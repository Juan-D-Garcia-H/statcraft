# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..utils.exceptions import DataValidationError, FormulaError
from ..utils.modelview import ModelView
from ._nls import nls


class _PredictorParamsAdapter:
    """
    Picklable adapter used when ``func`` has exactly 2 arguments, e.g.
    ``def f(x, p): ...``. It passes named, unpackable views so both
    ``Time, Dose = x`` and ``x[:, 0]`` / ``np.asarray(x)`` work; with one
    predictor ``x`` is used directly as the values, while ``time, = x`` still
    works. A one-parameter ``p`` likewise behaves as its value, with ``a, = p``
    still available. Each call is wrapped with the right shape for the backend.

    Defined at module level (not as a closure nested inside
    ``NLS.__init__``) specifically so instances are picklable: pickling a
    closure ("Can't pickle local object") is one of the classic reasons a
    fitted model can't be saved with ``pickle``/``joblib``. Picklability
    here only depends on whether the wrapped ``func`` itself is a
    module-level function — a ``lambda`` still can't be pickled, which is
    an inherent Python limitation (not specific to this class).
    """

    def __init__(self, func: Callable, predictor_cols: List[str], param_cols: List[str]) -> None:
        self.func = func
        self.predictor_cols = predictor_cols
        self.param_cols = param_cols
        self.__name__ = getattr(func, "__name__", "custom_func")

    def __call__(self, *args):
        n_exog = len(self.predictor_cols)
        x = ModelView(self.predictor_cols, args[:n_exog], label="x", columnar=True)
        p = ModelView(self.param_cols, args[n_exog:], label="p")
        return self.func(x, p)


class _NamedArgsAdapter:
    """
    Picklable adapter used when ``func`` already takes named exogenous/
    parameter arguments — just forwards the call, without overwriting
    ``__name__`` on the user's own function object (it might be a shared/
    reused function elsewhere). Defined at module level for the same
    picklability reason as ``_PredictorParamsAdapter`` above.
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.__name__ = getattr(func, "__name__", "custom_func")

    def __call__(self, *args):
        return self.func(*args)


class NLS(nls):
    """Statsmodels-style nonlinear regression from a plain Python function.

    Use this when the model is a ``def``/``lambda`` rather than a formula.
    The portable two-argument form is ``def f(x, p)``: ``x`` and ``p`` are
    named views that can be unpacked directly (``a, b = p``), indexed by
    position or name, or converted with ``np.asarray``. With a single entry,
    either view behaves directly as its value (``a = p`` and ``a, = p`` both
    work). With a single predictor, ``x`` can be used directly as the predictor
    values. The named form
    ``def f(x1, x2, a, b)`` is still called positionally as
    ``func(*exogenous, *parameters)``.

    Examples
    --------
    >>> from statcraft.nls import NLS
    >>> def growth(x, a, b):
    ...     return a * (1 - np.exp(-b * x))
    >>> fit = NLS(growth, df, response="y", predictors="x",
    ...           start={"a": 1.0, "b": 0.5}).fit()

    See Also
    --------
    nls : Formula-based nonlinear least squares.
    statcraft.nlme.gnls : Same single-curve model with correlation and
        variance structures.
    statcraft.nlme.nlme : Nonlinear mixed-effects model, for grouped data with
        random effects.
    """

    def __init__(
        self,
        func: Callable,
        data: pd.DataFrame,
        response: str,
        predictors: Optional[Union[str, List[str]]] = None,
        start: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
        n_params: Optional[int] = None,
        backend: str = "numpy",
    ) -> None:
        """Build an unfitted nonlinear model from a callable.

        Parameters
        ----------
        func : callable
            Model function. Either ``def f(x, p)`` (portable named views),
            or named arguments such as ``def f(x1, x2, a, b)``.
        data : pandas.DataFrame
            Data containing the response and the exogenous columns.
        response : str
            Name of the response column in ``data``.
        predictors : str or list of str, optional
            Name(s) of the exogenous column(s). Required (and matched in
            order to the function's first arguments) when ``func`` has
            more than two arguments; otherwise it defaults to the first
            argument's name.
        start : dict, list, array-like or None, optional
            Initial parameter values. For ``f(X, p)`` this (or
            ``n_params``) is required; for named-argument functions the
            names are inferred from the signature.
        n_params : int, optional
            Number of parameters for ``f(X, p)`` when ``start`` is not
            given; values default to ``1.0``.
        backend : {'numpy', 'jax'}, default 'numpy'
            Evaluation backend.
        """
        if not callable(func):
            raise TypeError("The first argument must be a callable function (def or lambda).")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if backend not in self._VALID_BACKENDS:
            raise FormulaError(f"backend must be one of {self._VALID_BACKENDS}, got '{backend}'.")
        if len(data) == 0:
            raise DataValidationError("data is empty (0 rows).")

        sig = inspect.signature(func)
        raw_params = list(sig.parameters.keys())

        if any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in sig.parameters.values()
        ):
            raise TypeError(
                f"'{getattr(func, '__name__', 'func')}' cannot have *args/**kwargs -- "
                "every argument must be explicitly named so exogenous "
                "variables can be distinguished from parameters."
            )

        if len(raw_params) < 2:
            raise FormulaError(
                f"Function '{func.__name__}' must take at least 2 arguments "
                "(e.g. (X, p) or (x1, x2, a, b))."
            )

        # --- 1. Normalize predictors ---
        if predictors is None:
            if len(raw_params) == 2:
                predictor_cols = [raw_params[0]]
            else:
                # With more than 2 arguments there's no safe way to guess
                # which are exogenous variables and which are parameters --
                # a default like "everything but the last" could
                # accidentally match real column names and build a model
                # with the wrong number of parameters without any warning.
                # Better to require it explicitly.
                raise FormulaError(
                    f"'{func.__name__}' has {len(raw_params)} arguments "
                    f"({raw_params}) -- with more than 2 arguments, 'predictors' "
                    "is required (e.g. predictors=['x1', 'x2']); there's no safe "
                    "way to automatically infer which are variables and which "
                    "are parameters."
                )
        elif isinstance(predictors, str):
            predictor_cols = [predictors]
        elif isinstance(predictors, (list, tuple)):
            predictor_cols = list(predictors)
        else:
            raise TypeError("predictors must be a str or a list/tuple of str.")

        if not predictor_cols:
            raise FormulaError(
                "predictors cannot be an empty list -- at least one exogenous variable is required."
            )

        # Validate that the columns exist
        missing_cols = [col for col in predictor_cols if col not in data.columns]
        if missing_cols:
            raise DataValidationError(
                f"The following exogenous variables do not exist in data: {missing_cols}"
            )

        if response not in data.columns:
            raise DataValidationError(f"Response variable '{response}' does not exist in data.")
        if response in predictor_cols:
            raise DataValidationError(
                f"Response variable '{response}' cannot also appear in predictors."
            )

        # --- 2. Process parameters ---
        if len(raw_params) == 2:
            if isinstance(start, dict):
                params_list = list(start.keys())
                start_dict: Any = start
            elif isinstance(start, (list, tuple, np.ndarray)):
                params_list = [f"p{i}" for i in range(len(start))]
                start_dict = dict(zip(params_list, start))
            elif n_params is not None and isinstance(n_params, int):
                if n_params < 1:
                    raise DataValidationError(f"n_params must be >= 1, got {n_params}.")
                params_list = [f"p{i}" for i in range(n_params)]
                start_dict = {p: 1.0 for p in params_list}
            else:
                raise DataValidationError(
                    "For functions shaped like 'def f(X, p):', you must provide 'start' "
                    "as a dict {'a': 1.0}, as a list/array [1.0, 2.0], "
                    "or pass 'n_params=N'."
                )
            if not params_list:
                raise DataValidationError(
                    "Cannot fit a model with no parameters (params_list is empty)."
                )

            # Picklable adapter (module-level class, see _PredictorParamsAdapter
            # above) for both NumPy and JAX
            target_func: Any = _PredictorParamsAdapter(func, predictor_cols, params_list)

        else:
            if raw_params[: len(predictor_cols)] != predictor_cols:
                raise FormulaError(
                    f"predictors={predictor_cols} must match, in the same order, the "
                    f"first {len(predictor_cols)} argument(s) of '{func.__name__}' "
                    f"({raw_params}) -- the function is called positionally as "
                    "func(*exogenous, *parameters), in that order."
                )
            params_list = raw_params[len(predictor_cols) :]
            if not params_list:
                raise FormulaError(
                    f"predictors={predictor_cols} covers every argument of "
                    f"'{func.__name__}' ({raw_params}) -- there's no parameter left to fit."
                )

            if isinstance(start, (list, tuple, np.ndarray)):
                if len(start) != len(params_list):
                    raise DataValidationError(
                        f"Expected {len(params_list)} value(s) in 'start', got {len(start)}."
                    )
                start_dict = dict(zip(params_list, start))
            elif isinstance(start, dict):
                if set(start.keys()) != set(params_list):
                    raise DataValidationError(
                        f"'start' keys {sorted(start.keys())} do not match "
                        f"the function's inferred parameters {sorted(params_list)}."
                    )
                start_dict = start
            else:
                start_dict = start

            # Picklable adapter (module-level class, see _NamedArgsAdapter
            # above): `func` is never reused directly as target_func, so
            # as not to overwrite __name__ on the object the user passed
            # in (it might be a shared/reused function elsewhere).
            target_func = _NamedArgsAdapter(func)

        # --- 3. Build the formula for FormulaParser ---
        for name in predictor_cols + params_list:
            if not str(name).isidentifier():
                raise FormulaError(
                    f"'{name}' is not a valid Python identifier -- exogenous "
                    "column and parameter names must be usable as formula "
                    "symbols (no spaces, dashes, etc.)."
                )

        func_name = getattr(func, "__name__", "custom_func")
        # Besides the reserved/generic names, any name starting with "SS"
        # collides with the self-starting-function detector
        # (`_RE_SPECIAL_FUNC` matches `^SS\w+\(...\)$` against the whole
        # formula): a user function named e.g. "SScustom" would generate
        # the formula "y ~ SScustom(x, a, b)", which FormulaParser would
        # treat as an unknown SS* and reject with a confusing error
        # ("Special function 'SScustom' is not supported") unrelated to
        # what the user actually intended.
        if func_name in ("<lambda>", "model", "func") or func_name.startswith("SS"):
            func_name = "custom_nls_func"

        target_func.__name__ = func_name

        extra_funcs = {
            func_name: target_func,
            "model": target_func,
            "custom_nls_func": target_func,
        }

        args_str = ", ".join(predictor_cols + params_list)
        formula_str = f"{response} ~ {func_name}({args_str})"

        # --- 4. Delegate to nls.__init__ (base class) ---
        # Building FormulaParser/Initializer/X/y/NaN-Inf checks/residual_fn
        # is deliberately NOT duplicated here: it lives in a single place
        # (nls.__init__), so any validation added there (or already there
        # and missed here) automatically applies to NLS too, instead of
        # the two construction paths being able to diverge over time.
        self.user_func = func
        super().__init__(
            formula=formula_str,
            data=data,
            start=start_dict,
            backend=backend,
            extra_functions=extra_funcs,
        )
