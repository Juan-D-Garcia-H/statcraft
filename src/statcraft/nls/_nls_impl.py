# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
nls
===
A thin wrapper around ``scipy.optimize.least_squares`` built on top of
``FormulaParser``/``Initializer``, styled after R's ``nls()`` function
(hence the lower-case class name — deliberate, so the API feels familiar
to anyone coming from R).

Why least_squares and not curve_fit
------------------------------------
``curve_fit`` is itself a thin wrapper around ``least_squares`` that
discards almost all of the result information (it only returns ``popt``
and ``pcov``) and does not make it as easy to pass an analytical
jacobian. Calling ``least_squares`` directly gives access to the full
``OptimizeResult`` (cost, optimality gradient, success/convergence
message, number of function/jacobian evaluations, etc.) and the option of
passing JAX's analytical jacobian when ``backend="jax"`` — instead of
having the optimizer approximate it via finite differences on every
iteration.

The jax backend
----------------
With ``backend="jax"``, a jit-compiled residual function and residual
jacobian are built once (reusing the jit-compiled, cached functions from
``FormulaParser`` — see ``FormulaParser._build_lambdify_jax``) and passed
to ``least_squares`` as ``fun``/``jac``. This avoids both the numerical
approximation of the jacobian (fewer function evaluations, more accurate)
and recompiling JAX on every iteration.

Compilation cost: JAX compiles the first time ``residual_fn``/``jac_fn``
run with a given input shape — that includes the first ``fit()`` call on
each instance. For a single small fit, that compilation cost can outweigh
the benefit (measured: ~0.45s compilation vs. ~0.008s for a small numpy
finite-difference fit). The real payoff shows up when: fitting repeatedly
on the same instance (subsequent ``fit()`` calls reuse the compiled cache
— measured: ~0.45s down to ~0.01s), problems with many parameters
(finite differences need ~n_params extra evaluations per jacobian;
autodiff does not), and large datasets where the analytical jacobian is
more accurate than the default finite-difference step.
"""

import json
import warnings
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import Bounds, least_squares

from ..plots.methods import PlottingMixin
from ..summary import Summary, build_summary
from ..utils.exceptions import (
    ConvergenceError,
    DataValidationError,
    FitError,
    NLSError,
    NotFittedError,
)
from ..utils.FormulaParser import FormulaNamespaceError, FormulaParser, _get_jax
from ..utils.Initializer import Initializer


class nls(PlottingMixin):
    """Nonlinear least squares regression specified with an R-style formula.

    The model is built from a ``y ~ f(x, a, b)`` formula, where the
    parameters (``a``, ``b``, ...) are the free symbols of the right-hand
    side that are not columns of ``data``. Call :meth:`fit` to estimate
    them.

    Examples
    --------
    >>> from statcraft.nls import nls
    >>> model = nls("y ~ a * exp(-b * x)", df)
    >>> fit = model.fit()
    >>> fit.coefficients
    >>> fit.summary()

    See Also
    --------
    NLS : statsmodels-style wrapper for a plain Python function.
    statcraft.nlme.gnls : Same single-curve model with correlation and
        variance structures (``corStruct``/``varStruct``).  Use it when the
        residuals are grouped or heteroscedastic.
    statcraft.nlme.nlme : Nonlinear mixed-effects model, for grouped data with
        random effects.
    """

    _VALID_BACKENDS = ("numpy", "jax")

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        start: Optional[Dict[str, float]] = None,
        backend: str = "numpy",
        extra_functions: Optional[Dict[str, Any]] = None,
    ):
        """Build an unfitted nonlinear least-squares model.

        Parameters
        ----------
        formula : str
            Model formula, e.g. ``"y ~ a * exp(-b * x)"``. The response is
            taken from the left-hand side; the parameters are the free
            symbols on the right-hand side that are not columns of ``data``.
        data : pandas.DataFrame
            Data containing the response and the exogenous columns.
        start : dict, optional
            Initial values keyed by parameter name. When omitted, the
            :class:`~statcraft.utils.Initializer` computes reasonable
            starting values heuristically.
        backend : {'numpy', 'jax'}, default 'numpy'
            Evaluation backend. ``'jax'`` provides a compiled analytical
            jacobian for faster and more accurate fitting.
        extra_functions : dict, optional
            Extra callables made available to the formula namespace (e.g.
            custom link or transform functions).
        """
        if backend not in self._VALID_BACKENDS:
            raise FitError(f"backend must be one of {self._VALID_BACKENDS}, got '{backend}'.")
        if len(data) == 0:
            raise DataValidationError("data is empty (0 rows).")

        self.formula = formula
        self.data = data
        self.backend = backend
        # Stored so an equivalent nls instance can be reconstructed later
        # (used by cross_validate) without losing any custom function
        # registered when this instance was built.
        self._extra_functions = extra_functions

        try:
            self._parser = FormulaParser(
                formula, data, extra_functions=extra_functions, backend=backend
            )
        except (ValueError, FormulaNamespaceError):
            # No print() here: it would clutter stdout and duplicate the
            # message the re-raised exception already carries.
            raise

        self._info = self._parser.get_info()

        self._endogenous = self._info["endogenous"]
        self._exogenous = self._info["exogenous"]
        self._parameters = self._info["parameters"]
        self._is_special = self._info["is_special_function"]

        # Initializer is ALWAYS invoked (whether or not the user passed
        # `start`), for every formula. Besides resolving a default p0 in
        # every case, this validates a user-supplied `start` (exact keys,
        # no NaN/Inf) instead of surfacing a context-free KeyError or a
        # cryptic failure from least_squares/curve_fit further down the
        # line. The already-built FormulaParser (self._parser) is passed
        # in so Initializer doesn't parse/validate the same formula again
        # from scratch (duplicate sympify + namespace checks). Initializer
        # only reads metadata off the parser (params, exogenous,
        # is_special, etc.) for its numpy heuristics — it never evaluates
        # backend functions — so reusing it is safe even when
        # backend='jax'.
        self.start: Dict[str, float] = Initializer(
            formula,
            data,
            p0=start,
            extra_functions=extra_functions,
            parser=self._parser,
        ).initial_values

        self._X = np.asarray(data[self._exogenous].values, dtype=float)
        self._y = np.asarray(data[self._endogenous].values, dtype=float)

        nan_cols = [c for c in self._exogenous if data[c].isna().any()]
        if data[self._endogenous].isna().any():
            nan_cols.append(self._endogenous)
        if nan_cols:
            raise DataValidationError(
                f"Column(s) {nan_cols} contain NaN values. Drop or impute them before fitting."
            )

        inf_cols = [c for c in self._exogenous if np.isinf(data[c].to_numpy(dtype=float)).any()]
        if np.isinf(data[self._endogenous].to_numpy(dtype=float)).any():
            inf_cols.append(self._endogenous)
        if inf_cols:
            raise DataValidationError(
                f"Column(s) {inf_cols} contain infinite values (inf/-inf). "
                "Drop or clip them before fitting."
            )

        # Residual/jacobian functions (jit-compiled if backend='jax') are
        # cached once here, so they aren't rebuilt on every fit() call.
        self._residual_fn, self._jac_fn = self._build_residual_and_jac()

    def __repr__(self) -> str:
        if not hasattr(self, "coefficients"):
            return f"nls('{self.formula}', backend='{self.backend}', not fitted)"
        coefs_str = ", ".join(f"{p}={self.coefficients[p]:.4g}" for p in self._parameters)
        return (
            f"nls('{self.formula}', backend='{self.backend}', fitted, "
            f"R²={self.rsquared:.4f}, {coefs_str})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    # ------------------------------------------------------------------

    def _build_residual_and_jac(self):
        """
        Returns (residual_fn, jac_fn). jac_fn is None when backend='numpy'
        (least_squares falls back to finite differences in that case).
        """
        callable_func = self._parser.to_callable()

        if self.backend != "jax":

            def residual_fn(params: np.ndarray) -> np.ndarray:
                return callable_func(self._X, *params) - self._y

            return residual_fn, None

        jax = _get_jax()
        jnp = jax.numpy

        # Reuses the functions FormulaParser already jit-compiled in its
        # constructor (predictions_fn / predictions_jacobian_fn) instead of
        # compiling the evaluation and jacobian again here with
        # to_jax_callable() + jax.jacfwd(...) from scratch. This is the
        # exact same derivative either way, since residual(p) = pred(p) - y
        # implies d(residual)/dp = d(pred)/dp (y is constant w.r.t. p) — so
        # reusing FormulaParser's cached jacobian avoids paying its
        # compilation cost a second time.
        pred_fn = self._parser.predictions_fn()  # jit, (exog_tuple, params) -> predictions
        pred_jac_fn = (
            self._parser.predictions_jacobian_fn()
        )  # jit, (exog_tuple, params) -> jacobian

        # Tuple of exogenous columns in jax, in the order of
        # self._exogenous (the order pred_fn/pred_jac_fn expect) — built
        # once here, not on every least_squares call.
        exog_tuple = tuple(jnp.asarray(col) for col in self._X.T)
        y_jax = jnp.asarray(self._y)

        @jax.jit
        def _residual_jax(params):
            return pred_fn(exog_tuple, params) - y_jax

        def residual_fn(params: np.ndarray) -> np.ndarray:
            return np.asarray(_residual_jax(jnp.asarray(params)))

        def jac_fn(params: np.ndarray) -> np.ndarray:
            # jac(residual) == jac(pred) — see note above. pred_jac_fn is
            # already jit-compiled and cached from the parser's
            # construction; nothing is recompiled here.
            return np.asarray(pred_jac_fn(exog_tuple, jnp.asarray(params)))

        return residual_fn, jac_fn

    # ------------------------------------------------------------------

    _VALID_COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3")

    def fit(
        self,
        method: str = "trf",
        loss: str = "linear",
        use_analytical_jacobian: Optional[bool] = None,
        cov_type: str = "nonrobust",
        raise_on_non_convergence: bool = False,
        **kwargs,
    ) -> "nls":
        """
        Fit the model with ``scipy.optimize.least_squares``.

        Parameters
        ----------
        method : {'trf', 'dogbox', 'lm'}, default 'trf'
            Algorithm passed to ``scipy.optimize.least_squares``. ``'lm'``
            only supports the default ``'linear'`` loss.
        loss : str, default 'linear'
            Robust loss function ('linear', 'soft_l1', 'huber', 'cauchy',
            'arctan'). 'lm' only supports 'linear' — a restriction of
            scipy itself, not of this class.
        use_analytical_jacobian : bool, optional
            If None (default), uses JAX's analytical jacobian when
            ``backend='jax'`` and finite differences ('2-point')
            otherwise. Forcing False with backend='jax' falls back to
            finite differences (useful for comparison/debugging).
        cov_type : {'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'}, default 'nonrobust'
            'nonrobust' (default): classical covariance, assumes constant
            error variance (homoscedasticity) — the same one ``curve_fit``
            computes.
            'HC0'-'HC3': heteroscedasticity-consistent sandwich
            covariance, generalized to NLS by replacing OLS's design
            matrix with the jacobian at the optimum (the same idea as R's
            ``sandwich::vcovHC`` applied to an ``nls`` fit). HC0 is the
            base version; HC1 applies a uniform degrees-of-freedom
            correction; HC2/HC3 weight each residual by its leverage (HC3
            is the most conservative, recommended with small samples —
            the same preference order as in OLS). Use this when you
            suspect the error variance isn't constant (common in
            dose-response/growth curves).
        raise_on_non_convergence : bool, default False
            If True, raise a ``ConvergenceError`` when
            ``OptimizeResult.success`` is False instead of only emitting a
            ``RuntimeWarning``. The fit still runs to completion either
            way — ``self.result``/``self.coefficients``/etc. are set
            before the exception is raised, so the (non-converged) fit
            remains inspectable on the instance even when this raises.
        **kwargs
            Forwarded to ``scipy.optimize.least_squares`` (bounds,
            x_scale, f_scale, max_nfev, ftol, xtol, gtol, etc.)

        Returns
        -------
        nls
            The fitted instance (``self``), so calls can be chained.

        Raises
        ------
        FitError
            If ``cov_type`` is invalid, an analytical jacobian is
            requested on a backend that doesn't support it, or
            ``least_squares`` itself raises.
        ConvergenceError
            If ``raise_on_non_convergence=True`` and the optimizer does
            not converge.

        Notes
        -----
        After fitting, the following become available: ``self.result``
        (the full OptimizeResult -- cost, optimality, status, message,
        success, nfev, njev, active_mask, fun, jac), ``self.coefficients``,
        ``self.covariance_matrix``, ``self.std_errors``, ``self.cov_type``,
        ``self.fitted_values``, ``self.residuals``.
        """
        if cov_type not in self._VALID_COV_TYPES:
            raise FitError(f"cov_type must be one of {self._VALID_COV_TYPES}, got '{cov_type}'.")
        if use_analytical_jacobian is None:
            use_analytical_jacobian = self.backend == "jax"
        if use_analytical_jacobian and self._jac_fn is None:
            raise FitError(
                "use_analytical_jacobian=True requires backend='jax' "
                f"(this instance was built with backend='{self.backend}')."
            )

        p0 = np.array([self.start[p] for p in self._parameters], dtype=float)
        result = self._run_least_squares(p0, method, loss, use_analytical_jacobian, kwargs)
        self._apply_fit_result(
            result,
            method,
            loss,
            use_analytical_jacobian,
            cov_type,
            raise_on_non_convergence=raise_on_non_convergence,
        )
        return self

    @staticmethod
    def _normalize_bounds(
        bounds, p0: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts (lb, ub) as arrays from ``bounds`` (a tuple or a
        ``scipy.optimize.Bounds``), only so ``fit_multistart``'s
        candidates can be clipped within the limits — the original
        ``bounds`` is passed through to ``least_squares`` unchanged,
        separately. Returns ``(None, None)`` if ``bounds`` is ``None``.
        """
        if bounds is None:
            return None, None

        if isinstance(bounds, Bounds):
            lb, ub = np.asarray(bounds.lb, dtype=float), np.asarray(bounds.ub, dtype=float)
        elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lb, ub = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
        else:
            raise FitError(
                "bounds must be a (lb, ub) tuple or a scipy.optimize.Bounds, "
                f"got {type(bounds).__name__}."
            )

        if lb.ndim == 0:
            lb = np.full_like(p0, float(lb))
        if ub.ndim == 0:
            ub = np.full_like(p0, float(ub))

        if lb.shape != p0.shape or ub.shape != p0.shape:
            raise FitError(
                f"bounds must have {len(p0)} element(s), one per parameter: "
                f"lb.shape={lb.shape}, ub.shape={ub.shape}."
            )
        if np.any(lb >= ub):
            raise FitError("Lower bounds must be strictly less than upper bounds.")
        if np.any(p0 < lb) or np.any(p0 > ub):
            raise FitError(
                f"self.start is outside the given bounds: "
                f"start={p0.tolist()}, lb={lb.tolist()}, ub={ub.tolist()}."
            )
        return lb, ub

    def _run_least_squares(self, p0, method, loss, use_analytical_jacobian, kwargs):
        """One least_squares run from p0. Does not touch instance state —
        only returns the OptimizeResult. Shared by fit() and
        fit_multistart()."""
        jac = self._jac_fn if use_analytical_jacobian else "2-point"
        try:
            return least_squares(
                fun=self._residual_fn,
                x0=p0,
                jac=jac,
                method=method,
                loss=loss,
                **kwargs,
            )
        except Exception as exc:
            raise FitError(
                f"least_squares failed for '{self.formula}' "
                f"(method='{method}', loss='{loss}', p0={dict(zip(self._parameters, p0))}): {exc}"
            ) from exc

    def _apply_fit_result(
        self,
        result,
        method,
        loss,
        use_analytical_jacobian,
        cov_type="nonrobust",
        raise_on_non_convergence: bool = False,
    ) -> None:
        """Sets every attribute derived from an already-resolved
        OptimizeResult (coefficients, covariance_matrix, std_errors,
        residuals, etc.) — the tail shared by fit() and
        fit_multistart()."""
        if not result.success:
            message = (
                f"least_squares did not converge for '{self.formula}': "
                f"{result.message} (status={result.status})."
            )
            if raise_on_non_convergence:
                # The result is still applied below before raising, so the
                # (non-converged) fit stays inspectable on the instance.
                non_convergence_message = message
            else:
                warnings.warn(message, RuntimeWarning)
                non_convergence_message = None
        else:
            non_convergence_message = None

        self.result = result  # full OptimizeResult, untrimmed
        self._fit_method = method
        self._fit_loss = loss
        self._fit_used_analytical_jac = use_analytical_jacobian
        self.cov_type = cov_type
        popt = result.x

        self.coefficients = dict(zip(self._parameters, popt))

        callable_func = self._parser.to_callable()
        self.fitted_values = callable_func(self._X, *popt)
        self.residuals = (
            self._y - self.fitted_values
        )  # needed by HC0-3, hence computed before covariance_matrix

        self.covariance_matrix = (
            self._compute_covariance(result)
            if cov_type == "nonrobust"
            else self._compute_robust_covariance(cov_type)
        )
        self._std_errors = dict(
            zip(
                self._parameters,
                np.sqrt(np.maximum(np.diag(self.covariance_matrix), 0.0)),
            )
        )

        if non_convergence_message is not None:
            raise ConvergenceError(non_convergence_message)

    def fit_multistart(
        self,
        n_starts: int = 10,
        perturbation: float = 0.5,
        method: str = "trf",
        loss: str = "linear",
        use_analytical_jacobian: Optional[bool] = None,
        random_state: Optional[int] = None,
        bounds: Optional[Union[Tuple[Any, Any], Bounds]] = None,
        cov_type: str = "nonrobust",
        raise_on_non_convergence: bool = False,
        **kwargs,
    ) -> "nls":
        """
        Like ``fit()``, but tries ``n_starts`` different starting points
        and keeps the lowest-cost one among those that converged —
        mitigates NLS's classic problem of landing on a local minimum
        depending on the initial ``p0``.

        The first attempt always uses ``self.start`` (the p0 resolved by
        Initializer, or whatever was passed to the constructor). The
        remaining ``n_starts - 1`` are log-normal multiplicative
        perturbations of that same point: ``p0_i = start * exp(N(0,
        perturbation))`` — multiplicative rather than additive so it
        doesn't break the sign/scale of very different parameters within
        the same formula (e.g. an amplitude ~10 and a rate ~0.01).

        Parameters
        ----------
        n_starts : int, default 10
            Number of starting points to try.
        perturbation : float, default 0.5
            Standard deviation (in log scale) of the multiplicative
            perturbation. Larger = wider search, but more attempts that
            fail to converge.
        random_state : int, optional
            Seed for reproducibility.
        bounds : tuple ``(lb, ub)`` or ``scipy.optimize.Bounds``, optional
            Per-parameter bounds, in the order of ``self.parameters``.
            Used for two things: (1) passed through as-is to
            ``least_squares`` on every attempt (same as passing them via
            ``**kwargs``), and (2) perturbed candidates are clipped
            (``np.clip``) to always fall within the bounds — without this,
            an out-of-range candidate makes ``least_squares`` fail with
            ``ValueError: x0 is infeasible`` instead of simply trying the
            next starting point. ``method='lm'`` does not support bounds
            (a restriction of scipy itself).
        raise_on_non_convergence : bool, default False
            Same meaning as in ``fit()``: raise a ``ConvergenceError``
            instead of only warning when no attempt converges (the
            lowest-cost, non-converged result is still applied to the
            instance first).


        Besides the attributes ``fit()`` sets, this also makes
        ``self.multistart_info`` available (a dict with ``n_starts``,
        ``n_converged``, ``costs`` — the final cost of every attempt, in
        the same order).

        Raises
        ------
        FitError
            If ``n_starts < 1``, ``cov_type`` is invalid, an analytical
            jacobian is requested on an unsupported backend, or every
            attempt raises an exception.
        ConvergenceError
            If ``raise_on_non_convergence=True`` and no attempt converges.
        """
        if n_starts < 1:
            raise FitError(f"n_starts must be >= 1, got {n_starts}.")
        if cov_type not in self._VALID_COV_TYPES:
            raise FitError(f"cov_type must be one of {self._VALID_COV_TYPES}, got '{cov_type}'.")
        if use_analytical_jacobian is None:
            use_analytical_jacobian = self.backend == "jax"
        if use_analytical_jacobian and self._jac_fn is None:
            raise FitError(
                "use_analytical_jacobian=True requires backend='jax' "
                f"(this instance was built with backend='{self.backend}')."
            )

        rng = np.random.default_rng(random_state)
        base_p0 = np.array([self.start[p] for p in self._parameters], dtype=float)
        lb, ub = self._normalize_bounds(bounds, base_p0)

        zero_mask = base_p0 == 0.0
        candidates = [base_p0]
        for _ in range(n_starts - 1):
            factors = np.exp(rng.normal(0.0, perturbation, size=base_p0.shape))
            candidate = base_p0 * factors
            # A multiplicative perturbation never moves a parameter whose
            # start is exactly 0 (0 * factor is always 0): for those
            # components, an additive perturbation of the same scale is
            # used instead, so multistart also explores starts at zero.
            if np.any(zero_mask):
                candidate[zero_mask] = rng.normal(0.0, perturbation, size=int(zero_mask.sum()))
            if lb is not None:
                candidate = np.clip(candidate, lb, ub)
            candidates.append(candidate)

        fit_kwargs = dict(kwargs)
        if bounds is not None:
            fit_kwargs["bounds"] = (
                bounds  # original object/tuple — scipy already handles both formats
            )

        all_results = []
        for p0 in candidates:
            try:
                r = self._run_least_squares(p0, method, loss, use_analytical_jacobian, fit_kwargs)
            except FitError:
                continue  # this starting point made the optimizer fail (e.g. overflow); try the next one
            all_results.append(r)

        if not all_results:
            raise FitError(
                f"fit_multistart: all {n_starts} attempts raised an exception for '{self.formula}'."
            )

        converged = [r for r in all_results if r.success]
        n_converged = len(converged)
        if converged:
            best_result = min(converged, key=lambda r: r.cost)
        else:
            best_result = min(all_results, key=lambda r: r.cost)
            message = (
                f"fit_multistart: none of the {n_starts} attempts converged "
                f"for '{self.formula}'; using the lowest-cost one (not converged)."
            )
            if raise_on_non_convergence:
                self._apply_fit_result(best_result, method, loss, use_analytical_jacobian, cov_type)
                self.multistart_info = {
                    "n_starts": n_starts,
                    "n_converged": n_converged,
                    "costs": [r.cost for r in all_results],
                }
                raise ConvergenceError(message)
            warnings.warn(message, RuntimeWarning)

        self._apply_fit_result(best_result, method, loss, use_analytical_jacobian, cov_type)
        self.multistart_info = {
            "n_starts": n_starts,
            "n_converged": n_converged,
            "costs": [r.cost for r in all_results],
        }
        return self

    @staticmethod
    def _compute_covariance(result) -> np.ndarray:
        """
        A faithful reimplementation of how ``scipy.optimize.curve_fit``
        derives ``pcov`` from a ``least_squares`` result (covariance not
        scaled by sigma — ``absolute_sigma=False``): pseudo-inverse of
        J^T J via SVD, scaled by the residual variance.
        """
        J = result.jac
        n_obs, n_params = J.shape
        _, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0] if s.size else 0.0
        keep = s > threshold
        s = s[keep]
        VT = VT[keep]
        pcov = (VT.T / s**2) @ VT

        dof = n_obs - n_params
        if dof > 0:
            s_sq = 2.0 * result.cost / dof  # least_squares cost = 0.5*sum(residuals**2)
        else:
            s_sq = 2.0 * result.cost
            warnings.warn(
                "Number of observations <= number of parameters; "
                "covariance matrix is not well-defined.",
                RuntimeWarning,
            )
        return pcov * s_sq

    def _compute_robust_covariance(self, cov_type: str) -> np.ndarray:
        """
        Heteroscedasticity-consistent sandwich covariance (HC0-HC3),
        generalized to NLS by replacing OLS's design matrix with the
        jacobian J at the optimum — the same idea as R's
        ``sandwich::vcovHC`` applied to an ``nls`` object:

            bread = (JᵀJ)⁻¹            (via SVD, same as leverage/_compute_covariance)
            meat  = Jᵀ diag(w) J        where w_i weights e_i² depending on the type:
                HC0: w_i = e_i²
                HC1: w_i = e_i² · n/(n-k)                  (uniform df correction)
                HC2: w_i = e_i² / (1-h_i)                  (weighted by leverage)
                HC3: w_i = e_i² / (1-h_i)²                 (more conservative, better for small n)
            Σ_robust = bread · meat · bread

        Requires ``self.residuals`` to already be computed (which is why
        ``_apply_fit_result`` computes residuals before calling this).
        """
        J = np.atleast_2d(self.result.jac)
        n, k = J.shape
        e = self.residuals

        U, s, VT = np.linalg.svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0] if s.size else 0.0
        keep = s > threshold
        s_k, VT_k = s[keep], VT[keep]
        bread = (VT_k.T / s_k**2) @ VT_k  # (JᵀJ)⁻¹, via SVD (same method as _compute_covariance)
        h = np.sum(U[:, keep] ** 2, axis=1)  # leverage, from the same SVD — not recomputed

        if cov_type == "HC0":
            w = e**2
        elif cov_type == "HC1":
            dof = max(n - k, 1)
            w = e**2 * (n / dof)
        elif cov_type == "HC2":
            w = e**2 / np.clip(1.0 - h, 1e-12, None)
        elif cov_type == "HC3":
            w = e**2 / np.clip(1.0 - h, 1e-12, None) ** 2
        else:
            raise FitError(f"Unknown robust cov_type: '{cov_type}'.")

        meat = J.T @ (w[:, None] * J)
        return bread @ meat @ bread

    # ------------------------------------------------------------------
    # Fit statistics — require fit() to have been called first.
    # ------------------------------------------------------------------

    def _require_fitted(self, prop_name: str) -> None:
        if not hasattr(self, "coefficients"):
            raise NotFittedError(
                f"'{prop_name}' is not available before calling .fit() — call model.fit() first."
            )

    @property
    def rss(self) -> float:
        """Residual sum of squares."""
        self._require_fitted("rss")
        return float(np.sum(self.residuals**2))

    @property
    def tss(self) -> float:
        """Total sum of squares (w.r.t. the mean of y)."""
        self._require_fitted("tss")
        return float(np.sum((self._y - np.mean(self._y)) ** 2))

    @property
    def rmse(self) -> float:
        """Root mean squared error: sqrt(rss / n)."""
        self._require_fitted("rmse")
        return float(np.sqrt(self.rss / len(self._y)))

    @property
    def mae(self) -> float:
        """Mean absolute error."""
        self._require_fitted("mae")
        return float(np.mean(np.abs(self.residuals)))

    @property
    def nrmse(self) -> float:
        """
        RMSE normalized by the range of y (max - min) — useful for
        comparing goodness of fit across variables with different scales
        (unlike rmse/mae, which are in y's own units).
        """
        self._require_fitted("nrmse")
        y_range = float(np.max(self._y) - np.min(self._y))
        if y_range == 0:
            warnings.warn(
                "The range of y is 0 (y is constant): nrmse is not well-defined.",
                RuntimeWarning,
            )
            return float("nan")
        return self.rmse / y_range

    @property
    def rsquared(self) -> float:
        """
        1 - RSS/TSS. For NLS models without an explicit intercept, this R²
        doesn't have the same interpretation as in linear OLS (it can even
        come out negative) — it's a heuristic goodness-of-fit measure, not
        a strict proportion of variance explained.
        """
        self._require_fitted("rsquared")
        tss = self.tss
        if tss == 0:
            warnings.warn(
                "tss == 0 (y is constant): rsquared is not well-defined.",
                RuntimeWarning,
            )
            return float("nan")
        return 1.0 - self.rss / tss

    @property
    def adj_rsquared(self) -> float:
        """R² adjusted for degrees of freedom: 1 - (RSS/TSS)*(n-1)/(n-k)."""
        self._require_fitted("adj_rsquared")
        n, k = len(self._y), len(self._parameters)
        if n - k <= 0:
            warnings.warn(
                "n <= number of parameters: adj_rsquared is not well-defined.",
                RuntimeWarning,
            )
            return float("nan")
        tss = self.tss
        if tss == 0:
            warnings.warn(
                "tss == 0 (y is constant): adj_rsquared is not well-defined.",
                RuntimeWarning,
            )
            return float("nan")
        return 1.0 - (self.rss / tss) * (n - 1) / (n - k)

    @property
    def logLik(self) -> float:
        """
        Gaussian log-likelihood of the fitted model (same convention as
        R's ``logLik.nls``: MLE of σ² = RSS/n).
        """
        self._require_fitted("logLik")
        n = len(self._y)
        rss = self.rss
        if rss <= 0:
            warnings.warn(
                "rss <= 0 (perfect or degenerate fit): logLik is not well-defined (log(0)).",
                RuntimeWarning,
            )
            return float("inf")
        return -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)

    @property
    def aic(self) -> float:
        """
        Akaike Information Criterion = -2*logLik + 2*df, with
        df = k + 1 (k regression parameters + 1 for the error variance
        estimated by MLE) — same convention as R's ``AIC()``. Defined in
        terms of logLik (rather than a separate formula) so the textbook
        identity AIC = -2*logLik + 2*df always holds, including when
        compared against logLik directly.
        """
        self._require_fitted("aic")
        k = len(self._parameters)
        return -2.0 * self.logLik + 2.0 * (k + 1)

    @property
    def bic(self) -> float:
        """BIC = -2*logLik + df*log(n), same df=k+1 convention as aic."""
        self._require_fitted("bic")
        n, k = len(self._y), len(self._parameters)
        return -2.0 * self.logLik + (k + 1) * np.log(n)

    @property
    def aicc(self) -> float:
        """
        AIC corrected for small sample size: ``aic + 2K(K+1)/(n-K-1)``
        with ``K = k+1`` (same degrees-of-freedom convention as
        ``aic``/``bic``). Preferable to ``aic`` when ``n/K`` isn't large
        (common rule of thumb: n/K < 40) — typical in NLS fits with
        limited data.
        """
        self._require_fitted("aicc")
        n, k = len(self._y), len(self._parameters)
        K = k + 1
        denom = n - K - 1
        if denom <= 0:
            warnings.warn(
                "n - K - 1 <= 0 (sample too small relative to the number "
                "of parameters): aicc is not well-defined.",
                RuntimeWarning,
            )
            return float("inf")
        return self.aic + (2.0 * K * (K + 1)) / denom

    @property
    def params(self) -> np.ndarray:
        """Estimated parameter values, in the order of self._parameters."""
        self._require_fitted("params")
        return np.asarray(list(self.coefficients.values()))

    @property
    def std_errors(self) -> np.ndarray:
        """Per-parameter standard errors (square root of covariance_matrix's diagonal)."""
        self._require_fitted("std_errors")
        return np.asarray(list(self._std_errors.values()))

    @property
    def resid_df(self) -> int:
        """Residual degrees of freedom (n - k)."""
        self._require_fitted("resid_df")
        return len(self._y) - len(self._parameters)

    @property
    def fvalue(self) -> float:
        """
        F-statistic of the fitted model against the null model
        "y = mean(y)" (there is no standard ANOVA decomposition for NLS
        like there is for linear OLS; this follows the convention reported
        by MATLAB's fitnlm/GraphPad Prism for nonlinear regression:
        F = ((TSS-RSS)/k) / (RSS/(n-k))).
        """
        self._require_fitted("fvalue")
        k, dof = len(self._parameters), self.resid_df
        if dof <= 0:
            warnings.warn(
                "n <= number of parameters: fvalue is not well-defined.",
                RuntimeWarning,
            )
            return float("nan")
        rss = self.rss
        if rss <= 0:
            warnings.warn(
                "rss <= 0 (perfect or degenerate fit): fvalue is not "
                "well-defined (division by zero).",
                RuntimeWarning,
            )
            return float("inf")
        return ((self.tss - rss) / k) / (rss / dof)

    @property
    def f_pvalue(self) -> float:
        """
        p-value of the F-statistic above (against the null model
        "y = mean(y)"), with (k, n-k) degrees of freedom.
        """
        self._require_fitted("f_pvalue")
        return float(stats.f.sf(self.fvalue, len(self._parameters), self.resid_df))

    @property
    def leverage(self) -> np.ndarray:
        """
        Leverage (hat-values) for each observation: diagonal of the local
        hat matrix ``H = J(JᵀJ)⁻¹Jᵀ``, where ``J`` is the jacobian at the
        optimum (the same matrix ``summary()`` uses for "Cond. No.").
        Computed via the SVD of J (h_ii = sum of U[i,:]² over the reduced
        SVD's columns) instead of forming H explicitly — avoids building
        an n×n matrix and stays numerically stable even with an
        ill-conditioned J.
        """
        self._require_fitted("leverage")
        J = np.atleast_2d(self.result.jac)
        U, _, _ = np.linalg.svd(J, full_matrices=False)
        return np.sum(U**2, axis=1)

    @property
    def cooks_distance(self) -> np.ndarray:
        """
        Cook's distance per observation: ``D_i = e_i² h_i / (k·s²·(1-h_i)²)``,
        with ``e`` the residuals, ``h`` the leverage, ``k`` the number of
        parameters and ``s² = rss/resid_df``. Measures how much the fit
        would change if that observation were excluded — the standard
        influence diagnostic, generalized to NLS via the local
        linearization at J.
        """
        self._require_fitted("cooks_distance")
        h = self.leverage
        k = len(self._parameters)
        s2 = self.rss / self.resid_df
        one_minus_h = 1.0 - h
        if np.any(one_minus_h <= 1e-12):
            warnings.warn(
                "Some observation has leverage ≈ 1 (total influence over "
                "its own prediction): cooks_distance may be "
                "infinite/unreliable for that point.",
                RuntimeWarning,
            )
        denom = k * s2 * one_minus_h**2
        with np.errstate(divide="ignore", invalid="ignore"):
            D = (self.residuals**2 * h) / denom
        return D

    @property
    def press(self) -> float:
        """
        PRESS (Predicted REsidual Sum of Squares): sum of squared
        leave-one-out residuals, approximated without refitting the model
        n times via ``e_i / (1 - h_i)`` (the exact formula for OLS; for
        NLS it's the standard approximation based on the local
        linearization at the optimum — the same one ``leverage``/
        ``cooks_distance`` use). See also ``cross_validate()`` for actual
        cross-validation (no local approximation) at the cost of
        refitting k times.
        """
        self._require_fitted("press")
        h = self.leverage
        one_minus_h = 1.0 - h
        if np.any(one_minus_h <= 1e-10):
            warnings.warn(
                "Some observation has leverage ≈ 1: press may be infinite/unreliable.",
                RuntimeWarning,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            loo_resid = self.residuals / one_minus_h
        return float(np.nansum(np.where(one_minus_h > 1e-10, loo_resid**2, np.nan)))

    @property
    def rsquared_pred(self) -> float:
        """
        Predictive R²: ``1 - PRESS/TSS``. Unlike ``rsquared`` (which only
        measures how well the model explains the data it was fit on),
        this approximates how well the model would predict a new
        observation — much more informative for judging overfitting. Can
        come out negative if the model generalizes worse than predicting
        the mean.
        """
        self._require_fitted("rsquared_pred")
        tss = self.tss
        if tss == 0:
            warnings.warn(
                "tss == 0 (y is constant): rsquared_pred is not well-defined.",
                RuntimeWarning,
            )
            return float("nan")
        return 1.0 - self.press / tss

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        (1-alpha) confidence intervals for each parameter, via the
        asymptotic normal-t approximation (coef ± t_{1-alpha/2,df} * se).

        Works for any number of parameters. Column labels reflect the
        actual `alpha` passed, rather than a fixed "2.5%"/"97.5%".
        """
        self._require_fitted("conf_int")
        t_val = stats.t.ppf(1 - alpha / 2, self.resid_df)
        params = self.params  # values, ndarray
        se = np.asarray([self._std_errors[p] for p in self._parameters])
        lower_label = f"{alpha / 2 * 100:.1f}%"
        upper_label = f"{(1 - alpha / 2) * 100:.1f}%"
        return pd.DataFrame(
            {
                lower_label: params - t_val * se,
                upper_label: params + t_val * se,
            },
            index=self._parameters,
        )

    def predict(
        self,
        newdata: Optional[pd.DataFrame] = None,
        interval: Optional[str] = None,
        alpha: float = 0.05,
    ):
        """
        Predictions for newdata (or for self.data if newdata is None).

        Parameters
        ----------
        interval : {None, "confidence", "prediction"}, default None
            - None: returns only the point estimate (``np.ndarray``).
            - "confidence": uncertainty band around the estimated mean,
              via the delta method (jacobian of the prediction at
              newdata, propagated through ``self.covariance_matrix``):
              ``Var(f(x0,θ)) ≈ J(x0) Σ J(x0)ᵀ``.
            - "prediction": same as "confidence" but adding the residual
              variance (``rss/resid_df``) — a band for a new observation,
              not just for the mean.

            With "confidence"/"prediction" this returns a
            ``pd.DataFrame`` (columns ``fit``, ``se``, ``lower``,
            ``upper``, same index as newdata).
        alpha : float, default 0.05
            Level for the lower/upper band (Student's t with
            ``self.resid_df`` degrees of freedom).

        With backend='jax' the jacobian at newdata is analytical
        (autodiff, via ``FormulaParser.jacobian(data=newdata)``). With
        backend='numpy' it's approximated with central differences over
        ``to_callable()`` — slower but doesn't require jax.

        Raises
        ------
        DataValidationError
            If ``newdata`` isn't a ``DataFrame``, is missing required
            exogenous columns, contains NaN values, or ``interval`` isn't
            one of the supported values.
        """
        self._require_fitted("predict")
        df = self.data if newdata is None else newdata

        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"newdata must be a pandas DataFrame, got {type(df).__name__}."
            )

        missing_cols = [c for c in self._exogenous if c not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"newdata is missing the required exogenous column(s): {missing_cols}."
            )

        nan_cols = [c for c in self._exogenous if df[c].isna().any()]
        if nan_cols:
            raise DataValidationError(f"Column(s) {nan_cols} in newdata contain NaN values.")

        X = np.asarray(df[self._exogenous].values, dtype=float)
        preds = self._parser.to_callable()(X, *self.params)

        if interval is None:
            return preds

        if interval not in ("confidence", "prediction"):
            raise DataValidationError(
                f"interval must be None, 'confidence' or 'prediction', got '{interval}'."
            )

        J = self._prediction_jacobian(df)  # (n_new, k)
        pred_var = np.einsum("ij,jk,ik->i", J, self.covariance_matrix, J)
        pred_var = np.maximum(pred_var, 0.0)  # guard: numerical noise can produce ~0 negatives
        if interval == "prediction":
            pred_var = pred_var + self.rss / self.resid_df

        se = np.sqrt(pred_var)
        t_val = stats.t.ppf(1 - alpha / 2, self.resid_df)
        return pd.DataFrame(
            {
                "fit": preds,
                "se": se,
                "lower": preds - t_val * se,
                "upper": preds + t_val * se,
            },
            index=df.index,
        )

    def _prediction_jacobian(self, df: pd.DataFrame) -> np.ndarray:
        """Jacobian of the predictions (not of the residuals) with respect
        to the parameters, evaluated at ``df`` — analytical with
        backend='jax', central differences with backend='numpy'."""
        if self.backend == "jax":
            param_dict = dict(zip(self._parameters, self.params))
            return np.asarray(self._parser.jacobian(param_dict, data=df))

        X = np.asarray(df[self._exogenous].values, dtype=float)
        f = self._parser.to_callable()
        theta = self.params
        base = f(X, *theta)
        J = np.zeros((len(base), len(theta)))
        for i in range(len(theta)):
            h = max(abs(theta[i]) * 1e-6, 1e-8)
            theta_plus, theta_minus = theta.copy(), theta.copy()
            theta_plus[i] += h
            theta_minus[i] -= h
            J[:, i] = (f(X, *theta_plus) - f(X, *theta_minus)) / (2 * h)
        return J

    def cross_validate(
        self,
        k: int = 5,
        method: str = "trf",
        loss: str = "linear",
        use_analytical_jacobian: Optional[bool] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        K-fold cross-validation: splits ``self.data`` into k folds, fits a
        NEW model (same formula/backend/extra_functions as this instance,
        with start resolved again by Initializer over each fold's
        training data) on k-1 folds, and evaluates RMSE/MAE/R² on the
        held-out fold.

        Unlike ``rsquared_pred``/``press`` (a cheap local approximation
        via leverage, based on the linearization at the optimum), this
        actually refits k times — more expensive but not dependent on
        that approximation.

        Parameters
        ----------
        k : int, default 5 — number of folds (2 <= k <= n_obs).
        random_state : int, optional — seed for the fold split.
        Remaining parameters: same as ``fit()``, used for every per-fold fit.

        Returns
        -------
        A dict with ``rmse``, ``mae``, ``r2`` (lists of length k, NaN for
        any fold that didn't converge/failed), their means
        (``rmse_mean``, ``mae_mean``, ``r2_mean`` — ignoring NaN), and
        ``n_folds_failed``.

        Raises
        ------
        DataValidationError
            If ``k < 2`` or ``k`` is greater than the number of observations.
        """
        n = len(self.data)
        if k < 2:
            raise DataValidationError(f"k must be >= 2, got {k}.")
        if k > n:
            raise DataValidationError(f"k ({k}) cannot be greater than n_obs ({n}).")

        rng = np.random.default_rng(random_state)
        idx = rng.permutation(n)
        folds = np.array_split(idx, k)

        rmses, maes, r2s = [], [], []
        n_failed = 0
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            train_df = self.data.iloc[train_idx].reset_index(drop=True)
            test_df = self.data.iloc[test_idx].reset_index(drop=True)

            try:
                fold_model = nls(
                    self.formula,
                    train_df,
                    backend=self.backend,
                    extra_functions=self._extra_functions,
                )
                fold_model.fit(
                    method=method,
                    loss=loss,
                    use_analytical_jacobian=use_analytical_jacobian,
                    **kwargs,
                )
                y_pred = fold_model.predict(test_df)
                y_true = np.asarray(test_df[self._endogenous].values, dtype=float)
                resid = y_true - y_pred
                tss_i = np.sum((y_true - np.mean(y_true)) ** 2)
                rmse_i = float(np.sqrt(np.mean(resid**2)))
                mae_i = float(np.mean(np.abs(resid)))
                r2_i = float(1.0 - np.sum(resid**2) / tss_i) if tss_i > 0 else float("nan")
            except Exception:
                n_failed += 1
                rmse_i = mae_i = r2_i = float("nan")

            rmses.append(rmse_i)
            maes.append(mae_i)
            r2s.append(r2_i)

        if n_failed == k:
            warnings.warn(
                f"cross_validate: all {k} folds failed; every metric is NaN.",
                RuntimeWarning,
            )

        return {
            "rmse": rmses,
            "mae": maes,
            "r2": r2s,
            "rmse_mean": float(np.nanmean(rmses)) if n_failed < k else float("nan"),
            "mae_mean": float(np.nanmean(maes)) if n_failed < k else float("nan"),
            "r2_mean": float(np.nanmean(r2s)) if n_failed < k else float("nan"),
            "n_folds_failed": n_failed,
        }

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this fitted NLS model.

        See :mod:`statcraft.plots` for the supported ``kind`` names.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def coef_table(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Coefficient table (coef, std err, t, P>|t|, lower/upper CI) as an
        actual ``pandas.DataFrame`` — unlike the table inside
        ``summary()`` (static HTML/text, meant only for visual reading),
        this is a normal DataFrame: cells can be copied individually
        (right-click on Jupyter's output, or ``.to_clipboard()``),
        indexed, exported to CSV, etc.
        """
        self._require_fitted("coef_table")
        se = np.asarray([self._std_errors[p] for p in self._parameters])
        coefs = self.params
        tvals = coefs / se
        pvals = 2.0 * stats.t.sf(np.abs(tvals), self.resid_df)
        ci = self.conf_int(alpha=alpha)
        return pd.DataFrame(
            {
                "coef": coefs,
                "std err": se,
                "t": tvals,
                "P>|t|": pvals,
                ci.columns[0]: ci.iloc[:, 0].values,
                ci.columns[1]: ci.iloc[:, 1].values,
            },
            index=self._parameters,
        )

    def summary(
        self,
        yname: Optional[str] = None,
        xname: Optional[list] = None,
        title: Optional[str] = None,
        alpha: float = 0.05,
    ) -> "Summary":
        """
        Summary table for the fit, with the same visual layout as
        ``statsmodels``' ``RegressionResults.summary()`` (model info
        block, coefficient table with CIs, residual diagnostics) — but
        self-contained: it doesn't add ``statsmodels`` as a dependency
        (not installed in this project), only ``numpy``/``scipy.stats``,
        which are already dependencies of ``nls``.

        Differences from statsmodels' OLS summary, because NLS doesn't
        have a fixed design matrix:

          - The "eigenvalues" count/condition and the "Cond. No." are
            computed on the jacobian at the optimum (``self.result.jac``),
            the model's local linear approximation — not on a fixed
            exogenous design matrix as in OLS.
          - There's no centered/uncentered R² distinction (there's no
            notion of a "constant" separate from the other parameters in a
            generic NLS model).
          - Coefficient p-values use the t distribution with ``resid_df``
            degrees of freedom (the standard asymptotic approximation for
            NLS, the same one R uses in ``summary.nls``).

        Parameters
        ----------
        yname : str, optional
            Response variable name.
        xname : list[str], optional
            Display names per parameter (must have the same length as
            ``self.parameters``).
        title : str, optional
            Title for the top table.
        alpha : float, default 0.05
            Significance level for the CIs.

        Raises
        ------
        DataValidationError
            If ``xname`` is given but its length doesn't match the number
            of parameters.
        """
        self._require_fitted("summary")

        n, k = len(self._y), len(self._parameters)
        yname = yname or self._endogenous
        xname = list(xname) if xname is not None else list(self._parameters)
        if len(xname) != k:
            raise DataValidationError(
                f"xname must have {k} elements (one per parameter), got {len(xname)}."
            )

        ct = self.coef_table(alpha=alpha)
        coefs = ct["coef"].values
        se = ct["std err"].values

        # Collinearity/local-identifiability diagnostics via the jacobian
        # at the optimum (the NLS analogue of the design matrix).
        J = np.atleast_2d(self.result.jac)
        s = np.linalg.svd(J, compute_uv=False)
        eigvals = np.sort(s**2)[::-1]
        condno = float(s.max() / s.min()) if s.min() > 0 else float("inf")

        resid = self.residuals
        skew = float(stats.skew(resid))
        kurtosis = float(
            stats.kurtosis(resid, fisher=False)
        )  # "Pearson" convention (~3 for normal data)
        omni, omnipv = stats.normaltest(resid)
        jb, jbpv = stats.jarque_bera(resid)
        dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2)) if n > 1 else float("nan")

        notes = []
        if self.cov_type != "nonrobust":
            _hc_desc = {
                "HC0": "no degrees-of-freedom correction.",
                "HC1": "uniform degrees-of-freedom correction (n/(n-k)).",
                "HC2": "each residual weighted by 1/(1-leverage).",
                "HC3": "each residual weighted by 1/(1-leverage)² — more conservative, recommended with small samples.",
            }
            notes.append(
                f"Standard Errors are heteroscedasticity-robust ({self.cov_type}): "
                f"{_hc_desc.get(self.cov_type, '')}"
            )
        if not self.result.success:
            notes.append(
                f"The optimizer did NOT converge: {self.result.message} "
                f"(status={self.result.status}). Results may not be "
                "reliable."
            )
        if eigvals[-1] < 1e-10:
            notes.append(
                f"The smallest eigenvalue of the jacobian is {eigvals[-1]:.3g}. "
                "This may indicate strong collinearity between parameters, or "
                "that the model isn't well identified at this point."
            )
        elif condno > 1000:
            notes.append(
                f"The condition number is large ({condno:.3g}). This may "
                "indicate collinearity between parameters or other numerical "
                "issues."
            )
        if self.backend == "jax" and not self._fit_used_analytical_jac:
            notes.append(
                "backend='jax' but the last fit() used finite differences "
                "(use_analytical_jacobian=False)."
            )

        model_label = (
            f"nls ({self._info.get('func_name_special') or 'formula'})"
            if self._is_special
            else "nls"
        )

        top_left = [
            ("Dep. Variable:", yname),
            ("Model:", model_label),
            ("Backend:", self.backend),
            ("Covariance Type:", self.cov_type),
            ("No. Observations:", str(n)),
            ("Df Residuals:", str(self.resid_df)),
            ("Df Model:", str(k)),
        ]
        top_right = [
            ("R-squared:", f"{self.rsquared:8.4f}"),
            ("Adj. R-squared:", f"{self.adj_rsquared:8.4f}"),
            ("F-statistic:", f"{self.fvalue:8.4g}"),
            ("Prob (F-statistic):", f"{self.f_pvalue:8.3g}"),
            # A fixed one-decimal format is used here (rather than 'g'
            # with the '#' flag) because logLik/AIC/BIC are always
            # "normal"-magnitude numbers for any real fit, and 'g'+'#'
            # produces a broken-looking trailing bare decimal point
            # (e.g. "1390.") whenever the value already has exactly as
            # many significant digits as the precision before the point.
            ("Log-Likelihood:", f"{self.logLik:8.1f}"),
            ("AIC:", f"{self.aic:8.1f}"),
            ("BIC:", f"{self.bic:8.1f}"),
        ]
        diagn_left = [
            ("Omnibus:", f"{omni:#8.3f}"),
            ("Prob(Omnibus):", f"{omnipv:#8.3f}"),
            ("Skew:", f"{skew:#8.3f}"),
            ("Kurtosis:", f"{kurtosis:#8.3f}"),
        ]
        diagn_right = [
            ("Durbin-Watson:", f"{dw:#8.3f}"),
            ("Jarque-Bera (JB):", f"{jb:#8.3f}"),
            ("Prob(JB):", f"{jbpv:8.3g}"),
            ("Cond. No. (jac):", f"{condno:8.3g}"),
        ]

        return build_summary(
            title=title or f"{model_label} Regression Results",
            subtitle=[
                f"Formula: {self.formula}",
                f"Method: least_squares (method='{self._fit_method}', loss='{self._fit_loss}')",
            ],
            left=top_left,
            right=top_right,
            names=xname,
            coefs=coefs,
            se=se,
            dist="t",
            df=self.resid_df,
            diag_left=diagn_left,
            diag_right=diagn_right,
            alpha=alpha,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this fitted model to a plain, JSON-serializable dict:
        formula, backend, training data, and every fit-derived attribute
        needed to fully restore a working instance with ``from_dict()``
        (without re-running the optimizer) — ``coefficients``,
        ``covariance_matrix``, ``fitted_values``, ``residuals``, enough of
        the ``OptimizeResult`` for ``leverage``/``cooks_distance``/
        ``summary()`` to keep working (``jac``, ``success``, ``message``,
        ``status``, ``cost``), and the fit configuration
        (``cov_type``, method, loss, whether the analytical jacobian was
        used).

        Only supports models built from the built-in function namespace
        (no ``extra_functions``) — this covers plain ``nls(...)`` usage.
        Arbitrary Python callables (as used by the ``NLS`` wrapper class,
        or by ``nls(..., extra_functions=...)``) cannot be safely
        represented as JSON; use Python's ``pickle`` module for those
        instead (this class is picklable — see ``__getstate__``):
        ``pickle.dump(model, open('model.pkl', 'wb'))``.

        See also ``save()`` to write this directly to a JSON file, and
        ``from_dict()``/``load()`` to restore it.

        Raises
        ------
        NotFittedError
            If called before ``fit()``/``fit_multistart()``.
        NLSError
            If this instance was built with ``extra_functions`` (custom
            functions can't be represented as JSON — use ``pickle``
            instead, as noted above).
        """
        self._require_fitted("to_dict")
        if self._extra_functions:
            raise NLSError(
                "to_dict()/save() only support models built from the "
                "built-in function namespace (no extra_functions), because "
                "arbitrary Python callables cannot be safely represented as "
                "JSON. Use Python's pickle module instead for models built "
                "with extra_functions or with the NLS wrapper class: "
                "pickle.dump(model, open('model.pkl', 'wb'))."
            )
        return {
            "__nls_format_version__": 1,
            "formula": self.formula,
            "backend": self.backend,
            "data": self.data.to_dict(orient="list"),
            "start": self.start,
            "fit_method": self._fit_method,
            "fit_loss": self._fit_loss,
            "fit_used_analytical_jac": self._fit_used_analytical_jac,
            "cov_type": self.cov_type,
            "coefficients": self.coefficients,
            "covariance_matrix": np.asarray(self.covariance_matrix).tolist(),
            "fitted_values": np.asarray(self.fitted_values).tolist(),
            "residuals": np.asarray(self.residuals).tolist(),
            "result": {
                "success": bool(self.result.success),
                "message": str(self.result.message),
                "status": int(self.result.status),
                "cost": float(self.result.cost),
                "jac": np.asarray(self.result.jac).tolist(),
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "nls":
        """
        Restores a fitted model previously serialized with ``to_dict()``.
        Rebuilds the instance through the normal, validated construction
        path (formula parsing, exogenous/NaN/Inf checks) using the stored
        ``start`` as the initial value, then applies the stored fit
        results directly — the optimizer is not re-run, so this is cheap
        and reproduces the exact model that was saved, without depending
        on ``least_squares`` converging to the same point again.

        See also ``load()`` to read directly from a JSON file.
        """
        data = pd.DataFrame(d["data"])
        model = cls(d["formula"], data, start=d["start"], backend=d.get("backend", "numpy"))

        r = d["result"]
        model.result = SimpleNamespace(
            success=r["success"],
            message=r["message"],
            status=r["status"],
            cost=r["cost"],
            jac=np.asarray(r["jac"]),
            x=np.asarray(list(d["coefficients"].values())),
        )
        model._fit_method = d["fit_method"]
        model._fit_loss = d["fit_loss"]
        model._fit_used_analytical_jac = d["fit_used_analytical_jac"]
        model.cov_type = d["cov_type"]
        model.coefficients = dict(d["coefficients"])
        model.covariance_matrix = np.asarray(d["covariance_matrix"])
        model.fitted_values = np.asarray(d["fitted_values"])
        model.residuals = np.asarray(d["residuals"])
        model._std_errors = dict(
            zip(
                model._parameters,
                np.sqrt(np.maximum(np.diag(model.covariance_matrix), 0.0)),
            )
        )
        return model

    def save(self, path: str) -> None:
        """Writes ``self.to_dict()`` to ``path`` as JSON. See ``to_dict()``
        for what is (and isn't) supported."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "nls":
        """Reads a model previously written with ``save()`` and restores
        it via ``from_dict()``."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """
        Makes ``nls`` (and ``NLS``) instances picklable
        (``pickle.dump``/``load``, ``copy.deepcopy``, ``joblib.dump``,
        multiprocessing, etc), including models built with
        ``extra_functions`` (e.g. via the ``NLS`` wrapper class) — unlike
        ``to_dict()``/``save()`` (JSON, portable, but limited to models
        with no custom functions), pickling can serialize the underlying
        Python function object itself, subject to Python's own pickling
        rules (module-level named functions pickle fine; lambdas and
        locally-defined functions generally do not).

        The cached residual/jacobian closures (``self._residual_fn``,
        ``self._jac_fn``) are functions built inside a method — not
        picklable as-is — so they're excluded here and rebuilt by
        ``__setstate__`` on unpickling via ``_build_residual_and_jac()``,
        which also relies on ``self._parser`` restoring its own cached
        functions the same way (see ``FormulaParser.__getstate__``).
        """
        state = self.__dict__.copy()
        state.pop("_residual_fn", None)
        state.pop("_jac_fn", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._residual_fn, self._jac_fn = self._build_residual_and_jac()
