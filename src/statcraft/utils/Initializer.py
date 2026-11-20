# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
Initializer
===========
Computes starting values (p0) for nonlinear least-squares estimation.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .exceptions import DataValidationError
from .FormulaParser import FormulaParser


class Initializer:
    """
    Computes initial parameter values for a nonlinear regression model.
    """

    # Maps an SS* function name to either:
    #  - a str: the name of a bound method defined on this class
    #    (resolved via getattr(self, name) — the built-in heuristics
    #    below), or
    #  - a callable: a dynamically registered initializer (see
    #    register_initializer()), invoked as init_fn(self) so it can use
    #    self._exog / self._endog / self.params / self._safe_return /
    #    self._require_distinct the same way the built-in ones do.
    _SS_INIT_MAP: Dict[str, Any] = {
        "SSlogis": "_init_SSlogis",
        "SSexp": "_init_SSexp",
        "SSmicmen": "_init_SSmicmen",
        "SSgompertz": "_init_SSgompertz",
        "SSweibull": "_init_SSweibull",
        "SSasymp": "_init_SSasymp",
        "SSfpl": "_init_SSfpl",
        "SSbiexp": "_init_SSbiexp",
        "SSlogis5": "_init_SSlogis5",
        "SSlin": "_init_SSlin",
        "SSquad": "_init_SSquad",
        "SSpower": "_init_SSpower",
        "SShill": "_init_SShill",
        "SSgauss": "_init_SSgauss",
    }

    @classmethod
    def register_initializer(
        cls,
        name: str,
        init_fn: Callable[["Initializer"], Dict[str, float]],
        overwrite: bool = False,
    ) -> None:
        """
        Registers a starting-value heuristic for a self-starting ``SS*``
        model, so ``Initializer`` can resolve a smart ``p0`` for it
        instead of falling back to 1.0 for every parameter.

        Pairs with ``FormulaParser.register_special_function()``: that
        method alone is enough to make a new ``SS*`` formula *usable*
        (with the 1.0-for-everything fallback); this one lets it also get
        a real initial-value guess.

        Parameters
        ----------
        name : str
            The ``SS*`` function name this initializer applies to — must
            match the ``name`` used in a prior
            ``FormulaParser.register_special_function()`` call (or one of
            the built-in ``SS*`` names, to override its heuristic).
        init_fn : callable
            ``init_fn(initializer) -> Dict[str, float]``, called with the
            ``Initializer`` instance itself, so it can use the same
            building blocks the built-in heuristics use:
            ``initializer._exog`` (exogenous data, 1-D array — only
            populated when there's exactly one exogenous variable, ``None``
            otherwise), ``initializer._endog`` (response data, 1-D array),
            ``initializer.params`` (parameter names, in formula order),
            ``initializer._safe_return(result, func_name)`` (validates the
            returned dict and falls back to 1.0-for-everything with a
            warning if it's malformed or non-finite — wrap your return
            value with it), and ``initializer._require_distinct(x, n,
            label)`` (raises ``DataValidationError`` if fewer than ``n``
            distinct x values are available — call it up front if your
            heuristic needs, say, a linear fit on a subset of points).
        overwrite : bool, default False
            If an initializer is already registered for ``name`` (built-in
            or registered earlier), raises ``DataValidationError`` unless
            ``overwrite=True``.

        Example
        -------
        >>> def _init_SSmymodel(init):
        ...     a_guess = float(np.max(init._endog))
        ...     b_guess = 1.0
        ...     return init._safe_return(
        ...         {"a": a_guess, "b": b_guess}, "SSmymodel"
        ...     )
        >>> Initializer.register_initializer("SSmymodel", _init_SSmymodel)
        """
        if name in cls._SS_INIT_MAP and not overwrite:
            raise DataValidationError(
                f"An initializer is already registered for '{name}'. "
                "Pass overwrite=True to replace it."
            )
        if not callable(init_fn):
            raise DataValidationError(
                "init_fn must be callable: init_fn(initializer) -> Dict[str, float]."
            )
        cls._SS_INIT_MAP[name] = init_fn

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        p0: Optional[Dict[str, float]] = None,
        extra_functions: Optional[Dict[str, Any]] = None,
        parser: Optional[FormulaParser] = None,
    ) -> None:
        # If the caller already built a FormulaParser (typically `nls`,
        # which needs one for itself anyway), it's reused instead of
        # parsing/validating the same formula again from scratch
        # (duplicate sympify + namespace checks per model instance, and
        # multiplied by k in cross_validate). It's safe to reuse a parser
        # with backend="jax" here: Initializer only reads metadata
        # (params, exogenous, is_special, etc.), it never evaluates the
        # numeric backend's functions.
        self.parser = (
            parser
            if parser is not None
            else FormulaParser(formula, data, extra_functions=extra_functions)
        )
        self.params: List[str] = self.parser.params

        self._endog: np.ndarray = np.asarray(
            data[self.parser.endogenous].values, dtype=float
        ).ravel()

        exog_cols = self.parser.exogenous_vars
        if len(exog_cols) == 1:
            self._exog: np.ndarray = np.asarray(data[exog_cols].values, dtype=float).flatten()
        else:
            self._exog = None  # type: ignore[assignment]  # sentinel: >1 exogenous column

        self._p0_user = self._validate_p0(p0)
        self._initial_values: Dict[str, float] = self._resolve()

    def _validate_p0(self, p0: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if p0 is None:
            return None

        if not isinstance(p0, dict):
            raise TypeError(f"p0 must be a dict, got {type(p0).__name__}.")

        expected = set(self.params)
        provided = set(p0.keys())

        extra = provided - expected
        missing = expected - provided

        messages = []
        if extra:
            messages.append(f"unexpected keys: {sorted(extra)}")
        if missing:
            messages.append(f"missing keys: {sorted(missing)}")

        if messages:
            raise DataValidationError(
                f"p0 does not match the formula parameters ({', '.join(messages)}). "
                f"Expected: {sorted(expected)}"
            )

        coerced: Dict[str, float] = {}
        bad: List[str] = []
        for k, v in p0.items():
            fv = float(v)
            if not np.isfinite(fv):
                bad.append(k)
            coerced[k] = fv
        if bad:
            raise DataValidationError(
                f"p0 contains non-finite value(s) for parameter(s) {sorted(bad)}. "
                "Starting values must be finite numbers."
            )
        return coerced

    def _resolve(self) -> Dict[str, float]:
        if self._p0_user is not None:
            return dict(self._p0_user)

        if self.parser.is_special:
            heuristic = self._get_special_initialization()
            if heuristic is not None:
                return heuristic

        return self._fallback()

    def _fallback(self) -> Dict[str, float]:
        return {p: 1.0 for p in self.params}

    def _get_special_initialization(self) -> Optional[Dict[str, float]]:
        func_name = self.parser.func_name_special
        if func_name is None:
            return None
        entry = self._SS_INIT_MAP.get(func_name)
        if entry is None:
            return None
        if isinstance(entry, str):
            method: Callable[[], Dict[str, float]] = getattr(self, entry)
            return method()
        # A dynamically registered callable (see register_initializer()):
        # invoked with `self` so it can use self._exog/_endog/params/
        # _safe_return/_require_distinct the same way the built-in
        # heuristics above do.
        return entry(self)

    def _require_distinct(self, x: np.ndarray, n: int, label: str) -> None:
        if len(np.unique(x)) < n:
            raise DataValidationError(
                f"{label}: need at least {n} distinct x values, got {len(np.unique(x))}."
            )

    def _safe_return(
        self,
        result: Dict[str, float],
        func_name: str,
    ) -> Dict[str, float]:
        if set(result.keys()) != set(self.params):
            warnings.warn(
                f"{func_name} initialiser returned wrong keys "
                f"({sorted(result.keys())} vs {sorted(self.params)}). "
                "Falling back to default values.",
                UserWarning,
                stacklevel=3,
            )
            return self._fallback()

        non_finite = [k for k, v in result.items() if not np.isfinite(v)]
        if non_finite:
            warnings.warn(
                f"{func_name} initialiser produced non-finite value(s) for "
                f"{sorted(non_finite)} (degenerate fit). "
                "Falling back to default values.",
                UserWarning,
                stacklevel=3,
            )
            return self._fallback()

        return {k: float(v) for k, v in result.items()}

    def _init_SSlogis(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
            self._require_distinct(x, 4, "SSlogis")
            if np.all(y == y[0]):
                raise ValueError("All y values are identical.")
            y_min, y_max = np.min(y), np.max(y)
            dy = y_max - y_min
            if dy == 0:
                raise ValueError("y range is zero.")
            z = (y - y_min + 0.05 * dy) / (1.1 * dy)
            logit_z = np.log(z / (1.0 - z))
            slope, intercept = np.polyfit(logit_z, x, 1)
            result = {
                self.params[0]: float(y_max),
                self.params[1]: float(intercept),
                self.params[2]: float(slope),
            }
            return self._safe_return(result, "SSlogis")
        except Exception as exc:
            warnings.warn(
                f"SSlogis initialisation failed ({exc}). Using default values (1.0).",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback()

    def _init_SSexp(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            mask = y > 0
            x_pos, y_pos = x[mask], y[mask]
            self._require_distinct(x_pos, 2, "SSexp")
            slope, intercept = np.polyfit(x_pos, np.log(y_pos), 1)
            result = {
                self.params[0]: float(np.exp(intercept)),
                self.params[1]: float(slope),
            }
            return self._safe_return(result, "SSexp")
        except Exception as exc:
            warnings.warn(
                f"SSexp initialisation failed ({exc}). Using default values (1.0 / 0.1).",
                UserWarning,
                stacklevel=2,
            )
            special_fallback = {self.params[0]: 1.0, self.params[1]: 0.1}
            return self._safe_return(special_fallback, "SSexp (fallback)")

    def _init_SSmicmen(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            self._require_distinct(x, 3, "SSmicmen")
            mask = (x != 0) & (y != 0)
            x_v, y_v = x[mask], y[mask]
            if len(x_v) < 2:
                rough = {
                    self.params[0]: float(np.max(y)),
                    self.params[1]: float(np.median(x)),
                }
                return self._safe_return(rough, "SSmicmen (rough fallback)")
            self._require_distinct(x_v, 2, "SSmicmen (post-mask)")
            slope, intercept = np.polyfit(1.0 / x_v, 1.0 / y_v, 1)
            if abs(intercept) < 1e-12:
                raise ValueError("Lineweaver-Burk intercept is near zero.")
            Vm = abs(1.0 / intercept)
            K = abs(slope / intercept)
            result = {self.params[0]: float(Vm), self.params[1]: float(K)}
            return self._safe_return(result, "SSmicmen")
        except Exception as exc:
            warnings.warn(
                f"SSmicmen initialisation failed ({exc}). Using default values (1.0).",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback()

    def _init_SSgompertz(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            Asym = float(np.max(y) * 1.1)
            y_clipped = np.clip(y, 1e-10, Asym * 0.9999)
            z = np.log(Asym / y_clipped)
            w = np.log(z)
            mask = np.isfinite(w) & (z > 0) & (y > 0.1 * Asym)
            if np.sum(mask) < 2:
                raise ValueError("Not enough valid points after double-log transform.")
            self._require_distinct(x[mask], 2, "SSgompertz (post-mask)")
            log_b3, log_b2 = np.polyfit(x[mask], w[mask], 1)
            result = {
                self.params[0]: float(Asym),
                self.params[1]: float(np.exp(log_b2)),
                self.params[2]: float(np.exp(log_b3)),
            }
            return self._safe_return(result, "SSgompertz")
        except Exception as exc:
            warnings.warn(
                f"SSgompertz initialisation failed ({exc}). Using default values (1.0).",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback()

    def _init_SSweibull(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            self._require_distinct(x, 5, "SSweibull")
            if np.any(x < 0):
                raise ValueError("All x values must be non-negative for SSweibull.")
            Rasym, Lasym = float(np.max(y)), float(np.min(y))
            mask = (x > 0) & (y != Rasym) & (y != Lasym)
            x_m, y_m = x[mask], y[mask]
            if len(x_m) < 2:
                raise ValueError("Not enough interior points for SSweibull fit.")
            denom = Rasym - Lasym
            if abs(denom) < 1e-12:
                raise ValueError("y range is near zero.")
            z = np.log(-np.log((Rasym - y_m) / denom))
            slope, intercept = np.polyfit(np.log(x_m), z, 1)
            result = {
                self.params[0]: float(Rasym),
                self.params[1]: float(Rasym - Lasym),
                self.params[2]: float(intercept),
                self.params[3]: float(slope),
            }
            return self._safe_return(result, "SSweibull")
        except Exception as exc:
            warnings.warn(
                f"SSweibull initialisation failed ({exc}). Using default values (1.0).",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback()

    def _init_SSasymp(self) -> Dict[str, float]:
        try:
            idx = np.argsort(self._exog)
            x, y = self._exog[idx], self._endog[idx]
            self._require_distinct(x, 3, "SSasymp")
            if len(x) > 3:
                Asym_hat = float(np.mean(y[-3:]))
                ydiff = np.abs(y - Asym_hat)
                ydiff = np.where(ydiff < 1e-10, 1e-10, ydiff)
                coef = np.polyfit(x, np.log(ydiff), 1)
                lrc = float(np.log(max(-coef[0], 1e-6)))
                rc = np.exp(lrc)
                exp_term = np.exp(-rc * x)
                X_mat = np.column_stack((1.0 - exp_term, exp_term))
                beta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
                result = {
                    self.params[0]: float(beta[0]),
                    self.params[1]: float(beta[1]),
                    self.params[2]: float(lrc),
                }
            else:
                ydiff = np.diff(y)
                if np.prod(ydiff) <= 0:
                    raise ValueError("y is not monotone — cannot use 3-point SSasymp init.")
                xdiff = np.diff(np.unique(x))
                if not np.isclose(xdiff[0], xdiff[1]):
                    raise ValueError("3-point SSasymp init requires evenly-spaced x.")
                frac = (y[2] - y[0]) / (y[1] - y[0])
                expmRd = frac - 1.0
                rc = -np.log(expmRd) / xdiff[0]
                lrc = float(np.log(max(rc, 1e-12)))
                expmRx1 = np.exp(-rc * x[0])
                bma = ydiff[0] / (expmRx1 * (expmRd - 1.0))
                Asym = float(y[0] - bma * expmRx1)
                R0 = float(bma + Asym)
                result = {self.params[0]: Asym, self.params[1]: R0, self.params[2]: lrc}
            return self._safe_return(result, "SSasymp")
        except Exception as exc:
            warnings.warn(
                f"SSasymp initialisation failed ({exc}). Using default values (1.0).",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback()

    def _init_SSfpl(self) -> Dict[str, float]:
        try:
            idx = np.argsort(self._exog)
            x, y = self._exog[idx], self._endog[idx]
            self._require_distinct(x, 5, "SSfpl")
            y_min, y_max = float(y.min()), float(y.max())
            drng = y_max - y_min
            if drng < 1e-12:
                raise ValueError("y range is near zero.")
            prop = (y - y_min + 0.05 * drng) / (1.1 * drng)
            z = np.log(prop / (1.0 - prop))
            ir1, ir0 = np.polyfit(z, x, 1)
            scal = float(abs(ir1))
            xmid = float(ir0)
            exp_term = 1.0 / (1.0 + np.exp((xmid - x) / max(scal, 1e-8)))
            X_mat = np.column_stack((np.ones(len(x)), exp_term))
            beta, *_ = np.linalg.lstsq(X_mat, y, rcond=None)
            A, B = float(beta[0]), float(beta[0] + beta[1])
            result = {
                self.params[0]: A,
                self.params[1]: B,
                self.params[2]: xmid,
                self.params[3]: scal,
            }
            return self._safe_return(result, "SSfpl")
        except Exception as exc:
            warnings.warn(
                f"SSfpl initialisation failed ({exc}). Using default values (1.0).",
                UserWarning,
                stacklevel=2,
            )
            return self._fallback()

    def _init_SSbiexp(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
            mask = y > 0
            x_pos, y_pos = x[mask], y[mask]
            self._require_distinct(x_pos, 5, "SSbiexp")
            mid_idx = len(x_pos) // 2
            x_slow, y_slow = x_pos[mid_idx:], y_pos[mid_idx:]
            x_fast, y_fast = x_pos[:mid_idx], y_pos[:mid_idx]
            slope_slow, intercept_slow = np.polyfit(x_slow, np.log(y_slow), 1)
            a2 = np.exp(intercept_slow)
            rate_slow = -slope_slow if slope_slow < 0 else 1e-3
            lrc2 = np.log(rate_slow)
            y_slow_pred = a2 * np.exp(-rate_slow * x_fast)
            residuals = y_fast - y_slow_pred
            res_mask = residuals > 1e-5
            if np.sum(res_mask) < 2:
                a1 = max(float(y_pos[0]) - a2, 1e-3)
                lrc1 = lrc2 + 1.0
            else:
                x_res, y_res = x_fast[res_mask], residuals[res_mask]
                slope_fast, intercept_fast = np.polyfit(x_res, np.log(y_res), 1)
                a1 = np.exp(intercept_fast)
                rate_fast = -slope_fast if slope_fast < 0 else rate_slow * 2.0
                rate_fast = max(rate_fast, rate_slow * 1.1)
                lrc1 = np.log(rate_fast)
            result = {
                self.params[0]: float(a1),
                self.params[1]: float(lrc1),
                self.params[2]: float(a2),
                self.params[3]: float(lrc2),
            }
            return self._safe_return(result, "SSbiexp")
        except Exception as exc:
            warnings.warn(
                f"SSbiexp initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            special_fallback = {
                self.params[0]: 1.0,
                self.params[1]: 0.0,
                self.params[2]: 1.0,
                self.params[3]: -1.0,
            }
            return self._safe_return(special_fallback, "SSbiexp (fallback)")

    def _init_SSlogis5(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
            self._require_distinct(x, 6, "SSlogis5")
            if np.all(y == y[0]):
                raise ValueError("All y values are identical.")
            n_points = max(1, len(y) // 10)
            a_init, b_init = float(np.mean(y[:n_points])), float(np.mean(y[-n_points:]))
            dy = b_init - a_init
            if abs(dy) < 1e-5:
                y_min, y_max = np.min(y), np.max(y)
                dy = y_max - y_min
                a_init, b_init = float(y_min), float(y_max)
                if abs(dy) < 1e-5:
                    raise ValueError("y range is zero.")
            g_init = 1.0
            y_min, y_max = min(a_init, b_init), max(a_init, b_init)
            global_dy = y_max - y_min if (y_max - y_min) > 1e-5 else 1.0
            z = (y - y_min + 0.05 * global_dy) / (1.1 * global_dy)
            z = np.clip(z, 1e-4, 1.0 - 1e-4)
            if b_init < a_init:
                z = 1.0 - z
            logit_z = np.log((1.0 - z) / z)
            slope, intercept = np.polyfit(logit_z, x, 1)
            scal_init = float(slope) if abs(slope) > 1e-5 else 1.0
            if b_init < a_init:
                scal_init = -scal_init
            xmid_init = float(intercept)
            result = {
                self.params[0]: a_init,
                self.params[1]: b_init,
                self.params[2]: xmid_init,
                self.params[3]: scal_init,
                self.params[4]: g_init,
            }
            return self._safe_return(result, "SSlogis5")
        except Exception as exc:
            warnings.warn(
                f"SSlogis5 initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            special_fallback = {
                self.params[0]: float(np.min(self._endog)),
                self.params[1]: float(np.max(self._endog)),
                self.params[2]: float(np.mean(self._exog)),
                self.params[3]: 1.0,
                self.params[4]: 1.0,
            }
            return self._safe_return(special_fallback, "SSlogis5 (fallback)")

    def _init_SSlin(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            idx = np.argsort(x_valid)
            x_valid, y_valid = x_valid[idx], y_valid[idx]
            self._require_distinct(x_valid, 2, "SSlin")
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            result = {self.params[0]: float(intercept), self.params[1]: float(slope)}
            return self._safe_return(result, "SSlin")
        except Exception as exc:
            warnings.warn(
                f"SSlin initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            y_clean = self._endog[np.isfinite(self._endog)]
            mean_y = float(np.mean(y_clean)) if len(y_clean) > 0 else 1.0
            special_fallback = {self.params[0]: mean_y, self.params[1]: 0.0}
            return self._safe_return(special_fallback, "SSlin (fallback)")

    def _init_SSquad(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            idx = np.argsort(x_valid)
            x_valid, y_valid = x_valid[idx], y_valid[idx]
            self._require_distinct(x_valid, 3, "SSquad")
            coefs = np.polyfit(x_valid, y_valid, 2)
            c_init, b_init, a_init = coefs[0], coefs[1], coefs[2]
            result = {
                self.params[0]: float(a_init),
                self.params[1]: float(b_init),
                self.params[2]: float(c_init),
            }
            return self._safe_return(result, "SSquad")
        except Exception as exc:
            warnings.warn(
                f"SSquad initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            y_clean = self._endog[np.isfinite(self._endog)]
            mean_y = float(np.mean(y_clean)) if len(y_clean) > 0 else 1.0
            special_fallback = {self.params[0]: mean_y, self.params[1]: 0.0, self.params[2]: 0.0}
            return self._safe_return(special_fallback, "SSquad (fallback)")

    def _init_SSpower(self) -> Dict[str, float]:
        default_names = ["a", "b", "c"]
        p_names = list(self.params) if hasattr(self, "params") else []
        while len(p_names) < 3:
            p_names.append(default_names[len(p_names)])
        try:
            x, y = self._exog, self._endog
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            pos_mask = (x_valid > 0) & (y_valid > 0)
            x_pos, y_pos = x_valid[pos_mask], y_valid[pos_mask]
            self._require_distinct(x_pos, 3, "SSpower")
            idx = np.argsort(x_pos)
            x_pos, y_pos = x_pos[idx], y_pos[idx]
            c_init = float(np.min(y_pos)) * 0.9
            y_shifted = y_pos - c_init
            slope, intercept = np.polyfit(np.log(x_pos), np.log(y_shifted), 1)
            b_init, a_init = float(slope), float(np.exp(intercept))
            result = {p_names[0]: a_init, p_names[1]: b_init, p_names[2]: c_init}
            return self._safe_return(result, "SSpower")
        except Exception as exc:
            warnings.warn(
                f"SSpower initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            y_data: Any = (
                self._endog if (hasattr(self, "_endog") and self._endog is not None) else []
            )
            c_fallback = float(np.min(y_data)) if len(y_data) > 0 else 0.0
            special_fallback = {p_names[0]: 1.0, p_names[1]: 0.5, p_names[2]: c_fallback}
            return self._safe_return(special_fallback, "SSpower (fallback)")

    def _init_SShill(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            pos_mask = (x_valid > 0) & (y_valid > 0)
            x_pos, y_pos = x_valid[pos_mask], y_valid[pos_mask]
            self._require_distinct(x_pos, 3, "SShill")
            idx = np.argsort(x_pos)
            x_pos, y_pos = x_pos[idx], y_pos[idx]
            n_points = max(1, len(y_pos) // 10)
            vm_init = float(np.mean(y_pos[-n_points:]))
            vm_init = max(vm_init, float(np.max(y_pos)) * 1.05)
            z = np.clip(y_pos / vm_init, 0.01, 0.99)
            slope, intercept = np.polyfit(np.log(x_pos), np.log(z / (1.0 - z)), 1)
            n_init = float(slope) if slope > 1e-3 else 1.0
            try:
                k_init = float(np.exp(-intercept / n_init))
            except (OverflowError, ZeroDivisionError):
                k_init = float(np.median(x_pos))
            k_init = np.clip(k_init, np.min(x_pos), np.max(x_pos))
            result = {self.params[0]: vm_init, self.params[1]: k_init, self.params[2]: n_init}
            return self._safe_return(result, "SShill")
        except Exception as exc:
            warnings.warn(
                f"SShill initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            y_clean = self._endog[np.isfinite(self._endog)] if len(self._endog) > 0 else [1.0]
            x_clean = self._exog[np.isfinite(self._exog)] if len(self._exog) > 0 else [1.0]
            special_fallback = {
                self.params[0]: float(np.max(y_clean)) * 1.1,
                self.params[1]: float(np.median(x_clean)),
                self.params[2]: 1.0,
            }
            return self._safe_return(special_fallback, "SShill (fallback)")

    def _init_SSgauss(self) -> Dict[str, float]:
        try:
            x, y = self._exog, self._endog
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            pos_mask = y_valid > 0
            x_pos, y_pos = x_valid[pos_mask], y_valid[pos_mask]
            self._require_distinct(x_pos, 3, "SSgauss")
            idx = np.argsort(x_pos)
            x_pos, y_pos = x_pos[idx], y_pos[idx]
            coefs = np.polyfit(x_pos, np.log(y_pos), 2)
            C, B, A = coefs[0], coefs[1], coefs[2]
            if C < -1e-8:
                c_init = float(np.sqrt(-1.0 / (2.0 * C)))
                b_init = float(B * (c_init**2))
                a_init = float(np.exp(A + (b_init**2) / (2.0 * (c_init**2))))
            else:
                raise ValueError("Log-linearization did not yield a downward parabola.")
            if a_init > np.max(y_pos) * 5.0 or a_init < np.max(y_pos) * 0.1:
                a_init = float(np.max(y_pos))
            if b_init < np.min(x_pos) or b_init > np.max(x_pos):
                b_init = float(x_pos[np.argmax(y_pos)])
            if c_init <= 0 or c_init > (np.max(x_pos) - np.min(x_pos)):
                c_init = float((np.max(x_pos) - np.min(x_pos)) / 4.0)
            result = {self.params[0]: a_init, self.params[1]: b_init, self.params[2]: c_init}
            return self._safe_return(result, "SSgauss")
        except Exception as exc:
            warnings.warn(
                f"SSgauss initialisation failed ({exc}). Using default values.",
                UserWarning,
                stacklevel=2,
            )
            y_clean = self._endog[np.isfinite(self._endog)] if len(self._endog) > 0 else [1.0]
            x_clean = self._exog[np.isfinite(self._exog)] if len(self._exog) > 0 else [0.0, 1.0]
            max_idx = np.argmax(y_clean) if len(y_clean) > 0 else 0
            special_fallback = {
                self.params[0]: float(np.max(y_clean)) if len(y_clean) > 0 else 1.0,
                self.params[1]: float(x_clean[max_idx]) if len(x_clean) > max_idx else 0.0,
                self.params[2]: float((np.max(x_clean) - np.min(x_clean)) / 4.0)
                if len(x_clean) > 0
                else 1.0,
            }
            return self._safe_return(special_fallback, "SSgauss (fallback)")

    @property
    def initial_values(self) -> Dict[str, float]:
        return dict(self._initial_values)

    @property
    def theta(self) -> List[float]:
        return [self._initial_values[p] for p in self.params]

    def __repr__(self) -> str:
        pairs = ", ".join(f"{p}={v:.4g}" for p, v in self._initial_values.items())
        return f"Initializer({self.parser.formula_original!r}, p0={{{pairs}}})"
