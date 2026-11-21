# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Normalise the many fitted-result objects of statcraft to one plot shape.

The plotting modules never branch on a concrete model class.  They ask this
adapter for response values, fitted values, residuals, leverage, coefficient
tables and random effects, and each result family is resolved exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "PlotData",
    "resolve",
    "is_fitted",
    "model_name",
    "response_name",
    "coefficient_table",
    "random_effects",
    "default_exponentiate",
    "has_coefficients",
    "has_random_effects",
    "family_name",
]


# ---------------------------------------------------------------------------
# Small, defensive accessors
# ---------------------------------------------------------------------------


def _value(obj: Any, name: str, *args, default=None, **kwargs):
    """Read a property, or call a method (the result API is inconsistent)."""
    if obj is None or not hasattr(obj, name):
        return default
    value = getattr(obj, name)
    if callable(value):
        try:
            return value(*args, **kwargs)
        except TypeError:
            return default
        except Exception:
            return default
    return value


def _attr(obj: Any, name: str, default=None):
    """Read an attribute defensively (some properties raise for certain families).

    For example ``GLM.linear_predictors`` is undefined for multinomial fits;
    ``getattr`` would propagate that error instead of returning ``None``.
    """
    if obj is None:
        return default
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _series(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(pd.Series(value), dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return _array(value)


def _first(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def is_fitted(result: Any) -> bool:
    """Whether ``result`` looks like a completed fit.

    ``GLM`` keeps an explicit ``_fitted`` flag; the NLS class exposes its
    ``coefficients`` only after ``fit()``; the other result classes are only
    constructed after a successful fit.
    """
    if result is None:
        return False
    if hasattr(result, "_fitted"):
        return bool(getattr(result, "_fitted"))
    name = type(result).__name__
    if name in ("nls", "NLS"):
        return hasattr(result, "coefficients")
    return True


def family_name(result: Any) -> str | None:
    """The family/class label used in titles."""
    fam = getattr(result, "family", None)
    if fam is None:
        return None
    return type(fam).__name__ if not isinstance(fam, str) else fam


def model_name(result: Any) -> str:
    """A compact, user-facing label for the fitted result."""
    name = type(result).__name__
    fam = family_name(result)
    labels = {
        "GLM": "GLM",
        "nls": "NLS",
        "NLS": "NLS",
        "LmeFit": "LME" if getattr(result, "kind", None) != "gls" else "GLS",
        "NlmeFit": "NLME" if getattr(result, "kind", None) not in ("gnls",) else "GNLS",
        "LmerFit": "LMER",
        "GlmerFit": "GLMER",
        "ListFit": "List fit",
        "SaemFit": "SAEM",
        "MCGLMResults": "MCGLM",
    }
    base = labels.get(name, name)
    if fam and name == "GLM":
        return f"{base} ({fam})"
    return base


def response_name(result: Any) -> str:
    """The name of the response variable, when the result records one."""
    if type(result).__name__ == "SaemFit":
        # SAEM keeps the modelling name in ``DataInfo``; falling through to the
        # formula would produce a generic ``response`` label in VPC/NPDE panels.
        y_col = getattr(getattr(result, "info", None), "y_col", None)
        if y_col:
            return str(y_col)
    if hasattr(result, "endog_names"):
        value = getattr(result, "endog_names")
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if value:
            return str(value)
    if hasattr(result, "model") and hasattr(result.model, "endog_names"):
        value = getattr(result.model, "endog_names")
        if isinstance(value, (list, tuple)) and value:
            return " + ".join(str(v) for v in value)
        if value:
            return str(value)
    if hasattr(result, "_endogenous"):
        return str(getattr(result, "_endogenous"))
    formula = getattr(result, "formula", None)
    if formula:
        return str(formula).split("~", 1)[0].strip()
    return "response"


def _fit_data(result: Any) -> pd.DataFrame | None:
    for attr in ("_data", "data"):
        value = getattr(result, attr, None)
        if isinstance(value, pd.DataFrame):
            return value
    return None


_GROUP_HINTS = (
    "plot", "subject", "group", "id", "unit", "curve", "series",
    "individual", "animal", "mare", "cluster", "site",
)
_TIME_LABELS = {"time", "times", "t", "date", "day", "age", "dose", "visit", "wave"}


def _inferred_gnls_groups(data: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """Pick the repeated-measures factor of an unstructured GNLS data frame.

    GNLS has no random effects, so a correlation-free fit does not record a
    grouping chain.  Longitudinal datasets nevertheless repeat a structural
    variable (time) inside plots/subjects; ACF lags crossing those boundaries
    are spurious.  Prefer explicitly named grouping columns and otherwise the
    categorical column with the most repeated levels.
    """
    best: tuple[tuple[int, int, int], str, np.ndarray] | None = None
    for col in data.columns:
        name = str(col)
        lower = name.lower()
        if lower in _TIME_LABELS or any(token in lower for token in ("time", "date", "dose", "age")):
            continue
        values = data[col]
        try:
            n_unique = int(values.nunique(dropna=False))
            counts = values.value_counts(dropna=False)
        except Exception:
            continue
        if n_unique < 2 or n_unique >= len(values) or len(values) == 0:
            continue
        if int(counts.max()) < 3:
            continue
        hinted = int(any(token in lower for token in _GROUP_HINTS))
        categorical = pd.api.types.is_object_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype)
        if not hinted and not categorical:
            continue
        score = (hinted, n_unique, int(counts.min()))
        if best is None or score > best[0]:
            best = (score, name, values.to_numpy())
    if best is None:
        return None, None
    return best[2], best[1]


def _outer_groups(result: Any) -> tuple[np.ndarray | None, str | None]:
    data = _fit_data(result)
    levels = getattr(result, "_pred_levels", None)
    if data is not None and levels:
        try:
            chain = levels[0].get("chain") or []
        except Exception:
            chain = []
        for col in chain:
            if col in data.columns:
                return data[col].to_numpy(), str(col)
    # GLS and correlation-only fits keep their grouping in the residual
    # correlation specification (for example ``~ 1 | Mare``).
    spec = getattr(result, "cor_spec", None)
    form = getattr(spec, "form", None) if spec is not None else None
    if form and data is not None and "|" in str(form):
        chain = str(form).split("|", 1)[1].strip().strip('"')
        for col in [part.strip() for part in chain.split("/")]:
            if col in data.columns:
                return data[col].to_numpy(), col
    if getattr(result, "kind", None) == "gnls" and data is not None:
        groups, group_name = _inferred_gnls_groups(data)
        if groups is not None:
            return groups, group_name
    index = getattr(result, "index", None)
    if isinstance(index, pd.MultiIndex) and index.nlevels:
        return np.asarray(index.get_level_values(0)), str(index.names[0] or "group")
    return None, None


def _time_values(result: Any, data: pd.DataFrame | None) -> tuple[np.ndarray | None, str | None]:
    name = type(result).__name__
    if name == "SaemFit":
        info = getattr(result, "info", None)
        if info is not None and info.frame is not None and info.time_col in info.frame.columns:
            return np.asarray(info.frame[info.time_col], dtype=float), str(info.time_col)
        raw = getattr(result, "res", None)
        if raw is not None and hasattr(raw, "data") and hasattr(raw.data, "time"):
            return np.asarray(raw.data.time, dtype=float), str(getattr(info, "time_col", "time"))
        return None, None
    if data is not None:
        for col in ("time", "Time", "TIME", "t", "dose", "Dose", "age", "Age"):
            if col in data.columns:
                return np.asarray(data[col], dtype=float), str(col)
    return None, None


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------


def _glm_residuals(result: Any, seed: int) -> tuple[np.ndarray | None, str]:
    attempts = [
        ("resid_quantile", ("Quantile residuals", {"seed": seed})),
        ("resid_sdeviance", ("Standardised deviance residuals", {})),
        ("resid_pearson", ("Pearson residuals", {})),
        ("resid_response", ("Raw residuals", {})),
    ]
    for name, (label, kwargs) in attempts:
        try:
            value = getattr(result, name, None)
            if value is None:
                continue
            value = value(**kwargs) if callable(value) else value
            arr = _array(value)
            if arr is not None and np.isfinite(arr).any():
                return arr, label
        except Exception:
            continue
    return None, "Residuals"


def _mixed_residuals(result: Any) -> tuple[np.ndarray | None, str]:
    name = type(result).__name__
    attempts: list[tuple[str, dict, str]] = []
    if name in ("LmeFit", "NlmeFit"):
        attempts = [
            ("normalized", "Normalised residuals"),
            ("pearson", "Pearson residuals"),
            ("response", "Raw residuals"),
        ]
        for kind, label in attempts:
            try:
                arr = _series(result.residuals(type=kind))
                if arr is not None and np.isfinite(arr).any():
                    return arr, label
            except Exception:
                continue
    elif name == "LmerFit":
        arr = _series(result.residuals())
        sigma = _array(getattr(result, "sigma", None))
        if arr is not None and sigma is not None and sigma.size:
            scale = float(sigma.reshape(-1)[0])
            if np.isfinite(scale) and scale > 0:
                return arr / scale, "Normalised residuals"
        return arr, "Raw residuals"
    elif name == "GlmerFit":
        for kind, label in (("deviance", "Deviance residuals"), ("pearson", "Pearson residuals"), ("response", "Raw residuals")):
            try:
                arr = _series(result.residuals(type=kind))
                if arr is not None and np.isfinite(arr).any():
                    return arr, label
            except Exception:
                continue
    return None, "Residuals"


# Simulation-derived fitted values for non-Gaussian SAEM families.
_SAEM_PRED_NSIM = 200
_SAEM_PRED_SEED = 12345


def _saem_predictions(result: Any, nsim: int = _SAEM_PRED_NSIM):
    """Conditional (IPRED) and population (PRED) means for a SAEM fit.

    ``SaemFit.fitted`` is only implemented for ``family='gaussian'``.  For
    discrete / censored families (Poisson, NB, ZIP, ordinal, TTE, ...) the
    predictive means are obtained by averaging simulations with and without
    fresh individual random effects.  The result is cached on the fit because
    every plotting function calls :func:`resolve` independently.
    """
    family = getattr(getattr(result, "model", None), "family", None)
    raw = getattr(result, "res", None)
    if family == "gaussian":
        return _array(getattr(raw, "fitted_ind", None)), _array(getattr(raw, "fitted_pop", None))

    store = getattr(result, "__dict__", None)
    key = "_plot_simulated_fitted"
    cached = store.get(key) if store is not None else None
    if cached is not None and cached[0] == int(nsim):
        return cached[1], cached[2]

    simulate = getattr(result, "_simulated_y", None)
    if not callable(simulate):
        return None, None
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            conditional = np.nanmean(
                np.asarray(simulate(nsim=nsim, seed=_SAEM_PRED_SEED, new_random_effects=False), dtype=float),
                axis=1,
            )
            population = np.nanmean(
                np.asarray(simulate(nsim=nsim, seed=_SAEM_PRED_SEED, new_random_effects=True), dtype=float),
                axis=1,
            )
    except Exception:
        return None, None
    if store is not None:
        store[key] = (int(nsim), conditional, population)
    return conditional, population


def _saem_residuals(result: Any, seed: int, nsim: int) -> tuple[np.ndarray | None, str]:
    cens = _array(getattr(getattr(getattr(result, "res", None), "data", None), "cens", None))
    censored = cens is not None and bool(np.any(cens != 0))
    if not censored:
        # Pearson residuals are the natural unit-scale check for Gaussian fits;
        # non-Gaussian families fail here and fall through to NPDE below.
        try:
            arr = _series(result.residuals(type="pearson"))
            if arr is not None and np.isfinite(arr).any():
                return arr, "Pearson residuals"
        except Exception:
            pass
    # For censored observations ``res.data.y`` is the censoring limit and
    # Pearson/raw residuals are NaN.  NPDE honours the censoring direction and
    # keeps every observation, so it is the only consistent plot residual.
    # Every panel calls ``resolve`` again; cache the draws like the simulated
    # predictions so the mosaic shows one consistent, single-cost residual set.
    store = getattr(result, "__dict__", None)
    key = "_plot_npde"
    cache_key = (int(nsim), int(seed))
    cached = store.get(key) if store is not None else None
    if cached is not None and cached[0] == cache_key:
        return cached[1], "NPDE"
    try:
        arr = _series(result.npde(nsim=nsim, seed=seed))
        if arr is not None and np.isfinite(arr).any():
            if store is not None:
                store[key] = (cache_key, arr)
            return arr, "NPDE"
    except Exception:
        pass
    try:
        arr = _series(result.residuals(type="raw"))
        if arr is not None and np.isfinite(arr).any():
            return arr, "Raw residuals"
    except Exception:
        pass
    return None, "Residuals"


# ---------------------------------------------------------------------------
# The resolved shape
# ---------------------------------------------------------------------------


@dataclass
class PlotData:
    """The common, model-independent view consumed by the plotting functions."""

    result: Any
    kind: str
    model_name: str
    response_name: str
    response: np.ndarray | None = None
    fitted: np.ndarray | None = None
    fitted_population: np.ndarray | None = None
    residuals: np.ndarray | None = None
    residual_label: str = "Residuals"
    leverage: np.ndarray | None = None
    cooks: np.ndarray | None = None
    linear_predictor: np.ndarray | None = None
    groups: np.ndarray | None = None
    group_name: str | None = None
    time: np.ndarray | None = None
    time_name: str | None = None
    n: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.n:
            for value in (self.response, self.fitted, self.residuals):
                if value is not None:
                    self.n = int(np.asarray(value).size)
                    break


def resolve(result: Any, *, seed: int = 42, npde_nsim: int = 500) -> PlotData:
    """Resolve any supported fitted result to a :class:`PlotData`."""
    if not is_fitted(result):
        raise RuntimeError(
            "The model must be fitted before plotting. Call `.fit()` first."
        )

    name = type(result).__name__
    label = model_name(result)
    yname = response_name(result)

    # ---- GLM ------------------------------------------------------------
    if name == "GLM":
        y = _array(getattr(result, "endog", None))
        fitted = _array(getattr(result, "fittedvalues", None))
        residuals, rlabel = _glm_residuals(result, seed)
        leverage = _array(getattr(result, "hat_matrix_diag", None))
        cooks = _array(_value(result, "cooks_distance", default=None))
        eta = _array(_attr(result, "linear_predictors", None))
        return PlotData(
            result=result,
            kind="glm",
            model_name=label,
            response_name=yname,
            response=y,
            fitted=fitted,
            fitted_population=fitted,
            residuals=residuals,
            residual_label=rlabel,
            leverage=leverage,
            cooks=cooks,
            linear_predictor=eta,
            n=int(y.size) if y is not None else 0,
            extras={
                "family": family_name(result),
                "has_cox_snell": _value(result, "resid_cox_snell", default=None) is not None,
                "is_binary": family_name(result) in ("Binomial", "QuasiBinomial"),
                "is_categorical": family_name(result) in ("Multinomial", "Ordinal"),
            },
        )

    # ---- Nonlinear least squares ----------------------------------------
    if name in ("nls", "NLS"):
        y = _array(getattr(result, "_y", None))
        fitted = _array(getattr(result, "fitted_values", None))
        residuals = _array(getattr(result, "residuals", None))
        return PlotData(
            result=result,
            kind="nls",
            model_name=label,
            response_name=yname,
            response=y,
            fitted=fitted,
            fitted_population=fitted,
            residuals=residuals,
            residual_label="Raw residuals",
            leverage=_array(getattr(result, "leverage", None)),
            cooks=_array(getattr(result, "cooks_distance", None)),
            n=int(y.size) if y is not None else 0,
        )

    # ---- Mixed-effects models -------------------------------------------
    if name in ("LmeFit", "NlmeFit", "LmerFit", "GlmerFit"):
        y = _array(getattr(result, "_y", None))
        if name in ("LmeFit", "NlmeFit"):
            fitted = _series(_value(result, "fitted", default=None))
            fitted_pop = _series(_value(result, "fitted", 0, default=None))
            eta = None
        elif name == "LmerFit":
            fitted = _series(result.fitted("all"))
            fitted_pop = _series(result.fitted(False))
            eta = None
        else:  # GlmerFit
            fitted = _series(result.fitted(type="response", re_form="all"))
            fitted_pop = _series(result.fitted(type="response", re_form=False))
            eta = _series(result.fitted(type="link", re_form="all"))
        residuals, rlabel = _mixed_residuals(result)
        groups, group_name = _outer_groups(result)
        time, time_name = _time_values(result, _fit_data(result))
        return PlotData(
            result=result,
            kind="mixed",
            model_name=label,
            response_name=yname,
            response=y,
            fitted=fitted,
            fitted_population=fitted_pop,
            residuals=residuals,
            residual_label=rlabel,
            linear_predictor=eta,
            groups=groups,
            group_name=group_name,
            time=time,
            time_name=time_name,
            n=int(y.size) if y is not None else 0,
            extras={
                "family": family_name(result),
                "has_random_effects": getattr(result, "ranef", None) is not None,
            },
        )

    # ---- SAEM ------------------------------------------------------------
    if name == "SaemFit":
        raw = getattr(result, "res", None)
        y = _array(getattr(getattr(raw, "data", None), "y", None))
        fitted, fitted_pop = _saem_predictions(result)
        if fitted is None:
            fitted = _series(_value(result, "fitted", "individual", default=None))
        if fitted_pop is None:
            fitted_pop = _series(_value(result, "fitted", "population", default=None))
        if y is not None:
            # Custom likelihoods cannot be simulated; keep the all-NaN shape so
            # prediction panels render an empty slot instead of raising.
            if fitted is None:
                fitted = np.full(y.shape, np.nan)
            if fitted_pop is None:
                fitted_pop = np.full(y.shape, np.nan)
        residuals, rlabel = _saem_residuals(result, seed, npde_nsim)
        groups, group_name = None, None
        info = getattr(result, "info", None)
        if info is not None:
            codes = _array(getattr(getattr(raw, "data", None), "id", None))
            if codes is not None and info.ids is not None:
                try:
                    idx = np.rint(codes).astype(int) - 1
                    valid = np.isfinite(codes) & (idx >= 0) & (idx < len(info.ids))
                    if bool(valid.all()):
                        groups = np.asarray(info.ids)[idx]
                except Exception:
                    groups = None
            if groups is None and info.frame is not None and info.id_col in info.frame.columns:
                groups = np.asarray(info.frame[info.id_col])
            group_name = getattr(info, "id_col", None)
        time, time_name = _time_values(result, None)
        return PlotData(
            result=result,
            kind="saem",
            model_name=label,
            response_name=yname,
            response=y,
            fitted=fitted,
            fitted_population=fitted_pop,
            residuals=residuals,
            residual_label=rlabel,
            groups=groups,
            group_name=group_name,
            time=time,
            time_name=time_name,
            n=int(y.size) if y is not None else 0,
            extras={
                "family": getattr(getattr(result, "model", None), "family", None),
                "has_random_effects": has_random_effects(result),
                "has_trace": bool(getattr(raw, "theta_trace", None)),
            },
        )

    # ---- MCGLM -----------------------------------------------------------
    if name == "MCGLMResults":
        K = int(getattr(result, "K", 1) or 1)
        y = _array(getattr(result, "y", None))
        fitted = _array(getattr(result, "fitted", None))
        residuals = _array(getattr(result, "resid_standardized", None))
        if residuals is None or not np.isfinite(residuals).any():
            residuals = _array(getattr(result, "resid_pearson", None))
        model = getattr(result, "model", None)
        endog = getattr(model, "endog_names", None) if model is not None else None
        if isinstance(endog, (list, tuple)):
            yname = " + ".join(str(v) for v in endog)
        elif endog:
            yname = str(endog)
        return PlotData(
            result=result,
            kind="mcglm",
            model_name=label,
            response_name=yname,
            response=y,
            fitted=fitted,
            fitted_population=fitted,
            residuals=residuals,
            residual_label="Standardised residuals",
            n=int(getattr(result, "nobs", 0) or 0),
            extras={
                "K": K,
                "responses": list(endog) if isinstance(endog, (list, tuple)) else [yname],
                "has_history": hasattr(result, "history"),
            },
        )

    # ---- List fits -------------------------------------------------------
    if name == "ListFit":
        return PlotData(
            result=result,
            kind="list",
            model_name=label,
            response_name=yname,
            n=0,
            extras={"groups": len(result.coef)} if hasattr(result, "coef") else 0,
        )

    # ---- Fallback: duck typing ------------------------------------------
    y = _first([_array(getattr(result, "endog", None)), _array(getattr(result, "_y", None))])
    fitted = _first(
        [
            _array(getattr(result, "fittedvalues", None)),
            _array(getattr(result, "fitted_values", None)),
            _series(_value(result, "fitted", default=None)),
        ]
    )
    residuals = _first(
        [
            _array(getattr(result, "residuals", None)),
            _array(getattr(result, "resid", None)),
        ]
    )
    return PlotData(
        result=result,
        kind="generic",
        model_name=label,
        response_name=yname,
        response=y,
        fitted=fitted,
        fitted_population=fitted,
        residuals=residuals,
        residual_label="Residuals",
        n=int(y.size) if y is not None else 0,
    )


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


def _standardise_frame(table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Map the many coefficient-table layouts to four common columns."""
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=["estimate", "se", "lower", "upper"])
    df = pd.DataFrame(table).copy()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("the coefficient table does not contain enough numeric columns")
    estimate = numeric.iloc[:, 0]
    se_name = next((c for c in numeric.columns if str(c).lower() in ("std err", "std. error", "se")), None)
    se = numeric[se_name] if se_name is not None else numeric.iloc[:, 1]
    lower = numeric.iloc[:, -2]
    upper = numeric.iloc[:, -1]
    p_name = next((c for c in numeric.columns if "p>" in str(c).lower() or str(c).lower() == "pvalue"), None)
    out = pd.DataFrame(
        {
            "estimate": np.asarray(estimate, dtype=float),
            "se": np.asarray(se, dtype=float),
            "lower": np.asarray(lower, dtype=float),
            "upper": np.asarray(upper, dtype=float),
        },
        index=df.index,
    )
    if p_name is not None:
        out["pvalue"] = np.asarray(numeric[p_name], dtype=float)
    out.attrs["alpha"] = alpha
    return out


def _glm_coef_table(result: Any, alpha: float) -> pd.DataFrame:
    try:
        table = result.coef_table(alpha=alpha)
    except TypeError:
        table = result.coef_table()
    return _standardise_frame(pd.DataFrame(table), alpha)


def _nls_coef_table(result: Any, alpha: float) -> pd.DataFrame:
    return _standardise_frame(pd.DataFrame(result.coef_table(alpha=alpha)), alpha)


def _fixed_table(result: Any, alpha: float) -> pd.DataFrame:
    from scipy import stats

    fixef = getattr(result, "fixef", None)
    se = getattr(result, "se", None)
    if fixef is None:
        raise TypeError(f"{type(result).__name__} does not expose fixed effects")
    names = list(fixef.index)
    est = np.asarray(fixef, dtype=float)
    ses = np.asarray(se, dtype=float) if se is not None else np.full(len(names), np.nan)
    fixdf = getattr(result, "_fixdf", None)
    if fixdf is None and type(result).__name__ == "LmerFit":
        try:  # lmerTest-style Satterthwaite df is available but expensive; use it when cheap/cached
            fixdf = getattr(result, "_satt", None)
        except Exception:
            fixdf = None
    if fixdf is not None:
        q = stats.t.ppf(1.0 - alpha / 2.0, np.maximum(np.asarray(fixdf, dtype=float), 1.0))
    else:
        q = np.full(len(names), stats.norm.ppf(1.0 - alpha / 2.0))
    return pd.DataFrame(
        {
            "estimate": est,
            "se": ses,
            "lower": est - q * ses,
            "upper": est + q * ses,
        },
        index=names,
    )


def _mcglm_table(result: Any, alpha: float, response: int) -> pd.DataFrame:
    K = int(getattr(result, "K", 1) or 1)
    k = int(response)
    if not 0 <= k < K:
        raise ValueError(f"response must be in [0, {K - 1}]")
    ix = np.asarray(result._ix_beta[k])
    names = [result.model.exog_names[k][j] for j in range(len(ix))]
    est = np.asarray(result.params, dtype=float)[ix]
    ci = np.asarray(result.conf_int(alpha=alpha), dtype=float)[ix]
    try:
        se = np.asarray(result.bse, dtype=float)[ix]
    except Exception:
        se = np.full(len(ix), np.nan)
    return pd.DataFrame(
        {"estimate": est, "se": se, "lower": ci[:, 0], "upper": ci[:, 1]},
        index=names,
    )


def _saem_table(result: Any, alpha: float) -> pd.DataFrame:
    ci = result.conf_int(alpha=alpha)
    est = np.asarray(ci["estimate"], dtype=float)
    lo = np.asarray(ci["lower"], dtype=float)
    hi = np.asarray(ci["upper"], dtype=float)
    try:
        se = np.asarray(result.res.se_coef, dtype=float)[: len(est)]
    except Exception:
        z = 1.959963984540054
        se = (hi - lo) / (2.0 * z)
    return pd.DataFrame(
        {"estimate": est, "se": se, "lower": lo, "upper": hi},
        index=list(ci.index),
    )


def coefficient_table(
    result: Any,
    *,
    alpha: float = 0.05,
    exponentiate: bool = False,
    response: int = 0,
) -> pd.DataFrame:
    """A uniform coefficient table with ``estimate/se/lower/upper`` columns.

    Parameters
    ----------
    result : fitted result
        Any supported result object.
    alpha : float, default 0.05
        Significance level for the Wald intervals.
    exponentiate : bool, default False
        Exponentiate the estimate and interval bounds (odds/rate ratios).
    response : int, default 0
        Response index for multivariate ``MCGLMResults``.
    """
    name = type(result).__name__
    if name == "GLM":
        table = _glm_coef_table(result, alpha)
    elif name in ("nls", "NLS"):
        table = _nls_coef_table(result, alpha)
    elif name in ("LmeFit", "NlmeFit", "LmerFit", "GlmerFit"):
        table = _fixed_table(result, alpha)
    elif name == "MCGLMResults":
        table = _mcglm_table(result, alpha, response)
    elif name == "SaemFit":
        table = _saem_table(result, alpha)
    elif name == "ListFit":
        raise TypeError(
            "ListFit has one coefficient table per group; use the list-coefficient plot instead."
        )
    else:
        # Duck-typed fallback: prefer a native coefficient table.
        if hasattr(result, "coef_table"):
            table = _standardise_frame(pd.DataFrame(result.coef_table(alpha=alpha)), alpha)
        elif hasattr(result, "fixef"):
            table = _fixed_table(result, alpha)
        else:
            raise TypeError(f"{name} does not expose a coefficient table")
    table = table.copy()
    if exponentiate:
        for column in ("estimate", "lower", "upper"):
            if column in table:
                table[column] = np.exp(np.asarray(table[column], dtype=float))
        table.attrs["exponentiated"] = True
    else:
        table.attrs["exponentiated"] = False
    return table


def has_coefficients(result: Any) -> bool:
    name = type(result).__name__
    if name == "ListFit":
        return False
    return any(hasattr(result, attr) for attr in ("coef_table", "fixef", "params"))


def default_exponentiate(result: Any) -> bool:
    """Whether a forest plot should show exponentials by default."""
    name = type(result).__name__
    if name != "GLM":
        return False
    fam = getattr(result, "family", None)
    link = getattr(fam, "link", None)
    return str(link).lower() in ("log", "logit", "cloglog", "cauchit")


def _has_numeric_ranef_columns(frame: pd.DataFrame) -> bool:
    """A ranef table with only index/label columns has no effect to plot."""
    ignore = {"id", "subject", "group"}
    return any(
        str(col).lower() not in ignore and pd.api.types.is_numeric_dtype(frame[col])
        for col in frame.columns
    )


def random_effects(result: Any) -> dict[str, pd.DataFrame] | None:
    """Normalise the random-effect tables to ``{group_name: DataFrame}``."""
    value = getattr(result, "ranef", None)
    if value is None:
        return None
    if callable(value) and not isinstance(value, pd.DataFrame):
        try:
            value = value()
        except Exception:
            return None
    if isinstance(value, dict):
        out: dict[str, pd.DataFrame] = {}
        for key, frame in value.items():
            if isinstance(frame, pd.DataFrame) and len(frame) and _has_numeric_ranef_columns(frame):
                out[str(key)] = frame
        return out or None
    if isinstance(value, pd.DataFrame) and len(value) and _has_numeric_ranef_columns(value):
        index_name = value.index.name or "group"
        return {str(index_name): value}
    return None


def has_random_effects(result: Any) -> bool:
    return random_effects(result) is not None
