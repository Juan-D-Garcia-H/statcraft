# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Plot registry: the ``plot(result, kind=...)`` entry point and data helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from importlib import import_module

# Import the submodules explicitly: the package re-exports several functions
# whose names collide with their module (for example ``diagnostics``), so
# ``from . import diagnostics`` can resolve to the function instead of the
# module.
_pkg = __package__
classification = import_module(".classification", _pkg)
coefficients = import_module(".coefficients", _pkg)
convergence = import_module(".convergence", _pkg)
diagnostics = import_module(".diagnostics", _pkg)
effects = import_module(".effects", _pkg)
influence = import_module(".influence", _pkg)
predictions = import_module(".predictions", _pkg)
from .adapters import (
    family_name,
    has_coefficients,
    has_random_effects,
    resolve,
)
from .foundation import PlotResult, pyplot

__all__ = [
    "plot",
    "available_kinds",
    "suggest_plots",
    "KIND_ALIASES",
    "CANONICAL_PLOTS",
    "rootogram_data",
    "npde_values",
    "empirical_variogram",
]


KIND_ALIASES: dict[str, tuple[str, Callable[..., PlotResult]]] = {}
CANONICAL_PLOTS: dict[str, Callable[..., PlotResult]] = {}


def _register(names: Iterable[str], function: Callable[..., PlotResult]):
    names = tuple(names)
    canonical = names[0]
    CANONICAL_PLOTS[canonical] = function
    for name in names:
        KIND_ALIASES[name] = (canonical, function)


# ---- diagnostics ----------------------------------------------------------
_register(["qq", "qq_plot", "residual_qq", "normal_qq"], diagnostics.qq_plot)
_register(
    ["qq_plot_with_envelope", "qq_with_envelope", "qq_envelope", "normal_qq_envelope"],
    diagnostics.qq_plot_with_envelope,
)
_register(
    [
        "normalized_residuals_vs_fitted",
        "normalised_residuals_vs_fitted",
        "residuals_vs_conditional",
        "lme_residual_trend",
    ],
    diagnostics.normalized_residuals_vs_fitted,
)
_register(
    [
        "normalized_residuals_vs_time",
        "normalised_residuals_vs_time",
        "nlme_residuals_vs_time",
        "nlme_residual_trend",
    ],
    diagnostics.normalized_residuals_vs_time,
)
_register(["residuals_vs_fitted", "residual_vs_fitted", "resid_vs_fitted", "rvf"], diagnostics.residuals_vs_fitted)
_register(
    ["residuals_vs_predictions", "residual_vs_predictions", "resid_vs_pred",
     "residuals_vs_time", "saem_resid_vs_pred", "npde_vs_time"],
    diagnostics.residuals_vs_predictions,
)
_register(["scale_location", "scale-location", "spread_location"], diagnostics.scale_location)
_register(["hist", "histogram", "residual_histogram"], diagnostics.residual_histogram)
_register(["binned", "binned_residuals", "binned_residual"], diagnostics.binned_residuals)
_register(
    [
        "multiclass_binned_residuals", "multinomial_binned_residuals",
        "binned_residuals_multiclass",
    ],
    diagnostics.multiclass_binned_residuals,
)
_register(
    ["binned_standardized_residuals", "binned_standardised_residuals",
     "binned_standardized_residual", "binned_std_residuals"],
    diagnostics.binned_standardized_residuals,
)
_register(["acf", "residual_acf", "autocorrelation"], diagnostics.acf_plot)
_register(
    ["normalized_residuals_and_spread", "normalised_residuals_and_spread",
     "residuals_and_spread", "gls_residual_spread", "gnls_residual_spread"],
    diagnostics.normalized_residuals_and_spread,
)
_register(
    ["grouped_acf", "grouped_acf_plot", "acf_by_group",
     "gls_residual_acf", "gnls_residual_acf"],
    diagnostics.grouped_acf_plot,
)
_register(["observed_vs_fitted", "obs_vs_fitted", "observed_fitted", "observed_vs_predicted"], diagnostics.observed_vs_fitted)
_register(["rootogram", "hanging_rootogram"], diagnostics.rootogram)
_register(["calibration", "calibration_plot", "reliability"], diagnostics.calibration_plot)
_register(["cox_snell", "cox-snell"], diagnostics.cox_snell)
_register(
    [
        "deviance_martingale_vs_eta", "deviance_martingale_residuals_vs_eta",
        "martingale_vs_eta", "deviance_martingale",
    ],
    diagnostics.deviance_martingale_vs_eta,
)
_register(["truncated", "truncated_fit"], diagnostics.truncated_fit)
_register(
    ["truncated_diagnostics", "truncated_mosaic", "truncated_panel", "truncated_qc"],
    diagnostics.truncated_diagnostics,
)
_register(["hurdle", "hurdle_continuous"], diagnostics.hurdle_continuous)
_register(["secondary", "secondary_distribution"], diagnostics.secondary_distribution)
_register(["diagnostics", "diagnostic", "panel"], diagnostics.diagnostics)
_register(["residuals", "residual_panel", "residual_panels"], diagnostics.residuals_panel)
_register(["pp", "pp_plot", "probability_probability"], diagnostics.pp_plot)
_register(["half_normal", "halfnormal", "half_normal_plot"], diagnostics.half_normal)
_register(["worm", "worm_plot", "detrended_qq"], diagnostics.worm_plot)
_register(["grouped_residuals", "residuals_by_group", "residual_by_group"], diagnostics.grouped_residuals)
_register(["variogram", "semi_variogram", "semivariogram"], diagnostics.variogram_plot)
_register(["pearson", "pearson_residuals"], diagnostics.pearson_residuals)
_register(["qq_residuals", "residual_qq"], diagnostics.qq_plot)
_register(
    ["randomized_quantile_qq", "randomised_quantile_qq", "dharma", "dharma_qq", "dharma_plot", "rq_qq"],
    diagnostics.randomized_quantile_qq,
)
_register(
    ["quantile_residuals_vs_fitted", "quantile_residuals_vs_predictions",
     "quantile_resid_vs_fitted", "quantile_resid_vs_pred",
     "dharma_vs_fitted", "dharma_vs_pred"],
    diagnostics.quantile_residuals_vs_fitted,
)
_register(
    ["binned_quantile_residuals_by_rank", "binned_quantile_residuals",
     "quantile_residuals_by_rank", "glmer_binned_quantile_residuals"],
    diagnostics.binned_quantile_residuals_by_rank,
)
_register(
    ["gls_diagnostics", "diagnose_gls", "gls_panel",
     "gnls_diagnostics", "diagnose_gnls", "gnls_panel",
     "glmer_diagnostics", "diagnose_glmer", "glmer_panel"],
    lambda result, **kwargs: (
        diagnostics.gls_diagnostics(result, **kwargs)
        if type(result).__name__ == "LmeFit"
        else diagnostics.gnls_diagnostics(result, **kwargs)
        if type(result).__name__ == "NlmeFit"
        else diagnostics.glmer_diagnostics(result, **kwargs)
    ),
)
_register(
    ["saem_diagnostics", "diagnose_saem", "saem_panel", "saem_mosaic"],
    diagnostics.saem_diagnostics,
)
_register(
    ["quantile_scale_location", "quantile_scale", "quantile_scale_vs_eta",
     "sqrt_quantile_scale", "qscale"],
    diagnostics.quantile_scale_location,
)
_register(
    [
        "secondary_vs_eta", "secondary_parameter_vs_eta", "phi_vs_eta",
        "sigma2_vs_eta", "dispersion_vs_eta",
    ],
    diagnostics.secondary_vs_eta,
)
_register(
    [
        "standardized_pearson_vs_fitted", "standardised_pearson_vs_fitted",
        "pearson_vs_fitted", "std_pearson_vs_fitted", "pearson_residuals_vs_fitted",
    ],
    diagnostics.standardized_pearson_vs_fitted,
)
_register(
    ["pearson_scale_location", "quasi_scale_location", "dispersion_scale_location"],
    diagnostics.pearson_scale_location,
)
_register(
    ["standardized_residual_qq", "standardised_residual_qq", "residual_qq_by_response", "mcglm_residual_qq"],
    diagnostics.standardized_residual_qq,
)
_register(
    ["standardized_residual_density", "standardised_residual_density", "residual_density_by_response"],
    diagnostics.standardized_residual_density,
)
_register(
    [
        "standardized_residuals_vs_fitted",
        "standardised_residuals_vs_fitted",
        "mcglm_residuals_vs_fitted",
        "residuals_vs_fitted_by_response",
    ],
    diagnostics.standardized_residuals_vs_fitted,
)
_register(
    ["binned_by_response", "binned_residuals_by_response", "binned_residual_by_response"],
    diagnostics.binned_residuals_by_response,
)

# ---- influence ------------------------------------------------------------
_register(["influence", "cook_leverage", "cooks_vs_leverage"], influence.influence_plot)
_register(
    ["residuals_vs_leverage", "resid_vs_leverage", "residual_vs_leverage", "leverage_vs_residuals"],
    influence.residuals_vs_leverage,
)
_register(["cooks", "cooks_distance", "cooks_distance_plot"], influence.cooks_distance_plot)
_register(["dffits", "dffits_plot"], influence.dffits_plot)
_register(["dfbetas", "dfbetas_heatmap"], influence.dfbetas_heatmap)
_register(["leverage", "leverage_plot", "hat_values"], influence.leverage_plot)

# ---- coefficients ---------------------------------------------------------
_register(["forest", "forest_plot", "coef", "coefficients", "fixef"], coefficients.forest_plot)
_register(["cor_fixef", "fixed_effect_correlation", "fixef_correlation"], coefficients.fixed_effect_correlation)
_register(["variance_components", "varcorr", "variance"], coefficients.variance_components)
_register(["covariance", "covariance_parameters", "cov_params"], coefficients.covariance_parameters)
_register(["correlation", "response_correlation", "correlation_matrix"], coefficients.response_correlation)
_register(["list_coefs", "list_coefficients", "list_coef"], coefficients.list_coefficients)

# ---- predictions ----------------------------------------------------------
_register(["population_vs_conditional", "pop_vs_cond", "population"], predictions.population_vs_conditional)
_register(["individuals", "individual_fits", "ipred"], predictions.individual_fits)
_register(["gof", "goodness_of_fit"], predictions.gof)
_register(["npde", "npde_panel"], predictions.npde_panel)
_register(["category_probs", "category_probabilities", "category"], predictions.category_probabilities)
_register(["ordinal", "ordinal_curves"], predictions.ordinal_curves)
_register(["curve", "fitted_curve", "response_curve"], predictions.fitted_curve)

# ---- effects --------------------------------------------------------------
_register(["caterpillar", "ranef_caterpillar", "ranef"], effects.ranef_caterpillar)
_register(["ranef_distribution", "ranef_hist"], effects.ranef_distribution)
_register(["qq_random", "qq_random_effects", "qq_ranef"], effects.qq_random_effects)
_register(["vpc", "visual_predictive_check"], effects.vpc)
_register(["lorenz", "lorenz_curve"], effects.lorenz_curve)
_register(["gengamma", "gengamma_submodels"], effects.gengamma_submodels)

# ---- convergence ----------------------------------------------------------
_register(["convergence", "trace"], convergence.convergence)
_register(["saem_trace"], convergence.saem_trace)
_register(["mcglm_convergence"], convergence.mcglm_convergence)

# ---- classification -------------------------------------------------------
_register(["roc", "roc_curve"], classification.roc_curve)
_register(["pr", "pr_curve", "precision_recall", "precision_recall_curve"], classification.precision_recall_curve)
_register(["confusion", "confusion_matrix", "confusion_matrix_plot"], classification.confusion_matrix_plot)
_register(["classification_calibration"], classification.calibration_curve)
_register(
    [
        "multiclass_calibration", "multiclass_calibration_curve",
        "one_vs_rest_calibration", "multinomial_calibration",
    ],
    classification.multiclass_calibration_curve,
)
_register(["gains", "gains_chart"], classification.gains_chart)
_register(["lift", "lift_chart"], classification.lift_chart)
_register(["threshold_metrics", "threshold_curves"], classification.threshold_metrics)


def _resolve_plot(kind: str):
    # Accept hyphenated plot names as well as the canonical underscore spelling.
    key = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    if key in KIND_ALIASES:
        return KIND_ALIASES[key]
    raise ValueError(f"unknown plot kind {kind!r}; use available_kinds() to list the supported names")


def plot(
    result: Any,
    kind: str | None = None,
    ax=None,
    *,
    show: bool = False,
    save: str | None = None,
    dpi: int | float | None = None,
    close: bool = False,
    block: bool | None = None,
    **kwargs,
) -> PlotResult:
    """Render a plot for any supported fitted result.

    Parameters
    ----------
    result : fitted result
        A fitted GLM, NLS, mixed-effects, SAEM, MCGLM or list-fit object.
    kind : str, optional
        Plot name or alias (see :func:`available_kinds`).  Defaults to the
        adaptive ``diagnostics`` panel.
    ax : matplotlib.axes.Axes or sequence of axes, optional
        Existing axes to draw into.  Multi-panel plots require one axes per
        panel; omit ``ax`` to create a new figure.
    show : bool, default False
        Call ``plt.show()`` after rendering.
    save : str, optional
        Write the figure to this path.
    dpi : int or float, optional
        Resolution used by ``save``.
    close : bool, default False
        Close the figure before returning (useful in batch scripts).
    block : bool, optional
        Passed to ``plt.show`` when ``show=True``.
    **kwargs
        Forwarded to the chosen plot (``alpha``, ``seed``, ``nsim``,
        ``exponentiate``, ``response``, ...).

    Returns
    -------
    PlotResult
        Wrapper with ``fig``, ``axes``, ``show()``, ``save()`` and ``close()``.
    """
    name = kind or "diagnostics"
    alias, function = _resolve_plot(name)
    # Some aliases such as "cox-snell" are display names; pass a normalised one.
    kwargs.pop("kind", None)
    rendered = function(result, ax=ax, **kwargs)

    if save is not None:
        rendered.save(save, dpi=dpi)
    if show:
        rendered.show(block=block)
    if close:
        rendered.close()
    return rendered


def available_kinds(result: Any = None) -> list[str]:
    """Sorted list of accepted ``kind`` names, optionally filtered for ``result``."""
    if result is None:
        return sorted(KIND_ALIASES)
    return suggest_plots(result)


def _finite_values(values) -> bool:
    if values is None:
        return False
    try:
        return bool(np.isfinite(np.asarray(values, dtype=float)).any())
    except (TypeError, ValueError):
        return False


def _saem_can_simulate(result: Any) -> bool:
    """Simulation-based plots are undefined for custom and hierarchical SAEM fits."""
    if not callable(getattr(result, "simulate", None)):
        return False
    if getattr(getattr(result, "model", None), "family", None) == "custom":
        return False
    if getattr(getattr(result, "info", None), "group_labels", None) is not None:
        return False
    return True


def suggest_plots(result: Any) -> list[str]:
    """Plot names that make sense for a particular fitted result."""
    try:
        data = resolve(result)
    except Exception:
        return ["diagnostics"]

    name = type(result).__name__
    kinds = [
        "diagnostics",
        "residuals",
        "qq",
        "pp",
        "half_normal",
        "worm",
        "residuals_vs_fitted",
        "scale_location",
        "residual_histogram",
        "observed_vs_fitted",
    ]
    if not _finite_values(data.residuals):
        residual_kinds = {
            "diagnostics", "residuals", "qq", "pp", "half_normal", "worm",
            "residuals_vs_fitted", "scale_location", "residual_histogram",
        }
        kinds = [kind for kind in kinds if kind not in residual_kinds]
    if has_coefficients(result):
        kinds.append("forest")
        if name not in ("nls", "NLS"):
            kinds.append("cor_fixef")
    if name == "ListFit":
        return ["list_coefficients"]
    if name == "MCGLMResults":
        kinds += [
            "pearson", "convergence", "covariance",
            "standardized_residual_qq", "standardized_residual_density",
            "standardized_residuals_vs_fitted", "binned_by_response",
        ]
        if int(data.extras.get("K", 1) or 1) > 1:
            kinds += ["forest", "correlation"]
        # Fall through to the common de-duplication at the end.

    fam = (family_name(result) or "").lower()
    # Substring tests must not confuse "negative binomial"/"hurdle negative
    # binomial" with a binary classifier or a plain count family.
    is_binary = fam.endswith("binomial") and "neg" not in fam
    if name == "GLM":
        from .diagnostics import (
            _continuous_quantile_panel_available,
            _is_continuous_truncated_glm,
            _secondary_parameter_info,
            has_randomized_quantile_residuals,
            supports_rootogram,
        )

        is_quasi = fam.startswith("quasi")
        if supports_rootogram(result) and not is_quasi:
            kinds.append("rootogram")
            if has_randomized_quantile_residuals(result):
                kinds += ["randomized_quantile_qq", "quantile_residuals_vs_fitted"]
        elif (
            not is_binary
            and not is_quasi
            and "multinomial" not in fam
            and "ordinal" not in fam
            and _continuous_quantile_panel_available(result, data, 42)
        ):
            kinds += [
                "quantile_scale_location",
                "randomized_quantile_qq",
                "quantile_residuals_vs_fitted",
            ]
        # No ACF: a GLM assumes independent observations.  The influence
        # panel below now uses residuals vs leverage with Cook contours.
        if is_quasi:
            kinds += ["standardized_pearson_vs_fitted", "pearson_scale_location"]
        if is_binary:
            kinds += ["calibration", "roc", "pr", "confusion", "gains", "lift", "threshold_metrics"]
            if not is_quasi and has_randomized_quantile_residuals(result):
                kinds += ["randomized_quantile_qq", "quantile_residuals_vs_fitted"]
        if any(word in fam for word in ("beta", "simplex")) or "hurdle" in fam:
            kinds.append("secondary")
            if any(word in fam for word in ("beta", "simplex")) and _secondary_parameter_info(result)[2]:
                kinds.append("secondary_vs_eta")
        if "generalizedgamma" in fam:
            kinds.append("gengamma")
        if any(word in fam for word in ("weibull", "lognormal", "loglogistic", "tweedie", "gamma", "pareto")):
            kinds.append("lorenz")
        if getattr(result, "resid_cox_snell", None) is not None:
            kinds.append("cox_snell")
        if getattr(result, "resid_deviance_martingale", None) is not None:
            kinds.append("deviance_martingale_vs_eta")
        if "truncated" in fam:
            kinds.append("truncated")
            if _is_continuous_truncated_glm(result):
                kinds.append("truncated_diagnostics")
        if "multinomial" in fam or "ordinal" in fam:
            kinds.append("category_probs")
        if "multinomial" in fam:
            # The multinomial quartet uses cell residuals Y - P and a
            # one-vs-rest reliability curve instead of a deviance-vs-μ panel.
            kinds += ["multiclass_binned_residuals", "multiclass_calibration"]
        if "ordinal" in fam:
            kinds.append("ordinal")
        if hasattr(result, "hat_matrix_diag"):
            kinds += ["influence", "residuals_vs_leverage", "cooks", "dffits", "leverage"]
    elif name in ("nls", "NLS"):
        if len(getattr(result, "_exogenous", None) or []) == 1:
            kinds.append("curve")
        kinds += ["influence", "cooks", "leverage"]
    elif name in ("LmeFit", "NlmeFit", "LmerFit", "GlmerFit"):
        discrete_glmer = name == "GlmerFit" and str(
            getattr(result, "family", "")
        ).lower() in ("binomial", "poisson", "negbin")
        kinds += [] if discrete_glmer else ["qq_residuals"]
        kinds += ["gof"]
        fit_kind = getattr(result, "kind", None)
        if name == "LmeFit" and fit_kind == "gls":
            kinds += ["gls_diagnostics", "observed_vs_fitted", "qq_plot_with_envelope",
                      "normalized_residuals_and_spread", "grouped_acf"]
        elif name == "NlmeFit" and fit_kind == "gnls":
            kinds += ["gnls_diagnostics", "observed_vs_fitted", "qq_plot_with_envelope",
                      "normalized_residuals_and_spread", "grouped_acf"]
        elif name == "GlmerFit":
            kinds += ["glmer_diagnostics", "population_vs_conditional"]
            if str(getattr(result, "family", "")).lower() in ("binomial", "poisson", "negbin"):
                kinds += ["randomized_quantile_qq", "binned_quantile_residuals_by_rank"]
            else:
                kinds.append("binned_standardized_residuals")
        else:
            kinds += ["population_vs_conditional", "binned_standardized_residuals"]
            if name == "LmeFit":
                kinds += ["normalized_residuals_vs_fitted", "qq_plot_with_envelope"]
            elif name == "NlmeFit":
                kinds += ["qq_plot_with_envelope", "normalized_residuals_vs_time"]
        if data.groups is not None:
            kinds.append("individuals")
        if callable(getattr(result, "simulate", None)):
            kinds.append("vpc")
        if getattr(result, "cor_spec", None) is not None or getattr(result, "correlation", None) is not None:
            kinds.append("variogram")
        if has_random_effects(result):
            kinds += ["caterpillar", "ranef_distribution", "qq_random"]
        if hasattr(result, "varcorr"):
            kinds.append("variance_components")
        if is_binary:
            kinds += ["roc", "pr", "calibration"]
    elif name == "SaemFit":
        kinds = ["diagnostics", "saem_diagnostics", "trace", "forest"]
        try:
            if _finite_values(result.vcov()):
                kinds.append("cor_fixef")
        except Exception:
            pass  # se_method='none' leaves no covariance to correlate
        if _finite_values(data.residuals):
            kinds += ["residuals_vs_predictions", "qq_plot_with_envelope", "grouped_acf"]
        if has_random_effects(result):
            kinds += ["caterpillar", "ranef_distribution", "qq_random"]
        if _finite_values(data.fitted):
            kinds += ["gof", "population_vs_conditional"]
            if data.groups is not None:
                kinds.append("individuals")
        if _saem_can_simulate(result):
            kinds += ["npde", "vpc"]
    if has_random_effects(result) and "caterpillar" not in kinds:
        kinds += ["caterpillar", "ranef_distribution"]
    if hasattr(result, "history") and "convergence" not in kinds:
        kinds.append("convergence")
    # Preserve order, drop duplicates.
    seen = set()
    out = []
    for kind in kinds:
        if kind not in seen and kind in KIND_ALIASES:
            seen.add(kind)
            out.append(kind)
    return out


# ---------------------------------------------------------------------------
# Data helpers shared by the plots and by user code
# ---------------------------------------------------------------------------


def rootogram_data(result: Any, max_count: int | None = None) -> pd.DataFrame:
    """Observed and expected count frequencies used by the hanging rootogram.

    Returns a DataFrame with ``count``, ``observed`` and ``expected`` columns.
    """
    data = resolve(result)
    from .diagnostics import _expected_count_frequencies

    y = np.asarray(data.response, dtype=float).reshape(-1)
    y = np.rint(y).astype(int)
    upper = int(y.max()) if max_count is None else int(max_count)
    counts = np.arange(0, upper + 1)
    observed = np.array([int(np.sum(y == k)) for k in counts], dtype=float)
    expected = _expected_count_frequencies(data.result, counts)
    return pd.DataFrame({"count": counts, "observed": observed, "expected": expected})


def npde_values(result: Any, *, nsim: int = 500, seed: int = 42) -> np.ndarray:
    """Normalised prediction distribution errors as a plain array."""
    npde = getattr(result, "npde", None)
    if not callable(npde):
        raise ValueError(f"{type(result).__name__} does not expose NPDE")
    try:
        values = npde(nsim=nsim, seed=seed)
    except TypeError:
        values = npde()
    if hasattr(values, "columns") and "npde" in values.columns:
        values = values["npde"]
    return np.asarray(values, dtype=float).reshape(-1)


def empirical_variogram(result: Any, *, n_lags: int = 20, max_lag: float | None = None) -> pd.DataFrame:
    """Empirical semivariogram of the residuals over time.

    Pairs of observations within the same group are formed at every time lag;
    the squared residual differences are averaged within ``n_lags`` lag bins.
    Results are returned as a DataFrame with ``lag``, ``semivariance`` and
    ``n_pairs`` columns.
    """
    data = resolve(result)
    residuals = data.residuals
    if residuals is None:
        raise ValueError(f"residuals are not available for {data.model_name}")
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    time = data.time
    if time is None:
        time = np.arange(residuals.size, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    groups = data.groups
    if groups is None:
        groups = np.zeros(residuals.size, dtype=int)
    groups = np.asarray(groups)

    lags, semis = [], []
    for gid in np.unique(groups):
        mask = (groups == gid) & np.isfinite(residuals) & np.isfinite(time)
        if mask.sum() < 2:
            continue
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(time[idx])]
        t = time[order]
        r = residuals[order]
        dt = np.abs(t[:, None] - t[None, :])
        dr2 = (r[:, None] - r[None, :]) ** 2
        upper = np.triu_indices(t.size, k=1)
        lags.append(dt[upper])
        semis.append(dr2[upper])
    if not lags:
        raise ValueError("not enough paired observations to build a variogram")
    lag_values = np.concatenate(lags)
    semi_values = np.concatenate(semis)
    keep = np.isfinite(lag_values) & np.isfinite(semi_values)
    lag_values, semi_values = lag_values[keep], semi_values[keep]
    if lag_values.size == 0:
        raise ValueError("not enough paired observations to build a variogram")
    if max_lag is None:
        max_lag = float(np.quantile(lag_values, 0.95))
    keep = lag_values <= max_lag
    lag_values, semi_values = lag_values[keep], semi_values[keep]
    if lag_values.size == 0:
        raise ValueError("no pairs fall within max_lag")

    edges = np.linspace(0.0, max_lag, n_lags + 1)
    rows = []
    for b in range(n_lags):
        mask = (lag_values >= edges[b]) & (lag_values < edges[b + 1])
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "lag": float(np.mean(lag_values[mask])),
                "semivariance": float(0.5 * np.mean(semi_values[mask])),
                "n_pairs": int(mask.sum()),
            }
        )
    if not rows:
        raise ValueError("no lag bin contained any pairs")
    return pd.DataFrame(rows)


def _noop():  # pragma: no cover - keeps linters from stripping lazy import
    pyplot()
