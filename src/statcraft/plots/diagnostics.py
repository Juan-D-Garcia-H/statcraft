# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Residual and goodness-of-fit diagnostics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from .adapters import PlotData, family_name, has_random_effects, resolve
from .foundation import (
    DEFAULT_COLORS,
    PlotResult,
    annotate_points,
    clean,
    finite,
    panel_axes,
    qq_points,
    single_axis,
    style_axis,
)
from .predictions import _multiclass_arrays

__all__ = [
    "qq_plot",
    "qq_plot_with_envelope",
    "normalized_residuals_vs_fitted",
    "normalized_residuals_vs_time",
    "normalized_residuals_and_spread",
    "grouped_acf_plot",
    "binned_quantile_residuals_by_rank",
    "gls_diagnostics",
    "gnls_diagnostics",
    "glmer_diagnostics",
    "saem_diagnostics",
    "residuals_vs_fitted",
    "residuals_vs_predictions",
    "scale_location",
    "residual_histogram",
    "binned_residuals",
    "multiclass_binned_residuals",
    "binned_standardized_residuals",
    "acf_plot",
    "observed_vs_fitted",
    "rootogram",
    "calibration_plot",
    "cox_snell",
    "deviance_martingale_vs_eta",
    "truncated_fit",
    "truncated_diagnostics",
    "hurdle_continuous",
    "secondary_distribution",
    "diagnostics",
    "residuals_panel",
    "pp_plot",
    "half_normal",
    "worm_plot",
    "grouped_residuals",
    "variogram_plot",
    "pearson_residuals",
    "randomized_quantile_qq",
    "dharma_plot",
    "quantile_residuals_vs_fitted",
    "quantile_scale_location",
    "secondary_vs_eta",
    "standardized_pearson_vs_fitted",
    "pearson_scale_location",
    "standardized_residual_qq",
    "standardized_residual_density",
    "standardized_residuals_vs_fitted",
    "binned_residuals_by_response",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _need(values, what: str, data: PlotData):
    if values is None:
        raise ValueError(f"{what} are not available for {data.model_name}.")
    return np.asarray(values, dtype=float).reshape(-1)


def _linear_limits(values: Any, *, pad: float = 0.08, include_zero: bool = False):
    """Padded linear limits from one variable's own finite values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    half_pad = span * pad if span > 0 else max(abs(hi) * pad, 1e-6)
    if include_zero and lo >= 0.0:
        # Nonnegative, heavily skewed responses should start at zero; padding
        # below zero only wastes plotting area and shifts the visual centre.
        return 0.0, hi + half_pad
    if include_zero and hi <= 0.0:
        return lo - half_pad, 0.0
    lower, upper = lo - half_pad, hi + half_pad
    if upper <= lower:
        lower, upper = lower - 0.5, upper + 0.5
    return lower, upper


def _padded_limits(values: Any, *, lower_pad: float = 0.05, upper_pad: float = 0.03):
    """Padded linear limits from one variable's own finite values.

    Positive ranges use multiplicative padding (``0.95 × min`` to ``1.03 ×
    max``), so a panel whose data live far from zero is not squashed by a zero
    baseline.  Ranges crossing zero use span-based padding instead.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    if lo > 0.0:
        lower, upper = lo * (1.0 - lower_pad), hi * (1.0 + upper_pad)
    else:
        span = hi - lo
        padding = span if span > 0 else max(abs(hi), 1.0)
        lower, upper = lo - lower_pad * padding, hi + upper_pad * padding
    if upper <= lower:
        lower, upper = lower - 0.5, upper + 0.5
    return lower, upper


def _lag_ticks(ax, lags):
    """Use exact integer ticks for short ACFs and an integer locator otherwise."""
    lags = np.asarray(lags)
    if lags.size <= 10:
        ax.set_xticks(lags)
        return
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _overlap_identity(ax, x: Any, y: Any, *, label: str = "Identity"):
    """Draw the identity line only where the two plotted variables overlap.

    Extending it to the union of both ranges lets an extreme response stretch
    the prediction axis far beyond the fitted values, squashing the cloud to
    one side.  The overlap segment is still a valid reference: inside it the
    equality ``y = ŷ`` is meaningful for both variables.
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        return
    lo = max(float(xs.min()), float(ys.min()))
    hi = min(float(xs.max()), float(ys.max()))
    if lo < hi:
        ax.plot(
            [lo, hi], [lo, hi], color=DEFAULT_COLORS["accent"],
            linestyle="--", linewidth=1.5, label=label,
        )


def _symmetric_log_scale(ax, x: Any, y: Any):
    """Use matching symlog axes for heavily skewed nonnegative data.

    A handful of very large responses otherwise occupies almost the whole
    linear Y range while the bulk of the points and the identity line collapse
    near zero.  Applying the same symmetric-log transform to both axes keeps
    every observation visible, including structural zeros, and leaves ``y = x``
    a straight diagonal.  The linear threshold is data-driven but never above
    one, so small bounded responses still get a readable scale.
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    positive = np.concatenate((xs[xs > 0], ys[ys > 0]))
    positive = positive[np.isfinite(positive)]
    linthresh = (
        float(np.clip(np.percentile(positive, 5.0), 1e-6, 1.0))
        if positive.size else 1.0
    )
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    x_limits = _linear_limits(xs, include_zero=True)
    y_limits = _linear_limits(ys, include_zero=True)
    if x_limits is not None:
        ax.set_xlim(0.0, x_limits[1])
    if y_limits is not None:
        ax.set_ylim(0.0, y_limits[1])


def _moving_average(x, y, *, fraction: float = 0.12, min_window: int = 5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_window + 1:
        return np.empty(0), np.empty(0)
    order = np.argsort(x)
    x, y = x[order], y[order]
    window = max(int(x.size * fraction), min_window)
    window = min(window, x.size)
    if window < 2:
        return np.empty(0), np.empty(0)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid"), np.convolve(y, kernel, mode="valid")


def _trend_line(ax, x, y, *, color=DEFAULT_COLORS["accent"], label="Trend"):
    xs, ys = _moving_average(x, y)
    if xs.size:
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label)


def _loess_curve(x, y, *, span: float = 0.12, n_grid: int = 100):
    """Local-linear smoother with pointwise 95% bands.

    Weighted local linear regressions evaluated on a grid follow the data
    across the full range without extrapolating past the first or last ``x``.
    The band uses the local fit's leverage and a homoskedastic residual
    variance estimated from the same smoother, which is enough for a visual
    calibration check without fitting a full generalized additive model.

    Returns ``None`` when fewer than five finite pairs are available, otherwise
    ``(grid, fitted, lower, upper)``.
    """
    x, y = clean(x, y)
    if x.size < 5:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    total = float(x[-1] - x[0])
    if total <= 0:
        return None
    bandwidth = max(total * span, 1e-12)
    grid = np.linspace(x[0], x[-1], int(n_grid))
    fitted = np.empty_like(grid)
    leverage = np.empty_like(grid)
    for i, x0 in enumerate(grid):
        dx = x - x0
        w = np.exp(-0.5 * (dx / bandwidth) ** 2)
        s0 = float(w.sum())
        if s0 <= 1e-12:
            fitted[i], leverage[i] = np.nan, np.nan
            continue
        s1 = float(np.sum(w * dx))
        s2 = float(np.sum(w * dx * dx))
        t0 = float(np.sum(w * y))
        t1 = float(np.sum(w * dx * y))
        det = s0 * s2 - s1 * s1
        if abs(det) <= 1e-12 * max(s0 * s2, 1.0):
            fitted[i], leverage[i] = t0 / s0, 1.0 / s0
        else:
            fitted[i] = (t0 * s2 - t1 * s1) / det
            leverage[i] = s2 / det
    good = np.isfinite(fitted) & np.isfinite(leverage)
    if not good.any():
        return None
    at_observations = np.interp(x, grid[good], fitted[good])
    residual_var = float(np.sum((y - at_observations) ** 2) / max(x.size - 2, 1))
    se = np.sqrt(np.maximum(residual_var * leverage, 0.0))
    lower = fitted - 1.96 * se
    upper = fitted + 1.96 * se
    return grid, fitted, lower, upper


def _loess_line(ax, x, y, *, color=DEFAULT_COLORS["accent"], label="Trend", span: float = 0.12):
    """Local-linear trend spanning the full range of ``x``.

    A centred moving average drops its edge windows, so with right-skewed
    predictions its line stops well before the last observations.  Weighted
    local linear regressions evaluated on a grid follow the data across the
    whole range without extrapolating past the first or last ``x``.
    """
    curve = _loess_curve(x, y, span=span)
    if curve is None:
        return
    grid, fitted, _, _ = curve
    good = np.isfinite(fitted)
    if good.any():
        ax.plot(grid[good], fitted[good], color=color, linewidth=2.0, label=label)


def _normal_density(probs):
    """Standard-normal density, used for quantile standard errors."""
    from scipy import stats

    return stats.norm.pdf(stats.norm.ppf(probs))


def _finish(fig, ax, title: str | None):
    if title:
        ax.set_title(title)
    return PlotResult(fig, title or "")


_RESPONSE_COLORS = ("#1f4e79", "#b45309", "#0f766e", "#7c3aed", "#be123c", "#15803d")


def _safe_attr(obj: Any, name: str, default=None):
    """Read a property, tolerating families where it raises."""
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _is_quasi_family(result) -> bool:
    """Whether the GLM family specifies only a mean-variance relationship."""
    return type(getattr(result, "family", None)).__name__.lower().startswith("quasi")


def _pearson_data(result, seed: int):
    """Fitted means, (standardised) Pearson residuals, label and dispersion.

    Quasi-likelihood fits have no CDF, and their preferred residuals in
    :func:`resolve` are deviance residuals.  dispersion checks therefore need
    the Pearson residuals explicitly, divided by ``phi (1 - h)`` when the
    hat matrix is available so that a well-fitting model has unit variance.
    """
    data = resolve(result, seed=seed)
    mu = _need(data.fitted, "Fitted values", data)

    res = _safe_attr(result, "resid_sspearson")
    label = "Standardised Pearson residuals"
    if res is None:
        res = _safe_attr(result, "resid_pearson")
        label = "Pearson residuals"
    if res is None and data.residuals is not None and "pearson" in data.residual_label.lower():
        res, label = data.residuals, data.residual_label
    if res is None:
        raise ValueError(f"Pearson residuals are not available for {data.model_name}.")
    res = np.asarray(res, dtype=float).reshape(-1)

    phi = None
    for attr in ("dispersion", "scale"):
        value = _safe_attr(result, attr)
        if value is None:
            continue
        try:
            value = float(np.asarray(value, dtype=float).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(value) and value > 0:
            phi = value
            break
    if phi is None:
        value = _safe_attr(result, "_phi")
        if callable(value):
            try:
                candidate = float(value())
                if np.isfinite(candidate) and candidate > 0:
                    phi = candidate
            except Exception:
                pass
    return data, mu, res, label, phi


def _by_response(result, seed: int):
    """Split a multivariate fit into ``(names, fitted, residuals)`` per response."""
    data = resolve(result, seed=seed)
    K = int(getattr(result, "K", 1) or 1)
    n = int(getattr(result, "nobs", 0) or 0)
    if K <= 1 or n <= 0 or data.fitted is None or data.residuals is None:
        return None
    try:
        fitted = np.asarray(data.fitted, dtype=float).reshape(K, n)
        residuals = np.asarray(data.residuals, dtype=float).reshape(K, n)
    except (TypeError, ValueError):
        return None
    endog = getattr(getattr(result, "model", None), "endog_names", None)
    names = [str(endog[k]) if k < len(endog) else f"Response {k + 1}" for k in range(K)]
    return names, fitted, residuals


_STANDARDIZED_WORDS = ("standard", "normalis", "normaliz", "pearson", "quantile", "npde", "deviance")


def _is_standardized(label: str) -> bool:
    lower = str(label).lower()
    return any(word in lower for word in _STANDARDIZED_WORDS)


def _standardized_residual_data(result, data: PlotData, seed: int):
    """Preferred unit-scale residuals for a fitted result."""
    label = data.residual_label
    res = data.residuals
    if res is not None and _is_standardized(label) and type(result).__name__ != "GlmerFit":
        return np.asarray(res, dtype=float).reshape(-1), label

    name = type(result).__name__
    if name in ("GlmerFit", "LmeFit", "NlmeFit"):
        kinds = (("pearson", "Pearson residuals"), ("deviance", "Deviance residuals")) if name == "GlmerFit" else (
            ("normalized", "Normalised residuals"), ("pearson", "Pearson residuals")
        )
        for kind, new_label in kinds:
            try:
                arr = np.asarray(result.residuals(type=kind), dtype=float).reshape(-1)
                if arr.size == data.n and np.isfinite(arr).any():
                    return arr, new_label
            except Exception:
                continue
    elif name == "GLM":
        for attr, new_label, kwargs in (
            ("resid_sdeviance", "Standardised deviance residuals", {}),
            ("resid_pearson", "Pearson residuals", {}),
            ("resid_quantile", "Quantile residuals", {"seed": seed}),
        ):
            try:
                value = getattr(result, attr, None)
                value = value(**kwargs) if callable(value) else value
                arr = np.asarray(value, dtype=float).reshape(-1)
                if arr.size == data.n and np.isfinite(arr).any():
                    return arr, new_label
            except Exception:
                continue
    if res is not None:
        return np.asarray(res, dtype=float).reshape(-1), label
    raise ValueError(f"standardised residuals are not available for {data.model_name}.")


def _quantile_bins(x, n_bins: int):
    """Stable rank-based quantile bin indices for ``x``."""
    size = int(np.asarray(x).size)
    bins = max(1, min(int(n_bins), size))
    ranks = np.empty(size, dtype=int)
    ranks[np.argsort(x, kind="mergesort")] = np.arange(size)
    return np.minimum((ranks * bins) // max(size, 1), bins - 1), bins


# ---------------------------------------------------------------------------
# Core residual diagnostics
# ---------------------------------------------------------------------------


def qq_plot(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    note: str | None = None,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Normal Q-Q plot of the model's preferred residuals.

    ``note`` is drawn in the upper-left corner; it is used by the Beta/simplex
    panel to show a fixed precision/dispersion without adding another panel.
    """
    data = resolve(result, seed=seed)
    values = _need(data.residuals, "Residuals", data)
    fig, ax, _ = single_axis(ax, figsize)
    osm, osr = qq_points(values)
    if osm.size:
        ax.scatter(osm, osr, alpha=0.62, color=DEFAULT_COLORS["band"], edgecolors="none", label="Residuals")
        slope, intercept = np.polyfit(osm, osr, 1)
        ax.plot(osm, slope * osm + intercept, color=DEFAULT_COLORS["accent"], linewidth=1.6, linestyle="--", label="Normal reference")
    if note:
        ax.text(
            0.03, 0.94, note, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78),
        )
    ax.set_xlabel("Theoretical N(0, 1) quantiles")
    ax.set_ylabel(f"Ordered {data.residual_label.lower()}")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Q-Q plot — {data.residual_label}")


def qq_plot_with_envelope(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_sim: int = 400,
    point_label: str = "Normalised residuals",
    value_label: str = "Ordered normalised residuals",
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Normal Q-Q of the preferred residuals with a 95% simulation envelope.

    The envelope is obtained by sorting ``n_sim`` standard-normal samples of
    the same length as the residual vector.  Unlike a single fitted reference
    line it accounts for the sampling variability of the order statistics,
    which avoids flagging ordinary edge fluctuations as outliers.  This is the
    Q-Q used in the truncated and mixed-effects diagnostic mosaics.  SAEM fits
    pass ``point_label='NPDE'`` when their residuals are simulation-based.
    """
    data = resolve(result, seed=seed)
    values = _need(data.residuals, "Residuals", data)
    fig, ax, _ = single_axis(ax, figsize)
    _qq_envelope_panel(ax, values, seed=seed, n_sim=n_sim, model_name=data.model_name,
                       point_label=point_label, value_label=value_label)
    if title:
        ax.set_title(title)
    return PlotResult(fig, title or f"Q-Q with 95% envelope — {data.model_name}")


def _residual_trend_panel(ax, x, residuals, *, xlabel: str, model_name: str, label: str):
    """Scatter of unit-scale residuals with a LOESS mean and 95% band."""
    x, residuals = clean(x, residuals)
    if x.size < 5:
        raise ValueError("not enough finite residuals for the residual-trend panel")
    fig, ax, _ = single_axis(ax)
    ax.scatter(x, residuals, s=17, alpha=0.55, color="#0e7490", edgecolors="none", label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    curve = _loess_curve(x, residuals)
    if curve is not None:
        grid, fitted, lower, upper = curve
        good = np.isfinite(fitted) & np.isfinite(lower) & np.isfinite(upper)
        if good.any():
            ax.fill_between(grid[good], lower[good], upper[good], color="#0e7490",
                            alpha=0.16, linewidth=0.0, label="95% LOESS band")
            ax.plot(grid[good], fitted[good], color="#0e7490", linewidth=2.2,
                    label="LOESS trend")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Standardised residual")
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    return _finish(fig, ax, f"Residual trend vs. {xlabel} — {model_name}")


def normalized_residuals_vs_fitted(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Normalised residuals against conditional fitted values (LME check).

    The smooth trend tests linearity on the mean scale; the spread of the
    points around it tests homoscedasticity.  Both are visible in one panel
    without the small-sample noise of one box per subject.
    """
    data = resolve(result, seed=seed)
    residuals = _need(data.residuals, "Residuals", data)
    fitted = _need(data.fitted, "Conditional fitted values", data)
    xlabel = "Conditional fitted value"
    fig, ax, _ = single_axis(ax, figsize)
    _residual_trend_panel(ax, fitted, residuals, xlabel=xlabel,
                          model_name=data.model_name, label=data.residual_label)
    if title:
        ax.set_title(title)
    return PlotResult(fig, title or f"Residual trend vs. conditional fit — {data.model_name}")


def normalized_residuals_vs_time(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Normalised residuals against the primary structural variable (NLME).

    In a nonlinear model, residuals sharing the same fitted value can come
    from opposite phases of the curve and cancel in a fitted-value plot.
    Plotting against time/dose/concentration keeps those phases separate.
    """
    data = resolve(result, seed=seed)
    residuals = _need(data.residuals, "Residuals", data)
    x = data.time
    if x is None:
        x = np.arange(residuals.size, dtype=float)
        xlabel = "Observation order"
    else:
        xlabel = str(data.time_name or "Time")
    fig, ax, _ = single_axis(ax, figsize)
    _residual_trend_panel(ax, x, residuals, xlabel=xlabel,
                          model_name=data.model_name, label=data.residual_label)
    if title:
        ax.set_title(title)
    return PlotResult(fig, title or f"Residual trend vs. {xlabel} — {data.model_name}")


def normalized_residuals_and_spread(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    use_time: bool = False,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Normalised residuals and their spread against the fitted mean.

    The left scale carries the unit-variance residuals with a LOESS mean
    curve; the right scale carries ``√|r|`` with its own smooth trend and 95%
    band.  A correctly specified variance function (``varPower``, ``varIdent``,
    ...) leaves both curves flat.  Axis limits are derived from the data, so a
    fan shape is never clipped to a fixed ``[-2, 2]`` window.  ``use_time=True``
    plots against the structural variable instead of the fitted value.
    """
    data = resolve(result, seed=seed)
    residuals = _need(data.residuals, "Residuals", data)
    x, xlabel = data.fitted, "Fitted value"
    if use_time and data.time is not None:
        x, xlabel = data.time, str(data.time_name or "Time")
    x = _need(x, xlabel, data)
    fig, ax, _ = single_axis(ax, figsize)
    _residual_spread_panel(
        ax, x, residuals, xlabel=xlabel, model_name=data.model_name,
        residual_label="Normalised residual", point_label="Normalised residuals",
        root_label="√|normalised residual|",
    )
    if title:
        ax.set_title(title)
    return PlotResult(fig, title or f"Normalised residuals & spread vs. {xlabel} — {data.model_name}")


def _acf_values(residuals: np.ndarray, groups: np.ndarray | None, order: np.ndarray | None,
                max_lag: int | None) -> tuple[np.ndarray, int]:
    """Pooled within-group ACF, falling back to a single-series ACF."""
    res = np.asarray(residuals, dtype=float).reshape(-1)
    if groups is None or len(groups) != res.size:
        centered = res - res.mean()
        denom = float(np.dot(centered, centered))
        if denom <= 0:
            raise ValueError("residuals are constant; the ACF is undefined")
        limit = min(res.size - 1, max_lag if max_lag is not None else min(40, res.size // 2))
        acf = np.array([1.0] + [float(np.dot(centered[:-k], centered[k:]) / denom)
                                for k in range(1, limit + 1)])
        return acf, res.size

    blocks: list[np.ndarray] = []
    for level in np.unique(groups):
        mask = np.asarray(groups) == level
        block = res[mask]
        if order is not None and order.size == res.size:
            block = block[np.argsort(order[mask])]
        if block.size >= 3:
            blocks.append(block - block.mean())
    if not blocks:
        raise ValueError("no group has enough residuals for an ACF")
    longest = max(block.size for block in blocks)
    default_lag = min(40, longest - 1)
    limit = max(1, min(longest - 1, default_lag if max_lag is None else int(max_lag)))
    acf = np.ones(limit + 1, dtype=float)
    for k in range(1, limit + 1):
        numerator = 0.0
        denominator = 0.0
        for block in blocks:
            if block.size <= k:
                continue
            numerator += float(np.dot(block[:-k], block[k:]))
            denominator += float(np.dot(block, block))
        acf[k] = numerator / denominator if denominator > 0 else np.nan
    return acf, res.size


def grouped_acf_plot(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    max_lag: int | None = None,
    figsize=(6.0, 4.0),
    title: str | None = None,
):
    """ACF of normalised residuals, computed within groups when available.

    Correlated GLS/GNLS fits carry a grouping factor in ``corStruct``; an
    unstructured GNLS fit infers the repeated-measures factor from its data
    frame (for example ``Plot`` or ``Subject``).  Pooling across those
    boundaries would create spurious autocorrelation at every lag, so the lag
    products are accumulated inside each group and the largest lag is capped at
    ``max(n_i) - 1``.  The band is the white-noise reference ``±1.96 / √N``
    using all finite residuals.
    """
    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    finite_mask = np.isfinite(res)
    res = res[finite_mask]
    groups = None if data.groups is None else np.asarray(data.groups)[finite_mask]
    order = None if data.time is None else np.asarray(data.time)[finite_mask]
    if res.size < 3:
        raise ValueError("at least three finite residuals are needed for an ACF plot")
    acf, n = _acf_values(res, groups, order, max_lag)
    if not np.isfinite(acf).any():
        raise ValueError("the residual ACF is undefined")

    fig, ax, _ = single_axis(ax, figsize)
    lags = np.arange(acf.size)
    markerline, stemlines, baseline = ax.stem(lags, acf, basefmt=" ")
    try:
        markerline.set_color(DEFAULT_COLORS["band"])
        stemlines.set_color(DEFAULT_COLORS["band"])
        stemlines.set_linewidth(1.5)
    except Exception:
        pass
    band = 1.96 / np.sqrt(n)
    label = "95% white-noise band"
    if groups is not None:
        label = "95% white-noise band (within group)"
    ax.axhspan(-band, band, color=DEFAULT_COLORS["neutral"], alpha=0.18, label=label)
    ax.axhline(0.0, color="black", linewidth=1.0)
    _lag_ticks(ax, lags)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Residual ACF — {data.model_name}")


def _quantile_residual_values(result: Any, data: PlotData, seed: int, n_sim: int | None = None):
    """Randomised quantile residuals on the normal-score scale, or ``None``."""
    rq = getattr(result, "resid_quantile", None)
    if callable(rq):
        try:
            raw = rq(seed=seed, n_sim=n_sim) if n_sim is not None else rq(seed=seed)
        except TypeError:
            raw = rq()
        if raw is not None:
            return np.asarray(raw, dtype=float).reshape(-1)
    if data.residuals is not None and "quantile" in data.residual_label.lower():
        return np.asarray(data.residuals, dtype=float).reshape(-1)
    return None


def randomized_quantile_qq(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_sim: int | None = None,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """DHARMa-style Q-Q of randomised quantile residuals on the uniform scale.

    Discrete responses have gaps in their distribution function, so raw
    residuals form rigid horizontal bands.  Randomised (Dunn-Smyth) quantile
    residuals replace each gap by a uniform draw inside ``(F(y-), F(y))``;
    under a correct model the transformed values are Uniform(0, 1).  This
    plot compares their order statistics with the theoretical uniform
    quantiles and draws a 95% simultaneous Kolmogorov-Smirnov envelope.

    Parameters
    ----------
    result : fitted result
        Needs ``resid_quantile`` (all count GLM families provide it).
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axis.
    seed : int, default 42
        Seed for the randomisation inside discrete distribution gaps.
    """
    data = resolve(result, seed=seed)
    values = _quantile_residual_values(result, data, seed, n_sim=n_sim)
    if values is None:
        raise ValueError(
            f"randomised quantile residuals are not available for {data.model_name}"
        )

    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("not enough finite quantile residuals to build a Q-Q plot")

    from scipy.stats import kstest, norm

    uniform = np.clip(norm.cdf(values), 0.0, 1.0)
    ordered = np.sort(uniform)
    n = ordered.size
    theoretical = (np.arange(n) + 0.5) / n
    band = 1.36 / np.sqrt(n)  # ~95% simultaneous KS band

    fig, ax, _ = single_axis(ax, figsize)
    ax.fill_between(
        theoretical,
        np.clip(theoretical - band, 0.0, 1.0),
        np.clip(theoretical + band, 0.0, 1.0),
        color=DEFAULT_COLORS["grid"],
        alpha=0.75,
        linewidth=0.0,
        label="95% KS envelope",
    )
    ax.scatter(theoretical, ordered, s=17, alpha=0.7, color="#4338ca", edgecolors="none", label="Randomised residuals")
    ax.plot([0.0, 1.0], [0.0, 1.0], color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.6, label="Uniform reference")
    try:
        pvalue = float(kstest(uniform, "uniform").pvalue)
        ax.text(0.04, 0.92, f"KS p = {pvalue:.3f}", transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    except Exception:
        pass
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Theoretical Uniform(0, 1) quantiles")
    ax.set_ylabel("Randomised quantile residuals")
    ax.legend(loc="lower right", fontsize=8)
    style_axis(ax)
    return _finish(fig, ax, title or f"DHARMa-style Q-Q — {data.model_name}")


# Common short alias used by DHARMa users.
dharma_plot = randomized_quantile_qq


def quantile_residuals_vs_fitted(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    use_linear_predictor: bool = False,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Randomised quantile residuals against the predicted value.

    The DHARMa-style Q-Q checks the *marginal* distribution of the residuals;
    this panel checks whether the same residuals drift with the prediction,
    which exposes a misspecified mean structure or variance function.  Values
    are on the normal-score scale returned by ``resid_quantile``, so the
    reference lines sit at 0 and ±2.  A local-linear trend covers the full
    prediction range, including sparse right-tail predictions.

    Parameters
    ----------
    result : fitted GLM
        Needs ``resid_quantile`` (all exact count families provide it).
    seed : int, default 42
        Randomisation seed inside discrete distribution gaps.  Use the same
        seed as the Q-Q panel so both show the same residuals.
    use_linear_predictor : bool, default False
        Plot against ``η`` instead of the fitted mean ``μ``.
    """
    data = resolve(result, seed=seed)
    res = _quantile_residual_values(result, data, seed)
    if res is None:
        raise ValueError(f"quantile residuals are not available for {data.model_name}")
    eta = data.linear_predictor if use_linear_predictor else None
    x = eta if (use_linear_predictor and eta is not None) else data.fitted
    x = _need(x, "Fitted values", data)
    x, res = clean(x, res)
    if x.size < 3:
        raise ValueError("not enough finite quantile residuals for this plot")

    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(x, res, s=17, alpha=0.6, color="#4338ca", edgecolors="none", label="Quantile residuals")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    _loess_line(ax, x, res)
    xlabel = "Linear predictor (η)" if use_linear_predictor else "Fitted mean (μ)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Quantile residuals (normal scores)")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Quantile residuals vs. {'η' if use_linear_predictor else 'μ'} — {data.model_name}")


def binned_quantile_residuals_by_rank(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_sim: int | None = None,
    bins: int = 20,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Mean quantile residuals in equally spaced bins of ``rank(fitted)``.

    Raw fitted values pile up near zero for binomial and count responses, so
    equal-width bins in ``ŷ`` leave most of the panel empty.  Mapping the
    fitted values to their empirical quantiles in ``[0, 1]`` spreads the bins
    evenly and makes systematic drift in the middle of the prediction range
    visible.  Error bars are ``±SE`` of the bin mean.
    """
    from scipy.stats import rankdata

    data = resolve(result, seed=seed)
    residuals = _quantile_residual_values(result, data, seed, n_sim=n_sim)
    if residuals is None:
        raise ValueError(f"quantile residuals are not available for {data.model_name}")
    fitted = _need(data.fitted, "Fitted values", data)
    fitted, residuals = clean(fitted, residuals)
    if fitted.size < 5:
        raise ValueError("not enough finite quantile residuals for binned residuals")
    ranks = rankdata(fitted, method="average") / fitted.size
    bins = int(np.clip(bins, 4, 20))
    bins = min(bins, max(4, fitted.size // 5))
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(ranks, edges[1:-1], right=False), 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan)
    ses = np.full(bins, np.nan)
    for b in range(bins):
        values = residuals[which == b]
        if values.size == 0:
            continue
        means[b] = float(np.mean(values))
        ses[b] = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0

    fig, ax, _ = single_axis(ax, figsize)
    keep = np.isfinite(means) & np.isfinite(ses)
    if keep.any():
        ax.errorbar(centers[keep], means[keep], yerr=ses[keep], fmt="o-", color="#4338ca",
                    ecolor="#4338ca", capsize=3, linewidth=1.6, label="Mean quantile residual")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fitted-value quantile")
    ax.set_ylabel("Quantile residual (normal score)")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Binned quantile residuals vs. rank(fitted) — {data.model_name}")


def quantile_scale_location(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    use_linear_predictor: bool = True,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Scale-location of quantile residuals against the linear predictor.

    Taking ``√|r|`` tames the skew of the absolute residuals, so a change in
    spread shows up as a change in the *level* of the smooth trend instead of
    as a widening fan.  Under a correct model the quantile residuals are
    N(0, 1), for which ``E[√|Z|] = √(2/π) ≈ 0.80``; a flat trend at that level
    supports the fitted variance function.  The natural GLM scale is ``η``;
    with no linear predictor the plot falls back to the fitted mean ``μ``.

    Parameters
    ----------
    result : fitted GLM
        Needs ``resid_quantile`` (all exact-distribution continuous families
        provide it).
    seed : int, default 42
        Use the same seed as the Q-Q panel so both refer to the same residuals.
    use_linear_predictor : bool, default True
        Plot against ``η`` instead of ``μ``.
    """
    data = resolve(result, seed=seed)
    res = _quantile_residual_values(result, data, seed)
    if res is None:
        raise ValueError(f"quantile residuals are not available for {data.model_name}")
    eta = data.linear_predictor if use_linear_predictor else None
    x = eta if eta is not None else data.fitted
    x = _need(x, "Fitted values", data)
    x, res = clean(x, res)
    if x.size < 3:
        raise ValueError("not enough finite quantile residuals for this plot")

    root = np.sqrt(np.abs(res))
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(x, root, s=17, alpha=0.55, color="#6366f1", edgecolors="none", label="√|Quantile residuals|")
    _loess_line(ax, x, root)
    expected = float(np.sqrt(2.0 / np.pi))
    ax.axhline(
        expected, color=DEFAULT_COLORS["neutral"], linestyle="--", linewidth=1.2,
        label=f"N(0,1) expectation ({expected:.2f})",
    )
    by_eta = eta is not None
    ax.set_xlabel("Linear predictor (η)" if by_eta else "Fitted mean (μ)")
    ax.set_ylabel("√|Quantile residuals|")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(
        fig, ax,
        title or f"Scale-location (quantile) vs. {'η' if by_eta else 'μ'} — {data.model_name}",
    )


def secondary_vs_eta(
    result: Any,
    ax=None,
    *,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Secondary distribution parameter (Beta ``φ`` / simplex ``σ²``) vs ``η``.

    This is the fourth panel for Beta/simplex fits whose precision/dispersion is
    itself modelled through covariates.  A flat trend means the covariate model
    adds nothing over a constant parameter; a trend or fan shows where the
    dispersion model is doing the work.
    """
    data = resolve(result)
    eta = _need(data.linear_predictor, "Linear predictor", data)
    try:
        values = np.asarray(result.predict(which="secondary"), dtype=float).reshape(-1)
    except Exception as exc:  # pragma: no cover - depends on the result class
        raise ValueError(
            f"secondary-parameter predictions are not available for {data.model_name}"
        ) from exc
    eta, values = clean(eta, values)
    if eta.size < 3:
        raise ValueError("not enough finite secondary-parameter values for this plot")

    name = family_name(result) or ""
    if name == "Beta":
        ylabel = "φ (precision)"
    elif name == "Simplex":
        ylabel = "σ² (dispersion)"
    else:
        ylabel = "Secondary parameter"

    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(eta, values, s=18, alpha=0.58, color="#0f766e", edgecolors="none", label=ylabel)
    _loess_line(ax, eta, values, label="Trend")
    ax.set_xlabel("Linear predictor (η)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"{ylabel} vs. η — {data.model_name}")


def residuals_vs_fitted(
    result: Any,
    ax=None,
    *,
    use_linear_predictor: bool | None = None,
    seed: int = 42,
    figsize=(5.5, 4.5),
    trend: bool = True,
    title: str | None = None,
):
    """Residuals against the linear predictor or the fitted mean.

    Standardised residuals get the usual ``±2`` reference lines; a local-linear
    trend spanning the whole predictor range makes curvature and
    heteroscedasticity easier to see.
    """
    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    if use_linear_predictor is None:
        use_linear_predictor = data.linear_predictor is not None
    x = data.linear_predictor if (use_linear_predictor and data.linear_predictor is not None) else data.fitted
    x = _need(x, "Fitted values", data)
    x, res = clean(x, res)
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(x, res, alpha=0.52, color=DEFAULT_COLORS["residual"], edgecolors="none")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    if any(word in data.residual_label.lower() for word in ("quantile", "pearson", "deviance", "standard", "normalised", "normalized", "npde")):
        ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
        ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    if trend:
        _loess_line(ax, x, res)
    xlabel = "Linear predictor (η)" if use_linear_predictor else "Fitted mean (μ)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(data.residual_label)
    ax.legend(loc="best", fontsize=8.5) if trend else None
    style_axis(ax)
    return _finish(fig, ax, title or f"{data.residual_label} vs. {'η' if use_linear_predictor else 'μ'} — {data.model_name}")


def residuals_vs_predictions(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    x: str | None = None,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Residuals against conditional predictions, or against time when unavailable.

    This is the homoscedasticity / trend check for SAEM fits.  Gaussian models
    provide Pearson residuals and conditional predictions (IPRED), so the
    x-axis is the fitted value.  Discrete/censored SAEM families (Poisson, NB,
    ZIP, ordinal, TTE, custom) have no residual-error parameters; their
    residuals are NPDE, plotted against observation time by default.  Passing
    ``x='time'`` or ``x='fitted'`` overrides the automatic choice (the fitted
    values of discrete fits are simulation-derived predictive means).
    """
    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    label = data.residual_label
    is_npde = any(word in label.lower() for word in ("npde", "quantile"))

    if x is None:
        if is_npde and _has_finite(getattr(data, "time", None)):
            x = "time"
        elif _has_finite(data.fitted):
            x = "fitted"
        elif _has_finite(getattr(data, "time", None)):
            x = "time"
        else:
            x = "fitted"
    axis = str(x).strip().lower()
    if axis in ("fitted", "fit", "pred", "prediction", "conditional", "ipred"):
        values = _need(data.fitted, "Conditional predictions", data)
        xlabel = "Conditional prediction (IPRED)"
    elif axis in ("time", "t"):
        values = getattr(data, "time", None)
        if values is None:
            values = np.arange(res.size, dtype=float)
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != res.size:
            values = np.arange(res.size, dtype=float)
        xlabel = getattr(data, "time_name", None) or "Time"
    else:
        raise ValueError("x must be 'fitted' or 'time'")

    xv, rv = clean(values, res)
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(xv, rv, s=18, alpha=0.55, color=DEFAULT_COLORS["residual"], edgecolors="none", label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    # NPDE are normal scores: the reference limits are the 2.5%/97.5% normal
    # quantiles, not the ±2 heuristic used for unit-scale Pearson residuals.
    band = 1.96 if is_npde else 2.0
    ax.axhline(band, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0, label=f"±{band:g}")
    ax.axhline(-band, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    _trend_line(ax, xv, rv, label="Trend")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    which = "time" if axis in ("time", "t") else "fitted"
    return _finish(fig, ax, title or f"{'NPDE' if is_npde else 'Residuals'} vs. {which} — {data.model_name}")


def scale_location(result: Any, ax=None, *, seed: int = 42, figsize=(5.5, 4.5), title: str | None = None):
    """Scale-location plot: :math:`\\sqrt{|r_i|}` against the fitted mean."""
    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    mu = _need(data.fitted, "Fitted values", data)
    mu, res = clean(mu, res)
    fig, ax, _ = single_axis(ax, figsize)
    root = np.sqrt(np.abs(res))
    ax.scatter(mu, root, alpha=0.5, color="#6366f1", edgecolors="none")
    _trend_line(ax, mu, root)
    expected = np.sqrt(2.0 / np.pi)
    ax.axhline(expected, color=DEFAULT_COLORS["neutral"], linestyle="--", linewidth=1.1, label=f"N(0,1) expectation ({expected:.2f})")
    ax.set_xlabel("Fitted mean (μ)")
    ax.set_ylabel(f"√|{data.residual_label.lower()}|")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Scale-location — {data.model_name}")


def standardized_pearson_vs_fitted(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    use_linear_predictor: bool = False,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Standardised Pearson residuals against ``η`` (or ``μ``).

    This is the main check for quasi-likelihood families: under
    ``Var(Y) = φ μ`` the scatter has constant spread across the whole range of
    the prediction.  A fanning-out pattern (or a trend in the local mean of
    ``|r|``) points to a variance function that grows with the mean, i.e. a
    negative-binomial or Tweedie model may be needed instead.
    """
    data, mu, res, label, phi = _pearson_data(result, seed)
    x = data.linear_predictor if use_linear_predictor and data.linear_predictor is not None else mu
    x, res = clean(x, res)
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(x, res, s=17, alpha=0.55, color=DEFAULT_COLORS["residual"], edgecolors="none", label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    _loess_line(ax, x, res, label="Trend")
    if phi is not None:
        detail = f"φ = {phi:.3g}"
    else:
        detail = "φ fixed at 1"
    ax.text(0.03, 0.94, detail, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    by_eta = use_linear_predictor and data.linear_predictor is not None
    ax.set_xlabel("Linear predictor (η)" if by_eta else "Fitted mean (μ)")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"{label} vs. {'η' if by_eta else 'μ'} — {data.model_name}")


def pearson_scale_location(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    use_linear_predictor: bool = False,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Scale-location check based on Pearson residuals: ``√|r|`` vs ``η``/``μ``.

    The smooth trend makes the behaviour of the estimated dispersion visible:
    a roughly flat line at ``E[√|Z|] = √(2/π)`` (for unit-variance standardised
    residuals) supports a constant ``φ``; a rising trend means the variance
    depends on the prediction.  The local-linear trend covers the whole range
    instead of stopping short where observations are sparse.
    """
    data, mu, res, label, phi = _pearson_data(result, seed)
    x = data.linear_predictor if use_linear_predictor and data.linear_predictor is not None else mu
    x, res = clean(x, res)
    root = np.sqrt(np.abs(res))
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(x, root, s=17, alpha=0.5, color="#6366f1", edgecolors="none", label=f"√|{label.lower()}|")
    _loess_line(ax, x, root, label="Trend √|r|")
    expected = np.sqrt(2.0 / np.pi)
    ax.axhline(expected, color=DEFAULT_COLORS["neutral"], linestyle="--", linewidth=1.1,
               label=f"Expected √|Z| ({expected:.2f})")
    if phi is not None:
        ax.text(0.03, 0.94, f"φ = {phi:.3g}", transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    by_eta = use_linear_predictor and data.linear_predictor is not None
    ax.set_xlabel("Linear predictor (η)" if by_eta else "Fitted mean (μ)")
    ax.set_ylabel(f"√|{label.lower()}|")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Pearson scale-location vs. {'η' if by_eta else 'μ'} — {data.model_name}")


def residual_histogram(result: Any, ax=None, *, seed: int = 42, bins: int = 30, figsize=(5.5, 4.0), title: str | None = None):
    """Histogram of residuals with a matched normal density."""
    from scipy import stats

    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    res = res[np.isfinite(res)]
    fig, ax, _ = single_axis(ax, figsize)
    ax.hist(res, bins=bins, density=True, color="#93c5fd", edgecolor="#1e3a8a", alpha=0.78)
    if res.size > 2 and np.std(res) > 0:
        grid = np.linspace(res.min(), res.max(), 200)
        ax.plot(grid, stats.norm.pdf(grid, res.mean(), res.std()), color=DEFAULT_COLORS["accent"], linewidth=1.8, label="Normal fit")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(data.residual_label)
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Residual distribution — {data.model_name}")


def binned_residuals(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_bins: int = 10,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Binned residual plot (Gelman & Hill).

    Observations are split into quantile bins of the fitted mean.  Each point is
    the mean residual in a bin, with an approximate ``±2`` standard-error band;
    this is especially useful for binary and count models, where individual
    residuals are inherently coarse.
    """
    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    fitted = _need(data.fitted, "Fitted values", data)
    fitted, res = clean(fitted, res)
    if fitted.size < n_bins:
        n_bins = max(2, fitted.size)
    edges = np.quantile(fitted, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(fitted, edges[1:-1], right=True), 0, len(edges) - 2)
    xs, means, ses, counts = [], [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        xs.append(float(np.mean(fitted[mask])))
        means.append(float(np.mean(res[mask])))
        ses.append(float(np.std(res[mask], ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0.0)
        counts.append(int(mask.sum()))
    fig, ax, _ = single_axis(ax, figsize)
    xs = np.asarray(xs)
    means = np.asarray(means)
    ses = np.asarray(ses)
    ax.errorbar(xs, means, yerr=2.0 * ses, fmt="o-", color=DEFAULT_COLORS["band"], ecolor=DEFAULT_COLORS["neutral"], capsize=3, linewidth=1.6)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Fitted mean (bin averages)")
    ax.set_ylabel("Mean residual")
    for x, y, n in zip(xs, means, counts):
        ax.annotate(
            str(n), (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color=DEFAULT_COLORS["neutral"],
            annotation_clip=False,
        )
    # Count labels sit above their points in *points*, so leave data-space
    # headroom or the top row of labels is clipped by the axes boundary.
    if means.size:
        hi = float(np.max(means + 2.0 * ses))
        lo = float(np.min(means - 2.0 * ses))
        pad = max(0.15 * (hi - lo), 0.08)
        ax.set_ylim(lo - pad, hi + pad)
    style_axis(ax)
    return _finish(fig, ax, title or f"Binned residuals — {data.model_name}")


def multiclass_binned_residuals(
    result: Any,
    ax=None,
    *,
    n_bins: int = 10,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Binned residuals over every ``(observation, category)`` cell.

    For a categorical GLM with predicted probabilities ``p_ik`` and one-hot
    response ``y_ik``, the cell residual ``y_ik - p_ik`` has conditional mean
    zero.  Using all ``N × K`` cells (rather than only the observed category's
    deviance residual) therefore centres the binned curve on zero and shows
    miscalibration separately in the low- and high-probability bins.  Each
    point is a bin mean with an approximate ``±2`` standard-error bar.
    """
    probs, one_hot, _, data = _multiclass_arrays(result)
    if type(getattr(data.result, "family", None)).__name__ != "Multinomial":
        raise ValueError(f"multiclass binned residuals need a Multinomial fit (got {data.model_name})")

    cell_p = probs.reshape(-1)
    cell_res = (one_hot - probs).reshape(-1)
    keep = np.isfinite(cell_p) & np.isfinite(cell_res)
    cell_p, cell_res = cell_p[keep], cell_res[keep]
    if cell_p.size < 2:
        raise ValueError("at least two finite category cells are needed for binned residuals")
    n_bins = max(2, min(int(n_bins), cell_p.size))
    edges = np.unique(np.quantile(cell_p, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("fitted category probabilities do not vary enough to form bins")
    idx = np.clip(np.digitize(cell_p, edges[1:-1], right=True), 0, edges.size - 2)

    xs, means, ses, counts = [], [], [], []
    for b in range(edges.size - 1):
        mask = idx == b
        if not mask.any():
            continue
        count = int(mask.sum())
        xs.append(float(np.mean(cell_p[mask])))
        means.append(float(np.mean(cell_res[mask])))
        ses.append(float(np.std(cell_res[mask], ddof=1) / np.sqrt(count)) if count > 1 else 0.0)
        counts.append(count)

    xs = np.asarray(xs, dtype=float)
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    fig, ax, _ = single_axis(ax, figsize)
    errorbar = ax.errorbar(
        xs, means, yerr=2.0 * ses, fmt="o-", color=DEFAULT_COLORS["band"],
        ecolor=DEFAULT_COLORS["neutral"], capsize=3, linewidth=1.6,
    )
    # Matplotlib stores the container's label separately from its data line;
    # expose it on the line so the plotted series is inspectable by label.
    errorbar[0].set_label("Bin mean ±2 SE")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    for x, y, n in zip(xs, means, counts):
        ann = ax.annotate(
            str(n), (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7.5, color=DEFAULT_COLORS["neutral"],
            annotation_clip=False,
        )
        ann.set_clip_on(False)
    if means.size:
        lo = float(np.min(means - 2.0 * ses))
        hi = float(np.max(means + 2.0 * ses))
        pad = max(0.20 * (hi - lo), 0.12)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Fitted category probability (bin averages)")
    ax.set_ylabel("Mean cell residual y − p̂")
    style_axis(ax)
    return _finish(fig, ax, title or f"Multiclass binned residuals — {data.model_name}")


def binned_standardized_residuals(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_bins: int = 10,
    x: str = "fitted",
    ylim: tuple[float, float] | None = (-2.0, 2.0),
    show_points: bool = False,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Binned unit-scale residuals against fitted values, time or index.

    Unlike :func:`binned_residuals`, this averages standardised (or Pearson)
    residuals inside each bin, so the vertical scale stays comparable across
    models and responses.  It replaces one boxplot per grouping level, which
    becomes unreadable with many subjects.  Each point is a bin mean with an
    approximate ``±2`` standard-error bar; the default ``ylim=(-2, 2)`` keeps
    every panel on the same visual scale.
    """
    data = resolve(result, seed=seed)
    res, label = _standardized_residual_data(result, data, seed)
    axis = str(x).strip().lower()
    if axis in ("fitted", "fit", "mu", "pred", "prediction"):
        values = _need(data.fitted, "Fitted values", data)
        xlabel = "Fitted mean (bin averages)"
    elif axis in ("time", "t"):
        values = getattr(data, "time", None)
        if values is None or np.asarray(values).size != res.size:
            values = np.arange(res.size, dtype=float)
        values = np.asarray(values, dtype=float).reshape(-1)
        xlabel = data.time_name or "Time"
    elif axis in ("index", "obs", "observation"):
        values = np.arange(res.size, dtype=float)
        xlabel = "Observation"
    else:
        raise ValueError("x must be 'fitted', 'time' or 'index'")

    values, res = clean(values, res)
    if values.size < 2:
        raise ValueError("at least two finite residuals are needed for binned residuals")
    idx, bins = _quantile_bins(values, n_bins)
    xs, means, ses, counts = [], [], [], []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        xs.append(float(np.median(values[mask])))
        means.append(float(np.mean(res[mask])))
        ses.append(float(np.std(res[mask], ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0.0)
        counts.append(int(mask.sum()))

    fig, ax, _ = single_axis(ax, figsize)
    if show_points:
        ax.scatter(values, res, s=9, alpha=0.16, color=DEFAULT_COLORS["residual"],
                   edgecolors="none", zorder=1)
    xs = np.asarray(xs)
    means = np.asarray(means)
    ses = np.asarray(ses)
    ax.errorbar(xs, means, yerr=2.0 * ses, fmt="o-", color=DEFAULT_COLORS["band"],
                ecolor=DEFAULT_COLORS["neutral"], capsize=3, linewidth=1.6, zorder=3)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=0.9)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=0.9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for x_value, y_value, count in zip(xs, means, counts):
        ax.annotate(str(count), (x_value, y_value), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=DEFAULT_COLORS["neutral"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Mean {label.lower()}")
    style_axis(ax)
    return _finish(fig, ax, title or f"Binned standardized residuals — {data.model_name}")


def standardized_residual_qq(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    density: bool = False,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Distribution of the standardised residuals, one series per response.

    For multivariate fits (e.g. MCGLM) the responses live on very different
    scales, so pooling them can hide a poor fit in the small-scale response.
    Each response is shown separately on the same standardised axis: a normal
    Q-Q by default, or a kernel density with the N(0, 1) reference when
    ``density=True``.
    """
    from scipy.stats import norm

    data = resolve(result, seed=seed)
    split = _by_response(result, seed)
    if split is None:
        series = [(data.response_name, np.asarray(data.fitted, dtype=float).reshape(-1),
                   _need(data.residuals, "Residuals", data))]
    else:
        names, fitted, residuals = split
        series = [(names[k], fitted[k], residuals[k]) for k in range(len(names))]

    fig, ax, _ = single_axis(ax, figsize)
    drawn = 0
    for k, (name, _, res) in enumerate(series):
        res = np.asarray(res, dtype=float).reshape(-1)
        res = res[np.isfinite(res)]
        if res.size < 3:
            continue
        color = _RESPONSE_COLORS[k % len(_RESPONSE_COLORS)]
        if density:
            from scipy.stats import gaussian_kde

            grid = np.linspace(res.min() - 0.5, res.max() + 0.5, 200)
            ax.plot(grid, gaussian_kde(res)(grid), color=color, linewidth=1.8,
                    label=f"{name} (n={res.size})")
        else:
            theoretical, ordered = qq_points(res)
            ax.scatter(theoretical, ordered, s=14, alpha=0.65, color=color,
                       edgecolors="none", label=f"{name} (n={res.size})")
        drawn += 1
    if drawn == 0:
        raise ValueError("no finite standardised residuals are available")

    if density:
        grid = np.linspace(-4.0, 4.0, 200)
        ax.plot(grid, norm.pdf(grid), color=DEFAULT_COLORS["neutral"], linestyle="--",
                linewidth=1.6, label="N(0, 1)")
        ax.set_xlabel("Standardised residual")
        ax.set_ylabel("Density")
        default_title = f"Standardised residual density — {data.model_name}"
    else:
        ax.plot([-4.0, 4.0], [-4.0, 4.0], color=DEFAULT_COLORS["neutral"], linestyle="--",
                linewidth=1.6, label="N(0, 1) reference")
        ax.set_xlabel("Theoretical N(0, 1) quantiles")
        ax.set_ylabel("Ordered standardised residual")
        default_title = f"Standardised residual Q-Q — {data.model_name}"
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    return _finish(fig, ax, title or default_title)


def standardized_residual_density(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Kernel density of standardised residuals, one curve per response."""
    return standardized_residual_qq(result, ax=ax, seed=seed, density=True,
                                    figsize=figsize, title=title)


def standardized_residuals_vs_fitted(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Standardised residuals against within-response fitted quantiles.

    Multivariate fits have responses on very different scales, so a shared
    ``μ`` axis lets one large-scale response squash the others.  Mapping each
    response's fitted values to their own quantile scale puts every series on
    the same horizontal frame, while the per-response colours match the Q-Q
    and binned-residual panels.
    """
    from scipy.stats import rankdata

    data = resolve(result, seed=seed)
    split = _by_response(result, seed)
    fig, ax, _ = single_axis(ax, figsize)

    if split is None:
        res = _standardized_residual_data(result, data, seed)[0]
        mu = _need(data.fitted, "Fitted values", data)
        mu, res = clean(mu, res)
        if mu.size < 3:
            raise ValueError("not enough finite residuals for this plot")
        quantile = rankdata(mu, method="average") / mu.size
        ax.scatter(quantile, res, s=16, alpha=0.58,
                   color=DEFAULT_COLORS["residual"], edgecolors="none",
                   label=f"{data.response_name} (n={mu.size})")
        default_title = f"{data.residual_label} vs. fitted quantile — {data.model_name}"
    else:
        names, fitted, residuals = split
        drawn = 0
        for k, name in enumerate(names):
            mu, res = clean(fitted[k], residuals[k])
            if mu.size < 3:
                continue
            color = _RESPONSE_COLORS[k % len(_RESPONSE_COLORS)]
            quantile = rankdata(mu, method="average") / mu.size
            ax.scatter(quantile, res, s=14, alpha=0.58, color=color,
                       edgecolors="none", label=f"{name} (n={mu.size})")
            drawn += 1
        if drawn == 0:
            raise ValueError("no finite standardised residuals are available")
        default_title = f"Standardised residuals vs. μ — {data.model_name}"

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fitted-value quantile (by response)")
    ax.set_ylabel("Standardised residual")
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    return _finish(fig, ax, title or default_title)


def binned_residuals_by_response(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_bins: int = 10,
    figsize=(6.0, 4.5),
    title: str | None = None,
):
    """Binned standardised residuals for every response, on a common scale.

    Each response is split into the same number of fitted-value quantile bins,
    so bin ``k`` means the same thing for all responses.  Averaging
    standardised residuals keeps the vertical axis unit-free and comparable;
    the shared ``(-2, 2)`` window aligns every series in the same frame.
    """
    data = resolve(result, seed=seed)
    split = _by_response(result, seed)
    if split is None:
        return binned_standardized_residuals(result, ax=ax, seed=seed, n_bins=n_bins,
                                             figsize=figsize, title=title)

    names, fitted, residuals = split
    pairs = []
    for k, name in enumerate(names):
        mu, res = clean(fitted[k], residuals[k])
        if mu.size >= 4:
            pairs.append((name, mu, res))
    if not pairs:
        raise ValueError("not enough observations to compute binned residuals")

    bins = min(n_bins, max(2, min(mu.size for _, mu, _ in pairs) // 5))
    bin_centers = (np.arange(bins) + 0.5) / bins
    fig, ax, _ = single_axis(ax, figsize)
    for k, (name, mu, res) in enumerate(pairs):
        idx, _ = _quantile_bins(mu, bins)
        means, ses = [], []
        for b in range(bins):
            mask = idx == b
            if not mask.any():
                means.append(np.nan)
                ses.append(0.0)
                continue
            means.append(float(np.mean(res[mask])))
            ses.append(float(np.std(res[mask], ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0.0)
        color = _RESPONSE_COLORS[k % len(_RESPONSE_COLORS)]
        ax.errorbar(bin_centers, means, yerr=2.0 * np.asarray(ses), fmt="o-", color=color,
                    ecolor=color, capsize=2.5, linewidth=1.5, alpha=0.9, label=name)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=0.9)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("Fitted-value quantile bin")
    ax.set_ylabel("Mean standardised residual")
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    return _finish(fig, ax, title or f"Binned standardised residuals by response — {data.model_name}")


def acf_plot(result: Any, ax=None, *, seed: int = 42, max_lag: int | None = None, figsize=(6.0, 4.0), title: str | None = None):
    """Autocorrelation function of residuals (ordered as supplied).

    Useful for time-ordered or grouped data; the dashed band is the usual
    ``±1.96 / √n`` white-noise reference.
    """
    data = resolve(result, seed=seed)
    res = _need(data.residuals, "Residuals", data)
    res = res[np.isfinite(res)]
    n = res.size
    if n < 3:
        raise ValueError("at least three finite residuals are needed for an ACF plot")
    centered = res - res.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0:
        raise ValueError("residuals are constant; the ACF is undefined")
    limit = min(n - 1, max_lag if max_lag is not None else min(40, n // 2))
    lags = np.arange(0, limit + 1)
    acf = np.array([1.0] + [float(np.dot(centered[:-k], centered[k:]) / denom) for k in range(1, limit + 1)])
    fig, ax, _ = single_axis(ax, figsize)
    markerline, stemlines, baseline = ax.stem(lags, acf, basefmt=" ")
    try:
        markerline.set_color(DEFAULT_COLORS["band"])
        stemlines.set_color(DEFAULT_COLORS["band"])
        stemlines.set_linewidth(1.5)
    except Exception:
        pass
    band = 1.96 / np.sqrt(n)
    ax.axhspan(-band, band, color=DEFAULT_COLORS["neutral"], alpha=0.18, label="95% white-noise band")
    ax.axhline(0.0, color="black", linewidth=1.0)
    _lag_ticks(ax, lags)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Residual ACF — {data.model_name}")


def observed_vs_fitted(result: Any, ax=None, *, seed: int = 42, log_scale: bool = False, symmetric_log: bool = False, loess: bool = False,
                       include_zero: bool = True, figsize=(5.5, 4.5), title: str | None = None):
    """Observed response against the fitted mean, with the identity line.

    ``symmetric_log`` uses matching symlog axes for skewed nonnegative
    responses, so extreme values do not flatten the cloud or the identity line.
    ``loess=True`` adds a local-linear calibration curve; it is used by the GLS
    mosaic, where no random effects can create a separate conditional fit.
    ``include_zero=False`` replaces the zero baseline with multiplicative
    limits derived from each axis's own data (``0.95 × min`` to ``1.03 ×
    max``), which fills the panel when the data live far from zero.
    """
    data = resolve(result, seed=seed)
    y = _need(data.response, "Observed values", data)
    mu = _need(data.fitted, "Fitted values", data)
    y, mu = clean(y, mu)
    if log_scale:
        keep = (y > 0) & (mu > 0)
        y, mu = y[keep], mu[keep]
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(mu, y, alpha=0.55, color=DEFAULT_COLORS["fitted"], edgecolors="none")
    _overlap_identity(ax, mu, y)
    if loess:
        _loess_line(ax, mu, y, color="#15803d", label="LOESS trend")
    if symmetric_log:
        _symmetric_log_scale(ax, mu, y)
    elif log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif include_zero:
        # Scale each axis from its own variable.  The identity line is drawn
        # on the overlap, so it no longer stretches the fitted axis to an
        # extreme observed response (common with continuous hurdle data).
        nonnegative = bool(y.size and np.all(y >= 0) and np.all(mu >= 0))
        x_limits = _linear_limits(mu, include_zero=nonnegative)
        y_limits = _linear_limits(y, include_zero=nonnegative)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
    else:
        x_limits = _padded_limits(mu)
        y_limits = _padded_limits(y)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
    ax.set_xlabel("Fitted mean E[y]")
    ax.set_ylabel("Observed response")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Observed vs. fitted — {data.model_name}")


def residuals_panel(result: Any, ax=None, *, seed: int = 42, figsize=(11.5, 4.6), title: str | None = None):
    """Two-panel residual summary: residuals vs. fitted and a normal Q-Q."""
    data = resolve(result, seed=seed)
    fig, axes = panel_axes(ax, 1, 2, figsize=figsize)
    left, right = axes.ravel()
    residuals_vs_fitted(result, ax=left, seed=seed)
    qq_plot(result, ax=right, seed=seed)
    left.set_title(f"{data.residual_label} vs. fitted")
    right.set_title("Normal Q-Q")
    label = title or f"Residual diagnostics — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def pp_plot(result: Any, ax=None, *, seed: int = 42, figsize=(5.5, 4.5), title: str | None = None):
    """Probability-probability plot of the residuals against N(0, 1)."""
    from scipy import stats

    data = resolve(result, seed=seed)
    values = _need(data.residuals, "Residuals", data)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite residuals are available")

    ordered = np.sort(values)
    probs = (np.arange(1, ordered.size + 1) - 0.375) / (ordered.size + 0.25)
    empirical = stats.norm.cdf(ordered, loc=np.mean(values), scale=np.std(values, ddof=1) or 1.0)

    fig, ax, _ = single_axis(ax, figsize)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.3, label="Perfect fit")
    ax.plot(probs, empirical, "o", color=DEFAULT_COLORS["band"], alpha=0.65, markersize=4, label="Residuals")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Theoretical cumulative probability")
    ax.set_ylabel("Empirical cumulative probability")
    ax.legend(loc="lower right", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"P-P plot — {data.residual_label}")


def half_normal(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_sim: int = 99,
    figsize=(5.8, 4.6),
    title: str | None = None,
):
    """Half-normal plot of absolute residuals with a simulated envelope.

    The envelope is built by drawing ``n_sim`` standard-normal samples of the
    same size as the data and keeping the 5th/95th percentiles of the ordered
    absolute values, which is a fast reference for "no systematic misfit".
    """
    from scipy import stats

    data = resolve(result, seed=seed)
    values = _need(data.residuals, "Residuals", data)
    values = np.abs(values[np.isfinite(values)])
    if values.size < 3:
        raise ValueError("at least three finite residuals are needed for a half-normal plot")
    ordered = np.sort(values)
    n = ordered.size
    theoretical = stats.halfnorm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    scale = np.std(values, ddof=1) or 1.0

    rng = np.random.default_rng(seed)
    sims = np.sort(np.abs(rng.standard_normal((int(n_sim), n))), axis=1) * scale
    lower = np.percentile(sims, 5, axis=0)
    upper = np.percentile(sims, 95, axis=0)

    fig, ax, _ = single_axis(ax, figsize)
    ax.fill_between(theoretical, lower, upper, color="#93c5fd", alpha=0.35, label="95% simulation envelope")
    ax.plot(theoretical, ordered, "o", color=DEFAULT_COLORS["band"], alpha=0.7, markersize=4, label="|Residuals|")
    slope = np.median(ordered / np.maximum(theoretical, 1e-12))
    ax.plot(theoretical, slope * theoretical, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.5, label="Reference")
    ax.set_xlabel("Half-normal quantiles")
    ax.set_ylabel(f"|{data.residual_label.lower()}|")
    ax.legend(loc="upper left", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Half-normal plot — {data.model_name}")


def worm_plot(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    n_intervals: int = 8,
    figsize=(5.8, 4.6),
    title: str | None = None,
):
    """Worm plot: local deviations of the Q-Q plot from the identity line."""
    data = resolve(result, seed=seed)
    values = _need(data.residuals, "Residuals", data)
    values = values[np.isfinite(values)]
    if values.size < 8:
        raise ValueError("at least eight finite residuals are needed for a worm plot")

    osm, osr = qq_points(values)
    n = osm.size
    n_intervals = max(int(n_intervals), 2)
    dev = np.empty_like(osr)
    for chunk in np.array_split(np.arange(n), n_intervals):
        if chunk.size < 2:
            dev[chunk] = 0.0
            continue
        x0, x1 = osm[chunk[0]], osm[chunk[-1]]
        y0, y1 = osr[chunk[0]], osr[chunk[-1]]
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        dev[chunk] = osr[chunk] - (y0 + slope * (osm[chunk] - x0))

    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    se = np.sqrt(p * (1.0 - p)) / np.maximum(_normal_density(p), 1e-12) / np.sqrt(n)

    fig, ax, _ = single_axis(ax, figsize)
    ax.axhspan(-2.0 * np.nanmedian(se), 2.0 * np.nanmedian(se), color="#93c5fd", alpha=0.3, label="±2 SE band")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.plot(osm, dev, "o-", color=DEFAULT_COLORS["band"], markersize=4, linewidth=1.4, label="Deviation")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Deviation from the Q-Q line")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Worm plot — {data.model_name}")


def grouped_residuals(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    max_groups: int = 60,
    sort: bool = True,
    figsize=None,
    title: str | None = None,
):
    """Residual boxplots by group, with the individual observations overlaid.

    Intended for a handful of groups.  For many subjects use
    :func:`binned_standardized_residuals` instead; it stays readable at any
    number of grouping levels.
    """
    import pandas as pd

    data = resolve(result, seed=seed)
    values = _need(data.residuals, "Residuals", data)
    if data.groups is None:
        raise ValueError(f"{data.model_name} does not expose a grouping variable")
    groups = np.asarray(data.groups)
    if groups.size != values.size:
        raise ValueError("the grouping variable does not match the observations")
    keep = np.isfinite(values)
    if keep.sum() == 0:
        raise ValueError("no finite residuals are available")
    levels, codes = np.unique(np.asarray(groups)[keep].astype(str), return_inverse=True)
    frame = pd.DataFrame({"group": levels[codes], "residual": values[keep]})
    order = frame.groupby("group")["residual"].median().sort_values().index if sort else levels
    labels = [str(g) for g in order]
    if len(labels) > max_groups:
        picks = np.linspace(0, len(labels) - 1, max_groups).round().astype(int)
        labels = [labels[i] for i in picks]
    blocks = [frame.loc[frame["group"] == label, "residual"].to_numpy() for label in labels]
    if figsize is None:
        figsize = (max(6.5, 0.25 * len(labels) + 3.5), 4.8)

    fig, ax, _ = single_axis(ax, figsize)
    try:
        bp = ax.boxplot(blocks, tick_labels=labels, patch_artist=True)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(blocks, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#93c5fd")
        patch.set_alpha(0.7)
    rng = np.random.default_rng(seed)
    for pos, block in enumerate(blocks, start=1):
        jitter = rng.uniform(-0.16, 0.16, size=block.size)
        ax.scatter(pos + jitter, block, s=12, color="#1e3a8a", alpha=0.55, edgecolors="none", zorder=3)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel(data.group_name or "Group")
    ax.set_ylabel(data.residual_label)
    if len(labels) > 12:
        ax.tick_params(axis="x", rotation=90, labelsize=8)
    style_axis(ax)
    return _finish(fig, ax, title or f"Residuals by group — {data.model_name}")


def variogram_plot(
    result: Any,
    ax=None,
    *,
    n_lags: int = 20,
    max_lag: float | None = None,
    figsize=(6.5, 4.6),
    title: str | None = None,
):
    """Empirical semi-variogram of the residuals over time."""
    from .registry import empirical_variogram

    data = resolve(result)
    frame = empirical_variogram(result, n_lags=n_lags, max_lag=max_lag)
    fig, ax, _ = single_axis(ax, figsize)
    sizes = 18.0 + 60.0 * frame["n_pairs"].to_numpy(dtype=float) / max(frame["n_pairs"].max(), 1)
    ax.plot(frame["lag"], frame["semivariance"], color=DEFAULT_COLORS["band"], linewidth=1.8, label="Semi-variogram")
    ax.scatter(frame["lag"], frame["semivariance"], s=sizes, color=DEFAULT_COLORS["band"], alpha=0.65, edgecolors="#064e3b")

    residuals = np.asarray(data.residuals, dtype=float)
    sill = float(np.nanvar(residuals))
    if np.isfinite(sill) and sill > 0:
        ax.axhline(sill, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.3, label=f"Residual variance ({sill:.3g})")
    ax.set_xlabel(data.time_name or "Lag")
    ax.set_ylabel("Semi-variance γ(h)")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Residual semi-variogram — {data.model_name}")


def pearson_residuals(
    result: Any,
    ax=None,
    *,
    figsize=None,
    title: str | None = None,
):
    """Pearson residuals against the fitted mean, one panel per response.

    Univariate fits produce a single panel.  Multivariate ``MCGLMResults``
    produce one panel per response, with the response names as titles.
    """
    data = resolve(result)
    model = data.result
    K = int(getattr(model, "K", 1) or 1)
    if K > 1 and hasattr(model, "resid_pearson") and hasattr(model, "nobs"):
        n = int(model.nobs)
        residuals = np.asarray(model.resid_pearson, dtype=float).reshape(K, n)
        fitted = np.asarray(model.fitted, dtype=float).reshape(K, n)
        exog = getattr(getattr(model, "model", None), "endog_names", None) or []
        names = [str(exog[k]) if k < len(exog) else f"Response {k + 1}" for k in range(K)]
    else:
        residuals = _need(data.residuals, "Residuals", data)[None, :]
        fitted = _need(data.fitted, "Fitted values", data)[None, :]
        names = [data.response_name]
        K = 1

    n_cols = min(K, 3)
    n_rows = int(np.ceil(K / n_cols))
    if figsize is None:
        figsize = (4.3 * n_cols, 3.6 * n_rows)
    fig, axes = panel_axes(ax, n_rows, n_cols, figsize=figsize)
    flat = axes.ravel()

    for k in range(K):
        panel = flat[k]
        res_k = residuals[k]
        mu_k = fitted[k]
        keep = np.isfinite(res_k) & np.isfinite(mu_k)
        panel.scatter(mu_k[keep], res_k[keep], alpha=0.55, color=DEFAULT_COLORS["residual"], edgecolors="none")
        panel.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        panel.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
        panel.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
        panel.set_title(f"[{names[k]}] Pearson residuals", fontsize=9.5)
        panel.set_xlabel("Fitted mean (μ)", fontsize=8.5)
        panel.set_ylabel("Pearson residual", fontsize=8.5)
        style_axis(panel)

    for k in range(K, n_rows * n_cols):
        flat[k].set_axis_off()

    label = title or f"Pearson residuals — {data.model_name}"
    if K == 1:
        flat[0].set_title(label)
    else:
        fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


# ---------------------------------------------------------------------------
# Count, binary and survival diagnostics
# ---------------------------------------------------------------------------


def _nb_theta(model) -> float:
    fam = getattr(model, "family", None)
    params = getattr(fam, "params", None)
    theta = getattr(params, "theta", None) if params is not None else None
    if theta is None:
        theta = getattr(fam, "theta", None)
    if theta is None:
        extra = getattr(model, "extra_param", None)
        if extra:
            theta = 1.0 / float(extra)
    if theta is None:
        raise NotImplementedError("the negative-binomial theta is not available for this fit")
    return float(theta)


_COUNT_FAMILIES = frozenset({
    "Poisson", "QuasiPoisson", "NegativeBinomial", "QuasiNegativeBinomial",
    "TruncatedPoisson", "TruncatedGeometric", "TruncatedNegativeBinomial",
    "ZeroInflatedPoisson", "ValueInflatedPoisson", "HurdlePoisson",
    "ZeroInflatedNegativeBinomial", "HurdleNegativeBinomial", "HurdleGeometric",
})


def supports_rootogram(model) -> bool:
    """Whether exact count probabilities are available for ``model``'s family."""
    return type(getattr(model, "family", None)).__name__ in _COUNT_FAMILIES


def _expected_count_frequencies(model, k_vals: np.ndarray) -> np.ndarray:
    """Expected counts summed over the sample for count / inflated / hurdle models."""
    from scipy import stats

    fam = getattr(model, "family", None)
    name = type(fam).__name__
    mu = np.asarray(model.fittedvalues, dtype=float)

    def _sum_pmf(pmf):
        return np.array([float(np.sum(pmf(k))) for k in k_vals])

    if name in ("Poisson", "QuasiPoisson"):
        return _sum_pmf(lambda k: stats.poisson.pmf(k, mu))
    if name in ("NegativeBinomial", "QuasiNegativeBinomial"):
        theta = _nb_theta(model)
        p = theta / (theta + mu)
        return _sum_pmf(lambda k: stats.nbinom.pmf(k, theta, p))
    if name == "TruncatedPoisson":
        p0 = stats.poisson.pmf(0, mu)
        return _sum_pmf(lambda k: np.where(k == 0, 0.0, stats.poisson.pmf(k, mu) / np.maximum(1.0 - p0, 1e-12)))
    if name == "TruncatedGeometric":
        p = 1.0 / (1.0 + mu)
        p0 = p
        return _sum_pmf(lambda k: np.where(k == 0, 0.0, (p * (1.0 - p) ** k) / np.maximum(1.0 - p0, 1e-12)))
    if name == "TruncatedNegativeBinomial":
        theta = _nb_theta(model)
        p = theta / (theta + mu)
        p0 = stats.nbinom.pmf(0, theta, p)
        return _sum_pmf(lambda k: np.where(k == 0, 0.0, stats.nbinom.pmf(k, theta, p) / np.maximum(1.0 - p0, 1e-12)))

    # Two-part models fitted by the MLE engine.
    if getattr(model, "_mle_supported", lambda: False)():
        lam = np.asarray(model.predict(which="location"), dtype=float)
        pi = np.asarray(model.predict(which="secondary"), dtype=float)
        if name == "ZeroInflatedPoisson":
            return _sum_pmf(lambda k: np.where(k == 0, pi + (1.0 - pi) * stats.poisson.pmf(k, lam), (1.0 - pi) * stats.poisson.pmf(k, lam)))
        if name == "ValueInflatedPoisson":
            target = int(round(float(getattr(fam, "target_value", 0))))
            return _sum_pmf(lambda k: np.where(k == target, pi + (1.0 - pi) * stats.poisson.pmf(k, lam), (1.0 - pi) * stats.poisson.pmf(k, lam)))
        if name == "HurdlePoisson":
            p0 = stats.poisson.pmf(0, lam)
            return _sum_pmf(lambda k: np.where(k == 0, 1.0 - pi, pi * stats.poisson.pmf(k, lam) / np.maximum(1.0 - p0, 1e-12)))
        if name in ("ZeroInflatedNegativeBinomial", "HurdleNegativeBinomial"):
            theta = _nb_theta(model)
            p = theta / (theta + lam)
            if name == "ZeroInflatedNegativeBinomial":
                return _sum_pmf(lambda k: np.where(k == 0, pi + (1.0 - pi) * stats.nbinom.pmf(k, theta, p), (1.0 - pi) * stats.nbinom.pmf(k, theta, p)))
            p0 = stats.nbinom.pmf(0, theta, p)
            return _sum_pmf(lambda k: np.where(k == 0, 1.0 - pi, pi * stats.nbinom.pmf(k, theta, p) / np.maximum(1.0 - p0, 1e-12)))
        if name == "HurdleGeometric":
            p = 1.0 / (1.0 + lam)
            return _sum_pmf(lambda k: np.where(k == 0, 1.0 - pi, pi * (p * (1.0 - p) ** k) / np.maximum(1.0 - p, 1e-12)))

    raise NotImplementedError(
        f"exact count probabilities are not available for family '{name}'; "
        "the rootogram is only defined for count models."
    )


def rootogram(result: Any, ax=None, *, max_count: int | None = None, figsize=(7.5, 4.5), title: str | None = None):
    """Hanging rootogram (Kleiber & Zeileis) for count models."""
    data = resolve(result)
    model = data.result
    y_obs = np.rint(_need(data.response, "Observed values", data)).astype(int)
    upper = int(y_obs.max()) if max_count is None else int(max_count)
    k_vals = np.arange(0, upper + 1)
    obs = np.array([int(np.sum(y_obs == k)) for k in k_vals], dtype=float)
    expected = _expected_count_frequencies(model, k_vals)
    fig, ax, _ = single_axis(ax, figsize)
    sqrt_obs = np.sqrt(obs)
    sqrt_exp = np.sqrt(expected)
    ax.bar(
        k_vals,
        sqrt_obs,
        bottom=sqrt_exp - sqrt_obs,
        width=0.8,
        color="#60a5fa",
        edgecolor="#1e3a8a",
        alpha=0.78,
        label="Observed (hanging)",
    )
    ax.plot(k_vals, sqrt_exp, "o-", color=DEFAULT_COLORS["accent"], linewidth=2.0, label="Expected")
    ax.axhline(0.0, color="black", linewidth=1.1)
    if len(k_vals) > 20:
        ticks = np.linspace(0, upper, 10, dtype=int)
    else:
        ticks = k_vals
    ax.set_xticks(ticks)
    ax.set_xlabel("Count (k)")
    ax.set_ylabel("√Frequency")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Hanging rootogram — {data.model_name}")


def calibration_plot(
    result: Any,
    ax=None,
    *,
    n_bins: int = 10,
    alpha: float = 0.05,
    exog=None,
    y_true=None,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Reliability diagram and Brier score for a binary-response model.

    Delegates to the evaluator's implementation so the panel and the standalone
    ``kind="calibration"`` share the same Wilson intervals and label headroom.
    """
    from .classification import ClassificationEvaluator

    evaluator = ClassificationEvaluator(result, exog=exog, y_true=y_true)
    return evaluator.plot_calibration(
        ax=ax, n_bins=n_bins, alpha=alpha, figsize=figsize, title=title,
    )


def _survival_status(result: Any, size: int) -> np.ndarray:
    """Event indicator (1 = event, 0 = right-censored) for survival GLM fits."""
    ctx = getattr(result, "_fit_ctx", {}) or {}
    raw = ctx.get("d")
    if raw is None:
        return np.ones(size, dtype=float)
    status = np.asarray(raw, dtype=float).reshape(-1)
    return status if status.size == size else np.ones(size, dtype=float)


def _survival_eta(result: Any, data: PlotData) -> np.ndarray:
    """Linear predictor of the mean/time submodel (μ for GeneralizedGamma)."""
    eta = data.linear_predictor
    if eta is None or family_name(result) == "GeneralizedGamma":
        fn = getattr(result, "_gengamma_mu_linear_predictor", None)
        if callable(fn):
            try:
                eta = fn()
            except Exception:  # pragma: no cover - depends on the result class
                pass
    return _need(eta, "Linear predictor", data)


def _log_limits(values: Any, *, pad: float = 0.10, min_pad_dex: float = 0.05):
    """Symmetric log-axis limits from ``values`` alone.

    The padding is in log10 units (decades), so a narrow range of fitted times
    gets the same visible breathing room as a wide range of observed times.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    lo, hi = float(np.log10(arr.min())), float(np.log10(arr.max()))
    half_pad = max((hi - lo) * pad, min_pad_dex)
    return float(10.0 ** (lo - half_pad)), float(10.0 ** (hi + half_pad))


def cox_snell(result: Any, ax=None, *, figsize=(6.0, 4.5), title: str | None = None):
    """Cox-Snell goodness-of-fit plot for censored AFT survival models.

    Under a correct model the Cox-Snell residuals ``r = -log S(t)`` are a
    unit-exponential sample, subject to the same censoring as ``t``.  The
    Nelson-Aalen estimate of their cumulative hazard should therefore follow
    the identity.  A pointwise 95% Greenwood band and a rug of censored
    residuals make departures and heavy censoring easier to read.
    """
    data = resolve(result)
    model = data.result
    rcs = getattr(model, "resid_cox_snell", None)
    if rcs is None:
        raise ValueError(f"Cox-Snell residuals are not available for {data.model_name}.")
    rcs = np.asarray(rcs, dtype=float).reshape(-1)
    events = _survival_status(model, rcs.size)
    keep = np.isfinite(rcs) & np.isfinite(events)
    rcs, events = rcs[keep], events[keep]
    if rcs.size == 0:
        raise ValueError(f"no finite Cox-Snell residuals for {data.model_name}.")

    order = np.argsort(rcs)
    rcs_sorted = rcs[order]
    events_sorted = events[order]
    n_at_risk = np.arange(rcs_sorted.size, 0, -1)
    hazard = np.cumsum(events_sorted / n_at_risk)
    greenwood = np.cumsum(events_sorted / n_at_risk**2)
    band = 1.96 * np.sqrt(np.maximum(greenwood, 0.0))

    # Start the step and its band at the origin, not at the first residual.
    x_step = np.concatenate(([0.0], rcs_sorted))
    y_step = np.concatenate(([0.0], hazard))
    b_step = np.concatenate(([0.0], band))

    fig, ax, _ = single_axis(ax, figsize)
    ax.fill_between(
        x_step, np.maximum(y_step - b_step, 0.0), y_step + b_step,
        step="post", color=DEFAULT_COLORS["band"], alpha=0.22, linewidth=0.0,
        label="95% Greenwood band",
    )
    ax.step(x_step, y_step, where="post", color="#047857", linewidth=2.0, label="Nelson-Aalen H(r)")
    censored = rcs_sorted[events_sorted < 0.5]
    if censored.size:
        ax.plot(
            censored, np.zeros_like(censored), linestyle="None", marker="|",
            markersize=8, color=DEFAULT_COLORS["neutral"], label="Censored r_CS",
        )
    # Both axes keep the same numeric scale so H(r) can be compared with r,
    # but the subplot is allowed to use its full rectangular box.  Forcing an
    # equal box aspect shrank the plot and made the curve look displaced.
    top = float(max(np.max(rcs), np.max(hazard + band), 1.0))
    limit = top * 1.10
    ax.plot([0, limit], [0, limit], color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.5, label="Unit exponential")
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("Cox-Snell residuals")
    ax.set_ylabel("Estimated cumulative hazard")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Cox-Snell goodness of fit — {data.model_name}")


def deviance_martingale_vs_eta(
    result: Any,
    ax=None,
    *,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Deviance-martingale residuals against the linear predictor.

    Unlike quantile residuals, martingale residuals are defined for censored
    rows too: every observation contributes.  Events and censored rows are
    drawn separately and a local-linear trend helps spot a misspecified
    functional form in the linear predictor.  These residuals are not normal
    scores, so no ``±2`` reference lines are drawn.
    """
    data = resolve(result)
    res = getattr(result, "resid_deviance_martingale", None)
    if res is None:
        raise ValueError(f"deviance-martingale residuals are not available for {data.model_name}.")
    eta = _survival_eta(result, data)
    res = _need(res, "Deviance-martingale residuals", data)
    status = _survival_status(result, res.size)
    eta, res, status = clean(eta, res, status)
    if eta.size < 3:
        raise ValueError("not enough finite deviance-martingale residuals for this plot")

    events = status > 0.5
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(eta[events], res[events], s=19, alpha=0.6, color="#7c3aed", edgecolors="none", label="Event")
    if np.any(~events):
        ax.scatter(
            eta[~events], res[~events], s=20, alpha=0.6, marker="x",
            color=DEFAULT_COLORS["neutral"], label="Censored",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    _loess_line(ax, eta, res)
    ax.margins(x=0.08, y=0.10)
    n_censored = int(np.count_nonzero(~events))
    ax.text(
        0.03, 0.94, f"{n_censored} censored", transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78),
    )
    ax.set_xlabel("Linear predictor (η)")
    ax.set_ylabel("Deviance-martingale residuals")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(
        fig, ax,
        title or f"Deviance-martingale residuals vs. η — {data.model_name}",
    )


def _survival_observed_vs_fitted(
    result: Any,
    ax=None,
    *,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Observed-vs-fitted for survival times, with censoring made explicit."""
    data = resolve(result)
    y = _need(data.response, "Observed values", data)
    mu = _need(data.fitted, "Fitted values", data)
    status = _survival_status(result, y.size)
    y, mu, status = clean(y, mu, status)
    keep = (y > 0) & (mu > 0)
    y, mu, status = y[keep], mu[keep], status[keep]
    if y.size == 0:
        raise ValueError(f"no positive observed/fitted times for {data.model_name}.")

    events = status > 0.5
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(mu[events], y[events], alpha=0.55, color=DEFAULT_COLORS["fitted"], edgecolors="none", label="Event")
    if np.any(~events):
        ax.scatter(
            mu[~events], y[~events], alpha=0.6, marker="x", s=32,
            color=DEFAULT_COLORS["neutral"], label="Censored",
        )
    # Scale each axis from its own variable.  Extending the identity line to
    # the most extreme observed time used to open the fitted axis far beyond
    # the fitted values, pushing the whole cloud against one side.  Draw the
    # reference only where both observed and fitted times overlap.
    overlap_lo = max(float(y.min()), float(mu.min()))
    overlap_hi = min(float(y.max()), float(mu.max()))
    if overlap_lo < overlap_hi:
        ax.plot(
            [overlap_lo, overlap_hi], [overlap_lo, overlap_hi],
            color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.5, label="Identity",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    x_limits, y_limits = _log_limits(mu), _log_limits(y)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    n_censored = int(np.count_nonzero(~events))
    ax.text(
        0.03, 0.94, f"{n_censored} censored", transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78),
    )
    ax.set_xlabel("Fitted mean E[T]")
    ax.set_ylabel("Observed time")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Observed vs. fitted — {data.model_name}")


def truncated_fit(result: Any, ax=None, *, bins: int = 25, figsize=(6.0, 4.5), title: str | None = None):
    """Observed vs. fitted response with truncation limits, for truncated families."""
    data = resolve(result)
    model = data.result
    y = _need(data.response, "Observed values", data)
    mu = _need(data.fitted, "Fitted values", data)
    ctx = getattr(model, "_fit_ctx", {}) or {}
    lo, hi = ctx.get("left"), ctx.get("right")
    fig, ax, _ = single_axis(ax, figsize)
    ax.hist(y, bins=bins, density=True, alpha=0.55, color="#7dd3fc", edgecolor="#0c4a6e", label="Observed")
    ax.hist(mu, bins=bins, density=True, alpha=0.45, color="#fb923c", edgecolor="#9a3412", label="Fitted mean")
    if lo is not None:
        ax.axvline(float(lo), color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=2.0, label=f"Left limit ({lo})")
    if hi is not None:
        ax.axvline(float(hi), color="#7f1d1d", linestyle="--", linewidth=2.0, label=f"Right limit ({hi})")
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Truncated fit — {data.model_name}")


def _is_continuous_truncated_glm(result: Any) -> bool:
    """Whether a fitted result is a GLM with a continuous truncated family."""
    if type(result).__name__ != "GLM" or not hasattr(result, "_mle_comp_full"):
        return False
    name = family_name(result) or type(getattr(result, "family", None)).__name__ or ""
    name = name.replace("_", "").replace("-", "").replace(" ", "").lower()
    if "truncated" not in name:
        return False
    return not any(word in name for word in ("poisson", "negativebinomial", "geometric"))


def _truncation_limits(result: Any) -> tuple[float | None, float | None]:
    """Read the fitted left and right truncation limits."""
    ctx = getattr(result, "_fit_ctx", {}) or {}
    lo, hi = ctx.get("left"), ctx.get("right")
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


def _qq_envelope_panel(ax, values, *, seed: int, n_sim: int, model_name: str,
                       point_label: str = "Quantile residuals",
                       value_label: str = "Ordered quantile residuals"):
    """Normal Q-Q of quantile residuals with a simulated 95% envelope."""
    from scipy.stats import norm

    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 5:
        raise ValueError("not enough finite quantile residuals for the Q-Q panel")
    theoretical, ordered = qq_points(values)
    rng = np.random.default_rng(seed)
    sims = np.sort(rng.standard_normal((int(n_sim), values.size)), axis=1)
    probs = (np.arange(1, values.size + 1) - 0.375) / (values.size + 0.25)
    sim_x = norm.ppf(probs)
    lower, upper = np.quantile(sims, [0.025, 0.975], axis=0)

    fig, ax, _ = single_axis(ax)
    ax.fill_between(
        sim_x, lower, upper, color=DEFAULT_COLORS["grid"], alpha=0.8,
        linewidth=0.0, label="95% simulation envelope",
    )
    ax.scatter(theoretical, ordered, s=17, alpha=0.7, color="#4338ca",
               edgecolors="none", label=point_label)
    low = float(min(theoretical.min(), sim_x.min(), lower.min()))
    high = float(max(theoretical.max(), sim_x.max(), upper.max()))
    ax.plot([low, high], [low, high], color=DEFAULT_COLORS["accent"],
            linestyle="--", linewidth=1.6, label="N(0,1) reference")
    ax.set_xlabel("Theoretical N(0, 1) quantiles")
    ax.set_ylabel(value_label)
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    _finish(fig, ax, f"Q-Q with 95% envelope — {model_name}")


def _residual_spread_panel(ax, x, residuals, *, xlabel: str, model_name: str,
                           residual_label: str = "Quantile residual",
                           point_label: str = "Quantile residuals",
                           root_label: str = "√|quantile residual|"):
    """Residuals vs a predictor with mean and spread trends in one panel.

    The two y axes are always scaled from the data; only reference lines at
    zero and ±2 are drawn.  A fixed ``[-2, 2]`` limit would hide the fan shape
    this panel is designed to expose.
    """
    x, residuals = clean(x, residuals)
    if x.size < 5:
        raise ValueError("not enough finite residuals for the residual-spread panel")
    root = np.sqrt(np.abs(residuals))

    fig, ax, _ = single_axis(ax)
    ax.scatter(x, residuals, s=17, alpha=0.62, color="#4338ca",
               edgecolors="none", label=point_label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.axhline(-2.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    _loess_line(ax, x, residuals, color=DEFAULT_COLORS["accent"], label="Mean trend")

    right = ax.twinx()
    right.scatter(x, root, s=10, alpha=0.18, color="#7c3aed", edgecolors="none")
    curve = _loess_curve(x, root)
    if curve is not None:
        grid, fitted, lower, upper = curve
        good = np.isfinite(fitted) & np.isfinite(lower) & np.isfinite(upper)
        if good.any():
            right.fill_between(grid[good], lower[good], upper[good], color="#7c3aed",
                               alpha=0.14, linewidth=0.0, label="95% spread band")
            right.plot(grid[good], fitted[good], color="#7c3aed",
                       linewidth=2.0, label="√|r| trend")
    expected = float(np.sqrt(2.0 / np.pi))
    right.axhline(expected, color="#7c3aed", linestyle=":", linewidth=1.0,
                  label="N(0,1) √|Z|")
    right.set_ylim(bottom=0.0)
    right.set_ylabel(root_label, color="#7c3aed")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(residual_label)
    handles, labels = ax.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    ax.legend(handles + right_handles, labels + right_labels, loc="upper left", fontsize=8)
    style_axis(ax)
    _finish(fig, ax, f"Residuals and spread vs. η — {model_name}")


def _density_hist_edges(y, requested: int):
    """Histogram edges for the observed density panel.

    Discrete or heavily rounded responses produce very tall bars with the
    default 30 bins, which makes the smooth predictive density unreadable.
    With at most 20 distinct values we therefore centre one bin on each value;
    otherwise Freedman-Diaconis and Sturges are combined and capped at 20 bins.
    """
    values = np.asarray(y, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    requested = max(1, int(requested))
    if values.size < 2:
        return requested

    unique = np.unique(values)
    if unique.size <= 20:
        if unique.size == 1:
            width = max(abs(float(unique[0])) * 0.1, 1.0)
        else:
            diffs = np.diff(unique)
            positive = diffs[diffs > 0]
            width = float(np.min(positive)) if positive.size else 1.0
        return np.concatenate((
            [unique[0] - width / 2.0],
            (unique[:-1] + unique[1:]) / 2.0,
            [unique[-1] + width / 2.0],
        ))

    q25, q75 = np.percentile(values, [25.0, 75.0])
    if q75 > q25:
        fd = int(np.histogram_bin_edges(values, bins="fd").size - 1)
    else:
        fd = requested
    sturges = int(np.histogram_bin_edges(values, bins="sturges").size - 1)
    target = min(fd if fd > 0 else requested, sturges if sturges > 0 else requested)
    return int(np.clip(target, 5, min(requested, 20)))


def _truncated_density_panel(result: Any, ax, data: PlotData, *, lo, hi, bins: int, grid_size: int):
    """Observed density against the mean per-row truncated predictive density."""
    from ..glm import mle_dist as _mle_dist

    y = _need(data.response, "Observed values", data)
    mu = _need(data.fitted, "Fitted values", data)
    y, mu = clean(y, mu)
    if y.size < 5:
        raise ValueError("not enough finite observations for the density panel")

    comp, _ = result._mle_comp_full()
    comp_mu = np.asarray(getattr(comp, "mu", mu), dtype=float).reshape(-1)
    if comp_mu.size != y.size:
        comp_mu = mu
    extra = np.asarray(getattr(comp, "extra", None), dtype=float).reshape(-1)
    if extra.size == 1:
        extra = np.full(comp_mu.shape, extra[0])
    elif extra.size != comp_mu.size:
        raise ValueError("the fitted scale parameter does not match the response length")

    values = np.concatenate((y, comp_mu))
    span = float(np.max(values) - np.min(values))
    pad = span * 0.08 if span > 0 else max(abs(float(np.mean(values))) * 0.08, 1e-6)
    left = float(np.min(values)) - pad if lo is None else float(lo)
    right = float(np.max(values)) + pad if hi is None else float(hi)
    if right <= left:
        width = max(abs(left) * 0.1, 1.0)
        left, right = left - width, right + width
    grid = np.linspace(left, right, int(grid_size))

    comp_plot = SimpleNamespace(
        mu=comp_mu[None, :],
        extra=extra[None, :],
        sec=getattr(comp, "sec", None),
        gamma=getattr(comp, "gamma", None),
        left=getattr(comp, "left", None),
        right=getattr(comp, "right", None),
    )
    log_density = _mle_dist.logpdf(result.family, comp_plot, grid[:, None])
    density = np.nanmean(np.exp(np.clip(log_density, -700.0, 700.0)), axis=1)

    fig, ax, _ = single_axis(ax)
    ax.hist(y, bins=_density_hist_edges(y, bins), density=True, alpha=0.45,
            color="#7dd3fc", edgecolor="#0c4a6e", label="Observed")
    ax.plot(grid, density, color=DEFAULT_COLORS["accent"], linewidth=2.2,
            label="Mean predictive density")
    if lo is not None:
        ax.axvline(float(lo), color="#0f766e", linestyle="--", linewidth=1.8,
                   label=f"Left limit ({lo:g})")
    if hi is not None:
        ax.axvline(float(hi), color="#7f1d1d", linestyle="--", linewidth=1.8,
                   label=f"Right limit ({hi:g})")
    ax.set_xlim(left, right)
    ax.set_xlabel("Response")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    _finish(fig, ax, f"Predictive density with truncation — {data.model_name}")


def _truncated_calibration_panel(ax, y, mu, *, model_name: str):
    """Observed response against the fitted truncated conditional mean."""
    y, mu = clean(y, mu)
    if y.size < 5:
        raise ValueError("not enough finite observations for the calibration panel")

    fig, ax, _ = single_axis(ax)
    ax.scatter(mu, y, s=20, alpha=0.62, color="#0e7490", edgecolors="none",
               label="Observed vs. fitted")
    _overlap_identity(ax, mu, y, label="Identity")
    curve = _loess_curve(mu, y)
    if curve is not None:
        grid, fitted, lower, upper = curve
        good = np.isfinite(fitted) & np.isfinite(lower) & np.isfinite(upper)
        if good.any():
            ax.fill_between(grid[good], lower[good], upper[good], color="#0e7490",
                            alpha=0.16, linewidth=0.0, label="95% CI")
            ax.plot(grid[good], fitted[good], color="#0e7490", linewidth=2.0,
                    label="Smooth trend")
    ax.set_xlabel("Predicted truncated mean E[y | a < y < b]")
    ax.set_ylabel("Observed y")
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    _finish(fig, ax, f"Truncated calibration — {model_name}")


def truncated_diagnostics(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    bins: int = 30,
    grid_size: int = 200,
    n_sim: int = 400,
    figsize=(13.0, 9.5),
    title: str | None = None,
):
    """A 2×2 diagnostic mosaic for continuous truncated GLM fits.

    The panels are chosen for the truncation problem rather than for the
    generic GLM workflow:

    1. a normal Q-Q of quantile residuals with a 95% simulation envelope,
       which checks the assumed distribution including its tails;
    2. quantile residuals against ``η`` with a mean trend and a separate
       spread trend, which checks the link and the fitted variance together;
    3. the observed density against the average per-row truncated predictive
       density, with dashed lines at the fitted truncation limits;
    4. observed responses against the fitted truncated conditional mean, with
       the identity line and a smoothed trend with a 95% interval.

    This is deliberately separate from ``kind="truncated"``: the old single
    panel compared raw responses with a histogram of fitted means and could
    hide exactly the edge behaviour this mosaic is designed to expose.

    Parameters
    ----------
    result : fitted GLM
        Must use a continuous truncated family (``TruncatedGaussian``).
    seed : int, default 42
        Seed for quantile residuals and the Q-Q envelope.  The same seed feeds
        both panels so they describe the same residual vector.
    bins : int, default 30
        Requested histogram bins for the observed-density panel. Discrete or
        rounded responses are binned adaptively and capped at 20 bars.
    grid_size : int, default 200
        Points used for the predictive density curve.
    n_sim : int, default 400
        Simulated normal samples behind the Q-Q envelope.
    """
    if not _is_continuous_truncated_glm(result):
        raise ValueError(
            "truncated_diagnostics supports continuous truncated GLM fits "
            "(TruncatedGaussian) only."
        )

    data = resolve(result, seed=seed)
    y_all = _need(data.response, "Observed values", data)
    mu_all = _need(data.fitted, "Fitted values", data)
    res_all = _quantile_residual_values(result, data, seed)
    if res_all is None:
        raise ValueError(f"quantile residuals are not available for {data.model_name}")

    eta_all = data.linear_predictor
    if eta_all is not None:
        eta_all = np.asarray(eta_all, dtype=float).reshape(-1)
        if eta_all.size != y_all.size:
            eta_all = None

    mask = finite(y_all, mu_all, res_all) if eta_all is None else finite(y_all, mu_all, res_all, eta_all)
    y, mu, residuals = y_all[mask], mu_all[mask], res_all[mask]
    eta = None if eta_all is None else eta_all[mask]
    if y.size < 5:
        raise ValueError("not enough finite observations for truncated diagnostics")
    x = eta if eta is not None else mu
    xlabel = "Linear predictor (η)" if eta is not None else "Fitted truncated mean (μ)"

    lo, hi = _truncation_limits(result)
    fig, axes = panel_axes(ax, 2, 2, figsize=figsize)
    a = axes.ravel()
    _qq_envelope_panel(a[0], residuals, seed=seed, n_sim=n_sim, model_name=data.model_name)
    _residual_spread_panel(a[1], x, residuals, xlabel=xlabel, model_name=data.model_name)
    _truncated_density_panel(result, a[2], data, lo=lo, hi=hi, bins=bins, grid_size=grid_size)
    _truncated_calibration_panel(a[3], y, mu, model_name=data.model_name)

    fig.suptitle(title or f"Truncated diagnostics — {data.model_name}", fontsize=14)
    return PlotResult(fig, title or "truncated diagnostics")


def hurdle_continuous(result: Any, ax=None, *, figsize=(6.0, 4.5), title: str | None = None):
    """Conditional positive component of a continuous hurdle model."""
    data = resolve(result)
    model = data.result
    y = _need(data.response, "Observed values", data)
    positive = y > 0
    if not np.any(positive):
        raise ValueError("the response has no positive observations")
    total = _need(data.fitted, "Fitted values", data)
    try:
        pi = np.asarray(model.predict(which="secondary"), dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError("the hurdle crossing probability is not available for this model") from exc
    conditional = total / np.maximum(pi, 1e-12)
    predicted, observed = clean(conditional[positive], y[positive])
    if predicted.size == 0:
        raise ValueError("no finite positive hurdle observations are available")
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(predicted, observed, alpha=0.6, color="#0e7490", edgecolors="none")
    _overlap_identity(ax, predicted, observed)
    # Continuous hurdle responses are heavily right-skewed.  Matching symlog
    # axes keep both extreme amounts and the bulk of the cloud readable.
    _symmetric_log_scale(ax, predicted, observed)
    ax.set_xlabel("Predicted E[y | y > 0]")
    ax.set_ylabel("Observed positive response")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Positive component — {data.model_name}")


def secondary_distribution(result: Any, ax=None, *, bins: int = 25, figsize=(6.0, 4.5), title: str | None = None):
    """Distribution of the secondary-submodel parameter (π, φ or σ²)."""
    data = resolve(result)
    model = data.result
    try:
        secondary = np.asarray(model.predict(which="secondary"), dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError("this model does not expose a secondary submodel") from exc
    secondary = secondary[np.isfinite(secondary)]
    fig, ax, _ = single_axis(ax, figsize)
    ax.hist(secondary, bins=bins, color="#34d399", edgecolor="#064e3b", alpha=0.82)
    ax.axvline(float(np.mean(secondary)), color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.6, label=f"Mean = {np.mean(secondary):.3f}")
    ax.set_xlabel("Secondary parameter (π / φ / σ²)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return _finish(fig, ax, title or f"Secondary submodel — {data.model_name}")


# ---------------------------------------------------------------------------
# Adaptive panel
# ---------------------------------------------------------------------------


def has_randomized_quantile_residuals(result) -> bool:
    """Whether an exact CDF is available to randomise discrete residuals.

    Quasi-likelihood families (e.g. QuasiPoisson) have no fully specified
    distribution, so they cannot provide randomised quantile residuals.
    """
    flag = getattr(result, "has_quantile_residuals", None)
    if flag is not None:
        try:
            return bool(flag)
        except Exception:
            return False
    return callable(getattr(result, "resid_quantile", None))


def _secondary_parameter_info(result):
    """Per-row secondary parameter, its label and whether it varies by row.

    For Beta/simplex the secondary parameter is the precision ``φ`` / dispersion
    ``σ²``.  Using the fitted per-row values (rather than the number of Z
    coefficients) also catches a one-column user-supplied Z that is not an
    intercept.
    """
    try:
        raw = result.predict(which="secondary")
    except Exception:
        return None, None, False
    values = np.asarray(raw, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None, False
    name = family_name(result) or ""
    label = "φ" if name == "Beta" else "σ²" if name == "Simplex" else "Secondary"
    spread = float(np.max(values) - np.min(values))
    varies = values.size > 1 and spread > 1e-8 * max(1.0, abs(float(np.mean(values))))
    return values, label, varies


def _format_secondary_value(label: str, value: float) -> str:
    """Compact but informative label, e.g. ``φ = 258.01`` or ``σ² = 0.0342``."""
    text = f"{value:.2f}" if abs(value) >= 100.0 else f"{value:.4g}"
    return f"{label} = {text}"


def _has_finite(values) -> bool:
    if values is None:
        return False
    try:
        return bool(np.isfinite(np.asarray(values, dtype=float)).any())
    except (TypeError, ValueError):
        return False


def _continuous_quantile_panel_available(result, data: PlotData, seed: int) -> bool:
    """Whether the quantile-residual panel applies to a continuous GLM.

    All four panels need quantile residuals on the normal-score scale, a
    complete linear predictor and leverage for the influence contour.  Censored
    AFT rows have undefined quantile residuals, so those fits keep their
    dedicated Cox-Snell panel instead of a half-empty residual panel.
    """
    if data.kind != "glm" or not has_randomized_quantile_residuals(result):
        return False
    if not _has_finite(data.leverage):
        return False
    eta = data.linear_predictor
    res = _quantile_residual_values(result, data, seed)
    if eta is None or res is None:
        return False
    eta = np.asarray(eta, dtype=float).reshape(-1)
    res = np.asarray(res, dtype=float).reshape(-1)
    if eta.size != res.size or eta.size < 5:
        return False
    return bool(np.isfinite(eta).all() and np.isfinite(res).all())


def _is_binary_family(data: PlotData, name: str) -> bool:
    fam = (data.extras.get("family") or "").lower()
    if fam:
        return fam.endswith("binomial") and "neg" not in fam
    family = str(getattr(data.result, "family", "") or "").lower()
    return "binomial" in family and "neg" not in family


def _render_mosaic(result: Any, ax, plan: list[tuple[Any, dict]], figsize, label: str) -> PlotResult:
    """Render a four-panel plan, degrading individual panels gracefully."""
    fig, axes = panel_axes(ax, 2, 2, figsize=figsize)
    for panel, (fn, kwargs) in zip(axes.ravel(), plan[:4]):
        try:
            fn(result, ax=panel, **kwargs)
        except Exception as exc:  # pragma: no cover - graceful degradation
            panel.text(0.5, 0.5, f"Panel unavailable\n{exc}", ha="center", va="center", transform=panel.transAxes)
            panel.set_axis_off()
    for panel in axes.ravel()[len(plan):]:
        panel.set_axis_off()
    fig.suptitle(label, fontsize=14)
    return PlotResult(fig, label)


def _mosaic_title(title: str | None, default: str) -> str:
    """Use the short mosaic title even if a caller passes a redundant suffix.

    Older callers sometimes supplied ``"GLMER diagnostics — GLMER"`` (and the
    equivalent for GLS/GNLS).  The acronym is already in the default title, so
    drop the duplicate instead of repeating it in the figure suptitle.
    """
    label = str(title) if title is not None else default
    suffix = f" — {default.split()[0]}"
    if label.endswith(suffix):
        return default
    return label


def gls_diagnostics(result: Any, ax=None, *, figsize=(13.0, 9.5), seed: int = 42, title: str | None = None):
    """Four-check mosaic for GLS (no random effects, no conditional fit)."""
    from .effects import ranef_caterpillar  # noqa: F401 - documented compatibility for mixed fits

    name = type(result).__name__
    if name != "LmeFit" or getattr(result, "kind", None) != "gls":
        raise ValueError("gls_diagnostics requires a fitted GLS result")
    data = resolve(result, seed=seed)
    plan = [
        (observed_vs_fitted, {"loess": True, "include_zero": False,
                              "title": f"Observed vs. fitted — {data.model_name}"}),
        (qq_plot_with_envelope, {"seed": seed, "title": f"Normalised residual Q-Q — {data.model_name}"}),
        (normalized_residuals_and_spread, {"seed": seed, "title": f"Normalised residuals & spread vs. fitted — {data.model_name}"}),
        (grouped_acf_plot, {"seed": seed, "title": f"Normalised residual ACF — {data.model_name}"}),
    ]
    label = _mosaic_title(title, "GLS diagnostics")
    return _render_mosaic(result, ax, plan, figsize, label)


def gnls_diagnostics(result: Any, ax=None, *, figsize=(13.0, 9.5), seed: int = 42, title: str | None = None):
    """Four-check mosaic for GNLS with weights/correlation structures."""
    name = type(result).__name__
    if name != "NlmeFit" or getattr(result, "kind", None) != "gnls":
        raise ValueError("gnls_diagnostics requires a fitted GNLS result")
    data = resolve(result, seed=seed)
    use_time = data.time is not None
    xlabel = str(data.time_name or "primary covariable") if use_time else "fitted"
    plan = [
        (observed_vs_fitted, {"loess": True, "title": f"Observed vs. fitted — {data.model_name}"}),
        (qq_plot_with_envelope, {"seed": seed, "title": f"Q-Q with 95% envelope — {data.model_name}"}),
        (normalized_residuals_and_spread, {"seed": seed, "use_time": use_time,
                                           "title": f"Normalised residuals & spread vs. {xlabel} — {data.model_name}"}),
        (grouped_acf_plot, {"seed": seed, "title": f"Normalised residual ACF — {data.model_name}"}),
    ]
    label = _mosaic_title(title, "GNLS diagnostics")
    return _render_mosaic(result, ax, plan, figsize, label)


def glmer_diagnostics(result: Any, ax=None, *, figsize=(13.0, 9.5), seed: int = 42,
                      n_sim: int = 500, title: str | None = None):
    """Golden-standard four-panel mosaic for discrete GLMER fits.

    Bernoulli, grouped binomial ``k/n``, Poisson and negative-binomial fits use
    simulated quantile residuals for the Q-Q and binned panels.  Continuous
    GLMER families keep the established prediction/residual/BLUP layout.
    """
    from .effects import ranef_caterpillar
    from .predictions import population_vs_conditional

    name = type(result).__name__
    if name != "GlmerFit":
        raise ValueError("glmer_diagnostics requires a fitted GLMER result")
    data = resolve(result, seed=seed)
    family = str(getattr(result, "family", "") or "").lower()
    if family in ("binomial", "poisson", "negbin"):
        plan = [
            (population_vs_conditional, {"title": f"Population vs. conditional — {data.model_name}"}),
            (randomized_quantile_qq, {"seed": seed, "n_sim": n_sim,
                                      "title": f"Simulated quantile Q-Q — {data.model_name}"}),
            (binned_quantile_residuals_by_rank, {"seed": seed, "n_sim": n_sim,
                                                 "title": f"Binned quantile residuals vs. rank(ŷ) — {data.model_name}"}),
            (ranef_caterpillar, {"title": f"Random effects caterpillar — {data.model_name}"}),
        ]
        label = _mosaic_title(title, "GLMER diagnostics")
    else:
        plan = [
            (population_vs_conditional, {"title": f"Population vs. conditional — {data.model_name}"}),
            (qq_plot, {"seed": seed, "title": f"Normal Q-Q — {data.model_name}"}),
            (binned_standardized_residuals, {"seed": seed,
                                             "title": f"Binned standardised residuals — {data.model_name}"}),
            (ranef_caterpillar, {"title": f"Random effects caterpillar — {data.model_name}"}),
        ]
        label = _mosaic_title(title, "GLMER diagnostics")
    return _render_mosaic(result, ax, plan, figsize, label)


def _saem_no_response_panel(
    result: Any,
    ax=None,
    *,
    title: str | None = None,
    figsize=None,
):
    """Placeholder for custom likelihoods with no simulations or residuals."""
    fig, ax, _ = single_axis(ax, figsize)
    ax.text(
        0.5, 0.5,
        "No response-level diagnostics available\n"
        "for this custom SAEM fit (no simulation or residuals).",
        ha="center", va="center", transform=ax.transAxes, fontsize=10,
    )
    ax.set_axis_off()
    label = title or "No response checks — SAEM"
    ax.set_title(label)
    return PlotResult(fig, label)


def saem_diagnostics(
    result: Any,
    ax=None,
    *,
    figsize=(13.0, 9.5),
    seed: int = 42,
    n_sim: int = 400,
    title: str | None = None,
):
    """Four-panel diagnostic mosaic for SAEM fits.

    The panels follow the longitudinal structure of the data:

    1. observed response against population (PRED) and conditional (IPRED)
       predictions;
    2. unit-scale residuals against IPRED for Gaussian fits, or NPDE against
       time for censored / non-Gaussian fits;
    3. a normal Q-Q with a 95% simulation envelope of the *same* residuals;
    4. a within-subject ACF, so lags never cross individual boundaries.

    Censored fits use NPDE everywhere: ``res.data.y`` stores the censoring
    limit rather than the unobserved response, and Pearson/raw residuals are
    NaN at those rows.  A pointless caterpillar is only added when the fit has
    actual numeric random effects and no residual panel is available; custom
    likelihoods fall back to a coefficient view (when Wald intervals exist)
    and an explicit placeholder rather than repeated forest plots.
    """
    from .adapters import coefficient_table
    from .coefficients import forest_plot
    from .effects import dominant_random_effect, ranef_caterpillar
    from .predictions import population_vs_conditional

    if type(result).__name__ != "SaemFit":
        raise ValueError("saem_diagnostics requires a fitted SAEM result")
    data = resolve(result, seed=seed)
    has_predictions = _has_finite(data.fitted) and _has_finite(data.fitted_population)
    has_residuals = _has_finite(data.residuals)
    is_npde = "npde" in str(data.residual_label).lower()
    cens = getattr(getattr(getattr(result, "res", None), "data", None), "cens", None)
    cens = np.asarray(cens, dtype=float).reshape(-1) if cens is not None else np.array([])
    censored = bool(np.any(cens != 0))
    has_ranef = has_random_effects(result)

    plan: list[tuple[Any, dict]] = []

    def add(fn, **kwargs):
        plan.append((fn, kwargs))

    if has_predictions:
        pred_title = "Population vs. conditional — SAEM"
        if censored:
            # The observed points are the censoring limits, not measured y.
            pred_title = "Population vs. conditional (censored limits) — SAEM"
        add(population_vs_conditional, title=pred_title)
    if has_residuals:
        use_time = is_npde and _has_finite(data.time)
        x = "time" if use_time or not _has_finite(data.fitted) else "fitted"
        which = "time" if x == "time" else "IPRED"
        add(residuals_vs_predictions, seed=seed, x=x,
            title=f"{data.residual_label} vs. {which} — SAEM")
        q = "NPDE" if is_npde else data.residual_label
        add(qq_plot_with_envelope, seed=seed, n_sim=n_sim,
            point_label=q, value_label=f"Ordered {q}",
            title="Q-Q with 95% envelope — SAEM")
        add(grouped_acf_plot, seed=seed,
            title=f"{'NPDE' if is_npde else 'Residual'} ACF — SAEM")
    if has_ranef:
        term = dominant_random_effect(result)
        options = {"title": "Random effects caterpillar — SAEM"}
        if term:
            options["term"] = term
        add(ranef_caterpillar, **options)

    # Custom likelihoods without simulation have no response-level checks.
    # Keep at most one coefficient view (only when Wald intervals exist) and
    # use an explicit placeholder instead of padding with duplicate panels.
    can_forest = False
    try:
        can_forest = len(coefficient_table(result)) > 0
    except Exception:
        pass
    if len(plan) < 4 and can_forest and forest_plot not in {fn for fn, _ in plan}:
        add(forest_plot, title="Fixed effects — SAEM")
    if not (has_predictions or has_residuals):
        add(_saem_no_response_panel)

    label = _mosaic_title(title, "SAEM diagnostics")
    return _render_mosaic(result, ax, plan, figsize, label)


def diagnostics(result: Any, ax=None, *, figsize=(13.0, 9.5), seed: int = 42, alpha: float = 0.05, title: str | None = None):
    """A 2×2 panel of *model checks* chosen for the fitted result family.

    Coefficient forest plots are deliberately not part of this quick view:
    every panel diagnoses the fit (residual behaviour or a family-specific
    check).  Discrete families never receive a normal Q-Q of raw residuals;
    they get their natural check instead.  Count GLMs use a DHARMa-style Q-Q,
    a hanging rootogram, quantile residuals vs the prediction and residuals
    vs leverage (no ACF: a GLM assumes independent observations).  Continuous
    GLMs with an exact distribution use a scale-location plot of quantile
    residuals against η, their Q-Q with a 95% envelope, quantile residuals vs
    η and residuals vs leverage.  Binary GLMs use the same shape: a Q-Q with a
    95% envelope, a calibration curve, quantile residuals vs η and leverage.
    Multinomial GLMs use category discrimination (boxplots of the probability
    assigned to the observed class), binned residuals over all ``N × K`` cells
    ``y_ik - p̂_ik``, one-vs-rest calibration curves and influence; the old
    deviance-vs-μ panel is intentionally replaced by calibration.  Tweedie
    GLMs use randomised quantile residuals as well, including the
    compound-Poisson mass at zero: scale-location vs η, Q-Q, residuals vs η and
    influence.  Beta/simplex fits keep observed-vs-fitted, a quantile-residual
    Q-Q (annotated with the fixed precision/dispersion when applicable) and
    quantile residuals vs η; the fourth panel is Cook's distance vs leverage
    for constant φ, or φᵢ/σ²ᵢ vs η when the dispersion is modelled.  Censored
    survival fits (AFT and GeneralizedGamma) use a censoring-aware observed-vs-
    fitted, Cox-Snell with a Greenwood band, and deviance-martingale residuals
    vs η; AFT fits add influence, while GeneralizedGamma keeps its submodels.
    Quasi-likelihood fits substitute Pearson-based checks.  SAEM delegates to
    :func:`saem_diagnostics`: prediction at both levels, Pearson/IPRED or
    censoring-aware NPDE, a Q-Q with a 95% envelope and a within-subject ACF.
    Continuous truncated GLM fits delegate to the dedicated
    :func:`truncated_diagnostics` mosaic; count-truncated fits keep the
    simpler `truncated` panel and the generic checks below.  LME mosaics show
    population vs. conditional predictions, normalised residuals against the
    conditional fit with a LOESS band, a Q-Q with a 95% envelope and a BLUP
    caterpillar with conditional intervals.  NLME replaces the fitted-value
    trend with residuals against the structural variable (time/dose), because
    opposite phases of a nonlinear curve can share the same fitted value, and
    selects the dominant random-effect term automatically.  GLS and GNLS
    delegate to :func:`gls_diagnostics` and :func:`gnls_diagnostics`, which use
    observed-vs-fitted, a Q-Q envelope, normalised residuals with a separate
    spread scale and a within-group ACF.  GLMER delegates to
    :func:`glmer_diagnostics`: discrete families use simulated quantile
    residuals and rank-binned checks, continuous families keep the prediction,
    normal Q-Q and BLUP layout.
    """
    from .classification import calibration_curve, multiclass_calibration_curve, roc_curve
    from .coefficients import covariance_parameters
    from .effects import dominant_random_effect, gengamma_submodels, ranef_caterpillar
    from .influence import cooks_distance_plot, influence_plot, residuals_vs_leverage
    from .predictions import (
        category_probabilities,
        fitted_curve,
        ordinal_curves,
        population_vs_conditional,
    )

    name = type(result).__name__
    fit_kind = getattr(result, "kind", None)
    if name == "LmeFit" and fit_kind == "gls":
        return gls_diagnostics(result, ax=ax, figsize=figsize, seed=seed, title=title)
    if name == "NlmeFit" and fit_kind == "gnls":
        return gnls_diagnostics(result, ax=ax, figsize=figsize, seed=seed, title=title)
    if name == "GlmerFit":
        return glmer_diagnostics(result, ax=ax, figsize=figsize, seed=seed, title=title)
    if name == "SaemFit":
        return saem_diagnostics(result, ax=ax, figsize=figsize, seed=seed, title=title)
    if _is_continuous_truncated_glm(result):
        # The generic quartet's observed-vs-fitted, leverage and raw residual
        # panels hide the edge behaviour that matters under truncation.  The
        # dedicated mosaic is therefore the default for this family.
        return truncated_diagnostics(result, ax=ax, seed=seed, figsize=figsize, title=title)

    if name == "ListFit":
        # A list fit stores per-group coefficient tables but no response-level
        # data, so the coefficient summary is its only meaningful overview.
        from .coefficients import list_coefficients

        return list_coefficients(result, ax=ax, alpha=alpha, figsize=figsize, title=title)

    data = resolve(result, seed=seed)
    fam = (data.extras.get("family") or "").lower()
    is_quasi = data.kind == "glm" and _is_quasi_family(result)
    is_count = data.kind == "glm" and supports_rootogram(result)
    is_binary = _is_binary_family(data, name)
    is_tweedie = data.kind == "glm" and (
        fam == "tweedie" or type(getattr(result, "family", None)).__name__ == "Tweedie"
    )
    is_beta_simplex = data.kind == "glm" and type(getattr(result, "family", None)).__name__ in (
        "Beta",
        "Simplex",
    )
    is_ordinal = "ordinal" in fam
    is_categorical = data.kind == "glm" and ("multinomial" in fam or is_ordinal)

    plan: list[tuple[Any, dict]] = []

    def add(fn, **kwargs):
        plan.append((fn, kwargs))

    if data.kind == "glm":
        if is_quasi:
            # Quasi-likelihood: no exact CDF, so the Q-Q/rootogram and the
            # quantile-residual scatter become Pearson/deviance checks on η.
            # The individual scatter uses *deviance* residuals: with binary
            # data a Pearson residual explodes as p̂ → 0 (|r| ≫ 30), squashing
            # the ±2 lines and the bulk of the points against zero.  Deviance
            # residuals stay on a readable ±4 scale, as in the leverage panel.
            add(pearson_scale_location, seed=seed, use_linear_predictor=True)
            add(residuals_vs_fitted, seed=seed, use_linear_predictor=True)
            add(binned_residuals, seed=seed)
            # Pearson residuals also feed the influence panel.  Independence
            # is assumed, so no ACF; the same Cook contours as every other
            # GLM panel keep the influence check comparable.
            add(residuals_vs_leverage, seed=seed)
        elif is_count:
            # Count GLM standard panel: residual distribution, count fit,
            # residuals against the prediction and influence.  Observed-vs-
            # fitted makes rigid horizontal bands for counts, so use
            # randomised quantile residuals whenever an exact CDF exists.
            # Independent observations make an ACF panel unnecessary, and a
            # quantile-residual scatter supersedes binned residuals.
            if has_randomized_quantile_residuals(result):
                add(randomized_quantile_qq, seed=seed)
                add(rootogram)
                add(quantile_residuals_vs_fitted, seed=seed)
            else:
                add(observed_vs_fitted)
                add(rootogram)
                add(binned_residuals, seed=seed)
            add(residuals_vs_leverage, seed=seed)
        elif is_binary:
            # Binary GLM standard panel, mirroring the count/continuous ones:
            # residual distribution (Q-Q with 95% band), calibration as the
            # family-specific fit check, quantile residuals on the natural η
            # scale, and influence.  ROC/PR remain available as separate kinds.
            if has_randomized_quantile_residuals(result):
                add(randomized_quantile_qq, seed=seed)
                add(calibration_curve)
                add(quantile_residuals_vs_fitted, seed=seed, use_linear_predictor=True)
            else:
                add(calibration_curve)
                add(roc_curve)
                add(binned_residuals, seed=seed)
            add(residuals_vs_leverage, seed=seed)
        elif is_tweedie and _continuous_quantile_panel_available(result, data, seed):
            # Tweedie standard panel.  The CDF is computable (compound
            # Poisson-gamma for 1 < p < 2, Gamma at p = 2), so randomised
            # quantile residuals remove the artificial break at zero and the
            # four panels match the continuous GLM layout.  Scale-location on
            # η makes a wrong variance exponent p visible as a trend.
            add(quantile_scale_location, seed=seed)
            add(randomized_quantile_qq, seed=seed)
            add(quantile_residuals_vs_fitted, seed=seed, use_linear_predictor=True)
            add(residuals_vs_leverage, seed=seed)
        elif is_categorical:
            if is_ordinal:
                add(ordinal_curves)
                add(binned_residuals, seed=seed)
                add(residuals_vs_fitted, seed=seed)
            else:
                # Multinomial quartet: discrimination, N×K cell residuals,
                # one-vs-rest calibration and influence.  The deviance-vs-μ
                # panel is deliberately replaced by the reliability curves.
                add(category_probabilities)
                add(multiclass_binned_residuals)
                add(multiclass_calibration_curve)
            add(influence_plot)
        elif data.extras.get("has_cox_snell"):
            # Censored-survival panel.  Cox-Snell is the primary check; the
            # residual-vs-η panel uses deviance-martingale residuals so that
            # censored rows are not silently dropped.  GeneralizedGamma keeps
            # its submodel panel as the fourth view instead of influence.
            add(_survival_observed_vs_fitted)
            add(cox_snell)
            add(deviance_martingale_vs_eta)
            if "generalizedgamma" in fam:
                add(gengamma_submodels)
            else:
                add(influence_plot)
        elif is_beta_simplex:
            # Beta/simplex standard panel.  Both have an exact continuous CDF,
            # so the Q-Q and residual-vs-η panels use quantile residuals.  With
            # constant φ/σ² the fourth panel is the usual influence check; the
            # fitted value is annotated on the Q-Q instead of using a panel.
            # When the secondary parameter is modelled through Z, plotting the
            # row-wise φᵢ/σ²ᵢ against η shows what that submodel contributes.
            secondary_values, secondary_label, secondary_varies = _secondary_parameter_info(result)
            note = (
                _format_secondary_value(secondary_label, float(np.mean(secondary_values)))
                if secondary_values is not None and not secondary_varies
                else None
            )
            add(observed_vs_fitted)
            add(qq_plot, seed=seed, note=note)
            add(quantile_residuals_vs_fitted, seed=seed, use_linear_predictor=True)
            if secondary_varies:
                add(secondary_vs_eta)
            else:
                add(influence_plot, title=f"Cook's distance vs. leverage — {data.model_name}")
        elif any(word in fam for word in ("beta", "simplex", "hurdlegamma", "hurdlelognormal", "hurdlebeta")):
            # Hurdle families keep their secondary (hurdle probability)
            # distribution panel; only plain Beta/simplex use the quartet above.
            # Their positive component is heavily skewed, so use symlog axes.
            add(observed_vs_fitted, symmetric_log=True)
            add(qq_plot, seed=seed)
            add(residuals_vs_fitted, seed=seed)
            add(secondary_distribution)
        elif "truncated" in fam:
            add(observed_vs_fitted)
            add(truncated_fit)
            add(residuals_vs_fitted, seed=seed)
            add(qq_plot, seed=seed)
        elif _continuous_quantile_panel_available(result, data, seed):
            # Standard continuous GLM panel.  Scale-location replaces the
            # observed-vs-fitted scatter (which only reflects the link and
            # the outcome's scale); the two residual plots use η, the natural
            # predictor scale, and all panels share the same residual seed.
            add(quantile_scale_location, seed=seed)
            add(randomized_quantile_qq, seed=seed)
            add(quantile_residuals_vs_fitted, seed=seed, use_linear_predictor=True)
            add(residuals_vs_leverage, seed=seed)
        else:
            # No exact CDF / η / leverage (e.g. Tweedie, Pareto, custom): keep
            # the generic Pearson/deviance-based panel.
            add(observed_vs_fitted)
            add(qq_plot, seed=seed)
            add(residuals_vs_fitted, seed=seed)
            add(influence_plot)
    elif name in ("nls", "NLS"):
        if len(getattr(result, "_exogenous", None) or []) == 1:
            add(fitted_curve)
        else:
            add(observed_vs_fitted)
        add(qq_plot, seed=seed)
        add(residuals_vs_fitted, seed=seed)
        add(cooks_distance_plot)
    elif name == "MCGLMResults":
        # Responses have very different scales: every residual panel uses
        # within-response quantiles and per-response colours so that a large
        # response cannot hide the smaller ones.  Covariance parameters are
        # split into correlations and dispersions below.
        add(standardized_residual_qq, seed=seed)
        add(binned_residuals_by_response, seed=seed)
        add(standardized_residuals_vs_fitted, seed=seed)
        add(covariance_parameters)
    elif name in ("LmeFit", "NlmeFit"):
        # LME/NLME mosaics cover both levels of the hierarchy: population vs.
        # conditional calibration, a level-1 residual check with a smooth
        # trend and Q-Q envelope, and a level-2 BLUP caterpillar with posterior
        # intervals.  The old per-subject bins hid small-sample behaviour.
        add(population_vs_conditional)
        if name == "NlmeFit":
            add(qq_plot_with_envelope, seed=seed)
            add(normalized_residuals_vs_time, seed=seed)
            if has_random_effects(result):
                term = dominant_random_effect(result)
                add(ranef_caterpillar, **({"term": term} if term else {}))
            else:
                add(acf_plot, seed=seed)
        else:
            add(normalized_residuals_vs_fitted, seed=seed)
            add(qq_plot_with_envelope, seed=seed)
            add(ranef_caterpillar if has_random_effects(result) else acf_plot)
    elif name in ("LmerFit", "GlmerFit"):
        from .coefficients import forest_plot

        add(population_vs_conditional)
        if is_binary:
            add(calibration_curve)
        else:
            add(qq_plot, seed=seed)
        add(binned_standardized_residuals, seed=seed)
        add(ranef_caterpillar if has_random_effects(result) else acf_plot)
    elif name == "SaemFit":
        from .coefficients import forest_plot

        # Population vs. conditional already carries the fitted values, so the
        # first slot checks residual behaviour instead: Pearson vs. IPRED for
        # Gaussian fits, NPDE vs. time for discrete/censored families (which
        # have no residual-error parameters).
        if _has_finite(data.residuals):
            add(residuals_vs_predictions, seed=seed)
            add(qq_plot, seed=seed)
        if data.groups is not None:
            add(population_vs_conditional)
        if has_random_effects(result):
            add(ranef_caterpillar)
        # Custom likelihoods and hierarchical fits have no residuals, fitted
        # values or simulation; the coefficient view is the useful fallback.
        while len(plan) < 4:
            add(forest_plot)
    else:  # generic fitted result
        add(observed_vs_fitted)
        add(qq_plot, seed=seed)
        add(residuals_vs_fitted, seed=seed)
        add(influence_plot)

    label = title or f"Diagnostics — {data.model_name}"
    return _render_mosaic(result, ax, plan, figsize, label)
