# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Leverage, Cook's distance and other influence diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import resolve
from .foundation import (
    DEFAULT_COLORS,
    PlotResult,
    annotate_points,
    clean,
    colorbar,
    single_axis,
    style_axis,
)

__all__ = [
    "influence_plot",
    "residuals_vs_leverage",
    "cooks_distance_plot",
    "dffits_plot",
    "dfbetas_heatmap",
    "leverage_plot",
]


def _n_params(model: Any) -> int:
    """Number of fitted coefficients, used by leverage/Cook reference lines."""
    for attribute in ("param_names", "exog_names", "params", "coefficients", "beta"):
        value = getattr(model, attribute, None)
        if value is None:
            continue
        try:
            count = len(value)
        except TypeError:
            continue
        if count:
            return count
    return 1


def _influence_arrays(result: Any):
    data = resolve(result)
    model = data.result
    leverage = data.leverage
    cooks = data.cooks
    if leverage is None and hasattr(model, "hat_matrix_diag"):
        leverage = getattr(model, "hat_matrix_diag")
    if cooks is None:
        value = getattr(model, "cooks_distance", None)
        cooks = value() if callable(value) else value
    leverage = None if leverage is None else np.asarray(leverage, dtype=float).reshape(-1)
    cooks = None if cooks is None else np.asarray(cooks, dtype=float).reshape(-1)
    return data, leverage, cooks


def influence_plot(result: Any, ax=None, *, top_n: int = 3, figsize=(5.5, 4.5), title: str | None = None):
    """Leverage against Cook's distance, annotating the most influential rows."""
    data, leverage, cooks = _influence_arrays(result)
    if leverage is None or cooks is None:
        raise ValueError(f"influence diagnostics are not available for {data.model_name}")
    fig, ax, _ = single_axis(ax, figsize)
    lev, ck = clean(leverage, cooks)
    ax.scatter(lev, ck, alpha=0.6, color="#7c3aed", edgecolors="none")
    n = ck.size
    threshold = 4.0 / max(n, 1)
    ax.axhline(threshold, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1, label=f"Cook's D = 4/n ({threshold:.3g})")
    # Fix the axes before annotating: the rendered-label collision check uses
    # display pixels, and a margin keeps labels near the largest Cook value or
    # leverage from being clipped at the panel edge.
    finite = np.isfinite(lev) & np.isfinite(ck)
    if finite.any():
        xlo, xhi = float(np.min(lev[finite])), float(np.max(lev[finite]))
        ylo, yhi = float(np.min(ck[finite])), max(float(np.max(ck[finite])), threshold)
        xpad = max((xhi - xlo) * 0.10, abs(xhi) * 0.06, 1e-6)
        ypad = max((yhi - ylo) * 0.10, abs(yhi) * 0.06, 1e-6)
        ax.set_xlim(min(0.0, xlo - xpad), xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
    if n:
        order = np.argsort(ck)[-max(int(top_n), 1) :]
        annotate_points(ax, lev[order], ck[order], [f"Obs {i}" for i in order], max_labels=top_n)
    ax.set_xlabel("Leverage (h_ii)")
    ax.set_ylabel("Cook's distance")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    panel_title = title or f"Influence — {data.model_name}"
    ax.set_title(panel_title)
    return PlotResult(fig, panel_title)


def residuals_vs_leverage(
    result: Any,
    ax=None,
    *,
    seed: int = 42,
    top_n: int = 3,
    figsize=(5.5, 4.5),
    title: str | None = None,
):
    """Residuals against leverage with Cook's-distance contours.

    This is the classic influence plot: it separates a large residual (poor
    fit) from a large hat value (unusual predictor values), and the contours
    show which points actually move the fit.  Discrete GLMs use randomised
    quantile residuals when available: they are already on a normal-score
    scale, so the contours use the usual ``D = r²h / (p(1-h))`` approximation.
    Contours include the classical ``4/n`` cutoff, which stays visible even
    when hat values are too small for the 0.5/1 guidelines to appear.

    Parameters
    ----------
    result : fitted result
        Needs residual and leverage (hat diagonal) information.
    seed : int, default 42
        Seed for residual randomisation; match the Q-Q panel's seed so both
        panels refer to the same residuals.
    top_n : int, default 3
        Number of largest Cook's-distance points to annotate.
    """
    data = resolve(result, seed=seed)
    leverage = data.leverage
    if leverage is None:
        leverage = getattr(data.result, "hat_matrix_diag", None)
    if leverage is None:
        raise ValueError(f"leverage values are not available for {data.model_name}")

    # Prefer quantile residuals: they are N(0, 1) under a correct model, which
    # is exactly the scale the Cook contours assume.  Fall back to whatever
    # residual the adapter selected for models without an exact CDF.
    label = data.residual_label
    values = data.residuals
    rq = getattr(data.result, "resid_quantile", None)
    if callable(rq):
        try:
            raw = rq(seed=seed)
        except TypeError:
            raw = rq()
        if raw is not None:
            candidate = np.asarray(raw, dtype=float).reshape(-1)
            if candidate.size == np.asarray(leverage).reshape(-1).size and np.isfinite(candidate).any():
                values, label = candidate, "Quantile residuals"
    if values is None:
        raise ValueError(f"residuals are not available for {data.model_name}")

    lev, res = clean(leverage, values)
    if lev.size < 3:
        raise ValueError("not enough finite leverage/residual pairs for this plot")

    p = _n_params(data.result)
    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(lev, res, s=18, alpha=0.6, color="#7c3aed", edgecolors="none", label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.1)

    span = max(float(np.nanmax(np.abs(res))) if res.size else 1.0, 1.0)
    ylim = (-1.25 * span, 1.25 * span)
    ax.set_ylim(*ylim)

    # MLE count families (e.g. zero-inflated) can yield h outside [0, 1];
    # keep those points visible instead of clipping them.  Set the limits
    # before annotating so the pixel-space label de-duplication sees them.
    upper = float(np.nanmax(lev))
    lower = float(np.nanmin(lev))
    ax.set_xlim(min(lower * 1.05, 0.0) if lower < 0 else 0.0, max(upper * 1.05, 0.01))

    # D = r² h / (p (1-h))  ->  r = ± sqrt(D p (1-h) / h).  The classical
    # 4/n cutoff goes first: with the tiny hat values of large samples the
    # 0.5/1 contours fall outside the residual range, so 4/n is the one that
    # is actually visible.
    positive = lev[lev > 0]
    lo = float(np.nanmin(positive)) if positive.size else max(upper / 2.0, 1e-6)
    hi = min(max(upper * 1.05, lo * 2.0), 0.999)
    reference = 4.0 / max(lev.size, 1)
    levels = [reference] + [lv for lv in (0.5, 1.0) if lv > reference * 1.5]
    if hi > lo:
        grid = np.linspace(lo, hi, 100)
        for level in levels:
            curve = np.sqrt(level * p * (1.0 - grid) / np.maximum(grid, 1e-12))
            # Do not advertise a contour that lies entirely off the y-axis.
            if not np.any((curve >= ylim[0]) & (curve <= ylim[1])):
                continue
            is_reference = level == reference
            style = dict(
                color=DEFAULT_COLORS["accent"] if is_reference else DEFAULT_COLORS["neutral"],
                linestyle="--" if is_reference else ":",
                linewidth=1.2 if is_reference else 1.0,
            )
            # Four significant digits keep 4/n readable (0.00102) without
            # exposing floating-point noise such as 0.00102015.
            shown = f"{level:.4g}"
            contour_label = f"Cook's D = {shown} (4/n)" if is_reference else f"Cook's D = {shown}"
            ax.plot(grid, curve, label=contour_label, **style)
            ax.plot(grid, -curve, **style)

    cook = np.where(lev < 1.0, res**2 * lev / (p * np.maximum(1.0 - lev, 1e-12)), np.nan)
    finite = np.isfinite(cook)
    if finite.any():
        order = np.argsort(np.where(finite, cook, -np.inf))[-max(int(top_n), 1):]
        annotate_points(ax, lev[order], res[order], [f"Obs {i}" for i in order], max_labels=top_n, stagger=True)

    ax.set_xlabel("Leverage (h_ii)")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    panel_title = title or f"Residuals vs. leverage — {data.model_name}"
    ax.set_title(panel_title)
    return PlotResult(fig, panel_title)


def cooks_distance_plot(result: Any, ax=None, *, top_n: int = 10, figsize=(7.0, 4.2), title: str | None = None):
    """Cook's distance by observation index, highlighting the largest values."""
    data, _, cooks = _influence_arrays(result)
    if cooks is None:
        raise ValueError(f"Cook's distance is not available for {data.model_name}")
    fig, ax, _ = single_axis(ax, figsize)
    idx = np.arange(cooks.size)
    ax.vlines(idx, 0.0, cooks, color=DEFAULT_COLORS["neutral"], linewidth=0.9, alpha=0.65)
    ax.scatter(idx, cooks, s=16, color="#7c3aed", zorder=3)
    threshold = 4.0 / max(cooks.size, 1)
    ax.axhline(threshold, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1, label=f"4/n = {threshold:.3g}")
    finite = np.isfinite(cooks)
    if finite.any():
        order = np.argsort(np.where(finite, cooks, -np.inf))[-max(int(top_n), 1) :]
        annotate_points(ax, idx[order], cooks[order], [f"{i}" for i in order], max_labels=top_n)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Cook's distance")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return PlotResult(fig, title or f"Cook's distance — {data.model_name}")


def dffits_plot(result: Any, ax=None, *, top_n: int = 8, figsize=(7.0, 4.2), title: str | None = None):
    """DFFITS by observation, with the conventional ``2√(p/n)`` threshold."""
    data = resolve(result)
    model = data.result
    dffits = getattr(model, "dffits", None)
    if callable(dffits):
        dffits = dffits()
    if dffits is None and data.leverage is not None and data.residuals is not None:
        # Standardised-deletion approximation from the residuals and the hat
        # diagonal, valid for NLS and other fits without a native DFFITS.
        h = np.asarray(data.leverage, dtype=float).reshape(-1)
        e = np.asarray(data.residuals, dtype=float).reshape(-1)
        sigma = getattr(model, "sigma", None)
        if sigma is None:
            sigma = np.nanstd(e, ddof=2) or 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            r_std = e / (float(sigma) * np.sqrt(np.maximum(1.0 - h, 1e-12)))
            dffits = r_std * np.sqrt(np.maximum(h, 0.0) / np.maximum(1.0 - h, 1e-12))
        dffits = np.where(np.isfinite(dffits), dffits, np.nan)
    if dffits is None:
        raise ValueError(f"DFFITS is not available for {data.model_name}")
    values = np.asarray(dffits, dtype=float).reshape(-1)
    fig, ax, _ = single_axis(ax, figsize)
    idx = np.arange(values.size)
    threshold = 2.0 * np.sqrt(max(len(getattr(model, "param_names", [])) or 1, 1) / max(values.size, 1))
    ax.vlines(idx, 0.0, values, color=DEFAULT_COLORS["neutral"], linewidth=0.9, alpha=0.65)
    ax.scatter(idx, values, s=16, color="#0f766e", zorder=3)
    ax.axhline(threshold, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1, label=f"+2√(p/n) = {threshold:.3g}")
    ax.axhline(-threshold, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1)
    finite = np.isfinite(values)
    if finite.any():
        order = np.argsort(np.abs(np.where(finite, values, 0.0)))[-max(int(top_n), 1) :]
        annotate_points(ax, idx[order], values[order], [f"{i}" for i in order], max_labels=top_n)
    ax.set_xlabel("Observation")
    ax.set_ylabel("DFFITS")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return PlotResult(fig, title or f"DFFITS — {data.model_name}")


def dfbetas_heatmap(result: Any, ax=None, *, max_rows: int = 120, figsize=(8.0, 6.0), cmap: str = "RdBu_r", title: str | None = None):
    """Heatmap of DFBETAS / standardised leave-one-out coefficient changes.

    Large absolute values identify observations that move a particular
    coefficient.  When the number of observations exceeds ``max_rows``, the
    rows with the largest maximum absolute change are shown.
    """
    data = resolve(result)
    model = data.result
    values = getattr(model, "dfbetas", None)
    if callable(values):
        values = values()
    names = None
    if values is None:
        values = getattr(model, "dfbeta_all", None)
        if callable(values):
            values = values()
        names = getattr(model, "all_param_names", None)
        if values is not None:
            values = np.asarray(values, dtype=float)
            scale = np.nanstd(values, axis=0)
            values = values / np.where(scale > 0, scale, 1.0)
    if values is None:
        raise ValueError(f"DFBETAS are not available for {data.model_name}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("DFBETAS must be a two-dimensional array")
    if names is None:
        names = getattr(model, "param_names", None)
    if names is None or len(names) != values.shape[1]:
        names = [f"β{j}" for j in range(values.shape[1])]
    if values.shape[0] > max_rows:
        score = np.nanmax(np.abs(values), axis=1)
        order = np.argsort(np.where(np.isfinite(score), score, -np.inf))[-max_rows:]
        order = np.sort(order)
        values = values[order]
        row_labels = [str(i) for i in order]
    else:
        row_labels = [str(i) for i in range(values.shape[0])]
    fig, ax, _ = single_axis(ax, figsize)
    vmax = float(np.nanpercentile(np.abs(values), 99)) if np.isfinite(values).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(n) for n in names], rotation=45, ha="right")
    if values.shape[0] <= 40:
        ax.set_yticks(np.arange(values.shape[0]))
        ax.set_yticklabels(row_labels, fontsize=7.5)
    else:
        ax.set_yticks([])
    ax.set_ylabel("Observation")
    colorbar(fig, im, ax, label="Standardised change")
    style_axis(ax, grid=False)
    return PlotResult(fig, title or f"DFBETAS — {data.model_name}")


def leverage_plot(
    result: Any,
    ax=None,
    *,
    top_n: int = 8,
    figsize=(7.0, 4.2),
    title: str | None = None,
):
    """Leverage (hat diagonal) by observation with a ``2p/n`` reference line."""
    data, leverage, cooks = _influence_arrays(result)
    if leverage is None:
        raise ValueError(f"leverage values are not available for {data.model_name}")
    n = leverage.size
    threshold = 2.0 * _n_params(data.result) / max(n, 1)
    fig, ax, _ = single_axis(ax, figsize)
    idx = np.arange(n)
    ax.vlines(idx, 0.0, leverage, color=DEFAULT_COLORS["neutral"], linewidth=0.9, alpha=0.65)
    ax.scatter(idx, leverage, s=16, color="#0e7490", zorder=3)
    ax.axhline(threshold, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1, label=f"2p/n = {threshold:.3g}")
    finite = np.isfinite(leverage)
    if finite.any():
        order = np.argsort(np.where(finite, leverage, -np.inf))[-max(int(top_n), 1):]
        annotate_points(ax, idx[order], leverage[order], [f"{i}" for i in order], max_labels=top_n)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Leverage (h_ii)")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    return PlotResult(fig, title or f"Leverage — {data.model_name}")
