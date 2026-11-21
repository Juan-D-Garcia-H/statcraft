# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Predictions: population vs. conditional fits, individual trajectories,
goodness-of-fit, NPDE and categorical response curves."""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import resolve
from .foundation import (
    DEFAULT_COLORS,
    PlotResult,
    clean,
    panel_axes,
    qq_points,
    single_axis,
    style_axis,
)

__all__ = [
    "population_vs_conditional",
    "individual_fits",
    "gof",
    "npde_panel",
    "category_probabilities",
    "ordinal_curves",
    "fitted_curve",
]


def _need(values, what: str, data):
    if values is None:
        raise ValueError(f"{what} are not available for {data.model_name}.")
    return np.asarray(values, dtype=float).reshape(-1)


def _identity_line(ax, *arrays, color=DEFAULT_COLORS["accent"], label="Identity"):
    finite = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    finite = np.concatenate([a[np.isfinite(a)] for a in finite if a.size])
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    ax.plot([lo, hi], [lo, hi], color=color, linestyle="--", linewidth=1.5, label=label)


def population_vs_conditional(result: Any, ax=None, *, figsize=(6.0, 5.0), title: str | None = None):
    """Observed response against the population and conditional predictions."""
    data = resolve(result)
    y = _need(data.response, "Observed values", data)
    conditional = _need(data.fitted, "Conditional predictions", data)
    population = data.fitted_population if data.fitted_population is not None else conditional
    population = _need(population, "Population predictions", data)
    y_c, population_c = clean(y, population)
    y_c2, conditional_c = clean(y, conditional)

    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(population_c, y_c, alpha=0.38, color="#3b82f6", edgecolors="none", label="Population (fixed effects)")
    ax.scatter(conditional_c, y_c2, alpha=0.48, color="#0f766e", edgecolors="none", label="Conditional (fixed + random)")
    _identity_line(ax, y, population, conditional)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Observed response")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    label = title or f"Population vs. conditional — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def individual_fits(
    result: Any,
    ax=None,
    *,
    subject_ids=None,
    n_subjects: int = 6,
    cols: int = 3,
    figsize=None,
    title: str | None = None,
):
    """Per-subject observed, individual (IPRED) and population (PRED) curves."""
    data = resolve(result)
    y = _need(data.response, "Observed values", data)
    conditional = _need(data.fitted, "Individual predictions", data)
    population = data.fitted_population if data.fitted_population is not None else conditional
    population = _need(population, "Population predictions", data)
    if data.groups is None:
        raise ValueError(f"{data.model_name} does not expose a grouping variable")
    groups = np.asarray(data.groups)
    if groups.size != y.size:
        raise ValueError("the grouping variable does not match the observations")
    time = data.time if data.time is not None and np.asarray(data.time).size == y.size else np.arange(y.size, dtype=float)
    time = np.asarray(time, dtype=float)

    unique = np.unique(groups)
    chosen = unique[: min(int(n_subjects), unique.size)] if subject_ids is None else list(subject_ids)
    if len(chosen) == 0:
        raise ValueError("no subjects selected")
    rows = int(np.ceil(len(chosen) / max(cols, 1)))
    if figsize is None:
        figsize = (4.4 * cols, 3.2 * rows)
    fig, axes = panel_axes(ax, rows, cols, figsize=figsize)
    axes = axes.ravel()

    for pos, sid in enumerate(chosen):
        panel = axes[pos]
        mask = groups == sid
        order = np.argsort(time[mask])
        t = time[mask][order]
        panel.scatter(t, y[mask][order], color="black", s=18, zorder=4, label="Observed" if pos == 0 else None)
        panel.plot(t, conditional[mask][order], color="#0f766e", linewidth=2.0, label="Individual" if pos == 0 else None)
        panel.plot(t, population[mask][order], color="#d97706", linestyle="--", linewidth=1.5, label="Population" if pos == 0 else None)
        panel.set_title(f"Subject: {sid}", fontsize=9.5)
        panel.set_xlabel(data.time_name or "Time", fontsize=8.5)
        panel.set_ylabel(data.response_name, fontsize=8.5)
        style_axis(panel)
        if pos == 0:
            panel.legend(loc="best", fontsize=7.5)

    for pos in range(len(chosen), rows * cols):
        axes[pos].set_axis_off()

    label = title or f"Individual fits — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def gof(result: Any, ax=None, *, log_scale: bool = False, figsize=(12.0, 5.0), title: str | None = None):
    """Goodness of fit: observed vs. population and individual predictions.

    When the model has no random effects (population and conditional predictions
    coincide), a single panel is produced.
    """
    data = resolve(result)
    y = _need(data.response, "Observed values", data)
    conditional = _need(data.fitted, "Fitted values", data)
    population = data.fitted_population if data.fitted_population is not None else conditional
    population = _need(population, "Population predictions", data)

    same = population.size == conditional.size and np.allclose(population, conditional, equal_nan=True)
    panels = 1 if same else 2
    if panels == 1:
        fig, ax0, _ = single_axis(ax, (6.2, 5.0))
        axes = [ax0]
    else:
        fig, axes_array = panel_axes(ax, 1, 2, figsize=figsize)
        axes = list(axes_array.ravel())

    configs = [(population, "Population (PRED)", "#2563eb")]
    if panels == 2:
        configs.append((conditional, "Individual (IPRED)", "#0f766e"))

    for panel, (pred, label, color) in zip(axes, configs):
        keep = np.isfinite(y) & np.isfinite(pred)
        if log_scale:
            keep &= (y > 0) & (pred > 0)
        panel.scatter(pred[keep], y[keep], alpha=0.55, color=color, edgecolors="none")
        _identity_line(panel, y[keep], pred[keep], label="Identity (y = x)")
        if log_scale:
            panel.set_xscale("log")
            panel.set_yscale("log")
        panel.set_xlabel(label)
        panel.set_ylabel(f"Observed {data.response_name}")
        style_axis(panel)
        panel.legend(loc="best", fontsize=8.5)

    label = title or f"Goodness of fit — {data.model_name}"
    if panels == 1:
        axes[0].set_title(label)
    else:
        fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def npde_panel(result: Any, ax=None, *, nsim: int = 500, seed: int = 42, figsize=(12.0, 4.6), title: str | None = None):
    """NPDE diagnostics: values against time and a normal Q-Q plot."""
    data = resolve(result)
    model = data.result
    npde_fn = getattr(model, "npde", None)
    if not callable(npde_fn):
        raise ValueError(f"NPDE are not available for {data.model_name}")
    try:
        values = npde_fn(nsim=nsim, seed=seed)
    except TypeError:
        values = npde_fn()
    if hasattr(values, "columns") and "npde" in values.columns:
        values = values["npde"]
    values = np.asarray(values, dtype=float).reshape(-1)

    time = data.time if data.time is not None else np.arange(values.size, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size != values.size:
        time = np.arange(values.size, dtype=float)
    keep = np.isfinite(values) & np.isfinite(time)

    fig, axes = panel_axes(ax, 1, 2, figsize=figsize)
    left, right = axes.ravel()
    left.scatter(time[keep], values[keep], alpha=0.55, color="#b45309", edgecolors="none")
    left.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    left.axhline(1.96, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0, label="±1.96")
    left.axhline(-1.96, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    left.set_xlabel(data.time_name or "Time")
    left.set_ylabel("NPDE")
    left.legend(loc="best", fontsize=8.5)
    style_axis(left)

    osm, osr = qq_points(values[keep])
    if osm.size:
        right.scatter(osm, osr, alpha=0.6, color="#7c3aed", edgecolors="none")
        slope, intercept = np.polyfit(osm, osr, 1)
        right.plot(osm, slope * osm + intercept, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.5)
    right.set_xlabel("Theoretical N(0, 1) quantiles")
    right.set_ylabel("Ordered NPDE")
    style_axis(right)

    label = title or f"NPDE — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def _category_names(model, n_cats: int) -> list[str]:
    cats = getattr(model, "_y_categories", None)
    if cats is not None and len(cats) == n_cats:
        return [str(c) for c in cats]
    return [f"Category {k}" for k in range(n_cats)]


def _observed_codes(model) -> np.ndarray:
    raw = getattr(model, "endog", None)
    arr = np.asarray(raw)
    if arr.ndim == 2:
        return np.argmax(arr, axis=1)
    return np.rint(arr.astype(float)).astype(int)


def _multiclass_arrays(result):
    """Return ``(P, Y, names, data)`` for a fitted categorical GLM.

    ``P`` is the ``(n, K)`` matrix of fitted category probabilities and ``Y``
    the matching one-hot response.  Their cellwise difference ``Y - P`` has
    exact mean zero across all ``N × K`` cells when the row probabilities sum
    to one, which is the basis of the multiclass binned-residual panel.
    """
    data = resolve(result)
    model = data.result
    prob_fn = getattr(model, "predict_proba", None)
    if not callable(prob_fn):
        raise ValueError(f"{data.model_name} does not expose category probabilities")
    probs = np.asarray(prob_fn(), dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("category probabilities must form a non-empty (n, K) matrix")
    codes = _observed_codes(model)
    if codes.size != probs.shape[0]:
        raise ValueError("observed categories and predicted probabilities have different lengths")
    keep = (
        np.isfinite(codes)
        & (codes >= 0)
        & (codes < probs.shape[1])
        & np.isfinite(probs).all(axis=1)
    )
    probs, codes = probs[keep], codes[keep].astype(int)
    if probs.shape[0] == 0:
        raise ValueError("no finite observed/predicted category rows are available")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), codes] = 1.0
    return probs, one_hot, _category_names(model, probs.shape[1]), data


def category_probabilities(result: Any, ax=None, *, figsize=(6.5, 4.6), title: str | None = None):
    """Boxplots of the predicted probability assigned to the observed category."""
    data = resolve(result)
    model = data.result
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ValueError(f"{data.model_name} does not expose category probabilities")
    probs = np.asarray(predict_proba(), dtype=float)
    codes = _observed_codes(model)
    n_cats = probs.shape[1]
    names = _category_names(model, n_cats)
    by_cat = [probs[codes == k, k] for k in range(n_cats)]
    if all(arr.size == 0 for arr in by_cat):
        raise ValueError("no observations are available for any category")

    fig, ax, _ = single_axis(ax, figsize)
    colors = [tuple(c) for c in _cmap(n_cats)]
    try:
        bp = ax.boxplot(by_cat, tick_labels=names, patch_artist=True)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(by_cat, labels=names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(1.0 / n_cats, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.2, label="Chance (1/K)")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Predicted probability of the observed class")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    label = title or f"Category discrimination — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def _cmap(n: int):
    from .foundation import pyplot

    cmap = pyplot().get_cmap("Set2")
    return [tuple(cmap(i)) for i in np.linspace(0.0, 1.0, max(n, 2))][:n]


def ordinal_curves(result: Any, ax=None, *, n_grid: int = 200, figsize=(6.5, 4.6), title: str | None = None):
    """Category-probability curves along the linear predictor for an ordinal model."""
    data = resolve(result)
    model = data.result
    gamma = getattr(model, "gamma", None)
    eta = getattr(model, "linear_predictors", None)
    family = getattr(model, "family", None)
    if gamma is None or eta is None or family is None or type(family).__name__ != "Ordinal":
        raise ValueError(f"ordinal curves are only available for Ordinal fits (got {data.model_name})")

    from ..glm import mle_dist as _dist

    eta = np.asarray(eta, dtype=float).reshape(-1)
    grid = np.linspace(eta.min() - 0.5, eta.max() + 0.5, n_grid)
    cuts = np.r_[-np.inf, np.asarray(gamma, dtype=float), np.inf]
    link = family.link
    edges = np.array(
        [
            np.where(np.isinf(c), float(c > 0), _dist.inv_link(link, c - grid))
            for c in cuts
        ]
    ).T
    probs = np.diff(edges, axis=1)
    n_cats = probs.shape[1]
    names = _category_names(model, n_cats)

    fig, ax, _ = single_axis(ax, figsize)
    colors = _cmap(n_cats)
    for k in range(n_cats):
        ax.plot(grid, probs[:, k], linewidth=2.0, color=colors[k], label=names[k])
    for idx, cut in enumerate(np.asarray(gamma, dtype=float)):
        ax.axvline(cut, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.1, label="Thresholds (γ)" if idx == 0 else None)
    ax.set_xlabel("Linear predictor (η = Xβ)")
    ax.set_ylabel("P(Y = k | η)")
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    label = title or "Ordinal probability curves"
    ax.set_title(label)
    return PlotResult(fig, label)


def fitted_curve(
    result: Any,
    ax=None,
    *,
    x: str | None = None,
    alpha: float = 0.05,
    show_confidence: bool = True,
    show_prediction: bool = True,
    newdata=None,
    figsize=(6.8, 4.8),
    title: str | None = None,
):
    """Fitted curve with delta-method confidence and prediction bands.

    Designed for nonlinear least squares fits (``statcraft.nls``), where the
    response is a smooth function of one predictor.  For models without
    prediction intervals the fitted values are drawn as a sorted line.
    """
    import pandas as pd

    data = resolve(result)
    model = data.result
    frame = newdata if newdata is not None else getattr(model, "data", None)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{data.model_name} does not keep the data needed for a curve plot")

    exog = getattr(model, "_exogenous", None) or []
    if x is None:
        if len(exog) != 1:
            raise ValueError(
                "the model has several predictors; pass x=<column name> to choose the curve axis"
            )
        xname = exog[0]
    else:
        xname = str(x)
    if xname not in frame.columns:
        raise ValueError(f"column {xname!r} is not present in the model data")

    ordered = frame.sort_values(xname).reset_index(drop=True)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError(f"{data.model_name} does not expose predict()")

    def _band(kind):
        try:
            out = predict(newdata=ordered, interval=kind, alpha=alpha)
        except TypeError:
            return None
        except Exception:
            return None
        if not isinstance(out, pd.DataFrame) or not {"fit", "lower", "upper"}.issubset(out.columns):
            return None
        return out

    confidence = _band("confidence") if show_confidence else None
    prediction = _band("prediction") if show_prediction else None
    baseline = confidence if confidence is not None else prediction
    if baseline is None:
        point = predict(newdata=ordered)
        point = pd.DataFrame({"fit": np.asarray(point, dtype=float).reshape(-1)})
    else:
        point = baseline

    x_grid = ordered[xname].to_numpy(dtype=float)
    fig, ax, _ = single_axis(ax, figsize)
    if show_prediction and prediction is not None:
        ax.fill_between(
            x_grid,
            prediction["lower"].to_numpy(dtype=float),
            prediction["upper"].to_numpy(dtype=float),
            color="#fbbf24",
            alpha=0.22,
            label=f"{(1 - alpha) * 100:.0f}% prediction band",
        )
    if show_confidence and confidence is not None:
        ax.fill_between(
            x_grid,
            confidence["lower"].to_numpy(dtype=float),
            confidence["upper"].to_numpy(dtype=float),
            color="#60a5fa",
            alpha=0.35,
            label=f"{(1 - alpha) * 100:.0f}% confidence band",
        )
    ax.plot(x_grid, point["fit"].to_numpy(dtype=float), color="#1e3a8a", linewidth=2.2, label="Fitted curve")

    y = data.response
    if y is not None and np.asarray(y).size == frame.shape[0]:
        ax.scatter(
            np.asarray(frame[xname], dtype=float),
            np.asarray(y, dtype=float),
            color="black",
            s=22,
            alpha=0.7,
            zorder=4,
            label="Observed",
        )
    ax.set_xlabel(xname)
    ax.set_ylabel(data.response_name)
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    label = title or f"Fitted curve — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)
