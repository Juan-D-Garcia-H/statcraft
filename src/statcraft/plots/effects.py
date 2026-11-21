# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Random effects, predictive checks and distributional shape plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .adapters import random_effects, resolve
from .foundation import (
    DEFAULT_COLORS,
    PlotResult,
    clean,
    panel_axes,
    single_axis,
    style_axis,
)

__all__ = [
    "dominant_random_effect",
    "ranef_caterpillar",
    "ranef_distribution",
    "qq_random_effects",
    "vpc",
    "lorenz_curve",
    "gengamma_submodels",
]


def _choose_group(table: dict[str, pd.DataFrame], group=None) -> tuple[str, pd.DataFrame]:
    if group is not None:
        key = str(group)
        if key not in table:
            raise KeyError(f"group {group!r} not found; available: {list(table)}")
        return key, table[key]
    key = next(iter(table))
    return key, table[key]


def _choose_term(frame: pd.DataFrame, term) -> str:
    if isinstance(term, (int, np.integer)):
        return str(frame.columns[int(term)])
    name = str(term)
    if name in frame.columns:
        return name
    if name in ("Intercept", "(Intercept)"):
        alt = "(Intercept)" if name == "Intercept" else "Intercept"
        if alt in frame.columns:
            return alt
    raise KeyError(f"column {term!r} not found; available: {list(frame.columns)}")


def _numeric_ranef_columns(frame: pd.DataFrame) -> list[str]:
    ignore = {"id", "subject", "group"}
    return [
        str(col)
        for col in frame.columns
        if str(col).lower() not in ignore and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _blup_conditional_se(result: Any, frame: pd.DataFrame, group=None) -> pd.DataFrame | None:
    """Posterior SDs of the BLUPs, when the covariance can be reconstructed.

    For a group with design ``Z``, random-effect covariance ``G`` and residual
    variance ``σ²``, the conditional covariance is
    ``(G⁻¹ + ZᵀZ/σ²)⁻¹``.  LME reconstructs ``Z`` from the random-effects model
    matrix, NLME uses the linearised derivative matrix ``raw.X_lin`` at the
    fitted parameters, and GLMER uses the GLM working weights.  Returns
    ``None`` when either ingredient is missing.
    """
    name = type(result).__name__
    if name == "GlmerFit":
        method = getattr(result, "blup_conditional_se", None)
        if not callable(method):
            return None
        try:
            tables = method()
        except Exception:
            return None
        table = tables.get(str(group)) if isinstance(tables, dict) else None
        if table is None and isinstance(tables, dict) and len(tables) == 1:
            table = next(iter(tables.values()))
        if table is None:
            return None
        columns = [str(c) for c in frame.columns]
        return table.reindex(index=frame.index, columns=columns) if set(columns) <= set(map(str, table.columns)) else None
    if name not in ("LmeFit", "NlmeFit"):
        return None
    raw = getattr(result, "raw", None)
    if raw is None:
        return None
    G = np.asarray(getattr(raw, "Sigma_b", None), dtype=float)
    sigma2 = np.asarray(getattr(raw, "sigma2", np.nan), dtype=float).reshape(-1)
    columns = _numeric_ranef_columns(frame)
    q = len(columns)
    if (
        G.ndim != 2
        or G.shape[0] < q
        or G.shape[1] < q
        or sigma2.size == 0
        or not np.isfinite(sigma2[0])
        or sigma2[0] <= 0
        or not np.isfinite(G[:q, :q]).all()
    ):
        return None
    G = G[:q, :q]
    data = getattr(result, "_data", None)
    if data is None:
        return None

    if name == "LmeFit":
        levels = getattr(result, "_pred_levels", None)
        if not levels:
            return None
        spec = levels[0].get("spec")
        chain = levels[0].get("chain") or []
        group_col = chain[0] if chain else (frame.index.name or "group")
        if spec is None:
            return None
        try:
            Z = np.asarray(spec.get_materializer(data).get_model_matrix(spec), dtype=float)
        except Exception:
            return None
    else:
        X = np.asarray(getattr(raw, "X_lin", None), dtype=float)
        names = [str(n) for n in (getattr(result, "names", None) or [])]
        indices: list[int] = []
        for column in columns:
            if column in names:
                indices.append(names.index(column))
            elif column in ("(Intercept)", "Intercept") and "Intercept" in names:
                indices.append(names.index("Intercept"))
            else:
                return None
        if X.ndim != 2 or not indices or max(indices) >= X.shape[1]:
            return None
        Z = X[:, indices]
        random_spec = getattr(result, "random_spec", None) or []
        if random_spec and isinstance(random_spec[0], (tuple, list)):
            group_col = random_spec[0][0]
        else:
            group_col = frame.index.name or "group"

    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != len(data) or Z.shape[1] != q or group_col not in data.columns:
        return None

    G_inv = np.linalg.pinv(G)
    labels = np.asarray(data[group_col])
    ses = np.full((len(frame), q), np.nan, dtype=float)
    for row, label in enumerate(frame.index):
        mask = labels == label
        Zi = Z[mask]
        if Zi.shape[0] == 0:
            continue
        covariance = np.linalg.pinv(G_inv + Zi.T @ Zi / sigma2[0])
        ses[row] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if not np.isfinite(ses).all():
        return None
    return pd.DataFrame(ses, index=frame.index, columns=columns)


def dominant_random_effect(result: Any) -> str | None:
    """Name of the random-effect term with the largest relative variability.

    Nonlinear fits often carry several random effects and one may collapse to
    a numerical zero.  Ranking by ``SD / (SD + |mean|)`` selects the
    informative term without being fooled by a large but constant offset.
    """
    table = random_effects(result)
    if not table:
        return None
    _, frame = _choose_group(table)
    columns = _numeric_ranef_columns(frame)
    if not columns:
        return None

    def score(column: str) -> float:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size < 2:
            return -np.inf
        sd = float(np.std(values, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return -np.inf
        return sd / (sd + abs(float(np.mean(values))))

    best = max(columns, key=score)
    return best if np.isfinite(score(best)) else None


def ranef_caterpillar(
    result: Any,
    ax=None,
    *,
    group=None,
    term=0,
    max_groups: int = 30,
    figsize=None,
    title: str | None = None,
):
    """Caterpillar plot of the estimated random effects (BLUPs/EBEs).

    Groups are sorted by their effect; when there are more than ``max_groups``
    rows, the two most extreme halves are shown.  When the covariance can be
    reconstructed, 95% conditional intervals (posterior SD of the BLUP) are
    drawn as error bars and the sample SD is shown in the title.
    """
    data = resolve(result)
    table = random_effects(result)
    if not table:
        raise ValueError(f"random effects are not available for {data.model_name}")
    key, frame = _choose_group(table, group)
    column = _choose_term(frame, term)
    series = frame[column].dropna().sort_values()
    if len(series) > max_groups:
        half = max(max_groups // 2, 1)
        series = pd.concat([series.iloc[:half], series.iloc[-half:]])
    se_frame = _blup_conditional_se(result, frame, group=key)
    se = None
    if se_frame is not None and column in se_frame.columns:
        se = se_frame[column].reindex(series.index).to_numpy(dtype=float)
    if figsize is None:
        figsize = (6.5, max(4.0, 0.26 * len(series)))

    fig, ax, _ = single_axis(ax, figsize)
    y_pos = np.arange(len(series))
    if se is not None and np.isfinite(se).all():
        ax.errorbar(
            series.to_numpy(dtype=float), y_pos, xerr=1.96 * se, fmt="none",
            ecolor="#16a34a", elinewidth=1.2, capsize=3, alpha=0.8, zorder=2,
            label="95% conditional CI",
        )
    ax.scatter(series.to_numpy(dtype=float), y_pos, color="#16a34a", edgecolor="#14532d", zorder=3)
    ax.hlines(y_pos, 0.0, series.to_numpy(dtype=float), color="#16a34a", alpha=0.5, linewidth=1.2)
    ax.axvline(0.0, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(idx) for idx in series.index], fontsize=8.5)
    ax.set_xlabel(f"Random effect ({column})")
    if se is not None and np.isfinite(se).all():
        ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    values = series.to_numpy(dtype=float)
    sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    label = title or f"Random effects: {key} [{column}] (SD = {sd:.3g})"
    ax.set_title(label)
    return PlotResult(fig, label)


def ranef_distribution(
    result: Any,
    ax=None,
    *,
    group=None,
    bins: int = 16,
    max_terms: int = 4,
    figsize=None,
    title: str | None = None,
):
    """Histograms of the random effects with a matched normal density."""
    from scipy import stats

    data = resolve(result)
    table = random_effects(result)
    if not table:
        raise ValueError(f"random effects are not available for {data.model_name}")
    key, frame = _choose_group(table, group)
    columns = [c for c in frame.columns if not str(c).lower() in ("id", "subject", "group")][:max_terms]
    if not columns:
        raise ValueError("the random-effect table has no numeric columns")
    if figsize is None:
        figsize = (4.4 * len(columns), 3.8)
    fig, axes = panel_axes(ax, 1, len(columns), figsize=figsize)
    axes = axes.ravel()

    for panel, column in zip(axes, columns):
        values = np.asarray(frame[column], dtype=float)
        values = values[np.isfinite(values)]
        panel.hist(values, bins=bins, density=True, color="#34d399", edgecolor="#064e3b", alpha=0.75)
        panel.axvline(0.0, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.1)
        if values.size > 2 and np.std(values) > 0:
            grid = np.linspace(values.min(), values.max(), 120)
            panel.plot(grid, stats.norm.pdf(grid, values.mean(), values.std()), color="black", linewidth=1.4)
        panel.set_title(f"{key}: {column}", fontsize=9.5)
        panel.set_xlabel("Random effect")
        panel.set_ylabel("Density")
        style_axis(panel)

    label = title or f"Random-effect distributions — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def qq_random_effects(
    result: Any,
    ax=None,
    *,
    group=None,
    term=None,
    figsize=None,
    title: str | None = None,
):
    """Normal Q-Q plots of the estimated random effects.

    One panel per random-effect term (or a single panel when ``term`` is
    given).  A straight-line pattern supports the assumed normal distribution
    of the BLUPs/EBEs.
    """
    from .foundation import qq_points

    data = resolve(result)
    table = random_effects(result)
    if not table:
        raise ValueError(f"random effects are not available for {data.model_name}")
    key, frame = _choose_group(table, group)
    if term is not None:
        columns = [_choose_term(frame, term)]
    else:
        columns = [str(c) for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
    if not columns:
        raise ValueError("the random-effect table has no numeric columns")

    if figsize is None:
        figsize = (4.4 * len(columns), 3.8)
    fig, axes = panel_axes(ax, 1, len(columns), figsize=figsize)
    axes = axes.ravel()

    for panel, column in zip(axes, columns):
        values = np.asarray(frame[column], dtype=float)
        values = values[np.isfinite(values)]
        osm, osr = qq_points(values)
        if osm.size:
            panel.scatter(osm, osr, alpha=0.65, color="#7c3aed", edgecolors="none", label="Random effects")
            slope, intercept = np.polyfit(osm, osr, 1)
            panel.plot(osm, slope * osm + intercept, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.5, label="Normal reference")
        panel.set_xlabel("Theoretical N(0, 1) quantiles")
        panel.set_ylabel("Ordered random effect")
        panel.set_title(f"{key}: {column}", fontsize=9.5)
        panel.legend(loc="best", fontsize=7.5)
        style_axis(panel)

    label = title or f"Random-effect Q-Q — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def _simulation_matrix(sims) -> np.ndarray:
    """Normalise simulated responses to ``(n_sim, n_obs)``.

    ``SaemFit.simulate`` returns one ``sim_k`` column per replication and one
    row per observation, so the explicit column convention takes precedence
    over any size-based guess.
    """
    if isinstance(sims, pd.DataFrame):
        cols = [c for c in sims.columns if str(c).startswith("sim_")]
        if cols:
            return sims[cols].to_numpy(dtype=float).T
        arr = sims.to_numpy(dtype=float)
    else:
        arr = np.asarray(sims, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr


def vpc(
    result: Any,
    ax=None,
    *,
    nsim: int = 200,
    n_bins: int = 8,
    seed: int = 42,
    figsize=(7.5, 5.0),
    title: str | None = None,
):
    """Visual predictive check using Monte Carlo simulations from the fit."""
    data = resolve(result)
    model = data.result
    simulate = getattr(model, "simulate", None)
    if not callable(simulate):
        raise ValueError(f"{data.model_name} cannot simulate new responses")
    try:
        sims = simulate(nsim=nsim, seed=seed)
    except TypeError:
        try:
            sims = simulate(nsim)
        except TypeError:
            sims = simulate()
    matrix = _simulation_matrix(sims)

    y = np.asarray(data.response, dtype=float).reshape(-1)
    time = data.time if data.time is not None else np.arange(y.size, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if matrix.shape[1] != y.size:
        raise ValueError("simulated responses are not aligned with the observed data")
    keep = np.isfinite(time) & np.isfinite(y)
    if keep.sum() < n_bins:
        raise ValueError("not enough finite observations to build a VPC")
    time, y = time[keep], y[keep]
    matrix = matrix[:, keep]

    edges = np.quantile(time, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(time.min(), time.max() + 1e-9, n_bins + 1)
    edges[-1] += 1e-9
    bin_idx = np.digitize(time, edges) - 1

    rows = {"t": [], "obs10": [], "obs50": [], "obs90": [], "sim10": [], "sim50": [], "sim90": [], "sim50_lo": [], "sim50_hi": []}
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.sum() < 3:
            continue
        block = matrix[:, mask]
        rows["t"].append(float(np.median(time[mask])))
        rows["obs10"].append(float(np.percentile(y[mask], 10)))
        rows["obs50"].append(float(np.percentile(y[mask], 50)))
        rows["obs90"].append(float(np.percentile(y[mask], 90)))
        p10 = np.percentile(block, 10, axis=1)
        p50 = np.percentile(block, 50, axis=1)
        p90 = np.percentile(block, 90, axis=1)
        rows["sim10"].append(float(np.median(p10)))
        rows["sim50"].append(float(np.median(p50)))
        rows["sim90"].append(float(np.median(p90)))
        rows["sim50_lo"].append(float(np.percentile(p50, 5)))
        rows["sim50_hi"].append(float(np.percentile(p50, 95)))

    if not rows["t"]:
        raise ValueError("no time bin had enough observations for a VPC")
    t = np.asarray(rows["t"])

    fig, ax, _ = single_axis(ax, figsize)
    ax.scatter(time, y, color="#64748b", alpha=0.22, s=14, label="Observations")
    ax.fill_between(t, rows["sim10"], rows["sim90"], color="#3b82f6", alpha=0.16, label="Simulated P10–P90")
    ax.fill_between(t, rows["sim50_lo"], rows["sim50_hi"], color="#dc2626", alpha=0.22, label="90% CI of simulated median")
    ax.plot(t, rows["sim50"], color="#dc2626", linestyle="--", linewidth=1.8, label="Simulated median")
    ax.plot(t, rows["obs50"], "ko-", linewidth=2.0, label="Observed median")
    ax.plot(t, rows["obs10"], "k:", linewidth=1.3, label="Observed P10 / P90")
    ax.plot(t, rows["obs90"], "k:", linewidth=1.3)
    ax.set_xlabel(data.time_name or "Time")
    ax.set_ylabel(data.response_name)
    ax.legend(loc="best", fontsize=8)
    style_axis(ax)
    label = title or f"Visual predictive check ({matrix.shape[0]} simulations)"
    ax.set_title(label)
    return PlotResult(fig, label)


def lorenz_curve(result: Any, ax=None, *, figsize=(5.8, 4.8), title: str | None = None):
    """Lorenz curve and Gini index of the response ordered by fitted values.

    Useful for severity/claims models (Tweedie, Gamma, Pareto): a curve that
    separates from the diagonal means the model ranks risk better than chance.
    """
    data = resolve(result)
    y = np.asarray(data.response, dtype=float).reshape(-1)
    mu = np.asarray(data.fitted, dtype=float).reshape(-1)
    y, mu = clean(y, mu)
    if y.size == 0:
        raise ValueError("no finite observations are available")
    if np.any(y < 0):
        raise ValueError("the Lorenz curve needs a non-negative response")
    order = np.argsort(mu)
    y_sorted = y[order]
    total = float(np.sum(y_sorted))
    if total <= 0:
        raise ValueError("the response sums to zero; the Lorenz curve is undefined")
    cum_pop = np.r_[0.0, np.arange(1, y_sorted.size + 1) / y_sorted.size]
    cum_share = np.r_[0.0, np.cumsum(y_sorted) / total]
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    gini = 1.0 - 2.0 * float(trapz(cum_share, cum_pop))

    fig, ax, _ = single_axis(ax, figsize)
    ax.plot(cum_pop, cum_share, color="#7e22ce", linewidth=2.2, label=f"Model ordering (Gini = {gini:.4f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Perfect equality")
    ax.set_xlabel("Cumulative share of observations (sorted by μ)")
    ax.set_ylabel("Cumulative share of the response")
    ax.legend(loc="upper left", fontsize=8.5)
    style_axis(ax)
    label = title or f"Lorenz curve — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def gengamma_submodels(result: Any, ax=None, *, figsize=(6.5, 4.6), title: str | None = None):
    """Boxplots of the row-wise location, scale and shape of a GeneralizedGamma fit."""
    data = resolve(result)
    model = data.result
    row_params = getattr(model, "_gengamma_row_params", None)
    if not callable(row_params):
        raise ValueError(f"generalized-gamma submodels are not available for {data.model_name}")
    theta = np.asarray(getattr(model, "all_params"), dtype=float)
    exog = np.asarray(getattr(model, "exog"), dtype=float)
    mu_i, sigma_i, q_i = row_params(theta, exog)

    fig, ax, _ = single_axis(ax, figsize)
    try:
        bp = ax.boxplot([mu_i, sigma_i, q_i], tick_labels=["Location (μ)", "Scale (σ)", "Shape (Q)"], patch_artist=True)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot([mu_i, sigma_i, q_i], labels=["Location (μ)", "Scale (σ)", "Shape (Q)"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#60a5fa", "#34d399", "#fb923c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.axhline(0.0, color=DEFAULT_COLORS["neutral"], linestyle=":", linewidth=1.0)
    ax.set_ylabel("Estimated value in the sample")
    style_axis(ax)
    label = title or "GeneralizedGamma submodels (μ, σ, Q)"
    ax.set_title(label)
    return PlotResult(fig, label)
