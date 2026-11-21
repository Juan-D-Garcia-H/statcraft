# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Coefficient, covariance and per-group estimate plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .adapters import coefficient_table, default_exponentiate, resolve
from .foundation import (
    DEFAULT_COLORS,
    PlotResult,
    colorbar,
    panel_axes,
    single_axis,
    style_axis,
)

__all__ = [
    "forest_plot",
    "fixed_effect_correlation",
    "response_correlation",
    "variance_components",
    "covariance_parameters",
    "list_coefficients",
]


def _is_intercept(name: Any) -> bool:
    text = str(name).strip()
    return text in ("(Intercept)", "Intercept", "const") or text.endswith("Intercept") or text.endswith("const")


def _boxplot(ax, data, labels, colors=None):
    """``Axes.boxplot`` that works across Matplotlib versions."""
    try:
        bp = ax.boxplot(data, tick_labels=list(labels), patch_artist=True)
    except TypeError:  # matplotlib < 3.9
        bp = ax.boxplot(data, labels=list(labels), patch_artist=True)
    if colors is not None:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
    return bp


def forest_plot(
    result: Any,
    ax=None,
    *,
    alpha: float = 0.05,
    exponentiate: bool | None = None,
    exclude_intercept: bool = True,
    response: int = 0,
    figsize=(6.5, 4.6),
    title: str | None = None,
):
    """Forest plot of the coefficients with Wald confidence intervals.

    Ratios (odds/rate/hazard) are shown on a log axis by default for GLMs with
    log or logit links; ``exponentiate`` overrides that choice.  ``MCGLMResults``
    accepts a ``response`` index to choose one of the stacked responses, and
    ``ListFit`` is redirected to :func:`list_coefficients`.
    """
    data = resolve(result)
    if data.kind == "list":
        raise TypeError("ListFit stores one coefficient table per group; use list_coefficients().")

    if exponentiate is None:
        exponentiate = default_exponentiate(result)
    table = coefficient_table(result, alpha=alpha, exponentiate=exponentiate, response=response)

    if exclude_intercept and not exponentiate:
        keep = [not _is_intercept(name) for name in table.index]
        kept = table[keep]
        if len(kept):
            table = kept
    if len(table) == 0:
        raise ValueError("no coefficients are available to plot")

    values = np.asarray(table["estimate"], dtype=float)
    lower = np.asarray(table["lower"], dtype=float)
    upper = np.asarray(table["upper"], dtype=float)
    names = [str(name) for name in table.index]
    y_pos = np.arange(len(names))

    fig, ax, _ = single_axis(ax, figsize)
    err = np.vstack([values - lower, upper - values])
    ax.errorbar(
        values,
        y_pos,
        xerr=err,
        fmt="o",
        color="#1e3a8a",
        ecolor="#60a5fa",
        elinewidth=2.0,
        capsize=4,
        markersize=5,
    )
    null = 1.0 if exponentiate else 0.0
    ax.axvline(null, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.2, label=f"Null ({null:g})")
    if exponentiate:
        ax.set_xscale("log")
        ax.set_xlabel(f"exp(estimate) [{int((1 - alpha) * 100)}% CI]")
    else:
        ax.set_xlabel(f"Estimate [{int((1 - alpha) * 100)}% CI]")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8.5)
    style_axis(ax)
    label = title or f"Fixed effects — {data.model_name}"
    if exponentiate:
        label = title or f"Effect ratios — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def _correlation_from_vcov(result: Any) -> tuple[np.ndarray, list[str]] | None:
    vcov = getattr(result, "vcov", None)
    names = None
    if isinstance(vcov, pd.DataFrame):
        corr = vcov.to_numpy(dtype=float)
        names = [str(n) for n in vcov.index]
    elif vcov is not None:
        corr = np.asarray(vcov, dtype=float)
        raw_names = getattr(result, "param_names", None) or getattr(result, "all_param_names", None)
        if raw_names is not None and len(raw_names) == corr.shape[0]:
            names = [str(n) for n in raw_names]
    else:
        return None
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return None
    std = np.sqrt(np.diag(corr))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = corr / np.outer(std, std)
    if names is None:
        names = [f"β{j}" for j in range(corr.shape[0])]
    return out, names


def fixed_effect_correlation(result: Any, ax=None, *, cmap: str = "RdBu_r", figsize=(5.8, 4.8), title: str | None = None):
    """Heatmap of the correlation matrix of the fixed effects."""
    data = resolve(result)
    corr, names = None, None
    if data.kind == "mcglm":
        try:
            indices = np.concatenate([np.asarray(ix, dtype=int).reshape(-1) for ix in result._ix_beta])
            cov_all = np.asarray(result.cov_params(), dtype=float)
            corr = cov_all[np.ix_(indices, indices)]
            exog = getattr(result.model, "exog_names", None) or []
            endog = getattr(result.model, "endog_names", None) or []
            names = []
            for k, ix in enumerate(result._ix_beta):
                for j in range(len(ix)):
                    prefix = f"{endog[k]}:" if k < len(endog) and len(endog) > 1 else ""
                    names.append(prefix + str(exog[k][j]) if k < len(exog) else f"β{j}")
        except Exception:
            corr, names = None, None
    native = getattr(result, "cor_fixef", None)
    if corr is None and callable(native):
        try:
            frame = native()
            if isinstance(frame, pd.DataFrame):
                corr = frame.to_numpy(dtype=float)
                names = [str(n) for n in frame.index]
        except Exception:
            corr, names = None, None
    if corr is None:
        found = _correlation_from_vcov(result)
        if found is None:
            raise ValueError(f"the fixed-effect covariance is not available for {data.model_name}")
        corr, names = found

    fig, ax, _ = single_axis(ax, figsize)
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap=cmap)
    colorbar(fig, im, ax, label="Correlation")
    tick = np.arange(len(names))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(corr[i, j]):
                color = "white" if abs(corr[i, j]) > 0.6 else "black"
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    style_axis(ax, grid=False)
    label = title or f"Fixed-effect correlation — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def _model_between_correlation(result: Any, K: int) -> np.ndarray:
    covariance = getattr(result, "covariance_matrix", None)
    if callable(covariance):
        try:
            info = covariance(full=False)
            matrix = np.asarray(info.Sigma_between, dtype=float)
            if matrix.shape == (K, K):
                return matrix
        except Exception:
            pass
    rho = np.asarray(getattr(result, "rho", []), dtype=float).reshape(-1)
    matrix = np.eye(K)
    pos = 0
    for i in range(K):
        for j in range(i + 1, K):
            if pos >= rho.size:
                break
            matrix[i, j] = matrix[j, i] = rho[pos]
            pos += 1
    return matrix


def response_correlation(
    result: Any,
    ax=None,
    *,
    cmap: str = "RdBu_r",
    figsize=(11.0, 4.6),
    title: str | None = None,
):
    """Observed and model-implied between-response correlation heatmaps.

    Only defined for multivariate ``MCGLMResults`` (two or more responses).
    The left panel is the sample correlation of the observed responses; the
    right panel is the correlation implied by the fitted joint covariance
    (``Sigma_between``).
    """
    data = resolve(result)
    model = data.result
    K = int(getattr(model, "K", 1) or 1)
    if K < 2:
        raise ValueError("the between-response correlation needs at least two responses")
    n = int(getattr(model, "nobs", 0) or 0)
    y = np.asarray(getattr(model, "y", None), dtype=float).reshape(K, n)
    observed = pd.DataFrame(y.T).corr().to_numpy(dtype=float)
    implied = _model_between_correlation(result, K)

    endog = getattr(getattr(model, "model", None), "endog_names", None) or []
    names = [str(endog[k]) if k < len(endog) else f"Response {k + 1}" for k in range(K)]

    fig, axes = panel_axes(ax, 1, 2, figsize=figsize)
    for panel, matrix, panel_title in zip(
        axes.ravel(), (observed, implied), ("Observed correlation", "Model-implied correlation")
    ):
        im = panel.imshow(matrix, vmin=-1.0, vmax=1.0, cmap=cmap)
        colorbar(fig, im, panel, label="Correlation")
        tick = np.arange(K)
        panel.set_xticks(tick)
        panel.set_yticks(tick)
        panel.set_xticklabels(names, rotation=45, ha="right")
        panel.set_yticklabels(names)
        panel.set_title(panel_title, fontsize=10.5)
        for i in range(K):
            for j in range(K):
                if np.isfinite(matrix[i, j]):
                    color = "white" if abs(matrix[i, j]) > 0.6 else "black"
                    panel.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
        style_axis(panel, grid=False)

    label = title or f"Between-response correlation — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def _variance_table(result: Any) -> tuple[np.ndarray, list[str]] | None:
    varcorr = getattr(result, "varcorr", None)
    if isinstance(varcorr, pd.DataFrame) and len(varcorr):
        frame = varcorr.reset_index()
        if "std_dev" in frame.columns:
            values = np.asarray(frame["std_dev"], dtype=float)
            labels = [
                f"{g} · {n}" if str(g) not in ("", "nan", "None") else str(n)
                for g, n in zip(frame.get("group", [""] * len(frame)), frame.get("name", frame.iloc[:, 0]))
            ]
            return values, labels
        if "estimate" in frame.columns:
            values = np.sqrt(np.abs(np.asarray(frame["estimate"], dtype=float)))
            labels = [str(n) for n in frame.get("name", frame.iloc[:, 0])]
            return values, labels
    # SAEM stores the random-effect standard deviations directly.
    sd = getattr(result, "sd", None)
    omega = getattr(result, "omega", None)
    if sd is not None:
        values = np.asarray(sd, dtype=float).reshape(-1)
        names = getattr(result, "params", None)
        labels = [str(n) for n in names] if names is not None and len(names) == values.size else [f"RE {j}" for j in range(values.size)]
        keep = values > 0
        return values[keep], [lab for lab, ok in zip(labels, keep) if ok]
    if callable(omega):
        try:
            frame = omega()
            values = np.sqrt(np.maximum(np.diag(frame.to_numpy(dtype=float)), 0.0))
            return values, [str(n) for n in frame.index]
        except Exception:
            return None
    return None


def variance_components(result: Any, ax=None, *, figsize=(6.5, 4.5), title: str | None = None):
    """Standard deviations of the random effects (variance components)."""
    data = resolve(result)
    found = _variance_table(result)
    if found is None:
        raise ValueError(f"variance components are not available for {data.model_name}")
    values, labels = found
    order = np.argsort(values)
    values = values[order]
    labels = [labels[i] for i in order]
    fig, ax, _ = single_axis(ax, figsize)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#0f766e", alpha=0.78, edgecolor="#134e4a")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standard deviation")
    style_axis(ax)
    label = title or f"Variance components — {data.model_name}"
    ax.set_title(label)
    return PlotResult(fig, label)


def _is_covariance_correlation(name: Any) -> bool:
    """Whether a covariance parameter label denotes a correlation (``ρ``)."""
    text = str(name).lower()
    return "rho" in text or "corr" in text


def _covariance_parameters_panel(ax, names, values, lower, upper, *, subtitle, alpha, corr):
    """Draw one group of covariance estimates and their confidence intervals."""
    y_pos = np.arange(len(names))
    ax.errorbar(
        values,
        y_pos,
        xerr=np.vstack([values - lower, upper - values]),
        fmt="s",
        color="#065f46",
        ecolor="#34d399",
        elinewidth=2.0,
        capsize=4,
    )
    ax.axvline(0.0, color=DEFAULT_COLORS["neutral"], linestyle="--", linewidth=1.1)
    if corr:
        # Correlations live on a fixed scale; keep their intervals readable
        # instead of letting a large dispersion compress them.
        ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"Estimate [{int((1 - alpha) * 100)}% CI]")
    if subtitle:
        ax.set_title(subtitle, fontsize=10.0)
    style_axis(ax)


def covariance_parameters(result: Any, ax=None, *, alpha: float = 0.05, figsize=(6.5, 4.5), title: str | None = None):
    """Estimated covariance / dispersion parameters with confidence intervals.

    Correlations (``rho``) are drawn on a fixed ``[-1, 1]`` axis above the
    dispersion and power parameters, which keep their natural scale.  When a
    fit has only one of the two groups the single group spans the whole panel.

    Works for ``MCGLMResults`` (rho, power, tau) and for any result exposing a
    ``param_names``/``conf_int`` pair where the covariance block can be
    separated from the regression coefficients.
    """
    data = resolve(result)
    names = getattr(result, "param_names", None)
    beta = getattr(result, "beta", None)
    cov = getattr(result, "cov", None)
    conf = getattr(result, "conf_int", None)
    if names is None or beta is None or cov is None or not callable(conf):
        raise ValueError(f"covariance parameters are not available for {data.model_name}")

    n_beta = np.asarray(beta).size
    cov = np.asarray(cov, dtype=float).reshape(-1)
    cov_names = [str(n) for n in names[n_beta : n_beta + cov.size]]
    ci = np.asarray(conf(alpha=alpha), dtype=float)[n_beta : n_beta + cov.size]
    lower, upper = ci[:, 0], ci[:, 1]
    if cov.size == 0:
        raise ValueError(f"covariance parameters are not available for {data.model_name}")

    label = title or f"Covariance parameters — {data.model_name}"
    corr_mask = np.array([_is_covariance_correlation(name) for name in cov_names])
    has_both = bool(corr_mask.any()) and bool((~corr_mask).any())

    if ax is None and has_both:
        fig, grid = panel_axes(None, 2, 1, figsize)
        axes = [grid[0, 0], grid[1, 0]]
        fig.suptitle(label)
    elif ax is None:
        fig, only = single_axis(None, figsize)
        only.set_title(label)
        axes = [only]
    else:
        fig, parent, _ = single_axis(ax)
        parent.set_axis_off()
        # The two inset panels need room for the top panel's x tick labels and
        # the bottom panel's title; lift the overall title clear of the top one.
        parent.set_title(label, y=1.10)
        rect = [0.03, 0.58, 0.94, 0.34] if has_both else [0.02, 0.04, 0.96, 0.90]
        axes = [parent.inset_axes(rect)]
        if has_both:
            axes.append(parent.inset_axes([0.03, 0.05, 0.94, 0.38]))

    if has_both:
        _covariance_parameters_panel(
            axes[0], [n for n, is_corr in zip(cov_names, corr_mask) if is_corr],
            cov[corr_mask], lower[corr_mask], upper[corr_mask],
            subtitle="Correlations (ρ)", alpha=alpha, corr=True,
        )
        _covariance_parameters_panel(
            axes[1], [n for n, is_corr in zip(cov_names, corr_mask) if not is_corr],
            cov[~corr_mask], lower[~corr_mask], upper[~corr_mask],
            subtitle="Dispersion / power (τ)", alpha=alpha, corr=False,
        )
    else:
        _covariance_parameters_panel(
            axes[0], cov_names, cov, lower, upper,
            subtitle=None, alpha=alpha, corr=bool(corr_mask.all()),
        )
    return PlotResult(fig, label)


def list_coefficients(result: Any, ax=None, *, alpha: float = 0.05, figsize=None, title: str | None = None):
    """Per-group coefficient panels for a ``ListFit`` (``lm_list`` / ``nls_list``).

    One panel per parameter; each row is a group estimate with its ``t``
    interval.  The dashed line marks the unweighted mean across groups.
    """
    from scipy import stats

    data = resolve(result)
    if data.kind != "list":
        raise TypeError("list_coefficients() is only defined for ListFit results")

    coefs = getattr(result, "coef", None)
    ses = getattr(result, "se", None)
    dfs = getattr(result, "df", None)
    if not isinstance(coefs, pd.DataFrame) or not isinstance(ses, pd.DataFrame):
        raise ValueError("the ListFit result does not expose coef/se tables")

    params = list(coefs.columns)
    groups = [str(g) for g in coefs.index]
    n_params = len(params)
    if figsize is None:
        figsize = (4.6 * n_params, max(4.0, 0.3 * len(groups)))
    fig, axes = panel_axes(ax, 1, n_params, figsize=figsize)
    axes = axes.ravel()
    y_pos = np.arange(len(groups))

    df_values = (
        np.asarray(dfs, dtype=float)
        if dfs is not None
        else np.full(len(groups), max(len(groups) - 1, 1), dtype=float)
    )
    q = stats.t.ppf(1.0 - alpha / 2.0, np.maximum(df_values, 1.0))

    for j, name in enumerate(params):
        panel = axes[j]
        values = np.asarray(coefs[name], dtype=float)
        errors = np.asarray(ses[name], dtype=float)
        panel.errorbar(values, y_pos, xerr=q * errors, fmt="o", color="#1e3a8a", ecolor="#60a5fa", capsize=3)
        mean = float(np.nanmean(values))
        panel.axvline(mean, color=DEFAULT_COLORS["accent"], linestyle="--", linewidth=1.2, label="Mean across groups")
        panel.set_yticks(y_pos)
        panel.set_yticklabels(groups if j == 0 else [])
        panel.invert_yaxis()
        panel.set_xlabel(str(name))
        panel.set_title(f"Parameter: {name}")
        style_axis(panel)
        if j == 0:
            panel.legend(loc="best", fontsize=8)

    sigma = getattr(result, "pooled_sigma", None)
    suptitle = title or "Per-group estimates"
    if sigma is not None:
        suptitle = f"{suptitle} (pooled σ = {float(sigma):.3f})"
    fig.suptitle(suptitle, fontsize=13)
    return PlotResult(fig, suptitle)
