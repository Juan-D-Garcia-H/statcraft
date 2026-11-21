# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Convergence traces for iterative estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import resolve
from .foundation import DEFAULT_COLORS, PlotResult, panel_axes, single_axis, style_axis

__all__ = ["convergence", "mcglm_convergence", "saem_trace"]


def _trace_matrix(trace) -> np.ndarray:
    if trace is None:
        return np.empty((0, 0))
    if isinstance(trace, np.ndarray):
        arr = trace
    else:
        rows = [np.asarray(row, dtype=float).reshape(-1) for row in trace]
        arr = np.vstack(rows) if rows else np.empty((0, 0))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _omega_label_groups(model: Any) -> list[list[str]] | None:
    """Labels for the Omega block of ``theta``, grouped by covariance component.

    The C++ covariance model orders components by their first random effect and
    encodes a full component as a lower-triangular Cholesky factor (column by
    column), or a partial component as log standard deviations followed by free
    correlations.  Mirroring that layout here lets the trace name every column.
    """
    params = [str(n) for n in getattr(model, "params", [])]
    if not params:
        return None
    random = {str(n) for n in (getattr(model, "random", None) or params)}
    names = [n for n in params if n in random]
    q = len(names)
    if q == 0:
        return []

    spec = getattr(model, "omega", "diagonal")
    pattern = np.zeros((q, q), dtype=bool)

    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "full":
            pattern = ~np.eye(q, dtype=bool)
        elif key != "diagonal":
            return None
    elif (
        isinstance(spec, (list, tuple))
        and spec
        and all(isinstance(g, (list, tuple)) and g and all(isinstance(n, str) for n in g) for g in spec)
    ):
        for group in spec:
            idx = [names.index(n) for n in group if n in names]
            for a in idx:
                for b in idx:
                    if a != b:
                        pattern[a, b] = True
    else:
        try:
            full = np.asarray(spec).astype(bool)
        except Exception:
            return None
        if full.shape == (len(params), len(params)):
            idx = [params.index(n) for n in names]
            full = full[np.ix_(idx, idx)]
        elif full.shape != (q, q):
            return None
        pattern = full | full.T
        np.fill_diagonal(pattern, False)

    parent = list(range(q))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*np.where(np.triu(pattern, 1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for i in range(q):
        groups.setdefault(find(i), []).append(i)

    labels: list[list[str]] = []
    for members in groups.values():
        m = len(members)
        pairs = [
            (a, b)
            for a in range(m)
            for b in range(a + 1, m)
            if pattern[members[a], members[b]]
        ]
        group_labels: list[str] = []
        if len(pairs) == m * (m - 1) // 2:
            # Cholesky: for each column j, diagonal j then rows i > j.
            for j in range(m):
                for i in range(j, m):
                    name = names[members[i]]
                    if i == j:
                        group_labels.append(f"sd({name})")
                    else:
                        group_labels.append(f"omega({name}, {names[members[j]]})")
        else:
            group_labels.extend(f"sd({names[i]})" for i in members)
            group_labels.extend(
                f"omega({names[members[a]]}, {names[members[b]]})" for a, b in pairs
            )
        labels.append(group_labels)
    return labels


def _flat_size(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(np.asarray(value).reshape(-1).size)
    except Exception:
        return 0


def _error_labels(result: Any, n: int) -> list[str]:
    """Labels for the residual-error and extra parameters after Omega."""
    from ..saem._utils import ERROR_PARS

    raw = getattr(result, "res", None)
    model = getattr(result, "model", None)
    errors = [str(e) for e in (getattr(model, "errors", None) or [])]
    error_pars = getattr(raw, "error_pars", None) or []
    labels: list[str] = []
    for r, ep in enumerate(error_pars):
        kind = errors[min(r, len(errors) - 1)] if errors else "constant"
        for pname in ERROR_PARS.get(kind, ("a",)):
            suffix = f"[{r + 1}]" if len(error_pars) > 1 else ""
            labels.append(f"sigma.{pname}{suffix}")
    for k in range(_flat_size(getattr(raw, "extra", None))):
        labels.append(f"extra{k}")
    return labels[:n] if n >= 0 else labels


def _trace_labels(result: Any, n_params: int) -> list[str]:
    """Human-readable names for every column of the SAEM theta trace."""
    raw = getattr(result, "res", None)
    names = [str(n) for n in (getattr(raw, "coef_names", []) or [])]
    if len(names) >= n_params:
        return names[:n_params]

    model = getattr(result, "model", None)
    tail: list[str] = []
    omega_groups = _omega_label_groups(model) or []
    for group in omega_groups:
        tail.extend(group)

    tail.extend(_error_labels(result, -1))

    if len(names) + len(tail) == n_params:
        return names + tail

    # Unknown layout: keep the fixed-effect names and fall back for the rest.
    return names + [
        tail[j - len(names)] if j - len(names) < len(tail) else f"Param {j}"
        for j in range(len(names), n_params)
    ]


def _saem_k1(result: Any, n_iter: int) -> int | None:
    """Effective end of the stochastic exploration phase (K1)."""
    control = getattr(result, "control", None)
    try:
        k1 = int(getattr(control, "step_k1", -1))
    except (TypeError, ValueError):
        return None
    if k1 < 0:
        k1 = min(int(0.6 * n_iter), 300)
    return k1 if 0 < k1 < n_iter else None


def _saem_title(result: Any, title: str | None) -> str:
    if title:
        return title
    name = getattr(getattr(result, "model", None), "name", None)
    name = str(name).strip() if name else ""
    if name and name.lower() not in {"saem", "saemfit", "model"}:
        return f"SAEM convergence — {name}"
    return "SAEM convergence"


def saem_trace(
    result: Any,
    ax=None,
    *,
    max_params: int = 12,
    figsize=None,
    title: str | None = None,
):
    """SAEM parameter traces, one panel per parameter.

    The vertical dashed line marks the end of the stochastic exploration phase
    (``K1``), where the Robbins-Monro smoothing begins.  If the fit did not store
    a trace, the fixed-effect forest plot is returned instead.
    """
    resolve(result)
    raw = getattr(result, "res", None)
    trace = getattr(raw, "theta_trace", None)
    matrix = _trace_matrix(trace)
    if matrix.size == 0:
        from .coefficients import forest_plot

        return forest_plot(result, ax=ax, title=title)

    n_iter, n_params = matrix.shape
    names = _trace_labels(result, n_params)
    shown = min(n_params, max_params)
    n_cols = min(3, shown)
    n_rows = int(np.ceil(shown / n_cols))
    if figsize is None:
        figsize = (4.4 * n_cols, 3.0 * n_rows)
    fig, axes = panel_axes(ax, n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()

    k1 = _saem_k1(result, n_iter)
    iters = np.arange(1, n_iter + 1)
    bottom_panels = {j % n_cols: j for j in range(shown)}

    label = _saem_title(result, title)
    if shown < n_params:
        label = f"{label} (showing {shown} of {n_params})"

    for j in range(shown):
        panel = axes[j]
        panel.plot(iters, matrix[:, j], color="#1e3a8a", linewidth=1.5)
        if k1 is not None:
            panel.axvline(
                k1,
                color=DEFAULT_COLORS["accent"],
                linestyle="--",
                linewidth=1.1,
                label="End of exploration (K1)" if j == 0 else None,
            )
        panel.set_title(names[j], fontsize=9.5)
        if j in bottom_panels.values():
            panel.set_xlabel("SAEM iteration", fontsize=8.5)
        style_axis(panel)
        if j == 0 and k1 is not None:
            panel.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    for j in range(shown, n_rows * n_cols):
        axes[j].set_axis_off()
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)



def mcglm_convergence(
    result: Any,
    ax=None,
    *,
    which: str = "cov",
    figsize=(12.0, 4.6),
    title: str | None = None,
):
    """Parameter trajectories and score norms of the MCGLM algorithm."""
    data = resolve(result)
    history = getattr(result, "history", None)
    if not isinstance(history, dict) or not history:
        raise ValueError(f"convergence history is not available for {data.model_name}")

    beta = np.asarray(history.get("beta", np.empty((0, 0))), dtype=float)
    cov = np.asarray(history.get("cov", np.empty((0, 0))), dtype=float)
    n_iter = max(beta.shape[0] if beta.ndim == 2 else 0, cov.shape[0] if cov.ndim == 2 else 0)
    if n_iter == 0:
        raise ValueError("the convergence history is empty")
    iters = np.arange(n_iter)

    param_names = [str(n) for n in (getattr(result, "param_names", None) or [])]
    n_beta = int(np.asarray(getattr(result, "beta", np.empty(0))).size)
    beta_names = param_names[:n_beta] or [f"β{j}" for j in range(beta.shape[1] if beta.ndim == 2 else 0)]
    cov_names = param_names[n_beta:] or [f"cov[{j}]" for j in range(cov.shape[1] if cov.ndim == 2 else 0)]

    fig, axes = panel_axes(ax, 1, 2, figsize=figsize)
    left, right = axes.ravel()

    if which == "beta":
        block, names = beta, beta_names
    else:
        block, names = (cov, cov_names) if cov.size else (beta, beta_names)
    block = np.asarray(block, dtype=float)
    if block.ndim == 1:
        block = block.reshape(-1, 1)
    for j in range(block.shape[1]):
        left.plot(iters[: block.shape[0]], block[:, j], marker="o", markersize=3, linewidth=1.6, label=names[j] if j < len(names) else f"p{j}")
    left.set_xlabel("Iteration")
    left.set_ylabel("Parameter value")
    left.set_title("Parameter trajectories (ρ, power, τ)" if which != "beta" else "Regression coefficients")
    if block.shape[1] <= 10:
        left.legend(loc="best", fontsize=8)
    style_axis(left)

    import warnings

    score_beta = np.asarray(history.get("score_beta", np.empty((0, 0))), dtype=float)
    score_cov = np.asarray(history.get("score_cov", np.empty((0, 0))), dtype=float)
    if score_beta.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN history rows
            sb = np.nanmax(np.abs(score_beta), axis=1)
        right.plot(np.arange(sb.size), np.maximum(np.nan_to_num(sb, nan=1e-16), 1e-16), "s-", color="#1e3a8a", linewidth=1.7, label="Max |score β|")
    if score_cov.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sc = np.nanmax(np.abs(score_cov), axis=1)
        right.plot(np.arange(sc.size), np.maximum(np.nan_to_num(sc, nan=1e-16), 1e-16), "^-", color=DEFAULT_COLORS["accent"], linewidth=1.7, label="Max |score cov|")
    options = getattr(result, "fit_options", {}) or {}
    tol = options.get("tol")
    if tol is not None:
        right.axhline(float(tol), color=DEFAULT_COLORS["neutral"], linestyle="--", linewidth=1.1, label=f"Tolerance ({float(tol):.0e})")
    right.set_yscale("log")
    right.set_xlabel("Iteration")
    right.set_ylabel("Max |score| (log scale)")
    right.set_title(f"Score convergence (converged: {bool(getattr(result, 'converged', False))})")
    right.legend(loc="best", fontsize=8)
    style_axis(right)

    label = title or f"Convergence — {data.model_name}"
    fig.suptitle(label, fontsize=13)
    return PlotResult(fig, label)


def convergence(result: Any, ax=None, *, figsize=(12.0, 4.6), title: str | None = None, **kwargs):
    """Dispatch to the convergence view that matches the fitted result."""
    data = resolve(result)
    raw = getattr(result, "res", None)
    if getattr(raw, "theta_trace", None):
        return saem_trace(result, ax=ax, figsize=figsize, title=title, **kwargs)
    if getattr(result, "history", None):
        return mcglm_convergence(result, ax=ax, figsize=figsize, title=title, **kwargs)

    # Generic history attributes used by other iterative fits.
    for attribute in ("history", "trace", "iter_params", "iter_fixef", "theta_hist", "chains"):
        value = getattr(result, attribute, None)
        if value is None:
            continue
        matrix = _trace_matrix(value)
        if matrix.size == 0:
            continue
        fig, panel, _ = single_axis(ax, figsize)
        iters = np.arange(1, matrix.shape[0] + 1)
        for j in range(matrix.shape[1]):
            panel.plot(iters, matrix[:, j], linewidth=1.5, label=f"Param {j}")
        panel.set_xlabel("Iteration")
        panel.set_ylabel("Value")
        panel.legend(loc="best", fontsize=8)
        style_axis(panel)
        label = title or f"Convergence — {data.model_name}"
        panel.set_title(label)
        return PlotResult(fig, label)

    raise ValueError(
        f"{data.model_name} does not store a convergence history; "
        "use its coefficient plot to inspect the fitted estimates."
    )
