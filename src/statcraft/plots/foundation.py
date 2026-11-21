# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Shared building blocks for :mod:`statcraft.plots`.

Matplotlib is an optional dependency: importing :mod:`statcraft.plots` never
imports it, and every function that needs a figure calls :func:`pyplot` first.
That keeps the core statistical engines usable without the ``viz`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "PlotResult",
    "pyplot",
    "new_figure",
    "new_axes",
    "as_float",
    "finite",
    "clean",
    "qq_points",
    "annotate_points",
    "colorbar",
    "safe_label",
    "DEFAULT_COLORS",
]


DEFAULT_COLORS = {
    "observed": "#1f4e79",
    "fitted": "#d97706",
    "residual": "#b45309",
    "accent": "#b91c1c",
    "reference": "#475569",
    "neutral": "#64748b",
    "band": "#0f766e",
    "grid": "#cbd5e1",
}


def pyplot():
    """Import and return ``matplotlib.pyplot`` with a helpful error message."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise ModuleNotFoundError(
            "statcraft.plots needs matplotlib. Install it with "
            '`pip install "statcraft[viz]"` (or `pip install matplotlib`).'
        ) from exc
    return plt


def _close_pending(fig) -> None:
    """Best-effort close of a figure after a save-only request."""
    try:
        pyplot().close(fig)
    except Exception:  # pragma: no cover - never mask the caller's error
        pass


@dataclass
class PlotResult:
    """A rendered figure together with a small, explicit lifetime API.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure that owns the panels.
    title : str, optional
        A short label used by :meth:`__repr__`.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The underlying Matplotlib figure.
    axes : list of matplotlib.axes.Axes
        All axes in the figure (a single-element list for one-panel plots).
    ax : matplotlib.axes.Axes
        The first axis, for convenience with one-panel plots.
    """

    fig: Any
    title: str = ""

    @property
    def axes(self) -> list[Any]:
        """All axes in the figure, in creation order."""
        return list(self.fig.axes)

    @property
    def ax(self):
        """The first axis (convenient for single-panel plots)."""
        axes = self.axes
        if not axes:
            raise RuntimeError("the figure has no axes")
        return axes[0]

    def show(self, block: bool | None = None):
        """Display the figure with the active Matplotlib backend."""
        plt = pyplot()
        if block is None:
            plt.show()
        else:
            plt.show(block=block)
        return self

    def save(self, path: str | Path, dpi: int | float | None = None, **kwargs):
        """Write the figure to ``path`` and return the resolved path.

        ``bbox_inches="tight"`` is used by default unless the caller passes
        one explicitly.
        """
        kwargs.setdefault("bbox_inches", "tight")
        if dpi is not None:
            kwargs["dpi"] = dpi
        out = Path(path)
        if out.parent and str(out.parent) not in ("", "."):
            out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, **kwargs)
        return out

    savefig = save

    def close(self):
        """Close the figure and release its memory."""
        pyplot().close(self.fig)
        return None

    def __enter__(self) -> "PlotResult":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        label = f" {self.title!r}" if self.title else ""
        return f"<PlotResult{label} axes={len(self.axes)}>"


# ---------------------------------------------------------------------------
# Figure construction and small numerical helpers
# ---------------------------------------------------------------------------


def new_figure(figsize: tuple[float, float] | None = None):
    """Create a figure with a single axis."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def new_axes(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    squeeze: bool = False,
):
    """Create a figure and an array of axes with constrained layout."""
    plt = pyplot()
    try:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=squeeze, layout="constrained")
    except TypeError:  # pragma: no cover - matplotlib < 3.6
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=squeeze)
        fig.tight_layout()
    return fig, axes


def as_float(values: Any, *, name: str = "values") -> np.ndarray | None:
    """Convert ``values`` to a 1-D float array, preserving ``None``."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def finite(*arrays: Any) -> np.ndarray:
    """Boolean mask selecting rows where every supplied array is finite."""
    mask = None
    for value in arrays:
        arr = np.asarray(value, dtype=float).reshape(-1)
        ok = np.isfinite(arr)
        mask = ok if mask is None else (mask & ok)
    if mask is None:
        return np.ones(0, dtype=bool)
    return mask


def clean(*arrays: Any) -> tuple[np.ndarray, ...]:
    """Drop rows with non-finite values from all arrays in lockstep."""
    mask = finite(*arrays)
    return tuple(np.asarray(a, dtype=float).reshape(-1)[mask] for a in arrays)


def qq_points(values: Iterable[float]):
    """Theoretical normal quantiles and sorted standardised values.

    Uses the Blom plotting positions ``(i - 3/8) / (n + 1/4)``, which are the
    positions R's ``qqnorm`` uses and are well behaved for small samples.
    """
    from scipy import stats

    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, arr
    probs = (np.arange(1, arr.size + 1) - 0.375) / (arr.size + 0.25)
    return stats.norm.ppf(probs), np.sort(arr)


def annotate_points(
    ax,
    x,
    y,
    labels: Sequence[Any],
    *,
    max_labels: int = 5,
    color: str | None = None,
    stagger: bool = True,
):
    """Annotate up to ``max_labels`` points without stacking nearby labels.

    Influential rows tend to sit in the same corner of an influence plot, so a
    fixed offset often makes their labels collide.  ``stagger`` first chooses
    from a few vertical offsets, then checks the *rendered* label boxes and
    nudges any remaining collisions apart.  The check uses display pixels, so
    it follows the actual figure size and font.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = list(labels)
    placed: list[tuple[float, float, float]] = []
    annotations: list[tuple[Any, float]] = []
    candidates = (4.0, -14.0, 18.0, -28.0, 30.0, -42.0)
    scale = float(getattr(ax.figure, "dpi", 100.0)) / 72.0
    seen = 0
    for xi, yi, label in zip(x, y, labels):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        if seen >= max_labels:
            break
        dy = candidates[0]
        if stagger:
            try:
                px, py = ax.transData.transform((xi, yi))
                for candidate in candidates:
                    if all(
                        abs(px - qx) >= 60.0
                        or abs((py + candidate * scale) - (qy + qdy * scale)) >= 13.0
                        for qx, qy, qdy in placed
                    ):
                        dy = candidate
                        break
                placed.append((px, py, dy))
            except Exception:  # pragma: no cover - unusual axes transforms
                pass
        ann = ax.annotate(
            str(label),
            (xi, yi),
            textcoords="offset points",
            xytext=(5, dy),
            fontsize=8.5,
            color=color or DEFAULT_COLORS["accent"],
        )
        # A point can sit at the edge of the axes.  Its label belongs outside
        # the data rectangle, and clipping it there is worse than letting it
        # use the panel margin reserved by the influence plots.
        ann.set_clip_on(False)
        annotations.append((ann, dy))
        seen += 1

    if not stagger or len(annotations) < 2:
        return
    # The point-based guess above is cheap but approximate: label widths and
    # the exact baseline depend on the renderer.  Resolve any real overlap by
    # moving the later label vertically, alternating above/below with growing
    # steps so a cluster of three or more labels also separates.
    try:
        canvas = ax.figure.canvas
        canvas.draw()
        renderer = canvas.get_renderer()
        for i, (ann, base_dy) in enumerate(annotations):
            for attempt in range(9):
                boxes = [a.get_window_extent(renderer) for a, _ in annotations[:i]]
                if not any(ann.get_window_extent(renderer).overlaps(b) for b in boxes):
                    break
                step = 16.0 * ((attempt // 2) + 1) * (1 if attempt % 2 == 0 else -1)
                ann.set_position((5, base_dy + step))
                canvas.draw()
    except Exception:  # pragma: no cover - headless renderers without draw()
        pass


def colorbar(fig, mappable, ax, label: str | None = None):
    """Attach a colorbar without relying on the global pyplot state."""
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label)
    return cb


def safe_label(*candidates: Any, default: str = "value") -> str:
    """Return the first non-empty candidate rendered as a short label."""
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return default


def single_axis(ax, figsize: tuple[float, float] | None = None):
    """Return ``(fig, ax, created)`` for a one-panel plot."""
    if ax is None:
        fig, ax = new_figure(figsize)
        return fig, ax, True
    return ax.figure, ax, False


def panel_axes(ax, nrows: int, ncols: int, figsize: tuple[float, float] | None = None):
    """Return ``(fig, axes)`` for a multi-panel plot.

    ``ax`` may be ``None`` (a new figure is created), a flat/nested sequence of
    exactly ``nrows * ncols`` axes, or a single axes (which is rejected with a
    clear message rather than silently dropping panels).
    """
    if ax is None:
        return new_axes(nrows, ncols, figsize=figsize, squeeze=False)
    arr = np.asarray(ax, dtype=object)
    if arr.size == 1:
        raise ValueError(
            "This plot has several panels; pass an array of axes or omit `ax` "
            "to create a new figure."
        )
    if arr.size != nrows * ncols:
        raise ValueError(f"expected {nrows * ncols} axes, got {arr.size}")
    axes = arr.reshape(nrows, ncols)
    return axes.flat[0].figure, axes


def style_axis(ax, *, grid: bool = True, zero_line: bool = False):
    """Apply the package's small, consistent axis defaults."""
    if grid:
        ax.grid(alpha=0.28, linewidth=0.8, color=DEFAULT_COLORS["grid"])
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)
    return ax
