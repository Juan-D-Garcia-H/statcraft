# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Plain-text table rendering (the statsmodels look: boxed tables made
of ``=``/``-`` rules and fixed-width, right-aligned numeric columns).
"""

from __future__ import annotations

import numpy as np

from ._compat import DIST_FUNCS
from .blocks import CoefBlock, TableBlock
from .layout import col_width

__all__ = ["two_col_text", "coef_table_text", "block_table_text", "table_block_text"]


def two_col_text(left: list, right: list) -> tuple[str, int]:
    """Text rendering of the statsmodels-style two-column key/value
    header block (Dep. Variable / Model / ... on the left, Log-Likelihood
    / AIC / ... on the right). Returns (text, width).
    """
    ll_w = col_width([lk for lk, _ in left], min_w=17)
    lv_w = col_width([lv for _, lv in left], min_w=10)
    if right:
        rl_w = col_width([rk for rk, _ in right], min_w=17)
        rv_w = col_width([rv for _, rv in right], min_w=10)
    else:
        rl_w = rv_w = 0
    gap = 3 if right else 0

    lines = []
    for i in range(max(len(left), len(right))):
        lk, lv = left[i] if i < len(left) else ("", "")
        left_part = f"{lk:<{ll_w}}{lv:>{lv_w}}"
        if right:
            rk, rv = right[i] if i < len(right) else ("", "")
            right_part = f"{rk:<{rl_w}}{rv:>{rv_w}}"
            lines.append(f"{left_part}{' ' * gap}{right_part}")
        else:
            lines.append(left_part)
    return "\n".join(lines), ll_w + lv_w + gap + rl_w + rv_w


def coef_table_text(
    block: CoefBlock, alpha: float, dist: str, df: float | None, stars: np.ndarray | None = None
) -> tuple[list[str], int]:
    """Text rows for one coefficient table (header row + one row per
    coefficient). The name column is sized to the longest name in this
    block, so long variable/category names are never cut off. When
    `stars` is given (one marker per row, see :mod:`.significance`),
    it's appended directly to the coef value ("0.7344***"), the
    convention in most published regression tables, rather than as its
    own column. Returns (lines, width).
    """
    names, coefs, se = block.as_arrays()
    name_w = col_width(names, min_w=10, pad=2)
    # Room for the widest possible marker ("***") so the coef column
    # stays aligned whether or not a given row's marker is shorter.
    star_w = max((len(s) for s in stars), default=0) if stars is not None else 0

    if se is None:
        col_w = 12 + star_w
        header = f"{'':<{name_w}}{'coef':>{col_w}}"
        lines = [header]
        for i, (name, c) in enumerate(zip(names, coefs)):
            cell = f"{c:.4f}{stars[i] if stars is not None else ''}"
            lines.append(f"{name:<{name_w}}{cell:>{col_w}}")
        return lines, len(header)

    df = float(df) if df is not None else float("nan")  # normal family ignores df
    stat_label, ppf_fn, sf_fn = DIST_FUNCS[dist]
    crit = ppf_fn(1.0 - alpha / 2.0, df)
    lo_label = f"[{alpha / 2 * 100:.1f}%"
    hi_label = f"{(1 - alpha / 2) * 100:.1f}%]"
    c1, c2 = 11, 12
    coef_w = c1 + star_w
    header = (
        f"{'':<{name_w}}{'coef':>{coef_w}}{'std err':>{c1}}{stat_label:>{c1}}"
        f"{'P>|' + stat_label + '|':>{c1}}{lo_label:>{c2}}{hi_label:>{c2}}"
    )

    # Vectorized over the whole block in one pass (see _compat.t_sf) --
    # a single scipy call per column instead of one per coefficient,
    # which matters once a model has dozens/hundreds of parameters.
    valid = np.isfinite(se) & (se > 0)
    stat = np.where(valid, coefs / np.where(valid, se, 1.0), np.nan)
    pvalue = np.where(np.isfinite(stat), 2.0 * sf_fn(np.abs(stat), df), np.nan)
    lo, hi = coefs - crit * se, coefs + crit * se

    lines = [header]
    for i, name in enumerate(names):
        coef_cell = f"{coefs[i]:.4f}{stars[i] if stars is not None else ''}"
        lines.append(
            f"{name:<{name_w}}{coef_cell:>{coef_w}}{se[i]:>{c1}.4f}{stat[i]:>{c1}.3f}"
            f"{pvalue[i]:>{c1}.3f}{lo[i]:>{c2}.4f}{hi[i]:>{c2}.4f}"
        )
    return lines, len(header)


def block_table_text(
    block: CoefBlock, alpha: float, dist: str, df: float | None, stars: np.ndarray | None = None
) -> tuple[str, int]:
    """Boxed text rendering of one coefficient block: optional heading,
    then a statsmodels-style bordered table (rule / header / rule /
    rows / rule). Returns (text, width).
    """
    lines, width = coef_table_text(block, alpha, dist=dist, df=df, stars=stars)
    dash = "-" * width
    out = []
    if block.heading:
        out.append(f"{block.heading}:")
    out.append(dash)
    out.append(lines[0])
    out.append(dash)
    out.extend(lines[1:])
    out.append(dash)
    return "\n".join(out), width


def _aligned(text: str, width: int, align: str) -> str:
    if align == "left":
        return f"{text:<{width}}"
    if align == "center":
        return f"{text:^{width}}"
    return f"{text:>{width}}"


def table_block_text(block: TableBlock) -> tuple[str, int]:
    """Boxed text rendering of one :class:`TableBlock`."""
    n = len(block.columns)
    label_w = (
        col_width([block.index_name] + [block.label_text(i) for i in range(len(block.rows))], min_w=5, pad=2)
        if block.index is not None
        else 0
    )
    col_ws = []
    for j in range(n):
        cells = [block.header_text(j)] + [block.cell_text(i, j) for i in range(len(block.rows))]
        col_ws.append(col_width(cells, min_w=7, pad=2))

    header = (f"{block.index_name:<{label_w}}" if block.index is not None else "")
    header += "".join(
        _aligned(block.header_text(j), col_ws[j], block.column_align(j)) for j in range(n)
    )
    lines = [header]
    for i in range(len(block.rows)):
        line = f"{block.label_text(i):<{label_w}}" if block.index is not None else ""
        line += "".join(
            _aligned(block.cell_text(i, j), col_ws[j], block.column_align(j)) for j in range(n)
        )
        lines.append(line)

    width = len(lines[0])
    dash = "-" * width
    out = []
    if block.heading:
        out.append(f"{block.heading}:")
    out.append(dash)
    out.append(lines[0])
    out.append(dash)
    out.extend(lines[1:])
    out.append(dash)
    return "\n".join(out), width
