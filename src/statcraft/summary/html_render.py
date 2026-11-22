# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""HTML table rendering, with the CSS that gives summaries their look
in notebooks and other rich displays.
"""

from __future__ import annotations

import numpy as np

from ._compat import DIST_FUNCS
from .blocks import CoefBlock, TableBlock

__all__ = ["esc_html", "html_two_col_table", "html_coef_table", "html_table_block", "CSS"]


def esc_html(s) -> str:
    """Minimal HTML-escaping for untrusted text going into attributes
    or element content (labels, names, notes -- anything that ultimately
    comes from the caller's data, not from this module).
    """
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def html_two_col_table(left: list, right: list) -> str:
    """HTML for the statsmodels-style top info block: the two-column
    key/value layout (e.g. "Dep. Variable" / "No. Observations"), left
    and right columns side by side in one table."""
    rows = []
    for i in range(max(len(left), len(right))):
        lk, lv = left[i] if i < len(left) else ("", "")
        rk, rv = right[i] if i < len(right) else ("", "")
        rows.append(
            f"<tr><td class='ss-lbl'>{esc_html(lk)}</td><td class='ss-val'>{esc_html(lv)}</td>"
            f"<td class='ss-gap'></td>"
            f"<td class='ss-lbl'>{esc_html(rk)}</td><td class='ss-val'>{esc_html(rv)}</td></tr>"
        )
    return "<table class='ss-info'><tbody>" + "".join(rows) + "</tbody></table>"


def html_coef_table(
    block: CoefBlock, alpha: float, dist: str, df: float | None, stars: np.ndarray | None = None
) -> str:
    """HTML rendering of one coefficient table. When `stars` is given
    (one marker per row, see :mod:`.significance`), it's appended
    directly inside the coef cell ("0.7344<b>***</b>"), matching the
    text renderer's convention rather than adding its own column.
    """
    names, coefs, se = block.as_arrays()

    def _coef_cell(i: int, c: float) -> str:
        marker = f"<b>{esc_html(stars[i])}</b>" if stars is not None else ""
        return f"<td>{c:.4f}{marker}</td>"

    if se is None:
        rows_html = "".join(
            f"<tr><td>{esc_html(name)}</td>{_coef_cell(i, c)}</tr>"
            for i, (name, c) in enumerate(zip(names, coefs))
        )
        return (
            "<table class='ss-coef'><thead><tr><th></th><th>coef</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    df = float(df) if df is not None else float("nan")  # normal family ignores df
    stat_label, ppf_fn, sf_fn = DIST_FUNCS[dist]
    crit = ppf_fn(1.0 - alpha / 2.0, df)
    lo_label = f"[{alpha / 2 * 100:.1f}%"
    hi_label = f"{(1 - alpha / 2) * 100:.1f}%]"

    # Vectorized over the whole block in one pass (see _compat.t_sf) --
    # a single scipy call per column instead of one per coefficient.
    valid = np.isfinite(se) & (se > 0)
    stat = np.where(valid, coefs / np.where(valid, se, 1.0), np.nan)
    pvalue = np.where(np.isfinite(stat), 2.0 * sf_fn(np.abs(stat), df), np.nan)
    lo, hi = coefs - crit * se, coefs + crit * se

    rows = []
    for i, name in enumerate(names):
        rows.append(
            f"<tr><td>{esc_html(name)}</td>{_coef_cell(i, coefs[i])}<td>{se[i]:.4f}</td>"
            f"<td>{stat[i]:.3f}</td><td>{pvalue[i]:.3f}</td><td>{lo[i]:.4f}</td><td>{hi[i]:.4f}</td></tr>"
        )
    return (
        "<table class='ss-coef'><thead><tr>"
        f"<th></th><th>coef</th><th>std err</th><th>{esc_html(stat_label)}</th>"
        f"<th>P&gt;|{esc_html(stat_label)}|</th>"
        f"<th>{esc_html(lo_label)}</th><th>{esc_html(hi_label)}</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def html_table_block(block: TableBlock) -> str:
    """HTML rendering of one :class:`TableBlock`."""
    n = len(block.columns)
    head_cells = ""
    if block.index is not None:
        head_cells += f"<th>{esc_html(block.index_name)}</th>"
    head_cells += "".join(f"<th>{esc_html(block.header_text(j))}</th>" for j in range(n))

    rows = []
    for i in range(len(block.rows)):
        cells = ""
        if block.index is not None:
            cells += f"<td style='text-align:left'>{esc_html(block.label_text(i))}</td>"
        for j in range(n):
            align = block.column_align(j)
            cells += f"<td style='text-align:{align}'>{esc_html(block.cell_text(i, j))}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<table class='ss-coef'><thead><tr>"
        + head_cells
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


CSS = (
    "<style>"
    ".ss-summary{"
    "  display:inline-block;"
    "  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;"
    "  font-size:15px; line-height:1.45; color:inherit;"
    "}"
    ".ss-summary h3{"
    "  font-size:1.15em; font-weight:700; text-align:left; letter-spacing:0.01em;"
    "  margin:0 0 0.2em 0;"
    "}"
    ".ss-summary h4{"
    "  font-size:1em; font-weight:700; margin:1.1em 0 0.35em 0; color:inherit;"
    "}"
    ".ss-summary .ss-formula{"
    "  text-align:left; font-style:italic; opacity:0.7; margin:0 0 0.9em 0; font-size:0.92em;"
    "}"
    # No fixed width and no fill/zebra on any row: each table is only as wide as its
    # own content needs (like statsmodels' SimpleTable). Everything here -- title,
    # subtitle, every table -- is left-aligned (statsmodels' own convention),
    # rather than centered: centering the title while a narrower table (the info
    # block, once a coefficient block widens from long variable names) stays
    # flush left made the two look misaligned. Left-aligning both sides of that
    # comparison removes the mismatch instead of trying to force one to match
    # the other's alignment.
    ".ss-summary table{border-collapse:collapse; margin:0 0 0.7em 0;}"
    ".ss-summary table td, .ss-summary table th{font-variant-numeric:tabular-nums;}"
    ".ss-summary table.ss-info{border-top:1px solid currentColor; border-bottom:1px solid currentColor;}"
    ".ss-summary table.ss-info td{padding:4px 10px; font-size:1em;}"
    ".ss-summary table.ss-info td.ss-lbl{text-align:left; white-space:nowrap;}"
    ".ss-summary table.ss-info td.ss-val{text-align:right; white-space:nowrap;}"
    ".ss-summary table.ss-info td.ss-gap{width:2.5em;}"
    ".ss-summary table.ss-coef{border-top:1px solid currentColor; border-bottom:1px solid currentColor;}"
    ".ss-summary table.ss-coef th{"
    "  border-bottom:1px solid currentColor; padding:6px 14px; text-align:right; font-weight:600; font-size:1em; white-space:nowrap;"
    "}"
    ".ss-summary table.ss-coef td{padding:5px 14px; text-align:right; font-size:1em; white-space:nowrap;}"
    ".ss-summary table.ss-coef td:first-child, .ss-summary table.ss-coef th:first-child{text-align:left;}"
    ".ss-summary .ss-extra{opacity:0.8; margin:0.25em 0; font-size:0.95em;}"
    ".ss-summary .ss-notes{opacity:0.75; font-size:0.9em; margin-top:0.7em;}"
    ".ss-summary .ss-notes ol{margin:0.2em 0 0 1.2em; padding:0;}"
    "</style>"
)
