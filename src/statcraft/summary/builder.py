# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""build_summary -- the package's single public entry point."""

from __future__ import annotations

import logging
import textwrap
from typing import Any, Mapping, Sequence

from .blocks import CoefBlock, TableBlock
from .exceptions import SummaryValidationError
from .frame import coef_block_frame, info_series, table_block_frame
from .html_render import CSS, esc_html, html_coef_table, html_table_block, html_two_col_table
from .significance import legend_for
from .tables import Summary, SummaryTable
from .text_render import block_table_text, table_block_text, two_col_text

logger = logging.getLogger(__name__)

__all__ = ["build_summary"]

_MIN_WIDTH = 78
_VALID_DISTS = ("normal", "t")


def _normalize_blocks(
    names,
    coefs,
    se,
    blocks: Sequence[Mapping[str, Any] | CoefBlock] | None,
    tables: Sequence[Mapping[str, Any] | TableBlock] | None = None,
) -> list[CoefBlock]:
    if blocks:
        return [b if isinstance(b, CoefBlock) else CoefBlock.from_dict(b) for b in blocks]
    if names is None or coefs is None:
        if tables:
            return []
        raise SummaryValidationError(
            "build_summary needs names/coefs (with optional se), blocks, or tables."
        )
    return [CoefBlock(names=names, coefs=coefs, se=se)]


def _normalize_tables(
    tables: Sequence[Mapping[str, Any] | TableBlock] | None,
) -> list[TableBlock]:
    return [t if isinstance(t, TableBlock) else TableBlock.from_dict(t) for t in tables or []]


def _validate_top_level(dist: str, alpha: float) -> None:
    if dist not in _VALID_DISTS:
        raise SummaryValidationError(f"dist must be one of {_VALID_DISTS!r}, got {dist!r}")
    if not (0.0 < alpha < 1.0):
        raise SummaryValidationError(f"alpha must be between 0 and 1, got {alpha!r}")


def build_summary(
    title: str,
    subtitle: str | Sequence[str] | None = None,
    left: list[tuple[str, str]] | None = None,
    right: list[tuple[str, str]] | None = None,
    names: Sequence[str] | None = None,
    coefs: Sequence[float] | None = None,
    se: Sequence[float] | None = None,
    blocks: Sequence[Mapping[str, Any] | CoefBlock] | None = None,
    tables: Sequence[Mapping[str, Any] | TableBlock] | None = None,
    diag_left: list[tuple[str, str]] | None = None,
    diag_right: list[tuple[str, str]] | None = None,
    dist: str = "normal",
    df: float | None = None,
    extra_lines: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
    notes: list[str] | None = None,
    stars: bool | dict[float, str] = False,
) -> Summary:
    """Build a statsmodels-style results summary from plain Python/NumPy
    data -- works for any fitted model, not just GLM.

    Parameters
    ----------
    title : str
        Report title, e.g. "Poisson Regression Results".
    subtitle : str or list of str, optional
        Line(s) shown under the title (formula, method/fit options,
        shape, ...). A list renders one line per entry (e.g. a formula
        line and a separate method line), each wrapped independently so
        they never run together.
    left, right : list of (str, str), optional
        Key/value pairs for the two-column header block (Dep. Variable,
        Model, No. Observations, ... on the left; Log-Likelihood, AIC,
        Converged, ... on the right). Either side can be shorter than
        the other or omitted entirely.
    names, coefs, se : optional
        Convenience path for a model with a single coefficient block:
        `names` is the list of coefficient/variable names, `coefs` the
        point estimates, and `se` the standard errors (or None to render
        a coef-only column with no stat/P-value/CI columns -- e.g. when
        SEs weren't computed). Ignored if `blocks` is given.
    blocks : list of dict or CoefBlock, optional
        For a model with more than one coefficient block (count + zero
        model, mean + precision, mu/sigma/Q, ...). May be omitted when
        `tables` are supplied (e.g. a list of per-group fits). Each entry is a
        :class:`~statcraft.utils.summary.blocks.CoefBlock`, or a dict with keys
        "heading" (str or None), "names", "coefs", and "se" (array-like
        or None) -- same meaning as the names/coefs/se path above. A
        block can also set its own "dist"/"df" to override the
        summary-wide `dist`/`df` for just that block (rarely needed --
        usually one fit uses one test statistic throughout).
    tables : list of dict or TableBlock, optional
        Free-form tables rendered after the coefficient block(s), each with
        arbitrary columns (for example the variance-components table of a
        mixed model: Group / Name / Variance / Std.Dev. / Corr). A dict has
        the keys ``columns``, ``rows`` and optionally ``heading``, ``index``,
        ``index_name``, ``align`` and ``formats``.
    diag_left, diag_right : list of (str, str), optional
        Optional second two-column block (residual diagnostics,
        condition number, ...), rendered after the coefficient table(s)
        and before the footer notes -- e.g. statsmodels' OLS
        Omnibus/Skew/Durbin-Watson/Cond. No. block. Gets its own entry
        in `summary.tables` when given.
    dist : {"normal", "t"}, default "normal"
        Which test statistic/column the coefficient table(s) report:
        "normal" gives a z-stat and "P>|z|" (asymptotic inference, as in
        most GLMs); "t" gives a t-stat and "P>|t|" using `df` residual
        degrees of freedom (finite-sample inference, as in NLS/OLS).
        Also used for the CI columns. Can be overridden per block (see
        `blocks` above).
    df : int or float, optional
        Residual degrees of freedom for the t distribution. Required
        when `dist="t"` (either here or per-block) and `se` is given.
    extra_lines : list of (str, str), optional
        Extra label/value lines shown in the footer, below the tables
        (e.g. "Rank-deficient columns dropped: ...").
    alpha : float, default 0.05
        Confidence level for the CI columns, only used where `se` is
        given (e.g. alpha=0.05 -> "[2.5%" / "97.5%]"). Must be in (0, 1).
    notes : list of str, optional
        Numbered footnotes shown at the bottom (convergence warnings,
        "no standard errors computed", caveats about the fit, ...).
    stars : bool or dict[float, str], default False
        Significance stars next to each coefficient's value (e.g.
        "0.7344***"), the standard shorthand from published regression
        tables. ``True`` uses the usual thresholds (``{0.01: "***",
        0.05: "**", 0.1: "*"}``); pass a dict to use different cutoffs
        or markers. A legend footnote is added automatically. Has no
        effect on a block with no standard errors (there's no p-value
        to threshold). The markers also land in `.tables[i].data["sig"]`
        alongside the numeric columns, so `frame[frame.sig != ""]` still
        works for programmatic filtering.

    Returns
    -------
    Summary
        `str(summary)`/`print(summary)` gives the plain-text report; in a
        notebook it renders as styled HTML. `summary.tables` holds each
        individual table (header block first, then one per coefficient
        block, then the diagnostics block if given) so a piece can be
        pulled out and reused on its own. Each table also carries its
        data as a real pandas object in `.tables[i].data` -- a DataFrame
        indexed by variable name for a coefficient block, a Series
        indexed by label for the header/diagnostics blocks -- so
        `summary.tables[1]["coef"]`, `summary.tables[1]["P>|z|"]`,
        `summary.tables[1].data.loc["x1"]`, or
        `summary.tables[1].data.to_csv(...)` all work directly, on top
        of `summary.tables[1]` still printing/rendering like before.

    Raises
    ------
    SummaryValidationError
        If the coefficient data is inconsistent (mismatched lengths,
        an unrecognized `dist`, or `dist="t"` without a usable `df`),
        or if `alpha` is outside (0, 1).
    """
    _validate_top_level(dist, alpha)

    left = left or []
    right = right or []
    diag_left = diag_left or []
    diag_right = diag_right or []
    extra_lines = extra_lines or []
    notes = list(notes or [])
    subtitle_lines = [subtitle] if isinstance(subtitle, str) else list(subtitle or [])
    # None disables stars entirely; {} enables them with the default
    # thresholds; a non-empty dict enables them with custom ones.
    stars_arg: dict[float, str] | None = (
        stars if isinstance(stars, dict) else ({} if stars else None)
    )

    block_list = _normalize_blocks(names, coefs, se, blocks, tables)
    for b in block_list:
        b.resolve(dist, df)

    table_blocks = _normalize_tables(tables)

    logger.debug(
        "Building summary %r with %d coefficient block(s), %d extra table(s), dist=%r",
        title, len(block_list), len(table_blocks), dist,
    )

    top_text, top_w = two_col_text(left, right)

    block_texts, block_widths, block_frames, block_stars = [], [], [], []
    any_stars_shown = False
    for b in block_list:
        frame = coef_block_frame(
            b, alpha, dist=b.effective_dist(dist), df=b.effective_df(df), stars=stars_arg
        )
        block_frames.append(frame)
        row_stars = frame["sig"].to_numpy() if "sig" in frame.columns else None
        block_stars.append(row_stars)
        any_stars_shown = any_stars_shown or row_stars is not None
        txt, w = block_table_text(
            b, alpha, dist=b.effective_dist(dist), df=b.effective_df(df), stars=row_stars
        )
        block_texts.append(txt)
        block_widths.append(w)

    table_texts, table_widths, table_frames = [], [], []
    for t in table_blocks:
        txt, w = table_block_text(t)
        table_texts.append(txt)
        table_widths.append(w)
        table_frames.append(table_block_frame(t))

    if any_stars_shown:
        notes.append(legend_for(stars_arg or None))

    has_diag = bool(diag_left) or bool(diag_right)
    diag_text, diag_w = two_col_text(diag_left, diag_right) if has_diag else ("", 0)

    width = max([top_w, diag_w, len(title) + 4, _MIN_WIDTH] + block_widths + table_widths)
    sep = "=" * width

    header_lines = [title.center(width)]
    if subtitle_lines:
        header_lines.append(sep)
        for line in subtitle_lines:
            header_lines.extend(textwrap.wrap(line, width) or [line])
    header_lines.append(sep)
    header_lines.append(top_text)
    header_lines.append(sep)
    header_text = "\n".join(header_lines)

    extra_txt: list[str] = []
    if extra_lines:
        extra_txt.extend(f"{label}: {val}" for label, val in extra_lines)
    if notes:
        extra_txt.append("Notes:")
        for i, n in enumerate(notes):
            extra_txt.extend(textwrap.wrap(f"[{i + 1}] {n}", width, subsequent_indent="    "))

    full_text = header_text + "\n" + "\n".join(block_texts)
    if table_texts:
        full_text += "\n" + "\n".join(table_texts)
    if has_diag:
        full_text += "\n" + sep + "\n" + diag_text + "\n" + sep
    if extra_txt:
        if not has_diag:
            full_text += "\n" + sep
        full_text += "\n" + "\n".join(extra_txt)

    header_html_parts = [CSS, f"<div class='ss-summary'><h3>{esc_html(title)}</h3>"]
    if subtitle_lines:
        joined = "<br>".join(esc_html(s) for s in subtitle_lines)
        header_html_parts.append(f"<p class='ss-formula'>{joined}</p>")
    header_html_parts.append(html_two_col_table(left, right))
    header_html = "".join(header_html_parts)

    block_htmls = []
    for b, row_stars in zip(block_list, block_stars):
        h4 = f"<h4>{esc_html(b.heading)}</h4>" if b.heading else ""
        block_htmls.append(
            h4
            + html_coef_table(
                b, alpha, dist=b.effective_dist(dist), df=b.effective_df(df), stars=row_stars
            )
        )

    diag_html = html_two_col_table(diag_left, diag_right) if has_diag else ""

    table_htmls = []
    for t in table_blocks:
        h4 = f"<h4>{esc_html(t.heading)}</h4>" if t.heading else ""
        table_htmls.append(h4 + html_table_block(t))

    footer_html = ""
    if extra_lines:
        footer_html += "".join(
            f"<div class='ss-extra'>{esc_html(label)}: {esc_html(val)}</div>"
            for label, val in extra_lines
        )
    if notes:
        items = "".join(f"<li>{esc_html(n)}</li>" for n in notes)
        footer_html += f"<div class='ss-notes'><b>Notes:</b><ol>{items}</ol></div>"

    full_html = (
        header_html
        + "".join(block_htmls)
        + "".join(table_htmls)
        + diag_html
        + footer_html
        + "</div>"
    )

    table_entries = [
        SummaryTable(
            header_text, header_html + "</div>", title="Model Info", data=info_series(left, right)
        )
    ]
    for b, txt, html, frame in zip(block_list, block_texts, block_htmls, block_frames):
        table_entries.append(
            SummaryTable(
                txt, CSS + f"<div class='ss-summary'>{html}</div>", title=b.heading, data=frame
            )
        )
    for t, txt, html, frame in zip(table_blocks, table_texts, table_htmls, table_frames):
        table_entries.append(
            SummaryTable(
                txt, CSS + f"<div class='ss-summary'>{html}</div>", title=t.heading, data=frame
            )
        )
    if has_diag:
        table_entries.append(
            SummaryTable(
                diag_text,
                CSS + f"<div class='ss-summary'>{diag_html}</div>",
                title="Diagnostics",
                data=info_series(diag_left, diag_right),
            )
        )

    return Summary(table_entries, extra_txt, full_text, full_html)
