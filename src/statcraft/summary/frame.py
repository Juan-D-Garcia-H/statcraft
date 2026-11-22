# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Structured (pandas) views of the tables, built alongside the text/HTML
ones so a table can be indexed and manipulated programmatically --
``summary.tables[i]["coef"]``, ``.loc["x1"]``, ``.to_csv(...)``, and the
rest of the pandas API -- instead of only being printable/renderable.
"""

from __future__ import annotations

import numpy as np

from ._compat import DIST_FUNCS
from .blocks import CoefBlock, TableBlock
from .significance import stars_for

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard dependency of statcraft itself
    pd = None

__all__ = ["coef_block_frame", "table_block_frame", "info_series", "HAS_PANDAS"]

HAS_PANDAS = pd is not None


def _require_pandas() -> None:
    if pd is None:
        raise ImportError(
            "Structured table access (summary.tables[i]['coef'], .data, .loc[...], "
            "to_csv(), ...) needs pandas. Install it with `pip install pandas`."
        )


def coef_block_frame(
    block: CoefBlock,
    alpha: float,
    dist: str,
    df: float | None,
    stars: dict[float, str] | None = None,
) -> "pd.DataFrame":
    """A pandas DataFrame for one coefficient block, indexed by variable
    name. Columns match the printed table exactly: just "coef" when the
    block has no standard errors, otherwise "coef", "std err", the
    stat column ("z" or "t"), its P-value column, and the two CI bound
    columns -- all as full-precision floats (the printed tables round
    to 3-4 decimals; this doesn't). When `stars` is given (even as an
    empty dict, to request the default thresholds), an extra "sig"
    column of ``""``, ``"*"``, ``"**"``, ``"***"`` markers is appended --
    see :mod:`.significance`.
    """
    _require_pandas()
    names, coefs, se = block.as_arrays()
    index = pd.Index(names, name="name")

    if se is None:
        return pd.DataFrame({"coef": coefs}, index=index)

    df = float(df) if df is not None else float("nan")  # normal family ignores df
    stat_label, ppf_fn, sf_fn = DIST_FUNCS[dist]
    crit = ppf_fn(1.0 - alpha / 2.0, df)
    lo_label = f"[{alpha / 2 * 100:.1f}%"
    hi_label = f"{(1 - alpha / 2) * 100:.1f}%]"

    valid = np.isfinite(se) & (se > 0)
    stat = np.where(valid, coefs / np.where(valid, se, 1.0), np.nan)
    # Vectorized over the whole column in one call (see _compat.t_sf) --
    # matters once a model has dozens/hundreds of coefficients (e.g.
    # many categorical dummies), where one scipy call per row used to
    # dominate the cost of building this table.
    pvalue = np.where(np.isfinite(stat), 2.0 * sf_fn(np.abs(stat), df), np.nan)
    lo = coefs - crit * se
    hi = coefs + crit * se

    data = {
        "coef": coefs,
        "std err": se,
        stat_label: stat,
        f"P>|{stat_label}|": pvalue,
        lo_label: lo,
        hi_label: hi,
    }
    if stars is not None:
        data["sig"] = stars_for(pvalue, stars or None)
    return pd.DataFrame(data, index=index)


def table_block_frame(block: TableBlock) -> "pd.DataFrame":
    """A pandas DataFrame for one :class:`TableBlock`.

    Columns mirror the printed table; cells are kept as the raw values given
    by the caller (numbers stay numeric).  When the block has row labels they
    become the DataFrame index.
    """
    _require_pandas()
    data = {str(block.columns[j]): [block.rows[i][j] for i in range(len(block.rows))] for j in range(len(block.columns))}
    index = (
        pd.Index([block.label_text(i) for i in range(len(block.rows))], name=block.index_name or None)
        if block.index is not None
        else None
    )
    return pd.DataFrame(data, index=index)


def info_series(left: list[tuple[str, str]], right: list[tuple[str, str]]) -> "pd.Series":
    """A pandas Series for a two-column key/value block (the header info
    block, or the diagnostics block), indexed by label with the trailing
    ":" stripped -- e.g. ``series["Log-Likelihood"]``. Values are kept
    exactly as given (already-formatted strings), since the caller
    controls their formatting and "n/a"-style placeholders are common.
    """
    _require_pandas()
    pairs = list(left) + list(right)
    labels = [str(k).rstrip(":").strip() for k, _ in pairs]
    values = [v for _, v in pairs]
    return pd.Series(values, index=pd.Index(labels, name="label"), name="value")
