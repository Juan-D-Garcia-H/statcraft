# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Typed, validated representations of the data that goes into a summary.

``build_summary`` used to accept a coefficient block as a bare dict
with string keys ("heading", "names", "coefs", "se"). That still works
(see :meth:`CoefBlock.from_dict`), but internally everything is
normalized into a :class:`CoefBlock` so the rest of the package can
rely on consistent, validated data instead of re-checking dicts
everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .exceptions import SummaryValidationError

logger = logging.getLogger(__name__)

__all__ = ["CoefBlock", "TableBlock"]

_VALID_DISTS = ("normal", "t")
_VALID_ALIGN = ("left", "right", "center")


@dataclass
class CoefBlock:
    """One coefficient table: a heading plus parallel arrays of
    variable names, point estimates, and (optionally) standard errors.

    Parameters
    ----------
    heading : str or None
        Sub-heading shown above this block (e.g. "Count Model (beta)").
        None for an unlabeled single-block summary.
    names : sequence of str
        Coefficient/variable names. Defines the length every other
        array in the block must match.
    coefs : sequence of float
        Point estimates, same length as ``names``.
    se : sequence of float or None
        Standard errors, same length as ``names``, or None to render a
        coef-only column with no stat/P-value/CI columns (e.g. when
        standard errors weren't computed).
    dist : {"normal", "t"} or None
        Overrides the summary-wide distribution for just this block.
        None means "use the summary-wide default".
    df : int, float, or None
        Residual degrees of freedom, required when this block's
        effective ``dist`` is "t". Overrides the summary-wide `df`
        when given.
    """

    names: Sequence[str]
    coefs: Sequence[float]
    se: Sequence[float] | None = None
    heading: str | None = None
    dist: str | None = None
    df: float | None = None

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0:
            raise SummaryValidationError("A coefficient block needs at least one name/coefficient.")
        if len(self.coefs) != n:
            raise SummaryValidationError(
                f"names and coefs must be the same length (got {n} names, {len(self.coefs)} coefs)"
                + (f" in block {self.heading!r}" if self.heading else "")
            )
        if self.se is not None and len(self.se) != n:
            raise SummaryValidationError(
                f"names and se must be the same length (got {n} names, {len(self.se)} se)"
                + (f" in block {self.heading!r}" if self.heading else "")
            )
        if self.dist is not None and self.dist not in _VALID_DISTS:
            raise SummaryValidationError(f"dist must be one of {_VALID_DISTS!r}, got {self.dist!r}")

        context = f" in block {self.heading!r}" if self.heading else ""

        try:
            np.asarray(self.coefs, dtype=float)
        except (TypeError, ValueError) as exc:
            raise SummaryValidationError(f"coefs must be numeric{context}: {exc}") from exc

        if self.se is not None:
            try:
                se_arr = np.asarray(self.se, dtype=float)
            except (TypeError, ValueError) as exc:
                raise SummaryValidationError(f"se must be numeric{context}: {exc}") from exc
            # NaN is allowed (e.g. a dropped/rank-deficient column with no
            # usable SE); a *finite negative* SE is never legitimate --
            # it means something upstream (the caller's own SE computation)
            # is broken, so this fails loudly here rather than quietly
            # producing a nonsensical stat/CI downstream.
            bad = np.isfinite(se_arr) & (se_arr < 0)
            if np.any(bad):
                bad_names = [str(self.names[int(i)]) for i in np.flatnonzero(bad)]
                raise SummaryValidationError(
                    f"se must be >= 0 (a standard error can't be negative){context}: "
                    f"negative for {bad_names}"
                )

        names_list = [str(x) for x in self.names]
        if len(set(names_list)) != len(names_list):
            dupes = sorted({x for x in names_list if names_list.count(x) > 1})
            logger.warning(
                "Duplicate coefficient name(s)%s: %s -- indexing summary.tables[i].data "
                "by name will return more than one row for these.",
                context,
                dupes,
            )

    def effective_dist(self, default_dist: str) -> str:
        return self.dist if self.dist is not None else default_dist

    def effective_df(self, default_df: float | None) -> float | None:
        return self.df if self.df is not None else default_df

    def resolve(self, default_dist: str, default_df: float | None) -> None:
        """Validate that a "t" distribution (from this block or the
        summary-wide default) has a usable, positive df. Called once
        the summary-wide defaults are known.
        """
        dist = self.effective_dist(default_dist)
        df = self.effective_df(default_df)
        if self.se is not None and dist == "t":
            context = f" in block {self.heading!r}" if self.heading else ""
            if df is None:
                raise SummaryValidationError(
                    f"dist='t' requires df (residual degrees of freedom){context}"
                )
            if not (df > 0):
                raise SummaryValidationError(f"df must be > 0 for dist='t', got {df!r}{context}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CoefBlock":
        """Build a CoefBlock from the legacy dict shape
        ``{"heading": ..., "names": ..., "coefs": ..., "se": ...}``,
        with "dist"/"df" as optional extra keys.
        """
        unknown = set(data) - {"heading", "names", "coefs", "se", "dist", "df"}
        if unknown:
            logger.debug("Ignoring unknown keys in coefficient block dict: %s", sorted(unknown))
        return cls(
            heading=data.get("heading"),
            names=data["names"],
            coefs=data["coefs"],
            se=data.get("se"),
            dist=data.get("dist"),
            df=data.get("df"),
        )

    def as_arrays(self) -> tuple[list[str], np.ndarray, np.ndarray | None]:
        """Names as a list of str, coefs and se (if any) as float arrays."""
        names = [str(n) for n in self.names]
        coefs = np.asarray(self.coefs, dtype=float)
        se = None if self.se is None else np.asarray(self.se, dtype=float)
        return names, coefs, se


@dataclass
class TableBlock:
    """A free-form table with arbitrary columns.

    Unlike :class:`CoefBlock` (which always renders the coefficient/SE/
    statistic/CI layout), a ``TableBlock`` is meant for pieces such as the
    variance-components table of a mixed model (Variance / Std.Dev. / Corr)
    or a one-row summary of standardized residuals.

    Parameters
    ----------
    columns : sequence of str
        Column headers, excluding the optional row-label column.
    rows : sequence of sequences
        One entry per row; each row must have ``len(columns)`` cells.  Cells
        may be numbers (formatted with the matching entry of ``formats``),
        strings (used verbatim) or ``None`` (rendered blank).
    heading : str or None
        Sub-heading shown above the table.
    index : sequence or None
        Optional row labels.  When omitted the table has no label column.
    index_name : str
        Header of the row-label column (only used when ``index`` is given).
    align : sequence of {"left", "right", "center"}, optional
        Alignment of each data column; defaults to right alignment.
    formats : sequence of str, optional
        Format specification per column for numeric cells (default
        ``".4f"``).
    """

    columns: Sequence[str]
    rows: Sequence[Sequence[Any]]
    heading: str | None = None
    index: Sequence[Any] | None = None
    index_name: str = ""
    align: Sequence[str] | None = None
    formats: Sequence[str] | None = None

    def __post_init__(self) -> None:
        n = len(self.columns)
        if n == 0:
            raise SummaryValidationError("A table block needs at least one column.")
        for r_i, row in enumerate(self.rows):
            if len(row) != n:
                raise SummaryValidationError(
                    f"Row {r_i} has {len(row)} cells but the table has {n} columns"
                    + (f" in block {self.heading!r}" if self.heading else "")
                )
        if self.index is not None and len(self.index) != len(self.rows):
            raise SummaryValidationError(
                f"index and rows must be the same length (got {len(self.index)} labels, "
                f"{len(self.rows)} rows)" + (f" in block {self.heading!r}" if self.heading else "")
            )
        if self.align is not None:
            if len(self.align) != n or any(a not in _VALID_ALIGN for a in self.align):
                raise SummaryValidationError(
                    f"align must have one of {_VALID_ALIGN!r} per column"
                    + (f" in block {self.heading!r}" if self.heading else "")
                )
        if self.formats is not None and len(self.formats) != n:
            raise SummaryValidationError(
                f"formats must have one entry per column"
                + (f" in block {self.heading!r}" if self.heading else "")
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableBlock":
        """Build a TableBlock from ``{columns, rows, heading, ...}``."""
        unknown = set(data) - {
            "heading", "columns", "rows", "index", "index_name", "align", "formats"
        }
        if unknown:
            logger.debug("Ignoring unknown keys in table block dict: %s", sorted(unknown))
        return cls(
            columns=data["columns"],
            rows=data["rows"],
            heading=data.get("heading"),
            index=data.get("index"),
            index_name=data.get("index_name", ""),
            align=data.get("align"),
            formats=data.get("formats"),
        )

    def cell_text(self, i: int, j: int) -> str:
        """Rendered text of cell ``(i, j)``."""
        value = self.rows[i][j]
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(v):
            return "NA"
        fmt = self.formats[j] if self.formats is not None else ".4f"
        return format(v, fmt)

    def header_text(self, j: int) -> str:
        """Header text of data column ``j``."""
        return str(self.columns[j])

    def label_text(self, i: int) -> str:
        """Text of row label ``i`` (empty when the block has no index)."""
        return "" if self.index is None else str(self.index[i])

    def column_align(self, j: int) -> str:
        """Alignment of data column ``j``."""
        return self.align[j] if self.align is not None else "right"
