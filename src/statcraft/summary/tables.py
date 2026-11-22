# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Output containers returned by :func:`statcraft.utils.summary.build_summary`."""

from __future__ import annotations

from pathlib import Path

__all__ = ["SummaryTable", "Summary"]


class SummaryTable:
    """One table within a Summary -- mirrors statsmodels'
    ``results.summary().tables[i]``: the header/info block is
    ``tables[0]``, each coefficient block is its own entry after that,
    so a single table can be pulled out, printed, or embedded on its
    own instead of the whole summary.

    Beyond printing, each table carries its data as a real pandas
    object in ``.data`` -- a DataFrame indexed by variable name for a
    coefficient block (columns "coef", "std err", "z"/"t", "P>|z|"/
    "P>|t|", and the two CI bounds -- or just "coef" when the block has
    no standard errors), or a Series indexed by label for the header/
    diagnostics blocks. Indexing the table itself forwards to that
    object, so ``table["coef"]`` or ``table["z"]`` returns the column
    as a pandas Series, and the full pandas API is available via
    ``.data`` directly: ``table.data.loc["x1"]``, ``table.data.round(3)``,
    ``table.data.to_csv(...)``, etc. ``.data`` is None for a table that
    was never given structured data.
    """

    def __init__(self, text: str, html: str, title: str | None = None, data=None):
        self._text = text
        self._html = html
        self.title = title
        self.data = data

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    def __eq__(self, other) -> bool:
        if not isinstance(other, SummaryTable):
            return NotImplemented
        return self._text == other._text and self._html == other._html and self.title == other.title

    def _repr_html_(self) -> str:
        return self._html

    def as_text(self) -> str:
        return self._text

    def as_html(self) -> str:
        return self._html

    def _require_data(self):
        if self.data is None:
            title = f" {self.title!r}" if self.title else ""
            raise TypeError(
                f"Table{title} has no structured data to index into -- "
                "use as_text()/as_html() to see its rendered form instead."
            )
        return self.data

    def __getitem__(self, key):
        """Column access for a coefficient block (``table["coef"]``,
        ``table["P>|z|"]``) or label access for a header/diagnostics
        block (``table["Log-Likelihood"]``). Returns whatever pandas
        returns for that key -- a Series for a DataFrame column, a
        scalar for a Series label.
        """
        return self._require_data()[key]

    def __len__(self) -> int:
        return len(self._require_data())

    def __iter__(self):
        return iter(self._require_data())

    def to_frame(self):
        """The underlying pandas object: a DataFrame for a coefficient
        block, a Series for the header/diagnostics blocks, or None if
        this table has no structured data.
        """
        return self.data


class Summary:
    """Statsmodels-style results summary. ``print(summary)``/``str(summary)``
    give the full plain-text report; in a notebook it renders as styled
    HTML. ``summary.tables`` holds each individual table -- ``tables[0]``
    is the header/info block, ``tables[1:]`` are the coefficient blocks
    -- so a piece can be extracted and reused on its own, e.g.
    ``summary.tables[1]`` for just the main coefficient table or
    ``summary.tables[1].as_html()`` to embed it elsewhere.
    ``summary.extra_txt`` holds the footer notes (convergence warnings,
    "no standard errors computed", etc.), also as a list so individual
    lines can be pulled out. Beyond the report itself, ``save_latex()``
    and ``save_excel()`` export every table's structured data (see
    ``SummaryTable.data``) to a ``.tex`` fragment/file or a multi-sheet
    ``.xlsx`` workbook, for pasting into a paper or handing off to
    someone who wants the numbers in a spreadsheet.
    """

    def __init__(self, tables: list[SummaryTable], extra_txt: list[str], text: str, html: str):
        self.tables = tables
        self.extra_txt = extra_txt
        self._text = text
        self._html = html

    def __repr__(self) -> str:
        return self._text

    def __str__(self) -> str:
        return self._text

    def _repr_html_(self) -> str:
        return self._html

    def as_text(self) -> str:
        return self._text

    def as_html(self) -> str:
        return self._html

    def save_text(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Write the plain-text report to ``path``."""
        Path(path).write_text(self._text, encoding=encoding)

    def save_html(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Write a standalone HTML file with the styled report to ``path``."""
        doc = f"<!DOCTYPE html><html><head><meta charset='{encoding}'></head><body>{self._html}</body></html>"
        Path(path).write_text(doc, encoding=encoding)

    def _tables_as_frames(self):
        """Yields (table, dataframe) for every table with structured
        data, promoting a header/diagnostics Series to a one-column
        "value" DataFrame so both kinds export the same way.
        """
        for table in self.tables:
            if table.data is None:
                continue
            data = table.data
            yield table, (data if hasattr(data, "columns") else data.to_frame(name="value"))

    def as_latex(self) -> str:
        """LaTeX fragment with one ``tabular`` per table that has
        structured data (see ``SummaryTable.data``) -- meant to be
        pasted into an existing document (a paper, a report). Use
        :meth:`save_latex` to write a standalone, compilable ``.tex``
        file instead. Numbers are rounded to 4 decimals for display,
        same as the printed text/HTML tables; use
        ``table.data.to_latex(...)`` directly on one table for full
        control over formatting.
        """
        parts = []
        for table, frame in self._tables_as_frames():
            parts.append(frame.to_latex(float_format="%.4f", caption=table.title or None))
        if self.extra_txt:
            parts.append("\n".join(f"% {line}" for line in self.extra_txt))
        return "\n".join(parts)

    def save_latex(
        self, path: str | Path, standalone: bool = False, encoding: str = "utf-8"
    ) -> None:
        """Write :meth:`as_latex`'s output to ``path``. With
        ``standalone=True``, wraps it in a minimal
        ``\\documentclass{article}`` preamble (with ``booktabs``, which
        pandas' table rules need) so the file compiles into a PDF on
        its own; otherwise it's a fragment meant to be pasted into an
        existing document.
        """
        body = self.as_latex()
        if standalone:
            body = (
                "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n"
                + body
                + "\n\\end{document}\n"
            )
        Path(path).write_text(body, encoding=encoding)

    @staticmethod
    def _sheet_name(title: str | None, index: int, used: set) -> str:
        # Excel sheet names: 31 chars max, and reject [ ] : * ? / \ .
        base = (title or f"Table {index}").strip() or f"Table {index}"
        for ch in "[]:*?/\\":
            base = base.replace(ch, "-")
        base = base[:31] or f"Table {index}"
        name, n = base, 1
        while name in used:
            suffix = f" ({n})"
            name = base[: 31 - len(suffix)] + suffix
            n += 1
        used.add(name)
        return name

    def save_excel(self, path: str | Path) -> None:
        """Write every table's structured data to its own sheet of an
        ``.xlsx`` workbook at ``path`` -- the header/diagnostics blocks
        as a one-column "value" sheet, each coefficient block as its
        own sheet of full-precision numbers (not the display-rounded
        text). Needs the optional ``openpyxl`` dependency
        (``pip install openpyxl`` or ``pip install statcraft[excel]``).
        """
        try:
            import openpyxl  # noqa: F401  -- imported only to fail here with a clear message
        except ImportError as exc:
            raise ImportError(
                "save_excel() needs openpyxl. Install it with `pip install openpyxl` "
                "or `pip install statcraft[excel]`."
            ) from exc
        import pandas as pd

        used_names: set = set()
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for i, (table, frame) in enumerate(self._tables_as_frames()):
                sheet = self._sheet_name(table.title, i, used_names)
                frame.to_excel(writer, sheet_name=sheet)
