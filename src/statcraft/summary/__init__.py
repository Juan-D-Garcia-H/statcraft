# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
statcraft.utils.summary -- general-purpose, statsmodels-style results
summary builder, used by ``GLM.summary()`` and ``NLS.summary()``.

Not tied to GLM or to NLS specifically: :func:`build_summary` takes
plain Python/NumPy data (a title, a two-column block of key/value
info, one or more coefficient blocks) and returns a :class:`Summary`
object with a plain-text report, a styled HTML report, and the
individual tables broken out separately -- the same shape as
statsmodels' ``results.summary()`` (``summary.tables[i]``,
``summary.extra_txt``).

This subpackage has no dependency on GLM/NLS internals, and only a
soft dependency on scipy (used opportunistically for the normal/t CDF
and PPF behind the stat/CI columns, with a pure-NumPy fallback when
scipy isn't installed -- see :mod:`statcraft.utils.summary._compat`).
scipy is already a hard dependency of ``statcraft`` itself (see
``pyproject.toml``), so in practice the exact z/t values are always
used here; the fallback exists so this subpackage stays usable on its
own if it's ever lifted out.

Module layout
-------------
- ``_compat.py``      -- z/t CDF/PPF, with and without scipy
- ``blocks.py``        -- ``CoefBlock``: validated dataclass for one coefficient block
- ``builder.py``        -- ``build_summary``: validation + render + assembly
- ``exceptions.py``      -- ``SummaryError``, ``SummaryValidationError``
- ``html_render.py``      -- HTML tables + embedded CSS
- ``layout.py``            -- column-width calculation
- ``tables.py``             -- ``SummaryTable``, ``Summary`` (with ``save_text``/``save_html``)
- ``text_render.py``         -- plain-text, statsmodels-style tables

Quick start
-----------
    from statcraft.utils.summary import build_summary

    summary = build_summary(
        title="My Model Results",
        subtitle="Formula: y ~ x1 + x2",
        left=[("Dep. Variable:", "y"), ("No. Observations:", "200")],
        right=[("Log-Likelihood:", "-120.5"), ("AIC:", "247.0")],
        names=["Intercept", "x1", "x2"], coefs=beta, se=se,
    )
    print(summary)                 # plain text
    summary                        # in a notebook: styled HTML
    summary.tables[1]               # just the coefficient table
    summary.tables[1]["coef"]        # that column, as a pandas Series indexed by name
    summary.tables[1].data.loc["x1"]  # that row, as a pandas Series
    summary.tables[1].data.to_csv("coefs.csv")
    summary.save_html("model.html")

Every table's underlying data is a real pandas object (see
``frame.py``): a DataFrame indexed by variable name for a coefficient
block, a Series indexed by label for the header/diagnostics blocks.
That means the full pandas API -- filtering, sorting, ``.round()``,
``.to_csv()``/``.to_dict()``, merging several models' tables, ... --
is available on any table via ``.data``, on top of it still printing
and rendering exactly as before. ``summary.save_latex(...)`` and
``summary.save_excel(...)`` export that same structured data to a
``.tex`` fragment or a multi-sheet ``.xlsx`` workbook. Passing
``stars=True`` adds significance markers ("*", "**", "***") next to
each coefficient, with an automatic legend footnote -- see
:mod:`.significance`.

See ``GLM.summary()`` / ``NLS.summary()`` for how this is actually
called within statcraft -- both pass ``blocks`` for multi-block models
(zero-inflated, generalized gamma, ordinal, multinomial, ...) and the
simpler ``names``/``coefs``/``se`` path otherwise.
"""

from .blocks import CoefBlock, TableBlock
from .builder import build_summary
from .exceptions import SummaryError, SummaryValidationError
from .significance import DEFAULT_STAR_THRESHOLDS
from .tables import Summary, SummaryTable

__all__ = [
    "build_summary",
    "Summary",
    "SummaryTable",
    "CoefBlock",
    "TableBlock",
    "SummaryError",
    "SummaryValidationError",
    "DEFAULT_STAR_THRESHOLDS",
]
