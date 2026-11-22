# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Significance stars (``"*"``, ``"**"``, ``"***"``), the standard shorthand for
p-value thresholds in econometric/statistical reporting (as in R's
``stargazer`` or Stata's estimation tables). Opt-in via
``build_summary(..., stars=True)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_STAR_THRESHOLDS", "stars_for", "legend_for"]

#: Significance-star thresholds, ascending in strictness (``p <= 0.01`` is the
#: strictest band and gets the most stars). Order matters for the
#: first-match-wins scan in ``stars_for``.
DEFAULT_STAR_THRESHOLDS: dict[float, str] = {0.01: "***", 0.05: "**", 0.1: "*"}


def stars_for(pvalue: np.ndarray, thresholds: dict[float, str] | None = None) -> np.ndarray:
    """Marker per p-value: the marker for the strictest (smallest)
    threshold `pvalue` satisfies, or "" if it clears none of them (or
    is NaN -- e.g. a coefficient with no usable standard error).
    """
    thresholds = dict(sorted((thresholds or DEFAULT_STAR_THRESHOLDS).items()))
    pvalue = np.asarray(pvalue, dtype=float)
    out = np.full(pvalue.shape, "", dtype=object)
    assigned = np.zeros(pvalue.shape, dtype=bool)
    for cutoff, marker in thresholds.items():
        hit = ~assigned & np.isfinite(pvalue) & (pvalue <= cutoff)
        out[hit] = marker
        assigned |= hit
    return out


def legend_for(thresholds: dict[float, str] | None = None) -> str:
    """Footnote spelling out what the stars mean, e.g.
    ``"Significance: * p<0.1, ** p<0.05, *** p<0.01"`` -- sorted so the
    marker with the fewest stars (loosest threshold) comes first,
    matching how it reads left-to-right.
    """
    thresholds = thresholds or DEFAULT_STAR_THRESHOLDS
    parts = [
        f"{marker} p<{cutoff:g}" for cutoff, marker in sorted(thresholds.items(), reverse=True)
    ]
    return "Significance: " + ", ".join(parts)
