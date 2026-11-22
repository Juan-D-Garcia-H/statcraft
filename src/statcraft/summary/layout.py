# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Small layout helpers shared by the text and HTML renderers."""

from __future__ import annotations

from typing import Iterable

__all__ = ["col_width"]


def col_width(items: Iterable, min_w: int = 8, pad: int = 1) -> int:
    """Column width that fits every item in ``items`` without truncating,
    at least ``min_w``. Used instead of a fixed width so long model
    names, link/method descriptions, or variable/category names widen
    the table rather than getting cut off mid-word.
    """
    lengths = [len(str(x)) for x in items]
    return max(lengths + [min_w]) + pad
