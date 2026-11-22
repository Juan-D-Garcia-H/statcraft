# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Exceptions raised by statcraft.utils.summary.

Kept in their own module (rather than raising bare ``ValueError``
everywhere) so callers can catch package-specific problems precisely:

    try:
        build_summary(...)
    except SummaryValidationError as exc:
        ...
"""

from __future__ import annotations

__all__ = [
    "SummaryError",
    "SummaryValidationError",
]


class SummaryError(Exception):
    """Base class for all errors raised by this package."""


class SummaryValidationError(SummaryError):
    """Raised when the data passed to :func:`build_summary` (or a
    :class:`~statcraft.utils.summary.blocks.CoefBlock`) is inconsistent -- e.g.
    mismatched lengths, an unknown distribution name, or a missing
    ``df`` for ``dist="t"``.
    """
