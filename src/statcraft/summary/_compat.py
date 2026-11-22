# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
_compat.py -- statistical primitives with an optional scipy dependency.

Every function here tries scipy first (exact, fast, well-tested) and
falls back to a pure-NumPy approximation when scipy isn't installed.
This keeps the whole package usable with nothing but NumPy, while
getting exact results automatically whenever scipy happens to be
available.

Nothing in this module depends on the rest of the package -- it can be
copied out and reused on its own.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "erf",
    "std_normal_cdf",
    "std_normal_sf",
    "std_normal_ppf",
    "t_cdf",
    "t_ppf",
    "t_sf",
    "DIST_FUNCS",
]


def erf(x: np.ndarray) -> np.ndarray:
    """Error function. Uses ``scipy.special.erf`` when available;
    otherwise the Abramowitz & Stegun 7.1.26 approximation
    (max absolute error ~1.5e-7, plenty for display-precision output).
    """
    try:
        from scipy.special import erf as _scipy_erf

        return _scipy_erf(x)
    except ImportError:
        logger.debug("scipy not available, using erf() polynomial approximation")

    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def std_normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, built on :func:`erf`."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def std_normal_sf(x: np.ndarray) -> np.ndarray:
    """Standard normal survival function (upper tail, ``1 - CDF``).

    Kept separate from ``1.0 - std_normal_cdf(x)`` (mathematically the
    same thing) because that subtraction loses precision for the small
    tail probabilities p-values usually are: ``1 - (something very
    close to 1)`` cancels significant digits. ``scipy.stats.norm.sf``
    is computed directly (via ``erfc``, without that subtraction), so
    it's used here whenever scipy is available; only the no-scipy
    fallback still goes through :func:`erf` and pays the (much smaller,
    display-precision-irrelevant) cancellation cost.
    """
    try:
        from scipy.stats import norm as _scipy_norm

        return _scipy_norm.sf(x)
    except ImportError:
        return 1.0 - std_normal_cdf(x)


def std_normal_ppf(p: float) -> float:
    """Standard normal quantile function (inverse CDF).

    Uses ``scipy.stats.norm.ppf`` when available; otherwise Newton's
    method on :func:`std_normal_cdf`, which converges in well under
    60 iterations for any p in (0, 1).
    """
    try:
        from scipy.stats import norm as _scipy_norm

        return float(_scipy_norm.ppf(p))
    except ImportError:
        logger.debug("scipy not available, using Newton's method for std_normal_ppf()")

    if not (0.0 < p < 1.0):
        return float("nan")

    x = 0.0
    for _ in range(60):
        cdf = std_normal_cdf(x)
        pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        if pdf < 1e-300:
            break
        step = (cdf - p) / pdf
        x -= step
        if abs(step) < 1e-14:
            break
    return float(x)


def t_cdf(x: np.ndarray, df: float) -> float:
    """Student's t CDF, for a single scalar `x` -- see :func:`t_sf` for
    the array-vectorized tail probability used when building a table.

    Needs scipy for an exact result (the incomplete beta function
    behind it isn't worth reimplementing here). Without scipy this
    falls back to the normal CDF -- exact only as ``df -> inf``, and
    noticeably off for small df, where the t distribution is normally
    used precisely because it is heavier-tailed than the normal. A
    warning is logged (once per call) so silent inaccuracy doesn't go
    unnoticed.
    """
    try:
        from scipy.stats import t as _scipy_t

        return float(_scipy_t.cdf(x, df))
    except ImportError:
        logger.warning(
            "scipy not installed: falling back to the normal distribution for "
            "t_cdf(), which is only exact as df -> inf. Install scipy for exact "
            "t-distribution p-values."
        )
        return std_normal_cdf(x)


def t_ppf(p: float, df: float) -> float:
    """Student's t quantile function -- see :func:`t_cdf` for the
    scipy-missing fallback caveat."""
    try:
        from scipy.stats import t as _scipy_t

        return float(_scipy_t.ppf(p, df))
    except ImportError:
        logger.debug("scipy not available, falling back to normal PPF for t_ppf()")
        return std_normal_ppf(p)


def t_sf(x: np.ndarray, df: float) -> np.ndarray:
    """Student's t survival function (the upper tail, ``1 - CDF``),
    vectorized over `x`.

    Kept separate from :func:`t_cdf` because p-values are needed for
    every coefficient in a table at once: computing them with one
    array call here (scipy vectorizes internally) instead of one
    Python-level scipy call per coefficient is the difference between
    O(1) and O(n) call overhead once a model has dozens or hundreds of
    parameters (e.g. many categorical dummies) -- the array form is a
    straight win, never slower, so there's no scalar-only fast path to
    preserve here the way there is for the once-per-table critical
    value in :func:`t_ppf`.
    """
    try:
        from scipy.stats import t as _scipy_t

        return _scipy_t.sf(x, df)
    except ImportError:
        logger.warning(
            "scipy not installed: falling back to the normal distribution for "
            "t_sf(), which is only exact as df -> inf. Install scipy for exact "
            "t-distribution p-values."
        )
        return 1.0 - std_normal_cdf(np.asarray(x, dtype=float))


# Maps a distribution name to (column label, ppf(p, df) -> scalar
# critical value, sf(x, df) -> array of one-sided tail probabilities,
# vectorized over x). The normal-family functions ignore `df` so both
# distributions share one call signature wherever a coefficient table
# is built.
DIST_FUNCS: dict[
    str, tuple[str, Callable[[float, float], float], Callable[[np.ndarray, float], np.ndarray]]
] = {
    "normal": ("z", lambda p, df: std_normal_ppf(p), lambda x, df: std_normal_sf(x)),
    "t": ("t", lambda p, df: t_ppf(p, df), lambda x, df: t_sf(x, df)),
}
