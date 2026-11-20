# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Randomized quantile residuals for the standard GLM families.

For a correctly specified model the randomized quantile residuals of Dunn &
Smyth (1996),

    u_i ~ U(F(y_i^-), F(y_i)),        r_i = Phi^-1(u_i),

are exactly standard normal for *any* response distribution. For continuous
families ``F(y^-) = F(y)`` and the residual is deterministic; for counts
(Poisson, negative binomial) and grouped binomial data the gap ``F(y) - F(y^-)``
is the point mass and ``rng`` breaks the tie.

This is what makes the normal Q-Q plot, P-P, half-normal and worm plots valid
for a discrete GLM, where the (asymptotically normal) deviance residuals leave a
staircase that the N(0, 1) envelope cannot describe.  Tweedie is supported too:
for ``1 < p < 2`` its CDF is the compound-Poisson/gamma series (including the
point mass at zero, which is randomised), ``p = 2`` is Gamma and ``p = 1`` is
Poisson.  For ``p > 2`` the positive-stable CDF is approximated by the
Lugannani-Rice saddlepoint formula.  Pareto, InverseGamma and the quasi
families return ``None`` so callers fall back to standardized deviance
residuals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

#: Families whose conditional law is available through scipy.
_FAMILIES = frozenset(
    {
        "Gaussian",
        "Binomial",
        "Poisson",
        "Exponential",
        "Gamma",
        "InverseGaussian",
        "NegativeBinomial",
        "Tweedie",
    }
)


def supports(family) -> bool:
    """Whether randomized quantile residuals are defined for *family*."""
    if type(family).__name__ == "Tweedie":
        p = getattr(getattr(family, "params", None), "var_power", np.nan)
        return bool(np.isfinite(p) and p >= 1.0)
    return type(family).__name__ in _FAMILIES


def _is_available(family, weights) -> bool:
    """``supports`` plus the data-dependent caveats (integer binomial trials)."""
    if not supports(family):
        return False
    if type(family).__name__ == "Binomial":
        w = np.asarray(weights, dtype=float)
        return bool(np.all(w == np.rint(w)) and np.all(w >= 1.0))
    return True


def _tweedie_lr_cdf(y, mu, phi, p):
    """Lugannani-Rice CDF for the positive-stable Tweedie law (``p > 2``).

    SciPy has no ``tweedie`` distribution, and exact inversion for every fitted
    row would be prohibitively slow in a diagnostic panel.  The saddlepoint
    approximation is accurate across the bulk of the distribution; the exact
    values at ``p = 1``, ``p = 2`` and ``1 < p < 2`` are handled separately.
    """
    y = np.maximum(np.asarray(y, dtype=float), 1e-300)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    phi = np.maximum(np.asarray(phi, dtype=float), 1e-300)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s = (y ** (1.0 - p) - mu ** (1.0 - p)) / ((1.0 - p) * phi)
        K = (y ** (2.0 - p) - mu ** (2.0 - p)) / ((2.0 - p) * phi)
        K2 = phi * y**p
        w = np.sign(s) * np.sqrt(np.maximum(2.0 * (s * y - K), 0.0))
        u = s * np.sqrt(K2)
        # Limit of the Lugannani-Rice correction as s -> 0. K'''(0) for a
        # Tweedie EDM is p * phi^2 * mu^(2p-1), giving this skewness term.
        skew = p * np.sqrt(phi) * mu ** (0.5 * p - 1.0)
        limit = 0.5 + stats.norm.pdf(0.0) * skew / 6.0
        safe_w = np.where(np.abs(w) < 1e-12, 1.0, w)
        safe_u = np.where(np.abs(u) < 1e-12, 1.0, u)
        F = stats.norm.cdf(w) + stats.norm.pdf(w) * (1.0 / safe_w - 1.0 / safe_u)
    F = np.where(np.abs(s) < 1e-8, limit, F)
    F = np.where(y <= 0.0, 0.0, F)
    return np.clip(np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _tweedie_cdf_bounds(family, y, mu, dispersion, weights):
    """``(F(y^-), F(y))`` for a Tweedie GLM with ``Var(Y) = phi * mu^p / w``."""
    p = float(getattr(getattr(family, "params", None), "var_power", np.nan))
    if not np.isfinite(p) or p < 1.0:
        return None, None

    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    w = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    phi = np.maximum(dispersion, 1e-300) / w

    if abs(p - 1.0) < 1e-10:
        # y/phi ~ Poisson(mu/phi) (only meaningful on multiples of phi).
        rate = mu / phi
        k = np.floor(np.maximum(y, 0.0) / phi)
        lo = stats.poisson.cdf(k - 1.0, rate)
        hi = stats.poisson.cdf(k, rate)
        return lo, hi

    if abs(p - 2.0) < 1e-10:
        F = stats.gamma.cdf(y, a=1.0 / phi, scale=mu * phi)
        return F, F

    if p < 2.0:
        # Compound Poisson-gamma: N ~ Poisson(lambda), each jump ~ Gamma(a,
        # theta), so the continuous part is a gamma mixture over N.  The mass
        # at zero is randomised by quantile_residuals().
        lam = mu ** (2.0 - p) / (phi * (2.0 - p))
        mass0 = np.exp(-lam)
        a = (2.0 - p) / (p - 1.0)
        theta = phi * mu ** (p - 1.0) * (p - 1.0)
        hi = np.array(mass0, dtype=float, copy=True)
        pos = y > 0.0
        if pos.any():
            lam_pos, y_pos, theta_pos = lam[pos], y[pos], theta[pos]
            # Truncate the mixture where the remaining Poisson mass is < 1e-12.
            nmax = np.minimum(
                np.maximum(stats.poisson.ppf(1.0 - 1e-12, lam_pos).astype(int) + 10, 10),
                10000,
            )
            n = np.arange(1, int(nmax.max()) + 1)[None, :]
            keep = n <= nmax[:, None]
            terms = stats.poisson.pmf(n, lam_pos[:, None]) * stats.gamma.cdf(
                y_pos[:, None], a * n, scale=theta_pos[:, None]
            )
            hi[pos] = mass0[pos] + np.sum(np.where(keep, terms, 0.0), axis=1)
        hi = np.clip(hi, 0.0, 1.0)
        lo = np.where(pos, hi, 0.0)
        return lo, hi

    hi = _tweedie_lr_cdf(y, mu, phi, p)
    lo = np.where(y > 0.0, hi, 0.0)
    return lo, hi


def cdf_bounds(family, y, mu, dispersion=1.0, weights=None):
    """Return ``(F(y^-), F(y))`` for *family*, or ``(None, None)`` if unavailable.

    ``weights`` are the prior weights (number of trials for Binomial, precision
    weights elsewhere) and ``dispersion`` the estimated Pearson phi (1 for
    Poisson / binomial / exponential / negative binomial).
    """
    name = type(family).__name__
    if not supports(family):
        return None, None

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    phi = float(dispersion) if dispersion else 1.0
    if not np.isfinite(phi) or phi <= 0.0:
        phi = 1.0
    w = np.maximum(w, 1e-300)

    if name == "Tweedie":
        return _tweedie_cdf_bounds(family, y, mu, phi, w)

    if name == "Gaussian":
        sigma = np.sqrt(phi / w)
        F = stats.norm.cdf(y, loc=mu, scale=sigma)
        return F, F

    if name == "Poisson":
        mu = np.maximum(mu, 1e-300)
        return stats.poisson.cdf(y - 1.0, mu), stats.poisson.cdf(y, mu)

    if name == "NegativeBinomial":
        params = getattr(family, "params", None)
        theta = float(getattr(params, "theta", 1.0) or 1.0)
        theta = max(theta, 1e-8)
        mu = np.maximum(mu, 1e-300)
        p = theta / (theta + mu)
        return stats.nbinom.cdf(y - 1.0, theta, p), stats.nbinom.cdf(y, theta, p)

    if name == "Binomial":
        m = np.rint(w)
        if not (np.all(w == m) and np.all(m >= 1.0)):
            return None, None
        p = np.clip(mu, 1e-15, 1.0 - 1e-15)
        return stats.binom.cdf(y * m - 1.0, m, p), stats.binom.cdf(y * m, m, p)

    if name == "Exponential":
        F = stats.expon.cdf(y, scale=np.maximum(mu, 1e-300))
        return F, F

    if name == "Gamma":
        shape = w / phi
        F = stats.gamma.cdf(y, a=shape, scale=np.maximum(mu, 1e-300) * phi / w)
        return F, F

    if name == "InverseGaussian":
        lam = w / phi
        F = stats.invgauss.cdf(y, mu=np.maximum(mu, 1e-300) * phi / w, scale=lam)
        return F, F

    return None, None


def quantile_residuals(family, y, mu, dispersion=1.0, weights=None, rng=None):
    """Randomized quantile residuals ``Phi^-1(u)`` (~ N(0, 1)), or ``None``.

    ``None`` is returned for families without a computable distribution function
    (or for grouped binomial data with non-integer trial counts), so callers can
    fall back to standardized deviance residuals.  Tweedie zeros are randomised
    over the compound-Poisson point mass at zero.
    """
    if not supports(family):
        return None
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    lo, hi = cdf_bounds(family, y, mu, dispersion, weights)
    if lo is None:
        return None
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    u = np.where(hi > lo + 1e-15, lo + rng.uniform(size=hi.shape) * (hi - lo), hi)
    return stats.norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15))
