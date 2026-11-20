# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostics for MLE families built on the per-observation log-likelihood (mle_dist.logpdf).

* quantile_residuals   Dunn & Smyth (1996): r = Phi^-1(u), u = F(y) (continuous) or u ~ U(F(y-), F(y)) (discrete,
                       zero-inflated, hurdle). Exactly N(0,1) if the model is right, for ANY of the families.
* deviance_residuals   r_D = sign * sqrt(2 (l_sat - l)), l_sat = max over the primary parameter mu with the other
                       parameters (pi, alpha, sigma, nu, phi_i ...) held at their fitted values.
* cox_snell / martingale / deviance_martingale   censored survival (AFT) families,
                        including GeneralizedGamma and left-truncated (delayed-entry) rows.
* information_measures leverage, leave-one-out parameter changes and Cook's distance from the observed
                       information, by numerical differentiation of l_i(theta):

      s_i = grad l_i,   I_i = -Hess l_i,   I = sum I_i,
      h_i = tr(I^-1 I_i)                       (information leverage: sum h_i = p; for a GLM it IS the hat value)
      Delta_i = (I - I_i)^-1 s_i               (one-step Newton estimate of theta_hat - theta_hat_(i))
      D_i = Delta_i' I Delta_i / p             (likelihood-distance version of Cook's D)
"""

import numpy as np
from scipy import stats

try:  # package import (glm.py) or plain-module import (stand-alone tests)
    from . import mle_dist as _dist
except ImportError:
    import mle_dist as _dist  # type: ignore[no-redef]


def quantile_residuals(family, c, y, rng=None):
    """Randomized quantile residuals: cdf-based normal quantiles. Discrete
    support gaps are broken by a uniform draw inside (F(y-), F(y));
    ``rng`` seeds that draw."""
    lo, hi = _dist.cdf_bounds(family, c, y)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    u = np.where(hi > lo + 1e-15, lo + rng.uniform(size=hi.shape) * (hi - lo), hi)
    return stats.norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15))


def deviance_residuals(family, c, y, mean, d=None, sign_ref=None):
    """sign(y - mean) sqrt(2 (l_sat - l)). `sign_ref` overrides the sign reference (e.g. the martingale residual)."""
    sat = _dist.saturated_loglik(family, c, y, d)
    dev = 2.0 * np.maximum(sat - _dist.logpdf(family, c, y, d), 0.0)
    sgn = np.sign(np.asarray(y, dtype=float) - mean) if sign_ref is None else np.sign(sign_ref)
    sgn = np.where(sgn == 0, 1.0, sgn)
    return sgn * np.sqrt(dev)


def log_survival(family, c, t, entry=None):
    """log S(t) for an AFT / GeneralizedGamma row, optionally conditional on delayed entry.

    ``c`` is either the usual MLE component namespace (AFT families) or a
    namespace with ``mu``/``sigma``/``Q`` arrays for GeneralizedGamma.
    Rows with ``entry >= t`` are invalid and get log S = -inf; their Cox-Snell
    residual is set to NaN by :func:`cox_snell`.
    """
    t = np.asarray(t, dtype=float)
    if type(family).__name__ == "GeneralizedGamma":
        logS = _dist.gengamma_logsurv(t, c.mu, c.sigma, c.Q)
        logS_entry = (
            None if entry is None else _dist.gengamma_logsurv(entry, c.mu, c.sigma, c.Q)
        )
    else:
        logS = _dist.logsurvival(family, c, t)
        logS_entry = None if entry is None else _dist.logsurvival(family, c, entry)
    if logS_entry is not None:
        logS = np.asarray(logS) - np.asarray(logS_entry)
        logS = np.where(np.asarray(entry) >= t, -np.inf, logS)
    return logS


def cox_snell(family, c, t, entry=None):
    """-log S(t): unit exponential if the model is right (censored rows are censored draws)."""
    r = -np.asarray(log_survival(family, c, t, entry), dtype=float)
    return np.where(np.isfinite(r), r, np.nan)


def martingale(family, c, t, d, entry=None):
    """Martingale residual for a survival/AFT model: ``d - (-log S(t))``,
    the difference between the event indicator and the integrated hazard."""
    return np.asarray(d, dtype=float) - cox_snell(family, c, t, entry)


def deviance_martingale_from_martingale(m, d):
    """Symmetric deviance transform of an already computed martingale residual."""
    m = np.asarray(m, dtype=float)
    d = np.asarray(d, dtype=float)
    inner = m + np.where(d > 0.5, np.log(np.maximum(d - m, 1e-300)), 0.0)
    out = np.sign(m) * np.sqrt(np.maximum(-2.0 * inner, 0.0))
    return np.where(np.isfinite(m), out, np.nan)


def deviance_martingale(family, c, t, d, entry=None):
    """Deviance residual for a survival/AFT model: the usual transformation
    ``sign(m) sqrt(-2 (m + d log(d - m)))`` of the martingale residual
    ``m``, symmetric on the positive side."""
    return deviance_martingale_from_martingale(
        martingale(family, c, t, d, entry), d
    )


def _num_grad(f, theta, steps):
    p = theta.shape[0]
    cols = []
    for j in range(p):
        e = np.zeros(p)
        e[j] = steps[j]
        cols.append((f(theta + e) - f(theta - e)) / (2.0 * steps[j]))
    return np.column_stack(cols)  # n x p


def information_measures(loglik_obs, theta):
    """loglik_obs: theta -> (n,) per-observation log-likelihood. Returns dict with the scores S (n x p), the
    per-observation informations Ii (n x p x p), the total information I (p x p), leverage h, dfbeta (n x p),
    and Cook's distance."""
    theta = np.asarray(theta, dtype=float)
    p = theta.shape[0]
    h1 = 1e-5 * np.maximum(1.0, np.abs(theta))
    h2 = 1e-4 * np.maximum(1.0, np.abs(theta))
    S = _num_grad(loglik_obs, theta, h1)
    n = S.shape[0]
    Hess = np.empty((n, p, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = h2[j]
        Hess[:, :, j] = (
            _num_grad(loglik_obs, theta + e, h1) - _num_grad(loglik_obs, theta - e, h1)
        ) / (2.0 * h2[j])
    Ii = -0.5 * (Hess + np.transpose(Hess, (0, 2, 1)))  # symmetrize
    I = Ii.sum(axis=0)
    Iinv = np.linalg.pinv(I)
    lev = np.einsum("ab,nba->n", Iinv, Ii)  # tr(I^-1 I_i)
    delta = np.empty((n, p))
    for i in range(n):
        delta[i] = np.linalg.solve(I - Ii[i] + 1e-12 * np.eye(p), S[i])
    cook = np.einsum("ni,ij,nj->n", delta, I, delta) / p
    return {"S": S, "Ii": Ii, "I": I, "hat": lev, "dfbeta": delta, "cooks": cook}
