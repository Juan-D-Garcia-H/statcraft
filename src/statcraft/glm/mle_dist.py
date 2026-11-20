# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
Distribution layer for the MLE families: mean and variance of y GIVEN ALL the
fitted parameters (linear-predictor blocks + secondary submodel + auxiliary
scalars), not just g^-1(X beta).

Why this exists
---------------
For most MLE families `g^-1(X beta)` is NOT E[y]:

    ZeroInflated*/ValueInflated*   E[y] = (1-pi) lambda  /  pi t + (1-pi) lambda
    Hurdle*                        E[y] = pi * E[Y | Y > 0]
    Truncated*                     mean of the truncated law
    LogNormal/Weibull/LogLogistic  g^-1(eta) is the LOCATION of log(T); E[T] needs sigma
    Gumbel                         E[y] = mu + euler_gamma * sigma

and Var[y] depends on pi, alpha, sigma, nu, phi_i ... so Pearson-type
residuals, prediction standard errors, etc. need every parameter. This module
is pure numpy/scipy (no _glmfit) so it can be checked against scipy.stats /
numerical integration independently of the C++ solver.

Parametrizations mirror log_likelihood.hpp exactly (see each family below).
"""

from types import SimpleNamespace

import numpy as np
from scipy import special as sp
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329

_PROB_EPS = 1e-15


# Inverse links for the secondary parameter (pi in (0,1), phi / sigma^2 > 0).
# Mirrors MleSolver::resolve_secondary_link (valid links) and the C++ inverse
# links / clamps. The PRIMARY link is evaluated by family.linkinv (compiled).
def positive_inverse(eta):
    """Smooth strictly-positive inverse, same as evaluate_positive_inverse_d2 (d = 0.1)."""
    eta = np.asarray(eta, dtype=float)
    d = 0.1
    eta_pos = 0.5 * (eta + np.sqrt(eta * eta + 4.0 * d * d))
    return 1.0 / eta_pos


def inv_link(name, eta):
    """Inverse link g^-1(eta) for the secondary-parameter links."""
    eta = np.asarray(eta, dtype=float)
    if name == "identity":
        return eta
    if name == "log":
        return np.exp(np.clip(eta, -50.0, 50.0))
    if name == "sqrt":
        return eta * eta
    if name == "inverse":
        return positive_inverse(eta)
    if name == "logit":
        return sp.expit(eta)
    if name == "probit":
        return norm.cdf(eta)
    if name == "cloglog":
        return -np.expm1(-np.exp(np.clip(eta, -50.0, 30.0)))
    if name == "loglog":
        return np.exp(-np.exp(np.clip(-eta, -50.0, 30.0)))
    if name == "cauchit":
        return 0.5 + np.arctan(eta) / np.pi
    raise ValueError(f"mle_dist.inv_link: unknown link '{name}'")


def as_prob(p):
    """Clip probabilities to the open interval (``_PROB_EPS``,
    ``1 - _PROB_EPS``) so link and score evaluations stay finite."""
    return np.clip(np.asarray(p, dtype=float), _PROB_EPS, 1.0 - _PROB_EPS)


# Building blocks
def _nb_mean_var_p0(mu, alpha, form="quadratic"):
    """Negative binomial: (mean, var, P(Y=0)). quadratic = NB2 (var = mu + alpha mu^2),
    linear = NB1 (var = mu (1 + alpha)); same as NegativeBinomialComponent."""
    mu = np.asarray(mu, dtype=float)
    if form == "linear":
        var = mu * (1.0 + alpha)
        p0 = np.exp((mu / alpha) * np.log(1.0 / (1.0 + alpha)))
    else:
        var = mu + alpha * mu * mu
        p0 = np.exp(-(1.0 / alpha) * np.log1p(alpha * mu))
    return mu, var, p0


def _hurdle_moments(pi, m, v, p0):
    """Hurdle: P(Y=0) = 1-pi, Y | Y>0 ~ base law truncated at 0 (base mean m, var v, P0 = p0)."""
    pi = as_prob(pi)
    denom = np.maximum(1.0 - p0, 1e-300)
    mean = pi * m / denom
    ey2 = pi * (v + m * m) / denom
    return mean, ey2 - mean * mean


def _mixture_at_value(pi, t, m, v):
    """Point-mass inflation: Y = t w.p. pi, else base law (mean m, var v)."""
    pi = as_prob(pi)
    mean = pi * t + (1.0 - pi) * m
    ey2 = pi * t * t + (1.0 - pi) * (v + m * m)
    return mean, ey2 - mean * mean


def _lognormal_mean_var(loc, sigma):
    s2 = sigma * sigma
    mean = np.exp(loc + 0.5 * s2)
    return mean, np.expm1(s2) * np.exp(2.0 * loc + s2)


def simplex_variance(mu, s2):
    """Variance of the simplex distribution S(mu, sigma^2) (Jorgensen 1997)::

        Var = mu(1-mu) - sqrt(pi/(2 s2)) exp(a) erfc(sqrt(a)),  a = 1/(2 s2 mu^2 (1-mu)^2)

    with E[Y] = mu. Uses the scaled erfcx for numerical stability."""
    mu = np.asarray(mu, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    a = 1.0 / (2.0 * s2 * mu * mu * (1.0 - mu) ** 2)
    # exp(a) * erfc(sqrt(a)) = erfcx(sqrt(a))
    return mu * (1.0 - mu) - np.sqrt(np.pi / (2.0 * s2)) * sp.erfcx(np.sqrt(a))


# Truncated counts: moments by summation of the (normalised) pmf
def _trunc_count_moments(logpmf, mu, left, right, kcap=200000, max_cells=4_000_000):
    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    L = max(int(left), 0)
    if right is not None and right >= 0:
        R = int(right)
    else:
        # generous upper bound of the support that carries the mass
        R = int(
            min(kcap, max(50, np.ceil(mu.max() * 3.0 + 60.0 * np.sqrt(max(mu.max(), 1.0)) + 50.0)))
        )
        # heavy tails (NB with big alpha) are handled by the caller's mu/alpha; capped by kcap
    ks = np.arange(L, R + 1, dtype=float)
    mean = np.empty(n)
    var = np.empty(n)
    chunk = max(1, int(max_cells // max(ks.size, 1)))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        lp = logpmf(ks[None, :], mu[s:e, None])
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        p /= p.sum(axis=1, keepdims=True)
        m1 = p @ ks
        m2 = p @ (ks * ks)
        mean[s:e] = m1
        var[s:e] = m2 - m1 * m1
    return mean, var


def _logpmf_poisson(k, lam):
    lam = np.maximum(lam, 1e-300)
    return -lam + k * np.log(lam) - sp.gammaln(k + 1.0)


def _logpmf_nb2(alpha):
    inv = 1.0 / alpha

    def f(k, mu):
        mu = np.maximum(mu, 1e-10)
        return (
            sp.gammaln(k + inv)
            - sp.gammaln(k + 1.0)
            - sp.gammaln(inv)
            + k * np.log(alpha * mu)
            - (k + inv) * np.log1p(alpha * mu)
        )

    return f


def _logpmf_geometric(k, mu):
    mu = np.maximum(mu, 1e-300)
    return k * np.log(mu) - (k + 1.0) * np.log1p(mu)


def _truncnorm_moments(mu, sigma, left, right, use_left, use_right):
    a = (left - mu) / sigma if use_left else -np.inf * np.ones_like(mu)
    b = (right - mu) / sigma if use_right else np.inf * np.ones_like(mu)
    Z = np.maximum(norm.cdf(b) - norm.cdf(a), 1e-300)
    pa = norm.pdf(a)
    pb = norm.pdf(b)
    a_fin = np.where(np.isfinite(a), a, 0.0)  # avoid inf * 0 at unbounded sides
    b_fin = np.where(np.isfinite(b), b, 0.0)
    a_pa = a_fin * pa
    b_pb = b_fin * pb
    ratio = (pa - pb) / Z
    mean = mu + sigma * ratio
    var = sigma * sigma * (1.0 + (a_pa - b_pb) / Z - ratio * ratio)
    return mean, var


# Public entry point
def make_components(mu=None, sec=None, gamma=None, extra=None, left=None, right=None):
    """Bundle of fitted quantities consumed by `mean_var`.

    mu     : g^-1(X beta) -- count mean / location / mean of the positive part
    sec    : g2^-1(Z gamma) (pi_i, phi_i or sigma^2_i) or None
    gamma  : raw gamma vector (Student-t / Skew-* keep their scalars here)
    extra  : natural-scale independent auxiliary scalar (alpha, sigma, shape, phi)
    left/right : truncation limits (Truncated* families); None = no limit
    """
    return SimpleNamespace(
        mu=None if mu is None else np.asarray(mu, dtype=float),
        sec=None if sec is None else np.asarray(sec, dtype=float),
        gamma=None if gamma is None else np.asarray(gamma, dtype=float),
        extra=extra,
        left=left,
        right=right,
    )


SUPPORTED = (
    "Beta",
    "Simplex",
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ValueInflatedPoisson",
    "ValueInflatedNegativeBinomial",
    "HurdlePoisson",
    "HurdleNegativeBinomial",
    "HurdleGamma",
    "HurdleLogNormal",
    "HurdleBeta",
    "HurdleGeometric",
    "TruncatedPoisson",
    "TruncatedNegativeBinomial",
    "TruncatedGeometric",
    "TruncatedGaussian",
    "StudentT",
    "SkewNormal",
    "SkewT",
    "LogisticDistribution",
    "Gumbel",
    "LogNormal",
    "Weibull",
    "LogLogistic",
)


def supports(family):
    """True if ``family``'s class name is in :data:`statcraft.glm.mle_dist.SUPPORTED`."""
    return type(family).__name__ in SUPPORTED


def mean_var(family, c):
    """(mean, variance) of y for `family` given the components `c`.
    Variance may be +inf where the moment does not exist (e.g. LogLogistic sigma >= 1/2)."""
    name = type(family).__name__
    mu = c.mu

    if name == "Beta":
        phi = np.maximum(c.sec, 1e-4)
        return mu, mu * (1.0 - mu) / (1.0 + phi)

    if name == "Simplex":
        return mu, simplex_variance(mu, np.maximum(c.sec, 1e-6))

    if name == "ZeroInflatedPoisson":
        pi = as_prob(c.sec)
        return (1.0 - pi) * mu, (1.0 - pi) * mu * (1.0 + pi * mu)

    if name == "ZeroInflatedNegativeBinomial":  # NB2, var = mu + alpha mu^2
        pi = as_prob(c.sec)
        a = c.extra
        return (1.0 - pi) * mu, (1.0 - pi) * mu * (1.0 + mu * (a + pi))

    if name == "ValueInflatedPoisson":
        return _mixture_at_value(c.sec, family.target_value, mu, mu)

    if name == "ValueInflatedNegativeBinomial":
        m, v, _ = _nb_mean_var_p0(mu, c.extra, family.var_form)
        return _mixture_at_value(c.sec, family.target_value, m, v)

    if name == "HurdlePoisson":
        return _hurdle_moments(c.sec, mu, mu, np.exp(-np.maximum(mu, 1e-300)))

    if name == "HurdleNegativeBinomial":  # NB2
        m, v, p0 = _nb_mean_var_p0(mu, c.extra, "quadratic")
        return _hurdle_moments(c.sec, m, v, p0)

    if name == "HurdleGamma":  # shape nu: Var = mu^2 / nu
        return _hurdle_moments(c.sec, mu, mu * mu / c.extra, 0.0)

    if name == "HurdleLogNormal":  # mu is the location of log(y)
        m, v = _lognormal_mean_var(mu, c.extra)
        return _hurdle_moments(c.sec, m, v, 0.0)

    if name == "HurdleBeta":  # scalar precision phi
        phi = max(c.extra, 1e-4)
        return _hurdle_moments(c.sec, mu, mu * (1.0 - mu) / (1.0 + phi), 0.0)

    if name == "HurdleGeometric":
        p0_base = 1.0 / (1.0 + np.maximum(mu, 1e-300))
        # base = geometric with mean mu: Var = mu (1 + mu)  (NOT mu, that is the Poisson variance)
        return _hurdle_moments(c.sec, mu, mu * (1.0 + mu), p0_base)

    if name == "TruncatedPoisson":
        return _trunc_count_moments(_logpmf_poisson, mu, c.left or 0, c.right)

    if name == "TruncatedNegativeBinomial":
        return _trunc_count_moments(_logpmf_nb2(c.extra), mu, c.left or 0, c.right)

    if name == "TruncatedGeometric":
        return _trunc_count_moments(_logpmf_geometric, mu, c.left or 0, c.right)

    if name == "TruncatedGaussian":
        return _truncnorm_moments(
            mu, c.extra, c.left, c.right, c.left is not None, c.right is not None
        )

    if name == "StudentT":  # gamma = (log sigma, log(nu - 2))
        sigma = np.exp(c.gamma[0])
        nu = np.exp(c.gamma[1]) + 2.0
        return mu, np.full_like(mu, sigma * sigma * nu / (nu - 2.0))

    if name == "SkewNormal":  # mean-centred; gamma = (log sigma, alpha)
        sigma = np.exp(c.gamma[0])
        alpha = c.gamma[1]
        delta = alpha / np.sqrt(1.0 + alpha * alpha)
        return mu, np.full_like(mu, sigma * sigma * (1.0 - 2.0 * delta * delta / np.pi))

    if name == "SkewT":  # mean-centred; gamma = (log sigma, alpha, log nu)
        sigma = np.exp(c.gamma[0])
        alpha = c.gamma[1]
        nu = np.exp(c.gamma[2])
        delta = alpha / np.sqrt(1.0 + alpha * alpha)
        if nu <= 2.0:
            return mu, np.full_like(mu, np.inf)
        b = np.sqrt(nu / np.pi) * np.exp(sp.gammaln(0.5 * (nu - 1.0)) - sp.gammaln(0.5 * nu))
        return mu, np.full_like(mu, sigma * sigma * (nu / (nu - 2.0) - (b * delta) ** 2))

    if name == "LogisticDistribution":
        return mu, np.full_like(mu, (c.extra * np.pi) ** 2 / 3.0)

    if name == "Gumbel":  # density exp(-(z + e^-z))/sigma  (Gumbel max)
        s = c.extra
        return mu + EULER_GAMMA * s, np.full_like(mu, (np.pi * s) ** 2 / 6.0)

    if name == "LogNormal":  # mu = location of log(T)
        return _lognormal_mean_var(mu, c.extra)

    if name == "Weibull":  # log T = mu + sigma W, W ~ min-Gumbel
        s = c.extra
        g1 = sp.gamma(1.0 + s)
        g2 = sp.gamma(1.0 + 2.0 * s)
        e = np.exp(mu)
        return e * g1, e * e * (g2 - g1 * g1)

    if name == "LogLogistic":  # log T = mu + sigma * standard logistic
        s = c.extra
        e = np.exp(mu)
        if s < 1.0:
            m1 = np.pi * s / np.sin(np.pi * s)
            mean = e * m1
        else:
            mean = np.full_like(mu, np.inf)
        if s < 0.5:
            m2 = 2.0 * np.pi * s / np.sin(2.0 * np.pi * s)
            var = e * e * (m2 - (np.pi * s / np.sin(np.pi * s)) ** 2)
        else:
            var = np.full_like(mu, np.inf)
        return mean, var

    raise NotImplementedError(f"mle_dist: no distribution layer for {name}")


# Per-observation log-likelihood, distribution function, survival, saturated log-likelihood
# (same parametrizations as mean_var / log_likelihood.hpp). Everything is vectorized over
# observations; `c` is the bundle from make_components().
from scipy import stats as _st
from scipy.integrate import cumulative_trapezoid as _cumtrapz
from scipy.integrate import quad as _quad

COUNT_FAMILIES = (
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ValueInflatedPoisson",
    "ValueInflatedNegativeBinomial",
    "HurdlePoisson",
    "HurdleNegativeBinomial",
    "HurdleGeometric",
    "TruncatedPoisson",
    "TruncatedNegativeBinomial",
    "TruncatedGeometric",
)
SURVIVAL_FAMILIES = ("LogNormal", "Weibull", "LogLogistic")  # censored AFT (indicator d)
# The Weibull / LogLogistic likelihoods in log_likelihood.hpp omit the Jacobian term -log(t):
# the proper log-likelihood differs from the reported number by sum(d * log t).
LL_OMITS_LOG_T = ("Weibull", "LogLogistic")


def _count_base(name, family, c):
    """Frozen scipy count law of the underlying (untruncated / uninflated) component."""
    mu = np.maximum(c.mu, 1e-300)
    if name in ("ZeroInflatedPoisson", "ValueInflatedPoisson", "HurdlePoisson", "TruncatedPoisson"):
        return _st.poisson(mu)
    if name in ("TruncatedGeometric", "HurdleGeometric"):  # P(k) = mu^k / (1+mu)^(k+1)  (mean mu)
        return _st.nbinom(1.0, 1.0 / (1.0 + mu))
    form = (
        getattr(family, "var_form", "quadratic")
        if name == "ValueInflatedNegativeBinomial"
        else "quadratic"
    )
    a = c.extra
    if form == "linear":
        return _st.nbinom(mu / a, np.full_like(mu, 1.0 / (1.0 + a)))
    return _st.nbinom(1.0 / a, 1.0 / (1.0 + a * mu))


def _simplex_pdf(y, mu, s2):
    y = np.clip(y, 1e-300, 1 - 1e-16)
    d = (y - mu) ** 2 / (y * (1 - y) * mu**2 * (1 - mu) ** 2)
    return (2 * np.pi * s2 * (y * (1 - y)) ** 3) ** -0.5 * np.exp(-d / (2 * s2))


def _skewt_logpdf(y, xi, om, al, nu):
    z = (y - xi) / om
    return (
        np.log(2.0)
        - np.log(om)
        + _st.t.logpdf(z, nu)
        + _st.t.logcdf(al * z * np.sqrt((nu + 1.0) / (nu + z * z)), nu + 1.0)
    )


def _skewt_logpdf_std(z, alpha, nu):
    """Log density of the *standard* skew-t (location 0, scale 1).

    Vectorised equivalent of ``_skewt_logpdf(z, 0.0, 1.0, alpha, nu)`` using
    ``scipy.special`` directly; accurate to machine precision but much faster
    than ``scipy.stats.t.logpdf/logcdf`` for large arrays.
    """
    z = np.asarray(z, dtype=float)
    log_t = (
        sp.gammaln(0.5 * (nu + 1.0))
        - sp.gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
    )
    w = alpha * z * np.sqrt((nu + 1.0) / (nu + z * z))
    df = nu + 1.0
    ib = sp.betainc(0.5 * df, 0.5, df / (df + w * w))
    log_tail = np.where(
        w <= 0.0,
        np.log(0.5) + np.log(np.maximum(ib, 1e-300)),
        np.log1p(-0.5 * ib),
    )
    return np.log(2.0) + log_t + log_tail


def _skewt_cdf_std(z, alpha, nu, *, n_grid: int = 20001) -> np.ndarray:
    """CDF of the standard skew-t, integrated once on a fixed tan grid.

    The former implementation called ``scipy.integrate.quad`` from -inf to y
    for every observation (about 24 s for one residual vector).  Integrating
    in ``theta`` after ``z = tan(theta)`` handles the tails analytically and
    lets a single vectorised pass serve all observations (~0.01 s).  The
    result agrees with direct quadrature to better than 2e-8.
    """
    z = np.asarray(z, dtype=float)
    theta = np.linspace(-np.pi / 2 + 1e-12, np.pi / 2 - 1e-12, int(n_grid))
    grid = np.tan(theta)
    jacobian = 1.0 / np.cos(theta) ** 2
    density = np.exp(_skewt_logpdf_std(grid, alpha, nu)) * jacobian
    cdf = _cumtrapz(density, theta, initial=0.0)
    cdf /= cdf[-1]
    return np.interp(z, grid, cdf)


def _skewt_shift(sigma, alpha, nu):
    delta = alpha / np.sqrt(1.0 + alpha * alpha)
    return (
        sigma
        * delta
        * np.sqrt(nu / np.pi)
        * np.exp(sp.gammaln(0.5 * (nu - 1.0)) - sp.gammaln(0.5 * nu))
    )


def _surv_params(name, c):
    """(location m of log T, scale sigma) for the AFT families."""
    return c.mu, c.extra


def logsurvival(family, c, t):
    """log S(t) for LogNormal / Weibull / LogLogistic (mu = location of log T, extra = sigma)."""
    name = type(family).__name__
    z = (np.log(np.maximum(t, 1e-300)) - c.mu) / c.extra
    if name == "LogNormal":
        return _st.norm.logsf(z)
    if name == "Weibull":
        return -np.exp(np.clip(z, -700, 700))
    if name == "LogLogistic":
        return -np.logaddexp(0.0, z)
    raise NotImplementedError(f"logsurvival: {name}")


def logpdf(family, c, y, d=None):
    """Per-observation log-likelihood l_i(theta). For the AFT families `d` is the event indicator
    (1 = event, 0 = right-censored; None = all events) and censored rows contribute log S(t).
    Proper log-density (Weibull / LogLogistic INCLUDE -log t; see LL_OMITS_LOG_T)."""
    name = type(family).__name__
    y = np.asarray(y, dtype=float)
    mu = c.mu

    if name == "Beta":
        phi = np.maximum(c.sec, 1e-4)
        yy = np.clip(y, 1e-12, 1 - 1e-12)
        return _st.beta.logpdf(yy, mu * phi, (1 - mu) * phi)

    if name == "Simplex":
        return np.log(np.maximum(_simplex_pdf(y, mu, np.maximum(c.sec, 1e-6)), 1e-300))

    if name in COUNT_FAMILIES:
        base = _count_base(name, family, c)
        if name.startswith("ZeroInflated"):
            pi = as_prob(c.sec)
            return np.where(
                y == 0, np.log(pi + (1 - pi) * base.pmf(0)), np.log1p(-pi) + base.logpmf(y)
            )
        if name.startswith("ValueInflated"):
            pi = as_prob(c.sec)
            t = family.target_value
            return np.where(
                y == t, np.log(pi + (1 - pi) * base.pmf(t)), np.log1p(-pi) + base.logpmf(y)
            )
        if name.startswith("Hurdle"):
            pi = as_prob(c.sec)
            p0 = base.pmf(0)
            return np.where(
                y == 0,
                np.log1p(-pi),
                np.log(pi) + base.logpmf(y) - np.log1p(-np.minimum(p0, 1 - 1e-16)),
            )
        # truncated counts on [L, R]
        L = int(c.left or 0)
        R = c.right if (c.right is not None and c.right >= 0) else None
        upper = base.cdf(R) if R is not None else 1.0
        mass = np.maximum(upper - base.cdf(L - 1), 1e-300)
        return base.logpmf(y) - np.log(mass)

    if name in ("HurdleGamma", "HurdleLogNormal", "HurdleBeta"):
        pi = as_prob(c.sec)
        pos = y > 0
        ys = np.where(pos, y, 1.0)
        if name == "HurdleGamma":
            nu = c.extra
            lp = _st.gamma.logpdf(ys, a=nu, scale=mu / nu)
        elif name == "HurdleLogNormal":
            lp = _st.lognorm.logpdf(ys, s=c.extra, scale=np.exp(mu))
        else:
            phi = max(c.extra, 1e-4)
            lp = _st.beta.logpdf(np.clip(ys, 1e-12, 1 - 1e-12), mu * phi, (1 - mu) * phi)
        return np.where(pos, np.log(pi) + lp, np.log1p(-pi))

    if name == "TruncatedGaussian":
        s = c.extra
        a = (c.left - mu) / s if c.left is not None else -np.inf
        b = (c.right - mu) / s if c.right is not None else np.inf
        return _st.truncnorm.logpdf(y, a, b, loc=mu, scale=s)

    if name == "StudentT":
        sigma = np.exp(c.gamma[0])
        nu = np.exp(c.gamma[1]) + 2.0
        return _st.t.logpdf(y, nu, loc=mu, scale=sigma)

    if name == "SkewNormal":
        sigma = np.exp(c.gamma[0])
        alpha = c.gamma[1]
        delta = alpha / np.sqrt(1 + alpha * alpha)
        xi = mu - sigma * delta * np.sqrt(2 / np.pi)
        return _st.skewnorm.logpdf(y, alpha, loc=xi, scale=sigma)

    if name == "SkewT":
        sigma = np.exp(c.gamma[0])
        alpha = c.gamma[1]
        nu = np.exp(c.gamma[2])
        return _skewt_logpdf(y, mu - _skewt_shift(sigma, alpha, nu), sigma, alpha, nu)

    if name == "LogisticDistribution":
        return _st.logistic.logpdf(y, loc=mu, scale=c.extra)

    if name == "Gumbel":
        return _st.gumbel_r.logpdf(y, loc=mu, scale=c.extra)

    if name in SURVIVAL_FAMILIES:
        t = np.maximum(y, 1e-300)
        s = c.extra
        z = (np.log(t) - mu) / s
        if name == "LogNormal":
            lf = _st.norm.logpdf(z) - np.log(s) - np.log(t)
        elif name == "Weibull":
            lf = -np.log(s) - np.log(t) + z - np.exp(np.clip(z, -700, 700))
        else:
            lf = -np.log(s) - np.log(t) + z - 2.0 * np.logaddexp(0.0, z)
        if d is None:
            return lf
        d = np.asarray(d, dtype=float)
        return np.where(d > 0.5, lf, logsurvival(family, c, t))

    raise NotImplementedError(f"mle_dist.logpdf: {name}")


def cdf_bounds(family, c, y):
    """(F(y-), F(y)): left limit and value of the distribution function at y. Equal for continuous laws; the gap
    is the point mass for discrete / zero-inflated / hurdle laws (used by randomized quantile residuals)."""
    name = type(family).__name__
    y = np.asarray(y, dtype=float)
    mu = c.mu

    if name in COUNT_FAMILIES:
        base = _count_base(name, family, c)
        if name.startswith("ZeroInflated"):
            pi = as_prob(c.sec)
            F = lambda k: np.where(k < 0, 0.0, pi + (1 - pi) * base.cdf(k))
            return F(y - 1), F(y)
        if name.startswith("ValueInflated"):
            pi = as_prob(c.sec)
            t = family.target_value
            F = lambda k: np.where(k < 0, 0.0, (1 - pi) * base.cdf(k) + pi * (k >= t))
            return F(y - 1), F(y)
        if name.startswith("Hurdle"):
            pi = as_prob(c.sec)
            p0 = np.minimum(base.pmf(0), 1 - 1e-16)
            F = lambda k: np.where(
                k < 0, 0.0, np.where(k < 1, 1 - pi, (1 - pi) + pi * (base.cdf(k) - p0) / (1 - p0))
            )
            return F(y - 1), F(y)
        L = int(c.left or 0)
        R = c.right if (c.right is not None and c.right >= 0) else None
        lowmass = base.cdf(L - 1)
        mass = np.maximum((base.cdf(R) if R is not None else 1.0) - lowmass, 1e-300)
        F = lambda k: np.where(k < L, 0.0, np.clip((base.cdf(k) - lowmass) / mass, 0.0, 1.0))
        return F(y - 1), F(y)

    if name in ("HurdleGamma", "HurdleLogNormal", "HurdleBeta"):
        pi = as_prob(c.sec)
        pos = y > 0
        ys = np.where(pos, y, 1.0)
        if name == "HurdleGamma":
            G = _st.gamma.cdf(ys, a=c.extra, scale=mu / c.extra)
        elif name == "HurdleLogNormal":
            G = _st.lognorm.cdf(ys, s=c.extra, scale=np.exp(mu))
        else:
            phi = max(c.extra, 1e-4)
            G = _st.beta.cdf(np.clip(ys, 1e-12, 1 - 1e-12), mu * phi, (1 - mu) * phi)
        F = (1 - pi) + pi * G
        return np.where(pos, F, 0.0), np.where(pos, F, 1 - pi)

    if name == "Beta":
        phi = np.maximum(c.sec, 1e-4)
        F = _st.beta.cdf(np.clip(y, 1e-12, 1 - 1e-12), mu * phi, (1 - mu) * phi)
    elif name == "Simplex":
        s2 = np.maximum(c.sec, 1e-6)
        F = np.empty_like(y)
        for i in range(y.shape[0]):
            yi, mi, si = float(np.clip(y[i], 1e-12, 1 - 1e-12)), float(mu[i]), float(s2[i])
            pts = [mi] if 0 < mi < yi else None
            F[i] = _quad(
                lambda u: float(_simplex_pdf(np.array(u), mi, si)),
                0.0,
                yi,
                points=pts,
                limit=200,
                epsabs=1e-13,
                epsrel=1e-11,
            )[0]
    elif name == "TruncatedGaussian":
        s = c.extra
        a = (c.left - mu) / s if c.left is not None else -np.inf
        b = (c.right - mu) / s if c.right is not None else np.inf
        F = _st.truncnorm.cdf(y, a, b, loc=mu, scale=s)
    elif name == "StudentT":
        F = _st.t.cdf(y, np.exp(c.gamma[1]) + 2.0, loc=mu, scale=np.exp(c.gamma[0]))
    elif name == "SkewNormal":
        sigma = np.exp(c.gamma[0])
        alpha = c.gamma[1]
        delta = alpha / np.sqrt(1 + alpha * alpha)
        F = _st.skewnorm.cdf(y, alpha, loc=mu - sigma * delta * np.sqrt(2 / np.pi), scale=sigma)
    elif name == "SkewT":
        sigma = np.exp(c.gamma[0])
        alpha = c.gamma[1]
        nu = np.exp(c.gamma[2])
        xi = mu - _skewt_shift(sigma, alpha, nu)
        F = _skewt_cdf_std((y - xi) / sigma, alpha, nu)
    elif name == "LogisticDistribution":
        F = _st.logistic.cdf(y, loc=mu, scale=c.extra)
    elif name == "Gumbel":
        F = _st.gumbel_r.cdf(y, loc=mu, scale=c.extra)
    elif name in SURVIVAL_FAMILIES:
        F = -np.expm1(logsurvival(family, c, y))
    else:
        raise NotImplementedError(f"mle_dist.cdf_bounds: {name}")
    return F, F


def _with_mu(c, mu):
    return SimpleNamespace(
        mu=mu, sec=c.sec, gamma=c.gamma, extra=c.extra, left=c.left, right=c.right
    )


def _mu_search_domain(name, c, y):
    """(lo, hi, transform) of the search interval for the primary parameter mu when computing the saturated
    log-likelihood. transform: 'log' (mu > 0), 'logit' (mu in (0,1)) or 'lin'."""
    y = np.asarray(y, dtype=float)
    scale = None
    if name in COUNT_FAMILIES:
        return np.full_like(y, 1e-8), 3.0 * y + 20.0, "log"
    if name == "HurdleGamma":
        return np.full_like(y, 1e-10), 20.0 * np.maximum(y, 1e-6) + 1.0, "log"
    if name in ("Beta", "Simplex", "HurdleBeta"):
        return np.full_like(y, 1e-9), np.full_like(y, 1 - 1e-9), "logit"
    if name in ("StudentT", "SkewNormal", "SkewT"):
        scale = np.exp(c.gamma[0])
    elif name in ("LogisticDistribution", "Gumbel", "TruncatedGaussian"):
        scale = c.extra
    elif name in SURVIVAL_FAMILIES or name == "HurdleLogNormal":
        scale = c.extra
        ly = np.log(np.maximum(y, 1e-300))
        return ly - 15.0 * scale, ly + 15.0 * scale, "lin"
    s = 15.0 * (scale if np.isscalar(scale) else np.max(scale))
    return y - s - 1.0, y + s + 1.0, "lin"


def saturated_loglik(family, c, y, d=None, n_iter=90):
    """Per-observation maximum of l_i over the primary parameter mu with every other parameter (pi, alpha, sigma,
    nu, phi_i ...) held at its fitted value; the reference for deviance residuals::

        d_i = 2 (l_i^sat - l_i),   r_D,i = sign(y_i - E[y_i]) sqrt(d_i).

    Golden-section search (vectorized over observations) on a log / logit / linear scale."""
    name = type(family).__name__
    y = np.asarray(y, dtype=float)
    lo, hi, tr = _mu_search_domain(name, c, y)
    if tr == "log":
        fwd, inv = np.log, np.exp
    elif tr == "logit":
        fwd, inv = (lambda m: np.log(m / (1 - m))), sp.expit
    else:
        fwd, inv = (lambda m: m), (lambda m: m)
    a, b = fwd(lo), fwd(hi)
    f = lambda t: logpdf(family, _with_mu(c, inv(t)), y, d)
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(n_iter):
        left = f1 > f2  # max in [a, x2]
        b = np.where(left, x2, b)
        a = np.where(left, a, x1)
        x1, x2 = b - gr * (b - a), a + gr * (b - a)
        f1, f2 = f(x1), f(x2)
    best = f(0.5 * (a + b))
    # the maximum may sit exactly on a boundary (e.g. ZIP with y = 0 -> mu -> 0): compare with the ends
    sat = np.maximum(best, np.maximum(f(fwd(lo)), f(fwd(hi))))
    if name in SURVIVAL_FAMILIES and d is not None:
        # a censored row contributes log S(t), whose supremum over the location is exactly 0 (S -> 1)
        sat = np.where(np.asarray(d, dtype=float) > 0.5, sat, 0.0)
    return sat


# Generalized Gamma (Prentice 1974 / flexsurv parametrization; matches gengamma.hpp exactly).
#   Q != 0 :  V = exp(Q W) / Q^2 ~ Gamma(shape = 1/Q^2, scale = 1),   W = (log T - mu) / sigma
#             T = exp(mu) * (Q^2)^(-sigma/Q) * V^(sigma/Q)
#   Q == 0 :  log T ~ Normal(mu, sigma^2)                              (the lognormal limit)
# Derived by substitution from the density in gengamma.hpp; verified below against scipy (Q -> 0, |Q| = 1
# reduces to Weibull/Gamma via known identities) and against numerical integration of the density itself.
def gengamma_logpdf(t, mu, sigma, Q):
    """Log density of the generalized gamma at ``t`` with parameters
    ``mu``, ``sigma`` and shape ``Q``; ``Q -> 0`` takes the lognormal
    limit exactly (see :func:`statcraft.glm.mle_dist.gengamma_logsurv`)."""
    t, mu, sigma, Q = np.broadcast_arrays(
        np.asarray(t, float), np.asarray(mu, float), np.asarray(sigma, float), np.asarray(Q, float)
    )
    Q = np.where(np.abs(Q) < 1e-4, np.where(Q >= 0, 1e-4, -1e-4), Q)
    y = np.log(np.maximum(t, 1e-300))
    w = (y - mu) / sigma
    small = np.abs(Q) < 1e-4
    out = np.empty(t.shape)
    if np.any(small):
        s = sigma[small]
        out[small] = -0.5 * np.log(2 * np.pi) - np.log(s) - y[small] - 0.5 * w[small] ** 2
    big = ~small
    if np.any(big):
        qi = 1.0 / Q[big] ** 2
        qw = Q[big] * w[big]
        out[big] = (
            -np.log(sigma[big] * t[big])
            + np.log(np.abs(Q[big]))
            + qi * np.log(qi)
            + qi * (qw - np.exp(qw))
            - sp.gammaln(qi)
        )
    return out


def gengamma_logsurv(t, mu, sigma, Q):
    """log S(t) = log P(T > t), matching gengamma::cdf(lower_tail=false)."""
    from scipy import special as _sp

    t, mu, sigma, Q = np.broadcast_arrays(
        np.asarray(t, float), np.asarray(mu, float), np.asarray(sigma, float), np.asarray(Q, float)
    )
    out = np.zeros(t.shape)
    pos = t > 0
    if not np.any(pos):
        return out
    y = np.log(t[pos])
    w = (y - mu[pos]) / sigma[pos]
    Qp = Q[pos]
    small = np.abs(Qp) < 1e-4
    r = np.empty(w.shape)
    if np.any(small):
        r[small] = _sp.log_ndtr(-w[small])
    if np.any(~small):
        qq = 1.0 / Qp[~small] ** 2
        expnu = qq * np.exp(Qp[~small] * w[~small])
        upper = Qp[~small] > 0  # upper tail of the gamma when Q > 0 (survival(lower_tail=False))
        r_big = np.empty(expnu.shape)
        r_big[upper] = _sp.gammaincc(qq[upper], expnu[upper])
        r_big[~upper] = _sp.gammainc(qq[~upper], expnu[~upper])
        r[~small] = np.log(np.clip(r_big, 1e-15, 1 - 1e-15))
    out[pos] = r
    return out


def gengamma_mean_var_truncated(mu, sigma, Q, entry):
    """(E[T | T > entry], Var[T | T > entry]) for entry > 0 (rows with entry <= 0 should use gengamma_mean_var
    instead: this function assumes P(T > entry) > 0 for every row).

    Using T = exp(mu) qi^(-sigma/Q) V^(sigma/Q), V ~ Gamma(qi, 1), qi = 1/Q^2 (see gengamma_mean_var): T > entry
    corresponds to V > v_e (if sigma/Q > 0) or V < v_e (if sigma/Q < 0), v_e = qi (entry exp(-mu))^(Q/sigma). Hence::

        E[T^r 1{T>entry}] = exp(mu r) qi^(-r sigma/Q) Gamma(qi + r sigma/Q) / Gamma(qi) * R(qi + r sigma/Q, v_e),

    R = the upper (sigma/Q > 0) or lower (sigma/Q < 0) regularized incomplete gamma function -- so, unlike the
    unconditional moments, these can be finite even where qi + r sigma/Q <= 0 (excluding V near 0, the
    non-existence region, whenever sigma/Q > 0 -- which is always finite for Q > 0 truncated OR not, and for
    Q < 0 only escapes non-existence for entry-truncation when sigma/Q > 0 too, i.e. never for Q < 0 -- kept as
    NaN there, exactly like the unconditional case). Divide by S(entry) = P(T > entry) to get the conditional moment."""
    from scipy import special as _sp

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    Q = np.asarray(Q, dtype=float)
    entry = np.asarray(entry, dtype=float)
    mu, sigma, Q, entry = np.broadcast_arrays(mu, sigma, Q, entry)
    out_m = np.empty(mu.shape)
    out_v = np.empty(mu.shape)
    logS = gengamma_logsurv(entry, mu, sigma, Q)
    small = np.abs(Q) < 1e-4
    if np.any(small):
        # lognormal limit: E[T^r | T>a] via the truncated-normal moment generating function of log T
        s2 = sigma[small] ** 2
        z = (np.log(np.maximum(entry[small], 1e-300)) - mu[small]) / sigma[small]
        Sr = np.exp(_sp.log_ndtr(-z))
        m1 = np.exp(mu[small] + s2 / 2.0) * np.exp(_sp.log_ndtr(-(z - sigma[small]))) / Sr
        m2 = np.exp(2 * mu[small] + 2 * s2) * np.exp(_sp.log_ndtr(-(z - 2 * sigma[small]))) / Sr
        out_m[small] = m1
        out_v[small] = m2 - m1**2
    big = ~small
    if np.any(big):
        m_, s_, q_, e_ = mu[big], sigma[big], Q[big], entry[big]
        qi = 1.0 / q_**2
        upper = (s_ / q_) > 0  # T increasing in V  <=>  T > entry means V > v_e
        v_e = qi * (e_ * np.exp(-m_)) ** (q_ / s_)
        Sc = np.exp(logS[big])

        def cond_moment(r):
            a = r * s_ / q_
            ok = qi + a > 0
            R = np.where(
                upper,
                _sp.gammaincc(np.where(ok, qi + a, 1.0), v_e),
                _sp.gammainc(np.where(ok, qi + a, 1.0), v_e),
            )
            val = np.full(a.shape, np.nan)
            good = ok & (Sc > 0)
            val[good] = (
                np.exp(m_[good] * r)
                * qi[good] ** (-a[good])
                * np.exp(_sp.gammaln(qi[good] + a[good]) - _sp.gammaln(qi[good]))
                * R[good]
                / Sc[good]
            )
            return val

        m1 = cond_moment(1.0)
        m2 = cond_moment(2.0)
        out_m[big] = m1
        out_v[big] = m2 - m1**2
    return out_m, out_v


def gengamma_mean_var(mu, sigma, Q):
    """(E[T], Var[T]) via E[T^r] = exp(mu r) (1/Q^2)^(-r sigma/Q) Gamma(1/Q^2 + r sigma/Q) / Gamma(1/Q^2),
    T = exp(mu) qi^(-sigma/Q) V^(sigma/Q), V ~ Gamma(qi, 1), qi = 1/Q^2 (see derivation above); the Q -> 0
    limit is the lognormal. NaN where the moment does not exist (qi + r sigma/Q <= 0)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    Q = np.asarray(Q, dtype=float)
    mu, sigma, Q = np.broadcast_arrays(mu, sigma, Q)
    mu, sigma, Q = mu.copy(), sigma.copy(), Q.copy()
    out_m = np.empty(mu.shape)
    out_v = np.empty(mu.shape)
    small = (
        np.abs(Q) < 1e-4
    )  # same cutover as gengamma_logpdf, so mean_var and logpdf agree on the lognormal boundary
    if np.any(small):
        m, v = _lognormal_mean_var(mu[small], sigma[small])
        out_m[small], out_v[small] = m, v
    big = ~small
    if np.any(big):
        m_, s_, q_ = mu[big], sigma[big], Q[big]
        qi = 1.0 / q_**2

        def moment(r):
            a = r * s_ / q_
            ok = qi + a > 0
            val = np.full(a.shape, np.nan)
            val[ok] = (
                np.exp(m_[ok] * r)
                * qi[ok] ** (-a[ok])
                * np.exp(sp.gammaln(qi[ok] + a[ok]) - sp.gammaln(qi[ok]))
            )
            return val

        m1 = moment(1.0)
        m2 = moment(2.0)
        out_m[big] = m1
        out_v[big] = m2 - m1**2
    return out_m, out_v
