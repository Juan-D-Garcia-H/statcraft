# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
family.py -- R-style Family/Link objects, in pure Python, on top of the
raw engine exposed by _glmfit (glm.hpp via pybind11).

FamilyCode is generated dynamically from the FAM_* attributes exposed by
the compiled module -- single source of truth: if families.hpp changes
and the binding is recompiled, this enum updates itself, with no
duplicated manual maintenance in Python.

Usage (direct import -- see "USAGE CONVENTION" in glm.py; family classes
are used this way, `from statcraft import Binomial, ...`, not namespaced):
    from statcraft import Binomial, NegativeBinomial, Tweedie, QuasiPoisson, Beta

    fam = Binomial()                  # logit (canonical) by default
    fam = Binomial(link="probit")     # R-style strings (glm(family=binomial(link="probit")))
    fam = NegativeBinomial(theta=5.0) # families with an extra parameter
    fam = Tweedie(var_power=1.5, link="log")
    fam = Gamma(link="power", link_power=0.25)  # Box-Cox/power link, custom exponent
    fam = QuasiPoisson()              # quasi family -- see below
    fam = Beta()                      # MLE family -- see below

    fam.code    -> FamilyCode (int)   ready for _glmfit.IrlsEngine(..., int(fam.code), fam.params)
    fam.link    -> str, the link exactly as passed, lowercased ("logit", "log", ...)
    fam.params  -> _glmfit.FamilyParams (theta/var_power), ALWAYS present
                   (empty/with defaults in families that don't use it)
    fam.family_type -> "irls" | "quasi" | "mle" | "mle_inflated" | "mle_gengamma"
                   -- which engine GLM.fit() (glm.py) uses internally for
                   this family. GLM's public interface (constructor,
                   .fit(), .coefficients, .fitted_values, .summary(), etc.)
                   is identical across all cases; family_type is the only
                   thing that changes the internal code path -- see
                   glm.py::GLM.fit().

Links (plain strings, nothing to import from this module to specify them):
    "identity", "log", "inverse", "logit", "probit", "cloglog", "sqrt",
    "invmu2", "loglog", "cauchit", "power" (Box-Cox/power link, mu =
    eta**(1/link_power) -- Gamma/InverseGaussian/NegativeBinomial/Tweedie
    only, requires link_power=... in the constructor), "logit_log" (alias
    for "logit", see Binomial below). Case-insensitive -- normalized to
    lowercase in Family.__init__. An invalid link for the family raises
    ValueError with the list of valid links for that family.

Secondary link (MLE families that model a 2nd parameter through their own
linear predictor Z -- constructor argument, plain string, same style as `link`):
    Beta(phi_link="log")             precision phi > 0     -> log | identity | sqrt | inverse
    Simplex(sigma_link="log")        dispersion sigma^2 > 0 -> log | identity | sqrt | inverse
    Hurdle*(zero_link="logit")       P(y > 0)              -> logit | probit | cloglog | loglog | cauchit
    ZeroInflated*(infl_link="logit") P(inflation)          -> logit | probit | cloglog | loglog | cauchit
    ValueInflated*(infl_link="logit")  (idem)
    fam.secondary_link -> the string in use (None for families without a 2nd predictor).
    An invalid link for that parameter raises ValueError at construction.
    Families whose extra parameters are plain scalars (StudentT, SkewNormal,
    SkewT, Weibull, ...) have no secondary link; GeneralizedGamma already
    takes link_mu / link_sigma / link_Q.

Five family groups, five different engines underneath the same GLM:

  1. Native IRLS (family_type="irls"): Gaussian, Binomial, Poisson, Gamma,
     InverseGaussian, NegativeBinomial, Tweedie, Exponential, Pareto,
     InverseGamma, CustomFamily. Full dispatch in families.hpp
     (linkinv/mu_eta/variance/dev_resids_sum) -> _glmfit.IrlsEngine.

  2. Quasi-likelihood (family_type="quasi"): QuasiPoisson, QuasiBinomial,
     QuasiGamma, QuasiInverseGaussian, QuasiGaussian. NO dispatch of its
     own in families.hpp -- linkinv/valid_eta/valid_mu are redirected to
     the BASE family (_glmfit.base_family_for_quasi), which does have full
     dispatch. GLM.fit() calls _glmfit.fit_quasi() (a single IRLS pass
     with the base family + rescaling vcov/se by Pearson dispersion).

  3. Direct MLE (family_type="mle"): Beta, LogNormal, Weibull, Gumbel,
     LogLogistic, StudentT, SkewNormal, SkewT, LogisticDistribution,
     Simplex, Multinomial, Ordinal, ZeroInflatedPoisson, ZeroInflatedNB,
     HurdlePoisson, HurdleNB, HurdleGamma, HurdleLogNormal, HurdleBeta,
     HurdleGeometric, TruncatedPoisson, TruncatedNB, TruncatedGeometric. linkinv/valid_eta/
     valid_mu implemented here in pure Python (they don't rely on the
     families.hpp switch, which is incomplete for these codes). GLM.fit()
     calls _glmfit.MleSolver().fit() -- L-BFGS-B over the full
     log-likelihood, which builds its own starting values internally (does
     not use GLM.initialize_parameters()). Does not compute se/vcov (no
     Hessian exposed) nor an IRLS-style deviance -- instead it exposes
     log_likelihood, plus gamma/extra_param/extra_param2 for the
     secondary submodel's parameters or the shape/scale parameters when
     the family has them.

  4. MLE inflated at an arbitrary value (family_type="mle_inflated"):
     ValueInflatedPoisson, ValueInflatedNegativeBinomial. Similar to
     group 3 but WITHOUT a FamilyCode -- GLM.fit() calls
     _glmfit.MleSolver().fit_poisson_inflated()/
     fit_negative_binomial_inflated() directly instead of
     .fit(family_code, ...). Not to be confused with
     ZeroInflatedPoisson/ZeroInflatedNB (group 3): those inflate
     specifically at 0 and have their own FamilyCode; these inflate at
     whatever `target_value` is passed in (0, 4, 10...) -- hence the name
     "ValueInflated*", so the two aren't mixed up at a glance.

  5. Multi-parameter survival MLE (family_type="mle_gengamma"):
     GeneralizedGamma. Same reason as group 4 for not having a
     FamilyCode (GLM.fit() calls
     _glmfit.MleSolver().fit_generalized_gamma() directly), but for a
     different reason: it's not that it lacks a FamilyCode for being "an
     arbitrary value" -- it has 3 submodels (mu/sigma/Q), each with its
     own link and its own covariate submatrix -- it doesn't fit into a
     single FamilyCode/link/X the way the rest of group 3 does. `d`
     (status) and `entry` (late entry / left truncation) go in
     GLM.fit(), same as `d` for LogNormal/Weibull/LogLogistic in group 3.
"""

from enum import IntEnum

import numpy as np
from scipy.special import gammaln  # used by loglike_obs() of Gaussian/Poisson/
from scipy.special import ndtr as _ndtr

# Binomial/Gamma/InverseGaussian/
# NegativeBinomial/Exponential below --
# scipy is already a hard dependency of the
# package (pyproject.toml).
from .. import _glmfit

# FamilyCode: generated from the compiled module, not by hand
FamilyCode = IntEnum(
    "FamilyCode",
    {name: val for name, val in vars(_glmfit).items() if name.startswith("FAM_")},
)


class Family:
    """Base class. Every subclass defines LINK_MAP: {"link_string": FamilyCode}."""

    LINK_MAP = {}  # each subclass overrides this
    family_type = "irls"  # "irls" | "quasi" | "mle" | "mle_inflated" | "mle_gengamma"
    # see the module docstring
    SUPPORT = "positive"  # "positive" | "unit" | "real" -- the typical
    # domain of y/mu, used by glm.py::GLM._mustart()
    # for the data-driven starting value (irls/quasi
    # only; MleSolver builds its own in C++)
    EXTRA_FREE = ()  # names (estimation scale) of the INDEPENDENT auxiliary
    # scalar parameters an MLE family estimates in addition
    # to beta and gamma, e.g. ("log_sigma",), ("log_alpha",).
    # `extra_param` is NOT always one of them: for Beta/
    # Simplex/StudentT/Skew* it is derived from gamma[0], and
    # for Ordinal it is the number of categories -- those
    # families leave EXTRA_FREE empty so parameter counts
    # (AIC/BIC/df) don't count the same parameter twice.
    DISPERSION = "fixed"  # "fixed" (phi=1, canonical: Binomial/Poisson/
    # NegativeBinomial/Exponential) | "estimate"
    # (phi free, estimated via Pearson and requires
    # rescaling vcov/se -- Gaussian/Gamma/
    # InverseGaussian/Tweedie/Pareto/InverseGamma).
    # Only used by GLM._fit_irls() (glm.py) -- see
    # there for why: glm.hpp::save_vcov()/save_se()
    # (C++) return (X'WX)^-1 WITHOUT scaling by phi,
    # which is correct only when phi=1 by definition
    # of the family.

    def __init__(self, link):
        self.link = link.lower()
        self.code = self._get_family_code()
        # Always present (even if left at FamilyParams' defaults) so that
        # the code building _glmfit.IrlsEngine(...) never has to ask "does this
        # family use params or not" -- it just passes them along.
        self.params = _glmfit.FamilyParams()

    def _get_family_code(self):
        family_class = self.__class__
        code = family_class.LINK_MAP.get(self.link)
        if code is None:
            valid = ", ".join(f'"{k}"' for k in family_class.LINK_MAP)
            raise ValueError(
                f"Link '{self.link}' is not valid for family "
                f"'{family_class.__name__}'. Valid links: {valid}."
            )
        return code

    def __repr__(self):
        return f"{self.__class__.__name__}(link='{self.link}')"

    def make_engine(
        self, X, y, weights, offset, tol=1e-6, maxit=100, type=1, is_big_matrix=False, firth=False
    ):
        """Builds the _glmfit.IrlsEngine object for this family. CustomFamily
        overrides this method to use the callback-based constructor --
        the code calling make_engine() doesn't need to know which is which."""
        return _glmfit.IrlsEngine(
            X,
            y,
            weights,
            offset,
            int(self.code),
            self.params,
            tol,
            maxit,
            type,
            is_big_matrix,
            firth,
        )

    def linkinv(self, eta):
        return _glmfit.linkinv(int(self.code), self.params, eta)

    def valid_eta(self, eta):
        return _glmfit.valid_eta(int(self.code), self.params, eta)

    def valid_mu(self, mu):
        return _glmfit.valid_mu(int(self.code), self.params, mu)

    def beta_names(self, feature_names, n_beta=None):
        return None

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return None

    def extra_param_label(self):
        return "extra_param"

    def extra_param2_label(self):
        return "extra_param2"

    def beta_heading(self):
        """Heading of the main regression block in GLM.summary()."""
        return f"{self.__class__.__name__} Model ({self.link_description()} link)"

    def gamma_heading(self):
        """Heading of the secondary submodel block in GLM.summary()."""
        return "Secondary Submodel"

    def link_description(self):
        return getattr(self, "link", "n/a")

    def loglike_obs(self, y, mu, weights, dispersion):
        return None

    # Conventions shared by the IRLS / quasi analysis code (glm.py)
    @property
    def estimates_dispersion(self):
        """True when the dispersion phi is ESTIMATED (Pearson chi2 / df) instead of fixed at 1: IRLS
        families with DISPERSION == "estimate" and every quasi-likelihood family. Such models use
        t (not normal) reference distributions for p-values / intervals and F (not chi2) tests,
        as R's summary.glm / anova.glm do."""
        return getattr(self, "DISPERSION", "fixed") == "estimate" or getattr(
            self, "family_type", "irls"
        ) in ("quasi", "quasi_custom")

    def ll_dispersion(self, y, mu, weights, deviance):
        """Dispersion at which `loglike_obs` is evaluated for logLik / AIC / BIC.

        deviance / sum(weights): the (approximate) MLE, as R's family$aic does for Gaussian, Gamma
        and inverse.gaussian (exact for Gaussian and inverse Gaussian). The Pearson estimate used to
        scale standard errors is NOT the maximizer of the likelihood, so evaluating logLik there
        under-states it and makes AIC inconsistent across families."""
        sw = float(np.sum(weights))
        return float(deviance) / sw if sw > 0 else 1.0

    def dev_resids(self, y, mu, weights):
        """Per-observation weighted deviance contributions w_i d(y_i, mu_i), from the C++ unit
        deviance (families.hpp). Needs no log-likelihood, so it exists for every IRLS and quasi
        family (Tweedie, Pareto, InverseGamma and FAM_QUASI_* included). None for families
        without a deviance (MLE families, CustomFamily)."""
        if getattr(self, "family_type", "irls") not in ("irls", "quasi", "nb_two_step"):
            return None
        code = getattr(self, "code", None)
        if code is None:
            return None
        return np.asarray(
            _glmfit.dev_resids(
                int(code),
                self.params,
                np.asarray(y, dtype=float),
                np.asarray(mu, dtype=float),
                np.asarray(weights, dtype=float),
            )
        )

    def copy(self):
        """Independent copy for one GLM instance. The FamilyParams are copied too: fitting a model must
        never change another model's family (the NB two-step fit stores its estimated theta in
        `params`, so a shared family let the null-model refit inside `llnull` overwrite the theta the
        full model's Pearson residuals / hat matrix rely on)."""
        import copy as _copy

        new = _copy.copy(self)
        old = getattr(self, "params", None)
        if old is not None:
            fresh = _glmfit.FamilyParams()
            for attr in ("theta", "var_power", "link_power"):
                if hasattr(old, attr):
                    setattr(fresh, attr, getattr(old, attr))
            new.params = fresh
        return new


def _tw_series_ld(T, alpha, kmax=3000):
    """log S and (log-magnitude of the largest term, digits lost to cancellation) for
        S = sum_{k>=1} (-1)^(k+1) Gamma(k a + 1) / k! * sin(k pi a) * T^k        (0 < a < 1),
    in extended precision with the largest term factored out (no overflow)."""
    k = np.arange(1, kmax + 1, dtype=np.longdouble)
    logabs = (
        gammaln((k * alpha + 1).astype(float)).astype(np.longdouble)
        - gammaln((k + 1).astype(float)).astype(np.longdouble)
        + k * np.log(np.longdouble(T))
    )
    sinv = np.sin(k * np.longdouble(np.pi) * np.longdouble(alpha))
    sign = np.where(k % 2 == 1, 1.0, -1.0) * np.sign(sinv)
    lm = logabs + np.log(np.maximum(np.abs(sinv), np.longdouble(1e-300)))
    big = np.max(lm)
    total = np.sum(sign * np.exp(lm - big) * (np.abs(sinv) > 0))
    if not total > 0:
        return np.nan, float(big), np.inf
    return float(big + np.log(total)), float(big), float(-np.log10(float(total)))


def _tw_saddlepoint_logpdf(y, mu, phi, p):
    """Saddlepoint approximation (2 pi phi y^p)^(-1/2) exp(-d/(2 phi)); used only where the exact series
    would need > ~1500 digits, i.e. for observations whose density is below ~1e-600."""
    d = 2.0 * (
        y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)
    )
    return -0.5 * np.log(2.0 * np.pi * phi * y**p) - d / (2.0 * phi)


def _tw_logpdf_talbot(y, mu, phi, p, dps):
    """log-density by numerical inversion (Talbot, mpmath) of the Laplace transform
        E[exp(-s Y)] = exp{ [ (mu^(1-p) + (p-1) phi s)^a - mu^((1-p) a) ] / ((2-p) phi) },  a = (2-p)/(1-p),
    valid for any p > 1 (exact to ~1e-15 in checks against the inverse Gaussian at p = 3, including a density of 1e-211)."""
    import mpmath as mp

    mp.mp.dps = dps
    y_, mu_, phi_, p_ = mp.mpf(y), mp.mpf(mu), mp.mpf(phi), mp.mpf(p)
    a = (2 - p_) / (1 - p_)
    lap = lambda s: mp.exp(
        ((mu_ ** (1 - p_) + (p_ - 1) * phi_ * s) ** a - mu_ ** ((1 - p_) * a)) / ((2 - p_) * phi_)
    )
    val = mp.invertlaplace(lap, y_, method="talbot")
    return float(mp.log(val)) if val > 0 else np.nan


def _tw_logpdf_gt2_one(y, mu, phi, p):
    """Fast path: the positive-stable series in long double while <= ~4 digits are lost to cancellation
    (the normal range of the data). Otherwise (observations far in the left tail, or p very close to 2 where
    the series needs thousands of digits) Talbot inversion at a precision guided by the saddlepoint magnitude
    of the density; the saddlepoint approximation itself only if that would need > 1200 digits or mpmath is missing."""
    alpha = (p - 2.0) / (p - 1.0)  # in (0, 1)
    lam = (p - 1.0) ** alpha * phi ** (alpha - 1.0) / (p - 2.0)
    T = lam * y ** (-alpha)
    log_s, _, lost = _tw_series_ld(T, alpha)
    if np.isfinite(log_s) and lost < 4.0:
        theta = mu ** (1.0 - p) / (1.0 - p)
        kappa = mu ** (2.0 - p) / (2.0 - p)
        return (y * theta - kappa) / phi - np.log(np.pi * y) + log_s
    approx = _tw_saddlepoint_logpdf(y, mu, phi, p)
    dps = int(max(40.0, -approx / np.log(10.0) + 40.0)) if np.isfinite(approx) else 400
    if dps > 1200:
        return approx
    try:
        val = _tw_logpdf_talbot(y, mu, phi, p, dps)
    except ImportError:
        return approx
    except Exception:
        return approx
    return val if np.isfinite(val) else approx


def _tweedie_logpdf(y, mu, phi, p):
    """log-density (log-probability at 0) of the Tweedie EDM, Var[y] = phi * mu^p.

    1 < p < 2 : compound Poisson-gamma, P(Y=0) = exp(-mu^(2-p) / (phi (2-p))) and the Dunn & Smyth
                (2005) series for y > 0 (verified: mass 1, mean mu, variance phi mu^p).
    p = 2     : Gamma(shape = 1/phi, scale = mu phi).
    p = 1     : y/phi ~ Poisson(mu/phi) (only meaningful on multiples of phi).
    p > 2     : y > 0 only. Positive-stable series (Dunn & Smyth 2005) in long double for the normal range of the
                data; Talbot inversion of the Laplace transform (mpmath) for far-tail observations and p close to 2;
                saddlepoint approximation only if mpmath is missing or > 1200 digits would be needed.
                Verified: p = 3 equals the inverse Gaussian to ~1e-15; mass 1 and mean mu for p = 2.2 ... 4.
    p < 1     : NaN (no distribution), so callers report "no likelihood".
    `phi` may be an array (precision weights: phi_i = phi / w_i).
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    y, mu, phi = np.broadcast_arrays(y, mu, np.asarray(phi, dtype=float))
    if abs(p - 1.0) < 1e-8:
        k, lam = y / phi, mu / phi
        return np.where(
            k >= 0, k * np.log(np.maximum(lam, 1e-300)) - lam - gammaln(k + 1.0), -np.inf
        )
    if abs(p - 2.0) < 1e-8:
        nu = 1.0 / phi
        return nu * np.log(nu * y / mu) - nu * y / mu - gammaln(nu) - np.log(y)
    if p > 2.0:
        out = np.full(y.shape, -np.inf)
        for idx in np.ndindex(y.shape):
            if y[idx] > 0:
                out[idx] = _tw_logpdf_gt2_one(
                    float(y[idx]), float(mu[idx]), float(phi[idx]), float(p)
                )
        return out
    if not (1.0 < p < 2.0):
        return np.full(y.shape, np.nan)

    alpha = (2.0 - p) / (1.0 - p)  # < 0
    yy, mm, ph = y.ravel(), mu.ravel(), phi.ravel()
    lam = mm ** (2.0 - p) / (ph * (2.0 - p))  # Poisson rate of the compound sum
    ll = np.empty(yy.shape)
    pos = yy > 0
    ll[~pos] = -lam[~pos]  # P(Y = 0)
    expo = (yy * mm ** (1.0 - p) / (1.0 - p) - mm ** (2.0 - p) / (2.0 - p)) / ph
    for i in np.flatnonzero(pos):
        yi, phi_i = yy[i], ph[i]
        jmax = yi ** (2.0 - p) / (phi_i * (2.0 - p))
        span = 10.0 * np.sqrt(max(jmax, 1.0)) + 25.0
        j = np.arange(max(1.0, np.floor(jmax - span)), np.ceil(jmax + span) + 1.0)
        logw = (
            -j * alpha * np.log(yi)
            + alpha * j * np.log(p - 1.0)
            - j * (1.0 - alpha) * np.log(phi_i)
            - j * np.log(2.0 - p)
            - gammaln(j + 1.0)
            - gammaln(-j * alpha)
        )
        m = logw.max()
        ll[i] = expo[i] - np.log(yi) + m + np.log(np.exp(logw - m).sum())
    return ll.reshape(y.shape)


def _reused_x_names(feature_names, n, prefix):
    if feature_names is None or n is None or n != len(feature_names):
        return None
    return [f"{prefix}:{name}" for name in feature_names]


def _z_submodel_names(n_gamma, prefix):
    if not n_gamma:
        return None
    if n_gamma == 1:
        return [f"{prefix}_intercept"]
    return [f"{prefix}_z{j}" for j in range(n_gamma)]


def _set_link_power(fam, link_power):
    """Power/Box-Cox link: mu = eta**(1/link_power), or log when
    link_power == 0. Used by Gamma/InverseGaussian/NegativeBinomial/
    Tweedie when link='power'. Requires link_power to be given explicitly
    -- there's no sensible default (1=identity, 0.5=sqrt, -1=inverse,
    0=log already have their own link names)."""
    if fam.link == "power":
        if link_power is None:
            raise ValueError(
                f"{fam.__class__.__name__}(link='power') requires link_power "
                f"(mu = eta**(1/link_power), or log when link_power == 0)."
            )
        fam.params.link_power = link_power
    elif link_power is not None:
        raise ValueError(
            f"{fam.__class__.__name__}: link_power only applies with link='power' "
            f"(got link='{fam.link}')."
        )


class Gaussian(Family):
    SUPPORT = "real"
    DISPERSION = "estimate"
    LINK_MAP = {
        "identity": FamilyCode.FAM_GAUSSIAN_IDENTITY,
        "log": FamilyCode.FAM_GAUSSIAN_LOG,
        "inverse": FamilyCode.FAM_GAUSSIAN_INVERSE,
        "sqrt": FamilyCode.FAM_GAUSSIAN_SQRT,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def loglike_obs(self, y, mu, weights, dispersion):
        w = np.asarray(weights, dtype=float)
        ssr = np.sum(w * (y - mu) ** 2)
        n_eff = np.sum(w)
        phi_mle = ssr / n_eff
        return -0.5 * w * np.log(2 * np.pi * phi_mle) - 0.5 * w * (y - mu) ** 2 / phi_mle


class Binomial(Family):
    SUPPORT = "unit"
    LINK_MAP = {
        "logit": FamilyCode.FAM_BINOMIAL_LOGIT,
        "probit": FamilyCode.FAM_BINOMIAL_PROBIT,
        "cloglog": FamilyCode.FAM_BINOMIAL_CLOGLOG,
        "log": FamilyCode.FAM_BINOMIAL_LOG,
        "loglog": FamilyCode.FAM_BINOMIAL_LOGLOG,
        "cauchit": FamilyCode.FAM_BINOMIAL_CAUCHIT,
        # "logit_log" is an alias for the real logit link: the two codes behave
        # identically across all 6 dispatch points in families.hpp (linkinv/mu_eta/
        # variance/dev_resids_sum/validmu/stable_weights), so the alias maps straight
        # to FAM_BINOMIAL_LOGIT. The string is accepted for backward compatibility
        # with existing callers.
        "logit_log": FamilyCode.FAM_BINOMIAL_LOGIT,
    }

    def __init__(self, link="logit"):
        super().__init__(link)

    def loglike_obs(self, y, mu, weights, phi=1.0):
        """Binomial log-likelihood with `weights` as the number of trials and `y` the proportion.

        Includes the binomial coefficient log C(m, m*y) whenever the weights are integers >= 1 and m*y is an
        integer (grouped / aggregated data), exactly what R's logLik does (R rounds m and m*y). It vanishes for
        Bernoulli data (m = 1), so nothing changes there. For non-integer weights (case / importance weights)
        no coefficient exists and the plain weighted log-likelihood is used."""
        y = np.asarray(y, dtype=float)
        w = np.asarray(weights, dtype=float)
        mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1.0 - 1e-15)

        term_y = np.where(y > 0, y * np.log(mu), 0.0)
        term_1y = np.where(y < 1, (1.0 - y) * np.log(1.0 - mu), 0.0)

        m = np.rint(w)
        k = np.rint(w * y)
        integer = (
            (np.abs(w - m) < 1e-8) & (m >= 1) & (np.abs(w * y - k) < 1e-8) & (k >= 0) & (k <= m)
        )
        coef = np.where(
            integer,
            gammaln(m + 1.0) - gammaln(k + 1.0) - gammaln(np.maximum(m - k, 0.0) + 1.0),
            0.0,
        )
        return w * (term_y + term_1y) + coef


class Poisson(Family):
    LINK_MAP = {
        "log": FamilyCode.FAM_POISSON_LOG,
        "identity": FamilyCode.FAM_POISSON_IDENTITY,
        "sqrt": FamilyCode.FAM_POISSON_SQRT,
        "inverse": FamilyCode.FAM_POISSON_INVERSE,
    }

    def __init__(self, link="log"):
        super().__init__(link)

    def loglike_obs(self, y, mu, weights, dispersion):
        return weights * (y * np.log(mu) - mu - gammaln(y + 1))


class Exponential(Family):
    LINK_MAP = {
        "log": FamilyCode.FAM_EXPONENTIAL_LOG,
        "identity": FamilyCode.FAM_EXPONENTIAL_IDENTITY,
        "inverse": FamilyCode.FAM_EXPONENTIAL_INVERSE,
        "sqrt": FamilyCode.FAM_EXPONENTIAL_SQRT,
    }

    def __init__(self, link="log"):
        super().__init__(link)

    def loglike_obs(self, y, mu, weights, dispersion):
        return weights * (-np.log(mu) - y / mu)


class Pareto(Family):
    # Fitted as a V(mu) = mu^2 quasi-likelihood model (Gamma unit deviance): deviance, deviance /
    # Pearson residuals and Wald inference are defined; there is NO log-likelihood (AIC/BIC = None).
    DISPERSION = "estimate"
    LINK_MAP = {
        "log": FamilyCode.FAM_PARETO_LOG,
        "inverse": FamilyCode.FAM_PARETO_INVERSE,
    }

    def __init__(self, link="log"):
        super().__init__(link)


class InverseGamma(Family):
    # Same status as Pareto: V(mu) = mu^2 quasi-likelihood, no log-likelihood (AIC/BIC = None).
    DISPERSION = "estimate"
    LINK_MAP = {
        "log": FamilyCode.FAM_INVGAMMA_LOG,
        "inverse": FamilyCode.FAM_INVGAMMA_INVERSE,
        "identity": FamilyCode.FAM_INVGAMMA_IDENTITY,
    }

    def __init__(self, link="inverse"):
        super().__init__(link)


class Gamma(Family):
    DISPERSION = "estimate"
    LINK_MAP = {
        "log": FamilyCode.FAM_GAMMA_LOG,
        "inverse": FamilyCode.FAM_GAMMA_INVERSE,
        "identity": FamilyCode.FAM_GAMMA_IDENTITY,
        "sqrt": FamilyCode.FAM_GAMMA_SQRT,
        "power": FamilyCode.FAM_GAMMA_POWER,
    }

    def __init__(self, link="inverse", link_power=None):
        super().__init__(link)
        _set_link_power(self, link_power)

    def __repr__(self):
        if self.link == "power":
            return f"Gamma(link='power', link_power={self.params.link_power})"
        return f"Gamma(link='{self.link}')"

    def loglike_obs(self, y, mu, weights, dispersion):
        ws = weights / dispersion
        return ws * np.log(ws * y / mu) - ws * y / mu - gammaln(ws) - np.log(y)


class InverseGaussian(Family):
    DISPERSION = "estimate"
    LINK_MAP = {
        "invmu2": FamilyCode.FAM_INVGAUSS_INVMU2,
        "log": FamilyCode.FAM_INVGAUSS_LOG,
        "identity": FamilyCode.FAM_INVGAUSS_IDENTITY,
        "inverse": FamilyCode.FAM_INVGAUSS_INVERSE,
        "power": FamilyCode.FAM_INVGAUSS_POWER,
    }

    def __init__(self, link="invmu2", link_power=None):
        super().__init__(link)
        _set_link_power(self, link_power)

    def __repr__(self):
        if self.link == "power":
            return f"InverseGaussian(link='power', link_power={self.params.link_power})"
        return f"InverseGaussian(link='{self.link}')"

    def loglike_obs(self, y, mu, weights, dispersion):
        ws = weights / dispersion
        return (
            -0.5 * ws * (y - mu) ** 2 / (y * mu**2)
            - 0.5 * np.log(dispersion / weights * y**3)
            - 0.5 * np.log(2 * np.pi)
        )


class NegativeBinomial(Family):
    LINK_MAP = {
        "log": FamilyCode.FAM_NB_LOG,
        "sqrt": FamilyCode.FAM_NB_SQRT,
        "identity": FamilyCode.FAM_NB_IDENTITY,
        "inverse": FamilyCode.FAM_NB_INVERSE,
        "power": FamilyCode.FAM_NB_POWER,
    }

    def __init__(self, theta=None, link="log", link_power=None):
        super().__init__(link)
        self.params.theta = theta if theta is not None else 1.0
        self.theta_is_fixed = theta is not None
        _set_link_power(self, link_power)
        self.family_type = "nb_two_step"

    @property
    def estimates_theta(self):
        """False when the user supplied theta (it is data, not a parameter): GLM.fit then keeps it fixed
        (`fit_nb_two_step(estimate_theta=False)`) and does not count it in AIC/BIC."""
        return not self.theta_is_fixed

    def __repr__(self):
        t_str = f"{self.params.theta}" if self.theta_is_fixed else "auto"
        if self.link == "power":
            return f"NegativeBinomial(theta={t_str}, link='power', link_power={self.params.link_power})"
        return f"NegativeBinomial(theta={t_str}, link='{self.link}')"

    def extra_param_label(self):
        return "theta (NB dispersion)"

    def loglike_obs(self, y, mu, weights, dispersion):
        alpha = 1.0 / self.params.theta
        return (
            y * np.log(alpha * mu / (1 + alpha * mu))
            - np.log(1 + alpha * mu) / alpha
            + gammaln(y + 1 / alpha)
            - gammaln(y + 1)
            - gammaln(1 / alpha)
        ) * weights


class Tweedie(Family):
    DISPERSION = "estimate"
    LINK_MAP = {
        "log": FamilyCode.FAM_TWEEDIE_LOG,
        "identity": FamilyCode.FAM_TWEEDIE_IDENTITY,
        "inverse": FamilyCode.FAM_TWEEDIE_INVERSE,
        "sqrt": FamilyCode.FAM_TWEEDIE_SQRT,
        "power": FamilyCode.FAM_TWEEDIE_POWER,
    }

    def __init__(self, var_power, link="log", link_power=None):
        super().__init__(link)
        self.params.var_power = var_power
        _set_link_power(self, link_power)

    def loglike_obs(self, y, mu, weights, dispersion):
        """Exact Tweedie log-density (Dunn-Smyth series) for p > 1 (p = 2 is the Gamma, p = 3 the inverse
        Gaussian) and p = 1. Weights are precision weights (y_i ~ Tweedie(mu_i, phi / w_i)). For p > 2 the
        support is y > 0: a zero makes the log-likelihood -inf and GLM then reports None."""
        w = np.asarray(weights, dtype=float)
        return _tweedie_logpdf(y, mu, float(dispersion) / w, float(self.params.var_power))

    def ll_dispersion(self, y, mu, weights, deviance):
        """phi at the MAXIMUM of the exact likelihood (1-D bounded search around deviance/sum(w), the
        saddlepoint estimate), so AIC/BIC use the maximized likelihood."""
        phi0 = super().ll_dispersion(y, mu, weights, deviance)
        p = float(self.params.var_power)
        if not (1.0 < p < 2.0 or p > 2.0):
            return phi0
        from scipy.optimize import minimize_scalar

        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        w = np.asarray(weights, dtype=float)

        def nll(log_phi):
            v = -np.sum(_tweedie_logpdf(y, mu, np.exp(log_phi) / w, p))
            return v if np.isfinite(v) else 1e300

        lo, hi = np.log(phi0) - 3.0, np.log(phi0) + 3.0
        res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8})
        return float(np.exp(res.x)) if res.fun < 1e299 else phi0

    def __repr__(self):
        if self.link == "power":
            return f"Tweedie(var_power={self.params.var_power}, link='power', link_power={self.params.link_power})"
        return f"Tweedie(var_power={self.params.var_power}, link='{self.link}')"


# Group 2: quasi-likelihood (family_type="quasi")
class QuasiFamily(Family):
    family_type = "quasi"

    @property
    def _base_code(self):
        return _glmfit.base_family_for_quasi(int(self.code))

    def linkinv(self, eta):
        return _glmfit.linkinv(self._base_code, self.params, eta)

    def valid_eta(self, eta):
        return _glmfit.valid_eta(self._base_code, self.params, eta)

    def valid_mu(self, mu):
        return _glmfit.valid_mu(self._base_code, self.params, mu)

    def make_engine(self, *args, **kwargs):
        raise NotImplementedError(
            "QuasiFamily does not build a _glmfit.IrlsEngine -- GLM.fit() (glm.py) "
            "must call _glmfit.fit_quasi() directly when "
            "family_type == 'quasi', not make_engine()."
        )


class QuasiPoisson(QuasiFamily):
    LINK_MAP = {
        "log": FamilyCode.FAM_QUASI_POISSON_LOG,
        "identity": FamilyCode.FAM_QUASI_POISSON_IDENTITY,
    }

    def __init__(self, link="log"):
        super().__init__(link)


class QuasiBinomial(QuasiFamily):
    SUPPORT = "unit"
    LINK_MAP = {
        "logit": FamilyCode.FAM_QUASI_BINOMIAL_LOGIT,
        "probit": FamilyCode.FAM_QUASI_BINOMIAL_PROBIT,
    }

    def __init__(self, link="logit"):
        super().__init__(link)


class QuasiGamma(QuasiFamily):
    LINK_MAP = {
        "log": FamilyCode.FAM_QUASI_GAMMA_LOG,
        "identity": FamilyCode.FAM_QUASI_GAMMA_IDENTITY,
        "inverse": FamilyCode.FAM_QUASI_GAMMA_INVERSE,
    }

    def __init__(self, link="log"):
        super().__init__(link)


class QuasiInverseGaussian(QuasiFamily):
    LINK_MAP = {
        "log": FamilyCode.FAM_QUASI_INVGAUSSIAN_LOG,
        "identity": FamilyCode.FAM_QUASI_INVGAUSSIAN_IDENTITY,
        "inverse": FamilyCode.FAM_QUASI_INVGAUSSIAN_INVERSE,
    }

    def __init__(self, link="log"):
        super().__init__(link)


class QuasiGaussian(QuasiFamily):
    SUPPORT = "real"
    LINK_MAP = {
        "identity": FamilyCode.FAM_QUASI_GAUSSIAN_IDENTITY,
        "log": FamilyCode.FAM_QUASI_GAUSSIAN_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)


class QuasiCustom(Family):
    """Quasi-likelihood with an arbitrary variance function V(mu), fit via
    QuasiSolver.fit_custom (L-BFGS-B/BFGS/NelderMead -- see `minimizer` in
    GLM.__init__). Unlike the other "quasi" families (QuasiPoisson,
    QuasiGamma, etc.), which go through fit_quasi() in irls_two_step.hpp
    with a fixed FamilyCode, this family has no FAM_QUASI_* code of its
    own -- variance_fn is evaluated directly as a Python callback on every
    optimizer iteration, so it supports any V(mu) the user wants, at the
    cost of being slower than the predefined quasi families (the callback
    crosses into Python for every observation on every iteration).

    Only link="log" or link="identity" are supported -- those are the only
    two QuasiSolver::fit_custom (quasi_solver.hpp) handles; validated here
    to give a clear error instead of letting the C++ exception surface
    later.

    Standard errors ARE available (unlike earlier versions of fit_custom):
    they're computed in C++ with the same asymptotic formula
    phi * (X'WX)^-1 the IRLS engine uses for the other quasi families,
    evaluated at beta_hat with the analytic link derivative.
    """

    family_type = "quasi_custom"

    _LINK_TO_BASE_CODE = {
        "log": FamilyCode.FAM_GAUSSIAN_LOG,
        "identity": FamilyCode.FAM_GAUSSIAN_IDENTITY,
    }

    @property
    def SUPPORT(self):
        # mu = exp(eta) with link="log" is always positive -- the
        # np.maximum(y, 0.1) starting value in _mustart() (glm.py) is
        # correct there. With link="identity", mu can be any real number,
        # same as Gaussian.
        return "positive" if self.link == "log" else "real"

    def __init__(self, variance_fn, link="log"):
        link = link.lower()
        if link not in self._LINK_TO_BASE_CODE:
            valid = ", ".join(f'"{k}"' for k in self._LINK_TO_BASE_CODE)
            raise ValueError(
                f"QuasiCustom(link='{link}'): only {valid} are supported "
                f"by the custom-variance optimizer "
                f"(QuasiSolver::fit_custom in quasi_solver.hpp)."
            )
        if not callable(variance_fn):
            raise TypeError("QuasiCustom: variance_fn must be callable (mu -> V(mu)).")
        self.link = link
        self.variance_fn = variance_fn
        self.code = None  # no FamilyCode -- fit_custom doesn't need one
        self.params = _glmfit.FamilyParams()
        self._base_code = self._LINK_TO_BASE_CODE[link]

    @property
    def quasi_link_type(self):
        return (
            _glmfit.QuasiLinkType.Log
            if self.link == "log"
            else _glmfit.QuasiLinkType.Identity
        )

    def linkinv(self, eta):
        return _glmfit.linkinv(int(self._base_code), self.params, eta)

    def valid_eta(self, eta):
        return _glmfit.valid_eta(int(self._base_code), self.params, eta)

    def valid_mu(self, mu):
        return _glmfit.valid_mu(int(self._base_code), self.params, mu)

    def make_engine(self, *args, **kwargs):
        raise NotImplementedError(
            "QuasiCustom does not build a _glmfit.IrlsEngine -- GLM.fit() "
            "(glm.py) must call QuasiSolver.fit_custom() directly when "
            "family_type == 'quasi_custom', not make_engine()."
        )

    def link_description(self):
        return f"{self.link} (custom variance)"

    def dev_resids(self, y, mu, weights):
        """w_i * d(y_i, mu_i) with the quasi-deviance d = 2 int_mu^y (y-u)/V(u) du, integrated with
        adaptive quadrature -- the same definition (and V floor of 1e-6) the C++ objective uses."""
        from scipy.integrate import quad

        V = self.variance_fn
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        w = np.asarray(weights, dtype=float)
        out = np.zeros(y.shape[0])
        for i in range(y.shape[0]):
            if abs(y[i] - mu[i]) < 1e-12:
                continue
            f = lambda u, yi=y[i]: (yi - u) / max(1e-6, float(V(u)))
            out[i] = w[i] * 2.0 * quad(f, mu[i], y[i], epsabs=1e-13, epsrel=1e-10, limit=200)[0]
        return out

    def __repr__(self):
        return f"QuasiCustom(link='{self.link}', variance_fn={self.variance_fn!r})"


# Group 3: direct MLE (family_type="mle")
def _erf_approx(x):
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def _mle_probit_linkinv(eta):
    # exact normal cdf (the Abramowitz-Stegun erf approximation is accurate only to ~1.5e-7)
    return _ndtr(np.asarray(eta, dtype=float))


_MLE_LINKINV = {
    "identity": lambda eta: np.asarray(eta, dtype=float),
    "log": lambda eta: np.exp(np.clip(eta, -700.0, 700.0)),
    "inverse": lambda eta: 1.0 / np.where(np.abs(eta) < 1e-12, 1e-12, eta),
    "sqrt": lambda eta: np.asarray(eta, dtype=float) ** 2,
    "logit": lambda eta: 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))),
    "probit": _mle_probit_linkinv,
    "cloglog": lambda eta: 1.0 - np.exp(-np.exp(np.clip(eta, -700.0, 700.0))),
    "loglog": lambda eta: np.exp(-np.exp(-np.clip(eta, -700.0, 700.0))),
    "cauchit": lambda eta: np.arctan(eta) / np.pi + 0.5,
}

_MLE_UNIT_LINKS = ("logit", "probit", "cloglog", "loglog", "cauchit")

# Links valid for the parameter modelled by the second predictor Z (must match
# MleSolver::resolve_secondary_link in mle_solver.hpp).
_PROBABILITY_LINKS = ("logit", "probit", "cloglog", "loglog", "cauchit")  # pi in (0, 1)
_POSITIVE_LINKS = ("log", "identity", "sqrt", "inverse")  # phi / sigma^2 > 0


def _check_secondary_link(value, valid, family_name, arg_name, what):
    """Normalizes (lowercase) and validates the link of a secondary parameter,
    failing at construction with the list of valid links instead of letting the
    error surface later from the C++ solver."""
    link = str(value).lower()
    if link not in valid:
        opts = ", ".join(f'"{k}"' for k in valid)
        raise ValueError(
            f"{family_name}: {arg_name}='{link}' is not valid for {what}. Valid links: {opts}."
        )
    return link


class MleFamily(Family):
    family_type = "mle"

    # Secondary link: link of the parameter modelled by the second predictor Z.
    # Subclasses that have one set these three and call _init_secondary_link()
    # from __init__; the resulting string is stored in `self.gamma_link`, which
    # GLM._fit_mle() (glm.py) forwards to MleSolver.fit(). (ZeroInflated*/
    # ValueInflated* keep their historical `infl_link` attribute instead.)
    SECONDARY_KIND = None  # "probability" | "positive" | None
    SECONDARY_ARG = None  # constructor argument name ("zero_link", "phi_link", ...)
    SECONDARY_DEFAULT = None  # "logit" for probability, "log" for positive
    MAIN_LABEL = None  # name of the main block in summary(), e.g. "Mean Model"

    def beta_heading(self):
        if self.MAIN_LABEL:
            return f"{self.MAIN_LABEL} ({self.link_description()} link)"
        return super().beta_heading()

    def _init_secondary_link(self, value):
        if value is None:
            value = self.SECONDARY_DEFAULT
        if self.SECONDARY_KIND == "probability":
            valid, what = _PROBABILITY_LINKS, "a probability parameter (zero hurdle / inflation)"
        else:
            valid, what = _POSITIVE_LINKS, "a strictly positive parameter (precision / dispersion)"
        self.gamma_link = _check_secondary_link(
            value, valid, self.__class__.__name__, self.SECONDARY_ARG, what
        )

    @property
    def secondary_link(self):
        """Link (str) of the parameter modelled by the second predictor Z, or
        None if the family has none. Covers both `gamma_link` (Beta, Simplex,
        Hurdle*) and `infl_link` (ZeroInflated*, ValueInflated*)."""
        return getattr(self, "gamma_link", None) or getattr(self, "infl_link", None)

    def __repr__(self):
        # Only shown when non-default, so existing reprs are unchanged.
        gl = getattr(self, "gamma_link", None)
        if self.SECONDARY_ARG and gl is not None and gl != self.SECONDARY_DEFAULT:
            return f"{self.__class__.__name__}(link='{self.link}', {self.SECONDARY_ARG}='{gl}')"
        return super().__repr__()

    def linkinv(self, eta):
        fn = _MLE_LINKINV.get(self.link)
        if fn is None:
            raise ValueError(f"MleFamily: no link implemented for '{self.link}'.")
        return fn(np.asarray(eta, dtype=float))

    def valid_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        if self.link == "inverse":
            return bool(np.all(np.isfinite(eta)) and np.all(eta != 0.0))
        return bool(np.all(np.isfinite(eta)))

    def valid_mu(self, mu):
        mu = np.asarray(mu, dtype=float)
        if not np.all(np.isfinite(mu)):
            return False
        if self.link in _MLE_UNIT_LINKS:
            return bool(np.all(mu > 0.0) and np.all(mu < 1.0))
        if self.link != "identity":
            return bool(np.all(mu > 0.0))
        return True

    def make_engine(self, *args, **kwargs):
        raise NotImplementedError(
            "MleFamily does not build a _glmfit.IrlsEngine -- GLM.fit() (glm.py) "
            "must call _glmfit.MleSolver().fit() directly when "
            "family_type == 'mle', not make_engine()."
        )


class Beta(MleFamily):
    MAIN_LABEL = "Mean Model"
    SUPPORT = "unit"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "positive", "phi_link", "log"
    LINK_MAP = {
        "logit": FamilyCode.FAM_BETA_LOGIT,
        "probit": FamilyCode.FAM_BETA_PROBIT,
        "cloglog": FamilyCode.FAM_BETA_CLOGLOG,
        "loglog": FamilyCode.FAM_BETA_LOGLOG,
        "cauchit": FamilyCode.FAM_BETA_CAUCHIT,
    }

    def __init__(self, mu_link="logit", phi_link="log"):
        super().__init__(mu_link)
        self._init_secondary_link(phi_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _z_submodel_names(n_gamma, "precision")

    def gamma_heading(self):
        return f"Precision Model (phi, {self.gamma_link} link)"

    def extra_param_label(self):
        return "phi (precision, natural scale)"


class LogNormal(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    LINK_MAP = {
        "identity": FamilyCode.FAM_LOGNORMAL_IDENTITY,
        "log": FamilyCode.FAM_LOGNORMAL_LOG,
        "inverse": FamilyCode.FAM_LOGNORMAL_INVERSE,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def extra_param_label(self):
        return "sigma (natural scale)"


class Weibull(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    LINK_MAP = {
        "log": FamilyCode.FAM_WEIBULL_LOG,
        "identity": FamilyCode.FAM_WEIBULL_IDENTITY,
    }

    def __init__(self, link="log"):
        super().__init__(link)

    def beta_heading(self):
        return f"Weibull Model ({self.link_description()} link)"

    def extra_param_label(self):
        return "sigma (Weibull scale, natural scale)"


class Gumbel(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    SUPPORT = "real"
    LINK_MAP = {
        "identity": FamilyCode.FAM_GUMBEL_IDENTITY,
        "log": FamilyCode.FAM_GUMBEL_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def extra_param_label(self):
        return "sigma (natural scale)"


class LogLogistic(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    LINK_MAP = {
        "identity": FamilyCode.FAM_LOGLOGISTIC_IDENTITY,
        "log": FamilyCode.FAM_LOGLOGISTIC_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def extra_param_label(self):
        return "sigma (natural scale)"


class StudentT(MleFamily):
    SUPPORT = "real"
    LINK_MAP = {
        "identity": FamilyCode.FAM_STUDENT_T_IDENTITY,
        "log": FamilyCode.FAM_STUDENT_T_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        if n_gamma != 2:
            return None
        return ["log(sigma)", "log(nu) (degrees of freedom)"]

    def gamma_heading(self):
        return "Auxiliary Parameters (sigma, degrees of freedom)"

    def extra_param_label(self):
        return "sigma (natural scale)"


class SkewNormal(MleFamily):
    SUPPORT = "real"
    LINK_MAP = {
        "identity": FamilyCode.FAM_SKEW_NORMAL_IDENTITY,
        "log": FamilyCode.FAM_SKEW_NORMAL_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        if n_gamma != 2:
            return None
        return ["log(sigma)", "alpha (skewness)"]

    def gamma_heading(self):
        return "Auxiliary Parameters (sigma, skewness)"

    def extra_param_label(self):
        return "sigma (natural scale)"


class SkewT(MleFamily):
    SUPPORT = "real"
    LINK_MAP = {
        "identity": FamilyCode.FAM_SKEW_T_IDENTITY,
        "log": FamilyCode.FAM_SKEW_T_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        if n_gamma != 3:
            return None
        return ["log(sigma)", "alpha (skewness)", "log(nu) (degrees of freedom)"]

    def gamma_heading(self):
        return "Auxiliary Parameters (sigma, skewness, degrees of freedom)"

    def extra_param_label(self):
        return "sigma (natural scale)"


class LogisticDistribution(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    SUPPORT = "real"
    LINK_MAP = {
        "identity": FamilyCode.FAM_LOGISTIC_IDENTITY,
        "log": FamilyCode.FAM_LOGISTIC_LOG,
    }

    def __init__(self, link="identity"):
        super().__init__(link)

    def extra_param_label(self):
        return "sigma (natural scale)"


class Simplex(MleFamily):
    MAIN_LABEL = "Mean Model"
    SUPPORT = "unit"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "positive", "sigma_link", "log"
    LINK_MAP = {
        "logit": FamilyCode.FAM_SIMPLEX_LOGIT,
        "probit": FamilyCode.FAM_SIMPLEX_PROBIT,
        "loglog": FamilyCode.FAM_SIMPLEX_LOGLOG,
        "cloglog": FamilyCode.FAM_SIMPLEX_CLOGLOG,
        "cauchit": FamilyCode.FAM_SIMPLEX_CAUCHIT,
    }

    def __init__(self, mu_link="logit", sigma_link="log"):
        super().__init__(mu_link)
        self._init_secondary_link(sigma_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _z_submodel_names(n_gamma, "sigma")

    def gamma_heading(self):
        return f"Dispersion Model (sigma^2, {self.gamma_link} link)"

    def extra_param_label(self):
        return "sigma (dispersion, natural scale)"


class Multinomial(MleFamily):
    LINK_MAP = {
        "logit": FamilyCode.FAM_MULTINOMIAL_LOGIT,
        "probit": FamilyCode.FAM_MULTINOMIAL_PROBIT,
        "cloglog": FamilyCode.FAM_MULTINOMIAL_CLOGLOG,
        "loglog": FamilyCode.FAM_MULTINOMIAL_LOGLOG,
        "cauchit": FamilyCode.FAM_MULTINOMIAL_CAUCHIT,
    }

    def __init__(self, link="logit"):
        super().__init__(link)

    def beta_names(self, feature_names, n_beta=None):
        p = len(feature_names)
        if not p or not n_beta or n_beta % p != 0:
            return None
        k_minus_1 = n_beta // p
        return [f"cat{j}:{name}" for j in range(k_minus_1) for name in feature_names]


class Ordinal(MleFamily):
    MAIN_LABEL = "Regression Coefficients"
    LINK_MAP = {
        "logit": FamilyCode.FAM_ORDINAL_LOGIT,
        "probit": FamilyCode.FAM_ORDINAL_PROBIT,
        "cloglog": FamilyCode.FAM_ORDINAL_CLOGLOG,
        "loglog": FamilyCode.FAM_ORDINAL_LOGLOG,
        "cauchit": FamilyCode.FAM_ORDINAL_CAUCHIT,
    }

    def __init__(self, link="logit"):
        super().__init__(link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        if y is None:
            return None
        cats = sorted(set(np.asarray(y).flatten().tolist()))

        def _fmt(v):
            return str(int(v)) if float(v).is_integer() else str(v)

        names = [f"{_fmt(cats[i])}|{_fmt(cats[i + 1])}" for i in range(len(cats) - 1)]
        if n_gamma is not None and len(names) != n_gamma:
            return None
        return names

    def gamma_heading(self):
        return "Threshold Coefficients"


class ZeroInflatedPoisson(MleFamily):
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "infl_link", "logit"
    MAIN_LABEL = "Count Model"
    LINK_MAP = {
        "log": FamilyCode.FAM_ZIP_LOG,
        "identity": FamilyCode.FAM_ZIP_IDENTITY,
        "sqrt": FamilyCode.FAM_ZIP_SQRT,
        "inverse": FamilyCode.FAM_ZIP_INVERSE,
    }

    def __init__(self, link="log", infl_link="logit"):
        super().__init__(link)
        self.infl_link = _check_secondary_link(
            infl_link,
            _PROBABILITY_LINKS,
            self.__class__.__name__,
            "infl_link",
            "the zero-inflation probability",
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(link='{self.link}', infl_link='{self.infl_link}')"

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "infl")

    def gamma_heading(self):
        return f"Zero-Inflation Model (P(y=0), {self.infl_link} link)"


class ZeroInflatedNegativeBinomial(MleFamily):
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "infl_link", "logit"
    EXTRA_FREE = ("log(alpha)",)
    MAIN_LABEL = "Count Model"
    LINK_MAP = {
        "log": FamilyCode.FAM_ZINB_LOG,
        "sqrt": FamilyCode.FAM_ZINB_SQRT,
        "identity": FamilyCode.FAM_ZINB_IDENTITY,
        "inverse": FamilyCode.FAM_ZINB_INVERSE,
    }

    def __init__(self, link="log", infl_link="logit"):
        super().__init__(link)
        self.infl_link = _check_secondary_link(
            infl_link,
            _PROBABILITY_LINKS,
            self.__class__.__name__,
            "infl_link",
            "the zero-inflation probability",
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(link='{self.link}', infl_link='{self.infl_link}')"

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "infl")

    def gamma_heading(self):
        return f"Zero-Inflation Model (P(y=0), {self.infl_link} link)"

    def extra_param_label(self):
        return "alpha (NB dispersion, natural scale)"


class ValueInflatedPoisson(MleFamily):
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "infl_link", "logit"
    MAIN_LABEL = "Count Model"
    family_type = "mle_inflated"

    def __init__(self, target_value, base_link="log", infl_link="logit"):
        self.target_value = target_value
        self.base_link = base_link.lower()
        self.infl_link = _check_secondary_link(
            infl_link,
            _PROBABILITY_LINKS,
            self.__class__.__name__,
            "infl_link",
            "the inflation probability",
        )
        self.link = self.base_link
        self.code = None
        self.params = _glmfit.FamilyParams()

    def __repr__(self):
        return (
            f"ValueInflatedPoisson(target_value={self.target_value}, "
            f"base_link='{self.base_link}', infl_link='{self.infl_link}')"
        )

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "infl")

    def gamma_heading(self):
        return f"Inflation Model (P(y={self.target_value!r}), {self.infl_link} link)"


class ValueInflatedNegativeBinomial(MleFamily):
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "infl_link", "logit"
    EXTRA_FREE = ("log(alpha)",)
    MAIN_LABEL = "Count Model"
    family_type = "mle_inflated"
    _VAR_FORMS = ("quadratic", "linear")

    def __init__(self, target_value, var_form="quadratic", base_link="log", infl_link="logit"):
        var_form = var_form.lower()
        if var_form not in self._VAR_FORMS:
            raise ValueError(
                f"ValueInflatedNegativeBinomial: var_form must be one of "
                f"{self._VAR_FORMS} (quadratic=NB2, linear=NB1), not '{var_form}'."
            )
        self.target_value = target_value
        self.var_form = var_form
        self.base_link = base_link.lower()
        self.infl_link = _check_secondary_link(
            infl_link,
            _PROBABILITY_LINKS,
            self.__class__.__name__,
            "infl_link",
            "the inflation probability",
        )
        self.link = self.base_link
        self.code = None
        self.params = _glmfit.FamilyParams()

    def __repr__(self):
        return (
            f"ValueInflatedNegativeBinomial(target_value={self.target_value}, "
            f"var_form='{self.var_form}', base_link='{self.base_link}', "
            f"infl_link='{self.infl_link}')"
        )

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "infl")

    def gamma_heading(self):
        return f"Inflation Model (P(y={self.target_value!r}), {self.infl_link} link)"

    def extra_param_label(self):
        return "alpha (NB dispersion, natural scale)"

    def extra_param2_label(self):
        return "alpha (NB dispersion, log scale)"


class GeneralizedGamma(MleFamily):
    family_type = "mle_gengamma"

    def __init__(
        self,
        mx_mu_cols=(),
        mx_sigma_cols=(),
        mx_Q_cols=(),
        link_mu="identity",
        link_sigma="log",
        link_Q="identity",
    ):
        self.mx_mu_cols = np.asarray(mx_mu_cols, dtype=np.int32)
        self.mx_sigma_cols = np.asarray(mx_sigma_cols, dtype=np.int32)
        self.mx_Q_cols = np.asarray(mx_Q_cols, dtype=np.int32)
        self.link_mu = link_mu.lower()
        self.link_sigma = link_sigma.lower()
        self.link_Q = link_Q.lower()
        self.link = self.link_mu
        self.code = None
        self.params = _glmfit.FamilyParams()

    def __repr__(self):
        return (
            f"GeneralizedGamma(mx_mu_cols={list(self.mx_mu_cols)}, "
            f"mx_sigma_cols={list(self.mx_sigma_cols)}, "
            f"mx_Q_cols={list(self.mx_Q_cols)}, "
            f"link_mu='{self.link_mu}', link_sigma='{self.link_sigma}', "
            f"link_Q='{self.link_Q}')"
        )

    def beta_names(self, feature_names, n_beta=None):
        names = []
        for j in self.mx_mu_cols:
            names.append(f"mu:{feature_names[j]}")
        for j in self.mx_sigma_cols:
            names.append(f"sigma:{feature_names[j]}")
        for j in self.mx_Q_cols:
            names.append(f"Q:{feature_names[j]}")
        return names

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        if n_gamma is not None and n_gamma != 3:
            return None
        return ["mu_intercept", "sigma_intercept", "Q_intercept"]

    def gamma_heading(self):
        return "Distribution Parameters (mu, sigma, Q intercepts)"

    def extra_param_label(self):
        return "Q (shape, eta scale -- same as gamma[2]/Q_intercept)"

    def link_description(self):
        return f"mu={self.link_mu}, sigma={self.link_sigma}, Q={self.link_Q}"


class HurdlePoisson(MleFamily):
    MAIN_LABEL = "Truncated Count Model"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "zero_link", "logit"
    LINK_MAP = {
        "log": FamilyCode.FAM_HURDLE_POISSON_LOG,
        "sqrt": FamilyCode.FAM_HURDLE_POISSON_SQRT,
    }

    def __init__(self, link="log", zero_link="logit"):
        super().__init__(link)
        self._init_secondary_link(zero_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "hurdle")

    def gamma_heading(self):
        return f"Zero Hurdle Model (P(y>0), {self.gamma_link} link)"


class HurdleNegativeBinomial(MleFamily):
    EXTRA_FREE = ("log(alpha)",)
    MAIN_LABEL = "Truncated Count Model"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "zero_link", "logit"
    LINK_MAP = {
        "log": FamilyCode.FAM_HURDLE_NB_LOG,
        "sqrt": FamilyCode.FAM_HURDLE_NB_SQRT,
    }

    def __init__(self, link="log", zero_link="logit"):
        super().__init__(link)
        self._init_secondary_link(zero_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "hurdle")

    def gamma_heading(self):
        return f"Zero Hurdle Model (P(y>0), {self.gamma_link} link)"

    def extra_param_label(self):
        return "alpha (NB dispersion, natural scale)"


class HurdleGamma(MleFamily):
    EXTRA_FREE = ("log(shape)",)
    MAIN_LABEL = "Positive Continuous Model"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "zero_link", "logit"
    LINK_MAP = {
        "log": FamilyCode.FAM_HURDLE_GAMMA_LOG,
        "inverse": FamilyCode.FAM_HURDLE_GAMMA_INVERSE,
    }

    def __init__(self, link="log", zero_link="logit"):
        super().__init__(link)
        self._init_secondary_link(zero_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "hurdle")

    def gamma_heading(self):
        return f"Zero Hurdle Model (P(y>0), {self.gamma_link} link)"

    def extra_param_label(self):
        return "shape (Gamma distribution shape, natural scale)"


class HurdleLogNormal(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    MAIN_LABEL = "Positive Continuous Model"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "zero_link", "logit"
    LINK_MAP = {
        "log": FamilyCode.FAM_HURDLE_LOGNORMAL_LOG,
        "identity": FamilyCode.FAM_HURDLE_LOGNORMAL_IDENTITY,
    }

    def __init__(self, link="log", zero_link="logit"):
        super().__init__(link)
        self._init_secondary_link(zero_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "hurdle")

    def gamma_heading(self):
        return f"Zero Hurdle Model (P(y>0), {self.gamma_link} link)"

    def extra_param_label(self):
        return "sigma (natural scale)"


class HurdleBeta(MleFamily):
    EXTRA_FREE = ("log_phi",)
    MAIN_LABEL = "Positive Continuous Model"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "zero_link", "logit"
    SUPPORT = "unit"
    LINK_MAP = {
        "logit": FamilyCode.FAM_HURDLE_BETA_LOGIT,
        "probit": FamilyCode.FAM_HURDLE_BETA_PROBIT,
    }

    def __init__(self, link="logit", zero_link="logit"):
        super().__init__(link)
        self._init_secondary_link(zero_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "hurdle")

    def gamma_heading(self):
        return f"Zero Hurdle Model (P(y>0), {self.gamma_link} link)"

    def extra_param_label(self):
        return "phi (precision, natural scale)"


class HurdleGeometric(MleFamily):
    MAIN_LABEL = "Truncated Geometric Count Model"
    SECONDARY_KIND, SECONDARY_ARG, SECONDARY_DEFAULT = "probability", "zero_link", "logit"
    LINK_MAP = {
        "log": FamilyCode.FAM_HURDLE_GEOMETRIC_LOG,
        "identity": FamilyCode.FAM_HURDLE_GEOMETRIC_IDENTITY,
        "sqrt": FamilyCode.FAM_HURDLE_GEOMETRIC_SQRT,
    }

    def __init__(self, link="log", zero_link="logit"):
        super().__init__(link)
        self._init_secondary_link(zero_link)

    def gamma_names(self, y=None, n_gamma=None, feature_names=None):
        return _reused_x_names(feature_names, n_gamma, "hurdle")

    def gamma_heading(self):
        return f"Zero Hurdle Model (P(y>0), {self.gamma_link} link)"


class TruncatedPoisson(MleFamily):
    LINK_MAP = {
        "log": FamilyCode.FAM_TRUNC_POISSON_LOG,
        "identity": FamilyCode.FAM_TRUNC_POISSON_IDENTITY,
    }

    def __init__(self, link="log"):
        super().__init__(link)


class TruncatedNegativeBinomial(MleFamily):
    EXTRA_FREE = ("log(alpha)",)
    LINK_MAP = {
        "log": FamilyCode.FAM_TRUNC_NB_LOG,
        "identity": FamilyCode.FAM_TRUNC_NB_IDENTITY,
    }

    def __init__(self, link="log"):
        super().__init__(link)

    def extra_param_label(self):
        return "alpha (NB dispersion, natural scale)"


class TruncatedGeometric(MleFamily):
    LINK_MAP = {
        "log": FamilyCode.FAM_TRUNC_GEOMETRIC_LOG,
    }

    def __init__(self, link="log"):
        super().__init__(link)


class TruncatedGaussian(MleFamily):
    EXTRA_FREE = ("log(sigma)",)
    LINK_MAP = {
        "identity": FamilyCode.FAM_TRUNC_GAUSSIAN_IDENTITY,
        "log": FamilyCode.FAM_TRUNC_GAUSSIAN_LOG,
        "inverse": FamilyCode.FAM_TRUNC_GAUSSIAN_INVERSE,
    }

    def __init__(self, link="identity"):
        if link not in self.LINK_MAP:
            valid = ", ".join(f'"{k}"' for k in self.LINK_MAP)
            raise ValueError(
                f"TruncatedGaussian: link '{link}' is not supported. Valid links: {valid}."
            )
        super().__init__(link)

    def link_description(self) -> str:
        return self.link


class CustomFamily(Family):
    def __init__(self, variance_fn, mu_eta_fn, linkinv_fn, dev_resids_fn, valideta_fn, validmu_fn):
        self.link = None
        self.code = None
        self.params = _glmfit.FamilyParams()
        self.variance_fn = variance_fn
        self.mu_eta_fn = mu_eta_fn
        self.linkinv_fn = linkinv_fn
        self.dev_resids_fn = dev_resids_fn
        self.valideta_fn = valideta_fn
        self.validmu_fn = validmu_fn

    def make_engine(
        self, X, y, weights, offset, tol=1e-6, maxit=100, type=1, is_big_matrix=False, firth=False
    ):
        return _glmfit.IrlsEngine(
            X,
            y,
            weights,
            offset,
            self.variance_fn,
            self.mu_eta_fn,
            self.linkinv_fn,
            self.dev_resids_fn,
            self.valideta_fn,
            self.validmu_fn,
            tol,
            maxit,
            type,
            is_big_matrix,
            firth,
        )

    def linkinv(self, eta):
        return self.linkinv_fn(eta)

    def valid_eta(self, eta):
        return self.valideta_fn(eta)

    def valid_mu(self, mu):
        return self.validmu_fn(mu)

    def __repr__(self):
        return "CustomFamily(...)"


#: Public family classes. Re-exported from ``statcraft.glm``, ``statcraft.nlme``
#: and ``statcraft.saem`` so callers can import them from the engine they use.
__all__ = [
    "FamilyCode",
    "Family",
    "CustomFamily",
    "Gaussian",
    "Binomial",
    "Poisson",
    "Gamma",
    "InverseGaussian",
    "NegativeBinomial",
    "Tweedie",
    "Exponential",
    "Pareto",
    "InverseGamma",
    "QuasiFamily",
    "QuasiPoisson",
    "QuasiBinomial",
    "QuasiGamma",
    "QuasiInverseGaussian",
    "QuasiGaussian",
    "QuasiCustom",
    "MleFamily",
    "Beta",
    "LogNormal",
    "Weibull",
    "Gumbel",
    "LogLogistic",
    "StudentT",
    "SkewNormal",
    "SkewT",
    "LogisticDistribution",
    "Simplex",
    "Multinomial",
    "Ordinal",
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
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
    "ValueInflatedPoisson",
    "ValueInflatedNegativeBinomial",
    "GeneralizedGamma",
]
