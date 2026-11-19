// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file families.hpp
 * @brief GLM families and links: one (variance, link) table, one kernel per variance and per link.
 * @details Every numeric family code (kept for the API / bindings) maps through `family_of()` to a pair
 *          (`Var`, `Link`). Mean, derivative, variance, deviance and validity kernels are written once per
 *          variance function and once per link, so a new family is one row in `detail::kRows`.
 *          `eval_block()` fuses linkinv / mu' / V / w / deviance for a block of rows and reuses mu between
 *          them (no repeated exp()).
 */

#ifndef _GLMFIT_FAMILIES_HPP
#define _GLMFIT_FAMILIES_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace glm {

struct FamilyParams {
    double theta      = 1.0;   ///< Negative Binomial shape parameter (theta)
    double var_power  = 1.5;   ///< Tweedie power parameter (p)
    double link_power = 0.0;   ///< Power link exponent
};

inline double thresh_eps() {
    return std::numeric_limits<double>::epsilon() * 100.0;
}

enum FamilyCode {
    FAM_UNKNOWN = -1,

    /* Gaussian */
    FAM_GAUSSIAN_IDENTITY = 0,
    FAM_GAUSSIAN_LOG      = 1,
    FAM_GAUSSIAN_INVERSE  = 2,
    FAM_GAUSSIAN_SQRT     = 200,

    /* Binomial */
    FAM_BINOMIAL_LOGIT    = 10,
    FAM_BINOMIAL_PROBIT   = 11,
    FAM_BINOMIAL_CLOGLOG  = 12,
    FAM_BINOMIAL_LOG      = 13,
    FAM_BINOMIAL_LOGLOG   = 113,
    FAM_BINOMIAL_CAUCHIT  = 112,

    /* Poisson */
    FAM_POISSON_LOG       = 20,
    FAM_POISSON_IDENTITY  = 21,
    FAM_POISSON_SQRT      = 22,
    FAM_POISSON_INVERSE   = 115,

    /* Gamma */
    FAM_GAMMA_LOG         = 30,
    FAM_GAMMA_INVERSE     = 31,
    FAM_GAMMA_IDENTITY    = 32,
    FAM_GAMMA_SQRT        = 201,
    FAM_GAMMA_POWER       = 205,

    /* Inverse Gaussian */
    FAM_INVGAUSS_INVMU2   = 40,
    FAM_INVGAUSS_LOG      = 41,
    FAM_INVGAUSS_IDENTITY = 42,
    FAM_INVGAUSS_INVERSE  = 43,
    FAM_INVGAUSS_POWER    = 206,

    /* Negative Binomial */
    FAM_NB_LOG            = 50,
    FAM_NB_SQRT           = 51,
    FAM_NB_IDENTITY       = 52,
    FAM_NB_INVERSE        = 116,
    FAM_NB_POWER          = 207,

    /* Tweedie */
    FAM_TWEEDIE_LOG       = 60,
    FAM_TWEEDIE_IDENTITY  = 61,
    FAM_TWEEDIE_INVERSE   = 62,
    FAM_TWEEDIE_SQRT      = 63,
    FAM_TWEEDIE_POWER     = 208,

    /* Exponential / Pareto / InvGamma */
    FAM_EXPONENTIAL_IDENTITY = 120,
    FAM_EXPONENTIAL_LOG      = 121,
    FAM_EXPONENTIAL_INVERSE  = 122,
    FAM_EXPONENTIAL_SQRT     = 202,
    FAM_PARETO_LOG           = 117,
    FAM_PARETO_INVERSE       = 203,
    FAM_INVGAMMA_LOG         = 118,
    FAM_INVGAMMA_INVERSE     = 119,
    FAM_INVGAMMA_IDENTITY    = 204,

    /* Quasi */
    FAM_QUASI_POISSON_LOG         = 100,
    FAM_QUASI_POISSON_IDENTITY    = 101,
    FAM_QUASI_BINOMIAL_LOGIT      = 102,
    FAM_QUASI_BINOMIAL_PROBIT     = 103,
    FAM_QUASI_GAUSSIAN_IDENTITY   = 104,
    FAM_QUASI_GAUSSIAN_LOG        = 105,
    FAM_QUASI_GAMMA_LOG           = 106,
    FAM_QUASI_GAMMA_IDENTITY      = 107,
    FAM_QUASI_GAMMA_INVERSE       = 108,
    FAM_QUASI_INVGAUSSIAN_LOG     = 109,
    FAM_QUASI_INVGAUSSIAN_IDENTITY = 110,
    FAM_QUASI_INVGAUSSIAN_INVERSE = 111,

    /* Beta */
    FAM_BETA_LOGIT     = 130,
    FAM_BETA_PROBIT    = 131,
    FAM_BETA_CLOGLOG   = 132,
    FAM_BETA_LOGLOG    = 133,
    FAM_BETA_CAUCHIT   = 212,

    /* Log-normal / Weibull / Gumbel / Log-logistic */
    FAM_LOGNORMAL_IDENTITY = 134,
    FAM_LOGNORMAL_LOG      = 135,
    FAM_LOGNORMAL_INVERSE  = 136,
    FAM_WEIBULL_IDENTITY   = 138,
    FAM_WEIBULL_LOG        = 137,
    FAM_GUMBEL_IDENTITY    = 139,
    FAM_GUMBEL_LOG         = 140,
    FAM_LOGLOGISTIC_IDENTITY = 141,
    FAM_LOGLOGISTIC_LOG      = 142,

    /* Simplex */
    FAM_SIMPLEX_LOGIT    = 143,
    FAM_SIMPLEX_PROBIT   = 144,
    FAM_SIMPLEX_LOGLOG   = 209,
    FAM_SIMPLEX_CLOGLOG  = 210,
    FAM_SIMPLEX_CAUCHIT  = 211,

    /* Multinomial / Ordinal */
    FAM_MULTINOMIAL_LOGIT    = 148,
    FAM_MULTINOMIAL_PROBIT   = 149,
    FAM_MULTINOMIAL_CLOGLOG  = 150,
    FAM_MULTINOMIAL_LOGLOG   = 146,
    FAM_MULTINOMIAL_CAUCHIT  = 147,
    FAM_ORDINAL_LOGIT        = 151,
    FAM_ORDINAL_PROBIT       = 152,
    FAM_ORDINAL_CLOGLOG      = 153,
    FAM_ORDINAL_LOGLOG       = 154,
    FAM_ORDINAL_CAUCHIT      = 155,

    /* Location-scale / others */
    FAM_STUDENT_T_IDENTITY  = 156,
    FAM_STUDENT_T_LOG       = 157,
    FAM_SKEW_NORMAL_IDENTITY = 158,
    FAM_SKEW_NORMAL_LOG      = 159,
    FAM_SKEW_T_IDENTITY      = 160,
    FAM_SKEW_T_LOG           = 161,
    FAM_LOGISTIC_IDENTITY    = 162,
    FAM_LOGISTIC_LOG         = 163,

    /* Zero-inflated / Hurdle */
    FAM_ZIP_LOG             = 164,
    FAM_ZIP_IDENTITY        = 165,
    FAM_ZIP_SQRT            = 166,
    FAM_ZIP_INVERSE         = 167,
    FAM_ZINB_LOG            = 168,
    FAM_ZINB_SQRT           = 169,
    FAM_ZINB_IDENTITY       = 170,
    FAM_ZINB_INVERSE        = 171,
    FAM_HURDLE_POISSON_LOG   = 172,
    FAM_HURDLE_POISSON_SQRT  = 173,
    FAM_HURDLE_NB_LOG        = 174,
    FAM_HURDLE_NB_SQRT       = 175,
    FAM_HURDLE_GAMMA_LOG     = 176,
    FAM_HURDLE_GAMMA_INVERSE = 177,
    FAM_HURDLE_LOGNORMAL_LOG = 178,
    FAM_HURDLE_LOGNORMAL_IDENTITY = 179,
    FAM_HURDLE_BETA_LOGIT    = 180,
    FAM_HURDLE_BETA_PROBIT   = 181,
    FAM_HURDLE_BETA_LOGLOG   = 213,
    FAM_HURDLE_BETA_CLOGLOG  = 214,
    FAM_HURDLE_BETA_CAUCHIT  = 215,
    FAM_HURDLE_GEOMETRIC_LOG      = 216,
    FAM_HURDLE_GEOMETRIC_IDENTITY = 217,
    FAM_HURDLE_GEOMETRIC_SQRT     = 218,

    /* Truncated */
    FAM_TRUNC_POISSON_LOG      = 182,
    FAM_TRUNC_POISSON_IDENTITY = 183,
    FAM_TRUNC_NB_LOG           = 184,
    FAM_TRUNC_NB_IDENTITY      = 185,
    FAM_TRUNC_GEOMETRIC_LOG    = 186,
    FAM_TRUNC_GAUSSIAN_IDENTITY = 187,
    FAM_TRUNC_GAUSSIAN_LOG      = 188,
    FAM_TRUNC_GAUSSIAN_INVERSE  = 189
};


// ============================================================================================
// (variance, link) decomposition
// ============================================================================================

/// Variance function / deviance family.
enum class Var : unsigned char { Gaussian, Binomial, Poisson, Gamma, InvGauss, NB, Tweedie };
/// Link function.
enum class Link : unsigned char { Identity, Log, Inverse, Sqrt, InvMu2, Logit, Probit, Cloglog, Loglog, Cauchit, Power };

/// `known == false` for codes without an IRLS kernel (they behave like Gaussian/identity, as before).
struct Family {
    Var  var   = Var::Gaussian;
    Link link  = Link::Identity;
    bool known = false;
};

namespace detail {

struct Row { int code; Var var; Link link; };

// Exponential / Pareto / InvGamma share the Gamma variance (V = mu^2) and deviance; quasi families share their base.
inline constexpr Row kRows[] = {
    // Gaussian
    {FAM_GAUSSIAN_IDENTITY, Var::Gaussian, Link::Identity}, {FAM_GAUSSIAN_LOG, Var::Gaussian, Link::Log},
    {FAM_GAUSSIAN_INVERSE, Var::Gaussian, Link::Inverse},   {FAM_GAUSSIAN_SQRT, Var::Gaussian, Link::Sqrt},
    {FAM_QUASI_GAUSSIAN_IDENTITY, Var::Gaussian, Link::Identity}, {FAM_QUASI_GAUSSIAN_LOG, Var::Gaussian, Link::Log},
    // Binomial
    {FAM_BINOMIAL_LOGIT, Var::Binomial, Link::Logit},   {FAM_BINOMIAL_PROBIT, Var::Binomial, Link::Probit},
    {FAM_BINOMIAL_CLOGLOG, Var::Binomial, Link::Cloglog}, {FAM_BINOMIAL_LOG, Var::Binomial, Link::Log},
    {FAM_BINOMIAL_LOGLOG, Var::Binomial, Link::Loglog}, {FAM_BINOMIAL_CAUCHIT, Var::Binomial, Link::Cauchit},
    {FAM_QUASI_BINOMIAL_LOGIT, Var::Binomial, Link::Logit}, {FAM_QUASI_BINOMIAL_PROBIT, Var::Binomial, Link::Probit},
    // Poisson
    {FAM_POISSON_LOG, Var::Poisson, Link::Log},       {FAM_POISSON_IDENTITY, Var::Poisson, Link::Identity},
    {FAM_POISSON_SQRT, Var::Poisson, Link::Sqrt},     {FAM_POISSON_INVERSE, Var::Poisson, Link::Inverse},
    {FAM_QUASI_POISSON_LOG, Var::Poisson, Link::Log}, {FAM_QUASI_POISSON_IDENTITY, Var::Poisson, Link::Identity},
    // Gamma (+ Exponential, Pareto, InvGamma)
    {FAM_GAMMA_LOG, Var::Gamma, Link::Log},         {FAM_GAMMA_INVERSE, Var::Gamma, Link::Inverse},
    {FAM_GAMMA_IDENTITY, Var::Gamma, Link::Identity}, {FAM_GAMMA_SQRT, Var::Gamma, Link::Sqrt},
    {FAM_GAMMA_POWER, Var::Gamma, Link::Power},
    {FAM_EXPONENTIAL_IDENTITY, Var::Gamma, Link::Identity}, {FAM_EXPONENTIAL_LOG, Var::Gamma, Link::Log},
    {FAM_EXPONENTIAL_INVERSE, Var::Gamma, Link::Inverse},   {FAM_EXPONENTIAL_SQRT, Var::Gamma, Link::Sqrt},
    {FAM_PARETO_LOG, Var::Gamma, Link::Log},        {FAM_PARETO_INVERSE, Var::Gamma, Link::Inverse},
    {FAM_INVGAMMA_LOG, Var::Gamma, Link::Log},      {FAM_INVGAMMA_INVERSE, Var::Gamma, Link::Inverse},
    {FAM_INVGAMMA_IDENTITY, Var::Gamma, Link::Identity},
    {FAM_QUASI_GAMMA_LOG, Var::Gamma, Link::Log},   {FAM_QUASI_GAMMA_IDENTITY, Var::Gamma, Link::Identity},
    {FAM_QUASI_GAMMA_INVERSE, Var::Gamma, Link::Inverse},
    // Inverse Gaussian
    {FAM_INVGAUSS_INVMU2, Var::InvGauss, Link::InvMu2}, {FAM_INVGAUSS_LOG, Var::InvGauss, Link::Log},
    {FAM_INVGAUSS_IDENTITY, Var::InvGauss, Link::Identity}, {FAM_INVGAUSS_INVERSE, Var::InvGauss, Link::Inverse},
    {FAM_INVGAUSS_POWER, Var::InvGauss, Link::Power},
    {FAM_QUASI_INVGAUSSIAN_LOG, Var::InvGauss, Link::Log}, {FAM_QUASI_INVGAUSSIAN_IDENTITY, Var::InvGauss, Link::Identity},
    {FAM_QUASI_INVGAUSSIAN_INVERSE, Var::InvGauss, Link::Inverse},
    // Negative binomial
    {FAM_NB_LOG, Var::NB, Link::Log}, {FAM_NB_SQRT, Var::NB, Link::Sqrt}, {FAM_NB_IDENTITY, Var::NB, Link::Identity},
    {FAM_NB_INVERSE, Var::NB, Link::Inverse}, {FAM_NB_POWER, Var::NB, Link::Power},
    // Tweedie
    {FAM_TWEEDIE_LOG, Var::Tweedie, Link::Log}, {FAM_TWEEDIE_IDENTITY, Var::Tweedie, Link::Identity},
    {FAM_TWEEDIE_INVERSE, Var::Tweedie, Link::Inverse}, {FAM_TWEEDIE_SQRT, Var::Tweedie, Link::Sqrt},
    {FAM_TWEEDIE_POWER, Var::Tweedie, Link::Power},
};

constexpr std::array<Family, 256> make_table() {
    std::array<Family, 256> t{};
    for (const Row& r : kRows) t[static_cast<std::size_t>(r.code)] = Family{r.var, r.link, true};
    return t;
}
inline constexpr std::array<Family, 256> kTable = make_table();

}  // namespace detail

/// O(1) lookup of the (variance, link) pair of a family code.
[[nodiscard]] inline Family family_of(int code) {
    return (code >= 0 && code < 256) ? detail::kTable[static_cast<std::size_t>(code)] : Family{};
}

inline bool is_irls_family(int code) {
    return (code >= FAM_GAUSSIAN_IDENTITY && code <= FAM_EXPONENTIAL_INVERSE) ||
           (code >= FAM_QUASI_POISSON_LOG && code <= FAM_QUASI_INVGAUSSIAN_INVERSE) ||
           (code >= FAM_GAUSSIAN_SQRT && code <= FAM_TWEEDIE_POWER);
}

inline bool has_incomplete_dispatch(int code) {
    (void)code;
    return false;
}

inline bool has_unit_dispersion(int code) {
    return (code >= FAM_BINOMIAL_LOGIT && code <= FAM_BINOMIAL_LOG) ||
           code == FAM_BINOMIAL_CAUCHIT || code == FAM_BINOMIAL_LOGLOG ||
           (code >= FAM_POISSON_LOG && code <= FAM_POISSON_SQRT) || code == FAM_POISSON_INVERSE ||
           (code >= FAM_NB_LOG && code <= FAM_NB_IDENTITY) || code == FAM_NB_INVERSE || code == FAM_NB_POWER ||
           (code >= FAM_EXPONENTIAL_IDENTITY && code <= FAM_EXPONENTIAL_INVERSE) || code == FAM_EXPONENTIAL_SQRT;
}

/// True for the families whose mean lives in (0, 1).
inline bool is_binomial_family(int c) {
    const Family f = family_of(c);
    return f.known && f.var == Var::Binomial;
}

namespace detail {

// Link predicates (kept for source compatibility); false for codes without a kernel.
inline bool has_link(int c, Link l) { const Family f = family_of(c); return f.known && f.link == l; }
inline bool is_log_link(int c)      { return has_link(c, Link::Log); }
inline bool is_identity_link(int c) { return has_link(c, Link::Identity); }
inline bool is_inverse_link(int c)  { return has_link(c, Link::Inverse); }
inline bool is_sqrt_link(int c)     { return has_link(c, Link::Sqrt); }
inline bool is_logit_link(int c)    { return has_link(c, Link::Logit); }
inline bool is_probit_link(int c)   { return has_link(c, Link::Probit); }
inline bool is_cloglog_link(int c)  { return has_link(c, Link::Cloglog); }
inline bool is_loglog_link(int c)   { return has_link(c, Link::Loglog); }
inline bool is_cauchit_link(int c)  { return has_link(c, Link::Cauchit); }
inline bool is_invmu2_link(int c)   { return has_link(c, Link::InvMu2); }
inline bool is_power_link(int c)    { return has_link(c, Link::Power); }

using CArr = Eigen::Ref<const Eigen::ArrayXd>;
using MArr = Eigen::Ref<Eigen::ArrayXd>;

inline constexpr double kEps       = std::numeric_limits<double>::epsilon();
inline constexpr double kProbitMax = 8.125890664701906;   // -qnorm(DBL_EPSILON)

// --------------------------------------------------------------------------------------------
// Links (written once; every family goes through these)
// --------------------------------------------------------------------------------------------

/// mu = g^-1(eta). Binomial links are clamped like R's C code so mu stays strictly inside (0, 1).
inline void link_inv(Link l, const FamilyParams& p, const CArr& eta, MArr mu) {
    switch (l) {
        case Link::Identity: mu = eta; break;
        case Link::Log:      mu = eta.exp(); break;
        case Link::Inverse:  mu = 1.0 / eta; break;
        case Link::Sqrt:     mu = eta.square(); break;
        case Link::InvMu2:   mu = 1.0 / eta.sqrt(); break;
        case Link::Logit: {   // R: THRESH = 30
            const Eigen::ArrayXd e = eta.exp();
            mu = (eta > 30.0).select(1.0 - kEps, (eta < -30.0).select(kEps, e / (1.0 + e)));
            break;
        }
        case Link::Probit:
            mu = 0.5 * (-eta.max(-kProbitMax).min(kProbitMax) / std::sqrt(2.0))
                           .unaryExpr([](double x) { return std::erfc(x); });
            break;
        case Link::Cloglog:  mu = (-(-eta.exp()).expm1()).max(kEps).min(1.0 - kEps); break;
        case Link::Loglog:   mu = (-eta.exp()).exp(); break;
        case Link::Cauchit:  mu = (eta.atan() / M_PI) + 0.5; break;
        case Link::Power:
            if (p.link_power == 0.0) mu = eta.exp();
            else                     mu = eta.pow(1.0 / p.link_power);
            break;
    }
}

/// eta = g(mu), the inverse of `link_inv`.
inline void link_fun(Link l, const FamilyParams& p, const CArr& mu, MArr eta);

/// d mu / d eta. If `mu` (= g^-1(eta)) is supplied it is reused instead of recomputing exp() etc.
inline void link_mu_eta(Link l, const FamilyParams& p, const CArr& eta, const CArr* mu, MArr d) {
    switch (l) {
        case Link::Identity: d.setOnes(); break;
        case Link::Log:      if (mu) d = *mu; else d = eta.exp(); break;
        case Link::Inverse:  d = -1.0 / eta.square(); break;
        case Link::Sqrt:     d = 2.0 * eta; break;
        case Link::InvMu2:   d = -0.5 / (eta * eta.sqrt()); break;
        case Link::Logit:
            if (mu) {
                d = (eta.abs() > 30.0).select(Eigen::ArrayXd::Constant(eta.size(), kEps), (*mu) * (1.0 - *mu));
            } else {
                const Eigen::ArrayXd e = eta.exp();
                d = (eta.abs() > 30.0).select(Eigen::ArrayXd::Constant(eta.size(), kEps), e / (1.0 + e).square());
            }
            break;
        case Link::Probit:   d = ((-0.5 * eta.square()).exp() / std::sqrt(2.0 * M_PI)).max(kEps); break;
        case Link::Cloglog: {
            const Eigen::ArrayXd ee = eta.min(700.0).exp();
            d = (ee * (-ee).exp()).max(kEps);
            break;
        }
        case Link::Loglog: {
            const Eigen::ArrayXd ee = eta.exp();
            if (mu) d = -ee * (*mu); else d = -ee * (-ee).exp();
            break;
        }
        case Link::Cauchit:  d = 1.0 / (M_PI * (1.0 + eta.square())); break;
        case Link::Power:
            if (p.link_power == 0.0) { if (mu) d = *mu; else d = eta.exp(); }
            else d = (1.0 / p.link_power) * eta.pow(1.0 / p.link_power - 1.0);
            break;
    }
}

/// d^2 mu / d eta^2 in closed form (the previous version used finite differences for most links).
/// `dmu` must be d mu / d eta at `eta` (as returned by `link_mu_eta`), `mu` the mean.
inline void link_d2(Link l, const FamilyParams& p, const CArr& eta, const CArr& mu, const CArr& dmu, MArr d2) {
    switch (l) {
        case Link::Identity: d2.setZero(); break;
        case Link::Log:      d2 = dmu; break;
        case Link::Inverse:  d2 = 2.0 / eta.cube(); break;
        case Link::Sqrt:     d2.setConstant(2.0); break;
        case Link::InvMu2:   d2 = 0.75 / (eta.square() * eta.sqrt()); break;
        case Link::Logit:    d2 = dmu * (1.0 - 2.0 * mu); break;
        case Link::Probit:   d2 = -eta * dmu; break;
        case Link::Cloglog:  d2 = dmu * (1.0 - eta.min(700.0).exp()); break;
        case Link::Loglog:   d2 = dmu * (1.0 - eta.exp()); break;
        case Link::Cauchit:  d2 = -2.0 * M_PI * eta * dmu.square(); break;
        case Link::Power:
            if (p.link_power == 0.0) d2 = dmu;
            else {
                const double a = 1.0 / p.link_power;
                d2 = a * (a - 1.0) * eta.pow(a - 2.0);
            }
            break;
    }
}

// --------------------------------------------------------------------------------------------
// Variance functions and deviances (written once per variance family)
// --------------------------------------------------------------------------------------------

inline void var_fun(Var v, const FamilyParams& p, const CArr& mu, MArr out) {
    switch (v) {
        case Var::Gaussian: out.setOnes(); break;
        case Var::Binomial: out = mu * (1.0 - mu); break;
        case Var::Poisson:  out = mu; break;
        case Var::Gamma:    out = mu.square(); break;
        case Var::InvGauss: out = mu.cube(); break;
        case Var::NB:       out = mu + (mu.square() / p.theta); break;
        case Var::Tweedie:  out = mu.pow(p.var_power); break;
    }
}

/// Unit deviance d(y, mu); 0*log(0) terms and non-finite logs of the saturated model count as 0.
inline void unit_dev(Var v, const FamilyParams& p, const CArr& y, const CArr& mu, MArr dev) {
    const double eps = thresh_eps();
    switch (v) {
        case Var::Gaussian: dev = (y - mu).square(); break;
        case Var::Poisson: {
            const Eigen::ArrayXd term = y * (y / mu.max(eps)).log() - (y - mu);
            dev = 2.0 * (y > 0.0).select(term, mu);
            break;
        }
        case Var::Binomial: {
            const Eigen::ArrayXd t1 = y * (y / mu.max(eps)).log();
            const Eigen::ArrayXd t2 = (1.0 - y) * ((1.0 - y) / (1.0 - mu).max(eps)).log();
            dev = 2.0 * (t1.isFinite().select(t1, 0.0) + t2.isFinite().select(t2, 0.0));
            break;
        }
        case Var::Gamma:
            dev = -2.0 * ((y.max(eps) / mu.max(eps)).log() - (y - mu) / mu.max(eps));
            break;
        case Var::InvGauss:
            dev = (y - mu).square() / (y * mu.square()).max(eps);
            break;
        case Var::NB: {
            const double th = p.theta;
            const Eigen::ArrayXd t1 = y * (y.max(eps) / mu.max(eps)).log();
            // log((mu+th)/(y+th)) = log1p((mu-y)/(y+th)): no cancellation for large theta (Poisson limit)
            const Eigen::ArrayXd t2 = (y + th) * ((mu - y) / (y + th)).log1p();
            dev = 2.0 * (t1.isFinite().select(t1, 0.0) + t2);
            break;
        }
        case Var::Tweedie: {
            const double vp = p.var_power;
            const Eigen::ArrayXd ms = mu.max(eps);
            if (std::abs(vp - 1.0) < 1e-8) {
                const Eigen::ArrayXd term = y * (y.max(eps) / ms).log() - (y - mu);
                dev = 2.0 * (y > 0.0).select(term, mu);
            } else if (std::abs(vp - 2.0) < 1e-8) {
                dev = -2.0 * ((y.max(eps) / ms).log() - (y - mu) / ms);
            } else {
                const Eigen::ArrayXd tpos = y.pow(2.0 - vp) / ((1.0 - vp) * (2.0 - vp))
                                          - y * ms.pow(1.0 - vp) / (1.0 - vp)
                                          + ms.pow(2.0 - vp) / (2.0 - vp);
                const Eigen::ArrayXd tzero = ms.pow(2.0 - vp) / (2.0 - vp);
                dev = 2.0 * (y > 0.0).select(tpos, tzero);
            }
            break;
        }
    }
}

}  // namespace detail

// ============================================================================================
// Public kernels (same names / signatures as before)
// ============================================================================================

inline void linkinv(int code, const FamilyParams &p,
                    const Eigen::Ref<const Eigen::ArrayXd> &eta,
                    Eigen::Ref<Eigen::ArrayXd> mu) {
    detail::link_inv(family_of(code).link, p, eta, mu);
}

inline void mu_eta(int code, const FamilyParams &p,
                   const Eigen::Ref<const Eigen::ArrayXd> &eta,
                   Eigen::Ref<Eigen::ArrayXd> dmu_deta) {
    detail::link_mu_eta(family_of(code).link, p, eta, nullptr, dmu_deta);
}

/// d mu/d eta reusing an already computed `mu = linkinv(eta)` (saves an exp() for log / logit / loglog links).
inline void mu_eta(int code, const FamilyParams &p,
                   const Eigen::Ref<const Eigen::ArrayXd> &eta,
                   const Eigen::ArrayXd &mu,
                   Eigen::Ref<Eigen::ArrayXd> dmu_deta) {
    const detail::CArr m(mu);
    detail::link_mu_eta(family_of(code).link, p, eta, &m, dmu_deta);
}

inline void variance(int code, const FamilyParams &p,
                     const Eigen::Ref<const Eigen::ArrayXd> &mu,
                     Eigen::Ref<Eigen::ArrayXd> v) {
    detail::var_fun(family_of(code).var, p, mu, v);
}

/// Numerically simplified weights w = sqrt(pw dmu^2 / V) for the (variance, link) pairs where it pays off.
/// Returns false when no closed form is available (use sqrt(pw * dmu^2 / V)).
inline bool stable_weights(int code, const FamilyParams &p,
                           const Eigen::Ref<const Eigen::ArrayXd> &mu,
                           const Eigen::Ref<const Eigen::ArrayXd> &dmu_deta,
                           const Eigen::Ref<const Eigen::ArrayXd> &pw,
                           Eigen::Ref<Eigen::ArrayXd> w) {
    (void)dmu_deta;
    const Family f = family_of(code);
    if (!f.known) return false;
    switch (f.var) {
        case Var::Binomial:
            if (f.link == Link::Logit) { w = (pw * mu * (1.0 - mu)).sqrt(); return true; }
            return false;
        case Var::Poisson:
            if (f.link == Link::Log)   { w = (pw * mu).sqrt(); return true; }
            return false;
        case Var::Gamma:
            if (f.link == Link::Log)   { w = pw.sqrt(); return true; }          // mu^2 / mu^2
            return false;
        case Var::Gaussian:
            if (f.link == Link::Identity) { w = pw.sqrt(); return true; }
            return false;
        case Var::NB: {
            const Eigen::ArrayXd tm = (p.theta + mu).max(thresh_eps());
            if (f.link == Link::Log)      { w = (pw * mu * p.theta / tm).sqrt(); return true; }
            if (f.link == Link::Sqrt)     { w = (pw * 4.0 * p.theta / tm).sqrt(); return true; }
            if (f.link == Link::Identity) {
                const Eigen::ArrayXd denom = (mu * tm).max(thresh_eps());
                w = (pw * p.theta / denom).sqrt();
                return true;
            }
            return false;
        }
        default: return false;
    }
}

inline void d2mu_deta2(int code, const FamilyParams &p,
                       const Eigen::Ref<const Eigen::ArrayXd> &eta,
                       const Eigen::Ref<const Eigen::ArrayXd> &mu,
                       const Eigen::Ref<const Eigen::ArrayXd> &dmu,
                       Eigen::Ref<Eigen::ArrayXd> d2mu) {
    detail::link_d2(family_of(code).link, p, eta, mu, dmu, d2mu);
}

inline void unit_deviance(int code, const FamilyParams &p,
                          const Eigen::Ref<const Eigen::ArrayXd> &y,
                          const Eigen::Ref<const Eigen::ArrayXd> &mu,
                          Eigen::Ref<Eigen::ArrayXd> dev_i) {
    detail::unit_dev(family_of(code).var, p, y, mu, dev_i);
}

inline void dev_resids(int code, const FamilyParams &p,
                       const Eigen::Ref<const Eigen::ArrayXd> &y,
                       const Eigen::Ref<const Eigen::ArrayXd> &mu,
                       const Eigen::Ref<const Eigen::ArrayXd> &weights,
                       Eigen::Ref<Eigen::ArrayXd> out) {
    unit_deviance(code, p, y, mu, out);
    out = weights * out;
}

inline double dev_resids_sum(int code, const FamilyParams &p,
                             const Eigen::Ref<const Eigen::ArrayXd> &y,
                             const Eigen::Ref<const Eigen::ArrayXd> &mu,
                             const Eigen::Ref<const Eigen::ArrayXd> &weights) {
    Eigen::ArrayXd dev_i(y.size());
    unit_deviance(code, p, y, mu, dev_i);
    return (weights * dev_i).sum();
}

/// eta must be finite, and inside the domain of the link (inverse: != 0; sqrt / invmu2: > 0; power: > 0 unless log).
inline bool valideta(int code, const FamilyParams &p,
                     const Eigen::Ref<const Eigen::ArrayXd> &eta) {
    if (!eta.isFinite().all()) return false;
    const Family f = family_of(code);
    if (!f.known) return true;
    switch (f.link) {
        case Link::Inverse: return (eta != 0.0).all();
        case Link::Sqrt:
        case Link::InvMu2:  return (eta > 0.0).all();
        case Link::Power:   return (p.link_power == 0.0) || (eta > 0.0).all();
        default:            return true;
    }
}

/// mu must be finite and inside the support of the family (binomial: (0,1); Gaussian: any; others: > 0).
inline bool validmu(int code, const FamilyParams &,
                    const Eigen::Ref<const Eigen::ArrayXd> &mu) {
    if (!mu.isFinite().all()) return false;
    const Family f = family_of(code);
    if (!f.known) return true;
    switch (f.var) {
        case Var::Gaussian: return true;
        case Var::Binomial: return (mu > 0.0 && mu < 1.0).all();
        default:            return (mu > 0.0).all();
    }
}

/**
 * @brief Fused evaluation for a block of rows: (optionally) mu = g^-1(eta), then mu', V, w and the weighted
 *        deviance sum, sharing mu between all of them.
 * @param recompute_mu if false the supplied `mu` is used as is and no validity checks are made.
 * @return false if eta or mu is invalid (nothing past the failing check is computed).
 */
inline bool eval_block(int code, const FamilyParams &p, bool recompute_mu,
                       const Eigen::Ref<const Eigen::ArrayXd> &eta,
                       const Eigen::Ref<const Eigen::ArrayXd> &y,
                       const Eigen::Ref<const Eigen::ArrayXd> &pw,
                       Eigen::Ref<Eigen::ArrayXd> mu, Eigen::Ref<Eigen::ArrayXd> dmu,
                       Eigen::Ref<Eigen::ArrayXd> var, Eigen::Ref<Eigen::ArrayXd> w,
                       double &dev_sum) {
    const Family f = family_of(code);
    if (recompute_mu) {
        if (!valideta(code, p, eta)) return false;
        detail::link_inv(f.link, p, eta, mu);
        if (!validmu(code, p, mu)) return false;
    }
    const detail::CArr muv(mu);                         // read-only view, shared by mu', V, w and the deviance
    detail::link_mu_eta(f.link, p, eta, &muv, dmu);
    detail::var_fun(f.var, p, muv, var);
    if (!stable_weights(code, p, muv, dmu, pw, w)) w = (pw * dmu.square() / var).sqrt();
    w = w.isFinite().select(w, 0.0);
    Eigen::ArrayXd buf(y.size());
    detail::unit_dev(f.var, p, y, muv, buf);
    dev_sum = (pw * buf).sum();
    return true;
}

// ---------------------------------------------------------------------------
// Link function, starting values and helpers used by the IRLS driver
// ---------------------------------------------------------------------------

namespace detail {
/// Standard normal quantile (Acklam's rational approximation + one Halley step; |err| ~ 1e-15).
inline double qnorm_(double p) {
    if (!(p > 0.0)) return -std::numeric_limits<double>::infinity();
    if (!(p < 1.0)) return  std::numeric_limits<double>::infinity();
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    double x;
    if (p < 0.02425) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else if (p > 1.0 - 0.02425) {
        const double q = std::sqrt(-2.0 * std::log1p(-p));
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    }
    const double e = 0.5 * std::erfc(-x / std::sqrt(2.0)) - p;
    const double u = e * std::sqrt(2.0 * M_PI) * std::exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

inline void link_fun(Link l, const FamilyParams& p, const CArr& mu, MArr eta) {
    switch (l) {
        case Link::Identity: eta = mu; break;
        case Link::Log:      eta = mu.log(); break;
        case Link::Inverse:  eta = 1.0 / mu; break;
        case Link::Sqrt:     eta = mu.sqrt(); break;
        case Link::InvMu2:   eta = 1.0 / mu.square(); break;
        case Link::Logit:    eta = (mu / (1.0 - mu)).log(); break;
        case Link::Probit:   eta = mu.unaryExpr([](double x) { return qnorm_(x); }); break;
        case Link::Cloglog:  eta = (-(1.0 - mu).log()).log(); break;
        case Link::Loglog:   eta = (-mu.log()).log(); break;
        case Link::Cauchit:  eta = (M_PI * (mu - 0.5)).tan(); break;
        case Link::Power:
            if (p.link_power == 0.0) eta = mu.log();
            else                     eta = mu.pow(p.link_power);
            break;
    }
}
}  // namespace detail

/// Link function g(mu) = eta, the inverse of `linkinv` for every family code.
inline void linkfun(int code, const FamilyParams &p,
                    const Eigen::Ref<const Eigen::ArrayXd> &mu,
                    Eigen::Ref<Eigen::ArrayXd> eta) {
    detail::link_fun(family_of(code).link, p, mu, eta);
}

/// Starting means in the spirit of R's family$initialize (binomial: (w y + 1/2)/(w + 1); count families:
/// y shifted away from 0; continuous families: y), forced into the valid mean range for the family.
inline void mustart(int code, const FamilyParams &,
                    const Eigen::Ref<const Eigen::ArrayXd> &y,
                    const Eigen::Ref<const Eigen::ArrayXd> &pw,
                    Eigen::Ref<Eigen::ArrayXd> mu) {
    const Family f = family_of(code);
    const bool binom = f.var == Var::Binomial;
    if (binom)                       mu = (pw * y + 0.5) / (pw + 1.0);
    else if (f.var == Var::NB)       mu = y + (y == 0.0).cast<double>() / 6.0;
    else if (f.var == Var::Poisson)  mu = y + 0.1;
    else if (f.var == Var::Tweedie)  mu = y + 0.1 * (y == 0.0).cast<double>();
    else                             mu = y;

    if (!binom && !(f.var == Var::Gaussian && f.link == Link::Identity)) {
        // positive-mean families (and Gaussian with a non-identity link) need mu > 0
        const double tiny = 1e-8;
        double s = 0.0; Eigen::Index m = 0;
        for (Eigen::Index i = 0; i < mu.size(); ++i) if (y[i] > 0.0 && pw[i] > 0.0) { s += y[i]; ++m; }
        const double fill = m > 0 ? s / static_cast<double>(m) : 1.0;
        mu = (mu > tiny).select(mu, fill);
    }
}

} // namespace glm


#endif // _GLMFIT_FAMILIES_HPP
