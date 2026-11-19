// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file mle_utils.hpp
 * @brief Auxiliary mathematical utilities and link function evaluation routines for MLE.
 * @details Implements link functions (Identity, Log, Sqrt, Logit, Probit, Cloglog, Inverse, etc.),
 *          standard normal CDF/PDF routines, digamma/trigamma asymptotic expansions, stable log-sum-exp,
 *          Inverse Mills Ratio evaluations, and generic weighted IRLS solver wrappers.
 */

#ifndef _GLMFIT_MLE_UTILS_HPP
#define _GLMFIT_MLE_UTILS_HPP

#include <cmath>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>

namespace glm {

/** @brief $1 / \sqrt{2\pi}$ constant. */
constexpr double glm_INV_SQRT_2PI = 0.3989422804014327;
/** @brief $1 / \sqrt{2}$ constant. */
constexpr double glm_SQRT_1_2     = 0.7071067811865475;
/** @brief $\ln\sqrt{2\pi}$ constant. */
constexpr double glm_LOG_SQRT_2PI = 0.9189385332046727;
/** @brief $\pi$ constant. */
constexpr double glm_LINK_PI      = 3.14159265358979323846;

/**
 * @brief Link function type enumeration.
 */
enum class LinkType {
    Identity,
    Log,
    Sqrt,
    Logit,
    Probit,
    Cloglog,
    Inverse,
    Loglog,
    Cauchit,
    InverseSquared
};

/**
 * @brief Variance formulation type (Linear vs Quadratic).
 */
enum class VarianceForm {
    Linear,
    Quadratic
};

/**
 * @brief Precision parameter formulation type.
 */
enum class PrecisionForm {
    Standard,
    Simis
};

// ---------------------------------------------------------------------------
// SPECIAL MATHEMATICAL FUNCTIONS
// ---------------------------------------------------------------------------

/**
 * @brief Standard normal probability density function $\phi(x)$.
 * @param x Input scalar.
 * @return $\frac{1}{\sqrt{2\pi}} e^{-x^2/2}$.
 */
inline double standard_normal_pdf(double x) {
    constexpr double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

/**
 * @brief Standard normal cumulative distribution function $\Phi(x)$.
 * @param x Input scalar.
 * @return $\frac{1}{2} \text{erfc}(-x / \sqrt{2})$.
 */
inline double standard_normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}



double log_standard_normal_cdf(double x) {
    if (x > 5.0) {
        return -standard_normal_cdf(-x);
    }
    if (x > -10.0) {
        return std::log(standard_normal_cdf(x));
    }
    double x2 = x * x;
    return -0.5 * x2 - 0.5 * std::log(2.0 * M_PI) - std::log(-x)
           + std::log(1.0 - 1.0/x2 + 3.0/(x2*x2) - 15.0/(x2*x2*x2));
}

/**
 * @brief Numerically stable logistic sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$.
 * @param x Input scalar.
 * @return Sigmoid function value in $(0, 1)$.
 */
inline double sigmoid_scalar(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

/**
 * @brief Numerically stable evaluation of $\ln(1 + e^x)$.
 * @param x Input scalar.
 * @return $\ln(1 + e^x)$ avoiding overflow for large positive or negative $x$.
 */
inline double log1p_exp_stable(double x) {
    return (x > 0.0) ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

/**
 * @brief Digamma function $\psi(x) = \frac{d}{dx}\ln\Gamma(x)$ by upward recurrence and asymptotic series.
 * @details Shifts $x$ to at least 7 with $\psi(x) = \psi(x+1) - 1/x$, then uses
 *          $\ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)$ (absolute error below $10^{-9}$).
 * @param x Evaluation scalar $x > 0$.
 * @return Digamma value.
 */
inline double digamma(double x) {
    double result = 0.0;
    while (x < 7.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double xx = 1.0 / x;
    double xx2 = xx * xx;
    double xx4 = xx2 * xx2;
    result += std::log(x) - 0.5 * xx - (1.0 / 12.0) * xx2 + (1.0 / 120.0) * xx4 - (1.0 / 252.0) * (xx4 * xx2);
    return result;
}

/**
 * @brief Asymptotic approximation of Trigamma function $\psi^{(1)}(x) = \frac{d^2}{dx^2}\ln\Gamma(x)$.
 * @param x Evaluation scalar $x > 0$.
 * @return Trigamma value.
 */
inline double trigamma(double x) {
    double result = 0.0;
    while (x < 7.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double inv_x4 = inv_x2 * inv_x2;
    result += inv_x + 0.5 * inv_x2 + (1.0 / 6.0) * (inv_x2 * inv_x) - (1.0 / 30.0) * (inv_x4 * inv_x) + (1.0 / 42.0) * (inv_x4 * inv_x2 * inv_x);
    return result;
}

/**
 * @brief Standard normal cumulative distribution function wrapper.
 * @param z Quantile value $z$.
 * @return $\Phi(z)$.
 */
inline double normal_cdf_scalar(double z) {
    return 0.5 * (1.0 + std::erf(z * glm_SQRT_1_2));
}

/**
 * @brief Numerically stable log survival function $\ln(1 - \Phi(z))$ for standard normal.
 * @param z Quantile value $z$.
 * @return $\ln(1 - \Phi(z))$.
 */
inline double log_survival_normal(double z) {
    if (z > 5.0) {
        const double z2 = z * z;
        return -glm_LOG_SQRT_2PI - 0.5 * z2 - std::log(z)
               + std::log(1.0 - 1.0 / z2 + 3.0 / (z2 * z2));
    }
    if (z < -5.0) {
        return std::log1p(-0.5 * std::erfc(-z * glm_SQRT_1_2));
    }
    const double s = 0.5 * std::erfc(z * glm_SQRT_1_2);
    return std::log(std::max(1e-300, s));
}

/**
 * @brief Inverse Mills Ratio $\lambda(z) = \frac{\phi(z)}{1 - \Phi(z)}$.
 * @param z Standardized score value $z$.
 * @return Inverse Mills Ratio scalar.
 */
inline double inverse_mills_ratio(double z) {
    if (z > 5.0) {
        const double z2 = z * z;
        return z / (1.0 - 1.0 / z2 + 3.0 / (z2 * z2));
    }
    if (z < -5.0) {
        return std::exp(-glm_LOG_SQRT_2PI - 0.5 * z * z);
    }
    const double phi = std::exp(-glm_LOG_SQRT_2PI - 0.5 * z * z);
    const double Phi_comp = 0.5 * std::erfc(z * glm_SQRT_1_2);
    return phi / std::max(1e-300, Phi_comp);
}

/**
 * @brief Derivative of Inverse Mills Ratio $\lambda'(z) = \lambda(z)(\lambda(z) - z)$.
 * @param z Score value $z$.
 * @param imr Pre-computed Inverse Mills Ratio $\lambda(z)$.
 * @return Derivative $\lambda'(z)$.
 */
inline double d_inverse_mills_ratio(double z, double imr) {
    return imr * (imr - z);
}

/**
 * @brief Standard Mills Ratio $R(x) = \frac{\phi(x)}{\Phi(x)}$.
 * @param x Input scalar.
 * @return Mills ratio value.
 */
inline double mills_ratio(double x) {
    if (x < -7.0) {
        const double x2 = x * x;
        return std::abs(x) / (1.0 - 1.0 / x2 + 3.0 / (x2 * x2));
    }
    const double phi_x = glm_INV_SQRT_2PI * std::exp(-0.5 * x * x);
    const double Phi_x = 0.5 * std::erfc(-x * glm_SQRT_1_2);
    if (Phi_x < 1e-30) return std::abs(x);
    return phi_x / Phi_x;
}

/**
 * @brief Smooth positive Inverse link function evaluation with second derivatives.
 * @details Uses hyperbolic softplus floor to maintain $C^\infty$ smoothness.
 * @param eta Linear predictor value $\eta$.
 * @param val Output inverse link value $\mu(\eta)$.
 * @param dval_deta Output first derivative $d\mu/d\eta$.
 * @param d2val_deta2 Output second derivative $d^2\mu/d\eta^2$.
 */
inline void evaluate_positive_inverse_d2(double eta, double& val, double& dval_deta, double& d2val_deta2) {
    const double d = 0.1;
    const double sq = std::sqrt(eta * eta + 4.0 * d * d);
    const double eta_pos = 0.5 * (eta + sq);
    const double deta_pos = 0.5 * (1.0 + eta / sq);
    const double d2eta_pos = 2.0 * d * d / (sq * sq * sq);
    val = 1.0 / eta_pos;
    dval_deta = -deta_pos / (eta_pos * eta_pos);
    d2val_deta2 = (-d2eta_pos * eta_pos + 2.0 * deta_pos * deta_pos)
                  / (eta_pos * eta_pos * eta_pos);
}

// ---------------------------------------------------------------------------
// LINK FUNCTION EVALUATION ROUTINES
// ---------------------------------------------------------------------------

/**
 * @brief Evaluates inverse link function $\mu = g^{-1}(\eta)$, first derivative $d\mu/d\eta$, and second derivative $d^2\mu/d\eta^2$.
 * @param eta Linear predictor value $\eta$.
 * @param link Link type enum.
 * @param val Output value $\mu$.
 * @param dval_deta Output derivative $d\mu/d\eta$.
 * @param d2val_deta2 Output second derivative $d^2\mu/d\eta^2$.
 */
inline void evaluate_link_d2(double eta, LinkType link,
                             double& val, double& dval_deta, double& d2val_deta2) {
    switch (link) {
        case LinkType::Identity: {
            val = eta;
            dval_deta = 1.0;
            d2val_deta2 = 0.0;
            break;
        }
        case LinkType::Log: {
            val = std::exp(std::min(eta, 23.0));
            dval_deta = val;
            d2val_deta2 = val;
            break;
        }
        case LinkType::Sqrt: {
            val = eta * eta;
            dval_deta = 2.0 * eta;
            d2val_deta2 = 2.0;
            break;
        }
        case LinkType::Logit: {
            val = sigmoid_scalar(eta);
            dval_deta = val * (1.0 - val);
            d2val_deta2 = dval_deta * (1.0 - 2.0 * val);
            break;
        }
        case LinkType::Probit: {
            const double c = std::max(-8.0, std::min(8.0, eta));
            val = 0.5 * (1.0 + std::erf(c * glm_SQRT_1_2));
            dval_deta = glm_INV_SQRT_2PI * std::exp(-0.5 * c * c);
            d2val_deta2 = -c * dval_deta;
            break;
        }
        case LinkType::Cloglog: {
            const double e = std::exp(std::max(-40.0, std::min(eta, 10.0)));
            val = 1.0 - std::exp(-e);
            dval_deta = e * (1.0 - val);
            d2val_deta2 = dval_deta * (1.0 - e);
            break;
        }
        case LinkType::Inverse: {
            if (std::abs(eta) < 1e-5) eta = (eta >= 0.0) ? 1e-5 : -1e-5;
            val = 1.0 / eta;
            dval_deta = -1.0 / (eta * eta);
            d2val_deta2 = 2.0 / (eta * eta * eta);
            break;
        }
        case LinkType::Loglog: {
            const double em = std::exp(std::max(-40.0, std::min(-eta, 10.0)));
            val = std::exp(-em);
            dval_deta = em * val;
            d2val_deta2 = dval_deta * (em - 1.0);
            break;
        }
        case LinkType::Cauchit: {
            val = 0.5 + std::atan(eta) / glm_LINK_PI;
            const double den = 1.0 + eta * eta;
            dval_deta = 1.0 / (glm_LINK_PI * den);
            d2val_deta2 = -2.0 * eta / (glm_LINK_PI * den * den);
            break;
        }
        default: {
            val = eta;
            dval_deta = 1.0;
            d2val_deta2 = 0.0;
            break;
        }
    }
}

/**
 * @brief Evaluates inverse link function $\mu = g^{-1}(\eta)$ and first derivative $d\mu/d\eta$.
 * @param eta Linear predictor value $\eta$.
 * @param link Link type enum.
 * @param val Output value $\mu$.
 * @param dval_deta Output derivative $d\mu/d\eta$.
 */
inline void evaluate_link(double eta, LinkType link, double& val, double& dval_deta) {
    double d2 = 0.0;
    evaluate_link_d2(eta, link, val, dval_deta, d2);
}

/**
 * @brief Evaluates inverse link function $\mu = g^{-1}(\eta)$.
 * @param eta Linear predictor value $\eta$.
 * @param link Link type enum.
 * @param val Output value $\mu$.
 */
inline void evaluate_link(double eta, LinkType link, double& val) {
    double d1 = 0.0;
    evaluate_link(eta, link, val, d1);
}

/**
 * @brief Evaluates link function for count response model with positive lower bound ($10^{-15}$).
 */
inline void evaluate_count_mean_d2(double eta, LinkType link,
                                   double& val, double& dval_deta, double& d2val_deta2) {
    evaluate_link_d2(eta, link, val, dval_deta, d2val_deta2);
    if (val < 1e-15) val = 1e-15;
}

/**
 * @brief Evaluates count link value and derivative with positive threshold safeguards.
 */
inline void evaluate_count_link(double eta, LinkType link, double& val, double& dval_deta) {
    double d2 = 0.0;
    evaluate_count_mean_d2(eta, link, val, dval_deta, d2);
}

/**
 * @brief Evaluates count link value, first, and second derivatives with positive threshold safeguards.
 */
inline void evaluate_count_link_d2(double eta, LinkType link,
                                   double& val, double& dval_deta, double& d2val_deta2) {
    evaluate_count_mean_d2(eta, link, val, dval_deta, d2val_deta2);
}

/**
 * @brief Evaluates logistic inflation link for zero-inflation components.
 */
inline void evaluate_inflation_link(double eta, double& pi, double& dpi_deta) {
    evaluate_link(eta, LinkType::Logit, pi, dpi_deta);
}

/** @brief Vectorized inverse link evaluation. */
inline Eigen::VectorXd evaluate_inverse_link(const Eigen::VectorXd& eta, LinkType link) {
    Eigen::VectorXd out(eta.size());
    for (Eigen::Index i = 0; i < eta.size(); ++i) {
        double v = 0.0, d1 = 0.0, d2 = 0.0;
        evaluate_link_d2(eta(i), link, v, d1, d2);
        out(i) = v;
    }
    return out;
}

/** @brief Vectorized first derivative of inverse link. */
inline Eigen::VectorXd evaluate_d_inverse_link(const Eigen::VectorXd& eta, LinkType link) {
    Eigen::VectorXd out(eta.size());
    for (Eigen::Index i = 0; i < eta.size(); ++i) {
        double v = 0.0, d1 = 0.0, d2 = 0.0;
        evaluate_link_d2(eta(i), link, v, d1, d2);
        out(i) = d1;
    }
    return out;
}

/** @brief Vectorized second derivative of inverse link. */
inline Eigen::VectorXd evaluate_d2_inverse_link(const Eigen::VectorXd& eta, LinkType link) {
    Eigen::VectorXd out(eta.size());
    for (Eigen::Index i = 0; i < eta.size(); ++i) {
        double v = 0.0, d1 = 0.0, d2 = 0.0;
        evaluate_link_d2(eta(i), link, v, d1, d2);
        out(i) = d2;
    }
    return out;
}

/** @brief Scalar second derivative of inverse link. */
inline double evaluate_d2_inverse_link_scalar(double eta, LinkType link) {
    double v = 0.0, d1 = 0.0, d2 = 0.0;
    evaluate_link_d2(eta, link, v, d1, d2);
    return d2;
}

/**
 * @brief Helper structure for bounded Beta regression link evaluations in $(0, 1)$.
 */
struct BetaMath {
    static inline double digamma(double x) { return ::glm::digamma(x); }
    static inline double trigamma(double x) { return ::glm::trigamma(x); }

    static inline void evaluate_link(double eta, LinkType link, double& mu, double& dmu_deta) {
        double d2 = 0.0;
        evaluate_link_hessian(eta, link, mu, dmu_deta, d2);
    }

    static inline void evaluate_link_hessian(double eta, LinkType link,
                                             double& mu, double& dmu_deta, double& d2mu_deta2) {
        switch (link) {
            case LinkType::Logit: {
                eta = std::max(-30.0, std::min(30.0, eta));
                break;
            }
            case LinkType::Probit: {
                eta = std::max(-5.0, std::min(5.0, eta));
                break;
            }
            case LinkType::Cloglog: {
                eta = std::max(-30.0, std::min(10.0, eta));
                break;
            }
            case LinkType::Log: {
                eta = std::max(-30.0, std::min(20.0, eta));
                break;
            }
            case LinkType::Loglog: {
                //  Mirror image of Cloglog: mu = exp(-exp(-eta)).
                eta = std::max(-10.0, std::min(30.0, eta));
                break;
            }
            case LinkType::Cauchit: {
                //  Heavy-tailed: reaching mu ~ 0/1 needs very large |eta|, so the usual
                //  +-30 clamp would cap mu at ~[0.011, 0.989]. Only guard against overflow.
                eta = std::max(-1e8, std::min(1e8, eta));
                break;
            }
            default:
                eta = std::max(-30.0, std::min(30.0, eta));
                if (link != LinkType::Inverse) link = LinkType::Logit;
                break;
        }
        ::glm::evaluate_link_d2(eta, link, mu, dmu_deta, d2mu_deta2);
        if (link == LinkType::Logit || link == LinkType::Probit || link == LinkType::Cloglog
            || link == LinkType::Loglog || link == LinkType::Cauchit) {
            mu = std::max(1e-14, std::min(1.0 - 1e-14, mu));
            if (link == LinkType::Logit) {
                dmu_deta = mu * (1.0 - mu);
                d2mu_deta2 = dmu_deta * (1.0 - 2.0 * mu);
            }
        }
    }

    /**
     * @brief Link evaluation for a strictly positive parameter (precision phi, dispersion sigma^2).
     * @details `evaluate_link_hessian` is designed for parameters in (0,1) and silently replaces
     *          unknown links (Identity, Sqrt) with Logit, which is wrong for phi > 0. This routine
     *          evaluates the links valid for a positive parameter:
     *          - Log:      delegated to `evaluate_link_hessian` (identical clamps as before).
     *          - Identity/Sqrt: evaluated directly.
     *          - Inverse:  smooth positive inverse (`evaluate_positive_inverse_d2`), so that phi = 1/eta_pos > 0.
     */
    static inline void evaluate_precision_link_d2(double eta, LinkType link,
                                                  double& phi, double& dphi_deta, double& d2phi_deta2) {
        switch (link) {
            case LinkType::Identity:
            case LinkType::Sqrt:
                eta = std::max(-1e6, std::min(1e6, eta));
                ::glm::evaluate_link_d2(eta, link, phi, dphi_deta, d2phi_deta2);
                break;
            case LinkType::Inverse:
                ::glm::evaluate_positive_inverse_d2(eta, phi, dphi_deta, d2phi_deta2);
                break;
            default:
                evaluate_link_hessian(eta, link, phi, dphi_deta, d2phi_deta2);
                break;
        }
    }

    /** @brief First-derivative-only variant of `evaluate_precision_link_d2`. */
    static inline void evaluate_precision_link(double eta, LinkType link, double& phi, double& dphi_deta) {
        double d2 = 0.0;
        evaluate_precision_link_d2(eta, link, phi, dphi_deta, d2);
    }
};

inline double apply_link(double value, LinkType link) {
    switch (link) {
        case LinkType::Log:
            return std::log(value);
        case LinkType::Inverse:
            return 1.0 / value;
        case LinkType::Identity:
        default:
            return value;
    }
}

inline double apply_inverse_link(double eta, LinkType link) {
    switch (link) {
        case LinkType::Log:     return std::exp(eta);
        case LinkType::Inverse: return 1.0 / eta;
        case LinkType::Identity:
        default:                return eta;
    }
}

inline double apply_inverse_link_deriv(double eta, LinkType link) {
    switch (link) {
        case LinkType::Log:
            return std::exp(eta);
        case LinkType::Inverse: {
            double inv = 1.0 / eta;
            return -inv * inv;
        }
        case LinkType::Identity:
        default:
            return 1.0;
    }
}

/**
 * @brief Generic Weighted IRLS solver using Fisher Scoring.
 * @details Solves for coefficients $\beta$ given generic target vector, link function, and variance callable `var_fn`.
 * @tparam VarianceFn Callable type returning variance $V(\mu)$ given mean $\mu$.
 * @param X Design matrix ($n \times p$).
 * @param target Response target vector ($n \times 1$).
 * @param prior_weight Prior weights vector ($n \times 1$).
 * @param link Link function type.
 * @param var_fn Variance function callable.
 * @param coef Initial coefficient vector $\beta$, updated in-place.
 * @param max_iter Maximum Fisher Scoring iterations.
 * @param tol Relative parameter change tolerance.
 */
template <typename VarianceFn>
inline void irls_glm(const Eigen::MatrixXd& X, const Eigen::VectorXd& target,
                      const Eigen::VectorXd& prior_weight, LinkType link,
                      VarianceFn var_fn, Eigen::VectorXd& coef,
                      int max_iter = 25, double tol = 1e-8) {
    const long n = X.rows();
    Eigen::VectorXd eta = X * coef;
    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd w(n), z(n);
        for (long i = 0; i < n; ++i) {
            double mu = 0.0, dmu = 0.0;
            evaluate_link(eta(i), link, mu, dmu);
            mu = std::max(1e-10, mu);
            if (std::abs(dmu) < 1e-10) dmu = (dmu >= 0.0) ? 1e-10 : -1e-10;
            const double V = std::max(1e-10, var_fn(mu));
            w(i) = prior_weight(i) * dmu * dmu / V;
            z(i) = eta(i) + (target(i) - mu) / dmu;
        }
        const Eigen::VectorXd wz = w.cwiseProduct(z);
        const Eigen::MatrixXd XtWX = X.transpose() * w.asDiagonal() * X;
        Eigen::VectorXd coef_new = XtWX.ldlt().solve(X.transpose() * wz);
        if (!coef_new.array().isFinite().all()) break;

        const Eigen::VectorXd eta_new = X * coef_new;
        const double delta = (eta_new - eta).norm() / std::max(1e-8, eta.norm());
        coef = coef_new;
        eta = eta_new;
        if (delta < tol) break;
    }
}

} // namespace glm

#endif // _GLMFIT_MLE_UTILS_HPP
