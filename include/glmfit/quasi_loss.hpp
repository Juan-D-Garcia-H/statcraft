// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file quasi_loss.hpp
 * @brief Vectorized Quasi-likelihood loss function implementation for QuasiSolver.
 * @details Defines link function types, variance function policies (Constant, Mu, MuSquared, MuCubed, MuPower/Tweedie, Custom),
 *          and vector/gradient/Hessian evaluations for Quasi-GLM models.
 */


#ifndef _GLMFIT_QUASI_LOSS_HPP
#define _GLMFIT_QUASI_LOSS_HPP

#include <Eigen/Core>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace glm {

namespace detail {

/**
 * @brief One Gauss-Kronrod G7/K15 panel on [a, b] (QUADPACK qk15 nodes/weights).
 * @param f Integrand callable double -> double.
 * @param err Output error estimate |K15 - G7|.
 * @return K15 estimate of the integral of f over [a, b] (signed: b < a gives the negated integral).
 */
template <typename F>
inline double gk15_panel(F&& f, double a, double b, double& err) {
    static const double xgk[8] = {
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245, 0.000000000000000000000000000000000};
    static const double wgk[8] = {
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static const double wg[4] = {
        0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
        0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

    const double c = 0.5 * (a + b);
    const double h = 0.5 * (b - a);

    const double fc = f(c);
    double resk = wgk[7] * fc;
    double resg = wg[3] * fc;
    for (int j = 0; j < 7; ++j) {
        const double dx = h * xgk[j];
        const double f1 = f(c - dx);
        const double f2 = f(c + dx);
        resk += wgk[j] * (f1 + f2);
        if (j % 2 == 1) resg += wg[j / 2] * (f1 + f2);
    }
    err = std::abs((resk - resg) * h);
    return resk * h;
}

/**
 * @brief Adaptive (bisection) Gauss-Kronrod integral of f over [a, b].
 * @details Converges to |error| <= max(abs_tol, rel_tol * |I|) or stops at `depth` bisections.
 */
template <typename F>
inline double integrate_gk15(F&& f, double a, double b, double abs_tol = 1e-13,
                             double rel_tol = 1e-10, int depth = 24) {
    double err = 0.0;
    const double val = gk15_panel(f, a, b, err);
    if (depth <= 0 || err <= std::max(abs_tol, rel_tol * std::abs(val))) return val;
    const double m = 0.5 * (a + b);
    return integrate_gk15(f, a, m, 0.5 * abs_tol, rel_tol, depth - 1)
         + integrate_gk15(f, m, b, 0.5 * abs_tol, rel_tol, depth - 1);
}

} // namespace detail

/**
 * @brief Link function enumeration for quasi-likelihood models.
 */
enum class QuasiLinkType { Identity, Log, Inverse, Logit, Probit };

/**
 * @brief Variance function enumeration for quasi-likelihood models.
 */
enum class QuasiVarianceType { Constant, Mu, MuMu, MuSquared, MuCubed, MuPower, Custom };

/**
 * @brief Default policy struct for $V(\mu) = \mu^p$ power variance functions (Tweedie family).
 * @details Uses $p = 1.5$ for default compound Poisson-Gamma distribution. C++17 compliant.
 */
struct DefaultMuPowerPolicy {
    static constexpr double p = 1.5;  ///< Power exponent $p$ ($1.0 < p < 2.0$)
};

/**
 * @brief Empty fallback policy struct for non-custom variance types.
 */
struct EmptyCustomPolicy {
    /** @brief Default constant variance fallback ($V(\mu) = 1.0$). */
    inline void variance(const Eigen::VectorXd&, Eigen::VectorXd& v_mu) const { v_mu.setConstant(1.0); }
    /** @brief Default deviance fallback ($0.0$). */
    inline double deviance(double, double) const { return 0.0; }
};

/**
 * @brief Custom user-defined variance policy wrapper.
 * @details Stores the user callback function by value to ensure thread safety during concurrent optimization calls.
 * @tparam VarFunc Callable type with signature `double(double mu)`.
 */
template <typename VarFunc>
struct CustomVariancePolicy {
    VarFunc fn; ///< Stored user-defined variance callback function

    /**
     * @brief Constructor for CustomVariancePolicy.
     * @param f Callable variance function $V(\mu)$.
     */
    explicit CustomVariancePolicy(VarFunc f) : fn(std::move(f)) {}

    /**
     * @brief Computes element-wise variance $V(\mu)$ for vector input.
     * @param mu Fitted mean vector ($\mu$).
     * @param v_mu Output variance vector ($V(\mu)$).
     */
    inline void variance(const Eigen::VectorXd& mu, Eigen::VectorXd& v_mu) const {
        const long n = mu.size();
        v_mu.resize(n);
        for (long i = 0; i < n; ++i) v_mu(i) = std::max(1e-6, fn(mu(i)));
    }

    /**
     * @brief Evaluates the quasi-deviance $d(y,\mu) = 2 \int_{\mu}^{y} \frac{y - u}{V(u)}\, du$ for arbitrary $V$.
     * @details Adaptive Gauss-Kronrod (G7/K15) to a relative tolerance of 1e-10. The previous 5-point
     *          Simpson rule was badly inaccurate whenever y and mu differ by a large factor (e.g. for
     *          V(u) = u^2, y = 10, mu = 1 it gave 18.0 instead of 13.39), which made the optimized
     *          objective disagree with its (exact) analytic gradient.
     * @param yi Observed response $y_i$.
     * @param mu Fitted mean value $\mu_i$.
     * @return Quasi-deviance value.
     */
    inline double deviance(double yi, double mu) const {
        if (std::abs(yi - mu) < 1e-7) return 0.0;

        auto integrando = [this, yi](double u) {
            return (yi - u) / std::max(1e-6, fn(u));
        };

        return 2.0 * detail::integrate_gk15(integrando, mu, yi);
    }
};

/**
 * @brief Vectorized Quasi-likelihood loss function structure.
 * @tparam Link Selected QuasiLinkType enum value.
 * @tparam Var Selected QuasiVarianceType enum value.
 * @tparam Policy Variance policy type (CustomVariancePolicy, DefaultMuPowerPolicy, or EmptyCustomPolicy).
 */
template <
    QuasiLinkType Link,
    QuasiVarianceType Var,
    typename Policy = std::conditional_t<Var == QuasiVarianceType::Custom, EmptyCustomPolicy, DefaultMuPowerPolicy>
>
struct QuasiLoss {
    Eigen::MatrixXd X;  ///< Design matrix ($n \times p$)
    Eigen::VectorXd y;  ///< Response vector ($n \times 1$)
    Eigen::VectorXd w;  ///< Observation weights vector ($n \times 1$)
    Policy policy;      ///< Variance function policy instance
    Eigen::VectorXd offset;  ///< Optional linear-predictor offset ($\eta = X\beta + \text{offset}$); empty = none

    /**
     * @brief Constructs a new QuasiLoss object.
     * @param exog Design matrix ($X$).
     * @param endog Response vector ($y$).
     * @param weights Prior weights vector ($w$).
     * @param policy_in Policy instance (default initialized).
     */
    QuasiLoss(Eigen::MatrixXd exog, Eigen::VectorXd endog, Eigen::VectorXd weights, Policy policy_in = Policy{})
        : X(std::move(exog)), y(std::move(endog)), w(std::move(weights)), policy(std::move(policy_in)) {}

    /** @brief Sets the linear-predictor offset (size must equal nrow(X), or empty for none). */
    void set_offset(Eigen::VectorXd off) { offset = std::move(off); }

    /** @brief $\eta = X\beta + \text{offset}$. */
    Eigen::VectorXd linear_predictor(const Eigen::VectorXd& beta) const {
        Eigen::VectorXd eta = X * beta;
        if (offset.size() == eta.size()) eta += offset;
        return eta;
    }

private:
    void compute_link(const Eigen::VectorXd& eta, Eigen::VectorXd& mu, Eigen::VectorXd& dmu_deta) const {
        const long n = eta.size();
        mu.resize(n);
        dmu_deta.resize(n);

        if constexpr (Link == QuasiLinkType::Identity) {
            mu = eta;
            dmu_deta.setConstant(1.0);
        }
        else if constexpr (Link == QuasiLinkType::Log) {
            Eigen::VectorXd eta_clamped = eta.array().cwiseMin(30.0);
            mu = eta_clamped.array().exp();
            dmu_deta = mu;
        }
        else if constexpr (Link == QuasiLinkType::Inverse) {
            const double d = 0.1;
            Eigen::ArrayXd disc = (eta.array().square() + 4.0 * d * d).sqrt();
            Eigen::ArrayXd eta_pos = 0.5 * (eta.array() + disc);
            Eigen::ArrayXd deta_pos_deta = 0.5 * (1.0 + eta.array() / disc);

            mu = eta_pos.inverse().matrix();
            dmu_deta = (-deta_pos_deta / eta_pos.square()).matrix();
        }
        else if constexpr (Link == QuasiLinkType::Logit) {
            Eigen::VectorXd clamped = eta.array().cwiseMax(-30.0).cwiseMin(30.0);
            mu = 1.0 / (1.0 + (-clamped).array().exp());
            dmu_deta = mu.array() * (1.0 - mu.array());
        }
        else if constexpr (Link == QuasiLinkType::Probit) {
            mu = eta.unaryExpr([](double val) {
                return 0.5 * (1.0 + std::erf(val * 0.7071067811865475));
            });
            dmu_deta = (-0.5 * eta.array().square()).array().exp() * 0.3989422804014327;
            dmu_deta = dmu_deta.array().cwiseMax(1e-10);
        }
    }

    void compute_variance(const Eigen::VectorXd& mu, Eigen::VectorXd& v_mu) const {
        const long n = mu.size();
        v_mu.resize(n);
        constexpr double eps = 1e-10;

        if constexpr (Var == QuasiVarianceType::Custom) {
            policy.variance(mu, v_mu);
        } else {
            if constexpr (Var == QuasiVarianceType::Constant) {
                v_mu.setConstant(1.0);
            } else if constexpr (Var == QuasiVarianceType::Mu) {
                v_mu = mu.array().abs().cwiseMax(eps);
            } else if constexpr (Var == QuasiVarianceType::MuMu) {
                Eigen::VectorXd c = mu.array().cwiseMax(eps).cwiseMin(1.0 - eps);
                v_mu = c.array() * (1.0 - c.array());
            } else if constexpr (Var == QuasiVarianceType::MuSquared) {
                v_mu = mu.array().square().cwiseMax(eps);
            } else if constexpr (Var == QuasiVarianceType::MuCubed) {
                v_mu = mu.array().abs().cube().cwiseMax(eps);
            } else if constexpr (Var == QuasiVarianceType::MuPower) {
                constexpr double p_val = Policy::p;
                v_mu = mu.array().abs().pow(p_val).cwiseMax(eps);
            }
        }

        auto pos_eps = Eigen::VectorXd::Constant(n, 1e-12);
        auto neg_eps = Eigen::VectorXd::Constant(n, -1e-12);

        v_mu = (v_mu.array().abs() < 1e-12).select(
            (v_mu.array() >= 0).select(pos_eps, neg_eps), v_mu
        );
    }

    inline double eval_deviance_scalar(double yi, double mu) const {
        constexpr double eps = 1e-10;
        if constexpr (Var == QuasiVarianceType::Custom) {
            return policy.deviance(yi, mu);
        } else {
            if constexpr (Var == QuasiVarianceType::Constant) {
                return (yi - mu) * (yi - mu);
            } else if constexpr (Var == QuasiVarianceType::Mu) {
                double mu_c = std::max(eps, std::abs(mu));
                if (yi > 0.0) return 2.0 * (yi * std::log(yi / mu_c) - (yi - mu_c));
                return 2.0 * mu_c;
            } else if constexpr (Var == QuasiVarianceType::MuMu) {
                double mu_c = std::max(eps, std::min(1.0 - eps, mu));
                if (yi <= 0.0) return 2.0 * std::log(1.0 / (1.0 - mu_c));
                if (yi >= 1.0) return 2.0 * std::log(1.0 / mu_c);
                return 2.0 * (yi * std::log(yi / mu_c) + (1.0 - yi) * std::log((1.0 - yi) / (1.0 - mu_c)));
            } else if constexpr (Var == QuasiVarianceType::MuSquared) {
                double mu_c = std::max(eps, std::abs(mu));
                double yi_c = std::max(eps, yi);
                return 2.0 * ((yi_c - mu_c) / mu_c - std::log(yi_c / mu_c));
            } else if constexpr (Var == QuasiVarianceType::MuCubed) {
                double mu_c = std::max(eps, std::abs(mu));
                double yi_c = std::max(eps, yi);
                return (yi_c - mu_c) * (yi_c - mu_c) / (yi_c * mu_c * mu_c);
            } else if constexpr (Var == QuasiVarianceType::MuPower) {
                constexpr double p = Policy::p;
                double mu_c = std::max(eps, std::abs(mu));
                double yi_c = std::max(0.0, yi);

                double term1 = 0.0;
                if (yi_c > 0.0) {
                    term1 = std::pow(yi_c, 2.0 - p) / ((1.0 - p) * (2.0 - p));
                }
                double term2 = (yi_c * std::pow(mu_c, 1.0 - p)) / (1.0 - p);
                double term3 = std::pow(mu_c, 2.0 - p) / (2.0 - p);

                return 2.0 * (term1 - term2 + term3);
            }
        }
        return 0.0;
    }

public:
    /**
     * @brief Computes per-observation IRLS information weights $(d\mu/d\eta)^2 / V(\mu)$ at a given $\beta$.
     * @details Used AFTER optimization converges to build the asymptotic covariance matrix
     *          $\hat\phi (X^T W X)^{-1}$ (L-BFGS-B/BFGS/NelderMead do not build a Hessian
     *          themselves, unlike the IRLS engine). Reuses the exact same `compute_link`/
     *          `compute_variance` used during optimization -- same analytic derivative,
     *          same variance formula, for every `Var` (including `Custom`), no duplicated logic.
     * @param beta Coefficient vector at which to evaluate the weights (typically $\hat\beta$).
     * @return Per-observation weight vector ($n \times 1$).
     */
    Eigen::VectorXd irls_weights(const Eigen::VectorXd& beta) const {
        Eigen::VectorXd eta = linear_predictor(beta);
        Eigen::VectorXd mu, dmu_deta, v_mu;
        compute_link(eta, mu, dmu_deta);
        compute_variance(mu, v_mu);
        return (dmu_deta.array().square() / v_mu.array()).matrix();
    }
    /**
     * @brief Evaluates overall quasi-likelihood loss value $f(\beta) = 0.5 \sum w_i d(y_i, \mu_i)$.
     * @param beta Coefficient vector $\beta$.
     * @return Half total weighted deviance scalar value.
     */
    double operator()(const Eigen::VectorXd& beta) const {
        const long n = X.rows();
        Eigen::VectorXd eta = linear_predictor(beta);
        Eigen::VectorXd mu, dmu_deta;
        compute_link(eta, mu, dmu_deta);

        double total = 0.0;
        for (long i = 0; i < n; ++i) {
            total += w(i) * eval_deviance_scalar(y(i), mu(i));
        }
        return 0.5 * total;
    }

    /**
     * @brief Computes quasi-likelihood gradient vector $g(\beta) = -X^T \left( W \frac{y - \mu}{V(\mu)} \frac{d\mu}{d\eta} \right)$.
     * @param beta Coefficient vector $\beta$.
     * @return Gradient vector ($p \times 1$).
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& beta) const {
        Eigen::VectorXd eta = linear_predictor(beta);

        Eigen::VectorXd mu, dmu_deta, v_mu;
        compute_link(eta, mu, dmu_deta);
        compute_variance(mu, v_mu);

        Eigen::VectorXd score_weights = -w.array() * (y.array() - mu.array()) * dmu_deta.array() / v_mu.array();
        return X.transpose() * score_weights;
    }

    /**
     * @brief Jointly computes Quasi-likelihood loss value and gradient vector.
     * @param beta Coefficient vector $\beta$.
     * @param grad Output gradient vector updated in-place.
     * @return Computed loss value.
     */
    double valueAndGradient(const Eigen::VectorXd& beta, Eigen::VectorXd& grad) const {
        const long n = X.rows();
        Eigen::VectorXd eta = linear_predictor(beta);

        Eigen::VectorXd mu, dmu_deta, v_mu;
        compute_link(eta, mu, dmu_deta);
        compute_variance(mu, v_mu);

        Eigen::VectorXd score_weights = -w.array() * (y.array() - mu.array()) * dmu_deta.array() / v_mu.array();
        grad = X.transpose() * score_weights;

        double total = 0.0;
        for (long i = 0; i < n; ++i) {
            total += w(i) * eval_deviance_scalar(y(i), mu(i));
        }
        return 0.5 * total;
    }
};

} // namespace glm

#endif // _GLMFIT_QUASI_LOSS_HPP
