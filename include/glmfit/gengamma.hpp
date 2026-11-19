// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file gengamma.hpp
 * @brief Generalized Gamma distribution probability functions (density and CDF).
 * @details Implements the Prentice (1974) reparameterization of the 3-parameter
 *          Generalized Gamma distribution ($\mu, \sigma, Q$) and its limiting
 *          Log-Normal behavior as $Q \to 0$.
 */

#ifndef _GLMFIT_GENGAMMA_H
#define _GLMFIT_GENGAMMA_H

#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>

#include <boost/math/special_functions/gamma.hpp>
#include <Eigen/Dense>

#include "distribution.hpp"

namespace gengamma {

/**
 * @brief Standard NaN constant for invalid numerical operations.
 */
const double NA_REAL_CPP = std::numeric_limits<double>::quiet_NaN();

/**
 * @brief Standard negative infinity constant.
 */
const double R_NEGINF_CPP = -std::numeric_limits<double>::infinity();

/**
 * @brief Checks parameter domain validity for Generalized Gamma distribution parameters.
 * @param mu Location parameter $\mu \in (-\infty, \infty)$.
 * @param sigma Scale parameter $\sigma > 0$.
 * @param Q Shape parameter $Q \in (-\infty, \infty)$.
 * @return True if any parameter is non-finite or $\sigma \le 0$; false otherwise.
 */
inline bool bad(const double mu,
                const double sigma,
                const double Q) {
    if (sigma <= 0.0 || !std::isfinite(sigma) || !std::isfinite(mu) || !std::isfinite(Q)) {
        return true;
    }
    return false;
}

/**
 * @brief Evaluates the log-density of the 3-parameter Generalized Gamma distribution.
 * @details Uses the exact Prentice (1974) parameterization matching the R package `flexsurv`.
 *          For $|Q| < 10^{-4}$, the evaluation smoothly transitions to the limiting Log-Normal density.
 */
class density {
public:
    typedef double result_type;

    /**
     * @brief Computes the log probability density function value at $x$.
     * @param x Observation value $x > 0$.
     * @param mu Location parameter $\mu$.
     * @param sigma Scale parameter $\sigma > 0$.
     * @param Q Shape parameter $Q$.
     * @return Log-density value as a double, or NA_REAL_CPP if parameters are invalid.
     */
    inline double operator()(const double x,
                             const double mu,
                             const double sigma,
                             const double Q) const {
        if (bad(mu, sigma, Q)) return NA_REAL_CPP;
        if (x <= 0.0) return R_NEGINF_CPP;

        const double abs_q = std::abs(Q);

        // When Q is near zero, evaluate standard Log-Normal limiting density
        if (abs_q < 1e-4) {
            const double log_x = std::log(x);
            const double normalized = (log_x - mu) / sigma;
            const double log_sqrt_2pi = 0.91893853320467274178; // 0.5 * log(2 * pi)

            return -log_sqrt_2pi - std::log(sigma) - log_x - 0.5 * normalized * normalized;
        } else {
            // Prentice (1974) exact Generalized Gamma parameterization
            const double y = std::log(x);
            const double w = (y - mu) / sigma;
            const double qi = 1.0 / (Q * Q);
            const double qw = Q * w;
            const double exp_qw = std::exp(qw);

            return -std::log(sigma * x)
                   + std::log(abs_q)
                   + qi * std::log(qi)
                   + qi * (qw - exp_qw)
                   - boost::math::lgamma(qi);
        }
    }
};

/**
 * @brief Cumulative distribution function (CDF) for the Generalized Gamma distribution.
 * @details Evaluates regularized lower or upper incomplete gamma functions, with tail options
 *          and log-scale return values. Automatically handles limiting Log-Normal CDF via `std::erf`.
 */
class cdf {
public:
    typedef double result_type;

    /**
     * @brief Constructor for Generalized Gamma CDF functor.
     * @param lower_tail_ If true, calculates $P(X \le q)$; if false, calculates $P(X > q)$.
     * @param give_log_ If true, returns log of the cumulative probability.
     */
    cdf(bool lower_tail_, bool give_log_) :
        lower_tail(lower_tail_),
        give_log(give_log_) {}

    /**
     * @brief Evaluates the CDF value for quantile $q$.
     * @param q Quantile threshold value $q$.
     * @param mu Location parameter $\mu$.
     * @param sigma Scale parameter $\sigma > 0$.
     * @param big_q Shape parameter $Q$.
     * @return Cumulative probability value (or log-probability).
     */
    inline double operator()(const double q,
                             const double mu,
                             const double sigma,
                             const double big_q) const {
        if (bad(mu, sigma, big_q)) return NA_REAL_CPP;
        if (q <= 0.0) return below_distribution(lower_tail, give_log);

        const double abs_q = std::abs(big_q);

        if (abs_q > 1e-4) {
            const double y = std::log(q);
            const double w = (y - mu) / sigma;
            const double qq = 1.0 / (big_q * big_q);
            const double expnu = qq * std::exp(big_q * w);

            // Inversion rule according to the sign of Q
            bool target_lower_tail = (big_q > 0) ? lower_tail : !lower_tail;

            double p = 0.0;
            if (target_lower_tail) {
                p = boost::math::gamma_p(qq, expnu);
            } else {
                p = boost::math::gamma_q(qq, expnu);
            }

            p = std::clamp(p, 1e-15, 1.0 - 1e-15);
            return give_log ? std::log(p) : p;
        } else {
            const double log_q = std::log(q);
            const double z = (log_q - mu) / (sigma * std::sqrt(2.0));

            double p = 0.0;
            if (lower_tail) {
                p = 0.5 * (1.0 + std::erf(z));
            } else {
                p = 0.5 * std::erfc(z);
            }

            p = std::clamp(p, 1e-15, 1.0 - 1e-15);
            return give_log ? std::log(p) : p;
        }
    }

private:
    bool lower_tail;
    bool give_log;
};

} // namespace gengamma

#endif // _GLMFIT_GENGAMMA_H
