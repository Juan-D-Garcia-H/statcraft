// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file links.hpp
/// @brief Inverse link functions and mean structure, with the numerical guards of R's `make.link`.
#pragma once
#include "types.hpp"
#include <cmath>
#include <limits>

namespace mcglm {
namespace detail {
constexpr double kEps = std::numeric_limits<double>::epsilon();
inline double pnorm_(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
inline double dnorm_(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI); }
/// Cauchy CDF, evaluated through atan(1/x) in the tails for accuracy.
inline double pcauchy_(double x) {
    if (std::fabs(x) > 1.0) { double y = std::atan(1.0 / x) / M_PI; return x > 0 ? 1.0 - y : -y; }
    return 0.5 + std::atan(x) / M_PI;
}
inline double dcauchy_(double x) { return 1.0 / (M_PI * (1.0 + x * x)); }
} // namespace detail

/// Mean structure of one response.
struct MeanOut {
    VectorXd mu;     ///< mean g^{-1}(eta)
    VectorXd dmu;    ///< derivative d mu / d eta
    MatrixXd D;      ///< n x p matrix diag(dmu) X, the derivative of mu with respect to beta
};

/// Evaluates the inverse link and its derivative.
///
/// Probabilities (logit, probit, cauchit, cloglog) are clamped away from 0 and 1 and derivatives are
/// bounded below by machine epsilon, exactly as in R's `make.link`.
///
/// @param link  link function
/// @param eta   linear predictor
/// @param[out] mu   g^{-1}(eta)
/// @param[out] dmu  d g^{-1} / d eta
inline void link_eval(Link link, const VectorXd& eta, VectorXd& mu, VectorXd& dmu) {
    using namespace detail;
    const Eigen::Index n = eta.size();
    mu.resize(n); dmu.resize(n);
    static const double thr_norm = 8.125890664701906;              // -qnorm(machine epsilon)
    static const double thr_cauchy = 1.0 / std::tan(M_PI * kEps);  // -qcauchy(machine epsilon)
    for (Eigen::Index i = 0; i < n; ++i) {
        double e = eta[i];
        switch (link) {
        case Link::Logit: {
            double t = e < -30 ? kEps : (e > 30 ? 1.0 / kEps : std::exp(e));
            mu[i] = t / (1.0 + t); dmu[i] = mu[i] * (1.0 - mu[i]); break; }
        case Link::Probit: {
            double ee = std::fmin(std::fmax(e, -thr_norm), thr_norm);
            mu[i] = pnorm_(ee); dmu[i] = std::fmax(dnorm_(e), kEps); break; }
        case Link::Cauchit: {
            double ee = std::fmin(std::fmax(e, -thr_cauchy), thr_cauchy);
            mu[i] = pcauchy_(ee); dmu[i] = std::fmax(dcauchy_(e), kEps); break; }
        case Link::Cloglog: {
            mu[i] = std::fmax(std::fmin(-std::expm1(-std::exp(e)), 1.0 - kEps), kEps);
            double ep = std::fmin(e, 700.0);
            dmu[i] = std::fmax(std::exp(ep) * std::exp(-std::exp(ep)), kEps); break; }
        case Link::Loglog:
            mu[i] = std::exp(-std::exp(-e)); dmu[i] = std::exp(-std::exp(-e) - e); break;
        case Link::Identity: mu[i] = e; dmu[i] = 1.0; break;
        case Link::Log: mu[i] = std::fmax(std::exp(e), kEps); dmu[i] = mu[i]; break;
        case Link::Sqrt: mu[i] = e * e; dmu[i] = 2.0 * e; break;
        case Link::InvMu2: mu[i] = 1.0 / std::sqrt(e); dmu[i] = -1.0 / (2.0 * std::pow(e, 1.5)); break;
        case Link::Inverse: mu[i] = 1.0 / e; dmu[i] = -1.0 / (e * e); break;
        }
    }
}

/// Computes mu = g^{-1}(X beta + offset) and D = diag(dmu) X.
/// @throws InvalidInput if beta and X have incompatible sizes
/// @throws NumericalError if the linear predictor is not finite
inline MeanOut mean_structure(const MatrixXd& X, const VectorXd& beta, const VectorXd& offset, Link link) {
    if (beta.size() != X.cols())
        throw InvalidInput("beta has " + std::to_string(beta.size()) + " entries but X has " +
                           std::to_string(X.cols()) + " columns");
    VectorXd eta = X * beta;
    if (offset.size() > 0) eta += offset;
    if (!eta.allFinite()) throw NumericalError("the linear predictor is not finite; check the starting values and the link");
    MeanOut o; link_eval(link, eta, o.mu, o.dmu);
    o.D = X.array().colwise() * o.dmu.array();
    return o;
}

} // namespace mcglm
