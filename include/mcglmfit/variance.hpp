// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file variance.hpp
/// @brief Variance functions V(mu; p) and their derivatives with respect to the exponents.
#pragma once
#include "types.hpp"
#include <cmath>

namespace mcglm {

/// Diagonal of V^{1/2} (or V^{-1/2}) and its derivatives with respect to the exponents.
struct VarOut {
    VectorXd v;       ///< diag(V^{1/2}), or diag(V^{-1/2}) when `inverse` was requested
    VectorXd dp, dq;  ///< derivative with respect to p (and q for `BinomialPQ`); empty unless requested
};

/// Checks that the mean lies in the support of the variance function.
/// @throws NumericalError if some mu <= 0 (or mu >= 1 when `binomial`)
inline void check_mu(const VectorXd& mu, bool binomial) {
    for (Eigen::Index i = 0; i < mu.size(); ++i) {
        if (!(mu[i] > 0)) throw NumericalError("the variance function requires mu > 0");
        if (binomial && !(mu[i] < 1)) throw NumericalError("the binomial variance function requires 0 < mu < 1");
    }
}

/// Evaluates V^{1/2} (or V^{-1/2}) for the given family.
///
/// - `Power`, `PoissonTweedie`, `GeomTweedie`: V = mu^p (one exponent).
/// - `BinomialP`: V = (mu (1 - mu))^p / ntrial (one exponent).
/// - `BinomialPQ`: V = mu^p (1 - mu)^q / ntrial (two exponents).
///
/// @param inverse       return V^{-1/2} instead of V^{1/2}
/// @param deriv_power   also return the derivatives with respect to the exponents
/// @throws InvalidInput on a wrong number of exponents or an unsupported family
/// @throws NumericalError if mu is outside the support
inline VarOut variance_function(Variance var, const VectorXd& mu, const VectorXd& power,
                                const VectorXd& ntrial, bool inverse, bool deriv_power) {
    const Eigen::Index n = mu.size();
    VarOut o;
    if (var == Variance::Power || var == Variance::PoissonTweedie || var == Variance::GeomTweedie) {
        check_mu(mu, false);
        if (power.size() != 1) throw InvalidInput("this variance function requires exactly one exponent");
        const double p = power[0];
        VectorXd mp = mu.array().pow(p);
        VectorXd s = mp.array().sqrt();
        VectorXd lg = mu.array().log();
        o.v = inverse ? VectorXd(1.0 / s.array()) : s;
        if (deriv_power)
            o.dp = inverse ? VectorXd(-(mp.array() * lg.array()) / (2.0 * mp.array().pow(1.5)))
                           : VectorXd((mp.array() * lg.array()) / (2.0 * s.array()));
        return o;
    }
    check_mu(mu, true);
    VectorXd nt = ntrial.size() ? ntrial : VectorXd::Ones(n);
    VectorXd cst = 1.0 / nt.array();
    if (var == Variance::BinomialP) {
        if (power.size() != 1) throw InvalidInput("binomialP requires exactly one exponent");
        const double p = power[0];
        VectorXd mp = mu.array().pow(p), mq = (1.0 - mu.array()).pow(p);
        VectorXd m1m = cst.array() * mp.array() * mq.array();
        VectorXd s = m1m.array().sqrt();
        o.v = inverse ? VectorXd(1.0 / s.array()) : s;
        if (deriv_power) {
            VectorXd num = (1.0 - mu.array()).log() * m1m.array() + mu.array().log() * m1m.array();
            o.dp = inverse ? VectorXd(-num.array() / (2.0 * m1m.array().pow(1.5)))
                           : VectorXd(num.array() / (2.0 * s.array()));
        }
        return o;
    }
    if (var == Variance::BinomialPQ) {
        if (power.size() != 2) throw InvalidInput("binomialPQ requires exactly two exponents");
        const double p = power[0], q = power[1];
        VectorXd mpq = mu.array().pow(p) * (1.0 - mu.array()).pow(q);
        VectorXd m1m = cst.array() * mpq.array();
        VectorXd s = m1m.array().sqrt();
        o.v = inverse ? VectorXd(1.0 / s.array()) : s;
        if (deriv_power) {
            VectorXd den = inverse ? VectorXd(2.0 * m1m.array().pow(1.5) * nt.array())
                                   : VectorXd(2.0 * s.array() * nt.array());
            double sg = inverse ? -1.0 : 1.0;
            o.dp = sg * (mpq.array() * mu.array().log()) / den.array();
            o.dq = sg * (mpq.array() * (1.0 - mu.array()).log()) / den.array();
        }
        return o;
    }
    throw InvalidInput("unsupported variance function");
}

/// True when the variance function depends on the mean (everything except `Constant`).
inline bool variance_uses_mu(Variance v) { return v != Variance::Constant; }

/// Number of variance exponents of a family (0 for `Constant`, 2 for `BinomialPQ`, otherwise 1).
inline int variance_n_power(Variance v) {
    switch (v) { case Variance::Constant: return 0; case Variance::BinomialPQ: return 2; default: return 1; }
}

} // namespace mcglm
