// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file initial.hpp
/// @brief Automatic starting values: a generalised linear model fitted by IRLS per response (quasi families with
/// the matching variance function, or binomial), whose dispersion initialises tau. rho starts at zero.
#pragma once
#include "links.hpp"
#include "variance.hpp"
#include <cmath>

namespace mcglm {
namespace detail {

/// Standard normal quantile (Acklam approximation refined by one Halley step; accuracy about 1e-15).
inline double qnorm_(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                               6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                               3.754408661907416e+00};
    double x;
    if (p < 0.02425) { double q = std::sqrt(-2 * std::log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    else if (p > 1 - 0.02425) { double q = std::sqrt(-2 * std::log(1 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    else { double q = p - 0.5, r = q * q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1); }
    double e = pnorm_(x) - p, u = e * std::sqrt(2 * M_PI) * std::exp(x * x / 2);
    return x - u / (1 + x * u / 2);
}

inline double linkfun(Link l, double mu) {
    switch (l) {
    case Link::Logit: return std::log(mu / (1 - mu));
    case Link::Probit: return qnorm_(mu);
    case Link::Cauchit: return std::tan(M_PI * (mu - 0.5));
    case Link::Cloglog: return std::log(-std::log(1 - mu));
    case Link::Loglog: return -std::log(-std::log(mu));
    case Link::Identity: return mu;
    case Link::Log: return std::log(mu);
    case Link::Sqrt: return std::sqrt(mu);
    case Link::InvMu2: return 1.0 / (mu * mu);
    case Link::Inverse: return 1.0 / mu;
    }
    return mu;
}
inline bool valid_eta(Link l, const VectorXd& e) {
    if (!e.allFinite()) return false;
    if (l == Link::Inverse || l == Link::InvMu2) return (e.array() != 0).all();
    if (l == Link::Sqrt) return (e.array() > 0).all();
    return true;
}

enum class GlmFam { QuasiConst, QuasiMu, QuasiMuOneMinusMu, Binomial };

struct GlmInit { VectorXd beta; double dispersion = 1; };

inline double dev_resid(GlmFam f, double y, double mu, double wt) {
    switch (f) {
    case GlmFam::QuasiConst: return wt * (y - mu) * (y - mu);
    case GlmFam::QuasiMu: return 2 * wt * (y * std::log(y == 0 ? 1.0 : y / mu) - (y - mu));
    default: return 2 * wt * (y * std::log(y == 0 ? 1.0 : y / mu) + (1 - y) * std::log(y == 1 ? 1.0 : (1 - y) / (1 - mu)));
    }
}
inline double var_fn(GlmFam f, double mu) {
    switch (f) { case GlmFam::QuasiConst: return 1.0; case GlmFam::QuasiMu: return mu; default: return mu * (1 - mu); }
}

/// IRLS fit of a GLM. `y` is on the response scale (a proportion for the binomial family) and `wt` are prior
/// weights (the number of trials for the binomial family).
inline GlmInit glm_fit(const MatrixXd& X, const VectorXd& y, const VectorXd& offset, Link link, GlmFam fam,
                       const VectorXd& wt) {
    const Eigen::Index n = X.rows(), p = X.cols();
    VectorXd off = offset.size() ? offset : VectorXd::Zero(n);
    VectorXd mu(n), eta(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        switch (fam) {
        case GlmFam::QuasiConst: mu[i] = y[i]; break;
        case GlmFam::QuasiMu: mu[i] = y[i] + 0.1 * (y[i] == 0); break;
        case GlmFam::QuasiMuOneMinusMu: mu[i] = std::fmax(0.001, std::fmin(0.999, y[i])); break;
        case GlmFam::Binomial: mu[i] = (wt[i] * y[i] + 0.5) / (wt[i] + 1); break;
        }
        eta[i] = linkfun(link, mu[i]);
    }
    VectorXd dmu, tmp; link_eval(link, eta, mu, dmu);
    auto deviance = [&](const VectorXd& m) { double d = 0; for (Eigen::Index i = 0; i < n; ++i) d += dev_resid(fam, y[i], m[i], wt[i]); return d; };
    double devold = deviance(mu), dev = devold;
    VectorXd coef = VectorXd::Zero(p), coefold; bool have_old = false;
    VectorXd wls_w(n);
    for (int iter = 0; iter < 25; ++iter) {
        VectorXd z(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            double v = var_fn(fam, mu[i]);
            wls_w[i] = std::sqrt(wt[i] * dmu[i] * dmu[i] / v);
            z[i] = (eta[i] - off[i]) + (y[i] - mu[i]) / dmu[i];
        }
        MatrixXd Xw = X.array().colwise() * wls_w.array();
        VectorXd zw = z.array() * wls_w.array();
        VectorXd start = Xw.colPivHouseholderQr().solve(zw);
        eta = X * start + off; link_eval(link, eta, mu, dmu);
        dev = deviance(mu);
        int half = 0;
        while (!(std::isfinite(dev) && valid_eta(link, eta)) && have_old && half++ < 25) {
            start = 0.5 * (start + coefold);
            eta = X * start + off; link_eval(link, eta, mu, dmu); dev = deviance(mu);
        }
        coef = start; coefold = start; have_old = true;
        if (std::fabs(dev - devold) / (std::fabs(dev) + 0.1) < 1e-8) break;
        devold = dev;
    }
    GlmInit g; g.beta = coef;
    if (fam == GlmFam::Binomial) g.dispersion = 1.0;
    else {
        double s = 0; Eigen::Index m = 0;
        for (Eigen::Index i = 0; i < n; ++i) if (wt[i] > 0) { s += wt[i] * (y[i] - mu[i]) * (y[i] - mu[i]) / var_fn(fam, mu[i]); ++m; }
        g.dispersion = s / double(m - p);
    }
    return g;
}
} // namespace detail

/// Starting values for `fit`.
/// @param specs    response specifications
/// @param y        stacked responses; NaN entries are skipped in the initial GLM fits
/// @param weights  observation weights (length K n, stacked) or empty; they are prior weights of the initial GLM
///                 (multiplied by the number of trials for the binomial family) and rows with weight 0 are skipped
/// @return regression coefficients, exponents (1), tau (dispersion on the scale of the covariance link) and rho = 0
/// @throws InvalidInput if `weights` has the wrong length or a negative entry
inline Params initial_values(const std::vector<ResponseSpec>& specs, const VectorXd& y,
                             const VectorXd& weights = VectorXd()) {
    using namespace detail;
    const int K = (int)specs.size(); const Eigen::Index n = specs[0].X.rows();
    if (weights.size() != 0) {
        if (weights.size() != K * n) throw InvalidInput("weights must have length K * n");
        if ((weights.array() < 0).any()) throw InvalidInput("weights must be non-negative");
    }
    Params p; p.beta.resize(K); p.power.resize(K); p.tau.resize(K);
    for (int k = 0; k < K; ++k) {
        const ResponseSpec& s = specs[k];
        VectorXd yk = y.segment(k * n, n);
        std::vector<Eigen::Index> keep;
        for (Eigen::Index i = 0; i < n; ++i)
            if (!std::isnan(yk[i]) && (weights.size() == 0 || weights[k * n + i] > 0)) keep.push_back(i);
        if (keep.empty()) throw InvalidInput("response " + std::to_string(k) + " has no observation with positive weight");
        const Eigen::Index m = (Eigen::Index)keep.size();
        MatrixXd X(m, s.X.cols()); VectorXd yy(m), off = s.offset.size() ? VectorXd(m) : VectorXd(), wt = VectorXd::Ones(m);
        if (weights.size() != 0) for (Eigen::Index i = 0; i < m; ++i) wt[i] = weights[k * n + keep[i]];
        for (Eigen::Index i = 0; i < m; ++i) { X.row(i) = s.X.row(keep[i]); yy[i] = yk[keep[i]]; if (off.size()) off[i] = s.offset[keep[i]]; }
        GlmFam fam; Link lk = s.link; VectorXd pw = wt;
        int np = variance_n_power(s.variance);
        switch (s.variance) {
        case Variance::Constant: fam = GlmFam::QuasiConst; p.power[k] = VectorXd::Zero(1); break;
        case Variance::Power: case Variance::PoissonTweedie: case Variance::GeomTweedie:
            fam = GlmFam::QuasiMu; p.power[k] = VectorXd::Ones(1); break;
        default: {
            p.power[k] = VectorXd::Ones(np);
            if (lk == Link::Loglog) lk = Link::Cloglog;
            if (s.ntrial.size()) {
                fam = GlmFam::Binomial;
                for (Eigen::Index i = 0; i < m; ++i) pw[i] = wt[i] * s.ntrial[keep[i]];
            } else fam = GlmFam::QuasiMuOneMinusMu;
        } }
        if (m <= s.X.cols()) throw InvalidInput("response " + std::to_string(k) + ": fewer observations with positive weight than coefficients");
        GlmInit g = glm_fit(X, yy, off, lk, fam, pw);
        p.beta[k] = g.beta;
        const size_t nz = s.Z.size();
        VectorXd tau = VectorXd::Zero(nz);
        switch (s.covariance) {
        case CovLink::Identity: tau[0] = g.dispersion; break;
        case CovLink::Inverse: tau[0] = 1.0 / g.dispersion; break;
        case CovLink::Expm: tau.setConstant(0.1); tau[0] = std::log(g.dispersion); break;
        }
        p.tau[k] = tau;
    }
    p.rho = K > 1 ? VectorXd::Zero(K * (K - 1) / 2) : VectorXd();
    return p;
}

} // namespace mcglm
