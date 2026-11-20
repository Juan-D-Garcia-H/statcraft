// SPDX-License-Identifier: Apache-2.0
//
// Self-starting nonlinear models: formula, analytic derivatives with respect to
// the parameters, and data-driven starting values. The formulas use each
// model's classical parameterization; the derivatives are analytic and the
// starting values come from model-specific linearizations (regressions on
// transformations of `y`) followed by a nonlinear least-squares refinement
// (Levenberg-Marquardt).
//
//   logis      : Asym / (1 + exp((xmid - x)/scal))
//   asymp      : Asym + (R0 - Asym) exp(-exp(lrc) x)
//   asympOff   : Asym (1 - exp(-exp(lrc) (x - c0)))
//   asympOrig  : Asym (1 - exp(-exp(lrc) x))
//   micmen     : Vm x / (K + x)
//   fpl        : A + (B - A) / (1 + exp((xmid - x)/scal))
//   gompertz   : Asym exp(-b2 b3^x)
//   weibull    : Asym - Drop exp(-exp(lrc) x^pwr)         (x > 0)
//
// Use with NLME: `make_ss_fns()` returns the model function and its analytic
// Jacobian for `NlmeModel`; with `A = I`, `beta0 = ss_initial(...)` provides
// the starting values for the fixed effects.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlme_fit.hpp"

namespace mixedfit {

/// Description of a self-starting nonlinear model.
struct SelfStart {
    std::string name;                   ///< Model name (e.g. "logis").
    int k = 0;                          ///< Number of parameters.
    std::vector<std::string> par;       ///< Parameter names, in column order of `Phi`.

    /// Predictions: row j of `Phi` (n x k) holds the parameters for
    /// observation j; `x` is the covariate vector.
    std::function<Eigen::VectorXd(const Eigen::MatrixXd& Phi, const Eigen::VectorXd& x)> f;

    /// Derivatives `d pred_j / d Phi(j, c)` (n x k).
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd& Phi, const Eigen::VectorXd& x)> df;

    /// Approximate starting values computed from all observations `(x, y)`.
    std::function<Eigen::VectorXd(const Eigen::VectorXd& x, const Eigen::VectorXd& y)> init;
};

namespace detail {

/// Ordinary least-squares fit of `v = intercept + slope * u`.
///
/// @param u Predictor values.
/// @param v Response values.
/// @return `{intercept, slope}`; the slope is zero when `u` has no spread.
inline std::pair<double, double> lin_fit(const std::vector<double>& u, const std::vector<double>& v) {
    const int n = static_cast<int>(u.size());
    double mu = 0, mv = 0;
    for (int i = 0; i < n; ++i) { mu += u[i]; mv += v[i]; }
    mu /= n; mv /= n;
    double suu = 0, suv = 0;
    for (int i = 0; i < n; ++i) { suu += (u[i] - mu) * (u[i] - mu); suv += (u[i] - mu) * (v[i] - mv); }
    const double b = suu > 0 ? suv / suu : 0.0;
    return {mv - b * mu, b};
}

/// Clamp `v` to the closed interval `[lo, hi]`.
inline double clipv(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

/// Logit-regression start: map `y` in `[lo, hi]` to `(x - xmid)/scal`.
///
/// @param x    Covariate values.
/// @param y    Response values.
/// @param lo   Lower asymptote.
/// @param hi   Upper asymptote.
/// @param xmid Output: location of the inflection point.
/// @param scal Output: scale parameter.
inline void logistic_start(const Eigen::VectorXd& x, const Eigen::VectorXd& y, double lo, double hi, double& xmid, double& scal) {
    std::vector<double> u, z;
    for (int i = 0; i < x.size(); ++i) {
        // Clamp the response to the open interval so the logit is finite.
        const double p = clipv((y[i] - lo) / (hi - lo), 0.02, 0.98);
        u.push_back(x[i]);
        z.push_back(std::log(p / (1.0 - p)));
    }
    const auto ab = lin_fit(u, z);
    double b = ab.second;
    if (std::abs(b) < 1e-12) b = 1e-12;
    scal = 1.0 / b;
    xmid = -ab.first / b;
}

}  // namespace detail

/// Logistic growth model: `Asym / (1 + exp((xmid - x)/scal))`.
inline SelfStart ss_logis() {
    SelfStart s;
    s.name = "logis"; s.k = 3; s.par = {"Asym", "xmid", "scal"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        return Eigen::VectorXd(P.col(0).array() / (1.0 + ((P.col(1).array() - x.array()) / P.col(2).array()).exp()));
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 3);
        for (int j = 0; j < n; ++j) {
            const double A = P(j, 0), xm = P(j, 1), sc = P(j, 2), e = std::exp((xm - x[j]) / sc), d = (1.0 + e) * (1.0 + e);
            J(j, 0) = 1.0 / (1.0 + e);
            J(j, 1) = -A * e / (sc * d);
            J(j, 2) = A * e * (xm - x[j]) / (sc * sc * d);
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd p(3);
        p[0] = 1.05 * y.maxCoeff();
        detail::logistic_start(x, y, 0.0, p[0], p[1], p[2]);
        return p;
    };
    return s;
}

/// Four-parameter logistic (FPL) model:
/// `A + (B - A) / (1 + exp((xmid - x)/scal))`.
inline SelfStart ss_fpl() {
    SelfStart s;
    s.name = "fpl"; s.k = 4; s.par = {"A", "B", "xmid", "scal"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        return Eigen::VectorXd(P.col(0).array() + (P.col(1) - P.col(0)).array() / (1.0 + ((P.col(2).array() - x.array()) / P.col(3).array()).exp()));
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 4);
        for (int j = 0; j < n; ++j) {
            const double A = P(j, 0), B = P(j, 1), xm = P(j, 2), sc = P(j, 3), e = std::exp((xm - x[j]) / sc), d = (1.0 + e) * (1.0 + e);
            J(j, 0) = 1.0 - 1.0 / (1.0 + e);
            J(j, 1) = 1.0 / (1.0 + e);
            J(j, 2) = -(B - A) * e / (sc * d);
            J(j, 3) = (B - A) * e * (xm - x[j]) / (sc * sc * d);
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd p(4);
        const double mn = y.minCoeff(), mx = y.maxCoeff(), rg = mx - mn;
        p[0] = mn - 0.05 * rg;
        p[1] = mx + 0.05 * rg;
        detail::logistic_start(x, y, p[0], p[1], p[2], p[3]);
        return p;
    };
    return s;
}

/// Asymptotic regression with an offset:
/// `Asym + (R0 - Asym) exp(-exp(lrc) x)`.
inline SelfStart ss_asymp() {
    SelfStart s;
    s.name = "asymp"; s.k = 3; s.par = {"Asym", "R0", "lrc"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        return Eigen::VectorXd(P.col(0).array() + (P.col(1) - P.col(0)).array() * (-(P.col(2).array().exp()) * x.array()).exp());
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 3);
        for (int j = 0; j < n; ++j) {
            const double A = P(j, 0), R0 = P(j, 1), lrc = P(j, 2), E = std::exp(-std::exp(lrc) * x[j]);
            J(j, 0) = 1.0 - E;
            J(j, 1) = E;
            J(j, 2) = -(R0 - A) * E * std::exp(lrc) * x[j];
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        const int n = static_cast<int>(x.size());
        std::vector<int> ord(n);
        for (int i = 0; i < n; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return x[a] < x[b]; });
        const int m = std::max(1, n / 10);
        // Use the extreme 10% of the covariate range to estimate both
        // asymptotes.
        double lo = 0, hi = 0;
        for (int i = 0; i < m; ++i) { lo += y[ord[i]]; hi += y[ord[n - 1 - i]]; }
        lo /= m; hi /= m;
        Eigen::VectorXd p(3);
        p[1] = lo;
        p[0] = hi + 0.1 * (hi - lo);
        // Linearize log|Asym - y| against x to estimate the rate.
        std::vector<double> u, z;
        for (int i = 0; i < n; ++i) {
            const double d = std::abs(p[0] - y[i]);
            if (d > 1e-8) { u.push_back(x[i]); z.push_back(std::log(d)); }
        }
        const double rc = (u.size() >= 2) ? -detail::lin_fit(u, z).second : 0.0;
        p[2] = std::log(rc > 1e-8 ? rc : 1.0 / std::max(1e-8, x.maxCoeff() - x.minCoeff()));
        return p;
    };
    return s;
}

/// Asymptotic regression with a shifted origin:
/// `Asym (1 - exp(-exp(lrc) (x - c0)))`.
inline SelfStart ss_asymp_off() {
    SelfStart s;
    s.name = "asympOff"; s.k = 3; s.par = {"Asym", "lrc", "c0"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        return Eigen::VectorXd(P.col(0).array() * (1.0 - (-(P.col(1).array().exp()) * (x.array() - P.col(2).array())).exp()));
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 3);
        for (int j = 0; j < n; ++j) {
            const double A = P(j, 0), lrc = P(j, 1), c0 = P(j, 2), rc = std::exp(lrc), E = std::exp(-rc * (x[j] - c0));
            J(j, 0) = 1.0 - E;
            J(j, 1) = A * E * rc * (x[j] - c0);
            J(j, 2) = -A * E * rc;
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd p(3);
        p[0] = 1.05 * y.maxCoeff();
        // Linearize log(1 - y/Asym) against x; the slope gives the rate and
        // the intercept the shift.
        std::vector<double> u, z;
        for (int i = 0; i < x.size(); ++i) { u.push_back(x[i]); z.push_back(std::log(1.0 - detail::clipv(y[i] / p[0], -0.5, 0.98))); }
        const auto ab = detail::lin_fit(u, z);
        double rc = -ab.second;
        if (!(rc > 1e-10)) rc = 1.0 / std::max(1e-8, x.maxCoeff() - x.minCoeff());
        p[1] = std::log(rc);
        p[2] = ab.first / rc;
        return p;
    };
    return s;
}

/// Asymptotic regression through the origin:
/// `Asym (1 - exp(-exp(lrc) x))`.
inline SelfStart ss_asymp_orig() {
    SelfStart s;
    s.name = "asympOrig"; s.k = 2; s.par = {"Asym", "lrc"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        return Eigen::VectorXd(P.col(0).array() * (1.0 - (-(P.col(1).array().exp()) * x.array()).exp()));
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 2);
        for (int j = 0; j < n; ++j) {
            const double A = P(j, 0), rc = std::exp(P(j, 1)), E = std::exp(-rc * x[j]);
            J(j, 0) = 1.0 - E;
            J(j, 1) = A * E * rc * x[j];
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd p(2);
        p[0] = 1.05 * y.maxCoeff();
        // Regress log(1 - y/Asym) on x through the origin to estimate the rate.
        double sxz = 0, sxx = 0;
        for (int i = 0; i < x.size(); ++i) { const double z = std::log(1.0 - detail::clipv(y[i] / p[0], -0.5, 0.98)); sxz += x[i] * z; sxx += x[i] * x[i]; }
        double rc = sxx > 0 ? -sxz / sxx : 0.0;
        if (!(rc > 1e-10)) rc = 1.0 / std::max(1e-8, x.maxCoeff());
        p[1] = std::log(rc);
        return p;
    };
    return s;
}

/// Michaelis-Menten model: `Vm x / (K + x)`.
inline SelfStart ss_micmen() {
    SelfStart s;
    s.name = "micmen"; s.k = 2; s.par = {"Vm", "K"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        return Eigen::VectorXd(P.col(0).array() * x.array() / (P.col(1).array() + x.array()));
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 2);
        for (int j = 0; j < n; ++j) {
            const double Vm = P(j, 0), K = P(j, 1), d = K + x[j];
            J(j, 0) = x[j] / d;
            J(j, 1) = -Vm * x[j] / (d * d);
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        // Lineweaver-Burk linearization: 1/y = 1/Vm + (K/Vm) (1/x).
        std::vector<double> u, z;
        for (int i = 0; i < x.size(); ++i) if (x[i] > 0 && y[i] > 0) { u.push_back(1.0 / x[i]); z.push_back(1.0 / y[i]); }
        const auto ab = detail::lin_fit(u, z);
        Eigen::VectorXd p(2);
        p[0] = ab.first > 0 ? 1.0 / ab.first : 1.05 * y.maxCoeff();
        p[1] = ab.first > 0 && ab.second > 0 ? ab.second / ab.first : x.mean();
        return p;
    };
    return s;
}

/// Gompertz growth model: `Asym exp(-b2 b3^x)`.
inline SelfStart ss_gompertz() {
    SelfStart s;
    s.name = "gompertz"; s.k = 3; s.par = {"Asym", "b2", "b3"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        Eigen::VectorXd out(x.size());
        for (int j = 0; j < x.size(); ++j) out[j] = P(j, 0) * std::exp(-P(j, 1) * std::pow(P(j, 2), x[j]));
        return out;
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 3);
        for (int j = 0; j < n; ++j) {
            const double A = P(j, 0), b2 = P(j, 1), b3 = P(j, 2), pw = std::pow(b3, x[j]), E = std::exp(-b2 * pw);
            J(j, 0) = E;
            J(j, 1) = -A * E * pw;
            J(j, 2) = -A * E * b2 * x[j] * std::pow(b3, x[j] - 1.0);
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd p(3);
        p[0] = 1.05 * y.maxCoeff();
        // Linearize log(-log(y/Asym)) against x.
        std::vector<double> u, z;
        for (int i = 0; i < x.size(); ++i) {
            const double q = detail::clipv(y[i] / p[0], 1e-3, 0.999);
            u.push_back(x[i]); z.push_back(std::log(-std::log(q)));
        }
        const auto ab = detail::lin_fit(u, z);
        p[1] = std::exp(ab.first);
        p[2] = std::exp(ab.second);
        return p;
    };
    return s;
}

/// Weibull growth model:
/// `Asym - Drop exp(-exp(lrc) x^pwr)` for `x > 0`.
inline SelfStart ss_weibull() {
    SelfStart s;
    s.name = "weibull"; s.k = 4; s.par = {"Asym", "Drop", "lrc", "pwr"};
    s.f = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        Eigen::VectorXd out(x.size());
        for (int j = 0; j < x.size(); ++j) out[j] = P(j, 0) - P(j, 1) * std::exp(-std::exp(P(j, 2)) * std::pow(x[j], P(j, 3)));
        return out;
    };
    s.df = [](const Eigen::MatrixXd& P, const Eigen::VectorXd& x) {
        const int n = static_cast<int>(x.size());
        Eigen::MatrixXd J(n, 4);
        for (int j = 0; j < n; ++j) {
            const double Dp = P(j, 1), rc = std::exp(P(j, 2)), xp = std::pow(x[j], P(j, 3)), E = std::exp(-rc * xp);
            J(j, 0) = 1.0;
            J(j, 1) = -E;
            J(j, 2) = Dp * E * rc * xp;
            J(j, 3) = Dp * E * rc * xp * std::log(x[j]);
        }
        return J;
    };
    s.init = [](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        Eigen::VectorXd p(4);
        const double mx = y.maxCoeff(), mn = y.minCoeff();
        p[0] = mx + 0.05 * (mx - mn);
        p[1] = p[0] - mn + 0.05 * (mx - mn);
        // Linearize log(-log((Asym - y)/Drop)) against log(x).
        std::vector<double> u, z;
        for (int i = 0; i < x.size(); ++i) {
            if (!(x[i] > 0)) continue;
            const double q = detail::clipv((p[0] - y[i]) / p[1], 1e-3, 0.999);
            u.push_back(std::log(x[i])); z.push_back(std::log(-std::log(q)));
        }
        const auto ab = detail::lin_fit(u, z);
        p[2] = ab.first;
        p[3] = ab.second;
        return p;
    };
    return s;
}

/// Pooled (single-curve) nonlinear least squares by Levenberg-Marquardt using
/// the analytic derivatives of `ss`.
///
/// @param ss      Self-starting model.
/// @param x       Covariate values.
/// @param y       Response values.
/// @param phi     Initial parameter vector.
/// @param rss_out Optional output for the final residual sum of squares.
/// @param maxit   Maximum number of iterations.
/// @return The refined parameter vector.
inline Eigen::VectorXd ss_nls(const SelfStart& ss, const Eigen::VectorXd& x, const Eigen::VectorXd& y, Eigen::VectorXd phi,
                              double* rss_out = nullptr, int maxit = 500) {
    const int n = static_cast<int>(x.size());
    auto rss_of = [&](const Eigen::VectorXd& ph) {
        const Eigen::MatrixXd P = Eigen::VectorXd::Ones(n) * ph.transpose();
        const Eigen::VectorXd r = y - ss.f(P, x);
        return r.allFinite() ? r.squaredNorm() : std::numeric_limits<double>::infinity();
    };
    double rss = rss_of(phi), lam = 1e-3;
    for (int it = 0; it < maxit && std::isfinite(rss); ++it) {
        const Eigen::MatrixXd P = Eigen::VectorXd::Ones(n) * phi.transpose();
        const Eigen::VectorXd r = y - ss.f(P, x);
        const Eigen::MatrixXd J = ss.df(P, x);
        const Eigen::MatrixXd JtJ = J.transpose() * J;
        const Eigen::VectorXd g = J.transpose() * r;
        bool improved = false;
        for (int tr = 0; tr < 30; ++tr) {
            // Damped normal equations; increase damping until the step lowers
            // the residual sum of squares.
            Eigen::MatrixXd A = JtJ;
            A.diagonal() += lam * (JtJ.diagonal().array() + 1e-12).matrix();
            const Eigen::VectorXd d = A.ldlt().solve(g);
            const Eigen::VectorXd cand = phi + d;
            const double rc = rss_of(cand);
            if (rc < rss) {
                const double rel = (rss - rc) / (rss + 1e-300);
                const double step = d.cwiseAbs().maxCoeff() / (1.0 + phi.cwiseAbs().maxCoeff());
                phi = cand; rss = rc; lam = std::max(lam / 3.0, 1e-12); improved = true;
                if (rel < 1e-15 || step < 1e-12) { it = maxit; }
                break;
            }
            lam *= 5.0;
        }
        if (!improved) break;
    }
    if (rss_out) *rss_out = rss;
    return phi;
}

/// Starting values for the model parameters: the model-specific initializer
/// followed by nonlinear least-squares refinement.
///
/// @param ss Self-starting model.
/// @param x  Covariate values.
/// @param y  Response values.
/// @return Refined parameter vector.
/// @throws std::runtime_error If the initializer produces non-finite values.
inline Eigen::VectorXd ss_initial(const SelfStart& ss, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    Eigen::VectorXd p0 = ss.init(x, y);
    if (!p0.allFinite()) throw std::runtime_error("selfStart " + ss.name + ": could not compute starting values");
    double rss = 0.0;
    return ss_nls(ss, x, y, p0, &rss);
}

/// Model function and analytic Jacobian ready for `NlmeModel`.
struct NlmeSSFns {
    NlmeFn f;                           ///< Model function.
    NlmeJacFn jac;                      ///< Analytic Jacobian.
};

/// Build the model function and analytic Jacobian for `NlmeModel`.
///
/// @param ss           Self-starting model.
/// @param x_by_group   Covariate vector for each group: `x_by_group[g]` holds the
///                     covariate values of the observations in group `g`.
/// @return A pair of callables suitable for `NlmeModel`.
inline NlmeSSFns make_ss_fns(const SelfStart& ss, std::vector<Eigen::VectorXd> x_by_group) {
    auto xs = std::make_shared<std::vector<Eigen::VectorXd>>(std::move(x_by_group));
    NlmeSSFns out;
    out.f = [ss, xs](const Eigen::MatrixXd& Phi, int g) { return ss.f(Phi, (*xs)[g]); };
    out.jac = [ss, xs](const Eigen::MatrixXd& Phi, int g) { return ss.df(Phi, (*xs)[g]); };
    return out;
}

}  // namespace mixedfit
