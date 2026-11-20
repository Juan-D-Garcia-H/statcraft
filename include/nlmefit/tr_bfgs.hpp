// SPDX-License-Identifier: Apache-2.0
//
// Trust-region quasi-Newton minimization.
//
// The method is a trust-region BFGS variant with Powell damping and a dogleg
// step. It follows Nocedal & Wright, "Numerical Optimization" (2nd edition):
//   * Algorithm 4.1: trust-region framework.
//   * Section 4.1: dogleg step.
//   * Section 18.3: damped BFGS update.
//   * Equation (6.20): initial scaling of the Hessian approximation.
//
// The method is particularly convenient for profiled likelihoods: a rejected
// step (for example, a non-positive-definite covariance matrix) only shrinks the
// trust radius, and the radius bounds the length of every step. This prevents
// large jumps into degenerate regions of the parameter space.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "bfgs_min.hpp"   // Objective, Gradient, BfgsOptions, BfgsResult

namespace mixedfit {

/// Trust-region-specific tuning parameters.
struct TrustRegionOptions {
    double initial_radius = 1.0;    ///< Initial trust radius (in parameter units).
    double max_radius = 100.0;      ///< Maximum trust radius.
    double eta = 1e-4;              ///< Minimum ratio of actual to predicted reduction for accepting a step.
    double radius_tol = 1e-10;      ///< Stop when the radius becomes this small relative to the parameter scale.
};

/// Minimize a scalar function with a trust-region BFGS method.
///
/// The quadratic model is minimized approximately by a dogleg step inside the
/// current trust region. The Hessian approximation is updated with Powell's
/// damping so that it remains positive definite.
///
/// @param f    Objective function.
/// @param x0   Starting point.
/// @param grad Optional analytic gradient. If empty, central differences are used.
/// @param o    BFGS options controlling iterations and tolerances.
/// @param t    Trust-region options.
/// @return The final point, objective value, gradient and termination status.
inline BfgsResult minimize_trust_bfgs(const Objective& f, const Eigen::VectorXd& x0, const Gradient& grad,
                                      const BfgsOptions& o = BfgsOptions(),
                                      const TrustRegionOptions& t = TrustRegionOptions()) {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    const int n = static_cast<int>(x0.size());
    constexpr double inf = std::numeric_limits<double>::infinity();

    BfgsResult r;
    r.x = x0;

    // Map non-finite objective values to +infinity so that they are rejected
    // in the same way as any other non-improving step.
    auto F = [&](const VectorXd& v) {
        ++r.f_evals;
        const double val = f(v);
        return std::isfinite(val) ? val : inf;
    };

    // Analytic gradient when available; otherwise central finite differences.
    auto G = [&](const VectorXd& v) {
        ++r.g_evals;
        VectorXd g(n);
        if (grad) {
            grad(v, g);
        } else {
            VectorXd w = v;
            for (int i = 0; i < n; ++i) {
                const double h = o.fd_step * std::max(1.0, std::abs(v[i]));
                w[i] = v[i] + h; const double fp = F(w);
                w[i] = v[i] - h; const double fm = F(w);
                w[i] = v[i];
                g[i] = (fp - fm) / (2.0 * h);
            }
        }
        return g;
    };

    VectorXd x = x0;
    double fx = F(x);
    if (!std::isfinite(fx)) {
        r.f = fx;
        r.message = "objective is not finite at the initial point";
        return r;
    }

    VectorXd g = G(x);
    MatrixXd B = MatrixXd::Identity(n, n);
    double radius = t.initial_radius;
    bool first = true;
    int small_steps = 0;

    for (int it = 1; it <= o.max_iter; ++it) {
        r.iterations = it - 1;

        if (g.cwiseAbs().maxCoeff() <= o.gtol) {
            r.converged = true;
            r.message = "gradient norm is small";
            break;
        }

        // Solve approximately: min g'p + 1/2 p'Bp subject to |p| <= radius.
        // A dogleg path is used because B is positive definite.
        VectorXd p;
        const VectorXd pB = B.ldlt().solve(-g);

        if (pB.norm() <= radius) {
            p = pB;                                    // unconstrained Newton step
        } else {
            const double gBg = g.dot(B * g);
            const VectorXd pU = -(g.squaredNorm() / gBg) * g;   // Cauchy point

            if (pU.norm() >= radius) {
                p = -(radius / g.norm()) * g;                   // step along -g
            } else {
                // Intersection of pU + tau (pB - pU) with |p| = radius.
                const VectorXd d = pB - pU;
                const double a = d.squaredNorm();
                const double b = 2.0 * pU.dot(d);
                const double c = pU.squaredNorm() - radius * radius;
                const double tau = (-b + std::sqrt(std::max(b * b - 4.0 * a * c, 0.0))) / (2.0 * a);
                p = pU + tau * d;
            }
        }

        const double pred = -(g.dot(p) + 0.5 * p.dot(B * p));   // predicted reduction (> 0)
        const VectorXd xn = x + p;
        const double fn = F(xn);

        double rho = -inf;
        if (std::isfinite(fn) && pred > 0.0) rho = (fx - fn) / pred;

        if (!(rho > t.eta)) {
            // Rejected step: shrink the radius and retry.
            radius = 0.25 * p.norm();
            if (radius < t.radius_tol * (1.0 + x.norm())) {
                r.message = "trust radius exhausted";
                r.converged = g.cwiseAbs().maxCoeff() <= 1e3 * o.gtol;
                break;
            }
            continue;
        }

        const VectorXd gn = G(xn);
        const VectorXd s = xn - x;
        VectorXd y = gn - g;

        // Powell damping keeps B positive definite even when the curvature
        // condition s'y > 0 fails.
        const VectorXd Bs = B * s;
        const double sBs = s.dot(Bs);
        double sy = s.dot(y);

        if (first && sy > 0.0) {
            B = (y.squaredNorm() / sy) * MatrixXd::Identity(n, n);
            first = false;
        } else if (first) {
            first = false;
        }

        const VectorXd Bs2 = B * s;
        const double sBs2 = s.dot(Bs2);
        (void)sBs;

        if (sy < 0.2 * sBs2) {
            const double theta = 0.8 * sBs2 / (sBs2 - sy);
            y = theta * y + (1.0 - theta) * Bs2;
            sy = s.dot(y);
        }

        if (sy > 1e-14) B += (y * y.transpose()) / sy - (Bs2 * Bs2.transpose()) / sBs2;

        if (rho > 0.75 && std::abs(p.norm() - radius) <= 1e-12 * radius)
            radius = std::min(2.0 * radius, t.max_radius);
        else if (rho < 0.25)
            radius = 0.25 * p.norm();

        const double fold = fx;
        x = xn;
        g = gn;
        fx = fn;
        r.iterations = it;

        if (std::abs(fold - fx) <= o.ftol * (1.0 + std::abs(fx))) {
            if (++small_steps >= 3) {
                r.converged = true;
                r.message = "relative change in the objective is small";
                break;
            }
        } else {
            small_steps = 0;
        }
    }

    r.x = x;
    r.f = fx;
    r.grad = g;
    if (r.message.empty()) r.message = "maximum number of iterations reached";
    return r;
}

}  // namespace mixedfit
