// SPDX-License-Identifier: Apache-2.0
//
// BFGS minimization with a strong Wolfe line search.
//
// The implementation follows Nocedal & Wright, "Numerical Optimization"
// (2nd edition):
//   * Algorithms 3.5 and 3.6: line search and zoom.
//   * Algorithm 6.1 and equation (6.20): BFGS update and initial scaling of the
//     inverse Hessian approximation.
//
// The optimizer is dependency-free apart from Eigen. Both the objective and an
// optional analytic gradient are supplied through `std::function`. When no
// gradient is provided, central finite differences are used.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>

namespace mixedfit {

/// Scalar objective function of an unconstrained minimization problem.
using Objective = std::function<double(const Eigen::VectorXd&)>;

/// Analytic gradient callback. It must write the gradient into the second
/// argument without resizing it.
using Gradient = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

/// Tuning parameters shared by the BFGS and trust-region optimizers.
struct BfgsOptions {
    int    max_iter = 200;      ///< Maximum number of BFGS iterations.
    double gtol     = 1e-8;     ///< Convergence threshold on `||g||_inf`.
    double ftol     = 1e-14;    ///< Relative objective-change tolerance.
    double c1       = 1e-4;     ///< Armijo sufficient-decrease constant.
    double c2       = 0.9;      ///< Strong-Wolfe curvature constant.
    double max_step = 1e3;      ///< Maximum line-search step length.
    double fd_step  = 1e-6;     ///< Relative step for central finite differences.
};

/// Result returned by the minimizers.
struct BfgsResult {
    Eigen::VectorXd x, grad;    ///< Final point and gradient.
    double      f = std::numeric_limits<double>::quiet_NaN();  ///< Final objective value.
    int         iterations = 0;  ///< Number of completed iterations.
    int         f_evals = 0;     ///< Number of objective evaluations.
    int         g_evals = 0;     ///< Number of gradient evaluations.
    bool        converged = false;  ///< Whether a convergence criterion was met.
    std::string message;            ///< Human-readable termination reason.
};

/// Minimize a scalar function with BFGS and a strong Wolfe line search.
///
/// At each iteration the inverse Hessian approximation `H` is updated with the
/// BFGS formula. If the resulting direction is not a descent direction, `H` is
/// reset to the identity and the steepest-descent direction is used instead.
/// A failed line search triggers one retry with the identity approximation; if
/// that also fails, the routine stops.
///
/// @param f    Objective function.
/// @param x0   Starting point.
/// @param grad Optional analytic gradient. If empty, central differences are used.
/// @param o    Optimizer options.
/// @return The final point, objective value, gradient and termination status.
inline BfgsResult minimize_bfgs(const Objective& f, const Eigen::VectorXd& x0,
                                const Gradient& grad = nullptr,
                                const BfgsOptions& o = BfgsOptions()) {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    const int n = static_cast<int>(x0.size());
    constexpr double inf = std::numeric_limits<double>::infinity();

    BfgsResult r;
    r.x = x0;

    // Objective wrapper: non-finite values are mapped to +infinity so that the
    // line search can reject them in a uniform way.
    auto F = [&](const VectorXd& v) {
        ++r.f_evals;
        const double val = f(v);
        return std::isfinite(val) ? val : inf;
    };

    // Gradient wrapper: uses the analytic callback when available, otherwise
    // central differences with a component-wise relative step.
    auto G = [&](const VectorXd& v) {
        ++r.g_evals;
        VectorXd g(n);
        if (grad) {
            grad(v, g);
        } else {
            VectorXd w = v;
            for (int i = 0; i < n; ++i) {
                const double h = o.fd_step * std::max(1.0, std::abs(v[i]));
                w[i] = v[i] + h;
                const double fp = F(w);
                w[i] = v[i] - h;
                const double fm = F(w);
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
    MatrixXd H = MatrixXd::Identity(n, n);
    bool first = true;

    VectorXd p(n), xt(n), gt(n);
    double ft = fx, alpha = 0.0;

    for (int it = 1; it <= o.max_iter; ++it) {
        r.iterations = it - 1;

        if (g.cwiseAbs().maxCoeff() <= o.gtol) {
            r.converged = true;
            r.message = "gradient norm is small";
            break;
        }

        bool accepted = false;

        // Two attempts: the first with the current inverse Hessian, the second
        // (if needed) with an identity reset.
        for (int attempt = 0; attempt < 2 && !accepted; ++attempt) {
            p = -(H * g);
            double d0 = g.dot(p);
            if (!(d0 < 0.0)) {
                H.setIdentity();
                p = -g;
                d0 = -g.squaredNorm();
            }

            auto phi = [&](double a) {
                xt = x + a * p;
                ft = F(xt);
                return ft;
            };
            auto dphi = [&](double a) {
                xt = x + a * p;
                gt = G(xt);
                return gt.dot(p);
            };

            // Zoom stage of the strong Wolfe line search (Algorithm 3.6).
            // Bisection is used to shrink the interval.
            auto zoom = [&](double lo, double hi, double flo) -> bool {
                for (int k = 0; k < 60; ++k) {
                    const double a  = 0.5 * (lo + hi);
                    const double fa = phi(a);

                    if (fa > fx + o.c1 * a * d0 || fa >= flo) {
                        hi = a;
                    } else {
                        const double da = dphi(a);
                        if (std::abs(da) <= -o.c2 * d0) {
                            alpha = a;
                            return true;
                        }
                        if (da * (hi - lo) >= 0.0) hi = lo;
                        lo = a;
                        flo = fa;
                    }
                    if (std::abs(hi - lo) < 1e-16 * std::max(1.0, std::abs(lo))) break;
                }

                if (lo > 0.0) {
                    const double fl = phi(lo);
                    dphi(lo);
                    alpha = lo;
                    return fl <= fx + o.c1 * lo * d0;
                }
                return false;
            };

            // Main line-search stage (Algorithm 3.5).
            auto line_search = [&]() -> bool {
                double aprev = 0.0, fprev = fx;
                double a = first ? std::min(1.0, 1.0 / g.norm()) : 1.0;

                for (int i = 0; i < 30; ++i) {
                    const double fa = phi(a);
                    if (fa > fx + o.c1 * a * d0 || (i > 0 && fa >= fprev))
                        return zoom(aprev, a, fprev);

                    const double da = dphi(a);
                    if (std::abs(da) <= -o.c2 * d0) {
                        alpha = a;
                        return true;
                    }
                    if (da >= 0.0) return zoom(a, aprev, fa);

                    aprev = a;
                    fprev = fa;
                    if (aprev >= o.max_step) break;
                    a = std::min(2.0 * a, o.max_step);
                }

                alpha = aprev;
                return aprev > 0.0;
            };

            accepted = line_search();
            if (!accepted) H.setIdentity();  // retry with steepest descent
        }

        if (!accepted) {
            r.message = "line search failed";
            break;
        }

        const VectorXd s = xt - x;
        const VectorXd y = gt - g;
        const double sy = s.dot(y);

        // BFGS inverse-Hessian update. The first update also rescales H using
        // equation (6.20) of Nocedal & Wright.
        if (sy > 1e-10 * s.norm() * y.norm()) {
            if (first) H = (sy / y.dot(y)) * MatrixXd::Identity(n, n);
            const double rho = 1.0 / sy;
            const VectorXd Hy = H * y;
            H += rho * ((1.0 + rho * y.dot(Hy)) * (s * s.transpose()) - Hy * s.transpose() -
                        s * Hy.transpose());
            first = false;
        }

        const double fold = fx;
        x = xt;
        g = gt;
        fx = ft;
        r.iterations = it;

        if (std::abs(fold - fx) <= o.ftol * (1.0 + std::abs(fx))) {
            r.converged = true;
            r.message = "relative change in the objective is small";
            break;
        }
        if (it == o.max_iter) r.message = "maximum number of iterations reached";
    }

    r.x = x;
    r.f = fx;
    r.grad = g;
    if (r.message.empty()) r.message = "maximum number of iterations reached";
    return r;
}

}  // namespace mixedfit
