// SPDX-License-Identifier: Apache-2.0
//
// Optimizer selection for the profiled likelihood.
//
// Available back ends:
//   * `Internal`:      in-house BFGS with a strong Wolfe line search
//                      (`bfgs_min.hpp`); no external dependencies.
//   * `TrustRegion`:   in-house trust-region BFGS with a dogleg step
//                      (`tr_bfgs.hpp`); no external dependencies.
//   * `LBFGSB`:        `min::lbfgsb` (L-BFGS-B 3.0, BSD-3). Supports the box
//                      constraint `|theta| <= theta_bound`, which prevents
//                      degenerate parameters from diverging to infinity.
//   * `ScipyBFGS`:     `min::bfgs` (a port of SciPy's BFGS, BSD-3).
//
// `LBFGSB` and `ScipyBFGS` require compiling with
// `-DMIXEDFIT_WITH_MINIMIZER` and adding `-I<directory containing minimizer/>`.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "bfgs_min.hpp"
#include "tr_bfgs.hpp"

#ifdef MIXEDFIT_WITH_MINIMIZER
#include "minimizer/bfgs.hpp"
#include "minimizer/lbfgsb.hpp"
#endif

namespace mixedfit {

/// Available optimization back ends.
enum class Optimizer { Internal, TrustRegion, LBFGSB, ScipyBFGS };

/// Run the selected optimizer on an objective.
///
/// `grad` may be empty, in which case central finite differences are used.
///
/// @param which  Optimizer to run.
/// @param f      Objective function.
/// @param grad   Optional analytic gradient.
/// @param x0     Starting point.
/// @param o      Optimizer options.
/// @param bound  Box bound used by L-BFGS-B (`|theta| <= bound`).
/// @param fscale Positive scaling factor applied before calling external
///               optimizers. External back ends minimize `f / fscale` and use
///               `gtol / fscale`. For objectives such as `-2 log L`, whose
///               magnitude is O(N), using `fscale = N` avoids the first
///               L-BFGS-B step being proportional to `|g|` and jumping outside
///               the box constraints.
/// @return The optimizer result, with objective and gradient rescaled back to
///         the caller's units.
/// @throws std::runtime_error if an external optimizer is requested but the
///         library was compiled without `MIXEDFIT_WITH_MINIMIZER`.
inline BfgsResult run_optimizer(Optimizer which, const Objective& f, const Gradient& grad,
                                const Eigen::VectorXd& x0, const BfgsOptions& o, double bound, double fscale = 1.0) {
    if (which == Optimizer::Internal) return minimize_bfgs(f, x0, grad, o);
    if (which == Optimizer::TrustRegion) return minimize_trust_bfgs(f, x0, grad, o);

#ifdef MIXEDFIT_WITH_MINIMIZER
    const int n = static_cast<int>(x0.size());

    // Scaled objective: infeasible regions are represented by a large but
    // finite value so that external optimizers handle them gracefully.
    auto fobj = [&](const Eigen::VectorXd& x) -> double {
        const double v = f(x);
        return std::isfinite(v) ? v / fscale : 1e30;
    };

    min::GradParam<double> jac = min::method::central;
    if (grad) {
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gf = [&](const Eigen::VectorXd& x) {
            Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
            grad(x, g);
            return Eigen::VectorXd(g / fscale);
        };
        jac = gf;
    }

    BfgsResult r;
    if (which == Optimizer::LBFGSB) {
        min::LbfgsbOptions<double> lo;
        lo.maxiter = o.max_iter;
        lo.gtol = o.gtol / fscale;
        lo.ftol = o.ftol;

        const Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, -bound);
        const Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, bound);
        const auto res = min::lbfgsb<double>(fobj, x0, lo, jac, lb, ub);

        r.x = res.x;
        r.f = res.fun * fscale;
        r.grad = res.jac * fscale;
        r.iterations = res.nit;
        r.f_evals = res.nfev;
        r.g_evals = res.njev;
        r.converged = res.success;
        r.message = res.message;
    } else {
        min::BfgsOptions<double> bo;
        bo.maxiter = o.max_iter;
        bo.gtol = o.gtol / fscale;

        const auto res = min::bfgs<double>(fobj, x0, bo, jac);

        r.x = res.x;
        r.f = res.fun * fscale;
        r.grad = res.jac * fscale;
        r.iterations = res.nit;
        r.f_evals = res.nfev;
        r.g_evals = res.njev;
        r.converged = res.success;
        r.message = res.message;
    }
    return r;
#else
    (void)bound;
    (void)fscale;
    throw std::runtime_error(
        "run_optimizer: compile with -DMIXEDFIT_WITH_MINIMIZER to use the LBFGSB or ScipyBFGS back ends");
#endif
}

}  // namespace mixedfit
