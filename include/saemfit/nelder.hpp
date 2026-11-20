// ============================================================================
// @file nelder.hpp
// @brief Internal Nelder-Mead adapter.
//
// The rest of the project uses a minimal API: "minimize fn(x) and return the
// optimal point". This header delegates the work to the COMPLETE implementation
// in the `min` library (see nelder_mead.hpp) and adds what the core does not:
//
//   * an initial simplex SCALED per coordinate (e.g. 0.3*sqrt(diag Omega)),
//     instead of 5% of |x0| (or 2.5e-4 if x0_j == 0) as in scipy;
//   * RESCUE of the starting point: if fn(x0) is not finite (inadmissible
//     region, logJoint = -inf), an admissible point is searched around x0; if
//     none exists, an exception is thrown instead of silently returning garbage;
//   * convergence diagnostics (nelderMeadEx).
//
// Reused by the individual MAP (map.hpp), llis / adaptive GH (loglik.hpp) and
// the per-cluster mode search (multilevel.hpp).
//
// Requires: Eigen3, C++20, nelder_mead.hpp (+ common.hpp).
// ============================================================================

#pragma once

#include "nelder_mead.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Outcome of an internal Nelder-Mead minimization.
 */
struct NMOutcome {
    Vec    x;                                  ///< Best point found.
    double fun = std::numeric_limits<double>::infinity(); ///< Objective value at @ref x.
    bool   converged = false;                  ///< Whether the ftol/xtol criteria were met.
    int    nfev = 0;                           ///< Total number of function evaluations.
    bool   rescued = false;                    ///< Whether x0 had to be moved to find an admissible region.
};

/**
 * @brief Minimize @p fn starting from @p x0.
 *
 * @param fn       Objective function to minimize.
 * @param x0       Initial point; must be nonempty.
 * @param scale    Initial simplex size per coordinate (length n); if empty,
 *                 uses max(0.1*|x0_j|, 0.1).
 * @param max_iter Iteration budget (per run).
 * @param ftol     Relative tolerance (used for both ftol and xtol).
 * @param restarts After convergence, the simplex is RESTARTED around the best
 *                 point (up to `restarts` times) while it keeps improving:
 *                 Nelder-Mead can stall in a degenerate simplex that satisfies
 *                 the criteria without being a minimum (McKinnon 1998;
 *                 Kelley 1999). 0 = no restarts.
 *
 * Adaptive (Gao-Han) coefficients are used when n >= 8.
 *
 * @return The minimization outcome, including the best point, whether it
 *         converged, the number of evaluations and whether rescue was needed.
 *
 * @throws std::invalid_argument If @p x0 is empty.
 * @throws std::runtime_error If no admissible starting point exists or if the
 *         returned optimum is not finite.
 */
inline NMOutcome nelderMeadEx(const std::function<double(const Vec&)>& fn, const Vec& x0,
                              const Vec& scale = Vec(), int max_iter = 2000,
                              double ftol = 1e-12, int restarts = 1) {
    const int n = static_cast<int>(x0.size());
    if (n == 0) throw std::invalid_argument("nelderMead: x0 is empty");

    Vec sc(n);
    for (int j = 0; j < n; ++j) {
        const double d = (scale.size() == n) ? scale[j] : std::max(0.1 * std::abs(x0[j]), 0.1);
        sc[j] = (std::isfinite(d) && d > 1e-12) ? d : 1e-12 + (std::isfinite(d) ? 0.0 : 0.1);
    }

    NMOutcome out;
    Vec xs = x0;
    int extra_fev = 0;
    double f0 = fn(xs); ++extra_fev;
    if (!std::isfinite(f0)) {
        // Rescue: deterministic points around x0 with increasing radii.
        std::mt19937 rng(20240601u);
        std::normal_distribution<double> nz(0.0, 1.0);
        static const double radii[] = {0.25, 0.5, 1.0, 2.0, 4.0};
        bool found = false;
        for (int a = 0; a < 250 && !found; ++a) {
            const double r = radii[a % 5];
            Vec cand = x0;
            for (int j = 0; j < n; ++j) cand[j] += r * sc[j] * nz(rng);
            const double fc = fn(cand); ++extra_fev;
            if (std::isfinite(fc)) { xs = cand; found = true; }
        }
        if (!found) throw std::runtime_error("nelderMead: no admissible initial point (fn is not finite)");
        out.rescued = true;
    }

    min::NelderMeadOptions<double> opt;
    opt.maxiter  = max_iter;
    opt.ftol     = ftol;
    opt.xtol     = ftol;
    opt.adaptive = (n >= 8);

    Vec best = xs;
    double fbest = std::numeric_limits<double>::infinity();
    bool conv = false;
    int nfev = extra_fev;
    for (int r = 0; r <= restarts; ++r) {
        Mat sim(n + 1, n);
        sim.row(0) = best.transpose();
        for (int j = 0; j < n; ++j) {
            sim.row(j + 1) = best.transpose();
            sim(j + 1, j) += sc[j] * ((r == 0) ? 1.0 : 0.5);   // restart: slightly smaller simplex
        }
        auto res = min::nelder_mead(fn, best, opt, Vec(), Vec(), std::optional<Mat>(sim));
        nfev += res.nfev;
        const double improvement = fbest - res.fun;            // +inf on the first run
        if (res.fun < fbest || r == 0) { best = res.x; fbest = res.fun; }
        conv = res.success;
        // Restart only if it converged and the last run still improved appreciably.
        if (!res.success) break;
        if (r > 0 && !(improvement > 10.0 * ftol * (std::abs(fbest) + 1.0))) break;
    }
    if (!std::isfinite(fbest)) throw std::runtime_error("nelderMead: optimum is not finite");
    out.x = best; out.fun = fbest; out.converged = conv; out.nfev = nfev;
    return out;
}

/**
 * @brief Compatibility interface: minimize @p fn and return the optimal point.
 *
 * @param fn        Objective function to minimize.
 * @param x0        Initial point.
 * @param max_iter  Iteration budget.
 * @param init_step Relative step for the initial simplex, used as
 *                  scale_j = init_step * max(|x0_j|, 1).
 * @param ftol      Relative tolerance.
 *
 * @return The optimal point found.
 *
 * @throws std::invalid_argument If @p x0 is empty.
 * @throws std::runtime_error If no admissible starting point exists or if the
 *         returned optimum is not finite.
 */
inline Vec nelderMead(const std::function<double(const Vec&)>& fn, const Vec& x0,
                      int max_iter = 2000, double init_step = 0.5,
                      double ftol = 1e-12) {
    Vec sc(x0.size());
    for (Eigen::Index j = 0; j < x0.size(); ++j) sc[j] = init_step * std::max(std::abs(x0[j]), 1.0);
    return nelderMeadEx(fn, x0, sc, max_iter, ftol).x;
}

} // namespace mcmckl
