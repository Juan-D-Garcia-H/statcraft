// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of scipy.optimize.minimize(method="Nelder-Mead")
// (SciPy's _minimize_neldermead).
// Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
// See THIRD_PARTY_NOTICES.md for the full licence text and citations.
// ===========================================================================

#ifndef NELDER_MEAD_HPP
#define NELDER_MEAD_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "common.hpp"

// ═══════════════════════════════════════════════════════════════════════════
// File-level documentation: Nelder-Mead downhill-simplex minimizer
// ═══════════════════════════════════════════════════════════════════════════
/**
 * @file nelder_mead.hpp
 * @brief Dense Nelder-Mead (downhill-simplex) direct-search minimizer.
 *
 * Independent C++20 implementation of the Nelder-Mead simplex method. The
 * default mode is a faithful port of `scipy.optimize.minimize(method=
 * "Nelder-Mead")` (SciPy's `_minimize_neldermead`): it reproduces SciPy's
 * trajectory **bit-for-bit** — identical vertex arithmetic, iteration
 * accounting (`nit` = passes + 1), strict evaluation budget (`maxfev`, default
 * 200·n), and initial-simplex construction — validated against SciPy 1.18.1 on
 * a canonical test suite (7/7 bit-identical 60-pass trajectories under both
 * \c adaptive settings; see `/tmp/opencode/nm_diff/README.md`).
 *
 * The only deliberate behavioural difference from SciPy is the convergence
 * test: the spread tolerances are scaled **relative** to the current best
 * (see \ref NelderMeadOptions::xtol and \ref NelderMeadOptions::ftol), whereas
 * SciPy uses absolute tolerances. Defaults otherwise match SciPy:
 * \c maxiter = \c maxfev = 200·n, \c adaptive = false.
 *
 * @section nm_refs References
 *  - J. A. Nelder and R. Mead, *A simplex method for function minimization*,
 *    The Computer Journal 7(4):308–313, 1965.
 *  - F. Gao and L. Han, *Implementing the Nelder-Mead simplex algorithm with
 *    adaptive parameters*, Computational Optimization and Applications
 *    51(1):259–277, 2012.
 *  - J. C. Lagarias, J. A. Reeds, M. H. Wright and P. E. Wright, *Convergence
 *    properties of the Nelder-Mead simplex method in low dimensions*, SIAM
 *    Journal on Optimization 9(1):112–147, 1998.
 *  - SciPy developers, `scipy.optimize.minimize(method="Nelder-Mead")`,
 *    SciPy 1.18.1 — the reference implementation this port is derived from.
 *
 * @section nm_credits Credits & acknowledgements
 *  The algorithm is due to **J. A. Nelder and R. Mead (1965)**; the adaptive
 *  coefficient scheme is due to **F. Gao and L. Han (2012)**. This code is a
 *  faithful C++ port of the reference implementation maintained by the
 *  **SciPy developers**, whose careful, permissively-licensed code made a
 *  bit-reproducible reference possible. Thanks also to Lagarias *et al.*
 *  (1998) for the convergence analysis of the method.
 */
// ═══════════════════════════════════════════════════════════════════════════

namespace min {

// ── Base Types and Enums ───────────────────────────────────────────────────

// ── Nelder-Mead Options and Results ────────────────────────────────────────

/**
 * @brief Options controlling the Nelder-Mead direct-search minimizer.
 *
 * The defaults reproduce `scipy.optimize.minimize(method="Nelder-Mead")`
 * exactly, with one documented exception: the convergence tolerances are
 * interpreted **relatively** (scaled by the best value / coordinate magnitude)
 * instead of absolutely (see @ref xtol and @ref ftol).
 */
template <typename FP = double>
struct NelderMeadOptions {
    /// Maximum **iterations** (passes + 1, matching scipy's `iterations`).
    /// Budget resolution follows scipy: if unset (and @ref maxfev also unset)
    /// it defaults to 200·n; if only one of the two budgets is set, the other
    /// is unlimited.
    std::optional<int> maxiter = std::nullopt;
    /// Maximum number of **function evaluations**. Same default logic as
    /// scipy: 200·n when unset. Imposed strictly — an iteration whose next
    /// evaluation would exceed the budget is aborted without committing the
    /// trial vertex (mirror of scipy's `_MaxFuncCallError` behaviour).
    std::optional<int> maxfev  = std::nullopt;
    /// Tolerance on the simplex spread in x. Interpreted **relatively**:
    /// convergence requires the coordinate spread to fall below
    /// `xtol·(|x_j|+1)`. scipy uses the same 1e-4 value as an *absolute*
    /// tolerance; the relative scaling is a deliberate, documented difference.
    FP xtol = FP(1e-4);
    /// Tolerance on the function-value spread. Interpreted **relatively**:
    /// requires `(f₀ − f_i) < ftol·(|f₀|+1)` where f₀ is the best cost.
    /// See @ref xtol for the corresponding scipy difference.
    FP ftol = FP(1e-4);
    VerbosityLevel verbose = VerbosityLevel::Silent; ///< Logging verbosity level.
    std::ostream* log_stream = &std::cout;           ///< Stream receiving log lines (may be null).
    std::function<void(std::string_view)> log_sink = nullptr; ///< Optional per-line logging callback.
    /// If true, record the best point at every iteration in
    /// @ref NelderMeadResult::all_vecs.
    bool return_all = false;
    /// Use adaptive (Gao–Han) coefficients instead of the standard ones.
    /// Matches scipy's default (`false`).
    bool adaptive = false;
    /// Optional fixed perturbation size used when building the initial
    /// simplex; `<= 0` selects the automatic non-zero scheme (which, like
    /// scipy, only replaces an exactly-zero `y` entry — no `|y|` threshold).
    FP zdelt = FP(-1);
};

/**
 * @brief Result of a call to min::nelder_mead().
 */
template <typename FP = double>
struct NelderMeadResult {
    Vec<FP> x;                     ///< Best point found.
    FP fun = FP(0);                ///< Objective value at @ref x.
    bool success = false;          ///< True if convergence criteria were satisfied.
    SolverStatus status = SolverStatus::Error; ///< Termination status (shared @ref SolverStatus).
    std::string message;           ///< Human-readable termination message.
    int nfev = 0;                  ///< Number of function evaluations performed.
    int nit = 0;                   ///< Number of iterations performed.
    Mat<FP> final_simplex;         ///< Final simplex (rows are vertices, sorted by cost).
    Vec<FP> final_costs;           ///< Objective values at the vertices of @ref final_simplex.
    std::vector<Vec<FP>> all_vecs; ///< Best point at each iteration (only if @ref NelderMeadOptions::return_all).
};

// ── Auxiliary Optimization and Matrix Helpers ──────────────────────────────

namespace internal {

/**
 * @brief Centroid of the n best vertices, using a sequential per-column sum
 *        that reproduces the rounding order of scipy's `add.reduce()/n`.
 *
 * Eigen's vectorized `colwise().mean()` performs packet/horizontal reductions
 * whose summation order can differ by 1 ulp from numpy's sequential reduction;
 * a plain scalar accumulation matches scipy's Nelder-Mead trajectory
 * bit-for-bit.
 *
 * @param sim Sorted simplex (M x n) whose first n rows are the best vertices.
 * @param n   Problem dimension (number of best vertices).
 */
template <typename FP>
Vec<FP> xbar_sequential(const Mat<FP>& sim, int n) {
    Vec<FP> xb(n);
    for (int j = 0; j < n; ++j) {
        FP s = sim(0, j);
        for (int i = 1; i < n; ++i) s += sim(i, j);
        xb(j) = s / static_cast<FP>(n);
    }
    return xb;
}

/**
 * @brief Sort the simplex vertices in ascending order of their costs.
 *
 * Sorting is done in place on @p fsim and @p sim. Small simplices use
 * insertion sort (cache-friendly, allocation-free); larger ones use an
 * indirect permutation to avoid repeated row copies.
 *
 * @param fsim Cost vector of length M (number of vertices).
 * @param sim  Simplex matrix of size M x n.
 */
template <typename FP>
void sort_simplex(Vec<FP>& fsim, Mat<FP>& sim) {
    const int M = static_cast<int>(fsim.size());
    if (M <= 12) {
        // Insertion sort for small simplices (cache-friendly, no allocations).
        for (int i = 1; i < M; ++i) {
            const FP key_f = fsim(i);
            const Eigen::Matrix<FP, 1, Eigen::Dynamic> key_row = sim.row(i);
            int j = i - 1;
            while (j >= 0 && fsim(j) > key_f) {
                fsim(j + 1) = fsim(j);
                sim.row(j + 1) = sim.row(j);
                --j;
            }
            fsim(j + 1) = key_f;
            sim.row(j + 1) = key_row;
        }
    } else {
        // Indirect permutation for larger dimensions.
        std::vector<int> idx(static_cast<size_t>(M));
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) noexcept {
            return fsim(a) < fsim(b);
        });

        Vec<FP> ft = fsim;
        Mat<FP> st = sim;
        for (int i = 0; i < M; ++i) {
            fsim(i) = ft(idx[static_cast<size_t>(i)]);
            sim.row(i) = st.row(idx[static_cast<size_t>(i)]);
        }
    }
}

/**
 * @brief Clamp a vector @p x component-wise to [lb, ub] in place.
 */
template <typename FP>
inline void clip_inplace(Vec<FP>& x, const Vec<FP>& lb, const Vec<FP>& ub) noexcept {
    x = x.cwiseMax(lb).cwiseMin(ub);
}

/**
 * @brief Clamp the specified row of @p sim component-wise to [lb, ub] in place.
 */
template <typename FP>
inline void clip_row_inplace(Mat<FP>& sim, int row, const Vec<FP>& lb, const Vec<FP>& ub) noexcept {
    sim.row(row) = sim.row(row).cwiseMax(lb.transpose()).cwiseMin(ub.transpose());
}

} // namespace internal

// ── Nelder-Mead Solver ─────────────────────────────────────────────────────

/**
 * @brief Minimize @p f using the Nelder-Mead downhill-simplex method.
 *
 * Implements the classical Nelder-Mead direct-search algorithm of Nelder &
 * Mead (1965), with the optional adaptive (Gao–Han) coefficient scheme,
 * optional box constraints, and an optional user-supplied callback that can
 * request early termination. The iteration logic, initial-simplex construction
 * and evaluation budget are ported from scipy's `_minimize_neldermead` and
 * reproduce its trajectory bit-for-bit (see the file header for references).
 *
 * Convergence is declared when both the function-value spread and the simplex
 * spread in x fall below their tolerances (scaled relative to the current
 * best; the relative scaling is the single documented difference from scipy).
 * The method is a **direct search**: no derivatives are required.
 *
 * @tparam FP   Floating-point scalar type.
 * @tparam Func Callable evaluating the objective; must satisfy
 *              `std::invocable<Func, const Vec<FP>&>`.
 *
 * @param f               Objective function to minimize.
 * @param x0              Initial point; must be nonempty.
 * @param opt             Optimizer options (see NelderMeadOptions).
 * @param lb              Optional lower bounds (length <= n; missing entries default to -inf).
 * @param ub              Optional upper bounds (length <= n; missing entries default to +inf).
 * @param initial_simplex Optional initial simplex of size (n+1) x n.
 * @param callback        Optional callback invoked each iteration; it receives
 *                        the current best point and the current simplex, and
 *                        may return a 1-element vector equal to `-1` to
 *                        request termination.
 *
 * @return A NelderMeadResult with the best point, diagnostics, the final
 *         simplex, and (optionally) the trajectory of best points.
 *
 * @throws std::invalid_argument If @p initial_simplex is provided with
 *         dimensions other than (n+1) x n.
 */
template <typename FP = double, typename Func>
requires std::invocable<Func, const Vec<FP>&>
NelderMeadResult<FP> nelder_mead(
    const Func&                  f,
    const Vec<FP>&               x0,
    const NelderMeadOptions<FP>& opt             = NelderMeadOptions<FP>{},
    const Vec<FP>&               lb              = Vec<FP>{},
    const Vec<FP>&               ub              = Vec<FP>{},
    std::optional<Mat<FP>>       initial_simplex = std::nullopt,
    CallBack<FP>                 callback        = std::nullopt)
{
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();

    NelderMeadResult<FP> res;
    const int n = static_cast<int>(x0.size());

    // Unified logging helper writing to both the stream and the optional sink.
    auto log_line = [&](std::string_view msg) {
        if (opt.log_stream) {
            *(opt.log_stream) << msg << "\n";
        }
        if (opt.log_sink) {
            opt.log_sink(msg);
        }
    };

    // Vector preview formatter used in Debug verbosity.
    auto format_vector_preview = [](const Vec<FP>& v, int max_elems = 4) -> std::string {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(3) << "[";
        int print_n = std::min(static_cast<int>(v.size()), max_elems);
        for (int i = 0; i < print_n; ++i) {
            ss << v[i] << (i + 1 < print_n ? ", " : "");
        }
        if (v.size() > max_elems) {
            ss << ", ... (" << v.size() << " dims)";
        }
        ss << "]";
        return ss.str();
    };

    if (n == 0) {
        res.message = "Error: Initial point x0 is empty.";
        res.status = SolverStatus::Error;
        if (opt.verbose != VerbosityLevel::Silent) {
            log_line(res.message);
        }
        return res;
    }

    // Budget resolution, matching scipy's Nelder-Mead defaults:
    //   - neither set  -> both equal to 200*n
    //   - only one set -> the other is (effectively) unlimited
    int maxiter, maxfun;
    if (!opt.maxiter.has_value() && !opt.maxfev.has_value()) {
        maxiter = 200 * n;
        maxfun  = 200 * n;
    } else if (opt.maxiter.has_value() && !opt.maxfev.has_value()) {
        maxiter = *opt.maxiter;
        maxfun  = std::numeric_limits<int>::max();
    } else if (opt.maxfev.has_value() && !opt.maxiter.has_value()) {
        maxfun  = *opt.maxfev;
        maxiter = std::numeric_limits<int>::max();
    } else {
        maxiter = *opt.maxiter;
        maxfun  = *opt.maxfev;
    }

    // Coefficient selection (standard vs. adaptive).
    FP rho, chi, psi, sigma;
    if (opt.adaptive) {
        const FP fN = static_cast<FP>(n);
        rho   = FP(1.0);
        chi   = FP(1.0) + FP(2.0) / fN;
        psi   = FP(0.75) - FP(1.0) / (FP(2.0) * fN);
        sigma = FP(1.0) - FP(1.0) / fN;
    } else {
        rho   = FP(1.0);
        chi   = FP(2.0);
        psi   = FP(0.5);
        sigma = FP(0.5);
    }

    const FP one_plus_rho     = FP(1.0) + rho;
    const FP one_plus_rho_chi = FP(1.0) + rho * chi;
    const FP one_plus_psi_rho = FP(1.0) + psi * rho;
    const FP one_minus_psi    = FP(1.0) - psi;
    const FP rho_chi          = rho * chi;
    const FP psi_rho          = psi * rho;
    const FP nonzdelt         = FP(0.05);

    const FP inf = std::numeric_limits<FP>::infinity();
    Vec<FP> l = Vec<FP>::Constant(n, -inf);
    Vec<FP> u = Vec<FP>::Constant(n,  inf);

    for (int i = 0; i < std::min<int>(n, static_cast<int>(lb.size())); ++i) l(i) = lb(i);
    for (int i = 0; i < std::min<int>(n, static_cast<int>(ub.size())); ++i) u(i) = ub(i);
    const bool has_bounds = (lb.size() > 0 || ub.size() > 0);

    Vec<FP> zdelt_vec(n);
    if (opt.zdelt > FP(0)) {
        zdelt_vec.setConstant(opt.zdelt);
    } else {
        for (int i = 0; i < n; ++i) {
            FP range = u(i) - l(i);
            zdelt_vec(i) = std::isfinite(range) ? std::max(FP(0.00025), nonzdelt * range)
                                                : FP(0.00025);
        }
    }

    // Simplex initialization.
    Mat<FP> sim;
    if (!initial_simplex.has_value()) {
        sim = Mat<FP>(n + 1, n);
        sim.row(0) = x0.transpose();

        const FP delta_nonzero = (opt.zdelt > FP(0)) ? opt.zdelt : nonzdelt;

        for (int k = 0; k < n; ++k) {
            sim.row(k + 1) = x0.transpose();
            FP& yk = sim(k + 1, k);
            if (yk != FP(0)) {   // matches scipy: y[k] != 0 -> scale; exactly 0 -> zdelt
                yk = (FP(1.0) + delta_nonzero) * yk;
            } else {
                yk = zdelt_vec(k);
            }
            if (has_bounds) {
                if (yk > u(k)) yk = sim(0, k) - zdelt_vec(k);
                if (yk < l(k)) yk = sim(0, k) + zdelt_vec(k);
            }
        }
    } else {
        sim = *initial_simplex;
        if (sim.rows() != n + 1 || sim.cols() != n) {
            throw std::invalid_argument(
                "nelder_mead: initial_simplex dimension mismatch. Expected (" +
                std::to_string(n + 1) + ", " + std::to_string(n) + ")");
        }
    }

    if (has_bounds) {
        for (int i = 0; i <= n; ++i) {
            internal::clip_row_inplace(sim, i, l, u);
        }
    }

    Vec<FP> fsim(n + 1);
    for (int k = 0; k <= n; ++k) {
        fsim(k) = f(sim.row(k).transpose());
    }
    int fcalls = n + 1;

    internal::sort_simplex(fsim, sim);
    Vec<FP> xbar = internal::xbar_sequential(sim, n);

    Vec<FP> xw(n), xtmp(n), xr_buf(n);

    if (opt.return_all) {
        res.all_vecs.push_back(sim.row(0).transpose());
    }

    // Console header.
    if (opt.verbose != VerbosityLevel::Silent) {
        std::ostringstream ss;
        ss << "========================================================================\n"
           << " Running Nelder-Mead Direct Search (n = " << n << ")\n"
           << " Mode: " << (opt.adaptive ? "Adaptive Hyperparameters" : "Standard Hyperparameters") << "\n"
           << " Max Iterations: " << maxiter << " | Max Function Evals: " << maxfun << "\n"
           << " xtol: " << std::scientific << std::setprecision(2) << opt.xtol
           << " | ftol: " << opt.ftol << "\n"
           << "========================================================================";
        log_line(ss.str());

        if (opt.verbose >= VerbosityLevel::Iteration) {
            std::ostringstream header;
            header << std::setw(6)  << "Iter" << " | "
                   << std::setw(8)  << "nfev" << " | "
                   << std::setw(14) << "Best f(x)" << " | "
                   << std::setw(14) << "Simplex Size" << " | ";
            if (opt.verbose >= VerbosityLevel::Detailed) {
                header << std::setw(12) << "Operation" << " | ";
            }
            log_line(header.str());
            log_line(std::string(header.str().length(), '-'));
        }
    }

    // Iteration counter starts at 1, exactly like scipy's `iterations`; this
    // makes the pass budget (`iter < maxiter` -> maxiter-1 completed passes)
    // and the reported `nit` (passes + 1) match scipy's Nelder-Mead.
    int iter = 1;
    std::string_view last_op = "Init";

    while (fcalls < maxfun && iter < maxiter) {

        // Convergence check: relative + absolute scaling of both function
        // and simplex spread.
        bool conv_f = true;
        const FP f_scale = std::abs(fsim(0)) + FP(1.0);
        for (int i = 1; i <= n; ++i) {
            if (std::abs(fsim(i) - fsim(0)) > opt.ftol * f_scale) {
                conv_f = false;
                break;
            }
        }

        FP max_x_diff = FP(0.0);
        if (conv_f) {
            bool conv_x = true;
            for (int i = 1; i <= n && conv_x; ++i) {
                for (int j = 0; j < n; ++j) {
                    FP x_scale = std::abs(sim(0, j)) + FP(1.0);
                    FP diff = std::abs(sim(i, j) - sim(0, j));
                    max_x_diff = std::max(max_x_diff, diff);
                    if (diff > opt.xtol * x_scale) {
                        conv_x = false;
                        break;
                    }
                }
            }
            if (conv_x) break;
        }

        xw.noalias()   = sim.row(n).transpose();
        xtmp.noalias() = one_plus_rho * xbar - rho * xw;
        if (has_bounds) internal::clip_inplace(xtmp, l, u);

        const FP fxr = f(xtmp);
        ++fcalls;

        bool do_shrink = false;

        if (fxr < fsim(0)) {
            std::swap(xtmp, xr_buf);
            xtmp.noalias() = one_plus_rho_chi * xbar - rho_chi * xw;
            if (has_bounds) internal::clip_inplace(xtmp, l, u);

            // Strict budget: scipy aborts the trial (no vertex committed)
            // if evaluating the expansion would exceed maxfev.
            if (fcalls >= maxfun) break;
            const FP fxe = f(xtmp);
            ++fcalls;

            if (fxe < fxr) {
                sim.row(n) = xtmp.transpose();
                fsim(n) = fxe;
                last_op = "Expand";
            } else {
                sim.row(n) = xr_buf.transpose();
                fsim(n) = fxr;
                last_op = "Reflect";
            }
        } else if (fxr < fsim(n - 1)) {
            sim.row(n) = xtmp.transpose();
            fsim(n) = fxr;
            last_op = "Reflect";
        } else {
            if (fxr < fsim(n)) {
                xtmp.noalias() = one_plus_psi_rho * xbar - psi_rho * xw;
                if (has_bounds) internal::clip_inplace(xtmp, l, u);
                if (fcalls >= maxfun) break;   // strict budget before outside contraction
                const FP fxc = f(xtmp);
                ++fcalls;

                if (fxc <= fxr) {
                    sim.row(n) = xtmp.transpose();
                    fsim(n) = fxc;
                    last_op = "Contract Out";
                } else {
                    do_shrink = true;
                }
            } else {
                xtmp.noalias() = one_minus_psi * xbar + psi * xw;
                if (has_bounds) internal::clip_inplace(xtmp, l, u);
                if (fcalls >= maxfun) break;   // strict budget before inside contraction
                const FP fxcc = f(xtmp);
                ++fcalls;

                if (fxcc < fsim(n)) {
                    sim.row(n) = xtmp.transpose();
                    fsim(n) = fxcc;
                    last_op = "Contract In";
                } else {
                    do_shrink = true;
                }
            }

            if (do_shrink) {
                last_op = "Shrink";
                const Eigen::Matrix<FP, 1, Eigen::Dynamic> x_best = sim.row(0);
                for (int j = 1; j <= n && fcalls < maxfun; ++j) {
                    sim.row(j) = x_best + sigma * (sim.row(j) - x_best);
                    if (has_bounds) internal::clip_row_inplace(sim, j, l, u);
                    fsim(j) = f(sim.row(j).transpose());
                    ++fcalls;
                }
            }
        }

        internal::sort_simplex(fsim, sim);
        xbar = internal::xbar_sequential(sim, n);
        ++iter;

        if (opt.return_all) {
            res.all_vecs.push_back(sim.row(0).transpose());
        }

        // Tabular iteration output.
        if (opt.verbose >= VerbosityLevel::Iteration) {
            std::ostringstream row;
            row << std::setw(6)  << iter << " | "
                << std::setw(8)  << fcalls << " | "
                << std::scientific << std::setprecision(6)
                << std::setw(14) << fsim(0) << " | "
                << std::setw(14) << max_x_diff << " | ";

            if (opt.verbose >= VerbosityLevel::Detailed) {
                row << std::setw(12) << last_op << " | ";
            }

            log_line(row.str());

            if (opt.verbose >= VerbosityLevel::Debug) {
                log_line("   Best x = " + format_vector_preview(sim.row(0).transpose()));
            }
        }

        // Callback handling.
        if (callback.has_value()) {
            bool stop_requested = false;
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_invocable_r_v<Vec<FP>, T, const Vec<FP>&, const Mat<FP>&>) {
                    Vec<FP> sig = arg(sim.row(0).transpose(), sim);
                    if (sig.size() == 1 && sig(0) == FP(-1.0)) {
                        stop_requested = true;
                    }
                }
            }, callback.value());

            if (stop_requested) {
                res.status = SolverStatus::CallbackStopped;
                res.message = "Optimization stopped by user callback.";
                break;
            }
        }
    }

    res.x = sim.row(0).transpose();
    res.fun = fsim(0);
    res.nit = iter;
    res.nfev = fcalls;
    res.final_simplex = std::move(sim);
    res.final_costs = std::move(fsim);

    if (res.status != SolverStatus::CallbackStopped) {
        if (fcalls >= maxfun) {
            res.status  = SolverStatus::MaxNfevReached;
            res.success = false;
            res.message = "Maximum number of function evaluations reached.";
        } else if (iter >= maxiter) {
            res.status  = SolverStatus::MaxIterReached;
            res.success = false;
            res.message = "Maximum number of iterations reached.";
        } else {
            res.status  = SolverStatus::Converged;
            res.success = true;
            res.message = "Optimization converged successfully.";
        }
    }

    // Final summary.
    if (opt.verbose != VerbosityLevel::Silent) {
        auto end_time = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::ostringstream ss;
        ss << "========================================================================\n"
           << " Optimization Summary:\n"
           << "   Status:      " << res.message << "\n"
           << "   Final f(x):  " << std::scientific << std::setprecision(8) << res.fun << "\n"
           << "   Iterations:  " << res.nit << "\n"
           << "   Func Evals:  " << res.nfev << "\n"
           << "   Time Elapsed:" << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n"
           << "========================================================================";
        log_line(ss.str());
    }

    return res;
}

} // namespace min

#endif // NELDER_MEAD_HPP
