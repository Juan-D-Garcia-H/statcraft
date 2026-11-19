// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_MODERN_HPP
#define LBFGSB_MODERN_HPP

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <optional>
#include <variant>
#include <functional>
#include <cmath>

#include "common.hpp"
#include "numdifference.hpp"
#include "lbfgsb/logger.hpp"
#include "lbfgsb/state.hpp"
#include "lbfgsb/mainlb.hpp"

/**
 * @file lbfgsb.hpp
 * @brief Public front-end of the modern L-BFGS-B minimizer.
 *
 * @details
 * Complete box-constrained limited-memory BFGS solver (Byrd, Lu, Nocedal &
 * Zhu 1995, with the improvements of Morales & Nocedal 2011) implemented in
 * C++/Eigen under a reverse-communication driver architecture. All algorithm
 * pieces live in the `lbfgsb/` directory and are organized into the four
 * modules defined below; this header exposes the high-level `min::lbfgsb()`
 * entry point and its options/result types, and performs the numerical-
 * differentiation dispatch for the gradient.
 *
 * @defgroup lbfgsb_driver L-BFGS-B: driver, diagnostics and bookkeeping
 * @brief Outer iteration control, validation, workspace and logging.
 * @details
 * Comprises the state machine (@ref mainlb.hpp), validation (@ref errclb.hpp),
 * bound initialization (@ref active.hpp), free/active bookkeeping
 * (@ref freev.hpp), the projected-gradient measure (@ref projgr.hpp), the
 * persistent solver state (@ref state.hpp) and the logging front-end
 * (@ref logger.hpp).
 */

/**
 * @defgroup lbfgsb_matrix L-BFGS-B: limited-memory matrix machinery
 * @brief Compact-representation updates and factorizations.
 * @details
 * History update (@ref matupd.hpp), the middle-matrix factor
 * (@ref formt.hpp), the subspace matrix \f$K\f$ (@ref formk.hpp) and its
 * application to vectors (@ref bmv.hpp).
 */

/**
 * @defgroup lbfgsb_direction L-BFGS-B: search-direction computation
 * @brief Cauchy point, subspace residual, truncated-Newton step and breakpoints.
 * @details
 * Generalized Cauchy point (@ref cauchy.hpp), breakpoint ordering helper
 * (@ref hpsolb.hpp), subspace residual (@ref cmprlb.hpp) and subspace
 * minimization (@ref subsm.hpp).
 */

/**
 * @defgroup lbfgsb_linesearch L-BFGS-B: More-Thuente line search
 * @brief Strong-Wolfe safeguarded line search.
 * @details
 * High-level line-search driver (@ref lnsrlb.hpp), the More-Thuente state
 * machine (@ref dcsrch.hpp) and the safeguarded interval update
 * (@ref dcstep.hpp).
 */

namespace min {

// ── Options and Result Types ───────────────────────────────────────────────

/**
 * @brief Options controlling the modern L-BFGS-B minimizer.
 */
template<typename FP = double>
struct LbfgsbOptions {
    int maxiter = 15000;                  ///< Maximum number of quasi-Newton iterations.
    std::optional<int> maxfun = std::nullopt; ///< Optional cap on function evaluations.
    int maxls = 20;                       ///< Maximum backtracking line-search steps per iteration.
    FP ftol = FP(1e-8);                   ///< Relative function-change tolerance.
    FP gtol = FP(1e-5);                   ///< Gradient-infinity-norm tolerance for convergence.
    int maxcor = 10;                      ///< L-BFGS memory size (number of stored correction pairs).
    std::optional<FP> step = std::nullopt; ///< Finite-difference step for numerical differentiation.
    VerbosityLevel verbose = VerbosityLevel::Silent; ///< Logging verbosity level.
    std::ostream* log_stream = &std::cout;           ///< Stream receiving log lines (may be null).
    std::function<void(std::string_view)> log_sink = nullptr; ///< Optional per-line logging callback.
};

/**
 * @brief Result of a call to min::lbfgsb().
 */
template<typename FP = double>
struct LbfgsbResult {
    Vec<FP> x;                    ///< Final iterate.
    FP fun = FP(0);               ///< Objective value at @ref x.
    Vec<FP> jac;                  ///< Gradient at @ref x.
    bool success = false;         ///< True if the solver reported convergence.
    SolverStatus status = SolverStatus::Error; ///< Termination status (shared @ref SolverStatus).
    ErrorCode error_code = ErrorCode::None;      ///< Detailed diagnostic code (if any).
    std::string message;          ///< Human-readable termination message.
    int nfev = 0;                 ///< Number of function evaluations performed.
    int njev = 0;                 ///< Number of gradient evaluations performed.
    int nit = 0;                  ///< Number of quasi-Newton iterations performed.
    FP optimality = FP(0);        ///< Final infinity-norm of the projected gradient.
};

// ── INTERNAL IMPLEMENTATION (high-level minimization function) ─────────────

/**
 * @brief Internal driver implementing the L-BFGS-B minimization loop.
 *
 * This function is not intended to be called directly by users; use
 * min::lbfgsb() instead. It sets up bound types, allocates workspace and
 * solver state, and repeatedly invokes the core `mainlb` routine, servicing
 * its requests for objective/gradient evaluations until convergence, an
 * iteration/function budget is exhausted, or an error occurs.
 *
 * @tparam FP       Floating-point scalar type.
 * @tparam Func     Callable evaluating the objective as `FP(const Vec<FP>&)`.
 * @tparam GradType Gradient specification: either a callable returning a
 *                  `Vec<FP>`, or a `GradParam<FP>` variant (method tag,
 *                  string, or callable).
 *
 * @param f    Objective function to minimize.
 * @param x0   Initial point; must be nonempty.
 * @param opt  Optimizer options.
 * @param jac  Gradient specification (callable or variant).
 * @param lb   Optional lower bounds (length <= n; missing entries default to -inf).
 * @param ub   Optional upper bounds (length <= n; missing entries default to +inf).
 *
 * @return An LbfgsbResult fully populated with the final iterate, diagnostics,
 *         and termination status. The object is valid even on failure.
 */
template<typename FP = double, typename Func, typename GradType>
LbfgsbResult<FP> _lbfgsb(
    const Func& f,
    const Vec<FP>& x0,
    const LbfgsbOptions<FP>& opt,
    const GradType& jac, // Accepts any type: string, tag, or lambda.
    const Vec<FP>& lb,
    const Vec<FP>& ub)
{
    LbfgsbResult<FP> res;
    const int n = static_cast<int>(x0.size());

    if (n == 0) {
        res.message    = "Error: x0 is empty.";
        res.status     = SolverStatus::Error;
        res.error_code = ErrorCode::InvalidN;
        return res;
    }

    // Gradient evaluator. If GradType is directly invocable as
    // Vec<FP>(const Vec<FP>&), call it directly; otherwise assume jac is a
    // GradParam<FP> variant and dispatch through std::visit.
    auto grad_eval = [&](const Vec<FP>& cx) -> Vec<FP> {
        if constexpr (std::is_invocable_r_v<Vec<FP>, GradType, const Vec<FP>&>) {
            // Case 1: user-supplied lambda/callable.
            return jac(cx);
        } else {
            // Case 2: GradParam<FP> variant (method tag, string, or callable).
            return std::visit([&](auto&& arg) -> Vec<FP> {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_invocable_r_v<Vec<FP>, T, const Vec<FP>&>) {
                    return arg(cx);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    FP step_val = opt.step.value_or(DefaultStep<FP, method::Central>::value());
                    return min::Gradient(f, cx, std::string_view(arg), std::optional<FP>(step_val),
                                         opt.verbose != VerbosityLevel::Silent);
                } else {
                    FP step_val = opt.step.value_or(DefaultStep<FP, T>::value());
                    return min::Gradient(f, cx, arg, step_val);
                }
            }, jac);
        }
    };

    // ── 2. Bounds & Bound Types ─────────────────────────────────────────────
    const FP inf = std::numeric_limits<FP>::infinity();

    Vec<FP> l = Vec<FP>::Constant(n, -inf);
    Vec<FP> u = Vec<FP>::Constant(n,  inf);
    for (int i = 0; i < std::min<int>(n, static_cast<int>(lb.size())); ++i) l(i) = lb(i);
    for (int i = 0; i < std::min<int>(n, static_cast<int>(ub.size())); ++i) u(i) = ub(i);

    Vec<BoundType> nbd(n);
    for (int i = 0; i < n; ++i) {
        const bool hl = l(i) > -inf;
        const bool hu = u(i) <  inf;
        nbd(i) = hl ? (hu ? BoundType::Both : BoundType::Lower)
                    : (hu ? BoundType::Upper : BoundType::Unbounded);
    }

    // ── 3. Workspace & Solver State ─────────────────────────────────────────
    const int m = opt.maxcor;
    Workspace<FP> ws_state;
    ws_state.resize(n, m);

    SolverState<FP> st;
    st.task = Task::Start;

    Vec<FP> x_d = x0;
    FP f_d = FP(0);
    Vec<FP> g_d = Vec<FP>::Zero(n);

    const FP eps   = std::numeric_limits<FP>::epsilon();
    const FP factr = opt.ftol / eps;

    Logger<FP> logger(opt.verbose, opt.log_stream, opt.log_sink);
    const int max_nfev = opt.maxfun.value_or(std::numeric_limits<int>::max());

    int nfev = 0, njev = 0;

    // ── 4. Main Loop ────────────────────────────────────────────────────────
    while (true) {
        mainlb<FP>(n, m, x_d, l, u, nbd, f_d, g_d, factr, opt.gtol, ws_state, st, logger);

        if (st.task == Task::EvaluateFG_Start || st.task == Task::EvaluateFG_LineSearch) {
            f_d = f(x_d);
            ++nfev;

            g_d = grad_eval(x_d);
            ++njev;

            if (nfev >= max_nfev) {
                st.task = Task::Stop;
                res.message = "Maximum function evaluations reached (maxfun).";
                res.status  = SolverStatus::MaxNfevReached;
                break;
            }
        } else if (st.task == Task::NewX) {
            res.nit = st.iter;

            if (res.nit >= opt.maxiter) {
                st.task = Task::Stop;
                res.message = "Maximum iterations reached (maxiter).";
                res.status  = SolverStatus::MaxIterReached;
                break;
            }
        } else if (st.task == Task::Convergence) {
            res.success = true;
            res.status  = SolverStatus::Converged;
            res.message = "Optimization converged successfully.";
            break;
        } else {
            res.status     = SolverStatus::Error;
            res.error_code = st.error_code;
            res.message    = "Optimization stopped due to error or abnormal termination.";
            break;
        }
    }

    // ── 5. Results ──────────────────────────────────────────────────────────
    res.x          = x_d;
    res.fun        = f_d;
    res.jac        = g_d;
    res.nfev       = nfev;
    res.njev       = njev;
    res.optimality = st.sbgnrm;
    res.error_code = st.error_code;

    return res;
}

/**
 * @brief Minimize @p f subject to optional box constraints using L-BFGS-B.
 *
 * Public entry point for the modern L-BFGS-B minimizer. Delegates to the
 * internal `_lbfgsb` driver after deducing the gradient type, avoiding
 * redefinition issues and allowing lambdas, method tags, or `GradParam`
 * variants to be passed unchanged.
 *
 * @tparam FP       Floating-point scalar type.
 * @tparam Func     Callable evaluating the objective as `FP(const Vec<FP>&)`.
 * @tparam GradType Gradient specification (callable, method tag, string, or
 *                  `GradParam<FP>` variant). Defaults to central differences.
 *
 * @param f    Objective function to minimize.
 * @param x0   Initial point; must be nonempty.
 * @param opt  Optimizer options (see LbfgsbOptions).
 * @param jac  Gradient specification; defaults to central finite differences.
 * @param lb   Optional lower bounds (missing entries default to -inf).
 * @param ub   Optional upper bounds (missing entries default to +inf).
 *
 * @return An LbfgsbResult with the final iterate, diagnostics, and status.
 *
 * @note The returned object is fully populated even on failure; consult
 *       @ref LbfgsbResult::success, @ref LbfgsbResult::status, and
 *       @ref LbfgsbResult::error_code to determine the outcome.
 */
template<typename FP = double, typename Func, typename GradType = GradParam<FP>>
inline LbfgsbResult<FP> lbfgsb(
    const Func& f,
    const Vec<FP>& x0,
    const LbfgsbOptions<FP>& opt = LbfgsbOptions<FP>{},
    const GradType& jac = method::central,
    const Vec<FP>& lb = Vec<FP>{},
    const Vec<FP>& ub = Vec<FP>{})
{
    return _lbfgsb<FP, Func, GradType>(f, x0, opt, jac, lb, ub);
}

} // namespace min

#endif // LBFGSB_MODERN_HPP