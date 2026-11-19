// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

#ifndef MINIMIZE_COMMON_HPP
#define MINIMIZE_COMMON_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include <optional>
#include <variant>
#include <algorithm>
#include <string_view>
#include <cstdint>

namespace min {

/**
 * @brief Logging verbosity for all optimizers in the library.
 *
 * Higher levels are strictly more verbose and include all output from lower
 * levels. The level is typically passed through an options struct such as
 * `BfgsOptions::verbose`.
 */
enum class VerbosityLevel : int {
    Silent    = 0, ///< No output at all.
    Summary   = 1, ///< Startup header and final summary (with timing).
    Iteration = 2, ///< One compact table row per iteration.
    Detailed  = 3, ///< Iteration rows enriched with step sizes and line-search info.
    Debug     = 4  ///< Everything, including vector previews of x and g.
};

// ── Matrix and Vector Type Aliases ─────────────────────────────────────────

/**
 * @brief Dense column vector of floating-point type @p FP.
 */
template<typename FP = double>
using Vec = Eigen::Matrix<FP, Eigen::Dynamic, 1>;

/**
 * @brief Dense matrix of floating-point type @p FP.
 */
template<typename FP = double>
using Mat = Eigen::Matrix<FP, Eigen::Dynamic, Eigen::Dynamic>;

/** @brief Boolean array (dynamic length), used for masks and activity flags. */
using VecBool = Eigen::Array<bool, Eigen::Dynamic, 1>;

/** @brief Integer vector (dynamic length), used for index/state arrays. */
using VecInt  = Eigen::VectorXi;

/**
 * @brief Optional callback specification for algorithms that accept
 *        user-provided functions.
 *
 * The variant supports:
 *  - a `std::function` taking the current iterate and a matrix, returning a
 *    vector (e.g. a user-provided gradient or preconditioner), or
 *  - a `std::string` naming a built-in method (resolved internally).
 *
 * The outer `std::optional` allows "no callback" to be represented.
 */
template<typename FP = double>
using CallBack = std::optional<
    std::variant<std::function<Vec<FP>(const Vec<FP>&, const Mat<FP>&)>, std::string>>;

/**
 * @brief Optional parameter-scaling specification.
 *
 * Either a concrete per-variable scaling vector, or a `std::string` naming a
 * built-in scaling scheme (e.g. `"auto"`). The outer `std::optional` allows
 * "use default scaling" to be represented.
 */
template<typename FP = double>
using ScaleParam = std::optional<std::variant<Vec<FP>, std::string>>;

/**
 * @brief Classification of a variable's bound status.
 *
 * Used by box-constrained solvers (e.g. L-BFGS-B) to describe which
 * combination of lower/upper bounds is present for each variable.
 */
enum class BoundType : int {
    Unbounded = 0, ///< No lower or upper bound.
    Lower     = 1, ///< Lower bound only.
    Both      = 2, ///< Both lower and upper bounds.
    Upper     = 3  ///< Upper bound only.
};

/**
 * @brief High-level control-flow tasks for the main solver loop.
 *
 * The main loop alternates between deciding what to do (@ref Start,
 * @ref NewX, convergence checks, etc.) and invoking the appropriate
 * sub-routines. This enum names the phases so the state machine is explicit.
 */
enum class Task : uint8_t {
    Start,                    ///< Initialize state and enter the loop.
    EvaluateFG_Start,         ///< Evaluate f and g at the current x (first time).
    EvaluateFG_LineSearch,    ///< Evaluate f and g inside a line search.
    NewX,                     ///< Accept a new iterate x and update state.
    Convergence,              ///< Check convergence criteria.
    Stop,                     ///< Terminate with a normal stop condition.
    Error,                    ///< Terminate due to an error condition.
    Abnormal                  ///< Terminate due to an abnormal condition.
};

/**
 * @brief Status codes for the More–Thuente line search (dcsrch).
 */
enum class LineSearchTask : uint8_t {
    Start,        ///< Begin a new line search.
    EvaluateFG,   ///< Request a function/gradient evaluation.
    Warning,      ///< Line search issued a warning (e.g. roundoff).
    Convergence,  ///< Line search found a point satisfying its criteria.
    Error         ///< Line search failed.
};

/**
 * @brief Subspace-minimization status (`iword` in the L-BFGS-B routine).
 */
enum class SubspaceStatus : uint8_t {
    Constant  = 0, ///< Free variables remain in the initial subspace.
    Boundary  = 1, ///< Free variables hit the bounds.
    Truncated = 5, ///< Truncated Newton direction was used.
    Unknown   = 255 ///< Unclassified / fallback value.
};

/**
 * @brief Unified termination status shared by every minimizer in the library.
 *
 * All solvers (`BFGS`, `L-BFGS-B`, `Nelder-Mead`) report through this single
 * enum so that status codes are comparable across methods and follow the
 * conventional library convention used e.g. by `scipy.optimize`:
 *
 *   - `0`  always means success (converged); `success` is true exactly when
 *          @ref Converged is returned.
 *   - positive values are non-fatal terminations (budgets exhausted, etc.).
 *   - negative values indicate errors.
 *
 * Not every solver can produce every value; each uses only the subset that
 * applies to its own termination logic.
 */
enum class SolverStatus : int {
    Converged        = 0, ///< Convergence criteria satisfied (success).
    MaxIterReached   = 1, ///< Iteration budget exhausted.
    MaxNfevReached   = 2, ///< Function-evaluation budget exhausted.
    LineSearchFailed = 3, ///< Line search could not be satisfied (BFGS).
    CallbackStopped  = 4, ///< User-supplied callback requested termination.
    Error            = -1 ///< Solver terminated due to an error.
};

/**
 * @brief Diagnostic and error codes returned by the L-BFGS-B driver.
 *
 * Negative values indicate errors; zero means no error.
 */
enum class ErrorCode : int {
    None                   = 0,  ///< No error.
    NonPosDefCholesky1     = -1, ///< Matrix in the 1st Cholesky (formk) not positive definite.
    NonPosDefCholesky2     = -2, ///< Matrix in the 2nd Cholesky (formk) not positive definite.
    NonPosDefCholesky3     = -3, ///< Matrix in Cholesky (formt) not positive definite.
    DerivativeNonNegative  = -4, ///< Line-search derivative >= 0 (search direction is not a descent direction).
    LineSearchMaxFeval     = -5, ///< More than 10 f/g evaluations in the last line search.
    InvalidNbd             = -6, ///< `nbd(k)` value is invalid.
    InfeasibleBounds       = -7, ///< Lower bound l(k) > upper bound u(k).
    SingularTriangular     = -8, ///< Triangular system is singular.
    LineSearchFailed       = -9, ///< Line search failed after 20 evaluations.
    InvalidN               = -10,///< Problem dimension n <= 0.
    InvalidM               = -11,///< L-BFGS memory size m <= 0.
    InvalidFactr           = -12 ///< Tolerance factor factr < 0.
};


/**
 * @brief Numerical differentiation method tags.
 *
 * These tags are used in @ref GradParam to request a specific scheme for
 * approximating gradients (and Jacobians/Hessians where applicable).
 */
namespace method {
    /** @brief One-sided (forward) finite differences. */
    struct Forward        {};
    /** @brief Centered finite differences. */
    struct Central        {};
    /** @brief Richardson-extrapolated forward differences. */
    struct RichardsonFwd  {};
    /** @brief Richardson-extrapolated centered differences. */
    struct RichardsonCen  {};

    inline constexpr Forward       forward;             ///< Convenience instance of Forward.
    inline constexpr Central       central;             ///< Convenience instance of Central.
    inline constexpr RichardsonFwd richardson_forward;  ///< Convenience instance of RichardsonFwd.
    inline constexpr RichardsonCen richardson_central;  ///< Convenience instance of RichardsonCen.
}

/**
 * @brief Specification of how to obtain gradients in the optimization routines.
 *
 * The variant supports:
 *  - a numerical-differentiation method tag (see @ref method),
 *  - a `std::string` naming a built-in method (resolved internally),
 *  - a user-supplied callable returning a gradient vector.
 *
 * This allows every optimizer in the library to accept the same gradient
 * argument, whether the gradient is analytical or approximated.
 */
template<typename FP = double>
using GradParam = std::variant<
    method::Forward,
    method::Central,
    method::RichardsonFwd,
    method::RichardsonCen,
    std::string,
    std::function<Vec<FP>(const Vec<FP>&)>>;

} // namespace min

#endif // MINIMIZE_COMMON_HPP