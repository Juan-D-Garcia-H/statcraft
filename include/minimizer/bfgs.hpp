// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of scipy.optimize.minimize(method="BFGS") (SciPy's _minimize_bfgs)
// and of its strong-Wolfe line search, which derives from the MINPACK-2
// dcsrch/dcstep algorithm as packaged by SciPy.
// Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
// See THIRD_PARTY_NOTICES.md for the full licence text and citations.
// ===========================================================================

#ifndef MIN_HESSIAN_UPDATE_HPP
#define MIN_HESSIAN_UPDATE_HPP

#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <Eigen/Dense>

#include "common.hpp"
#include "numdifference.hpp"

// ═══════════════════════════════════════════════════════════════════════════
// File-level documentation: BFGS quasi-Newton minimizer
// ═══════════════════════════════════════════════════════════════════════════
/**
 * @file bfgs.hpp
 * @brief BFGS quasi-Newton minimizer with strong-Wolfe backtracking line search.
 *
 * Dense BFGS implementation that can maintain and update either the Hessian
 * approximation B ≈ ∇²f or its inverse H ≈ (∇²f)⁻¹ (see @ref ApproxType).
 * The default mode (`InverseHessian`) is a faithful port of
 * `scipy.optimize.minimize(method="BFGS")` (SciPy's `_minimize_bfgs`): the
 * driver starts from Hk = I and never auto-scales, and the line search is a
 * C++ port of SciPy's strong-Wolfe `line_search_wolfe1`/`wolfe2` + `_zoom`
 * machinery (which in turn derives from MINPACK-2's DCSRCH, `dcstep`).
 * Validated against SciPy 1.18.1 and R's `optim(method="BFGS")`. On the canonical test battery the
 * C++ and SciPy solvers follow the same trajectory (identical it/ev/f); the
 * C++ port is the most robust of the three (105/105 converged on the
 * perturbed-start battery).
 *
 * @section bfgs_refs References
 *  - C. G. Broyden (1970), R. Fletcher (1970), D. Goldfarb (1970) and
 *    D. F. Shanno (1970) — the independent derivations of the BFGS update.
 *  - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
 *    2006 — Algorithms 3.5–3.6 (line search with strong-Wolfe conditions and
 *    safeguarded cubic interpolation) and the standard (inverse-)BFGS formulas.
 *  - J. J. Moré and D. J. Thuente, *Line search algorithms with guaranteed
 *    sufficient decrease*, ACM TOMS 20(3):286–307, 1994 — the MINPACK-2 DCSRCH
 *    bracketing/zoom scheme (`dcstep`) that SciPy ports verbatim.
 *  - SciPy developers, `scipy.optimize.minimize(method="BFGS")`, SciPy 1.18.1 —
 *    the reference implementation this port is derived from.
 *
 * @section bfgs_credits Credits & acknowledgements
 *  The BFGS method is named after its four independent inventors:
 *  **C. G. Broyden, R. Fletcher, D. Goldfarb and D. F. Shanno** (1970).
 *  This code is a faithful C++ port of the reference implementation
 *  maintained by the **SciPy developers**, which itself builds on the
 *  numerical-optimization canon of **Nocedal & Wright** and on the
 *  **MINPACK** line-search work of **Moré & Thuente**. Thanks to all of them
 *  for making high-quality, permissively-licensed references available.
 */
// ═══════════════════════════════════════════════════════════════════════════

namespace min {

// ── Base Types and Enums ───────────────────────────────────────────────────

/**
 * @brief Which matrix the quasi-Newton update maintains.
 *
 * - Hessian:        the strategy stores and updates B ≈ ∇²f(x).
 * - InverseHessian: the strategy stores and updates H ≈ (∇²f(x))⁻¹.
 *
 * The aliases `hess` and `inv_hess` are provided for convenience.
 */
enum class ApproxType {
    Hessian,
    InverseHessian,
    hess = Hessian,
    inv_hess = InverseHessian
};

/**
 * @brief Behavior when the curvature condition (yᵀs > min_curvature · sᵀBs) fails.
 *
 * - SkipUpdate: silently skip the update for this iteration.
 * - DampUpdate: apply Powell-style damping to the gradient difference so the
 *   update remains positive definite.
 */
enum class ExceptionStrategy {
    SkipUpdate,
    DampUpdate
};

// ── Scale Initialization Variants ──────────────────────────────────────────

/**
 * @brief Variant describing how to initialize the Hessian approximation.
 *
 * - FP:      use the given scalar as the initial scaling factor (`s·I`).
 * - Mat<FP>: use the given symmetric matrix as the initial approximation.
 *
 * Defaults to `FP(1.0)`, i.e. an identity Hessian/inverse-Hessian, matching
 * scipy `_minimize_bfgs`, which starts from Hk = I and never auto-scales.
 */
template <typename FP = double>
using InitScale = std::variant<FP, Mat<FP>>;

// ── L-BFGS-B-Style Options and Results ─────────────────────────────────────

/**
 * @brief Options controlling the BFGS minimizer.
 *
 * The defaults mirror `scipy.optimize.minimize(method="BFGS")` where
 * applicable (e.g. `gtol = 1e-5`, identity inverse-Hessian seed). Convergence
 * is declared when the infinity-norm of the gradient falls below @ref gtol; a
 * relative function-change / step-size fallback is also honoured when the line
 * search stalls at machine precision, matching scipy and R's `vmmin`.
 */
template <typename FP = double>
struct BfgsOptions {
    /// Maximum number of quasi-Newton iterations.
    int maxiter = 1000;
    /// Optional cap on objective (and gradient) evaluations. If unset the
    /// budget is effectively unlimited (driven by @ref maxiter).
    std::optional<int> maxfun = std::nullopt;
    /// Maximum number of line-search evaluations per iteration (zoom limit,
    /// as in scipy/mcstep).
    int maxls = 20;
    /// Relative function-change tolerance: accepts a candidate if the relative
    /// decrease falls below `ftol` (reltol, as in scipy / R `vmmin`).
    FP ftol = FP(1e-8);
    /// Gradient-infinity-norm tolerance for convergence (matches scipy's
    /// `gtol = 1e-5`).
    FP gtol = FP(1e-5);
    /// Finite-difference step for numerical differentiation; unset selects the
    /// automatic default of the configured @p jac method.
    std::optional<FP> step = std::nullopt;
    /// Which matrix the quasi-Newton update maintains: the Hessian B or its
    /// inverse H. The default (`InverseHessian`) reproduces scipy's BFGS;
    /// `Hessian` is a direct-approximation mode without a scipy analog.
    ApproxType approx_type = ApproxType::InverseHessian;
    /// Behavior when the curvature condition fails: skip the update, or apply
    /// Powell-style damping so B/H stays positive definite.
    ExceptionStrategy exception_strategy = ExceptionStrategy::SkipUpdate;
    /// Curvature threshold used by the guarded update; defaults depend on
    /// @ref exception_strategy (1e-8 for `SkipUpdate`, 0.2 for `DampUpdate`).
    std::optional<FP> min_curvature = std::nullopt;
    /// Initial scale of B or H: a scalar `s` starts from `s·I`, or a full
    /// symmetric matrix can be supplied. Defaults to the identity, matching
    /// scipy `_minimize_bfgs` (Hk = I, never auto-scaled).
    InitScale<FP> init_scale = FP(1.0);
    VerbosityLevel verbose = VerbosityLevel::Silent; ///< Logging verbosity level.
    std::ostream* log_stream = &std::cout;           ///< Stream receiving log lines (may be null).
    std::function<void(std::string_view)> log_sink = nullptr; ///< Optional per-line logging callback.
};

/**
 * @brief Result of a call to min::bfgs().
 */
template <typename FP = double>
struct BfgsResult {
    Vec<FP> x;                    ///< Final iterate.
    FP fun = FP(0);               ///< Objective value at @ref x.
    Vec<FP> jac;                  ///< Gradient at @ref x.
    Mat<FP> hessian_approx;       ///< Final Hessian (or inverse-Hessian) approximation.
    bool success = false;         ///< True if the optimizer reported convergence.
    SolverStatus status = SolverStatus::Error; ///< Termination status (shared @ref SolverStatus).
    std::string message;          ///< Human-readable termination message.
    int nfev = 0;                 ///< Number of function evaluations performed.
    int njev = 0;                 ///< Number of gradient evaluations performed.
    int nit = 0;                  ///< Number of quasi-Newton iterations performed.
    FP optimality = FP(0);        ///< Final infinity-norm of the gradient.
};

// ── Abstract Base Class: HessianUpdateStrategy ─────────────────────────────

/**
 * @brief Abstract interface for Hessian (or inverse-Hessian) update strategies.
 *
 * Concrete strategies maintain either B ≈ ∇²f or H ≈ (∇²f)⁻¹ and expose it
 * through a matrix-vector product (`dot`) and a dense `get_matrix()` accessor.
 */
template <typename FP = double>
class HessianUpdateStrategy {
public:
    using Vector = Vec<FP>;
    using Matrix = Mat<FP>;

    virtual ~HessianUpdateStrategy() = default;

    /**
     * @brief Reset the strategy for a problem of dimension @p n.
     * @param n           Problem dimension.
     * @param approx_type Whether to maintain the Hessian or its inverse.
     */
    virtual void initialize(int n, ApproxType approx_type) = 0;

    /**
     * @brief Incorporate a new (s, y) pair into the approximation.
     * @param delta_x    Step vector s = x_{k+1} - x_k.
     * @param delta_grad Gradient difference y = g_{k+1} - g_k.
     */
    virtual void update(const Vector& delta_x, const Vector& delta_grad) = 0;

    /**
     * @brief Apply the stored matrix to @p p.
     * @return B * p (or H * p), depending on @ref ApproxType.
     */
    [[nodiscard]] virtual Vector dot(const Vector& p) const = 0;

    /**
     * @brief Return the stored approximation as a dense symmetric matrix.
     */
    [[nodiscard]] virtual Matrix get_matrix() const = 0;

    /** @brief Convenience operator: `strategy * p` is equivalent to `strategy.dot(p)`. */
    [[nodiscard]] Vector operator*(const Vector& p) const {
        return dot(p);
    }
};

// ── Intermediate Base Class: FullHessianUpdateStrategy ─────────────────────

/**
 * @brief Common machinery shared by dense quasi-Newton update strategies.
 *
 * Handles:
 *  - Allocation and initial scaling of B or H on first update.
 *  - Auto-scaling and user-provided initialization (scalar or full matrix).
 *  - Symmetric matrix-vector products via `selfadjointView<Eigen::Lower>()`.
 *  - Mirroring the lower triangle to the upper triangle in `get_matrix()`.
 *
 * Derived classes only implement `update_implementation()`.
 */
template <typename FP = double>
class FullHessianUpdateStrategy : public HessianUpdateStrategy<FP> {
public:
    using typename HessianUpdateStrategy<FP>::Vector;
    using typename HessianUpdateStrategy<FP>::Matrix;

    /**
     * @param init_scale Policy used to initialize B or H on the first update.
     */
    explicit FullHessianUpdateStrategy(InitScale<FP> init_scale = FP(1.0))
        : init_scale_(std::move(init_scale)) {}

    /**
     * @brief Reset the strategy for a problem of dimension @p n.
     *
     * Allocates an identity matrix for the matrix being maintained and clears
     * the other one. Sets `first_iteration_` so the initial scaling is applied
     * on the next call to update().
     */
    void initialize(int n, ApproxType approx_type) override {
        first_iteration_ = true;
        n_ = n;
        approx_type_ = approx_type;

        if (approx_type_ == ApproxType::Hessian) {
            B_ = Matrix::Identity(n, n);
            H_.resize(0, 0);
        } else {
            H_ = Matrix::Identity(n, n);
            B_.resize(0, 0);
        }
    }

    /**
     * @brief Apply initial scaling (if needed) and delegate to the concrete update.
     *
     * The first call after initialize() applies the initialization policy from
     * @ref init_scale_ (scalar or full matrix). Subsequent calls only run the
     * concrete update.
     */
    void update(const Vector& delta_x, const Vector& delta_grad) override {
        if (delta_x.isZero(0.0)) {
            return;
        }

        if (delta_grad.isZero(0.0)) {
            std::cerr << "Warning: delta_grad == 0.0. Check if the approximated function is linear.\n";
            return;
        }

        if (first_iteration_) {
            FP scalar_scale = FP(1.0);
            bool replace = false;
            Matrix matrix_scale;

            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_arithmetic_v<T>) {
                    scalar_scale = static_cast<FP>(arg);
                } else if constexpr (std::is_same_v<T, Matrix>) {
                    replace = true;
                    if (arg.rows() != n_ || arg.cols() != n_) {
                        throw std::invalid_argument("Matrix dimension mismatch.");
                    }
                    if (!arg.isApprox(arg.transpose())) {
                        throw std::invalid_argument("Initial scale matrix must be symmetric.");
                    }
                    matrix_scale = arg;
                }
            }, init_scale_);

            if (approx_type_ == ApproxType::Hessian) {
                B_ = replace ? matrix_scale : (B_ * scalar_scale);
            } else {
                H_ = replace ? matrix_scale : (H_ * scalar_scale);
            }

            first_iteration_ = false;
        }

        update_implementation(delta_x, delta_grad);
    }

    /**
     * @brief Multiply the current approximation by @p p (using only the lower triangle).
     */
    [[nodiscard]] Vector dot(const Vector& p) const override {
        if (approx_type_ == ApproxType::Hessian) {
            return B_.template selfadjointView<Eigen::Lower>() * p;
        } else {
            return H_.template selfadjointView<Eigen::Lower>() * p;
        }
    }

    /**
     * @brief Return the current approximation as a dense, fully symmetric matrix.
     *
     * The internal storage keeps only the lower triangle; the upper triangle
     * is filled by mirroring the lower one.
     */
    [[nodiscard]] Matrix get_matrix() const override {
        Matrix M = (approx_type_ == ApproxType::Hessian) ? B_ : H_;
        M.template triangularView<Eigen::StrictlyUpper>() = M.transpose().template triangularView<Eigen::StrictlyUpper>();
        return M;
    }

protected:
    /**
     * @brief Concrete update to be implemented by derived classes.
     *
     * Called with the current (s, y) pair after initialization and any
     * necessary scaling.
     */
    virtual void update_implementation(const Vector& delta_x, const Vector& delta_grad) = 0;

    InitScale<FP> init_scale_;                        ///< Initialization policy.
    bool first_iteration_ = true;                     ///< True until the first update has been applied.
    int n_ = 0;                                       ///< Problem dimension.
    ApproxType approx_type_ = ApproxType::Hessian;    ///< Whether B_ or H_ is being maintained.
    Matrix B_;                                        ///< Hessian approximation (lower triangle).
    Matrix H_;                                        ///< Inverse-Hessian approximation (lower triangle).
};

// ── Derived Class: BFGS ────────────────────────────────────────────────────

/**
 * @brief Dense BFGS update strategy.
 *
 * Implements the standard rank-two BFGS update (Broyden, Fletcher, Goldfarb &
 * Shanno, 1970; formulas as in Nocedal & Wright 2006, eqs. 6.17/6.19) for
 * either the Hessian B or its inverse H, with two safeguards:
 *  - If wᵀMw ≤ 0 (non-positive curvature detected), the approximation is
 *    reset to a scaled identity before applying the update.
 *  - If the curvature condition wz ≤ min_curvature · wMw fails, the update is
 *    either skipped or damped (Powell-style damping; Powell 1978) depending on
 *    the configured ExceptionStrategy.
 *
 * Note that the default (inverse-Hessian) mode reproduces scipy's
 * `_minimize_bfgs` seed and update; the Hessian mode is a direct-approximation
 * variant without a scipy analog.
 */
template <typename FP = double>
class BFGS : public FullHessianUpdateStrategy<FP> {
public:
    using typename FullHessianUpdateStrategy<FP>::Vector;
    using typename FullHessianUpdateStrategy<FP>::Matrix;
    using FullHessianUpdateStrategy<FP>::approx_type_;
    using FullHessianUpdateStrategy<FP>::n_;
    using FullHessianUpdateStrategy<FP>::B_;
    using FullHessianUpdateStrategy<FP>::H_;

    /**
     * @param exception_strategy Behavior when the curvature condition fails.
     * @param min_curvature      Optional curvature threshold; defaults to 1e-8
     *                           for SkipUpdate and 0.2 for DampUpdate.
     * @param init_scale         Scalar (`s·I`) or symmetric-matrix initial approximation.
     */
    explicit BFGS(ExceptionStrategy exception_strategy = ExceptionStrategy::SkipUpdate,
                  std::optional<FP> min_curvature = std::nullopt,
                  InitScale<FP> init_scale = FP(1.0))
        : FullHessianUpdateStrategy<FP>(std::move(init_scale)),
          exception_strategy_(exception_strategy) {

        if (min_curvature.has_value()) {
            min_curvature_ = min_curvature.value();
        } else {
            min_curvature_ = (exception_strategy_ == ExceptionStrategy::SkipUpdate) ? FP(1e-8) : FP(0.2);
        }
    }

    /** @brief Construct from a BfgsOptions instance. */
    explicit BFGS(const BfgsOptions<FP>& opt)
        : BFGS(opt.exception_strategy, opt.min_curvature, opt.init_scale) {}

protected:
    /**
     * @brief Rank-two BFGS update for the inverse Hessian H.
     *
     * H ← H - (s·(Hy)ᵀ + (Hy)·sᵀ)/ys + ((ys + yHy)/(ys·ys))·s·sᵀ
     */
    void update_inverse_hessian(FP ys, const Vector& Hy, FP yHy, const Vector& s) {
        const FP inv_ys = FP(1.0) / ys;
        const FP scale_ss = (ys + yHy) / (ys * ys);

        H_.template selfadjointView<Eigen::Lower>().rankUpdate(s, Hy, -inv_ys);
        H_.template selfadjointView<Eigen::Lower>().rankUpdate(s, scale_ss);
    }

    /**
     * @brief Rank-two BFGS update for the Hessian B.
     *
     * B ← B + (y·yᵀ)/ys - (Bs·(Bs)ᵀ)/(sᵀBs)
     */
    void update_hessian(FP ys, const Vector& Bs, FP sBs, const Vector& y) {
        const FP inv_ys = FP(1.0) / ys;
        const FP inv_sBs = FP(-1.0) / sBs;

        B_.template selfadjointView<Eigen::Lower>().rankUpdate(y, inv_ys);
        B_.template selfadjointView<Eigen::Lower>().rankUpdate(Bs, inv_sBs);
    }

    /**
     * @brief Apply the guarded BFGS rank-two update.
     *
     * Chooses (w, z) according to whether B or H is being updated, checks
     * positive definiteness and the curvature condition, and dispatches to
     * the appropriate rank-two routine.
     */
    void update_implementation(const Vector& delta_x, const Vector& delta_grad) override {
        Vector w = (approx_type_ == ApproxType::Hessian) ? delta_x : delta_grad;
        Vector z = (approx_type_ == ApproxType::Hessian) ? delta_grad : delta_x;

        FP wz = w.dot(z);
        Vector Mw = this->dot(w);
        FP wMw = Mw.dot(w);

        if (wMw <= FP(0.0)) {
            // Non-positive curvature: restart from the identity (scipy applies
            // the update unconditionally; a plain identity reset is the safest
            // available safeguard once the approximation is not SPD).
            if (approx_type_ == ApproxType::Hessian) {
                B_ = Matrix::Identity(n_, n_);
            } else {
                H_ = Matrix::Identity(n_, n_);
            }
            Mw = this->dot(w);
            wMw = Mw.dot(w);
        }

        if (wz <= min_curvature_ * wMw) {
            if (exception_strategy_ == ExceptionStrategy::SkipUpdate) {
                return;
            } else if (exception_strategy_ == ExceptionStrategy::DampUpdate) {
                FP update_factor = (FP(1.0) - min_curvature_) / (FP(1.0) - wz / wMw);
                z = update_factor * z + (FP(1.0) - update_factor) * Mw;
                wz = w.dot(z);
            }
        }

        if (approx_type_ == ApproxType::Hessian) {
            update_hessian(wz, Mw, wMw, z);
        } else {
            update_inverse_hessian(wz, Mw, wMw, z);
        }
    }

private:
    ExceptionStrategy exception_strategy_; ///< Behavior on curvature failure.
    FP min_curvature_;                     ///< Curvature threshold.
};

// ── Minimizer Interface Homogeneous with L-BFGS-B ──────────────────────────

/**
 * @brief Minimize @p f using a BFGS quasi-Newton method with backtracking line search.
 *
 * The routine implements a full BFGS update on either the Hessian or its
 * inverse (selected via @ref BfgsOptions::approx_type), with safeguards for
 * non-positive curvature, and a strong-Wolfe backtracking line search
 * (Armijo sufficient decrease plus curvature condition) with safeguarded
 * quadratic/cubic interpolation that reproduces scipy's `_zoom` behaviour.
 * In the default configuration this driver is a faithful port of
 * `scipy.optimize.minimize(method="BFGS")` (see the file header for the
 * algorithm references: Broyden, Fletcher, Goldfarb & Shanno 1970; Nocedal
 * & Wright 2006, Algs. 3.5–3.6; Moré & Thuente 1994).
 *
 * @tparam FP       Floating-point scalar type.
 * @tparam Func     Callable type evaluating the objective; must be callable
 *                  as `FP(const Vec<FP>&)`.
 * @tparam GradType Gradient specification. Either a callable returning a
 *                  `Vec<FP>`, or a method tag/string accepted by the internal
 *                  numerical-differentiation helper. Defaults to central
 *                  differences.
 *
 * @param f    Objective function to minimize.
 * @param x0   Initial point; must be nonempty.
 * @param opt  Optimizer options (see BfgsOptions).
 * @param jac  Gradient specification (analytical callable, method tag, or
 *             finite-difference method identifier).
 *
 * @return A BfgsResult containing the final iterate, function value, gradient,
 *         Hessian approximation, status, and diagnostics.
 *
 * @note The returned object is fully populated even when the routine fails to
 *       converge; inspect @ref BfgsResult::success and
 *       @ref BfgsResult::status to determine the outcome.
 */
template <typename FP = double, typename Func, typename GradType = GradParam<FP>>
BfgsResult<FP> bfgs(
    const Func& f,
    const Vec<FP>& x0,
    const BfgsOptions<FP>& opt = BfgsOptions<FP>{},
    GradType&& jac = method::central)
{
    using Vector = Vec<FP>;
    using Clock = std::chrono::high_resolution_clock;

    auto start_time = Clock::now();
    BfgsResult<FP> res;
    const int n = static_cast<int>(x0.size());

    // Logging helper: writes to both the stream and the optional sink.
    auto log_line = [&](std::string_view msg) {
        if (opt.log_stream) {
            *(opt.log_stream) << msg << "\n";
        }
        if (opt.log_sink) {
            opt.log_sink(msg);
        }
    };

    // Vector preview formatter (Debug verbosity).
    auto format_vector_preview = [](const Vector& v, int max_elems = 4) -> std::string {
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

    // Explicitly convert the generic gradient argument to the internal GradParam<FP> type.
    GradParam<FP> internal_jac = std::forward<GradType>(jac);

    // Gradient evaluator: dispatches on the variant alternative.
    auto grad_eval = [&](const Vector& cx) -> Vector {
        return std::visit([&](auto&& arg) -> Vector {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_invocable_r_v<Vector, T, const Vector&>) {
                return arg(cx);
            } else if constexpr (std::is_same_v<T, std::string>) {
                FP step_val = opt.step.value_or(DefaultStep<FP, method::Central>::value());
                return min::Gradient(f, cx, std::string_view(arg), std::optional<FP>(step_val),
                                     opt.verbose != VerbosityLevel::Silent);
            } else {
                FP step_val = opt.step.value_or(DefaultStep<FP, T>::value());
                return min::Gradient(f, cx, arg, step_val);
            }
        }, internal_jac);
    };

    // Initial scale: defaults to the identity (FP(1.0)), so by construction the
    // driver reproduces scipy `_minimize_bfgs` (Hk = I, never auto-scaled).
    // Explicit FP/Mat values supplied by the caller are honored as-is.
    BFGS<FP> bfgs(opt);
    bfgs.initialize(n, opt.approx_type);

    Vector x = x0;
    FP f_val = f(x);
    Vector g = grad_eval(x);

    res.nfev = 1;
    res.njev = 1;
    res.nit = 0;
    res.optimality = g.template lpNorm<Eigen::Infinity>();

    // Value of f at the previous accepted iterate. scipy's BFGS seeds the step
    // length with `old_old_fval = old_fval + ||g0||/2` and afterwards uses the
    // last accepted f, giving the first-trial estimate
    //   alpha1 = min(1, 1.01 * 2 * (f(xk) - f(xkm1)) / (gk . dk)).
    FP f_prev = f_val + FP(0.5) * g.template lpNorm<2>();

    const int max_evals = opt.maxfun.value_or(std::numeric_limits<int>::max());

    // Print summary header.
    if (opt.verbose != VerbosityLevel::Silent) {
        std::ostringstream ss;
        ss << "========================================================================\n"
           << " Running BFGS Minimizer (n = " << n << ")\n"
           << " Mode: " << (opt.approx_type == ApproxType::InverseHessian ? "Inverse Hessian Update" : "Direct Hessian Update") << "\n"
           << " Max Iterations: " << opt.maxiter << " | gtol: " << std::scientific << std::setprecision(2) << opt.gtol << "\n"
           << "========================================================================";
        log_line(ss.str());

        if (opt.verbose >= VerbosityLevel::Iteration) {
            std::ostringstream header;
            header << std::setw(6)  << "Iter" << " | "
                   << std::setw(8)  << "nfev" << " | "
                   << std::setw(14) << "f(x)" << " | "
                   << std::setw(12) << "||proj g||" << " | ";
            if (opt.verbose >= VerbosityLevel::Detailed) {
                header << std::setw(12) << "Step size" << " | "
                       << std::setw(6)  << "LS evals" << " | ";
            }
            log_line(header.str());
            log_line(std::string(header.str().length(), '-'));
        }
    }

    // Strong-Wolfe line search (Nocedal & Wright, Numerical Optimization, 2nd
    // ed., Algorithms 3.5-3.6): a faithful C++ port of scipy's
    // `line_search_wolfe2` / `scalar_search_wolfe2` / `_zoom`.
    struct LsResult { bool success; Vector x_next; Vector g_next; FP f_next; FP alpha; int steps; };

    const FP c1 = FP(1e-4);   // Armijo sufficient-decrease parameter
    const FP c2 = FP(0.9);    // strong-Wolfe curvature parameter

    // Minimizer of the cubic interpolant through (a, fa, fpa), (b, fb), (c, fc)
    // (scipy `_cubicmin`). Returns NaN when no minimizer can be found.
    auto cubic_min = [](FP a, FP fa, FP fpa, FP b, FP fb, FP c, FP fc) -> FP {
        const FP db = b - a;
        const FP dc = c - a;
        const FP denom = (db * dc) * (db * dc) * (db - dc);
        if (denom == FP(0) || !std::isfinite(denom)) {
            return std::numeric_limits<FP>::quiet_NaN();
        }
        const FP rhs1 = fb - fa - fpa * db;
        const FP rhs2 = fc - fa - fpa * dc;
        const FP A = (dc * dc * rhs1 - db * db * rhs2) / denom;
        const FP B = (-(dc * dc * dc) * rhs1 + db * db * db * rhs2) / denom;
        const FP radical = B * B - FP(3.0) * A * fpa;
        if (radical < FP(0) || A == FP(0) || !std::isfinite(radical)) {
            return std::numeric_limits<FP>::quiet_NaN();
        }
        const FP xmin = a + (-B + std::sqrt(radical)) / (FP(3.0) * A);
        return std::isfinite(xmin) ? xmin : std::numeric_limits<FP>::quiet_NaN();
    };

    // Minimizer of the quadratic interpolant through (a, fa, fpa) and (b, fb)
    // (scipy `_quadmin`). Returns NaN when no minimizer can be found.
    auto quad_min = [](FP a, FP fa, FP fpa, FP b, FP fb) -> FP {
        const FP db = b - a;
        if (db == FP(0)) {
            return std::numeric_limits<FP>::quiet_NaN();
        }
        const FP B = (fb - fa - fpa * db) / (db * db);
        if (B == FP(0) || !std::isfinite(B)) {
            return std::numeric_limits<FP>::quiet_NaN();
        }
        const FP xmin = a - fpa / (FP(2.0) * B);
        return std::isfinite(xmin) ? xmin : std::numeric_limits<FP>::quiet_NaN();
    };

    auto backtrack = [&](const Vector& d, FP gtd, FP alpha0_guess) -> LsResult {
        LsResult bad{false, x, g, f_val, FP(0), 0};

        const FP phi0 = f_val;
        const FP derphi0 = gtd;

        // Last evaluated trial, kept so that a failed search can report a
        // meaningful relative-function-change value to the caller.
        int steps = 0;
        FP last_alpha = FP(0);
        FP last_phi = f_val;
        Vector last_x = x;
        Vector last_g = g;

        Vector X, G;
        FP phi_a, dphi_a;
        // Evaluate phi(alpha) = f(x + alpha*d) and phi'(alpha) = g(x + alpha*d).d,
        // caching the full gradient G of the trial point.
        auto eval = [&](FP alpha) {
            X = x + alpha * d;
            const FP fx = f(X);
            ++res.nfev;
            G = grad_eval(X);
            ++res.njev;
            phi_a = fx;
            dphi_a = G.dot(d);
            ++steps;
            last_alpha = alpha;
            last_phi = phi_a;
            last_x = X;
            last_g = G;
        };
        auto fail = [&]() -> LsResult {
            return LsResult{false, std::move(last_x), std::move(last_g), last_phi, last_alpha, steps};
        };

        // zoom(a_lo, a_hi): `a_lo` is an Armijo-satisfying point, `a_hi`
        // brackets a minimizer. Cubic interpolation with quadratic/bisection
        // fallbacks (scipy `_zoom`).
        auto zoom = [&](FP a_lo, FP a_hi, FP phi_lo, FP phi_hi, FP derphi_lo,
                        FP phi0_c, FP derphi0_c) -> LsResult {
            const int maxiter = opt.maxls;
            const FP delta1 = FP(0.2), delta2 = FP(0.1);
            FP phi_rec = phi0_c;
            FP a_rec = FP(0);
            FP a_j = FP(0);
            for (int i = 0; i <= maxiter; ++i) {
                const FP dalpha = a_hi - a_lo;
                const FP a = std::min(a_lo, a_hi);
                const FP b = std::max(a_lo, a_hi);
                const FP qchk = delta2 * dalpha;
                if (i > 0) {
                    const FP cchk = delta1 * dalpha;
                    a_j = cubic_min(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec);
                    if (!std::isfinite(a_j) || a_j > b - cchk || a_j < a + cchk) {
                        a_j = quad_min(a_lo, phi_lo, derphi_lo, a_hi, phi_hi);
                        if (!std::isfinite(a_j) || a_j > b - qchk || a_j < a + qchk) {
                            a_j = a_lo + FP(0.5) * dalpha;
                        }
                    }
                } else {
                    a_j = quad_min(a_lo, phi_lo, derphi_lo, a_hi, phi_hi);
                    if (!std::isfinite(a_j) || a_j > b - qchk || a_j < a + qchk) {
                        a_j = a_lo + FP(0.5) * dalpha;
                    }
                }
                eval(a_j);
                if (!std::isfinite(phi_a) || !std::isfinite(dphi_a)) {
                    return fail();
                }
                if (phi_a > phi0_c + c1 * a_j * derphi0_c || phi_a >= phi_lo) {
                    phi_rec = phi_hi;
                    a_rec = a_hi;
                    a_hi = a_j;
                    phi_hi = phi_a;
                } else {
                    if (std::abs(dphi_a) <= -c2 * derphi0_c) {
                        return LsResult{true, std::move(X), std::move(G), phi_a, a_j, steps};
                    }
                    if (dphi_a * (a_hi - a_lo) >= FP(0)) {
                        phi_rec = phi_hi;
                        a_rec = a_hi;
                        a_hi = a_lo;
                        phi_hi = phi_lo;
                    } else {
                        phi_rec = phi_lo;
                        a_rec = a_lo;
                    }
                    a_lo = a_j;
                    phi_lo = phi_a;
                    derphi_lo = dphi_a;
                }
                if (i == maxiter) {
                    break;
                }
            }
            return fail();
        };

        if (!(derphi0 < FP(0))) {
            return bad;
        }

        // Initial trial step (shared by wolfe1 and wolfe2): scipy's
        // `scalar_search_wolfe1`/`_2` set
        //   alpha1 = min(1, 1.01 * 2 * (f(xk) - f(xkm1)) / (gk . dk))
        // implemented via the caller-provided `alpha0_guess`.
        FP alpha1 = alpha0_guess;
        if (!(alpha1 > FP(0)) || !std::isfinite(alpha1)) {
            alpha1 = FP(1.0);
        }
        if (alpha1 > FP(1.0)) {
            alpha1 = FP(1.0);
        }

        // =====================================================================
        // Primary line search: DCSRCH from MINPACK (Moré & Thuente 1994), the
        // algorithm behind scipy's `line_search_wolfe1`. scipy's `_minimize_bfgs`
        // runs this first; the safeguarded cubic/section/bisection stepping
        // produces better-conditioned steps than plain backtracking on
        // ill-conditioned problems.
        // =====================================================================
        const FP stpmin_ = FP(1e-100);
        const FP stpmax_ = FP(1e100);
        const FP xtol1_ = FP(1e-14);
        const int wolfe1_maxiter = std::max(opt.maxls, 100);

        // DCSRCH persistent state (attributes of the Python DCSRCH class).
        int dstage = 1;
        bool dbrackt = false;
        FP dfinit = FP(0), dginit = FP(0), dgtest = FP(0);
        FP dfx = FP(0), dgx = FP(0), dfy = FP(0), dgy = FP(0);
        FP dstx = FP(0), dsty = FP(0), dstmin = FP(0), dstmax = FP(0);
        FP dwidth = FP(0), dwidth1 = FP(0);

        // dcstep: MINPACK safeguarded step (scipy `dcstep`). The dynamic
        // bracketing bounds stmin/stmax must be passed in: scipy calls
        // dcstep(..., self.stmin, self.stmax) and uses them when a cubic is
        // not available (stpc = stmax/stmin) and for the safeguarded clip of
        // stpf. Using the global [stpmin_, stpmax_] there (as an earlier port
        // did) selects different steps and derails the trajectory from scipy.
        auto dcstep = [&](FP& stx, FP& fx, FP& dx, FP& sty, FP& fy, FP& dy,
                          FP& stp, FP fp, FP dp, bool& brackt,
                          FP stmin, FP stmax) {
            const FP sgn_dp = (dp > FP(0)) ? FP(1) : (dp < FP(0)) ? FP(-1) : FP(0);
            const FP sgn_dx = (dx > FP(0)) ? FP(1) : (dx < FP(0)) ? FP(-1) : FP(0);
            const FP sgnd = sgn_dp * sgn_dx;
            FP stpf;
            if (fp > fx) {
                // Higher function value: the minimum is bracketed.
                const FP theta = FP(3.0) * (fx - fp) / (stp - stx) + dx + dp;
                const FP s = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
                FP gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
                if (stp < stx) {
                    gamma *= -1;
                }
                const FP p = (gamma - dx) + theta;
                const FP q = ((gamma - dx) + gamma) + dp;
                const FP r = p / q;
                const FP stpc = stx + r * (stp - stx);
                const FP stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / FP(2.0)) * (stp - stx);
                if (std::abs(stpc - stx) <= std::abs(stpq - stx)) {
                    stpf = stpc;
                } else {
                    stpf = stpc + (stpq - stpc) / FP(2.0);
                }
                brackt = true;
            } else if (sgnd < FP(0)) {
                // Opposite-sign derivatives: the minimum is bracketed.
                const FP theta = FP(3.0) * (fx - fp) / (stp - stx) + dx + dp;
                const FP s = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
                FP gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
                if (stp > stx) {
                    gamma *= -1;
                }
                const FP p = (gamma - dp) + theta;
                const FP q = ((gamma - dp) + gamma) + dx;
                const FP r = p / q;
                const FP stpc = stp + r * (stx - stp);
                const FP stpq = stp + (dp / (dp - dx)) * (stx - stp);
                if (std::abs(stpc - stp) > std::abs(stpq - stp)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }
                brackt = true;
            } else if (std::abs(dp) < std::abs(dx)) {
                // Same-sign derivatives with decreasing magnitude.
                const FP theta = FP(3.0) * (fx - fp) / (stp - stx) + dx + dp;
                const FP s = std::max({std::abs(theta), std::abs(dx), std::abs(dp)});
                FP gamma = s * std::sqrt(std::max(FP(0), (theta / s) * (theta / s) - (dx / s) * (dp / s)));
                if (stp > stx) {
                    gamma = -gamma;
                }
                const FP p = (gamma - dp) + theta;
                const FP q = (gamma + (dx - dp)) + gamma;
                const FP r = p / q;
                FP stpc;
                if (r < FP(0) && gamma != FP(0)) {
                    stpc = stp + r * (stx - stp);
                } else if (stp > stx) {
                    stpc = stmax;
                } else {
                    stpc = stmin;
                }
                const FP stpq = stp + (dp / (dp - dx)) * (stx - stp);
                if (brackt) {
                    if (std::abs(stpc - stp) < std::abs(stpq - stp)) {
                        stpf = stpc;
                    } else {
                        stpf = stpq;
                    }
                    if (stp > stx) {
                        stpf = std::min(stp + FP(0.66) * (sty - stp), stpf);
                    } else {
                        stpf = std::max(stp + FP(0.66) * (sty - stp), stpf);
                    }
                } else {
                    if (std::abs(stpc - stp) > std::abs(stpq - stp)) {
                        stpf = stpc;
                    } else {
                        stpf = stpq;
                    }
                    stpf = std::min(std::max(stpf, stmin), stmax);
                }
            } else {
                // Same-sign derivatives with non-decreasing magnitude.
                if (brackt) {
                    const FP theta = FP(3.0) * (fp - fy) / (sty - stp) + dy + dp;
                    const FP s = std::max({std::abs(theta), std::abs(dy), std::abs(dp)});
                    FP gamma = s * std::sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                    if (stp > sty) {
                        gamma = -gamma;
                    }
                    const FP p = (gamma - dp) + theta;
                    const FP q = ((gamma - dp) + gamma) + dy;
                    const FP r = p / q;
                    stpf = stp + r * (sty - stp);
                } else if (stp > stx) {
                    stpf = stmax;
                } else {
                    stpf = stmin;
                }
            }
            // Update the interval that contains a minimizer.
            if (fp > fx) {
                sty = stp;
                fy = fp;
                dy = dp;
            } else {
                if (sgnd < FP(0)) {
                    sty = stx;
                    fy = fx;
                    dy = dx;
                }
                stx = stp;
                fx = fp;
                dx = dp;
            }
            stp = stpf;
        };

        // One DCSRCH macro-iteration (scipy DCSRCH._iterate).
        auto dcsrch_iterate = [&](FP& stp, FP f_in, FP g_in, std::string& task) {
            if (task == "START") {
                if (stp < stpmin_ || stp > stpmax_ || g_in >= FP(0) ||
                    c1 < FP(0) || c2 < FP(0) || xtol1_ < FP(0) ||
                    stpmin_ < FP(0) || stpmax_ < stpmin_) {
                    task = "ERROR: INPUT ARGUMENTS";
                    return;
                }
                dbrackt = false;
                dstage = 1;
                dfinit = f_in;
                dginit = g_in;
                dgtest = c1 * dginit;
                dwidth = stpmax_ - stpmin_;
                dwidth1 = dwidth / FP(0.5);
                dstx = FP(0);
                dfx = dfinit;
                dgx = dginit;
                dsty = FP(0);
                dfy = dfinit;
                dgy = dginit;
                dstmin = FP(0);
                dstmax = stp + FP(4.0) * stp;
                task = "FG";
                return;
            }

            const FP ftest = dfinit + stp * dgtest;
            if (dstage == 1 && f_in <= ftest && g_in >= FP(0)) {
                dstage = 2;
            }
            // Test for warnings, then convergence (convergence overrides).
            task = "FG";
            if (dbrackt && (stp <= dstmin || stp >= dstmax)) {
                task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS";
            }
            if (dbrackt && dstmax - dstmin <= xtol1_ * dstmax) {
                task = "WARNING: XTOL TEST SATISFIED";
            }
            if (stp == stpmax_ && f_in <= ftest && g_in <= dgtest) {
                task = "WARNING: STP = STPMAX";
            }
            if (stp == stpmin_ && (f_in > ftest || g_in >= dgtest)) {
                task = "WARNING: STP = STPMIN";
            }
            if (f_in <= ftest && std::abs(g_in) <= c2 * (-dginit)) {
                task = "CONVERGENCE";
            }
            if (task != "FG") {
                return;
            }

            // First stage uses a modified function to predict the step.
            if (dstage == 1 && f_in <= dfx && f_in > ftest) {
                const FP fm = f_in - stp * dgtest;
                FP fxm = dfx - dstx * dgtest;
                FP fym = dfy - dsty * dgtest;
                const FP gm = g_in - dgtest;
                FP gxm = dgx - dgtest;
                FP gym = dgy - dgtest;
                bool b = dbrackt;
                dcstep(dstx, fxm, gxm, dsty, fym, gym, stp, fm, gm, b, dstmin, dstmax);
                dbrackt = b;
                dfx = fxm + dstx * dgtest;
                dfy = fym + dsty * dgtest;
                dgx = gxm + dgtest;
                dgy = gym + dgtest;
            } else {
                dcstep(dstx, dfx, dgx, dsty, dfy, dgy, stp, f_in, g_in, dbrackt, dstmin, dstmax);
            }

            // Decide if a bisection step is needed.
            if (dbrackt) {
                if (std::abs(dsty - dstx) >= FP(0.66) * dwidth1) {
                    stp = dstx + FP(0.5) * (dsty - dstx);
                }
                dwidth1 = dwidth;
                dwidth = std::abs(dsty - dstx);
            }
            // Set the minimum and maximum steps allowed for stp.
            if (dbrackt) {
                dstmin = std::min(dstx, dsty);
                dstmax = std::max(dstx, dsty);
            } else {
                dstmin = stp + FP(1.1) * (stp - dstx);
                dstmax = stp + FP(4.0) * (stp - dstx);
            }
            // Force the step within the bounds stpmax and stpmin.
            stp = std::min(std::max(stp, stpmin_), stpmax_);
            // If further progress is not possible, let stp be the best point.
            if ((dbrackt && (stp <= dstmin || stp >= dstmax)) ||
                (dbrackt && dstmax - dstmin <= xtol1_ * dstmax)) {
                stp = dstx;
            }
            task = "FG";
        };

        auto run_wolfe1 = [&](FP a1) -> LsResult {
            dstage = 1;
            dbrackt = false;
            dfinit = dginit = dgtest = FP(0);
            dfx = dgx = dfy = dgy = FP(0);
            dstx = dsty = dstmin = dstmax = FP(0);
            dwidth = dwidth1 = FP(0);
            std::string task = "START";
            FP stp = a1;
            FP f_here = phi0;
            FP g_here = derphi0;
            for (int it = 0; it < wolfe1_maxiter; ++it) {
                dcsrch_iterate(stp, f_here, g_here, task);
                if (task.rfind("ERROR", 0) == 0 || !std::isfinite(stp)) {
                    return fail();
                }
                if (task != "FG") {
                    break;  // CONVERGENCE or WARNING
                }
                eval(stp);
                if (!std::isfinite(phi_a) || !std::isfinite(dphi_a)) {
                    return fail();
                }
                f_here = phi_a;
                g_here = dphi_a;
            }
            if (task == "CONVERGENCE") {
                return LsResult{true, std::move(X), std::move(G), f_here, stp, steps};
            }
            // WARNING or failure: MINPACK returns stp=None; fall back to wolfe2.
            return fail();
        };

        LsResult r1 = run_wolfe1(alpha1);
        if (r1.success) {
            return r1;
        }

        // =====================================================================
        // Fallback: `scalar_search_wolfe2` + `_zoom` (pure-Python scipy, the
        // public `scipy.optimize.line_search`).
        // =====================================================================
        auto run_wolfe2 = [&](FP a1) -> LsResult {
            eval(a1);
            if (!std::isfinite(phi_a) || !std::isfinite(dphi_a)) {
                return fail();
            }

            FP alpha0 = FP(0);
            FP phi_a0 = phi0;
            FP derphi_a0 = derphi0;
            FP alpha_cur = a1;
            FP phi_a1 = phi_a;
            FP dphi_a1 = dphi_a;

            // Bracketing phase (scipy `scalar_search_wolfe2`).
            for (int i = 0; i < opt.maxls; ++i) {
                if (alpha_cur == FP(0)) {
                    return fail();
                }
                const bool not_first = (i > 0);
                if (phi_a1 > phi0 + c1 * alpha_cur * derphi0 || (not_first && phi_a1 >= phi_a0)) {
                    return zoom(alpha0, alpha_cur, phi_a0, phi_a1, derphi_a0, phi0, derphi0);
                }
                if (std::abs(dphi_a1) <= -c2 * derphi0) {
                    // Strong-Wolfe conditions satisfied.
                    return LsResult{true, std::move(X), std::move(G), phi_a1, alpha_cur, steps};
                }
                if (dphi_a1 >= FP(0)) {
                    return zoom(alpha_cur, alpha0, phi_a1, phi_a0, dphi_a1, phi0, derphi0);
                }
                // Still a descent direction at the trial point: double the step
                // and keep expanding until the minimizer is bracketed.
                const FP alpha2 = FP(2.0) * alpha_cur;
                if (!std::isfinite(alpha2) || alpha2 > FP(1e8)) {
                    return fail();
                }
                alpha0 = alpha_cur;
                alpha_cur = alpha2;
                phi_a0 = phi_a1;
                derphi_a0 = dphi_a1;
                eval(alpha_cur);
                if (!std::isfinite(phi_a) || !std::isfinite(dphi_a)) {
                    return fail();
                }
                phi_a1 = phi_a;
                dphi_a1 = dphi_a;
            }
            // maxiter reached in the bracketing loop: scipy
            // (`scalar_search_wolfe2`) ACCEPTS the last trial step (it satisfies
            // Armijo by construction; the curvature derivative is then recomputed
            // by the caller = our cached G).
            return LsResult{true, std::move(X), std::move(G), phi_a1, alpha_cur, steps};
        };
        return run_wolfe2(alpha1);
    };

    while (res.nit < opt.maxiter) {
        res.optimality = g.template lpNorm<Eigen::Infinity>();

        if (res.optimality < opt.gtol) {
            res.success = true;
            res.status = SolverStatus::Converged;
            res.message = "Optimization converged successfully (gtol satisfied).";
            break;
        }

        if (res.nfev >= max_evals) {
            res.status = SolverStatus::MaxNfevReached;
            res.message = "Maximum function evaluations reached.";
            break;
        }

        Vector d;
        if (opt.approx_type == ApproxType::InverseHessian) {
            d = -bfgs.dot(g);
        } else {
            d = bfgs.get_matrix().ldlt().solve(-g);
        }

        FP gtd = g.dot(d);

        // Non-descent direction: the Hessian approximation became indefinite
        // or corrupted. Reset to identity and retry.
        bool reset_this_iter = false;
        if (!(gtd < FP(0.0))) {
            bfgs.initialize(n, opt.approx_type);
            d = -g;
            gtd = g.dot(d);
            reset_this_iter = true;
        }

        // Initial step-length guess (scipy `scalar_search_wolfe2`):
        //   alpha1 = min(1, 1.01 * 2 * (f(xk) - f(xkm1)) / (gk . dk)).
        // Starts the line search near the prospective line minimizer instead of
        // blindly using alpha = 1.
        FP alpha0 = FP(1.0);
        if (gtd != FP(0.0)) {
            const FP est = FP(1.01) * FP(2.0) * (f_val - f_prev) / gtd;
            if (std::isfinite(est) && est > FP(0.0)) {
                alpha0 = est;
            }
        }
        if (alpha0 > FP(1.0)) {
            alpha0 = FP(1.0);
        }

        auto ls = backtrack(d, gtd, alpha0);

        if (!ls.success && !reset_this_iter) {
            bfgs.initialize(n, opt.approx_type);
            Vector d_sd = -g;
            FP gtd_sd = g.dot(d_sd);
            FP alpha0_sd = FP(1.0);
            if (gtd_sd != FP(0.0)) {
                const FP est = FP(1.01) * FP(2.0) * (f_val - f_prev) / gtd_sd;
                if (std::isfinite(est) && est > FP(0.0)) {
                    alpha0_sd = est;
                }
            }
            if (alpha0_sd > FP(1.0)) {
                alpha0_sd = FP(1.0);
            }
            auto ls_retry = backtrack(d_sd, gtd_sd, alpha0_sd);
            if (ls_retry.success) {
                d = std::move(d_sd);
                gtd = gtd_sd;
                ls = std::move(ls_retry);
                reset_this_iter = true;
            }
        }

        // Emit iteration row according to verbosity level.
        if (opt.verbose >= VerbosityLevel::Iteration) {
            std::ostringstream row;
            row << std::setw(6)  << res.nit << " | "
                << std::setw(8)  << res.nfev << " | "
                << std::scientific << std::setprecision(6)
                << std::setw(14) << f_val << " | "
                << std::setw(12) << res.optimality << " | ";

            if (opt.verbose >= VerbosityLevel::Detailed) {
                row << std::setw(12) << ls.alpha << " | "
                    << std::setw(8)  << ls.steps << " | ";
            }

            log_line(row.str());
            if (reset_this_iter && opt.verbose >= VerbosityLevel::Detailed) {
                log_line("   (Hessian reset to identity -- steepest descent for this iteration)");
            }

            if (opt.verbose >= VerbosityLevel::Debug) {
                log_line("   x = " + format_vector_preview(x));
                log_line("   g = " + format_vector_preview(g));
            }
        }

        if (!ls.success) {
            // Relative tolerance check (ftol/reltol) as in scipy / R vmmin.
            FP rel_diff = std::abs(f_val - ls.f_next) / (std::abs(f_val) + FP(1.0));
            if (res.optimality < opt.gtol * FP(100.0) || rel_diff < opt.ftol || ls.alpha < FP(1e-8)) {
                res.success = true;
                res.status = SolverStatus::Converged;
                res.message = "Optimization converged (line search step size or function delta reached floating point tolerance limit).";
            } else {
                res.status = SolverStatus::LineSearchFailed;
                res.message = "Line search failed to satisfy the strong-Wolfe conditions (even after resetting to steepest descent).";
            }
            break;
        }

        Vector s = ls.x_next - x;
        Vector y = ls.g_next - g;

        bfgs.update(s, y);

        x = std::move(ls.x_next);
        f_prev = f_val;
        f_val = ls.f_next;
        g = std::move(ls.g_next);
        res.nit++;
    }

    if (res.nit >= opt.maxiter && res.status == SolverStatus::Error) {
        res.status = SolverStatus::MaxIterReached;
        res.message = "Maximum iterations reached.";
    }

    res.x = x;
    res.fun = f_val;
    res.jac = g;
    res.hessian_approx = bfgs.get_matrix();
    res.optimality = g.template lpNorm<Eigen::Infinity>();

    // Final result logging (Summary and above).
    if (opt.verbose != VerbosityLevel::Silent) {
        auto end_time = Clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::ostringstream ss;
        ss << "========================================================================\n"
           << " Optimization Summary:\n"
           << "   Status:      " << res.message << "\n"
           << "   Final f(x):  " << std::scientific << std::setprecision(8) << res.fun << "\n"
           << "   Optimality:  " << std::scientific << std::setprecision(4) << res.optimality << "\n"
           << "   Iterations:  " << res.nit << "\n"
           << "   Func Evals:  " << res.nfev << "\n"
           << "   Grad Evals:  " << res.njev << "\n"
           << "   Time Elapsed:" << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n"
           << "========================================================================";
        log_line(ss.str());
    }

    return res;
}

} // namespace min

#endif // MIN_HESSIAN_UPDATE_HPP
