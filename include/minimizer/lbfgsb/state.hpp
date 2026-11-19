// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

#ifndef LBFGSB_STATE_HPP
#define LBFGSB_STATE_HPP

#include "../common.hpp"

/**
 * @file state.hpp
 * @ingroup lbfgsb_driver
 * @brief Persistent solver state, line-search state and dense workspace.
 *
 * @details
 * Strongly-typed replacements for the historical Fortran workspace arrays of
 * L-BFGS-B:
 * - @ref LineSearchState replaces `csave`/`isave_ls`/`dsave_ls` and carries
 *   everything @ref dcsrch needs to resume across evaluations;
 * - @ref SolverState replaces the integer/real/logical saves of the driver
 *   (`isave`/`dsave`/`lsave`) and holds every counter, flag and diagnostic of
 *   one solve;
 * - @ref Workspace replaces the low-level `wy`, `ws`, `sy`, `ss`, `wt`, `wn`,
 *   `snd` matrix bank plus the `z`-..-`wa` vector bank of the original driver.
 *
 * An instance of each is created once per solve and threaded through @ref
 * mainlb; call @ref Workspace::resize once before solving and @ref
 * LineSearchState::reset before every new line search.
 */

namespace min {

template<typename FP = double>
struct LineSearchState {
    LineSearchTask task = LineSearchTask::Start; ///< Current phase of the line search.
    int  stage = 0;                              ///< Internal stage counter.
    bool bracket = false;                        ///< True once a valid bracket has been established.
    int  infoc = 0;                              ///< Legacy status code (kept for compatibility; written only by reset()).

    FP ginit  = FP(0); ///< Initial directional derivative at step 0.
    FP gtest  = FP(0); ///< Sufficient-decrease threshold, gtest = ftol * ginit.
    FP fx     = FP(0); ///< Objective value at the current best point.
    FP fx1    = FP(0); ///< Objective value at stx for the modified-function work (initially f(0)).
    FP fy     = FP(0); ///< Objective value at the trial point.
    FP fy1    = FP(0); ///< Legacy field (kept for compatibility; written only by reset()).
    FP gx     = FP(0); ///< Directional derivative at the current best point.
    FP gx1    = FP(0); ///< Legacy field (kept for compatibility; written only by reset()).
    FP gy     = FP(0); ///< Directional derivative at the trial point.
    FP gy1    = FP(0); ///< Legacy field (kept for compatibility; written only by reset()).
    FP stx    = FP(0); ///< Current best step length.
    FP sty    = FP(0); ///< Trial step length.
    FP stmin  = FP(0); ///< Minimum allowed step length.
    FP stmax  = FP(0); ///< Maximum allowed step length.
    FP width  = FP(0); ///< Width of the current bracket.
    FP width1 = FP(0); ///< Width of the previous bracket.

    /**
     * @brief Reset all fields to their initial values.
     *
     * Sets the task back to @ref LineSearchTask::Start and clears every
     * scalar field. Call this before starting a fresh line search.
     */
    void reset() {
        task = LineSearchTask::Start;
        stage = 0;
        bracket = false;
        infoc = 0;
        ginit = gtest = fx = fx1 = fy = fy1 = FP(0);
        gx = gx1 = gy = gy1 = stx = sty = stmin = stmax = width = width1 = FP(0);
    }
};

/**
 * @brief Main solver state for the L-BFGS-B driver.
 *
 * Replaces the historical Fortran integer and real workspace arrays
 * (`isave`, `dsave`) plus the logical flags array (`lsave`) with a
 * strongly-typed C++ object. An instance is created once per solve and
 * threaded through every call to the main driver routine.
 *
 * The struct is intentionally plain-old-data: no invariants are enforced
 * here, and callers are expected to set fields directly when a routine
 * needs to communicate a status back to the driver.
 */
template<typename FP = double>
struct SolverState {
    // ── Flags ───────────────────────────────────────────────────────────────
    bool projected   = false; ///< True if the initial point had to be clipped onto the feasible bounds (infeasible x0); set by active().
    bool constrained = false; ///< True if any variable has a lower or upper bound.
    bool boxed       = false; ///< True if EVERY variable has both a lower and an upper bound; set by active().
    bool updated     = false; ///< True if the L-BFGS memory was updated this iteration.
    bool wrk         = false; ///< True if the subspace workspace wn/wn1 must be rebuilt; set by freev().

    // ── Integer Counters & Indices ─────────────────────────────────────────
    int iter   = 0; ///< Current iteration index.
    int nfgv   = 0; ///< Cumulative number of function/gradient evaluations.
    int nintol = 0; ///< Cumulative number of Cauchy-point segments processed across all iterations.
    int nskip  = 0; ///< Number of skipped updates in the L-BFGS memory.
    int head   = 0; ///< Circular-buffer head index in the L-BFGS memory.
    int col    = 0; ///< Number of active correction pairs; equals min(iupdat, m) after an update.
    int itail  = 0; ///< Circular-buffer tail index in the L-BFGS memory.
    int iupdat = 0; ///< Total number of L-BFGS updates performed so far (may exceed m; the history then wraps).
    int nseg   = 0; ///< Number of segments in the Cauchy-point search.
    int ifun   = 0; ///< Counter used inside the line search.
    int nfree  = 0; ///< Number of currently free (non-active) variables.
    int nact   = 0; ///< Number of currently active bound constraints.
    int ileave = 0; ///< Start index of the leaving-variable segment in indx2 after freev() (n - number of leavers).
    int nenter = 0; ///< Number of variables entering the free set this iteration (bound -> free).
    int iback  = 0; ///< Number of backtracking steps performed in the last line search.
    int k      = 0; ///< Index of the last erroneous parameter when errclb() fails.

    // ── Floating-Point Metrics ─────────────────────────────────────────────
    FP theta  = FP(1.0); ///< Scaling factor for the initial Hessian approximation.
    FP fold   = FP(0.0); ///< Objective value at the previous iterate.
    FP tol    = FP(0.0); ///< Working tolerance (derived from factr * epsmch).
    FP dnorm  = FP(0.0); ///< Euclidean norm of the search direction.
    FP epsmch = FP(0.0); ///< Machine epsilon.
    FP cpu1   = FP(0.0); ///< Legacy CPU timestamp (kept for compatibility; unused by the current driver).
    FP cachyt = FP(0.0); ///< Accumulated time spent in the Cauchy-point search (seconds).
    FP sbtime = FP(0.0); ///< Accumulated time spent in subspace minimization (seconds).
    FP lnscht = FP(0.0); ///< Accumulated time spent in line search (seconds).
    FP time1  = FP(0.0); ///< Start-of-iteration CPU timestamp.
    FP gd     = FP(0.0); ///< Current directional derivative gᵀd.
    FP stpmx  = FP(0.0); ///< Maximum allowed line-search step length.
    FP sbgnrm = FP(0.0); ///< Infinity-norm of the projected gradient.
    FP stp    = FP(0.0); ///< Line-search step length from the last call.
    FP gdold  = FP(0.0); ///< Previous directional derivative gᵀd.
    FP dtd    = FP(0.0); ///< Inner product dᵀd of the search direction.
    FP xstep  = FP(0.0); ///< Norm of the last parameter update ||dx||.

    // ── Enums ──────────────────────────────────────────────────────────────
    Task task = Task::Start;                                  ///< Current phase requested by the driver.
    SubspaceStatus subspace_status = SubspaceStatus::Unknown; ///< Status reported by subspace minimization.
    ErrorCode error_code = ErrorCode::None;                   ///< Diagnostic code (any error encountered).
    LineSearchState<FP> ls_state;                             ///< Persistent state for the More-Thuente line search.
};

/**
 * @brief Workspace matrices and vectors for the L-BFGS-B driver.
 *
 * Groups every dense matrix and vector the algorithm needs, sized cleanly
 * for the problem dimension n and the L-BFGS memory size m. The layout is
 * a direct translation of the L-BFGS-B Fortran `WORK` array, but with
 * descriptive names, dedicated segment buffers, and zero-initialized
 * storage (which is convenient for debugging and avoids surprises when a
 * buffer is used before being fully written).
 *
 * Call @ref resize once before the first solve to allocate all members
 * with the correct dimensions.
 */
template<typename FP = double>
struct Workspace {
    Mat<FP> ws;   ///< n x m: history of s vectors (parameter differences).
    Mat<FP> wy;   ///< n x m: history of y vectors (gradient differences).
    Mat<FP> sy;   ///< m x m: matrix of sᵀy inner products.
    Mat<FP> ss;   ///< m x m: matrix of sᵀs inner products.
    Mat<FP> wt;   ///< m x m: upper-triangular Cholesky factor J' of (theta*S'S + L D^-1 L'); built by formt().
    Mat<FP> wn;   ///< 2m x 2m: LEL^T factorization of the indefinite subspace matrix K; built by formk().
    Mat<FP> snd;  ///< 2m x 2m: cached wn1 sub-blocks of K enabling incremental updates; built by formk().

    Vec<FP> z;    ///< n: generalized Cauchy point / subspace result ("Newton point").
    Vec<FP> r;    ///< n: gradient-difference and subspace-residual buffer.
    Vec<FP> d;    ///< n: search direction.
    Vec<FP> t;    ///< n: iterate snapshot for line-search failure restoration.
    Vec<FP> xp;   ///< n: copy of the Cauchy point safeguarding the truncated Newton direction in subsm().
    Vec<FP> wa8m; ///< 8m: workspace shared by cmprlb() (bmv result) and subsm() (wv solve buffer).

    VecInt index;  ///< n: free/active partition of coordinate indices used by freev()/subsm()/cmprlb().
    VecInt iwhere; ///< n: per-variable bound/free classification maintained by active() and refined by cauchy().
    VecInt indx2;  ///< n: breakpoint ordering and entering/leaving records (iorder/indx2).

    // ── Temporary Segment Buffers for the Cauchy Routine ───────────────────
    Vec<FP> p_wa; ///< 2m: segment buffer for the Cauchy point (p component).
    Vec<FP> c_wa; ///< 2m: segment buffer for the Cauchy point (c component).
    Vec<FP> wbp;  ///< 2m: segment buffer for the Cauchy point (w·bᵀ product).
    Vec<FP> v_wa; ///< 2m: segment buffer for the Cauchy point (v component).

    /**
     * @brief Allocate and zero-initialize every member for the given dimensions.
     *
     * After this call, all matrices have the shapes documented above and all
     * vectors and matrices are zero-filled. It is safe to call @ref resize
     * repeatedly (for example, when restarting a solve with different
     * dimensions); existing contents are discarded.
     *
     * @param n Problem dimension (number of variables).
     * @param m L-BFGS memory size (number of correction pairs).
     */
    void resize(int n, int m) {
        ws  = Mat<FP>::Zero(n, m);
        wy  = Mat<FP>::Zero(n, m);
        sy  = Mat<FP>::Zero(m, m);
        ss  = Mat<FP>::Zero(m, m);
        wt  = Mat<FP>::Zero(m, m);
        wn  = Mat<FP>::Zero(2 * m, 2 * m);
        snd = Mat<FP>::Zero(2 * m, 2 * m);

        z    = Vec<FP>::Zero(n);
        r    = Vec<FP>::Zero(n);
        d    = Vec<FP>::Zero(n);
        t    = Vec<FP>::Zero(n);
        xp   = Vec<FP>::Zero(n);
        wa8m = Vec<FP>::Zero(8 * m);

        index  = VecInt::Zero(n);
        iwhere = VecInt::Zero(n);
        indx2  = VecInt::Zero(n);

        p_wa = Vec<FP>::Zero(2 * m);
        c_wa = Vec<FP>::Zero(2 * m);
        wbp  = Vec<FP>::Zero(2 * m);
        v_wa = Vec<FP>::Zero(2 * m);
    }
};

} // namespace min

#endif // LBFGSB_STATE_HPP