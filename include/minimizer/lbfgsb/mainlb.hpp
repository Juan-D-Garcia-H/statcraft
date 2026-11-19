// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_MAINLB_HPP
#define LBFGSB_MAINLB_HPP

/**
 * @file mainlb.hpp
 * @ingroup lbfgsb_driver
 * @brief Reverse-communication core driver of the L-BFGS-B solver.
 *
 * @details
 * Orchestrates one full step of the box-constrained limited-memory BFGS
 * algorithm: input validation (@ref errclb) and bound initialization
 * (@ref active), the generalized Cauchy point (@ref cauchy), free/active-set
 * bookkeeping (@ref freev), the factorization machinery (@ref formt, @ref
 * formk), the subspace residual and truncated-Newton step (@ref cmprlb,
 * @ref subsm), the feasibility-aware line search (@ref lnsrlb), and the
 * L-BFGS history update (@ref matupd), interleaved with convergence tests on
 * the projected gradient and on the relative reduction of the objective.
 *
 * It is a state machine driven through @ref SolverState::task: the caller
 * performs objective/gradient evaluations on demand (`EvaluateFG_*`) and
 * re-enters; the driver returns control after each such request.
 *
 * When a sub-solver reports a numerical failure (cauchy/formk/formt/subsm/
 * lnsrlb), the routine refreshes the L-BFGS memory (or, in the first-iteration
 * line-search case, terminates with `ErrorCode::LineSearchFailed`).
 *
 * See `_lbfgsb` in lbfgsb.hpp for the public front-end that services the
 * evaluation requests.
 */


#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "../common.hpp"
#include "state.hpp"
#include "logger.hpp"

#include "active.hpp"
#include "cauchy.hpp"
#include "cmprlb.hpp"
#include "errclb.hpp"
#include "formk.hpp"
#include "formt.hpp"
#include "freev.hpp"
#include "lnsrlb.hpp"
#include "matupd.hpp"
#include "projgr.hpp"
#include "subsm.hpp"

namespace min {

/**
 * @brief Advance the L-BFGS-B state machine by one step.
 *
 * @details
 * The task token in @p st decides the branch:
 * - **`Task::Start`**: model setup. Machine epsilon and all counters are
 *   initialized, @ref errclb validates the input, the header is printed, and
 *   @ref active clips the initial point and classifies its bound status. The
 *   routine then requests the first function/gradient evaluation
 *   (`EvaluateFG_Start`).
 * - **`Task::EvaluateFG_Start`** (first evaluation): computes the projected
 *   gradient norm @ref SolverState::sbgnrm and tests
 *   \f$\mathrm{sbgnrm} \le \mathrm{pgtol}\f$ (immediate convergence).
 * - **`Task::NewX`**: convergence analysis after a new iterate. Tests
 *   \f$\mathrm{sbgnrm} \le \mathrm{pgtol}\f$ and the relative function change
 *   \f$(\mathrm{fold}-f) \le \mathrm{factr}\,\mathrm{eps}\,\max(|\mathrm{fold}|,|f|,1)\f$;
 *   if more than 10 evaluations were used in the final line search, the latter
 *   is reported with `ErrorCode::LineSearchMaxFeval`. Otherwise the L-BFGS
 *   history is updated (gradient difference \f$y = g-g_\mathrm{old}\f$,
 *   curvature \f$s^Ty\f$ test, @ref matupd then @ref formt); skips with a log
 *   message when the curvature condition fails, and refreshes the memory when
 *   @ref formt reports a non-positive-definite factor.
 * - **Main solver branch** (any other task): computes the search direction —
 *   for unconstrained problems with stored history the "Newton point" is just
 *   \f$x\f$; otherwise @ref cauchy produces the GCP and @ref freev the free
 *   index set, then @ref formk/@ref cmprlb/@ref subsm build and solve the
 *   subspace problem, and finally \f$d = z - x\f$. Numerical failures at any
 *   sub-step discard the memory and retry.
 * - **Line search**: @ref lnsrlb drives @ref dcsrch. On `info != 0` or
 *   \f$\mathrm{iback} \ge 20\f$ the iterate is restored (`x = t`, `g = r`,
 *   `f = fold`): if no history exists this is terminal
 *   (`ErrorCode::LineSearchFailed`, `Task::Abnormal`), otherwise the memory is
 *   refreshed and the direction recomputed. If the line search requests an
 *   evaluation, control returns with `EvaluateFG_LineSearch`. Otherwise the
 *   new iterate is accepted (`Task::NewX`).
 *
 * @tparam FP      Floating-point scalar type.
 * @param n        Problem dimension.
 * @param m        L-BFGS memory size (number of stored correction pairs).
 * @param x        In/out: current iterate.
 * @param l        Lower bounds (length @p n).
 * @param u        Upper bounds (length @p n).
 * @param nbd      Per-variable bound kind, see @ref BoundType.
 * @param f        In/out: objective value at @p x.
 * @param g        In/out: gradient at @p x.
 * @param factr    Stopping-tolerance factor (\f$\ge 0\f$); combined with the
 *                 machine epsilon into the working tolerance
 *                 \f$\mathrm{tol} = \mathrm{factr}\,\mathrm{eps}\f$.
 * @param pgtol    Convergence threshold for the projected-gradient norm.
 * @param ws_state Workspace matrices/vectors (see @ref Workspace).
 * @param st       Persistent solver state (see @ref SolverState); carries the
 *                 task token and all counters in/out.
 * @param logger   Logger sink for header/iteration/summary/info output.
 */


template<typename FP = double>
inline void mainlb(int n,
                   int m,
                   Vec<FP>& x,
                   const Vec<FP>& l,
                   const Vec<FP>& u,
                   const Vec<BoundType>& nbd,
                   FP& f,
                   Vec<FP>& g,
                   FP factr,
                   FP pgtol,
                   Workspace<FP>& ws_state,
                   SolverState<FP>& st,
                   const Logger<FP>& logger)
{
    auto now_sec = []() -> FP {
        using namespace std::chrono;
        return duration<FP>(steady_clock::now().time_since_epoch()).count();
    };

    // ── 1. START ─────────────────────────────────────────────────────────────
    if (st.task == Task::Start) {
        st.epsmch = std::numeric_limits<FP>::epsilon();
        st.time1  = now_sec();

        st.col    = 0;   st.head   = 0;   st.theta  = FP(1.0);
        st.iupdat = 0;   st.updated = false;
        st.iback  = 0;   st.itail  = 0;   st.subspace_status = SubspaceStatus::Constant;
        st.nact   = 0;   st.ileave = 0;   st.nenter = 0;
        st.fold   = FP(0.0); st.dnorm  = FP(0.0); st.cpu1   = FP(0.0);
        st.gd     = FP(0.0); st.stpmx  = FP(0.0); st.sbgnrm = FP(0.0);
        st.stp    = FP(0.0); st.gdold  = FP(0.0); st.dtd    = FP(0.0);

        st.iter   = 0;   st.nfgv   = 0;   st.nseg   = 0;
        st.nintol = 0;   st.nskip  = 0;   st.nfree  = n;   st.ifun = 0;

        st.tol    = factr * st.epsmch;
        st.cachyt = FP(0.0); st.sbtime = FP(0.0); st.lnscht = FP(0.0);
        st.error_code = ErrorCode::None;

        errclb(n, m, factr, l, u, nbd, st.task, st.error_code, st.k);
        if (st.task == Task::Error) {
            FP t_total = now_sec() - st.time1;
            logger.print_summary(n, st.iter, st.nfgv, st.nact, st.sbgnrm, f,
                                 st.error_code, "Error during setup verification",
                                 t_total, st.cachyt, st.sbtime, st.lnscht);
            return;
        }

        logger.print_header(n, m, st.epsmch, l, u, x);
        active(n, l, u, nbd, x, ws_state.iwhere, logger, st.projected, st.constrained, st.boxed);

        st.task = Task::EvaluateFG_Start;
        return;
    }

    // ── 2. RESUMPTION HANDLING ───────────────────────────────────────────────
    if (st.task == Task::EvaluateFG_Start) {
        if (st.iter == 0) {
            st.nfgv = 1;
            projgr(n, l, u, nbd, x, g, st.sbgnrm);

            logger.print_table_header();
            logger.print_iteration(0, st.nfgv, st.nact, 0, SubspaceStatus::Constant, 0, FP(0), FP(0), st.sbgnrm, f, x, g);

            if (st.sbgnrm <= pgtol) {
                st.task = Task::Convergence;
                FP t_total = now_sec() - st.time1;
                logger.print_summary(n, st.iter, st.nfgv, st.nact, st.sbgnrm, f,
                                     ErrorCode::None, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
                                     t_total, st.cachyt, st.sbtime, st.lnscht);
                return;
            }
        }
    } else if (st.task == Task::NewX) {
        // Test convergence after NewX
        if (st.sbgnrm <= pgtol) {
            st.task = Task::Convergence;
            FP t_total = now_sec() - st.time1;
            logger.print_summary(n, st.iter, st.nfgv, st.nact, st.sbgnrm, f,
                                 ErrorCode::None, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
                                 t_total, st.cachyt, st.sbtime, st.lnscht);
            return;
        }

        FP ddum = std::max({std::abs(st.fold), std::abs(f), FP(1.0)});
        if ((st.fold - f) <= st.tol * ddum) {
            st.task = Task::Convergence;
            if (st.iback >= 10) st.error_code = ErrorCode::LineSearchMaxFeval;
            FP t_total = now_sec() - st.time1;
            logger.print_summary(n, st.iter, st.nfgv, st.nact, st.sbgnrm, f,
                                 st.error_code, "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
                                 t_total, st.cachyt, st.sbtime, st.lnscht);
            return;
        }

        ws_state.r = g - ws_state.r;
        FP rr = ws_state.r.dot(ws_state.r);

        FP dr;
        if (st.stp == FP(1.0)) {
            dr   = st.gd - st.gdold;
            ddum = -st.gdold;
        } else {
            dr   = (st.gd - st.gdold) * st.stp;
            ws_state.d *= st.stp;
            ddum = -st.gdold * st.stp;
        }

        if (dr <= st.epsmch * ddum) {
            ++st.nskip;
            st.updated = false;
            logger.print_info_message("BFGS matrix update skipped due to insufficient curvature.");
        } else {
            st.updated = true;
            ++st.iupdat;

            matupd(n, m, ws_state.ws, ws_state.wy, ws_state.sy, ws_state.ss,
                   ws_state.d, ws_state.r, st.itail, st.iupdat, st.col, st.head,
                   st.theta, rr, dr, st.stp, st.dtd);

            int info = 0;
            formt(m, ws_state.wt, ws_state.sy, ws_state.ss, st.col, st.theta, info);
            if (info != 0) {
                logger.print_info_message("Non-positive Cholesky in formt; refreshing memory.");
                info = 0; st.col = 0; st.head = 0; st.theta = FP(1.0);
                st.iupdat = 0; st.updated = false;
            }
        }
    }

    // ── 3. MAIN SOLVER STEP ──────────────────────────────────────────────────
    while (true) {
        if (st.task != Task::EvaluateFG_LineSearch) {
            st.subspace_status = SubspaceStatus::Unknown;

            if (!st.constrained && st.col > 0) {
                ws_state.z = x;
                st.wrk    = st.updated;
                st.nseg   = 0;
            } else {
                // Generalized Cauchy Point
                FP t_c1 = now_sec();
                int info = 0;
                cauchy(n, x, l, u, nbd, g, ws_state.indx2, ws_state.iwhere, ws_state.t, ws_state.d, ws_state.z,
                       m, ws_state.wy, ws_state.ws, ws_state.sy, ws_state.wt, st.theta, st.col, st.head,
                       ws_state.p_wa, ws_state.c_wa, ws_state.wbp, ws_state.v_wa, st.nseg,
                       logger, st.sbgnrm, info, st.epsmch);

                if (info != 0) {
                    logger.print_info_message("Singular system in Cauchy search; refreshing memory.");
                    info = 0; st.col = 0; st.head = 0; st.theta = FP(1.0);
                    st.iupdat = 0; st.updated = false;
                    st.cachyt += now_sec() - t_c1;
                    continue;
                }
                st.cachyt += now_sec() - t_c1;
                st.nintol += st.nseg;

                freev(n, st.nfree, ws_state.index, st.nenter, st.ileave, ws_state.indx2,
                      ws_state.iwhere, st.wrk, st.updated, st.constrained, st.iter);
                st.nact = n - st.nfree;
            }

            if (st.nfree > 0 && st.col > 0) {
                FP t_s1 = now_sec();
                int info = 0;
                if (st.wrk) {
                    formk(n, st.nfree, ws_state.index, st.nenter, st.ileave, ws_state.indx2, st.iupdat,
                          st.updated, ws_state.wn, ws_state.snd, m, ws_state.ws, ws_state.wy, ws_state.sy,
                          st.theta, st.col, st.head, info);
                }

                if (info != 0) {
                    logger.print_info_message("Non-positive Cholesky in formk; refreshing memory.");
                    info = 0; st.col = 0; st.head = 0; st.theta = FP(1.0);
                    st.iupdat = 0; st.updated = false;
                    st.sbtime += now_sec() - t_s1;
                    continue;
                }

                cmprlb(n, m, x, g, ws_state.ws, ws_state.wy, ws_state.sy, ws_state.wt,
                       ws_state.z, ws_state.r, ws_state.c_wa, ws_state.wa8m, ws_state.index, st.theta,
                       st.col, st.head, st.nfree, st.constrained, info);

                if (info == 0) {
                    subsm(n, m, st.nfree, ws_state.index, l, u, nbd, ws_state.z, ws_state.r,
                          ws_state.xp, ws_state.ws, ws_state.wy, st.theta, x, g,
                          st.col, st.head, st.subspace_status, ws_state.wa8m, ws_state.wn, info);
                }

                if (info != 0) {
                    logger.print_info_message("Singular triangular system in subsm; refreshing memory.");
                    info = 0; st.col = 0; st.head = 0; st.theta = FP(1.0);
                    st.iupdat = 0; st.updated = false;
                    st.sbtime += now_sec() - t_s1;
                    continue;
                }
                st.sbtime += now_sec() - t_s1;
            }

            ws_state.d = ws_state.z - x;
        }

        // Perform or resume line search
        FP t_l1 = now_sec();
        int info = 0;
        lnsrlb(n, l, u, nbd, x, f, st.fold, st.gd, st.gdold, g,
               ws_state.d, ws_state.r, ws_state.t, ws_state.z,
               st.stp, st.dnorm, st.dtd, st.xstep, st.stpmx,
               st.iter, st.ifun, st.iback, st.nfgv, info,
               st.task, st.boxed, st.constrained, st.ls_state);

        st.lnscht += now_sec() - t_l1;

        if (info != 0 || st.iback >= 20) {
            x = ws_state.t; g = ws_state.r; f = st.fold;
            if (st.col == 0) {
                if (info == 0) info = -9;
                st.error_code = ErrorCode::LineSearchFailed;
                st.task = Task::Abnormal;
                ++st.iter;
                FP t_total = now_sec() - st.time1;
                logger.print_summary(n, st.iter, st.nfgv, st.nact, st.sbgnrm, f,
                                     st.error_code, "ABNORMAL TERMINATION IN LINE SEARCH",
                                     t_total, st.cachyt, st.sbtime, st.lnscht);
                return;
            } else {
                logger.print_info_message("Bad direction in line search; resetting memory and restarting search direction.");
                if (info == 0) --st.nfgv;
                info = 0; st.col = 0; st.head = 0; st.theta = FP(1.0);
                st.iupdat = 0; st.updated = false;
                st.task = Task::EvaluateFG_Start;
                continue;
            }
        } else if (st.task == Task::EvaluateFG_LineSearch) {
            return;
        } else {
            // Line search complete -> New iterate found
            ++st.iter;
            projgr(n, l, u, nbd, x, g, st.sbgnrm);

            logger.print_iteration(st.iter, st.nfgv, st.nact, st.nseg,
                                   st.subspace_status, st.iback, st.stp, st.xstep,
                                   st.sbgnrm, f, x, g);
            st.task = Task::NewX;
            return;
        }
    }
}

} // namespace min

#endif // LBFGSB_MAINLB_HPP
