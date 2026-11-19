// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_DCSRCH_HPP
#define LBFGSB_DCSRCH_HPP

/**
 * @file dcsrch.hpp
 * @ingroup lbfgsb_linesearch
 * @brief Huber-Moré-Thuente safeguarded line search (strong Wolfe conditions).
 *
 * @details
 * Implements the MINPACK-2 `dcsrch` algorithm: given a trial step @p stp and
 * its function value/derivative, it updates a bracketing interval and either
 * reports convergence (strong Wolfe conditions satisfied), requests a new
 * function/gradient evaluation, or warns that the search cannot make progress.
 *
 * The search is driven by the modified function
 * \f$\psi(\sigma) = f(\sigma) - f(0) - \mathrm{ftol}\,\sigma f'(0)\f$ (stage 1);
 * once a point with \f$\psi \le 0\f$ and \f$f' \ge 0\f$ is found, the search
 * switches to minimizer-bracketing of \f$f\f$ itself (stage 2).
 *
 * Derived from the classic MINPACK-1 project (More' & Thuente, 1983) and the
 * MINPACK-2 project (Averick, Carter & More', 1993).
 *
 * @see dcstep.hpp (interval update), lnsrlb.hpp (caller), LineSearchState
 */


#include <algorithm>
#include <cmath>
#include "dcstep.hpp"
#include "../common.hpp"
#include "state.hpp"

namespace min {

/**
 * @brief Advance the More–Thuente line search by one function/gradient evaluation.
 *
 * @details
 * Reverse-communication protocol: the caller provides the value/derivative
 * pair (\f$f, g\f$) at the current trial step @p stp and inspects @p task on
 * return. On `LineSearchTask::EvaluateFG` it must evaluate the objective and
 * its directional derivative at the *new* @p stp and re-enter; repeat until
 * `Convergence`, `Warning` or `Error`.
 *
 * - **Start**: the arguments are validated (`stp` within `[stpmin, stpmax]`,
 *   \f$g < 0\f$, non-negative tolerances, feasible bounds) and the persistent
 *   @p ls_state is initialized. Internal constants: sufficient-decrease test
 *   \f$\mathrm{gtest} = \mathrm{ftol}\,g(0)\f$, initial interval width scaled
 *   by \f$1/0.5\f$, max step \f$\mathrm{stmax} = \mathrm{stp}(1+4)\f$.
 * - **Iteration**: computes \f$\mathrm{ftest} = f(0) + \mathrm{stp}\,\mathrm{gtest}\f$
 *   and advances `stage` from 1 to 2 once \f$f \le \mathrm{ftest}\f$ and
 *   \f$g \ge 0\f$. Convergence is declared when
 *   \f$f \le \mathrm{ftest}\f$ and \f$|g| \le \mathrm{gtol}|g(0)|\f$.
 *   Warnings fire on interval collapse (\f$\mathrm{stmax}-\mathrm{stmin} \le
 *   \mathrm{xtol}\,\mathrm{stmax}\f$ when bracketed), leaving the bracket
 *   range, or hitting `stpmax`/`stpmin` without enough decrease. The interval
 *   is updated by @ref dcstep (on the modified function in stage 1, on \f$f\f$
 *   in stage 2) and the new trial step is safeguarded into
 *   \f$[\mathrm{stpmin},\mathrm{stpmax}]\f$. A final safeguard drops the step
 *   back to \f$\mathrm{stp} = \mathrm{stx}\f$ whenever the bracketed interval
 *   collapsed or the trial step left the bracket.
 *
 * @note The user must not alter @p ls_state between calls of the same line
 *       search; it persists all bracketing state.
 *
 * @note The driver @ref lnsrlb fixes \f$\mathrm{ftol}=10^{-3}\f$,
 *       \f$\mathrm{gtol}=0.9\f$, \f$\mathrm{xtol}=0.1\f$; the defaults of the
 *       classic FORTRAN driver are reproduced here so no external tuning is
 *       exposed.
 *
 * @tparam FP     Floating-point scalar type.
 * @param f       In: objective value at @p stp (or at 0 on Start).
 *                Out: unchanged.
 * @param g       In: directional derivative at @p stp (or at 0 on Start).
 *                Out: unchanged.
 * @param stp     In/out: trial step; updated to the next trial point on
 *                `EvaluateFG`, the accepted step on `Convergence`, the best
 *                step found on `Warning`.
 * @param ftol    Nonnegative sufficient-decrease tolerance.
 * @param gtol    Nonnegative curvature tolerance.
 * @param xtol    Nonnegative relative bracket-width tolerance for warnings.
 * @param stpmin  Nonnegative lower bound for the step.
 * @param stpmax  Upper bound for the step (>= @p stpmin).
 * @param task    In/out: `LineSearchTask::Start` on first entry; on exit
 *                one of `EvaluateFG`, `Convergence`, `Warning` or `Error`.
 * @param ls_state Persistent state structure holding the bracket, stage, and
 *                saved values (see @ref LineSearchState).
 */


template<typename FP = double>
inline void dcsrch(FP& f,
                   FP& g,
                   FP& stp,
                   FP  ftol,
                   FP  gtol,
                   FP  xtol,
                   FP  stpmin,
                   FP  stpmax,
                   LineSearchTask& task,
                   LineSearchState<FP>& ls_state)
{
    constexpr FP zero = FP(0.0), p5 = FP(0.5), p66 = FP(0.66), xtrapl = FP(1.1), xtrapu = FP(4.0);

    if (task == LineSearchTask::Start) {
        if (stp    < stpmin) { task = LineSearchTask::Error; return; }
        if (stp    > stpmax) { task = LineSearchTask::Error; return; }
        if (g      >= zero ) { task = LineSearchTask::Error; return; }
        if (ftol   < zero  ) { task = LineSearchTask::Error; return; }
        if (gtol   < zero  ) { task = LineSearchTask::Error; return; }
        if (xtol   < zero  ) { task = LineSearchTask::Error; return; }
        if (stpmin < zero  ) { task = LineSearchTask::Error; return; }
        if (stpmax < stpmin) { task = LineSearchTask::Error; return; }

        ls_state.bracket = false;
        ls_state.stage   = 1;
        ls_state.fx      = f; // finit
        ls_state.ginit   = g;
        ls_state.gtest   = ftol * ls_state.ginit;

        ls_state.width  = stpmax - stpmin;
        ls_state.width1 = ls_state.width / p5;

        ls_state.stx = zero;
        ls_state.fx1 = f;
        ls_state.gx  = ls_state.ginit;

        ls_state.sty = zero;
        ls_state.fy  = f;
        ls_state.gy  = ls_state.ginit;

        ls_state.stmin = zero;
        ls_state.stmax = stp + xtrapu * stp;

        task = LineSearchTask::EvaluateFG;
        return;
    }

    const FP finit = ls_state.fx;
    const FP ftest = finit + stp * ls_state.gtest;

    if (ls_state.stage == 1 && f <= ftest && g >= zero) {
        ls_state.stage = 2;
    }

    if (ls_state.bracket && (stp <= ls_state.stmin || stp >= ls_state.stmax)) {
        task = LineSearchTask::Warning;
    }
    if (ls_state.bracket && (ls_state.stmax - ls_state.stmin) <= xtol * ls_state.stmax) {
        task = LineSearchTask::Warning;
    }
    if (stp == stpmax && f <= ftest && g <= ls_state.gtest) {
        task = LineSearchTask::Warning;
    }
    if (stp == stpmin && (f > ftest || g >= ls_state.gtest)) {
        task = LineSearchTask::Warning;
    }
    if (f <= ftest && std::abs(g) <= gtol * (-ls_state.ginit)) {
        task = LineSearchTask::Convergence;
    }

    if (task == LineSearchTask::Warning || task == LineSearchTask::Convergence) {
        return;
    }

    if (ls_state.stage == 1 && f <= ls_state.fx1 && f > ftest) {
        FP fm  = f - stp * ls_state.gtest;
        FP fxm = ls_state.fx1 - ls_state.stx * ls_state.gtest;
        FP fym = ls_state.fy - ls_state.sty * ls_state.gtest;

        FP gm  = g - ls_state.gtest;
        FP gxm = ls_state.gx - ls_state.gtest;
        FP gym = ls_state.gy - ls_state.gtest;

        dcstep(ls_state.stx, fxm, gxm,
               ls_state.sty, fym, gym,
               stp, fm, gm,
               ls_state.bracket, ls_state.stmin, ls_state.stmax);

        ls_state.fx1 = fxm + ls_state.stx * ls_state.gtest;
        ls_state.fy  = fym + ls_state.sty * ls_state.gtest;
        ls_state.gx  = gxm + ls_state.gtest;
        ls_state.gy  = gym + ls_state.gtest;
    } else {
        dcstep(ls_state.stx, ls_state.fx1, ls_state.gx,
               ls_state.sty, ls_state.fy,  ls_state.gy,
               stp, f, g,
               ls_state.bracket, ls_state.stmin, ls_state.stmax);
    }

    if (ls_state.bracket) {
        if (std::abs(ls_state.sty - ls_state.stx) >= p66 * ls_state.width1) {
            stp = ls_state.stx + p5 * (ls_state.sty - ls_state.stx);
        }
        ls_state.width1 = ls_state.width;
        ls_state.width  = std::abs(ls_state.sty - ls_state.stx);
    }

    if (ls_state.bracket) {
        ls_state.stmin = std::min(ls_state.stx, ls_state.sty);
        ls_state.stmax = std::max(ls_state.stx, ls_state.sty);
    } else {
        ls_state.stmin = stp + xtrapl * (stp - ls_state.stx);
        ls_state.stmax = stp + xtrapu * (stp - ls_state.stx);
    }

    stp = std::max(stp, stpmin);
    stp = std::min(stp, stpmax);

    if (ls_state.bracket && (stp <= ls_state.stmin || stp >= ls_state.stmax || (ls_state.stmax - ls_state.stmin) <= xtol * ls_state.stmax)) {
        stp = ls_state.stx;
    }
    task = LineSearchTask::EvaluateFG;
}

} // namespace min

#endif // LBFGSB_DCSRCH_HPP
