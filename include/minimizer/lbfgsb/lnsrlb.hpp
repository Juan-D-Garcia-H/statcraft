// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_LNSRLB_HPP
#define LBFGSB_LNSRLB_HPP

/**
 * @file lnsrlb.hpp
 * @ingroup lbfgsb_linesearch
 * @brief Drive the More–Thuente line search within the L-BFGS-B iteration.
 *
 * @details
 * Reverse-communication layer between the outer L-BFGS-B loop and the
 * low-level @ref dcsrch line search. It computes the directional derivative
 * \f$g^T d\f$, initializes the trial step (unit step for bounded problems,
 * scaled to \f$O(1/\|d\|)\f$ otherwise), bounds the maximum step so the trial
 * points stay feasible, and updates the iterate \f$x = t + \mathrm{stp}\,d\f$
 * (or \f$x = z\f$ when \f$\mathrm{stp}=1\f$).
 *
 * All state that must survive across `EvaluateFG` re-entries lives in the
 * caller-provided @ref LineSearchState, replacing the historical Fortran
 * `csave`/`isave`/`dsave` arrays.
 *
 * @see dcsrch.hpp, dcstep.hpp, mainlb.hpp (caller)
 */


#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#include "dcsrch.hpp"
#include "../common.hpp"
#include "state.hpp"

namespace min {

template<typename FP = double>
/**
 * @brief Coordinate the safeguarded line search for one outer iteration.
 *
 * @details
 * On the **first entry** (`task == Task::Start`): computes
 * \f$\mathrm{dtd} = d^Td\f$ and \f$\mathrm{dnorm} = \|d\|\f$; if the problem is
 * bounded, scans the active bounds to tighten @p stpmx so that
 * \f$x+\mathrm{stpmx}\,d\f$ stays feasible (with a unit cap on the very first
 * iteration), otherwise \f$\mathrm{stpmx}=10^{10}\f$; seeds the trial step
 * \f$\mathrm{stp} = \min(1/\mathrm{dnorm},\,\mathrm{stpmx})\f$ when
 * \f$i=0\f$ and the problem is not fully boxed, else \f$\mathrm{stp}=1\f$; and
 * snapshots \f$t = x\f$, \f$r = g\f$, \f$\mathrm{fold}=f\f$ for the failure
 * restoration path, resetting \f$\mathrm{ifun}=0\f$, \f$\mathrm{iback}=0\f$ and
 * @p ls_state.
 *
 * On **every entry** (including `EvaluateFG_LineSearch` re-entries) it
 * recomputes \f$\mathrm{gd} = g^T d\f$, saves \f$\mathrm{gdold}\f$ on the first
 * evaluation and reports `info = -4` when \f$\mathrm{gd} \ge 0\f$ (the
 * direction is not a descent direction), then delegates to @ref dcsrch with
 * the internal tolerances \f$\mathrm{ftol}=10^{-3}\f$, \f$\mathrm{gtol}=0.9\f$,
 * \f$\mathrm{xtol}=0.1\f$. While the search runs, the routine returns
 * `Task::EvaluateFG_LineSearch` after materializing the new trial iterate and
 * incrementing the evaluation counters (`ifun`, `nfgv`, `iback = ifun-1`). When
 * @ref dcsrch reports `Convergence`/`Warning`, the task becomes `Task::NewX`.
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param l       Lower bounds (length @p n).
 * @param u       Upper bounds (length @p n).
 * @param nbd     Per-variable bound kind, see @ref BoundType.
 * @param x       In/out: trial iterate (updated on `EvaluateFG_LineSearch`
 *                exits).
 * @param f       In/out: objective value at @p x.
 * @param fold    On exit, value of @p f at the start of this line search.
 * @param gd      In/out: directional derivative at the current trial step.
 * @param gdold   On exit, directional derivative at \f$\mathrm{stp}=0\f$
 *                (first evaluation of this line search).
 * @param g       Gradient at @p x.
 * @param d       Search direction (length @p n).
 * @param r       Workspace: gradient snapshot for the failure restoration.
 * @param t       Workspace: iterate snapshot for the failure restoration.
 * @param z       Pre-projected candidate (Cauchy/subspace output); used
 *                verbatim when \f$\mathrm{stp}=1\f$.
 * @param stp     In/out: trial step length.
 * @param dnorm   On exit, \f$\|d\|\f$.
 * @param dtd     On exit, \f$d^Td\f$.
 * @param xstep   On exit, \f$\mathrm{stp}\,\mathrm{dnorm}\f$.
 * @param stpmx   In/out: maximum feasible step (tightened by the bound scan).
 * @param iter    Outer iteration index (seeds the first-iteration step cap).
 * @param ifun    In/out: function/gradient evaluations in **this** line search.
 * @param iback   In/out: number of backtracks (\f$\mathrm{ifun}-1\f$); the
 *                driver aborts when \f$\mathrm{iback} \ge 20\f$.
 * @param nfgv    In/out: cumulative evaluation counter across the whole solve.
 * @param info    On exit: `0` on success; `-4` when there is no descent
 *                direction at the start.
 * @param task    In/out: `Task::Start` on entry; `Task::EvaluateFG_LineSearch`
 *                while evaluations are requested; `Task::NewX` once the search
 *                completed.
 * @param boxed   True iff every variable has both bounds (unit first step).
 * @param cnstnd  True iff the problem has at least one bound (enables the
 *                feasibility scan).
 * @param ls_state Persistent line-search state (see @ref LineSearchState).
 */
inline void lnsrlb(int n,
                   const Vec<FP>& l,
                   const Vec<FP>& u,
                   const Vec<BoundType>& nbd,
                   Vec<FP>& x,
                   FP& f,
                   FP& fold,
                   FP& gd,
                   FP& gdold,
                   const Vec<FP>& g,
                   Vec<FP>& d,
                   Vec<FP>& r,
                   Vec<FP>& t,
                   Vec<FP>& z,
                   FP& stp,
                   FP& dnorm,
                   FP& dtd,
                   FP& xstep,
                   FP& stpmx,
                   int& iter,
                   int& ifun,
                   int& iback,
                   int& nfgv,
                   int& info,
                   Task& task,
                   bool& boxed,
                   bool& cnstnd,
                   LineSearchState<FP>& ls_state)
{
    constexpr FP one  = FP(1.0);
    constexpr FP zero = FP(0.0);
    constexpr FP big  = FP(1.0e10);
    constexpr FP ftol = FP(1.0e-3);
    constexpr FP gtol = FP(0.9);
    constexpr FP xtol = FP(0.1);

    if (task == Task::EvaluateFG_LineSearch) {
        goto label556;
    }

    // ── First entry ──────────────────────────────────────────────────────────
    dtd   = d.dot(d);
    dnorm = std::sqrt(dtd);

    stpmx = big;
    if (cnstnd) {
        if (iter == 0) {
            stpmx = one;
        } else {
            for (int i = 0; i < n; ++i) {
                const FP a1 = d(i);
                const auto bnd = nbd(i);
                if (bnd != BoundType::Unbounded) {
                    if (a1 < zero && (bnd == BoundType::Lower || bnd == BoundType::Both)) {
                        const FP a2 = l(i) - x(i);
                        if (a2 >= zero)
                            stpmx = zero;
                        else if (a1 * stpmx < a2)
                            stpmx = a2 / a1;
                    } else if (a1 > zero && (bnd == BoundType::Both || bnd == BoundType::Upper)) {
                        const FP a2 = u(i) - x(i);
                        if (a2 <= zero)
                            stpmx = zero;
                        else if (a1 * stpmx > a2)
                            stpmx = a2 / a1;
                    }
                }
            }
        }
    }

    stp = (iter == 0 && !boxed) ? std::min(one / dnorm, stpmx) : one;

    t     = x;
    r     = g;
    fold  = f;
    ifun  = 0;
    iback = 0;
    ls_state.reset();

label556:
    gd = g.dot(d);

    if (ifun == 0) {
        gdold = gd;
        if (gd >= zero) {
            info = -4;
            return;
        }
    }

    dcsrch(f, gd, stp, ftol, gtol, xtol, zero, stpmx, ls_state.task, ls_state);

    xstep = stp * dnorm;

    if (ls_state.task != LineSearchTask::Convergence && ls_state.task != LineSearchTask::Warning) {
        task  = Task::EvaluateFG_LineSearch;
        ++ifun;
        ++nfgv;
        iback = ifun - 1;

        if (stp == one) {
            x = z;
        } else {
            for (int i = 0; i < n; ++i)
                x(i) = stp * d(i) + t(i);
        }
    } else {
        task = Task::NewX;
    }
}

} // namespace min

#endif // LBFGSB_LNSRLB_HPP
