// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_CAUCHY_HPP
#define LBFGSB_CAUCHY_HPP

/**
 * @file cauchy.hpp
 * @ingroup lbfgsb_direction
 * @brief Compute the generalized Cauchy point (GCP) along the projected gradient.
 *
 * @details
 * Given \f$x, l, u, g\f$ and the limited-memory BFGS matrix \f$B\f$ defined by
 * `wy`, `ws`, `wt` and the scalars `head`, `col`, `theta`, this routine finds
 * the first local minimizer of
 * \f[
 *   Q(x+s) = g^T s + \tfrac12 s^T B s
 * \f]
 * along the projected-gradient direction \f$P(x-\tau g)\f$, i.e. the first
 * minimum hit while walking in the (clipped) direction of steepest descent.
 * The result (the GCP) is returned in @p xcp together with the refined
 * bound/free classification @p iwhere and the breakpoint ordering @p iorder
 * used by the subsequent subspace phase.
 *
 * The search is piecewise quadratic: the path is divided at the bound
 * breakpoints (where a coordinate becomes clamped). The routine tracks the
 * accumulated curvature corrections \f$p = W^T d\f$ and \f$c = W^T(x-x_k)\f$,
 * using @ref bmv for every \f$M^{-1}\f$ application.
 *
 * This is the port of the classic `cauchy` of Byrd, Lu, Nocedal & Zhu (1995).
 *
 * @see bmv.hpp, hpsolb.hpp, mainlb.hpp (caller)
 */


#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "bmv.hpp"
#include "hpsolb.hpp"
#include "../common.hpp"
#include "logger.hpp"

namespace min {

/**
 * @brief Compute the generalized Cauchy point \f$x_{cp}\f$.
 *
 * @details
 * Three phases:
 * 1. **Bound classification and breakpoints.** For each coordinate the search
 *    direction is \f$d_i = -g_i\f$ unless the coordinate is permanently fixed
 *    (`iwhere == 3`) or always free (`iwhere == -1`). For bounded, moving
 *    coordinates the distance-to-bound breakpoints
 *    \f$\tau_i = (x_i - l_i)/(-g_i)\f$ (upper move, \f$g_i<0\f$) or
 *    \f$\tau_i = (u_i - x_i)/g_i\f$ (lower move, \f$g_i>0\f$) are recorded in
 *    @p t together with their indices in @p iorder; the smallest one seeds the
 *    first segment (`bkmin`/`ibkmin`). Free coordinates with no bound along
 *    the direction set `bnded = false` and are moved without clipping.
 * 2. **GCP walk.** Starting with the unconstrained minimizer estimate
 *    \f$\tau_m = -f_1/f_2\f$ (where \f$f_1 = -\|g\|^2\f$-type and
 *    \f$f_2 = -\theta f_1 - p^T M^{-1}p\f$), the routine advances segment by
 *    segment. At each breakpoint the hitting coordinate is clamped
 *    (\f$iwhere = 1\f$/`2`), the curvature corrections `c`/`p`/`wbp` are
 *    updated through @ref bmv, and \f$f_1, f_2, \tau_m\f$ are propagated. The
 *    walk stops when the next breakpoint lies beyond \f$\tau_m\f$, or after the
 *    last breakpoint when all coordinates are bounded (`bnded`), or when the
 *    accumulated step \f$\sum \tau\f$ is used up.
 * 3. **Finalize.** \f$x_{cp} = x + \bar\tau\,d\f$ with the residual step
 *    \f$\bar\tau = \max(0, \tau_m)\f$, and the correction buffer
 *    \f$c = W^T(x_{cp}-x)\f$ is completed for the caller.
 *
 * If \f$\mathrm{sbgnrm} \le 0\f$ the GCP is simply \f$x\f$ itself (no descent
 * direction). On a failing @ref bmv call (`info != 0`) the routine returns
 * early; @ref mainlb then discards the L-BFGS memory and restarts.
 *
 * @note The breakpoint ordering contract with @ref hpsolb is the classic one:
 *       after every call the *smallest* unprocessed breakpoint sits at
 *       `t(nleft-1)` (the spot read below at `tj = t(nleft-1)`), followed by the
 *       remaining elements as a min-heap. `hpsolb` is a faithful 1:1 port of
 *       the reference Fortran `hpsolb.f` (verified bit-for-bit against it), so
 *       this precondition holds for any number of breakpoints.
 *
 * @note Unlike the Fortran original, this routine takes a @ref Logger
 *       reference (@p logger), not an `iprint` integer/`iterate.dat` scheme,
 *       and keeps the @p info status parameter to forward @ref bmv failures to
 *       its caller.
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param x       Current iterate (unmodified).
 * @param l       Lower bounds (length @p n).
 * @param u       Upper bounds (length @p n).
 * @param nbd     Per-variable bound kind, see @ref BoundType.
 * @param g       Gradient at @p x (must be non-zero to make progress).
 * @param iorder  Working buffer; on exit, ordering of breakpoints / free
 *                variables as described below the function.
 * @param iwhere  In: `3` (fixed) / `-1` (free) markers from @ref active.
 *                Out: refined status per coordinate
 *                (`-3` bounded-free-moved, `0` bounded-free-still, `1`/`2`
 *                clamped at `l`/`u`, `3` always fixed, `-1` always free).
 * @param t       Working buffer for the breakpoint values \f$\tau\f$.
 * @param d       On exit, the piecewise projected-gradient direction
 *                \f$P(x-\tau g)-x\f$.
 * @param xcp     On exit, the generalized Cauchy point.
 * @param m       L-BFGS memory size.
 * @param wy      Y history matrix (\f$n\times m\f$).
 * @param ws      S history matrix (\f$n\times m\f$).
 * @param sy      Matrix of \f$S'Y\f$ inner products (\f$m\times m\f$).
 * @param wt      Cholesky factor from @ref formt (\f$m\times m\f$).
 * @param theta   L-BFGS scaling \f$B_0 = \theta I\f$.
 * @param col     Number of active pairs.
 * @param head    Oldest stored correction index.
 * @param p       Length-\f$2m\f$ buffer for \f$W^T d\f$.
 * @param c       Length-\f$2m\f$ buffer for \f$W^T(x_{cp}-x)\f$.
 * @param wbp     Length-\f$2m\f$ buffer for the row of \f$W\f$ of a breakpoint.
 * @param v       Length-\f$2m\f$ buffer reused for every @ref bmv result.
 * @param nseg    On exit, number of quadratic segments explored.
 * @param logger  Logger for diagnostics (replaces the legacy `iprint`).
 * @param sbgnrm  Norm of the projected gradient at @p x; when \f$\le 0\f$ the
 *                GCP degenerates to \f$x\f$.
 * @param info    On exit: `0` on success; non-zero when an internal @ref bmv
 *                call failed.
 * @param epsmch  Machine epsilon; used as the floor for the quadratic
 *                coefficient \f$\max(\varepsilon f_2^{(0)}, f_2)\f$.
 */


template<typename FP = double>
inline void cauchy(int n,
                   const Vec<FP>& x,
                   const Vec<FP>& l,
                   const Vec<FP>& u,
                   const Vec<BoundType>& nbd,
                   const Vec<FP>& g,
                   VecInt& iorder,
                   VecInt& iwhere,
                   Vec<FP>& t,
                   Vec<FP>& d,
                   Vec<FP>& xcp,
                   int m,
                   const Mat<FP>& wy,
                   const Mat<FP>& ws,
                   const Mat<FP>& sy,
                   const Mat<FP>& wt,
                   FP theta,
                   int col,
                   int head,
                   Vec<FP>& p,
                   Vec<FP>& c,
                   Vec<FP>& wbp,
                   Vec<FP>& v,
                   int& nseg,
                   [[maybe_unused]] const Logger<FP>& logger,
                   FP sbgnrm,
                   int& info,
                   FP epsmch)
{
    const int col2 = 2 * col;

    if (sbgnrm <= FP(0.0)) {
        xcp = x;
        return;
    }

    bool bnded = true;
    int nfree  = n, nbreak = 0, ibkmin = -1;
    FP bkmin   = FP(0.0), f1 = FP(0.0);

    if (col2 > 0) {
        p.head(col2).setZero();
        c.head(col2).setZero();
    }

    for (int i = 0; i < n; ++i) {
        const FP neggi = -g(i);
        const auto bnd = nbd(i);

        if (iwhere(i) != 3 && iwhere(i) != -1) {
            FP tl = FP(0.0), tu = FP(0.0);
            if (bnd == BoundType::Lower || bnd == BoundType::Both) tl = x(i) - l(i);
            if (bnd == BoundType::Both  || bnd == BoundType::Upper) tu = u(i) - x(i);
            const bool xlower = (bnd == BoundType::Lower || bnd == BoundType::Both) && (tl <= FP(0.0));
            const bool xupper = (bnd == BoundType::Both  || bnd == BoundType::Upper) && (tu <= FP(0.0));
            iwhere(i) = 0;
            if      (xlower && neggi <= FP(0.0)) iwhere(i) = 1;
            else if (xupper && neggi >= FP(0.0)) iwhere(i) = 2;
            else if (std::abs(neggi) <= FP(0.0)) iwhere(i) = -3;
        }

        if (iwhere(i) != 0 && iwhere(i) != -1) {
            d(i) = FP(0.0);
        } else {
            d(i) = neggi;
            f1  -= neggi * neggi;
            int pointr = head;
            for (int j = 0; j < col; ++j) {
                p(j)       += wy(i, pointr) * neggi;
                p(col + j) += ws(i, pointr) * neggi;
                pointr      = (pointr + 1) % m;
            }
            const FP tl_i = x(i) - l(i), tu_i = u(i) - x(i);
            if ((bnd == BoundType::Lower || bnd == BoundType::Both) && neggi < FP(0.0)) {
                iorder(nbreak) = i;
                t(nbreak) = tl_i / (-neggi);
                if (nbreak == 0 || t(nbreak) < bkmin) { bkmin = t(nbreak); ibkmin = nbreak; }
                ++nbreak;
            } else if ((bnd == BoundType::Both || bnd == BoundType::Upper) && neggi > FP(0.0)) {
                iorder(nbreak) = i;
                t(nbreak) = tu_i / neggi;
                if (nbreak == 0 || t(nbreak) < bkmin) { bkmin = t(nbreak); ibkmin = nbreak; }
                ++nbreak;
            } else {
                --nfree;
                iorder(nfree) = i;
                if (std::abs(neggi) > FP(0.0)) bnded = false;
            }
        }
    }

    if (theta != FP(1.0) && col > 0) {
        p.segment(col, col) *= theta;
    }

    xcp = x;
    if (nbreak == 0 && nfree == n) return;

    FP f2     = -theta * f1;
    FP f2_org = f2;

    if (col > 0) {
        bmv(m, sy, wt, col, p, v, info);
        if (info != 0) return;
        f2 -= p.head(col2).dot(v.head(col2));
    }

    FP dtm = -f1 / f2, tsum = FP(0.0);
    nseg = 1;
    if (nbreak == 0) goto label888;

    {
        int nleft = nbreak, iter = 1, ibp = -1;
        FP tj = FP(0.0);

        while (true) {
            FP tj0 = tj;
            if (iter == 1) {
                tj = bkmin; ibp = iorder(ibkmin);
            } else {
                if (iter == 2 && ibkmin != nbreak - 1) {
                    t(ibkmin) = t(nbreak - 1); iorder(ibkmin) = iorder(nbreak - 1);
                }
                hpsolb(nleft, t, iorder, iter - 2);
                tj = t(nleft - 1); ibp = iorder(nleft - 1);
            }
            FP dt = tj - tj0;
            if (dtm < dt) goto label888;

            tsum += dt; --nleft; ++iter;
            const FP dibp = d(ibp);
            d(ibp) = FP(0.0);
            FP zibp;
            if (dibp > FP(0.0)) { zibp = u(ibp) - x(ibp); xcp(ibp) = u(ibp); iwhere(ibp) = 2; }
            else                { zibp = l(ibp) - x(ibp); xcp(ibp) = l(ibp); iwhere(ibp) = 1; }
            if (nleft == 0 && nbreak == n) { dtm = dt; goto label999; }

            ++nseg;
            const FP dibp2 = dibp * dibp;
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
            f2 = f2 - theta * dibp2;

            if (col > 0) {
                c.head(col2) += dt * p.head(col2);
                int pointr = head;
                for (int j = 0; j < col; ++j) {
                    wbp(j)       = wy(ibp, pointr);
                    wbp(col + j) = theta * ws(ibp, pointr);
                    pointr = (pointr + 1) % m;
                }
                bmv(m, sy, wt, col, wbp, v, info);
                if (info != 0) return;
                const FP wmc = c.head(col2).dot(v.head(col2));
                const FP wmp = p.head(col2).dot(v.head(col2));
                const FP wmw = wbp.head(col2).dot(v.head(col2));
                p.head(col2) -= dibp * wbp.head(col2);
                f1 += dibp * wmc;
                f2 += FP(2.0) * dibp * wmp - dibp2 * wmw;
            }

            f2 = std::max(epsmch * f2_org, f2);
            if (nleft > 0) { dtm = -f1 / f2; continue; }
            else if (bnded) { f1 = FP(0.0); f2 = FP(0.0); dtm = FP(0.0); }
            else            { dtm = -f1 / f2; }
            break;
        }
    }

label888:
    if (dtm <= FP(0.0)) dtm = FP(0.0);
    tsum += dtm;
    xcp += tsum * d;

label999:
    if (col > 0) {
        c.head(col2) += dtm * p.head(col2);
    }
}

} // namespace min

#endif // LBFGSB_CAUCHY_HPP
