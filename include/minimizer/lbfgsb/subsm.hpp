// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_SUBSM_HPP
#define LBFGSB_SUBSM_HPP

/**
 * @file subsm.hpp
 * @ingroup lbfgsb_direction
 * @brief Solve the bounded subspace (Newton) minimization problem.
 *
 * @details
 * Given the generalized Cauchy point \f$x_{cp}\f$ and the residual
 * \f$r = -Z'B(x_{cp}-x)-Z'g\f$ of @ref cmprlb, this routine computes an
 * approximate minimizer of the quadratic
 * \f$\min_{x}\; r'(x-x_{cp}) + \tfrac12 (x-x_{cp})' B (x-x_{cp}) \quad\text{s.t.}\quad l \le x \le u,\; x_i = x_{cp,i}\ \text{for}\ i \notin \text{free}\f$
 * along the subspace-truncated Newton direction
 * \f$d = -(Z'BZ)^{-1}r\f$, expressed via the Sherman–Morrison–Woodbury form
 * \f$d = \tfrac1\theta r + \tfrac1{\theta^2} Z' W K^{-1} W' Z r\f$ with
 * \f$K\f$ the factorized matrix produced by @ref formk.
 *
 * Implements the improved (2011) version of Morales & Nocedal: the truncated
 * direction is safeguarded by the positivity test
 * \f$(x_\text{truncated}-x)'g > 0\f$; when it fails, the direction is instead
 * truncated to the first bound hit (restoring the Cauchy point).
 *
 * @see cmprlb.hpp (producer of @p r), formk.hpp (factorization of \f$K\f$),
 *      mainlb.hpp (caller)
 */


#include <Eigen/Dense>
#include <algorithm>
#include "../common.hpp"

namespace min {

/**
 * @brief Compute the truncated Newton step over the free-variable subspace.
 *
 * @details
 * Steps performed:
 * 1. **Gather.** \f$wv = W'Z d\f$ is formed from the current direction @p d,
 *    the free indices @p ind and the `ws`/`wy` histories.
 * 2. **Solve.** The triangular system \f$K\,wv = wv\f$ is solved with the two
 *    triangular views of @p wn (the \f$LEL^T\f$ factors from @ref formk); the
 *    first half is negated (block-diagonal \f$E\f$). `info` is reset to `0`
 *    (the routine never reports failure — the factorization guarantees
 *    nonsingularity).
 * 3. **Expand.** \f$d \leftarrow \tfrac1\theta d + \tfrac1{\theta^2} Z' W wv\f$.
 * 4. **Clip.** The free coordinates of \f$x\f$ are projected onto the box; any
 *    coordinate that hits a bound sets IWORD to @ref SubspaceStatus::Boundary.
 * 5. **Safeguard.** If the truncated step increased the *linearized* value
 *    (\f$\sum_i (x_i - x_i^{\text{prev}}) g_i > 0\f$), the direction is
 *    truncated to the very first bound crossing instead: the Cauchy point is
 *    restored (@p xp) and the new direction is scaled down by
 *    \f$\alpha = \min$ ratio to the first bound, with the bound coordinate set
 *    exactly at its limit.
 *
 * @note @p m and @p info: `m` is not referenced by the body (kept for
 *       interface symmetry). `info` is always written as `0`; it is never a
 *       failure channel here. @ref SubspaceStatus::Truncated is defined in the
 *       shared enum but never produced by this routine.
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param m       L-BFGS memory size (unused by the body).
 * @param nsub    Number of free variables (subspace size). If \f$\le 0\f$ the
 *                routine returns immediately.
 * @param ind     Indices of the free variables (length @p nsub).
 * @param l       Lower bounds (length @p n).
 * @param u       Upper bounds (length @p n).
 * @param nbd     Per-variable bound kind, see @ref BoundType.
 * @param x       In/out: on entry the Cauchy point \f$x_{cp}\f$; on exit the
 *                subspace minimizer (free coordinates updated).
 * @param d       In/out: on entry the reduced residual (in @ref cmprlb terms);
 *                on exit the (truncated) Newton direction over the subspace.
 * @param xp      Scratch: copy of the Cauchy point used to restore it in the
 *                safeguard branch.
 * @param ws      S history matrix (\f$n\times m\f$).
 * @param wy      Y history matrix (\f$n\times m\f$).
 * @param theta   L-BFGS scaling \f$B_0 = \theta I\f$.
 * @param xx      Current outer iterate \f$x\f$ (unmodified).
 * @param gg      Gradient at @p xx (unmodified).
 * @param col     Number of active pairs.
 * @param head    Oldest stored correction index.
 * @param iword   On exit: @ref SubspaceStatus::Constant when no free coordinate
 *                reached a bound, @ref SubspaceStatus::Boundary otherwise.
 * @param wv      Length-\f$2m\f$ scratch for \f$W'Zd\f$ and the solved vector.
 * @param wn      2m x 2m \f$LEL^T\f$ factorization of \f$K\f$ from @ref formk.
 * @param info    On exit, always `0`.
 */


template<typename FP = double>
inline void subsm(int n,
                  int m,
                  int nsub,
                  const VecInt& ind,
                  const Vec<FP>& l,
                  const Vec<FP>& u,
                  const Vec<BoundType>& nbd,
                  Vec<FP>& x,
                  Vec<FP>& d,
                  Vec<FP>& xp,
                  const Mat<FP>& ws,
                  const Mat<FP>& wy,
                  FP theta,
                  const Vec<FP>& xx,
                  const Vec<FP>& gg,
                  int col,
                  int head,
                  SubspaceStatus& iword,
                  Vec<FP>& wv,
                  const Mat<FP>& wn,
                  int& info)
{
    if (nsub <= 0) return;

    const int col2 = 2 * col;

    // wv = W' Z d
    {
        int pointr = head;
        for (int i = 0; i < col; ++i) {
            FP t1 = FP(0.0), t2 = FP(0.0);
            for (int j = 0; j < nsub; ++j) {
                const int k = ind(j);
                t1 += wy(k, pointr) * d(j);
                t2 += ws(k, pointr) * d(j);
            }
            wv(i)       = t1;
            wv(col + i) = theta * t2;
            pointr = (pointr + 1) % m;
        }
    }

    // wv = K^{-1} wv
    {
        Mat<FP> wn_loc = wn.topLeftCorner(col2, col2);
        wv.head(col2) =
            wn_loc.template triangularView<Eigen::Upper>().transpose()
                  .solve(wv.head(col2));
        info = 0;
        wv.head(col) = -wv.head(col);
        wv.head(col2) =
            wn_loc.template triangularView<Eigen::Upper>()
                  .solve(wv.head(col2));
        info = 0;
    }

    // d = (1/theta)*d + (1/theta^2) Z'W wv
    {
        int pointr = head;
        for (int jy = 0; jy < col; ++jy) {
            const FP wvjy = wv(jy), wvjs = wv(col + jy);
            for (int i = 0; i < nsub; ++i) {
                const int k = ind(i);
                d(i) += wy(k, pointr) * wvjy / theta
                      + ws(k, pointr) * wvjs;
            }
            pointr = (pointr + 1) % m;
        }
    }
    d.head(nsub) /= theta;

    iword = SubspaceStatus::Constant;
    xp    = x;

    for (int i = 0; i < nsub; ++i) {
        const int k = ind(i); const FP dk = d(i); const FP xk = x(k);
        const auto bnd = nbd(k);
        if (bnd != BoundType::Unbounded) {
            if (bnd == BoundType::Lower) {
                x(k) = std::max(l(k), xk + dk);
                if (x(k) == l(k)) iword = SubspaceStatus::Boundary;
            } else if (bnd == BoundType::Both) {
                FP xn = std::max(l(k), xk + dk);
                x(k)  = std::min(u(k), xn);
                if (x(k) == l(k) || x(k) == u(k)) iword = SubspaceStatus::Boundary;
            } else if (bnd == BoundType::Upper) {
                x(k) = std::min(u(k), xk + dk);
                if (x(k) == u(k)) iword = SubspaceStatus::Boundary;
            }
        } else {
            x(k) = xk + dk;
        }
    }

    if (iword == SubspaceStatus::Constant) goto label911;

    {
        FP dd_p = FP(0.0);
        for (int i = 0; i < n; ++i) dd_p += (x(i) - xx(i)) * gg(i);
        if (dd_p > FP(0.0)) {
            x = xp;
        } else {
            goto label911;
        }
    }

    {
        FP alpha = FP(1.0), temp1 = alpha;
        int ibd  = 0;
        for (int i = 0; i < nsub; ++i) {
            const int k = ind(i); const FP dk = d(i);
            const auto bnd = nbd(k);
            if (bnd != BoundType::Unbounded) {
                FP temp2;
                if (dk < FP(0.0) && (bnd == BoundType::Lower || bnd == BoundType::Both)) {
                    temp2 = l(k) - x(k);
                    if      (temp2 >= FP(0.0))      temp1 = FP(0.0);
                    else if (dk * alpha < temp2)    temp1 = temp2 / dk;
                } else if (dk > FP(0.0) && (bnd == BoundType::Both || bnd == BoundType::Upper)) {
                    temp2 = u(k) - x(k);
                    if      (temp2 <= FP(0.0))      temp1 = FP(0.0);
                    else if (dk * alpha > temp2)    temp1 = temp2 / dk;
                }
                if (temp1 < alpha) { alpha = temp1; ibd = i; }
            }
        }
        if (alpha < FP(1.0)) {
            const int k = ind(ibd); const FP dk = d(ibd);
            if      (dk > FP(0.0)) { x(k) = u(k); d(ibd) = FP(0.0); }
            else if (dk < FP(0.0)) { x(k) = l(k); d(ibd) = FP(0.0); }
        }
        for (int i = 0; i < nsub; ++i) x(ind(i)) += alpha * d(i);
    }

label911:
    return;
}

} // namespace min

#endif // LBFGSB_SUBSM_HPP
