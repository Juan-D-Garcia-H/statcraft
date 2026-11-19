// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_CMPRLB_HPP
#define LBFGSB_CMPRLB_HPP

/**
 * @file cmprlb.hpp
 * @ingroup lbfgsb_direction
 * @brief Compute the subspace residual \f$r = -Z'B(x_{cp}-x) - Z'g\f$.
 *
 * @details
 * Given the generalized Cauchy point @p z (called \f$x_{cp}\f$), the gradient
 * and the L-BFGS matrix, this routine evaluates
 * \f$r = -Z^T B\,(z-x) - Z^T g\f$ restricted to the free variables listed in
 * @p index, i.e. the reduced gradient of the quadratic subspace model at the
 * GCP. The \f$W^T(z-x)\f$ products computed by @ref cauchy are consumed here:
 * the stored `c_wa` block is mapped by the middle matrix through @ref bmv and
 * folded back into @p r with the `ws`/`wy` histories.
 *
 * @see cauchy.hpp (producer of @p z and the `c_wa` block), bmv.hpp,
 *      subsm.hpp (consumer of @p r)
 */


#include <Eigen/Dense>
#include "bmv.hpp"
#include "../common.hpp"

namespace min {

template<typename FP = double>
/**
 * @brief Compress the residual onto the free variables for the subspace problem.
 *
 * @details
 * On the unconstrained shortcut path (`!cnstnd && col > 0`) the whole
 * \f$r = -g\f$ is returned and the routine exits immediately. Otherwise the
 * residual for each free variable \f$i = \mathrm{index}(k)\f$ starts as
 * \f$r_i = -\theta\,(z_i - x_i) - g_i\f$. If \f$\mathrm{col} > 0\f$ the
 * \f$W^T(z-x)\f$ block in `c_wa` is mapped through the middle matrix via
 * @ref bmv (copied into @p wa8m) and the correction
 * \f$\sum_j \big(wy_{i,j}\,a_{1,j} + ws_{i,j}\,\theta a_{2,j}\big)\f$ is
 * added, reconstructing \f$-Z'B(z-x)\f$.
 *
 * @note The classic Fortran workspace segments `wa(1..2m)` and `wa(2m+1..4m)`
 *       appear here as the separate buffers @p wa8m (bmv result) and
 *       @p c_wa (input block).
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param m       L-BFGS memory size.
 * @param x       Current iterate.
 * @param g       Gradient at @p x.
 * @param ws      S history matrix (\f$n\times m\f$).
 * @param wy      Y history matrix (\f$n\times m\f$).
 * @param sy      Matrix of \f$S'Y\f$ products (\f$m\times m\f$; forwarded to @ref bmv).
 * @param wt      L-BFGS factor from @ref formt (\f$m\times m\f$; forwarded to @ref bmv).
 * @param z       The generalized Cauchy point \f$x_{cp}\f$.
 * @param r       On exit, the residual restricted to the free variables
 *                (length \f$\ge \mathrm{nfree}\f$).
 * @param c_wa    Input: \f$W^T(z-x)\f$ in its first \f$2\mathrm{col}\f$ entries.
 * @param wa8m    Scratch: receives the \f$M^{-1}W^T(z-x)\f$ result of @ref bmv.
 * @param index   Indices of the free variables (first @p nfree entries).
 * @param theta   L-BFGS scaling \f$B_0 = \theta I\f$.
 * @param col     Number of active pairs.
 * @param head    Oldest stored correction index (walk start).
 * @param nfree   Number of free variables (size of the subspace problem).
 * @param cnstnd  True if the problem has bounds; enables the shortcut path
 *                described above.
 * @param info    On exit: `0` on success; `-8` when the internal @ref bmv
 *                call reports a dimensional failure.
 */
inline void cmprlb(int n,
                   int m,
                   const Vec<FP>& x,
                   const Vec<FP>& g,
                   const Mat<FP>& ws,
                   const Mat<FP>& wy,
                   const Mat<FP>& sy,
                   const Mat<FP>& wt,
                   const Vec<FP>& z,
                   Vec<FP>& r,
                   const Vec<FP>& c_wa,
                   Vec<FP>& wa8m,
                   const VecInt& index,
                   FP theta,
                   int col,
                   int head,
                   int nfree,
                   bool cnstnd,
                   int& info)
{
    if (!cnstnd && col > 0) {
        r.head(n) = -g;
        return;
    }

    for (int i = 0; i < nfree; ++i) {
        const int k = index(i);
        r(i) = -theta * (z(k) - x(k)) - g(k);
    }

    if (col == 0) return;

    {
        Vec<FP> v_in  = c_wa.head(2 * col);
        Vec<FP> v_out = Vec<FP>::Zero(2 * col);
        bmv(m, sy, wt, col, v_in, v_out, info);
        if (info != 0) { info = -8; return; }
        wa8m.head(2 * col) = v_out;
    }

    int pointr = head;
    for (int j = 0; j < col; ++j) {
        const FP a1 = wa8m(j);
        const FP a2 = theta * wa8m(col + j);
        for (int i = 0; i < nfree; ++i) {
            const int k = index(i);
            r(i) += wy(k, pointr) * a1 + ws(k, pointr) * a2;
        }
        pointr = (pointr + 1) % m;
    }
}

} // namespace min

#endif // LBFGSB_CMPRLB_HPP
