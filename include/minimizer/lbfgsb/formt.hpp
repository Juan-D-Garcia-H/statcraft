// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_FORMT_HPP
#define LBFGSB_FORMT_HPP

/**
 * @file formt.hpp
 * @ingroup lbfgsb_matrix
 * @brief Form and factor \f$T = \theta SS' + L D^{-1} L'\f$ for the L-BFGS update.
 *
 * @details
 * Given the accumulated \f$S'S\f$ and \f$S'Y\f$ inner-product matrices, this
 * routine assembles the upper triangle of the symmetric positive-definite
 * matrix \f$T\f$ used in the compact L-BFGS-B representation, then computes its
 * Cholesky factorization \f$T = JJ'\f$ and stores \f$J'\f$ (the upper-triangular
 * factor) back into @p wt. @ref bmv subsequently uses this factor to apply
 * \f$M^{-1}\f$. A failing or non-positive-definite factorization is signalled
 * through @p info, never by throwing.
 *
 * @see bmv.hpp (consumer of the factor), matupd.hpp (matching formk counterpart)
 */


#include <Eigen/Dense>
#include <cmath>
#include "../common.hpp"

namespace min {

template<typename FP = double>
/**
 * @brief Build \f$T\f$ and overwrite @p wt with its Cholesky factor \f$J'\f$.
 *
 * @details
 * Fills the upper triangle of the leading \f$\mathrm{col}\times\mathrm{col}\f$
 * block of @p wt with
 * \f$T_{ij} = \sum_{k < i} \frac{(S'Y)_{ik}(S'Y)_{jk}}{(S'Y)_{kk}}
 * + \theta\,(S'S)_{ij}\f$
 * (mirroring `formt.f`: the summation index runs strictly below the smaller
 * row/column index, so the first row and the diagonal carry no \f$S'Y\f$ terms;
 * the \f$k = i\f$ contribution enters the factorization only through the
 * strictly-lower part of \f$S'Y\f$), then factorizes it:
 * - for \f$\mathrm{col}=1\f$ the factor is the plain square root, and a
 *   non-positive \f$T_{00}\f$ yields `info = -3`;
 * - for \f$\mathrm{col} > 1\f$ an Eigen \f$LL^T\f$ factorization of the upper
 *   self-adjoint view is used; failure (or a thrown LAPACK-style exception)
 *   yields `info = -3`.
 *
 * On entry `info` is always reset to `0`, so a previous value never leaks.
 * Failure modes by priority: `-1` undersized @p wt, `-2` undersized @p sy,
 * `-3` undersized @p ss **or** non-positive-definite factor.
 *
 * @tparam FP    Floating-point scalar type.
 * @param m      L-BFGS memory size; not read by the body (kept for interface
 *               symmetry), the leading @p col block drives everything.
 * @param wt     In/out: on exit the upper triangle of its leading
 *               \f$\mathrm{col}\times\mathrm{col}\f$ block holds \f$J'\f$.
 * @param sy     Inner products \f$S'Y\f$ (leading \f$\mathrm{col}\f$ block used).
 * @param ss     Inner products \f$S'S\f$ (leading \f$\mathrm{col}\f$ block used).
 * @param col    Number of active correction pairs. If \f$\le 0\f$, returns
 *               with `info = 0` immediately.
 * @param theta  Initial-Hessian scaling \f$B_0 = \theta I\f$.
 * @param info   On exit: `0` success, `-1`/`-2`/`-3` as described above.
 */
inline void formt([[maybe_unused]] int m,
                  Mat<FP>& wt,
                  const Mat<FP>& sy,
                  const Mat<FP>& ss,
                  int col,
                  FP theta,
                  int& info)
{
    info = 0;
    if (col <= 0) return;

    if (wt.rows() < col || wt.cols() < col) { info = -1; return; }
    if (sy.rows() < col || sy.cols() < col) { info = -2; return; }
    if (ss.rows() < col || ss.cols() < col) { info = -3; return; }

    wt.topLeftCorner(col, col).setZero();

    // Mirror formt.f: wt(i,j) = theta*ss(i,j) + sum_{k<i} sy(i,k)*sy(j,k)/sy(k,k)
    // (k runs strictly below min(i,j); the first row and the diagonal carry no
    // sy terms -- the k = i term belongs to the L*D^{-1}*L' factorization only
    // through the strictly lower part of sy).
    for (int i = 0; i < col; ++i) {
        for (int j = i; j < col; ++j) {
            FP sum = FP(0.0);
            for (int k = 0; k < i; ++k) {
                sum += sy(i, k) * sy(j, k) / sy(k, k);
            }
            wt(i, j) = sum + theta * ss(i, j);
        }
    }

    if (col == 1) {
        if (wt(0, 0) <= FP(0.0)) {
            info = -3;
            return;
        }
        wt(0, 0) = std::sqrt(wt(0, 0));
        return;
    }

    try {
        Eigen::LLT<Mat<FP>> llt;
        llt.compute(wt.topLeftCorner(col, col).template selfadjointView<Eigen::Upper>());

        if (llt.info() != Eigen::Success) {
            info = -3;
            return;
        }
        wt.topLeftCorner(col, col).template triangularView<Eigen::Upper>() = llt.matrixU();
    } catch (...) {
        info = -3;
        return;
    }
}

} // namespace min

#endif // LBFGSB_FORMT_HPP
