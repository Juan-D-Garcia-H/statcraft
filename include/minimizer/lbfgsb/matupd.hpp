// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_MATUPD_HPP
#define LBFGSB_MATUPD_HPP

/**
 * @file matupd.hpp
 * @ingroup lbfgsb_matrix
 * @brief Append a new (s, y) correction pair to the L-BFGS circular history.
 *
 * @details
 * After a successful line search, the driver calls this routine to record the
 * accepted step \f$s = \mathrm{stp}\,d\f$ and the gradient difference
 * \f$y = g_{k+1} - g_k\f$ as new columns of the L-BFGS history matrices
 * `ws`/`wy`, and to refresh the associated inner-product matrices `sy` and
 * `ss`. When the memory is full (\f$\mathrm{iupdat} > m\f$), the oldest column
 * is discarded (cyclic wrap) and the stored inner products are shifted.
 *
 * The next-BFGS scaling \f$\theta\f$ is also updated to the standard
 * \f$\theta = y^T y / (s^T y)\f$.
 *
 * @see matupd is the matching counterpart of @ref formt/@ref formk
 */


#include <Eigen/Dense>
#include "../common.hpp"

namespace min {


template<typename FP = double>
/**
 * @brief Update the L-BFGS history with a new (s, y) pair.
 *
 * @details
 * The new `s = stp*d` and `y` vectors are written into columns `itail` of
 * `ws`/`wy`. For \f$\mathrm{iupdat} \le m\f$ the history simply grows:
 * `col = iupdat` and `itail = (head + iupdat - 1) % m`. Once
 * \f$\mathrm{iupdat} > m\f$ the write wraps around and the inner-product rows
 * of `ss`/`sy` are shifted one slot to discard the outgoing column. The
 * remaining `ss`/`sy` entries of the new column are filled by dot products
 * against the stored `ws`/`wy` columns walked from `head`, and the diagonal
 * values are set from the caller-supplied scalars:
 * \f$\mathrm{ss}(\cdot,\cdot) = \mathrm{stp}^2\,\mathrm{dtd}\f$
 * (with the \f$\mathrm{stp}=1\f$ shortcut) and
 * \f$\mathrm{sy}(\cdot,\cdot) = \mathrm{dr}\f$. Finally
 * \f$\theta = \mathrm{rr}/\mathrm{dr}\f$.
 *
 * @tparam FP    Floating-point scalar type.
 * @param n      Problem dimension; the length of the new step/`r` vectors
 *               (not otherwise used by the body).
 * @param m      L-BFGS memory size; bounds every modulo computation.
 * @param ws     In/out: \f$n\times m\f$ S history (new column written).
 * @param wy     In/out: \f$n\times m\f$ Y history (new column written).
 * @param sy     In/out: \f$m\times m\f$ \f$S'Y\f$ inner-product matrix.
 * @param ss     In/out: \f$m\times m\f$ \f$S'S\f$ inner-product matrix.
 * @param d      The accepted search direction (pre-scaled by `stp` when
 *               `stp != 1`; see @ref mainlb).
 * @param r      The accepted gradient difference \f$y\f$.
 * @param itail  In/out: column index written by this update.
 * @param iupdat Total number of updates performed so far (the driver
 *               increments it before the call).
 * @param col    In/out: number of active correction pairs after this update.
 * @param head   In/out: index of the oldest stored correction.
 * @param theta  In/out: overwritten with \f$\mathrm{rr}/\mathrm{dr}\f$.
 * @param rr     Squared 2-norm of @p r: \f$y^T y\f$.
 * @param dr     Curvature \f$s^T y = d^T r \cdot \mathrm{stp}\f$; the caller
 *               only calls this routine when it satisfies the curvature
 *               safeguard \f$\mathrm{dr} > \epsilon\,\mathrm{ddum}\f$.
 * @param stp    Line-search step length (used to recover \f$s = \mathrm{stp}\,d\f$
 *               in the `ss` diagonal).
 * @param dtd    Squared 2-norm of @p d: \f$d^T d\f$.
 */
inline void matupd([[maybe_unused]] int n,
                   int m,
                   Mat<FP>& ws,
                   Mat<FP>& wy,
                   Mat<FP>& sy,
                   Mat<FP>& ss,
                   const Vec<FP>& d,
                   const Vec<FP>& r,
                   int& itail,
                   int iupdat,
                   int& col,
                   int& head,
                   FP& theta,
                   FP rr,
                   FP dr,
                   FP stp,
                   FP dtd)
{
    if (iupdat <= m) {
        col   = iupdat;
        itail = (head + iupdat - 1) % m;
    } else {
        itail = (itail + 1) % m;
        head  = (head  + 1) % m;
    }

    ws.col(itail) = d;
    wy.col(itail) = r;
    theta = rr / dr;

    if (iupdat > m) {
        for (int j = 0; j < col - 1; ++j) {
            ss.col(j).head(j + 1)         = ss.col(j + 1).segment(1, j + 1);
            sy.col(j).segment(j, col - 1 - j) = sy.col(j + 1).segment(j + 1, col - 1 - j);
        }
    }

    int pointr = head;
    for (int j = 0; j < col - 1; ++j) {
        sy(col - 1, j) = d.dot(wy.col(pointr));
        ss(j, col - 1) = ws.col(pointr).dot(d);
        pointr = (pointr + 1) % m;
    }

    ss(col - 1, col - 1) = (stp == FP(1.0)) ? dtd : stp * stp * dtd;
    sy(col - 1, col - 1) = dr;
}

} // namespace min

#endif // LBFGSB_MATUPD_HPP
