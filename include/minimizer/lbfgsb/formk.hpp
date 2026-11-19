// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_FORMK_HPP
#define LBFGSB_FORMK_HPP

/**
 * @file formk.hpp
 * @ingroup lbfgsb_matrix
 * @brief Assemble and factor the indefinite subspace matrix \f$K\f$.
 *
 * @details
 * Builds the \f$2\mathrm{col}\times 2\mathrm{col}\f$ matrix
 * \f$K = \begin{pmatrix} -D & L_a' - R_z'\\ L_a - R_z & \theta S'_{AA}S \end{pmatrix}\f$
 * — equal to \f$M^{-1}N\f$ of Byrd et al. (1995) — from the accumulated inner
 * products, and computes its \f$LEL^T\f$ factorization. The two blocks are
 * assembled separately; re-used sub-blocks are kept in @p wn1 so that updates
 * only touch the newly entered/left variables. The factorized form is consumed
 * by @ref subsm to solve the subspace Newton system. Failures of either
 * Cholesky step are reported through @p info, never thrown.
 *
 * @see subsm.hpp (consumer), freev.hpp (entering/leaving information), mainlb.hpp
 */


#include <Eigen/Dense>
#include "../common.hpp"

namespace min {

/**
 * @brief Form the \f$LEL^T\f$ factorization of the indefinite matrix \f$K\f$.
 *
 * @details
 * Three phases:
 * 1. **Update phase** (when @p updatd): if \f$\mathrm{iupdat} > m\f$, the rows
 *    of @p wn1 are shifted to drop the outgoing history column; then the new
 *    row/column inner products involving the subspace variables
 *    (\f$Y'Z Z'Y\f$, \f$S'_{AA} S\f$, \f$S'_{AA} Y\f$) are computed and stored.
 * 2. **Incremental phase**: for \f$i,j\f$ in the entering variables
 *    (@p indx2[0..nenter)) and leaving variables (@p indx2[ileave..n)), the
 *    stored sub-blocks are updated using the identity
 *    \f$W'A_{(new)}W - W'A_{(old)}W\f$ contribution per entering/leaving
 *    variable, and the blocks of @p wn are filled/scaled by \f$\theta\f$.
 * 3. **Factorization**: the leading \f$\mathrm{col}\f$ block is Cholesky-
 *    factorized (`info = -1` on failure), the mixed block is reduced by
 *    triangular solves, and the trailing block is updated and factorized
 *    (`info = -2` on failure).
 *
 * When \f$\mathrm{col} \le 0\f$ the routine returns immediately with `info=0`,
 * leaving @p wn untouched.
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension (defines the trailing loop extent).
 * @param nsub    Number of free (subspace) variables.
 * @param ind     Indices of the free variables (length @p nsub).
 * @param nenter  Number of variables entering the free set.
 * @param ileave  Start index of the leaving segment in @p indx2.
 * @param indx2   Lists entering variables in \f$[0,\mathrm{nenter})\f$ and
 *                leaving ones in \f$[\mathrm{ileave},n)\f$.
 * @param iupdat  Total number of L-BFGS updates performed so far.
 * @param updatd  True if the L-BFGS memory was updated this iteration.
 * @param wn      In/out: on exit stores the \f$LEL^T\f$ factorization of
 *                \f$K\f$ in its upper triangle.
 * @param wn1     In/out: cached sub-blocks of \f$K\f$ enabling incremental
 *                updates across iterations.
 * @param m       L-BFGS memory size.
 * @param ws      S history matrix (\f$n\times m\f$).
 * @param wy      Y history matrix (\f$n\times m\f$).
 * @param sy      Matrix of \f$S'Y\f$ inner products (\f$m\times m\f$).
 * @param theta   Initial-Hessian scaling \f$B_0 = \theta I\f$.
 * @param col     Number of active correction pairs.
 * @param head    Index of the oldest stored correction (start of the circular
 *                walk over the history columns).
 * @param info    On exit: `0` success, `-1` first Cholesky failed, `-2` second
 *                Cholesky failed.
 * @note The viewing bounds checks embedded in the loops make the routine safe
 *       against undersized workspace, skipping the affected inner products.
 */


template<typename FP = double>
inline void formk(int n,
                  int nsub,
                  const VecInt& ind,
                  int nenter,
                  int ileave,
                  const VecInt& indx2,
                  int iupdat,
                  bool updatd,
                  Mat<FP>& wn,
                  Mat<FP>& wn1,
                  int m,
                  const Mat<FP>& ws,
                  const Mat<FP>& wy,
                  const Mat<FP>& sy,
                  FP theta,
                  int col,
                  int head,
                  int& info)
{
    info = 0;
    if (col <= 0) return;

    const int pbegin = 0, pend = nsub;
    const int dbegin = nsub, dend = n;

    if (updatd) {
        if (iupdat > m) {
            for (int jy = 0; jy < m - 1; ++jy) {
                int js = m + jy;
                if (jy < wn1.rows() && jy < wn1.cols() && 
                    jy + 1 < wn1.rows() && jy + 1 < wn1.cols() &&
                    js < wn1.rows() && js < wn1.cols() &&
                    js + 1 < wn1.rows() && js + 1 < wn1.cols() &&
                    m < wn1.rows() && jy < wn1.cols() &&
                    m + 1 < wn1.rows() && jy + 1 < wn1.cols())
                {
                    wn1.block(jy, jy, m - 1 - jy, 1) = wn1.block(jy + 1, jy + 1, m - 1 - jy, 1);
                    wn1.block(js, js, m - 1 - jy, 1) = wn1.block(js + 1, js + 1, m - 1 - jy, 1);
                    wn1.block(m,  jy, m - 1,    1) = wn1.block(m + 1,  jy + 1, m - 1,    1);
                }
            }
        }

        const int iy    = col - 1;
        const int is    = m + col - 1;
        int       ipntr = (head + col - 1) % m;
        int       jpntr = head;

        for (int jy = 0; jy < col; ++jy) {
            const int js = m + jy;
            FP t1 = FP(0.0), t2 = FP(0.0), t3 = FP(0.0);
            for (int k = pbegin; k < pend; ++k) {
                if (k < ind.size()) {
                    const int k1 = ind(k);
                    if (k1 < wy.rows() && ipntr < wy.cols() && jpntr < wy.cols()) {
                        t1 += wy(k1, ipntr) * wy(k1, jpntr);
                    }
                }
            }
            for (int k = dbegin; k < dend; ++k) {
                if (k < ind.size()) {
                    const int k1 = ind(k);
                    if (k1 < ws.rows() && ipntr < ws.cols() && jpntr < ws.cols()) {
                        t2 += ws(k1, ipntr) * ws(k1, jpntr);
                        t3 += ws(k1, ipntr) * wy(k1, jpntr);
                    }
                }
            }
            if (iy < wn1.rows() && jy < wn1.cols()) wn1(iy, jy) = t1;
            if (is < wn1.rows() && js < wn1.cols()) wn1(is, js) = t2;
            if (is < wn1.rows() && jy < wn1.cols()) wn1(is, jy) = t3;
            jpntr = (jpntr + 1) % m;
        }

        {
            const int jy_col = col - 1;
            int jpntr2 = (head + col - 1) % m;
            int ipntr2 = head;
            for (int i = 0; i < col; ++i) {
                const int is2 = m + i;
                FP t3 = FP(0.0);
                for (int k = pbegin; k < pend; ++k) {
                    if (k < ind.size()) {
                        const int k1 = ind(k);
                        if (k1 < ws.rows() && ipntr2 < ws.cols() && jpntr2 < wy.cols()) {
                            t3 += ws(k1, ipntr2) * wy(k1, jpntr2);
                        }
                    }
                }
                ipntr2 = (ipntr2 + 1) % m;
                if (is2 < wn1.rows() && jy_col < wn1.cols()) wn1(is2, jy_col) = t3;
            }
        }
    }

    const int upcl = updatd ? col - 1 : col;

    if (upcl > 0) {
        int ipntr = head;
        for (int iy = 0; iy < upcl; ++iy) {
            const int is = m + iy;
            int jpntr = head;
            for (int jy = 0; jy <= iy; ++jy) {
                const int js = m + jy;
                FP t1 = FP(0), t2 = FP(0), t3 = FP(0), t4 = FP(0);
                for (int k = 0; k < nenter; ++k) {
                    if (k < indx2.size()) {
                        const int k1 = indx2(k);
                        if (k1 < wy.rows() && ipntr < wy.cols() && jpntr < wy.cols()) {
                            t1 += wy(k1, ipntr) * wy(k1, jpntr);
                            t2 += ws(k1, ipntr) * ws(k1, jpntr);
                        }
                    }
                }
                for (int k = ileave; k < n; ++k) {
                    if (k < indx2.size()) {
                        const int k1 = indx2(k);
                        if (k1 < wy.rows() && ipntr < wy.cols() && jpntr < wy.cols()) {
                            t3 += wy(k1, ipntr) * wy(k1, jpntr);
                            t4 += ws(k1, ipntr) * ws(k1, jpntr);
                        }
                    }
                }
                if (iy < wn1.rows() && jy < wn1.cols()) wn1(iy, jy) += t1 - t3;
                if (is < wn1.rows() && js < wn1.cols()) wn1(is, js) += -t2 + t4;
                jpntr = (jpntr + 1) % m;
            }
            ipntr = (ipntr + 1) % m;
        }

        int ipntr2 = head;
        for (int is = m; is < m + upcl; ++is) {
            int jpntr = head;
            for (int jy = 0; jy < upcl; ++jy) {
                FP t1 = FP(0), t3 = FP(0);
                for (int k = 0; k < nenter; ++k) {
                    if (k < indx2.size()) {
                        const int k1 = indx2(k);
                        if (k1 < ws.rows() && ipntr2 < ws.cols() && jpntr < wy.cols()) {
                            t1 += ws(k1, ipntr2) * wy(k1, jpntr);
                        }
                    }
                }
                for (int k = ileave; k < n; ++k) {
                    if (k < indx2.size()) {
                        const int k1 = indx2(k);
                        if (k1 < ws.rows() && ipntr2 < ws.cols() && jpntr < wy.cols()) {
                            t3 += ws(k1, ipntr2) * wy(k1, jpntr);
                        }
                    }
                }
                if (is < wn1.rows() && jy < wn1.cols()) {
                    wn1(is, jy) += (is <= jy + m) ? (t1 - t3) : (-t1 + t3);
                }
                jpntr = (jpntr + 1) % m;
            }
            ipntr2 = (ipntr2 + 1) % m;
        }
    }

    for (int iy = 0; iy < col; ++iy) {
        const int is  = col + iy;
        const int is1 = m + iy;
        for (int jy = 0; jy <= iy; ++jy) {
            const int js  = col + jy;
            const int js1 = m + jy;
            if (jy < wn.cols() && iy < wn.rows()) wn(jy, iy) = wn1(iy, jy) / theta;
            if (js < wn.cols() && is < wn.rows()) wn(js, is) = wn1(is1, js1) * theta;
        }
        for (int jy = 0; jy < iy; ++jy) {
            if (jy < wn.cols() && is < wn.rows()) wn(jy, is) = -wn1(is1, jy);
        }
        for (int jy = iy; jy < col; ++jy) {
            if (jy < wn.cols() && is < wn.rows()) wn(jy, is) = wn1(is1, jy);
        }
        if (iy < wn.rows() && iy < wn.cols() && iy < sy.rows() && iy < sy.cols()) {
            wn(iy, iy) += sy(iy, iy);
        }
    }

    if (col > 0) {
        Eigen::LLT<Mat<FP>> llt;
        llt.compute(wn.topLeftCorner(col, col).template selfadjointView<Eigen::Upper>());
        if (llt.info() != Eigen::Success) { info = -1; return; }
        wn.topLeftCorner(col, col).template triangularView<Eigen::Upper>() = llt.matrixU();

        for (int js = col; js < 2 * col && js < wn.cols(); ++js) {
            wn.col(js).head(col) = wn.topLeftCorner(col, col)
                .template triangularView<Eigen::Upper>().transpose().solve(wn.col(js).head(col));
        }

        for (int is = col; is < 2 * col && is < wn.rows(); ++is) {
            for (int js = is; js < 2 * col && js < wn.cols(); ++js) {
                wn(is, js) += wn.col(is).head(col).dot(wn.col(js).head(col));
            }
        }

        Eigen::LLT<Mat<FP>> llt2;
        llt2.compute(wn.block(col, col, col, col).template selfadjointView<Eigen::Upper>());
        if (llt2.info() != Eigen::Success) { info = -2; return; }
        wn.block(col, col, col, col).template triangularView<Eigen::Upper>() = llt2.matrixU();
    }
}

} // namespace min

#endif // LBFGSB_FORMK_HPP
