// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_BMV_HPP
#define LBFGSB_BMV_HPP

/**
 * @file bmv.hpp
 * @ingroup lbfgsb_matrix
 * @brief Compute the product \f$Mv\f$ of the 2m x 2m L-BFGS "middle matrix" with a vector.
 *
 * @details
 * The compact L-BFGS-B representation defines the inverse of the middle matrix
 * \f$M = \begin{pmatrix} -D & L'\\ L & \theta S'S \end{pmatrix}\f$. This routine
 * applies \f$M\f$ to a length-\f$2\mathrm{col}\f$ vector by solving the two
 * block-triangular systems with the Cholesky factor of
 * \f$\theta S'S + L D^{-1} L'\f$ (stored in @p wt by @ref formt) and reading
 * the diagonal of @p sy. It is used both by @ref cauchy (Cauchy-point update)
 * and by @ref cmprlb (subspace residual).
 *
 * The routine never allocates; it reuses fixed-size block operations on the
 * inputs. It reports dimension mismatches via @p info instead of throwing.
 *
 * @see formt.hpp, cauchy.hpp, cmprlb.hpp
 */


#include <Eigen/Dense>
#include <cmath>

#include "../common.hpp"

namespace min {

/**
 * @brief Compute \f$p \leftarrow Mv\f$ where \f$M\f$ is the L-BFGS middle matrix.
 *
 * @details
 * Algorithm (matches the classic L-BFGS-B `bmv`):
 * - if \f$\mathrm{col} \le 0\f$: sets `info = 0` and returns without touching
 *   @p p;
 * - if any operand is too small for a \f$2\mathrm{col}\f$ block — @p v or @p p
 *   shorter than \f$2\mathrm{col}\f$, or @p sy/@p wt smaller than
 *   @p col x @p col — sets `info = -1` and returns;
 * - otherwise `info = 0`, @p p is zeroed and filled with
 *   \f$a = L^{-T}D^{1/2}v\f$-type block products; specifically
 *   \f$M v = \begin{pmatrix} a\\ b\end{pmatrix}\f$ with
 *   \f$a = D^{-1/2}v_1\f$ and
 *   \f$b =J^{-1}\left(v_2 + S'Y D^{-1} v_1\right)\f$, then folded back with the
 *   off-diagonal terms of \f$S'Y\f$. The enclosing caller (@ref cauchy /
 *   @ref cmprlb) maps `info != 0` onto its own error handling.
 *
 * @tparam FP   Floating-point scalar type.
 * @param m     L-BFGS memory size (number of stored correction pairs); not used
 *              by the computation itself, kept for interface symmetry with the
 *              other matrix routines.
 * @param sy    Matrix of \f$s_i^T y_j\f$ products (\f$m\times m\f$); only the
 *              leading \f$\mathrm{col}\times\mathrm{col}\f$ block is read.
 * @param wt    Upper triangle of the Cholesky factor of
 *              \f$\theta S'S + LD^{-1}L'\f$ built by @ref formt.
 * @param col   Number of active correction pairs (\f$0 \le \mathrm{col} \le m\f$).
 * @param v     Input vector (at least \f$2\mathrm{col}\f$ entries).
 * @param p     On exit, the product \f$Mv\f$ (first \f$2\mathrm{col}\f$
 *              entries are meaningful).
 * @param info  On exit: `0` on success, `-1` when an operand is too small.
 * @note The leading \f$2\mathrm{col}\f$ entries of @p v are the only ones read.
 */


template<typename FP = double>
inline void bmv([[maybe_unused]] int m,
                const Mat<FP>& sy,
                const Mat<FP>& wt,
                int col,
                const Vec<FP>& v,
                Vec<FP>& p,
                int& info)
{
    if (col <= 0) { info = 0; return; }
    if (v.size() < 2 * col || p.size() < 2 * col ||
        sy.rows() < col   || sy.cols() < col   ||
        wt.rows() < col   || wt.cols() < col)
    {
        info = -1;
        return;
    }

    info = 0;
    p.setZero();

    p(col) = v(col);
    for (int i = 1; i < col; ++i) {
        FP sum = FP(0.0);
        for (int k = 0; k < i; ++k) {
            sum += sy(i, k) * v(k) / sy(k, k);
        }
        p(col + i) = v(col + i) + sum;
    }

    {
        // Solve J p2 = rhs.  formt stores the Cholesky factor J' in the UPPER
        // triangle of wt, so the transposed-upper view gives J (dtrsm
        // 'L','U','T','N').  (The lower triangle of wt is never written; reading
        // it here would silently degrade the solve to a diagonal one.)
        Mat<FP> wt_loc = wt.topLeftCorner(col, col);
        p.segment(col, col) =
            wt_loc.template triangularView<Eigen::Upper>()
                  .transpose()
                  .template solve<Eigen::Lower>(p.segment(col, col));
    }

    for (int i = 0; i < col; ++i) {
        p(i) = v(i) / std::sqrt(sy(i, i));
    }

    {
        Mat<FP> wt_loc = wt.topLeftCorner(col, col);
        p.segment(col, col) =
            wt_loc.template triangularView<Eigen::Upper>()
                  .template solve<Eigen::Lower>(p.segment(col, col));
    }

    for (int i = 0; i < col; ++i) {
        p(i) = -p(i) / std::sqrt(sy(i, i));
    }

    for (int i = 0; i < col; ++i) {
        FP sum = FP(0.0);
        for (int k = i + 1; k < col; ++k) {
            sum += sy(k, i) * p(col + k) / sy(i, i);
        }
        p(i) += sum;
    }
}

} // namespace min

#endif // LBFGSB_BMV_HPP
