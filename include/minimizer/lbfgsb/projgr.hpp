// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_PROJGR_HPP
#define LBFGSB_PROJGR_HPP

/**
 * @file projgr.hpp
 * @ingroup lbfgsb_driver
 * @brief Compute the infinity norm of the projected gradient.
 *
 * @details
 * Projects the gradient onto the feasible box defined by the bound vectors
 * @p l / @p u and the per-variable bound types @p nbd, then returns the largest
 * absolute component of the projected gradient. The projected gradient being
 * zero is the KKT condition for a stationary point of the box-constrained
 * problem, so the driver uses the value produced here as its convergence
 * measure (compared against the user tolerance `pgtol`).
 *
 * The projection is purely componentwise:
 * - if \f$g_i < 0\f$ and variable \f$i\f$ has an upper (or both) bound, the
 *   projected component becomes \f$\max\left(x_i-u_i,\;g_i\right)\f$;
 * - if \f$g_i \ge 0\f$ and variable \f$i\f$ has a lower (or both) bound, the
 *   projected component becomes \f$\min\left(x_i-l_i,\;g_i\right)\f$;
 * - otherwise the component is simply \f$g_i\f$.
 *
 * @see mainlb.hpp (caller), active.hpp (bound pre-processing)
 */


#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include "../common.hpp"

namespace min {

/**
 * @brief Compute \f$\|P(x-g)\|_\infty\f$, the infinity norm of the projected gradient.
 *
 * @details
 * Called once per outer iteration by @ref mainlb, both on the first evaluation
 * and after every new iterate, to test the convergence condition
 * \f$\mathrm{sbgnrm} \le \mathrm{pgtol}\f$.
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param l       Lower bounds (length @p n).
 * @param u       Upper bounds (length @p n).
 * @param nbd     Per-variable bound kind, see @ref BoundType.
 * @param x       Current iterate.
 * @param g       Gradient at @p x.
 * @param sbgnrm  On exit, the largest absolute projected-gradient component
 *                (always non-negative); overwritten regardless of the input.
 * @note The input vectors @p x and @p g are not modified.
 */
template<typename FP = double>
inline void projgr(int n,
                   const Vec<FP>& l,
                   const Vec<FP>& u,
                   const Vec<BoundType>& nbd,
                   const Vec<FP>& x,
                   const Vec<FP>& g,
                   FP& sbgnrm)
{
    sbgnrm = FP(0.0);
    for (int i = 0; i < n; ++i) {
        FP gi = g(i);
        const auto bnd = nbd(i);
        if (bnd != BoundType::Unbounded) {
            if (gi < FP(0.0)) {
                if (bnd == BoundType::Both || bnd == BoundType::Upper) {
                    gi = std::max(x(i) - u(i), gi);
                }
            } else {
                if (bnd == BoundType::Lower || bnd == BoundType::Both) {
                    gi = std::min(x(i) - l(i), gi);
                }
            }
        }
        sbgnrm = std::max(sbgnrm, std::abs(gi));
    }
}

} // namespace min

#endif // LBFGSB_PROJGR_HPP
