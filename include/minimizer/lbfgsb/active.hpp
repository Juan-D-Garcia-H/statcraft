// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================


#ifndef LBFGSB_ACTIVE_HPP
#define LBFGSB_ACTIVE_HPP

/**
 * @file active.hpp
 * @ingroup lbfgsb_driver
 * @brief Initialize the bound-status array and clip the initial point.
 *
 * @details
 * One-time preprocessing step run by @ref mainlb before the first iteration:
 * - projects the user-supplied starting point onto the feasible box, flagging
 *   with @p prjctd whether any component had to be clipped;
 * - derives the per-variable classification stored in @p iwhere
 *   (permanently fixed variables are tagged `3`, unbounded ones `-1`, all
 *   others `0`);
 * - reports whether the problem is constrained / fully boxed.
 *
 * The code in @ref cauchy later refines the @p iwhere markers with finer
 * gradations. Original algorithm by Ciyou Zhu et al. (NEOS, November 1994;
 * latest revision June 1996), reimplemented in C++.
 *
 * @see cauchy.hpp (refines iwhere), mainlb.hpp (caller)
 */


#include "../common.hpp"
#include "logger.hpp"

namespace min {

/**
 * @brief Clip the initial point to the feasible region and set up @p iwhere.
 *
 * @details
 * Runs once per solve, right after the input validation of @ref errclb. In the
 * first loop each component of @p x is clipped onto its bounds when it strictly
 * violates them (exact bound hits leave @p x untouched); the flag @p prjctd
 * records whether any strict violation occurred. In the second loop the
 * per-variable classification is computed:
 *
 * | `nbd(i)`               | `iwhere(i)`                           |
 * |------------------------|---------------------------------------|
 * | `Unbounded`            | `-1` (always free)                    |
 * | `Both` with \f$u\f$ =\f$l\f$ | `3`  (permanently fixed)         |
 * | any other bounded kind | `0`  (bounded, to be refined)         |
 *
 * The output flags describe the whole problem: @p cnstnd is true iff at least
 * one variable carries a bound, and @p boxed is true iff **every** variable has
 * both a lower and an upper bound. Informational log messages are emitted
 * through @p logger (there is no legacy `iprint` flag).
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param l       Lower bounds (length @p n).
 * @param u       Upper bounds (length @p n).
 * @param nbd     Per-variable bound kind, see @ref BoundType.
 * @param x       In/out: initial point; clipped onto the feasible box on exit.
 * @param iwhere  On exit, per-variable classification `-1`, `3` or `0`
 *                (see table above).
 * @param logger  Logger sink for the informational messages.
 * @param prjctd  On exit, true iff some component of @p x strictly violated a
 *                bound and had to be projected.
 * @param cnstnd  On exit, true iff at least one variable has a bound.
 * @param boxed   On exit, true iff every variable has both bounds.
 */
template<typename FP = double>
inline void active(int n,
                   const Vec<FP>& l,
                   const Vec<FP>& u,
                   const Vec<BoundType>& nbd,
                   Vec<FP>& x,
                   VecInt& iwhere,
                   const Logger<FP>& logger,
                   bool& prjctd,
                   bool& cnstnd,
                   bool& boxed)
{
    prjctd   = false;
    cnstnd   = false;
    boxed    = true;

    for (int i = 0; i < n; ++i) {
        const auto bnd = nbd(i);
        if (bnd != BoundType::Unbounded) {
            if ((bnd == BoundType::Lower || bnd == BoundType::Both) && x(i) <= l(i)) {
                if (x(i) < l(i)) { prjctd = true; x(i) = l(i); }
            } else if ((bnd == BoundType::Both || bnd == BoundType::Upper) && x(i) >= u(i)) {
                if (x(i) > u(i)) { prjctd = true; x(i) = u(i); }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        const auto bnd = nbd(i);
        if (bnd != BoundType::Both) boxed = false;
        if (bnd == BoundType::Unbounded) {
            iwhere(i) = -1;
        } else {
            cnstnd    = true;
            iwhere(i) = (bnd == BoundType::Both && (u(i) - l(i)) <= FP(0.0)) ? 3 : 0;
        }
    }

    if (prjctd) {
        logger.print_info_message("Initial point x0 was infeasible; projected onto feasible bounds.");
    }
    if (!cnstnd) {
        logger.print_info_message("Problem has no active bound constraints.");
    }
}

} // namespace min

#endif // LBFGSB_ACTIVE_HPP
