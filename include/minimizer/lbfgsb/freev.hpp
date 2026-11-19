// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_FREEV_HPP
#define LBFGSB_FREEV_HPP

/**
 * @file freev.hpp
 * @ingroup lbfgsb_driver
 * @brief Track variables entering/leaving the free set and re-index free vs. active variables.
 *
 * @details
 * Post-processing step run by @ref mainlb right after the generalized Cauchy
 * point (GCP) computation:
 * - when @p iter > 0 (there is a previous iteration to compare against), counts
 *   the variables that changed status since then and records them in @p indx2;
 * - decides whether the subspace workspace must be rebuilt (@p wrk);
 * - finally re-partitions @p index so that the @p nfree free variables come
 *   first, followed by the active (at-bound) variables.
 *
 * The `index`/`iwhere` conventions consumed here are produced by @ref cauchy.
 *
 * @see cauchy.hpp, mainlb.hpp (caller), formk.hpp
 */


#include <Eigen/Dense>
#include "../common.hpp"

namespace min {

template<typename FP = double>
/**
 * @brief Count variables entering/leaving the free set and re-sort @p index.
 *
 * @details
 * Three tasks are performed:
 * 1. **Change detection** (only when @p iter > 0 and @p cnstnd): the free
 *    variables from the *previous* iteration (given by the current contents of
 *    @p index and @p nfree) are compared against the current classification
 *    @p iwhere. Those that became at-bound are moved to the end of @p indx2
 *    (segment \f$[\mathrm{ileave},\,n)\f$), and those previously at-bound that
 *    became free occupy the front of @p indx2 (segment
 *    \f$[0,\,\mathrm{nenter})\f$).
 * 2. **Workspace flag**: @p wrk is set when any variable entered/left the free
 *    set or the L-BFGS memory was updated, signalling @ref mainlb that the
 *    subspace matrix must be rebuilt via @ref formk.
 * 3. **Re-indexing**: @p index is rewritten so
 *    \f$\mathrm{index}[0,\mathrm{nfree})\f$ lists the free variables (in index
 *    order) and \f$\mathrm{index}[\mathrm{nfree},n)\f$ the at-bound ones (in
 *    reverse index order).
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Problem dimension.
 * @param nfree   In: number of free variables from the previous iteration.
 *                Out: number of free variables at the GCP.
 * @param index   In/out: permutation of `0..n-1` described above.
 * @param nenter  On exit, number of variables that entered the free set.
 * @param ileave  On exit, start index of the leaving segment in @p indx2
 *                (decremented from @p n as leavers are recorded; equals @p n
 *                when none).
 * @param indx2   On exit, lists the entering variables in
 *                \f$[0,\mathrm{nenter})\f$ and the leaving ones in
 *                \f$[\mathrm{ileave},n)\f$.
 * @param iwhere  Per-variable status at the GCP produced by @ref cauchy
 *                (`<= 0` free, `> 0` at-bound).
 * @param wrk     On exit, true iff the subspace workspace `wn`/`wn1` must be
 *                rebuilt.
 * @param updatd  True if the L-BFGS memory was updated since the previous
 *                iteration.
 * @param cnstnd  True if the problem has at least one bound; when false the
 *                change-detection loop is skipped.
 * @param iter    Current outer iteration index (change detection runs only for
 *                `iter > 0`).
 */
inline void freev(int n,
                  int& nfree,
                  VecInt& index,
                  int& nenter,
                  int& ileave,
                  VecInt& indx2,
                  const VecInt& iwhere,
                  bool& wrk,
                  bool updatd,
                  bool cnstnd,
                  int iter)
{
    nenter = 0;
    ileave = n;

    if (iter > 0 && cnstnd) {
        for (int i = 0; i < nfree; ++i) {
            const int k = index(i);
            if (iwhere(k) > 0) {
                --ileave;
                indx2(ileave) = k;
            }
        }
        for (int i = nfree; i < n; ++i) {
            const int k = index(i);
            if (iwhere(k) <= 0) {
                indx2(nenter) = k;
                ++nenter;
            }
        }
    }

    wrk   = (ileave < n) || (nenter > 0) || updatd;
    nfree = 0;
    int iact = n;

    for (int i = 0; i < n; ++i) {
        if (iwhere(i) <= 0) {
            index(nfree) = i;
            ++nfree;
        } else {
            --iact;
            index(iact) = i;
        }
    }
}

} // namespace min

#endif // LBFGSB_FREEV_HPP
