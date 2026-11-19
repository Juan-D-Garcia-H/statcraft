// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_HPSOLB_HPP
#define LBFGSB_HPSOLB_HPP

/**
 * @file hpsolb.hpp
 * @ingroup lbfgsb_direction
 * @brief Breakpoint ordering helper for the Cauchy-point search.
 *
 * @details
 * Maintains the working arrays @p t (breakpoint positions) and @p iorder
 * (original indices) consumed by the Cauchy-point loop of @ref cauchy. Per the
 * classic algorithm, after every step the *smallest* unprocessed breakpoint is
 * expected at `t[nleft - 1]`; this routine keeps the array in that shape by
 * building (or repairing) a min-heap and moving the heap root — the minimum —
 * to the last active slot.
 *
 * This is a faithful 1:1 port of the reference Fortran `hpsolb.f` (Algorithm
 * 232 of CACM, J. W. J. Williams: HEAPSORT), so its behavior — including the
 * exact permutation and the tie-breaking between equal breakpoints — matches
 * the reference implementation bit for bit.
 *
 * @see cauchy.hpp (the only caller)
 */

#include <Eigen/Dense>
#include <algorithm>
#include "../common.hpp"

namespace min {

/**
 * @brief Sort the least element of the active region to `t[n - 1]` and keep
 *        the remaining elements as a min-heap.
 *
 * @details
 * Two operations, mirroring the reference `hpsolb.f`:
 * 1. **Heap build** (only when @p iheap == 0): the elements of
 *    `t[0..n-1]` are re-arranged into a binary min-heap by successive
 *    insertions — each element is appended and bubbled up while it is smaller
 *    than its parent (`t[(k+1)/2 - 1]`). On completion the root `t[0]` holds
 *    the minimum.
 * 2. **Extraction** (always, when `n > 1`): the root (the least member) is
 *    saved, the last element `t[n - 1]` is sifted down from the root through
 *    the region `[0, n - 2]` (always descending into the smaller child), and
 *    the saved minimum is finally placed at `t[n - 1]`.
 *
 * Net effect: on exit `t[n - 1]` holds the *smallest* remaining breakpoint and
 * `t[0..n-2]` the rest as a min-heap, exactly what the Cauchy loop reads next.
 * The accompanying permutation @p iorder is moved in lock-step. For `n <= 1`
 * the routine returns immediately.
 *
 * @tparam FP     Floating-point scalar type.
 * @param n       Number of elements in @p t/@p iorder.
 * @param t       In/out: breakpoint values; permuted on exit.
 * @param iorder  In/out: original index of each breakpoint; permuted in
 *                lock-step with @p t.
 * @param iheap   If zero, @p t is assumed to be *not* in heap form and the
 *                heap-build pass runs first; if non-zero, @p t is assumed to
 *                already be a min-heap on entry and only the extraction step
 *                is performed (matches how @ref cauchy calls this routine
 *                from its second iteration onward).
 */
template<typename FP = double>
inline void hpsolb(int n,
                   Vec<FP>& t,
                   VecInt& iorder,
                   int iheap)
{
    if (iheap == 0) {
        // Heap build by successive insertions (verbatim hpsolb.f, do 20 k=2,n).
        for (int k = 1; k < n; ++k) {
            FP ddum = t(k);
            const int indxin = iorder(k);
            int i = k;
            while (i > 0) {
                const int j = (i + 1) / 2 - 1;      // parent of 0-based node i
                if (ddum < t(j)) {
                    t(i) = t(j);
                    iorder(i) = iorder(j);
                    i = j;
                } else {
                    break;
                }
            }
            t(i) = ddum;
            iorder(i) = indxin;
        }
    }

    if (n > 1) {
        // Extract the heap root (the minimum) and reheapify the rest onto
        // element n-1 (verbatim hpsolb.f, lines "Assign to 'out' ...").
        const FP out   = t(0);
        const int indxou = iorder(0);
        FP ddum       = t(n - 1);
        const int indxin = iorder(n - 1);

        int i = 0;
        while (true) {
            int j = 2 * i + 1;
            if (j > n - 2) break;                   // no children within [0, n-2]
            if (j + 1 <= n - 2 && t(j + 1) < t(j)) j = j + 1;  // smaller child
            if (t(j) < ddum) {
                t(i) = t(j);
                iorder(i) = iorder(j);
                i = j;
            } else {
                break;
            }
        }
        t(i) = ddum;
        iorder(i) = indxin;

        t(n - 1) = out;
        iorder(n - 1) = indxou;
    }
}

} // namespace min

#endif // LBFGSB_HPSOLB_HPP