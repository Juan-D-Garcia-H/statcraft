// ===========================================================================
// SPDX-License-Identifier: BSD-3-Clause
//
// C++ port of the Fortran L-BFGS-B code (version 3.0) by Ciyou Zhu,
// Richard H. Byrd, Jorge Nocedal and Jose Luis Morales, released by its
// authors under the BSD 3-Clause License.
// Upstream: https://users.iems.northwestern.edu/~nocedal/lbfgsb.html
// See THIRD_PARTY_NOTICES.md for the full licence text and the papers to cite.
// ===========================================================================

#ifndef LBFGSB_ERRCLB_HPP
#define LBFGSB_ERRCLB_HPP

/**
 * @file errclb.hpp
 * @ingroup lbfgsb_driver
 * @brief Validate the user-supplied bounds and stopping tolerances.
 *
 * @details
 * Performs the setup verification for the L-BFGS-B driver: problem dimension,
 * L-BFGS memory size, stopping-tolerance factor, bound types, and consistency
 * of the lower/upper bound vectors. On the first failure, it sets
 * @p task = Task::Error and records the corresponding @ref ErrorCode,
 * leaving any subsequent checks unexecuted. On success the task state (and
 * @p error_code) is left untouched.
 *
 * @see mainlb.hpp (caller), ErrorCode, BoundType
 */


#include "../common.hpp"

namespace min {

/**
 * @brief Check the validity of the input data for the L-BFGS-B driver.
 *
 * @details
 * Executes the checks in this order, each short-circuiting on failure:
 * - \f$n \le 0\f$          -> @ref ErrorCode::InvalidN;
 * - \f$m \le 0\f$          -> @ref ErrorCode::InvalidM;
 * - \f$\mathrm{factr} < 0\f$ -> @ref ErrorCode::InvalidFactr;
 * - an `nbd(i)` value outside @ref BoundType -> @ref ErrorCode::InvalidNbd
 *   with @p k set to the offending index;
 * - a variable with both bounds and \f$l(i) > u(i)\f$ -> @ref ErrorCode::InfeasibleBounds
 *   with @p k set to the offending index.
 *
 * On success neither @p task nor @p error_code is modified; @p k keeps its
 * previous value.
 *
 * @tparam FP       Floating-point scalar type.
 * @param n         Problem dimension.
 * @param m         L-BFGS memory size (number of stored correction pairs).
 * @param factr     Scaling factor of the stopping tolerance (must be >= 0).
 * @param l         Lower bounds (length @p n).
 * @param u         Upper bounds (length @p n).
 * @param nbd       Per-variable bound kind, see @ref BoundType.
 * @param task      In/out control token; set to Task::Error on failure.
 * @param error_code On exit, the error code of the first failure found
 *                  (unmodified on success).
 * @param k         On a bound-related failure, the index of the first
 *                  erroneous parameter.
 */
template<typename FP = double>
inline void errclb(int n, int m, FP factr,
                   const Vec<FP>& l, const Vec<FP>& u,
                   const Vec<BoundType>& nbd,
                   Task& task, ErrorCode& error_code, int& k)
{
    if (n <= 0) {
        task = Task::Error;
        error_code = ErrorCode::InvalidN;
        return;
    }
    if (m <= 0) {
        task = Task::Error;
        error_code = ErrorCode::InvalidM;
        return;
    }
    if (factr < FP(0.0)) {
        task = Task::Error;
        error_code = ErrorCode::InvalidFactr;
        return;
    }

    for (int i = 0; i < n; ++i) {
        const auto bnd = nbd(i);
        if (bnd != BoundType::Unbounded && bnd != BoundType::Lower &&
            bnd != BoundType::Both      && bnd != BoundType::Upper)
        {
            task = Task::Error;
            error_code = ErrorCode::InvalidNbd;
            k = i;
            return;
        }
        if (bnd == BoundType::Both && l(i) > u(i)) {
            task = Task::Error;
            error_code = ErrorCode::InfeasibleBounds;
            k = i;
            return;
        }
    }
}

} // namespace min

#endif // LBFGSB_ERRCLB_HPP
