// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file distribution.hpp
 * @brief Numerical distribution utilities and exponential helper functions.
 * @details Provides mathematical constants and vectorized/scalar evaluation functions
 *          used across probability distribution implementations in the gengamma module.
 */

#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <cmath>
#include <limits>
#include <Eigen/Dense>

/**
 * @namespace gengamma
 * @brief Namespace containing Generalized Gamma distribution statistical routines.
 */
namespace gengamma {

/**
 * @brief Native constant representing negative infinity ($\{-\infty\}$).
 */
const double DISTRIBUTION_NEGINF = -std::numeric_limits<double>::infinity();

/**
 * @brief Evaluates boundary probability values outside the distribution support.
 * @details Computes log or natural space boundary values depending on tail selection.
 * @param lower_tail If true, evaluates lower tail probabilities; otherwise upper tail.
 * @param give_log If true, returns probability on the log scale.
 * @return Boundary probability value as a double.
 */
inline double below_distribution(bool lower_tail, bool give_log) {
    if (lower_tail) {
        if (give_log) { return DISTRIBUTION_NEGINF; } else { return 0.0; }
    } else {
        if (give_log) { return 0.0; } else { return 1.0; }
    } 
}

/**
 * @brief Vectorized conditional exponentiation utility using Eigen.
 * @details Element-wise conditional transformation that returns the original vector if log scale
 *          is requested, or exponentiates each element using Eigen array operations.
 * @param y Input Eigen column vector.
 * @param log Boolean flag indicating whether values are on log scale.
 * @return Evaluated Eigen::VectorXd result.
 */
inline Eigen::VectorXd perhaps_exp(const Eigen::VectorXd& y, bool log) {
    if (log) {
        return y;
    } else {
        return y.array().exp();
    }
}
  
/**
 * @brief Scalar overload of conditional exponentiation function.
 * @details Scalar version for individual parameter calculations, matching the vectorized interface.
 * @param y Input scalar value.
 * @param log Boolean flag indicating whether the input is on the log scale.
 * @return Computed exponentiated or identity scalar value.
 */
#ifndef DOXYGEN_SHOULD_SKIP_THIS
inline double perhaps_exp(double y, bool log) {
    if (log) {
        return y;
    } else {
        return std::exp(y);
    }
}
#endif
  
} // namespace gengamma

#endif // DISTRIBUTION_H
