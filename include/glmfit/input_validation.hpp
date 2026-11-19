// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file input_validation.hpp
 * @brief Input validation functions shared across MleSolver and QuasiSolver classes.
 * @details Performs pre-optimization checks for matrix dimensions, finite values (NaN/Inf detection),
 *          column rank deficiency via QR pivoting, parameter identification bounds, binary indicator validity,
 *          and positivity constraints.
 */

#ifndef _GLMFIT_INPUT_VALIDATION_HPP
#define _GLMFIT_INPUT_VALIDATION_HPP

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace glm {

/**
 * @brief Verifies that all entries in a matrix or vector reference are finite.
 * @details Checks for `NaN` or `Inf` values and throws `std::invalid_argument` with row/column indices of the first non-finite entry found.
 * @param M Const reference to Eigen MatrixXd object.
 * @param name Variable identifier for exception message context.
 * @throws std::invalid_argument If non-finite values are detected.
 */
inline void check_finite(const Eigen::Ref<const Eigen::MatrixXd>& M, const std::string& name) {
    if (M.allFinite()) return;
    Eigen::Index bad_row = -1, bad_col = -1;
    for (Eigen::Index j = 0; j < M.cols() && bad_row < 0; ++j) {
        for (Eigen::Index i = 0; i < M.rows(); ++i) {
            if (!std::isfinite(M(i, j))) { bad_row = i; bad_col = j; break; }
        }
    }
    if (M.cols() == 1) {
        throw std::invalid_argument(
            name + " contains non-finite values (NaN/Inf); first occurrence at index " +
            std::to_string(bad_row) + ".");
    }
    throw std::invalid_argument(
        name + " contains non-finite values (NaN/Inf); first occurrence at row " +
        std::to_string(bad_row) + ", column " + std::to_string(bad_col) + ".");
}

/**
 * @brief Optional vector wrapper for `check_finite`.
 * @param v Optional Eigen vector to validate if present.
 * @param name Variable identifier for exception message context.
 */
inline void check_finite_opt(const std::optional<Eigen::VectorXd>& v, const std::string& name) {
    if (v.has_value()) check_finite(Eigen::Ref<const Eigen::MatrixXd>(*v), name);
}

/**
 * @brief Optional matrix wrapper for `check_finite`.
 * @param M Optional Eigen matrix to validate if present.
 * @param name Variable identifier for exception message context.
 */
inline void check_finite_opt(const std::optional<Eigen::MatrixXd>& M, const std::string& name) {
    if (M.has_value()) check_finite(Eigen::Ref<const Eigen::MatrixXd>(*M), name);
}

/**
 * @brief Validates matching row dimensions between two data structures.
 * @param n_rows_a Row count of first structure.
 * @param n_rows_b Row count of second structure.
 * @param name_a Identifier for first structure.
 * @param name_b Identifier for second structure.
 * @param fn_name Name of calling function.
 * @throws std::invalid_argument If row dimensions do not match.
 */
inline void check_dims(long n_rows_a, long n_rows_b, const std::string& name_a,
                        const std::string& name_b, const std::string& fn_name) {
    if (n_rows_a != n_rows_b) {
        throw std::invalid_argument(
            fn_name + ": " + name_a + " and " + name_b + " have mismatched row dimensions (" +
            std::to_string(n_rows_a) + " vs " + std::to_string(n_rows_b) + ").");
    }
}

/**
 * @brief Validates optional vector dimension against sample size $n$.
 * @param v Optional vector to check.
 * @param n Required number of elements.
 * @param name Identifier for optional vector.
 * @param fn_name Name of calling function.
 */
inline void check_opt_dims(const std::optional<Eigen::VectorXd>& v, long n,
                            const std::string& name, const std::string& fn_name) {
    if (v.has_value() && v->size() != n) {
        throw std::invalid_argument(
            fn_name + ": " + name + " must have size equal to nrow(X) (" + std::to_string(n) +
            "), but has size " + std::to_string(v->size()) + ".");
    }
}

/**
 * @brief Validates optional matrix row dimension against sample size $n$.
 * @param M Optional matrix to check.
 * @param n Required number of rows.
 * @param name Identifier for optional matrix.
 * @param fn_name Name of calling function.
 */
inline void check_opt_dims(const std::optional<Eigen::MatrixXd>& M, long n,
                            const std::string& name, const std::string& fn_name) {
    if (M.has_value() && M->rows() != n) {
        throw std::invalid_argument(
            fn_name + ": " + name + " must have number of rows equal to nrow(X) (" +
            std::to_string(n) + "), but has " + std::to_string(M->rows()) + ".");
    }
}

/**
 * @brief Conservative identification check ensuring parameter count $k < n$.
 * @param n Number of observations ($n$).
 * @param k Number of design matrix columns ($k$).
 * @param fn_name Name of calling function.
 * @throws std::invalid_argument If $k \ge n$.
 */
inline void check_identifiable(long n, long k, const std::string& fn_name) {
    if (k >= n) {
        throw std::invalid_argument(
            fn_name + ": column dimension of X (" + std::to_string(k) +
            ") must be strictly less than observation count (" + std::to_string(n) +
            "); model is not identifiable.");
    }
}

/**
 * @brief Verifies full column rank of design matrix X via pivoted QR decomposition.
 * @details Solvers using full parameter vector optimizers (such as L-BFGS-B) require full column rank.
 * @param X Design matrix ($n \times p$).
 * @param fn_name Calling function context.
 * @return Computed column rank of matrix X.
 * @throws std::invalid_argument If rank $< p$.
 */
inline int check_full_rank(const Eigen::MatrixXd& X, const std::string& fn_name) {
    if (X.cols() == 0) return 0;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
    const int rank = static_cast<int>(qr.rank());
    if (rank < X.cols()) {
        throw std::invalid_argument(
            fn_name + ": X is rank-deficient (rank " + std::to_string(rank) +
            " of " + std::to_string(X.cols()) + " columns). Redundant linear combinations must "
            "be removed prior to optimization.");
    }
    return rank;
}

/**
 * @brief Validates that a vector contains strictly binary values ($0$ or $1$).
 * @param d Event indicator vector.
 * @param name Variable identifier.
 * @param fn_name Calling function name.
 * @throws std::invalid_argument If any element is neither $0.0$ nor $1.0$.
 */
inline void check_binary_indicator(const Eigen::VectorXd& d, const std::string& name,
                                    const std::string& fn_name) {
    for (Eigen::Index i = 0; i < d.size(); ++i) {
        if (d(i) != 0.0 && d(i) != 1.0) {
            throw std::invalid_argument(
                fn_name + ": " + name + " must contain binary 0/1 values only; "
                "found " + std::to_string(d(i)) + " at position " +
                std::to_string(i) + ".");
        }
    }
}

/**
 * @brief Optional wrapper for `check_binary_indicator`.
 * @param d Optional binary vector.
 * @param name Variable identifier.
 * @param fn_name Calling function name.
 */
inline void check_binary_indicator(const std::optional<Eigen::VectorXd>& d, const std::string& name,
                                    const std::string& fn_name) {
    if (d.has_value()) check_binary_indicator(*d, name, fn_name);
}

/**
 * @brief Checks that all vector elements are strictly positive ($y_i > 0$).
 * @param y Input response vector.
 * @param name Variable identifier.
 * @param fn_name Calling function name.
 * @throws std::invalid_argument If any element $y_i \le 0$.
 */
inline void check_positive(const Eigen::VectorXd& y, const std::string& name,
                            const std::string& fn_name) {
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        if (!(y(i) > 0.0)) {
            throw std::invalid_argument(
                fn_name + ": " + name + " must be strictly positive (log-scale model); "
                "found " + std::to_string(y(i)) + " at position " +
                std::to_string(i) + ".");
        }
    }
}

/**
 * @brief Checks range interval consistency for truncation or censoring limits.
 * @param left Left bound limit.
 * @param right Right bound limit.
 * @param use_left Flag indicating if left bound is active.
 * @param use_right Flag indicating if right bound is active.
 * @param fn_name Calling function name.
 */
inline void check_range(double left, double right, bool use_left, bool use_right,
                         const std::string& fn_name) {
    if (use_left && !std::isfinite(left)) {
        throw std::invalid_argument(fn_name + ": left_limit must be finite when use_left=true.");
    }
    if (use_right && !std::isfinite(right)) {
        throw std::invalid_argument(fn_name + ": right_limit must be finite when use_right=true.");
    }
    if (use_left && use_right && left >= right) {
        throw std::invalid_argument(
            fn_name + ": left_limit (" + std::to_string(left) + ") must be strictly less than "
            "right_limit (" + std::to_string(right) + ").");
    }
}

/**
 * @brief Checks valid range for truncated discrete count models.
 * @param left Non-negative integer lower bound.
 * @param right Upper bound limit, or -1 for unbounded right tail.
 * @param fn_name Calling function context.
 */
inline void check_count_range(long left, long right, const std::string& fn_name) {
    if (left < 0) {
        throw std::invalid_argument(
            fn_name + ": left_limit must be >= 0 for count families (received " +
            std::to_string(left) + ").");
    }
    if (right >= 0 && right < left) {
        throw std::invalid_argument(
            fn_name + ": right_limit (" + std::to_string(right) + ") must be >= left_limit (" +
            std::to_string(left) + "), or -1 for unbounded upper tail.");
    }
}

} // namespace glm

#endif // _GLMFIT_INPUT_VALIDATION_HPP
