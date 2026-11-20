// SPDX-License-Identifier: Apache-2.0
//
// Rank-revealing QR decomposition with column pivoting.
//
// This is a thin, numerically robust wrapper around Eigen::ColPivHouseholderQR.
// It exposes the pieces needed by the mixed-model code: the numerical rank, the
// column permutation, the leading triangular factor, log|det| of the retained
// block, and least-squares solves in which coefficients of dependent columns are
// set to zero.
//
// A column is treated as numerically dependent when its diagonal entry of R is
// small relative to the first pivot, which is the usual relative threshold used
// by rank-revealing QR.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

namespace mixedfit {

/// Rank-revealing QR factorization of a dense matrix.
///
/// The factorization is computed once at construction time:
/// \f$ A P = Q R \f$, where `P` is a column permutation and `R` is upper
/// triangular. Only the leading `rank()` diagonal entries of `R` are considered
/// numerically nonzero.
class RankQR {
public:
    /// Compute the factorization.
    ///
    /// @param A   Matrix to factorize.
    /// @param tol Relative pivot threshold. A value `<= 0` selects
    ///            `sqrt(epsilon)` (the Eigen default). A column is marked
    ///            dependent when `|r_ii| <= tol * |r_00|`, i.e. the threshold is
    ///            relative to the first pivot.
    explicit RankQR(const Eigen::MatrixXd& A, double tol = 0.0) : qr_(A.rows(), A.cols()) {
        if (tol <= 0.0) tol = std::sqrt(std::numeric_limits<double>::epsilon());
        qr_.setThreshold(tol);
        qr_.compute(A);
        rank_ = static_cast<int>(qr_.rank());
        cols_ = static_cast<int>(A.cols());
    }

    /// Numerical rank of the factorized matrix.
    int rank() const { return rank_; }

    /// Number of columns of the factorized matrix.
    int cols() const { return cols_; }

    /// Column permutation: `pivot()[i]` is the original index of the column
    /// placed at position `i`.
    std::vector<int> pivot() const {
        const auto& idx = qr_.colsPermutation().indices();
        return std::vector<int>(idx.data(), idx.data() + idx.size());
    }

    /// Leading `rank() x rank()` upper-triangular block of `R`.
    Eigen::MatrixXd R_leading() const {
        return qr_.matrixQR().topLeftCorner(rank_, rank_).triangularView<Eigen::Upper>();
    }

    /// Natural logarithm of the absolute determinant of the leading block.
    double log_abs_det() const {
        double acc = 0.0;
        for (int i = 0; i < rank_; ++i) acc += std::log(std::abs(qr_.matrixQR()(i, i)));
        return acc;
    }

    /// Apply the transposed orthogonal factor: returns \f$ Q^T B \f$.
    Eigen::MatrixXd apply_Qt(const Eigen::MatrixXd& B) const {
        return qr_.householderQ().adjoint() * B;
    }

    /// Basic least-squares solution.
    ///
    /// Coefficients corresponding to dependent columns are set to zero, which
    /// yields the minimum-norm-like solution on the retained column subset.
    Eigen::MatrixXd solve(const Eigen::MatrixXd& B) const {
        Eigen::MatrixXd coef = Eigen::MatrixXd::Zero(cols_, B.cols());
        if (rank_ == 0) return coef;

        const Eigen::MatrixXd qtb = apply_Qt(B);
        const Eigen::MatrixXd z = qr_.matrixQR()
                                      .topLeftCorner(rank_, rank_)
                                      .triangularView<Eigen::Upper>()
                                      .solve(qtb.topRows(rank_));

        const auto piv = pivot();
        for (int i = 0; i < rank_; ++i) coef.row(piv[static_cast<size_t>(i)]) = z.row(i);
        return coef;
    }

private:
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
    int rank_ = 0;
    int cols_ = 0;
};

}  // namespace mixedfit
