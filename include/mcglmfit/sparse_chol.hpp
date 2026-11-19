// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file sparse_chol.hpp
/// @brief Sparse Cholesky factorisation with exact solves and the selected inverse (Takahashi recursions).
///
/// For a sparse symmetric positive-definite matrix A the factor L (with a fill-reducing permutation P,
/// P A P^T = L L^T) gives exact solves at cost O(nnz(L)) per right-hand side. The selected inverse is the set of
/// entries of A^{-1} on the sparsity pattern of L + L^T, obtained without forming the dense inverse in
/// O(sum_j m_j^2) with m_j the number of off-diagonal entries of column j of L. Every entry of A on the pattern of A
/// is contained in that set, so quantities such as tr(A^{-1} B) with B having the pattern of A are exact.
#pragma once
#include "types.hpp"
#include <algorithm>

namespace mcglm {

/// Sparse SPD solver: one symbolic analysis, many numeric factorisations on the same pattern.
class SparseSPD {
public:
    /// Symbolic analysis of the pattern of `A` (full symmetric storage; only the lower triangle is used).
    void analyze(const SpMat& A) {
        llt_.analyzePattern(A);
        if (llt_.info() != Eigen::Success) throw NumericalError("symbolic analysis of the sparse Cholesky factorisation failed");
        analyzed_ = true; factored_ = false;
    }
    /// Numeric factorisation of `A` (same pattern as in `analyze`).
    /// @throws NumericalError if A is not positive definite
    void factorize(const SpMat& A) {
        if (!analyzed_) analyze(A);
        llt_.factorize(A);
        if (llt_.info() != Eigen::Success)
            throw NumericalError("sparse Cholesky factorisation failed: the covariance matrix is not positive definite");
        factored_ = true; zinv_ready_ = false;
    }
    /// Exact solution of A x = b.
    VectorXd solve(const VectorXd& b) const { return llt_.solve(b); }
    /// Exact solution of A X = B for several right-hand sides.
    MatrixXd solve(const MatrixXd& B) const { return llt_.solve(B); }

    /// Number of non-zeros of the factor L (fill-in indicator).
    Eigen::Index nnz_factor() const { return factored_ ? tix(llt_.matrixL().nestedExpression().nonZeros()) : 0; }

    /// Computes the selected inverse for the current factorisation.
    void compute_selected_inverse() {
        if (!factored_) throw NumericalError("selected inverse requested before factorisation");
        // L in compressed column form with sorted row indices (the double transpose sorts them).
        SpMat Lraw = llt_.matrixL().nestedExpression();
        SpMat Lt = Lraw.transpose(); L_ = Lt.transpose(); L_.makeCompressed();
        const int n = toi(L_.cols());
        const int* outer = L_.outerIndexPtr(); const int* inner = L_.innerIndexPtr(); const double* val = L_.valuePtr();
        Z_.assign(uz(L_.nonZeros()), 0.0);
        // Takahashi recursions, column j from n-1 down to 0. With R = rows of column j below the diagonal, l = L(R, j):
        //   Z(R, j) = -(1/d) Z(R, R) l,      Z(j, j) = 1/d^2 - (1/d) l' Z(R, j).
        // Z(R, R) is gathered by merging the sorted row lists (chordality guarantees every needed entry is stored).
        std::vector<double> sub, acc;
        for (int j = n - 1; j >= 0; --j) {
            const int lo = outer[j], hi = outer[j + 1];               // inner[lo] == j (diagonal)
            const double d = val[lo];
            const int m = hi - lo - 1;
            if (m == 0) { Z_[uz(lo)] = 1.0 / (d * d); continue; }
            const int* R = inner + lo + 1; const double* l = val + lo + 1;
            sub.assign(uz(m) * uz(m), 0.0); acc.assign(uz(m), 0.0);
            for (int t = 0; t < m; ++t) {                               // lower triangle of Z(R, R), column t
                const int c = R[t]; int pp = outer[c]; const int pe = outer[c + 1];
                for (int s = t; s < m; ++s) {
                    while (pp < pe && inner[pp] < R[s]) ++pp;
                    if (pp == pe || inner[pp] != R[s]) throw NumericalError("selected inverse: pattern is not chordal");
                    sub[uz(s) + uz(t) * uz(m)] = Z_[uz(pp)]; ++pp;
                }
            }
            for (int t = 0; t < m; ++t) {                               // acc = Z(R, R) l using symmetry
                const double* col = sub.data() + uz(t) * uz(m);
                acc[uz(t)] += col[t] * l[t];
                for (int s = t + 1; s < m; ++s) { acc[uz(t)] += col[s] * l[s]; acc[uz(s)] += col[s] * l[t]; }
            }
            double dot = 0.0;
            for (int t = 0; t < m; ++t) { const double z = -acc[uz(t)] / d; Z_[uz(lo + 1 + t)] = z; dot += l[t] * z; }
            Z_[uz(lo)] = 1.0 / (d * d) - dot / d;
        }
        P_ = llt_.permutationP().indices();                           // original index -> permuted index (see `at`)
        zinv_ready_ = true;
    }

    /// Entry (i, j) of A^{-1} in the original ordering; (i, j) must belong to the pattern of A (or of L + L^T).
    double inverse_at(int i, int j) const {
        if (!zinv_ready_) throw NumericalError("selected inverse not computed");
        int pi = P_[i], pj = P_[j];
        if (pi < pj) std::swap(pi, pj);                                // lower triangle: row >= col
        const int* outer = L_.outerIndexPtr(); const int* inner = L_.innerIndexPtr();
        const int* b = inner + outer[pj]; const int* e = inner + outer[pj + 1];
        const int* p = std::lower_bound(b, e, pi);
        if (p == e || *p != pi) throw NumericalError("entry outside the pattern of the selected inverse");
        return Z_[uz(p - inner)];
    }

private:
    Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::AMDOrdering<int>> llt_;
    SpMat L_;
    std::vector<double> Z_;
    Eigen::VectorXi P_;
    bool analyzed_ = false, factored_ = false, zinv_ready_ = false;
};

} // namespace mcglm
