// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file linalg.hpp
/// @brief Dense linear algebra kernels.
///
/// With `-DMCGLM_USE_LAPACKE` (link `lapacke` and `openblas`) the Cholesky factorisation, SPD inverse and
/// triangular inverse call LAPACK (`potrf`, `potri`, `trtri`) directly. Otherwise Eigen is used, and Eigen's
/// products can still be routed to BLAS with `-DEIGEN_USE_BLAS`.
#pragma once
#include "types.hpp"
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif
#ifdef MCGLM_USE_LAPACKE
#include <lapacke.h>
#endif

namespace mcglm { namespace la {

/// Physical memory in bytes, or 0 when it cannot be determined.
inline double physical_memory_bytes() {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && size > 0) return tod(pages) * tod(size);
#endif
    return 0.0;
}

/// Refuses a dense computation that cannot fit in memory.
/// @param N       dimension of the dense matrices (K n)
/// @param copies  rough number of N x N matrices alive at the same time
/// @throws ResourceError if the estimate exceeds 90% of the physical memory
inline void require_dense_memory(Eigen::Index N, int copies) {
    const double need = tod(copies) * 8.0 * tod(N) * tod(N);
    const double have = physical_memory_bytes();
    if (have > 0 && need > 0.9 * have) {
        const double gib = 1024.0 * 1024.0 * 1024.0;
        throw ResourceError("the dense computation needs about " + std::to_string(static_cast<long long>(need / gib)) + " GiB for N = " +
                            std::to_string(static_cast<long long>(N)) + " (physical memory: " + std::to_string(static_cast<long long>(have / gib)) +
                            " GiB). The covariance couples all units in a single block, so the exact estimating "
                            "equations need dense N x N matrices. Options: a Z structure that splits into independent "
                            "groups, fewer observations, or, for one response with the identity covariance link and a "
                            "sparse Z, the approximate sparse method (structure = 'sparse')");
    }
}

/// Inverse of a symmetric positive-definite matrix, optionally returning its lower Cholesky factor.
/// @param S     symmetric positive-definite matrix
/// @param[out] inv   S^{-1}
/// @param[out] Lout  if non-null, the lower factor L with S = L L^T
/// @throws NumericalError if S is not positive definite
inline void spd_inverse(const MatrixXd& S, MatrixXd& inv, MatrixXd* Lout = nullptr) {
#ifdef MCGLM_USE_LAPACKE
    const lapack_int n = static_cast<lapack_int>(S.rows());
    inv = S;
    if (LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, inv.data(), n) != 0)
        throw NumericalError("Cholesky factorisation failed: the matrix is not positive definite");
    if (Lout) { *Lout = inv.triangularView<Eigen::Lower>(); }
    if (LAPACKE_dpotri(LAPACK_COL_MAJOR, 'L', n, inv.data(), n) != 0)
        throw NumericalError("inversion from the Cholesky factor failed (dpotri)");
    inv.triangularView<Eigen::StrictlyUpper>() = inv.transpose().triangularView<Eigen::StrictlyUpper>();
#else
    Eigen::LLT<MatrixXd> llt(S);
    if (llt.info() != Eigen::Success)
        throw NumericalError("Cholesky factorisation failed: the matrix is not positive definite");
    if (Lout) *Lout = llt.matrixL();
    inv = llt.solve(MatrixXd::Identity(S.rows(), S.rows()));
#endif
}

/// Lower Cholesky factor L of a symmetric positive-definite matrix (S = L L^T).
/// @throws NumericalError if S is not positive definite
inline MatrixXd chol_lower(const MatrixXd& S) {
    Eigen::LLT<MatrixXd> llt(S);
    if (llt.info() != Eigen::Success)
        throw NumericalError("Cholesky factorisation failed: the matrix is not positive definite");
    return llt.matrixL();
}

/// Inverse of a lower-triangular matrix.
/// @throws NumericalError if the matrix is singular (LAPACK path)
inline MatrixXd tri_inverse_lower(const MatrixXd& L) {
#ifdef MCGLM_USE_LAPACKE
    MatrixXd A = L; const lapack_int n = static_cast<lapack_int>(L.rows());
    if (LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', n, A.data(), n) != 0)
        throw NumericalError("triangular inversion failed (dtrtri): singular factor");
    return A.triangularView<Eigen::Lower>();
#else
    return L.triangularView<Eigen::Lower>().solve(MatrixXd::Identity(L.rows(), L.cols()));
#endif
}

/// Weighted trace sum_{g,h} w_g A(h,g) B(g,h) = tr(W B A) with W = diag(w), without forming the product.
/// Blocked over 64 x 64 tiles for cache efficiency; O(n^2).
inline double weighted_trace_BA(const MatrixXd& A, const MatrixXd& B, const VectorXd& w) {
    const Eigen::Index n = A.rows(); constexpr Eigen::Index T = 64;
    double s = 0;
    for (Eigen::Index jb = 0; jb < n; jb += T) {
        const Eigen::Index je = std::min(n, jb + T);
        for (Eigen::Index ib = 0; ib < n; ib += T) {
            const Eigen::Index ie = std::min(n, ib + T);
            for (Eigen::Index g = jb; g < je; ++g) {
                double acc = 0;
                const double* Ag = A.data() + g * n;           // column g of A: A(h, g)
                for (Eigen::Index h = ib; h < ie; ++h) acc += Ag[h] * B(g, h);
                s += w[g] * acc;
            }
        }
    }
    return s;
}

}} // namespace mcglm::la
