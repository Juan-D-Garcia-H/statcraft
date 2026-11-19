// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file extras.hpp
/// @brief Model covariance structure at given parameters (diagonal of C, and optionally the full matrix).
///
///     C = L (Sigma_b (x) I) L^T,   block (a, b) = Sigma_b[a, b] F_a^T F_b,   Sigma_k = F_k^T F_k.
#pragma once
#include "covariance.hpp"
#include "estimating.hpp"
#include "partition.hpp"

namespace mcglm {

/// Covariance structure of the model.
struct CovarianceOut {
    VectorXd mu;                    ///< stacked mean (length K n)
    VectorXd diag_C;                ///< diag(C), always computed (cheap)
    MatrixXd C;                     ///< full (K n) x (K n) matrix; only if `full`
    std::vector<MatrixXd> Sigma;    ///< dense Sigma_k; only if `full`
    MatrixXd Sigma_between;         ///< K x K between-response correlation matrix
};

/// Computes the covariance structure at the given parameters.
/// @param full  also build the dense C and Sigma_k (O((K n)^2) memory)
/// @param part  optional block decomposition (see partition.hpp); with it `diag_C` costs O(sum m_b^3) and a sparse Z
///              is never expanded to n x n (only `full` allocates the dense result)
/// @throws NumericalError if Sigma_b or some Sigma_k is not positive definite
inline CovarianceOut covariance_matrices(const std::vector<ResponseSpec>& specs_in, const Params& par, bool full,
                                         const Partition* part = nullptr) {
    const int K = (int)specs_in.size(); const Eigen::Index n = specs_in[0].X.rows();
    if (part && part->useful) {
        CovarianceOut o; o.diag_C.resize(K * n); o.mu.resize(K * n);
        for (const Batch& B : part->batches) {
            CovarianceOut b = covariance_matrices(B.specs, par, full);
            const Eigen::Index m = (Eigen::Index)B.units.size();
            auto g = [&](int k, Eigen::Index c) { return k * n + B.units[c]; };
            for (int k = 0; k < K; ++k)
                for (Eigen::Index c = 0; c < m; ++c) { o.diag_C[g(k, c)] = b.diag_C[k * m + c]; o.mu[g(k, c)] = b.mu[k * m + c]; }
            o.Sigma_between = b.Sigma_between;
            if (!full) continue;
            if (o.C.size() == 0) { o.C = MatrixXd::Zero(K * n, K * n); o.Sigma.assign(K, MatrixXd::Zero(n, n)); }
            for (int a2 = 0; a2 < K; ++a2)
                for (int b2 = 0; b2 < K; ++b2)
                    for (Eigen::Index cc = 0; cc < m; ++cc)
                        for (Eigen::Index rr = 0; rr < m; ++rr) o.C(g(a2, rr), g(b2, cc)) = b.C(a2 * m + rr, b2 * m + cc);
            for (int k = 0; k < K; ++k)
                for (Eigen::Index cc = 0; cc < m; ++cc)
                    for (Eigen::Index rr = 0; rr < m; ++rr) o.Sigma[k](B.units[rr], B.units[cc]) = b.Sigma[k](rr, cc);
        }
        return o;
    }
    std::vector<ResponseSpec> expanded;                 // sparse Z are expanded for the single-block computation
    const std::vector<ResponseSpec>* sp = &specs_in;
    for (const auto& s : specs_in) for (const auto& z : s.Z) if (z.is_sparse) { expanded = specs_in; sp = &expanded; break; }
    if (sp == &expanded) for (auto& s : expanded) for (auto& z : s.Z) if (z.is_sparse) z = z.densified();
    const std::vector<ResponseSpec>& specs = *sp;
    MeanState ms = mean_state(specs, par.beta);
    CovarianceOut o; o.mu = ms.mu; o.diag_C.resize(K * n);
    o.Sigma_between = sigma_between(par.rho, K);
    Eigen::LLT<MatrixXd> chk(o.Sigma_between);
    if (chk.info() != Eigen::Success) throw NumericalError("the between-response correlation matrix is not positive definite (invalid rho)");
    std::vector<SigmaBlock> blocks; blocks.reserve(K);
    SigmaRequest sr; sr.need_inverse = false; sr.need_factor = full && K > 1; sr.need_deriv = false;
    for (int k = 0; k < K; ++k) {
        ResponseSpec rs_local; const ResponseSpec* rs = &specs[k];
        if (rs->ntrial.size() == 0 && (rs->variance == Variance::BinomialP || rs->variance == Variance::BinomialPQ)) {
            rs_local = *rs; rs_local.ntrial = VectorXd::Ones(n); rs = &rs_local;
        }
        blocks.push_back(build_sigma(*rs, ms.mu_k[k], par.tau[k], par.power[k], sr));
        o.diag_C.segment(k * n, n) = blocks[k].sdiag;
    }
    if (!full) return o;
    o.C = MatrixXd::Zero(K * n, K * n); o.Sigma.resize(K);
    std::vector<MatrixXd> F(K);
    for (int k = 0; k < K; ++k) {
        const SigmaBlock& b = blocks[k];
        o.Sigma[k] = b.diagonal ? MatrixXd(b.sdiag.asDiagonal()) : b.Sigma;
        if (K > 1) F[k] = b.diagonal ? MatrixXd(b.sdiag.cwiseSqrt().asDiagonal()) : b.F;
    }
    if (K == 1) { o.C = o.Sigma[0]; return o; }
    for (int a = 0; a < K; ++a) {
        o.C.block(a * n, a * n, n, n) = o.Sigma[a];
        for (int b = a + 1; b < K; ++b) {
            MatrixXd X = o.Sigma_between(a, b) * (F[a].transpose() * F[b]);
            o.C.block(a * n, b * n, n, n) = X;
            o.C.block(b * n, a * n, n, n) = X.transpose();
        }
    }
    return o;
}

} // namespace mcglm
