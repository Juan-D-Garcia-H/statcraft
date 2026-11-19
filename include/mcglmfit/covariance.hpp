// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file covariance.hpp
/// @brief Joint covariance matrix C of the stacked response: its inverse, diagonal and derivatives.
///
///     C = L (Sigma_b (x) I_n) L^T,   L = blockdiag(F_k^T),   Sigma_k = F_k^T F_k.
///
/// Two representations are used:
///  - `obs_block`: every Sigma_k is diagonal, so C decouples across observations into K x K matrices;
///    all quantities are stored as (K*K) x n arrays and cost O(n K^3).
///  - dense: N x N matrices with N = K n, computed with LAPACK/BLAS.
#pragma once
#include "sigma.hpp"
#include "links.hpp"

namespace mcglm {

/// Inverse of C, diag(C) and the derivatives D_C of C with respect to every covariance parameter,
/// ordered as (rho, then for each response: exponents, tau).
struct CovFeatures {
    int K = 1; Eigen::Index n = 0, N = 0;
    bool obs_block = false;               ///< true: per-observation representation is active
    VectorXd diagC;                       ///< diag(C), length N
    // dense representation
    MatrixXd inv_C;                       ///< C^{-1} (N x N)
    std::vector<SymMat> DC;               ///< dC/d(parameter), N x N each (may reference data owned by `blocks`)
    std::vector<SigmaBlock> blocks;       ///< per-response Sigma data; keeps the references of `DC` alive (K = 1)
    // per-observation representation
    MatrixXd invC_o;                      ///< (K*K) x n: column i holds the K x K block of C^{-1} for observation i
    std::vector<MatrixXd> DC_o;           ///< each (K*K) x n
    /// Number of covariance parameters with derivatives available.
    size_t n_cov() const { return obs_block ? DC_o.size() : DC.size(); }
};

/// K x K between-response correlation matrix with unit diagonal and `rho` in row-major upper-triangle order.
/// @throws InvalidInput if rho does not have K(K-1)/2 entries
inline MatrixXd sigma_between(const VectorXd& rho, int K) {
    MatrixXd S = MatrixXd::Identity(K, K);
    if (K > 1) {
        if (rho.size() != K * (K - 1) / 2) throw InvalidInput("expected " + std::to_string(K * (K - 1) / 2) + " rho value(s), got " + std::to_string(rho.size()));
        int r = 0;
        for (int i = 0; i < K; ++i) for (int j = i + 1; j < K; ++j) { S(i, j) = S(j, i) = rho[r++]; }
    }
    return S;
}

/// What `build_C` should compute besides C^{-1}.
struct BuildRequest { bool deriv = false; };  ///< `deriv`: also compute D_C

/// Builds the joint covariance features for the current parameters.
/// @param specs  response specifications
/// @param mu_k   mean of each response
/// @param par    current parameters
/// @param rq     requested outputs
/// @param cb     cross-block derivative convention (relevant for K > 1 with dense Z)
/// @throws NumericalError if Sigma_b or any Sigma_k is not positive definite
inline CovFeatures build_C(const std::vector<ResponseSpec>& specs, const std::vector<VectorXd>& mu_k,
                           const Params& par, const BuildRequest& rq, CrossBlock cb) {
    const int K = toi(specs.size());
    const Eigen::Index n = mu_k[0].size();
    CovFeatures cf; cf.K = K; cf.n = n; cf.N = K * n;
    const bool need_factor = K > 1;
    cf.blocks.reserve(uz(K));
    SigmaRequest sr; sr.need_inverse = true; sr.need_factor = need_factor; sr.need_deriv = rq.deriv;
    for (int k = 0; k < K; ++k)
        cf.blocks.push_back(build_sigma(specs[uz(k)], mu_k[uz(k)], par.tau[uz(k)], par.power[uz(k)], sr));
    bool all_diag = true;
    for (auto& b : cf.blocks) all_diag = all_diag && b.diagonal;
    MatrixXd Sb = sigma_between(par.rho, K);
    Eigen::LLT<MatrixXd> sbllt(Sb);
    if (sbllt.info() != Eigen::Success) throw NumericalError("the between-response correlation matrix is not positive definite (invalid rho)");
    MatrixXd Sb_inv = sbllt.solve(MatrixXd::Identity(K, K));
    std::vector<std::pair<int,int>> pairs;
    for (int i = 0; i < K; ++i) for (int j = i + 1; j < K; ++j) pairs.push_back({i, j});

    cf.diagC.resize(cf.N);
    for (int k = 0; k < K; ++k) cf.diagC.segment(k * n, n) = cf.blocks[uz(k)].sdiag;

    // ============================ per-observation path ============================
    if (all_diag) {
        cf.obs_block = true;
        cf.invC_o.resize(K * K, n);
        MatrixXd sq(K, n);
        for (int k = 0; k < K; ++k) sq.row(k) = cf.blocks[uz(k)].sdiag.array().sqrt().transpose();
        for (Eigen::Index i = 0; i < n; ++i)
            for (int a = 0; a < K; ++a) for (int b = 0; b < K; ++b)
                cf.invC_o(a + b * K, i) = Sb_inv(a, b) / (sq(a, i) * sq(b, i));
        if (rq.deriv) {
            for (auto& pr : pairs) {                                     // rho
                MatrixXd m = MatrixXd::Zero(K * K, n);
                for (Eigen::Index i = 0; i < n; ++i) {
                    double v = sq(pr.first, i) * sq(pr.second, i);
                    m(pr.first + pr.second * K, i) = v; m(pr.second + pr.first * K, i) = v;
                }
                cf.DC_o.push_back(std::move(m));
            }
            for (int k = 0; k < K; ++k)
                for (auto& ds : cf.blocks[uz(k)].dSigma) {
                    MatrixXd m = MatrixXd::Zero(K * K, n);
                    for (Eigen::Index i = 0; i < n; ++i) {
                        m(k + k * K, i) = ds.d[i];
                        for (int b = 0; b < K; ++b) if (b != k) {
                            double v = Sb(k, b) * ds.d[i] / (2.0 * sq(k, i)) * sq(b, i);
                            m(k + b * K, i) = v; m(b + k * K, i) = v;
                        }
                    }
                    cf.DC_o.push_back(std::move(m));
                }
        }
        return cf;
    }

    // ================================== dense path =====================================

    if (K == 1) {
        cf.inv_C = cf.blocks[0].Sinv;
        if (rq.deriv) cf.DC = std::move(cf.blocks[0].dSigma);   // K = 1: D_C = dSigma
        return cf;
    }
    const Eigen::Index N = cf.N;
    // A response with diagonal Sigma next to a dense one: materialise its inverse and factors (F = F^T = sqrt(Sigma)).
    for (auto& b : cf.blocks) {
        if (!b.diagonal) continue;
        b.Sinv = MatrixXd(b.sdiag.cwiseInverse().asDiagonal());
        b.F = MatrixXd(b.sdiag.cwiseSqrt().asDiagonal());
        b.Finv = MatrixXd(b.sdiag.cwiseSqrt().cwiseInverse().asDiagonal());
    }
    cf.inv_C.resize(N, N);
    for (int a = 0; a < K; ++a) {
        cf.inv_C.block(a * n, a * n, n, n) = Sb_inv(a, a) * cf.blocks[uz(a)].Sinv;
        for (int b = a + 1; b < K; ++b) {
            MatrixXd X = Sb_inv(a, b) * (cf.blocks[uz(a)].Finv * cf.blocks[uz(b)].Finv.transpose());
            cf.inv_C.block(a * n, b * n, n, n) = X;
            cf.inv_C.block(b * n, a * n, n, n) = X.transpose();
        }
    }
    if (rq.deriv) {
        for (auto& pr : pairs) {                                        // rho: blocks (a,b) and (b,a)
            SymMat s; s.own = MatrixXd::Zero(N, N);
            MatrixXd X = cf.blocks[uz(pr.first)].F.transpose() * cf.blocks[uz(pr.second)].F;
            s.own.block(pr.first * n, pr.second * n, n, n) = X;
            s.own.block(pr.second * n, pr.first * n, n, n) = X.transpose();
            cf.DC.push_back(std::move(s));
        }
        for (int k = 0; k < K; ++k) {
            const SigmaBlock& sk = cf.blocks[uz(k)];
            for (auto& ds : sk.dSigma) {
                MatrixXd dS = ds.dense();
                MatrixXd G;
                if (cb == CrossBlock::RCompatible) {                    // R convention: F * tril_half(F^-1 dS F^-T)
                    MatrixXd T = sk.Finv * dS * sk.Finv.transpose();
                    MatrixXd Tl = T.triangularView<Eigen::Lower>(); Tl.diagonal() *= 0.5;
                    G = sk.F * Tl;
                } else {                                                // exact: F^T * tril_half(F^-T dS F^-1)
                    MatrixXd T = sk.Finv.transpose() * dS * sk.Finv;
                    MatrixXd Tl = T.triangularView<Eigen::Lower>(); Tl.diagonal() *= 0.5;
                    G = sk.F.transpose() * Tl;
                }
                SymMat s; s.own = MatrixXd::Zero(N, N);
                for (int b = 0; b < K; ++b) {
                    const double rkb = (b == k) ? 1.0 : Sb(k, b);
                    MatrixXd H = (cb == CrossBlock::RCompatible)
                                     ? MatrixXd(rkb * (G * cf.blocks[uz(b)].F.transpose()))
                                     : MatrixXd(rkb * (G * cf.blocks[uz(b)].F));
                    if (b == k) s.own.block(k * n, k * n, n, n) = H + H.transpose();
                    else {
                        s.own.block(k * n, b * n, n, n) = H;
                        s.own.block(b * n, k * n, n, n) = H.transpose();
                    }
                }
                cf.DC.push_back(std::move(s));
            }
        }
    }
    return cf;
}

} // namespace mcglm
