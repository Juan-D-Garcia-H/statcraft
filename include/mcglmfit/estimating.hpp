// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file estimating.hpp
/// @brief Estimating functions: quasi-score for the regression parameters and Pearson estimating function for
/// the covariance parameters, together with their sensitivity and variability matrices.
///
/// With r = y - mu, W = diag(w) (observation weights), C^{-1} the inverse covariance and D = dmu/dbeta:
///
///   quasi-score      psi_beta = D^T C^{-1} W r,           sensitivity S_beta = -D^T C^{-1} W D
///   Pearson          psi_a    = r^T P_a W C^{-1} r - tr(P_a W)   (- bias correction),   P_a = C^{-1} dC/dtheta_a
///   sensitivity      S_ab     = -tr(W P_a P_b)
///   variability      V_ab     = -2 tr(W^2 P_a P_b) + sum_i w_i^2 k4_i (P_a C^{-1})_ii (P_b C^{-1})_ii
///
/// with k4_i = r_i^4 - 3 C_ii^2 the fourth-cumulant term. For the per-observation representation all products
/// are evaluated block by block (K x K per observation) without forming N x N matrices.
#pragma once
#include "covariance.hpp"

namespace mcglm {

/// Mean structure of all responses at the current regression parameters.
struct MeanState {
    std::vector<VectorXd> mu_k;        ///< mean of each response
    std::vector<MatrixXd> D_k;         ///< n x p_k derivative of mu_k with respect to beta_k
    VectorXd mu;                       ///< stacked mean (length K n)
};

/// Evaluates the mean structure for every response.
inline MeanState mean_state(const std::vector<ResponseSpec>& specs, const std::vector<VectorXd>& beta) {
    MeanState ms; const size_t K = specs.size();
    ms.mu_k.resize(K); ms.D_k.resize(K);
    Eigen::Index n = specs[0].X.rows();
    ms.mu.resize(tix(K) * n);
    for (size_t k = 0; k < K; ++k) {
        MeanOut mo = mean_structure(specs[k].X, beta[k], specs[k].offset, specs[k].link);
        ms.mu_k[k] = mo.mu; ms.D_k[k] = std::move(mo.D);
        ms.mu.segment(tix(k) * n, n) = ms.mu_k[k];
    }
    return ms;
}

// ---------------------------- operators on CovFeatures ----------------------------

/// M = C^{-1} blockdiag(D_1, ..., D_K)   (N x P, P = total number of regression parameters).
inline MatrixXd apply_invC(const CovFeatures& cf, const std::vector<MatrixXd>& Dk) {
    const int K = cf.K; const Eigen::Index n = cf.n;
    Eigen::Index P = 0; for (auto& d : Dk) P += d.cols();
    MatrixXd M = MatrixXd::Zero(cf.N, P);
    Eigen::Index c0 = 0;
    for (int l = 0; l < K; ++l) {
        const Eigen::Index p = Dk[uz(l)].cols();
        if (!cf.obs_block) {
            M.middleCols(c0, p).noalias() = cf.inv_C.middleCols(l * n, n) * Dk[uz(l)];
        } else {
            for (int k = 0; k < K; ++k)
                M.block(k * n, c0, n, p) = Dk[uz(l)].array().colwise() * cf.invC_o.row(k + l * K).transpose().array();
        }
        c0 += p;
    }
    return M;
}

/// (dC/dtheta_a) M   (N x P).
inline MatrixXd apply_DC(const CovFeatures& cf, size_t a, const MatrixXd& M) {
    if (!cf.obs_block) {
        const SymMat& s = cf.DC[a];
        if (s.diag) return M.array().colwise() * s.d.array();
        return s.mat() * M;
    }
    const int K = cf.K; const Eigen::Index n = cf.n;
    MatrixXd out = MatrixXd::Zero(cf.N, M.cols());
    for (int k = 0; k < K; ++k)
        for (int l = 0; l < K; ++l)
            out.middleRows(k * n, n).array() += M.middleRows(l * n, n).array().colwise() *
                                                cf.DC_o[a].row(k + l * K).transpose().array();
    return out;
}

/// C^{-1} x.
inline VectorXd apply_invC_vec(const CovFeatures& cf, const VectorXd& x) {
    const int K = cf.K; const Eigen::Index n = cf.n;
    if (!cf.obs_block) return cf.inv_C * x;
    VectorXd out = VectorXd::Zero(cf.N);
    for (int k = 0; k < K; ++k)
        for (int l = 0; l < K; ++l)
            out.segment(k * n, n).array() += cf.invC_o.row(k + l * K).transpose().array() * x.segment(l * n, n).array();
    return out;
}

// ------------------------------------- quasi-score -------------------------------------

/// Quasi-score of beta with its sensitivity and (optionally) variability; `M` = C^{-1} blockdiag(D_k).
struct BetaScore { VectorXd score; MatrixXd sens, var; MatrixXd M; };

/// Quasi-score estimating function of the regression parameters.
/// @param need_var  also compute the variability D^T C^{-1} W^2 D
inline BetaScore quasi_score(const CovFeatures& cf, const MeanState& ms, const VectorXd& y,
                             const VectorXd& w, bool need_var) {
    const int K = cf.K; const Eigen::Index n = cf.n;
    BetaScore b; b.M = apply_invC(cf, ms.D_k);
    const Eigen::Index P = b.M.cols();
    VectorXd res = y - ms.mu;
    b.score = b.M.transpose() * (w.array() * res.array()).matrix();
    b.sens = MatrixXd::Zero(P, P); if (need_var) b.var = MatrixXd::Zero(P, P);
    Eigen::Index c0 = 0;
    for (int k = 0; k < K; ++k) {
        const Eigen::Index p = ms.D_k[uz(k)].cols();
        auto wk = w.segment(k * n, n);
        MatrixXd WD = ms.D_k[uz(k)].array().colwise() * wk.array();
        b.sens.middleCols(c0, p) = -(b.M.middleRows(k * n, n).transpose() * WD);
        if (need_var) {
            MatrixXd W2D = WD.array().colwise() * wk.array();
            b.var.middleCols(c0, p) = b.M.middleRows(k * n, n).transpose() * W2D;
        }
        c0 += p;
    }
    return b;
}

// -------------------------------------- Pearson --------------------------------------

/// P_a = C^{-1} dC/dtheta_a for each covariance parameter: N x N (dense) or (K*K) x n (per-observation).
struct Products { std::vector<MatrixXd> P; };

/// Forms the products P_a. A diagonal dC/dtheta_a costs O(N^2) instead of a matrix product.
inline Products make_products(const CovFeatures& cf) {
    Products pr; pr.P.reserve(cf.n_cov());
    if (!cf.obs_block) {
        for (const SymMat& s : cf.DC) {
            // order matters for dense Z: P_a = C^{-1} dC (not dC C^{-1})
            if (s.diag) pr.P.push_back(cf.inv_C.array().rowwise() * s.d.transpose().array());
            else { MatrixXd P(cf.N, cf.N); P.noalias() = cf.inv_C * s.mat(); pr.P.push_back(std::move(P)); }
        }
    } else {
        const int K = cf.K;
        for (size_t a = 0; a < cf.DC_o.size(); ++a) {
            MatrixXd Po(K * K, cf.n);
            for (Eigen::Index i = 0; i < cf.n; ++i) {
                Eigen::Map<const MatrixXd> d(cf.DC_o[a].data() + i * K * K, K, K);
                Eigen::Map<const MatrixXd> ic(cf.invC_o.data() + i * K * K, K, K);
                Eigen::Map<MatrixXd>(Po.data() + i * K * K, K, K).noalias() = ic * d;
            }
            pr.P.push_back(std::move(Po));
        }
    }
    return pr;
}

/// tr(W P_b P_a) without forming the product.
inline double pair_trace(const CovFeatures& cf, const Products& pr, size_t a, size_t b, const VectorXd& w) {
    if (!cf.obs_block) return la::weighted_trace_BA(pr.P[a], pr.P[b], w);
    const int K = cf.K; const Eigen::Index n = cf.n; double s = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const double* A = pr.P[a].data() + i * K * K; const double* B = pr.P[b].data() + i * K * K;
        for (int k = 0; k < K; ++k) {
            double acc = 0;
            for (int l = 0; l < K; ++l) acc += A[l + k * K] * B[k + l * K];
            s += w[k * n + i] * acc;
        }
    }
    return s;
}

/// r^T P_a (w .* C^{-1} r) - sum_g P_a(g,g) w_g, with `wv` = w .* C^{-1} r.
inline double score_term(const CovFeatures& cf, const Products& pr, size_t a, const VectorXd& res,
                         const VectorXd& w, const VectorXd& wv) {
    if (!cf.obs_block) {
        const MatrixXd& P = pr.P[a];
        double t1 = res.dot(P * wv);
        double tr = 0; for (Eigen::Index g = 0; g < cf.N; ++g) tr += P(g, g) * w[g];
        return t1 - tr;
    }
    const int K = cf.K; const Eigen::Index n = cf.n; double s = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        const double* A = pr.P[a].data() + i * K * K;
        for (int k = 0; k < K; ++k) {
            double acc = 0;
            for (int l = 0; l < K; ++l) acc += A[k + l * K] * wv[l * n + i];
            s += res[k * n + i] * acc - A[k + k * K] * w[k * n + i];
        }
    }
    return s;
}

/// Output of the Pearson estimating function.
struct PearsonOut {
    VectorXd score;                  ///< estimating function values (one per covariance parameter)
    MatrixXd sens, var;              ///< sensitivity and variability (when requested)
    Products products;               ///< the products P_a, kept for reuse
    VectorXd u;                      ///< u_a = r^T P_a C^{-1} W r, used for the (beta, cov) cross variability
};

/// Pearson estimating function of the covariance parameters.
/// @param inv_J_beta        inverse sensitivity of beta (used by the bias correction)
/// @param correct           include the bias-correction term -sum((D^T C^{-1} dC_a C^{-1} D) .* inv_J_beta)
/// @param need_sens         compute the sensitivity matrix
/// @param need_var          compute the variability matrix (implies the sensitivity)
/// @param need_cross        compute the cross-variability vector `u`
/// @param diag_variability  zero the off-diagonal terms of the variability (see Options::r_diag_variability)
inline PearsonOut pearson(const CovFeatures& cf, const VectorXd& y, const MeanState& ms, const MatrixXd* inv_J_beta,
                          bool correct, bool need_sens, bool need_var, bool need_cross, const VectorXd& w,
                          bool diag_variability = true) {
    const size_t q = cf.n_cov();
    PearsonOut o; o.score.resize(tix(q));
    VectorXd res = y - ms.mu;
    VectorXd v = apply_invC_vec(cf, res);
    VectorXd wv = (w.array() * v.array()).matrix();
    o.products = make_products(cf);
    for (size_t a = 0; a < q; ++a) o.score[tix(a)] = score_term(cf, o.products, a, res, w, wv);

    if (correct) {                    // bias correction
        MatrixXd M = apply_invC(cf, ms.D_k);
        for (size_t a = 0; a < q; ++a) {
            MatrixXd DM = apply_DC(cf, a, M);
            MatrixXd T = M.transpose() * DM;
            o.score[tix(a)] += -(T.array() * inv_J_beta->array()).sum();
        }
    }
    if (need_sens || need_var) {
        o.sens.resize(tix(q), tix(q));
        for (size_t a = 0; a < q; ++a)
            for (size_t b = 0; b <= a; ++b)
                o.sens(tix(a), tix(b)) = o.sens(tix(b), tix(a)) = -pair_trace(cf, o.products, a, b, w);
    }
    if (need_var) {
        VectorXd w2 = w.array().square();
        VectorXd k4 = res.array().pow(4) - 3.0 * cf.diagC.array().square();
        std::vector<VectorXd> dwe(q);
        for (size_t a = 0; a < q; ++a) {
            if (!cf.obs_block) dwe[a] = (o.products.P[a].array() * cf.inv_C.array()).rowwise().sum();
            else {
                const int K = cf.K; const Eigen::Index n = cf.n; dwe[a].resize(cf.N);
                for (Eigen::Index i = 0; i < n; ++i) {
                    Eigen::Map<const MatrixXd> A(o.products.P[a].data() + i * K * K, K, K);
                    Eigen::Map<const MatrixXd> ic(cf.invC_o.data() + i * K * K, K, K);
                    MatrixXd X = A * ic;    // diag(inv_C D_C inv_C)
                    for (int k = 0; k < K; ++k) dwe[a][k * n + i] = X(k, k);
                }
            }
        }
        o.var.resize(tix(q), tix(q));
        for (size_t a = 0; a < q; ++a)
            for (size_t b = 0; b <= a; ++b) {
                double s2 = -pair_trace(cf, o.products, a, b, w2);
                double t = (k4.array() * dwe[a].array() * dwe[b].array() * w2.array()).sum();
                o.var(tix(a), tix(b)) = o.var(tix(b), tix(a)) = (diag_variability && a != b) ? 0.0 : -2.0 * s2 + t;
            }
    }
    if (need_cross) {                 // u_a = r^T P_a (C^{-1} (w .* r))
        VectorXd z = apply_invC_vec(cf, (w.array() * res.array()).matrix());
        o.u.resize(tix(q));
        for (size_t a = 0; a < q; ++a) {
            double s = 0;
            if (!cf.obs_block) s = res.dot(o.products.P[a] * z);
            else {
                const int K = cf.K; const Eigen::Index n = cf.n;
                for (Eigen::Index i = 0; i < n; ++i) {
                    const double* A = o.products.P[a].data() + i * K * K;
                    for (int k = 0; k < K; ++k) {
                        double acc = 0; for (int l = 0; l < K; ++l) acc += A[k + l * K] * z[l * n + i];
                        s += res[k * n + i] * acc;
                    }
                }
            }
            o.u[tix(a)] = s;
        }
    }
    return o;
}


namespace detail {

/// Sums of the beta quantities over the units of a (sub)problem.
struct BetaPass { VectorXd score; MatrixXd sens, var; };

/// Sums of the Pearson quantities over the units of a (sub)problem.
struct PearsonSum { VectorXd score; MatrixXd sens, var; VectorXd u; };

inline void accumulate(BetaPass& a, const BetaPass& b) {
    a.score += b.score; a.sens += b.sens; if (b.var.size()) a.var += b.var;
}
inline void accumulate(PearsonSum& a, const PearsonSum& b) {
    a.score += b.score;
    if (b.sens.size()) a.sens += b.sens;
    if (b.var.size()) a.var += b.var;
    if (b.u.size()) a.u += b.u;
}

} // namespace detail

} // namespace mcglm
