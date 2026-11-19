// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file sparse_backend.hpp
/// @brief Approximate method for one large connected block: sparse Cholesky + selected inverse + stochastic traces.
///
/// Applies to one response (K = 1) with the identity covariance link and diagonal or sparse Z, so that
/// C = V Omega V (+ diag(mu) terms) is sparse. The dense inverse of C that the exact equations need is avoided:
///
///  exact (up to rounding)   quasi-score and its sensitivity (solves with the sparse factor), the Pearson estimating
///                           function including its trace term tr(W C^{-1} dC/dtheta), computed from the selected
///                           inverse (Takahashi), the bias correction and the cross-variability vector;
///  stochastic               the traces of products of two inverses, tr(W P_a P_b) and tr(W^2 P_a P_b), and the
///                           diagonals diag(C^{-1} dC_a C^{-1}), estimated with fixed Rademacher probes (Hutchinson).
///
/// Because the estimating function is exact, the converged estimates are the exact ones: the stochastic quantities
/// only enter the Newton-type step (an inexact Jacobian, which changes the speed of convergence, not the limit) and
/// the sensitivity/variability matrices of the covariance parameters, hence the standard errors of those parameters.
/// The probes are fixed (same for every iteration), so the algorithm is deterministic. The relative Monte Carlo
/// standard error of the stochastic quantities is reported in `FitResult::mc_rel_error`; it decreases like
/// 1/sqrt(probes * effective rank), so it is small for large n.
#pragma once
#include "estimating.hpp"
#include "sparse_chol.hpp"
#include "variance.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace mcglm { namespace detail {

/// True when the sparse method can handle the model (K = 1, identity covariance link).
inline bool sparse_method_applicable(const std::vector<ResponseSpec>& specs) {
    return specs.size() == 1 && specs[0].covariance == CovLink::Identity;
}

/// Largest fraction of non-zero entries among the non-diagonal Z matrices (0 when all are diagonal).
inline double z_density(const std::vector<ResponseSpec>& specs) {
    double d = 0.0;
    for (const auto& s : specs) for (const auto& z : s.Z) {
        if (z.is_diag) continue;
        const double n = tod(z.n());
        if (z.is_sparse) d = std::max(d, tod(z.sparse.nonZeros()) / (n * n));
        else {
            long long nz = 0; const double* p = z.dense.data();
            for (long long i = 0, m = static_cast<long long>(z.n()) * static_cast<long long>(z.n()); i < m; ++i) nz += (p[i] != 0.0);
            d = std::max(d, tod(nz) / (n * n));
        }
    }
    return d;
}

inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31);
}

/// Optional phase timing (set MCGLM_PROFILE=1) printed to stderr, to locate the cost of the sparse method.
struct PhaseTimer {
    bool on = std::getenv("MCGLM_PROFILE") != nullptr;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void lap(const char* what) {
        if (!on) return;
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "  [sparse] %-18s %.3fs\n", what, std::chrono::duration<double>(t1 - t0).count());
        t0 = t1;
    }
};

class SparseBackend {
public:
    SparseBackend(const std::vector<ResponseSpec>& specs, const VectorXd& y, const VectorXd& w, const Options& opt)
        : specs_(&specs), y_(y), w_(w), w2_(w.array().square()) {
        if (!sparse_method_applicable(specs))
            throw InvalidInput("the sparse method requires one response with the identity covariance link");
        spec_ = &specs[0]; n_ = spec_->X.rows();
        probes_loop_ = std::max(2, opt.n_probes); probes_final_ = std::max(8, std::max(probes_loop_, opt.n_probes_final));
        seed_ = static_cast<std::uint64_t>(opt.probe_seed);
        w_ones_ = (w_.array() == 1.0).all();
        build_pattern();
    }

    BetaPass beta_pass(const Params& par, bool need_var, bool /*keep*/) {
        PhaseTimer pt;
        ms_ = mean_state(*specs_, par.beta);
        assemble(par, ms_.mu_k[0]);
        pt.lap("assemble+factorize");
        const MatrixXd& D = ms_.D_k[0];
        MatrixXd M = solver_.solve(D);
        VectorXd res = y_ - ms_.mu;
        BetaPass b;
        b.score = M.transpose() * w_.cwiseProduct(res);
        MatrixXd WD = D.array().colwise() * w_.array();
        b.sens = -(M.transpose() * WD);
        if (need_var) b.var = M.transpose() * (WD.array().colwise() * w_.array()).matrix();
        return b;
    }

    PearsonSum cov_pass(const Params& par, const MatrixXd* inv_J, bool correct, bool need_sens, bool need_var,
                        bool need_cross, bool diag_variability, bool reuse) {
        PhaseTimer pt;
        MeanState ms2 = mean_state(*specs_, par.beta);
        if (!reuse) assemble(par, ms2.mu_k[0]);
        pt.lap("assemble");
        const size_t q = A_.size(); const Eigen::Index qi = tix(q);
        const MatrixXd& D = ms2.D_k[0];
        VectorXd res = y_ - ms2.mu;
        PearsonSum o; o.score.resize(qi);

        // ---- exact part: score = s1' A_a (w .* s1) - tr(W C^{-1} A_a), with s1 = C^{-1} r
        VectorXd s1 = solver_.solve(res), wv = w_.cwiseProduct(s1);
        pt.lap("solve+setup");
        solver_.compute_selected_inverse();
        pt.lap("selected inverse");
        std::vector<double> zpat(nnz_);
        for (size_t p = 0; p < nnz_; ++p) zpat[p] = solver_.inverse_at(row_[p], col_[p]);
        VectorXd tmp(n_);
        for (size_t a = 0; a < q; ++a) {
            spmv(A_[a], wv, tmp);
            double tr = 0.0;
            for (size_t p = 0; p < nnz_; ++p) tr += A_[a][p] * zpat[p] * w_[col_[p]];
            o.score[tix(a)] = s1.dot(tmp) - tr;
        }
        if (correct) {                                         // -sum( (M' A_a M) .* inv_J ),  M = C^{-1} D
            MatrixXd M = solver_.solve(D), AM(n_, M.cols());
            for (size_t a = 0; a < q; ++a) {
                for (int c = 0; c < M.cols(); ++c) { VectorXd col = M.col(c); spmv(A_[a], col, tmp); AM.col(c) = tmp; }
                o.score[tix(a)] -= ((M.transpose() * AM).array() * inv_J->array()).sum();
            }
        }
        if (need_cross) {                                      // u_a = s1' A_a C^{-1}(w .* r)
            VectorXd z2 = solver_.solve(VectorXd(w_.cwiseProduct(res)));
            o.u.resize(qi);
            for (size_t a = 0; a < q; ++a) { spmv(A_[a], z2, tmp); o.u[tix(a)] = s1.dot(tmp); }
        }
        pt.lap("exact score/correction");
        if (!(need_sens || need_var)) return o;

        // ---- stochastic part: tr(W P_a P_b), tr(W^2 P_a P_b), diag(C^{-1} A_a C^{-1})
        const bool final_quality = need_var || need_cross;
        int S = final_quality ? probes_final_ : probes_loop_;
        if (need_var) S = std::max(8, S / 8 * 8);                  // 4 replicates x 2 independent probe sets
        VectorXd z(n_), cz(n_);
        MatrixXd T(n_, qi), R(n_, qi), R2(n_, qi), B(n_, qi);
        // One probe: e1 = sym(r_a . p_b) estimates tr(W P_a P_b); with `var` also e2 (for W^2) and zy = z .* diag-probe.
        auto one_probe = [&](int i, bool var, MatrixXd& e1, MatrixXd& e2, MatrixXd& zy) {
            probe(i, z);
            for (size_t b = 0; b < q; ++b) { spmv(A_[b], z, tmp); T.col(tix(b)) = tmp; }
            MatrixXd Pm = solver_.solve(T);                                    // p_b = C^{-1} A_b z
            VectorXd u = solver_.solve(VectorXd(w_.cwiseProduct(z)));          // C^{-1} W z
            for (size_t a = 0; a < q; ++a) { spmv(A_[a], u, tmp); R.col(tix(a)) = tmp; }
            e1 = R.transpose() * Pm; e1 = 0.5 * (e1 + e1.transpose());
            if (!var) return;
            if (w_ones_) R2 = R;
            else {
                VectorXd u2 = solver_.solve(VectorXd(w2_.cwiseProduct(z)));
                for (size_t a = 0; a < q; ++a) { spmv(A_[a], u2, tmp); R2.col(tix(a)) = tmp; }
            }
            e2 = R2.transpose() * Pm; e2 = 0.5 * (e2 + e2.transpose());
            cz = solver_.solve(z);
            for (size_t a = 0; a < q; ++a) { spmv(A_[a], cz, tmp); B.col(tix(a)) = tmp; }
            MatrixXd Y = solver_.solve(B);                                     // C^{-1} A_a C^{-1} z
            zy = Y.array().colwise() * z.array();
        };
        MatrixXd e1(qi, qi), e2(qi, qi), zy(n_, qi);
        MatrixXd sum1 = MatrixXd::Zero(qi, qi), sq1 = MatrixXd::Zero(qi, qi);
        MatrixXd Vrep[4];
        if (!need_var) {
            for (int i = 0; i < S; ++i) { one_probe(i, false, e1, e2, zy); sum1 += e1; sq1 += e1.cwiseAbs2(); }
        } else {
            const VectorXd k4 = res.array().pow(4) - 3.0 * diagC_.array().square();
            const VectorXd kw = k4.cwiseProduct(w2_);
            for (int r = 0; r < 4; ++r) {
                MatrixXd Dsum[2] = {MatrixXd::Zero(n_, qi), MatrixXd::Zero(n_, qi)}, tr2 = MatrixXd::Zero(qi, qi);
                int cnt[2] = {0, 0}, ntr = 0;
                for (int s2 = 0; s2 < 2; ++s2)
                    for (int i = r + 4 * s2; i < S; i += 8) {
                        one_probe(i, true, e1, e2, zy);
                        sum1 += e1; sq1 += e1.cwiseAbs2(); tr2 += e2; ++ntr;
                        Dsum[s2] += zy; ++cnt[s2];
                    }
                MatrixXd t = (Dsum[0] / cnt[0]).transpose() * ((Dsum[1] / cnt[1]).array().colwise() * kw.array()).matrix();
                t = 0.5 * (t + t.transpose());                                 // independent probe sets: unbiased product
                Vrep[r] = 2.0 * tr2 / ntr + t;
            }
        }
        pt.lap("probes");
        // relative Monte Carlo standard error: per-probe variance for S, replicate spread for the variability
        const MatrixXd Smean = sum1 / S;
        const MatrixXd Svar = (sq1 / S - Smean.cwiseAbs2()) * (tod(S) / std::max(1, S - 1));
        auto rel = [&](const MatrixXd& se, const MatrixXd& scale) {
            double worst = 0.0;
            for (size_t a = 0; a < q; ++a) for (size_t b = a; b < q; ++b) {
                const double sc = std::sqrt(std::fabs(scale(tix(a), tix(a)) * scale(tix(b), tix(b))));
                if (sc > 0) worst = std::max(worst, se(tix(a), tix(b)) / sc);
            }
            return worst;
        };
        mc_error_ = rel(Svar.cwiseMax(0.0).cwiseSqrt() / std::sqrt(tod(S)), Smean); probes_used_ = S;
        o.sens = -Smean;
        if (need_var) {
            MatrixXd V = (Vrep[0] + Vrep[1] + Vrep[2] + Vrep[3]) / 4.0, ss = MatrixXd::Zero(qi, qi);
            for (int r = 0; r < 4; ++r) ss += (Vrep[r] - V).cwiseAbs2();
            const MatrixXd se = (ss / 3.0).cwiseSqrt() / 2.0;                    // sample std of the replicates / sqrt(4)
            mc_error_ = std::max(mc_error_, rel(se, V));
            o.var = V;
            if (diag_variability) { MatrixXd dg = o.var.diagonal().asDiagonal(); o.var = dg; }
        }
        return o;
    }

    VectorXd mu(const Params& par) const { return mean_state(*specs_, par.beta).mu; }
    double mc_error() const { return mc_error_; }
    int probes_used() const { return probes_used_; }

private:
    const std::vector<ResponseSpec>* specs_;
    const ResponseSpec* spec_ = nullptr;
    VectorXd y_, w_, w2_;
    Eigen::Index n_ = 0;
    bool w_ones_ = false;
    int probes_loop_ = 32, probes_final_ = 128;
    std::uint64_t seed_ = 0;

    // pattern shared by Omega, Sigma and every dSigma (full symmetric storage, diagonal always present)
    SpMat Sigma_;
    size_t nnz_ = 0;
    std::vector<int> row_, col_, outer_;
    std::vector<std::vector<double>> zval_;          // values of each Z on the pattern
    std::vector<std::vector<double>> A_;             // dSigma/d(parameter) on the pattern
    std::vector<int> diag_pos_;
    VectorXd diagC_;
    SparseSPD solver_;
    MeanState ms_;
    double mc_error_ = 0.0; int probes_used_ = 0;

    void probe(int i, VectorXd& z) const {
        const std::uint64_t base = mix64(seed_ + static_cast<std::uint64_t>(0x632BE59BD9B4E019ULL) * static_cast<std::uint64_t>(i + 1));
        for (Eigen::Index j = 0; j < n_; ++j) z[j] = (mix64(base ^ static_cast<std::uint64_t>(j)) & 1ULL) ? 1.0 : -1.0;
    }
    void spmv(const std::vector<double>& val, const VectorXd& x, VectorXd& out) const {
        out.setZero(n_);
        for (size_t c = 0; c < uz(n_); ++c) {
            const double xc = x[tix(c)];
            for (size_t p = uz(outer_[c]); p < uz(outer_[c + 1]); ++p) out[row_[p]] += val[p] * xc;
        }
    }

    void build_pattern() {
        std::vector<Eigen::Triplet<double>> trip;
        for (Eigen::Index i = 0; i < n_; ++i) trip.emplace_back(toi(i), toi(i), 1.0);
        for (const ZMat& z : spec_->Z) {
            if (z.is_diag) continue;
            if (z.is_sparse) {
                for (int c = 0; c < z.sparse.outerSize(); ++c)
                    for (SpMat::InnerIterator it(z.sparse, c); it; ++it) trip.emplace_back(it.row(), c, 1.0);
            } else {
                for (Eigen::Index c = 0; c < n_; ++c) for (Eigen::Index r = 0; r < n_; ++r)
                    if (z.dense(r, c) != 0.0) trip.emplace_back(toi(r), toi(c), 1.0);
            }
        }
        Sigma_.resize(toi(n_), toi(n_));
        Sigma_.setFromTriplets(trip.begin(), trip.end());
        Sigma_.makeCompressed();
        nnz_ = uz(Sigma_.nonZeros());
        outer_.assign(Sigma_.outerIndexPtr(), Sigma_.outerIndexPtr() + n_ + 1);
        row_.assign(Sigma_.innerIndexPtr(), Sigma_.innerIndexPtr() + nnz_);
        col_.resize(nnz_);
        for (size_t c = 0; c < uz(n_); ++c) for (size_t p = uz(outer_[c]); p < uz(outer_[c + 1]); ++p) col_[p] = toi(c);
        diag_pos_.assign(uz(n_), -1);
        for (size_t p = 0; p < nnz_; ++p) if (row_[p] == col_[p]) diag_pos_[uz(row_[p])] = toi(p);
        auto position = [&](int r, int c) {
            const int* b = row_.data() + outer_[uz(c)]; const int* e = row_.data() + outer_[uz(c) + 1];
            return uz(std::lower_bound(b, e, r) - row_.data());
        };
        zval_.assign(spec_->Z.size(), std::vector<double>(nnz_, 0.0));
        for (size_t j = 0; j < spec_->Z.size(); ++j) {
            const ZMat& z = spec_->Z[j]; auto& v = zval_[j];
            if (z.is_diag) { for (Eigen::Index i = 0; i < n_; ++i) v[uz(diag_pos_[uz(i)])] = z.diag[i]; }
            else if (z.is_sparse) {
                for (int c = 0; c < z.sparse.outerSize(); ++c)
                    for (SpMat::InnerIterator it(z.sparse, c); it; ++it) v[position(toi(it.row()), c)] += it.value();
            } else {
                for (Eigen::Index c = 0; c < n_; ++c) for (Eigen::Index r = 0; r < n_; ++r)
                    if (z.dense(r, c) != 0.0) v[position(toi(r), toi(c))] = z.dense(r, c);
            }
        }
        solver_.analyze(Sigma_);
    }

    /// Sigma, its derivatives and the factorisation at the given mean and parameters.
    void assemble(const Params& par, const VectorXd& mu) {
        const VectorXd& tau = par.tau[0];
        if (uz(tau.size()) != zval_.size())
            throw InvalidInput("expected " + std::to_string(zval_.size()) + " tau parameter(s), got " + std::to_string(tau.size()));
        const bool use_v = spec_->variance != Variance::Constant;
        const bool deriv_power = !spec_->power_fixed && use_v;
        VarOut vo; VectorXd v = VectorXd::Ones(n_);
        if (use_v) { vo = variance_function(spec_->variance, mu, par.power[0], spec_->ntrial, false, deriv_power); v = vo.v; }
        std::vector<double> om(nnz_, 0.0);
        for (size_t j = 0; j < zval_.size(); ++j) { const double t = tau[tix(j)]; for (size_t p = 0; p < nnz_; ++p) om[p] += t * zval_[j][p]; }
        double* sv = Sigma_.valuePtr();
        for (size_t p = 0; p < nnz_; ++p) sv[p] = v[row_[p]] * v[col_[p]] * om[p];
        if (spec_->variance == Variance::PoissonTweedie) for (Eigen::Index i = 0; i < n_; ++i) sv[uz(diag_pos_[uz(i)])] += mu[i];
        if (spec_->variance == Variance::GeomTweedie) for (Eigen::Index i = 0; i < n_; ++i) sv[uz(diag_pos_[uz(i)])] += mu[i] * mu[i];
        diagC_.resize(n_);
        for (Eigen::Index i = 0; i < n_; ++i) diagC_[i] = sv[uz(diag_pos_[uz(i)])];
        const int nd = deriv_power ? (spec_->variance == Variance::BinomialPQ ? 2 : 1) : 0;
        A_.assign(uz(nd) + zval_.size(), std::vector<double>(nnz_, 0.0));
        for (int k = 0; k < nd; ++k) {
            const VectorXd& dv = (k == 0) ? vo.dp : vo.dq;
            for (size_t p = 0; p < nnz_; ++p) A_[uz(k)][p] = (dv[row_[p]] * v[col_[p]] + v[row_[p]] * dv[col_[p]]) * om[p];
        }
        for (size_t j = 0; j < zval_.size(); ++j)
            for (size_t p = 0; p < nnz_; ++p) A_[uz(nd) + j][p] = v[row_[p]] * v[col_[p]] * zval_[j][p];
        solver_.factorize(Sigma_);
    }
};

}} // namespace mcglm::detail
