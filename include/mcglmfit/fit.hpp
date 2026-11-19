// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file fit.hpp
/// @brief Fitting algorithm (chaser, rc, gradient) and joint sandwich covariance of the estimators.
///
/// Each iteration alternates (i) a Newton-type step for beta from the quasi-score and (ii) an update of the
/// covariance parameters from the Pearson estimating function evaluated at the new beta:
///
///     beta^{(i+1)} = beta^{(i)} - S_beta^{-1} psi_beta
///     chaser:    theta^{(i+1)} = theta^{(i)} - tuning * S_theta^{-1} psi_theta
///     rc:        theta^{(i+1)} = theta^{(i)} - (tuning psi psi^T V^{-1} S + S)^{-1} psi
///     gradient:  theta^{(i+1)} = theta^{(i)} - tuning * psi
///
/// The joint covariance of (beta, theta) is the sandwich J^{-1} V J^{-T} with J = blockdiag(S_beta, S_theta)
/// and V the joint variability (cross term of rank one).
#pragma once
#include "estimating.hpp"
#include "partition.hpp"
#include "sparse_backend.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mcglm {

namespace detail {
/// Number of variance exponents that are estimated for a response.
inline int n_power_free(const ResponseSpec& r) { return r.power_fixed ? 0 : variance_n_power(r.variance); }

/// Stacks the regression coefficients of all responses.
inline VectorXd pack_beta(const Params& p) {
    Eigen::Index m = 0; for (auto& b : p.beta) m += b.size();
    VectorXd v(m); Eigen::Index o = 0;
    for (auto& b : p.beta) { v.segment(o, b.size()) = b; o += b.size(); }
    return v;
}
/// Stacks the covariance parameters: rho, then for each response the free exponents and tau.
inline VectorXd pack_cov(const Params& p, const std::vector<ResponseSpec>& s) {
    std::vector<double> c;
    if (s.size() > 1) for (int i = 0; i < p.rho.size(); ++i) c.push_back(p.rho[i]);
    for (size_t k = 0; k < s.size(); ++k) {
        for (int i = 0; i < n_power_free(s[k]); ++i) c.push_back(p.power[k][i]);
        for (int i = 0; i < p.tau[k].size(); ++i) c.push_back(p.tau[k][i]);
    }
    return Eigen::Map<VectorXd>(c.data(), tix(c.size()));
}
/// Inverse of pack_beta.
inline void unpack_beta(Params& p, const VectorXd& v) {
    Eigen::Index o = 0;
    for (auto& b : p.beta) { b = v.segment(o, b.size()); o += b.size(); }
}
/// Inverse of pack_cov.
inline void unpack_cov(Params& p, const std::vector<ResponseSpec>& s, const VectorXd& v) {
    Eigen::Index o = 0;
    if (s.size() > 1) { p.rho = v.segment(o, p.rho.size()); o += p.rho.size(); }
    for (size_t k = 0; k < s.size(); ++k) {
        for (int i = 0; i < n_power_free(s[k]); ++i) p.power[k][i] = v[o++];
        for (int i = 0; i < p.tau[k].size(); ++i) p.tau[k][i] = v[o++];
    }
}
/// Solves A x = b by partial-pivot LU.
/// @throws NumericalError (mentioning `what`) if the solution is not finite
inline VectorXd solve_lu(const MatrixXd& A, const VectorXd& b, const char* what) {
    Eigen::PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    if (!x.allFinite()) throw NumericalError(std::string("singular linear system in the ") + what);
    return x;
}
/// Inverse by partial-pivot LU.
/// @throws NumericalError (mentioning `what`) if the inverse is not finite
inline MatrixXd inverse_lu(const MatrixXd& A, const char* what) {
    MatrixXd I = A.partialPivLu().inverse();
    if (!I.allFinite()) throw NumericalError(std::string("singular matrix in the ") + what);
    return I;
}
} // namespace detail

namespace detail {

/// Evaluates the estimating functions on one (sub)problem; the blocks of a decomposed problem are SubModels too.
/// All outputs are sums over the units of the sub-problem, so the totals of a decomposition are plain sums.
class SubModel {
public:
    const std::vector<ResponseSpec>* specs = nullptr;
    VectorXd y, w;                              ///< responses and weights of this sub-problem (stacked)
    std::vector<Eigen::Index> gidx;             ///< global position of each stacked entry (empty: identity)
    CrossBlock cb = CrossBlock::RCompatible;

    /// Quasi-score at `par`; keeps the covariance features when `keep` (they are reused by `cov_pass`).
    BetaPass beta_pass(const Params& par, bool need_var, bool keep) {
        ms_ = mean_state(*specs, par.beta);
        BuildRequest rq; rq.deriv = keep;
        CovFeatures cf = build_C(*specs, ms_.mu_k, par, rq, cb);
        BetaScore bs = quasi_score(cf, ms_, y, w, need_var);
        if (keep) cf_ = std::move(cf); else cf_ = CovFeatures();
        return BetaPass{std::move(bs.score), std::move(bs.sens), std::move(bs.var)};
    }
    /// Pearson estimating function at `par`. `reuse` takes the covariance features kept by `beta_pass`
    /// (valid when they do not depend on beta: constant variance, or the same parameters).
    PearsonSum cov_pass(const Params& par, const MatrixXd* inv_J, bool correct, bool need_sens, bool need_var,
                        bool need_cross, bool diag_variability, bool reuse) {
        MeanState ms2 = mean_state(*specs, par.beta);
        CovFeatures own; const CovFeatures* cf = &cf_;
        if (!reuse) { BuildRequest rq; rq.deriv = true; own = build_C(*specs, ms2.mu_k, par, rq, cb); cf = &own; }
        PearsonOut o = pearson(*cf, y, ms2, inv_J, correct, need_sens, need_var, need_cross, w, diag_variability);
        return PearsonSum{std::move(o.score), std::move(o.sens), std::move(o.var), std::move(o.u)};
    }
    /// Stacked mean of this sub-problem.
    VectorXd mu(const Params& par) const { return mean_state(*specs, par.beta).mu; }

private:
    MeanState ms_;
    CovFeatures cf_;
};

/// Runs `f(b)` for every block and adds the results. Blocks are split into contiguous chunks, one per thread, and
/// the partial sums are added in thread order, so the result depends only on the number of threads.
template <class R, class F>
R sum_over_blocks(size_t nb, int threads, F&& f) {
    int nt = 1;
#ifdef _OPENMP
    nt = threads > 0 ? threads : omp_get_max_threads();
#else
    (void)threads;
#endif
    nt = std::max(1, std::min<int>(nt, toi(nb)));
    const size_t nts = uz(nt);
    std::vector<R> part(nts); std::vector<char> has(nts, 0);
    std::exception_ptr err;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
    {
        int tid_i = 0;
#ifdef _OPENMP
        tid_i = omp_get_thread_num();
#endif
        const size_t tid = uz(tid_i);
        const size_t lo = nb * tid / nts, hi = nb * (tid + 1) / nts;
        try {
            for (size_t b = lo; b < hi; ++b) {
                R r = f(b);
                if (!has[tid]) { part[tid] = std::move(r); has[tid] = 1; } else accumulate(part[tid], r);
            }
        } catch (...) {
#ifdef _OPENMP
#pragma omp critical(mcglm_block_error)
#endif
            { if (!err) err = std::current_exception(); }
        }
    }
    if (err) std::rethrow_exception(err);
    R total; bool any = false;
    for (size_t t = 0; t < nts; ++t) {
        if (!has[t]) continue;
        if (!any) { total = std::move(part[t]); any = true; } else accumulate(total, part[t]);
    }
    return total;
}

/// Whole problem as a single block (diagonal per-observation path or dense).
class DenseBackend {
public:
    DenseBackend(const std::vector<ResponseSpec>& specs, const VectorXd& y, const VectorXd& w, CrossBlock cb) {
        m_.specs = &specs; m_.y = y; m_.w = w; m_.cb = cb;
    }
    BetaPass beta_pass(const Params& p, bool need_var, bool keep) { return m_.beta_pass(p, need_var, keep); }
    PearsonSum cov_pass(const Params& p, const MatrixXd* J, bool c, bool s, bool v, bool x, bool d, bool reuse) {
        return m_.cov_pass(p, J, c, s, v, x, d, reuse);
    }
    VectorXd mu(const Params& p) const { return m_.mu(p); }
private:
    SubModel m_;
};

/// Problem decomposed into independent blocks (see partition.hpp).
class BlockBackend {
public:
    BlockBackend(std::shared_ptr<const Partition> part, const VectorXd& y, const VectorXd& w, CrossBlock cb, int threads)
        : part_(std::move(part)), threads_(threads) {
        const int K = part_->K; const Eigen::Index n = part_->n;
        subs_.resize(part_->batches.size());
        for (size_t b = 0; b < subs_.size(); ++b) {
            const Batch& B = part_->batches[b]; SubModel& s = subs_[b];
            const Eigen::Index m = tix(B.units.size());
            s.specs = &B.specs; s.cb = cb;
            s.y.resize(K * m); s.w.resize(K * m); s.gidx.resize(uz(K * m));
            for (int k = 0; k < K; ++k)
                for (Eigen::Index c = 0; c < m; ++c) {
                    const Eigen::Index g = k * n + B.units[uz(c)];
                    s.y[k * m + c] = y[g]; s.w[k * m + c] = w[g]; s.gidx[uz(k * m + c)] = g;
                }
        }
    }
    BetaPass beta_pass(const Params& p, bool need_var, bool keep) {
        return sum_over_blocks<BetaPass>(subs_.size(), threads_, [&](size_t b) { return subs_[b].beta_pass(p, need_var, keep); });
    }
    PearsonSum cov_pass(const Params& p, const MatrixXd* J, bool c, bool s, bool v, bool x, bool d, bool reuse) {
        return sum_over_blocks<PearsonSum>(subs_.size(), threads_, [&](size_t b) { return subs_[b].cov_pass(p, J, c, s, v, x, d, reuse); });
    }
    VectorXd mu(const Params& p) const {
        VectorXd out(part_->K * part_->n);
        for (const SubModel& s : subs_) { VectorXd m = s.mu(p); for (Eigen::Index i = 0; i < m.size(); ++i) out[s.gidx[uz(i)]] = m[i]; }
        return out;
    }
private:
    std::shared_ptr<const Partition> part_;
    std::vector<SubModel> subs_;
    int threads_;
};

/// The fitting loop, independent of how the covariance is organised.
template <class Backend>
FitResult run_fit(Backend& be, const std::vector<ResponseSpec>& specs, const VectorXd& y, const Params& init, const Options& opt) {
    const int K = toi(specs.size());
    Params par = init;
    FitResult R;
    R.n_rho = K > 1 ? toi(par.rho.size()) : 0;
    for (int k = 0; k < K; ++k) {
        R.n_beta.push_back(toi(par.beta[uz(k)].size())); R.n_tau.push_back(toi(par.tau[uz(k)].size()));
        R.n_power.push_back(n_power_free(specs[uz(k)]));
    }
    VectorXd beta_ini = pack_beta(par), cov_ini = pack_cov(par, specs);
    const Eigen::Index Pb = beta_ini.size(), Pc = cov_ini.size();
    R.iter_beta.setConstant(opt.max_iter, Pb, std::nan("")); R.iter_cov.setConstant(opt.max_iter, Pc, std::nan(""));
    R.score_beta.setConstant(opt.max_iter, Pb, std::nan("")); R.score_cov.setConstant(opt.max_iter, Pc, std::nan(""));
    R.iter_beta.row(0) = beta_ini; R.iter_cov.row(0) = cov_ini;

    bool all_const = true;
    for (auto& s : specs) all_const = all_const && s.variance == Variance::Constant;

    int rows = 1; R.converged = false;
    for (int it = 1; it < opt.max_iter; ++it) {
        // ---- step 1: quasi-score at the current mean structure ----
        // With constant variance Sigma does not depend on mu, so the features are kept and reused in step 2.
        BetaPass bs = be.beta_pass(par, false, all_const);
        VectorXd beta_new = beta_ini - solve_lu(bs.sens, bs.score, "sensitivity of beta");
        R.score_beta.row(it) = bs.score;
        unpack_beta(par, beta_new);
        // ---- step 2: covariance parameters at beta^{(i+1)} ----
        MatrixXd inv_J = inverse_lu(bs.sens, "sensitivity of beta");
        VectorXd step;
        if (Pc > 0) {
            if (opt.method == Method::Chaser) {
                PearsonSum pe = be.cov_pass(par, &inv_J, opt.correct, true, false, false, opt.r_diag_variability, all_const);
                R.score_cov.row(it) = pe.score;
                step = opt.tuning * solve_lu(pe.sens, pe.score, "Pearson sensitivity");
            } else if (opt.method == Method::Gradient) {
                PearsonSum pe = be.cov_pass(par, &inv_J, opt.correct, false, false, false, opt.r_diag_variability, all_const);
                R.score_cov.row(it) = pe.score; step = opt.tuning * pe.score;
            } else {
                PearsonSum pe = be.cov_pass(par, &inv_J, opt.correct, true, true, false, opt.r_diag_variability, all_const);
                R.score_cov.row(it) = pe.score;
                MatrixXd Vi = inverse_lu(pe.var, "Pearson variability");
                MatrixXd A = opt.tuning * pe.score * pe.score.transpose() * Vi * pe.sens + pe.sens;
                step = solve_lu(A, pe.score, "rc update");
            }
        } else step = VectorXd(0);
        VectorXd cov_next = cov_ini - step;
        unpack_cov(par, specs, cov_next);
        if (opt.verbose) std::cerr << "iter " << it + 1 << " cov = " << cov_next.transpose() << "\n";
        VectorXd tol_v(Pb + Pc);
        tol_v << (beta_new - beta_ini).cwiseAbs(), (cov_next - cov_ini).cwiseAbs();
        beta_ini = beta_new; cov_ini = cov_next;
        R.iter_beta.row(it) = beta_ini; R.iter_cov.row(it) = cov_ini;
        rows = it + 1;
        if (!beta_ini.allFinite() || !cov_ini.allFinite()) throw NumericalError("the parameter estimates are not finite");
        const bool stop = opt.on_iteration ? opt.on_iteration(it + 1, beta_ini, cov_ini) : false;   // also called on the final iteration
        if ((tol_v.array() <= opt.tol).all()) { R.converged = true; break; }
        if (stop) { R.stopped = true; break; }
    }
    R.iterations = rows;
    R.iter_beta.conservativeResize(rows, Pb); R.iter_cov.conservativeResize(rows, Pc);
    R.score_beta.conservativeResize(rows, Pb); R.score_cov.conservativeResize(rows, Pc);

    // ---- final quantities ----
    BetaPass bsf = be.beta_pass(par, opt.compute_vcov, true);
    MatrixXd inv_Jf = inverse_lu(bsf.sens, "final sensitivity of beta");
    R.beta = beta_ini; R.cov = cov_ini; R.params = par;
    R.fitted = be.mu(par); R.residuals = y - R.fitted; R.inv_S_beta = inv_Jf;
    if (Pc > 0) {
        PearsonSum pf = be.cov_pass(par, &inv_Jf, false, true, opt.compute_vcov, opt.compute_vcov, opt.r_diag_variability, true);
        R.S_cov = pf.sens;
        if (opt.compute_vcov) {
            R.V_beta = bsf.var; R.V_cov = pf.var;
            MatrixXd Vcb = bsf.score * pf.u.transpose();                  // rank-one cross variability: psi_beta u^T
            MatrixXd Vj(Pb + Pc, Pb + Pc);
            Vj << bsf.var, Vcb, Vcb.transpose(), pf.var;
            MatrixXd Jinv = MatrixXd::Zero(Pb + Pc, Pb + Pc);
            Jinv.topLeftCorner(Pb, Pb) = inv_Jf;
            Jinv.bottomRightCorner(Pc, Pc) = inverse_lu(pf.sens, "final Pearson sensitivity");
            R.vcov = Jinv * Vj * Jinv.transpose();
        }
    } else if (opt.compute_vcov) {
        R.V_beta = bsf.var; R.vcov = inv_Jf * bsf.var * inv_Jf.transpose();
    }
    return R;
}

} // namespace detail

/// Decides whether the decomposition into blocks is used and returns it (null: dense / diagonal path).
/// Building it costs O(nnz) (O(n^2) to scan a dense Z), so callers that fit repeatedly should cache it in
/// `Options::partition`.
inline std::shared_ptr<const Partition> choose_partition(const std::vector<ResponseSpec>& specs, const Options& opt) {
    if (opt.structure == Structure::Dense || opt.structure == Structure::Sparse) return nullptr;
    std::shared_ptr<const Partition> part = opt.partition;
    if (!part) {
        if (opt.structure == Structure::Auto && !has_off_diagonal_z(specs)) return nullptr;
        part = build_partition(specs, opt.block_target);
    }
    if (part->K != toi(specs.size()) || part->n != specs[0].X.rows())
        throw InvalidInput("the precomputed partition does not match the model");
    if (opt.structure == Structure::Auto && !part->useful) return nullptr;
    if (part->batches.empty()) part = build_partition(specs, opt.block_target, true);      // forced decomposition
    return part;
}

/// Fits the model.
///
/// The covariance is handled in one of four ways, chosen automatically (see `Options::structure`):
///  - all Z diagonal: per-observation fast path, O(n K^3) per iteration;
///  - Z with several independent groups of units (dense or sparse): exact block decomposition, O(sum m_b^3);
///  - one connected block, K = 1, identity covariance link, sparse Z and n > `sparse_dense_max_n`: the sparse method
///    (exact estimates, stochastic sensitivity/variability of the covariance parameters; see sparse_backend.hpp);
///  - otherwise: dense N x N computation with LAPACK/BLAS, O(N^3).
/// `FitResult::structure` reports which one was used.
///
/// @param specs_in  one specification per response (all with the same number of observations n)
/// @param y         stacked responses (length K n). NaN entries are treated as missing: weight 0, value 0
/// @param init      starting values (see `initial_values`)
/// @param weights   observation weights (length K n); empty means all ones
/// @param opt       fitting options
/// @return estimates, history, fitted values, sensitivities and (optionally) the joint covariance
/// @throws InvalidInput   inconsistent dimensions
/// @throws NumericalError a numerical step failed (singular system, non-positive-definite matrix, non-finite values)
/// @throws ResourceError  the dense computation would not fit in memory
inline FitResult fit(const std::vector<ResponseSpec>& specs_in, VectorXd y, const Params& init,
                     VectorXd weights = VectorXd(), const Options& opt = Options()) {
    using namespace detail;
    const int K = toi(specs_in.size());
    if (K == 0) throw InvalidInput("at least one response is required");
    const Eigen::Index n = specs_in[0].X.rows();
    bool any_sparse = false;
    for (auto& s : specs_in) {
        if (s.X.rows() != n) throw InvalidInput("all responses must have the same number of observations");
        if (s.Z.empty()) throw InvalidInput("every response needs at least one Z matrix");
        for (auto& z : s.Z) { if (z.n() != n) throw InvalidInput("a Z matrix has the wrong size"); any_sparse = any_sparse || z.is_sparse; }
    }
    if (y.size() != K * n) throw InvalidInput("y must have length K * n (responses stacked)");
    VectorXd w;
    if (weights.size() == 0) {
        w = VectorXd::Ones(y.size());
        for (Eigen::Index i = 0; i < y.size(); ++i) if (std::isnan(y[i])) { w[i] = 0; y[i] = 0; }
    } else {
        if (weights.size() != y.size()) throw InvalidInput("weights must have length K * n");
        w = weights; for (Eigen::Index i = 0; i < y.size(); ++i) if (std::isnan(y[i])) y[i] = 0;
    }

    const bool sparse_ok = sparse_method_applicable(specs_in);
    bool use_sparse = false;
    if (opt.structure == Structure::Sparse) {
        if (!sparse_ok) throw InvalidInput("the sparse method requires one response (K = 1) with the identity covariance link");
        use_sparse = true;
    }
    std::shared_ptr<const Partition> part = use_sparse ? nullptr : choose_partition(specs_in, opt);
    if (!part && !use_sparse && opt.structure == Structure::Auto && sparse_ok && has_off_diagonal_z(specs_in) &&
        n > opt.sparse_dense_max_n && z_density(specs_in) <= 0.25)
        use_sparse = true;
    FitResult R;
    if (use_sparse) {
        SparseBackend be(specs_in, y, w, opt);
        R = run_fit(be, specs_in, y, init, opt);
        R.structure = "sparse"; R.mc_rel_error = be.mc_error(); R.n_probes = be.probes_used();
        return R;
    }
    if (part) {
        BlockBackend be(part, y, w, opt.cross_block, opt.block_threads);
        R = run_fit(be, specs_in, y, init, opt);
        R.structure = "blocks"; R.n_blocks = toi(part->n_batches); R.max_block = toi(part->max_batch_units());
        return R;
    }
    // Single block: expand sparse Z and fill in ntrial only when needed, to avoid copying large matrices.
    std::vector<ResponseSpec> specs_work;
    const std::vector<ResponseSpec>* sp = &specs_in;
    bool need_ntrial = false;
    for (auto& s : specs_in)
        need_ntrial = need_ntrial || (s.ntrial.size() == 0 && (s.variance == Variance::BinomialP || s.variance == Variance::BinomialPQ));
    const bool dense_work = has_off_diagonal_z(specs_in);
    if (dense_work) {
        int q = 0; for (auto& s : specs_in) q += toi(s.Z.size()) + 2;
        la::require_dense_memory(K * n, (K > 1 ? 12 : 8) + q);
    }
    if (any_sparse || need_ntrial) {
        specs_work = specs_in; sp = &specs_work;
        for (auto& s : specs_work) {
            for (auto& z : s.Z) if (z.is_sparse) z = z.densified();
            if (s.ntrial.size() == 0 && (s.variance == Variance::BinomialP || s.variance == Variance::BinomialPQ)) s.ntrial = VectorXd::Ones(n);
        }
    }
    DenseBackend be(*sp, y, w, opt.cross_block);
    R = run_fit(be, *sp, y, init, opt);
    R.structure = dense_work ? "dense" : "diagonal";
    return R;
}

} // namespace mcglm
