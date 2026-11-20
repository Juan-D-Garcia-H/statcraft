// SPDX-License-Identifier: Apache-2.0
//
// Linear mixed models with crossed or partially crossed random effects
// (`lme4::lmer`) using sparse linear algebra.
//
//     y = X beta + sum_t Z_t b_t + e,
//     b_t = (b_t1, ..., b_t,n_t),   b_tl ~ N(0, sigma^2 D_t),
//     e ~ N(0, sigma^2 W^-1)
//
// Each term `t` has a grouping factor (`n_t` levels), a per-row design `Z_t`
// (`N x q_t`) and a relative covariance `D_t = L_t L_t^T`, parameterized by any
// `PdSpec` (log-Cholesky, diagonal, block-diagonal, user-defined, ...). Terms
// are independent of each other and may share a factor (for example
// `(1|g) + (0+x|g)`). Nested models are a special case (the factor is the
// interaction), so this engine also handles them without the per-group block
// restriction.
//
// Estimation follows Bates et al. (2015), "Fitting linear mixed-effects models
// using lme4". With `Z~ = Z Lambda` (sparse, `N x q`) and
//
//     A = Z~^T Z~ + I   (SPD, sparse; Cholesky with AMD ordering, symbolic
//                        analysis performed once),
//
// `beta` and `sigma^2` are profiled out analytically and only `theta` (the
// parameters of `L_t`) is optimized:
//
//     u(beta) = A^-1 Z~^T (y - X beta)
//     S       = X^T X - (Z~^T X)^T A^-1 (Z~^T X)
//     beta    = S^-1 (X^T y - (Z~^T X)^T A^-1 Z~^T y)
//     pwrss   = ||y - X beta - Z~ u||^2 + ||u||^2
//
//     ML  : dev = N (1 + log(2 pi pwrss / N)) + log|A| - sum log w
//     REML: dev = (N-p)(1 + log(2 pi pwrss / (N-p))) + log|A| + log|S|
//                 - sum log w,   sigma^2 = pwrss / (N-p)
//
// Linearly dependent columns of `X` are dropped (their coefficients become
// NaN), as in lme4. The gradient with respect to `theta` uses finite
// differences (like lme4's derivative-free optimizer); each evaluation is one
// sparse factorization.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "lme_fit.hpp"

namespace mixedfit {

/// One random-effect term in a crossed model.
struct ReTerm {
    std::string name;
    std::vector<int> group;   ///< N entries: level (0..n_levels-1) of the factor on each row.
    int n_levels = 0;
    Eigen::MatrixXd Z;        ///< N x q design of this term, one row per observation.
    PdSpec pd = PdSpec(PdKind::LogCholesky);
};

/// Specification of a crossed random-effects model.
struct CrossedSpec {
    bool reml = true;
    Optimizer optimizer = Optimizer::Internal;
    double theta_bound = 15.0;
    BfgsOptions opt = [] {
        BfgsOptions o;
        o.gtol = 1e-6;
        o.fd_step = 1e-5;
        o.ftol = 1e-13;
        o.max_iter = 300;
        return o;
    }();
};

/// Profiled quantities at a given `theta`.
///
/// Used both for the deviance and, from Python, for Satterthwaite degrees of
/// freedom.
struct CrossedProfile {
    double dev = std::numeric_limits<double>::infinity();
    double logdetA = 0.0, logdetS = 0.0, pwrss = 0.0, sumlogw = 0.0;
    int N = 0, p = 0;                  ///< `p` is the rank of `X`.
    Eigen::VectorXd beta;              ///< Rank-length coefficients of the non-aliased columns.
    Eigen::MatrixXd Sinv;              ///< (rank x rank) `S^-1`; `vcov(beta) = sigma^2 S^-1`.
    Eigen::VectorXd u;                 ///< Spherical conditional modes (`q`).
    bool ok = false;
};

/// Result of a crossed random-effects fit.
struct CrossedResult {
    Eigen::VectorXd beta, se;                        ///< Length `p`; NaN on aliased columns.
    Eigen::MatrixXd vcov_beta;                       ///< `p x p` coefficient covariance.
    double sigma2 = 0.0;
    Eigen::VectorXd theta;
    std::vector<Eigen::MatrixXd> D_terms, Sigma_terms;  ///< Relative `D` and `Sigma = sigma^2 D` per term.
    std::vector<Eigen::MatrixXd> blup;                  ///< Per term: `n_levels x q` (`b = Lambda u`).
    double loglik = 0.0, deviance = 0.0;
    bool reml = true, singular = false;
    int rank = 0, N = 0, n_re = 0;
    std::vector<int> aliased;                        ///< Columns of `X` that were dropped.
    Eigen::VectorXd fitted_pop, fitted_cond, resid;  ///< On the original (unweighted) scale.
    BfgsResult opt;
    std::string warning;
};

/// Crossed / partially crossed linear mixed model.
class CrossedModel {
public:
    /// Construct the model.
    ///
    /// @param y       Response vector of length `N`.
    /// @param X       Fixed-effects design with `N` rows.
    /// @param terms   Random-effect terms.
    /// @param spec    Model specification.
    /// @param weights Optional prior weights (strictly positive).
    /// @param offset  Optional offset added to the linear predictor.
    /// @throws std::invalid_argument on inconsistent dimensions, non-finite
    ///         inputs, or invalid weights.
    CrossedModel(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, std::vector<ReTerm> terms,
                 CrossedSpec spec = {}, const Eigen::VectorXd& weights = Eigen::VectorXd(),
                 const Eigen::VectorXd& offset = Eigen::VectorXd())
        : y0_(y), X0_(X), terms_(std::move(terms)), spec_(std::move(spec)) {
        N_ = static_cast<int>(y.size());

        if (X.rows() != N_) throw std::invalid_argument("CrossedModel: X must have N rows");
        if (terms_.empty()) throw std::invalid_argument("CrossedModel: at least one random-effect term is required");
        if (weights.size() != 0 && weights.size() != N_)
            throw std::invalid_argument("CrossedModel: 'weights' has the wrong length");
        if (offset.size() != 0 && offset.size() != N_)
            throw std::invalid_argument("CrossedModel: 'offset' has the wrong length");

        offset_ = offset.size() ? offset : Eigen::VectorXd::Zero(N_);
        sw_ = Eigen::VectorXd::Ones(N_);
        if (weights.size()) {
            for (int i = 0; i < N_; ++i) {
                if (!(weights[i] > 0.0))
                    throw std::invalid_argument("CrossedModel: weights must be > 0");
                sw_[i] = std::sqrt(weights[i]);
                sumlogw_ += std::log(weights[i]);
            }
        }
        if (!y.allFinite() || !X.allFinite() || !offset_.allFinite())
            throw std::invalid_argument("CrossedModel: y, X and offset must be finite");

        const Eigen::VectorXd yw = (y - offset_).cwiseProduct(sw_);
        Eigen::MatrixXd Xw = X;
        for (int i = 0; i < N_; ++i) Xw.row(i) *= sw_[i];

        // Keep the independent columns of X from left to right, as lme4 does:
        // the first column of a dependent set is retained and the later ones
        // are dropped. REML depends on the chosen basis of X (for example, on
        // the scale of the column that is kept).
        {
            std::vector<int> keep;
            Eigen::MatrixXd Qb(N_, 0);
            for (int j = 0; j < Xw.cols(); ++j) {
                const double nv = Xw.col(j).norm();
                if (nv == 0.0) continue;
                Eigen::VectorXd r = Xw.col(j);
                for (int pass = 0; pass < 2 && Qb.cols() > 0; ++pass) r -= Qb * (Qb.transpose() * r);
                if (r.norm() > 1e-7 * nv) {
                    keep.push_back(j);
                    Qb.conservativeResize(N_, Qb.cols() + 1);
                    Qb.col(Qb.cols() - 1) = r / r.norm();
                }
            }
            rank_ = static_cast<int>(keep.size());
            keep_ = keep;
        }

        for (int j = 0; j < X.cols(); ++j)
            if (!std::binary_search(keep_.begin(), keep_.end(), j)) aliased_.push_back(j);

        Xk_ = Xw(Eigen::placeholders::all, keep_);
        yw_ = yw;
        XtX_ = Xk_.transpose() * Xk_;
        Xty_ = Xk_.transpose() * yw_;
        yty_ = yw_.squaredNorm();

        // Random-effect terms.
        q_total_ = 0;
        n_theta_ = 0;
        for (auto& t : terms_) {
            if (static_cast<int>(t.group.size()) != N_ || t.Z.rows() != N_)
                throw std::invalid_argument("CrossedModel: a random-effect term has inconsistent dimensions");
            if (t.n_levels < 1 || t.Z.cols() < 1)
                throw std::invalid_argument("CrossedModel: empty random-effect term");
            for (int g : t.group)
                if (g < 0 || g >= t.n_levels)
                    throw std::invalid_argument("CrossedModel: level index out of range");
            if (!t.Z.allFinite())
                throw std::invalid_argument("CrossedModel: Z contains non-finite values");

            const int q = static_cast<int>(t.Z.cols());
            off_.push_back(q_total_);
            q_total_ += t.n_levels * q;
            th_off_.push_back(n_theta_);
            n_theta_ += pd_nparams(t.pd, q);
        }

        // Fixed sparsity pattern of Z~ (N x q_total) and symbolic analysis of A,
        // performed once.
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            for (int i = 0; i < N_; ++i)
                for (int c = 0; c < q; ++c) {
                    rows_.push_back(i);
                    cols_.push_back(off_[t] + terms_[t].group[static_cast<size_t>(i)] * q + c);
                }
        }

        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(rows_.size());
        for (size_t k = 0; k < rows_.size(); ++k) trip.emplace_back(rows_[k], cols_[k], 1.0);

        Eigen::SparseMatrix<double> Z1(N_, q_total_);
        Z1.setFromTriplets(trip.begin(), trip.end());

        Eigen::SparseMatrix<double> A1 = Z1.transpose() * Z1;
        for (int k = 0; k < q_total_; ++k) A1.coeffRef(k, k) += 1.0;
        A1.makeCompressed();
        nnzA_ = A1.nonZeros();
        ldlt_.analyzePattern(A1);
    }

    /// Number of covariance parameters.
    int n_theta() const { return n_theta_; }

    /// Number of observations.
    int N() const { return N_; }

    /// Numerical rank of the fixed-effects design.
    int rank() const { return rank_; }

    /// Total number of random coefficients.
    int n_random() const { return q_total_; }

    /// Indices of the aliased columns of `X`.
    const std::vector<int>& aliased() const { return aliased_; }

    /// Random-effect terms.
    const std::vector<ReTerm>& terms() const { return terms_; }

    /// Default covariance parameters (`D_t = I`).
    Eigen::VectorXd default_theta() const {
        Eigen::VectorXd th(n_theta_);
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            th.segment(th_off_[t], pd_nparams(terms_[t].pd, q)) = pd_start(terms_[t].pd, q);
        }
        return th;
    }

    /// Profiled quantities at `theta`.
    ///
    /// The deviance is `+infinity` if the factorization fails.
    ///
    /// @throws std::invalid_argument if `theta` has the wrong length.
    CrossedProfile profile(const Eigen::VectorXd& theta) const {
        CrossedProfile P;
        P.N = N_;
        P.p = rank_;
        P.sumlogw = sumlogw_;
        if (theta.size() != n_theta_)
            throw std::invalid_argument("CrossedModel::profile: incorrect theta length");

        // Z~ = Z Lambda, where Lambda holds L_t for each level.
        std::vector<Eigen::MatrixXd> L(terms_.size());
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            L[t] = pd_factor(terms_[t].pd, q, theta.segment(th_off_[t], pd_nparams(terms_[t].pd, q)));
        }

        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(rows_.size());
        size_t k = 0;
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            for (int i = 0; i < N_; ++i) {
                const Eigen::RowVectorXd zi = terms_[t].Z.row(i) * sw_[i];
                const Eigen::RowVectorXd zl = zi * L[t];
                for (int c = 0; c < q; ++c, ++k) trip.emplace_back(rows_[k], cols_[k], zl[c]);
            }
        }

        Eigen::SparseMatrix<double> Zs(N_, q_total_);
        Zs.setFromTriplets(trip.begin(), trip.end());
        const Eigen::SparseMatrix<double> Zst = Zs.transpose();

        Eigen::SparseMatrix<double> A = Zst * Zs;
        for (int j = 0; j < q_total_; ++j) A.coeffRef(j, j) += 1.0;
        A.makeCompressed();

        auto& ldlt = ldlt_;   // Reused factorization (not thread-safe).
        if (A.nonZeros() == nnzA_) {
            ldlt.factorize(A);   // Pattern unchanged: reuse the symbolic analysis.
        } else {
            ldlt.compute(A);     // Different pattern (should not happen): reanalyze.
            nnzA_ = A.nonZeros();
        }
        if (ldlt.info() != Eigen::Success) return P;

        const Eigen::VectorXd D = ldlt.vectorD();
        if (!(D.array() > 0.0).all()) return P;
        P.logdetA = D.array().log().sum();

        const int pk = rank_;
        Eigen::MatrixXd ZtX = Zst * Xk_;
        const Eigen::VectorXd Zty = Zst * yw_;
        Eigen::MatrixXd rhs(q_total_, pk + 1);
        rhs.leftCols(pk) = ZtX;
        rhs.col(pk) = Zty;

        const Eigen::MatrixXd sol = ldlt.solve(rhs);
        const Eigen::MatrixXd M = sol.leftCols(pk);
        const Eigen::VectorXd v = sol.col(pk);

        Eigen::MatrixXd S = XtX_ - ZtX.transpose() * M;
        S = 0.5 * (S + S.transpose());
        Eigen::LDLT<Eigen::MatrixXd> sl(S);
        if (pk > 0) {
            if (sl.info() != Eigen::Success || !(sl.vectorD().array() > 0.0).all()) return P;
            P.logdetS = sl.vectorD().array().log().sum();
            P.beta = sl.solve(Eigen::VectorXd(Xty_ - ZtX.transpose() * v));
            P.Sinv = sl.solve(Eigen::MatrixXd::Identity(pk, pk));
        } else {
            P.beta = Eigen::VectorXd(0);
            P.Sinv = Eigen::MatrixXd(0, 0);
        }

        P.u = v - M * P.beta;
        const Eigen::VectorXd r = yw_ - Xk_ * P.beta - Zs * P.u;
        P.pwrss = r.squaredNorm() + P.u.squaredNorm();

        const double dof = spec_.reml ? static_cast<double>(N_ - pk) : static_cast<double>(N_);
        if (!(P.pwrss > 0.0) || dof <= 0.0) return P;

        P.dev = dof * (1.0 + std::log(2.0 * M_PI * P.pwrss / dof)) + P.logdetA - sumlogw_ +
                (spec_.reml ? P.logdetS : 0.0);
        P.ok = std::isfinite(P.dev);
        return P;
    }

    /// Profiled deviance at `theta`.
    double deviance(const Eigen::VectorXd& theta) const { return profile(theta).dev; }

    /// Fit the model.
    ///
    /// @param start Optional initial `theta`.
    /// @throws std::invalid_argument if `start` has the wrong length.
    /// @throws std::runtime_error if the likelihood is not finite at the optimum.
    CrossedResult fit(const Eigen::VectorXd* start = nullptr) const {
        Eigen::VectorXd th0 = start ? *start : default_theta();
        if (th0.size() != n_theta_)
            throw std::invalid_argument("CrossedModel::fit: incorrect 'start' length");

        Objective f = [&](const Eigen::VectorXd& t) {
            if (t.cwiseAbs().maxCoeff() > spec_.theta_bound)
                return std::numeric_limits<double>::infinity();
            double d;
            try {
                d = deviance(t);
            } catch (const std::exception&) {
                d = std::numeric_limits<double>::infinity();
            }
            return d;
        };

        CrossedResult res;
        res.reml = spec_.reml;
        res.opt = run_optimizer(spec_.optimizer, f, nullptr, th0, spec_.opt, spec_.theta_bound,
                                static_cast<double>(N_));

        // If an external optimizer stopped early, finish with the internal BFGS.
        if (spec_.optimizer != Optimizer::Internal && !res.opt.converged) {
            BfgsResult r2 = minimize_bfgs(f, res.opt.x, nullptr, spec_.opt);
            if (std::isfinite(r2.f) && r2.f <= res.opt.f) res.opt = r2;
        }

        const Eigen::VectorXd th = res.opt.x;
        const CrossedProfile P = profile(th);
        if (!P.ok)
            throw std::runtime_error("CrossedModel: the likelihood is not finite at the optimum");

        res.theta = th;
        res.N = N_;
        res.rank = rank_;
        res.n_re = q_total_;
        res.aliased = aliased_;
        res.deviance = P.dev;
        res.loglik = -0.5 * P.dev;

        const double dof = spec_.reml ? static_cast<double>(N_ - rank_) : static_cast<double>(N_);
        res.sigma2 = P.pwrss / dof;

        const int p = static_cast<int>(X0_.cols());
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        res.beta = Eigen::VectorXd::Constant(p, nan);
        res.se = res.beta;
        res.vcov_beta = Eigen::MatrixXd::Constant(p, p, nan);

        for (int a = 0; a < rank_; ++a) {
            res.beta[keep_[static_cast<size_t>(a)]] = P.beta[a];
            for (int b = 0; b < rank_; ++b)
                res.vcov_beta(keep_[static_cast<size_t>(a)], keep_[static_cast<size_t>(b)]) = res.sigma2 * P.Sinv(a, b);
            res.se[keep_[static_cast<size_t>(a)]] = std::sqrt(std::max(res.vcov_beta(keep_[static_cast<size_t>(a)], keep_[static_cast<size_t>(a)]), 0.0));
        }

        // Terms: D, Sigma, BLUP (b = Lambda u), and fitted values.
        Eigen::VectorXd fit_pop = offset_;
        for (int a = 0; a < rank_; ++a) fit_pop += X0_.col(keep_[static_cast<size_t>(a)]) * P.beta[a];
        Eigen::VectorXd fit_cond = fit_pop;

        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            const Eigen::MatrixXd Lt =
                pd_factor(terms_[t].pd, q, th.segment(th_off_[t], pd_nparams(terms_[t].pd, q)));
            res.D_terms.push_back(Lt * Lt.transpose());
            res.Sigma_terms.push_back(res.sigma2 * res.D_terms.back());

            Eigen::MatrixXd b(terms_[t].n_levels, q);
            for (int l = 0; l < terms_[t].n_levels; ++l)
                b.row(l) = (Lt * P.u.segment(off_[t] + l * q, q)).transpose();

            for (int i = 0; i < N_; ++i)
                fit_cond[i] += terms_[t].Z.row(i).dot(b.row(terms_[t].group[static_cast<size_t>(i)]));

            res.blup.push_back(b);

            Eigen::LLT<Eigen::MatrixXd> chol(res.D_terms.back());
            double mind = std::numeric_limits<double>::infinity();
            for (int c = 0; c < q; ++c) mind = std::min(mind, std::abs(Lt(c, c)));

            // Same idea as lme4's isSingular (relative diagonal of L <= 1e-4).
            // A threshold of 1e-3 is used because the log-Cholesky
            // parameterization approaches the boundary without reaching it.
            if (chol.info() != Eigen::Success || mind < 1e-3) res.singular = true;
        }

        res.fitted_pop = fit_pop;
        res.fitted_cond = fit_cond;
        res.resid = y0_ - fit_cond;

        if (!res.opt.converged)
            res.warning = "the optimizer did not converge: " + res.opt.message;
        if (res.singular)
            res.warning += std::string(res.warning.empty() ? "" : "; ") +
                           "singular fit (a random-effect covariance is nearly degenerate)";
        return res;
    }

private:
    Eigen::VectorXd y0_;
    Eigen::MatrixXd X0_;
    std::vector<ReTerm> terms_;
    CrossedSpec spec_;
    int N_ = 0, rank_ = 0, q_total_ = 0, n_theta_ = 0;
    mutable long nnzA_ = 0;
    double sumlogw_ = 0.0;
    Eigen::VectorXd offset_, sw_, yw_, Xty_;
    Eigen::MatrixXd Xk_, XtX_;
    double yty_ = 0.0;
    std::vector<int> keep_, aliased_, off_, th_off_, rows_, cols_;
    mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
};

}  // namespace mixedfit
