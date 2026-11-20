// SPDX-License-Identifier: Apache-2.0
//
// Generalized linear mixed models (`lme4::glmer`) with sparse algebra: the
// Laplace approximation (`nAGQ = 1`) and adaptive Gauss-Hermite quadrature
// (`nAGQ > 1`, only with a single scalar random effect, as in `glmer`).
//
//   y_i | b ~ family(mu_i, wt_i),   g(mu) = eta = X beta + offset + Z b,
//   b = Lambda(theta) u,  u ~ N(0, I)
//
// Families: binomial (logit, probit, cloglog, cauchit; `y` as a proportion
// with weights = number of trials), Poisson, negative binomial (fixed or
// estimated theta, as in `glmer.nb`), Gaussian, Gamma and inverse Gaussian.
// The continuous families estimate (or fix) a dispersion parameter on the log
// scale. The random terms are the same `ReTerm`/`PdSpec` objects used by the
// crossed linear engine.
//
// Estimation (Bates et al. 2015, sec. 4): two stages, as in lme4.
//   Stage 1 (`nAGQ = 0`): optimize theta only; for each theta, run a joint
//                         PIRLS in (beta, u): Fisher steps on the penalized
//                         deviance pdev = -2 loglik(mu) + ||u||^2 with
//                         A = Z~' W Z~ + I (W = wt mu.eta^2 / V(mu),
//                         Z~ = Z Lambda).
//   Stage 2 (Laplace): optimize (theta, beta [, log theta_nb]); for each point
//                         run PIRLS over u only (conditional mode u*):
//                         dev = pdev(u*) + log|A(u*)|. (The log-likelihood
//                         includes all constants, as lme4's `logLik`.)
//   AGQ: dev = -2 sum_j log[ sqrt(2/h_j) sum_k w_k exp(x_k^2 + l_j(u*_j + sqrt(2/h_j) x_k)) ] + J log(2 pi),
//        l_j = sum_i loglik_i - u^2/2, h_j = A_jj (Fisher weights). Note: for
//        nAGQ > 1, lme4 reports a deviance-based logLik (without the saturated
//        constant); here `loglik` is the actual likelihood and `ll_sat` is the
//        constant, so loglik_lme4 = loglik - ll_sat.
// Standard errors follow lme4: (X'WX - X'WZ~ A^-1 Z~'WX)^-1 at the solution.
// The optimizer gradient is numerical (central differences); each evaluation is
// a PIRLS with a sparse Cholesky factorization (symbolic analysis performed once).
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>

#include "crossed_fit.hpp"

namespace mixedfit {

/// Response distribution of a generalized linear mixed model.
enum class GlmFamily { Binomial, Poisson, NegBin, Gaussian, Gamma, InvGauss };

/// Link function connecting the linear predictor to the mean.
enum class GlmLink { Logit, Probit, Cloglog, Cauchit, Log, Sqrt, Identity, Inverse, Invmu2 };

/// Specification of a `glmer`-style model.
struct GlmmSpec {
    GlmFamily family = GlmFamily::Binomial;     ///< Response family.
    GlmLink link = GlmLink::Logit;              ///< Link function.
    double theta_nb = 1.0;                      ///< Negative binomial: size (fixed, or starting value if `estimate_theta`).
    bool estimate_theta = false;                ///< Estimate the negative binomial size parameter.
    double dispersion = 1.0;                    ///< Continuous families: dispersion (sigma^2 for Gaussian, 1/shape for Gamma/InvGaussian).
    bool estimate_dispersion = false;           ///< Estimate the dispersion on the log scale (continuous families).
    int nAGQ = 1;                               ///< Quadrature points: 1 = Laplace, >1 = adaptive Gauss-Hermite.
    bool init_stage = true;                     ///< Run stage 1 (theta with joint PIRLS) before full optimization.
    Optimizer optimizer = Optimizer::Internal;  ///< Optimization back end.
    double theta_bound = 15.0;                  ///< Box bound on the variance parameters.
    int max_pirls = 100;                        ///< Maximum PIRLS iterations.
    double pirls_tol = 1e-12;                   ///< PIRLS convergence tolerance on the deviance.
    double pirls_step_tol = 1e-10;              ///< PIRLS convergence tolerance on the max step (infinity norm).
    BfgsOptions opt = [] {
        BfgsOptions o;
        o.gtol = 1e-5;
        o.fd_step = 1e-4;
        o.ftol = 1e-14;
        o.max_iter = 300;
        return o;
    }();
};

namespace glm_detail {

/// A small positive constant used to keep links and variances finite.
constexpr double kEps = 2.220446049250313e-16;

/// Inverse link: map the linear predictor `eta` to the mean `mu`.
inline double linkinv(GlmLink l, double eta) {
    switch (l) {
        case GlmLink::Logit: { const double mu = 1.0 / (1.0 + std::exp(-std::max(-30.0, std::min(30.0, eta)))); return std::min(std::max(mu, kEps), 1.0 - kEps); }
        case GlmLink::Probit: { const double mu = 0.5 * std::erfc(-std::max(-8.2, std::min(8.2, eta)) / std::sqrt(2.0)); return std::min(std::max(mu, kEps), 1.0 - kEps); }
        case GlmLink::Cloglog: { const double mu = -std::expm1(-std::exp(std::min(eta, 30.0))); return std::min(std::max(mu, kEps), 1.0 - kEps); }
        case GlmLink::Cauchit: { const double mu = 0.5 + std::atan(eta) / M_PI; return std::min(std::max(mu, kEps), 1.0 - kEps); }
        case GlmLink::Log: return std::max(std::exp(std::min(eta, 700.0)), kEps);
        case GlmLink::Sqrt: return eta * eta;
        case GlmLink::Identity: return eta;
        case GlmLink::Inverse: return 1.0 / std::max(eta, 1e-6);
        case GlmLink::Invmu2: return 1.0 / std::sqrt(std::max(eta, 1e-6));
    }
    return eta;
}

/// Link function: map the mean `mu` to the linear predictor `eta`.
inline double linkfun(GlmLink l, double mu) {
    switch (l) {
        case GlmLink::Logit: return std::log(mu / (1.0 - mu));
        case GlmLink::Probit: return std::sqrt(2.0) * -0.5 * 0.0 + [&] { // normal quantile: Newton on erfc (only for starting values)
            double x = 0.0;
            for (int i = 0; i < 60; ++i) { const double f = 0.5 * std::erfc(-x / std::sqrt(2.0)) - mu, d = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI); x -= f / std::max(d, 1e-300); x = std::max(-8.2, std::min(8.2, x)); }
            return x; }();
        case GlmLink::Cloglog: return std::log(-std::log1p(-mu));
        case GlmLink::Cauchit: return std::tan(M_PI * (mu - 0.5));
        case GlmLink::Log: return std::log(mu);
        case GlmLink::Sqrt: return std::sqrt(mu);
        case GlmLink::Identity: return mu;
        case GlmLink::Inverse: return 1.0 / std::max(mu, kEps);
        case GlmLink::Invmu2: return 1.0 / std::max(mu * mu, kEps);
    }
    return mu;
}

/// Derivative `d mu / d eta` of the inverse link.
inline double mu_eta(GlmLink l, double eta) {
    switch (l) {
        case GlmLink::Logit: { const double e = std::exp(-std::abs(std::max(-30.0, std::min(30.0, eta)))); return std::max(e / ((1.0 + e) * (1.0 + e)), kEps); }
        case GlmLink::Probit: return std::max(std::exp(-0.5 * eta * eta) / std::sqrt(2.0 * M_PI), kEps);
        case GlmLink::Cloglog: { const double e = std::min(eta, 30.0); return std::max(std::exp(e) * std::exp(-std::exp(e)), kEps); }
        case GlmLink::Cauchit: return std::max(1.0 / (M_PI * (1.0 + eta * eta)), kEps);
        case GlmLink::Log: return std::max(std::exp(std::min(eta, 700.0)), kEps);
        case GlmLink::Sqrt: return 2.0 * eta;
        case GlmLink::Identity: return 1.0;
        case GlmLink::Inverse: { const double e = std::max(eta, 1e-6); return -1.0 / (e * e); }
        case GlmLink::Invmu2: { const double e = std::max(eta, 1e-6); return -0.5 * std::pow(e, -1.5); }
    }
    return 1.0;
}

/// Variance function `V(mu)` of the response family (the dispersion is kept
/// out of `V` and enters the log-likelihood, as in GLM IRLS).
inline double variance(const GlmmSpec& s, double mu, double tnb) {
    switch (s.family) {
        case GlmFamily::Binomial: return mu * (1.0 - mu);
        case GlmFamily::Poisson: return mu;
        case GlmFamily::NegBin: return mu + mu * mu / tnb;
        case GlmFamily::Gaussian: return 1.0;
        case GlmFamily::Gamma: return mu * mu;
        case GlmFamily::InvGauss: return mu * mu * mu;
    }
    return 1.0;
}

/// `x * log(y)` with the convention `0 * log(0) = 0`.
inline double xlogy(double x, double y) { return x == 0.0 ? 0.0 : x * std::log(y); }

/// Full log-density `log f(y | mu)` (including constants), multiplied by the
/// prior weight. `tnb` is the negative-binomial size; `disp` is the dispersion
/// of the continuous families (sigma^2 for Gaussian, 1/shape for Gamma and
/// inverse Gaussian).
inline double loglik1(const GlmmSpec& s, double y, double mu, double wt, double tnb, double disp) {
    switch (s.family) {
        case GlmFamily::Binomial: {
            const double m = std::round(wt), ys = std::round(wt * y);
            return std::lgamma(m + 1.0) - std::lgamma(ys + 1.0) - std::lgamma(m - ys + 1.0) + xlogy(ys, mu) + (m - ys == 0.0 ? 0.0 : (m - ys) * std::log1p(-mu));
        }
        case GlmFamily::Poisson: return wt * (xlogy(y, mu) - mu - std::lgamma(y + 1.0));
        case GlmFamily::NegBin:
            return wt * (std::lgamma(y + tnb) - std::lgamma(tnb) - std::lgamma(y + 1.0) + tnb * (std::log(tnb) - std::log(tnb + mu)) + xlogy(y, mu) - y * std::log(tnb + mu));
        case GlmFamily::Gaussian:
            return -0.5 * wt * (std::log(2.0 * M_PI * disp) + (y - mu) * (y - mu) / disp);
        case GlmFamily::Gamma: {
            if (!(y > 0.0) || !(mu > 0.0) || !(disp > 0.0)) return -std::numeric_limits<double>::infinity();
            const double k = 1.0 / disp;
            return wt * (k * std::log(k / mu) - std::lgamma(k) + (k - 1.0) * std::log(y) - k * y / mu);
        }
        case GlmFamily::InvGauss: {
            if (!(y > 0.0) || !(mu > 0.0) || !(disp > 0.0)) return -std::numeric_limits<double>::infinity();
            const double lambda = 1.0 / disp;
            return 0.5 * wt * (std::log(lambda / (2.0 * M_PI * y * y * y)) - lambda * (y - mu) * (y - mu) / (mu * mu * y));
        }
    }
    return 0.0;
}

/// Gauss-Hermite nodes and weights (weight `exp(-x^2)`) via Golub-Welsch.
inline void gauss_hermite(int n, Eigen::VectorXd& x, Eigen::VectorXd& w) {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    for (int k = 0; k + 1 < n; ++k) J(k, k + 1) = J(k + 1, k) = std::sqrt((k + 1) / 2.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
    x = es.eigenvalues();
    w.resize(n);
    for (int k = 0; k < n; ++k) w[k] = std::sqrt(M_PI) * es.eigenvectors()(0, k) * es.eigenvectors()(0, k);
}

}  // namespace glm_detail

/// Results of a generalized linear mixed-model fit.
struct GlmmResult {
    Eigen::VectorXd beta, se;               ///< Fixed effects and standard errors (NaN for aliased columns).
    Eigen::MatrixXd vcov_beta;              ///< p x p beta block of 2 H^-1 (H = numerical Hessian of the
                                            ///< deviance in (theta, beta)), as `vcov` in glmer; falls back to
                                            ///< the RX approximation when H is not positive definite.
    Eigen::MatrixXd vcov_rx;                ///< (X'WX - X'WZ~ A^-1 Z~'WX)^-1: lme4's RX approximation
                                            ///< (`use.hessian = FALSE`).
    bool vcov_hessian = false;              ///< Whether `vcov_beta` came from the Hessian.
    Eigen::VectorXd theta;                  ///< Covariance parameters (concatenated by term).
    double theta_nb = std::numeric_limits<double>::quiet_NaN();  ///< Estimated negative binomial size.
    double dispersion = std::numeric_limits<double>::quiet_NaN();  ///< Estimated/fixed dispersion of continuous families.
    std::vector<Eigen::MatrixXd> D_terms;   ///< Random-effect covariances (unit scale).
    std::vector<Eigen::MatrixXd> blup;      ///< Per term: n_levels x q (b = Lambda u).
    double loglik = 0.0, deviance = 0.0, ll_sat = 0.0;  ///< Log-likelihood, deviance, and saturated log-likelihood.
    int rank = 0, N = 0, n_re = 0, nAGQ = 1;  ///< Design rank, observations, random effects, quadrature order.
    std::vector<int> aliased;               ///< Indices of aliased columns of X.
    Eigen::VectorXd eta, mu, eta_pop, mu_pop, W;  ///< Linear predictor and mean (with and without random effects), working weights.
    bool singular = false;                  ///< A random-effect covariance is nearly degenerate.
    BfgsResult opt;                         ///< Optimizer diagnostics.
    int stage1_evals = 0;                   ///< Function evaluations in stage 1.
    std::string warning;                    ///< Non-fatal diagnostic messages.
};

/// Generalized linear mixed model (`glmer`-style).
class GlmmModel {
public:
    /// Construct the model.
    ///
    /// @param y        Response vector.
    /// @param X        Fixed-effect design matrix (N rows).
    /// @param terms    Random-effect terms.
    /// @param spec     Family, link, and optimizer settings.
    /// @param weights  Optional prior weights (N).
    /// @param offset   Optional offset (N).
    /// @throws std::invalid_argument On inconsistent dimensions, non-finite
    ///         inputs, invalid binomial responses, invalid weights, or an
    ///         unsupported `nAGQ` configuration.
    GlmmModel(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, std::vector<ReTerm> terms, GlmmSpec spec = {},
              const Eigen::VectorXd& weights = Eigen::VectorXd(), const Eigen::VectorXd& offset = Eigen::VectorXd())
        : y_(y), X0_(X), terms_(std::move(terms)), spec_(std::move(spec)) {
        N_ = static_cast<int>(y.size());
        if (X.rows() != N_) throw std::invalid_argument("GlmmModel: X must have N rows");
        if (terms_.empty()) throw std::invalid_argument("GlmmModel: at least one random term is required");
        if (weights.size() != 0 && weights.size() != N_) throw std::invalid_argument("GlmmModel: weights have the wrong length");
        if (offset.size() != 0 && offset.size() != N_) throw std::invalid_argument("GlmmModel: offset has the wrong length");
        wt_ = weights.size() ? weights : Eigen::VectorXd::Ones(N_);
        off_ = offset.size() ? offset : Eigen::VectorXd::Zero(N_);
        if (!y.allFinite() || !X.allFinite() || !off_.allFinite() || !wt_.allFinite()) throw std::invalid_argument("GlmmModel: y, X, weights, and offset must be finite");
        const bool continuous = spec_.family == GlmFamily::Gaussian || spec_.family == GlmFamily::Gamma || spec_.family == GlmFamily::InvGauss;
        for (int i = 0; i < N_; ++i) {
            if (!(wt_[i] > 0.0)) throw std::invalid_argument("GlmmModel: weights must be > 0");
            if (spec_.family == GlmFamily::Binomial && (y[i] < 0.0 || y[i] > 1.0)) throw std::invalid_argument("GlmmModel: for the binomial family y must be in [0, 1] (proportion) with weights = trials");
            if ((spec_.family == GlmFamily::Poisson || spec_.family == GlmFamily::NegBin) && y[i] < 0.0) throw std::invalid_argument("GlmmModel: y must be >= 0 for count families");
            if ((spec_.family == GlmFamily::Gamma || spec_.family == GlmFamily::InvGauss) && !(y[i] > 0.0)) throw std::invalid_argument("GlmmModel: y must be > 0 for the Gamma and inverse Gaussian families");
        }
        if (spec_.nAGQ < 1) throw std::invalid_argument("GlmmModel: nAGQ >= 1");
        if (spec_.estimate_theta && spec_.family != GlmFamily::NegBin) throw std::invalid_argument("GlmmModel: estimate_theta is only available for the negative binomial family");
        if (spec_.estimate_dispersion && !continuous) throw std::invalid_argument("GlmmModel: estimate_dispersion is only available for Gaussian, Gamma and inverse Gaussian families");
        if (!(spec_.dispersion > 0.0)) throw std::invalid_argument("GlmmModel: dispersion must be > 0");
        // Independent columns of X (left to right, as in lme4).
        {
            std::vector<int> keep;
            Eigen::MatrixXd Qb(N_, 0);
            for (int j = 0; j < X.cols(); ++j) {
                const double nv = X.col(j).norm();
                if (nv == 0.0) continue;
                Eigen::VectorXd r = X.col(j);
                for (int pass = 0; pass < 2 && Qb.cols() > 0; ++pass) r -= Qb * (Qb.transpose() * r);
                if (r.norm() > 1e-7 * nv) {
                    keep.push_back(j);
                    Qb.conservativeResize(N_, Qb.cols() + 1);
                    Qb.col(Qb.cols() - 1) = r / r.norm();
                }
            }
            keep_ = keep;
        }
        rank_ = static_cast<int>(keep_.size());
        for (int j = 0; j < X.cols(); ++j)
            if (!std::binary_search(keep_.begin(), keep_.end(), j)) aliased_.push_back(j);
        Xk_ = X(Eigen::placeholders::all, keep_);
        q_total_ = 0;
        n_theta_ = 0;
        for (auto& t : terms_) {
            if (static_cast<int>(t.group.size()) != N_ || t.Z.rows() != N_) throw std::invalid_argument("GlmmModel: term with inconsistent dimensions");
            if (t.n_levels < 1 || t.Z.cols() < 1) throw std::invalid_argument("GlmmModel: empty term");
            for (int g : t.group)
                if (g < 0 || g >= t.n_levels) throw std::invalid_argument("GlmmModel: level index out of range");
            const int q = static_cast<int>(t.Z.cols());
            off_t_.push_back(q_total_);
            q_total_ += t.n_levels * q;
            th_off_.push_back(n_theta_);
            n_theta_ += pd_nparams(t.pd, q);
        }
        if (spec_.nAGQ > 1 && !(terms_.size() == 1 && terms_[0].Z.cols() == 1))
            throw std::invalid_argument("GlmmModel: nAGQ > 1 is only supported with a single scalar random effect (as in glmer)");
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            for (int i = 0; i < N_; ++i)
                for (int c = 0; c < q; ++c) {
                    rows_.push_back(i);
                    cols_.push_back(off_t_[t] + terms_[t].group[static_cast<size_t>(i)] * q + c);
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
        u_warm_ = Eigen::VectorXd::Zero(q_total_);
        if (spec_.nAGQ > 1) glm_detail::gauss_hermite(spec_.nAGQ, ghx_, ghw_);
    }

    /// Number of covariance parameters.
    int n_theta() const { return n_theta_; }
    /// Number of random effects (flattened).
    int n_random() const { return q_total_; }
    /// Rank of the fixed-effect design.
    int rank() const { return rank_; }
    /// Indices of aliased columns of the original design.
    const std::vector<int>& aliased() const { return aliased_; }

    /// Fit the model.
    ///
    /// @return The fit results, including fixed effects, covariance
    ///         parameters, BLUPs, log-likelihood, and diagnostics.
    /// @throws std::runtime_error If the deviance is not finite at the optimum
    ///         or if PIRLS fails to converge there.
    GlmmResult fit() const {
        using namespace glm_detail;
        const int p = rank_;
        Eigen::VectorXd theta(n_theta_);
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            theta.segment(th_off_[t], pd_nparams(terms_[t].pd, q)) = pd_start(terms_[t].pd, q);
        }
        // Initial beta: least squares on eta0 = g(mu0), as in glm().
        Eigen::VectorXd eta0(N_);
        for (int i = 0; i < N_; ++i) {
            double mu0;
            if (spec_.family == GlmFamily::Binomial) mu0 = (wt_[i] * y_[i] + 0.5) / (wt_[i] + 1.0);
            else if (spec_.family == GlmFamily::Gaussian && spec_.link == GlmLink::Identity) mu0 = y_[i];
            else mu0 = std::max(y_[i] + 0.1, 1e-3);
            eta0[i] = linkfun(spec_.link, mu0) - off_[i];
        }
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
        if (p > 0) beta = Eigen::MatrixXd(Xk_.transpose() * Xk_).ldlt().solve(Eigen::VectorXd(Xk_.transpose() * eta0));
        // Inverse links have a restricted eta > 0 domain.  Starting the
        // covariance scales at ~10% of eta0 keeps the conditional modes inside
        // it; the default sd of 1 is far too large on the 1/mu^2 scale.
        if (spec_.link == GlmLink::Inverse || spec_.link == GlmLink::Invmu2) {
            const double log_sd = std::log(std::max(1e-8, 0.1 * std::abs(eta0.mean())));
            for (int i = 0; i < theta.size(); ++i)
                if (theta[i] >= 0.0) theta[i] = std::min(theta[i], log_sd);
        }
        double tnb = spec_.theta_nb, disp = spec_.dispersion;
        if (spec_.estimate_dispersion) {
            // Method-of-moments starting value: far better than phi = 1 for
            // Gamma and inverse Gaussian, where the natural scale can be tiny.
            const double my = std::max(y_.mean(), kEps);
            const double vy = std::max((y_.array() - my).square().sum() / std::max(N_ - 1, 1), kEps);
            if (spec_.family == GlmFamily::Gaussian) disp = vy;
            else if (spec_.family == GlmFamily::Gamma) disp = vy / (my * my);
            else disp = vy / (my * my * my);
        }
        u_warm_.setZero();
        GlmmResult res;
        res.nAGQ = spec_.nAGQ;
        // ---- Stage 1: theta with joint PIRLS ----
        if (spec_.init_stage) {
            Eigen::VectorXd bw = beta, uw = u_warm_;
            Objective f1 = [&](const Eigen::VectorXd& th) {
                if (th.cwiseAbs().maxCoeff() > spec_.theta_bound) return std::numeric_limits<double>::infinity();
                Eigen::VectorXd b = bw, u = uw;
                double d;
                try { d = dev_joint(th, b, u, tnb, disp); } catch (const std::exception&) { d = std::numeric_limits<double>::infinity(); }
                if (std::isfinite(d)) { bw = b; uw = u; }
                return d;
            };
            BfgsResult r1 = run_optimizer(spec_.optimizer, f1, nullptr, theta, spec_.opt, spec_.theta_bound, static_cast<double>(N_));
            res.stage1_evals = r1.f_evals;
            if (std::isfinite(r1.f)) { theta = r1.x; Eigen::VectorXd b = bw, u = uw; (void)dev_joint(theta, b, u, tnb, disp); beta = b; u_warm_ = u; }
        }
        // ---- Stage 2: (theta, beta [, log theta_nb] [, log dispersion]) ----
        const int theta_idx = n_theta_ + p;
        const int disp_idx = theta_idx + (spec_.estimate_theta ? 1 : 0);
        const int nx = disp_idx + (spec_.estimate_dispersion ? 1 : 0);
        Eigen::VectorXd x0(nx);
        x0.head(n_theta_) = theta;
        x0.segment(n_theta_, p) = beta;
        if (spec_.estimate_theta) x0[theta_idx] = std::log(tnb);
        if (spec_.estimate_dispersion) x0[disp_idx] = std::log(disp);
        Objective f2 = [&](const Eigen::VectorXd& x) {
            if (x.head(n_theta_).cwiseAbs().maxCoeff() > spec_.theta_bound) return std::numeric_limits<double>::infinity();
            const double tn = spec_.estimate_theta ? std::exp(x[theta_idx]) : tnb;
            if (spec_.estimate_theta && (x[theta_idx] < -12.0 || x[theta_idx] > 15.0)) return std::numeric_limits<double>::infinity();
            const double dp = spec_.estimate_dispersion ? std::exp(x[disp_idx]) : disp;
            if (spec_.estimate_dispersion && (x[disp_idx] < -12.0 || x[disp_idx] > 15.0)) return std::numeric_limits<double>::infinity();
            double d;
            try { d = dev_stage2(x.head(n_theta_), x.segment(n_theta_, p), tn, dp); } catch (const std::exception&) { d = std::numeric_limits<double>::infinity(); }
            return d;
        };
        res.opt = run_optimizer(spec_.optimizer, f2, nullptr, x0, spec_.opt, spec_.theta_bound, static_cast<double>(N_));
        if (spec_.optimizer != Optimizer::Internal && !res.opt.converged) {
            // Fall back to the internal BFGS if the external backend failed.
            BfgsResult r2 = minimize_bfgs(f2, res.opt.x, nullptr, spec_.opt);
            if (std::isfinite(r2.f) && r2.f <= res.opt.f) res.opt = r2;
        }
        const Eigen::VectorXd xb = res.opt.x;
        theta = xb.head(n_theta_);
        beta = xb.segment(n_theta_, p);
        if (spec_.estimate_theta) tnb = std::exp(xb[theta_idx]);
        if (spec_.estimate_dispersion) disp = std::exp(xb[disp_idx]);
        const double dev = dev_stage2(theta, beta, tnb, disp);
        if (!std::isfinite(dev)) throw std::runtime_error("GlmmModel: deviance is not finite at the optimum");
        res.deviance = dev;
        res.loglik = -0.5 * dev;
        res.theta = theta;
        res.dispersion = disp;
        if (spec_.family == GlmFamily::NegBin) res.theta_nb = tnb;
        res.N = N_;
        res.rank = rank_;
        res.n_re = q_total_;
        res.aliased = aliased_;
        // ---- Results at the solution: conditional mode, standard errors (as lme4), BLUPs ----
        const Eigen::SparseMatrix<double> Zs = make_Zs(theta), Zst = Zs.transpose();
        Eigen::VectorXd u = u_warm_, mu, W;
        double pdev, ldA;
        const Eigen::VectorXd base = Xk_ * beta + off_;
        if (!pirls_u(Zs, Zst, base, tnb, disp, u, pdev, mu, W, ldA)) throw std::runtime_error("GlmmModel: PIRLS did not converge at the optimum");
        Eigen::VectorXd eta = base + Zs * u;
        Eigen::MatrixXd B = Zst * W.asDiagonal() * Xk_;
        Eigen::MatrixXd C = Xk_.transpose() * W.asDiagonal() * Xk_;
        Eigen::MatrixXd S = C - B.transpose() * Eigen::MatrixXd(ldlt_.solve(B));
        S = 0.5 * (S + S.transpose());
        const Eigen::MatrixXd Sinv = p > 0 ? Eigen::MatrixXd(S.ldlt().solve(Eigen::MatrixXd::Identity(p, p))) : Eigen::MatrixXd(0, 0);
        // vcov as glmer: beta block of the inverse Hessian of the deviance
        // (theta and beta; theta_nb held fixed).
        Eigen::MatrixXd Vb = Sinv;
        {
            const int nh = n_theta_ + p;
            Eigen::VectorXd xh(nh);
            xh.head(n_theta_) = theta;
            xh.tail(p) = beta;
            auto fd = [&](const Eigen::VectorXd& x) { return dev_stage2(x.head(n_theta_), x.tail(p), tnb, disp); };
            const double f0 = fd(xh);
            Eigen::MatrixXd H(nh, nh);
            bool fin = std::isfinite(f0);
            std::vector<double> hs(static_cast<size_t>(nh));
            for (int i = 0; i < nh; ++i) hs[static_cast<size_t>(i)] = 2e-4 * std::max(1.0, std::abs(xh[i]));
            // Numerical Hessian by central differences.
            for (int i = 0; i < nh && fin; ++i) {
                Eigen::VectorXd e = xh;
                e[i] += hs[static_cast<size_t>(i)]; const double fp = fd(e);
                e[i] = xh[i] - hs[static_cast<size_t>(i)]; const double fm = fd(e);
                H(i, i) = (fp - 2.0 * f0 + fm) / (hs[static_cast<size_t>(i)] * hs[static_cast<size_t>(i)]);
                for (int j = 0; j < i; ++j) {
                    Eigen::VectorXd w = xh;
                    w[i] = xh[i] + hs[static_cast<size_t>(i)]; w[j] = xh[j] + hs[static_cast<size_t>(j)]; const double fpp = fd(w);
                    w[j] = xh[j] - hs[static_cast<size_t>(j)]; const double fpm = fd(w);
                    w[i] = xh[i] - hs[static_cast<size_t>(i)]; const double fmm = fd(w);
                    w[j] = xh[j] + hs[static_cast<size_t>(j)]; const double fmp = fd(w);
                    H(i, j) = H(j, i) = (fpp - fpm - fmp + fmm) / (4.0 * hs[static_cast<size_t>(i)] * hs[static_cast<size_t>(j)]);
                }
                fin = fin && std::isfinite(fp) && std::isfinite(fm);
            }
            if (fin && H.allFinite()) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
                if (es.eigenvalues().minCoeff() > 1e-8 * std::max(1.0, es.eigenvalues().maxCoeff())) {
                    const Eigen::MatrixXd Hi = 2.0 * H.inverse();
                    Vb = Hi.bottomRightCorner(p, p);
                    res.vcov_hessian = true;
                }
            }
            u_warm_ = u;
        }
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        const int pf = static_cast<int>(X0_.cols());
        res.beta = Eigen::VectorXd::Constant(pf, nan);
        res.se = res.beta;
        res.vcov_beta = Eigen::MatrixXd::Constant(pf, pf, nan);
        res.vcov_rx = res.vcov_beta;
        for (int a = 0; a < p; ++a) {
            res.beta[keep_[static_cast<size_t>(a)]] = beta[a];
            for (int b = 0; b < p; ++b) { res.vcov_beta(keep_[static_cast<size_t>(a)], keep_[static_cast<size_t>(b)]) = Vb(a, b); res.vcov_rx(keep_[static_cast<size_t>(a)], keep_[static_cast<size_t>(b)]) = Sinv(a, b); }
            res.se[keep_[static_cast<size_t>(a)]] = std::sqrt(std::max(Vb(a, a), 0.0));
        }
        for (size_t t = 0; t < terms_.size(); ++t) {
            const int q = static_cast<int>(terms_[t].Z.cols());
            const Eigen::MatrixXd Lt = pd_factor(terms_[t].pd, q, theta.segment(th_off_[t], pd_nparams(terms_[t].pd, q)));
            res.D_terms.push_back(Lt * Lt.transpose());
            Eigen::MatrixXd b(terms_[t].n_levels, q);
            for (int l = 0; l < terms_[t].n_levels; ++l) b.row(l) = (Lt * u.segment(off_t_[t] + l * q, q)).transpose();
            res.blup.push_back(b);
            // A near-zero diagonal entry of the Cholesky factor indicates a
            // singular random-effect covariance.
            double mind = std::numeric_limits<double>::infinity();
            for (int c = 0; c < q; ++c) mind = std::min(mind, std::abs(Lt(c, c)));
            if (mind < 1e-3) res.singular = true;
        }
        res.eta = eta;
        res.mu = mu;
        res.W = W;
        res.eta_pop = Xk_ * beta + off_;
        res.mu_pop.resize(N_);
        for (int i = 0; i < N_; ++i) res.mu_pop[i] = linkinv(spec_.link, res.eta_pop[i]);
        // Saturated log-likelihood.
        double sat = 0.0;
        for (int i = 0; i < N_; ++i) {
            const double yi = spec_.family == GlmFamily::Binomial ? std::min(std::max(y_[i], kEps), 1.0 - kEps) : std::max(y_[i], kEps);
            const double mus = spec_.family == GlmFamily::Binomial ? (y_[i] <= 0.0 ? kEps : (y_[i] >= 1.0 ? 1.0 - kEps : y_[i])) : (y_[i] <= 0.0 ? 0.0 : y_[i]);
            (void)yi;
            if ((spec_.family == GlmFamily::Poisson || spec_.family == GlmFamily::NegBin) && y_[i] == 0.0) {
                sat += loglik1(spec_, 0.0, 0.0, wt_[i], tnb, disp);   // y = mu = 0: Poisson -> 0 (xlogy), NB -> lgamma terms
            } else {
                sat += loglik1(spec_, y_[i], mus, wt_[i], tnb, disp);
            }
        }
        res.ll_sat = sat;
        if (!res.opt.converged) res.warning = "the optimizer did not converge: " + res.opt.message;
        if (res.singular) res.warning += std::string(res.warning.empty() ? "" : "; ") + "singular fit (a random-effect covariance is nearly degenerate)";
        return res;
    }

    /// Laplace deviance at the given parameter values.
    double laplace_deviance(const Eigen::VectorXd& theta, const Eigen::VectorXd& beta, double tnb, double disp = 1.0) const { return dev_stage2(theta, beta, tnb, disp); }

private:
    /// Build the sparse random-effect matrix `Z~ = Z Lambda(theta)`.
    Eigen::SparseMatrix<double> make_Zs(const Eigen::VectorXd& theta) const {
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
                const Eigen::RowVectorXd zl = terms_[t].Z.row(i) * L[t];
                for (int c = 0; c < q; ++c, ++k) trip.emplace_back(rows_[k], cols_[k], zl[c]);
            }
        }
        Eigen::SparseMatrix<double> Zs(N_, q_total_);
        Zs.setFromTriplets(trip.begin(), trip.end());
        return Zs;
    }

    /// Compute the mean, working weights `W`, working residual
    /// `WR = W * (y - mu) / mu.eta`, and total log-likelihood for a linear
    /// predictor.
    double evaluate(const Eigen::VectorXd& eta, double tnb, double disp, Eigen::VectorXd& mu, Eigen::VectorXd& W, Eigen::VectorXd& WR) const {
        using namespace glm_detail;
        const bool continuous = spec_.family == GlmFamily::Gaussian || spec_.family == GlmFamily::Gamma || spec_.family == GlmFamily::InvGauss;
        const double scale = continuous ? 1.0 / disp : 1.0;   // the likelihood of the continuous families is phi^{-1} times the canonical one
        mu.resize(N_); W.resize(N_); WR.resize(N_);
        if (spec_.link == GlmLink::Inverse || spec_.link == GlmLink::Invmu2) {
            // Both inverse links are defined only for eta > 0.  Without a hard
            // barrier the clamping inside linkinv() creates a flat region with
            // an enormous dmu/deta, which derails the optimizer.
            for (int i = 0; i < N_; ++i) {
                if (!(eta[i] > 1e-8)) {
                    mu.setZero(); W.setOnes(); WR.setZero();
                    return -std::numeric_limits<double>::infinity();
                }
            }
        }
        double ll = 0.0;
        for (int i = 0; i < N_; ++i) {
            const double m = linkinv(spec_.link, eta[i]);
            const double me = mu_eta(spec_.link, eta[i]);
            const double v = std::max(variance(spec_, m, tnb), kEps);
            mu[i] = m;
            W[i] = scale * wt_[i] * me * me / v;
            WR[i] = scale * wt_[i] * me / v * (y_[i] - m);
            ll += loglik1(spec_, y_[i], m, wt_[i], tnb, disp);
        }
        return ll;
    }

    /// Factor the sparse matrix `A = Z~' W Z~ + I` and return `log|A|`.
    ///
    /// Reuses the symbolic pattern when the sparsity structure is unchanged.
    ///
    /// @return `false` if the factorization fails or `A` is not positive
    ///         definite.
    bool factor(const Eigen::SparseMatrix<double>& Zs, const Eigen::SparseMatrix<double>& Zst, const Eigen::VectorXd& W, double& logdet) const {
        Eigen::SparseMatrix<double> A = Zst * W.asDiagonal() * Zs;
        for (int j = 0; j < q_total_; ++j) A.coeffRef(j, j) += 1.0;
        A.makeCompressed();
        if (A.nonZeros() == nnzA_) ldlt_.factorize(A);
        else { ldlt_.compute(A); nnzA_ = A.nonZeros(); }
        if (ldlt_.info() != Eigen::Success) return false;
        const Eigen::VectorXd D = ldlt_.vectorD();
        if (!(D.array() > 0.0).all()) return false;
        logdet = D.array().log().sum();
        return true;
    }

    /// PIRLS over `u` with theta and beta fixed.
    ///
    /// Leaves `ldlt_` factorized at the solution and returns the penalized
    /// deviance, `mu`, `W`, and `log|A|`.
    bool pirls_u(const Eigen::SparseMatrix<double>& Zs, const Eigen::SparseMatrix<double>& Zst, const Eigen::VectorXd& base, double tnb, double disp,
                 Eigen::VectorXd& u, double& pdev, Eigen::VectorXd& mu, Eigen::VectorXd& W, double& logdetA) const {
        Eigen::VectorXd eta = base + Zs * u, WR;
        double ll = evaluate(eta, tnb, disp, mu, W, WR);
        pdev = -2.0 * ll + u.squaredNorm();
        if (!std::isfinite(pdev)) { u.setZero(); eta = base; ll = evaluate(eta, tnb, disp, mu, W, WR); pdev = -2.0 * ll; if (!std::isfinite(pdev)) return false; }
        double ld = 0.0;
        for (int it = 0; it < spec_.max_pirls; ++it) {
            if (!factor(Zs, Zst, W, ld)) return false;
            const Eigen::VectorXd delta = ldlt_.solve(Eigen::VectorXd(Zst * WR - u));
            double s = 1.0, pnew = std::numeric_limits<double>::infinity();
            Eigen::VectorXd unew, munew, Wnew, WRnew;
            bool ok = false;
            // Step-halving line search.
            for (int h = 0; h < 25; ++h, s *= 0.5) {
                unew = u + s * delta;
                const double lln = evaluate(base + Zs * unew, tnb, disp, munew, Wnew, WRnew);
                pnew = -2.0 * lln + unew.squaredNorm();
                if (std::isfinite(pnew) && pnew <= pdev + 1e-12 * (std::abs(pdev) + 1.0)) { ok = true; break; }
            }
            if (!ok) break;
            const double diff = pdev - pnew;
            const double stp = (s * delta).cwiseAbs().maxCoeff();
            u = unew; mu = munew; W = Wnew; WR = WRnew; pdev = pnew;
            // Converge on both the deviance change AND the step size: with
            // non-canonical links Fisher scoring converges linearly, and
            // stopping on the deviance alone leaves u with an error that
            // pollutes the finite-difference Hessian.
            if (diff < spec_.pirls_tol * (std::abs(pdev) + 1.0) && diff >= -1e-9 && stp < spec_.pirls_step_tol) break;
        }
        return factor(Zs, Zst, W, logdetA);
    }

    /// Laplace (`nAGQ = 1`) or AGQ deviance for `(theta, beta)`.
    double dev_stage2(const Eigen::VectorXd& theta, const Eigen::VectorXd& beta, double tnb, double disp) const {
        const Eigen::SparseMatrix<double> Zs = make_Zs(theta), Zst = Zs.transpose();
        const Eigen::VectorXd base = Xk_ * beta + off_;
        Eigen::VectorXd u = u_warm_, mu, W;
        double pdev, ldA;
        if (!pirls_u(Zs, Zst, base, tnb, disp, u, pdev, mu, W, ldA)) return std::numeric_limits<double>::infinity();
        u_warm_ = u;
        if (spec_.nAGQ <= 1) return pdev + ldA;
        return dev_agq(theta, base, tnb, disp, Zs, u, W);
    }

    /// AGQ for a single scalar random effect: `Zs` has one entry per row
    /// (column = level).
    double dev_agq(const Eigen::VectorXd& theta, const Eigen::VectorXd& base, double tnb, double disp, const Eigen::SparseMatrix<double>& Zs, const Eigen::VectorXd& ustar,
                   const Eigen::VectorXd& W) const {
        using namespace glm_detail;
        (void)theta;
        const int J = terms_[0].n_levels;
        Eigen::VectorXd zt(N_);                             // Z~ per row
        for (int k = 0; k < Zs.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(Zs, k); it; ++it) zt[it.row()] = it.value();
        const std::vector<int>& g = terms_[0].group;
        Eigen::VectorXd h = Eigen::VectorXd::Ones(J);
        for (int i = 0; i < N_; ++i) h[g[static_cast<size_t>(i)]] += W[i] * zt[i] * zt[i];
        const int K = static_cast<int>(ghx_.size());
        Eigen::MatrixXd lse(J, K);
        for (int k = 0; k < K; ++k) {
            Eigen::VectorXd ll = Eigen::VectorXd::Zero(J);
            Eigen::VectorXd uk(J);
            for (int j = 0; j < J; ++j) uk[j] = ustar[j] + std::sqrt(2.0 / h[j]) * ghx_[k];
            for (int i = 0; i < N_; ++i) {
                const double eta = base[i] + zt[i] * uk[g[static_cast<size_t>(i)]];
                ll[g[static_cast<size_t>(i)]] += loglik1(spec_, y_[i], linkinv(spec_.link, eta), wt_[i], tnb, disp);
            }
            for (int j = 0; j < J; ++j) lse(j, k) = std::log(ghw_[k]) + ghx_[k] * ghx_[k] + ll[j] - 0.5 * uk[j] * uk[j];
        }
        // Log-sum-exp for numerical stability.
        double dev = J * std::log(2.0 * M_PI);
        for (int j = 0; j < J; ++j) {
            const double m = lse.row(j).maxCoeff();
            dev -= 2.0 * (m + std::log((lse.row(j).array() - m).exp().sum()) + 0.5 * std::log(2.0 / h[j]));
        }
        return dev;
    }

    /// Joint PIRLS in `(beta, u)` for a given theta (stage 1).
    ///
    /// @return `pdev + log|A|`.
    double dev_joint(const Eigen::VectorXd& theta, Eigen::VectorXd& beta, Eigen::VectorXd& u, double tnb, double disp) const {
        const int p = rank_;
        const Eigen::SparseMatrix<double> Zs = make_Zs(theta), Zst = Zs.transpose();
        Eigen::VectorXd eta = Xk_ * beta + off_ + Zs * u, mu, W, WR;
        double ll = evaluate(eta, tnb, disp, mu, W, WR);
        double pdev = -2.0 * ll + u.squaredNorm();
        if (!std::isfinite(pdev)) return std::numeric_limits<double>::infinity();
        double ld = 0.0;
        for (int it = 0; it < spec_.max_pirls; ++it) {
            if (!factor(Zs, Zst, W, ld)) return std::numeric_limits<double>::infinity();
            const Eigen::VectorXd ru = Zst * WR - u, rb = Xk_.transpose() * WR;
            Eigen::MatrixXd Bm = Zst * W.asDiagonal() * Xk_;
            const Eigen::MatrixXd C = Xk_.transpose() * W.asDiagonal() * Xk_;
            const Eigen::MatrixXd AiB = ldlt_.solve(Bm);
            const Eigen::VectorXd Airu = ldlt_.solve(ru);
            Eigen::MatrixXd S = C - Bm.transpose() * AiB;
            S = 0.5 * (S + S.transpose());
            Eigen::VectorXd db = p > 0 ? Eigen::VectorXd(S.ldlt().solve(Eigen::VectorXd(rb - Bm.transpose() * Airu))) : Eigen::VectorXd(0);
            const Eigen::VectorXd du = Airu - AiB * db;
            double s = 1.0, pnew = std::numeric_limits<double>::infinity();
            Eigen::VectorXd bn, un, munew, Wnew, WRnew;
            bool ok = false;
            // Step-halving line search.
            for (int h = 0; h < 25; ++h, s *= 0.5) {
                bn = beta + s * db; un = u + s * du;
                const double lln = evaluate(Xk_ * bn + off_ + Zs * un, tnb, disp, munew, Wnew, WRnew);
                pnew = -2.0 * lln + un.squaredNorm();
                if (std::isfinite(pnew) && pnew <= pdev + 1e-12 * (std::abs(pdev) + 1.0)) { ok = true; break; }
            }
            if (!ok) break;
            const double diff = pdev - pnew;
            const double stp = std::max((s * db).cwiseAbs().maxCoeff(), (s * du).cwiseAbs().maxCoeff());
            beta = bn; u = un; mu = munew; W = Wnew; WR = WRnew; pdev = pnew;
            if (diff < spec_.pirls_tol * (std::abs(pdev) + 1.0) && diff >= -1e-9 && stp < spec_.pirls_step_tol) break;
        }
        if (!factor(Zs, Zst, W, ld)) return std::numeric_limits<double>::infinity();
        return pdev + ld;
    }

    Eigen::VectorXd y_, wt_, off_;                          ///< Response, prior weights, offset.
    Eigen::MatrixXd X0_, Xk_;                               ///< Original and full-rank designs.
    std::vector<ReTerm> terms_;                             ///< Random-effect terms.
    GlmmSpec spec_;                                         ///< Model specification.
    int N_ = 0, rank_ = 0, q_total_ = 0, n_theta_ = 0;      ///< Data size, rank, random effects, covariance parameters.
    mutable long nnzA_ = 0;                                 ///< Stored nonzeros of the factorized matrix.
    std::vector<int> keep_, aliased_, off_t_, th_off_, rows_, cols_;  ///< Column and block bookkeeping.
    mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;  ///< Sparse Cholesky of A.
    mutable Eigen::VectorXd u_warm_;                        ///< Warm start for the conditional mode.
    Eigen::VectorXd ghx_, ghw_;                             ///< Gauss-Hermite nodes and weights.
};

}  // namespace mixedfit
