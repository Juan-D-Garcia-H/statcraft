// SPDX-License-Identifier: Apache-2.0
//
// `gls()` and `gnls()` equivalents from R/nlme: generalized least squares
// (linear or nonlinear) with within-group correlation and variance structures
// but WITHOUT random effects:
//
//     y_i = X_i beta + e_i                       (gls)
//     y_i = f(phi_i) + e_i,  phi_i = A_i beta    (gnls)
//     e_i ~ N(0, sigma^2 S_i R_i S_i)
//
// Here `S_i` is the variance function, `R_i` is the correlation matrix (any of
// the `mixedfit` structures, including user-defined ones), and the groups are
// independent.
//
// Both models reuse `LmeModel` / `NlmeModel` with an empty random-effects level
// (`q = 0`). They therefore share the likelihood, analytic gradient,
// optimizers, fixed parameters, fixed sigma, `apVar`, and identifiability
// diagnostics.
//
//   * `GlsModel`:  ML or REML (R default: `method = "REML"`). Returns
//                  `LmeResult` (empty `D`; marginal residuals and fitted
//                  values).
//   * `GnlsModel`: ML (R default: `method = "ML"`; `gnls()` only fits by ML).
//                  With `reml = true`, REML is applied to the linearized model
//                  at the solution (an extension not offered by R). Returns
//                  `NlmeResult`.
//
// The estimator is exact maximum likelihood: at the fixed point of the
// alternating algorithm (PNLS plus an LME step), `beta` solves the nonlinear
// generalized least-squares problem and `theta` annihilates the gradient of the
// profiled likelihood. No algorithm beyond `NlmeModel` is therefore required.
#pragma once

#include "lme_fit.hpp"
#include "nlme_fit.hpp"

namespace mixedfit {

/// Specification of a `gls` model.
struct GlsSpec {
    CorSpec cor;                          ///< Within-group correlation structure.
    VarSpec var;                          ///< Within-group variance function.
    bool reml = true;                     ///< Use REML instead of ML.
    bool reml_constant = false;           ///< Add 1/2 log|X'X| to the REML log-likelihood.
    std::vector<FixedTheta> fixed;        ///< Unconstrained parameters held fixed.
    double sigma_fixed = 0.0;             ///< If > 0, hold sigma fixed at this value.
    Optimizer optimizer = Optimizer::Internal;  ///< Optimization back end.
    double theta_bound = 15.0;            ///< Box bound for L-BFGS-B.
    bool analytic_grad = true;            ///< Use the analytic profiled gradient.
    bool diagnose = true;                 ///< Run the identifiability diagnostic.
    double var_max_ratio = 1e4;           ///< Maximum allowed max(s)/min(s) ratio.
    double var_shape_bound = 10.0;        ///< Bound on the shape parameter of `varConstPower`.
    BfgsOptions opt = [] {
        BfgsOptions o;
        o.gtol = 1e-6;
        o.fd_step = 1e-5;
        o.ftol = 1e-13;
        return o;
    }();
};

/// Group stacked data for `gls`.
///
/// If `labels` is empty, a single group containing all rows is created. The
/// correlation and variance structures act within groups; observations in
/// different groups are independent.
inline std::vector<LmeGroup> make_gls_groups(const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
                                             const std::vector<int>& labels = {},
                                             const Eigen::VectorXd& time = Eigen::VectorXd(),
                                             const std::vector<int>& strata = {},
                                             const Eigen::VectorXd& vcov = Eigen::VectorXd(),
                                             const Eigen::MatrixXd& coords = Eigen::MatrixXd()) {
    const int N = static_cast<int>(y.size());
    const std::vector<int> lab = labels.empty() ? std::vector<int>(static_cast<std::size_t>(N), 0) : labels;
    return make_groups(y, X, Eigen::MatrixXd::Zero(N, 0), lab, time, strata, vcov, coords);
}

/// Generalized least-squares model (linear mean, no random effects).
class GlsModel {
public:
    /// Construct from already grouped data.
    GlsModel(std::vector<LmeGroup> groups, GlsSpec spec)
        : spec_(std::move(spec)), model_(prepare(std::move(groups)), to_lme(spec_)) {}

    /// Fit the model.
    LmeResult fit(const Eigen::VectorXd* start = nullptr) const { return model_.fit(start); }

    /// Access the underlying linear mixed model (random effects removed).
    const LmeModel& model() const { return model_; }

    /// Number of covariance parameters in `theta`.
    int n_theta() const { return model_.n_theta(); }

private:
    /// Remove any random-effect design from the supplied groups.
    static std::vector<LmeGroup> prepare(std::vector<LmeGroup> gs) {
        for (auto& g : gs) {
            g.Z = Eigen::MatrixXd::Zero(g.y.size(), 0);
            g.inner.clear();
        }
        return gs;
    }

    /// Translate a `GlsSpec` into the corresponding `LmeSpec`.
    static LmeSpec to_lme(const GlsSpec& s) {
        LmeSpec l;
        l.cor = s.cor;
        l.var = s.var;
        l.reml = s.reml;
        l.reml_constant = s.reml_constant;
        l.fixed = s.fixed;
        l.sigma_fixed = s.sigma_fixed;
        l.optimizer = s.optimizer;
        l.theta_bound = s.theta_bound;
        l.analytic_grad = s.analytic_grad;
        l.diagnose = s.diagnose;
        l.var_max_ratio = s.var_max_ratio;
        l.var_shape_bound = s.var_shape_bound;
        l.opt = s.opt;
        l.em_iter = 0;   // no random effects, hence no EM step
        return l;
    }

    GlsSpec spec_;
    LmeModel model_;
};

/// Specification of a `gnls` model.
struct GnlsSpec {
    CorSpec cor;                    ///< Within-group correlation structure.
    VarSpec var;                    ///< Within-group variance function.
    bool var_use_fitted = false;    ///< Use fitted values as the variance covariate (`varPower(form = ~fitted(.))`).
    bool reml = false;              ///< R only offers ML in `gnls()`.
    double sigma_fixed = 0.0;       ///< If > 0, hold sigma fixed at this value.
    bool analytic_grad = true;      ///< Use the analytic profiled gradient.
    Optimizer optimizer = Optimizer::Internal;  ///< Optimization back end.
    double theta_bound = 15.0;      ///< Box bound for L-BFGS-B.
    int max_outer = 50;             ///< Maximum number of outer iterations.
    int max_pnls = 50;              ///< Maximum number of PNLS iterations.
    double tol = 1e-6;              ///< Outer convergence tolerance.
    double pnls_tol = 0.0;          ///< PNLS relative-offset stopping tolerance (0 = strict).
    bool parallel = false;          ///< Parallelize group loops with OpenMP.
    BfgsOptions opt = [] {
        BfgsOptions o;
        o.gtol = 1e-6;
        o.fd_step = 1e-5;
        o.ftol = 1e-13;
        return o;
    }();
};

/// Groups for `gnls`, where `phi = A beta` and there are no random effects.
///
/// `A` is the `k x p` fixed-effects design shared by all groups.
inline std::vector<NlmeGroup> make_gnls_groups(const Eigen::VectorXd& y, const std::vector<int>& labels,
                                               const Eigen::MatrixXd& A,
                                               const Eigen::VectorXd& time = Eigen::VectorXd(),
                                               const std::vector<int>& strata = {},
                                               const Eigen::VectorXd& vcov = Eigen::VectorXd(),
                                               const Eigen::MatrixXd& coords = Eigen::MatrixXd()) {
    const std::vector<int> lab = labels.empty() ? std::vector<int>(static_cast<std::size_t>(y.size()), 0) : labels;
    return make_nlme_groups(y, lab, A, Eigen::MatrixXd::Zero(A.rows(), 0), time, strata, vcov, coords);
}

/// Nonlinear generalized least-squares model (no random effects).
class GnlsModel {
public:
    /// Construct from grouped data, a model function and an optional analytic
    /// Jacobian.
    GnlsModel(std::vector<NlmeGroup> groups, NlmeFn f, GnlsSpec spec, NlmeJacFn jac = nullptr)
        : model_(prepare(std::move(groups)), std::move(f), to_nlme(spec), std::move(jac)) {}

    /// Fit the model from initial fixed effects `beta0` and optional initial
    /// covariance parameters.
    NlmeResult fit(const Eigen::VectorXd& beta0, const Eigen::VectorXd* theta0 = nullptr) const {
        return model_.fit(beta0, theta0);
    }

    /// Access the underlying nonlinear mixed model (random effects removed).
    const NlmeModel& model() const { return model_; }

private:
    /// Remove any random-effect design from the supplied groups.
    static std::vector<NlmeGroup> prepare(std::vector<NlmeGroup> gs) {
        for (auto& g : gs) {
            g.B = Eigen::MatrixXd::Zero(g.A.rows(), 0);
            g.B_row = Eigen::MatrixXd();
            g.inner.clear();
        }
        return gs;
    }

    /// Translate a `GnlsSpec` into the corresponding `NlmeSpec`.
    static NlmeSpec to_nlme(const GnlsSpec& s) {
        NlmeSpec n;
        n.cor = s.cor;
        n.var = s.var;
        n.var_use_fitted = s.var_use_fitted;
        n.reml = s.reml;
        n.sigma_fixed = s.sigma_fixed;
        n.analytic_grad = s.analytic_grad;
        n.optimizer = s.optimizer;
        n.theta_bound = s.theta_bound;
        n.max_outer = s.max_outer;
        n.max_pnls = s.max_pnls;
        n.tol = s.tol;
        n.pnls_tol = s.pnls_tol;
        n.parallel = s.parallel;
        n.opt = s.opt;
        return n;
    }

    NlmeModel model_;
};

}  // namespace mixedfit
