// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file quasi_solver.hpp
 * @brief Quasi-likelihood solver class and fit dispatchers for GLM models.
 * @details Implements the QuasiSolver class supporting pre-defined quasi families (Quasi-Poisson, Quasi-Binomial,
 *          Quasi-Gaussian, Quasi-Gamma, Quasi-Inverse-Gaussian) and user-defined custom variance functions via L-BFGS-B, BFGS, or Nelder-Mead.
 */

#ifndef _GLMFIT_QUASI_SOLVER_HPP
#define _GLMFIT_QUASI_SOLVER_HPP

#include "quasi_loss.hpp"
#include "families.hpp"
#include "input_validation.hpp"
#include "minimizer_choice.hpp"
#include "mle_vcov.hpp"
#include "../minimizer/lbfgsb.hpp"
#include "../minimizer/bfgs.hpp"
#include "../minimizer/nelder_mead.hpp"

#include <Eigen/Dense>
#include <optional>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <string>
#include <utility>
#include <functional>

namespace glm {

/**
 * @brief Output result structure for QuasiSolver optimization.
 */
struct QuasiResult {
    bool success = false;                 ///< Optimization convergence flag
    std::string message;                  ///< Solver status message
    double quasi_deviance = 0.0;          ///< Final optimized quasi-deviance loss
    double dispersion = 1.0;              ///< Estimated Pearson dispersion scale factor ($\phi$)
    int iterations = 0;                   ///< Number of solver iterations
    int function_evaluations = 0;         ///< Total function evaluations performed

    Eigen::VectorXd beta;                 ///< Fitted coefficient vector ($\beta$)

    std::optional<Eigen::MatrixXd> vcov;  ///< Asymptotic covariance matrix $\hat\phi (X^TWX)^{-1}$, if computed
    std::optional<Eigen::VectorXd> se;    ///< Standard errors (sqrt of vcov diagonal), if computed
};

/**
 * @brief Main Quasi-likelihood solver class.
 */
class QuasiSolver {
private:
    bool fit_intercept_;                  ///< Flag indicating whether to automatically prepend intercept column
    Minimizer minimizer_ = Minimizer::LBFGSB;  ///< Selected optimization minimizer backend

    Eigen::MatrixXd add_intercept(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd X_new(X.rows(), X.cols() + 1);
        X_new.col(0) = Eigen::VectorXd::Ones(X.rows());
        X_new.rightCols(X.cols()) = X;
        return X_new;
    }

    /**
     * @brief Computes warm-start linear initial parameters $\beta_0$ via least-squares on transformed response $g(y)$.
     * @param link Selected QuasiLinkType.
     * @param X Design matrix ($n \times k$).
     * @param y Response vector ($n \times 1$).
     * @return Initial starting parameter vector $\beta_0$.
     */
    static Eigen::VectorXd compute_warm_start(QuasiLinkType link, const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                              const Eigen::VectorXd& offset = Eigen::VectorXd()) {
        const long k = X.cols();
        Eigen::VectorXd eta0(y.size());

        switch (link) {
            case QuasiLinkType::Identity:
                eta0 = y;
                break;
            case QuasiLinkType::Log:
                eta0 = y.array().max(0.1).log();
                break;
            case QuasiLinkType::Inverse:
                eta0 = y.array().max(0.1).inverse();
                break;
            case QuasiLinkType::Logit: {
                Eigen::ArrayXd p = y.array().max(0.02).min(0.98);
                eta0 = (p / (1.0 - p)).log();
                break;
            }
            case QuasiLinkType::Probit: {
                Eigen::ArrayXd p = y.array().max(0.02).min(0.98);
                eta0 = (p / (1.0 - p)).log() * 0.5875;
                break;
            }
        }

        // eta = X beta + offset  =>  regress g(y) - offset on X
        if (offset.size() == eta0.size()) eta0 -= offset;

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr = X.colPivHouseholderQr();
        if (qr.rank() == k) {
            return qr.solve(eta0);
        }
        return Eigen::VectorXd::Zero(k);
    }

    static void resolve_quasi(FamilyCode code, QuasiLinkType& link, QuasiVarianceType& var) {
        switch (code) {
            case FAM_QUASI_POISSON_LOG:
                link = QuasiLinkType::Log;      var = QuasiVarianceType::Mu;       return;
            case FAM_QUASI_POISSON_IDENTITY:
                link = QuasiLinkType::Identity; var = QuasiVarianceType::Mu;       return;
            case FAM_QUASI_BINOMIAL_LOGIT:
                link = QuasiLinkType::Logit;    var = QuasiVarianceType::MuMu;     return;
            case FAM_QUASI_BINOMIAL_PROBIT:
                link = QuasiLinkType::Probit;   var = QuasiVarianceType::MuMu;     return;
            case FAM_QUASI_GAUSSIAN_IDENTITY:
                link = QuasiLinkType::Identity; var = QuasiVarianceType::Constant; return;
            case FAM_QUASI_GAUSSIAN_LOG:
                link = QuasiLinkType::Log;      var = QuasiVarianceType::Constant; return;
            case FAM_QUASI_GAMMA_LOG:
                link = QuasiLinkType::Log;      var = QuasiVarianceType::MuSquared; return;
            case FAM_QUASI_GAMMA_IDENTITY:
                link = QuasiLinkType::Identity; var = QuasiVarianceType::MuSquared; return;
            case FAM_QUASI_GAMMA_INVERSE:
                link = QuasiLinkType::Inverse;  var = QuasiVarianceType::MuSquared; return;
            case FAM_QUASI_INVGAUSSIAN_LOG:
                link = QuasiLinkType::Log;      var = QuasiVarianceType::MuCubed; return;
            case FAM_QUASI_INVGAUSSIAN_IDENTITY:
                link = QuasiLinkType::Identity; var = QuasiVarianceType::MuCubed; return;
            case FAM_QUASI_INVGAUSSIAN_INVERSE:
                link = QuasiLinkType::Inverse;  var = QuasiVarianceType::MuCubed; return;
            default:
                throw std::invalid_argument("QuasiSolver: specified family code is not a recognized FAM_QUASI_* family.");
        }
    }

    /**
     * @brief Computes Pearson dispersion scale factor $\phi = \frac{1}{n-p} \sum \frac{w_i (y_i - \mu_i)^2}{V(\mu_i)}$.
     */
    template <QuasiLinkType Link, QuasiVarianceType Var, typename Policy>
    static double pearson_dispersion_vectorized(const QuasiLoss<Link, Var, Policy>& loss_fn,
                                                const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                                const Eigen::VectorXd& w, const Eigen::VectorXd& beta) {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd eta = X * beta;
        if (loss_fn.offset.size() == eta.size()) eta += loss_fn.offset;
        Eigen::VectorXd mu_v(n);

        if constexpr (Link == QuasiLinkType::Identity) {
            mu_v = eta;
        } else if constexpr (Link == QuasiLinkType::Log) {
            mu_v = eta.array().cwiseMin(30.0).exp();
        } else if constexpr (Link == QuasiLinkType::Inverse) {
            const double d = 0.1;
            Eigen::ArrayXd disc = (eta.array().square() + 4.0 * d * d).sqrt();
            Eigen::ArrayXd eta_pos = 0.5 * (eta.array() + disc);
            mu_v = eta_pos.inverse().matrix();
        } else if constexpr (Link == QuasiLinkType::Logit) {
            Eigen::VectorXd clamped = eta.array().cwiseMax(-30.0).cwiseMin(30.0);
            mu_v = 1.0 / (1.0 + (-clamped).array().exp());
        } else if constexpr (Link == QuasiLinkType::Probit) {
            mu_v = eta.unaryExpr([](double val) { return 0.5 * (1.0 + std::erf(val * 0.7071067811865475)); });
        }

        Eigen::VectorXd v_mu_v(n);
        constexpr double eps = 1e-10;
        if constexpr (Var == QuasiVarianceType::Custom) {
            loss_fn.policy.variance(mu_v, v_mu_v);
        } else {
            if constexpr (Var == QuasiVarianceType::Constant) {
                v_mu_v.setConstant(1.0);
            } else if constexpr (Var == QuasiVarianceType::Mu) {
                v_mu_v = mu_v.array().abs().cwiseMax(eps);
            } else if constexpr (Var == QuasiVarianceType::MuMu) {
                Eigen::VectorXd c = mu_v.array().cwiseMax(eps).cwiseMin(1.0 - eps);
                v_mu_v = c.array() * (1.0 - c.array());
            } else if constexpr (Var == QuasiVarianceType::MuSquared) {
                v_mu_v = mu_v.array().square().cwiseMax(eps);
            } else if constexpr (Var == QuasiVarianceType::MuCubed) {
                v_mu_v = mu_v.array().abs().cube().cwiseMax(eps);
            } else if constexpr (Var == QuasiVarianceType::MuPower) {
                constexpr double p_val = Policy::p;
                v_mu_v = mu_v.array().abs().pow(p_val).cwiseMax(eps);
            }
        }

        auto pos_eps = Eigen::VectorXd::Constant(n, 1e-12);
        auto neg_eps = Eigen::VectorXd::Constant(n, -1e-12);
        v_mu_v = (v_mu_v.array().abs() < 1e-12).select((v_mu_v.array() >= 0).select(pos_eps, neg_eps), v_mu_v);

        Eigen::VectorXd resid = y.array() - mu_v.array();
        double sum = (w.array() * resid.array().square() / v_mu_v.array()).sum();

        // residual df = (observations with positive prior weight) - rank, as in R's glm.fit
        const long n_ok = (w.array() > 0.0).count();
        long dof = n_ok - k;
        return dof > 0 ? sum / static_cast<double>(dof) : std::numeric_limits<double>::quiet_NaN();
    }

    /**
     * @brief Attaches asymptotic covariance/SE to a QuasiResult after optimization converges.
     * @details $\widehat{\mathrm{vcov}} = \hat\phi (X^T W X)^{-1}$ with $W = \mathrm{diag}(w_i \cdot (d\mu/d\eta)_i^2 / V(\mu_i))$
     *          -- same formula the IRLS engine uses (see irls.hpp::update_w and fit_quasi in
     *          irls_two_step.hpp), evaluated here at $\hat\beta$ since L-BFGS-B/BFGS/NelderMead
     *          never build this matrix themselves. Uses `invert_hessian` (mle_vcov.hpp) for the
     *          same Cholesky/LDLT/Tikhonov/SVD fallback chain already used for the MLE families,
     *          so a rank-deficient or near-singular $X^TWX$ degrades gracefully instead of throwing.
     *          `res.dispersion` must already be set (call after `pearson_dispersion_vectorized`).
     */
    template <typename LossFunc>
    static void attach_quasi_vcov(QuasiResult& res, const LossFunc& loss_fn,
                                   const Eigen::MatrixXd& X, const Eigen::VectorXd& w,
                                   const Eigen::VectorXd& beta_opt) {
        const Eigen::VectorXd info_w = loss_fn.irls_weights(beta_opt);
        const Eigen::VectorXd combined_w = (w.array() * info_w.array()).matrix();
        Eigen::MatrixXd XtWX = X.transpose() * combined_w.asDiagonal() * X;
        Eigen::MatrixXd vcov = invert_hessian(XtWX);
        if (std::isfinite(res.dispersion)) vcov *= res.dispersion;
        res.vcov = vcov;
        res.se = vcov_to_se(vcov);
    }

protected:
    /**
     * @brief Dispatches optimization loop using selected minimizer algorithm.
     */
    template <typename LossFunc>
    std::pair<QuasiResult, Eigen::VectorXd> run_optimization(const LossFunc& loss_fn,
                                                              const Eigen::VectorXd& x0,
                                                              [[maybe_unused]] int total_params,
                                                              int maxiter,
                                                              double ftol,
                                                              double gtol,
                                                              int maxcor,
                                                              int verbose)
    {
        QuasiResult res;

        const double huge = std::numeric_limits<double>::max() / 1e6;
        auto f = [&](const Eigen::VectorXd& x) -> double {
            const double v = loss_fn(x);
            return std::isfinite(v) ? v : huge;
        };
        // If the gradient is non-finite, L-BFGS-B receives gd = g·d = 0 (>= 0)
        // and, with empty memory (col==0), produces Task::Abnormal on the 1st iteration.
        // Instead, we return a penalty gradient pointing toward the origin:
        // it provides a valid descent direction and lets the solver escape the
        // degenerate region without ending in error.
        auto jac = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd g = loss_fn.gradient(x);
            if (!g.allFinite()) {
                // Penalty: the gradient points from x toward the origin,
                // scaled to be consistent with the magnitude of 'huge'.
                g = -x * (huge / std::max(1.0, x.norm()));
                if (!g.allFinite()) g = Eigen::VectorXd::Zero(x.size());
            }
            return g;
        };

        const min::VerbosityLevel vlevel =
            (verbose > 0) ? min::VerbosityLevel::Iteration : min::VerbosityLevel::Silent;

        double fun = 0.0;
        Eigen::VectorXd x_final = x0;

        switch (minimizer_) {
            case Minimizer::BFGS: {
                min::BfgsOptions<double> opt;
                opt.maxiter = maxiter;
                opt.ftol    = ftol;
                opt.gtol    = gtol;
                opt.verbose = vlevel;

                auto r = min::bfgs<double>(f, x0, opt, jac);
                res.success              = r.success;
                res.message              = r.message;
                res.iterations           = r.nit;
                res.function_evaluations = r.nfev;
                fun                      = r.fun;
                x_final                  = r.x;
                break;
            }
            case Minimizer::NelderMead: {
                min::NelderMeadOptions<double> opt;
                opt.maxiter = maxiter;
                opt.ftol    = ftol;
                opt.xtol    = std::max(ftol, 1e-6);
                opt.adaptive = true;  // keeps the previous behavior (the default is now false, as in scipy)
                opt.verbose = vlevel;

                auto r = min::nelder_mead<double>(f, x0, opt);
                res.success              = r.success;
                res.message              = r.message;
                res.iterations           = r.nit;
                res.function_evaluations = r.nfev;
                fun                      = r.fun;
                x_final                  = r.x;
                break;
            }
            case Minimizer::LBFGSB:
            default: {
                min::LbfgsbOptions<double> opt;
                opt.maxiter = maxiter;
                opt.ftol    = ftol;
                opt.gtol    = gtol;
                opt.maxcor  = maxcor;
                opt.verbose = vlevel;

                auto r = min::lbfgsb<double>(f, x0, opt, jac);
                res.success              = r.success;
                res.message              = r.message;
                res.iterations           = r.nit;
                res.function_evaluations = r.nfev;
                fun                      = r.fun;
                x_final                  = r.x;
                break;
            }
        }

        if (!std::isfinite(fun)) {
            res.success = false;
            res.message = "Numerical failure: Loss function returned non-finite value (NaN/Inf).";
        }

        res.quasi_deviance = fun;
        return std::make_pair(std::move(res), std::move(x_final));
    }

public:
    /**
     * @brief Constructs a new QuasiSolver instance.
     * @param fit_intercept Automatically prepend intercept column of ones if true.
     * @param minimizer Minimizer algorithm choice (default = LBFGSB).
     */
    explicit QuasiSolver(bool fit_intercept = true, Minimizer minimizer = Minimizer::LBFGSB)
        : fit_intercept_(fit_intercept), minimizer_(minimizer) {}

    /** @brief Sets optimization minimizer choice. */
    void set_minimizer(Minimizer m) { minimizer_ = m; }
    /** @brief Gets current optimization minimizer choice. */
    Minimizer get_minimizer() const { return minimizer_; }

    /**
     * @brief Fits predefined Quasi-likelihood GLM model.
     * @param family Family code (e.g. `FAM_QUASI_POISSON_LOG`, `FAM_QUASI_GAMMA_LOG`).
     * @param X_raw Raw design matrix ($n \times k$).
     * @param y Response vector ($n \times 1$).
     * @param weights Optional observation weights.
     * @param maxiter Maximum optimizer iterations.
     * @param ftol Function value change tolerance.
     * @param gtol Gradient norm tolerance.
     * @param maxcor Memory corrections for L-BFGS-B.
     * @param verbose Verbosity level.
     * @return `QuasiResult` object containing fitted coefficients and deviance.
     */
    QuasiResult fit(FamilyCode family,
                    const Eigen::MatrixXd& X_raw,
                    const Eigen::VectorXd& y,
                    std::optional<Eigen::VectorXd> weights = std::nullopt,
                    int maxiter = 1000,
                    double ftol = 1e-12,
                    double gtol = 1e-6,
                    int maxcor = 10,
                    int verbose = 0,
                    std::optional<Eigen::VectorXd> offset = std::nullopt)
    {
        QuasiLinkType link;
        QuasiVarianceType var;
        resolve_quasi(family, link, var);

        check_finite(X_raw, "X");
        check_finite(y, "y");

        Eigen::MatrixXd X = fit_intercept_ ? add_intercept(X_raw) : X_raw;
        const long n = X.rows();
        const long k = X.cols();

        if (y.size() != n)
            throw std::invalid_argument("QuasiSolver::fit: X and y row dimensions mismatch.");

        Eigen::VectorXd w = weights.value_or(Eigen::VectorXd::Ones(n));
        if (w.size() != n)
            throw std::invalid_argument("QuasiSolver::fit: weights size must match nrow(X).");
        check_finite(w, "weights");

        check_full_rank(X, "QuasiSolver::fit");
        check_identifiable(n, k, "QuasiSolver::fit");

        Eigen::VectorXd off = offset.value_or(Eigen::VectorXd::Zero(n));
        if (off.size() != n)
            throw std::invalid_argument("QuasiSolver::fit: offset size must match nrow(X).");
        check_finite(off, "offset");

        Eigen::VectorXd x0 = compute_warm_start(link, X, y, off);

        QuasiResult res;
        Eigen::VectorXd beta_opt;

        switch (family) {
            case FAM_QUASI_POISSON_LOG: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Log, glm::QuasiVarianceType::Mu>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_POISSON_IDENTITY: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Identity, glm::QuasiVarianceType::Mu>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_BINOMIAL_LOGIT: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Logit, glm::QuasiVarianceType::MuMu>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_BINOMIAL_PROBIT: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Probit, glm::QuasiVarianceType::MuMu>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_GAUSSIAN_IDENTITY: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Identity, glm::QuasiVarianceType::Constant>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_GAUSSIAN_LOG: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Log, glm::QuasiVarianceType::Constant>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_GAMMA_LOG: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Log, glm::QuasiVarianceType::MuSquared>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_GAMMA_IDENTITY: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Identity, glm::QuasiVarianceType::MuSquared>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_GAMMA_INVERSE: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Inverse, glm::QuasiVarianceType::MuSquared>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_INVGAUSSIAN_LOG: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Log, glm::QuasiVarianceType::MuCubed>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_INVGAUSSIAN_IDENTITY: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Identity, glm::QuasiVarianceType::MuCubed>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            case FAM_QUASI_INVGAUSSIAN_INVERSE: {
                using Loss = glm::QuasiLoss<glm::QuasiLinkType::Inverse, glm::QuasiVarianceType::MuCubed>;
                Loss loss_fn(X, y, w);
                loss_fn.set_offset(off);
                std::tie(res, beta_opt) = run_optimization(loss_fn, x0, static_cast<int>(k), maxiter, ftol, gtol, maxcor, verbose);
                if (beta_opt.size() == static_cast<long>(k)) {
                    res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                    attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
                }
                break;
            }
            default:
                throw std::invalid_argument("QuasiSolver: family code is not a valid FAM_QUASI_* code.");
        }

        res.beta = beta_opt;
        return res;
    }

    /**
     * @brief Fits Quasi-likelihood model with a user-supplied custom variance function $V(\mu)$.
     * @tparam VarianceFunc Callable type $V(\mu)$.
     * @param X_raw Raw design matrix.
     * @param y Response vector.
     * @param v_mu_lambda Custom variance function callable $V(\mu)$.
     * @param weights Optional weights.
     * @param link Link type enum.
     * @param maxiter Maximum optimizer iterations.
     * @param ftol Function tolerance.
     * @param gtol Gradient tolerance.
     * @return `QuasiResult` object.
     */
    template<typename VarianceFunc>
    QuasiResult fit_custom(const Eigen::MatrixXd& X_raw,
                           const Eigen::VectorXd& y,
                           VarianceFunc&& v_mu_lambda,
                           std::optional<Eigen::VectorXd> weights = std::nullopt,
                           QuasiLinkType link = QuasiLinkType::Log,
                           int maxiter = 1000,
                           double ftol = 1e-12,
                           double gtol = 1e-6,
                           std::optional<Eigen::VectorXd> offset = std::nullopt)
    {
        using RawVarFunc = std::decay_t<VarianceFunc>;
        using Policy = CustomVariancePolicy<RawVarFunc>;
        Policy policy(std::forward<VarianceFunc>(v_mu_lambda));

        check_finite(X_raw, "X");
        check_finite(y, "y");

        Eigen::MatrixXd X = fit_intercept_ ? add_intercept(X_raw) : X_raw;
        const long n = X.rows();
        const long k = X.cols();

        if (y.size() != n) {
            throw std::invalid_argument("QuasiSolver::fit_custom: X and y row dimensions mismatch.");
        }
        check_opt_dims(weights, n, "weights", "QuasiSolver::fit_custom");
        check_finite_opt(weights, "weights");
        check_full_rank(X, "QuasiSolver::fit_custom");
        check_identifiable(n, k, "QuasiSolver::fit_custom");

        Eigen::VectorXd w = weights.value_or(Eigen::VectorXd::Ones(n));
        Eigen::VectorXd off = offset.value_or(Eigen::VectorXd::Zero(n));
        if (off.size() != n)
            throw std::invalid_argument("QuasiSolver::fit_custom: offset size must match nrow(X).");
        check_finite(off, "offset");
        Eigen::VectorXd x0 = compute_warm_start(link, X, y, off);

        QuasiResult res;
        Eigen::VectorXd beta_opt;

        if (link == QuasiLinkType::Log) {
            using Loss = glm::QuasiLoss<QuasiLinkType::Log,
                                         QuasiVarianceType::Custom,
                                         Policy>;
            Loss loss_fn(X, y, w, policy);
            loss_fn.set_offset(off);
            std::tie(res, beta_opt) = this->run_optimization(loss_fn, x0,
                                                             static_cast<int>(k),
                                                             maxiter, ftol, gtol,
                                                             10, 0);
            if (beta_opt.size() == static_cast<long>(k)) {
                res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
            }
        } else if (link == QuasiLinkType::Identity) {
            using Loss = glm::QuasiLoss<QuasiLinkType::Identity,
                                         QuasiVarianceType::Custom,
                                         Policy>;
            Loss loss_fn(X, y, w, policy);
            loss_fn.set_offset(off);
            std::tie(res, beta_opt) = this->run_optimization(loss_fn, x0,
                                                             static_cast<int>(k),
                                                             maxiter, ftol, gtol,
                                                             10, 0);
            if (beta_opt.size() == static_cast<long>(k)) {
                res.dispersion = pearson_dispersion_vectorized(loss_fn, X, y, w, beta_opt);
                attach_quasi_vcov(res, loss_fn, X, w, beta_opt);
            }
        } else {
            throw std::invalid_argument("fit_custom: Link function not supported for custom optimization.");
        }

        res.beta = beta_opt;
        return res;
    }

};

} // namespace glm

#endif // _GLMFIT_QUASI_SOLVER_HPP
