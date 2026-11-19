// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file mle_solver.hpp
 * @brief Multi-component Maximum Likelihood Estimation (MLE) driver class.
 * @details Implements the `MleSolver` class supporting parametric regression model fitting for survival AFT models
 *          (Log-Normal, Weibull, Log-Logistic), inflated count models (ZIP, ZINB, Hurdle Poisson/NB/Gamma/Beta),
 *          Beta regression, Simplex, Multinomial, Ordinal, Skew-Normal, Skew-t, Student-t, Truncated distributions,
 *          and Generalized Gamma (Prentice 1974) multi-parameter survival models.
 *
 * @par Secondary link
 *      Families that model a second parameter through its own linear predictor (`Z`) accept a
 *      link for that parameter via `fit(..., secondary_link, ...)`:
 *      - probability parameter pi in (0,1) -- ZIP, ZINB (inflation) and Hurdle Poisson/NB/Gamma/LogNormal/Beta
 *        (barrier P(Y>0)): Logit (default), Probit, Cloglog, Loglog, Cauchit.
 *      - positive parameter -- Beta regression (precision phi) and Simplex (dispersion sigma^2):
 *        Log (default), Identity, Sqrt, Inverse.
 *      See `MleSolver::supports_secondary_link` / `MleSolver::default_secondary_link`.
 */

#ifndef _GLMFIT_MLE_SOLVE_HPP
#define _GLMFIT_MLE_SOLVE_HPP

#include <vector>
#include <numeric>
#include "log_likelihood.hpp"
#include "mle_vcov.hpp"
#include "em_warmstart.hpp"
#include "input_validation.hpp"
#include "minimizer_choice.hpp"
#include "../minimizer/lbfgsb.hpp"
#include "../minimizer/bfgs.hpp"
#include "../minimizer/nelder_mead.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <optional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>
#include <utility>
#include <memory>
#include <mutex>

namespace glm {

/** @brief $\pi$ constant definition. */
constexpr double glm_PI = 3.14159265358979323846;

/**
 * @brief High-precision rational approximation of the standard normal quantile function $\Phi^{-1}(p)$.
 * @param p Cumulative probability scalar $p \in (0, 1)$.
 * @return Quantile scalar value $z$.
 */
inline double pure_qnorm(double p) {
    p = std::clamp(p, 1e-15, 1.0 - 1e-15);
    double q = p - 0.5;
    if (std::abs(q) <= 0.42) {
        double r = q * q;
        return q * (((-25.44106049637 * r + 41.39119773534) * r - 18.61500062529) * r + 2.50662823884) /
               ((((3.13082909833 * r - 21.06224101826) * r + 23.08336743743) * r - 8.47351093090) * r + 1.0);
    }
    double r = p < 0.5 ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double x = (((0.33747573768 * r + 0.97616488914) * r + 0.16079797149) * r - 2.78718931138) /
               ((0.00039237019 * r + 0.37518216202) * r + 1.0);
    return p < 0.5 ? -x : x;
}

/**
 * @brief Uniform result container for MLE estimation results.
 */
struct MleResult {
    bool success = false;                 ///< Optimization convergence flag
    std::string message;                  ///< Solver status / termination message
    double log_likelihood = 0.0;          ///< Final maximized log-likelihood value ($\ell$)
    int iterations = 0;                   ///< Number of optimizer iterations executed
    int function_evaluations = 0;         ///< Number of objective function evaluations

    Eigen::VectorXd beta;                 ///< Main component fitted parameter vector ($\beta$)
    Eigen::VectorXd gamma;                ///< Zero-inflation / secondary component parameter vector ($\gamma$)
    double extra_param = 0.0;             ///< Additional scalar parameter (e.g. dispersion $\alpha$, scale $\sigma$, shape $Q$)
    double extra_param2 = 0.0;            ///< Secondary additional scalar parameter (e.g. $\ln\alpha$)

    Eigen::VectorXd x_opt;                ///< Full raw parameter vector from optimization backend

    std::optional<Eigen::MatrixXd> vcov;  ///< Asymptotic variance-covariance matrix (populated if `compute_se=true`)
    std::optional<Eigen::VectorXd> se;    ///< Standard error estimates vector (populated if `compute_se=true`)
};

/**
 * @brief Automated multi-component Maximum Likelihood Estimation solver class.
 */
class MleSolver {
private:
    bool fit_intercept_;                  ///< Flag indicating whether to automatically prepend intercept column
    Minimizer minimizer_ = Minimizer::LBFGSB;  ///< Selected optimization backend minimizer

    Eigen::MatrixXd add_intercept(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd X_new(X.rows(), X.cols() + 1);
        X_new.col(0) = Eigen::VectorXd::Ones(X.rows());
        X_new.rightCols(X.cols()) = X;
        return X_new;
    }

    /**
     * @brief Computes forward link transformation for a single scalar.
     */
    static double compute_single_link_forward(double mu, LinkType link) {
        switch (link) {
            case LinkType::Identity:
                return mu;
            case LinkType::Log:
                return std::log(std::max(mu, 1e-6));
            case LinkType::Sqrt:
                return std::sqrt(std::max(mu, 0.0));
            case LinkType::Inverse: {
                double abs_mu = std::abs(mu);
                if (abs_mu < 1e-6) return (mu >= 0.0) ? 1e6 : -1e6;
                return 1.0 / mu;
            }
            case LinkType::Logit: {
                double p = std::clamp(mu, 1e-5, 1.0 - 1e-5);
                return std::log(p / (1.0 - p));
            }
            case LinkType::Probit: {
                double p = std::clamp(mu, 1e-5, 1.0 - 1e-5);
                return pure_qnorm(p);
            }
            case LinkType::Cloglog: {
                double p = std::clamp(mu, 1e-5, 1.0 - 1e-5);
                return std::log(-std::log(1.0 - p));
            }
            case LinkType::Loglog: {
                double p = std::clamp(mu, 1e-5, 1.0 - 1e-5);
                return -std::log(-std::log(p));
            }
            case LinkType::Cauchit: {
                double p = std::clamp(mu, 1e-5, 1.0 - 1e-5);
                return std::tan(glm_PI * (p - 0.5));
            }
            default:
                return mu;
        }
    }

    /**
     * @brief Transforms vector of mean response values to linear predictor scale via link function.
     */
    static Eigen::VectorXd transform_link_forward(const Eigen::VectorXd& mu_vec, LinkType link) {
        Eigen::VectorXd eta(mu_vec.size());
        for (Eigen::Index i = 0; i < mu_vec.size(); ++i) {
            eta(i) = compute_single_link_forward(mu_vec(i), link);
        }
        return eta;
    }


    /**
     * @brief Maps family code to default link type enum.
     */
    static LinkType resolve_link(FamilyCode code) {
        switch (code) {
            case FAM_GAUSSIAN_IDENTITY:
            case FAM_POISSON_IDENTITY:
            case FAM_GAMMA_IDENTITY:
            case FAM_INVGAUSS_IDENTITY:
            case FAM_TWEEDIE_IDENTITY:
            case FAM_EXPONENTIAL_IDENTITY:
            case FAM_NB_IDENTITY:
            case FAM_LOGNORMAL_IDENTITY:
            case FAM_WEIBULL_IDENTITY:
            case FAM_GUMBEL_IDENTITY:
            case FAM_LOGLOGISTIC_IDENTITY:
            case FAM_QUASI_POISSON_IDENTITY:
            case FAM_QUASI_GAUSSIAN_IDENTITY:
            case FAM_STUDENT_T_IDENTITY:
            case FAM_SKEW_NORMAL_IDENTITY:
            case FAM_SKEW_T_IDENTITY:
            case FAM_LOGISTIC_IDENTITY:
            case FAM_ZIP_IDENTITY:
            case FAM_ZINB_IDENTITY:
            case FAM_HURDLE_LOGNORMAL_IDENTITY:
            case FAM_TRUNC_POISSON_IDENTITY:
            case FAM_TRUNC_NB_IDENTITY:
            case FAM_TRUNC_GAUSSIAN_IDENTITY:
                return LinkType::Identity;

            case FAM_POISSON_SQRT:
            case FAM_TWEEDIE_SQRT:
            case FAM_NB_SQRT:
            case FAM_ZIP_SQRT:
            case FAM_ZINB_SQRT:
            case FAM_HURDLE_POISSON_SQRT:
            case FAM_HURDLE_NB_SQRT:
                return LinkType::Sqrt;

            case FAM_GAUSSIAN_INVERSE:
            case FAM_POISSON_INVERSE:
            case FAM_GAMMA_INVERSE:
            case FAM_INVGAUSS_INVERSE:
            case FAM_TWEEDIE_INVERSE:
            case FAM_EXPONENTIAL_INVERSE:
            case FAM_NB_INVERSE:
            case FAM_LOGNORMAL_INVERSE:
            case FAM_INVGAMMA_INVERSE:
            case FAM_ZIP_INVERSE:
            case FAM_ZINB_INVERSE:
            case FAM_HURDLE_GAMMA_INVERSE:
            case FAM_TRUNC_GAUSSIAN_INVERSE:
                return LinkType::Inverse;

            default:
                break;
        }

        if ((code >= FAM_BINOMIAL_LOGIT && code <= FAM_BINOMIAL_LOG) ||
            code == FAM_BINOMIAL_CAUCHIT || code == FAM_BINOMIAL_LOGLOG)
        {
            if (code == FAM_BINOMIAL_PROBIT)   return LinkType::Probit;
            if (code == FAM_BINOMIAL_LOGIT)    return LinkType::Logit;
            if (code == FAM_BINOMIAL_LOG)      return LinkType::Log;
            return LinkType::Logit;
        }

        if (code == FAM_BETA_LOGIT || code == FAM_BETA_PROBIT
            || code == FAM_BETA_LOGLOG || code == FAM_BETA_CLOGLOG
            || code == FAM_BETA_CAUCHIT) {

            if (code == FAM_BETA_PROBIT) return LinkType::Probit;
            if (code == FAM_BETA_LOGLOG) return LinkType::Loglog;
            if (code == FAM_BETA_CLOGLOG) return LinkType::Cloglog;
            if (code == FAM_BETA_CAUCHIT) return LinkType::Cauchit;

            return LinkType::Logit;
        }

        if (code == FAM_SIMPLEX_LOGIT || code == FAM_SIMPLEX_PROBIT
            || code == FAM_SIMPLEX_LOGLOG || code == FAM_SIMPLEX_CLOGLOG
            || code == FAM_SIMPLEX_CAUCHIT) {

            if (code == FAM_SIMPLEX_PROBIT) return LinkType::Probit;
            if (code == FAM_SIMPLEX_LOGLOG) return LinkType::Loglog;
            if (code == FAM_SIMPLEX_CLOGLOG) return LinkType::Cloglog;
            if (code == FAM_SIMPLEX_CAUCHIT) return LinkType::Cauchit;

            return LinkType::Logit;
        }

        if (code == FAM_HURDLE_BETA_LOGIT || code == FAM_HURDLE_BETA_PROBIT
            || code == FAM_HURDLE_BETA_LOGLOG || code == FAM_HURDLE_BETA_CLOGLOG
            || code == FAM_HURDLE_BETA_CAUCHIT) {

            if (code == FAM_HURDLE_BETA_PROBIT) return LinkType::Probit;
            if (code == FAM_HURDLE_BETA_LOGLOG) return LinkType::Loglog;
            if (code == FAM_HURDLE_BETA_CLOGLOG) return LinkType::Cloglog;
            if (code == FAM_HURDLE_BETA_CAUCHIT) return LinkType::Cauchit;
            return LinkType::Logit;
        }

        if (code == FAM_HURDLE_GEOMETRIC_LOG)      return LinkType::Log;
        if (code == FAM_HURDLE_GEOMETRIC_IDENTITY) return LinkType::Identity;
        if (code == FAM_HURDLE_GEOMETRIC_SQRT)     return LinkType::Sqrt;


        if (code >= FAM_MULTINOMIAL_LOGIT && code <= FAM_ORDINAL_CAUCHIT) {
            if (code == FAM_MULTINOMIAL_PROBIT  || code == FAM_ORDINAL_PROBIT)  return LinkType::Probit;
            if (code == FAM_MULTINOMIAL_CLOGLOG || code == FAM_ORDINAL_CLOGLOG) return LinkType::Cloglog;
            if (code == FAM_MULTINOMIAL_LOGLOG  || code == FAM_ORDINAL_LOGLOG)  return LinkType::Loglog;
            if (code == FAM_MULTINOMIAL_CAUCHIT || code == FAM_ORDINAL_CAUCHIT) return LinkType::Cauchit;

            return LinkType::Logit;
        }

        if (code == FAM_INVGAUSS_INVMU2) {
            return LinkType::Inverse;
        }

        return LinkType::Log;
    }


    // -----------------------------------------------------------------------
    // SECONDARY LINK (link of the parameter modelled by the second predictor Z)
    // -----------------------------------------------------------------------

    /** @brief Kind of parameter modelled by the second linear predictor. */
    enum class SecondaryKind {
        None,         ///< Family has no second linear predictor (secondary_link is ignored)
        Probability,  ///< pi in (0,1): zero-inflation probability / hurdle barrier P(Y>0)
        Positive      ///< strictly positive: Beta precision phi, Simplex dispersion sigma^2
    };

    static SecondaryKind secondary_kind(FamilyCode code) {
        switch (code) {
            case FAM_ZIP_LOG:
            case FAM_ZIP_IDENTITY:
            case FAM_ZIP_SQRT:
            case FAM_ZIP_INVERSE:
            case FAM_ZINB_LOG:
            case FAM_ZINB_SQRT:
            case FAM_ZINB_IDENTITY:
            case FAM_ZINB_INVERSE:
            case FAM_HURDLE_POISSON_LOG:
            case FAM_HURDLE_POISSON_SQRT:
            case FAM_HURDLE_NB_LOG:
            case FAM_HURDLE_NB_SQRT:
            case FAM_HURDLE_GAMMA_LOG:
            case FAM_HURDLE_GAMMA_INVERSE:
            case FAM_HURDLE_LOGNORMAL_LOG:
            case FAM_HURDLE_LOGNORMAL_IDENTITY:

            case FAM_HURDLE_BETA_LOGIT:
            case FAM_HURDLE_BETA_PROBIT:
            case FAM_HURDLE_BETA_LOGLOG:
            case FAM_HURDLE_BETA_CLOGLOG:
            case FAM_HURDLE_BETA_CAUCHIT:

            case FAM_HURDLE_GEOMETRIC_LOG:
            case FAM_HURDLE_GEOMETRIC_IDENTITY:
            case FAM_HURDLE_GEOMETRIC_SQRT:
                return SecondaryKind::Probability;

            case FAM_BETA_LOGIT:
            case FAM_BETA_PROBIT:
            case FAM_BETA_CLOGLOG:
            case FAM_BETA_LOGLOG:
            case FAM_BETA_CAUCHIT:

            case FAM_SIMPLEX_LOGIT:
            case FAM_SIMPLEX_PROBIT:
            case FAM_SIMPLEX_LOGLOG:
            case FAM_SIMPLEX_CLOGLOG:
            case FAM_SIMPLEX_CAUCHIT:
                return SecondaryKind::Positive;

            default:
                return SecondaryKind::None;
        }
    }

    static bool is_probability_link(LinkType l) {
        return l == LinkType::Logit  || l == LinkType::Probit  || l == LinkType::Cloglog ||
               l == LinkType::Loglog || l == LinkType::Cauchit;
    }

    static bool is_positive_link(LinkType l) {
        return l == LinkType::Log  || l == LinkType::Identity ||
               l == LinkType::Sqrt || l == LinkType::Inverse;
    }

    static const char* link_name(LinkType l) {
        switch (l) {
            case LinkType::Identity:       return "identity";
            case LinkType::Log:            return "log";
            case LinkType::Sqrt:           return "sqrt";
            case LinkType::Logit:          return "logit";
            case LinkType::Probit:         return "probit";
            case LinkType::Cloglog:        return "cloglog";
            case LinkType::Inverse:        return "inverse";
            case LinkType::Loglog:         return "loglog";
            case LinkType::Cauchit:        return "cauchit";
            case LinkType::InverseSquared: return "inverse_squared";
        }
        return "unknown";
    }

    /** @brief Throws std::invalid_argument if `link` is not valid for a secondary parameter of the given kind. */
    static void validate_secondary_link(LinkType link, SecondaryKind kind, const char* context) {
        if (kind == SecondaryKind::Probability && !is_probability_link(link)) {
            throw std::invalid_argument(
                std::string(context) + ": link '" + link_name(link) + "' is not valid for a probability parameter "
                "(zero-inflation / hurdle barrier). Allowed: logit, probit, cloglog, loglog, cauchit.");
        }
        if (kind == SecondaryKind::Positive && !is_positive_link(link)) {
            throw std::invalid_argument(
                std::string(context) + ": link '" + link_name(link) + "' is not valid for a strictly positive parameter "
                "(Beta precision / Simplex dispersion). Allowed: log, identity, sqrt, inverse.");
        }
    }

    /**
     * @brief Resolves and validates the link of the secondary parameter.
     * @details Families without a second linear predictor ignore `requested` (as `infl_link` did before).
     *          Defaults: Logit for probability parameters, Log for positive parameters.
     */
    static LinkType resolve_secondary_link(FamilyCode family, std::optional<LinkType> requested, const char* context) {
        const SecondaryKind kind = secondary_kind(family);
        if (kind == SecondaryKind::None) return LinkType::Logit;   // unused for this family
        const LinkType link = requested.value_or(kind == SecondaryKind::Probability ? LinkType::Logit
                                                                                     : LinkType::Log);
        validate_secondary_link(link, kind, context);
        return link;
    }

    /**
     * @brief Linear-predictor value that maps to `value` (> 0) under a link valid for a positive parameter.
     * @details Same as `compute_single_link_forward` except for Inverse, which must invert the smooth positive
     *          inverse used by the loss functions (eta -> 1/eta_pos with eta_pos = (eta + sqrt(eta^2 + 4 d^2)) / 2,
     *          d = 0.1), i.e. eta = 1/value - d^2 * value.
     */
    static double compute_positive_link_forward(double value, LinkType link) {
        if (link == LinkType::Inverse) {
            const double v = std::max(value, 1e-6);
            return 1.0 / v - 0.01 * v;
        }
        return compute_single_link_forward(value, link);
    }

    /** @brief Inverse link for a strictly positive parameter (consistent with the loss functors). */
    static double evaluate_positive_link(double eta, LinkType link) {
        double v = 0.0, d1 = 0.0, d2 = 0.0;
        if (link == LinkType::Inverse) {
            evaluate_positive_inverse_d2(eta, v, d1, d2);
        } else {
            evaluate_link_d2(eta, link, v, d1, d2);
        }
        return v;
    }

    bool skewt_multistart_ = true;        ///< Skew-t: fit from several starts and keep the best log-likelihood

    /** @brief Quantile of a SORTED sample (linear interpolation, same as R type=7). */
    static double quantile_sorted(const std::vector<double>& s, double p) {
        const size_t m = s.size();
        if (m == 0) return 0.0;
        const double h = (static_cast<double>(m) - 1.0) * p;
        const size_t lo = static_cast<size_t>(std::floor(h));
        const size_t hi = std::min(lo + 1, m - 1);
        return s[lo] + (h - static_cast<double>(lo)) * (s[hi] - s[lo]);
    }

    /** @brief Trimmed mean (p in each tail) of a SORTED sample. */
    static double trimmed_mean_sorted(const std::vector<double>& s, double p) {
        const size_t m = s.size();
        if (m == 0) return 0.0;
        const size_t cut = static_cast<size_t>(std::floor(p * static_cast<double>(m)));
        double acc = 0.0; size_t cnt = 0;
        for (size_t i = cut; i + cut < m; ++i) { acc += s[i]; ++cnt; }
        return cnt ? acc / static_cast<double>(cnt) : 0.0;
    }

    /**
     * @brief Median (LAD, L1) regression via iteratively reweighted least squares (IRLS).
     * @details Similar in spirit to R's quantreg::rq(tau=0.5) as used by st.prelimFit: robust to heavy tails.
     */
    static Eigen::VectorXd lad_irls(const Eigen::MatrixXd& X, const Eigen::VectorXd& t,
                                    const Eigen::VectorXd& b_init, int maxit = 100) {
        Eigen::VectorXd b = b_init;
        Eigen::VectorXd r = t - X * b;
        const double delta = 1e-6 * (1.0 + r.cwiseAbs().mean());
        for (int it = 0; it < maxit; ++it) {
            r = t - X * b;
            Eigen::VectorXd w = r.cwiseAbs().cwiseMax(delta).cwiseInverse();
            Eigen::MatrixXd XtW = X.transpose() * w.asDiagonal();
            Eigen::VectorXd bn = (XtW * X).ldlt().solve(XtW * t);
            if (!bn.allFinite()) break;
            const bool done = (bn - b).norm() <= 1e-8 * (1.0 + b.norm());
            b = bn;
            if (done) break;
        }
        return b;
    }

public:
    /** @brief Enable/disable multi-start initialization for the skew-t family (enabled by default). */
    void set_skewt_multistart(bool on) { skewt_multistart_ = on; }

private:
    /**
     * @brief Optimization dispatch wrapper executing L-BFGS-B, BFGS, or Nelder-Mead on objective loss function.
     */
    template <typename LossFunc>
    std::pair<MleResult, Eigen::VectorXd> run_optimization(const LossFunc& loss_fn,
                                                           const Eigen::VectorXd& x0,
                                                           [[maybe_unused]] int total_params,
                                                           int maxiter,
                                                           double ftol,
                                                           double gtol,
                                                           int maxcor,
                                                           int verbose)
    {
        MleResult res;

        // Penalty value for non-finite f. It must be large but moderate: with max()/1e6 (~1e302) the
        // cubic/quadratic interpolation of the line search overflows (inf/NaN) and sends the parameters
        // (e.g. log sigma) to absurd values.
        const double huge = 1e20;

        // Best point actually evaluated: guards against optimizers returning an x that does not
        // match the reported f (the last trial point instead of the accepted one).
        Eigen::VectorXd best_x;
        double best_f = std::numeric_limits<double>::infinity();
        auto f = [&](const Eigen::VectorXd& x) -> double {
            const double v  = loss_fn(x);
            const double vv = std::isfinite(v) ? v : huge;
            if (vv < best_f) { best_f = vv; best_x = x; }
            return vv;
        };
        auto jac = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd g = loss_fn.gradient(x);
            if (!g.allFinite()) g.setZero();
            return g;
        };

        const min::VerbosityLevel vlevel =
            (verbose > 0) ? min::VerbosityLevel::Iteration : min::VerbosityLevel::Silent;

        double fun = 0.0;

        switch (minimizer_) {
            case Minimizer::BFGS: {
                min::BfgsOptions<double> opt;
                opt.maxiter = maxiter;
                opt.ftol    = ftol;
                opt.gtol    = gtol;
                opt.verbose = vlevel;

                auto r = min::bfgs<double>(f, x0, opt, jac);
                res.success               = r.success;
                res.message               = r.message;
                res.iterations            = r.nit;
                res.function_evaluations  = r.nfev;
                fun                       = r.fun;
                res.x_opt                 = r.x;
                break;
            }
            case Minimizer::NelderMead: {
                min::NelderMeadOptions<double> opt;
                opt.maxiter = maxiter;
                opt.maxfev  = maxiter * 50;
                opt.ftol    = ftol;
                opt.xtol    = std::max(ftol, 1e-6);
                opt.adaptive = true;  // preserve previous behavior (the default is now false, as in scipy)
                opt.verbose = vlevel;

                auto r = min::nelder_mead<double>(f, x0, opt);
                res.success               = r.success;
                res.message               = r.message;
                res.iterations            = r.nit;
                res.function_evaluations  = r.nfev;
                fun                       = r.fun;
                res.x_opt                 = r.x;
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
                res.success               = r.success;
                res.message               = r.message;
                res.iterations            = r.nit;
                res.function_evaluations  = r.nfev;
                fun                       = r.fun;
                res.x_opt                 = r.x;
                break;
            }
        }

        // (x, f) consistency: re-evaluate f at the returned x; if it is worse than the best point
        // visited, the optimizer returned an x that does not match its f -> use the best point.
        if (best_x.size() == x0.size() && res.x_opt.size() == x0.size()) {
            const double f_ret = f(res.x_opt);
            if (!std::isfinite(f_ret) || f_ret > best_f + 1e-8 * (1.0 + std::abs(best_f))) {
                res.x_opt = best_x;
                fun       = best_f;
                res.message += " [x corrected: the optimizer returned a point worse than the best visited]";
            } else {
                fun = f_ret;
            }
        }

        if (!std::isfinite(fun)) {
            res.success = false;
            res.message = "Numerical failure: Loss function returned non-finite value (NaN/Inf).";
        }

        res.log_likelihood = -fun;
        Eigen::VectorXd x_final = res.x_opt;
        return std::make_pair(std::move(res), std::move(x_final));
    }

public:
    /**
     * @brief Constructs a new MleSolver instance.
     * @param fit_intercept Automatically prepend intercept column if true.
     * @param minimizer Selected minimizer choice (default = LBFGSB).
     */
    explicit MleSolver(bool fit_intercept = true, Minimizer minimizer = Minimizer::LBFGSB)
        : fit_intercept_(fit_intercept), minimizer_(minimizer) {}

    /**
     * @brief True if the family models a second parameter through its own linear predictor and link
     *        (i.e. `fit(..., secondary_link, ...)` has an effect for it).
     */
    static bool supports_secondary_link(FamilyCode family) {
        return secondary_kind(family) != SecondaryKind::None;
    }

    /**
     * @brief Default link of the secondary parameter (Logit for probabilities, Log for positive parameters),
     *        or `std::nullopt` if the family has no second linear predictor.
     */
    static std::optional<LinkType> default_secondary_link(FamilyCode family) {
        switch (secondary_kind(family)) {
            case SecondaryKind::Probability: return LinkType::Logit;
            case SecondaryKind::Positive:    return LinkType::Log;
            case SecondaryKind::None:        break;
        }
        return std::nullopt;
    }

    /** @brief Sets optimization minimizer algorithm choice. */
    void set_minimizer(Minimizer m) { minimizer_ = m; }
    /** @brief Gets current optimization minimizer backend. */
    Minimizer get_minimizer() const { return minimizer_; }

    /**
     * @brief Fits a parametric model using Maximum Likelihood Estimation.
     * @param family FamilyCode identifier.
     * @param X_raw Raw design matrix ($n \times k$).
     * @param y Response vector ($n \times 1$).
     * @param d Optional event indicator / censoring vector.
     * @param Z Optional design matrix for zero-inflation or precision component.
     * @param left_limit Optional left truncation bound.
     * @param right_limit Optional right truncation bound.
     * @param maxiter Maximum optimizer iterations.
     * @param ftol Function change tolerance.
     * @param gtol Gradient norm tolerance.
     * @param maxcor Memory corrections for L-BFGS-B.
     * @param verbose Verbosity level.
     * @param compute_se Compute asymptotic covariance and standard errors if true.
     * @param secondary_link Optional link of the parameter modelled by `Z` (was `infl_link`; same position).
     *        Probability parameter (ZIP, ZINB, Hurdle*): logit (default), probit, cloglog, loglog, cauchit.
     *        Positive parameter (Beta precision, Simplex dispersion): log (default), identity, sqrt, inverse.
     *        Ignored by families without a second linear predictor; throws `std::invalid_argument` if the
     *        link is not valid for the family's secondary parameter.
     * @param use_left Flag indicating if left truncation bound is active.
     * @param use_right Flag indicating if right truncation bound is active.
     * @return `MleResult` object populated with estimated parameters and log-likelihood.
     */
    MleResult fit(FamilyCode family,
                  const Eigen::MatrixXd& X_raw,
                  const Eigen::VectorXd& y,
                  std::optional<Eigen::VectorXd> d = std::nullopt,
                  std::optional<Eigen::MatrixXd> Z = std::nullopt,
                  std::optional<double> left_limit = std::nullopt,
                  std::optional<double> right_limit = std::nullopt,
                  int maxiter = 1000,
                  double ftol = 1e-12,
                  double gtol = 1e-6,
                  int maxcor = 10,
                  int verbose = 0,
                  bool compute_se = true,
                  std::optional<LinkType> secondary_link = std::nullopt,
                  bool use_left = false, bool use_right = false)
    {
        LinkType link = resolve_link(family);
        const LinkType sec_link = resolve_secondary_link(family, secondary_link, "MleSolver::fit");

        check_finite(X_raw, "X");
        check_finite(y, "y");
        check_dims(X_raw.rows(), y.size(), "X", "y", "MleSolver::fit");
        check_opt_dims(d, X_raw.rows(), "d", "MleSolver::fit");
        check_finite_opt(d, "d");
        check_opt_dims(Z, X_raw.rows(), "Z", "MleSolver::fit");
        check_finite_opt(Z, "Z");

        Eigen::MatrixXd X = fit_intercept_ ? add_intercept(X_raw) : X_raw;
        const long k = X.cols();
        const long n = X.rows();

        check_full_rank(X, "MleSolver::fit");
        check_identifiable(n, k, "MleSolver::fit");

        switch (family) {

            // 1. LOG-NORMAL AFT
            case FAM_LOGNORMAL_IDENTITY:
            case FAM_LOGNORMAL_LOG:
            case FAM_LOGNORMAL_INVERSE: {
                check_positive(y, "y", "MleSolver::fit (LogNormal)");
                const int total_params = static_cast<int>(k + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
                Eigen::VectorXd cens_d = d.value_or(Eigen::VectorXd::Ones(n));
                check_binary_indicator(cens_d, "d", "MleSolver::fit (LogNormal)");

                Eigen::VectorXd log_y = y.array().cwiseMax(1e-5).log();
                Eigen::VectorXd mu_target = log_y;
                for (long i = 0; i < n; ++i) {
                    if (cens_d(i) < 0.5) mu_target(i) += 0.12 * std::max(1.0, std::abs(mu_target(i)));
                }
                Eigen::VectorXd eta_target = transform_link_forward(mu_target, link);

                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd log_mu_hat = evaluate_inverse_link(X * x0.head(k), link);
                Eigen::VectorXd residuals = log_y - log_mu_hat;
                double variance_res = (residuals.array() - residuals.mean()).square().sum() / static_cast<double>(std::max<long>(n - k, 1L));
                double initial_sigma = std::sqrt(std::max(variance_res, 0.05));

                x0(k) = std::log(std::max(1e-4, initial_sigma));

                LogNormalLoss loss_fn(X, y, cens_d, link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 2. WEIBULL AFT
            case FAM_WEIBULL_IDENTITY:
            case FAM_WEIBULL_LOG: {
                check_positive(y, "y", "MleSolver::fit (Weibull)");
                const int total_params = static_cast<int>(k + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
                Eigen::VectorXd cens_d = d.value_or(Eigen::VectorXd::Ones(n));
                check_binary_indicator(cens_d, "d", "MleSolver::fit (Weibull)");

                Eigen::VectorXd log_y = y.array().cwiseMax(1e-5).log();
                Eigen::VectorXd mu_target = log_y;
                for (long i = 0; i < n; ++i) {
                    if (cens_d(i) < 0.5) mu_target(i) += 0.12 * std::max(1.0, std::abs(mu_target(i)));
                }
                Eigen::VectorXd eta_target = transform_link_forward(mu_target, link);

                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd log_mu_hat = evaluate_inverse_link(X * x0.head(k), link);
                Eigen::VectorXd residuals = log_y - log_mu_hat;
                double variance_res = (residuals.array() - residuals.mean()).square().sum() / static_cast<double>(std::max<long>(n - k, 1L));
                double initial_sigma = std::sqrt(std::max(variance_res, 0.05)) * 0.7796968012336783;

                x0(k) = std::log(std::max(1e-4, initial_sigma));

                WeibullTLoss loss_fn(X, y, cens_d, link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 3. LOG-LOGISTIC AFT
            case FAM_LOGLOGISTIC_IDENTITY:
            case FAM_LOGLOGISTIC_LOG: {
                check_positive(y, "y", "MleSolver::fit (LogLogistic)");
                const int total_params = static_cast<int>(k + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
                Eigen::VectorXd cens_d = d.value_or(Eigen::VectorXd::Ones(n));
                check_binary_indicator(cens_d, "d", "MleSolver::fit (LogLogistic)");

                Eigen::VectorXd log_y = y.array().cwiseMax(1e-5).log();
                Eigen::VectorXd mu_target = log_y;
                for (long i = 0; i < n; ++i) {
                    if (cens_d(i) < 0.5) mu_target(i) += 0.12 * std::max(1.0, std::abs(mu_target(i)));
                }
                Eigen::VectorXd eta_target = transform_link_forward(mu_target, link);

                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd log_mu_hat = evaluate_inverse_link(X * x0.head(k), link);
                Eigen::VectorXd residuals = log_y - log_mu_hat;
                double variance_res = (residuals.array() - residuals.mean()).square().sum() / static_cast<double>(std::max<long>(n - k, 1L));
                double initial_sigma = std::sqrt(std::max(variance_res, 0.05)) * 0.5513288954217921;
                x0(k) = std::log(std::max(1e-4, initial_sigma));

                LogLogisticTLoss loss_fn(X, y, cens_d, link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 3b. GUMBEL
            case FAM_GUMBEL_IDENTITY:
            case FAM_GUMBEL_LOG: {
                const int total_params = static_cast<int>(k + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double mean_y = y.mean();
                double variance_y = (y.array() - mean_y).square().sum() / static_cast<double>(std::max<long>(n - 1, 1));

                double initial_sigma = std::sqrt(std::max(variance_y, 0.1) * 6.0 / (glm_PI * glm_PI));
                constexpr double euler_gamma = 0.5772156649015329;
                Eigen::VectorXd mu_target = y.array() - initial_sigma * euler_gamma;
                Eigen::VectorXd eta_target = transform_link_forward(mu_target, link);

                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }
                x0(k) = std::log(std::max(1e-4, initial_sigma));

                GumbelLoss loss_fn(X, y, link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 4. ZERO-INFLATED POISSON (ZIP)
            case FAM_ZIP_LOG:
            case FAM_ZIP_IDENTITY:
            case FAM_ZIP_SQRT:
            case FAM_ZIP_INVERSE: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const LinkType actual_infl_link = sec_link;
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double init_gamma0 = compute_single_link_forward(zero_prop, actual_infl_link);
                double mean_pos = 0.0;
                long pos_cnt = 0;
                for (long i = 0; i < y.size(); ++i) {
                    if (y(i) > 0.0) {
                        mean_pos += y(i);
                        pos_cnt++;
                    }
                }
                double avg_pos = (pos_cnt > 0) ? (mean_pos / static_cast<double>(pos_cnt)) : 1.0;
                double init_beta0 = compute_single_link_forward(avg_pos, link);

                if (fit_intercept_ && kc > 0 && kz > 0) {
                    x0(0) = init_beta0;
                    x0(kc) = init_gamma0;
                }

                {
                    Eigen::VectorXd beta0 = x0.head(kc);
                    Eigen::VectorXd gamma0 = x0.tail(kz);
                    zip_em_warmstart(X, Z_mat, y, link, beta0, gamma0, actual_infl_link);
                    x0.head(kc) = beta0;
                    x0.tail(kz) = gamma0;
                }

                ZeroInflatedPoissonLoss loss_fn(X, Z_mat, y, link, actual_infl_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.tail(kz);
                return std::move(res);
            }

            // 5. ZERO-INFLATED NEGATIVE BINOMIAL (ZINB)
            case FAM_ZINB_LOG:
            case FAM_ZINB_SQRT:
            case FAM_ZINB_IDENTITY:
            case FAM_ZINB_INVERSE: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const LinkType actual_infl_link = sec_link;
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double init_gamma0 = compute_single_link_forward(zero_prop, actual_infl_link);

                double mean_pos = 0.0;
                long pos_cnt = 0;
                for (long i = 0; i < y.size(); ++i) {
                    if (y(i) > 0.0) {
                        mean_pos += y(i);
                        pos_cnt++;
                    }
                }

                double init_beta0 = 1.0;
                if (pos_cnt > 0) {
                    double avg_pos = mean_pos / static_cast<double>(pos_cnt);
                    double sq_sum = 0.0;
                    for (long i = 0; i < y.size(); ++i) {
                        if (y(i) > 0.0) sq_sum += (y(i) - avg_pos) * (y(i) - avg_pos);
                    }
                    double variance_pos = sq_sum / static_cast<double>(std::max(1L, pos_cnt - 1));

                    init_beta0 = compute_single_link_forward(avg_pos, link);

                    if (variance_pos > avg_pos) {
                        double initial_alpha = (variance_pos - avg_pos) / (avg_pos * avg_pos);
                        x0(kc + kz) = std::log(std::clamp(initial_alpha, 0.05, 5.0));
                    } else {
                        x0(kc + kz) = std::log(0.15);
                    }
                } else {
                    x0(kc + kz) = std::log(0.5);
                }

                if (fit_intercept_ && kc > 0 && kz > 0) {
                    x0(0) = init_beta0;
                    x0(kc) = init_gamma0;
                }

                {
                    Eigen::VectorXd beta0 = x0.head(kc);
                    Eigen::VectorXd gamma0 = x0.segment(kc, kz);
                    const double alpha0 = std::exp(x0(kc + kz));
                    zinb_em_warmstart(X, Z_mat, y, link, alpha0, beta0, gamma0, actual_infl_link);
                    x0.head(kc) = beta0;
                    x0.segment(kc, kz) = gamma0;
                }

                ZeroInflatedNBLoss loss_fn(X, Z_mat, y, link, VarianceForm::Quadratic, actual_infl_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.segment(kc, kz);
                res.extra_param = std::exp(x_opt(kc + kz));
                return std::move(res);
            }

            // 6. HURDLE POISSON
            case FAM_HURDLE_POISSON_LOG:
            case FAM_HURDLE_POISSON_SQRT: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double positive_prop = 1.0 - zero_prop;
                double init_gamma0 = compute_single_link_forward(positive_prop, sec_link);

                LinkType count_link = resolve_link(family);

                std::vector<long> pos_idx;
                for (long i = 0; i < n; ++i) { if (y(i) > 0.0) pos_idx.push_back(i); }

                if (!pos_idx.empty()) {
                    const long n_pos = static_cast<long>(pos_idx.size());
                    Eigen::MatrixXd X_pos(n_pos, kc);
                    Eigen::VectorXd y_pos(n_pos);
                    for (long i = 0; i < n_pos; ++i) {
                        X_pos.row(i) = X.row(pos_idx[static_cast<size_t>(i)]);
                        y_pos(i) = y(pos_idx[static_cast<size_t>(i)]);
                    }
                    Eigen::VectorXd eta_pos = transform_link_forward(y_pos, count_link);
                    Eigen::VectorXd beta_init = (X_pos.transpose() * X_pos).ldlt().solve(X_pos.transpose() * eta_pos);
                    if (beta_init.allFinite()) {
                        x0.head(kc) = beta_init;
                    } else if (fit_intercept_ && kc > 0) {
                        x0(0) = eta_pos.mean();
                    }
                }

                if (fit_intercept_ && kz > 0) {
                    x0(kc) = init_gamma0;
                }

                HurdlePoissonLoss loss_fn(X, Z_mat, y, count_link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.tail(kz);
                return std::move(res);
            }

            // 7. HURDLE NEGATIVE BINOMIAL
            case FAM_HURDLE_NB_LOG:
            case FAM_HURDLE_NB_SQRT: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double positive_prop = 1.0 - zero_prop;
                double init_gamma0 = compute_single_link_forward(positive_prop, sec_link);

                LinkType count_link = resolve_link(family);

                double mean_pos = 0.0;
                long pos_cnt = 0;
                std::vector<long> pos_idx;
                for (long i = 0; i < n; ++i) {
                    if (y(i) > 0.0) {
                        mean_pos += y(i);
                        pos_cnt++;
                        pos_idx.push_back(i);
                    }
                }

                if (!pos_idx.empty()) {
                    const long n_pos = static_cast<long>(pos_idx.size());
                    Eigen::MatrixXd X_pos(n_pos, kc);
                    Eigen::VectorXd y_pos(n_pos);
                    for (long i = 0; i < n_pos; ++i) {
                        X_pos.row(i) = X.row(pos_idx[static_cast<size_t>(i)]);
                        y_pos(i) = y(pos_idx[static_cast<size_t>(i)]);
                    }
                    Eigen::VectorXd eta_pos = transform_link_forward(y_pos, count_link);
                    Eigen::VectorXd beta_init = (X_pos.transpose() * X_pos).ldlt().solve(X_pos.transpose() * eta_pos);
                    if (beta_init.allFinite()) {
                        x0.head(kc) = beta_init;
                    } else if (fit_intercept_ && kc > 0) {
                        x0(0) = eta_pos.mean();
                    }

                    if (pos_cnt > 1) {
                        double avg_pos = mean_pos / static_cast<double>(pos_cnt);
                        double sq_sum = 0.0;
                        for (long i = 0; i < n; ++i) {
                            if (y(i) > 0.0) sq_sum += (y(i) - avg_pos) * (y(i) - avg_pos);
                        }
                        double variance_pos = sq_sum / static_cast<double>(pos_cnt - 1);
                        if (variance_pos > avg_pos) {
                            double initial_alpha = (variance_pos - avg_pos) / (avg_pos * avg_pos);
                            x0(kc + kz) = std::log(std::clamp(initial_alpha, 0.05, 5.0));
                        } else {
                            x0(kc + kz) = std::log(0.1);
                        }
                    } else {
                        x0(kc + kz) = std::log(0.5);
                    }
                }

                if (fit_intercept_ && kz > 0) {
                    x0(kc) = init_gamma0;
                }

                HurdleNBLoss loss_fn(X, Z_mat, y, count_link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.segment(kc, kz);
                res.extra_param = std::exp(x_opt(kc + kz));
                return std::move(res);
            }

            // 8. BETA REGRESSION
            case FAM_BETA_LOGIT:
            case FAM_BETA_PROBIT:
            case FAM_BETA_CLOGLOG:
            case FAM_BETA_LOGLOG:
            case FAM_BETA_CAUCHIT: {
                Eigen::MatrixXd Z_mat = Z.value_or(Eigen::MatrixXd::Ones(n, 1));
                const long k_mu = k;
                const long k_phi = Z_mat.cols();
                const int total_params = static_cast<int>(k_mu + k_phi);

                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                Eigen::VectorXd y_clamp = y.array().cwiseMax(1e-4).cwiseMin(1.0 - 1e-4);
                Eigen::VectorXd eta_target = transform_link_forward(y_clamp, link);

                x0.head(k_mu) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k_mu).allFinite()) {
                    x0.head(k_mu).setZero();
                    if (fit_intercept_ && k_mu > 0) x0(0) = eta_target.mean();
                }

                double mean_y = std::clamp(y.mean(), 0.01, 0.99);
                double var_y = (y.array() - mean_y).square().sum() / static_cast<double>(std::max<long>(n - 1, 1));
                var_y = std::max(1e-4, std::min(var_y, mean_y * (1.0 - mean_y) - 1e-4));

                double initial_phi = (mean_y * (1.0 - mean_y) / var_y) - 1.0;
                initial_phi = std::clamp(initial_phi, 0.5, 200.0);

                x0(k_mu) = compute_positive_link_forward(initial_phi, sec_link);

                BetaRegressionLoss loss_fn(X, Z_mat, y, link, sec_link, PrecisionForm::Standard);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k_mu);
                res.gamma = x_opt.segment(k_mu, k_phi);
                res.extra_param = evaluate_positive_link(x_opt(k_mu), sec_link);   // phi (intercept-only Z); = exp(.) for Log
                return std::move(res);
            }

            // 9. HURDLE GAMMA
            case FAM_HURDLE_GAMMA_LOG:
            case FAM_HURDLE_GAMMA_INVERSE: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double positive_prop = 1.0 - zero_prop;
                double init_gamma0 = compute_single_link_forward(positive_prop, sec_link);

                LinkType mean_link = resolve_link(family);

                double mean_pos = 0.0;
                long pos_cnt = 0;
                std::vector<long> pos_idx;
                for (long i = 0; i < n; ++i) {
                    if (y(i) > 0.0) { mean_pos += y(i); pos_cnt++; pos_idx.push_back(i); }
                }

                double init_shape = 1.0;
                if (!pos_idx.empty()) {
                    const long n_pos = static_cast<long>(pos_idx.size());
                    Eigen::MatrixXd X_pos(n_pos, kc);
                    Eigen::VectorXd y_pos(n_pos);
                    for (long i = 0; i < n_pos; ++i) {
                        X_pos.row(i) = X.row(pos_idx[static_cast<size_t>(i)]);
                        y_pos(i) = y(pos_idx[static_cast<size_t>(i)]);
                    }
                    Eigen::VectorXd eta_pos = transform_link_forward(y_pos, mean_link);
                    Eigen::VectorXd beta_init = (X_pos.transpose() * X_pos).ldlt().solve(X_pos.transpose() * eta_pos);
                    if (beta_init.allFinite()) {
                        x0.head(kc) = beta_init;
                    } else if (fit_intercept_ && kc > 0) {
                        x0(0) = eta_pos.mean();
                    }

                    if (pos_cnt > 1) {
                        double avg_pos = mean_pos / static_cast<double>(pos_cnt);
                        double sq_sum = 0.0;
                        for (long i = 0; i < n; ++i) {
                            if (y(i) > 0.0) sq_sum += (y(i) - avg_pos) * (y(i) - avg_pos);
                        }
                        double variance_pos = sq_sum / static_cast<double>(pos_cnt - 1);
                        if (variance_pos > 1e-8) {
                            init_shape = std::clamp((avg_pos * avg_pos) / variance_pos, 0.05, 50.0);
                        }
                    }
                }

                if (fit_intercept_ && kz > 0) {
                    x0(kc) = init_gamma0;
                }
                x0(kc + kz) = std::log(init_shape);

                HurdleGammaLoss loss_fn(X, Z_mat, y, mean_link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.segment(kc, kz);
                res.extra_param = std::exp(x_opt(kc + kz));
                return std::move(res);
            }

            // 10. HURDLE LOGNORMAL
            case FAM_HURDLE_LOGNORMAL_LOG:
            case FAM_HURDLE_LOGNORMAL_IDENTITY: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double positive_prop = 1.0 - zero_prop;
                double init_gamma0 = compute_single_link_forward(positive_prop, sec_link);

                LinkType mean_link = resolve_link(family);

                double mean_log_pos = 0.0;
                long pos_cnt = 0;
                std::vector<long> pos_idx;
                for (long i = 0; i < n; ++i) {
                    if (y(i) > 0.0) { mean_log_pos += std::log(y(i)); pos_cnt++; pos_idx.push_back(i); }
                }

                double init_sigma = 1.0;
                if (!pos_idx.empty()) {
                    const long n_pos = static_cast<long>(pos_idx.size());
                    Eigen::MatrixXd X_pos(n_pos, kc);
                    Eigen::VectorXd log_y_pos(n_pos);
                    for (long i = 0; i < n_pos; ++i) {
                        X_pos.row(i) = X.row(pos_idx[static_cast<size_t>(i)]);
                        log_y_pos(i) = std::log(y(pos_idx[static_cast<size_t>(i)]));
                    }
                    Eigen::VectorXd eta_pos = transform_link_forward(log_y_pos, mean_link);
                    Eigen::VectorXd beta_init = (X_pos.transpose() * X_pos).ldlt().solve(X_pos.transpose() * eta_pos);
                    if (beta_init.allFinite()) {
                        x0.head(kc) = beta_init;
                    } else if (fit_intercept_ && kc > 0) {
                        x0(0) = eta_pos.mean();
                    }

                    if (pos_cnt > 1) {
                        double avg_log_pos = mean_log_pos / static_cast<double>(pos_cnt);
                        double sq_sum = 0.0;
                        for (long i = 0; i < n; ++i) {
                            if (y(i) > 0.0) {
                                double diff = std::log(y(i)) - avg_log_pos;
                                sq_sum += diff * diff;
                            }
                        }
                        init_sigma = std::sqrt(std::max(sq_sum / static_cast<double>(pos_cnt - 1), 0.05));
                    }
                }

                if (fit_intercept_ && kz > 0) {
                    x0(kc) = init_gamma0;
                }
                x0(kc + kz) = std::log(init_sigma);

                HurdleLogNormalLoss loss_fn(X, Z_mat, y, mean_link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.segment(kc, kz);
                res.extra_param = std::exp(x_opt(kc + kz));
                return std::move(res);
            }

            // 11. HURDLE BETA
            case FAM_HURDLE_BETA_LOGIT:
            case FAM_HURDLE_BETA_PROBIT:
            case FAM_HURDLE_BETA_LOGLOG:
            case FAM_HURDLE_BETA_CLOGLOG:
            case FAM_HURDLE_BETA_CAUCHIT: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double positive_prop = 1.0 - zero_prop;
                double init_gamma0 = compute_single_link_forward(positive_prop, sec_link);

                LinkType mean_link = resolve_link(family);

                double mean_pos = 0.0;
                long pos_cnt = 0;
                std::vector<long> pos_idx;
                for (long i = 0; i < n; ++i) {
                    if (y(i) > 0.0) { mean_pos += y(i); pos_cnt++; pos_idx.push_back(i); }
                }

                double init_phi = 10.0;
                if (!pos_idx.empty()) {
                    const long n_pos = static_cast<long>(pos_idx.size());
                    Eigen::MatrixXd X_pos(n_pos, kc);
                    Eigen::VectorXd y_pos_clamped(n_pos);
                    for (long i = 0; i < n_pos; ++i) {
                        X_pos.row(i) = X.row(pos_idx[static_cast<size_t>(i)]);
                        y_pos_clamped(i) = std::clamp(y(pos_idx[static_cast<size_t>(i)]), 1e-4, 1.0 - 1e-4);
                    }
                    Eigen::VectorXd eta_pos = transform_link_forward(y_pos_clamped, mean_link);
                    Eigen::VectorXd beta_init = (X_pos.transpose() * X_pos).ldlt().solve(X_pos.transpose() * eta_pos);
                    if (beta_init.allFinite()) {
                        x0.head(kc) = beta_init;
                    } else if (fit_intercept_ && kc > 0) {
                        x0(0) = eta_pos.mean();
                    }

                    if (pos_cnt > 1) {
                        double avg_pos = std::clamp(mean_pos / static_cast<double>(pos_cnt), 1e-4, 1.0 - 1e-4);
                        double sq_sum = 0.0;
                        for (long i = 0; i < n; ++i) {
                            if (y(i) > 0.0) sq_sum += (y(i) - avg_pos) * (y(i) - avg_pos);
                        }
                        double variance_pos = sq_sum / static_cast<double>(pos_cnt - 1);
                        if (variance_pos > 1e-8) {
                            init_phi = std::clamp(avg_pos * (1.0 - avg_pos) / variance_pos - 1.0, 0.5, 200.0);
                        }
                    }
                }

                if (fit_intercept_ && kz > 0) {
                    x0(kc) = init_gamma0;
                }
                x0(kc + kz) = std::log(init_phi);

                HurdleBetaLoss loss_fn(X, Z_mat, y, mean_link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.segment(kc, kz);
                res.extra_param = std::exp(x_opt(kc + kz));
                return std::move(res);
            }


            case FAM_HURDLE_GEOMETRIC_LOG:
            case FAM_HURDLE_GEOMETRIC_IDENTITY:
            case FAM_HURDLE_GEOMETRIC_SQRT: {
                Eigen::MatrixXd Z_mat = Z.value_or(X);
                const long kc = X.cols();
                const long kz = Z_mat.cols();
                const int total_params = static_cast<int>(kc + kz);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                // 1. Initialize the barrier P(Y > 0)
                double zero_prop = std::clamp((y.array() == 0.0).cast<double>().mean(), 0.05, 0.95);
                double positive_prop = 1.0 - zero_prop;
                double init_gamma0 = compute_single_link_forward(positive_prop, sec_link);

                LinkType count_link = resolve_link(family);

                std::vector<long> pos_idx;
                for (long i = 0; i < n; ++i) {
                    if (y(i) > 0.0) pos_idx.push_back(i);
                }

                if (!pos_idx.empty()) {
                    const long n_pos = static_cast<long>(pos_idx.size());
                    Eigen::MatrixXd X_pos(n_pos, kc);
                    Eigen::VectorXd y_pos_adj(n_pos);

                    for (long i = 0; i < n_pos; ++i) {
                        X_pos.row(i) = X.row(pos_idx[static_cast<size_t>(i)]);
                        // Crucial adjustment: for the truncated geometric model, shift y_i by -1
                        // (with a floor value for logarithm stability)
                        y_pos_adj(i) = std::max(y(pos_idx[static_cast<size_t>(i)]) - 1.0, 0.1);
                    }

                    Eigen::VectorXd eta_pos = transform_link_forward(y_pos_adj, count_link);
                    Eigen::VectorXd beta_init = (X_pos.transpose() * X_pos).ldlt().solve(X_pos.transpose() * eta_pos);

                    if (beta_init.allFinite()) {
                        x0.head(kc) = beta_init;
                    } else if (fit_intercept_ && kc > 0) {
                        x0(0) = eta_pos.mean();
                    }
                }

                if (fit_intercept_ && kz > 0) {
                    x0(kc) = init_gamma0;
                }

                HurdleGeometricLoss loss_fn(X, Z_mat, y, count_link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(kc);
                res.gamma = x_opt.tail(kz);
                return std::move(res);
            }


            case FAM_TRUNC_GAUSSIAN_IDENTITY:
            case FAM_TRUNC_GAUSSIAN_LOG:
            case FAM_TRUNC_GAUSSIAN_INVERSE: {
                double left  = left_limit.value_or(0.0);
                double right = right_limit.value_or(-1.0);
                check_range(left, right, use_left, use_right, "MleSolver::fit (TruncatedGaussian)");

                const int total_params = static_cast<int>(k + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                LinkType final_link = (family == FAM_TRUNC_GAUSSIAN_IDENTITY) ? LinkType::Identity :
                                     (family == FAM_TRUNC_GAUSSIAN_LOG)      ? LinkType::Log      :
                                                                                LinkType::Inverse;

                Eigen::VectorXd eta_init(n);
                for (long i = 0; i < n; ++i) {
                    double yi = y(i);
                    switch (final_link) {
                        case LinkType::Identity:
                            eta_init(i) = yi;
                            break;
                        case LinkType::Log:
                            eta_init(i) = std::log(std::max(yi, 1e-5));
                            break;
                        case LinkType::Inverse:
                            eta_init(i) = 1.0 / (std::abs(yi) > 1e-7 ? yi : (yi >= 0 ? 1e-7 : -1e-7));
                            break;
                        default:
                            eta_init(i) = yi;
                    }
                }

                {
                    auto sorted = eta_init;
                    std::sort(sorted.data(), sorted.data() + n);
                    double lo = sorted(static_cast<long>(0.05 * static_cast<double>(n)));
                    double hi = sorted(static_cast<long>(0.95 * static_cast<double>(n)));
                    for (long i = 0; i < n; ++i)
                        eta_init(i) = std::clamp(eta_init(i), lo, hi);
                }

                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_init);

                Eigen::VectorXd residuals = eta_init - (X * x0.head(k));
                double var_res = residuals.array().square().sum()
                                 / static_cast<double>(std::max(n - k, 1L));
                double init_sigma = std::sqrt(std::max(var_res, 1e-4));
                x0(k) = std::log(init_sigma);

                TruncatedGaussianLoss loss_fn(X, y, final_link, left, right,
                                              use_left, use_right);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params,
                                                     maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta        = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 12. TRUNCATED POISSON
            case FAM_TRUNC_POISSON_LOG:
            case FAM_TRUNC_POISSON_IDENTITY: {
                long left = static_cast<long>(left_limit.value_or(0));
                long right = static_cast<long>(right_limit.value_or(-1));
                check_count_range(left, right, "MleSolver::fit (TruncatedPoisson)");
                const int total_params = static_cast<int>(k);

                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
                LinkType final_link = (family == FAM_TRUNC_POISSON_IDENTITY) ? LinkType::Identity : LinkType::Log;

                double mean_y = std::max(y.mean(), 0.1);
                double init_beta0 = (final_link == LinkType::Identity) ? mean_y : std::log(mean_y);

                if (k > 0) {
                    x0(0) = init_beta0;
                }

                TruncatedPoissonLoss loss_fn(X, y, left, right, final_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt;
                return std::move(res);
            }

            // 13. TRUNCATED NEGATIVE BINOMIAL
            case FAM_TRUNC_NB_LOG:
            case FAM_TRUNC_NB_IDENTITY: {
                long left = static_cast<long>(left_limit.value_or(0));
                long right = static_cast<long>(right_limit.value_or(-1));
                check_count_range(left, right, "MleSolver::fit (TruncatedNB)");
                const int total_params = static_cast<int>(k + 1);

                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
                LinkType final_link = (family == FAM_TRUNC_NB_IDENTITY) ? LinkType::Identity : LinkType::Log;

                double mean_y = std::max(y.mean(), 0.1);
                double init_beta0 = (final_link == LinkType::Identity) ? mean_y : std::log(mean_y);

                if (k > 0) {
                    x0(0) = init_beta0;
                }

                x0(k) = std::log(0.5);

                TruncatedNBLoss loss_fn(X, y, left, right, final_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 14. TRUNCATED GEOMETRIC
            case FAM_TRUNC_GEOMETRIC_LOG: {
                long left = static_cast<long>(left_limit.value_or(0));
                long right = static_cast<long>(right_limit.value_or(-1));
                check_count_range(left, right, "MleSolver::fit (TruncatedGeometric)");
                const int total_params = static_cast<int>(k);

                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
                double mean_y = y.mean();
                if (fit_intercept_ && k > 0) x0(0) = std::log(std::max(mean_y, 0.1));

                TruncatedGeometricLoss loss_fn(X, y, left, right, link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt;
                return std::move(res);
            }

            // 15. STUDENT-T
            case FAM_STUDENT_T_LOG:
            case FAM_STUDENT_T_IDENTITY: {
                const int total_params = static_cast<int>(k + 2);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                LinkType final_link = resolve_link(family);
                Eigen::VectorXd eta_target = transform_link_forward(y, final_link);
                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd mu_hat = evaluate_inverse_link(X * x0.head(k), final_link);
                Eigen::VectorXd residuals = y - mu_hat;
                double var_res = residuals.array().square().sum() / static_cast<double>(std::max<long>(n - k, 1L));
                double initial_sigma = std::sqrt(std::max(var_res, 0.05));

                x0(k) = std::log(initial_sigma);

                x0(k + 1) = std::log(4.0);

                StudentTLoss loss_fn(X, y, final_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.gamma = x_opt.segment(k, 2);

                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }


            // 16. SKEW-NORMAL
            case FAM_SKEW_NORMAL_IDENTITY:
            case FAM_SKEW_NORMAL_LOG: {
                const int total_params = static_cast<int>(k + 2);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                LinkType final_link = resolve_link(family);
                Eigen::VectorXd eta_target = transform_link_forward(y, final_link);
                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd mu_hat = evaluate_inverse_link(X * x0.head(k), final_link);
                Eigen::VectorXd residuals = y - mu_hat;
                double var_res = residuals.array().square().sum() / static_cast<double>(std::max<long>(n - k, 1L));
                double initial_sigma = std::sqrt(std::max(var_res, 0.1));

                x0(k) = std::log(initial_sigma);

                double mean_res = residuals.mean();
                double cube_sum = 0.0;
                double sq_sum = 0.0;
                for (long i = 0; i < n; ++i) {
                    double dev = residuals(i) - mean_res;
                    cube_sum += dev * dev * dev;
                    sq_sum += dev * dev;
                }
                double skewness = (cube_sum / static_cast<double>(n)) / std::pow(std::max(1e-6, sq_sum / static_cast<double>(n)), 1.5);

                x0(k + 1) = (skewness >= 0.0) ? 0.5 : -0.5;

                SkewNormalLoss loss_fn(X, y, final_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.gamma = x_opt.segment(k, 2);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // 17. SKEW-T
            case FAM_SKEW_T_IDENTITY:
            case FAM_SKEW_T_LOG: {
                const int total_params = static_cast<int>(k + 3);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                LinkType final_link = resolve_link(family);
                Eigen::VectorXd eta_target = transform_link_forward(y, final_link);
                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd mu_hat = evaluate_inverse_link(X * x0.head(k), final_link);
                Eigen::VectorXd residuals = y - mu_hat;
                double var_res = residuals.array().square().sum() / static_cast<double>(std::max<long>(n - k, 1L));
                double initial_sigma = std::sqrt(std::max(var_res, 0.1));
                x0(k) = std::log(initial_sigma);

                double mean_res = residuals.mean();
                double cube_sum = 0.0; double sq_sum = 0.0;
                for (long i = 0; i < n; ++i) {
                    double dev = residuals(i) - mean_res;
                    cube_sum += dev * dev * dev; sq_sum += dev * dev;
                }
                double skewness = (cube_sum / static_cast<double>(n)) / std::pow(std::max(1e-6, sq_sum / static_cast<double>(n)), 1.5);
                x0(k + 1) = (skewness >= 0.0) ? 0.4 : -0.4;

                x0(k + 2) = std::log(5.0);

                SkewTLoss loss_fn(X, y, final_link);

                // ── Starts: (0) the classic one (OLS, nu=5) + (1..3) ROBUST starts in the style of R's st.prelimFit:
                //    median regression, scale by IQR and alpha by Galton's measure, with nu in {2.5, 6, 20}.
                std::vector<Eigen::VectorXd> starts;
                starts.push_back(x0);
                if (skewt_multistart_ && n >= 10) {
                    Eigen::VectorXd b_lad = lad_irls(X, eta_target, x0.head(k));
                    if (b_lad.allFinite()) {
                        Eigen::VectorXd res_lad = y - evaluate_inverse_link(X * b_lad, final_link);
                        std::vector<double> rs(res_lad.data(), res_lad.data() + res_lad.size());
                        std::sort(rs.begin(), rs.end());
                        // The parameterization is for the MEAN: shift the intercept by the trimmed mean
                        // of the residuals (the LAD median is ~0). Only if column 0 is constant 1.
                        if (k > 0 && (X.col(0).array() == 1.0).all()) b_lad(0) += trimmed_mean_sorted(rs, 0.1);
                        const double q1 = quantile_sorted(rs, 0.25);
                        const double q2 = quantile_sorted(rs, 0.50);
                        const double q3 = quantile_sorted(rs, 0.75);
                        const double iqr = q3 - q1;
                        const double sig_robust = (iqr > 0.0) ? iqr / 1.349
                                                              : std::sqrt(std::max(var_res, 1e-6));
                        const double galton = (iqr > 0.0) ? (q3 + q1 - 2.0 * q2) / iqr : 0.0;
                        const double a_sign = (galton < 0.0) ? -1.0 : 1.0;
                        const double a0 = a_sign * std::min(4.0, std::max(0.3, 15.0 * std::abs(galton)));
                        for (double nu0 : {2.5, 6.0, 20.0}) {
                            Eigen::VectorXd xs(total_params);
                            xs.head(k) = b_lad;
                            // IQR of a t(nu) ~ 1.349*(1+0.42/nu) relative to the normal
                            const double s0 = std::max(sig_robust / (1.0 + 0.42 / nu0), 1e-3);
                            xs(k)     = std::log(s0);
                            xs(k + 1) = a0;
                            xs(k + 2) = std::log(nu0);
                            if (xs.allFinite()) starts.push_back(std::move(xs));
                        }
                    }
                }

                MleResult res;
                Eigen::VectorXd x_opt;
                {
                    bool have = false;
                    double best_ll = -std::numeric_limits<double>::infinity();
                    int best_idx = -1, total_fev = 0, total_it = 0;
                    for (size_t s = 0; s < starts.size(); ++s) {
                        auto pr = run_optimization(loss_fn, starts[s], total_params, maxiter, ftol, gtol, maxcor, verbose);
                        total_fev += pr.first.function_evaluations;
                        total_it  += pr.first.iterations;
                        const double ll = pr.first.log_likelihood;
                        if (!std::isfinite(ll)) continue;
                        if (!have || ll > best_ll + 1e-9) {
                            have = true; best_ll = ll; best_idx = static_cast<int>(s);
                            res = std::move(pr.first);
                            x_opt = std::move(pr.second);
                        }
                    }
                    if (!have) {   // all starts failed: rerun the classic start to return its diagnostics
                        auto pr = run_optimization(loss_fn, starts[0], total_params, maxiter, ftol, gtol, maxcor, verbose);
                        res = std::move(pr.first);
                        x_opt = std::move(pr.second);
                        best_idx = 0;
                    }
                    if (starts.size() > 1) {
                        res.function_evaluations = total_fev;
                        res.iterations = total_it;
                        res.message += " [multi-start: " + std::to_string(starts.size()) +
                                       " starts, best #" + std::to_string(best_idx) + "]";
                    }
                }

                // Auxiliary parameters are ALWAYS reported within the range where the loss evaluates
                // them, and the log-likelihood is recomputed at that same point: consistent (parameters, LL).
                {
                    const double ls_lo = SkewTLoss::LS_MIN, ls_hi = SkewTLoss::LS_MAX;
                    const double a_lo  = SkewTLoss::A_MIN,  a_hi  = SkewTLoss::A_MAX;
                    const double ln_lo = SkewTLoss::LN_MIN, ln_hi = SkewTLoss::LN_MAX;
                    x_opt(k)     = std::min(std::max(x_opt(k),     ls_lo), ls_hi);
                    x_opt(k + 1) = std::min(std::max(x_opt(k + 1), a_lo),  a_hi);
                    x_opt(k + 2) = std::min(std::max(x_opt(k + 2), ln_lo), ln_hi);
                    const double f_final = loss_fn(x_opt);
                    if (std::isfinite(f_final)) res.log_likelihood = -f_final;
                }

                res.x_opt = x_opt;
                if (compute_se) {
                    // Raw observed Hessian (no eigenvalue clipping) for vcov/SE.
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.gamma = x_opt.segment(k, 3);       // [log sigma, alpha, log nu]
                res.extra_param = std::exp(x_opt(k));
                return res;
            }

            // 18. LOGISTIC (CONTINUOUS)
            case FAM_LOGISTIC_IDENTITY:
            case FAM_LOGISTIC_LOG: {
                const int total_params = static_cast<int>(k + 1);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                LinkType final_link = resolve_link(family);
                Eigen::VectorXd eta_target = transform_link_forward(y, final_link);
                x0.head(k) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k).allFinite()) {
                    x0.head(k).setZero();
                    if (fit_intercept_ && k > 0) x0(0) = eta_target.mean();
                }

                Eigen::VectorXd mu_hat = evaluate_inverse_link(X * x0.head(k), final_link);
                Eigen::VectorXd residuals = y - mu_hat;
                double var_res = residuals.array().square().sum() / static_cast<double>(std::max<long>(n - k, 1L));

                double initial_sigma = std::sqrt(std::max(var_res, 0.05)) * 0.5513288954217921;
                x0(k) = std::log(initial_sigma);

                LogisticLoss loss_fn(X, y, final_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k);
                res.extra_param = std::exp(x_opt(k));
                return std::move(res);
            }

            // SIMPLEX REGRESSION
            case FAM_SIMPLEX_LOGIT:
            case FAM_SIMPLEX_PROBIT:
            case FAM_SIMPLEX_LOGLOG:
            case FAM_SIMPLEX_CLOGLOG:
            case FAM_SIMPLEX_CAUCHIT: {
                Eigen::MatrixXd Z_mat = Z.value_or(Eigen::MatrixXd::Ones(n, 1));
                const long k_mu = k;
                const long k_sigma = Z_mat.cols();
                const int total_params = static_cast<int>(k_mu + k_sigma);

                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                Eigen::VectorXd y_clamp = y.array().cwiseMax(1e-4).cwiseMin(1.0 - 1e-4);
                Eigen::VectorXd eta_target = transform_link_forward(y_clamp, link);

                x0.head(k_mu) = (X.transpose() * X).ldlt().solve(X.transpose() * eta_target);
                if (!x0.head(k_mu).allFinite()) {
                    x0.head(k_mu).setZero();
                    if (fit_intercept_ && k_mu > 0) x0(0) = eta_target.mean();
                }

                double mean_y = std::clamp(y.mean(), 0.01, 0.99);
                double var_y = (y.array() - mean_y).square().sum() / static_cast<double>(std::max<long>(n - 1, 1));
                var_y = std::max(1e-6, var_y);

                double mu3 = mean_y * mean_y * mean_y;
                double mu1_3 = (1.0 - mean_y) * (1.0 - mean_y) * (1.0 - mean_y);
                double initial_sigma_sq = var_y / (mu3 * mu1_3);
                double initial_sigma = std::sqrt(std::clamp(initial_sigma_sq, 0.01, 10.0));

                x0(k_mu) = compute_positive_link_forward(initial_sigma, sec_link);

                SimplexLoss loss_fn(X, Z_mat, y, link, sec_link);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                res.beta = x_opt.head(k_mu);
                res.gamma = x_opt.segment(k_mu, k_sigma);
                res.extra_param = evaluate_positive_link(x_opt(k_mu), sec_link);   // sigma^2 (intercept-only Z); = exp(.) for Log
                return std::move(res);
            }

            // MULTINOMIAL REGRESSION
            case FAM_MULTINOMIAL_LOGIT:
            case FAM_MULTINOMIAL_PROBIT:
            case FAM_MULTINOMIAL_CLOGLOG:
            case FAM_MULTINOMIAL_LOGLOG:
            case FAM_MULTINOMIAL_CAUCHIT:
            {
                LinkType link_mode = LinkType::Logit;
                if (family == FAM_MULTINOMIAL_PROBIT)   link_mode = LinkType::Probit;
                if (family == FAM_MULTINOMIAL_CLOGLOG)  link_mode = LinkType::Cloglog;
                if (family == FAM_MULTINOMIAL_LOGLOG)   link_mode = LinkType::Loglog;
                if (family == FAM_MULTINOMIAL_CAUCHIT)  link_mode = LinkType::Cauchit;

                Eigen::VectorXi y_int = y.cast<int>();

                const int K = y_int.maxCoeff() + 1;
                const int K_minus_1 = K - 1;
                const int total_params = static_cast<int>(k * K_minus_1);

                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                Eigen::VectorXd count_cat = Eigen::VectorXd::Zero(K);
                for (long i = 0; i < n; ++i) count_cat(y_int(i)) += 1.0;
                double ref_count = std::max(count_cat(K - 1), 0.5);

                auto xTx_fact = (X.transpose() * X).ldlt();
                Eigen::Map<Eigen::MatrixXd> beta_init(x0.data(), k, K_minus_1);

                for (int j = 0; j < K_minus_1; ++j) {
                    double cat_j_count = std::max(count_cat(j), 0.5);
                    double p_j = cat_j_count / (cat_j_count + ref_count);
                    double eta_j = compute_single_link_forward(p_j, link_mode);

                    Eigen::VectorXd y_pseudo = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < n; ++i) {
                        if (y_int(i) == j) {
                            y_pseudo(i) = 0.5;
                        } else if (y_int(i) == K - 1) {
                            y_pseudo(i) = -0.5;
                        }
                    }
                    Eigen::VectorXd slope_init = xTx_fact.solve(X.transpose() * y_pseudo);
                    if (slope_init.allFinite()) {
                        beta_init.col(j) = slope_init;
                    }
                    if (fit_intercept_ && k > 0) {
                        beta_init(0, j) = eta_j;
                    }
                }

                if (link_mode != LinkType::Logit) {
                    beta_init *= 0.5;
                }

                MultinomialMultiLinkLoss loss_fn(X, y_int, K, link_mode);
                auto [res_opt, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res_opt.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res_opt, loss_fn.hessian(x_opt));
                }

                res_opt.beta = x_opt;
                res_opt.extra_param = 0.0;

                return std::move(res_opt);
            }

            // ORDINAL REGRESSION
            case FAM_ORDINAL_LOGIT:
            case FAM_ORDINAL_PROBIT:
            case FAM_ORDINAL_CLOGLOG:
            case FAM_ORDINAL_LOGLOG:
            case FAM_ORDINAL_CAUCHIT:
            {
                LinkType link_mode = LinkType::Logit;
                if (family == FAM_ORDINAL_PROBIT)   link_mode = LinkType::Probit;
                if (family == FAM_ORDINAL_CLOGLOG)  link_mode = LinkType::Cloglog;
                if (family == FAM_ORDINAL_LOGLOG)   link_mode = LinkType::Loglog;
                if (family == FAM_ORDINAL_CAUCHIT)  link_mode = LinkType::Cauchit;

                Eigen::VectorXi y_int = y.cast<int>();
                const int M = y_int.maxCoeff() + 1;
                if (M < 3) {
                    throw std::invalid_argument(
                        "MleSolver: Ordinal regression requires at least 3 categories (0..M-1); use Binomial for 2 categories.");
                }

                if (k > 0 && X.col(0).isApprox(Eigen::VectorXd::Ones(n), 1e-4)) {
                    throw std::invalid_argument(
                        "MleSolver: Intercept column not allowed in design matrix X for Ordinal family.");
                }

                const int num_thresholds = M - 1;
                const int total_params = static_cast<int>(k + num_thresholds);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);

                Eigen::VectorXd counts = Eigen::VectorXd::Zero(M);
                for (long i = 0; i < n; ++i) counts(y_int(i)) += 1.0;

                auto pure_qcauchy = [](double p) -> double {
                    return std::tan(glm_PI * (p - 0.5));
                };

                Eigen::VectorXd alpha0(num_thresholds);
                double running = 0.0;
                for (int m = 0; m < num_thresholds; ++m) {
                    running += counts(m);
                    double p = std::clamp(running / static_cast<double>(n), 1e-3, 1.0 - 1e-3);

                    if (link_mode == LinkType::Logit) {
                        alpha0(m) = std::log(p / (1.0 - p));
                    } else if (link_mode == LinkType::Probit) {
                        alpha0(m) = pure_qnorm(p);
                    } else if (link_mode == LinkType::Cloglog) {
                        alpha0(m) = std::log(-std::log(1.0 - p));
                    } else if (link_mode == LinkType::Loglog) {
                        alpha0(m) = -std::log(-std::log(p));
                    } else if (link_mode == LinkType::Cauchit) {
                        alpha0(m) = pure_qcauchy(p);
                    }
                }

                x0(k) = alpha0(0);
                for (int m = 1; m < num_thresholds; ++m) {
                    double diff = std::max(alpha0(m) - alpha0(m - 1), 1e-3);
                    x0(k + m) = std::log(diff);
                }

                Eigen::VectorXd y_double = y_int.cast<double>();
                OrdinalLoss loss_fn(X, y_double, M, link_mode);
                auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

                res.x_opt = x_opt;
                if (compute_se) {
                    attach_vcov_se(res, loss_fn.hessian(x_opt));
                }

                Eigen::VectorXd alpha_final(num_thresholds);
                loss_fn.reconstruct_thresholds(x_opt, static_cast<int>(k), alpha_final);

                res.beta = x_opt.head(k);
                res.gamma = alpha_final;
                res.extra_param = static_cast<double>(M);
                return std::move(res);
            }

            default: {
                throw std::invalid_argument("MleSolver: Unsupported family code.");
            }
        }
    }

    /** @brief Fits Poisson Inflated model at target response value. */
    MleResult fit_poisson_inflated(const Eigen::MatrixXd& X_raw,
                                   const Eigen::VectorXd& y,
                                   double target_value,
                                   LinkType base_link = LinkType::Log,
                                   LinkType infl_link = LinkType::Logit,
                                   int maxiter = 1000,
                                   double ftol = 1e-12,
                                   double gtol = 1e-6,
                                   int maxcor = 10,
                                   int verbose = 0,
                                   bool compute_se = true,
                                   std::optional<Eigen::MatrixXd> Z = std::nullopt)
    {
        check_finite(X_raw, "X");
        check_finite(y, "y");
        check_dims(X_raw.rows(), y.size(), "X", "y", "MleSolver::fit_poisson_inflated");
        check_opt_dims(Z, X_raw.rows(), "Z", "MleSolver::fit_poisson_inflated");
        check_finite_opt(Z, "Z");
        validate_secondary_link(infl_link, SecondaryKind::Probability, "MleSolver::fit_poisson_inflated");

        Eigen::MatrixXd X = fit_intercept_ ? add_intercept(X_raw) : X_raw;
        Eigen::MatrixXd Z_mat = Z.value_or(X);
        const long kc = X.cols();
        const long kz = Z_mat.cols();
        const int total_params = static_cast<int>(kc + kz);

        check_full_rank(X, "MleSolver::fit_poisson_inflated (count submodel)");
        check_full_rank(Z_mat, "MleSolver::fit_poisson_inflated (inflation submodel)");
        check_identifiable(X.rows(), total_params, "MleSolver::fit_poisson_inflated");

        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
        double mean_y = y.mean();
        if (kc > 0 && kz > 0) {
            x0(0) = compute_single_link_forward(mean_y, base_link);
            double prop_target = (y.array() == target_value).cast<double>().mean();
            double p_clamped = std::clamp(prop_target, 0.01, 0.99);
            x0(kc) = compute_single_link_forward(p_clamped, infl_link);
        }

        PoissonComponent base_dist;
        PoissonInflatedLoss<PoissonComponent> loss_fn(X, Z_mat, y, target_value, base_link, infl_link, base_dist);
        auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

        res.x_opt = x_opt;
        if (compute_se) {
            attach_vcov_se(res, loss_fn.hessian(x_opt));
        }

        res.beta = x_opt.head(kc);
        res.gamma = x_opt.tail(kz);
        return std::move(res);
    }

    /** @brief Fits Negative Binomial Inflated model at target response value. */
    MleResult fit_negative_binomial_inflated(const Eigen::MatrixXd& X_raw,
                                             const Eigen::VectorXd& y,
                                             double target_value,
                                             VarianceForm var_form = VarianceForm::Quadratic,
                                             LinkType base_link = LinkType::Log,
                                             LinkType infl_link = LinkType::Logit,
                                             int maxiter = 1000,
                                             double ftol = 1e-12,
                                             double gtol = 1e-6,
                                             int maxcor = 10,
                                             int verbose = 0,
                                             bool compute_se = true,
                                             std::optional<Eigen::MatrixXd> Z = std::nullopt)
    {
        check_finite(X_raw, "X");
        check_finite(y, "y");
        check_dims(X_raw.rows(), y.size(), "X", "y", "MleSolver::fit_negative_binomial_inflated");
        check_opt_dims(Z, X_raw.rows(), "Z", "MleSolver::fit_negative_binomial_inflated");
        check_finite_opt(Z, "Z");
        validate_secondary_link(infl_link, SecondaryKind::Probability, "MleSolver::fit_negative_binomial_inflated");

        Eigen::MatrixXd X = fit_intercept_ ? add_intercept(X_raw) : X_raw;
        Eigen::MatrixXd Z_mat = Z.value_or(X);
        const long kc = X.cols();
        const long kz = Z_mat.cols();

        const int total_params = static_cast<int>(kc + kz + 1);

        check_full_rank(X, "MleSolver::fit_negative_binomial_inflated (count submodel)");
        check_full_rank(Z_mat, "MleSolver::fit_negative_binomial_inflated (inflation submodel)");
        check_identifiable(X.rows(), total_params, "MleSolver::fit_negative_binomial_inflated");

        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
        double mean_y = y.mean();

        if (kc > 0 && kz > 0) {
            x0(0) = compute_single_link_forward(mean_y, base_link);

            double prop_target = (y.array() == target_value).cast<double>().mean();
            double p_clamped = std::clamp(prop_target, 0.01, 0.99);
            x0(kc) = compute_single_link_forward(p_clamped, infl_link);
        }

        x0(kc + kz) = std::log(0.5);

        NegativeBinomialComponent base_dist(var_form);
        NegativeBinomialInflatedLoss<NegativeBinomialComponent> loss_fn(X, Z_mat, y, target_value, base_link, infl_link, base_dist);

        auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

        res.x_opt = x_opt;

        if (compute_se) {
            attach_vcov_se(res, loss_fn.hessian(x_opt));
        }

        res.beta = x_opt.head(kc);
        res.gamma = x_opt.segment(kc, kz);

        res.extra_param = std::exp(x_opt(kc + kz));
        res.extra_param2 = x_opt(kc + kz);

        return std::move(res);
    }

    /**
     * @brief Fits multi-parameter Prentice (1974) Generalized Gamma distribution model.
     */
    MleResult fit_generalized_gamma(const Eigen::MatrixXd& X_raw,
                                     const Eigen::MatrixXd& Y,
                                     const Eigen::VectorXi& mx_mu_cols,
                                     const Eigen::VectorXi& mx_sigma_cols,
                                     const Eigen::VectorXi& mx_Q_cols,
                                     std::optional<Eigen::VectorXd> weights = std::nullopt,
                                     LinkType link_mu = LinkType::Identity,
                                     LinkType link_sigma = LinkType::Log,
                                     LinkType link_Q = LinkType::Identity,
                                     int maxiter = 1000,
                                     double ftol = 1e-12,
                                     double gtol = 1e-6,
                                     int maxcor = 10,
                                     int verbose = 0,
                                     bool compute_se = true)
    {
        const long n = X_raw.rows();
        check_finite(X_raw, "X");
        check_finite(Y, "Y");
        check_dims(X_raw.rows(), Y.rows(), "X", "Y", "MleSolver::fit_generalized_gamma");
        if (Y.cols() < 2) {
            throw std::invalid_argument(
                "MleSolver::fit_generalized_gamma: Y must have at least 2 columns (start, time1).");
        }
        check_opt_dims(weights, n, "weights", "MleSolver::fit_generalized_gamma");
        check_finite_opt(weights, "weights");

        const long p = X_raw.cols();
        auto check_cols_in_range = [&](const Eigen::VectorXi& cols, const char* which) {
            for (Eigen::Index i = 0; i < cols.size(); ++i) {
                if (cols(i) < 0 || cols(i) >= p) {
                    throw std::invalid_argument(
                        std::string("MleSolver::fit_generalized_gamma: ") + which +
                        " has column index out of range for X.");
                }
            }
        };
        check_cols_in_range(mx_mu_cols, "mx_mu_cols");
        check_cols_in_range(mx_sigma_cols, "mx_sigma_cols");
        check_cols_in_range(mx_Q_cols, "mx_Q_cols");

        const int total_params = static_cast<int>(3 + mx_mu_cols.size() + mx_sigma_cols.size() + mx_Q_cols.size());
        check_identifiable(n, total_params, "MleSolver::fit_generalized_gamma");

        auto extract_cols = [&](const Eigen::VectorXi& cols) {
            Eigen::MatrixXd sub(n, cols.size());
            for (Eigen::Index j = 0; j < cols.size(); ++j) sub.col(j) = X_raw.col(cols(j));
            return sub;
        };
        if (mx_mu_cols.size() > 0)
            check_full_rank(extract_cols(mx_mu_cols), "MleSolver::fit_generalized_gamma (mu submodel)");
        if (mx_sigma_cols.size() > 0)
            check_full_rank(extract_cols(mx_sigma_cols), "MleSolver::fit_generalized_gamma (sigma submodel)");
        if (mx_Q_cols.size() > 0)
            check_full_rank(extract_cols(mx_Q_cols), "MleSolver::fit_generalized_gamma (Q submodel)");

        Eigen::VectorXd w = weights.value_or(Eigen::VectorXd::Ones(n));

        double sum_log_t = 0.0;
        long count_ev = 0;
        for (long i = 0; i < n; ++i) {
            if (Y.coeff(i, 0) >= Y.coeff(i, 1) - 1e-5) continue;
            double t1 = Y.coeff(i, 1);
            if (t1 > 0.0) {
                sum_log_t += std::log(t1);
                ++count_ev;
            }
        }
        double mean_log_t = (count_ev > 0) ? (sum_log_t / static_cast<double>(count_ev)) : 0.0;

        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(total_params);
        x0(0) = compute_single_link_forward(mean_log_t, link_mu);
        if (link_sigma == LinkType::Log) {
            x0(1) = std::log(0.8);
        } else if (link_sigma == LinkType::Identity) {
            x0(1) = 0.8;
        } else if (link_sigma == LinkType::Inverse) {
            x0(1) = 1.0 / 0.8;
        } else {
            x0(1) = std::log(0.8);
        }
        x0(2) = 0.0;

        GeneralizedGammaLoss loss_fn(X_raw, Y, w, mx_mu_cols, mx_sigma_cols, mx_Q_cols,
                                      link_mu, link_sigma, link_Q);
        auto [res, x_opt] = run_optimization(loss_fn, x0, total_params, maxiter, ftol, gtol, maxcor, verbose);

        res.x_opt = x_opt;
        if (compute_se) {
            attach_vcov_se(res, loss_fn.hessian(x_opt));
        }

        res.gamma = x_opt.head(3);
        res.beta = x_opt.tail(total_params - 3);
        res.extra_param = x_opt.coeff(2);

        return std::move(res);
    }
};

} // namespace glm

#endif // _GLMFIT_MLE_SOLVE_HPP
