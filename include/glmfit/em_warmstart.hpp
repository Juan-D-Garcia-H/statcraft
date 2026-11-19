// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file em_warmstart.hpp
 * @brief Expectation-Maximization (EM) warm-start routines for Zero-Inflated models.
 * @details Implements EM warm-start algorithms for Zero-Inflated Poisson (ZIP) and
 *          Zero-Inflated Negative Binomial (ZINB) regression models, mimicking `pscl::zeroinfl`
 *          (`control$EM = TRUE` in R). Decouples count and inflation linear predictors during initialization
 *          to avoid optimization traps in flat log-likelihood regions.
 */

#ifndef _GMLFIT_EM_WARMSTART_HPP
#define _GMLFIT_EM_WARMSTART_HPP

#include <cmath>
#include <Eigen/Dense>

#include "mle_utils.hpp"

namespace glm {

/**
 * @brief Performs EM warm-start initialization for Zero-Inflated Poisson (ZIP) models.
 * @details Refines count coefficients ($\beta$) and inflation coefficients ($\gamma$) in-place.
 *          Iterates E-step posterior zero probability calculations and M-step weighted IRLS fits.
 * @param Xc Design matrix for count component ($n \times p_c$).
 * @param Zi Design matrix for inflation component ($n \times p_i$).
 * @param y Observed non-negative integer response vector ($n \times 1$).
 * @param link Link function type for count component (typically Log).
 * @param beta Parameter vector for count model, updated in-place.
 * @param gamma Parameter vector for zero-inflation model, updated in-place.
 * @param infl_link Link function type for zero-inflation component (typically Logit).
 * @param max_em_iter Maximum EM iterations (default = 15).
 * @param reltol Relative log-likelihood change tolerance for termination (default = 1e-4).
 */
inline void zip_em_warmstart(const Eigen::MatrixXd& Xc, const Eigen::MatrixXd& Zi,
                              const Eigen::VectorXd& y, LinkType link,
                              Eigen::VectorXd& beta, Eigen::VectorXd& gamma,
                              LinkType infl_link = LinkType::Logit,
                              int max_em_iter = 15, double reltol = 1e-4) {
    const long n = Xc.rows();
    Eigen::VectorXd probi = Eigen::VectorXd::Zero(n); // P(structural zero | y_i = 0)

    auto compute_ll = [&]() {
        double ll = 0.0;
        const Eigen::VectorXd eta_lam = Xc * beta;
        const Eigen::VectorXd eta_pi  = Zi * gamma;
        for (long i = 0; i < n; ++i) {
            double lam = 0.0, dlam = 0.0;
            evaluate_count_link(eta_lam(i), link, lam, dlam);
            double pi = 0.0, dpi = 0.0;
            evaluate_link(eta_pi(i), infl_link, pi, dpi);
            if (y(i) == 0.0) {
                ll += std::log(std::max(pi + (1.0 - pi) * std::exp(-lam), 1e-300));
            } else {
                ll += -log1p_exp_stable(eta_pi(i)) - lam + y(i) * std::log(lam) - std::lgamma(y(i) + 1.0);
            }
        }
        return ll;
    };

    double ll_old = compute_ll();
    double ll_new = 2.0 * ll_old;

    for (int it = 0; it < max_em_iter && std::abs((ll_old - ll_new) / std::max(1e-300, std::abs(ll_old))) > reltol; ++it) {
        ll_old = compute_ll();

        // E-step: Posterior probability of structural zero
        const Eigen::VectorXd eta_lam = Xc * beta;
        const Eigen::VectorXd eta_pi  = Zi * gamma;
        for (long i = 0; i < n; ++i) {
            if (y(i) != 0.0) { probi(i) = 0.0; continue; }
            double lam = 0.0, dlam = 0.0;
            evaluate_count_link(eta_lam(i), link, lam, dlam);
            double pi = 0.0, dpi = 0.0;
            evaluate_link(eta_pi(i), infl_link, pi, dpi);
            probi(i) = pi / std::max(pi + (1.0 - pi) * std::exp(-lam), 1e-300);
        }

        // M-step: Poisson fit on Xc weighted by (1 - probi); fractional logistic fit on Zi with target probi
        irls_glm(Xc, y, (1.0 - probi.array()).matrix(), link,
                 [](double mu) { return mu; }, beta);
        irls_glm(Zi, probi, Eigen::VectorXd::Ones(n), infl_link,
                 [](double mu) { return mu * (1.0 - mu); }, gamma);

        ll_new = compute_ll();
    }
}

/**
 * @brief Performs EM warm-start initialization for Zero-Inflated Negative Binomial (ZINB) models.
 * @details Refines count coefficients ($\beta$) and inflation coefficients ($\gamma$) under fixed dispersion ($\alpha$).
 *          In the M-step, count parameters are updated via NB2 weighted IRLS and inflation via fractional logistic regression.
 * @param Xc Design matrix for count component ($n \times p_c$).
 * @param Zi Design matrix for inflation component ($n \times p_i$).
 * @param y Observed non-negative response vector ($n \times 1$).
 * @param link Link function type for count component.
 * @param alpha Fixed dispersion parameter $\alpha$ ($Var(Y) = \mu + \alpha\mu^2$).
 * @param beta Count component parameter vector, updated in-place.
 * @param gamma Zero-inflation component parameter vector, updated in-place.
 * @param infl_link Link function type for zero-inflation component (default = Logit).
 * @param max_em_iter Maximum EM iterations (default = 15).
 * @param reltol Relative log-likelihood tolerance threshold (default = 1e-4).
 */
inline void zinb_em_warmstart(const Eigen::MatrixXd& Xc, const Eigen::MatrixXd& Zi,
                               const Eigen::VectorXd& y, LinkType link, double alpha,
                               Eigen::VectorXd& beta, Eigen::VectorXd& gamma,
                               LinkType infl_link = LinkType::Logit,
                               int max_em_iter = 15, double reltol = 1e-4) {
    const long n = Xc.rows();
    Eigen::VectorXd probi = Eigen::VectorXd::Zero(n);
    const double inv_alpha = 1.0 / std::max(1e-8, alpha);

    auto nb_log_pmf0 = [&](double lam) {
        // P(Y=0) under NB2: size = 1/alpha, prob = 1/(1 + alpha * lam)
        const double prob_param = 1.0 / (1.0 + alpha * lam);
        return inv_alpha * std::log(prob_param);
    };
    auto nb_log_pmf = [&](double yi, double lam) {
        const double prob_param = 1.0 / (1.0 + alpha * lam);
        return std::lgamma(yi + inv_alpha) - std::lgamma(inv_alpha) - std::lgamma(yi + 1.0)
             + inv_alpha * std::log(prob_param) + yi * std::log(1.0 - prob_param);
    };

    auto compute_ll = [&]() {
        double ll = 0.0;
        const Eigen::VectorXd eta_lam = Xc * beta;
        const Eigen::VectorXd eta_pi  = Zi * gamma;
        for (long i = 0; i < n; ++i) {
            double lam = 0.0, dlam = 0.0;
            evaluate_link(eta_lam(i), link, lam, dlam);
            lam = std::max(1e-10, std::min(lam, 1e6));
            double pi = 0.0, dpi = 0.0;
            evaluate_link(eta_pi(i), infl_link, pi, dpi);
            if (y(i) == 0.0) {
                const double p0 = std::exp(nb_log_pmf0(lam));
                ll += std::log(std::max(pi + (1.0 - pi) * p0, 1e-300));
            } else {
                ll += -log1p_exp_stable(eta_pi(i)) + nb_log_pmf(y(i), lam);
            }
        }
        return ll;
    };

    double ll_old = compute_ll();
    double ll_new = 2.0 * ll_old;

    for (int it = 0; it < max_em_iter && std::abs((ll_old - ll_new) / std::max(1e-300, std::abs(ll_old))) > reltol; ++it) {
        ll_old = compute_ll();

        // E-step: Posterior structural zero probability
        const Eigen::VectorXd eta_lam = Xc * beta;
        const Eigen::VectorXd eta_pi  = Zi * gamma;
        for (long i = 0; i < n; ++i) {
            if (y(i) != 0.0) { probi(i) = 0.0; continue; }
            double lam = 0.0, dlam = 0.0;
            evaluate_link(eta_lam(i), link, lam, dlam);
            lam = std::max(1e-10, std::min(lam, 1e6));
            double pi = 0.0, dpi = 0.0;
            evaluate_link(eta_pi(i), infl_link, pi, dpi);
            const double p0 = std::exp(nb_log_pmf0(lam));
            probi(i) = pi / std::max(pi + (1.0 - pi) * p0, 1e-300);
        }

        // M-step: NB2 fit on Xc weighted by (1 - probi) with fixed alpha; fractional logit fit on Zi
        irls_glm(Xc, y, (1.0 - probi.array()).matrix(), link,
                 [alpha](double mu) { return mu + alpha * mu * mu; }, beta);
        irls_glm(Zi, probi, Eigen::VectorXd::Ones(n), infl_link,
                 [](double mu) { return mu * (1.0 - mu); }, gamma);

        ll_new = compute_ll();
    }
}

} // namespace glm

#endif // _GMLFIT_EM_WARMSTART_HPP
