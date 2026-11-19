// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file log_likelihood.hpp
 * @brief Negative log-likelihood functors (value, gradient, Hessian) for the maximum-likelihood
 *        regression families of the GLM module.
 *
 * @details Every family is a small `struct` that behaves as a *loss functor*: it is constructed once from
 * the data (design matrices, response, optional censoring/truncation information and link choices) and then
 * evaluated at arbitrary parameter vectors by a numerical optimiser or by the variance estimators
 * (see `mle_solver.hpp`).
 *
 * ## Common interface
 * | Member                                       | Role                                                                           |
 * |----------------------------------------------|--------------------------------------------------------------------------------|
 * | `double operator()(const VectorXd& p) const`  | negative log-likelihood (NLL) at the parameter vector `p`                      |
 * | `VectorXd gradient(const VectorXd& p) const`  | gradient of the NLL (analytical, or finite differences where stated)           |
 * | `MatrixXd hessian(const VectorXd& p) const`   | observed Hessian of the NLL (analytical, or finite differences where stated)   |
 *
 * All functors return the **negative** log-likelihood, so they are *minimised*. Unless a struct states
 * otherwise:
 * - the parameter vector is the concatenation of one block of regression coefficients per linear predictor
 *   (location/mean first) followed by the scalar nuisance parameters (scale, dispersion, shape, ...), which
 *   are normally stored on the **log scale** so that they are unconstrained;
 * - the data are held through **non-owning const references**: the matrices and vectors passed to the
 *   constructor must outlive the functor, and no dimension or domain validation is performed;
 * - links are selected with `LinkType` and evaluated by the helpers of `mle_utils.hpp`
 *   (`evaluate_link*`, `evaluate_inverse_link*`, ...); the exact helper, and how it treats links that are not
 *   meaningful for the family, is documented in each struct;
 * - non-finite values are replaced by large finite sentinels (typically `1e10`) or by zero derivatives, and
 *   Hessians are post-processed (finite mask, symmetrisation, small diagonal ridge) as detailed per family.
 *   These safeguards make optimisers robust but mean that the returned Hessian is not always the exact
 *   derivative of the returned gradient; every struct lists its caveats in its notes.
 *
 * ## Families
 * | Group                       | Structs |
 * |-----------------------------|---------|
 * | @ref glm_ll_survival        | LogNormalLoss, WeibullTLoss, LogLogisticTLoss, GeneralizedGammaLoss |
 * | @ref glm_ll_zero_inflated   | ZeroInflatedPoissonLoss, ZeroInflatedNBLoss, PoissonInflatedLoss, NegativeBinomialInflatedLoss (policies: PoissonComponent, NegativeBinomialComponent) |
 * | @ref glm_ll_hurdle          | HurdlePoissonLoss, HurdleNBLoss, HurdleGeometricLoss, HurdleGammaLoss, HurdleLogNormalLoss, HurdleBetaLoss |
 * | @ref glm_ll_truncated       | TruncatedGaussianLoss, TruncatedPoissonLoss, TruncatedNBLoss, TruncatedGeometricLoss |
 * | @ref glm_ll_location_scale  | StudentTLoss, SkewNormalLoss, SkewTLoss, LogisticLoss, GumbelLoss |
 * | @ref glm_ll_proportion      | BetaRegressionLoss, SimplexLoss |
 * | @ref glm_ll_categorical     | MultinomialMultiLinkLoss, OrdinalLoss |
 *
 * @see mle_utils.hpp (link functions and special functions), gengamma.hpp (Generalized Gamma density and CDF),
 *      families.hpp (IRLS families).
 */

#ifndef _GLMFIT_LOG_LIKELIHOOD_HPP
#define _GLMFIT_LOG_LIKELIHOOD_HPP

#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/policies/policy.hpp>

#include "families.hpp"
#include "mle_utils.hpp"
#include "gengamma.hpp"

/**
 * @defgroup glm_ll_root Maximum-likelihood loss functors
 * @brief Negative log-likelihood functors used by the MLE solvers of the GLM module.
 */

/**
 * @defgroup glm_ll_survival Survival (accelerated failure time) families
 * @ingroup glm_ll_root
 * @brief Survival (accelerated failure time) families: LogNormalLoss, WeibullTLoss, LogLogisticTLoss, GeneralizedGammaLoss.
 */

/**
 * @defgroup glm_ll_zero_inflated Zero- and value-inflated count families
 * @ingroup glm_ll_root
 * @brief Zero- and value-inflated count families: ZeroInflatedPoissonLoss, ZeroInflatedNBLoss, PoissonInflatedLoss, NegativeBinomialInflatedLoss, PoissonComponent, NegativeBinomialComponent.
 */

/**
 * @defgroup glm_ll_hurdle Hurdle (two-part) families
 * @ingroup glm_ll_root
 * @brief Hurdle (two-part) families: HurdlePoissonLoss, HurdleNBLoss, HurdleGeometricLoss, HurdleGammaLoss, HurdleLogNormalLoss, HurdleBetaLoss.
 */

/**
 * @defgroup glm_ll_truncated Truncated families
 * @ingroup glm_ll_root
 * @brief Truncated families: TruncatedGaussianLoss, TruncatedPoissonLoss, TruncatedNBLoss, TruncatedGeometricLoss.
 */

/**
 * @defgroup glm_ll_location_scale Continuous location-scale families
 * @ingroup glm_ll_root
 * @brief Continuous location-scale families: StudentTLoss, SkewNormalLoss, SkewTLoss, LogisticLoss, GumbelLoss.
 */

/**
 * @defgroup glm_ll_proportion Proportion families (responses in the unit interval)
 * @ingroup glm_ll_root
 * @brief Proportion families (responses in the unit interval): BetaRegressionLoss, SimplexLoss.
 */

/**
 * @defgroup glm_ll_categorical Categorical families
 * @ingroup glm_ll_root
 * @brief Categorical families: MultinomialMultiLinkLoss, OrdinalLoss.
 */

namespace glm {

/**
 * @brief Negative log-likelihood of the Log-Normal accelerated failure time (AFT) model with right censoring.
 *
 * @details Models a strictly positive survival time \f$ y_i \f$ whose logarithm is Gaussian:
 * \f[ \log y_i = m_i + \sigma\,\varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,1), \qquad
 *     m_i = g^{-1}(\mathbf{x}_i^\top \boldsymbol\beta), \qquad \sigma = e^{\log\sigma}. \f]
 * The linear predictor is mapped to the *location of \f$ \log y \f$* by the inverse link
 * (`Identity` gives \f$ m_i = \eta_i \f$). Observations are either observed events or right-censored.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                          | Scale                     |
 * |------------|----------------------------------|---------------------------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor scale |
 * | `k`        | \f$ \log\sigma \f$               | log (so \f$ \sigma > 0 \f$ automatically) |
 *
 * where \f$ k \f$ is the number of columns of the design matrix.
 *
 * ### Log-likelihood
 * With the standardised residual \f$ z_i = (\log y_i - m_i)/\sigma \f$, \f$ \phi \f$ and \f$ \Phi \f$ the
 * standard normal density and CDF:
 * - observed event (\f$ d_i = 1 \f$):
 *   \f$ \ell_i = -\log\sigma - \tfrac12\log(2\pi) - \tfrac12 z_i^2 - \log y_i \f$
 *   (the \f$ -\log y_i \f$ Jacobian term makes the value comparable with other AFT families);
 * - right-censored (\f$ d_i = 0 \f$): \f$ \ell_i = \log\{1 - \Phi(z_i)\} \f$, evaluated with
 *   `log_survival_normal()` for accuracy in the upper tail.
 *
 * The functor returns \f$ -\sum_i \ell_i \f$.
 *
 * ### Derivatives
 * Let \f$ \lambda(z) = \phi(z)/(1-\Phi(z)) \f$ be the inverse Mills ratio. Per observation,
 * \f$ \partial\,\mathrm{NLL}_i/\partial z_i = z_i \f$ (event) or \f$ \lambda(z_i) \f$ (censored), and
 * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial\eta_i} = -\frac{1}{\sigma}\frac{\partial\,\mathrm{NLL}_i}{\partial z_i}\,
 *     \frac{d m_i}{d\eta_i}, \qquad
 *     \frac{\partial\,\mathrm{NLL}_i}{\partial\log\sigma} = \mathbb{1}_{d_i=1} - z_i\frac{\partial\,\mathrm{NLL}_i}{\partial z_i}. \f]
 *
 * @note The object stores **non-owning references** to the design matrix, the response and the event
 *       indicator: they must outlive the functor. No validation is performed; \f$ y_i > 0 \f$ is required
 *       (\f$ \log y_i \f$ is taken directly).
 * @see WeibullTLoss, LogLogisticTLoss for the other AFT survival families.
 *
 * @ingroup glm_ll_survival
 */
struct LogNormalLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the location model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed positive survival/censoring times (length \f$ n \f$).
    const Eigen::VectorXd& d;  ///< Event indicator: a value above 0.5 means observed event, otherwise right-censored.
    LinkType link_type;  ///< Inverse link mapping \f$ \eta \f$ to the location \f$ m \f$ of \f$ \log y \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Positive response vector \f$ y \f$ (length \f$ n \f$).
     * @param cens Event indicator \f$ d \f$ (length \f$ n \f$): 1 = event, 0 = right-censored.
     * @param link Link type whose inverse maps \f$ \eta \f$ to the location of \f$ \log y \f$.
     * @pre `exog`, `endog` and `cens` have the same number of rows and outlive the functor.
     */
    LogNormalLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog,
                  const Eigen::VectorXd& cens, LinkType link)
        : X(exog), y(endog), d(cens), link_type(link) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite (a large finite value keeps
     *         line-search based optimisers away from invalid regions).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd log_mu = evaluate_inverse_link(eta, link_type);
        Eigen::VectorXd log_y = y.array().log();

        double total_neg_ll = 0.0;
        const double CONST_NORMAL = 0.9189385332046727;

        for (long i = 0; i < n; ++i) {
            double zi = (log_y(i) - log_mu(i)) / sigma;
            if (d(i) > 0.5) {
                double log_pdf_i = -log_sigma - CONST_NORMAL - (zi * zi / 2.0) - log_y(i);
                total_neg_ll -= log_pdf_i;
            } else {
                total_neg_ll -= log_survival_normal(zi);
            }
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Vector of length \f$ k+1 \f$: \f$ \sum_i \partial\mathrm{NLL}_i/\partial\eta_i\,\mathbf{x}_i \f$
     *         followed by \f$ \sum_i \partial\mathrm{NLL}_i/\partial\log\sigma \f$. Non-finite components
     *         are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd log_mu = evaluate_inverse_link(eta, link_type);
        Eigen::VectorXd dlogmu_deta = evaluate_d_inverse_link(eta, link_type);
        Eigen::VectorXd log_y = y.array().log();

        Eigen::VectorXd dNLL_dbeta = Eigen::VectorXd::Zero(k);
        double dNLL_dlogsigma = 0.0;

        for (long i = 0; i < n; ++i) {
            double zi = (log_y(i) - log_mu(i)) / sigma;

            double dNLL_dzi = 0.0;
            if (d(i) > 0.5) {
                dNLL_dzi = zi;
            } else {
                dNLL_dzi = inverse_mills_ratio(zi);
            }

            if (d(i) > 0.5) {
                dNLL_dlogsigma += 1.0 + dNLL_dzi * (-zi);
            } else {
                dNLL_dlogsigma += dNLL_dzi * (-zi);
            }

            double dNLL_dlogmu_i = dNLL_dzi * (-1.0 / sigma);

            double dNLL_deta_i = dNLL_dlogmu_i * dlogmu_deta(i);

            dNLL_dbeta += dNLL_deta_i * X.row(i).transpose();
        }

        Eigen::VectorXd grad(k + 1);
        grad.head(k) = dNLL_dbeta;
        grad(k) = dNLL_dlogsigma;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) {
                grad(i) = 0.0;
            }
        }

        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @details Block structure
     * \f[ H = \begin{pmatrix} H_{\beta\beta} & h_{\beta,\log\sigma} \\ h_{\beta,\log\sigma}^\top & h_{\log\sigma,\log\sigma} \end{pmatrix}, \f]
     * with \f$ H_{\beta\beta} = X^\top \mathrm{diag}(\partial^2\mathrm{NLL}_i/\partial\eta_i^2)\,X \f$ (this includes the
     * link-curvature term \f$ \partial\mathrm{NLL}_i/\partial m_i \cdot d^2m_i/d\eta_i^2 \f$). The second derivative
     * with respect to \f$ z \f$ is 1 for events and \f$ \lambda'(z) \f$ (`d_inverse_mills_ratio()`) for censored
     * observations; for \f$ \log\sigma \f$ the diagonal term is \f$ 2z^2 \f$ (event) or
     * \f$ z\lambda + z^2\lambda' \f$ (censored).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Symmetric \f$ (k+1)\times(k+1) \f$ matrix. Non-finite entries are set to 0, the matrix is symmetrised
     *         and a ridge of `1e-6` is added to the diagonal so that it can be factorised safely.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long k = X.cols();
        const long n = X.rows();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd log_mu = evaluate_inverse_link(eta, link_type);
        Eigen::VectorXd dlogmu_deta = evaluate_d_inverse_link(eta, link_type);
        Eigen::VectorXd d2logmu_deta2 = evaluate_d2_inverse_link(eta, link_type);
        Eigen::VectorXd log_y = y.array().log();

        Eigen::VectorXd z = (log_y - log_mu) / sigma;

        Eigen::VectorXd imr(n);
        Eigen::VectorXd d_imr(n);
        for (long i = 0; i < n; ++i) {
            double zi = z(i);
            imr(i) = inverse_mills_ratio(zi);

            d_imr(i) = d_inverse_mills_ratio(zi, imr(i));
        }

        auto mask_event = (d.array() > 0.5);

        Eigen::VectorXd dNLL_dz = mask_event.select(z, imr);
        Eigen::VectorXd d2NLL_dz2 = mask_event.select(Eigen::VectorXd::Ones(n), d_imr);

        Eigen::VectorXd dNLL_dlogmu = dNLL_dz * (-1.0 / sigma);
        Eigen::VectorXd d2NLL_dlogmu2 = d2NLL_dz2 / (sigma * sigma);
        Eigen::VectorXd d2NLL_deta2 = d2NLL_dlogmu2.cwiseProduct(dlogmu_deta.cwiseAbs2()) +
                                      dNLL_dlogmu.cwiseProduct(d2logmu_deta2);

        Eigen::MatrixXd H_beta_beta = X.transpose() * d2NLL_deta2.asDiagonal() * X;

        Eigen::VectorXd d2NLL_dlogmu_dlogsigma = (1.0 / sigma) * (dNLL_dz + z.cwiseProduct(d2NLL_dz2));
        Eigen::VectorXd d2NLL_deta_dlogsigma = d2NLL_dlogmu_dlogsigma.cwiseProduct(dlogmu_deta);
        Eigen::VectorXd H_beta_logsigma = X.transpose() * d2NLL_deta_dlogsigma;

        Eigen::VectorXd H_logsigma_diag = mask_event.select(
            2.0 * z.array().square(),
            z.array() * imr.array() + z.array().square() * d_imr.array()
        );
        double H_logsigma_logsigma = H_logsigma_diag.sum();

        Eigen::MatrixXd hess(k + 1, k + 1);
        hess.topLeftCorner(k, k) = H_beta_beta;
        hess.topRightCorner(k, 1) = H_beta_logsigma;
        hess.bottomLeftCorner(1, k) = H_beta_logsigma.transpose();
        hess(k, k) = H_logsigma_logsigma;

        hess = hess.array().isFinite().select(hess, 0.0);

        hess = 0.5 * (hess + hess.transpose());

        hess.diagonal().array() += 1e-6;

        return hess;
    }
};

/**
 * @brief Negative log-likelihood of the Weibull accelerated failure time (AFT) model with right censoring.
 *
 * @details Models a strictly positive survival time \f$ y_i \f$ whose logarithm follows a location-scale
 * model with standard minimum-extreme-value (Gumbel-min) errors:
 * \f[ \log y_i = \mu_i + \sigma\,\varepsilon_i, \qquad \mu_i = g^{-1}(\mathbf{x}_i^\top \boldsymbol\beta),
 *     \qquad \sigma = e^{\log\sigma}. \f]
 * Equivalently \f$ y_i \f$ is Weibull with scale \f$ e^{\mu_i} \f$ and shape \f$ 1/\sigma \f$, with survival
 * function \f$ S(y_i) = \exp(-e^{z_i}) \f$ where \f$ z_i = (\log y_i - \mu_i)/\sigma \f$. The linear predictor
 * \f$ \eta_i = \mathbf{x}_i^\top \boldsymbol\beta \f$ is mapped by the inverse link \f$ g^{-1} \f$ (evaluated with
 * `evaluate_link()` / `evaluate_link_d2()`, member `mu_link`) to the *location \f$ \mu_i \f$ of \f$ \log y \f$*
 * (not to the mean of \f$ y \f$); with the default `Identity` link, \f$ \mu_i = \eta_i \f$. Observations are
 * either observed events or right-censored; there are no observation weights.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                          | Scale                                      |
 * |------------|--------------------------------------------------|--------------------------------------------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor scale (before the link)   |
 * | `k`        | \f$ \log\sigma \f$                               | log (so \f$ \sigma > 0 \f$ automatically)  |
 *
 * where \f$ k \f$ is the number of columns of the design matrix. The parameter vector therefore has length
 * \f$ k+1 \f$. There are no optional, fixed or penalised parameters.
 *
 * ### Log-likelihood
 * With \f$ z_i \f$ as above (clamped to \f$ [-30, 30] \f$) and the event indicator \f$ d_i \f$, the per-observation
 * NLL is
 * - observed event (\f$ d_i > 0.5 \f$): \f$ \mathrm{NLL}_i = \log\sigma - z_i + e^{z_i} \f$;
 * - right-censored: \f$ \mathrm{NLL}_i = e^{z_i} \f$.
 *
 * The functor returns \f$ \sum_i \mathrm{NLL}_i \f$. The Jacobian term \f$ \log y_i \f$ of the density of \f$ y \f$
 * is **dropped** (it does not depend on the parameters), so the value differs by \f$ \sum_{i:\,d_i=1}\log y_i \f$
 * from the true NLL of \f$ y \f$ and from the value returned by LogNormalLoss. The response is floored at
 * `1e-15` before taking the logarithm.
 *
 * ### Derivatives
 * Let \f$ g(z) = \partial\,\mathrm{NLL}_i/\partial z_i = -d_i + e^{z_i} \f$ (the indicator enters numerically,
 * without thresholding) and \f$ g'(z) = e^{z_i} \f$. By the chain rule (\f$ \partial z/\partial\mu = -1/\sigma \f$,
 * \f$ \partial z/\partial\log\sigma = -z \f$):
 * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial\eta_i} = -\frac{g(z_i)}{\sigma}\,\frac{d\mu_i}{d\eta_i}, \qquad
 *     \frac{\partial\,\mathrm{NLL}_i}{\partial\log\sigma} = d_i - z_i\,g(z_i). \f]
 * The gradient is analytical. The Hessian is the analytical *observed* Hessian (not the expected information),
 * including the link-curvature term \f$ d^2\mu/d\eta^2 \f$; it is finite-masked, symmetrised and
 * ridge-regularised, but not projected onto the positive semidefinite cone (see hessian()).
 *
 * ### Numerical safeguards
 * - \f$ y_i \f$ is floored at `1e-15` before the logarithm.
 * - \f$ z_i \f$ is clamped to \f$ [-30, 30] \f$ (so \f$ e^{z_i} \le e^{30} \f$ is finite). The clamp is *not*
 *   differentiated: gradient() and hessian() use the clamped \f$ z_i \f$ in the unclamped formulas.
 * - operator() returns `1e10` when the sum is not finite; gradient() replaces non-finite components by 0;
 *   hessian() sets non-finite entries to 0, symmetrises and adds `1e-6` to the diagonal.
 *
 * @note The object stores **non-owning references** to the design matrix, the response and the event
 *       indicator: they must outlive the functor. No validation is performed (dimensions, \f$ y_i > 0 \f$,
 *       \f$ d_i \in \{0,1\} \f$). The event test in operator() is \f$ d_i > 0.5 \f$, whereas gradient() and
 *       hessian() use \f$ d_i \f$ numerically; both agree only when \f$ d_i \in \{0,1\} \f$.
 * @see LogNormalLoss, LogLogisticTLoss for the other AFT survival families.
 *
 * @ingroup glm_ll_survival
 */
struct WeibullTLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the location model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed positive survival/censoring times (length \f$ n \f$).
    const Eigen::VectorXd& d;  ///< Event indicator (length \f$ n \f$): 1 = observed event, 0 = right-censored (see class note on thresholding).
    LinkType mu_link;  ///< Inverse link mapping \f$ \eta \f$ to the location \f$ \mu \f$ of \f$ \log y \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Positive response vector \f$ y \f$ (length \f$ n \f$).
     * @param cens Event indicator \f$ d \f$ (length \f$ n \f$): 1 = event, 0 = right-censored.
     * @param link Link type whose inverse maps \f$ \eta \f$ to the location of \f$ \log y \f$ (default `LinkType::Identity`).
     * @pre `exog`, `endog` and `cens` have the same number of rows and outlive the functor.
     */
    WeibullTLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, const Eigen::VectorXd& cens, LinkType link = LinkType::Identity)
        : X(exog), y(endog), d(cens), mu_link(link) {}

    /**
     * @brief Evaluates the total negative log-likelihood (up to the dropped \f$ \log y_i \f$ term).
     * @details For each observation \f$ \mu_i = g^{-1}(\eta_i) \f$, \f$ z_i = \mathrm{clamp}\{(\log\max(y_i,10^{-15})
     * - \mu_i)/\sigma,\,-30,\,30\} \f$ and the term \f$ \log\sigma - z_i + e^{z_i} \f$ (event, \f$ d_i > 0.5 \f$) or
     * \f$ e^{z_i} \f$ (censored) is accumulated.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ \sum_i \mathrm{NLL}_i \f$, or `1e10` if the sum is not finite (a large finite value keeps
     *         line-search based optimisers away from invalid regions).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        double total_neg_ll = 0.0;

        // accumulate per-observation NLL terms
        for (long i = 0; i < n; ++i) {
            double mu = 0.0;
            double dmu_deta = 0.0;
            evaluate_link(eta(i), mu_link, mu, dmu_deta);

            double log_yi = std::log(std::max(1e-15, y(i)));
            double zi = (log_yi - mu) / sigma;

            zi = std::max(-30.0, std::min(30.0, zi));
            double exp_zi = std::exp(zi);

            if (d(i) > 0.5) {
                total_neg_ll += log_sigma - zi + exp_zi;
            } else {
                total_neg_ll += exp_zi;
            }
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood.
     * @details Per observation, with \f$ g = -d_i + e^{z_i} \f$ and the clamped \f$ z_i \f$:
     * \f$ \partial\mathrm{NLL}_i/\partial\eta_i = -(g/\sigma)\,d\mu_i/d\eta_i \f$ and
     * \f$ \partial\mathrm{NLL}_i/\partial\log\sigma = d_i - z_i\,g \f$. The coefficient block is
     * \f$ X^\top (\partial\mathrm{NLL}/\partial\boldsymbol\eta) \f$. The indicator \f$ d_i \f$ is used numerically here.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return Vector of length \f$ k+1 \f$ (coefficients, then \f$ \log\sigma \f$). Non-finite components are
     *         replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;

        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlog_sigma = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu = 0.0;
            double dmu_deta = 0.0;
            evaluate_link(eta(i), mu_link, mu, dmu_deta);

            double log_yi = std::log(std::max(1e-15, y(i)));
            double zi = (log_yi - mu) / sigma;

            zi = std::max(-30.0, std::min(30.0, zi));
            double exp_zi = std::exp(zi);

            double dNLL_dzi = -d(i) + exp_zi;
            double dNLL_dmu = -dNLL_dzi / sigma;
            dNLL_deta(i) = dNLL_dmu * dmu_deta;

            dNLL_dlog_sigma += d(i) - zi * dNLL_dzi;
        }

        // assemble the gradient: X^T dNLL/deta, then dNLL/dlog(sigma)
        Eigen::VectorXd grad(k + 1);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k)      = dNLL_dlog_sigma;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @details Block structure
     * \f[ H = \begin{pmatrix} H_{\beta\beta} & h_{\beta,\log\sigma} \\ h_{\beta,\log\sigma}^\top & h_{\log\sigma,\log\sigma} \end{pmatrix}, \f]
     * with \f$ g = -d_i + e^{z_i} \f$, \f$ g' = e^{z_i} \f$ and
     * - \f$ H_{\beta\beta} = X^\top \mathrm{diag}(w_i)\,X \f$,
     *   \f$ w_i = (g'/\sigma^2)(d\mu_i/d\eta_i)^2 - (g/\sigma)\,d^2\mu_i/d\eta_i^2 \f$ (the second derivatives of the
     *   link come from `evaluate_link_d2()`);
     * - \f$ h_{\beta,\log\sigma} = X^\top \mathrm{diag}(d\mu_i/d\eta_i)\,\mathbf{c} \f$ with
     *   \f$ c_i = \sigma^{-1}\{g + z_i e^{z_i}\} \f$ (the exact derivative of the \f$ \log\sigma \f$ gradient component with respect to \f$ \mu_i \f$);
     * - \f$ h_{\log\sigma,\log\sigma} = \sum_i \{ z_i\,g + z_i^2 e^{z_i} \} \f$.
     *
     * Quantities are evaluated at the clamped \f$ z_i \in [-30,30] \f$ and the clamp is not differentiated.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return Symmetric \f$ (k+1)\times(k+1) \f$ matrix. Non-finite entries are set to 0, the matrix is symmetrised
     *         and a ridge of `1e-6` is added to the diagonal so that it can be factorised safely. It is not
     *         projected onto the positive semidefinite cone.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;

        Eigen::VectorXd mu_vec(n);
        Eigen::VectorXd dmu_deta_vec(n);
        Eigen::VectorXd d2mu_deta2_vec(n);

        for (long i = 0; i < n; ++i) {
            evaluate_link_d2(eta(i), mu_link, mu_vec(i), dmu_deta_vec(i), d2mu_deta2_vec(i));
        }

        Eigen::VectorXd log_y = y.array().cwiseMax(1e-15).log();
        Eigen::VectorXd z = (log_y - mu_vec) / sigma;

        z = z.cwiseMax(-30.0).cwiseMin(30.0);
        Eigen::VectorXd exp_z = z.array().exp();

        // derivatives of the NLL with respect to z
        Eigen::VectorXd dNLL_dz = -d + exp_z;
        Eigen::VectorXd d2NLL_dz2 = exp_z;

        Eigen::VectorXd d2NLL_dmu2 = d2NLL_dz2 / (sigma * sigma);
        Eigen::VectorXd dNLL_dmu = -dNLL_dz / sigma;

        Eigen::VectorXd d2NLL_deta2 = d2NLL_dmu2.cwiseProduct(dmu_deta_vec.cwiseAbs2()) +
                                      dNLL_dmu.cwiseProduct(d2mu_deta2_vec);

        Eigen::MatrixXd H_beta_beta = X.transpose() * d2NLL_deta2.asDiagonal() * X;

        Eigen::VectorXd d2NLL_dmu_dlogsigma = (1.0 / sigma) * (dNLL_dz + z.cwiseProduct(exp_z));
        Eigen::VectorXd d2NLL_deta_dlogsigma = d2NLL_dmu_dlogsigma.cwiseProduct(dmu_deta_vec);
        Eigen::VectorXd H_beta_logsigma = X.transpose() * d2NLL_deta_dlogsigma;

        Eigen::VectorXd H_logsigma_diag = (z.array() * dNLL_dz.array()) +
                                          (z.array().square() * exp_z.array());
        double H_logsigma_logsigma = H_logsigma_diag.sum();

        // assemble Hessian blocks, then finite-mask, symmetrise and add the ridge
        Eigen::MatrixXd hess(k + 1, k + 1);
        hess.topLeftCorner(k, k) = H_beta_beta;
        hess.topRightCorner(k, 1) = H_beta_logsigma;
        hess.bottomLeftCorner(1, k) = H_beta_logsigma.transpose();
        hess(k, k) = H_logsigma_logsigma;

        hess = hess.array().isFinite().select(hess, 0.0);
        hess = 0.5 * (hess + hess.transpose());
        hess.diagonal().array() += 1e-6;

        return hess;
    }
};

/**
 * @brief Negative log-likelihood of the Log-Logistic accelerated failure time (AFT) model with right censoring.
 *
 * @details Models a strictly positive survival time \f$ y_i \f$ whose logarithm follows a location-scale
 * model with standard logistic errors:
 * \f[ \log y_i = \mu_i + \sigma\,\varepsilon_i, \qquad \varepsilon_i \sim \mathrm{Logistic}(0,1), \qquad
 *     \mu_i = g^{-1}(\mathbf{x}_i^\top \boldsymbol\beta), \qquad \sigma = e^{\log\sigma}. \f]
 * With \f$ z_i = (\log y_i - \mu_i)/\sigma \f$ the survival function is \f$ S(y_i) = 1/(1+e^{z_i}) \f$ and the density is
 * \f$ f(y_i) = e^{z_i}/\{\sigma\,y_i\,(1+e^{z_i})^2\} \f$ (log-logistic with scale \f$ e^{\mu_i} \f$ and shape
 * \f$ 1/\sigma \f$). The linear predictor \f$ \eta_i = \mathbf{x}_i^\top \boldsymbol\beta \f$ is mapped by the inverse
 * link \f$ g^{-1} \f$ (`evaluate_link()` / `evaluate_link_d2()`, member `mu_link`) to the *location \f$ \mu_i \f$ of
 * \f$ \log y \f$*; with the default `Identity` link \f$ \mu_i = \eta_i \f$. Observations are either observed events
 * or right-censored; there are no observation weights.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                          | Scale                                      |
 * |------------|--------------------------------------------------|--------------------------------------------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor scale (before the link)   |
 * | `k`        | \f$ \log\sigma \f$                               | log (so \f$ \sigma > 0 \f$ automatically)  |
 *
 * where \f$ k \f$ is the number of columns of the design matrix (length \f$ k+1 \f$ in total). There are no
 * optional, fixed or penalised parameters.
 *
 * ### Log-likelihood
 * With \f$ z_i \f$ clamped to \f$ [-50, 50] \f$ and \f$ \mathrm{sp}(z) = \log(1+e^{z}) \f$ (evaluated stably by
 * `log1p_exp_stable()`), the per-observation NLL is
 * \f[ \mathrm{NLL}_i = d_i\,(\log\sigma - z_i) + (1 + d_i)\,\mathrm{sp}(z_i). \f]
 * For an event (\f$ d_i = 1 \f$) this is \f$ \log\sigma - z_i + 2\,\mathrm{sp}(z_i) \f$ and for a right-censored
 * observation (\f$ d_i = 0 \f$) it is \f$ \mathrm{sp}(z_i) = -\log S(y_i) \f$. The indicator enters the formula
 * numerically (no thresholding). The functor returns \f$ \sum_i \mathrm{NLL}_i \f$. The Jacobian term
 * \f$ \log y_i \f$ of the density of \f$ y \f$ is **dropped**, so the value differs from the true NLL of
 * \f$ y \f$ by \f$ \sum_i d_i \log y_i \f$ and from LogNormalLoss. The response is floored at `1e-15` before the logarithm.
 *
 * ### Derivatives
 * Let \f$ \varsigma(z) = 1/(1+e^{-z}) \f$ (logistic sigmoid), \f$ g(z) = \partial\mathrm{NLL}_i/\partial z_i =
 * -d_i + (1+d_i)\,\varsigma(z_i) \f$ and \f$ g'(z) = (1+d_i)\,\varsigma(z_i)\{1-\varsigma(z_i)\} \f$. Then
 * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial\eta_i} = -\frac{g(z_i)}{\sigma}\,\frac{d\mu_i}{d\eta_i}, \qquad
 *     \frac{\partial\,\mathrm{NLL}_i}{\partial\log\sigma} = d_i - z_i\,g(z_i). \f]
 * The gradient is analytical. The Hessian is the analytical *observed* Hessian including the link-curvature
 * term; only exactly-zero diagonal entries are replaced and non-finite entries masked (see hessian()).
 *
 * ### Numerical safeguards
 * - \f$ y_i \f$ is floored at `1e-15` before the logarithm.
 * - \f$ z_i \f$ is clamped to \f$ [-50, 50] \f$ in operator() and hessian(). In gradient() an observation whose
 *   unclamped \f$ \lvert z_i\rvert > 50 \f$ contributes **nothing** (zero to the \f$ \eta \f$-derivative and to the
 *   \f$ \log\sigma \f$ component), i.e. the clamp is treated as zero derivative.
 * - operator() does **not** guard against non-finite values (unlike WeibullTLoss); gradient() does not sanitise
 *   its result either.
 * - hessian() sets non-finite entries to 0 and replaces diagonal entries equal to exactly 0 by `1e-4`; it does not
 *   symmetrise explicitly (the assembled matrix is symmetric by construction) and adds no ridge.
 *
 * @note The object stores **non-owning references** to the design matrix, the response and the event
 *       indicator: they must outlive the functor. No validation is performed (dimensions, \f$ y_i > 0 \f$,
 *       \f$ d_i \in \{0,1\} \f$).
 * @note gradient() skips clamped observations whereas hessian() keeps them (evaluated at \f$ z = \pm 50 \f$), so the two are
 *       mutually inconsistent in the tails.
 * @see LogNormalLoss, WeibullTLoss for the other AFT survival families.
 *
 * @ingroup glm_ll_survival
 */
struct LogLogisticTLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the location model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed positive survival/censoring times (length \f$ n \f$).
    const Eigen::VectorXd& d;  ///< Event indicator (length \f$ n \f$): 1 = observed event, 0 = right-censored; used numerically in the formulas.
    LinkType mu_link;  ///< Inverse link mapping \f$ \eta \f$ to the location \f$ \mu \f$ of \f$ \log y \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Positive response vector \f$ y \f$ (length \f$ n \f$).
     * @param cens Event indicator \f$ d \f$ (length \f$ n \f$): 1 = event, 0 = right-censored.
     * @param link Link type whose inverse maps \f$ \eta \f$ to the location of \f$ \log y \f$ (default `LinkType::Identity`).
     * @pre `exog`, `endog` and `cens` have the same number of rows and outlive the functor.
     */
    LogLogisticTLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, const Eigen::VectorXd& cens, LinkType link = LinkType::Identity)
        : X(exog), y(endog), d(cens), mu_link(link) {}

    /**
     * @brief Evaluates the total negative log-likelihood (up to the dropped \f$ \log y_i \f$ term).
     * @details For each observation \f$ \mu_i = g^{-1}(\eta_i) \f$,
     * \f$ z_i = \mathrm{clamp}\{(\log\max(y_i,10^{-15}) - \mu_i)/\sigma,\,-50,\,50\} \f$ and the term
     * \f$ d_i(\log\sigma - z_i) + (1+d_i)\log(1+e^{z_i}) \f$ is accumulated.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ \sum_i \mathrm{NLL}_i \f$ (returned as is, without a non-finite fallback).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        double total_neg_ll = 0.0;

        // accumulate per-observation NLL terms
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0;
            double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double log_yi = std::log(std::max(1e-15, y(i)));
            double zi = (log_yi - mu_i) / sigma;

            if (zi > 50.0) zi = 50.0;
            if (zi < -50.0) zi = -50.0;

            double log_one_plus_exp_zi = log1p_exp_stable(zi);

            double loss_i = d(i) * (log_sigma - zi) + (1.0 + d(i)) * log_one_plus_exp_zi;
            total_neg_ll += loss_i;
        }
        return total_neg_ll;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood.
     * @details Per observation, with \f$ g = -d_i + (1+d_i)\varsigma(z_i) \f$ (`sigmoid_scalar()`):
     * \f$ \partial\mathrm{NLL}_i/\partial\eta_i = -(g/\sigma)\,d\mu_i/d\eta_i \f$ and
     * \f$ \partial\mathrm{NLL}_i/\partial\log\sigma = d_i - z_i\,g \f$. If \f$ z_i \f$ had to be clamped to
     * \f$ [-50,50] \f$, the observation is skipped (zero contribution to both parts). The coefficient block is
     * \f$ X^\top (\partial\mathrm{NLL}/\partial\boldsymbol\eta) \f$.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return Vector of length \f$ k+1 \f$ (coefficients, then \f$ \log\sigma \f$); no non-finite sanitising is applied.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;

        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlog_sigma = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0;
            double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double log_yi = std::log(std::max(1e-15, y(i)));
            double zi = (log_yi - mu_i) / sigma;

            bool clamped = false;
            if (zi > 50.0) { zi = 50.0; clamped = true; }
            if (zi < -50.0) { zi = -50.0; clamped = true; }

            if (clamped) {
                dNLL_deta(i) = 0.0;
                continue;
            }

            double sig_z = sigmoid_scalar(zi);

            double dNLL_dzi = -d(i) + (1.0 + d(i)) * sig_z;

            double dNLL_dmu = -dNLL_dzi / sigma;
            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;

            dNLL_dlog_sigma += d(i) - zi * dNLL_dzi;
        }

        // assemble the gradient: X^T dNLL/deta, then dNLL/dlog(sigma)
        Eigen::VectorXd grad(k + 1);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k)      = dNLL_dlog_sigma;
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @details Block structure
     * \f[ H = \begin{pmatrix} H_{\beta\beta} & h_{\beta,\log\sigma} \\ h_{\beta,\log\sigma}^\top & h_{\log\sigma,\log\sigma} \end{pmatrix}, \f]
     * with \f$ g = -d_i + (1+d_i)\varsigma(z_i) \f$, \f$ g' = (1+d_i)\varsigma(z_i)\{1-\varsigma(z_i)\} \f$ and
     * - \f$ H_{\beta\beta} = X^\top \mathrm{diag}(w_i)\,X \f$,
     *   \f$ w_i = (g'/\sigma^2)(d\mu_i/d\eta_i)^2 - (g/\sigma)\,d^2\mu_i/d\eta_i^2 \f$ (link second derivatives from
     *   `evaluate_link_d2()`);
     * - \f$ h_{\beta,\log\sigma} = X^\top \mathrm{diag}(d\mu_i/d\eta_i)\,\mathbf{c} \f$ with
     *   \f$ c_i = \sigma^{-1}\{g + z_i\,g'\} \f$;
     * - \f$ h_{\log\sigma,\log\sigma} = \sum_i \{ z_i\,g + z_i^2\,g' \} \f$.
     *
     * All terms are evaluated at the clamped \f$ z_i \in [-50,50] \f$; unlike gradient(), clamped observations are
     * not dropped.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return Matrix of size \f$ (k+1)\times(k+1) \f$. Non-finite entries are set to 0 and diagonal entries equal to exactly 0
     *         are replaced by `1e-4`. No ridge is added and the matrix is not projected onto the positive semidefinite cone.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;

        Eigen::VectorXd mu_vec(n);
        Eigen::VectorXd dmu_deta_vec(n);
        Eigen::VectorXd d2mu_deta2_vec(n);

        for (long i = 0; i < n; ++i) {
            evaluate_link_d2(eta(i), mu_link, mu_vec(i), dmu_deta_vec(i), d2mu_deta2_vec(i));
        }

        Eigen::VectorXd log_y = y.array().cwiseMax(1e-15).log();
        Eigen::VectorXd z = (log_y - mu_vec) / sigma;

        z = z.cwiseMax(-50.0).cwiseMin(50.0);

        Eigen::VectorXd exp_neg_z = (-z).array().exp();
        Eigen::VectorXd sig_z = 1.0 / (1.0 + exp_neg_z.array());

        // derivatives of the NLL with respect to z
        Eigen::VectorXd dNLL_dz = -d.array() + (1.0 + d.array()) * sig_z.array();
        Eigen::VectorXd d2NLL_dz2 = (1.0 + d.array()) * sig_z.array() * (1.0 - sig_z.array());

        Eigen::VectorXd d2NLL_dmu2 = d2NLL_dz2 / (sigma * sigma);
        Eigen::VectorXd dNLL_dmu = -dNLL_dz / sigma;

        Eigen::VectorXd d2NLL_deta2 = d2NLL_dmu2.cwiseProduct(dmu_deta_vec.cwiseAbs2()) +
                                      dNLL_dmu.cwiseProduct(d2mu_deta2_vec);

        Eigen::MatrixXd H_beta_beta = X.transpose() * d2NLL_deta2.asDiagonal() * X;

        Eigen::VectorXd d2NLL_dmu_dlogsigma = (1.0 / sigma) * (dNLL_dz.array() + z.array() * d2NLL_dz2.array());
        Eigen::VectorXd d2NLL_deta_dlogsigma = d2NLL_dmu_dlogsigma.cwiseProduct(dmu_deta_vec);

        Eigen::MatrixXd H_beta_logsigma = X.transpose() * d2NLL_deta_dlogsigma;

        Eigen::VectorXd H_logsigma_diag = (z.array() * dNLL_dz.array()) +
                                          (z.array().square() * d2NLL_dz2.array());
        double H_logsigma_logsigma = H_logsigma_diag.sum();

        // assemble Hessian blocks, then finite-mask and replace zero diagonals
        Eigen::MatrixXd hess(k + 1, k + 1);
        hess.topLeftCorner(k, k) = H_beta_beta;
        hess.topRightCorner(k, 1) = H_beta_logsigma;
        hess.bottomLeftCorner(1, k) = H_beta_logsigma.transpose();
        hess(k, k) = H_logsigma_logsigma;

        hess = hess.array().isFinite().select(hess, 0.0);
        for (long i = 0; i < k + 1; ++i) {
            if (hess(i, i) == 0.0) hess(i, i) = 1e-4;
        }

        return hess;
    }

};

/**
 * @brief Negative log-likelihood of the zero-inflated Poisson (ZIP) regression model.
 *
 * @details Models a count response \f$ y_i \in \{0,1,2,\dots\} \f$ as a two-component mixture: with probability
 * \f$ \pi_i \f$ the observation is a structural zero, otherwise it is drawn from \f$ \mathrm{Poisson}(\lambda_i) \f$:
 * \f[ P(Y_i = 0) = \pi_i + (1-\pi_i)\,e^{-\lambda_i}, \qquad
 *     P(Y_i = y) = (1-\pi_i)\,\frac{\lambda_i^{y} e^{-\lambda_i}}{y!}, \quad y \ge 1. \f]
 * Two linear predictors are used:
 * - the *count* predictor \f$ \eta^{c}_i = \mathbf{x}_i^\top \boldsymbol\beta \f$ gives the Poisson mean
 *   \f$ \lambda_i = g_c^{-1}(\eta^{c}_i) \f$ (member `count_link`, default `Log`, i.e. \f$ \lambda_i = e^{\min(\eta^c_i,23)} \f$);
 * - the *inflation* predictor \f$ \eta^{z}_i = \mathbf{z}_i^\top \boldsymbol\gamma \f$ gives the structural-zero
 *   probability \f$ \pi_i = g_z^{-1}(\eta^{z}_i) \f$ (member `infl_link`, default `Logit`, i.e. the logistic sigmoid).
 *
 * Both inverse links are evaluated with `evaluate_link()` (first derivative) and
 * `evaluate_d2_inverse_link_scalar()` (second derivative).
 *
 * ### Parameter vector layout
 * | Index                 | Meaning                                                   | Scale                                         |
 * |-----------------------|-----------------------------------------------------------|-----------------------------------------------|
 * | `0 .. kc-1`           | count coefficients \f$ \boldsymbol\beta \f$ (`X` columns) | predictor scale of `count_link` (log for `Log`) |
 * | `kc .. kc+kz-1`       | inflation coefficients \f$ \boldsymbol\gamma \f$ (`Z` columns) | predictor scale of `infl_link` (logit for `Logit`) |
 *
 * where \f$ k_c \f$ and \f$ k_z \f$ are the numbers of columns of `X` and `Z`; the vector has length
 * \f$ k_c + k_z \f$. The code reads \f$ \boldsymbol\beta \f$ with `head(kc)` and \f$ \boldsymbol\gamma \f$ with
 * `tail(kz)`. There are no dispersion parameters and no optional, fixed or penalised parameters.
 *
 * ### Log-likelihood
 * After clamping \f$ \lambda_i \f$ and \f$ \pi_i \f$ (see below), the per-observation log-probability is
 * - \f$ y_i = 0 \f$ (exact floating-point comparison):
 *   \f$ \log\{\pi_i + (1-\pi_i)e^{-\lambda_i}\} \f$, computed by log-sum-exp of
 *   \f$ a = \log\pi_i \f$ and \f$ b = \log(1-\pi_i) - \lambda_i \f$;
 * - \f$ y_i \ne 0 \f$: \f$ \log(1-\pi_i) - \lambda_i + y_i\log\lambda_i - \log\Gamma(y_i+1) \f$
 *   (full pmf, the factorial constant is **kept**, `std::lgamma`).
 *
 * The functor returns the NLL \f$ -\sum_i \log P(Y_i = y_i) \f$. There are no observation weights.
 *
 * ### Derivatives
 * Write \f$ f_0 = \pi + (1-\pi)e^{-\lambda} \f$ and \f$ \pi' = d\pi/d\eta^z \f$, \f$ \lambda' = d\lambda/d\eta^c \f$.
 * For \f$ y = 0 \f$: \f$ \partial\mathrm{NLL}/\partial\pi = -(1-e^{-\lambda})/f_0 \f$ and
 * \f$ \partial\mathrm{NLL}/\partial\lambda = (1-\pi)e^{-\lambda}/f_0 \f$; for \f$ y > 0 \f$:
 * \f$ \partial\mathrm{NLL}/\partial\pi = 1/(1-\pi) \f$ and \f$ \partial\mathrm{NLL}/\partial\lambda = 1 - y/\lambda \f$.
 * The gradient w.r.t. \f$ \eta \f$ multiplies these by \f$ \pi' \f$ and \f$ \lambda' \f$ and the coefficient
 * gradients are \f$ X^\top(\cdot) \f$ and \f$ Z^\top(\cdot) \f$. The Hessian is the analytical *observed* Hessian
 * including the link-curvature terms (\f$ d^2\lambda/d\eta^2 \f$, \f$ d^2\pi/d\eta^2 \f$). It is block-structured
 * with a nonzero cross block between \f$ \boldsymbol\beta \f$ and \f$ \boldsymbol\gamma \f$ only through the
 * zero observations (see hessian()).
 *
 * ### Numerical safeguards
 * - \f$ \lambda_i \f$ is clamped to \f$ [10^{-10}, 10^{6}] \f$ and \f$ \pi_i \f$ to \f$ [10^{-15}, 1-10^{-15}] \f$.
 *   The link derivatives \f$ \lambda', \pi' \f$ (and second derivatives) are those of the *unclamped* link values;
 *   they are not zeroed when a clamp is active.
 * - The `Log` count link itself saturates at \f$ \eta^c = 23 \f$ (see `evaluate_link_d2()`).
 * - operator() and gradient() return their results without a non-finite fallback; hessian() sets non-finite
 *   entries to 0 and replaces diagonal entries equal to exactly 0 by `1e-4`.
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y`: they must outlive the functor. No validation
 *       is performed (equal row counts, \f$ y_i \ge 0 \f$ integer-valued). The zero test is `y(i) == 0.0`, and
 *       non-integer positive values are passed to `lgamma` as is. `X` and `Z` may be the same matrix.
 * @note Internal variable names (`eta_nb`, `dNLL_deta_nb`, `W_nb_*`, `H_cruzado`) refer to the count part
 *       (a leftover of the negative-binomial implementation; no dispersion is involved here).
 * @see ZeroInflatedNBLoss, HurdlePoissonLoss for related count-with-excess-zeros models.
 *
 * @ingroup glm_ll_zero_inflated
 */
struct ZeroInflatedPoissonLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the count (Poisson mean) model (\f$ n \times k_c \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the zero-inflation model (\f$ n \times k_z \f$).
    const Eigen::VectorXd& y;  ///< Observed non-negative counts (length \f$ n \f$).
    LinkType count_link;  ///< Inverse link mapping the count predictor \f$ X\beta \f$ to the Poisson mean \f$ \lambda \f$.
    LinkType infl_link;  ///< Inverse link mapping the inflation predictor \f$ Z\gamma \f$ to the structural-zero probability \f$ \pi \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog_count Count design matrix \f$ X \f$ (\f$ n \times k_c \f$).
     * @param exog_infl Inflation design matrix \f$ Z \f$ (\f$ n \times k_z \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$, non-negative integers stored as doubles).
     * @param c_link Count link type (default `LinkType::Log`); its inverse maps \f$ X\beta \f$ to \f$ \lambda \f$.
     * @param i_link Inflation link type (default `LinkType::Logit`); its inverse should map \f$ Z\gamma \f$ into \f$ (0,1) \f$.
     * @pre `exog_count`, `exog_infl` and `endog` have the same number of rows and outlive the functor.
     */
    ZeroInflatedPoissonLoss(const Eigen::MatrixXd& exog_count, const Eigen::MatrixXd& exog_infl,
                            const Eigen::VectorXd& endog, LinkType c_link = LinkType::Log,
                            LinkType i_link = LinkType::Logit)
        : X(exog_count), Z(exog_infl), y(endog), count_link(c_link), infl_link(i_link) {}

    /**
     * @brief Evaluates the total negative log-likelihood of the zero-inflated Poisson model.
     * @details Zeros use the log-sum-exp form of \f$ \log\{\pi + (1-\pi)e^{-\lambda}\} \f$; positive counts use
     * \f$ \log(1-\pi) - \lambda + y\log\lambda - \log\Gamma(y+1) \f$, with \f$ \lambda \f$ and \f$ \pi \f$ clamped as described
     * in the class documentation.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return The value \f$ -\sum_i \log P(Y_i = y_i) \f$ (returned as is, without a non-finite fallback).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        double total_neg_ll = 0.0;

        // accumulate per-observation log-probabilities
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0, dlam_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_i);
            lambda_i = std::max(1e-10, std::min(lambda_i, 1e6));

            double pi_i = 0.0, dpi_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            double log_prob = 0.0;
            double clogdens0 = -lambda_i;

            if (y(i) == 0.0) {

                double a = std::log(pi_i);
                double b = std::log(1.0 - pi_i) + clogdens0;
                double max_ab = std::max(a, b);
                log_prob = max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
            } else {
                double log_1m_pi = std::log(1.0 - pi_i);
                double log_pmf_pois = -lambda_i + y(i) * std::log(lambda_i) - std::lgamma(y(i) + 1.0);
                log_prob = log_1m_pi + log_pmf_pois;
            }

            total_neg_ll -= log_prob;
        }

        return total_neg_ll;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood.
     * @details Per observation, with \f$ f_0 = \pi + (1-\pi)e^{-\lambda} \f$:
     * - \f$ y = 0 \f$: \f$ \partial\mathrm{NLL}/\partial\eta^z = -\pi'(1-e^{-\lambda})/f_0 \f$ and
     *   \f$ \partial\mathrm{NLL}/\partial\eta^c = \lambda'(1-\pi)e^{-\lambda}/f_0 \f$;
     * - \f$ y > 0 \f$: \f$ \partial\mathrm{NLL}/\partial\eta^z = \pi'/(1-\pi) \f$ and
     *   \f$ \partial\mathrm{NLL}/\partial\eta^c = \lambda'(\lambda - y)/\lambda \f$.
     *
     * The result is assembled as \f$ (X^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^c,\;
     * Z^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^z) \f$.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return Vector of length \f$ k_c + k_z \f$ (count block, then inflation block); no non-finite sanitising is applied.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        Eigen::VectorXd dNLL_deta_nb = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_infl = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0, dlam_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_i);
            lambda_i = std::max(1e-10, std::min(lambda_i, 1e6));

            double pi_i = 0.0, dpi_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            double clogdens0 = -lambda_i;
            double dens0 = pi_i + (1.0 - pi_i) * std::exp(clogdens0);

            if (y(i) == 0.0) {
                double exp_c = std::exp(clogdens0);

                double wres_infl = (dpi_i - exp_c * dpi_i) / dens0;
                dNLL_deta_infl(i) = -wres_infl;

                double wres_count = -std::exp(-std::log(dens0) + std::log(1.0 - pi_i) + clogdens0 + std::log(lambda_i)) * dlam_i / lambda_i;
                dNLL_deta_nb(i) = -wres_count;
            } else {
                double wres_infl = -1.0 / (1.0 - pi_i) * dpi_i;
                dNLL_deta_infl(i) = -wres_infl;

                double wres_count = (y(i) - lambda_i) / lambda_i * dlam_i;
                dNLL_deta_nb(i) = -wres_count;
            }
        }

        // assemble the gradient: count block X^T dNLL/deta_c, inflation block Z^T dNLL/deta_z
        Eigen::VectorXd grad(kc + kz);
        grad.head(kc) = X.transpose() * dNLL_deta_nb;
        grad.tail(kz) = Z.transpose() * dNLL_deta_infl;
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @details Block structure
     * \f[ H = \begin{pmatrix} X^\top W_{cc} X & X^\top W_{cz} Z \\ Z^\top W_{cz} X & Z^\top W_{zz} Z \end{pmatrix}, \f]
     * where \f$ W_{cc}, W_{zz}, W_{cz} \f$ are diagonal matrices holding per observation
     * \f[ W_{cc} = \frac{\partial^2\mathrm{NLL}}{\partial\lambda^2}\lambda'^2 + \frac{\partial\mathrm{NLL}}{\partial\lambda}\lambda'', \quad
     *     W_{zz} = \frac{\partial^2\mathrm{NLL}}{\partial\pi^2}\pi'^2 + \frac{\partial\mathrm{NLL}}{\partial\pi}\pi'', \quad
     *     W_{cz} = \frac{\partial^2\mathrm{NLL}}{\partial\pi\,\partial\lambda}\pi'\lambda'. \f]
     * With \f$ f_0 = \pi + (1-\pi)e^{-\lambda} \f$, \f$ f_\pi = 1-e^{-\lambda} \f$, \f$ f_\lambda = -(1-\pi)e^{-\lambda} \f$:
     * - \f$ y = 0 \f$: \f$ \partial^2\mathrm{NLL}/\partial\pi^2 = f_\pi^2/f_0^2 \f$,
     *   \f$ \partial^2\mathrm{NLL}/\partial\lambda^2 = f_\lambda^2/f_0^2 - (1-\pi)e^{-\lambda}/f_0 \f$,
     *   \f$ \partial^2\mathrm{NLL}/\partial\pi\partial\lambda = f_\pi f_\lambda/f_0^2 - e^{-\lambda}/f_0 \f$
     *   (first derivatives as in gradient());
     * - \f$ y > 0 \f$: \f$ \partial^2\mathrm{NLL}/\partial\pi^2 = 1/(1-\pi)^2 \f$,
     *   \f$ \partial^2\mathrm{NLL}/\partial\lambda^2 = y/\lambda^2 \f$ and the cross term is 0.
     *
     * The link second derivatives \f$ \lambda'', \pi'' \f$ come from `evaluate_d2_inverse_link_scalar()`.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return Symmetric \f$ (k_c+k_z)\times(k_c+k_z) \f$ matrix (symmetric by construction: the lower-left block is the
     *         transpose of the upper-right one). Non-finite entries are set to 0 and diagonal entries equal to exactly 0
     *         are replaced by `1e-4`. No ridge is added and the matrix is not projected onto the positive semidefinite cone.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();
        const long total_params = kc + kz;

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        Eigen::VectorXd W_nb_nb   = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_inf_inf = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_nb_inf  = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0, dlam_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_i);
            lambda_i = std::max(1e-10, std::min(lambda_i, 1e6));
            double d2lam_i = evaluate_d2_inverse_link_scalar(eta_nb(i), count_link);

            double pi_i = 0.0, dpi_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));
            double d2pi_i = evaluate_d2_inverse_link_scalar(eta_infl(i), infl_link);

            double clogdens0 = -lambda_i;
            double exp_c = std::exp(clogdens0);
            double dens0 = pi_i + (1.0 - pi_i) * exp_c;

            if (y(i) == 0.0) {
                double inv_dens0 = 1.0 / dens0;
                double d_dens0_dpi = 1.0 - exp_c;
                double d_dens0_dlambda = -(1.0 - pi_i) * exp_c;

                double dL_dpi = -d_dens0_dpi * inv_dens0;
                double d2L_dpi2 = (d_dens0_dpi * d_dens0_dpi) * (inv_dens0 * inv_dens0);

                double dL_dlambda = -d_dens0_dlambda * inv_dens0;
                double d2_dens0_dlambda2 = (1.0 - pi_i) * exp_c;
                double d2L_dlambda2 = (d_dens0_dlambda * d_dens0_dlambda) * (inv_dens0 * inv_dens0) - d2_dens0_dlambda2 * inv_dens0;

                double d2_dens0_dpidlambda = exp_c;
                double d2L_dpidlambda = (d_dens0_dpi * d_dens0_dlambda) * (inv_dens0 * inv_dens0) - d2_dens0_dpidlambda * inv_dens0;

                W_inf_inf(i) = d2L_dpi2 * (dpi_i * dpi_i) + dL_dpi * d2pi_i;
                W_nb_nb(i)   = d2L_dlambda2 * (dlam_i * dlam_i) + dL_dlambda * d2lam_i;
                W_nb_inf(i)  = d2L_dpidlambda * dpi_i * dlam_i;
            } else {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));

                double dL_dlambda = 1.0 - (y(i) / lambda_i);
                double d2L_dlambda2 = y(i) / (lambda_i * lambda_i);

                W_inf_inf(i) = d2L_dpi2 * (dpi_i * dpi_i) + dL_dpi * d2pi_i;
                W_nb_nb(i)   = d2L_dlambda2 * (dlam_i * dlam_i) + dL_dlambda * d2lam_i;
                W_nb_inf(i)  = 0.0;
            }
        }

        // assemble Hessian blocks: count-count, inflation-inflation and the cross block
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(total_params, total_params);
        H.block(0, 0, kc, kc) = X.transpose() * W_nb_nb.asDiagonal() * X;
        H.block(kc, kc, kz, kz) = Z.transpose() * W_inf_inf.asDiagonal() * Z;

        Eigen::MatrixXd H_cruzado = X.transpose() * W_nb_inf.asDiagonal() * Z;
        H.block(0, kc, kc, kz) = H_cruzado;
        H.block(kc, 0, kz, kc) = H_cruzado.transpose();

        H = H.array().isFinite().select(H, 0.0);
        for (long i = 0; i < total_params; ++i) {
            if (H(i, i) == 0.0) H(i, i) = 1e-4;
        }

        return H;
    }
};

/**
 * @brief Poisson base distribution policy (mean parameterisation) for the inflated-count losses.
 *
 * @details Stateless "component" object that supplies the pieces of the Poisson law required by
 * PoissonInflatedLoss: a lower bound for the mean and the log-pmf with its first two derivatives with respect to
 * the mean \f$ \mu \f$. For a count \f$ y \f$ and mean \f$ \mu > 0 \f$
 * \f[ \log f(y;\mu) = -\mu + y\log\mu - \log\Gamma(y+1), \qquad
 *     \frac{\partial \log f}{\partial\mu} = \frac{y}{\mu} - 1, \qquad
 *     \frac{\partial^2 \log f}{\partial\mu^2} = -\frac{y}{\mu^2}. \f]
 *
 * ### Interface
 * Any type usable as the `BaseDistribution` template argument of PoissonInflatedLoss must provide the same four
 * `const` member functions (`constrain_parameter`, `log_pmf_mu`, `d_log_pmf_dmu`, `d2_log_pmf_dmu2`).
 *
 * @note No validation is performed: \f$ y \f$ is expected to be a non-negative integer stored as `double`
 *       (\f$ \log\Gamma(y+1) \f$ is taken directly) and \f$ \mu \f$ must have been passed through
 *       `constrain_parameter()` first, otherwise \f$ \log\mu \f$ and \f$ 1/\mu \f$ are undefined for \f$ \mu \le 0 \f$.
 * @see PoissonInflatedLoss, NegativeBinomialComponent
 *
 * @ingroup glm_ll_zero_inflated
 */
struct PoissonComponent {
    /**
     * @brief Clamps the mean away from zero.
     * @param mu Mean \f$ \mu \f$, modified in place: replaced by `1e-6` if \f$ \mu < 10^{-6} \f$ (no upper bound).
     */
    inline void constrain_parameter(double& mu) const {
        if (mu < 1e-6) mu = 1e-6;
    }

    /**
     * @brief Poisson log-pmf as a function of the mean.
     * @param yi Observed count \f$ y \f$ (non-negative integer stored as `double`).
     * @param mu Mean \f$ \mu > 0 \f$ (already constrained).
     * @return The value \f$ -\mu + y\log\mu - \log\Gamma(y+1) \f$, with the full normalising constant, computed with
     *         `std::lgamma`.
     */
    double log_pmf_mu(double yi, double mu) const {
        return -mu + yi * std::log(mu) - std::lgamma(yi + 1.0);
    }

    /**
     * @brief First derivative of the log-pmf with respect to the mean.
     * @param yi Observed count \f$ y \f$.
     * @param mu Mean \f$ \mu > 0 \f$ (already constrained).
     * @return The score \f$ y/\mu - 1 \f$.
     */
    double d_log_pmf_dmu(double yi, double mu) const {
        return (yi / mu) - 1.0;
    }

    /**
     * @brief Second derivative of the log-pmf with respect to the mean.
     * @param yi Observed count \f$ y \f$.
     * @param mu Mean \f$ \mu > 0 \f$ (already constrained).
     * @return The curvature \f$ -y/\mu^2 \f$ (independent of the clamp: it is the exact derivative at the given \f$ \mu \f$).
     */
    double d2_log_pmf_dmu2(double yi, double mu) const {
        return -yi / (mu * mu);
    }
};

/**
 * @brief Negative log-likelihood of a count model inflated at an arbitrary value `k_target`, with a pluggable base distribution.
 *
 * @details Generalises zero inflation to inflation at any value \f$ k \f$ (`k_target`). With inflation probability
 * \f$ \pi_i \f$ and a base count law \f$ f(\cdot;\mu_i) \f$ (policy `BaseDistribution`, e.g. PoissonComponent):
 * \f[ P(y_i = k) = \pi_i + (1-\pi_i)\,f(k;\mu_i), \qquad
 *     P(y_i = j) = (1-\pi_i)\,f(j;\mu_i) \quad (j \ne k). \f]
 * Two linear predictors are used:
 * - the *base* predictor \f$ \eta^{b}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ gives the base mean
 *   \f$ \mu_i = g_b^{-1}(\eta^{b}_i) \f$ (`base_link`, evaluated with `evaluate_link()`; then
 *   `base_dist.constrain_parameter()` is applied, e.g. a lower bound);
 * - the *inflation* predictor \f$ \eta^{\pi}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ gives the inflation probability
 *   \f$ \pi_i = g_\pi^{-1}(\eta^{\pi}_i) \f$ (`infl_link`, typically `Logit`).
 *
 * An observation is treated as the inflated value when \f$ |y_i - k| < 10^{-9} \f$.
 *
 * ### Parameter vector layout
 * | Index                | Meaning                                   | Scale                    |
 * |----------------------|-------------------------------------------|--------------------------|
 * | `0 .. kc-1`          | base coefficients \f$ \boldsymbol\beta \f$ | base linear predictor    |
 * | `kc .. kc+kz-1`      | inflation coefficients \f$ \boldsymbol\gamma \f$ | inflation linear predictor |
 *
 * where \f$ k_c \f$ = `X.cols()` and \f$ k_z \f$ = `Z.cols()`. The vector has length \f$ k_c+k_z \f$ and there are
 * no extra (dispersion) parameters: any shape parameter must be stored inside `base_dist`.
 *
 * ### Log-likelihood
 * With \f$ \pi_i \f$ clamped to \f$ [10^{-15}, 1-10^{-15}] \f$ and \f$ B_i = f(k;\mu_i) \f$:
 * - inflated value (\f$ y_i = k \f$): \f$ \ell_i = \log\max\{\pi_i + (1-\pi_i)B_i,\,10^{-15}\} \f$;
 * - otherwise: \f$ \ell_i = \log(1-\pi_i) + \log f(y_i;\mu_i) \f$, where for `Logit` inflation link
 *   \f$ \log(1-\pi_i) = -\log(1+e^{\eta^{\pi}_i}) \f$ is evaluated stably with `log1p_exp_stable()` (no clamp) and for
 *   other links as `std::log(1.0 - pi_i)`.
 *
 * The functor returns \f$ -\sum_i \ell_i \f$. No weights and no truncation are applied; the base pmf is not
 * renormalised.
 *
 * ### Derivatives
 * Let \f$ P_i = \max\{\pi_i + (1-\pi_i)B_i,\,10^{-15}\} \f$ and \f$ s_i = \partial\log f/\partial\mu \f$ (`d_log_pmf_dmu`).
 * Per observation:
 * - inflated value:
 *   \f$ \partial\mathrm{NLL}_i/\partial\mu_i = -(1-\pi_i)B_i s_i/P_i \f$,
 *   \f$ \partial\mathrm{NLL}_i/\partial\pi_i = -(1-B_i)/P_i \f$;
 * - otherwise: \f$ \partial\mathrm{NLL}_i/\partial\mu_i = -s_i \f$ and
 *   \f$ \partial\mathrm{NLL}_i/\partial\pi_i = 1/(1-\pi_i) \f$ (for `Logit`, directly
 *   \f$ \partial\mathrm{NLL}_i/\partial\eta^{\pi}_i = \pi_i \f$).
 *
 * The chain rule multiplies by \f$ d\mu_i/d\eta^{b}_i \f$ and \f$ d\pi_i/d\eta^{\pi}_i \f$, and the result is
 * accumulated as \f$ X^\top(\cdot) \f$ and \f$ Z^\top(\cdot) \f$. The Hessian is the exact analytical observed
 * Hessian (second derivatives of \f$ P_i \f$ plus link-curvature terms), including the \f$ \beta \f$-\f$ \gamma \f$
 * cross block, which is non-zero only for inflated-value observations.
 *
 * ### Numerical safeguards
 * - \f$ \pi_i \f$ is clamped to \f$ [10^{-15}, 1-10^{-15}] \f$ in `operator()` and `gradient()`, **but not** in `hessian()`.
 * - The probability of the inflated value is floored at `1e-15` in the value, the gradient and the Hessian.
 * - Derivatives of the clamps are ignored (\f$ d\mu/d\eta \f$ and \f$ d\pi/d\eta \f$ are not zeroed when a clamp is
 *   active).
 * - `operator()` and `gradient()` do **not** replace non-finite results (no `1e10` fallback and no masking).
 * - `hessian()` replaces non-finite entries by 0 and every diagonal entry that is exactly 0 by `1e-4`; it is not
 *   symmetrised explicitly (it is symmetric by construction), regularised by a ridge, or PSD-projected.
 *
 * @tparam BaseDistribution Policy type providing `void constrain_parameter(double&) const`,
 *         `double log_pmf_mu(double y, double mu) const`, `double d_log_pmf_dmu(double y, double mu) const` and
 *         `double d2_log_pmf_dmu2(double y, double mu) const` (see PoissonComponent).
 * @note The object stores **non-owning references** to the design matrices and the response: they must outlive the
 *       functor. No validation is performed: \f$ y_i \f$ must be a valid support point of the base law (for
 *       PoissonComponent a non-negative integer) and `params` must have length \f$ k_c + k_z \f$ (`head(kc)` and
 *       `tail(kz)` are used, so a longer vector would make the two blocks overlap). The member `k_target` is `const`,
 *       so the functor is not assignable.
 * @see ZeroInflatedPoissonLoss, NegativeBinomialInflatedLoss, PoissonComponent
 *
 * @ingroup glm_ll_zero_inflated
 */
template <typename BaseDistribution>

struct PoissonInflatedLoss {
    const Eigen::MatrixXd& X;  ///< Base design matrix (\f$ n \times k_c \f$) of the base-mean linear predictor.
    const Eigen::MatrixXd& Z;  ///< Inflation design matrix (\f$ n \times k_z \f$) of the inflation-probability linear predictor.
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$).
    const double k_target;  ///< Inflated value \f$ k \f$: observations with \f$ |y_i - k| < 10^{-9} \f$ carry the point mass.
    LinkType base_link;  ///< Link whose inverse maps the base linear predictor to the base mean \f$ \mu \f$.
    LinkType infl_link;  ///< Link whose inverse maps the inflation linear predictor to the inflation probability \f$ \pi \f$ (typically `Logit`).
    BaseDistribution base_dist;  ///< Base count distribution policy (copied by value): mean constraint, log-pmf and its derivatives.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied; `dist` is copied).
     * @param exog Base design matrix \f$ X \f$ (\f$ n \times k_c \f$).
     * @param exog_infl Inflation design matrix \f$ Z \f$ (\f$ n \times k_z \f$).
     * @param endog Response vector \f$ y \f$ (length \f$ n \f$), counts.
     * @param target_val Inflated value \f$ k \f$ (e.g. 0 for zero inflation).
     * @param blink Link type whose inverse maps the base linear predictor to the base mean.
     * @param ilink Link type whose inverse maps the inflation linear predictor to the inflation probability
     *        (`Logit` enables the stable log-complement and the simplified derivatives).
     * @param dist Base distribution policy object (e.g. PoissonComponent).
     * @pre `exog`, `exog_infl` and `endog` have the same number of rows and outlive the functor.
     */
    PoissonInflatedLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_infl, const Eigen::VectorXd& endog,
                         double target_val, LinkType blink, LinkType ilink, BaseDistribution dist)
        : X(exog), Z(exog_infl), y(endog), k_target(target_val), base_link(blink), infl_link(ilink), base_dist(dist) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return The value \f$ -\sum_i \ell_i \f$ with the per-observation terms described in the class documentation.
     *         The result is returned as is (no fallback if it is not finite).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long kc = X.cols();
        const long kz = Z.cols();
        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors
        Eigen::VectorXd eta_base = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;
        double total_neg_ll = 0.0;

        // accumulate per-observation NLL terms
        for (long i = 0; i < X.rows(); ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta_base(i), base_link, mu_i, dmu_deta_i);
            base_dist.constrain_parameter(mu_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            double log_prob_i = 0.0;
            if (std::abs(y(i) - k_target) < 1e-9) {
                double log_P_base_target = base_dist.log_pmf_mu(k_target, mu_i);
                double prob_target = pi_i + (1.0 - pi_i) * std::exp(log_P_base_target);
                log_prob_i = std::log(std::max(prob_target, 1e-15));
            } else {
                double log_not_pi = (infl_link == LinkType::Logit) ? -log1p_exp_stable(eta_infl(i)) : std::log(1.0 - pi_i);
                double log_P_base_yi = base_dist.log_pmf_mu(y(i), mu_i);
                log_prob_i = log_not_pi + log_P_base_yi;
            }
            total_neg_ll -= log_prob_i;
        }
        return total_neg_ll;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return Vector of length \f$ k_c + k_z \f$: \f$ X^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{b} \f$ followed by
     *         \f$ Z^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{\pi} \f$. Entries are not masked if non-finite.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long kc = X.cols();
        const long kz = Z.cols();
        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors
        Eigen::VectorXd eta_base = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        Eigen::VectorXd dNLL_deta_base = Eigen::VectorXd::Zero(X.rows());
        Eigen::VectorXd dNLL_deta_infl = Eigen::VectorXd::Zero(X.rows());

        // per-observation derivatives w.r.t. the linear predictors
        for (long i = 0; i < X.rows(); ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta_base(i), base_link, mu_i, dmu_deta_i);
            base_dist.constrain_parameter(mu_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            if (std::abs(y(i) - k_target) < 1e-9) {
                double log_P_base_target = base_dist.log_pmf_mu(k_target, mu_i);
                double P_base_target = std::exp(log_P_base_target);
                double prob_target = std::max(pi_i + (1.0 - pi_i) * P_base_target, 1e-15);

                double dLogP_dmu = base_dist.d_log_pmf_dmu(k_target, mu_i);
                double dL_dmu = -((1.0 - pi_i) * P_base_target * dLogP_dmu) / prob_target;
                dNLL_deta_base(i) = dL_dmu * dmu_deta_i;

                double dL_dpi = -(1.0 - P_base_target) / prob_target;
                dNLL_deta_infl(i) = dL_dpi * dpi_deta_i;
            } else {
                double dL_dmu = -base_dist.d_log_pmf_dmu(y(i), mu_i);
                dNLL_deta_base(i) = dL_dmu * dmu_deta_i;

                if (infl_link == LinkType::Logit) {
                    dNLL_deta_infl(i) = pi_i;
                } else {
                    double dL_dpi = 1.0 / (1.0 - pi_i);
                    dNLL_deta_infl(i) = dL_dpi * dpi_deta_i;
                }
            }
        }

        // chain rule through the design matrices
        Eigen::VectorXd grad(kc + kz);
        grad.head(kc) = X.transpose() * dNLL_deta_base;
        grad.tail(kz) = Z.transpose() * dNLL_deta_infl;
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @details Block structure
     * \f[ H = \begin{pmatrix} X^\top D_{bb} X & X^\top D_{b\pi} Z \\ Z^\top D_{b\pi} X & Z^\top D_{\pi\pi} Z \end{pmatrix}, \f]
     * with diagonal weight matrices built per observation (\f$ L = \mathrm{NLL}_i \f$):
     * - inflated value: \f$ D_{bb} = L_{\mu\mu}\mu'^2 + L_\mu\mu'' \f$, \f$ D_{\pi\pi} = L_{\pi\pi}\pi'^2 + L_\pi\pi'' \f$,
     *   \f$ D_{b\pi} = L_{\mu\pi}\mu'\pi' \f$, where with \f$ P = \pi+(1-\pi)B \f$,
     *   \f$ P_\mu = (1-\pi)Bs \f$, \f$ P_\pi = 1-B \f$, \f$ P_{\mu\mu} = (1-\pi)B(s'+s^2) \f$, \f$ P_{\pi\pi}=0 \f$,
     *   \f$ P_{\mu\pi} = -Bs \f$ and \f$ L_{xy} = -(P_{xy}P - P_xP_y)/P^2 \f$ (`s` and `s'` are the first and second
     *   derivatives of the base log-pmf);
     * - otherwise: \f$ D_{bb} = -\partial^2\log f/\partial\mu^2\,\mu'^2 - (\partial\log f/\partial\mu)\mu'' \f$,
     *   \f$ D_{b\pi} = 0 \f$ and \f$ D_{\pi\pi} = \pi' \f$ for `Logit` (since \f$ \pi' = \pi(1-\pi) \f$) or
     *   \f$ \pi'^2/(1-\pi)^2 + \pi''/(1-\pi) \f$ for other links.
     *
     * Link derivatives come from `evaluate_link_d2()`.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return Symmetric square matrix of size \f$ k_c+k_z \f$. Non-finite entries are set to 0 and diagonal entries that
     *         are exactly 0 are set to `1e-4`. Unlike the value and the gradient, \f$ \pi_i \f$ is not clamped here and the
     *         derivatives of the mean clamp are ignored.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        Eigen::VectorXd eta_base = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        Eigen::VectorXd H_base_base = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd H_infl_infl = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd H_base_infl = Eigen::VectorXd::Zero(n);

        // per-observation Hessian weights
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0; double d2mu_deta2_i = 0.0;
            evaluate_link_d2(eta_base(i), base_link, mu_i, dmu_deta_i, d2mu_deta2_i);
            base_dist.constrain_parameter(mu_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0; double d2pi_deta2_i = 0.0;
            evaluate_link_d2(eta_infl(i), infl_link, pi_i, dpi_deta_i, d2pi_deta2_i);

            if (std::abs(y(i) - k_target) < 1e-9) {

                double log_P_base_target = base_dist.log_pmf_mu(k_target, mu_i);
                double P_base_target = std::exp(log_P_base_target);
                double prob_target = std::max(pi_i + (1.0 - pi_i) * P_base_target, 1e-15);

                double dLogP_dmu = base_dist.d_log_pmf_dmu(k_target, mu_i);
                double d2LogP_dmu2 = base_dist.d2_log_pmf_dmu2(k_target, mu_i);

                double dP_dmu = (1.0 - pi_i) * P_base_target * dLogP_dmu;
                double dP_dpi = 1.0 - P_base_target;

                double d2P_dmu2 = (1.0 - pi_i) * P_base_target * (d2LogP_dmu2 + dLogP_dmu * dLogP_dmu);
                double d2P_dpi2 = 0.0;
                double d2P_dmu_dpi = -P_base_target * dLogP_dmu;

                double d2NLL_dmu2 = -(d2P_dmu2 * prob_target - dP_dmu * dP_dmu) / (prob_target * prob_target);
                double d2NLL_dpi2 = -(d2P_dpi2 * prob_target - dP_dpi * dP_dpi) / (prob_target * prob_target);
                double d2NLL_dmu_dpi = -(d2P_dmu_dpi * prob_target - dP_dmu * dP_dpi) / (prob_target * prob_target);

                double dNLL_dmu = -dP_dmu / prob_target;
                double dNLL_dpi = -dP_dpi / prob_target;

                H_base_base(i) = d2NLL_dmu2 * dmu_deta_i * dmu_deta_i + dNLL_dmu * d2mu_deta2_i;
                H_infl_infl(i) = d2NLL_dpi2 * dpi_deta_i * dpi_deta_i + dNLL_dpi * d2pi_deta2_i;
                H_base_infl(i) = d2NLL_dmu_dpi * dmu_deta_i * dpi_deta_i;

            } else {

                double dL_dmu = -base_dist.d_log_pmf_dmu(y(i), mu_i);
                double d2L_dmu2 = -base_dist.d2_log_pmf_dmu2(y(i), mu_i);
                H_base_base(i) = d2L_dmu2 * dmu_deta_i * dmu_deta_i + dL_dmu * d2mu_deta2_i;

                if (infl_link == LinkType::Logit) {
                    H_infl_infl(i) = dpi_deta_i;
                } else {
                    double dL_dpi = 1.0 / (1.0 - pi_i);
                    double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));
                    H_infl_infl(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
                }

                H_base_infl(i) = 0.0;
            }
        }

        // assemble Hessian blocks
        Eigen::MatrixXd H_bb = X.transpose() * H_base_base.asDiagonal() * X;
        Eigen::MatrixXd H_gg = Z.transpose() * H_infl_infl.asDiagonal() * Z;
        Eigen::MatrixXd H_bg = X.transpose() * H_base_infl.asDiagonal() * Z;

        Eigen::MatrixXd hess(kc + kz, kc + kz);
        hess.topLeftCorner(kc, kc)     = H_bb;
        hess.topRightCorner(kc, kz)    = H_bg;
        hess.bottomLeftCorner(kz, kc)  = H_bg.transpose();
        hess.bottomRightCorner(kz, kz) = H_gg;

        hess = hess.array().isFinite().select(hess, 0.0);
        for (long i = 0; i < kc + kz; ++i) {
            if (hess(i, i) == 0.0) hess(i, i) = 1e-4;
        }

        return hess;
    }

};

/**
 * @brief Negative log-likelihood of the Poisson hurdle model (zero-truncated Poisson for the positive part).
 *
 * @details Two-part model for a count \f$ y_i \in \{0,1,2,\dots\} \f$. A binary "hurdle" decides whether the count is
 * positive and, if so, the positive count follows a zero-truncated Poisson law:
 * \f[ P(y_i = 0) = 1-\pi_i, \qquad
 *     P(y_i = j) = \pi_i\,\frac{e^{-\lambda_i}\lambda_i^{j}/j!}{1-e^{-\lambda_i}} \quad (j \ge 1). \f]
 * - the *count* predictor \f$ \eta^{c}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ gives the Poisson mean
 *   \f$ \lambda_i = g_c^{-1}(\eta^{c}_i) \f$ (`count_link`, default `Log`);
 * - the *hurdle* predictor \f$ \eta^{z}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ gives
 *   \f$ \pi_i = g_z^{-1}(\eta^{z}_i) = P(y_i > 0) \f$ (`zero_link`, default `Logit`).
 *
 * Note that \f$ \pi_i \f$ is the probability of a **positive** count (not of a structural zero); both links are
 * evaluated with the free function `evaluate_link()`.
 *
 * ### Parameter vector layout
 * | Index             | Meaning                                         | Scale                       |
 * |-------------------|-------------------------------------------------|-----------------------------|
 * | `0 .. kc-1`       | count coefficients \f$ \boldsymbol\beta \f$     | count linear predictor      |
 * | `kc .. kc+kz-1`   | hurdle coefficients \f$ \boldsymbol\gamma \f$   | hurdle linear predictor     |
 *
 * with \f$ k_c \f$ = `X.cols()` and \f$ k_z \f$ = `Z.cols()` (length \f$ k_c + k_z \f$, no dispersion parameter).
 *
 * ### Log-likelihood
 * With \f$ \lambda_i \f$ floored at \f$ 10^{-5} \f$ and \f$ \pi_i \f$ clamped to \f$ [10^{-12}, 1-10^{-12}] \f$:
 * - zero (\f$ y_i < 0.5 \f$): \f$ \ell_i = \log(1-\pi_i) \f$;
 * - positive: \f$ \ell_i = \log\pi_i - \lambda_i + y_i\log\lambda_i - \log\Gamma(y_i+1)
 *   - \log\max\{1-e^{-\lambda_i},\,10^{-12}\} \f$ (full Poisson constant kept).
 *
 * The functor returns \f$ -\sum_i \ell_i \f$, or `1e10` if that sum is not finite. No weights are used.
 *
 * ### Derivatives
 * The NLL is additively separable in \f$ \pi_i \f$ and \f$ \lambda_i \f$, so \f$ \boldsymbol\beta \f$ and
 * \f$ \boldsymbol\gamma \f$ are orthogonal. Per observation, with \f$ \pi'=d\pi/d\eta^z \f$, \f$ \lambda'=d\lambda/d\eta^c \f$:
 * - zero: \f$ \partial\mathrm{NLL}_i/\partial\eta^z_i = \pi'/(1-\pi_i) \f$, no dependence on \f$ \lambda_i \f$;
 * - positive: \f$ \partial\mathrm{NLL}_i/\partial\eta^z_i = -\pi'/\pi_i \f$ and
 *   \f[ \frac{\partial\mathrm{NLL}_i}{\partial\eta^c_i} = \Bigl(1 - \frac{y_i}{\lambda_i} + \frac{1}{e^{\lambda_i}-1}\Bigr)\lambda'. \f]
 *   For \f$ \lambda_i < 10^{-3} \f$ the truncation term \f$ 1/(e^{\lambda}-1) \f$ is replaced by its series
 *   \f$ 1/\lambda - 1/2 + \lambda/12 \f$.
 *
 * The Hessian is block diagonal (no \f$ \beta \f$-\f$ \gamma \f$ cross terms): \f$ X^\top D_c X \f$ and
 * \f$ Z^\top D_z Z \f$ with link-curvature terms from `evaluate_link_d2()`.
 *
 * ### Numerical safeguards
 * - \f$ \lambda_i \ge 10^{-5} \f$, \f$ \pi_i \in [10^{-12}, 1-10^{-12}] \f$, truncation factor
 *   \f$ \ge 10^{-12} \f$ (value only). Derivatives of the clamps are ignored.
 * - `operator()` returns `1e10` if the total is not finite; `gradient()` replaces non-finite components by 0;
 *   `hessian()` replaces non-finite entries by 0 and diagonal entries that are exactly 0 by `1e-4`.
 * - The Hessian is not symmetrised, ridge-regularised or PSD-projected (it is symmetric by construction).
 *
 * @note The object stores **non-owning references** to the design matrices and the response: they must outlive the
 *       functor. No validation is performed: \f$ y_i \f$ must be a non-negative integer stored as `double`
 *       (\f$ y_i < 0.5 \f$ is treated as zero) and `params` must have length \f$ k_c + k_z \f$.
 * @note For \f$ \lambda_i < 10^{-3} \f$ the truncation term \f$ 1/(e^{\lambda}-1) \f$ is replaced by its series
 *       \f$ 1/\lambda - 1/2 + \lambda/12 \f$ in the gradient, and by the derivative of that series,
 *       \f$ -(1/\lambda^2 - 1/12) \f$, in the Hessian; for larger \f$ \lambda \f$ the exact \f$ -e^{\lambda}/(e^{\lambda}-1)^2 \f$ is used.
 * @see HurdleNBLoss, HurdleGeometricLoss, HurdleGammaLoss, HurdleLogNormalLoss
 *
 * @ingroup glm_ll_hurdle
 */
struct HurdlePoissonLoss {
    const Eigen::MatrixXd& X;  ///< Count design matrix (\f$ n \times k_c \f$) of the Poisson-mean linear predictor.
    const Eigen::MatrixXd& Z;  ///< Hurdle design matrix (\f$ n \times k_z \f$) of the linear predictor for \f$ P(y>0) \f$.
    const Eigen::VectorXd& y;  ///< Observed non-negative counts (length \f$ n \f$); values below 0.5 count as zeros.
    LinkType count_link;  ///< Link whose inverse maps the count linear predictor to the Poisson mean \f$ \lambda \f$.
    LinkType zero_link;  ///< Link whose inverse maps the hurdle linear predictor to \f$ \pi = P(y>0) \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Count design matrix \f$ X \f$ (\f$ n \times k_c \f$).
     * @param exog_zero Hurdle design matrix \f$ Z \f$ (\f$ n \times k_z \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$).
     * @param blink Count link type (default `LinkType::Log`); its inverse gives \f$ \lambda \f$.
     * @param zlink Hurdle link type (default `LinkType::Logit`); its inverse gives \f$ P(y>0) \f$.
     * @pre `exog`, `exog_zero` and `endog` have the same number of rows and outlive the functor.
     */
    HurdlePoissonLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_zero, const Eigen::VectorXd& endog,
                      LinkType blink = LinkType::Log, LinkType zlink = LinkType::Logit)
        : X(exog), Z(exog_zero), y(endog), count_link(blink), zero_link(zlink) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors
        Eigen::VectorXd eta_poisson = X * beta;
        Eigen::VectorXd eta_logit = Z * gamma;

        double total_neg_ll = 0.0;

        // accumulate per-observation NLL terms
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_poisson(i), count_link, lambda_i, dlam_deta_i);
            if (lambda_i < 1e-5) lambda_i = 1e-5;

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_logit(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            double log_prob_i = 0.0;

            if (y(i) < 0.5) {
                log_prob_i = std::log(1.0 - pi_i);
            } else {
                double log_barrera = std::log(pi_i);
                double truncation_factor = std::max(1.0 - std::exp(-lambda_i), 1e-12);
                double log_truncation = std::log(truncation_factor);
                double log_poisson_base = -lambda_i + y(i) * std::log(lambda_i) - std::lgamma(y(i) + 1.0);

                log_prob_i = log_barrera - log_truncation + log_poisson_base;
            }
            total_neg_ll -= log_prob_i;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return Vector of length \f$ k_c + k_z \f$: \f$ X^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{c} \f$ followed by
     *         \f$ Z^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{z} \f$. Non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors
        Eigen::VectorXd eta_poisson = X * beta;
        Eigen::VectorXd eta_logit = Z * gamma;

        Eigen::VectorXd dNLL_deta_poisson = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_zero = Eigen::VectorXd::Zero(n);

        // per-observation derivatives w.r.t. the linear predictors
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_poisson(i), count_link, lambda_i, dlam_deta_i);
            if (lambda_i < 1e-5) lambda_i = 1e-5;

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_logit(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 0.5) {

                double dL_dpi = 1.0 / (1.0 - pi_i);
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;
                dNLL_deta_poisson(i) = 0.0;
            } else {

                double dL_dpi = -1.0 / pi_i;
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;

                double truncation_grad = 0.0;
                if (lambda_i < 1e-3) {
                    truncation_grad = 1.0 / lambda_i - 0.5 + lambda_i / 12.0;
                } else {
                    truncation_grad = 1.0 / (std::exp(lambda_i) - 1.0);
                }

                double dL_dlam = 1.0 - (y(i) / lambda_i) + truncation_grad;
                dNLL_deta_poisson(i) = dL_dlam * dlam_deta_i;
            }
        }

        // chain rule through the design matrices
        Eigen::VectorXd grad(kc + kz);
        grad.head(kc) = X.transpose() * dNLL_deta_poisson;
        grad.tail(kz) = Z.transpose() * dNLL_deta_zero;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @details Block diagonal: \f$ H_{\beta\beta} = X^\top\mathrm{diag}(D_c)X \f$ and
     * \f$ H_{\gamma\gamma} = Z^\top\mathrm{diag}(D_z)Z \f$, the off-diagonal blocks are exactly zero because the NLL
     * separates in \f$ \pi_i \f$ and \f$ \lambda_i \f$. Each weight is \f$ L''(\cdot)\,u'^2 + L'(\cdot)\,u'' \f$ for
     * \f$ u \in \{\pi,\lambda\} \f$, with
     * - hurdle part: zero \f$ L'=1/(1-\pi) \f$, \f$ L''=1/(1-\pi)^2 \f$; positive \f$ L'=-1/\pi \f$, \f$ L''=1/\pi^2 \f$;
     * - count part (positive observations only, zero for \f$ y_i<0.5 \f$):
     *   \f$ L' = 1 - y/\lambda + 1/(e^\lambda-1) \f$ and
     *   \f$ L'' = y/\lambda^2 - e^\lambda/(e^\lambda-1)^2 \f$ (small-\f$ \lambda \f$ series for \f$ \lambda<10^{-3} \f$, see
     *   the class note).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_c + k_z \f$.
     * @return Symmetric square matrix of size \f$ k_c+k_z \f$. Non-finite entries are set to 0 and diagonal entries that
     *         are exactly 0 are set to `1e-4`.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        Eigen::VectorXd eta_poisson = X * beta;
        Eigen::VectorXd eta_logit = Z * gamma;

        Eigen::VectorXd H_poisson_poisson = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd H_zero_zero = Eigen::VectorXd::Zero(n);

        // per-observation Hessian weights
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0; double d2lam_deta2_i = 0.0;
            evaluate_link_d2(eta_poisson(i), count_link, lambda_i, dlam_deta_i, d2lam_deta2_i);
            if (lambda_i < 1e-5) lambda_i = 1e-5;

            double pi_i = 0.0; double dpi_deta_i = 0.0; double d2pi_deta2_i = 0.0;
            evaluate_link_d2(eta_logit(i), zero_link, pi_i, dpi_deta_i, d2pi_deta2_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 0.5) {

                double dL_dpi = 1.0 / (1.0 - pi_i);
                double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));

                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
                H_poisson_poisson(i) = 0.0;
            } else {

                double dL_dpi = -1.0 / pi_i;
                double d2L_dpi2 = 1.0 / (pi_i * pi_i);
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;

                double dL_dlam = 1.0 - (y(i) / lambda_i);
                double d2L_dlam2 = y(i) / (lambda_i * lambda_i);

                if (lambda_i < 1e-3) {

                    double truncation_grad = 1.0 / lambda_i - 0.5 + lambda_i / 12.0;
                    dL_dlam += truncation_grad;
                    d2L_dlam2 -= (1.0 / (lambda_i * lambda_i) - 1.0 / 12.0);
                } else {
                    double exp_lam = std::exp(lambda_i);
                    double denom = exp_lam - 1.0;
                    dL_dlam += 1.0 / denom;
                    d2L_dlam2 -= exp_lam / (denom * denom);
                }

                H_poisson_poisson(i) = d2L_dlam2 * dlam_deta_i * dlam_deta_i + dL_dlam * d2lam_deta2_i;
            }
        }

        // assemble the block-diagonal Hessian
        Eigen::MatrixXd H_beta_beta   = X.transpose() * H_poisson_poisson.asDiagonal() * X;
        Eigen::MatrixXd H_gamma_gamma = Z.transpose() * H_zero_zero.asDiagonal() * Z;

        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(kc + kz, kc + kz);
        hess.topLeftCorner(kc, kc) = H_beta_beta;

        hess.bottomRightCorner(kz, kz) = H_gamma_gamma;

        hess = hess.array().isFinite().select(hess, 0.0);
        for (long i = 0; i < kc + kz; ++i) {
            if (hess(i, i) == 0.0) hess(i, i) = 1e-4;
        }

        return hess;
    }

};

/**
 * @brief Negative log-likelihood of the Negative Binomial (NB2) hurdle model with a zero-truncated NB positive part.
 *
 * @details Two-part model for a count \f$ y_i \in \{0,1,2,\dots\} \f$:
 * \f[ P(y_i = 0) = 1-\pi_i, \qquad
 *     P(y_i = j) = \pi_i\,\frac{\mathrm{NB}(j;\lambda_i,\alpha)}{1-p_{0,i}} \quad (j \ge 1), \f]
 * where the NB2 pmf with mean \f$ \lambda_i \f$ and dispersion \f$ \alpha > 0 \f$ (variance
 * \f$ \lambda_i + \alpha\lambda_i^2 \f$) is
 * \f[ \mathrm{NB}(j;\lambda,\alpha) = \frac{\Gamma(j+\alpha^{-1})}{\Gamma(\alpha^{-1})\,j!}
 *     \Bigl(\frac{1}{1+\alpha\lambda}\Bigr)^{1/\alpha}\Bigl(\frac{\alpha\lambda}{1+\alpha\lambda}\Bigr)^{j}, \qquad
 *     p_{0,i} = (1+\alpha\lambda_i)^{-1/\alpha}. \f]
 * - the *count* predictor \f$ \eta^{c}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ gives
 *   \f$ \lambda_i = g_c^{-1}(\eta^{c}_i) \f$ (`count_link`, default `Log`);
 * - the *hurdle* predictor \f$ \eta^{z}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ gives
 *   \f$ \pi_i = g_z^{-1}(\eta^{z}_i) = P(y_i>0) \f$ (`zero_link`, default `Logit`; probability of a **positive** count).
 *
 * Both links use the free function `evaluate_link()`.
 *
 * ### Parameter vector layout
 * | Index                 | Meaning                                       | Scale                          |
 * |-----------------------|-----------------------------------------------|--------------------------------|
 * | `0 .. kc-1`           | count coefficients \f$ \boldsymbol\beta \f$   | count linear predictor         |
 * | `kc .. kc+kz-1`       | hurdle coefficients \f$ \boldsymbol\gamma \f$ | hurdle linear predictor        |
 * | `kc+kz`               | \f$ \log\alpha \f$                            | log; clamped to \f$ [-10,10] \f$ before exponentiation |
 *
 * with \f$ k_c \f$ = `X.cols()`, \f$ k_z \f$ = `Z.cols()`; total length \f$ k_c+k_z+1 \f$.
 *
 * ### Log-likelihood
 * With \f$ \alpha = \exp(\mathrm{clamp}(\log\alpha,-10,10)) \f$, \f$ \lambda_i \ge 10^{-5} \f$ and
 * \f$ \pi_i \in [10^{-12}, 1-10^{-12}] \f$:
 * - zero (\f$ y_i < 0.5 \f$): \f$ \ell_i = \log(1-\pi_i) \f$;
 * - positive: \f$ \ell_i = \log\pi_i + \log\mathrm{NB}(y_i;\lambda_i,\alpha) - \log\max\{1-p_{0,i},\,10^{-12}\} \f$, with the
 *   full NB constant (three `std::lgamma` terms) kept.
 *
 * The functor returns \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite. No weights are used.
 *
 * ### Derivatives
 * The NLL separates additively in \f$ \pi_i \f$ and \f$ (\lambda_i,\alpha) \f$.
 * - Hurdle part (analytical): zero \f$ \pi'/(1-\pi_i) \f$, positive \f$ -\pi'/\pi_i \f$ with respect to \f$ \eta^z_i \f$.
 * - Count part (analytical, positive observations only): with \f$ \partial p_0/\partial\lambda = -p_0/(1+\alpha\lambda) \f$,
 *   \f[ \frac{\partial\mathrm{NLL}_i}{\partial\eta^c_i} = \Bigl[-\frac{y_i}{\lambda_i}
 *       + \frac{(y_i+\alpha^{-1})\alpha}{1+\alpha\lambda_i}
 *       + \frac{1}{\max\{1-p_0,10^{-12}\}}\frac{p_0}{1+\alpha\lambda_i}\Bigr]\lambda'. \f]
 * - \f$ \partial\mathrm{NLL}/\partial\log\alpha \f$ is **not** analytical: it is a central finite difference of
 *   `operator()` with step `1e-5` (so it inherits the clamp of \f$ \log\alpha \f$ and the `1e10` fallback).
 *
 * The Hessian is hybrid: the \f$ \gamma\gamma \f$ block is analytical, the blocks involving \f$ \beta \f$ and
 * \f$ \log\alpha \f$ are central finite differences (step `1e-5`) of `gradient()`, and cross blocks between
 * \f$ \boldsymbol\gamma \f$ and the other parameters are exactly zero (exact by separability).
 *
 * ### Numerical safeguards
 * - \f$ \log\alpha \in [-10,10] \f$, \f$ \lambda_i \ge 10^{-5} \f$, \f$ \pi_i \in [10^{-12},1-10^{-12}] \f$, truncation
 *   factor \f$ \ge 10^{-12} \f$ (floor applied in the value and in the gradient); derivatives of the clamps are ignored.
 * - `operator()` returns `1e10` when not finite; `gradient()` replaces non-finite entries by 0;
 *   `hessian()` replaces non-finite entries by 0 and diagonal entries that are exactly 0 by `1e-4`.
 * - The Hessian is not PSD-projected or ridge-regularised, and the \f$ \beta \f$-\f$ \beta \f$ block is not explicitly
 *   symmetrised (symmetric only up to finite-difference error).
 *
 * @note The object stores **non-owning references** to the design matrices and the response: they must outlive the
 *       functor. No validation is performed: \f$ y_i \f$ must be a non-negative integer stored as `double`
 *       (\f$ y_i < 0.5 \f$ is treated as zero) and `params` must have length \f$ k_c + k_z + 1 \f$.
 * @note `hessian()` costs \f$ 2(k_c+1) \f$ gradient evaluations, each of which costs two `operator()` evaluations
 *       (for \f$ \log\alpha \f$).
 * @see HurdlePoissonLoss, HurdleGeometricLoss, HurdleGammaLoss, ZeroInflatedNBLoss
 *
 * @ingroup glm_ll_hurdle
 */
struct HurdleNBLoss {
    const Eigen::MatrixXd& X;  ///< Count design matrix (\f$ n \times k_c \f$) of the NB-mean linear predictor.
    const Eigen::MatrixXd& Z;  ///< Hurdle design matrix (\f$ n \times k_z \f$) of the linear predictor for \f$ P(y>0) \f$.
    const Eigen::VectorXd& y;  ///< Observed non-negative counts (length \f$ n \f$); values below 0.5 count as zeros.
    LinkType count_link;  ///< Link whose inverse maps the count linear predictor to the NB mean \f$ \lambda \f$.
    LinkType zero_link;  ///< Link whose inverse maps the hurdle linear predictor to \f$ \pi = P(y>0) \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Count design matrix \f$ X \f$ (\f$ n \times k_c \f$).
     * @param exog_zero Hurdle design matrix \f$ Z \f$ (\f$ n \times k_z \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$).
     * @param blink Count link type (default `LinkType::Log`); its inverse gives the NB mean \f$ \lambda \f$.
     * @param zlink Hurdle link type (default `LinkType::Logit`); its inverse gives \f$ P(y>0) \f$.
     * @pre `exog`, `exog_zero` and `endog` have the same number of rows and outlive the functor.
     */
    HurdleNBLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_zero, const Eigen::VectorXd& endog,
                 LinkType blink = LinkType::Log, LinkType zlink = LinkType::Logit)
        : X(exog), Z(exog_zero), y(endog), count_link(blink), zero_link(zlink) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_alpha = params(kc + kz);
        double alpha = std::exp(std::max(-10.0, std::min(10.0, log_alpha)));
        double inv_alpha = 1.0 / alpha;

        // linear predictors
        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        double total_neg_ll = 0.0;
        // accumulate per-observation NLL terms
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_deta_i);
            lambda_i = std::max(1e-5, lambda_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            double nb_zero = std::pow(1.0 / (1.0 + alpha * lambda_i), inv_alpha);
            double log_prob_i = 0.0;

            if (y(i) < 0.5) {
                log_prob_i = std::log(1.0 - pi_i);
            } else {
                double log_barrera = std::log(pi_i);
                double log_nb_pmf = std::lgamma(y(i) + inv_alpha)
                                  - std::lgamma(inv_alpha)
                                  - std::lgamma(y(i) + 1.0)
                                  + inv_alpha * std::log(1.0 / (1.0 + alpha * lambda_i))
                                  + y(i) * std::log(alpha * lambda_i / (1.0 + alpha * lambda_i));
                double log_truncation = std::log(std::max(1.0 - nb_zero, 1e-12));
                log_prob_i = log_barrera + log_nb_pmf - log_truncation;
            }
            total_neg_ll -= log_prob_i;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Gradient of the negative log-likelihood (analytical for \f$ \beta,\gamma \f$, finite difference for \f$ \log\alpha \f$).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return Vector of length \f$ k_c + k_z + 1 \f$: \f$ X^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{c} \f$,
     *         \f$ Z^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{z} \f$, and the central difference
     *         \f$ [\mathrm{NLL}(\log\alpha+\varepsilon)-\mathrm{NLL}(\log\alpha-\varepsilon)]/(2\varepsilon) \f$ with
     *         \f$ \varepsilon = 10^{-5} \f$. Non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_alpha = params(kc + kz);
        double alpha = std::exp(std::max(-10.0, std::min(10.0, log_alpha)));
        double inv_alpha = 1.0 / alpha;

        // linear predictors
        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        Eigen::VectorXd dNLL_deta_nb = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_zero = Eigen::VectorXd::Zero(n);

        // per-observation derivatives w.r.t. the linear predictors
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_deta_i);
            lambda_i = std::max(1e-5, lambda_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            double nb_zero = std::pow(1.0 / (1.0 + alpha * lambda_i), inv_alpha);

            if (y(i) < 0.5) {
                dNLL_deta_zero(i) = (1.0 / (1.0 - pi_i)) * dpi_deta_i;
                dNLL_deta_nb(i) = 0.0;
            } else {
                dNLL_deta_zero(i) = (-1.0 / pi_i) * dpi_deta_i;

                double truncation_factor = std::max(1.0 - nb_zero, 1e-12);
                double dnb_dlam = -nb_zero / (1.0 + alpha * lambda_i);
                double dNLL_dlam_trunc = -(1.0 / truncation_factor) * dnb_dlam;
                double dNLL_dlam_pmf = -(y(i) / lambda_i) + (y(i) + inv_alpha) * alpha / (1.0 + alpha * lambda_i);

                dNLL_deta_nb(i) = (dNLL_dlam_pmf + dNLL_dlam_trunc) * dlam_deta_i;
            }
        }

        // log(alpha) derivative by central finite difference of the NLL
        double eps = 1e-5;
        Eigen::VectorXd params_plus = params;  params_plus(kc + kz) += eps;
        Eigen::VectorXd params_minus = params; params_minus(kc + kz) -= eps;
        double dNLL_dlog_alpha = (this->operator()(params_plus) - this->operator()(params_minus)) / (2.0 * eps);

        Eigen::VectorXd grad(kc + kz + 1);
        grad.head(kc) = X.transpose() * dNLL_deta_nb;
        grad.segment(kc, kz) = Z.transpose() * dNLL_deta_zero;
        grad(kc + kz) = dNLL_dlog_alpha;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Approximate observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$.
     * @details Assembled as follows:
     * - \f$ H_{\gamma\gamma} = Z^\top\mathrm{diag}(L''u'^2 + L'u'')Z \f$ analytically, with
     *   \f$ L' = 1/(1-\pi), L''=1/(1-\pi)^2 \f$ for zeros and \f$ L'=-1/\pi, L''=1/\pi^2 \f$ for positives;
     * - for each \f$ j<k_c \f$, the column \f$ \partial\nabla\mathrm{NLL}/\partial\beta_j \f$ by a central difference
     *   (step `1e-5`) of `gradient()`; its \f$ \beta \f$ rows fill \f$ H_{\beta\beta}(\cdot,j) \f$ and its
     *   \f$ \log\alpha \f$ component fills both \f$ H(\log\alpha,j) \f$ and \f$ H(j,\log\alpha) \f$;
     * - \f$ H_{\log\alpha,\log\alpha} \f$ from the central difference of the \f$ \log\alpha \f$ component of `gradient()`;
     * - all blocks coupling \f$ \boldsymbol\gamma \f$ with \f$ \boldsymbol\beta \f$ or \f$ \log\alpha \f$ are set to 0.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return Square matrix of size \f$ k_c+k_z+1 \f$. Non-finite entries are set to 0 and diagonal entries that are
     *         exactly 0 are set to `1e-4` (negative diagonals are kept).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);

        Eigen::VectorXd eta_zero = Z * gamma;
        Eigen::VectorXd H_zero_zero = Eigen::VectorXd::Zero(n);

        // analytical Hessian weights of the hurdle part
        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0; double dpi_deta_i = 0.0; double d2pi_deta2_i = 0.0;
            evaluate_link_d2(eta_zero(i), zero_link, pi_i, dpi_deta_i, d2pi_deta2_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 0.5) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            } else {
                double dL_dpi = -1.0 / pi_i;
                double d2L_dpi2 = 1.0 / (pi_i * pi_i);
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            }
        }
        Eigen::MatrixXd H_gamma_gamma = Z.transpose() * H_zero_zero.asDiagonal() * Z;

        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(kc + kz + 1, kc + kz + 1);
        hess.block(kc, kc, kz, kz) = H_gamma_gamma;

        // finite differences of the gradient for the beta and log(alpha) blocks
        double eps = 1e-5;

        for (long j = 0; j < kc; ++j) {
            Eigen::VectorXd p_plus = params;  p_plus(j) += eps;
            Eigen::VectorXd p_minus = params; p_minus(j) -= eps;

            Eigen::VectorXd g_plus = this->gradient(p_plus);
            Eigen::VectorXd g_minus = this->gradient(p_minus);
            Eigen::VectorXd d2NLL_dtheta_dbetaj = (g_plus - g_minus) / (2.0 * eps);

            hess.block(0, j, kc, 1) = d2NLL_dtheta_dbetaj.head(kc);
            hess(kc + kz, j) = d2NLL_dtheta_dbetaj(kc + kz);
            hess(j, kc + kz) = d2NLL_dtheta_dbetaj(kc + kz);
        }

        Eigen::VectorXd p_alpha_plus = params;  p_alpha_plus(kc + kz) += eps;
        Eigen::VectorXd p_alpha_minus = params; p_alpha_minus(kc + kz) -= eps;

        Eigen::VectorXd g_alpha_plus = this->gradient(p_alpha_plus);
        Eigen::VectorXd g_alpha_minus = this->gradient(p_alpha_minus);

        hess(kc + kz, kc + kz) = (g_alpha_plus(kc + kz) - g_alpha_minus(kc + kz)) / (2.0 * eps);

        // gamma is orthogonal to the other parameters
        hess.block(0, kc, kc, kz) = Eigen::MatrixXd::Zero(kc, kz);
        hess.block(kc, 0, kz, kc) = Eigen::MatrixXd::Zero(kz, kc);
        hess.block(kc, kc + kz, kz, 1) = Eigen::MatrixXd::Zero(kz, 1);
        hess.block(kc + kz, kc, 1, kz) = Eigen::MatrixXd::Zero(1, kz);

        hess = hess.array().isFinite().select(hess, 0.0);
        for (long i = 0; i < kc + kz + 1; ++i) {
            if (hess(i, i) == 0.0) hess(i, i) = 1e-4;
        }

        return hess;
    }

};

/**
 * @brief Negative log-likelihood of the Gamma hurdle model (point mass at zero plus a Gamma density for the positive part).
 *
 * @details Two-part model for a non-negative continuous response: with probability \f$ 1-\pi_i \f$ the response is
 * exactly zero, otherwise it is positive with a Gamma density parameterised by mean \f$ \mu_i \f$ and shape
 * \f$ \kappa \f$ (variance \f$ \mu_i^2/\kappa \f$):
 * \f[ f(y;\mu,\kappa) = \frac{1}{\Gamma(\kappa)}\Bigl(\frac{\kappa}{\mu}\Bigr)^{\kappa} y^{\kappa-1} e^{-\kappa y/\mu}. \f]
 * - the *mean* predictor \f$ \eta^{m}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ gives
 *   \f$ \mu_i = g_m^{-1}(\eta^{m}_i) \f$ (`mean_link`, default `Log`);
 * - the *hurdle* predictor \f$ \eta^{z}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ gives
 *   \f$ \pi_i = g_z^{-1}(\eta^{z}_i) = P(y_i>0) \f$ (`zero_link`, default `Logit`; probability of a **positive**
 *   response);
 * - the shape \f$ \kappa = \exp(\log\kappa) \f$ is constant across observations.
 *
 * Links are evaluated with the class's own static `evaluate_link()` / `evaluate_link_d2()`, which behave like the free
 * functions except that `LinkType::Inverse` uses the smooth positive inverse (`evaluate_positive_inverse_d2()`), for
 * **both** `mean_link` and `zero_link`.
 *
 * ### Parameter vector layout
 * | Index             | Meaning                                       | Scale                          |
 * |-------------------|-----------------------------------------------|--------------------------------|
 * | `0 .. kc-1`       | mean coefficients \f$ \boldsymbol\beta \f$    | mean linear predictor          |
 * | `kc .. kc+kz-1`   | hurdle coefficients \f$ \boldsymbol\gamma \f$ | hurdle linear predictor        |
 * | `kc+kz`           | \f$ \log\kappa \f$                            | log; clamped to \f$ [-10,10] \f$ before exponentiation |
 *
 * with \f$ k_c \f$ = `X.cols()`, \f$ k_z \f$ = `Z.cols()`; total length \f$ k_c+k_z+1 \f$.
 *
 * ### Log-likelihood
 * With \f$ \mu_i \ge 10^{-8} \f$ and \f$ \pi_i \in [10^{-12}, 1-10^{-12}] \f$:
 * - zero (\f$ y_i < 10^{-12} \f$): \f$ \ell_i = \log(1-\pi_i) \f$;
 * - positive: \f$ \ell_i = \log\pi_i + \kappa\log(\kappa/\mu_i) - \log\Gamma(\kappa) + (\kappa-1)\log y_i
 *   - \kappa y_i/\mu_i \f$, with \f$ y_i \f$ floored at \f$ 10^{-300} \f$ inside the logarithm and the product.
 *
 * The functor returns \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite. No weights are used.
 *
 * ### Derivatives
 * The NLL separates in \f$ \pi_i \f$ and \f$ (\mu_i,\kappa) \f$.
 * - Hurdle part (analytical): \f$ \pi'/(1-\pi_i) \f$ for zeros, \f$ -\pi'/\pi_i \f$ for positives.
 * - Mean part (analytical, positive observations): \f$ \partial\mathrm{NLL}_i/\partial\mu_i = \kappa(\mu_i-y_i)/\mu_i^2 \f$,
 *   multiplied by \f$ d\mu_i/d\eta^m_i \f$.
 * - \f$ \partial\mathrm{NLL}/\partial\log\kappa \f$ is a central finite difference of `operator()` (step `1e-5`).
 *
 * The Hessian is hybrid: the \f$ \gamma\gamma \f$ block is analytical; the \f$ \beta \f$ and \f$ \log\kappa \f$ blocks are
 * central finite differences (step `1e-5`) of `gradient()`; cross blocks between \f$ \boldsymbol\gamma \f$ and the
 * other parameters are exactly zero (separability).
 *
 * ### Numerical safeguards
 * - \f$ \log\kappa \in [-10,10] \f$, \f$ \mu_i \ge 10^{-8} \f$, \f$ \pi_i \in [10^{-12},1-10^{-12}] \f$,
 *   \f$ y_i \ge 10^{-300} \f$ in the positive branch. Derivatives of the clamps are ignored.
 * - `operator()` returns `1e10` when not finite; `gradient()` replaces non-finite entries by 0;
 *   `hessian()` replaces non-finite entries by 0 and diagonal entries that are \f$ \le 0 \f$ by `1e-4`.
 * - The Hessian is not ridge-regularised or explicitly symmetrised (symmetric up to finite-difference error).
 *
 * @note The object stores **non-owning references** to the design matrices and the response: they must outlive the
 *       functor. No validation is performed: \f$ y_i \ge 0 \f$ is required (\f$ y_i < 10^{-12} \f$ is treated as zero)
 *       and `params` must have length \f$ k_c + k_z + 1 \f$.
 * @note `hessian()` costs \f$ 2(k_c+1) \f$ gradient evaluations, each of which costs two `operator()` evaluations
 *       (for \f$ \log\kappa \f$).
 * @see HurdleLogNormalLoss, HurdlePoissonLoss, HurdleNBLoss, HurdleBetaLoss
 *
 * @ingroup glm_ll_hurdle
 */
struct HurdleGammaLoss {
    const Eigen::MatrixXd& X;  ///< Mean design matrix (\f$ n \times k_c \f$) of the Gamma-mean linear predictor.
    const Eigen::MatrixXd& Z;  ///< Hurdle design matrix (\f$ n \times k_z \f$) of the linear predictor for \f$ P(y>0) \f$.
    const Eigen::VectorXd& y;  ///< Observed non-negative responses (length \f$ n \f$); values below \f$ 10^{-12} \f$ count as zeros.
    LinkType mean_link;  ///< Link whose inverse maps the mean linear predictor to the Gamma mean \f$ \mu \f$.
    LinkType zero_link;  ///< Link whose inverse maps the hurdle linear predictor to \f$ \pi = P(y>0) \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Mean design matrix \f$ X \f$ (\f$ n \times k_c \f$).
     * @param exog_zero Hurdle design matrix \f$ Z \f$ (\f$ n \times k_z \f$).
     * @param endog Non-negative response \f$ y \f$ (length \f$ n \f$).
     * @param blink Mean link type (default `LinkType::Log`); its inverse gives the Gamma mean \f$ \mu \f$.
     * @param zlink Hurdle link type (default `LinkType::Logit`); its inverse gives \f$ P(y>0) \f$.
     * @pre `exog`, `exog_zero` and `endog` have the same number of rows and outlive the functor.
     */
    HurdleGammaLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_zero, const Eigen::VectorXd& endog,
                    LinkType blink = LinkType::Log, LinkType zlink = LinkType::Logit)
        : X(exog), Z(exog_zero), y(endog), mean_link(blink), zero_link(zlink) {}

    /**
     * @brief Inverse link value and first derivative (class-local override of the free function).
     * @details Forwards to the class's `evaluate_link_d2()` discarding the second derivative, so that
     * `LinkType::Inverse` uses the smooth positive inverse.
     * @param eta Linear predictor value \f$ \eta \f$.
     * @param link Link type.
     * @param val Output: inverse link value \f$ g^{-1}(\eta) \f$.
     * @param dval_deta Output: derivative \f$ d g^{-1}/d\eta \f$.
     */
    static inline void evaluate_link(double eta, LinkType link, double& val, double& dval_deta) {
        double d2 = 0.0;
        evaluate_link_d2(eta, link, val, dval_deta, d2);
    }

    /**
     * @brief Inverse link value and first two derivatives with a smooth positive inverse link.
     * @details For `LinkType::Inverse` it calls `evaluate_positive_inverse_d2()`, i.e.
     * \f$ g^{-1}(\eta) = 1/\eta_+ \f$ with \f$ \eta_+ = \tfrac12\bigl(\eta + \sqrt{\eta^2 + 4d^2}\bigr) \f$, \f$ d=0.1 \f$ (a
     * smooth, strictly positive floor of \f$ \eta \f$, so the value is positive and \f$ C^\infty \f$). Every other link is
     * delegated to the free function `::glm::evaluate_link_d2()`.
     * @param eta Linear predictor value \f$ \eta \f$.
     * @param link Link type.
     * @param val Output: inverse link value \f$ g^{-1}(\eta) \f$.
     * @param dval_deta Output: first derivative \f$ d g^{-1}/d\eta \f$.
     * @param d2val_deta2 Output: second derivative \f$ d^2 g^{-1}/d\eta^2 \f$.
     */
    static inline void evaluate_link_d2(double eta, LinkType link, double& val, double& dval_deta, double& d2val_deta2) {
        if (link == LinkType::Inverse) {
            evaluate_positive_inverse_d2(eta, val, dval_deta, d2val_deta2);
            return;
        }
        ::glm::evaluate_link_d2(eta, link, val, dval_deta, d2val_deta2);
    }

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\kappa) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_shape = params(kc + kz);
        double shape = std::exp(std::max(-10.0, std::min(10.0, log_shape)));

        // linear predictors
        Eigen::VectorXd eta_mean = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        double total_neg_ll = 0.0;
        // accumulate per-observation NLL terms
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta_mean(i), mean_link, mu_i, dmu_deta_i);
            mu_i = std::max(1e-8, mu_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            double log_prob_i = 0.0;
            if (y(i) < 1e-12) {
                log_prob_i = std::log(1.0 - pi_i);
            } else {
                double yi = std::max(y(i), 1e-300);
                double log_gamma_pdf = shape * std::log(shape / mu_i) - std::lgamma(shape)
                                      + (shape - 1.0) * std::log(yi) - shape * yi / mu_i;
                log_prob_i = std::log(pi_i) + log_gamma_pdf;
            }
            total_neg_ll -= log_prob_i;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Gradient of the negative log-likelihood (analytical for \f$ \beta,\gamma \f$, finite difference for \f$ \log\kappa \f$).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\kappa) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return Vector of length \f$ k_c + k_z + 1 \f$: \f$ X^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{m} \f$,
     *         \f$ Z^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{z} \f$ and the central difference of `operator()`
     *         with respect to \f$ \log\kappa \f$ (step `1e-5`). Non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_shape = params(kc + kz);
        double shape = std::exp(std::max(-10.0, std::min(10.0, log_shape)));

        // linear predictors
        Eigen::VectorXd eta_mean = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        Eigen::VectorXd dNLL_deta_mean = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_zero = Eigen::VectorXd::Zero(n);

        // per-observation derivatives w.r.t. the linear predictors
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta_mean(i), mean_link, mu_i, dmu_deta_i);
            mu_i = std::max(1e-8, mu_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 1e-12) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;
                dNLL_deta_mean(i) = 0.0;
            } else {
                double dL_dpi = -1.0 / pi_i;
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;

                double yi = std::max(y(i), 1e-300);
                double dNLL_dmu = shape * (mu_i - yi) / (mu_i * mu_i);
                dNLL_deta_mean(i) = dNLL_dmu * dmu_deta_i;
            }
        }

        Eigen::VectorXd grad(kc + kz + 1);
        grad.head(kc) = X.transpose() * dNLL_deta_mean;
        grad.segment(kc, kz) = Z.transpose() * dNLL_deta_zero;

        // log(shape) derivative by central finite difference of the NLL
        double eps = 1e-5;
        Eigen::VectorXd params_plus = params;  params_plus(kc + kz) += eps;
        Eigen::VectorXd params_minus = params; params_minus(kc + kz) -= eps;
        grad(kc + kz) = (this->operator()(params_plus) - this->operator()(params_minus)) / (2.0 * eps);

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Approximate observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\kappa) \f$.
     * @details Assembled as follows:
     * - \f$ H_{\gamma\gamma} = Z^\top\mathrm{diag}(L''\pi'^2 + L'\pi'')Z \f$ analytically, with
     *   \f$ L' = 1/(1-\pi), L''=1/(1-\pi)^2 \f$ for zeros and \f$ L'=-1/\pi, L''=1/\pi^2 \f$ for positives
     *   (link derivatives from the class's `evaluate_link_d2()`);
     * - for each \f$ j<k_c \f$, a central difference (step `1e-5`) of `gradient()` with respect to \f$ \beta_j \f$ fills
     *   \f$ H_{\beta\beta}(\cdot,j) \f$ and both \f$ H(\log\kappa,j) \f$ and \f$ H(j,\log\kappa) \f$;
     * - \f$ H_{\log\kappa,\log\kappa} \f$ from the central difference of the \f$ \log\kappa \f$ component of `gradient()`;
     * - all blocks coupling \f$ \boldsymbol\gamma \f$ with \f$ \boldsymbol\beta \f$ or \f$ \log\kappa \f$ are set to 0.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\kappa) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return Square matrix of size \f$ k_c+k_z+1 \f$. Non-finite entries are set to 0 and diagonal entries that are
     *         \f$ \le 0 \f$ are set to `1e-4`.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);

        Eigen::VectorXd eta_zero = Z * gamma;
        Eigen::VectorXd H_zero_zero = Eigen::VectorXd::Zero(n);

        // analytical Hessian weights of the hurdle part
        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0; double dpi_deta_i = 0.0; double d2pi_deta2_i = 0.0;
            evaluate_link_d2(eta_zero(i), zero_link, pi_i, dpi_deta_i, d2pi_deta2_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 1e-12) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            } else {
                double dL_dpi = -1.0 / pi_i;
                double d2L_dpi2 = 1.0 / (pi_i * pi_i);
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            }
        }
        Eigen::MatrixXd H_gamma_gamma = Z.transpose() * H_zero_zero.asDiagonal() * Z;

        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(kc + kz + 1, kc + kz + 1);
        hess.block(kc, kc, kz, kz) = H_gamma_gamma;

        // finite differences of the gradient for the beta and log(shape) blocks
        double eps = 1e-5;

        for (long j = 0; j < kc; ++j) {
            Eigen::VectorXd p_plus = params;  p_plus(j) += eps;
            Eigen::VectorXd p_minus = params; p_minus(j) -= eps;

            Eigen::VectorXd g_plus = this->gradient(p_plus);
            Eigen::VectorXd g_minus = this->gradient(p_minus);
            Eigen::VectorXd d2NLL_dtheta_dbetaj = (g_plus - g_minus) / (2.0 * eps);

            hess.block(0, j, kc, 1) = d2NLL_dtheta_dbetaj.head(kc);
            hess(kc + kz, j) = d2NLL_dtheta_dbetaj(kc + kz);
            hess(j, kc + kz) = d2NLL_dtheta_dbetaj(kc + kz);
        }

        Eigen::VectorXd p_shape_plus = params;  p_shape_plus(kc + kz) += eps;
        Eigen::VectorXd p_shape_minus = params; p_shape_minus(kc + kz) -= eps;

        Eigen::VectorXd g_shape_plus = this->gradient(p_shape_plus);
        Eigen::VectorXd g_shape_minus = this->gradient(p_shape_minus);

        hess(kc + kz, kc + kz) = (g_shape_plus(kc + kz) - g_shape_minus(kc + kz)) / (2.0 * eps);

        // gamma is orthogonal to the other parameters
        hess.block(0, kc, kc, kz).setZero();
        hess.block(kc, 0, kz, kc).setZero();
        hess.block(kc, kc + kz, kz, 1).setZero();
        hess.block(kc + kz, kc, 1, kz).setZero();

        hess = hess.array().isFinite().select(hess, 0.0);
        for (long i = 0; i < kc + kz + 1; ++i) {
            if (hess(i, i) <= 0.0) hess(i, i) = 1e-4;
        }

        return hess;
    }
};

/**
 * @brief Negative log-likelihood of the Log-Normal hurdle model (point mass at zero plus a Log-Normal density for the positive part).
 *
 * @details Two-part model for a non-negative continuous response: with probability \f$ 1-\pi_i \f$ the response is
 * exactly zero, otherwise it is positive with \f$ \log y_i \sim \mathcal{N}(m_i,\sigma^2) \f$:
 * \f[ f(y;m,\sigma) = \frac{1}{y\,\sigma\sqrt{2\pi}}\exp\Bigl\{-\tfrac12\Bigl(\frac{\log y - m}{\sigma}\Bigr)^2\Bigr\}. \f]
 * - the *location* predictor \f$ \eta^{m}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ is mapped by the inverse link
 *   (`mean_link`, default `Identity`) to the **location of \f$ \log y \f$**, \f$ m_i = g_m^{-1}(\eta^{m}_i) \f$ (it is
 *   not the mean of \f$ y \f$; the code names it `log_mu_i`);
 * - the *hurdle* predictor \f$ \eta^{z}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ gives
 *   \f$ \pi_i = g_z^{-1}(\eta^{z}_i) = P(y_i>0) \f$ (`zero_link`, default `Logit`; probability of a **positive**
 *   response);
 * - \f$ \sigma = e^{\log\sigma} \f$ is constant across observations.
 *
 * Both links use the free function `evaluate_link()`.
 *
 * ### Parameter vector layout
 * | Index             | Meaning                                       | Scale                          |
 * |-------------------|-----------------------------------------------|--------------------------------|
 * | `0 .. kc-1`       | location coefficients \f$ \boldsymbol\beta \f$ | location linear predictor     |
 * | `kc .. kc+kz-1`   | hurdle coefficients \f$ \boldsymbol\gamma \f$ | hurdle linear predictor        |
 * | `kc+kz`           | \f$ \log\sigma \f$                            | log (not clamped)              |
 *
 * with \f$ k_c \f$ = `X.cols()`, \f$ k_z \f$ = `Z.cols()`; total length \f$ k_c+k_z+1 \f$.
 *
 * ### Log-likelihood
 * With \f$ \pi_i \in [10^{-12}, 1-10^{-12}] \f$, \f$ z_i = (\log y_i - m_i)/\sigma \f$ and
 * \f$ \log y_i \f$ evaluated as \f$ \log\max(y_i,10^{-300}) \f$:
 * - zero (\f$ y_i < 10^{-12} \f$): \f$ \ell_i = \log(1-\pi_i) \f$;
 * - positive: \f$ \ell_i = \log\pi_i - \log\sigma - \tfrac12\log(2\pi) - \tfrac12 z_i^2 - \log y_i \f$ (the constant
 *   \f$ \tfrac12\log(2\pi) = 0.9189385332046727 \f$ and the Jacobian term \f$ -\log y_i \f$ are kept).
 *
 * The functor returns \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite. No weights are used.
 *
 * ### Derivatives
 * The NLL separates in \f$ \pi_i \f$ and \f$ (m_i,\sigma) \f$. Per observation:
 * - hurdle part: \f$ \pi'/(1-\pi_i) \f$ for zeros, \f$ -\pi'/\pi_i \f$ for positives (with respect to \f$ \eta^z_i \f$);
 * - positive observations: \f$ \partial\mathrm{NLL}_i/\partial m_i = -z_i/\sigma \f$, hence
 *   \f$ \partial\mathrm{NLL}_i/\partial\eta^m_i = -z_i\,(dm_i/d\eta^m_i)/\sigma \f$, and
 *   \f$ \partial\mathrm{NLL}_i/\partial\log\sigma = 1 - z_i^2 \f$ (zero observations contribute 0).
 *
 * The gradient is fully analytical. The Hessian is hybrid: the \f$ \gamma\gamma \f$ block is analytical, the blocks
 * involving \f$ \beta \f$ and \f$ \log\sigma \f$ are central finite differences (step `1e-5`) of the analytical
 * `gradient()`, symmetrised by averaging; cross blocks between \f$ \boldsymbol\gamma \f$ and the other parameters
 * are exactly zero (separability).
 *
 * ### Numerical safeguards
 * - \f$ \pi_i \in [10^{-12},1-10^{-12}] \f$ and \f$ y_i \ge 10^{-300} \f$ inside the logarithm; the location and
 *   \f$ \log\sigma \f$ are **not** clamped (apart from the cap of the `Log` link inside `evaluate_link()`).
 *   Derivatives of the clamps are ignored.
 * - `operator()` returns `1e10` when not finite; `gradient()` replaces non-finite entries by 0;
 *   `hessian()` replaces non-finite entries by 0 and diagonal entries that are \f$ \le 0 \f$ by `1e-4`.
 * - No ridge regularisation or PSD projection is applied.
 *
 * @note The object stores **non-owning references** to the design matrices and the response: they must outlive the
 *       functor. No validation is performed: \f$ y_i \ge 0 \f$ is required (\f$ y_i < 10^{-12} \f$ is treated as zero)
 *       and `params` must have length \f$ k_c + k_z + 1 \f$.
 * @note `hessian()` costs \f$ 2(k_c+1) \f$ gradient evaluations.
 * @see HurdleGammaLoss, LogNormalLoss, HurdlePoissonLoss, HurdleNBLoss
 *
 * @ingroup glm_ll_hurdle
 */
struct HurdleLogNormalLoss {
    const Eigen::MatrixXd& X;  ///< Location design matrix (\f$ n \times k_c \f$) of the linear predictor for the location of \f$ \log y \f$.
    const Eigen::MatrixXd& Z;  ///< Hurdle design matrix (\f$ n \times k_z \f$) of the linear predictor for \f$ P(y>0) \f$.
    const Eigen::VectorXd& y;  ///< Observed non-negative responses (length \f$ n \f$); values below \f$ 10^{-12} \f$ count as zeros.
    LinkType mean_link;  ///< Link whose inverse maps the location linear predictor to the location \f$ m \f$ of \f$ \log y \f$ (default `Identity`).
    LinkType zero_link;  ///< Link whose inverse maps the hurdle linear predictor to \f$ \pi = P(y>0) \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Location design matrix \f$ X \f$ (\f$ n \times k_c \f$).
     * @param exog_zero Hurdle design matrix \f$ Z \f$ (\f$ n \times k_z \f$).
     * @param endog Non-negative response \f$ y \f$ (length \f$ n \f$).
     * @param blink Location link type (default `LinkType::Identity`); its inverse gives the location of \f$ \log y \f$.
     * @param zlink Hurdle link type (default `LinkType::Logit`); its inverse gives \f$ P(y>0) \f$.
     * @pre `exog`, `exog_zero` and `endog` have the same number of rows and outlive the functor.
     */
    HurdleLogNormalLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_zero, const Eigen::VectorXd& endog,
                        LinkType blink = LinkType::Identity, LinkType zlink = LinkType::Logit)
        : X(exog), Z(exog_zero), y(endog), mean_link(blink), zero_link(zlink) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\sigma) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_sigma = params(kc + kz);
        double sigma = std::exp(log_sigma);

        // linear predictors
        Eigen::VectorXd eta_mean = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        const double CONST_NORMAL = 0.9189385332046727;
        double total_neg_ll = 0.0;

        // accumulate per-observation NLL terms
        for (long i = 0; i < n; ++i) {
            double log_mu_i = 0.0; double dlogmu_deta_i = 0.0;
            evaluate_link(eta_mean(i), mean_link, log_mu_i, dlogmu_deta_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            double log_prob_i = 0.0;
            if (y(i) < 1e-12) {
                log_prob_i = std::log(1.0 - pi_i);
            } else {
                double log_yi = std::log(std::max(y(i), 1e-300));
                double zi = (log_yi - log_mu_i) / sigma;
                double log_pdf_i = -log_sigma - CONST_NORMAL - 0.5 * zi * zi - log_yi;
                log_prob_i = std::log(pi_i) + log_pdf_i;
            }
            total_neg_ll -= log_prob_i;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\sigma) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return Vector of length \f$ k_c + k_z + 1 \f$: \f$ X^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{m} \f$,
     *         \f$ Z^\top \partial\mathrm{NLL}/\partial\boldsymbol\eta^{z} \f$ and
     *         \f$ \sum_{i:\,y_i\ge 10^{-12}} (1 - z_i^2) \f$. Non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_sigma = params(kc + kz);
        double sigma = std::exp(log_sigma);

        // linear predictors
        Eigen::VectorXd eta_mean = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        Eigen::VectorXd dNLL_deta_mean = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_zero = Eigen::VectorXd::Zero(n);
        double dNLL_dlogsigma = 0.0;

        // per-observation derivatives w.r.t. the linear predictors
        for (long i = 0; i < n; ++i) {
            double log_mu_i = 0.0; double dlogmu_deta_i = 0.0;
            evaluate_link(eta_mean(i), mean_link, log_mu_i, dlogmu_deta_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 1e-12) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;
                dNLL_deta_mean(i) = 0.0;
            } else {
                double dL_dpi = -1.0 / pi_i;
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;

                double log_yi = std::log(std::max(y(i), 1e-300));
                double zi = (log_yi - log_mu_i) / sigma;

                double dNLL_dlogmu_i = zi * (-1.0 / sigma);
                dNLL_deta_mean(i) = dNLL_dlogmu_i * dlogmu_deta_i;

                dNLL_dlogsigma += 1.0 - zi * zi;
            }
        }

        Eigen::VectorXd grad(kc + kz + 1);
        grad.head(kc) = X.transpose() * dNLL_deta_mean;
        grad.segment(kc, kz) = Z.transpose() * dNLL_deta_zero;
        grad(kc + kz) = dNLL_dlogsigma;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Approximate observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\sigma) \f$.
     * @details Assembled as follows:
     * - \f$ H_{\gamma\gamma} = Z^\top\mathrm{diag}(L''\pi'^2 + L'\pi'')Z \f$ analytically, with
     *   \f$ L' = 1/(1-\pi), L''=1/(1-\pi)^2 \f$ for zeros and \f$ L'=-1/\pi, L''=1/\pi^2 \f$ for positives;
     * - for each \f$ j<k_c \f$, a central difference (step `1e-5`) of the analytical `gradient()` with respect to
     *   \f$ \beta_j \f$ fills the \f$ \beta \f$ rows of column \f$ j \f$ and the \f$ \log\sigma \f$ entry;
     * - the central difference with respect to \f$ \log\sigma \f$ gives \f$ H_{\log\sigma,\log\sigma} \f$ and a second
     *   estimate of the cross terms; the cross entries \f$ H(\log\sigma,j) = H(j,\log\sigma) \f$ are the average of the two
     *   estimates;
     * - the \f$ \beta\beta \f$ block is symmetrised by averaging \f$ H(r,c) \f$ and \f$ H(c,r) \f$;
     * - all blocks coupling \f$ \boldsymbol\gamma \f$ with \f$ \boldsymbol\beta \f$ or \f$ \log\sigma \f$ are set to 0.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\sigma) \f$ of length \f$ k_c + k_z + 1 \f$.
     * @return Symmetric square matrix of size \f$ k_c+k_z+1 \f$. Non-finite entries are set to 0 and diagonal entries that
     *         are \f$ \le 0 \f$ are set to `1e-4`.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);

        Eigen::VectorXd eta_zero = Z * gamma;
        Eigen::VectorXd H_zero_zero = Eigen::VectorXd::Zero(n);

        // analytical Hessian weights of the hurdle part
        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0; double dpi_deta_i = 0.0; double d2pi_deta2_i = 0.0;
            evaluate_link_d2(eta_zero(i), zero_link, pi_i, dpi_deta_i, d2pi_deta2_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 1e-12) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            } else {
                double dL_dpi = -1.0 / pi_i;
                double d2L_dpi2 = 1.0 / (pi_i * pi_i);
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            }
        }
        Eigen::MatrixXd H_gamma_gamma = Z.transpose() * H_zero_zero.asDiagonal() * Z;

        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(kc + kz + 1, kc + kz + 1);
        hess.block(kc, kc, kz, kz) = H_gamma_gamma;

        // finite differences of the gradient for the beta and log(sigma) blocks
        double eps = 1e-5;

        for (long j = 0; j < kc; ++j) {
            Eigen::VectorXd p_plus = params;  p_plus(j) += eps;
            Eigen::VectorXd p_minus = params; p_minus(j) -= eps;

            Eigen::VectorXd g_plus = this->gradient(p_plus);
            Eigen::VectorXd g_minus = this->gradient(p_minus);
            Eigen::VectorXd d2NLL_dtheta_dbetaj = (g_plus - g_minus) / (2.0 * eps);

            hess.block(0, j, kc, 1) = d2NLL_dtheta_dbetaj.head(kc);
            hess(kc + kz, j) = d2NLL_dtheta_dbetaj(kc + kz);
            hess(j, kc + kz) = d2NLL_dtheta_dbetaj(kc + kz);
        }

        {
            Eigen::VectorXd p_sigma_plus = params;  p_sigma_plus(kc + kz) += eps;
            Eigen::VectorXd p_sigma_minus = params; p_sigma_minus(kc + kz) -= eps;

            Eigen::VectorXd g_sigma_plus = this->gradient(p_sigma_plus);
            Eigen::VectorXd g_sigma_minus = this->gradient(p_sigma_minus);
            Eigen::VectorXd d2NLL_dtheta_dlogsigma = (g_sigma_plus - g_sigma_minus) / (2.0 * eps);

            hess(kc + kz, kc + kz) = d2NLL_dtheta_dlogsigma(kc + kz);

            for (long j = 0; j < kc; ++j) {
                double avg_cross = 0.5 * (hess(kc + kz, j) + d2NLL_dtheta_dlogsigma(j));
                hess(kc + kz, j) = avg_cross;
                hess(j, kc + kz) = avg_cross;
            }
        }

        // symmetrise the beta-beta block
        for (long r = 0; r < kc; ++r) {
            for (long c = 0; c < r; ++c) {
                double avg_beta = 0.5 * (hess(r, c) + hess(c, r));
                hess(r, c) = avg_beta;
                hess(c, r) = avg_beta;
            }
        }

        // gamma is orthogonal to the other parameters
        hess.block(0, kc, kc, kz).setZero();
        hess.block(kc, 0, kz, kc).setZero();
        hess.block(kc, kc + kz, kz, 1).setZero();
        hess.block(kc + kz, kc, 1, kz).setZero();

        hess = hess.array().isFinite().select(hess, Eigen::MatrixXd::Zero(kc + kz + 1, kc + kz + 1));
        for (long i = 0; i < kc + kz + 1; ++i) {
            if (hess(i, i) <= 0.0) {
                hess(i, i) = 1e-4;
            }
        }

        return hess;
    }

};

/**
 * @brief Negative log-likelihood of a hurdle model with a zero-truncated geometric count component.
 *
 * @details Models a non-negative integer count \f$ y_i \in \{0,1,2,\dots\} \f$ in two parts. A binary "hurdle"
 * part decides whether the count is positive, and a zero-truncated geometric distribution describes the positive
 * counts:
 * \f[ P(y_i = 0) = 1 - \pi_i, \qquad
 *     P(y_i = k) = \pi_i\,\frac{\mu_i^{\,k-1}}{(1+\mu_i)^{k}}, \quad k \ge 1. \f]
 * The second factor is the pmf \f$ \mu^k/(1+\mu)^{k+1} \f$ of a geometric variable with mean \f$ \mu \f$
 * (failures-before-success parametrisation) conditioned on \f$ k \ge 1 \f$; the truncated component has mean
 * \f$ 1 + \mu_i \f$.
 * - Hurdle part: the linear predictor \f$ \eta^{(z)}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ is mapped by the
 *   inverse link `zero_link` to \f$ \pi_i = P(y_i > 0) \f$ (probability of **crossing** the hurdle, not the
 *   probability of a zero). The default link is `Logit`.
 * - Count part: the linear predictor \f$ \eta^{(c)}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ is mapped by the inverse
 *   link `count_link` to \f$ \mu_i \f$ (default `Log`, i.e. \f$ \mu_i = e^{\eta^{(c)}_i} \f$).
 *
 * ### Parameter vector layout
 * | Index                 | Meaning                                              | Scale                            |
 * |-----------------------|------------------------------------------------------|----------------------------------|
 * | `0 .. kc-1`           | count coefficients \f$ \boldsymbol\beta \f$ (columns of `X`) | linear predictor of \f$ \mu \f$ |
 * | `kc .. kc+kz-1`       | hurdle coefficients \f$ \boldsymbol\gamma \f$ (columns of `Z`) | linear predictor of \f$ \pi \f$ |
 *
 * There are no dispersion/shape parameters. The vector must have exactly \f$ kc + kz \f$ entries
 * (`params.head(kc)` and `params.tail(kz)` are used).
 *
 * ### Log-likelihood
 * Observations with \f$ y_i < 0.5 \f$ are treated as zeros, all others as positive counts (no integrality check):
 * - zero: \f$ \ell_i = \log(1 - \pi_i) \f$;
 * - positive: \f$ \ell_i = \log\pi_i + (y_i - 1)\log\mu_i - y_i\log(1+\mu_i) \f$.
 *
 * The pmf has no additive constant, so nothing is dropped. The functor returns \f$ -\sum_i \ell_i \f$.
 *
 * ### Derivatives
 * The likelihood is additively separable in \f$ \boldsymbol\beta \f$ and \f$ \boldsymbol\gamma \f$, so
 * the gradient is \f$ (X^\top u,\; Z^\top v) \f$ with \f$ u_i = \partial\mathrm{NLL}_i/\partial\eta^{(c)}_i \f$ and
 * \f$ v_i = \partial\mathrm{NLL}_i/\partial\eta^{(z)}_i \f$:
 * - zero: \f$ v_i = \pi_i'/(1-\pi_i) \f$, \f$ u_i = 0 \f$;
 * - positive: \f$ v_i = -\pi_i'/\pi_i \f$ and
 *   \f$ u_i = \big[-(y_i-1)/\mu_i + y_i/(1+\mu_i)\big]\mu_i' \f$ (for the `Log` link this reduces to
 *   \f$ y_i\,\mathrm{sigmoid}(\eta^{(c)}_i) - (y_i - 1) \f$),
 *
 * where \f$ \pi' = d\pi/d\eta^{(z)} \f$ and \f$ \mu' = d\mu/d\eta^{(c)} \f$. The Hessian is the exact observed
 * Hessian, **block diagonal** (the \f$ \beta \f$-\f$ \gamma \f$ block is identically zero because of the
 * separability): \f$ H_{\beta\beta} = X^\top\mathrm{diag}(a)\,X \f$ and
 * \f$ H_{\gamma\gamma} = Z^\top\mathrm{diag}(b)\,Z \f$, with link-curvature terms included
 * (\f$ a_i = \mathrm{NLL}_{\mu\mu}\mu'^2 + \mathrm{NLL}_\mu\mu'' \f$, \f$ b_i = \mathrm{NLL}_{\pi\pi}\pi'^2 +
 * \mathrm{NLL}_\pi\pi'' \f$). It is neither regularised nor finite-masked.
 *
 * ### Numerical safeguards
 * - `operator()` returns `1e10` if the sum is not finite; `gradient()` replaces non-finite entries by 0.
 * - \f$ \pi_i \f$ is obtained with `BetaMath::evaluate_link_hessian` (which clamps \f$ \eta \f$ per link, e.g.
 *   \f$ [-30,30] \f$ for `Logit`, and clamps \f$ \pi \f$ to \f$ [10^{-14}, 1-10^{-14}] \f$ for the bounded links) and is
 *   then clamped again to \f$ [10^{-12}, 1-10^{-12}] \f$ by `barrier()`; \f$ \pi' \f$ and \f$ \pi'' \f$ are **not**
 *   adjusted after this clamp.
 * - \f$ \mu_i \f$: see `count_mean()` (clamp of \f$ \eta \f$ for `Log`, floor \f$ 10^{-8} \f$ otherwise; derivatives
 *   are not adjusted).
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y`; they must outlive the functor. No
 *       validation is performed: `X`, `Z`, `y` must have the same number of rows, and \f$ y_i \ge 0 \f$ (positive values
 *       should be integers \f$ \ge 1 \f$). Observations with \f$ y_i < 0.5 \f$ contribute nothing to the count block of
 *       the gradient and Hessian.
 * @note The static helper `log1p_exp()` is not used by any method of this struct.
 * @see HurdlePoissonLoss, HurdleNBLoss, TruncatedGeometricLoss for the other hurdle / truncated count models.
 *
 * @ingroup glm_ll_hurdle
 */
struct HurdleGeometricLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the count (geometric mean) model (\f$ n \times kc \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the hurdle (probability of a positive count) model (\f$ n \times kz \f$).
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$); values below 0.5 are treated as zeros.
    LinkType count_link;  ///< Inverse link mapping \f$ \eta^{(c)} \f$ to the geometric mean parameter \f$ \mu \f$.
    LinkType zero_link;  ///< Inverse link mapping \f$ \eta^{(z)} \f$ to the hurdle probability \f$ \pi = P(y>0) \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Count design matrix \f$ X \f$ (\f$ n \times kc \f$).
     * @param exog_zero Hurdle design matrix \f$ Z \f$ (\f$ n \times kz \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$), non-negative.
     * @param blink Link whose inverse maps \f$ \eta^{(c)} \f$ to the geometric mean \f$ \mu \f$ (default `Log`).
     * @param zlink Link whose inverse maps \f$ \eta^{(z)} \f$ to \f$ \pi = P(y>0) \f$ (default `Logit`).
     * @pre `exog`, `exog_zero` and `endog` have the same number of rows and outlive the functor.
     */
    HurdleGeometricLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_zero, const Eigen::VectorXd& endog,
                        LinkType blink = LinkType::Log, LinkType zlink = LinkType::Logit)
        : X(exog), Z(exog_zero), y(endog), count_link(blink), zero_link(zlink) {}

    /**
     * @brief Overflow-safe \f$ \log(1 + e^{\eta}) \f$ (softplus).
     * @details Returns \f$ \eta \f$ for \f$ \eta > 30 \f$, \f$ e^{\eta} \f$ for \f$ \eta < -30 \f$ and
     * \f$ \mathrm{log1p}(e^{\eta}) \f$ otherwise. Not used by any method of this struct.
     * @param eta Argument.
     * @return Approximation of \f$ \log(1+e^{\eta}) \f$.
     */
    static inline double log1p_exp(double eta) {
        if (eta > 30.0) return eta;
        if (eta < -30.0) return std::exp(eta);
        return std::log1p(std::exp(eta));
    }

    /**
     * @brief Logistic function with hard saturation.
     * @param eta Argument.
     * @return The value 1 if \f$ \eta > 30 \f$, 0 if \f$ \eta < -30 \f$, otherwise
     *         \f$ 1/(1+e^{-\eta}) \f$. Used in the closed-form `Log`-link derivatives, where
     *         \f$ \mu/(1+\mu) = \mathrm{sigmoid}(\eta) \f$.
     */
    static inline double sigmoid(double eta) {
        if (eta > 30.0) return 1.0;
        if (eta < -30.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-eta));
    }

    /**
     * @brief Evaluates the hurdle probability \f$ \pi = g^{-1}(\eta) \f$ with its link derivatives, kept strictly inside \f$ (0,1) \f$.
     * @details Delegates to `BetaMath::evaluate_link_hessian`, then clamps \f$ \pi \f$ to
     * \f$ [10^{-12}, 1 - 10^{-12}] \f$ so that \f$ \log\pi \f$ and \f$ \log(1-\pi) \f$ are finite. The derivatives
     * \f$ d\pi/d\eta \f$ and \f$ d^2\pi/d\eta^2 \f$ returned by the link helper are left untouched by this clamp.
     * @param eta_zero Linear predictor \f$ \eta^{(z)}_i \f$.
     * @param link Link type whose inverse is evaluated.
     * @param[out] pi Clamped probability \f$ \pi_i \f$.
     * @param[out] dpi Derivative \f$ d\pi/d\eta \f$.
     * @param[out] d2pi Second derivative \f$ d^2\pi/d\eta^2 \f$.
     */
    static inline void barrier(double eta_zero, LinkType link, double& pi, double& dpi, double& d2pi) {
        BetaMath::evaluate_link_hessian(eta_zero, link, pi, dpi, d2pi);
        pi = std::max(1e-12, std::min(1.0 - 1e-12, pi));
    }

    /**
     * @brief Evaluates the geometric mean \f$ \mu = g^{-1}(\eta) \f$ and its first two derivatives.
     * @details For the `Log` link, \f$ \eta \f$ is clamped to \f$ [-30, 30] \f$ and
     * \f$ \mu = \mu' = \mu'' = e^{\eta} \f$ (the derivatives are not set to zero outside the clamp). For every other
     * link `evaluate_link_d2` is used and \f$ \mu \f$ is floored at \f$ 10^{-8} \f$ (derivatives unchanged).
     * @param eta Linear predictor \f$ \eta^{(c)}_i \f$.
     * @param[out] mu Mean parameter \f$ \mu_i \f$.
     * @param[out] dmu Derivative \f$ d\mu/d\eta \f$.
     * @param[out] d2mu Second derivative \f$ d^2\mu/d\eta^2 \f$.
     */
    inline void count_mean(double eta, double& mu, double& dmu, double& d2mu) const {
        if (count_link == LinkType::Log) {
            const double e = std::max(-30.0, std::min(30.0, eta));
            mu = std::exp(e); dmu = mu; d2mu = mu;
            return;
        }
        evaluate_link_d2(eta, count_link, mu, dmu, d2mu);
        if (mu < 1e-8) mu = 1e-8;
    }

    /**
     * @brief Log-pmf of the zero-truncated geometric distribution.
     * @details The log-pmf is \f$ \log f(y;\mu) = (y-1)\log\mu - y\log(1+\mu) \f$ for \f$ y \ge 1 \f$, i.e. the geometric pmf
     * \f$ \mu^y/(1+\mu)^{y+1} \f$ renormalised by \f$ 1 - 1/(1+\mu) = \mu/(1+\mu) \f$.
     * @param y Positive count.
     * @param mu Mean parameter \f$ \mu > 0 \f$ of the untruncated geometric law.
     * @return The log-pmf value.
     */
    static inline double log_trunc_geom(double y, double mu) {
        return (y - 1.0) * std::log(mu) - y * std::log1p(mu);
    }

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ kc + kz \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors of the count and hurdle parts
        Eigen::VectorXd eta_geom = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        double total_neg_ll = 0.0;

        // accumulate per-observation log-likelihood terms
        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0, dpi_i = 0.0, d2pi_i = 0.0;
            barrier(eta_zero(i), zero_link, pi_i, dpi_i, d2pi_i);

            double log_prob_i = 0.0;

            if (y(i) < 0.5) {

                log_prob_i = std::log(1.0 - pi_i);
            } else {
                double log_pi = std::log(pi_i);

                double mu_i = 0.0, dmu_i = 0.0, d2mu_i = 0.0;
                count_mean(eta_geom(i), mu_i, dmu_i, d2mu_i);
                double log_geom = log_trunc_geom(y(i), mu_i);

                log_prob_i = log_pi + log_geom;
            }

            total_neg_ll -= log_prob_i;
        }

        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @return Vector of length \f$ kc + kz \f$: \f$ X^\top u \f$ followed by \f$ Z^\top v \f$ with
     *         \f$ u_i, v_i \f$ the per-observation derivatives with respect to the count and hurdle linear predictors
     *         (see the struct documentation). Non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors of the count and hurdle parts
        Eigen::VectorXd eta_geom = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        Eigen::VectorXd dNLL_deta_geom = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_zero = Eigen::VectorXd::Zero(n);

        // per-observation derivatives with respect to both linear predictors
        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0, dpi_i = 0.0, d2pi_i = 0.0;
            barrier(eta_zero(i), zero_link, pi_i, dpi_i, d2pi_i);

            if (y(i) < 0.5) {

                dNLL_deta_zero(i) = dpi_i / (1.0 - pi_i);
                dNLL_deta_geom(i) = 0.0;
            } else {

                dNLL_deta_zero(i) = -dpi_i / pi_i;

                if (count_link == LinkType::Log) {

                    double prob_success = sigmoid(std::max(-30.0, std::min(30.0, eta_geom(i))));
                    dNLL_deta_geom(i) = y(i) * prob_success - (y(i) - 1.0);
                } else {

                    double mu_i = 0.0, dmu_i = 0.0, d2mu_i = 0.0;
                    count_mean(eta_geom(i), mu_i, dmu_i, d2mu_i);
                    const double dL_dmu = -(y(i) - 1.0) / mu_i + y(i) / (1.0 + mu_i);
                    dNLL_deta_geom(i) = dL_dmu * dmu_i;
                }
            }
        }

        Eigen::VectorXd grad(kc + kz);
        grad.head(kc) = X.transpose() * dNLL_deta_geom;
        grad.tail(kz) = Z.transpose() * dNLL_deta_zero;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Exact observed Hessian of the negative log-likelihood (block diagonal).
     * @details For a zero observation, with \f$ \mathrm{NLL}_i = -\log(1-\pi_i) \f$, the weight of the hurdle block is
     * \f$ \pi'^2/(1-\pi)^2 + \pi''/(1-\pi) \f$; for a positive one, with \f$ -\log\pi_i \f$, it is
     * \f$ \pi'^2/\pi^2 - \pi''/\pi \f$. The count block receives (positive observations only)
     * \f$ \mathrm{NLL}_{\mu\mu}\mu'^2 + \mathrm{NLL}_\mu\mu'' \f$ with
     * \f$ \mathrm{NLL}_\mu = -(y-1)/\mu + y/(1+\mu) \f$ and
     * \f$ \mathrm{NLL}_{\mu\mu} = (y-1)/\mu^2 - y/(1+\mu)^2 \f$; for the `Log` link this is
     * \f$ y\,p(1-p) \f$ with \f$ p = \mathrm{sigmoid}(\eta^{(c)}) \f$.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @return Symmetric \f$ (kc+kz)\times(kc+kz) \f$ matrix with blocks \f$ X^\top\mathrm{diag}(a)X \f$ and
     *         \f$ Z^\top\mathrm{diag}(b)Z \f$ on the diagonal and zeros off the diagonal. No regularisation or
     *         finite-masking is applied.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.tail(kz);

        // linear predictors of the count and hurdle parts
        Eigen::VectorXd eta_geom = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        Eigen::VectorXd H_geom_geom = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd H_zero_zero = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0, dpi_i = 0.0, d2pi_i = 0.0;
            barrier(eta_zero(i), zero_link, pi_i, dpi_i, d2pi_i);

            if (y(i) < 0.5) {
                const double dL  = 1.0 / (1.0 - pi_i);
                const double d2L = dL * dL;
                H_zero_zero(i) = d2L * dpi_i * dpi_i + dL * d2pi_i;
            } else {
                const double dL  = -1.0 / pi_i;
                const double d2L = dL * dL;
                H_zero_zero(i) = d2L * dpi_i * dpi_i + dL * d2pi_i;

                if (count_link == LinkType::Log) {

                    double p_geom = sigmoid(std::max(-30.0, std::min(30.0, eta_geom(i))));
                    H_geom_geom(i) = y(i) * p_geom * (1.0 - p_geom);
                } else {

                    double mu_i = 0.0, dmu_i = 0.0, d2mu_i = 0.0;
                    count_mean(eta_geom(i), mu_i, dmu_i, d2mu_i);
                    const double dLg = -(y(i) - 1.0) / mu_i + y(i) / (1.0 + mu_i);
                    const double d2Lg = (y(i) - 1.0) / (mu_i * mu_i) - y(i) / ((1.0 + mu_i) * (1.0 + mu_i));
                    H_geom_geom(i) = d2Lg * dmu_i * dmu_i + dLg * d2mu_i;
                }
            }
        }

        // assemble the two diagonal blocks
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kc + kz, kc + kz);
        H.block(0, 0, kc, kc) = X.transpose() * H_geom_geom.asDiagonal() * X;
        H.block(kc, kc, kz, kz) = Z.transpose() * H_zero_zero.asDiagonal() * Z;

        return H;
    }
};

/**
 * @brief Negative log-likelihood of the zero-inflated negative binomial (ZINB) regression model.
 *
 * @details Models a non-negative integer count as a mixture of a point mass at zero (weight \f$ \pi_i \f$) and a
 * negative binomial (NB) variable with mean \f$ \lambda_i \f$ and dispersion \f$ \alpha \f$:
 * \f[ P(y_i = 0) = \pi_i + (1-\pi_i)\,p_0(\lambda_i,\alpha), \qquad
 *     P(y_i = k) = (1-\pi_i)\,\mathrm{NB}(k;\lambda_i,\alpha), \quad k \ge 1, \f]
 * where \f$ p_0 = \mathrm{NB}(0;\lambda_i,\alpha) \f$.
 * - Count part: \f$ \eta^{(c)}_i = \mathbf{x}_i^\top\boldsymbol\beta \f$ and \f$ \lambda_i = g^{-1}(\eta^{(c)}_i) \f$ via
 *   `evaluate_link` with `count_link`.
 * - Inflation part: \f$ \eta^{(z)}_i = \mathbf{z}_i^\top\boldsymbol\gamma \f$ and \f$ \pi_i \f$ (probability of a
 *   structural zero) \f$ = g_z^{-1}(\eta^{(z)}_i) \f$ via `evaluate_link` with `infl_link` (default `Logit`).
 * - Dispersion \f$ \alpha = e^{\log\alpha} > 0 \f$, a single scalar shared by all observations.
 *
 * The NB is written with `size` \f$ r \f$ and success probability \f$ p \f$
 * (\f$ \mathrm{NB}(k) = \frac{\Gamma(k+r)}{\Gamma(r)\,k!}\,p^{r}(1-p)^{k} \f$), selected by `var_form`:
 * | `VarianceForm` | \f$ r \f$ | \f$ p \f$ | Variance of the NB part |
 * |----------------|-----------|-----------|--------------------------|
 * | `Linear` (NB1) | \f$ \lambda/\alpha \f$ | \f$ 1/(1+\alpha) \f$ | \f$ \lambda(1+\alpha) \f$ |
 * | `Quadratic` (NB2) | \f$ 1/\alpha \f$ | \f$ 1/(1+\alpha\lambda) \f$ | \f$ \lambda + \alpha\lambda^2 \f$ |
 *
 * ### Parameter vector layout
 * | Index             | Meaning                                              | Scale                              |
 * |-------------------|------------------------------------------------------|------------------------------------|
 * | `0 .. kc-1`       | count coefficients \f$ \boldsymbol\beta \f$ (columns of `X`) | linear predictor of \f$ \lambda \f$ |
 * | `kc .. kc+kz-1`   | inflation coefficients \f$ \boldsymbol\gamma \f$ (columns of `Z`) | linear predictor of \f$ \pi \f$ |
 * | `kc+kz`           | \f$ \log\alpha \f$                                   | log; clamped to \f$ [-10, 10] \f$ before use |
 *
 * ### Log-likelihood
 * With \f$ L_0 = \max\{\pi_i + (1-\pi_i)p_0,\,10^{-15}\} \f$:
 * - \f$ y_i = 0 \f$ (exact comparison): \f$ \ell_i = \log L_0 \f$, with \f$ p_0 = p^{\,r} \f$ computed by `std::pow`;
 * - otherwise: \f$ \ell_i = \log(1-\pi_i) + \log\Gamma(y_i+r) - \log\Gamma(r) - \log\Gamma(y_i+1) + r\log p + y_i\log(1-p) \f$.
 *   For the `Logit` inflation link \f$ \log(1-\pi_i) = -\log(1+e^{\eta^{(z)}_i}) \f$ is evaluated with
 *   `log1p_exp_stable` from the unclamped \f$ \eta^{(z)} \f$; for other links \f$ \log(1-\pi_i) \f$ uses the clamped
 *   \f$ \pi_i \f$.
 *
 * The full pmf is kept (including \f$ -\log\Gamma(y_i+1) \f$). The functor returns \f$ -\sum_i \ell_i \f$.
 *
 * ### Derivatives
 * `gradient()` is analytical and exact for the (unclamped) model. Let \f$ \mathrm{NLL}_i \f$ be the per-observation
 * contribution. For \f$ y_i = 0 \f$,
 * \f$ \partial\mathrm{NLL}_i/\partial\pi = -(1-p_0)/L_0 \f$ and \f$ \partial\mathrm{NLL}_i/\partial p_0 = -(1-\pi)/L_0 \f$,
 * combined with \f$ \partial p_0/\partial\lambda \f$ and \f$ \partial p_0/\partial\alpha \f$ in closed form
 * (`Linear` and `Quadratic` cases coded separately). For \f$ y_i \ge 1 \f$ the inflation derivative is
 * \f$ \pi'/(1-\pi) \f$ and the count/dispersion derivatives are minus those of the NB log-pmf
 * (`digamma` based for `Linear`; for `Quadratic`,
 * \f$ \partial\mathrm{NLL}_i/\partial\lambda = -(y_i-\lambda)/(\lambda(1+\alpha\lambda)) \f$ written in an equivalent form).
 * Chain rule: \f$ \partial/\partial\eta^{(c)} = (\cdot)\,\lambda' \f$, \f$ \partial/\partial\eta^{(z)} = (\cdot)\,\pi' \f$ and
 * \f$ \partial/\partial\log\alpha = \alpha\,\partial/\partial\alpha \f$.
 *
 * `hessian()` is the analytical observed Hessian with the \f$ (kc+kz+1)\times(kc+kz+1) \f$ block structure
 * \f[ H = \begin{pmatrix} H_{\beta\beta} & H_{\beta\gamma} & h_{\beta,\log\alpha} \\
 *                        H_{\gamma\beta} & H_{\gamma\gamma} & h_{\gamma,\log\alpha} \\
 *                        h_{\beta,\log\alpha}^\top & h_{\gamma,\log\alpha}^\top & h_{\log\alpha,\log\alpha} \end{pmatrix}, \f]
 * with \f$ H_{\beta\beta} = X^\top\mathrm{diag}(W_{nb,nb})X \f$, \f$ H_{\gamma\gamma} = Z^\top\mathrm{diag}(W_{lg,lg})Z \f$,
 * \f$ H_{\beta\gamma} = X^\top\mathrm{diag}(W_{nb,lg})Z \f$, and the \f$ \log\alpha \f$ row/column built from
 * \f$ X^\top W_{nb,\alpha} \f$, \f$ Z^\top W_{lg,\alpha} \f$ and a scalar. The cross terms between the count and inflation
 * parts (\f$ W_{nb,lg} \f$, \f$ W_{lg,\alpha} \f$) are non-zero only for \f$ y_i = 0 \f$. The matrix is symmetric by
 * construction (blocks are mirrored explicitly); it is not regularised and not finite-masked.
 *
 * ### Numerical safeguards
 * - \f$ \log\alpha \f$ is clamped to \f$ [-10, 10] \f$ in all three methods; derivatives with respect to
 *   \f$ \log\alpha \f$ are **not** zeroed when the clamp is active.
 * - \f$ \lambda_i \f$ is clamped to \f$ [10^{-10}, 10^{6}] \f$; when it is clamped, \f$ \lambda' \f$ (and
 *   \f$ \lambda'' \f$) are set to 0 in `gradient()` and `hessian()`.
 * - \f$ \pi_i \f$ is clamped to \f$ [10^{-15}, 1 - 10^{-15}] \f$ (its derivatives are not adjusted); the zero-class
 *   probability \f$ L_0 \f$ is floored at \f$ 10^{-15} \f$.
 * - Unlike most other losses, `operator()` has **no** `1e10` fallback and `gradient()`/`hessian()` do not mask
 *   non-finite values.
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y`; they must outlive the functor. No
 *       validation is done; \f$ y_i \ge 0 \f$ is assumed (any value different from exactly 0 is treated as a positive
 *       count, non-integers included).
 * @note In `hessian()` the link curvatures \f$ \lambda'' \f$ (count link) and \f$ \pi'' \f$ (inflation link) are obtained
 *       for any link with `evaluate_d2_inverse_link_scalar`; \f$ \lambda'' \f$ is set to 0 only where the \f$ \lambda \f$ clamp is active.
 * @note The gradient uses `digamma` / `trigamma` from `mle_utils.hpp` (recurrence plus asymptotic series, absolute error
 *       below \f$ 10^{-9} \f$), while the value uses `std::lgamma`.
 * @note The private helper `dg_nb2()` is not used.
 * @see NegativeBinomialInflatedLoss, HurdleNBLoss, ZeroInflatedPoissonLoss
 *
 * @ingroup glm_ll_zero_inflated
 */
struct ZeroInflatedNBLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the NB count model (\f$ n \times kc \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the zero-inflation model (\f$ n \times kz \f$).
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$); exact zeros form the inflated class.
    LinkType count_link;  ///< Inverse link mapping \f$ \eta^{(c)} \f$ to the NB mean \f$ \lambda \f$.
    VarianceForm var_form;  ///< NB variance form: `Linear` (NB1) or `Quadratic` (NB2).
    LinkType infl_link;  ///< Inverse link mapping \f$ \eta^{(z)} \f$ to the structural-zero probability \f$ \pi \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog_count Count design matrix \f$ X \f$ (\f$ n \times kc \f$).
     * @param exog_infl Inflation design matrix \f$ Z \f$ (\f$ n \times kz \f$).
     * @param endog Non-negative count response \f$ y \f$ (length \f$ n \f$).
     * @param link Count link whose inverse maps \f$ \eta^{(c)} \f$ to the NB mean.
     * @param form NB variance form (`Linear` = NB1, `Quadratic` = NB2).
     * @param infl_l Inflation link whose inverse maps \f$ \eta^{(z)} \f$ to \f$ \pi \f$ (default `Logit`).
     * @pre `exog_count`, `exog_infl` and `endog` have the same number of rows and outlive the functor.
     */
    ZeroInflatedNBLoss(const Eigen::MatrixXd& exog_count, const Eigen::MatrixXd& exog_infl,
                       const Eigen::VectorXd& endog, LinkType link, VarianceForm form,
                       LinkType infl_l = LinkType::Logit)
        : X(exog_count), Z(exog_infl), y(endog), count_link(link), var_form(form), infl_link(infl_l) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$ of length \f$ kc+kz+1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$ (no finite check, no fallback value).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_alpha = std::max(-10.0, std::min(10.0, params(kc + kz)));
        double alpha = std::exp(log_alpha);

        // linear predictors of the count and inflation parts
        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_logit = Z * gamma;

        double total_neg_ll = 0.0;
        // accumulate per-observation terms (zero class vs positive counts)
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_deta_i);
            lambda_i = std::max(1e-10, std::min(lambda_i, 1e6));

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_logit(i), infl_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            double log_prob;

            double size_param, prob_param;
            if (var_form == VarianceForm::Linear) {
                size_param = lambda_i / alpha;
                prob_param = 1.0 / (1.0 + alpha);
            } else {
                size_param = 1.0 / alpha;
                prob_param = 1.0 / (1.0 + alpha * lambda_i);
            }

            if (y(i) == 0.0) {
                double nb_zero_prob = std::pow(prob_param, size_param);
                double prob_zero = std::max(pi_i + (1.0 - pi_i) * nb_zero_prob, 1e-15);
                log_prob = std::log(prob_zero);
            } else {

                double log_not_pi = (infl_link == LinkType::Logit) ? -log1p_exp_stable(eta_logit(i))
                                                                   : std::log(1.0 - pi_i);
                double log_pmf_nb = std::lgamma(y(i) + size_param)
                                  - std::lgamma(size_param)
                                  - std::lgamma(y(i) + 1.0)
                                  + size_param * std::log(prob_param)
                                  + y(i) * std::log(1.0 - prob_param);
                log_prob = log_not_pi + log_pmf_nb;
            }
            total_neg_ll -= log_prob;
        }
        return total_neg_ll;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$.
     * @return Vector of length \f$ kc+kz+1 \f$: \f$ X^\top(\partial\mathrm{NLL}/\partial\eta^{(c)}) \f$,
     *         \f$ Z^\top(\partial\mathrm{NLL}/\partial\eta^{(z)}) \f$, and
     *         \f$ \sum_i \alpha\,\partial\mathrm{NLL}_i/\partial\alpha \f$ (derivative with respect to \f$ \log\alpha \f$).
     *         Non-finite components are not replaced.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_alpha = std::max(-10.0, std::min(10.0, params(kc + kz)));
        double alpha = std::exp(log_alpha);
        double inv_alpha = 1.0 / alpha;

        // linear predictors of the count and inflation parts
        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_logit = Z * gamma;

        Eigen::VectorXd dNLL_deta_nb = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_logit = Eigen::VectorXd::Zero(n);
        double dNLL_dlog_alpha = 0.0;

        // per-observation derivatives (chain rule through the links)
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_deta_i);

            bool clipped = (lambda_i > 1e6 || lambda_i < 1e-10);
            lambda_i = std::max(1e-10, std::min(lambda_i, 1e6));
            double dlam_i = clipped ? 0.0 : dlam_deta_i;

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_logit(i), infl_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            double size_param, prob_param;
            if (var_form == VarianceForm::Linear) {
                size_param = lambda_i / alpha;
                prob_param = 1.0 / (1.0 + alpha);
            } else {
                size_param = 1.0 / alpha;
                prob_param = 1.0 / (1.0 + alpha * lambda_i);
            }

            if (y(i) == 0.0) {
                double nb_zero_prob = std::pow(prob_param, size_param);
                double prob_zero = std::max(pi_i + (1.0 - pi_i) * nb_zero_prob, 1e-15);

                double dL_dpi = -(1.0 - nb_zero_prob) / prob_zero;
                dNLL_deta_logit(i) = dL_dpi * dpi_deta_i;

                double dL_dmu = 0.0; double dL_dalpha = 0.0;
                if (var_form == VarianceForm::Linear) {
                    dL_dmu = -((1.0 - pi_i) * nb_zero_prob * (inv_alpha * std::log(prob_param))) / prob_zero;
                    double dnb_dalpha = nb_zero_prob * (-(lambda_i * inv_alpha / alpha) * std::log(prob_param) - (size_param / (1.0 + alpha)));
                    dL_dalpha = -((1.0 - pi_i) * dnb_dalpha) / prob_zero;
                } else {
                    dL_dmu = ((1.0 - pi_i) * nb_zero_prob / (1.0 + alpha * lambda_i)) / prob_zero;
                    double dnb_dalpha = nb_zero_prob * ((inv_alpha / alpha) * std::log1p(alpha * lambda_i) - (lambda_i * inv_alpha) / (1.0 + alpha * lambda_i));
                    dL_dalpha = -((1.0 - pi_i) * dnb_dalpha) / prob_zero;
                }
                dNLL_deta_nb(i) = dL_dmu * dlam_i;
                dNLL_dlog_alpha += dL_dalpha * alpha;
            } else {

                dNLL_deta_logit(i) = dpi_deta_i / (1.0 - pi_i);

                double dL_dmu = 0.0; double dpmf_dalpha = 0.0;
                if (var_form == VarianceForm::Linear) {
                    dL_dmu = -((inv_alpha * (digamma(y(i) + size_param) - digamma(size_param) + std::log(prob_param))));
                    dpmf_dalpha = -(lambda_i * inv_alpha / alpha) * (digamma(y(i) + size_param) - digamma(size_param) + std::log(prob_param))
                                  - (size_param / (1.0 + alpha)) + (y(i) * inv_alpha) - (y(i) / (1.0 + alpha));
                } else {
                    dL_dmu = -((y(i) / lambda_i) - (y(i) + inv_alpha) * (alpha / (1.0 + alpha * lambda_i)));
                    dpmf_dalpha = -inv_alpha * inv_alpha * digamma(y(i) + inv_alpha)
                                   + inv_alpha * inv_alpha * digamma(inv_alpha)
                                   + inv_alpha * inv_alpha * std::log1p(alpha * lambda_i)
                                   - (y(i) + inv_alpha) * (lambda_i / (1.0 + alpha * lambda_i))
                                   + y(i) * inv_alpha;
                }
                dNLL_deta_nb(i) = dL_dmu * dlam_i;
                dNLL_dlog_alpha += -(dpmf_dalpha * alpha);
            }
        }

        Eigen::VectorXd grad(kc + kz + 1);
        grad.head(kc) = X.transpose() * dNLL_deta_nb;
        grad.segment(kc, kz) = Z.transpose() * dNLL_deta_logit;
        grad(kc + kz) = dNLL_dlog_alpha;
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$.
     * @details Per observation the second derivatives with respect to \f$ (\pi, \lambda, \alpha) \f$ are formed in closed
     * form (for \f$ y_i = 0 \f$ through the zero-class probability \f$ p_0 \f$ and its first/second derivatives; for
     * \f$ y_i \ge 1 \f$ through `digamma` and `trigamma`), then mapped to the linear predictors with the link
     * derivatives. The \f$ \log\alpha \f$ diagonal entry accumulates \f$ \alpha^2\,\partial^2_\alpha\mathrm{NLL}_i + \alpha\,\partial_\alpha\mathrm{NLL}_i \f$.
     * See the struct documentation for the block layout and for the limitation concerning the count-link curvature.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$.
     * @return Symmetric \f$ (kc+kz+1)\times(kc+kz+1) \f$ matrix (no regularisation, no finite masking).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();
        const long total_params = kc + kz + 1;

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_alpha = std::max(-10.0, std::min(10.0, params(kc + kz)));
        double alpha = std::exp(log_alpha);
        double inv_alpha = 1.0 / alpha;

        // linear predictors of the count and inflation parts
        Eigen::VectorXd eta_nb = X * beta;
        Eigen::VectorXd eta_logit = Z * gamma;

        Eigen::VectorXd W_nb_nb = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_lg_lg = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_nb_lg = Eigen::VectorXd::Zero(n);

        Eigen::VectorXd W_nb_alpha = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_lg_alpha = Eigen::VectorXd::Zero(n);
        double d2NLL_dlogalpha2 = 0.0;

        // per-observation weights W_* and log-alpha accumulators
        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0; double dlam_deta_i = 0.0;
            evaluate_link(eta_nb(i), count_link, lambda_i, dlam_deta_i);

            bool clipped = (lambda_i > 1e6 || lambda_i < 1e-10);
            lambda_i = std::max(1e-10, std::min(lambda_i, 1e6));
            double dlam_i = clipped ? 0.0 : dlam_deta_i;
            double d2lam_i = clipped ? 0.0 : evaluate_d2_inverse_link_scalar(eta_nb(i), count_link);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            evaluate_link(eta_logit(i), infl_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));
            double d2pi_i = evaluate_d2_inverse_link_scalar(eta_logit(i), infl_link);

            double size_param, prob_param;
            if (var_form == VarianceForm::Linear) {
                size_param = lambda_i / alpha;
                prob_param = 1.0 / (1.0 + alpha);
            } else {
                size_param = 1.0 / alpha;
                prob_param = 1.0 / (1.0 + alpha * lambda_i);
            }

            if (y(i) == 0.0) {
                double nb_zero_prob = std::pow(prob_param, size_param);
                double prob_zero = std::max(pi_i + (1.0 - pi_i) * nb_zero_prob, 1e-15);

                double dL_dpi = -(1.0 - nb_zero_prob) / prob_zero;
                double dL_dnbzero = -(1.0 - pi_i) / prob_zero;

                double dnbzero_dmu = 0.0;
                double dnbzero_dalpha = 0.0;
                double d2nbzero_dmu2 = 0.0;
                double d2nbzero_dalpha2 = 0.0;
                double d2nbzero_dmudalpha = 0.0;

                if (var_form == VarianceForm::Linear) {
                    double log_p = std::log(prob_param);
                    dnbzero_dmu = nb_zero_prob * (inv_alpha * log_p);
                    dnbzero_dalpha = nb_zero_prob * (-(lambda_i * inv_alpha / alpha) * log_p - (size_param / (1.0 + alpha)));

                    d2nbzero_dmu2 = dnbzero_dmu * dnbzero_dmu / nb_zero_prob;
                    d2nbzero_dalpha2 = dnbzero_dalpha * dnbzero_dalpha / nb_zero_prob
                        + nb_zero_prob * ((2.0 * lambda_i / (alpha * alpha * alpha)) * log_p + (2.0 * lambda_i * inv_alpha / (alpha * (1.0 + alpha))) + size_param / ((1.0 + alpha) * (1.0 + alpha)));
                    d2nbzero_dmudalpha = dnbzero_dmu * dnbzero_dalpha / nb_zero_prob
                        + nb_zero_prob * (-(inv_alpha / alpha) * log_p - inv_alpha / (1.0 + alpha));
                } else {
                    double den = 1.0 + alpha * lambda_i;
                    dnbzero_dmu = -nb_zero_prob * (1.0 / den);
                    dnbzero_dalpha = nb_zero_prob * ((inv_alpha / alpha) * std::log1p(alpha * lambda_i) - (lambda_i * inv_alpha) / den);

                    d2nbzero_dmu2 = dnbzero_dmu * dnbzero_dmu / nb_zero_prob + nb_zero_prob * (alpha / (den * den));
                    d2nbzero_dalpha2 = dnbzero_dalpha * dnbzero_dalpha / nb_zero_prob
                        + nb_zero_prob * (-(2.0 / (alpha * alpha * alpha)) * std::log1p(alpha * lambda_i) + (2.0 * lambda_i / (alpha * alpha * den)) + (lambda_i * lambda_i * inv_alpha) / (den * den));
                    d2nbzero_dmudalpha = dnbzero_dmu * dnbzero_dalpha / nb_zero_prob + nb_zero_prob * (lambda_i / (den * den));
                }

                double dL_dmu = dL_dnbzero * dnbzero_dmu;
                double dL_dalpha = dL_dnbzero * dnbzero_dalpha;

                double d2L_dpi2 = (1.0 - nb_zero_prob) * (1.0 - nb_zero_prob) / (prob_zero * prob_zero);
                double d2L_dpidnbzero = 1.0 / (prob_zero * prob_zero);
                double d2L_dnbzero2 = (1.0 - pi_i) * (1.0 - pi_i) / (prob_zero * prob_zero);

                double d2L_dmu2 = d2L_dnbzero2 * dnbzero_dmu * dnbzero_dmu + dL_dnbzero * d2nbzero_dmu2;
                double d2L_dalpha2 = d2L_dnbzero2 * dnbzero_dalpha * dnbzero_dalpha + dL_dnbzero * d2nbzero_dalpha2;
                double d2L_dmudalpha = d2L_dnbzero2 * dnbzero_dmu * dnbzero_dalpha + dL_dnbzero * d2nbzero_dmudalpha;

                W_lg_lg(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_i;
                W_nb_nb(i) = d2L_dmu2 * dlam_i * dlam_i + dL_dmu * d2lam_i;
                W_nb_lg(i) = d2L_dpidnbzero * dnbzero_dmu * dpi_deta_i * dlam_i;

                W_nb_alpha(i) = d2L_dmudalpha * dlam_i * alpha;
                W_lg_alpha(i) = d2L_dpidnbzero * dnbzero_dalpha * dpi_deta_i * alpha;

                d2NLL_dlogalpha2 += d2L_dalpha2 * alpha * alpha + dL_dalpha * alpha;

            } else {

                W_lg_lg(i) = d2pi_i / (1.0 - pi_i) + (dpi_deta_i * dpi_deta_i) / ((1.0 - pi_i) * (1.0 - pi_i));
                W_nb_lg(i) = 0.0;
                W_lg_alpha(i) = 0.0;

                double dL_dmu = 0.0;
                double d2L_dmu2 = 0.0;
                double dL_dalpha = 0.0;
                double d2L_dalpha2 = 0.0;
                double d2L_dmudalpha = 0.0;

                double yi = y(i);

                if (var_form == VarianceForm::Linear) {
                    double dg = digamma(yi + size_param) - digamma(size_param) + std::log(prob_param);
                    double tg = trigamma(yi + size_param) - trigamma(size_param);

                    dL_dmu = -(inv_alpha * dg);
                    d2L_dmu2 = -(inv_alpha * inv_alpha * tg);
                    d2L_dmudalpha = (inv_alpha / alpha) * dg + (lambda_i * inv_alpha * inv_alpha / alpha) * tg + inv_alpha / (1.0 + alpha);

                    dL_dalpha = (lambda_i * inv_alpha / alpha) * dg + (lambda_i - yi) / (alpha * (1.0 + alpha));
                    d2L_dalpha2 = -(2.0 * lambda_i * inv_alpha / (alpha * alpha)) * dg
                                  - (lambda_i * lambda_i * inv_alpha * inv_alpha / (alpha * alpha)) * tg
                                  + (yi - 2.0 * lambda_i) / (alpha * alpha * (1.0 + alpha))
                                  - (lambda_i - yi) / (alpha * (1.0 + alpha) * (1.0 + alpha));
                } else {
                    double den = 1.0 + alpha * lambda_i;
                    double tg = trigamma(yi + inv_alpha) - trigamma(inv_alpha);
                    double dg = digamma(yi + inv_alpha) - digamma(inv_alpha) - std::log(den);

                    dL_dmu = (lambda_i - yi) / (lambda_i * den);
                    d2L_dmu2 = (yi / (lambda_i * lambda_i)) - alpha * (1.0 + alpha * yi) / (den * den);
                    d2L_dmudalpha = (yi - lambda_i) / (den * den);

                    dL_dalpha = inv_alpha * inv_alpha * dg + (lambda_i - yi) / (alpha * den);
                    d2L_dalpha2 = -(2.0 * inv_alpha * inv_alpha * inv_alpha) * dg
                                  - inv_alpha * inv_alpha * inv_alpha * inv_alpha * tg
                                  + (yi - 2.0 * lambda_i) / (alpha * alpha * den)
                                  - lambda_i * (lambda_i - yi) / (alpha * den * den);
                }

                W_nb_nb(i) = d2L_dmu2 * dlam_i * dlam_i + dL_dmu * d2lam_i;
                W_nb_alpha(i) = d2L_dmudalpha * dlam_i * alpha;
                d2NLL_dlogalpha2 += d2L_dalpha2 * alpha * alpha + dL_dalpha * alpha;
            }
        }

        // assemble the Hessian blocks (mirrored to keep symmetry)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(total_params, total_params);

        H.block(0, 0, kc, kc) = X.transpose() * W_nb_nb.asDiagonal() * X;
        H.block(kc, kc, kz, kz) = Z.transpose() * W_lg_lg.asDiagonal() * Z;

        Eigen::MatrixXd H_cruzado = X.transpose() * W_nb_lg.asDiagonal() * Z;
        H.block(0, kc, kc, kz) = H_cruzado;
        H.block(kc, 0, kz, kc) = H_cruzado.transpose();

        H.block(0, kc + kz, kc, 1) = X.transpose() * W_nb_alpha;
        H.block(kc + kz, 0, 1, kc) = (X.transpose() * W_nb_alpha).transpose();

        H.block(kc, kc + kz, kz, 1) = Z.transpose() * W_lg_alpha;
        H.block(kc + kz, kc, 1, kz) = (Z.transpose() * W_lg_alpha).transpose();

        H(kc + kz, kc + kz) = d2NLL_dlogalpha2;

        return H;
    }

private:

    /**
     * @brief Unused helper evaluating \f$ \psi(y+1/a) - \psi(1/a) - \log(\mathrm{den}) + (y+1/a)\,\mu/\mathrm{den} - y \f$.
     * @details Intended for the NB2 dispersion derivative; it is not referenced by any method of this struct.
     * @param yi Count \f$ y \f$.
     * @param inv_a Inverse dispersion \f$ 1/\alpha \f$.
     * @param den Denominator \f$ 1 + \alpha\mu \f$.
     * @param mu Mean \f$ \mu \f$.
     * @return The value of the expression above, with \f$ \psi \f$ the `digamma` function.
     */
    static inline double dg_nb2(double yi, double inv_a, double den, double mu) {
        return digamma(yi + inv_a) - digamma(inv_a) - std::log(den) + (yi + inv_a) * (mu / den) - yi;
    }
};

/**
 * @brief Negative binomial pmf policy (log-pmf and its mean derivatives) used as `BaseDistribution` of NegativeBinomialInflatedLoss.
 *
 * @details Provides the NB log-pmf with mean \f$ \mu \f$ and dispersion \f$ \alpha \f$, its first and second
 * derivatives with respect to \f$ \mu \f$, and the parameter constraints used by the inflated loss. With `size` \f$ r \f$
 * and success probability \f$ p \f$,
 * \f[ \log\mathrm{NB}(y) = \log\Gamma(y+r) - \log\Gamma(r) - \log\Gamma(y+1) + r\log p + y\log(1-p), \f]
 * | `var_form`          | \f$ r \f$ | \f$ p \f$ | Variance |
 * |---------------------|-----------|-----------|----------|
 * | `Linear` (NB1)      | \f$ \mu/\alpha \f$ | \f$ 1/(1+\alpha) \f$ | \f$ \mu(1+\alpha) \f$ |
 * | `Quadratic` (NB2)   | \f$ 1/\alpha \f$ | \f$ 1/(1+\alpha\mu) \f$ | \f$ \mu+\alpha\mu^2 \f$ |
 *
 * The full pmf (all constants) is used.
 *
 * @note The methods do not apply the constraints themselves; the caller must call `constrain_parameter()` and
 *       `constrain_dispersion()` first. Derivatives use `digamma`/`trigamma` of `mle_utils.hpp` (recurrence plus
 *       asymptotic series, absolute error below \f$ 10^{-9} \f$).
 * @see NegativeBinomialInflatedLoss
 *
 * @ingroup glm_ll_zero_inflated
 */
struct NegativeBinomialComponent {
    VarianceForm var_form;  ///< NB variance form: `Linear` (NB1) or `Quadratic` (NB2).

    /**
     * @brief Constructs the component.
     * @param form Variance form (`Linear` = NB1, `Quadratic` = NB2); default `Quadratic`.
     */
    NegativeBinomialComponent(VarianceForm form = VarianceForm::Quadratic) : var_form(form) {}

    /**
     * @brief Floors the mean to keep the pmf finite.
     * @param[in,out] mu Mean \f$ \mu \f$; replaced by \f$ 10^{-6} \f$ if smaller.
     */
    inline void constrain_parameter(double& mu) const {
        if (mu < 1e-6) mu = 1e-6;
    }

    /**
     * @brief Clamps the dispersion to a safe range.
     * @param[in,out] alpha Dispersion \f$ \alpha \f$; clamped to \f$ [10^{-6}, 10^{3}] \f$.
     */
    inline void constrain_dispersion(double& alpha) const {
        if (alpha < 1e-6) alpha = 1e-6;
        if (alpha > 1e3)  alpha = 1e3;
    }

    /**
     * @brief Log-pmf of the NB distribution parametrised by its mean.
     * @details Computes \f$ \log\Gamma(y+r) - \log\Gamma(r) - \log\Gamma(y+1) + r\log p + y\log(1-p) \f$ with \f$ (r,p) \f$ given by
     * `var_form` (see the struct documentation); `std::lgamma` is used.
     * @param yi Observed count \f$ y \f$.
     * @param mu Mean \f$ \mu > 0 \f$.
     * @param alpha Dispersion \f$ \alpha > 0 \f$.
     * @return The log-pmf value.
     */
    double log_pmf_mu(double yi, double mu, double alpha) const {
        double size_param = (var_form == VarianceForm::Linear) ? (mu / alpha) : (1.0 / alpha);
        double prob_param = (var_form == VarianceForm::Linear) ? (1.0 / (1.0 + alpha)) : (1.0 / (1.0 + alpha * mu));

        return std::lgamma(yi + size_param) - std::lgamma(size_param) - std::lgamma(yi + 1.0)
               + size_param * std::log(prob_param) + yi * std::log(1.0 - prob_param);
    }

    /**
     * @brief First derivative of the log-pmf with respect to the mean \f$ \mu \f$ (fixed \f$ \alpha \f$).
     * @details `Linear`: \f$ \frac{1}{\alpha}\big[\psi(y+\mu/\alpha) - \psi(\mu/\alpha) + \log\frac{1}{1+\alpha}\big] \f$.
     * `Quadratic`: \f$ \frac{y-\mu}{\mu(1+\alpha\mu)} \f$.
     * @param yi Observed count \f$ y \f$.
     * @param mu Mean \f$ \mu > 0 \f$.
     * @param alpha Dispersion \f$ \alpha > 0 \f$.
     * @return The derivative \f$ \partial\log\mathrm{NB}(y)/\partial\mu \f$.
     */
    double d_log_pmf_dmu(double yi, double mu, double alpha) const {
        if (var_form == VarianceForm::Linear) {
            return (1.0 / alpha) * (digamma(yi + mu / alpha) - digamma(mu / alpha) + std::log(1.0 / (1.0 + alpha)));
        } else {
            return (yi - mu) / (mu * (1.0 + alpha * mu));
        }
    }

    /**
     * @brief Second derivative of the log-pmf with respect to the mean \f$ \mu \f$ (fixed \f$ \alpha \f$).
     * @details `Linear`: \f$ \frac{1}{\alpha^2}\big[\psi_1(y+\mu/\alpha) - \psi_1(\mu/\alpha)\big] \f$ with
     * \f$ \psi_1 \f$ the trigamma function. `Quadratic`:
     * \f$ \frac{\alpha\mu^2 - y(1+2\alpha\mu)}{\mu^2(1+\alpha\mu)^2} \f$.
     * @param yi Observed count \f$ y \f$.
     * @param mu Mean \f$ \mu > 0 \f$.
     * @param alpha Dispersion \f$ \alpha > 0 \f$.
     * @return The second derivative \f$ \partial^2\log\mathrm{NB}(y)/\partial\mu^2 \f$.
     */
    double d2_log_pmf_dmu2(double yi, double mu, double alpha) const {
        if (var_form == VarianceForm::Linear) {
            double inv_a = 1.0 / alpha;
            return (inv_a * inv_a) * (trigamma(yi + mu / alpha) - trigamma(mu / alpha));
        } else {
            double den = 1.0 + alpha * mu;
            return (alpha * mu * mu - yi * (1.0 + 2.0 * alpha * mu)) / (mu * mu * den * den);
        }
    }
};

/**
 * @brief Negative log-likelihood of a negative binomial model inflated at an arbitrary count value \f$ k \f$.
 *
 * @details Mixture of a point mass at the target count \f$ k \f$ (weight \f$ \pi_i \f$) and an untruncated count
 * distribution with mean \f$ \mu_i \f$ and dispersion \f$ \alpha \f$ supplied by `BaseDistribution`
 * (typically NegativeBinomialComponent):
 * \f[ P(y_i = k) = \pi_i + (1-\pi_i)\,f(k;\mu_i,\alpha), \qquad
 *     P(y_i = y) = (1-\pi_i)\,f(y;\mu_i,\alpha), \quad y \ne k. \f]
 * \f$ k = \f$ `k_target` can be 0 (zero inflation) or any other value (e.g. one-inflation); the base pmf is **not**
 * truncated (this is inflation, not a hurdle). Observations are classified as \f$ y_i = k \f$ when
 * \f$ |y_i - k| < 10^{-9} \f$.
 * - Base part: \f$ \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$ with `evaluate_link` and `base_link`, then
 *   `base_dist.constrain_parameter()` (for NegativeBinomialComponent: floor \f$ 10^{-6} \f$).
 * - Inflation part: \f$ \pi_i = g_z^{-1}(\mathbf{z}_i^\top\boldsymbol\gamma) \f$ with `infl_link`, clamped to
 *   \f$ [10^{-15}, 1-10^{-15}] \f$.
 * - Dispersion: \f$ \alpha = \exp(\texttt{params}[kc+kz]) \f$ followed by `base_dist.constrain_dispersion()` (for
 *   NegativeBinomialComponent: clamp to \f$ [10^{-6}, 10^{3}] \f$).
 *
 * ### Parameter vector layout
 * | Index             | Meaning                                         | Scale                                    |
 * |-------------------|-------------------------------------------------|------------------------------------------|
 * | `0 .. kc-1`       | base coefficients \f$ \boldsymbol\beta \f$ (columns of `X`) | linear predictor of \f$ \mu \f$ |
 * | `kc .. kc+kz-1`   | inflation coefficients \f$ \boldsymbol\gamma \f$ (columns of `Z`) | linear predictor of \f$ \pi \f$ |
 * | `kc+kz`           | \f$ \log\alpha \f$ (raw, not clamped)           | log; \f$ \alpha \f$ is clamped after `exp` by the base distribution |
 *
 * ### Log-likelihood
 * - \f$ y_i = k \f$: \f$ \ell_i = \log\max\{\pi_i + (1-\pi_i)f(k;\mu_i,\alpha),\,10^{-15}\} \f$, with
 *   \f$ f = \exp(\texttt{log\_pmf\_mu}) \f$;
 * - otherwise: \f$ \ell_i = \log(1-\pi_i) + \log f(y_i;\mu_i,\alpha) \f$, where \f$ \log(1-\pi_i) = -\log(1+e^{\eta^{(z)}_i}) \f$
 *   (`log1p_exp_stable`, unclamped \f$ \eta^{(z)} \f$) for the `Logit` inflation link and \f$ \log(1-\pi_i) \f$ of the
 *   clamped \f$ \pi_i \f$ otherwise.
 *
 * The functor returns \f$ -\sum_i \ell_i \f$ (no finite check, no `1e10` fallback).
 *
 * ### Derivatives
 * - `gradient()`: the derivatives with respect to \f$ \eta^{(c)} \f$ and \f$ \eta^{(z)} \f$ are analytical. With
 *   \f$ L = \max\{\pi + (1-\pi)P,\,10^{-15}\} \f$, \f$ P = f(k) \f$ and \f$ s = \partial\log f/\partial\mu \f$: for
 *   \f$ y = k \f$, \f$ \partial_{\eta^{(c)}}\mathrm{NLL} = -(1-\pi)Ps\,\mu'/L \f$ and
 *   \f$ \partial_{\eta^{(z)}}\mathrm{NLL} = -(1-P)\,\pi'/L \f$; for \f$ y \ne k \f$, \f$ -s\,\mu' \f$ and
 *   (`Logit`) \f$ \pi \f$ or (other links) \f$ \pi'/(1-\pi) \f$.
 *   The derivative with respect to \f$ \log\alpha \f$ is obtained by a **central finite difference in \f$ \alpha \f$**
 *   with step \f$ \varepsilon = 10^{-5} \f$ and multiplied by \f$ \alpha \f$.
 * - `hessian()`: the \f$ \beta\beta \f$, \f$ \gamma\gamma \f$ and \f$ \beta\gamma \f$ blocks are analytical
 *   (including link curvature, via `evaluate_link_d2`); the \f$ \beta\beta \f$ and \f$ \gamma\gamma \f$ weights use
 *   `base_dist.d2_log_pmf_dmu2`. The \f$ \log\alpha \f$ row/column are built with finite differences with step
 *   \f$ \varepsilon = 10^{-5} \f$ in \f$ \alpha \f$: a three-point stencil for the second derivative and central
 *   differences of the \f$ \eta \f$-gradients for the cross terms, all multiplied by \f$ \alpha \f$ (and
 *   \f$ \alpha^2 \f$, plus the \f$ \alpha\,\partial_\alpha \f$ term, for the diagonal entry). The \f$ \beta\gamma \f$
 *   cross term is non-zero only for \f$ y_i = k \f$. The matrix is symmetric by construction, not regularised, not
 *   finite-masked.
 *
 * ### Numerical safeguards
 * - Constraints on \f$ \mu \f$ and \f$ \alpha \f$ are delegated to `BaseDistribution`; the derivatives
 *   \f$ \mu' \f$, \f$ \mu'' \f$ are not adjusted when \f$ \mu \f$ is floored, and the \f$ \log\alpha \f$ derivatives are not zeroed
 *   when \f$ \alpha \f$ is clamped.
 * - \f$ \pi_i \f$ is clamped to \f$ [10^{-15}, 1-10^{-15}] \f$ (link derivatives not adjusted); \f$ L \f$ is floored at
 *   \f$ 10^{-15} \f$.
 * - The finite-difference stencil evaluates the pmf at \f$ \alpha \pm 10^{-5} \f$ without re-applying
 *   `constrain_dispersion()`; for \f$ \alpha \le 10^{-5} \f$ the lower point is non-positive and the result is not
 *   meaningful. The second-difference quotient divides by \f$ \varepsilon^2 = 10^{-10} \f$ and is therefore subject to
 *   round-off noise.
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y` (they must outlive the functor), and keeps a
 *       **copy** of `base_dist`. No validation is done.
 * @note Template requirements: `BaseDistribution` must provide `const`-callable `constrain_parameter(double&)`,
 *       `constrain_dispersion(double&)`, `log_pmf_mu(y, mu, alpha)`, `d_log_pmf_dmu(y, mu, alpha)` and
 *       `d2_log_pmf_dmu2(y, mu, alpha)` (as NegativeBinomialComponent does), because the call operators are `const`.
 * @note Compared with ZeroInflatedNBLoss, the inflated value is configurable, \f$ \lambda \f$ is not capped above, and
 *       the dispersion derivatives are numerical.
 * @tparam BaseDistribution Policy type describing the base count pmf (e.g. NegativeBinomialComponent).
 * @see NegativeBinomialComponent, ZeroInflatedNBLoss
 *
 * @ingroup glm_ll_zero_inflated
 */
template <typename BaseDistribution>

struct NegativeBinomialInflatedLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the base (NB mean) model (\f$ n \times kc \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the inflation model (\f$ n \times kz \f$).
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$).
    const double k_target;  ///< Inflated count value \f$ k \f$; an observation is in the inflated class if \f$ |y_i - k| < 10^{-9} \f$.
    LinkType base_link;  ///< Inverse link mapping the base linear predictor to the mean \f$ \mu \f$.
    LinkType infl_link;  ///< Inverse link mapping the inflation linear predictor to \f$ \pi \f$.
    BaseDistribution base_dist;  ///< Base count distribution policy (copy): log-pmf, mean derivatives and parameter constraints.

    /**
     * @brief Builds the loss functor (matrices and response are referenced, `dist` is copied).
     * @param exog Base design matrix \f$ X \f$ (\f$ n \times kc \f$).
     * @param exog_infl Inflation design matrix \f$ Z \f$ (\f$ n \times kz \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$).
     * @param target_val Inflated count value \f$ k \f$ (e.g. 0 for zero inflation).
     * @param blink Link whose inverse maps the base linear predictor to \f$ \mu \f$.
     * @param ilink Link whose inverse maps the inflation linear predictor to \f$ \pi \f$.
     * @param dist Base distribution policy (e.g. NegativeBinomialComponent with the desired variance form).
     * @pre `exog`, `exog_infl` and `endog` have the same number of rows and outlive the functor.
     */
    NegativeBinomialInflatedLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_infl, const Eigen::VectorXd& endog,
                                 double target_val, LinkType blink, LinkType ilink, BaseDistribution dist)
        : X(exog), Z(exog_infl), y(endog), k_target(target_val), base_link(blink), infl_link(ilink), base_dist(dist) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$ of length \f$ kc+kz+1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$ (no finite check, no fallback value).
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long kc = X.cols();
        const long kz = Z.cols();
        const long n = X.rows();
        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double alpha_raw = params(kc + kz);
        double alpha = std::exp(alpha_raw);
        base_dist.constrain_dispersion(alpha);

        // linear predictors of the base and inflation parts
        Eigen::VectorXd eta_base = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;
        double total_neg_ll = 0.0;

        // accumulate per-observation terms (inflated class vs other counts)
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_i = 0.0;
            evaluate_link(eta_base(i), base_link, mu_i, dmu_i);
            base_dist.constrain_parameter(mu_i);

            double pi_i = 0.0; double dpi_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_i);
            double pi_c = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            double log_prob = 0.0;
            if (std::abs(y(i) - k_target) < 1e-9) {
                double p_base = std::exp(base_dist.log_pmf_mu(k_target, mu_i, alpha));
                log_prob = std::log(std::max(pi_c + (1.0 - pi_c) * p_base, 1e-15));
            } else {
                double log_not_pi = (infl_link == LinkType::Logit) ? -log1p_exp_stable(eta_infl(i)) : std::log(1.0 - pi_c);
                log_prob = log_not_pi + base_dist.log_pmf_mu(y(i), mu_i, alpha);
            }
            total_neg_ll -= log_prob;
        }
        return total_neg_ll;
    }

    /**
     * @brief Gradient of the negative log-likelihood (analytical in \f$ \boldsymbol\beta,\boldsymbol\gamma \f$, finite-difference in \f$ \log\alpha \f$).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$.
     * @return Vector of length \f$ kc+kz+1 \f$: \f$ X^\top(\partial\mathrm{NLL}/\partial\eta^{(b)}) \f$,
     *         \f$ Z^\top(\partial\mathrm{NLL}/\partial\eta^{(z)}) \f$ and
     *         \f$ \sum_i \alpha\,[\mathrm{NLL}_i(\alpha+\varepsilon) - \mathrm{NLL}_i(\alpha-\varepsilon)]/(2\varepsilon) \f$ with
     *         \f$ \varepsilon = 10^{-5} \f$. No finite masking is applied.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long kc = X.cols();
        const long kz = Z.cols();
        const long n = X.rows();
        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double alpha_raw = params(kc + kz);
        double alpha = std::exp(alpha_raw);
        base_dist.constrain_dispersion(alpha);

        // linear predictors of the base and inflation parts
        Eigen::VectorXd eta_base = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        Eigen::VectorXd dNLL_deta_base = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_infl = Eigen::VectorXd::Zero(n);
        double dNLL_dalpha_raw = 0.0;

        const double eps = 1e-5;

        // per-observation eta-weights and finite-difference dispersion terms
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_i = 0.0;
            evaluate_link(eta_base(i), base_link, mu_i, dmu_i);
            base_dist.constrain_parameter(mu_i);

            double pi_i = 0.0; double dpi_i = 0.0;
            evaluate_link(eta_infl(i), infl_link, pi_i, dpi_i);
            double pi_c = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            if (std::abs(y(i) - k_target) < 1e-9) {
                double p_base = std::exp(base_dist.log_pmf_mu(k_target, mu_i, alpha));
                double L_i = std::max(pi_c + (1.0 - pi_c) * p_base, 1e-15);
                double dlogP = base_dist.d_log_pmf_dmu(k_target, mu_i, alpha);
                dNLL_deta_base(i) = (-((1.0 - pi_c) * p_base * dlogP) / L_i) * dmu_i;
                dNLL_deta_infl(i) = (-(1.0 - p_base) / L_i) * dpi_i;
            } else {
                dNLL_deta_base(i) = -base_dist.d_log_pmf_dmu(y(i), mu_i, alpha) * dmu_i;
                if (infl_link == LinkType::Logit) {
                    dNLL_deta_infl(i) = pi_c;
                } else {
                    dNLL_deta_infl(i) = (1.0 / (1.0 - pi_c)) * dpi_i;
                }
            }

            // per-observation NLL as a function of alpha, for the central difference
            auto compute_NLL_alpha = [&](double a_val) {
                if (std::abs(y(i) - k_target) < 1e-9) {
                    return -std::log(std::max(pi_c + (1.0 - pi_c) * std::exp(base_dist.log_pmf_mu(k_target, mu_i, a_val)), 1e-15));
                } else {
                    double log_not_pi = (infl_link == LinkType::Logit) ? -log1p_exp_stable(eta_infl(i)) : std::log(1.0 - pi_c);
                    return -(log_not_pi + base_dist.log_pmf_mu(y(i), mu_i, a_val));
                }
            };
            double grad_alpha_i = (compute_NLL_alpha(alpha + eps) - compute_NLL_alpha(alpha - eps)) / (2.0 * eps);
            dNLL_dalpha_raw += grad_alpha_i * alpha;
        }

        Eigen::VectorXd grad(kc + kz + 1);
        grad.head(kc) = X.transpose() * dNLL_deta_base;
        grad.segment(kc, kz) = Z.transpose() * dNLL_deta_infl;
        grad(kc + kz) = dNLL_dalpha_raw;
        return grad;
    }

    /**
     * @brief Hessian of the negative log-likelihood (analytical eta-blocks, finite-difference dispersion row/column).
     * @details Per observation the weights \f$ w_{bb}, w_{gg}, w_{bg} \f$ are second derivatives with respect to
     * \f$ (\eta^{(b)}, \eta^{(z)}) \f$. For \f$ y = k \f$, with \f$ L = \pi + (1-\pi)P \f$ (floored at \f$ 10^{-15} \f$),
     * \f$ \mathrm{NLL} = -\log L \f$ and \f$ \partial_a\partial_b\mathrm{NLL} = (L_aL_b - L\,L_{ab})/L^2 \f$ with
     * \f$ L_\mu = (1-\pi)P s \f$, \f$ L_{\mu\mu} = (1-\pi)P(s' + s^2) \f$, \f$ L_\pi = 1-P \f$,
     * \f$ L_{\mu\pi} = -Ps \f$, where \f$ s = \partial_\mu\log f \f$ and \f$ s' = \partial^2_\mu\log f \f$. For
     * \f$ y \ne k \f$: \f$ w_{bb} = -s'\mu'^2 - s\,\mu'' \f$, \f$ w_{bg} = 0 \f$ and \f$ w_{gg} = \pi(1-\pi) \f$ for the
     * `Logit` link (otherwise \f$ \pi''/(1-\pi) + \pi'^2/(1-\pi)^2 \f$). The \f$ \alpha \f$-related entries use the
     * stencil described in the struct documentation (step \f$ 10^{-5} \f$ in \f$ \alpha \f$, multiplied by \f$ \alpha \f$ to
     * convert to \f$ \log\alpha \f$).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\alpha) \f$.
     * @return Symmetric \f$ (kc+kz+1)\times(kc+kz+1) \f$ matrix (blocks mirrored; no regularisation, no finite masking).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long kc = X.cols();
        const long kz = Z.cols();
        const long n = X.rows();
        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double alpha_raw = params(kc + kz);
        double alpha = std::exp(alpha_raw);
        base_dist.constrain_dispersion(alpha);

        // linear predictors of the base and inflation parts
        Eigen::VectorXd eta_base = X * beta;
        Eigen::VectorXd eta_infl = Z * gamma;

        Eigen::VectorXd w_bb = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd w_gg = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd w_bg = Eigen::VectorXd::Zero(n);

        double d2NLL_dalpha_raw2 = 0.0;
        Eigen::VectorXd d2NLL_dbeta_dalpha = Eigen::VectorXd::Zero(kc);
        Eigen::VectorXd d2NLL_dgamma_dalpha = Eigen::VectorXd::Zero(kz);
        const double eps = 1e-5;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_i = 0.0; double d2mu_i = 0.0;
            evaluate_link_d2(eta_base(i), base_link, mu_i, dmu_i, d2mu_i);
            base_dist.constrain_parameter(mu_i);

            double pi_i = 0.0; double dpi_i = 0.0; double d2pi_i = 0.0;
            evaluate_link_d2(eta_infl(i), infl_link, pi_i, dpi_i, d2pi_i);
            double pi_c = std::max(1e-15, std::min(1.0 - 1e-15, pi_i));

            if (std::abs(y(i) - k_target) < 1e-9) {

                double log_P_base = base_dist.log_pmf_mu(k_target, mu_i, alpha);
                double P_base = std::exp(log_P_base);
                double L_i = std::max(pi_c + (1.0 - pi_c) * P_base, 1e-15);

                double dlogP = base_dist.d_log_pmf_dmu(k_target, mu_i, alpha);
                double d2logP = base_dist.d2_log_pmf_dmu2(k_target, mu_i, alpha);

                double dL_dmu = (1.0 - pi_c) * P_base * dlogP;
                double d2L_dmu2 = (1.0 - pi_c) * P_base * (d2logP + dlogP * dlogP);
                double dL_dpi = 1.0 - P_base;
                double d2L_dmudpi = -P_base * dlogP;

                double dNLL_dmu = -dL_dmu / L_i;
                double d2NLL_dmu2 = (dL_dmu * dL_dmu - d2L_dmu2 * L_i) / (L_i * L_i);

                double dNLL_dpi = -dL_dpi / L_i;
                double d2NLL_dpi2 = (dL_dpi * dL_dpi) / (L_i * L_i);

                double d2NLL_dmudpi = (dL_dmu * dL_dpi - d2L_dmudpi * L_i) / (L_i * L_i);

                w_bb(i) = d2NLL_dmu2 * dmu_i * dmu_i + dNLL_dmu * d2mu_i;
                w_gg(i) = d2NLL_dpi2 * dpi_i * dpi_i + dNLL_dpi * d2pi_i;
                w_bg(i) = d2NLL_dmudpi * dmu_i * dpi_i;

            } else {

                double dlogP = base_dist.d_log_pmf_dmu(y(i), mu_i, alpha);
                double d2logP = base_dist.d2_log_pmf_dmu2(y(i), mu_i, alpha);

                double dNLL_dmu = -dlogP;
                double d2NLL_dmu2 = -d2logP;

                w_bb(i) = d2NLL_dmu2 * dmu_i * dmu_i + dNLL_dmu * d2mu_i;

                if (infl_link == LinkType::Logit) {

                    w_gg(i) = pi_c * (1.0 - pi_c);
                } else {
                    double dNLL_dpi = 1.0 / (1.0 - pi_c);
                    double d2NLL_dpi2 = 1.0 / ((1.0 - pi_c) * (1.0 - pi_c));
                    w_gg(i) = d2NLL_dpi2 * dpi_i * dpi_i + dNLL_dpi * d2pi_i;
                }

                w_bg(i) = 0.0;
            }

            auto compute_NLL_alpha = [&](double a_val) {
                if (std::abs(y(i) - k_target) < 1e-9) {
                    return -std::log(std::max(pi_c + (1.0 - pi_c) * std::exp(base_dist.log_pmf_mu(k_target, mu_i, a_val)), 1e-15));
                } else {
                    double log_not_pi = (infl_link == LinkType::Logit) ? -log1p_exp_stable(eta_infl(i)) : std::log(1.0 - pi_c);
                    return -(log_not_pi + base_dist.log_pmf_mu(y(i), mu_i, a_val));
                }
            };

            // finite differences in alpha: second derivative of the NLL
            double nll_0 = compute_NLL_alpha(alpha);
            double nll_plus = compute_NLL_alpha(alpha + eps);
            double nll_minus = compute_NLL_alpha(alpha - eps);

            double grad_alpha_i = (nll_plus - nll_minus) / (2.0 * eps);
            double hess_alpha_i = (nll_plus - 2.0 * nll_0 + nll_minus) / (eps * eps);

            d2NLL_dalpha_raw2 += hess_alpha_i * alpha * alpha + grad_alpha_i * alpha;

            // eta-gradient re-evaluated at alpha +/- eps, for the cross derivatives
            auto get_dNLL_deta_at_alpha = [&](double a_val, double& g_base, double& g_infl) {
                if (std::abs(y(i) - k_target) < 1e-9) {
                    double p_b = std::exp(base_dist.log_pmf_mu(k_target, mu_i, a_val));
                    double L_local = std::max(pi_c + (1.0 - pi_c) * p_b, 1e-15);
                    g_base = (-((1.0 - pi_c) * p_b * base_dist.d_log_pmf_dmu(k_target, mu_i, a_val)) / L_local) * dmu_i;
                    g_infl = (-(1.0 - p_b) / L_local) * dpi_i;
                } else {
                    g_base = -base_dist.d_log_pmf_dmu(y(i), mu_i, a_val) * dmu_i;
                    g_infl = (infl_link == LinkType::Logit) ? pi_c : (1.0 / (1.0 - pi_c)) * dpi_i;
                }
            };

            double g_base_p, g_infl_p, g_base_m, g_infl_m;
            get_dNLL_deta_at_alpha(alpha + eps, g_base_p, g_infl_p);
            get_dNLL_deta_at_alpha(alpha - eps, g_base_m, g_infl_m);

            d2NLL_dbeta_dalpha += ((g_base_p - g_base_m) / (2.0 * eps) * alpha) * X.row(i).transpose();
            d2NLL_dgamma_dalpha += ((g_infl_p - g_infl_m) / (2.0 * eps) * alpha) * Z.row(i).transpose();
        }

        // assemble the Hessian blocks (mirrored to keep symmetry)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kc + kz + 1, kc + kz + 1);

        H.block(0, 0, kc, kc) = X.transpose() * w_bb.asDiagonal() * X;
        H.block(kc, kc, kz, kz) = Z.transpose() * w_gg.asDiagonal() * Z;

        Eigen::MatrixXd H_cross = X.transpose() * w_bg.asDiagonal() * Z;
        H.block(0, kc, kc, kz) = H_cross;
        H.block(kc, 0, kz, kc) = H_cross.transpose();

        H.block(0, kc + kz, kc, 1) = d2NLL_dbeta_dalpha;
        H.block(kc + kz, 0, 1, kc) = d2NLL_dbeta_dalpha.transpose();

        H.block(kc, kc + kz, kz, 1) = d2NLL_dgamma_dalpha;
        H.block(kc + kz, kc, 1, kz) = d2NLL_dgamma_dalpha.transpose();

        H(kc + kz, kc + kz) = d2NLL_dalpha_raw2;

        return H;
    }
};

/**
 * @brief Negative log-likelihood of the Beta regression model with covariate-dependent mean and precision.
 *
 * @details Models a response \f$ y_i \in (0,1) \f$ with
 * \f$ y_i \sim \mathrm{Beta}(p_i, q_i) \f$ and density
 * \f$ \frac{\Gamma(p+q)}{\Gamma(p)\Gamma(q)}\,y^{p-1}(1-y)^{q-1} \f$. The mean model is
 * \f$ \mu_i = g_\mu^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$ and the precision/dispersion model is
 * \f$ \phi_i = g_\phi^{-1}(\mathbf{z}_i^\top\boldsymbol\gamma) \f$, both evaluated by `BetaMath` helpers (defaults: `Logit`
 * for \f$ \mu \f$, `Log` for \f$ \phi \f$). The shape parameters depend on `prec_form`:
 * | `PrecisionForm` | \f$ p \f$ | \f$ q \f$ | \f$ p+q \f$ | Variance |
 * |-----------------|-----------|-----------|-------------|----------|
 * | `Standard`      | \f$ \mu\phi \f$ | \f$ (1-\mu)\phi \f$ | \f$ \phi \f$ | \f$ \mu(1-\mu)/(1+\phi) \f$ |
 * | `Simis`         | \f$ \mu/\phi \f$ | \f$ (1-\mu)/\phi \f$ | \f$ 1/\phi \f$ | \f$ \mu(1-\mu)\,\phi/(1+\phi) \f$ |
 *
 * (`Standard`: \f$ \phi \f$ is the precision; `Simis`: \f$ \phi \f$ enters as the reciprocal of the precision.)
 *
 * ### Parameter vector layout
 * | Index                 | Meaning                                                    | Scale                           |
 * |-----------------------|------------------------------------------------------------|---------------------------------|
 * | `0 .. k_mu-1`         | mean coefficients \f$ \boldsymbol\beta \f$ (columns of `X`)  | linear predictor of \f$ \mu \f$  |
 * | `k_mu ..`  (rest)     | precision coefficients \f$ \boldsymbol\gamma \f$ (columns of `Z`) | linear predictor of \f$ \phi \f$ |
 *
 * In `operator()` \f$ \boldsymbol\gamma \f$ is "everything after the first `k_mu` entries"; in `gradient()` and `hessian()` it is the
 * last `Z.cols()` entries. Both agree when the vector has length \f$ k_\mu + k_\phi \f$. There are no other parameters.
 *
 * ### Log-likelihood
 * With \f$ \tilde y_i = \min\{\max\{y_i,10^{-10}\},1-10^{-10}\} \f$:
 * \f[ \ell_i = \log\Gamma(p+q) - \log\Gamma(p) - \log\Gamma(q) + (p-1)\log\tilde y_i + (q-1)\log(1-\tilde y_i). \f]
 * The full density is used (no constant dropped). The functor returns \f$ -\sum_i \ell_i \f$.
 *
 * ### Derivatives
 * Let \f$ N_p = \psi(p) - \psi(p+q) - \log\tilde y \f$, \f$ N_q = \psi(q) - \psi(p+q) - \log(1-\tilde y) \f$ be the
 * derivatives of \f$ \mathrm{NLL}_i \f$ with respect to \f$ p \f$ and \f$ q \f$, and
 * \f$ N_{pp} = \psi_1(p)-\psi_1(p+q) \f$, \f$ N_{qq} = \psi_1(q)-\psi_1(p+q) \f$, \f$ N_{pq} = -\psi_1(p+q) \f$.
 * - `Standard`: \f$ N_\mu = (N_p - N_q)\phi \f$, \f$ N_\phi = \mu N_p + (1-\mu)N_q \f$.
 * - `Simis`: \f$ N_\mu = (N_p - N_q)/\phi \f$, \f$ N_\phi = -[\mu N_p + (1-\mu)N_q]/\phi^2 \f$.
 *
 * Gradient: \f$ (X^\top (N_\mu\,\mu'),\; Z^\top (N_\phi\,\phi')) \f$ with \f$ \mu' = d\mu/d\eta^{(\mu)} \f$ and
 * \f$ \phi' = d\phi/d\eta^{(\phi)} \f$. The Hessian is the exact analytical observed Hessian (second derivatives
 * \f$ N_{\mu\mu}, N_{\mu\phi}, N_{\phi\phi} \f$ coded separately for both forms): \f$ H_{\mu\mu} = X^\top\mathrm{diag}(N_{\mu\mu}\mu'^2 + N_\mu\mu'')X \f$,
 * \f$ H_{\phi\phi} = Z^\top\mathrm{diag}(N_{\phi\phi}\phi'^2 + N_\phi\phi'')Z \f$ and
 * \f$ H_{\mu\phi} = X^\top\mathrm{diag}(N_{\mu\phi}\mu'\phi')Z \f$ (no link-curvature term in the cross block). Non-finite entries
 * are replaced by 0; there is no ridge and no PSD projection; the matrix is symmetric by construction.
 *
 * ### Numerical safeguards
 * - `operator()` returns `1e10` if the sum is not finite; `gradient()` and `hessian()` replace non-finite entries by 0.
 * - Response clipped to \f$ [10^{-10}, 1-10^{-10}] \f$ (in all three methods).
 * - \f$ \phi \f$ is floored at \f$ 10^{-4} \f$ (`Standard`) or \f$ 10^{-3} \f$ (`Simis`) after the link evaluation; \f$ \phi' \f$
 *   and \f$ \phi'' \f$ are not adjusted by the floor.
 * - Mean link (`BetaMath::evaluate_link[_hessian]`): \f$ \eta \f$ is clamped per link (`Logit` \f$ [-30,30] \f$, `Probit`
 *   \f$ [-5,5] \f$, `Cloglog` \f$ [-30,10] \f$, `Log` \f$ [-30,20] \f$, `Loglog` \f$ [-10,30] \f$, `Cauchit`
 *   \f$ \pm 10^{8} \f$); for the bounded links \f$ \mu \f$ is clamped to \f$ [10^{-14}, 1-10^{-14}] \f$. `Identity`, `Sqrt` and
 *   other unlisted links are silently replaced by `Logit` (with \f$ [-30,30] \f$ clamp); `Inverse` is kept (and its \f$ \mu \f$ is not
 *   confined to \f$ (0,1) \f$).
 * - Precision link (`BetaMath::evaluate_precision_link[_d2]`): `Identity`/`Sqrt` use \f$ \eta \f$ clamped to
 *   \f$ [-10^{6}, 10^{6}] \f$ directly; `Inverse` uses a smooth positive inverse; other links (e.g. `Log`) go through the mean-link
 *   helper with its clamps.
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y`; they must outlive the functor. No
 *       validation is done: \f$ y \f$ should lie in \f$ (0,1) \f$ (boundary values are silently clipped).
 * @note The analytical gradient and Hessian use `BetaMath::digamma`/`trigamma` (recurrence plus asymptotic series,
 *       absolute error below \f$ 10^{-9} \f$), while the value uses `std::lgamma`.
 * @note The link evaluation loops are parallelised with OpenMP (`n > 5000`) in `gradient()` and `hessian()`.
 * @see HurdleBetaLoss, SimplexLoss
 *
 * @ingroup glm_ll_proportion
 */
struct BetaRegressionLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the mean model (\f$ n \times k_\mu \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the precision/dispersion model (\f$ n \times k_\phi \f$).
    const Eigen::VectorXd& y;  ///< Response in \f$ (0,1) \f$ (length \f$ n \f$); clipped to \f$ [10^{-10}, 1-10^{-10}] \f$ when used.
    LinkType mu_link;  ///< Link whose inverse maps \f$ \eta^{(\mu)} \f$ to the mean \f$ \mu \in (0,1) \f$.
    LinkType phi_link;  ///< Link whose inverse maps \f$ \eta^{(\phi)} \f$ to the positive precision parameter \f$ \phi \f$.
    PrecisionForm prec_form;  ///< Parametrisation of the shape parameters: `Standard` (\f$ p=\mu\phi \f$) or `Simis` (\f$ p=\mu/\phi \f$).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog_mu Mean design matrix \f$ X \f$ (\f$ n \times k_\mu \f$).
     * @param exog_phi Precision design matrix \f$ Z \f$ (\f$ n \times k_\phi \f$).
     * @param endog Response \f$ y \f$ (length \f$ n \f$) with values in \f$ (0,1) \f$.
     * @param m_link Link of the mean (default `Logit`).
     * @param p_link Link of the precision/dispersion parameter (default `Log`).
     * @param form Precision parametrisation (default `Standard`).
     * @pre `exog_mu`, `exog_phi` and `endog` have the same number of rows and outlive the functor.
     */
    BetaRegressionLoss(const Eigen::MatrixXd& exog_mu, const Eigen::MatrixXd& exog_phi, const Eigen::VectorXd& endog,
                       LinkType m_link = LinkType::Logit, LinkType p_link = LinkType::Log,
                       PrecisionForm form = PrecisionForm::Standard)
        : X(exog_mu), Z(exog_phi), y(endog), mu_link(m_link), phi_link(p_link), prec_form(form) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_\mu + k_\phi \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if it is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_mu = X.cols();

        Eigen::VectorXd beta = params.head(k_mu);
        Eigen::VectorXd gamma = params.tail(params.size() - k_mu);

        // linear predictors of the mean and precision models
        Eigen::VectorXd eta_mu = X * beta;
        Eigen::VectorXd eta_phi = Z * gamma;

        double total_neg_ll = 0.0;

        // accumulate per-observation log-likelihood terms
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            BetaMath::evaluate_link(eta_mu(i), mu_link, mu_i, dmu_deta_i);

            double phi_i = 0.0; double dphi_deta_phi_i = 0.0;
            BetaMath::evaluate_precision_link(eta_phi(i), phi_link, phi_i, dphi_deta_phi_i);

            const double phi_min = (prec_form == PrecisionForm::Standard) ? 1e-4 : 1e-3;
            if (phi_i < phi_min) phi_i = phi_min;

            double p = 0.0; double q = 0.0; double sum_pq = 0.0;
            if (prec_form == PrecisionForm::Standard) {
                p = mu_i * phi_i;
                q = (1.0 - mu_i) * phi_i;
                sum_pq = phi_i;
            } else {
                p = mu_i / phi_i;
                q = (1.0 - mu_i) / phi_i;
                sum_pq = 1.0 / phi_i;
            }

            double yi = std::max(1e-10, std::min(1.0 - 1e-10, y(i)));

            double log_lik_i = std::lgamma(sum_pq) - std::lgamma(p) - std::lgamma(q)
                             + (p - 1.0) * std::log(yi) + (q - 1.0) * std::log(1.0 - yi);

            total_neg_ll -= log_lik_i;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood (vectorised implementation).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @return Vector of length \f$ k_\mu + k_\phi \f$: \f$ X^\top (N_\mu\,\mu') \f$ followed by \f$ Z^\top (N_\phi\,\phi') \f$ (see the struct
     *         documentation); non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_mu = X.cols();
        const long k_phi = Z.cols();

        Eigen::Ref<const Eigen::VectorXd> beta = params.head(k_mu);
        Eigen::Ref<const Eigen::VectorXd> gamma = params.tail(k_phi);

        // linear predictors of the mean and precision models
        Eigen::VectorXd eta_mu = X * beta;
        Eigen::VectorXd eta_phi = Z * gamma;

        Eigen::VectorXd mu(n), dmu_deta(n), phi(n), dphi_deta_phi(n);

        // link values and first derivatives (parallel for large n)
        #pragma omp parallel for if(n > 5000)
        for (long i = 0; i < n; ++i) {
            BetaMath::evaluate_link(eta_mu(i), mu_link, mu(i), dmu_deta(i));
            BetaMath::evaluate_precision_link(eta_phi(i), phi_link, phi(i), dphi_deta_phi(i));
        }

        const double phi_min = (prec_form == PrecisionForm::Standard) ? 1e-4 : 1e-3;
        phi = phi.cwiseMax(phi_min);

        auto mu_arr = mu.array();
        auto phi_arr = phi.array();

        Eigen::ArrayXd p(n), q(n), sum_pq(n);
        if (prec_form == PrecisionForm::Standard) {
            p = mu_arr * phi_arr;
            q = (1.0 - mu_arr) * phi_arr;
            sum_pq = phi_arr;
        } else {
            Eigen::ArrayXd phi_inv = phi_arr.inverse();
            p = mu_arr * phi_inv;
            q = (1.0 - mu_arr) * phi_inv;
            sum_pq = phi_inv;
        }

        Eigen::ArrayXd yi = y.array().cwiseMax(1e-10).cwiseMin(1.0 - 1e-10);
        Eigen::ArrayXd log_yi = yi.log();
        Eigen::ArrayXd log_one_minus_yi = (1.0 - yi).log();

        auto digamma_lambda = [](double x) { return BetaMath::digamma(x); };
        Eigen::ArrayXd digam_sum = sum_pq.unaryExpr(digamma_lambda);
        Eigen::ArrayXd digam_p   = p.unaryExpr(digamma_lambda);
        Eigen::ArrayXd digam_q   = q.unaryExpr(digamma_lambda);

        // derivatives with respect to the beta shape parameters p and q
        Eigen::ArrayXd dNLL_dp = digam_p - digam_sum - log_yi;
        Eigen::ArrayXd dNLL_dq = digam_q - digam_sum - log_one_minus_yi;

        Eigen::ArrayXd dNLL_dmu(n), dNLL_dphi(n);

        if (prec_form == PrecisionForm::Standard) {
            dNLL_dmu = (dNLL_dp - dNLL_dq) * phi_arr;
            dNLL_dphi = dNLL_dp * mu_arr + dNLL_dq * (1.0 - mu_arr);
        } else {
            dNLL_dmu = (dNLL_dp - dNLL_dq) / phi_arr;

            dNLL_dphi = -(dNLL_dp * mu_arr + dNLL_dq * (1.0 - mu_arr)) / phi_arr.square();
        }

        Eigen::VectorXd dNLL_deta_mu  = (dNLL_dmu * dmu_deta.array()).matrix();
        Eigen::VectorXd dNLL_deta_phi = (dNLL_dphi * dphi_deta_phi.array()).matrix();

        // chain rule to the regression coefficients
        Eigen::VectorXd grad(k_mu + k_phi);
        grad.head(k_mu) = X.transpose() * dNLL_deta_mu;
        grad.tail(k_phi) = Z.transpose() * dNLL_deta_phi;

        auto finite_lambda = [](double x) { return std::isfinite(x) ? x : 0.0; };
        return grad.unaryExpr(finite_lambda);
    }

    /**
     * @brief Exact analytical observed Hessian of the negative log-likelihood.
     * @details Builds the per-observation second derivatives \f$ N_{\mu\mu}, N_{\mu\phi}, N_{\phi\phi} \f$ from digamma and
     * trigamma values for the chosen `prec_form`, then forms the weights
     * \f$ W_{\mu\mu} = N_{\mu\mu}\mu'^2 + N_\mu\mu'' \f$, \f$ W_{\phi\phi} = N_{\phi\phi}\phi'^2 + N_\phi\phi'' \f$ and
     * \f$ W_{\mu\phi} = N_{\mu\phi}\mu'\phi' \f$ and the blocks
     * \f$ X^\top\mathrm{diag}(W_{\mu\mu})X \f$, \f$ Z^\top\mathrm{diag}(W_{\phi\phi})Z \f$, \f$ X^\top\mathrm{diag}(W_{\mu\phi})Z \f$.
     * The link second derivatives come from `BetaMath::evaluate_link_hessian` and `BetaMath::evaluate_precision_link_d2`.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @return Symmetric \f$ (k_\mu+k_\phi)\times(k_\mu+k_\phi) \f$ matrix; non-finite entries are set to 0. No ridge, no
     *         PSD projection.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_mu = X.cols();
        const long k_phi = Z.cols();

        Eigen::Ref<const Eigen::VectorXd> beta = params.head(k_mu);
        Eigen::Ref<const Eigen::VectorXd> gamma = params.tail(k_phi);

        // linear predictors of the mean and precision models
        Eigen::VectorXd eta_mu = X * beta;
        Eigen::VectorXd eta_phi = Z * gamma;

        Eigen::VectorXd mu(n), dmu_deta(n), d2mu_deta2(n);
        Eigen::VectorXd phi(n), dphi_deta_phi(n), d2phi_deta2(n);

        // link values with first and second derivatives (parallel for large n)
        #pragma omp parallel for if(n > 5000)
        for (long i = 0; i < n; ++i) {
            BetaMath::evaluate_link_hessian(eta_mu(i), mu_link, mu(i), dmu_deta(i), d2mu_deta2(i));
            BetaMath::evaluate_precision_link_d2(eta_phi(i), phi_link, phi(i), dphi_deta_phi(i), d2phi_deta2(i));
        }

        const double phi_min = (prec_form == PrecisionForm::Standard) ? 1e-4 : 1e-3;
        phi = phi.cwiseMax(phi_min);

        auto mu_arr = mu.array();
        auto phi_arr = phi.array();

        Eigen::ArrayXd p(n), q(n), sum_pq(n);
        if (prec_form == PrecisionForm::Standard) {
            p = mu_arr * phi_arr;
            q = (1.0 - mu_arr) * phi_arr;
            sum_pq = phi_arr;
        } else {
            Eigen::ArrayXd phi_inv = phi_arr.inverse();
            p = mu_arr * phi_inv;
            q = (1.0 - mu_arr) * phi_inv;
            sum_pq = phi_inv;
        }

        Eigen::ArrayXd yi = y.array().cwiseMax(1e-10).cwiseMin(1.0 - 1e-10);
        Eigen::ArrayXd log_yi = yi.log();
        Eigen::ArrayXd log_one_minus_yi = (1.0 - yi).log();

        auto digamma_lambda = [](double x) { return BetaMath::digamma(x); };
        auto trigamma_lambda = [](double x) { return BetaMath::trigamma(x); };

        Eigen::ArrayXd digam_sum = sum_pq.unaryExpr(digamma_lambda);
        Eigen::ArrayXd digam_p   = p.unaryExpr(digamma_lambda);
        Eigen::ArrayXd digam_q   = q.unaryExpr(digamma_lambda);

        Eigen::ArrayXd trigam_sum = sum_pq.unaryExpr(trigamma_lambda);
        Eigen::ArrayXd trigam_p   = p.unaryExpr(trigamma_lambda);
        Eigen::ArrayXd trigam_q   = q.unaryExpr(trigamma_lambda);

        Eigen::ArrayXd dNLL_dp = digam_p - digam_sum - log_yi;
        Eigen::ArrayXd dNLL_dq = digam_q - digam_sum - log_one_minus_yi;

        // second derivatives with respect to the beta shape parameters
        Eigen::ArrayXd d2NLL_dp2  = trigam_p - trigam_sum;
        Eigen::ArrayXd d2NLL_dq2  = trigam_q - trigam_sum;
        Eigen::ArrayXd d2NLL_dpdq = -trigam_sum;

        Eigen::ArrayXd dNLL_dmu(n), dNLL_dphi(n);
        Eigen::ArrayXd d2NLL_dmu2(n), d2NLL_dphi2(n), d2NLL_dmudphi(n);

        if (prec_form == PrecisionForm::Standard) {

            dNLL_dmu = (dNLL_dp - dNLL_dq) * phi_arr;
            dNLL_dphi = dNLL_dp * mu_arr + dNLL_dq * (1.0 - mu_arr);

            d2NLL_dmu2 = (d2NLL_dp2 - 2.0 * d2NLL_dpdq + d2NLL_dq2) * phi_arr.square();

            d2NLL_dmudphi = (dNLL_dp - dNLL_dq)
                          + phi_arr * (mu_arr * d2NLL_dp2 + (1.0 - 2.0 * mu_arr) * d2NLL_dpdq - (1.0 - mu_arr) * d2NLL_dq2);

            d2NLL_dphi2 = d2NLL_dp2 * mu_arr.square()
                        + 2.0 * d2NLL_dpdq * mu_arr * (1.0 - mu_arr)
                        + d2NLL_dq2 * (1.0 - mu_arr).square();
        } else {
            Eigen::ArrayXd phi_sq = phi_arr.square();
            Eigen::ArrayXd phi_cub = phi_sq * phi_arr;

            dNLL_dmu = (dNLL_dp - dNLL_dq) / phi_arr;
            dNLL_dphi = -(dNLL_dp * mu_arr + dNLL_dq * (1.0 - mu_arr)) / phi_sq;

            d2NLL_dmu2 = (d2NLL_dp2 - 2.0 * d2NLL_dpdq + d2NLL_dq2) / phi_sq;

            d2NLL_dmudphi = -(dNLL_dp - dNLL_dq) / phi_sq
                          - (mu_arr * d2NLL_dp2 + (1.0 - 2.0 * mu_arr) * d2NLL_dpdq - (1.0 - mu_arr) * d2NLL_dq2) / phi_cub;

            d2NLL_dphi2 = 2.0 * (dNLL_dp * mu_arr + dNLL_dq * (1.0 - mu_arr)) / phi_cub
                        + (d2NLL_dp2 * mu_arr.square() + 2.0 * d2NLL_dpdq * mu_arr * (1.0 - mu_arr) + d2NLL_dq2 * (1.0 - mu_arr).square()) / (phi_sq * phi_sq);
        }

        Eigen::ArrayXd dmu_arr = dmu_deta.array();
        Eigen::ArrayXd dphi_arr = dphi_deta_phi.array();

        // per-observation weights including link curvature
        Eigen::VectorXd W_mumu = (d2NLL_dmu2 * dmu_arr.square() + dNLL_dmu * d2mu_deta2.array()).matrix();
        Eigen::VectorXd W_phiphi = (d2NLL_dphi2 * dphi_arr.square() + dNLL_dphi * d2phi_deta2.array()).matrix();
        Eigen::VectorXd W_muphi = (d2NLL_dmudphi * dmu_arr * dphi_arr).matrix();

        // assemble Hessian blocks (cross block mirrored)
        Eigen::MatrixXd H(k_mu + k_phi, k_mu + k_phi);

        H.block(0, 0, k_mu, k_mu) = X.transpose() * W_mumu.asDiagonal() * X;

        H.block(k_mu, k_mu, k_phi, k_phi) = Z.transpose() * W_phiphi.asDiagonal() * Z;

        Eigen::MatrixXd H_bg = X.transpose() * W_muphi.asDiagonal() * Z;
        H.block(0, k_mu, k_mu, k_phi) = H_bg;
        H.block(k_mu, 0, k_phi, k_mu) = H_bg.transpose();

        auto finite_lambda = [](double x) { return std::isfinite(x) ? x : 0.0; };
        return H.unaryExpr(finite_lambda);
    }

};

/**
 * @brief Negative log-likelihood of a two-part (hurdle) Beta regression for responses in \f$ [0,1) \f$ with point mass at zero.
 *
 * @details Models a proportion \f$ y_i \in [0,1) \f$ as a mixture of a point mass at 0 and a Beta density:
 * \f[ \Pr(y_i = 0) = 1-\pi_i, \qquad y_i \mid y_i > 0 \sim \mathrm{Beta}(\mu_i\phi,\,(1-\mu_i)\phi), \f]
 * so that \f$ \mathbb{E}[y_i \mid y_i>0] = \mu_i \f$ and \f$ \phi \f$ is the precision (\f$ p_i = \mu_i\phi \f$,
 * \f$ q_i = (1-\mu_i)\phi \f$ are the usual Beta shape parameters). Two linear predictors are used:
 * - the **hurdle (zero) part**: \f$ \pi_i = g_z^{-1}(\mathbf{z}_i^\top\boldsymbol\gamma) \f$ is the probability that
 *   the response is *strictly positive* (the hurdle is crossed); `zero_link` is \f$ g_z \f$ (default Logit);
 * - the **mean part**: \f$ \mu_i = g_\mu^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$ is the mean of the Beta component;
 *   `mean_link` is \f$ g_\mu \f$ (default Logit).
 *
 * Both inverse links are evaluated with `BetaMath::evaluate_link` (value and first derivative) or
 * `BetaMath::evaluate_link_hessian` (value, first and second derivative). That helper clamps \f$ \eta \f$ before
 * applying the link (Logit \f$ \pm 30 \f$, Probit \f$ \pm 5 \f$, Cloglog \f$ [-30,10] \f$, Log \f$ [-30,20] \f$,
 * Loglog \f$ [-10,30] \f$, Cauchit \f$ \pm 10^8 \f$), clamps the result to \f$ [10^{-14}, 1-10^{-14}] \f$ for the
 * bounded links (Logit, Probit, Cloglog, Loglog, Cauchit) and silently replaces every unsupported link
 * (Identity, Sqrt, InverseSquared) by Logit (with \f$ \eta \f$ clamped to \f$ \pm 30 \f$). `Inverse` is evaluated as such
 * (\f$ \eta \f$ clamped to \f$ \pm 30 \f$). The `Log` link is *not* clamped into \f$ (0,1) \f$.
 *
 * ### Parameter vector layout
 * | Index                 | Meaning                                                     | Scale |
 * |-----------------------|-------------------------------------------------------------|-------|
 * | `0 .. kc-1`           | mean coefficients \f$ \boldsymbol\beta \f$ (columns of `X`)  | linear predictor of \f$ \mu \f$ |
 * | `kc .. kc+kz-1`       | hurdle coefficients \f$ \boldsymbol\gamma \f$ (columns of `Z`) | linear predictor of \f$ \pi \f$ |
 * | `kc+kz`               | \f$ \log\phi \f$                                              | log; \f$ \phi = \max\{\exp(\mathrm{clamp}(\log\phi,-10,10)),\,10^{-4}\} \f$ |
 *
 * where \f$ kc \f$ = `X.cols()` and \f$ kz \f$ = `Z.cols()`. The total length is \f$ kc+kz+1 \f$.
 *
 * ### Log-likelihood
 * With \f$ \pi_i \f$ clamped to \f$ [10^{-12}, 1-10^{-12}] \f$ and \f$ \tilde y_i = \mathrm{clamp}(y_i, 10^{-10}, 1-10^{-10}) \f$:
 * - zero observation (\f$ y_i < 10^{-12} \f$): \f$ \ell_i = \log(1-\pi_i) \f$;
 * - positive observation:
 *   \f[ \ell_i = \log\pi_i + \log\Gamma(\phi) - \log\Gamma(p_i) - \log\Gamma(q_i)
 *       + (p_i-1)\log\tilde y_i + (q_i-1)\log(1-\tilde y_i). \f]
 *
 * The functor returns \f$ -\sum_i \ell_i \f$. All constants are kept (no weights, no censoring).
 *
 * ### Derivatives
 * The model is separable: the hurdle part depends only on \f$ \boldsymbol\gamma \f$ and the Beta part only on
 * \f$ (\boldsymbol\beta, \log\phi) \f$. Per observation, with \f$ \psi \f$ the digamma function (`BetaMath::digamma`),
 * - hurdle part: \f$ \partial\mathrm{NLL}_i/\partial\pi_i = 1/(1-\pi_i) \f$ (zero) or \f$ -1/\pi_i \f$ (positive),
 *   chained through \f$ d\pi_i/d\eta^{(z)}_i \f$;
 * - Beta part (positive observations only):
 *   \f$ \partial\mathrm{NLL}_i/\partial p_i = \psi(p_i)-\psi(\phi)-\log\tilde y_i \f$,
 *   \f$ \partial\mathrm{NLL}_i/\partial q_i = \psi(q_i)-\psi(\phi)-\log(1-\tilde y_i) \f$,
 *   \f$ \partial\mathrm{NLL}_i/\partial\mu_i = \phi\,(\partial_p - \partial_q) \f$ and
 *   \f$ \partial\mathrm{NLL}_i/\partial\phi = \mu_i\,\partial_p + (1-\mu_i)\,\partial_q \f$, then chained through
 *   \f$ d\mu_i/d\eta^{(\mu)}_i \f$ and \f$ d\phi/d\log\phi = \phi \f$.
 *
 * The gradient is analytic. The Hessian is *hybrid*: the \f$ \boldsymbol\gamma \f$ block is analytic, the
 * \f$ (\boldsymbol\beta,\log\phi) \f$ block is obtained by central finite differences of the analytic gradient
 * (step \f$ 10^{-5} \f$), and the cross blocks between \f$ \boldsymbol\gamma \f$ and the rest are exactly zero.
 *
 * ### Numerical safeguards
 * - \f$ \phi \f$ is floored at \f$ 10^{-4} \f$ and \f$ \log\phi \f$ is clamped to \f$ [-10,10] \f$; \f$ \pi_i \f$ and
 *   \f$ y_i \f$ are clamped as described above (the derivatives of the links are *not* zeroed when a clamp is active).
 * - `operator()` returns `1e10` when the sum is not finite.
 * - `gradient` replaces non-finite components by 0.
 * - `hessian` replaces non-finite entries by 0 and sets every non-positive diagonal entry to `1e-4`.
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y`; they must outlive the functor. No
 *       validation is performed: the three inputs must have the same number of rows, \f$ y_i \in [0,1) \f$ is
 *       assumed (values \f$ <10^{-12} \f$, including negative ones, are treated as zeros; \f$ y_i \ge 1 \f$ is
 *       clamped to \f$ 1-10^{-10} \f$ inside the Beta density).
 * @note With the `Log` mean link \f$ \mu_i \f$ may exceed 1, making \f$ q_i \f$ negative and the density undefined
 *       (`lgamma` of a negative argument); use a bounded link for \f$ \mu \f$.
 * @see HurdlePoissonLoss, HurdleNBLoss, HurdleGammaLoss, HurdleLogNormalLoss, BetaRegressionLoss
 *
 * @ingroup glm_ll_hurdle
 */
struct HurdleBetaLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the Beta mean model (\f$ n \times kc \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the hurdle (probability of a positive response) model (\f$ n \times kz \f$).
    const Eigen::VectorXd& y;  ///< Response in \f$ [0,1) \f$ (length \f$ n \f$); values below \f$ 10^{-12} \f$ are treated as zeros.
    LinkType mean_link;  ///< Link of the Beta mean \f$ \mu \f$, evaluated through `BetaMath::evaluate_link`.
    LinkType zero_link;  ///< Link of the positive-response probability \f$ \pi \f$, evaluated through `BetaMath`.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ of the Beta mean (\f$ n \times kc \f$).
     * @param exog_zero Design matrix \f$ Z \f$ of the hurdle part (\f$ n \times kz \f$).
     * @param endog Response \f$ y \f$ (length \f$ n \f$), values in \f$ [0,1) \f$ with exact zeros allowed.
     * @param blink Link of the Beta mean (default `LinkType::Logit`); bounded links are recommended.
     * @param zlink Link of the probability of a positive response (default `LinkType::Logit`).
     * @pre `exog`, `exog_zero` and `endog` have the same number of rows and outlive the functor.
     */
    HurdleBetaLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& exog_zero, const Eigen::VectorXd& endog,
                   LinkType blink = LinkType::Logit, LinkType zlink = LinkType::Logit)
        : X(exog), Z(exog_zero), y(endog), mean_link(blink), zero_link(zlink) {}

    /**
     * @brief Evaluates the total negative log-likelihood of the hurdle Beta model.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\phi) \f$ of length \f$ kc+kz+1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$ (zero observations contribute \f$ -\log(1-\pi_i) \f$, positive ones
     *         \f$ -\log\pi_i - \log f_{\mathrm{Beta}}(\tilde y_i) \f$), or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        // unpack parameters and linear predictors
        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_phi = params(kc + kz);
        double phi = std::max(std::exp(std::max(-10.0, std::min(10.0, log_phi))), 1e-4);

        Eigen::VectorXd eta_mean = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        double total_neg_ll = 0.0;
        for (long i = 0; i < n; ++i) {
            // inverse links of the mean and of the positive-response probability
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            BetaMath::evaluate_link(eta_mean(i), mean_link, mu_i, dmu_deta_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            BetaMath::evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            // zero mass vs. Beta density of a positive response
            double log_prob_i = 0.0;
            if (y(i) < 1e-12) {
                log_prob_i = std::log(1.0 - pi_i);
            } else {
                double p = mu_i * phi;
                double q = (1.0 - mu_i) * phi;
                double yi = std::max(1e-10, std::min(1.0 - 1e-10, y(i)));
                double log_beta_pdf = std::lgamma(phi) - std::lgamma(p) - std::lgamma(q)
                                     + (p - 1.0) * std::log(yi) + (q - 1.0) * std::log(1.0 - yi);
                log_prob_i = std::log(pi_i) + log_beta_pdf;
            }
            total_neg_ll -= log_prob_i;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytic gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\phi) \f$.
     * @return Vector of length \f$ kc+kz+1 \f$: \f$ X^\top \mathbf{g}^{(\mu)} \f$ for \f$ \boldsymbol\beta \f$,
     *         \f$ Z^\top \mathbf{g}^{(\pi)} \f$ for \f$ \boldsymbol\gamma \f$ and
     *         \f$ \sum_i \phi\,\partial\mathrm{NLL}_i/\partial\phi \f$ for \f$ \log\phi \f$ (zero observations do not
     *         contribute to the \f$ \boldsymbol\beta \f$ and \f$ \log\phi \f$ components). Non-finite components are
     *         replaced by 0.
     * @note The chain-rule factor \f$ d\phi/d\log\phi = \phi \f$ uses the clamped \f$ \phi \f$; the gradient is not
     *       zeroed when \f$ \log\phi \f$ lies outside \f$ [-10,10] \f$.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);
        double log_phi = params(kc + kz);
        double phi = std::max(std::exp(std::max(-10.0, std::min(10.0, log_phi))), 1e-4);

        Eigen::VectorXd eta_mean = X * beta;
        Eigen::VectorXd eta_zero = Z * gamma;

        Eigen::VectorXd dNLL_deta_mean = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_zero = Eigen::VectorXd::Zero(n);
        double dNLL_dlogphi = 0.0;

        // per-observation derivatives w.r.t. the two linear predictors and log(phi)
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            BetaMath::evaluate_link(eta_mean(i), mean_link, mu_i, dmu_deta_i);

            double pi_i = 0.0; double dpi_deta_i = 0.0;
            BetaMath::evaluate_link(eta_zero(i), zero_link, pi_i, dpi_deta_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 1e-12) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;
                dNLL_deta_mean(i) = 0.0;
            } else {
                double dL_dpi = -1.0 / pi_i;
                dNLL_deta_zero(i) = dL_dpi * dpi_deta_i;

                double p = mu_i * phi;
                double q = (1.0 - mu_i) * phi;
                double yi = std::max(1e-10, std::min(1.0 - 1e-10, y(i)));
                double digam_sum = BetaMath::digamma(phi);
                double digam_p   = BetaMath::digamma(p);
                double digam_q   = BetaMath::digamma(q);

                double dNLL_dp = digam_p - digam_sum - std::log(yi);
                double dNLL_dq = digam_q - digam_sum - std::log(1.0 - yi);

                double dNLL_dmu    = (dNLL_dp - dNLL_dq) * phi;
                double dNLL_dphi_i = dNLL_dp * mu_i + dNLL_dq * (1.0 - mu_i);

                dNLL_deta_mean(i) = dNLL_dmu * dmu_deta_i;
                dNLL_dlogphi += dNLL_dphi_i * phi;
            }
        }

        Eigen::VectorXd grad(kc + kz + 1);
        grad.head(kc) = X.transpose() * dNLL_deta_mean;
        grad.segment(kc, kz) = Z.transpose() * dNLL_deta_zero;
        grad(kc + kz) = dNLL_dlogphi;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Hybrid analytic/finite-difference Hessian of the negative log-likelihood.
     * @details Structure of the \f$ (kc+kz+1)\times(kc+kz+1) \f$ matrix:
     * - \f$ \boldsymbol\gamma\boldsymbol\gamma \f$ block: exact, \f$ Z^\top\mathrm{diag}(w_i)\,Z \f$ with
     *   \f$ w_i = \partial^2\mathrm{NLL}_i/\partial\pi_i^2\,(d\pi_i/d\eta_i)^2 + \partial\mathrm{NLL}_i/\partial\pi_i\,d^2\pi_i/d\eta_i^2 \f$,
     *   where \f$ \partial^2\mathrm{NLL}_i/\partial\pi_i^2 = 1/(1-\pi_i)^2 \f$ (zero) or \f$ 1/\pi_i^2 \f$ (positive)
     *   and the link curvature comes from `BetaMath::evaluate_link_hessian`;
     * - columns/rows of \f$ \boldsymbol\beta \f$ and the \f$ \log\phi \f$ entry: central differences
     *   \f$ [\nabla\mathrm{NLL}(\theta+\epsilon e_j)-\nabla\mathrm{NLL}(\theta-\epsilon e_j)]/(2\epsilon) \f$ with
     *   \f$ \epsilon = 10^{-5} \f$, using only the \f$ (\boldsymbol\beta,\log\phi) \f$ rows of the gradient;
     *   the \f$ (\boldsymbol\beta,\log\phi) \f$ cross entries are written symmetrically, the
     *   \f$ \boldsymbol\beta\boldsymbol\beta \f$ block itself is not explicitly symmetrised;
     * - all blocks coupling \f$ \boldsymbol\gamma \f$ with \f$ \boldsymbol\beta \f$ or \f$ \log\phi \f$ are set to zero
     *   (exact, since the model is separable).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma, \log\phi) \f$.
     * @return Matrix of size \f$ (kc+kz+1)^2 \f$ with non-finite entries set to 0 and every diagonal entry
     *         \f$ \le 0 \f$ replaced by `1e-4` (not otherwise regularised or PSD-projected).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long kc = X.cols();
        const long kz = Z.cols();

        Eigen::VectorXd beta = params.head(kc);
        Eigen::VectorXd gamma = params.segment(kc, kz);

        // exact gamma-gamma block
        Eigen::VectorXd eta_zero = Z * gamma;
        Eigen::VectorXd H_zero_zero = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double pi_i = 0.0; double dpi_deta_i = 0.0; double d2pi_deta2_i = 0.0;
            BetaMath::evaluate_link_hessian(eta_zero(i), zero_link, pi_i, dpi_deta_i, d2pi_deta2_i);
            pi_i = std::max(1e-12, std::min(1.0 - 1e-12, pi_i));

            if (y(i) < 1e-12) {
                double dL_dpi = 1.0 / (1.0 - pi_i);
                double d2L_dpi2 = 1.0 / ((1.0 - pi_i) * (1.0 - pi_i));
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            } else {
                double dL_dpi = -1.0 / pi_i;
                double d2L_dpi2 = 1.0 / (pi_i * pi_i);
                H_zero_zero(i) = d2L_dpi2 * dpi_deta_i * dpi_deta_i + dL_dpi * d2pi_deta2_i;
            }
        }

        Eigen::MatrixXd H_gamma_gamma = Z.transpose() * H_zero_zero.asDiagonal() * Z;
        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(kc + kz + 1, kc + kz + 1);
        hess.block(kc, kc, kz, kz) = H_gamma_gamma;

        // finite differences of the analytic gradient for the (beta, log phi) rows and columns
        double eps = 1e-5;
        for (long j = 0; j < kc; ++j) {
            Eigen::VectorXd p_plus = params;  p_plus(j) += eps;
            Eigen::VectorXd p_minus = params; p_minus(j) -= eps;

            Eigen::VectorXd g_plus = this->gradient(p_plus);
            Eigen::VectorXd g_minus = this->gradient(p_minus);
            Eigen::VectorXd d2NLL_dtheta_dbetaj = (g_plus - g_minus) / (2.0 * eps);

            hess.block(0, j, kc, 1) = d2NLL_dtheta_dbetaj.head(kc);
            hess(kc + kz, j) = d2NLL_dtheta_dbetaj(kc + kz);
            hess(j, kc + kz) = d2NLL_dtheta_dbetaj(kc + kz);
        }

        Eigen::VectorXd p_phi_plus = params;  p_phi_plus(kc + kz) += eps;
        Eigen::VectorXd p_phi_minus = params; p_phi_minus(kc + kz) -= eps;

        Eigen::VectorXd g_phi_plus = this->gradient(p_phi_plus);
        Eigen::VectorXd g_phi_minus = this->gradient(p_phi_minus);

        hess(kc + kz, kc + kz) = (g_phi_plus(kc + kz) - g_phi_minus(kc + kz)) / (2.0 * eps);

        // beta/log(phi) and gamma are decoupled
        hess.block(0, kc, kc, kz).setZero();
        hess.block(kc, 0, kz, kc).setZero();
        hess.block(kc, kc + kz, kz, 1).setZero();
        hess.block(kc + kz, kc, 1, kz).setZero();

        hess = hess.array().isFinite().select(hess, 0.0);
        for (long i = 0; i < kc + kz + 1; ++i) {
            if (hess(i, i) <= 0.0) hess(i, i) = 1e-4;
        }

        return hess;
    }
};

/**
 * @brief Negative log-likelihood of a Gaussian regression truncated to an interval (left, right or both).
 *
 * @details Each response is a Normal variable \f$ \mathcal{N}(\mu_i,\sigma^2) \f$ observed only inside the interval
 * \f$ [a, b] \f$:
 * \f[ f(y_i) = \frac{\sigma^{-1}\phi\big((y_i-\mu_i)/\sigma\big)}{\Phi(\beta_i)-\Phi(\alpha_i)},
 *     \qquad \alpha_i = \frac{a-\mu_i}{\sigma}, \quad \beta_i = \frac{b-\mu_i}{\sigma}, \f]
 * with \f$ \phi,\Phi \f$ the standard normal density and CDF (`standard_normal_pdf`, `standard_normal_cdf`).
 * Here \f$ \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$ is the mean (location) of the *untruncated* Normal,
 * not the mean of the truncated variable; \f$ \sigma = e^{\log\sigma} \f$ is the scale of the untruncated Normal.
 * The lower limit \f$ a = \f$ `left_limit` is active only if `has_left` is true (otherwise \f$ a=-\infty \f$,
 * \f$ \Phi(\alpha_i)=0 \f$); the upper limit \f$ b = \f$ `right_limit` only if `has_right` (otherwise \f$ b=+\infty \f$,
 * \f$ \Phi(\beta_i)=1 \f$). With both flags false the model reduces to ordinary Gaussian regression.
 *
 * The inverse link \f$ g^{-1} \f$ is computed by the member `evaluate_mean_link_d2`, which supports only the
 * `Identity`, `Log` and `Inverse` links.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                         | Scale |
 * |------------|-------------------------------------------------|-------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor of \f$ \mu \f$ |
 * | `k`        | \f$ \log\sigma \f$                                | log (\f$ \sigma>0 \f$ automatically; **not clamped**) |
 *
 * where \f$ k \f$ = `X.cols()`.
 *
 * ### Log-likelihood
 * With \f$ r_i = (y_i-\mu_i)/\sigma \f$ and \f$ S_i = \max\{\Phi(\beta_i)-\Phi(\alpha_i),\,10^{-15}\} \f$:
 * \f[ \ell_i = -\log\sigma - \tfrac12\log(2\pi) - \tfrac12 r_i^2 - \log S_i, \f]
 * and the functor returns \f$ -\sum_i \ell_i \f$. All constants are kept; there are no weights.
 *
 * ### Derivatives
 * Let \f$ \lambda_i = [\phi(\alpha_i)-\phi(\beta_i)]/S_i \f$, \f$ \lambda^{(1)}_i = [\alpha_i\phi(\alpha_i)-\beta_i\phi(\beta_i)]/S_i \f$
 * and analogously \f$ \lambda^{(2)}_i,\lambda^{(3)}_i \f$ with \f$ \alpha^2,\beta^2 \f$ and \f$ \alpha^3,\beta^3 \f$ as
 * weights (terms of an inactive limit are dropped). Then
 * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial\mu_i} = \frac{-r_i+\lambda_i}{\sigma}, \qquad
 *     \frac{\partial\,\mathrm{NLL}_i}{\partial\log\sigma} = 1 - r_i^2 + \lambda^{(1)}_i, \f]
 * and \f$ \partial\mathrm{NLL}_i/\partial\eta_i = \partial\mathrm{NLL}_i/\partial\mu_i\cdot d\mu_i/d\eta_i \f$.
 * The second derivatives implemented are
 * \f$ \partial^2\mathrm{NLL}_i/\partial\mu_i^2 = \sigma^{-2}(1+\lambda^{(1)}_i-\lambda_i^2) \f$ and
 * \f$ \partial^2\mathrm{NLL}_i/\partial(\log\sigma)^2 = 2r_i^2-\lambda^{(1)}_i+\lambda^{(3)}_i-(\lambda^{(1)}_i)^2 \f$;
 * the Hessian is analytic.
 *
 * ### Numerical safeguards
 * - \f$ S_i \f$ is floored at \f$ 10^{-15} \f$ (also in the derivatives).
 * - `evaluate_mean_link_d2` clamps \f$ \mu \f$ for the `Log` link to \f$ [10^{-7},10^{8}] \f$ and moves
 *   \f$ |\eta|<10^{-7} \f$ to \f$ \pm10^{-7} \f$ for the `Inverse` link.
 * - `operator()` returns `1e10` if the sum is not finite; `gradient` performs no finiteness check.
 * - `hessian` raises every diagonal entry \f$ \le 10^{-5} \f$ to `1e-4`.
 *
 * @note The object stores **non-owning references** to `X` and `y`; they must outlive the functor. No validation
 *       is performed: \f$ y_i \f$ is *not* checked to lie inside \f$ [a,b] \f$ and `left_limit < right_limit` is
 *       assumed.
 * @note For link types other than `Identity`, `Log` and `Inverse`, `evaluate_mean_link_d2` has no matching case and
 *       leaves its outputs untouched; all callers initialise them to 0, hence \f$ \mu_i = d\mu_i/d\eta_i = 0 \f$ and the
 *       model degenerates. Only the three supported links should be used.
 * @see TruncatedPoissonLoss, TruncatedNBLoss, TruncatedGeometricLoss for the truncated count models.
 *
 * @ingroup glm_ll_truncated
 */
struct TruncatedGaussianLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the mean model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed response (length \f$ n \f$), assumed to lie inside the truncation interval.
    LinkType mean_link;  ///< Link whose inverse maps \f$ \eta \f$ to the mean \f$ \mu \f$ of the untruncated Normal (Identity, Log or Inverse).

    double left_limit;  ///< Lower truncation point \f$ a \f$; used only if `has_left` is true.
    double right_limit;  ///< Upper truncation point \f$ b \f$; used only if `has_right` is true.
    bool has_left;  ///< Whether the lower truncation point is active (otherwise \f$ a=-\infty \f$).
    bool has_right;  ///< Whether the upper truncation point is active (otherwise \f$ b=+\infty \f$).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Response vector \f$ y \f$ (length \f$ n \f$), assumed inside the truncation interval.
     * @param link Mean link: `Identity` (default), `Log` or `Inverse` (other values are not supported).
     * @param left Lower truncation point \f$ a \f$ (default 0.0, ignored unless `use_left`).
     * @param right Upper truncation point \f$ b \f$ (default 0.0, ignored unless `use_right`).
     * @param use_left Activates the lower truncation (default false).
     * @param use_right Activates the upper truncation (default false).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor; if both limits are active,
     *      \f$ a<b \f$.
     */
    TruncatedGaussianLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog,
                          LinkType link = LinkType::Identity,
                          double left = 0.0, double right = 0.0,
                          bool use_left = false, bool use_right = false)
        : X(exog), y(endog), mean_link(link), left_limit(left), right_limit(right),
          has_left(use_left), has_right(use_right) {}

    /**
     * @brief Evaluates the inverse link and its first two derivatives for the mean.
     * @details Implemented cases (all other `LinkType` values leave the outputs unchanged):
     * - `Identity`: \f$ \mu=\eta,\ \mu'=1,\ \mu''=0 \f$;
     * - `Log`: \f$ \mu=\mathrm{clamp}(e^{\eta},10^{-7},10^{8}) \f$ and \f$ \mu'=\mu''=\mu \f$ (the clamped value, so
     *   the derivatives are not zeroed when the clamp is active);
     * - `Inverse`: \f$ \mu=1/\eta,\ \mu'=-1/\eta^2,\ \mu''=2/\eta^3 \f$ after moving \f$ |\eta|<10^{-7} \f$ to
     *   \f$ \pm10^{-7} \f$ (the sign of 0 is taken as positive).
     * @param eta Linear predictor \f$ \eta \f$.
     * @param link Link type.
     * @param mu Output: mean \f$ \mu = g^{-1}(\eta) \f$.
     * @param dmu_deta Output: first derivative \f$ d\mu/d\eta \f$.
     * @param d2mu_deta2 Output: second derivative \f$ d^2\mu/d\eta^2 \f$.
     */
    void evaluate_mean_link_d2(double eta, LinkType link, double& mu, double& dmu_deta, double& d2mu_deta2) const {
        switch (link) {
            case LinkType::Identity:
                mu = eta;
                dmu_deta = 1.0;
                d2mu_deta2 = 0.0;
                break;
            case LinkType::Log:
                mu = std::exp(eta);

                mu = std::max(1e-7, std::min(mu, 1e8));
                dmu_deta = mu;
                d2mu_deta2 = mu;
                break;
            case LinkType::Inverse:

                if (std::abs(eta) < 1e-7) eta = (eta >= 0) ? 1e-7 : -1e-7;
                mu = 1.0 / eta;
                dmu_deta = -1.0 / (eta * eta);
                d2mu_deta2 = 2.0 / (eta * eta * eta);
                break;
            default:
                break;
        }
    }

    /**
     * @brief Evaluates the total negative log-likelihood of the truncated Gaussian model.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$ with \f$ \ell_i = \log f(y_i) - \log S_i \f$ (normal log-density minus the
     *         log of the floored truncation mass), or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        // linear predictors
        Eigen::VectorXd eta = X * beta;
        double total_nll = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, d1 = 0.0, d2 = 0.0;
            evaluate_mean_link_d2(eta(i), mean_link, mu_i, d1, d2);

            // untruncated log-density and truncation mass of the observation
            double res = (y(i) - mu_i) / sigma;
            double log_f = -log_sigma - 0.5 * std::log(2.0 * M_PI) - 0.5 * res * res;

            double alpha = has_left  ? (left_limit - mu_i) / sigma  : -std::numeric_limits<double>::infinity();
            double delta = has_right ? (right_limit - mu_i) / sigma : std::numeric_limits<double>::infinity();

            double Phi_alpha = has_left  ? standard_normal_cdf(alpha) : 0.0;
            double Phi_delta = has_right ? standard_normal_cdf(delta) : 1.0;
            double S_i = std::max(Phi_delta - Phi_alpha, 1e-15);

            total_nll -= (log_f - std::log(S_i));
        }

        return std::isfinite(total_nll) ? total_nll : 1e10;
    }

    /**
     * @brief Analytic gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Vector of length \f$ k+1 \f$: \f$ X^\top\mathbf{g}_\eta \f$ with
     *         \f$ g_{\eta,i} = (-r_i+\lambda_i)\sigma^{-1}\,d\mu_i/d\eta_i \f$, followed by
     *         \f$ \sum_i(1-r_i^2+\lambda^{(1)}_i) \f$ (see the class description for the notation). No finiteness
     *         sanitising is applied.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd dNLL_deta(n);
        double dNLL_dlogsigma = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu_deta_i = 0.0, d2mu_deta2_i = 0.0;
            evaluate_mean_link_d2(eta(i), mean_link, mu_i, dmu_deta_i, d2mu_deta2_i);

            double res = (y(i) - mu_i) / sigma;

            double alpha = has_left  ? (left_limit - mu_i) / sigma  : -std::numeric_limits<double>::infinity();
            double delta = has_right ? (right_limit - mu_i) / sigma : std::numeric_limits<double>::infinity();

            double phi_alpha = has_left  ? standard_normal_pdf(alpha) : 0.0;
            double phi_delta = has_right ? standard_normal_pdf(delta) : 0.0;
            double Phi_alpha = has_left  ? standard_normal_cdf(alpha) : 0.0;
            double Phi_delta = has_right ? standard_normal_cdf(delta) : 1.0;
            double S_i = std::max(Phi_delta - Phi_alpha, 1e-15);

            // truncation correction terms (inverse-Mills-like ratios)
            double lambda_i = (phi_alpha - phi_delta) / S_i;
            double lambda_sigma_i = ((has_left ? alpha * phi_alpha : 0.0) - (has_right ? delta * phi_delta : 0.0)) / S_i;

            double dNLL_dmu = -(res / sigma) + (lambda_i / sigma);

            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;

            dNLL_dlogsigma += 1.0 - (res * res) + lambda_sigma_i;
        }

        Eigen::VectorXd grad(k + 1);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k) = dNLL_dlogsigma;

        return grad;
    }

    /**
     * @brief Analytic Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @details Three blocks are filled:
     * - \f$ H_{\beta\beta} = X^\top\mathrm{diag}(W_i)\,X \f$ with
     *   \f$ W_i = \partial^2\mathrm{NLL}_i/\partial\mu_i^2\,(d\mu_i/d\eta_i)^2 + \partial\mathrm{NLL}_i/\partial\mu_i\,d^2\mu_i/d\eta_i^2 \f$;
     * - \f$ H_{\beta,\log\sigma} = X^\top \mathbf{w} \f$ with
     *   \f$ w_i = \sigma^{-1}(2r_i-\lambda_i+\lambda^{(2)}_i-\lambda_i\lambda^{(1)}_i)\,d\mu_i/d\eta_i \f$ (`W_eta_sigma`);
     * - \f$ H_{\log\sigma,\log\sigma} = \sum_i [2r_i^2-\lambda^{(1)}_i+\lambda^{(3)}_i-(\lambda^{(1)}_i)^2] \f$ (set to 0 if not finite).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Matrix of size \f$ (k+1)\times(k+1) \f$ in which every diagonal entry \f$ \le 10^{-5} \f$ is replaced by
     *         `1e-4`. The \f$ \boldsymbol\beta \f$ block is not checked for finiteness.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = params(k);
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;

        // per-observation Hessian weights
        Eigen::VectorXd W_eta2 = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_eta_sigma = Eigen::VectorXd::Zero(n);
        double d2NLL_dlogsigma2 = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu_deta_i = 0.0, d2mu_deta2_i = 0.0;
            evaluate_mean_link_d2(eta(i), mean_link, mu_i, dmu_deta_i, d2mu_deta2_i);

            double res = (y(i) - mu_i) / sigma;

            double alpha = has_left  ? (left_limit - mu_i) / sigma  : -std::numeric_limits<double>::infinity();
            double delta = has_right ? (right_limit - mu_i) / sigma : std::numeric_limits<double>::infinity();

            double phi_alpha = has_left  ? standard_normal_pdf(alpha) : 0.0;
            double phi_delta = has_right ? standard_normal_pdf(delta) : 0.0;
            double Phi_alpha = has_left  ? standard_normal_cdf(alpha) : 0.0;
            double Phi_delta = has_right ? standard_normal_cdf(delta) : 1.0;
            double S_i = std::max(Phi_delta - Phi_alpha, 1e-15);

            double lambda_i = (phi_alpha - phi_delta) / S_i;
            double lambda_sigma_i = ((has_left ? alpha * phi_alpha : 0.0) - (has_right ? delta * phi_delta : 0.0)) / S_i;
            double lambda_sigma2_i = ((has_left ? alpha * alpha * phi_alpha : 0.0) - (has_right ? delta * delta * phi_delta : 0.0)) / S_i;
            double lambda_sigma3_i = ((has_left ? std::pow(alpha, 3) * phi_alpha : 0.0) - (has_right ? std::pow(delta, 3) * phi_delta : 0.0)) / S_i;

            double dNLL_dmu = -(res / sigma) + (lambda_i / sigma);

            double d2NLL_dmu2 = (1.0 / (sigma * sigma)) * (1.0 + lambda_sigma_i - (lambda_i * lambda_i));

            double val_eta2 = d2NLL_dmu2 * (dmu_deta_i * dmu_deta_i) + dNLL_dmu * d2mu_deta2_i;
            W_eta2(i) = val_eta2;

            double d2NLL_dmu_dlogsigma = (1.0 / sigma) * (2.0 * res - lambda_i + lambda_sigma2_i - (lambda_i * lambda_sigma_i));
            W_eta_sigma(i) = d2NLL_dmu_dlogsigma * dmu_deta_i;

            d2NLL_dlogsigma2 += (2.0 * res * res) - lambda_sigma_i + lambda_sigma3_i - (lambda_sigma_i * lambda_sigma_i);
        }

        // assemble the beta-beta, beta-log(sigma) and log(sigma)-log(sigma) blocks
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(k + 1, k + 1);

        H.block(0, 0, k, k) = X.transpose() * W_eta2.asDiagonal() * X;

        Eigen::VectorXd H_beta_logsigma = X.transpose() * W_eta_sigma;
        H.block(0, k, k, 1) = H_beta_logsigma;
        H.block(k, 0, 1, k) = H_beta_logsigma.transpose();

        H(k, k) = std::isfinite(d2NLL_dlogsigma2) ? d2NLL_dlogsigma2 : 0.0;

        for (long j = 0; j <= k; ++j) {
            if (H(j, j) <= 1e-5) H(j, j) = 1e-4;
        }

        return H;
    }
};

/**
 * @brief Negative log-likelihood of a Poisson regression truncated to the count range \f$ \{L,\dots,U\} \f$.
 *
 * @details Counts \f$ y_i \f$ follow a Poisson law with mean \f$ \lambda_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$
 * conditioned on \f$ L \le y_i \le U \f$, with \f$ L = \f$ `left_limit` and \f$ U = \f$ `right_limit`:
 * \f[ \Pr(y_i=y) = \frac{e^{-\lambda_i}\lambda_i^{y}/y!}{F(U;\lambda_i)-F(L-1;\lambda_i)}, \qquad L \le y \le U, \f]
 * where \f$ F(\cdot;\lambda) \f$ is the Poisson CDF. Conventions: the lower truncation is active only if
 * \f$ L>0 \f$ (so the default \f$ L=0 \f$ means no lower truncation; \f$ L=1 \f$ gives the zero-truncated Poisson) and the
 * upper truncation is active only if \f$ U\ge0 \f$ (default \f$ U=-1 \f$ means no upper truncation). The inverse link
 * \f$ g^{-1} \f$ is `link` (default Log), evaluated with the generic `evaluate_link` (any `LinkType`, the `Log`
 * link uses \f$ \exp(\min(\eta,23)) \f$).
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                         | Scale |
 * |------------|-------------------------------------------------|-------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor of \f$ \lambda \f$ |
 *
 * where \f$ k \f$ = `X.cols()`. There are no extra parameters.
 *
 * ### Log-likelihood
 * With \f$ \lambda_i \f$ clamped to \f$ [10^{-5},10^{6}] \f$ and the truncation mass
 * \f$ S_i = \max\{F(U;\lambda_i)-F(L-1;\lambda_i),\,10^{-12}\} \f$ (with \f$ F(U)=1 \f$ if there is no upper limit and
 * \f$ F(L-1)=0 \f$ if there is no lower limit):
 * \f[ \ell_i = -\lambda_i + y_i\log\lambda_i - \log\Gamma(y_i+1) - \log S_i, \f]
 * and the functor returns \f$ -\sum_i \ell_i \f$ (constants kept, no weights; \f$ y_i \f$ enters `lgamma` as a double).
 *
 * ### Derivatives
 * Let \f$ P(k) \f$ be the Poisson pmf. Since \f$ \partial F(k;\lambda)/\partial\lambda = -P(k) \f$,
 * \f[ \frac{\partial S_i}{\partial\lambda_i} = P(L-1)-P(U), \qquad
 *     \frac{\partial^2 S_i}{\partial\lambda_i^2} = P(L-2)-P(L-1)-P(U-1)+P(U), \f]
 * (terms of an inactive limit, or with a negative argument, are 0), hence
 * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial\lambda_i} = 1-\frac{y_i}{\lambda_i}+\frac{S_i'}{S_i}, \qquad
 *     \frac{\partial^2\mathrm{NLL}_i}{\partial\lambda_i^2} = \frac{y_i}{\lambda_i^2}+\frac{S_i''S_i-(S_i')^2}{S_i^2}. \f]
 * The **gradient is not analytic**: it differentiates the per-observation NLL with respect to \f$ \eta_i \f$ by central
 * finite differences (step \f$ 10^{-6} \f$). The **Hessian is analytic** and uses the formulas above.
 *
 * ### Numerical safeguards
 * - \f$ \lambda_i \f$ clamped to \f$ [10^{-5},10^{6}] \f$; \f$ S_i \f$ floored at \f$ 10^{-12} \f$.
 * - `poisson_cdf_scalar` stops summing once the partial sum exceeds \f$ 0.9999999999 \f$ and returns exactly 1.
 * - `operator()` returns `1e10` if the sum is not finite; `gradient` performs no finiteness check.
 * - `hessian` replaces non-finite entries by 0 and every non-positive diagonal entry by `1e-4`.
 *
 * @note The object stores **non-owning references** to `X` and `y`; they must outlive the functor. No validation
 *       is performed: \f$ y_i \f$ should be a non-negative integer inside \f$ [L,U] \f$. The CDF is computed by direct
 *       summation, so the cost per observation grows linearly with the truncation points.
 * @note Because the gradient is evaluated through the clamped \f$ \lambda_i \f$, it vanishes where a clamp is active,
 *       whereas the Hessian chain rule uses the unclamped link derivatives \f$ d\lambda/d\eta,\,d^2\lambda/d\eta^2 \f$ with
 *       the clamped \f$ \lambda_i \f$ (the clamp is not differentiated). The Hessian uses `evaluate_count_link_d2`
 *       (which additionally floors \f$ \lambda \f$ at \f$ 10^{-15} \f$) while the value uses `evaluate_link`.
 * @see TruncatedNBLoss, TruncatedGeometricLoss, HurdlePoissonLoss
 *
 * @ingroup glm_ll_truncated
 */
struct TruncatedPoissonLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$), stored as doubles; assumed integers in \f$ [L,U] \f$.
    LinkType count_link;  ///< Link whose inverse maps \f$ \eta \f$ to the Poisson mean \f$ \lambda \f$.

    long left_limit;  ///< Lower truncation point \f$ L \f$: counts below \f$ L \f$ are excluded; active only if \f$ L>0 \f$.
    long right_limit;  ///< Upper truncation point \f$ U \f$: counts above \f$ U \f$ are excluded; active only if \f$ U \ge 0 \f$ (negative = no upper limit).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$), integers in \f$ [L,U] \f$.
     * @param left Lower truncation point \f$ L \f$ (default 0 = no lower truncation; 1 = zero-truncated).
     * @param right Upper truncation point \f$ U \f$ (default -1 = no upper truncation; any negative value disables it).
     * @param link Link of the Poisson mean (default `LinkType::Log`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor; if both limits are active,
     *      \f$ L\le U \f$.
     */
    TruncatedPoissonLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog,
                         long left = 0, long right = -1, LinkType link = LinkType::Log)
        : X(exog), y(endog), count_link(link), left_limit(left), right_limit(right) {}

    /**
     * @brief Poisson probability mass function.
     * @details Computes \f$ P(k;\lambda)=\exp\{-\lambda+k\log\lambda-\log\Gamma(k+1)\} \f$.
     * @param k Count (a negative count has probability 0).
     * @param lambda Poisson mean \f$ \lambda>0 \f$ (not validated; \f$ \log\lambda \f$ is taken directly).
     * @return Probability \f$ P(k;\lambda) \f$, or 0 if \f$ k<0 \f$.
     */
    static inline double poisson_pmf_scalar(long k, double lambda) {
        if (k < 0) return 0.0;
        return std::exp(-lambda + static_cast<double>(k) * std::log(lambda) - std::lgamma(static_cast<double>(k) + 1.0));
    }

    /**
     * @brief Poisson cumulative distribution function by direct summation.
     * @details Computes \f$ F(k;\lambda)=\sum_{i=0}^{k}P(i;\lambda) \f$ and stops early, returning exactly 1, as soon
     *          as the partial sum exceeds \f$ 0.9999999999 \f$; the result is capped at 1.
     * @param k Upper count (a negative value gives 0).
     * @param lambda Poisson mean \f$ \lambda>0 \f$.
     * @return Probability \f$ F(k;\lambda) \f$, or 0 if \f$ k<0 \f$.
     */
    static inline double poisson_cdf_scalar(long k, double lambda) {
        if (k < 0) return 0.0;
        double sum = 0.0;
        for (long i = 0; i <= k; ++i) {
            sum += poisson_pmf_scalar(i, lambda);
            if (sum > 0.9999999999) return 1.0;
        }
        return std::min(sum, 1.0);
    }

    /**
     * @brief Per-observation negative log-likelihood given the linear predictor.
     * @details Evaluates \f$ \lambda=g^{-1}(\eta) \f$ with `evaluate_link`, clamps it to \f$ [10^{-5},10^{6}] \f$ and returns
     * \f[ \lambda - y\log\lambda + \log\Gamma(y+1) + \log\max\{F(U;\lambda)-F(L-1;\lambda),\,10^{-12}\}. \f]
     * This helper is what `operator()` sums and what `gradient` differentiates numerically.
     * @param eta_i Linear predictor \f$ \eta_i=\mathbf{x}_i^\top\boldsymbol\beta \f$.
     * @param y_i Observed count \f$ y_i \f$.
     * @return The contribution \f$ -\ell_i \f$ of one observation.
     */
    double evaluate_single_nll(double eta_i, double y_i) const {
        double lambda_i = 0.0;
        evaluate_link(eta_i, count_link, lambda_i);
        lambda_i = std::max(1e-5, std::min(lambda_i, 1e6));

        double log_poisson = -lambda_i + y_i * std::log(lambda_i) - std::lgamma(y_i + 1.0);

        double cdf_u = (right_limit >= 0) ? poisson_cdf_scalar(right_limit, lambda_i) : 1.0;
        double cdf_l = (left_limit > 0) ? poisson_cdf_scalar(left_limit - 1, lambda_i) : 0.0;

        double prob_mass = std::max(cdf_u - cdf_l, 1e-12);
        return -(log_poisson - std::log(prob_mass));
    }

    /**
     * @brief Evaluates the total negative log-likelihood of the truncated Poisson model.
     * @param params Coefficient vector \f$ \boldsymbol\beta \f$ of length \f$ k \f$.
     * @return Sum \f$ \sum_i \mathrm{evaluate\_single\_nll}(\eta_i,y_i) \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        double total_neg_ll = 0.0;
        Eigen::VectorXd eta = X * params;

        for (long i = 0; i < n; ++i) {
            total_neg_ll += evaluate_single_nll(eta(i), y(i));
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Gradient of the negative log-likelihood by central finite differences in \f$ \eta_i \f$.
     * @details For each observation \f$ \partial\mathrm{NLL}_i/\partial\eta_i \approx
     *          [\mathrm{NLL}_i(\eta_i+\epsilon)-\mathrm{NLL}_i(\eta_i-\epsilon)]/(2\epsilon) \f$ with \f$ \epsilon=10^{-6} \f$,
     *          then \f$ \nabla = X^\top\,\partial\mathrm{NLL}/\partial\boldsymbol\eta \f$ (the \f$ \epsilon \f$-perturbation acts
     *          on \f$ \eta \f$, not on \f$ \boldsymbol\beta \f$).
     * @param params Coefficient vector \f$ \boldsymbol\beta \f$.
     * @return Vector of length \f$ k \f$; not sanitised for non-finite values.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        Eigen::VectorXd eta = X * params;
        Eigen::VectorXd dNLL_deta(n);

        const double eps = 1e-6;

        for (long i = 0; i < n; ++i) {
            double eta_plus  = eta(i) + eps;
            double eta_minus = eta(i) - eps;

            double nll_plus  = evaluate_single_nll(eta_plus, y(i));
            double nll_minus = evaluate_single_nll(eta_minus, y(i));

            dNLL_deta(i) = (nll_plus - nll_minus) / (2.0 * eps);
        }

        return X.transpose() * dNLL_deta;
    }

    /**
     * @brief Analytic Hessian \f$ X^\top\mathrm{diag}(W_i)\,X \f$ of the negative log-likelihood.
     * @details With the derivatives of the class description,
     * \f[ W_i = \frac{\partial^2\mathrm{NLL}_i}{\partial\lambda_i^2}\Big(\frac{d\lambda_i}{d\eta_i}\Big)^2
     *         + \frac{\partial\mathrm{NLL}_i}{\partial\lambda_i}\,\frac{d^2\lambda_i}{d\eta_i^2}. \f]
     * The needed pmf values \f$ P(U),P(U-1),P(L-1),P(L-2) \f$ are included only when the corresponding limit and
     * argument are valid (\f$ U\ge0 \f$, \f$ U>0 \f$, \f$ L>0 \f$, \f$ L>1 \f$ respectively).
     * @param params Coefficient vector \f$ \boldsymbol\beta \f$.
     * @return Symmetric \f$ k\times k \f$ matrix with non-finite entries set to 0 and non-positive diagonal entries
     *         replaced by `1e-4` (no other regularisation or PSD projection).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        Eigen::VectorXd eta = X * params;
        // per-observation weights from the truncated-Poisson second derivatives
        Eigen::VectorXd W = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double lambda_i = 0.0, dlam_deta_i = 0.0, d2lam_deta2_i = 0.0;
            evaluate_count_link_d2(eta(i), count_link, lambda_i, dlam_deta_i, d2lam_deta2_i);
            lambda_i = std::max(1e-5, std::min(lambda_i, 1e6));

            double P_u = (right_limit >= 0) ? poisson_pmf_scalar(right_limit, lambda_i) : 0.0;
            double P_u_minus1 = (right_limit > 0) ? poisson_pmf_scalar(right_limit - 1, lambda_i) : 0.0;

            double P_l_minus1 = (left_limit > 0) ? poisson_pmf_scalar(left_limit - 1, lambda_i) : 0.0;
            double P_l_minus2 = (left_limit > 1) ? poisson_pmf_scalar(left_limit - 2, lambda_i) : 0.0;

            double cdf_u = (right_limit >= 0) ? poisson_cdf_scalar(right_limit, lambda_i) : 1.0;
            double cdf_l = (left_limit > 0) ? poisson_cdf_scalar(left_limit - 1, lambda_i) : 0.0;
            double S_i = std::max(cdf_u - cdf_l, 1e-12);

            // derivatives of the truncation mass S with respect to lambda
            double dS_dlambda = P_l_minus1 - P_u;
            double dNLL_dlambda = 1.0 - (y(i) / lambda_i) + (dS_dlambda / S_i);

            double d2S_dlambda2 = P_l_minus2 - P_l_minus1 - P_u_minus1 + P_u;
            double d2NLL_dlambda2 = (y(i) / (lambda_i * lambda_i))
                                  + (d2S_dlambda2 * S_i - dS_dlambda * dS_dlambda) / (S_i * S_i);

            W(i) = d2NLL_dlambda2 * (dlam_deta_i * dlam_deta_i) + dNLL_dlambda * d2lam_deta2_i;
        }

        Eigen::MatrixXd H = X.transpose() * W.asDiagonal() * X;

        H = H.array().isFinite().select(H, 0.0);
        for (long j = 0; j < k; ++j) {
            if (H(j, j) <= 0.0) H(j, j) = 1e-4;
        }

        return H;
    }
};

/**
 * @brief Negative log-likelihood of a Negative Binomial (NB2) regression truncated to the count range \f$ \{L,\dots,U\} \f$.
 *
 * @details Counts \f$ y_i \f$ follow an NB2 law with mean \f$ \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$,
 * variance \f$ \mu_i+\alpha\mu_i^2 \f$ and dispersion \f$ \alpha = e^{\log\alpha} > 0 \f$ (\f$ r=1/\alpha \f$ is the size parameter),
 * conditioned on \f$ L\le y_i\le U \f$ (\f$ L = \f$ `left_limit`, \f$ U = \f$ `right_limit`):
 * \f[ P(y;\mu,\alpha) = \frac{\Gamma(y+r)}{y!\,\Gamma(r)}\Big(\frac{\alpha\mu}{1+\alpha\mu}\Big)^{y}(1+\alpha\mu)^{-r}, \qquad
 *     \Pr(y_i=y)=\frac{P(y;\mu_i,\alpha)}{F(U)-F(L-1)}, \f]
 * with \f$ F \f$ the NB CDF. The lower truncation is active only if \f$ L>0 \f$ and the upper truncation only if
 * \f$ U\ge0 \f$ (default \f$ U=-1 \f$: none). The inverse link is evaluated with `evaluate_count_link_d2`
 * (any `LinkType`; \f$ \mu \f$ floored at \f$ 10^{-15} \f$, no upper clamp).
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                         | Scale |
 * |------------|-------------------------------------------------|-------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor of \f$ \mu \f$ |
 * | `k`        | \f$ \log\alpha \f$                                | log; clamped to \f$ [-10,10] \f$ before use |
 *
 * where \f$ k \f$ = `X.cols()`.
 *
 * ### Log-likelihood
 * With \f$ S_i = \max\{F(U;\mu_i,\alpha)-F(L-1;\mu_i,\alpha),\,10^{-12}\} \f$ (\f$ F(U)=1 \f$ without upper limit,
 * \f$ F(L-1)=0 \f$ without lower limit):
 * \f[ \ell_i = \log\Gamma(y_i+r)-\log\Gamma(y_i+1)-\log\Gamma(r)+y_i\log(\alpha\mu_i)-(y_i+r)\log(1+\alpha\mu_i)-\log S_i, \f]
 * and the functor returns \f$ -\sum_i \ell_i \f$ (constants kept, no weights).
 *
 * ### Derivatives
 * Per observation (all derivatives of \f$ \log P \f$ are with respect to \f$ \mu \f$ and \f$ \log\alpha \f$):
 * \f[ \frac{\partial\log P(y)}{\partial\mu}=\frac{y-\mu}{\mu(1+\alpha\mu)}, \qquad
 *     \frac{\partial\log P(y)}{\partial\log\alpha}=-r\Big[\psi(y+r)-\psi(r)+\log\frac{r}{r+\mu}+\frac{\mu-y}{r+\mu}\Big], \f]
 * with \f$ \psi \f$ the digamma function (`glm::digamma`). The truncation-mass derivatives are the
 * \f$ P \f$-weighted sums of these scores over the retained support,
 * \f$ \partial S/\partial\theta=\sum_{y=L}^{U}P(y)\,\partial\log P(y)/\partial\theta \f$, and
 * \f$ \partial\mathrm{NLL}_i/\partial\theta=-\partial\log P(y_i)/\partial\theta+(\partial S/\partial\theta)/S_i \f$.
 * The gradient is analytic. The \f$ \boldsymbol\beta\boldsymbol\beta \f$ block of the Hessian is analytic (via the
 * second \f$ \mu \f$-derivative of \f$ S \f$), whereas the row/column of \f$ \log\alpha \f$ comes from central
 * finite differences of the gradient (see `hessian`).
 *
 * ### Numerical safeguards
 * - \f$ \log\alpha \f$ clamped to \f$ [-10,10] \f$ (derivatives are *not* zeroed outside that range);
 *   \f$ S_i \f$ floored at \f$ 10^{-12} \f$.
 * - The gradient sums the support from \f$ \max(L,0) \f$ to \f$ U \f$ (or, without upper limit, to
 *   \f$ \lfloor\mu+15\sqrt{\mu+\alpha\mu^2}\rfloor \f$), capped at index 1000, and breaks early when
 *   \f$ P<10^{-15} \f$ beyond the mean.
 * - `operator()` returns `1e10` if the sum is not finite; `gradient` replaces non-finite components by 0;
 *   `hessian` replaces non-finite entries by 0 and non-positive diagonal entries by `1e-4`.
 *
 * @note The object stores **non-owning references** to `X` and `y`; they must outlive the functor. No validation
 *       is performed: \f$ y_i \f$ should be a non-negative integer inside \f$ [L,U] \f$. The CDF is evaluated by direct
 *       summation over \f$ 0,\dots,k \f$ (cost linear in the limit, no early exit).
 * @note Unlike `TruncatedPoissonLoss`/`TruncatedGeometricLoss`, `left` has no default value in the constructor,
 *       and \f$ \mu \f$ is not clamped from above. The helper `nb_pmf_scalar` floors \f$ \mu \f$ at \f$ 10^{-10} \f$, whereas
 *       the log-density in `operator()` uses the (\f$ 10^{-15} \f$-floored) \f$ \mu \f$ directly.
 * @see TruncatedPoissonLoss, TruncatedGeometricLoss, HurdleNBLoss, ZeroInflatedNBLoss
 *
 * @ingroup glm_ll_truncated
 */
struct TruncatedNBLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$), stored as doubles; assumed integers in \f$ [L,U] \f$.
    LinkType count_link;  ///< Link whose inverse maps \f$ \eta \f$ to the NB mean \f$ \mu \f$.
    long left_limit;  ///< Lower truncation point \f$ L \f$: counts below \f$ L \f$ are excluded; active only if \f$ L>0 \f$.
    long right_limit;  ///< Upper truncation point \f$ U \f$: counts above \f$ U \f$ are excluded; active only if \f$ U \ge 0 \f$ (negative = no upper limit).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$), integers in \f$ [L,U] \f$.
     * @param left Lower truncation point \f$ L \f$ (no default; 0 = no lower truncation, 1 = zero-truncated).
     * @param right Upper truncation point \f$ U \f$ (default -1 = no upper truncation; any negative value disables it).
     * @param link Link of the NB mean (default `LinkType::Log`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor; if both limits are active,
     *      \f$ L\le U \f$.
     */
    TruncatedNBLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog,
                    long left, long right = -1, LinkType link = LinkType::Log)
        : X(exog), y(endog), count_link(link), left_limit(left), right_limit(right) {}

    /**
     * @brief Negative Binomial (NB2) probability mass function.
     * @details Computes \f$ P(k;\mu,\alpha)=\exp\{\log\Gamma(k+r)-\log\Gamma(k+1)-\log\Gamma(r)+k\log(\alpha\mu)-(k+r)\log(1+\alpha\mu)\} \f$
     *          with \f$ r=1/\alpha \f$ (the log-denominator uses `log1p`).
     * @param k Count (a negative count has probability 0).
     * @param mu Mean \f$ \mu \f$; values below \f$ 10^{-10} \f$ are raised to \f$ 10^{-10} \f$.
     * @param alpha Dispersion \f$ \alpha>0 \f$ (not validated).
     * @return Probability \f$ P(k;\mu,\alpha) \f$, or 0 if \f$ k<0 \f$.
     */
    inline double nb_pmf_scalar(long k, double mu, double alpha) const {
        if (k < 0) return 0.0;
        double inv_alpha = 1.0 / alpha;

        if (mu < 1e-10) mu = 1e-10;

        double log_pmf = std::lgamma(static_cast<double>(k) + inv_alpha)
                    - std::lgamma(static_cast<double>(k) + 1.0)
                    - std::lgamma(inv_alpha)
                    + static_cast<double>(k) * std::log(alpha * mu)
                    - (static_cast<double>(k) + inv_alpha) * std::log1p(alpha * mu);
        return std::exp(log_pmf);
    }

    /**
     * @brief Negative Binomial cumulative distribution function by direct summation.
     * @details Computes \f$ F(k;\mu,\alpha)=\sum_{i=0}^{k}P(i;\mu,\alpha) \f$, capped at 1 (no early exit).
     * @param k Upper count (a negative value gives 0).
     * @param mu Mean \f$ \mu \f$.
     * @param alpha Dispersion \f$ \alpha>0 \f$.
     * @return Probability \f$ F(k;\mu,\alpha) \f$, or 0 if \f$ k<0 \f$.
     */
    inline double nb_cdf_scalar(long k, double mu, double alpha) const {
        if (k < 0) return 0.0;

        double sum = 0.0;
        for (long i = 0; i <= k; ++i) {
            sum += nb_pmf_scalar(i, mu, alpha);
        }

        return (sum > 1.0) ? 1.0 : sum;
    }

    /**
     * @brief Evaluates the total negative log-likelihood of the truncated NB model.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\alpha) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ -\sum_i \ell_i \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_alpha = std::max(-10.0, std::min(10.0, params(k)));
        double alpha = std::exp(log_alpha);
        double inv_alpha = 1.0 / alpha;

        // linear predictor and untruncated NB log-pmf plus truncation mass
        Eigen::VectorXd eta = X * beta;
        double total_neg_ll = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu = 0.0, d2mu = 0.0;
            evaluate_count_link_d2(eta(i), count_link, mu_i, dmu, d2mu);

            double log_nb = std::lgamma(y(i) + inv_alpha) - std::lgamma(y(i) + 1.0) - std::lgamma(inv_alpha) +
                            y(i) * std::log(alpha * mu_i) - (y(i) + inv_alpha) * std::log1p(alpha * mu_i);

            double cdf_u = (right_limit >= 0) ? nb_cdf_scalar(right_limit, mu_i, alpha) : 1.0;
            double cdf_l = (left_limit > 0) ? nb_cdf_scalar(left_limit - 1, mu_i, alpha) : 0.0;

            double prob_mass = std::max(cdf_u - cdf_l, 1e-12);
            total_neg_ll -= (log_nb - std::log(prob_mass));
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytic gradient of the negative log-likelihood.
     * @details The truncation-mass derivatives are accumulated by summing \f$ P(y)\,\partial\log P(y)/\partial\theta \f$
     *          over \f$ y\in[\max(L,0),\,k_{\max}] \f$, where \f$ k_{\max}=U \f$ if \f$ U\ge0 \f$ and
     *          \f$ \lfloor\mu_i+15\sqrt{\mu_i+\alpha\mu_i^2}\rfloor \f$ otherwise, in both cases capped at 1000; the sum
     *          stops early once \f$ P<10^{-15} \f$ and \f$ y>\mu_i \f$. In particular an upper limit above 1000 is truncated to
     *          1000 in this routine.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\alpha) \f$.
     * @return Vector of length \f$ k+1 \f$: \f$ X^\top\partial\mathrm{NLL}/\partial\boldsymbol\eta \f$ followed by
     *         \f$ \sum_i\partial\mathrm{NLL}_i/\partial\log\alpha \f$; non-finite components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_alpha = std::max(-10.0, std::min(10.0, params(k)));
        double alpha = std::exp(log_alpha);
        double r = 1.0 / alpha;

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlog_alpha = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu_deta_i = 0.0, d2mu = 0.0;
            evaluate_count_link_d2(eta(i), count_link, mu_i, dmu_deta_i, d2mu);

            double denom = 1.0 + alpha * mu_i;
            double cdf_u = (right_limit >= 0) ? nb_cdf_scalar(right_limit, mu_i, alpha) : 1.0;
            double cdf_l = (left_limit > 0) ? nb_cdf_scalar(left_limit - 1, mu_i, alpha) : 0.0;
            double S_i = std::max(cdf_u - cdf_l, 1e-12);

            // scores of the NB log-pmf at the observed count
            double dlog_pmf_dmu = (y(i) - mu_i) / (mu_i * denom);
            double dg_yi = digamma(y(i) + r) - digamma(r) + std::log(r / (r + mu_i)) + (mu_i - y(i)) / (r + mu_i);
            double dlog_pmf_dtheta = -r * dg_yi;

            double dS_dmu = 0.0;
            double dS_dtheta = 0.0;

            // derivatives of the truncation mass over the retained support
            long k_start = (left_limit > 0) ? left_limit : 0;
            long k_end = (right_limit >= 0) ? right_limit : static_cast<long>(mu_i + 15.0 * std::sqrt(mu_i + alpha * mu_i * mu_i));
            if (k_end > 1000) k_end = 1000;

            for (long k_val = k_start; k_val <= k_end; ++k_val) {
                double P_k = nb_pmf_scalar(k_val, mu_i, alpha);
                if (P_k < 1e-15 && static_cast<double>(k_val) > mu_i) break;

                double dL_dmu = (static_cast<double>(k_val) - mu_i) / (mu_i * denom);
                double dg_k = digamma(static_cast<double>(k_val) + r) - digamma(r) + std::log(r / (r + mu_i)) + (mu_i - static_cast<double>(k_val)) / (r + mu_i);
                double dL_dtheta = -r * dg_k;

                dS_dmu += P_k * dL_dmu;
                dS_dtheta += P_k * dL_dtheta;
            }

            double dNLL_dmu = -dlog_pmf_dmu + (dS_dmu / S_i);
            double dNLL_dtheta = -dlog_pmf_dtheta + (dS_dtheta / S_i);

            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;
            dNLL_dlog_alpha += dNLL_dtheta;
        }

        Eigen::VectorXd grad(k + 1);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k) = dNLL_dlog_alpha;

        return grad.array().isFinite().select(grad, 0.0);
    }

    /**
     * @brief Hybrid analytic/finite-difference Hessian of the negative log-likelihood.
     * @details
     * - \f$ \boldsymbol\beta\boldsymbol\beta \f$ block: \f$ X^\top\mathrm{diag}(W_i)\,X \f$ with
     *   \f$ W_i = \partial^2\mathrm{NLL}_i/\partial\mu_i^2\,(d\mu_i/d\eta_i)^2 + \partial\mathrm{NLL}_i/\partial\mu_i\,d^2\mu_i/d\eta_i^2 \f$,
     *   \f[ \frac{\partial^2\mathrm{NLL}_i}{\partial\mu^2}=-\frac{\partial^2\log P(y_i)}{\partial\mu^2}+\frac{S''S-(S')^2}{S^2},
     *       \quad \frac{\partial^2\log P(y)}{\partial\mu^2}=-\frac{y(1+\alpha\mu)+(y-\mu)\alpha\mu}{\mu^2(1+\alpha\mu)^2}, \f]
     *   \f$ S'=\sum_y P\,\ell'_y \f$, \f$ S''=\sum_y P\,(\ell'^2_y+\ell''_y) \f$ over the retained support
     *   (here \f$ y \f$ runs over \f$ 0,\dots,1000 \f$ restricted to \f$ [L,U] \f$, with early exit when \f$ P<10^{-15} \f$ and
     *   \f$ y>\mu+10\sqrt{\mu+\alpha\mu^2} \f$);
     * - the last row/column (\f$ \log\alpha \f$) and its diagonal entry: central finite differences of `gradient` with
     *   step \f$ 10^{-5} \f$ in \f$ \log\alpha \f$; the cross entries are written symmetrically.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\alpha) \f$.
     * @return Matrix of size \f$ (k+1)\times(k+1) \f$ with non-finite entries set to 0 and non-positive diagonal entries
     *         replaced by `1e-4` (no other regularisation or PSD projection).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_alpha = std::max(-10.0, std::min(10.0, params(k)));
        double alpha = std::exp(log_alpha);

        Eigen::VectorXd eta = X * beta;
        // analytic beta-beta block from the mu-derivatives of the truncated NB
        Eigen::VectorXd W_mumu = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu_deta_i = 0.0, d2mu_deta2_i = 0.0;
            evaluate_count_link_d2(eta(i), count_link, mu_i, dmu_deta_i, d2mu_deta2_i);

            double denom = 1.0 + alpha * mu_i;
            double cdf_u = (right_limit >= 0) ? nb_cdf_scalar(right_limit, mu_i, alpha) : 1.0;
            double cdf_l = (left_limit > 0) ? nb_cdf_scalar(left_limit - 1, mu_i, alpha) : 0.0;
            double S_i = std::max(cdf_u - cdf_l, 1e-12);

            double dlog_pmf_dmu = (y(i) - mu_i) / (mu_i * denom);
            double d2log_pmf_dmu2 = -(y(i) * denom + (y(i) - mu_i) * alpha * mu_i) / (mu_i * mu_i * denom * denom);

            double dS_dmu = 0.0, d2S_dmu2 = 0.0;
            for (long k_val = 0; k_val <= 1000; ++k_val) {
                bool in_range = true;
                if (left_limit > 0 && k_val < left_limit) in_range = false;
                if (right_limit >= 0 && k_val > right_limit) in_range = false;

                if (in_range) {
                    double P_k = nb_pmf_scalar(k_val, mu_i, alpha);
                    if (P_k < 1e-15 && static_cast<double>(k_val) > mu_i + 10.0 * std::sqrt(mu_i + alpha * mu_i * mu_i)) break;

                    double dL_dmu = (static_cast<double>(k_val) - mu_i) / (mu_i * denom);
                    double d2L_dmu2 = -(static_cast<double>(k_val) * denom + (static_cast<double>(k_val) - mu_i) * alpha * mu_i) / (mu_i * mu_i * denom * denom);

                    double dPk_dmu = P_k * dL_dmu;
                    dS_dmu += dPk_dmu;
                    d2S_dmu2 += (dPk_dmu * dL_dmu + P_k * d2L_dmu2);
                }
            }

            double dNLL_dmu = -dlog_pmf_dmu + (dS_dmu / S_i);
            double d2NLL_dmu2 = -d2log_pmf_dmu2 + (d2S_dmu2 * S_i - dS_dmu * dS_dmu) / (S_i * S_i);

            W_mumu(i) = d2NLL_dmu2 * (dmu_deta_i * dmu_deta_i) + dNLL_dmu * d2mu_deta2_i;
        }

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(k + 1, k + 1);
        H.block(0, 0, k, k) = X.transpose() * W_mumu.asDiagonal() * X;

        // log(alpha) row/column and diagonal by central differences of the gradient
        double h = 1e-5;
        Eigen::VectorXd p_plus = params;  p_plus(k) += h;
        Eigen::VectorXd p_minus = params; p_minus(k) -= h;

        Eigen::VectorXd g_plus = this->gradient(p_plus);
        Eigen::VectorXd g_minus = this->gradient(p_minus);

        Eigen::VectorXd d_grad_dalpha = (g_plus - g_minus) / (2.0 * h);

        H.block(0, k, k, 1) = d_grad_dalpha.head(k);
        H.block(k, 0, 1, k) = d_grad_dalpha.head(k).transpose();
        H(k, k) = d_grad_dalpha(k);

        H = H.array().isFinite().select(H, 0.0);
        for (long j = 0; j <= k; ++j) {
            if (H(j, j) <= 0.0) H(j, j) = 1e-4;
        }

        return H;
    }
};

/**
 * @brief Negative log-likelihood of a geometric regression truncated to the count range \f$ \{L,\dots,U\} \f$.
 *
 * @details Counts \f$ y_i \in \{0,1,2,\dots\} \f$ follow the geometric law with mean
 * \f$ \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$, i.e. (with \f$ q_i = \mu_i/(1+\mu_i) \f$)
 * \f[ P(y;\mu_i) = \frac{\mu_i^{y}}{(1+\mu_i)^{y+1}} = (1-q_i)\,q_i^{y}, \qquad
 *     \Pr(y\ge m) = q_i^{m}, \f]
 * conditioned on \f$ L\le y_i\le U \f$ (\f$ L = \f$ `left_limit`, \f$ U = \f$ `right_limit`), so that the truncation mass
 * is \f$ S_i = q_i^{L}-q_i^{U+1} \f$. The lower truncation is active only if \f$ L>0 \f$ (otherwise
 * \f$ q^{L}\equiv1 \f$) and the upper only if \f$ U\ge0 \f$ (default \f$ U=-1 \f$: none, \f$ q^{U+1}\equiv0 \f$).
 * The geometric law is the NB2 law with dispersion \f$ \alpha=1 \f$. The inverse link \f$ g^{-1} \f$ is evaluated with the
 * generic `evaluate_link_d2` (any `LinkType`; the `Log` link uses \f$ \exp(\min(\eta,23)) \f$).
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                         | Scale |
 * |------------|-------------------------------------------------|-------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor of \f$ \mu \f$ |
 *
 * where \f$ k \f$ = `X.cols()`. There are no extra parameters.
 *
 * ### Log-likelihood
 * With \f$ \mu_i \f$ clamped to \f$ [10^{-5},10^{6}] \f$ and \f$ S_i=\max\{q_i^{L}-q_i^{U+1},\,10^{-12}\} \f$:
 * \f[ \ell_i = y_i\log\mu_i-(y_i+1)\log(1+\mu_i)-\log S_i, \f]
 * and the functor returns \f$ -\sum_i \ell_i \f$ (no weights; no extra constants since the pmf has none).
 *
 * ### Derivatives
 * With \f$ \partial q^{m}/\partial\mu = m\,q^{m}/[\mu(1+\mu)] \f$,
 * \f[ S_i' = \frac{L\,q_i^{L}-(U+1)\,q_i^{U+1}}{\mu_i(1+\mu_i)}, \qquad
 *     S_i'' = \frac{L(L-1-2\mu_i)\,q_i^{L}-(U+1)(U-2\mu_i)\,q_i^{U+1}}{\mu_i^2(1+\mu_i)^2} \f]
 * (terms of an inactive limit are dropped), and
 * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial\mu_i}=\frac{y_i+1}{1+\mu_i}-\frac{y_i}{\mu_i}+\frac{S_i'}{S_i}, \qquad
 *     \frac{\partial^2\mathrm{NLL}_i}{\partial\mu_i^2}=\frac{y_i}{\mu_i^2}-\frac{y_i+1}{(1+\mu_i)^2}+\frac{S_i''S_i-(S_i')^2}{S_i^2}. \f]
 * The gradient is \f$ X^\top[\partial\mathrm{NLL}/\partial\mu\cdot d\mu/d\eta] \f$ and the Hessian is
 * \f$ X^\top\mathrm{diag}(W_i)X \f$ with
 * \f$ W_i=\partial^2\mathrm{NLL}_i/\partial\mu_i^2\,(d\mu_i/d\eta_i)^2+\partial\mathrm{NLL}_i/\partial\mu_i\,d^2\mu_i/d\eta_i^2 \f$.
 * Both are analytic.
 *
 * ### Numerical safeguards
 * - \f$ \mu_i \f$ clamped to \f$ [10^{-5},10^{6}] \f$ (the link derivatives are *not* zeroed when the clamp is active);
 *   \f$ S_i \f$ floored at \f$ 10^{-12} \f$ (also in the derivatives).
 * - `operator()` returns `1e10` if the sum is not finite; `gradient` performs no finiteness check.
 * - `hessian` replaces non-finite entries by 0 and non-positive diagonal entries by `1e-4`.
 *
 * @note The object stores **non-owning references** to `X` and `y`; they must outlive the functor. No validation
 *       is performed: \f$ y_i \f$ should be a non-negative integer inside \f$ [L,U] \f$.
 * @see TruncatedPoissonLoss, TruncatedNBLoss, HurdleGeometricLoss
 *
 * @ingroup glm_ll_truncated
 */
struct TruncatedGeometricLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Observed counts (length \f$ n \f$), stored as doubles; assumed integers in \f$ [L,U] \f$.
    LinkType count_link;  ///< Link whose inverse maps \f$ \eta \f$ to the geometric mean \f$ \mu \f$.
    long left_limit;  ///< Lower truncation point \f$ L \f$: counts below \f$ L \f$ are excluded; active only if \f$ L>0 \f$.
    long right_limit;  ///< Upper truncation point \f$ U \f$: counts above \f$ U \f$ are excluded; active only if \f$ U \ge 0 \f$ (negative = no upper limit).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Count response \f$ y \f$ (length \f$ n \f$), integers in \f$ [L,U] \f$.
     * @param left Lower truncation point \f$ L \f$ (default 0 = no lower truncation; 1 = zero-truncated).
     * @param right Upper truncation point \f$ U \f$ (default -1 = no upper truncation; any negative value disables it).
     * @param link Link of the geometric mean (default `LinkType::Log`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor; if both limits are active,
     *      \f$ L\le U \f$.
     */
    TruncatedGeometricLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog,
                           long left = 0, long right = -1, LinkType link = LinkType::Log)
        : X(exog), y(endog), count_link(link), left_limit(left), right_limit(right) {}

    /**
     * @brief Evaluates the total negative log-likelihood of the truncated geometric model.
     * @param params Coefficient vector \f$ \boldsymbol\beta \f$ of length \f$ k \f$.
     * @return The value \f$ -\sum_i\{y_i\log\mu_i-(y_i+1)\log(1+\mu_i)-\log S_i\} \f$ with \f$ S_i=\max\{q_i^{L}-q_i^{U+1},10^{-12}\} \f$,
     *         or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        Eigen::VectorXd eta = X * params;
        double total_neg_ll = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu = 0.0, d2mu = 0.0;
            evaluate_link_d2(eta(i), count_link, mu_i, dmu, d2mu);
            mu_i = std::max(1e-5, std::min(mu_i, 1e6));

            // untruncated log-pmf
            double log_geom = y(i) * std::log(mu_i) - (y(i) + 1.0) * std::log1p(mu_i);

            // truncation mass q^L - q^(U+1)
            double p_ratio = mu_i / (1.0 + mu_i);
            double S_L = (left_limit > 0) ? std::pow(p_ratio, static_cast<double>(left_limit)) : 1.0;
            double S_U = (right_limit >= 0) ? std::pow(p_ratio, static_cast<double>(right_limit + 1)) : 0.0;

            double prob_mass = std::max(S_L - S_U, 1e-12);
            total_neg_ll -= (log_geom - std::log(prob_mass));
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytic gradient of the negative log-likelihood.
     * @param params Coefficient vector \f$ \boldsymbol\beta \f$.
     * @return Vector \f$ X^\top\mathbf{g} \f$ of length \f$ k \f$ with
     *         \f$ g_i=[(y_i+1)/(1+\mu_i)-y_i/\mu_i+S_i'/S_i]\,d\mu_i/d\eta_i \f$; no finiteness sanitising is applied.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        Eigen::VectorXd eta = X * params;
        Eigen::VectorXd dNLL_deta(n);

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu_deta_i = 0.0, d2mu = 0.0;
            evaluate_link_d2(eta(i), count_link, mu_i, dmu_deta_i, d2mu);
            mu_i = std::max(1e-5, std::min(mu_i, 1e6));

            double p_ratio = mu_i / (1.0 + mu_i);
            double L_val = static_cast<double>(left_limit);
            double U_val = static_cast<double>(right_limit + 1);

            double term_L = (left_limit > 0) ? std::pow(p_ratio, L_val) : 1.0;
            double term_U = (right_limit >= 0) ? std::pow(p_ratio, U_val) : 0.0;

            double S_i = std::max(term_L - term_U, 1e-12);

            double dS_dmu = 0.0;
            if (left_limit > 0) dS_dmu += L_val * term_L;
            if (right_limit >= 0) dS_dmu -= U_val * term_U;
            dS_dmu /= (mu_i * (1.0 + mu_i));

            double dNLL_dmu = (y(i) + 1.0) / (1.0 + mu_i) - (y(i) / mu_i) + (dS_dmu / S_i);

            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;
        }

        return X.transpose() * dNLL_deta;
    }

    /**
     * @brief Analytic Hessian \f$ X^\top\mathrm{diag}(W_i)\,X \f$ of the negative log-likelihood.
     * @param params Coefficient vector \f$ \boldsymbol\beta \f$.
     * @return Symmetric \f$ k\times k \f$ matrix with non-finite entries set to 0 and non-positive diagonal entries
     *         replaced by `1e-4` (no other regularisation or PSD projection).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        Eigen::VectorXd eta = X * params;
        Eigen::VectorXd W = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0, dmu_deta_i = 0.0, d2mu_deta2_i = 0.0;
            evaluate_link_d2(eta(i), count_link, mu_i, dmu_deta_i, d2mu_deta2_i);
            mu_i = std::max(1e-5, std::min(mu_i, 1e6));

            double p_ratio = mu_i / (1.0 + mu_i);
            double L_val = static_cast<double>(left_limit);
            double U_val = static_cast<double>(right_limit + 1);

            double term_L = (left_limit > 0) ? std::pow(p_ratio, L_val) : 1.0;
            double term_U = (right_limit >= 0) ? std::pow(p_ratio, U_val) : 0.0;

            double S_i = std::max(term_L - term_U, 1e-12);

            double dS_dmu = 0.0;
            if (left_limit > 0) dS_dmu += L_val * term_L;
            if (right_limit >= 0) dS_dmu -= U_val * term_U;
            dS_dmu /= (mu_i * (1.0 + mu_i));

            double dNLL_dmu = (y(i) + 1.0) / (1.0 + mu_i) - (y(i) / mu_i) + (dS_dmu / S_i);

            // second derivative of the truncation mass
            double d2S_dmu2 = 0.0;
            if (left_limit > 0) d2S_dmu2 += L_val * (L_val - 1.0 - 2.0 * mu_i) * term_L;
            if (right_limit >= 0) d2S_dmu2 -= U_val * (U_val - 1.0 - 2.0 * mu_i) * term_U;
            d2S_dmu2 /= (mu_i * mu_i * (1.0 + mu_i) * (1.0 + mu_i));

            double d2NLL_dmu2 = (y(i) / (mu_i * mu_i)) - ((y(i) + 1.0) / ((1.0 + mu_i) * (1.0 + mu_i)))
                              + (d2S_dmu2 * S_i - dS_dmu * dS_dmu) / (S_i * S_i);

            W(i) = d2NLL_dmu2 * (dmu_deta_i * dmu_deta_i) + dNLL_dmu * d2mu_deta2_i;
        }

        Eigen::MatrixXd H = X.transpose() * W.asDiagonal() * X;

        H = H.array().isFinite().select(H, 0.0);
        for (long j = 0; j < k; ++j) {
            if (H(j, j) <= 0.0) H(j, j) = 1e-4;
        }

        return H;
    }
};

/**
 * @brief Negative log-likelihood of a Student-t location-scale regression with estimated degrees of freedom.
 *
 * @details Models a real response with heavy tails:
 * \f[ y_i = \mu_i + \sigma\,\varepsilon_i, \qquad \varepsilon_i \sim t_\nu, \qquad
 *     \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta), \qquad \sigma=e^{\log\sigma}, \quad \nu=e^{\log\nu}. \f]
 * The linear predictor models the *location* \f$ \mu_i \f$ (the mean when \f$ \nu>1 \f$) through `mu_link`
 * (default Identity; any `LinkType` handled by `evaluate_link`/`evaluate_link_d2`). Both the scale \f$ \sigma \f$ and the
 * degrees of freedom \f$ \nu \f$ are free scalar parameters shared by all observations.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                         | Scale |
 * |------------|-------------------------------------------------|-------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor of \f$ \mu \f$ |
 * | `k`        | \f$ \log\sigma \f$                                | log; clamped to \f$ [-10,10] \f$ |
 * | `k+1`      | \f$ \log\nu \f$                                   | log; clamped to \f$ [-4.6,10] \f$, i.e. \f$ \nu\in[0.010,\,2.2\times10^{4}] \f$ |
 *
 * where \f$ k \f$ = `X.cols()`.
 *
 * ### Log-likelihood
 * With \f$ z_i=(y_i-\mu_i)/\sigma \f$ the per-observation log-density is
 * \f[ \ell_i = \log\Gamma\big(\tfrac{\nu+1}{2}\big)-\log\Gamma\big(\tfrac{\nu}{2}\big)-\tfrac12\log(\pi\nu)-\log\sigma
 *     -\tfrac{\nu+1}{2}\log\Big(1+\frac{z_i^2}{\nu}\Big), \f]
 * and the functor returns \f$ -\sum_i\ell_i \f$ (all constants kept, no weights).
 *
 * ### Derivatives
 * Let \f$ w_i=(\nu+1)/(\nu+z_i^2) \f$ and \f$ \psi,\psi_1 \f$ the digamma/trigamma functions (`BetaMath::digamma`,
 * `BetaMath::trigamma`). Then
 * \f[ \frac{\partial\mathrm{NLL}}{\partial\mu_i}=-\frac{w_i(y_i-\mu_i)}{\sigma^2}, \qquad
 *     \frac{\partial\mathrm{NLL}}{\partial\log\sigma}=n-\sum_i w_iz_i^2, \f]
 * \f[ \frac{\partial\mathrm{NLL}}{\partial\nu}=-\tfrac n2\psi\big(\tfrac{\nu+1}{2}\big)+\tfrac n2\psi\big(\tfrac\nu2\big)+\frac{n}{2\nu}
 *     +\sum_i\Big[\tfrac12\log\Big(1+\frac{z_i^2}{\nu}\Big)-\frac{w_iz_i^2}{2\nu}\Big], \f]
 * with \f$ \partial/\partial\log\nu=\nu\,\partial/\partial\nu \f$ and the \f$ \mu \f$-derivative chained through
 * \f$ d\mu_i/d\eta_i \f$. The Hessian is exact (analytic observed information, including the link-curvature term and
 * the trigamma term for \f$ \nu \f$), with a positivity floor only on the \f$ \log\nu \f$ diagonal entry.
 *
 * ### Numerical safeguards
 * - \f$ \log\sigma \f$ and \f$ \log\nu \f$ are clamped as above; derivatives are *not* zeroed outside the clamped range
 *   (they are evaluated at the clamped values).
 * - `operator()` returns `1e10` if the sum is not finite.
 * - `gradient` replaces non-finite components by 0.
 * - `hessian` replaces non-finite entries by 0 (final step) and sets \f$ H_{\log\nu,\log\nu} \f$ to
 *   \f$ \max(10^{-4},\cdot) \f$, or `1e-4` if it is not finite.
 *
 * @note The object stores **non-owning references** to `X` and `y`; they must outlive the functor. No validation is
 *       performed (`y` must have `X.rows()` entries and `params` at least \f$ k+2 \f$ entries).
 * @see SkewTLoss, SkewNormalLoss, LogisticLoss for other location-scale families.
 *
 * @ingroup glm_ll_location_scale
 */
struct StudentTLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the location model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Real-valued response (length \f$ n \f$).
    LinkType mu_link;  ///< Link whose inverse maps \f$ \eta \f$ to the location \f$ \mu \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Response vector \f$ y \f$ (length \f$ n \f$).
     * @param link Location link (default `LinkType::Identity`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor.
     */
    StudentTLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, LinkType link = LinkType::Identity)
        : X(exog), y(endog), mu_link(link) {}

    /**
     * @brief Evaluates the total negative log-likelihood of the Student-t model.
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\log\nu) \f$ of length \f$ k+2 \f$.
     * @return The value \f$ -\sum_i\ell_i \f$ computed as \f$ n\,c(\nu,\sigma)+\tfrac{\nu+1}{2}\sum_i\log(1+z_i^2/\nu) \f$ with
     *         \f$ c=-\log\Gamma(\frac{\nu+1}{2})+\log\Gamma(\frac\nu2)+\frac12\log(\pi\nu)+\log\sigma \f$, or `1e10` if the
     *         sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        // clamp log-scale parameters, then linear predictor and normalising constant
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));

        double log_nu = std::max(-4.6, std::min(10.0, params(k + 1)));

        double sigma = std::exp(log_sigma);
        double nu = std::exp(log_nu);

        Eigen::VectorXd beta = params.head(k);
        Eigen::VectorXd eta = X * beta;

        double total_neg_ll = 0.0;
        double term_constant = -std::lgamma(0.5 * (nu + 1.0)) + std::lgamma(0.5 * nu)
                               + 0.5 * std::log(M_PI * nu) + log_sigma;

        for (long i = 0; i < n; ++i) {
            // accumulate per-observation terms
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double diff = y(i) - mu_i;
            double z = diff / sigma;

            double log_density_nll = term_constant + 0.5 * (nu + 1.0) * std::log1p((z * z) / nu);
            total_neg_ll += log_density_nll;
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytic gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\log\nu) \f$.
     * @return Vector of length \f$ k+2 \f$: \f$ X^\top[\partial\mathrm{NLL}/\partial\mu\cdot d\mu/d\eta] \f$, then
     *         \f$ \partial\mathrm{NLL}/\partial\log\sigma \f$ and \f$ \nu\,\partial\mathrm{NLL}/\partial\nu \f$. Non-finite
     *         components are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double log_nu = std::max(-4.6, std::min(10.0, params(k + 1)));

        double sigma = std::exp(log_sigma);
        double nu = std::exp(log_nu);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlog_sigma = static_cast<double>(n);
        double dNLL_dnu = static_cast<double>(n) * (-0.5 * BetaMath::digamma(0.5 * (nu + 1.0))
                                                    + 0.5 * BetaMath::digamma(0.5 * nu)
                                                    + 0.5 / nu);

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double diff = y(i) - mu_i;
            double z = diff / sigma;
            double z2 = z * z;
            double denom = nu + z2;
            double w_i = (nu + 1.0) / denom;

            double dNLL_dmu = -w_i * diff / (sigma * sigma);
            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;

            dNLL_dlog_sigma -= w_i * z2;
            dNLL_dnu += 0.5 * std::log1p(z2 / nu) - 0.5 * w_i * z2 / nu;
        }

        Eigen::VectorXd grad(k + 2);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k) = dNLL_dlog_sigma;

        grad(k + 1) = dNLL_dnu * nu;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Exact analytic observed Hessian of the negative log-likelihood.
     * @details Per observation, with \f$ f=\partial\mathrm{NLL}_i/\partial z_i=w_iz_i \f$ and
     * \f$ f'=(\nu+1)(\nu-z_i^2)/(\nu+z_i^2)^2 \f$:
     * - \f$ \partial^2/\partial\mu^2 = f'/\sigma^2 \f$, giving \f$ W_{\mu\mu,i}=f'\sigma^{-2}(d\mu/d\eta)^2-(f/\sigma)\,d^2\mu/d\eta^2 \f$;
     * - \f$ \partial^2/\partial\mu\partial\log\sigma = (f+z_if')/\sigma \f$ (times \f$ d\mu/d\eta \f$);
     * - \f$ \partial^2/\partial\mu\partial\nu = -z_i(z_i^2-1)/[\sigma(\nu+z_i^2)^2] \f$ (times \f$ \nu\,d\mu/d\eta \f$ for \f$ \log\nu \f$);
     * - \f$ \partial^2/\partial(\log\sigma)^2 = \sum_i 2\nu w_iz_i^2/(\nu+z_i^2) \f$;
     * - \f$ \partial^2/\partial\log\sigma\,\partial\nu = -\sum_i z_i^2(z_i^2-1)/(\nu+z_i^2)^2 \f$ (times \f$ \nu \f$);
     * - \f$ H_{\log\nu,\log\nu}=\nu^2\,\partial^2\mathrm{NLL}/\partial\nu^2+\nu\,\partial\mathrm{NLL}/\partial\nu \f$, where
     *   \f$ \partial^2\mathrm{NLL}/\partial\nu^2 = n[-\tfrac14\psi_1(\tfrac{\nu+1}{2})+\tfrac14\psi_1(\tfrac\nu2)-\tfrac1{2\nu^2}]
     *   +\sum_i[\tfrac{z_i^2}{2\nu^2(\nu+z_i^2)}-\tfrac{z_i^2(z_i^2-1)}{2\nu(\nu+z_i^2)^2}] \f$.
     *
     * The blocks are assembled as \f$ H_{\beta\beta}=X^\top\mathrm{diag}(W_{\mu\mu})X \f$, the \f$ \beta \f$--\f$ \log\sigma \f$ and
     * \f$ \beta \f$--\f$ \log\nu \f$ columns as \f$ X^\top W \f$ (mirrored to the rows), and the 2x2 scalar block.
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\log\nu) \f$.
     * @return Symmetric \f$ (k+2)\times(k+2) \f$ matrix. Non-finite entries are set to 0; \f$ H_{\log\nu,\log\nu} \f$ is
     *         floored at `1e-4`. The matrix is not otherwise regularised and need not be positive definite.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));

        double log_nu = std::max(-4.6, std::min(10.0, params(k + 1)));

        double sigma = std::exp(log_sigma);
        double nu = std::exp(log_nu);

        Eigen::VectorXd eta = X * beta;

        // accumulators of the per-observation second derivatives
        Eigen::VectorXd W_mumu = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_musigma = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_mulognu = Eigen::VectorXd::Zero(n);

        double d2NLL_dlogsigma2 = 0.0;
        double d2NLL_dlogsigmadnu = 0.0;

        double dNLL_dnu_const = static_cast<double>(n) * (-0.5 * BetaMath::digamma(0.5 * (nu + 1.0))
                                                          + 0.5 * BetaMath::digamma(0.5 * nu)
                                                          + 0.5 / nu);

        double d2NLL_dnu2_const = static_cast<double>(n) * (-0.25 * BetaMath::trigamma(0.5 * (nu + 1.0))
                                                             + 0.25 * BetaMath::trigamma(0.5 * nu)
                                                             - 0.5 / (nu * nu));

        double dNLL_dnu = dNLL_dnu_const;
        double d2NLL_dnu2 = d2NLL_dnu2_const;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0; double d2mu_deta2_i = 0.0;
            evaluate_link_d2(eta(i), mu_link, mu_i, dmu_deta_i, d2mu_deta2_i);

            double diff = y(i) - mu_i;
            double z = diff / sigma;
            double z2 = z * z;
            double denom = nu + z2;
            double denom2 = denom * denom;
            double w_i = (nu + 1.0) / denom;

            double dNLL_dz = w_i * z;
            double d2NLL_dz2 = (nu + 1.0) * (nu - z2) / denom2;

            double dNLL_dmu = -dNLL_dz / sigma;
            double d2NLL_dmu2 = d2NLL_dz2 / (sigma * sigma);
            double d2NLL_dmudlogsigma = (dNLL_dz + z * d2NLL_dz2) / sigma;
            double d2NLL_dmudnu = -z * (z2 - 1.0) / (sigma * denom2);

            W_mumu(i) = d2NLL_dmu2 * dmu_deta_i * dmu_deta_i + dNLL_dmu * d2mu_deta2_i;
            W_musigma(i) = d2NLL_dmudlogsigma * dmu_deta_i;
            W_mulognu(i) = (d2NLL_dmudnu * dmu_deta_i) * nu;

            d2NLL_dlogsigma2 += 2.0 * nu * w_i * z2 / denom;
            d2NLL_dlogsigmadnu -= z2 * (z2 - 1.0) / denom2;

            dNLL_dnu += 0.5 * std::log1p(z2 / nu) - 0.5 * w_i * z2 / nu;
            d2NLL_dnu2 += (z2 / (2.0 * nu * nu * denom)) - (z2 * (z2 - 1.0) / (2.0 * nu * denom2));
        }

        // assemble Hessian blocks
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(k + 2, k + 2);

        H.block(0, 0, k, k) = X.transpose() * W_mumu.asDiagonal() * X;

        Eigen::VectorXd H_beta_logsigma = X.transpose() * W_musigma;
        H.block(0, k, k, 1) = H_beta_logsigma;
        H.block(k, 0, 1, k) = H_beta_logsigma.transpose();

        Eigen::VectorXd H_beta_lognu = X.transpose() * W_mulognu;
        H.block(0, k + 1, k, 1) = H_beta_lognu;
        H.block(k + 1, 0, 1, k) = H_beta_lognu.transpose();

        H(k, k) = d2NLL_dlogsigma2;

        double d2NLL_dlogsigmadlognu = d2NLL_dlogsigmadnu * nu;
        H(k, k + 1) = d2NLL_dlogsigmadlognu;
        H(k + 1, k) = d2NLL_dlogsigmadlognu;

        double h_nu_nu = d2NLL_dnu2 * nu * nu + dNLL_dnu * nu;
        H(k + 1, k + 1) = std::isfinite(h_nu_nu) ? std::max(1e-4, h_nu_nu) : 1e-4;

        return H.array().isFinite().select(H, 0.0);
    }

};

/**
 * @brief Negative log-likelihood of a mean-parameterised skew-normal regression (Azzalini family, centred parameterisation).
 *
 * @details The response follows an Azzalini skew-normal law
 * \f[ f(y)=\frac{2}{\sigma}\,\phi(z)\,\Phi(\alpha z), \qquad z=\frac{y-\xi}{\sigma}, \f]
 * with shape \f$ \alpha \f$ and scale \f$ \sigma \f$, whose location \f$ \xi \f$ is chosen so that the **mean equals the
 * regression mean**. With \f$ \delta=\alpha/\sqrt{1+\alpha^2} \f$ and \f$ c=\sqrt{2/\pi}\,\delta \f$ the mean of
 * \f$ f \f$ is \f$ \xi+\sigma c \f$, hence
 * \f[ \xi_i=\mu_i-\sigma c, \qquad \mu_i=g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta), \qquad
 *     z_i=\frac{y_i-\mu_i}{\sigma}+c=u_i. \f]
 * The linear predictor therefore models the *mean* \f$ \mu_i \f$ through `mu_link` (default Identity; any `LinkType`
 * supported by `evaluate_link`/`evaluate_link_d2`). \f$ \sigma \f$ and \f$ \alpha \f$ are scalar parameters shared by all
 * observations; \f$ \alpha>0 \f$ gives right skew, \f$ \alpha<0 \f$ left skew, \f$ \alpha=0 \f$ the Normal.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                         | Scale |
 * |------------|-------------------------------------------------|-------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor of the mean \f$ \mu \f$ |
 * | `k`        | \f$ \log\sigma \f$                                | log; clamped to \f$ \pm \f$`LOG_SIGMA_MAX` \f$ = \pm10 \f$ |
 * | `k+1`      | \f$ \alpha \f$ (shape)                            | identity scale (not logged); clamped to \f$ \pm \f$`ALPHA_MAX` \f$ = \pm15 \f$ |
 *
 * where \f$ k \f$ = `X.cols()`. Decoding and clamping are done by `unpack`.
 *
 * ### Log-likelihood
 * With \f$ u_i \f$ as above, the full NLL is
 * \f[ -\sum_i\ell_i = n\,(\log\sigma+\tfrac12\log(2\pi)-\log2)+\sum_i\Big[\tfrac12u_i^2-\log\Phi(\alpha u_i)\Big], \f]
 * where \f$ \log\Phi \f$ is evaluated by the stable routine `log_Phi_cdf`. All constants are kept; no weights.
 *
 * ### Derivatives
 * Let \f$ w_i=(y_i-\mu_i)/\sigma \f$, \f$ u_i=w_i+c \f$, \f$ R(x)=\phi(x)/\Phi(x) \f$ (`mills`), \f$ f_i=u_i-\alpha R(\alpha u_i) \f$,
 * \f$ c'=dc/d\alpha=\sqrt{2/\pi}(1+\alpha^2)^{-3/2} \f$ and \f$ c''=-3\alpha\sqrt{2/\pi}(1+\alpha^2)^{-5/2} \f$. Then
 * \f[ \frac{\partial\mathrm{NLL}_i}{\partial\eta_i}=-\frac{f_i}{\sigma}\frac{d\mu_i}{d\eta_i}, \qquad
 *     \frac{\partial\mathrm{NLL}}{\partial\log\sigma}=n-\sum_iw_if_i, \qquad
 *     \frac{\partial\mathrm{NLL}}{\partial\alpha}=\sum_i\big[c'f_i-R(\alpha u_i)\,u_i\big]. \f]
 * The Hessian is analytic and exact (it also contains the link-curvature term and \f$ c'' \f$). A regularised
 * (PSD-projected) variant is provided by `hessian_regularized`.
 *
 * ### Numerical safeguards
 * - Parameters outside the clamp ranges are *saturated*: `unpack` records `sat_s` / `sat_a` and uses the clamped values.
 *   For a saturated parameter `gradient` returns 0 in its component and `hessian` replaces its row and column by the
 *   corresponding identity row/column (diagonal 1, off-diagonal 0).
 * - `log_Phi_cdf`, `mills` and `x_plus_mills` switch to an asymptotic series for \f$ x\le-20 \f$ (`ASYMP_THRESH`) to avoid
 *   underflow/cancellation in the lower tail.
 * - `operator()` returns `1e10` if the NLL is not finite; the \f$ \log\sigma \f$ and \f$ \alpha \f$ gradient components are
 *   set to 0 if not finite (the \f$ \beta \f$ components are not checked).
 * - `hessian` replaces non-finite entries by 0.
 *
 * @note The object stores **non-owning references** to `X` and `y`; they must outlive the functor. No validation is
 *       performed. Note that \f$ \alpha \f$ is on its natural (not log) scale and bounded to \f$ [-15,15] \f$.
 * @see SkewTLoss, StudentTLoss
 *
 * @ingroup glm_ll_location_scale
 */
struct SkewNormalLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the mean model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Real-valued response (length \f$ n \f$).
    LinkType mu_link;  ///< Link whose inverse maps \f$ \eta \f$ to the mean \f$ \mu \f$ of the response.

    static constexpr double LOG_SIGMA_MAX = 10.0;  ///< Bound on \f$ |\log\sigma| \f$: parameter values outside \f$ [-10,10] \f$ are clamped and flagged as saturated.
    static constexpr double ALPHA_MAX     = 15.0;  ///< Bound on \f$ |\alpha| \f$: shape values outside \f$ [-15,15] \f$ are clamped and flagged as saturated.
    static constexpr double SQRT_2_OVER_PI = 0.7978845608028654;  ///< Constant \f$ \sqrt{2/\pi} \f$.
    static constexpr double HALF_LOG_2PI   = 0.9189385332046727;  ///< Constant \f$ \tfrac12\log(2\pi) \f$.
    static constexpr double LOG_2          = 0.6931471805599453;  ///< Constant \f$ \log2 \f$.
    static constexpr double ASYMP_THRESH   = -20.0;  ///< Threshold \f$ -20 \f$: for \f$ x \f$ below it the lower-tail routines use the asymptotic series.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Response vector \f$ y \f$ (length \f$ n \f$).
     * @param link Mean link (default `LinkType::Identity`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor.
     */
    SkewNormalLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog,
                   LinkType link = LinkType::Identity)
        : X(exog), y(endog), mu_link(link) {}

    /**
     * @brief Asymptotic series for the lower-tail Mills factor of the normal CDF.
     * @details For \f$ x\to-\infty \f$, \f$ \Phi(x)\approx\phi(x)\,S(x)/(-x) \f$ with
     * \f$ S(x)=1-t+3t^2-15t^3+105t^4-945t^5+10395t^6 \f$, \f$ t=1/x^2 \f$ (Horner form).
     * @param x Argument, intended for \f$ x\le-20 \f$.
     * @return The truncated series \f$ S(x) \f$.
     */
    static inline double asymp_S(double x) {
        const double t = 1.0 / (x * x);
        return 1.0 + t * (-1.0 + t * (3.0 + t * (-15.0 + t * (105.0 + t * (-945.0 + t * 10395.0)))));
    }

    /**
     * @brief Cancellation-free evaluation of \f$ 1-S(x) \f$ for the asymptotic series.
     * @details Returns \f$ t-3t^2+15t^3-105t^4+945t^5-10395t^6 \f$ with \f$ t=1/x^2 \f$ (Horner form), i.e.
     *          exactly \f$ 1-S(x) \f$ without subtracting numbers close to 1.
     * @param x Argument, intended for \f$ x\le-20 \f$.
     * @return The value \f$ 1-S(x) \f$.
     */
    static inline double asymp_1mS(double x) {
        const double t = 1.0 / (x * x);
        return t * (1.0 - t * (3.0 - t * (15.0 - t * (105.0 - t * (945.0 - t * 10395.0)))));
    }

    /**
     * @brief Numerically stable \f$ \log\Phi(x) \f$ for the standard normal CDF.
     * @details For \f$ x>-20 \f$: \f$ \log\{\tfrac12\,\mathrm{erfc}(-x/\sqrt2)\} \f$. For \f$ x\le-20 \f$ the asymptotic form
     *          \f$ -\tfrac12\log(2\pi)-\tfrac12x^2-\log(-x)+\log S(x) \f$ (see `asymp_S`), which avoids underflow of `erfc`.
     * @param x Argument.
     * @return The value \f$ \log\Phi(x) \f$.
     */
    static inline double log_Phi_cdf(double x) {
        if (x > ASYMP_THRESH) {

            return std::log(0.5 * std::erfc(-x * 0.7071067811865475));
        }
        return -HALF_LOG_2PI - 0.5 * x * x - std::log(-x) + std::log(asymp_S(x));
    }

    /**
     * @brief Lower-tail inverse Mills ratio \f$ R(x)=\phi(x)/\Phi(x) \f$.
     * @details For \f$ x>-20 \f$: \f$ \exp\{-\tfrac12x^2-\tfrac12\log(2\pi)-\log\Phi(x)\} \f$; for \f$ x\le-20 \f$ the asymptotic
     *          form \f$ -x/S(x) \f$.
     * @param x Argument.
     * @return The ratio \f$ R(x) \f$ (always positive).
     */
    static inline double mills(double x) {
        if (x > ASYMP_THRESH) return std::exp(-0.5 * x * x - HALF_LOG_2PI - log_Phi_cdf(x));
        return -x / asymp_S(x);
    }

    /**
     * @brief Cancellation-safe evaluation of \f$ x+R(x) \f$.
     * @details For \f$ x>-20 \f$ the plain sum; for \f$ x\le-20 \f$ the equivalent form \f$ (-x)\,[1-S(x)]/S(x) \f$ (via
     *          `asymp_1mS`), since \f$ x \f$ and \f$ R(x) \f$ nearly cancel in the far lower tail. Appears in
     *          \f$ R'(x)=-R(x)\,[x+R(x)] \f$.
     * @param x Argument.
     * @return The value \f$ x+R(x) \f$.
     */
    static inline double x_plus_mills(double x) {
        if (x > ASYMP_THRESH) return x + mills(x);
        return (-x) * asymp_1mS(x) / asymp_S(x);
    }

    /**
     * @brief Decoded and clamped parameter set produced by `unpack`.
     */
    struct Pars {
        Eigen::VectorXd beta;  ///< Regression coefficients \f$ \boldsymbol\beta \f$ (first \f$ k \f$ entries of the parameter vector).
        double log_sigma, alpha, sigma, c, dc, d2c;  ///< Clamped \f$ \log\sigma \f$ and \f$ \alpha \f$; \f$ \sigma=e^{\log\sigma} \f$; \f$ c=\sqrt{2/\pi}\,\alpha/\sqrt{1+\alpha^2} \f$ (mean correction per unit \f$ \sigma \f$) and its derivatives \f$ dc/d\alpha \f$, \f$ d^2c/d\alpha^2 \f$.
        bool sat_s, sat_a;  ///< True if the raw \f$ \log\sigma \f$ (resp. \f$ \alpha \f$) lay outside its clamp range (saturated parameter).
    };
    /**
     * @brief Splits the parameter vector, clamps \f$ \log\sigma \f$ and \f$ \alpha \f$ and precomputes the mean-correction terms.
     * @details With \f$ s=\sqrt{1+\alpha^2} \f$:
     * \f$ c=\sqrt{2/\pi}\,\alpha/s \f$, \f$ c'=\sqrt{2/\pi}/s^3 \f$, \f$ c''=-3\alpha\sqrt{2/\pi}/s^5 \f$. The saturation flags are
     * set from the *raw* values (strictly outside the closed clamp interval).
     * @param p Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\alpha) \f$ of length \f$ k+2 \f$.
     * @return The decoded `Pars` structure.
     */
    Pars unpack(const Eigen::VectorXd& p) const {
        const long k = X.cols();
        Pars q;
        q.beta = p.head(k);
        q.sat_s = (p(k) < -LOG_SIGMA_MAX) || (p(k) > LOG_SIGMA_MAX);
        q.sat_a = (p(k + 1) < -ALPHA_MAX) || (p(k + 1) > ALPHA_MAX);
        q.log_sigma = std::max(-LOG_SIGMA_MAX, std::min(LOG_SIGMA_MAX, p(k)));
        q.alpha     = std::max(-ALPHA_MAX, std::min(ALPHA_MAX, p(k + 1)));
        q.sigma = std::exp(q.log_sigma);
        const double s = std::sqrt(1.0 + q.alpha * q.alpha);
        q.c   = SQRT_2_OVER_PI * q.alpha / s;
        q.dc  = SQRT_2_OVER_PI / (s * s * s);
        q.d2c = -3.0 * q.alpha * SQRT_2_OVER_PI / (s * s * s * s * s);
        return q;
    }

    /**
     * @brief Per-observation NLL term without the \f$ \log\sigma+\tfrac12\log(2\pi)-\log2 \f$ constant.
     * @details Evaluates \f$ \mu=g^{-1}(\eta) \f$ with `evaluate_link`, the location \f$ \xi=\mu-\texttt{mean\_correction} \f$,
     *          \f$ z=(y-\xi)/\sigma \f$ and returns \f$ \tfrac12z^2-\log\Phi(\alpha z) \f$.
     * @param eta_i Linear predictor \f$ \eta_i \f$.
     * @param y_i Observed response.
     * @param sigma Scale \f$ \sigma>0 \f$.
     * @param alpha Shape \f$ \alpha \f$.
     * @param mean_correction Value \f$ \sigma c \f$ subtracted from the mean to obtain the location \f$ \xi \f$.
     * @return The value \f$ \tfrac12z^2-\log\Phi(\alpha z) \f$.
     */
    double evaluate_single_nll(double eta_i, double y_i, double sigma, double alpha,
                               double mean_correction) const {
        double mu_base = 0.0, dummy = 0.0;
        evaluate_link(eta_i, mu_link, mu_base, dummy);
        const double z = (y_i - (mu_base - mean_correction)) / sigma;
        return 0.5 * z * z - log_Phi_cdf(alpha * z);
    }

    /**
     * @brief Evaluates the total negative log-likelihood of the skew-normal model.
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\alpha) \f$ of length \f$ k+2 \f$ (clamped by `unpack`).
     * @return The value \f$ n(\log\sigma+\tfrac12\log(2\pi)-\log2)+\sum_i\mathrm{evaluate\_single\_nll}(\eta_i,y_i,\sigma,\alpha,\sigma c) \f$,
     *         or `1e10` if it is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const Pars q = unpack(params);
        // location xi = mu - sigma*c so that the model mean is mu
        const double mean_correction = q.sigma * q.c;
        const Eigen::VectorXd eta = X * q.beta;

        double nll = static_cast<double>(n) * (q.log_sigma + HALF_LOG_2PI - LOG_2);
        for (long i = 0; i < n; ++i)
            nll += evaluate_single_nll(eta(i), y(i), q.sigma, q.alpha, mean_correction);
        return std::isfinite(nll) ? nll : 1e10;
    }

    /**
     * @brief Analytic gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\alpha) \f$.
     * @return Vector of length \f$ k+2 \f$: \f$ X^\top[-f_i\sigma^{-1}d\mu_i/d\eta_i] \f$, then
     *         \f$ n-\sum_iw_if_i \f$ for \f$ \log\sigma \f$ and \f$ \sum_i[c'f_i-R(\alpha u_i)u_i] \f$ for \f$ \alpha \f$. The
     *         \f$ \log\sigma \f$ (resp. \f$ \alpha \f$) component is 0 if that parameter is saturated or the value is not finite.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        const Pars q = unpack(params);
        const Eigen::VectorXd eta = X * q.beta;

        Eigen::VectorXd dNLL_deta(n);
        double g_s = static_cast<double>(n);
        double g_a = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu = 0.0, dmu = 0.0;
            evaluate_link(eta(i), mu_link, mu, dmu);

            // per-observation score terms w, u, R, f_u
            const double w = (y(i) - mu) / q.sigma;
            const double u = w + q.c;
            const double R = mills(q.alpha * u);
            const double f_u = u - q.alpha * R;

            dNLL_deta(i) = -f_u * dmu / q.sigma;
            g_s -= w * f_u;
            g_a += q.dc * f_u - R * u;
        }

        Eigen::VectorXd grad(k + 2);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k)     = (q.sat_s || !std::isfinite(g_s)) ? 0.0 : g_s;
        grad(k + 1) = (q.sat_a || !std::isfinite(g_a)) ? 0.0 : g_a;
        return grad;
    }

    /**
     * @brief Exact analytic observed Hessian of the negative log-likelihood.
     * @details With \f$ a=\alpha u \f$, \f$ R=R(a) \f$, \f$ \rho=R\,(a+R) \f$ (so that \f$ R'(a)=-\rho \f$),
     * \f$ f_u=u-\alpha R \f$, \f$ f_{uu}=1+\alpha^2\rho \f$ and \f$ F_\alpha=c'f_{uu}-R+\alpha\rho u \f$ (the \f$ \alpha \f$-derivative
     * of \f$ f_u \f$), the per-observation weights are
     * - \f$ W^{ee}_i=f_{uu}(d\mu/d\eta)^2/\sigma^2-f_u\,(d^2\mu/d\eta^2)/\sigma \f$,
     * - \f$ W^{e\sigma}_i=(d\mu/d\eta)(w_if_{uu}+f_u)/\sigma \f$,
     * - \f$ W^{e\alpha}_i=-(d\mu/d\eta)F_\alpha/\sigma \f$,
     *
     * and the scalar entries are \f$ H_{\sigma\sigma}=\sum_i(w_if_u+w_i^2f_{uu}) \f$, \f$ H_{\sigma\alpha}=-\sum_iw_iF_\alpha \f$ and
     * \f$ H_{\alpha\alpha}=\sum_i[c''f_u+c'F_\alpha+\rho(u+\alpha c')u-Rc'] \f$ (with respect to \f$ \log\sigma \f$ and \f$ \alpha \f$).
     * Blocks: \f$ H_{\beta\beta}=X^\top\mathrm{diag}(W^{ee})X \f$, \f$ H_{\beta\sigma}=X^\top W^{e\sigma} \f$,
     * \f$ H_{\beta\alpha}=X^\top W^{e\alpha} \f$ (mirrored to the rows).
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\alpha) \f$.
     * @return Symmetric \f$ (k+2)\times(k+2) \f$ matrix. Non-finite entries are set to 0; for a saturated \f$ \log\sigma \f$ or
     *         \f$ \alpha \f$ its row and column are zeroed and the diagonal set to 1. No positive-definiteness is enforced
     *         (see `hessian_regularized`).
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        const Pars q = unpack(params);
        const Eigen::VectorXd eta = X * q.beta;

        Eigen::VectorXd W_ee(n), W_es(n), W_ea(n);
        double H_ss = 0.0, H_sa = 0.0, H_aa = 0.0;
        const double a2 = q.alpha * q.alpha;

        for (long i = 0; i < n; ++i) {
            double mu = 0.0, dmu = 0.0, d2mu = 0.0;
            evaluate_link_d2(eta(i), mu_link, mu, dmu, d2mu);

            // per-observation derivatives of the score f_u and the weights
            const double w = (y(i) - mu) / q.sigma;
            const double u = w + q.c;
            const double a = q.alpha * u;
            const double R = mills(a);
            const double Rapr = R * x_plus_mills(a);

            const double f_u  = u - q.alpha * R;
            const double f_uu = 1.0 + a2 * Rapr;
            const double F_a  = q.dc * f_uu + (-R + q.alpha * Rapr * u);

            W_ee(i) = f_uu * dmu * dmu / (q.sigma * q.sigma) - f_u * d2mu / q.sigma;
            W_es(i) = (dmu / q.sigma) * (w * f_uu + f_u);
            W_ea(i) = -(dmu / q.sigma) * F_a;

            H_ss += w * f_u + w * w * f_uu;
            H_sa -= w * F_a;
            H_aa += q.d2c * f_u + q.dc * F_a + Rapr * (u + q.alpha * q.dc) * u - R * q.dc;
        }

        // assemble Hessian blocks
        Eigen::MatrixXd H(k + 2, k + 2);
        H.block(0, 0, k, k) = X.transpose() * W_ee.asDiagonal() * X;
        const Eigen::VectorXd h_bs = X.transpose() * W_es;
        const Eigen::VectorXd h_ba = X.transpose() * W_ea;
        H.block(0, k, k, 1) = h_bs;      H.block(k, 0, 1, k) = h_bs.transpose();
        H.block(0, k + 1, k, 1) = h_ba;  H.block(k + 1, 0, 1, k) = h_ba.transpose();
        H(k, k) = H_ss;
        H(k, k + 1) = H(k + 1, k) = H_sa;
        H(k + 1, k + 1) = H_aa;

        H = H.array().isFinite().select(H, 0.0).matrix();

        // decouple saturated parameters
        for (int j : {0, 1}) {
            if ((j == 0 && q.sat_s) || (j == 1 && q.sat_a)) {
                const long idx = k + j;
                H.row(idx).setZero();
                H.col(idx).setZero();
                H(idx, idx) = 1.0;
            }
        }
        return H;
    }

    /**
     * @brief Positive-definite regularisation of the Hessian by eigenvalue clipping.
     * @details Symmetrises \f$ H=\tfrac12(H_0+H_0^\top) \f$ with \f$ H_0 \f$ = `hessian(params)`, computes the eigendecomposition
     *          \f$ H=V\Lambda V^\top \f$ and returns \f$ V\max(\Lambda,\epsilon)V^\top \f$ with \f$ \epsilon= \f$ `min_eig`. If the
     *          eigensolver fails, \f$ H+\epsilon I \f$ is returned instead.
     * @param params Parameter vector \f$ (\boldsymbol\beta,\log\sigma,\alpha) \f$.
     * @param min_eig Smallest allowed eigenvalue \f$ \epsilon \f$ (default \f$ 10^{-8} \f$).
     * @return Symmetric \f$ (k+2)\times(k+2) \f$ matrix whose eigenvalues are all \f$ \ge\epsilon \f$ (up to rounding).
     */
    Eigen::MatrixXd hessian_regularized(const Eigen::VectorXd& params, double min_eig = 1e-8) const {
        Eigen::MatrixXd H = hessian(params);
        H = 0.5 * (H + H.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        if (es.info() != Eigen::Success) return H + min_eig * Eigen::MatrixXd::Identity(H.rows(), H.cols());
        Eigen::VectorXd ev = es.eigenvalues().cwiseMax(min_eig);
        return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }
};

/**
 * @brief Negative log-likelihood of a linear regression model with Azzalini skew-Student-t errors, parameterised through the mean.
 *
 * @details Models a real-valued response \f$ y_i \in \mathbb{R} \f$ with the Azzalini skew-\f$ t \f$ density
 * \f[ f(y_i) = \frac{2}{\sigma}\, t_\nu(z_i)\; T_{\nu+1}\!\Big( \alpha z_i \sqrt{\tfrac{\nu+1}{\nu+z_i^2}} \Big),
 *     \qquad z_i = \frac{y_i - \xi_i}{\sigma}, \f]
 * where \f$ t_\nu \f$ and \f$ T_{\nu} \f$ are the density and CDF of the standard Student-\f$ t \f$ distribution
 * with \f$ \nu \f$ degrees of freedom, \f$ \sigma > 0 \f$ is the scale, \f$ \alpha \in \mathbb{R} \f$ the
 * (unbounded) skewness (slant) parameter and \f$ \nu > 0 \f$ the degrees of freedom.
 *
 * The linear predictor \f$ \eta_i = \mathbf{x}_i^\top \boldsymbol\beta \f$ models the **mean** of \f$ y_i \f$ rather
 * than the location: \f$ \mu_i = g^{-1}(\eta_i) \f$ (inverse link evaluated by `evaluate_link`, default `Identity`) and the
 * location is shifted by the mean of the skew-\f$ t \f$ distribution,
 * \f[ \xi_i = \mu_i - c, \qquad c = \sigma\,\delta\,\sqrt{\nu/\pi}\;\frac{\Gamma((\nu-1)/2)}{\Gamma(\nu/2)},
 *     \qquad \delta = \frac{\alpha}{\sqrt{1+\alpha^2}}, \f]
 * so that \f$ E[y_i] = \mu_i \f$ (valid for \f$ \nu > 1 \f$; the code uses \f$ c = 0 \f$ if \f$ \nu \le 1 \f$).
 * The scale, the skewness and the degrees of freedom are common to all observations.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                          | Scale / transform                                   |
 * |------------|--------------------------------------------------|-----------------------------------------------------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor scale (not clamped)                |
 * | `k`        | \f$ \log\sigma \f$                               | log; clamped to \f$ [-10, 10] \f$ (`LS_MIN`, `LS_MAX`) |
 * | `k+1`      | \f$ \alpha \f$                                   | identity; clamped to \f$ [-20, 20] \f$ (`A_MIN`, `A_MAX`) |
 * | `k+2`      | \f$ \log\nu \f$                                  | log; clamped to \f$ [0.1, 7] \f$ (`LN_MIN`, `LN_MAX`), i.e. \f$ \nu \in [e^{0.1}, e^{7}] \approx [1.105, 1097] \f$ |
 *
 * where \f$ k \f$ is the number of columns of the design matrix; the vector has length \f$ k+3 \f$.
 * Values outside the clamp interval are evaluated at the nearest bound and additionally penalised
 * (see "Numerical safeguards").
 *
 * ### Log-likelihood
 * With \f$ \xi_i \f$, \f$ z_i \f$ as above and \f$ \rho_i = \sqrt{(\nu+1)/(\nu+z_i^2)} \f$ the NLL is
 * \f[ \mathrm{NLL} = \sum_{i=1}^n \Big[ \log\sigma - \log 2 - \log t_\nu(z_i) - \log T_{\nu+1}(\alpha z_i \rho_i) \Big]
 *     + \text{penalty}, \f]
 * with \f$ \log t_\nu(x) = \log\Gamma\big(\tfrac{\nu+1}{2}\big) - \log\Gamma\big(\tfrac{\nu}{2}\big)
 * - \tfrac12\log(\pi\nu) - \tfrac{\nu+1}{2}\log\big(1 + x^2/\nu\big) \f$. No constant is dropped. The term
 * \f$ n(\log\sigma - \log 2) \f$ is added once and the per-observation part is computed by `evaluate_single_nll()`.
 * All observations have unit weight; there is no censoring or truncation.
 *
 * ### Derivatives
 * - **Gradient**: analytical chain rule through \f$ z_i \f$, \f$ \xi_i \f$ and \f$ \eta_i \f$, except for the derivative
 *   of \f$ \log T_{\nu+1} \f$ with respect to its degrees of freedom, which is a central finite difference
 *   (step `1e-5` on the argument \f$ \nu+1 \f$) of `log_T_cdf()`. The \f$ \nu \f$-derivative therefore is only
 *   approximately analytical.
 * - **Hessian**: *not* analytical; `hessian()` is a central finite difference of `gradient()` (see `hessian_raw()`),
 *   symmetrised and finite-masked, without regularisation. `hessian_psd()` additionally projects it onto the positive
 *   definite cone by flooring eigenvalues at `1e-8`.
 * - `check_gradient()` compares the gradient against central finite differences of the objective.
 *
 * ### Numerical safeguards
 * - Hard clamps of \f$ \log\sigma \f$, \f$ \alpha \f$, \f$ \log\nu \f$ (constants `LS_MIN`/`LS_MAX`, `A_MIN`/`A_MAX`,
 *   `LN_MIN`/`LN_MAX`) in the objective and the gradient.
 * - Quadratic exterior penalty \f$ \tfrac12 W e^2 \f$ with \f$ W = 100 \f$ (`PEN_W`), \f$ e \f$ being the distance of the
 *   raw parameter to its interval, added to the objective; its derivative is added to the gradient. The likelihood
 *   part of the gradient of a parameter is set to 0 when that parameter is not strictly inside its interval
 *   (the clamped objective does not depend on it there).
 * - A non-finite per-observation NLL contributes `1e10` to the objective; an observation with any non-finite gradient
 *   contribution is skipped in the gradient (its \f$ \eta \f$-derivative is 0, but its constant `1` in the
 *   \f$ \log\sigma \f$ derivative is kept).
 * - Non-finite totals of the gradient (for the three scalar parameters) are replaced by 0; non-finite Hessian
 *   entries are replaced by 0.
 * - Special-function calls use boost.math with the policy `Pol` (errors give `errno`/NaN/Inf instead of exceptions);
 *   the Student-\f$ t \f$ CDF falls back to a continued fraction in the far tails.
 *
 * @note The object stores **non-owning references** to the design matrix and the response: they must outlive the
 *       functor. No validation is performed (sizes of `params`, finiteness of data). Because
 *       \f$ \log\nu \ge 0.1 \f$ always holds after clamping, \f$ \nu > 1 \f$ and the mean correction is always active
 *       in practice; the \f$ \nu \le 1 \f$ branches are unreachable through the parameter vector.
 * @see StudentTLoss, SkewNormalLoss for the symmetric and the skew-normal special cases.
 *
 * @ingroup glm_ll_location_scale
 */
struct SkewTLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the mean model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Real-valued response vector (length \f$ n \f$).
    LinkType mu_link;  ///< Link type whose inverse maps \f$ \eta_i \f$ to the mean \f$ \mu_i \f$ of \f$ y_i \f$.

    /**
     * @brief Boost.math error policy used by `log_gamma`, `digamma`, `ibeta` and friends.
     * @details Domain, pole, overflow and evaluation errors are reported through `errno` (a NaN/Inf is returned instead of
     * throwing), underflow errors are ignored, and arguments are not promoted to higher precision (`promote_double<false>`).
     */
    using Pol = boost::math::policies::policy<
        boost::math::policies::domain_error<boost::math::policies::errno_on_error>,
        boost::math::policies::pole_error<boost::math::policies::errno_on_error>,
        boost::math::policies::overflow_error<boost::math::policies::errno_on_error>,
        boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
        boost::math::policies::evaluation_error<boost::math::policies::errno_on_error>,
        boost::math::policies::promote_double<false>>;

    static constexpr double LS_MIN = -10.0, LS_MAX = 10.0;  ///< Lower and upper clamp of \f$ \log\sigma \f$ (`params(k)`): \f$ -10 \f$ and \f$ 10 \f$.
    static constexpr double A_MIN  = -20.0, A_MAX  = 20.0;  ///< Lower and upper clamp of the skewness \f$ \alpha \f$ (`params(k+1)`): \f$ -20 \f$ and \f$ 20 \f$.
    static constexpr double LN_MIN = 0.1,   LN_MAX = 7.0;  ///< Lower and upper clamp of \f$ \log\nu \f$ (`params(k+2)`): \f$ 0.1 \f$ and \f$ 7 \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$) of the mean model.
     * @param endog Real-valued response vector \f$ y \f$ (length \f$ n \f$).
     * @param link Link type whose inverse maps \f$ \eta \f$ to the mean \f$ \mu \f$ (default `LinkType::Identity`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor.
     */
    SkewTLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, LinkType link = LinkType::Identity)
        : X(exog), y(endog), mu_link(link) {}

    static constexpr double PEN_W = 100.0;  ///< Weight \f$ W = 100 \f$ of the quadratic bound penalty.
    /**
     * @brief Quadratic exterior penalty for a parameter leaving its admissible interval.
     * @param v Raw (unclamped) parameter value.
     * @param lo Lower bound of the interval.
     * @param hi Upper bound of the interval.
     * @return Penalty \f$ \tfrac12 W e^2 \f$ with \f$ W = \f$ `PEN_W` and \f$ e = \max(lo - v,\, v - hi,\, 0) \f$ (0 inside the interval).
     */
    static inline double pen_val(double v, double lo, double hi) {
        const double e = v < lo ? lo - v : (v > hi ? v - hi : 0.0);
        return 0.5 * PEN_W * e * e;
    }
    /**
     * @brief Derivative of `pen_val()` with respect to the raw parameter.
     * @param v Raw (unclamped) parameter value.
     * @param lo Lower bound of the interval.
     * @param hi Upper bound of the interval.
     * @return `PEN_W * (v - lo)` if \f$ v < lo \f$, `PEN_W * (v - hi)` if \f$ v > hi \f$, and 0 otherwise.
     */
    static inline double pen_grad(double v, double lo, double hi) {
        return v < lo ? PEN_W * (v - lo) : (v > hi ? PEN_W * (v - hi) : 0.0);
    }

    /**
     * @brief Clamps a scalar to an interval.
     * @param v Value to clamp.
     * @param lo Lower bound.
     * @param hi Upper bound.
     * @return The value `v` limited to `[lo, hi]` (a NaN input is returned unchanged).
     */
    static inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

    /**
     * @brief Log-gamma function \f$ \log\Gamma(x) \f$ via `boost::math::lgamma` with policy `Pol`.
     * @param x Argument.
     * @return Value \f$ \log\Gamma(x) \f$ (non-finite on error, see `Pol`).
     */
    static inline double log_gamma(double x) { return boost::math::lgamma(x, Pol()); }
    /**
     * @brief Digamma function \f$ \psi(x) = \Gamma'(x)/\Gamma(x) \f$ via `boost::math::digamma` with policy `Pol`.
     * @param x Argument.
     * @return Value \f$ \psi(x) \f$ (non-finite on error, see `Pol`).
     */
    static inline double digamma(double x)   { return boost::math::digamma(x, Pol()); }

    /**
     * @brief Log-density of the standard Student-\f$ t \f$ distribution.
     * @param x Evaluation point.
     * @param nu Degrees of freedom \f$ \nu > 0 \f$.
     * @return Log-density \f$ \log\Gamma\big(\tfrac{\nu+1}{2}\big) - \log\Gamma\big(\tfrac{\nu}{2}\big) - \tfrac12\log(\pi\nu)
     *         - \tfrac{\nu+1}{2}\log\big(1 + x^2/\nu\big) \f$, using `log1p` for accuracy.
     */
    static inline double log_t_pdf(double x, double nu) {
        double term1 = log_gamma((nu + 1.0) / 2.0) - log_gamma(nu / 2.0);
        double term2 = 0.5 * std::log(M_PI * nu);
        double term3 = -((nu + 1.0) / 2.0) * std::log1p((x * x) / nu);
        return term1 - term2 + term3;
    }

    /**
     * @brief Continued fraction for the regularised incomplete beta function (modified Lentz algorithm).
     * @details Evaluates the continued fraction \f$ \mathrm{cf}(a,b,x) \f$ such that
     * \f[ I_x(a,b) = \frac{x^a (1-x)^b}{a\,B(a,b)}\;\mathrm{cf}(a,b,x), \f]
     * iterating at most `MAXIT = 500` times and stopping when the correction factor is within machine epsilon of 1.
     * Intermediate values with magnitude below `FPMIN` (`min()/eps`) are replaced by `FPMIN`. It converges quickly for
     * \f$ x < (a+1)/(a+b+2) \f$.
     * @param a First shape parameter, \f$ a > 0 \f$.
     * @param b Second shape parameter, \f$ b > 0 \f$.
     * @param x Argument in \f$ (0,1) \f$.
     * @return The continued-fraction value \f$ \mathrm{cf}(a,b,x) \f$.
     */
    static inline double betacf(double a, double b, double x) {
        const int MAXIT = 500;
        const double EPS = std::numeric_limits<double>::epsilon();
        const double FPMIN = std::numeric_limits<double>::min() / EPS;
        double qab = a + b, qap = a + 1.0, qam = a - 1.0, c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (std::abs(d) < FPMIN) d = FPMIN;
        d = 1.0 / d; double h = d;
        for (int m = 1; m <= MAXIT; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
            c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
            d = 1.0 / d; h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
            c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
            d = 1.0 / d; double del = d * c; h *= del;
            if (std::abs(del - 1.0) <= EPS) break;
        }
        return h;
    }

    /**
     * @brief Tail-accurate log-CDF of the Student-\f$ t \f$ distribution based on `betacf()`.
     * @details With \f$ a = \nu/2 \f$, \f$ b = 1/2 \f$, \f$ v = \nu/(\nu+x^2) \f$ one has
     * \f$ I_v(a,b) = P(|T_\nu| > |x|) \f$. The code evaluates \f$ \log I_v(a,b) \f$ with the continued fraction directly if
     * \f$ v < (a+1)/(a+b+2) \f$, otherwise through the symmetry \f$ I_v(a,b) = 1 - I_{1-v}(b,a) \f$ and `log1p`.
     * The result is \f$ \log\tfrac12 + \log I_v(a,b) \f$ for \f$ x < 0 \f$ (lower tail) and
     * \f$ \log\{1 - \tfrac12 I_v(a,b)\} \f$ for \f$ x \ge 0 \f$.
     * @param x Evaluation point (finite and non-zero; \f$ x = 0 \f$ would give \f$ \log 0 \f$ inside the prefactor).
     * @param nu Degrees of freedom \f$ \nu > 0 \f$.
     * @return Value \f$ \log T_\nu(x) \f$.
     */
    static inline double log_T_cdf_tail(double x, double nu) {
        const double a = 0.5 * nu, b = 0.5;
        const double x2  = x * x;
        const double v   = nu / (nu + x2);
        const double omv = x2 / (nu + x2);

        const double log_bt = log_gamma(a + b) - log_gamma(a) - log_gamma(b)
                            + a * std::log(v) + b * std::log(omv);
        double log_I;
        if (v < (a + 1.0) / (a + b + 2.0))
            log_I = log_bt + std::log(betacf(a, b, v)) - std::log(a);
        else
            log_I = std::log1p(-std::exp(log_bt + std::log(betacf(b, a, omv)) - std::log(b)));

        return (x < 0.0) ? std::log(0.5) + log_I
                         : std::log1p(-0.5 * std::exp(log_I));
    }

    /**
     * @brief Log-CDF \f$ \log T_\nu(x) \f$ of the standard Student-\f$ t \f$ distribution.
     * @details Special cases: NaN returns NaN, \f$ x = -\infty \f$ returns \f$ -\infty \f$, \f$ x = +\infty \f$ returns 0, and
     * \f$ x = 0 \f$ returns \f$ \log\tfrac12 \f$. Otherwise \f$ I = I_v(\nu/2, 1/2) = P(|T_\nu|>|x|) \f$ with
     * \f$ v = \nu/(\nu+x^2) \f$ is computed with `boost::math::ibeta` if \f$ x^2 > \nu \f$, else with `boost::math::ibetac`
     * (complementary form, better conditioned near the centre), and
     * \f[ \log T_\nu(x) = \begin{cases} \log(1 - I/2), & x > 0, \\ \log(I/2), & x < 0. \end{cases} \f]
     * `log_T_cdf_tail()` is used when `I` is not finite or when \f$ x < 0 \f$ and \f$ I \le 10^{-290} \f$
     * (avoiding underflow of the lower tail).
     * @param x Evaluation point.
     * @param nu Degrees of freedom \f$ \nu > 0 \f$.
     * @return Value \f$ \log T_\nu(x) \f$.
     */
    static inline double log_T_cdf(double x, double nu) {
        if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
        if (std::isinf(x)) return (x < 0.0) ? -std::numeric_limits<double>::infinity() : 0.0;
        if (x == 0.0) return std::log(0.5);

        const double a = 0.5 * nu, x2 = x * x;
        const double v = nu / (nu + x2), omv = x2 / (nu + x2);

        const double I = (x2 > nu) ? boost::math::ibeta(a, 0.5, v, Pol())
                                   : boost::math::ibetac(0.5, a, omv, Pol());
        if (!std::isfinite(I)) return log_T_cdf_tail(x, nu);

        if (x > 0.0) return std::log1p(-0.5 * I);
        if (I > 1e-290) return std::log(0.5 * I);
        return log_T_cdf_tail(x, nu);
    }

    /**
     * @brief Gamma-function ratio appearing in the skew-\f$ t \f$ mean.
     * @param nu Degrees of freedom \f$ \nu \f$ (meaningful for \f$ \nu > 1 \f$).
     * @return Ratio \f$ \Gamma((\nu-1)/2)/\Gamma(\nu/2) \f$, computed as `boost::math::tgamma_delta_ratio(0.5*(nu-1), 0.5, Pol())`.
     */
    static inline double gamma_ratio_fn(double nu) {
        return boost::math::tgamma_delta_ratio(0.5 * (nu - 1.0), 0.5, Pol());
    }

    /**
     * @brief Shift between the mean and the location of the skew-\f$ t \f$ distribution.
     * @param sigma Scale \f$ \sigma > 0 \f$.
     * @param alpha Skewness \f$ \alpha \f$.
     * @param nu Degrees of freedom \f$ \nu \f$.
     * @return Shift \f$ c = \sigma\,\delta\sqrt{\nu/\pi}\,\Gamma((\nu-1)/2)/\Gamma(\nu/2) \f$ with
     *         \f$ \delta = \alpha/\sqrt{1+\alpha^2} \f$, or 0 if \f$ \nu \le 1 \f$ (the mean does not exist).
     *         The location is \f$ \xi = \mu - c \f$.
     */
    inline double get_mean_correction(double sigma, double alpha, double nu) const {
        if (nu <= 1.0) return 0.0;
        double delta = alpha / std::sqrt(1.0 + alpha * alpha);
        return sigma * delta * std::sqrt(nu / M_PI) * gamma_ratio_fn(nu);
    }

    /**
     * @brief Per-observation NLL without the \f$ \log\sigma - \log 2 \f$ terms.
     * @details Computes \f$ \mu = g^{-1}(\eta) \f$ with `evaluate_link`, \f$ z = (y - \mu + c)/\sigma \f$ and returns
     * \f[ -\big[\log t_\nu(z) + \log T_{\nu+1}(\alpha z \sqrt{(\nu+1)/(\nu+z^2)})\big]. \f]
     * @param eta_i Linear predictor \f$ \eta_i \f$ of the observation.
     * @param y_i Observed response \f$ y_i \f$.
     * @param sigma Scale \f$ \sigma \f$ (already exponentiated and clamped).
     * @param alpha Skewness \f$ \alpha \f$ (already clamped).
     * @param nu Degrees of freedom \f$ \nu \f$ (already exponentiated and clamped).
     * @param mean_correction Shift \f$ c \f$ returned by `get_mean_correction()` for the same \f$ (\sigma,\alpha,\nu) \f$.
     * @return The value above (may be non-finite; the caller replaces it by `1e10`).
     */
    double evaluate_single_nll(double eta_i, double y_i, double sigma, double alpha, double nu,
                               double mean_correction) const {
        double mu_base = 0.0; double dummy = 0.0;
        evaluate_link(eta_i, mu_link, mu_base, dummy);

        double mu_adjusted = mu_base - mean_correction;
        double z = (y_i - mu_adjusted) / sigma;
        double nu_plus_1 = nu + 1.0;
        double rescale = std::sqrt(nu_plus_1 / (nu + z * z));
        double argument_cdf = alpha * z * rescale;

        return -(log_t_pdf(z, nu) + log_T_cdf(argument_cdf, nu_plus_1));
    }

    /**
     * @brief Evaluates the total negative log-likelihood including the bound penalty.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma, \alpha, \log\nu) \f$ of length \f$ k+3 \f$.
     * @return Total \f$ n(\log\sigma - \log 2) + \sum_i \mathrm{NLL}_i + \text{penalty} \f$, evaluated at the clamped
     *         \f$ (\log\sigma, \alpha, \log\nu) \f$, where the penalty is the sum of three `pen_val()` terms on the raw
     *         values. Rows whose NLL is not finite contribute `1e10`; there is no final fallback for the whole sum.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = clampd(params(k),     LS_MIN, LS_MAX);
        double alpha     = clampd(params(k + 1), A_MIN,  A_MAX);
        double log_nu    = clampd(params(k + 2), LN_MIN, LN_MAX);

        double sigma = std::exp(log_sigma);
        double nu    = std::exp(log_nu);
        double mean_correction = get_mean_correction(sigma, alpha, nu);

        // linear predictors; the log-sigma/alpha/log-nu clamps were applied above
        Eigen::VectorXd eta = X * beta;
        double total_neg_ll = static_cast<double>(n) * log_sigma - static_cast<double>(n) * 0.6931471805599453;

        // accumulate per-observation terms (non-finite rows count as 1e10)
        for (long i = 0; i < n; ++i) {
            double row_nll = evaluate_single_nll(eta(i), y(i), sigma, alpha, nu, mean_correction);
            total_neg_ll += std::isfinite(row_nll) ? row_nll : 1e10;
        }
        total_neg_ll += pen_val(params(k),     LS_MIN, LS_MAX)
                      + pen_val(params(k + 1), A_MIN,  A_MAX)
                      + pen_val(params(k + 2), LN_MIN, LN_MAX);
        return total_neg_ll;
    }

    /**
     * @brief Student-\f$ t \f$ density \f$ t_\nu(x) = \exp(\log t_\nu(x)) \f$.
     * @param x Evaluation point.
     * @param nu Degrees of freedom.
     * @return The density value (not used by the loss itself).
     */
    static inline double t_pdf_value(double x, double nu) { return std::exp(log_t_pdf(x, nu)); }
    /**
     * @brief Student-\f$ t \f$ CDF \f$ T_\nu(x) = \exp(\log T_\nu(x)) \f$ via `log_T_cdf()`.
     * @param x Evaluation point.
     * @param nu Degrees of freedom.
     * @return The CDF value (not used by the loss itself).
     */
    static inline double t_cdf_value(double x, double nu) { return std::exp(log_T_cdf(x, nu)); }

    /**
     * @brief Student-\f$ t \f$ Mills-type ratio \f$ t_\nu(x)/T_\nu(x) \f$, i.e. \f$ \partial \log T_\nu(x)/\partial x \f$.
     * @param x Evaluation point.
     * @param nu Degrees of freedom of both density and CDF.
     * @return Ratio \f$ \exp\{\log t_\nu(x) - \log T_\nu(x)\} \f$, evaluated in log scale for stability.
     */
    static inline double t_mills(double x, double nu) {
        return std::exp(log_t_pdf(x, nu) - log_T_cdf(x, nu));
    }

    /**
     * @brief Gradient of the negative log-likelihood (analytical, except the degrees-of-freedom term of the CDF).
     * @details With \f$ M_i = t_{\nu+1}(a_i)/T_{\nu+1}(a_i) \f$, \f$ a_i = \alpha z_i \rho_i \f$,
     * \f$ \rho_i = \sqrt{(\nu+1)/(\nu+z_i^2)} \f$ and
     * \f[ \frac{\partial\,\mathrm{NLL}_i}{\partial z_i} = \frac{(\nu+1) z_i}{\nu+z_i^2} - M_i\,
     *     \frac{\alpha\,\nu\sqrt{\nu+1}}{(\nu+z_i^2)^{3/2}}, \f]
     * the per-observation derivatives are
     * - \f$ \partial/\partial\eta_i = -\tfrac1\sigma\,(\partial\mathrm{NLL}_i/\partial z_i)\,d\mu_i/d\eta_i \f$;
     * - \f$ \partial/\partial\log\sigma = (\partial\mathrm{NLL}_i/\partial z_i)\,(C - z_i) \f$ plus the constant 1 from
     *   \f$ \log\sigma \f$, where \f$ C = c/\sigma = \delta\sqrt{\nu/\pi}\,\Gamma((\nu-1)/2)/\Gamma(\nu/2) \f$;
     * - \f$ \partial/\partial\alpha = (\partial\mathrm{NLL}_i/\partial z_i)\,\partial C/\partial\alpha - M_i z_i\rho_i \f$,
     *   with \f$ \partial C/\partial\alpha = \sqrt{\nu/\pi}\,\Gamma((\nu-1)/2)/\Gamma(\nu/2)\,(1+\alpha^2)^{-3/2} \f$;
     * - \f$ \partial/\partial\log\nu = \nu\,[\,\partial_\nu(-\log t_\nu(z_i)) + (\partial\mathrm{NLL}_i/\partial z_i)\,
     *   \partial C/\partial\nu - M_i\,\partial_\nu a_i - \partial_{\mathrm{df}}\log T(a_i)\,] \f$ with
     *   \f$ \partial C/\partial\nu = C\,[\tfrac{1}{2\nu} + \tfrac12(\psi(\tfrac{\nu-1}{2}) - \psi(\tfrac{\nu}{2}))] \f$
     *   and \f$ \partial_{\mathrm{df}}\log T \f$ the central difference
     *   \f$ [\log T_{\nu+1+h}(a_i) - \log T_{\nu+1-h}(a_i)]/(2h) \f$, \f$ h = 10^{-5} \f$.
     *
     * The regression block is \f$ X^\top (\partial\mathrm{NLL}/\partial\boldsymbol\eta) \f$ and the derivative of the
     * bound penalty (`pen_grad()`) is added to the three scalar components.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma, \alpha, \log\nu) \f$ of length \f$ k+3 \f$.
     * @return Vector of length \f$ k+3 \f$. An observation with any non-finite contribution is skipped (see class notes).
     *         The likelihood part of the \f$ \log\sigma \f$, \f$ \alpha \f$ and \f$ \log\nu \f$ components is 0 when that
     *         parameter is not strictly inside its clamp interval, and non-finite totals are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        // a parameter has likelihood gradient only if strictly inside its clamp interval
        const bool ls_free = params(k)     > LS_MIN && params(k)     < LS_MAX;
        const bool a_free  = params(k + 1) > A_MIN  && params(k + 1) < A_MAX;
        const bool nu_free = params(k + 2) > LN_MIN && params(k + 2) < LN_MAX;

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = clampd(params(k),     LS_MIN, LS_MAX);
        double alpha     = clampd(params(k + 1), A_MIN,  A_MAX);
        double log_nu    = clampd(params(k + 2), LN_MIN, LN_MAX);

        const double sigma = std::exp(log_sigma);
        const double nu    = std::exp(log_nu);
        const double nu1   = nu + 1.0;
        const double sqrt_nu1 = std::sqrt(nu1);

        const double sqrt_1pa2 = std::sqrt(1.0 + alpha * alpha);
        const double delta = alpha / sqrt_1pa2;
        const bool has_mean = nu > 1.0;

        const double sqrt_nu_pi = std::sqrt(nu / M_PI);
        const double gamma_ratio = has_mean ? gamma_ratio_fn(nu) : 0.0;
        const double C_skewt = delta * sqrt_nu_pi * gamma_ratio;
        const double mean_correction = sigma * C_skewt;

        const double dC_dalpha = sqrt_nu_pi * gamma_ratio / (sqrt_1pa2 * sqrt_1pa2 * sqrt_1pa2);

        const double psi_half_nu1 = digamma(0.5 * nu1);
        const double psi_half_nu  = digamma(0.5 * nu);
        double dC_dnu = 0.0;
        if (has_mean) {
            const double psi_half_nm1 = digamma(0.5 * (nu - 1.0));
            dC_dnu = C_skewt * (0.5 / nu + 0.5 * (psi_half_nm1 - psi_half_nu));
        }
        const double dtpdf_dnu_const = -0.5 * (psi_half_nu1 - psi_half_nu) + 0.5 / nu;
        const double eps_df = 1e-5;

        // linear predictors and accumulators
        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlogsigma = static_cast<double>(n);
        double dNLL_dalpha = 0.0;
        double dNLL_dlognu = 0.0;

        // per-observation derivatives w.r.t. eta, log(sigma), alpha, log(nu)
        for (long i = 0; i < n; ++i) {
            double mu_base = 0.0, dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_base, dmu_deta_i);

            const double mu_adj = mu_base - mean_correction;
            const double z  = (y(i) - mu_adj) / sigma;
            const double z2 = z * z;
            const double denom = nu + z2;
            const double sqrt_denom = std::sqrt(denom);

            const double sqrt_ratio = sqrt_nu1 / sqrt_denom;
            const double arg = alpha * z * sqrt_ratio;
            const double M_i = t_mills(arg, nu1);

            const double darg_dz = alpha * nu * sqrt_nu1 / (denom * sqrt_denom);
            const double dNLLtpdf_dz = nu1 * z / denom;
            const double dNLL_dz = dNLLtpdf_dz - M_i * darg_dz;

            const double d_eta = -dNLL_dz / sigma * dmu_deta_i;

            const double d_ls = dNLL_dz * (-(z - C_skewt));

            const double d_a = dNLL_dz * dC_dalpha - M_i * z * sqrt_ratio;

            const double dNLLtpdf_dnu = dtpdf_dnu_const
                                      + 0.5 * std::log1p(z2 / nu)
                                      - 0.5 * nu1 * z2 / (nu * denom);
            const double darg_dnu_partial = alpha * z * (z2 - 1.0) / (2.0 * denom * sqrt_nu1 * sqrt_denom);
            const double dlogT_ddf = (log_T_cdf(arg, nu1 + eps_df) - log_T_cdf(arg, nu1 - eps_df))
                                     / (2.0 * eps_df);
            const double dnu_i = dNLLtpdf_dnu + dNLL_dz * dC_dnu
                               - M_i * darg_dnu_partial - dlogT_ddf;
            const double d_ln = dnu_i * nu;

            if (!(std::isfinite(d_eta) && std::isfinite(d_ls) &&
                  std::isfinite(d_a)   && std::isfinite(d_ln))) continue;

            dNLL_deta(i)    = d_eta;
            dNLL_dlogsigma += d_ls;
            dNLL_dalpha    += d_a;
            dNLL_dlognu    += d_ln;
        }

        // assemble gradient and add bound-penalty derivatives
        Eigen::VectorXd grad(k + 3);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k)      = ((ls_free && std::isfinite(dNLL_dlogsigma)) ? dNLL_dlogsigma : 0.0)
                       + pen_grad(params(k),     LS_MIN, LS_MAX);
        grad(k + 1)  = ((a_free  && std::isfinite(dNLL_dalpha))    ? dNLL_dalpha    : 0.0)
                       + pen_grad(params(k + 1), A_MIN,  A_MAX);
        grad(k + 2)  = ((nu_free && std::isfinite(dNLL_dlognu))    ? dNLL_dlognu    : 0.0)
                       + pen_grad(params(k + 2), LN_MIN, LN_MAX);
        return grad;
    }

    /**
     * @brief Finite-difference Hessian of the NLL built from the analytical gradient.
     * @details Column \f$ j \f$ is \f$ [\nabla f(\theta + h_j e_j) - \nabla f(\theta - h_j e_j)]/(2h_j) \f$ with
     * \f$ h_j = 10^{-4}\max(1, |\theta_j|) \f$. The matrix is symmetrised as \f$ (H+H^\top)/2 \f$ and non-finite entries are
     * set to 0. Because it differentiates `gradient()`, it inherits the finite-difference degrees-of-freedom term, the
     * clamp-dependent masking and the penalty derivative (curvature \f$ W = 100 \f$ outside the intervals).
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma, \alpha, \log\nu) \f$.
     * @return Symmetric \f$ (k+3)\times(k+3) \f$ matrix; costs \f$ 2(k+3) \f$ gradient evaluations.
     */
    Eigen::MatrixXd hessian_raw(const Eigen::VectorXd& params) const {
        const long p = params.size();
        Eigen::MatrixXd H(p, p);
        for (long j = 0; j < p; ++j) {
            const double h = 1e-4 * std::max(1.0, std::abs(params(j)));
            Eigen::VectorXd pp = params, pm = params;
            pp(j) += h; pm(j) -= h;
            H.col(j) = (gradient(pp) - gradient(pm)) / (2.0 * h);
        }
        H = 0.5 * (H + H.transpose());
        return H.array().isFinite().select(H, 0.0);
    }

    /**
     * @brief Hessian of the NLL used by the optimiser; identical to `hessian_raw()`.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma, \alpha, \log\nu) \f$.
     * @return Symmetric \f$ (k+3)\times(k+3) \f$ finite-difference Hessian, without ridge or PSD projection.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const { return hessian_raw(params); }

    /**
     * @brief Positive-definite version of the finite-difference Hessian.
     * @details Eigen-decomposes `hessian_raw()` with `SelfAdjointEigenSolver`, floors every eigenvalue at `1e-8` and
     * reassembles \f$ V\,\mathrm{diag}(\max(\lambda_j, 10^{-8}))\,V^\top \f$. The solver status is not checked.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma, \alpha, \log\nu) \f$.
     * @return Symmetric positive-definite \f$ (k+3)\times(k+3) \f$ matrix.
     */
    Eigen::MatrixXd hessian_psd(const Eigen::VectorXd& params) const {
        Eigen::MatrixXd H = hessian_raw(params);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        Eigen::VectorXd ev = es.eigenvalues().cwiseMax(1e-8);
        return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }

    /**
     * @brief Diagnostic: maximal relative discrepancy between `gradient()` and finite differences of the objective.
     * @details For each component \f$ j \f$ a central difference of `operator()` with step
     * \f$ h\max(1,|\theta_j|) \f$ gives \f$ \mathrm{fd}_j \f$ and the error is
     * \f$ |g_j - \mathrm{fd}_j| / \max(1, |\mathrm{fd}_j|) \f$.
     * @param params Parameter vector at which the gradient is checked.
     * @param h Relative finite-difference step (default `1e-6`).
     * @return The maximum of the relative errors over all components.
     */
    double check_gradient(const Eigen::VectorXd& params, double h = 1e-6) const {
        const Eigen::VectorXd g = gradient(params);
        double max_err = 0.0;
        for (long j = 0; j < params.size(); ++j) {
            const double step = h * std::max(1.0, std::abs(params(j)));
            Eigen::VectorXd pp = params, pm = params;
            pp(j) += step; pm(j) -= step;
            const double fd = ((*this)(pp) - (*this)(pm)) / (2.0 * step);
            max_err = std::max(max_err, std::abs(g(j) - fd) / std::max(1.0, std::abs(fd)));
        }
        return max_err;
    }
};

/**
 * @brief Negative log-likelihood of a linear regression model with logistic errors (location-scale logistic distribution).
 *
 * @details Models a real-valued response \f$ y_i \in \mathbb{R} \f$ with density
 * \f[ f(y_i) = \frac{1}{\sigma}\,\frac{e^{-z_i}}{(1+e^{-z_i})^2}, \qquad z_i = \frac{y_i - \mu_i}{\sigma}, \qquad
 *     \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta), \qquad \sigma = e^{\log\sigma}. \f]
 * The linear predictor is mapped to the **location (= mean = median)** \f$ \mu_i \f$ by the inverse link
 * (`evaluate_link`, default `Identity`); \f$ \sigma \f$ is a common scale (the standard deviation is
 * \f$ \sigma\pi/\sqrt3 \f$).
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                          | Scale                                         |
 * |------------|--------------------------------------------------|-----------------------------------------------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor scale                        |
 * | `k`        | \f$ \log\sigma \f$                               | log; clamped to \f$ [-10, 10] \f$ before use  |
 *
 * where \f$ k \f$ is the number of columns of the design matrix.
 *
 * ### Log-likelihood
 * \f[ \mathrm{NLL} = \sum_{i=1}^n \Big[ \log\sigma + z_i + 2\log\big(1 + e^{-z_i}\big) \Big], \f]
 * evaluated in the overflow-free form \f$ |z_i| + 2\log1p(e^{-|z_i|}) \f$ (the two branches `z >= 0` and `z < 0` of the
 * code). No constant is dropped; all observations have unit weight.
 *
 * ### Derivatives
 * With \f$ F(z) = 1/(1+e^{-z}) \f$ (the logistic CDF, evaluated in a stable form) the per-observation derivatives are
 * \f$ \partial\mathrm{NLL}_i/\partial z_i = 2F(z_i) - 1 \f$ and \f$ \partial^2\mathrm{NLL}_i/\partial z_i^2 = 2F(1-F) \f$:
 * \f[ \frac{\partial\mathrm{NLL}_i}{\partial\eta_i} = -\frac{2F(z_i)-1}{\sigma}\frac{d\mu_i}{d\eta_i}, \qquad
 *     \frac{\partial\mathrm{NLL}_i}{\partial\log\sigma} = 1 - z_i\,(2F(z_i)-1). \f]
 * The Hessian is exact (observed information, including the link-curvature term) and is described in `hessian()`.
 *
 * ### Numerical safeguards
 * - \f$ \log\sigma \f$ is clamped to \f$ [-10, 10] \f$ in the objective, gradient and Hessian (silent clamp, no
 *   penalty); the derivatives are those of the unclamped formulas evaluated at the clamped value.
 * - The objective returns `1e10` if the sum is not finite.
 * - The Hessian replaces non-finite entries by 0; the gradient is not masked.
 *
 * @note The object stores **non-owning references** to the design matrix and the response: they must outlive the
 *       functor. No validation is performed.
 * @see GumbelLoss, SkewTLoss, StudentTLoss for other real-valued location-scale families.
 *
 * @ingroup glm_ll_location_scale
 */
struct LogisticLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the location model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Real-valued response vector (length \f$ n \f$).
    LinkType mu_link;  ///< Link type whose inverse maps \f$ \eta_i \f$ to the location \f$ \mu_i \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Real-valued response vector \f$ y \f$ (length \f$ n \f$).
     * @param link Link type whose inverse maps \f$ \eta \f$ to the location \f$ \mu \f$ (default `LinkType::Identity`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor.
     */
    LogisticLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, LinkType link = LinkType::Identity)
        : X(exog), y(endog), mu_link(link) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ n\log\sigma + \sum_i [\,|z_i| + 2\log1p(e^{-|z_i|})\,] \f$, i.e. the NLL above, or
     *         `1e10` if it is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double sigma = std::exp(log_sigma);

        // linear predictors; n*log(sigma) is the scale term of all rows
        Eigen::VectorXd eta = X * beta;
        double total_neg_ll = static_cast<double>(n) * log_sigma;

        // accumulate per-observation terms in an overflow-free form
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double diff = y(i) - mu_i;
            double z = diff / sigma;

            if (z >= 0.0) {
                total_neg_ll += z + 2.0 * std::log1p(std::exp(-z));
            } else {
                total_neg_ll += -z + 2.0 * std::log1p(std::exp(z));
            }
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Vector of length \f$ k+1 \f$: \f$ \sum_i \partial\mathrm{NLL}_i/\partial\eta_i\,\mathbf{x}_i \f$ followed by
     *         \f$ \sum_i (1 - z_i\,[2F(z_i)-1]) \f$. Not masked for non-finite values.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlog_sigma = static_cast<double>(n);

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double diff = y(i) - mu_i;
            double z = diff / sigma;

            double F_z = (z >= 0.0) ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
            double dNLL_dz = 2.0 * F_z - 1.0;

            double dNLL_dmu = -dNLL_dz / sigma;
            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;
            dNLL_dlog_sigma -= z * dNLL_dz;
        }

        Eigen::VectorXd grad(k + 1);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k)      = dNLL_dlog_sigma;
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @details With \f$ g = 2F(z)-1 \f$ and \f$ g' = 2F(1-F) \f$ (derivatives with respect to \f$ z \f$):
     * \f[ H = \begin{pmatrix} X^\top \mathrm{diag}(w_{\mu\mu})\,X & X^\top w_{\mu s} \\ w_{\mu s}^\top X & h_{ss} \end{pmatrix}, \f]
     * \f[ w_{\mu\mu,i} = \frac{g'}{\sigma^2}\Big(\frac{d\mu_i}{d\eta_i}\Big)^2 - \frac{g}{\sigma}\frac{d^2\mu_i}{d\eta_i^2},
     *     \quad
     *     w_{\mu s,i} = \frac{g + z_i g'}{\sigma}\frac{d\mu_i}{d\eta_i},
     *     \quad
     *     h_{ss} = \sum_i \big( z_i g + z_i^2 g' \big). \f]
     * The second derivative of the inverse link comes from `evaluate_link_d2`.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Symmetric \f$ (k+1)\times(k+1) \f$ matrix; non-finite entries are replaced by 0. No ridge is added and the
     *         matrix is not projected onto the positive-definite cone.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd W_mumu = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_musigma = Eigen::VectorXd::Zero(n);
        double d2NLL_dlogsigma2 = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0; double d2mu_deta2_i = 0.0;
            evaluate_link_d2(eta(i), mu_link, mu_i, dmu_deta_i, d2mu_deta2_i);

            double diff = y(i) - mu_i;
            double z = diff / sigma;

            double F_z = (z >= 0.0) ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
            double dNLL_dz = 2.0 * F_z - 1.0;
            double d2NLL_dz2 = 2.0 * F_z * (1.0 - F_z);

            double dNLL_dmu = -dNLL_dz / sigma;
            double d2NLL_dmu2 = d2NLL_dz2 / (sigma * sigma);
            double d2NLL_dmudlogsigma = (dNLL_dz + z * d2NLL_dz2) / sigma;

            W_mumu(i) = d2NLL_dmu2 * dmu_deta_i * dmu_deta_i + dNLL_dmu * d2mu_deta2_i;
            W_musigma(i) = d2NLL_dmudlogsigma * dmu_deta_i;

            d2NLL_dlogsigma2 += z * dNLL_dz + z * z * d2NLL_dz2;
        }

        // assemble Hessian blocks (beta-beta, beta-log sigma, log sigma-log sigma)
        Eigen::MatrixXd H(k + 1, k + 1);
        H.block(0, 0, k, k) = X.transpose() * W_mumu.asDiagonal() * X;
        Eigen::VectorXd H_beta_sigma = X.transpose() * W_musigma;
        H.block(0, k, k, 1) = H_beta_sigma;
        H.block(k, 0, 1, k) = H_beta_sigma.transpose();
        H(k, k) = d2NLL_dlogsigma2;

        return H.array().isFinite().select(H, 0.0);
    }
};

/**
 * @brief Negative log-likelihood of a linear regression model with Gumbel (type-I extreme value, maximum) errors.
 *
 * @details Models a real-valued response \f$ y_i \in \mathbb{R} \f$ with density
 * \f[ f(y_i) = \frac{1}{\sigma}\exp\!\big(-z_i - e^{-z_i}\big), \qquad z_i = \frac{y_i - \mu_i}{\sigma}, \qquad
 *     \mu_i = g^{-1}(\mathbf{x}_i^\top\boldsymbol\beta), \qquad \sigma = e^{\log\sigma}. \f]
 * The linear predictor is mapped to the **location (mode)** \f$ \mu_i \f$ by the inverse link (`evaluate_link`,
 * default `Identity`); note that the mean is \f$ \mu_i + \gamma_E\,\sigma \f$ (\f$ \gamma_E \f$ Euler's constant),
 * so \f$ \mu_i \f$ is not the mean. The distribution is right-skewed; \f$ \sigma \f$ is a common scale.
 *
 * ### Parameter vector layout
 * | Index      | Meaning                                          | Scale                                         |
 * |------------|--------------------------------------------------|-----------------------------------------------|
 * | `0 .. k-1` | regression coefficients \f$ \boldsymbol\beta \f$ | linear predictor scale                        |
 * | `k`        | \f$ \log\sigma \f$                               | log; clamped to \f$ [-10, 10] \f$ before use  |
 *
 * where \f$ k \f$ is the number of columns of the design matrix.
 *
 * ### Log-likelihood
 * With the standardised residual clamped to \f$ z_i \in [-40, 40] \f$,
 * \f[ \mathrm{NLL} = \sum_{i=1}^n \Big[ \log\sigma + z_i + e^{-z_i} \Big]. \f]
 * No constant is dropped; all observations have unit weight.
 *
 * ### Derivatives
 * Per observation \f$ \partial\mathrm{NLL}_i/\partial z_i = 1 - e^{-z_i} \f$ and
 * \f$ \partial^2\mathrm{NLL}_i/\partial z_i^2 = e^{-z_i} \f$, hence
 * \f[ \frac{\partial\mathrm{NLL}_i}{\partial\eta_i} = -\frac{1 - e^{-z_i}}{\sigma}\frac{d\mu_i}{d\eta_i}, \qquad
 *     \frac{\partial\mathrm{NLL}_i}{\partial\log\sigma} = 1 - z_i\,(1 - e^{-z_i}). \f]
 * The Hessian is exact (observed information, including the link-curvature term); see `hessian()`.
 *
 * ### Numerical safeguards
 * - \f$ \log\sigma \f$ is clamped to \f$ [-10, 10] \f$ and \f$ z_i \f$ to \f$ [-40, 40] \f$ in the objective, gradient
 *   and Hessian (silent clamps, no penalty). The derivative formulas are those of the unclamped function evaluated at
 *   the clamped values, so they are not the derivatives of the clamped function when \f$ |z_i| > 40 \f$.
 * - The objective returns `1e10` if the sum is not finite.
 * - Non-finite gradient components are replaced by 0; non-finite Hessian entries are replaced by 0.
 *
 * @note The object stores **non-owning references** to the design matrix and the response: they must outlive the
 *       functor. No validation is performed.
 * @see LogisticLoss, SkewTLoss for other real-valued location-scale families.
 *
 * @ingroup glm_ll_location_scale
 */
struct GumbelLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the location model (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Real-valued response vector (length \f$ n \f$).
    LinkType mu_link;  ///< Link type whose inverse maps \f$ \eta_i \f$ to the Gumbel location \f$ \mu_i \f$.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog Real-valued response vector \f$ y \f$ (length \f$ n \f$).
     * @param link Link type whose inverse maps \f$ \eta \f$ to the Gumbel location \f$ \mu \f$ (default `LinkType::Identity`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor.
     */
    GumbelLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, LinkType link = LinkType::Identity)
        : X(exog), y(endog), mu_link(link) {}

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$ of length \f$ k+1 \f$.
     * @return The value \f$ \sum_i [\log\sigma + z_i + e^{-z_i}] \f$ with clamped \f$ z_i \f$, or `1e10` if it is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double sigma = std::exp(log_sigma);

        // linear predictors; n*log(sigma) is the scale term of all rows
        Eigen::VectorXd eta = X * beta;
        double total_neg_ll = static_cast<double>(n) * log_sigma;

        // accumulate per-observation terms with clamped z
        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double z = (y(i) - mu_i) / sigma;
            z = std::max(-40.0, std::min(40.0, z));
            total_neg_ll += z + std::exp(-z);
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Vector of length \f$ k+1 \f$: \f$ \sum_i \partial\mathrm{NLL}_i/\partial\eta_i\,\mathbf{x}_i \f$ followed by
     *         \f$ \sum_i (1 - z_i\,[1 - e^{-z_i}]) \f$ (the clamped \f$ z_i \f$ is used). Non-finite components are
     *         replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd dNLL_deta = Eigen::VectorXd::Zero(n);
        double dNLL_dlogsigma = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0;
            evaluate_link(eta(i), mu_link, mu_i, dmu_deta_i);

            double z = (y(i) - mu_i) / sigma;
            z = std::max(-40.0, std::min(40.0, z));
            double exp_neg_z = std::exp(-z);

            double dNLL_dz = 1.0 - exp_neg_z;

            double dNLL_dmu = -dNLL_dz / sigma;
            dNLL_deta(i) = dNLL_dmu * dmu_deta_i;

            dNLL_dlogsigma += 1.0 - z * dNLL_dz;
        }

        Eigen::VectorXd grad(k + 1);
        grad.head(k) = X.transpose() * dNLL_deta;
        grad(k) = dNLL_dlogsigma;

        for (long i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) grad(i) = 0.0;
        }
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @details With \f$ g = 1 - e^{-z} \f$ and \f$ g' = e^{-z} \f$ (derivatives with respect to \f$ z \f$):
     * \f[ H = \begin{pmatrix} X^\top \mathrm{diag}(w_{\mu\mu})\,X & X^\top w_{\mu s} \\ w_{\mu s}^\top X & h_{ss} \end{pmatrix}, \f]
     * \f[ w_{\mu\mu,i} = \frac{g'}{\sigma^2}\Big(\frac{d\mu_i}{d\eta_i}\Big)^2 - \frac{g}{\sigma}\frac{d^2\mu_i}{d\eta_i^2},
     *     \quad
     *     w_{\mu s,i} = \frac{g + z_i g'}{\sigma}\frac{d\mu_i}{d\eta_i},
     *     \quad
     *     h_{ss} = \sum_i \big( z_i g + z_i^2 g' \big). \f]
     * The second derivative of the inverse link comes from `evaluate_link_d2`.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \log\sigma) \f$.
     * @return Symmetric \f$ (k+1)\times(k+1) \f$ matrix; non-finite entries are replaced by 0. No ridge is added and the
     *         matrix is not projected onto the positive-definite cone.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();

        Eigen::VectorXd beta = params.head(k);
        double log_sigma = std::max(-10.0, std::min(10.0, params(k)));
        double sigma = std::exp(log_sigma);

        Eigen::VectorXd eta = X * beta;
        Eigen::VectorXd W_mumu = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_musigma = Eigen::VectorXd::Zero(n);
        double d2NLL_dlogsigma2 = 0.0;

        for (long i = 0; i < n; ++i) {
            double mu_i = 0.0; double dmu_deta_i = 0.0; double d2mu_deta2_i = 0.0;
            evaluate_link_d2(eta(i), mu_link, mu_i, dmu_deta_i, d2mu_deta2_i);

            double z = (y(i) - mu_i) / sigma;
            z = std::max(-40.0, std::min(40.0, z));
            double exp_neg_z = std::exp(-z);

            double dNLL_dz = 1.0 - exp_neg_z;
            double d2NLL_dz2 = exp_neg_z;

            double dNLL_dmu = -dNLL_dz / sigma;

            double d2NLL_dmu2 = d2NLL_dz2 / (sigma * sigma);

            double d2NLL_dmudlogsigma = (dNLL_dz + z * d2NLL_dz2) / sigma;

            W_mumu(i) = d2NLL_dmu2 * dmu_deta_i * dmu_deta_i + dNLL_dmu * d2mu_deta2_i;
            W_musigma(i) = d2NLL_dmudlogsigma * dmu_deta_i;

            d2NLL_dlogsigma2 += z * dNLL_dz + z * z * d2NLL_dz2;
        }

        // assemble Hessian blocks (beta-beta, beta-log sigma, log sigma-log sigma)
        Eigen::MatrixXd H(k + 1, k + 1);
        H.block(0, 0, k, k) = X.transpose() * W_mumu.asDiagonal() * X;
        Eigen::VectorXd H_beta_sigma = X.transpose() * W_musigma;
        H.block(0, k, k, 1) = H_beta_sigma;
        H.block(k, 0, 1, k) = H_beta_sigma.transpose();
        H(k, k) = d2NLL_dlogsigma2;

        return H.array().isFinite().select(H, 0.0);
    }
};

/**
 * @brief Negative log-likelihood of the simplex-distribution regression model for responses in \f$ (0,1) \f$ with a regression on the dispersion.
 *
 * @details Models a proportion \f$ y_i \in (0,1) \f$ with the simplex density
 * \f[ f(y_i) = \big[2\pi\sigma_i^2\,\{y_i(1-y_i)\}^3\big]^{-1/2}\exp\!\Big( -\frac{d(y_i;\mu_i)}{2\sigma_i^2} \Big),
 *     \qquad d(y;\mu) = \frac{(y-\mu)^2}{y(1-y)\,\mu^2(1-\mu)^2}. \f]
 * Two linear predictors are used:
 * - the mean \f$ \mu_i = g_\mu^{-1}(\mathbf{x}_i^\top\boldsymbol\beta) \f$ with `mu_link` (default `Logit`), evaluated
 *   with `evaluate_inverse_link`;
 * - the **dispersion \f$ \sigma_i^2 \f$ itself** (not its logarithm or the standard deviation):
 *   \f$ \sigma_i^2 = g_\gamma^{-1}(\mathbf{z}_i^\top\boldsymbol\gamma) \f$ with `gamma_link` (default `Log`), evaluated by
 *   `eval_gamma_link()` (the `Inverse` link uses a smooth positive inverse instead of the plain one).
 *
 * Here \f$ \boldsymbol\gamma \f$ denotes the dispersion regression coefficients (it has nothing to do with the Gamma
 * distribution).
 *
 * ### Parameter vector layout
 * | Index                 | Meaning                                                   | Scale                                     |
 * |-----------------------|-----------------------------------------------------------|-------------------------------------------|
 * | `0 .. k_x-1`          | mean coefficients \f$ \boldsymbol\beta \f$ (`X.cols()` entries)       | linear predictor of `mu_link`     |
 * | `k_x .. k_x+k_z-1`    | dispersion coefficients \f$ \boldsymbol\gamma \f$ (`Z.cols()` entries) | linear predictor of `gamma_link` |
 *
 * There are no extra scalar parameters; the vector has length \f$ k_x + k_z \f$.
 *
 * ### Log-likelihood
 * With the clamped quantities \f$ \tilde y_i, \tilde\mu_i \in [10^{-6}, 1-10^{-6}] \f$ and
 * \f$ \tilde\sigma_i^2 = \max(\sigma_i^2, 10^{-6}) \f$,
 * \f$ D_y = \tilde y_i(1-\tilde y_i) \f$, \f$ D_\mu = \tilde\mu_i(1-\tilde\mu_i) \f$:
 * \f[ \ell_i = -\tfrac12\log(2\pi) - \tfrac12\log\tilde\sigma_i^2 - \tfrac32\log D_y
 *     - \frac{(\tilde y_i - \tilde\mu_i)^2}{2\tilde\sigma_i^2\,D_y\,D_\mu^2}, \qquad
 *     \mathrm{NLL} = -\sum_i \ell_i. \f]
 * No constant is dropped; all observations have unit weight; there is no censoring.
 *
 * ### Derivatives
 * With \f$ r = \tilde y - \tilde\mu \f$, \f$ s = \tilde\sigma^2 \f$ and \f$ D_i = r^2/(2 D_y D_\mu^2) \f$ (so that
 * \f$ \ell_i = \text{const} - \tfrac12\log s - \tfrac32\log D_y - D_i/s \f$):
 * \f[ \frac{\partial\mathrm{NLL}_i}{\partial\mu} = -\frac{r\,[D_\mu + r(1-2\mu)]}{s\,D_y\,D_\mu^3}, \qquad
 *     \frac{\partial\mathrm{NLL}_i}{\partial s} = \frac{1}{2s} - \frac{D_i}{s^2}, \f]
 * and the chain rule through \f$ d\mu/d\eta_\mu \f$ and \f$ ds/d\eta_\gamma \f$ gives the gradient
 * \f$ (X^\top u_\mu,\; Z^\top u_\gamma) \f$. The Hessian is the exact observed Hessian (including link curvature
 * terms) assembled from the 2x2 per-observation blocks, see `hessian()`. Derivatives are those of the unclamped formulas
 * evaluated at the clamped values \f$ \tilde y, \tilde\mu, \tilde\sigma^2 \f$ (the clamps are not differentiated).
 *
 * ### Numerical safeguards
 * - Clamps: \f$ y \f$ and \f$ \mu \f$ to \f$ [10^{-6}, 1-10^{-6}] \f$; \f$ \sigma^2 \f$ from below at `1e-6` (no upper
 *   bound).
 * - In the objective a non-finite log-density adds `1e8` (not `1e10`) to the NLL for that row; the total is not checked.
 * - In the gradient and the Hessian, non-finite per-observation weights are replaced by 0, and the assembled Hessian
 *   has remaining non-finite entries replaced by 0. No ridge and no PSD projection.
 *
 * @note The object stores **non-owning references** to `X`, `Z` and `y`: they must outlive the functor. No validation is
 *       performed; \f$ y_i \f$ outside \f$ (0,1) \f$ is silently clamped. `Z.cols()` must equal `params.size() - X.cols()`
 *       and `X`, `Z`, `y` must have the same number of rows.
 * @see BetaRegressionLoss for another model of proportions.
 *
 * @ingroup glm_ll_proportion
 */
struct SimplexLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix of the mean model (\f$ n \times k_x \f$).
    const Eigen::MatrixXd& Z;  ///< Design matrix of the dispersion model (\f$ n \times k_z \f$).
    const Eigen::VectorXd& y;  ///< Response vector of proportions in \f$ (0,1) \f$ (length \f$ n \f$).

    LinkType mu_link;  ///< Link type whose inverse maps \f$ \eta_\mu \f$ to the mean \f$ \mu \f$ (default `Logit`).
    LinkType gamma_link;  ///< Link type whose inverse maps \f$ \eta_\gamma \f$ to the dispersion \f$ \sigma^2 \f$ (default `Log`).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog_X Design matrix \f$ X \f$ of the mean (\f$ n \times k_x \f$).
     * @param exog_Z Design matrix \f$ Z \f$ of the dispersion \f$ \sigma^2 \f$ (\f$ n \times k_z \f$).
     * @param endog_y Response vector \f$ y \f$ (length \f$ n \f$) with values in \f$ (0,1) \f$.
     * @param link_mu Link type whose inverse maps \f$ \eta_\mu \f$ to \f$ \mu \in (0,1) \f$ (default `LinkType::Logit`).
     * @param link_gamma Link type whose inverse maps \f$ \eta_\gamma \f$ to \f$ \sigma^2 > 0 \f$ (default `LinkType::Log`).
     * @pre `exog_X`, `exog_Z` and `endog_y` have the same number of rows and outlive the functor.
     */
    SimplexLoss(const Eigen::MatrixXd& exog_X,
                const Eigen::MatrixXd& exog_Z,
                const Eigen::VectorXd& endog_y,
                LinkType link_mu = LinkType::Logit,
                LinkType link_gamma = LinkType::Log)
        : X(exog_X), Z(exog_Z), y(endog_y), mu_link(link_mu), gamma_link(link_gamma) {}

    /**
     * @brief Evaluates the inverse link of the dispersion with its first two derivatives.
     * @details For `LinkType::Inverse` it calls `evaluate_positive_inverse_d2()`, i.e.
     * \f$ \sigma^2 = 1/\eta_+ \f$ with \f$ \eta_+ = \tfrac12(\eta + \sqrt{\eta^2 + 0.04}) > 0 \f$, which keeps the dispersion
     * positive and smooth. Every other link is delegated to `evaluate_link_d2()` (e.g. `Log` gives \f$ e^{\min(\eta,23)} \f$).
     * @param eta Linear predictor value \f$ \eta_\gamma \f$.
     * @param link Link type.
     * @param v Output: inverse link value \f$ \sigma^2 \f$.
     * @param d1 Output: first derivative \f$ d\sigma^2/d\eta \f$.
     * @param d2 Output: second derivative \f$ d^2\sigma^2/d\eta^2 \f$.
     */
    static inline void eval_gamma_link(double eta, LinkType link, double& v, double& d1, double& d2) {
        if (link == LinkType::Inverse) {
            evaluate_positive_inverse_d2(eta, v, d1, d2);
            return;
        }
        ::glm::evaluate_link_d2(eta, link, v, d1, d2);
    }
    /**
     * @brief Vectorised dispersion inverse link.
     * @param eta Vector of linear predictors \f$ \eta_\gamma \f$ (length \f$ n \f$).
     * @return Vector of \f$ \sigma_i^2 \f$ values computed with `eval_gamma_link()`.
     */
    Eigen::VectorXd gamma_inverse_link(const Eigen::VectorXd& eta) const {
        Eigen::VectorXd out(eta.size());
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            double v = 0.0, d1 = 0.0, d2 = 0.0;
            eval_gamma_link(eta(i), gamma_link, v, d1, d2);
            out(i) = v;
        }
        return out;
    }
    /**
     * @brief Vectorised first derivative of the dispersion inverse link.
     * @param eta Vector of linear predictors \f$ \eta_\gamma \f$ (length \f$ n \f$).
     * @return Vector of \f$ d\sigma_i^2/d\eta_i \f$ values computed with `eval_gamma_link()`.
     */
    Eigen::VectorXd gamma_d_inverse_link(const Eigen::VectorXd& eta) const {
        Eigen::VectorXd out(eta.size());
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            double v = 0.0, d1 = 0.0, d2 = 0.0;
            eval_gamma_link(eta(i), gamma_link, v, d1, d2);
            out(i) = d1;
        }
        return out;
    }
    /**
     * @brief Scalar second derivative of the dispersion inverse link.
     * @param eta Linear predictor value \f$ \eta_\gamma \f$.
     * @return Value \f$ d^2\sigma^2/d\eta^2 \f$ computed with `eval_gamma_link()`.
     */
    double gamma_d2_inverse_link_scalar(double eta) const {
        double v = 0.0, d1 = 0.0, d2 = 0.0;
        eval_gamma_link(eta, gamma_link, v, d1, d2);
        return d2;
    }

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_x + k_z \f$
     *        (the dispersion block is taken as everything after the first `X.cols()` entries).
     * @return The value \f$ -\sum_i \ell_i \f$ with clamped \f$ \tilde y_i, \tilde\mu_i, \tilde\sigma_i^2 \f$; each row
     *         with non-finite log-density adds `1e8` instead.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_x = X.cols();

        Eigen::VectorXd beta = params.head(k_x);
        Eigen::VectorXd gamma = params.tail(params.size() - k_x);

        // linear predictors, mean and dispersion
        Eigen::VectorXd eta_mu = X * beta;
        Eigen::VectorXd eta_gamma = Z * gamma;

        Eigen::VectorXd mu = evaluate_inverse_link(eta_mu, mu_link);
        Eigen::VectorXd sigma_sq = gamma_inverse_link(eta_gamma);

        double total_neg_ll = 0.0;

        // accumulate per-observation log-densities (clamped y, mu, sigma^2)
        for (long i = 0; i < n; ++i) {
            double yi = std::max(1e-6, std::min(1.0 - 1e-6, y(i)));
            double mui = std::max(1e-6, std::min(1.0 - 1e-6, mu(i)));
            double s2i = std::max(1e-6, sigma_sq(i));

            double den_y = yi * (1.0 - yi);
            double den_mu = mui * (1.0 - mui);
            double diff = yi - mui;

            double log_density = -0.5 * std::log(2.0 * M_PI)
                                 - 0.5 * std::log(s2i)
                                 - 1.5 * std::log(den_y)
                                 - (diff * diff) / (2.0 * s2i * den_y * den_mu * den_mu);

            if (std::isfinite(log_density)) {
                total_neg_ll -= log_density;
            } else {
                total_neg_ll += 1e8;
            }
        }
        return total_neg_ll;
    }

    /**
     * @brief Exact analytical gradient of the negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_x + k_z \f$.
     * @return Vector of length \f$ k_x + k_z \f$: \f$ X^\top u_\mu \f$ followed by \f$ Z^\top u_\gamma \f$, where
     *         \f$ u_{\mu,i} = (\partial\mathrm{NLL}_i/\partial\mu_i)\,d\mu_i/d\eta_{\mu,i} \f$ and
     *         \f$ u_{\gamma,i} = (\partial\mathrm{NLL}_i/\partial\sigma_i^2)\,d\sigma_i^2/d\eta_{\gamma,i} \f$.
     *         Non-finite per-observation entries of \f$ u_\mu, u_\gamma \f$ are replaced by 0.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_x = X.cols();
        const long k_z = Z.cols();

        Eigen::VectorXd beta = params.head(k_x);
        Eigen::VectorXd gamma = params.tail(k_z);

        Eigen::VectorXd eta_mu = X * beta;
        Eigen::VectorXd eta_gamma = Z * gamma;

        Eigen::VectorXd mu = evaluate_inverse_link(eta_mu, mu_link);
        Eigen::VectorXd sigma_sq = gamma_inverse_link(eta_gamma);

        Eigen::VectorXd dmu_deta = evaluate_d_inverse_link(eta_mu, mu_link);
        Eigen::VectorXd dsigma2_deta = gamma_d_inverse_link(eta_gamma);

        Eigen::VectorXd dNLL_deta_mu = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dNLL_deta_gamma = Eigen::VectorXd::Zero(n);

        // per-observation derivatives w.r.t. the two linear predictors
        for (long i = 0; i < n; ++i) {
            double yi = std::max(1e-6, std::min(1.0 - 1e-6, y(i)));
            double mui = std::max(1e-6, std::min(1.0 - 1e-6, mu(i)));
            double s2i = std::max(1e-6, sigma_sq(i));

            double diff = yi - mui;
            double den_y = yi * (1.0 - yi);
            double den_mu = mui * (1.0 - mui);

            double den_mu_cube = den_mu * den_mu * den_mu;
            double num_mu = diff * (den_mu + diff * (1.0 - 2.0 * mui));
            double dNLL_dmu = -num_mu / (s2i * den_y * den_mu_cube);

            double grad_eta_mu = dNLL_dmu * dmu_deta(i);
            dNLL_deta_mu(i) = std::isfinite(grad_eta_mu) ? grad_eta_mu : 0.0;

            double dNLL_dsigma2 = 0.5 / s2i - (diff * diff) / (2.0 * s2i * s2i * den_y * den_mu * den_mu);

            double grad_eta_gamma = dNLL_dsigma2 * dsigma2_deta(i);
            dNLL_deta_gamma(i) = std::isfinite(grad_eta_gamma) ? grad_eta_gamma : 0.0;
        }

        Eigen::VectorXd grad(k_x + k_z);
        grad.head(k_x) = X.transpose() * dNLL_deta_mu;
        grad.tail(k_z) = Z.transpose() * dNLL_deta_gamma;
        return grad;
    }

    /**
     * @brief Analytical observed Hessian of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$.
     * @details Per observation, with \f$ N = r\,[D_\mu + r(1-2\mu)] \f$, \f$ s = \tilde\sigma^2 \f$ and \f$ D_i \f$ as in the
     * class description:
     * \f[ \frac{\partial^2\mathrm{NLL}_i}{\partial\mu^2} = \frac{-N'\,D_\mu + 3(1-2\mu)\,N}{s\,D_y\,D_\mu^4}, \quad
     *     N' = -D_\mu - r(1-2\mu) - 2r^2, \f]
     * \f[ \frac{\partial^2\mathrm{NLL}_i}{\partial s^2} = -\frac{1}{2s^2} + \frac{2D_i}{s^3}, \qquad
     *     \frac{\partial^2\mathrm{NLL}_i}{\partial\mu\,\partial s} = \frac{N}{s^2 D_y D_\mu^3}. \f]
     * The weights include the link curvature:
     * \f$ W_{\mu\mu} = \partial^2_\mu\mathrm{NLL}\,(d\mu/d\eta_\mu)^2 + \partial_\mu\mathrm{NLL}\,d^2\mu/d\eta_\mu^2 \f$,
     * \f$ W_{ss} = \partial^2_s\mathrm{NLL}\,(ds/d\eta_\gamma)^2 + \partial_s\mathrm{NLL}\,d^2s/d\eta_\gamma^2 \f$ and
     * \f$ W_{\mu s} = \partial_\mu\partial_s\mathrm{NLL}\;(d\mu/d\eta_\mu)(ds/d\eta_\gamma) \f$, and the blocks are
     * \f[ H = \begin{pmatrix} X^\top \mathrm{diag}(W_{\mu\mu}) X & X^\top \mathrm{diag}(W_{\mu s}) Z \\
     *     Z^\top \mathrm{diag}(W_{\mu s}) X & Z^\top \mathrm{diag}(W_{ss}) Z \end{pmatrix}. \f]
     * @param params Parameter vector \f$ (\boldsymbol\beta, \boldsymbol\gamma) \f$ of length \f$ k_x + k_z \f$.
     * @return Symmetric \f$ (k_x+k_z)\times(k_x+k_z) \f$ matrix. Non-finite per-observation weights are set to 0 and
     *         non-finite matrix entries are replaced by 0; no ridge and no PSD projection.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_x = X.cols();
        const long k_z = Z.cols();

        Eigen::VectorXd beta = params.head(k_x);
        Eigen::VectorXd gamma = params.tail(k_z);

        Eigen::VectorXd eta_mu = X * beta;
        Eigen::VectorXd eta_gamma = Z * gamma;

        Eigen::VectorXd mu = evaluate_inverse_link(eta_mu, mu_link);
        Eigen::VectorXd sigma_sq = gamma_inverse_link(eta_gamma);

        Eigen::VectorXd dmu_deta = evaluate_d_inverse_link(eta_mu, mu_link);
        Eigen::VectorXd dsigma2_deta = gamma_d_inverse_link(eta_gamma);

        // per-observation Hessian weights for the mean, dispersion and cross blocks
        Eigen::VectorXd W_mm = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_ss = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd W_ms = Eigen::VectorXd::Zero(n);

        for (long i = 0; i < n; ++i) {
            double yi = std::max(1e-6, std::min(1.0 - 1e-6, y(i)));
            double mui = std::max(1e-6, std::min(1.0 - 1e-6, mu(i)));
            double s2i = std::max(1e-6, sigma_sq(i));

            double diff = yi - mui;
            double diff2 = diff * diff;
            double den_y = yi * (1.0 - yi);
            double den_mu = mui * (1.0 - mui);
            double den_mu2 = den_mu * den_mu;
            double den_mu3 = den_mu2 * den_mu;

            double D_i = diff2 / (2.0 * den_y * den_mu2);

            double num_mu = diff * (den_mu + diff * (1.0 - 2.0 * mui));
            double dNLL_dmu = -num_mu / (s2i * den_y * den_mu3);

            double f = diff;
            double g = den_mu;
            double g_prime = 1.0 - 2.0 * mui;

            double dN_dmu = -g - f * g_prime - 2.0 * f * f;
            double d2D_dmu2 = (-dN_dmu * g + 3.0 * num_mu * g_prime) / (den_y * den_mu2 * den_mu2);

            double d2NLL_dmu2 = d2D_dmu2 / s2i;

            double d2mu_deta2 = evaluate_d2_inverse_link_scalar(eta_mu(i), mu_link);
            W_mm(i) = d2NLL_dmu2 * dmu_deta(i) * dmu_deta(i) + dNLL_dmu * d2mu_deta2;
            if (!std::isfinite(W_mm(i))) W_mm(i) = 0.0;

            double d2NLL_ds2s2 = -0.5 / (s2i * s2i) + 2.0 * D_i / (s2i * s2i * s2i);

            double dNLL_ds2 = 0.5 / s2i - D_i / (s2i * s2i);
            double d2s2_deta2 = gamma_d2_inverse_link_scalar(eta_gamma(i));
            W_ss(i) = d2NLL_ds2s2 * dsigma2_deta(i) * dsigma2_deta(i) + dNLL_ds2 * d2s2_deta2;
            if (!std::isfinite(W_ss(i))) W_ss(i) = 0.0;

            double d2NLL_dmu_ds2 = num_mu / (s2i * s2i * den_y * den_mu3);

            W_ms(i) = d2NLL_dmu_ds2 * dmu_deta(i) * dsigma2_deta(i);
            if (!std::isfinite(W_ms(i))) W_ms(i) = 0.0;
        }

        // assemble Hessian blocks
        const long n_params = k_x + k_z;
        Eigen::MatrixXd H(n_params, n_params);

        H.block(0, 0, k_x, k_x) = X.transpose() * W_mm.asDiagonal() * X;

        H.block(k_x, k_x, k_z, k_z) = Z.transpose() * W_ss.asDiagonal() * Z;

        Eigen::MatrixXd H_cross = X.transpose() * W_ms.asDiagonal() * Z;
        H.block(0, k_x, k_x, k_z) = H_cross;
        H.block(k_x, 0, k_z, k_x) = H_cross.transpose();

        return H.array().isFinite().select(H, 0.0);
    }
};

/**
 * @brief Negative log-likelihood of a multinomial regression with \f$ K \f$ nominal classes and a selectable link (softmax or sequential binary links).
 *
 * @details Models a categorical response \f$ y_i \in \{0,\dots,K-1\} \f$. Class \f$ K-1 \f$ is the *reference class*:
 * only \f$ K-1 \f$ linear predictors are estimated,
 * \f[ \eta_{ij} = \mathbf{x}_i^\top \boldsymbol\beta_j, \qquad j = 0,\dots,K-2, \qquad \eta_{i,K-1} \equiv 0 . \f]
 * The class probabilities depend on `link_function`:
 * - `LinkType::Logit` (default): multinomial logit (softmax with the reference score fixed at 0),
 *   \f[ p_{ij} = \frac{e^{\eta_{ij}}}{1 + \sum_{l=0}^{K-2} e^{\eta_{il}}}, \qquad
 *       p_{i,K-1} = \frac{1}{1 + \sum_{l=0}^{K-2} e^{\eta_{il}}} . \f]
 *   It is evaluated stably by subtracting the row maximum of the \f$ K \f$ scores (the zero score included).
 * - `LinkType::Probit`, `Cloglog`, `Loglog`, `Cauchit`: *sequential (continuation-ratio / stopping-ratio)* model. With
 *   \f$ F_{ij} = F(\eta_{ij}) \f$ the CDF associated with the link (`evaluate_link_cdf()`),
 *   \f[ p_{ij} = F_{ij}\prod_{l<j}(1 - F_{il}), \quad j \le K-2, \qquad
 *       p_{i,K-1} = \prod_{l=0}^{K-2}(1 - F_{il}) , \f]
 *   i.e. \f$ F_{ij} \f$ is the probability of stopping in class \f$ j \f$ given that classes \f$ 0,\dots,j-1 \f$ were
 *   "passed". The sequential model is only an \f$ \eta \f$-separable likelihood: each class column has its own
 *   \f$ \boldsymbol\beta_j \f$ and there are no shared thresholds.
 *
 * ### Parameter vector layout
 * | Index                          | Meaning                                                    | Scale |
 * |--------------------------------|------------------------------------------------------------|-------|
 * | `j*k + c`, `c = 0..k-1`, `j = 0..K-2` | coefficient of design column \f$ c \f$ for class \f$ j \f$ (column-major flattening of a \f$ k \times (K-1) \f$ matrix) | linear predictor scale |
 *
 * where \f$ k \f$ is the number of columns of the design matrix. The total length is \f$ k(K-1) \f$; there are no
 * auxiliary parameters, and no intercept is added automatically (include a constant column in `X` if required).
 *
 * ### Log-likelihood
 * \f[ \ell_i = \log p_{i,y_i}, \qquad \mathrm{NLL} = -\sum_{i} \log\bigl(\mathrm{clip}(p_{i,y_i},\,10^{-15},\,1-10^{-15})\bigr) . \f]
 * No observation weights, censoring or constants are involved.
 *
 * ### Derivatives
 * - Logit: \f$ \partial\mathrm{NLL}_i/\partial\eta_{ij} = p_{ij} - \mathbb{1}_{y_i=j} \f$ for \f$ j \le K-2 \f$.
 * - Other links: the NLL of observation \f$ i \f$ is \f$ -\log F_{iy_i} - \sum_{l<y_i}\log(1-F_{il}) \f$ (the first term is
 *   absent if \f$ y_i = K-1 \f$), so only the predictors \f$ j \le \min(y_i, K-2) \f$ contribute:
 *   \f$ -f_{ij}/F_{ij} \f$ for \f$ j = y_i \f$ and \f$ f_{ij}/(1-F_{ij}) \f$ for \f$ j < y_i \f$, where
 *   \f$ f = F' \f$ is `evaluate_d_inverse_link()`.
 * - Chain rule: the gradient block of class \f$ j \f$ is \f$ \sum_i \partial\mathrm{NLL}_i/\partial\eta_{ij}\,\mathbf{x}_i \f$.
 * - The Hessian is block structured with blocks \f$ X^\top \mathrm{diag}(W)\,X \f$: full \f$ (K-1)\times(K-1) \f$ block
 *   matrix for Logit (exact, \f$ W_{jj} = p_j(1-p_j) \f$, \f$ W_{jm} = -p_jp_m \f$), block-diagonal for the sequential
 *   links (exact, the cross-class second derivatives are identically zero).
 *
 * ### Numerical safeguards
 * - Link CDF clipped to \f$ [10^{-12}, 1-10^{-12}] \f$; link density floored at \f$ 10^{-10} \f$; Cloglog / Loglog use
 *   \f$ \min(\eta, 20) \f$ before exponentiation.
 * - Probability of the observed class clipped to \f$ [10^{-15}, 1-10^{-15}] \f$ in the NLL (no `1e10` fallback is used).
 * - Per-observation gradient entries: non-finite replaced by 0, then clipped to \f$ [-30, 30] \f$.
 * - Sequential Hessian weights: non-finite replaced by 0, then clipped to \f$ [-10^{6}, 10^{6}] \f$.
 *
 * @note The object stores **non-owning references** to `X` and `y`, which must outlive the functor. No validation is
 *       performed: every `y(i)` must lie in `[0, K-1]` (it is used directly as a column index) and `params` must have
 *       length \f$ k(K-1) \f$.
 * @note The sequential Hessian is not PSD-projected and, unlike the Logit branch, the clipped gradient is not the exact
 *       derivative of the NLL where the clipping is active.
 * @see OrdinalLoss for the cumulative-link model with ordered categories.
 *
 * @ingroup glm_ll_categorical
 */
struct MultinomialMultiLinkLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix (\f$ n \times k \f$), shared by all class predictors.
    const Eigen::VectorXi& y;  ///< Observed class labels in \f$ \{0,\dots,K-1\} \f$ (length \f$ n \f$); class \f$ K-1 \f$ is the reference.
    const int num_classes;  ///< Number of classes \f$ K \f$ (so \f$ K-1 \f$ coefficient vectors are estimated).
    LinkType link_function;  ///< Link selecting the model: `Logit` = multinomial logit, `Probit`/`Cloglog`/`Loglog`/`Cauchit` = sequential model.

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog_X Design matrix \f$ X \f$ (\f$ n \times k \f$).
     * @param endog_y Integer class labels (length \f$ n \f$), each in \f$ \{0,\dots,K-1\} \f$.
     * @param K Number of classes \f$ K \ge 2 \f$.
     * @param link Link function: `Logit` (default, softmax) or one of `Probit`, `Cloglog`, `Loglog`, `Cauchit` (sequential model).
     * @pre `exog_X` and `endog_y` have the same number of rows and outlive the functor; labels are in range.
     */
    MultinomialMultiLinkLoss(const Eigen::MatrixXd& exog_X,
                             const Eigen::VectorXi& endog_y,
                             int K,
                             LinkType link = LinkType::Logit)
        : X(exog_X), y(endog_y), num_classes(K), link_function(link) {}

    /**
     * @brief Evaluates the link CDF \f$ F(\eta) \f$ used by the sequential (non-Logit) models.
     * @details Implemented on the first \f$ K-1 \f$ columns of `eta_matrix`:
     * - `Probit`: \f$ \tfrac12\{1 + \mathrm{erf}(\eta/\sqrt2)\} \f$;
     * - `Cloglog`: \f$ 1 - \exp(-e^{\min(\eta,20)}) \f$;
     * - `Loglog`: \f$ \exp(-e^{-\min(\eta,20)}) \f$;
     * - `Cauchit`: \f$ \tfrac12 + \arctan(\eta)/\pi \f$;
     * - any other link (including `Logit`) leaves the matrix at 0 before clipping, so every entry becomes `1e-12`.
     * @param eta_matrix Score matrix (\f$ n \times K \f$); only the first \f$ K-1 \f$ columns are used.
     * @return Matrix (\f$ n \times (K-1) \f$) of CDF values clipped to \f$ [10^{-12}, 1-10^{-12}] \f$.
     */
    inline Eigen::MatrixXd evaluate_link_cdf(const Eigen::MatrixXd& eta_matrix) const {
        const long n = eta_matrix.rows();
        const int K_minus_1 = num_classes - 1;
        Eigen::MatrixXd F_eta = Eigen::MatrixXd::Zero(n, K_minus_1);
        auto eta_sub = eta_matrix.leftCols(K_minus_1).array();

        switch (link_function) {
            case LinkType::Probit: {
                static const double INV_SQRT_2 = 1.0 / std::sqrt(2.0);
                F_eta = (0.5 * (1.0 + (eta_sub * INV_SQRT_2).erf())).matrix();
                break;
            }
            case LinkType::Cloglog:
                F_eta = (1.0 - (-(eta_sub.cwiseMin(20.0)).exp()).exp()).matrix();
                break;
            case LinkType::Loglog:
                F_eta = ((- (-eta_sub.cwiseMin(20.0)).exp()).exp()).matrix();
                break;
            case LinkType::Cauchit:
                F_eta = (0.5 + (1.0 / M_PI) * eta_sub.atan()).matrix();
                break;
            default:
                break;
        }

        return F_eta.cwiseMax(1e-12).cwiseMin(1.0 - 1e-12);
    }

    /**
     * @brief Computes the class probability matrix from the score matrix.
     * @details For `Logit`: row-wise stable softmax where the last column (reference class, score 0) is set to
     * \f$ e^{-m_i} \f$ with \f$ m_i \f$ the row maximum over all \f$ K \f$ scores, and every entry is divided by the row sum.
     * For the other links: sequential stick-breaking \f$ p_{ij} = F_{ij}\,r_{ij} \f$ with remainder
     * \f$ r_{i,j+1} = r_{ij}(1-F_{ij}) \f$, \f$ r_{i0}=1 \f$, and \f$ p_{i,K-1} = r_{i,K-1} \f$. Probabilities are not clipped here.
     * @param eta_matrix Score matrix (\f$ n \times K \f$) whose last column is 0.
     * @return Probability matrix (\f$ n \times K \f$) with rows summing to 1.
     */
    inline Eigen::MatrixXd evaluate_inverse_link(const Eigen::MatrixXd& eta_matrix) const {
        const long n = eta_matrix.rows();
        const int K_minus_1 = num_classes - 1;
        Eigen::MatrixXd probs = Eigen::MatrixXd::Zero(n, num_classes);

        if (link_function == LinkType::Logit) {

            Eigen::VectorXd max_scores = eta_matrix.rowwise().maxCoeff();
            Eigen::MatrixXd exp_matrix = (eta_matrix.array().colwise() - max_scores.array()).exp().matrix();
            exp_matrix.rightCols(1) = (-max_scores.array()).exp().matrix();
            Eigen::VectorXd row_sums = exp_matrix.rowwise().sum();
            probs = exp_matrix.array().colwise() / row_sums.array();
        }
        else {

            Eigen::MatrixXd F_eta = evaluate_link_cdf(eta_matrix);

            Eigen::VectorXd rem = Eigen::VectorXd::Ones(n);
            for (int j = 0; j < K_minus_1; ++j) {
                probs.col(j) = rem.array() * F_eta.col(j).array();
                rem.array() *= (1.0 - F_eta.col(j).array());
            }
            probs.col(num_classes - 1) = rem;
        }
        return probs;
    }

    /**
     * @brief Evaluates the link density \f$ f(\eta) = F'(\eta) \f$ for the sequential (non-Logit) models.
     * @details `Probit`: \f$ \varphi(\eta) \f$; `Cloglog`: \f$ e^{\eta}\exp(-e^{\eta}) \f$ with \f$ \eta \f$ capped at 20;
     * `Loglog`: \f$ e^{-\eta}\exp(-e^{-\eta}) \f$ with \f$ \eta \f$ capped at 20; `Cauchit`: \f$ 1/\{\pi(1+\eta^2)\} \f$.
     * `Logit` (and any link not listed) yields 0 before flooring.
     * @param eta_matrix Score matrix (\f$ n \times K \f$); only the first \f$ K-1 \f$ columns are used.
     * @return Matrix (\f$ n \times (K-1) \f$) of densities floored at `1e-10`.
     */
    inline Eigen::MatrixXd evaluate_d_inverse_link(const Eigen::MatrixXd& eta_matrix) const {
        const int K_minus_1 = num_classes - 1;
        Eigen::MatrixXd dP_deta = Eigen::MatrixXd::Zero(eta_matrix.rows(), K_minus_1);
        auto eta_sub = eta_matrix.leftCols(K_minus_1).array();

        switch (link_function) {
            case LinkType::Logit:
                break;
            case LinkType::Probit: {
                static const double INV_SQRT_2PI = 1.0 / std::sqrt(2.0 * M_PI);
                dP_deta = (INV_SQRT_2PI * (-0.5 * eta_sub.square()).exp()).matrix();
                break;
            }
            case LinkType::Cloglog: {
                auto exp_eta = eta_sub.cwiseMin(20.0).exp();
                dP_deta = (exp_eta * (-exp_eta).exp()).matrix();
                break;
            }
            case LinkType::Loglog: {
                auto exp_neg_eta = (-eta_sub.cwiseMin(20.0)).exp();
                dP_deta = (exp_neg_eta * (-exp_neg_eta).exp()).matrix();
                break;
            }
            case LinkType::Cauchit: {
                dP_deta = (1.0 / (M_PI * (1.0 + eta_sub.square()))).matrix();
                break;
            }
            default:
                break;
        }
        return dP_deta.cwiseMax(1e-10);
    }

    /**
     * @brief Evaluates the second derivative \f$ F''(\eta) = f'(\eta) \f$ of the link CDF for the sequential models.
     * @details Expressed through the density \f$ f \f$: `Probit`: \f$ -\eta f \f$; `Cloglog`: \f$ f\,(1-e^{\eta}) \f$;
     * `Loglog`: \f$ f\,(e^{-\eta}-1) \f$; `Cauchit`: \f$ -2\eta f/(1+\eta^2) \f$ (with the same cap of 20 on \f$ \eta \f$ for the
     * exponential links). `Logit` and any other link yield 0. No clipping is applied.
     * @param eta_matrix Score matrix (\f$ n \times K \f$); only the first \f$ K-1 \f$ columns are used.
     * @param f_eta Link density matrix (\f$ n \times (K-1) \f$) as returned by `evaluate_d_inverse_link()`.
     * @return Matrix (\f$ n \times (K-1) \f$) of second derivatives of the CDF.
     */
    inline Eigen::MatrixXd evaluate_d2_inverse_link(const Eigen::MatrixXd& eta_matrix, const Eigen::MatrixXd& f_eta) const {
        const int K_minus_1 = num_classes - 1;
        Eigen::MatrixXd d2P_deta2 = Eigen::MatrixXd::Zero(eta_matrix.rows(), K_minus_1);
        auto eta_sub = eta_matrix.leftCols(K_minus_1).array();
        auto f_arr = f_eta.array();

        switch (link_function) {
            case LinkType::Probit:
                d2P_deta2 = (-eta_sub * f_arr).matrix();
                break;
            case LinkType::Cloglog: {
                auto exp_eta = eta_sub.cwiseMin(20.0).exp();
                d2P_deta2 = (f_arr * (1.0 - exp_eta)).matrix();
                break;
            }
            case LinkType::Loglog: {
                auto exp_neg_eta = (-eta_sub.cwiseMin(20.0)).exp();
                d2P_deta2 = (f_arr * (exp_neg_eta - 1.0)).matrix();
                break;
            }
            case LinkType::Cauchit: {
                auto eta_sq = eta_sub.square();
                auto factor = -2.0 * eta_sub / (1.0 + eta_sq);
                d2P_deta2 = (f_arr * factor).matrix();
                break;
            }
            default:
                break;
        }

        return d2P_deta2;
    }

    /**
     * @brief Computes the score (linear predictor) matrix \f$ \eta_{ij} = \mathbf{x}_i^\top\boldsymbol\beta_j \f$.
     * @details The parameter vector is reinterpreted (column-major, no copy) as the \f$ k \times (K-1) \f$ matrix
     * \f$ (\boldsymbol\beta_0,\dots,\boldsymbol\beta_{K-2}) \f$.
     * @param params Flattened coefficient vector of length \f$ k(K-1) \f$.
     * @return Matrix (\f$ n \times K \f$): first \f$ K-1 \f$ columns equal \f$ X B \f$, the last (reference) column is 0.
     */
    inline Eigen::MatrixXd compute_scores(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_x = X.cols();
        const int K_minus_1 = num_classes - 1;

        Eigen::Map<const Eigen::MatrixXd> beta_flat(params.data(), k_x, K_minus_1);
        Eigen::MatrixXd score_matrix = Eigen::MatrixXd::Zero(n, num_classes);
        score_matrix.leftCols(K_minus_1) = X * beta_flat;

        return score_matrix;
    }

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Flattened coefficient vector of length \f$ k(K-1) \f$ (see the layout table).
     * @return The value \f$ -\sum_i \log p_{i,y_i} \f$ with each probability clipped to \f$ [10^{-15}, 1-10^{-15}] \f$.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        // linear predictors and class probabilities
        Eigen::MatrixXd score_matrix = compute_scores(params);
        Eigen::MatrixXd probs = evaluate_inverse_link(score_matrix);

        // accumulate -log p of the observed class
        double total_neg_ll = 0.0;
        for (long i = 0; i < n; ++i) {
            double p_i = std::max(1e-15, std::min(1.0 - 1e-15, probs(i, y(i))));
            total_neg_ll -= std::log(p_i);
        }
        return total_neg_ll;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood.
     * @param params Flattened coefficient vector of length \f$ k(K-1) \f$.
     * @return Vector of length \f$ k(K-1) \f$ in the same layout as `params`: \f$ X^\top G \f$ with \f$ G_{ij} = \partial\mathrm{NLL}_i/\partial\eta_{ij} \f$
     *         (\f$ p_{ij}-\mathbb{1}_{y_i=j} \f$ for Logit, \f$ -f/F \f$ / \f$ f/(1-F) \f$ otherwise). Entries of \f$ G \f$
     *         that are non-finite are set to 0 and the others clipped to \f$ [-30,30] \f$.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_x = X.cols();
        const int K_minus_1 = num_classes - 1;

        Eigen::MatrixXd score_matrix = compute_scores(params);
        Eigen::MatrixXd probs = evaluate_inverse_link(score_matrix);
        // per-observation derivative of the NLL with respect to each predictor
        Eigen::MatrixXd dNLL_deta = Eigen::MatrixXd::Zero(n, K_minus_1);

        if (link_function == LinkType::Logit) {
            dNLL_deta = probs.leftCols(K_minus_1);
            for (long i = 0; i < n; ++i) {
                if (y(i) < K_minus_1) dNLL_deta(i, y(i)) -= 1.0;
            }
        }
        else {
            Eigen::MatrixXd F_eta = evaluate_link_cdf(score_matrix);
            Eigen::MatrixXd f_eta = evaluate_d_inverse_link(score_matrix);

            for (long i = 0; i < n; ++i) {
                const int true_class = y(i);
                const int limit = std::min(true_class, K_minus_1 - 1);
                for (int j = 0; j <= limit; ++j) {
                    if (true_class == j) {
                        dNLL_deta(i, j) = -f_eta(i, j) / F_eta(i, j);
                    } else {
                        dNLL_deta(i, j) = f_eta(i, j) / (1.0 - F_eta(i, j));
                    }
                }
            }
        }

        dNLL_deta = dNLL_deta.unaryExpr([](double v) {
            if (!std::isfinite(v)) return 0.0;
            return std::max(-30.0, std::min(30.0, v));
        });

        // chain rule to the coefficients and column-major flattening
        Eigen::MatrixXd grad_beta = X.transpose() * dNLL_deta;
        Eigen::VectorXd grad(k_x * K_minus_1);
        Eigen::Map<Eigen::MatrixXd>(grad.data(), k_x, K_minus_1) = grad_beta;

        return grad;
    }

    /**
     * @brief Analytical Hessian of the negative log-likelihood with respect to the flattened coefficients.
     * @details Block \f$ (j,m) \f$ (each \f$ k \times k \f$) equals \f$ X^\top \mathrm{diag}(W^{(jm)})\,X \f$.
     * - Logit: \f$ W^{(jj)} = p_j(1-p_j) \f$ and \f$ W^{(jm)} = -p_jp_m \f$ for \f$ j \ne m \f$ (exact, independent of the observed
     *   class); the blocks are computed for \f$ m \ge j \f$ and mirrored.
     * - Other links: only diagonal blocks, with \f$ W_i = (f^2 - F''F)/F^2 \f$ if \f$ y_i = j \f$,
     *   \f$ W_i = (F''(1-F) + f^2)/(1-F)^2 \f$ if \f$ y_i > j \f$ and 0 if \f$ y_i < j \f$ (the derivative of the
     *   per-observation gradient entries without the \f$ \pm30 \f$ clipping). Weights are set to 0 when non-finite and clipped to
     *   \f$ [-10^{6}, 10^{6}] \f$.
     * @param params Flattened coefficient vector of length \f$ k(K-1) \f$.
     * @return Symmetric matrix of size \f$ k(K-1) \times k(K-1) \f$. No ridge, symmetrisation or finite masking is applied
     *         to the final matrix.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k_x = X.cols();
        const int K_minus_1 = num_classes - 1;

        Eigen::MatrixXd score_matrix = compute_scores(params);
        Eigen::MatrixXd probs = evaluate_inverse_link(score_matrix);

        // assemble the Hessian blocks X' diag(W) X
        Eigen::MatrixXd hessian_matrix = Eigen::MatrixXd::Zero(k_x * K_minus_1, k_x * K_minus_1);

        if (link_function == LinkType::Logit) {

            for (int j = 0; j < K_minus_1; ++j) {
                for (int m = j; m < K_minus_1; ++m) {
                    Eigen::VectorXd W;
                    if (j == m) {
                        W = (probs.col(j).array() * (1.0 - probs.col(j).array())).matrix();
                    } else {
                        W = (-probs.col(j).array() * probs.col(m).array()).matrix();
                    }
                    Eigen::MatrixXd block = X.transpose() * W.asDiagonal() * X;

                    hessian_matrix.block(j * k_x, m * k_x, k_x, k_x) = block;
                    if (j != m) {
                        hessian_matrix.block(m * k_x, j * k_x, k_x, k_x) = block;
                    }
                }
            }
        }
        else {

            Eigen::MatrixXd F_eta = evaluate_link_cdf(score_matrix);
            Eigen::MatrixXd f_eta = evaluate_d_inverse_link(score_matrix);
            Eigen::MatrixXd d2_eta = evaluate_d2_inverse_link(score_matrix, f_eta);

            for (int j = 0; j < K_minus_1; ++j) {

                Eigen::Array<bool, Eigen::Dynamic, 1> mask_j_eq_y = (y.array() == j);
                Eigen::Array<bool, Eigen::Dynamic, 1> mask_j_lt_y = (y.array() > j);

                Eigen::ArrayXd f = f_eta.col(j).array();
                Eigen::ArrayXd F = F_eta.col(j).array();
                Eigen::ArrayXd d2 = d2_eta.col(j).array();
                Eigen::ArrayXd omF = 1.0 - F;

                Eigen::ArrayXd W_eq = (f * f - d2 * F) / (F * F);
                Eigen::ArrayXd W_lt = (d2 * omF + f * f) / (omF * omF);

                Eigen::VectorXd W = Eigen::VectorXd::Zero(n);
                W.array() = mask_j_eq_y.select(W_eq, W.array());
                W.array() = mask_j_lt_y.select(W_lt, W.array());

                W = W.unaryExpr([](double v) {
                    if (!std::isfinite(v)) return 0.0;
                    return std::max(-1e6, std::min(1e6, v));
                });

                hessian_matrix.block(j * k_x, j * k_x, k_x, k_x) = X.transpose() * W.asDiagonal() * X;
            }
        }

        return hessian_matrix;
    }

};

/**
 * @brief Negative log-likelihood of the cumulative-link ordinal regression model with ordered thresholds.
 *
 * @details Models an ordered response \f$ y_i \in \{0,\dots,M-1\} \f$ with \f$ M \f$ categories through a latent variable
 * with CDF \f$ F \f$ selected by `mu_link` (logit, probit, complementary log-log, log-log or Cauchy):
 * \f[ P(y_i \le c) = F(\alpha_c - \eta_i), \qquad \eta_i = \mathbf{x}_i^\top\boldsymbol\beta, \qquad
 *     \alpha_0 < \alpha_1 < \dots < \alpha_{M-2}, \f]
 * so \f$ P(y_i = c) = F(\alpha_c - \eta_i) - F(\alpha_{c-1} - \eta_i) \f$ with the conventions
 * \f$ F(\alpha_{-1}-\eta) = 0 \f$ and \f$ F(\alpha_{M-1}-\eta) = 1 \f$. A larger \f$ \eta_i \f$ shifts the mass towards
 * higher categories. The design matrix must not contain an intercept (it is absorbed by the thresholds).
 * The ordering of the thresholds is enforced by a reparametrisation: the first threshold is free and the
 * others are built from log-increments (`reconstruct_thresholds()`).
 *
 * ### Parameter vector layout
 * | Index                  | Meaning                                                   | Scale |
 * |------------------------|-----------------------------------------------------------|-------|
 * | `0 .. k-1`             | regression coefficients \f$ \boldsymbol\beta \f$           | linear predictor scale |
 * | `k`                    | first threshold \f$ \alpha_0 \f$                           | identity (unconstrained) |
 * | `k+m`, `m = 1..M-2`    | \f$ \gamma_m = \log(\alpha_m - \alpha_{m-1}) \f$           | log increment (clamped to \f$ [-15,15] \f$ when used) |
 *
 * where \f$ k \f$ is the number of design columns; the total length is \f$ k + M - 1 \f$ and
 * \f$ \alpha_m = \alpha_0 + \sum_{j=1}^{m} e^{\gamma_j} \f$.
 *
 * ### Log-likelihood
 * \f[ \ell_i = \log\max\bigl\{10^{-15},\ F(\alpha_{y_i}-\eta_i) - F(\alpha_{y_i-1}-\eta_i)\bigr\}, \qquad
 *     \mathrm{NLL} = -\sum_i \ell_i . \f]
 * No weights or censoring are supported.
 *
 * ### Derivatives
 * With \f$ u_i = \alpha_{y_i}-\eta_i \f$, \f$ l_i = \alpha_{y_i-1}-\eta_i \f$, \f$ f = F' \f$ and \f$ P_i \f$ the (floored)
 * category probability, the per-observation derivatives with respect to the CDF arguments are
 * \f$ \partial\mathrm{NLL}_i/\partial u_i = -f(u_i)/P_i \f$ and \f$ \partial\mathrm{NLL}_i/\partial l_i = f(l_i)/P_i \f$
 * (the term of a missing boundary, i.e. \f$ u \f$ for the last and \f$ l \f$ for the first category, is 0). Then
 * \f$ \partial\mathrm{NLL}_i/\partial\eta_i = -\partial_u - \partial_l \f$, the gradients with respect to
 * \f$ \alpha_{y_i} \f$ and \f$ \alpha_{y_i-1} \f$ are \f$ \partial_u \f$ and \f$ \partial_l \f$, and these are mapped
 * to \f$ (\alpha_0,\gamma_1,\dots) \f$ by \f$ \partial/\partial\alpha_0 = \sum_m \partial/\partial\alpha_m \f$,
 * \f$ \partial/\partial\gamma_m = e^{\gamma_m}\sum_{l\ge m}\partial/\partial\alpha_l \f$.
 *
 * The Hessian is built from the exact second derivatives in the \f$ (\boldsymbol\beta,\boldsymbol\alpha) \f$ coordinates,
 * transformed to \f$ (\boldsymbol\beta,\alpha_0,\boldsymbol\gamma) \f$ with the Jacobian \f$ J = \partial\boldsymbol\alpha/\partial\boldsymbol\gamma \f$
 * and the curvature of the map, and finally re-expressed in \f$ (\boldsymbol\beta,\boldsymbol\alpha) \f$ coordinates (see `hessian()`:
 * the result is the exact threshold-coordinate Hessian at the maximum-likelihood estimate, not elsewhere).
 *
 * ### Numerical safeguards
 * - Category probability floored at `1e-15` (value, gradient, Hessian); the NLL returns `1e10` if not finite.
 * - Link-specific clamps of \f$ \eta \f$ inside the CDF/density: Probit \f$ [-5,5] \f$, Cloglog \f$ [-30,3.5] \f$, Loglog
 *   \f$ [-5,30] \f$; Logit uses a branch-wise stable sigmoid, Cauchit is unclamped.
 * - \f$ \gamma_m \f$ clamped to \f$ [-15, 15] \f$ in every exponentiation (the derivatives do not account for the clamp).
 * - Gradient: non-finite per-observation terms and non-finite final components are set to 0. Hessian: non-finite entries set to 0
 *   and the result is symmetrised; no ridge is added.
 *
 * @note The object stores **non-owning references** to `X` and `y`, which must outlive the functor. No validation is
 *       performed: `y(i)` must be an integer-valued double in `[0, M-1]` (cast with `static_cast<int>` and used as an index)
 *       and \f$ M \ge 2 \f$.
 * @note The data members `mu_link` and `num_categories` are declared in a different order than they appear in the
 *       constructor initialiser list (harmless, members are initialised in declaration order).
 * @see MultinomialMultiLinkLoss for the nominal (unordered) multi-class model.
 *
 * @ingroup glm_ll_categorical
 */
struct OrdinalLoss {
    const Eigen::MatrixXd& X;  ///< Design matrix without intercept (\f$ n \times k \f$).
    const Eigen::VectorXd& y;  ///< Ordered category labels stored as doubles in \f$ \{0,\dots,M-1\} \f$ (length \f$ n \f$).
    LinkType mu_link;  ///< Link selecting the latent CDF \f$ F \f$ (`Logit`, `Probit`, `Cloglog`, `Loglog`, `Cauchit`).
    int num_categories;  ///< Number of ordered categories \f$ M \f$ (there are \f$ M-1 \f$ thresholds).

    /**
     * @brief Builds the loss functor (the data are referenced, not copied).
     * @param exog Design matrix \f$ X \f$ (\f$ n \times k \f$), without intercept column.
     * @param endog Category labels (length \f$ n \f$) with integer values in \f$ \{0,\dots,M-1\} \f$.
     * @param M Number of categories \f$ M \ge 2 \f$.
     * @param link Link whose CDF defines the latent distribution (default `Logit`).
     * @pre `exog` and `endog` have the same number of rows and outlive the functor.
     */
    OrdinalLoss(const Eigen::MatrixXd& exog, const Eigen::VectorXd& endog, int M, LinkType link = LinkType::Logit)
        : X(exog), y(endog), mu_link(link), num_categories(M) {}

    /**
     * @brief Evaluates the latent CDF \f$ F(\eta) \f$ for the chosen link.
     * @details `Logit`: numerically stable logistic function (branch on the sign of \f$ \eta \f$); `Probit`:
     * \f$ \tfrac12\{1+\mathrm{erf}(\eta_c/\sqrt2)\} \f$ with \f$ \eta_c = \mathrm{clamp}(\eta,-5,5) \f$; `Cloglog`:
     * \f$ 1-\exp(-e^{\eta_c}) \f$ with \f$ \eta_c \in [-30, 3.5] \f$; `Loglog`: \f$ \exp(-e^{-\eta_c}) \f$ with
     * \f$ \eta_c \in [-5, 30] \f$; `Cauchit`: \f$ \tfrac12 + \arctan(\eta)/\pi \f$ (unclamped).
     * @param eta Argument \f$ \alpha_c - \eta_i \f$.
     * @param link Link type.
     * @return CDF value in \f$ [0,1] \f$; `0.5` for any unsupported link type.
     */
    static inline double evaluate_ordinal_cdf(double eta, LinkType link) {
        switch (link) {
            case LinkType::Logit: {
                if (eta >= 0.0) return 1.0 / (1.0 + std::exp(-eta));
                double exp_eta = std::exp(eta);
                return exp_eta / (1.0 + exp_eta);
            }
            case LinkType::Probit: {
                double c_eta = std::max(-5.0, std::min(5.0, eta));
                return 0.5 * (1.0 + std::erf(c_eta * 0.7071067811865475));
            }
            case LinkType::Cloglog: {
                double c_eta = std::max(-30.0, std::min(3.5, eta));
                return 1.0 - std::exp(-std::exp(c_eta));
            }
            case LinkType::Loglog: {
                double c_eta = std::max(-5.0, std::min(30.0, eta));
                return std::exp(-std::exp(-c_eta));
            }
            case LinkType::Cauchit: {
                return 0.5 + (std::atan(eta) * 0.31830988618379067);
            }
            default:
                return 0.5;
        }
    }

    /**
     * @brief Evaluates the latent density \f$ f(\eta) = F'(\eta) \f$ for the chosen link.
     * @details `Logit`: \f$ F(1-F) \f$; `Probit`: \f$ \varphi(\eta_c) \f$ with \f$ \eta_c \in [-5,5] \f$; `Cloglog`:
     * \f$ e^{\eta_c}\exp(-e^{\eta_c}) \f$ with \f$ \eta_c \in [-30,3.5] \f$; `Loglog`: \f$ e^{-\eta_c}\exp(-e^{-\eta_c}) \f$ with
     * \f$ \eta_c \in [-5,30] \f$; `Cauchit`: \f$ 1/\{\pi(1+\eta^2)\} \f$. The density is evaluated at the clamped argument, so it
     * does not vanish beyond the clamps although the CDF is constant there.
     * @param eta Argument \f$ \alpha_c - \eta_i \f$.
     * @param link Link type.
     * @return Density value; `0` for any unsupported link type.
     */
    static inline double evaluate_ordinal_pdf(double eta, LinkType link) {
        switch (link) {
            case LinkType::Logit: {
                double cdf = evaluate_ordinal_cdf(eta, LinkType::Logit);
                return cdf * (1.0 - cdf);
            }
            case LinkType::Probit: {
                double c_eta = std::max(-5.0, std::min(5.0, eta));
                return 0.3989422804014327 * std::exp(-0.5 * c_eta * c_eta);
            }
            case LinkType::Cloglog: {
                double c_eta = std::max(-30.0, std::min(3.5, eta));
                double exp_eta = std::exp(c_eta);
                return exp_eta * std::exp(-exp_eta);
            }
            case LinkType::Loglog: {
                double c_eta = std::max(-5.0, std::min(30.0, eta));
                double exp_neg_eta = std::exp(-c_eta);
                return exp_neg_eta * std::exp(-exp_neg_eta);
            }
            case LinkType::Cauchit: {
                return 0.31830988618379067 / (1.0 + eta * eta);
            }
            default:
                return 0.0;
        }
    }

    /**
     * @brief Evaluates the second derivative \f$ F''(\eta) = f'(\eta) \f$ of the latent CDF.
     * @details `Logit`: \f$ F(1-F)(1-2F) \f$; `Probit`: \f$ -\eta_c\,\varphi(\eta_c) \f$; `Cloglog`: \f$ f\,(1-e^{\eta_c}) \f$;
     * `Loglog`: \f$ f\,(e^{-\eta_c}-1) \f$; `Cauchit`: \f$ f\cdot(-2\eta/(1+\eta^2)) \f$, with the same clamps as
     * `evaluate_ordinal_cdf()` (Cauchit and Logit unclamped).
     * @param eta Argument \f$ \alpha_c - \eta_i \f$.
     * @param link Link type.
     * @return Second derivative of the CDF; `0` for any unsupported link type.
     */
    static inline double evaluate_ordinal_d2_cdf(double eta, LinkType link) {
        switch (link) {
            case LinkType::Logit: {
                double cdf = evaluate_ordinal_cdf(eta, LinkType::Logit);
                return cdf * (1.0 - cdf) * (1.0 - 2.0 * cdf);
            }
            case LinkType::Probit: {
                double c_eta = std::max(-5.0, std::min(5.0, eta));
                double pdf = 0.3989422804014327 * std::exp(-0.5 * c_eta * c_eta);
                return -c_eta * pdf;
            }
            case LinkType::Cloglog: {
                double c_eta = std::max(-30.0, std::min(3.5, eta));
                double pdf = evaluate_ordinal_pdf(c_eta, LinkType::Cloglog);
                return pdf * (1.0 - std::exp(c_eta));
            }
            case LinkType::Loglog: {
                double c_eta = std::max(-5.0, std::min(30.0, eta));
                double pdf = evaluate_ordinal_pdf(c_eta, LinkType::Loglog);
                return pdf * (std::exp(-c_eta) - 1.0);
            }
            case LinkType::Cauchit: {
                double eta_sq = eta * eta;
                double pdf = 0.31830988618379067 / (1.0 + eta_sq);
                return pdf * (-2.0 * eta / (1.0 + eta_sq));
            }
            default:
                return 0.0;
        }
    }

    /**
     * @brief Rebuilds the ordered thresholds \f$ \alpha \f$ from the unconstrained parameters.
     * @details The first threshold is \f$ \alpha_0 = \mathtt{params}(k) \f$ and, for \f$ m \ge 1 \f$,
     * \f$ \alpha_m = \alpha_{m-1} + \exp\{\mathrm{clamp}(\mathtt{params}(k+m), -15, 15)\} \f$, which guarantees a strictly increasing
     * sequence.
     * @param params Full parameter vector \f$ (\boldsymbol\beta, \alpha_0, \gamma_1,\dots,\gamma_{M-2}) \f$.
     * @param k Number of design columns (offset of the threshold block in `params`).
     * @param alpha Output vector of length \f$ M-1 \f$ (must be pre-sized by the caller), overwritten with the thresholds.
     */
    void reconstruct_thresholds(const Eigen::VectorXd& params, int k, Eigen::VectorXd& alpha) const {
        int num_thresholds = num_categories - 1;
        alpha(0) = params(k);
        for (int m = 1; m < num_thresholds; ++m) {
            double gamma = std::max(-15.0, std::min(15.0, params(k + m)));
            alpha(m) = alpha(m - 1) + std::exp(gamma);
        }
    }

    /**
     * @brief Evaluates the total negative log-likelihood.
     * @param params Parameter vector \f$ (\boldsymbol\beta, \alpha_0, \gamma_1,\dots,\gamma_{M-2}) \f$ of length \f$ k+M-1 \f$.
     * @return The value \f$ -\sum_i \log\max\{10^{-15}, P(y_i)\} \f$, or `1e10` if the sum is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        int num_thresholds = num_categories - 1;

        Eigen::VectorXd beta = params.head(k);
        Eigen::VectorXd alpha(num_thresholds);
        reconstruct_thresholds(params, static_cast<int>(k), alpha);

        Eigen::VectorXd eta = X * beta;
        // accumulate -log P(y_i) over observations
        double total_neg_ll = 0.0;

        for (long i = 0; i < n; ++i) {
            int cat = static_cast<int>(y(i));
            double eta_i = eta(i);

            double cdf_upper = (cat == num_categories - 1) ? 1.0 : evaluate_ordinal_cdf(alpha(cat) - eta_i, mu_link);
            double cdf_lower = (cat == 0) ? 0.0 : evaluate_ordinal_cdf(alpha(cat - 1) - eta_i, mu_link);

            double prob = cdf_upper - cdf_lower;
            if (prob < 1e-15) prob = 1e-15;

            total_neg_ll -= std::log(prob);
        }
        return std::isfinite(total_neg_ll) ? total_neg_ll : 1e10;
    }

    /**
     * @brief Analytical gradient of the negative log-likelihood with respect to \f$ (\boldsymbol\beta, \alpha_0, \boldsymbol\gamma) \f$.
     * @details The coefficient part is \f$ \partial\mathrm{NLL}/\partial\boldsymbol\beta = \sum_i (-\partial_u-\partial_l)\,\mathbf{x}_i \f$; the threshold part
     * is obtained by accumulating the contributions to each \f$ \alpha_m \f$ and applying the reverse cumulative sum
     * \f$ \partial/\partial\gamma_m = e^{\gamma_m}\sum_{l\ge m}\partial/\partial\alpha_l \f$ (\f$ m\ge1 \f$) and
     * \f$ \partial/\partial\alpha_0 = \sum_{l\ge0}\partial/\partial\alpha_l \f$.
     * @param params Parameter vector of length \f$ k+M-1 \f$.
     * @return Gradient vector of the same length. Non-finite terms are set to 0 (per observation and per final component).
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        int num_thresholds = num_categories - 1;

        Eigen::VectorXd beta = params.head(k);
        Eigen::VectorXd alpha(num_thresholds);
        reconstruct_thresholds(params, static_cast<int>(k), alpha);

        Eigen::VectorXd eta = X * beta;

        // accumulate derivatives with respect to beta and the thresholds alpha
        Eigen::VectorXd dNLL_dbeta = Eigen::VectorXd::Zero(k);
        Eigen::VectorXd dNLL_dalpha = Eigen::VectorXd::Zero(num_thresholds);

        for (long i = 0; i < n; ++i) {
            int cat = static_cast<int>(y(i));
            double eta_i = eta(i);

            double arg_upper = (cat == num_categories - 1) ? 0.0 : alpha(cat) - eta_i;
            double arg_lower = (cat == 0) ? 0.0 : alpha(cat - 1) - eta_i;

            double cdf_upper = (cat == num_categories - 1) ? 1.0 : evaluate_ordinal_cdf(arg_upper, mu_link);
            double cdf_lower = (cat == 0) ? 0.0 : evaluate_ordinal_cdf(arg_lower, mu_link);

            double prob = std::max(1e-15, cdf_upper - cdf_lower);

            double pdf_upper = (cat == num_categories - 1) ? 0.0 : evaluate_ordinal_pdf(arg_upper, mu_link);
            double pdf_lower = (cat == 0) ? 0.0 : evaluate_ordinal_pdf(arg_lower, mu_link);

            double dNLL_darg_upper = -pdf_upper / prob;
            double dNLL_darg_lower =  pdf_lower / prob;

            if (!std::isfinite(dNLL_darg_upper)) dNLL_darg_upper = 0.0;
            if (!std::isfinite(dNLL_darg_lower)) dNLL_darg_lower = 0.0;

            double dNLL_deta_i = -dNLL_darg_upper - dNLL_darg_lower;
            dNLL_dbeta += X.row(i).transpose() * dNLL_deta_i;

            if (cat < num_categories - 1) dNLL_dalpha(cat) += dNLL_darg_upper;
            if (cat > 0)                  dNLL_dalpha(cat - 1) += dNLL_darg_lower;
        }

        Eigen::VectorXd grad(k + num_thresholds);
        grad.head(k) = dNLL_dbeta;

        // chain rule alpha -> (alpha_0, gamma) via reverse cumulative sums
        double running_sum_dalpha = 0.0;
        for (int m = num_thresholds - 1; m >= 1; --m) {
            running_sum_dalpha += dNLL_dalpha(m);
            double gamma_m = std::max(-15.0, std::min(15.0, params(k + m)));
            grad(k + m) = running_sum_dalpha * std::exp(gamma_m);
        }

        running_sum_dalpha += dNLL_dalpha(0);
        grad(k) = running_sum_dalpha;

        for (int i = 0; i < grad.size(); ++i) {
            if (!std::isfinite(grad(i))) {
                grad(i) = 0.0;
            }
        }

        return grad;
    }

    /**
     * @brief Analytical Hessian of the negative log-likelihood, assembled in threshold coordinates and then re-mapped.
     * @details Per observation the second derivatives with respect to the CDF arguments are
     * \f$ g_{uu} = (-F''(u)P + f(u)^2)/P^2 \f$, \f$ g_{ll} = (F''(l)P + f(l)^2)/P^2 \f$ and \f$ g_{ul} = -f(u)f(l)/P^2 \f$,
     * giving \f$ \partial^2\mathrm{NLL}_i/\partial\eta^2 = g_{uu}+g_{ll}+2g_{ul} \f$, the \f$ \boldsymbol\alpha \f$ diagonal terms
     * \f$ g_{uu} \f$ (at \f$ \alpha_{y_i} \f$), \f$ g_{ll} \f$ (at \f$ \alpha_{y_i-1} \f$), the neighbour term \f$ g_{ul} \f$ and the
     * cross terms \f$ \mathbf{x}_i(-g_{uu}-g_{ul}) \f$, \f$ \mathbf{x}_i(-g_{ll}-g_{ul}) \f$ with the thresholds. The blocks
     * \f$ H_{\beta\beta},\,H_{\beta\alpha},\,H_{\alpha\alpha} \f$ are then transformed with
     * \f$ J_{mj} = \partial\alpha_m/\partial\gamma_j \f$ (\f$ J_{m0}=1 \f$, \f$ J_{mj}=e^{\gamma_j} \f$ for \f$ 1\le j\le m \f$):
     * \f$ H_{\beta\gamma} = H_{\beta\alpha}J \f$ and
     * \f$ H_{\gamma\gamma} = J^\top H_{\alpha\alpha}J + \mathrm{diag}\bigl(0,\, e^{\gamma_j}\sum_{l\ge j}\partial\mathrm{NLL}/\partial\alpha_l\bigr) \f$,
     * which is the exact Hessian in \f$ (\boldsymbol\beta,\alpha_0,\boldsymbol\gamma) \f$ coordinates.
     * @note The code then multiplies the whole matrix on both sides by \f$ J_{\mathrm{tot}}^{-\top} \f$ and
     *       \f$ J_{\mathrm{tot}}^{-1} \f$ (with \f$ J^{-1}_{mm}=e^{-\gamma_m} \f$, \f$ J^{-1}_{m,m-1}=-e^{-\gamma_m} \f$, \f$ J^{-1}_{00}=1 \f$),
     *       which returns it to \f$ (\boldsymbol\beta,\boldsymbol\alpha) \f$ coordinates: the threshold block becomes
     *       \f$ H_{\alpha\alpha} + J^{-\top} D J^{-1} \f$ and the cross block \f$ H_{\beta\alpha} \f$. The extra term
     *       \f$ D \f$ is proportional to the gradient \f$ \partial\mathrm{NLL}/\partial\boldsymbol\alpha \f$, so **at a stationary point
     *       (the maximum-likelihood estimate) the returned matrix is exactly the observed Hessian in
     *       \f$ (\boldsymbol\beta,\boldsymbol\alpha) \f$ coordinates**, which is what the solver uses (`MleSolver` evaluates it at
     *       the optimum to get standard errors of the thresholds \f$ \boldsymbol\alpha \f$). Away from the optimum it is *not* the
     *       Hessian with respect to the parameter vector the gradient refers to (nor exactly the \f$ \boldsymbol\alpha \f$-Hessian).
     * @param params Parameter vector of length \f$ k+M-1 \f$.
     * @return Symmetric \f$ (k+M-1)\times(k+M-1) \f$ matrix; non-finite entries are set to 0, then it is symmetrised as
     *         \f$ (H+H^\top)/2 \f$. No ridge or PSD projection is applied.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        const long k = X.cols();
        int num_thresholds = num_categories - 1;
        int total_params = static_cast<int>(k + num_thresholds);

        Eigen::VectorXd beta = params.head(k);
        Eigen::VectorXd alpha(num_thresholds);
        reconstruct_thresholds(params, static_cast<int>(k), alpha);

        Eigen::VectorXd eta = X * beta;

        // accumulate Hessian blocks in (beta, alpha) coordinates
        Eigen::MatrixXd H_beta_beta = Eigen::MatrixXd::Zero(k, k);
        Eigen::MatrixXd H_beta_alpha = Eigen::MatrixXd::Zero(k, num_thresholds);
        Eigen::MatrixXd H_alpha_alpha = Eigen::MatrixXd::Zero(num_thresholds, num_thresholds);
        Eigen::VectorXd dNLL_dalpha = Eigen::VectorXd::Zero(num_thresholds);

        for (long i = 0; i < n; ++i) {
            int cat = static_cast<int>(y(i));
            double eta_i = eta(i);

            double arg_upper = (cat == num_categories - 1) ? 0.0 : alpha(cat) - eta_i;
            double arg_lower = (cat == 0) ? 0.0 : alpha(cat - 1) - eta_i;

            double cdf_upper = (cat == num_categories - 1) ? 1.0 : evaluate_ordinal_cdf(arg_upper, mu_link);
            double cdf_lower = (cat == 0) ? 0.0 : evaluate_ordinal_cdf(arg_lower, mu_link);
            double prob = std::max(1e-15, cdf_upper - cdf_lower);

            double pdf_upper = (cat == num_categories - 1) ? 0.0 : evaluate_ordinal_pdf(arg_upper, mu_link);
            double pdf_lower = (cat == 0) ? 0.0 : evaluate_ordinal_pdf(arg_lower, mu_link);

            double d2cdf_upper = (cat == num_categories - 1) ? 0.0 : evaluate_ordinal_d2_cdf(arg_upper, mu_link);
            double d2cdf_lower = (cat == 0) ? 0.0 : evaluate_ordinal_d2_cdf(arg_lower, mu_link);

            double dg_u = -pdf_upper / prob;
            double dg_l =  pdf_lower / prob;

            if (cat < num_categories - 1) dNLL_dalpha(cat) += dg_u;
            if (cat > 0)                  dNLL_dalpha(cat - 1) += dg_l;

            double d2g_uu = (-d2cdf_upper * prob + pdf_upper * pdf_upper) / (prob * prob);
            double d2g_ll = ( d2cdf_lower * prob + pdf_lower * pdf_lower) / (prob * prob);
            double d2g_ul = (-pdf_upper * pdf_lower) / (prob * prob);

            double d2NLL_deta2 = d2g_uu + d2g_ll + 2.0 * d2g_ul;
            H_beta_beta += X.row(i).transpose() * X.row(i) * d2NLL_deta2;

            if (cat < num_categories - 1) {
                H_alpha_alpha(cat, cat) += d2g_uu;
                H_beta_alpha.col(cat) += X.row(i).transpose() * (-d2g_uu - d2g_ul);
            }
            if (cat > 0) {
                H_alpha_alpha(cat - 1, cat - 1) += d2g_ll;
                H_beta_alpha.col(cat - 1) += X.row(i).transpose() * (-d2g_ll - d2g_ul);
            }

            if (cat > 0 && cat < num_categories - 1) {
                H_alpha_alpha(cat, cat - 1) += d2g_ul;
                H_alpha_alpha(cat - 1, cat) += d2g_ul;
            }
        }

        // Jacobian dalpha/dgamma
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_thresholds, num_thresholds);
        for (int m = 0; m < num_thresholds; ++m) {
            J(m, 0) = 1.0;
            for (int j = 1; j <= m; ++j) {
                double gamma_j = std::max(-15.0, std::min(15.0, params(k + j)));
                J(m, j) = std::exp(gamma_j);
            }
        }

        Eigen::MatrixXd hess = Eigen::MatrixXd::Zero(total_params, total_params);
        hess.topLeftCorner(k, k) = H_beta_beta;

        Eigen::MatrixXd H_beta_gamma = H_beta_alpha * J;
        hess.topRightCorner(k, num_thresholds) = H_beta_gamma;
        hess.bottomLeftCorner(num_thresholds, k) = H_beta_gamma.transpose();

        Eigen::MatrixXd H_gamma_gamma = J.transpose() * H_alpha_alpha * J;

        double cum_grad = 0.0;
        for (int j = num_thresholds - 1; j >= 1; --j) {
            cum_grad += dNLL_dalpha(j);
            double gamma_j = std::max(-15.0, std::min(15.0, params(k + j)));

            H_gamma_gamma(j, j) += cum_grad * std::exp(gamma_j);
        }

        hess.bottomRightCorner(num_thresholds, num_thresholds) = H_gamma_gamma;
        // inverse Jacobian and back-transformation of the whole matrix
        Eigen::MatrixXd J_inv = Eigen::MatrixXd::Zero(num_thresholds, num_thresholds);
        J_inv(0, 0) = 1.0;
        for (int m = 1; m < num_thresholds; ++m) {
            double gamma_m = std::max(-15.0, std::min(15.0, params(k + m)));
            double exp_gamma = std::exp(gamma_m);
            J_inv(m, m - 1) = -1.0 / exp_gamma;
            J_inv(m, m)     =  1.0 / exp_gamma;
        }

        Eigen::MatrixXd J_total_inv = Eigen::MatrixXd::Identity(total_params, total_params);
        J_total_inv.bottomRightCorner(num_thresholds, num_thresholds) = J_inv;
        hess = J_total_inv.transpose() * hess * J_total_inv;

        for (int i = 0; i < hess.rows(); ++i) {
            for (int j = 0; j < hess.cols(); ++j) {
                if (!std::isfinite(hess(i, j))) {
                    hess(i, j) = 0.0;
                }
            }
        }

        // symmetrise
        hess = (hess + hess.transpose()) * 0.5;
        return hess;
    }

};

/**
 * @brief Negative log-likelihood of the Generalized Gamma (Prentice 1974) accelerated failure time model with right censoring, left truncation and observation weights.
 *
 * @details Models a positive survival time \f$ T_i \f$ whose logarithm follows the Prentice parameterisation of the
 * Generalized Gamma distribution (as in the R package `flexsurv`) with location \f$ \mu_i \f$, scale \f$ \sigma_i > 0 \f$
 * and shape \f$ Q_i \f$. All three parameters have their own linear predictor,
 * \f[ \eta^{\mu}_i = \theta_0 + \sum_{c \in \mathtt{mx\_mu}} x_{ic}\,\beta^{\mu}_c, \quad
 *     \eta^{\sigma}_i = \theta_1 + \sum_{c \in \mathtt{mx\_sigma}} x_{ic}\,\beta^{\sigma}_c, \quad
 *     \eta^{Q}_i = \theta_2 + \sum_{c \in \mathtt{mx\_Q}} x_{ic}\,\beta^{Q}_c , \f]
 * \f[ \mu_i = g_\mu^{-1}(\eta^{\mu}_i), \qquad \sigma_i = g_\sigma^{-1}(\eta^{\sigma}_i), \qquad Q_i = g_Q^{-1}(\eta^{Q}_i), \f]
 * where the inverse links come from `apply_inverse_link()`: `Log` gives \f$ e^{\eta} \f$, `Inverse` gives \f$ 1/\eta \f$ and
 * **every other link type behaves as the identity**. Defaults: identity for \f$ \mu \f$ and \f$ Q \f$, `Log` for \f$ \sigma \f$.
 * The design matrix is shared; each predictor selects its columns through the integer index vectors `mx_mu`,
 * `mx_sigma`, `mx_Q`. The intercepts \f$ \theta_0,\theta_1,\theta_2 \f$ are explicit parameters (not columns of `X`).
 * With \f$ w = (\log t - \mu)/\sigma \f$ and \f$ q = 1/Q^2 \f$ the log-density (`gengamma::density`) is
 * \f[ \log f(t) = -\log(\sigma t) + \log|Q| + q\log q + q\,(Qw - e^{Qw}) - \log\Gamma(q), \f]
 * and the survival function (`gengamma::cdf` upper tail) is the regularised incomplete gamma function
 * \f$ S(t) = \Gamma_{\mathrm{upper}}(q,\, q e^{Qw}) \f$ for \f$ Q>0 \f$ and \f$ \Gamma_{\mathrm{lower}}(q,\, q e^{Qw}) \f$ for \f$ Q<0 \f$.
 *
 * ### Parameter vector layout
 * | Index                                                      | Meaning                                                          | Scale |
 * |------------------------------------------------------------|------------------------------------------------------------------|-------|
 * | `0`                                                        | intercept \f$ \theta_0 \f$ of \f$ \eta^{\mu} \f$                  | predictor scale of \f$ \mu \f$ |
 * | `1`                                                        | intercept \f$ \theta_1 \f$ of \f$ \eta^{\sigma} \f$               | predictor scale of \f$ \sigma \f$ (log scale for the default `Log` link) |
 * | `2`                                                        | intercept \f$ \theta_2 \f$ of \f$ \eta^{Q} \f$                    | predictor scale of \f$ Q \f$ |
 * | `3 .. 2+m1`                                                | \f$ \boldsymbol\beta^{\mu} \f$ for the columns `mx_mu` (in this order) | linear predictor |
 * | `3+m1 .. 2+m1+m2`                                          | \f$ \boldsymbol\beta^{\sigma} \f$ for the columns `mx_sigma`      | linear predictor |
 * | `3+m1+m2 .. 2+m1+m2+m3`                                    | \f$ \boldsymbol\beta^{Q} \f$ for the columns `mx_Q`               | linear predictor |
 *
 * with \f$ m_1, m_2, m_3 \f$ the lengths of `mx_mu`, `mx_sigma`, `mx_Q`. The total length is \f$ 3+m_1+m_2+m_3 \f$.
 *
 * ### Log-likelihood
 * Each row of `Y` supplies a left-truncation (entry) time \f$ s_i \f$ (column 0), the observed stop time \f$ t_i \f$ (column 1)
 * and the event indicator \f$ d_i \f$ (column 3); column 2 is never used. Rows with \f$ s_i \ge t_i - 10^{-5} \f$ are skipped
 * entirely. Otherwise
 * - event (\f$ d_i > 0.5 \f$): \f$ \ell_i = \log f(t_i) \f$;
 * - right-censored: \f$ \ell_i = \log\max\{S(t_i), 10^{-15}\} \f$;
 * - if \f$ s_i > 10^{-8} \f$ (delayed entry): \f$ \ell_i \mathrel{-}= \log\max\{S(s_i), 10^{-15}\} \f$.
 *
 * The functor returns \f$ -\sum_i w_i\,\ell_i \f$ with the observation weights \f$ w_i \f$. In the NLL the shape is
 * bounded away from zero: \f$ |Q_i| < 10^{-4} \f$ is replaced by \f$ \pm 10^{-4} \f$ (sign of \f$ Q_i \f$, positive for 0).
 *
 * ### Derivatives
 * **The gradient and the Hessian are computed by finite differences, not analytically.** For each observation the
 * derivative of \f$ \ell_i \f$ is taken with respect to the three predictors \f$ (\eta^{\mu},\eta^{\sigma},\eta^{Q}) \f$
 * (the inverse links, the \f$ Q \f$ bound and the censoring/truncation terms are inside the differenced function), and
 * mapped to the parameters by the chain rule \f$ \partial\mathrm{NLL}/\partial\theta_r = \sum_i \partial\mathrm{NLL}_i/\partial\eta^{(r)}_i \f$,
 * \f$ \partial\mathrm{NLL}/\partial\beta^{(r)}_c = \sum_i x_{ic}\,\partial\mathrm{NLL}_i/\partial\eta^{(r)}_i \f$.
 * The Hessian uses nested central differences of the per-observation NLL with respect to the three predictors, giving a
 * \f$ 3\times3 \f$ matrix \f$ H^{\eta}_i \f$ that is mapped with products of the design entries (see `hessian()`).
 *
 * ### Numerical safeguards
 * - `operator()`: returns `1e10` if `params` has fewer than 3 entries, immediately returns `1e10` if any non-skipped
 *   observation has \f$ \sigma_i \le 10^{-8} \f$ or a non-finite \f$ \sigma_i \f$, \f$ \mu_i \f$ or \f$ Q_i \f$, and
 *   returns `1e10` if the total is not finite.
 * - Survival values are floored at `1e-15` (and the underlying CDF is clipped to \f$ [10^{-15}, 1-10^{-15}] \f$).
 * - Inside the finite-difference helpers an invalid parameter point returns \f$ -10^{10} \f$ (log-likelihood, gradient) or
 *   \f$ +10^{10} \f$ (NLL, Hessian), which produces huge difference quotients rather than an exception.
 * - Step sizes: gradient \f$ \varepsilon = 10^{-6}(|\eta|+1) \f$ per predictor with \f$ \varepsilon_Q \ge 2\cdot10^{-4} \f$;
 *   Hessian absolute \f$ \varepsilon = 10^{-5} \f$ for both difference levels.
 *
 * @note The object stores **non-owning references** to `X`, `Y`, `weights`, `mx_mu`, `mx_sigma` and `mx_Q`, which must
 *       outlive the functor. No validation is performed: \f$ Y \f$ needs at least 4 columns, the column indices in
 *       `mx_*` must be valid columns of `X`, \f$ Y \f$ and `weights` must have \f$ n \f$ rows, and `params` must have exactly
 *       \f$ 3+m_1+m_2+m_3 \f$ entries (only `operator()` checks the minimum size 3; `gradient()` and `hessian()` do not).
 * @note Oddities: (i) with the bound \f$ |Q|\ge 10^{-4} \f$, \f$ Q = \pm10^{-4} \f$ exactly takes the Prentice branch in
 *       `gengamma::density` (threshold `abs_q < 1e-4`) but the log-normal limiting branch in `gengamma::cdf` (threshold
 *       `abs_q > 1e-4`); (ii) `hessian()` is not declared `const`; (iii) the gradient step uses relative increments
 *       while the Hessian uses absolute ones and has no \f$ Q \f$ floor on the step, so differences near \f$ Q=0 \f$ may straddle
 *       the \f$ \pm10^{-4} \f$ clamp; (iv) neither `gradient()` nor `hessian()` masks non-finite values.
 * @see LogNormalLoss, WeibullTLoss, LogLogisticTLoss for the other accelerated failure time families.
 *
 * @ingroup glm_ll_survival
 */
struct GeneralizedGammaLoss {
    const Eigen::MatrixXd& X;  ///< Shared design matrix (\f$ n \times p \f$), columns selected by `mx_mu`, `mx_sigma`, `mx_Q`.
    const Eigen::MatrixXd& Y;  ///< Survival data (\f$ n \times \ge 4 \f$): column 0 = entry/left-truncation time, column 1 = stop time, column 3 = event indicator (column 2 unused).
    const Eigen::VectorXd& weights;  ///< Non-negative observation weights \f$ w_i \f$ (length \f$ n \f$) multiplying each log-likelihood contribution.
    const Eigen::VectorXi& mx_mu;  ///< Column indices of `X` entering the \f$ \mu \f$ predictor (length \f$ m_1 \f$).
    const Eigen::VectorXi& mx_sigma;  ///< Column indices of `X` entering the \f$ \sigma \f$ predictor (length \f$ m_2 \f$).
    const Eigen::VectorXi& mx_Q;  ///< Column indices of `X` entering the \f$ Q \f$ predictor (length \f$ m_3 \f$).

    const LinkType link_mu;  ///< Link type of the location \f$ \mu \f$ (only `Log`, `Inverse`, otherwise identity).
    const LinkType link_sigma;  ///< Link type of the scale \f$ \sigma \f$ (only `Log`, `Inverse`, otherwise identity).
    const LinkType link_Q;  ///< Link type of the shape \f$ Q \f$ (only `Log`, `Inverse`, otherwise identity).

    gengamma::density log_pdf;  ///< Functor returning the Generalized Gamma log-density \f$ \log f(t;\mu,\sigma,Q) \f$.
    gengamma::cdf survival;  ///< Functor returning the Generalized Gamma survival function \f$ S(t;\mu,\sigma,Q) \f$ (upper tail, not on log scale).

    /**
     * @brief Builds the loss functor (all data are referenced, not copied) and the density / survival helpers.
     * @param exog Design matrix \f$ X \f$ (\f$ n \times p \f$).
     * @param endog Survival matrix \f$ Y \f$ (\f$ n \times 4 \f$ or more): entry time, stop time, (unused), event indicator.
     * @param w Observation weights (length \f$ n \f$).
     * @param mx_mu_cols Indices of the columns of `exog` used in the \f$ \mu \f$ predictor.
     * @param mx_sigma_cols Indices of the columns of `exog` used in the \f$ \sigma \f$ predictor.
     * @param mx_Q_cols Indices of the columns of `exog` used in the \f$ Q \f$ predictor.
     * @param l_mu Link for \f$ \mu \f$ (default `Identity`).
     * @param l_sigma Link for \f$ \sigma \f$ (default `Log`).
     * @param l_Q Link for \f$ Q \f$ (default `Identity`).
     * @pre All referenced objects outlive the functor and the index vectors contain valid column indices of `exog`.
     */
    GeneralizedGammaLoss(const Eigen::MatrixXd& exog, const Eigen::MatrixXd& endog, const Eigen::VectorXd& w,
                         const Eigen::VectorXi& mx_mu_cols, const Eigen::VectorXi& mx_sigma_cols, const Eigen::VectorXi& mx_Q_cols,
                         LinkType l_mu = LinkType::Identity,
                         LinkType l_sigma = LinkType::Log,
                         LinkType l_Q = LinkType::Identity)
        : X(exog), Y(endog), weights(w),
          mx_mu(mx_mu_cols), mx_sigma(mx_sigma_cols), mx_Q(mx_Q_cols),
          link_mu(l_mu), link_sigma(l_sigma), link_Q(l_Q),
          log_pdf(), survival(false, false) {}

    /**
     * @brief Evaluates the total weighted negative log-likelihood.
     * @param params Parameter vector \f$ (\theta_0,\theta_1,\theta_2,\boldsymbol\beta^{\mu},\boldsymbol\beta^{\sigma},\boldsymbol\beta^{Q}) \f$ of
     *        length \f$ 3+m_1+m_2+m_3 \f$.
     * @return The value \f$ -\sum_i w_i\,\ell_i \f$ over the non-skipped rows, or `1e10` if `params` has fewer than 3 entries, if an
     *         invalid \f$ (\mu_i,\sigma_i,Q_i) \f$ is met, or if the total is not finite.
     */
    double operator()(const Eigen::VectorXd& params) const {
        if (params.size() < 3) return 1e10;
        const long n = X.rows();

        const double init_mu    = params.coeff(0);
        const double init_sigma = params.coeff(1);
        const double init_Q     = params.coeff(2);

        Eigen::VectorXd beta = params.tail(params.size() - 3);
        double total_neg_loglik = 0.0;

        for (long i = 0; i < n; ++i) {
            const double start  = Y.coeff(i, 0);
            const double time1  = Y.coeff(i, 1);
            const double status = Y.coeff(i, 3);

            if (start >= time1 - 1e-5) {
                continue;
            }

            double pred_mu    = init_mu;
            double pred_sigma = init_sigma;
            double pred_Q     = init_Q;

            // linear predictors of the three parameters
            int beta_idx = 0;
            for (int j = 0; j < mx_mu.size(); ++j)    { pred_mu    += X.coeff(i, mx_mu.coeff(j))    * beta.coeff(beta_idx++); }
            for (int j = 0; j < mx_sigma.size(); ++j) { pred_sigma += X.coeff(i, mx_sigma.coeff(j)) * beta.coeff(beta_idx++); }
            for (int j = 0; j < mx_Q.size(); ++j)     { pred_Q     += X.coeff(i, mx_Q.coeff(j))     * beta.coeff(beta_idx++); }

            double mu_i    = apply_inverse_link(pred_mu, link_mu);
            double sigma_i = apply_inverse_link(pred_sigma, link_sigma);
            double Q_i     = apply_inverse_link(pred_Q, link_Q);

            if (sigma_i <= 1e-8 || !std::isfinite(sigma_i)) return 1e10;
            if (!std::isfinite(mu_i) || !std::isfinite(Q_i)) return 1e10;

            if (std::abs(Q_i) < 1e-4) Q_i = (Q_i >= 0) ? 1e-4 : -1e-4;

            // event: log-density; right-censored: log-survival
            double loglik_i = 0.0;
            if (status > 0.5) {
                loglik_i = log_pdf(time1, mu_i, sigma_i, Q_i);
            } else {
                double s_min = survival(time1, mu_i, sigma_i, Q_i);
                loglik_i = std::log(std::max(s_min, 1e-15));
            }

            if (start > 1e-8) {
                double pobs = survival(start, mu_i, sigma_i, Q_i);
                loglik_i -= std::log(std::max(pobs, 1e-15));
            }
            total_neg_loglik -= loglik_i * weights.coeff(i);
        }

        return std::isfinite(total_neg_loglik) ? total_neg_loglik : 1e10;
    }

    /**
     * @brief Gradient of the negative log-likelihood by per-observation central finite differences in the predictor space.
     * @details For each non-skipped row the derivative of \f$ \ell_i \f$ with respect to \f$ \eta^{\mu},\eta^{\sigma},\eta^{Q} \f$ is
     * \f$ (\ell_i(\eta+\varepsilon)-\ell_i(\eta-\varepsilon))/(2\varepsilon) \f$ with
     * \f$ \varepsilon = 10^{-6}(|\eta|+1) \f$ (and \f$ \varepsilon_Q = \max\{10^{-6}(|\eta^Q|+1),\,2\cdot10^{-4}\} \f$ for the shape, so
     * that the step crosses the \f$ |Q| < 10^{-4} \f$ clamp). The per-row NLL derivative is
     * \f$ -w_i\,\partial\ell_i/\partial\eta \f$; the three intercept entries are its sums over rows, and each
     * coefficient entry is the dot product of the corresponding design column with these per-row values.
     * @param params Parameter vector of length \f$ 3+m_1+m_2+m_3 \f$ (not size-checked).
     * @return Gradient of the same length and layout as `params`. Non-finite values are not masked.
     */
    Eigen::VectorXd gradient(const Eigen::VectorXd& params) const {
        const long n = X.rows();
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());

        const double init_mu    = params.coeff(0);
        const double init_sigma = params.coeff(1);
        const double init_Q     = params.coeff(2);

        Eigen::VectorXd beta = params.tail(params.size() - 3);

        Eigen::VectorXd dL_deta_mu    = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dL_deta_sigma = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd dL_deta_Q     = Eigen::VectorXd::Zero(n);

        const double rel_eps = 1e-6;

        for (long i = 0; i < n; ++i) {
            const double start  = Y.coeff(i, 0);
            const double time1  = Y.coeff(i, 1);
            const double status = Y.coeff(i, 3);

            if (start >= time1 - 1e-5) continue;

            double pred_mu    = init_mu;
            double pred_sigma = init_sigma;
            double pred_Q     = init_Q;

            // linear predictors of the three parameters
            int beta_idx = 0;
            for (int j = 0; j < mx_mu.size(); ++j)    { pred_mu    += X.coeff(i, mx_mu.coeff(j))    * beta.coeff(beta_idx++); }
            for (int j = 0; j < mx_sigma.size(); ++j) { pred_sigma += X.coeff(i, mx_sigma.coeff(j)) * beta.coeff(beta_idx++); }
            for (int j = 0; j < mx_Q.size(); ++j)     { pred_Q     += X.coeff(i, mx_Q.coeff(j))     * beta.coeff(beta_idx++); }

            auto eval_row_eta = [&](double eta_m, double eta_s, double eta_q) -> double {
                double mu    = apply_inverse_link(eta_m, link_mu);
                double sigma = apply_inverse_link(eta_s, link_sigma);
                double Q     = apply_inverse_link(eta_q, link_Q);

                if (sigma <= 1e-8 || !std::isfinite(sigma) || !std::isfinite(mu) || !std::isfinite(Q)) return -1e10;

                if (std::abs(Q) < 1e-4) Q = (Q >= 0) ? 1e-4 : -1e-4;

                double ll = (status > 0.5) ? log_pdf(time1, mu, sigma, Q)
                                        : std::log(std::max(survival(time1, mu, sigma, Q), 1e-15));
                if (start > 1e-8) {
                    ll -= std::log(std::max(survival(start, mu, sigma, Q), 1e-15));
                }
                return ll;
            };

            // central differences of the row log-likelihood in each predictor
            double eps_m = rel_eps * (std::abs(pred_mu) + 1.0);
            double eps_s = rel_eps * (std::abs(pred_sigma) + 1.0);
            double eps_q = rel_eps * (std::abs(pred_Q) + 1.0);

            eps_q = std::max(eps_q, 2e-4);

            double d_eta_mu    = (eval_row_eta(pred_mu + eps_m, pred_sigma, pred_Q) - eval_row_eta(pred_mu - eps_m, pred_sigma, pred_Q)) / (2.0 * eps_m);
            double d_eta_sigma = (eval_row_eta(pred_mu, pred_sigma + eps_s, pred_Q) - eval_row_eta(pred_mu, pred_sigma - eps_s, pred_Q)) / (2.0 * eps_s);
            double d_eta_Q     = (eval_row_eta(pred_mu, pred_sigma, pred_Q + eps_q) - eval_row_eta(pred_mu, pred_sigma, pred_Q - eps_q)) / (2.0 * eps_q);

            double dw = weights.coeff(i);

            dL_deta_mu.coeffRef(i)    = -d_eta_mu    * dw;
            dL_deta_sigma.coeffRef(i) = -d_eta_sigma * dw;
            dL_deta_Q.coeffRef(i)     = -d_eta_Q     * dw;
        }

        // chain rule to intercepts and coefficients
        grad(0) = dL_deta_mu.sum();
        grad(1) = dL_deta_sigma.sum();
        grad(2) = dL_deta_Q.sum();

        int p_idx = 3;
        for (int j = 0; j < mx_mu.size(); ++j)    { grad(p_idx++) = X.col(mx_mu.coeff(j)).dot(dL_deta_mu); }
        for (int j = 0; j < mx_sigma.size(); ++j) { grad(p_idx++) = X.col(mx_sigma.coeff(j)).dot(dL_deta_sigma); }
        for (int j = 0; j < mx_Q.size(); ++j)     { grad(p_idx++) = X.col(mx_Q.coeff(j)).dot(dL_deta_Q); }

        return grad;
    }

    /**
     * @brief Hessian of the negative log-likelihood by nested central finite differences (non-`const` member).
     * @details For each non-skipped row the \f$ 3\times3 \f$ matrix \f$ H^{\eta}_i \f$ of second derivatives of the per-row
     * NLL with respect to \f$ (\eta^{\mu},\eta^{\sigma},\eta^{Q}) \f$ is obtained by differencing a central-difference
     * gradient (absolute step \f$ \varepsilon = 10^{-5} \f$ at both levels) and is symmetrised, \f$ (H^\eta+H^{\eta\top})/2 \f$.
     * It is then accumulated with the weight \f$ w_i \f$ into
     * - the \f$ 3\times3 \f$ intercept block,
     * - the intercept/coefficient blocks: entry \f$ (r, \text{coefficient } c \text{ of predictor } s) \f$ gets
     *   \f$ w_i H^{\eta}_{rs}\,x_{ic} \f$,
     * - the coefficient/coefficient blocks: \f$ w_i H^{\eta}_{rs}\,x_{ic}x_{id} \f$ for predictor pairs \f$ (r,s) \f$
     *   (\f$ \mu\mu \f$, \f$ \sigma\sigma \f$, \f$ QQ \f$, \f$ \mu\sigma \f$, \f$ \mu Q \f$, \f$ \sigma Q \f$).
     *
     * Only the upper triangle is accumulated off the diagonal blocks; the lower triangle is finally overwritten by the transpose
     * of the upper one.
     * @param params Parameter vector of length \f$ 3+m_1+m_2+m_3 \f$ (not size-checked).
     * @return Symmetric \f$ p\times p \f$ matrix (\f$ p \f$ = `params.size()`). No ridge, finite masking or PSD projection is
     *         applied; invalid parameter points inside the differences contribute \f$ 10^{10} \f$ terms.
     */
    Eigen::MatrixXd hessian(const Eigen::VectorXd& params) {
        const long n = X.rows();
        const long p = params.size();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(p, p);

        const double init_mu    = params.coeff(0);
        const double init_sigma = params.coeff(1);
        const double init_Q     = params.coeff(2);

        Eigen::VectorXd beta = params.tail(params.size() - 3);

        const double eps = 1e-5;

        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> map_mu(mx_mu.size());
        for (Eigen::Index j = 0; j < mx_mu.size(); ++j) map_mu[j] = 3 + j;

        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> map_sigma(mx_sigma.size());
        for (Eigen::Index j = 0; j < mx_sigma.size(); ++j) map_sigma[j] = 3 + mx_mu.size() + j;

        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> map_Q(mx_Q.size());
        for (Eigen::Index j = 0; j < mx_Q.size(); ++j) map_Q[j] = 3 + mx_mu.size() + mx_sigma.size() + j;

        for (long i = 0; i < n; ++i) {
            const double start  = Y.coeff(i, 0);
            const double time1  = Y.coeff(i, 1);
            const double status = Y.coeff(i, 3);

            if (start >= time1 - 1e-5) continue;

            double pred_mu    = init_mu;
            double pred_sigma = init_sigma;
            double pred_Q     = init_Q;

            // linear predictors of the three parameters
            int beta_idx = 0;
            for (Eigen::Index j = 0; j < mx_mu.size(); ++j)    { pred_mu    += X.coeff(i, mx_mu.coeff(j))    * beta.coeff(beta_idx++); }
            for (Eigen::Index j = 0; j < mx_sigma.size(); ++j) { pred_sigma += X.coeff(i, mx_sigma.coeff(j)) * beta.coeff(beta_idx++); }
            for (Eigen::Index j = 0; j < mx_Q.size(); ++j)     { pred_Q     += X.coeff(i, mx_Q.coeff(j))     * beta.coeff(beta_idx++); }

            auto eval_row_eta = [&](double eta_m, double eta_s, double eta_q) -> double {
                double mu    = apply_inverse_link(eta_m, link_mu);
                double sigma = apply_inverse_link(eta_s, link_sigma);
                double Q     = apply_inverse_link(eta_q, link_Q);

                if (sigma <= 1e-8 || !std::isfinite(sigma) || !std::isfinite(mu) || !std::isfinite(Q)) return 1e10;
                if (std::abs(Q) < 1e-4) Q = (Q >= 0) ? 1e-4 : -1e-4;

                double ll = (status > 0.5) ? log_pdf(time1, mu, sigma, Q)
                                        : std::log(std::max(survival(time1, mu, sigma, Q), 1e-15));
                if (start > 1e-8) {
                    ll -= std::log(std::max(survival(start, mu, sigma, Q), 1e-15));
                }

                return -ll;
            };

            auto local_grad = [&](double em, double es, double eq) -> Eigen::Vector3d {
                double g_m = (eval_row_eta(em + eps, es, eq) - eval_row_eta(em - eps, es, eq)) / (2.0 * eps);
                double g_s = (eval_row_eta(em, es + eps, eq) - eval_row_eta(em, es - eps, eq)) / (2.0 * eps);
                double g_q = (eval_row_eta(em, es, eq + eps) - eval_row_eta(em, es, eq - eps)) / (2.0 * eps);
                return Eigen::Vector3d(g_m, g_s, g_q);
            };

            // 3x3 predictor-space Hessian of the row NLL by differencing the finite-difference gradient
            Eigen::Matrix3d H_eta = Eigen::Matrix3d::Zero();

            Eigen::Vector3d g_m_p = local_grad(pred_mu + eps, pred_sigma, pred_Q);
            Eigen::Vector3d g_m_m = local_grad(pred_mu - eps, pred_sigma, pred_Q);
            H_eta.col(0) = (g_m_p - g_m_m) / (2.0 * eps);

            Eigen::Vector3d g_s_p = local_grad(pred_mu, pred_sigma + eps, pred_Q);
            Eigen::Vector3d g_s_m = local_grad(pred_mu, pred_sigma - eps, pred_Q);
            H_eta.col(1) = (g_s_p - g_s_m) / (2.0 * eps);

            Eigen::Vector3d g_q_p = local_grad(pred_mu, pred_sigma, pred_Q + eps);
            Eigen::Vector3d g_q_m = local_grad(pred_mu, pred_sigma, pred_Q - eps);
            H_eta.col(2) = (g_q_p - g_q_m) / (2.0 * eps);

            H_eta = 0.5 * (H_eta + H_eta.transpose());

            double w_i = weights.coeff(i);

            // scatter H_eta into the parameter Hessian
            H.block<3, 3>(0, 0) += w_i * H_eta;

            Eigen::VectorXd X_m(map_mu.size());
            for (Eigen::Index j = 0; j < map_mu.size(); ++j) X_m(j) = X.coeff(i, mx_mu.coeff(j));

            Eigen::VectorXd X_s(map_sigma.size());
            for (Eigen::Index j = 0; j < map_sigma.size(); ++j) X_s(j) = X.coeff(i, mx_sigma.coeff(j));

            Eigen::VectorXd X_q(map_Q.size());
            for (Eigen::Index j = 0; j < map_Q.size(); ++j) X_q(j) = X.coeff(i, mx_Q.coeff(j));

            for (Eigen::Index j = 0; j < map_mu.size(); ++j)    H(0, map_mu[j])    += w_i * H_eta(0, 0) * X_m(j);
            for (Eigen::Index j = 0; j < map_sigma.size(); ++j) H(0, map_sigma[j]) += w_i * H_eta(0, 1) * X_s(j);
            for (Eigen::Index j = 0; j < map_Q.size(); ++j)     H(0, map_Q[j])     += w_i * H_eta(0, 2) * X_q(j);

            for (Eigen::Index j = 0; j < map_mu.size(); ++j)    H(1, map_mu[j])    += w_i * H_eta(1, 0) * X_m(j);
            for (Eigen::Index j = 0; j < map_sigma.size(); ++j) H(1, map_sigma[j]) += w_i * H_eta(1, 1) * X_s(j);
            for (Eigen::Index j = 0; j < map_Q.size(); ++j)     H(1, map_Q[j])     += w_i * H_eta(1, 2) * X_q(j);

            for (Eigen::Index j = 0; j < map_mu.size(); ++j)    H(2, map_mu[j])    += w_i * H_eta(2, 0) * X_m(j);
            for (Eigen::Index j = 0; j < map_sigma.size(); ++j) H(2, map_sigma[j]) += w_i * H_eta(2, 1) * X_s(j);
            for (Eigen::Index j = 0; j < map_Q.size(); ++j)     H(2, map_Q[j])     += w_i * H_eta(2, 2) * X_q(j);

            for (Eigen::Index r = 0; r < map_mu.size(); ++r) {
                for (Eigen::Index c = 0; c < map_mu.size(); ++c)
                    H(map_mu[r], map_mu[c]) += w_i * H_eta(0, 0) * X_m(r) * X_m(c);
            }
            for (Eigen::Index r = 0; r < map_sigma.size(); ++r) {
                for (Eigen::Index c = 0; c < map_sigma.size(); ++c)
                    H(map_sigma[r], map_sigma[c]) += w_i * H_eta(1, 1) * X_s(r) * X_s(c);
            }
            for (Eigen::Index r = 0; r < map_Q.size(); ++r) {
                for (Eigen::Index c = 0; c < map_Q.size(); ++c)
                    H(map_Q[r], map_Q[c]) += w_i * H_eta(2, 2) * X_q(r) * X_q(c);
            }

            for (Eigen::Index r = 0; r < map_mu.size(); ++r) {
                for (Eigen::Index c = 0; c < map_sigma.size(); ++c)
                    H(map_mu[r], map_sigma[c]) += w_i * H_eta(0, 1) * X_m(r) * X_s(c);
            }

            for (Eigen::Index r = 0; r < map_mu.size(); ++r) {
                for (Eigen::Index c = 0; c < map_Q.size(); ++c)
                    H(map_mu[r], map_Q[c]) += w_i * H_eta(0, 2) * X_m(r) * X_q(c);
            }

            for (Eigen::Index r = 0; r < map_sigma.size(); ++r) {
                for (Eigen::Index c = 0; c < map_Q.size(); ++c)
                    H(map_sigma[r], map_Q[c]) += w_i * H_eta(1, 2) * X_s(r) * X_q(c);
            }
        }

        // mirror the upper triangle
        H.triangularView<Eigen::Lower>() = H.transpose();
        return H;
    }

};

}

#endif // _GLMFIT_LOG_LIKELIHOOD_HPP
