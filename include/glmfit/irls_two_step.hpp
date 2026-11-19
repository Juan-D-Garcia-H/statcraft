// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file irls_two_step.hpp
 * @brief Two-step alternating IRLS fits for Negative Binomial and quasi-likelihood GLMs.
 * @details Built on `glm::IrlsSolver` (irls.hpp). Public function signatures are unchanged except for
 *          one trailing optional `IrlsOptions` argument; `TwoStepResult` only gained fields at the end.
 */

#ifndef _GLMFIT_IRLS_TWO_STEP_HPP
#define _GLMFIT_IRLS_TWO_STEP_HPP

#include "irls.hpp"
#include "families.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace glm {

/// Digamma psi(x), x > 0: recurrence up to x >= 10, then asymptotic series (error < 1e-14).
[[nodiscard]] inline double digamma_(double x)
{
    double result = 0.0;
    while (x < 10.0) { result -= 1.0 / x; x += 1.0; }
    const double x2 = 1.0 / (x * x);
    result += std::log(x) - 0.5 / x
            - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0 - x2 * (1.0/240.0 - x2 * (1.0/132.0)))));
    return result;
}

/// Trigamma psi'(x), x > 0: recurrence up to x >= 10, then asymptotic series.
[[nodiscard]] inline double trigamma_(double x)
{
    double result = 0.0;
    while (x < 10.0) { result += 1.0 / (x * x); x += 1.0; }
    const double xi = 1.0 / x;
    const double x2 = xi * xi;
    result += xi + 0.5 * x2
            + x2 * xi * (1.0/6.0 - x2 * (1.0/30.0 - x2 * (1.0/42.0 - x2 * (1.0/30.0 - x2 * (5.0/66.0)))));
    return result;
}

/// Result of the NB shape (theta) maximum-likelihood estimation.
struct ThetaFit {
    double theta;      ///< estimated theta (V = mu + mu^2/theta); last valid iterate if the fit failed
    int    iters;      ///< Newton iterations performed
    bool   converged;  ///< true if converged (also true when theta reaches the Poisson limit); false on any failure
};

namespace detail_two_step {

/// Largest integer response for which psi(y+theta)-psi(theta) is summed term by term (exact, no cancellation).
inline constexpr double kThetaSumMaxY = 64.0;

/// Distinct response values with their total prior weight: the y-only part of the NB score/information
/// (psi(y+theta) - psi(theta), psi'(theta) - psi'(y+theta)) is evaluated once per distinct y, not per row.
struct NbYTable {
    std::vector<double> y;   ///< distinct y (rows with positive weight), ascending
    std::vector<double> W;   ///< total prior weight of each distinct y (kept for diagnostics)
    std::vector<int>    idx; ///< per row: index of its y in `y`, -1 for rows with non-positive weight
};

[[nodiscard]] inline NbYTable make_y_table(const Eigen::ArrayXd& y, const Eigen::ArrayXd& prior_w)
{
    const std::size_t n = static_cast<std::size_t>(y.size());
    std::vector<std::size_t> ord;
    ord.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (prior_w[static_cast<Eigen::Index>(i)] > 0.0) ord.push_back(i);
    std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
        return y[static_cast<Eigen::Index>(a)] < y[static_cast<Eigen::Index>(b)]; });
    NbYTable t;
    t.idx.assign(n, -1);
    for (const std::size_t i : ord) {
        const double yi = y[static_cast<Eigen::Index>(i)];
        if (t.y.empty() || t.y.back() != yi) { t.y.push_back(yi); t.W.push_back(0.0); }
        t.W.back() += prior_w[static_cast<Eigen::Index>(i)];
        t.idx[i] = static_cast<int>(t.y.size()) - 1;
    }
    return t;
}

/// psi(y+theta)-psi(theta) and psi'(theta)-psi'(y+theta): exact term-by-term sums for integer y <= 64.
inline void nb_y_terms(double y, double theta, double dg_t, double tg_t, double& dpsi, double& dtri)
{
    if (y <= kThetaSumMaxY && y == std::floor(y)) {
        dpsi = 0.0; dtri = 0.0;
        for (double k = 0.0; k < y; k += 1.0) {
            const double inv = 1.0 / (theta + k);
            dpsi += inv;
            dtri += inv * inv;
        }
    } else {
        dpsi = digamma_(y + theta) - dg_t;
        dtri = tg_t - trigamma_(y + theta);
    }
}

/**
 * @brief NB score and information in theta, written to avoid cancellation for large theta.
 * @details Per row, with tm = theta + mu:
 *   score_i = [psi(y+theta) - psi(theta)] - log1p(mu/theta) + (mu - y)/tm
 *   info_i  = [psi'(theta) - psi'(y+theta)] - mu/(theta tm) + (mu - y)/tm^2
 * The bracketed y-only terms are evaluated once per distinct y (`NbYTable`) but combined with the mu terms
 * row by row: the O(1/theta) parts cancel inside each row, so no large partial sums are ever accumulated.
 * Returns false if either sum is non-finite.
 */
inline bool nb_theta_sums(const Eigen::ArrayXd& y, const Eigen::ArrayXd& mu, const Eigen::ArrayXd& prior_w,
                          const NbYTable& tab, double theta, double& score_out, double& info_out)
{
    const double dg_t = digamma_(theta);
    const double tg_t = trigamma_(theta);

    const std::size_t m = tab.y.size();
    std::vector<double> dpsi(m), dtri(m);
    for (std::size_t k = 0; k < m; ++k) nb_y_terms(tab.y[k], theta, dg_t, tg_t, dpsi[k], dtri[k]);

    double score = 0.0, info = 0.0;
    const Eigen::Index n = y.size();
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : score, info) schedule(static) if (n > 4096)
#endif
    for (Eigen::Index i = 0; i < n; ++i) {
        const int k = tab.idx[static_cast<std::size_t>(i)];
        if (k < 0) continue;
        const double wi = prior_w[i];
        const double mi = mu[i], tm = theta + mi, dm = mi - y[i];
        score += wi * (dpsi[static_cast<std::size_t>(k)] - std::log1p(mi / theta) + dm / tm);
        info  += wi * (dtri[static_cast<std::size_t>(k)] - mi / (theta * tm) + dm / (tm * tm));
    }
    score_out = score;
    info_out  = info;
    return std::isfinite(score) && std::isfinite(info);
}

}  // namespace detail_two_step

/**
 * @brief Safeguarded Newton-Raphson MLE of the Negative Binomial shape theta for fixed means.
 * @details Uses cancellation-free score/information tabulated by distinct y (see `detail_two_step::nb_theta_sums`).
 *          Safeguards: each step is limited to theta/10 .. 10*theta; if the observed information is not
 *          positive the step falls back to doubling (score > 0) or halving (score <= 0) and cannot
 *          signal convergence. theta is capped at 1e10 (Poisson limit): the cap counts as converged
 *          only if the score there is still positive (likelihood increasing). Any non-finite value
 *          returns `converged = false` with the last valid theta, so callers must check the flag.
 */
[[nodiscard]] inline ThetaFit mle_theta_nb(const Eigen::ArrayXd& y,
                                           const Eigen::ArrayXd& mu,
                                           const Eigen::ArrayXd& prior_w,
                                           double theta0,
                                           int maxit = 100,
                                           double tol = 1e-8)
{
    constexpr double theta_max = 1e10;
    if (!std::isfinite(theta0)) return ThetaFit{ theta0, 0, false };
    double theta = std::clamp(theta0, 1e-4, theta_max);
    const detail_two_step::NbYTable tab = detail_two_step::make_y_table(y, prior_w);
    int it = 0;

    for (; it < maxit; ++it) {
        double score, info;
        if (!detail_two_step::nb_theta_sums(y, mu, prior_w, tab, theta, score, info))
            return ThetaFit{ theta, it, false };

        // Poisson limit: still increasing at the cap -> theta = cap is the (boundary) maximum.
        if (theta >= theta_max && score > 0.0) return ThetaFit{ theta_max, it + 1, true };

        const bool newton_ok = info > 0.0;
        double step = newton_ok ? score / info : (score > 0.0 ? theta : -0.5 * theta);
        const double lo = -0.9 * theta, hi = 9.0 * theta;   // theta_new in [theta/10, 10 theta]
        const bool limited = !(step >= lo && step <= hi);
        step = std::clamp(step, lo, hi);

        double theta_new = std::min(theta + step, theta_max);
        const double rel = std::abs(theta_new - theta) / (theta + tol);
        theta = theta_new;
        if (newton_ok && !limited && rel < tol) return ThetaFit{ theta, it + 1, true };
    }
    return ThetaFit{ theta, it, false };
}

/// Output container shared by the NB and quasi fits.
struct TwoStepResult {
    Eigen::VectorXd beta;       ///< coefficients
    Eigen::VectorXd se;         ///< standard errors (NaN for aliased columns)
    Eigen::MatrixXd vcov;       ///< covariance matrix (NaN rows/cols for aliased columns)
    double theta = std::numeric_limits<double>::quiet_NaN();  ///< NB shape (NaN for quasi)
    double dispersion = 1.0;    ///< phi (1 for NB, Pearson estimate for quasi)
    int outer_iters = 0;        ///< outer alternating iterations
    bool converged = false;     ///< global convergence flag
    int rank = -1;              ///< detected rank of the design
    double deviance = std::numeric_limits<double>::quiet_NaN();  ///< sum w_i d(y_i, mu_i) at the fit (NB: at final theta)
    int n_ok = 0;               ///< observations with positive prior weight (residual df = n_ok - rank)
    // --- added fields ---
    int inner_iters = 0;                    ///< total IRLS iterations across all passes
    std::vector<int> aliased;               ///< 0-based aliased columns (pivoted back ends)
    Eigen::VectorXd mu;                     ///< fitted means
    Eigen::VectorXd eta;                    ///< linear predictor
};

namespace detail_two_step {

template <class XMat>
inline void check_inputs(const XMat& X,
                         const Eigen::Map<const Eigen::VectorXd>& Y,
                         const Eigen::Map<const Eigen::VectorXd>& w,
                         const Eigen::Map<const Eigen::VectorXd>& off,
                         const Eigen::VectorXd& beta_start, const char* who)
{
    const std::string name(who);
    if (Y.size() != X.rows() || w.size() != X.rows() || off.size() != X.rows())
        throw std::invalid_argument(name + ": Y, weights and offset must have nrow(X) entries");
    if (beta_start.size() != X.cols())
        throw std::invalid_argument(name + ": beta_start must have ncol(X) entries");
}

/// beta_start == 0: family-aware defaults (mustart / linkfun, as in R). Otherwise start at beta_start.
template <class XMat>
inline void init_model(IrlsSolver& model, const XMat& X,
                       const Eigen::Map<const Eigen::VectorXd>& offset, int code,
                       const FamilyParams& params, const Eigen::VectorXd& beta_start)
{
    if (beta_start.isZero(0.0)) { model.init_default(); return; }
    Eigen::ArrayXd eta0 = (X * beta_start).array() + offset.array();
    Eigen::ArrayXd mu0(X.rows());
    glm::linkinv(code, params, eta0, mu0);
    model.init_parms(Eigen::Map<const Eigen::VectorXd>(beta_start.data(), beta_start.size()),
                     Eigen::Map<const Eigen::VectorXd>(mu0.data(), mu0.size()),
                     Eigen::Map<const Eigen::VectorXd>(eta0.data(), eta0.size()));
}

inline void fill_result(TwoStepResult& out, const IrlsSolver& m)
{
    out.beta     = m.get_beta();
    out.mu       = m.get_mu();
    out.eta      = m.get_eta();
    out.rank     = m.get_rank();
    out.aliased  = m.get_aliased();
}

}  // namespace detail_two_step

/**
 * @brief Alternating IRLS / theta-MLE fit for Negative Binomial families.
 * @param estimate_theta if false, `theta_start` is treated as known and kept fixed (one IRLS fit).
 * @param link_power exponent of the power link, used only for FAM_NB_POWER (1 = identity, 0 = log).
 * @param opts optional solver knobs (rank tolerance, chunk rows, observed Newton, LM, ...).
 * @note X may be a dense `Eigen::Map<const MatrixXd>` or a sparse CSC `Eigen::SparseMatrix<double>` (must outlive the call).
 * @note Outer convergence requires a converged theta step; a failed theta MLE never counts as convergence.
 *       se/vcov treat theta as known (as in R's glm.nb).
 */
template <class XMat>   // Eigen::Map<const MatrixXd> (dense) or Eigen::SparseMatrix<double> (CSC)
inline TwoStepResult fit_nb_two_step(
    const XMat& X,
    const Eigen::Map<const Eigen::VectorXd>& Y,
    const Eigen::Map<const Eigen::VectorXd>& weights,
    const Eigen::Map<const Eigen::VectorXd>& offset,
    int fam_code,
    const Eigen::VectorXd& beta_start,
    double theta_start = 1.0,
    int outer_maxit = 25,
    double outer_tol = 1e-6,
    int inner_maxit = 100,
    double inner_tol = 1e-8,
    int type = 6,
    bool is_big_matrix = false,
    double link_power = 1.0,
    bool estimate_theta = true,
    const IrlsOptions& opts = IrlsOptions())
{
    if (!(fam_code == FAM_NB_LOG || fam_code == FAM_NB_SQRT ||
          fam_code == FAM_NB_IDENTITY || fam_code == FAM_NB_INVERSE ||
          fam_code == FAM_NB_POWER)) {
        throw std::invalid_argument("fit_nb_two_step: specified fam_code is not a recognized NB family");
    }
    detail_two_step::check_inputs(X, Y, weights, offset, beta_start, "fit_nb_two_step");
    if (!(std::isfinite(theta_start) && theta_start > 0.0))
        throw std::invalid_argument("fit_nb_two_step: theta_start must be finite and positive");
    if ((Y.array() < 0.0).any())
        throw std::invalid_argument("fit_nb_two_step: the response must be non-negative counts");

    FamilyParams params;
    params.theta = std::max(theta_start, 1e-4);
    if (fam_code == FAM_NB_POWER) params.link_power = link_power;

    IrlsSolver model(X, Y, weights, offset, fam_code, params, inner_tol, inner_maxit,
                     type, is_big_matrix, false, opts);
    detail_two_step::init_model(model, X, offset, fam_code, params, beta_start);

    const Eigen::ArrayXd y_arr = Y.array();
    const Eigen::ArrayXd w_arr = weights.array();

    double theta_prev = params.theta;
    bool converged = false;
    int outer_it = 0;
    int inner_total = 0;

    if (!estimate_theta) {
        model.set_fam_params(params);
        inner_total += model.solve(inner_maxit);
        converged = true;                 // theta is data: nothing outer to converge
        outer_it = 1;
    } else {
        double rel_prev = 1.0;
        for (outer_it = 0; outer_it < outer_maxit; ++outer_it) {
            // loose inner solves while theta is still moving, strict once it settles
            model.set_tol(std::max(inner_tol, std::min(1e-4, 1e-2 * rel_prev)));
            model.set_fam_params(params);
            inner_total += model.solve(inner_maxit);

            const Eigen::ArrayXd mu = model.get_mu().array();
            const ThetaFit tf = mle_theta_nb(y_arr, mu, w_arr, params.theta);
            params.theta = tf.theta;

            const double rel = std::abs(params.theta - theta_prev) / (theta_prev + outer_tol);
            theta_prev = params.theta;
            rel_prev = rel;
            if (tf.converged && rel < outer_tol) { converged = true; ++outer_it; break; }
            if (!tf.converged && rel < outer_tol) { ++outer_it; break; }  // theta fit failed and is stuck: give up, not converged
        }
        // Refit beta at the final theta (strict tolerance) so beta, se, vcov, deviance and theta describe the same model.
        model.set_tol(inner_tol);
        model.set_fam_params(params);
        inner_total += model.solve(inner_maxit);
    }

    TwoStepResult out;
    detail_two_step::fill_result(out, model);
    out.se          = model.get_se();
    out.vcov        = model.get_vcov();
    out.theta       = params.theta;
    out.dispersion  = 1.0;
    out.outer_iters = outer_it;
    out.inner_iters = inner_total;
    out.converged   = converged && model.get_converged();
    out.deviance    = model.get_std_dev();
    out.n_ok        = static_cast<int>((weights.array() > 0.0).count());
    return out;
}

/// Maps a quasi family code to the family that supplies its variance function, link and deviance.
[[nodiscard]] inline int base_family_for_quasi(int quasi_code)
{
    switch (quasi_code) {
        case FAM_QUASI_POISSON_LOG:          return FAM_POISSON_LOG;
        case FAM_QUASI_POISSON_IDENTITY:        return FAM_POISSON_IDENTITY;
        case FAM_QUASI_BINOMIAL_LOGIT:       return FAM_BINOMIAL_LOGIT;
        case FAM_QUASI_BINOMIAL_PROBIT:      return FAM_BINOMIAL_PROBIT;
        case FAM_QUASI_GAUSSIAN_IDENTITY:       return FAM_GAUSSIAN_IDENTITY;
        case FAM_QUASI_GAUSSIAN_LOG:         return FAM_GAUSSIAN_LOG;
        case FAM_QUASI_GAMMA_LOG:            return FAM_GAMMA_LOG;
        case FAM_QUASI_GAMMA_IDENTITY:          return FAM_GAMMA_IDENTITY;
        case FAM_QUASI_GAMMA_INVERSE:        return FAM_GAMMA_INVERSE;
        case FAM_QUASI_INVGAUSSIAN_LOG:      return FAM_INVGAUSS_LOG;
        case FAM_QUASI_INVGAUSSIAN_IDENTITY:    return FAM_INVGAUSS_IDENTITY;
        case FAM_QUASI_INVGAUSSIAN_INVERSE:  return FAM_INVGAUSS_INVERSE;
        default:
            throw std::invalid_argument("base_family_for_quasi: specified code is not a quasi family");
    }
}

/// Response checks that R's quasipoisson / quasibinomial perform (rows with zero weight are ignored).
inline void check_quasi_response(int base_code, const Eigen::Map<const Eigen::VectorXd>& Y,
                                 const Eigen::Map<const Eigen::VectorXd>& w)
{
    const bool poisson  = (base_code == FAM_POISSON_LOG || base_code == FAM_POISSON_IDENTITY);
    const bool binomial = (base_code == FAM_BINOMIAL_LOGIT || base_code == FAM_BINOMIAL_PROBIT);
    if (!poisson && !binomial) return;
    for (Eigen::Index i = 0; i < Y.size(); ++i) {
        if (!(w[i] > 0.0)) continue;
        if (poisson && Y[i] < 0.0)
            throw std::invalid_argument("fit_quasi: negative values not allowed for the quasi-Poisson family");
        if (binomial && (Y[i] < 0.0 || Y[i] > 1.0))
            throw std::invalid_argument("fit_quasi: y values must be in [0, 1] for the quasi-binomial family");
    }
}

/**
 * @brief Quasi-likelihood fit: IRLS of the base family plus Pearson dispersion
 *        phi = sum w_i (y_i - mu_i)^2 / V(mu_i) / (n_ok - rank); vcov and se are scaled by phi.
 */
template <class XMat>   // Eigen::Map<const MatrixXd> (dense) or Eigen::SparseMatrix<double> (CSC)
inline TwoStepResult fit_quasi(
    const XMat& X,
    const Eigen::Map<const Eigen::VectorXd>& Y,
    const Eigen::Map<const Eigen::VectorXd>& weights,
    const Eigen::Map<const Eigen::VectorXd>& offset,
    int quasi_code,
    const Eigen::VectorXd& beta_start,
    int maxit = 100,
    double tol = 1e-8,
    int type = 6,
    bool is_big_matrix = false,
    const IrlsOptions& opts = IrlsOptions())
{
    const int base_code = base_family_for_quasi(quasi_code);
    detail_two_step::check_inputs(X, Y, weights, offset, beta_start, "fit_quasi");
    check_quasi_response(base_code, Y, weights);
    FamilyParams params;

    IrlsSolver model(X, Y, weights, offset, base_code, params, tol, maxit,
                     type, is_big_matrix, false, opts);
    detail_two_step::init_model(model, X, offset, base_code, params, beta_start);
    const int iters = model.solve(maxit);

    const Eigen::ArrayXd mu = model.get_mu().array();
    Eigen::ArrayXd V(mu.size());
    glm::variance(base_code, params, mu, V);

    // Residual df = n_ok - rank, as in R's glm.fit (zero-weight rows carry no dispersion information).
    const Eigen::Index n_ok = static_cast<Eigen::Index>((weights.array() > 0.0).count());
    const int rank = model.get_rank();
    const double pearson_chi2 =
        (weights.array() * (Y.array() - mu).square() / V.max(glm::thresh_eps())).sum();
    const double phi = pearson_chi2 / static_cast<double>(std::max<Eigen::Index>(n_ok - rank, 1));

    TwoStepResult out;
    detail_two_step::fill_result(out, model);
    out.vcov        = model.get_vcov() * phi;
    out.se          = (model.get_se().array() * std::sqrt(phi)).matrix();
    out.dispersion  = phi;
    out.outer_iters = 1;
    out.inner_iters = iters;
    out.converged   = model.get_converged();
    out.deviance    = model.get_std_dev();   // quasi-deviance = deviance of the base family
    out.n_ok        = static_cast<int>(n_ok);
    return out;
}

}  // namespace glm

#endif  // _GLMFIT_IRLS_TWO_STEP_HPP
