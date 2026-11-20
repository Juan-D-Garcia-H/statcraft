// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file loglik.hpp
 * @brief Marginal log-likelihood utilities for post-SAEM goodness of fit.
 *
 * The marginal log-likelihood is
 * @f[
 *   \ell(\theta) = \sum_i \log \int p(y_i \mid \phi_i)\,
 *                  p(\phi_i \mid \theta)\,d\phi_i.
 * @f]
 *
 * Two estimators are provided:
 *  - @ref llgq : Gauss-Hermite quadrature, accurate in low dimensions;
 *  - @ref llis : importance sampling with a Gaussian or Student-t proposal
 *    centered at the conditional mode.
 *
 * Both rely on the @ref LogLikModel interface:
 *  - @c logJoint(i, phi, theta) = @f$\log p(y_i \mid \phi) + \log p(\phi \mid \theta)@f$;
 *  - @c priorParams(theta) returns @f$(\mu, \Omega)@f$ for the population
 *    distribution @f$N(\mu, \Omega)@f$.
 *
 * Requires: Eigen3, C++20, nelder.hpp.
 */

#pragma once
#include "numcast.hpp"

#include "nelder.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <map>
#include <numbers>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Model interface required by the marginal-likelihood estimators.
 */
struct LogLikModel {
    int dimChi = 0;       ///< Dimension of the individual random effect.
    int nSubjects = 0;    ///< Number of subjects.
    /**
     * @brief Complete-data log density
     *        @f$\log p(y_i \mid \phi) + \log p(\phi \mid \theta)@f$.
     */
    std::function<double(int i, const Vec& phi, const Vec& theta)> logJoint;
    /**
     * @brief Extracts the population mean and covariance from @f$\theta@f$.
     */
    std::function<void(const Vec& theta, Vec& mu, Mat& Omega)> priorParams;
};

/**
 * @brief Computes the lower Cholesky factor of a positive-definite matrix.
 * @param A Symmetric positive-definite matrix.
 * @return The lower triangular factor @c L with @f$A = LL^\top@f$.
 * @throws std::runtime_error If @p A is not positive definite.
 */
inline Mat cholLower(const Mat& A) {
    Eigen::LLT<Mat> llt(A);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("matrix is not positive definite in Cholesky factorization");
    return llt.matrixL();
}

/**
 * @brief Computes @f$\log\det(A)@f$ from a Cholesky factor.
 * @param L Lower Cholesky factor of @f$A@f$.
 * @return @f$\log\det(A)@f$.
 */
inline double logDetFromChol(const Mat& L) {
    return 2.0 * L.diagonal().array().log().sum();
}

/**
 * @brief Evaluates the Gaussian log density @f$\log N(\phi; \mu, \Omega)@f$.
 * @param phi Evaluation point.
 * @param mu Mean vector.
 * @param Omega Covariance matrix.
 * @return The log density.
 * @throws std::runtime_error If @p Omega is not positive definite.
 */
inline double logPrior(const Vec& phi, const Vec& mu, const Mat& Omega) {
    const Mat L = cholLower(Omega);
    const Vec d = L.triangularView<Eigen::Lower>().solve(phi - mu);
    const double d_ = double(phi.size());
    return -0.5 * (d.squaredNorm() + logDetFromChol(L) + d_ * std::log(2.0 * std::numbers::pi));
}

/**
 * @brief Numerically stable @f$\log\left(\frac{1}{M}\sum_m e^{v_m}\right)@f$.
 * @param v Log-scale values.
 * @return The log mean, or @c -infinity for an empty/all-@c -infinity input.
 */
inline double logMeanExp(const std::vector<double>& v) {
    if (v.empty()) return -std::numeric_limits<double>::infinity();
    double mx = -std::numeric_limits<double>::infinity();
    for (double x : v) if (x > mx) mx = x;
    if (!std::isfinite(mx)) return mx;            // All -inf (avoids -inf - -inf = NaN).
    double s = 0.0;
    for (double x : v) s += std::exp(x - mx);
    return mx + std::log(s / double(v.size()));
}

/**
 * @brief Numerically stable @f$\log\sum_m e^{v_m}@f$.
 * @param v Log-scale values.
 * @return The log sum, or @c -infinity for an empty/all-@c -infinity input.
 */
inline double logSumExp(const std::vector<double>& v) {
    double m = -std::numeric_limits<double>::infinity();
    for (double x : v) if (x > m) m = x;
    if (!std::isfinite(m)) return m;
    double s = 0.0;
    for (double x : v) s += std::exp(x - m);
    return m + std::log(s);
}

/**
 * @brief Computes Gauss-Hermite nodes and weights (Golub-Welsch).
 *
 * Approximates @f$\int f(x)e^{-x^2}dx \approx \sum_k w_k f(x_k)@f$.
 *
 * @param n Number of nodes.
 * @param x Output nodes.
 * @param w Output weights.
 * @throws std::invalid_argument If @p n < 1.
 */
inline void gaussHermite(int n, Vec& x, Vec& w) {
    if (n < 1) throw std::invalid_argument("gaussHermite: n must be >= 1");
    Mat J = Mat::Zero(n, n);
    for (int i = 0; i < n - 1; ++i) {
        const double b = std::sqrt((i + 1) / 2.0);
        J(i, i + 1) = b;
        J(i + 1, i) = b;
    }
    Eigen::SelfAdjointEigenSolver<Mat> es(J);
    x = es.eigenvalues();
    Vec v = es.eigenvectors().row(0).transpose();
    w = std::sqrt(std::numbers::pi) * v.array().square();
}

/**
 * @brief Thread-local cached version of @ref gaussHermite.
 *
 * The eigendecomposition is expensive and is requested on every inner
 * evaluation of nested quadrature, so results are cached per thread and node
 * count.
 *
 * @param n Number of nodes.
 * @param x Output nodes.
 * @param w Output weights.
 * @throws std::invalid_argument If @p n < 1.
 */
inline void gaussHermiteCached(int n, Vec& x, Vec& w) {
    thread_local std::map<int, std::pair<Vec, Vec>> cache;
    auto it = cache.find(n);
    if (it == cache.end()) {
        Vec xx, ww;
        gaussHermite(n, xx, ww);
        it = cache.emplace(n, std::make_pair(std::move(xx), std::move(ww))).first;
    }
    x = it->second.first; w = it->second.second;
}

/**
 * @brief Iterates over a @p d-dimensional Gauss-Hermite grid.
 * @tparam F Callable with signature @c cb(const Vec& z, double wprod).
 * @param xg One-dimensional nodes.
 * @param wg One-dimensional weights.
 * @param d Dimension of the tensor grid.
 * @param cb Callback invoked with each node and the product of its weights.
 */
template <class F>
inline void ghGrid(const Vec& xg, const Vec& wg, int d, F&& cb) {
    const int n = static_cast<int>(xg.size());
    std::vector<int> idx(uz(d), 0);
    while (true) {
        double wprod = 1.0;
        Vec z(d);
        for (int j = 0; j < d; ++j) { z[j] = xg[idx[uz(j)]]; wprod *= wg[idx[uz(j)]]; }
        cb(z, wprod);
        int k = 0;
        while (k < d) { if (++idx[uz(k)] < n) break; idx[uz(k)] = 0; ++k; }
        if (k == d) break;
    }
}

/**
 * @brief Cholesky factorization with a bounded relative jitter.
 *
 * Factors @f$S + \text{jitter}\cdot\overline{\operatorname{diag}(S)}I@f$,
 * increasing the jitter geometrically. Returns @c false when factorization
 * still fails, so the caller can fall back to another proposal (for example the
 * prior).
 *
 * @param S Matrix to factor.
 * @param L Output lower Cholesky factor.
 * @param logdet Output log-determinant of the jittered matrix.
 * @return @c true on success.
 */
inline bool cholWithJitter(const Mat& S, Mat& L, double& logdet) {
    const double scale = std::max(1e-300, S.diagonal().cwiseAbs().mean());
    double jitter = 0.0;
    for (int a = 0; a < 14; ++a) {
        Mat T = S;
        if (jitter > 0.0) T.diagonal().array() += jitter * scale;
        Eigen::LLT<Mat> llt(T);
        if (llt.info() == Eigen::Success) {
            L = llt.matrixL(); logdet = 2.0 * L.diagonal().array().log().sum();
            return true;
        }
        jitter = (jitter == 0.0) ? 1e-12 : jitter * 10.0;
    }
    return false;
}

/**
 * @brief Robust inverse of the negative Hessian.
 *
 * Computes @f$\Sigma = (-H)^{-1}@f$ through an eigendecomposition with absolute
 * eigenvalues and a relative floor. If @f$-H@f$ is indefinite (a poorly located
 * mode or noisy numerical curvature), a naive inverse would produce a
 * meaningless covariance; the spectral floor instead assigns finite, reasonable
 * variance to badly conditioned or negative-curvature directions.
 *
 * @param H Hessian of the log posterior at the mode.
 * @return The covariance estimate, or an empty matrix when @p H is not finite
 *         or the eigendecomposition fails.
 */
inline Mat sigmaFromNegHessian(const Mat& H) {
    if (!H.allFinite()) return Mat();
    const Mat A = 0.5 * ((-H) + (-H).transpose());
    Eigen::SelfAdjointEigenSolver<Mat> es(A);
    if (es.info() != Eigen::Success) return Mat();
    Vec ev = es.eigenvalues();
    const double fl = 1e-8 * std::max(1.0, ev.cwiseAbs().maxCoeff());
    Vec inv = ev.cwiseAbs().cwiseMax(fl).cwiseInverse();
    Mat S = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
    return 0.5 * (S + S.transpose());
}

/**
 * @brief Fixed Gauss-Hermite marginal log-likelihood for one subject.
 *
 * Integrates over the prior @f$N(\mu,\Omega)@f$ using a tensor-product grid in
 * @c dimChi dimensions. Accumulation is performed on the log scale to avoid
 * underflow and overflow.
 *
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param i Subject index.
 * @param n_nodes Number of quadrature nodes per dimension.
 * @return @f$\log p(y_i \mid \theta)@f$.
 * @throws std::runtime_error If the prior covariance is not positive definite.
 */
inline double llgqSubject(const LogLikModel& m, const Vec& theta, int i, int n_nodes) {
    Vec mu; Mat Omega;
    m.priorParams(theta, mu, Omega);
    const Mat L = cholLower(Omega);
    const int d = m.dimChi;

    Vec xg, wg;
    gaussHermiteCached(n_nodes, xg, wg);

    // Log-scale accumulation (avoids underflow/overflow for large n_i).
    std::vector<double> terms;
    ghGrid(xg, wg, d, [&](const Vec& z, double wprod) {
        if (!(wprod > 0.0)) return;
        const Vec phi = mu + std::sqrt(2.0) * (L * z);
        const double lp = m.logJoint(i, phi, theta) - logPrior(phi, mu, Omega);
        if (std::isfinite(lp)) terms.push_back(std::log(wprod) + lp);
    });
    if (terms.empty()) return -std::numeric_limits<double>::infinity();
    return logSumExp(terms) - 0.5 * double(d) * std::log(std::numbers::pi);
}

/**
 * @brief Fixed Gauss-Hermite marginal log-likelihood over all subjects.
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param n_nodes Number of quadrature nodes per dimension.
 * @return The total marginal log-likelihood.
 */
inline double llgq(const LogLikModel& m, const Vec& theta, int n_nodes) {
    double s = 0.0;
    for (int i = 0; i < m.nSubjects; ++i) s += llgqSubject(m, theta, i, n_nodes);
    return s;
}

/**
 * @brief Computes the conditional mode of @f$\phi_i@f$ by Nelder-Mead.
 *
 * Maximizes @f$\log p(\phi \mid y_i, \theta)@f$ starting from the prior mean,
 * with an initial simplex scaled to 30% of the prior standard deviations.
 *
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param i Subject index.
 * @return The conditional mode.
 */
inline Vec mapForLL(const LogLikModel& m, const Vec& theta, int i) {
    Vec mu; Mat Omega;
    m.priorParams(theta, mu, Omega);
    auto neg = [&](const Vec& p) { return -m.logJoint(i, p, theta); };
    // Initial simplex at the population scale: 0.3 * sd(Omega).
    const Vec sc = 0.3 * Omega.diagonal().cwiseMax(1e-12).cwiseSqrt();
    return nelderMeadEx(neg, mu, sc, 1000).x;
}

/**
 * @brief Central-difference numerical Hessian.
 * @param f Scalar function.
 * @param x Evaluation point.
 * @param h Relative step size.
 * @return The Hessian matrix at @p x.
 */
inline Mat numericalHessian(const std::function<double(const Vec&)>& f, const Vec& x, double h = 1e-4) {
    const int n = static_cast<int>(x.size());
    Mat H = Mat::Zero(n, n);
    Vec xp = x, xm = x;
    for (int j = 0; j < n; ++j) {
        const double hj = h * std::max(1.0, std::abs(x[j]));
        xp = x; xm = x; xp[j] += hj; xm[j] -= hj;
        H(j, j) = (f(xp) - 2.0 * f(x) + f(xm)) / (hj * hj);
    }
    for (int j = 0; j < n; ++j) for (int k = j + 1; k < n; ++k) {
        const double hj = h * std::max(1.0, std::abs(x[j]));
        const double hk = h * std::max(1.0, std::abs(x[k]));
        Vec a = x, b = x, c = x, e = x;
        a[j] += hj; a[k] += hk;
        b[j] += hj; b[k] -= hk;
        c[j] -= hj; c[k] += hk;
        e[j] -= hj; e[k] -= hk;
        const double v = (f(a) - f(b) - f(c) + f(e)) / (4.0 * hj * hk);
        H(j, k) = v; H(k, j) = v;
    }
    return H;
}

/**
 * @brief Adaptive Gauss-Hermite marginal log-likelihood for one subject.
 *
 * Fixed Gauss-Hermite quadrature (see @ref llgqSubject) integrates over the
 * prior @f$N(\mu,\Omega)@f$. When the likelihood is much narrower than the prior
 * (small residual error), the fixed nodes do not resolve the peak and the
 * integral -- especially its derivatives -- becomes unstable (the dominant node
 * switches, producing kinks) and corrupts the FIM and standard errors.
 *
 * The adaptive version changes variables to a Gaussian centered at the
 * conditional mode with covariance equal to the inverse negative Hessian of
 * @f$\log p(\phi \mid y_i)@f$:
 * @f[
 *   \phi = \text{mode} + \sqrt{2}Lz,\qquad L = \operatorname{chol}((-H)^{-1}).
 * @f]
 * Then few nodes (10-20) suffice and numerical derivatives of the
 * log-likelihood remain stable.
 *
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param i Subject index.
 * @param n_nodes Number of quadrature nodes per dimension.
 * @return @f$\log p(y_i \mid \theta)@f$.
 * @throws std::runtime_error If the fallback prior covariance is not positive
 *         definite.
 */
inline double llgqSubjectAdaptive(const LogLikModel& m, const Vec& theta,
                                  int i, int n_nodes) {
    const int d = m.dimChi;
    Vec mu; Mat Omega;
    m.priorParams(theta, mu, Omega);

    // Conditional mode (maximize log p(phi | y_i)) and its Hessian.
    const Vec mode = mapForLL(m, theta, i);
    auto lj = [&](const Vec& p) { return m.logJoint(i, p, theta); };
    const Mat H = numericalHessian(lj, mode);      // Hessian of log p(phi | y).

    // Sigma = (-H)^{-1} with a spectral floor; fall back to the prior if unusable.
    Mat Sigma = sigmaFromNegHessian(H);
    Mat L; double logdetS = 0.0;
    if (Sigma.size() == 0 || !cholWithJitter(Sigma, L, logdetS)) {
        L = cholLower(Omega); logdetS = logDetFromChol(L);
    }

    Vec xg, wg;
    gaussHermiteCached(n_nodes, xg, wg);

    std::vector<double> terms;
    ghGrid(xg, wg, d, [&](const Vec& z, double wprod) {
        if (!(wprod > 0.0)) return;
        const Vec phi = mode + std::sqrt(2.0) * (L * z);
        const Vec dv  = L.triangularView<Eigen::Lower>().solve(phi - mode);
        const double logq = -0.5 * double(d) * std::log(2.0 * std::numbers::pi) - 0.5 * logdetS
                            - 0.5 * dv.squaredNorm();
        const double v = m.logJoint(i, phi, theta) - logq;
        if (std::isfinite(v)) terms.push_back(std::log(wprod) + v);
    });
    if (terms.empty()) return -std::numeric_limits<double>::infinity();
    return logSumExp(terms) - 0.5 * double(d) * std::log(std::numbers::pi);
}

/**
 * @brief Adaptive Gauss-Hermite marginal log-likelihood over all subjects.
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param n_nodes Number of quadrature nodes per dimension.
 * @return The total marginal log-likelihood.
 */
inline double llgqAdaptive(const LogLikModel& m, const Vec& theta, int n_nodes) {
    double s = 0.0;
    for (int i = 0; i < m.nSubjects; ++i) s += llgqSubjectAdaptive(m, theta, i, n_nodes);
    return s;
}

/**
 * @brief Importance-sampling marginal log-likelihood for one subject.
 *
 * Uses a Student-t proposal @f$t_\nu(\text{mode}, \Sigma)@f$, where
 * @f$\Sigma@f$ is the inverse negative Hessian of @f$\log p(\phi \mid y)@f$ at
 * the conditional mode. Random numbers are seeded per subject so that
 * evaluations at different @f$\theta@f$ are smooth, which is required for
 * numerical differentiation.
 *
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param i Subject index.
 * @param M Number of Monte Carlo draws.
 * @param seed Random seed.
 * @param nu Degrees of freedom of the Student-t proposal (must be > 0).
 * @return @f$\log p(y_i \mid \theta)@f$.
 * @throws std::invalid_argument If @p nu <= 0.
 * @throws std::runtime_error If the prior covariance is not positive definite.
 */
inline double llisSubject(const LogLikModel& m, const Vec& theta, int i, int M,
                          unsigned seed, double nu = 4.0) {
    if (!(nu > 0.0)) throw std::invalid_argument("llis: nu must be > 0");
    const int d = m.dimChi;
    const Vec mode = mapForLL(m, theta, i);
    auto lj = [&](const Vec& p) { return m.logJoint(i, p, theta); };
    const Mat H = numericalHessian(lj, mode);
    Vec mu0; Mat Omega;
    m.priorParams(theta, mu0, Omega);
    Mat Sigma = sigmaFromNegHessian(H);
    Mat L; double logdetS = 0.0;
    if (Sigma.size() == 0 || !cholWithJitter(Sigma, L, logdetS)) {
        L = cholLower(Omega); logdetS = logDetFromChol(L);
    }

    const double lognorm_q = std::lgamma(0.5 * (nu + d)) - std::lgamma(0.5 * nu)
                           - 0.5 * d * std::log(nu * std::numbers::pi) - 0.5 * logdetS;

    // Fixed seed per subject: common random numbers across theta evaluations
    // keep the estimator smooth in theta (needed for numerical derivatives).
    std::mt19937 rng(seed + 7919u * unsigned(i + 1));
    std::normal_distribution<double> nz(0.0, 1.0);
    std::gamma_distribution<double> gam(0.5 * nu, 2.0);   // chi^2_nu, real nu.
    std::vector<double> lw(uz(M));
    for (int s = 0; s < M; ++s) {
        Vec z(d);
        for (int j = 0; j < d; ++j) z[j] = nz(rng);
        const double chi2 = gam(rng);
        const Vec phi = mode + L * (z * std::sqrt(nu / chi2));
        const Vec dv = L.triangularView<Eigen::Lower>().solve(phi - mode);
        const double q = dv.squaredNorm();
        const double logq = lognorm_q - 0.5 * (nu + d) * std::log(1.0 + q / nu);
        lw[uz(s)] = m.logJoint(i, phi, theta) - logq;
    }
    return logMeanExp(lw);
}

/**
 * @brief Importance-sampling marginal log-likelihood over all subjects.
 * @param m Model interface.
 * @param theta Parameter vector.
 * @param M Number of Monte Carlo draws per subject.
 * @param seed Random seed.
 * @param nu Degrees of freedom of the Student-t proposal.
 * @return The total marginal log-likelihood.
 * @throws std::invalid_argument If @p nu <= 0.
 */
inline double llis(const LogLikModel& m, const Vec& theta, int M, unsigned seed, double nu = 4.0) {
    double s = 0.0;
    for (int i = 0; i < m.nSubjects; ++i) s += llisSubject(m, theta, i, M, seed, nu);
    return s;
}

} // namespace mcmckl
