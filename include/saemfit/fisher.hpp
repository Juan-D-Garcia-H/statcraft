// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file fisher.hpp
 * @brief Fisher information matrix (FIM) and standard errors.
 *
 * The FIM of @f$\theta@f$ is obtained by numerical differentiation of the
 * individual score:
 * @f[
 *   s_i(\theta) = \frac{\partial}{\partial \theta}\log p(y_i \mid \theta),
 * @f]
 * @f[
 *   I(\theta) = \sum_i s_i(\theta)s_i(\theta)^\top
 *             \quad\text{(BHHH / outer product of gradients)},
 * @f]
 * or, using the observed information,
 * @f[
 *   I(\theta) = -\sum_i \frac{\partial^2}{\partial \theta^2}
 *               \log p(y_i \mid \theta).
 * @f]
 *
 * The user supplies @f$\log p(y_i \mid \theta)@f$ through @ref IndivLogLik
 * (for example @c llgqSubject or @c llisSubject from loglik.hpp). Scores are
 * approximated by central finite differences.
 *
 * Requires: Eigen3, C++20.
 */

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Per-subject log-likelihood @f$\log p(y_i \mid \theta)@f$.
 *
 * The first argument is the subject index and the second is the parameter
 * vector.
 */
using IndivLogLik = std::function<double(int i, const Vec& theta)>;

/**
 * @brief Computes the individual score by central finite differences.
 *
 * Uses the relative step
 * @f$h_j = h\max(1, |\theta_j|)@f$ for each coordinate.
 *
 * @param ll Per-subject log-likelihood.
 * @param i Subject index.
 * @param theta Parameter vector.
 * @param h Relative step size.
 * @return The score vector @f$s_i(\theta)@f$.
 */
inline Vec scoreFD(const IndivLogLik& ll, int i, const Vec& theta, double h = 1e-4) {
    const int p = static_cast<int>(theta.size());
    Vec g(p);
    Vec xp = theta, xm = theta;
    for (int j = 0; j < p; ++j) {
        const double hj = h * std::max(1.0, std::abs(theta[j]));
        xp = theta; xm = theta;
        xp[j] += hj; xm[j] -= hj;
        g[j] = (ll(i, xp) - ll(i, xm)) / (2.0 * hj);
    }
    return g;
}

/**
 * @brief Builds the FIM from the outer product of individual scores.
 *
 * Computes @f$I = \sum_i s_i s_i^\top@f$ (BHHH estimator).
 *
 * @param ll Per-subject log-likelihood.
 * @param n Number of subjects.
 * @param theta Parameter vector.
 * @param h Relative step size used by @ref scoreFD.
 * @return The BHHH Fisher information matrix.
 */
inline Mat fimBHHH(const IndivLogLik& ll, int n, const Vec& theta, double h = 1e-4) {
    const int p = static_cast<int>(theta.size());
    Mat I = Mat::Zero(p, p);
    for (int i = 0; i < n; ++i) {
        const Vec s = scoreFD(ll, i, theta, h);
        I += s * s.transpose();
    }
    return I;
}

/**
 * @brief Observed FIM from the numerical Hessian of the total log-likelihood.
 *
 * Computes @f$I = -\nabla^2 \sum_i \log p(y_i \mid \theta)@f$ with central
 * differences.
 *
 * @note The second difference amplifies log-likelihood noise by @f$1/h^2@f$.
 *       If the log-likelihood comes from a Nelder-Mead mode (@c tol ~ 1e-12) or
 *       from Monte Carlo, @f$h=10^{-4}@f$ leaves the Hessian dominated by noise;
 *       the default is therefore @f$h=10^{-3}@f$. See also
 *       @ref fimObservedFromScores.
 *
 * @param ll Per-subject log-likelihood.
 * @param n Number of subjects.
 * @param theta Parameter vector.
 * @param h Relative step size.
 * @return The observed Fisher information matrix.
 */
inline Mat fimObserved(const IndivLogLik& ll, int n, const Vec& theta, double h = 1e-3) {
    const int p = static_cast<int>(theta.size());
    Mat H = Mat::Zero(p, p);
    Vec xp = theta, xm = theta;
    auto total = [&](const Vec& th) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += ll(i, th);
        return s;
    };
    const double f0 = total(theta);
    // Diagonal.
    for (int j = 0; j < p; ++j) {
        const double hj = h * std::max(1.0, std::abs(theta[j]));
        xp = theta; xm = theta; xp[j] += hj; xm[j] -= hj;
        H(j, j) = (total(xp) - 2.0 * f0 + total(xm)) / (hj * hj);
    }
    // Off-diagonal.
    for (int j = 0; j < p; ++j) for (int k = j + 1; k < p; ++k) {
        const double hj = h * std::max(1.0, std::abs(theta[j]));
        const double hk = h * std::max(1.0, std::abs(theta[k]));
        Vec a = theta, b = theta, c = theta, d = theta;
        a[j] += hj; a[k] += hk;
        b[j] += hj; b[k] -= hk;
        c[j] -= hj; c[k] += hk;
        d[j] -= hj; d[k] -= hk;
        const double v = (total(a) - total(b) - total(c) + total(d)) / (4.0 * hj * hk);
        H(j, k) = v; H(k, j) = v;
    }
    return -H;   // Observed information.
}

/**
 * @brief Observed FIM from differences of scores.
 *
 * Computes @f$H_{jk} = \partial s_j/\partial\theta_k@f$, where
 * @f$s = \sum_i \text{scoreFD}_i@f$ is a first-order central difference. The
 * log-likelihood noise is amplified by @f$1/(h h_2)@f$ instead of
 * @f$1/h^2@f$, so a larger @p h_hess can be used for the second level without
 * biasing the inner score. The result is symmetrized.
 *
 * @param ll Per-subject log-likelihood.
 * @param n Number of subjects.
 * @param theta Parameter vector.
 * @param h_score Relative step for the inner score.
 * @param h_hess Relative step for the Hessian.
 * @return The observed Fisher information matrix.
 */
inline Mat fimObservedFromScores(const IndivLogLik& ll, int n, const Vec& theta,
                                 double h_score = 1e-3, double h_hess = 1e-2) {
    const int p = static_cast<int>(theta.size());
    auto total_score = [&](const Vec& th) {
        Vec s = Vec::Zero(p);
        for (int i = 0; i < n; ++i) s += scoreFD(ll, i, th, h_score);
        return s;
    };
    Mat H(p, p);
    for (int k = 0; k < p; ++k) {
        const double hk = h_hess * std::max(1.0, std::abs(theta[k]));
        Vec tp = theta, tm = theta;
        tp[k] += hk; tm[k] -= hk;
        H.col(k) = (total_score(tp) - total_score(tm)) / (2.0 * hk);
    }
    return -0.5 * (H + H.transpose());
}

/**
 * @brief Computes standard errors from a Fisher information matrix.
 *
 * Returns @f$\sqrt{\operatorname{diag}(I^{-1})}@f$. The inverse is computed
 * through the spectral decomposition with a positive floor, which avoids NaNs
 * when the FIM is nearly singular or has small negative eigenvalues caused by
 * numerical noise.
 *
 * @param fim Fisher information matrix.
 * @return Vector of standard errors; entries are NaN when the corresponding
 *         variance is not positive.
 */
inline Vec stdErrors(const Mat& fim) {
    Eigen::SelfAdjointEigenSolver<Mat> es(0.5 * (fim + fim.transpose()));
    Vec d = es.eigenvalues();
    Mat V = es.eigenvectors();
    const double dmax = d.cwiseAbs().maxCoeff();
    const double floor_eps = std::max(1e-12, 1e-8 * (dmax > 0.0 ? dmax : 1.0));
    Vec dinv(d.size());
    for (int i = 0; i < d.size(); ++i) dinv[i] = 1.0 / std::max(d[i], floor_eps);
    const Mat inv = V * dinv.asDiagonal() * V.transpose();
    Vec se(dinv.size());
    for (int i = 0; i < inv.rows(); ++i)
        se[i] = (inv(i, i) > 0.0) ? std::sqrt(inv(i, i)) : std::numeric_limits<double>::quiet_NaN();
    return se;
}

} // namespace mcmckl
