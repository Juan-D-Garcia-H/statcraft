/**
 * @file map.hpp
 * @brief Post-fit individual MAP estimates for the generic likelihood model.
 *
 * For each subject the maximum a posteriori estimate of the individual
 * parameter is computed as
 *
 *   phi_i^MAP = argmax_phi p(phi_i | y_i, theta_hat)
 *             = argmax_phi log p(y_i | phi_i) + log p(phi_i | theta_hat)
 *
 * because the marginal normalization p(y_i | theta) does not depend on phi.
 * The full conditional log density supplied by LogLikModel::logJoint is
 * maximized with the derivative-free Nelder-Mead method.
 *
 * Requires: Eigen3, C++20, nelder.hpp, loglik.hpp.
 */

#pragma once
#include "numcast.hpp"

#include "nelder.hpp"
#include "loglik.hpp"

#include <Eigen/Dense>
#include <vector>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Computes the MAP estimate of the individual parameter for one subject.
 *
 * Maximizes the full conditional log density provided by
 * LogLikModel::logJoint using Nelder-Mead. The initial simplex is scaled by
 * 0.3 times the square root of the diagonal of Omega (floored at 1e-12).
 *
 * @param m Likelihood model providing logJoint and priorParams.
 * @param theta Parameter vector at which the conditional is evaluated.
 * @param i Subject index.
 * @param phi0 Starting point of the optimization.
 * @param max_iter Maximum number of Nelder-Mead iterations.
 * @param converged Optional output; when non-null it receives the convergence
 *        diagnostic reported by the optimizer.
 * @return The MAP estimate of phi_i.
 *
 * @note This function keeps no mutable state of its own; thread safety
 *       depends on the supplied model and its callbacks.
 */
inline Vec mapEstimate(const LogLikModel& m, const Vec& theta, int i,
                       const Vec& phi0, int max_iter = 1000, bool* converged = nullptr) {
    auto neg = [&](const Vec& p) { return -m.logJoint(i, p, theta); };
    Vec mu; Mat Omega;
    m.priorParams(theta, mu, Omega);
    const Vec sc = 0.3 * Omega.diagonal().cwiseMax(1e-12).cwiseSqrt();
    const NMOutcome r = nelderMeadEx(neg, phi0, sc, max_iter);
    if (converged) *converged = r.converged;
    return r.x;
}

/**
 * @brief Computes the MAP estimate of the individual parameter starting from
 *        the population mean.
 *
 * @param m Likelihood model providing logJoint and priorParams.
 * @param theta Parameter vector at which the conditional is evaluated.
 * @param i Subject index.
 * @param max_iter Maximum number of Nelder-Mead iterations.
 * @return The MAP estimate of phi_i.
 */
inline Vec mapEstimate(const LogLikModel& m, const Vec& theta, int i, int max_iter = 1000) {
    Vec mu; Mat Omega;
    m.priorParams(theta, mu, Omega);
    return mapEstimate(m, theta, i, mu, max_iter);
}

/**
 * @brief Computes MAP estimates for all subjects.
 *
 * Every subject is started from the population mean returned by
 * LogLikModel::priorParams.
 *
 * @param m Likelihood model providing logJoint and priorParams.
 * @param theta Parameter vector at which the conditionals are evaluated.
 * @param max_iter Maximum number of Nelder-Mead iterations per subject.
 * @return A vector with one MAP estimate per subject.
 */
inline std::vector<Vec> mapEstimates(const LogLikModel& m, const Vec& theta, int max_iter = 1000) {
    std::vector<Vec> out(uz(m.nSubjects));
    Vec mu; Mat Omega;
    m.priorParams(theta, mu, Omega);
    for (int i = 0; i < m.nSubjects; ++i)
        out[uz(i)] = mapEstimate(m, theta, i, mu, max_iter, nullptr);
    return out;
}

/**
 * @brief Evaluates the log posterior of phi_i at the mode.
 *
 * Returns log p(phi_i | y_i, theta) up to the additive constant
 * log p(y_i | theta), which does not depend on phi.
 *
 * @param m Likelihood model providing logJoint.
 * @param theta Parameter vector.
 * @param i Subject index.
 * @param phi Individual parameter value.
 * @return The full conditional log density at phi.
 */
inline double mapLogPost(const LogLikModel& m, const Vec& theta, int i, const Vec& phi) {
    return m.logJoint(i, phi, theta);
}

} // namespace mcmckl
