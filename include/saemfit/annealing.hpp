// ============================================================================
// @file annealing.hpp
// @brief Simulated annealing for SAEM (exploratory phase).
//
// Simulated Annealing for SAEM (exploratory phase) following Kuhn & Lavielle
// (2005) / saemix: it prevents PREMATURE convergence of the random-effect
// variances (Omega, Psi).
//
// MECHANISM (the part that changes the estimate):
//   In the M-step, during the SA phase, the variances CANNOT fall faster than
//   a factor `shrink` per iteration:
//       Omega_{k+1} >= shrink * Omega_k          (shrink ~ 0.98)
//   i.e. log sd_{k+1} >= log sd_k + 0.5*log(shrink)   (theta stores log sd).
//   The phase switches itself off: it stays active while beta_k is
//   appreciably > 1.
//
// OPTIONAL (efficiency only, does NOT change the estimate):
//   `inflate_proposal` also inflates the independence proposal of the MCMC
//   with beta_k (Omega_k = beta_k * Omega). The stationary distribution of the
//   kernel is unchanged; it only helps exploration. Disabled by default.
//
// Schedules for the factor beta_k >= 1 (governs when the SA phase ends):
//   Geometric : beta_k = 1 + (beta0 - 1) * r^k   (r = 0.95 by default)
//   Exponential: beta_k = 1 + (beta0 - 1) * exp(-k / T)
//   Power     : beta_k = 1 + (beta0 - 1) * max(0, 1 - k/K)^p
//   Inverse   : beta_k = 1 + (beta0 - 1) / (1 + k / T)
//
// Requires: Eigen3 (header only), C++20.
// ============================================================================

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Configuration and controls for the simulated-annealing phase.
 *
 * All members are public configuration knobs; the accompanying free functions
 * (@ref saBeta, @ref saActive, @ref saPhaseLength, @ref inflateOmega,
 * @ref covDiagIndices and @ref saApplyVarianceFloor) implement the schedules
 * and the variance floor.
 */
struct SAControl {
    /**
     * @brief Annealing schedule used to decay the inflation factor beta_k.
     */
    enum class Schedule {
        Geometric,   ///< Geometric decay, beta_k = 1 + (beta0-1)*r^k.
        Exponential, ///< Exponential decay, beta_k = 1 + (beta0-1)*exp(-k/T).
        Power,       ///< Power decay, beta_k = 1 + (beta0-1)*max(0, 1-k/K)^p.
        Inverse      ///< Inverse decay, beta_k = 1 + (beta0-1)/(1+k/T).
    };

    Schedule schedule = Schedule::Geometric; ///< Annealing schedule.
    double beta0 = 10.0;   ///< Initial inflation (beta0 >= 1); beta0 == 1 disables SA.
    double rate  = 0.95;   ///< Geometric: r in (0, 1). 0.95 -> ~135 iterations of SA (0.97 -> ~227).
    double T     = 30.0;   ///< Exponential / Inverse: time constant.
    double K     = 200.0;  ///< Power: iterations until switch-off.
    double power = 2.0;    ///< Power: exponent.

    // --- SA in the M-step (K&L mechanism) ---
    double shrink = 0.98;  ///< Omega_{k+1} >= shrink * Omega_k during the SA phase, in (0, 1].

    // --- optional: additionally widen the independence proposal ---
    bool inflate_proposal = false; ///< Whether to inflate the MCMC proposal.
    bool diagonal_only = false;    ///< Inflate only the (diagonal) variances of the proposal; by default the whole matrix.
};

/**
 * @brief Compute the inflation factor beta_k (>= 1, tending to 1).
 *
 * @param k Iteration index.
 * @param c Annealing controls.
 * @return The inflation factor beta_k.
 *
 * @throws std::invalid_argument If `c.beta0 < 1`.
 */
inline double saBeta(int k, const SAControl& c) {
    if (c.beta0 < 1.0) throw std::invalid_argument("beta0 must be >= 1");
    double decay;
    switch (c.schedule) {
        case SAControl::Schedule::Geometric:   decay = std::pow(c.rate, double(k));                       break;
        case SAControl::Schedule::Exponential: decay = std::exp(-double(k) / c.T);                        break;
        case SAControl::Schedule::Power:
            decay = std::pow(std::max(0.0, 1.0 - double(k) / c.K), c.power);                              break;
        case SAControl::Schedule::Inverse:     decay = 1.0 / (1.0 + double(k) / c.T);                     break;
        default: decay = 0.0;
    }
    return 1.0 + (c.beta0 - 1.0) * decay;
}

/**
 * @brief Whether the SA phase is still active at iteration @p k.
 *
 * The phase is active while the excess of beta_k exceeds 0.1% of its initial
 * value.
 *
 * @param k Iteration index.
 * @param c Annealing controls.
 * @return True if the SA phase is active.
 */
inline bool saActive(int k, const SAControl& c) {
    if (!(c.beta0 > 1.0)) return false;
    return (saBeta(k, c) - 1.0) > 1e-3 * (c.beta0 - 1.0);
}

/**
 * @brief Number of iterations of the SA phase (bounded by @p max_iter).
 *
 * @param c        Annealing controls.
 * @param max_iter Maximum number of iterations to consider.
 * @return The number of active SA iterations.
 */
inline int saPhaseLength(const SAControl& c, int max_iter) {
    int k = 0;
    while (k < max_iter && saActive(k, c)) ++k;
    return k;
}

/**
 * @brief Inflated Omega at iteration @p k (only for the MCMC proposal).
 *
 * @param Omega Covariance matrix to inflate.
 * @param k     Iteration index.
 * @param c     Annealing controls.
 * @return The matrix beta_k * Omega, or only its diagonal scaled by beta_k when
 *         @ref SAControl::diagonal_only is set.
 */
inline Mat inflateOmega(const Mat& Omega, int k, const SAControl& c) {
    const double beta = saBeta(k, c);
    if (c.diagonal_only) {
        Mat out = Omega;
        out.diagonal() *= beta;
        return out;
    }
    return beta * Omega;
}

/**
 * @brief Indices (in theta) of the log-sd entries of a covariance block.
 *
 * The block is parameterized as in model.hpp/multilevel.hpp: diagonal ->
 * log sd; full -> Cholesky (diagonal in log, off-diagonal raw, by columns).
 *
 * @param off  Offset of the block within theta.
 * @param p    Dimension of the covariance block.
 * @param full Whether the block is a full Cholesky factorization.
 * @return The theta indices holding the log-sd (diagonal) entries.
 */
inline std::vector<int> covDiagIndices(int off, int p, bool full) {
    std::vector<int> idx;
    if (!full) {
        for (int j = 0; j < p; ++j) idx.push_back(off + j);
        return idx;
    }
    int k = off;
    for (int j = 0; j < p; ++j) for (int i = j; i < p; ++i) {
        if (i == j) idx.push_back(k);
        ++k;
    }
    return idx;
}

/**
 * @brief Apply the SA floor of the M-step: log sd_new >= log sd_old + 0.5*log(shrink).
 *
 * @param theta_new Updated parameter vector, modified in place.
 * @param theta_old Previous parameter vector.
 * @param idx       Indices of the log-sd entries to constrain.
 * @param k         Iteration index.
 * @param c         Annealing controls.
 * @return True if @p theta_new was modified.
 *
 * @throws std::invalid_argument If `c.shrink` is not in (0, 1].
 */
inline bool saApplyVarianceFloor(Vec& theta_new, const Vec& theta_old,
                                 const std::vector<int>& idx, int k, const SAControl& c) {
    if (idx.empty() || !saActive(k, c)) return false;
    if (!(c.shrink > 0.0 && c.shrink <= 1.0)) throw std::invalid_argument("shrink must be in (0, 1]");
    const double lo = 0.5 * std::log(c.shrink);
    bool changed = false;
    for (int j : idx) {
        const double fl = theta_old[j] + lo;
        if (theta_new[j] < fl) { theta_new[j] = fl; changed = true; }
    }
    return changed;
}

} // namespace mcmckl