// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file robbins.hpp
 * @brief Robbins-Monro stochastic approximation for the SAEM M-step.
 *
 * Robbins-Monro (RM) solves a root-finding problem of the form
 * @f[
 *   h(\theta) = \mathbb{E}[H(\theta, Z)] = 0
 * @f]
 * with the stochastic recursion
 * @f[
 *   \theta_{t+1} = \theta_t + \gamma_t\, H(\theta_t, Z_t),
 * @f]
 * where @f$Z_t@f$ is the MCMC simulation produced by the SAEM S-step. In the
 * SAEM M-step the target root is
 * @f$\nabla_\theta Q(\theta) = 0@f$, with
 * @f$Q(\theta) = \mathbb{E}[\log p(y, \phi; \theta) \mid y]@f$.
 *
 * Convergence (Robbins-Monro conditions) requires
 * @f$\sum_t \gamma_t = \infty@f$ and @f$\sum_t \gamma_t^2 < \infty@f$.
 * This implementation uses
 * @f$\gamma_t = \gamma_0 / (t + t_0)^\alpha@f$ with
 * @f$1/2 < \alpha \le 1@f$.
 *
 * @par Robustness
 * Large steps or atypical MCMC draws can destabilize the recursion. The
 * following safeguards are available:
 *  - clipping of the stochastic gradient (bounded L2 norm);
 *  - truncation of iterates to a compact box @f$K@f$ (projection, Borkar);
 *  - Kesten's rule: decrease the step only when the update direction reverses
 *    (a sign that noise dominates, which typically happens near the optimum);
 *  - diagonal preconditioning (AdaGrad/RMSprop) for parameters with different
 *    scales.
 *
 * @par Optimality
 *  - The asymptotically optimal step exponent is @f$\alpha \approx 1@f$.
 *  - Polyak-Ruppert averaging,
 *    @f$\bar\theta_t = (1/t)\sum_{k \le t}\theta_k@f$, reaches the optimal
 *    asymptotic variance without sacrificing the convergence rate.
 *
 * Requires: Eigen3 (header only), C++20.
 */

#pragma once
#include "numcast.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Selects the SAEM M-step strategy.
 *
 * - @ref Gradient: @f$\theta \mathrel{+}= \gamma_k \nabla \log p(y,\phi;\theta)@f$
 *   (Robbins-Monro; see robbins.hpp).
 * - @ref SufficientStats: classical SAEM,
 *   @f$s_k = s_{k-1} + \gamma_k (S(\phi_k) - s_{k-1})@f$, followed by
 *   @f$\theta_k = \arg\max_\theta Q(\theta; s_k)@f$ (closed-form M-step). It has
 *   no step sizes to tune and is considerably more stable.
 * - @ref Auto: use @ref SufficientStats when the model implements it, otherwise
 *   fall back to @ref Gradient.
 */
enum class MStep { Auto, Gradient, SufficientStats };

/**
 * @brief Step-size schedule for the classical SAEM algorithm.
 *
 * Following Delyon, Lavielle & Moulines (1999) and Kuhn & Lavielle (2005):
 * @f[
 *   \gamma_k =
 *   \begin{cases}
 *     1, & k < K_1 \quad\text{(exploration phase)},\\
 *     (k - K_1 + 1)^{-\alpha}, & k \ge K_1 \quad\text{(smoothing phase)}.
 *   \end{cases}
 * @f]
 * The smoothing exponent must satisfy @f$\alpha \in (1/2, 1]@f$.
 */
struct ClassicStepControl {
    int    k1     = -1;   ///< Explicit @f$K_1@f$; negative selects the automatic rule.
    double alpha  = 1.0;  ///< Smoothing exponent, must lie in @f$(0.5, 1]@f$.
    /**
     * @brief Length of the simulated-annealing phase set by the engine.
     *
     * While the SA variance floor is active, @f$\gamma=1@f$ prevents @f$\theta@f$
     * from converging. The automatic rule therefore uses
     * @f$K_1 = \min(60\%\,n_{\text{iter}},\,300,\,sa\_len + 25)@f$.
     */
    int    sa_len = 0;

    /**
     * @brief Returns the exploration length @f$K_1@f$.
     * @param n_iter Total number of SAEM iterations.
     * @return The explicit @ref k1 when non-negative, otherwise the automatic
     *         value described above.
     */
    int K1(int n_iter) const {
        if (k1 >= 0) return k1;
        int K = std::min(static_cast<int>(0.6 * n_iter), 300);
        if (sa_len > 0) K = std::min(K, sa_len + 25);
        return K;
    }

    /**
     * @brief Evaluates the classical SAEM step size @f$\gamma_k@f$.
     * @param k Zero-based iteration index.
     * @param n_iter Total number of iterations (used by the automatic rule).
     * @return @f$1@f$ during exploration, then a power-decayed step.
     * @throws std::invalid_argument If @ref alpha is not in @f$(0.5, 1]@f$.
     */
    double gamma(int k, int n_iter) const {
        if (!(alpha > 0.5 && alpha <= 1.0)) throw std::invalid_argument("ClassicStep: alpha must be in (0.5, 1]");
        const int K1v = K1(n_iter);
        if (k < K1v) return 1.0;
        return std::pow(double(k - K1v + 1), -alpha);
    }
};

/**
 * @brief Configuration for the Robbins-Monro solver.
 *
 * All fields are optional; the defaults provide a robust power-decay schedule
 * with gradient clipping.
 */
struct RMControl {
    /** @brief Step-size schedule used by @ref RobbinsMonro. */
    enum class Schedule { Constant, OneOverT, PowerDecay, Kesten };

    Schedule schedule = Schedule::PowerDecay; ///< Step-size schedule.
    double   gamma0   = 1.0; ///< Step scale @f$\gamma_0@f$.
    double   alpha    = 0.6; ///< Decay exponent; convergence requires @f$0.5 < \alpha \le 1@f$.
    int      t0       = 1;   ///< Delay: @f$\gamma_t = \gamma_0/(t+t_0)^\alpha@f$.

    // --- Robustness ---------------------------------------------------------
    bool   clip_gradient = true;  ///< Clip the stochastic-gradient norm.
    double grad_clip     = 10.0;  ///< Maximum L2 norm of the gradient.
    bool   truncate      = false; ///< Project @f$\theta@f$ onto [lower, upper].
    Vec    lower, upper;          ///< Projection box (used when @ref truncate is true).

    // --- Optimality / scaling ----------------------------------------------
    bool   diagonal_precond = false; ///< Enable diagonal RMSprop preconditioning.
    double precond_decay    = 0.9;   ///< RMSprop decay factor.
    double precond_eps      = 1e-8;  ///< RMSprop stabilizer.

    // --- Statistical optimality: Polyak-Ruppert averaging -------------------
    bool averaging = false; ///< Enable Polyak-Ruppert averaging.
    int  avg_start = 1;     ///< First iteration included in the average.

    // --- Stopping criterion -------------------------------------------------
    int    conv_window = 200;  ///< Number of recent iterates in the convergence window.
    double conv_tol    = 1e-6; ///< Maximum per-coordinate range for convergence.
};

/**
 * @brief Robbins-Monro stochastic-approximation solver.
 *
 * The class is stateful. Typical use is:
 * @code
 * RobbinsMonro rm(RMControl{});
 * rm.init(theta0);
 * for (int k = 0; k < nIter; ++k) rm.step(stochasticGradient());
 * Vec thetaHat = rm.estimate();
 * @endcode
 */
class RobbinsMonro {
public:
    /**
     * @brief Constructs the solver and validates the configuration.
     * @param ctrl Solver configuration.
     * @throws std::invalid_argument If the configuration is inconsistent.
     */
    explicit RobbinsMonro(RMControl ctrl = RMControl()) : ctrl_(std::move(ctrl)) { validate(); }

    /**
     * @brief Resets the solver to a new starting point.
     * @param theta0 Initial parameter vector.
     */
    void init(const Vec& theta0) {
        theta_ = theta0;
        v_.setOnes(theta0.size());
        avg_ = theta0;
        avg_n_ = 0;
        prev_step_.resize(0);
        kesten_ = 0;
        t_ = 0;
        hist_.assign(uz(std::max(1, ctrl_.conv_window)), theta0);
        hist_n_ = 0;
    }

    /** @brief Returns the number of completed iterations. */
    int iteration() const { return t_; }

    /**
     * @brief Overwrites the last iterate.
     *
     * This is used, for example, to enforce the variance floor of simulated
     * annealing. Polyak averaging and the convergence window remain coherent,
     * and the previous Kesten direction is reset.
     *
     * @param th New parameter vector.
     * @throws std::invalid_argument If @p th has the wrong dimension.
     */
    void setTheta(const Vec& th) {
        if (th.size() != theta_.size()) throw std::invalid_argument("setTheta: incorrect dimension");
        if (ctrl_.averaging && avg_n_ > 0) avg_ += (th - theta_) / double(avg_n_);
        theta_ = th;
        prev_step_.resize(0);
        if (!hist_.empty() && hist_n_ > 0) hist_[(uz(hist_n_ - 1)) % hist_.size()] = th;
    }

    /**
     * @brief Performs one Robbins-Monro update.
     * @param grad_in Stochastic gradient of the objective at the current iterate.
     * @return The updated parameter vector @f$\theta_{t+1}@f$.
     */
    Vec step(const Vec& grad_in) {
        Vec g = grad_in;

        // (1) Robustness: gradient clipping.
        if (ctrl_.clip_gradient) {
            double n = g.norm();
            if (n > ctrl_.grad_clip && n > 0.0) g *= ctrl_.grad_clip / n;
        }

        // (2) Per-coordinate scaling (RMSprop), making the update scale-free.
        if (ctrl_.diagonal_precond) {
            v_ = (ctrl_.precond_decay * v_.array()
                  + (1.0 - ctrl_.precond_decay) * g.array().square()).matrix();
            g = (g.array() / (v_.array().sqrt() + ctrl_.precond_eps)).matrix();
        }

        // (3) Robustness (Kesten): advance the counter only on direction reversal.
        if (ctrl_.schedule == RMControl::Schedule::Kesten &&
            prev_step_.size() == g.size() && g.dot(prev_step_) < 0.0) {
            ++kesten_;
        }

        // (4) Step.
        double gamma = gamma_t();
        Vec theta_new = theta_ + gamma * g;

        // (5) Robustness: truncation to the compact box K.
        if (ctrl_.truncate && ctrl_.lower.size() == theta_.size() &&
            ctrl_.upper.size() == theta_.size()) {
            theta_new = theta_new.cwiseMax(ctrl_.lower).cwiseMin(ctrl_.upper);
        }

        prev_step_ = theta_new - theta_;
        theta_ = theta_new;
        ++t_;

        // (6) Optimality: Polyak-Ruppert averaging.
        if (ctrl_.averaging && t_ >= ctrl_.avg_start) {
            ++avg_n_;
            avg_ += (theta_ - avg_) / double(avg_n_);
        }

        pushHistory(theta_);
        return theta_;
    }

    /**
     * @brief Standalone driver for stochastic root finding.
     * @tparam GradFn Callable returning the stochastic gradient at a point.
     * @param grad Gradient oracle.
     * @param n_iter Number of iterations.
     * @return The recommended estimate (@ref estimate).
     */
    template <class GradFn>
    Vec solve(GradFn&& grad, int n_iter) {
        for (int t = 0; t < n_iter; ++t) step(grad(theta_));
        return estimate();
    }

    /** @brief Returns the current iterate. */
    const Vec& theta()    const { return theta_; }
    /** @brief Returns the Polyak-Ruppert average. */
    const Vec& averaged() const { return avg_; }
    /** @brief Returns the number of iterates included in the average. */
    int  avgCount()       const { return avg_n_; }

    /**
     * @brief Returns the recommended estimate.
     * @return The Polyak average when averaging is active and non-empty;
     *         otherwise the current iterate.
     */
    Vec  estimate()       const { return (ctrl_.averaging && avg_n_ > 0) ? avg_ : theta_; }

    /** @brief Returns the step size that would be used at the current iteration. */
    double currentStep() const { return gamma_t(); }

    /**
     * @brief Tests convergence using the recent iterate window.
     * @return @c true when the maximum per-coordinate range over the last
     *         @ref RMControl::conv_window iterates is below
     *         @ref RMControl::conv_tol.
     */
    bool converged() const {
        if (hist_n_ < toi(hist_.size())) return false;
        Vec lo = hist_.front(), hi = hist_.front();
        for (const auto& h : hist_) { lo = lo.cwiseMin(h); hi = hi.cwiseMax(h); }
        return (hi - lo).cwiseAbs().maxCoeff() < ctrl_.conv_tol;
    }

    /** @brief Returns the active configuration. */
    const RMControl& control() const { return ctrl_; }

private:
    RMControl ctrl_;
    Vec theta_, avg_, v_, prev_step_;
    int t_ = 0, avg_n_ = 0, kesten_ = 0;
    std::vector<Vec> hist_;
    int hist_n_ = 0;

    void validate() const {
        if (ctrl_.gamma0 <= 0) throw std::invalid_argument("gamma0 must be > 0");
        if (ctrl_.alpha <= 0)  throw std::invalid_argument("alpha must be > 0");
        if (ctrl_.schedule == RMControl::Schedule::Kesten && ctrl_.alpha <= 0.5)
            throw std::invalid_argument("Kesten schedule requires alpha > 0.5");
        if (ctrl_.truncate && ctrl_.lower.size() != ctrl_.upper.size())
            throw std::invalid_argument("lower and upper must have the same length");
    }

    double gamma_t() const {
        double t = double(t_);
        switch (ctrl_.schedule) {
            case RMControl::Schedule::Constant: return ctrl_.gamma0;
            case RMControl::Schedule::OneOverT: return ctrl_.gamma0 / (t + 1.0 + ctrl_.t0);
            case RMControl::Schedule::PowerDecay:
                return ctrl_.gamma0 / std::pow(t + 1.0 + ctrl_.t0, ctrl_.alpha);
            case RMControl::Schedule::Kesten:
                return ctrl_.gamma0 / std::pow(2.0 + kesten_, ctrl_.alpha);
        }
        return ctrl_.gamma0;
    }

    void pushHistory(const Vec& th) {
        if (hist_.empty()) return;
        hist_[uz(hist_n_) % hist_.size()] = th;
        ++hist_n_;
    }
};

} // namespace mcmckl
