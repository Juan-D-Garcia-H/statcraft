// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file mcmc.hpp
 * @brief Metropolis-Hastings with a three-kernel mixture.
 *
 * The implementation follows the kernel-mixture strategy of Kuhn & Lavielle
 * (2004, Statistics & Computing), "Coupling a stochastic approximation version
 * of EM with an MCMC algorithm".
 *
 * @par Main idea (Kuhn & Lavielle)
 * At each iteration one of several MCMC kernels is chosen at random. Each kernel
 * leaves the same target distribution @f$\pi(x)@f$ invariant, so the mixture is
 * also a @f$\pi@f$-reversible Markov kernel, while the complementary kernels
 * correct each other's weaknesses:
 *
 *  - @b Kernel 0 - Random-walk MH: proposal
 *    @f$g(x) = x + \sigma Z@f$, @f$Z \sim N(0,I)@f$. Good for local
 *    exploration; poor for multimodal or strongly elongated targets.
 *
 *  - @b Kernel 1 - Independence-sampler MH (used by K&L in SAEM-MCMC with the
 *    Gaussian posterior approximation or conditional mode): fixed proposal
 *    @f$q(x) = N(m,S)@f$. It jumps between modes and crosses low-density
 *    regions when @f$q@f$ is well centered (for example at the EBEs or at the
 *    posterior mode with its Hessian).
 *
 *  - @b Kernel 2 - Metropolis-within-Gibbs, coordinate by coordinate: updates
 *    each coordinate with a one-dimensional random-walk MH step. It breaks the
 *    coordinate correlation that slows down the multivariate random walk.
 *
 *  - @b Kernel 3 - Independence sampler with a Laplace/MAP proposal
 *    @f$N(\text{mode},(-H)^{-1})@f$; enabled through @ref
 *    MCMCKuhnLavielle::setMapProposal and weighted by @ref MCMCControl::w_map.
 *
 * Each kernel alone leaves @f$\pi@f$ invariant; the random mixture
 * @f$\pi_{\text{kernel}} = \sum_k p_k K_k@f$ does as well.
 *
 * @par API
 * @code
 * MCMCKuhnLavielle mcmc(logp, dim, seed);   // required arguments plus seed
 * mcmc.run(x0);                             // default control
 * mcmc.run(x0, control);                    // explicit control
 * @endcode
 *
 * Requires: Eigen3 (header only), C++20.
 */

#pragma once
#include "numcast.hpp"

#include <Eigen/Dense>
#include <functional>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace mcmckl {

using Vec  = Eigen::VectorXd;
using Mat  = Eigen::MatrixXd;

/** @brief Unnormalized target log density. */
using LogDens = std::function<double(const Vec&)>;

/**
 * @brief MCMC control options.
 *
 * All fields are optional. Required inputs (log density, dimension, and the
 * initial point) are passed to the constructor and to @ref
 * MCMCKuhnLavielle::run; the seed is also a constructor argument because it is
 * almost always needed.
 */
struct MCMCControl {
    // --- Kernel mixture (relative weights, normalized internally) -----------
    double w_rw    = 1.0;   ///< Kernel 0: random-walk MH.
    double w_indep = 1.0;   ///< Kernel 1: independence sampler.
    double w_gibbs = 1.0;   ///< Kernel 2: Metropolis-within-Gibbs.
    double w_map   = 0.0;   ///< Kernel 3: Laplace/MAP independence proposal (see setMapProposal).

    // --- Proposal step sizes (empty -> 0.1 for every coordinate) ------------
    Vec rw_sigma;           ///< Kernel 0 step sizes.
    Vec gibbs_sigma;        ///< Kernel 2 step sizes (empty -> @ref rw_sigma).

    // --- Independence proposal (empty -> N(0, I)) ---------------------------
    // Typical K&L choice: mean = posterior mode (EBE), covariance = inv(Hessian).
    Vec indep_mean;         ///< Independence-proposal mean.
    Mat indep_cov;          ///< Independence-proposal covariance.

    // --- Step adaptation during burn-in (per kernel) ------------------------
    bool   adaptive    = false; ///< Enable burn-in adaptation.
    double target_rate = 0.30;  ///< Target acceptance rate.
    int    adapt_every = 100;   ///< Iterations between adaptation updates.

    // --- Online adaptation (sampleFrom only, e.g. inside SAEM) --------------
    // Robbins-Monro on log(rw_sigma) with a decreasing (diminishing) step
    // toward target_rate; gibbs_sigma is rescaled by the same factor.
    bool adapt_online = false;  ///< Enable online step adaptation.

    // --- Chain length -------------------------------------------------------
    int n_iter   = 20000; ///< Total number of iterations.
    int n_burnin = 5000;  ///< Number of burn-in iterations.
    int thin     = 1;     ///< Thinning interval.
};

/**
 * @brief Output of a completed chain.
 */
struct ChainResult {
    Eigen::MatrixXd samples;        ///< Kept samples, @c n_kept x @c dim.
    std::vector<int>  kernel_used;  ///< Kernel chosen at each iteration.
    double acceptance_total = 0.0;  ///< Overall acceptance rate.
    Eigen::VectorXd      acceptance_per_kernel; ///< Rate for kernels 0..3 (NaN when unused).
    int n_iter = 0, n_burnin = 0, thin = 0;      ///< Chain configuration used.
};

/**
 * @brief Metropolis-Hastings sampler with a Kuhn-Lavielle kernel mixture.
 *
 * A sampler instance owns its RNG and must not be used concurrently from
 * multiple threads; use one instance per thread.
 */
class MCMCKuhnLavielle {
public:
    /**
     * @brief Constructs the sampler.
     * @param log_density Unnormalized target log density.
     * @param dim Dimension of the state space.
     * @param seed Random seed.
     * @param control Sampler options.
     * @throws std::invalid_argument If @p dim <= 0 or the control is invalid.
     * @throws std::runtime_error If the independence covariance is not positive
     *         definite.
     */
    MCMCKuhnLavielle(LogDens log_density, int dim, unsigned seed = 12345,
                     MCMCControl control = {})
        : logp_(std::move(log_density)), dim_(dim), rng_(seed), ctrl_(std::move(control))
    {
        if (dim_ <= 0) throw std::invalid_argument("dim must be > 0");
        prepare();                              // Validate control and derive state.
    }

    // --- Options ------------------------------------------------------------

    /**
     * @brief Replaces the full control object and rebuilds derived state.
     * @param control New options.
     * @throws std::invalid_argument If the new control is inconsistent.
     * @throws std::runtime_error If the independence covariance is not positive
     *         definite.
     */
    void setControl(MCMCControl control) { ctrl_ = std::move(control); prepare(); }

    /** @brief Returns the active control object. */
    const MCMCControl& control() const { return ctrl_; }

    // --- Execution ----------------------------------------------------------

    /**
     * @brief Runs the chain using the current control.
     * @param x0 Initial state.
     * @return The kept samples and acceptance diagnostics.
     */
    ChainResult run(const Vec& x0) { return run(x0, ctrl_); }

    /**
     * @brief Runs the chain with an explicit control.
     * @param x0 Initial state.
     * @param control Options for this run.
     * @return The kept samples and acceptance diagnostics.
     * @throws std::invalid_argument If @p x0 has the wrong dimension or the
     *         control is inconsistent.
     * @throws std::runtime_error If the initial log density is not finite or the
     *         independence covariance is not positive definite.
     */
    ChainResult run(const Vec& x0, MCMCControl control) {
        applyControl(std::move(control));
        return runImpl(x0);
    }

    /**
     * @brief Advances the chain by @p n_steps transitions and returns the final
     *        state.
     *
     * No burn-in is performed and no samples are stored. This is intended for
     * the SAEM S-step: advance the individual-effect chain by a few transitions
     * per iteration.
     *
     * @param x0 Initial state.
     * @param n_steps Number of transitions.
     * @return The final state.
     * @throws std::invalid_argument If @p x0 has the wrong dimension or
     *         @p n_steps is negative.
     * @throws std::runtime_error If no admissible point with finite density can
     *         be found.
     */
    Vec sampleFrom(const Vec& x0, int n_steps) {
        if (x0.size() != dim_) throw std::invalid_argument("x0 has incorrect dimension");
        if (n_steps < 0) throw std::invalid_argument("n_steps must be >= 0");
        Vec x = x0;
        double lp = logp_(x);
        if (!std::isfinite(lp)) {                    // Rescue: find an admissible point.
            x = ctrl_.indep_mean; lp = logp_(x);
            // If the mean is also degenerate, draw from the independence proposal
            // N(indep_mean, indep_cov) until a finite density is found.
            for (int a = 0; a < 500 && !std::isfinite(lp); ++a) {
                Vec cand = ctrl_.indep_mean + indep_chol_ * standardNormal();
                double lc = logp_(cand);
                if (std::isfinite(lc)) { x = cand; lp = lc; }
            }
            if (!std::isfinite(lp))
                throw std::runtime_error("non-finite log density in sampleFrom (no admissible point)");
        }
        for (int t = 0; t < n_steps; ++t) {
            switch (choose_(rng_)) {
                case 0: {
                    const bool acc = kernelRandomWalk(x, lp);
                    if (ctrl_.adapt_online) adaptOnline(acc);
                    break;
                }
                case 1: kernelIndependence(x, lp); break;
                case 2: kernelGibbsSweep(x, lp);   break;
                case 3: kernelMapIndependence(x, lp); break;
            }
        }
        return x;
    }

    /**
     * @brief Changes only the independence proposal.
     *
     * The full control is not revalidated and the kernel mixture is not
     * rebuilt. Intended for SAEM.
     *
     * @param mean Proposal mean.
     * @param cov Proposal covariance.
     * @throws std::runtime_error If @p cov is not positive definite.
     * @throws std::invalid_argument If the dimensions are inconsistent.
     */
    void setIndependence(const Vec& mean, const Mat& cov) {
        Eigen::LLT<Mat> llt(cov);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("indep_cov is not positive definite");
        setIndependenceChol(mean, cov, Mat(llt.matrixL()));
    }

    /**
     * @brief Changes the independence proposal using a precomputed Cholesky
     *        factor.
     *
     * Useful when the factor is shared across subjects.
     *
     * @param mean Proposal mean.
     * @param cov Proposal covariance.
     * @param chol Lower Cholesky factor of @p cov.
     * @throws std::invalid_argument If the dimensions are inconsistent.
     */
    void setIndependenceChol(const Vec& mean, const Mat& cov, const Mat& chol) {
        if (mean.size() != dim_ || cov.rows() != dim_ || cov.cols() != dim_ ||
            chol.rows() != dim_ || chol.cols() != dim_)
            throw std::invalid_argument("independence proposal has incorrect dimension");
        ctrl_.indep_mean = mean; ctrl_.indep_cov = cov; indep_chol_ = chol;
    }

    /**
     * @brief Sets the Laplace/MAP proposal of kernel 3,
     *        @f$N(\text{mean}, LL^\top)@f$.
     *
     * The kernel weight is @ref MCMCControl::w_map.
     *
     * @param mean Proposal mean.
     * @param chol Lower Cholesky factor of the proposal covariance.
     * @throws std::invalid_argument If the dimensions are inconsistent.
     */
    void setMapProposal(const Vec& mean, const Mat& chol) {
        if (mean.size() != dim_ || chol.rows() != dim_ || chol.cols() != dim_)
            throw std::invalid_argument("MAP proposal has incorrect dimension");
        map_mean_ = mean; map_chol_ = chol; map_ready_ = true;
        map_logdet_ = 2.0 * map_chol_.diagonal().array().log().sum();
    }

    /**
     * @brief Changes the random-walk and Metropolis-within-Gibbs step sizes.
     * @param rw Random-walk step sizes.
     * @param gibbs Gibbs step sizes.
     * @throws std::invalid_argument If a dimension is wrong or a step is
     *         non-positive.
     */
    void setSteps(const Vec& rw, const Vec& gibbs) {
        if (rw.size() != dim_ || gibbs.size() != dim_) throw std::invalid_argument("step vectors have incorrect dimension");
        if (rw.minCoeff() <= 0 || gibbs.minCoeff() <= 0) throw std::invalid_argument("step sizes must be > 0");
        ctrl_.rw_sigma = rw; ctrl_.gibbs_sigma = gibbs;
    }

    // State accessors (for coupling with SAEM).
    /** @brief Returns the independence-proposal mean. */
    const Vec&  independenceMean()  const { return ctrl_.indep_mean; }
    /** @brief Returns the independence-proposal covariance. */
    const Mat&  independenceCov()   const { return ctrl_.indep_cov; }

private:
    LogDens logp_;
    int dim_;
    std::mt19937 rng_;
    MCMCControl ctrl_;

    // State derived from the control (recomputed in prepare()).
    std::vector<double> kernel_weights_;
    Vec  map_mean_;                            // Laplace proposal: N(map_mean_, L L').
    Mat  map_chol_;
    double map_logdet_ = 0.0;                  // Log-det of the MAP covariance (precomputed).
    double indep_logdet_ = 0.0;                // Same, for the independence proposal.
    bool map_ready_ = false;
    Vec  wz_, wx_, wd_;                        // Workspace (one sampler per thread at a time).
    std::discrete_distribution<int> choose_;   // Rebuilt only in prepare().
    long n_adapt_ = 0;                         // Online-adaptation counter.
    Mat indep_chol_;                           // Cholesky factor of the independence proposal.

    // Validates the control and computes defaults/derived values.
    void prepare() {
        // Normalized weights.
        double s = ctrl_.w_rw + ctrl_.w_indep + ctrl_.w_gibbs + ctrl_.w_map;
        if (s <= 0) throw std::invalid_argument("kernel weights must sum to > 0");
        kernel_weights_ = { ctrl_.w_rw / s, ctrl_.w_indep / s, ctrl_.w_gibbs / s, ctrl_.w_map / s };
        choose_ = std::discrete_distribution<int>(kernel_weights_.begin(), kernel_weights_.end());

        // Default step sizes.
        if (ctrl_.rw_sigma.size() == 0)    ctrl_.rw_sigma.setConstant(dim_, 0.1);
        if (ctrl_.gibbs_sigma.size() == 0) ctrl_.gibbs_sigma = ctrl_.rw_sigma;
        if (ctrl_.rw_sigma.size()    != dim_ ||
            ctrl_.gibbs_sigma.size() != dim_)
            throw std::invalid_argument("rw_sigma/gibbs_sigma have incorrect dimension");
        if (ctrl_.rw_sigma.minCoeff()    <= 0 ||
            ctrl_.gibbs_sigma.minCoeff() <= 0)
            throw std::invalid_argument("step sizes must be > 0");

        // Default independence proposal: N(0, I).
        if (ctrl_.indep_mean.size() == 0) ctrl_.indep_mean.setZero(dim_);
        if (ctrl_.indep_cov.size() == 0)  ctrl_.indep_cov.setIdentity(dim_, dim_);
        if (ctrl_.indep_mean.size() != dim_ ||
            ctrl_.indep_cov.rows()  != dim_ || ctrl_.indep_cov.cols() != dim_)
            throw std::invalid_argument("indep_mean/indep_cov have incorrect dimension");

        // Cholesky factorization once (Eigen).
        Eigen::LLT<Mat> llt(ctrl_.indep_cov);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("indep_cov is not positive definite");
        indep_chol_ = llt.matrixL();
        indep_logdet_ = 2.0 * indep_chol_.diagonal().array().log().sum();

        // Chain validation.
        if (ctrl_.n_iter <= ctrl_.n_burnin)
            throw std::invalid_argument("n_iter must exceed n_burnin");
        if (ctrl_.thin < 1) throw std::invalid_argument("thin must be >= 1");
        if (ctrl_.adapt_every < 1) throw std::invalid_argument("adapt_every must be >= 1");
    }

    // For run(x0, control): avoid repeating setControl (which already calls prepare).
    void applyControl(MCMCControl control) { ctrl_ = std::move(control); prepare(); }

    ChainResult runImpl(const Vec& x0) {
        if (x0.size() != dim_)
            throw std::invalid_argument("x0 has incorrect dimension");

        const int n_iter = ctrl_.n_iter, n_burnin = ctrl_.n_burnin, thin = ctrl_.thin;

        Vec x = x0;
        double lp = logp_(x);
        if (!std::isfinite(lp)) throw std::runtime_error("initial log density is not finite");

        // Acceptance accumulators per kernel (totals, for reporting).
        std::vector<int>  n_try(4, 0), n_acc(4, 0);
        // Adaptation window per kernel (reset every adapt_every iterations).
        std::vector<int>  w_try(4, 0), w_acc(4, 0);

        ChainResult res;
        res.samples.resize((n_iter - n_burnin) / thin + 1, dim_);
        res.n_iter = n_iter; res.n_burnin = n_burnin; res.thin = thin;
        int kept = 0, acc_total = 0;

        for (int it = 0; it < n_iter; ++it) {
            int k = choose_(rng_);            // K&L: random kernel mixture.
            bool acc = false;
            switch (k) {
                case 0: acc = kernelRandomWalk(x, lp);        break;
                case 1: acc = kernelIndependence(x, lp);      break;
                case 2: acc = kernelGibbsSweep(x, lp);        break;
                case 3: acc = kernelMapIndependence(x, lp);   break;
            }
            ++n_try[uz(k)]; n_acc[uz(k)] += acc; acc_total += acc;
            ++w_try[uz(k)]; w_acc[uz(k)] += acc;
            res.kernel_used.push_back(k);

            // Adaptation ONLY during burn-in and PER KERNEL: the RW step is
            // adjusted using the kernel-0 rate and the Gibbs step using the
            // kernel-2 rate. (The independence kernel has a fixed proposal and
            // is not adapted.) After burn-in the kernel is frozen, so the chain
            // is stationary.
            if (ctrl_.adaptive && it < n_burnin && (it + 1) % ctrl_.adapt_every == 0) {
                if (w_try[0] > 0) {
                    double r = double(w_acc[0]) / w_try[0];
                    double f = std::min(2.0, std::max(0.5, std::exp(0.5 * (r - ctrl_.target_rate))));
                    ctrl_.rw_sigma *= f;
                }
                if (w_try[2] > 0) {
                    double r = double(w_acc[2]) / w_try[2];
                    double f = std::min(2.0, std::max(0.5, std::exp(0.5 * (r - ctrl_.target_rate))));
                    ctrl_.gibbs_sigma *= f;
                }
                std::fill(w_try.begin(), w_try.end(), 0);
                std::fill(w_acc.begin(), w_acc.end(), 0);
            }

            if (it >= n_burnin && ((it - n_burnin) % thin == 0) && kept < res.samples.rows()) {
                res.samples.row(kept++) = x.transpose();
            }
        }
        res.samples.conservativeResize(kept, Eigen::NoChange);
        res.acceptance_total = static_cast<double>(acc_total) / n_iter;
        res.acceptance_per_kernel.resize(4);
        for (int k = 0; k < 4; ++k)
            res.acceptance_per_kernel[k] = n_try[uz(k)] ? static_cast<double>(n_acc[uz(k)]) / n_try[uz(k)]
                                                    : std::numeric_limits<double>::quiet_NaN();
        return res;
    }

    // Online adaptation of the RW step (Robbins-Monro, step 1/(10+n)^0.6).
    void adaptOnline(bool accepted) {
        const double g = 1.0 / std::pow(10.0 + double(n_adapt_++), 0.6);
        const double f = std::exp(g * ((accepted ? 1.0 : 0.0) - ctrl_.target_rate));
        ctrl_.rw_sigma    = (ctrl_.rw_sigma * f).cwiseMax(1e-10).cwiseMin(1e10);
        ctrl_.gibbs_sigma = (ctrl_.gibbs_sigma * f).cwiseMax(1e-10).cwiseMin(1e10);
    }

    double logUniform() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return std::log(u(rng_));
    }

    Vec standardNormal() {
        std::normal_distribution<double> n(0.0, 1.0);
        Vec z(dim_);
        for (int i = 0; i < dim_; ++i) z[i] = n(rng_);
        return z;
    }

    // Same as standardNormal() (same random stream) but without allocating memory.
    void fillNormal(Vec& z) {
        std::normal_distribution<double> n(0.0, 1.0);
        z.resize(dim_);
        for (int i = 0; i < dim_; ++i) z[i] = n(rng_);
    }

    // --- Kernel 0: Random-Walk Metropolis -----------------------------------
    bool kernelRandomWalk(Vec& x, double& lp) {
        fillNormal(wz_);
        wx_.resize(dim_);
        for (int i = 0; i < dim_; ++i) wx_[i] = x[i] + ctrl_.rw_sigma[i] * wz_[i];
        double lpp = logp_(wx_);
        if (!std::isfinite(lpp)) return false;
        if (logUniform() < lpp - lp) { x = wx_; lp = lpp; return true; }
        return false;
    }

    // --- Kernel 1: Independence Sampler -------------------------------------
    // q(x' | x) = N(indep_mean, indep_cov)  ->  r = exp(lp' - lp) * q(x)/q(x').
    bool kernelIndependence(Vec& x, double& lp) {
        fillNormal(wz_);
        wx_.noalias() = indep_chol_ * wz_;
        wx_ += ctrl_.indep_mean;
        double lpp = logp_(wx_);
        if (!std::isfinite(lpp)) return false;
        double lq_x  = logGaussian(x,   ctrl_.indep_mean, indep_chol_, indep_logdet_);
        double lq_xp = logGaussian(wx_, ctrl_.indep_mean, indep_chol_, indep_logdet_);
        if (logUniform() < (lpp - lp) + (lq_x - lq_xp)) { x = wx_; lp = lpp; return true; }
        return false;
    }

    static double logGaussian(const Vec& z, const Vec& mu, const Mat& cholL) {
        Vec d = cholL.triangularView<Eigen::Lower>().solve(z - mu);
        double quad = d.squaredNorm();
        double logdet = 2.0 * cholL.diagonal().array().log().sum();
        return -0.5 * (quad + logdet + double(mu.size()) * std::log(2.0 * std::numbers::pi));
    }

    // Allocation-free version with a precomputed log-determinant (same value).
    double logGaussian(const Vec& z, const Vec& mu, const Mat& cholL, double logdet) {
        wd_ = z - mu;
        cholL.triangularView<Eigen::Lower>().solveInPlace(wd_);
        return -0.5 * (wd_.squaredNorm() + logdet + double(mu.size()) * std::log(2.0 * std::numbers::pi));
    }

    // --- Kernel 3: independence with a Laplace/MAP proposal ------------------
    // q(x') = N(conditional mode, (-H)^{-1}): the proposal depends on theta and
    // the data but NOT on the chain state, so it is a valid independence kernel.
    // Useful when the data are much more informative than the prior.
    bool kernelMapIndependence(Vec& x, double& lp) {
        if (!map_ready_) return false;
        fillNormal(wz_);
        wx_.noalias() = map_chol_ * wz_;
        wx_ += map_mean_;
        double lpp = logp_(wx_);
        if (!std::isfinite(lpp)) return false;
        double lq_x  = logGaussian(x,   map_mean_, map_chol_, map_logdet_);
        double lq_xp = logGaussian(wx_, map_mean_, map_chol_, map_logdet_);
        if (logUniform() < (lpp - lp) + (lq_x - lq_xp)) { x = wx_; lp = lpp; return true; }
        return false;
    }

    // --- Kernel 2: Metropolis-within-Gibbs (coordinate sweep) ---------------
    bool kernelGibbsSweep(Vec& x, double& lp) {
        bool any = false;
        std::normal_distribution<double> n(0.0, 1.0);
        for (int i = 0; i < dim_; ++i) {
            const double old = x[i];                    // Modify in place and restore on rejection.
            x[i] = old + ctrl_.gibbs_sigma[i] * n(rng_);
            double lpp = logp_(x);
            if (std::isfinite(lpp) && logUniform() < lpp - lp) {
                lp = lpp; any = true;
            } else x[i] = old;
        }
        return any;
    }
};

} // namespace mcmckl
