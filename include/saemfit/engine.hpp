// ============================================================================
// @file engine.hpp
// @brief SAEM (Stochastic Approximation EM) of Kuhn & Lavielle.
//
//   S/E-step : simulation of the individual effect chi_i ~ p(chi_i | y_i, theta)
//              with an MCMC of up to 4 kernels (see mcmc.hpp): random walk,
//              independence N(prior), Metropolis-within-Gibbs and independence
//              with a LAPLACE/MAP proposal (conditional mode + Hessian).
//              Subjects (and chains) are simulated IN PARALLEL (n_threads).
//   M-step   : (a) CLASSIC SAEM: stochastic approximation of sufficient
//                 statistics + closed-form maximization (+ numerical step for
//                 parameters without a closed form), or
//              (b) Robbins-Monro on the stochastic gradient (robbins.hpp).
//   Simulated Annealing (K&L): variance shrinkage floor.
//   Convergence diagnostics and optional automatic stopping.
//
// The model is injected through SaemModel.
//
// Requires: Eigen3 (header only), C++20, mcmc.hpp, robbins.hpp, annealing.hpp,
//           nelder.hpp, loglik.hpp, parallel.hpp.
// ============================================================================

#pragma once
#include "numcast.hpp"

#include "mcmc.hpp"
#include "robbins.hpp"
#include "annealing.hpp"
#include "nelder.hpp"
#include "loglik.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mcmckl {

// ---------------------------------------------------------------------------
// SAEM control (all fields optional).
// ---------------------------------------------------------------------------

/// @brief Runtime controls for the SAEM algorithm.
struct SAEMControl {
    MCMCControl mcmc;              ///< Sampler options (S-step).
    RMControl   rm;                ///< Gradient M-step options (Robbins-Monro).
    MStep       mstep = MStep::Auto;   ///< Gradient | SufficientStats (classic SAEM) | Auto.
    ClassicStepControl step;           ///< Classic SAEM step schedule.
    int  n_saem_iter   = 300;      ///< Total iterations (maximum when auto_stop is set).
    int  n_mcmc_sweeps = 3;        ///< MCMC transitions per subject and iteration.
    int  n_mcmc_burnin = 40;       ///< Extra transitions on the first iteration.
    bool verbose       = false;    ///< Print progress when true.
    int  print_every   = 50;       ///< Iterations between progress prints.

    // Simulated Annealing (K&L): variance shrinkage floor.
    bool     simulated_annealing = false; ///< Enable simulated annealing.
    SAControl sa;                         ///< Simulated annealing controls.

    int  n_chains = 1;             ///< MCMC chains per subject (gradient/statistics are averaged).

    // RW/Gibbs steps: initial population scale + online adaptation.
    bool adapt_rw = true;

    // Parallelism over subjects: 1 = serial (default), 0 = all threads.
    // The user prediction must be thread-safe.
    int  n_threads = 1;

    // LAPLACE/MAP kernel: independence centered at each subject's conditional
    // mode with covariance (-H)^{-1}; refreshed every `map_every` iterations.
    bool   map_kernel = true;
    double map_weight = 1.0;
    int    map_every  = 25;

    // Automatic stopping by theta drift (disabled by default).
    bool   auto_stop       = false;
    double stop_tol        = 1e-3;     ///< Maximum relative drift in a window.
    int    stop_window     = 50;       ///< Window length in iterations.
    int    stop_min_phase2 = 50;       ///< Minimum iterations after the exploratory phase.
};

// ---------------------------------------------------------------------------
// Interface of the model to fit (the concrete model is injected).
// ---------------------------------------------------------------------------

/// @brief Abstract model interface consumed by SAEM.
class SaemModel {
public:
    virtual ~SaemModel() = default;

    /// @brief Dimension of theta.
    virtual int dimTheta() const = 0;
    /// @brief Dimension of the sampled variable chi_i.
    virtual int dimChi()   const = 0;
    /// @brief Number of subjects.
    virtual int nSubjects() const = 0;

    /// @brief Starting point of theta.
    virtual Vec initialTheta() const = 0;
    /// @brief Starting value of chi_i.
    /// @param i Subject index.
    virtual Vec initialChi(int i) const = 0;

    /// @brief Log joint density log p(y_i, chi_i; theta).
    /// @param i     Subject index.
    /// @param chi   Individual effect.
    /// @param theta Parameter vector.
    virtual double logJoint(int i, const Vec& chi, const Vec& theta) const = 0;

    /// @brief Gradient of log p(y_i, chi_i; theta) with respect to theta,
    ///        averaged over subjects.
    /// @param theta Parameter vector.
    /// @param phis  Individual effects, one per subject.
    virtual Vec mStepGradient(const Vec& theta, const std::vector<Vec>& phis) const = 0;

    /// @brief Independence proposal N(mean, cov) for the MCMC of chi_i.
    /// @param theta Parameter vector.
    /// @param mean  Output mean vector.
    /// @param cov   Output covariance matrix.
    virtual void independenceProposal(const Vec& theta, Vec& mean, Mat& cov) const = 0;

    /// @brief Indices of theta that are log standard deviations of the random
    ///        effects (used by the simulated annealing floor).
    virtual std::vector<int> varianceIndices() const { return {}; }

    // --- Closed-form M-step with sufficient statistics (classic SAEM) -------

    /// @brief Whether the model implements sufficient statistics.
    virtual bool hasSufficientStats() const { return false; }

    /// @brief S(chi) averaged over subjects.
    ///
    /// @param phis  Individual effects, one per subject.
    /// @param theta Current value (e.g. to impute censored observations with the
    ///              current error parameters).
    /// @return The averaged sufficient-statistic vector.
    /// @throws std::logic_error By default, because the method is not implemented.
    virtual Vec  sufficientStats(const std::vector<Vec>& /*phis*/, const Vec& /*theta*/) const {
        throw std::logic_error("sufficientStats not implemented");
    }

    /// @brief Recomputes theta from sufficient statistics.
    /// @param s     Sufficient-statistic vector.
    /// @param theta Current parameter vector.
    /// @return The updated parameter vector.
    /// @throws std::logic_error By default, because the method is not implemented.
    virtual Vec  thetaFromStats(const Vec& /*s*/, const Vec& /*theta*/) const {
        throw std::logic_error("thetaFromStats not implemented");
    }

    /// @brief Numerical step for parameters without a closed form (fixed effects
    ///        without random effects, custom likelihood parameters, ...):
    ///        theta_nb <- theta_nb + gamma (argmax Q(.; chi_k) - theta_nb).
    ///
    /// @param theta Parameter vector, updated in place.
    /// @param phis  Chains per subject: phis[chain][subject].
    /// @param gamma Step size.
    /// @note The default implementation does nothing.
    virtual void numericMStep(Vec& /*theta*/, const std::vector<std::vector<Vec>>& /*phis*/,
                              double /*gamma*/) const {}

    /// @brief Robust generalized M-step (MStep::Gradient in models that implement it).
    ///
    /// At each iteration it maximizes Q(theta; chi_k) with damped Newton
    /// (finite derivatives, backtracking, bounded step) and then applies
    ///   theta <- theta + gamma (theta* - theta).
    /// It replaces Robbins-Monro on the gradient (which is sensitive to the
    /// scale of each coordinate).
    virtual bool hasGemStep() const { return false; }

    /// @brief Performs one generalized M-step.
    /// @param theta Parameter vector, updated in place.
    /// @param phis  Chains per subject: phis[chain][subject].
    /// @param gamma Step size.
    /// @note The default implementation does nothing.
    virtual void gemStep(Vec& /*theta*/, const std::vector<std::vector<Vec>>& /*phis*/, double /*gamma*/) const {}
};

// ---------------------------------------------------------------------------
// ESS (sum of autocorrelations truncated at the first non-positive one),
// minimum across coordinates of a trace.
// ---------------------------------------------------------------------------

/// @brief Minimum effective sample size across coordinates of a trace.
///
/// Computes the integrated autocorrelation time per coordinate and returns the
/// smallest T / tau over coordinates.
///
/// @param tr Trace of vectors (one vector per iteration).
/// @return The minimum effective sample size.
inline double essMin(const std::vector<Vec>& tr) {
    const int T = static_cast<int>(tr.size());
    if (T < 4) return double(T);
    const int d = static_cast<int>(tr[0].size());
    double best = std::numeric_limits<double>::infinity();
    for (int j = 0; j < d; ++j) {
        double mu = 0.0; for (const auto& x : tr) mu += x[j]; mu /= T;
        double c0 = 0.0; for (const auto& x : tr) c0 += (x[j] - mu) * (x[j] - mu); c0 /= T;
        if (c0 <= 0.0) { best = std::min(best, 1.0); continue; }
        double tau = 1.0;
        for (int lag = 1; lag < T / 2; ++lag) {
            double c = 0.0;
            for (int t = lag; t < T; ++t) c += (tr[uz(t)][j] - mu) * (tr[uz(t - lag)][j] - mu);
            c /= T;
            const double rho = c / c0;
            if (rho <= 0.0) break;
            tau += 2.0 * rho;
        }
        best = std::min(best, double(T) / tau);
    }
    return best;
}

/// @brief Split Gelman-Rubin R-hat for several sequences of one variable.
///
/// Each chain is split in half before computing the potential scale reduction
/// factor.
///
/// @param chains Sequences to compare.
/// @return The split R-hat, or NaN when a sequence is too short or the
///         within-chain variance is not positive.
inline double splitRhat(const std::vector<std::vector<double>>& chains) {
    std::vector<std::vector<double>> seqs;
    for (const auto& c : chains) {
        const size_t h = c.size() / 2;
        if (h < 2) return std::numeric_limits<double>::quiet_NaN();
        seqs.emplace_back(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(h));
        seqs.emplace_back(c.begin() + static_cast<std::ptrdiff_t>(h), c.begin() + static_cast<std::ptrdiff_t>(2 * h));
    }
    const double m = double(seqs.size()), n = double(seqs[0].size());
    std::vector<double> means(seqs.size()), vars(seqs.size());
    for (size_t k = 0; k < seqs.size(); ++k) {
        double mu = 0.0; for (double v : seqs[k]) mu += v; mu /= n; means[k] = mu;
        double s2 = 0.0; for (double v : seqs[k]) s2 += (v - mu) * (v - mu); vars[k] = s2 / (n - 1.0);
    }
    double gm = 0.0; for (double v : means) gm += v; gm /= m;
    double B = 0.0; for (double v : means) B += (v - gm) * (v - gm); B *= n / (m - 1.0);
    double W = 0.0; for (double v : vars) W += v; W /= m;
    if (!(W > 0.0)) return 1.0;
    const double varplus = (n - 1.0) / n * W + B / n;
    return std::sqrt(varplus / W);
}

// ---------------------------------------------------------------------------
// SAEM convergence diagnostics.
// ---------------------------------------------------------------------------

/// @brief Convergence summary of a SAEM run.
struct ConvergenceInfo {
    int    iterations   = 0;       ///< Number of iterations executed.
    bool   stoppedEarly = false;   ///< Whether automatic stopping was triggered.
    Vec    drift;                  ///< Per-coordinate |theta_end - theta_75%| / (|theta_end| + 0.05).
    double maxDrift     = 0.0;     ///< Maximum of `drift` (small = stable in the last 25%).
};

// ---------------------------------------------------------------------------
// SAEM.
// ---------------------------------------------------------------------------

/// @brief Stochastic Approximation EM driver.
///
/// The driver owns the chains, samplers and parameter trace, and delegates the
/// model-specific work to a SaemModel.
///
/// @note Sampler lambdas capture `this`, so the object cannot be copied or moved.
class SAEM {
public:
    /// @brief Constructs the driver and initializes chains and samplers.
    /// @param model Model to fit.
    /// @param ctrl  Runtime controls.
    /// @param seed  Base random seed.
    /// @throws std::invalid_argument If n_chains < 1, initialTheta() has the
    ///         wrong dimension, or sufficient statistics are requested but not
    ///         implemented by the model.
    SAEM(const SaemModel& model, SAEMControl ctrl = {}, unsigned seed = 12345)
      : model_(model), ctrl_(std::move(ctrl)), seed_(seed), rm_(effectiveRM(ctrl_))
    {
        if (ctrl_.n_chains < 1) throw std::invalid_argument("n_chains >= 1");
        theta_ = model_.initialTheta();
        if (theta_.size() != model_.dimTheta())
            throw std::invalid_argument("initialTheta() has incorrect dimension");

        rm_.init(theta_);
        ctrl_.step.sa_len = ctrl_.simulated_annealing ? saPhaseLength(ctrl_.sa, ctrl_.n_saem_iter) : 0;
        varIdx_ = model_.varianceIndices();
        if (ctrl_.mstep == MStep::SufficientStats && !model_.hasSufficientStats())
            throw std::invalid_argument("the model does not implement sufficient statistics");
        useStats_ = (ctrl_.mstep == MStep::SufficientStats) ||
                    (ctrl_.mstep == MStep::Auto && model_.hasSufficientStats());
        useGem_ = !useStats_ && ctrl_.mstep == MStep::Gradient && model_.hasGemStep();

        const int n = model_.nSubjects();
        const int M = ctrl_.n_chains;
        phis_.assign(uz(M), std::vector<Vec>(uz(n)));
        samplers_.resize(uz(M));
        MCMCControl mcc = ctrl_.mcmc;
        if (ctrl_.adapt_rw) mcc.adapt_online = true;
        mcc.w_map = ctrl_.map_kernel ? ctrl_.map_weight : 0.0;
        for (int m = 0; m < M; ++m) {
            samplers_[uz(m)].reserve(uz(n));
            for (int i = 0; i < n; ++i) {
                phis_[uz(m)][uz(i)] = model_.initialChi(i);
                auto logp = [this, i](const Vec& phi) {
                    return model_.logJoint(i, phi, theta_);
                };
                samplers_[uz(m)].push_back(std::make_unique<MCMCKuhnLavielle>(
                    logp, model_.dimChi(), seed_ + static_cast<unsigned>(i) + 1000003u * static_cast<unsigned>(m), mcc));
            }
        }
    }

    // The sampler lambdas capture `this`: copying or moving would leave
    // dangling pointers.
    SAEM(const SAEM&) = delete;
    SAEM& operator=(const SAEM&) = delete;
    SAEM(SAEM&&) = delete;
    SAEM& operator=(SAEM&&) = delete;

    /// @brief Whether the run uses closed-form sufficient statistics.
    bool usesSufficientStats() const { return useStats_; }

    /// @brief Runs the full SAEM loop.
    ///
    /// Each iteration performs the S/E-step, then either the closed-form,
    /// generalized or Robbins-Monro M-step, appends the current theta to the
    /// trace, optionally prints progress, and optionally checks the automatic
    /// stopping criterion.
    ///
    /// @throws std::runtime_error If the M-step produces a non-finite theta.
    void run() {
        const int n = model_.nSubjects();
        const int M = ctrl_.n_chains;
        const int K = ctrl_.n_saem_iter;
        thetaTrace_.clear(); thetaTrace_.reserve(uz(K));
        stoppedEarly_ = false;
        int stable = 0;
        for (int k = 0; k < K; ++k) {
            updateProposals(k);

            // S/E-step: each (chain, subject) has its own sampler and RNG.
            const int sweeps = (k == 0) ? (ctrl_.n_mcmc_burnin + ctrl_.n_mcmc_sweeps)
                                        :  ctrl_.n_mcmc_sweeps;
            parallelFor(M * n, ctrl_.n_threads, [&](int idx) {
                const int m = idx / n, i = idx % n;
                phis_[uz(m)][uz(i)] = samplers_[uz(m)][uz(i)]->sampleFrom(phis_[uz(m)][uz(i)], sweeps);
            });

            const Vec theta_old = theta_;
            if (useStats_) {
                // CLOSED-FORM M-step (classic SAEM).
                Vec S = model_.sufficientStats(phis_[0], theta_);
                for (int m = 1; m < M; ++m) S += model_.sufficientStats(phis_[uz(m)], theta_);
                if (M > 1) S /= double(M);
                const double gam = ctrl_.step.gamma(k, K);
                if (k == 0 || s_.size() != S.size()) s_ = S;
                else s_ += gam * (S - s_);
                Vec th = model_.thetaFromStats(s_, theta_);
                model_.numericMStep(th, phis_, gam);
                if (!th.allFinite()) throw std::runtime_error("closed-form M-step: theta is not finite");
                if (ctrl_.simulated_annealing) saApplyVarianceFloor(th, theta_old, varIdx_, k, ctrl_.sa);
                theta_ = th;
            } else if (useGem_) {
                Vec th = theta_;
                model_.gemStep(th, phis_, ctrl_.step.gamma(k, K));
                if (!th.allFinite()) throw std::runtime_error("generalized M-step: theta is not finite");
                if (ctrl_.simulated_annealing) saApplyVarianceFloor(th, theta_old, varIdx_, k, ctrl_.sa);
                theta_ = th;
            } else {
                Vec grad = model_.mStepGradient(theta_, phis_[0]);
                for (int m = 1; m < M; ++m) grad += model_.mStepGradient(theta_, phis_[uz(m)]);
                if (M > 1) grad /= double(M);
                theta_ = rm_.step(grad);
                if (ctrl_.simulated_annealing) {
                    Vec th = theta_;
                    if (saApplyVarianceFloor(th, theta_old, varIdx_, k, ctrl_.sa)) {
                        rm_.setTheta(th);
                        theta_ = th;
                    }
                }
            }
            thetaTrace_.push_back(theta_);

            if (ctrl_.verbose && (k % ctrl_.print_every == 0 || k == K - 1)) {
                std::cout << "  it " << std::setw(4) << k
                          << "  gamma " << std::scientific << std::setprecision(2)
                          << ((useStats_ || useGem_) ? ctrl_.step.gamma(k, K) : rm_.currentStep())
                          << "  theta " << std::fixed << std::setprecision(4)
                          << theta_.transpose();
                if (!useStats_ && !useGem_ && ctrl_.rm.averaging)
                    std::cout << "   | avg " << rm_.estimate().transpose();
                std::cout << "\n";
            }

            // Automatic stopping: small relative drift in a window, twice in a row.
            if (ctrl_.auto_stop) {
                const int k1 = useStats_ ? ctrl_.step.K1(K) : K / 2;
                const int w = ctrl_.stop_window;
                if (k >= k1 + ctrl_.stop_min_phase2 && k >= w && (k % w) == 0) {
                    const Vec& a = thetaTrace_[uz(k)]; const Vec& b = thetaTrace_[uz(k - w)];
                    const double dr = ((a - b).cwiseAbs().array() / (a.cwiseAbs().array() + 0.05)).maxCoeff();
                    stable = (dr < ctrl_.stop_tol) ? stable + 1 : 0;
                    if (stable >= 2) { stoppedEarly_ = true; break; }
                }
            }
        }
    }

    /// @brief Last theta iterate.
    const Vec& theta() const { return theta_; }

    /// @brief Recommended estimate: Polyak average (gradient M-step, when active)
    ///        or the last iterate (classic SAEM: statistics are already averaged).
    Vec  thetaAveraged() const { return (useStats_ || useGem_) ? theta_ : rm_.estimate(); }

    /// @brief Individual effects of chain 0.
    const std::vector<Vec>& phis() const { return phis_[0]; }

    /// @brief Individual effects of all chains: [chain][subject].
    const std::vector<std::vector<Vec>>& allPhis() const { return phis_; }

    /// @brief Robbins-Monro state.
    const RobbinsMonro& rm() const { return rm_; }

    /// @brief Trace of theta over iterations.
    const std::vector<Vec>& thetaTrace() const { return thetaTrace_; }

    /// @brief Computes convergence diagnostics from the theta trace.
    /// @return Drift information for the final iterates.
    ConvergenceInfo convergence() const {
        ConvergenceInfo ci;
        ci.iterations = static_cast<int>(thetaTrace_.size());
        ci.stoppedEarly = stoppedEarly_;
        if (ci.iterations >= 8) {
            const Vec& a = thetaTrace_.back();
            const Vec& b = thetaTrace_[static_cast<size_t>(0.75 * (ci.iterations - 1))];
            ci.drift = ((a - b).cwiseAbs().array() / (a.cwiseAbs().array() + 0.05)).matrix();
            ci.maxDrift = ci.drift.maxCoeff();
        }
        return ci;
    }

    // -----------------------------------------------------------------------
    // Conditional distribution p(chi_i | y_i, theta_hat): MEAN and SD per
    // subject, with theta FIXED at theta_hat (all chains). After the call,
    // theta() == theta_hat. The minimum ESS and the maximum (split, across
    // chains) R-hat are also returned as diagnostics of the conditional
    // sampling.
    // -----------------------------------------------------------------------

    /// @brief Conditional estimates for all subjects.
    struct CondEstimate {
        std::vector<Vec> mean, sd;                ///< Per-subject conditional mean and SD.
        std::vector<std::vector<Vec>> draws;      ///< [subject][sample] (only if keep_draws > 0).
        double ess_min = 0.0;                     ///< Minimum effective sample size.
        double rhat_max = std::numeric_limits<double>::quiet_NaN(); ///< Maximum split R-hat.
    };

    /// @brief Samples the conditional distribution p(chi_i | y_i, theta_hat).
    ///
    /// @param theta_hat Fixed parameter vector.
    /// @param n_samples Number of retained samples per chain.
    /// @param n_burn    Number of burn-in transitions.
    /// @param thin      Thinning interval between retained samples.
    /// @param keep_draws Number of draws to keep per subject (0 = none).
    /// @return Conditional means, standard deviations and diagnostics.
    /// @throws std::invalid_argument If n_samples < 4, thin < 1 or n_burn < 0.
    CondEstimate conditionalSample(const Vec& theta_hat, int n_samples = 1000,
                                   int n_burn = 200, int thin = 2, int keep_draws = 0) {
        if (n_samples < 4 || thin < 1 || n_burn < 0) throw std::invalid_argument("conditionalSample: invalid arguments");
        theta_ = theta_hat;
        updateProposals(ctrl_.n_saem_iter + 1);
        refreshMap();
        const int n = model_.nSubjects(), M = ctrl_.n_chains, d = model_.dimChi();
        CondEstimate out; out.mean.assign(uz(n), Vec::Zero(d)); out.sd.assign(uz(n), Vec::Zero(d));
        if (keep_draws > 0) out.draws.assign(uz(n), std::vector<Vec>());
        std::vector<double> essI(uz(n), 0.0), rhatI(uz(n), std::numeric_limits<double>::quiet_NaN());
        parallelFor(n, ctrl_.n_threads, [&](int i) {
            std::vector<std::vector<Vec>> tr(uz(M));
            for (int m = 0; m < M; ++m) {
                auto& sp = *samplers_[uz(m)][uz(i)];
                phis_[uz(m)][uz(i)] = sp.sampleFrom(phis_[uz(m)][uz(i)], n_burn);
                MCMCControl c = sp.control(); c.adapt_online = false; sp.setControl(c);   // Freeze adaptation.
                tr[uz(m)].reserve(uz(n_samples));
                for (int s = 0; s < n_samples; ++s) {
                    phis_[uz(m)][uz(i)] = sp.sampleFrom(phis_[uz(m)][uz(i)], thin);
                    tr[uz(m)].push_back(phis_[uz(m)][uz(i)]);
                }
            }
            Vec mean = Vec::Zero(d), m2 = Vec::Zero(d);
            long cnt = 0;
            for (int m = 0; m < M; ++m) for (const Vec& x : tr[uz(m)]) {
                ++cnt;
                const Vec dl = x - mean;
                mean += dl / double(cnt);
                m2 += dl.cwiseProduct(x - mean);
            }
            out.mean[uz(i)] = mean;
            out.sd[uz(i)] = (m2 / double(std::max<long>(1, cnt - 1))).cwiseSqrt();
            if (keep_draws > 0) {                       // Uniform subsampling of the traces (all chains).
                std::vector<const Vec*> all;
                for (int m = 0; m < M; ++m) for (const Vec& x : tr[uz(m)]) all.push_back(&x);
                const long tot = static_cast<long>(all.size());
                const int kd = static_cast<int>(std::min<long>(keep_draws, tot));
                for (int j = 0; j < kd; ++j) out.draws[uz(i)].push_back(*all[static_cast<size_t>(double(j) * double(tot) / double(kd))]);
            }
            double ess = 0.0;
            for (int m = 0; m < M; ++m) ess += essMin(tr[uz(m)]);
            essI[uz(i)] = ess;
            double rh = 0.0; bool ok = true;
            for (int j = 0; j < d; ++j) {
                std::vector<std::vector<double>> seqs(uz(M));
                for (int m = 0; m < M; ++m) { seqs[uz(m)].reserve(uz(n_samples)); for (const Vec& x : tr[uz(m)]) seqs[uz(m)].push_back(x[j]); }
                const double r = splitRhat(seqs);
                if (std::isnan(r)) { ok = false; break; }
                rh = std::max(rh, r);
            }
            if (ok) rhatI[uz(i)] = rh;
        });
        out.ess_min = *std::min_element(essI.begin(), essI.end());
        double rmax = 0.0; bool any = false;
        for (double r : rhatI) if (!std::isnan(r)) { rmax = std::max(rmax, r); any = true; }
        if (any) out.rhat_max = rmax;
        return out;
    }

private:
    const SaemModel& model_;
    SAEMControl ctrl_;
    unsigned seed_;

    Vec theta_;
    RobbinsMonro rm_;
    bool useStats_ = false;
    bool useGem_ = false;
    bool stoppedEarly_ = false;
    Vec  s_;                                  // Approximate sufficient statistics (classic SAEM).
    Vec  propSd_;                             // Population proposal SD (MAP simplex scale).
    std::vector<int> varIdx_;
    std::vector<std::vector<Vec>> phis_;                              // [chain][subject].
    std::vector<std::vector<std::unique_ptr<MCMCKuhnLavielle>>> samplers_;
    std::vector<Vec> thetaTrace_;

    // Effective RM: with SA active, the Polyak average only starts AFTER the
    // exploratory phase.
    static RMControl effectiveRM(const SAEMControl& c) {
        RMControl r = c.rm;
        if (c.simulated_annealing && r.averaging)
            r.avg_start = std::max(r.avg_start, saPhaseLength(c.sa, c.n_saem_iter) + 1);
        return r;
    }

    // Proposal N(mean, Omega) and its Cholesky factor: the same for all subjects.
    void updateProposals(int k) {
        Vec mean; Mat cov;
        model_.independenceProposal(theta_, mean, cov);
        const Mat cov_raw = cov;
        if (ctrl_.simulated_annealing && ctrl_.sa.inflate_proposal)
            cov = inflateOmega(cov, k, ctrl_.sa);
        Eigen::LLT<Mat> llt(cov);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("independenceProposal: cov is not positive definite");
        const Mat L = llt.matrixL();

        const int d = model_.dimChi();
        propSd_ = cov_raw.diagonal().cwiseMax(1e-12).cwiseSqrt();
        Vec rw, gb;
        if (k == 0 && ctrl_.adapt_rw) {
            rw = (2.4 / std::sqrt(double(d))) * propSd_;
            gb = 2.4 * propSd_;
        }
        for (auto& chain : samplers_)
            for (auto& sp : chain) {
                sp->setIndependenceChol(mean, cov, L);
                if (rw.size() == d) sp->setSteps(rw, gb);
            }
        if (ctrl_.map_kernel && ctrl_.map_every > 0 && (k == 0 || k % ctrl_.map_every == 0)) refreshMap();
    }

    // Per-subject Laplace/MAP proposal: conditional mode (Nelder-Mead starting
    // from the current state) + robust covariance (-H)^{-1}. It depends on theta
    // and the data, not on the state of the chain, so it is a valid independence
    // kernel.
    void refreshMap() {
        if (!ctrl_.map_kernel) return;
        const int n = model_.nSubjects(), M = ctrl_.n_chains;
        parallelFor(n, ctrl_.n_threads, [&](int i) {
            try {
                auto lj = [&](const Vec& x) { return model_.logJoint(i, x, theta_); };
                auto neg = [&](const Vec& x) {
                    const double v = lj(x);
                    return std::isfinite(v) ? -v : std::numeric_limits<double>::infinity();
                };
                const NMOutcome o = nelderMeadEx(neg, phis_[0][uz(i)], 0.3 * propSd_, 300, 1e-9, 0);
                const Mat H = numericalHessian(lj, o.x, 1e-4);
                const Mat Sg = sigmaFromNegHessian(H);
                Mat L; double ld = 0.0;
                if (Sg.size() == 0 || !cholWithJitter(Sg, L, ld)) return;
                for (int m = 0; m < M; ++m) samplers_[uz(m)][uz(i)]->setMapProposal(o.x, L);
            } catch (const std::exception&) { /* No MAP proposal for this subject. */ }
        });
    }
};

} // namespace mcmckl
