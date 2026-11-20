// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file expdecay.hpp
 * @brief Nonlinear mixed-effects exponential-decay model used by the demos.
 *
 * The model implements @ref SaemModel:
 * @f[
 *   \phi_i = (\phi_{1i}, \phi_{2i}) \sim N(\mu, \operatorname{diag}(w^2)),
 * @f]
 * @f[
 *   y_{ij} = \phi_{1i}\exp(-\phi_{2i}t_{ij}) + \varepsilon_{ij},
 *   \qquad \varepsilon_{ij} \sim N(0, \sigma^2),
 * @f]
 * with parameter vector
 * @f[
 *   \theta = (\mu_1, \mu_2, \log w_1, \log w_2, \log\sigma).
 * @f]
 *
 * Requires: Eigen3, C++20, engine.hpp.
 */

#pragma once

#include "engine.hpp"

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace mcmckl {

/**
 * @brief Exponential-decay nonlinear mixed-effects model.
 *
 * The class provides closed-form and gradient-based M-step implementations so
 * it can be used to exercise every SAEM mode supported by @ref SaemModel.
 */
class ExpDecayModel : public SaemModel {
public:
    /** @brief Observations: one time vector and one response vector per subject. */
    struct Data {
        std::vector<std::vector<double>> t, y;
    };

    /**
     * @brief Constructs the model.
     * @param data Observations.
     * @param mu_true True population mean (used for reference/diagnostics).
     * @param w_true True random-effect standard deviations.
     * @param sigma_true True residual standard deviation.
     */
    ExpDecayModel(Data data, Vec mu_true, Vec w_true, double sigma_true)
      : d_(std::move(data)), n_(static_cast<int>(d_.t.size())),
        mu_true_(std::move(mu_true)), w_true_(std::move(w_true)), sigma_(sigma_true) {}

    /** @brief Returns the number of fixed/global parameters (5). */
    int dimTheta() const override { return 5; }
    /** @brief Returns the number of individual random effects (2). */
    int dimChi()   const override { return 2; }
    /** @brief Returns the number of subjects. */
    int nSubjects() const override { return n_; }

    /** @brief Returns the default starting values. */
    Vec initialTheta() const override {
        Vec th(5); th << 5.0, 1.0, 0.0, 0.0, 0.0; return th;
    }
    /** @brief Returns the default starting values for an individual effect. */
    Vec initialChi(int) const override { return initialTheta().head(2); }

    /**
     * @brief Evaluates the complete-data log density for one subject.
     * @param i Subject index.
     * @param phi Individual effects.
     * @param th Global parameter vector.
     * @return @f$\log p(y_i, \phi_i; \theta)@f$.
     */
    double logJoint(int i, const Vec& phi, const Vec& th) const override {
        const double mu1 = th[0], mu2 = th[1];
        const double w1 = std::exp(th[2]), w2 = std::exp(th[3]), s = std::exp(th[4]);
        double lp = -0.5 * std::pow((phi[0] - mu1) / w1, 2) - std::log(w1)
                  - 0.5 * std::pow((phi[1] - mu2) / w2, 2) - std::log(w2);
        for (size_t j = 0; j < d_.t[i].size(); ++j) {
            const double f = phi[0] * std::exp(-phi[1] * d_.t[i][j]);
            const double r = (d_.y[i][j] - f) / s;
            lp += -0.5 * r * r - std::log(s);
        }
        return lp;
    }

    /**
     * @brief Evaluates the Monte Carlo M-step gradient.
     * @param th Global parameter vector.
     * @param phis Individual effects sampled during the S-step.
     * @return Gradient of the averaged complete-data log density.
     */
    Vec mStepGradient(const Vec& th, const std::vector<Vec>& phis) const override {
        const double mu1 = th[0], mu2 = th[1];
        const double w1 = std::exp(th[2]), w2 = std::exp(th[3]), s = std::exp(th[4]);
        Vec g = Vec::Zero(5);
        for (int i = 0; i < n_; ++i) {
            const Vec& phi = phis[i];
            g[0] += (phi[0] - mu1) / (w1 * w1);
            g[1] += (phi[1] - mu2) / (w2 * w2);
            g[2] += -1.0 + std::pow((phi[0] - mu1) / w1, 2);
            g[3] += -1.0 + std::pow((phi[1] - mu2) / w2, 2);
            double ss = 0.0;
            for (size_t j = 0; j < d_.t[i].size(); ++j) {
                const double f = phi[0] * std::exp(-phi[1] * d_.t[i][j]);
                ss += std::pow((d_.y[i][j] - f) / s, 2);
            }
            g[4] += -double(d_.t[i].size()) + ss;
        }
        return g / double(n_);
    }

    /**
     * @brief Gaussian independence proposal @f$N(\mu, \operatorname{diag}(w^2))@f$.
     * @param th Global parameter vector.
     * @param mean Output proposal mean.
     * @param cov Output proposal covariance.
     */
    void independenceProposal(const Vec& th, Vec& mean, Mat& cov) const override {
        mean = th.head(2);
        cov  = Mat::Zero(2, 2);
        cov.diagonal() << std::exp(2.0 * th[2]), std::exp(2.0 * th[3]);
    }

    // --- Reference values and utilities ------------------------------------
    /** @brief Returns the true population mean. */
    Vec    muTrue()    const { return mu_true_; }
    /** @brief Returns the true random-effect standard deviations. */
    Vec    wTrue()     const { return w_true_; }
    /** @brief Returns the true residual standard deviation. */
    double sigmaTrue() const { return sigma_; }

    /**
     * @brief Closed-form maximum-likelihood M-step.
     *
     * Computes the MLE of @f$\mu@f$, the random-effect variances, and the
     * residual variance from the sampled individual effects. The M-step of
     * SAEM maximizes the likelihood, so the divisors are the number of subjects
     * and the number of observations (not @f$n-1@f$).
     *
     * @param phis Individual effects sampled during the S-step.
     * @return Updated global parameter vector.
     */
    Vec closedFormMStep(const std::vector<Vec>& phis) const {
        Vec mean = Vec::Zero(2);
        for (const auto& p : phis) mean += p;
        mean /= double(n_);
        Vec var = Vec::Zero(2);
        double ss = 0.0;
        long nobs = 0;
        for (int i = 0; i < n_; ++i) {
            var += (phis[i] - mean).cwiseAbs2();
            for (size_t j = 0; j < d_.t[i].size(); ++j) {
                const double f = phis[i][0] * std::exp(-phis[i][1] * d_.t[i][j]);
                ss += std::pow(d_.y[i][j] - f, 2);
                ++nobs;
            }
        }
        // The SAEM M-step maximizes the likelihood: divide by the number of
        // subjects for the variances and by the number of observations for the
        // residual variance (previously ss / n_subjects inflated sigma by about
        // sqrt(n_times), and var / (n-1) was not the MLE).
        var /= double(n_);
        ss  /= double(nobs);
        Vec th(5);
        th << mean[0], mean[1],
              std::log(std::sqrt(var[0])), std::log(std::sqrt(var[1])),
              std::log(std::sqrt(ss));
        return th;
    }

private:
    Data d_;
    int n_;
    Vec mu_true_, w_true_;
    double sigma_;
};

/**
 * @brief Generates synthetic data from the exponential-decay model.
 *
 * @param n_sub Number of subjects.
 * @param n_times Number of observations per subject.
 * @param dt Spacing between observation times.
 * @param sigma Residual standard deviation.
 * @param mu Population mean used for simulation.
 * @param w Random-effect standard deviations used for simulation.
 * @param seed Random seed.
 * @param add_outliers When true, injects symmetric 5-sigma outliers with
 *        probability 5% per observation.
 * @return Simulated observations.
 */
inline ExpDecayModel::Data makeData(int n_sub, int n_times, double dt, double sigma,
                                    const Vec& mu, const Vec& w, unsigned seed,
                                    bool add_outliers = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nz(0.0, 1.0);
    std::uniform_real_distribution<double> ut(0.0, 1.0);

    ExpDecayModel::Data d;
    d.t.resize(n_sub); d.y.resize(n_sub);
    for (int i = 0; i < n_sub; ++i) {
        const double p1 = mu[0] + w[0] * nz(rng);
        const double p2 = std::max(0.05, mu[1] + w[1] * nz(rng));
        for (int j = 0; j < n_times; ++j) {
            const double t = dt * (j + 1);
            const double f = p1 * std::exp(-p2 * t);
            double y = f + sigma * nz(rng);
            if (add_outliers && ut(rng) < 0.05) y += 5.0 * sigma * (ut(rng) < 0.5 ? 1 : -1);
            d.t[i].push_back(t);
            d.y[i].push_back(y);
        }
    }
    return d;
}

} // namespace mcmckl
