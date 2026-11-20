// ============================================================================
// @file likelihoods.hpp
// @brief Ready-made likelihoods for NON-continuous data.
//
// Each factory returns a `lik::Lik` containing:
//   .logLik : log p(y | psi, x, cov, extra)       (-> Model.logLikObs)
//   .sample : y ~ p(. | psi, x, cov, extra)       (-> Model.simulateObs, used by smx::simulate)
//   .nExtra : number of extra parameters           (-> Model.nExtra, values in Control.extra0)
//
// The user supplies the "structural" part as a function of (psi, x, cov):
//
//   bernoulli / bernoulliLogit : binary response (0/1)
//   poisson                    : counts
//   negbin                     : overdispersed counts; extra[0] = log(size r)
//   zip                        : zero-inflated Poisson; returns (lambda, pi)
//   ordinal                    : ordered category under a proportional-odds model
//   categorical                : nominal categories with given probabilities
//   tte                        : time to event: delta*log h(t) - [H(t) - H(t_ini)]
//                                (repeated events / interval censoring with
//                                 `tstart` as a predictor column)
//
// Usage:
//   auto L = lik::poisson([](const Vec& psi, auto x, const Vec&) {
//       return std::exp(psi[0] + psi[1] * x[0]);
//   });
//   model.logLikObs = L.logLik; model.simulateObs = L.sample;
//
// Requires: Eigen3, C++20, obslik.hpp.
// ============================================================================

#pragma once
#include "numcast.hpp"

#include "obslik.hpp"

#include <cmath>
#include <functional>
#include <random>
#include <span>

namespace mcmckl {
namespace lik {

/** @brief Read-only view of one observation row. */
using Xs = std::span<const double>;

/**
 * @brief Draws one observation from a likelihood.
 * @param psi Individual parameters.
 * @param x Predictor row, with x[0] as the time column where applicable.
 * @param cov Subject covariates.
 * @param extra Extra parameters (see Lik::nExtra).
 * @param rng Random number generator to use.
 * @return The simulated observation.
 */
using SampleObs = std::function<double(const Vec& psi, Xs x, const Vec& cov, const Vec& extra, std::mt19937& rng)>;

/**
 * @brief Bundle returned by every likelihood factory.
 */
struct Lik {
    LogLikObs logLik;   ///< log p(y | psi, x, cov, extra) per observation.
    SampleObs sample;   ///< Simulator y ~ p(. | psi, x, cov, extra).
    int nExtra = 0;     ///< Number of extra parameters consumed from `extra`.
};

namespace detail {
/** @brief Clamps a probability to the open interval (0, 1). */
inline double clampP(double p) { return std::min(std::max(p, 1e-12), 1.0 - 1e-12); }
/** @brief Numerically stable logistic function. */
inline double logistic(double z) { return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z)); }
}

/**
 * @brief Bernoulli likelihood.
 * @param P Callable returning P(y = 1) for a given (psi, x, cov).
 * @return A Lik whose log-likelihood is log P(y = 1) or log(1 - P(y = 1)).
 * @note Probabilities outside [0, 1] or non-finite values yield -inf.
 */
inline Lik bernoulli(std::function<double(const Vec&, Xs, const Vec&)> P) {
    Lik L;
    L.logLik = [P](const Vec& psi, Xs x, const Vec& cov, double y, const Vec&) {
        const double p = P(psi, x, cov);
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) return -std::numeric_limits<double>::infinity();
        return y >= 0.5 ? std::log(detail::clampP(p)) : std::log(detail::clampP(1.0 - p));
    };
    L.sample = [P](const Vec& psi, Xs x, const Vec& cov, const Vec&, std::mt19937& rng) {
        std::bernoulli_distribution b(detail::clampP(P(psi, x, cov))); return b(rng) ? 1.0 : 0.0;
    };
    return L;
}
/**
 * @brief Bernoulli likelihood parametrized by the logit.
 * @param eta Callable returning logit P(y = 1) = eta.
 * @return A Bernoulli Lik using P(y = 1) = logistic(eta).
 */
inline Lik bernoulliLogit(std::function<double(const Vec&, Xs, const Vec&)> eta) {
    return bernoulli([eta](const Vec& psi, Xs x, const Vec& cov) { return detail::logistic(eta(psi, x, cov)); });
}

/**
 * @brief Poisson likelihood.
 * @param lambda Callable returning the Poisson mean.
 * @return A Lik with log p(y) = y log(lambda) - lambda - log Gamma(y + 1).
 * @note Non-positive or non-finite means, or negative counts, yield -inf.
 */
inline Lik poisson(std::function<double(const Vec&, Xs, const Vec&)> lambda) {
    Lik L;
    L.logLik = [lambda](const Vec& psi, Xs x, const Vec& cov, double y, const Vec&) {
        const double lam = lambda(psi, x, cov);
        if (!(lam > 0.0) || !std::isfinite(lam) || y < 0.0) return -std::numeric_limits<double>::infinity();
        return y * std::log(lam) - lam - std::lgamma(y + 1.0);
    };
    L.sample = [lambda](const Vec& psi, Xs x, const Vec& cov, const Vec&, std::mt19937& rng) {
        std::poisson_distribution<int> po(std::max(lambda(psi, x, cov), 1e-12)); return double(po(rng));
    };
    return L;
}

/**
 * @brief Negative-binomial likelihood for overdispersed counts.
 * @param mean Callable returning the mean mu.
 * @return A Lik with size r = exp(extra[0]) and Lik::nExtra == 1.
 * @note Non-positive or non-finite means, or negative counts, yield -inf.
 */
inline Lik negbin(std::function<double(const Vec&, Xs, const Vec&)> mean) {
    Lik L; L.nExtra = 1;
    L.logLik = [mean](const Vec& psi, Xs x, const Vec& cov, double y, const Vec& ex) {
        const double mu = mean(psi, x, cov), r = std::exp(ex[0]);
        if (!(mu > 0.0) || !std::isfinite(mu) || y < 0.0) return -std::numeric_limits<double>::infinity();
        return std::lgamma(y + r) - std::lgamma(r) - std::lgamma(y + 1.0) + r * std::log(r / (r + mu)) + y * std::log(mu / (r + mu));
    };
    L.sample = [mean](const Vec& psi, Xs x, const Vec& cov, const Vec& ex, std::mt19937& rng) {
        const double mu = std::max(mean(psi, x, cov), 1e-12), r = std::exp(ex[0]);
        std::gamma_distribution<double> g(r, mu / r); std::poisson_distribution<int> po(std::max(g(rng), 1e-12)); return double(po(rng));
    };
    return L;
}

/**
 * @brief Zero-inflated Poisson likelihood.
 * @param f Callable returning (lambda, pi), where pi = P(structural zero).
 * @return A Lik combining a point mass at zero with a Poisson component.
 * @note Requires lambda > 0 and 0 <= pi < 1; otherwise the log-likelihood is -inf.
 */
inline Lik zip(std::function<std::pair<double,double>(const Vec&, Xs, const Vec&)> f) {
    Lik L;
    L.logLik = [f](const Vec& psi, Xs x, const Vec& cov, double y, const Vec&) {
        const auto [lam, pi] = f(psi, x, cov);
        if (!(lam > 0.0) || !(pi >= 0.0 && pi < 1.0)) return -std::numeric_limits<double>::infinity();
        if (y < 0.5) return std::log(pi + (1.0 - pi) * std::exp(-lam));
        return std::log(1.0 - pi) + y * std::log(lam) - lam - std::lgamma(y + 1.0);
    };
    L.sample = [f](const Vec& psi, Xs x, const Vec& cov, const Vec&, std::mt19937& rng) {
        const auto [lam, pi] = f(psi, x, cov);
        std::bernoulli_distribution z(std::min(std::max(pi, 0.0), 1.0)); if (z(rng)) return 0.0;
        std::poisson_distribution<int> po(std::max(lam, 1e-12)); return double(po(rng));
    };
    return L;
}

/**
 * @brief Proportional-odds ordinal likelihood.
 *
 * The response takes values in @f$\{0,\ldots,K\}@f$ with cumulative
 * probabilities
 * @f[
 *   P(y \le k) = \operatorname{logistic}(c_k - \eta),
 *   \qquad c_0 < \cdots < c_{K-1}.
 * @f]
 *
 * @param eta Callable returning the linear predictor @f$\eta@f$.
 * @param cuts Callable returning the ordered cut points
 *        @f$(c_0,\ldots,c_{K-1})@f$ from the parameters.
 * @return A Lik with @f$K+1@f$ categories.
 */
inline Lik ordinal(std::function<double(const Vec&, Xs, const Vec&)> eta,
                   std::function<Vec(const Vec&)> cuts) {
    Lik L;
    auto probs = [eta, cuts](const Vec& psi, Xs x, const Vec& cov) {
        const double e = eta(psi, x, cov); const Vec c = cuts(psi);
        const int K = static_cast<int>(c.size());
        Vec p(K + 1); double prev = 0.0;
        for (int k = 0; k < K; ++k) { const double cum = detail::logistic(c[k] - e); p[k] = cum - prev; prev = cum; }
        p[K] = 1.0 - prev; return p;
    };
    L.logLik = [probs](const Vec& psi, Xs x, const Vec& cov, double y, const Vec&) {
        const Vec p = probs(psi, x, cov); const int k = static_cast<int>(std::lround(y));
        if (k < 0 || k >= p.size() || !(p[k] > 0.0)) return -std::numeric_limits<double>::infinity();
        return std::log(p[k]);
    };
    L.sample = [probs](const Vec& psi, Xs x, const Vec& cov, const Vec&, std::mt19937& rng) {
        const Vec p = probs(psi, x, cov); std::discrete_distribution<int> d(p.data(), p.data() + p.size()); return double(d(rng));
    };
    return L;
}

/**
 * @brief Nominal categorical likelihood.
 * @param prob Callable returning a vector of probabilities summing to one.
 * @return A Lik with one category per probability-vector entry.
 */
inline Lik categorical(std::function<Vec(const Vec&, Xs, const Vec&)> prob) {
    Lik L;
    L.logLik = [prob](const Vec& psi, Xs x, const Vec& cov, double y, const Vec&) {
        const Vec p = prob(psi, x, cov); const int k = static_cast<int>(std::lround(y));
        if (k < 0 || k >= p.size() || !(p[k] > 0.0)) return -std::numeric_limits<double>::infinity();
        return std::log(p[k]);
    };
    L.sample = [prob](const Vec& psi, Xs x, const Vec& cov, const Vec&, std::mt19937& rng) {
        const Vec p = prob(psi, x, cov); std::discrete_distribution<int> d(p.data(), p.data() + p.size()); return double(d(rng));
    };
    return L;
}

/**
 * @brief Time-to-event likelihood for interval-censored or repeated events.
 *
 * Each row stores @c x[0] = @f$t@f$ (end of the interval) and
 * @c y = 1 when the event occurs at @f$t@f$, or @c y = 0 for censoring.
 * @p tstartCol selects the column of @c x containing the interval start
 * (a negative value means zero). The log-likelihood is
 * @f[
 *   \log L = y\log h(t) - [H(t) - H(t_{\text{start}})].
 * @f]
 *
 * @param hazard Callable @c hazard(psi, t, x, cov) returning @f$h(t)@f$.
 * @param cumhazard Callable @c cumhazard(psi, t, x, cov) returning @f$H(t)@f$.
 * @param tstartCol Column of @c x holding the interval start, or -1.
 * @return A Lik for time-to-event observations.
 */
inline Lik tte(std::function<double(const Vec&, double, Xs, const Vec&)> hazard,
               std::function<double(const Vec&, double, Xs, const Vec&)> cumhazard, int tstartCol = -1) {
    Lik L;
    L.logLik = [hazard, cumhazard, tstartCol](const Vec& psi, Xs x, const Vec& cov, double y, const Vec&) {
        const double t = x[0], t0 = (tstartCol >= 0) ? x[uz(tstartCol)] : 0.0;
        const double dH = cumhazard(psi, t, x, cov) - (t0 > 0.0 ? cumhazard(psi, t0, x, cov) : 0.0);
        if (!std::isfinite(dH) || dH < 0.0) return -std::numeric_limits<double>::infinity();
        double ll = -dH;
        if (y >= 0.5) { const double h = hazard(psi, t, x, cov); if (!(h > 0.0)) return -std::numeric_limits<double>::infinity(); ll += std::log(h); }
        return ll;
    };
    // Sampler for an INTERVAL row (tstart, t]: y = 1 with probability
    // 1 - exp(-(H(t) - H(tstart))).
    L.sample = [cumhazard, tstartCol](const Vec& psi, Xs x, const Vec& cov, const Vec&, std::mt19937& rng) {
        const double t = x[0], t0 = (tstartCol >= 0) ? x[uz(tstartCol)] : 0.0;
        const double dH = cumhazard(psi, t, x, cov) - (t0 > 0.0 ? cumhazard(psi, t0, x, cov) : 0.0);
        std::bernoulli_distribution b(std::min(std::max(1.0 - std::exp(-std::max(dH, 0.0)), 0.0), 1.0));
        return b(rng) ? 1.0 : 0.0;
    };
    return L;
}

/**
 * @brief Samples the time of the next event after @p t0 by inverting the
 *        cumulative hazard.
 *
 * Solves @f$H(T) = H(t_0) - \log U@f$ by bisection. The predictor row @c x is
 * passed by value and its time column @c x[0] is overwritten during the search.
 *
 * @param cumhazard Callable returning @f$H(t)@f$.
 * @param psi Individual parameters.
 * @param x Predictor row; @c x[0] is the time column.
 * @param cov Subject covariates.
 * @param t0 Time from which the search starts.
 * @param tmax Latest admissible event time.
 * @param rng Random number generator.
 * @return The sampled event time, or @c +infinity when no event occurs before
 *         @p tmax.
 */
inline double sampleEventTime(const std::function<double(const Vec&, double, Xs, const Vec&)>& cumhazard,
                              const Vec& psi, std::vector<double> x, const Vec& cov,
                              double t0, double tmax, std::mt19937& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double target = (t0 > 0.0 ? cumhazard(psi, t0, Xs(x.data(), x.size()), cov) : 0.0) - std::log(1.0 - U(rng));
    auto H = [&](double t) { x[0] = t; return cumhazard(psi, t, Xs(x.data(), x.size()), cov); };
    if (!(H(tmax) >= target)) return std::numeric_limits<double>::infinity();
    double lo = t0, hi = tmax;
    for (int it = 0; it < 80; ++it) { const double mid = 0.5 * (lo + hi); (H(mid) < target ? lo : hi) = mid; }
    return 0.5 * (lo + hi);
}

} // namespace lik
} // namespace mcmckl
