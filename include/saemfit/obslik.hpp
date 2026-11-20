// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file obslik.hpp
 * @brief Observation likelihood given the individual effect @f$\psi_i@f$.
 *
 * The effect is expressed on the natural scale (that is, after applying the
 * parameter transformation). The header supports:
 *
 *  - multiple predictors per observation
 *    (@c x = [t, p1, p2, ..., ytype]) and subject-level covariates;
 *  - multiple response types (@c ytype), each with its own error model and
 *    parameters;
 *  - error models: constant, proportional, exponential, combined,
 *    combined-quadratic, and power (see @ref resmod::ErrorKind);
 *  - censored data (@c cens = -1 left, +1 right, optional limit for
 *    interval censoring):
 *      * exact likelihood @f$P(Y \in \text{region})@f$ (M3 method);
 *      * M-step by stochastic imputation of the censored observation from a
 *        truncated normal (Samson, Lavielle & Mentre, 2006), which preserves
 *        the closed-form residual-error updates;
 *  - a custom per-observation likelihood (binary, count, time-to-event, ...)
 *    with optional extra parameters;
 *  - M-step for the error parameters from sufficient statistics.
 *
 * Error parameters in @f$\theta@f$ (per response):
 *  - Constant/Exponential: @f$[\log a]@f$;
 *  - Proportional: @f$[\log b]@f$;
 *  - Combined/Combined2: @f$[\log a, \log b]@f$;
 *  - Power: @f$[\log a, \log b, c]@f$;
 * followed by the @c nExtra extra parameters (untransformed).
 *
 * Requires: Eigen3, C++20, residual.hpp, normal.hpp, nelder.hpp.
 */

#pragma once
#include "numcast.hpp"

#include "residual.hpp"
#include "normal.hpp"
#include "nelder.hpp"
#include "ode.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

/**
 * @brief Row-wise prediction callback.
 *
 * Receives @c x = [predictors..., ytype], with @c ytype encoded as a double in
 * the last position.
 */
using PredictRow = std::function<double(const Vec& psi, std::span<const double> x, const Vec& cov)>;

/**
 * @brief Read-only view of one subject's observations.
 *
 * The predictor matrix has @c n rows and @c stride columns
 * (@c [t, predictors..., ytype]). The dose record is optional and may be null
 * or empty.
 */
struct ObsView {
    int n = 0;                       ///< Number of observations.
    int stride = 0;                  ///< Number of predictor columns per row.
    const double* x = nullptr;       ///< Row-major predictor storage.
    const std::vector<ode::Dose>* doses = nullptr; ///< Optional dose record.

    /** @brief Returns the predictor row @p j. */
    std::span<const double> row(int j) const { return std::span<const double>(x + uz(j) * uz(stride), uz(stride)); }
    /** @brief Returns the time of observation @p j (first column). */
    double time(int j) const { return x[uz(j) * uz(stride)]; }
    /** @brief Returns the response type of observation @p j (last column). */
    double ytype(int j) const { return x[uz(j) * uz(stride) + uz(stride) - 1]; }
    /** @brief Returns the dose record, or an empty list when none is present. */
    const std::vector<ode::Dose>& doseList() const { static const std::vector<ode::Dose> none; return doses ? *doses : none; }
};

/**
 * @brief Vectorized prediction callback for a whole subject.
 *
 * Fills @c f(0..n-1) with all predictions for the subject in one call, for
 * example with a single ODE integration over every observation time.
 */
using PredictSubject = std::function<void(const Vec& psi, const ObsView& obs, const Vec& cov, Vec& f)>;

/**
 * @brief Per-observation log-likelihood
 *        @f$\log p(y \mid \psi, x, cov, extra)@f$.
 */
using LogLikObs  = std::function<double(const Vec& psi, std::span<const double> x, const Vec& cov,
                                        double y, const Vec& extra)>;

/**
 * @brief Error-model configuration for one response.
 */
struct ErrorSpec {
    resmod::ErrorKind type = resmod::ErrorKind::Proportional; ///< Error model type.
    resmod::Pres init = resmod::Pres(0.1, 0.1, 1.0); ///< Initial error parameters.
    std::vector<char> fixed;                        ///< Per-parameter fixed flags (empty = all free).
};

/**
 * @brief Flat storage for one subject's observations.
 */
struct ObsSubject {
    int stride = 0;                   ///< Columns of @c x per observation (@c npred + 1).
    std::vector<double> x;            ///< @c n * stride row-major predictors.
    std::vector<double> y;            ///< Observations.
    std::vector<int>    ytype;        ///< Response type of each observation.
    std::vector<int>    cens;         ///< Censoring: -1 left, 0 exact, +1 right.
    std::vector<double> lim;          ///< Optional other limit (NaN = none).
    std::vector<ode::Dose> doses;     ///< Optional dose record for the subject.
    /** @brief Returns the number of observations. */
    int n() const { return static_cast<int>(y.size()); }
};

/**
 * @brief Decoded error and extra parameters.
 */
struct ErrParams {
    std::vector<resmod::Pres> pres;   ///< Error parameters per response.
    Vec extra;                        ///< Extra parameters (custom likelihood).
};

/**
 * @brief Observation likelihood for a full data set.
 *
 * The class owns the data and pre-computes the parameter layout for the error
 * models. It provides the likelihood, prediction, sufficient-statistic, and
 * closed-form M-step operations used by the SAEM engine.
 */
class ObsLik {
public:
    /**
     * @brief Construction specification.
     */
    struct Spec {
        PredictRow predict;             ///< Row-wise prediction function.
        PredictSubject predictSubject;  ///< Optional vectorized prediction (takes priority).
        LogLikObs  logLik;              ///< Optional custom likelihood (replaces the Gaussian model).
        int  nExtra = 0;                ///< Number of extra parameters.
        Vec  extra0;                    ///< Initial extra parameters.
        std::vector<char> extraFixed;   ///< Fixed flags for the extra parameters.
        std::vector<ErrorSpec> errors;  ///< One error specification per response (empty -> one proportional).
    };

    ObsLik() = default;

    /**
     * @brief Constructs the likelihood object and validates the data.
     * @param spec Model specification.
     * @param subj Observations, one entry per subject.
     * @param cov Subject covariates.
     * @throws std::invalid_argument If required callbacks are missing, the
     *         parameter layout is inconsistent, or the data arrays have
     *         incompatible lengths.
     */
    ObsLik(Spec spec, std::vector<ObsSubject> subj, std::vector<Vec> cov)
      : spec_(std::move(spec)), S_(std::move(subj)), cov_(std::move(cov)), rng_(0x5eed1234u)
    {
        if (!spec_.predict && !spec_.predictSubject) throw std::invalid_argument("ObsLik: predict or predictSubject is required");
        if (cov_.size() != S_.size()) cov_.resize(S_.size());
        custom_ = static_cast<bool>(spec_.logLik);
        if (!custom_ && spec_.errors.empty()) spec_.errors.push_back(ErrorSpec{});
        if (custom_) spec_.errors.clear();
        const int R = static_cast<int>(spec_.errors.size());
        if (spec_.extra0.size() == 0) spec_.extra0 = Vec::Zero(spec_.nExtra);
        if (spec_.extra0.size() != spec_.nExtra) throw std::invalid_argument("extra0 has incorrect dimension");
        // Error-parameter layout.
        errOff_.assign(uz(R), 0); nErr_ = 0;
        for (int r = 0; r < R; ++r) {
            errOff_[uz(r)] = nErr_;
            nErr_ += resmod::nParams(spec_.errors[uz(r)].type);
            if (spec_.errors[uz(r)].fixed.empty()) spec_.errors[uz(r)].fixed.assign(uz(resmod::nParams(spec_.errors[uz(r)].type)), 0);
            if (static_cast<int>(spec_.errors[uz(r)].fixed.size()) != resmod::nParams(spec_.errors[uz(r)].type))
                throw std::invalid_argument("ErrorSpec.fixed has incorrect dimension");
        }
        extraOff_ = nErr_; nErr_ += spec_.nExtra;
        if (spec_.extraFixed.empty()) spec_.extraFixed.assign(uz(spec_.nExtra), 0);
        // Validate data and count observations per response.
        nObsResp_.assign(uz(std::max(R, 1)), 0);
        for (auto& s : S_) {
            const int n = s.n();
            if (s.ytype.empty()) s.ytype.assign(uz(n), 0);
            if (s.cens.empty())  s.cens.assign(uz(n), 0);
            if (s.lim.empty())   s.lim.assign(uz(n), std::numeric_limits<double>::quiet_NaN());
            if (static_cast<int>(s.x.size()) != n * s.stride || static_cast<int>(s.ytype.size()) != n ||
                static_cast<int>(s.cens.size()) != n || static_cast<int>(s.lim.size()) != n)
                throw std::invalid_argument("ObsLik: subject has arrays of different lengths");
            for (int j = 0; j < n; ++j) {
                if (!std::isfinite(s.y[uz(j)])) throw std::invalid_argument("ObsLik: y is not finite");
                const int r = s.ytype[uz(j)];
                if (!custom_ && (r < 0 || r >= R)) throw std::invalid_argument("ObsLik: ytype is out of range");
                if (!custom_) {
                    ++nObsResp_[uz(r)];
                    if (s.cens[uz(j)] != 0) hasCens_ = true;
                    if (spec_.errors[uz(r)].type == resmod::ErrorKind::Exponential && s.cens[uz(j)] == 0 && !(s.y[uz(j)] > 0.0))
                        throw std::invalid_argument("exponential error model requires y > 0");
                }
                ++nObs_;
            }
        }
        // Sufficient-statistic layout.
        statOff_.assign(uz(R), 0); nStats_ = 0;
        for (int r = 0; r < R; ++r) {
            statOff_[uz(r)] = nStats_;
            nStats_ += scalarStat(spec_.errors[uz(r)].type) ? 1 : 2 * static_cast<int>(nObsResp_[uz(r)]);
        }
    }

    // --- Sizes --------------------------------------------------------------
    /** @brief Returns the number of subjects. */
    int  nSubjects() const { return static_cast<int>(S_.size()); }
    /** @brief Returns the total number of observations. */
    long nObs() const { return nObs_; }
    /** @brief Returns the number of responses. */
    int  nResp() const { return static_cast<int>(spec_.errors.size()); }
    /** @brief Returns the number of error parameters (all responses plus extra). */
    int  nErrParams() const { return nErr_; }
    /** @brief Returns the number of extra parameters. */
    int  nExtra() const { return spec_.nExtra; }
    /** @brief Returns the number of sufficient statistics (zero for custom likelihoods). */
    int  nStats() const { return custom_ ? 0 : nStats_; }
    /** @brief Returns true when a custom likelihood is used. */
    bool custom() const { return custom_; }
    /** @brief Returns true when the data contain censored observations. */
    bool hasCensoring() const { return hasCens_; }
    /** @brief Returns the observations of subject @p i. */
    const ObsSubject& subject(int i) const { return S_[uz(i)]; }
    /** @brief Returns the covariates of subject @p i. */
    const Vec& subjectCov(int i) const { return cov_[uz(i)]; }
    /** @brief Returns the error specification of response @p r. */
    const ErrorSpec& errorSpec(int r) const { return spec_.errors[uz(r)]; }
    /** @brief Returns the offset of response @p r inside the error block. */
    int  errOffset(int r) const { return errOff_[uz(r)]; }
    /** @brief Returns the offset of the extra parameters inside the error block. */
    int  extraOffset() const { return extraOff_; }
    /** @brief Returns the number of observations of response @p r. */
    long nObsOfResponse(int r) const { return nObsResp_[uz(r)]; }

    // --- Error parameters in theta ------------------------------------------
    /**
     * @brief Builds the initial error-parameter vector.
     * @return Vector of length @ref nErrParams.
     */
    Vec initialErr() const {
        Vec t = Vec::Zero(nErr_);
        for (int r = 0; r < nResp(); ++r) encodeOne(spec_.errors[uz(r)].type, spec_.errors[uz(r)].init, t, errOff_[uz(r)]);
        if (spec_.nExtra > 0) t.segment(extraOff_, spec_.nExtra) = spec_.extra0;
        return t;
    }

    /**
     * @brief Returns the fixed-parameter mask for the error block.
     * @return One flag per error parameter (length @ref nErrParams).
     */
    std::vector<char> fixedMask() const {
        std::vector<char> m(uz(nErr_), 0);
        for (int r = 0; r < nResp(); ++r)
            for (size_t k = 0; k < spec_.errors[uz(r)].fixed.size(); ++k) m[uz(errOff_[uz(r)]) + k] = spec_.errors[uz(r)].fixed[k];
        for (int k = 0; k < spec_.nExtra; ++k) m[uz(extraOff_ + k)] = spec_.extraFixed[uz(k)];
        return m;
    }

    /**
     * @brief Decodes error and extra parameters from @f$\theta@f$.
     * @param theta Full parameter vector.
     * @param off Offset of the error block in @p theta.
     * @return The decoded parameters.
     */
    ErrParams decode(const Vec& theta, int off) const {
        ErrParams ep;
        ep.pres.resize(uz(nResp()));
        for (int r = 0; r < nResp(); ++r) ep.pres[uz(r)] = decodeOne(spec_.errors[uz(r)].type, theta, off + errOff_[uz(r)]);
        if (spec_.nExtra > 0) ep.extra = theta.segment(off + extraOff_, spec_.nExtra);
        return ep;
    }

    // --- Prediction ---------------------------------------------------------
    /**
     * @brief Returns a view over the observations of subject @p i.
     * @param i Subject index.
     * @return The observation view.
     */
    ObsView view(int i) const {
        const auto& s = S_[uz(i)];
        ObsView v; v.n = s.n(); v.stride = s.stride; v.x = s.x.data(); v.doses = &s.doses;
        return v;
    }

    /**
     * @brief Computes all predictions of subject @p i.
     * @param i Subject index.
     * @param psi Individual effects.
     * @param f Output prediction vector (resized to the subject's row count).
     * @throws std::runtime_error If a vectorized predictor returns the wrong
     *         number of predictions.
     */
    void predictAll(int i, const Vec& psi, Vec& f) const {
        const auto& s = S_[uz(i)];
        const int n = s.n();
        f.resize(n);
        if (spec_.predictSubject) {
            spec_.predictSubject(psi, view(i), cov_[uz(i)], f);
            if (f.size() != n) throw std::runtime_error("predictSubject returned an incorrect number of predictions");
            return;
        }
        for (int j = 0; j < n; ++j)
            f[j] = spec_.predict(psi, std::span<const double>(s.x.data() + uz(j) * uz(s.stride), uz(s.stride)), cov_[uz(i)]);
    }

    /**
     * @brief Computes the prediction for one observation.
     * @param i Subject index.
     * @param j Observation index.
     * @param psi Individual effects.
     * @return The model prediction.
     */
    double predictAt(int i, int j, const Vec& psi) const {
        if (spec_.predictSubject) { Vec f; predictAll(i, psi, f); return f[j]; }
        const auto& s = S_[uz(i)];
        return spec_.predict(psi, std::span<const double>(s.x.data() + uz(j) * uz(s.stride), uz(s.stride)), cov_[uz(i)]);
    }

    /**
     * @brief Returns the residual standard deviation of observation @p j.
     * @param i Subject index.
     * @param j Observation index.
     * @param f Model prediction.
     * @param ep Decoded error parameters.
     * @return The error scale @f$g(f)@f$.
     */
    double errSd(int i, int j, double f, const ErrParams& ep) const {
        const int r = S_[uz(i)].ytype[uz(j)];
        return resmod::error(f, ep.pres[uz(r)], spec_.errors[uz(r)].type);
    }

    /**
     * @brief Returns true when observation @p j uses the exponential error model.
     * @param i Subject index.
     * @param j Observation index.
     */
    bool isExp(int i, int j) const {
        return !custom_ && spec_.errors[uz(S_[uz(i)].ytype[uz(j)])].type == resmod::ErrorKind::Exponential;
    }

    // --- log p(y_i | psi) ---------------------------------------------------
    /**
     * @brief Computes @f$\log p(y_i \mid \psi_i)@f$ for subject @p i.
     * @param i Subject index.
     * @param psi Individual effects.
     * @param ep Decoded error parameters.
     * @return The subject log-likelihood.
     */
    double logLik(int i, const Vec& psi, const ErrParams& ep) const {
        Vec fall;
        return logLik(i, psi, ep, fall);
    }

    /**
     * @brief Computes @f$\log p(y_i \mid \psi_i)@f$ using caller-provided scratch space.
     * @param i Subject index.
     * @param psi Individual effects.
     * @param ep Decoded error parameters.
     * @param fall Reusable prediction workspace (avoids an allocation per evaluation).
     * @return The subject log-likelihood.
     */
    double logLik(int i, const Vec& psi, const ErrParams& ep, Vec& fall) const {
        const double ninf = -std::numeric_limits<double>::infinity();
        const auto& s = S_[uz(i)];
        const double cst = 0.5 * std::log(2.0 * std::numbers::pi);
        double lp = 0.0;
        const int n = s.n();
        if (!custom_) predictAll(i, psi, fall);
        for (int j = 0; j < n; ++j) {
            std::span<const double> x(s.x.data() + uz(j) * uz(s.stride), uz(s.stride));
            if (custom_) {
                const double v = spec_.logLik(psi, x, cov_[uz(i)], s.y[uz(j)], ep.extra);
                if (!std::isfinite(v)) return ninf;
                lp += v;
                continue;
            }
            const double f = fall[j];
            if (!std::isfinite(f)) return ninf;
            const int r = s.ytype[uz(j)];
            const resmod::ErrorKind type = spec_.errors[uz(r)].type;
            const bool ex = (type == resmod::ErrorKind::Exponential);
            if (ex && !(f > 0.0)) return ninf;
            const double g = resmod::error(f, ep.pres[uz(r)], type);
            if (!(g > 0.0) || !std::isfinite(g)) return ninf;
            const double mean = ex ? std::log(f) : f;
            const int c = s.cens[uz(j)];
            if (c == 0) {
                const double yv = ex ? std::log(s.y[uz(j)]) : s.y[uz(j)];
                const double z = (yv - mean) / g;
                lp += -0.5 * z * z - std::log(g) - cst;
                if (ex) lp -= std::log(s.y[uz(j)]);                  // Jacobian d(log y)/dy.
            } else {
                double zlo, zhi;
                if (!censBounds(s, j, ex, mean, g, zlo, zhi)) return ninf;
                const double v = normLogProbInterval(zlo, zhi);
                if (!std::isfinite(v)) return ninf;
                lp += v;
            }
        }
        return lp;
    }

    // --- Residual-error sufficient statistics --------------------------------
    //   Constant/Exponential : mean of r^2                (1 scalar per response)
    //   Proportional         : mean of (r/f)^2            (1 scalar)
    //   Combined/Comb2/Power : (r^2_k, |f|_k) per observation
    // Censored observations are IMPUTED from the truncated normal N(f, g^2)
    // restricted to the censored region (uses the internal RNG: call serially).
    /**
     * @brief Computes the residual-error sufficient statistics.
     * @param psis Individual effects sampled during the S-step.
     * @param theta Full parameter vector.
     * @param off Offset of the error block in @p theta.
     * @return The sufficient-statistic vector (empty for custom likelihoods).
     * @throws std::runtime_error If a prediction is non-finite or inadmissible.
     */
    Vec stats(const std::vector<Vec>& psis, const Vec& theta, int off) const {
        if (custom_) return Vec();
        Vec rs = Vec::Zero(nStats_);
        const ErrParams ep = decode(theta, off);
        std::vector<long> k(uz(nResp()), 0);
        for (int i = 0; i < nSubjects(); ++i) {
            const auto& s = S_[uz(i)];
            Vec fall; predictAll(i, psis[uz(i)], fall);
            for (int j = 0; j < s.n(); ++j) {
                const int r = s.ytype[uz(j)];
                const resmod::ErrorKind type = spec_.errors[uz(r)].type;
                const bool ex = (type == resmod::ErrorKind::Exponential);
                const double f = fall[j];
                if (!std::isfinite(f) || (ex && !(f > 0.0)))
                    throw std::runtime_error("non-finite sufficient statistic (inadmissible prediction)");
                const double g = resmod::error(f, ep.pres[uz(r)], type);
                const double mean = ex ? std::log(f) : f;
                double res;
                if (s.cens[uz(j)] == 0) {
                    res = (ex ? std::log(s.y[uz(j)]) : s.y[uz(j)]) - mean;
                } else {
                    double zlo, zhi;
                    if (!censBounds(s, j, ex, mean, g, zlo, zhi) || !(g > 0.0)) res = 0.0;
                    else res = g * sampleTruncStdNormal(zlo, zhi, rng_);
                }
                const double invN = 1.0 / double(std::max<long>(1, nObsResp_[uz(r)]));
                if (scalarStat(type)) {
                    if (type == resmod::ErrorKind::Proportional) {
                        const double af = std::max(std::abs(f), 1e-12);
                        rs[statOff_[uz(r)]] += invN * (res / af) * (res / af);
                    } else rs[statOff_[uz(r)]] += invN * res * res;
                } else {
                    rs[statOff_[uz(r)] + 2 * k[uz(r)]]     = res * res;
                    rs[statOff_[uz(r)] + 2 * k[uz(r)] + 1] = std::abs(f);
                }
                ++k[uz(r)];
            }
        }
        return rs;
    }

    /**
     * @brief Closed-form or numerical M-step for the error parameters.
     *
     * Updates @c theta[off + ...] with the error parameters (on the log scale)
     * that maximize @f$Q@f$ given the sufficient statistics.
     *
     * @param rs Sufficient statistics returned by @ref stats.
     * @param theta Parameter vector to update in place.
     * @param off Offset of the error block in @p theta.
     */
    void mstep(const Vec& rs, Vec& theta, int off) const {
        constexpr double lo = -18.0, hi = 12.0;
        for (int r = 0; r < nResp(); ++r) {
            const ErrorSpec& es = spec_.errors[uz(r)];
            const int o = off + errOff_[uz(r)];
            if (scalarStat(es.type)) {
                if (!es.fixed[0]) theta[o] = std::clamp(0.5 * std::log(std::max(rs[statOff_[uz(r)]], 1e-300)), lo, hi);
                continue;
            }
            // Numerical: minimize sum_k [ s_k / g_k^2 + 2 log g_k ],
            // with g_k = error(F_k; a, b, c).
            const int np = resmod::nParams(es.type);
            std::vector<int> freeIdx;
            for (int k2 = 0; k2 < np; ++k2) if (!es.fixed[uz(k2)]) freeIdx.push_back(k2);
            if (freeIdx.empty()) continue;
            const long N = nObsResp_[uz(r)];
            const int so = statOff_[uz(r)];
            Vec base = theta.segment(o, np);
            auto obj = [&](const Vec& q) -> double {
                Vec full = base;
                for (size_t m = 0; m < freeIdx.size(); ++m) full[freeIdx[m]] = q[tix(m)];
                resmod::Pres pr(std::exp(full[0]), std::exp(full[1]), np > 2 ? full[2] : 1.0);
                double v = 0.0;
                for (long k2 = 0; k2 < N; ++k2) {
                    const double g = resmod::error(rs[so + 2 * k2 + 1], pr, es.type);
                    if (!(g > 0.0) || !std::isfinite(g)) return std::numeric_limits<double>::infinity();
                    v += rs[so + 2 * k2] / (g * g) + 2.0 * std::log(g);
                }
                return v;
            };
            Vec q0(freeIdx.size());
            for (size_t m = 0; m < freeIdx.size(); ++m) q0[tix(m)] = base[freeIdx[m]];
            try {
                const Vec sc = Vec::Constant(q0.size(), 0.3);
                const NMOutcome res = nelderMeadEx(obj, q0, sc, 400, 1e-10);
                for (size_t m = 0; m < freeIdx.size(); ++m)
                    theta[o + freeIdx[m]] = std::clamp(res.x[tix(m)], freeIdx[m] == 2 ? -10.0 : lo, hi);
            } catch (const std::exception&) { /* Keep the current value. */ }
        }
    }

private:
    Spec spec_;
    std::vector<ObsSubject> S_;
    std::vector<Vec> cov_;
    bool custom_ = false, hasCens_ = false;
    int nErr_ = 0, extraOff_ = 0, nStats_ = 0;
    long nObs_ = 0;
    std::vector<int> errOff_, statOff_;
    std::vector<long> nObsResp_;
    mutable std::mt19937 rng_;

    static bool scalarStat(resmod::ErrorKind t) {
        return t == resmod::ErrorKind::Constant || t == resmod::ErrorKind::Proportional || t == resmod::ErrorKind::Exponential;
    }

    // Standardized (z) limits of the censored region of observation j.
    static bool censBounds(const ObsSubject& s, int j, bool ex, double mean, double g,
                           double& zlo, double& zhi) {
        const double inf = std::numeric_limits<double>::infinity();
        double lo = -inf, hi = inf;
        const double yv = s.y[uz(j)], L = s.lim[uz(j)];
        if (s.cens[uz(j)] < 0) { hi = yv; if (!std::isnan(L)) lo = L; }
        else               { lo = yv; if (!std::isnan(L)) hi = L; }
        if (ex) {                                              // Log scale.
            if (hi <= 0.0) return false;
            lo = (lo > 0.0) ? std::log(lo) : -inf;
            hi = std::isinf(hi) ? inf : std::log(hi);
        }
        zlo = (lo - mean) / g;
        zhi = (hi - mean) / g;
        return zhi > zlo;
    }

    static resmod::Pres decodeOne(resmod::ErrorKind t, const Vec& th, int o) {
        switch (t) {
            case resmod::ErrorKind::Constant:
            case resmod::ErrorKind::Exponential:  return resmod::Pres(std::exp(th[o]), 0.0);
            case resmod::ErrorKind::Proportional: return resmod::Pres(0.0, std::exp(th[o]));
            case resmod::ErrorKind::Combined:
            case resmod::ErrorKind::Combined2:   return resmod::Pres(std::exp(th[o]), std::exp(th[o + 1]));
            case resmod::ErrorKind::Power:     return resmod::Pres(std::exp(th[o]), std::exp(th[o + 1]), th[o + 2]);
        }
        return resmod::Pres();
    }

    static void encodeOne(resmod::ErrorKind t, const resmod::Pres& p, Vec& th, int o) {
        const double eps = 1e-12;
        switch (t) {
            case resmod::ErrorKind::Constant:
            case resmod::ErrorKind::Exponential:  th[o] = std::log(std::max(eps, p.a)); break;
            case resmod::ErrorKind::Proportional: th[o] = std::log(std::max(eps, p.b)); break;
            case resmod::ErrorKind::Combined:
            case resmod::ErrorKind::Combined2:
                th[o] = std::log(std::max(eps, p.a)); th[o + 1] = std::log(std::max(eps, p.b)); break;
            case resmod::ErrorKind::Power:
                th[o] = std::log(std::max(eps, p.a)); th[o + 1] = std::log(std::max(eps, p.b)); th[o + 2] = p.c; break;
        }
    }
};

} // namespace mcmckl
