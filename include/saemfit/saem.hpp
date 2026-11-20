/**
 * @file saem.hpp
 * @brief Public SAEM interface for single-level and hierarchical nonlinear
 *        mixed-effects models.
 *
 * This header provides a high-level facade in which the caller describes:
 *  - @ref smx::Data: observations in long format (id, time, y, optional
 *    additional predictors, response type, censoring information, and
 *    subject-level covariates).
 *  - @ref smx::Model: the prediction function f(psi, x) (or a custom
 *    likelihood), per-response error models, parameter transformations,
 *    partial random effects, the covariate model, and block covariance
 *    structure.
 *  - @ref smx::Control: initial values (or automatic initialization), fixed
 *    parameters, and algorithm options.
 *
 * The main entry points are @ref smx::saem and @ref smx::fit, which return a
 * @ref smx::Results object.
 *
 * @par Single-level model
 * @f[
 *   \phi_i = D_i \beta + \eta_i, \qquad \eta_i \sim N(0, \Omega)
 * @f]
 * applied only to parameters with a random effect,
 * @f[
 *   \psi_i = h(\phi_i)
 * @f]
 * where @f$h@f$ is the identity, log, logit, or probit transformation selected
 * per parameter, and @f$y_{ij} \mid \psi_i@f$ follows a constant,
 * proportional, exponential, combined, combined-quadratic, or power error
 * model, a censoring model (M3, imputed during the M-step), or a custom
 * likelihood.
 *
 * @par Estimation
 * Classical SAEM (sufficient statistics, closed-form/GLS M-step, and numerical
 * step), a parallelizable MCMC sampler with four kernels (random walk,
 * independence, Gibbs, and Laplace/MAP), simulated annealing, multiple chains,
 * and automatic stopping.
 *
 * @par Inference
 * Linearized FIM (FOCE with interaction) for all parameters, log-likelihood by
 * adaptive Gauss-Hermite quadrature, importance sampling, or linearization,
 * conditional means (EBE) with standard deviations, ESS, and R-hat.
 *
 * @par Hierarchical models
 * Nested two-level models are enabled by filling @ref smx::Data::group. The
 * multilevel engine (`multilevel.hpp`) supports parameter transformations. The
 * remaining advanced options (extra predictors, censoring, ytype, covariates,
 * parameters without random effects, and so on) are available only for the
 * single-level model.
 *
 * @note Requires Eigen3, C++20, and the project headers.
 */

#pragma once
#include "numcast.hpp"

#include "nlme.hpp"
#include "model.hpp"
#include "map.hpp"
#include "loglik.hpp"
#include "fisher.hpp"
#include "multilevel.hpp"
#include "ode.hpp"
#include "likelihoods.hpp"
#include "nlme_ml.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// The namespace is `smx` so that the fitting function can be named `saem`
// without ambiguity (a namespace `saem` and a function `saem` would collide).
namespace smx {

/** @brief Dense vector of optimizer parameters and observations. */
using Vec = Eigen::VectorXd;

/** @brief Dense matrix used for covariance and information matrices. */
using Mat = Eigen::MatrixXd;

/** @brief Parameter transformation type used by the modeling engine. */
using mcmckl::Transform;
using mcmckl::uz;
using mcmckl::toi;
using mcmckl::tix;
using mcmckl::tod;

// ===========================================================================
// Residual error model
// ===========================================================================

/**
 * @brief Available residual error models.
 *
 * The C++ enumerator names are English; the Python bindings keep the stable
 * lowercase names ``constant``, ``proportional``, ``exponential``, ``combined``,
 * ``combined2`` and ``power``.
 */
enum class ErrorModel { Constant, Proportional, Exponential, Combined, Combined2, Power };

/**
 * @brief Convert a public @ref ErrorModel to the internal residual model type.
 *
 * @param e Public error model.
 * @return The corresponding `resmod::ErrorKind` value. If @p e does not match a
 *         known enumerator, `resmod::ErrorKind::Proportional` is returned.
 */
inline resmod::ErrorKind toResid(ErrorModel e) {
    switch (e) {
        case ErrorModel::Constant:     return resmod::ErrorKind::Constant;
        case ErrorModel::Proportional: return resmod::ErrorKind::Proportional;
        case ErrorModel::Exponential:  return resmod::ErrorKind::Exponential;
        case ErrorModel::Combined:     return resmod::ErrorKind::Combined;
        case ErrorModel::Combined2:    return resmod::ErrorKind::Combined2;
        case ErrorModel::Power:        return resmod::ErrorKind::Power;
    }
    return resmod::ErrorKind::Proportional;
}

// ===========================================================================
// Data (long format)
// ===========================================================================

/**
 * @brief Observations and subject-level metadata in long format.
 *
 * Each element of @ref id, @ref time, and @ref y describes one observation.
 * Optional vectors must either be empty or contain exactly one value per
 * observation; this invariant is enforced by @ref check.
 *
 * Call @ref check before fitting. The fitting functions also validate the data
 * internally.
 */
struct Data {
    /** @brief Subject identifier for each observation. */
    std::vector<int>    id;

    /** @brief Observation time for each row (the first prediction predictor). */
    std::vector<double> time;

    /** @brief Observed response value. */
    std::vector<double> y;

    /**
     * @brief Cluster (center, site, and so on) for each observation.
     *
     * Optional. When non-empty, enables the hierarchical two-level model.
     * The value must be constant within each subject.
     */
    std::vector<int>    group;

    // --- per observation [OPTIONAL] ---

    /** @brief Response index in `0..R-1`; empty means every row uses response 0. */
    std::vector<int>    ytype;

    /**
     * @brief Censoring indicator.
     *
     * `-1` means left-censored (`Y <= y`), `+1` means right-censored
     * (`Y >= y`), and `0` means the observation is observed exactly.
     */
    std::vector<int>    cens;

    /**
     * @brief Optional censoring limit (`NaN` means none).
     *
     * Together with @ref y and @ref cens it defines the censoring interval
     * `(limit, y]` or `[y, limit)`.
     */
    std::vector<double> limit;

    /**
     * @brief Additional per-observation predictors.
     *
     * Examples include dose, infusion rate, and time-varying covariates. The
     * prediction function receives
     * `x = [time, predictors in predictorOrder..., ytype]`.
     */
    std::map<std::string, std::vector<double>> predictors;

    /** @brief Order in which @ref predictors are appended to the prediction input. */
    std::vector<std::string> predictorOrder;

    /**
     * @brief Add or replace a named predictor and register its order.
     *
     * @param name Predictor name.
     * @param values One value per observation.
     *
     * The name is appended to @ref predictorOrder only if it is not already
     * present.
     */
    void addPredictor(const std::string& name, std::vector<double> values) {
        predictors[name] = std::move(values);
        if (std::find(predictorOrder.begin(), predictorOrder.end(), name) == predictorOrder.end())
            predictorOrder.push_back(name);
    }

    // --- NONMEM-style dosing records (optional) ---

    /**
     * @brief One dosing record.
     *
     * Observations remain in @ref id, @ref time, and @ref y. The model receives
     * each subject's dose list through `ObsView::doseList()` (see
     * `ode::integrate`). @ref cmt is 1-based, as in NONMEM.
     *
     * - `rate > 0`: infusion of duration `amt / rate`; `rate = 0`: bolus.
     * - `rate = -1 / -2`: modeled rate R1 / modeled duration D1, resolved by
     *   `ode::resolveDoses`.
     * - `addl / ii`: @ref addl additional doses every @ref ii.
     * - `ss = 1 / 2`: steady state (requires `ii > 0`); replaces or adds to the
     *   previous state.
     */
    struct EventRow {
        int id;
        double time;
        int evid = 1;
        double amt = 0.0;
        int cmt = 1;
        double rate = 0.0;
        int addl = 0;
        double ii = 0.0;
        int ss = 0;
    };

    /** @brief Dosing records; see @ref EventRow. */
    std::vector<EventRow> events;

    /**
     * @brief Append a dosing record.
     *
     * @param subject Subject identifier.
     * @param t Dose time.
     * @param amt Dose amount.
     * @param cmt Compartment (1-based).
     * @param rate Infusion rate, or a modeled-rate code.
     * @param addl Number of additional doses.
     * @param ii Interdose interval for additional doses.
     */
    void addDose(int subject, double t, double amt, int cmt = 1, double rate = 0.0, int addl = 0, double ii = 0.0) {
        EventRow e; e.id = subject; e.time = t; e.amt = amt; e.cmt = cmt; e.rate = rate; e.addl = addl; e.ii = ii;
        events.push_back(e);
    }

    /**
     * @brief Return the expanded doses (ADDL/II) for subject @p sid, sorted by time.
     *
     * Steady-state handling (`ss = 1/2`) requires `ii > 0`; see `ode.hpp` (the
     * first dose carries the steady-state flag, while ADDL doses are normal
     * doses). Modeled rates/durations (`rate = -1 / -2`, R1/D1) must be
     * resolved with `ode::resolveDoses`.
     *
     * @param sid Subject identifier.
     * @return Expanded doses in ascending time order.
     * @throws std::invalid_argument If a dosing record has an unsupported
     *         `EVID`, invalid `SS`, `RATE`, `CMT`, or ADDL/SS without `II > 0`.
     */
    std::vector<ode::Dose> dosesOf(int sid) const {
        std::vector<ode::Dose> out;
        for (const auto& e : events) {
            if (e.id != sid) continue;
            if (e.evid != 1) throw std::invalid_argument("Data.events: only EVID = 1 (dose) is supported");
            if (e.ss < 0 || e.ss > 2) throw std::invalid_argument("Data.events: SS must be 0, 1 or 2");
            if (e.rate < 0.0 && e.rate != -1.0 && e.rate != -2.0) throw std::invalid_argument("Data.events: RATE may only be > 0, 0, -1 or -2");
            if (e.cmt < 1) throw std::invalid_argument("Data.events: CMT must be >= 1");
            if ((e.addl > 0 || e.ss != 0) && !(e.ii > 0.0)) throw std::invalid_argument("Data.events: ADDL > 0 / SS != 0 require II > 0");
            const int modeled = (e.rate == -1.0) ? -1 : (e.rate == -2.0 ? -2 : 0);
            const double dur = e.rate > 0.0 ? e.amt / e.rate : 0.0;
            for (int k = 0; k <= std::max(0, e.addl); ++k) {
                ode::Dose d{e.time + k * e.ii, e.cmt - 1, e.amt, dur};
                d.modeled = modeled;
                if (k == 0 && e.ss != 0) { d.ss = e.ss; d.ii = e.ii; }
                out.push_back(d);
            }
        }
        std::stable_sort(out.begin(), out.end(), [](const ode::Dose& a, const ode::Dose& b) { return a.time < b.time; });
        return out;
    }

    /**
     * @brief Column-oriented NONMEM table, one row per record.
     *
     * `EVID = 0` denotes an observation and `EVID = 1` denotes a dose.
     */
    struct NonmemTable {
        std::vector<int> id, evid, mdv, cmt, addl, ss, cens;
        std::vector<double> time, dv, amt, rate, ii, limit;
    };

    /**
     * @brief Build a @ref Data object from a NONMEM-style table.
     *
     * Observation records (`EVID = 0`) with `MDV != 0` are skipped. Dose
     * records (`EVID = 1`) are copied into @ref events.
     *
     * @param t Column-oriented NONMEM table.
     * @return The corresponding long-format data set.
     * @throws std::invalid_argument If an `EVID` other than 0 or 1 is present.
     */
    static Data fromNonmem(const NonmemTable& t) {
        Data d;
        const size_t N = t.id.size();
        auto at = [&](const auto& v, size_t i, auto def) { return v.size() == N ? v[i] : def; };
        bool anyCens = !t.cens.empty(), anyLim = !t.limit.empty();
        for (size_t i = 0; i < N; ++i) {
            const int ev = at(t.evid, i, 0);
            if (ev == 0) {
                if (at(t.mdv, i, 0) != 0) continue;
                d.id.push_back(t.id[i]); d.time.push_back(t.time[i]); d.y.push_back(t.dv[i]);
                if (anyCens) d.cens.push_back(t.cens[i]);
                if (anyLim) d.limit.push_back(t.limit[i]);
            } else if (ev == 1) {
                EventRow e; e.id = t.id[i]; e.time = t.time[i]; e.amt = at(t.amt, i, 0.0); e.cmt = at(t.cmt, i, 1);
                e.rate = at(t.rate, i, 0.0); e.addl = at(t.addl, i, 0); e.ii = at(t.ii, i, 0.0); e.ss = at(t.ss, i, 0);
                d.events.push_back(e);
            } else throw std::invalid_argument("fromNonmem: EVID " + std::to_string(ev) + " not supported (only 0 and 1)");
        }
        return d;
    }

    // CATEGORICAL covariates per subject: labels (one per subject, in subjectIds() order).
    // They are encoded automatically (0/1 variables per level, relative to `catReference` or the
    // first level in alphabetical order) when they appear in Model.covariateModel.
    std::map<std::string, std::vector<std::string>> catCov;
    std::map<std::string, std::string> catReference;

    // Per-subject (constant) covariates: cov[name][k] = value of subject k.
    std::map<std::string, Vec> cov;
    std::vector<std::string>   covOrder;   // concatenation order (empty = alphabetical)

    int nObs() const { return static_cast<int>(id.size()); }
    bool isHierarchical() const { return !group.empty(); }

    std::vector<int> subjectIds() const {
        std::vector<int> s;
        std::unordered_set<int> seen;
        for (int v : id) if (seen.insert(v).second) s.push_back(v);
        return s;
    }
    int nSubjects() const { return static_cast<int>(subjectIds().size()); }

    void check() const {
        const size_t N = id.size();
        if (N != time.size() || N != y.size())
            throw std::invalid_argument("Data: id, time and y must have the same size");
        if (!group.empty() && group.size() != N) throw std::invalid_argument("Data: group must have one value per observation");
        if (!ytype.empty() && ytype.size() != N) throw std::invalid_argument("Data: ytype must have one value per observation");
        if (!cens.empty()  && cens.size()  != N) throw std::invalid_argument("Data: cens must have one value per observation");
        if (!limit.empty() && limit.size() != N) throw std::invalid_argument("Data: limit must have one value per observation");
        for (const auto& nm : predictorOrder) {
            auto it = predictors.find(nm);
            if (it == predictors.end() || it->second.size() != N)
                throw std::invalid_argument("Data: predictor '" + nm + "' does not have one value per observation");
        }
        const int ns = nSubjects();
        for (auto& kv : catCov)
            if (static_cast<int>(kv.second.size()) != ns)
                throw std::invalid_argument("Data: categorical covariate '" + kv.first + "' does not have one label per subject");
        for (auto& kv : cov)
            if (static_cast<int>(kv.second.size()) != ns)
                throw std::invalid_argument("Data: covariate '" + kv.first + "' does not have one value per subject");
        if (!group.empty()) {
            std::map<int,int> g0;
            for (size_t r = 0; r < N; ++r) {
                auto it = g0.find(id[r]);
                if (it == g0.end()) g0[id[r]] = group[r];
                else if (it->second != group[r])
                    throw std::invalid_argument("Data: subject " + std::to_string(id[r]) + " belongs to more than one cluster");
            }
        }
    }

    std::vector<int> subjectGroup() const {
        std::vector<int> sids = subjectIds();
        std::map<int,int> sid2k;
        for (size_t k = 0; k < sids.size(); ++k) sid2k[sids[k]] = int(k);
        std::vector<int> g(sids.size(), -1);
        for (int r = 0; r < nObs(); ++r) g[uz(sid2k[id[uz(r)]])] = group[uz(r)];
        return g;
    }

    // Effective covariate order.
    std::vector<std::string> covNames() const {
        if (!covOrder.empty()) return covOrder;
        std::vector<std::string> v;
        for (const auto& kv : cov) v.push_back(kv.first);
        return v;
    }
    int covIndex(const std::string& name) const {
        const auto v = covNames();
        for (size_t j = 0; j < v.size(); ++j) if (v[j] == name) return static_cast<int>(j);
        throw std::invalid_argument("Data: covariate '" + name + "' not found");
    }
    // Covariates of the k-th subject (according to covNames).
    Vec covariatesOf(int k) const {
        const auto names = covNames();
        if (names.empty()) return Vec();
        Vec x(names.size());
        for (size_t j = 0; j < names.size(); ++j) {
            auto it = cov.find(names[j]);
            x[tix(j)] = (it == cov.end() ? 0.0 : it->second[k]);
        }
        return x;
    }
};

// ===========================================================================
// Model
// ===========================================================================
/** @brief One covariate effect in the covariate model. */
struct CovariateTerm {
    int         param;         // parameter (0..nPhi-1) that the covariate acts on
    std::string covariate;     // name in Data.cov
};

/** @brief User description of the nonlinear mixed-effects model. */
struct Model {
    // --- prediction (priority: predictRow > predict) ---
    // f(psi, t, x_subject)   [simple form]
    std::function<double(const Vec& psi, double t, const Vec& x)> predict;
    // f(psi, x, cov): x = [t, predictors..., ytype], cov = subject covariates
    mcmckl::PredictRow predictRow;
    // f(psi, obs, cov, f): ALL predictions of a subject in one call (takes priority over
    // predictRow). Essential for ODEs: a single integration per subject and evaluation.
    mcmckl::PredictSubject predictSubject;
    // Custom per-observation likelihood (replaces the Gaussian error):
    //   log p(y | psi, x, cov, extra)
    mcmckl::LogLikObs logLikObs;
    // Simulator of one observation (for smx::simulate with a custom likelihood); see lik::*
    std::function<double(const Vec& psi, std::span<const double> x, const Vec& cov, const Vec& extra, std::mt19937& rng)> simulateObs;
    int nExtra = 0;                    // extra parameters of logLikObs (values in Control.extra0)
    // Time to event: hazard functions (for smx::simulateEvents); tteStartCol = column of x with the
    // start of the interval (the same as in lik::tte), -1 = 0
    std::function<double(const Vec& psi, double t, std::span<const double> x, const Vec& cov)> tteCumHazard;
    int tteStartCol = -1;

    int nPhi = 1;
    ErrorModel error = ErrorModel::Proportional;   // response 0 if `errors` is empty
    std::vector<ErrorModel> errors;                // one per response (ytype)

    std::vector<Transform> transform;      // per parameter (empty = identity)
    std::vector<char>      randomEffect;   // per parameter (empty = all with a random effect)
    // Block covariance of the random effects: same id = full block;
    // empty = diagonal. Example: {0,0,1} -> corr. between the first 2, 3rd independent.
    std::vector<int>       omegaBlock;
    // Alternative: p x p pattern (non-zero = free covariance) for arbitrary structures
    std::vector<std::vector<char>> covariancePattern;
    // Hierarchical: structure of Psi (cluster effects), same id = full block (empty = diagonal)
    std::vector<int>       psiBlock;
    std::vector<CovariateTerm> covariateModel;   // covariate coefficients (phi scale)
    std::vector<std::string>   paramNames;       // optional (only for printing)
};

// ===========================================================================
// Control (all options have default values)
// ===========================================================================
/** @brief Method used to compute the standard errors. */
enum class SEMethod { Auto, Linearization, GaussHermite, Louis, Laplace, None };
/** @brief Method used to compute the log likelihood. */
enum class LLMethod { Auto, GaussHermite, ImportanceSampling, Linearization, Laplace, None };

/** @brief Fitting options. */
struct Control {
    // --- initial estimates ---
    // beta0 on the psi (natural) scale; empty -> AUTOMATIC (individual fits, see autoInit)
    Vec    beta0;
    Vec    betaCov0;       // initial covariate coefficients (len = number of terms; default 0)
    Vec    sd0;            // initial sds (phi scale) of the random effects (len nPhi)
    double a0 = -1.0;      // initial error of response 0 (a); < 0 -> automatic (individual-fit residuals)
    double b0 = -1.0;      // same (b)
    double c0 = 1.0;       // initial exponent (power model)
    std::vector<resmod::Pres> errorInit;   // per response (takes priority over a0/b0/c0)
    Vec    extra0;         // initial extra parameters (Model.nExtra)
    bool   autoInit = true;

    // --- FIXED parameters (not estimated; they stay at their initial value) ---
    std::vector<char> fixedBeta;    // len nPhi (intercept) or nPhi + n_terms
    std::vector<char> fixedSd;      // len nPhi (single-parameter blocks only)
    std::vector<char> fixedError;   // flat: error parameters of all responses, in order
    std::vector<char> fixedExtra;

    // --- multilevel (only if Data.group is present) ---
    Vec    sdCluster0;

    // --- algorithm ---
    int      nbIterations = 1000;
    int      nbBurnIn     = 200;
    int      nMCMCsweeps  = 5;
    unsigned seed         = 12345;
    // K&L variance floor during the exploratory phase. ON by default: together with
    // nChains = 2 it eliminates the variance collapse of classical SAEM (0/60 vs 3/60).
    bool     simulatedAnnealing = true;
    int      nChains      = 2;
    mcmckl::MStep mstep = mcmckl::MStep::Auto;
    mcmckl::ClassicStepControl step;
    int      nThreads     = 1;      // 1 = serial; 0 = all threads (predict must be thread-safe)
    bool     mapKernel    = true;   // Laplace/MAP kernel
    int      mapEvery     = 25;
    bool     autoStop     = false;  // stop on theta drift
    double   stopTol      = 1e-3;
    bool     adaptRW      = true;
    bool     verbose      = false;
    int      printEvery   = 100;

    // --- conditional mean (EBE) ---
    bool     computeCond  = true;
    int      nCondSamples = 1000;
    int      condBurnIn   = 200;
    int      condThin     = 2;

    // --- inference ---
    bool     computeSE    = true;
    SEMethod seMethod     = SEMethod::Auto;    // Auto: linearization (without censoring or custom likelihood); otherwise LAPLACE (deterministic)
    LLMethod llMethod     = LLMethod::Auto;    // Auto: adaptive GH (dim<=3) or importance sampling
    int      gqNodes      = 15;
    int      gqInnerNodes = 16;
    int      isSamples    = 2000;              // importance-sampling samples
    int      louisDraws   = 150;               // posterior samples per subject for Louis' formula
    int      louisISSamples = 4000;            // hierarchical: i.i.d. samples per cluster (importance sampling) for Louis

    // --- advanced ---
    mcmckl::RMControl  rm = defaultRM();
    mcmckl::SAControl  sa;

    static mcmckl::RMControl defaultRM() {
        mcmckl::RMControl r;
        r.schedule        = mcmckl::RMControl::Schedule::PowerDecay;
        r.gamma0          = 2.0;
        r.alpha           = 0.6;
        r.clip_gradient   = true;  r.grad_clip = 8.0;
        r.diagonal_precond = true;
        r.averaging       = true;  r.avg_start = 0;     // <= 0: second half of the iterations
        return r;
    }
    mcmckl::RMControl effectiveRM() const {
        mcmckl::RMControl r = rm;
        if (r.averaging && r.avg_start <= 0) r.avg_start = std::max(1, nbIterations / 2);
        return r;
    }
};

// ===========================================================================
// Results
// ===========================================================================
/** @brief Result of a fit. */
struct Results {
    // ---- fixed effects ----
    Vec        theta;         // full optimizer parameter vector (see nlme.hpp)
    std::vector<Vec> thetaTrace;  // theta at every SAEM iteration (for convergence plots)
    std::vector<std::string> coefNames;   // name of each coefficient in `coef`
    Vec        beta;          // intercept of each parameter, phi scale (len nPhi)
    Vec        psiPop;        // h(beta): population value on the natural scale
    Vec        coef;          // ALL coefficients (nPhi intercepts + covariates)
    Mat        Omega;         // covariance (nPhi x nPhi; 0 for parameters without a random effect)
    Vec        sd;            // sqrt(diag Omega)
    double     a = 0.0, b = 0.0, c = 1.0;   // error of response 0
    std::vector<resmod::Pres> errPres;       // error of each response
    Vec        extra;         // extra parameters (custom likelihood)
    ErrorModel error = ErrorModel::Proportional;

    // ---- multilevel ----
    bool        hierarchical = false;
    Mat         Psi;
    Vec         sdCluster;
    int         nClusters = 0;
    std::vector<int> subjectCluster;
    std::vector<Vec> xi;

    // ---- individual effects (phi scale, nPhi) ----
    //   phi   : MAP;  ebe : conditional MEAN E[phi_i | y_i, theta_hat];  phiCondSd : conditional SD
    std::vector<Vec> phi, ebe, phiCondSd;
    std::vector<Vec> psiEbe;                 // h(ebe): natural scale
    std::vector<Vec> xiCondMean, xiCondSd;
    bool   condComputed = false;
    double condEssMin   = 0.0;
    double condRhatMax  = std::numeric_limits<double>::quiet_NaN();

    // ---- likelihood ----
    double logLik = 0.0;      // according to llMethodUsed
    bool   llComputed = false;
    std::string llMethodUsed;
    double logLikLin = std::numeric_limits<double>::quiet_NaN();   // by linearization (FOCE)

    // ---- standard errors ----
    Vec    se;                // SE of beta (intercept, phi scale)   [compat]
    bool   seComputed = false;
    std::string seMethodUsed;
    Vec    seCoef;            // all coefficients
    Vec    sePsiPop;          // delta: |h'(beta)| * se
    Vec    seSd;              // SE of the random-effect sds
    Vec    seSdCluster;       // SE of the cluster-effect sds (hierarchical)
    Mat    Corr, seOmega, seCorr;   // correlations and SE of the Omega elements and of the correlations
    std::vector<double> seErr;   // SE of the error parameters (a, b on the natural scale; c raw)
    Vec    seExtra;
    Mat    covTheta;          // covariance of theta (all coordinates; 0 for fixed ones)

    // ---- diagnostics ----
    mcmckl::ConvergenceInfo convergence;
    std::vector<std::string> warnings;

    int nSubjects = 0, nObs = 0;

    // ---- data/model and per-observation predictions (Data order) ----
    Data data;
    Model model;
    Control control;            // options used (for smx::simulate)
    std::vector<double> fittedPop;   // f(psi_pop(cov)) : population prediction
    std::vector<double> fittedInd;   // f(psi_i(EBE))   : individual prediction

    std::vector<double> fitted() const { return fittedPop; }
    std::vector<double> residuals() const {
        std::vector<double> r(uz(data.nObs()));
        for (int i = 0; i < data.nObs(); ++i) r[uz(i)] = data.y[uz(i)] - fittedPop[uz(i)];
        return r;
    }
    std::vector<double> residualsInd() const {
        std::vector<double> r(uz(data.nObs()));
        for (int i = 0; i < data.nObs(); ++i) r[uz(i)] = data.y[uz(i)] - fittedInd[uz(i)];
        return r;
    }

    void print(std::ostream& os = std::cout) const {
        auto nm = [&](int j) { return j < toi(model.paramNames.size()) ? model.paramNames[uz(j)] : "p" + std::to_string(j); };
        os << std::fixed << std::setprecision(4);
        os << "---- SAEM Results ----\n";
        os << "subjects: " << nSubjects << "   observations: " << nObs << "\n\n";
        for (const auto& w : warnings) os << "WARNING: " << w << "\n";
        os << "Fixed effects (phi scale | natural scale):\n";
        for (int j = 0; j < beta.size(); ++j) {
            os << "  " << std::setw(8) << nm(j) << " = " << std::setw(10) << beta[j];
            if (seComputed && j < seCoef.size() && std::isfinite(seCoef[j])) os << " (SE " << seCoef[j] << ")";
            if (j < psiPop.size()) os << "   psi = " << psiPop[j];
            os << "\n";
        }
        for (int ci = toi(beta.size()); ci < coef.size(); ++ci) {
            os << "  " << std::setw(14) << (ci < toi(coefNames.size()) ? coefNames[uz(ci)] : "cov" + std::to_string(ci)) << " = " << std::setw(10) << coef[ci];
            if (seComputed && ci < seCoef.size() && std::isfinite(seCoef[ci])) os << " (SE " << seCoef[ci] << ")";
            os << "\n";
        }
        os << "\nRandom effects (sd):\n";
        for (int j = 0; j < sd.size(); ++j) {
            os << "  " << std::setw(8) << nm(j) << " = " << std::setw(10) << sd[j];
            if (seComputed && j < seSd.size() && std::isfinite(seSd[j])) os << " (SE " << seSd[j] << ")";
            os << "\n";
        }
        if (hierarchical) os << "CLUSTER random effects (sd): " << sdCluster.transpose() << "   [" << nClusters << " clusters]\n";
        for (size_t r = 0; r < errPres.size(); ++r)
            os << "Error resp. " << r << ": a=" << errPres[r].a << "  b=" << errPres[r].b << "  c=" << errPres[r].c << "\n";
        if (extra.size()) os << "Extra parameters: " << extra.transpose() << "\n";
        if (llComputed) os << "log-likelihood (" << llMethodUsed << "): " << logLik << "\n";
        if (std::isfinite(logLikLin)) os << "log-likelihood (linearization): " << logLikLin << "\n";
        if (seComputed) os << "SE by: " << seMethodUsed << "\n";
        os << "convergence: " << convergence.iterations << " it"
           << (convergence.stoppedEarly ? " (automatic stop)" : "") << ", max drift (last 25%) = " << convergence.maxDrift << "\n";
        if (condComputed) os << "EBE: min ESS " << condEssMin << ", max R-hat " << condRhatMax << "\n";
        os << "-------------------------\n";
    }
};

// ===========================================================================
// Internal fitting utilities
// ===========================================================================
namespace detail {

inline std::vector<Transform> transformsOf(const Model& m) {
    std::vector<Transform> t = m.transform;
    if (t.empty()) t.assign(uz(m.nPhi), Transform::Identity);
    if (static_cast<int>(t.size()) != m.nPhi) throw std::invalid_argument("Model.transform has the wrong dimension");
    return t;
}

// psi -> phi with a clear message if the value is outside the transformation domain
inline double toPhi(double psi, Transform t, int j) {
    try { return mcmckl::transphi(psi, t); }
    catch (const std::domain_error& e) {
        throw std::invalid_argument("beta0[" + std::to_string(j) + "] = " + std::to_string(psi) +
                                    " is outside the transformation domain (" + e.what() + ")");
    }
}

inline double medianOf(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Information from individual fits (for automatic initial values)
struct IndivFits {
    std::vector<Vec> phi;            // per subject (empty if it failed)
    std::vector<char> ok;
    int nOk = 0;
};

// Individual ML fit of each subject (no prior) with Nelder-Mead over phi, with the
// error parameters at their provisional value. Several deterministic starts.
inline IndivFits individualFits(const mcmckl::ObsLik& obs, const std::vector<Transform>& tr,
                                const Vec& phi0, const mcmckl::ErrParams& ep0, int nthreads) {
    const int ns = obs.nSubjects(), p = static_cast<int>(phi0.size());
    IndivFits F; F.phi.assign(uz(ns), Vec()); F.ok.assign(uz(ns), 0);
    mcmckl::parallelFor(ns, nthreads, [&](int i) {
        auto neg = [&](const Vec& phi) {
            Vec psi(p);
            for (int j = 0; j < p; ++j) psi[j] = mcmckl::transpsi(phi[j], tr[uz(j)]);
            const double ll = obs.logLik(i, psi, ep0);
            return std::isfinite(ll) ? -ll : std::numeric_limits<double>::infinity();
        };
        std::mt19937 rng(777u + 31u * unsigned(i));
        std::normal_distribution<double> nz(0.0, 1.0);
        Vec sc(p);
        for (int j = 0; j < p; ++j) sc[j] = 0.3 * std::max(std::abs(phi0[j]), 0.5);
        double best = std::numeric_limits<double>::infinity();
        Vec bx;
        for (int start = 0; start < 5; ++start) {
            Vec x0 = phi0;
            if (start > 0) for (int j = 0; j < p; ++j) x0[j] += (start == 1 ? 0.0 : 1.0) * sc[j] * nz(rng) * (start == 1 ? 1.0 : 2.0);
            if (start == 1) for (int j = 0; j < p; ++j) x0[j] = phi0[j] * 1.5 + 0.1;
            try {
                const mcmckl::NMOutcome o = mcmckl::nelderMeadEx(neg, x0, sc, 800, 1e-9, 0);
                if (o.fun < best) { best = o.fun; bx = o.x; }
                if (o.converged && std::isfinite(o.fun) && start >= 0) break;
            } catch (const std::exception&) { /* next start */ }
        }
        if (std::isfinite(best)) { F.phi[uz(i)] = bx; F.ok[uz(i)] = 1; }
    });
    for (char c : F.ok) F.nOk += (c != 0);
    return F;
}

// Split the posterior samples into two halves (even / odd indices)
template <class Draws>
inline Draws splitDraws(const Draws& d, int part) {
    Draws out(d.size());
    for (size_t i = 0; i < d.size(); ++i)
        for (size_t s = 0; s < d[i].size(); ++s) if (static_cast<int>(s % 2) == part) out[i].push_back(d[i][s]);
    return out;
}

// Louis FIM estimated from the two sample halves: returns the mean and measures the Monte
// Carlo NOISE as the maximum relative difference between the diagonals of both halves (free coordinates).
// Louis subtracts two large terms (complete information - missing information) and is noisy when
// most of the information is "lost" (e.g. fixed effects with few clusters).
template <class Model, class Draws>
inline mcmckl::NlmeModel::FimResult louisChecked(const Model& model, const Vec& th, const Draws& draws,
                                                 const std::vector<char>& fixedMask, double& noise) {
    auto f1 = model.louisFIM(th, splitDraws(draws, 0));
    auto f2 = model.louisFIM(th, splitDraws(draws, 1));
    mcmckl::NlmeModel::FimResult out;
    out.ok = f1.ok && f2.ok;
    out.I = 0.5 * (f1.I + f2.I);
    noise = 0.0;
    for (int j = 0; j < out.I.rows(); ++j) {
        if (fixedMask[uz(j)]) continue;
        const double sc = std::max(std::abs(out.I(j, j)), 1e-12);
        noise = std::max(noise, std::abs(f1.I(j, j) - f2.I(j, j)) / sc);
    }
    return out;
}

// Robust inverse of the FIM restricted to the free coordinates
inline bool invertFIM(const Mat& I, const std::vector<char>& fixed, Mat& C, bool& wasSingular) {
    const int D = static_cast<int>(I.rows());
    std::vector<int> fr;
    for (int j = 0; j < D; ++j) if (!fixed[uz(j)]) fr.push_back(j);
    const int m = static_cast<int>(fr.size());
    Mat Ifr(m, m);
    for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) Ifr(a, b) = I(fr[uz(a)], fr[uz(b)]);
    if (!Ifr.allFinite()) return false;
    Eigen::SelfAdjointEigenSolver<Mat> es(0.5 * (Ifr + Ifr.transpose()));
    Vec d = es.eigenvalues();
    const double dmax = d.cwiseAbs().maxCoeff();
    const double fl = std::max(1e-12, 1e-9 * (dmax > 0 ? dmax : 1.0));
    wasSingular = d.minCoeff() <= fl;
    Vec dinv = d.cwiseMax(fl).cwiseInverse();
    const Mat inv = es.eigenvectors() * dinv.asDiagonal() * es.eigenvectors().transpose();
    C = Mat::Zero(D, D);
    for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) C(fr[uz(a)], fr[uz(b)]) = inv(a, b);
    return true;
}

} // namespace detail

// ===========================================================================
// Single-level model assembly (shared by the fit and by simulate)
// ===========================================================================
namespace detail {

// Expands the categorical covariates of Model.covariateModel into 0/1 variables per level
// (relative to the reference level) and returns the names of all coefficients.
inline void expandCategorical(Data& data, Model& model, std::vector<std::string>& coefNames) {
    std::vector<CovariateTerm> terms;
    const int ns = data.nSubjects();
    for (const auto& ct : model.covariateModel) {
        auto it = data.catCov.find(ct.covariate);
        if (it == data.catCov.end() || data.cov.count(ct.covariate)) { terms.push_back(ct); continue; }
        std::vector<std::string> levels = it->second;
        std::sort(levels.begin(), levels.end());
        levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
        std::string ref = levels.front();
        auto rt = data.catReference.find(ct.covariate);
        if (rt != data.catReference.end()) {
            ref = rt->second;
            if (std::find(levels.begin(), levels.end(), ref) == levels.end())
                throw std::invalid_argument("catReference['" + ct.covariate + "'] = '" + ref + "' is not a level");
        }
        for (const auto& lv : levels) {
            if (lv == ref) continue;
            const std::string nm = ct.covariate + "=" + lv;
            if (!data.cov.count(nm)) {
                Vec x(ns);
                for (int k = 0; k < ns; ++k) x[k] = (it->second[uz(k)] == lv) ? 1.0 : 0.0;
                data.cov[nm] = x;
                if (!data.covOrder.empty()) data.covOrder.push_back(nm);
            }
            terms.push_back({ct.param, nm});
        }
    }
    model.covariateModel = terms;
    coefNames.clear();
    for (int j = 0; j < model.nPhi; ++j) coefNames.push_back(j < toi(model.paramNames.size()) ? model.paramNames[uz(j)] : "p" + std::to_string(j));
    for (const auto& t : terms)
        coefNames.push_back((t.param < toi(model.paramNames.size()) ? model.paramNames[uz(t.param)] : "p" + std::to_string(t.param)) + "~" + t.covariate);
}

struct Assembled {
    std::unique_ptr<mcmckl::NlmeModel> nl;               // single-level model
    std::unique_ptr<mcmckl::NlmeMultilevel> ml;          // hierarchical model (contains its NlmeModel)
    mcmckl::NlmeModel& core() { return ml ? ml->core() : *nl; }
    std::vector<int> rowSubj, rowIdx;
    std::vector<Transform> trs;
    std::vector<ErrorModel> ems;
    int R = 0;
    bool custom = false;
};

// Builds ObsLik + NlmeModel. beta0Phi (len nPhi, phi scale) and sd0 (len nPhi) are the initial
// values; errInit is one per response. `data` and `model` must already have the categoricals expanded.
inline Assembled assemble(const Data& data, const Model& model, const Control& control,
                          const std::vector<resmod::Pres>& errInit, const Vec& beta0Phi, const Vec& sd0,
                          const std::vector<int>* cluster = nullptr) {
    using namespace mcmckl;
    Assembled A;
    const int p = model.nPhi;
    A.trs = transformsOf(model);
    const std::vector<int> sids = data.subjectIds();
    const int ns = static_cast<int>(sids.size());
    std::map<int,int> sid2k;
    for (int k = 0; k < ns; ++k) sid2k[sids[uz(k)]] = k;
    const int P = static_cast<int>(data.predictorOrder.size());
    std::vector<ObsSubject> subj(uz(ns));
    for (auto& s : subj) s.stride = P + 2;
    A.rowSubj.resize(uz(data.nObs())); A.rowIdx.resize(uz(data.nObs()));
    for (int r = 0; r < data.nObs(); ++r) {
        const int k = sid2k[data.id[uz(r)]];
        ObsSubject& s = subj[uz(k)];
        A.rowSubj[uz(r)] = k; A.rowIdx[uz(r)] = s.n();
        s.x.push_back(data.time[uz(r)]);
        for (const auto& nm : data.predictorOrder) s.x.push_back(data.predictors.at(nm)[uz(r)]);
        s.x.push_back(data.ytype.empty() ? 0.0 : double(data.ytype[uz(r)]));
        s.y.push_back(data.y[uz(r)]);
        s.ytype.push_back(data.ytype.empty() ? 0 : data.ytype[uz(r)]);
        s.cens.push_back(data.cens.empty() ? 0 : data.cens[uz(r)]);
        s.lim.push_back(data.limit.empty() ? std::numeric_limits<double>::quiet_NaN() : data.limit[uz(r)]);
    }
    for (int k = 0; k < ns; ++k) subj[uz(k)].doses = data.dosesOf(sids[uz(k)]);
    std::vector<Vec> subjCov(uz(ns));
    for (int k = 0; k < ns; ++k) subjCov[uz(k)] = data.covariatesOf(k);

    ObsLik::Spec ospec;
    ospec.predictSubject = model.predictSubject;
    if (model.predictRow) ospec.predict = model.predictRow;
    else if (model.predict) ospec.predict = [f = model.predict](const Vec& psi, std::span<const double> x, const Vec& cov) { return f(psi, x[0], cov); };
    else ospec.predict = [](const Vec&, std::span<const double>, const Vec&) { return 0.0; };
    ospec.logLik = model.logLikObs;
    ospec.nExtra = model.nExtra;
    ospec.extra0 = control.extra0;
    ospec.extraFixed = control.fixedExtra;
    A.custom = static_cast<bool>(model.logLikObs);
    A.ems = model.errors.empty() ? std::vector<ErrorModel>{model.error} : model.errors;
    A.R = A.custom ? 0 : static_cast<int>(A.ems.size());
    size_t flat = 0;
    for (int r = 0; r < A.R; ++r) {
        ErrorSpec es; es.type = toResid(A.ems[uz(r)]); es.init = errInit[uz(r)];
        const int np = resmod::nParams(es.type);
        if (!control.fixedError.empty()) {
            es.fixed.assign(uz(np), 0);
            for (int k = 0; k < np; ++k) { if (flat < control.fixedError.size()) es.fixed[uz(k)] = control.fixedError[flat]; ++flat; }
        }
        ospec.errors.push_back(es);
    }
    ObsLik obs(ospec, subj, subjCov);

    NlmeModel::Spec nspec;
    nspec.dimPhi = p;
    nspec.transform = A.trs;
    nspec.randomEffect = model.randomEffect.empty() ? std::vector<char>(uz(p), 1) : model.randomEffect;
    nspec.omegaBlock = model.omegaBlock;
    nspec.covPattern = model.covariancePattern;
    for (const auto& ct : model.covariateModel) nspec.covTerms.push_back({ct.param, data.covIndex(ct.covariate)});
    const int T = static_cast<int>(nspec.covTerms.size());
    const int nb = p + T;
    nspec.beta0 = Vec::Zero(nb);
    nspec.beta0.head(p) = beta0Phi;
    if (control.betaCov0.size() == T) nspec.beta0.tail(T) = control.betaCov0;
    nspec.sd0 = sd0;
    nspec.betaFixed.assign(uz(nb), 0);
    if (static_cast<int>(control.fixedBeta.size()) == p)  for (int j = 0; j < p; ++j) nspec.betaFixed[uz(j)] = control.fixedBeta[uz(j)];
    if (static_cast<int>(control.fixedBeta.size()) == nb) nspec.betaFixed = control.fixedBeta;
    nspec.sdFixed = control.fixedSd.empty() ? std::vector<char>(uz(p), 0) : control.fixedSd;
    if (static_cast<int>(nspec.sdFixed.size()) != p) throw std::invalid_argument("Control.fixedSd has the wrong dimension");
    if (cluster) {
        NlmeMultilevel::Spec ms; ms.core = nspec; ms.psiBlock = model.psiBlock;
        if (control.sdCluster0.size() == p) ms.sdPsi0 = control.sdCluster0;
        A.ml = std::make_unique<NlmeMultilevel>(ms, std::move(obs), *cluster);
        A.ml->threads = control.nThreads; A.ml->core().threads = control.nThreads;
    } else {
        A.nl = std::make_unique<NlmeModel>(nspec, std::move(obs));
        A.nl->threads = control.nThreads;
    }
    return A;
}

} // namespace detail

namespace detail {

struct Prep {
    Data data; Model model;
    std::vector<std::string> coefNames, initWarn;
    int p = 0, T = 0, nb = 0, ns = 0, R = 0;
    std::vector<Transform> trs;
    std::vector<ErrorModel> ems;
    bool custom = false;
    std::vector<resmod::Pres> errInit;
    Vec beta0, sd0;
};

// Validation, categoricals, and automatic or user-provided initial values (beta, sd, error)
inline Prep prepare(const Data& data_in, const Model& model_in, const Control& control) {
    using namespace mcmckl;
    Prep P;
    P.data = data_in; P.model = model_in;
    Data& data = P.data; Model& model = P.model;
    data.check();
    if (model.nPhi <= 0) throw std::invalid_argument("nPhi > 0");
    if (!model.predict && !model.predictRow && !model.predictSubject && !model.logLikObs)
        throw std::invalid_argument("Model.predict, predictRow, predictSubject or logLikObs is required");
    std::vector<std::string>& coefNames = P.coefNames;
    detail::expandCategorical(data, model, coefNames);
    const int p = model.nPhi;
    const int T = static_cast<int>(model.covariateModel.size());
    const int nb = p + T;
    const std::vector<Transform> trs = detail::transformsOf(model);
    const int ns = data.nSubjects();
    const bool custom = static_cast<bool>(model.logLikObs);
    std::vector<ErrorModel> ems = model.errors.empty() ? std::vector<ErrorModel>{model.error} : model.errors;
    const int R = custom ? 0 : static_cast<int>(ems.size());

    // ---------- initial error values ----------
    std::vector<resmod::Pres> errInit(uz(R), resmod::Pres(0.1, 0.1, control.c0));
    bool errAuto = false;
    if (static_cast<int>(control.errorInit.size()) == R && R > 0) errInit = control.errorInit;
    else if (R > 0) {
        if (control.a0 > 0) errInit[0].a = control.a0; else errAuto = true;
        if (control.b0 > 0) errInit[0].b = control.b0; else errAuto = true;
    }

    // ---------- initial values of beta and sd ----------
    Vec phi0(p);
    const bool haveBeta = (control.beta0.size() == p);
    for (int j = 0; j < p; ++j) {
        const double psi0 = haveBeta ? control.beta0[j]
                          : ((trs[uz(j)] == Transform::Logit || trs[uz(j)] == Transform::Probit) ? 0.5 : 1.0);
        phi0[j] = detail::toPhi(psi0, trs[uz(j)], j);
    }
    Vec beta0 = phi0;
    Vec sd0 = control.sd0;
    std::vector<std::string>& initWarn = P.initWarn;
    const bool needSd = (sd0.size() != p);
    if (control.autoInit && (!haveBeta || needSd || errAuto) && !custom) {
        detail::Assembled prov = detail::assemble(data, model, control, errInit, phi0, Vec::Ones(p));
        const ObsLik& obsProv = prov.nl->obs();
        ErrParams ep0 = obsProv.decode(obsProv.initialErr(), 0);
        detail::IndivFits F = detail::individualFits(obsProv, trs, phi0, ep0, control.nThreads);
        if (F.nOk >= std::max(2, ns / 2)) {
            Vec med(p), mad(p);
            for (int j = 0; j < p; ++j) {
                std::vector<double> v;
                for (int i = 0; i < ns; ++i) if (F.ok[uz(i)]) v.push_back(F.phi[uz(i)][j]);
                med[j] = detail::medianOf(v);
                for (double& x : v) x = std::abs(x - med[j]);
                mad[j] = 1.4826 * detail::medianOf(v);
            }
            if (!haveBeta) beta0 = med;
            if (needSd) { sd0.resize(p); for (int j = 0; j < p; ++j) sd0[j] = std::max(mad[j], 0.05 * std::max(std::abs(med[j]), 0.1)); }
            if (errAuto) {
                std::vector<double> s2(uz(R), 0.0), r2(uz(R), 0.0); std::vector<long> cnt(uz(R), 0);
                for (int i = 0; i < ns; ++i) if (F.ok[uz(i)]) {
                    Vec psi(p);
                    for (int j = 0; j < p; ++j) psi[j] = transpsi(F.phi[uz(i)][j], trs[uz(j)]);
                    const auto& S = obsProv.subject(i);
                    Vec fall; obsProv.predictAll(i, psi, fall);
                    for (int j = 0; j < S.n(); ++j) {
                        if (S.cens[uz(j)] != 0) continue;
                        const double f = fall[j];
                        if (!std::isfinite(f)) continue;
                        const int r = S.ytype[uz(j)];
                        const bool ex = (ems[uz(r)] == ErrorModel::Exponential);
                        if (ex && !(f > 0.0)) continue;
                        const double res = ex ? std::log(S.y[uz(j)]) - std::log(f) : S.y[uz(j)] - f;
                        s2[uz(r)] += res * res; r2[uz(r)] += res * res / std::max(f * f, 1e-12); ++cnt[uz(r)];
                    }
                }
                for (int r = 0; r < R; ++r) {
                    if (cnt[uz(r)] < 3) continue;
                    const double dfc = std::sqrt(double(cnt[uz(r)]) / std::max(1.0, double(cnt[uz(r)]) - double(ns) * p * double(cnt[uz(r)]) / double(std::max(1L, obsProv.nObs()))));
                    const double a = std::sqrt(s2[uz(r)] / tod(cnt[uz(r)])) * std::min(dfc, 3.0), b = std::sqrt(r2[uz(r)] / tod(cnt[uz(r)])) * std::min(dfc, 3.0);
                    const resmod::ErrorKind ty = toResid(ems[uz(r)]);
                    resmod::Pres pr = errInit[uz(r)];
                    const bool userA = (r == 0 && control.a0 > 0), userB = (r == 0 && control.b0 > 0);
                    switch (ty) {
                        case resmod::ErrorKind::Constant:
                        case resmod::ErrorKind::Exponential:  if (!userA) pr.a = std::max(a, 1e-6); break;
                        case resmod::ErrorKind::Proportional: if (!userB) pr.b = std::max(b, 1e-6); break;
                        default: if (!userA) pr.a = std::max(0.5 * a, 1e-6); if (!userB) pr.b = std::max(0.5 * b, 1e-6); break;
                    }
                    errInit[uz(r)] = pr;
                }
            }
        } else {
            initWarn.push_back("automatic initialization: too many individual fits failed; using default values (provide beta0)");
        }
    }
    if (sd0.size() != p) {
        sd0.resize(p);
        for (int j = 0; j < p; ++j) sd0[j] = (trs[uz(j)] == Transform::Identity) ? std::max(0.3 * std::abs(beta0[j]), 0.1) : 0.5;
    }

    P.p = p; P.T = T; P.nb = nb; P.ns = ns; P.R = R; P.trs = trs; P.ems = ems; P.custom = custom;
    P.errInit = errInit; P.beta0 = beta0; P.sd0 = sd0;
    return P;
}

} // namespace detail

// ===========================================================================
// Core: SINGLE-level fit (NlmeModel)
// ===========================================================================
inline Results fitSingleLevel(const Data& data_in, const Model& model_in, const Control& control) {
    using namespace mcmckl;
    detail::Prep P = detail::prepare(data_in, model_in, control);
    Data& data = P.data; Model& model = P.model;
    std::vector<std::string>& coefNames = P.coefNames;
    const int p = P.p, T = P.T, nb = P.nb, ns = P.ns;
    const std::vector<Transform>& trs = P.trs;
    const bool custom = P.custom;
    (void)T;
    const std::vector<resmod::Pres>& errInit = P.errInit;
    const Vec& beta0 = P.beta0; const Vec& sd0 = P.sd0;
    const std::vector<std::string>& initWarn = P.initWarn;
    detail::Assembled A = detail::assemble(data, model, control, errInit, beta0, sd0);
    NlmeModel& nl = *A.nl;
    const std::vector<int>& rowSubj = A.rowSubj;
    const std::vector<int>& rowIdx = A.rowIdx;
    const int q = nl.q(), D = nl.dimTheta();

    // ---------- SAEM ----------
    SAEMControl sc;
    sc.n_saem_iter = control.nbIterations; sc.n_mcmc_burnin = control.nbBurnIn; sc.n_mcmc_sweeps = control.nMCMCsweeps;
    sc.verbose = control.verbose; sc.print_every = control.printEvery;
    sc.rm = control.effectiveRM(); sc.mstep = control.mstep; sc.step = control.step;
    sc.simulated_annealing = control.simulatedAnnealing; sc.sa = control.sa;
    sc.n_chains = control.nChains; sc.adapt_rw = control.adaptRW; sc.n_threads = control.nThreads;
    sc.map_kernel = control.mapKernel; sc.map_every = control.mapEvery;
    sc.auto_stop = control.autoStop; sc.stop_tol = control.stopTol;
    sc.mcmc.rw_sigma = Vec::Constant(q, 0.5);
    SAEM eng(nl, sc, control.seed);
    eng.run();
    const Vec th = eng.thetaAveraged();

    // ---------- method for SEs (decided here: Louis needs posterior samples) ----------
    SEMethod sem = control.seMethod;
    if (!control.computeSE) sem = SEMethod::None;
    const bool cens = nl.obs().hasCensoring();
    if (sem == SEMethod::Auto) sem = (!custom && !cens) ? SEMethod::Linearization : SEMethod::Laplace;
    if (sem == SEMethod::Linearization && custom) sem = SEMethod::Laplace;

    // ---------- individuals: MAP and conditional mean ----------
    LogLikModel lm;
    lm.dimChi = q; lm.nSubjects = ns;
    lm.logJoint = [&](int i, const Vec& chi, const Vec& t) { return nl.logJoint(i, chi, t); };
    lm.priorParams = [&](const Vec& t, Vec& mu, Mat& Om) { mu = Vec::Zero(q); Om = nl.omegaOf(t); };
    std::vector<Vec> etaMap(uz(ns), Vec::Zero(q));
    parallelFor(ns, control.nThreads, [&](int i) {
        try { etaMap[uz(i)] = mapEstimate(lm, th, i, Vec::Zero(q), 1000); } catch (const std::exception&) {}
    });
    Results res;
    std::vector<Vec> etaEbe = etaMap;
    std::vector<std::vector<Vec>> louisDraws;
    res.convergence = eng.convergence();
    if (control.computeCond || sem == SEMethod::Louis) {
        auto ce = eng.conditionalSample(th, std::max(control.nCondSamples, sem == SEMethod::Louis ? 2 * control.louisDraws : 0),
                                        control.condBurnIn, control.condThin, sem == SEMethod::Louis ? control.louisDraws : 0);
        etaEbe = ce.mean;
        louisDraws = std::move(ce.draws);
        res.condComputed = true; res.condEssMin = ce.ess_min; res.condRhatMax = ce.rhat_max;
        res.phiCondSd.assign(uz(ns), Vec::Zero(p));
        for (int i = 0; i < ns; ++i) for (int l = 0; l < q; ++l) res.phiCondSd[uz(i)][nl.reIdx()[uz(l)]] = ce.sd[uz(i)][l];
    }

    // ---------- package ----------
    res.data = data; res.model = model; res.control = control;
    res.nSubjects = ns; res.nObs = data.nObs();
    res.error = model.error;
    res.theta = th; res.thetaTrace = eng.thetaTrace(); res.coefNames = coefNames;
    res.beta = th.head(p); res.coef = th.head(nb);
    res.psiPop.resize(p);
    for (int j = 0; j < p; ++j) res.psiPop[j] = transpsi(res.beta[j], trs[uz(j)]);
    res.Omega = nl.omegaFull(th);
    res.sd = res.Omega.diagonal().cwiseSqrt();
    res.Corr = Mat::Zero(p, p);
    for (int a = 0; a < p; ++a) for (int b = 0; b < p; ++b)
        res.Corr(a, b) = (a == b) ? 1.0 : ((res.sd[a] > 0 && res.sd[b] > 0) ? res.Omega(a, b) / (res.sd[a] * res.sd[b]) : 0.0);
    {
        const ErrParams ep = nl.obs().decode(th, nl.offError());
        res.errPres = ep.pres; res.extra = ep.extra;
        if (!ep.pres.empty()) { res.a = ep.pres[0].a; res.b = ep.pres[0].b; res.c = ep.pres[0].c; }
    }
    res.phi.resize(uz(ns)); res.ebe.resize(uz(ns)); res.psiEbe.resize(uz(ns));
    Vec phiTmp;
    for (int i = 0; i < ns; ++i) {
        nl.phiOf(i, th, etaMap[uz(i)], phiTmp); res.phi[uz(i)] = phiTmp;
        nl.phiOf(i, th, etaEbe[uz(i)], phiTmp); res.ebe[uz(i)] = phiTmp; res.psiEbe[uz(i)] = nl.psiOf(phiTmp);
    }
    res.fittedPop.assign(uz(data.nObs()), std::numeric_limits<double>::quiet_NaN());
    res.fittedInd = res.fittedPop;
    if (!custom) {
        std::vector<Vec> fPop(uz(ns)), fInd(uz(ns));
        for (int i = 0; i < ns; ++i) {
            nl.phiOf(i, th, Vec::Zero(q), phiTmp);
            nl.obs().predictAll(i, nl.psiOf(phiTmp), fPop[uz(i)]);
            nl.obs().predictAll(i, res.psiEbe[uz(i)], fInd[uz(i)]);
        }
        for (int r = 0; r < data.nObs(); ++r) {
            res.fittedPop[uz(r)] = fPop[uz(rowSubj[uz(r)])][rowIdx[uz(r)]];
            res.fittedInd[uz(r)] = fInd[uz(rowSubj[uz(r)])][rowIdx[uz(r)]];
        }
    }

    for (int j = 0; j < p; ++j)
        if (nl.reOf()[uz(j)] >= 0 && res.sd[j] < 1e-3 * std::max(std::abs(res.beta[j]), 1e-8))
            res.warnings.push_back("sd(p" + std::to_string(j) + ") ~ 0 relative to |beta|: no real variability or SAEM collapse "
                                   "(try nChains >= 2, simulatedAnnealing, more iterations)");
    for (const auto& w : initWarn) res.warnings.push_back(w);
    if (res.convergence.iterations >= 8 && res.convergence.maxDrift > 0.05)
        res.warnings.push_back("theta still drifts in the last 25% of the iterations (max drift " +
                               std::to_string(res.convergence.maxDrift) + "): consider more iterations");
    if (res.condComputed && std::isfinite(res.condRhatMax) && res.condRhatMax > 1.1)
        res.warnings.push_back("conditional-sampling R-hat > 1.1: use more conditional samples or more chains");

    // Variance parameters on the boundary: excluded from the FIM (their SEs stay NaN).
    {
        const std::vector<char> bm = nl.boundaryMask(th);
        bool any = false; for (char b : bm) any = any || b;
        if (any && sem != SEMethod::None) {
            nl.markFixed(bm);
            res.warnings.push_back("variance(s) on the boundary (sd ~ 0): excluded from the FIM; their standard errors are undefined");
        }
    }

    // ---------- likelihood ----------
    const bool linOK = !custom;
    NlmeModel::FimResult fr;
    bool frDone = false;
    auto doLin = [&]() { if (!frDone) { fr = nl.linearizedFIM(th, etaEbe); frDone = true; } };
    if (linOK) { doLin(); if (fr.ok) res.logLikLin = fr.ll; }

    LLMethod llm = control.llMethod;
    if (llm == LLMethod::Auto) llm = (q <= 3) ? LLMethod::GaussHermite : LLMethod::ImportanceSampling;
    if (llm == LLMethod::GaussHermite || llm == LLMethod::ImportanceSampling) {
        std::vector<double> lls(uz(ns), 0.0);
        parallelFor(ns, control.nThreads, [&](int i) {
            try {
                lls[uz(i)] = (llm == LLMethod::GaussHermite) ? llgqSubjectAdaptive(lm, th, i, control.gqNodes)
                                                         : llisSubject(lm, th, i, control.isSamples, control.seed);
            } catch (const std::exception&) { lls[uz(i)] = std::numeric_limits<double>::quiet_NaN(); }
        });
        double s = 0.0; for (double v : lls) s += v;
        res.logLik = s; res.llComputed = std::isfinite(s);
        res.llMethodUsed = (llm == LLMethod::GaussHermite) ? "adaptive Gauss-Hermite" : "importance sampling";
    } else if (llm == LLMethod::Linearization && std::isfinite(res.logLikLin)) {
        res.logLik = res.logLikLin; res.llComputed = true; res.llMethodUsed = "linearization";
    } else if (llm == LLMethod::Laplace) {
        std::vector<Vec> modes = etaEbe;
        res.logLik = nl.laplaceLogLik(th, modes);
        res.llComputed = std::isfinite(res.logLik); res.llMethodUsed = "Laplace";
    }

    // ---------- standard errors ----------
    if (sem != SEMethod::None) {
        Mat I; bool ok = false;
        if (sem == SEMethod::Linearization) {
            doLin(); I = fr.I; ok = fr.ok;
            if (!fr.ok) res.warnings.push_back("linearized FIM: some subjects could not be linearized (V not positive definite)");
            res.seMethodUsed = "linearized FIM (FOCE with interaction)";
        } else if (sem == SEMethod::Laplace) {
            auto lf = nl.laplaceFIM(th, etaEbe);
            I = lf.I; ok = lf.ok && lf.I.allFinite();
            if (!lf.ok) res.warnings.push_back("Laplace: could not compute some subject's contribution to the FIM");
            res.seMethodUsed = "observed FIM (numerical Hessian of the Laplace logLik, deterministic)";
        } else if (sem == SEMethod::Louis) {
            double noise = 0.0;
            auto lf = detail::louisChecked(nl, th, louisDraws, nl.fixedMask(), noise);
            I = lf.I; ok = lf.ok && lf.I.allFinite();
            if (!lf.ok) res.warnings.push_back("Louis' formula: some subjects have no valid posterior samples");
            if (noise > 0.5) res.warnings.push_back("noisy Louis' formula (relative difference " + std::to_string(noise) +
                                                    " between sample halves): increase louisDraws");
            res.seMethodUsed = "observed FIM (Louis' formula, " + std::to_string(control.louisDraws) + " samples/subject)";
        } else {
            IndivLogLik ll = [&](int i, const Vec& t) { return llgqSubjectAdaptive(lm, t, i, control.gqNodes); };
            I = fimObserved(ll, ns, th, 1e-3); ok = I.allFinite();
            res.seMethodUsed = "observed FIM (adaptive Gauss-Hermite)";
        }
        Mat C; bool singular = false;
        if (ok && detail::invertFIM(I, nl.fixedMask(), C, singular)) {
            if (singular) res.warnings.push_back("nearly singular FIM: some SEs are unreliable (poorly identifiable parameters)");
            const double nan = std::numeric_limits<double>::quiet_NaN();
            auto seOf = [&](int j) { return nl.fixedMask()[uz(j)] ? nan : (C(j, j) > 0 ? std::sqrt(C(j, j)) : nan); };
            res.covTheta = C;
            res.seCoef.resize(nb);
            for (int c = 0; c < nb; ++c) res.seCoef[c] = seOf(c);
            res.se = res.seCoef.head(p);
            res.sePsiPop.resize(p);
            for (int j = 0; j < p; ++j) res.sePsiPop[j] = std::abs(derivpsi(res.beta[j], trs[uz(j)])) * res.se[j];
            // delta method with a finite-difference Jacobian over the Omega parameters
            auto deltaSE = [&](auto&& fn) {
                Vec g = Vec::Zero(D);
                for (int k = nl.offOmega(); k < nl.offOmega() + nl.nOmega(); ++k) {
                    if (nl.fixedMask()[uz(k)]) continue;
                    const double h = 1e-6 * std::max(1.0, std::abs(th[k]));
                    Vec tp = th, tm = th; tp[k] += h; tm[k] -= h;
                    g[k] = (fn(tp) - fn(tm)) / (2.0 * h);
                }
                const double v = g.dot(C * g);
                return v >= 0 ? std::sqrt(v) : nan;
            };
            res.seSd = Vec::Constant(p, nan);
            res.seOmega = Mat::Constant(p, p, nan);
            res.seCorr = Mat::Constant(p, p, nan);
            for (int a = 0; a < p; ++a) {
                if (nl.reOf()[uz(a)] < 0) continue;
                res.seSd[a] = deltaSE([&](const Vec& t) { return std::sqrt(nl.omegaFull(t)(a, a)); });
                for (int b = a; b < p; ++b) {
                    if (nl.reOf()[uz(b)] < 0) continue;
                    const double so = deltaSE([&](const Vec& t) { return nl.omegaFull(t)(a, b); });
                    res.seOmega(a, b) = so; res.seOmega(b, a) = so;
                    if (a != b) {
                        const double sc2 = deltaSE([&](const Vec& t) { const Mat O = nl.omegaFull(t); return O(a, b) / std::sqrt(O(a, a) * O(b, b)); });
                        res.seCorr(a, b) = sc2; res.seCorr(b, a) = sc2;
                    }
                }
            }
            const ObsLik& ob = nl.obs();
            for (int r = 0; r < ob.nResp(); ++r)
                for (int k = 0; k < resmod::nParams(ob.errorSpec(r).type); ++k) {
                    const int idx = nl.offError() + ob.errOffset(r) + k;
                    const bool raw = (ob.errorSpec(r).type == resmod::ErrorKind::Power && k == 2);
                    const double se = seOf(idx);
                    res.seErr.push_back(raw ? se : se * std::exp(th[idx]));
                }
            res.seExtra = Vec::Constant(ob.nExtra(), nan);
            for (int k = 0; k < ob.nExtra(); ++k) res.seExtra[k] = seOf(nl.offError() + ob.extraOffset() + k);
            res.seComputed = true;
        } else res.warnings.push_back("could not invert the FIM: no standard errors");
    }
    return res;
}

// ===========================================================================
// HIERARCHICAL two-level fit (NlmeMultilevel): same capabilities as single level
// ===========================================================================
inline Results fitHierarchical(const Data& data_in, const Model& model_in, const Control& control) {
    using namespace mcmckl;
    detail::Prep P = detail::prepare(data_in, model_in, control);
    Data& data = P.data; Model& model = P.model;
    const int p = P.p, nb = P.nb, ns = P.ns;
    const std::vector<Transform>& trs = P.trs;
    const std::vector<int> grp = data.subjectGroup();
    detail::Assembled A = detail::assemble(data, model, control, P.errInit, P.beta0, P.sd0, &grp);
    NlmeMultilevel& ml = *A.ml;
    NlmeModel& nl = ml.core();
    const int q = nl.q(), D = ml.dimTheta();
    ml.innerNodes = control.gqInnerNodes; ml.outerNodes = control.gqNodes;

    MultilevelControl mc;
    mc.n_saem_iter = control.nbIterations; mc.n_mcmc_burnin = control.nbBurnIn; mc.n_mcmc_sweeps = control.nMCMCsweeps;
    mc.verbose = control.verbose; mc.print_every = control.printEvery;
    mc.rm = control.effectiveRM(); mc.mstep = control.mstep; mc.step = control.step;
    mc.simulated_annealing = control.simulatedAnnealing; mc.sa = control.sa;
    mc.n_chains = control.nChains; mc.adapt_rw = control.adaptRW; mc.n_threads = control.nThreads;
    mc.mcmc.rw_sigma = Vec::Constant(q, 0.5);
    SAEMMultilevel eng(ml, mc, control.seed);
    eng.run();
    const Vec th = eng.thetaAveraged();

    SEMethod sem = control.seMethod;
    if (!control.computeSE) sem = SEMethod::None;
    const bool hCens = nl.obs().hasCensoring();
    if (sem == SEMethod::Auto) sem = (!P.custom && !hCens) ? SEMethod::Linearization : SEMethod::Laplace;
    if (sem == SEMethod::GaussHermite) sem = SEMethod::Laplace;                // no FIM via nested GH
    if (sem == SEMethod::Linearization && P.custom) sem = SEMethod::Laplace;
    LLMethod llm = control.llMethod;
    if (llm == LLMethod::Auto) llm = (q <= 3) ? LLMethod::GaussHermite : LLMethod::ImportanceSampling;
    if (llm == LLMethod::Linearization && P.custom) llm = LLMethod::None;
    int maxMem = 0; for (int c = 0; c < ml.nClusters(); ++c) maxMem = std::max(maxMem, toi(ml.membersOf(c).size()));
    const int dJoint = q * (1 + maxMem);
    int keep = 0;
    if (sem == SEMethod::Louis) keep = std::max(control.louisDraws, 2 * dJoint + 10);
    if (llm == LLMethod::ImportanceSampling) keep = std::max(keep, std::max(control.louisDraws, 2 * dJoint + 10));
    auto ce = eng.conditionalSample(th, std::max(control.nCondSamples, 2 * keep),
                                    control.condBurnIn, control.condThin, keep);

    Results res;
    res.hierarchical = true;
    res.data = data; res.model = model; res.control = control;
    res.nSubjects = ns; res.nObs = data.nObs(); res.error = model.error;
    res.theta = th; res.thetaTrace = eng.thetaTrace(); res.coefNames = P.coefNames;
    res.beta = th.head(p); res.coef = th.head(nb);
    res.psiPop.resize(p);
    for (int j = 0; j < p; ++j) res.psiPop[j] = transpsi(res.beta[j], trs[uz(j)]);
    res.Omega = nl.omegaFull(th);
    res.sd = res.Omega.diagonal().cwiseSqrt();
    res.Corr = Mat::Zero(p, p);
    for (int a = 0; a < p; ++a) for (int b = 0; b < p; ++b)
        res.Corr(a, b) = (a == b) ? 1.0 : ((res.sd[a] > 0 && res.sd[b] > 0) ? res.Omega(a, b) / (res.sd[a] * res.sd[b]) : 0.0);
    {
        const Mat Ps = ml.psiOf(th);
        res.Psi = Mat::Zero(p, p);
        for (int a = 0; a < q; ++a) for (int b = 0; b < q; ++b) res.Psi(nl.reIdx()[uz(a)], nl.reIdx()[uz(b)]) = Ps(a, b);
        res.sdCluster = res.Psi.diagonal().cwiseSqrt();
        const ErrParams ep = nl.obs().decode(th, nl.offError());
        res.errPres = ep.pres; res.extra = ep.extra;
        if (!ep.pres.empty()) { res.a = ep.pres[0].a; res.b = ep.pres[0].b; res.c = ep.pres[0].c; }
    }
    res.nClusters = ml.nClusters(); res.subjectCluster = grp;
    res.condComputed = true;
    res.phi.resize(uz(ns)); res.ebe.resize(uz(ns)); res.psiEbe.resize(uz(ns)); res.phiCondSd.assign(uz(ns), Vec::Zero(p));
    std::vector<Vec> fInd(uz(ns)), fPop(uz(ns));
    Vec phiTmp;
    for (int i = 0; i < ns; ++i) {
        nl.phiOf(i, th, Vec::Zero(q), phiTmp);
        fPop[uz(i)].resize(0);
        if (!P.custom) nl.obs().predictAll(i, nl.psiOf(phiTmp), fPop[uz(i)]);
        for (int l = 0; l < q; ++l) { phiTmp[nl.reIdx()[uz(l)]] = ce.phiMean[uz(i)][l]; res.phiCondSd[uz(i)][nl.reIdx()[uz(l)]] = ce.phiSd[uz(i)][l]; }
        res.phi[uz(i)] = phiTmp; res.ebe[uz(i)] = phiTmp; res.psiEbe[uz(i)] = nl.psiOf(phiTmp);
        if (!P.custom) nl.obs().predictAll(i, res.psiEbe[uz(i)], fInd[uz(i)]);
    }
    res.xi = ce.xiMean; res.xiCondMean = ce.xiMean; res.xiCondSd = ce.xiSd;
    res.fittedPop.assign(uz(data.nObs()), std::numeric_limits<double>::quiet_NaN()); res.fittedInd = res.fittedPop;
    if (!P.custom)
        for (int r = 0; r < data.nObs(); ++r) {
            const int k = A.rowSubj[uz(r)], j = A.rowIdx[uz(r)];
            res.fittedPop[uz(r)] = fPop[uz(k)][j]; res.fittedInd[uz(r)] = fInd[uz(k)][j];
        }
    res.convergence = eng.convergence();
    for (int j = 0; j < p; ++j) {
        if (nl.reOf()[uz(j)] >= 0 && res.sd[j] < 1e-3 * std::max(std::abs(res.beta[j]), 1e-8))
            res.warnings.push_back("sd(subject p" + std::to_string(j) + ") ~ 0 relative to |beta|: no real variability or collapse");
        if (nl.reOf()[uz(j)] >= 0 && res.sdCluster[j] < 1e-3 * std::max(std::abs(res.beta[j]), 1e-8))
            res.warnings.push_back("sd(cluster p" + std::to_string(j) + ") ~ 0 relative to |beta|: no real variability or collapse");
    }
    for (const auto& w : P.initWarn) res.warnings.push_back(w);

    {
        const std::vector<char> bm = ml.boundaryMask(th);
        bool any = false; for (char b : bm) any = any || b;
        if (any && sem != SEMethod::None) {
            ml.markFixed(bm);
            res.warnings.push_back("variance(s) on the boundary (sd ~ 0): excluded from the FIM; their standard errors are undefined");
        }
    }
    // ---------- linearization (FOCE) ----------
    NlmeModel::FimResult hfr;
    if (!P.custom) {
        hfr = ml.linearizedFIM(th, ce.phiMean);
        if (hfr.ok) res.logLikLin = hfr.ll;
    }

    // ---------- marginal likelihood ----------
    if (llm == LLMethod::GaussHermite) {
        res.logLik = ml.logMarginal(th, control.gqNodes);
        res.llComputed = std::isfinite(res.logLik); res.llMethodUsed = "adaptive nested Gauss-Hermite";
    } else if (llm == LLMethod::ImportanceSampling) {
        res.logLik = ml.logMarginalIS(th, ce.draws, control.isSamples, control.seed);
        res.llComputed = std::isfinite(res.logLik); res.llMethodUsed = "joint importance sampling (t) per cluster";
    } else if (llm == LLMethod::Linearization && std::isfinite(res.logLikLin)) {
        res.logLik = res.logLikLin; res.llComputed = true; res.llMethodUsed = "linearization";
    } else if (llm == LLMethod::Laplace) {
        res.logLik = ml.laplaceLogLik(th, ml.laplaceStarts(ce.phiMean, ce.xiMean));
        res.llComputed = std::isfinite(res.logLik); res.llMethodUsed = "Laplace";
    }

    // ---------- standard errors (linearization, Laplace or Louis) ----------
    if (sem == SEMethod::Linearization || sem == SEMethod::Louis || sem == SEMethod::Laplace) {
        NlmeModel::FimResult lf;
        if (sem == SEMethod::Linearization) { lf = hfr; if (!lf.ok) res.warnings.push_back("hierarchical linearized FIM: some clusters could not be linearized"); }
        else if (sem == SEMethod::Laplace) {
            lf = ml.laplaceFIM(th, ml.laplaceStarts(ce.phiMean, ce.xiMean));
            if (!lf.ok) res.warnings.push_back("hierarchical Laplace: could not compute some cluster's contribution");
        }
        else {
            // Louis with i.i.d. importance sampling (without MCMC autocorrelation) and block-wise derivatives
            double noise = 0.0, essFrac = 1.0;
            lf = ml.louisIS(th, ce.draws, control.louisISSamples, control.seed, noise, essFrac);
            if (!lf.ok || essFrac < 0.02) {
                res.warnings.push_back("hierarchical Louis by importance sampling is unreliable (minimum ESS/M " + std::to_string(essFrac) +
                                       "): using the (noisy) MCMC samples");
                lf = detail::louisChecked(ml, th, ce.draws, ml.fixedMask(), noise);
            }
            if (noise > 0.3) res.warnings.push_back("hierarchical Louis' formula with Monte Carlo noise (relative difference " + std::to_string(noise) +
                                                    " between halves): increase louisISSamples or use SEMethod::Linearization");
        }
        Mat C; bool singular = false;
        if (lf.I.size() > 0 && lf.I.allFinite() && detail::invertFIM(lf.I, ml.fixedMask(), C, singular)) {
            res.seMethodUsed = (sem == SEMethod::Linearization) ? "hierarchical linearized FIM (FOCE with interaction)"
                             : (sem == SEMethod::Laplace) ? "observed FIM (numerical Hessian of the Laplace logLik, deterministic)"
                             : "observed FIM (Louis' formula per cluster)";
            if (singular) res.warnings.push_back("nearly singular FIM: some SEs are unreliable");
            const double nan = std::numeric_limits<double>::quiet_NaN();
            auto seOf = [&](int j) { return ml.fixedMask()[uz(j)] ? nan : (C(j, j) > 0 ? std::sqrt(C(j, j)) : nan); };
            res.covTheta = C;
            res.seCoef.resize(nb); for (int c = 0; c < nb; ++c) res.seCoef[c] = seOf(c);
            res.se = res.seCoef.head(p);
            res.sePsiPop.resize(p);
            for (int j = 0; j < p; ++j) res.sePsiPop[j] = std::abs(derivpsi(res.beta[j], trs[uz(j)])) * res.se[j];
            auto deltaSE = [&](int k0, int k1, auto&& fn) {
                Vec g = Vec::Zero(D);
                for (int k = k0; k < k1; ++k) {
                    if (ml.fixedMask()[uz(k)]) continue;
                    const double h = 1e-6 * std::max(1.0, std::abs(th[k]));
                    Vec tp = th, tm = th; tp[k] += h; tm[k] -= h;
                    g[k] = (fn(tp) - fn(tm)) / (2.0 * h);
                }
                const double v = g.dot(C * g);
                return v >= 0 ? std::sqrt(v) : nan;
            };
            const int o0 = nl.offOmega(), o1 = o0 + nl.nOmega(), s0 = ml.offPsi(), s1 = s0 + ml.nPsi();
            res.seSd = Vec::Constant(p, nan); res.seSdCluster = Vec::Constant(p, nan);
            res.seOmega = Mat::Constant(p, p, nan); res.seCorr = Mat::Constant(p, p, nan);
            auto psiFull = [&](const Vec& t) { const Mat Ps = ml.psiOf(t); Mat F = Mat::Zero(p, p); for (int a = 0; a < q; ++a) for (int b = 0; b < q; ++b) F(nl.reIdx()[uz(a)], nl.reIdx()[uz(b)]) = Ps(a, b); return F; };
            for (int a = 0; a < p; ++a) {
                if (nl.reOf()[uz(a)] < 0) continue;
                res.seSd[a] = deltaSE(o0, o1, [&](const Vec& t) { return std::sqrt(nl.omegaFull(t)(a, a)); });
                res.seSdCluster[a] = deltaSE(s0, s1, [&](const Vec& t) { return std::sqrt(psiFull(t)(a, a)); });
                for (int b = a; b < p; ++b) {
                    if (nl.reOf()[uz(b)] < 0) continue;
                    const double so = deltaSE(o0, o1, [&](const Vec& t) { return nl.omegaFull(t)(a, b); });
                    res.seOmega(a, b) = so; res.seOmega(b, a) = so;
                    if (a != b) {
                        const double sc2 = deltaSE(o0, o1, [&](const Vec& t) { const Mat O = nl.omegaFull(t); return O(a, b) / std::sqrt(O(a, a) * O(b, b)); });
                        res.seCorr(a, b) = sc2; res.seCorr(b, a) = sc2;
                    }
                }
            }
            const ObsLik& ob = nl.obs();
            for (int r = 0; r < ob.nResp(); ++r)
                for (int k = 0; k < resmod::nParams(ob.errorSpec(r).type); ++k) {
                    const int idx = nl.offError() + ob.errOffset(r) + k;
                    const bool raw = (ob.errorSpec(r).type == resmod::ErrorKind::Power && k == 2);
                    const double se = seOf(idx);
                    res.seErr.push_back(raw ? se : se * std::exp(th[idx]));
                }
            res.seExtra = Vec::Constant(ob.nExtra(), nan);
            for (int k = 0; k < ob.nExtra(); ++k) res.seExtra[k] = seOf(nl.offError() + ob.extraOffset() + k);
            res.seComputed = true;
        } else res.warnings.push_back("hierarchical: could not compute the standard errors");
    }
    return res;
}

// ===========================================================================
// Simulation of new data with the estimated parameters (for predictive checks)
// ===========================================================================
/** @brief Options for @ref simulate. */
struct SimOptions {
    unsigned seed = 1;
    int nrep = 1;
    double lloq = std::numeric_limits<double>::quiet_NaN();   // simulated y < lloq -> y = lloq, cens = -1
    double uloq = std::numeric_limits<double>::quiet_NaN();   // simulated y > uloq -> y = uloq, cens = +1
    bool newRandomEffects = true;                             // false: use the subjects' EBEs
};

// Returns `nrep` copies of res.data with a simulated y: eta_i ~ N(0, Omega) (or EBE), model
// prediction, and noise according to the error model (or Model.simulateObs if a custom likelihood is used).
inline std::vector<Data> simulate(const Results& res, const SimOptions& opt = SimOptions()) {
    using namespace mcmckl;
    if (res.theta.size() == 0) throw std::invalid_argument("simulate: Results without theta (single-level fit required)");
    if (res.data.isHierarchical()) throw std::invalid_argument("simulate: hierarchical models are not supported");
    Data data = res.data; Model model = res.model;
    std::vector<std::string> cn;
    detail::expandCategorical(data, model, cn);
    const int p = model.nPhi;
    const bool custom = static_cast<bool>(model.logLikObs);
    if (custom && !model.simulateObs) throw std::invalid_argument("simulate: a custom likelihood requires Model.simulateObs (see lik::*)");
    const int R = custom ? 0 : (model.errors.empty() ? 1 : static_cast<int>(model.errors.size()));
    detail::Assembled A = detail::assemble(data, model, res.control, std::vector<resmod::Pres>(uz(R), resmod::Pres(0.1, 0.1, 1.0)),
                                           Vec::Zero(p), Vec::Ones(p));
    NlmeModel& nl = *A.nl;
    const Vec& th = res.theta;
    const int ns = nl.nSubjects(), q = nl.q();
    const Mat Om = nl.omegaOf(th);
    Eigen::LLT<Mat> llt(Om);
    const Mat L = (llt.info() == Eigen::Success) ? Mat(llt.matrixL()) : Mat(Mat::Zero(q, q));
    const ErrParams ep = nl.obs().decode(th, nl.offError());
    std::vector<Data> out;
    for (int rep = 0; rep < opt.nrep; ++rep) {
        std::mt19937 rng(opt.seed + 7919u * unsigned(rep));
        std::normal_distribution<double> nz(0.0, 1.0);
        Data d = res.data;
        if (!std::isnan(opt.lloq) || !std::isnan(opt.uloq)) { d.cens.assign(uz(d.nObs()), 0); d.limit.clear(); }
        std::vector<Vec> psi(uz(ns)), fpred(uz(ns));
        for (int i = 0; i < ns; ++i) {
            Vec eta(q);
            if (opt.newRandomEffects) { Vec z(q); for (int l = 0; l < q; ++l) z[l] = nz(rng); eta = L * z; }
            else { Vec phi = res.ebe[uz(i)]; Vec base; nl.phiOf(i, th, Vec::Zero(q), base); for (int l = 0; l < q; ++l) eta[l] = phi[nl.reIdx()[uz(l)]] - base[nl.reIdx()[uz(l)]]; }
            Vec phi; nl.phiOf(i, th, eta, phi);
            psi[uz(i)] = nl.psiOf(phi);
            if (!custom) nl.obs().predictAll(i, psi[uz(i)], fpred[uz(i)]);
        }
        for (int r = 0; r < d.nObs(); ++r) {
            const int k = A.rowSubj[uz(r)], j = A.rowIdx[uz(r)];
            double y;
            if (custom) {
                const auto& S = nl.obs().subject(k);
                y = model.simulateObs(psi[uz(k)], std::span<const double>(S.x.data() + uz(j) * uz(S.stride), uz(S.stride)), nl.obs().subjectCov(k), ep.extra, rng);
            } else {
                const double f = fpred[uz(k)][j];
                const int rr = nl.obs().subject(k).ytype[uz(j)];
                const double g = nl.obs().errSd(k, j, f, ep);
                y = nl.obs().isExp(k, j) ? f * std::exp(g * nz(rng)) : f + g * nz(rng);
                (void)rr;
            }
            if (!std::isnan(opt.lloq) && y < opt.lloq) { y = opt.lloq; d.cens[uz(r)] = -1; }
            else if (!std::isnan(opt.uloq) && y > opt.uloq) { y = opt.uloq; d.cens[uz(r)] = 1; }
            d.y[uz(r)] = y;
        }
        out.push_back(std::move(d));
    }
    return out;
}

// Simulates TIME-TO-EVENT data with the estimated parameters (Model.tteCumHazard): eta_i ~ N(0, Omega) (or EBE),
// and the event time by inverting H(t). One row per event (y = 1, tstart = previous event) and a final
// censored row at tmax (y = 0) if there are no more events. `repeated = false`: only the first event.
// Each subject's predictors are copied from its first row (except tstart, which is updated).
inline std::vector<Data> simulateEvents(const Results& res, double tmax, bool repeated = false, const SimOptions& opt = SimOptions()) {
    using namespace mcmckl;
    if (!res.model.tteCumHazard) throw std::invalid_argument("simulateEvents: Model.tteCumHazard is required");
    if (res.theta.size() == 0 || res.data.isHierarchical()) throw std::invalid_argument("simulateEvents: single-level models only");
    Data data = res.data; Model model = res.model;
    std::vector<std::string> cn;
    detail::expandCategorical(data, model, cn);
    const int p = model.nPhi;
    detail::Assembled A = detail::assemble(data, model, res.control, std::vector<resmod::Pres>(), Vec::Zero(p), Vec::Ones(p));
    NlmeModel& nl = *A.nl;
    const Vec& th = res.theta;
    const int ns = nl.nSubjects(), q = nl.q();
    Eigen::LLT<Mat> llt(nl.omegaOf(th));
    const Mat L = (llt.info() == Eigen::Success) ? Mat(llt.matrixL()) : Mat(Mat::Zero(q, q));
    const std::vector<int> sids = data.subjectIds();
    const int P = static_cast<int>(data.predictorOrder.size());
    int tcol = model.tteStartCol - 1;                       // position in predictorOrder
    if (model.tteStartCol >= 0 && (tcol < 0 || tcol >= P)) throw std::invalid_argument("simulateEvents: tteStartCol is outside the predictors");
    std::vector<Data> out;
    for (int rep = 0; rep < opt.nrep; ++rep) {
        std::mt19937 rng(opt.seed + 104729u * unsigned(rep));
        std::normal_distribution<double> nz(0.0, 1.0);
        Data d;
        d.cov = data.cov; d.covOrder = data.covOrder; d.catCov = data.catCov; d.catReference = data.catReference;
        d.events = data.events; d.predictorOrder = data.predictorOrder;
        for (const auto& nm : data.predictorOrder) d.predictors[nm] = {};
        for (int k = 0; k < ns; ++k) {
            Vec eta(q);
            if (opt.newRandomEffects) { Vec z(q); for (int l = 0; l < q; ++l) z[l] = nz(rng); eta = L * z; }
            else { Vec base; nl.phiOf(k, th, Vec::Zero(q), base); for (int l = 0; l < q; ++l) eta[l] = res.ebe[uz(k)][nl.reIdx()[uz(l)]] - base[nl.reIdx()[uz(l)]]; }
            Vec phi; nl.phiOf(k, th, eta, phi);
            const Vec psi = nl.psiOf(phi);
            const auto& S = nl.obs().subject(k);
            std::vector<double> x(S.x.begin(), S.x.begin() + S.stride);      // subject's first row
            double t0 = 0.0;
            for (;;) {
                if (tcol >= 0) x[uz(1 + tcol)] = t0;
                const double T = lik::sampleEventTime(model.tteCumHazard, psi, x, nl.obs().subjectCov(k), t0, tmax, rng);
                const bool ev = std::isfinite(T);
                d.id.push_back(sids[uz(k)]); d.time.push_back(ev ? T : tmax); d.y.push_back(ev ? 1.0 : 0.0);
                for (int c = 0; c < P; ++c) d.predictors[data.predictorOrder[uz(c)]].push_back(x[uz(1 + c)]);
                if (!ev || !repeated) break;
                t0 = T;
            }
        }
        out.push_back(std::move(d));
    }
    return out;
}

// ===========================================================================
// Entry point: dispatches depending on whether Data.group is present
// ===========================================================================
inline Results fit(const Data& data, const Model& model, const Control& control) {
    data.check();
    return data.isHierarchical() ? fitHierarchical(data, model, control) : fitSingleLevel(data, model, control);
}

// Level 2/3 - explicit control
inline Results saem(const Data& data, const Model& model, const Control& control = Control()) {
    return fit(data, model, control);
}

// Level 1 - minimal: prediction as a lambda f(psi,t) and effect dimension
inline Results saem(const Data& data,
                    std::function<double(const Vec&, double)> predict,
                    int nPhi, ErrorModel error = ErrorModel::Proportional,
                    Control control = Control()) {
    Model m;
    m.nPhi = nPhi;
    m.error = error;
    m.predict = [predict](const Vec& psi, double t, const Vec&) { return predict(psi, t); };
    return fit(data, m, control);
}

} // namespace smx
