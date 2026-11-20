/**
 * @file nlme.hpp
 * @brief Complete one-level nonlinear mixed-effects model and its SAEM support.
 *
 * The model is defined as
 * @code
 *   phi_i = D_i beta + E eta_i ,    eta_i ~ N(0, Omega)        (Gaussian "phi" scale)
 *   psi_i = h(phi_i)                                           (h: identity/log/logit/probit per parameter)
 *   y_ij | psi_i  ~  observation model (obslik.hpp)
 * @endcode
 *
 * Features:
 *   * Per-parameter TRANSFORMATIONS (psi = h(phi)).
 *   * Parameters WITHOUT a random effect (eta_j = 0): the sampled variable is
 *     chi = eta restricted to the p_re parameters that have a random effect.
 *   * COVARIATE model: phi_j += sum_t beta_t * c_i[cov_t] (estimated coefficients).
 *   * Block covariance structure (diagonal, full, diagonal blocks).
 *   * FIXED parameters (coefficients, random-effect sd, errors).
 *   * M-step: closed-form GLS for coefficients of parameters with a random
 *     effect, block-wise closed-form Omega, residual error (obslik.hpp), and a
 *     NUMERICAL (Newton) step for coefficients of parameters without a random
 *     effect and for extra parameters of a custom likelihood.
 *   * LINEARIZED FIM (FOCE with interaction) and logLik by linearization.
 *
 * The parameter vector is laid out as
 * @code
 *   theta = [ beta (nb = p + n_cov_terms) ; Omega (per-block Cholesky) ;
 *             error / extra parameters (obslik.hpp) ]
 * @endcode
 *
 * The variable sampled by SAEM is chi_i = eta_i (dimension q) with prior N(0, Omega).
 *
 * Requires: Eigen3, C++20, engine.hpp, model.hpp, obslik.hpp, transform.hpp.
 */

#pragma once
#include "numcast.hpp"

#include "engine.hpp"
#include "model.hpp"
#include "covmodel.hpp"
#include "obslik.hpp"
#include "transform.hpp"
#include "parallel.hpp"

#include <atomic>

namespace mcmckl {

/**
 * @brief One-level nonlinear mixed-effects model.
 *
 * Implements the SAEM model interface (SaemModel) together with closed-form
 * sufficient-statistic updates, numerical M-steps, and several Fisher
 * information approximations.
 *
 * A single instance owns its specification and observation model. Log-joint
 * evaluations use a thread-local cache of the prior decomposition keyed by the
 * current theta, so repeated evaluations at the same theta are cheap. The cache
 * is per thread; distinct model instances are distinguished by an internal id.
 */
class NlmeModel : public SaemModel {
public:
    /// Pairing of an estimated coefficient with a subject-level covariate column.
    struct CovTerm { int param; int covariate; };      // covariate: column of the per-subject covariates

    /**
     * @brief Model specification.
     *
     * Empty optional vectors are filled with their documented defaults by the
     * constructor.
     */
    struct Spec {
        int dimPhi = 1;                                ///< Number of model parameters p; must be > 0.
        std::vector<Transform> transform;              ///< Size p (empty = identity for all parameters).
        std::vector<char>      randomEffect;           ///< Size p (empty = every parameter has a random effect).
        std::vector<int>       omegaBlock;             ///< Size p: equal id = full block (empty = diagonal).
        std::vector<std::vector<char>> covPattern;     ///< Alternative to omegaBlock: p x p 0/1 pattern of free covariances.
        std::vector<CovTerm>   covTerms;               ///< Covariate terms added to the linear predictor.
        Vec beta0;                                     ///< Size nb, on the phi scale (empty = zeros).
        Vec sd0;                                       ///< Size p (empty = ones).
        std::vector<char> betaFixed;                   ///< Size nb; nonzero entries are held fixed.
        std::vector<char> sdFixed;                     ///< Size p (only for blocks with a single parameter).
    };

    int threads = 1;                                   ///< Number of threads for the numerical step.

    /**
     * @brief Builds the model from a specification and an observation model.
     * @param spec Model specification; empty optional fields are defaulted in place.
     * @param obs Observation likelihood describing the subjects.
     * @throws std::invalid_argument If @p spec is inconsistent (dimensions,
     *         ranges, no random-effect parameter, or mismatched initial values).
     *
     * Thread-safety: construction is not thread-safe with respect to concurrent
     * use of the same object; after construction the object is immutable except
     * for the explicitly mutating helpers (for example markFixed()).
     */
    NlmeModel(Spec spec, ObsLik obs)
      : spec_(std::move(spec)), obs_(std::move(obs)), p_(spec_.dimPhi), id_(nextId())
    {
        if (p_ <= 0) throw std::invalid_argument("dimPhi > 0");
        n_ = obs_.nSubjects();
        if (spec_.transform.empty()) spec_.transform.assign(uz(p_), Transform::Identity);
        if (spec_.randomEffect.empty()) spec_.randomEffect.assign(uz(p_), 1);
        if (spec_.omegaBlock.empty()) { spec_.omegaBlock.resize(uz(p_)); for (int j = 0; j < p_; ++j) spec_.omegaBlock[uz(j)] = j; }
        if (static_cast<int>(spec_.transform.size()) != p_ || static_cast<int>(spec_.randomEffect.size()) != p_ ||
            static_cast<int>(spec_.omegaBlock.size()) != p_)
            throw std::invalid_argument("transform/randomEffect/omegaBlock have incorrect dimension");

        reOf_.assign(uz(p_), -1);
        for (int j = 0; j < p_; ++j) if (spec_.randomEffect[uz(j)]) { reOf_[uz(j)] = static_cast<int>(reIdx_.size()); reIdx_.push_back(j); }
        q_ = static_cast<int>(reIdx_.size());
        if (q_ == 0) throw std::invalid_argument("at least one parameter with a random effect is required");

        T_ = static_cast<int>(spec_.covTerms.size());
        nb_ = p_ + T_;
        coefParam_.resize(uz(nb_)); coefCov_.assign(uz(nb_), -1);
        for (int j = 0; j < p_; ++j) coefParam_[uz(j)] = j;
        const int ncov = (n_ > 0) ? static_cast<int>(obs_.subjectCov(0).size()) : 0;
        for (int t = 0; t < T_; ++t) {
            const auto& ct = spec_.covTerms[uz(t)];
            if (ct.param < 0 || ct.param >= p_) throw std::invalid_argument("covTerm.param is out of range");
            if (ct.covariate < 0 || ct.covariate >= ncov) throw std::invalid_argument("covTerm.covariate is out of range");
            coefParam_[uz(p_ + t)] = ct.param; coefCov_[uz(p_ + t)] = ct.covariate;
        }
        covVals_.assign(uz(n_), Vec::Zero(T_));
        for (int i = 0; i < n_; ++i)
            for (int t = 0; t < T_; ++t) covVals_[uz(i)][t] = obs_.subjectCov(i)[coefCov_[uz(p_ + t)]];
        for (int c = 0; c < nb_; ++c) if (reOf_[uz(coefParam_[uz(c)])] >= 0) reCoef_.push_back(c);

        // Omega structure over the parameters that carry a random effect
        {
            std::vector<std::vector<char>> pat(uz(q_), std::vector<char>(uz(q_), 0));
            if (!spec_.covPattern.empty()) {
                if (static_cast<int>(spec_.covPattern.size()) != p_) throw std::invalid_argument("covPattern must be p x p");
                for (int a = 0; a < q_; ++a) for (int b2 = 0; b2 < q_; ++b2)
                    pat[uz(a)][uz(b2)] = (a != b2) && (spec_.covPattern[uz(reIdx_[uz(a)])][uz(reIdx_[uz(b2)])] || spec_.covPattern[uz(reIdx_[uz(b2)])][uz(reIdx_[uz(a)])]);
            } else {
                for (int a = 0; a < q_; ++a) for (int b2 = 0; b2 < q_; ++b2)
                    pat[uz(a)][uz(b2)] = (a != b2) && spec_.omegaBlock[uz(reIdx_[uz(a)])] == spec_.omegaBlock[uz(reIdx_[uz(b2)])];
            }
            cov_ = CovModel(pat);
        }
        nOm_ = cov_.nParams();
        offOm_ = nb_; offErr_ = nb_ + nOm_;

        // G = (1/n) sum_i x_i x_i'
        G_ = Mat::Zero(nb_, nb_);
        for (int i = 0; i < n_; ++i) {
            Vec x(nb_);
            for (int c = 0; c < nb_; ++c) x[c] = xcoef(i, c);
            G_ += x * x.transpose() / double(std::max(1, n_));
        }

        // Initial theta and fixed-parameter mask
        if (spec_.beta0.size() == 0) spec_.beta0 = Vec::Zero(nb_);
        if (spec_.sd0.size() == 0) spec_.sd0 = Vec::Ones(p_);
        if (spec_.beta0.size() != nb_ || spec_.sd0.size() != p_) throw std::invalid_argument("beta0/sd0 have incorrect dimension");
        if (spec_.betaFixed.empty()) spec_.betaFixed.assign(uz(nb_), 0);
        if (spec_.sdFixed.empty()) spec_.sdFixed.assign(uz(p_), 0);
        theta0_ = Vec::Zero(dimTheta());
        theta0_.head(nb_) = spec_.beta0;
        {
            Vec sdRE(q_);
            for (int l = 0; l < q_; ++l) sdRE[l] = spec_.sd0[reIdx_[uz(l)]];
            cov_.encodeSd(theta0_, offOm_, sdRE);
        }
        theta0_.tail(obs_.nErrParams()) = obs_.initialErr();
        fixed_.assign(uz(dimTheta()), 0);
        for (int c = 0; c < nb_; ++c) fixed_[uz(c)] = spec_.betaFixed[uz(c)];
        sdFixedRE_.assign(uz(q_), 0);
        for (int l = 0; l < q_; ++l)
            if (cov_.singleton(l) && spec_.sdFixed[uz(reIdx_[uz(l)])]) { fixed_[uz(offOm_ + cov_.singletonParam(l))] = 1; sdFixedRE_[uz(l)] = 1; }
        {
            const auto fe = obs_.fixedMask();
            for (size_t k = 0; k < fe.size(); ++k) fixed_[uz(offErr_) + k] = fe[k];
        }
        // Numerical block: coefficients of parameters WITHOUT a random effect and free extra parameters
        for (int c = 0; c < nb_; ++c) if (reOf_[uz(coefParam_[uz(c)])] < 0 && !fixed_[uz(c)]) nbIdx_.push_back(c);
        for (int k = 0; k < obs_.nExtra(); ++k) {
            const int idx = offErr_ + obs_.extraOffset() + k;
            if (!fixed_[uz(idx)]) nbIdx_.push_back(idx);
        }
    }

    // --- SaemModel ----------------------------------------------------------
    /// @return Total number of theta coordinates.
    int dimTheta() const override { return nb_ + nOm_ + obs_.nErrParams(); }
    /// @return Dimension q of the sampled random effects.
    int dimChi()   const override { return q_; }
    /// @return Number of subjects.
    int nSubjects() const override { return n_; }
    /// @return Initial parameter vector theta0.
    Vec initialTheta() const override { return theta0_; }
    /// @return Zero initial value for the random effects of subject @p (ignored).
    Vec initialChi(int) const override { return Vec::Zero(q_); }

    /**
     * @brief Evaluates log p(y_i, eta | theta) for one subject.
     * @param i Subject index.
     * @param eta Random-effect vector (size q).
     * @param theta Parameter vector.
     * @return Log joint density, or -infinity if the prior is invalid or the
     *         observation log-likelihood is not finite.
     *
     * Thread-safety: safe for concurrent calls with different subjects or
     * threads; a per-thread cache holds the prior decomposition and workspace.
     */
    double logJoint(int i, const Vec& eta, const Vec& theta) const override {
        const PriorCache& c = cacheFor(theta);
        if (!c.ok) return -std::numeric_limits<double>::infinity();
        // Per-thread workspace (next to the cache): no allocation per evaluation
        Vec& phi = c.wPhi; Vec& psi = c.wPsi; Vec& d = c.wD;
        phiOf(i, theta, eta, phi);
        psi.resize(p_);
        for (int j = 0; j < p_; ++j) psi[j] = transpsi(phi[j], spec_.transform[uz(j)]);
        const double ll = obs_.logLik(i, psi, c.ep, c.wF);
        if (!std::isfinite(ll)) return -std::numeric_limits<double>::infinity();
        d = eta;
        c.L.triangularView<Eigen::Lower>().solveInPlace(d);
        return -0.5 * (d.squaredNorm() + c.logdet + double(q_) * std::log(2.0 * std::numbers::pi)) + ll;
    }

    /**
     * @brief Stochastic M-step gradient by central differences over free coordinates.
     * @param theta Current parameter vector.
     * @param etas Random effects, one vector per subject.
     * @return Gradient of the mean log joint with respect to theta, with eta held fixed.
     *
     * This is the "legacy" M-step (Robbins-Monro): with fixed eta the beta
     * gradient comes from the likelihood (non-centered parametrization, noisier),
     * and scaling across coordinates is delicate. The classical SAEM
     * (MStep::SufficientStats, the default) is much more robust.
     */
    Vec mStepGradient(const Vec& theta, const std::vector<Vec>& etas) const override {
        const int d = dimTheta();
        Vec g = Vec::Zero(d);
        auto F = [&](const Vec& th) {
            double s = 0.0;
            for (int i = 0; i < n_; ++i) s += logJoint(i, etas[uz(i)], th);
            return s / double(n_);
        };
        for (int j = 0; j < d; ++j) {
            if (fixed_[uz(j)]) continue;
            const double h = 1e-6 * std::max(1.0, std::abs(theta[j]));
            Vec tp = theta, tm = theta; tp[j] += h; tm[j] -= h;
            const double fp = F(tp), fm = F(tm);
            if (std::isfinite(fp) && std::isfinite(fm)) g[j] = (fp - fm) / (2.0 * h);
        }
        return g;
    }

    /**
     * @brief Independence proposal for the SAEM sampler.
     * @param theta Current parameter vector.
     * @param[out] mean Proposal mean (zero).
     * @param[out] cov Proposal covariance Omega(theta).
     */
    void independenceProposal(const Vec& theta, Vec& mean, Mat& cov) const override {
        mean = Vec::Zero(q_);
        cov = omegaOf(theta);
    }

    /// @return Indices of the theta coordinates that represent variances.
    std::vector<int> varianceIndices() const override { return cov_.varianceIndices(offOm_); }

    // --- Closed-form M-step ------------------------------------------------
    //   s = [ M (nbRE x q) | S2 (q x q) | error statistics ]
    //   M_kl = E[x_ic_k * phi_l],   S2_lm = E[phi_l phi_m]     (phi = linear prediction + eta, l,m in RE)

    /// @return Always true: this model provides sufficient statistics.
    bool hasSufficientStats() const override { return true; }

    /**
     * @brief Computes the sufficient statistics for the closed-form M-step.
     * @param etas Random effects, one vector per subject.
     * @param theta Current parameter vector.
     * @return Concatenation of population statistics and observation-model statistics.
     */
    Vec sufficientStats(const std::vector<Vec>& etas, const Vec& theta) const override {
        std::vector<Vec> phiRE(uz(n_)), psis(uz(n_));
        Vec phi;
        for (int i = 0; i < n_; ++i) {
            phiOf(i, theta, etas[uz(i)], phi);
            psis[uz(i)] = psiOf(phi);
            phiRE[uz(i)].resize(q_);
            for (int l = 0; l < q_; ++l) phiRE[uz(i)][l] = phi[reIdx_[uz(l)]];
        }
        const Vec sp = popStats(phiRE);
        Vec s(sp.size() + obs_.nStats());
        s.head(sp.size()) = sp;
        if (obs_.nStats() > 0) s.tail(obs_.nStats()) = obs_.stats(psis, theta, offErr_);
        return s;
    }

    /**
     * @brief Population statistics [ M (nbRE x q) | S2 (q x q) ] from the subject phi_RE values.
     * @param phiRE Per-subject random-effect components of phi.
     * @return Vectorized M followed by vectorized S2.
     *
     * In the hierarchical model, phi_RE - xi_c is used (without the cluster part).
     */
    Vec popStats(const std::vector<Vec>& phiRE) const {
        const int nbRE = static_cast<int>(reCoef_.size());
        Vec s = Vec::Zero(nbRE * q_ + q_ * q_);
        const double w = 1.0 / double(std::max(1, n_));
        for (int i = 0; i < n_; ++i) {
            for (int k = 0; k < nbRE; ++k) {
                const double x = xcoef(i, reCoef_[uz(k)]);
                for (int l = 0; l < q_; ++l) s[k * q_ + l] += w * x * phiRE[uz(i)][l];
            }
            for (int l = 0; l < q_; ++l) for (int m = 0; m < q_; ++m) s[nbRE * q_ + l * q_ + m] += w * phiRE[uz(i)][l] * phiRE[uz(i)][m];
        }
        return s;
    }

    /// @brief Observation-model statistics for the given transformed predictions.
    Vec errStats(const std::vector<Vec>& psis, const Vec& theta) const {
        return obs_.nStats() > 0 ? obs_.stats(psis, theta, offErr_) : Vec();
    }
    /// @return Number of observation-model sufficient statistics.
    int nStatsErr() const { return obs_.nStats(); }
    /// @return Number of population sufficient statistics.
    int nPopStats() const { return static_cast<int>(reCoef_.size()) * q_ + q_ * q_; }

    /**
     * @brief Maps sufficient statistics to a new theta (closed-form M-step).
     * @param s Sufficient statistics produced by sufficientStats().
     * @param theta Current parameter vector.
     * @return Updated parameter vector with fixed coordinates restored to their initial values.
     */
    Vec thetaFromStats(const Vec& s, const Vec& theta) const override {
        const int nbRE = static_cast<int>(reCoef_.size());
        Vec th = theta;
        Mat M(nbRE, q_), S2(q_, q_);
        for (int k = 0; k < nbRE; ++k) for (int l = 0; l < q_; ++l) M(k, l) = s[k * q_ + l];
        for (int l = 0; l < q_; ++l) for (int m = 0; m < q_; ++m) S2(l, m) = s[nbRE * q_ + l * q_ + m];

        // (1) Coefficients of parameters with a random effect: GLS with W = current Omega^{-1}
        const Mat Om = omegaOf(theta);
        Eigen::LDLT<Mat> ld(Om);
        const Mat W = ld.solve(Mat::Identity(q_, q_));
        std::vector<int> U, K;
        for (int k = 0; k < nbRE; ++k) (fixed_[uz(reCoef_[uz(k)])] ? K : U).push_back(k);
        if (!U.empty() && W.allFinite()) {
            Mat A(nbRE, nbRE); Vec b(nbRE);
            for (int k = 0; k < nbRE; ++k) {
                const int lk = reOf_[uz(coefParam_[uz(reCoef_[uz(k)])])];
                b[k] = W.row(lk).dot(M.row(k));
                for (int k2 = 0; k2 < nbRE; ++k2)
                    A(k, k2) = W(lk, reOf_[uz(coefParam_[uz(reCoef_[uz(k2)])])]) * G_(reCoef_[uz(k)], reCoef_[uz(k2)]);
            }
            Vec rhs(U.size()); Mat AUU(U.size(), U.size());
            for (size_t a = 0; a < U.size(); ++a) {
                rhs[tix(a)] = b[U[a]];
                for (int kk : K) rhs[tix(a)] -= A(U[a], kk) * theta[reCoef_[uz(kk)]];
                for (size_t c2 = 0; c2 < U.size(); ++c2) AUU(tix(a), tix(c2)) = A(U[a], U[c2]);
            }
            Vec x = AUU.completeOrthogonalDecomposition().solve(rhs);
            if (x.allFinite()) for (size_t a = 0; a < U.size(); ++a) th[reCoef_[uz(U[a])]] = x[tix(a)];
        }

        // (2) Block Omega: E[(phi - D beta)(phi - D beta)'] with the new beta
        Mat T1 = Mat::Zero(q_, q_), T2 = Mat::Zero(q_, q_);
        for (int k = 0; k < nbRE; ++k) {
            const int c = reCoef_[uz(k)], m = reOf_[uz(coefParam_[uz(c)])];
            for (int l = 0; l < q_; ++l) T1(l, m) += th[c] * M(k, l);
            for (int k2 = 0; k2 < nbRE; ++k2) {
                const int c2 = reCoef_[uz(k2)], m2 = reOf_[uz(coefParam_[uz(c2)])];
                T2(m, m2) += th[c] * th[c2] * G_(c, c2);
            }
        }
        Mat Oh = S2 - T1 - T1.transpose() + T2;
        Oh = 0.5 * (Oh + Oh.transpose());
        cov_.encodeFromMatrix(th, offOm_, Oh, sdFixedRE_);

        // (3) Residual error
        if (obs_.nStats() > 0) obs_.mstep(s.tail(obs_.nStats()), th, offErr_);
        for (int j = 0; j < dimTheta(); ++j) if (fixed_[uz(j)]) th[j] = theta0_[j];
        return th;
    }

    /**
     * @brief NUMERICAL step (Newton with finite differences and line search) over the
     *        coefficients of parameters without a random effect and the extra parameters.
     * @param[in,out] theta Parameter vector updated in place.
     * @param etas Random effects per chain and subject.
     * @param gamma Step size in (0, 1].
     *
     * Performs nu <- nu + gamma (argmax_nu sum_i log p(y_i | psi(nu, eta_i)) - nu).
     */
    void numericMStep(Vec& theta, const std::vector<std::vector<Vec>>& etas, double gamma) const override {
        if (nbIdx_.empty()) return;
        std::vector<std::vector<Vec>> phiRE(etas.size(), std::vector<Vec>(uz(n_)));
        Vec phi;
        for (size_t m = 0; m < etas.size(); ++m)
            for (int i = 0; i < n_; ++i) {
                phiOf(i, theta, etas[m][uz(i)], phi);
                phiRE[m][uz(i)].resize(q_);
                for (int l = 0; l < q_; ++l) phiRE[m][uz(i)][l] = phi[reIdx_[uz(l)]];
            }
        numericMStepPhiRE(theta, phiRE, gamma);
    }

    /**
     * @brief Same as numericMStep(), but starting from fixed phi_RE values per chain and subject.
     * @param[in,out] theta Parameter vector updated in place.
     * @param phiRE phi_RE values per chain and subject.
     * @param gamma Step size in (0, 1].
     *
     * Only coefficients without a random effect and the extra parameters change,
     * so the random-effect components of phi are held fixed.
     */
    void numericMStepPhiRE(Vec& theta, const std::vector<std::vector<Vec>>& phiRE, double gamma) const {
        if (nbIdx_.empty()) return;
        const int d = static_cast<int>(nbIdx_.size());
        const int M = static_cast<int>(phiRE.size());
        auto F = [&](const Vec& nu) -> double {
            Vec th = theta;
            for (int k = 0; k < d; ++k) th[nbIdx_[uz(k)]] = nu[k];
            const ErrParams ep = obs_.decode(th, offErr_);
            std::vector<double> part(uz(n_), 0.0);
            parallelFor(n_, threads, [&](int i) {
                Vec phi; double acc = 0.0;
                for (int m = 0; m < M; ++m) {
                    phiOf(i, th, Vec::Zero(q_), phi);
                    for (int l = 0; l < q_; ++l) phi[reIdx_[uz(l)]] = phiRE[uz(m)][uz(i)][l];
                    acc += obs_.logLik(i, psiOf(phi), ep);
                }
                part[uz(i)] = acc / double(M);
            });
            double s = 0.0;
            for (double v : part) { if (!std::isfinite(v)) return std::numeric_limits<double>::infinity(); s += v; }
            return -s / double(std::max(1, n_));
        };
        Vec nu0(d);
        for (int k = 0; k < d; ++k) nu0[k] = theta[nbIdx_[uz(k)]];
        const Vec nu = newtonMin(F, nu0, 3);
        for (int k = 0; k < d; ++k) theta[nbIdx_[uz(k)]] += gamma * (nu[k] - nu0[k]);
    }

    // --- Pieces for the hierarchical model (nlme_ml.hpp) ------------------------
    /// @brief Random-effect part (q) of D_i beta.
    Vec baseRE(int i, const Vec& theta) const {
        Vec b; phiOf(i, theta, Vec::Zero(q_), b);
        Vec r(q_); for (int l = 0; l < q_; ++l) r[l] = b[reIdx_[uz(l)]];
        return r;
    }
    /// @brief log p(y_i | full phi) using the thread cache for the error parameters.
    double logLikPhi(int i, const Vec& phi, const Vec& theta) const {
        const PriorCache& c = cacheFor(theta);
        return obs_.logLik(i, psiOf(phi), c.ep);
    }
    /// @brief log N(d; 0, Omega) using the thread cache; -inf if Omega is not positive definite.
    double logNormalOmega(const Vec& d, const Vec& theta) const {
        const PriorCache& c = cacheFor(theta);
        if (!c.ok) return -std::numeric_limits<double>::infinity();
        const Vec z = c.L.triangularView<Eigen::Lower>().solve(d);
        return -0.5 * (z.squaredNorm() + c.logdet + double(q_) * std::log(2.0 * std::numbers::pi));
    }

    // --- Robust generalized M-step (see SaemModel::gemStep) -------------------------------
    // Maximizes the CENTERED complete-data log-likelihood (with phi_RE fixed across chains) over
    // the free coordinates using damped Newton (<= 4 iterations, step <= 2 per coordinate).
    /// @return Always true: a generalized M-step is available.
    bool hasGemStep() const override { return true; }

    /**
     * @brief Generalized M-step using damped Newton on the centered complete-data log-likelihood.
     * @param[in,out] theta Parameter vector updated in place.
     * @param etas Random effects per chain and subject.
     * @param gamma Step size in (0, 1].
     */
    void gemStep(Vec& theta, const std::vector<std::vector<Vec>>& etas, double gamma) const override {
        const int D = dimTheta(), M = static_cast<int>(etas.size());
        std::vector<int> fr;
        for (int j = 0; j < D; ++j) if (!fixed_[uz(j)]) fr.push_back(j);
        const int m = static_cast<int>(fr.size());
        if (m == 0) return;
        std::vector<std::vector<Vec>> phiRE(uz(M), std::vector<Vec>(uz(n_)));
        Vec phi;
        for (int c = 0; c < M; ++c) for (int i = 0; i < n_; ++i) {
            phiOf(i, theta, etas[uz(c)][uz(i)], phi);
            phiRE[uz(c)][uz(i)].resize(q_);
            for (int l = 0; l < q_; ++l) phiRE[uz(c)][uz(i)][l] = phi[reIdx_[uz(l)]];
        }
        auto F = [&](const Vec& x) -> double {
            Vec th = theta;
            for (int a = 0; a < m; ++a) th[fr[uz(a)]] = x[a];
            Eigen::LLT<Mat> llt(omegaOf(th));
            if (llt.info() != Eigen::Success) return std::numeric_limits<double>::infinity();
            const Mat L = llt.matrixL();
            const double logdet = 2.0 * L.diagonal().array().log().sum();
            const ErrParams ep = obs_.decode(th, offErr_);
            std::vector<double> part(uz(n_), 0.0);
            parallelFor(n_, threads, [&](int i) {
                Vec base, ph, dv(q_); double acc = 0.0;
                phiOf(i, th, Vec::Zero(q_), base);
                for (int c = 0; c < M; ++c) {
                    ph = base;
                    for (int l = 0; l < q_; ++l) { dv[l] = phiRE[uz(c)][uz(i)][l] - base[reIdx_[uz(l)]]; ph[reIdx_[uz(l)]] = phiRE[uz(c)][uz(i)][l]; }
                    const double ll = obs_.logLik(i, psiOf(ph), ep);
                    if (!std::isfinite(ll)) { part[uz(i)] = std::numeric_limits<double>::quiet_NaN(); return; }
                    const Vec z = L.triangularView<Eigen::Lower>().solve(dv);
                    acc += -0.5 * (z.squaredNorm() + logdet + double(q_) * std::log(2.0 * std::numbers::pi)) + ll;
                }
                part[uz(i)] = acc / double(M);
            });
            double s = 0.0;
            for (double v : part) { if (!std::isfinite(v)) return std::numeric_limits<double>::infinity(); s += v; }
            return -s / double(std::max(1, n_));
        };
        Vec x0(m);
        for (int a = 0; a < m; ++a) x0[a] = theta[fr[uz(a)]];
        const Vec xs = newtonMin(F, x0, 4, 2.0);
        for (int a = 0; a < m; ++a) theta[fr[uz(a)]] += gamma * (xs[a] - x0[a]);
    }

    /**
     * @brief Damped Newton minimization with finite-difference derivatives and backtracking.
     * @tparam Fn Callable with signature double(const Vec&); may return +infinity.
     * @param F Objective function to minimize.
     * @param x Starting point; returned unchanged on failure.
     * @param maxit Maximum number of Newton iterations.
     * @param maxStep Maximum change per coordinate per iteration (trust region);
     *        infinity disables the cap.
     * @return The (possibly updated) minimizer estimate.
     *
     * The Hessian is regularized by clipping its eigenvalues from below before
     * forming the step, and a backtracking line search ensures monotone decrease.
     */
    template <class Fn>
    static Vec newtonMin(Fn&& F, Vec x, int maxit, double maxStep = std::numeric_limits<double>::infinity()) {
        const int d = static_cast<int>(x.size());
        double f0 = F(x);
        if (!std::isfinite(f0)) return x;
        for (int it = 0; it < maxit; ++it) {
            Vec h(d);
            for (int k = 0; k < d; ++k) h[k] = 1e-4 * std::max(1.0, std::abs(x[k]));
            Vec g(d); Mat H(d, d);
            std::vector<double> fp(uz(d)), fm(uz(d));
            for (int k = 0; k < d; ++k) {
                Vec a = x, b = x; a[k] += h[k]; b[k] -= h[k];
                fp[uz(k)] = F(a); fm[uz(k)] = F(b);
                if (!std::isfinite(fp[uz(k)]) || !std::isfinite(fm[uz(k)])) return x;
                g[k] = (fp[uz(k)] - fm[uz(k)]) / (2.0 * h[k]);
                H(k, k) = (fp[uz(k)] - 2.0 * f0 + fm[uz(k)]) / (h[k] * h[k]);
            }
            for (int k = 0; k < d; ++k) for (int l = k + 1; l < d; ++l) {
                Vec a = x, b = x, c = x, e = x;
                a[k] += h[k]; a[l] += h[l]; b[k] += h[k]; b[l] -= h[l];
                c[k] -= h[k]; c[l] += h[l]; e[k] -= h[k]; e[l] -= h[l];
                const double v = (F(a) - F(b) - F(c) + F(e)) / (4.0 * h[k] * h[l]);
                if (!std::isfinite(v)) return x;
                H(k, l) = v; H(l, k) = v;
            }
            Eigen::SelfAdjointEigenSolver<Mat> es(H);
            Vec ev = es.eigenvalues();
            const double fl = 1e-8 * std::max(1.0, ev.cwiseAbs().maxCoeff());
            const Vec step = -(es.eigenvectors() * ev.cwiseAbs().cwiseMax(fl).cwiseInverse().asDiagonal() *
                               es.eigenvectors().transpose() * g);
            Vec step2 = step;
            const double mx = step2.cwiseAbs().maxCoeff();
            if (mx > maxStep) step2 *= maxStep / mx;                      // trust region
            double t = 1.0; bool improved = false;
            for (int ls = 0; ls < 8; ++ls, t *= 0.5) {
                const Vec cand = x + t * step2;
                const double fc = F(cand);
                if (std::isfinite(fc) && fc < f0) { x = cand; f0 = fc; improved = true; break; }
            }
            if (!improved || step2.cwiseAbs().maxCoeff() * t < 1e-8) break;
        }
        return x;
    }

    /**
     * @brief Flags variance parameters that lie on the boundary.
     * @param th Parameter vector.
     * @param tol Relative tolerance comparing sd to |beta| of the parameter.
     * @return Mask of size dimTheta() with ones for boundary variance parameters.
     *
     * A parameter is considered degenerate when sd ~ 0 relative to |beta|; the
     * likelihood is flat/degenerate there and its score has huge variance, so
     * these parameters are excluded from the FIM (treated as fixed).
     */
    std::vector<char> boundaryMask(const Vec& th, double tol = 1e-3) const {
        std::vector<char> m(uz(dimTheta()), 0);
        const Mat Om = omegaOf(th);
        for (const auto& c : cov_.comps()) {
            bool deg = false;
            for (int l : c.members) deg = deg || (std::sqrt(std::max(Om(l, l), 0.0)) < tol * std::max(std::abs(th[reIdx_[uz(l)]]), 1e-8));
            if (deg) for (int k = 0; k < c.nPar; ++k) m[uz(offOm_ + c.off + k)] = 1;
        }
        return m;
    }

    /**
     * @brief Marks entries as fixed (intended for the FIM computation after fitting).
     * @param m Mask; entries that are nonzero are marked fixed.
     */
    void markFixed(const std::vector<char>& m) { for (size_t j = 0; j < m.size() && j < fixed_.size(); ++j) if (m[j]) fixed_[j] = 1; }

    // --- Accessors -----------------------------------------------------------
    /// @return The observation model.
    const ObsLik& obs() const { return obs_; }
    /// @return Number of model parameters p.
    int p() const { return p_; }
    /// @return Number of random-effect parameters q.
    int q() const { return q_; }
    /// @return Number of fixed-effect coefficients nb.
    int nb() const { return nb_; }
    /// @return Number of covariate terms.
    int nTerms() const { return T_; }
    /// @return Offset of the Omega block in theta.
    int offOmega() const { return offOm_; }
    /// @return Number of Omega parameters.
    int nOmega() const { return nOm_; }
    /// @return Offset of the error parameters in theta.
    int offError() const { return offErr_; }
    /// @return Indices of parameters with a random effect.
    const std::vector<int>& reIdx() const { return reIdx_; }
    /// @return For each parameter, its random-effect index or -1.
    const std::vector<int>& reOf() const { return reOf_; }
    /// @return The block covariance model.
    const CovModel& covModel() const { return cov_; }
    /// @return Parameter index associated with coefficient @p c.
    int coefParam(int c) const { return coefParam_[uz(c)]; }
    /// @return Fixed-parameter mask over theta.
    const std::vector<char>& fixedMask() const { return fixed_; }
    /// @return Transformation of parameter @p j.
    Transform transform(int j) const { return spec_.transform[uz(j)]; }

    /// @brief q x q covariance of the random effects.
    Mat omegaOf(const Vec& th) const { return cov_.decode(th, offOm_); }
    /// @brief p x p covariance (zeros for parameters without a random effect).
    Mat omegaFull(const Vec& th) const {
        const Mat Om = omegaOf(th);
        Mat F = Mat::Zero(p_, p_);
        for (int a = 0; a < q_; ++a) for (int b = 0; b < q_; ++b) F(reIdx_[uz(a)], reIdx_[uz(b)]) = Om(a, b);
        return F;
    }
    /// @brief phi_i (Gaussian scale) = D_i beta + E eta.
    void phiOf(int i, const Vec& theta, const Vec& eta, Vec& phi) const {
        phi.resize(p_);
        for (int j = 0; j < p_; ++j) phi[j] = theta[j];
        for (int t = 0; t < T_; ++t) phi[spec_.covTerms[uz(t)].param] += theta[p_ + t] * covVals_[uz(i)][t];
        for (int l = 0; l < q_; ++l) phi[reIdx_[uz(l)]] += eta[l];
    }
    /// @brief Applies the per-parameter transformations to a full phi vector.
    Vec psiOf(const Vec& phi) const {
        Vec psi(p_);
        for (int j = 0; j < p_; ++j) psi[j] = transpsi(phi[j], spec_.transform[uz(j)]);
        return psi;
    }
    /// @brief Design coefficient x_ic.
    double xcoef(int i, int c) const { return c < p_ ? 1.0 : covVals_[uz(i)][c - p_]; }

    /// Result of a Fisher information computation.
    struct FimResult { Mat I; double ll = std::numeric_limits<double>::quiet_NaN(); bool ok = false; };

    // ----------------------------------------------------------------------
    // LAPLACE approximation of log p(y_i | theta) = log ∫ p(y_i, eta | theta) d eta:
    //     f(eta^) + (q/2) log 2 pi - 0.5 log det(-H),  eta^ = conditional mode (Newton, finite differences)
    // `x` is the starting point (EBE) and is updated with the mode. Returns -inf on failure.
    // ----------------------------------------------------------------------
    /**
     * @brief Laplace approximation of the marginal log-likelihood of one subject.
     * @param i Subject index.
     * @param th Parameter vector.
     * @param[in,out] x Starting point (EBE); updated with the conditional mode.
     * @return Laplace log-likelihood, or -infinity on failure.
     */
    double laplaceSubject(int i, const Vec& th, Vec& x) const {
        const double ninf = -std::numeric_limits<double>::infinity();
        auto f = [&](const Vec& e) { return logJoint(i, e, th); };
        double f0 = f(x);
        if (!std::isfinite(f0)) return ninf;
        Mat H(q_, q_); Vec g(q_);
        auto derivs = [&](const Vec& e, double fe) -> bool {
            for (int a = 0; a < q_; ++a) {
                const double h = 1e-4 * std::max(1.0, std::abs(e[a]));
                Vec ep = e, em = e; ep[a] += h; em[a] -= h;
                const double fp = f(ep), fm = f(em);
                if (!std::isfinite(fp) || !std::isfinite(fm)) return false;
                g[a] = (fp - fm) / (2.0 * h);
                H(a, a) = (fp - 2.0 * fe + fm) / (h * h);
                for (int b = a + 1; b < q_; ++b) {
                    const double hb = 1e-4 * std::max(1.0, std::abs(e[b]));
                    Vec e1 = e, e2 = e, e3 = e, e4 = e;
                    e1[a] += h; e1[b] += hb; e2[a] += h; e2[b] -= hb; e3[a] -= h; e3[b] += hb; e4[a] -= h; e4[b] -= hb;
                    const double v = (f(e1) - f(e2) - f(e3) + f(e4)) / (4.0 * h * hb);
                    if (!std::isfinite(v)) return false;
                    H(a, b) = v; H(b, a) = v;
                }
            }
            return true;
        };
        for (int it = 0; it < 12; ++it) {
            if (!derivs(x, f0)) return ninf;
            const Mat Sg = sigmaFromNegHessian(H);
            if (Sg.size() == 0) return ninf;
            const Vec step = Sg * g;
            double t = 1.0; bool ok = false;
            for (int ls = 0; ls < 10; ++ls, t *= 0.5) {
                const Vec cand = x + t * step;
                const double fc = f(cand);
                if (std::isfinite(fc) && fc >= f0) { x = cand; f0 = fc; ok = true; break; }
            }
            if (!ok || step.cwiseAbs().maxCoeff() * t < 1e-10 * (1.0 + x.cwiseAbs().maxCoeff())) break;
        }
        if (!derivs(x, f0)) return ninf;
        Eigen::LLT<Mat> llt(-H);
        if (llt.info() != Eigen::Success) return ninf;
        const double ld = 2.0 * Mat(llt.matrixL()).diagonal().array().log().sum();
        return f0 + 0.5 * double(q_) * std::log(2.0 * std::numbers::pi) - 0.5 * ld;
    }

    /**
     * @brief Marginal log-likelihood by Laplace approximation, summed over subjects.
     * @param th Parameter vector.
     * @param[in,out] modes Starting points (EBE), one per subject; updated with the modes.
     * @return Sum of per-subject Laplace log-likelihoods.
     */
    double laplaceLogLik(const Vec& th, std::vector<Vec>& modes) const {
        std::vector<double> part(uz(n_), 0.0);
        parallelFor(n_, threads, [&](int i) { part[uz(i)] = laplaceSubject(i, th, modes[uz(i)]); });
        double s = 0.0; for (double v : part) s += v;
        return s;
    }

    // OBSERVED FIM = -Hessian (finite differences) of the marginal Laplace logLik. DETERMINISTIC
    // (no Monte Carlo noise); valid for a custom or censored likelihood and any q. The bias is
    // that of the Laplace approximation (small with informative per-subject data).
    /**
     * @brief Observed FIM from the Laplace-approximated marginal log-likelihood.
     * @param theta Parameter vector.
     * @param etaHat Starting points (EBE) per subject.
     * @return FimResult with the observed information and ok=false if any subject failed.
     *
     * The computation is deterministic (no Monte Carlo noise) and supports custom
     * or censored likelihoods and any random-effect dimension q. The bias is
     * that of the Laplace approximation.
     */
    FimResult laplaceFIM(const Vec& theta, const std::vector<Vec>& etaHat) const {
        FimResult out;
        const int D = dimTheta();
        std::vector<int> fr;
        for (int j = 0; j < D; ++j) if (!fixed_[uz(j)]) fr.push_back(j);
        const int m = static_cast<int>(fr.size());
        std::vector<Mat> parts(uz(n_), Mat::Zero(m, m));
        std::vector<char> okv(uz(n_), 1);
        Vec hs(m);
        for (int a = 0; a < m; ++a) hs[a] = 5e-4 * std::max(1.0, std::abs(theta[fr[uz(a)]]));
        parallelFor(n_, threads, [&](int i) {
            try {
                Vec x0 = etaHat[uz(i)];
                auto f = [&](const Vec& th) { Vec x = x0; return laplaceSubject(i, th, x); };
                const double f0 = f(theta);
                if (!std::isfinite(f0)) { okv[uz(i)] = 0; return; }
                std::vector<double> fp(uz(m)), fm(uz(m));
                Mat H(m, m);
                for (int a = 0; a < m; ++a) {
                    Vec tp = theta, tm = theta; tp[fr[uz(a)]] += hs[a]; tm[fr[uz(a)]] -= hs[a];
                    fp[uz(a)] = f(tp); fm[uz(a)] = f(tm);
                    if (!std::isfinite(fp[uz(a)]) || !std::isfinite(fm[uz(a)])) { okv[uz(i)] = 0; return; }
                    H(a, a) = (fp[uz(a)] - 2.0 * f0 + fm[uz(a)]) / (hs[a] * hs[a]);
                }
                for (int a = 0; a < m; ++a) for (int b = a + 1; b < m; ++b) {
                    Vec t1 = theta, t2 = theta, t3 = theta, t4 = theta;
                    t1[fr[uz(a)]] += hs[a]; t1[fr[uz(b)]] += hs[b]; t2[fr[uz(a)]] += hs[a]; t2[fr[uz(b)]] -= hs[b];
                    t3[fr[uz(a)]] -= hs[a]; t3[fr[uz(b)]] += hs[b]; t4[fr[uz(a)]] -= hs[a]; t4[fr[uz(b)]] -= hs[b];
                    const double v = (f(t1) - f(t2) - f(t3) + f(t4)) / (4.0 * hs[a] * hs[b]);
                    if (!std::isfinite(v)) { okv[uz(i)] = 0; return; }
                    H(a, b) = v; H(b, a) = v;
                }
                parts[uz(i)] = -H;
            } catch (const std::exception&) { okv[uz(i)] = 0; }
        });
        out.I = Mat::Zero(D, D); out.ok = true;
        for (int i = 0; i < n_; ++i) {
            if (!okv[uz(i)]) { out.ok = false; continue; }
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) out.I(fr[uz(a)], fr[uz(b)]) += parts[uz(i)](a, b);
        }
        return out;
    }

    // log p(y_i, phi_RE ; theta) with phi_RE FIXED (CENTERED parametrization): the prior
    // N(phi_RE; D beta, Omega) supplies the score for beta and Omega; the likelihood supplies
    // that of the error / no-random-effect parameters. Basis of the Louis formula.
    /**
     * @brief Centered complete-data log joint for one subject with fixed phi_RE.
     * @param i Subject index.
     * @param phiRE Random-effect components of phi (size q).
     * @param th Parameter vector.
     * @return log p(y_i, phi_RE ; theta), or -infinity if Omega is not positive
     *         definite or the observation log-likelihood is not finite.
     */
    double centeredLogJoint(int i, const Vec& phiRE, const Vec& th) const {
        const double ninf = -std::numeric_limits<double>::infinity();
        Eigen::LLT<Mat> llt(omegaOf(th));
        if (llt.info() != Eigen::Success) return ninf;
        const Mat L = llt.matrixL();
        const double logdet = 2.0 * L.diagonal().array().log().sum();
        const ErrParams ep = obs_.decode(th, offErr_);
        Vec base; phiOf(i, th, Vec::Zero(q_), base);
        Vec dv(q_), phiP = base;
        for (int l = 0; l < q_; ++l) { dv[l] = phiRE[l] - base[reIdx_[uz(l)]]; phiP[reIdx_[uz(l)]] = phiRE[l]; }
        const double ll = obs_.logLik(i, psiOf(phiP), ep);
        if (!std::isfinite(ll)) return ninf;
        const Vec z = L.triangularView<Eigen::Lower>().solve(dv);
        return -0.5 * (z.squaredNorm() + logdet + double(q_) * std::log(2.0 * std::numbers::pi)) + ll;
    }

    // ----------------------------------------------------------------------
    // OBSERVED FIM by the Louis formula with posterior samples
    //   I_obs = sum_i ( -E[ d2 l_c ] - Cov[ d l_c ] ),   l_c = log p(y_i, phi_i ; theta)
    // (expectations under p(chi_i | y_i, theta^); `draws[i]` = samples of eta_i).
    // Valid for ANY likelihood (custom, censored) and any random-effect dimension:
    // replaces Gauss-Hermite quadrature when q > 3. Derivatives by finite
    // differences of l_c.
    // ----------------------------------------------------------------------
    /**
     * @brief Observed FIM by the Louis formula using posterior samples.
     * @param theta Parameter vector.
     * @param draws Posterior samples of eta, one vector per subject.
     * @return FimResult with the observed information; ok=false if any subject
     *         has fewer than four usable samples or a derivative is non-finite.
     *
     * Supports any likelihood and any random-effect dimension (in particular
     * q > 3, where Gauss-Hermite quadrature becomes impractical).
     */
    FimResult louisFIM(const Vec& theta, const std::vector<std::vector<Vec>>& draws) const {
        FimResult out;
        const int D = dimTheta();
        std::vector<int> fr;
        for (int j = 0; j < D; ++j) if (!fixed_[uz(j)]) fr.push_back(j);
        const int m = static_cast<int>(fr.size());
        std::vector<Mat> parts(uz(n_), Mat::Zero(m, m));
        std::vector<char> okv(uz(n_), 1);
        parallelFor(n_, threads, [&](int i) {
            try {
                const auto& dr = draws[uz(i)];
                Vec base; phiOf(i, theta, Vec::Zero(q_), base);
                Vec hs(m);
                for (int a = 0; a < m; ++a) hs[a] = 2e-4 * std::max(1.0, std::abs(theta[fr[uz(a)]]));
                std::vector<Vec> G; Mat Hsum = Mat::Zero(m, m);
                Vec phiRE(q_);
                for (const Vec& eta : dr) {
                    for (int l = 0; l < q_; ++l) phiRE[l] = base[reIdx_[uz(l)]] + eta[l];
                    auto f = [&](const Vec& th) { return centeredLogJoint(i, phiRE, th); };
                    const double f0 = f(theta);
                    if (!std::isfinite(f0)) continue;
                    Vec g(m); Mat H(m, m);
                    std::vector<double> fp(uz(m)), fm(uz(m));
                    bool bad = false;
                    for (int a = 0; a < m && !bad; ++a) {
                        Vec tp = theta, tm = theta; tp[fr[uz(a)]] += hs[a]; tm[fr[uz(a)]] -= hs[a];
                        fp[uz(a)] = f(tp); fm[uz(a)] = f(tm);
                        if (!std::isfinite(fp[uz(a)]) || !std::isfinite(fm[uz(a)])) { bad = true; break; }
                        g[a] = (fp[uz(a)] - fm[uz(a)]) / (2.0 * hs[a]);
                        H(a, a) = (fp[uz(a)] - 2.0 * f0 + fm[uz(a)]) / (hs[a] * hs[a]);
                    }
                    for (int a = 0; a < m && !bad; ++a) for (int b2 = a + 1; b2 < m && !bad; ++b2) {
                        Vec t1 = theta, t2 = theta, t3 = theta, t4 = theta;
                        t1[fr[uz(a)]] += hs[a]; t1[fr[uz(b2)]] += hs[b2]; t2[fr[uz(a)]] += hs[a]; t2[fr[uz(b2)]] -= hs[b2];
                        t3[fr[uz(a)]] -= hs[a]; t3[fr[uz(b2)]] += hs[b2]; t4[fr[uz(a)]] -= hs[a]; t4[fr[uz(b2)]] -= hs[b2];
                        const double v = (f(t1) - f(t2) - f(t3) + f(t4)) / (4.0 * hs[a] * hs[b2]);
                        if (!std::isfinite(v)) { bad = true; break; }
                        H(a, b2) = v; H(b2, a) = v;
                    }
                    if (bad) continue;
                    G.push_back(g); Hsum += H;
                }
                const int cnt = static_cast<int>(G.size());
                if (cnt < 4) { okv[uz(i)] = 0; return; }
                Vec gm = Vec::Zero(m); for (const Vec& g : G) gm += g / double(cnt);
                Mat Cg = Mat::Zero(m, m);
                for (const Vec& g : G) Cg += (g - gm) * (g - gm).transpose() / double(cnt - 1);
                parts[uz(i)] = -Hsum / double(cnt) - Cg;
            } catch (const std::exception&) { okv[uz(i)] = 0; }
        });
        out.I = Mat::Zero(D, D); out.ok = true;
        for (int i = 0; i < n_; ++i) {
            if (!okv[uz(i)]) { out.ok = false; continue; }
            for (int a = 0; a < m; ++a) for (int b2 = 0; b2 < m; ++b2) out.I(fr[uz(a)], fr[uz(b2)]) += parts[uz(i)](a, b2);
        }
        return out;
    }

    // ----------------------------------------------------------------------
    // LINEARIZED FIM (FOCE with interaction) and logLik by linearization.
    //   y_i ~ N(E_i(theta), V_i(theta)),  E_i = f(phi^_i) + J_phi D (beta - beta^) - J_E eta^_i
    //                                     V_i = J_E Omega J_E' + R_i(f^_i)
    // with J = numerical derivative of the prediction (log scale if the error is
    // exponential) at the individual estimate eta^_i (conditional mean). Censored
    // observations are ignored. Does not apply to a custom likelihood.
    // ----------------------------------------------------------------------

    /**
     * @brief Linearized (FOCE with interaction) observed FIM and log-likelihood.
     * @param theta Parameter vector.
     * @param etaHat Individual estimates (conditional means), one per subject.
     * @return FimResult with the linearized information, total logLik, and ok
     *         flag; ok=false if any subject failed.
     *
     * Returns an empty result for a custom observation likelihood. Censored
     * observations are ignored.
     */
    FimResult linearizedFIM(const Vec& theta, const std::vector<Vec>& etaHat) const {
        FimResult out;
        if (obs_.custom()) return out;
        const int D = dimTheta();
        std::vector<Mat> parts(uz(n_), Mat::Zero(D, D));
        std::vector<double> lls(uz(n_), 0.0);
        std::vector<char> okv(uz(n_), 1);
        parallelFor(n_, threads, [&](int i) {
            try {
                const auto& S = obs_.subject(i);
                std::vector<int> idx;
                for (int j = 0; j < S.n(); ++j) if (S.cens[uz(j)] == 0) idx.push_back(j);
                const int ni = static_cast<int>(idx.size());
                if (ni == 0) return;
                Vec phiH; phiOf(i, theta, etaHat[uz(i)], phiH);
                auto predLin = [&](const Vec& phi, Vec& f, Vec& fval) {
                    const Vec psi = psiOf(phi);
                    Vec fa; obs_.predictAll(i, psi, fa);
                    f.resize(ni); fval.resize(ni);
                    for (int a = 0; a < ni; ++a) {
                        const double v = fa[idx[uz(a)]];
                        fval[a] = v;
                        f[a] = obs_.isExp(i, idx[uz(a)]) ? std::log(v) : v;
                    }
                };
                Vec fh, fv; predLin(phiH, fh, fv);
                if (!fh.allFinite()) { okv[uz(i)] = 0; return; }
                Mat Jphi(ni, p_);
                for (int a = 0; a < p_; ++a) {
                    const double h = 1e-5 * std::max(1.0, std::abs(phiH[a]));
                    Vec pp = phiH, pm = phiH; pp[a] += h; pm[a] -= h;
                    Vec fp, fpv, fm, fmv; predLin(pp, fp, fpv); predLin(pm, fm, fmv);
                    Jphi.col(a) = (fp - fm) / (2.0 * h);
                }
                Mat JE(ni, q_);
                for (int l = 0; l < q_; ++l) JE.col(l) = Jphi.col(reIdx_[uz(l)]);
                Vec yy(ni); double jac = 0.0;
                for (int a = 0; a < ni; ++a) {
                    const double yv = S.y[uz(idx[uz(a)])];
                    if (obs_.isExp(i, idx[uz(a)])) { yy[a] = std::log(yv); jac -= std::log(yv); } else yy[a] = yv;
                }
                auto Vfun = [&](const Vec& th) {
                    const ErrParams ep = obs_.decode(th, offErr_);
                    const Mat Om = omegaOf(th);
                    Mat V = JE * Om * JE.transpose();
                    for (int a = 0; a < ni; ++a) {
                        const double g = obs_.errSd(i, idx[uz(a)], fv[a], ep);
                        V(a, a) += g * g;
                    }
                    return V;
                };
                const Mat V0 = Vfun(theta);
                Eigen::LLT<Mat> llt(V0);
                if (llt.info() != Eigen::Success) { okv[uz(i)] = 0; return; }
                const Mat Vi = llt.solve(Mat::Identity(ni, ni));
                const Vec E0 = fh - JE * etaHat[uz(i)];
                const Vec res = yy - E0;
                const double logdet = 2.0 * Mat(llt.matrixL()).diagonal().array().log().sum();
                lls[uz(i)] = -0.5 * (double(ni) * std::log(2.0 * std::numbers::pi) + logdet + res.dot(Vi * res)) + jac;
                // Derivatives
                std::vector<Vec> dE(uz(D), Vec());
                std::vector<Mat> dV(uz(D));
                std::vector<int> act;
                for (int a = 0; a < D; ++a) {
                    if (fixed_[uz(a)]) continue;
                    if (a < nb_) {
                        const double x = xcoef(i, a);
                        dE[uz(a)] = Jphi.col(coefParam_[uz(a)]) * x;
                    } else if (a < offErr_ + obs_.extraOffset()) {
                        const double h = 1e-5 * std::max(1.0, std::abs(theta[a]));
                        Vec tp = theta, tm = theta; tp[a] += h; tm[a] -= h;
                        dV[uz(a)] = (Vfun(tp) - Vfun(tm)) / (2.0 * h);
                    } else continue;
                    act.push_back(a);
                }
                Mat& I = parts[uz(i)];
                std::vector<Mat> ViDV(uz(D));
                for (int a : act) if (dV[uz(a)].size() > 0) ViDV[uz(a)] = Vi * dV[uz(a)];
                for (int a : act) for (int b : act) {
                    double v = 0.0;
                    if (dE[uz(a)].size() > 0 && dE[uz(b)].size() > 0) v += dE[uz(a)].dot(Vi * dE[uz(b)]);
                    if (ViDV[uz(a)].size() > 0 && ViDV[uz(b)].size() > 0) v += 0.5 * (ViDV[uz(a)] * ViDV[uz(b)]).trace();
                    I(a, b) = v;
                }
            } catch (const std::exception&) { okv[uz(i)] = 0; }
        });
        out.I = Mat::Zero(D, D); out.ll = 0.0; out.ok = true;
        for (int i = 0; i < n_; ++i) {
            if (!okv[uz(i)]) { out.ok = false; continue; }
            out.I += parts[uz(i)]; out.ll += lls[uz(i)];
        }
        return out;
    }

private:
    /// Thread-local cache of the prior decomposition and logJoint workspace.
    struct PriorCache {
        unsigned long owner = 0; bool valid = false, ok = false;
        Vec theta; Mat L; double logdet = 0.0; ErrParams ep;
        mutable Vec wPhi, wPsi, wD, wF;     // logJoint workspace (per thread)
    };
    /// @return A fresh unique model identifier.
    static unsigned long nextId() { static std::atomic<unsigned long> c{0}; return ++c; }

    Spec spec_;
    ObsLik obs_;
    int p_, n_ = 0, q_ = 0, T_ = 0, nb_ = 0, nOm_ = 0, offOm_ = 0, offErr_ = 0;
    unsigned long id_;
    std::vector<int> reIdx_, reOf_, coefParam_, coefCov_, reCoef_, nbIdx_;
    CovModel cov_;
    std::vector<char> sdFixedRE_;
    std::vector<Vec> covVals_;
    Mat G_;
    Vec theta0_;
    std::vector<char> fixed_;

    // Per-THREAD cache of the prior decomposition (the same theta is evaluated many times)
    /// @return The calling thread's prior cache.
    static PriorCache& tl() { thread_local PriorCache c; return c; }
    /**
     * @brief Returns a cache for @p theta, recomputing it if the theta changed.
     * @param theta Parameter vector.
     * @return Reference to the thread-local cache.
     *
     * The cache is keyed by model id and exact theta value. On recomputation it
     * stores the Cholesky factor of Omega, its log-determinant, the decoded error
     * parameters, and whether Omega is positive definite.
     */
    const PriorCache& cacheFor(const Vec& theta) const {
        PriorCache& c = tl();
        if (c.valid && c.owner == id_ && c.theta.size() == theta.size() && (c.theta.array() == theta.array()).all())
            return c;
        const Mat Om = omegaOf(theta);
        Eigen::LLT<Mat> llt(Om);
        c.ok = (llt.info() == Eigen::Success);
        if (c.ok) { c.L = llt.matrixL(); c.logdet = 2.0 * c.L.diagonal().array().log().sum(); }
        c.ep = obs_.decode(theta, offErr_);
        c.theta = theta; c.owner = id_; c.valid = true;
        return c;
    }

};

} // namespace mcmckl
