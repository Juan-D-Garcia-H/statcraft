// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file multilevel.hpp
 * @brief Hierarchical / multilevel model with nested random effects.
 *
 * The model has two levels of clustering, following the same generic design as
 * the rest of the project. Observation @f$j@f$ of subject @f$i@f$, where the
 * subject belongs to cluster @f$c(i)@f$, is modelled as
 *
 * @f[
 *   \xi_c  \sim N(0, \Psi)                 \quad\text{(cluster-level random effect)}\\
 *   \eta_i \sim N(0, \Omega)               \quad\text{(subject-level random effect)}\\
 *   \phi_{c i} = \mu + \xi_c + \eta_i      \quad\text{(structural parameter of the subject)}\\
 *   y_{ij} = f(\phi_{c i}, t_{ij}) + \varepsilon_{ij}
 * @f]
 *
 * Subjects in the same cluster share @f$\xi_c@f$ and are therefore correlated,
 * which is the key feature of the multilevel model. Marginally
 * @f$\phi_{c i} \sim N(\mu, \Psi + \Omega)@f$, but the intra-cluster
 * correlation means that the marginal likelihood does @b not factorize over
 * subjects: the shared @f$\xi_c@f$ must be integrated out.
 *
 * @par Parameterization of theta
 * @f[
 *   \theta = [\,\mu\ (p)\ ;\ \text{params}_\Omega\ (m)\ ;\
 *              \text{params}_\Psi\ (m)\ ;\ \text{params}_\text{error}\ (r)\,]
 * @f]
 * For @f$\Omega@f$ and @f$\Psi@f$: diagonal means @f$m = p@f$ (log standard
 * deviations); full means @f$m = p(p+1)/2@f$ (Cholesky factors).
 *
 * @par S step (MCMC)
 * Two levels, both using the three-kernel scheme of Kuhn & Lavielle:
 *  - @b Subject @f$i@f$: target
 *    @f$\log p(y_i\mid\phi) + \log N(\phi; \mu + \xi_c, \Omega)@f$, with an
 *    independence proposal @f$N(\mu + \xi_c, \Omega)@f$.
 *  - @b Cluster @f$c@f$: target
 *    @f$\log N(\xi;0,\Psi) + \sum_{i\in c}\log N(\phi_i;\mu+\xi,\Omega)@f$,
 *    with an independence proposal equal to the exact conditional
 *    @f$N(P^{-1}b, P^{-1})@f$ (conjugate, hence exact sampling; see the note
 *    below).
 *
 * @par M step
 * Robbins-Monro on the gradient of the complete-data log density
 * @f[
 *   \sum_c \log N(\xi_c;0,\Psi) +
 *   \sum_i \big[\log N(\phi_i;\mu+\xi_{c(i)},\Omega) + \log p(y_i\mid\phi_i)\big],
 * @f]
 * computed by finite differences (generic for any @f$f@f$).
 *
 * @par Exactness of the cluster level
 * Since the cluster target is Gaussian in @f$\xi@f$, its exact conditional is
 * @f$N(P^{-1}b, P^{-1})@f$ with
 * @f[
 *   P = \Psi^{-1} + n_c\,\Omega^{-1}, \qquad
 *   b = \Omega^{-1}\sum_{i\in c}(\phi_i - \mu).
 * @f]
 * Using that conditional as the independence proposal makes the MH ratio
 * exactly one, so every step is an exact draw from @f$\xi_c\mid\phi@f$.
 *
 * Requires Eigen3, C++20, model.hpp (logLikelihood), loglik.hpp
 * (gaussHermite), mcmc.hpp and robbins.hpp.
 */

#pragma once
#include "numcast.hpp"

#include "model.hpp"     // GenericModel::logLikelihood, logMultiNormal
#include "loglik.hpp"     // gaussHermite

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// (decodeCov/encodeCov and the covariance-statistics helpers live in model.hpp)

// (logSumExp, ghGrid, gaussHermiteCached, cholWithJitter and
//  sigmaFromNegHessian live in loglik.hpp)

// ---------------------------------------------------------------------------
// Multilevel model interface (the concrete model is injected)
// ---------------------------------------------------------------------------

/**
 * @brief Abstract interface of a two-level hierarchical model.
 *
 * A concrete model provides the observation model @f$f(\phi,t)@f$ together with
 * the initial values and the target densities used by the two MCMC levels.
 * The subject and cluster random effects have dimension @f$p@f$.
 */
class MultilevelModel {
public:
    virtual ~MultilevelModel() = default;

    /** @brief Total dimension of the parameter vector @f$\theta@f$. */
    virtual int dimTheta() const = 0;
    /** @brief Dimension @f$p@f$ of @f$\phi@f$ and of @f$\xi@f$. */
    virtual int dimPhi()   const = 0;
    /** @brief Number of subjects. */
    virtual int nSubjects() const = 0;
    /** @brief Number of clusters. */
    virtual int nClusters() const = 0;
    /** @brief Cluster index of subject @f$i@f$. */
    virtual int clusterOf(int i) const = 0;

    /**
     * @brief Indices of @f$\theta@f$ that are log standard deviations.
     *
     * These entries (the @f$\Omega@f$ and @f$\Psi@f$ parameters) are the ones
     * floored by simulated annealing.
     */
    virtual std::vector<int> varianceIndices() const { return {}; }

    /** @brief Whether a closed-form M step from sufficient statistics exists. */
    virtual bool hasSufficientStats() const { return false; }

    /**
     * @brief Sufficient statistics of the complete-data model.
     * @param phis Subject effects @f$\phi_i@f$.
     * @param xis  Cluster effects @f$\xi_c@f$.
     * @param theta Current parameter vector (may be ignored).
     * @throws std::logic_error If the model does not implement them.
     */
    virtual Vec  sufficientStats(const std::vector<Vec>&, const std::vector<Vec>&, const Vec& /*theta*/) const {
        throw std::logic_error("sufficientStats not implemented");
    }

    /**
     * @brief Numerical M step for parameters without a closed form.
     *
     * See SaemModel::numericMStep. @p phis and @p xis are indexed as
     * `[chain][subject|cluster]`.
     */
    virtual void numericMStep(Vec& /*theta*/, const std::vector<std::vector<Vec>>& /*phis*/,
                              const std::vector<std::vector<Vec>>& /*xis*/, double /*gamma*/) const {}
    /** @brief Subjects belonging to cluster @p c. */
    virtual const std::vector<int>& membersOf(int c) const = 0;

    /** @brief Whether a generalized (robust) M step is available. */
    virtual bool hasGemStep() const { return false; }

    /**
     * @brief Generalized (robust) M step.
     *
     * See SaemModel::gemStep. @p phis and @p xis are indexed as
     * `[chain][subject|cluster]`.
     */
    virtual void gemStep(Vec& /*theta*/, const std::vector<std::vector<Vec>>& /*phis*/,
                         const std::vector<std::vector<Vec>>& /*xis*/, double /*gamma*/) const {}

    /**
     * @brief Map sufficient statistics back to @f$\theta@f$.
     * @throws std::logic_error If the model does not implement it.
     */
    virtual Vec  thetaFromStats(const Vec&, const Vec&) const {
        throw std::logic_error("thetaFromStats not implemented");
    }

    /** @brief Starting value of @f$\theta@f$. */
    virtual Vec initialTheta() const = 0;
    /** @brief Starting value of @f$\phi_i@f$. */
    virtual Vec initialPhi(int i) const = 0;
    /** @brief Starting value of @f$\xi_c@f$. */
    virtual Vec initialXi(int c)   const = 0;

    /**
     * @brief MCMC target of subject @f$i@f$ given the @f$\xi@f$ of its cluster.
     * @return @f$\log p(y_i\mid\phi) + \log N(\phi;\mu+\xi,\Omega)@f$.
     */
    virtual double logJointSubject(int i, const Vec& phi, const Vec& theta,
                                   const Vec& xi) const = 0;

    /**
     * @brief MCMC target of cluster @f$c@f$ given the @f$\phi@f$ of its subjects.
     * @return @f$\log N(\xi;0,\Psi) + \sum_{i\in c}\log N(\phi_i;\mu+\xi,\Omega)@f$.
     */
    virtual double logJointCluster(int c, const Vec& xi, const Vec& theta,
                                   const std::vector<Vec>& phis) const = 0;

    /** @brief Independence proposal for the subject level. */
    virtual void subjProposal(int i, const Vec& theta, const Vec& xi, Vec& mean, Mat& cov) const = 0;
    /** @brief Independence proposal for the cluster level. */
    virtual void clusProposal(int c, const Vec& theta, const std::vector<Vec>& phis,
                              Vec& mean, Mat& cov) const = 0;

    /** @brief (Averaged) gradient of the complete-data log density w.r.t. @f$\theta@f$. */
    virtual Vec mStepGradient(const Vec& theta, const std::vector<Vec>& phis,
                              const std::vector<Vec>& xis) const = 0;

    /** @brief Marginal log likelihood of one cluster (for goodness of fit / FIM). */
    virtual double logMarginalCluster(int c, const Vec& theta, int n_nodes) const = 0;

    /** @brief Total marginal log likelihood (for goodness of fit / FIM). */
    virtual double logMarginal(const Vec& theta, int n_nodes) const {
        double s = 0.0;
        for (int c = 0; c < nClusters(); ++c) s += logMarginalCluster(c, theta, n_nodes);
        return s;
    }
};

// ---------------------------------------------------------------------------
// Generic implementation: any f(phi,t[,x]) plus residual error
// ---------------------------------------------------------------------------

/**
 * @brief Generic two-level model built from a prediction function.
 *
 * The observation model is any @f$f(\phi,t)@f$ (or @f$f(\phi,t,x)@f$ with
 * subject covariates), combined with the residual-error model selected by
 * @ref Spec::residType.
 */
class GenericMultilevelModel : public MultilevelModel {
public:
    /** @brief Prediction function @f$f(\phi,t)@f$. */
    using Predict     = std::function<double(const Vec& phi, double t)>;
    /** @brief Covariate-aware prediction function @f$f(\phi,t,x)@f$ (takes priority). */
    using PredictCov  = std::function<double(const Vec& phi, double t, const Vec& x)>;

    /** @brief Construction options. */
    struct Spec {
        Predict    predict;                              ///< f(phi, t)
        PredictCov predictCov;                           ///< f(phi, t, x) [takes priority]
        std::function<Vec(int i)> subjectCovariates;     ///< i -> covariates
        int    dimPhi   = 1;                             ///< p
        bool   fullOmega = false;                        ///< full Omega (subject level)
        bool   fullPsi   = false;                        ///< full Psi (cluster level)
        resmod::ErrorKind residType = resmod::ErrorKind::Proportional;

        /// Optional starting values.
        Vec          mu0;        ///< initial mu
        Vec          sdOmega0;   ///< diagonal Omega (dim p)
        Vec          sdPsi0;     ///< diagonal Psi (dim p)
        Mat          Omega0;     ///< used when fullOmega
        Mat          Psi0;       ///< used when fullPsi
        resmod::Pres resid0;     ///< initial residual parameters
    };

    /**
     * @brief Build the model from a specification and the data.
     * @param spec      Model specification.
     * @param t         Observation times, per subject.
     * @param y         Observations, per subject.
     * @param clusterOf Cluster label of each subject.
     * @throws std::invalid_argument On inconsistent dimensions or missing
     *         prediction function.
     */
    GenericMultilevelModel(Spec spec,
                           std::vector<std::vector<double>> t,
                           std::vector<std::vector<double>> y,
                           std::vector<int> clusterOf)
      : spec_(std::move(spec)),
        t_(std::move(t)), y_(std::move(y)),
        p_(spec_.dimPhi),
        n_(static_cast<int>(t_.size())),
        residType_(spec_.residType),
        like_(makeLike(spec_, t_, y_))
    {
        if (p_ <= 0) throw std::invalid_argument("dimPhi > 0");
        if (!spec_.predict && !spec_.predictCov) throw std::invalid_argument("predict is required");
        if (static_cast<int>(y_.size()) != n_) throw std::invalid_argument("t and y have different sizes");
        if (static_cast<int>(clusterOf.size()) != n_)
            throw std::invalid_argument("clusterOf must have one value per subject");

        // Normalize cluster labels to 0..C-1 and build the membership lists.
        std::vector<int> labels = clusterOf;
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        C_ = static_cast<int>(labels.size());
        clusterOf_.resize(uz(n_));
        members_.assign(uz(C_), {});
        for (int i = 0; i < n_; ++i) {
            auto it = std::find(labels.begin(), labels.end(), clusterOf[uz(i)]);
            if (it == labels.end()) throw std::invalid_argument("unknown cluster");
            const int c = static_cast<int>(it - labels.begin());
            clusterOf_[uz(i)] = c;
            members_[uz(c)].push_back(i);
        }
    }

    // --- MultilevelModel ----------------------------------------------------
    int dimTheta() const override { return p_ + nCov(spec_.fullOmega) + nCov(spec_.fullPsi) + nResid(); }
    int dimPhi()   const override { return p_; }
    int nSubjects() const override { return n_; }
    int nClusters() const override { return C_; }
    int clusterOf(int i) const override { return clusterOf_[uz(i)]; }
    const std::vector<int>& membersOf(int c) const override { return members_[uz(c)]; }

    Vec initialTheta() const override {
        Vec th = Vec::Zero(dimTheta());
        int off = 0;
        if (spec_.mu0.size() == p_) th.segment(off, p_) = spec_.mu0;
        off += p_;
        // Omega
        if (!spec_.fullOmega) {
            Vec sd = (spec_.sdOmega0.size() == p_) ? spec_.sdOmega0 : Vec::Constant(p_, 1.0);
            for (int j = 0; j < p_; ++j) th[off + j] = std::log(std::max(1e-12, sd[j]));
        } else {
            Mat Om = (spec_.Omega0.rows() == p_) ? spec_.Omega0 : Mat::Identity(p_, p_);
            encodeCov(th, off, Om, p_, true);
        }
        off += nCov(spec_.fullOmega);
        // Psi
        if (!spec_.fullPsi) {
            Vec sd = (spec_.sdPsi0.size() == p_) ? spec_.sdPsi0 : Vec::Constant(p_, 1.0);
            for (int j = 0; j < p_; ++j) th[off + j] = std::log(std::max(1e-12, sd[j]));
        } else {
            Mat Ps = (spec_.Psi0.rows() == p_) ? spec_.Psi0 : Mat::Identity(p_, p_);
            encodeCov(th, off, Ps, p_, true);
        }
        off += nCov(spec_.fullPsi);
        encodeResid(th, off, spec_.resid0);
        return th;
    }
    Vec initialPhi(int) const override {
        Vec th = initialTheta();
        return th.head(p_);
    }
    Vec initialXi(int) const override { return Vec::Zero(p_); }

    // NOTE: these use a mutable cache of the last theta, so they are not
    // thread-safe across different theta values.
    double logJointSubject(int i, const Vec& phi, const Vec& theta, const Vec& xi) const override {
        const Cache& c = cacheFor(theta);
        if (!c.okO) return -std::numeric_limits<double>::infinity();
        const double ll = like_.logLikelihood(i, phi, c.pres);
        if (!std::isfinite(ll)) return -std::numeric_limits<double>::infinity();
        return logNormChol(phi, c.mu + xi, c.LO, c.ldO) + ll;
    }

    double logJointCluster(int cl, const Vec& xi, const Vec& theta,
                           const std::vector<Vec>& phis) const override {
        const Cache& c = cacheFor(theta);
        if (!c.okO || !c.okP) return -std::numeric_limits<double>::infinity();
        double lp = logNormChol(xi, Vec::Zero(p_), c.LP, c.ldP);
        const Vec m = c.mu + xi;
        for (int i : membersOf(cl)) {
            const double v = logNormChol(phis[uz(i)], m, c.LO, c.ldO);
            if (!std::isfinite(v)) return -std::numeric_limits<double>::infinity();
            lp += v;
        }
        return lp;
    }

    void subjProposal(int, const Vec& theta, const Vec& xi, Vec& mean, Mat& cov) const override {
        Vec mu; Mat Omega, Psi; resmod::Pres pres;
        unpack(theta, mu, Omega, Psi, pres);
        mean = mu + xi; cov = Omega;
    }

    // Exact conditional of xi_c given the phi values of the cluster
    // (Gaussian conjugacy).
    void clusProposal(int c, const Vec& theta, const std::vector<Vec>& phis,
                      Vec& mean, Mat& cov) const override {
        Vec mu; Mat Omega, Psi; resmod::Pres pres;
        unpack(theta, mu, Omega, Psi, pres);
        Eigen::LLT<Mat> llP(Psi), llO(Omega);
        if (llP.info() != Eigen::Success || llO.info() != Eigen::Success) {
            mean = Vec::Zero(p_); cov = Psi; return;
        }
        const Mat Pinv_Oinv = llO.solve(Mat::Identity(p_, p_));        // Omega^-1
        const Mat Psi_inv   = llP.solve(Mat::Identity(p_, p_));        // Psi^-1
        const int nc = static_cast<int>(membersOf(c).size());
        Mat prec = Psi_inv + double(nc) * Pinv_Oinv;
        prec = 0.5 * (prec + prec.transpose());
        // Sum of (phi_i - mu) over the cluster.
        Vec rhs = Vec::Zero(p_);
        for (int i : membersOf(c)) rhs += (phis[uz(i)] - mu);
        rhs = Pinv_Oinv * rhs;                                          // Omega^-1 * sum
        Eigen::LLT<Mat> llPrec(prec);
        if (llPrec.info() != Eigen::Success) { mean = Vec::Zero(p_); cov = Psi; return; }
        cov   = prec.inverse();
        cov   = 0.5 * (cov + cov.transpose());
        mean  = llPrec.solve(rhs);
    }

    Vec mStepGradient(const Vec& theta, const std::vector<Vec>& phis,
                      const std::vector<Vec>& xis) const override {
        auto total = [&](const Vec& th) -> double {
            Vec mu; Mat Omega, Psi; resmod::Pres pres;
            unpack(th, mu, Omega, Psi, pres);
            // Cholesky ONCE per theta (previously once per subject and evaluation).
            Eigen::LLT<Mat> lO(Omega), lP(Psi);
            if (lO.info() != Eigen::Success || lP.info() != Eigen::Success)
                return -std::numeric_limits<double>::infinity();
            const Mat LO = lO.matrixL(), LP = lP.matrixL();
            const double ldO = 2.0 * LO.diagonal().array().log().sum();
            const double ldP = 2.0 * LP.diagonal().array().log().sum();
            double s = 0.0;
            for (int i = 0; i < n_; ++i) {
                const double ll = like_.logLikelihood(i, phis[uz(i)], pres);
                if (!std::isfinite(ll)) return -std::numeric_limits<double>::infinity();
                s += logNormChol(phis[uz(i)], mu + xis[uz(clusterOf_[uz(i)])], LO, ldO) + ll;
            }
            for (int c = 0; c < C_; ++c)
                s += logNormChol(xis[uz(c)], Vec::Zero(p_), LP, ldP);
            return s;
        };

        const int q = static_cast<int>(theta.size());
        Vec g(q);
        const double f0 = total(theta);
        for (int j = 0; j < q; ++j) {
            const double h = 1e-6 * std::max(1.0, std::abs(theta[j]));
            Vec tp = theta, tm = theta;
            tp[j] += h; tm[j] -= h;
            const double fp = total(tp), fm = total(tm);
            if (std::isfinite(fp) && std::isfinite(fm))
                g[j] = (fp - fm) / (2.0 * h) / double(n_);
            else if (std::isfinite(fp) && std::isfinite(f0))
                g[j] = (fp - f0) / h / double(n_);
            else if (std::isfinite(fm) && std::isfinite(f0))
                g[j] = (f0 - fm) / h / double(n_);
            else
                g[j] = 0.0;
        }
        return g;
    }

    // --- Marginal likelihood (GH integration over the shared xi) -------------
    // log ∫ N(xi;0,Psi) prod_{i in c} [ ∫ p(y_i|phi) N(phi; mu+xi, Omega) dphi ] dxi
    double logMarginalCluster(int c, const Vec& theta, int n_nodes) const override {
        Vec mu; Mat Omega, Psi; resmod::Pres pres;
        unpack(theta, mu, Omega, Psi, pres);
        Eigen::LLT<Mat> llt(Psi);
        if (llt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
        const Mat Lp = llt.matrixL();

        Vec xg, wg;
        gaussHermiteCached(n_nodes, xg, wg);

        std::vector<double> terms;
        ghGrid(xg, wg, p_, [&](const Vec& z, double wprod) {
            if (!(wprod > 0.0)) return;
            const Vec xi = std::sqrt(2.0) * (Lp * z);
            double inner = 0.0;
            for (int i : membersOf(c)) inner += logMargSubjectGivenXi(i, theta, xi, n_nodes);
            if (std::isfinite(inner)) terms.push_back(std::log(wprod) + inner);
        });
        if (terms.empty()) return -std::numeric_limits<double>::infinity();
        return logSumExp(terms) - 0.5 * double(p_) * std::log(std::numbers::pi);
    }

    // ADAPTIVE GH over the SHARED cluster xi.
    //
    // The fixed GH above integrates over the prior N(0,Psi). But the posterior
    // of xi given the cluster (n_c subjects) is much narrower and its mode is
    // far from 0; with fixed nodes the integral --and especially its
    // derivatives-- is unstable (kinks when the dominant node changes), which
    // ruins the FIM/SE.
    //
    // Solution: change variables to the Gaussian centered at the posterior mode
    // of xi with covariance equal to the inverse of the negative Hessian of
    //   h(xi) = log N(xi;0,Psi) + sum_{i in c} log m_i(xi),
    //   m_i(xi) = ∫ p(y_i|phi) N(phi; mu+xi, Omega) dphi.
    double clusterLogIntegrand(int c, const Vec& theta, const Vec& xi, int n_nodes) const {
        const Cache& cc = cacheFor(theta);
        if (!cc.okP) return -std::numeric_limits<double>::infinity();
        double lp = logNormChol(xi, Vec::Zero(p_), cc.LP, cc.ldP);
        if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();
        for (int i : membersOf(c)) {
            const double lm = logMargSubjectGivenXi(i, theta, xi, n_nodes);
            if (!std::isfinite(lm)) return -std::numeric_limits<double>::infinity();
            lp += lm;
        }
        return lp;
    }

    // Center (mode and Cholesky of the covariance) of a cluster quadrature.
    // It is computed ONCE at theta_hat and kept FIXED while differentiating
    // with respect to theta: re-optimizing the mode at every theta perturbation
    // injects noise into the numerical Hessian and destabilizes the FIM/SE.
    struct ClusterQuad { Vec mode; Mat L; double logdet = 0.0; };

    ClusterQuad clusterQuadrature(int c, const Vec& theta, int n_nodes) const {
        Vec mu; Mat Omega, Psi; resmod::Pres pres;
        unpack(theta, mu, Omega, Psi, pres);

        auto h = [&](const Vec& xi) { return clusterLogIntegrand(c, theta, xi, n_nodes); };
        // Initial simplex at the scale of Psi; rescues the case h(0) = -inf.
        const Vec sc = 0.3 * Psi.diagonal().cwiseMax(1e-12).cwiseSqrt();
        const Vec mode = nelderMeadEx([&](const Vec& xi) { return -h(xi); },
                                      Vec::Zero(p_), sc, 1000).x;
        const Mat H = numericalHessian(h, mode);
        const Mat Sigma = sigmaFromNegHessian(H);     // spectral floor (H may be indefinite)

        ClusterQuad q;
        q.mode = mode;
        if (Sigma.size() == 0 || !cholWithJitter(Sigma, q.L, q.logdet)) {
            q.L = cholLower(Psi); q.logdet = logDetFromChol(q.L);     // fallback: the prior
        }
        return q;
    }

    // Evaluates log p(y_c | theta) with a FIXED quadrature center.
    double logMarginalClusterQuad(int c, const Vec& theta, const ClusterQuad& q,
                                  int n_nodes) const {
        Vec xg, wg;
        gaussHermiteCached(n_nodes, xg, wg);
        std::vector<double> terms;
        ghGrid(xg, wg, p_, [&](const Vec& z, double wprod) {
            if (!(wprod > 0.0)) return;
            const Vec xi = q.mode + std::sqrt(2.0) * (q.L * z);
            const Vec dv = q.L.triangularView<Eigen::Lower>().solve(xi - q.mode);
            const double logq = -0.5 * double(p_) * std::log(2.0 * std::numbers::pi) - 0.5 * q.logdet
                                - 0.5 * dv.squaredNorm();
            const double hv = clusterLogIntegrand(c, theta, xi, n_nodes);
            if (std::isfinite(hv)) terms.push_back(std::log(wprod) + hv - logq);
        });
        if (terms.empty()) return -std::numeric_limits<double>::infinity();
        return logSumExp(terms) - 0.5 * double(p_) * std::log(std::numbers::pi);
    }

    double logMarginalClusterAdaptive(int c, const Vec& theta, int n_nodes) const {
        const ClusterQuad q = clusterQuadrature(c, theta, n_nodes);
        return logMarginalClusterQuad(c, theta, q, n_nodes);
    }

    double logMarginalAdaptive(const Vec& theta, int n_nodes) const {
        double s = 0.0;
        for (int c = 0; c < C_; ++c) s += logMarginalClusterAdaptive(c, theta, n_nodes);
        return s;
    }

    // --- Access -------------------------------------------------------------
    /** @brief Split @f$\theta@f$ into @f$\mu@f$, @f$\Omega@f$, @f$\Psi@f$ and residual parameters. */
    void unpack(const Vec& theta, Vec& mu, Mat& Omega, Mat& Psi, resmod::Pres& pres) const {
        int off = 0;
        mu = theta.segment(off, p_); off += p_;
        Omega = decodeCov(theta, off, p_, spec_.fullOmega); off += nCov(spec_.fullOmega);
        Psi   = decodeCov(theta, off, p_, spec_.fullPsi);   off += nCov(spec_.fullPsi);
        decodeResid(theta, off, pres);
    }
    /** @brief Residual-error type used by this model. */
    const resmod::ErrorKind& residType() const { return residType_; }

    // --- Closed-form M step ------------------------------------------------
    //   phi_i = mu + xi_c(i) + eps_i,  eps ~ N(0,Omega),  xi_c ~ N(0,Psi)
    //   s = [ E_i[phi-xi] | E_i[(phi-xi)(phi-xi)'] | E_c[xi xi'] | error ]
    //   mu = E[phi-xi], Omega = E[rr'] - mu mu', Psi = E[xi xi']
    bool hasSufficientStats() const override { return true; }

    Vec sufficientStats(const std::vector<Vec>& phis, const std::vector<Vec>& xis, const Vec& /*theta*/) const override {
        const int nO = covStatsSize(p_, spec_.fullOmega), nP = covStatsSize(p_, spec_.fullPsi);
        const int nr = like_.nResidStats();
        Vec s = Vec::Zero(p_ + nO + nP + nr);
        for (int i = 0; i < n_; ++i) {
            const Vec r = phis[uz(i)] - xis[uz(clusterOf_[uz(i)])];
            s.head(p_) += r / double(n_);
            addCovStats(s, p_, r, spec_.fullOmega, 1.0 / double(n_));
        }
        for (int c = 0; c < C_; ++c) addCovStats(s, p_ + nO, xis[uz(c)], spec_.fullPsi, 1.0 / double(C_));
        s.tail(nr) = like_.residualStats(phis);
        return s;
    }

    Vec thetaFromStats(const Vec& s, const Vec& theta) const override {
        const int nO = covStatsSize(p_, spec_.fullOmega), nP = covStatsSize(p_, spec_.fullPsi);
        const int nr = like_.nResidStats();
        Vec th = theta;
        const Vec m = s.head(p_);
        th.head(p_) = m;
        encodeCov(th, p_, covFromStats(s, p_, p_, spec_.fullOmega, m), p_, spec_.fullOmega);
        encodeCov(th, p_ + nCov(spec_.fullOmega),
                  covFromStats(s, p_ + nO, p_, spec_.fullPsi, Vec::Zero(p_)), p_, spec_.fullPsi);
        like_.residualThetaFromStats(s.tail(nr), th, p_ + nCov(spec_.fullOmega) + nCov(spec_.fullPsi));
        (void)nP;
        return th;
    }

    std::vector<int> varianceIndices() const override {
        std::vector<int> idx = covDiagIndices(p_, p_, spec_.fullOmega);
        const std::vector<int> ip = covDiagIndices(p_ + nCov(spec_.fullOmega), p_, spec_.fullPsi);
        idx.insert(idx.end(), ip.begin(), ip.end());
        return idx;
    }

    /**
     * @brief Number of nodes for the INNER GH quadrature (integral over
     *        @f$\phi@f$ given @f$\xi@f$).
     *
     * If 0, the same value as the outer quadrature is used. Increasing it
     * improves the accuracy of the multilevel marginal at the cost of time
     * (cost @f$O(\text{innerNodes}^p)@f$ per subject and outer node).
     */
    int innerNodes = 0;

private:
    Spec spec_;
    std::vector<std::vector<double>> t_, y_;
    int p_, n_, C_ = 0;
    resmod::ErrorKind residType_;
    std::vector<int> clusterOf_;             // subject -> cluster
    std::vector<std::vector<int>> members_;  // cluster -> subjects
    GenericModel like_;                      // only for logLikelihood

    // Cache of the factorization of (mu, Omega, Psi, pres) for the last theta.
    struct Cache {
        Vec theta, mu; Mat Omega, Psi, LO, LP;
        double ldO = 0.0, ldP = 0.0; resmod::Pres pres;
        bool okO = false, okP = false, valid = false;
    };
    mutable Cache cc_;

    const Cache& cacheFor(const Vec& theta) const {
        if (cc_.valid && cc_.theta.size() == theta.size() &&
            (cc_.theta.array() == theta.array()).all())
            return cc_;
        unpack(theta, cc_.mu, cc_.Omega, cc_.Psi, cc_.pres);
        Eigen::LLT<Mat> lO(cc_.Omega), lP(cc_.Psi);
        cc_.okO = (lO.info() == Eigen::Success);
        cc_.okP = (lP.info() == Eigen::Success);
        if (cc_.okO) { cc_.LO = lO.matrixL(); cc_.ldO = 2.0 * cc_.LO.diagonal().array().log().sum(); }
        if (cc_.okP) { cc_.LP = lP.matrixL(); cc_.ldP = 2.0 * cc_.LP.diagonal().array().log().sum(); }
        cc_.theta = theta; cc_.valid = true;
        return cc_;
    }

    // log N(x; m, LL') with the Cholesky factor already computed.
    double logNormChol(const Vec& x, const Vec& m, const Mat& L, double logdet) const {
        const Vec d = L.triangularView<Eigen::Lower>().solve(x - m);
        return -0.5 * (d.squaredNorm() + logdet + double(p_) * std::log(2.0 * std::numbers::pi));
    }

    static GenericModel makeLike(const Spec& spec,
                                 const std::vector<std::vector<double>>& t,
                                 const std::vector<std::vector<double>>& y) {
        GenericModel::Spec ls;
        ls.predict = spec.predict;
        ls.predictCov = spec.predictCov;
        ls.subjectCovariates = spec.subjectCovariates;
        ls.dimPhi = spec.dimPhi;
        ls.residType = spec.residType;
        ls.mu0 = spec.mu0;
        return GenericModel(std::move(ls), t, y);
    }

    int nCov(bool full) const { return full ? p_ * (p_ + 1) / 2 : p_; }
    int nResid() const { return resmod::nParams(residType_); }

    // ∫ p(y_i|phi) N(phi; mu+xi, Omega) dphi via GH (base = that Gaussian).
    double logMargSubjectGivenXi(int i, const Vec& theta, const Vec& xi, int n_nodes) const {
        const Cache& c = cacheFor(theta);
        if (!c.okO) return -std::numeric_limits<double>::infinity();
        const Vec m = c.mu + xi;
        const Mat& L = c.LO;

        const int nn = (innerNodes > 0) ? innerNodes : n_nodes;
        Vec xg, wg;
        gaussHermiteCached(nn, xg, wg);
        std::vector<double> terms;
        ghGrid(xg, wg, p_, [&](const Vec& z, double wprod) {
            if (!(wprod > 0.0)) return;
            const Vec phi = m + std::sqrt(2.0) * (L * z);
            const double ll = like_.logLikelihood(i, phi, c.pres);
            if (std::isfinite(ll)) terms.push_back(std::log(wprod) + ll);
        });
        if (terms.empty()) return -std::numeric_limits<double>::infinity();
        return logSumExp(terms) - 0.5 * double(p_) * std::log(std::numbers::pi);
    }

    void decodeResid(const Vec& th, int off, resmod::Pres& pres) const {
        switch (residType_) {
            default: break;
            case resmod::ErrorKind::Constant:    pres = resmod::Pres(std::exp(th[off]), 0.0); break;
            case resmod::ErrorKind::Exponential:  pres = resmod::Pres(std::exp(th[off]), 0.0); break;
            case resmod::ErrorKind::Proportional: pres = resmod::Pres(0.0, std::exp(th[off])); break;
            case resmod::ErrorKind::Combined:
                pres = resmod::Pres(std::exp(th[off]), std::exp(th[off + 1])); break;
        }
    }
    void encodeResid(Vec& th, int off, const resmod::Pres& pres) const {
        const double eps = 1e-12;
        switch (residType_) {
            default: break;
            case resmod::ErrorKind::Constant:
            case resmod::ErrorKind::Exponential:
                th[off] = std::log(std::max(eps, pres.a)); break;
            case resmod::ErrorKind::Proportional:
                th[off] = std::log(std::max(eps, pres.b)); break;
            case resmod::ErrorKind::Combined:
                th[off]     = std::log(std::max(eps, pres.a));
                th[off + 1] = std::log(std::max(eps, pres.b));
                break;
        }
    }
};

// ---------------------------------------------------------------------------
// Multilevel SAEM control (all fields optional)
// ---------------------------------------------------------------------------

/** @brief Control options for @ref SAEMMultilevel. */
struct MultilevelControl {
    MCMCControl mcmc;                 ///< SUBJECT-level kernels (3 K&L kernels)
    MCMCControl mcmcCluster = independenceOnly();  ///< CLUSTER level (exact)
    RMControl   rm;                   ///< Robbins-Monro step of the M step
    int  n_saem_iter   = 300;         ///< number of SAEM iterations
    int  n_mcmc_sweeps = 3;           ///< MCMC sweeps per iteration
    int  n_mcmc_burnin = 40;          ///< burn-in sweeps at the first iteration
    bool verbose = false;             ///< print progress
    int  print_every = 50;            ///< progress-printing period

    /// Simulated annealing (K&L): shrinkage floor of Omega and Psi in the M step.
    bool      simulated_annealing = false;
    SAControl sa;

    /// Subject-level RW/Gibbs steps: start at the Omega scale and adapt online.
    bool adapt_rw = true;

    /// M step: Robbins-Monro on the gradient or classic SAEM (sufficient statistics).
    MStep mstep = MStep::Auto;
    ClassicStepControl step;

    /// Independent MCMC chains per subject and per cluster (the gradient /
    /// statistics are averaged over chains).
    int n_chains = 1;

    /// Threads for the subject / cluster sweeps (1 = serial; 0 = all).
    int n_threads = 1;

    /// The cluster level is conjugate: only the independence kernel with the
    /// exact conditional proposal is used, so every step is an exact draw.
    static MCMCControl independenceOnly() {
        MCMCControl c;
        c.w_rw = 0.0; c.w_gibbs = 0.0; c.w_indep = 1.0;
        return c;
    }
};

// ---------------------------------------------------------------------------
// SAEM for the multilevel model (two levels of random effects)
// ---------------------------------------------------------------------------

/**
 * @brief Stochastic-approximation EM for a @ref MultilevelModel.
 *
 * Each iteration runs, for every independent chain, a subject-level MCMC sweep
 * (@f$\phi_i\mid\xi@f$) followed by a cluster-level sweep
 * (@f$\xi_c\mid\phi@f$), and then updates @f$\theta@f$ with the configured
 * M step (sufficient statistics, generalized step, or Robbins-Monro on the
 * finite-difference gradient).
 *
 * The object owns MCMC kernels whose log-density lambdas capture `this`, so it
 * is non-copyable and non-movable.
 */
class SAEMMultilevel {
public:
    /**
     * @brief Construct the sampler.
     * @param model Multilevel model (must outlive this object).
     * @param ctrl  Control options.
     * @param seed  Base random seed.
     * @throws std::invalid_argument On invalid control values or unsupported
     *         M-step configuration.
     */
    SAEMMultilevel(const MultilevelModel& model, MultilevelControl ctrl = {}, unsigned seed = 12345)
      : model_(model), ctrl_(std::move(ctrl)), seed_(seed), rm_(effectiveRM(ctrl_))
    {
        if (ctrl_.n_chains < 1) throw std::invalid_argument("n_chains >= 1");
        theta_ = model_.initialTheta();
        if (theta_.size() != model_.dimTheta())
            throw std::invalid_argument("initialTheta() has the wrong dimension");
        rm_.init(theta_);
        ctrl_.step.sa_len = ctrl_.simulated_annealing ? saPhaseLength(ctrl_.sa, ctrl_.n_saem_iter) : 0;
        varIdx_ = model_.varianceIndices();
        if (ctrl_.mstep == MStep::SufficientStats && !model_.hasSufficientStats())
            throw std::invalid_argument("the model does not implement sufficient statistics");
        useStats_ = (ctrl_.mstep == MStep::SufficientStats) ||
                    (ctrl_.mstep == MStep::Auto && model_.hasSufficientStats());
        useGem_ = !useStats_ && ctrl_.mstep == MStep::Gradient && model_.hasGemStep();

        const int n = model_.nSubjects(), C = model_.nClusters();
        const int p = model_.dimPhi(), M = ctrl_.n_chains;
        phis_.assign(uz(M), std::vector<Vec>(uz(n)));
        xis_.assign(uz(M), std::vector<Vec>(uz(C)));
        subjSamp_.resize(uz(M)); clusSamp_.resize(uz(M));

        if (ctrl_.mcmcCluster.rw_sigma.size() != p)
            ctrl_.mcmcCluster.rw_sigma = Vec::Constant(p, 0.5);
        MCMCControl mcs = ctrl_.mcmc;
        if (ctrl_.adapt_rw) mcs.adapt_online = true;

        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < n; ++i) phis_[uz(m)][uz(i)] = model_.initialPhi(i);
            for (int c = 0; c < C; ++c) xis_[uz(m)][uz(c)] = model_.initialXi(c);
            // One chain per subject (reads theta_ and xis_[m][c(i)] via `this`).
            subjSamp_[uz(m)].reserve(uz(n));
            for (int i = 0; i < n; ++i) {
                const int c = model_.clusterOf(i);
                auto logp = [this, i, c, m](const Vec& phi) {
                    return model_.logJointSubject(i, phi, theta_, xis_[uz(m)][uz(c)]);
                };
                subjSamp_[uz(m)].push_back(std::make_unique<MCMCKuhnLavielle>(
                    logp, p, seed_ + static_cast<unsigned>(i) + 1000003u * static_cast<unsigned>(m), mcs));
            }
            // One chain per cluster (reads theta_ and phis_[m] via `this`).
            clusSamp_[uz(m)].reserve(uz(C));
            for (int c = 0; c < C; ++c) {
                auto logp = [this, c, m](const Vec& xi) {
                    return model_.logJointCluster(c, xi, theta_, phis_[uz(m)]);
                };
                clusSamp_[uz(m)].push_back(std::make_unique<MCMCKuhnLavielle>(
                    logp, p, seed_ + 1000003u * unsigned(c + 1) + 7u * unsigned(m), ctrl_.mcmcCluster));
            }
        }
    }

    // The lambdas capture `this`: copying/moving would leave dangling pointers.
    SAEMMultilevel(const SAEMMultilevel&) = delete;
    SAEMMultilevel& operator=(const SAEMMultilevel&) = delete;
    SAEMMultilevel(SAEMMultilevel&&) = delete;
    SAEMMultilevel& operator=(SAEMMultilevel&&) = delete;

    /** @brief Whether the closed-form sufficient-statistics M step is used. */
    bool usesSufficientStats() const { return useStats_; }

    /** @brief Run the full SAEM loop. */
    void run() {
        const int M = ctrl_.n_chains;
        for (int k = 0; k < ctrl_.n_saem_iter; ++k) {
            const int sweeps = (k == 0) ? (ctrl_.n_mcmc_burnin + ctrl_.n_mcmc_sweeps)
                                        :  ctrl_.n_mcmc_sweeps;
            refreshSubjectProposals(k == 0 && ctrl_.adapt_rw);
            for (int m = 0; m < M; ++m) sweepChain(m, sweeps);

            // M step (+ simulated-annealing floor on the variances).
            const Vec theta_old = theta_;
            if (useStats_) {
                Vec S = model_.sufficientStats(phis_[0], xis_[0], theta_);
                for (int m = 1; m < M; ++m) S += model_.sufficientStats(phis_[uz(m)], xis_[uz(m)], theta_);
                if (M > 1) S /= double(M);
                const double gam = ctrl_.step.gamma(k, ctrl_.n_saem_iter);
                if (k == 0 || s_.size() != S.size()) s_ = S;
                else s_ += gam * (S - s_);
                Vec th = model_.thetaFromStats(s_, theta_);
                model_.numericMStep(th, phis_, xis_, gam);
                if (!th.allFinite()) throw std::runtime_error("closed-form M step: theta is not finite");
                if (ctrl_.simulated_annealing) saApplyVarianceFloor(th, theta_old, varIdx_, k, ctrl_.sa);
                theta_ = th;
            } else if (useGem_) {
                Vec th = theta_;
                model_.gemStep(th, phis_, xis_, ctrl_.step.gamma(k, ctrl_.n_saem_iter));
                if (!th.allFinite()) throw std::runtime_error("generalized M step: theta is not finite");
                if (ctrl_.simulated_annealing) saApplyVarianceFloor(th, theta_old, varIdx_, k, ctrl_.sa);
                theta_ = th;
            } else {
                Vec grad = model_.mStepGradient(theta_, phis_[0], xis_[0]);
                for (int m = 1; m < M; ++m) grad += model_.mStepGradient(theta_, phis_[uz(m)], xis_[uz(m)]);
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
            if (ctrl_.verbose && (k % ctrl_.print_every == 0 || k == ctrl_.n_saem_iter - 1)) {
                std::cout << "  it " << std::setw(4) << k
                          << "  gamma " << std::scientific << std::setprecision(2)
                          << ((useStats_ || useGem_) ? ctrl_.step.gamma(k, ctrl_.n_saem_iter) : rm_.currentStep())
                          << "  theta " << std::fixed << std::setprecision(4)
                          << theta_.transpose();
                if (!useStats_ && !useGem_ && ctrl_.rm.averaging) std::cout << "   | avg " << rm_.estimate().transpose();
                std::cout << "\n";
            }
        }
    }

    /** @brief Current parameter estimate. */
    const Vec& theta() const { return theta_; }
    /** @brief Polyak-averaged estimate (the current value for stats/gem steps). */
    Vec thetaAveraged() const { return (useStats_ || useGem_) ? theta_ : rm_.estimate(); }
    /** @brief Trace of theta over iterations. */
    const std::vector<Vec>& thetaTrace() const { return thetaTrace_; }
    /** @brief Subject effects of chain 0. */
    const std::vector<Vec>& phis() const { return phis_[0]; }   // chain 0
    /** @brief Convergence diagnostics derived from the theta trace. */
    ConvergenceInfo convergence() const {
        ConvergenceInfo ci;
        ci.iterations = static_cast<int>(thetaTrace_.size());
        if (ci.iterations >= 8) {
            const Vec& a = thetaTrace_.back();
            const Vec& b = thetaTrace_[static_cast<size_t>(0.75 * (ci.iterations - 1))];
            ci.drift = ((a - b).cwiseAbs().array() / (a.cwiseAbs().array() + 0.05)).matrix();
            ci.maxDrift = ci.drift.maxCoeff();
        }
        return ci;
    }
    /** @brief Cluster effects of chain 0. */
    const std::vector<Vec>& xis()  const { return xis_[0]; }

    /** @brief Conditional draws for one cluster (chi follows membersOf(c) order). */
    struct ClusterDraw { std::vector<Vec> chi; Vec xi; };
    /** @brief Conditional estimates returned by @ref conditionalSample. */
    struct CondEstimate { std::vector<Vec> phiMean, phiSd, xiMean, xiSd; std::vector<std::vector<ClusterDraw>> draws; };

    /**
     * @brief Conditional means and SDs of @f$\phi_i@f$ and @f$\xi_c@f$
     *        given the data, with @f$\theta@f$ fixed at @p theta_hat.
     *
     * Uses all chains. After the call, theta() == @p theta_hat.
     *
     * @param theta_hat Fixed parameter vector.
     * @param n_samples Number of retained samples per chain (>= 2).
     * @param n_burn    Burn-in sweeps (>= 0).
     * @param thin      Thinning between samples (>= 1).
     * @param keep_draws If > 0, number of draws to store per cluster.
     * @throws std::invalid_argument On invalid arguments.
     */
    CondEstimate conditionalSample(const Vec& theta_hat, int n_samples = 1000,
                                   int n_burn = 200, int thin = 2, int keep_draws = 0) {
        if (n_samples < 2 || thin < 1 || n_burn < 0) throw std::invalid_argument("conditionalSample: invalid arguments");
        theta_ = theta_hat;
        const int n = model_.nSubjects(), C = model_.nClusters(), p = model_.dimPhi(), M = ctrl_.n_chains;
        refreshSubjectProposals(false);
        // Freeze adaptation after burn-in.
        for (int m = 0; m < M; ++m) sweepChain(m, n_burn);
        for (int m = 0; m < M; ++m)
            for (auto& sp : subjSamp_[uz(m)]) { MCMCControl c = sp->control(); c.adapt_online = false; sp->setControl(c); }

        CondEstimate out;
        if (keep_draws > 0) out.draws.assign(uz(C), std::vector<ClusterDraw>());
        const long stride = std::max<long>(1, (long(n_samples) * M) / std::max(1, keep_draws));
        long counter = 0;
        out.phiMean.assign(uz(n), Vec::Zero(p)); out.phiSd.assign(uz(n), Vec::Zero(p));
        out.xiMean.assign(uz(C), Vec::Zero(p));  out.xiSd.assign(uz(C), Vec::Zero(p));
        std::vector<Vec> m2p(uz(n), Vec::Zero(p)), m2x(uz(C), Vec::Zero(p));
        long cnt = 0;
        for (int s = 0; s < n_samples; ++s)
            for (int m = 0; m < M; ++m) {
                refreshSubjectProposals(false);
                sweepChain(m, thin);
                ++cnt;
                if (keep_draws > 0 && (counter++ % stride) == 0) {
                    for (int c = 0; c < C; ++c) {
                        ClusterDraw cd; cd.xi = xis_[uz(m)][uz(c)];
                        for (int i : model_.membersOf(c)) cd.chi.push_back(phis_[uz(m)][uz(i)]);
                        out.draws[uz(c)].push_back(std::move(cd));
                    }
                }
                for (int i = 0; i < n; ++i) {
                    const Vec d = phis_[uz(m)][uz(i)] - out.phiMean[uz(i)];
                    out.phiMean[uz(i)] += d / double(cnt);
                    m2p[uz(i)] += d.cwiseProduct(phis_[uz(m)][uz(i)] - out.phiMean[uz(i)]);
                }
                for (int c = 0; c < C; ++c) {
                    const Vec d = xis_[uz(m)][uz(c)] - out.xiMean[uz(c)];
                    out.xiMean[uz(c)] += d / double(cnt);
                    m2x[uz(c)] += d.cwiseProduct(xis_[uz(m)][uz(c)] - out.xiMean[uz(c)]);
                }
            }
        for (int i = 0; i < n; ++i) out.phiSd[uz(i)] = (m2p[uz(i)] / double(cnt - 1)).cwiseSqrt();
        for (int c = 0; c < C; ++c) out.xiSd[uz(c)]  = (m2x[uz(c)] / double(cnt - 1)).cwiseSqrt();
        return out;
    }

private:
    const MultilevelModel& model_;
    MultilevelControl ctrl_;
    unsigned seed_;

    Vec theta_;
    RobbinsMonro rm_;
    bool useStats_ = false;
    bool useGem_ = false;
    Vec  s_;
    std::vector<Vec> thetaTrace_;
    std::vector<int> varIdx_;
    std::vector<std::vector<Vec>> phis_, xis_;                       // [chain][subject|cluster]
    std::vector<std::vector<std::unique_ptr<MCMCKuhnLavielle>>> subjSamp_, clusSamp_;

    // With SA enabled, the Polyak average starts after the exploratory phase.
    static RMControl effectiveRM(const MultilevelControl& c) {
        RMControl r = c.rm;
        if (c.simulated_annealing && r.averaging)
            r.avg_start = std::max(r.avg_start, saPhaseLength(c.sa, c.n_saem_iter) + 1);
        return r;
    }

    // Subject-level independence proposals (depend on the xi of each chain).
    void refreshSubjectProposals(bool init_steps) {
        const int n = model_.nSubjects(), p = model_.dimPhi();
        for (int m = 0; m < ctrl_.n_chains; ++m)
            for (int i = 0; i < n; ++i) {
                Vec mean; Mat cov;
                model_.subjProposal(i, theta_, xis_[uz(m)][uz(model_.clusterOf(i))], mean, cov);
                subjSamp_[uz(m)][uz(i)]->setIndependence(mean, cov);
                if (init_steps) {
                    const Vec sd = cov.diagonal().cwiseMax(1e-12).cwiseSqrt();
                    subjSamp_[uz(m)][uz(i)]->setSteps((2.4 / std::sqrt(double(p))) * sd, 2.4 * sd);
                }
            }
    }

    // One sweep of chain m: phi_i | xi, then xi_c | phi (proposal = conditional).
    void sweepChain(int m, int sweeps) {
        const int n = model_.nSubjects(), C = model_.nClusters();
        parallelFor(n, ctrl_.n_threads, [&](int i) {
            phis_[uz(m)][uz(i)] = subjSamp_[uz(m)][uz(i)]->sampleFrom(phis_[uz(m)][uz(i)], sweeps);
        });
        parallelFor(C, ctrl_.n_threads, [&](int c) {
            Vec mean; Mat cov;
            model_.clusProposal(c, theta_, phis_[uz(m)], mean, cov);
            clusSamp_[uz(m)][uz(c)]->setIndependence(mean, cov);
            xis_[uz(m)][uz(c)] = clusSamp_[uz(m)][uz(c)]->sampleFrom(xis_[uz(m)][uz(c)], sweeps);
        });
    }
};

} // namespace mcmckl
