/**
 * @file nlme_ml.hpp
 * @brief Two-level hierarchical nonlinear mixed-effects model with all features
 *        of the single-level model (nlme.hpp).
 *
 * The model is defined by
 * @code
 *   phi_i = D_i beta + xi_c(i) + eta_i      (only for parameters with a random effect)
 *   xi_c  ~ N(0, Psi)    (cluster effect)
 *   eta_i ~ N(0, Omega)  (subject effect)
 *   psi_i = h(phi_i);  y_ij | psi_i ~ observation model (obslik.hpp)
 * @endcode
 *
 * The variable sampled per subject is chi_i, the random-effect part of phi_i,
 * whose prior is N(base_i + xi_c, Omega). For each cluster, xi_c is sampled
 * using its exact Gaussian conditional as the independence proposal.
 *
 * The full parameter vector is
 * @code
 *   theta = [ theta_core (beta, Omega, error/extra: see NlmeModel) ; Psi (CovModel) ]
 * @endcode
 *
 * Closed-form M-step: the population part (GLS estimate of beta, Omega and the
 * error parameters) reuses NlmeModel with phi_RE - xi_c for each subject, while
 * Psi = E[xi xi']. A numerical step handles coefficients without a random
 * effect and extra parameters. The observed FIM is computed with Louis'
 * formula per cluster.
 *
 * Requires: Eigen3, C++20, nlme.hpp, multilevel.hpp.
 */

#pragma once
#include "numcast.hpp"

#include "nlme.hpp"
#include "multilevel.hpp"

namespace mcmckl {

/**
 * @brief Computes log(∫ exp(f(x)) dx) by adaptive Gauss-Hermite quadrature.
 *
 * The mode is located with Nelder-Mead starting from @p x0 and the covariance
 * is taken from the robust negative Hessian (-H)^{-1}. The quadrature grid is
 * then centered at the mode.
 *
 * @param f        Integrand in log scale.
 * @param x0       Starting point for the mode search.
 * @param sc       Per-coordinate scale used by the mode search and as a
 *                 fallback covariance if the Hessian is not usable.
 * @param n_nodes  Number of Gauss-Hermite nodes per dimension.
 * @param nm_iter  Maximum number of Nelder-Mead iterations.
 * @return The log integral, or -infinity if no admissible region is found
 *         (mode search failure, invalid covariance, or no finite terms).
 */
inline double logIntegralAdaptive(const std::function<double(const Vec&)>& f, const Vec& x0, const Vec& sc,
                                  int n_nodes, int nm_iter = 200) {
    const double ninf = -std::numeric_limits<double>::infinity();
    const int d = static_cast<int>(x0.size());
    auto neg = [&](const Vec& x) { const double v = f(x); return std::isfinite(v) ? -v : std::numeric_limits<double>::infinity(); };
    Vec mode;
    try { mode = nelderMeadEx(neg, x0, sc, nm_iter, 1e-9, 0).x; } catch (const std::exception&) { return ninf; }
    const Mat H = numericalHessian(f, mode, 1e-4);
    Mat Sigma = sigmaFromNegHessian(H), L; double logdetS = 0.0;
    if (Sigma.size() == 0 || !cholWithJitter(Sigma, L, logdetS)) {
        Mat D = Mat::Zero(d, d); for (int j = 0; j < d; ++j) D(j, j) = sc[j] * sc[j];
        if (!cholWithJitter(D, L, logdetS)) return ninf;
    }
    Vec xg, wg; gaussHermiteCached(n_nodes, xg, wg);
    std::vector<double> terms;
    ghGrid(xg, wg, d, [&](const Vec& z, double wprod) {
        if (!(wprod > 0.0)) return;
        const Vec x = mode + std::sqrt(2.0) * (L * z);
        const double v = f(x);
        if (!std::isfinite(v)) return;
        // integral = sum w * exp(f(x) + |z|^2) * 2^{d/2} |L|
        terms.push_back(std::log(wprod) + v + z.squaredNorm() + 0.5 * double(d) * std::log(2.0) + 0.5 * logdetS);
    });
    if (terms.empty()) return ninf;
    return logSumExp(terms);
}

/**
 * @brief Two-level hierarchical nonlinear mixed-effects model.
 *
 * Extends MultilevelModel with per-cluster random effects xi_c ~ N(0, Psi) on
 * top of the subject-level random effects. See the file header for the full
 * model specification and parameter layout.
 */
class NlmeMultilevel : public MultilevelModel {
public:
    /**
     * @brief Model specification.
     */
    struct Spec {
        NlmeModel::Spec core;                 ///< Subject-level model (same as the single-level model).
        std::vector<int> psiBlock;            ///< Size p: same id = full block in Psi (empty = diagonal).
        Vec sdPsi0;                           ///< Size p (empty = 0.3).
    };

    int threads = 1;                          ///< Number of worker threads used by parallel loops.
    int innerNodes = 9, outerNodes = 9;       ///< Quadrature nodes for the marginal log-likelihood.

    /**
     * @brief Constructs the hierarchical model.
     *
     * Normalizes the cluster labels, builds the per-cluster member lists and
     * the Psi covariance model, and initializes theta from the core model plus
     * the initial Psi standard deviations.
     *
     * @param spec       Model specification.
     * @param obs        Observation likelihood.
     * @param clusterOf  Cluster label for each subject; must contain one entry
     *                   per subject.
     * @throws std::invalid_argument If @p clusterOf does not contain exactly one
     *         entry per subject.
     */
    NlmeMultilevel(Spec spec, ObsLik obs, std::vector<int> clusterOf)
      : core_(spec.core, std::move(obs)), id_(nextId())
    {
        n_ = core_.nSubjects(); q_ = core_.q();
        if (static_cast<int>(clusterOf.size()) != n_) throw std::invalid_argument("clusterOf must have one value per subject");
        std::vector<int> labels = clusterOf;
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        C_ = static_cast<int>(labels.size());
        clusterOf_.resize(uz(n_)); members_.assign(uz(C_), {});
        for (int i = 0; i < n_; ++i) {
            clusterOf_[uz(i)] = static_cast<int>(std::lower_bound(labels.begin(), labels.end(), clusterOf[uz(i)]) - labels.begin());
            members_[uz(clusterOf_[uz(i)])].push_back(i);
        }
        // Psi over the q random effects
        std::vector<int> blk(uz(q_));
        for (int l = 0; l < q_; ++l) {
            const int j = core_.reIdx()[uz(l)];
            blk[uz(l)] = spec.psiBlock.empty() ? l : spec.psiBlock[uz(j)];
        }
        psiCov_ = CovModel::fromBlocks(blk);
        if (spec.psiBlock.empty()) psiCov_ = CovModel::diagonal(q_);
        offPsi_ = core_.dimTheta(); nPsi_ = psiCov_.nParams();
        Vec sdp = Vec::Constant(q_, 0.3);
        if (spec.sdPsi0.size() == core_.p()) for (int l = 0; l < q_; ++l) sdp[l] = spec.sdPsi0[core_.reIdx()[uz(l)]];
        theta0_ = Vec::Zero(offPsi_ + nPsi_);
        theta0_.head(offPsi_) = core_.initialTheta();
        psiCov_.encodeSd(theta0_, offPsi_, sdp);
        fixed_ = core_.fixedMask(); fixed_.resize(uz(offPsi_ + nPsi_), 0);
    }

    /// @return Reference to the underlying subject-level model.
    NlmeModel& core() { return core_; }
    /// @return Const reference to the underlying subject-level model.
    const NlmeModel& core() const { return core_; }
    /// @return Offset of the Psi parameters in theta.
    int offPsi() const { return offPsi_; }
    /// @return Number of parameters in the Psi covariance model.
    int nPsi() const { return nPsi_; }
    /// @return Psi covariance matrix decoded from @p th.
    Mat psiOf(const Vec& th) const { return psiCov_.decode(th, offPsi_); }
    /// @return Covariance model used for Psi.
    const CovModel& psiModel() const { return psiCov_; }
    /// @return Fixed-parameter mask (non-zero marks a fixed parameter).
    const std::vector<char>& fixedMask() const { return fixed_; }

    // --- MultilevelModel interface --------------------------------------------
    /// @return Total number of model parameters (core plus Psi).
    int dimTheta() const override { return offPsi_ + nPsi_; }
    /// @return Dimension q of the per-subject random-effect vector.
    int dimPhi()   const override { return q_; }
    /// @return Number of subjects.
    int nSubjects() const override { return n_; }
    /// @return Number of clusters.
    int nClusters() const override { return C_; }
    /// @return Cluster index of subject @p i.
    int clusterOf(int i) const override { return clusterOf_[uz(i)]; }
    /// @return Subject indices belonging to cluster @p c.
    const std::vector<int>& membersOf(int c) const override { return members_[uz(c)]; }
    /// @return Initial parameter vector.
    Vec initialTheta() const override { return theta0_; }
    /// @return Initial subject-level random effects for subject @p i.
    Vec initialPhi(int i) const override { return core_.baseRE(i, theta0_); }
    /// @return Initial cluster effect (zero vector).
    Vec initialXi(int) const override { return Vec::Zero(q_); }

    /// @return Indices of the variance parameters (core plus Psi).
    std::vector<int> varianceIndices() const override {
        std::vector<int> v = core_.varianceIndices();
        const auto w = psiCov_.varianceIndices(offPsi_);
        v.insert(v.end(), w.begin(), w.end());
        return v;
    }

    double logJointSubject(int i, const Vec& chi, const Vec& theta, const Vec& xi) const override {
        const double ninf = -std::numeric_limits<double>::infinity();
        Vec phi; core_.phiOf(i, theta, Vec::Zero(q_), phi);
        Vec d(q_);
        for (int l = 0; l < q_; ++l) d[l] = chi[l] - phi[core_.reIdx()[uz(l)]] - xi[l];
        const double pr = core_.logNormalOmega(d, theta);
        if (!std::isfinite(pr)) return ninf;
        for (int l = 0; l < q_; ++l) phi[core_.reIdx()[uz(l)]] = chi[l];
        const double ll = core_.logLikPhi(i, phi, theta);
        return std::isfinite(ll) ? pr + ll : ninf;
    }

    double logJointCluster(int c, const Vec& xi, const Vec& theta, const std::vector<Vec>& phis) const override {
        const double ninf = -std::numeric_limits<double>::infinity();
        Eigen::LLT<Mat> lp(psiOf(theta));
        if (lp.info() != Eigen::Success) return ninf;
        const Mat Lp = lp.matrixL();
        const Vec z = Lp.triangularView<Eigen::Lower>().solve(xi);
        double s = -0.5 * (z.squaredNorm() + 2.0 * Lp.diagonal().array().log().sum() + double(q_) * std::log(2.0 * std::numbers::pi));
        for (int i : members_[uz(c)]) {
            const Vec d = phis[uz(i)] - core_.baseRE(i, theta) - xi;
            const double v = core_.logNormalOmega(d, theta);
            if (!std::isfinite(v)) return ninf;
            s += v;
        }
        return s;
    }

    void subjProposal(int i, const Vec& theta, const Vec& xi, Vec& mean, Mat& cov) const override {
        mean = core_.baseRE(i, theta) + xi;
        cov = core_.omegaOf(theta);
    }

    // Exact conditional of xi_c | chi:  P = Psi^-1 + n_c Omega^-1,
    // b = Omega^-1 sum (chi_i - base_i).
    void clusProposal(int c, const Vec& theta, const std::vector<Vec>& phis, Vec& mean, Mat& cov) const override {
        const Mat Psi = psiOf(theta), Om = core_.omegaOf(theta);
        Eigen::LLT<Mat> lP(Psi), lO(Om);
        if (lP.info() != Eigen::Success || lO.info() != Eigen::Success) { mean = Vec::Zero(q_); cov = Psi; return; }
        const Mat Oinv = lO.solve(Mat::Identity(q_, q_)), Pinv = lP.solve(Mat::Identity(q_, q_));
        Vec sum = Vec::Zero(q_);
        for (int i : members_[uz(c)]) sum += phis[uz(i)] - core_.baseRE(i, theta);
        const Mat prec = Pinv + double(members_[uz(c)].size()) * Oinv;
        Eigen::LLT<Mat> lq(prec);
        if (lq.info() != Eigen::Success) { mean = Vec::Zero(q_); cov = Psi; return; }
        cov = lq.solve(Mat::Identity(q_, q_)); cov = 0.5 * (cov + cov.transpose());
        mean = lq.solve(Oinv * sum);
    }

    // --- Gradient M step (legacy), centered parameterization ---------------------
    Vec mStepGradient(const Vec& theta, const std::vector<Vec>& phis, const std::vector<Vec>& xis) const override {
        const int D = dimTheta();
        Vec g = Vec::Zero(D);
        auto F = [&](const Vec& th) {
            return completeLogLik(th, phis, xis, nullptr) / double(n_);
        };
        for (int j = 0; j < D; ++j) {
            if (fixed_[uz(j)]) continue;
            const double h = 1e-6 * std::max(1.0, std::abs(theta[j]));
            Vec tp = theta, tm = theta; tp[j] += h; tm[j] -= h;
            const double fp = F(tp), fm = F(tm);
            if (std::isfinite(fp) && std::isfinite(fm)) g[j] = (fp - fm) / (2.0 * h);
        }
        return g;
    }

    // --- Robust generalized M step: damped Newton on the centered complete logLik ----------------
    bool hasGemStep() const override { return true; }
    void gemStep(Vec& theta, const std::vector<std::vector<Vec>>& phis, const std::vector<std::vector<Vec>>& xis, double gamma) const override {
        const int D = dimTheta(), M = static_cast<int>(phis.size());
        std::vector<int> fr;
        for (int j = 0; j < D; ++j) if (!fixed_[uz(j)]) fr.push_back(j);
        const int m = static_cast<int>(fr.size());
        if (m == 0) return;
        auto F = [&](const Vec& x) -> double {
            Vec th = theta;
            for (int a = 0; a < m; ++a) th[fr[uz(a)]] = x[a];
            double s = 0.0;
            for (int c = 0; c < M; ++c) {
                const double v = completeLogLik(th, phis[uz(c)], xis[uz(c)], nullptr);
                if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
                s += v;
            }
            return -s / double(std::max(1, n_) * M);
        };
        Vec x0(m);
        for (int a = 0; a < m; ++a) x0[a] = theta[fr[uz(a)]];
        const Vec xs = NlmeModel::newtonMin(F, x0, 4, 2.0);
        for (int a = 0; a < m; ++a) theta[fr[uz(a)]] += gamma * (xs[a] - x0[a]);
    }

    // --- Closed-form M step ----------------------------------------------------------
    //   s = [ pop (M | S2 of chi - xi) | E[xi xi'] (q x q) | error ]
    bool hasSufficientStats() const override { return true; }

    Vec sufficientStats(const std::vector<Vec>& phis, const std::vector<Vec>& xis, const Vec& theta) const override {
        std::vector<Vec> eff(uz(n_)), psis(uz(n_));
        Vec phi;
        for (int i = 0; i < n_; ++i) {
            eff[uz(i)] = phis[uz(i)] - xis[uz(clusterOf_[uz(i)])];
            core_.phiOf(i, theta, Vec::Zero(q_), phi);
            for (int l = 0; l < q_; ++l) phi[core_.reIdx()[uz(l)]] = phis[uz(i)][l];
            psis[uz(i)] = core_.psiOf(phi);
        }
        const Vec sp = core_.popStats(eff);
        const Vec se = core_.errStats(psis, theta);
        Vec s(sp.size() + q_ * q_ + se.size());
        s.head(sp.size()) = sp;
        Mat Sx = Mat::Zero(q_, q_);
        for (int c = 0; c < C_; ++c) Sx += xis[uz(c)] * xis[uz(c)].transpose() / double(C_);
        for (int l = 0; l < q_; ++l) for (int m = 0; m < q_; ++m) s[sp.size() + l * q_ + m] = Sx(l, m);
        if (se.size() > 0) s.tail(se.size()) = se;
        return s;
    }

    Vec thetaFromStats(const Vec& s, const Vec& theta) const override {
        const int np = core_.nPopStats(), ne = core_.nStatsErr();
        Vec sc(np + ne);
        sc.head(np) = s.head(np);
        if (ne > 0) sc.tail(ne) = s.tail(ne);
        Vec th = theta;
        th.head(offPsi_) = core_.thetaFromStats(sc, theta.head(offPsi_));
        Mat Sx(q_, q_);
        for (int l = 0; l < q_; ++l) for (int m = 0; m < q_; ++m) Sx(l, m) = s[np + l * q_ + m];
        psiCov_.encodeFromMatrix(th, offPsi_, 0.5 * (Sx + Sx.transpose()), {});
        return th;
    }

    void numericMStep(Vec& theta, const std::vector<std::vector<Vec>>& phis, const std::vector<std::vector<Vec>>&, double gamma) const override {
        Vec thc = theta.head(offPsi_);
        core_.numericMStepPhiRE(thc, phis, gamma);
        theta.head(offPsi_) = thc;
    }

    // Complete log p(y, chi, xi ; theta) (centered parameterization). If `lik_only` != nullptr,
    // the likelihood part is also returned.
    double completeLogLik(const Vec& th, const std::vector<Vec>& phis, const std::vector<Vec>& xis, double* lik_only) const {
        const double ninf = -std::numeric_limits<double>::infinity();
        Eigen::LLT<Mat> lO(core_.omegaOf(th)), lP(psiOf(th));
        if (lO.info() != Eigen::Success || lP.info() != Eigen::Success) return ninf;
        const Mat LO = lO.matrixL(), LP = lP.matrixL();
        const double ldO = 2.0 * LO.diagonal().array().log().sum(), ldP = 2.0 * LP.diagonal().array().log().sum();
        const double k2 = double(q_) * std::log(2.0 * std::numbers::pi);
        const ErrParams ep = core_.obs().decode(th, core_.offError());
        double s = 0.0, sl = 0.0;
        Vec base, phi;
        for (int i = 0; i < n_; ++i) {
            core_.phiOf(i, th, Vec::Zero(q_), base);
            phi = base;
            Vec d(q_);
            for (int l = 0; l < q_; ++l) { d[l] = phis[uz(i)][l] - base[core_.reIdx()[uz(l)]] - xis[uz(clusterOf_[uz(i)])][l]; phi[core_.reIdx()[uz(l)]] = phis[uz(i)][l]; }
            const double ll = core_.obs().logLik(i, core_.psiOf(phi), ep);
            if (!std::isfinite(ll)) return ninf;
            const Vec z = LO.triangularView<Eigen::Lower>().solve(d);
            s += -0.5 * (z.squaredNorm() + ldO + k2) + ll; sl += ll;
        }
        for (int c = 0; c < C_; ++c) {
            const Vec z = LP.triangularView<Eigen::Lower>().solve(xis[uz(c)]);
            s += -0.5 * (z.squaredNorm() + ldP + k2);
        }
        if (lik_only) *lik_only = sl;
        return s;
    }

    // --- Marginal log likelihood per cluster (adaptive nested quadrature) ------------------
    double logMarginalCluster(int c, const Vec& theta, int n_nodes) const override {
        const double ninf = -std::numeric_limits<double>::infinity();
        const Mat Psi = psiOf(theta), Om = core_.omegaOf(theta);
        Eigen::LLT<Mat> lp(Psi);
        if (lp.info() != Eigen::Success) return ninf;
        const Mat Lp = lp.matrixL();
        const double ldP = 2.0 * Lp.diagonal().array().log().sum();
        const Vec scXi = 0.3 * Psi.diagonal().cwiseMax(1e-12).cwiseSqrt();
        const Vec scChi = 0.3 * Om.diagonal().cwiseMax(1e-12).cwiseSqrt();
        const int nIn = (innerNodes > 0) ? innerNodes : n_nodes;
        auto inner = [&](int i, const Vec& xi) {
            const Vec bre = core_.baseRE(i, theta);
            Vec phi0; core_.phiOf(i, theta, Vec::Zero(q_), phi0);
            auto f = [&](const Vec& chi) {
                Vec d = chi - bre - xi;
                const double pr = core_.logNormalOmega(d, theta);
                if (!std::isfinite(pr)) return -std::numeric_limits<double>::infinity();
                Vec phi = phi0;
                for (int l = 0; l < q_; ++l) phi[core_.reIdx()[uz(l)]] = chi[l];
                const double ll = core_.logLikPhi(i, phi, theta);
                return std::isfinite(ll) ? pr + ll : -std::numeric_limits<double>::infinity();
            };
            return logIntegralAdaptive(f, bre + xi, scChi, nIn, 120);
        };
        auto h = [&](const Vec& xi) {
            const Vec z = Lp.triangularView<Eigen::Lower>().solve(xi);
            double v = -0.5 * (z.squaredNorm() + ldP + double(q_) * std::log(2.0 * std::numbers::pi));
            for (int i : members_[uz(c)]) { const double m = inner(i, xi); if (!std::isfinite(m)) return -std::numeric_limits<double>::infinity(); v += m; }
            return v;
        };
        return logIntegralAdaptive(h, Vec::Zero(q_), scXi, (outerNodes > 0) ? outerNodes : n_nodes, 200);
    }
    double logMarginal(const Vec& theta, int n_nodes) const override {
        std::vector<double> part(uz(C_), 0.0);
        parallelFor(C_, threads, [&](int c) { part[uz(c)] = logMarginalCluster(c, theta, n_nodes); });
        double s = 0.0; for (double v : part) s += v;
        return s;
    }

    // --- Observed FIM via Louis' formula (per cluster) ------------------------------
    // draws[c] = posterior samples (chi of the members in membersOf(c) order, xi)
    template <class Draws>
    NlmeModel::FimResult louisFIM(const Vec& theta, const Draws& draws) const {
        NlmeModel::FimResult out;
        const int D = dimTheta();
        std::vector<int> fr;
        for (int j = 0; j < D; ++j) if (!fixed_[uz(j)]) fr.push_back(j);
        const int m = static_cast<int>(fr.size());
        std::vector<Mat> parts(uz(C_), Mat::Zero(m, m));
        std::vector<char> okv(uz(C_), 1);
        parallelFor(C_, threads, [&](int c) {
            try {
                Vec hs(m);
                for (int a = 0; a < m; ++a) hs[a] = 2e-4 * std::max(1.0, std::abs(theta[fr[uz(a)]]));
                std::vector<Vec> G; Mat Hsum = Mat::Zero(m, m);
                const auto& mem = members_[uz(c)];
                for (const auto& dr : draws[uz(c)]) {
                    auto f = [&](const Vec& th) {
                        Eigen::LLT<Mat> lO(core_.omegaOf(th)), lP(psiOf(th));
                        if (lO.info() != Eigen::Success || lP.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
                        const Mat LO = lO.matrixL(), LP = lP.matrixL();
                        const double ldO = 2.0 * LO.diagonal().array().log().sum(), ldP = 2.0 * LP.diagonal().array().log().sum();
                        const double k2 = double(q_) * std::log(2.0 * std::numbers::pi);
                        const ErrParams ep = core_.obs().decode(th, core_.offError());
                        const Vec zc = LP.triangularView<Eigen::Lower>().solve(dr.xi);
                        double s = -0.5 * (zc.squaredNorm() + ldP + k2);
                        Vec base, phi;
                        for (size_t a = 0; a < mem.size(); ++a) {
                            const int i = mem[a];
                            core_.phiOf(i, th, Vec::Zero(q_), base);
                            phi = base; Vec dv(q_);
                            for (int l = 0; l < q_; ++l) { dv[l] = dr.chi[a][l] - base[core_.reIdx()[uz(l)]] - dr.xi[l]; phi[core_.reIdx()[uz(l)]] = dr.chi[a][l]; }
                            const double ll = core_.obs().logLik(i, core_.psiOf(phi), ep);
                            if (!std::isfinite(ll)) return -std::numeric_limits<double>::infinity();
                            const Vec z = LO.triangularView<Eigen::Lower>().solve(dv);
                            s += -0.5 * (z.squaredNorm() + ldO + k2) + ll;
                        }
                        return s;
                    };
                    const double f0 = f(theta);
                    if (!std::isfinite(f0)) continue;
                    Vec g(m); Mat H(m, m); bool bad = false;
                    std::vector<double> fp(uz(m)), fm(uz(m));
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
                if (cnt < 4) { okv[uz(c)] = 0; return; }
                Vec gm = Vec::Zero(m); for (const Vec& g : G) gm += g / double(cnt);
                Mat Cg = Mat::Zero(m, m);
                for (const Vec& g : G) Cg += (g - gm) * (g - gm).transpose() / double(cnt - 1);
                parts[uz(c)] = -Hsum / double(cnt) - Cg;
            } catch (const std::exception&) { okv[uz(c)] = 0; }
        });
        out.I = Mat::Zero(D, D); out.ok = true;
        for (int c = 0; c < C_; ++c) {
            if (!okv[uz(c)]) { out.ok = false; continue; }
            for (int a = 0; a < m; ++a) for (int b2 = 0; b2 < m; ++b2) out.I(fr[uz(a)], fr[uz(b2)]) += parts[uz(c)](a, b2);
        }
        return out;
    }

    // See NlmeModel::boundaryMask: it also includes the Psi parameters.
    std::vector<char> boundaryMask(const Vec& th, double tol = 1e-3) const {
        std::vector<char> m(uz(dimTheta()), 0);
        const std::vector<char> mo = core_.boundaryMask(th, tol);
        for (size_t j = 0; j < mo.size(); ++j) m[j] = mo[j];
        const Mat Ps = psiOf(th);
        for (const auto& c : psiCov_.comps()) {
            bool deg = false;
            for (int l : c.members) deg = deg || (std::sqrt(std::max(Ps(l, l), 0.0)) < tol * std::max(std::abs(th[core_.reIdx()[uz(l)]]), 1e-8));
            if (deg) for (int k = 0; k < c.nPar; ++k) m[uz(offPsi_ + c.off + k)] = 1;
        }
        return m;
    }
    void markFixed(const std::vector<char>& m) { for (size_t j = 0; j < m.size() && j < fixed_.size(); ++j) if (m[j]) fixed_[j] = 1; }

    // --- LINEARIZED FIM (FOCE with interaction), hierarchical, and logLik per linearization ----------
    //   y_c ~ N(E_c, V_c),  V_c = blockdiag_i(J_iE Omega J_iE' + R_i) + J_E Psi J_E'   (J_E = stacked J_iE)
    //   E_i = f(phi^_i) - J_iE (chi^_i - base_i) [+ J_phi D (beta - beta^)]
    // at the conditional estimates (chi^_i, xi^_c already included in chi^). It ignores censored
    // observations and does not apply to a custom likelihood.
    NlmeModel::FimResult linearizedFIM(const Vec& theta, const std::vector<Vec>& chiHat) const {
        NlmeModel::FimResult out;
        if (core_.obs().custom()) return out;
        const int D = dimTheta(), p = core_.p(), nb = core_.nb();
        const ObsLik& ob = core_.obs();
        std::vector<Mat> parts(uz(C_), Mat::Zero(D, D));
        std::vector<double> lls(uz(C_), 0.0);
        std::vector<char> okv(uz(C_), 1);
        const int errEnd = core_.offError() + ob.extraOffset();
        parallelFor(C_, threads, [&](int c) {
            try {
                const auto& mem = members_[uz(c)];
                const int m = static_cast<int>(mem.size());
                std::vector<std::vector<int>> idx(uz(m));
                std::vector<int> off(uz(m + 1), 0);
                for (int a = 0; a < m; ++a) {
                    const auto& S = ob.subject(mem[uz(a)]);
                    for (int j = 0; j < S.n(); ++j) if (S.cens[uz(j)] == 0) idx[uz(a)].push_back(j);
                    off[uz(a + 1)] = off[uz(a)] + static_cast<int>(idx[uz(a)].size());
                }
                const int N = off[uz(m)];
                if (N == 0) return;
                Mat Jphi(N, p), JE(N, q_);
                Vec fh(N), fv(N), yy(N), E0(N);
                double jac = 0.0;
                std::vector<char> isEx(uz(N), 0);
                for (int a = 0; a < m; ++a) {
                    const int i = mem[uz(a)], ni = static_cast<int>(idx[uz(a)].size());
                    if (ni == 0) continue;
                    Vec phiH; core_.phiOf(i, theta, Vec::Zero(q_), phiH);
                    const Vec bre = core_.baseRE(i, theta);
                    for (int l = 0; l < q_; ++l) phiH[core_.reIdx()[uz(l)]] = chiHat[uz(i)][l];
                    auto predLin = [&](const Vec& phi, Vec& f, Vec& fval) {
                        Vec fa; ob.predictAll(i, core_.psiOf(phi), fa);
                        f.resize(ni); fval.resize(ni);
                        for (int b = 0; b < ni; ++b) { const double v = fa[idx[uz(a)][uz(b)]]; fval[b] = v; f[b] = ob.isExp(i, idx[uz(a)][uz(b)]) ? std::log(v) : v; }
                    };
                    Vec f0, fv0; predLin(phiH, f0, fv0);
                    if (!f0.allFinite()) { okv[uz(c)] = 0; return; }
                    for (int pp = 0; pp < p; ++pp) {
                        const double h = 1e-5 * std::max(1.0, std::abs(phiH[pp]));
                        Vec php = phiH, phm = phiH; php[pp] += h; phm[pp] -= h;
                        Vec fp, fpv, fm, fmv; predLin(php, fp, fpv); predLin(phm, fm, fmv);
                        Jphi.block(off[uz(a)], pp, ni, 1) = (fp - fm) / (2.0 * h);
                    }
                    for (int l = 0; l < q_; ++l) JE.block(off[uz(a)], l, ni, 1) = Jphi.block(off[uz(a)], core_.reIdx()[uz(l)], ni, 1);
                    Vec chiBase(q_); for (int l = 0; l < q_; ++l) chiBase[l] = chiHat[uz(i)][l] - bre[l];
                    const Vec Ei = f0 - JE.block(off[uz(a)], 0, ni, q_) * chiBase;
                    for (int b = 0; b < ni; ++b) {
                        fh[off[uz(a)] + b] = f0[b]; fv[off[uz(a)] + b] = fv0[b]; E0[off[uz(a)] + b] = Ei[b];
                        const double yv = ob.subject(i).y[uz(idx[uz(a)][uz(b)])];
                        isEx[uz(off[uz(a)] + b)] = ob.isExp(i, idx[uz(a)][uz(b)]);
                        if (isEx[uz(off[uz(a)] + b)]) { yy[off[uz(a)] + b] = std::log(yv); jac -= std::log(yv); } else yy[off[uz(a)] + b] = yv;
                    }
                }
                auto Vfun = [&](const Vec& th) {
                    const ErrParams ep = ob.decode(th, core_.offError());
                    const Mat Om = core_.omegaOf(th), Ps = psiOf(th);
                    Mat V = JE * Ps * JE.transpose();
                    for (int a = 0; a < m; ++a) {
                        const int ni = off[uz(a + 1)] - off[uz(a)];
                        if (ni == 0) continue;
                        const Mat Ja = JE.block(off[uz(a)], 0, ni, q_);
                        V.block(off[uz(a)], off[uz(a)], ni, ni) += Ja * Om * Ja.transpose();
                        for (int b = 0; b < ni; ++b) { const double g = ob.errSd(mem[uz(a)], idx[uz(a)][uz(b)], fv[off[uz(a)] + b], ep); V(off[uz(a)] + b, off[uz(a)] + b) += g * g; }
                    }
                    return V;
                };
                const Mat V0 = Vfun(theta);
                Eigen::LLT<Mat> llt(V0);
                if (llt.info() != Eigen::Success) { okv[uz(c)] = 0; return; }
                const Mat Vi = llt.solve(Mat::Identity(N, N));
                const Vec res = yy - E0;
                const double logdet = 2.0 * Mat(llt.matrixL()).diagonal().array().log().sum();
                lls[uz(c)] = -0.5 * (double(N) * std::log(2.0 * std::numbers::pi) + logdet + res.dot(Vi * res)) + jac;
                std::vector<Vec> dE(uz(D)); std::vector<Mat> ViDV(uz(D)); std::vector<int> act;
                for (int a = 0; a < D; ++a) {
                    if (fixed_[uz(a)]) continue;
                    if (a < nb) {
                        Vec de(N);
                        for (int b = 0; b < m; ++b) {
                            const int ni = off[uz(b + 1)] - off[uz(b)];
                            if (ni == 0) continue;
                            de.segment(off[uz(b)], ni) = Jphi.block(off[uz(b)], core_.coefParam(a), ni, 1) * core_.xcoef(mem[uz(b)], a);
                        }
                        dE[uz(a)] = de;
                    } else if (a < errEnd || (a >= offPsi_)) {
                        const double h = 1e-5 * std::max(1.0, std::abs(theta[a]));
                        Vec tp = theta, tm = theta; tp[a] += h; tm[a] -= h;
                        ViDV[uz(a)] = Vi * ((Vfun(tp) - Vfun(tm)) / (2.0 * h));
                    } else continue;
                    act.push_back(a);
                }
                Mat& I = parts[uz(c)];
                for (int a : act) for (int b : act) {
                    double v = 0.0;
                    if (dE[uz(a)].size() > 0 && dE[uz(b)].size() > 0) v += dE[uz(a)].dot(Vi * dE[uz(b)]);
                    if (ViDV[uz(a)].size() > 0 && ViDV[uz(b)].size() > 0) v += 0.5 * (ViDV[uz(a)] * ViDV[uz(b)]).trace();
                    I(a, b) = v;
                }
            } catch (const std::exception&) { okv[uz(c)] = 0; }
        });
        out.I = Mat::Zero(D, D); out.ll = 0.0; out.ok = true;
        for (int c = 0; c < C_; ++c) {
            if (!okv[uz(c)]) { out.ok = false; continue; }
            out.I += parts[uz(c)]; out.ll += lls[uz(c)];
        }
        return out;
    }

    // --- Marginal logLik per cluster by JOINT IMPORTANCE SAMPLING (xi, chi_1..chi_m) ------------
    // t_nu proposal centered at the mean/covariance of the posterior cluster samples (works for
    // any q: it replaces the nested quadrature when q > 3).
    template <class Draws>
    double logMarginalIS(const Vec& theta, const Draws& draws, int M, unsigned seed, double nu = 5.0, double infl = 1.3) const {
        std::vector<double> part(uz(C_), 0.0);
        parallelFor(C_, threads, [&](int c) {
            const double ninf = -std::numeric_limits<double>::infinity();
            try {
                const auto& mem = members_[uz(c)];
                const int m = static_cast<int>(mem.size()), d = q_ * (1 + m);
                const auto& dr = draws[uz(c)];
                const int S = static_cast<int>(dr.size());
                if (S < 4) { part[uz(c)] = std::numeric_limits<double>::quiet_NaN(); return; }
                Mat X(S, d);
                for (int s = 0; s < S; ++s) {
                    X.block(s, 0, 1, q_) = dr[uz(s)].xi.transpose();
                    for (int a = 0; a < m; ++a) X.block(s, q_ * (1 + a), 1, q_) = dr[uz(s)].chi[uz(a)].transpose();
                }
                const Vec mu = X.colwise().mean().transpose();
                Mat Xc = X.rowwise() - mu.transpose();
                Mat Sg = Mat::Zero(d, d);
                if (S >= d + 2) Sg = Xc.transpose() * Xc / double(S - 1);
                else for (int j = 0; j < d; ++j) Sg(j, j) = Xc.col(j).squaredNorm() / double(S - 1);
                Sg *= infl;
                Sg.diagonal().array() += 1e-10 * std::max(1.0, Sg.diagonal().mean());
                Eigen::LLT<Mat> llt(Sg);
                if (llt.info() != Eigen::Success) { part[uz(c)] = std::numeric_limits<double>::quiet_NaN(); return; }
                const Mat L = llt.matrixL();
                const double logdetS = 2.0 * L.diagonal().array().log().sum();
                const double lognorm = std::lgamma(0.5 * (nu + d)) - std::lgamma(0.5 * nu) - 0.5 * d * std::log(nu * std::numbers::pi) - 0.5 * logdetS;
                // pieces of log p(y_c, chi, xi; theta)
                Eigen::LLT<Mat> lO(core_.omegaOf(theta)), lP(psiOf(theta));
                if (lO.info() != Eigen::Success || lP.info() != Eigen::Success) { part[uz(c)] = ninf; return; }
                const Mat LO = lO.matrixL(), LP = lP.matrixL();
                const double ldO = 2.0 * LO.diagonal().array().log().sum(), ldP = 2.0 * LP.diagonal().array().log().sum();
                const double k2 = double(q_) * std::log(2.0 * std::numbers::pi);
                const ErrParams ep = core_.obs().decode(theta, core_.offError());
                std::vector<Vec> base(uz(m));
                for (int a = 0; a < m; ++a) core_.phiOf(mem[uz(a)], theta, Vec::Zero(q_), base[uz(a)]);
                std::mt19937 rng(seed + 1000003u * unsigned(c + 1));
                std::normal_distribution<double> nz(0.0, 1.0);
                std::gamma_distribution<double> gam(0.5 * nu, 2.0);
                std::vector<double> lw;
                lw.reserve(uz(M));
                for (int s = 0; s < M; ++s) {
                    Vec z(d); for (int j = 0; j < d; ++j) z[j] = nz(rng);
                    const double w = gam(rng) / nu;
                    const Vec x = mu + L * z / std::sqrt(w);
                    const double qd = z.squaredNorm() / w;
                    const double lq = lognorm - 0.5 * (nu + d) * std::log1p(qd / nu);
                    const Vec xi = x.head(q_);
                    const Vec zc = LP.triangularView<Eigen::Lower>().solve(xi);
                    double v = -0.5 * (zc.squaredNorm() + ldP + k2);
                    bool bad = false;
                    for (int a = 0; a < m && !bad; ++a) {
                        const int i = mem[uz(a)];
                        const Vec chi = x.segment(q_ * (1 + a), q_);
                        Vec phi = base[uz(a)], dv(q_);
                        for (int l = 0; l < q_; ++l) { dv[l] = chi[l] - base[uz(a)][core_.reIdx()[uz(l)]] - xi[l]; phi[core_.reIdx()[uz(l)]] = chi[l]; }
                        const double ll = core_.obs().logLik(i, core_.psiOf(phi), ep);
                        if (!std::isfinite(ll)) { bad = true; break; }
                        const Vec zz = LO.triangularView<Eigen::Lower>().solve(dv);
                        v += -0.5 * (zz.squaredNorm() + ldO + k2) + ll;
                    }
                    lw.push_back(bad ? ninf : v - lq);
                }
                part[uz(c)] = logMeanExp(lw);
            } catch (const std::exception&) { part[uz(c)] = std::numeric_limits<double>::quiet_NaN(); }
        });
        double s = 0.0; for (double v : part) s += v;
        return s;
    }

    // --- LAPLACE per cluster (STRUCTURED Newton in x = (xi, chi_1..chi_m)) ----------------------------
    // log p(y_c | theta) ≈ l(x^) + (d/2) log 2 pi - 0.5 log det(-H),  d = q (1 + m_c).
    // Gradient and Hessian: analytic prior parts (Omega^-1, Psi^-1, arrow structure) plus finite
    // differences of ONLY log p(y_i | chi_i) for each subject (dimension q). `x` = start and is updated.
    double laplaceCluster(int c, const Vec& th, Vec& x) const {
        const double ninf = -std::numeric_limits<double>::infinity();
        const auto& mem = members_[uz(c)];
        const int m = static_cast<int>(mem.size()), d = q_ * (1 + m);
        Eigen::LLT<Mat> lO(core_.omegaOf(th)), lP(psiOf(th));
        if (lO.info() != Eigen::Success || lP.info() != Eigen::Success) return ninf;
        const Mat Oi = lO.solve(Mat::Identity(q_, q_)), Pi = lP.solve(Mat::Identity(q_, q_));
        const double ldO = 2.0 * Mat(lO.matrixL()).diagonal().array().log().sum(), ldP = 2.0 * Mat(lP.matrixL()).diagonal().array().log().sum();
        const double k2 = double(q_) * std::log(2.0 * std::numbers::pi);
        const ErrParams ep = core_.obs().decode(th, core_.offError());
        std::vector<Vec> phi0(uz(m)), bre(uz(m));
        for (int a = 0; a < m; ++a) { core_.phiOf(mem[uz(a)], th, Vec::Zero(q_), phi0[uz(a)]); bre[uz(a)] = core_.baseRE(mem[uz(a)], th); }
        auto ll = [&](int a, const Vec& chi) {
            Vec phi = phi0[uz(a)];
            for (int l = 0; l < q_; ++l) phi[core_.reIdx()[uz(l)]] = chi[l];
            return core_.obs().logLik(mem[uz(a)], core_.psiOf(phi), ep);
        };
        auto lfun = [&](const Vec& xx) {
            const Vec xi = xx.head(q_);
            double v = -0.5 * (xi.dot(Pi * xi) + ldP + k2);
            for (int a = 0; a < m; ++a) {
                const Vec chi = xx.segment(q_ * (1 + a), q_);
                const Vec r = chi - bre[uz(a)] - xi;
                const double l1 = ll(a, chi);
                if (!std::isfinite(l1)) return ninf;
                v += -0.5 * (r.dot(Oi * r) + ldO + k2) + l1;
            }
            return v;
        };
        Vec g(d); Mat H(d, d);
        auto derivs = [&](const Vec& xx) -> bool {
            g.setZero(); H.setZero();
            const Vec xi = xx.head(q_);
            Vec sr = Vec::Zero(q_);
            for (int a = 0; a < m; ++a) {
                const Vec chi = xx.segment(q_ * (1 + a), q_);
                const Vec r = chi - bre[uz(a)] - xi;
                sr += r;
                // finite differences of ll(a, chi)
                const double l0 = ll(a, chi);
                if (!std::isfinite(l0)) return false;
                Vec gl(q_); Mat Hl(q_, q_);
                for (int u = 0; u < q_; ++u) {
                    const double h = 1e-4 * std::max(1.0, std::abs(chi[u]));
                    Vec cp = chi, cm = chi; cp[u] += h; cm[u] -= h;
                    const double fp = ll(a, cp), fm = ll(a, cm);
                    if (!std::isfinite(fp) || !std::isfinite(fm)) return false;
                    gl[u] = (fp - fm) / (2.0 * h);
                    Hl(u, u) = (fp - 2.0 * l0 + fm) / (h * h);
                    for (int w = u + 1; w < q_; ++w) {
                        const double hw = 1e-4 * std::max(1.0, std::abs(chi[w]));
                        Vec c1 = chi, c2 = chi, c3 = chi, c4 = chi;
                        c1[u] += h; c1[w] += hw; c2[u] += h; c2[w] -= hw; c3[u] -= h; c3[w] += hw; c4[u] -= h; c4[w] -= hw;
                        const double v = (ll(a, c1) - ll(a, c2) - ll(a, c3) + ll(a, c4)) / (4.0 * h * hw);
                        if (!std::isfinite(v)) return false;
                        Hl(u, w) = v; Hl(w, u) = v;
                    }
                }
                const int o = q_ * (1 + a);
                g.segment(o, q_) = -Oi * r + gl;
                H.block(o, o, q_, q_) = -Oi + Hl;
                H.block(0, o, q_, q_) = Oi; H.block(o, 0, q_, q_) = Oi;
            }
            g.head(q_) = -Pi * xi + Oi * sr;
            H.block(0, 0, q_, q_) = -Pi - double(m) * Oi;
            return true;
        };
        double f0 = lfun(x);
        if (!std::isfinite(f0)) return ninf;
        for (int it = 0; it < 12; ++it) {
            if (!derivs(x)) return ninf;
            const Mat Sg = sigmaFromNegHessian(H);
            if (Sg.size() == 0) return ninf;
            const Vec step = Sg * g;
            double t = 1.0; bool ok = false;
            for (int ls = 0; ls < 10; ++ls, t *= 0.5) {
                const Vec cand = x + t * step;
                const double fc = lfun(cand);
                if (std::isfinite(fc) && fc >= f0) { x = cand; f0 = fc; ok = true; break; }
            }
            if (!ok || step.cwiseAbs().maxCoeff() * t < 1e-10 * (1.0 + x.cwiseAbs().maxCoeff())) break;
        }
        if (!derivs(x)) return ninf;
        Eigen::LLT<Mat> llt(-H);
        if (llt.info() != Eigen::Success) return ninf;
        const double ld = 2.0 * Mat(llt.matrixL()).diagonal().array().log().sum();
        return f0 + 0.5 * double(d) * std::log(2.0 * std::numbers::pi) - 0.5 * ld;
    }

    // Per-cluster starting point (xi, chi) from the conditional means.
    template <class XiVec>
    std::vector<Vec> laplaceStarts(const std::vector<Vec>& chiHat, const XiVec& xiHat) const {
        std::vector<Vec> st(uz(C_));
        for (int c = 0; c < C_; ++c) {
            const auto& mem = members_[uz(c)];
            Vec x(q_ * (1 + static_cast<int>(mem.size())));
            x.head(q_) = xiHat[uz(c)];
            for (size_t a = 0; a < mem.size(); ++a) x.segment(q_ * tix(1 + a), q_) = chiHat[uz(mem[a])];
            st[uz(c)] = x;
        }
        return st;
    }
    double laplaceLogLik(const Vec& th, std::vector<Vec> starts) const {
        std::vector<double> part(uz(C_), 0.0);
        parallelFor(C_, threads, [&](int c) { part[uz(c)] = laplaceCluster(c, th, starts[uz(c)]); });
        double s = 0.0; for (double v : part) s += v;
        return s;
    }
    // FIM = -Hessian per cluster of the Laplace logLik (deterministic), by finite differences.
    NlmeModel::FimResult laplaceFIM(const Vec& theta, const std::vector<Vec>& starts) const {
        NlmeModel::FimResult out;
        const int D = dimTheta();
        std::vector<int> fr;
        for (int j = 0; j < D; ++j) if (!fixed_[uz(j)]) fr.push_back(j);
        const int m = static_cast<int>(fr.size());
        std::vector<Mat> parts(uz(C_), Mat::Zero(m, m));
        std::vector<char> okv(uz(C_), 1);
        Vec hs(m);
        for (int a = 0; a < m; ++a) hs[a] = 5e-4 * std::max(1.0, std::abs(theta[fr[uz(a)]]));
        parallelFor(C_, threads, [&](int c) {
            try {
                auto f = [&](const Vec& th) { Vec x = starts[uz(c)]; return laplaceCluster(c, th, x); };
                const double f0 = f(theta);
                if (!std::isfinite(f0)) { okv[uz(c)] = 0; return; }
                std::vector<double> fp(uz(m)), fm(uz(m)); Mat H(m, m);
                for (int a = 0; a < m; ++a) {
                    Vec tp = theta, tm = theta; tp[fr[uz(a)]] += hs[a]; tm[fr[uz(a)]] -= hs[a];
                    fp[uz(a)] = f(tp); fm[uz(a)] = f(tm);
                    if (!std::isfinite(fp[uz(a)]) || !std::isfinite(fm[uz(a)])) { okv[uz(c)] = 0; return; }
                    H(a, a) = (fp[uz(a)] - 2.0 * f0 + fm[uz(a)]) / (hs[a] * hs[a]);
                }
                for (int a = 0; a < m; ++a) for (int b = a + 1; b < m; ++b) {
                    Vec t1 = theta, t2 = theta, t3 = theta, t4 = theta;
                    t1[fr[uz(a)]] += hs[a]; t1[fr[uz(b)]] += hs[b]; t2[fr[uz(a)]] += hs[a]; t2[fr[uz(b)]] -= hs[b];
                    t3[fr[uz(a)]] -= hs[a]; t3[fr[uz(b)]] += hs[b]; t4[fr[uz(a)]] -= hs[a]; t4[fr[uz(b)]] -= hs[b];
                    const double v = (f(t1) - f(t2) - f(t3) + f(t4)) / (4.0 * hs[a] * hs[b]);
                    if (!std::isfinite(v)) { okv[uz(c)] = 0; return; }
                    H(a, b) = v; H(b, a) = v;
                }
                parts[uz(c)] = -H;
            } catch (const std::exception&) { okv[uz(c)] = 0; }
        });
        out.I = Mat::Zero(D, D); out.ok = true;
        for (int c = 0; c < C_; ++c) {
            if (!okv[uz(c)]) { out.ok = false; continue; }
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) out.I(fr[uz(a)], fr[uz(b)]) += parts[uz(c)](a, b);
        }
        return out;
    }

    // --- Louis FIM WITHOUT MCMC noise: i.i.d. importance sampling + block-wise derivatives ----------
    // I_obs = sum_c ( -E[d2 l_c] - Cov[d l_c] ) with expectations under the cluster posterior, approximated
    // by i.i.d. samples of a t_nu fitted to the MCMC samples (self-normalized weights).
    // Parameters are split into two groups with a full BLOCK-DIAGONAL Hessian:
    //   prior (coefficients of parameters with a random effect, Omega, Psi): l_prior(chi, xi) = cheap
    //   likelihood (coefficients without a random effect, error, extra)    : l_lik(chi) = one model evaluation
    // and the covariance of the complete score couples both. Two independent halves measure the residual noise.
    // `ess` = minimum ESS/M fraction across clusters.
    template <class Draws>
    NlmeModel::FimResult louisIS(const Vec& theta, const Draws& draws, int M, unsigned seed, double& noise, double& ess,
                                 double nu = 4.0, double infl = 1.5) const {
        NlmeModel::FimResult out;
        const int D = dimTheta(), nb = core_.nb();
        const ObsLik& ob = core_.obs();
        std::vector<int> fr, kind;                       // kind: 0 = prior, 1 = likelihood
        for (int j = 0; j < D; ++j) {
            if (fixed_[uz(j)]) continue;
            fr.push_back(j);
            bool prior;
            if (j < nb) prior = core_.reOf()[uz(core_.coefParam(j))] >= 0;
            else if (j >= offPsi_) prior = true;
            else if (j < core_.offError()) prior = true;          // Omega
            else prior = false;                                    // error / extra
            kind.push_back(prior ? 0 : 1);
        }
        const int m = static_cast<int>(fr.size());
        std::vector<int> gp, gl;                         // positions in fr of each group
        for (int a = 0; a < m; ++a) (kind[uz(a)] == 0 ? gp : gl).push_back(a);
        std::vector<Mat> part(uz(2 * C_), Mat::Zero(m, m));  // [half][cluster]
        std::vector<double> essC(uz(C_), 1.0);
        std::vector<char> okv(uz(C_), 1);
        parallelFor(C_, threads, [&](int c) {
            try {
                const auto& mem = members_[uz(c)];
                const int mm = static_cast<int>(mem.size()), d = q_ * (1 + mm);
                const auto& dr = draws[uz(c)];
                const int S0 = static_cast<int>(dr.size());
                if (S0 < 4) { okv[uz(c)] = 0; return; }
                Mat X(S0, d);
                for (int s = 0; s < S0; ++s) {
                    X.block(s, 0, 1, q_) = dr[uz(s)].xi.transpose();
                    for (int a = 0; a < mm; ++a) X.block(s, q_ * (1 + a), 1, q_) = dr[uz(s)].chi[uz(a)].transpose();
                }
                const Vec mu = X.colwise().mean().transpose();
                const Mat Xc = X.rowwise() - mu.transpose();
                Mat Sg = Mat::Zero(d, d);
                if (S0 >= d + 2) Sg = Xc.transpose() * Xc / double(S0 - 1);
                else for (int j = 0; j < d; ++j) Sg(j, j) = Xc.col(j).squaredNorm() / double(S0 - 1);
                Sg *= infl; Sg.diagonal().array() += 1e-10 * std::max(1.0, Sg.diagonal().mean());
                Eigen::LLT<Mat> llt(Sg);
                if (llt.info() != Eigen::Success) { okv[uz(c)] = 0; return; }
                const Mat L = llt.matrixL();
                const double logdetS = 2.0 * L.diagonal().array().log().sum();
                const double lognorm = std::lgamma(0.5 * (nu + d)) - std::lgamma(0.5 * nu) - 0.5 * d * std::log(nu * std::numbers::pi) - 0.5 * logdetS;
                const double k2 = double(q_) * std::log(2.0 * std::numbers::pi);
                const ErrParams ep0 = ob.decode(theta, core_.offError());

                // l_prior(th; xi, chi) and l_lik(th; chi)  (centered parameterization)
                auto lPrior = [&](const Vec& th, const Vec& xi, const std::vector<Vec>& chi) {
                    Eigen::LLT<Mat> lO(core_.omegaOf(th)), lP(psiOf(th));
                    if (lO.info() != Eigen::Success || lP.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
                    const Mat LO = lO.matrixL(), LP = lP.matrixL();
                    const Vec zc = LP.triangularView<Eigen::Lower>().solve(xi);
                    double v = -0.5 * (zc.squaredNorm() + 2.0 * LP.diagonal().array().log().sum() + k2);
                    const double ldO = 2.0 * LO.diagonal().array().log().sum();
                    for (int a = 0; a < mm; ++a) {
                        const Vec dv = chi[uz(a)] - core_.baseRE(mem[uz(a)], th) - xi;
                        const Vec z = LO.triangularView<Eigen::Lower>().solve(dv);
                        v += -0.5 * (z.squaredNorm() + ldO + k2);
                    }
                    return v;
                };
                auto lLik = [&](const Vec& th, const std::vector<Vec>& chi) {
                    const ErrParams ep = ob.decode(th, core_.offError());
                    double v = 0.0; Vec phi;
                    for (int a = 0; a < mm; ++a) {
                        core_.phiOf(mem[uz(a)], th, Vec::Zero(q_), phi);
                        for (int l = 0; l < q_; ++l) phi[core_.reIdx()[uz(l)]] = chi[uz(a)][l];
                        const double ll = ob.logLik(mem[uz(a)], core_.psiOf(phi), ep);
                        if (!std::isfinite(ll)) return -std::numeric_limits<double>::infinity();
                        v += ll;
                    }
                    return v;
                };
                (void)ep0;

                const int Mh = std::max(8, M / 2);
                std::mt19937 rng(seed + 1000003u * unsigned(c + 1));
                std::normal_distribution<double> nz(0.0, 1.0);
                std::gamma_distribution<double> gam(0.5 * nu, 2.0);
                double essMin = 1.0;
                for (int half = 0; half < 2; ++half) {
                    std::vector<Vec> xs; std::vector<double> lw; xs.reserve(uz(Mh)); lw.reserve(uz(Mh));
                    for (int s = 0; s < Mh; ++s) {
                        Vec z(d); for (int j = 0; j < d; ++j) z[j] = nz(rng);
                        const double w = gam(rng) / nu;
                        const Vec x = mu + L * z / std::sqrt(w);
                        const double lq = lognorm - 0.5 * (nu + d) * std::log1p(z.squaredNorm() / w / nu);
                        const Vec xi = x.head(q_);
                        std::vector<Vec> chi(uz(mm));
                        for (int a = 0; a < mm; ++a) chi[uz(a)] = x.segment(q_ * (1 + a), q_);
                        const double v = lPrior(theta, xi, chi) + lLik(theta, chi);
                        lw.push_back(std::isfinite(v) ? v - lq : -std::numeric_limits<double>::infinity());
                        xs.push_back(x);
                    }
                    double lmax = -std::numeric_limits<double>::infinity();
                    for (double v : lw) lmax = std::max(lmax, v);
                    if (!std::isfinite(lmax)) { okv[uz(c)] = 0; return; }
                    double W0 = 0.0, W2 = 0.0;
                    std::vector<double> wn(uz(Mh));
                    for (int s = 0; s < Mh; ++s) { wn[uz(s)] = std::exp(lw[uz(s)] - lmax); W0 += wn[uz(s)]; }
                    // TRUNCATED weights (Ionides 2008): w <= mean * sqrt(M); controls heavy weight tails
                    {
                        const double cap = (W0 / double(Mh)) * std::sqrt(double(Mh));
                        W0 = 0.0;
                        for (int s = 0; s < Mh; ++s) { wn[uz(s)] = std::min(wn[uz(s)], cap); W0 += wn[uz(s)]; W2 += wn[uz(s)] * wn[uz(s)]; }
                    }
                    essMin = std::min(essMin, (W0 * W0 / W2) / double(Mh));
                    // weighted accumulators (West)
                    double Wt = 0.0; Vec gm = Vec::Zero(m); Mat S2 = Mat::Zero(m, m), Hs = Mat::Zero(m, m);
                    for (int s = 0; s < Mh; ++s) {
                        if (wn[uz(s)] < 1e-8 * W0 / double(Mh)) continue;          // negligible weight
                        const Vec& x = xs[uz(s)];
                        const Vec xi = x.head(q_);
                        std::vector<Vec> chi(uz(mm));
                        for (int a = 0; a < mm; ++a) chi[uz(a)] = x.segment(q_ * (1 + a), q_);
                        Vec g(m); Mat H = Mat::Zero(m, m); bool bad = false;
                        auto grp = [&](const std::vector<int>& idx, auto&& f) {
                            const int k = static_cast<int>(idx.size());
                            if (k == 0) return;
                            const double f0 = f(theta);
                            std::vector<double> fp(uz(k)), fm(uz(k)), h(uz(k));
                            for (int a = 0; a < k && !bad; ++a) {
                                h[uz(a)] = 2e-4 * std::max(1.0, std::abs(theta[fr[uz(idx[uz(a)])]]));
                                Vec tp = theta, tm = theta; tp[fr[uz(idx[uz(a)])]] += h[uz(a)]; tm[fr[uz(idx[uz(a)])]] -= h[uz(a)];
                                fp[uz(a)] = f(tp); fm[uz(a)] = f(tm);
                                if (!std::isfinite(fp[uz(a)]) || !std::isfinite(fm[uz(a)])) { bad = true; break; }
                                g[idx[uz(a)]] = (fp[uz(a)] - fm[uz(a)]) / (2.0 * h[uz(a)]);
                                H(idx[uz(a)], idx[uz(a)]) = (fp[uz(a)] - 2.0 * f0 + fm[uz(a)]) / (h[uz(a)] * h[uz(a)]);
                            }
                            for (int a = 0; a < k && !bad; ++a) for (int b2 = a + 1; b2 < k && !bad; ++b2) {
                                Vec t1 = theta, t2 = theta, t3 = theta, t4 = theta;
                                const int ja = fr[uz(idx[uz(a)])], jb = fr[uz(idx[uz(b2)])];
                                t1[ja] += h[uz(a)]; t1[jb] += h[uz(b2)]; t2[ja] += h[uz(a)]; t2[jb] -= h[uz(b2)];
                                t3[ja] -= h[uz(a)]; t3[jb] += h[uz(b2)]; t4[ja] -= h[uz(a)]; t4[jb] -= h[uz(b2)];
                                const double v = (f(t1) - f(t2) - f(t3) + f(t4)) / (4.0 * h[uz(a)] * h[uz(b2)]);
                                if (!std::isfinite(v)) { bad = true; break; }
                                H(idx[uz(a)], idx[uz(b2)]) = v; H(idx[uz(b2)], idx[uz(a)]) = v;
                            }
                        };
                        grp(gp, [&](const Vec& th) { return lPrior(th, xi, chi); });
                        if (!bad) grp(gl, [&](const Vec& th) { return lLik(th, chi); });
                        if (bad) continue;
                        const double w = wn[uz(s)];
                        Wt += w;
                        const Vec dl = g - gm;
                        gm += dl * (w / Wt);
                        S2 += w * dl * (g - gm).transpose();
                        Hs += w * H;
                    }
                    if (Wt <= 0.0) { okv[uz(c)] = 0; return; }
                    part[uz(2 * c + half)] = -Hs / Wt - S2 / Wt;
                }
                essC[uz(c)] = essMin;
            } catch (const std::exception&) { okv[uz(c)] = 0; }
        });
        Mat I1 = Mat::Zero(D, D), I2 = Mat::Zero(D, D);
        out.ok = true; ess = 1.0;
        for (int c = 0; c < C_; ++c) {
            if (!okv[uz(c)]) { out.ok = false; continue; }
            ess = std::min(ess, essC[uz(c)]);
            for (int a = 0; a < m; ++a) for (int b2 = 0; b2 < m; ++b2) {
                I1(fr[uz(a)], fr[uz(b2)]) += part[uz(2 * c)](a, b2); I2(fr[uz(a)], fr[uz(b2)]) += part[uz(2 * c + 1)](a, b2);
            }
        }
        out.I = 0.5 * (I1 + I2);
        // Noise measured on the scale of the STANDARD ERRORS of each half (what the user sees).
        auto seOf = [&](const Mat& I) {
            Mat Ifr(m, m);
            for (int a = 0; a < m; ++a) for (int b2 = 0; b2 < m; ++b2) Ifr(a, b2) = I(fr[uz(a)], fr[uz(b2)]);
            Eigen::SelfAdjointEigenSolver<Mat> es(0.5 * (Ifr + Ifr.transpose()));
            Vec ev = es.eigenvalues();
            const double fl = std::max(1e-12, 1e-9 * ev.cwiseAbs().maxCoeff());
            const Mat inv = es.eigenvectors() * ev.cwiseMax(fl).cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
            Vec se(m); for (int a = 0; a < m; ++a) se[a] = std::sqrt(std::max(inv(a, a), 0.0));
            return se;
        };
        const Vec s1 = seOf(I1), s2 = seOf(I2);
        noise = 0.0;
        for (int a = 0; a < m; ++a) noise = std::max(noise, std::abs(s1[a] - s2[a]) / std::max(0.5 * (s1[a] + s2[a]), 1e-12));
        return out;
    }

private:
    NlmeModel core_;
    unsigned long id_;
    int n_ = 0, q_ = 0, C_ = 0, offPsi_ = 0, nPsi_ = 0;
    std::vector<int> clusterOf_;
    std::vector<std::vector<int>> members_;
    CovModel psiCov_;
    Vec theta0_;
    std::vector<char> fixed_;
    static unsigned long nextId() { static std::atomic<unsigned long> c{0}; return ++c; }
};

} // namespace mcmckl
