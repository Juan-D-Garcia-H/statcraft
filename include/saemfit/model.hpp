// ============================================================================
// @file model.hpp
// @brief Generic nonlinear mixed-effects model built from a user-supplied
//        prediction function.
//
// The model is
//
//   y_ij = f(phi_i, t_ij) + residual error   (constant / proportional /
//                                             exponential / combined)
//   phi_i ~ N(mu, Omega)
//
// The user only supplies:
//   - predict : f(phi, t)  ->  double
//   - dimPhi
//   - the residual error model (residual.hpp)
//
// The framework automatically builds:
//   - logJoint(i, phi, theta)            (likelihood + prior)
//   - mStepGradient(theta, phis)         (M-step gradient, by finite differences
//                                         or analytical when supplied by the user)
//   - independenceProposal(theta)        -> N(mu, Omega)
//   - the parameter vector theta:
//        theta = [ mu (p) ; paramsOmega (m) ; paramsError (r) ]
//     Omega diagonal -> m = p   (log sd)
//     Omega full     -> m = p(p+1)/2  (Cholesky: diagonal in log, off-diagonal raw)
//     paramsError    -> log of the parameter (a and/or b depending on the type)
//
// This works with SAEM, MAP, llis/llgq and the FIM (fisher.hpp) unchanged.
//
// Requires: Eigen3, C++20, engine.hpp, residual.hpp.
// ============================================================================

#pragma once
#include "numcast.hpp"

#include "engine.hpp"
#include "residual.hpp"
#include "nelder.hpp"

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace mcmckl {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// ---------------------------------------------------------------------------
// Covariance parameterization utilities (shared by Omega and Psi).
// ---------------------------------------------------------------------------

/// @brief Decodes a covariance matrix from a parameter slice.
///
/// For a diagonal covariance the parameter holds log standard deviations.
/// For a full covariance the parameter holds the Cholesky factor: diagonal
/// entries are logs and off-diagonal entries are used as-is.
///
/// @param th   Parameter vector.
/// @param off  Offset of the covariance block in @p th.
/// @param p    Dimension of the covariance matrix.
/// @param full Whether the covariance is full or diagonal.
/// @return The decoded covariance matrix.
inline Mat decodeCov(const Vec& th, int off, int p, bool full) {
    if (!full) {
        Mat M = Mat::Zero(p, p);
        for (int j = 0; j < p; ++j) M(j, j) = std::exp(2.0 * th[off + j]);
        return M;
    }
    Mat L = Mat::Zero(p, p);
    int k = off;
    for (int j = 0; j < p; ++j) for (int i = j; i < p; ++i) {
        const double v = th[k++];
        L(i, j) = (i == j) ? std::exp(v) : v;
    }
    return L * L.transpose();
}

/// @brief Encodes a covariance matrix into a parameter slice.
///
/// This is the inverse of decodeCov. If the Cholesky decomposition fails, an
/// identity factor is used for the off-diagonal encoding.
///
/// @param th  Parameter vector to write into.
/// @param off Offset of the covariance block in @p th.
/// @param cov Covariance matrix to encode.
/// @param p   Dimension of the covariance matrix.
/// @param full Whether the covariance is full or diagonal.
inline void encodeCov(Vec& th, int off, const Mat& cov, int p, bool full) {
    Eigen::LLT<Mat> llt(cov);
    const Mat L = (llt.info() == Eigen::Success) ? Mat(llt.matrixL()) : Mat::Identity(p, p);
    if (!full) {
        for (int j = 0; j < p; ++j)
            th[off + j] = std::log(std::max(1e-12, std::sqrt(std::max(1e-300, cov(j, j)))));
        return;
    }
    int k = off;
    for (int j = 0; j < p; ++j) for (int i = j; i < p; ++i)
        th[k++] = (i == j) ? std::log(std::max(1e-12, L(i, j))) : L(i, j);
}

// ---------------------------------------------------------------------------
// Sufficient statistics of a covariance block (for the closed-form M-step).
// Layout: diagonal -> E[x_j^2] (p); full -> lower-triangular E[x_i x_j], by
// columns (same order as the Cholesky parameterization of theta).
// ---------------------------------------------------------------------------

/// @brief Number of sufficient statistics for a covariance block.
/// @param p    Dimension of the covariance matrix.
/// @param full Whether the covariance is full or diagonal.
/// @return p for a diagonal covariance, p(p+1)/2 for a full covariance.
inline int covStatsSize(int p, bool full) { return full ? p * (p + 1) / 2 : p; }

/// @brief Accumulates one weighted sample into covariance sufficient statistics.
///
/// @param s    Statistic vector to update.
/// @param off  Offset of the covariance block in @p s.
/// @param x    Sample vector of size p.
/// @param full Whether the covariance is full or diagonal.
/// @param w    Weight applied to this sample.
inline void addCovStats(Vec& s, int off, const Vec& x, bool full, double w) {
    const int p = static_cast<int>(x.size());
    if (!full) { for (int j = 0; j < p; ++j) s[off + j] += w * x[j] * x[j]; return; }
    int k = off;
    for (int j = 0; j < p; ++j) for (int i = j; i < p; ++i) s[k++] += w * x[i] * x[j];
}

/// @brief Builds Cov = E[xx'] - m m' from sufficient statistics.
///
/// A floor is applied so the result is positive definite. The closed-form
/// M-step with averaged statistics can produce eigenvalues near zero or
/// negative due to rounding when a variance collapses.
///
/// @param s    Statistic vector.
/// @param off  Offset of the covariance block in @p s.
/// @param p    Dimension of the covariance matrix.
/// @param full Whether the covariance is full or diagonal.
/// @param m    Mean vector used to center the second moment.
/// @return A positive-definite covariance matrix.
inline Mat covFromStats(const Vec& s, int off, int p, bool full, const Vec& m) {
    Mat C = Mat::Zero(p, p);
    if (!full) {
        for (int j = 0; j < p; ++j) {
            const double ex2 = s[off + j];
            C(j, j) = std::max(ex2 - m[j] * m[j], 1e-12 * std::max(1.0, std::abs(ex2)));
        }
        return C;
    }
    int k = off;
    for (int j = 0; j < p; ++j) for (int i = j; i < p; ++i) {
        const double v = s[k++] - m[i] * m[j];
        C(i, j) = v; C(j, i) = v;
    }
    Eigen::SelfAdjointEigenSolver<Mat> es(C);
    Vec ev = es.eigenvalues();
    const double fl = 1e-10 * std::max(1.0, ev.cwiseAbs().maxCoeff());
    return es.eigenvectors() * ev.cwiseMax(fl).asDiagonal() * es.eigenvectors().transpose();
}

/// @brief Generic nonlinear mixed-effects model with a user-supplied prediction.
///
/// The class owns the observation times and responses, validates the data and
/// the specification, and exposes the SaemModel interface used by SAEM, MAP and
/// related algorithms.
///
/// @note The model keeps a mutable cache of the prior decomposition for the
///       last evaluated theta. It is therefore not thread-safe for concurrent
///       calls that use different theta vectors; use one model copy per thread.
class GenericModel : public SaemModel {
public:
    /// @brief Prediction function f(phi, t) -> response (arbitrary nonlinear).
    using Predict = std::function<double(const Vec& phi, double t)>;
    /// @brief Prediction function f(phi, t, x) -> response with subject covariates.
    /// Takes priority over Predict when both are provided.
    using PredictCov = std::function<double(const Vec& phi, double t, const Vec& x)>;
    /// @brief Optional analytical gradient of logJoint with respect to theta.
    using GradTheta = std::function<Vec(int i, const Vec& phi, const Vec& theta)>;

    /// @brief Specification of the model and its optional initial values.
    struct Spec {
        Predict predict;                               ///< f(phi, t)
        PredictCov predictCov;                         ///< f(phi, t, x) [priority]
        std::function<Vec(int i)> subjectCovariates;   ///< i -> subject covariates
        int     dimPhi     = 1;                        ///< Dimension of phi.
        bool    fullOmega  = false;                    ///< Full or diagonal Omega.
        resmod::ErrorKind residType = resmod::ErrorKind::Proportional; ///< Residual error type.

        // Optional initial values.
        Vec          mu0;       ///< Initial mean (size dimPhi).
        Vec          sd0;       ///< Initial standard deviations (diagonal, size dimPhi).
        Mat          Omega0;    ///< Initial Omega when fullOmega is true.
        resmod::Pres resid0;    ///< Initial residual error parameters.
        GradTheta    gradTheta; ///< Optional analytical gradient (empty -> finite differences).
    };

    /// @brief Constructs a model from a specification, observation times and responses.
    ///
    /// @param spec Model specification.
    /// @param t    Observation times, one vector per subject.
    /// @param y    Responses, one vector per subject.
    /// @throws std::invalid_argument If dimPhi <= 0, no prediction function is set,
    ///         subject counts differ, the specification is inconsistent, or the
    ///         data are invalid.
    GenericModel(Spec spec,
                 std::vector<std::vector<double>> t,
                 std::vector<std::vector<double>> y)
      : spec_(std::move(spec)),
        t_(std::move(t)), y_(std::move(y)),
        p_(spec_.dimPhi), n_(static_cast<int>(t_.size())),
        residType_(spec_.residType)
    {
        if (p_ <= 0) throw std::invalid_argument("dimPhi > 0");
        if (!spec_.predict && !spec_.predictCov) throw std::invalid_argument("predict is required");
        if (static_cast<int>(y_.size()) != n_) throw std::invalid_argument("t and y have different sizes");
        validateSpec();
        validateData();
        nObs_ = 0;
        for (int i = 0; i < n_; ++i) nObs_ += static_cast<long>(y_[uz(i)].size());
        // Subject covariates: evaluated ONCE (previously, one assignment per
        // logLikelihood evaluation inside the MCMC).
        hasCov_ = spec_.predictCov && spec_.subjectCovariates;
        if (hasCov_) {
            cov_.resize(uz(n_));
            for (int i = 0; i < n_; ++i) cov_[uz(i)] = spec_.subjectCovariates(i);
        }
    }

    // --- SaemModel ----------------------------------------------------------

    /// @brief Total dimension of theta.
    /// @return dimPhi + Omega parameters + residual parameters.
    int dimTheta() const override { return p_ + nOmega() + nResid(); }

    /// @brief Dimension of the sampled individual effect.
    /// @return dimPhi.
    int dimChi()   const override { return p_; }

    /// @brief Number of subjects.
    int nSubjects() const override { return n_; }

    /// @brief Builds the initial theta from the optional initial values.
    /// @return Initial parameter vector.
    Vec initialTheta() const override {
        Vec th = Vec::Zero(dimTheta());
        // mu
        if (spec_.mu0.size() == p_) th.head(p_) = spec_.mu0;
        // Omega
        if (!spec_.fullOmega) {
            Vec sd = (spec_.sd0.size() == p_) ? spec_.sd0 : Vec::Constant(p_, 1.0);
            for (int j = 0; j < p_; ++j) th[p_ + j] = std::log(sd[j]);
        } else {
            Mat Om = (spec_.Omega0.rows() == p_) ? spec_.Omega0 : Mat::Identity(p_, p_);
            Eigen::LLT<Mat> llt(Om);
            Mat L = llt.matrixL();
            int k = p_;
            for (int j = 0; j < p_; ++j) for (int i = j; i < p_; ++i)
                th[k++] = (i == j) ? std::log(L(i, j)) : L(i, j);
        }
        // Residual error (log).
        encodeResid(th, p_ + nOmega(), spec_.resid0);
        return th;
    }

    /// @brief Initial individual effect for a subject.
    /// @param Subject index (ignored; the population mean is used for all subjects).
    /// @return The first dimPhi entries of initialTheta().
    Vec initialChi(int) const override {
        Vec th = initialTheta();
        return th.head(p_);
    }

    /// @brief Log joint density log p(y_i, phi_i; theta).
    ///
    /// Uses a mutable cache of mu, chol(Omega) and residual parameters for the
    /// last evaluated theta.
    ///
    /// @param i     Subject index.
    /// @param phi   Individual effect.
    /// @param theta Parameter vector.
    /// @return The log joint density, or -infinity when the prior decomposition
    ///         fails or the likelihood is not finite.
    ///
    /// @note Not thread-safe across different theta values; use one model copy
    ///       per thread.
    double logJoint(int i, const Vec& phi, const Vec& theta) const override {
        const PriorCache& pc = cacheFor(theta);
        if (!pc.ok) return -std::numeric_limits<double>::infinity();
        const double ll = logLikelihood(i, phi, pc.pres);
        if (!std::isfinite(ll)) return -std::numeric_limits<double>::infinity();
        const Vec d = pc.L.triangularView<Eigen::Lower>().solve(phi - pc.mu);
        return -0.5 * (d.squaredNorm() + pc.logdet + double(phi.size()) * std::log(2.0 * std::numbers::pi)) + ll;
    }

    /// @brief Log likelihood log p(y_i | phi) only, without the prior.
    ///
    /// Reused by the multilevel model, which supplies its own prior with
    /// two-level random effects.
    ///
    /// @param i    Subject index.
    /// @param phi  Individual effect.
    /// @param pres Residual error parameters.
    /// @return The log likelihood, or -infinity when a prediction or the error
    ///         scale is not admissible.
    double logLikelihood(int i, const Vec& phi, const resmod::Pres& pres) const {
        const double cst = 0.5 * std::log(2.0 * std::numbers::pi);
        static const Vec none;
        const Vec& x = hasCov_ ? cov_[uz(i)] : none;
        double lp = 0.0;
        for (size_t j = 0; j < t_[uz(i)].size(); ++j) {
            const double f = spec_.predictCov ? spec_.predictCov(phi, t_[uz(i)][j], x)
                                              : spec_.predict(phi, t_[uz(i)][j]);
            // Non-admissible region -> zero density (no exception): robust to MCMC.
            if (!std::isfinite(f)) return -std::numeric_limits<double>::infinity();
            if (residType_ == resmod::ErrorKind::Exponential && !(f > 0.0))
                return -std::numeric_limits<double>::infinity();
            const double g = resmod::error(f, pres, residType_);
            if (!(g > 0.0)) return -std::numeric_limits<double>::infinity();
            const double r = resmod::resid(y_[uz(i)][j], f, residType_);
            lp += -0.5 * (r / g) * (r / g) - std::log(g) - cst;
            if (residType_ == resmod::ErrorKind::Exponential)
                lp -= std::log(y_[uz(i)][j]);              // Jacobian d(log y)/dy.
        }
        return lp;
    }

    /// @brief Log density of the multivariate normal N(x; mu, Omega).
    /// @param x     Evaluation point.
    /// @param mu    Mean vector.
    /// @param Omega Covariance matrix.
    /// @return The log density, or -infinity if the decomposition fails.
    static double logMultiNormal(const Vec& x, const Vec& mu, const Mat& Omega) {
        Eigen::LLT<Mat> llt(Omega);
        if (llt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
        const Mat L = llt.matrixL();
        const Vec d = L.triangularView<Eigen::Lower>().solve(x - mu);
        const double logdet = 2.0 * L.diagonal().array().log().sum();
        return -0.5 * (d.squaredNorm() + logdet + double(x.size()) * std::log(2.0 * std::numbers::pi));
    }

    /// @brief Gradient of the M-step objective averaged over subjects.
    ///
    /// Uses the user-supplied analytical gradient when available; otherwise
    /// computes central finite differences, falling back to a one-sided
    /// difference when only one side is admissible.
    ///
    /// @param theta Parameter vector.
    /// @param phis  Individual effects, one per subject.
    /// @return Gradient vector with the same size as @p theta.
    Vec mStepGradient(const Vec& theta, const std::vector<Vec>& phis) const override {
        // Analytical gradient when supplied by the user.
        if (spec_.gradTheta) {
            Vec g = Vec::Zero(theta.size());
            for (int i = 0; i < n_; ++i) g += spec_.gradTheta(i, phis[uz(i)], theta);
            return g / double(n_);
        }
        // Central differences (generic, any f).
        const int p = static_cast<int>(theta.size());
        Vec g(p);
        Vec tp = theta, tm = theta;
        for (int j = 0; j < p; ++j) {
            const double h = 1e-6 * std::max(1.0, std::abs(theta[j]));
            tp = theta; tm = theta;
            tp[j] += h; tm[j] -= h;
            double fp = 0.0, fm = 0.0;
            for (int i = 0; i < n_; ++i) {
                fp += logJoint(i, phis[uz(i)], tp);
                fm += logJoint(i, phis[uz(i)], tm);
            }
            if (std::isfinite(fp) && std::isfinite(fm))
                g[j] = (fp - fm) / (2.0 * h) / double(n_);
            else if (std::isfinite(fp)) {                 // One admissible side.
                double f0 = 0.0;
                for (int i = 0; i < n_; ++i) f0 += logJoint(i, phis[uz(i)], theta);
                g[j] = (fp - f0) / h / double(n_);
            } else if (std::isfinite(fm)) {
                double f0 = 0.0;
                for (int i = 0; i < n_; ++i) f0 += logJoint(i, phis[uz(i)], theta);
                g[j] = (f0 - fm) / h / double(n_);
            } else {
                g[j] = 0.0;
            }
        }
        return g;
    }

    /// @brief Independence proposal N(mean, Omega) for the MCMC of chi_i.
    /// @param theta Parameter vector.
    /// @param mean  Output mean vector.
    /// @param Omega Output covariance matrix.
    void independenceProposal(const Vec& theta, Vec& mean, Mat& Omega) const override {
        resmod::Pres pres;
        unpack(theta, mean, Omega, pres);
    }

    // --- Closed-form M-step with sufficient statistics (classic SAEM) --------
    //   s = [ E[phi] (p) | E[phi phi'] (Omega layout) | error statistics ]
    //   mu = E[phi],  Omega = E[phi phi'] - mu mu'   (MLE of the complete normal)
    //   error: constant/proportional/exponential -> closed form (mean of r^2,
    //          r^2/f^2, r_log^2); combined -> numerical minimization of 2 parameters
    //          over stochastically approximated r^2_ij and |f|_ij (per observation).

    /// @brief Whether this model implements sufficient statistics.
    /// @return Always true.
    bool hasSufficientStats() const override { return true; }

    /// @brief Averages sufficient statistics over subjects.
    /// @param phis  Individual effects, one per subject.
    /// @param theta Current parameter vector (unused).
    /// @return The averaged sufficient-statistic vector.
    Vec sufficientStats(const std::vector<Vec>& phis, const Vec& /*theta*/) const override {
        const int nc = covStatsSize(p_, spec_.fullOmega);
        const int nr = nResidStats();
        Vec s = Vec::Zero(p_ + nc + nr);
        const double w = 1.0 / double(n_);
        for (int i = 0; i < n_; ++i) {
            s.head(p_) += w * phis[uz(i)];
            addCovStats(s, p_, phis[uz(i)], spec_.fullOmega, w);
        }
        s.tail(nr) = residualStats(phis);
        return s;
    }

    /// @brief Recomputes theta from sufficient statistics.
    /// @param s     Sufficient-statistic vector.
    /// @param theta Current parameter vector, used for unchanged entries.
    /// @return The updated parameter vector.
    Vec thetaFromStats(const Vec& s, const Vec& theta) const override {
        Vec th = theta;
        const Vec m = s.head(p_);
        th.head(p_) = m;
        encodeCov(th, p_, covFromStats(s, p_, p_, spec_.fullOmega, m), p_, spec_.fullOmega);
        residualThetaFromStats(s.tail(nResidStats()), th, p_ + nOmega());
        return th;
    }

    /// @brief Number of residual error sufficient statistics.
    /// @return 2 * number of observations for the combined model, otherwise 1.
    int nResidStats() const {
        return residType_ == resmod::ErrorKind::Combined ? 2 * static_cast<int>(nObs_) : 1;
    }

    /// @brief Computes residual error statistics for a set of individual effects.
    /// @param phis Individual effects, one per subject.
    /// @return Residual statistic vector.
    /// @throws std::runtime_error If a prediction is not finite or is inadmissible
    ///         for the exponential error model.
    Vec residualStats(const std::vector<Vec>& phis) const {
        Vec rs = Vec::Zero(nResidStats());
        const double invN = 1.0 / double(std::max<long>(1, nObs_));
        long k = 0;
        for (int i = 0; i < n_; ++i) {
            for (size_t j = 0; j < t_[uz(i)].size(); ++j, ++k) {
                const double f = predictAt(i, static_cast<int>(j), phis[uz(i)]);
                if (!std::isfinite(f) || (residType_ == resmod::ErrorKind::Exponential && !(f > 0.0)))
                    throw std::runtime_error("non-finite sufficient statistic (inadmissible prediction)");
                const double r = resmod::resid(y_[uz(i)][j], f, residType_);
                switch (residType_) {
            default: break;
                    case resmod::ErrorKind::Constant:
                    case resmod::ErrorKind::Exponential:
                        rs[0] += invN * r * r; break;
                    case resmod::ErrorKind::Proportional: {
                        const double af = std::max(std::abs(f), 1e-12);
                        rs[0] += invN * (r / af) * (r / af); break;
                    }
                    case resmod::ErrorKind::Combined:
                        rs[2 * k] = r * r; rs[2 * k + 1] = std::abs(f); break;
                }
            }
        }
        return rs;
    }

    /// @brief Writes residual error parameters (in log) from statistics.
    ///
    /// For non-combined error models the scalar statistic gives a closed-form
    /// log-standard-deviation. For the combined model the two parameters are
    /// obtained by numerical minimization and clamped to a safe range. If the
    /// minimization throws, the current values are kept.
    ///
    /// @param rs    Residual statistics.
    /// @param theta Parameter vector to write into.
    /// @param off   Offset of the residual parameters in @p theta.
    void residualThetaFromStats(const Vec& rs, Vec& theta, int off) const {
        constexpr double lo = -18.0, hi = 12.0;
        if (residType_ != resmod::ErrorKind::Combined) {
            theta[off] = std::clamp(0.5 * std::log(std::max(rs[0], 1e-300)), lo, hi);
            return;
        }
        // Combined: g = a + b|f|;  min sum [ s_ij/g^2 + 2 log g ]  in (log a, log b).
        const long N = nObs_;
        auto obj = [&](const Vec& q) -> double {
            const double a = std::exp(q[0]), b = std::exp(q[1]);
            double v = 0.0;
            for (long k = 0; k < N; ++k) {
                const double g = a + b * rs[2 * k + 1];
                if (!(g > 0.0)) return std::numeric_limits<double>::infinity();
                v += rs[2 * k] / (g * g) + 2.0 * std::log(g);
            }
            return v;
        };
        Vec q0(2); q0 << theta[off], theta[off + 1];
        try {
            Vec sc = Vec::Constant(2, 0.3);
            const NMOutcome o = nelderMeadEx(obj, q0, sc, 400, 1e-10);
            theta[off]     = std::clamp(o.x[0], lo, hi);
            theta[off + 1] = std::clamp(o.x[1], lo, hi);
        } catch (const std::exception&) { /* Keep the current value. */ }
    }

    /// @brief Indices of theta that are log standard deviations of Omega.
    ///
    /// Used by the Simulated Annealing variance floor.
    ///
    /// @return Indices of the log-sd entries.
    std::vector<int> varianceIndices() const override {
        return covDiagIndices(p_, p_, spec_.fullOmega);
    }

    // --- Convenience access -------------------------------------------------

    /// @brief Unpacks theta into (mu, Omega, residual parameters).
    /// @param theta Parameter vector.
    /// @param mu    Output mean vector.
    /// @param Omega Output covariance matrix.
    /// @param pres  Output residual error parameters.
    void unpack(const Vec& theta, Vec& mu, Mat& Omega, resmod::Pres& pres) const {
        mu = theta.head(p_);
        const int nOm = nOmega();
        if (!spec_.fullOmega) {
            Omega = Mat::Zero(p_, p_);
            for (int j = 0; j < p_; ++j) Omega(j, j) = std::exp(2.0 * theta[p_ + j]);
        } else {
            Mat L = Mat::Zero(p_, p_);
            int k = p_;
            for (int j = 0; j < p_; ++j) for (int i = j; i < p_; ++i) {
                const double v = theta[k++];
                L(i, j) = (i == j) ? std::exp(v) : v;
            }
            Omega = L * L.transpose();
        }
        decodeResid(theta, p_ + nOm, pres);
    }

    /// @brief Configured residual error type.
    const resmod::ErrorKind& residType() const { return residType_; }

    /// @brief Dimension of the individual effect.
    int dimPhi() const { return p_; }

private:
    Spec spec_;
    std::vector<std::vector<double>> t_, y_;
    int p_, n_;
    resmod::ErrorKind residType_;
    bool hasCov_ = false;
    std::vector<Vec> cov_;                     // Subject covariates.
    long nObs_ = 0;

    double predictAt(int i, int j, const Vec& phi) const {
        static const Vec none;
        const Vec& x = hasCov_ ? cov_[uz(i)] : none;
        return spec_.predictCov ? spec_.predictCov(phi, t_[uz(i)][uz(j)], x) : spec_.predict(phi, t_[uz(i)][uz(j)]);
    }

    // Cache of the prior decomposition for the last theta.
    struct PriorCache {
        Vec theta, mu; Mat Omega, L; double logdet = 0.0;
        resmod::Pres pres; bool ok = false, valid = false;
    };
    mutable PriorCache pc_;

    const PriorCache& cacheFor(const Vec& theta) const {
        if (pc_.valid && pc_.theta.size() == theta.size() &&
            (pc_.theta.array() == theta.array()).all())
            return pc_;
        unpack(theta, pc_.mu, pc_.Omega, pc_.pres);
        Eigen::LLT<Mat> llt(pc_.Omega);
        pc_.ok = (llt.info() == Eigen::Success);
        if (pc_.ok) {
            pc_.L = llt.matrixL();
            pc_.logdet = 2.0 * pc_.L.diagonal().array().log().sum();
        }
        pc_.theta = theta; pc_.valid = true;
        return pc_;
    }

    void validateData() const {
        for (int i = 0; i < n_; ++i) {
            if (t_[uz(i)].size() != y_[uz(i)].size())
                throw std::invalid_argument("subject with t and y of different lengths");
            for (size_t j = 0; j < y_[uz(i)].size(); ++j) {
                if (!std::isfinite(y_[uz(i)][j]) || !std::isfinite(t_[uz(i)][j]))
                    throw std::invalid_argument("non-finite t/y");
                // resmod::resid throws domain_error if y <= 0 in the exponential
                // model: reject it here, not inside the MCMC.
                if (residType_ == resmod::ErrorKind::Exponential && !(y_[uz(i)][j] > 0.0))
                    throw std::invalid_argument("exponential error model requires y > 0");
            }
        }
    }

    int nOmega() const { return spec_.fullOmega ? p_ * (p_ + 1) / 2 : p_; }
    int nResid() const { return resmod::nParams(residType_); }

    void validateSpec() const {
        if (residType_ == resmod::ErrorKind::Combined2 || residType_ == resmod::ErrorKind::Power)
            throw std::invalid_argument("GenericModel does not support this error model: use smx::saem (NlmeModel)");
        if (spec_.mu0.size() != 0 && spec_.mu0.size() != p_)
            throw std::invalid_argument("mu0 has incorrect dimension");
        if (spec_.sd0.size() != 0 && spec_.sd0.size() != p_)
            throw std::invalid_argument("sd0 has incorrect dimension");
        if (spec_.fullOmega && spec_.Omega0.size() != 0 &&
            (spec_.Omega0.rows() != p_ || spec_.Omega0.cols() != p_))
            throw std::invalid_argument("Omega0 has incorrect dimension");
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
                th[off] = std::log(std::max(eps, pres.a)); break;
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

} // namespace mcmckl
