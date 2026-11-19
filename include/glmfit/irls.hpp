// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file irls.hpp
 * @brief Standalone, robust IRLS solver for generalized linear models (C++20, Eigen).
 *
 * Public contract (same as the previous solver):
 *   - Inputs : X, Y, prior weights, offset, either a `fam_code` (+ FamilyParams) resolved through
 *              families.hpp or six user callbacks, tol, maxit, `type`, `is_big_matrix`, `firth_flag`.
 *   - Outputs: get_beta, get_se, get_vcov, get_eta, get_mu, get_weights (prior), get_w (working W),
 *              get_dev, get_std_dev, get_log_det_XtWX, get_rank, get_converged, get_firth.
 *   - Extras : get_iterations, get_aliased, result() (all of the above in one struct).
 *
 * `type` selects the linear-algebra back end: 0 ColPivQR, 1 HouseholderQR, 2 LLT, 3 LDLT,
 * 4 FullPivQR, 5 SVD, 6 Auto (default: Jacobi-scaled Cholesky with automatic QR fallback; see wls_kernels.hpp). Types 1-3 fall back automatically to the pivoted QR path
 * when the weighted design is rank deficient, so rank deficiency never aborts a fit.
 *
 * Convention for rank-deficient designs (types 0, 1, 2, 3, 4): aliased coefficients are 0 and their
 * se / vcov rows+columns are NaN. Type 5 returns the minimum-norm solution and a pseudo-inverse vcov.
 *
 * Firth (Jeffreys-prior) bias reduction uses the exact adjusted score for any link:
 *   xi_i = phi h_i (2 mu''_i / mu'_i - V'(mu_i) mu'_i / V_i) / (2 W_i),  z*_i = z_i + xi_i,
 * with h_i the leverages of the current weighted design. For canonical links this reduces to the
 * usual phi h_i V'(mu_i) / (2 W_i) term.
 */

#ifndef _GLMFIT_IRLS_HPP
#define _GLMFIT_IRLS_HPP

#include "families.hpp"
#include "wls_kernels.hpp"
#include "design.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace glm {

/// Optional tuning knobs (all have sensible defaults; aggregate, so designated initialisers work).
struct IrlsOptions {
    double       rank_tol     = 1e-11;  ///< relative pivot / singular-value cut-off for rank detection (per equilibrated column)
    int          max_halvings = 30;     ///< max step halvings per iteration
    double       max_step     = 5.0;    ///< Firth only: cap on max|delta beta| per iteration
    Eigen::Index chunk_rows   = 0;      ///< rows per streaming block; 0 = auto (FASTGLM_CHUNK_ROWS / 16384)
    bool         equilibrate  = true;   ///< scale columns of diag(w) X to unit norm before factorising
    bool         freeze_aliased = true; ///< detect structural rank once and drop aliased columns from the design
    double       max_eta_step = 0.0;    ///< > 0: cap on max|Delta eta| of the first trial step of an iteration (0 = off)
    Eigen::Index eval_chunk   = 8192;   ///< rows per block of the generic (array) family evaluation
    // --- v3 ---
    bool         observed_newton = true; ///< non-canonical links: switch to the observed information after `observed_start` iterations
    int          observed_start  = 2;    ///< Fisher-scoring iterations before trying observed Newton
    bool         levenberg_marquardt = true; ///< damped fallback when the line search fails
    double       lm_lambda0      = 1e-4; ///< first LM damping (on the Jacobi-scaled Gram matrix); x10 per retry
    int          lm_tries        = 16;       ///< lambda goes from lm_lambda0 up to lm_lambda0 * 10^(lm_tries-1)
    bool         compact_zero_weights = true; ///< drop rows with prior weight 0 from every pass
    bool         fused_kernels   = false; ///< opt-in scalar single-pass kernels (Poisson/NB/binomial-logit/Gaussian/Gamma-log). Bit-compatible with
                                          ///< the array path, but measured ~1.4x SLOWER on x86: Eigen's SIMD exp/log in the blocked path beat libm
};

/// Everything a fit produces, in one place.
struct IrlsResult {
    Eigen::VectorXd  coefficients;      ///< beta
    Eigen::VectorXd  std_errors;        ///< sqrt(diag((X'WX)^-1)), NaN for aliased columns
    Eigen::MatrixXd  vcov;              ///< (X'WX)^-1 (expected information), NaN rows/cols for aliased columns
    Eigen::VectorXd  eta;               ///< linear predictor
    Eigen::VectorXd  mu;                ///< fitted means
    Eigen::VectorXd  prior_weights;     ///< user weights
    Eigen::VectorXd  working_weights;   ///< W = w^2 of the last iteration
    double           deviance      = 0.0;  ///< objective (penalised deviance when Firth is on)
    double           std_deviance  = 0.0;  ///< plain deviance sum(w_i d_i)
    double           log_det_xtwx  = 0.0;
    int              rank          = 0;
    int              iterations    = 0;
    bool             converged     = false;
    bool             firth         = false;
    std::vector<int> aliased;           ///< 0-based indices of aliased columns
    int              n_boundary    = 0; ///< binomial: rows with fitted mean numerically 0 or 1 (possible separation)
};

namespace detail {

/// Families with a scalar single-pass kernel (see `kernel_row`).
enum class Kern { None, PoisLog, NbLog, BinLogit, GaussId, GammaLog };

[[nodiscard]] inline Kern kernel_for(int code) {
    switch (code) {
        case FAM_POISSON_LOG: case FAM_QUASI_POISSON_LOG:        return Kern::PoisLog;
        case FAM_NB_LOG:                                         return Kern::NbLog;
        case FAM_BINOMIAL_LOGIT: case FAM_QUASI_BINOMIAL_LOGIT:  return Kern::BinLogit;
        case FAM_GAUSSIAN_IDENTITY: case FAM_QUASI_GAUSSIAN_IDENTITY: return Kern::GaussId;
        case FAM_GAMMA_LOG: case FAM_QUASI_GAMMA_LOG:            return Kern::GammaLog;
        default:                                                 return Kern::None;
    }
}

/// Canonical (family, link) pairs: observed and expected information coincide, so Fisher scoring is exact Newton.
[[nodiscard]] inline bool is_canonical(int code) {
    switch (code) {
        case FAM_GAUSSIAN_IDENTITY: case FAM_QUASI_GAUSSIAN_IDENTITY:
        case FAM_POISSON_LOG: case FAM_QUASI_POISSON_LOG:
        case FAM_BINOMIAL_LOGIT: case FAM_QUASI_BINOMIAL_LOGIT:
        case FAM_GAMMA_INVERSE: case FAM_QUASI_GAMMA_INVERSE:
        case FAM_INVGAUSS_INVMU2:
            return true;
        default: return false;
    }
}

/**
 * @brief One row of the fused evaluation: mu (optionally), mu', V, w, weighted working residual and the
 *        unit deviance, with exactly the semantics of the array kernels in families.hpp.
 * @return false if eta/mu are invalid (non-finite eta, or mu outside the family's range).
 */
template <Kern K>
inline bool kernel_row(double eta, double y, double pw, double theta, bool recompute,
                       double& mu, double& dmu, double& var, double& w, double& r, double& dev)
{
    constexpr double eps  = 100.0 * std::numeric_limits<double>::epsilon();   // thresh_eps()
    constexpr double deps = std::numeric_limits<double>::epsilon();

    if (recompute) {
        if (!std::isfinite(eta)) return false;
        if constexpr (K == Kern::BinLogit) {
            const double t = eta > 30.0 ? 1.0 / deps : (eta < -30.0 ? deps : std::exp(eta));
            mu = t / (1.0 + t);
        } else if constexpr (K == Kern::GaussId) {
            mu = eta;
        } else {
            mu = std::exp(eta);
        }
        if (!std::isfinite(mu)) return false;
        if constexpr (K == Kern::BinLogit) { if (!(mu > 0.0 && mu < 1.0)) return false; }
        else if constexpr (K != Kern::GaussId) { if (!(mu > 0.0)) return false; }
    }

    if constexpr (K == Kern::BinLogit) {
        if (std::abs(eta) > 30.0) dmu = deps;
        else { const double e = std::exp(eta), q = 1.0 + e; dmu = e / (q * q); }
    } else if constexpr (K == Kern::GaussId) {
        dmu = 1.0;
    } else {
        dmu = recompute ? mu : std::exp(eta);   // d mu / d eta = mu for the log link
    }

    if constexpr (K == Kern::PoisLog) {
        var = mu; w = std::sqrt(pw * mu);
    } else if constexpr (K == Kern::NbLog) {
        var = mu + mu * mu / theta;
        const double tm = std::max(theta + mu, eps);
        w = std::sqrt(pw * mu * theta / tm);
    } else if constexpr (K == Kern::BinLogit) {
        var = mu * (1.0 - mu); w = std::sqrt(pw * mu * (1.0 - mu));
    } else if constexpr (K == Kern::GaussId) {
        var = 1.0; w = std::sqrt(pw * dmu * dmu / var);
    } else {  // GammaLog
        var = mu * mu; w = std::sqrt(pw * dmu * dmu / var);
    }
    if (!std::isfinite(w)) w = 0.0;
    double rr = std::sqrt(pw / var) * (y - mu) * (dmu >= 0.0 ? 1.0 : -1.0);
    if (!std::isfinite(rr)) rr = 0.0;
    r = (w > 0.0) ? rr : 0.0;

    if constexpr (K == Kern::PoisLog) {
        dev = y > 0.0 ? 2.0 * (y * std::log(y / std::max(mu, eps)) - (y - mu)) : 2.0 * mu;
    } else if constexpr (K == Kern::NbLog) {
        double t1 = y * std::log(std::max(y, eps) / std::max(mu, eps));
        if (!std::isfinite(t1)) t1 = 0.0;
        dev = 2.0 * (t1 + (y + theta) * std::log1p((mu - y) / (y + theta)));
    } else if constexpr (K == Kern::BinLogit) {
        double t1 = y * std::log(y / std::max(mu, eps));
        double t2 = (1.0 - y) * std::log((1.0 - y) / std::max(1.0 - mu, eps));
        if (!std::isfinite(t1)) t1 = 0.0;
        if (!std::isfinite(t2)) t2 = 0.0;
        dev = 2.0 * (t1 + t2);
    } else if constexpr (K == Kern::GaussId) {
        dev = (y - mu) * (y - mu);
    } else {
        dev = -2.0 * (std::log(std::max(y, eps) / std::max(mu, eps)) - (y - mu) / std::max(mu, eps));
    }
    return true;
}

}  // namespace detail

class IrlsSolver {
public:
    using Vector      = Eigen::VectorXd;
    using Matrix      = Eigen::MatrixXd;
    using LinkFun     = std::function<void(const Eigen::ArrayXd&, Eigen::ArrayXd&)>;
    using DevResidFun = std::function<double(const Eigen::ArrayXd&, const Eigen::ArrayXd&, const Eigen::ArrayXd&)>;
    using ValidFun    = std::function<bool(const Eigen::ArrayXd&)>;
    using SparseMatrix = Eigen::SparseMatrix<double>;

    /// Callback-based constructor (dense design; use `fam_code >= 0` to prefer the built-in families.hpp kernels).
    IrlsSolver(const Eigen::Map<const Matrix>& X,
               const Eigen::Map<const Vector>& Y,
               const Eigen::Map<const Vector>& weights,
               const Eigen::Map<const Vector>& offset,
               LinkFun variance_fun,
               LinkFun mu_eta_fun,
               LinkFun linkinv_fun,
               DevResidFun dev_resids_fun,
               ValidFun valideta_fun,
               ValidFun validmu_fun,
               double tol = 1e-6,
               int maxit = 100,
               int type = 6,
               bool is_big_matrix = false,
               int fam_code = -1,
               const FamilyParams& fam_params = FamilyParams(),
               bool firth_flag = false,
               const IrlsOptions& opts = IrlsOptions())
        : IrlsSolver(std::make_unique<detail::DenseDesign>(X), Y, weights, offset,
                     std::move(variance_fun), std::move(mu_eta_fun), std::move(linkinv_fun),
                     std::move(dev_resids_fun), std::move(valideta_fun), std::move(validmu_fun),
                     tol, maxit, type, is_big_matrix, fam_code, fam_params, firth_flag, opts) {}

    /// Constructor driven by a FamilyCode, dense design (the usual entry point).
    IrlsSolver(const Eigen::Map<const Matrix>& X,
               const Eigen::Map<const Vector>& Y,
               const Eigen::Map<const Vector>& weights,
               const Eigen::Map<const Vector>& offset,
               int fam_code,
               const FamilyParams& fam_params = FamilyParams(),
               double tol = 1e-6,
               int maxit = 100,
               int type = 6,
               bool is_big_matrix = false,
               bool firth_flag = false,
               const IrlsOptions& opts = IrlsOptions())
        : IrlsSolver(std::make_unique<detail::DenseDesign>(X), Y, weights, offset,
                     LinkFun{}, LinkFun{}, LinkFun{}, DevResidFun{}, ValidFun{}, ValidFun{},
                     tol, maxit, type, is_big_matrix, fam_code, fam_params, firth_flag, opts) {}

    /// Constructor driven by a FamilyCode, sparse (CSC) design. `X` must outlive the solver.
    /// X'WX is formed with sparse products and the small p x p system is solved densely.
    IrlsSolver(const SparseMatrix& X,
               const Eigen::Map<const Vector>& Y,
               const Eigen::Map<const Vector>& weights,
               const Eigen::Map<const Vector>& offset,
               int fam_code,
               const FamilyParams& fam_params = FamilyParams(),
               double tol = 1e-6,
               int maxit = 100,
               int type = 6,
               bool is_big_matrix = false,
               bool firth_flag = false,
               const IrlsOptions& opts = IrlsOptions())
        : IrlsSolver(std::make_unique<detail::SparseDesign>(X), Y, weights, offset,
                     LinkFun{}, LinkFun{}, LinkFun{}, DevResidFun{}, ValidFun{}, ValidFun{},
                     tol, maxit, type, is_big_matrix, fam_code, fam_params, firth_flag, opts) {}

    /// Sets the starting coefficients, fitted means and linear predictor (must be called before solve()).
    /// `mu_in` / `eta_in` always have one entry per row of X (zero-weight rows are dropped internally).
    void init_parms(const Eigen::Map<const Vector>& start,
                    const Eigen::Map<const Vector>& mu_in,
                    const Eigen::Map<const Vector>& eta_in) {
        if (start.size() != p_ || mu_in.size() != nfull_ || eta_in.size() != nfull_)
            throw std::invalid_argument("init_parms: start/mu/eta have inconsistent sizes.");
        beta_      = start;
        beta_prev_ = start;
        if (rows_.empty()) {
            eta_ = eta_in.array();
            mu_  = mu_in.array();
        } else {
            for (Eigen::Index k = 0; k < nw_; ++k) {
                eta_[k] = eta_in[rows_[static_cast<std::size_t>(k)]];
                mu_[k]  = mu_in[rows_[static_cast<std::size_t>(k)]];
            }
        }
        eta_consistent_ = false;
        logdet_    = 0.0;
        phi_       = 1.0;
        refresh_state();
        devold_      = dev_;
        initialized_ = true;
        solved_      = false;
        rank_        = static_cast<int>(p_);
        aliased_.clear();
    }

    /// Family-aware defaults (like R's `mustart`): beta = 0, mu = mustart(y, w), eta = g(mu).
    /// Much safer than mu = g^-1(offset) for count / positive families. Built-in families only.
    void init_default() {
        if (fam_code_ < 0)
            throw std::invalid_argument("init_default: needs a built-in family code (fam_code >= 0).");
        ArrayXd mu(nfull_), eta(nfull_);
        glm::mustart(fam_code_, fam_params_, y_full_, pw_full_, mu);
        glm::linkfun(fam_code_, fam_params_, mu, eta);
        const Vector z0 = Vector::Zero(p_);
        const Vector mv = mu.matrix(), ev = eta.matrix();
        init_parms(Eigen::Map<const Vector>(z0.data(), p_),
                   Eigen::Map<const Vector>(mv.data(), nfull_),
                   Eigen::Map<const Vector>(ev.data(), nfull_));
    }

    /// Replaces family hyper-parameters (e.g. NB theta) and recomputes weights and deviance under them.
    void set_fam_params(const FamilyParams& p) {
        fam_params_ = p;
        if (initialized_) { refresh_state(); devold_ = dev_; }
    }

    /// Changes the convergence tolerance (e.g. loose inner solves inside an outer loop).
    void set_tol(double tol) {
        if (!(tol > 0.0)) throw std::invalid_argument("IrlsSolver::set_tol: tol must be positive.");
        tol_ = tol;
    }

    /**
     * @brief Runs IRLS.
     * @param maxit_param max iterations (<= 0 uses the constructor's maxit).
     * @return iterations performed.
     * @details Each iteration tries, in order: observed-information Newton (non-canonical links, after a
     *          few Fisher iterations), Fisher scoring with step halving, Levenberg-Marquardt damping, and
     *          finally the Fisher step accepted after halving (the classical glm.fit behaviour).
     * @throws std::runtime_error if no valid iterate can be found; std::logic_error without init_parms().
     */
    int solve(int maxit_param = -1) {
        if (!initialized_)
            throw std::logic_error("IrlsSolver::solve: call init_parms() first.");
        if (!rank_checked_) setup_active_set();
        const int limit = maxit_param > 0 ? maxit_param : maxit_;
        conv_ = false;
        iterations_ = 0;
        n_obs_steps_ = 0;
        n_lm_steps_  = 0;

        for (int it = 0; it < limit; ++it) {
            beta_prev_ = beta_;
            eta_prev_  = eta_;
            devold_    = dev_;
            const bool zform = !eta_consistent_;
            const bool use_obs = opts_.observed_newton && !firth_ && !zform && noncanonical_ &&
                                 it >= opts_.observed_start;
            const bool use_lm  = opts_.levenberg_marquardt && !firth_ && !zform;

            Vector sol;
            double decrement = 0.0;
            LS status = LS::Invalid;

            if (use_obs) {
                if (observed_direction(sol)) status = line_search(sol, zform, /*strict=*/true, opts_.max_halvings, decrement, it);
                if (status != LS::Ok) restore_state(); else ++n_obs_steps_;
            }
            if (status != LS::Ok) {
                fisher_direction(zform, sol, it);
                status = line_search(sol, zform, /*strict=*/use_lm, opts_.max_halvings, decrement, it);
                if (status != LS::Ok && use_lm) {
                    restore_state();
                    status = lm_search(decrement, it);
                    if (status == LS::Ok) ++n_lm_steps_;
                    if (status != LS::Ok) {
                        restore_state();
                        fisher_direction(zform, sol, it);
                        status = line_search(sol, zform, /*strict=*/false, opts_.max_halvings, decrement, it);
                    }
                }
            }
            if (status == LS::Invalid) throw_nonfinite(it);

            eta_consistent_ = true;
            iterations_ = it + 1;

            bool done;
            if (firth_) {
                done = (beta_ - beta_prev_).cwiseAbs().maxCoeff() < tol_;
            } else {
                const bool dev_ok = std::abs(dev_ - devold_) / (0.1 + std::abs(dev_)) < tol_;
                // a flat deviance with a large Newton decrement means the line search stalled: keep going
                const bool dec_ok = zform || decrement / (0.1 + std::abs(dev_)) < std::sqrt(tol_);
                done = dev_ok && dec_ok;
            }
            if (done) { conv_ = true; break; }
        }
        finalize();
        return iterations_;
    }

    // ---- results (same names as the previous solver) -----------------------------------------
    [[nodiscard]] Vector get_beta()    const { return beta_; }
    [[nodiscard]] Vector get_eta()     const { return rows_.empty() ? Vector(eta_.matrix()) : Vector(eta_out_.matrix()); }
    [[nodiscard]] Vector get_se()      const { return se_; }
    [[nodiscard]] Vector get_mu()      const { return rows_.empty() ? Vector(mu_.matrix()) : Vector(mu_out_.matrix()); }
    [[nodiscard]] Vector get_weights() const { return pw_full_.matrix(); }                 ///< prior weights
    [[nodiscard]] Vector get_w()       const {                                              ///< working weights W = w^2
        return rows_.empty() ? Vector(w_.square().matrix()) : Vector(w_out_.square().matrix());
    }
    [[nodiscard]] Matrix get_vcov()    const { return vcov_; }
    [[nodiscard]] double get_dev()     const noexcept { return dev_; }
    [[nodiscard]] double get_std_dev() const noexcept { return std_dev_; }
    [[nodiscard]] double get_log_det_XtWX() const noexcept { return logdet_; }
    [[nodiscard]] int    get_rank()    const noexcept { return rank_; }
    [[nodiscard]] bool   get_converged() const noexcept { return conv_; }
    [[nodiscard]] bool   get_firth()   const noexcept { return firth_; }
    [[nodiscard]] int    get_iterations() const noexcept { return iterations_; }
    [[nodiscard]] const std::vector<int>& get_aliased() const noexcept { return aliased_; }
    /// Binomial families: rows (prior weight > 0) whose fitted mean is within 10 eps of 0 or 1.
    [[nodiscard]] int    get_boundary_count() const noexcept { return n_boundary_; }
    /// Rows actually used by the passes (rows with prior weight > 0 when compaction is active).
    [[nodiscard]] Eigen::Index get_active_rows() const noexcept { return nw_; }
    /// Diagnostics: iterations accepted with the observed-information step / the Levenberg-Marquardt step.
    [[nodiscard]] int    get_observed_steps() const noexcept { return n_obs_steps_; }
    [[nodiscard]] int    get_lm_steps() const noexcept { return n_lm_steps_; }

    [[nodiscard]] IrlsResult result() const {
        IrlsResult r;
        r.coefficients    = beta_;
        r.std_errors      = se_;
        r.vcov            = vcov_;
        r.eta             = get_eta();
        r.mu              = get_mu();
        r.prior_weights   = pw_full_.matrix();
        r.working_weights = get_w();
        r.deviance        = dev_;
        r.std_deviance    = std_dev_;
        r.log_det_xtwx    = logdet_;
        r.rank            = rank_;
        r.iterations      = iterations_;
        r.converged       = conv_;
        r.firth           = firth_;
        r.aliased         = aliased_;
        r.n_boundary      = n_boundary_;
        return r;
    }

private:
    using ArrayXd = Eigen::ArrayXd;
    using Index   = Eigen::Index;
    enum class LS { Ok, Increased, Invalid };

    /// Main constructor (all public ones delegate here).
    IrlsSolver(std::unique_ptr<detail::Design> design,
               const Eigen::Map<const Vector>& Y,
               const Eigen::Map<const Vector>& weights,
               const Eigen::Map<const Vector>& offset,
               LinkFun variance_fun, LinkFun mu_eta_fun, LinkFun linkinv_fun, DevResidFun dev_resids_fun,
               ValidFun valideta_fun, ValidFun validmu_fun,
               double tol, int maxit, int type, bool is_big_matrix, int fam_code,
               const FamilyParams& fam_params, bool firth_flag, const IrlsOptions& opts)
        : variance_fun_(std::move(variance_fun)), mu_eta_fun_(std::move(mu_eta_fun)),
          linkinv_fun_(std::move(linkinv_fun)), dev_resids_fun_(std::move(dev_resids_fun)),
          valideta_fun_(std::move(valideta_fun)), validmu_fun_(std::move(validmu_fun)),
          fam_code_(fam_code), fam_params_(fam_params),
          tol_(tol), maxit_(maxit), firth_(firth_flag), opts_(opts),
          wls_(design->cols(), to_method(type), opts.rank_tol, opts.equilibrate)
    {
        base_  = std::move(design);
        nfull_ = base_->rows();
        nw_    = nfull_;
        p_     = base_->cols();
        y_full_  = Y.array();
        pw_full_ = weights.array();
        off_full_ = offset.array();
        y_ = y_full_; pw_ = pw_full_; off_ = off_full_;
        validate_inputs();
        validate_family();
        kernel_ = (opts_.fused_kernels && fam_code_ >= 0) ? detail::kernel_for(fam_code_) : detail::Kern::None;
        noncanonical_ = fam_code_ < 0 ? true : !detail::is_canonical(fam_code_);

        qr_chunk_ = opts_.chunk_rows > 0 ? opts_.chunk_rows
                                         : (is_big_matrix ? detail::default_chunk_rows() : nfull_);
        ne_chunk_ = opts_.chunk_rows > 0 ? opts_.chunk_rows : detail::default_chunk_rows();

        beta_      = Vector::Zero(p_);
        beta_prev_ = Vector::Zero(p_);
        delta_     = Vector::Zero(p_);
        vcov_      = Matrix::Zero(p_, p_);
        se_        = Vector::Zero(p_);
        resize_state(nfull_);
        act_.resize(static_cast<std::size_t>(p_));
        std::iota(act_.begin(), act_.end(), 0);
        if (fam_code_ < 0) { y_cb_ = y_; pw_cb_ = pw_; }
    }

    // ---- data ---------------------------------------------------------------------------------
    std::unique_ptr<detail::Design> base_;   // full design (all rows / columns)
    std::unique_ptr<detail::Design> sub_;    // compacted rows + active columns (when needed)
    ArrayXd y_full_, pw_full_, off_full_;    // as supplied
    ArrayXd y_, pw_, off_;                   // work copies (rows with prior weight 0 removed when compacted)
    Index nfull_ = 0, nw_ = 0, p_ = 0;
    std::vector<Index> rows_;                // kept rows when compacted (empty = all rows)

    LinkFun variance_fun_, mu_eta_fun_, linkinv_fun_;
    DevResidFun dev_resids_fun_;
    ValidFun valideta_fun_, validmu_fun_;
    ArrayXd y_cb_, pw_cb_;  // owned copies, only used with callbacks

    int fam_code_;
    FamilyParams fam_params_;
    double tol_;
    int maxit_;
    bool firth_;
    IrlsOptions opts_;
    detail::Kern kernel_ = detail::Kern::None;
    bool noncanonical_ = true;
    Index qr_chunk_ = 0, ne_chunk_ = 0;

    detail::WlsSolver wls_;

    // ---- active column set (structural rank, frozen) -----------------------------------------
    std::vector<int> act_;          // indices of the columns kept in the design
    std::vector<int> aliased0_;     // structurally aliased columns (dropped)
    bool rank_checked_ = false;
    bool fisher_valid_ = false;     // wls_ currently holds the Fisher (expected-information) factorisation

    // ---- state --------------------------------------------------------------------------------
    Vector  beta_, beta_prev_, delta_, se_;
    Matrix  vcov_;
    ArrayXd eta_, eta_prev_, mu_, dmu_, var_, w_, r_, xd_, tw_;
    ArrayXd h_, d2_, vprime_, xi_;       // second-order / Firth work arrays
    ArrayXd eta_out_, mu_out_, w_out_;   // full-length results when rows were compacted
    double dev_ = 0.0, devold_ = 0.0, std_dev_ = 0.0, logdet_ = 0.0, phi_ = 1.0;
    int  rank_ = 0, iterations_ = 0, n_boundary_ = 0, n_obs_steps_ = 0, n_lm_steps_ = 0;
    bool conv_ = false, initialized_ = false, solved_ = false, eta_consistent_ = false;
    std::vector<int> aliased_;

    [[nodiscard]] const detail::Design& D() const { return sub_ ? *sub_ : *base_; }

    void resize_state(Index n) {
        eta_.setZero(n); eta_prev_.setZero(n); mu_.setZero(n); dmu_.setZero(n); var_.setZero(n);
        w_.setZero(n); r_.setZero(n); xd_.setZero(n); tw_.setZero(n);
    }

    // ---- validation ---------------------------------------------------------------------------
    static detail::Method to_method(int t) {
        if (t < 0 || t > 6)
            throw std::invalid_argument("IrlsSolver: type must be in 0..6, got " + std::to_string(t) + ".");
        return static_cast<detail::Method>(t);
    }

    void validate_inputs() const {
        if (nfull_ <= 0 || p_ <= 0) throw std::invalid_argument("IrlsSolver: X must be non-empty.");
        if (y_full_.size() != nfull_ || pw_full_.size() != nfull_ || off_full_.size() != nfull_)
            throw std::invalid_argument("IrlsSolver: Y, weights and offset must have nrow(X) entries.");
        if (!(tol_ > 0.0)) throw std::invalid_argument("IrlsSolver: tol must be positive.");
        if (!y_full_.isFinite().all() || !off_full_.isFinite().all())
            throw std::invalid_argument("IrlsSolver: Y and offset must be finite.");
        if (!pw_full_.isFinite().all() || (pw_full_ < 0.0).any())
            throw std::invalid_argument("IrlsSolver: prior weights must be finite and non-negative.");
    }

    void validate_family() const {
        if (fam_code_ >= 0) {
            if (!is_irls_family(fam_code_))
                throw std::invalid_argument(
                    "IrlsSolver: fam_code " + std::to_string(fam_code_) + " is not a valid IRLS family. "
                    "MLE families (Simplex, Beta, ZIP, Hurdle, Truncated, ...) need a dedicated solver.");
        } else if (!variance_fun_ || !mu_eta_fun_ || !linkinv_fun_ || !dev_resids_fun_) {
            throw std::invalid_argument(
                "IrlsSolver: without fam_code, variance/mu_eta/linkinv/dev_resids callbacks are required.");
        }
    }

    // ---- active set, row compaction -----------------------------------------------------------
    /// One-off set-up: drop zero-weight rows, detect structural rank (ordered, R-style) and freeze the
    /// retained columns; builds the reduced design only when something was dropped.
    void setup_active_set() {
        rank_checked_ = true;
        act_.assign(static_cast<std::size_t>(p_), 0);
        std::iota(act_.begin(), act_.end(), 0);
        aliased0_.clear();
        sub_.reset();
        rows_.clear();

        if (opts_.freeze_aliased) {
            const ArrayXd w0 = pw_full_.sqrt();
            detail::WlsSolver tmp(p_, detail::Method::ColPivQR, opts_.rank_tol, opts_.equilibrate);
            const std::vector<int> al = base_->aliased(tmp, w0, qr_chunk_);
            if (static_cast<Index>(al.size()) == p_)
                throw std::runtime_error("Numerical error: design matrix has rank 0 under the current weights "
                                         "(all working weights are zero or non-finite).");
            if (!al.empty()) {
                aliased0_ = al;
                std::vector<char> drop(static_cast<std::size_t>(p_), 0);
                for (int j : aliased0_) drop[static_cast<std::size_t>(j)] = 1;
                act_.clear();
                for (int j = 0; j < p_; ++j) if (!drop[static_cast<std::size_t>(j)]) act_.push_back(j);
                for (int j : aliased0_) { beta_[j] = 0.0; beta_prev_[j] = 0.0; }
                eta_consistent_ = false;   // the start eta may have contained the dropped columns
            }
        }

        if (opts_.compact_zero_weights) {
            std::vector<Index> keep;
            keep.reserve(static_cast<std::size_t>(nfull_));
            for (Index i = 0; i < nfull_; ++i) if (pw_full_[i] > 0.0) keep.push_back(i);
            if (keep.empty())
                throw std::runtime_error("Numerical error: all prior weights are zero.");
            if (static_cast<Index>(keep.size()) < nfull_) rows_ = std::move(keep);
        }

        if (!aliased0_.empty() || !rows_.empty())
            sub_ = base_->subset(rows_.empty() ? nullptr : &rows_, act_);

        if (!rows_.empty()) {
            nw_ = static_cast<Index>(rows_.size());
            const auto sel = [&](ArrayXd& a) {
                ArrayXd b(nw_);
                for (Index k = 0; k < nw_; ++k) b[k] = a[rows_[static_cast<std::size_t>(k)]];
                a.swap(b);
            };
            sel(y_); sel(pw_); sel(off_); sel(eta_); sel(mu_);
            dmu_.setZero(nw_); var_.setZero(nw_); w_.setZero(nw_); r_.setZero(nw_);
            xd_.setZero(nw_); tw_.setZero(nw_); eta_prev_.setZero(nw_);
            if (fam_code_ < 0) { y_cb_ = y_; pw_cb_ = pw_; }
            refresh_state();
            devold_ = dev_;
        }
        wls_.reset(static_cast<Index>(act_.size()));
    }

    // ---- family dispatch (families.hpp fast path or user callbacks) ---------------------------
    void eval_linkinv(const ArrayXd& eta, ArrayXd& mu) const {
        if (fam_code_ >= 0) glm::linkinv(fam_code_, fam_params_, eta, mu);
        else                linkinv_fun_(eta, mu);
    }
    void eval_mu_eta(const ArrayXd& eta, ArrayXd& d) const {
        if (fam_code_ >= 0) glm::mu_eta(fam_code_, fam_params_, eta, d);
        else                mu_eta_fun_(eta, d);
    }
    void eval_variance(const ArrayXd& mu, ArrayXd& v) const {
        if (fam_code_ >= 0) glm::variance(fam_code_, fam_params_, mu, v);
        else                variance_fun_(mu, v);
    }
    [[nodiscard]] bool eta_valid() const {
        if (fam_code_ >= 0) return glm::valideta(fam_code_, fam_params_, eta_);
        return eta_.isFinite().all() && (!valideta_fun_ || valideta_fun_(eta_));
    }
    [[nodiscard]] bool mu_valid() const {
        if (fam_code_ >= 0) return glm::validmu(fam_code_, fam_params_, mu_);
        return mu_.isFinite().all() && (!validmu_fun_ || validmu_fun_(mu_));
    }

    /// Weighted working residual r = w (y-mu)/mu' = sgn(mu') sqrt(pw/V) (y-mu), zero on zero-weight rows.
    static void working_residual(const Eigen::Ref<const ArrayXd>& pw, const Eigen::Ref<const ArrayXd>& v,
                                 const Eigen::Ref<const ArrayXd>& d, const Eigen::Ref<const ArrayXd>& y,
                                 const Eigen::Ref<const ArrayXd>& m, const Eigen::Ref<const ArrayXd>& w,
                                 Eigen::Ref<ArrayXd> r) {
        const ArrayXd sgn = (d >= 0.0).cast<double>() * 2.0 - 1.0;
        ArrayXd rr = (pw / v).sqrt() * (y - m) * sgn;
        rr = rr.isFinite().select(rr, 0.0);
        r = (w > 0.0).select(rr, 0.0);
    }

    /**
     * @brief Evaluation at the current eta_: (optionally) mu, then mu', V, w, working residual and the
     *        deviance in one go. Scalar single-pass kernels for the common families, block-wise array
     *        kernels for the rest. @return false if eta/mu are invalid or non-finite.
     */
    bool evaluate(bool recompute_mu, double& dev_sum) {
        if (fam_code_ < 0) return evaluate_generic(recompute_mu, dev_sum);
        switch (kernel_) {
            case detail::Kern::PoisLog:  return evaluate_kernel<detail::Kern::PoisLog>(recompute_mu, dev_sum);
            case detail::Kern::NbLog:    return evaluate_kernel<detail::Kern::NbLog>(recompute_mu, dev_sum);
            case detail::Kern::BinLogit: return evaluate_kernel<detail::Kern::BinLogit>(recompute_mu, dev_sum);
            case detail::Kern::GaussId:  return evaluate_kernel<detail::Kern::GaussId>(recompute_mu, dev_sum);
            case detail::Kern::GammaLog: return evaluate_kernel<detail::Kern::GammaLog>(recompute_mu, dev_sum);
            default:                     return evaluate_blocks(recompute_mu, dev_sum);
        }
    }

    template <detail::Kern K>
    bool evaluate_kernel(bool recompute_mu, double& dev_sum) {
        double devsum = 0.0;
        int bad = 0;
        const double theta = fam_params_.theta;
        const Index n = nw_;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : devsum, bad) schedule(static) if (n > 16384)
#endif
        for (Index i = 0; i < n; ++i) {
            double mu = recompute_mu ? 0.0 : mu_[i];
            double d, v, w, r, dv;
            if (!detail::kernel_row<K>(eta_[i], y_[i], pw_[i], theta, recompute_mu, mu, d, v, w, r, dv)) {
                ++bad;
                continue;
            }
            mu_[i] = mu; dmu_[i] = d; var_[i] = v; w_[i] = w; r_[i] = r;
            devsum += pw_[i] * dv;
        }
        dev_sum = devsum;
        return bad == 0 && std::isfinite(devsum);
    }

    bool evaluate_blocks(bool recompute_mu, double& dev_sum) {
        const Index chunk = std::max<Index>(256, opts_.eval_chunk);
        const Index nch = (nw_ + chunk - 1) / chunk;
        double devsum = 0.0;
        int bad = 0;
        const int code = fam_code_;
        const FamilyParams prm = fam_params_;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : devsum, bad) schedule(static) if (nch > 1)
#endif
        for (Index c = 0; c < nch; ++c) {
            const Index r0 = c * chunk;
            const Index k  = std::min(chunk, nw_ - r0);
            auto e  = eta_.segment(r0, k);
            auto m  = mu_.segment(r0, k);
            auto d  = dmu_.segment(r0, k);
            auto v  = var_.segment(r0, k);
            auto w  = w_.segment(r0, k);
            const auto pw = pw_.segment(r0, k);
            const auto yy = y_.segment(r0, k);
            double dv = 0.0;
            if (!glm::eval_block(code, prm, recompute_mu, e, yy, pw, m, d, v, w, dv)) { ++bad; continue; }
            auto rseg = r_.segment(r0, k);
            working_residual(pw, v, d, yy, m, w, rseg);
            devsum += dv;
        }
        dev_sum = devsum;
        return bad == 0 && std::isfinite(devsum);
    }

    bool evaluate_generic(bool recompute_mu, double& dev_sum) {
        if (recompute_mu) {
            if (!eta_valid()) return false;
            eval_linkinv(eta_, mu_);
            if (!mu_valid()) return false;
        }
        eval_mu_eta(eta_, dmu_);
        eval_variance(mu_, var_);
        w_ = (pw_ * dmu_.square() / var_).sqrt();
        w_ = w_.isFinite().select(w_, 0.0);
        working_residual(pw_, var_, dmu_, y_, mu_, w_, r_);
        dev_sum = dev_resids_fun_(y_cb_, mu_, pw_cb_);
        return std::isfinite(dev_sum);
    }

    /// Recomputes w, residual, deviance (and dev_) from the stored mu_/eta_ (no validity checks).
    void refresh_state() {
        double d = 0.0;
        (void)evaluate(false, d);
        std_dev_ = d;
        dev_     = firth_ ? std_dev_ - phi_ * logdet_ : std_dev_;
    }

    /// Returns to the accepted iterate (eta_prev_, beta_prev_) after a failed attempt.
    void restore_state() {
        beta_ = beta_prev_;
        eta_  = eta_prev_;
        double d = 0.0;
        (void)evaluate(true, d);
        std_dev_ = d;
        dev_     = devold_;
    }

    /// Classical working response for the very first iteration: tw = w * (eta - off + (y-mu)/mu').
    void build_z_target() {
        constexpr double tiny = std::numeric_limits<double>::min();
        const ArrayXd lin = eta_ - off_;
        ArrayXd z = (dmu_.abs() > tiny).select(lin + (y_ - mu_) / dmu_, lin);
        z = z.isFinite().select(z, lin);
        tw_ = w_ * z;
    }

    [[nodiscard]] Vector gather(const Vector& full) const {
        Vector a(static_cast<Index>(act_.size()));
        for (std::size_t k = 0; k < act_.size(); ++k) a[static_cast<Index>(k)] = full[act_[k]];
        return a;
    }

    [[noreturn]] void throw_nonfinite(int it) const {
        if (it == 0 && !eta_consistent_)
            throw std::runtime_error("Numerical error: No valid starting values found.");
        throw std::runtime_error("Numerical error: Deviance became non-finite at iteration " +
                                 std::to_string(it) + " and step-halving failed to recover.");
    }

    // ---- search directions --------------------------------------------------------------------
    /// Fisher scoring direction in active coordinates (full solution in the very first iteration).
    void fisher_direction(bool zform, Vector& sol, int it) {
        if (zform) {
            build_z_target();
            D().wls(wls_, w_, tw_, qr_chunk_, ne_chunk_, sol);
            sol -= gather(beta_prev_);
        } else {
            D().wls(wls_, w_, r_, qr_chunk_, ne_chunk_, sol);
        }
        solved_ = true;
        fisher_valid_ = true;
        logdet_ = wls_.log_det();
        rank_   = static_cast<int>(wls_.rank());
        if (rank_ == 0)
            throw std::runtime_error("Numerical error: design matrix has rank 0 under the current weights "
                                     "(all working weights are zero or non-finite).");
        if (firth_) firth_adjust(sol);
        if (!sol.allFinite()) {
            if (it == 0 && !eta_consistent_)
                throw std::runtime_error("Numerical error: No valid starting values found.");
            throw std::runtime_error("Numerical error: non-finite coefficients at iteration " +
                                     std::to_string(it) + ".");
        }
    }

    /// d2mu/deta2 and V'(mu) (central difference) at the current iterate.
    void compute_second_order() {
        d2_.resize(nw_);
        if (fam_code_ >= 0) {
            glm::d2mu_deta2(fam_code_, fam_params_, eta_, mu_, dmu_, d2_);
        } else {
            const double base = std::cbrt(std::numeric_limits<double>::epsilon());
            const ArrayXd hh = base * eta_.abs().max(1.0);
            ArrayXd ep = eta_ + hh, em = eta_ - hh, dp(nw_), dm(nw_);
            mu_eta_fun_(ep, dp);
            mu_eta_fun_(em, dm);
            d2_ = (dp - dm) / (2.0 * hh);
        }
        const ArrayXd hm = 1e-5 * mu_.abs().max(1e-8);
        ArrayXd mp = mu_ + hm, mm = mu_ - hm, vp(nw_), vm(nw_);
        eval_variance(mp, vp);
        eval_variance(mm, vm);
        vprime_ = (vp - vm) / (2.0 * hm);
    }

    /**
     * @brief Newton direction with the observed information X' W_obs X,
     *        W_obs = W - pw (y-mu) (mu''/V - mu'^2 V'/V^2), floored at 1e-3 W so the system stays positive
     *        definite. Returns false if it cannot be formed or is rank deficient.
     */
    [[nodiscard]] bool observed_direction(Vector& sol) {
        compute_second_order();
        const ArrayXd Wf = w_.square();
        ArrayXd Wo = Wf - pw_ * (y_ - mu_) * (d2_ / var_ - dmu_.square() * vprime_ / var_.square());
        Wo = Wo.isFinite().select(Wo, Wf);
        Wo = Wo.max(1e-3 * Wf);
        const ArrayXd wo = Wo.sqrt();
        const ArrayXd tw = (wo > 0.0).select(w_ * r_ / wo, 0.0);
        D().wls(wls_, wo, tw, qr_chunk_, ne_chunk_, sol);
        fisher_valid_ = false;
        return sol.allFinite() && wls_.rank() == static_cast<Index>(act_.size());
    }

    // ---- line search ----------------------------------------------------------------------------
    /**
     * @brief Backtracking on alpha along `sol` (active coordinates). The state must be valid at
     *        (eta_prev_, beta_prev_) on entry.
     * @param strict if true a step that still increases the deviance after `max_halv` halvings is a failure
     *        (`LS::Increased`); otherwise it is accepted like glm.fit does.
     */
    LS line_search(const Vector& sol, bool zform, bool strict, int max_halv, double& decrement, int /*it*/) {
        delta_.setZero();
        for (std::size_t k = 0; k < act_.size(); ++k) delta_[act_[k]] = sol[static_cast<Index>(k)];
        if (firth_) limit_step();

        double alpha = 1.0;
        decrement = 0.0;
        if (!zform) {
            D().matvec(gather(delta_), xd_);
            decrement = (w_ * xd_).square().sum();
            if (opts_.max_eta_step > 0.0) {
                const double m = xd_.abs().maxCoeff();
                if (std::isfinite(m) && m > opts_.max_eta_step) alpha = opts_.max_eta_step / m;
            }
        }
        int halvings = 0;
        bool ok = false, inc = false;
        double devc = 0.0;
        for (;;) {
            beta_ = beta_prev_ + alpha * delta_;
            if (zform) { D().matvec(gather(beta_), eta_); eta_ += off_; }
            else       { eta_ = eta_prev_ + alpha * xd_; }
            ok = evaluate(true, devc);
            if (ok) {
                std_dev_ = devc;
                dev_     = firth_ ? std_dev_ - phi_ * logdet_ : std_dev_;
            }
            inc = ok && !firth_ && !zform && (dev_ - devold_) > tol_ * (0.1 + std::abs(dev_));
            if ((ok && !inc) || halvings >= max_halv) break;
            alpha *= 0.5;
            ++halvings;
        }
        if (!ok || !std::isfinite(dev_)) return LS::Invalid;
        if (inc) return strict ? LS::Increased : LS::Ok;
        return LS::Ok;
    }

    /// Levenberg-Marquardt: (A_s + lambda I) d = b_s on the Jacobi-scaled Gram matrix, lambda grows x10 per retry.
    LS lm_search(double& decrement, int it) {
        Matrix A; Vector b;
        D().gram(w_, r_, ne_chunk_, A, b);
        double lambda = opts_.lm_lambda0;
        LS status = LS::Invalid;
        for (int k = 0; k < opts_.lm_tries; ++k, lambda *= 10.0) {
            Vector sol;
            if (!wls_.damped_gram(A, b, lambda, sol)) continue;
            status = line_search(sol, /*zform=*/false, /*strict=*/true, /*max_halv=*/0, decrement, it);
            if (status == LS::Ok) return status;
            restore_state();
        }
        return status;
    }

    // ---- Firth --------------------------------------------------------------------------------
    /// Adds the bias-reduction adjustment to the WLS solution `sol` (active coordinates).
    void firth_adjust(Vector& sol) {
        constexpr double tiny = std::numeric_limits<double>::min();
        const Matrix Ainv = wls_.inverse_information();

        h_.resize(nw_);
        D().leverages(w_, Ainv, ne_chunk_, h_);
        compute_second_order();

        double phi = 1.0;
        if (!unit_dispersion() && nw_ > rank_) {
            phi = std_dev_ / static_cast<double>(nw_ - rank_);
            if (!(phi > std::numeric_limits<double>::epsilon()) || !std::isfinite(phi)) phi = 1.0;
        }
        phi_ = phi;

        const ArrayXd W   = w_.square();
        const ArrayXd num = 2.0 * d2_ / dmu_ - vprime_ * dmu_ / var_;
        xi_ = ((W > tiny) && (dmu_.abs() > tiny) && (var_ > 0.0))
                  .select(phi * h_ * num / (2.0 * W), 0.0);
        xi_ = xi_.isFinite().select(xi_, 0.0);

        // delta <- delta_WLS + (X'WX)^-1 X' W xi
        const ArrayXd wxi = W * xi_;
        Vector g;
        D().rmatvec(wxi, g);
        sol.noalias() += Ainv * g;
    }

    [[nodiscard]] bool unit_dispersion() const {
        return fam_code_ < 0 || glm::has_unit_dispersion(fam_code_);
    }

    void limit_step() {
        const double m = delta_.cwiseAbs().maxCoeff();
        if (std::isfinite(m) && m > opts_.max_step) delta_ *= (opts_.max_step / m);
    }

    // ---- covariance and outputs ---------------------------------------------------------------
    void finalize() {
        if (!solved_) return;
        constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
        if (!fisher_valid_ && eta_consistent_) {
            // the last factorisation used the observed information: redo it with the expected one (R's vcov)
            Vector sol;
            D().wls(wls_, w_, r_, qr_chunk_, ne_chunk_, sol);
            fisher_valid_ = true;
            logdet_ = wls_.log_det();
            rank_   = static_cast<int>(wls_.rank());
        }
        const Matrix Ainv = wls_.inverse_information();   // active coordinates, original units

        vcov_ = Matrix::Constant(p_, p_, qnan);
        for (std::size_t a = 0; a < act_.size(); ++a)
            for (std::size_t b = 0; b < act_.size(); ++b)
                vcov_(act_[a], act_[b]) = Ainv(static_cast<Index>(a), static_cast<Index>(b));
        se_ = Vector::Constant(p_, qnan);
        for (std::size_t a = 0; a < act_.size(); ++a)
            se_[act_[a]] = std::sqrt(std::max(0.0, Ainv(static_cast<Index>(a), static_cast<Index>(a))));

        aliased_ = aliased0_;
        for (const int j : wls_.aliased()) aliased_.push_back(act_[static_cast<std::size_t>(j)]);
        std::sort(aliased_.begin(), aliased_.end());
        aliased_.erase(std::unique(aliased_.begin(), aliased_.end()), aliased_.end());
        for (const int j : aliased_) {
            vcov_.row(j).setConstant(qnan);
            vcov_.col(j).setConstant(qnan);
            se_[j] = qnan;
        }

        n_boundary_ = 0;
        if (fam_code_ >= 0 && glm::is_binomial_family(fam_code_)) {
            const double eps = 10.0 * DBL_EPSILON;
            for (Index i = 0; i < nw_; ++i)
                if (pw_[i] > 0.0 && (mu_[i] < eps || mu_[i] > 1.0 - eps)) ++n_boundary_;
        }

        if (!rows_.empty()) {   // expand eta / mu / W to every input row (zero-weight rows get predictions)
            ArrayXd eta(nfull_);
            base_->matvec(beta_, eta);
            eta += off_full_;
            ArrayXd mu(nfull_);
            eval_linkinv(eta, mu);
            eta_out_ = eta;
            mu_out_  = mu;
            w_out_   = ArrayXd::Zero(nfull_);
            for (Index k = 0; k < nw_; ++k) w_out_[rows_[static_cast<std::size_t>(k)]] = w_[k];
        }
    }
};

}  // namespace glm

#endif  // _GLMFIT_IRLS_HPP
