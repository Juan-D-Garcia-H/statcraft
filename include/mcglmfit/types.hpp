// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file types.hpp
/// @brief Public types of the MCGLM engine: enumerations, model specification, options and results.
///
/// Model (Bonat & Jørgensen, 2016, JRSS-C 65(5)): for K responses observed on n units,
/// the stacked response vector y (length N = K n) has
///
///     E(y) = mu,                 g_k(mu_k) = X_k beta_k + offset_k
///     Var(y) = C = L (Sigma_b (x) I_n) L^T,     Sigma_k = F_k^T F_k = V_k^{1/2} Omega_k V_k^{1/2} [+ diag(mu_k) terms]
///     Omega_k = h_k( tau_k0 Z_k0 + tau_k1 Z_k1 + ... )
///
/// where g_k is the link, V_k the variance function (power / binomial families), h_k the covariance link
/// (identity, inverse, matrix exponential), Z_kj known symmetric matrices, Sigma_b the K x K between-response
/// correlation matrix (parameters rho) and L = blockdiag(F_k^T).
///
/// Parameters are estimated with quasi-score (regression) and Pearson (covariance) estimating functions.
#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcglm {

/// Thrown when a numerical step fails (non-positive-definite matrix, singular system, non-finite values).
struct NumericalError : std::runtime_error { using std::runtime_error::runtime_error; };

/// Thrown when an argument is invalid (inconsistent dimensions, NaN/inf, unsupported combination).
struct InvalidInput : std::runtime_error { using std::runtime_error::runtime_error; };

/// Thrown when a computation would need more memory than the machine has.
struct ResourceError : std::runtime_error { using std::runtime_error::runtime_error; };

/// Explicit numeric conversions (no -Wold-style-cast / -Wsign-conversion noise; the value is not range-checked).
template <class T> constexpr std::size_t uz(T v) noexcept { return static_cast<std::size_t>(v); }
template <class T> constexpr int toi(T v) noexcept { return static_cast<int>(v); }
template <class T> constexpr double tod(T v) noexcept { return static_cast<double>(v); }
template <class T> constexpr Eigen::Index tix(T v) noexcept { return static_cast<Eigen::Index>(v); }

using Eigen::MatrixXd;
using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct Partition;   // block decomposition of the observations (see partition.hpp)

/// Link function g with mu = g^{-1}(eta).
enum class Link { Logit, Probit, Cauchit, Cloglog, Loglog, Identity, Log, Sqrt, InvMu2, Inverse };

/// Variance function. `Power` is the Tweedie-type variance mu^p. `BinomialP` is (mu(1-mu))^p / ntrial,
/// `BinomialPQ` is mu^p (1-mu)^q / ntrial. `PoissonTweedie` and `GeomTweedie` add mu and mu^2 to the diagonal.
enum class Variance { Constant, Power, BinomialP, BinomialPQ, PoissonTweedie, GeomTweedie };

/// Covariance link h: Omega = tau-combination (Identity), its inverse is the combination (Inverse),
/// or Omega = expm(combination) (Expm).
enum class CovLink { Identity, Inverse, Expm };

/// Update rule for the covariance parameters: Newton-like `Chaser`, `Rc` (regularised chaser using the
/// Pearson variability) or plain `Gradient` ascent.
enum class Method { Chaser, Rc, Gradient };

/// Known symmetric matrix Z of the matrix linear predictor, stored as its diagonal (O(n)), dense, or sparse.
///
/// A sparse Z is never expanded to n x n unless the observations form a single connected block; otherwise the
/// engine works on the connected components of the sparsity pattern (see partition.hpp).
struct ZMat {
    bool is_diag = true;   ///< true: only `diag` is used
    bool is_sparse = false;///< true (with !is_diag): only `sparse` is used
    VectorXd diag;         ///< diagonal entries (length n)
    MatrixXd dense;        ///< full n x n matrix (when !is_diag && !is_sparse)
    SpMat sparse;          ///< compressed sparse matrix (when is_sparse)

    /// Identity matrix of size n (diagonal representation).
    static ZMat identity(Eigen::Index n) { ZMat z; z.diag = VectorXd::Ones(n); return z; }
    /// Diagonal matrix with entries `d`.
    static ZMat diagonal(const VectorXd& d) { ZMat z; z.diag = d; return z; }
    /// Dense matrix; if `detect_diagonal` and all off-diagonal entries are exactly zero, the cheaper
    /// diagonal representation is used.
    static ZMat from_dense(const MatrixXd& M, bool detect_diagonal = true) {
        ZMat z;
        if (detect_diagonal && M.rows() == M.cols()) {
            MatrixXd off = M; off.diagonal().setZero();
            if (off.cwiseAbs().maxCoeff() == 0.0) { z.diag = M.diagonal(); return z; }
        }
        z.is_diag = false; z.dense = M; return z;
    }
    /// Sparse matrix; if `detect_diagonal` and every stored entry is on the diagonal, the diagonal form is used.
    static ZMat from_sparse(const SpMat& S, bool detect_diagonal = true) {
        ZMat z;
        if (detect_diagonal) {
            bool diag_only = true;
            for (int c = 0; c < S.outerSize() && diag_only; ++c)
                for (SpMat::InnerIterator it(S, c); it; ++it)
                    if (it.row() != c && it.value() != 0.0) { diag_only = false; break; }
            if (diag_only) {
                z.diag = VectorXd::Zero(S.rows());
                for (int c = 0; c < S.outerSize(); ++c)
                    for (SpMat::InnerIterator it(S, c); it; ++it) if (it.row() == c) z.diag[c] = it.value();
                return z;
            }
        }
        z.is_diag = false; z.is_sparse = true; z.sparse = S; z.sparse.makeCompressed(); return z;
    }
    /// Number of rows (= columns).
    Eigen::Index n() const { return is_diag ? diag.size() : (is_sparse ? tix(sparse.rows()) : dense.rows()); }
    /// Dense copy of a sparse matrix (other forms are returned unchanged).
    ZMat densified() const {
        if (!is_sparse) return *this;
        ZMat z; z.is_diag = false; z.dense = MatrixXd(sparse); return z;
    }
};

/// Specification of one response (everything that does not change between fits).
struct ResponseSpec {
    MatrixXd X;                 ///< n x p design matrix
    VectorXd offset;            ///< length n; empty = no offset
    VectorXd ntrial;            ///< length n number of trials (binomial variances); empty = 1
    Link link = Link::Identity;
    Variance variance = Variance::Constant;
    CovLink covariance = CovLink::Identity;
    bool power_fixed = true;    ///< true: variance exponents are held at their starting values
    std::vector<ZMat> Z;        ///< one or more Z matrices; tau has one entry per Z
};

/// Model parameters, one entry per response (rho: K(K-1)/2 entries, empty when K = 1).
struct Params {
    std::vector<VectorXd> beta;   ///< regression coefficients
    std::vector<VectorXd> power;  ///< variance exponents (1 or 2 values; one dummy value for `Constant`)
    std::vector<VectorXd> tau;    ///< dispersion parameters, one per Z matrix
    VectorXd rho;                 ///< between-response correlations, row-major upper triangle
};

/// How the derivatives of C with respect to the within-response parameters are formed in the cross blocks
/// when K > 1 and some Z is dense: `RCompatible` reproduces the R package `mcglm`; `Exact` uses the
/// exact derivative of C.
enum class CrossBlock { RCompatible, Exact };

/// How the observations are organised for the computation.
/// `Auto` uses the connected components of the sparsity pattern of the Z matrices when that is cheaper
/// (exact: C is block diagonal across components); `Dense` forces one block; `Blocks` forces the decomposition.
/// `Sparse` forces the approximate sparse method (single block, K = 1, identity covariance link); see sparse_backend.hpp.
enum class Structure { Auto, Dense, Blocks, Sparse };

/// Fitting options.
struct Options {
    int max_iter = 20;            ///< maximum number of iterations (including the starting point)
    double tol = 1e-4;            ///< convergence tolerance on the max absolute parameter change
    Method method = Method::Chaser;
    double tuning = 1.0;          ///< step scaling of the covariance update
    bool correct = true;          ///< bias-correction term in the Pearson estimating function
    bool compute_vcov = true;     ///< compute the joint sandwich covariance of (beta, rho, power, tau)
    bool verbose = false;         ///< print the covariance parameters at each iteration to std::cerr
    CrossBlock cross_block = CrossBlock::RCompatible;
    /// If true, the off-diagonal terms of the Pearson variability are set to zero, matching the R package
    /// (its `forceSymmetric` call discards them). If false the full matrix is used.
    bool r_diag_variability = true;
    /// Optional per-iteration hook `(iteration, beta, cov) -> stop`. Returning true stops the fit
    /// (`FitResult::stopped`). Exceptions thrown by the hook propagate out of `fit`.
    std::function<bool(int, const VectorXd&, const VectorXd&)> on_iteration;
    Structure structure = Structure::Auto;
    /// Components with few units are merged into batches of about this many units (zero padding is exact);
    /// larger values mean fewer, larger dense blocks.
    int block_target = 32;
    /// Threads used to process blocks (OpenMP builds only); 0 uses the OpenMP default. The result is
    /// deterministic for a given thread count.
    int block_threads = 0;
    /// Precomputed decomposition (must match the specifications); built on demand when null.
    std::shared_ptr<const Partition> partition;
    /// With `Structure::Auto`, a single block with n above this and a sparse enough Z uses the sparse method
    /// (below it the exact dense computation is cheap).
    int sparse_dense_max_n = 4000;
    /// Sparse method: random probes of the stochastic trace estimates during the iterations and in the final
    /// sensitivity / covariance (>= 2). They are fixed (deterministic for `probe_seed`).
    int n_probes = 32;
    int n_probes_final = 128;
    unsigned long long probe_seed = 12345;
};

/// Result of a fit.
struct FitResult {
    VectorXd beta;                ///< stacked regression coefficients
    VectorXd cov;                 ///< covariance parameters: rho, then (power_k, tau_k) for each response
    Params params;                ///< the same parameters in per-response form
    MatrixXd iter_beta, iter_cov; ///< parameter history, one row per iteration performed
    MatrixXd score_beta, score_cov; ///< estimating-function values at each iteration (NaN where not evaluated)
    int iterations = 0;           ///< rows of the history
    bool converged = false;
    bool stopped = false;         ///< stopped by `Options::on_iteration`
    VectorXd fitted, residuals;   ///< mu and y - mu (stacked)
    MatrixXd vcov;                ///< joint covariance of (beta, cov); empty if !compute_vcov
    MatrixXd inv_S_beta;          ///< inverse sensitivity of beta
    MatrixXd S_cov;               ///< Pearson sensitivity of the covariance parameters
    MatrixXd V_beta, V_cov;       ///< variabilities (only if compute_vcov)
    std::vector<int> n_beta, n_tau, n_power; ///< parameter counts per response
    int n_rho = 0;                ///< number of between-response correlations
    std::string structure = "dense";  ///< "diagonal", "dense", "blocks" or "sparse": how the covariance was handled
    int n_blocks = 0;             ///< number of blocks when structure == "blocks"
    int max_block = 0;            ///< units in the largest block
    double mc_rel_error = 0.0;    ///< sparse method: largest relative Monte Carlo standard error of the stochastic traces
    int n_probes = 0;             ///< sparse method: probes used for the final sensitivity / covariance
};

} // namespace mcglm
