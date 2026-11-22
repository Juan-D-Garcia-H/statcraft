// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

// nanobind bindings of the MCGLM engine (header-only C++20, Eigen, OpenBLAS/LAPACKE).
//
// Design
//  * `Problem` owns the model specification (design matrices, Z matrices, links, ...) once, so the same
//    problem can be refitted many times (new responses, new weights, bootstrap) without copying dense Z again.
//  * `Problem.fit` releases the GIL, so fits can run concurrently from Python threads. Ctrl-C is honoured between
//    iterations and an optional Python callback is invoked after each iteration.
//  * Results are exposed as read-only NumPy views over the memory of the `FitResult` (no copies); the view keeps
//    the result alive.
//  * Inputs: any float64 ndarray (strided, C or Fortran order); other dtypes are converted. Everything is
//    validated and reported through `InvalidInput` (a ValueError) before the engine is entered.
//  * Numerical failures raise `NumericalError` (an ArithmeticError); allocation failures raise MemoryError.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../include/mcglmfit/mcglm.hpp"

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mutex>
#include <shared_mutex>
#include <sstream>

#ifdef MCGLM_HAVE_OPENBLAS
extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
char* openblas_get_config(void);
}
#endif

namespace nb = nanobind;
using namespace nb::literals;
using namespace mcglm;

using Arr = nb::ndarray<const double, nb::device::cpu>;

namespace {

// ------------------------------------------------------------------------------ validated conversions

/// Copies a 1-D array into a vector, rejecting NaN/inf (NaN is allowed when `allow_nan`, e.g. missing responses).
VectorXd to_vec(const Arr& a, const char* name, bool allow_nan = false) {
    if (a.ndim() != 1) throw InvalidInput(std::string(name) + " must be 1-D, got " + std::to_string(a.ndim()) + "-D");
    const Eigen::Index n = (Eigen::Index)a.shape(0);
    VectorXd v(n);
    const double* p = a.data(); const int64_t st = a.stride(0);
    for (Eigen::Index i = 0; i < n; ++i) v[i] = p[i * st];
    if (allow_nan) {
        for (Eigen::Index i = 0; i < n; ++i)
            if (std::isinf(v[i])) throw InvalidInput(std::string(name) + " contains inf");
    } else if (!v.allFinite()) {
        throw InvalidInput(std::string(name) + " contains NaN or inf");
    }
    return v;
}

/// Copies a 2-D array (any strides) into a dense matrix with a single copy, rejecting NaN/inf.
MatrixXd to_mat(const Arr& a, const char* name) {
    if (a.ndim() != 2) throw InvalidInput(std::string(name) + " must be 2-D, got " + std::to_string(a.ndim()) + "-D");
    const Eigen::Index r = (Eigen::Index)a.shape(0), c = (Eigen::Index)a.shape(1);
    using Str = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<const MatrixXd, 0, Str> m(a.data(), r, c, Str((Eigen::Index)a.stride(1), (Eigen::Index)a.stride(0)));
    MatrixXd M = m;
    if (!M.allFinite()) throw InvalidInput(std::string(name) + " contains NaN or inf");
    return M;
}

// ------------------------------------------------------------------------------------------ SparseZ

/// Validated sparse symmetric matrix built from the CSC arrays of a SciPy sparse matrix.
struct SparseZ { SpMat S; };

using I64 = nb::ndarray<const int64_t, nb::ndim<1>, nb::device::cpu>;
using F64 = nb::ndarray<const double, nb::ndim<1>, nb::device::cpu>;

SparseZ make_sparse_z(long n, const I64& indptr, const I64& indices, const F64& data) {
    if (n < 1) throw InvalidInput("the sparse Z must have at least one row");
    if (n > 2000000000L) throw InvalidInput("the sparse Z is too large (more than 2^31 rows)");
    if ((long)indptr.shape(0) != n + 1) throw InvalidInput("indptr must have length n + 1 = " + std::to_string(n + 1));
    const int64_t nnz = indptr.data()[n];
    if (indptr.data()[0] != 0) throw InvalidInput("indptr[0] must be 0");
    for (long c = 0; c < n; ++c)
        if (indptr.data()[c + 1] < indptr.data()[c] || indptr.data()[c + 1] > nnz)
            throw InvalidInput("indptr must be non-decreasing");
    if ((int64_t)indices.shape(0) != nnz || (int64_t)data.shape(0) != nnz)
        throw InvalidInput("indices and data must have length indptr[-1] = " + std::to_string(nnz));
    std::vector<Eigen::Triplet<double>> trip; trip.reserve((size_t)nnz);
    for (long c = 0; c < n; ++c) {
        const int64_t lo = indptr.data()[c], hi = indptr.data()[c + 1];
        for (int64_t p = lo; p < hi; ++p) {
            const int64_t r = indices.data()[p]; const double v = data.data()[p];
            if (r < 0 || r >= n) throw InvalidInput("a row index of the sparse Z is out of range [0, n)");
            if (!std::isfinite(v)) throw InvalidInput("the sparse Z contains NaN or inf");
            if (v != 0.0) trip.emplace_back((int)r, (int)c, v);
        }
    }
    SparseZ z; z.S.resize((int)n, (int)n); z.S.setFromTriplets(trip.begin(), trip.end());
    double scale = 1.0;
    for (int c = 0; c < z.S.outerSize(); ++c) for (SpMat::InnerIterator it(z.S, c); it; ++it) scale = std::fmax(scale, std::fabs(it.value()));
    SpMat D = z.S - SpMat(z.S.transpose());
    double mx = 0.0;
    for (int c = 0; c < D.outerSize(); ++c) for (SpMat::InnerIterator it(D, c); it; ++it) mx = std::fmax(mx, std::fabs(it.value()));
    if (mx > 1e-8 * scale) throw InvalidInput("the sparse Z must be symmetric");
    z.S.makeCompressed();
    return z;
}

// -------------------------------------------------------------------------------------------- Problem

/// Model specification shared by all fits. Reads (`fit`, `covariance`) take a shared lock and writes
/// (`add_response`) an exclusive one, so concurrent use from several threads is safe.
struct Problem {
    std::vector<ResponseSpec> specs;
    Eigen::Index n = -1;
    mutable std::shared_mutex mtx;

    void add_response(const Arr& X, Link link, Variance var, CovLink cov, const std::vector<nb::object>& Z,
                      bool power_fixed, const std::optional<Arr>& offset, const std::optional<Arr>& ntrial,
                      bool detect_diagonal) {
        ResponseSpec rs;
        rs.X = to_mat(X, "X");
        const Eigen::Index m = rs.X.rows();
        if (rs.X.cols() < 1) throw InvalidInput("X must have at least one column");
        if (m < 2) throw InvalidInput("at least 2 observations are required, got " + std::to_string(m));
        {   std::shared_lock<std::shared_mutex> rd(mtx);
            if (n >= 0 && m != n)
                throw InvalidInput("all responses must have the same number of observations (expected " +
                                   std::to_string(n) + ", got " + std::to_string(m) + ")");
        }
        rs.link = link; rs.variance = var; rs.covariance = cov; rs.power_fixed = power_fixed;
        if (offset) {
            rs.offset = to_vec(*offset, "offset");
            if (rs.offset.size() != m) throw InvalidInput("offset must have length n = " + std::to_string(m));
        }
        if (ntrial) {
            rs.ntrial = to_vec(*ntrial, "ntrial");
            if (rs.ntrial.size() != m) throw InvalidInput("ntrial must have length n = " + std::to_string(m));
            if ((rs.ntrial.array() <= 0).any()) throw InvalidInput("ntrial must be strictly positive");
        }
        if (Z.empty()) throw InvalidInput("every response needs at least one Z matrix");
        for (size_t j = 0; j < Z.size(); ++j) {
            const std::string nm = "Z[" + std::to_string(j) + "]";
            if (nb::isinstance<SparseZ>(Z[j])) {
                const SparseZ& sz = nb::cast<const SparseZ&>(Z[j]);
                if (sz.S.rows() != m) throw InvalidInput(nm + " must be n x n with n = " + std::to_string(m));
                rs.Z.push_back(ZMat::from_sparse(sz.S, detect_diagonal));
                continue;
            }
            Arr zj = nb::cast<Arr>(Z[j]);
            if (zj.ndim() == 1) {
                VectorXd d = to_vec(zj, nm.c_str());
                if (d.size() != m) throw InvalidInput(nm + " must have length n = " + std::to_string(m));
                rs.Z.push_back(ZMat::diagonal(d));
            } else {
                MatrixXd M = to_mat(zj, nm.c_str());
                if (M.rows() != m || M.cols() != m) throw InvalidInput(nm + " must be n x n with n = " + std::to_string(m));
                const double scale = std::fmax(1.0, M.cwiseAbs().maxCoeff());
                if ((M - M.transpose()).cwiseAbs().maxCoeff() > 1e-8 * scale)
                    throw InvalidInput(nm + " must be symmetric");
                rs.Z.push_back(ZMat::from_dense(M, detect_diagonal));
            }
        }
        // Identical or proportional Z matrices make the corresponding tau parameters unidentifiable.
        for (size_t j = 0; j < rs.Z.size(); ++j)
            for (size_t l = j + 1; l < rs.Z.size(); ++l) {
                const ZMat &a = rs.Z[j], &b = rs.Z[l];
                if (a.is_diag != b.is_diag || a.is_sparse != b.is_sparse) continue;
                double na, nb2, d;
                if (a.is_sparse) {
                    na = a.sparse.squaredNorm(); nb2 = b.sparse.squaredNorm(); d = a.sparse.cwiseProduct(b.sparse).sum();
                } else {
                    const Eigen::Index len = a.is_diag ? m : m * m;
                    Eigen::Map<const VectorXd> va(a.is_diag ? a.diag.data() : a.dense.data(), len);
                    Eigen::Map<const VectorXd> vb(b.is_diag ? b.diag.data() : b.dense.data(), len);
                    na = va.squaredNorm(); nb2 = vb.squaredNorm(); d = va.dot(vb);
                }
                if (na > 0 && nb2 > 0 && d * d >= (1.0 - 1e-12) * na * nb2)
                    throw InvalidInput("Z[" + std::to_string(j) + "] and Z[" + std::to_string(l) +
                                       "] are identical or proportional, so their tau parameters are not identifiable. "
                                       "To give each response its own Z use one list per response: z=[[Z1], [Z2], ...]");
            }
        std::unique_lock<std::shared_mutex> wr(mtx);
        if (n >= 0 && m != n)
            throw InvalidInput("all responses must have the same number of observations (expected " +
                               std::to_string(n) + ", got " + std::to_string(m) + ")");
        n = m; specs.push_back(std::move(rs));
    }
    void require_ready() const {
        if (specs.empty()) throw InvalidInput("the problem has no responses; call add_response first");
    }

    /// Block decomposition for the given merge target (null when every Z is diagonal). Cached until a response is
    /// added or the target changes. The caller must hold the shared lock on `mtx`.
    std::shared_ptr<const Partition> partition(int target) const {
        if (!has_off_diagonal_z(specs)) return nullptr;
        std::lock_guard<std::mutex> g(part_mtx);
        if (!part_cache || part_target != target || part_stamp != specs.size()) {
            part_cache = build_partition(specs, target); part_target = target; part_stamp = specs.size();
        }
        return part_cache;
    }

private:
    mutable std::mutex part_mtx;
    mutable std::shared_ptr<const Partition> part_cache;
    mutable int part_target = -1;
    mutable size_t part_stamp = 0;
};

/// Sets the OpenBLAS thread count to 1 while blocks are processed by several OpenMP threads (avoids
/// oversubscription) and restores it afterwards.
struct BlasThreadGuard {
    int saved = 0;
    explicit BlasThreadGuard(bool active) {
#ifdef MCGLM_HAVE_OPENBLAS
        if (active) { saved = openblas_get_num_threads(); if (saved > 1) openblas_set_num_threads(1); else saved = 0; }
#else
        (void)active;
#endif
    }
    ~BlasThreadGuard() {
#ifdef MCGLM_HAVE_OPENBLAS
        if (saved > 1) openblas_set_num_threads(saved);
#endif
    }
};

int effective_block_threads(int requested) {
#ifdef _OPENMP
    return requested > 0 ? requested : omp_get_max_threads();
#else
    (void)requested; return 1;
#endif
}

VectorXd check_y(const Problem& p, const Arr& y) {
    VectorXd v = to_vec(y, "y", /*allow_nan=*/true);
    const Eigen::Index expected = (Eigen::Index)p.specs.size() * p.n;
    if (v.size() != expected)
        throw InvalidInput("y must have length K * n = " + std::to_string(expected) + " (responses stacked), got " +
                           std::to_string(v.size()));
    return v;
}
VectorXd check_weights(const Problem& p, const std::optional<Arr>& w) {
    if (!w) return VectorXd();
    VectorXd v = to_vec(*w, "weights");
    const Eigen::Index expected = (Eigen::Index)p.specs.size() * p.n;
    if (v.size() != expected)
        throw InvalidInput("weights must have length K * n = " + std::to_string(expected) + ", got " +
                           std::to_string(v.size()));
    if ((v.array() < 0).any()) throw InvalidInput("weights must be non-negative");
    return v;
}
void check_params(const Problem& p, const Params& q) {
    const size_t K = p.specs.size();
    if (q.beta.size() != K || q.tau.size() != K || q.power.size() != K)
        throw InvalidInput("Params must have one entry per response (" + std::to_string(K) + ")");
    for (size_t k = 0; k < K; ++k) {
        const std::string r = "[" + std::to_string(k) + "]";
        if (q.beta[k].size() != p.specs[k].X.cols())
            throw InvalidInput("Params.beta" + r + " has " + std::to_string(q.beta[k].size()) + " entries but X has " +
                               std::to_string(p.specs[k].X.cols()) + " columns");
        if (q.tau[k].size() != (Eigen::Index)p.specs[k].Z.size())
            throw InvalidInput("Params.tau" + r + " has " + std::to_string(q.tau[k].size()) + " entries but the response has " +
                               std::to_string(p.specs[k].Z.size()) + " Z matrices");
        const int need = std::max(1, variance_n_power(p.specs[k].variance));
        if (q.power[k].size() < need)
            throw InvalidInput("Params.power" + r + " needs at least " + std::to_string(need) + " value(s)");
        if (!q.beta[k].allFinite() || !q.tau[k].allFinite() || !q.power[k].allFinite())
            throw InvalidInput("Params contains NaN or inf");
    }
    if (K > 1 && q.rho.size() != (Eigen::Index)(K * (K - 1) / 2))
        throw InvalidInput("Params.rho must have K(K-1)/2 = " + std::to_string(K * (K - 1) / 2) + " entries, got " +
                           std::to_string(q.rho.size()));
}

FitResult do_fit(const Problem& p, const Arr& y_in, const std::optional<Params>& init, const std::optional<Arr>& weights,
                 const Options& opt, const std::optional<nb::callable>& cb) {
    p.require_ready();
    VectorXd y = check_y(p, y_in);
    VectorXd w = check_weights(p, weights);
    if (init) check_params(p, *init);
    Options o = opt;
    nb::callable pycb = cb ? *cb : nb::callable();
    const bool has_cb = (bool)cb;
    o.on_iteration = [pycb, has_cb](int it, const VectorXd& b, const VectorXd& c) -> bool {
        nb::gil_scoped_acquire gil;
        if (PyErr_CheckSignals() != 0) throw nb::python_error();            // Ctrl-C / pending signal handlers
        if (has_cb) {
            nb::object r = pycb(it, nb::cast(b, nb::rv_policy::copy), nb::cast(c, nb::rv_policy::copy));
            return !r.is_none() && nb::cast<bool>(r);
        }
        return false;
    };
    std::shared_lock<std::shared_mutex> lock(p.mtx);
    nb::gil_scoped_release release;                                          // the engine never touches Python objects
    Params p0 = init ? *init : initial_values(p.specs, y, w);
    if (o.structure != Structure::Dense && !o.partition) o.partition = p.partition(o.block_target);
    const bool parallel_blocks = o.partition && o.partition->useful && effective_block_threads(o.block_threads) > 1;
    BlasThreadGuard blas(parallel_blocks);
    return mcglm::fit(p.specs, y, p0, w, o);
}

std::string repr_options(const Options& o) {
    std::ostringstream s;
    s << "Options(max_iter=" << o.max_iter << ", tol=" << o.tol << ", tuning=" << o.tuning
      << ", correct=" << (o.correct ? "True" : "False") << ", compute_vcov=" << (o.compute_vcov ? "True" : "False") << ")";
    return s.str();
}

} // namespace

NB_MODULE(_mcglmfit, m) {
    m.doc() = "Low-level bindings of the MCGLM engine (C++20, Eigen, OpenBLAS/LAPACKE). "
              "Most users should use the high-level classes in `mcglm_cpp` (MCGLM, MCGLMResults).";
    nb::exception<NumericalError>(m, "NumericalError", PyExc_ArithmeticError);
    nb::exception<InvalidInput>(m, "InvalidInput", PyExc_ValueError);
    nb::exception<ResourceError>(m, "ResourceError", PyExc_MemoryError);

    nb::enum_<Link>(m, "Link", "Link function g, with mu = g^{-1}(eta).")
        .value("Logit", Link::Logit).value("Probit", Link::Probit).value("Cauchit", Link::Cauchit)
        .value("Cloglog", Link::Cloglog).value("Loglog", Link::Loglog).value("Identity", Link::Identity)
        .value("Log", Link::Log).value("Sqrt", Link::Sqrt).value("InvMu2", Link::InvMu2).value("Inverse", Link::Inverse);
    nb::enum_<Variance>(m, "Variance",
                        "Variance function. Power: mu^p. BinomialP: (mu(1-mu))^p / ntrial. BinomialPQ: mu^p (1-mu)^q / ntrial. "
                        "PoissonTweedie and GeomTweedie add mu and mu^2 to the diagonal of Sigma.")
        .value("Constant", Variance::Constant).value("Power", Variance::Power).value("BinomialP", Variance::BinomialP)
        .value("BinomialPQ", Variance::BinomialPQ).value("PoissonTweedie", Variance::PoissonTweedie)
        .value("GeomTweedie", Variance::GeomTweedie);
    nb::enum_<CovLink>(m, "CovLink",
                       "Covariance link: the combination of Z matrices is Omega (Identity), Omega^{-1} (Inverse) or log(Omega) (Expm).")
        .value("Identity", CovLink::Identity).value("Inverse", CovLink::Inverse).value("Expm", CovLink::Expm);
    nb::enum_<Method>(m, "Method", "Update rule for the covariance parameters.")
        .value("Chaser", Method::Chaser).value("Rc", Method::Rc).value("Gradient", Method::Gradient);
    nb::enum_<CrossBlock>(m, "CrossBlock",
                          "Derivative convention for the cross blocks of C when there are several responses and some Z is dense: "
                          "RCompatible reproduces the R package, Exact uses the exact derivative.")
        .value("RCompatible", CrossBlock::RCompatible).value("Exact", CrossBlock::Exact);

    nb::enum_<Structure>(m, "Structure",
                         "How the covariance is organised: Auto picks the cheapest exact organisation, Dense forces one block, "
                         "Blocks forces the decomposition into connected components of the Z sparsity pattern, Sparse forces the approximate "
                         "sparse method (one response, identity covariance link).")
        .value("Auto", Structure::Auto).value("Dense", Structure::Dense).value("Blocks", Structure::Blocks)
        .value("Sparse", Structure::Sparse);

    nb::class_<Options>(m, "Options", "Fitting options.")
        .def(nb::init<>())
        .def_rw("max_iter", &Options::max_iter, "Maximum number of iterations (including the starting point).")
        .def_rw("tol", &Options::tol, "Convergence tolerance on the maximum absolute change of any parameter.")
        .def_rw("method", &Options::method, "Covariance update rule (Method).")
        .def_rw("tuning", &Options::tuning, "Step scaling of the covariance update.")
        .def_rw("correct", &Options::correct, "Include the bias-correction term of the Pearson estimating function.")
        .def_rw("compute_vcov", &Options::compute_vcov, "Compute the joint sandwich covariance of the estimators.")
        .def_rw("verbose", &Options::verbose, "Print the covariance parameters at every iteration to stderr.")
        .def_rw("cross_block", &Options::cross_block, "Cross-block derivative convention (CrossBlock).")
        .def_rw("r_diag_variability", &Options::r_diag_variability,
                "Zero the off-diagonal terms of the Pearson variability (as the R package does).")
        .def_rw("structure", &Options::structure, "Covariance organisation (Structure).")
        .def_rw("block_target", &Options::block_target,
                "Small connected components are merged into batches of about this many units.")
        .def_rw("block_threads", &Options::block_threads,
                "Threads used to process blocks (OpenMP builds); 0 = OpenMP default.")
        .def_rw("sparse_dense_max_n", &Options::sparse_dense_max_n,
                "Auto: a single block with more units than this and a sparse Z uses the sparse method.")
        .def_rw("n_probes", &Options::n_probes, "Sparse method: random probes during the iterations (>= 2).")
        .def_rw("n_probes_final", &Options::n_probes_final, "Sparse method: probes for the final sensitivity and covariance (>= 8).")
        .def_rw("probe_seed", &Options::probe_seed, "Sparse method: seed of the (fixed) probes.")
        .def("__repr__", &repr_options);

    nb::class_<Params>(m, "Params", "Model parameters, one entry per response (rho has K(K-1)/2 entries).")
        .def(nb::init<>())
        .def("__init__", [](Params* self, std::vector<VectorXd> beta, std::vector<VectorXd> power, std::vector<VectorXd> tau,
                            std::optional<VectorXd> rho) {
                 new (self) Params();
                 self->beta = std::move(beta); self->power = std::move(power); self->tau = std::move(tau);
                 self->rho = rho ? *rho : VectorXd();
             }, "beta"_a, "power"_a, "tau"_a, "rho"_a = nb::none(),
             "Parameters: `beta`, `power` and `tau` are lists with one array per response; `rho` is None for one response.")
        .def_rw("beta", &Params::beta, "Regression coefficients, one array per response.")
        .def_rw("power", &Params::power, "Variance exponents, one array per response.")
        .def_rw("tau", &Params::tau, "Dispersion parameters (one per Z matrix), one array per response.")
        .def_rw("rho", &Params::rho, "Between-response correlations (upper triangle, row-major).");

    nb::class_<FitResult>(m, "FitResult", "Result of a fit. Arrays are read-only views that keep the result alive.")
        .def_ro("beta", &FitResult::beta, "Stacked regression coefficients.")
        .def_ro("cov", &FitResult::cov, "Covariance parameters: rho, then (power, tau) for each response.")
        .def_ro("params", &FitResult::params, "The estimates as a `Params` object.")
        .def_ro("iter_beta", &FitResult::iter_beta, "Regression coefficients at every iteration (one row each).")
        .def_ro("iter_cov", &FitResult::iter_cov, "Covariance parameters at every iteration.")
        .def_ro("score_beta", &FitResult::score_beta, "Quasi-score at every iteration (row 0 is NaN).")
        .def_ro("score_cov", &FitResult::score_cov, "Pearson estimating function at every iteration (row 0 is NaN).")
        .def_ro("iterations", &FitResult::iterations, "Number of rows of the history.")
        .def_ro("converged", &FitResult::converged, "Whether the tolerance was reached.")
        .def_ro("stopped", &FitResult::stopped, "Whether the callback stopped the fit.")
        .def_ro("fitted", &FitResult::fitted, "Fitted means, responses stacked.")
        .def_ro("residuals", &FitResult::residuals, "Raw residuals y - mu, responses stacked.")
        .def_ro("vcov", &FitResult::vcov, "Joint covariance of (beta, cov); empty if compute_vcov=False.")
        .def_ro("inv_S_beta", &FitResult::inv_S_beta, "Inverse sensitivity matrix of beta.")
        .def_ro("S_cov", &FitResult::S_cov, "Pearson sensitivity matrix of the covariance parameters.")
        .def_ro("V_beta", &FitResult::V_beta, "Variability matrix of beta (empty if compute_vcov=False).")
        .def_ro("V_cov", &FitResult::V_cov, "Pearson variability matrix (empty if compute_vcov=False).")
        .def_ro("n_beta", &FitResult::n_beta, "Number of regression coefficients per response.")
        .def_ro("n_tau", &FitResult::n_tau, "Number of dispersion parameters per response.")
        .def_ro("n_power", &FitResult::n_power, "Number of estimated variance exponents per response.")
        .def_ro("n_rho", &FitResult::n_rho, "Number of between-response correlations.")
        .def_ro("structure", &FitResult::structure, "Covariance organisation used: 'diagonal', 'dense', 'blocks' or 'sparse'.")
        .def_ro("mc_rel_error", &FitResult::mc_rel_error,
                "Sparse method: largest relative Monte Carlo standard error of the stochastic traces (0 otherwise).")
        .def_ro("n_probes", &FitResult::n_probes, "Sparse method: probes used for the final quantities.")
        .def_ro("n_blocks", &FitResult::n_blocks, "Number of blocks (structure == 'blocks').")
        .def_ro("max_block", &FitResult::max_block, "Units in the largest block.");

    nb::class_<CovarianceOut>(m, "CovarianceOut", "Covariance structure of the model at given parameters.")
        .def_ro("mu", &CovarianceOut::mu, "Stacked mean.")
        .def_ro("diag_C", &CovarianceOut::diag_C, "Diagonal of C (variances), stacked.")
        .def_ro("C", &CovarianceOut::C, "Full covariance matrix (empty unless full=True).")
        .def_ro("Sigma", &CovarianceOut::Sigma, "Within-response matrices Sigma_k (empty unless full=True).")
        .def_ro("Sigma_between", &CovarianceOut::Sigma_between, "Between-response correlation matrix.");

    nb::class_<SparseZ>(m, "SparseZ",
                        "Sparse symmetric Z matrix built from CSC arrays (indptr, indices, data) of a SciPy sparse matrix.")
        .def("__init__", [](SparseZ* self, long n, const I64& indptr, const I64& indices, const F64& data) {
                 new (self) SparseZ(make_sparse_z(n, indptr, indices, data));
             }, "n"_a, "indptr"_a, "indices"_a, "data"_a,
             "Validates the arrays (ranges, finite values, symmetry) and stores the matrix.")
        .def_prop_ro("n", [](const SparseZ& z) { return (long)z.S.rows(); }, "Number of rows (= columns).")
        .def_prop_ro("nnz", [](const SparseZ& z) { return (long)z.S.nonZeros(); }, "Number of stored non-zero entries.");

    nb::class_<Problem>(m, "Problem",
                        "Model specification that can be fitted many times. Thread-safe: concurrent fits release the GIL.")
        .def(nb::init<>())
        .def("add_response", &Problem::add_response,
             "X"_a, "link"_a, "variance"_a, "covariance"_a, "Z"_a, "power_fixed"_a = true,
             "offset"_a = nb::none(), "ntrial"_a = nb::none(), "detect_diagonal"_a = true,
             "Adds a response.\n\n"
             "Parameters\n----------\n"
             "X : (n, p) array\n    Design matrix.\n"
             "link, variance, covariance : Link, Variance, CovLink\n"
             "Z : list of arrays\n    Each 1-D array is a diagonal Z; each 2-D array is a dense symmetric n x n Z. "
             "Identical or proportional Z matrices are rejected.\n"
             "power_fixed : bool\n    Hold the variance exponents at their starting values.\n"
             "offset : (n,) array, optional\n    Added to the linear predictor.\n"
             "ntrial : (n,) array, optional\n    Number of trials (binomial variances); defaults to 1.\n"
             "detect_diagonal : bool\n    Store a dense Z with an all-zero off-diagonal as a diagonal.")
        .def_prop_ro("n", [](const Problem& p) { return (long)p.n; }, "Number of observations per response.")
        .def_prop_ro("n_responses", [](const Problem& p) { return p.specs.size(); }, "Number of responses K.")
        .def_prop_ro("n_beta", [](const Problem& p) { std::vector<long> v; for (auto& s : p.specs) v.push_back((long)s.X.cols()); return v; },
                     "Number of regression coefficients per response.")
        .def_prop_ro("n_z", [](const Problem& p) { std::vector<long> v; for (auto& s : p.specs) v.push_back((long)s.Z.size()); return v; },
                     "Number of Z matrices per response.")
        .def_prop_ro("z_kinds", [](const Problem& p) {
            std::vector<std::vector<std::string>> v;
            for (auto& s : p.specs) {
                std::vector<std::string> r;
                for (auto& z : s.Z) r.push_back(z.is_diag ? "diagonal" : (z.is_sparse ? "sparse" : "dense"));
                v.push_back(r);
            }
            return v; }, "For every response, the storage of each Z: 'diagonal', 'dense' or 'sparse'.")
        .def("structure", [](const Problem& p, int target, int sparse_dense_max_n) {
                p.require_ready();
                std::shared_lock<std::shared_mutex> lk(p.mtx);
                nb::dict d;
                auto part = p.partition(target);
                if (!part) { d["kind"] = "diagonal"; d["n_blocks"] = 0L; d["max_block"] = 0L; d["n_components"] = (long)p.n;
                             d["n_singletons"] = (long)p.n; d["max_component"] = 1L; return d; }
                d["kind"] = part->useful ? "blocks" : "dense";
                if (!part->useful && detail::sparse_method_applicable(p.specs) && p.n > sparse_dense_max_n &&
                    detail::z_density(p.specs) <= 0.25)
                    d["kind"] = "sparse";
                d["n_blocks"] = (long)part->n_batches; d["max_block"] = (long)part->max_batch_units();
                d["n_components"] = (long)part->n_components; d["n_singletons"] = (long)part->n_singletons;
                d["max_component"] = (long)part->max_component;
                return d; }, "block_target"_a = 32, "sparse_dense_max_n"_a = 4000,
             "Analysis of the covariance structure: 'diagonal' (per-observation path), 'blocks' (exact decomposition into "
             "independent groups of units), 'sparse' (one large connected block handled by the approximate sparse method) or "
             "'dense' (a single connected block, exact dense computation), with the number and size of the blocks.")
        .def_prop_ro("z_is_diagonal", [](const Problem& p) {
            std::vector<std::vector<bool>> v;
            for (auto& s : p.specs) { std::vector<bool> r; for (auto& z : s.Z) r.push_back(z.is_diag); v.push_back(r); }
            return v; }, "For every response, whether each Z is stored as a diagonal.")
        .def("initial_values", [](const Problem& p, const Arr& y, const std::optional<Arr>& weights) {
                p.require_ready(); VectorXd yv = check_y(p, y); VectorXd wv = check_weights(p, weights);
                std::shared_lock<std::shared_mutex> lk(p.mtx);
                nb::gil_scoped_release rel; return initial_values(p.specs, yv, wv); }, "y"_a, "weights"_a = nb::none(),
             "Automatic starting values from per-response GLM fits. `y` is the stacked response vector (NaN allowed); "
             "`weights` (optional) are prior weights of those fits and rows with weight 0 are skipped.")
        .def("fit", &do_fit, "y"_a, "init"_a = nb::none(), "weights"_a = nb::none(), "options"_a = Options(),
             "callback"_a = nb::none(),
             "Fits the model.\n\n"
             "Parameters\n----------\n"
             "y : (K*n,) array\n    Responses stacked; NaN marks a missing value (weight 0).\n"
             "init : Params, optional\n    Starting values; computed automatically if omitted.\n"
             "weights : (K*n,) array, optional\n    Non-negative observation weights.\n"
             "options : Options\n"
             "callback : callable, optional\n    Called as callback(iteration, beta, cov) after each iteration; "
             "return True to stop. Exceptions propagate. Ctrl-C is honoured between iterations.\n\n"
             "Raises\n------\n"
             "InvalidInput\n    Invalid arguments (a ValueError).\n"
             "NumericalError\n    A numerical step failed (an ArithmeticError).")
        .def("covariance", [](const Problem& p, const Params& q, bool full) {
                p.require_ready(); check_params(p, q);
                std::shared_lock<std::shared_mutex> lk(p.mtx);
                nb::gil_scoped_release rel;
                auto part = p.partition(32);
                return covariance_matrices(p.specs, q, full, part.get()); },
             "params"_a, "full"_a = false,
             "Covariance structure at `params`. With full=True also returns the dense C and Sigma_k "
             "(memory O((K n)^2)).");

    m.def("link_eval", [](Link l, const VectorXd& eta) {
              VectorXd mu, dmu; link_eval(l, eta, mu, dmu); return nb::make_tuple(mu, dmu); },
          "link"_a, "eta"_a, "Inverse link: returns (mu, dmu/deta) for the linear predictor `eta`.");

    m.def("set_num_threads", [](int num_threads) {
#ifdef MCGLM_HAVE_OPENBLAS
        if (num_threads < 1) throw InvalidInput("the number of threads must be >= 1");
        openblas_set_num_threads(num_threads);
#else
        (void)num_threads;
#endif
    }, "num_threads"_a, "Sets the number of OpenBLAS threads (no effect if built without OpenBLAS).");
    m.def("get_num_threads", []() {
#ifdef MCGLM_HAVE_OPENBLAS
        return openblas_get_num_threads();
#else
        return 1;
#endif
    }, "Number of OpenBLAS threads in use.");
    m.def("build_info", []() {
        nb::dict d;
#ifdef MCGLM_HAVE_OPENBLAS
        d["blas"] = std::string(openblas_get_config());
        d["openblas"] = true;
#else
        d["blas"] = "Eigen (no external BLAS)"; d["openblas"] = false;
#endif
#ifdef MCGLM_USE_LAPACKE
        d["lapacke"] = true;
#else
        d["lapacke"] = false;
#endif
#ifdef MCGLM_NATIVE
        d["march_native"] = true;
#else
        d["march_native"] = false;
#endif
        d["eigen"] = std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." + std::to_string(EIGEN_MINOR_VERSION);
        d["cxx_standard"] = (long)__cplusplus;
        return d;
    }, "Build configuration: BLAS library, LAPACKE, -march=native, Eigen version, C++ standard.");
}
