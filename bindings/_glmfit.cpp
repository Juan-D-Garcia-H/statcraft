// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

// _glmfit.cpp -- nanobind bindings for the native GLM engine.
//
// Exposed components:
//   1. IrlsEngine  : classical and custom-family IRLS            (irls.hpp)
//   2. MleSolver   : direct maximum likelihood via L-BFGS-B/BFGS (mle_solver.hpp)
//   3. QuasiSolver : quasi-likelihood optimization                (quasi_solver.hpp)
//   4. Drivers     : two-step estimation (NB theta, quasi families) (irls_two_step.hpp)
//
// nanobind note: STL type casters are opt-in. If a bound member or return type
// uses std::map, std::tuple, ..., include the matching <nanobind/stl/...> header.

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../include/glmfit/irls.hpp"
#include "../include/glmfit/irls_two_step.hpp"
#include "../include/glmfit/minimizer_choice.hpp"
#include "../include/glmfit/mle_solver.hpp"
#include "../include/glmfit/quasi_solver.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace nb = nanobind;


namespace {

// Adapts a Python callable (ndarray -> ndarray of the same length) to the
// engine's vectorized function signature (used for variance, mu_eta and linkinv).
std::function<void(const Eigen::ArrayXd&, Eigen::ArrayXd&)> wrap_link(nb::callable f)
{
    return [f](const Eigen::ArrayXd &in, Eigen::ArrayXd &out) {
        Eigen::VectorXd in_v = in.matrix();
        Eigen::VectorXd out_v = nb::cast<Eigen::VectorXd>(f(in_v));
        if (out_v.size() != in.size())
            throw std::runtime_error(
                "_glmsolve (custom callable): the function returned an array of size "
                + std::to_string(out_v.size()) + ", expected "
                + std::to_string(in.size()));
        out = out_v.array();
    };
}

// Adapts a Python callable f(y, mu, weights) -> float returning the total deviance.
std::function<double(const Eigen::ArrayXd&, const Eigen::ArrayXd&, const Eigen::ArrayXd&)> wrap_dev_resids(nb::callable f)
{
    return [f](const Eigen::ArrayXd &y, const Eigen::ArrayXd &mu,
               const Eigen::ArrayXd &w) {
        Eigen::VectorXd yv = y.matrix(), muv = mu.matrix(), wv = w.matrix();
        return nb::cast<double>(f(yv, muv, wv));
    };
}

// Adapts a Python callable f(v) -> truthy (used for valideta and validmu).
std::function<bool(const Eigen::ArrayXd&)> wrap_valid(nb::callable f)
{
    return [f](const Eigen::ArrayXd &v) {
        Eigen::VectorXd vv = v.matrix();
        nb::object res = f(vv);
        // Python truthiness: accepts bool, numpy.bool_, int, ...
        int r = PyObject_IsTrue(res.ptr());
        if (r < 0)
            throw nb::python_error();
        return r == 1;
    };
}

// Owns copies of the data (X, Y, weights, offset) so the engine can keep
// non-owning Eigen::Map views for its whole lifetime, and forwards to `glm::IrlsSolver`.
// Two constructors: built-in family (FAM_* code) or user-supplied callables.
class PyIrlsEngine
{
public:
    PyIrlsEngine(const Eigen::MatrixXd &X,
                 const Eigen::VectorXd &Y,
                 const Eigen::VectorXd &weights,
                 const Eigen::VectorXd &offset,
                 int fam_code,
                 const glm::FamilyParams &fam_params,
                 double tol, int maxit, int type,
                 bool is_big_matrix, bool firth,
                 const glm::IrlsOptions &options)
        : X_(X), Y_(Y), weights_(weights), offset_(offset)
    {
        check_dims();
        Eigen::Map<const Eigen::MatrixXd> Xm(X_.data(), X_.rows(), X_.cols());
        Eigen::Map<const Eigen::VectorXd> Ym(Y_.data(), Y_.size());
        Eigen::Map<const Eigen::VectorXd> Wm(weights_.data(), weights_.size());
        Eigen::Map<const Eigen::VectorXd> Om(offset_.data(), offset_.size());

        engine_ = std::make_unique<glm::IrlsSolver>(Xm, Ym, Wm, Om,
                                         fam_code, fam_params,
                                         tol, maxit, type,
                                         is_big_matrix, firth, options);
    }

    PyIrlsEngine(const Eigen::MatrixXd &X,
                 const Eigen::VectorXd &Y,
                 const Eigen::VectorXd &weights,
                 const Eigen::VectorXd &offset,
                 nb::callable variance_fn, nb::callable mu_eta_fn,
                 nb::callable linkinv_fn, nb::callable dev_resids_fn,
                 nb::callable valideta_fn, nb::callable validmu_fn,
                 double tol, int maxit, int type,
                 bool is_big_matrix, bool firth,
                 const glm::IrlsOptions &options)
        : X_(X), Y_(Y), weights_(weights), offset_(offset)
    {
        check_dims();
        Eigen::Map<const Eigen::MatrixXd> Xm(X_.data(), X_.rows(), X_.cols());
        Eigen::Map<const Eigen::VectorXd> Ym(Y_.data(), Y_.size());
        Eigen::Map<const Eigen::VectorXd> Wm(weights_.data(), weights_.size());
        Eigen::Map<const Eigen::VectorXd> Om(offset_.data(), offset_.size());

        engine_ = std::make_unique<glm::IrlsSolver>(
            Xm, Ym, Wm, Om,
            wrap_link(variance_fn), wrap_link(mu_eta_fn), wrap_link(linkinv_fn),
            wrap_dev_resids(dev_resids_fn),
            wrap_valid(valideta_fn), wrap_valid(validmu_fn),
            tol, maxit, type, is_big_matrix,
            glm::FAM_UNKNOWN, glm::FamilyParams(), firth, options);
    }

    void init_parms(const Eigen::VectorXd &start,
                    const Eigen::VectorXd &mu,
                    const Eigen::VectorXd &eta)
    {
        if (start.size() != X_.cols())
            throw std::invalid_argument("IrlsEngine.init_parms: start must have size p = ncol(X)");
        if (mu.size() != X_.rows() || eta.size() != X_.rows())
            throw std::invalid_argument("IrlsEngine.init_parms: mu and eta must have size n = nrow(X)");
        Eigen::Map<const Eigen::VectorXd> s(start.data(), start.size());
        Eigen::Map<const Eigen::VectorXd> m(mu.data(), mu.size());
        Eigen::Map<const Eigen::VectorXd> e(eta.data(), eta.size());
        engine_->init_parms(s, m, e);
    }

    int solve(int maxit) { return engine_->solve(maxit); }

    Eigen::VectorXd get_beta()          { return engine_->get_beta(); }
    Eigen::VectorXd get_se()            { return engine_->get_se(); }
    Eigen::VectorXd get_mu()            { return engine_->get_mu(); }
    Eigen::VectorXd get_eta()           { return engine_->get_eta(); }
    Eigen::VectorXd get_weights()       { return engine_->get_weights(); }
    Eigen::VectorXd get_w()             { return engine_->get_w(); }
    double           get_dev()          { return engine_->get_dev(); }
    double           get_std_dev()      { return engine_->get_std_dev(); }
    int              get_rank()         { return engine_->get_rank(); }
    Eigen::MatrixXd get_vcov()          { return engine_->get_vcov(); }
    bool             get_converged()    { return engine_->get_converged(); }
    double           get_log_det_XtWX() const { return engine_->get_log_det_XtWX(); }
    bool             get_firth()        const { return engine_->get_firth(); }
    int              get_iterations()   const { return engine_->get_iterations(); }
    std::vector<int> get_aliased()      const { return engine_->get_aliased(); }
    void set_fam_params(const glm::FamilyParams &p) { engine_->set_fam_params(p); }

private:
    void check_dims() const
    {
        if (X_.rows() != Y_.size())
            throw std::invalid_argument("IrlsEngine: X and Y have a different number of rows");
        if (weights_.size() != X_.rows())
            throw std::invalid_argument("IrlsEngine: weights must have size nrow(X)");
        if (offset_.size() != X_.rows())
            throw std::invalid_argument("IrlsEngine: offset must have size nrow(X)");
    }

    Eigen::MatrixXd X_;
    Eigen::VectorXd Y_, weights_, offset_;
    std::unique_ptr<glm::IrlsSolver> engine_;
};

} // namespace


NB_MODULE(_glmfit, m)
{
    m.doc() =
        "Native C++ engine for generalized linear models.\n"
        "\n"
        "Components: IrlsEngine (IRLS), MleSolver (direct maximum likelihood), "
        "QuasiSolver (quasi-likelihood), and the two-step drivers fit_nb_two_step "
        "and fit_quasi.\n"
        "\n"
        "Module constants: FAM_<FAMILY>_<LINK> are integer family codes accepted by "
        "IrlsEngine, the family helpers (linkinv, variance, dev_resids, valid_eta, "
        "valid_mu) and the fit drivers; FAM_QUASI_* select quasi-likelihood families; "
        "TYPE_* select the matrix decomposition used by IRLS (0=column-pivoted QR, "
        "1=QR, 2=Cholesky, 3=LDLT, 4=full-pivot QR, 5=SVD).";

    nb::class_<glm::FamilyParams>(m, "FamilyParams",
            "Auxiliary family parameters used by IrlsEngine and the family helpers.\n"
            "\n"
            "Attributes:\n"
            "  theta       Negative-binomial dispersion: V(mu) = mu + mu^2 / theta.\n"
            "  var_power   Tweedie variance power: V(mu) = mu^var_power.\n"
            "  link_power  Exponent of the power / Box-Cox link: mu = eta^(1/link_power),\n"
            "              or the log link when link_power == 0. Used by FAM_*_POWER.")
        .def(nb::init<>(), "Create parameters with the engine's default values.")
        .def_rw("theta", &glm::FamilyParams::theta,
                "Negative-binomial dispersion theta.")
        .def_rw("var_power", &glm::FamilyParams::var_power,
                "Tweedie variance power.")
        .def_rw("link_power", &glm::FamilyParams::link_power,
                "Exponent of the power / Box-Cox link (0 means log link).");

    nb::class_<glm::IrlsOptions>(m, "IrlsOptions",
            "Optional tuning knobs of the IRLS engine (all have sensible defaults).\n"
            "\n"
            "Attributes:\n"
            "  rank_tol      Relative pivot / singular-value cut-off used to detect rank deficiency (default 1e-11).\n"
            "  max_halvings  Maximum step halvings per iteration (default 30).\n"
            "  max_step      Firth only: cap on max|delta beta| per iteration (default 5).\n"
            "  chunk_rows    Rows per streaming block; 0 = automatic (env FASTGLM_CHUNK_ROWS or 16384).")
        .def(nb::init<>(), "Create options with the default values.")
        .def_rw("rank_tol", &glm::IrlsOptions::rank_tol, "Relative rank-detection tolerance.")
        .def_rw("max_halvings", &glm::IrlsOptions::max_halvings, "Maximum step halvings per iteration.")
        .def_rw("max_step", &glm::IrlsOptions::max_step, "Firth: cap on the per-iteration coefficient change.")
        .def_rw("chunk_rows", &glm::IrlsOptions::chunk_rows, "Rows per streaming block (0 = automatic).");

    // Integer family codes: FAM_<FAMILY>_<LINK>
    m.attr("FAM_UNKNOWN")           = static_cast<int>(glm::FAM_UNKNOWN);
    m.attr("FAM_GAUSSIAN_IDENTITY") = static_cast<int>(glm::FAM_GAUSSIAN_IDENTITY);
    m.attr("FAM_GAUSSIAN_LOG")      = static_cast<int>(glm::FAM_GAUSSIAN_LOG);
    m.attr("FAM_GAUSSIAN_INVERSE")  = static_cast<int>(glm::FAM_GAUSSIAN_INVERSE);
    m.attr("FAM_BINOMIAL_LOGIT")    = static_cast<int>(glm::FAM_BINOMIAL_LOGIT);
    m.attr("FAM_BINOMIAL_PROBIT")   = static_cast<int>(glm::FAM_BINOMIAL_PROBIT);
    m.attr("FAM_BINOMIAL_CLOGLOG")  = static_cast<int>(glm::FAM_BINOMIAL_CLOGLOG);
    m.attr("FAM_BINOMIAL_LOG")      = static_cast<int>(glm::FAM_BINOMIAL_LOG);
    m.attr("FAM_POISSON_LOG")       = static_cast<int>(glm::FAM_POISSON_LOG);
    m.attr("FAM_POISSON_IDENTITY")  = static_cast<int>(glm::FAM_POISSON_IDENTITY);
    m.attr("FAM_POISSON_SQRT")      = static_cast<int>(glm::FAM_POISSON_SQRT);
    m.attr("FAM_GAMMA_LOG")         = static_cast<int>(glm::FAM_GAMMA_LOG);
    m.attr("FAM_GAMMA_INVERSE")     = static_cast<int>(glm::FAM_GAMMA_INVERSE);
    m.attr("FAM_GAMMA_IDENTITY")    = static_cast<int>(glm::FAM_GAMMA_IDENTITY);
    m.attr("FAM_INVGAUSS_INVMU2")   = static_cast<int>(glm::FAM_INVGAUSS_INVMU2);
    m.attr("FAM_INVGAUSS_LOG")      = static_cast<int>(glm::FAM_INVGAUSS_LOG);
    m.attr("FAM_INVGAUSS_IDENTITY") = static_cast<int>(glm::FAM_INVGAUSS_IDENTITY);
    m.attr("FAM_INVGAUSS_INVERSE")  = static_cast<int>(glm::FAM_INVGAUSS_INVERSE);
    m.attr("FAM_NB_LOG")            = static_cast<int>(glm::FAM_NB_LOG);
    m.attr("FAM_NB_SQRT")           = static_cast<int>(glm::FAM_NB_SQRT);
    m.attr("FAM_NB_IDENTITY")       = static_cast<int>(glm::FAM_NB_IDENTITY);
    m.attr("FAM_TWEEDIE_LOG")       = static_cast<int>(glm::FAM_TWEEDIE_LOG);
    m.attr("FAM_TWEEDIE_IDENTITY")  = static_cast<int>(glm::FAM_TWEEDIE_IDENTITY);
    m.attr("FAM_TWEEDIE_INVERSE")   = static_cast<int>(glm::FAM_TWEEDIE_INVERSE);
    m.attr("FAM_TWEEDIE_SQRT")      = static_cast<int>(glm::FAM_TWEEDIE_SQRT);

    m.attr("FAM_QUASI_POISSON_LOG")     = static_cast<int>(glm::FAM_QUASI_POISSON_LOG);
    m.attr("FAM_QUASI_POISSON_IDENTITY") = static_cast<int>(glm::FAM_QUASI_POISSON_IDENTITY);
    m.attr("FAM_QUASI_POISSON_IDENT") = static_cast<int>(glm::FAM_QUASI_POISSON_IDENTITY);  // legacy Python name
    m.attr("FAM_QUASI_BINOMIAL_LOGIT")  = static_cast<int>(glm::FAM_QUASI_BINOMIAL_LOGIT);
    m.attr("FAM_QUASI_BINOMIAL_PROBIT") = static_cast<int>(glm::FAM_QUASI_BINOMIAL_PROBIT);
    m.attr("FAM_QUASI_GAUSSIAN_IDENTITY") = static_cast<int>(glm::FAM_QUASI_GAUSSIAN_IDENTITY);
    m.attr("FAM_QUASI_GAUSSIAN_IDENT") = static_cast<int>(glm::FAM_QUASI_GAUSSIAN_IDENTITY);  // legacy Python name
    m.attr("FAM_QUASI_GAUSSIAN_LOG")    = static_cast<int>(glm::FAM_QUASI_GAUSSIAN_LOG);
    m.attr("FAM_QUASI_GAMMA_LOG")       = static_cast<int>(glm::FAM_QUASI_GAMMA_LOG);
    m.attr("FAM_QUASI_GAMMA_IDENTITY") = static_cast<int>(glm::FAM_QUASI_GAMMA_IDENTITY);
    m.attr("FAM_QUASI_GAMMA_IDENT") = static_cast<int>(glm::FAM_QUASI_GAMMA_IDENTITY);  // legacy Python name
    m.attr("FAM_QUASI_GAMMA_INVERSE")   = static_cast<int>(glm::FAM_QUASI_GAMMA_INVERSE);
    m.attr("FAM_QUASI_INVGAUSSIAN_LOG")     = static_cast<int>(glm::FAM_QUASI_INVGAUSSIAN_LOG);
    m.attr("FAM_QUASI_INVGAUSSIAN_IDENTITY") = static_cast<int>(glm::FAM_QUASI_INVGAUSSIAN_IDENTITY);
    m.attr("FAM_QUASI_INVGAUSSIAN_IDENT") = static_cast<int>(glm::FAM_QUASI_INVGAUSSIAN_IDENTITY);  // legacy Python name
    m.attr("FAM_QUASI_INVGAUSSIAN_INVERSE") = static_cast<int>(glm::FAM_QUASI_INVGAUSSIAN_INVERSE);

    m.attr("FAM_BINOMIAL_CAUCHIT")   = static_cast<int>(glm::FAM_BINOMIAL_CAUCHIT);
    m.attr("FAM_BINOMIAL_LOGLOG")    = static_cast<int>(glm::FAM_BINOMIAL_LOGLOG);
    m.attr("FAM_POISSON_INVERSE")    = static_cast<int>(glm::FAM_POISSON_INVERSE);
    m.attr("FAM_NB_INVERSE")         = static_cast<int>(glm::FAM_NB_INVERSE);
    m.attr("FAM_PARETO_LOG")         = static_cast<int>(glm::FAM_PARETO_LOG);
    m.attr("FAM_INVGAMMA_LOG")       = static_cast<int>(glm::FAM_INVGAMMA_LOG);
    m.attr("FAM_INVGAMMA_INVERSE")   = static_cast<int>(glm::FAM_INVGAMMA_INVERSE);
    m.attr("FAM_EXPONENTIAL_IDENTITY") = static_cast<int>(glm::FAM_EXPONENTIAL_IDENTITY);
    m.attr("FAM_EXPONENTIAL_LOG")      = static_cast<int>(glm::FAM_EXPONENTIAL_LOG);
    m.attr("FAM_EXPONENTIAL_INVERSE")  = static_cast<int>(glm::FAM_EXPONENTIAL_INVERSE);

    m.attr("FAM_BETA_LOGIT")   = static_cast<int>(glm::FAM_BETA_LOGIT);
    m.attr("FAM_BETA_PROBIT")  = static_cast<int>(glm::FAM_BETA_PROBIT);
    m.attr("FAM_BETA_CLOGLOG") = static_cast<int>(glm::FAM_BETA_CLOGLOG);
    m.attr("FAM_BETA_LOGLOG")  = static_cast<int>(glm::FAM_BETA_LOGLOG);
    m.attr("FAM_BETA_CAUCHIT") = static_cast<int>(glm::FAM_BETA_CAUCHIT);

    m.attr("FAM_LOGNORMAL_IDENTITY") = static_cast<int>(glm::FAM_LOGNORMAL_IDENTITY);
    m.attr("FAM_LOGNORMAL_LOG")      = static_cast<int>(glm::FAM_LOGNORMAL_LOG);
    m.attr("FAM_LOGNORMAL_INVERSE")  = static_cast<int>(glm::FAM_LOGNORMAL_INVERSE);
    m.attr("FAM_WEIBULL_LOG")        = static_cast<int>(glm::FAM_WEIBULL_LOG);
    m.attr("FAM_WEIBULL_IDENTITY")   = static_cast<int>(glm::FAM_WEIBULL_IDENTITY);
    m.attr("FAM_GUMBEL_IDENTITY")    = static_cast<int>(glm::FAM_GUMBEL_IDENTITY);
    m.attr("FAM_GUMBEL_LOG")         = static_cast<int>(glm::FAM_GUMBEL_LOG);
    m.attr("FAM_LOGLOGISTIC_IDENTITY") = static_cast<int>(glm::FAM_LOGLOGISTIC_IDENTITY);
    m.attr("FAM_LOGLOGISTIC_LOG")      = static_cast<int>(glm::FAM_LOGLOGISTIC_LOG);

    m.attr("FAM_SIMPLEX_LOGIT")      = static_cast<int>(glm::FAM_SIMPLEX_LOGIT);
    m.attr("FAM_SIMPLEX_PROBIT")     = static_cast<int>(glm::FAM_SIMPLEX_PROBIT);
    m.attr("FAM_SIMPLEX_LOGLOG")     = static_cast<int>(glm::FAM_SIMPLEX_LOGLOG);
    m.attr("FAM_SIMPLEX_CLOGLOG")    = static_cast<int>(glm::FAM_SIMPLEX_CLOGLOG);
    m.attr("FAM_SIMPLEX_CAUCHIT")    = static_cast<int>(glm::FAM_SIMPLEX_CAUCHIT);

    m.attr("FAM_MULTINOMIAL_LOGLOG") = static_cast<int>(glm::FAM_MULTINOMIAL_LOGLOG);
    m.attr("FAM_MULTINOMIAL_CAUCHIT") = static_cast<int>(glm::FAM_MULTINOMIAL_CAUCHIT);
    m.attr("FAM_MULTINOMIAL_LOGIT")  = static_cast<int>(glm::FAM_MULTINOMIAL_LOGIT);
    m.attr("FAM_MULTINOMIAL_PROBIT") = static_cast<int>(glm::FAM_MULTINOMIAL_PROBIT);
    m.attr("FAM_MULTINOMIAL_CLOGLOG") = static_cast<int>(glm::FAM_MULTINOMIAL_CLOGLOG);
    m.attr("FAM_ORDINAL_LOGIT")      = static_cast<int>(glm::FAM_ORDINAL_LOGIT);
    m.attr("FAM_ORDINAL_PROBIT")     = static_cast<int>(glm::FAM_ORDINAL_PROBIT);
    m.attr("FAM_ORDINAL_CLOGLOG")    = static_cast<int>(glm::FAM_ORDINAL_CLOGLOG);
    m.attr("FAM_ORDINAL_LOGLOG")     = static_cast<int>(glm::FAM_ORDINAL_LOGLOG);
    m.attr("FAM_ORDINAL_CAUCHIT")    = static_cast<int>(glm::FAM_ORDINAL_CAUCHIT);
    m.attr("FAM_STUDENT_T_IDENTITY")   = static_cast<int>(glm::FAM_STUDENT_T_IDENTITY);
    m.attr("FAM_STUDENT_T_LOG")        = static_cast<int>(glm::FAM_STUDENT_T_LOG);
    m.attr("FAM_SKEW_NORMAL_IDENTITY") = static_cast<int>(glm::FAM_SKEW_NORMAL_IDENTITY);
    m.attr("FAM_SKEW_NORMAL_LOG")      = static_cast<int>(glm::FAM_SKEW_NORMAL_LOG);
    m.attr("FAM_SKEW_T_IDENTITY")      = static_cast<int>(glm::FAM_SKEW_T_IDENTITY);
    m.attr("FAM_SKEW_T_LOG")           = static_cast<int>(glm::FAM_SKEW_T_LOG);
    m.attr("FAM_LOGISTIC_IDENTITY")    = static_cast<int>(glm::FAM_LOGISTIC_IDENTITY);
    m.attr("FAM_LOGISTIC_LOG")         = static_cast<int>(glm::FAM_LOGISTIC_LOG);
    m.attr("FAM_ZIP_LOG")      = static_cast<int>(glm::FAM_ZIP_LOG);
    m.attr("FAM_ZIP_IDENTITY") = static_cast<int>(glm::FAM_ZIP_IDENTITY);
    m.attr("FAM_ZIP_SQRT")     = static_cast<int>(glm::FAM_ZIP_SQRT);
    m.attr("FAM_ZIP_INVERSE")  = static_cast<int>(glm::FAM_ZIP_INVERSE);
    m.attr("FAM_ZINB_LOG")      = static_cast<int>(glm::FAM_ZINB_LOG);
    m.attr("FAM_ZINB_SQRT")     = static_cast<int>(glm::FAM_ZINB_SQRT);
    m.attr("FAM_ZINB_IDENTITY") = static_cast<int>(glm::FAM_ZINB_IDENTITY);
    m.attr("FAM_ZINB_INVERSE")  = static_cast<int>(glm::FAM_ZINB_INVERSE);
    m.attr("FAM_HURDLE_POISSON_LOG")   = static_cast<int>(glm::FAM_HURDLE_POISSON_LOG);
    m.attr("FAM_HURDLE_POISSON_SQRT")  = static_cast<int>(glm::FAM_HURDLE_POISSON_SQRT);
    m.attr("FAM_HURDLE_NB_LOG")        = static_cast<int>(glm::FAM_HURDLE_NB_LOG);
    m.attr("FAM_HURDLE_NB_SQRT")       = static_cast<int>(glm::FAM_HURDLE_NB_SQRT);
    m.attr("FAM_HURDLE_GAMMA_LOG")     = static_cast<int>(glm::FAM_HURDLE_GAMMA_LOG);
    m.attr("FAM_HURDLE_GAMMA_INVERSE") = static_cast<int>(glm::FAM_HURDLE_GAMMA_INVERSE);
    m.attr("FAM_HURDLE_LOGNORMAL_LOG") = static_cast<int>(glm::FAM_HURDLE_LOGNORMAL_LOG);
    m.attr("FAM_HURDLE_LOGNORMAL_IDENTITY") = static_cast<int>(glm::FAM_HURDLE_LOGNORMAL_IDENTITY);
    m.attr("FAM_HURDLE_LOGNORMAL_ID") = static_cast<int>(glm::FAM_HURDLE_LOGNORMAL_IDENTITY);  // legacy Python name
    m.attr("FAM_HURDLE_BETA_LOGIT")    = static_cast<int>(glm::FAM_HURDLE_BETA_LOGIT);
    m.attr("FAM_HURDLE_BETA_PROBIT")   = static_cast<int>(glm::FAM_HURDLE_BETA_PROBIT);
    m.attr("FAM_HURDLE_BETA_LOGLOG")   = static_cast<int>(glm::FAM_HURDLE_BETA_LOGLOG);
    m.attr("FAM_HURDLE_BETA_CLOGLOG")  = static_cast<int>(glm::FAM_HURDLE_BETA_CLOGLOG);
    m.attr("FAM_HURDLE_BETA_CAUCHIT")  = static_cast<int>(glm::FAM_HURDLE_BETA_CAUCHIT);

    m.attr("FAM_HURDLE_GEOMETRIC_LOG")  = static_cast<int>(glm::FAM_HURDLE_GEOMETRIC_LOG);
    m.attr("FAM_HURDLE_GEOMETRIC_IDENTITY")  = static_cast<int>(glm::FAM_HURDLE_GEOMETRIC_IDENTITY);
    m.attr("FAM_HURDLE_GEOMETRIC_SQRT")  = static_cast<int>(glm::FAM_HURDLE_GEOMETRIC_SQRT);

    m.attr("FAM_TRUNC_POISSON_LOG")   = static_cast<int>(glm::FAM_TRUNC_POISSON_LOG);
    m.attr("FAM_TRUNC_POISSON_IDENTITY") = static_cast<int>(glm::FAM_TRUNC_POISSON_IDENTITY);
    m.attr("FAM_TRUNC_POISSON_IDENT") = static_cast<int>(glm::FAM_TRUNC_POISSON_IDENTITY);  // legacy Python name
    m.attr("FAM_TRUNC_NB_LOG")        = static_cast<int>(glm::FAM_TRUNC_NB_LOG);
    m.attr("FAM_TRUNC_NB_IDENTITY")   = static_cast<int>(glm::FAM_TRUNC_NB_IDENTITY);
    m.attr("FAM_TRUNC_GEOMETRIC_LOG") = static_cast<int>(glm::FAM_TRUNC_GEOMETRIC_LOG);
    m.attr("FAM_TRUNC_GAUSSIAN_IDENTITY") = static_cast<int>(glm::FAM_TRUNC_GAUSSIAN_IDENTITY);
    m.attr("FAM_TRUNC_GAUSSIAN_LOG") = static_cast<int>(glm::FAM_TRUNC_GAUSSIAN_LOG);
    m.attr("FAM_TRUNC_GAUSSIAN_INVERSE") = static_cast<int>(glm::FAM_TRUNC_GAUSSIAN_INVERSE);

    // Extended links (codes 200-204)
    m.attr("FAM_GAUSSIAN_SQRT")     = static_cast<int>(glm::FAM_GAUSSIAN_SQRT);
    m.attr("FAM_GAMMA_SQRT")        = static_cast<int>(glm::FAM_GAMMA_SQRT);
    m.attr("FAM_EXPONENTIAL_SQRT")  = static_cast<int>(glm::FAM_EXPONENTIAL_SQRT);
    m.attr("FAM_PARETO_INVERSE")    = static_cast<int>(glm::FAM_PARETO_INVERSE);
    m.attr("FAM_INVGAMMA_IDENTITY") = static_cast<int>(glm::FAM_INVGAMMA_IDENTITY);

    // Generic power / Box-Cox link (codes 205-208; uses FamilyParams::link_power)
    m.attr("FAM_GAMMA_POWER")    = static_cast<int>(glm::FAM_GAMMA_POWER);
    m.attr("FAM_INVGAUSS_POWER") = static_cast<int>(glm::FAM_INVGAUSS_POWER);
    m.attr("FAM_NB_POWER")       = static_cast<int>(glm::FAM_NB_POWER);
    m.attr("FAM_TWEEDIE_POWER")  = static_cast<int>(glm::FAM_TWEEDIE_POWER);

    // Matrix decomposition types for IRLS (argument `type`)
    m.attr("TYPE_COLPIV_QR")     = 0;
    m.attr("TYPE_QR")            = 1;
    m.attr("TYPE_CHOLESKY")      = 2;
    m.attr("TYPE_LDLT")          = 3;
    m.attr("TYPE_FULLPIV_QR")    = 4;
    m.attr("TYPE_SVD")           = 5;

    // -----------------------------------------------------------------
    // 1. IRLS engine (IrlsEngine)
    // -----------------------------------------------------------------
    nb::class_<PyIrlsEngine>(m, "IrlsEngine",
            "Native iteratively reweighted least squares (IRLS) engine.\n"
            "\n"
            "Typical use: construct, call init_parms(start, mu, eta), call solve(maxit),\n"
            "then read the results with the get_* methods. Two constructors are available:\n"
            "one for built-in families (FAM_* code + FamilyParams) and one for custom\n"
            "families defined by Python callables. The engine keeps its own copy of the data.")
        .def(nb::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&,
                      int, const glm::FamilyParams&,
                      double, int, int, bool, bool, const glm::IrlsOptions&>(),
             nb::arg("X"), nb::arg("Y"), nb::arg("weights"), nb::arg("offset"),
             nb::arg("fam_code"),
             nb::arg("fam_params") = glm::FamilyParams(),
             nb::arg("tol") = 1e-6,
             nb::arg("maxit") = 100,
             nb::arg("type") = 1,
             nb::arg("is_big_matrix") = false,
             nb::arg("firth") = false,
             nb::arg("options") = glm::IrlsOptions(),
             "Create an IRLS engine for a built-in family.\n"
             "\n"
             "Parameters:\n"
             "  X              (n, p) design matrix.\n"
             "  Y              (n,) response.\n"
             "  weights        (n,) prior weights.\n"
             "  offset         (n,) offset added to the linear predictor.\n"
             "  fam_code       Family/link code (one of the FAM_* constants).\n"
             "  fam_params     FamilyParams (theta, var_power, link_power).\n"
             "  tol            Convergence tolerance.\n"
             "  maxit          Maximum number of iterations.\n"
             "  type           Matrix decomposition (TYPE_* constant; default 1 = QR).\n"
             "  is_big_matrix  Select the code path intended for large design matrices.\n"
             "  firth          Use Firth's penalized likelihood (bias reduction; exact for any link).\n"
             "  options        IrlsOptions (rank tolerance, step halvings, Firth step cap, chunk rows).\n"
             "\n"
             "Rank-deficient designs do not abort the fit: aliased coefficients are 0 and their se / vcov are NaN\n"
             "(type 5 returns the minimum-norm solution). See get_aliased().\n"
             "Raises ValueError if X, Y, weights and offset have inconsistent sizes.")
        .def(nb::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, const Eigen::VectorXd&,
                      nb::callable, nb::callable, nb::callable, nb::callable,
                      nb::callable, nb::callable,
                      double, int, int, bool, bool, const glm::IrlsOptions&>(),
             nb::arg("X"), nb::arg("Y"), nb::arg("weights"), nb::arg("offset"),
             nb::arg("variance_fn"), nb::arg("mu_eta_fn"), nb::arg("linkinv_fn"),
             nb::arg("dev_resids_fn"), nb::arg("valideta_fn"), nb::arg("validmu_fn"),
             nb::arg("tol") = 1e-6,
             nb::arg("maxit") = 100,
             nb::arg("type") = 1,
             nb::arg("is_big_matrix") = false,
             nb::arg("firth") = false,
             nb::arg("options") = glm::IrlsOptions(),
             "Create an IRLS engine for a custom family defined by Python callables.\n"
             "\n"
             "Each callable receives NumPy arrays and mirrors the members of an R family object:\n"
             "  variance_fn(mu)             -> ndarray, variance function V(mu).\n"
             "  mu_eta_fn(eta)              -> ndarray, derivative d mu / d eta.\n"
             "  linkinv_fn(eta)             -> ndarray, inverse link g^-1(eta).\n"
             "  dev_resids_fn(y, mu, w)     -> float, total deviance.\n"
             "  valideta_fn(eta)            -> truthy value, True if eta is in the valid domain.\n"
             "  validmu_fn(mu)              -> truthy value, True if mu is in the valid domain.\n"
             "Array-valued callables must return an array of the same length as their input.\n"
             "The remaining parameters are described in the built-in-family constructor.\n"
             "\n"
             "Raises ValueError if X, Y, weights and offset have inconsistent sizes.")
        .def("init_parms", &PyIrlsEngine::init_parms,
             nb::arg("start"), nb::arg("mu"), nb::arg("eta"),
             "Initialize the iteration state.\n"
             "\n"
             "Parameters:\n"
             "  start  (p,) starting coefficients.\n"
             "  mu     (n,) starting means.\n"
             "  eta    (n,) starting linear predictor.\n"
             "\n"
             "Raises ValueError if the sizes do not match ncol(X) / nrow(X).")
        .def("solve", &PyIrlsEngine::solve, nb::arg("maxit"),
             "Run IRLS for at most `maxit` iterations and return the integer reported by the native solver (iteration count). Call init_parms first.")
        .def("get_beta", &PyIrlsEngine::get_beta,
             "Coefficient estimates, shape (p,).")
        .def("get_se", &PyIrlsEngine::get_se,
             "Standard errors of the coefficient estimates, shape (p,).")
        .def("get_mu", &PyIrlsEngine::get_mu,
             "Fitted means, shape (n,).")
        .def("get_eta", &PyIrlsEngine::get_eta,
             "Linear predictor, shape (n,).")
        .def("get_weights", &PyIrlsEngine::get_weights,
             "Prior observation weights held by the engine, shape (n,).")
        .def("get_w", &PyIrlsEngine::get_w,
             "Working weights of the last IRLS iteration, shape (n,).")
        .def("get_dev", &PyIrlsEngine::get_dev,
             "Deviance minimized by the engine. With Firth's penalty this is the penalized deviance (the log|X'WX| term is subtracted).")
        .def("get_std_dev", &PyIrlsEngine::get_std_dev,
             "Standard deviance sum(w_i * d_i), WITHOUT the Firth penalty.")
        .def("get_rank", &PyIrlsEngine::get_rank,
             "Numerical rank of the weighted design matrix.")
        .def("get_vcov", &PyIrlsEngine::get_vcov,
             "Covariance matrix of the coefficient estimates, shape (p, p).")
        .def("get_converged", &PyIrlsEngine::get_converged,
             "True if the last solve() met the convergence criterion.")
        .def("get_log_det_XtWX", &PyIrlsEngine::get_log_det_XtWX,
             "Log-determinant of X'WX at the current iterate.")
        .def("get_firth", &PyIrlsEngine::get_firth,
             "True if Firth's penalty is active.")
        .def("get_iterations", &PyIrlsEngine::get_iterations,
             "Number of IRLS iterations performed by the last solve().")
        .def("get_aliased", &PyIrlsEngine::get_aliased,
             "0-based indices of aliased (collinear) columns detected by the pivoted back ends.")
        .def("set_fam_params", &PyIrlsEngine::set_fam_params, nb::arg("params"),
             "Replace the family parameters (e.g. update theta between IRLS runs).");

    // Family helpers: evaluate link, variance and deviance from Python
    m.def("linkinv",
        [](int fam_code, const glm::FamilyParams &p, const Eigen::VectorXd &eta) {
            Eigen::VectorXd mu(eta.size());
            Eigen::Map<const Eigen::ArrayXd> eta_arr(eta.data(), eta.size());
            Eigen::Map<Eigen::ArrayXd>       mu_arr(mu.data(), mu.size());
            glm::linkinv(fam_code, p, eta_arr, mu_arr);
            return mu;
        }, nb::arg("fam_code"), nb::arg("fam_params"), nb::arg("eta"),
        "Inverse link: mean mu = g^-1(eta) for the given family code.\n"
        "\n"
        "Parameters: fam_code (FAM_* constant), fam_params (FamilyParams), eta (n,) linear predictor.\n"
        "Returns an (n,) array of means.");

    m.def("variance",
        [](int fam_code, const glm::FamilyParams &p, const Eigen::VectorXd &mu) {
            Eigen::VectorXd v(mu.size());
            Eigen::Map<const Eigen::ArrayXd> mu_arr(mu.data(), mu.size());
            Eigen::Map<Eigen::ArrayXd>       v_arr(v.data(), v.size());
            glm::variance(fam_code, p, mu_arr, v_arr);
            return v;
        }, nb::arg("fam_code"), nb::arg("fam_params"), nb::arg("mu"),
        "Variance function V(mu) for the given family code.\n"
        "\n"
        "Parameters: fam_code (FAM_* constant), fam_params (FamilyParams), mu (n,) means.\n"
        "Returns an (n,) array.");

    m.def("dev_resids",
        [](int fam_code, const glm::FamilyParams &p, const Eigen::VectorXd &y,
           const Eigen::VectorXd &mu, const Eigen::VectorXd &weights) {
            if (y.size() != mu.size() || y.size() != weights.size())
                throw std::invalid_argument("dev_resids: y, mu and weights must have the same size");
            Eigen::VectorXd out(y.size());
            Eigen::Map<const Eigen::ArrayXd> y_a(y.data(), y.size());
            Eigen::Map<const Eigen::ArrayXd> mu_a(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> w_a(weights.data(), weights.size());
            Eigen::Map<Eigen::ArrayXd>       out_a(out.data(), out.size());
            glm::dev_resids(fam_code, p, y_a, mu_a, w_a, out_a);
            return out;
        }, nb::arg("fam_code"), nb::arg("fam_params"), nb::arg("y"), nb::arg("mu"), nb::arg("weights"),
                "Per-observation deviance contributions w_i * d(y_i, mu_i).\n"
        "\n"
        "They are defined for every IRLS and quasi family (no log-likelihood needed).\n"
        "Deviance residuals are sign(y - mu) * sqrt(contribution).\n"
        "\n"
        "Parameters: fam_code (FAM_* constant), fam_params (FamilyParams), y, mu and\n"
        "weights (all (n,) arrays of the same size; ValueError otherwise).\n"
        "Returns an (n,) array.");

    m.def("valid_eta",
        [](int fam_code, const glm::FamilyParams &p, const Eigen::VectorXd &eta) {
            Eigen::Map<const Eigen::ArrayXd> eta_arr(eta.data(), eta.size());
            return glm::valideta(fam_code, p, eta_arr);
        }, nb::arg("fam_code"), nb::arg("fam_params"), nb::arg("eta"),
        "Check that the linear predictor eta lies in the valid domain of the family/link.\n"
        "\n"
        "Parameters: fam_code (FAM_* constant), fam_params (FamilyParams), eta (n,).\n"
        "Returns a bool.");

    m.def("valid_mu",
        [](int fam_code, const glm::FamilyParams &p, const Eigen::VectorXd &mu) {
            Eigen::Map<const Eigen::ArrayXd> mu_arr(mu.data(), mu.size());
            return glm::validmu(fam_code, p, mu_arr);
        }, nb::arg("fam_code"), nb::arg("fam_params"), nb::arg("mu"),
        "Check that the means mu lie in the valid domain of the family.\n"
        "\n"
        "Parameters: fam_code (FAM_* constant), fam_params (FamilyParams), mu (n,).\n"
        "Returns a bool.");

    // -----------------------------------------------------------------
    // Two-step drivers
    // -----------------------------------------------------------------
    nb::class_<glm::ThetaFit>(m, "ThetaFit",
        "Result of the maximum-likelihood estimation of the negative-binomial theta.")
        .def_ro("theta", &glm::ThetaFit::theta,
            "Estimated negative-binomial theta.")
        .def_ro("iters", &glm::ThetaFit::iters,
            "Number of iterations used.")
        .def_ro("converged", &glm::ThetaFit::converged,
            "True if the estimation converged.");

    nb::class_<glm::TwoStepResult>(m, "TwoStepResult",
        "Result of a two-step fit (fit_nb_two_step / fit_quasi).")
        .def_ro("beta", &glm::TwoStepResult::beta,
            "Coefficient estimates, shape (p,).")
        .def_ro("se", &glm::TwoStepResult::se,
            "Standard errors of the coefficients, shape (p,).")
        .def_ro("vcov", &glm::TwoStepResult::vcov,
            "Covariance matrix of the coefficients, shape (p, p).")
        .def_ro("theta", &glm::TwoStepResult::theta,
            "Estimated negative-binomial theta (meaningful for NB fits).")
        .def_ro("dispersion", &glm::TwoStepResult::dispersion,
            "Estimated dispersion parameter.")
        .def_ro("outer_iters", &glm::TwoStepResult::outer_iters,
            "Number of outer (alternating) iterations.")
        .def_ro("converged", &glm::TwoStepResult::converged,
            "True if the fit converged.")
        .def_ro("rank", &glm::TwoStepResult::rank,
            "Numerical rank of the weighted design matrix.")
        .def_ro("deviance", &glm::TwoStepResult::deviance,
            "Final deviance.")
        .def_ro("n_ok", &glm::TwoStepResult::n_ok,
            "Number of valid observations used in the fit.")
        .def_ro("inner_iters", &glm::TwoStepResult::inner_iters,
            "Total IRLS iterations across all passes.")
        .def_ro("aliased", &glm::TwoStepResult::aliased,
            "0-based indices of aliased (collinear) columns.")
        .def_ro("mu", &glm::TwoStepResult::mu,
            "Fitted means, shape (n,).")
        .def_ro("eta", &glm::TwoStepResult::eta,
            "Linear predictor, shape (n,).");

    m.def("fit_nb_two_step",
        [](const Eigen::MatrixXd &X, const Eigen::VectorXd &Y,
           const Eigen::VectorXd &weights, const Eigen::VectorXd &offset,
           int fam_code, const Eigen::VectorXd &beta_start,
           double theta_start, int outer_maxit, double outer_tol,
           int inner_maxit, double inner_tol, int type, bool is_big_matrix,
           double link_power, bool estimate_theta, const glm::IrlsOptions &options)
        {
            if (X.rows() != Y.size())
                throw std::invalid_argument("fit_nb_two_step: X and Y have a different number of rows");
            if (weights.size() != X.rows())
                throw std::invalid_argument("fit_nb_two_step: weights must have size nrow(X)");
            if (offset.size() != X.rows())
                throw std::invalid_argument("fit_nb_two_step: offset must have size nrow(X)");
            if (beta_start.size() != X.cols())
                throw std::invalid_argument("fit_nb_two_step: beta_start must have size p = ncol(X)");

            Eigen::Map<const Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<const Eigen::VectorXd> Ym(Y.data(), Y.size());
            Eigen::Map<const Eigen::VectorXd> Wm(weights.data(), weights.size());
            Eigen::Map<const Eigen::VectorXd> Om(offset.data(), offset.size());

            return glm::fit_nb_two_step(Xm, Ym, Wm, Om, fam_code, beta_start,
                                          theta_start, outer_maxit, outer_tol,
                                          inner_maxit, inner_tol, type, is_big_matrix,
                                          link_power, estimate_theta, options);
        },
        nb::arg("X"), nb::arg("Y"), nb::arg("weights"), nb::arg("offset"),
        nb::arg("fam_code"), nb::arg("beta_start"),
        nb::arg("theta_start") = 1.0,
        nb::arg("outer_maxit") = 25, nb::arg("outer_tol") = 1e-6,
        nb::arg("inner_maxit") = 100, nb::arg("inner_tol") = 1e-8,
        nb::arg("type") = 0, nb::arg("is_big_matrix") = false,
        nb::arg("link_power") = 1.0,
        nb::arg("estimate_theta") = true,
        nb::arg("options") = glm::IrlsOptions(),
        "Two-step negative-binomial fit: alternates an IRLS fit of beta (for fixed theta) with maximum-likelihood estimation of theta until convergence.\n"
        "\n"
        "Parameters:\n"
        "  X, Y, weights, offset  Data: (n, p) design matrix and (n,) vectors.\n"
        "  fam_code               Negative-binomial family/link code (FAM_NB_*).\n"
        "  beta_start             (p,) starting coefficients.\n"
        "  theta_start            Starting theta (default 1.0).\n"
        "  outer_maxit, outer_tol Limits for the alternating loop (defaults 25, 1e-6).\n"
        "  inner_maxit, inner_tol Limits for each IRLS run (defaults 100, 1e-8).\n"
        "  type                   Matrix decomposition (TYPE_* constant; default 0).\n"
        "  is_big_matrix          Select the code path intended for large matrices.\n"
        "  link_power             Exponent for FAM_NB_POWER (default 1.0).\n"
        "  estimate_theta         If False, theta stays fixed at theta_start.\n"
        "  options                IrlsOptions forwarded to every inner IRLS fit.\n"
        "\n"
        "Returns the fit result object. Raises ValueError on inconsistent sizes.");

    m.def("fit_quasi",
        [](const Eigen::MatrixXd &X, const Eigen::VectorXd &Y,
           const Eigen::VectorXd &weights, const Eigen::VectorXd &offset,
           int quasi_code, const Eigen::VectorXd &beta_start,
           int maxit, double tol, int type, bool is_big_matrix,
           const glm::IrlsOptions &options)
        {
            if (X.rows() != Y.size())
                throw std::invalid_argument("fit_quasi: X and Y have a different number of rows");
            if (weights.size() != X.rows())
                throw std::invalid_argument("fit_quasi: weights must have size nrow(X)");
            if (offset.size() != X.rows())
                throw std::invalid_argument("fit_quasi: offset must have size nrow(X)");
            if (beta_start.size() != X.cols())
                throw std::invalid_argument("fit_quasi: beta_start must have size p = ncol(X)");

            Eigen::Map<const Eigen::MatrixXd> Xm(X.data(), X.rows(), X.cols());
            Eigen::Map<const Eigen::VectorXd> Ym(Y.data(), Y.size());
            Eigen::Map<const Eigen::VectorXd> Wm(weights.data(), weights.size());
            Eigen::Map<const Eigen::VectorXd> Om(offset.data(), offset.size());

            return glm::fit_quasi(Xm, Ym, Wm, Om, quasi_code, beta_start, maxit, tol,
                                   type, is_big_matrix, options);
        },
        nb::arg("X"), nb::arg("Y"), nb::arg("weights"), nb::arg("offset"),
        nb::arg("quasi_code"), nb::arg("beta_start"),
        nb::arg("maxit") = 100, nb::arg("tol") = 1e-8,
        nb::arg("type") = 0, nb::arg("is_big_matrix") = false,
        nb::arg("options") = glm::IrlsOptions(),
        "Quasi-likelihood fit by IRLS for a FAM_QUASI_* family.\n"
        "\n"
        "Parameters:\n"
        "  X, Y, weights, offset  Data: (n, p) design matrix and (n,) vectors.\n"
        "  quasi_code             Quasi family/link code (FAM_QUASI_*).\n"
        "  beta_start             (p,) starting coefficients.\n"
        "  maxit, tol             Iteration limit and tolerance (defaults 100, 1e-8).\n"
        "  type                   Matrix decomposition (TYPE_* constant; default 0).\n"
        "  is_big_matrix          Select the code path intended for large matrices.\n"
        "  options                IrlsOptions (rank tolerance, chunk rows, ...).\n"
        "\n"
        "Returns the fit result object. Raises ValueError on inconsistent sizes.");

    m.def("base_family_for_quasi", &glm::base_family_for_quasi, nb::arg("quasi_code"),
            "Return the code of the base family associated with a FAM_QUASI_* code.");

    m.def("mle_theta_nb",
        [](const Eigen::VectorXd &y, const Eigen::VectorXd &mu,
           const Eigen::VectorXd &prior_w, double theta0, int maxit, double tol)
        {
            return glm::mle_theta_nb(y.array(), mu.array(), prior_w.array(),
                                       theta0, maxit, tol);
        },
        nb::arg("y"), nb::arg("mu"), nb::arg("prior_w"), nb::arg("theta0"),
        nb::arg("maxit") = 50, nb::arg("tol") = 1e-8,
        "Maximum-likelihood estimate of the negative-binomial theta for fixed means.\n"
        "\n"
        "Parameters:\n"
        "  y        (n,) counts.\n"
        "  mu       (n,) fitted means.\n"
        "  prior_w  (n,) prior weights.\n"
        "  theta0   Starting value for theta.\n"
        "  maxit    Maximum iterations (default 50).\n"
        "  tol      Convergence tolerance (default 1e-8).\n"
        "\n"
        "Returns a ThetaFit.");

    // -----------------------------------------------------------------
    // 2. Direct maximum-likelihood engine (MleSolver)
    // -----------------------------------------------------------------
    nb::enum_<glm::LinkType>(m, "LinkType",
            "Link functions available to MleSolver: Identity, Log, Sqrt, Logit, Probit, Cloglog, Cauchit, Loglog and Inverse (g(mu) = 1/mu).")
        .value("Identity", glm::LinkType::Identity)
        .value("Log",      glm::LinkType::Log)
        .value("Sqrt",     glm::LinkType::Sqrt)
        .value("Logit",    glm::LinkType::Logit)
        .value("Probit",   glm::LinkType::Probit)
        .value("Cloglog",  glm::LinkType::Cloglog)
        .value("Cauchit",  glm::LinkType::Cauchit)
        .value("Loglog",   glm::LinkType::Loglog)
        .value("Inverse",  glm::LinkType::Inverse);

    nb::enum_<glm::VarianceForm>(m, "VarianceForm",
            "Variance form of the negative-binomial component: Quadratic (V = mu + mu^2/theta, NB2) or Linear (variance proportional to mu, NB1-type). Values are also exported to the module namespace.")
        .value("Quadratic", glm::VarianceForm::Quadratic)
        .value("Linear", glm::VarianceForm::Linear)
        .export_values();

    nb::enum_<glm::Minimizer>(m, "Minimizer",
            "Optimizer used internally by MleSolver and QuasiSolver:\n"
            "  LBFGSB      Limited-memory BFGS with box constraints and analytic gradient (default; the most robust in general).\n"
            "  BFGS        Unconstrained BFGS with a full Hessian approximation.\n"
            "  NelderMead  Derivative-free simplex search (slower; useful as a fallback).\n"
            "Values are also exported to the module namespace.")
        .value("LBFGSB", glm::Minimizer::LBFGSB)
        .value("BFGS", glm::Minimizer::BFGS)
        .value("NelderMead", glm::Minimizer::NelderMead)
        .export_values();

    nb::class_<glm::MleResult>(m, "MleResult",
        "Result of a direct maximum-likelihood fit (MleSolver).")
        .def_ro("success", &glm::MleResult::success,
            "True if the optimizer reported success.")
        .def_ro("message", &glm::MleResult::message,
            "Optimizer termination message.")
        .def_ro("log_likelihood", &glm::MleResult::log_likelihood,
            "Log-likelihood at the solution.")
        .def_ro("iterations", &glm::MleResult::iterations,
            "Number of optimizer iterations.")
        .def_ro("function_evaluations", &glm::MleResult::function_evaluations,
            "Number of objective evaluations.")
        .def_ro("beta", &glm::MleResult::beta,
            "Coefficients of the main (mean) predictor.")
        .def_ro("gamma", &glm::MleResult::gamma,
            "Coefficients of the secondary predictor (Z), when the family has one.")
        .def_ro("extra_param", &glm::MleResult::extra_param,
            "Additional family-specific scalar parameter (e.g. dispersion or shape).")
        .def_ro("extra_param2", &glm::MleResult::extra_param2,
            "Second additional family-specific scalar parameter, when applicable.")
        .def_ro("vcov", &glm::MleResult::vcov,
            "Covariance matrix of the estimated parameters (if compute_se=True).")
        .def_ro("se", &glm::MleResult::se,
            "Standard errors of the estimated parameters (if compute_se=True).")
        .def("__repr__", [](const glm::MleResult &r) {
            return "<_glmsolve.MleResult success=" + std::string(r.success ? "True" : "False")
                 + " log_likelihood=" + std::to_string(r.log_likelihood)
                 + " iterations=" + std::to_string(r.iterations) + ">";
        });

    nb::class_<glm::MleSolver>(m, "MleSolver",
            "Direct maximum-likelihood solver for count, survival, zero-inflated, hurdle, truncated and related families.\n"
            "\n"
            "Methods: fit (generic families selected by FAM_* code), fit_poisson_inflated, fit_negative_binomial_inflated and fit_generalized_gamma. The optimizer is chosen with the `minimizer` argument or set_minimizer().")
        .def(nb::init<bool, glm::Minimizer>(),
             nb::arg("fit_intercept") = true,
             nb::arg("minimizer") = glm::Minimizer::LBFGSB,
             "Create a solver.\n"
             "\n"
             "Parameters:\n"
             "  fit_intercept  Whether the solver fits an intercept term (default True).\n"
             "  minimizer      Minimizer enum value (default Minimizer.LBFGSB).")
        .def("set_minimizer", &glm::MleSolver::set_minimizer, nb::arg("minimizer"),
             "Set the minimizer used by subsequent fit() / fit_*() calls.")
        .def("get_minimizer", &glm::MleSolver::get_minimizer,
             "Return the current Minimizer.")
        .def("fit",
            [](glm::MleSolver &self, int family_code,
               const Eigen::MatrixXd &X_raw, const Eigen::VectorXd &y,
               std::optional<Eigen::VectorXd> d,
               std::optional<Eigen::MatrixXd> Z,
               std::optional<double> left_limit,
               std::optional<double> right_limit,
               int maxiter, double ftol, double gtol, int maxcor, int verbose,
               bool compute_se,
               std::optional<glm::LinkType> secondary_link,
               bool use_left, bool use_right)
            {
                if (X_raw.rows() != y.size())
                    throw std::invalid_argument("MleSolver.fit: X and y have a different number of rows");
                if (d.has_value() && d->size() != y.size())
                    throw std::invalid_argument("MleSolver.fit: d (censoring indicator) must have the same size as y");
                if (Z.has_value() && Z->rows() != X_raw.rows())
                    throw std::invalid_argument("MleSolver.fit: Z must have the same number of rows as X");

                return self.fit(static_cast<glm::FamilyCode>(family_code),
                                 X_raw, y, std::move(d), std::move(Z),
                                 left_limit, right_limit,
                                 maxiter, ftol, gtol, maxcor, verbose, compute_se,
                                 secondary_link, use_left, use_right);
            },
            nb::arg("family_code"), nb::arg("X"), nb::arg("y"),
            nb::arg("d") = nb::none(),
            nb::arg("Z") = nb::none(),
            nb::arg("left_limit") = nb::none(),
            nb::arg("right_limit") = nb::none(),
            nb::arg("maxiter") = 1000,
            nb::arg("ftol") = 1e-12,
            nb::arg("gtol") = 1e-6,
            nb::arg("maxcor") = 10,
            nb::arg("verbose") = 0,
            nb::arg("compute_se") = true,
            nb::arg("secondary_link") = nb::none(),
            nb::arg("use_left") = false,
            nb::arg("use_right") = false,
            "Fit a family selected by its FAM_* code by direct maximum likelihood.\n"
            "\n"
            "Parameters:\n"
            "  family_code     FAM_* constant selecting the family.\n"
            "  X               (n, p) design matrix of the main predictor.\n"
            "  y               (n,) response.\n"
            "  d               Optional (n,) censoring / event indicator for families that support it.\n"
            "  Z               Optional (n, q) design matrix of the secondary predictor.\n"
            "  left_limit      Optional lower limit (e.g. truncation bound); used only if use_left is True.\n"
            "  right_limit     Optional upper limit; used only if use_right is True.\n"
            "  maxiter         Maximum optimizer iterations (default 1000).\n"
            "  ftol            Function-value tolerance (default 1e-12).\n"
            "  gtol            Gradient tolerance (default 1e-6).\n"
            "  maxcor          Number of stored correction pairs (L-BFGS memory; default 10).\n"
            "  verbose         Verbosity level (default 0).\n"
            "  compute_se      Compute vcov and standard errors (default True).\n"
            "  secondary_link  LinkType of the parameter modeled by the secondary predictor Z.\n"
            "                  Families with a secondary predictor: ZIP/ZINB and Hurdle* (probability:\n"
            "                  logit/probit/cloglog/loglog/cauchit), Beta (phi) and Simplex (sigma^2)\n"
            "                  (positive: log/identity/sqrt/inverse). None uses the family default;\n"
            "                  families without a secondary predictor ignore it; an invalid link\n"
            "                  raises ValueError.\n"
            "  use_left        Apply left_limit (default False).\n"
            "  use_right       Apply right_limit (default False).\n"
            "\n"
            "Returns an MleResult. Raises ValueError if sizes are inconsistent.")

        .def("fit_poisson_inflated",
            [](glm::MleSolver &self,
               const Eigen::MatrixXd &X_raw, const Eigen::VectorXd &y,
               double target_value,
               glm::LinkType base_link, glm::LinkType infl_link,
               int maxiter, double ftol, double gtol, int maxcor, int verbose,
               bool compute_se,
               std::optional<Eigen::MatrixXd> Z)
            {
                if (X_raw.rows() != y.size())
                    throw std::invalid_argument("MleSolver.fit_poisson_inflated: X and y have a different number of rows");
                if (Z.has_value() && Z->rows() != X_raw.rows())
                    throw std::invalid_argument("MleSolver.fit_poisson_inflated: Z must have the same number of rows as X");
                return self.fit_poisson_inflated(X_raw, y, target_value,
                                                  base_link, infl_link,
                                                  maxiter, ftol, gtol, maxcor, verbose,
                                                  compute_se, std::move(Z));
            },
            nb::arg("X"), nb::arg("y"), nb::arg("target_value"),
            nb::arg("base_link") = glm::LinkType::Log,
            nb::arg("infl_link") = glm::LinkType::Logit,
            nb::arg("maxiter") = 1000,
            nb::arg("ftol") = 1e-12,
            nb::arg("gtol") = 1e-6,
            nb::arg("maxcor") = 10,
            nb::arg("verbose") = 0,
            nb::arg("compute_se") = true,
            nb::arg("Z") = nb::none(),
            "Fit a Poisson model inflated at `target_value` by direct maximum likelihood.\n"
            "\n"
            "Parameters:\n"
            "  X            (n, p) design matrix of the count (base) predictor.\n"
            "  y            (n,) counts.\n"
            "  target_value Value that is inflated (0 gives the zero-inflated model).\n"
            "  base_link    LinkType of the count mean (default Log).\n"
            "  infl_link    LinkType of the inflation probability (default Logit).\n"
            "  maxiter, ftol, gtol, maxcor, verbose, compute_se  As in fit().\n"
            "  Z            Optional (n, q) design matrix of the inflation predictor.\n"
            "\n"
            "Returns an MleResult. Raises ValueError if sizes are inconsistent.")

        .def("fit_negative_binomial_inflated",
            [](glm::MleSolver &self,
            const Eigen::MatrixXd &X_raw,
            const Eigen::VectorXd &y,
            double target_value,
            glm::VarianceForm var_form,
            glm::LinkType base_link,
            glm::LinkType infl_link,
            int maxiter,
            double ftol,
            double gtol,
            int maxcor,
            int verbose,
            bool compute_se,
            std::optional<Eigen::MatrixXd> Z)
            {
                if (X_raw.rows() != y.size())
                    throw std::invalid_argument("MleSolver.fit_negative_binomial_inflated: X and y have a different number of rows");
                if (Z.has_value() && Z->rows() != X_raw.rows())
                    throw std::invalid_argument("MleSolver.fit_negative_binomial_inflated: Z must have the same number of rows as X");

                return self.fit_negative_binomial_inflated(
                    X_raw, y, target_value,
                    var_form, base_link, infl_link,
                    maxiter, ftol, gtol, maxcor, verbose,
                    compute_se, std::move(Z));
            },
            nb::arg("X"), nb::arg("y"), nb::arg("target_value"),
            nb::arg("var_form") = glm::VarianceForm::Quadratic,
            nb::arg("base_link") = glm::LinkType::Log,
            nb::arg("infl_link") = glm::LinkType::Logit,
            nb::arg("maxiter") = 1000,
            nb::arg("ftol") = 1e-12,
            nb::arg("gtol") = 1e-6,
            nb::arg("maxcor") = 10,
            nb::arg("verbose") = 0,
            nb::arg("compute_se") = true,
            nb::arg("Z") = nb::none(),
            "Fit a negative-binomial model inflated at `target_value` by direct maximum likelihood.\n"
            "\n"
            "Parameters (same as fit_poisson_inflated, plus):\n"
            "  X            (n, p) design matrix of the count (base) predictor.\n"
            "  y            (n,) counts.\n"
            "  target_value Value that is inflated (0 gives the zero-inflated model).\n"
            "  base_link    LinkType of the count mean (default Log).\n"
            "  infl_link    LinkType of the inflation probability (default Logit).\n"
            "  maxiter, ftol, gtol, maxcor, verbose, compute_se  As in fit().\n"
            "  Z            Optional (n, q) design matrix of the inflation predictor.\n"
            "  var_form     VarianceForm of the NB component (default Quadratic).\n"
            "\n"
            "Returns an MleResult. Raises ValueError if sizes are inconsistent.")

        .def("fit_generalized_gamma",
            [](glm::MleSolver &self,
               const Eigen::MatrixXd &X_raw, const Eigen::MatrixXd &Y,
               const Eigen::VectorXi &mx_mu_cols, const Eigen::VectorXi &mx_sigma_cols,
               const Eigen::VectorXi &mx_Q_cols,
               std::optional<Eigen::VectorXd> weights,
               glm::LinkType link_mu, glm::LinkType link_sigma, glm::LinkType link_Q,
               int maxiter, double ftol, double gtol, int maxcor, int verbose,
               bool compute_se)
            {
                if (X_raw.rows() != Y.rows())
                    throw std::invalid_argument("MleSolver.fit_generalized_gamma: X and Y have a different number of rows");
                if (Y.cols() < 4)
                    throw std::invalid_argument("MleSolver.fit_generalized_gamma: Y must have columns [start, time1, (unused), status]");
                if (weights.has_value() && weights->size() != X_raw.rows())
                    throw std::invalid_argument("MleSolver.fit_generalized_gamma: weights must have size nrow(X)");
                return self.fit_generalized_gamma(X_raw, Y, mx_mu_cols, mx_sigma_cols, mx_Q_cols,
                                                   weights, link_mu, link_sigma, link_Q,
                                                   maxiter, ftol, gtol, maxcor, verbose,
                                                   compute_se);
            },
            nb::arg("X"), nb::arg("Y"),
            nb::arg("mx_mu_cols"), nb::arg("mx_sigma_cols"), nb::arg("mx_Q_cols"),
            nb::arg("weights") = nb::none(),
            nb::arg("link_mu") = glm::LinkType::Identity,
            nb::arg("link_sigma") = glm::LinkType::Log,
            nb::arg("link_Q") = glm::LinkType::Identity,
            nb::arg("maxiter") = 1000,
            nb::arg("ftol") = 1e-12,
            nb::arg("gtol") = 1e-6,
            nb::arg("maxcor") = 10,
            nb::arg("verbose") = 0,
            nb::arg("compute_se") = true,
            "Fit a generalized gamma survival model by direct maximum likelihood.\n"
            "\n"
            "Parameters:\n"
            "  X               (n, p) design matrix shared by the three predictors.\n"
            "  Y               (n, >=4) survival matrix with columns [start, time1, (unused), status].\n"
            "  mx_mu_cols      Integer indices of the columns of X used by the mu predictor.\n"
            "  mx_sigma_cols   Integer indices of the columns of X used by the sigma predictor.\n"
            "  mx_Q_cols       Integer indices of the columns of X used by the Q (shape) predictor.\n"
            "  weights         Optional (n,) observation weights.\n"
            "  link_mu         LinkType for mu (default Identity).\n"
            "  link_sigma      LinkType for sigma (default Log).\n"
            "  link_Q          LinkType for Q (default Identity).\n"
            "  maxiter, ftol, gtol, maxcor, verbose, compute_se  As in fit().\n"
            "\n"
            "Returns an MleResult. Raises ValueError if sizes are inconsistent.");

    // -----------------------------------------------------------------
    // 3. Quasi-likelihood engine (QuasiSolver)
    // -----------------------------------------------------------------
    nb::enum_<glm::QuasiLinkType>(m, "QuasiLinkType",
            "Link functions available to QuasiSolver.fit_custom: Identity, Log, Inverse, Logit and Probit.")
        .value("Identity", glm::QuasiLinkType::Identity)
        .value("Log",      glm::QuasiLinkType::Log)
        .value("Inverse",  glm::QuasiLinkType::Inverse)
        .value("Logit",    glm::QuasiLinkType::Logit)
        .value("Probit",   glm::QuasiLinkType::Probit);

    nb::class_<glm::QuasiResult>(m, "QuasiResult",
        "Result of a quasi-likelihood fit (QuasiSolver).")
        .def_ro("success", &glm::QuasiResult::success,
            "True if the optimizer reported success.")
        .def_ro("message", &glm::QuasiResult::message,
            "Optimizer termination message.")
        .def_ro("quasi_deviance", &glm::QuasiResult::quasi_deviance,
            "Quasi-deviance at the solution.")
        .def_ro("dispersion", &glm::QuasiResult::dispersion,
            "Estimated dispersion parameter.")
        .def_ro("iterations", &glm::QuasiResult::iterations,
            "Number of optimizer iterations.")
        .def_ro("function_evaluations", &glm::QuasiResult::function_evaluations,
            "Number of objective evaluations.")
        .def_ro("beta", &glm::QuasiResult::beta,
            "Coefficient estimates.")
        .def_ro("vcov", &glm::QuasiResult::vcov,
            "Covariance matrix of the coefficients.")
        .def_ro("se", &glm::QuasiResult::se,
            "Standard errors of the coefficients.")
        .def("__repr__", [](const glm::QuasiResult &r) {
            return "<_glmsolve.QuasiResult success=" + std::string(r.success ? "True" : "False")
                 + " dispersion=" + std::to_string(r.dispersion)
                 + " iterations=" + std::to_string(r.iterations) + ">";
        });

    nb::class_<glm::QuasiSolver>(m, "QuasiSolver",
            "Quasi-likelihood solver.\n"
            "\n"
            "Methods: fit (built-in FAM_QUASI_* families) and fit_custom (user-defined variance function). Optimization uses the selected Minimizer.")
        .def(nb::init<bool, glm::Minimizer>(),
             nb::arg("fit_intercept") = true,
             nb::arg("minimizer") = glm::Minimizer::LBFGSB,
             "Create a solver.\n"
             "\n"
             "Parameters:\n"
             "  fit_intercept  Whether the solver fits an intercept term (default True).\n"
             "  minimizer      Minimizer enum value (default Minimizer.LBFGSB).")
        .def("set_minimizer", &glm::QuasiSolver::set_minimizer, nb::arg("minimizer"),
             "Set the minimizer used by subsequent fit() / fit_custom() calls.")
        .def("get_minimizer", &glm::QuasiSolver::get_minimizer,
             "Return the current Minimizer.")
        .def("fit",
            [](glm::QuasiSolver &self, int family_code,
               const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
               std::optional<Eigen::VectorXd> weights,
               int maxiter, double ftol, double gtol, int maxcor, int verbose,
               std::optional<Eigen::VectorXd> offset)
            {
                if (X.rows() != y.size())
                    throw std::invalid_argument("QuasiSolver.fit: X and y have a different number of rows");
                if (weights.has_value() && weights->size() != y.size())
                    throw std::invalid_argument("QuasiSolver.fit: weights must have the same size as y");
                if (offset.has_value() && offset->size() != y.size())
                    throw std::invalid_argument("QuasiSolver.fit: offset must have the same size as y");

                return self.fit(static_cast<glm::FamilyCode>(family_code), X, y,
                                 std::move(weights), maxiter, ftol, gtol, maxcor, verbose,
                                 std::move(offset));
            },
            nb::arg("family_code"), nb::arg("X"), nb::arg("y"),
            nb::arg("weights") = nb::none(),
            nb::arg("maxiter") = 1000,
            nb::arg("ftol") = 1e-12,
            nb::arg("gtol") = 1e-6,
            nb::arg("maxcor") = 10,
            nb::arg("verbose") = 0,
            nb::arg("offset") = nb::none(),
            "Fit a built-in quasi-likelihood family.\n"
            "\n"
            "Parameters:\n"
            "  family_code  FAM_QUASI_* constant.\n"
            "  X            (n, p) design matrix.\n"
            "  y            (n,) response.\n"
            "  weights      Optional (n,) prior weights.\n"
            "  maxiter, ftol, gtol, maxcor, verbose  Optimizer controls (defaults 1000, 1e-12, 1e-6, 10, 0).\n"
            "  offset       Optional (n,) offset.\n"
            "\n"
            "Returns a QuasiResult. Raises ValueError if sizes are inconsistent.")
        .def("fit_custom",
            [](glm::QuasiSolver &self,
               const Eigen::MatrixXd &X, const Eigen::VectorXd &y,
               nb::callable variance_fn,
               std::optional<Eigen::VectorXd> weights,
               glm::QuasiLinkType link,
               int maxiter, double ftol, double gtol,
               std::optional<Eigen::VectorXd> offset)
            {
                if (X.rows() != y.size())
                    throw std::invalid_argument("QuasiSolver.fit_custom: X and y have a different number of rows");
                if (weights.has_value() && weights->size() != y.size())
                    throw std::invalid_argument("QuasiSolver.fit_custom: weights must have the same size as y");
                if (offset.has_value() && offset->size() != y.size())
                    throw std::invalid_argument("QuasiSolver.fit_custom: offset must have the same size as y");

                auto cpp_var_fn = [variance_fn](double mu) -> double {
                    return nb::cast<double>(variance_fn(mu));
                };

                return self.fit_custom(X, y, cpp_var_fn, std::move(weights),
                                        link, maxiter, ftol, gtol, std::move(offset));
            },
            nb::arg("X"), nb::arg("y"), nb::arg("variance_fn"),
            nb::arg("weights") = nb::none(),
            nb::arg("link") = glm::QuasiLinkType::Log,
            nb::arg("maxiter") = 1000,
            nb::arg("ftol") = 1e-12,
            nb::arg("gtol") = 1e-6,
            nb::arg("offset") = nb::none(),
            "Fit a quasi-likelihood model with a user-defined variance function.\n"
            "\n"
            "Parameters:\n"
            "  X            (n, p) design matrix.\n"
            "  y            (n,) response.\n"
            "  variance_fn  Python callable V(mu: float) -> float, the variance function.\n"
            "  weights      Optional (n,) prior weights.\n"
            "  link         QuasiLinkType (default Log).\n"
            "  maxiter, ftol, gtol  Optimizer controls (defaults 1000, 1e-12, 1e-6).\n"
            "  offset       Optional (n,) offset.\n"
            "\n"
            "Returns a QuasiResult. Raises ValueError if sizes are inconsistent.");
}
