// nanobind layer of mixedfit: exposes models, specifications and results with NumPy
// arrays. Design-matrix construction (formulas) is done in Python
// (mixedfit/formula.py); this layer only performs the fit.
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "../include/nlmefit/crossed_fit.hpp"
#include "../include/nlmefit/glmm_fit.hpp"
#include "../include/nlmefit/gls_fit.hpp"
#include "../include/nlmefit/lm_list.hpp"
#include "../include/nlmefit/selfstart.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mixedfit;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace {

std::vector<int> to_ivec(const VectorXi& v) { return std::vector<int>(v.data(), v.data() + v.size()); }
VectorXd opt_vec(const std::optional<VectorXd>& v) { return v ? *v : VectorXd(); }
MatrixXd opt_mat(const std::optional<MatrixXd>& m) { return m ? *m : MatrixXd(); }

// ---- structures defined from Python ----
CorSpec make_custom_cor(const std::string& name, int n_params, nb::callable fn, const std::optional<VectorXd>& start, nb::object deriv) {
    CustomCor c;
    c.name = name;
    c.n_params = n_params;
    c.start = opt_vec(start);
    c.matrix = [fn](const VectorXd& th, const VectorXd& t, const MatrixXd& coords) -> MatrixXd {
        nb::object r = fn(nb::cast(th, nb::rv_policy::copy), nb::cast(t, nb::rv_policy::copy), nb::cast(coords, nb::rv_policy::copy));
        return nb::cast<MatrixXd>(r);
    };
    if (!deriv.is_none())
        c.deriv = [deriv](const VectorXd& th, const VectorXd& t, const MatrixXd& coords, int j) -> MatrixXd {
            return nb::cast<MatrixXd>(deriv(nb::cast(th, nb::rv_policy::copy), nb::cast(t, nb::rv_policy::copy), nb::cast(coords, nb::rv_policy::copy), j));
        };
    return CorSpec::Custom(std::move(c));
}

VarSpec make_custom_var(const std::string& name, int n_params, bool needs_covariate, nb::callable fn, const std::optional<VectorXd>& start, nb::object dscales) {
    CustomVar c;
    c.name = name;
    c.n_params = n_params;
    c.needs_covariate = needs_covariate;
    c.start = opt_vec(start);
    c.scales = [fn](const VectorXd& th, const VectorXd& cov, const std::vector<int>& strata, int n) -> VectorXd {
        nb::object r = fn(nb::cast(th, nb::rv_policy::copy), nb::cast(cov, nb::rv_policy::copy), nb::cast(strata), n);
        return nb::cast<VectorXd>(r);
    };
    if (!dscales.is_none())
        c.dscales = [dscales](const VectorXd& th, const VectorXd& cov, const std::vector<int>& strata, int n, int j) -> VectorXd {
            return nb::cast<VectorXd>(dscales(nb::cast(th, nb::rv_policy::copy), nb::cast(cov, nb::rv_policy::copy), nb::cast(strata), n, j));
        };
    return VarSpec::Custom(std::move(c));
}

PdSpec make_custom_pd(const std::string& name, nb::callable nparams, nb::callable factor, nb::object start, nb::object from_D, nb::object dfactor) {
    CustomPd c;
    c.name = name;
    c.nparams = [nparams](int q) { return nb::cast<int>(nparams(q)); };
    c.factor = [factor](int q, const VectorXd& th) { return nb::cast<MatrixXd>(factor(q, nb::cast(th, nb::rv_policy::copy))); };
    if (!start.is_none()) c.start = [start](int q) { return nb::cast<VectorXd>(start(q)); };
    if (!from_D.is_none()) c.from_D = [from_D](int q, const MatrixXd& D) { return nb::cast<VectorXd>(from_D(q, nb::cast(D, nb::rv_policy::copy))); };
    if (!dfactor.is_none()) c.dfactor = [dfactor](int q, const VectorXd& th, int j) { return nb::cast<MatrixXd>(dfactor(q, nb::cast(th, nb::rv_policy::copy), j)); };
    return PdSpec::Custom(std::move(c));
}

// ---- NLME model function: Python callable or native selfStart ----
struct SelfStartFns { NlmeFn f; NlmeJacFn jac; std::string name; };

SelfStart ss_by_name(const std::string& n) {
    if (n == "logis") return ss_logis();
    if (n == "fpl") return ss_fpl();
    if (n == "asymp") return ss_asymp();
    if (n == "asymp_off") return ss_asymp_off();
    if (n == "asymp_orig") return ss_asymp_orig();
    if (n == "micmen") return ss_micmen();
    if (n == "gompertz") return ss_gompertz();
    if (n == "weibull") return ss_weibull();
    throw std::invalid_argument("unknown selfStart: " + n);
}

NlmeFn wrap_f(nb::callable fn) {
    return [fn](const MatrixXd& Phi, int g) -> VectorXd {
        return nb::cast<VectorXd>(fn(nb::cast(Phi, nb::rv_policy::copy), g));
    };
}
NlmeJacFn wrap_jac(nb::callable fn) {
    return [fn](const MatrixXd& Phi, int g) -> MatrixXd {
        return nb::cast<MatrixXd>(fn(nb::cast(Phi, nb::rv_policy::copy), g));
    };
}

struct FnPair { NlmeFn f; NlmeJacFn jac; bool python = false; };
FnPair resolve_fn(nb::object f, nb::object jac) {
    FnPair p;
    if (nb::isinstance<SelfStartFns>(f)) {
        const auto& s = nb::cast<const SelfStartFns&>(f);
        p.f = s.f;
        p.jac = s.jac;
        return p;
    }
    if (!PyCallable_Check(f.ptr())) throw std::invalid_argument("f must be callable(Phi, group) or a SelfStartFns");
    p.python = true;
    p.f = wrap_f(nb::borrow<nb::callable>(f));
    if (!jac.is_none()) p.jac = wrap_jac(nb::borrow<nb::callable>(jac));
    return p;
}

// ---- group construction ----
template <class G>
void set_slots(std::vector<G>& groups, const std::vector<VectorXi>& strata_slots, const std::vector<VectorXd>& cov_slots) {
    for (auto& g : groups) {
        g.vd.strata.resize(strata_slots.size());
        g.vd.cov.resize(cov_slots.size());
        for (size_t s = 0; s < strata_slots.size(); ++s) {
            g.vd.strata[s].resize(g.rows.size());
            for (size_t k = 0; k < g.rows.size(); ++k) g.vd.strata[s][k] = strata_slots[s][g.rows[k]];
        }
        for (size_t s = 0; s < cov_slots.size(); ++s) g.vd.cov[s] = cov_slots[s](g.rows);
    }
}

}  // namespace

NB_MODULE(_nlmefit, m) {
    m.doc() = "nlmefit: nonlinear mixed-effects models (C++/Eigen) for Python";

    nb::enum_<PdKind>(m, "PdKind").value("Cholesky", PdKind::Cholesky).value("LogCholesky", PdKind::LogCholesky)
        .value("MatrixLog", PdKind::MatrixLog).value("Natural", PdKind::Natural).value("CompoundSymmetry", PdKind::CompoundSymmetry)
        .value("Diag", PdKind::Diag).value("Ident", PdKind::Ident);
    nb::enum_<CorKind>(m, "CorKind").value("Identity", CorKind::Identity).value("AR1", CorKind::AR1).value("CAR1", CorKind::CAR1)
        .value("CompoundSymmetry", CorKind::CompoundSymmetry).value("ARMA", CorKind::ARMA).value("Symm", CorKind::Symm)
        .value("Exp", CorKind::Exp).value("Gaus", CorKind::Gaus).value("Lin", CorKind::Lin).value("Ratio", CorKind::Ratio)
        .value("Spher", CorKind::Spher).value("Custom", CorKind::Custom);
    nb::enum_<SpatialMetric>(m, "SpatialMetric").value("Euclidean", SpatialMetric::Euclidean).value("Maximum", SpatialMetric::Maximum)
        .value("Manhattan", SpatialMetric::Manhattan);
    nb::enum_<VarKind>(m, "VarKind").value("None_", VarKind::None).value("Fixed", VarKind::Fixed).value("Ident", VarKind::Ident)
        .value("Power", VarKind::Power).value("Exp", VarKind::Exp).value("ConstPower", VarKind::ConstPower)
        .value("ConstProp", VarKind::ConstProp).value("Comb", VarKind::Comb).value("Custom", VarKind::Custom);
    nb::enum_<Optimizer>(m, "Optimizer").value("Internal", Optimizer::Internal).value("TrustRegion", Optimizer::TrustRegion)
        .value("LBFGSB", Optimizer::LBFGSB).value("ScipyBFGS", Optimizer::ScipyBFGS);

    nb::class_<BfgsOptions>(m, "BfgsOptions").def(nb::init<>())
        .def_rw("max_iter", &BfgsOptions::max_iter).def_rw("gtol", &BfgsOptions::gtol).def_rw("ftol", &BfgsOptions::ftol)
        .def_rw("c1", &BfgsOptions::c1).def_rw("c2", &BfgsOptions::c2).def_rw("max_step", &BfgsOptions::max_step)
        .def_rw("fd_step", &BfgsOptions::fd_step);
    nb::class_<BfgsResult>(m, "BfgsResult").def_ro("x", &BfgsResult::x).def_ro("grad", &BfgsResult::grad).def_ro("f", &BfgsResult::f)
        .def_ro("iterations", &BfgsResult::iterations).def_ro("f_evals", &BfgsResult::f_evals).def_ro("g_evals", &BfgsResult::g_evals)
        .def_ro("converged", &BfgsResult::converged).def_ro("message", &BfgsResult::message);

    nb::class_<CorSpec>(m, "CorSpec").def(nb::init<>())
        .def_rw("kind", &CorSpec::kind).def_rw("p", &CorSpec::p).def_rw("q", &CorSpec::q).def_rw("nugget", &CorSpec::nugget)
        .def_rw("metric", &CorSpec::metric).def_rw("range_start", &CorSpec::range_start);
    nb::class_<VarSpec>(m, "VarSpec").def(nb::init<>())
        .def_rw("kind", &VarSpec::kind).def_rw("by_stratum", &VarSpec::by_stratum).def_rw("fitted", &VarSpec::fitted)
        .def_rw("strata_slot", &VarSpec::strata_slot).def_rw("cov_slot", &VarSpec::cov_slot).def_rw("comb", &VarSpec::comb)
        .def_static("combine", &VarSpec::Combine);
    nb::class_<PdSpec>(m, "PdSpec").def(nb::init<>()).def(nb::init<PdKind>())
        .def_rw("kind", &PdSpec::kind).def_rw("block_sizes", &PdSpec::block_sizes).def_rw("block_kinds", &PdSpec::block_kinds)
        .def_static("blocked", &PdSpec::Blocked);
    nb::implicitly_convertible<PdKind, PdSpec>();
    m.def("make_custom_cor", &make_custom_cor, "name"_a, "n_params"_a, "matrix"_a, "start"_a = nb::none(), "deriv"_a.none() = nb::none());
    m.def("make_custom_var", &make_custom_var, "name"_a, "n_params"_a, "needs_covariate"_a, "scales"_a, "start"_a = nb::none(), "dscales"_a.none() = nb::none());
    m.def("make_custom_pd", &make_custom_pd, "name"_a, "nparams"_a, "factor"_a, "start"_a.none() = nb::none(), "from_D"_a.none() = nb::none(), "dfactor"_a.none() = nb::none());
    nb::class_<FixedTheta>(m, "FixedTheta").def(nb::init<>()).def_rw("label", &FixedTheta::label).def_rw("value", &FixedTheta::value)
        .def_rw("index", &FixedTheta::index);

    nb::class_<LmeSpec>(m, "LmeSpec").def(nb::init<>())
        .def_rw("re_kind", &LmeSpec::re_kind).def_rw("inner_kinds", &LmeSpec::inner_kinds).def_rw("pd_specs", &LmeSpec::pd_specs)
        .def_rw("cor", &LmeSpec::cor).def_rw("var", &LmeSpec::var).def_rw("reml", &LmeSpec::reml)
        .def_rw("reml_constant", &LmeSpec::reml_constant).def_rw("analytic_grad", &LmeSpec::analytic_grad)
        .def_rw("var_max_ratio", &LmeSpec::var_max_ratio).def_rw("var_shape_bound", &LmeSpec::var_shape_bound)
        .def_rw("diagnose", &LmeSpec::diagnose).def_rw("em_iter", &LmeSpec::em_iter).def_rw("fixed", &LmeSpec::fixed)
        .def_rw("sigma_fixed", &LmeSpec::sigma_fixed).def_rw("optimizer", &LmeSpec::optimizer)
        .def_rw("theta_bound", &LmeSpec::theta_bound).def_rw("opt", &LmeSpec::opt);
    nb::class_<GlsSpec>(m, "GlsSpec").def(nb::init<>())
        .def_rw("cor", &GlsSpec::cor).def_rw("var", &GlsSpec::var).def_rw("reml", &GlsSpec::reml)
        .def_rw("reml_constant", &GlsSpec::reml_constant).def_rw("fixed", &GlsSpec::fixed).def_rw("sigma_fixed", &GlsSpec::sigma_fixed)
        .def_rw("optimizer", &GlsSpec::optimizer).def_rw("theta_bound", &GlsSpec::theta_bound)
        .def_rw("analytic_grad", &GlsSpec::analytic_grad).def_rw("diagnose", &GlsSpec::diagnose)
        .def_rw("var_max_ratio", &GlsSpec::var_max_ratio).def_rw("var_shape_bound", &GlsSpec::var_shape_bound).def_rw("opt", &GlsSpec::opt);
    nb::class_<NlmeSpec>(m, "NlmeSpec").def(nb::init<>())
        .def_rw("re_kind", &NlmeSpec::re_kind).def_rw("inner_kinds", &NlmeSpec::inner_kinds).def_rw("pd_specs", &NlmeSpec::pd_specs)
        .def_rw("cor", &NlmeSpec::cor).def_rw("var", &NlmeSpec::var).def_rw("var_use_fitted", &NlmeSpec::var_use_fitted)
        .def_rw("sigma_fixed", &NlmeSpec::sigma_fixed).def_rw("parallel", &NlmeSpec::parallel).def_rw("reml", &NlmeSpec::reml)
        .def_rw("opt", &NlmeSpec::opt).def_rw("analytic_grad", &NlmeSpec::analytic_grad).def_rw("optimizer", &NlmeSpec::optimizer)
        .def_rw("theta_bound", &NlmeSpec::theta_bound).def_rw("max_outer", &NlmeSpec::max_outer).def_rw("max_pnls", &NlmeSpec::max_pnls)
        .def_rw("tol", &NlmeSpec::tol).def_rw("pnls_tol", &NlmeSpec::pnls_tol);
    nb::class_<GnlsSpec>(m, "GnlsSpec").def(nb::init<>())
        .def_rw("cor", &GnlsSpec::cor).def_rw("var", &GnlsSpec::var).def_rw("var_use_fitted", &GnlsSpec::var_use_fitted)
        .def_rw("reml", &GnlsSpec::reml).def_rw("sigma_fixed", &GnlsSpec::sigma_fixed).def_rw("analytic_grad", &GnlsSpec::analytic_grad)
        .def_rw("optimizer", &GnlsSpec::optimizer).def_rw("theta_bound", &GnlsSpec::theta_bound).def_rw("max_outer", &GnlsSpec::max_outer)
        .def_rw("max_pnls", &GnlsSpec::max_pnls).def_rw("tol", &GnlsSpec::tol).def_rw("pnls_tol", &GnlsSpec::pnls_tol)
        .def_rw("opt", &GnlsSpec::opt);

    nb::class_<LmeIdent>(m, "LmeIdent").def_ro("eigenvalues", &LmeIdent::eigenvalues).def_ro("n_weak", &LmeIdent::n_weak)
        .def_ro("saddle", &LmeIdent::saddle).def_ro("evaluated", &LmeIdent::evaluated).def_ro("message", &LmeIdent::message);
    nb::class_<LmeDetails>(m, "LmeDetails")
        .def_ro("beta", &LmeDetails::beta).def_ro("cov_unscaled", &LmeDetails::cov_unscaled)
        .def_ro("rss", &LmeDetails::rss).def_ro("logdetW", &LmeDetails::logdetW)
        .def_ro("var_ratio", &LmeDetails::var_ratio).def_ro("rank", &LmeDetails::rank);
    nb::class_<VarcompEntry>(m, "VarcompEntry").def_ro("name", &VarcompEntry::name).def_ro("est", &VarcompEntry::est).def_ro("se", &VarcompEntry::se);
    nb::class_<LmeResult>(m, "LmeResult")
        .def_ro("beta", &LmeResult::beta).def_ro("se", &LmeResult::se).def_ro("sigma2", &LmeResult::sigma2)
        .def_ro("theta", &LmeResult::theta).def_ro("theta_cor", &LmeResult::theta_cor).def_ro("theta_var", &LmeResult::theta_var)
        .def_ro("theta_re", &LmeResult::theta_re).def_ro("D_levels", &LmeResult::D_levels).def_ro("Sigma_levels", &LmeResult::Sigma_levels)
        .def_ro("D", &LmeResult::D).def_ro("Sigma_b", &LmeResult::Sigma_b).def_ro("loglik", &LmeResult::loglik)
        .def_ro("reml", &LmeResult::reml).def_ro("rank", &LmeResult::rank).def_ro("N", &LmeResult::N).def_ro("ident", &LmeResult::ident)
        .def_ro("vcov_beta", &LmeResult::vcov_beta).def_ro("vcov_theta", &LmeResult::vcov_theta)
        .def_ro("natural_names", &LmeResult::natural_names).def_ro("theta_natural", &LmeResult::theta_natural)
        .def_ro("vcov_natural", &LmeResult::vcov_natural).def_ro("varcomp", &LmeResult::varcomp)
        .def_ro("var_iterations", &LmeResult::var_iterations).def_ro("warning", &LmeResult::warning).def_ro("labels", &LmeResult::labels)
        .def_ro("blup", &LmeResult::blup).def_ro("blup_inner", &LmeResult::blup_inner).def_ro("fitted_pop", &LmeResult::fitted_pop)
        .def_ro("fitted_cond", &LmeResult::fitted_cond).def_ro("resid", &LmeResult::resid).def_ro("order", &LmeResult::order)
        .def_ro("opt", &LmeResult::opt);
    nb::class_<LmeModel>(m, "LmeModel")
        .def("profile", [](const LmeModel& self, const VectorXd& theta) {
            LmeDetails d;
            self.neg2ll(theta, &d);
            return d;
        }, "theta"_a)
        .def("neg2ll", [](const LmeModel& self, const VectorXd& theta) { return self.neg2ll(theta); }, "theta"_a)
        .def("theta_labels", &LmeModel::theta_labels)
        .def("free_theta", &LmeModel::free_theta)
        .def("sigma_is_fixed", &LmeModel::sigma_is_fixed)
        .def("n_theta", &LmeModel::n_theta)
        .def("n_obs", &LmeModel::n_obs);
    nb::class_<NlmeResult>(m, "NlmeResult")
        .def_ro("beta", &NlmeResult::beta).def_ro("se", &NlmeResult::se).def_ro("vcov_beta", &NlmeResult::vcov_beta).def_ro("vcov_theta", &NlmeResult::vcov_theta).def_ro("beta_pnls", &NlmeResult::beta_pnls)
        .def_ro("beta_lme", &NlmeResult::beta_lme).def_ro("sigma2", &NlmeResult::sigma2).def_ro("theta", &NlmeResult::theta)
        .def_ro("D_levels", &NlmeResult::D_levels).def_ro("Sigma_levels", &NlmeResult::Sigma_levels).def_ro("D", &NlmeResult::D)
        .def_ro("Sigma_b", &NlmeResult::Sigma_b).def_ro("b", &NlmeResult::b).def_ro("loglik", &NlmeResult::loglik)
        .def_ro("converged", &NlmeResult::converged).def_ro("outer_iterations", &NlmeResult::outer_iterations)
        .def_ro("warning", &NlmeResult::warning).def_ro("fitted_pop", &NlmeResult::fitted_pop).def_ro("fitted_cond", &NlmeResult::fitted_cond)
        .def_ro("order", &NlmeResult::order).def_prop_ro("used", [](const NlmeResult& r) { return std::vector<int>(r.used.begin(), r.used.end()); }).def_ro("n_used", &NlmeResult::n_used)
        .def_ro("X_lin", &NlmeResult::X_lin);
    nb::class_<NlmeModel>(m, "NlmeModel")
        .def("linearized_lme", &NlmeModel::linearized_lme, "beta"_a, "b"_a);
    nb::class_<GnlsModel>(m, "GnlsModel")
        .def("model", [](const GnlsModel& self) { return self.model(); });

    // ---------------- selfStart ----------------
    nb::class_<SelfStartFns>(m, "SelfStartFns").def_ro("name", &SelfStartFns::name);
    m.def("selfstart_names", [] { return std::vector<std::string>{"logis", "fpl", "asymp", "asymp_off", "asymp_orig", "micmen", "gompertz", "weibull"}; });
    m.def("selfstart_params", [](const std::string& n) { return ss_by_name(n).par; }, "name"_a);
    m.def("selfstart_initial", [](const std::string& n, const VectorXd& x, const VectorXd& y) { return ss_initial(ss_by_name(n), x, y); },
          "name"_a, "x"_a, "y"_a);
    m.def("selfstart_eval", [](const std::string& n, const MatrixXd& Phi, const VectorXd& x) { return VectorXd(ss_by_name(n).f(Phi, x)); },
          "name"_a, "Phi"_a, "x"_a);
    m.def("make_selfstart_fns", [](const std::string& n, const std::vector<VectorXd>& x_by_group) {
        NlmeSSFns r = make_ss_fns(ss_by_name(n), x_by_group);
        return SelfStartFns{r.f, r.jac, n};
    }, "name"_a, "x_by_group"_a);

    // ---------------- fits ----------------
    m.def("fit_lme", [](const VectorXd& y, const MatrixXd& X, const MatrixXd& Z, const VectorXi& labels, const LmeSpec& spec,
                        const std::optional<VectorXd>& time, const std::optional<MatrixXd>& coords,
                        const std::vector<VectorXi>& strata_slots, const std::vector<VectorXd>& cov_slots,
                        const std::vector<MatrixXd>& inner_Z, const std::vector<VectorXi>& inner_sub,
                        const std::optional<VectorXi>& cor_sub, const std::optional<VectorXd>& start) {
        auto gs = make_groups(y, X, Z, to_ivec(labels), opt_vec(time), {}, VectorXd(), opt_mat(coords));
        for (size_t l = 0; l < inner_Z.size(); ++l) add_inner_level(gs, inner_Z[l], to_ivec(inner_sub[l]));
        if (cor_sub) add_cor_sub(gs, to_ivec(*cor_sub));
        set_slots(gs, strata_slots, cov_slots);
        LmeModel model(gs, spec);
        LmeResult r = model.fit(start ? &*start : nullptr);
        return std::make_tuple(std::move(r), model.theta_labels(), model);
    }, "y"_a, "X"_a, "Z"_a, "labels"_a, "spec"_a, "time"_a = nb::none(), "coords"_a = nb::none(), "strata_slots"_a = std::vector<VectorXi>{},
       "cov_slots"_a = std::vector<VectorXd>{}, "inner_Z"_a = std::vector<MatrixXd>{}, "inner_sub"_a = std::vector<VectorXi>{},
       "cor_sub"_a = nb::none(), "start"_a = nb::none());

    m.def("fit_gls", [](const VectorXd& y, const MatrixXd& X, const VectorXi& labels, const GlsSpec& spec,
                        const std::optional<VectorXd>& time, const std::optional<MatrixXd>& coords,
                        const std::vector<VectorXi>& strata_slots, const std::vector<VectorXd>& cov_slots,
                        const std::optional<VectorXd>& start) {
        auto gs = make_gls_groups(y, X, to_ivec(labels), opt_vec(time), {}, VectorXd(), opt_mat(coords));
        set_slots(gs, strata_slots, cov_slots);
        GlsModel model(gs, spec);
        LmeResult r = model.fit(start ? &*start : nullptr);
        return std::make_tuple(std::move(r), model.model().theta_labels(), model.model());
    }, "y"_a, "X"_a, "labels"_a, "spec"_a, "time"_a = nb::none(), "coords"_a = nb::none(), "strata_slots"_a = std::vector<VectorXi>{},
       "cov_slots"_a = std::vector<VectorXd>{}, "start"_a = nb::none());

    m.def("fit_nlme", [](const VectorXd& y, const VectorXi& labels, const std::vector<MatrixXd>& A_list, const std::vector<MatrixXd>& B_list,
                         nb::object f, nb::object jac, const NlmeSpec& spec_in, const VectorXd& beta0,
                         const std::optional<VectorXd>& time, const std::optional<MatrixXd>& coords,
                         const std::vector<VectorXi>& strata_slots, const std::vector<VectorXd>& cov_slots,
                         const std::vector<MatrixXd>& inner_B, const std::vector<VectorXi>& inner_sub,
                         const std::optional<VectorXi>& cor_sub, const std::optional<VectorXi>& use,
                         const std::optional<VectorXd>& theta0, bool auto_start, const std::vector<MatrixXd>& A_rows,
                         const std::vector<MatrixXd>& B_rows, const std::vector<std::vector<MatrixXd>>& inner_B_rows) {
        const int N = static_cast<int>(y.size());
        const std::vector<int> lab = to_ivec(labels);
        // one group per distinct label, in ascending order (like make_nlme_groups); A and B per group
        auto gs = make_nlme_groups(y, lab, A_list.at(0), B_list.at(0), opt_vec(time), {}, VectorXd(), opt_mat(coords));
        if (gs.size() != A_list.size() || gs.size() != B_list.size()) throw std::invalid_argument("fit_nlme: A_list/B_list must have one element per group");
        for (size_t g = 0; g < gs.size(); ++g) { gs[g].A = A_list[g]; gs[g].B = B_list[g]; }
        for (size_t l = 0; l < inner_B.size(); ++l) add_nlme_inner_level(gs, inner_B[l], to_ivec(inner_sub[l]));
        if (!A_rows.empty()) {   // row-wise covariates (phi_j = A_j beta + B_j b): (n_g*k) x p / q per group
            if (A_rows.size() != gs.size()) throw std::invalid_argument("fit_nlme: A_rows must have one element per group");
            for (size_t g = 0; g < gs.size(); ++g) gs[g].A_row = A_rows[g];
        }
        if (!B_rows.empty()) {
            if (B_rows.size() != gs.size()) throw std::invalid_argument("fit_nlme: B_rows must have one element per group");
            for (size_t g = 0; g < gs.size(); ++g) gs[g].B_row = B_rows[g];
        }
        for (size_t l = 0; l < inner_B_rows.size(); ++l) {
            if (inner_B_rows[l].empty()) continue;
            if (inner_B_rows[l].size() != gs.size()) throw std::invalid_argument("fit_nlme: inner_B_rows[l] must have one element per group");
            for (size_t g = 0; g < gs.size(); ++g) gs[g].inner.at(l).B_row = inner_B_rows[l][g];
        }
        if (cor_sub) add_nlme_cor_sub(gs, to_ivec(*cor_sub));
        if (use) {
            std::vector<char> keep(N);
            for (int i = 0; i < N; ++i) keep[i] = (*use)[i] != 0;
            set_na_pattern(gs, keep);
        }
        set_slots(gs, strata_slots, cov_slots);
        FnPair fp = resolve_fn(f, jac);
        NlmeSpec spec = spec_in;
        if (fp.python) spec.parallel = false;   // Python callbacks are not thread-safe
        NlmeModel model(gs, fp.f, spec, fp.jac);
        NlmeStart st;
        NlmeResult r;
        if (auto_start) { st = nlme_start_from_nls(model, beta0); r = model.fit(st.beta, &st.theta); }
        else r = model.fit(beta0, theta0 ? &*theta0 : nullptr);
        return std::make_tuple(std::move(r), st.beta, st.D0, st.sigma, model);
    }, "y"_a, "labels"_a, "A_list"_a, "B_list"_a, "f"_a, "jac"_a.none(), "spec"_a, "beta0"_a, "time"_a = nb::none(), "coords"_a = nb::none(),
       "strata_slots"_a = std::vector<VectorXi>{}, "cov_slots"_a = std::vector<VectorXd>{}, "inner_B"_a = std::vector<MatrixXd>{},
       "inner_sub"_a = std::vector<VectorXi>{}, "cor_sub"_a = nb::none(), "use"_a = nb::none(), "theta0"_a = nb::none(), "auto_start"_a = false,
       "A_rows"_a = std::vector<MatrixXd>{}, "B_rows"_a = std::vector<MatrixXd>{}, "inner_B_rows"_a = std::vector<std::vector<MatrixXd>>{});

    m.def("fit_gnls", [](const VectorXd& y, const VectorXi& labels, const std::vector<MatrixXd>& A_list, nb::object f, nb::object jac,
                         const GnlsSpec& spec_in, const VectorXd& beta0, const std::optional<VectorXd>& time,
                         const std::optional<MatrixXd>& coords, const std::vector<VectorXi>& strata_slots,
                         const std::vector<VectorXd>& cov_slots, const std::optional<VectorXi>& use, const std::vector<MatrixXd>& A_rows) {
        const int N = static_cast<int>(y.size());
        auto gs = make_gnls_groups(y, to_ivec(labels), A_list.at(0), opt_vec(time), {}, VectorXd(), opt_mat(coords));
        if (gs.size() != A_list.size()) throw std::invalid_argument("fit_gnls: A_list must have one element per group");
        for (size_t g = 0; g < gs.size(); ++g) gs[g].A = A_list[g];
        if (!A_rows.empty()) {
            if (A_rows.size() != gs.size()) throw std::invalid_argument("fit_gnls: A_rows must have one element per group");
            for (size_t g = 0; g < gs.size(); ++g) gs[g].A_row = A_rows[g];
        }
        if (use) {
            std::vector<char> keep(N);
            for (int i = 0; i < N; ++i) keep[i] = (*use)[i] != 0;
            set_na_pattern(gs, keep);
        }
        set_slots(gs, strata_slots, cov_slots);
        FnPair fp = resolve_fn(f, jac);
        GnlsSpec spec = spec_in;
        GnlsModel model(gs, fp.f, spec, fp.jac);
        return std::make_pair(model.fit(beta0), model.model());
    }, "y"_a, "labels"_a, "A_list"_a, "f"_a, "jac"_a.none(), "spec"_a, "beta0"_a, "time"_a = nb::none(), "coords"_a = nb::none(),
       "strata_slots"_a = std::vector<VectorXi>{}, "cov_slots"_a = std::vector<VectorXd>{}, "use"_a = nb::none(),
       "A_rows"_a = std::vector<MatrixXd>{});

    // ---------------- lmList / nlsList ----------------
    nb::class_<LmListResult>(m, "LmListResult").def_ro("labels", &LmListResult::labels).def_ro("coef", &LmListResult::coef)
        .def_ro("se", &LmListResult::se).def_ro("vcov", &LmListResult::vcov).def_ro("sigma", &LmListResult::sigma)
        .def_ro("rss", &LmListResult::rss).def_ro("df", &LmListResult::df).def_ro("rank", &LmListResult::rank)
        .def_ro("pooled_sigma", &LmListResult::pooled_sigma).def_ro("pooled_df", &LmListResult::pooled_df);
    nb::class_<NlsListResult>(m, "NlsListResult").def_ro("labels", &NlsListResult::labels).def_ro("coef", &NlsListResult::coef)
        .def_ro("se", &NlsListResult::se).def_ro("vcov", &NlsListResult::vcov).def_ro("sigma", &NlsListResult::sigma)
        .def_ro("rss", &NlsListResult::rss).def_ro("df", &NlsListResult::df).def_ro("iterations", &NlsListResult::iterations)
        .def_ro("converged", &NlsListResult::converged).def_ro("pooled_sigma", &NlsListResult::pooled_sigma)
        .def_ro("pooled_df", &NlsListResult::pooled_df);
    m.def("lm_list", [](const VectorXd& y, const MatrixXd& X, const VectorXi& labels) {
        return lm_list(make_gls_groups(y, X, to_ivec(labels)));
    }, "y"_a, "X"_a, "labels"_a);
    m.def("nls_list", [](const VectorXd& y, const VectorXi& labels, const std::vector<MatrixXd>& A_list, nb::object f, nb::object jac,
                         const VectorXd& beta0, const std::optional<VectorXi>& use, double tol, int max_iter) {
        auto gs = make_gnls_groups(y, to_ivec(labels), A_list.at(0));
        for (size_t g = 0; g < gs.size(); ++g) gs[g].A = A_list.at(g);
        if (use) {
            std::vector<char> keep(y.size());
            for (int i = 0; i < y.size(); ++i) keep[i] = (*use)[i] != 0;
            set_na_pattern(gs, keep);
        }
        FnPair fp = resolve_fn(f, jac);
        NlsListControl ctl; ctl.tol = tol; ctl.max_iter = max_iter;
        return nls_list(gs, fp.f, fp.jac, beta0, ctl);
    }, "y"_a, "labels"_a, "A_list"_a, "f"_a, "jac"_a.none(), "beta0"_a, "use"_a = nb::none(), "tol"_a = 1e-8, "max_iter"_a = 200);


    // ---------------- crossed effects (sparse algebra) ----------------
    nb::class_<ReTerm>(m, "ReTerm").def_ro("name", &ReTerm::name).def_ro("n_levels", &ReTerm::n_levels);
    m.def("make_re_term", [](const std::string& name, const VectorXi& group, int n_levels, const MatrixXd& Z, const PdSpec& pd) {
        ReTerm t;
        t.name = name; t.group = to_ivec(group); t.n_levels = n_levels; t.Z = Z; t.pd = pd;
        return t;
    }, "name"_a, "group"_a, "n_levels"_a, "Z"_a, "pd"_a);
    nb::class_<CrossedSpec>(m, "CrossedSpec").def(nb::init<>())
        .def_rw("reml", &CrossedSpec::reml).def_rw("optimizer", &CrossedSpec::optimizer)
        .def_rw("theta_bound", &CrossedSpec::theta_bound).def_rw("opt", &CrossedSpec::opt);
    nb::class_<CrossedProfile>(m, "CrossedProfile").def_ro("dev", &CrossedProfile::dev).def_ro("logdetA", &CrossedProfile::logdetA)
        .def_ro("logdetS", &CrossedProfile::logdetS).def_ro("pwrss", &CrossedProfile::pwrss).def_ro("sumlogw", &CrossedProfile::sumlogw)
        .def_ro("N", &CrossedProfile::N).def_ro("p", &CrossedProfile::p).def_ro("beta", &CrossedProfile::beta)
        .def_ro("Sinv", &CrossedProfile::Sinv).def_ro("u", &CrossedProfile::u).def_ro("ok", &CrossedProfile::ok);
    nb::class_<CrossedResult>(m, "CrossedResult")
        .def_ro("beta", &CrossedResult::beta).def_ro("se", &CrossedResult::se).def_ro("vcov_beta", &CrossedResult::vcov_beta)
        .def_ro("sigma2", &CrossedResult::sigma2).def_ro("theta", &CrossedResult::theta).def_ro("D_terms", &CrossedResult::D_terms)
        .def_ro("Sigma_terms", &CrossedResult::Sigma_terms).def_ro("blup", &CrossedResult::blup).def_ro("loglik", &CrossedResult::loglik)
        .def_ro("deviance", &CrossedResult::deviance).def_ro("reml", &CrossedResult::reml).def_ro("singular", &CrossedResult::singular)
        .def_ro("rank", &CrossedResult::rank).def_ro("N", &CrossedResult::N).def_ro("n_re", &CrossedResult::n_re)
        .def_ro("aliased", &CrossedResult::aliased).def_ro("fitted_pop", &CrossedResult::fitted_pop)
        .def_ro("fitted_cond", &CrossedResult::fitted_cond).def_ro("resid", &CrossedResult::resid).def_ro("opt", &CrossedResult::opt)
        .def_ro("warning", &CrossedResult::warning);
    nb::class_<CrossedModel>(m, "CrossedModel")
        .def("__init__", [](CrossedModel* self, const VectorXd& y, const MatrixXd& X, const std::vector<ReTerm>& terms, const CrossedSpec& spec,
                            const VectorXd& weights, const VectorXd& offset) { new (self) CrossedModel(y, X, terms, spec, weights, offset); },
             "y"_a, "X"_a, "terms"_a, "spec"_a, "weights"_a = VectorXd(), "offset"_a = VectorXd())
        .def("fit", [](const CrossedModel& mod, const std::optional<VectorXd>& start) { return mod.fit(start ? &*start : nullptr); }, "start"_a = nb::none())
        .def("profile", &CrossedModel::profile, "theta"_a)
        .def("deviance", &CrossedModel::deviance, "theta"_a)
        .def("default_theta", &CrossedModel::default_theta)
        .def_prop_ro("n_theta", &CrossedModel::n_theta).def_prop_ro("rank", &CrossedModel::rank).def_prop_ro("n_random", &CrossedModel::n_random)
        .def_prop_ro("aliased", &CrossedModel::aliased);

    // ---------------- GLMM (Laplace / AGQ) ----------------
    nb::enum_<GlmFamily>(m, "GlmFamily")
        .value("Binomial", GlmFamily::Binomial).value("Poisson", GlmFamily::Poisson).value("NegBin", GlmFamily::NegBin)
        .value("Gaussian", GlmFamily::Gaussian).value("Gamma", GlmFamily::Gamma).value("InvGauss", GlmFamily::InvGauss);
    nb::enum_<GlmLink>(m, "GlmLink").value("Logit", GlmLink::Logit).value("Probit", GlmLink::Probit).value("Cloglog", GlmLink::Cloglog)
        .value("Cauchit", GlmLink::Cauchit).value("Log", GlmLink::Log).value("Sqrt", GlmLink::Sqrt)
        .value("Identity", GlmLink::Identity).value("Inverse", GlmLink::Inverse).value("Invmu2", GlmLink::Invmu2);
    nb::class_<GlmmSpec>(m, "GlmmSpec").def(nb::init<>())
        .def_rw("family", &GlmmSpec::family).def_rw("link", &GlmmSpec::link).def_rw("theta_nb", &GlmmSpec::theta_nb)
        .def_rw("estimate_theta", &GlmmSpec::estimate_theta).def_rw("dispersion", &GlmmSpec::dispersion)
        .def_rw("estimate_dispersion", &GlmmSpec::estimate_dispersion).def_rw("nAGQ", &GlmmSpec::nAGQ).def_rw("init_stage", &GlmmSpec::init_stage)
        .def_rw("optimizer", &GlmmSpec::optimizer).def_rw("theta_bound", &GlmmSpec::theta_bound).def_rw("max_pirls", &GlmmSpec::max_pirls)
        .def_rw("pirls_tol", &GlmmSpec::pirls_tol).def_rw("pirls_step_tol", &GlmmSpec::pirls_step_tol).def_rw("opt", &GlmmSpec::opt);
    nb::class_<GlmmResult>(m, "GlmmResult")
        .def_ro("beta", &GlmmResult::beta).def_ro("se", &GlmmResult::se).def_ro("vcov_beta", &GlmmResult::vcov_beta).def_ro("vcov_rx", &GlmmResult::vcov_rx)
        .def_ro("vcov_hessian", &GlmmResult::vcov_hessian).def_ro("theta", &GlmmResult::theta).def_ro("theta_nb", &GlmmResult::theta_nb)
        .def_ro("dispersion", &GlmmResult::dispersion)
        .def_ro("D_terms", &GlmmResult::D_terms).def_ro("blup", &GlmmResult::blup).def_ro("loglik", &GlmmResult::loglik)
        .def_ro("deviance", &GlmmResult::deviance).def_ro("ll_sat", &GlmmResult::ll_sat).def_ro("rank", &GlmmResult::rank).def_ro("N", &GlmmResult::N)
        .def_ro("n_re", &GlmmResult::n_re).def_ro("nAGQ", &GlmmResult::nAGQ).def_ro("aliased", &GlmmResult::aliased).def_ro("eta", &GlmmResult::eta)
        .def_ro("mu", &GlmmResult::mu).def_ro("eta_pop", &GlmmResult::eta_pop).def_ro("mu_pop", &GlmmResult::mu_pop).def_ro("W", &GlmmResult::W)
        .def_ro("singular", &GlmmResult::singular).def_ro("opt", &GlmmResult::opt).def_ro("stage1_evals", &GlmmResult::stage1_evals)
        .def_ro("warning", &GlmmResult::warning);
    nb::class_<GlmmModel>(m, "GlmmModel")
        .def("__init__", [](GlmmModel* self, const VectorXd& y, const MatrixXd& X, const std::vector<ReTerm>& terms, const GlmmSpec& spec,
                            const VectorXd& weights, const VectorXd& offset) { new (self) GlmmModel(y, X, terms, spec, weights, offset); },
             "y"_a, "X"_a, "terms"_a, "spec"_a, "weights"_a = VectorXd(), "offset"_a = VectorXd())
        .def("fit", &GlmmModel::fit)
        .def_prop_ro("n_theta", &GlmmModel::n_theta).def_prop_ro("rank", &GlmmModel::rank).def_prop_ro("n_random", &GlmmModel::n_random)
        .def_prop_ro("aliased", &GlmmModel::aliased);

    m.def("build_info", [] {
        nb::dict d;
#ifdef MIXEDFIT_WITH_MINIMIZER
        d["minimizer"] = true;
#else
        d["minimizer"] = false;
#endif
#ifdef _OPENMP
        d["openmp"] = true;
#else
        d["openmp"] = false;
#endif
        return d;
    });
}
