// ============================================================================
// _saemfit.cpp  —  nanobind binding for saemfit (header-only SAEM, namespace smx)
//
// Low-level layer: exposes Data / Model / Control / Results / simulate almost 1:1.
// The R-style interface (formulas, DataFrames, summary(), ...) lives in Python.
//
// Design decisions
//  * Python callbacks (prediction, likelihood) are invoked with the GIL held.
//    A Python exception inside a callback is SAVED (Guard) and aborts the fit;
//    when unwinding, the ORIGINAL error is re-raised even if the engine caught
//    the intermediate C++ exception (several `catch (std::exception&)` blocks in
//    the engine would otherwise swallow it and leave a silently degraded fit).
//  * With Python callbacks the fit runs on 1 thread and retains the GIL; with
//    compiled models (builtin) the GIL is released and Control.n_threads is used.
//  * Ctrl-C works inside callbacks (PyErr_CheckSignals).
//  * C++ stdout (verbose) is redirected to sys.stdout (Jupyter / pytest capsys).
// ============================================================================

#include <nanobind/nanobind.h>
#include <unordered_map>
#include <cstring>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include "../include/saemfit/saem.hpp"
#include "../include/saemfit/expr.hpp"

#include <cmath>
#include <memory>
#include <numeric>
#include <sstream>
#include <streambuf>

namespace nb = nanobind;
using namespace nb::literals;

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Xs  = std::span<const double>;

// ===========================================================================
// NumPy utilities
// ===========================================================================
namespace {

template <class T>
nb::object vec_to_np(const std::vector<T>& v) {
    auto* d = new std::vector<T>(v);
    nb::capsule owner(d, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(d->data(), {d->size()}, owner));
}

nb::object np1(const double* p, std::size_t n) {
    auto* d = new std::vector<double>(p, p + n);
    nb::capsule owner(d, [](void* q) noexcept { delete static_cast<std::vector<double>*>(q); });
    return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<1>>(d->data(), {n}, owner));
}

nb::object np2(const double* p, std::size_t r, std::size_t c) {
    auto* d = new std::vector<double>(p, p + r * c);
    nb::capsule owner(d, [](void* q) noexcept { delete static_cast<std::vector<double>*>(q); });
    return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<2>>(d->data(), {r, c}, owner));
}

using DArr1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;
using IArr1 = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig>;
using DArr2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig>;

std::vector<double> to_vd(const DArr1& a) { return std::vector<double>(a.data(), a.data() + a.shape(0)); }
std::vector<int> to_vi(const IArr1& a) {
    std::vector<int> v(a.shape(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int>(a.data()[i]);
    return v;
}
Vec to_eig(const DArr1& a) { return Eigen::Map<const Vec>(a.data(), static_cast<Eigen::Index>(a.shape(0))); }

std::vector<char> to_flags(const std::vector<int>& v) { return std::vector<char>(v.begin(), v.end()); }
std::vector<int> from_flags(const std::vector<char>& v) { return std::vector<int>(v.begin(), v.end()); }

nb::object doses_to_np(const std::vector<ode::Dose>& dl) {
    // columns: [time, cmt (1-based), amt, duration, ii, ss, modeled]
    const std::size_t m = dl.size();
    std::vector<double> buf(m * 7);
    for (std::size_t i = 0; i < m; ++i) {
        const auto& d = dl[i];
        buf[i * 7 + 0] = d.time; buf[i * 7 + 1] = d.cmt + 1; buf[i * 7 + 2] = d.amount;
        buf[i * 7 + 3] = d.duration; buf[i * 7 + 4] = d.ii; buf[i * 7 + 5] = d.ss; buf[i * 7 + 6] = d.modeled;
    }
    return np2(buf.data(), m, 7);
}

std::vector<ode::Dose> doses_from_np(const DArr2& a) {
    if (a.shape(1) < 3) throw std::invalid_argument("doses: expected columns [time, cmt, amt, dur, ii, ss, modeled]");
    std::vector<ode::Dose> out;
    const std::size_t m = a.shape(0), c = a.shape(1);
    for (std::size_t i = 0; i < m; ++i) {
        const double* r = a.data() + i * c;
        ode::Dose d{r[0], static_cast<int>(std::lround(r[1])) - 1, r[2], c > 3 ? r[3] : 0.0};
        if (c > 4) d.ii = r[4];
        if (c > 5) d.ss = static_cast<int>(std::lround(r[5]));
        if (c > 6) d.modeled = static_cast<int>(std::lround(r[6]));
        out.push_back(d);
    }
    return out;
}

// ===========================================================================
// Guard: Python error raised inside a callback
// ===========================================================================
struct Guard {
    bool failed = false;
    std::unique_ptr<nb::python_error> err;
    std::string msg;
    void reset() { failed = false; err.reset(); msg.clear(); }
    void record(const nb::python_error& e) { if (!failed) { failed = true; err = std::make_unique<nb::python_error>(e); } }
    void record(const std::string& m) { if (!failed) { failed = true; msg = m; } }
    // Re-raise (and clear) the saved error, if any.
    void rethrow_if_failed() {
        if (!failed) return;
        if (err) { nb::python_error e(*err); reset(); throw e; }
        std::string m = msg; reset();
        throw std::runtime_error(m);
    }
};
Guard& G() { static Guard g; return g; }

template <class F>
auto guarded(F&& f) -> decltype(f()) {
    nb::gil_scoped_acquire gil;
    Guard& g = G();
    if (g.failed) throw std::runtime_error("user callback aborted due to a previous error");
    try {
        if (PyErr_CheckSignals() != 0) throw nb::python_error();
        return f();
    } catch (nb::python_error& e) {
        g.record(e);
        throw std::runtime_error("Python exception in a model callback");
    } catch (const std::exception& e) {
        g.record(std::string("error in a model callback: ") + e.what());
        throw;
    }
}

// Cache of NumPy arrays for the per-subject prediction callback. A subject's data (x, cov,
// doses) do not change during the fit: rebuilding 3 NumPy arrays per call used to be the largest
// part of the bridge cost. Each entry is VALIDATED by content on every call (if the user mutates the
// array in place, or the address is reused for another subject, it is rebuilt). Only touched with the GIL.
struct PredCache {
    // arr: NumPy array handed to the user; live: its REAL buffer. Validation compares that buffer (which
    // the user may have mutated in place) with the original data on every call.
    struct Ent { nb::object arr; double* live = nullptr; std::size_t cnt = 0; };
    std::unordered_map<const void*, Ent> xs, covs, doses;
    nb::object psiObj;                 // reusable array (overwritten in place)
    double* psiPtr = nullptr;
    std::size_t psiN = 0;

    static void trim(std::unordered_map<const void*, Ent>& m) { if (m.size() > 4096) m.clear(); }
    static double* liveOf(const nb::object& o) { return nb::cast<nb::ndarray<nb::numpy, double>>(o).data(); }

    nb::object get(std::unordered_map<const void*, Ent>& m, const void* key, const double* p, std::size_t r, std::size_t c, bool two) {
        const std::size_t cnt = r * c;
        auto it = m.find(key);
        if (it != m.end() && it->second.cnt == cnt &&
            (cnt == 0 || std::memcmp(it->second.live, p, cnt * sizeof(double)) == 0))
            return it->second.arr;
        trim(m);
        Ent e;
        e.arr = two ? np2(p, r, c) : np1(p, cnt);
        e.live = liveOf(e.arr); e.cnt = cnt;
        nb::object out = e.arr;
        m[key] = std::move(e);
        return out;
    }
    nb::object psi(const double* p, std::size_t n) {
        if (psiN != n || psiObj.is_none() || !psiObj.is_valid()) {
            psiObj = np1(p, n);
            psiPtr = liveOf(psiObj);
            psiN = n;
            return psiObj;
        }
        std::memcpy(psiPtr, p, n * sizeof(double));
        return psiObj;
    }
    nb::object dosesArr(const std::vector<ode::Dose>& dl) {
        const std::size_t m = dl.size();
        auto it = doses.find(dl.data());
        if (it != doses.end() && it->second.cnt == m * 7) {
            bool same = true;
            const double* b = it->second.live;
            for (std::size_t i = 0; i < m && same; ++i) {
                const auto& d = dl[i];
                same = b[i*7] == d.time && b[i*7+1] == d.cmt + 1 && b[i*7+2] == d.amount && b[i*7+3] == d.duration &&
                       b[i*7+4] == d.ii && b[i*7+5] == d.ss && b[i*7+6] == d.modeled;
            }
            if (same) return it->second.arr;
        }
        trim(doses);
        Ent e; e.arr = doses_to_np(dl);
        e.live = liveOf(e.arr); e.cnt = m * 7;
        nb::object out = e.arr;
        doses[dl.data()] = std::move(e);
        return out;
    }
};

using Fn = std::shared_ptr<nb::object>;
Fn hold(nb::handle h) {
    return Fn(new nb::object(nb::borrow(h)), [](nb::object* o) { nb::gil_scoped_acquire g; delete o; });
}

// ===========================================================================
// Redirect std::cout -> sys.stdout
// ===========================================================================
class PyOutBuf : public std::streambuf {
public:
    PyOutBuf() { write_ = nb::module_::import_("sys").attr("stdout").attr("write"); }
protected:
    int_type overflow(int_type c) override {
        if (c != traits_type::eof()) { buf_.push_back(static_cast<char>(c)); if (c == '\n') flush_(); }
        return c;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        buf_.append(s, static_cast<std::size_t>(n));
        if (buf_.find('\n') != std::string::npos) flush_();
        return n;
    }
    int sync() override { flush_(); return 0; }
private:
    void flush_() {
        if (buf_.empty()) return;
        nb::gil_scoped_acquire g;
        write_(nb::str(buf_.c_str(), buf_.size()));
        buf_.clear();
    }
    nb::object write_;
    std::string buf_;
};
struct CoutRedirect {
    std::unique_ptr<PyOutBuf> buf;
    std::streambuf* old = nullptr;
    CoutRedirect() { buf = std::make_unique<PyOutBuf>(); old = std::cout.rdbuf(buf.get()); }
    ~CoutRedirect() { std::cout.flush(); std::cout.rdbuf(old); }
};

// ===========================================================================
// Compiled models (thread-safe, no GIL)
// ===========================================================================
struct BuiltinInfo { int nPhi; };
const std::map<std::string, BuiltinInfo>& builtinTable() {
    static const std::map<std::string, BuiltinInfo> t = {
        {"exp_decay", {2}},       // A, k
        {"biexp", {4}},           // A1, k1, A2, k2
        {"emax", {3}},            // E0, Emax, EC50
        {"sigmoid_emax", {4}},    // E0, Emax, EC50, gamma
        {"logistic", {3}},        // Asym, xmid, scal
        {"gompertz", {3}},        // Asym, b2, b3
        {"michaelis_menten", {2}},// Vmax, Km
        {"pk1_iv", {2}},          // CL, V
        {"pk1_oral", {3}},        // ka, CL, V
        {"pk2_iv", {4}},          // CL, V1, Q, V2
        {"pk2_oral", {5}},        // ka, CL, V1, Q, V2
    };
    return t;
}

mcmckl::PredictSubject makeBuiltin(const std::string& name, int xcol, int doseCov) {
    const auto& tab = builtinTable();
    if (!tab.count(name)) throw std::invalid_argument("unknown compiled model: '" + name + "'");
    const int np = tab.at(name).nPhi;

    // --- algebraic models: f_j = g(psi, x_j) ---
    auto algebraic = [=](std::function<double(const Vec&, double)> g) -> mcmckl::PredictSubject {
        return [=](const Vec& psi, const mcmckl::ObsView& o, const Vec&, Vec& f) {
            if (psi.size() != np) throw std::invalid_argument("model '" + name + "': expected " + std::to_string(np) + " parameters");
            if (xcol < 0 || xcol > o.stride - 2) throw std::invalid_argument("x_col out of range");
            f.resize(o.n);
            for (int j = 0; j < o.n; ++j) f[j] = g(psi, o.x[std::size_t(j) * o.stride + xcol]);
        };
    };
    if (name == "exp_decay")
        return algebraic([](const Vec& p, double x) { return p[0] * std::exp(-p[1] * x); });
    if (name == "biexp")
        return algebraic([](const Vec& p, double x) { return p[0] * std::exp(-p[1] * x) + p[2] * std::exp(-p[3] * x); });
    if (name == "emax")
        return algebraic([](const Vec& p, double x) { return p[0] + p[1] * x / (p[2] + x); });
    if (name == "sigmoid_emax")
        return algebraic([](const Vec& p, double x) {
            const double xg = std::pow(std::max(x, 0.0), p[3]);
            return p[0] + p[1] * xg / (std::pow(p[2], p[3]) + xg);
        });
    if (name == "logistic")
        return algebraic([](const Vec& p, double x) { return p[0] / (1.0 + std::exp((p[1] - x) / p[2])); });
    if (name == "gompertz")
        return algebraic([](const Vec& p, double x) { return p[0] * std::exp(-p[1] * std::pow(p[2], x)); });
    if (name == "michaelis_menten")
        return algebraic([](const Vec& p, double x) { return p[0] * x / (p[1] + x); });

    // --- linear compartmental models: exact matrix exponential (ode.hpp, linear = true) ---
    const bool oral = name.find("oral") != std::string::npos;
    const bool two = name.rfind("pk2", 0) == 0;
    const int ncmt = (two ? 2 : 1) + (oral ? 1 : 0);
    const int cen = oral ? 1 : 0;                      // central compartment (0-based)
    return [=](const Vec& psi, const mcmckl::ObsView& o, const Vec& cov, Vec& f) {
        if (psi.size() != np) throw std::invalid_argument("model '" + name + "': expected " + std::to_string(np) + " parameters");
        double ka = 0, CL, V1, Q = 0, V2 = 1;
        int k = 0;
        if (oral) ka = psi[k++];
        CL = psi[k++]; V1 = psi[k++];
        if (two) { Q = psi[k++]; V2 = psi[k++]; }
        f.resize(o.n);
        if (!(CL > 0 && V1 > 0 && (!oral || ka > 0) && (!two || (Q > 0 && V2 > 0)))) { f.setConstant(std::numeric_limits<double>::quiet_NaN()); return; }
        // doses: no copy when the subject already carries its record (a local one is only built for the covariate)
        thread_local std::vector<ode::Dose> localDoses;
        const std::vector<ode::Dose>* dp = &o.doseList();
        if (dp->empty()) {
            if (doseCov < 0 || doseCov >= cov.size())
                throw std::invalid_argument("model '" + name + "': no doses (use dose events or dose=<covariate>)");
            localDoses.assign(1, ode::Dose{0.0, 0, cov[doseCov], 0.0});
            dp = &localDoses;
        }
        const std::vector<ode::Dose>& doses = *dp;
        for (const auto& d : doses) {
            if (d.modeled != 0) throw std::invalid_argument("model '" + name + "': RATE = -1/-2 (modeled rate/duration) is not supported for compiled models");
            if (d.cmt < 0 || d.cmt >= ncmt) throw std::invalid_argument("model '" + name + "': dose cmt out of range (1.." + std::to_string(ncmt) + ")");
        }
        // observations in time order (they almost always already are: sorting is skipped)
        thread_local std::vector<int> ord;
        thread_local std::vector<double> ts;
        ts.resize(o.n);
        bool sorted = true;
        for (int j = 0; j < o.n; ++j) { ts[j] = o.time(j); if (j > 0 && ts[j] < ts[j - 1]) sorted = false; }
        if (!sorted) {
            ord.resize(o.n);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return o.time(a) < o.time(b); });
            for (int j = 0; j < o.n; ++j) ts[j] = o.time(ord[j]);
        }
        double t0 = 0.0;
        for (int j = 0; j < o.n; ++j) t0 = std::min(t0, ts[j]);
        for (const auto& d : doses) t0 = std::min(t0, d.time);
        const double k10 = CL / V1, k12 = Q / V1, k21 = Q / V2;
        // closed-form modal decomposition (or expm if the eigenvalues are nearly equal / ill-conditioned)
        ode::detail::LinSys L = two ? (oral ? ode::detail::pk2Oral(ka, k10, k12, k21) : ode::detail::pk2Iv(k10, k12, k21))
                                    : (oral ? ode::detail::pk1Oral(ka, k10) : ode::detail::pk1Iv(k10));
        const double invV1 = 1.0 / V1;
        ode::integrateLinear(L, t0, ts.data(), o.n, doses, [&](int k, const double* y) {
            f[sorted ? k : ord[k]] = y[cen] * invV1;
        });
    };
}

// ===========================================================================
// Enum helpers
// ===========================================================================
template <class E>
std::string enum_name(E e, const std::vector<std::pair<E, const char*>>& t) {
    for (auto& p : t) if (p.first == e) return p.second;
    return "?";
}

} // namespace

// ===========================================================================
// MODULE
// ===========================================================================
NB_MODULE(_saemfit, m) {
    m.attr("EXPR_NOPS") = static_cast<int>(mcmckl::E_COUNT);
    m.doc() = "saemfit: C++ core (SAEM) - low-level layer";
    // Results holds Python callbacks (which may in turn reference globals that contain the fit):
    // an invisible GC cycle that only surfaces as a warning when the interpreter shuts down.
    nb::set_leak_warnings(false);

    // ---------------- enums ----------------
    nb::enum_<smx::ErrorModel>(m, "ErrorModel")
        .value("constant", smx::ErrorModel::Constant)
        .value("proportional", smx::ErrorModel::Proportional)
        .value("exponential", smx::ErrorModel::Exponential)
        .value("combined", smx::ErrorModel::Combined)
        .value("combined2", smx::ErrorModel::Combined2)
        .value("power", smx::ErrorModel::Power);
    nb::enum_<smx::Transform>(m, "Transform")
        .value("identity", smx::Transform::Identity)
        .value("log", smx::Transform::Log)
        .value("logit", smx::Transform::Logit)
        .value("probit", smx::Transform::Probit);
    nb::enum_<smx::SEMethod>(m, "SEMethod")
        .value("auto", smx::SEMethod::Auto)
        .value("linearization", smx::SEMethod::Linearization)
        .value("gauss_hermite", smx::SEMethod::GaussHermite)
        .value("louis", smx::SEMethod::Louis)
        .value("laplace", smx::SEMethod::Laplace)
        .value("none", smx::SEMethod::None);
    nb::enum_<smx::LLMethod>(m, "LLMethod")
        .value("auto", smx::LLMethod::Auto)
        .value("gauss_hermite", smx::LLMethod::GaussHermite)
        .value("importance_sampling", smx::LLMethod::ImportanceSampling)
        .value("linearization", smx::LLMethod::Linearization)
        .value("laplace", smx::LLMethod::Laplace)
        .value("none", smx::LLMethod::None);
    nb::enum_<mcmckl::MStep>(m, "MStep")
        .value("auto", mcmckl::MStep::Auto)
        .value("gradient", mcmckl::MStep::Gradient)
        .value("sufficient_stats", mcmckl::MStep::SufficientStats);
    nb::enum_<mcmckl::RMControl::Schedule>(m, "RMSchedule")
        .value("constant", mcmckl::RMControl::Schedule::Constant)
        .value("one_over_t", mcmckl::RMControl::Schedule::OneOverT)
        .value("power_decay", mcmckl::RMControl::Schedule::PowerDecay)
        .value("kesten", mcmckl::RMControl::Schedule::Kesten);
    nb::enum_<mcmckl::SAControl::Schedule>(m, "SASchedule")
        .value("geometric", mcmckl::SAControl::Schedule::Geometric)
        .value("exponential", mcmckl::SAControl::Schedule::Exponential)
        .value("power", mcmckl::SAControl::Schedule::Power)
        .value("inverse", mcmckl::SAControl::Schedule::Inverse);

    // ---------------- Control building blocks ----------------
    nb::class_<resmod::Pres>(m, "ErrorPars")
        .def(nb::init<>())
        .def(nb::init<double, double, double>(), "a"_a, "b"_a, "c"_a = 1.0)
        .def_rw("a", &resmod::Pres::a)
        .def_rw("b", &resmod::Pres::b)
        .def_rw("c", &resmod::Pres::c);

    nb::class_<mcmckl::ClassicStepControl>(m, "ClassicStepControl")
        .def(nb::init<>())
        .def_rw("k1", &mcmckl::ClassicStepControl::k1)
        .def_rw("alpha", &mcmckl::ClassicStepControl::alpha);

    nb::class_<mcmckl::RMControl>(m, "RMControl")
        .def(nb::init<>())
        .def_rw("schedule", &mcmckl::RMControl::schedule)
        .def_rw("gamma0", &mcmckl::RMControl::gamma0)
        .def_rw("alpha", &mcmckl::RMControl::alpha)
        .def_rw("t0", &mcmckl::RMControl::t0)
        .def_rw("clip_gradient", &mcmckl::RMControl::clip_gradient)
        .def_rw("grad_clip", &mcmckl::RMControl::grad_clip)
        .def_rw("truncate", &mcmckl::RMControl::truncate)
        .def_rw("lower", &mcmckl::RMControl::lower)
        .def_rw("upper", &mcmckl::RMControl::upper)
        .def_rw("diagonal_precond", &mcmckl::RMControl::diagonal_precond)
        .def_rw("precond_decay", &mcmckl::RMControl::precond_decay)
        .def_rw("precond_eps", &mcmckl::RMControl::precond_eps)
        .def_rw("averaging", &mcmckl::RMControl::averaging)
        .def_rw("avg_start", &mcmckl::RMControl::avg_start)
        .def_rw("conv_window", &mcmckl::RMControl::conv_window)
        .def_rw("conv_tol", &mcmckl::RMControl::conv_tol);

    nb::class_<mcmckl::SAControl>(m, "SAControl")
        .def(nb::init<>())
        .def_rw("schedule", &mcmckl::SAControl::schedule)
        .def_rw("beta0", &mcmckl::SAControl::beta0)
        .def_rw("rate", &mcmckl::SAControl::rate)
        .def_rw("T", &mcmckl::SAControl::T)
        .def_rw("K", &mcmckl::SAControl::K)
        .def_rw("power", &mcmckl::SAControl::power)
        .def_rw("shrink", &mcmckl::SAControl::shrink)
        .def_rw("inflate_proposal", &mcmckl::SAControl::inflate_proposal)
        .def_rw("diagonal_only", &mcmckl::SAControl::diagonal_only);

    nb::class_<mcmckl::ConvergenceInfo>(m, "ConvergenceInfo")
        .def_ro("iterations", &mcmckl::ConvergenceInfo::iterations)
        .def_ro("stopped_early", &mcmckl::ConvergenceInfo::stoppedEarly)
        .def_ro("drift", &mcmckl::ConvergenceInfo::drift)
        .def_ro("max_drift", &mcmckl::ConvergenceInfo::maxDrift);

    // ---------------- Control ----------------
    using C = smx::Control;
    nb::class_<C>(m, "Control")
        .def(nb::init<>())
        .def_rw("beta0", &C::beta0)
        .def_rw("beta_cov0", &C::betaCov0)
        .def_rw("sd0", &C::sd0)
        .def_rw("a0", &C::a0)
        .def_rw("b0", &C::b0)
        .def_rw("c0", &C::c0)
        .def_prop_rw("error_init",
            [](const C& c) { return c.errorInit; },
            [](C& c, const std::vector<resmod::Pres>& v) { c.errorInit = v; })
        .def_rw("extra0", &C::extra0)
        .def_rw("auto_init", &C::autoInit)
        .def_prop_rw("fixed_beta", [](const C& c) { return from_flags(c.fixedBeta); }, [](C& c, const std::vector<int>& v) { c.fixedBeta = to_flags(v); })
        .def_prop_rw("fixed_sd", [](const C& c) { return from_flags(c.fixedSd); }, [](C& c, const std::vector<int>& v) { c.fixedSd = to_flags(v); })
        .def_prop_rw("fixed_error", [](const C& c) { return from_flags(c.fixedError); }, [](C& c, const std::vector<int>& v) { c.fixedError = to_flags(v); })
        .def_prop_rw("fixed_extra", [](const C& c) { return from_flags(c.fixedExtra); }, [](C& c, const std::vector<int>& v) { c.fixedExtra = to_flags(v); })
        .def_rw("sd_cluster0", &C::sdCluster0)
        .def_rw("n_iterations", &C::nbIterations)
        .def_rw("n_burn_in", &C::nbBurnIn)
        .def_rw("n_mcmc_sweeps", &C::nMCMCsweeps)
        .def_rw("seed", &C::seed)
        .def_rw("simulated_annealing", &C::simulatedAnnealing)
        .def_rw("n_chains", &C::nChains)
        .def_rw("mstep", &C::mstep)
        .def_rw("step", &C::step)
        .def_rw("n_threads", &C::nThreads)
        .def_rw("map_kernel", &C::mapKernel)
        .def_rw("map_every", &C::mapEvery)
        .def_rw("auto_stop", &C::autoStop)
        .def_rw("stop_tol", &C::stopTol)
        .def_rw("adapt_rw", &C::adaptRW)
        .def_rw("verbose", &C::verbose)
        .def_rw("print_every", &C::printEvery)
        .def_rw("compute_cond", &C::computeCond)
        .def_rw("n_cond_samples", &C::nCondSamples)
        .def_rw("cond_burn_in", &C::condBurnIn)
        .def_rw("cond_thin", &C::condThin)
        .def_rw("compute_se", &C::computeSE)
        .def_rw("se_method", &C::seMethod)
        .def_rw("ll_method", &C::llMethod)
        .def_rw("gq_nodes", &C::gqNodes)
        .def_rw("gq_inner_nodes", &C::gqInnerNodes)
        .def_rw("is_samples", &C::isSamples)
        .def_rw("louis_draws", &C::louisDraws)
        .def_rw("louis_is_samples", &C::louisISSamples)
        .def_rw("rm", &C::rm)
        .def_rw("sa", &C::sa)
        .def_static("default_rm", &C::defaultRM);

    // ---------------- Data ----------------
    using D = smx::Data;
    nb::class_<D>(m, "Data")
        .def("__init__", [](D* self, const IArr1& id, const DArr1& time, const DArr1& y) {
            new (self) D();
            self->id = to_vi(id); self->time = to_vd(time); self->y = to_vd(y);
            if (self->id.size() != self->time.size() || self->id.size() != self->y.size())
                throw std::invalid_argument("Data: id, time and y must have the same length");
        }, "id"_a, "time"_a, "y"_a)
        .def("set_group", [](D& d, const IArr1& g) { d.group = to_vi(g); }, "group"_a)
        .def("set_ytype", [](D& d, const IArr1& g) { d.ytype = to_vi(g); }, "ytype"_a)
        .def("set_cens", [](D& d, const IArr1& g) { d.cens = to_vi(g); }, "cens"_a)
        .def("set_limit", [](D& d, const DArr1& g) { d.limit = to_vd(g); }, "limit"_a)
        .def("add_predictor", [](D& d, const std::string& name, const DArr1& v) { d.addPredictor(name, to_vd(v)); }, "name"_a, "values"_a)
        .def("add_covariate", [](D& d, const std::string& name, const DArr1& v) {
            d.cov[name] = to_eig(v);
            if (std::find(d.covOrder.begin(), d.covOrder.end(), name) == d.covOrder.end()) d.covOrder.push_back(name);
        }, "name"_a, "values"_a)
        .def("add_categorical", [](D& d, const std::string& name, const std::vector<std::string>& labels,
                                   const std::optional<std::string>& reference) {
            d.catCov[name] = labels;
            if (reference) d.catReference[name] = *reference;
        }, "name"_a, "labels"_a, "reference"_a = nb::none())
        .def("add_events", [](D& d, const IArr1& id, const DArr1& time, const DArr1& amt, const IArr1& cmt,
                              const DArr1& rate, const IArr1& addl, const DArr1& ii, const IArr1& ss) {
            const std::size_t n = id.shape(0);
            for (const std::size_t s : {time.shape(0), amt.shape(0), cmt.shape(0), rate.shape(0), addl.shape(0), ii.shape(0), ss.shape(0)})
                if (s != n) throw std::invalid_argument("add_events: all columns must have the same length");
            for (std::size_t i = 0; i < n; ++i) {
                D::EventRow e; e.id = int(id.data()[i]); e.time = time.data()[i]; e.amt = amt.data()[i];
                e.cmt = int(cmt.data()[i]); e.rate = rate.data()[i]; e.addl = int(addl.data()[i]);
                e.ii = ii.data()[i]; e.ss = int(ss.data()[i]);
                d.events.push_back(e);
            }
        }, "id"_a, "time"_a, "amt"_a, "cmt"_a, "rate"_a, "addl"_a, "ii"_a, "ss"_a)
        .def("check", [](const D& d) { d.check(); })
        .def_prop_ro("n_obs", &D::nObs)
        .def_prop_ro("n_subjects", &D::nSubjects)
        .def_prop_ro("is_hierarchical", &D::isHierarchical)
        .def_prop_ro("subject_ids", [](const D& d) { return vec_to_np(d.subjectIds()); })
        .def_prop_ro("id", [](const D& d) { return vec_to_np(d.id); })
        .def_prop_ro("time", [](const D& d) { return vec_to_np(d.time); })
        .def_prop_ro("y", [](const D& d) { return vec_to_np(d.y); })
        .def_prop_ro("ytype", [](const D& d) { return vec_to_np(d.ytype); })
        .def_prop_ro("cens", [](const D& d) { return vec_to_np(d.cens); })
        .def_prop_ro("limit", [](const D& d) { return vec_to_np(d.limit); })
        .def_prop_ro("group", [](const D& d) { return vec_to_np(d.group); })
        .def_prop_ro("predictor_names", [](const D& d) { return d.predictorOrder; })
        .def_prop_ro("covariate_names", [](const D& d) { return d.covNames(); })
        .def("predictor", [](const D& d, const std::string& n) {
            auto it = d.predictors.find(n);
            if (it == d.predictors.end()) throw std::invalid_argument("predictor '" + n + "' not found");
            return vec_to_np(it->second);
        }, "name"_a)
        .def("covariate", [](const D& d, const std::string& n) {
            auto it = d.cov.find(n);
            if (it == d.cov.end()) throw std::invalid_argument("covariate '" + n + "' not found");
            return np1(it->second.data(), std::size_t(it->second.size()));
        }, "name"_a);

    // ---------------- Model ----------------
    using M = smx::Model;
    nb::class_<mcmckl::ExprProgram>(m, "Expr")
        .def("__init__", [](mcmckl::ExprProgram* self, std::vector<int> op, std::vector<int> a, std::vector<int> b,
                            std::vector<int> c, std::vector<double> cst, std::vector<int> outs) {
            new (self) mcmckl::ExprProgram(std::move(op), std::move(a), std::move(b), std::move(c), std::move(cst), std::move(outs));
        }, "op"_a, "a"_a, "b"_a, "c"_a, "cst"_a, "outs"_a)
        .def_prop_ro("n_out", &mcmckl::ExprProgram::nOut)
        // Gaussian prediction: eval(psi, x[n,stride], cov, doses|None) -> vector(n)
        .def("eval", [](const mcmckl::ExprProgram& p, const Vec& psi, const DArr2& x, const Vec& cov, std::optional<DArr2> doses) {
            std::vector<ode::Dose> dl;
            if (doses) dl = doses_from_np(*doses);
            Vec f(Eigen::Index(x.shape(0)));
            p.eval(psi.data(), int(psi.size()), cov.data(), int(cov.size()), x.data(), int(x.shape(0)), int(x.shape(1)), &dl, f.data());
            return f;
        }, "psi"_a, "x"_a, "cov"_a, "doses"_a = nb::none())
        // single row: eval_row(psi, x_row, cov, y, extra, t|None) -> vector(n_out)
        .def("eval_row", [](const mcmckl::ExprProgram& p, const Vec& psi, const Vec& row, const Vec& cov, double y,
                            const Vec& extra, std::optional<double> t) {
            Vec o(p.nOut());
            double tv = t ? *t : 0.0;
            p.evalRow(psi.data(), int(psi.size()), cov.data(), int(cov.size()), row.size() ? row.data() : nullptr, int(row.size()), y,
                      extra.data(), int(extra.size()), t ? &tv : nullptr, o.data());
            return o;
        }, "psi"_a, "row"_a, "cov"_a, "y"_a = 0.0, "extra"_a = Vec(), "t"_a = nb::none());

    nb::class_<M>(m, "Model")
        .def(nb::init<>())
        .def_rw("n_phi", &M::nPhi)
        .def_prop_rw("error", [](const M& s) { return s.error; }, [](M& s, smx::ErrorModel e) { s.error = e; })
        .def_prop_rw("errors", [](const M& s) { return s.errors; }, [](M& s, const std::vector<smx::ErrorModel>& v) { s.errors = v; })
        .def_prop_rw("transform", [](const M& s) { return s.transform; }, [](M& s, const std::vector<smx::Transform>& v) { s.transform = v; })
        .def_prop_rw("random_effect", [](const M& s) { return from_flags(s.randomEffect); }, [](M& s, const std::vector<int>& v) { s.randomEffect = to_flags(v); })
        .def_prop_rw("omega_block", [](const M& s) { return s.omegaBlock; }, [](M& s, const std::vector<int>& v) { s.omegaBlock = v; })
        .def_prop_rw("covariance_pattern",
            [](const M& s) { std::vector<std::vector<int>> o; for (auto& r : s.covariancePattern) o.push_back(from_flags(r)); return o; },
            [](M& s, const std::vector<std::vector<int>>& v) { s.covariancePattern.clear(); for (auto& r : v) s.covariancePattern.push_back(to_flags(r)); })
        .def_prop_rw("psi_block", [](const M& s) { return s.psiBlock; }, [](M& s, const std::vector<int>& v) { s.psiBlock = v; })
        .def_prop_rw("covariate_model",
            [](const M& s) { std::vector<std::pair<int, std::string>> o; for (auto& t : s.covariateModel) o.emplace_back(t.param, t.covariate); return o; },
            [](M& s, const std::vector<std::pair<int, std::string>>& v) { s.covariateModel.clear(); for (auto& p : v) s.covariateModel.push_back({p.first, p.second}); })
        .def_rw("param_names", &M::paramNames)
        .def_rw("n_extra", &M::nExtra)
        .def_rw("tte_start_col", &M::tteStartCol)
        .def_prop_ro("has_predictor", [](const M& s) { return bool(s.predictSubject); })
        .def_prop_ro("has_custom_loglik", [](const M& s) { return bool(s.logLikObs); })
        // --- vectorized per-subject prediction (Python callback) ---
        // fn(psi, x[n, stride], cov, doses|None) -> array(n)    x = [t, predictors..., ytype]
        .def("set_predict_subject", [](M& s, nb::callable fn) {
            Fn h = hold(fn);
            auto cache = std::make_shared<PredCache>();
            s.predictSubject = [h, cache](const Vec& psi, const mcmckl::ObsView& obs, const Vec& cov, Vec& f) {
                guarded([&] {
                    nb::object pa = cache->psi(psi.data(), std::size_t(psi.size()));
                    nb::object xa = cache->get(cache->xs, obs.x, obs.x, std::size_t(obs.n), std::size_t(obs.stride), true);
                    nb::object ca = cache->get(cache->covs, cov.data(), cov.data(), 1, std::size_t(cov.size()), false);
                    nb::object da = nb::none();
                    const auto& dl = obs.doseList();
                    if (!dl.empty()) da = cache->dosesArr(dl);
                    nb::object r = (*h)(pa, xa, ca, da);
                    auto arr = nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>>(r);
                    if (int(arr.shape(0)) != obs.n)
                        throw std::runtime_error("the model function returned " + std::to_string(arr.shape(0)) +
                                                 " values but the subject has " + std::to_string(obs.n) + " observations");
                    f = Eigen::Map<const Vec>(arr.data(), obs.n);
                    return 0;
                });
            };
        }, "fn"_a)
        // --- expression graphs compiled from Python (no GIL, multithreaded); see expr.hpp and saem/_trace.py ---
        // per-subject Gaussian prediction
        .def("set_expr", [](M& s, std::shared_ptr<mcmckl::ExprProgram> prog) {
            s.predictSubject = [prog](const Vec& psi, const mcmckl::ObsView& obs, const Vec& cov, Vec& f) {
                f.resize(obs.n);
                prog->eval(psi.data(), int(psi.size()), cov.data(), int(cov.size()), obs.x, obs.n, obs.stride,
                           &obs.doseList(), f.data());
            };
        }, "expr"_a)
        // custom likelihood log p(y | psi, x, cov, extra): the graph is the likelihood; `sample` stays in Python
        .def("set_expr_loglik", [](M& s, std::shared_ptr<mcmckl::ExprProgram> prog, std::optional<nb::callable> sample, int n_extra) {
            if (prog->nOut() != 1) throw std::invalid_argument("loglik: the graph must have exactly one output");
            s.logLikObs = [prog](const Vec& psi, Xs x, const Vec& cov, double y, const Vec& extra) -> double {
                double o = 0.0;
                prog->evalRow(psi.data(), int(psi.size()), cov.data(), int(cov.size()), x.data(), int(x.size()), y,
                              extra.data(), int(extra.size()), nullptr, &o);
                return o;
            };
            s.nExtra = n_extra;
            if (sample) {
                Fn hs = hold(*sample);
                s.simulateObs = [hs](const Vec& psi, Xs x, const Vec& cov, const Vec& extra, std::mt19937& rng) -> double {
                    return guarded([&] {
                        const std::int64_t seed = static_cast<std::int64_t>(rng());
                        nb::object r = (*hs)(np1(psi.data(), std::size_t(psi.size())), np1(x.data(), x.size()),
                                             np1(cov.data(), std::size_t(cov.size())), np1(extra.data(), std::size_t(extra.size())), seed);
                        return nb::cast<double>(r);
                    });
                };
            } else s.simulateObs = nullptr;
        }, "expr"_a, "sample"_a = nb::none(), "n_extra"_a = 0)
        // non-continuous families with f (and g) compiled: same signatures as set_family
        .def("set_expr_family", [](M& s, const std::string& kind, std::shared_ptr<mcmckl::ExprProgram> pf,
                                   std::shared_ptr<mcmckl::ExprProgram> pg, int tte_start_col) {
            auto scalar3 = [pf](const Vec& psi, Xs x, const Vec& cov) -> double {
                double o = 0.0;
                pf->evalRow(psi.data(), int(psi.size()), cov.data(), int(cov.size()), x.data(), int(x.size()), 0.0, nullptr, 0, nullptr, &o);
                return o;
            };
            mcmckl::lik::Lik L;
            if (kind == "bernoulli") L = mcmckl::lik::bernoulli(scalar3);
            else if (kind == "bernoulli_logit") L = mcmckl::lik::bernoulliLogit(scalar3);
            else if (kind == "poisson") L = mcmckl::lik::poisson(scalar3);
            else if (kind == "negbin") L = mcmckl::lik::negbin(scalar3);
            else if (kind == "zip") {
                if (pf->nOut() != 2) throw std::invalid_argument("zip: f must return (lambda, pi)");
                L = mcmckl::lik::zip([pf](const Vec& psi, Xs x, const Vec& cov) -> std::pair<double, double> {
                    double o[2];
                    pf->evalRow(psi.data(), int(psi.size()), cov.data(), int(cov.size()), x.data(), int(x.size()), 0.0, nullptr, 0, nullptr, o);
                    return {o[0], o[1]};
                });
            } else if (kind == "ordinal") {
                if (!pg) throw std::invalid_argument("ordinal: g(psi) -> cut points is required");
                L = mcmckl::lik::ordinal(scalar3, [pg](const Vec& psi) -> Vec {
                    Vec o(pg->nOut());
                    pg->evalRow(psi.data(), int(psi.size()), nullptr, 0, nullptr, 0, 0.0, nullptr, 0, nullptr, o.data());
                    return o;
                });
            } else if (kind == "categorical") {
                L = mcmckl::lik::categorical([pf](const Vec& psi, Xs x, const Vec& cov) -> Vec {
                    Vec o(pf->nOut());
                    pf->evalRow(psi.data(), int(psi.size()), cov.data(), int(cov.size()), x.data(), int(x.size()), 0.0, nullptr, 0, nullptr, o.data());
                    return o;
                });
            } else if (kind == "tte") {
                if (!pg) throw std::invalid_argument("tte: g(psi, t, x, cov) -> cumulative hazard is required");
                auto mk = [](std::shared_ptr<mcmckl::ExprProgram> p) {
                    return [p](const Vec& psi, double t, Xs x, const Vec& cov) -> double {
                        double o = 0.0;
                        p->evalRow(psi.data(), int(psi.size()), cov.data(), int(cov.size()), x.data(), int(x.size()), 0.0, nullptr, 0, &t, &o);
                        return o;
                    };
                };
                auto haz = mk(pf);
                auto cum = mk(pg);
                L = mcmckl::lik::tte(haz, cum, tte_start_col);
                s.tteCumHazard = cum;
                s.tteStartCol = tte_start_col;
            } else throw std::invalid_argument("unknown family: '" + kind + "'");
            s.logLikObs = L.logLik;
            s.simulateObs = L.sample;
            s.nExtra = L.nExtra;
        }, "kind"_a, "f"_a, "g"_a = nb::none(), "tte_start_col"_a = -1)
        // --- compiled model (no GIL, multithreaded) ---
        .def("set_builtin", [](M& s, const std::string& name, int x_col, int dose_cov) {
            s.predictSubject = makeBuiltin(name, x_col, dose_cov);
            s.nPhi = builtinTable().at(name).nPhi;
        }, "name"_a, "x_col"_a = 0, "dose_cov"_a = -1)
        // --- custom likelihood: fn(psi, x_row, cov, y, extra) -> float ---
        //     sample(psi, x_row, cov, extra, seed) -> float   (optional)
        .def("set_loglik", [](M& s, nb::callable fn, std::optional<nb::callable> sample, int n_extra) {
            Fn h = hold(fn);
            s.logLikObs = [h](const Vec& psi, Xs x, const Vec& cov, double y, const Vec& extra) -> double {
                return guarded([&] {
                    nb::object r = (*h)(np1(psi.data(), std::size_t(psi.size())), np1(x.data(), x.size()),
                                        np1(cov.data(), std::size_t(cov.size())), y, np1(extra.data(), std::size_t(extra.size())));
                    return nb::cast<double>(r);
                });
            };
            s.nExtra = n_extra;
            if (sample) {
                Fn hs = hold(*sample);
                s.simulateObs = [hs](const Vec& psi, Xs x, const Vec& cov, const Vec& extra, std::mt19937& rng) -> double {
                    return guarded([&] {
                        const std::int64_t seed = static_cast<std::int64_t>(rng());
                        nb::object r = (*hs)(np1(psi.data(), std::size_t(psi.size())), np1(x.data(), x.size()),
                                             np1(cov.data(), std::size_t(cov.size())), np1(extra.data(), std::size_t(extra.size())), seed);
                        return nb::cast<double>(r);
                    });
                };
            } else s.simulateObs = nullptr;
        }, "fn"_a, "sample"_a = nb::none(), "n_extra"_a = 0)
        // --- non-continuous families (see likelihoods.hpp) ---
        //   bernoulli: f -> P(y=1)        bernoulli_logit: f -> eta        poisson: f -> lambda
        //   negbin: f -> mean (1 extra: log r)       zip: f -> (lambda, pi)
        //   ordinal: f -> eta, g(psi) -> cut points       categorical: f -> probability vector
        //   tte: f(psi, t, x, cov) -> hazard, g(psi, t, x, cov) -> cumulative hazard
        .def("set_family", [](M& s, const std::string& kind, nb::callable f, std::optional<nb::callable> g, int tte_start_col) {
            Fn hf = hold(f);
            Fn hg = g ? hold(*g) : Fn();
            auto scalar3 = [hf](const Vec& psi, Xs x, const Vec& cov) -> double {
                return guarded([&] {
                    return nb::cast<double>((*hf)(np1(psi.data(), std::size_t(psi.size())), np1(x.data(), x.size()), np1(cov.data(), std::size_t(cov.size()))));
                });
            };
            mcmckl::lik::Lik L;
            if (kind == "bernoulli") L = mcmckl::lik::bernoulli(scalar3);
            else if (kind == "bernoulli_logit") L = mcmckl::lik::bernoulliLogit(scalar3);
            else if (kind == "poisson") L = mcmckl::lik::poisson(scalar3);
            else if (kind == "negbin") L = mcmckl::lik::negbin(scalar3);
            else if (kind == "zip") {
                L = mcmckl::lik::zip([hf](const Vec& psi, Xs x, const Vec& cov) -> std::pair<double, double> {
                    return guarded([&] {
                        nb::object r = (*hf)(np1(psi.data(), std::size_t(psi.size())), np1(x.data(), x.size()), np1(cov.data(), std::size_t(cov.size())));
                        return std::pair<double, double>(nb::cast<double>(r[0]), nb::cast<double>(r[1]));
                    });
                });
            } else if (kind == "ordinal") {
                if (!hg) throw std::invalid_argument("ordinal: g(psi) -> cut points is required");
                L = mcmckl::lik::ordinal(scalar3, [hg](const Vec& psi) -> Vec {
                    return guarded([&] {
                        auto a = nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>>((*hg)(np1(psi.data(), std::size_t(psi.size()))));
                        return Vec(Eigen::Map<const Vec>(a.data(), Eigen::Index(a.shape(0))));
                    });
                });
            } else if (kind == "categorical") {
                L = mcmckl::lik::categorical([hf](const Vec& psi, Xs x, const Vec& cov) -> Vec {
                    return guarded([&] {
                        auto a = nb::cast<nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig>>(
                            (*hf)(np1(psi.data(), std::size_t(psi.size())), np1(x.data(), x.size()), np1(cov.data(), std::size_t(cov.size()))));
                        return Vec(Eigen::Map<const Vec>(a.data(), Eigen::Index(a.shape(0))));
                    });
                });
            } else if (kind == "tte") {
                if (!hg) throw std::invalid_argument("tte: g(psi, t, x, cov) -> cumulative hazard is required");
                auto haz = [hf](const Vec& psi, double t, Xs x, const Vec& cov) -> double {
                    return guarded([&] {
                        return nb::cast<double>((*hf)(np1(psi.data(), std::size_t(psi.size())), t, np1(x.data(), x.size()), np1(cov.data(), std::size_t(cov.size()))));
                    });
                };
                auto cum = [hg](const Vec& psi, double t, Xs x, const Vec& cov) -> double {
                    return guarded([&] {
                        return nb::cast<double>((*hg)(np1(psi.data(), std::size_t(psi.size())), t, np1(x.data(), x.size()), np1(cov.data(), std::size_t(cov.size()))));
                    });
                };
                L = mcmckl::lik::tte(haz, cum, tte_start_col);
                s.tteCumHazard = cum;
                s.tteStartCol = tte_start_col;
            } else throw std::invalid_argument("unknown family: '" + kind + "'");
            s.logLikObs = L.logLik;
            s.simulateObs = L.sample;
            s.nExtra = L.nExtra;
        }, "kind"_a, "f"_a, "g"_a = nb::none(), "tte_start_col"_a = -1)
        // direct evaluation (for predict() with new data): f(psi, x[n,stride], cov, doses|None)
        .def("eval", [](const M& s, const Vec& psi, const DArr2& x, const Vec& cov, std::optional<DArr2> doses) {
            if (!s.predictSubject) throw std::invalid_argument("the model has no prediction function");
            std::vector<ode::Dose> dl;
            if (doses) dl = doses_from_np(*doses);
            mcmckl::ObsView ov;
            ov.n = int(x.shape(0)); ov.stride = int(x.shape(1)); ov.x = x.data(); ov.doses = &dl;
            Vec f(ov.n);
            G().reset();
            s.predictSubject(psi, ov, cov, f);
            G().rethrow_if_failed();
            return f;
        }, "psi"_a, "x"_a, "cov"_a, "doses"_a = nb::none());

    // ---------------- Results ----------------
    using R = smx::Results;
    nb::class_<R>(m, "Results")
        .def_ro("theta", &R::theta)
        .def_prop_ro("theta_trace", [](const R& r) {
            nb::list out;
            for (const Vec& v : r.thetaTrace) out.append(np1(v.data(), std::size_t(v.size())));
            return out;
        }, "theta at every SAEM iteration (list of 1-D arrays)")
        .def_ro("coef_names", &R::coefNames)
        .def_ro("beta", &R::beta)
        .def_ro("psi_pop", &R::psiPop)
        .def_ro("coef", &R::coef)
        .def_ro("omega", &R::Omega)
        .def_ro("sd", &R::sd)
        .def_ro("error_pars", &R::errPres)
        .def_ro("extra", &R::extra)
        .def_ro("hierarchical", &R::hierarchical)
        .def_ro("psi_cluster", &R::Psi)
        .def_ro("sd_cluster", &R::sdCluster)
        .def_ro("n_clusters", &R::nClusters)
        .def_ro("subject_cluster", &R::subjectCluster)
        .def_ro("xi", &R::xi)
        .def_ro("phi", &R::phi)
        .def_ro("ebe", &R::ebe)
        .def_ro("phi_cond_sd", &R::phiCondSd)
        .def_ro("psi_ebe", &R::psiEbe)
        .def_ro("xi_cond_mean", &R::xiCondMean)
        .def_ro("xi_cond_sd", &R::xiCondSd)
        .def_ro("cond_computed", &R::condComputed)
        .def_ro("cond_ess_min", &R::condEssMin)
        .def_ro("cond_rhat_max", &R::condRhatMax)
        .def_ro("loglik", &R::logLik)
        .def_ro("loglik_computed", &R::llComputed)
        .def_ro("loglik_method", &R::llMethodUsed)
        .def_ro("loglik_lin", &R::logLikLin)
        .def_ro("se", &R::se)
        .def_ro("se_computed", &R::seComputed)
        .def_ro("se_method", &R::seMethodUsed)
        .def_ro("se_coef", &R::seCoef)
        .def_ro("se_psi_pop", &R::sePsiPop)
        .def_ro("se_sd", &R::seSd)
        .def_ro("se_sd_cluster", &R::seSdCluster)
        .def_ro("corr", &R::Corr)
        .def_ro("se_omega", &R::seOmega)
        .def_ro("se_corr", &R::seCorr)
        .def_ro("se_err", &R::seErr)
        .def_ro("se_extra", &R::seExtra)
        .def_ro("cov_theta", &R::covTheta)
        .def_ro("convergence", &R::convergence)
        .def_ro("warnings", &R::warnings)
        .def_ro("n_subjects", &R::nSubjects)
        .def_ro("n_obs", &R::nObs)
        .def_prop_ro("data", [](const R& r) -> const D& { return r.data; }, nb::rv_policy::reference_internal)
        .def_prop_ro("model", [](const R& r) -> const M& { return r.model; }, nb::rv_policy::reference_internal)
        .def_prop_ro("control", [](const R& r) -> const C& { return r.control; }, nb::rv_policy::reference_internal)
        .def_prop_ro("fitted_pop", [](const R& r) { return vec_to_np(r.fittedPop); })
        .def_prop_ro("fitted_ind", [](const R& r) { return vec_to_np(r.fittedInd); })
        .def("to_text", [](const R& r) { std::ostringstream os; r.print(os); return os.str(); });

    // ---------------- fitting and simulation ----------------
    m.def("fit", [](const D& data, const M& model, const C& control, bool release_gil) {
        // models with Python callbacks do not support threads (the GIL would serialize them anyway)
        C ctrl = control;
        if (!release_gil) ctrl.nThreads = 1;
        G().reset();
        CoutRedirect redirect;
        R res;
        try {
            if (release_gil) { nb::gil_scoped_release rel; res = smx::fit(data, model, ctrl); }
            else res = smx::fit(data, model, ctrl);
        } catch (...) {
            // if a Python callback failed, that is the real error (not the engine's derivative)
            G().rethrow_if_failed();
            throw;
        }
        G().rethrow_if_failed();
        return res;
    }, "data"_a, "model"_a, "control"_a, "release_gil"_a = false);

    m.def("simulate", [](const R& res, unsigned seed, int nrep, double lloq, double uloq, bool new_random_effects, bool release_gil) {
        smx::SimOptions o; o.seed = seed; o.nrep = nrep; o.lloq = lloq; o.uloq = uloq; o.newRandomEffects = new_random_effects;
        G().reset();
        std::vector<D> out;
        try {
            if (release_gil) { nb::gil_scoped_release rel; out = smx::simulate(res, o); }
            else out = smx::simulate(res, o);
        } catch (...) { G().rethrow_if_failed(); throw; }
        G().rethrow_if_failed();
        return out;
    }, "res"_a, "seed"_a = 1u, "nrep"_a = 1, "lloq"_a = std::numeric_limits<double>::quiet_NaN(),
       "uloq"_a = std::numeric_limits<double>::quiet_NaN(), "new_random_effects"_a = true, "release_gil"_a = false);

    m.def("simulate_events", [](const R& res, double tmax, bool repeated, unsigned seed, int nrep, bool new_random_effects) {
        smx::SimOptions o; o.seed = seed; o.nrep = nrep; o.newRandomEffects = new_random_effects;
        G().reset();
        std::vector<D> out;
        try { out = smx::simulateEvents(res, tmax, repeated, o); }
        catch (...) { G().rethrow_if_failed(); throw; }
        G().rethrow_if_failed();
        return out;
    }, "res"_a, "tmax"_a, "repeated"_a = false, "seed"_a = 1u, "nrep"_a = 1, "new_random_effects"_a = true);

    m.def("builtin_models", []() {
        std::map<std::string, int> o;
        for (auto& kv : builtinTable()) o[kv.first] = kv.second.nPhi;
        return o;
    });
    m.def("hardware_threads", []() { return mcmckl::hardwareThreads(); });
}
