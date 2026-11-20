// ============================================================================
// @file ode.hpp
// @brief ODE integrators for PK/PD models with no external dependencies.
//
// This header provides the following functionality:
//   * Dormand-Prince 5(4) with adaptive step size (for non-stiff systems).
//   * TR-BDF2 (L-stable, second order, simplified Newton with a numerical
//     Jacobian and step control by duplication) for STIFF systems.
//   * Method::Auto: starts with RK45 and switches to TR-BDF2 when stiffness is
//     detected (step budget exhausted).
//   * Options::linear = true (EXACT FAST PATH): the system is dy/dt = A y + b,
//     with A and b constant (linear compartmental models). A and b are extracted
//     from the rhs (n+1 evaluations, with a linearity check) and the solution is
//     propagated with the MATRIX EXPONENTIAL (machine precision, no adaptive
//     steps): boluses, infusions, ADDL and STEADY STATE
//     (I - e^{A tau})^{-1}. Typically 10-50 times faster than RK45 and much more
//     so at steady state.
//   * integrate(): states at observation times, with dose EVENTS:
//       bolus     (duration = 0) : y[cmt] += amount at `time`
//       infusion  (duration > 0) : constant rate amount/duration over
//                                  [time, time+duration]
//     Convention: an observation at the instant of a dose sees the state
//     AFTER applying the dose.
//   * STEADY STATE (Dose::ss, with interval `ii`): the dose repeats every `ii`
//     until steady state (Newton with a numerical Jacobian on the one-interval
//     map; fallback: fixed-point iteration).
//       ss = 1 : the state is REPLACED by the pre-dose steady state, then the
//                dose is given
//       ss = 2 : the pre-dose steady state is ADDED to the current state
//                (superposition)
//     A steady-state infusion must last less than `ii`.
//   * DOSES WITH MODELED RATE / DURATION (NONMEM RATE = -1 / -2):
//     Dose::modeled = -1 (the rate R1 is a parameter) or -2 (the duration D1 is a
//     parameter); they are resolved with resolveDoses().
//
// Typical use inside a thread-safe predict (the integrators have no global
// state):
//     ode::Rhs rhs = [&](double, const Vec& y, Vec& dy) { dy[0] = -ka*y[0]; dy[1] = ka*y[0] - k*y[1]; };
//     Mat Y = ode::integrate(rhs, Vec::Zero(2), 0.0, times, {{0.0, 0, 100.0, 0.0}});
//
// Requires: Eigen3, C++20.
// ============================================================================

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ode {

/// @brief Dense vector type used for states and right-hand sides.
using Vec = Eigen::VectorXd;
/// @brief Dense matrix type used for Jacobians and matrix exponentials.
using Mat = Eigen::MatrixXd;
/// @brief Right-hand side of the ODE: computes dydt at time t for state y.
using Rhs = std::function<void(double t, const Vec& y, Vec& dydt)>;

/// @brief Integration method selection.
enum class Method {
    Auto,  ///< Start with RK45 and switch to the stiff integrator on stiffness.
    RK45,  ///< Dormand-Prince 5(4), non-stiff.
    Stiff  ///< TR-BDF2, L-stable, for stiff systems.
};

/// @brief Integrator configuration.
struct Options {
    double rtol = 1e-8;                 ///< Relative tolerance.
    double atol = 1e-10;                ///< Absolute tolerance.
    double hmax = std::numeric_limits<double>::infinity(); ///< Maximum step size.
    double h0   = 0.0;                  ///< Initial step; 0 selects it automatically.
    int    maxSteps = 200000;           ///< Step budget for the selected method.
    int    stiffSwitchSteps = 4000;     ///< Auto: RK45 steps before declaring stiffness.
    Method method = Method::Auto;       ///< Integration method.
    bool   linear = false;              ///< RHS is affine linear and autonomous: use the matrix exponential (see above).
    bool   linearModal = true;          ///< With linear: propagate in modal coordinates (real, distinct
                                        ///< eigenvalues) instead of segment-wise expm; fall back to expm if invalid.
};

/// @brief A dosing event.
struct Dose {
    double time;                        ///< Dose time.
    int    cmt;                         ///< Target compartment index.
    double amount;                      ///< Dose amount.
    double duration = 0.0;              ///< 0 = bolus.
    double ii = 0.0;                    ///< Dosing interval (for steady state).
    int    ss = 0;                      ///< 0 = no steady state; 1 = replace state; 2 = add to state.
    int    modeled = 0;                 ///< 0 = duration given; -1 = modeled rate; -2 = modeled duration.
};

/// @brief Resolves doses with a modeled rate or duration.
///
/// For each dose with `modeled != 0`, `value(cmt)` returns R1 (when
/// `modeled == -1`, giving `duration = amount / R1`) or D1 (when
/// `modeled == -2`, giving `duration = D1`) for the dose compartment. Resolved
/// doses are returned with `modeled` reset to 0.
///
/// @param doses Input doses; may contain modeled rate/duration entries.
/// @param value Callback mapping a compartment index to R1 or D1.
/// @return A copy of the doses with modeled entries resolved.
/// @throws std::invalid_argument If the supplied rate or duration is not finite
///         or not strictly positive.
/// @note This function has no global state and is thread-safe when `value` is.
inline std::vector<Dose> resolveDoses(std::vector<Dose> doses, const std::function<double(int cmt)>& value) {
    for (auto& d : doses) {
        if (d.modeled == 0) continue;
        const double v = value(d.cmt);
        if (!(v > 0.0) || !std::isfinite(v)) throw std::invalid_argument("resolveDoses: modeled rate/duration is not positive");
        d.duration = (d.modeled == -1) ? d.amount / v : v;
        d.modeled = 0;
    }
    return doses;
}

/// @brief Integration statistics accumulated across calls.
struct Stats {
    long nsteps = 0;                    ///< Accepted steps.
    long nrejected = 0;                 ///< Rejected steps.
    long nfev = 0;                      ///< Right-hand-side evaluations.
    bool usedStiff = false;             ///< Whether the stiff integrator was used.
};

namespace detail {

inline double wnorm(const Vec& e, const Vec& y0, const Vec& y1, double rtol, double atol) {
    double s = 0.0;
    for (int i = 0; i < e.size(); ++i) {
        const double sc = atol + rtol * std::max(std::abs(y0[i]), std::abs(y1[i]));
        s += (e[i] / sc) * (e[i] / sc);
    }
    return std::sqrt(s / double(std::max<int>(1, int(e.size()))));
}

// Initial step selection (Hairer, Norsett & Wanner).
inline double initialStep(const Rhs& f, double t, const Vec& y, const Vec& f0, double dir, const Options& o, Stats& st) {
    Vec sc = (o.atol + o.rtol * y.cwiseAbs().array()).matrix();
    const double d0 = (y.array() / sc.array()).matrix().norm() / std::sqrt(double(y.size()));
    const double d1 = (f0.array() / sc.array()).matrix().norm() / std::sqrt(double(y.size()));
    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
    Vec y1 = y + dir * h0 * f0, f1(y.size());
    f(t + dir * h0, y1, f1); ++st.nfev;
    const double d2 = (((f1 - f0).array() / sc.array()).matrix().norm() / std::sqrt(double(y.size()))) / h0;
    const double h1 = (std::max(d1, d2) <= 1e-15) ? std::max(1e-6, h0 * 1e-3) : std::pow(0.01 / std::max(d1, d2), 0.2);
    return std::min(100.0 * h0, h1);
}

// ---------------------------------------------------------------------------
// Dormand-Prince 5(4), FSAL. Advances from t0 to t1 (t1 > t0). Returns false if
// the step budget is exhausted (y, t, h remain at the last accepted point).
// ---------------------------------------------------------------------------
inline bool rk45(const Rhs& f, double& t, Vec& y, double t1, double& h, const Options& o,
                 Stats& st, long maxSteps) {
    const int n = static_cast<int>(y.size());
    Vec k1(n), k2(n), k3(n), k4(n), k5(n), k6(n), k7(n), yn(n), err(n), tmp(n);
    f(t, y, k1); ++st.nfev;
    if (h <= 0.0) h = initialStep(f, t, y, k1, 1.0, o, st);
    long steps = 0;
    while (t < t1) {
        if (steps++ >= maxSteps) return false;
        if (t + h > t1) h = t1 - t;
        h = std::min(h, o.hmax);
        tmp = y + h * (1.0 / 5.0) * k1;                                   f(t + h / 5.0, tmp, k2);
        tmp = y + h * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2);               f(t + 3.0 * h / 10.0, tmp, k3);
        tmp = y + h * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3); f(t + 4.0 * h / 5.0, tmp, k4);
        tmp = y + h * (19372.0 / 6561.0 * k1 - 25360.0 / 2187.0 * k2 + 64448.0 / 6561.0 * k3 - 212.0 / 729.0 * k4);
        f(t + 8.0 * h / 9.0, tmp, k5);
        tmp = y + h * (9017.0 / 3168.0 * k1 - 355.0 / 33.0 * k2 + 46732.0 / 5247.0 * k3 + 49.0 / 176.0 * k4 - 5103.0 / 18656.0 * k5);
        f(t + h, tmp, k6);
        yn = y + h * (35.0 / 384.0 * k1 + 500.0 / 1113.0 * k3 + 125.0 / 192.0 * k4 - 2187.0 / 6784.0 * k5 + 11.0 / 84.0 * k6);
        f(t + h, yn, k7);
        st.nfev += 6;
        err = h * (71.0 / 57600.0 * k1 - 71.0 / 16695.0 * k3 + 71.0 / 1920.0 * k4 - 17253.0 / 339200.0 * k5 + 22.0 / 525.0 * k6 - 1.0 / 40.0 * k7);
        const double e = wnorm(err, y, yn, o.rtol, o.atol);
        if (!std::isfinite(e)) { h *= 0.2; ++st.nrejected; if (h < 1e-14 * std::max(1.0, std::abs(t))) throw std::runtime_error("ode: step collapsed (non-finite)"); continue; }
        if (e <= 1.0) {
            t += h; y = yn; k1 = k7; ++st.nsteps;
            h *= std::min(5.0, std::max(0.2, 0.9 * std::pow(std::max(e, 1e-10), -0.2)));
        } else {
            ++st.nrejected;
            h *= std::max(0.2, 0.9 * std::pow(e, -0.2));
            if (h < 1e-14 * std::max(1.0, std::abs(t))) throw std::runtime_error("ode: step collapsed");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// TR-BDF2 (gamma = 2 - sqrt 2). One step of size h: returns false if Newton does
// not converge.
// ---------------------------------------------------------------------------
inline bool trbdf2Step(const Rhs& f, double t, const Vec& y, double h, Vec& ynew, const Options& o, Stats& st) {
    const int n = static_cast<int>(y.size());
    const double gam = 2.0 - std::sqrt(2.0);
    const double d = 0.5 * gam;                                  // = (1-gam)/(2-gam)
    const double a1 = 1.0 / (gam * (2.0 - gam)), a0 = (1.0 - gam) * (1.0 - gam) / (gam * (2.0 - gam));
    Vec f0(n); f(t, y, f0); ++st.nfev;
    // Numerical Jacobian at (t, y).
    Mat J(n, n); Vec fp(n), yp = y;
    for (int j = 0; j < n; ++j) {
        const double dj = std::sqrt(std::numeric_limits<double>::epsilon()) * std::max(1.0, std::abs(y[j]));
        yp[j] = y[j] + dj; f(t, yp, fp); ++st.nfev;
        J.col(j) = (fp - f0) / dj; yp[j] = y[j];
    }
    Eigen::PartialPivLU<Mat> lu(Mat::Identity(n, n) - d * h * J);
    auto newton = [&](auto&& resid, Vec z) -> std::pair<bool, Vec> {
        for (int it = 0; it < 10; ++it) {
            const Vec r = resid(z);
            const Vec dz = lu.solve(r);
            z -= dz;
            if (!z.allFinite()) return {false, z};
            if (wnorm(dz, y, z, o.rtol, o.atol) < 1e-3) return {true, z};
        }
        return {false, z};
    };
    // Stage 1 (trapezoidal up to t + gam h): Y - y - (gam h/2)(f0 + f(Y)) = 0.
    Vec fz(n);
    auto r1 = [&](const Vec& z) { f(t + gam * h, z, fz); ++st.nfev; return Vec(z - y - 0.5 * gam * h * (f0 + fz)); };
    auto [ok1, Y] = newton(r1, y + gam * h * f0);
    if (!ok1) return false;
    // Stage 2 (BDF2): y1 - [a1 Y - a0 y + d h f(y1)] = 0.
    auto r2 = [&](const Vec& z) { f(t + h, z, fz); ++st.nfev; return Vec(z - (a1 * Y - a0 * y + d * h * fz)); };
    auto [ok2, y1] = newton(r2, Y);
    if (!ok2) return false;
    ynew = y1;
    return true;
}

inline bool stiff(const Rhs& f, double& t, Vec& y, double t1, double& h, const Options& o, Stats& st, long maxSteps) {
    st.usedStiff = true;
    Vec f0(y.size());
    if (h <= 0.0) { f(t, y, f0); ++st.nfev; h = initialStep(f, t, y, f0, 1.0, o, st); }
    long steps = 0;
    Vec yfull, yhalf, y2;
    while (t < t1) {
        if (steps++ >= maxSteps) return false;
        if (t + h > t1) h = t1 - t;
        h = std::min(h, o.hmax);
        // One step of h vs. two steps of h/2 (error estimate by duplication).
        const bool a = trbdf2Step(f, t, y, h, yfull, o, st);
        bool b = false;
        if (a) { b = trbdf2Step(f, t, y, 0.5 * h, yhalf, o, st) && trbdf2Step(f, t + 0.5 * h, yhalf, 0.5 * h, y2, o, st); }
        if (!a || !b) { h *= 0.25; ++st.nrejected; if (h < 1e-15 * std::max(1.0, std::abs(t))) throw std::runtime_error("ode: Newton did not converge (step collapsed)"); continue; }
        const Vec err = (y2 - yfull) / 3.0;                       // Order 2: local error ~ (y_h - y_{h/2}) / (2^2 - 1).
        const double e = wnorm(err, y, y2, o.rtol, o.atol);
        if (e <= 1.0 || !std::isfinite(e)) {
            if (std::isfinite(e)) { t += h; y = y2 + err; ++st.nsteps; h *= std::min(4.0, std::max(0.25, 0.9 * std::pow(std::max(e, 1e-10), -1.0 / 3.0))); }
            else { h *= 0.25; ++st.nrejected; }
        } else {
            ++st.nrejected;
            h *= std::max(0.25, 0.9 * std::pow(e, -1.0 / 3.0));
            if (h < 1e-15 * std::max(1.0, std::abs(t))) throw std::runtime_error("ode: step collapsed");
        }
    }
    return true;
}

} // namespace detail

/// @brief Advances the state y from t0 to t1 (t1 >= t0).
///
/// The step size `h` is an in/out parameter and is preserved between calls.
/// The method is selected by `o.method`; `Method::Auto` starts with RK45 and
/// switches to the stiff integrator if the RK45 step budget is exhausted.
///
/// @param f Right-hand side of the ODE.
/// @param t0 Initial time.
/// @param y State vector, updated in place.
/// @param t1 Final time.
/// @param h Step size, updated in place.
/// @param o Integrator options.
/// @param stats Optional statistics sink; a local accumulator is used if null.
/// @throws std::invalid_argument If t1 < t0.
/// @throws std::runtime_error If the step budget is exhausted or the step
///         collapses.
/// @note This function has no global state and is thread-safe as long as `f` is.
inline void advance(const Rhs& f, double t0, Vec& y, double t1, double& h, const Options& o, Stats* stats = nullptr) {
    Stats local; Stats& st = stats ? *stats : local;
    if (t1 < t0) throw std::invalid_argument("ode::advance: t1 < t0");
    double t = t0;
    if (t1 == t0) return;
    if (o.method == Method::Stiff) {
        if (!detail::stiff(f, t, y, t1, h, o, st, o.maxSteps)) throw std::runtime_error("ode: maxSteps exhausted (stiff)");
        return;
    }
    const long budget = (o.method == Method::Auto) ? o.stiffSwitchSteps : o.maxSteps;
    if (detail::rk45(f, t, y, t1, h, o, st, budget)) return;
    if (o.method == Method::RK45) throw std::runtime_error("ode: maxSteps exhausted (RK45)");
    // Auto: the problem is stiff -> finish with TR-BDF2 from the last accepted point.
    h = 0.0;
    if (!detail::stiff(f, t, y, t1, h, o, st, o.maxSteps)) throw std::runtime_error("ode: maxSteps exhausted (stiff)");
}

/// @brief Matrix exponential.
///
/// Uses order-8 Pade approximation with scaling and squaring
/// (Golub & Van Loan, 11.3).
///
/// @param A Square matrix.
/// @return The matrix exponential of A.
/// @note This function has no global state and is thread-safe.
inline Mat expm(const Mat& A) {
    const int n = static_cast<int>(A.rows());
    const double nrm = A.cwiseAbs().rowwise().sum().maxCoeff();
    int sq = 0;
    if (nrm > 0.5) sq = std::max(0, static_cast<int>(std::ceil(std::log2(nrm / 0.5))));
    const Mat As = A / std::ldexp(1.0, sq);
    const int q = 8;
    Mat X = Mat::Identity(n, n), N = Mat::Identity(n, n), D = Mat::Identity(n, n);
    double c = 1.0;
    for (int k = 1; k <= q; ++k) {
        c = c * double(q - k + 1) / (double(2 * q - k + 1) * double(k));
        X = As * X;
        N += c * X;
        D += ((k % 2) ? -c : c) * X;
    }
    Mat F = D.partialPivLu().solve(N);
    for (int i = 0; i < sq; ++i) F = F * F;
    return F;
}

namespace detail {

// Maximum capacity (compartments) of the modal path: data lives in stack arrays
// with no dynamic allocation.
constexpr int kModalMax = 4;
// Maximum admitted condition number of V (error ~ eps * cond); above this the
// expm path is used.
constexpr double kModalMaxCond = 1e4;

// Affine linear system y' = A y + b (autonomous).
//   * EXPM PATH  : A (n x n) and b (n) as dynamic Eigen objects.
//   * MODAL PATH : A = V diag(lam) Vi with REAL, well-separated eigenvalues;
//     everything lives in stack arrays (for PK compartmental models, A has only
//     real eigenvalues: compartment matrix).
struct LinSys {
    Mat A; Vec b;                       // expm path (empty if only modal and b == 0)
    int n = 0;
    bool modal = false;
    double lam[kModalMax] = {};
    double V[kModalMax][kModalMax] = {};
    double Vi[kModalMax][kModalMax] = {};
    double bm[kModalMax] = {};          // b (modal path)
};

// Affine linear system y' = A y + b (autonomous) extracted from the rhs; throws
// if the rhs is not of this form.
inline LinSys extractLinear(const Rhs& rhs, int n, double t0) {
    LinSys L; L.A.resize(n, n); L.n = n;
    Vec z = Vec::Zero(n), f0(n), fj(n);
    rhs(t0, z, f0);
    L.b = f0;
    for (int j = 0; j < n; ++j) { Vec e = Vec::Zero(n); e[j] = 1.0; rhs(t0, e, fj); L.A.col(j) = fj - f0; }
    // Check f(v) = A v + b at two distinct times.
    Vec v(n); for (int j = 0; j < n; ++j) v[j] = 0.37 + 0.21 * j;
    for (double dt : {0.0, 1.7}) {
        rhs(t0 + dt, v, fj);
        const Vec pred = L.A * v + L.b;
        if (!((fj - pred).cwiseAbs().maxCoeff() <= 1e-8 * (1.0 + pred.cwiseAbs().maxCoeff())))
            throw std::invalid_argument("ode::Options::linear: rhs is not affine linear and autonomous");
    }
    return L;
}

// Validates an already populated modal decomposition (lam, V, Vi): separated
// eigenvalues and a well-conditioned V. If it fails, the system uses the expm
// path.
inline bool acceptModal(LinSys& L) {
    const int n = L.n;
    if (n < 1 || n > kModalMax) return false;
    double scale = 0.0;
    for (int i = 0; i < n; ++i) { if (!std::isfinite(L.lam[i])) return false; scale = std::max(scale, std::abs(L.lam[i])); }
    if (!(scale > 0.0)) return false;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j)
        if (!(std::abs(L.lam[i] - L.lam[j]) > 1e-4 * scale)) return false;
    // Normalize the columns of V (1-norm = 1) and the rows of Vi accordingly, so
    // the condition number becomes invariant to the arbitrary eigenvector scale.
    double nvi = 0.0;
    for (int j = 0; j < n; ++j) {
        double cv = 0.0;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(L.V[i][j]) || !std::isfinite(L.Vi[i][j])) return false;
            cv += std::abs(L.V[i][j]);
        }
        if (!(cv > 0.0) || !std::isfinite(cv)) return false;
        for (int i = 0; i < n; ++i) { L.V[i][j] /= cv; L.Vi[j][i] *= cv; }
    }
    for (int j = 0; j < n; ++j) {
        double cvi = 0.0;
        for (int i = 0; i < n; ++i) cvi += std::abs(L.Vi[i][j]);
        nvi = std::max(nvi, cvi);
    }
    if (!(nvi < kModalMaxCond) || !std::isfinite(nvi)) return false;   // cond1(V) = nvi (||V||1 = 1)
    L.modal = true;
    return true;
}

// Generic modal decomposition of A (n <= kModalMax) with EigenSolver; false if not possible
inline bool buildModal(LinSys& L) {
    const int n = static_cast<int>(L.A.rows());
    if (n < 1 || n > kModalMax) return false;
    Eigen::EigenSolver<Mat> es(L.A, true);
    if (es.info() != Eigen::Success) return false;
    const auto ev = es.eigenvalues();
    double scale = 0.0;
    for (int i = 0; i < n; ++i) scale = std::max(scale, std::abs(ev[i]));
    for (int i = 0; i < n; ++i) if (std::abs(ev[i].imag()) > 1e-10 * std::max(scale, 1e-300)) return false;
    const Mat Vm = es.eigenvectors().real();
    Eigen::FullPivLU<Mat> lu(Vm);
    if (!lu.isInvertible()) return false;
    const Mat Vim = lu.inverse();
    L.n = n;
    for (int i = 0; i < n; ++i) {
        L.lam[i] = ev[i].real();
        L.bm[i] = L.b.size() == n ? L.b[i] : 0.0;
        for (int j = 0; j < n; ++j) { L.V[i][j] = Vm(i, j); L.Vi[i][j] = Vim(i, j); }
    }
    return acceptModal(L);
}

// E = e^{A dt} (n x n), modal or expm
inline Mat expmAt(const LinSys& L, double dt) {
    if (!L.modal) return expm(L.A * dt);
    const int n = L.n;
    Mat E(n, n);
    double e[kModalMax];
    for (int m = 0; m < n; ++m) e[m] = std::exp(L.lam[m] * dt);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int m = 0; m < n; ++m) s += L.V[i][m] * e[m] * L.Vi[m][j];
        E(i, j) = s;
    }
    return E;
}

// y(t+dt) = E y + Gamma u,  with u = b + infusion rate (constant over the segment)
inline Vec propagateLinear(const LinSys& L, const Vec& y, double dt, const Vec& u) {
    const int n = static_cast<int>(y.size());
    if (L.modal) {
        double z[kModalMax];
        for (int m = 0; m < n; ++m) {
            double zs = 0.0, ws = 0.0;
            for (int j = 0; j < n; ++j) { zs += L.Vi[m][j] * y[j]; ws += L.Vi[m][j] * u[j]; }
            const double x = L.lam[m] * dt;
            z[m] = std::exp(x) * zs + ws * (L.lam[m] == 0.0 ? dt : std::expm1(x) / L.lam[m]);
        }
        Vec out(n);
        for (int i = 0; i < n; ++i) { double s = 0.0; for (int m = 0; m < n; ++m) s += L.V[i][m] * z[m]; out[i] = s; }
        return out;
    }
    if (u.cwiseAbs().maxCoeff() == 0.0) return expm(L.A * dt) * y;
    Mat M = Mat::Zero(n + 1, n + 1);
    M.topLeftCorner(n, n) = L.A * dt;
    M.topRightCorner(n, 1) = u * dt;
    const Mat E = expm(M);
    return E.topLeftCorner(n, n) * y + E.topRightCorner(n, 1);
}

/**
 * @brief Pre-dose steady state for a linear system under repeated dosing.
 *
 * Computes @f$y = (I - e^{A\tau})^{-1}\Phi(0)@f$ for a dose repeated every
 * @c d.ii.
 *
 * @param L Linear system.
 * @param d Dose event defining the interval and amount.
 * @return The steady-state state immediately before the dose.
 * @throws std::invalid_argument If @c d.ii is not positive or a steady-state
 *         infusion lasts at least @c d.ii.
 * @throws std::runtime_error If the steady state does not exist (no
 *         elimination).
 */
inline Vec steadyStateLinear(const LinSys& L, const Dose& d) {
    const int n = L.n;
    if (!(d.ii > 0.0)) throw std::invalid_argument("ode: steady state requires II > 0");
    if (d.duration >= d.ii) throw std::invalid_argument("ode: steady-state infusion with duration >= II is not supported");
    Vec bv = Vec::Zero(n);
    if (L.modal) { for (int i = 0; i < n; ++i) bv[i] = L.bm[i]; } else bv = L.b;
    Vec y = Vec::Zero(n);
    if (d.duration <= 0.0) {
        y[d.cmt] += d.amount;
        y = propagateLinear(L, y, d.ii, bv);
    } else {
        Vec u = bv; u[d.cmt] += d.amount / d.duration;
        y = propagateLinear(L, y, d.duration, u);
        y = propagateLinear(L, y, d.ii - d.duration, bv);
    }
    Eigen::FullPivLU<Mat> lu(Mat::Identity(n, n) - expmAt(L, d.ii));
    if (!lu.isInvertible()) throw std::runtime_error("ode: steady state does not exist (no elimination)");
    return lu.solve(y);
}

/**
 * @brief Pre-dose steady state for a general (possibly nonlinear) system.
 *
 * Finds the fixed point of the one-interval dosing map using Newton's method
 * with a numerical Jacobian, falling back to fixed-point iteration.
 *
 * @param rhs Right-hand side of the ODE.
 * @param n State dimension.
 * @param d Dose event defining the interval and amount.
 * @param T Time at which the repeated dosing starts.
 * @param o Integrator options.
 * @return The steady-state state immediately before the dose.
 * @throws std::invalid_argument If @c d.ii is not positive or a steady-state
 *         infusion lasts at least @c d.ii.
 * @throws std::runtime_error If the fixed-point iteration does not converge.
 */
inline Vec steadyStateBefore(const Rhs& rhs, int n, const Dose& d, double T, const Options& o) {
    if (!(d.ii > 0.0)) throw std::invalid_argument("ode: steady state requires II > 0");
    if (d.duration >= d.ii) throw std::invalid_argument("ode: steady-state infusion with duration >= II is not supported");
    // Internal steady-state integrations use a slightly looser relative
    // tolerance (>= 1e-7): the pre-dose steady-state error is of the order of
    // the tolerance.
    Options oo = o; oo.rtol = std::max(o.rtol, 1e-7);
    auto phi = [&](const Vec& y0) {
        Vec y = y0; double h = 0.0;
        if (d.duration <= 0.0) {
            y[d.cmt] += d.amount;
            advance(rhs, T, y, T + d.ii, h, oo);
        } else {
            Vec rate = Vec::Zero(n); rate[d.cmt] = d.amount / d.duration;
            Rhs eff = [&](double t, const Vec& yy, Vec& dy) { rhs(t, yy, dy); dy += rate; };
            advance(eff, T, y, T + d.duration, h, oo);
            h = 0.0;
            advance(rhs, T + d.duration, y, T + d.ii, h, oo);
        }
        return y;
    };
    auto wn = [&](const Vec& r, const Vec& y) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) { const double sc = o.atol + o.rtol * std::abs(y[i]); s += (r[i] / sc) * (r[i] / sc); }
        return std::sqrt(s / double(std::max(1, n)));
    };
    Vec y = Vec::Zero(n);
    // Newton on F(y) = phi(y) - y (exact in one step for linear systems).
    for (int it = 0; it < 12; ++it) {
        const Vec py = phi(y), F = py - y;
        if (wn(F, y) < 1.0) return py;          // F is already at tolerance: py is the best estimate.
        Mat J(n, n);
        for (int j = 0; j < n; ++j) {
            Vec yp = y; const double dj = 1e-6 * std::max(1.0, std::abs(y[j]));
            yp[j] += dj; J.col(j) = (phi(yp) - py) / dj;
        }
        Eigen::FullPivLU<Mat> lu(J - Mat::Identity(n, n));
        if (!lu.isInvertible()) break;
        const Vec ynew = y - lu.solve(F);
        if (!ynew.allFinite()) break;
        y = ynew;
    }
    // Fallback: fixed-point iteration.
    y = Vec::Zero(n);
    for (int it = 0; it < 2000; ++it) {
        const Vec yn = phi(y);
        if (wn(yn - y, yn) < 1e-6) return yn;
        y = yn;
    }
    throw std::runtime_error("ode: steady state did not converge");
}

} // namespace detail

namespace detail {

/**
 * @brief Core of the linear fast path.
 *
 * Walks the sorted, unique breakpoints @p bp (observation times, dose times,
 * and infusion end times), applies the events, and calls @c sink(k, y) for each
 * observation @c k. The @p times array must be non-decreasing and @c y points
 * to @c n values. Convention: an observation at the instant of a dose sees the
 * state AFTER the dose is applied.
 */
template <class Sink>
inline void runLinear(const LinSys& L, const double* y0, double t0, const std::vector<double>& bp,
                      const double* times, int nTimes, const std::vector<Dose>& doses, Sink&& sink) {
    const int n = L.n;
    int next = 0;
    double tL = t0;
    if (!L.modal) {                                           // EXPM path (reference).
        Vec yL = Vec::Zero(n), rateL = Vec::Zero(n);
        if (y0) for (int i = 0; i < n; ++i) yL[i] = y0[i];
        for (double T : bp) {
            if (T > tL) { yL = propagateLinear(L, yL, T - tL, L.b + rateL); tL = T; }
            for (const auto& d : doses) {
                if (d.time != T || d.ss == 0) continue;
                const Vec yss = steadyStateLinear(L, d);
                if (d.ss == 1) yL = yss; else yL += yss;
            }
            for (const auto& d : doses) if (d.time == T && d.duration == 0.0) yL[d.cmt] += d.amount;
            rateL.setZero();
            for (const auto& d : doses) if (d.duration > 0.0 && d.time <= T && T < d.time + d.duration) rateL[d.cmt] += d.amount / d.duration;
            while (next < nTimes && times[next] == T) { sink(next, yL.data()); ++next; }
        }
        return;
    }
    // MODAL path: the state lives in modal coordinates z = Vi y (each mode
    // evolves independently through an exponential).
    double z[kModalMax] = {}, w[kModalMax] = {}, wb[kModalMax] = {};
    for (int m = 0; m < n; ++m) {
        double s = 0.0, sb = 0.0;
        for (int j = 0; j < n; ++j) { if (y0) s += L.Vi[m][j] * y0[j]; sb += L.Vi[m][j] * L.bm[j]; }
        z[m] = s; wb[m] = sb; w[m] = sb;                       // w = Vi (b + rate); without infusion: Vi b.
    }
    double yv[kModalMax];
    bool forced = false;                                      // There is an affine term or an active infusion.
    for (int m = 0; m < n; ++m) if (w[m] != 0.0) forced = true;
    for (double T : bp) {
        if (T > tL) {
            const double dt = T - tL;
            if (forced) {
                for (int m = 0; m < n; ++m) {
                    const double x = L.lam[m] * dt;
                    z[m] = std::exp(x) * z[m] + w[m] * (L.lam[m] == 0.0 ? dt : std::expm1(x) / L.lam[m]);
                }
            } else {
                for (int m = 0; m < n; ++m) z[m] *= std::exp(L.lam[m] * dt);   // Free decay.
            }
            tL = T;
        }
        for (const auto& d : doses) {
            if (d.time != T || d.ss == 0) continue;
            const Vec yss = steadyStateLinear(L, d);
            if (d.ss == 1) for (int m = 0; m < n; ++m) z[m] = 0.0;
            for (int m = 0; m < n; ++m) { double s = 0.0; for (int j = 0; j < n; ++j) s += L.Vi[m][j] * yss[j]; z[m] += s; }
        }
        for (const auto& d : doses)
            if (d.time == T && d.duration == 0.0) for (int m = 0; m < n; ++m) z[m] += L.Vi[m][d.cmt] * d.amount;
        // Infusion rate active from T onward.
        double rate[kModalMax] = {};
        bool anyRate = false;
        for (const auto& d : doses)
            if (d.duration > 0.0 && d.time <= T && T < d.time + d.duration) { rate[d.cmt] += d.amount / d.duration; anyRate = true; }
        if (anyRate) {
            for (int m = 0; m < n; ++m) {
                double s = wb[m];
                for (int j = 0; j < n; ++j) s += L.Vi[m][j] * rate[j];
                w[m] = s;
            }
        } else for (int m = 0; m < n; ++m) w[m] = wb[m];
        forced = false;
        for (int m = 0; m < n; ++m) if (w[m] != 0.0) forced = true;
        while (next < nTimes && times[next] == T) {
            for (int i = 0; i < n; ++i) { double s = 0.0; for (int m = 0; m < n; ++m) s += L.V[i][m] * z[m]; yv[i] = s; }
            sink(next, yv); ++next;
        }
    }
}

// ---- Closed-form modal decompositions of linear (mammillary) PK models ------
// Every function returns a LinSys with modal = true when the decomposition is
// reliable; otherwise (nearly equal eigenvalues, poor conditioning, large
// residuals) modal = false and A is filled for the expm path.

/**
 * @brief Finalizes a hand-built modal decomposition.
 *
 * Validates conditioning (@c acceptModal) and the residuals
 * (@f$Av_j = \lambda_j v_j@f$ and @f$V V^{-1} = I@f$). If any check fails, the
 * system falls back to the expm path using the supplied matrix @p A.
 */
inline void finishModal(LinSys& L, const double (&A)[kModalMax][kModalMax]) {
    const int n = L.n;
    bool ok = acceptModal(L);
    if (ok) {
        double scale = 0.0;
        for (int i = 0; i < n; ++i) scale = std::max(scale, std::abs(L.lam[i]));
        for (int j = 0; j < n && ok; ++j)
            for (int i = 0; i < n; ++i) {
                double av = 0.0;
                for (int k = 0; k < n; ++k) av += A[i][k] * L.V[k][j];
                if (!(std::abs(av - L.lam[j] * L.V[i][j]) <= 1e-13 * scale)) { ok = false; break; }
            }
        for (int i = 0; i < n && ok; ++i)
            for (int j = 0; j < n; ++j) {
                double s2 = 0.0;
                for (int k = 0; k < n; ++k) s2 += L.V[i][k] * L.Vi[k][j];
                if (!(std::abs(s2 - (i == j ? 1.0 : 0.0)) <= 1e-12)) { ok = false; break; }
            }
    }
    if (!ok) {
        L.modal = false;
        L.A = Mat::Zero(n, n); L.b = Vec::Zero(n);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) L.A(i, j) = A[i][j];
    }
}

inline LinSys pk1Iv(double k10) {
    LinSys L; L.n = 1;
    L.lam[0] = -k10; L.V[0][0] = 1.0; L.Vi[0][0] = 1.0;
    double A[kModalMax][kModalMax] = {}; A[0][0] = -k10;
    finishModal(L, A);
    return L;
}
// Compartments: 0 = depot, 1 = central.
inline LinSys pk1Oral(double ka, double k10) {
    LinSys L; L.n = 2;
    double A[kModalMax][kModalMax] = {};
    A[0][0] = -ka; A[1][0] = ka; A[1][1] = -k10;
    const double dlt = k10 - ka;
    if (std::abs(dlt) > 0.0) {
        L.lam[0] = -ka; L.lam[1] = -k10;
        L.V[0][0] = dlt; L.V[0][1] = 0.0; L.V[1][0] = ka; L.V[1][1] = 1.0;
        L.Vi[0][0] = 1.0 / dlt; L.Vi[0][1] = 0.0; L.Vi[1][0] = -ka / dlt; L.Vi[1][1] = 1.0;
    }
    if (!(std::abs(dlt) > 0.0)) { finishModal(L, A); L.modal = false; if (L.A.size() == 0) { L.A = Mat::Zero(2, 2); L.b = Vec::Zero(2); for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) L.A(i, j) = A[i][j]; } }
    else finishModal(L, A);
    return L;
}
// Compartments: 0 = central, 1 = peripheral. Roots of
// l^2 + s l + k10 k21 = 0 (numerically stable).
inline void pk2Roots(double k10, double k12, double k21, double& l1, double& l2) {
    const double s = k10 + k12 + k21, p = k10 * k21;
    const double disc = std::max(0.0, (k10 + k12 - k21) * (k10 + k12 - k21) + 4.0 * k12 * k21);
    l1 = -0.5 * (s + std::sqrt(disc));                          // The more negative root (no cancellation).
    l2 = (l1 != 0.0) ? p / l1 : 0.0;
}

// Eigenvector (0?, k21, a) of the central/peripheral block for eigenvalue l,
// with a = l + k10 + k12. Since (l + c)(l + k21) = k12 k21 (characteristic
// equation), the small factor is obtained from the large one without
// cancellation.
inline double pk2Vec(double l, double k10, double k12, double k21) {
    const double c = k10 + k12;
    const double a = l + c, b = l + k21;
    return (std::abs(a) >= std::abs(b)) ? a : (k12 * k21) / b;
}
inline LinSys pk2Iv(double k10, double k12, double k21) {
    LinSys L; L.n = 2;
    double A[kModalMax][kModalMax] = {};
    A[0][0] = -(k10 + k12); A[0][1] = k21; A[1][0] = k12; A[1][1] = -k21;
    double l1, l2; pk2Roots(k10, k12, k21, l1, l2);
    L.lam[0] = l1; L.lam[1] = l2;
    L.V[0][0] = k21; L.V[0][1] = k21;
    L.V[1][0] = pk2Vec(l1, k10, k12, k21); L.V[1][1] = pk2Vec(l2, k10, k12, k21);
    const double det = L.V[0][0] * L.V[1][1] - L.V[0][1] * L.V[1][0];
    if (std::abs(det) > 0.0 && std::isfinite(det)) {
        L.Vi[0][0] = L.V[1][1] / det; L.Vi[0][1] = -L.V[0][1] / det;
        L.Vi[1][0] = -L.V[1][0] / det; L.Vi[1][1] = L.V[0][0] / det;
        finishModal(L, A);
    } else { L.modal = false; L.A = Mat::Zero(2, 2); L.b = Vec::Zero(2); for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) L.A(i, j) = A[i][j]; }
    return L;
}
// Compartments: 0 = depot, 1 = central, 2 = peripheral.
inline LinSys pk2Oral(double ka, double k10, double k12, double k21) {
    LinSys L; L.n = 3;
    double A[kModalMax][kModalMax] = {};
    A[0][0] = -ka; A[1][0] = ka; A[1][1] = -(k10 + k12); A[1][2] = k21; A[2][1] = k12; A[2][2] = -k21;
    double l1, l2; pk2Roots(k10, k12, k21, l1, l2);
    L.lam[0] = -ka; L.lam[1] = l1; L.lam[2] = l2;
    const double c = k10 + k12;
    // Depot mode: (v1, k21 - ka, k12).
    const double v2 = k21 - ka, v3 = k12;
    const double v1 = -((ka - c) * v2 + k21 * v3) / ka;
    L.V[0][0] = v1; L.V[1][0] = v2; L.V[2][0] = v3;
    L.V[0][1] = 0.0; L.V[1][1] = k21; L.V[2][1] = pk2Vec(l1, k10, k12, k21);
    L.V[0][2] = 0.0; L.V[1][2] = k21; L.V[2][2] = pk2Vec(l2, k10, k12, k21);
    Eigen::Matrix3d Vm;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Vm(i, j) = L.V[i][j];
    const double det = Vm.determinant();
    if (std::isfinite(det) && std::abs(det) > 0.0 && ka > 0.0) {
        const Eigen::Matrix3d Vim = Vm.inverse();
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) L.Vi[i][j] = Vim(i, j);
        finishModal(L, A);
    } else { L.modal = false; L.A = Mat::Zero(3, 3); L.b = Vec::Zero(3); for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) L.A(i, j) = A[i][j]; }
    return L;
}

} // namespace detail

/**
 * @brief Fast path for linear models that are already decomposed.
 *
 * See @c detail::pk1Iv and related builders. @p times must be non-decreasing
 * and at least @p t0; @c sink(k, const double* y) receives the state (@c n
 * values) at @c times[k]. The modal path performs no per-call heap allocation
 * (except when @c ss != 0).
 *
 * @throws std::invalid_argument If a dose has an invalid compartment, occurs
 *         before @p t0, uses an unresolved modeled rate/duration, or has a
 *         negative duration.
 */
template <class Sink>
inline void integrateLinear(const detail::LinSys& L, double t0, const double* times, int nTimes,
                            const std::vector<Dose>& doses, Sink&& sink) {
    const int n = L.n;
    for (const auto& d : doses) {
        if (d.cmt < 0 || d.cmt >= n) throw std::invalid_argument("ode::integrateLinear: compartment out of range");
        if (d.time < t0) throw std::invalid_argument("ode::integrateLinear: dose before t0");
        if (d.modeled != 0) throw std::invalid_argument("ode::integrateLinear: dose with unresolved modeled rate/duration");
        if (d.duration < 0.0) throw std::invalid_argument("ode::integrateLinear: duration < 0");
    }
    thread_local std::vector<double> bp;                         // Reused across calls.
    bp.assign(times, times + nTimes);
    for (const auto& d : doses) { bp.push_back(d.time); if (d.duration > 0.0) bp.push_back(d.time + d.duration); }
    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end()), bp.end());
    detail::runLinear(L, nullptr, t0, bp, times, nTimes, doses, std::forward<Sink>(sink));
}

/**
 * @brief Integrates the ODE and returns the state at the requested times.
 *
 * @param rhs Right-hand side of the ODE.
 * @param y0 Initial state at @p t0.
 * @param t0 Initial time.
 * @param times Non-decreasing observation times, all >= @p t0.
 * @param doses Dose events.
 * @param o Integrator options.
 * @param stats Optional integration statistics.
 * @return Matrix of shape @c n_times x @c dim(y0).
 * @throws std::invalid_argument If @p times is decreasing, precedes @p t0, or a
 *         dose is invalid or unresolved.
 */
inline Mat integrate(const Rhs& rhs, const Vec& y0, double t0, const std::vector<double>& times,
                     std::vector<Dose> doses = {}, const Options& o = Options(), Stats* stats = nullptr) {
    const int n = static_cast<int>(y0.size());
    for (size_t k = 1; k < times.size(); ++k)
        if (times[k] < times[k - 1]) throw std::invalid_argument("ode::integrate: times must be non-decreasing");
    if (!times.empty() && times.front() < t0) throw std::invalid_argument("ode::integrate: times < t0");
    for (const auto& d : doses) {
        if (d.cmt < 0 || d.cmt >= n) throw std::invalid_argument("ode::integrate: compartment out of range");
        if (d.time < t0) throw std::invalid_argument("ode::integrate: dose before t0");
        if (d.modeled != 0) throw std::invalid_argument("ode::integrate: dose with unresolved modeled rate/duration (use ode::resolveDoses)");
        if (d.duration < 0.0) throw std::invalid_argument("ode::integrate: duration < 0");
    }
    // Breakpoints.
    std::vector<double> bp(times.begin(), times.end());
    for (const auto& d : doses) { bp.push_back(d.time); if (d.duration > 0.0) bp.push_back(d.time + d.duration); }
    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end()), bp.end());

    if (o.linear) {                                           // Exact path (modal or matrix exponential).
        detail::LinSys L = detail::extractLinear(rhs, n, t0);
        if (o.linearModal) detail::buildModal(L);
        Mat outL(static_cast<int>(times.size()), n);
        detail::runLinear(L, y0.data(), t0, bp, times.data(), static_cast<int>(times.size()), doses,
                          [&](int k, const double* yv) { for (int i = 0; i < n; ++i) outL(k, i) = yv[i]; });
        return outL;
    }
    Mat out(static_cast<int>(times.size()), n);
    Vec y = y0, rate = Vec::Zero(n);
    Rhs eff = [&](double t, const Vec& yy, Vec& dy) { rhs(t, yy, dy); dy += rate; };
    double t = t0, h = o.h0;
    size_t nextObs = 0;
    for (double T : bp) {
        if (T > t) { advance(eff, t, y, T, h, o, stats); t = T; }
        // Pre-dose steady state for doses with ss != 0 at T.
        for (const auto& d : doses) {
            if (d.time != T || d.ss == 0) continue;
            const Vec yss = detail::steadyStateBefore(rhs, n, d, T, o);
            if (d.ss == 1) y = yss; else y += yss;
        }
        // Apply events at T.
        for (const auto& d : doses) {
            if (d.time == T && d.duration == 0.0) y[d.cmt] += d.amount;
        }
        rate.setZero();
        for (const auto& d : doses) {
            if (d.duration > 0.0 && d.time <= T && T < d.time + d.duration) rate[d.cmt] += d.amount / d.duration;
        }
        while (nextObs < times.size() && times[nextObs] == T) { out.row(static_cast<int>(nextObs)) = y.transpose(); ++nextObs; }
    }
    return out;
}

/**
 * @brief Integrates the ODE and returns the final state.
 * @param rhs Right-hand side of the ODE.
 * @param y0 Initial state at @p t0.
 * @param t0 Initial time.
 * @param t1 Final time.
 * @param o Integrator options.
 * @param stats Optional integration statistics.
 * @return The state at @p t1 (no doses are applied).
 */
inline Vec solve(const Rhs& rhs, const Vec& y0, double t0, double t1, const Options& o = Options(), Stats* stats = nullptr) {
    Vec y = y0; double h = o.h0;
    advance(rhs, t0, y, t1, h, o, stats);
    return y;
}

} // namespace ode
