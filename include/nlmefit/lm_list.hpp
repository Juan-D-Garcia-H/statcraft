// SPDX-License-Identifier: Apache-2.0
//
// `lmList()` and `nlsList()` from R/nlme: one fit per group (linear or
// nonlinear least squares) and the starting values that `nlme()` obtains from
// `nlsList` (fixed effects and the covariance of the random effects).
//
//     lm_list(groups)                     OLS per group (aliased columns -> NaN),
//                                         per-group sigma, and pooled sigma
//     nls_list(groups, f, jac, beta0)     nonlinear least squares per group
//                                         (Levenberg-Marquardt); the starting
//                                         point is phi_i = A_i beta0
//     nlme_start_from_nls(model, beta0)   beta = regression of the per-group
//                                         phi_i on A_i; D0 = covariance of the
//                                         estimated b_i / sigma^2
//     nlme_fit_auto(model, beta0)         NlmeModel::fit with those starting values
//
// Fidelity: `lmList` and `nlsList` reproduce R's coefficients and sigma (up to
// optimizer tolerance). R's initial value for D in `nlme()` is an undocumented
// heuristic; here we use the natural choice (the empirical covariance of the
// per-group random effects), which leads to the same optimum.
#pragma once

#include "lme_fit.hpp"
#include "nlme_fit.hpp"

namespace mixedfit {

/// Results of an `lmList` fit: one ordinary least-squares fit per group.
struct LmListResult {
    std::vector<int> labels;                    ///< Group labels, in input order.
    std::vector<Eigen::VectorXd> coef, se;      ///< Coefficients and standard errors (NaN for aliased columns).
    std::vector<Eigen::MatrixXd> vcov;          ///< Coefficient covariance matrices.
    std::vector<double> sigma, rss;             ///< sigma = sqrt(rss / df); NaN when df = 0.
    std::vector<int> df, rank;                  ///< Residual degrees of freedom and design rank per group.
    double pooled_sigma = std::numeric_limits<double>::quiet_NaN();  ///< R's `pooledSD`.
    int pooled_df = 0;                          ///< Degrees of freedom behind `pooled_sigma`.

    /// Stack the per-group coefficients into a (groups x p) matrix.
    Eigen::MatrixXd coef_matrix() const {       // groups x p
        Eigen::MatrixXd M(coef.size(), coef.empty() ? 0 : coef[0].size());
        for (size_t i = 0; i < coef.size(); ++i) M.row(i) = coef[i].transpose();
        return M;
    }
};

/// Fit an ordinary least-squares model separately within each group.
///
/// The design is factorized with a rank-revealing QR decomposition, so rank
/// deficiency is handled gracefully: aliased columns receive `NaN`
/// coefficients, standard errors, and covariances instead of arbitrary values.
///
/// @param groups Grouped data (`y`, `X`, and label per group).
/// @return Per-group coefficients, standard errors, covariances, residual
///         summaries, and the pooled residual standard deviation.
inline LmListResult lm_list(const std::vector<LmeGroup>& groups) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    LmListResult out;
    double rss_tot = 0.0;
    int df_tot = 0;
    for (const auto& g : groups) {
        const int n = static_cast<int>(g.y.size()), p = static_cast<int>(g.X.cols());
        RankQR qr(g.X);
        const int rk = qr.rank();
        Eigen::VectorXd coef = Eigen::VectorXd::Constant(p, nan), se = coef;
        Eigen::MatrixXd V = Eigen::MatrixXd::Constant(p, p, nan);
        const Eigen::MatrixXd b = qr.solve(g.y);
        const double rss = (g.y - g.X * b.col(0)).squaredNorm();
        const int df = n - rk;
        const double sg = df > 0 ? std::sqrt(rss / df) : nan;
        const auto piv = qr.pivot();
        for (int i = 0; i < rk; ++i) coef[piv[i]] = b(piv[i], 0);
        if (rk > 0) {
            const Eigen::MatrixXd Rl = qr.R_leading();
            const Eigen::MatrixXd Ri = Rl.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(rk, rk));
            const Eigen::MatrixXd Cp = Ri * Ri.transpose();
            for (int i = 0; i < rk; ++i)
                for (int j = 0; j < rk; ++j) V(piv[i], piv[j]) = sg * sg * Cp(i, j);
            for (int i = 0; i < rk; ++i) se[piv[i]] = std::sqrt(V(piv[i], piv[i]));
        }
        out.labels.push_back(g.label);
        out.coef.push_back(coef);
        out.se.push_back(se);
        out.vcov.push_back(V);
        out.sigma.push_back(sg);
        out.rss.push_back(rss);
        out.df.push_back(df);
        out.rank.push_back(rk);
        if (df > 0) { rss_tot += rss; df_tot += df; }
    }
    if (df_tot > 0) { out.pooled_sigma = std::sqrt(rss_tot / df_tot); out.pooled_df = df_tot; }
    return out;
}

/// Tuning parameters for `nls_list`.
struct NlsListControl {
    int max_iter = 200;     ///< Maximum number of Levenberg-Marquardt iterations per group.
    double tol = 1e-8;      ///< Relative-offset convergence criterion of Bates & Watts.
};

/// Results of an `nlsList` fit: one nonlinear least-squares fit per group.
struct NlsListResult {
    std::vector<int> labels;                    ///< Group labels, in input order.
    std::vector<Eigen::VectorXd> coef, se;      ///< k parameters per group (NaN when the group did not converge).
    std::vector<Eigen::MatrixXd> vcov;          ///< Parameter covariance matrices.
    std::vector<double> sigma, rss;             ///< Residual standard deviation and residual sum of squares.
    std::vector<int> df, iterations;            ///< Residual degrees of freedom and iteration count.
    std::vector<char> converged;                ///< 1 if the group converged, 0 otherwise.
    double pooled_sigma = std::numeric_limits<double>::quiet_NaN();  ///< Pooled residual standard deviation.
    int pooled_df = 0;                          ///< Degrees of freedom behind `pooled_sigma`.

    /// Number of groups that converged.
    int n_converged() const { int c = 0; for (char v : converged) c += v ? 1 : 0; return c; }
};

namespace detail {

/// Nonlinear fit of a single group:
/// `y_u = f([1 phi^T])[u] + e`, with parameters `phi` (k) shared by all rows
/// of the group.
struct NlsOne {
    bool ok = false;                            ///< Whether the fit converged.
    int iter = 0;                               ///< Iterations performed.
    Eigen::VectorXd phi, se;                    ///< Estimates and standard errors (NaN on failure).
    Eigen::MatrixXd V;                          ///< Parameter covariance matrix.
    double rss = std::numeric_limits<double>::quiet_NaN();    ///< Residual sum of squares.
    double sigma = std::numeric_limits<double>::quiet_NaN();  ///< Residual standard deviation.
    int df = 0;                                 ///< Residual degrees of freedom.
};

/// Fit one group by Levenberg-Marquardt nonlinear least squares.
///
/// Only the rows selected by `g.use` (the NA pattern) contribute to the
/// objective. When no analytic Jacobian is supplied, a central finite-difference
/// approximation is used. Any exception raised by the model function aborts the
/// fit for this group and leaves `ok == false`.
///
/// @param g    Group data.
/// @param gi   Group index passed through to the model function.
/// @param f    Model function `f(Phi, gi)` returning the full-length mean vector.
/// @param jac  Optional analytic Jacobian `jac(Phi, gi)`.
/// @param phi  Initial parameter vector (k).
/// @param ctl  Iteration controls.
/// @return The per-group fit, with `ok == false` if it did not converge.
inline NlsOne nls_one(const NlmeGroup& g, int gi, const NlmeFn& f, const NlmeJacFn& jac, Eigen::VectorXd phi,
                      const NlsListControl& ctl) {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    NlsOne r;
    const int n = static_cast<int>(g.y.size()), k = static_cast<int>(phi.size());
    std::vector<int> u;
    for (int i = 0; i < n; ++i) if (g.use.empty() || g.use[i]) u.push_back(i);
    const int nu = static_cast<int>(u.size());
    r.df = nu - k;
    r.phi = Eigen::VectorXd::Constant(k, nan);
    r.se = r.phi;
    r.V = Eigen::MatrixXd::Constant(k, k, nan);
    if (nu == 0) return r;
    Eigen::VectorXd yu(nu);
    for (int i = 0; i < nu; ++i) yu[i] = g.y[u[i]];
    // Expand a single parameter vector to the per-row parameter matrix expected
    // by the model function: every row gets the same `phi`.
    auto Phi_of = [&](const Eigen::VectorXd& p) { return Eigen::MatrixXd(Eigen::VectorXd::Ones(n) * p.transpose()); };
    auto pred = [&](const Eigen::VectorXd& p) {
        const Eigen::VectorXd fv = f(Phi_of(p), gi);
        if (fv.size() != n) throw std::runtime_error("nls_list: f returned the wrong length");
        return Eigen::VectorXd(fv(u));
    };
    auto jacm = [&](const Eigen::VectorXd& p) {
        const Eigen::MatrixXd P = Phi_of(p);
        Eigen::MatrixXd J(n, k);
        if (jac) {
            J = jac(P, gi);
        } else {
            // Central differences with a step scaled to each parameter.
            Eigen::MatrixXd Q = P;
            const double base = std::cbrt(std::numeric_limits<double>::epsilon());
            for (int c = 0; c < k; ++c) {
                const double h = base * std::max(1.0, std::abs(p[c]));
                Q.col(c).array() += h;
                const Eigen::VectorXd fp = f(Q, gi);
                Q.col(c) = P.col(c);
                Q.col(c).array() -= h;
                const Eigen::VectorXd fm = f(Q, gi);
                Q.col(c) = P.col(c);
                J.col(c) = (fp - fm) / (2.0 * h);
            }
        }
        return Eigen::MatrixXd(J(u, Eigen::placeholders::all));
    };
    try {
        Eigen::VectorXd res = yu - pred(phi);
        double S = res.squaredNorm();
        if (!std::isfinite(S)) return r;
        double lam = 1e-3;
        bool conv = false;
        for (int it = 1; it <= ctl.max_iter; ++it) {
            r.iter = it;
            const Eigen::MatrixXd J = jacm(phi);
            const Eigen::MatrixXd JtJ = J.transpose() * J;
            const Eigen::VectorXd gr = J.transpose() * res;
            // Relative offset: ||Q1' r||^2 = gr' (J'J)^-1 gr.
            Eigen::LDLT<Eigen::MatrixXd> ldl(JtJ);
            if (ldl.info() != Eigen::Success) break;
            const Eigen::VectorXd d0 = ldl.solve(gr);
            const double proj = std::max(gr.dot(d0), 0.0);
            const double perp = std::max(S - proj, 0.0);
            const bool tiny = S <= 1e-24 * (1.0 + yu.squaredNorm());
            const double off = (nu > k && perp > 0.0) ? std::sqrt(proj / k) / std::sqrt(perp / (nu - k)) : (proj <= 1e-20 * S ? 0.0 : 1.0);
            if (off < ctl.tol || tiny) { conv = true; break; }
            bool accepted = false;
            for (int h = 0; h < 60; ++h) {
                // Levenberg-Marquardt damping of the normal equations.
                Eigen::MatrixXd A = JtJ;
                for (int c = 0; c < k; ++c) A(c, c) += lam * std::max(JtJ(c, c), 1e-12);
                Eigen::LDLT<Eigen::MatrixXd> l2(A);
                if (l2.info() != Eigen::Success) { lam *= 10.0; continue; }
                const Eigen::VectorXd dl = l2.solve(gr);
                const Eigen::VectorXd pn = phi + dl;
                double Sn = std::numeric_limits<double>::infinity();
                Eigen::VectorXd rn;
                try { rn = yu - pred(pn); Sn = rn.squaredNorm(); } catch (...) {}
                if (std::isfinite(Sn) && Sn < S) {
                    phi = pn; res = rn; S = Sn; lam = std::max(lam / 10.0, 1e-14); accepted = true;
                    break;
                }
                lam *= 10.0;
                if (lam > 1e14) break;
            }
            if (!accepted) { conv = (off < 1e-4); break; }   // no further improvement: accept only if already very close
        }
        if (!conv) return r;
        r.ok = true;
        r.phi = phi;
        r.rss = S;
        if (nu > k) {
            r.sigma = std::sqrt(S / (nu - k));
            // Approximate parameter covariance from the Gauss-Newton matrix at
            // the solution: V = sigma^2 (J'J)^-1.
            const Eigen::MatrixXd J = jacm(phi);
            Eigen::LDLT<Eigen::MatrixXd> ldl(J.transpose() * J);
            if (ldl.info() == Eigen::Success) {
                r.V = r.sigma * r.sigma * ldl.solve(Eigen::MatrixXd::Identity(k, k));
                for (int c = 0; c < k; ++c) r.se[c] = std::sqrt(std::max(r.V(c, c), 0.0));
            }
        }
    } catch (const std::exception&) {
        r.ok = false;
    }
    return r;
}

}  // namespace detail

/// Fit a separate nonlinear least-squares model to each group.
///
/// @param groups Grouped data.
/// @param f      Model function `f(Phi, group_index)`.
/// @param jac    Optional analytic Jacobian; finite differences are used when empty.
/// @param phi0   One initial parameter vector (k) per group.
/// @param ctl    Iteration controls.
/// @return Per-group estimates. Groups that do not converge keep
///         `converged == 0` and `NaN` coefficients (R returns `NULL` for them).
/// @throws std::invalid_argument If `phi0` does not have one entry per group.
inline NlsListResult nls_list(const std::vector<NlmeGroup>& groups, const NlmeFn& f, const NlmeJacFn& jac,
                              const std::vector<Eigen::VectorXd>& phi0, const NlsListControl& ctl = {}) {
    if (phi0.size() != groups.size()) throw std::invalid_argument("nls_list: expected one initial value per group");
    NlsListResult out;
    double rss_tot = 0.0;
    int df_tot = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        const detail::NlsOne r = detail::nls_one(groups[i], static_cast<int>(i), f, jac, phi0[i], ctl);
        out.labels.push_back(groups[i].label);
        out.coef.push_back(r.phi);
        out.se.push_back(r.se);
        out.vcov.push_back(r.V);
        out.sigma.push_back(r.sigma);
        out.rss.push_back(r.rss);
        out.df.push_back(r.df);
        out.iterations.push_back(r.iter);
        out.converged.push_back(r.ok ? 1 : 0);
        if (r.ok && r.df > 0) { rss_tot += r.rss; df_tot += r.df; }
    }
    if (df_tot > 0) { out.pooled_sigma = std::sqrt(rss_tot / df_tot); out.pooled_df = df_tot; }
    return out;
}

/// Variant that starts from a common point in fixed-effect space:
/// `phi0_i = A_i beta0`.
///
/// @param groups Grouped data.
/// @param f      Model function.
/// @param jac    Optional analytic Jacobian.
/// @param beta0  Fixed-effect starting values.
/// @param ctl    Iteration controls.
/// @return The per-group fits.
inline NlsListResult nls_list(const std::vector<NlmeGroup>& groups, const NlmeFn& f, const NlmeJacFn& jac,
                              const Eigen::VectorXd& beta0, const NlsListControl& ctl = {}) {
    std::vector<Eigen::VectorXd> p0;
    for (const auto& g : groups) p0.push_back(g.A * beta0);
    return nls_list(groups, f, jac, p0, ctl);
}

/// Starting values for `nlme()` derived from `nlsList`.
struct NlmeStart {
    Eigen::VectorXd beta, theta;                ///< Fixed effects and variance parameters.
    Eigen::MatrixXd D0;                         ///< Relative covariance (Sigma_b / sigma^2) of level 0.
    double sigma = std::numeric_limits<double>::quiet_NaN();  ///< Pooled `nlsList` sigma.
    int n_used = 0, n_failed = 0;               ///< Groups used in the start and groups that failed.
    NlsListResult nls;                          ///< Underlying per-group nonlinear fits.
};

/// Build starting values for `nlme()` from a per-group `nlsList`.
///
/// Steps:
///  1. Fit `nlsList` with the common fixed-effect start `beta0`.
///  2. Obtain `beta` by regressing the per-group `phi_i` on their design
///     matrices `A_i` (least squares over the groups that converged).
///  3. Set the level-0 random-effect covariance to the empirical covariance of
///     `b_i = (B'B)^-1 B' (phi_i - A_i beta)`, divided by `sigma^2`.
///
/// If the groups carry row-level fixed- or random-effect covariates, the
/// parameters are not constant within a group and `nlsList` does not apply;
/// in that case the function falls back to `beta0` and the default theta.
///
/// @param model Nonlinear mixed-effects model.
/// @param beta0 Fixed-effect starting values.
/// @param ctl   Iteration controls for the per-group fits.
/// @return The derived starting values.
/// @throws std::runtime_error If no group converged in `nlsList`.
inline NlmeStart nlme_start_from_nls(const NlmeModel& model, const Eigen::VectorXd& beta0, const NlsListControl& ctl = {}) {
    const auto& gs = model.groups();
    NlmeStart st;
    for (const auto& g : gs)
        if (g.A_row.size() || g.B_row.size()) {   // row-level covariates: phi is not constant within a group, nlsList does not apply
            st.beta = beta0;
            st.theta = model.default_theta();
            return st;
        }
    st.nls = nls_list(gs, model.fn(), model.jac_fn(), beta0, ctl);
    const int p = model.n_fixed();
    Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd Atp = Eigen::VectorXd::Zero(p);
    std::vector<int> good;
    for (size_t i = 0; i < gs.size(); ++i) {
        if (!st.nls.converged[i]) { ++st.n_failed; continue; }
        good.push_back(static_cast<int>(i));
        AtA += gs[i].A.transpose() * gs[i].A;
        Atp += gs[i].A.transpose() * st.nls.coef[i];
    }
    st.n_used = static_cast<int>(good.size());
    if (good.empty()) throw std::runtime_error("nlme_start_from_nls: no group converged in nlsList");
    st.beta = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(AtA).solve(Atp);
    st.sigma = st.nls.pooled_sigma;
    const int q0 = static_cast<int>(gs[0].B.cols());
    if (q0 > 0) {
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(q0, q0);
        for (int i : good) {
            const Eigen::MatrixXd& B = gs[i].B;
            const Eigen::VectorXd e = st.nls.coef[i] - gs[i].A * st.beta;
            const Eigen::VectorXd bh = (B.transpose() * B).ldlt().solve(B.transpose() * e);
            S += bh * bh.transpose();
        }
        S /= static_cast<double>(good.size());
        const double s2 = (std::isfinite(st.sigma) && st.sigma > 0.0) ? st.sigma * st.sigma : 1.0;
        Eigen::MatrixXd D = S / s2;
        auto pd = [&](const Eigen::MatrixXd& M) { return Eigen::LLT<Eigen::MatrixXd>(M).info() == Eigen::Success; };
        if (!pd(D)) {                                  // too few clusters: regularize, then fall back to the diagonal
            const double tr = std::max(D.trace() / q0, 1e-8);
            D += 1e-3 * tr * Eigen::MatrixXd::Identity(q0, q0);
            if (!pd(D)) {
                Eigen::MatrixXd Dd = Eigen::MatrixXd::Zero(q0, q0);
                for (int c = 0; c < q0; ++c) Dd(c, c) = std::max(S(c, c) / s2, 1e-8);
                D = Dd;
            }
        }
        st.D0 = D;
    }
    st.theta = model.theta_with_D0(st.D0);
    return st;
}

/// Fit an `NlmeModel` starting from the values derived by
/// `nlme_start_from_nls`.
///
/// @param model     Nonlinear mixed-effects model.
/// @param beta0     Fixed-effect starting values.
/// @param start_out Optional output for the derived starting values.
/// @return The fit result.
inline NlmeResult nlme_fit_auto(const NlmeModel& model, const Eigen::VectorXd& beta0, NlmeStart* start_out = nullptr) {
    NlmeStart st = nlme_start_from_nls(model, beta0);
    NlmeResult r = model.fit(st.beta, &st.theta);
    if (start_out) *start_out = std::move(st);
    return r;
}

}  // namespace mixedfit
