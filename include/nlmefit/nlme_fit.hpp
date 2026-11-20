// SPDX-License-Identifier: Apache-2.0
//
// Nonlinear mixed-effects model with nested levels:
//     y_ij = f(phi_ij) + e_ij,
//     phi_ij = A_i beta + B_0 b_i + sum_{l>=1} B_l b_{l,s_l(j)},
//     b_i ~ N(0, sigma^2 D_0), b_{l,s} ~ N(0, sigma^2 D_l),
//     e_i ~ N(0, sigma^2 S_i R_i S_i).
//
// Alternating algorithm of Lindstrom & Bates (1990), Biometrics 46:673-687:
//   1. PNLS: given theta, minimize over (beta, b) the penalized sum of squares
//        sum_i ||Lam_i^-1/2 (y_i - f_i)||^2 + sum_units b^T D_l^-1 b
//      with Gauss-Newton plus step reduction. Each iteration solves the linear
//      least-squares problem by eliminating the random effects group by group
//      with a QR decomposition (arrow structure): cost
//      O(sum_i n_i (p + nb_i)^2), linear in the number of groups.
//   2. LME: linearize f at (beta, b) with working response
//        w = y - f + Xhat beta + sum_l Zhat_l b_l,
//      and re-estimate (theta, sigma^2) with the profiled likelihood of
//      `LmeModel`.
//
// Model-function contract: `f(Phi, g)` receives the per-observation parameter
// matrix (n x k) of group g and returns the n predictions. Prediction j is
// assumed to depend only on row j of `Phi` (or, when all rows are equal because
// there is a single level, on any function of the common phi).
//
// Variance-function covariate: the one stored in the group, or the current
// conditional fitted values when `var_use_fitted` is set (updated on every
// outer iteration).
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <vector>

#include "lme_fit.hpp"

namespace mixedfit {

/// One nested random-effects level of an `NlmeGroup`.
struct NlmeLevel {
    Eigen::MatrixXd B;        ///< k x q_l design for this level.
    /// Optional row-varying random design: (n*k) x q_l, where rows
    /// [j*k, (j+1)*k) form B_j (row j of the group). Empty means B is used for
    /// every row; B still defines the dimensions (k x q_l).
    Eigen::MatrixXd B_row;
    std::vector<int> sub;     ///< Subgroup index of each row within the enclosing group.
};

/// Data for one group of a nonlinear mixed-effects model.
struct NlmeGroup {
    int label = 0;                          ///< Group label.
    Eigen::VectorXd y;                      ///< Response values.
    Eigen::MatrixXd A;                      ///< k x p: fixed effects -> parameters.
    Eigen::MatrixXd B;                      ///< k x q_0: outer random-effect level.
    /// Row-varying covariates (as in `Asym ~ x` in nlme): phi_j = A_j beta + B_j b.
    /// When non-empty, A_row is (n*k) x p and B_row is (n*k) x q_0, with
    /// A_j = A_row.middleRows(j*k, k); A and B still define the dimensions.
    Eigen::MatrixXd A_row, B_row;
    std::vector<NlmeLevel> inner;           ///< Nested levels 1..
    Eigen::VectorXd time;                   ///< Time values (used by correlation structures).
    Eigen::MatrixXd coords;                 ///< Coordinates for spatial correlation (empty -> time).
    std::vector<int> cor_sub;               ///< Correlation at an inner level: block id of each row (empty -> whole group).
    /// naPattern: use[i] = 1 if row i enters the fit, 0 if f is only evaluated
    /// there (it contributes neither residual nor correlation nor variance).
    /// Empty -> all rows. Useful with missing responses: see
    /// `na_pattern_from_response()`.
    std::vector<char> use;
    std::vector<int> strata;                ///< Variance strata.
    Eigen::VectorXd vcov;                   ///< Variance covariate.
    VarData vd;                             ///< Strata and covariates per slot (see var_func.hpp).
    std::vector<int> rows;                  ///< Original row indices, used to align group data with the full data set.
};

/// Model function: `f(Phi, group)` returns the predictions for one group.
using NlmeFn = std::function<Eigen::VectorXd(const Eigen::MatrixXd& Phi, int group)>;

/// Optional analytic Jacobian of f with respect to the parameters: (n x k),
/// column c = d pred_j / d Phi(j, c).
using NlmeJacFn = std::function<Eigen::MatrixXd(const Eigen::MatrixXd& Phi, int group)>;

/// Specification of an `nlme`-style model.
struct NlmeSpec {
    PdKind re_kind = PdKind::LogCholesky;           ///< Covariance parameterization of level 0.
    std::vector<PdKind> inner_kinds;                ///< Parameterizations of inner levels.
    std::vector<PdSpec> pd_specs;                   ///< Per-level custom parameterizations (see LmeSpec::pd_specs).
    CorSpec cor;                                    ///< Within-group correlation structure.
    VarSpec var;                                    ///< Within-group variance function.
    bool var_use_fitted = false;                    ///< Shorthand: mark the variance covariate as fitted(.) (unless Comb).
    double sigma_fixed = 0.0;                       ///< > 0: hold sigma fixed in the LME step.
    /// Parallelize (OpenMP) the per-group loops of PNLS and linearization. This
    /// requires f(Phi, g) to be thread-safe (no shared mutable state).
    bool parallel = false;
    bool reml = false;                              ///< Use REML instead of ML.
    BfgsOptions opt = [] {
        BfgsOptions o;
        o.gtol = 1e-6;
        o.fd_step = 1e-5;
        o.ftol = 1e-13;
        return o;
    }();
    bool analytic_grad = true;                      ///< Analytic gradient of the profile in the LME step.
    Optimizer optimizer = Optimizer::Internal;      ///< Optimization back end.
    double theta_bound = 15.0;                      ///< Box bound on the variance parameters.
    int max_outer = 50;                             ///< Maximum outer (PNLS + LME) iterations.
    int max_pnls = 50;                              ///< Maximum PNLS iterations per outer iteration.
    double tol = 1e-5;                              ///< Outer convergence tolerance.
    /// Relative-offset criterion (Bates & Watts 1988) used to stop PNLS before
    /// taking the step:
    /// sqrt(||J d||^2/ncol) / sqrt(||e + J d||^2/(nrow-ncol)) < pnls_tol.
    /// 0 = run PNLS to strict convergence. nlme uses 1e-3 by default.
    double pnls_tol = 0.0;
};

/// Results of an `nlme` fit.
struct NlmeResult {
    Eigen::VectorXd beta, se;                       ///< Fixed effects and standard errors.
    Eigen::VectorXd beta_pnls, beta_lme;            ///< beta from the last PNLS step / last LME (linearized GLS) step.
    Eigen::MatrixXd vcov_beta;                      ///< Variance-covariance matrix of beta.
    Eigen::MatrixXd vcov_theta;                     ///< Asymptotic covariance of (free theta, log sigma).
    double sigma2 = std::numeric_limits<double>::quiet_NaN();  ///< Residual variance.
    Eigen::VectorXd theta;                          ///< Covariance parameters.
    std::vector<Eigen::MatrixXd> D_levels, Sigma_levels;  ///< Relative and absolute covariance matrices per level.
    Eigen::MatrixXd D, Sigma_b;                     ///< Level-0 covariance matrices.
    std::vector<Eigen::VectorXd> b;                 ///< Per group: [b_0 | level-1 units | ...].
    double loglik = std::numeric_limits<double>::quiet_NaN();  ///< LME approximation at the solution.
    bool converged = false;                         ///< Whether the outer loop converged.
    int outer_iterations = 0;                       ///< Outer iterations performed.
    std::string warning;                            ///< Non-fatal diagnostic messages.
    Eigen::VectorXd fitted_pop, fitted_cond;        ///< Fitted values (all rows, including those excluded by naPattern), groups concatenated.
    std::vector<int> order;                         ///< Original row index of each stacked fitted value.
    std::vector<char> used;                         ///< Aligned with `order`: 1 if the row was used in the fit.
    Eigen::MatrixXd X_lin;                          ///< (N used) x p: d f / d beta at the solution, stacked by group (used-row order).
    int n_used = 0;                                 ///< Number of observations that entered the fit.
};

/// Build groups from stacked data.
///
/// @param y       Response vector.
/// @param labels  Group label per row.
/// @param A       Fixed-effect design (k x p).
/// @param B0      Level-0 random-effect design (k x q_0).
/// @param time    Optional time values.
/// @param strata  Optional variance strata per row.
/// @param vcov    Optional variance covariate per row.
/// @param coords  Optional spatial coordinates per row.
/// @return One `NlmeGroup` per distinct label, preserving first-appearance order.
inline std::vector<NlmeGroup> make_nlme_groups(const Eigen::VectorXd& y, const std::vector<int>& labels,
                                               const Eigen::MatrixXd& A, const Eigen::MatrixXd& B0,
                                               const Eigen::VectorXd& time = Eigen::VectorXd(),
                                               const std::vector<int>& strata = {},
                                               const Eigen::VectorXd& vcov = Eigen::VectorXd(),
                                               const Eigen::MatrixXd& coords = Eigen::MatrixXd()) {
    const int N = static_cast<int>(y.size());
    std::map<int, int> pos;
    std::vector<std::vector<int>> idx;
    std::vector<int> lab;
    for (int i = 0; i < N; ++i) {
        auto it = pos.find(labels[static_cast<size_t>(i)]);
        if (it == pos.end()) {
            it = pos.emplace(labels[static_cast<size_t>(i)], static_cast<int>(idx.size())).first;
            idx.emplace_back();
            lab.push_back(labels[static_cast<size_t>(i)]);
        }
        idx[static_cast<size_t>(it->second)].push_back(i);
    }
    std::vector<NlmeGroup> out(idx.size());
    for (size_t g = 0; g < idx.size(); ++g) {
        const auto& r = idx[g];
        out[g].label = lab[g];
        out[g].rows = r;
        out[g].y = y(r);
        out[g].A = A;
        out[g].B = B0;
        if (time.size() != 0) out[g].time = time(r);
        if (vcov.size() != 0) out[g].vcov = vcov(r);
        if (coords.cols() > 0) out[g].coords = coords(r, Eigen::placeholders::all);
        if (!strata.empty()) {
            out[g].strata.resize(r.size());
            for (size_t k = 0; k < r.size(); ++k) out[g].strata[k] = strata[static_cast<size_t>(r[k])];
        }
    }
    return out;
}

/// Add an inner random-effects level to every group.
///
/// @param groups Groups to modify.
/// @param B      Design matrix (k x q_l), shared by all groups.
/// @param sub    Subgroup index per original row; each group takes the entries
///               corresponding to its own rows.
inline void add_nlme_inner_level(std::vector<NlmeGroup>& groups, const Eigen::MatrixXd& B,
                                 const std::vector<int>& sub) {
    for (auto& g : groups) {
        NlmeLevel lv;
        lv.B = B;
        lv.sub.resize(g.rows.size());
        for (size_t k = 0; k < g.rows.size(); ++k) lv.sub[k] = sub[static_cast<size_t>(g.rows[k])];
        g.inner.push_back(std::move(lv));
    }
}

/// Correlation at an inner level: assign each row the block `sub[original_row]`
/// (see `add_cor_sub` in lme_fit.hpp).
///
/// @param groups Groups to modify.
/// @param sub    Block id per original row.
/// @throws std::invalid_argument If the groups do not store their original row indices.
inline void add_nlme_cor_sub(std::vector<NlmeGroup>& groups, const std::vector<int>& sub) {
    for (auto& g : groups) {
        if (g.rows.empty()) throw std::invalid_argument("add_nlme_cor_sub: groups do not store row indices");
        g.cor_sub.resize(g.rows.size());
        for (size_t k = 0; k < g.rows.size(); ++k) g.cor_sub[k] = sub[static_cast<size_t>(g.rows[k])];
        // With no explicit times, use increasing ordinals within each block.
        if (g.time.size() == 0) detail::block_ordinal_time(g.cor_sub, g.time);
    }
}

/// naPattern = ~ !is.na(y): rows with a non-finite response are excluded from
/// the fit, but f is still evaluated on them (groups keep all their rows; see
/// `NlmeGroup::use`). `NaN` responses are replaced by 0 so they do not
/// contaminate any computation.
///
/// @param groups Groups to modify.
inline void na_pattern_from_response(std::vector<NlmeGroup>& groups) {
    for (auto& g : groups) {
        g.use.assign(static_cast<size_t>(g.y.size()), 1);
        for (Eigen::Index i = 0; i < g.y.size(); ++i)
            if (!std::isfinite(g.y[i])) { g.use[static_cast<size_t>(i)] = 0; g.y[i] = 0.0; }
    }
}

/// General naPattern: `keep[original_row] != 0` marks the row as used. Requires
/// the groups to store their original row indices.
///
/// @param groups Groups to modify.
/// @param keep   Per-original-row inclusion flags.
/// @throws std::invalid_argument If the groups do not store their original row indices.
inline void set_na_pattern(std::vector<NlmeGroup>& groups, const std::vector<char>& keep) {
    for (auto& g : groups) {
        if (g.rows.empty()) throw std::invalid_argument("set_na_pattern: groups do not store row indices");
        g.use.resize(g.rows.size());
        for (size_t k = 0; k < g.rows.size(); ++k) g.use[k] = keep[static_cast<size_t>(g.rows[k])] ? 1 : 0;
    }
}

/// Nonlinear mixed-effects model with nested random-effect levels.
class NlmeModel {
public:
    /// Construct the model and precompute the per-group bookkeeping.
    ///
    /// @param groups Grouped data.
    /// @param f      Model function.
    /// @param spec   Model specification.
    /// @param jac    Optional analytic Jacobian; finite differences are used
    ///               when empty.
    /// @throws std::invalid_argument On missing groups or inconsistent
    ///         dimensions of A, B, inner levels, `use`, `cor_sub`, or row designs.
    NlmeModel(std::vector<NlmeGroup> groups, NlmeFn f, NlmeSpec spec, NlmeJacFn jac = nullptr)
        : g_(std::move(groups)), f_(std::move(f)), jac_(std::move(jac)), spec_(std::move(spec)) {
        if (g_.empty()) throw std::invalid_argument("NlmeModel: no groups");
        p_ = static_cast<int>(g_[0].A.cols());
        k_ = static_cast<int>(g_[0].A.rows());
        nlev_ = 1 + static_cast<int>(g_[0].inner.size());
        q_.assign(static_cast<size_t>(nlev_), 0);
        q_[0] = static_cast<int>(g_[0].B.cols());
        for (int l = 1; l < nlev_; ++l) q_[static_cast<size_t>(l)] = static_cast<int>(g_[0].inner[static_cast<size_t>(l - 1)].B.cols());
        kinds_ = resolve_pd_specs(spec_.pd_specs, spec_.re_kind, spec_.inner_kinds, nlev_);

        int mx = 2, maxpos = 0;
        N_ = 0;
        info_.resize(g_.size());
        for (size_t gi = 0; gi < g_.size(); ++gi) {
            auto& g = g_[gi];
            auto& in = info_[gi];
            const int n = static_cast<int>(g.y.size());
            if (g.A.cols() != p_ || g.A.rows() != k_ || g.B.rows() != k_ || g.B.cols() != q_[0] ||
                static_cast<int>(g.inner.size()) != nlev_ - 1)
                throw std::invalid_argument("NlmeModel: inconsistent A/B/levels");
            if (g.time.size() == 0) g.time = Eigen::VectorXd::LinSpaced(n, 0, n - 1);
            for (int i = 0; i < n; ++i) maxpos = std::max(maxpos, static_cast<int>(std::lround(g.time[i])));
            if (g.vd.strata.empty() && !g.strata.empty()) g.vd.strata.push_back(g.strata);
            if (g.vd.cov.empty() && g.vcov.size() > 0) g.vd.cov.push_back(g.vcov);
            if (!g.use.empty() && static_cast<int>(g.use.size()) != n) throw std::invalid_argument("NlmeModel: use (naPattern) has the wrong length");
            if (!g.cor_sub.empty() && static_cast<int>(g.cor_sub.size()) != n) throw std::invalid_argument("NlmeModel: cor_sub has the wrong length");
            for (int i = 0; i < n; ++i)
                if (g.use.empty() || g.use[static_cast<size_t>(i)]) in.uidx.push_back(i);
            in.nu = static_cast<int>(in.uidx.size());
            if (in.nu == 0) throw std::invalid_argument("NlmeModel: group has no used observations");
            in.all_used = (in.nu == n);
            in.pos.assign(static_cast<size_t>(n), -1);
            for (int k = 0; k < in.nu; ++k) in.pos[static_cast<size_t>(in.uidx[static_cast<size_t>(k)])] = k;
            in.yu = Eigen::VectorXd(g.y(in.uidx));
            in.timeu = Eigen::VectorXd(g.time(in.uidx));
            if (g.coords.cols() > 0) in.coordsu = Eigen::MatrixXd(g.coords(in.uidx, Eigen::placeholders::all));
            in.vdu = g.vd;
            for (auto& st : in.vdu.strata) if (!st.empty()) { std::vector<int> t2; for (int i : in.uidx) t2.push_back(st[static_cast<size_t>(i)]); st = std::move(t2); }
            for (auto& cv : in.vdu.cov) if (cv.size() == n) cv = Eigen::VectorXd(cv(in.uidx));
            if (!g.cor_sub.empty()) for (int i : in.uidx) in.cor_sub_u.push_back(g.cor_sub[static_cast<size_t>(i)]);
            in.cor_blocks = make_cor_blocks(in.cor_sub_u);
            in.n = n;
            in.row_off = N_;
            N_ += in.nu;
            N_all_ += n;
            {
                int bmax = in.nu;
                if (!in.cor_blocks.empty()) { bmax = 0; for (const auto& b : in.cor_blocks) bmax = std::max(bmax, static_cast<int>(b.size())); }
                mx = std::max(mx, bmax);
            }
            in.sub_u.resize(static_cast<size_t>(nlev_ - 1));
            if (g.A_row.size()) {
                if (g.A_row.rows() != static_cast<Eigen::Index>(n) * k_ || g.A_row.cols() != p_) throw std::invalid_argument("NlmeModel: A_row must be (n*k) x p");
                in.Aru = in.all_used ? g.A_row : reduce_blocks(g.A_row, in.uidx, k_);
            }
            if (g.B_row.size()) {
                if (g.B_row.rows() != static_cast<Eigen::Index>(n) * k_ || g.B_row.cols() != q_[0]) throw std::invalid_argument("NlmeModel: B_row must be (n*k) x q0");
                in.Bru = in.all_used ? g.B_row : reduce_blocks(g.B_row, in.uidx, k_);
            }
            in.Binru.assign(static_cast<size_t>(nlev_ - 1), Eigen::MatrixXd());
            for (int l = 1; l < nlev_; ++l)
                if (g.inner[static_cast<size_t>(l - 1)].B_row.size()) {
                    const Eigen::MatrixXd& Br = g.inner[static_cast<size_t>(l - 1)].B_row;
                    if (Br.rows() != static_cast<Eigen::Index>(n) * k_ || Br.cols() != q_[static_cast<size_t>(l)]) throw std::invalid_argument("NlmeModel: B_row of an inner level must be (n*k) x q_l");
                    in.Binru[static_cast<size_t>(l - 1)] = in.all_used ? Br : reduce_blocks(Br, in.uidx, k_);
                }
            in.nb = 0;
            if (q_[0] > 0) { in.units.push_back({0, 0, in.nb, q_[0]}); in.nb += q_[0]; }
            in.sub_rows.resize(static_cast<size_t>(nlev_ - 1));
            in.unit_of.resize(static_cast<size_t>(nlev_ - 1));
            for (int l = 1; l < nlev_; ++l) {
                const auto& lv = g.inner[static_cast<size_t>(l - 1)];
                if (lv.B.rows() != k_ || lv.B.cols() != q_[static_cast<size_t>(l)] || static_cast<int>(lv.sub.size()) != n)
                    throw std::invalid_argument("NlmeModel: inconsistent inner level");
                std::map<int, int> pos;
                auto& sr = in.sub_rows[static_cast<size_t>(l - 1)];
                for (int i = 0; i < n; ++i) {
                    auto it = pos.find(lv.sub[static_cast<size_t>(i)]);
                    if (it == pos.end()) { it = pos.emplace(lv.sub[static_cast<size_t>(i)], static_cast<int>(sr.size())).first; sr.emplace_back(); }
                    sr[static_cast<size_t>(it->second)].push_back(i);
                }
                for (int i : in.uidx) in.sub_u[static_cast<size_t>(l - 1)].push_back(lv.sub[static_cast<size_t>(i)]);
                for (size_t s = 0; s < sr.size(); ++s) {
                    if (q_[static_cast<size_t>(l)] > 0) {
                        in.unit_of[static_cast<size_t>(l - 1)].push_back(static_cast<int>(in.units.size()));
                        in.units.push_back({l, static_cast<int>(s), in.nb, q_[static_cast<size_t>(l)]});
                        in.nb += q_[static_cast<size_t>(l)];
                    } else {
                        in.unit_of[static_cast<size_t>(l - 1)].push_back(-1);
                    }
                }
            }
        }
        spec_.cor.max_group_size = mx;
        if (spec_.cor.spatial()) {
            std::vector<const Eigen::MatrixXd*> cs;
            std::vector<const Eigen::VectorXd*> ts;
            for (const auto& in : info_) { cs.push_back(&in.coordsu); ts.push_back(&in.timeu); }
            set_spatial_start(spec_.cor, cs, ts);
        }
        if (spec_.cor.kind == CorKind::Symm) spec_.cor.n_pos = maxpos + 1;
        if (spec_.var_use_fitted && spec_.var.kind != VarKind::Comb && spec_.var.needs_covariate()) spec_.var.fitted = true;
        {
            std::vector<const VarData*> vds;
            for (const auto& in : info_) vds.push_back(&in.vdu);
            configure_var(spec_.var, vds);
        }
    }

    /// Accessors (used by lm_list.hpp and diagnostics).
    const std::vector<NlmeGroup>& groups() const { return g_; }
    const NlmeFn& fn() const { return f_; }
    const NlmeJacFn& jac_fn() const { return jac_; }
    int n_fixed() const { return p_; }
    int n_params() const { return k_; }
    int n_obs() const { return N_; }
    int n_obs_all() const { return N_all_; }

    /// Default starting values for theta.
    Eigen::VectorXd default_theta() const { return theta_start(kinds_, q_, spec_.cor, spec_.var); }

    /// Starting theta with the given relative level-0 covariance
    /// (`D0 = Sigma_b / sigma^2`); the remaining entries keep their defaults.
    Eigen::VectorXd theta_with_D0(const Eigen::MatrixXd& D0) const {
        Eigen::VectorXd th = default_theta();
        if (q_[0] > 0 && D0.rows() == q_[0] && D0.cols() == q_[0]) {
            const Eigen::VectorXd t0 = detail::pd_theta_from_D(kinds_[0], q_[0], D0);
            if (t0.size() == pd_nparams(kinds_[0], q_[0])) th.segment(0, t0.size()) = t0;
        }
        return th;
    }

    /// Parameter matrix (k x n) of each row of a group for `(beta, b)`; useful
    /// for carrying `nlsList` results to the group level.
    Eigen::MatrixXd phi(int gi, const Eigen::VectorXd& beta, const Eigen::VectorXd& b) const { return phi_matrix(gi, beta, b); }

    /// Fit the model by the alternating PNLS/LME algorithm.
    ///
    /// @param beta0  Fixed-effect starting values (length p).
    /// @param theta0 Optional covariance-parameter starting values; ignored if
    ///               the length does not match.
    /// @return The fit results.
    /// @throws std::invalid_argument If `beta0` has the wrong length.
    /// @throws std::runtime_error If the within-group covariance is not positive
    ///         definite, the theta dimension is inconsistent, the model function
    ///         returns the wrong length, or the initial values are infeasible.
    NlmeResult fit(const Eigen::VectorXd& beta0, const Eigen::VectorXd* theta0 = nullptr) const {
        const int m = static_cast<int>(g_.size());
        if (beta0.size() != p_) throw std::invalid_argument("NlmeModel::fit: beta0 has the wrong length");

        Eigen::VectorXd beta = beta0, beta_prev = beta0;
        std::vector<Eigen::VectorXd> b(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) b[static_cast<size_t>(i)] = Eigen::VectorXd::Zero(info_[static_cast<size_t>(i)].nb);
        Eigen::VectorXd theta = theta_start(kinds_, q_, spec_.cor, spec_.var);
        if (theta0 && theta0->size() == theta.size()) theta = *theta0;

        LmeSpec ls;
        ls.re_kind = spec_.re_kind;
        ls.inner_kinds = spec_.inner_kinds;
        ls.pd_specs = spec_.pd_specs;
        ls.cor = spec_.cor;
        ls.var = strip_fitted(spec_.var);
        ls.sigma_fixed = spec_.sigma_fixed;
        ls.reml = spec_.reml;
        ls.opt = spec_.opt;
        ls.analytic_grad = spec_.analytic_grad;
        ls.diagnose = false;                          // diagnose once, at the end
        ls.optimizer = spec_.optimizer;
        ls.theta_bound = spec_.theta_bound;

        int off_cor = 0;
        std::vector<int> re_off(static_cast<size_t>(nlev_), 0), re_n(static_cast<size_t>(nlev_), 0);
        for (int l = 0; l < nlev_; ++l) {
            re_n[static_cast<size_t>(l)] = q_[static_cast<size_t>(l)] > 0 ? pd_nparams(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)]) : 0;
            re_off[static_cast<size_t>(l)] = off_cor;
            off_cor += re_n[static_cast<size_t>(l)];
        }
        const int n_cor = spec_.cor.n_params(), n_var = spec_.var.n_params();

        NlmeResult out;
        LmeResult lres;
        Eigen::MatrixXd x_lin;
        for (int it = 1; it <= spec_.max_outer; ++it) {
            out.outer_iterations = it;

            // ---- theta-dependent matrices ----
            std::vector<Eigen::MatrixXd> LD(static_cast<size_t>(nlev_)), LDinv(static_cast<size_t>(nlev_));
            for (int l = 0; l < nlev_; ++l) {
                if (q_[static_cast<size_t>(l)] == 0) continue;
                LD[static_cast<size_t>(l)] = pd_factor(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], theta.segment(re_off[static_cast<size_t>(l)], re_n[static_cast<size_t>(l)]));
                LDinv[static_cast<size_t>(l)] = LD[static_cast<size_t>(l)].triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(q_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)]));
            }
            const Eigen::VectorXd tc = theta.segment(off_cor, n_cor), tv = theta.segment(off_cor + n_cor, n_var);
            std::vector<VarData> vc(static_cast<size_t>(m));
            std::vector<std::pair<const VarSpec*, int>> vcomps;
            var_components(spec_.var, 0, vcomps);
            // Within-group variance weights: with a fitted(.) covariate they
            // depend on (beta, b).
            auto make_LR = [&](const Eigen::VectorXd& be, const std::vector<Eigen::VectorXd>& bb,
                               std::vector<VarData>& vco) {
                std::vector<Eigen::MatrixXd> LRo(static_cast<size_t>(m));
                for (int i = 0; i < m; ++i) {
                    const auto& in = info_[static_cast<size_t>(i)];
                    const int n = in.nu;
                    vco[static_cast<size_t>(i)] = in.vdu;
                    if (spec_.var.has_fitted()) {
                        const Eigen::VectorXd fv = sel(f_(phi_matrix(i, be, bb[static_cast<size_t>(i)]), i), in);
                        for (const auto& c : vcomps)
                            if (c.first->fitted) {
                                if (vco[static_cast<size_t>(i)].cov.size() <= static_cast<size_t>(c.first->cov_slot)) vco[static_cast<size_t>(i)].cov.resize(static_cast<size_t>(c.first->cov_slot + 1));
                                vco[static_cast<size_t>(i)].cov[static_cast<size_t>(c.first->cov_slot)] = fv;
                            }
                    }
                    const Eigen::VectorXd s = var_scales(spec_.var, tv, vco[static_cast<size_t>(i)], n);
                    const Eigen::MatrixXd Lam = s.asDiagonal() * cor_matrix_blocks(spec_.cor, tc, in.timeu, in.coordsu, in.cor_blocks) * s.asDiagonal();
                    Eigen::LLT<Eigen::MatrixXd> llt(Lam);
                    if (llt.info() != Eigen::Success) throw std::runtime_error("NLME: within-group covariance is not positive definite");
                    LRo[static_cast<size_t>(i)] = llt.matrixL();
                }
                return LRo;
            };
            std::vector<Eigen::MatrixXd> LR = make_LR(beta, b, vc);
            const bool refit_w = spec_.var.has_fitted();
            std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&, const std::vector<Eigen::VectorXd>&)> lr_fn;
            if (refit_w)
                lr_fn = [&](const Eigen::VectorXd& be, const std::vector<Eigen::VectorXd>& bb) {
                    std::vector<VarData> tmp(static_cast<size_t>(m));
                    return make_LR(be, bb, tmp);
                };

            // ---- PNLS step ----
            pnls(beta, b, LD, LDinv, LR, refit_w ? &lr_fn : nullptr);
            if (refit_w) LR = make_LR(beta, b, vc);      // fitted(.) covariate at the PNLS solution

            // ---- linearization and LME step ----
            std::vector<LmeGroup> lg = linearize_groups(beta, b, vc);
            x_lin.resize(N_, p_);          // linearized design (d f / d beta) of the used rows, stacked by group; used by nlme's getFixDF
            {
                Eigen::Index r0 = 0;
                for (const auto& g : lg) { x_lin.middleRows(r0, g.X.rows()) = g.X; r0 += g.X.rows(); }
            }
            LmeModel lm(std::move(lg), ls);
            lres = lm.fit(&theta);
            if (lres.theta.size() != theta.size()) throw std::runtime_error("NLME: theta size mismatch");

            const Eigen::VectorXd thn = lres.theta;
            const double dth = theta.size() ? (thn - theta).cwiseAbs().maxCoeff() : 0.0;
            const double dbeta = ((beta - beta_prev).array().abs() / (1.0 + beta.array().abs())).maxCoeff();
            theta = thn;
            if (it > 1 && dth < spec_.tol && dbeta < spec_.tol) {
                out.converged = true;
                break;
            }
            beta_prev = beta;
        }

        out.X_lin = x_lin;
        out.beta_pnls = beta;
        out.beta_lme = lres.beta;
        out.beta = beta;
        out.se = lres.se;
        out.vcov_beta = lres.vcov_beta;
        out.vcov_theta = lres.vcov_theta;
        out.sigma2 = lres.sigma2;
        out.theta = lres.theta;
        out.D_levels = lres.D_levels;
        out.Sigma_levels = lres.Sigma_levels;
        out.D = lres.D;
        out.Sigma_b = lres.Sigma_b;
        out.loglik = lres.loglik;
        out.warning = lres.warning;
        out.b = b;
        out.fitted_pop.resize(N_all_);
        out.fitted_cond.resize(N_all_);
        out.order.resize(static_cast<size_t>(N_all_));
        out.used.assign(static_cast<size_t>(N_all_), 1);
        out.n_used = N_;
        int off_all = 0;
        for (int i = 0; i < m; ++i) {
            const auto& in = info_[static_cast<size_t>(i)];
            const Eigen::VectorXd zero = Eigen::VectorXd::Zero(in.nb);
            out.fitted_pop.segment(off_all, in.n) = f_(phi_matrix(i, beta, zero), i);
            out.fitted_cond.segment(off_all, in.n) = f_(phi_matrix(i, beta, b[static_cast<size_t>(i)]), i);
            for (int r = 0; r < in.n; ++r) {
                out.order[static_cast<size_t>(off_all + r)] = g_[static_cast<size_t>(i)].rows.empty() ? off_all + r : g_[static_cast<size_t>(i)].rows[static_cast<size_t>(r)];
                out.used[static_cast<size_t>(off_all + r)] = in.pos[static_cast<size_t>(r)] >= 0 ? 1 : 0;
            }
            off_all += in.n;
        }
        return out;
    }

    /// Working linear model (`LmeModel`) linearized at `(beta, b)`; intended for
    /// diagnostics.
    LmeModel linearized_lme(const Eigen::VectorXd& beta, const std::vector<Eigen::VectorXd>& b) const {
        LmeSpec ls;
        ls.re_kind = spec_.re_kind; ls.inner_kinds = spec_.inner_kinds; ls.pd_specs = spec_.pd_specs; ls.cor = spec_.cor;
        ls.var = strip_fitted(spec_.var); ls.reml = spec_.reml; ls.opt = spec_.opt; ls.sigma_fixed = spec_.sigma_fixed;
        std::vector<VarData> vc(g_.size());
        for (size_t i = 0; i < g_.size(); ++i) vc[i] = g_[i].vd;
        return LmeModel(linearize_groups(beta, b, vc), ls);
    }

private:
    /// One unit of random effects (a level and a subgroup within it).
    struct Unit { int level; int sub; int off; int q; };

    /// Per-group bookkeeping after applying the NA pattern.
    struct Info {
        int n = 0, nu = 0, nb = 0, row_off = 0;   ///< Group rows, used rows, random effects, and offset among used rows.
        std::vector<int> uidx, pos;                ///< Used rows (ascending) and row -> index among used rows (-1 if unused).
        bool all_used = true;                      ///< Whether every row is used.
        Eigen::MatrixXd Aru, Bru;                  ///< A_row / B_row restricted to used rows (empty when the design is constant).
        std::vector<Eigen::MatrixXd> Binru;        ///< Same for each inner level.
        Eigen::VectorXd yu, timeu;                 ///< Used responses and times.
        Eigen::MatrixXd coordsu;                   ///< Used coordinates.
        VarData vdu;                               ///< Variance data restricted to used rows.
        std::vector<int> cor_sub_u;                ///< Correlation block ids of used rows.
        std::vector<std::vector<int>> cor_blocks;  ///< Correlation blocks.
        std::vector<std::vector<int>> sub_u;       ///< [level-1] -> subgroup id of the used rows.
        std::vector<Unit> units;                   ///< Random-effect units.
        std::vector<std::vector<std::vector<int>>> sub_rows;   ///< [level-1][sub] -> rows.
        std::vector<std::vector<int>> unit_of;                 ///< [level-1][sub] -> index in `units`.
    };


    /// Groups of the working linear model at `(beta, b)`:
    /// X = J A, Z_l = J B_l, w = y - f + X beta + sum_l Z_l b_l.
    std::vector<LmeGroup> linearize_groups(const Eigen::VectorXd& beta, const std::vector<Eigen::VectorXd>& b,
                                           const std::vector<VarData>& vc) const {
        const int m = static_cast<int>(g_.size());
        std::vector<LmeGroup> lg(static_cast<size_t>(m));
        std::exception_ptr ep;
#pragma omp parallel for schedule(dynamic) if (spec_.parallel)
        for (int i = 0; i < m; ++i) {
          try {
                const auto& g = g_[static_cast<size_t>(i)];
                const auto& in = info_[static_cast<size_t>(i)];
                const int n = in.nu;
                const Eigen::MatrixXd Phi = phi_matrix(i, beta, b[static_cast<size_t>(i)]);
                const Eigen::VectorXd pred = sel(f_(Phi, i), in);
                const Eigen::MatrixXd Jp = selrows(jac(Phi, i), in);
                LmeGroup& L = lg[static_cast<size_t>(i)];
                L.label = g.label;
                if (!g.rows.empty()) { L.rows.clear(); for (int r : in.uidx) L.rows.push_back(g.rows[static_cast<size_t>(r)]); }
                L.time = in.timeu;
                L.coords = in.coordsu;
                L.cor_sub = in.cor_sub_u;
                L.vd = vc[static_cast<size_t>(i)];
                L.X = prodA(Jp, g, in);
                L.Z = q_[0] > 0 ? prodB0(Jp, g, in) : Eigen::MatrixXd::Zero(n, 0);
                L.y = in.yu - pred + L.X * beta;
                if (q_[0] > 0) L.y += L.Z * b[static_cast<size_t>(i)].segment(0, q_[0]);
                for (int l = 1; l < nlev_; ++l) {
                    LmeLevel lv;
                    lv.Z = q_[static_cast<size_t>(l)] > 0 ? prodBl(Jp, g, in, l) : Eigen::MatrixXd::Zero(n, 0);
                    lv.sub = in.sub_u[static_cast<size_t>(l - 1)];
                    if (q_[static_cast<size_t>(l)] > 0)
                        for (size_t s = 0; s < in.sub_rows[static_cast<size_t>(l - 1)].size(); ++s) {
                            const Unit& u = in.units[static_cast<size_t>(in.unit_of[static_cast<size_t>(l - 1)][s])];
                            for (int r : in.sub_rows[static_cast<size_t>(l - 1)][s]) {
                                const int pr = in.pos[static_cast<size_t>(r)];
                                if (pr >= 0) L.y[pr] += lv.Z.row(pr).dot(b[static_cast<size_t>(i)].segment(u.off, u.q));
                            }
                        }
                    L.inner.push_back(std::move(lv));
                }
          } catch (...) {
#pragma omp critical(mixedfit_nlme_flag)
            if (!ep) ep = std::current_exception();
          }
        }
        if (ep) std::rethrow_exception(ep);
        return lg;
    }

    /// Row-wise product: row j of J (n x k) times block S.middleRows(j*k, k)
    /// (k x m) -> n x m.
    static Eigen::MatrixXd rowprod(const Eigen::MatrixXd& J, const Eigen::MatrixXd& S, int k) {
        Eigen::MatrixXd out(J.rows(), S.cols());
        for (Eigen::Index j = 0; j < J.rows(); ++j) out.row(j) = J.row(j) * S.middleRows(j * k, k);
        return out;
    }

    /// Keep the k-row blocks corresponding to the given rows.
    static Eigen::MatrixXd reduce_blocks(const Eigen::MatrixXd& S, const std::vector<int>& idx, int k) {
        Eigen::MatrixXd out(static_cast<Eigen::Index>(idx.size()) * k, S.cols());
        for (size_t r = 0; r < idx.size(); ++r) out.middleRows(static_cast<Eigen::Index>(r) * k, k) = S.middleRows(static_cast<Eigen::Index>(idx[r]) * k, k);
        return out;
    }

    /// d f/d phi (used rows) times the fixed or row-varying designs (A, B) or
    /// (A_row, B_row).
    Eigen::MatrixXd prodA(const Eigen::MatrixXd& J, const NlmeGroup& g, const Info& in) const { return in.Aru.size() ? rowprod(J, in.Aru, k_) : Eigen::MatrixXd(J * g.A); }
    Eigen::MatrixXd prodB0(const Eigen::MatrixXd& J, const NlmeGroup& g, const Info& in) const { return in.Bru.size() ? rowprod(J, in.Bru, k_) : Eigen::MatrixXd(J * g.B); }
    Eigen::MatrixXd prodBl(const Eigen::MatrixXd& J, const NlmeGroup& g, const Info& in, int l) const {
        return in.Binru[static_cast<size_t>(l - 1)].size() ? rowprod(J, in.Binru[static_cast<size_t>(l - 1)], k_) : Eigen::MatrixXd(J * g.inner[static_cast<size_t>(l - 1)].B);
    }

    /// Used rows (naPattern) of a vector / matrix defined over all rows of the
    /// group.
    static Eigen::VectorXd sel(const Eigen::VectorXd& v, const Info& in) {
        return in.all_used ? v : Eigen::VectorXd(v(in.uidx));
    }
    static Eigen::MatrixXd selrows(const Eigen::MatrixXd& M, const Info& in) {
        return in.all_used ? M : Eigen::MatrixXd(M(in.uidx, Eigen::placeholders::all));
    }

    /// Per-row parameter matrix (n x k) of a group for `(beta, b)`.
    Eigen::MatrixXd phi_matrix(int gi, const Eigen::VectorXd& beta, const Eigen::VectorXd& b) const {
        const auto& g = g_[static_cast<size_t>(gi)];
        const auto& in = info_[static_cast<size_t>(gi)];
        using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::MatrixXd Phi;
        if (g.A_row.size()) {                                   // row-level covariates in the fixed effects
            const Eigen::VectorXd v = g.A_row * beta;
            Phi = Eigen::Map<const RowMat>(v.data(), in.n, k_);
        } else {
            Phi = Eigen::VectorXd::Ones(in.n) * Eigen::VectorXd(g.A * beta).transpose();
        }
        if (q_[0] > 0) {
            if (g.B_row.size()) {
                const Eigen::VectorXd v = g.B_row * b.segment(0, q_[0]);
                Phi += Eigen::Map<const RowMat>(v.data(), in.n, k_);
            } else {
                Phi.rowwise() += Eigen::VectorXd(g.B * b.segment(0, q_[0])).transpose();
            }
        }
        for (int l = 1; l < nlev_; ++l) {
            if (q_[static_cast<size_t>(l)] == 0) continue;
            const Eigen::MatrixXd& Br = g.inner[static_cast<size_t>(l - 1)].B_row;
            for (size_t s = 0; s < in.sub_rows[static_cast<size_t>(l - 1)].size(); ++s) {
                const Unit& u = in.units[static_cast<size_t>(in.unit_of[static_cast<size_t>(l - 1)][s])];
                if (Br.size()) {
                    for (int r : in.sub_rows[static_cast<size_t>(l - 1)][s]) Phi.row(r) += (Br.middleRows(static_cast<Eigen::Index>(r) * k_, k_) * b.segment(u.off, u.q)).transpose();
                } else {
                    const Eigen::VectorXd c = g.inner[static_cast<size_t>(l - 1)].B * b.segment(u.off, u.q);
                    for (int r : in.sub_rows[static_cast<size_t>(l - 1)][s]) Phi.row(r) += c.transpose();
                }
            }
        }
        return Phi;
    }

    /// d f / d phi: analytic when `jac_` was supplied, otherwise by central
    /// differences (column c = derivative with respect to phi_c).
    Eigen::MatrixXd jac(const Eigen::MatrixXd& Phi, int gi) const {
        if (jac_) return jac_(Phi, gi);
        const int n = info_[static_cast<size_t>(gi)].n, k = static_cast<int>(Phi.cols());
        Eigen::MatrixXd J(n, k), P = Phi;
        const double base = std::cbrt(std::numeric_limits<double>::epsilon());
        for (int c = 0; c < k; ++c) {
            const double h = base * std::max(1.0, Phi.col(c).cwiseAbs().maxCoeff());
            P.col(c).array() += h;
            const Eigen::VectorXd fp = f_(P, gi);
            P.col(c) = Phi.col(c);
            P.col(c).array() -= h;
            const Eigen::VectorXd fm = f_(P, gi);
            P.col(c) = Phi.col(c);
            J.col(c) = (fp - fm) / (2.0 * h);
        }
        return J;
    }

    /// Penalized sum of squares at `(beta, b)`, including the within-group
    /// whitening by `LR` and the random-effect penalty by `LD`.
    double pen_ss(const Eigen::VectorXd& be, const std::vector<Eigen::VectorXd>& bb,
                  const std::vector<Eigen::MatrixXd>& LD, const std::vector<Eigen::MatrixXd>& LR) const {
        const int m = static_cast<int>(g_.size());
        double s = 0.0;
        bool nonfinite = false;
        std::exception_ptr ep;
#pragma omp parallel for reduction(+ : s) schedule(dynamic) if (spec_.parallel)
        for (int i = 0; i < m; ++i) {
            try {
                const auto& g = g_[static_cast<size_t>(i)];
                const Eigen::VectorXd predf = f_(phi_matrix(i, be, bb[static_cast<size_t>(i)]), i);
                if (predf.size() != g.y.size()) throw std::runtime_error("NLME: f returned the wrong length");
                const Eigen::VectorXd res = info_[static_cast<size_t>(i)].yu - sel(predf, info_[static_cast<size_t>(i)]);
                if (!res.allFinite()) {
#pragma omp critical(mixedfit_nlme_flag)
                    nonfinite = true;
                    continue;
                }
                double si = LR[static_cast<size_t>(i)].triangularView<Eigen::Lower>().solve(res).squaredNorm();
                for (const Unit& u : info_[static_cast<size_t>(i)].units)
                    si += LD[static_cast<size_t>(u.level)].triangularView<Eigen::Lower>().solve(Eigen::VectorXd(bb[static_cast<size_t>(i)].segment(u.off, u.q))).squaredNorm();
                s += si;
            } catch (...) {
#pragma omp critical(mixedfit_nlme_flag)
                if (!ep) ep = std::current_exception();
            }
        }
        if (ep) std::rethrow_exception(ep);
        return nonfinite ? std::numeric_limits<double>::infinity() : s;
    }

    /// Callback that recomputes the within-group whitening factors at
    /// `(beta, b)`.
    using LrFn = std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&, const std::vector<Eigen::VectorXd>&)>;

    /// Gauss-Newton with step reduction; each linear step eliminates b_i by QR.
    void pnls(Eigen::VectorXd& beta, std::vector<Eigen::VectorXd>& b, const std::vector<Eigen::MatrixXd>& LD,
              const std::vector<Eigen::MatrixXd>& LDinv, const std::vector<Eigen::MatrixXd>& LR0,
              const LrFn* lr_fn = nullptr) const {
        const int m = static_cast<int>(g_.size());
        std::vector<Eigen::MatrixXd> LR = LR0;
        double S0 = pen_ss(beta, b, LD, LR);
        if (!std::isfinite(S0)) throw std::runtime_error("NLME: infeasible initial values");

        for (int it = 0; it < spec_.max_pnls; ++it) {
            if (lr_fn && it > 0) {                       // variance weights from the current fitted values
                LR = (*lr_fn)(beta, b);
                S0 = pen_ss(beta, b, LD, LR);
            }
            Eigen::MatrixXd Red(N_, p_);
            Eigen::VectorXd rhs(N_);
            std::vector<Eigen::MatrixXd> Rbs(static_cast<size_t>(m)), tops(static_cast<size_t>(m));
            std::exception_ptr ep;
#pragma omp parallel for schedule(dynamic) if (spec_.parallel)
            for (int gi = 0; gi < m; ++gi) {
              try {
                const auto& g = g_[static_cast<size_t>(gi)];
                const auto& in = info_[static_cast<size_t>(gi)];
                const int n = in.nu, nb = in.nb, rows = n + nb;
                const int rr = in.row_off;
                const Eigen::MatrixXd Phi = phi_matrix(gi, beta, b[static_cast<size_t>(gi)]);
                const Eigen::VectorXd res = in.yu - sel(f_(Phi, gi), in);
                const Eigen::MatrixXd Jw = LR[static_cast<size_t>(gi)].triangularView<Eigen::Lower>().solve(selrows(jac(Phi, gi), in));
                Eigen::MatrixXd M = Eigen::MatrixXd::Zero(rows, p_ + 1);   // [J_beta | e]
                M.block(0, 0, n, p_) = -prodA(Jw, g, in);
                M.col(p_).head(n) = LR[static_cast<size_t>(gi)].triangularView<Eigen::Lower>().solve(res);
                Eigen::MatrixXd Jb = Eigen::MatrixXd::Zero(rows, nb);
                for (const Unit& u : in.units) {
                    if (u.level == 0) {
                        Jb.block(0, u.off, n, u.q) = -prodB0(Jw, g, in);
                    } else {
                        const Eigen::MatrixXd Mb = prodBl(Jw, g, in, u.level);
                        for (int r : in.sub_rows[static_cast<size_t>(u.level - 1)][static_cast<size_t>(u.sub)]) {
                            const int pr = in.pos[static_cast<size_t>(r)];
                            if (pr >= 0) Jb.block(pr, u.off, 1, u.q) = -Mb.row(pr);
                        }
                    }
                    Jb.block(n + u.off, u.off, u.q, u.q) = LDinv[static_cast<size_t>(u.level)];
                    M.block(n + u.off, p_, u.q, 1) =
                        LD[static_cast<size_t>(u.level)].triangularView<Eigen::Lower>().solve(Eigen::VectorXd(b[static_cast<size_t>(gi)].segment(u.off, u.q)));
                }
                if (nb == 0) {
                    Red.middleRows(rr, n) = M.leftCols(p_);
                    rhs.segment(rr, n) = M.col(p_);
                    continue;
                }
                // Eliminate the group's random effects with a QR decomposition
                // (arrow structure): only the n x p reduced design is stacked.
                Eigen::HouseholderQR<Eigen::MatrixXd> qr(Jb);
                const Eigen::MatrixXd QtM = qr.householderQ().adjoint() * M;
                Rbs[static_cast<size_t>(gi)] = qr.matrixQR().topRows(nb).triangularView<Eigen::Upper>();
                tops[static_cast<size_t>(gi)] = QtM.topRows(nb);
                Red.middleRows(rr, n) = QtM.bottomRows(n).leftCols(p_);
                rhs.segment(rr, n) = QtM.bottomRows(n).col(p_);
              } catch (...) {
#pragma omp critical(mixedfit_nlme_flag)
                if (!ep) ep = std::current_exception();
              }
            }
            if (ep) std::rethrow_exception(ep);
            const Eigen::VectorXd dbeta = Red.colPivHouseholderQr().solve(-rhs);
            if (spec_.pnls_tol > 0.0) {
                // Bates & Watts relative-offset stopping rule.
                int nbt = 0;
                for (int gi = 0; gi < m; ++gi) nbt += info_[static_cast<size_t>(gi)].nb;
                const double resid2 = (Red * dbeta + rhs).squaredNorm();
                const double jd2 = std::max(S0 - resid2, 0.0);
                const double ncol = p_ + nbt, nrow = N_ + nbt;
                const double ro = std::sqrt(jd2 / ncol) / std::sqrt(resid2 / (nrow - ncol));
                if (ro < spec_.pnls_tol) break;
            }
            std::vector<Eigen::VectorXd> db(static_cast<size_t>(m));
            for (int gi = 0; gi < m; ++gi) {
                if (info_[static_cast<size_t>(gi)].nb == 0) { db[static_cast<size_t>(gi)] = Eigen::VectorXd(0); continue; }
                const Eigen::VectorXd r = tops[static_cast<size_t>(gi)].leftCols(p_) * dbeta + tops[static_cast<size_t>(gi)].col(p_);
                db[static_cast<size_t>(gi)] = -Rbs[static_cast<size_t>(gi)].triangularView<Eigen::Upper>().solve(r);
            }

            // Step reduction: accept the largest step that does not increase
            // the penalized sum of squares.
            double lam = 1.0, S1 = S0;
            Eigen::VectorXd bt = beta;
            std::vector<Eigen::VectorXd> btb = b;
            bool ok = false;
            for (int h = 0; h < 40; ++h) {
                bt = beta + lam * dbeta;
                for (int gi = 0; gi < m; ++gi) btb[static_cast<size_t>(gi)] = b[static_cast<size_t>(gi)] + lam * db[static_cast<size_t>(gi)];
                S1 = pen_ss(bt, btb, LD, LR);
                if (std::isfinite(S1) && S1 <= S0) { ok = true; break; }
                lam *= 0.5;
            }
            if (!ok) break;
            double step = lam * dbeta.cwiseAbs().maxCoeff();
            for (int gi = 0; gi < m; ++gi)
                if (db[static_cast<size_t>(gi)].size()) step = std::max(step, lam * db[static_cast<size_t>(gi)].cwiseAbs().maxCoeff());
            beta = bt;
            b = btb;
            const double dS = S0 - S1;
            S0 = S1;
            if (dS <= 1e-11 * (S0 + 1e-11) || step < 1e-10) break;
        }
    }

    std::vector<NlmeGroup> g_;                  ///< Grouped data.
    NlmeFn f_;                                  ///< Model function.
    NlmeJacFn jac_;                             ///< Optional analytic Jacobian.
    NlmeSpec spec_;                             ///< Model specification.
    std::vector<Info> info_;                    ///< Per-group bookkeeping.
    std::vector<int> q_;                        ///< Random-effect dimensions per level.
    std::vector<PdSpec> kinds_;                 ///< Resolved covariance parameterizations.
    int p_ = 0, k_ = 0, N_ = 0, N_all_ = 0, nlev_ = 1;  ///< Fixed effects, parameters, used rows, all rows, levels.
};

}  // namespace mixedfit
