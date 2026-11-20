// SPDX-License-Identifier: Apache-2.0
//
// Linear mixed-effects model with nested grouping levels:
//     y_i = X_i beta + Z_i b_i + sum_{l>=1} sum_s Z_{l,s} b_{l,s} + e_i,
//     b_i ~ N(0, sigma^2 D_0),  b_{l,s} ~ N(0, sigma^2 D_l)  (independent),
//     e_i ~ N(0, sigma^2 S_i R_i S_i)
// where i indexes the outer group (independent of the others), l>=1 are inner
// levels (e.g. Side within Dog), and S_i comes from a variance function.
//
// Profiled likelihood (Lindstrom & Bates 1988; Pinheiro & Bates 2000, ch. 2):
// with W_i = S R S + sum_l Z_l D_l Z_l^T = C_i C_i^T, whiten X~ = C^-1 X and
// y~ = C^-1 y, and concentrate beta and sigma^2 out:
//     ML  : -2 l = N [log 2pi + log(RSS/N) + 1] + sum log|W_i|
//     REML: -2 l = (N-p)[log 2pi + log(RSS/(N-p)) + 1] + sum log|W_i| + log|X~^T X~|
//           (- log|X^T X| if reml_constant)
// Groups are processed in parallel when compiled with -fopenmp.
//
// theta = [re level 0, re level 1, ..., cor, var].   Requires Eigen >= 3.4.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "bfgs_min.hpp"
#include "optim_backend.hpp"
#include "cor_struct.hpp"
#include "pd_param.hpp"
#include "qr_rank.hpp"
#include "var_func.hpp"

namespace mixedfit {

/// One inner (nested) random-effects level of an `LmeGroup`.
struct LmeLevel {
    Eigen::MatrixXd Z;       ///< n x q_l design (rows of the outer group).
    std::vector<int> sub;    ///< sub[r] = subgroup id of row r within the group.
};

/// Data for one group of a linear mixed-effects model.
struct LmeGroup {
    int label = 0;                        ///< Group label.
    Eigen::VectorXd y;                    ///< Response values.
    Eigen::MatrixXd X, Z;                 ///< Fixed effects and outer (level 0) random effects.
    std::vector<LmeLevel> inner;          ///< Nested levels 1..
    Eigen::VectorXd time;                 ///< Positions for the correlation structure (empty -> 0..n-1).
    Eigen::MatrixXd coords;               ///< Coordinates (n x d) for spatial correlation (empty -> time is used).
    std::vector<int> cor_sub;             ///< Correlation at an inner level: block id of each row (empty -> whole group is one block).
    std::vector<int> strata;              ///< Variance strata (empty -> 0).
    Eigen::VectorXd vcov;                 ///< Variance-function covariate (slot 0; shortcut).
    VarData vd;                           ///< Strata and covariates per slot (`varComb` with several covariates).
    std::vector<int> rows;                ///< Original row indices (optional).
};

/// A theta parameter pinned to a value (unconstrained scale). Identified by
/// `label` (see `LmeModel::theta_labels()`) or by `index` (>= 0).
struct FixedTheta {
    std::string label;                    ///< Parameter label.
    double value = 0.0;                   ///< Value on the unconstrained scale.
    int index = -1;                       ///< Parameter index (alternative to `label`).
};

/// Usual unconstrained values derived from the constrained value.
inline double theta_from_ar1(double phi) { return std::atanh(phi); }          ///< corAR1: phi = tanh(theta)
inline double theta_from_logit(double x) { return std::log(x / (1.0 - x)); }  ///< corCAR1 (phi), nugget
inline double theta_from_log(double x) { return std::log(x); }                ///< range, constants, varIdent ratios

/// Specification of an `lme` model.
struct LmeSpec {
    PdKind re_kind = PdKind::LogCholesky;          ///< Level-0 covariance parameterization.
    std::vector<PdKind> inner_kinds;               ///< Levels 1.. (default: LogCholesky).
    std::vector<PdSpec> pd_specs;                  ///< If non-empty: one PdSpec per level (allows pdBlocked); overrides re_kind/inner_kinds.
    CorSpec cor;                                   ///< Within-group correlation structure.
    VarSpec var;                                   ///< Within-group variance function.
    bool reml = true;                              ///< Use REML instead of ML.
    bool reml_constant = false;                    ///< Add 1/2 log|X^T X| to the REML log-likelihood.
    bool analytic_grad = true;                     ///< Analytic profile gradient (otherwise finite differences).
    /// Heteroscedasticity bound: max(s)/min(s) ratio of the variance scales
    /// over the whole data set. Beyond it the region is treated as infeasible
    /// (this avoids degenerate suprema of the likelihood).
    double var_max_ratio = 1e4;
    /// varConstPower: |delta| <= var_shape_bound. As delta -> -inf the family
    /// c + |v|^delta tends to a step function (different variance only for the
    /// smallest v) and the likelihood grows without an interior optimum.
    double var_shape_bound = 10.0;
    bool diagnose = true;                          ///< Identifiability diagnostic (Hessian of theta).
    int em_iter = 25;                              ///< EM iterations (random-effect covariance only) for the starting theta (as lmeControl(niterEM = 25)).
    std::vector<FixedTheta> fixed;                 ///< Fixed unconstrained parameters (by label, e.g. "cor[0]", "var[1]", "re0[2]").
    double sigma_fixed = 0.0;                      ///< > 0: hold sigma fixed (as lmeControl(sigma=)); needed e.g. with varConstProp.
    Optimizer optimizer = Optimizer::Internal;     ///< Optimization back end (see optim_backend.hpp).
    double theta_bound = 15.0;                     ///< Bounds |theta| <= theta_bound (L-BFGS-B only).
    BfgsOptions opt = [] {
        BfgsOptions o;
        o.gtol = 1e-6;
        o.fd_step = 1e-5;
        o.ftol = 1e-13;
        return o;
    }();
};

/// Details returned by the profiled objective `neg2ll`.
struct LmeDetails {
    Eigen::VectorXd beta;               ///< GLS estimate at theta.
    Eigen::MatrixXd cov_unscaled;       ///< (X~^T X~)^-1, NaN for aliased columns.
    double rss = 0.0, logdetW = 0.0, var_ratio = 1.0;  ///< Residual sum of squares, log|W|, and variance-scale ratio.
    int rank = 0;                       ///< Rank of the whitened design.
};

/// Identifiability diagnostics based on the profile Hessian.
struct LmeIdent {
    Eigen::VectorXd eigenvalues;        ///< Ascending eigenvalues of the Hessian of -2 l (profiled) with respect to theta.
    Eigen::MatrixXd eigenvectors;       ///< Corresponding eigenvectors.
    int n_weak = 0;                     ///< Directions with |lambda| <= 1e-6 lambda_max (nearly unidentified).
    bool saddle = false;                ///< A negative eigenvalue: not a local minimum.
    bool evaluated = false;             ///< Whether the diagnostic was computed.
    std::string message;                ///< Human-readable summary.
};

/// A variance-component estimate with its standard error.
struct VarcompEntry { std::string name; double est = 0.0, se = std::numeric_limits<double>::quiet_NaN(); };

/// Results of an `lme` fit.
struct LmeResult {
    Eigen::VectorXd beta, se;                       ///< Fixed effects and standard errors.
    double sigma2 = std::numeric_limits<double>::quiet_NaN();  ///< Residual variance.
    Eigen::VectorXd theta, theta_cor, theta_var;    ///< Full, correlation, and variance parameter vectors.
    std::vector<Eigen::VectorXd> theta_re;          ///< Random-effect parameters per level.
    std::vector<Eigen::MatrixXd> D_levels, Sigma_levels;  ///< D_l and sigma^2 * D_l.
    Eigen::MatrixXd D, Sigma_b;                     ///< Level-0 covariance matrices (shortcut).
    Eigen::VectorXd R_cor_theta() const { return theta_cor; }  ///< Correlation parameters.
    double loglik = std::numeric_limits<double>::quiet_NaN();  ///< Log-likelihood at the solution.
    bool reml = true;                               ///< Whether REML was used.
    int rank = 0, N = 0;                            ///< Design rank and number of observations.
    LmeIdent ident;                                 ///< Identifiability diagnostic.
    Eigen::MatrixXd vcov_beta;                      ///< Covariance of beta (NaN for aliased columns); convention sigma^2 = RSS/(N-p).
    /// apVar: asymptotic covariance of (theta, log sigma) = 2 * (Hessian of -2 l)^-1
    /// (last entry: log sigma).
    Eigen::MatrixXd vcov_theta;
    /// The same covariance on nlme's natural scale: random effects in
    /// (log absolute sd = log relative sd + log sigma, x_ij with rho = tanh(x/2)
    /// for upper-triangle entries), AR1 -> 2*atanh(phi), CAR1 -> logit(phi),
    /// otherwise as theta; last entry: log sigma.
    std::vector<std::string> natural_names;
    Eigen::VectorXd theta_natural;
    Eigen::MatrixXd vcov_natural;
    std::vector<VarcompEntry> varcomp;          ///< sd, correlations, cor/var parameters, and sigma, with standard errors (delta method).
    int var_iterations = 0;                     ///< Outer iterations when the variance uses fitted(.).
    std::string warning;                        ///< Non-empty when some unconstrained parameter ended at |theta| > 30.
    std::vector<int> labels;                    ///< Group labels.
    std::vector<Eigen::VectorXd> blup;                                  ///< [group] level 0.
    std::vector<std::vector<std::vector<Eigen::VectorXd>>> blup_inner;  ///< [group][level-1][subgroup].
    Eigen::VectorXd fitted_pop, fitted_cond, resid;                     ///< Fitted values (population and conditional) and residuals, groups concatenated.
    std::vector<int> order;                                             ///< Original row index of each stacked value.
    BfgsResult opt;                                                     ///< Optimizer diagnostics.
};


/// Starting value for the spatial range: median of the positive
/// within-group distances.
inline void set_spatial_start(CorSpec& cor, const std::vector<const Eigen::MatrixXd*>& coords,
                              const std::vector<const Eigen::VectorXd*>& times) {
    if (!cor.spatial()) return;
    std::vector<double> d;
    for (size_t g = 0; g < coords.size() && d.size() < 200000; ++g) {
        const int n = static_cast<int>(times[g]->size());
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) {
                const double v = spatial_dist(cor.metric, *coords[g], *times[g], i, j);
                if (v > 0.0) d.push_back(v);
            }
    }
    if (d.empty()) return;
    std::nth_element(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(d.size() / 2), d.end());
    cor.range_start = d[d.size() / 2];
}


/// Set `n_strata` of each variance-function component from the strata present
/// in its slot.
inline void configure_var(VarSpec& spec, const std::vector<const VarData*>& vds) {
    std::vector<VarSpec*> comps;
    var_components_mut(spec, comps);
    for (VarSpec* c : comps) {
        int ms = 0;
        bool any = false;
        for (const VarData* vd : vds)
            if (c->strata_slot < static_cast<int>(vd->strata.size()))
                for (int s : vd->strata[static_cast<size_t>(c->strata_slot)]) { ms = std::max(ms, s); any = true; }
        c->n_strata = (any || c->kind == VarKind::Ident) ? ms + 1 : 1;
        if (c->kind == VarKind::Power && !c->fitted)
            for (const VarData* vd : vds)
                if (c->cov_slot < static_cast<int>(vd->cov.size()) && (vd->cov[static_cast<size_t>(c->cov_slot)].array() == 0.0).any())
                    throw std::invalid_argument("varPower with a covariate equal to 0");
    }
}

/// Remove the `fitted` flag recursively from a variance specification.
inline VarSpec strip_fitted(VarSpec s) {
    s.fitted = false;
    for (auto& c : s.comb) c = strip_fitted(c);
    return s;
}


/// Build human-readable labels for the theta parameters.
inline std::vector<std::string> make_theta_labels(const std::vector<PdSpec>& pds, const std::vector<int>& qs,
                                                  const CorSpec& cor, const VarSpec& var) {
    std::vector<std::string> nm;
    for (size_t l = 0; l < pds.size(); ++l) {
        const int n = qs[l] > 0 ? pd_nparams(pds[l], qs[l]) : 0;
        for (int j = 0; j < n; ++j) nm.push_back("re" + std::to_string(l) + "[" + std::to_string(j) + "]");
    }
    for (int j = 0; j < cor.n_params(); ++j) nm.push_back("cor[" + std::to_string(j) + "]");
    for (int j = 0; j < var.n_params(); ++j) nm.push_back("var[" + std::to_string(j) + "]");
    return nm;
}

/// Apply the fixed parameters to theta; return the fixed indices (sorted).
///
/// @throws std::invalid_argument If a label is unknown or an index is out of range.
inline std::vector<int> apply_fixed_theta(Eigen::VectorXd& th, const std::vector<FixedTheta>& fx,
                                          const std::vector<std::string>& labels) {
    std::vector<int> idx;
    for (const auto& f : fx) {
        int k = f.index;
        if (k < 0) {
            for (size_t i = 0; i < labels.size(); ++i) if (labels[i] == f.label) { k = static_cast<int>(i); break; }
            if (k < 0) throw std::invalid_argument("unknown fixed parameter: " + f.label);
        }
        if (k >= th.size()) throw std::invalid_argument("fixed parameter index out of range");
        th[k] = f.value;
        idx.push_back(k);
    }
    std::sort(idx.begin(), idx.end());
    return idx;
}

namespace detail {
/// Inverse of `pd_factor` for the covariance D: return theta such that
/// `pd_sigma(spec, q, theta) ~ D` (empty if D is not positive definite).
inline Eigen::VectorXd pd_theta_from_D(const PdSpec& s, int q, const Eigen::MatrixXd& D) {
    if (s.custom) return s.custom->from_D ? s.custom->from_D(q, D) : Eigen::VectorXd();
    if (s.blocked()) {
        Eigen::VectorXd t(pd_nparams(s, q));
        int off = 0, pos = 0;
        for (size_t b = 0; b < s.block_sizes.size(); ++b) {
            const int sz = s.block_sizes[b];
            const Eigen::VectorXd tb = pd_theta_from_D(PdSpec(s.block_kinds[b]), sz, D.block(off, off, sz, sz));
            if (tb.size() == 0) return Eigen::VectorXd();
            t.segment(pos, tb.size()) = tb;
            off += sz; pos += static_cast<int>(tb.size());
        }
        return t;
    }
    Eigen::VectorXd t(pd_nparams(s.kind, q));
    switch (s.kind) {
        case PdKind::Diag:
            for (int i = 0; i < q; ++i) { if (!(D(i, i) > 0)) return Eigen::VectorXd(); t[i] = 0.5 * std::log(D(i, i)); }
            return t;
        case PdKind::Ident: {
            const double m = D.diagonal().mean();
            if (!(m > 0)) return Eigen::VectorXd();
            t[0] = 0.5 * std::log(m);
            return t;
        }
        case PdKind::LogCholesky:
        case PdKind::Cholesky: {
            Eigen::LLT<Eigen::MatrixXd> llt(D);
            if (llt.info() != Eigen::Success) return Eigen::VectorXd();
            const Eigen::MatrixXd L = llt.matrixL();
            int k = 0;
            for (int i = 0; i < q; ++i)
                for (int j = 0; j <= i; ++j) t[k++] = (i == j && s.kind == PdKind::LogCholesky) ? std::log(L(i, i)) : L(i, j);
            return t;
        }
        case PdKind::MatrixLog: {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(D);
            if (es.info() != Eigen::Success || !(es.eigenvalues().minCoeff() > 0)) return Eigen::VectorXd();
            const Eigen::MatrixXd A = es.eigenvectors() * es.eigenvalues().array().log().matrix().asDiagonal() * es.eigenvectors().transpose();
            int k = 0;
            for (int i = 0; i < q; ++i) for (int j = 0; j <= i; ++j) t[k++] = A(i, j);
            return t;
        }
        case PdKind::Natural: {
            Eigen::VectorXd sd(q);
            for (int i = 0; i < q; ++i) { if (!(D(i, i) > 0)) return Eigen::VectorXd(); sd[i] = std::sqrt(D(i, i)); t[i] = std::log(sd[i]); }
            const Eigen::MatrixXd C = sd.cwiseInverse().asDiagonal() * D * sd.cwiseInverse().asDiagonal();
            Eigen::LLT<Eigen::MatrixXd> llt(C);
            if (llt.info() != Eigen::Success) return Eigen::VectorXd();
            const Eigen::MatrixXd Lc = llt.matrixL();
            int k = q;
            for (int i = 0; i < q; ++i) {
                double rem = 1.0;
                for (int j = 0; j < i; ++j) {
                    double v = Lc(i, j) / std::sqrt(std::max(rem, 1e-300));
                    v = std::max(-1.0 + 1e-9, std::min(1.0 - 1e-9, v));
                    t[k++] = std::atanh(v);
                    rem -= Lc(i, j) * Lc(i, j);
                }
            }
            return t;
        }
        case PdKind::CompoundSymmetry: {
            const double m = D.diagonal().mean();
            if (!(m > 0)) return Eigen::VectorXd();
            t[0] = 0.5 * std::log(m);
            if (q > 1) {
                double mo = 0.0;
                for (int i = 0; i < q; ++i) for (int j = 0; j < q; ++j) if (i != j) mo += D(i, j);
                mo /= (q * (q - 1.0));
                const double lo = -1.0 / (q - 1.0);
                const double rho = std::max(lo + 1e-6, std::min(1.0 - 1e-6, mo / m));
                t[1] = std::log((rho - lo) / (1.0 - rho));
            }
            return t;
        }
    }
    return Eigen::VectorXd();
}
}  // namespace detail

/// Resolve one covariance parameterization per level.
///
/// @param given       Explicit per-level specifications (overrides the rest).
/// @param re_kind     Level-0 parameterization.
/// @param inner_kinds Parameterizations for inner levels 1..
/// @param nlev        Number of levels.
/// @throws std::invalid_argument If `given` is non-empty and does not have one
///         entry per level.
inline std::vector<PdSpec> resolve_pd_specs(const std::vector<PdSpec>& given, PdKind re_kind,
                                            const std::vector<PdKind>& inner_kinds, int nlev) {
    if (!given.empty()) {
        if (static_cast<int>(given.size()) != nlev) throw std::invalid_argument("pd_specs: one PdSpec per level is required");
        return given;
    }
    std::vector<PdSpec> out(static_cast<size_t>(nlev), PdSpec(PdKind::LogCholesky));
    out[0] = PdSpec(re_kind);
    for (int l = 1; l < nlev; ++l)
        if (l - 1 < static_cast<int>(inner_kinds.size())) out[static_cast<size_t>(l)] = PdSpec(inner_kinds[static_cast<size_t>(l - 1)]);
    return out;
}

/// Initial theta with D_l = I, zero correlation, and homogeneous variance.
inline Eigen::VectorXd pd_start(PdKind k, int q);
inline Eigen::VectorXd pd_start(const PdSpec& s, int q) {
    if (s.custom) {
        Eigen::VectorXd t = s.custom->start ? s.custom->start(q) : Eigen::VectorXd::Zero(pd_nparams(s, q));
        if (t.size() != pd_nparams(s, q)) throw std::invalid_argument("pd custom: start has the wrong length");
        return t;
    }
    if (!s.blocked()) return pd_start(s.kind, q);
    Eigen::VectorXd t(pd_nparams(s, q));
    int pos = 0;
    for (size_t b = 0; b < s.block_sizes.size(); ++b) {
        const Eigen::VectorXd tb = pd_start(s.block_kinds[b], s.block_sizes[b]);
        t.segment(pos, tb.size()) = tb;
        pos += static_cast<int>(tb.size());
    }
    return t;
}
inline Eigen::VectorXd pd_start(PdKind k, int q) {
    Eigen::VectorXd t = Eigen::VectorXd::Zero(pd_nparams(k, q));
    if (k == PdKind::Cholesky) {
        int idx = 0;
        for (int i = 0; i < q; ++i)
            for (int j = 0; j <= i; ++j, ++idx)
                if (i == j) t[idx] = 1.0;
    } else if (k == PdKind::CompoundSymmetry && q > 1) {
        t[1] = -std::log(q - 1.0);
    }
    return t;
}

/// Assemble the full starting theta vector from the per-level, correlation,
/// and variance parameterizations.
///
/// Spatial correlations start from the median within-group distance; custom
/// structures use their supplied starts; `varConstPower` starts with a small
/// constant (`c = exp(-4)`) to avoid its degenerate likelihood supremum.
inline Eigen::VectorXd theta_start(const std::vector<PdSpec>& kinds, const std::vector<int>& qs,
                                   const CorSpec& cor, const VarSpec& var) {
    std::vector<Eigen::VectorXd> parts;
    int tot = 0;
    for (size_t l = 0; l < kinds.size(); ++l) {
        parts.push_back(qs[l] > 0 ? pd_start(kinds[l], qs[l]) : Eigen::VectorXd(0));
        tot += static_cast<int>(parts.back().size());
    }
    tot += cor.n_params() + var.n_params();
    Eigen::VectorXd t = Eigen::VectorXd::Zero(tot);
    int off = 0;
    for (auto& p : parts) { t.segment(off, p.size()) = p; off += static_cast<int>(p.size()); }
    if (cor.spatial()) {
        const int off_c = tot - cor.n_params() - var.n_params();
        t[off_c] = std::log(cor.range_start);
        if (cor.nugget) t[off_c + 1] = -2.2;               // initial nugget ~ 0.1
    }
    if (cor.kind == CorKind::Custom && cor.custom && cor.custom->start.size() == cor.n_params()) {
        const int off_c = tot - cor.n_params() - var.n_params();
        t.segment(off_c, cor.n_params()) = cor.custom->start;
    }
    {
        std::vector<std::pair<const VarSpec*, int>> comps;
        var_components(var, 0, comps);
        for (const auto& c : comps)
            if (c.first->kind == VarKind::Custom && c.first->custom && c.first->custom->start.size() == c.first->n_params())
                t.segment(tot - var.n_params() + c.second, c.first->n_params()) = c.first->custom->start;
    }
    // varConstPower: the likelihood has a degenerate supremum (scale -> 0)
    // when the constant starts large; start with a small constant (c = exp(-4)).
    {
        std::vector<std::pair<const VarSpec*, int>> comps;
        var_components(var, 0, comps);
        for (const auto& c : comps)
            if (c.first->kind == VarKind::ConstPower)
                for (int k = 0; k < c.first->n_per(); ++k) t[tot - var.n_params() + c.second + k] = -4.0;
    }
    return t;
}

/// Group stacked data (order of first appearance of each label).
///
/// @throws std::invalid_argument If the input dimensions are inconsistent.
inline std::vector<LmeGroup> make_groups(const Eigen::VectorXd& y, const Eigen::MatrixXd& X,
                                         const Eigen::MatrixXd& Z, const std::vector<int>& labels,
                                         const Eigen::VectorXd& time = Eigen::VectorXd(),
                                         const std::vector<int>& strata = {},
                                         const Eigen::VectorXd& vcov = Eigen::VectorXd(),
                                         const Eigen::MatrixXd& coords = Eigen::MatrixXd()) {
    const int N = static_cast<int>(y.size());
    if (X.rows() != N || Z.rows() != N || static_cast<int>(labels.size()) != N ||
        (time.size() != 0 && time.size() != N) || (!strata.empty() && static_cast<int>(strata.size()) != N) ||
        (vcov.size() != 0 && vcov.size() != N))
        throw std::invalid_argument("make_groups: inconsistent dimensions");
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
    std::vector<LmeGroup> out(idx.size());
    for (size_t g = 0; g < idx.size(); ++g) {
        const auto& r = idx[g];
        out[g].label = lab[g];
        out[g].rows = r;
        out[g].y = y(r);
        out[g].X = X(r, Eigen::placeholders::all);
        out[g].Z = Z(r, Eigen::placeholders::all);
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

/// Add an inner (nested) level to groups created by `make_groups`.
///
/// @throws std::invalid_argument If the groups do not store their original row indices.
inline void add_inner_level(std::vector<LmeGroup>& groups, const Eigen::MatrixXd& Z,
                            const std::vector<int>& sub) {
    for (auto& g : groups) {
        if (g.rows.empty()) throw std::invalid_argument("add_inner_level: groups do not store row indices");
        LmeLevel lv;
        lv.Z = Z(g.rows, Eigen::placeholders::all);
        lv.sub.resize(g.rows.size());
        for (size_t k = 0; k < g.rows.size(); ++k) lv.sub[k] = sub[static_cast<size_t>(g.rows[k])];
        g.inner.push_back(std::move(lv));
    }
}


namespace detail {
/// Ordinal position (0, 1, ...) of each row within its block: nlme's default
/// `time` with `form = ~1 | g1/g2`.
inline void block_ordinal_time(const std::vector<int>& sub, Eigen::VectorXd& time) {
    time.resize(static_cast<Eigen::Index>(sub.size()));
    std::map<int, int> cnt;
    for (size_t i = 0; i < sub.size(); ++i) time[static_cast<Eigen::Index>(i)] = cnt[sub[i]]++;
}
}  // namespace detail

/// na.omit-style utility: like `make_groups` but dropping rows with any
/// non-finite value (NaN/Inf) in y, X, Z, time, vcov, or coords.
///
/// `rows` keeps the ORIGINAL indices (into the full vectors), so
/// `add_inner_level`, `add_var_strata`, `add_var_covariate`, `add_cor_sub`, and
/// `LmeResult::order` still index the original data. `dropped` (optional)
/// receives the discarded indices. Groups left without rows disappear.
///
/// @throws std::invalid_argument If the input dimensions are inconsistent or
///         no complete row remains.
inline std::vector<LmeGroup> make_groups_na(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Z,
                                            const std::vector<int>& labels, const Eigen::VectorXd& time = Eigen::VectorXd(),
                                            const std::vector<int>& strata = {}, const Eigen::VectorXd& vcov = Eigen::VectorXd(),
                                            const Eigen::MatrixXd& coords = Eigen::MatrixXd(), std::vector<int>* dropped = nullptr) {
    const int N = static_cast<int>(y.size());
    if (X.rows() != N || Z.rows() != N || static_cast<int>(labels.size()) != N ||
        (time.size() != 0 && time.size() != N) || (!strata.empty() && static_cast<int>(strata.size()) != N) ||
        (vcov.size() != 0 && vcov.size() != N) || (coords.cols() > 0 && coords.rows() != N))
        throw std::invalid_argument("make_groups_na: inconsistent dimensions");
    std::vector<int> keep, drop;
    for (int i = 0; i < N; ++i) {
        bool ok = std::isfinite(y[i]) && X.row(i).allFinite() && Z.row(i).allFinite();
        if (ok && time.size() != 0) ok = std::isfinite(time[i]);
        if (ok && vcov.size() != 0) ok = std::isfinite(vcov[i]);
        if (ok && coords.cols() > 0) ok = coords.row(i).allFinite();
        (ok ? keep : drop).push_back(i);
    }
    if (dropped) *dropped = drop;
    if (keep.empty()) throw std::invalid_argument("make_groups_na: no complete rows remain");
    std::vector<int> lab(keep.size()), st;
    for (size_t k = 0; k < keep.size(); ++k) lab[k] = labels[static_cast<size_t>(keep[k])];
    if (!strata.empty()) for (int i : keep) st.push_back(strata[static_cast<size_t>(i)]);
    Eigen::VectorXd tm = time.size() ? Eigen::VectorXd(time(keep)) : Eigen::VectorXd();
    Eigen::VectorXd vc = vcov.size() ? Eigen::VectorXd(vcov(keep)) : Eigen::VectorXd();
    Eigen::MatrixXd cd = coords.cols() > 0 ? Eigen::MatrixXd(coords(keep, Eigen::placeholders::all)) : Eigen::MatrixXd();
    std::vector<LmeGroup> gs = make_groups(Eigen::VectorXd(y(keep)), Eigen::MatrixXd(X(keep, Eigen::placeholders::all)),
                                           Eigen::MatrixXd(Z(keep, Eigen::placeholders::all)), lab, tm, st, vc, cd);
    for (auto& g : gs)
        for (auto& r : g.rows) r = keep[static_cast<size_t>(r)];
    return gs;
}

/// Correlation at an inner level: assign each row the block
/// `sub[original_row]` (as `form = ~ t | g1/g2` in nlme). Rows in different
/// blocks are uncorrelated; within a block the correlation structure applies.
///
/// @throws std::invalid_argument If the groups do not store their original row indices.
inline void add_cor_sub(std::vector<LmeGroup>& groups, const std::vector<int>& sub) {
    for (auto& g : groups) {
        if (g.rows.empty()) throw std::invalid_argument("add_cor_sub: groups do not store row indices");
        g.cor_sub.resize(g.rows.size());
        for (size_t k = 0; k < g.rows.size(); ++k) g.cor_sub[k] = sub[static_cast<size_t>(g.rows[k])];
        if (g.time.size() == 0) detail::block_ordinal_time(g.cor_sub, g.time);
    }
}

namespace detail {

/// Append the natural-scale representation of one covariance level.
inline void pd_natural_append(const PdSpec& s, int q, const Eigen::VectorXd& th, double logsig, const std::string& pre,
                              std::vector<double>& v, std::vector<std::string>& nm) {
    if (s.custom) {
        for (int j = 0; j < th.size(); ++j) { v.push_back(th[j]); nm.push_back(pre + s.custom->name + "[" + std::to_string(j) + "]"); }
        return;
    }
    if (s.blocked()) {
        int pos = 0;
        for (size_t b = 0; b < s.block_sizes.size(); ++b) {
            const int sz = s.block_sizes[b], np = pd_nparams(s.block_kinds[b], sz);
            pd_natural_append(PdSpec(s.block_kinds[b]), sz, th.segment(pos, np), logsig, pre + "b" + std::to_string(b) + ".", v, nm);
            pos += np;
        }
        return;
    }
    if (s.kind == PdKind::Diag) {
        for (int i = 0; i < q; ++i) { v.push_back(th[i] + logsig); nm.push_back(pre + "logsd[" + std::to_string(i) + "]"); }
        return;
    }
    if (s.kind == PdKind::Ident) { v.push_back(th[0] + logsig); nm.push_back(pre + "logsd"); return; }
    const Eigen::MatrixXd D = pd_sigma(s.kind, q, th);
    for (int i = 0; i < q; ++i) { v.push_back(0.5 * std::log(D(i, i)) + logsig); nm.push_back(pre + "logsd[" + std::to_string(i) + "]"); }
    for (int i = 0; i < q; ++i)
        for (int j = i + 1; j < q; ++j) {
            const double rho = D(i, j) / std::sqrt(D(i, i) * D(j, j));
            if (s.kind == PdKind::CompoundSymmetry) {
                if (i == 0 && j == 1) { v.push_back(rho); nm.push_back(pre + "rho"); }
            } else {
                v.push_back(std::log((1.0 + rho) / (1.0 - rho)));
                nm.push_back(pre + "x[" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
}

/// Append absolute standard deviations and (constrained) correlations of one level.
inline void pd_constrained_append(const PdSpec& s, int q, const Eigen::VectorXd& th, double sigma, const std::string& pre,
                                  std::vector<double>& v, std::vector<std::string>& nm) {
    const Eigen::MatrixXd D = pd_sigma(s, q, th);
    for (int i = 0; i < q; ++i) { v.push_back(sigma * std::sqrt(D(i, i))); nm.push_back(pre + "sd[" + std::to_string(i) + "]"); }
    std::vector<std::pair<int, int>> ranges;
    if (s.blocked()) {
        int off = 0;
        for (size_t b = 0; b < s.block_sizes.size(); ++b) {
            if (s.block_kinds[b] != PdKind::Diag && s.block_kinds[b] != PdKind::Ident) ranges.push_back({off, s.block_sizes[b]});
            off += s.block_sizes[b];
        }
    } else if (s.kind != PdKind::Diag && s.kind != PdKind::Ident) {
        ranges.push_back({0, q});
    }
    for (const auto& rg : ranges)
        for (int i = rg.first; i < rg.first + rg.second; ++i)
            for (int j = i + 1; j < rg.first + rg.second; ++j) {
                v.push_back(D(i, j) / std::sqrt(D(i, i) * D(j, j)));
                nm.push_back(pre + "cor[" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
}

}  // namespace detail

/// Add a strata factor or a variance covariate to each group in the given slot
/// (uses `LmeGroup::rows`).
inline void add_var_strata(std::vector<LmeGroup>& groups, const std::vector<int>& labels, int slot) {
    for (auto& g : groups) {
        if (g.vd.strata.size() <= static_cast<size_t>(slot)) g.vd.strata.resize(static_cast<size_t>(slot + 1));
        g.vd.strata[static_cast<size_t>(slot)].resize(g.rows.size());
        for (size_t k = 0; k < g.rows.size(); ++k) g.vd.strata[static_cast<size_t>(slot)][k] = labels[static_cast<size_t>(g.rows[k])];
    }
}

/// Add a variance covariate to each group in the given slot (uses `LmeGroup::rows`).
inline void add_var_covariate(std::vector<LmeGroup>& groups, const Eigen::VectorXd& v, int slot) {
    for (auto& g : groups) {
        if (g.vd.cov.size() <= static_cast<size_t>(slot)) g.vd.cov.resize(static_cast<size_t>(slot + 1));
        g.vd.cov[static_cast<size_t>(slot)] = v(g.rows);
    }
}

/// Linear mixed-effects model.
class LmeModel {
public:
    /// Construct the model and precompute the per-group bookkeeping.
    ///
    /// @param groups Grouped data.
    /// @param spec   Model specification.
    /// @throws std::invalid_argument On missing groups or inconsistent
    ///         dimensions of X, Z, time, cor_sub, or inner levels.
    LmeModel(std::vector<LmeGroup> groups, LmeSpec spec) : g_(std::move(groups)), spec_(std::move(spec)) {
        if (g_.empty()) throw std::invalid_argument("LmeModel: no groups");
        p_ = static_cast<int>(g_[0].X.cols());
        nlev_ = 1 + static_cast<int>(g_[0].inner.size());
        q_.assign(static_cast<size_t>(nlev_), 0);
        q_[0] = static_cast<int>(g_[0].Z.cols());
        for (int l = 1; l < nlev_; ++l) q_[static_cast<size_t>(l)] = static_cast<int>(g_[0].inner[static_cast<size_t>(l - 1)].Z.cols());
        kinds_ = resolve_pd_specs(spec_.pd_specs, spec_.re_kind, spec_.inner_kinds, nlev_);

        int mx = 2, maxpos = 0;
        N_ = 0;
        cache_.resize(g_.size());
        for (size_t gi = 0; gi < g_.size(); ++gi) {
            auto& g = g_[gi];
            const int n = static_cast<int>(g.y.size());
            if (g.X.rows() != n || g.Z.rows() != n || g.X.cols() != p_ || g.Z.cols() != q_[0] ||
                static_cast<int>(g.inner.size()) != nlev_ - 1)
                throw std::invalid_argument("LmeModel: inconsistent group dimensions");
            if (g.time.size() == 0) g.time = Eigen::VectorXd::LinSpaced(n, 0, n - 1);
            if (g.time.size() != n) throw std::invalid_argument("LmeModel: time has the wrong length");
            for (int i = 0; i < n; ++i) maxpos = std::max(maxpos, static_cast<int>(std::lround(g.time[i])));
            if (g.vd.strata.empty() && !g.strata.empty()) g.vd.strata.push_back(g.strata);
            if (g.vd.cov.empty() && g.vcov.size() > 0) g.vd.cov.push_back(g.vcov);
            cache_[gi].row_off = N_;
            N_ += n;
            if (!g.cor_sub.empty() && static_cast<int>(g.cor_sub.size()) != n)
                throw std::invalid_argument("LmeModel: cor_sub has the wrong length");
            cache_[gi].cor_blocks = make_cor_blocks(g.cor_sub);
            {
                int bmax = n;
                if (!cache_[gi].cor_blocks.empty()) {
                    bmax = 0;
                    for (const auto& b : cache_[gi].cor_blocks) bmax = std::max(bmax, static_cast<int>(b.size()));
                }
                mx = std::max(mx, bmax);
            }
            // Subgroups of each inner level.
            cache_[gi].sub_rows.resize(static_cast<size_t>(nlev_ - 1));
            cache_[gi].sub_Z.resize(static_cast<size_t>(nlev_ - 1));
            for (int l = 1; l < nlev_; ++l) {
                const auto& lv = g.inner[static_cast<size_t>(l - 1)];
                if (lv.Z.rows() != n || lv.Z.cols() != q_[static_cast<size_t>(l)] || static_cast<int>(lv.sub.size()) != n)
                    throw std::invalid_argument("LmeModel: inconsistent inner level");
                std::map<int, int> pos;
                auto& sr = cache_[gi].sub_rows[static_cast<size_t>(l - 1)];
                for (int i = 0; i < n; ++i) {
                    auto it = pos.find(lv.sub[static_cast<size_t>(i)]);
                    if (it == pos.end()) { it = pos.emplace(lv.sub[static_cast<size_t>(i)], static_cast<int>(sr.size())).first; sr.emplace_back(); }
                    sr[static_cast<size_t>(it->second)].push_back(i);
                }
                for (const auto& idx : sr) cache_[gi].sub_Z[static_cast<size_t>(l - 1)].push_back(lv.Z(idx, Eigen::placeholders::all));
            }
        }
        spec_.cor.max_group_size = mx;
        if (spec_.cor.spatial()) {
            std::vector<const Eigen::MatrixXd*> cs;
            std::vector<const Eigen::VectorXd*> ts;
            for (const auto& g : g_) { cs.push_back(&g.coords); ts.push_back(&g.time); }
            set_spatial_start(spec_.cor, cs, ts);
        }
        if (spec_.cor.kind == CorKind::Symm) spec_.cor.n_pos = maxpos + 1;
        {
            std::vector<const VarData*> vds;
            for (const auto& g : g_) vds.push_back(&g.vd);
            configure_var(spec_.var, vds);
        }

        re_off_.assign(static_cast<size_t>(nlev_), 0);
        n_re_l_.assign(static_cast<size_t>(nlev_), 0);
        int off = 0;
        for (int l = 0; l < nlev_; ++l) {
            n_re_l_[static_cast<size_t>(l)] = q_[static_cast<size_t>(l)] > 0 ? pd_nparams(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)]) : 0;
            re_off_[static_cast<size_t>(l)] = off;
            off += n_re_l_[static_cast<size_t>(l)];
        }
        cor_off_ = off;
        n_cor_ = spec_.cor.n_params();
        var_off_ = cor_off_ + n_cor_;
        n_var_ = spec_.var.n_params();
        n_theta_ = var_off_ + n_var_;
        {
            Eigen::VectorXd dummy = Eigen::VectorXd::Zero(n_theta_);
            fixed_idx_ = apply_fixed_theta(dummy, spec_.fixed, theta_labels());
            fixed_vals_ = dummy;
            free_idx_.clear();
            for (int k = 0; k < n_theta_; ++k) if (!std::binary_search(fixed_idx_.begin(), fixed_idx_.end(), k)) free_idx_.push_back(k);
        }

        // REML constant: log|X^T X| of the stacked X matrix.
        Eigen::MatrixXd Xs(N_, p_);
        for (size_t gi = 0; gi < g_.size(); ++gi) Xs.middleRows(cache_[gi].row_off, g_[gi].y.size()) = g_[gi].X;
        logdet_xtx_ = 2.0 * RankQR(Xs).log_abs_det();
    }

    /// Number of covariance parameters.
    int n_theta() const { return n_theta_; }
    /// Number of observations.
    int n_obs() const { return N_; }
    /// Number of grouping levels.
    int n_levels() const { return nlev_; }
    /// Grouped data.
    const std::vector<LmeGroup>& groups() const { return g_; }
    /// Model specification.
    const LmeSpec& spec() const { return spec_; }
    /// Indices of the free (estimated) theta parameters.
    const std::vector<int>& free_theta() const { return free_idx_; }
    /// Whether the residual variance is held fixed.
    bool sigma_is_fixed() const { return spec_.sigma_fixed > 0.0; }

    /// Default starting theta, with fixed parameters applied.
    Eigen::VectorXd default_start() const {
        Eigen::VectorXd t = theta_start(kinds_, q_, spec_.cor, spec_.var);
        for (int k : fixed_idx_) t[k] = fixed_vals_[k];
        return t;
    }

    /// Profiled -2 log-likelihood. Returns +inf if theta is infeasible.
    ///
    /// @param theta Parameter vector (length `n_theta()`).
    /// @param det   Optional output for intermediate quantities at theta.
    /// @throws std::invalid_argument If `theta` has the wrong length.
    double neg2ll(const Eigen::VectorXd& theta, LmeDetails* det = nullptr) const {
        constexpr double inf = std::numeric_limits<double>::infinity();
        if (theta.size() != n_theta_) throw std::invalid_argument("neg2ll: theta length");
        try {
            std::vector<Eigen::MatrixXd> Ds = make_D(theta);
            const Eigen::VectorXd tc = theta.segment(cor_off_, n_cor_), tv = theta.segment(var_off_, n_var_);

            Eigen::MatrixXd Xt(N_, p_);
            Eigen::VectorXd yt(N_);
            const int m = static_cast<int>(g_.size());
            std::vector<double> ld(static_cast<size_t>(m), 0.0), smin(static_cast<size_t>(m), 1.0), smax(static_cast<size_t>(m), 1.0);
            std::atomic<bool> bad(false);
#pragma omp parallel for schedule(dynamic)
            for (int gi = 0; gi < m; ++gi) {
                if (bad.load()) continue;
                try {
                    const auto& g = g_[static_cast<size_t>(gi)];
                    if (n_var_ > 0) {
                        const Eigen::VectorXd sc = var_scales(spec_.var, tv, g.vd, static_cast<int>(g.y.size()));
                        if (!sc.allFinite() || sc.minCoeff() <= 0.0) { bad = true; continue; }
                        smin[static_cast<size_t>(gi)] = sc.minCoeff();
                        smax[static_cast<size_t>(gi)] = sc.maxCoeff();
                    }
                    const Eigen::MatrixXd W = build_W(gi, Ds, tc, tv);
                    Eigen::LLT<Eigen::MatrixXd> llt(W);
                    if (llt.info() != Eigen::Success) { bad = true; continue; }
                    ld[static_cast<size_t>(gi)] = 2.0 * llt.matrixLLT().diagonal().array().log().sum();
                    const int n = static_cast<int>(g.y.size());
                    Xt.middleRows(cache_[static_cast<size_t>(gi)].row_off, n) = llt.matrixL().solve(g.X);
                    yt.segment(cache_[static_cast<size_t>(gi)].row_off, n) = llt.matrixL().solve(g.y);
                } catch (...) {
                    bad = true;
                }
            }
            if (bad.load()) return inf;
            const double logdetW = std::accumulate(ld.begin(), ld.end(), 0.0);
            const double var_ratio = *std::max_element(smax.begin(), smax.end()) / *std::min_element(smin.begin(), smin.end());
            if (n_var_ > 0 && var_ratio > spec_.var_max_ratio) return inf;
            {
                std::vector<std::pair<const VarSpec*, int>> comps;
                var_components(spec_.var, 0, comps);
                for (const auto& c : comps)
                    if (c.first->kind == VarKind::ConstPower)
                        for (int k = c.first->n_per(); k < 2 * c.first->n_per(); ++k)
                            if (std::abs(tv[c.second + k]) > spec_.var_shape_bound) return inf;
            }

            RankQR qr(Xt);
            const int rk = qr.rank();
            const Eigen::MatrixXd bet = qr.solve(yt);
            const Eigen::VectorXd res = yt - Xt * bet.col(0);
            const double rss = res.squaredNorm();
            const double dof = spec_.reml ? static_cast<double>(N_ - rk) : static_cast<double>(N_);
            if (!(rss > 0.0) || !(dof > 0.0)) return inf;

            double v;
            if (spec_.sigma_fixed > 0.0) {
                const double s2 = spec_.sigma_fixed * spec_.sigma_fixed;
                v = dof * (std::log(2.0 * M_PI) + std::log(s2)) + logdetW + rss / s2;
            } else {
                v = dof * (std::log(2.0 * M_PI) + std::log(rss / dof) + 1.0) + logdetW;
            }
            if (spec_.reml) {
                v += 2.0 * qr.log_abs_det();
                if (spec_.reml_constant) v -= logdet_xtx_;
            }

            if (det) {
                det->beta = bet.col(0);
                det->rss = rss;
                det->rank = rk;
                det->logdetW = logdetW;
                det->var_ratio = var_ratio;
                Eigen::MatrixXd C = Eigen::MatrixXd::Constant(p_, p_, std::numeric_limits<double>::quiet_NaN());
                if (rk > 0) {
                    const Eigen::MatrixXd Rl = qr.R_leading();
                    const Eigen::MatrixXd Ri =
                        Rl.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(rk, rk));
                    const Eigen::MatrixXd Cp = Ri * Ri.transpose();
                    const auto piv = qr.pivot();
                    for (int i = 0; i < rk; ++i)
                        for (int j = 0; j < rk; ++j) C(piv[static_cast<size_t>(i)], piv[static_cast<size_t>(j)]) = Cp(i, j);
                }
                det->cov_unscaled = C;
            }
            return v;
        } catch (const std::exception&) {
            return inf;
        }
    }

    /// Gradient of the profiled -2 log-likelihood with respect to theta.
    ///
    /// Given the solution beta(theta), the envelope theorem means only W_i must
    /// be differentiated:
    ///   d logdet W      = sum_i tr(W_i^-1 dW_i)
    ///   d RSS           = - sum_i a_i^T dW_i a_i,                a_i = W_i^-1 (y_i - X_i beta)
    ///   d log|X'W^-1X|  = - sum_i tr(M^-1 X_i^T W_i^-1 dW_i W_i^-1 X_i),   M = X'W^-1X
    /// The traces are exact; derivatives of the small matrices D_l(theta),
    /// R_i(theta), and s_i(theta) are computed by central 5-point differences
    /// (relative error ~1e-11). `det` must come from `neg2ll(theta, &det)`.
    ///
    /// @return `false` if theta is infeasible.
    bool neg2ll_grad(const Eigen::VectorXd& theta, const LmeDetails& det, Eigen::VectorXd& grad,
                     Eigen::VectorXd* drss = nullptr) const {
        grad = Eigen::VectorXd::Zero(n_theta_);
        if (drss) *drss = Eigen::VectorXd::Zero(n_theta_);
        if (n_theta_ == 0) return true;
        try {
            const std::vector<Eigen::MatrixXd> Ds = make_D(theta);
            const Eigen::VectorXd tc = theta.segment(cor_off_, n_cor_), tv = theta.segment(var_off_, n_var_);

            std::vector<Eigen::MatrixXd> dD(static_cast<size_t>(n_theta_));
            std::vector<int> lev_of(static_cast<size_t>(n_theta_), -1);
            for (int l = 0; l < nlev_; ++l) {
                const Eigen::VectorXd seg = theta.segment(re_off_[static_cast<size_t>(l)], n_re_l_[static_cast<size_t>(l)]);
                for (int j = 0; j < n_re_l_[static_cast<size_t>(l)]; ++j) {
                    const int k = re_off_[static_cast<size_t>(l)] + j;
                    lev_of[static_cast<size_t>(k)] = l;
                    if (kinds_[static_cast<size_t>(l)].custom && kinds_[static_cast<size_t>(l)].custom->dfactor) {       // analytic derivative of the custom pdMat
                        const Eigen::MatrixXd L = pd_factor(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], seg);
                        const Eigen::MatrixXd dL = kinds_[static_cast<size_t>(l)].custom->dfactor(q_[static_cast<size_t>(l)], seg, j);
                        if (dL.rows() != q_[static_cast<size_t>(l)] || dL.cols() != q_[static_cast<size_t>(l)]) throw std::invalid_argument("pd custom: dfactor must be q x q");
                        dD[static_cast<size_t>(k)] = dL * L.transpose() + L * dL.transpose();
                        continue;
                    }
                    dD[static_cast<size_t>(k)] = diff5(seg, j, [&](const Eigen::VectorXd& t) {
                        const Eigen::MatrixXd L = pd_factor(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], t);
                        return Eigen::MatrixXd(L * L.transpose());
                    });
                }
            }
            std::vector<int> ident;
            for (int c = 0; c < p_; ++c) if (std::isfinite(det.cov_unscaled(c, c))) ident.push_back(c);
            const Eigen::MatrixXd Minv = det.cov_unscaled(ident, ident);

            const int m = static_cast<int>(g_.size());
            std::vector<double> gld(static_cast<size_t>(m) * static_cast<size_t>(n_theta_), 0.0), grss(gld), gxx(gld);
            std::atomic<bool> bad(false);
#pragma omp parallel for schedule(dynamic)
            for (int gi = 0; gi < m; ++gi) {
                if (bad.load()) continue;
                try {
                    const auto& g = g_[static_cast<size_t>(gi)];
                    const int n = static_cast<int>(g.y.size());
                    const Eigen::MatrixXd W = build_W(gi, Ds, tc, tv);
                    Eigen::LLT<Eigen::MatrixXd> llt(W);
                    if (llt.info() != Eigen::Success) { bad = true; continue; }
                    const Eigen::MatrixXd Winv = llt.solve(Eigen::MatrixXd::Identity(n, n));
                    const Eigen::VectorXd a = Winv * (g.y - g.X * det.beta);
                    Eigen::MatrixXd U, UM;
                    if (spec_.reml) {
                        const Eigen::MatrixXd Xp = g.X(Eigen::placeholders::all, ident);
                        U = Winv * Xp;
                        UM = U * Minv;
                    }
                    const Eigen::VectorXd s = var_scales(spec_.var, tv, g.vd, n);
                    const Eigen::MatrixXd R = n_var_ > 0 ? cor_R(gi, tc) : Eigen::MatrixXd();
                    for (int k = 0; k < n_theta_; ++k) {
                        Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(n, n);
                        if (k < cor_off_) {
                            const int l = lev_of[static_cast<size_t>(k)];
                            if (l == 0) {
                                dW = g.Z * dD[static_cast<size_t>(k)] * g.Z.transpose();
                            } else {
                                const auto& sr = cache_[static_cast<size_t>(gi)].sub_rows[static_cast<size_t>(l - 1)];
                                for (size_t b = 0; b < sr.size(); ++b) {
                                    const Eigen::MatrixXd& Zs = cache_[static_cast<size_t>(gi)].sub_Z[static_cast<size_t>(l - 1)][b];
                                    const Eigen::MatrixXd M = Zs * dD[static_cast<size_t>(k)] * Zs.transpose();
                                    for (size_t i = 0; i < sr[b].size(); ++i)
                                        for (size_t j = 0; j < sr[b].size(); ++j) dW(sr[b][i], sr[b][j]) += M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
                                }
                            }
                        } else if (k < var_off_) {
                            const bool cust_d = spec_.cor.kind == CorKind::Custom && spec_.cor.custom && spec_.cor.custom->deriv;
                            const Eigen::MatrixXd dR = spec_.cor.spatial()
                                ? spatial_cor_deriv_blocks(spec_.cor, tc, g.time, g.coords, k - cor_off_, cache_[static_cast<size_t>(gi)].cor_blocks)
                                : cust_d ? custom_cor_deriv_blocks(spec_.cor, tc, g.time, g.coords, k - cor_off_, cache_[static_cast<size_t>(gi)].cor_blocks)
                                : diff5(tc, k - cor_off_, [&](const Eigen::VectorXd& t) {
                                      return cor_matrix_blocks(spec_.cor, t, g.time, g.coords, cache_[static_cast<size_t>(gi)].cor_blocks);
                                  });
                            dW = s.asDiagonal() * dR * s.asDiagonal();
                        } else {
                            Eigen::VectorXd dS;
                            if (!var_scales_deriv(spec_.var, tv, g.vd, n, k - var_off_, dS)) {
                                const Eigen::MatrixXd dSm = diff5(tv, k - var_off_, [&](const Eigen::VectorXd& t) {
                                    return Eigen::MatrixXd(var_scales(spec_.var, t, g.vd, n));
                                });
                                dS = dSm.col(0);
                            }
                            dW = dS.asDiagonal() * R * s.asDiagonal() + s.asDiagonal() * R * dS.asDiagonal();
                        }
                        const size_t idx = static_cast<size_t>(gi) * static_cast<size_t>(n_theta_) + static_cast<size_t>(k);
                        gld[idx] = Winv.cwiseProduct(dW).sum();
                        grss[idx] = -a.dot(dW * a);
                        if (spec_.reml) gxx[idx] = -(dW * U).cwiseProduct(UM).sum();
                    }
                } catch (...) {
                    bad = true;
                }
            }
            if (bad.load()) return false;
            const double dof = spec_.reml ? static_cast<double>(N_ - det.rank) : static_cast<double>(N_);
            const double fac = spec_.sigma_fixed > 0.0 ? 1.0 / (spec_.sigma_fixed * spec_.sigma_fixed) : dof / det.rss;
            for (int k = 0; k < n_theta_; ++k) {
                double sl = 0, sr = 0, sx = 0;
                for (int gi = 0; gi < m; ++gi) {
                    const size_t idx = static_cast<size_t>(gi) * static_cast<size_t>(n_theta_) + static_cast<size_t>(k);
                    sl += gld[idx]; sr += grss[idx]; sx += gxx[idx];
                }
                grad[k] = fac * sr + sl + (spec_.reml ? sx : 0.0);
                if (drss) (*drss)[k] = sr;
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    /// Labels of the theta parameters (for messages).
    std::vector<std::string> theta_labels() const { return make_theta_labels(kinds_, q_, spec_.cor, spec_.var); }

    /// Identifiability diagnostic: Hessian of the profiled -2 log-likelihood
    /// with respect to theta, by central differences of the analytic gradient.
    /// A direction with curvature ~0 is a parameter combination the likelihood
    /// cannot distinguish (e.g. random intercept + corSymm, or the constant of
    /// varConstPower).
    Eigen::MatrixXd sub_free(const Eigen::MatrixXd& H) const {
        const int nf = static_cast<int>(free_idx_.size());
        Eigen::MatrixXd S(nf, nf);
        for (int i = 0; i < nf; ++i) for (int j = 0; j < nf; ++j) S(i, j) = H(free_idx_[static_cast<size_t>(i)], free_idx_[static_cast<size_t>(j)]);
        return S;
    }

    /// Hessian of the profiled -2 log-likelihood (over theta) by central
    /// differences of the analytic gradient.
    ///
    /// `drss` (optional): d RSS / d theta at theta (used to complete the
    /// Hessian with log sigma).
    bool profile_hessian(const Eigen::VectorXd& theta, Eigen::MatrixXd& H, Eigen::VectorXd* drss = nullptr) const {
        const int n = n_theta_;
        H = Eigen::MatrixXd::Zero(n, n);
        if (drss) {
            LmeDetails d0;
            Eigen::VectorXd g0;
            if (!std::isfinite(neg2ll(theta, &d0)) || !neg2ll_grad(theta, d0, g0, drss)) return false;
        }
        for (int k = 0; k < n; ++k) {
            const double h = 1e-4 * std::max(1.0, std::abs(theta[k]));
            Eigen::VectorXd gp, gm, tp = theta, tm = theta;
            tp[k] += h; tm[k] -= h;
            LmeDetails dp, dm;
            if (!std::isfinite(neg2ll(tp, &dp)) || !std::isfinite(neg2ll(tm, &dm)) ||
                !neg2ll_grad(tp, dp, gp) || !neg2ll_grad(tm, dm, gm))
                return false;
            H.col(k) = (gp - gm) / (2.0 * h);
        }
        H = 0.5 * (H + H.transpose());
        return true;
    }

    /// Build identifiability diagnostics from a Hessian and the parameter
    /// indices it refers to.
    LmeIdent ident_from_hessian(const Eigen::MatrixXd& H, const std::vector<int>& idx) const {
        LmeIdent id;
        const int n = static_cast<int>(H.rows());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        id.eigenvalues = es.eigenvalues();
        id.eigenvectors = es.eigenvectors();
        id.evaluated = true;
        const double lmax = id.eigenvalues.cwiseAbs().maxCoeff();
        const auto nm_all = theta_labels();
        std::vector<std::string> nm(static_cast<size_t>(n));
        for (int k = 0; k < n; ++k) nm[static_cast<size_t>(k)] = nm_all[static_cast<size_t>(idx[static_cast<size_t>(k)])];
        for (int i = 0; i < n; ++i) {
            const double lam = id.eigenvalues[i];
            if (lam < -1e-6 * lmax) id.saddle = true;
            if (std::abs(lam) <= 1e-6 * lmax) {
                ++id.n_weak;
                std::string comp;
                for (int k = 0; k < n; ++k)
                    if (std::abs(id.eigenvectors(k, i)) > 0.3) comp += (comp.empty() ? "" : ", ") + nm[static_cast<size_t>(k)];
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.1e", std::abs(lam) / lmax);
                id.message += (id.message.empty() ? "" : "; ") + std::string("nearly unidentified direction (curvature/max = ") +
                              buf + ") over {" + comp + "}";
            }
        }
        if (id.saddle) id.message += std::string(id.message.empty() ? "" : "; ") + "Hessian with a negative eigenvalue: not a local minimum";
        return id;
    }

    /// Run the identifiability diagnostic at theta.
    LmeIdent identifiability(const Eigen::VectorXd& theta) const {
        LmeIdent id;
        if (n_theta_ == 0) { id.message = "no covariance parameters"; return id; }
        Eigen::MatrixXd H;
        if (!profile_hessian(theta, H)) { id.message = "not evaluated: theta at the boundary of the feasible region"; return id; }
        return ident_from_hessian(sub_free(H), free_idx_);
    }

    /// Covariance of (theta, log sigma): 2 * H_full^-1 with
    ///   H_full = [[H_prof + b b'/(4 dof), b], [b', 4 dof]],  b = -2 (dof/RSS) dRSS/dtheta
    /// (the profiled Hessian is completed with the log-sigma direction).
    /// Returns false if H_full is not positive definite.
    bool apvar(const Eigen::MatrixXd& Hprof, const Eigen::VectorXd& drss, double dof, double rss, Eigen::MatrixXd& cov) const {
        const int n = static_cast<int>(Hprof.rows());
        if (spec_.sigma_fixed > 0.0) {                  // fixed sigma: theta only
            if (n == 0) { cov = Eigen::MatrixXd(0, 0); return true; }
            Eigen::LLT<Eigen::MatrixXd> l0(Hprof);
            if (l0.info() != Eigen::Success) return false;
            cov = 2.0 * l0.solve(Eigen::MatrixXd::Identity(n, n));
            return true;
        }
        Eigen::MatrixXd Hf(n + 1, n + 1);
        const Eigen::VectorXd b = (n > 0) ? Eigen::VectorXd(-2.0 * dof / rss * drss) : Eigen::VectorXd(0);
        if (n > 0) {
            Hf.topLeftCorner(n, n) = Hprof + b * b.transpose() / (4.0 * dof);
            Hf.topRightCorner(n, 1) = b;
            Hf.bottomLeftCorner(1, n) = b.transpose();
        }
        Hf(n, n) = 4.0 * dof;
        Eigen::LLT<Eigen::MatrixXd> llt(Hf);
        if (llt.info() != Eigen::Success) return false;
        cov = 2.0 * llt.solve(Eigen::MatrixXd::Identity(n + 1, n + 1));
        return true;
    }

    /// Parameters on nlme's natural scale (see `LmeResult::vcov_natural`).
    void natural_params(const Eigen::VectorXd& th, double logsig, Eigen::VectorXd& out, std::vector<std::string>& names) const {
        std::vector<double> v;
        names.clear();
        for (int l = 0; l < nlev_; ++l)
            if (q_[static_cast<size_t>(l)] > 0) detail::pd_natural_append(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], th.segment(re_off_[static_cast<size_t>(l)], n_re_l_[static_cast<size_t>(l)]), logsig, "re" + std::to_string(l) + ".", v, names);
        for (int j = 0; j < n_cor_; ++j) {
            double x = th[cor_off_ + j];
            if (spec_.cor.kind == CorKind::AR1) x *= 2.0;
            v.push_back(x);
            names.push_back("cor[" + std::to_string(j) + "]");
        }
        for (int j = 0; j < n_var_; ++j) { v.push_back(th[var_off_ + j]); names.push_back("var[" + std::to_string(j) + "]"); }
        if (spec_.sigma_fixed <= 0.0) { v.push_back(logsig); names.push_back("lSigma"); }
        out = Eigen::Map<Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
    }

    /// Absolute standard deviations, correlations, and cor/var parameters on
    /// the constrained scale, plus sigma.
    void constrained_params(const Eigen::VectorXd& th, double logsig, Eigen::VectorXd& out, std::vector<std::string>& names) const {
        std::vector<double> v;
        names.clear();
        const double sigma = std::exp(logsig);
        for (int l = 0; l < nlev_; ++l)
            if (q_[static_cast<size_t>(l)] > 0) detail::pd_constrained_append(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], th.segment(re_off_[static_cast<size_t>(l)], n_re_l_[static_cast<size_t>(l)]), sigma, "re" + std::to_string(l) + ".", v, names);
        const Eigen::VectorXd tc = th.segment(cor_off_, n_cor_), tv = th.segment(var_off_, n_var_);
        switch (spec_.cor.kind) {
            case CorKind::AR1: v.push_back(std::tanh(tc[0])); names.push_back("cor.phi"); break;
            case CorKind::CAR1: v.push_back(detail::logistic(tc[0])); names.push_back("cor.phi"); break;
            case CorKind::CompoundSymmetry: {
                const double lo = -1.0 / (std::max(spec_.cor.max_group_size, 2) - 1.0);
                v.push_back(lo + (1.0 - lo) * detail::logistic(tc[0]));
                names.push_back("cor.rho");
                break;
            }
            case CorKind::Exp: case CorKind::Gaus: case CorKind::Lin: case CorKind::Ratio: case CorKind::Spher:
                v.push_back(std::exp(tc[0])); names.push_back("cor.range");
                if (spec_.cor.nugget) { v.push_back(detail::logistic(tc[1])); names.push_back("cor.nugget"); }
                break;
            case CorKind::ARMA: {
                const Eigen::VectorXd ar = pacf_to_ar(tc.head(spec_.cor.p)), ma = -pacf_to_ar(tc.tail(spec_.cor.q));
                for (int i = 0; i < ar.size(); ++i) { v.push_back(ar[i]); names.push_back("cor.ar[" + std::to_string(i + 1) + "]"); }
                for (int i = 0; i < ma.size(); ++i) { v.push_back(ma[i]); names.push_back("cor.ma[" + std::to_string(i + 1) + "]"); }
                break;
            }
            default:
                for (int j = 0; j < n_cor_; ++j) { v.push_back(tc[j]); names.push_back("cor[" + std::to_string(j) + "] (theta)"); }
        }
        {
            std::vector<std::pair<const VarSpec*, int>> comps;
            var_components(spec_.var, 0, comps);
            int ci = 0;
            for (const auto& c : comps) {
                const VarSpec& vs = *c.first;
                const int np = vs.n_per(), npar = vs.n_params();
                const std::string pre = "var" + std::to_string(ci++) + ".";
                for (int j = 0; j < npar; ++j) {
                    const double x = tv[c.second + j];
                    switch (vs.kind) {
                        case VarKind::Ident:      v.push_back(std::exp(x)); names.push_back(pre + "ratio[" + std::to_string(j) + "]"); break;
                        case VarKind::ConstPower: v.push_back(j < np ? std::exp(x) : x); names.push_back(pre + (j < np ? "const[" : "power[") + std::to_string(j % np) + "]"); break;
                        case VarKind::ConstProp:  v.push_back(std::exp(x)); names.push_back(pre + (j < np ? "const[" : "prop[") + std::to_string(j % np) + "]"); break;
                        default:                  v.push_back(x); names.push_back(pre + "delta[" + std::to_string(j) + "]"); break;
                    }
                }
            }
        }
        v.push_back(sigma);
        names.push_back("sigma");
        out = Eigen::Map<Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
    }

    /// Marginal covariance of the observations in group gi:
    /// sigma^2 (S R S + sum Z D Z').
    Eigen::MatrixXd marginal_cov(const LmeResult& r, int gi) const {
        const std::vector<Eigen::MatrixXd> Ds = make_D(r.theta);
        return r.sigma2 * build_W(gi, Ds, r.theta_cor, r.theta_var);
    }

    /// Fit the model.
    ///
    /// If the variance function uses fitted(.), the fit is iterated: the
    /// covariate is replaced by the conditional fitted values of the previous
    /// fit (the first one without a variance function) until theta stabilizes.
    ///
    /// @param start Optional starting theta.
    /// @return The fit results.
    /// @throws std::runtime_error If the fit is infeasible.
    LmeResult fit(const Eigen::VectorXd* start = nullptr) const {
        if (!spec_.var.has_fitted()) return fit_core(start);
        std::vector<std::pair<const VarSpec*, int>> comps;
        var_components(spec_.var, 0, comps);
        LmeSpec s0 = spec_;
        s0.var = VarSpec();
        s0.diagnose = false;
        LmeResult cur = LmeModel(g_, s0).fit_core();
        Eigen::VectorXd theta_prev;
        LmeResult last;
        for (int it = 0; it < 50; ++it) {
            std::vector<LmeGroup> gs = g_;
            int row = 0;
            for (auto& g : gs) {
                const int n = static_cast<int>(g.y.size());
                const Eigen::VectorXd fv = cur.fitted_cond.segment(row, n);
                for (const auto& c : comps)
                    if (c.first->fitted) {
                        if (g.vd.cov.size() <= static_cast<size_t>(c.first->cov_slot)) g.vd.cov.resize(static_cast<size_t>(c.first->cov_slot + 1));
                        g.vd.cov[static_cast<size_t>(c.first->cov_slot)] = fv;
                    }
                row += n;
            }
            LmeSpec s1 = spec_;
            s1.var = strip_fitted(spec_.var);
            s1.diagnose = false;
            LmeModel m(gs, s1);
            const Eigen::VectorXd st = (it == 0 && start && start->size() == m.n_theta()) ? *start
                                       : (theta_prev.size() ? theta_prev : m.default_start());
            last = m.fit_core(&st);
            const bool conv = theta_prev.size() && (last.theta - theta_prev).cwiseAbs().maxCoeff() < 1e-7;
            theta_prev = last.theta;
            // fitted values in the model's group order
            cur = last;
            last.var_iterations = it + 1;
            if (conv) break;
        }
        // Final result: last fixed covariate, with the full diagnostic.
        {
            std::vector<LmeGroup> gs = g_;
            int row = 0;
            for (auto& g : gs) {
                const int n = static_cast<int>(g.y.size());
                const Eigen::VectorXd fv = cur.fitted_cond.segment(row, n);
                for (const auto& c : comps)
                    if (c.first->fitted) {
                        if (g.vd.cov.size() <= static_cast<size_t>(c.first->cov_slot)) g.vd.cov.resize(static_cast<size_t>(c.first->cov_slot + 1));
                        g.vd.cov[static_cast<size_t>(c.first->cov_slot)] = fv;
                    }
                row += n;
            }
            LmeSpec s1 = spec_;
            s1.var = strip_fitted(spec_.var);
            LmeResult fin = LmeModel(gs, s1).fit_core(&theta_prev);
            fin.var_iterations = last.var_iterations;
            return fin;
        }
    }

    /// One EM update of the relative covariance of each level (Laird & Ware
    /// 1982; REML version if `spec.reml`), keeping beta, sigma^2, and the
    /// cor/var parameters fixed at their current values.
    ///
    /// Returns the new theta (levels with fixed parameters are not updated).
    /// Useful as a starting point; at the ML optimum it is a fixed point.
    Eigen::VectorXd em_step(const Eigen::VectorXd& th) const {
        const int m = static_cast<int>(g_.size());
        const std::vector<Eigen::MatrixXd> Ds = make_D(th);
        const Eigen::VectorXd tc = th.segment(cor_off_, n_cor_), tv = th.segment(var_off_, n_var_);
        std::vector<Eigen::MatrixXd> Winv(static_cast<size_t>(m));
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(p_, p_);
        Eigen::VectorXd c = Eigen::VectorXd::Zero(p_);
        for (int gi = 0; gi < m; ++gi) {
            const auto& g = g_[static_cast<size_t>(gi)];
            Eigen::LLT<Eigen::MatrixXd> llt(build_W(gi, Ds, tc, tv));
            if (llt.info() != Eigen::Success) return th;
            Winv[static_cast<size_t>(gi)] = llt.solve(Eigen::MatrixXd::Identity(g.y.size(), g.y.size()));
            M.noalias() += g.X.transpose() * Winv[static_cast<size_t>(gi)] * g.X;
            c.noalias() += g.X.transpose() * Winv[static_cast<size_t>(gi)] * g.y;
        }
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(M);
        const Eigen::MatrixXd Minv = cod.pseudoInverse();
        const Eigen::VectorXd beta = Minv * c;
        double rss = 0.0;
        for (int gi = 0; gi < m; ++gi) {
            const Eigen::VectorXd r = g_[static_cast<size_t>(gi)].y - g_[static_cast<size_t>(gi)].X * beta;
            rss += r.dot(Winv[static_cast<size_t>(gi)] * r);
        }
        const double dof = spec_.reml ? double(N_ - static_cast<int>(cod.rank())) : double(N_);
        const double s2 = spec_.sigma_fixed > 0.0 ? spec_.sigma_fixed * spec_.sigma_fixed : rss / dof;
        std::vector<Eigen::MatrixXd> acc(static_cast<size_t>(nlev_));
        std::vector<double> cnt(static_cast<size_t>(nlev_), 0.0);
        for (int l = 0; l < nlev_; ++l) acc[static_cast<size_t>(l)] = Eigen::MatrixXd::Zero(q_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)]);
        for (int gi = 0; gi < m; ++gi) {
            const auto& g = g_[static_cast<size_t>(gi)];
            const Eigen::VectorXd a = Winv[static_cast<size_t>(gi)] * (g.y - g.X * beta);
            auto unit = [&](int l, const Eigen::MatrixXd& Zu, const std::vector<int>* idx) {
                const int q = q_[static_cast<size_t>(l)];
                Eigen::VectorXd au;
                Eigen::MatrixXd Wuu, WX;
                if (!idx) { au = a; Wuu = Winv[static_cast<size_t>(gi)]; WX = Winv[static_cast<size_t>(gi)] * g.X; }
                else {
                    au.resize(static_cast<Eigen::Index>(idx->size()));
                    Wuu.resize(static_cast<Eigen::Index>(idx->size()), static_cast<Eigen::Index>(idx->size()));
                    for (size_t i = 0; i < idx->size(); ++i) {
                        au[static_cast<Eigen::Index>(i)] = a[(*idx)[i]];
                        for (size_t j = 0; j < idx->size(); ++j) Wuu(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = Winv[static_cast<size_t>(gi)]((*idx)[i], (*idx)[j]);
                    }
                    Eigen::MatrixXd Xu(idx->size(), p_);
                    for (size_t i = 0; i < idx->size(); ++i) Xu.row(static_cast<Eigen::Index>(i)) = g.X.row((*idx)[i]);
                    // (W^-1 X) restricted to the rows of the unit
                    WX.resize(static_cast<Eigen::Index>(idx->size()), p_);
                    for (size_t i = 0; i < idx->size(); ++i) WX.row(static_cast<Eigen::Index>(i)) = (Winv[static_cast<size_t>(gi)].row((*idx)[i]) * g.X);
                }
                const Eigen::VectorXd b = Ds[static_cast<size_t>(l)] * Zu.transpose() * au;
                Eigen::MatrixXd ZPZ = Zu.transpose() * Wuu * Zu;
                if (spec_.reml) {
                    const Eigen::MatrixXd ZWX = Zu.transpose() * WX;
                    ZPZ -= ZWX * Minv * ZWX.transpose();
                }
                acc[static_cast<size_t>(l)] += b * b.transpose() / s2 + Ds[static_cast<size_t>(l)] - Ds[static_cast<size_t>(l)] * ZPZ * Ds[static_cast<size_t>(l)];
                cnt[static_cast<size_t>(l)] += 1.0;
                (void)q;
            };
            if (q_[0] > 0) unit(0, g.Z, nullptr);
            for (int l = 1; l < nlev_; ++l) {
                if (q_[static_cast<size_t>(l)] == 0) continue;
                const auto& sr = cache_[static_cast<size_t>(gi)].sub_rows[static_cast<size_t>(l - 1)];
                for (size_t k = 0; k < sr.size(); ++k) unit(l, cache_[static_cast<size_t>(gi)].sub_Z[static_cast<size_t>(l - 1)][k], &sr[k]);
            }
        }
        Eigen::VectorXd out = th;
        for (int l = 0; l < nlev_; ++l) {
            if (q_[static_cast<size_t>(l)] == 0 || cnt[static_cast<size_t>(l)] == 0) continue;
            bool has_fixed = false;
            for (int k : fixed_idx_) if (k >= re_off_[static_cast<size_t>(l)] && k < re_off_[static_cast<size_t>(l)] + n_re_l_[static_cast<size_t>(l)]) has_fixed = true;
            if (has_fixed) continue;
            const Eigen::MatrixXd Dn = 0.5 * (acc[static_cast<size_t>(l)] + acc[static_cast<size_t>(l)].transpose()) / cnt[static_cast<size_t>(l)];
            const Eigen::VectorXd tn = detail::pd_theta_from_D(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], Dn);
            if (tn.size() == n_re_l_[static_cast<size_t>(l)]) out.segment(re_off_[static_cast<size_t>(l)], n_re_l_[static_cast<size_t>(l)]) = tn;
        }
        return out;
    }

    /// Optimization core: starting values, EM warm-up, and the optimizer.
    LmeResult fit_core(const Eigen::VectorXd* start = nullptr) const {
        const bool given = start && start->size() == n_theta_;
        Eigen::VectorXd th = given ? *start : default_start();
        for (int k : fixed_idx_) th[k] = fixed_vals_[k];
        if (!std::isfinite(neg2ll(th))) { th = default_start(); }
        bool has_re = false;
        for (int qq : q_) has_re = has_re || qq > 0;
        if (spec_.em_iter > 0 && !given && n_theta_ > 0 && has_re) {
            Eigen::VectorXd t2 = th;
            for (int it = 0; it < spec_.em_iter; ++it) t2 = em_step(t2);
            if (std::isfinite(neg2ll(t2))) th = t2;
        }

        LmeResult res;
        const int nf = static_cast<int>(free_idx_.size());
        if (nf > 0) {
            const Eigen::VectorXd base = th;
            auto expand = [&](const Eigen::VectorXd& u) {
                Eigen::VectorXd t = base;
                for (int i = 0; i < nf; ++i) t[free_idx_[static_cast<size_t>(i)]] = u[i];
                return t;
            };
            Eigen::VectorXd u0(nf);
            for (int i = 0; i < nf; ++i) u0[i] = th[free_idx_[static_cast<size_t>(i)]];

            struct Memo { Eigen::VectorXd th; double f = 0; LmeDetails det; bool ok = false; } memo;
            auto fobj = [&](const Eigen::VectorXd& u) {
                const Eigen::VectorXd t = expand(u);
                LmeDetails d;
                const double v = neg2ll(t, spec_.analytic_grad ? &d : nullptr);
                if (spec_.analytic_grad) { memo.th = t; memo.f = v; memo.det = std::move(d); memo.ok = std::isfinite(v); }
                return v;
            };
            Gradient gobj;
            if (spec_.analytic_grad)
                gobj = [&](const Eigen::VectorXd& u, Eigen::VectorXd& g) {
                    const Eigen::VectorXd t = expand(u);
                    if (!(memo.ok && memo.th.size() == t.size() && (memo.th - t).cwiseAbs().maxCoeff() == 0.0)) {
                        LmeDetails d;
                        const double v = neg2ll(t, &d);
                        memo.th = t; memo.f = v; memo.det = std::move(d); memo.ok = std::isfinite(v);
                    }
                    Eigen::VectorXd gf;
                    // Infeasible region: f = inf (external optimizers see 1e30)
                    // and the gradient is finite and zero.
                    if (!memo.ok || !neg2ll_grad(t, memo.det, gf)) gf = Eigen::VectorXd::Zero(n_theta_);
                    g.resize(nf);
                    for (int i = 0; i < nf; ++i) g[i] = gf[free_idx_[static_cast<size_t>(i)]];
                };
            res.opt = run_optimizer(spec_.optimizer, fobj, gobj, u0, spec_.opt, spec_.theta_bound, static_cast<double>(N_));
            // Optimality check with the analytic gradient: an external optimizer
            // that stops with a large gradient outside the bounds (bad scaling,
            // infeasible region) is finished off with the internal BFGS.
            if (spec_.optimizer != Optimizer::Internal && spec_.analytic_grad) {
                Eigen::VectorXd gchk(nf);
                gobj(res.opt.x, gchk);
                double pg = 0.0;
                for (int k = 0; k < gchk.size(); ++k) {
                    const bool at_bound = spec_.optimizer == Optimizer::LBFGSB &&
                                          std::abs(res.opt.x[k]) >= 0.999 * spec_.theta_bound;
                    if (!at_bound) pg = std::max(pg, std::abs(gchk[k]));
                }
                if (pg > 1e-3) {
                    BfgsResult r2 = minimize_bfgs(fobj, res.opt.x, gobj, spec_.opt);
                    if (std::isfinite(r2.f) && r2.f <= res.opt.f) {
                        r2.message = "polished with internal BFGS (the external optimizer stopped with |g|=" +
                                     std::to_string(pg) + "): " + r2.message;
                        res.opt = r2;
                    }
                }
            }
            th = expand(res.opt.x);
        }
        finalize(th, res);
        return res;
    }


private:
    /// Derivative with respect to x[k] of a matrix-valued function, by central
    /// 5-point differences.
    template <class F>
    static Eigen::MatrixXd diff5(const Eigen::VectorXd& x, int k, const F& fun, double h = 1e-3) {
        auto at = [&](double d) { Eigen::VectorXd t = x; t[k] += d; return Eigen::MatrixXd(fun(t)); };
        return (-at(2 * h) + 8.0 * at(h) - 8.0 * at(-h) + at(-2 * h)) / (12.0 * h);
    }

    /// Per-group precomputed structure.
    struct Cache {
        int row_off = 0;                                         ///< Offset among stacked observations.
        std::vector<std::vector<std::vector<int>>> sub_rows;     ///< [level-1][sub] -> rows.
        std::vector<std::vector<Eigen::MatrixXd>> sub_Z;         ///< [level-1][sub] -> Z(rows).
        std::vector<std::vector<int>> cor_blocks;                ///< Correlation blocks (empty -> a single block).
    };

    /// Relative covariance matrices D_l at theta.
    std::vector<Eigen::MatrixXd> make_D(const Eigen::VectorXd& theta) const {
        std::vector<Eigen::MatrixXd> Ds(static_cast<size_t>(nlev_));
        for (int l = 0; l < nlev_; ++l) {
            if (q_[static_cast<size_t>(l)] > 0) {
                const Eigen::MatrixXd L = pd_factor(kinds_[static_cast<size_t>(l)], q_[static_cast<size_t>(l)], theta.segment(re_off_[static_cast<size_t>(l)], n_re_l_[static_cast<size_t>(l)]));
                Ds[static_cast<size_t>(l)] = L * L.transpose();
            } else {
                Ds[static_cast<size_t>(l)] = Eigen::MatrixXd::Zero(0, 0);
            }
        }
        return Ds;
    }

    /// Within-group correlation matrix (blockwise when the group has cor_sub).
    Eigen::MatrixXd cor_R(int gi, const Eigen::VectorXd& tc) const {
        const auto& g = g_[static_cast<size_t>(gi)];
        return cor_matrix_blocks(spec_.cor, tc, g.time, g.coords, cache_[static_cast<size_t>(gi)].cor_blocks);
    }

    /// Relative marginal covariance of group gi.
    Eigen::MatrixXd build_W(int gi, const std::vector<Eigen::MatrixXd>& Ds, const Eigen::VectorXd& tc,
                            const Eigen::VectorXd& tv) const {
        const auto& g = g_[static_cast<size_t>(gi)];
        const int n = static_cast<int>(g.y.size());
        const Eigen::VectorXd s = var_scales(spec_.var, tv, g.vd, n);
        const Eigen::MatrixXd R = cor_R(gi, tc);
        Eigen::MatrixXd W = s.asDiagonal() * R * s.asDiagonal();
        if (q_[0] > 0) W.noalias() += g.Z * Ds[0] * g.Z.transpose();
        for (int l = 1; l < nlev_; ++l) {
            if (q_[static_cast<size_t>(l)] == 0) continue;
            const auto& sr = cache_[static_cast<size_t>(gi)].sub_rows[static_cast<size_t>(l - 1)];
            for (size_t k = 0; k < sr.size(); ++k) {
                const Eigen::MatrixXd& Zs = cache_[static_cast<size_t>(gi)].sub_Z[static_cast<size_t>(l - 1)][k];
                const Eigen::MatrixXd M = Zs * Ds[static_cast<size_t>(l)] * Zs.transpose();
                for (size_t a = 0; a < sr[k].size(); ++a)
                    for (size_t b = 0; b < sr[k].size(); ++b) W(sr[k][a], sr[k][b]) += M(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
            }
        }
        return W;
    }

    /// Assemble the full result from the final theta.
    ///
    /// @throws std::runtime_error If the fit is infeasible at theta.
    void finalize(const Eigen::VectorXd& th, LmeResult& res) const {
        LmeDetails det;
        const double v = neg2ll(th, &det);
        if (!std::isfinite(v)) throw std::runtime_error("LmeModel: infeasible fit");

        res.reml = spec_.reml;
        res.N = N_;
        res.rank = det.rank;
        res.loglik = -0.5 * v;
        res.beta = det.beta;
        const double dof = spec_.reml ? N_ - det.rank : N_;
        res.sigma2 = spec_.sigma_fixed > 0.0 ? spec_.sigma_fixed * spec_.sigma_fixed : det.rss / dof;
        // Standard-error convention as in nlme: sigma^2 = RSS/(N-p) also under ML.
        // nlme convention: sigma^2 * dof/(N-p) with dof = N (ML) or N-p (REML);
        // with free sigma this equals RSS/(N-p).
        const double s2se = res.sigma2 * (spec_.reml ? 1.0 : double(N_) / (N_ - det.rank));
        res.se = (s2se * det.cov_unscaled.diagonal().array()).sqrt();
        res.theta = th;
        const double thr = spec_.optimizer == Optimizer::LBFGSB ? std::min(30.0, 0.999 * spec_.theta_bound) : 30.0;
        if (th.size() > 0 && th.cwiseAbs().maxCoeff() > thr)
            res.warning = "some unconstrained parameter is at the limit (|theta| > threshold): possible degeneracy or unidentified parameter";
        if (det.var_ratio > 0.9 * spec_.var_max_ratio && n_var_ > 0)
            res.warning += std::string(res.warning.empty() ? "" : " | ") + "heteroscedasticity at the limit (var_max_ratio): the likelihood grows toward a degenerate supremum";
        {
            std::vector<std::pair<const VarSpec*, int>> comps;
            var_components(spec_.var, 0, comps);
            for (const auto& c : comps)
                if (c.first->kind == VarKind::ConstPower)
                    for (int k = c.first->n_per(); k < 2 * c.first->n_per(); ++k)
                        if (std::abs(th[var_off_ + c.second + k]) > 0.9 * spec_.var_shape_bound)
                            res.warning += std::string(res.warning.empty() ? "" : " | ") +
                                   "varConstPower shape parameter at the limit (var_shape_bound): the likelihood has no interior optimum (it tends to a step variance)";
        }
        res.theta_cor = th.segment(cor_off_, n_cor_);
        res.theta_var = th.segment(var_off_, n_var_);
        res.theta_re.clear();
        for (int l = 0; l < nlev_; ++l) res.theta_re.push_back(th.segment(re_off_[static_cast<size_t>(l)], n_re_l_[static_cast<size_t>(l)]));

        const std::vector<Eigen::MatrixXd> Ds = make_D(th);
        res.D_levels = Ds;
        res.Sigma_levels.clear();
        for (const auto& D : Ds) res.Sigma_levels.push_back(res.sigma2 * D);
        res.D = Ds[0];
        res.Sigma_b = res.sigma2 * Ds[0];

        res.vcov_beta = res.sigma2 * (spec_.reml ? 1.0 : double(N_) / (N_ - det.rank)) * det.cov_unscaled;
        Eigen::MatrixXd Hprof;
        Eigen::VectorXd drss;
        bool have_H = false;
        const int nfree = static_cast<int>(free_idx_.size());
        if (nfree == 0) {
            have_H = true;
        } else if (spec_.diagnose && n_theta_ <= 40 && res.warning.find("limit") == std::string::npos) {
            // Skip the diagnostic when a parameter already sits at a bound.
            Eigen::MatrixXd Hfull;
            Eigen::VectorXd dfull;
            have_H = profile_hessian(th, Hfull, &dfull);
            if (have_H) {
                Hprof = sub_free(Hfull);
                drss.resize(nfree);
                for (int i = 0; i < nfree; ++i) drss[i] = dfull[free_idx_[static_cast<size_t>(i)]];
                res.ident = ident_from_hessian(Hprof, free_idx_);
                if (res.ident.n_weak > 0 || res.ident.saddle)
                    res.warning += std::string(res.warning.empty() ? "" : " | ") + "identifiability: " + res.ident.message;
            } else {
                res.ident.message = "not evaluated: theta at the boundary of the feasible region";
            }
        }
        if (have_H) {
            Eigen::MatrixXd cov;
            if (apvar(Hprof, drss, spec_.reml ? double(N_ - det.rank) : double(N_), det.rss, cov)) {
                res.vcov_theta = cov;
                const double logsig = 0.5 * std::log(res.sigma2);
                const bool sf = spec_.sigma_fixed > 0.0;
                const int next = nfree + (sf ? 0 : 1);
                // theta_ext = (free theta, [log sigma]); fixed entries stay constant
                Eigen::VectorXd ext(next);
                for (int i = 0; i < nfree; ++i) ext[i] = th[free_idx_[static_cast<size_t>(i)]];
                if (!sf) ext[nfree] = logsig;
                auto theta_of = [&](const Eigen::VectorXd& e) {
                    Eigen::VectorXd t = th;
                    for (int i = 0; i < nfree; ++i) t[free_idx_[static_cast<size_t>(i)]] = e[i];
                    return t;
                };
                auto lsig = [&](const Eigen::VectorXd& e) { return sf ? logsig : e[nfree]; };
                // Natural scale: skipped if any random-effect parameter is fixed;
                // fixed cor/var entries are dropped.
                bool re_fixed = false;
                for (int k : fixed_idx_) if (k < cor_off_) re_fixed = true;
                if (!re_fixed) {
                    Eigen::VectorXd nat0;
                    std::vector<std::string> nn0;
                    natural_params(th, logsig, nat0, nn0);
                    // Rows to keep: all except fixed cor[j]/var[j].
                    std::vector<int> keep;
                    for (int i = 0; i < nat0.size(); ++i) {
                        bool drop = false;
                        for (int k : fixed_idx_) {
                            if (k >= cor_off_ && k < var_off_ && nn0[static_cast<size_t>(i)] == "cor[" + std::to_string(k - cor_off_) + "]") drop = true;
                            if (k >= var_off_ && nn0[static_cast<size_t>(i)] == "var[" + std::to_string(k - var_off_) + "]") drop = true;
                        }
                        if (!drop) keep.push_back(i);
                    }
                    Eigen::MatrixXd J(keep.size(), next);
                    for (int k = 0; k < next; ++k) {
                        const double h = 1e-6 * std::max(1.0, std::abs(ext[k]));
                        Eigen::VectorXd ep = ext, em = ext, np_, nm_;
                        ep[k] += h; em[k] -= h;
                        std::vector<std::string> dummy;
                        natural_params(theta_of(ep), lsig(ep), np_, dummy);
                        natural_params(theta_of(em), lsig(em), nm_, dummy);
                        for (size_t r = 0; r < keep.size(); ++r) J(static_cast<Eigen::Index>(r), k) = (np_[keep[r]] - nm_[keep[r]]) / (2.0 * h);
                    }
                    res.vcov_natural = J * cov * J.transpose();
                    res.theta_natural.resize(static_cast<Eigen::Index>(keep.size()));
                    for (size_t r = 0; r < keep.size(); ++r) { res.theta_natural[static_cast<Eigen::Index>(r)] = nat0[keep[r]]; res.natural_names.push_back(nn0[static_cast<size_t>(keep[r])]); }
                }
                Eigen::VectorXd cp;
                std::vector<std::string> cn;
                constrained_params(th, logsig, cp, cn);
                Eigen::MatrixXd Jc(cp.size(), next);
                for (int k = 0; k < next; ++k) {
                    const double h = 1e-6 * std::max(1.0, std::abs(ext[k]));
                    Eigen::VectorXd ep = ext, em = ext, a, b2;
                    ep[k] += h; em[k] -= h;
                    std::vector<std::string> dummy;
                    constrained_params(theta_of(ep), lsig(ep), a, dummy);
                    constrained_params(theta_of(em), lsig(em), b2, dummy);
                    Jc.col(k) = (a - b2) / (2.0 * h);
                }
                const Eigen::MatrixXd Vc = Jc * cov * Jc.transpose();
                res.varcomp.clear();
                for (int i = 0; i < cp.size(); ++i) res.varcomp.push_back({cn[static_cast<size_t>(i)], cp[i], std::sqrt(std::max(Vc(i, i), 0.0))});
            } else {
                res.warning += std::string(res.warning.empty() ? "" : " | ") + "apVar unavailable: the likelihood Hessian is not positive definite";
            }
        }
        res.fitted_pop.resize(N_);
        res.fitted_cond.resize(N_);
        res.resid.resize(N_);
        res.order.resize(static_cast<size_t>(N_));
        res.blup.clear();
        res.blup_inner.clear();
        res.labels.clear();
        const Eigen::VectorXd bet = res.beta;
        for (size_t gi = 0; gi < g_.size(); ++gi) {
            const auto& g = g_[gi];
            const int n = static_cast<int>(g.y.size());
            const int row = cache_[gi].row_off;
            Eigen::LLT<Eigen::MatrixXd> llt(build_W(static_cast<int>(gi), Ds, res.theta_cor, res.theta_var));
            const Eigen::VectorXd pop = g.X * bet;
            const Eigen::VectorXd wr = llt.solve(g.y - pop);
            Eigen::VectorXd re_fit = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd b0 = Eigen::VectorXd::Zero(q_[0]);
            if (q_[0] > 0) {
                b0 = Ds[0] * g.Z.transpose() * wr;
                re_fit += g.Z * b0;
            }
            std::vector<std::vector<Eigen::VectorXd>> inner(static_cast<size_t>(nlev_ - 1));
            for (int l = 1; l < nlev_; ++l) {
                const auto& sr = cache_[gi].sub_rows[static_cast<size_t>(l - 1)];
                for (size_t k = 0; k < sr.size(); ++k) {
                    Eigen::VectorXd bk = Eigen::VectorXd::Zero(q_[static_cast<size_t>(l)]);
                    if (q_[static_cast<size_t>(l)] > 0) {
                        Eigen::VectorXd ws(sr[k].size());
                        for (size_t a = 0; a < sr[k].size(); ++a) ws[static_cast<Eigen::Index>(a)] = wr[sr[k][a]];
                        bk = Ds[static_cast<size_t>(l)] * cache_[gi].sub_Z[static_cast<size_t>(l - 1)][k].transpose() * ws;
                        const Eigen::VectorXd contrib = cache_[gi].sub_Z[static_cast<size_t>(l - 1)][k] * bk;
                        for (size_t a = 0; a < sr[k].size(); ++a) re_fit[sr[k][a]] += contrib[static_cast<Eigen::Index>(a)];
                    }
                    inner[static_cast<size_t>(l - 1)].push_back(bk);
                }
            }
            res.fitted_pop.segment(row, n) = pop;
            res.fitted_cond.segment(row, n) = pop + re_fit;
            res.resid.segment(row, n) = g.y - pop - re_fit;
            for (int i = 0; i < n; ++i) res.order[static_cast<size_t>(row + i)] = g.rows.empty() ? row + i : g.rows[static_cast<size_t>(i)];
            res.blup.push_back(b0);
            res.blup_inner.push_back(std::move(inner));
            res.labels.push_back(g.label);
        }
    }

    std::vector<LmeGroup> g_;                   ///< Grouped data.
    LmeSpec spec_;                              ///< Model specification.
    std::vector<Cache> cache_;                  ///< Per-group precomputed structure.
    std::vector<int> q_, re_off_, n_re_l_;      ///< Random-effect dimensions and offsets per level.
    std::vector<PdSpec> kinds_;                 ///< Resolved covariance parameterizations.
    int p_ = 0, N_ = 0, nlev_ = 1, cor_off_ = 0, n_cor_ = 0, var_off_ = 0, n_var_ = 0, n_theta_ = 0;  ///< Sizes and offsets.
    double logdet_xtx_ = 0.0;                   ///< log|X^T X| for the REML constant.
    std::vector<int> fixed_idx_, free_idx_;     ///< Fixed and free theta indices.
    Eigen::VectorXd fixed_vals_;                ///< Fixed theta values.
};

}  // namespace mixedfit
