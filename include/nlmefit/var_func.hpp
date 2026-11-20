// SPDX-License-Identifier: Apache-2.0
//
// Within-group variance functions.
//
// For observation `j` the residual standard deviation is
//
//     sd_j = sigma * s_j,
//     Cov(e_i) = sigma^2 * S_i R_i S_i,   S_i = diag(s).
//
// The relative scale `s_j = g(v_j, stratum_j; delta)` is one of:
//
//   * `Fixed`:      s = sqrt(v)                     (no parameters; variance proportional to v).
//   * `Ident`:      s = 1 for stratum 0, exp(theta_k) for strata k > 0.
//   * `Power`:      s = |v|^delta                   (one delta per stratum, optional).
//   * `Exp`:        s = exp(delta * v).
//   * `ConstPower`: s = exp(c) + |v|^delta          (c and delta per stratum, optional).
//   * `ConstProp`:  s = sqrt(a^2 + b^2 v^2), with a = exp(theta_a), b = exp(theta_b)
//                   (additive plus proportional error).
//   * `Comb`:       product of the scales of its components (`varComb`).
//
// Parameters are unconstrained (Pinheiro & Bates 2000, section 5.2).
//
// Each component reads its covariate and stratum factor from a "slot"
// (`cov_slot`, `strata_slot`) of `VarData`. This lets a `varComb` combine, for
// example, `varIdent(~1|Sex)` (strata, slot 0) with `varPower(~age)`
// (covariate, slot 0) or with a second covariate (slot 1). The flag
// `fitted = true` means that the covariate is `fitted(.)`: the current fitted
// values, which the model updates iteratively.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mixedfit {

/// Built-in within-group variance functions.
enum class VarKind { None, Fixed, Ident, Power, Exp, ConstPower, ConstProp, Comb, Custom };

/// Per-group variance data: stratum factors and covariates, indexed by slot.
struct VarData {
    std::vector<std::vector<int>> strata;   ///< Stratum labels, one vector per slot.
    std::vector<Eigen::VectorXd> cov;       ///< Covariates, one vector per slot.
};

/// User-defined variance function.
///
/// Equivalent to a new `varFunc` class in R. `scales` returns the `n` relative
/// standard deviations `s_j > 0` of one group from the unconstrained
/// parameters, the covariate (empty when `needs_covariate = false`), the
/// strata (empty when there are none) and `n`. The gradient uses five-point
/// finite differences unless `dscales` is supplied.
struct CustomVar {
    std::string name = "custom";
    int n_params = 0;
    bool needs_covariate = false;
    std::function<Eigen::VectorXd(const Eigen::VectorXd& th, const Eigen::VectorXd& cov,
                                 const std::vector<int>& strata, int n)> scales;
    Eigen::VectorXd start;   ///< Empty -> zeros.
    /// Optional `d s / d theta_j` (`n` values, same signature as `scales` plus
    /// the index `j`). Empty -> finite differences.
    std::function<Eigen::VectorXd(const Eigen::VectorXd& th, const Eigen::VectorXd& cov,
                                 const std::vector<int>& strata, int n, int j)> dscales;
};

/// Specification of the within-group variance structure.
struct VarSpec {
    VarKind kind = VarKind::None;

    /// Build a user-defined specification.
    ///
    /// @throws std::invalid_argument if `scales` is missing or `start` has the
    ///         wrong length.
    static VarSpec Custom(CustomVar c) {
        if (!c.scales) throw std::invalid_argument("VarSpec::Custom: 'scales' is required");
        if (c.start.size() != 0 && c.start.size() != c.n_params)
            throw std::invalid_argument("VarSpec::Custom: 'start' has the wrong length");
        VarSpec s;
        s.kind = VarKind::Custom;
        s.custom = std::make_shared<const CustomVar>(std::move(c));
        return s;
    }

    int n_strata = 1;           ///< Set by the model from the strata that are present.
    bool by_stratum = false;    ///< Power/Exp/ConstPower/ConstProp: separate parameters per stratum.
    bool fitted = false;        ///< The covariate is `fitted(.)`.
    int strata_slot = 0;        ///< Slot in `VarData::strata`.
    int cov_slot = 0;           ///< Slot in `VarData::cov`.
    std::vector<VarSpec> comb;  ///< Components of `Comb`.
    std::shared_ptr<const CustomVar> custom;  ///< Only for `kind == Custom`; kept last to preserve aggregate initialization.

    /// Build a combined (`varComb`) specification.
    static VarSpec Combine(std::vector<VarSpec> parts) {
        VarSpec s;
        s.kind = VarKind::Comb;
        s.comb = std::move(parts);
        return s;
    }

    /// Number of parameter sets (one per stratum when `by_stratum`, otherwise one).
    int n_per() const { return by_stratum ? n_strata : 1; }

    /// Number of unconstrained parameters.
    int n_params() const {
        switch (kind) {
            case VarKind::None:
            case VarKind::Fixed:      return 0;
            case VarKind::Custom:     return custom ? custom->n_params : 0;
            case VarKind::Ident:      return n_strata - 1;
            case VarKind::Power:
            case VarKind::Exp:        return n_per();
            case VarKind::ConstPower:
            case VarKind::ConstProp:  return 2 * n_per();
            case VarKind::Comb: {
                int n = 0;
                for (const auto& c : comb) n += c.n_params();
                return n;
            }
        }
        return 0;
    }

    /// Whether at least one component needs a covariate.
    bool needs_covariate() const {
        if (kind == VarKind::Comb) {
            for (const auto& c : comb) if (c.needs_covariate()) return true;
            return false;
        }
        if (kind == VarKind::Custom) return custom && custom->needs_covariate;
        return kind == VarKind::Fixed || kind == VarKind::Power || kind == VarKind::Exp ||
               kind == VarKind::ConstPower || kind == VarKind::ConstProp;
    }

    /// Whether at least one component uses `fitted(.)` as its covariate.
    bool has_fitted() const {
        if (kind == VarKind::Comb) {
            for (const auto& c : comb) if (c.has_fitted()) return true;
            return false;
        }
        return fitted && needs_covariate();
    }

    /// Whether this is the identity (no variance function) specification.
    bool is_none() const { return kind == VarKind::None; }
};

/// Flatten a `VarSpec` into its simple components, recording the offset of the
/// parameters of each component inside `theta_var`.
inline void var_components(const VarSpec& s, int base, std::vector<std::pair<const VarSpec*, int>>& out) {
    if (s.kind == VarKind::Comb) {
        int off = base;
        for (const auto& c : s.comb) {
            var_components(c, off, out);
            off += c.n_params();
        }
    } else if (s.kind != VarKind::None) {
        out.push_back({&s, base});
    }
}

/// Mutable variant used by the model to set `n_strata` on each component.
inline void var_components_mut(VarSpec& s, std::vector<VarSpec*>& out) {
    if (s.kind == VarKind::Comb) {
        for (auto& c : s.comb) var_components_mut(c, out);
    } else if (s.kind != VarKind::None) {
        out.push_back(&s);
    }
}

namespace detail {

/// Relative scales of one simple (non-combined) component.
///
/// @throws std::invalid_argument if the required covariate is missing or a
///         user-defined function returns the wrong length.
inline Eigen::VectorXd scales_simple(const VarSpec& s, const Eigen::VectorXd& th, const VarData& vd, int n) {
    Eigen::VectorXd sc = Eigen::VectorXd::Ones(n);
    if (s.kind == VarKind::None) return sc;

    const std::vector<int>* strata =
        (s.strata_slot < static_cast<int>(vd.strata.size())) ? &vd.strata[static_cast<size_t>(s.strata_slot)] : nullptr;
    const Eigen::VectorXd* v =
        (s.cov_slot < static_cast<int>(vd.cov.size())) ? &vd.cov[static_cast<size_t>(s.cov_slot)] : nullptr;

    if (s.needs_covariate() && (!v || v->size() != n))
        throw std::invalid_argument("var: the variance function is missing its covariate");

    if (s.kind == VarKind::Custom) {
        if (!s.custom) throw std::invalid_argument("var: custom variance function has no definition");
        const Eigen::VectorXd sc2 = s.custom->scales(
            th, v ? *v : Eigen::VectorXd(), strata ? *strata : std::vector<int>(), n);
        if (sc2.size() != n)
            throw std::invalid_argument("var custom: 'scales' must return n values");
        return sc2;
    }

    auto st = [&](int i) { return (strata && !strata->empty()) ? (*strata)[static_cast<size_t>(i)] : 0; };
    auto k = [&](int i) { return s.by_stratum ? st(i) : 0; };
    const int np = s.n_per();

    for (int i = 0; i < n; ++i) {
        switch (s.kind) {
            case VarKind::None:
            case VarKind::Custom:
            case VarKind::Comb: break;
            case VarKind::Fixed:      sc[i] = std::sqrt((*v)[i]); break;
            case VarKind::Ident:      sc[i] = st(i) == 0 ? 1.0 : std::exp(th[st(i) - 1]); break;
            case VarKind::Power:      sc[i] = std::pow(std::abs((*v)[i]), th[k(i)]); break;
            case VarKind::Exp:        sc[i] = std::exp(th[k(i)] * (*v)[i]); break;
            case VarKind::ConstPower: sc[i] = std::exp(th[k(i)]) + std::pow(std::abs((*v)[i]), th[np + k(i)]); break;
            case VarKind::ConstProp: {
                const double a = std::exp(th[k(i)]), b = std::exp(th[np + k(i)]);
                sc[i] = std::sqrt(a * a + b * b * (*v)[i] * (*v)[i]);
                break;
            }
        }
    }
    return sc;
}

}  // namespace detail

/// Relative scales `s_j` of a group of `n` observations.
///
/// For `Comb` the scales are the product of the component scales.
inline Eigen::VectorXd var_scales(const VarSpec& s, const Eigen::VectorXd& th, const VarData& vd, int n) {
    if (s.kind != VarKind::Comb) return detail::scales_simple(s, th, vd, n);

    std::vector<std::pair<const VarSpec*, int>> comps;
    var_components(s, 0, comps);

    Eigen::VectorXd sc = Eigen::VectorXd::Ones(n);
    for (const auto& c : comps)
        sc.array() *= detail::scales_simple(*c.first, th.segment(c.second, c.first->n_params()), vd, n).array();
    return sc;
}

/// Analytic derivative of the scales with respect to `theta[j]`.
///
/// This is available only when the component that owns parameter `j` is a
/// `Custom` component with a `dscales` callback. The other components of a
/// `Comb` multiply the derivative.
///
/// @return `true` if an analytic derivative was produced; `false` means the
///         caller must fall back to finite differences.
/// @throws std::invalid_argument if a required covariate is missing or a
///         user-defined callback returns the wrong length.
inline bool var_scales_deriv(const VarSpec& s, const Eigen::VectorXd& th, const VarData& vd, int n, int j,
                             Eigen::VectorXd& out) {
    std::vector<std::pair<const VarSpec*, int>> comps;
    var_components(s, 0, comps);

    int owner = -1;
    for (size_t c = 0; c < comps.size(); ++c)
        if (j >= comps[c].second && j < comps[c].second + comps[c].first->n_params())
            owner = static_cast<int>(c);

    if (owner < 0) return false;

    const VarSpec& oc = *comps[static_cast<size_t>(owner)].first;
    if (oc.kind != VarKind::Custom || !oc.custom || !oc.custom->dscales) return false;

    const std::vector<int>* strata =
        (oc.strata_slot < static_cast<int>(vd.strata.size())) ? &vd.strata[static_cast<size_t>(oc.strata_slot)] : nullptr;
    const Eigen::VectorXd* v =
        (oc.cov_slot < static_cast<int>(vd.cov.size())) ? &vd.cov[static_cast<size_t>(oc.cov_slot)] : nullptr;

    if (oc.needs_covariate() && (!v || v->size() != n))
        throw std::invalid_argument("var: the variance function is missing its covariate");

    out = oc.custom->dscales(th.segment(comps[static_cast<size_t>(owner)].second, oc.n_params()),
                             v ? *v : Eigen::VectorXd(),
                             strata ? *strata : std::vector<int>(),
                             n, j - comps[static_cast<size_t>(owner)].second);
    if (out.size() != n)
        throw std::invalid_argument("var custom: 'dscales' must return n values");

    for (size_t c = 0; c < comps.size(); ++c)
        if (static_cast<int>(c) != owner)
            out.array() *= detail::scales_simple(*comps[c].first,
                                                 th.segment(comps[c].second, comps[c].first->n_params()),
                                                 vd, n).array();
    return true;
}

/// Convenience overload for a single variance function with `(strata, v)` in
/// slot 0.
inline Eigen::VectorXd var_scales(const VarSpec& s, const Eigen::VectorXd& th, const std::vector<int>& strata,
                                  const Eigen::VectorXd& v, int n) {
    VarData vd;
    vd.strata.push_back(strata);
    vd.cov.push_back(v);
    return var_scales(s, th, vd, n);
}

}  // namespace mixedfit
