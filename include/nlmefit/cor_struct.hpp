// SPDX-License-Identifier: Apache-2.0
//
// Within-group correlation structures, implemented from the underlying theory:
//
//   * `AR1`:  R_ij = phi^|lag|, with phi = tanh(theta) and integer positions.
//   * `CAR1`: R_ij = phi^|t_i - t_j|, with phi in (0, 1) and continuous times.
//   * `CS`:   compound symmetry, with rho in (-1/(n-1), 1).
//   * `ARMA(p,q)`: exact autocorrelation function of a stationary and
//                  invertible ARMA process.
//
// For ARMA, the coefficients are obtained from unconstrained partial
// autocorrelations `z -> tanh(z)` through the Durbin-Levinson recursion
// (Jones 1980; Monahan 1984). This guarantees stationarity of the AR part and
// invertibility of the MA part for every theta. The autocorrelation function is
// computed by solving the generalized Yule-Walker equations
// (Brockwell & Davis, "Time Series: Theory and Methods", section 3.3).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pd_param.hpp"

namespace mixedfit {

/// Built-in correlation structures.
///
/// The last five entries are spatial correlation functions, corresponding to
/// `corExp`, `corGaus`, `corLin`, `corRatio` and `corSpher` in R/nlme.
enum class CorKind {
    Identity,
    AR1,
    CAR1,
    CompoundSymmetry,
    ARMA,
    Symm,
    Exp,
    Gaus,
    Lin,
    Ratio,
    Spher,
    Custom
};

/// Distance metric used by the spatial correlation functions.
enum class SpatialMetric { Euclidean, Maximum, Manhattan };

/// User-defined correlation structure.
///
/// Equivalent to a new `corStruct` class in R. `matrix` receives the
/// unconstrained parameters, the positions/times `t`, and the coordinates
/// (empty when there are none) of one block of observations. It must return
/// the block correlation matrix (symmetric, unit diagonal, positive definite).
/// The analytic gradient of the model uses five-point finite differences over
/// this callback unless `deriv` is supplied.
struct CustomCor {
    std::string name = "custom";
    int n_params = 0;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& th, const Eigen::VectorXd& t,
                                 const Eigen::MatrixXd& coords)> matrix;
    Eigen::VectorXd start;   ///< Empty -> zeros.
    /// Optional `d(correlation matrix)/d theta_j` of one block (same signature
    /// as `matrix` plus the index `j`). Empty -> finite differences.
    std::function<Eigen::MatrixXd(const Eigen::VectorXd& th, const Eigen::VectorXd& t,
                                 const Eigen::MatrixXd& coords, int j)> deriv;
};

/// Specification of a within-group correlation structure.
struct CorSpec {
    CorKind kind = CorKind::Identity;

    /// Build a user-defined specification.
    ///
    /// @throws std::invalid_argument if `matrix` is missing or `start` has the
    ///         wrong length.
    static CorSpec Custom(CustomCor c) {
        if (!c.matrix) throw std::invalid_argument("CorSpec::Custom: 'matrix' is required");
        if (c.start.size() != 0 && c.start.size() != c.n_params)
            throw std::invalid_argument("CorSpec::Custom: 'start' has the wrong length");
        CorSpec s;
        s.kind = CorKind::Custom;
        s.custom = std::make_shared<const CustomCor>(std::move(c));
        return s;
    }

    int p = 0;                 ///< AR order (ARMA only).
    int q = 0;                 ///< MA order (ARMA only).
    int max_group_size = 2;    ///< Set by the model; defines the lower bound of the CS correlation.
    int n_pos = 0;             ///< `Symm`: number of positions (0..n_pos-1); set by the model.
    bool nugget = false;       ///< Spatial: add a nugget effect rho = (1 - nugget) rho0(d) for d > 0.
    SpatialMetric metric = SpatialMetric::Euclidean;
    double range_start = 1.0;  ///< Spatial: initial range (set by the model to the median distance).
    std::shared_ptr<const CustomCor> custom;   ///< Only for `kind == Custom`; kept last to preserve aggregate initialization.

    /// Whether this is one of the spatial correlation functions.
    bool spatial() const {
        return kind == CorKind::Exp || kind == CorKind::Gaus || kind == CorKind::Lin ||
               kind == CorKind::Ratio || kind == CorKind::Spher;
    }

    /// Number of unconstrained correlation parameters.
    int n_params() const {
        switch (kind) {
            case CorKind::Identity: return 0;
            case CorKind::ARMA:     return p + q;
            case CorKind::Symm:     return n_pos * (n_pos - 1) / 2;
            case CorKind::Custom:   return custom ? custom->n_params : 0;
            case CorKind::Exp: case CorKind::Gaus: case CorKind::Lin:
            case CorKind::Ratio: case CorKind::Spher:
                return nugget ? 2 : 1;
            default:                return 1;
        }
    }
};

/// Convert unconstrained partial autocorrelations to stationary AR
/// coefficients.
///
/// The autoregressive polynomial is `1 - sum_j phi_j z^j`.
inline Eigen::VectorXd pacf_to_ar(const Eigen::VectorXd& z) {
    const int p = static_cast<int>(z.size());
    Eigen::VectorXd a = Eigen::VectorXd::Zero(p), prev(p);
    for (int k = 1; k <= p; ++k) {
        const double r = std::tanh(z[k - 1]);
        prev = a;
        for (int j = 1; j < k; ++j) a[j - 1] = prev[j - 1] - r * prev[k - j - 1];
        a[k - 1] = r;
    }
    return a;
}

/// Build a general `n x n` correlation matrix from unconstrained partial
/// correlations `z` (`n(n-1)/2` values, row-wise).
///
/// The construction uses a Cholesky factor with unit-diagonal rows
/// (Lewandowski, Kurowicka & Joe 2009) and is positive definite by design.
inline Eigen::MatrixXd corr_from_pcor(int n, const Eigen::VectorXd& z) {
    Eigen::MatrixXd Lc = Eigen::MatrixXd::Zero(n, n);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        double rem = 1.0;
        for (int j = 0; j < i; ++j) {
            Lc(i, j) = std::tanh(z[k++]) * std::sqrt(rem);
            rem -= Lc(i, j) * Lc(i, j);
        }
        Lc(i, i) = std::sqrt(std::max(rem, 0.0));
    }
    return Lc * Lc.transpose();
}

/// Theoretical autocorrelation `rho(0..maxlag)` of an ARMA(p,q) process:
///
///     phi(B) x_t = theta(B) e_t,
///     theta(B) = 1 + theta_1 B + ... + theta_q B^q.
inline Eigen::VectorXd arma_acf(const Eigen::VectorXd& phi, const Eigen::VectorXd& theta, int maxlag) {
    const int p = static_cast<int>(phi.size());
    const int q = static_cast<int>(theta.size());
    const int L = std::max(maxlag, p);

    // psi weights psi_0..psi_q of the MA(infinity) representation.
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(q + 1);
    psi[0] = 1.0;
    for (int j = 1; j <= q; ++j) {
        double s = theta[j - 1];
        for (int i = 1; i <= std::min(j, p); ++i) s += phi[i - 1] * psi[j - i];
        psi[j] = s;
    }

    auto th = [&](int j) { return j == 0 ? 1.0 : theta[j - 1]; };

    // Right-hand side sum_{j=k}^{q} theta_j psi_{j-k} (0 when k > q).
    auto rhs = [&](int k) {
        double s = 0.0;
        for (int j = k; j <= q; ++j) s += th(j) * psi[j - k];
        return s;
    };

    // Generalized Yule-Walker equations for k = 0..p, giving gamma(0..p):
    //   gamma(k) - sum_j phi_j gamma(|k-j|) = rhs(k).
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(p + 1, p + 1);
    Eigen::VectorXd b(p + 1);
    for (int k = 0; k <= p; ++k) {
        b[k] = rhs(k);
        for (int j = 1; j <= p; ++j) A(k, std::abs(k - j)) -= phi[j - 1];
    }

    Eigen::VectorXd g = Eigen::VectorXd::Zero(L + 1);
    g.head(p + 1) = A.colPivHouseholderQr().solve(b);
    for (int k = p + 1; k <= L; ++k) {
        double s = rhs(k);
        for (int j = 1; j <= p; ++j) s += phi[j - 1] * g[k - j];
        g[k] = s;
    }
    return (g / g[0]).head(maxlag + 1);
}

/// Distance between observations `i` and `j` of a group.
///
/// The distance is computed from the coordinate rows of `P`. When `P` has no
/// columns, the absolute difference of the times is used instead.
inline double spatial_dist(SpatialMetric m, const Eigen::MatrixXd& P, const Eigen::VectorXd& t, int i, int j) {
    if (P.cols() == 0) return std::abs(t[i] - t[j]);

    const Eigen::VectorXd d = (P.row(i) - P.row(j)).transpose().cwiseAbs();
    switch (m) {
        case SpatialMetric::Maximum:   return d.maxCoeff();
        case SpatialMetric::Manhattan: return d.sum();
        default:                       return d.norm();
    }
}

/// Base spatial correlation function `rho0(r)` for `r = distance / range`.
inline double spatial_rho0(CorKind k, double r) {
    switch (k) {
        case CorKind::Exp:   return std::exp(-r);
        case CorKind::Gaus:  return std::exp(-r * r);
        case CorKind::Lin:   return r < 1.0 ? 1.0 - r : 0.0;
        case CorKind::Ratio: return 1.0 - r * r / (1.0 + r * r);
        case CorKind::Spher: return r < 1.0 ? 1.0 - 1.5 * r + 0.5 * r * r * r : 0.0;
        default:             return 0.0;
    }
}

/// Derivative `d rho0 / d r` of the spatial correlation functions.
inline double spatial_drho0(CorKind k, double r) {
    switch (k) {
        case CorKind::Exp:   return -std::exp(-r);
        case CorKind::Gaus:  return -2.0 * r * std::exp(-r * r);
        case CorKind::Lin:   return r < 1.0 ? -1.0 : 0.0;
        case CorKind::Ratio: { const double q = 1.0 + r * r; return -2.0 * r / (q * q); }
        case CorKind::Spher: return r < 1.0 ? -1.5 + 1.5 * r * r : 0.0;
        default:             return 0.0;
    }
}

/// Analytic derivative of the spatial correlation matrix with respect to
/// parameter `j`.
///
/// Parameter 0 is `log(range)`; parameter 1, when present, is `logit(nugget)`.
inline Eigen::MatrixXd spatial_cor_deriv(const CorSpec& s, const Eigen::VectorXd& th,
                                         const Eigen::VectorXd& t, const Eigen::MatrixXd& coords, int j) {
    const int n = static_cast<int>(t.size());
    Eigen::MatrixXd dR = Eigen::MatrixXd::Zero(n, n);
    const double range = std::exp(th[0]);
    const double nug = s.nugget ? detail::logistic(th[1]) : 0.0;

    for (int a = 0; a < n; ++a)
        for (int b = 0; b < a; ++b) {
            const double r = spatial_dist(s.metric, coords, t, a, b) / range;
            const double v = (j == 0)
                                 ? (1.0 - nug) * spatial_drho0(s.kind, r) * (-r)
                                 : -nug * (1.0 - nug) * spatial_rho0(s.kind, r);
            dR(a, b) = dR(b, a) = v;
        }
    return dR;
}

/// Correlation matrix of one group.
///
/// @param s      Correlation specification.
/// @param th     Unconstrained correlation parameters.
/// @param t      Positions or times of the observations.
/// @param coords Optional coordinates; when non-empty, distances are computed
///               from these rows instead of from `t`.
/// @throws std::invalid_argument if the parameter length, the coordinate
///         dimensions or the positions are invalid.
/// @throws std::runtime_error if a user-defined matrix contains non-finite
///         values.
inline Eigen::MatrixXd cor_matrix(const CorSpec& s, const Eigen::VectorXd& th, const Eigen::VectorXd& t,
                                  const Eigen::MatrixXd& coords) {
    if (th.size() != s.n_params())
        throw std::invalid_argument("cor: incorrect theta length");

    const int n = static_cast<int>(t.size());
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(n, n);

    if (s.spatial()) {
        const double range = std::exp(th[0]);
        const double nug = s.nugget ? detail::logistic(th[1]) : 0.0;
        if (coords.cols() > 0 && coords.rows() != n)
            throw std::invalid_argument("cor: 'coords' must have one row per observation");

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j) {
                const double d = spatial_dist(s.metric, coords, t, i, j);
                R(i, j) = R(j, i) = (1.0 - nug) * spatial_rho0(s.kind, d / range);
            }
        return R;
    }

    switch (s.kind) {
        case CorKind::Identity:
            return R;

        case CorKind::AR1: {
            const double phi = std::tanh(th[0]);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < i; ++j)
                    R(i, j) = R(j, i) = std::pow(phi, std::round(std::abs(t[i] - t[j])));
            return R;
        }

        case CorKind::CAR1: {
            const double phi = detail::logistic(th[0]);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < i; ++j)
                    R(i, j) = R(j, i) = std::pow(phi, std::abs(t[i] - t[j]));
            return R;
        }

        case CorKind::CompoundSymmetry: {
            const double lo = -1.0 / (std::max(s.max_group_size, 2) - 1.0);
            const double rho = lo + (1.0 - lo) * detail::logistic(th[0]);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < i; ++j) R(i, j) = R(j, i) = rho;
            return R;
        }

        case CorKind::Exp: case CorKind::Gaus: case CorKind::Lin:
        case CorKind::Ratio: case CorKind::Spher:
            break;   // handled above

        case CorKind::Symm: {
            // Observations occupy distinct integer positions 0..n_pos-1.
            // The block matrix is the corresponding submatrix of the full
            // correlation matrix.
            const Eigen::MatrixXd Cf = corr_from_pcor(s.n_pos, th);
            std::vector<int> pos(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                pos[static_cast<size_t>(i)] = static_cast<int>(std::lround(t[i]));
                if (pos[static_cast<size_t>(i)] < 0 || pos[static_cast<size_t>(i)] >= s.n_pos)
                    throw std::invalid_argument("cor: position out of range");
                for (int j = 0; j < i; ++j)
                    if (pos[static_cast<size_t>(j)] == pos[static_cast<size_t>(i)])
                        throw std::invalid_argument("cor: repeated positions within a group");
            }
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) R(i, j) = Cf(pos[static_cast<size_t>(i)], pos[static_cast<size_t>(j)]);
            return R;
        }

        case CorKind::Custom: {
            if (!s.custom) throw std::invalid_argument("cor: custom correlation has no definition");
            Eigen::MatrixXd C = s.custom->matrix(th, t, coords);
            if (C.rows() != n || C.cols() != n)
                throw std::invalid_argument("cor custom: matrix must be n x n");
            if (!C.allFinite())
                throw std::runtime_error("cor custom: matrix contains non-finite values");
            return 0.5 * (C + C.transpose());
        }

        case CorKind::ARMA: {
            if (s.p + s.q == 0) throw std::invalid_argument("cor: ARMA(0,0) is not valid");
            const Eigen::VectorXd ar = pacf_to_ar(th.head(s.p));
            const Eigen::VectorXd ma = -pacf_to_ar(th.tail(s.q));

            int maxlag = 0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < i; ++j)
                    maxlag = std::max(maxlag, static_cast<int>(std::lround(std::abs(t[i] - t[j]))));

            const Eigen::VectorXd rho = arma_acf(ar, ma, maxlag);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < i; ++j)
                    R(i, j) = R(j, i) = rho[static_cast<int>(std::lround(std::abs(t[i] - t[j])))];
            return R;
        }
    }
    throw std::logic_error("cor: unknown correlation structure");
}

/// Convenience overload without coordinates.
inline Eigen::MatrixXd cor_matrix(const CorSpec& s, const Eigen::VectorXd& th, const Eigen::VectorXd& t) {
    return cor_matrix(s, th, t, Eigen::MatrixXd());
}

/// Block-diagonal correlation matrix for an inner level.
///
/// Observations are partitioned into blocks (`blocks[b]` contains the row
/// indices of block `b`); the correlation matrix is block diagonal, so
/// observations in different blocks are uncorrelated. This matches the
/// correlation of `nlme` with `form = ~ t | g1/g2`. An empty `blocks` vector is
/// equivalent to a single block containing every row.
inline Eigen::MatrixXd cor_matrix_blocks(const CorSpec& s, const Eigen::VectorXd& th, const Eigen::VectorXd& t,
                                         const Eigen::MatrixXd& coords,
                                         const std::vector<std::vector<int>>& blocks) {
    if (blocks.empty()) return cor_matrix(s, th, t, coords);

    const int n = static_cast<int>(t.size());
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

    for (const auto& idx : blocks) {
        const int m = static_cast<int>(idx.size());
        Eigen::VectorXd tb(m);
        Eigen::MatrixXd cb;
        if (coords.cols() > 0) cb.resize(m, coords.cols());
        for (int k = 0; k < m; ++k) {
            tb[k] = t[idx[static_cast<size_t>(k)]];
            if (coords.cols() > 0) cb.row(k) = coords.row(idx[static_cast<size_t>(k)]);
        }
        const Eigen::MatrixXd Rb = cor_matrix(s, th, tb, cb);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) R(idx[static_cast<size_t>(i)], idx[static_cast<size_t>(j)]) = Rb(i, j);
    }
    return R;
}

/// Block-wise version of `spatial_cor_deriv`.
inline Eigen::MatrixXd spatial_cor_deriv_blocks(const CorSpec& s, const Eigen::VectorXd& th, const Eigen::VectorXd& t,
                                                const Eigen::MatrixXd& coords, int j,
                                                const std::vector<std::vector<int>>& blocks) {
    if (blocks.empty()) return spatial_cor_deriv(s, th, t, coords, j);

    const int n = static_cast<int>(t.size());
    Eigen::MatrixXd dR = Eigen::MatrixXd::Zero(n, n);

    for (const auto& idx : blocks) {
        const int m = static_cast<int>(idx.size());
        Eigen::VectorXd tb(m);
        Eigen::MatrixXd cb;
        if (coords.cols() > 0) cb.resize(m, coords.cols());
        for (int k = 0; k < m; ++k) {
            tb[k] = t[idx[static_cast<size_t>(k)]];
            if (coords.cols() > 0) cb.row(k) = coords.row(idx[static_cast<size_t>(k)]);
        }
        const Eigen::MatrixXd Db = spatial_cor_deriv(s, th, tb, cb, j);
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) dR(idx[static_cast<size_t>(a)], idx[static_cast<size_t>(b)]) = Db(a, b);
    }
    return dR;
}

/// Analytic derivative of a user-defined correlation structure
/// (`CustomCor::deriv`), computed block by block.
///
/// @throws std::invalid_argument if no analytic derivative is available or the
///         derivative has the wrong dimensions.
inline Eigen::MatrixXd custom_cor_deriv_blocks(const CorSpec& s, const Eigen::VectorXd& th,
                                               const Eigen::VectorXd& t, const Eigen::MatrixXd& coords, int j,
                                               const std::vector<std::vector<int>>& blocks) {
    if (!s.custom || !s.custom->deriv)
        throw std::invalid_argument("cor custom: analytic derivative is not available");

    const int n = static_cast<int>(t.size());
    auto one = [&](const Eigen::VectorXd& tb, const Eigen::MatrixXd& cb) {
        Eigen::MatrixXd D = s.custom->deriv(th, tb, cb, j);
        if (D.rows() != tb.size() || D.cols() != tb.size())
            throw std::invalid_argument("cor custom: derivative must be n x n");
        return Eigen::MatrixXd(0.5 * (D + D.transpose()));
    };

    if (blocks.empty()) return one(t, coords);

    Eigen::MatrixXd dR = Eigen::MatrixXd::Zero(n, n);
    for (const auto& idx : blocks) {
        const int m = static_cast<int>(idx.size());
        Eigen::VectorXd tb(m);
        Eigen::MatrixXd cb;
        if (coords.cols() > 0) cb.resize(m, coords.cols());
        for (int k = 0; k < m; ++k) {
            tb[k] = t[idx[static_cast<size_t>(k)]];
            if (coords.cols() > 0) cb.row(k) = coords.row(idx[static_cast<size_t>(k)]);
        }
        const Eigen::MatrixXd Db = one(tb, cb);
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) dR(idx[static_cast<size_t>(a)], idx[static_cast<size_t>(b)]) = Db(a, b);
    }
    return dR;
}

/// Group row indices `0..n-1` by the identifier in `sub`.
///
/// The groups are returned in order of first appearance. An empty result means
/// that there is a single block, so callers can skip the block machinery.
inline std::vector<std::vector<int>> make_cor_blocks(const std::vector<int>& sub) {
    std::vector<std::vector<int>> out;
    if (sub.empty()) return out;

    std::map<int, int> pos;
    for (int i = 0; i < static_cast<int>(sub.size()); ++i) {
        auto it = pos.find(sub[static_cast<size_t>(i)]);
        if (it == pos.end()) {
            it = pos.emplace(sub[static_cast<size_t>(i)], static_cast<int>(out.size())).first;
            out.emplace_back();
        }
        out[static_cast<size_t>(it->second)].push_back(i);
    }
    if (out.size() == 1) out.clear();
    return out;
}

}  // namespace mixedfit
