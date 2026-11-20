// SPDX-License-Identifier: Apache-2.0
//
// Unconstrained parameterizations of positive-definite covariance matrices.
//
// Every parameterization returns a lower-triangular factor `L` such that
// `Sigma = L * L^T`. The packing convention used by all parameter vectors is
// row-wise over the lower triangle:
//
//   (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...
//
// References:
//   * Pinheiro & Bates (1996), "Unconstrained parametrizations for
//     variance-covariance matrices".
//   * Lewandowski, Kurowicka & Joe (2009), for the partial-correlation
//     parameterization of correlation matrices.
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

/// Built-in covariance parameterizations.
enum class PdKind {
    Cholesky,          ///< Direct lower-triangular factor (diagonal unrestricted).
    LogCholesky,       ///< Lower-triangular factor with log-transformed diagonal.
    MatrixLog,         ///< Matrix logarithm: `Sigma = expm(A)` with `A` symmetric.
    Natural,           ///< Log standard deviations plus partial correlations.
    CompoundSymmetry,  ///< Compound symmetry: one variance and one correlation.
    Diag,              ///< Diagonal matrix, parameterized by log standard deviations.
    Ident             ///< Scaled identity: one common log standard deviation.
};

/// Number of unconstrained parameters used by a built-in parameterization.
///
/// @param kind Covariance parameterization.
/// @param q    Dimension of the covariance matrix.
/// @return Number of free parameters.
/// @throws std::invalid_argument if `q < 1`.
inline int pd_nparams(PdKind kind, int q) {
    if (q < 1) throw std::invalid_argument("pd: q must be >= 1");
    if (kind == PdKind::CompoundSymmetry) return q == 1 ? 1 : 2;
    if (kind == PdKind::Diag) return q;      // log standard deviations (pdDiag)
    if (kind == PdKind::Ident) return 1;     // common log standard deviation (pdIdent)
    return q * (q + 1) / 2;
}

namespace detail {

/// Unpack a lower-triangular matrix from a row-wise packed vector.
inline Eigen::MatrixXd unpack_lower(int q, const Eigen::VectorXd& t) {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(q, q);
    int k = 0;
    for (int i = 0; i < q; ++i)
        for (int j = 0; j <= i; ++j) M(i, j) = t[k++];
    return M;
}

/// Cholesky factor of a positive-definite matrix.
inline Eigen::MatrixXd cholesky_lower(const Eigen::MatrixXd& S) {
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("pd: matrix is not positive definite");
    return llt.matrixL();
}

/// Numerically stable logistic function.
inline double logistic(double t) { return 1.0 / (1.0 + std::exp(-t)); }

}  // namespace detail

/// Build the lower-triangular factor `L` with `Sigma = L L^T`.
///
/// @param kind  Covariance parameterization.
/// @param q     Dimension of the covariance matrix.
/// @param theta Unconstrained parameter vector.
/// @return Lower-triangular factor.
/// @throws std::invalid_argument if the parameter vector has the wrong length.
/// @throws std::runtime_error if a numerical decomposition fails.
inline Eigen::MatrixXd pd_factor(PdKind kind, int q, const Eigen::VectorXd& theta) {
    if (theta.size() != pd_nparams(kind, q))
        throw std::invalid_argument("pd: incorrect theta length");

    switch (kind) {
        case PdKind::Cholesky:
            return detail::unpack_lower(q, theta);

        case PdKind::LogCholesky: {
            Eigen::MatrixXd L = detail::unpack_lower(q, theta);
            for (int i = 0; i < q; ++i) L(i, i) = std::exp(L(i, i));
            return L;
        }

        case PdKind::MatrixLog: {
            // Sigma = expm(A), where A is symmetric and the lower triangle of A
            // is given by theta.
            Eigen::MatrixXd A = detail::unpack_lower(q, theta);
            A = A.selfadjointView<Eigen::Lower>();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
            if (es.info() != Eigen::Success)
                throw std::runtime_error("pd: eigenvalue decomposition did not converge");
            Eigen::MatrixXd S = es.eigenvectors() *
                                es.eigenvalues().array().exp().matrix().asDiagonal() *
                                es.eigenvectors().transpose();
            return detail::cholesky_lower(S);
        }

        case PdKind::Diag: {
            Eigen::MatrixXd L = Eigen::MatrixXd::Zero(q, q);
            for (int i = 0; i < q; ++i) L(i, i) = std::exp(theta[i]);
            return L;
        }

        case PdKind::Ident:
            return std::exp(theta[0]) * Eigen::MatrixXd::Identity(q, q);

        case PdKind::Natural: {
            // theta[0..q-1] are log standard deviations. The remaining entries
            // are z = atanh(partial correlation), stored row-wise. This always
            // produces a valid (positive-definite) correlation matrix.
            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(q, q);
            int k = q;
            for (int i = 0; i < q; ++i) {
                double rem = 1.0;
                for (int j = 0; j < i; ++j) {
                    C(i, j) = std::tanh(theta[k++]) * std::sqrt(rem);
                    rem -= C(i, j) * C(i, j);
                }
                C(i, i) = std::sqrt(std::max(rem, 0.0));
            }
            for (int i = 0; i < q; ++i) C.row(i) *= std::exp(theta[i]);
            return C;
        }

        case PdKind::CompoundSymmetry: {
            // Sigma = s^2 [(1 - r) I + r 11^T], with r in (-1/(q-1), 1).
            const double s = std::exp(theta[0]);
            double r = 0.0;
            if (q > 1) {
                const double lo = -1.0 / (q - 1.0);
                r = lo + (1.0 - lo) * detail::logistic(theta[1]);
            }
            Eigen::MatrixXd S = Eigen::MatrixXd::Constant(q, q, r);
            S.diagonal().setOnes();
            return std::abs(s) * detail::cholesky_lower(S);
        }
    }
    throw std::logic_error("pd: unknown covariance parameterization");
}

/// Build the covariance matrix `Sigma = L L^T`.
inline Eigen::MatrixXd pd_sigma(PdKind kind, int q, const Eigen::VectorXd& theta) {
    const Eigen::MatrixXd L = pd_factor(kind, q, theta);
    return L * L.transpose();
}

/// User-defined covariance parameterization.
///
/// This is the analogue of a new `pdMat` class in R. The user supplies:
///   * `nparams`: number of unconstrained parameters for dimension `q`;
///   * `factor`:   lower-triangular factor `L(theta)` with `D = L L^T`
///                 (which must be positive definite for every `theta`);
///   * optionally `start`: an initial `theta`;
///   * optionally `from_D`: an approximate inverse map `D -> theta`, used only
///     to initialize the EM step.
struct CustomPd {
    std::string name = "custom";
    std::function<int(int q)> nparams;
    std::function<Eigen::MatrixXd(int q, const Eigen::VectorXd& theta)> factor;
    std::function<Eigen::VectorXd(int q)> start;                            ///< Empty -> zeros.
    std::function<Eigen::VectorXd(int q, const Eigen::MatrixXd& D)> from_D; ///< Empty -> EM leaves this level unchanged.
    /// Optional analytic derivative of `L` with respect to `theta[j]` (`q x q`).
    /// Empty -> five-point finite differences on `factor`.
    std::function<Eigen::MatrixXd(int q, const Eigen::VectorXd& theta, int j)> dfactor;
};

/// Complete specification of the covariance of one grouping level.
///
/// It can be a simple built-in parameterization, a block-diagonal
/// (`pdBlocked`) specification, or a user-defined parameterization.
struct PdSpec {
    PdKind kind = PdKind::LogCholesky;
    std::shared_ptr<const CustomPd> custom;   ///< If non-null, overrides `kind` and the block fields.
    std::vector<int> block_sizes;             ///< Only for `pdBlocked`: block sizes (must sum to `q`).
    std::vector<PdKind> block_kinds;          ///< Only for `pdBlocked`: parameterization of each block.

    PdSpec() = default;
    PdSpec(PdKind k) : kind(k) {}             ///< Implicit conversion from `PdKind`.

    /// Build a block-diagonal specification.
    static PdSpec Blocked(std::vector<int> sizes, std::vector<PdKind> kinds) {
        PdSpec s;
        s.block_sizes = std::move(sizes);
        s.block_kinds = std::move(kinds);
        return s;
    }

    /// Build a user-defined specification.
    ///
    /// @throws std::invalid_argument if `nparams` or `factor` is missing.
    static PdSpec Custom(CustomPd c) {
        if (!c.nparams || !c.factor)
            throw std::invalid_argument("PdSpec::Custom: 'nparams' and 'factor' are required");
        PdSpec s;
        s.custom = std::make_shared<const CustomPd>(std::move(c));
        return s;
    }

    /// Whether this is a non-custom block-diagonal specification.
    bool blocked() const { return !custom && !block_sizes.empty(); }
};

/// Number of unconstrained parameters for a full specification.
///
/// @throws std::invalid_argument if the block specification is inconsistent.
inline int pd_nparams(const PdSpec& s, int q) {
    if (s.custom) {
        if (q < 1) throw std::invalid_argument("pd: q must be >= 1");
        return s.custom->nparams(q);
    }
    if (!s.blocked()) return pd_nparams(s.kind, q);

    int tot = 0, n = 0;
    if (s.block_kinds.size() != s.block_sizes.size())
        throw std::invalid_argument("pdBlocked: block sizes and block kinds have different lengths");
    for (size_t b = 0; b < s.block_sizes.size(); ++b) {
        tot += s.block_sizes[b];
        n += pd_nparams(s.block_kinds[b], s.block_sizes[b]);
    }
    if (tot != q) throw std::invalid_argument("pdBlocked: block sizes do not sum to q");
    return n;
}

/// Build the lower-triangular factor for a full specification.
inline Eigen::MatrixXd pd_factor(const PdSpec& s, int q, const Eigen::VectorXd& theta) {
    if (s.custom) {
        if (theta.size() != pd_nparams(s, q))
            throw std::invalid_argument("pd: incorrect theta length");
        Eigen::MatrixXd L = s.custom->factor(q, theta);
        if (L.rows() != q || L.cols() != q)
            throw std::invalid_argument("pd custom: factor must be q x q");
        if (!L.allFinite())
            throw std::runtime_error("pd custom: factor contains non-finite values");
        return L;
    }

    if (!s.blocked()) return pd_factor(s.kind, q, theta);

    if (theta.size() != pd_nparams(s, q))
        throw std::invalid_argument("pd: incorrect theta length");

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(q, q);
    int off = 0, pos = 0;
    for (size_t b = 0; b < s.block_sizes.size(); ++b) {
        const int sz = s.block_sizes[b], np = pd_nparams(s.block_kinds[b], sz);
        L.block(off, off, sz, sz) = pd_factor(s.block_kinds[b], sz, theta.segment(pos, np));
        off += sz;
        pos += np;
    }
    return L;
}

/// Build the covariance matrix for a full specification.
inline Eigen::MatrixXd pd_sigma(const PdSpec& s, int q, const Eigen::VectorXd& theta) {
    const Eigen::MatrixXd L = pd_factor(s, q, theta);
    return L * L.transpose();
}

}  // namespace mixedfit
