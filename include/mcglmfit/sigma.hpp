// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file sigma.hpp
/// @brief Within-response covariance matrix Sigma_k, its derivatives and its factors.
///
/// For response k, with mean mu_k, variance exponents p and dispersion parameters tau:
///
///     Sigma_k = V^{1/2} Omega V^{1/2}  [+ diag(mu) for poisson_tweedie] [+ diag(mu^2) for geom_tweedie]
///
/// where Omega = sum_j tau_j Z_j (covariance link `Identity`), Omega^{-1} = sum_j tau_j Z_j (`Inverse`), or
/// Omega = expm(sum_j tau_j Z_j) (`Expm`). When every Z_j is diagonal (any covariance link) a diagonal fast path
/// with O(n) cost is used. Otherwise dense BLAS/LAPACK kernels are used.
#pragma once
#include "linalg.hpp"
#include "types.hpp"
#include "variance.hpp"
#include <memory>

namespace mcglm {

/// Symmetric n x n matrix stored as a diagonal vector, a non-owning reference to an existing matrix
/// (typically a user-supplied Z), or an owned dense matrix.
struct SymMat {
    bool diag = false;
    VectorXd d;
    const MatrixXd* ref = nullptr;
    MatrixXd own;
    const MatrixXd& mat() const { return ref ? *ref : own; }
    Eigen::Index n() const { return diag ? d.size() : mat().rows(); }
    MatrixXd dense() const { return diag ? MatrixXd(d.asDiagonal()) : mat(); }
};

/// Sigma_k and the quantities derived from it.
struct SigmaBlock {
    Eigen::Index n = 0;
    bool diagonal = false;        ///< Sigma is diagonal (fast path); only `sdiag` is then populated
    VectorXd sdiag;               ///< diag(Sigma)
    MatrixXd Sigma;               ///< dense Sigma (empty when `diagonal`)
    MatrixXd Sinv;                ///< Sigma^{-1} (only if requested)
    MatrixXd F, Finv;             ///< Sigma = F^T F and F^{-1}; F is upper triangular (lower for covariance `Inverse`)
    std::vector<SymMat> dSigma;   ///< derivatives of Sigma: exponents first (if estimated), then one per tau
};

namespace detail {

/// True when every Z matrix is stored as a diagonal.
inline bool all_diag(const std::vector<ZMat>& Z) {
    for (auto& z : Z) if (!z.is_diag) return false;
    return true;
}

/// Dense sum_j tau_j Z_j.
/// @throws InvalidInput if the number of tau parameters differs from the number of Z matrices
inline MatrixXd linear_predictor_dense(const std::vector<ZMat>& Z, const VectorXd& tau, Eigen::Index n) {
    if (uz(tau.size()) != Z.size()) throw InvalidInput("expected " + std::to_string(Z.size()) + " tau parameter(s), got " + std::to_string(tau.size()));
    MatrixXd O = MatrixXd::Zero(n, n);
    for (size_t j = 0; j < Z.size(); ++j) {
        if (Z[j].is_diag) O.diagonal() += tau[tix(j)] * Z[j].diag;
        else O += tau[tix(j)] * Z[j].dense;
    }
    return O;
}
/// Diagonal of sum_j tau_j Z_j when every Z_j is diagonal.
inline VectorXd linear_predictor_diag(const std::vector<ZMat>& Z, const VectorXd& tau, Eigen::Index n) {
    if (uz(tau.size()) != Z.size()) throw InvalidInput("expected " + std::to_string(Z.size()) + " tau parameter(s), got " + std::to_string(tau.size()));
    VectorXd d = VectorXd::Zero(n);
    for (size_t j = 0; j < Z.size(); ++j) d += tau[tix(j)] * Z[j].diag;
    return d;
}
/// diag(a) M diag(b) without forming the diagonal matrices.
inline MatrixXd vMv(const VectorXd& a, const MatrixXd& M, const VectorXd& b) {
    MatrixXd R = M;
    for (Eigen::Index j = 0; j < R.cols(); ++j) R.col(j) *= b[j];
    for (Eigen::Index j = 0; j < R.cols(); ++j) R.col(j).array() *= a.array();
    return R;
}
/// V Z V with V = diag(v).
inline SymMat scaled_sym(const ZMat& z, const VectorXd& v) {
    SymMat s;
    if (z.is_diag) { s.diag = true; s.d = z.diag.array() * v.array().square(); }
    else s.own = vMv(v, z.dense, v);
    return s;
}
/// Derivative of V Omega V with respect to a variance exponent: V' Omega V + V Omega V' (V' = diag(dv)).
inline SymMat power_sandwich(const MatrixXd& mid, const VectorXd& v, const VectorXd& dv) {
    SymMat s; MatrixXd H = vMv(v, mid, dv); s.own = H + H.transpose(); return s;
}
/// -S X S, the derivative of an inverse: d(A^{-1}) = -A^{-1} dA A^{-1}.
inline SymMat neg_sandwich(const MatrixXd& S, const SymMat& X) {
    SymMat s;
    MatrixXd T = X.diag ? MatrixXd(S * X.d.asDiagonal()) : MatrixXd(S * X.mat());
    s.own = -(T * S); return s;
}

/// Spectral data of a symmetric matrix U = Q diag(q) Q^T together with Omega = expm(U).
struct ExpmInfo { MatrixXd Q; VectorXd q; MatrixXd Omega; };

/// Matrix exponential of a symmetric matrix through its eigendecomposition.
/// @throws NumericalError if the eigensolver fails
inline ExpmInfo expm_sym(const MatrixXd& U) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(U);
    if (es.info() != Eigen::Success) throw NumericalError("eigendecomposition failed while computing the matrix exponential");
    ExpmInfo e; e.Q = es.eigenvectors(); e.q = es.eigenvalues();
    VectorXd eq = e.q.array().exp();
    e.Omega = e.Q * eq.asDiagonal() * e.Q.transpose();
    e.Omega = 0.5 * (e.Omega + e.Omega.transpose());
    return e;
}
/// Frechet derivative of expm at U in the direction Z, using divided differences of exp on the spectrum:
/// Q [ (Q^T Z Q) .* f(q_i, q_j) ] Q^T with f(a, b) = (e^a - e^b) / (a - b) and f(a, a) = e^a.
inline MatrixXd expm_derivative(const ExpmInfo& e, const ZMat& z) {
    const Eigen::Index n = e.q.size();
    MatrixXd Zd = z.is_diag ? MatrixXd(z.diag.asDiagonal()) : z.dense;
    MatrixXd H = e.Q.transpose() * Zd * e.Q;
    VectorXd eq = e.q.array().exp();
    MatrixXd P(n, n);
    for (Eigen::Index j = 0; j < n; ++j)
        for (Eigen::Index i = 0; i < n; ++i) {
            double dq = e.q[i] - e.q[j];
            double f = (i == j || std::fabs(dq) < 1e-12 * (1.0 + std::fabs(e.q[i]))) ? eq[i]
                                                                                     : (eq[i] - eq[j]) / dq;
            P(i, j) = H(i, j) * f;
        }
    MatrixXd D = e.Q * P * e.Q.transpose();
    return 0.5 * (D + D.transpose());
}

} // namespace detail

/// What `build_sigma` should compute besides Sigma itself.
struct SigmaRequest {
    bool need_inverse = true;     ///< Sigma^{-1}
    bool need_factor = false;     ///< F and F^{-1} (needed for K > 1)
    bool need_deriv = false;      ///< derivatives of Sigma with respect to the exponents and tau
};

/// Builds Sigma_k for one response.
/// @param rs     response specification (link, variance, covariance link, Z matrices)
/// @param mu     mean of the response
/// @param tau    dispersion parameters (one per Z)
/// @param power  variance exponents
/// @param rq     which derived quantities to compute
/// @throws NumericalError if Sigma is not positive definite
/// @throws InvalidInput on a wrong number of parameters
inline SigmaBlock build_sigma(const ResponseSpec& rs, const VectorXd& mu, const VectorXd& tau,
                              const VectorXd& power, const SigmaRequest& rq) {
    using namespace detail;
    const Eigen::Index n = mu.size();
    SigmaBlock sb; sb.n = n;
    const bool use_v = rs.variance != Variance::Constant;
    const bool deriv_power = rq.need_deriv && !rs.power_fixed && use_v;
    for (const ZMat& z : rs.Z)
        if (z.is_sparse) throw InvalidInput("a sparse Z reached the dense kernel; it must be expanded or decomposed first");
    const bool diag_ok = all_diag(rs.Z);

    // ---------- diagonal fast path (every Z diagonal; any covariance link) ----------
    // With diagonal Z all matrices commute, so the covariance link acts elementwise:
    //   identity: Omega = sum_j tau_j z_j;  expm: Omega = exp(sum_j tau_j z_j);  inverse: Omega = 1 / sum_j tau_j z_j.
    if (diag_ok) {
        VectorXd lin = linear_predictor_diag(rs.Z, tau, n);
        VectorXd om, dlink;                       // Omega and the factor with d Omega / d tau_j = dlink .* z_j
        switch (rs.covariance) {
        case CovLink::Identity: om = lin; dlink = VectorXd::Ones(n); break;
        case CovLink::Expm: om = lin.array().exp(); dlink = om; break;
        case CovLink::Inverse:
            for (Eigen::Index i = 0; i < n; ++i)
                if (!(lin[i] > 0)) throw NumericalError("the precision combination (inverse covariance link) is not strictly positive; check tau");
            om = 1.0 / lin.array(); dlink = -om.array().square(); break;
        }
        VectorXd s = om;
        VarOut vo; VectorXd v;
        if (use_v) {
            vo = variance_function(rs.variance, mu, power, rs.ntrial, false, deriv_power);
            v = vo.v; s = v.array().square() * om.array();
            if (rs.variance == Variance::PoissonTweedie) s += mu;
            if (rs.variance == Variance::GeomTweedie) s += mu.cwiseAbs2();
        }
        for (Eigen::Index i = 0; i < n; ++i)
            if (!(s[i] > 0) || !std::isfinite(s[i]))
                throw NumericalError("the diagonal of Sigma is not strictly positive and finite; check tau and the variance function");
        sb.diagonal = true; sb.sdiag = s;
        if (rq.need_deriv) {
            if (deriv_power) {
                SymMat sp; sp.diag = true; sp.d = 2.0 * v.array() * om.array() * vo.dp.array();
                sb.dSigma.push_back(std::move(sp));
                if (rs.variance == Variance::BinomialPQ) {
                    SymMat sq; sq.diag = true; sq.d = 2.0 * v.array() * om.array() * vo.dq.array();
                    sb.dSigma.push_back(std::move(sq));
                }
            }
            for (auto& z : rs.Z) {
                SymMat t; t.diag = true;
                VectorXd dz = z.diag.array() * dlink.array();
                t.d = use_v ? VectorXd(dz.array() * v.array().square()) : dz;
                sb.dSigma.push_back(std::move(t));
            }
        }
        return sb;
    }

    // ---------- dense path ----------
    MatrixXd Sigma;
    MatrixXd Sinv; bool have_sinv = false;
    MatrixXd Lfac;                // lower Cholesky factor of whatever was factorised
    bool factor_lower = false;    // true: F = Linv^{-1} is lower triangular (covariance link `Inverse`)
    VectorXd v, vinv; VarOut vo;
    const bool inverse_cov = rs.covariance == CovLink::Inverse;
    const bool pt_like = rs.variance == Variance::PoissonTweedie || rs.variance == Variance::GeomTweedie;

    MatrixXd Omega;               // Omega (identity / expm) or Omega^{-1} (inverse)
    ExpmInfo ex;
    if (rs.covariance == CovLink::Expm) {
        ex = expm_sym(linear_predictor_dense(rs.Z, tau, n)); Omega = ex.Omega;
    } else {
        Omega = linear_predictor_dense(rs.Z, tau, n);
    }

    if (!inverse_cov || pt_like) {
        MatrixXd Om = Omega;
        if (inverse_cov) { MatrixXd Oi; la::spd_inverse(Omega, Oi); Om = Oi; }   // Omega = (Omega^{-1})^{-1}
        if (use_v) {
            vo = variance_function(rs.variance, mu, power, rs.ntrial, false, deriv_power);
            v = vo.v; Sigma = vMv(v, Om, v);
            if (rs.variance == Variance::PoissonTweedie) Sigma.diagonal() += mu;
            if (rs.variance == Variance::GeomTweedie) Sigma.diagonal() += mu.cwiseAbs2();
        } else Sigma = Om;
        Sigma = 0.5 * (Sigma + Sigma.transpose());
        if (rq.need_deriv) {
            if (deriv_power) {
                sb.dSigma.push_back(power_sandwich(Om, v, vo.dp));
                if (rs.variance == Variance::BinomialPQ) sb.dSigma.push_back(power_sandwich(Om, v, vo.dq));
            }
            for (size_t j = 0; j < rs.Z.size(); ++j) {
                if (rs.covariance == CovLink::Expm) {
                    SymMat s; MatrixXd dO = expm_derivative(ex, rs.Z[j]);
                    if (use_v) dO = vMv(v, dO, v);
                    s.own = std::move(dO); sb.dSigma.push_back(std::move(s));
                } else if (inverse_cov) {      // poisson/geom_tweedie with inverse link: dOmega = -Om Z Om
                    SymMat s; MatrixXd T = rs.Z[j].is_diag ? MatrixXd(Om * rs.Z[j].diag.asDiagonal())
                                                           : MatrixXd(Om * rs.Z[j].dense);
                    MatrixXd dO = -(T * Om); s.own = vMv(v, dO, v); sb.dSigma.push_back(std::move(s));
                } else if (use_v) {
                    sb.dSigma.push_back(scaled_sym(rs.Z[j], v));
                } else {
                    SymMat s; if (rs.Z[j].is_diag) { s.diag = true; s.d = rs.Z[j].diag; } else s.ref = &rs.Z[j].dense;
                    sb.dSigma.push_back(std::move(s));
                }
            }
        }
        if (rq.need_inverse || rq.need_factor) la::spd_inverse(Sigma, Sinv, &Lfac), have_sinv = true;
    } else {
        // inverse covariance link with constant / power / binomial variances
        MatrixXd InvSig;
        if (use_v) {
            vo = variance_function(rs.variance, mu, power, rs.ntrial, true, deriv_power);
            vinv = vo.v; InvSig = vMv(vinv, Omega, vinv);
        } else InvSig = Omega;
        InvSig = 0.5 * (InvSig + InvSig.transpose());
        MatrixXd Linv;                                         // InvSig = Linv Linv^T
        la::spd_inverse(InvSig, Sigma, &Linv);                 // Sigma = InvSig^{-1}
        Sigma = 0.5 * (Sigma + Sigma.transpose());
        Sinv = InvSig; have_sinv = true; Lfac = Linv; factor_lower = true;
        if (rq.need_deriv) {
            if (deriv_power) {
                SymMat a = power_sandwich(Omega, vinv, vo.dp); sb.dSigma.push_back(neg_sandwich(Sigma, a));
                if (rs.variance == Variance::BinomialPQ) {
                    SymMat b = power_sandwich(Omega, vinv, vo.dq); sb.dSigma.push_back(neg_sandwich(Sigma, b));
                }
            }
            for (auto& z : rs.Z) {
                SymMat X = use_v ? scaled_sym(z, vinv) : SymMat();
                if (!use_v) { if (z.is_diag) { X.diag = true; X.d = z.diag; } else X.ref = &z.dense; }
                sb.dSigma.push_back(neg_sandwich(Sigma, X));
            }
        }
    }

    sb.Sigma = std::move(Sigma); sb.sdiag = sb.Sigma.diagonal();
    if (have_sinv && rq.need_inverse) sb.Sinv = Sinv;
    if (rq.need_factor) {
        if (!factor_lower) {                       // F = L^T (upper), Finv = F^{-1} = (L^{-1})^T
            sb.F = Lfac.transpose();
            sb.Finv = la::tri_inverse_lower(Lfac).transpose();
        } else {                                   // F = Linv^{-1} (lower), Finv = Linv
            sb.Finv = Lfac;
            sb.F = la::tri_inverse_lower(Lfac);
        }
    }
    return sb;
}

} // namespace mcglm
