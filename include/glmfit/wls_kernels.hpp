// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file wls_kernels.hpp
 * @brief Weighted-least-squares kernels used by the IRLS driver (internal header).
 *
 * Every WLS step minimises ||diag(w) (z - X b)||^2 and is reduced to a small p x p problem:
 *
 *   - QR family  : blocked TSQR of [diag(w) X | w.*z] -> R (p x p) and c = Q'(w.*z).
 *                  Pivoted / SVD variants then work on R (same singular values as diag(w) X),
 *                  never on the n x p matrix.
 *   - Normal eqs : X' W X and X' W z accumulated over row chunks (no n x p temporary).
 *                  If the system is not safely positive definite the step silently falls
 *                  back to the pivoted-QR path.
 *
 * Every path works on column-equilibrated data (each column of diag(w) X scaled to unit norm), so the
 * rank decision is invariant to the units of the covariates (a pivot is judged against its own column).
 * Right-hand sides are the already weighted targets `tw` (w.*z for a full solve, w.*(y-mu)/mu' for an
 * increment), and coefficients are returned in the original column units.
 *
 * All paths expose the same results: coefficients, rank, aliased columns, log|X'WX| and
 * (X'WX)^-1 (zero rows/cols for aliased columns), which is all the driver needs.
 */

#ifndef _GLMFIT_WLS_KERNELS_HPP
#define _GLMFIT_WLS_KERNELS_HPP

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glm::detail {

using Eigen::ArrayXd;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/// Linear-algebra back end selected through the `type` constructor argument.
enum class Method : int {
    ColPivQR      = 0,  ///< column-pivoted QR, rank revealing (aliased columns -> beta = 0, se/vcov = NaN)
    HouseholderQR = 1,  ///< unpivoted QR (falls back to ColPivQR if rank deficient)
    LLT           = 2,  ///< Cholesky of X'WX (falls back to ColPivQR if not safely positive definite)
    LDLT          = 3,  ///< LDL' of X'WX     (falls back to ColPivQR if not safely positive definite)
    FullPivQR     = 4,  ///< full-pivot QR, rank revealing
    SVD           = 5,  ///< BDC-SVD, minimum-norm solution for rank deficient problems
    Auto          = 6   ///< fast default: Jacobi-scaled Cholesky, automatic fallback to ColPivQR when ill-conditioned
};

[[nodiscard]] constexpr bool is_normal_equation_method(Method m) noexcept {
    return m == Method::LLT || m == Method::LDLT || m == Method::Auto;
}

/// Rows per chunk for streaming kernels. `FASTGLM_CHUNK_ROWS` overrides the 16384 default.
[[nodiscard]] inline Index default_chunk_rows() {
    static const Index cached = [] {
        Index v = 16384;
        if (const char* env = std::getenv("FASTGLM_CHUNK_ROWS")) {
            const long parsed = std::strtol(env, nullptr, 10);
            if (parsed > 0) v = static_cast<Index>(parsed);
        }
        return v;
    }();
    return cached;
}

#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
using SvdType = Eigen::BDCSVD<MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV>;
inline void svd_compute(SvdType& s, const MatrixXd& m) { s.compute(m); }
#else
using SvdType = Eigen::BDCSVD<MatrixXd>;
inline void svd_compute(SvdType& s, const MatrixXd& m) {
    s.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
}
#endif

// ---------------------------------------------------------------------------
// Streaming kernels
// ---------------------------------------------------------------------------

/// In-place Householder QR of a k x q block; returns its q x q upper-triangular factor
/// (zero padded when k < q).
[[nodiscard]] inline MatrixXd reduce_qr_block(Eigen::Ref<MatrixXd> M) {
    const Index k = M.rows();
    const Index q = M.cols();
    Eigen::HouseholderQR<Eigen::Ref<MatrixXd>> qr(M);
    const Index m = std::min(k, q);
    MatrixXd R = MatrixXd::Zero(q, q);
    R.topRows(m) = qr.matrixQR().topRows(m).triangularView<Eigen::Upper>();
    return R;
}

/// R factor of the stacked pair [a; b].
[[nodiscard]] inline MatrixXd merge_r(const MatrixXd& a, const MatrixXd& b) {
    const Index q = a.cols();
    MatrixXd s(2 * q, q);
    s.topRows(q)    = a;
    s.bottomRows(q) = b;
    return reduce_qr_block(s);
}

/**
 * @brief Tall-skinny QR of the augmented matrix [diag(w) X | w.*z].
 * @return (p+1) x (p+1) upper-triangular R_aug. Its leading p x p block is R (with X'W X = R'R) and
 *         column p, rows 0..p-1, is Q'(w.*z).
 * @param chunk rows per block; `chunk >= n` means a single in-place QR using `scratch`.
 * @param scratch reusable n x (p+1) buffer for the single-block case.
 */
[[nodiscard]] inline MatrixXd tsqr_augmented(const Eigen::Ref<const MatrixXd>& X,
                                             const Eigen::Ref<const ArrayXd>& w,
                                             const Eigen::Ref<const ArrayXd>& tw,
                                             const Eigen::Ref<const ArrayXd>& inv_s,
                                             Index chunk, MatrixXd& scratch) {
    const Index n = X.rows();
    const Index p = X.cols();
    const Index q = p + 1;

    if (chunk >= n) {
        scratch.resize(n, q);
        scratch.leftCols(p).array() = (X.array().colwise() * w).rowwise() * inv_s.transpose();
        scratch.col(p).array()      = tw;
        return reduce_qr_block(scratch);
    }

    chunk = std::max(chunk, q);
    const Index nchunks = (n + chunk - 1) / chunk;

    Index wave = 2;
#ifdef _OPENMP
    wave = std::max<Index>(2, 2 * static_cast<Index>(omp_get_max_threads()));
#endif
    std::vector<MatrixXd> parts(static_cast<std::size_t>(std::min(wave, nchunks)));

    MatrixXd acc;
    bool have = false;
    for (Index c0 = 0; c0 < nchunks; c0 += wave) {
        const Index cnt = std::min(wave, nchunks - c0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (Index j = 0; j < cnt; ++j) {
            const Index r0 = (c0 + j) * chunk;
            const Index k  = std::min(chunk, n - r0);
            MatrixXd M(k, q);
            M.leftCols(p).array() = (X.middleRows(r0, k).array().colwise() * w.segment(r0, k)).rowwise() * inv_s.transpose();
            M.col(p).array()      = tw.segment(r0, k);
            parts[static_cast<std::size_t>(j)] = reduce_qr_block(M);
        }
        for (Index j = 0; j < cnt; ++j) {
            acc  = have ? merge_r(acc, parts[static_cast<std::size_t>(j)])
                        : parts[static_cast<std::size_t>(j)];
            have = true;
        }
    }
    return acc;
}

/// Weighted column norms ||diag(w) X_j||.
[[nodiscard]] inline VectorXd weighted_col_norms(const Eigen::Ref<const MatrixXd>& X,
                                                 const Eigen::Ref<const ArrayXd>& w) {
    const Index p = X.cols();
    VectorXd s(p);
    for (Index j = 0; j < p; ++j) s[j] = std::sqrt((X.col(j).array() * w).square().sum());
    return s;
}

/// A = X' diag(w^2) X (lower triangle only is valid) and b = (diag(w) X)' tw, chunked.
inline void normal_equations(const Eigen::Ref<const MatrixXd>& X,
                             const Eigen::Ref<const ArrayXd>& w,
                             const Eigen::Ref<const ArrayXd>& tw,
                             Index chunk, MatrixXd& A, VectorXd& b) {
    const Index n = X.rows();
    const Index p = X.cols();
    chunk = std::max<Index>(1, std::min(chunk, n));
    A.setZero(p, p);
    b.setZero(p);

    MatrixXd WB(chunk, p);
    for (Index r = 0; r < n; r += chunk) {
        const Index k = std::min(chunk, n - r);
        WB.topRows(k).array() = X.middleRows(r, k).array().colwise() * w.segment(r, k);
        A.selfadjointView<Eigen::Lower>().rankUpdate(WB.topRows(k).transpose());
        b.noalias() += WB.topRows(k).transpose() * tw.segment(r, k).matrix();
    }
}

/// Hat-matrix diagonal h_i = w_i^2 x_i' Ainv x_i, chunked.
inline void leverages(const Eigen::Ref<const MatrixXd>& X,
                      const Eigen::Ref<const ArrayXd>& w,
                      const MatrixXd& Ainv, Index chunk,
                      Eigen::Ref<ArrayXd> h) {
    const Index n = X.rows();
    const Index p = X.cols();
    chunk = std::max<Index>(1, std::min(chunk, n));
    MatrixXd WB(chunk, p);
    MatrixXd T(chunk, p);
    for (Index r = 0; r < n; r += chunk) {
        const Index k = std::min(chunk, n - r);
        WB.topRows(k).array() = X.middleRows(r, k).array().colwise() * w.segment(r, k);
        T.topRows(k).noalias() = WB.topRows(k) * Ainv;
        h.segment(r, k) = (T.topRows(k).array() * WB.topRows(k).array()).rowwise().sum();
    }
}

// ---------------------------------------------------------------------------
// WLS solver state
// ---------------------------------------------------------------------------

/// Pseudo-inverse of the information matrix from a rank-revealing pivoted QR of R.
template <class PivotedQR>
[[nodiscard]] MatrixXd pivoted_inverse(const PivotedQR& qr, Index p, Index r) {
    MatrixXd padded = MatrixXd::Zero(p, p);
    if (r > 0) {
        const MatrixXd Z = qr.matrixQR().topLeftCorner(r, r).template triangularView<Eigen::Upper>()
                               .solve(MatrixXd::Identity(r, r));
        padded.topLeftCorner(r, r).noalias() = Z * Z.transpose();
    }
    const auto& P = qr.colsPermutation();
    MatrixXd out = P * padded * P.transpose();
    return out;
}

/// log|A| restricted to the r leading pivots of a pivoted QR (A = P R2'R2 P').
template <class PivotedQR>
[[nodiscard]] double pivoted_logdet(const PivotedQR& qr, Index r) {
    double s = 0.0;
    for (Index i = 0; i < r; ++i) s += std::log(std::abs(qr.matrixQR()(i, i)));
    return 2.0 * s;
}

class WlsSolver {
public:
    WlsSolver(Index p, Method m, double rank_tol, bool equilibrate = true)
        : p_(p), requested_(m), used_(m), eff_(m == Method::Auto ? Method::LLT : m),
          rank_tol_(rank_tol), equil_(equilibrate), rank_(0), logdet_(0.0), solved_(false) {}

    /// Re-dimensions the solver (used when the active column set shrinks); clears all state.
    void reset(Index p) {
        p_ = p; rank_ = 0; logdet_ = 0.0; solved_ = false; aliased_.clear(); gram_pinv_ = false;
        used_ = eff_;
    }

    /// Solve min ||tw - diag(w) X b||; `tw` is the weighted target. `beta` is returned in the original
    /// column units. `qr_chunk` is the TSQR block, `ne_chunk` the normal-equation block.
    void solve(const Eigen::Ref<const MatrixXd>& X, const Eigen::Ref<const ArrayXd>& w,
               const Eigen::Ref<const ArrayXd>& tw, Index qr_chunk, Index ne_chunk, VectorXd& beta) {
        aliased_.clear();
        gram_pinv_ = false;
        used_ = eff_;
        beta.resize(p_);
        if (is_normal_equation_method(eff_)) {
            if (solve_normal(X, w, tw, ne_chunk, beta)) { solved_ = true; return; }
            used_ = Method::ColPivQR;  // not safely positive definite -> rank-revealing path
        }
        solve_qr(X, w, tw, qr_chunk, beta);
        solved_ = true;
    }

    /**
     * @brief Structural rank in column order, R-style (LINPACK dqrdc2): a column is aliased when it is
     *        (numerically) in the span of the earlier retained columns, so later duplicates are the ones dropped.
     * @details Works on the TSQR triangle of the equilibrated diag(w) X (modified Gram-Schmidt, two passes,
     *          on the small p x p factor). Returns the 0-based aliased columns, ascending.
     */
    [[nodiscard]] std::vector<int> detect_aliased_ordered(const Eigen::Ref<const MatrixXd>& X,
                                                          const Eigen::Ref<const ArrayXd>& w,
                                                          Index chunk) {
        // Fast exit: a well-conditioned equilibrated Gram matrix (one syrk pass) proves full column rank.
        {
            MatrixXd G; VectorXd g;
            normal_equations(X, w, ArrayXd::Zero(X.rows()), chunk, G, g);
            const VectorXd dg = G.diagonal();
            if (dg.allFinite() && (dg.array() > 0.0).all()) {
                const VectorXd sc = dg.cwiseSqrt().cwiseInverse();
                for (Index j = 0; j < p_; ++j)
                    for (Index i = j; i < p_; ++i) G(i, j) *= sc[i] * sc[j];
                Eigen::LLT<MatrixXd> llt(G);
                if (llt.info() == Eigen::Success) {
                    const ArrayXd d = llt.matrixLLT().diagonal().array();
                    if (d.minCoeff() > 1e-3) return {};   // every pivot >= 1e-3 of its (unit) column
                }
            }
        }
        VectorXd s = equil_ ? weighted_col_norms(X, w) : VectorXd::Ones(p_);
        for (Index j = 0; j < p_; ++j) if (!(s[j] > 0.0) || !std::isfinite(s[j])) s[j] = 1.0;
        const VectorXd inv = s.cwiseInverse();
        const ArrayXd zero = ArrayXd::Zero(X.rows());
        const MatrixXd Raug = tsqr_augmented(X, w, zero, inv.array(), chunk, scratch_);
        const MatrixXd R = Raug.topLeftCorner(p_, p_);
        std::vector<int> aliased;
        std::vector<VectorXd> basis;           // orthonormal basis of the retained columns
        for (Index j = 0; j < p_; ++j) {
            VectorXd v = R.col(j);
            const double n0 = v.norm();
            for (int pass = 0; pass < 2; ++pass)
                for (const VectorXd& q : basis) v -= q * q.dot(v);
            const double nr = v.norm();
            if (!(n0 > 0.0) || !(nr > rank_tol_ * n0) || !(nr > rank_tol_ * (equil_ ? 1.0 : n0))) {
                aliased.push_back(static_cast<int>(j));
            } else {
                basis.push_back(v / nr);
            }
        }
        return aliased;
    }

    /**
     * @brief Solve from a precomputed Gram matrix (sparse designs): A = X' diag(w^2) X (lower triangle valid)
     *        and b = X'(w .* tw). Equilibrated Cholesky with a condition check; otherwise an eigen
     *        pseudo-inverse (rank-revealing, min-norm). `beta` is returned in the original column units.
     */
    void solve_gram(MatrixXd A, VectorXd b, VectorXd& beta) {
        aliased_.clear();
        gram_pinv_ = false;
        used_ = Method::LLT;
        beta.resize(p_);
        const VectorXd dg = A.diagonal();
        VectorXd s = dg.cwiseMax(0.0).cwiseSqrt();
        for (Index j = 0; j < p_; ++j) if (!(s[j] > 0.0) || !std::isfinite(s[j])) s[j] = 1.0;
        if (!equil_) s.setOnes();
        inv_s_ = s.cwiseInverse();
        logs_  = s.array().log().matrix();
        for (Index j = 0; j < p_; ++j)
            for (Index i = j; i < p_; ++i) A(i, j) *= inv_s_[i] * inv_s_[j];
        b.array() *= inv_s_.array();
        A_ = A;

        llt_.compute(A_);
        if (llt_.info() == Eigen::Success) {
            const ArrayXd d = llt_.matrixLLT().diagonal().array();
            const double dmax = d.maxCoeff(), dmin = d.minCoeff();
            if (dmin > 0.0 && std::isfinite(dmax) && (dmin * dmin) / (dmax * dmax) >= std::max(rank_tol_, 1e-8)) {
                const VectorXd bt = llt_.solve(b);
                if (bt.allFinite()) {
                    beta = bt.cwiseProduct(inv_s_);
                    logdet_ = 2.0 * d.log().sum() + 2.0 * logs_.sum();
                    rank_ = p_;
                    solved_ = true;
                    return;
                }
            }
        }
        // pseudo-inverse through the symmetric eigen-decomposition of the scaled Gram matrix
        MatrixXd S = A_.template selfadjointView<Eigen::Lower>();
        Eigen::SelfAdjointEigenSolver<MatrixXd> es(S);
        const VectorXd ev = es.eigenvalues();
        const double cut = std::max(rank_tol_, 1e-12) * std::max(ev.maxCoeff(), 0.0);
        Index r = 0;
        pinv_ = MatrixXd::Zero(p_, p_);
        logdet_ = 2.0 * logs_.sum();
        for (Index k = 0; k < p_; ++k) {
            if (ev[k] > cut && ev[k] > 0.0) {
                pinv_.noalias() += es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose() / ev[k];
                logdet_ += std::log(ev[k]);
                ++r;
            }
        }
        rank_ = r;
        gram_pinv_ = true;
        used_ = Method::SVD;
        beta = (pinv_ * b).cwiseProduct(inv_s_);
        solved_ = true;
    }

    /// Levenberg-Marquardt step: (A_s + lambda I) d_s = b_s on the Jacobi-scaled Gram matrix. Stateless
    /// (does not disturb the factorisation used for the covariance). Returns false if not solvable.
    [[nodiscard]] bool damped_gram(MatrixXd A, VectorXd b, double lambda, VectorXd& beta) const {
        const VectorXd dg = A.diagonal();
        VectorXd s = dg.cwiseMax(0.0).cwiseSqrt();
        for (Index j = 0; j < p_; ++j) if (!(s[j] > 0.0) || !std::isfinite(s[j])) s[j] = 1.0;
        const VectorXd inv = s.cwiseInverse();
        for (Index j = 0; j < p_; ++j)
            for (Index i = j; i < p_; ++i) A(i, j) *= inv[i] * inv[j];
        A.diagonal().array() += lambda;
        b.array() *= inv.array();
        Eigen::LLT<MatrixXd> llt(A);
        if (llt.info() != Eigen::Success) return false;
        const VectorXd bt = llt.solve(b);
        if (!bt.allFinite()) return false;
        beta = bt.cwiseProduct(inv);
        return true;
    }

    /// Ordered structural rank from a Gram matrix (sparse designs): sequential Cholesky of the equilibrated
    /// Gram, dropping a column whose residual variance is (numerically) zero. Later duplicates are dropped.
    [[nodiscard]] std::vector<int> detect_aliased_gram(const MatrixXd& G) const {
        std::vector<int> aliased;
        const double tol = std::max(rank_tol_, 1e-12);
        std::vector<Index> kept;
        MatrixXd L = MatrixXd::Zero(p_, p_);   // rows of the Cholesky factor of the kept columns
        for (Index j = 0; j < p_; ++j) {
            const double gjj = G(j, j);
            if (!(gjj > 0.0) || !std::isfinite(gjj)) { aliased.push_back(static_cast<int>(j)); continue; }
            const Index k = static_cast<Index>(kept.size());
            VectorXd l(k);
            for (Index a = 0; a < k; ++a) {
                const Index ia = kept[static_cast<std::size_t>(a)];
                const double gij = (ia >= j ? G(ia, j) : G(j, ia));
                double v = gij / std::sqrt(G(ia, ia) * gjj);
                for (Index c = 0; c < a; ++c) v -= L(a, c) * l[c];
                l[a] = v / L(a, a);
            }
            const double d = 1.0 - l.squaredNorm();
            if (d > tol) {
                for (Index c = 0; c < k; ++c) L(k, c) = l[c];
                L(k, k) = std::sqrt(d);
                kept.push_back(j);
            } else {
                aliased.push_back(static_cast<int>(j));
            }
        }
        return aliased;
    }

    [[nodiscard]] bool   solved() const noexcept { return solved_; }
    [[nodiscard]] Index  rank() const noexcept { return rank_; }
    [[nodiscard]] double log_det() const noexcept { return logdet_; }
    [[nodiscard]] const std::vector<int>& aliased() const noexcept { return aliased_; }
    [[nodiscard]] Method used_method() const noexcept { return used_; }

    /// (X'WX)^-1 in the original column units (pseudo-inverse if rank deficient; aliased rows/cols are zero).
    [[nodiscard]] MatrixXd inverse_information() const {
        MatrixXd out = MatrixXd::Zero(p_, p_);
        switch (used_) {
            case Method::LLT:  out = llt_.solve(MatrixXd::Identity(p_, p_)); break;
            case Method::LDLT: out = ldlt_.solve(MatrixXd::Identity(p_, p_)); break;
            case Method::HouseholderQR: {
                const MatrixXd Ri = R_.triangularView<Eigen::Upper>().solve(MatrixXd::Identity(p_, p_));
                out = Ri * Ri.transpose();
                break;
            }
            case Method::ColPivQR:  out = pivoted_inverse(cpqr_, p_, rank_); break;
            case Method::FullPivQR: out = pivoted_inverse(fpqr_, p_, rank_); break;
            case Method::SVD: {
                if (gram_pinv_) { out = pinv_; break; }
                const Index r = rank_;
                if (r > 0) {
                    const MatrixXd Vr = svd_.matrixV().leftCols(r);
                    const ArrayXd  s  = svd_.singularValues().head(r).array();
                    const VectorXd d  = (1.0 / s.square()).matrix();
                    out = Vr * d.asDiagonal() * Vr.transpose();
                }
                break;
            }
            default: break;
        }
        if (inv_s_.size() == p_) out = inv_s_.asDiagonal() * out * inv_s_.asDiagonal();
        return out;
    }

private:
    Index  p_;
    Method requested_;
    Method used_;
    Method eff_;
    double rank_tol_;
    bool   equil_;

    Index  rank_;
    double logdet_;
    bool   solved_;
    std::vector<int> aliased_;

    bool     gram_pinv_ = false;  // true when the last solve_gram used the eigen pseudo-inverse
    MatrixXd pinv_;               // scaled pseudo-inverse (gram_pinv_ only)
    VectorXd inv_s_;    // 1 / column scale
    VectorXd logs_;     // log column scale

    MatrixXd scratch_;  // n x (p+1) buffer for single-block QR
    MatrixXd R_;        // p x p upper triangular factor (scaled)
    VectorXd c_;        // Q'tw, first p entries
    VectorXd qtc_;
    MatrixXd A_;
    VectorXd b_;

    Eigen::ColPivHouseholderQR<MatrixXd>  cpqr_;
    Eigen::FullPivHouseholderQR<MatrixXd> fpqr_;
    SvdType                               svd_;
    Eigen::LLT<MatrixXd>                  llt_;
    Eigen::LDLT<MatrixXd>                 ldlt_;

    void set_unit_scales() { inv_s_ = VectorXd::Ones(p_); logs_ = VectorXd::Zero(p_); }

    bool solve_normal(const Eigen::Ref<const MatrixXd>& X, const Eigen::Ref<const ArrayXd>& w,
                      const Eigen::Ref<const ArrayXd>& tw, Index chunk, VectorXd& beta) {
        normal_equations(X, w, tw, chunk, A_, b_);
        if (!b_.allFinite()) return false;
        const VectorXd dg = A_.diagonal();
        if (!dg.allFinite() || !(dg.array() > 0.0).all()) return false;   // empty column -> QR path

        if (equil_) {
            const VectorXd s = dg.cwiseSqrt();
            inv_s_ = s.cwiseInverse();
            logs_  = s.array().log().matrix();
            for (Index j = 0; j < p_; ++j)
                for (Index i = j; i < p_; ++i) A_(i, j) *= inv_s_[i] * inv_s_[j];
            b_.array() *= inv_s_.array();
        } else {
            set_unit_scales();
        }

        const double cond_lim = std::max(rank_tol_, 1e-8);   // normal equations square the condition number
        VectorXd bt;
        if (eff_ == Method::LLT) {
            llt_.compute(A_);
            if (llt_.info() != Eigen::Success) return false;
            const ArrayXd d = llt_.matrixLLT().diagonal().array();
            const double dmax = d.maxCoeff();
            const double dmin = d.minCoeff();
            if (!(dmin > 0.0) || !std::isfinite(dmax)) return false;
            if ((dmin * dmin) / (dmax * dmax) < cond_lim) return false;
            bt = llt_.solve(b_);
            if (!bt.allFinite()) return false;
            logdet_ = 2.0 * d.log().sum() + 2.0 * logs_.sum();
        } else {
            ldlt_.compute(A_);
            if (ldlt_.info() != Eigen::Success || !ldlt_.isPositive()) return false;
            const ArrayXd d = ldlt_.vectorD().array();
            const double dmax = d.maxCoeff();
            const double dmin = d.minCoeff();
            if (!(dmin > 0.0) || !std::isfinite(dmax)) return false;
            if (dmin / dmax < cond_lim) return false;
            bt = ldlt_.solve(b_);
            if (!bt.allFinite()) return false;
            logdet_ = d.log().sum() + 2.0 * logs_.sum();
        }
        beta = bt.cwiseProduct(inv_s_);
        rank_ = p_;
        return true;
    }

    void solve_qr(const Eigen::Ref<const MatrixXd>& X, const Eigen::Ref<const ArrayXd>& w,
                  const Eigen::Ref<const ArrayXd>& tw, Index chunk, VectorXd& beta) {
        if (equil_) {
            VectorXd s = weighted_col_norms(X, w);
            for (Index j = 0; j < p_; ++j) if (!(s[j] > 0.0) || !std::isfinite(s[j])) s[j] = 1.0;
            inv_s_ = s.cwiseInverse();
            logs_  = s.array().log().matrix();
        } else {
            set_unit_scales();
        }
        const MatrixXd Raug = tsqr_augmented(X, w, tw, inv_s_.array(), chunk, scratch_);
        R_ = Raug.topLeftCorner(p_, p_);
        c_ = Raug.col(p_).head(p_);

        VectorXd bt = VectorXd::Zero(p_);
        Method m = used_;
        if (m == Method::HouseholderQR) {
            const ArrayXd d = R_.diagonal().array().abs();
            const double dmax = d.maxCoeff();
            if (std::isfinite(dmax) && d.minCoeff() > rank_tol_ * dmax) {
                bt      = R_.triangularView<Eigen::Upper>().solve(c_);
                rank_   = p_;
                logdet_ = 2.0 * d.log().sum() + 2.0 * logs_.sum();
                used_   = Method::HouseholderQR;
                beta    = bt.cwiseProduct(inv_s_);
                return;
            }
            m = Method::ColPivQR;  // rank deficient -> pivoted path
        }
        used_ = m;

        switch (m) {
            case Method::FullPivQR: {
                fpqr_.compute(R_);
                fpqr_.setThreshold(rank_tol_);
                rank_ = fpqr_.rank();
                qtc_  = fpqr_.matrixQ().adjoint() * c_;
                finish_pivoted(fpqr_, bt);
                break;
            }
            case Method::SVD: {
                svd_compute(svd_, R_);
                const ArrayXd s = svd_.singularValues().array();
                const double cut = (s.size() > 0) ? rank_tol_ * s(0) : 0.0;
                Index r = 0;
                while (r < s.size() && s(r) > cut && s(r) > 0.0) ++r;
                rank_ = r;
                logdet_ = 2.0 * logs_.sum();
                if (r > 0) {
                    const VectorXd coef =
                        ((svd_.matrixU().leftCols(r).transpose() * c_).array() / s.head(r)).matrix();
                    bt = svd_.matrixV().leftCols(r) * coef;
                    logdet_ += 2.0 * s.head(r).log().sum();
                }
                break;
            }
            case Method::ColPivQR:
            default: {
                used_ = Method::ColPivQR;
                cpqr_.compute(R_);
                cpqr_.setThreshold(rank_tol_);
                rank_ = cpqr_.rank();
                qtc_  = cpqr_.householderQ().adjoint() * c_;
                finish_pivoted(cpqr_, bt);
                break;
            }
        }
        beta = bt.cwiseProduct(inv_s_);
    }

    template <class PivotedQR>
    void finish_pivoted(const PivotedQR& qr, VectorXd& bt) {
        const Index r = rank_;
        VectorXd gamma = VectorXd::Zero(p_);
        if (r > 0) {
            gamma.head(r) = qr.matrixQR().topLeftCorner(r, r).template triangularView<Eigen::Upper>()
                                .solve(qtc_.head(r));
        }
        bt = qr.colsPermutation() * gamma;
        const auto& P = qr.colsPermutation();
        for (Index j = r; j < p_; ++j) aliased_.push_back(static_cast<int>(P.indices()[j]));
        std::sort(aliased_.begin(), aliased_.end());
        logdet_ = pivoted_logdet(qr, r);
        for (Index i = 0; i < r; ++i) logdet_ += 2.0 * logs_[P.indices()[i]];
    }
};

}  // namespace glm::detail

#endif  // _GLMFIT_WLS_KERNELS_HPP
