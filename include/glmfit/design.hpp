// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file design.hpp
 * @brief Design-matrix abstraction for the IRLS solver: dense (Eigen::Map) and sparse (CSC) back ends.
 * @details The solver only needs a handful of operations on X: X*b, X'*v, the weighted least-squares
 *          solve, the Gram matrix (for Levenberg-Marquardt), leverages (Firth), ordered structural rank
 *          detection and row/column sub-setting. Dense designs keep the TSQR / normal-equation kernels;
 *          sparse designs build the Gram matrix with sparse products (O(sum nnz_row^2)) and solve the
 *          small dense p x p system.
 */

#ifndef _GLMFIT_DESIGN_HPP
#define _GLMFIT_DESIGN_HPP

#include "wls_kernels.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>
#include <utility>
#include <vector>

namespace glm::detail {

using SpMat = Eigen::SparseMatrix<double>;   // column-major (CSC)

struct Design {
    virtual ~Design() = default;
    [[nodiscard]] virtual Index rows() const = 0;
    [[nodiscard]] virtual Index cols() const = 0;
    /// out = X b
    virtual void matvec(const VectorXd& b, ArrayXd& out) const = 0;
    /// out = X' v
    virtual void rmatvec(const ArrayXd& v, VectorXd& out) const = 0;
    /// Weighted LS on the weighted target tw (see WlsSolver::solve).
    virtual void wls(WlsSolver& s, const ArrayXd& w, const ArrayXd& tw, Index qr_chunk, Index ne_chunk,
                     VectorXd& beta) const = 0;
    /// Gram matrix A = X'diag(w^2)X (at least the lower triangle) and b = X'(w .* tw).
    virtual void gram(const ArrayXd& w, const ArrayXd& tw, Index ne_chunk, MatrixXd& A, VectorXd& b) const = 0;
    /// h_i = w_i^2 x_i' Ainv x_i
    virtual void leverages(const ArrayXd& w, const MatrixXd& Ainv, Index chunk, ArrayXd& h) const = 0;
    /// Structurally aliased columns (ascending), later duplicates first to go (R convention).
    [[nodiscard]] virtual std::vector<int> aliased(WlsSolver& s, const ArrayXd& w0, Index qr_chunk) const = 0;
    /// Copy restricted to `cols` (and to `rows` when non-null).
    [[nodiscard]] virtual std::unique_ptr<Design> subset(const std::vector<Index>* rows,
                                                         const std::vector<int>& cols) const = 0;
    [[nodiscard]] virtual bool is_sparse() const noexcept = 0;
};

// ------------------------------------------------------------------------------------------------
class DenseDesign final : public Design {
public:
    explicit DenseDesign(const Eigen::Map<const MatrixXd>& X)
        : n_(X.rows()), p_(X.cols()), data_(X.data()) {}
    explicit DenseDesign(MatrixXd&& m) : own_(std::move(m)) {
        n_ = own_.rows(); p_ = own_.cols(); data_ = own_.data();
    }

    [[nodiscard]] Index rows() const override { return n_; }
    [[nodiscard]] Index cols() const override { return p_; }
    [[nodiscard]] bool is_sparse() const noexcept override { return false; }

    void matvec(const VectorXd& b, ArrayXd& out) const override { out.matrix().noalias() = M() * b; }
    void rmatvec(const ArrayXd& v, VectorXd& out) const override { out.noalias() = M().transpose() * v.matrix(); }
    void wls(WlsSolver& s, const ArrayXd& w, const ArrayXd& tw, Index qr_chunk, Index ne_chunk,
             VectorXd& beta) const override {
        s.solve(M(), w, tw, qr_chunk, ne_chunk, beta);
    }
    void gram(const ArrayXd& w, const ArrayXd& tw, Index ne_chunk, MatrixXd& A, VectorXd& b) const override {
        normal_equations(M(), w, tw, ne_chunk, A, b);
        for (Index j = 0; j < p_; ++j) for (Index i = 0; i < j; ++i) A(i, j) = A(j, i);
    }
    void leverages(const ArrayXd& w, const MatrixXd& Ainv, Index chunk, ArrayXd& h) const override {
        detail::leverages(M(), w, Ainv, chunk, h);
    }
    [[nodiscard]] std::vector<int> aliased(WlsSolver& s, const ArrayXd& w0, Index qr_chunk) const override {
        return s.detect_aliased_ordered(M(), w0, qr_chunk);
    }
    [[nodiscard]] std::unique_ptr<Design> subset(const std::vector<Index>* rows,
                                                 const std::vector<int>& cols) const override {
        const Index nr = rows ? static_cast<Index>(rows->size()) : n_;
        const Index nc = static_cast<Index>(cols.size());
        MatrixXd out(nr, nc);
        const auto X = M();
        for (Index j = 0; j < nc; ++j) {
            const auto src = X.col(cols[static_cast<std::size_t>(j)]);
            if (rows) for (Index i = 0; i < nr; ++i) out(i, j) = src[(*rows)[static_cast<std::size_t>(i)]];
            else      out.col(j) = src;
        }
        return std::make_unique<DenseDesign>(std::move(out));
    }

private:
    Index n_ = 0, p_ = 0;
    const double* data_ = nullptr;
    MatrixXd own_;
    [[nodiscard]] Eigen::Map<const MatrixXd> M() const { return Eigen::Map<const MatrixXd>(data_, n_, p_); }
};

// ------------------------------------------------------------------------------------------------
class SparseDesign final : public Design {
public:
    /// Non-owning view: `X` must outlive the solver.
    explicit SparseDesign(const SpMat& X) : ext_(&X) {}
    explicit SparseDesign(SpMat&& X) : own_(std::move(X)) { own_.makeCompressed(); }

    [[nodiscard]] Index rows() const override { return X().rows(); }
    [[nodiscard]] Index cols() const override { return X().cols(); }
    [[nodiscard]] bool is_sparse() const noexcept override { return true; }

    void matvec(const VectorXd& b, ArrayXd& out) const override { out.matrix() = X() * b; }
    void rmatvec(const ArrayXd& v, VectorXd& out) const override { out = X().transpose() * v.matrix(); }

    void gram(const ArrayXd& w, const ArrayXd& tw, Index, MatrixXd& A, VectorXd& b) const override {
        SpMat Xw = w.matrix().asDiagonal() * X();
        const SpMat G = Xw.transpose() * Xw;
        A = MatrixXd(G);
        b = Xw.transpose() * tw.matrix();
    }
    void wls(WlsSolver& s, const ArrayXd& w, const ArrayXd& tw, Index, Index ne_chunk,
             VectorXd& beta) const override {
        MatrixXd A; VectorXd b;
        gram(w, tw, ne_chunk, A, b);
        s.solve_gram(std::move(A), std::move(b), beta);
    }
    void leverages(const ArrayXd& w, const MatrixXd& Ainv, Index, ArrayXd& h) const override {
        const Eigen::SparseMatrix<double, Eigen::RowMajor> R(X());
        const Index n = R.rows();
        h.resize(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 4096)
#endif
        for (Index i = 0; i < n; ++i) {
            double s = 0.0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator a(R, i); a; ++a)
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator b(R, i); b; ++b)
                    s += a.value() * Ainv(a.col(), b.col()) * b.value();
            h[i] = w[i] * w[i] * s;
        }
    }
    [[nodiscard]] std::vector<int> aliased(WlsSolver& s, const ArrayXd& w0, Index) const override {
        MatrixXd G; VectorXd b;
        gram(w0, ArrayXd::Zero(rows()), 0, G, b);
        return s.detect_aliased_gram(G);
    }
    [[nodiscard]] std::unique_ptr<Design> subset(const std::vector<Index>* rows,
                                                 const std::vector<int>& cols) const override {
        const SpMat& S = X();
        std::vector<int> rmap;
        Index nr = S.rows();
        if (rows) {
            rmap.assign(static_cast<std::size_t>(S.rows()), -1);
            for (std::size_t k = 0; k < rows->size(); ++k) rmap[static_cast<std::size_t>((*rows)[k])] = static_cast<int>(k);
            nr = static_cast<Index>(rows->size());
        }
        std::vector<Eigen::Triplet<double>> t;
        t.reserve(static_cast<std::size_t>(S.nonZeros()));
        for (std::size_t k = 0; k < cols.size(); ++k)
            for (SpMat::InnerIterator it(S, cols[k]); it; ++it) {
                const Index r = rows ? rmap[static_cast<std::size_t>(it.row())] : it.row();
                if (r >= 0) t.emplace_back(static_cast<int>(r), static_cast<int>(k), it.value());
            }
        SpMat out(nr, static_cast<Index>(cols.size()));
        out.setFromTriplets(t.begin(), t.end());
        return std::make_unique<SparseDesign>(std::move(out));
    }

private:
    const SpMat* ext_ = nullptr;
    SpMat own_;
    [[nodiscard]] const SpMat& X() const { return ext_ ? *ext_ : own_; }
};

}  // namespace glm::detail

#endif  // _GLMFIT_DESIGN_HPP
