// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/**
 * @file covmodel.hpp
 * @brief Covariance structures for random effects.
 *
 * Given a symmetric 0/1 pattern indicating which pairs have a free covariance,
 * the class builds the connected components and parameterizes each component:
 *
 *  - FULL (all pairs free, e.g. a dense block or a single parameter):
 *    Cholesky parameterization, the log of the diagonal plus the off-diagonal
 *    elements. The matrix is positive definite by construction and smooth in
 *    @f$\theta@f$.
 *
 *  - PARTIAL (arbitrary pattern with zeros inside the component):
 *    @f$\Omega = D R D@f$, with @f$D = \operatorname{diag}(\exp(\log sd))@f$
 *    and @f$R@f$ having ones on the diagonal and free correlations on the pairs
 *    selected by the pattern. The closed-form M-step projects the sample
 *    covariance onto the pattern and, if @f$R@f$ is not positive definite,
 *    shrinks the correlations until it is.
 *
 * Requires: Eigen3, C++20, model.hpp (@c decodeCov/@c encodeCov).
 */

#pragma once
#include "numcast.hpp"

#include "model.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace mcmckl {

/**
 * @brief Parameterization of a structured random-effects covariance matrix.
 *
 * The class is immutable after construction except for the parameter vectors
 * passed to its encode/decode helpers. Every component is parameterized
 * independently, which keeps the parameter layout block-diagonal.
 */
class CovModel {
public:
    /**
     * @brief One connected component of the covariance pattern.
     */
    struct Comp {
        std::vector<int> members;                  ///< Global positions (0..q-1).
        bool chol = true;                          ///< True for a full (Cholesky) block.
        std::vector<std::pair<int,int>> pairs;     ///< Free local pairs (a<b) when @c !chol.
        int off = 0, nPar = 0;                     ///< Offset relative to the block and parameter count.
    };

    CovModel() = default;

    /**
     * @brief Builds the model from a symmetric covariance pattern.
     * @param pat @f$q \times q@f$ pattern; a non-zero off-diagonal entry means
     *        that the corresponding covariance is free. The diagonal is always
     *        free.
     */
    explicit CovModel(const std::vector<std::vector<char>>& pat) : q_(static_cast<int>(pat.size())) {
        std::vector<int> par(uz(q_)); std::iota(par.begin(), par.end(), 0);
        std::function<int(int)> find = [&](int x) { return par[uz(x)] == x ? x : par[uz(x)] = find(par[uz(x)]); };
        for (int a = 0; a < q_; ++a) for (int b = a + 1; b < q_; ++b)
            if (pat[uz(a)][uz(b)] || pat[uz(b)][uz(a)]) par[uz(find(a))] = find(b);
        std::vector<int> compIdx(uz(q_), -1);
        compOf_.assign(uz(q_), -1);
        for (int l = 0; l < q_; ++l) {
            const int r = find(l);
            if (compIdx[uz(r)] < 0) { compIdx[uz(r)] = static_cast<int>(comps_.size()); comps_.emplace_back(); }
            comps_[uz(compIdx[uz(r)])].members.push_back(l);
            compOf_[uz(l)] = compIdx[uz(r)];
        }
        nPar_ = 0;
        for (auto& c : comps_) {
            const int m = static_cast<int>(c.members.size());
            for (int a = 0; a < m; ++a) for (int b = a + 1; b < m; ++b)
                if (pat[uz(c.members[uz(a)])][uz(c.members[uz(b)])] || pat[uz(c.members[uz(b)])][uz(c.members[uz(a)])]) c.pairs.emplace_back(a, b);
            c.chol = (static_cast<int>(c.pairs.size()) == m * (m - 1) / 2);
            c.off = nPar_;
            c.nPar = c.chol ? m * (m + 1) / 2 : m + static_cast<int>(c.pairs.size());
            nPar_ += c.nPar;
        }
    }

    /**
     * @brief Builds an independent (diagonal) covariance model.
     * @param q Number of random effects.
     * @return A model with @f$q@f$ singleton components.
     */
    static CovModel diagonal(int q) { return CovModel(std::vector<std::vector<char>>(uz(q), std::vector<char>(uz(q), 0))); }

    /**
     * @brief Builds a block-diagonal model from component identifiers.
     * @param id Component identifier for each random effect; equal identifiers
     *        form a full block.
     * @return The corresponding covariance model.
     */
    static CovModel fromBlocks(const std::vector<int>& id) {
        const int q = static_cast<int>(id.size());
        std::vector<std::vector<char>> pat(uz(q), std::vector<char>(uz(q), 0));
        for (int a = 0; a < q; ++a) for (int b = 0; b < q; ++b) pat[uz(a)][uz(b)] = (a != b && id[uz(a)] == id[uz(b)]);
        return CovModel(pat);
    }

    /** @brief Returns the number of random effects. */
    int q() const { return q_; }
    /** @brief Returns the number of covariance parameters. */
    int nParams() const { return nPar_; }
    /** @brief Returns the connected components. */
    const std::vector<Comp>& comps() const { return comps_; }
    /** @brief Returns the component index of random effect @p l. */
    int compOf(int l) const { return compOf_[uz(l)]; }
    /** @brief Returns true when random effect @p l is a singleton component. */
    bool singleton(int l) const { return comps_[uz(compOf_[uz(l)])].members.size() == 1; }
    /** @brief Returns the relative index of the log-sd parameter of effect @p l. */
    int singletonParam(int l) const { return comps_[uz(compOf_[uz(l)])].off; }

    /**
     * @brief Decodes a covariance matrix from the parameter vector.
     * @param th Full parameter vector.
     * @param off Offset of the covariance block in @p th.
     * @return The @f$q \times q@f$ covariance matrix @f$\Omega@f$.
     */
    Mat decode(const Vec& th, int off) const {
        Mat Om = Mat::Zero(q_, q_);
        for (const auto& c : comps_) {
            const int m = static_cast<int>(c.members.size());
            Mat B;
            if (c.chol) B = decodeCov(th, off + c.off, m, true);
            else {
                Vec sd(m);
                for (int a = 0; a < m; ++a) sd[a] = std::exp(th[off + c.off + a]);
                B = Mat::Zero(m, m);
                for (int a = 0; a < m; ++a) B(a, a) = sd[a] * sd[a];
                for (size_t k = 0; k < c.pairs.size(); ++k) {
                    const auto [a, b] = c.pairs[k];
                    const double v = th[off + c.off + m + tix(k)] * sd[a] * sd[b];
                    B(a, b) = v; B(b, a) = v;
                }
            }
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) Om(c.members[uz(a)], c.members[uz(b)]) = B(a, b);
        }
        return Om;
    }

    /**
     * @brief Encodes initial standard deviations with zero correlations.
     * @param th Parameter vector to fill.
     * @param off Offset of the covariance block in @p th.
     * @param sd Standard deviation of each random effect.
     */
    void encodeSd(Vec& th, int off, const Vec& sd) const {
        for (const auto& c : comps_) {
            const int m = static_cast<int>(c.members.size());
            if (c.chol) {
                Mat B = Mat::Zero(m, m);
                for (int a = 0; a < m; ++a) B(a, a) = sd[c.members[uz(a)]] * sd[c.members[uz(a)]];
                encodeCov(th, off + c.off, B, m, true);
            } else {
                for (int a = 0; a < m; ++a) th[off + c.off + a] = std::log(std::max(sd[c.members[uz(a)]], 1e-12));
                for (size_t k = 0; k < c.pairs.size(); ++k) th[off + c.off + m + tix(k)] = 0.0;
            }
        }
    }

    /**
     * @brief Closed-form M-step: projects a sample covariance onto the structure.
     *
     * For full blocks the projection uses an eigendecomposition with a positive
     * eigenvalue floor. For partial blocks it projects correlations and shrinks
     * them by bisection until the correlation matrix is positive definite.
     *
     * @param th Parameter vector to fill.
     * @param off Offset of the covariance block in @p th.
     * @param S Sample covariance matrix (@f$q \times q@f$).
     * @param skip When entry @c l is non-zero, the singleton component of
     *        random effect @c l is left unchanged (fixed standard deviation).
     */
    void encodeFromMatrix(Vec& th, int off, const Mat& S, const std::vector<char>& skip) const {
        for (const auto& c : comps_) {
            const int m = static_cast<int>(c.members.size());
            if (m == 1 && !skip.empty() && skip[uz(c.members[0])]) continue;
            Mat sub(m, m);
            for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) sub(a, b) = S(c.members[uz(a)], c.members[uz(b)]);
            sub = 0.5 * (sub + sub.transpose());
            if (c.chol) {
                Eigen::SelfAdjointEigenSolver<Mat> es(sub);
                Vec ev = es.eigenvalues();
                const double fl = 1e-10 * std::max(1.0, ev.cwiseAbs().maxCoeff());
                sub = es.eigenvectors() * ev.cwiseMax(fl).asDiagonal() * es.eigenvectors().transpose();
                encodeCov(th, off + c.off, sub, m, true);
            } else {
                Vec sd(m);
                for (int a = 0; a < m; ++a) { sd[a] = std::sqrt(std::max(sub(a, a), 1e-20)); th[off + c.off + a] = std::log(sd[a]); }
                Vec rho(c.pairs.size());
                for (size_t k = 0; k < c.pairs.size(); ++k) {
                    const auto [a, b] = c.pairs[k];
                    rho[tix(k)] = std::clamp(sub(a, b) / (sd[a] * sd[b]), -0.999, 0.999);
                }
                // Shrink correlations until R is positive definite.
                auto pd = [&](double t) {
                    Mat R = Mat::Identity(m, m);
                    for (size_t k = 0; k < c.pairs.size(); ++k) { R(c.pairs[k].first, c.pairs[k].second) = t * rho[tix(k)]; R(c.pairs[k].second, c.pairs[k].first) = t * rho[tix(k)]; }
                    Eigen::SelfAdjointEigenSolver<Mat> es(R);
                    return es.eigenvalues().minCoeff() > 1e-6;
                };
                double t = 1.0;
                if (!pd(1.0)) { double lo = 0.0, hi = 1.0; for (int it = 0; it < 40; ++it) { t = 0.5 * (lo + hi); if (pd(t)) lo = t; else hi = t; } t = lo; }
                for (size_t k = 0; k < c.pairs.size(); ++k) th[off + c.off + m + tix(k)] = t * rho[tix(k)];
            }
        }
    }

    /**
     * @brief Returns the indices of the log-sd parameters.
     *
     * These indices are used by the simulated-annealing variance floor.
     *
     * @param off Offset of the covariance block in the parameter vector.
     * @return Absolute parameter indices of the log standard deviations.
     */
    std::vector<int> varianceIndices(int off) const {
        std::vector<int> idx;
        for (const auto& c : comps_) {
            const int m = static_cast<int>(c.members.size());
            if (c.chol) { const auto v = covDiagIndices(off + c.off, m, true); idx.insert(idx.end(), v.begin(), v.end()); }
            else for (int a = 0; a < m; ++a) idx.push_back(off + c.off + a);
        }
        return idx;
    }

private:
    int q_ = 0, nPar_ = 0;
    std::vector<Comp> comps_;
    std::vector<int> compOf_;
};

} // namespace mcmckl
