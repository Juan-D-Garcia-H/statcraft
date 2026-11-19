// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Juan Garcia

/// @file partition.hpp
/// @brief Exact block decomposition of the observations from the sparsity pattern of the Z matrices.
///
/// Two units i and j are connected when some Z matrix of some response has a non-zero entry (i, j). In every
/// connected component the covariance couples the units only among themselves, so after a permutation
/// C = blockdiag(C_1, ..., C_B). Every quantity of the estimating equations (quasi-score, Pearson function,
/// sensitivities, variabilities, bias correction, cross variability) is then a sum over the blocks, and each block
/// is processed with the dense kernels at cost O(sum_b m_b^3) instead of O(n^3), and memory O(sum_b m_b^2).
///
/// The result is mathematically identical to the dense computation (the zero entries are exact zeros).
///
/// Components with a single unit are collected into one diagonal batch (processed with the per-observation fast
/// path). Small components are merged into batches of about `target` units: the zero padding between merged
/// components is exact and avoids the per-block overhead. Units keep their increasing order inside each component,
/// which makes the R-compatible cross-block convention identical to the dense one.
///
/// A single connected component (for example a band matrix over one long series) cannot be decomposed; there the
/// exact estimating equations need the dense inverse of C and the engine falls back to the dense path.
#pragma once
#include "linalg.hpp"
#include "types.hpp"
#include <algorithm>
#include <numeric>

namespace mcglm {

/// One block of units with its own (small) specification.
struct Batch {
    std::vector<ResponseSpec> specs;   ///< X, offset, ntrial and Z restricted to the units of the block
    std::vector<int> units;            ///< global unit indices (component by component, ascending inside each)
    bool diag_only = false;            ///< all Z of the block are diagonal (singleton units)
};

/// Decomposition of a problem into independent blocks.
struct Partition {
    int K = 0;                         ///< number of responses
    Eigen::Index n = 0;                ///< number of units
    int target = 32;                   ///< merge target used to build the batches
    bool useful = false;               ///< true when decomposing is worthwhile (at least two batches and some coupling)
    Eigen::Index n_components = 0;     ///< connected components (singletons counted individually)
    Eigen::Index n_singletons = 0;     ///< components with a single unit
    Eigen::Index max_component = 0;    ///< units in the largest connected component
    Eigen::Index n_batches = 0;        ///< number of batches (the restricted problems are built only when `batches` is filled)
    Eigen::Index max_batch = 0;        ///< units in the largest batch
    std::vector<Batch> batches;        ///< empty when the decomposition is not useful and was not forced

    /// Units in the largest batch.
    Eigen::Index max_batch_units() const { return max_batch; }
};

namespace detail {

/// Union-find with path halving and union by size.
struct UnionFind {
    std::vector<int> parent, size;
    Eigen::Index components;
    explicit UnionFind(Eigen::Index n) : parent(uz(n)), size(uz(n), 1), components(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[uz(x)] != x) { parent[uz(x)] = parent[uz(parent[uz(x)])]; x = parent[uz(x)]; }
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (size[uz(a)] < size[uz(b)]) std::swap(a, b);
        parent[uz(b)] = a; size[uz(a)] += size[uz(b)]; --components;
    }
};

} // namespace detail

/// True when some Z of some response is not diagonal.
inline bool has_off_diagonal_z(const std::vector<ResponseSpec>& specs) {
    for (const auto& s : specs) for (const auto& z : s.Z) if (!z.is_diag) return true;
    return false;
}

/// Builds the decomposition.
/// @param specs   response specifications (all with the same n); Z may be diagonal, dense or sparse
/// @param target  merge target in units (>= 1)
/// @param force   build the restricted problems even when the decomposition is not useful (otherwise only the
///                statistics are computed, which avoids expanding a single large block)
/// @throws InvalidInput on inconsistent sizes
/// @throws ResourceError if a forced single block would not fit in memory
inline std::shared_ptr<const Partition> build_partition(const std::vector<ResponseSpec>& specs, int target, bool force = false) {
    if (specs.empty()) throw InvalidInput("at least one response is required");
    if (target < 1) throw InvalidInput("block_target must be >= 1");
    auto part = std::make_shared<Partition>();
    const Eigen::Index n = specs[0].X.rows();
    part->K = toi(specs.size()); part->n = n; part->target = target;

    // ---- connected components of the union of all sparsity patterns
    detail::UnionFind uf(n);
    bool coupled = false;
    for (const auto& s : specs)
        for (const auto& z : s.Z) {
            if (z.is_diag) continue;
            if (z.n() != n) throw InvalidInput("a Z matrix has the wrong size");
            if (z.is_sparse) {
                for (int c = 0; c < z.sparse.outerSize(); ++c)
                    for (SpMat::InnerIterator it(z.sparse, c); it; ++it)
                        if (it.row() != c && it.value() != 0.0) { uf.unite(toi(it.row()), c); coupled = true; }
            } else {
                for (Eigen::Index c = 1; c < n && uf.components > 1; ++c) {
                    const double* col = z.dense.data() + c * n;
                    for (Eigen::Index r = 0; r < c; ++r)
                        if (col[r] != 0.0) { uf.unite(toi(r), toi(c)); coupled = true; }
                }
            }
            if (uf.components == 1) break;
        }
    part->n_components = uf.components;

    // ---- list the members of every component (ascending inside each; components ordered by first member)
    std::vector<int> comp_of(uz(n), -1), comp_size;
    std::vector<int> root_to_comp(uz(n), -1);
    for (Eigen::Index i = 0; i < n; ++i) {
        const int r = uf.find(toi(i));
        if (root_to_comp[uz(r)] < 0) { root_to_comp[uz(r)] = toi(comp_size.size()); comp_size.push_back(0); }
        comp_of[uz(i)] = root_to_comp[uz(r)]; ++comp_size[uz(comp_of[uz(i)])];
    }
    const size_t nc = comp_size.size();
    std::vector<std::vector<int>> members(nc);
    for (size_t c = 0; c < nc; ++c) members[c].reserve(uz(comp_size[c]));
    for (Eigen::Index i = 0; i < n; ++i) members[uz(comp_of[uz(i)])].push_back(toi(i));

    // ---- batches: singletons together, larger components binned up to `target` units
    std::vector<std::vector<int>> batch_units;
    std::vector<bool> batch_diag;
    std::vector<int> singles, cur;
    Eigen::Index max_comp = 0;
    for (size_t c = 0; c < nc; ++c) {
        max_comp = std::max<Eigen::Index>(max_comp, comp_size[c]);
        if (comp_size[c] == 1) { singles.push_back(members[c][0]); ++part->n_singletons; continue; }
        if (!cur.empty() && tix(cur.size()) + comp_size[c] > target) { batch_units.push_back(std::move(cur)); batch_diag.push_back(false); cur.clear(); }
        cur.insert(cur.end(), members[c].begin(), members[c].end());
        if (tix(cur.size()) >= target) { batch_units.push_back(std::move(cur)); batch_diag.push_back(false); cur.clear(); }
    }
    if (!cur.empty()) { batch_units.push_back(std::move(cur)); batch_diag.push_back(false); }
    if (!singles.empty()) { batch_units.push_back(std::move(singles)); batch_diag.push_back(true); }
    part->max_component = max_comp;
    part->useful = coupled && batch_units.size() >= 2;
    part->n_batches = tix(batch_units.size());
    for (const auto& u : batch_units) part->max_batch = std::max<Eigen::Index>(part->max_batch, tix(u.size()));
    if (!part->useful && !force) return part;                 // statistics only
    if (part->max_batch > 2000 && coupled) la::require_dense_memory(tix(specs.size()) * part->max_batch, 6);

    // ---- restricted specifications
    std::vector<int> local_of(uz(n), -1), batch_of(uz(n), -1);
    for (size_t b = 0; b < batch_units.size(); ++b)
        for (size_t c = 0; c < batch_units[b].size(); ++c) {
            local_of[uz(batch_units[b][c])] = toi(c); batch_of[uz(batch_units[b][c])] = toi(b);
        }

    part->batches.resize(batch_units.size());
    for (size_t b = 0; b < batch_units.size(); ++b) {
        Batch& B = part->batches[b];
        B.units = std::move(batch_units[b]); B.diag_only = batch_diag[b];
        const Eigen::Index m = tix(B.units.size());
        B.specs.resize(specs.size());
        for (size_t k = 0; k < specs.size(); ++k) {
            const ResponseSpec& s = specs[k]; ResponseSpec& r = B.specs[k];
            r.link = s.link; r.variance = s.variance; r.covariance = s.covariance; r.power_fixed = s.power_fixed;
            r.X.resize(m, s.X.cols());
            for (Eigen::Index c = 0; c < m; ++c) r.X.row(c) = s.X.row(B.units[uz(c)]);
            if (s.offset.size()) { r.offset.resize(m); for (Eigen::Index c = 0; c < m; ++c) r.offset[c] = s.offset[B.units[uz(c)]]; }
            if (s.ntrial.size()) { r.ntrial.resize(m); for (Eigen::Index c = 0; c < m; ++c) r.ntrial[c] = s.ntrial[B.units[uz(c)]]; }
            else if (s.variance == Variance::BinomialP || s.variance == Variance::BinomialPQ) r.ntrial = VectorXd::Ones(m);
            for (const ZMat& z : s.Z) {
                if (z.is_diag || B.diag_only) {                       // only the diagonal entries can be non-zero here
                    VectorXd d(m);
                    for (Eigen::Index c = 0; c < m; ++c) {
                        const int u = B.units[uz(c)];
                        d[c] = z.is_diag ? z.diag[u] : (z.is_sparse ? z.sparse.coeff(u, u) : z.dense(u, u));
                    }
                    r.Z.push_back(ZMat::diagonal(d));
                    continue;
                }
                MatrixXd blk = MatrixXd::Zero(m, m);
                if (z.is_sparse) {
                    for (Eigen::Index c = 0; c < m; ++c)
                        for (SpMat::InnerIterator it(z.sparse, B.units[uz(c)]); it; ++it) {
                            if (batch_of[uz(it.row())] != toi(b))
                                throw InvalidInput("internal error: an entry of Z couples units of different blocks");
                            blk(local_of[uz(it.row())], c) = it.value();
                        }
                } else {
                    for (Eigen::Index c = 0; c < m; ++c)
                        for (Eigen::Index r2 = 0; r2 < m; ++r2) blk(r2, c) = z.dense(B.units[uz(r2)], B.units[uz(c)]);
                }
                r.Z.push_back(ZMat::from_dense(blk, true));
            }
        }
    }
    return part;
}

} // namespace mcglm
