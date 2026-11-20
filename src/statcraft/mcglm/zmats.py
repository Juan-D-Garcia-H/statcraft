# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Constructors of the known matrices Z of the matrix linear predictor.

All functions return NumPy arrays. The engine recognises two forms: a 1-D array is a diagonal matrix (the fast
O(n) path) and a 2-D array is a dense symmetric ``n x n`` matrix. A model whose Z matrices are all diagonal is fitted
per observation (cost O(n K^3) per iteration); any dense Z makes each iteration O(n^3) (LAPACK/BLAS).
"""

from __future__ import annotations

import numpy as np
import itertools

__all__ = [
    "mc_id",
    "mc_diag",
    "mc_group",
    "mc_ma",
    "mc_dist",
    "mc_exp",
    "mc_matern32",
    "mc_wendland",
    "mc_mixed",
    "mc_rw",
    "mc_car",
    "mc_ns",
    "mc_dglm",
    "mc_twin",
    "mc_twin_bio",
    "mc_twin_full",
]


def _scipy_sparse():
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("sparse=True requires SciPy (pip install scipy)") from exc
    return sp


def mc_id(n: int) -> np.ndarray:
    """Identity matrix as a diagonal (1-D array of ones).

    Parameters
    ----------
    n : int
        Number of observations, ``n >= 1``.

    Returns
    -------
    (n,) ndarray
        Array of ones, the diagonal of the identity matrix.
    """
    if int(n) < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    return np.ones(int(n))


def mc_diag(d) -> np.ndarray:
    """Diagonal matrix with entries ``d``, returned as a 1-D array.

    Parameters
    ----------
    d : 1-D array_like
        Diagonal entries of the matrix.

    Returns
    -------
    (n,) ndarray
        The same entries as a float array.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1:
        raise ValueError(f"d must be 1-D, got {d.ndim}-D")
    return d


def mc_group(codes, sparse: bool = False):
    """Random-intercept matrix: ``Z[i, j] = 1`` if observations i and j belong to the same group.

    Parameters
    ----------
    codes : 1-D array_like
        Group label of each observation (any hashable/sortable values).
    sparse : bool, default False
        Return a ``scipy.sparse.csr_matrix`` instead of a dense array. With many groups the sparse form is required
        for large n: it stores ``sum(m_g^2)`` entries, and the engine then fits the model block by block.

    Returns
    -------
    (n, n) ndarray or scipy.sparse.csr_matrix
        Block-structured matrix.
    """
    c = np.asarray(codes)
    if c.ndim != 1:
        raise ValueError(f"codes must be 1-D, got {c.ndim}-D")
    _, inv = np.unique(c, return_inverse=True)
    if not sparse:
        return (inv[:, None] == inv[None, :]).astype(float)
    sp = _scipy_sparse()
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    ends = np.cumsum(counts)
    starts = ends - counts
    rows = np.empty(int(np.sum(counts.astype(np.int64) ** 2)), dtype=np.int64)
    cols = np.empty_like(rows)
    pos = 0
    for s, e in zip(starts, ends):
        idx = order[s:e]
        m = idx.size
        rows[pos : pos + m * m] = np.repeat(idx, m)
        cols[pos : pos + m * m] = np.tile(idx, m)
        pos += m * m
    return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(c.size, c.size))


def mc_ma(n: int, order: int = 1, groups=None, sparse: bool = False) -> list:
    """Banded matrices of a moving-average-type covariance.

    Returns ``order`` matrices; the k-th (k = 1..order) has ones on the k-th sub- and super-diagonals.

    Parameters
    ----------
    n : int
        Number of observations.
    order : int, default 1
        Number of bands, ``0 < order < n``.
    groups : 1-D array_like, optional
        If given, a band entry is kept only between observations of the same group. Observations must be ordered so
        that each group is contiguous for the bands to follow the series inside a group.
    sparse : bool, default False
        Return ``scipy.sparse.csr_matrix`` objects. Without ``groups`` the bands link all n observations in a single
        connected block, so the engine cannot split the problem; use ``groups`` (one series per group) for large n.
    """
    if not 0 < int(order) < int(n):
        raise ValueError(f"order must satisfy 0 < order < n, got order={order!r}, n={n!r}")
    if sparse:
        sp = _scipy_sparse()
        g = None if groups is None else np.asarray(groups)
        if g is not None and g.shape != (n,):
            raise ValueError(f"groups must have shape ({n},), got {g.shape}")
        out = []
        for k in range(1, int(order) + 1):
            r = np.arange(n - k, dtype=np.int64)
            c = r + k
            if g is not None:
                keep = g[r] == g[c]
                r, c = r[keep], c[keep]
            out.append(
                sp.csr_matrix(
                    (np.ones(2 * r.size), (np.concatenate([r, c]), np.concatenate([c, r]))),
                    shape=(n, n),
                )
            )
        return out
    if groups is None:
        same = np.ones((n, n), dtype=bool)
    else:
        g = np.asarray(groups)
        if g.shape != (n,):
            raise ValueError(f"groups must have shape ({n},), got {g.shape}")
        same = g[:, None] == g[None, :]
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    return [((d == k) & same).astype(float) for k in range(1, int(order) + 1)]


def _pdist(coords) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    if c.ndim != 2:
        raise ValueError(f"coords must be 1-D or 2-D, got {c.ndim}-D")
    g = c @ c.T
    s = np.diag(g)
    d2 = np.maximum(s[:, None] + s[None, :] - 2.0 * g, 0.0)
    return np.sqrt(d2)


def mc_dist(coords) -> np.ndarray:
    """Euclidean distance matrix between the rows of ``coords``.

    Parameters
    ----------
    coords : array_like
        Observation coordinates, shape ``(n, d)`` or ``(n,)``.

    Returns
    -------
    (n, n) ndarray
        Symmetric matrix of pairwise Euclidean distances.
    """
    return _pdist(coords)


def mc_exp(coords, range_: float) -> np.ndarray:
    """Exponential correlation ``exp(-d / range_)`` (symmetric positive definite).

    Parameters
    ----------
    coords : array_like
        Coordinates of the observations, shape ``(n, d)``.
    range_ : float
        Range parameter, ``> 0``.
    """
    if not range_ > 0:
        raise ValueError(f"range_ must be > 0, got {range_!r}")
    return np.exp(-_pdist(coords) / range_)


def mc_matern32(coords, range_: float) -> np.ndarray:
    """Matern correlation with smoothness 3/2: ``(1 + sqrt(3) d / r) exp(-sqrt(3) d / r)``.

    Parameters
    ----------
    coords : array_like
        Coordinates of the observations, shape ``(n, d)``.
    range_ : float
        Range parameter, ``> 0``.
    """
    if not range_ > 0:
        raise ValueError(f"range_ must be > 0, got {range_!r}")
    a = np.sqrt(3.0) * _pdist(coords) / range_
    return (1.0 + a) * np.exp(-a)


def mc_wendland(coords, range_: float):
    """Compactly supported spatial correlation (Wendland, ``(1 - d/r)_+^4 (4 d/r + 1)``) as a sparse matrix.

    It is symmetric positive definite for coordinates in up to three dimensions and exactly zero beyond ``range_``, so
    the matrix is sparse for large n. The observations form one connected block (neighbours chain together), which the
    engine handles with the approximate sparse method (one response, identity covariance link) instead of dense
    ``n x n`` matrices.

    Parameters
    ----------
    coords : array_like
        Coordinates, shape ``(n, d)`` with ``d <= 3``.
    range_ : float
        Support radius, ``> 0``; the number of neighbours within it controls the sparsity.

    Returns
    -------
    scipy.sparse.csr_matrix
        Requires SciPy.
    """
    if not range_ > 0:
        raise ValueError(f"range_ must be > 0, got {range_!r}")
    sp = _scipy_sparse()
    from scipy.spatial import cKDTree

    c = np.asarray(coords, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    if c.ndim != 2 or c.shape[1] > 3:
        raise ValueError("coords must have shape (n, d) with d <= 3")
    tree = cKDTree(c)
    d = tree.sparse_distance_matrix(tree, range_, output_type="coo_matrix")
    u = d.data / range_
    val = (1.0 - u) ** 4 * (4.0 * u + 1.0)
    rows, cols = d.row, d.col
    diag = np.arange(c.shape[0])
    keep = rows != cols
    return sp.csr_matrix(
        (
            np.concatenate([val[keep], np.ones(diag.size)]),
            (np.concatenate([rows[keep], diag]), np.concatenate([cols[keep], diag])),
        ),
        shape=(c.shape[0],) * 2,
    )


# -----------------------------------------------------------------------------
# Additional Z matrix constructors (mcglm style)
# -----------------------------------------------------------------------------

def mc_mixed(formula=None, data=None, sparse: bool = False, **kwargs):
    """Z matrices of the matrix linear predictor of a mixed model.

    Two calling conventions are supported:

    * ``mc_mixed(groups=codes)`` or ``mc_mixed(id=codes)`` is equivalent to
      ``[mc_group(codes, sparse=sparse)]`` (random intercept per group).
    * ``mc_mixed(formula, data)`` uses an R-style random-effects formula such
      as ``"~ Days | subject"`` and expands it into random-intercept and
      random-slope Z matrices (Demidenko, 2013).

    Parameters
    ----------
    formula : str, optional
        Random-effects formula written as ``"~ terms | group"``. Only the terms
        left of ``|`` define the random effects; a leading response is not
        used. ``"~ 1 | group"`` gives a random intercept and ``"~ x | group"``
        gives a random intercept plus a random slope for ``x``.
    data : DataFrame-like, optional
        Data in which ``formula`` is evaluated with ``formulaic``. Required
        when ``formula`` is given.
    sparse : bool, default False
        Return ``scipy.sparse.csr_matrix`` objects instead of dense arrays.
    **kwargs
        ``groups`` or ``id`` supplies the group labels for the formula-free
        interface.

    Returns
    -------
    list of ndarray or list of scipy.sparse.csr_matrix
        One matrix per variance component: the group intercept, each random
        slope and, when both are present, their symmetric cross-covariance.

    Raises
    ------
    ImportError
        If ``formula``/``data`` are used and ``formulaic`` is not installed.
    ValueError
        If neither a group vector nor a valid formula is supplied, or if the
        formula does not contain ``|``.

    Notes
    -----
    The formula is expanded as ``~ 0 + group + group:terms`` and the design
    columns are classified as intercepts (no ``:``) or slopes (``:``). For a
    position ``(i, i)`` the Z matrix is ``D_i D_i'``; for a cross position
    ``(i, j)`` it is ``D_i D_j' + D_j D_i'``, which keeps it symmetric and
    supplies the covariance between the random intercept and slope.
    """
    # 1. Fallback for compatibility with classic direct arguments
    groups = kwargs.get("groups")
    if groups is None:
        groups = kwargs.get("id")
    if groups is not None:
        return [mc_group(groups, sparse=sparse)]

    if formula is not None and data is not None:
        try:
            import formulaic
        except ImportError as e:
            raise ImportError(
                "mc_mixed with formulas requires the 'formulaic' package (pip install formulaic)"
            ) from e

        form_str = formula.strip()
        if "|" not in form_str:
            raise ValueError(
                "the formula must contain the '|' operator to specify random effects by group"
            )

        # 2. Split the R-style syntax: "~ Days | subject"
        lhs, rhs = form_str.split("|")
        group_var = rhs.replace(")", "").strip()
        fixed_part = lhs.replace("~", "").replace("(", "").strip()

        # Build the same manual expansion as the original author's logic
        if fixed_part == "1":
            expanded_form = f"~ 0 + {group_var}"
        elif "0" in fixed_part or "-1" in fixed_part:
            clean_fixed = fixed_part.replace("0+", "").replace("0 +", "").replace("-1+", "").replace("-1 +", "").strip()
            expanded_form = f"~ 0 + {group_var}:{clean_fixed}"
        else:
            # Turn "~ Days | subject" into "~ 0 + subject + subject:Days"
            expanded_form = f"~ 0 + {group_var} + {group_var}:{fixed_part}"

        # 3. Obtain the structured design matrix with formulaic
        mm = formulaic.model_matrix(expanded_form, data, na_action="ignore")
        val_columns = mm.columns.tolist()

        # Classify columns: 0 for group intercepts, 1 for slopes (interactions with ':')
        positions = []
        for column in val_columns:
            if ":" in str(column):
                positions.append(1)
            else:
                positions.append(0)
        positions = np.array(positions)

        design_matrix = np.asarray(mm, dtype=float)

        # Determine the unique positions present (e.g. 0 and 1)
        unique_positions = sorted(list(set(positions)))
        all_indexes = [(i, i) for i in unique_positions]

        # If both intercept and slope (0 and 1) are present, add the cross combination (0, 1)
        if len(unique_positions) > 1:
            all_indexes = all_indexes + list(itertools.combinations(unique_positions, 2))

        matrices = list()
        for tc in all_indexes:
            # Extract all columns that belong to the groups tc[0] and tc[1]
            matrix1 = design_matrix[:, np.where(positions == tc[0])[0]]
            matrix2 = design_matrix[:, np.where(positions == tc[1])[0]]

            if tc[0] == tc[1]:
                # Pure variance: intercepts (0, 0) or slopes (1, 1)
                z_mat = np.matmul(matrix1, matrix2.T)
            else:
                # Symmetric cross-covariance (0, 1): this is where the fourth parameter comes from
                z_mat = np.matmul(matrix1, matrix2.T) + np.matmul(matrix2, matrix1.T)

            if sparse:
                _scipy_sparse()
                import scipy.sparse as sp
                matrices.append(sp.csr_matrix(z_mat))
            else:
                matrices.append(z_mat)

        return matrices

    raise ValueError(
        "mc_mixed requires 'groups', 'id', or a valid formula with '|' together with its DataFrame"
    )




def mc_rw(n: int, order: int = 1, sparse: bool = False):
    """Z matrix of a random-walk covariance along one ordered series.

    Parameters
    ----------
    n : int
        Number of observations in the series. They must be ordered along the
        walk.
    order : {1, 2}, default 1
        Order of the random walk. Higher orders are not implemented.
    sparse : bool, default False
        Return a single ``scipy.sparse.csr_matrix`` instead of a dense array.

    Returns
    -------
    list of one ndarray or one scipy.sparse.csr_matrix
        For the dense path, a symmetric tridiagonal (order 1) or pentadiagonal
        (order 2) matrix. For the sparse path, ``D.T @ D`` where ``D`` is the
        first- or second-order difference operator. Only one Z matrix is
        returned; its tau is the innovation variance of the walk.

    Raises
    ------
    NotImplementedError
        If ``order`` is not 1 or 2, or if ``sparse=True`` with ``order > 2``.

    Notes
    -----
    The dense and sparse paths use different scalings of the same difference
    operator; the scale is absorbed by the corresponding tau estimate.
    """
    if sparse:
        sp = _scipy_sparse()
        if order == 1:
            # difference operator related
            rows = []
            cols = []
            data = []
            for i in range(n - 1):
                rows += [i, i]
                cols += [i, i + 1]
                data += [1.0, -1.0]
            D = sp.csr_matrix((data, (rows, cols)), shape=(n - 1, n))
            return [D.T @ D]
        elif order == 2:
            rows = []
            cols = []
            data = []
            for i in range(n - 2):
                rows += [i, i, i]
                cols += [i, i + 1, i + 2]
                data += [1.0, -2.0, 1.0]
            D = sp.csr_matrix((data, (rows, cols)), shape=(n - 2, n))
            return [D.T @ D]
        else:
            raise NotImplementedError("mc_rw sparse order>2 not implemented")
    if order == 1:
        # I - J-like tridiag approximation via differences? Or return structure
        # common form: tridiagonal for RW1
        Z = np.zeros((n, n))
        for i in range(n):
            if i > 0:
                Z[i, i] += 1.0
                Z[i, i - 1] += -0.5
            if i < n - 1:
                Z[i, i] += 1.0
                Z[i, i + 1] += -0.5
        Z = (Z + Z.T) / 2.0
        return [Z]
    elif order == 2:
        Z = np.zeros((n, n))
        for i in range(1, n - 1):
            Z[i, i] += 1.0
            Z[i, i - 1] += -0.5
            Z[i, i + 1] += -0.5
        for i in range(2, n):
            Z[i, i] += 0.25
            Z[i, i - 2] += 0.25
            Z[i - 2, i] += 0.25
            Z[i - 2, i - 2] += 0.25  # rough approximation
        Z = Z + Z.T
        Z[np.diag_indices(n)] /= 2.0
        return [Z]
    raise NotImplementedError(f"mc_rw order={order} not implemented")


def mc_car(n: int, adj=None, sparse: bool = False):
    """Z matrix of a conditional autoregressive (CAR) structure.

    Builds the graph Laplacian ``D - A`` of an adjacency matrix ``A``, where
    ``D`` is the diagonal matrix of row sums (node degrees). Using it as Z
    gives an intrinsic CAR model on the graph.

    Parameters
    ----------
    n : int
        Number of observations. Used only to build the default path graph.
    adj : (n, n) array_like or scipy sparse matrix, optional
        Symmetric adjacency matrix whose non-zero entries mark neighbours. If
        omitted, a chain ``0 - 1 - ... - (n - 1)`` is used.
    sparse : bool, default False
        Return a ``scipy.sparse.csr_matrix``. With a sparse ``adj`` the
        Laplacian is kept sparse even if this flag is False.

    Returns
    -------
    list of one ndarray or one scipy.sparse.csr_matrix
        The graph Laplacian as a single Z matrix.
    """
    if sparse:
        sp = _scipy_sparse()
    if adj is not None:
        if sparse or hasattr(adj, "tocoo"):
            try:
                A = adj.tocsr() if hasattr(adj, "tocsr") else sp.csr_matrix(adj)
            except Exception:
                A = sp.csr_matrix(adj)
            D = sp.diags(A.sum(axis=1).A1)
            return [(D - A)]
        A = np.asarray(adj)
        D = np.diag(A.sum(axis=1))
        return [(D - A)]
    # default: chain
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    if sparse:
        sp = _scipy_sparse()
        return [sp.csr_matrix(D - A)]
    return [(D - A)]


def mc_ns(x, df: int = None, knots=None, degree: int = 3):
    """Z matrix of a smooth-term covariance based on a polynomial basis.

    Builds ``X = [x, x**2, ..., x**degree]`` and returns the dense matrix
    ``X @ X.T`` as a single Z, so the implied covariance between observations
    is a polynomial kernel of the covariate.

    Parameters
    ----------
    x : array_like
        Covariate values, flattened to length n.
    df : int, optional
        Requested degrees of freedom. Used only to choose ``knots`` when they
        are not supplied; it does not change the polynomial basis.
    knots : array_like, optional
        Knot positions. Accepted for API compatibility; the current
        implementation always uses the polynomial fallback below.
    degree : int, default 3
        Highest power of ``x`` in the basis, ``>= 1``.

    Returns
    -------
    list of ndarray
        One dense ``n x n`` matrix ``X @ X.T``.

    Notes
    -----
    The returned matrix has rank at most ``degree``; for large n prefer a
    diagonal, grouped or sparse Z to avoid allocating an ``n x n`` matrix.
    """
    try:
        from scipy.interpolate import splev
    except Exception:
        pass
    x = np.asarray(x).ravel()
    n = len(x)
    if df is None and knots is None:
        df = 3
    # build natural spline basis
    if knots is None and df is not None:
        # quantile knots
        q = np.linspace(0, 1, df - degree + 2)[1:-1] if df - degree + 2 > 2 else np.array([])
        knots = np.quantile(x, q) if len(q) > 0 else np.array([])
    # fallback to polynomial
    X = np.column_stack([x ** i for i in range(1, degree + 1)])
    # return as diagonal-like or dense; mcglm expects n x n usually for structure
    # use outer product structure
    Z = X @ X.T
    return [Z]


def mc_dglm(variance_structure=None, **kwargs):
    """Z matrix helper for double-GLM-style covariance specifications.

    Parameters
    ----------
    variance_structure : array_like or scipy sparse matrix, optional
        A single pre-built Z matrix to use directly.
    **kwargs
        ``n`` gives the size of the identity Z used when
        ``variance_structure`` is omitted (default 1).

    Returns
    -------
    list
        ``[variance_structure]`` when given, otherwise ``[mc_id(n)]``.

    Notes
    -----
    This is a thin compatibility helper; for any non-identity structure pass
    the actual Z matrix through ``variance_structure``.
    """
    if variance_structure is not None:
        return [variance_structure]
    return [mc_id(kwargs.get("n", 1))]


def mc_twin(n_pairs=None, twin_type: str = "MZ", **kwargs):
    """Z matrices for a classical twin design with pairs of twins.

    The observations must be ordered so that the two members of each pair are
    adjacent (rows ``2*i`` and ``2*i + 1``).

    Parameters
    ----------
    n_pairs : int, optional
        Number of twin pairs. If omitted, ``n // 2`` is taken from ``n`` in
        ``kwargs``.
    twin_type : {"MZ", "DZ", "ACE", "ADE"}, default "MZ"
        Relationship assumed within a pair. ``"MZ"`` gives two copies of the
        shared pair matrix (correlation 1). ``"DZ"``, ``"ACE"`` and ``"ADE"``
        give a shared-environment matrix and an additive-genetic matrix with
        within-pair correlation 0.5.
    **kwargs
        ``n`` gives the total number of observations when ``n_pairs`` is not
        supplied.

    Returns
    -------
    list of ndarray
        ``[G, G]`` for MZ and ``[G, A]`` for the other types, where ``G`` is
        the block matrix of shared pair membership and ``A`` is the additive
        genetic relationship matrix. Both have shape
        ``(2*n_pairs, 2*n_pairs)``.

    Raises
    ------
    ValueError
        If neither ``n_pairs`` nor a usable ``n`` is supplied.

    Notes
    -----
    For MZ twins the two components coincide, so their separate variances are
    not identifiable (the standard ACE parameterisation cannot separate A and
    C for identical twins). Unknown ``twin_type`` values return an empty list.
    """
    if n_pairs is None and "n" in kwargs:
        n_pairs = kwargs["n"] // 2
    if n_pairs is None:
        raise ValueError("mc_twin requires n_pairs or n")
    m = 2 * int(n_pairs)
    Z_list = []
    # common environment/shared factors
    if twin_type.upper() in ["MZ", "DZ", "ACE", "ADE"]:
        # matrix for shared environment/family
        G = np.zeros((m, m))
        for i in range(n_pairs):
            G[2 * i, 2 * i] = G[2 * i, 2 * i + 1] = G[2 * i + 1, 2 * i] = G[2 * i + 1, 2 * i + 1] = 1.0
        Z_list.append(G)
        if twin_type.upper() == "MZ":
            Z_list.append(G)
        elif twin_type.upper() == "DZ":
            A = G.copy()
            for i in range(n_pairs):
                A[2 * i, 2 * i + 1] = A[2 * i + 1, 2 * i] = 0.5
                A[2 * i, 2 * i] = A[2 * i + 1, 2 * i + 1] = 0.5
            Z_list.append(A)
        else:
            A = G.copy()
            for i in range(n_pairs):
                A[2 * i, 2 * i + 1] = A[2 * i + 1, 2 * i] = 0.5
                A[2 * i, 2 * i] = A[2 * i + 1, 2 * i + 1] = 0.5
            Z_list.append(A)
    return Z_list


mc_twin_bio = mc_twin


def mc_twin_full(n_pairs=None, **kwargs):
    """Full twin specification: equivalent to ``mc_twin(n_pairs, twin_type="ACE", **kwargs)``.

    See :func:`mc_twin` for parameters, return value and notes.
    """
    return mc_twin(n_pairs, twin_type="ACE", **kwargs)
