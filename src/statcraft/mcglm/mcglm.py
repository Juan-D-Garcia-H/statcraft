# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""High-level interface to the MCGLM engine.

Multivariate Covariance Generalized Linear Models (Bonat & Jørgensen, 2016, JRSS-C 65(5)) model the mean of one or
several responses through a link function and their covariance through a linear combination of known matrices Z:

    E(y_k) = g_k^{-1}(X_k beta_k),      Var(y_k) = V_k^{1/2} Omega_k(tau_k) V_k^{1/2},      Omega_k = h_k(sum_j tau_kj Z_kj)

with a between-response correlation matrix parametrised by rho. Parameters are estimated with quasi-score and
Pearson estimating functions, and the joint covariance of the estimators is the sandwich of their sensitivity and
variability matrices.

Quick start
-----------
>>> import numpy as np, mcglm_cpp as mc
>>> rng = np.random.default_rng(0); n = 500
>>> x = rng.random(n); X = np.column_stack([np.ones(n), x])
>>> y = rng.poisson(np.exp(0.5 + 0.8 * x)).astype(float)
>>> res = mc.MCGLM(y, X, [mc.mc_id(n)], link="log", variance="tweedie").fit()
>>> res.converged
True
"""

from __future__ import annotations

import math
import warnings
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .. import _mcglmfit as _ms
from ..plots.methods import PlottingMixin


__all__ = ["MCGLM", "MCGLMResults", "mcglm"]

_LINKS = {
    "logit": _ms.Link.Logit,
    "probit": _ms.Link.Probit,
    "cauchit": _ms.Link.Cauchit,
    "cloglog": _ms.Link.Cloglog,
    "loglog": _ms.Link.Loglog,
    "identity": _ms.Link.Identity,
    "log": _ms.Link.Log,
    "sqrt": _ms.Link.Sqrt,
    "1/mu^2": _ms.Link.InvMu2,
    "inverse": _ms.Link.Inverse,
}
_VARS = {
    "constant": _ms.Variance.Constant,
    "tweedie": _ms.Variance.Power,
    "power": _ms.Variance.Power,
    "binomialp": _ms.Variance.BinomialP,
    "binomialpq": _ms.Variance.BinomialPQ,
    "poisson_tweedie": _ms.Variance.PoissonTweedie,
    "geom_tweedie": _ms.Variance.GeomTweedie,
}
_COVS = {"identity": _ms.CovLink.Identity, "inverse": _ms.CovLink.Inverse, "expm": _ms.CovLink.Expm}
_METHODS = {"chaser": _ms.Method.Chaser, "rc": _ms.Method.Rc, "gradient": _ms.Method.Gradient}
_CROSS = {"r": _ms.CrossBlock.RCompatible, "exact": _ms.CrossBlock.Exact}
_STRUCTURE = {
    "auto": _ms.Structure.Auto,
    "dense": _ms.Structure.Dense,
    "blocks": _ms.Structure.Blocks,
    "sparse": _ms.Structure.Sparse,
}
_N_POWER = {_ms.Variance.Constant: 0, _ms.Variance.BinomialPQ: 2}
# pAIC/pBIC need the dense covariance of the stacked model; above this many entries the summary reports
# them as unavailable instead of allocating an impractically large matrix (the fast diagonal path is exempt).
_MAX_PSEUDO_ELEMENTS = 25_000_000


def _get_build_summary():
    """Return ``build_summary`` (shared results-summary builder in ``statcraft.summary``)."""
    from ..summary import build_summary

    return build_summary


def _enum(table: Mapping[str, Any], v: Any, what: str):
    """Resolve a case-insensitive string option to its engine enum (enum members pass through)."""
    if isinstance(v, str):
        key = v.lower()
        if key not in table:
            raise ValueError(f"unknown {what} {v!r}; valid options: {sorted(table)}")
        return table[key]
    return v


def _per_response(v, K: int, name: str) -> list:
    """Broadcast a scalar or a list to one value per response."""
    if isinstance(v, (list, tuple)):
        if len(v) == 1 and K > 1:
            return list(v) * K
        if len(v) != K:
            raise ValueError(f"{name}: expected {K} element(s) (one per response), got {len(v)}")
        return list(v)
    return [v] * K


def _arr(x, name: str, ndim: int) -> np.ndarray:
    """Convert to a float64 array of the required dimensionality."""
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from exc
    if a.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got {a.ndim}-D")
    return a


def _is_sparse(x) -> bool:
    """True for SciPy sparse matrices/arrays (duck-typed, so SciPy is only needed when they are used)."""
    return hasattr(x, "tocsc") and hasattr(x, "nnz") and hasattr(x, "shape")


def _as_z(x, name: str = "z"):
    """Normalise one Z matrix: float64 ndarray (1-D diagonal, 2-D dense) or an engine ``SparseZ``."""
    if isinstance(x, _ms.SparseZ):
        return x
    if _is_sparse(x):
        if x.ndim != 2 or x.shape[0] != x.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got shape {x.shape}")
        csc = x.tocsc(copy=True)
        csc.sum_duplicates()
        csc.sort_indices()
        return _ms.SparseZ(
            int(x.shape[0]),
            np.ascontiguousarray(csc.indptr, dtype=np.int64),
            np.ascontiguousarray(csc.indices, dtype=np.int64),
            np.ascontiguousarray(csc.data, dtype=np.float64),
        )
    try:
        return np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from exc


def _z_equal(a, b) -> bool:
    """Exact equality of two Z specifications (used to recognise ``[Z0, Z0, ...]``)."""
    if _is_sparse(a) or _is_sparse(b):
        if not (_is_sparse(a) and _is_sparse(b)) or a.shape != b.shape:
            return False
        return (a != b).nnz == 0
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _names_of(x, default: Sequence[str]) -> list[str]:
    """Column names of a DataFrame-like object, or ``default``."""
    cols = getattr(x, "columns", None)
    return [str(c) for c in cols] if cols is not None else list(default)


def _unique_join(values) -> str:
    """Join values in first-seen order, dropping repeats (compact summary header)."""
    return ", ".join(dict.fromkeys(str(v) for v in values))


class MCGLM:
    """A Multivariate Covariance GLM ready to be fitted, for one or several responses.

    The model specification (design matrices, Z matrices, links) is stored once in the C++ engine, so ``fit`` can
    be called repeatedly (different ``y``, ``weights`` or options) without copying the data again.

    Parameters
    ----------
    endog : array_like, list of array_like, or 2-D array
        Response(s). A list or a 2-D array defines K responses observed on the same n units. NaN marks a missing
        value, which receives weight 0.
    exog : 2-D array_like or list of such arrays
        Design matrix. A single matrix is shared by all responses.
    z : list
        Known symmetric matrices of the matrix linear predictor of the covariance. A 1-D array is a diagonal matrix
        (fast path), a 2-D array a dense symmetric n x n matrix, and a SciPy sparse matrix a sparse one (never expanded
        to n x n when the units split into independent groups; see :meth:`structure`).
        A flat list of arrays is shared by all responses
        (``z=[Z0]`` gives every response the single matrix Z0; ``z=[I, Zd]`` gives every response two). A list of
        lists gives each response its own matrices (``z=[[Z1], [Z2]]``). K identical arrays (``z=[Z0, Z0, Z0]``) are
        read as one Z per response, with a warning. Identical or proportional Z matrices within a response raise
        ``ValueError``, because their tau parameters would not be identifiable.
    link : str or list of str, default "identity"
        Link function(s): "logit", "probit", "cauchit", "cloglog", "loglog", "identity", "log", "sqrt", "1/mu^2",
        "inverse".
    variance : str or list of str, default "constant"
        Variance function(s): "constant", "tweedie" (mu^p), "binomialP", "binomialPQ", "poisson_tweedie",
        "geom_tweedie".
    covariance : str or list of str, default "identity"
        Covariance link(s): "identity", "inverse", "expm".
    power_fixed : bool or list of bool, default True
        Hold the variance exponent(s) at their starting value(s) instead of estimating them.
    ntrial : array_like or list, optional
        Number of trials for the binomial variance functions (scalar or length n); defaults to 1.
    offset : array_like or list, optional
        Offset added to the linear predictor.
    weights : array_like, optional
        Non-negative observation weights, length n (repeated for every response) or K n (stacked).
    endog_names, exog_names : list of str, optional
        Labels used in the summary. ``exog_names`` is a list of names for K = 1 and a list of lists otherwise;
        DataFrame column names are used when available.
    detect_diagonal : bool, default True
        Store a dense Z whose off-diagonal entries are all zero as a diagonal matrix.

    Attributes
    ----------
    K, n : int
        Number of responses and observations per response.
    problem : mcglm_cpp.Problem
        The underlying engine object.
    formulas : list of str or None
        The formulas, when built with :meth:`from_formula`.

    Raises
    ------
    ValueError
        On inconsistent shapes, non-finite values, unknown options or non-identifiable Z matrices.
    """

    def __init__(
        self,
        endog,
        exog,
        z,
        *,
        link="identity",
        variance="constant",
        covariance="identity",
        power_fixed=True,
        ntrial=None,
        offset=None,
        weights=None,
        endog_names=None,
        exog_names=None,
        detect_diagonal: bool = True,
    ):
        series_name = getattr(endog, "name", None)
        if isinstance(endog, np.ndarray) and endog.ndim == 2:
            endog = [endog[:, j] for j in range(endog.shape[1])]
        elif not isinstance(endog, (list, tuple)):
            endog = [endog]
        K = len(endog)
        if K == 0:
            raise ValueError("endog must contain at least one response")
        ys = [_arr(e, f"endog[{k}]", 1) for k, e in enumerate(endog)]
        n = ys[0].size
        if any(y.size != n for y in ys):
            raise ValueError("all responses must have the same number of observations")
        self.K, self.n = K, n
        self.y = np.concatenate(ys)

        exogs = (
            _per_response(list(exog), K, "exog") if isinstance(exog, (list, tuple)) else [exog] * K
        )
        if exog_names is not None and K == 1 and len(exog_names) and isinstance(exog_names[0], str):
            exog_names = [list(exog_names)]
        self.exog_names, Xs = [], []
        for k in range(K):
            Xa = _arr(exogs[k], f"exog[{k}]", 2)
            if Xa.shape[0] != n:
                raise ValueError(
                    f"exog[{k}] has {Xa.shape[0]} row(s) but the responses have {n} observation(s)"
                )
            names = (
                exog_names[k]
                if exog_names is not None
                else _names_of(exogs[k], [f"x{j}" for j in range(Xa.shape[1])])
            )
            if len(names) != Xa.shape[1]:
                raise ValueError(
                    f"exog_names[{k}] has {len(names)} name(s) but exog[{k}] has {Xa.shape[1]} column(s)"
                )
            self.exog_names.append([str(s) for s in names])
            Xs.append(Xa)
        if endog_names is None:
            endog_names = (
                [str(series_name) if series_name is not None else "y"]
                if K == 1
                else [f"y{k + 1}" for k in range(K)]
            )
        if len(endog_names) != K:
            raise ValueError(f"endog_names must have {K} element(s), got {len(endog_names)}")
        self.endog_names = [str(s) for s in endog_names]

        zs = list(z)
        if len(zs) == 0:
            raise ValueError("z must contain at least one matrix")
        per_response = len(zs) == K and all(isinstance(e, (list, tuple)) for e in zs)
        if per_response:
            zs_per = [list(e) for e in zs]
        else:
            if any(isinstance(e, (list, tuple)) for e in zs):
                raise ValueError(
                    "z mixes lists and arrays; give either one list of matrices per response "
                    "or a flat list of arrays shared by all responses"
                )
            if K > 1 and len(zs) == K and all(_z_equal(zs[0], e) for e in zs[1:]):
                warnings.warn(
                    "z holds K identical matrices; interpreting it as one Z per response "
                    "(z=[[Z0], [Z0], ...]). Pass z=[Z0] to share a single Z explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
                zs_per = [[zs[0]] for _ in range(K)]
            else:
                zs_per = [zs] * K

        self.links = [_enum(_LINKS, v, "link") for v in _per_response(link, K, "link")]
        self.variances = [
            _enum(_VARS, v, "variance") for v in _per_response(variance, K, "variance")
        ]
        self.covariances = [
            _enum(_COVS, v, "covariance") for v in _per_response(covariance, K, "covariance")
        ]
        pf = _per_response(power_fixed, K, "power_fixed")
        nts = _per_response(ntrial, K, "ntrial")
        offs = _per_response(offset, K, "offset")
        self.problem = _ms.Problem()
        for k in range(K):
            nt = None
            if nts[k] is not None:
                a = np.asarray(nts[k], dtype=float)
                if a.size not in (1, n):
                    raise ValueError(
                        f"ntrial must be a scalar or have length n = {n}, got {a.size}"
                    )
                nt = np.broadcast_to(a.reshape(-1), (n,)).copy()
            of = None if offs[k] is None else _arr(offs[k], "offset", 1)
            zk = [_as_z(e, f"z[{j}]") for j, e in enumerate(zs_per[k])]
            self.problem.add_response(
                Xs[k],
                self.links[k],
                self.variances[k],
                self.covariances[k],
                zk,
                bool(pf[k]),
                of,
                nt,
                detect_diagonal,
            )
        self.exogs = Xs
        if weights is None:
            self.weights = None
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = np.tile(w, K) if (w.size == n and K > 1) else w
        self.power_fixed = [bool(v) for v in pf]
        self._offsets = offs
        self.formulas: list[str] | None = None

    # ------------------------------------------------------------------ construction from formulas
    # @classmethod
    # def from_formula(cls, formula: str | Sequence[str], data, z, **kw) -> "MCGLM":
    #     """Build a model from formula(s) evaluated on ``data`` with ``formulaic``.

    #     Parameters
    #     ----------
    #     formula : str or list of str
    #         One formula per response, e.g. ``"y ~ x1 + C(g)"``. In a single string, responses are separated by
    #         ``|``: ``"y1 ~ x | y2 ~ x + z"``.
    #     data : DataFrame-like
    #         Data accepted by ``formulaic.model_matrix``. Missing values in the response are allowed (weight 0).
    #     z : list
    #         See :class:`MCGLM`.
    #     **kw
    #         Remaining keyword arguments of :class:`MCGLM` (``link``, ``variance``, ...).

    #     Raises
    #     ------
    #     ImportError
    #         If ``formulaic`` is not installed.
    #     """
    #     try:
    #         import formulaic
    #     except ImportError as e:  # pragma: no cover
    #         raise ImportError(
    #             "MCGLM.from_formula requires the 'formulaic' package (pip install formulaic)"
    #         ) from e
    #     fs = [s.strip() for s in formula.split("|")] if isinstance(formula, str) else list(formula)
    #     ys, Xs, xn, yn = [], [], [], []
    #     for f in fs:
    #         mm = formulaic.model_matrix(f, data, na_action="ignore")
    #         ys.append(np.asarray(mm.lhs, dtype=float)[:, 0])
    #         Xs.append(np.asarray(mm.rhs, dtype=float))
    #         xn.append([str(c) for c in mm.rhs.columns])
    #         yn.append(str(mm.lhs.columns[0]))
    #     kw.setdefault("endog_names", yn)
    #     model = cls(ys, Xs, z, exog_names=xn, **kw)
    #     model.formulas = [" ".join(f.split()) for f in fs]
    #     return model

    @classmethod
    def from_formula(cls, formulas: list[str], data, z, **kw) -> "MCGLM":
        """Build a model from a list of formulas evaluated on ``data`` with ``formulaic``.

        Parameters
        ----------
        formulas : list of str
            A list of formulas, one per response variable, e.g. ``["y1 ~ x1", "y2 ~ x2"]``.
        data : DataFrame-like
            Data accepted by ``formulaic.model_matrix``. Missing values in the response
            are allowed (assigned weight 0).
        z : list
            See :class:`MCGLM`.
        **kw
            Remaining keyword arguments for :class:`MCGLM` (``link``, ``variance``, ...).

        Raises
        ------
        ImportError
            If the ``formulaic`` package is not installed.
        """
        try:
            import formulaic
        except ImportError as e:
            raise ImportError(
                "MCGLM.from_formula requires the 'formulaic' package (pip install formulaic)"
            ) from e

        ys, Xs, xn, yn = [], [], [], []
        for f in formulas:
            mm = formulaic.model_matrix(f, data, na_action="ignore")
            ys.append(np.asarray(mm.lhs, dtype=float)[:, 0])
            Xs.append(np.asarray(mm.rhs, dtype=float))
            xn.append([str(c) for c in mm.rhs.columns])
            yn.append(str(mm.lhs.columns[0]))

        kw.setdefault("endog_names", yn)
        model = cls(ys, Xs, z, exog_names=xn, **kw)
        model.formulas = [" ".join(f.split()) for f in formulas]
        return model



    # ------------------------------------------------------------------ fitting
    def _start(self, start):
        """Normalise user-provided starting values to the engine's ``Params``."""
        if start is None:
            return None
        if isinstance(start, _ms.Params):
            return start
        K = self.K

        def norm(v, name):  # for K = 1 a flat vector is accepted
            if (
                K == 1
                and isinstance(v, (list, tuple, np.ndarray))
                and len(v) > 0
                and np.ndim(v[0]) == 0
            ):
                return [v]
            return _per_response(v, K, name)

        try:
            beta = [np.asarray(b, float) for b in norm(start["beta"], "start['beta']")]
            tau = [np.atleast_1d(np.asarray(t, float)) for t in norm(start["tau"], "start['tau']")]
        except KeyError as exc:
            raise ValueError(
                f"start must contain the key {exc}; expected keys: beta, tau, [power], [rho]"
            ) from None
        pw = start.get("power")
        pw = (
            [np.atleast_1d(np.asarray(p, float)) for p in norm(pw, "start['power']")]
            if pw is not None
            else [np.ones(max(1, _N_POWER.get(v, 1))) for v in self.variances]
        )
        rho = np.asarray(start.get("rho", []), float) if K > 1 else None
        return _ms.Params(beta, pw, tau, rho)

    def structure(self, block_target: int = 32, sparse_dense_max_n: int = 4000) -> dict:
        """How the covariance will be handled with ``structure="auto"``, without fitting.

        Returns a dict with ``kind``: "diagonal" (per-observation path), "blocks" (exact decomposition into independent
        groups of units), "sparse" (one large connected block handled by the approximate sparse method: one response,
        identity covariance link, sparse Z, ``n > sparse_dense_max_n``) or "dense" (exact dense computation); plus
        ``n_blocks``, ``max_block`` (units in the largest block), ``n_components``, ``n_singletons`` and
        ``max_component``. The units are connected when some Z has a non-zero entry between them.
        """
        return self.problem.structure(int(block_target), int(sparse_dense_max_n))

    def fit(
        self,
        *,
        method: str = "chaser",
        max_iter: int = 20,
        tol: float = 1e-4,
        tuning: float = 1.0,
        correct: bool = True,
        compute_vcov: bool = True,
        cross_block: str = "r",
        r_diag_variability: bool = True,
        start=None,
        callback: Callable | None = None,
        verbose: bool = False,
        retry: Sequence[float] = (),
        y=None,
        weights=None,
        structure: str = "auto",
        block_target: int = 32,
        block_threads: int = 0,
        sparse_dense_max_n: int = 4000,
        n_probes: int = 32,
        n_probes_final: int = 128,
        probe_seed: int = 12345,
    ) -> "MCGLMResults":
        """Fit the model.

        Parameters
        ----------
        method : {"chaser", "rc", "gradient"}, default "chaser"
            Update rule of the covariance parameters.
        max_iter : int, default 20
            Maximum number of iterations (the starting point counts as the first).
        tol : float, default 1e-4
            Convergence tolerance on the largest absolute parameter change.
        tuning : float, default 1.0
            Step scaling of the covariance update (use a smaller value for difficult problems).
        correct : bool, default True
            Include the bias-correction term of the Pearson estimating function.
        compute_vcov : bool, default True
            Compute the joint sandwich covariance of the estimators (needed for standard errors).
        cross_block : {"r", "exact"}, default "r"
            Convention for the derivatives of C in the cross blocks (several responses with a dense Z):
            "r" reproduces the R package; "exact" uses the exact derivative and usually converges better.
        r_diag_variability : bool, default True
            Zero the off-diagonal terms of the Pearson variability, as the R package does.
        start : mcglm_cpp.Params or dict, optional
            Starting values; a dict with keys ``beta``, ``tau`` and optionally ``power``, ``rho``. Computed
            automatically from per-response GLM fits if omitted.
        callback : callable, optional
            ``callback(iteration, beta, cov)`` is called after every iteration; returning True stops the fit
            (``result.stopped``). Exceptions propagate. Ctrl-C interrupts the fit between iterations.
        verbose : bool, default False
            Print the covariance parameters at every iteration to stderr.
        retry : sequence of float, optional
            If a ``NumericalError`` occurs, refit with ``tuning`` multiplied by each factor in turn
            (a ``RuntimeWarning`` is issued for every retry); the last error is raised if all attempts fail.
        y, weights : array_like, optional
            Replace the responses (length K n, stacked) or weights for this fit only. Weights also act as prior
            weights in the automatic starting values.
        structure : {"auto", "dense", "blocks", "sparse"}, default "auto"
            How the covariance is organised. "auto" uses the exact block decomposition when the units split into
            independent groups (cost ``O(sum m_b^3)`` instead of ``O(n^3)``, memory ``O(sum m_b^2)``), the
            per-observation path when every Z is diagonal, the approximate sparse method for one large connected block
            (see below) and the exact dense computation otherwise. "dense" forces one block; "blocks" forces the
            decomposition; "sparse" forces the sparse method. Dense, blocks and diagonal give the same estimates.

            The sparse method needs one response, the identity covariance link and a sparse Z. It solves with a sparse
            Cholesky factor and computes the trace term of the Pearson function exactly from the selected inverse, so
            the **estimates are exact**; only the sensitivity and variability of the covariance parameters use
            stochastic trace estimates (fixed random probes), which affect the standard errors of those parameters by
            the relative Monte Carlo error reported in ``result.mc_error``. Increase ``n_probes_final`` to reduce it.
        block_target : int, default 32
            Small independent groups are merged into batches of about this many units (exact, fewer allocations).
        block_threads : int, default 0
            Threads used to process blocks (OpenMP builds); 0 uses the OpenMP default. BLAS is limited to one thread
            while several block threads run. Results are deterministic for a fixed thread count.
        sparse_dense_max_n : int, default 4000
            With ``structure="auto"``, a single connected block with more units than this and a sparse Z (at most 25%
            non-zeros) uses the sparse method; smaller problems keep the exact dense computation.
        n_probes, n_probes_final : int, default 32 and 128
            Sparse method: random probes of the stochastic traces during the iterations (>= 2) and for the final
            sensitivity and covariance (>= 8, rounded down to a multiple of 8).
        probe_seed : int, default 12345
            Seed of the probes. They are fixed, so the fit is deterministic.

        Returns
        -------
        MCGLMResults

        Raises
        ------
        ValueError
            Invalid arguments (the engine raises ``mcglm_cpp.InvalidInput``, a ``ValueError``).
        mcglm_cpp.NumericalError
            A numerical step failed (singular system, non-positive-definite matrix, non-finite values).
            It is an ``ArithmeticError``.
        """
        if int(max_iter) < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
        if not float(tol) > 0:
            raise ValueError(f"tol must be > 0, got {tol!r}")
        if not float(tuning) > 0:
            raise ValueError(f"tuning must be > 0, got {tuning!r}")
        if str(cross_block).lower() not in _CROSS:
            raise ValueError(
                f"unknown cross_block {cross_block!r}; valid options: {sorted(_CROSS)}"
            )
        meth = _enum(_METHODS, method, "method")
        if str(structure).lower() not in _STRUCTURE:
            raise ValueError(
                f"unknown structure {structure!r}; valid options: {sorted(_STRUCTURE)}"
            )
        if int(block_target) < 1:
            raise ValueError(f"block_target must be >= 1, got {block_target!r}")
        if int(block_threads) < 0:
            raise ValueError(f"block_threads must be >= 0, got {block_threads!r}")
        if int(n_probes) < 2 or int(n_probes_final) < 8:
            raise ValueError(
                f"n_probes must be >= 2 and n_probes_final >= 8, got {n_probes!r} and {n_probes_final!r}"
            )
        if int(sparse_dense_max_n) < 0:
            raise ValueError(f"sparse_dense_max_n must be >= 0, got {sparse_dense_max_n!r}")
        yv = self.y if y is None else np.asarray(y, dtype=float)
        wv = self.weights if weights is None else np.asarray(weights, dtype=float)
        init = self._start(start)
        factors = (1.0,) + tuple(float(f) for f in retry)
        last: Exception | None = None
        for i, f in enumerate(factors):
            o = _ms.Options()
            o.method = meth
            o.max_iter = int(max_iter)
            o.tol = float(tol)
            o.tuning = float(tuning) * f
            o.correct = bool(correct)
            o.compute_vcov = bool(compute_vcov)
            o.verbose = bool(verbose)
            o.cross_block = _CROSS[str(cross_block).lower()]
            o.r_diag_variability = bool(r_diag_variability)
            o.structure = _STRUCTURE[str(structure).lower()]
            o.block_target = int(block_target)
            o.block_threads = int(block_threads)
            o.sparse_dense_max_n = int(sparse_dense_max_n)
            o.n_probes = int(n_probes)
            o.n_probes_final = int(n_probes_final)
            o.probe_seed = int(probe_seed)
            try:
                raw = self.problem.fit(yv, init, wv, o, callback)
            except _ms.NumericalError as e:
                last = e
                if i + 1 == len(factors):
                    raise
                warnings.warn(
                    f"NumericalError ({e}); retrying with tuning={float(tuning) * factors[i + 1]:.3g}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            used = dict(
                method=str(method).lower(),
                max_iter=int(max_iter),
                tol=float(tol),
                tuning=o.tuning,
                correct=bool(correct),
                compute_vcov=bool(compute_vcov),
                cross_block=str(cross_block).lower(),
                r_diag_variability=bool(r_diag_variability),
                structure=str(structure).lower(),
                block_target=int(block_target),
                block_threads=int(block_threads),
                n_probes=int(n_probes),
                n_probes_final=int(n_probes_final),
                sparse_dense_max_n=int(sparse_dense_max_n),
            )
            return MCGLMResults(self, raw, yv, used)
        if last is None:  # pragma: no cover - defensive
            raise RuntimeError("MCGLM fit failed without a captured error")
        raise last  # pragma: no cover

    def initial_values(self) -> _ms.Params:
        """Automatic starting values (per-response GLM fits) for the stored responses and weights."""
        return self.problem.initial_values(self.y, self.weights)


class MCGLMResults(PlottingMixin):
    """Results of :meth:`MCGLM.fit`.

    Arrays are read-only views of the engine's memory (no copies). Parameters are ordered as
    ``[beta (all responses), rho, then for each response: variance exponents (if estimated), tau]``;
    ``param_names`` gives the matching labels.

    Attributes
    ----------
    model : MCGLM
        The fitted model.
    raw : mcglm_cpp.FitResult
        The engine result with every low-level quantity.
    param_names : list of str
        Labels of ``params``.
    fit_options : dict
        The options used for the fit.
    K : int
        Number of responses.
    """

    def __init__(
        self, model: MCGLM, raw: _ms.FitResult, y: np.ndarray, fit_options: dict | None = None
    ):
        self.model, self.raw, self.y = model, raw, y
        self.fit_options = dict(fit_options or {})
        self._pseudo_terms_cache: tuple[float, int] | None = None
        K = model.K
        self.K = K
        names = []
        for k in range(K):
            pre = f"{model.endog_names[k]}:" if K > 1 else ""
            names += [pre + nm for nm in model.exog_names[k]]
        for i in range(K):
            for j in range(i + 1, K):
                names.append(f"rho_{i + 1}{j + 1}")
        for k in range(K):
            pre = f"{model.endog_names[k]}:" if K > 1 else ""
            npw = raw.n_power[k]
            for q in range(npw):
                names.append(pre + ("power" if npw == 1 else ("power_p", "power_q")[q]))
            names += [pre + f"tau{j}" for j in range(raw.n_tau[k])]
        self.param_names = names
        if len(names) != raw.beta.size + raw.cov.size:  # pragma: no cover
            raise RuntimeError("internal error: parameter labels do not match the estimates")
        # positions of each block inside ``params``
        o = 0
        self._ix_beta = []
        for k in range(K):
            self._ix_beta.append(np.arange(o, o + raw.n_beta[k]))
            o += raw.n_beta[k]
        self._ix_rho = np.arange(o, o + raw.n_rho)
        o += raw.n_rho
        self._ix_power, self._ix_tau = [], []
        for k in range(K):
            self._ix_power.append(np.arange(o, o + raw.n_power[k]))
            o += raw.n_power[k]
            self._ix_tau.append(np.arange(o, o + raw.n_tau[k]))
            o += raw.n_tau[k]

    # ------------------------------------------------------------------ estimates
    @property
    def params(self) -> np.ndarray:
        """All estimates: ``[beta, rho, (power, tau) per response]``."""
        return np.concatenate([self.raw.beta, self.raw.cov])

    @property
    def beta(self) -> np.ndarray:
        """Regression coefficients of all responses, stacked."""
        return np.asarray(self.raw.beta)

    @property
    def cov(self) -> np.ndarray:
        """Covariance parameters: ``[rho, (power, tau) per response]``."""
        return np.asarray(self.raw.cov)

    @property
    def fitted(self) -> np.ndarray:
        """Fitted means, responses stacked."""
        return np.asarray(self.raw.fitted)

    @property
    def resid(self) -> np.ndarray:
        """Raw residuals ``y - fitted``, responses stacked (NaN responses give ``-fitted``)."""
        return np.asarray(self.raw.residuals)

    @property
    def resid_pearson(self) -> np.ndarray:
        """Pearson residuals ``(y - mu) / sqrt(diag(C))`` using the fitted marginal variances.

        Computed from :meth:`covariance_matrix` with ``full=False`` (no dense covariance matrix is built).
        Entries where the response, mean or marginal variance is missing/non-positive are NaN.
        """
        r = np.asarray(self.resid, dtype=float)
        try:
            cov = self.covariance_matrix(full=False)
            dc = np.asarray(cov.diag_C, dtype=float).reshape(r.shape)
        except Exception:
            return np.full_like(r, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = r / np.sqrt(dc)
        out[~(np.isfinite(r) & np.isfinite(dc) & (dc > 0))] = np.nan
        return out

    @property
    def resid_standardized(self) -> np.ndarray:
        """Standardized (Cholesky/Mahalanobis) residuals ``L^{-1}(y - mu)`` with ``C = L L^T``.

        Under a correct covariance structure these behave as uncorrelated, unit-variance noise.
        Non-missing observations are decorrelated jointly; missing entries remain NaN. For diagonal
        or block-diagonal-by-unit covariance structures the solve is done without forming the dense
        stacked covariance when possible.
        """
        K, n = self.K, self.model.n
        y = np.asarray(self.y, dtype=float).reshape(K, n)
        mu = np.asarray(self.fitted, dtype=float).reshape(K, n)
        e = y - mu
        keep = np.isfinite(e)
        out = np.full((K, n), np.nan, dtype=float)
        params = self.raw.params
        try:
            if self.raw.structure == "diagonal" and K == 1:
                dc = np.asarray(self.covariance_matrix(full=False).diag_C, dtype=float).reshape(K, n)
                m = keep[0] & (dc[0] > 0)
                out[0, m] = e[0, m] / np.sqrt(dc[0, m])
                return out.reshape(-1)

            if self.raw.structure == "diagonal":
                cinfo = self.model.problem.covariance(params, False)
                dc = np.asarray(cinfo.diag_C, dtype=float).reshape(K, n)
                S = np.asarray(cinfo.Sigma_between, dtype=float)
                for i in range(n):
                    m = keep[:, i] & (dc[:, i] > 0)
                    if not m.any():
                        continue
                    Ls = np.linalg.cholesky(S[np.ix_(m, m)])
                    out[m, i] = np.linalg.solve(Ls, e[m, i] / np.sqrt(dc[m, i]))
                return out.reshape(-1)

            # dense / blocks / other structures: use the joint covariance on finite observations
            rflat = e.reshape(-1)
            kflat = np.isfinite(rflat)
            b = rflat[kflat]
            if b.size == 0:
                return out.reshape(-1)
            if self.raw.structure == "sparse":
                raise ValueError(
                    "standardized residuals are not available with the sparse covariance method "
                    "(they require the model covariance)"
                )
            if K * n > _MAX_PSEUDO_ELEMENTS:
                raise ValueError(
                    f"the dense covariance is too large for standardized residuals "
                    f"({K * n} x {K * n}); use a diagonal/blocked structure or n <= "
                    f"{_MAX_PSEUDO_ELEMENTS // max(K, 1)}"
                )
            C = np.asarray(self.model.problem.covariance(params, True).C, dtype=float)
            if not kflat.all():
                C = C[np.ix_(kflat, kflat)]
            L = np.linalg.cholesky(C)
            z = np.linalg.solve(L, b)
            flat_out = out.reshape(-1)
            flat_out[kflat] = z
            return flat_out
        except Exception:
            return out.reshape(-1)

    @property
    def fitted_by_response(self) -> list[np.ndarray]:
        """Fitted means split by response, as a list of length ``K`` with arrays of length ``n``."""
        n = self.nobs
        mu = np.asarray(self.fitted, dtype=float)
        return [mu[k * n : (k + 1) * n].copy() for k in range(self.K)]

    @property
    def resid_by_response(self) -> list[np.ndarray]:
        """Raw residuals split by response, as a list of length ``K`` with arrays of length ``n``."""
        n = self.nobs
        r = np.asarray(self.resid, dtype=float)
        return [r[k * n : (k + 1) * n].copy() for k in range(self.K)]

    @property
    def nobs(self) -> int:
        """Number of observations per response."""
        return int(np.asarray(self.y).size // self.K)

    @property
    def iterations(self) -> int:
        """Number of iterations performed (rows of the history, including the starting point)."""
        return self.raw.iterations

    @property
    def converged(self) -> bool:
        """Whether the convergence tolerance was reached."""
        return self.raw.converged

    @property
    def stopped(self) -> bool:
        """Whether the fit was stopped by the callback."""
        return self.raw.stopped

    @property
    def structure(self) -> dict:
        """Covariance organisation used: ``kind`` ("diagonal", "dense", "blocks" or "sparse"), ``n_blocks`` and ``max_block``."""
        return {
            "kind": self.raw.structure,
            "n_blocks": self.raw.n_blocks,
            "max_block": self.raw.max_block,
        }

    @property
    def mc_error(self) -> float:
        """Sparse method: largest relative Monte Carlo standard error of the stochastic traces (0 for exact methods).

        It bounds the relative error of the sensitivity and variability of the covariance parameters, hence of their
        standard errors; the estimates themselves are exact."""
        return float(self.raw.mc_rel_error)

    @property
    def params_struct(self) -> _ms.Params:
        """The estimates as a :class:`mcglm_cpp.Params` (usable as ``start`` for another fit)."""
        return self.raw.params

    @property
    def rho(self) -> np.ndarray:
        """Between-response correlations (empty for one response)."""
        return np.asarray(self.raw.params.rho)

    @property
    def tau(self) -> list[np.ndarray]:
        """Dispersion parameters, one array per response."""
        return list(self.raw.params.tau)

    @property
    def power(self) -> list[np.ndarray]:
        """Variance exponents, one array per response."""
        return list(self.raw.params.power)

    # ------------------------------------------------------------------ pseudo-likelihood criteria
    @property
    def df(self) -> int:
        """Number of estimated parameters (``len(params)``): beta, rho, power and tau."""
        return int(self.params.size)

    def _pseudo_terms(self) -> tuple[float, int]:
        """Gaussian pseudo log-likelihood and the number of observations used (cached).

        Uses the mean and covariance of the fitted model, as the R package ``mcglm`` does for ``plogLik``:
        ``-N/2 log(2 pi) - 1/2 log|C| - 1/2 (y - mu)' C^{-1} (y - mu)``. Missing responses (NaN) are excluded.
        """
        if self._pseudo_terms_cache is None:
            K, n = self.K, self.model.n
            y = np.asarray(self.y, dtype=float).reshape(K, n)
            mu = np.asarray(self.fitted, dtype=float).reshape(K, n)
            r = (y - mu).reshape(-1)
            keep = np.isfinite(r)
            b = r[keep]
            N = int(b.size)
            if N == 0:
                raise ValueError(
                    "the pseudo log-likelihood needs at least one non-missing observation"
                )
            params = self.raw.params
            if self.raw.structure == "diagonal" and K == 1:
                # C is diagonal: log|C| and b' C^{-1} b are O(N), no dense matrix needed.
                dc = np.asarray(self.model.problem.covariance(params, False).diag_C, dtype=float)[
                    keep
                ]
                if not np.all(dc > 0):
                    raise np.linalg.LinAlgError(
                        "the model covariance matrix is not positive definite"
                    )
                logdet = float(np.log(dc).sum())
                quad = float(np.sum(b * b / dc))
            elif self.raw.structure == "diagonal":
                # Every Z is diagonal, so C is block diagonal by unit: n independent K x K blocks
                # C_i = D_i^{1/2} Sigma_b D_i^{1/2}. This avoids the dense (K n) x (K n) matrix.
                cinfo = self.model.problem.covariance(params, False)
                dc = np.asarray(cinfo.diag_C, dtype=float).reshape(K, n)
                S = np.asarray(cinfo.Sigma_between, dtype=float)
                e = y - mu
                if not keep.all():
                    logdet = 0.0
                    quad = 0.0
                    N = 0
                    for i in range(n):
                        m = np.isfinite(e[:, i])
                        if not m.any():
                            continue
                        di = dc[m, i]
                        if not np.all(di > 0):
                            raise np.linalg.LinAlgError(
                                "the model covariance matrix is not positive definite"
                            )
                        Ls = np.linalg.cholesky(S[np.ix_(m, m)])
                        logdet += 2.0 * float(np.log(np.diag(Ls)).sum()) + float(np.log(di).sum())
                        z = np.linalg.solve(Ls, e[m, i] / np.sqrt(di))
                        quad += float(z @ z)
                        N += int(m.sum())
                else:
                    if not np.all(dc > 0):
                        raise np.linalg.LinAlgError(
                            "the model covariance matrix is not positive definite"
                        )
                    Ls = np.linalg.cholesky(S)
                    logdet = n * 2.0 * float(np.log(np.diag(Ls)).sum()) + float(
                        np.log(dc).sum()
                    )
                    z = np.linalg.solve(Ls, e / np.sqrt(dc))
                    quad = float(np.sum(z * z))
                    N = K * n
            else:
                if self.raw.structure == "sparse":
                    raise ValueError(
                        "pAIC/pBIC are not available with the sparse covariance method "
                        "(they need the dense model covariance)"
                    )
                if K * n > _MAX_PSEUDO_ELEMENTS:
                    raise ValueError(
                        f"the dense covariance is too large for pAIC/pBIC "
                        f"({K * n} x {K * n}); use a diagonal/blocked structure or n <= "
                        f"{_MAX_PSEUDO_ELEMENTS // max(K, 1)}"
                    )
                C = np.asarray(self.model.problem.covariance(params, True).C, dtype=float)
                if not keep.all():
                    C = C[np.ix_(keep, keep)]
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError as exc:
                    raise np.linalg.LinAlgError(
                        "the model covariance matrix is not positive definite"
                    ) from exc
                logdet = 2.0 * float(np.log(np.diag(L)).sum())
                z = np.linalg.solve(L, b)
                quad = float(z @ z)
            ll = -0.5 * (N * math.log(2.0 * math.pi) + logdet + quad)
            self._pseudo_terms_cache = (float(ll), N)
        return self._pseudo_terms_cache

    @property
    def pseudo_loglik(self) -> float:
        """Gaussian pseudo log-likelihood evaluated at the estimates.

        The stacked response is assumed multivariate normal with mean ``fitted`` and covariance given by the
        fitted model (the same convention as ``plogLik`` in the R package). It is not a true likelihood unless
        the response is Gaussian; use it only to compare models fitted to the same data.
        """
        return self._pseudo_terms()[0]

    @property
    def pAIC(self) -> float:
        """Pseudo AIC: ``-2 * pseudo_loglik + 2 * df``.

        A likelihood-free model-comparison criterion for McGLMs (Bonat, 2018); smaller is better and only
        differences between models fitted to the same response data are meaningful.
        """
        return 2.0 * self.df - 2.0 * self.pseudo_loglik

    @property
    def pBIC(self) -> float:
        """Pseudo BIC: ``-2 * pseudo_loglik + df * log(N)``, with ``N`` the number of observations used.

        Stronger penalty than :attr:`pAIC`; see its caveats.
        """
        ll, N = self._pseudo_terms()
        return self.df * math.log(N) - 2.0 * ll

    # ------------------------------------------------------------------ algorithm internals
    @property
    def history(self) -> dict[str, np.ndarray]:
        """Per-iteration values: ``beta``, ``cov``, ``score_beta`` and ``score_cov`` (one row per iteration)."""
        r = self.raw
        return {
            "beta": np.asarray(r.iter_beta),
            "cov": np.asarray(r.iter_cov),
            "score_beta": np.asarray(r.score_beta),
            "score_cov": np.asarray(r.score_cov),
        }

    @property
    def sensitivity(self) -> dict[str, np.ndarray]:
        """Final sensitivity matrices: ``inv_S_beta`` (inverse) and ``S_cov`` (Pearson)."""
        r = self.raw
        return {"inv_S_beta": np.asarray(r.inv_S_beta), "S_cov": np.asarray(r.S_cov)}

    @property
    def variability(self) -> dict[str, np.ndarray]:
        """Final variability matrices ``V_beta`` and ``V_cov`` (empty if ``compute_vcov=False``)."""
        r = self.raw
        return {"V_beta": np.asarray(r.V_beta), "V_cov": np.asarray(r.V_cov)}

    # ------------------------------------------------------------------ inference
    def cov_params(self) -> np.ndarray:
        """Joint covariance matrix of ``params`` (sandwich estimator).

        Raises
        ------
        ValueError
            If the model was fitted with ``compute_vcov=False``.
        """
        if self.raw.vcov.size == 0:
            raise ValueError(
                "the covariance of the estimators was not computed (fit was run with compute_vcov=False)"
            )
        return np.asarray(self.raw.vcov)

    @property
    def bse(self) -> np.ndarray:
        """Standard errors of ``params``. A negative variance estimate gives NaN and a ``RuntimeWarning``."""
        v = np.diag(self.cov_params())
        if (v < 0).any():
            bad = [self.param_names[i] for i in np.flatnonzero(v < 0)]
            warnings.warn(
                f"negative variance estimate for {bad}: the standard error is reported as NaN "
                "(the mcglm variability formula can produce this, as in the R package)",
                RuntimeWarning,
                stacklevel=2,
            )
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.where(v >= 0, v, np.nan))

    @property
    def zvalues(self) -> np.ndarray:
        """Wald statistics ``params / bse``."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.params / self.bse

    @property
    def pvalues(self) -> np.ndarray:
        """Two-sided p-values of the Wald statistics under the standard normal distribution."""
        z = self.zvalues
        return np.array(
            [math.erfc(abs(v) / math.sqrt(2.0)) if np.isfinite(v) else np.nan for v in z]
        )

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this MCGLM fit.

        See :mod:`statcraft.plots` for the supported ``kind`` names.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Wald confidence intervals, shape ``(len(params), 2)``.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level, in (0, 1).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        q = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        b = self.bse
        return np.column_stack([self.params - q * b, self.params + q * b])

    def wald_test(self, r_matrix, q_matrix=None):
        """Wald test for the linear hypothesis ``R * theta = q``.

        Parameters
        ----------
        r_matrix : array_like
            Restriction matrix ``R`` with ``len(params)`` columns.
        q_matrix : array_like, optional
            Right-hand side ``q``. Defaults to zeros.

        Returns
        -------
        dict
            ``statistic`` (chi-squared), ``df`` and ``pvalue``.
        """
        if self.raw.vcov.size == 0:
            raise ValueError(
                "MCGLM.wald_test: the covariance of the estimators was not computed "
                "(fit was run with compute_vcov=False)."
            )
        theta = np.asarray(self.params, dtype=float)
        V = np.asarray(self.cov_params(), dtype=float)
        R = np.atleast_2d(np.asarray(r_matrix, dtype=float))
        if R.shape[1] != theta.shape[0]:
            raise ValueError(
                f"MCGLM.wald_test: r_matrix has {R.shape[1]} columns, expected {theta.shape[0]}."
            )
        q = (
            np.asarray(q_matrix, dtype=float).reshape(-1)
            if q_matrix is not None
            else np.zeros(R.shape[0], dtype=float)
        )
        if q.shape[0] != R.shape[0]:
            raise ValueError(
                f"MCGLM.wald_test: q_matrix has length {q.shape[0]}, expected {R.shape[0]}."
            )
        diff = R @ theta - q
        cov_r = R @ V @ R.T
        stat = float(diff.T @ np.linalg.solve(cov_r, diff))
        df = int(R.shape[0])
        try:
            from scipy.stats import chi2 as _chi2

            pvalue = float(_chi2.sf(max(stat, 0.0), df))
        except Exception:
            pvalue = math.erfc(math.sqrt(max(stat, 0.0) / 2.0)) if df == 1 else float("nan")
        return {"statistic": stat, "df": df, "pvalue": pvalue}

    def anova(self, other):
        """Compare this model with a nested one using the pseudo log-likelihood (pseudo-LRT).

        The model with more estimated parameters is treated as the full model. Only meaningful for
        models fitted to the same response data.

        Parameters
        ----------
        other : MCGLMResults
            Another fitted MCGLM result.

        Returns
        -------
        pandas.DataFrame
            Table with model df, pseudo log-likelihood, df difference, chi-squared statistic and p-value.
        """
        import pandas as pd

        if not isinstance(other, MCGLMResults):
            raise TypeError("MCGLMResults.anova: 'other' must be an MCGLMResults instance.")
        y1 = np.asarray(self.y, dtype=float)
        y2 = np.asarray(other.y, dtype=float)
        if y1.shape != y2.shape:
            raise ValueError("MCGLMResults.anova: models must be fitted to the same response data.")
        finite = np.isfinite(y1) & np.isfinite(y2)
        if finite.any() and not np.allclose(y1[finite], y2[finite], equal_nan=True):
            raise ValueError("MCGLMResults.anova: models must be fitted to the same response data.")
        ll1 = float(self.pseudo_loglik)
        ll2 = float(other.pseudo_loglik)
        df1 = int(self.df)
        df2 = int(other.df)
        if df1 == df2:
            raise ValueError(
                "MCGLMResults.anova: the models have the same number of parameters; nothing to test."
            )
        if df1 >= df2:
            ll_big, ll_small, df_big, df_small = ll1, ll2, df1, df2
        else:
            ll_big, ll_small, df_big, df_small = ll2, ll1, df2, df1
        ddf = df_big - df_small
        stat = 2.0 * (ll_big - ll_small)
        try:
            from scipy.stats import chi2 as _chi2

            pval = float(_chi2.sf(max(stat, 0.0), ddf))
        except Exception:
            pval = float("nan")
        return pd.DataFrame(
            {
                "Df model": [df_small, df_big],
                "Pseudo log-Lik.": [ll_small, ll_big],
                "Df": [np.nan, ddf],
                "Chisq": [np.nan, stat],
                "Pr(>Chi)": [np.nan, pval],
            },
            index=["small", "big"],
        )

    def get_prediction(self, exog=None, response=0, alpha=0.05):
        """Predicted mean with standard error and confidence interval for one response.

        Parameters
        ----------
        exog : (m, p) array_like, optional
            New design matrix; the estimation sample is used if omitted.
        response : int, default 0
            Index of the response.
        alpha : float, default 0.05
            Significance level for the Wald confidence interval.

        Returns
        -------
        pandas.DataFrame
            Columns ``mean``, ``mean_se``, ``mean_ci_lower`` and ``mean_ci_upper``.
        """
        import pandas as pd

        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        k = int(response)
        if not 0 <= k < self.K:
            raise ValueError(f"response must be in [0, {self.K - 1}], got {response!r}")
        beta_k = np.asarray(self.raw.params.beta[k], dtype=float)
        if exog is None:
            X = np.asarray(self.model.exogs[k], dtype=float)
            off = self.model._offsets[k]
        else:
            X = _arr(exog, "exog", 2)
            off = None
            if X.shape[1] != beta_k.size:
                raise ValueError(
                    f"exog has {X.shape[1]} column(s) but the model has {beta_k.size} coefficient(s)"
                )
        eta = X @ beta_k
        if off is not None:
            off = np.asarray(off, dtype=float)
            if off.shape != eta.shape:
                raise ValueError(f"offset must have shape {eta.shape}, got {off.shape}")
            eta = eta + off
        mu, dmu = _ms.link_eval(self.model.links[k], np.ascontiguousarray(eta))
        mu = np.asarray(mu, dtype=float)
        dmu = np.asarray(dmu, dtype=float)
        if self.raw.vcov.size == 0:
            raise ValueError(
                "MCGLM.get_prediction: the covariance of the estimators was not computed "
                "(fit was run with compute_vcov=False)."
            )
        V = np.asarray(self.cov_params(), dtype=float)
        ix = self._ix_beta[k]
        Vb = V[np.ix_(ix, ix)]
        var_eta = np.einsum("ij,jk,ik->i", X, Vb, X)
        se_mean = np.abs(dmu) * np.sqrt(np.maximum(var_eta, 0.0))
        q = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        lo = mu - q * se_mean
        hi = mu + q * se_mean
        return pd.DataFrame(
            {"mean": mu, "mean_se": se_mean, "mean_ci_lower": lo, "mean_ci_upper": hi}
        )

    # ------------------------------------------------------------------ model covariance and prediction
    def covariance_matrix(self, full: bool = False):
        """Covariance structure of the model at the fitted parameters.

        Parameters
        ----------
        full : bool, default False
            If False only the mean and the diagonal of C are computed. If True the dense C
            ``(K n) x (K n)`` and the within-response matrices ``Sigma_k`` are also built (``O((K n)^2)`` memory).

        Returns
        -------
        mcglm_cpp.CovarianceOut
            With attributes ``mu``, ``diag_C``, ``C``, ``Sigma`` and ``Sigma_between``.
        """
        return self.model.problem.covariance(self.raw.params, full)

    def predict(self, exog=None, *, response: int = 0, offset=None) -> np.ndarray:
        """Expected value ``g^{-1}(X beta + offset)`` of one response.

        Parameters
        ----------
        exog : (m, p) array_like, optional
            New design matrix; the estimation sample is used if omitted.
        response : int, default 0
            Index of the response.
        offset : array_like, optional
            Offset for ``exog`` (ignored when ``exog`` is omitted, where the model's offset is used).
        """
        k = int(response)
        if not 0 <= k < self.K:
            raise ValueError(f"response must be in [0, {self.K - 1}], got {response!r}")
        if exog is None:
            X = self.model.exogs[k]
            off = self.model._offsets[k]
        else:
            X = _arr(exog, "exog", 2)
            off = offset
            if X.shape[1] != self.raw.params.beta[k].size:
                raise ValueError(
                    f"exog has {X.shape[1]} column(s) but the model has {self.raw.params.beta[k].size} coefficient(s)"
                )
        eta = X @ np.asarray(self.raw.params.beta[k])
        if off is not None:
            off = np.asarray(off, dtype=float)
            if off.shape != eta.shape:
                raise ValueError(f"offset must have shape {eta.shape}, got {off.shape}")
            eta = eta + off
        mu, _ = _ms.link_eval(self.model.links[k], np.ascontiguousarray(eta))
        return np.asarray(mu)

    def refit(self, **kw) -> "MCGLMResults":
        """Fit again starting from these estimates; keyword arguments are those of :meth:`MCGLM.fit`."""
        kw.setdefault("start", self.raw.params)
        return self.model.fit(**kw)

    # ------------------------------------------------------------------ presentation
    def to_frame(self, response: int | None = None):
        """Estimates as a pandas DataFrame indexed by parameter name.

        Parameters
        ----------
        response : int, optional
            If given, return only the parameters of that response: its regression coefficients
            (beta), and when applicable its variance-power and dispersion (tau) parameters. This
            is analogous to selecting a response in :meth:`predict`.

        Returns
        -------
        pandas.DataFrame
            Columns: ``coef`` and, if the covariance was computed, ``std err``, ``z`` and ``P>|z|``.
        """
        import pandas as pd

        cols = {"coef": self.params}
        if self.raw.vcov.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cols.update({"std err": self.bse, "z": self.zvalues, "P>|z|": self.pvalues})
        df = pd.DataFrame(cols, index=self.param_names)
        if response is None:
            return df
        k = int(response)
        if not 0 <= k < self.K:
            raise ValueError(f"response must be in [0, {self.K - 1}], got {response!r}")
        ix = np.concatenate([self._ix_beta[k], self._ix_power[k], self._ix_tau[k]]).astype(int)
        return df.iloc[ix]

    # ------------------------------------------------------------------ summary
    _LINK_NAMES = {v: k for k, v in _LINKS.items()}
    _VAR_NAMES = {
        "Constant": "constant",
        "Power": "tweedie",
        "BinomialP": "binomialP",
        "BinomialPQ": "binomialPQ",
        "PoissonTweedie": "poisson_tweedie",
        "GeomTweedie": "geom_tweedie",
    }
    _COV_NOTE = {"identity": "", "inverse": ", precision scale", "expm": ", log scale"}

    def summary(
        self,
        *,
        alpha: float = 0.05,
        stars: bool | dict = False,
        title: str = "MCGLM Results",
        diagnostics: bool = True,
    ):
        """Statsmodels-style results summary.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level of the confidence intervals, in (0, 1).
        stars : bool or dict, default False
            Add significance stars next to the coefficients. ``True`` uses the default thresholds
            (``{0.01: "***", 0.05: "**", 0.1: "*"}``); a dict gives custom thresholds and markers.
        title : str, default "MCGLM Results"
            Report title.
        diagnostics : bool, default True
            Include the algorithm diagnostics block (final scores, last step, condition number of the covariance).

        Returns
        -------
        Summary
            ``str(summary)`` gives the plain-text report, a notebook renders it as HTML, ``summary.tables[i].data``
            holds each table as a pandas object, and ``save_html`` / ``save_latex`` / ``save_excel`` export it.
            Blocks: between-response correlations (K > 1) and, for each response, regression, variance-power
            (if estimated) and dispersion parameters. Inference is asymptotic (normal). The header also
            reports the Gaussian pseudo log-likelihood and the pseudo information criteria ``pAIC`` and
            ``pBIC`` (also available as properties of those names).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        m, raw, K = self.model, self.raw, self.K
        has_v = raw.vcov.size > 0
        P = self.params
        notes: list[str] = []
        if has_v:
            v = np.diag(self.cov_params())
            with np.errstate(invalid="ignore"):
                se = np.sqrt(np.where(v >= 0, v, np.nan))
            neg = [self.param_names[i] for i in np.flatnonzero(v < 0)]
            if neg:
                notes.append(
                    "Negative variance estimate for "
                    + ", ".join(neg)
                    + ": standard error reported as NaN "
                    "(the mcglm variability formula can give it, as in R; more data or a less dense Z "
                    "usually stabilizes it)."
                )
            vb = (
                "Standard errors from the joint sandwich covariance (sensitivity/variability) of (beta, rho, power, tau); "
                "z-based inference (asymptotic)."
            )
            if self.raw.structure == "sparse":
                notes.append(
                    f"Sparse method: the estimates are exact; the standard errors of the covariance parameters use "
                    f"stochastic trace estimates ({self.raw.n_probes} probes, relative Monte Carlo error about "
                    f"{self.raw.mc_rel_error:.1%}). Increase n_probes_final to reduce it."
                )
            if self.fit_options.get("r_diag_variability", True):
                vb += " Pearson variability is diagonal."
            notes.append(vb)
        else:
            se = None
            notes.append("Standard errors were not computed (compute_vcov=False).")
        if not self.converged:
            notes.append(
                "The algorithm "
                + (
                    "was stopped by the callback."
                    if self.stopped
                    else f"did not converge within max_iter={self.fit_options.get('max_iter', '?')} iterations."
                )
                + " Estimates may be unreliable; consider more iterations, a smaller tuning, or method='rc'."
            )
        if any(c.name == "Expm" for c in m.covariances):
            notes.append(
                "covariance='expm': dispersion parameters are on the log (matrix exponential) scale."
            )
        if any(c.name == "Inverse" for c in m.covariances):
            notes.append("covariance='inverse': dispersion parameters are on the precision scale.")

        # ---- coefficient blocks
        def sl(ix):
            return (P[ix], None if se is None else se[ix])

        blocks = []
        if K > 1 and self._ix_rho.size:
            c, e = sl(self._ix_rho)
            blocks.append(
                dict(
                    heading="Correlation between responses (rho)",
                    names=[
                        f"rho_{i + 1}{j + 1} ({m.endog_names[i]}, {m.endog_names[j]})"
                        for i in range(K)
                        for j in range(i + 1, K)
                    ],
                    coefs=c,
                    se=e,
                )
            )
        for k in range(K):
            pre = f"[{m.endog_names[k]}] " if K > 1 else ""
            c, e = sl(self._ix_beta[k])
            blocks.append(
                dict(
                    heading=f"{pre}Regression parameters (link: {self._LINK_NAMES[m.links[k]]})",
                    names=m.exog_names[k],
                    coefs=c,
                    se=e,
                )
            )
            if self._ix_power[k].size:
                npw = self._ix_power[k].size
                c, e = sl(self._ix_power[k])
                blocks.append(
                    dict(
                        heading=f"{pre}Power parameters (variance: {self._VAR_NAMES[m.variances[k].name]})",
                        names=["power"] if npw == 1 else ["power_p", "power_q"],
                        coefs=c,
                        se=e,
                    )
                )
            c, e = sl(self._ix_tau[k])
            cov_name = m.covariances[k].name.lower()
            blocks.append(
                dict(
                    heading=f"{pre}Dispersion parameters (covariance: {cov_name}{self._COV_NOTE[cov_name]})",
                    names=[f"tau{j}" for j in range(self._ix_tau[k].size)],
                    coefs=c,
                    se=e,
                )
            )

        # ---- header
        yk = np.asarray(self.y).reshape(K, m.n)
        n_miss = int(np.sum(~np.isfinite(yk)))
        fo = self.fit_options
        # pseudo-likelihood information criteria (R's plogLik / pAIC / pBIC)
        try:
            info = {
                "Pseudo log-Lik.:": f"{self.pseudo_loglik:.2f}",
                "pAIC:": f"{self.pAIC:.2f}",
                "pBIC:": f"{self.pBIC:.2f}",
            }
            notes.append(
                "pAIC/pBIC use the Gaussian pseudo log-likelihood at the fitted mean and covariance; "
                "compare only models fitted to the same response data."
            )
        except (ValueError, ArithmeticError, MemoryError) as exc:
            info = {"Pseudo log-Lik.:": "n/a", "pAIC:": "n/a", "pBIC:": "n/a"}
            notes.append(f"pAIC/pBIC were not computed: {exc}")
        left = [
            ("Dep. Variable(s):", ", ".join(m.endog_names)),
            ("Model:", "MCGLM"),
            ("Method:", fo.get("method", "chaser")),
            ("Link(s):", _unique_join(self._LINK_NAMES[c] for c in m.links)),
            ("Variance(s):", _unique_join(self._VAR_NAMES[c.name] for c in m.variances)),
            ("Covariance(s):", _unique_join(c.name.lower() for c in m.covariances)),
            ("No. Responses:", str(K)),
            ("No. Observations:", str(m.n)),
        ]
        if n_miss:
            left.append(("Missing (weight 0):", str(n_miss)))
        left += [
            ("Df Model (beta):", str(self.beta.size)),
            ("Df Model (cov):", str(self.cov.size)),
        ]
        right = [
            ("Iterations:", str(self.iterations)),
            ("Converged:", "Yes" if self.converged else "No"),
            ("Tolerance:", f"{fo['tol']:.0e}" if "tol" in fo else "n/a"),
            ("Tuning:", f"{fo['tuning']:g}" if "tuning" in fo else "n/a"),
            ("Correction term:", ("Yes" if fo.get("correct", True) else "No")),
            ("Covariance type:", "Sandwich (joint)" if has_v else "not computed"),
            ("Pseudo log-Lik.:", info["Pseudo log-Lik.:"]),
            ("pAIC:", info["pAIC:"]),
            ("pBIC:", info["pBIC:"]),
            (
                "Z structure:",
                self.structure["kind"]
                if self.structure["kind"] != "blocks"
                else f"blocks ({self.structure['n_blocks']})",
            ),
        ]
        subtitle = []
        if m.formulas:
            subtitle = [
                f"Formula: {f}" if K == 1 else f"[{m.endog_names[i]}] {f}"
                for i, f in enumerate(m.formulas)
            ]

        # ---- algorithm diagnostics
        diag_left = diag_right = None
        if diagnostics:
            sb, sc = np.asarray(raw.score_beta), np.asarray(raw.score_cov)
            diag_left = [
                ("Max |score| (beta):", f"{np.nanmax(np.abs(sb[-1])):.3e}" if sb.size else "n/a"),
                (
                    "Max |score| (cov):",
                    f"{np.nanmax(np.abs(sc[-1])):.3e}"
                    if sc.size and np.isfinite(sc[-1]).any()
                    else "n/a",
                ),
            ]
            ib, ic = np.asarray(raw.iter_beta), np.asarray(raw.iter_cov)
            if ib.shape[0] >= 2:
                last = max(
                    np.abs(ib[-1] - ib[-2]).max(), np.abs(ic[-1] - ic[-2]).max() if ic.size else 0.0
                )
                diag_right = [("Last step max |Δ|:", f"{last:.3e}")]
            if has_v and np.isfinite(raw.vcov).all():
                diag_right = (diag_right or []) + [
                    ("Cond. no. (vcov):", f"{np.linalg.cond(np.asarray(raw.vcov)):.3e}")
                ]
        return _get_build_summary()(
            title=title,
            subtitle=subtitle or None,
            left=left,
            right=right,
            blocks=blocks,
            diag_left=diag_left,
            diag_right=diag_right,
            dist="normal",
            alpha=alpha,
            notes=notes,
            stars=stars,
        )

    def __repr__(self) -> str:
        return f"<MCGLMResults K={self.K} n={self.model.n} iterations={self.iterations} converged={self.converged}>"


mcglm = MCGLM.from_formula
