"""High-level interface over the C++ engine: lme, gls, nlme, gnls, lm_list, nls_list.

Models are constructed unfitted and are fitted with ``.fit()``::

    res = lme("y ~ x", data, random="~ 1 | g").fit()
    print(res.summary())

``nlme`` and ``gnls`` also accept a plain Python function instead of a
formula::

    def growth(x, Asym, xmid, scal):
        return Asym / (1 + np.exp((xmid - x) / scal))

    res = nlme(growth, data, response="y", predictors="x",
               random="Asym ~ 1 | g", start={"Asym": 1, "xmid": 1, "scal": 1}).fit()

``params`` is optional for that form: the keys of ``start`` provide the
parameter names and their order. Pass ``params`` explicitly when a parameter
has a covariate formula, e.g. ``params={"Asym": "Variety", "xmid": "1"}``.
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import _nlmefit as core
from ..plots.methods import PlottingMixin
from ..summary import TableBlock, build_summary
from ..utils.modelview import PARAM_VIEW_NAMES, ModelView
from .formula import (FormParts, collect_columns, design, design_new, design_spec, eval_expr, factor_codes, handle_na, names_in,
                      nested_codes, parse_form, r_to_py, _FUNCS, _IDENT)
from .structs import CorStruct, PdStruct, VarStruct, as_pd
from ._summary import (_fmt_value, _residual_sigma, balanced_info, fixed_block, lm_random_tables, lm_varcorr_blocks,
                        nlme_random_tables, r2_icc, standardized_residuals_table, structure_details, structure_text)

_OPT = {"internal": "Internal", "trust": "TrustRegion", "lbfgsb": "LBFGSB", "scipy": "ScipyBFGS"}
_SS = {"SSlogis": "logis", "SSfpl": "fpl", "SSasymp": "asymp", "SSasympOff": "asymp_off", "SSasympOrig": "asymp_orig",
       "SSmicmen": "micmen", "SSgompertz": "gompertz", "SSweibull": "weibull"}


def _F(a):  # matrix in Fortran order (Eigen) and float64
    return np.asfortranarray(np.asarray(a, dtype=float))


def _V(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())


def _I(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.int32).ravel())


def _as_df(data) -> pd.DataFrame:
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


# ----------------------------------------------------------------------------------------------------------------------
# structures -> engine
# ----------------------------------------------------------------------------------------------------------------------
def _cor_spec(cor: Optional[CorStruct]):
    s = core.CorSpec()
    if cor is None:
        return s
    if cor.kind == "Custom":
        c = cor.custom
        return core.make_custom_cor(c["name"], c["n_params"], c["matrix"], c["start"], c.get("deriv"))
    s.kind = getattr(core.CorKind, cor.kind)
    s.p, s.q, s.nugget = cor.p, cor.q, cor.nugget
    s.metric = getattr(core.SpatialMetric, cor.metric.capitalize())
    return s


@dataclass
class _VarBuild:
    spec: Any
    strata: list
    cov: list
    fitted: bool
    forms: list


def _var_forms(var: Optional[VarStruct]) -> list[str]:
    if var is None:
        return []
    if var.kind == "Comb":
        return [f for p in var.parts for f in _var_forms(p)]
    return [var.form] if var.form else []


def _r_strata(codes: np.ndarray, gcodes: Optional[np.ndarray]) -> np.ndarray:
    """Recode the strata as R does: the reference stratum (code 0) is the first one to
    appear when the data are ordered by group (lme/gls order rows by the grouping factor),
    and the remaining strata follow in order of appearance."""
    idx = np.arange(len(codes)) if gcodes is None else np.argsort(gcodes, kind="stable")
    seen: dict = {}
    for c in codes[idx]:
        if c not in seen:
            seen[c] = len(seen)
    return np.array([seen[c] for c in codes], dtype=np.int32)


def _var_spec(var: Optional[VarStruct], df: pd.DataFrame, gcodes: Optional[np.ndarray] = None) -> _VarBuild:
    strata: list = []
    cov: list = []
    fitted = [False]

    def leaf(v: VarStruct):
        if v.kind == "Comb":
            return core.VarSpec.combine([leaf(p) for p in v.parts])
        fp = parse_form(v.form or "~1")
        rhs = fp.rhs.strip()
        has_g = bool(fp.groups)
        s = core.VarSpec()
        if v.kind == "Custom":
            c = v.custom
            s = core.make_custom_var(c["name"], c["n_params"], c["needs_cov"], c["scales"], c["start"], c.get("dscales"))
            if c["needs_cov"] and rhs not in ("1", ""):
                s.cov_slot = len(cov); cov.append(_V(eval_expr(rhs, df)))
            if has_g:
                s.strata_slot = len(strata); strata.append(_I(_r_strata(nested_codes(df, fp.groups), gcodes)))
            return s
        s.kind = {"Ident": core.VarKind.Ident, "Fixed": core.VarKind.Fixed, "Power": core.VarKind.Power, "Exp": core.VarKind.Exp,
                  "ConstPower": core.VarKind.ConstPower, "ConstProp": core.VarKind.ConstProp}[v.kind]
        if v.kind == "Ident":
            if not has_g:
                raise ValueError("varIdent needs strata: form='~1 | g'")
            s.strata_slot = len(strata); strata.append(_I(_r_strata(nested_codes(df, fp.groups), gcodes)))
            return s
        if v.kind == "Fixed":
            s.cov_slot = len(cov); cov.append(_V(eval_expr(rhs, df)))
            return s
        if "fitted" in rhs:
            s.fitted = True; fitted[0] = True
        else:
            s.cov_slot = len(cov); cov.append(_V(eval_expr(rhs, df)))
        if has_g:
            s.by_stratum = True
            s.strata_slot = len(strata); strata.append(_I(_r_strata(nested_codes(df, fp.groups), gcodes)))
        return s

    if var is None:
        return _VarBuild(core.VarSpec(), strata, cov, False, [])
    spec = leaf(var)
    return _VarBuild(spec, strata, cov, fitted[0], _var_forms(var))


def _var_kind(spec: Any) -> str:
    """Leaf name of a nanobind ``VarKind`` enum."""
    return str(spec.kind).rsplit(".", 1)[-1]


def _py_var_leaves(var: Optional[VarStruct]) -> list[VarStruct]:
    """Python variance descriptors in engine order (depth-first for ``varComb``)."""
    if var is None:
        return []
    if var.kind == "Comb":
        return [leaf for part in var.parts for leaf in _py_var_leaves(part)]
    return [var]


def _cpp_var_leaves(spec: Any) -> list[Any]:
    """Engine variance leaves in parameter order (depth-first for ``varComb``)."""
    if _var_kind(spec) == "Comb":
        return [leaf for part in spec.comb for leaf in _cpp_var_leaves(part)]
    return [spec]


def _leaf_nparams(spec: Any, py_spec: Optional[VarStruct], strata: np.ndarray | None) -> int:
    """Number of variance parameters of one leaf, matching the C++ layout."""
    kind = _var_kind(spec)
    if kind in ("None", "Fixed"):
        return 0
    if kind == "Custom":
        custom = getattr(py_spec, "custom", None) if py_spec is not None else None
        return int(custom.get("n_params", 0)) if custom else 0
    n_strata = int(np.unique(strata).size) if strata is not None and strata.size else 1
    n_per = n_strata if spec.by_stratum else 1
    if kind == "Ident":
        return max(n_strata - 1, 0)
    if kind in ("Power", "Exp"):
        return n_per
    if kind in ("ConstPower", "ConstProp"):
        return 2 * n_per
    return 0


def _variance_scales(fit: Any, fitted: np.ndarray) -> np.ndarray:
    """Reconstruct the relative residual SDs ``g(mu_i, delta)`` of a mixed fit.

    The engine reports ``sigma`` on the *whitened* scale: with a variance
    function the observation SD is ``sigma * g(mu_i, delta)``.  Reconstructing
    that factor from the saved structure and fitted values lets residual
    diagnostics use ``(y - yhat) / (sigma * g)``, i.e. unit-variance residuals
    on the same scale as the N(0, 1) reference line.
    """
    mu = np.asarray(fitted, dtype=float).reshape(-1)
    n = mu.size
    var = getattr(fit, "var_spec", None)
    if var is None or n == 0:
        return np.ones(n)
    data = getattr(fit, "_data", None)
    if data is None or len(data) != n:
        return np.ones(n)
    try:
        vb = _var_spec(var, data, getattr(fit, "var_gcodes", None))
        covs = [np.asarray(c, dtype=float).reshape(-1) for c in vb.cov]
        strata = [np.asarray(s).reshape(-1) for s in vb.strata]

        # The engine overwrites a fitted(.) covariate slot with the current
        # fitted values at every outer iteration; mirror that convention.
        components = _cpp_var_leaves(vb.spec)
        py_leaves = _py_var_leaves(var)
        if len(components) != len(py_leaves):
            return np.ones(n)
        for cspec in components:
            if cspec.fitted:
                slot = int(cspec.cov_slot)
                while len(covs) <= slot:
                    covs.append(np.empty(0))
                covs[slot] = mu

        pairs: list[tuple[Any, Optional[VarStruct], int, np.ndarray | None, np.ndarray | None, int]] = []
        offset = 0
        for cspec, py_spec in zip(components, py_leaves):
            slot = int(cspec.strata_slot)
            st = strata[slot] if slot < len(strata) and strata[slot].size == n else None
            nleaf = _leaf_nparams(cspec, py_spec, st)
            cs = int(cspec.cov_slot)
            cov = covs[cs] if cs < len(covs) and covs[cs].size == n else None
            pairs.append((cspec, py_spec, offset, cov, st, nleaf))
            offset += nleaf

        theta = np.asarray(getattr(fit, "theta", []), dtype=float).reshape(-1)
        if theta.size < offset:
            return np.ones(n)
        var_theta = theta[theta.size - offset:] if offset else np.empty(0)

        scales = np.ones(n)
        for cspec, py_spec, off, cov, st, nleaf in pairs:
            kind = _var_kind(cspec)
            th = var_theta[off:off + nleaf]
            if kind == "None":
                leaf = np.ones(n)
            elif kind == "Fixed":
                if cov is None:
                    return np.ones(n)
                leaf = np.sqrt(np.abs(cov))
            elif kind == "Custom":
                custom = (py_spec.custom if py_spec is not None else None) or {}
                scales_fn = custom.get("scales")
                if not callable(scales_fn):
                    return np.ones(n)
                leaf = np.asarray(scales_fn(
                    th,
                    cov if cov is not None else np.empty(0),
                    st if st is not None else np.empty(0, dtype=int),
                    n,
                ), dtype=float).reshape(-1)
            else:
                needs_cov = kind in ("Power", "Exp", "ConstPower", "ConstProp", "Fixed")
                if needs_cov and (cov is None or not np.all(np.isfinite(cov))):
                    return np.ones(n)
                st_values = st if st is not None else np.zeros(n, dtype=int)
                key = np.asarray(st_values, dtype=int) if cspec.by_stratum else np.zeros(n, dtype=int)
                if kind == "Ident":
                    leaf = np.where(st_values == 0, 1.0, np.exp(th[np.maximum(st_values - 1, 0)]))
                elif kind == "Power":
                    leaf = np.power(np.abs(cov), th[key])
                elif kind == "Exp":
                    leaf = np.exp(th[key] * cov)
                elif kind == "ConstPower":
                    leaf = np.exp(th[key]) + np.power(np.abs(cov), th[nleaf // 2 + key])
                elif kind == "ConstProp":
                    a = np.exp(th[key])
                    b = np.exp(th[nleaf // 2 + key])
                    leaf = np.sqrt(a * a + b * b * cov * cov)
                else:
                    return np.ones(n)
            if leaf.size != n or not np.all(np.isfinite(leaf)) or np.any(leaf <= 0):
                return np.ones(n)
            scales *= leaf
        return scales
    except Exception:
        return np.ones(n)


def _pd_spec(p: PdStruct):
    if p.kind == "Blocked":
        return core.PdSpec.blocked(list(p.sizes), [getattr(core.PdKind, k) for k in p.kinds])
    if p.kind == "Custom":
        c = p.custom
        return core.make_custom_pd(c["name"], c["nparams"], c["factor"], c["start"], c["from_D"], c.get("dfactor"))
    return core.PdSpec(getattr(core.PdKind, p.kind))


def _apply_control(spec, control: Optional[Mapping], kind: str):
    if not control:
        return {}
    ctl = dict(control)
    extra = {}
    for k, v in ctl.items():
        if k == "optimizer":
            spec.optimizer = getattr(core.Optimizer, _OPT[str(v).lower()])
        elif k in ("max_iter", "gtol", "ftol", "fd_step", "max_step"):
            setattr(spec.opt, k, v)
        elif k == "sigma":
            spec.sigma_fixed = float(v)
        elif k == "fixed_theta":
            lst = []
            for lab, val in dict(v).items():
                ft = core.FixedTheta(); ft.label = lab; ft.value = float(val); lst.append(ft)
            spec.fixed = lst
        elif k in ("auto_start", "ident"):
            extra[k] = v
        elif hasattr(spec, k):
            setattr(spec, k, v)
        else:
            raise ValueError(f"unknown control option for {kind}: {k!r}")
    return extra


# ----------------------------------------------------------------------------------------------------------------------
# correlation
# ----------------------------------------------------------------------------------------------------------------------
def _cor_inputs(cor: Optional[CorStruct], df: pd.DataFrame, chain: Sequence[str], cor_prefix: bool = False):
    """Translate a correlation structure into engine inputs.

    Parameters
    ----------
    cor : CorStruct or None
        Correlation structure to translate.
    df : pandas.DataFrame
        Data providing the time/position or spatial covariates.
    chain : sequence of str
        Random-effects grouping chain ordered from outer to inner; empty for
        :func:`gls`.
    cor_prefix : bool, optional
        If ``True``, allow a correlation grouping that is a strict prefix of
        the random-effects chain (an extension over R's ``lme``).

    Returns
    -------
    cspec : engine CorSpec
        Correlation specification passed to the engine.
    time : numpy.ndarray or None
        Position/time covariate for non-spatial structures.
    coords : numpy.ndarray or None
        Spatial coordinates for spatial structures.
    cor_sub : numpy.ndarray or None
        Sub-group codes used by the correlation blocks.
    groups : list[str]
        Correlation grouping variables declared in the formula.

    Raises
    ------
    ValueError
        If a non-spatial structure has several position covariates or if the
        correlation grouping is incompatible with the random-effects grouping.
    """
    if cor is None:
        return core.CorSpec(), None, None, None, []
    fp = parse_form(cor.form or "~1")
    exprs = [e.strip() for e in fp.rhs.split("+") if e.strip()]
    time = coords = None
    if exprs and exprs != ["1"]:
        if len(exprs) == 1:
            time = _V(eval_expr(exprs[0], df))
        else:
            if cor.kind not in ("Exp", "Gaus", "Lin", "Ratio", "Spher", "Custom"):
                raise ValueError("a non-spatial correlation accepts only one position covariate")
            coords = _F(np.column_stack([eval_expr(e, df) for e in exprs]))
    cor_sub = None
    groups = fp.groups
    if chain:
        # As in R's lme: without '|' the correlation uses the full random-effects chain
        # (innermost level); a FINER grouping (which starts with the whole chain:
        # random ~1|a, correlation ~1|a/b) is valid; a COARSER grouping (a prefix of the
        # chain) is replaced by R with the full chain plus a warning. cor_prefix=True (an
        # extension not present in R) keeps it as given: observations from different
        # prefix blocks remain uncorrelated even if they share lower levels.
        use = list(groups) if groups else list(chain)
        ch = list(chain)
        if use[:len(ch)] == ch:
            pass
        elif use and use == ch[:len(use)]:
            if not cor_prefix:
                import warnings
                warnings.warn(f"cannot use a correlation grouping coarser ({'/'.join(use)}) than the random-effects one "
                              f"({'/'.join(ch)}): the latter is used (as in R's lme); cor_prefix=True applies the given grouping", stacklevel=3)
                use = ch
        else:
            raise ValueError(f"correlation grouping {use} is incompatible with the random-effects grouping {ch}: it must start with it (or be a prefix with cor_prefix=True)")
        if len(use) > 1:
            cor_sub = _I(nested_codes(df, use))
    return _cor_spec(cor), time, coords, cor_sub, groups


# ----------------------------------------------------------------------------------------------------------------------
# containment degrees of freedom (nlme::getFixDF) and prediction utilities
# ----------------------------------------------------------------------------------------------------------------------
def _is_const(x: np.ndarray, tol: float = math.sqrt(np.finfo(float).eps)) -> bool:
    if len(x) < 1:
        return True
    x = np.asarray(x, float)
    return bool(np.all(np.abs(x if x[0] == 0 else x / x[0] - 1.0) < tol))


def _get_fix_df(X: np.ndarray, gcodes: Sequence[np.ndarray]) -> np.ndarray:
    """Denominator degrees of freedom of each coefficient by the containment method of
    Pinheiro & Bates (nlme's getFixDF). gcodes[k]: group id of level k+1 (nested chain
    from outer to inner) for each row of X."""
    N, p = X.shape
    Q = len(gcodes)
    ngrps = [len(np.unique(g)) for g in gcodes]
    dfX = np.array(ngrps + [N], float) - np.array([0] + ngrps, float)     # strata: g1, g2/g1, ..., Residual
    val = np.zeros(p)
    notint = np.array([not _is_const(X[:, j]) for j in range(p)])
    if notint.any():
        strat = np.zeros(p, int)                                          # stratum of each column: the innermost one where it varies
        for j in range(p):
            for k in range(Q):
                g = gcodes[k]
                order = np.argsort(g, kind="stable")
                xs, gs = X[order, j], g[order]
                varies = np.any((gs[1:] == gs[:-1]) & (xs[1:] != xs[:-1]))
                if varies:
                    strat[j] = k + 1
        counts = np.bincount(strat[notint], minlength=Q + 1)
        dfX = dfX - counts
        if not notint.all():
            dfX[0] -= 1
        else:
            dfX[1:] += 1
        val[notint] = dfX[strat[notint]]
    if not notint.all():
        val[~notint] = dfX.max()
    return val


def _t_table(fixef: pd.Series, se: pd.Series, dfv) -> pd.DataFrame:
    from scipy import stats
    t = pd.DataFrame({"Value": fixef, "Std.Error": se})
    t["DF"] = np.asarray(dfv, float)
    t["t-value"] = t["Value"] / t["Std.Error"]
    t["p-value"] = 2.0 * stats.t.sf(np.abs(t["t-value"]), t["DF"])
    return t


def _lookup_b(maps: dict, keys: list, q: int) -> np.ndarray:
    """Random effects for each row according to its group. A group absent from the fit
    yields NaN (R's predict.lme/nlme return NA)."""
    out = np.full((len(keys), q), np.nan)
    for i, k in enumerate(keys):
        v = maps.get(k)
        if v is not None:
            out[i] = v
    return out


def _row_keys(nd: pd.DataFrame, chain: Sequence[str]) -> list:
    return list(zip(*[nd[c] for c in chain])) if len(chain) > 1 else list(nd[chain[0]])


# ----------------------------------------------------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------------------------------------------------
def _summary_df(dfv) -> Optional[float]:
    """Single df value for build_summary. nlme uses a per-coefficient df (containment);
    the summary table only accepts one, so the common value or the median is used."""
    a = np.asarray(dfv, dtype=float)
    if a.size == 0:
        return None
    return float(a[0]) if np.allclose(a, a[0], equal_nan=True) else float(np.median(a))


def _compact_fit_repr(result) -> str:
    """One-line diagnostic representation; full tables live in ``summary()``."""
    parts = []
    formula = getattr(result, "formula", None)
    if formula:
        parts.append(f"'{formula}'")
    for attr, label in (("nobs", "n"), ("npar", "k")):
        value = getattr(result, attr, None)
        if value is not None:
            parts.append(f"{label}={value}")
    loglik = getattr(result, "loglik", None)
    if loglik is not None:
        parts.append(f"logLik={float(loglik):.4g}")
    converged = getattr(result, "converged", None)
    if converged is not None:
        parts.append("converged" if converged else "not converged")
    detail = f" {', '.join(parts)}" if parts else ""
    return f"<{type(result).__name__}{detail}>"


class _FitBase(PlottingMixin):
    """Common behavior shared by the fitted-model result classes.

    Subclasses expose ``loglik``, ``npar`` and ``nobs`` and may override
    :meth:`_bic_n` when the effective sample size used by BIC differs from the
    number of observations (for example REML fits).
    """

    def __repr__(self):
        return _compact_fit_repr(self)

    def fit(self):
        """Return the already-fitted result itself (chainable API: lme(...).fit())."""
        return self

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this fitted model.

        See :mod:`statcraft.plots` for the supported ``kind`` names; the
        default is the adaptive 2x2 diagnostics panel.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def aic(self):
        """Akaike information criterion, ``-2 * logLik + 2 * npar``."""
        return -2.0 * self.loglik + 2.0 * self.npar

    def bic(self):
        """Bayesian information criterion, ``-2 * logLik + log(n) * npar``."""
        return -2.0 * self.loglik + math.log(self._bic_n()) * self.npar

    def _bic_n(self):
        """Effective sample size used by :meth:`bic` (number of observations)."""
        return self.nobs

    def cor_fixef(self) -> pd.DataFrame:
        """Correlation matrix of the fixed effects derived from ``vcov``."""
        cov = self.vcov.to_numpy(dtype=float)
        std = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(std, std)
        return pd.DataFrame(corr, index=self.vcov.index, columns=self.vcov.columns)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for fixed effects and variance components.

        Fixed effects use t-based intervals with the model's degrees of freedom when
        available; variance components use normal (Wald) intervals from reported
        standard errors when available.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        pandas.DataFrame
            Indexed by parameter/component name with columns ``estimate``, ``lower`` and ``upper``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        from scipy import stats

        rows = []
        index = []
        # fixed effects
        fixef = getattr(self, "fixef", None)
        se = getattr(self, "se", None)
        fixdf = getattr(self, "_fixdf", None)
        if fixef is not None and se is not None:
            vals = np.asarray(fixef, dtype=float)
            ses = np.asarray(se, dtype=float)
            if fixdf is not None:
                dfs = np.asarray(fixdf, dtype=float)
                qs = stats.t.ppf(1.0 - alpha / 2.0, np.maximum(dfs, 1.0))
            else:
                qs = np.full_like(vals, stats.norm.ppf(1.0 - alpha / 2.0), dtype=float)
            for name, v, s, q in zip(list(getattr(fixef, "index", range(len(vals)))), vals, ses, qs):
                rows.append([v, v - q * s, v + q * s])
                index.append(str(name))

        # variance components
        varcorr = getattr(self, "varcorr", None)
        if isinstance(varcorr, pd.DataFrame) and {"estimate", "se"}.issubset(varcorr.columns):
            z = stats.norm.ppf(1.0 - alpha / 2.0)
            for name, row in varcorr.iterrows():
                est = float(row["estimate"])
                s = row["se"]
                s = float(s) if np.isfinite(s) else np.nan
                if np.isfinite(s):
                    lo, hi = est - z * s, est + z * s
                else:
                    lo = hi = np.nan
                rows.append([est, lo, hi])
                index.append(str(name))
        elif not index:
            # fallback: theta without SEs
            theta = getattr(self, "theta", None)
            if theta is not None:
                if isinstance(theta, pd.Series):
                    items = list(theta.items())
                else:
                    items = [(f"theta[{i}]", v) for i, v in enumerate(np.asarray(theta, dtype=float).ravel())]
                for name, v in items:
                    rows.append([float(v), np.nan, np.nan])
                    index.append(str(name))

        return pd.DataFrame(rows, index=index, columns=["estimate", "lower", "upper"])

    def wald_test(self, r_matrix, q_matrix=None):
        """Wald test for the linear hypothesis ``R * beta = q`` on fixed effects.

        Parameters
        ----------
        r_matrix : array_like
            Restriction matrix with as many columns as fixed-effect coefficients.
        q_matrix : array_like, optional
            Right-hand side ``q``. Defaults to zeros.

        Returns
        -------
        dict
            ``statistic`` (chi-squared), ``df`` and ``pvalue``.
        """
        fixef = getattr(self, "fixef", None)
        vcov = getattr(self, "vcov", None)
        if fixef is None or vcov is None:
            raise ValueError("wald_test: fixed effects and their covariance matrix are required.")
        beta = np.asarray(fixef, dtype=float)
        V = np.asarray(vcov, dtype=float)
        R = np.atleast_2d(np.asarray(r_matrix, dtype=float))
        if R.shape[1] != beta.shape[0]:
            raise ValueError(
                f"wald_test: r_matrix has {R.shape[1]} columns, expected {beta.shape[0]}."
            )
        q = (
            np.asarray(q_matrix, dtype=float).reshape(-1)
            if q_matrix is not None
            else np.zeros(R.shape[0], dtype=float)
        )
        if q.shape[0] != R.shape[0]:
            raise ValueError(f"wald_test: q_matrix has length {q.shape[0]}, expected {R.shape[0]}.")
        diff = R @ beta - q
        cov_r = R @ V @ R.T
        stat = float(diff.T @ np.linalg.solve(cov_r, diff))
        df = int(R.shape[0])
        try:
            from scipy.stats import chi2 as _chi2

            pvalue = float(_chi2.sf(max(stat, 0.0), df))
        except Exception:
            pvalue = math.erfc(math.sqrt(max(stat, 0.0) / 2.0)) if df == 1 else float("nan")
        return {"statistic": stat, "df": df, "pvalue": pvalue}

    def anova(self, other, ddf: str = "chi2"):
        """Compare this model with another fitted model on the same response.

        ``ddf='chi2'`` (the default) keeps the likelihood-ratio comparison.
        For linear mixed models, ``ddf='satterthwaite'`` instead tests the extra
        fixed effects with a Wald F test whose denominator df uses the
        Satterthwaite approximation (both models must share the same random and
        residual covariance structure and be fitted by REML).

        Parameters
        ----------
        other : _FitBase
            Another fitted model.
        ddf : {'chi2', 'satterthwaite', 'residual'}, default 'chi2'
            Degrees-of-freedom method for the test.

        Returns
        -------
        pandas.DataFrame
            LRT table for ``ddf='chi2'``; F-test table otherwise.
        """
        method = str(ddf).strip().lower().replace("-", "_").replace(" ", "_")
        if method not in ("chi2", "lrt", "satterthwaite", "residual"):
            raise ValueError("anova: ddf must be 'chi2', 'satterthwaite' or 'residual'")
        if not hasattr(other, "loglik") or not hasattr(other, "npar"):
            raise TypeError("anova: 'other' must be a fitted model with loglik and npar.")
        y1 = np.asarray(getattr(self, "_y", []), dtype=float)
        y2 = np.asarray(getattr(other, "_y", []), dtype=float)
        if y1.shape != y2.shape:
            raise ValueError("anova: models must be fitted to the same response data.")
        finite = np.isfinite(y1) & np.isfinite(y2)
        if finite.any() and not np.allclose(y1[finite], y2[finite], equal_nan=True):
            raise ValueError("anova: models must be fitted to the same response data.")
        ll1, ll2 = float(self.loglik), float(other.loglik)
        df1, df2 = int(self.npar), int(other.npar)
        if df1 == df2:
            raise ValueError("anova: the models have the same number of parameters; nothing to test.")
        if df1 >= df2:
            big, small = self, other
            ll_big, ll_small, df_big, df_small = ll1, ll2, df1, df2
        else:
            big, small = other, self
            ll_big, ll_small, df_big, df_small = ll2, ll1, df2, df1

        if method in ("satterthwaite", "residual"):
            from ._ddf import anova_f_test, require_reml

            big = require_reml(big, "Satterthwaite ANOVA") if method == "satterthwaite" else big
            small = require_reml(small, "Satterthwaite ANOVA") if method == "satterthwaite" else small
            ll_big, ll_small = float(big.loglik), float(small.loglik)
            df_big, df_small = int(big.npar), int(small.npar)
            res = anova_f_test(small, big, method=method)
            return pd.DataFrame(
                {
                    "Df": [df_small, df_big],
                    "logLik": [ll_small, ll_big],
                    "Df diff": [np.nan, res["num_df"]],
                    "F": [np.nan, res["statistic"]],
                    "Den Df": [np.nan, res["den_df"]],
                    "Pr(>F)": [np.nan, res["pvalue"]],
                },
                index=["small", "big"],
            )

        diff_df = df_big - df_small
        stat = 2.0 * (ll_big - ll_small)
        try:
            from scipy.stats import chi2 as _chi2

            pvalue = float(_chi2.sf(max(stat, 0.0), diff_df))
        except Exception:
            pvalue = float("nan")
        return pd.DataFrame(
            {
                "Df": [df_small, df_big],
                "logLik": [ll_small, ll_big],
                "Df diff": [np.nan, diff_df],
                "Chisq": [np.nan, stat],
                "Pr(>Chi)": [np.nan, pvalue],
            },
            index=["small", "big"],
        )


class LmeFit(_FitBase):
    """Result of :func:`lme` and :func:`gls`.

    The object exposes the estimated fixed effects, the variance/correlation
    components, the random effects and the usual inference helpers.  The raw
    C++ engine result remains available through :attr:`raw` for advanced use.

    Attributes
    ----------
    fixef : pandas.Series
        Fixed-effect estimates indexed by coefficient name.
    se : pandas.Series
        Standard errors of the fixed effects.
    vcov : pandas.DataFrame
        Covariance matrix of the fixed effects.
    sigma : float
        Residual standard deviation.
    loglik : float
        Maximized log-likelihood (REML or ML, depending on ``method``).
    nobs : int
        Number of observations used in the fit.
    theta : pandas.Series
        Estimated variance/correlation parameters with the engine labels.
    varcorr : pandas.DataFrame
        Variance and correlation components (``estimate`` and ``se`` columns).
    ranef
        BLUP random effects: a :class:`pandas.DataFrame` for a single grouping
        level or a dictionary keyed by grouping chain for nested models.
    kind : str
        ``"lme"`` or ``"gls"``.
    method : str
        ``"REML"`` or ``"ML"``.
    warning : str or None
        Convergence or numerical warning emitted by the engine.
    converged : bool
        Whether the optimizer reported convergence.
    raw
        The underlying engine result object.
    """

    def __init__(self, raw, theta_labels, names, index, y, kind, method, spec_info):
        self.raw, self.theta_labels, self.names, self.index, self._y, self.kind, self.method = raw, list(theta_labels), names, index, y, kind, method
        self.__dict__.update(spec_info)
        self.fixef = pd.Series(np.array(raw.beta), index=names, name="Value")
        self.se = pd.Series(np.array(raw.se), index=names, name="Std.Error")
        self.vcov = pd.DataFrame(np.array(raw.vcov_beta), index=names, columns=names)
        self.sigma = math.sqrt(raw.sigma2)
        self.loglik = raw.loglik
        self.nobs = len(y)
        self.theta = pd.Series(np.array(raw.theta), index=self.theta_labels) if len(raw.theta) else pd.Series(dtype=float)
        self.varcorr = pd.DataFrame([(v.name, v.est, v.se) for v in raw.varcomp], columns=["name", "estimate", "se"]).set_index("name")
        self.warning = raw.warning
        self.converged = bool(raw.opt.converged)
        self.npar = int(raw.rank) + len(raw.theta) + (0 if spec_info.get("sigma_fixed") else 1) - spec_info.get("n_fixed_theta", 0)

    def _bic_n(self):
        return self.nobs - (int(self.raw.rank) if self.method == "REML" else 0)

    def _unorder(self, v):
        out = np.empty(self.nobs); out[np.array(self.raw.order)] = np.array(v)
        return pd.Series(out, index=self.index)

    def fitted(self, level: Optional[int] = None) -> pd.Series:
        """Fitted values as in fitted.lme: level=0 is the population level (fixed effects
        only); level=k adds the random effects of the first k grouping levels (outer ->
        inner); the default is the innermost level (all of them)."""
        Q = len(getattr(self, "_pred_levels", []))
        level = Q if level is None else level
        if level == 0 or Q == 0:
            return self._unorder(self.raw.fitted_pop)
        if level == Q:
            return self._unorder(self.raw.fitted_cond)
        return self.predict(None, level=level)

    def residuals(self, type="response", level: Optional[int] = None) -> pd.Series:
        """Residuals at the requested random-effect level.

        Parameters
        ----------
        type : {"response", "pearson", "normalized"}, default "response"
            Residual type. ``"response"`` returns raw residuals ``y - fitted(level)``.
            ``"pearson"`` scales residuals by ``sigma * g(mu_i, delta)`` when a
            variance function is present. ``"normalized"`` returns residuals
            standardized by the marginal covariance ``V^{-1/2}(y - mu)`` when the
            covariance structure can be reconstructed (including variance
            functions), falling back to the same scale otherwise.
        level : int, optional
            ``0`` uses population-level fitted values; ``k`` adds the random effects
            of the first ``k`` levels. Defaults to all levels. For backward
            compatibility, ``residuals(level_int)`` is accepted.

        Returns
        -------
        pandas.Series
            Residuals on the original data index.
        """
        if isinstance(type, int) and not isinstance(type, bool):
            level, type = type, "response"
        if type is None:
            type = "response"
        type = str(type).lower()
        raw_resid = pd.Series(self._y - self.fitted(level).to_numpy(), index=self.index)
        if type in ("response", "raw"):
            return raw_resid
        sigma = _residual_sigma(self)
        mu = np.asarray(self.fitted(level).to_numpy(), dtype=float)
        scale = _variance_scales(self, mu)
        denom = sigma * scale
        if type == "pearson":
            with np.errstate(divide="ignore", invalid="ignore"):
                return raw_resid / denom
        if type == "normalized":
            # Best-effort marginal normalization for LME without correlation.
            try:
                if (
                    getattr(self, "kind", None) == "lme"
                    and (level is None or level == 0)
                    and not getattr(self, "cor_spec", None)
                    and getattr(self, "_pred_levels", None)
                ):
                    y = np.asarray(self._y, dtype=float)
                    r = y - mu
                    order = np.asarray(self.raw.order, dtype=int)
                    inv_order = np.empty_like(order)
                    inv_order[order] = np.arange(order.size)
                    r_e = r[order]
                    s_e = scale[order]
                    labels = np.asarray(self.raw.labels, dtype=int)
                    Z0 = design_new(self._pred_levels[0]["spec"], self._data.reset_index(drop=True))
                    Z0_e = np.asarray(Z0, dtype=float)[order]
                    Sb = np.asarray(self.raw.Sigma_b, dtype=float)
                    sigma2 = float(self.raw.sigma2)
                    out_e = np.full_like(r_e, np.nan, dtype=float)
                    for g in np.unique(labels):
                        idx = np.flatnonzero(labels == g)
                        Zi = Z0_e[idx]
                        s_i = s_e[idx]
                        Vi = Zi @ Sb @ Zi.T + sigma2 * np.outer(s_i, s_i)
                        L = np.linalg.cholesky(Vi)
                        out_e[idx] = np.linalg.solve(L, r_e[idx])
                    return pd.Series(out_e[inv_order], index=self.index)
            except Exception:
                pass
            with np.errstate(divide="ignore", invalid="ignore"):
                return raw_resid / denom
        raise ValueError("type must be 'response', 'pearson' or 'normalized'")

    def predict(self, newdata: Optional[pd.DataFrame] = None, level: Optional[int] = None) -> pd.Series:
        """Like predict.lme / predict.gls: level=0 uses fixed effects only; level=k adds
        the random effects of the first k levels. Groups in newdata that are absent from
        the fit yield NaN with level >= 1 (like R's NA). A fixed factor level not seen in
        the fit is an error; missing values yield NaN. Without newdata, predicts the data
        used in the fit."""
        levels = getattr(self, "_pred_levels", [])
        Q = len(levels)
        level = Q if level is None else level
        if not 0 <= level <= Q:
            raise ValueError(f"level must be between 0 and {Q}")
        if newdata is None:
            if level == 0 or level == Q or Q == 0:
                return self.fitted(0 if level == 0 else None)
            nd, idx = self._data.reset_index(drop=True), self.index
        else:
            nd = pd.DataFrame(newdata).reset_index(drop=True); idx = nd.index
        pred = design_new(self._fixed_spec, nd) @ np.nan_to_num(self._beta_vec)
        pred = np.where(np.isnan(design_new(self._fixed_spec, nd)).any(axis=1), np.nan, pred)
        for lv in levels[:level]:
            Z = design_new(lv["spec"], nd)
            b = _lookup_b(lv["map"], _row_keys(nd, lv["chain"]), Z.shape[1])
            pred = pred + (Z * b).sum(axis=1)
        return pd.Series(pred, index=idx)

    def tTable(self, ddf: str = "containment", *, df: str | None = None) -> pd.DataFrame:
        """Coefficient table with optional Satterthwaite denominator df.

        Parameters
        ----------
        ddf : {'containment', 'residual', 'satterthwaite'}, default 'containment'
            ``'containment'`` keeps the R ``nlme`` containment df (``N - p`` for
            ``gls``); ``'satterthwaite'`` uses the REML covariance model and
            requires (or refits in) REML.  ``'kr'`` is not implemented yet.
        df : str, optional
            Backwards-compatible alias for ``ddf``.
        """
        if df is not None:
            ddf = df
        method = str(ddf).strip().lower().replace("-", "_").replace(" ", "_")
        if method in ("kr", "kenward", "kenward_roger"):
            raise NotImplementedError("Kenward-Roger df are not implemented yet")
        if method not in ("containment", "residual", "satterthwaite", "satt"):
            raise ValueError("ddf must be 'containment', 'residual' or 'satterthwaite'")
        if method in ("satterthwaite", "satt"):
            from ._ddf import ddf_for_fit, require_reml

            fit = require_reml(self, "Satterthwaite degrees of freedom")
            eye = np.eye(len(fit.fixef))
            dfs = pd.Series(ddf_for_fit(fit, eye), index=fit.fixef.index, name="DF")
            return _t_table(fit.fixef, fit.se, dfs)
        return _t_table(self.fixef, self.se, self._fixdf)

    @property
    def ranef(self):
        """BLUP random-effects DataFrame for the outer level (or a dict level -> DataFrame when nested)."""
        return self._ranef

    def summary(self, alpha: float = 0.05):
        """statsmodels-style summary (statcraft.summary) of the lme/gls fit.

        The table uses a single t df (the common value or the median of ``_fixdf``);
        ``tTable()`` keeps the exact df per coefficient.
        """
        is_lme = self.kind == "lme"
        lhs = str(getattr(self, "formula", "")).split("~", 1)[0].strip() or "y"
        left = [
            ("Dep. Variable:", lhs),
            ("Model:", "LME" if is_lme else "GLS"),
            ("Method:", self.method),
            ("No. Observations:", str(self.nobs)),
        ]
        if is_lme:
            groups = self.ranef if isinstance(self.ranef, dict) else {self._pred_levels[0]["chain"][0]: self.ranef}
            group_label = self._pred_levels[0]["chain"][0] if getattr(self, "_pred_levels", None) else "Group"
            left.append(("No. Groups:", f"{len(next(iter(groups.values())))} ({group_label})"))
        else:
            group_label = "Group"
            left.append(("Structures:", structure_text(self)))
        right = [
            ("Log-Likelihood:", f"{self.loglik:.6f}"),
            ("AIC:", f"{self.aic():.4f}"),
            ("BIC:", f"{self.bic():.4f}"),
            ("Residual SE:", _fmt_value(self.sigma)),
        ]
        stats = r2_icc(self)
        if stats:
            if "r2c" in stats:
                right.append(("R2 Marginal / Cond.:", f"{stats['r2m']:.3f} / {stats['r2c']:.3f}"))
                icc = stats.get("icc")
                if icc is not None and np.isfinite(icc):
                    right.append((f"ICC ({group_label}):", f"{icc:.3f}"))
            else:
                right.append(("R2:", f"{stats['r2m']:.3f}"))
        right.append(("Converged:", str(self.converged)))
        left_extra = [
            ("Fixed effects:", str(len(self.fixef))),
            ("Variance params:", str(len(getattr(self, "theta", [])))),
            ("Free parameters:", str(getattr(self, "npar", len(self.fixef)))),
            ("Residual df:", str(max(self.nobs - len(self.fixef), 1))),
        ]
        right_extra = [("Scale (variance):", _fmt_value(self.sigma ** 2))]
        left, right = balanced_info(left, right, left_extra, right_extra)
        df = _summary_df(getattr(self, "_fixdf", None))
        random_spec = getattr(self, "random_spec", None)
        level_names = [list(getattr(pl.get("spec"), "column_names", []) or []) for pl in getattr(self, "_pred_levels", [])]
        blocks = [fixed_block(self.fixef, self.se, "t", df)]
        blocks += lm_varcorr_blocks(
            self.varcorr,
            random_spec=random_spec,
            chain=[g for g, _ in random_spec] if random_spec else [],
            level_names=level_names,
            cor_spec=getattr(self, "cor_spec", None),
            var_spec=getattr(self, "var_spec", None),
            data=getattr(self, "_data", None),
            var_gcodes=getattr(self, "var_gcodes", None),
            random_blocks=False,
            residual=False,
        )
        # lme: one table per grouping level plus the residual row;
        # gls: a single residual-only table (no random effects).
        tables = list(lm_random_tables(self))
        resid_table = standardized_residuals_table(self)
        if resid_table is not None:
            tables.append(resid_table)
        notes = [self.warning] if self.warning else []
        details = structure_details(self)
        if details and details != structure_text(self):
            notes.append(f"Structures: {details}.")
        return build_summary(
            title="Linear Mixed-Effects Model Results" if is_lme else "Generalized Least Squares Results",
            subtitle=f"Formula: {getattr(self, 'formula', '')}",
            left=left,
            right=right,
            blocks=blocks,
            tables=tables,
            dist="t",
            df=df,
            alpha=alpha,
            notes=notes,
        )


class NlmeFit(_FitBase):
    """Result of :func:`nlme` and :func:`gnls`.

    Nonlinear fits store the fixed-effect parameters of the nonlinear model
    (``fixef``), the variance/correlation components, the random effects for
    :func:`nlme` and the prediction metadata required to evaluate the model on
    new data.

    Attributes
    ----------
    fixef : pandas.Series
        Estimated fixed-effect parameters (one entry per parameter coefficient).
    se : pandas.Series
        Standard errors of the fixed effects.
    sigma : float
        Residual standard deviation.
    loglik : float
        Maximized log-likelihood.
    nobs : int
        Number of observations used in the fit.
    converged : bool
        Whether the optimizer reported convergence.
    theta : numpy.ndarray
        Estimated variance/correlation parameters.
    warning : str or None
        Convergence or numerical warning emitted by the engine.
    kind : str
        ``"nlme"`` or ``"gnls"``.
    method : str
        ``"REML"`` or ``"ML"``.
    ranef
        BLUP random effects (``nlme``) or ``None`` (``gnls``).
    raw
        The underlying engine result object.
    """

    def __init__(self, raw, names, index, y, kind, method, extra):
        self.raw, self.names, self.index, self._y, self.kind, self.method = raw, names, index, y, kind, method
        self.__dict__.update(extra)
        self.fixef = pd.Series(np.array(raw.beta), index=names, name="Value")
        self.se = pd.Series(np.array(raw.se), index=names, name="Std.Error")

        if hasattr(raw, "vcov_beta") and np.asarray(raw.vcov_beta).size > 0:
            self.vcov = pd.DataFrame(np.array(raw.vcov_beta), index=names, columns=names)
        else:
            self.vcov = pd.DataFrame(np.nan, index=names, columns=names)

        self.sigma = math.sqrt(raw.sigma2)            # ML: RSS/N (R's gnls reports RSS/(N-p))
        self.loglik = raw.loglik
        self.converged = bool(raw.converged)
        self.theta = np.array(raw.theta)
        self.nobs = int(raw.n_used)
        self.warning = raw.warning
        self.npar = len(names) + len(raw.theta) + 1

    def _unorder(self, v):
        out = np.empty(len(self.index)); out[np.array(self.raw.order)] = np.array(v)
        return pd.Series(out, index=self.index)

    def fitted(self, level: Optional[int] = None) -> pd.Series:
        """level=0 is the population level; level=k adds the random effects of the first k levels; the default is all of them."""
        Q = len(getattr(self, "_pred_levels", []))
        level = Q if level is None else level
        if level == 0 or Q == 0:
            return self._unorder(self.raw.fitted_pop)
        if level == Q:
            return self._unorder(self.raw.fitted_cond)
        return self.predict(None, level=level)

    def residuals(self, type="response", level: Optional[int] = None) -> pd.Series:
        """Residuals at the requested random-effect level.

        Parameters
        ----------
        type : {"response", "pearson", "normalized"}, default "response"
            Residual type. ``"response"`` returns raw residuals ``y - fitted(level)``.
            ``"pearson"`` and ``"normalized"`` both return
            ``(y - yhat) / (sigma * g(mu_i, delta))``, where ``g`` is the fitted
            variance function (1 when there is none).  This puts GNLS/NLME
            residuals on the N(0, 1) scale used by the diagnostic Q-Q plots.
        level : int, optional
            ``0`` uses population-level fitted values; ``k`` adds the random effects
            of the first ``k`` levels. Defaults to all levels. For backward
            compatibility, ``residuals(level_int)`` is accepted.

        Returns
        -------
        pandas.Series
            Residuals on the original data index.
        """
        if isinstance(type, int) and not isinstance(type, bool):
            level, type = type, "response"
        if type is None:
            type = "response"
        type = str(type).lower()
        raw_resid = pd.Series(self._y - self.fitted(level).to_numpy(), index=self.index)
        if type in ("response", "raw"):
            return raw_resid
        sigma = _residual_sigma(self)
        scale = _variance_scales(self, np.asarray(self.fitted(level).to_numpy(), dtype=float))
        denom = sigma * scale
        if type in ("pearson", "normalized"):
            with np.errstate(divide="ignore", invalid="ignore"):
                return raw_resid / denom
        raise ValueError("type must be 'response', 'pearson' or 'normalized'")

    def predict(self, newdata: Optional[pd.DataFrame] = None, level: Optional[int] = None) -> pd.Series:
        """Like predict.nlme / predict.gnls: evaluate the nonlinear model on newdata (only
        the covariates are needed) with the fixed-effect parameters (level=0) plus the
        random effects of the first k levels. New groups: NaN with level >= 1 (as in R)."""
        levels = getattr(self, "_pred_levels", [])
        Q = len(levels)
        level = Q if level is None else level
        if not 0 <= level <= Q:
            raise ValueError(f"level must be between 0 and {Q}")
        if newdata is None:
            if level == 0 or level == Q or Q == 0:
                return self.fitted(0 if level == 0 else None)
            nd, idx = self._data.reset_index(drop=True), self.index
        else:
            nd = pd.DataFrame(newdata).reset_index(drop=True); idx = nd.index
        n = len(nd)
        nm = self._nm
        try:
            if nm.ss:
                xnew = eval_expr(nm.xexpr, nd)
                f = None
            else:
                f, _ = nm.make(nd, [np.arange(n)])
        except KeyError as e:
            raise ValueError(f"newdata does not have the column {e}") from e
        beta = self.fixef.to_numpy()
        Phi = np.zeros((n, len(nm.params)))
        for j, (spec, w, off) in enumerate(self._param_specs):
            Phi[:, j] = design_new(spec, nd) @ beta[off:off + w]
        for lv in levels[:level]:
            b = _lookup_b(lv["map"], _row_keys(nd, lv["chain"]), lv["q"])
            for (j, spec, w, c0) in lv["blocks"]:
                Phi[:, j] += (design_new(spec, nd) * b[:, c0:c0 + w]).sum(axis=1)
        vals = np.asarray(core.selfstart_eval(nm.ss, _F(Phi), _V(xnew)), float) if nm.ss else np.asarray(f(_F(Phi), 0), float)
        return pd.Series(vals, index=idx)

    def tTable(self, ddf: str = "containment", *, df: str | None = None) -> pd.DataFrame:
        """Coefficient table with optional Satterthwaite denominator df.

        The nonlinear model uses its final linearized working model, so the
        approximation follows the same REML covariance model as ``lme``.
        """
        if df is not None:
            ddf = df
        method = str(ddf).strip().lower().replace("-", "_").replace(" ", "_")
        if method in ("kr", "kenward", "kenward_roger"):
            raise NotImplementedError("Kenward-Roger df are not implemented yet")
        if method not in ("containment", "residual", "satterthwaite", "satt"):
            raise ValueError("ddf must be 'containment', 'residual' or 'satterthwaite'")
        if method in ("satterthwaite", "satt"):
            from ._ddf import ddf_for_fit, require_reml

            fit = require_reml(self, "Satterthwaite degrees of freedom")
            eye = np.eye(len(fit.fixef))
            dfs = pd.Series(ddf_for_fit(fit, eye), index=fit.fixef.index, name="DF")
            return _t_table(fit.fixef, fit.se, dfs)
        return _t_table(self.fixef, self.se, self._fixdf)

    @property
    def ranef(self):
        """Estimated random effects, as a DataFrame indexed by group."""
        return self._ranef

    def summary(self, alpha: float = 0.05):
        """statsmodels-style summary (statcraft.summary) of the nlme/gnls fit.

        The table uses a single t df (the common value or the median of ``_fixdf``);
        ``tTable()`` keeps the exact df per coefficient.
        """
        is_nlme = self.kind == "nlme"
        lhs = str(getattr(self, "formula", "")).split("~", 1)[0].strip() or "y"
        left = [
            ("Dep. Variable:", lhs),
            ("Model:", "NLME" if is_nlme else "GNLS"),
            ("Method:", self.method),
            ("No. Observations:", str(self.nobs)),
        ]
        group_label = "Group"
        if is_nlme:
            random_spec = list(getattr(self, "random_spec", None) or [])
            group_label = random_spec[0][0] if random_spec else "Group"
            ranef = getattr(self, "_ranef", None)
            first = ranef.get(group_label) if isinstance(ranef, dict) else ranef
            if first is not None:
                left.append(("No. Groups:", f"{len(first)} ({group_label})"))
        else:
            left.append(("Structures:", structure_text(self)))
        right = [
            ("Log-Likelihood:", f"{self.loglik:.6f}"),
            ("AIC:", f"{self.aic():.4f}"),
            ("BIC:", f"{self.bic():.4f}"),
            ("Residual SE:", _fmt_value(_residual_sigma(self))),
        ]
        stats = r2_icc(self)
        if stats:
            if "r2c" in stats:
                right.append(("R2 Marginal / Cond.:", f"{stats['r2m']:.3f} / {stats['r2c']:.3f}"))
                icc = stats.get("icc")
                if icc is not None and np.isfinite(icc):
                    right.append((f"ICC ({group_label}):", f"{icc:.3f}"))
            else:
                right.append(("R2:", f"{stats['r2m']:.3f}"))
        right.append(("Converged:", str(self.converged)))
        left_extra = [
            ("Fixed effects:", str(len(self.fixef))),
            ("Variance params:", str(len(getattr(self, "theta", [])))),
            ("Free parameters:", str(getattr(self, "npar", len(self.fixef)))),
            ("Residual df:", str(max(self.nobs - len(self.fixef), 1))),
        ]
        right_extra = [("Scale (variance):", _fmt_value(_residual_sigma(self) ** 2))]
        left, right = balanced_info(left, right, left_extra, right_extra)
        df = _summary_df(getattr(self, "_fixdf", None))
        blocks = [fixed_block(self.fixef, self.se, "t", df)]
        tables = list(nlme_random_tables(self))
        resid_table = standardized_residuals_table(self)
        if resid_table is not None:
            tables.append(resid_table)
        notes = [self.warning] if self.warning else []
        details = structure_details(self)
        if details and details != structure_text(self):
            notes.append(f"Structures: {details}.")
        return build_summary(
            title="Nonlinear Mixed-Effects Model Results" if is_nlme else "Generalized Nonlinear Least Squares Results",
            subtitle=f"Formula: {getattr(self, 'formula', '')}",
            left=left,
            right=right,
            blocks=blocks,
            tables=tables,
            dist="t",
            df=df,
            alpha=alpha,
            notes=notes,
        )


# ----------------------------------------------------------------------------------------------------------------------
# lme
# ----------------------------------------------------------------------------------------------------------------------
def _parse_random(random, groups, fixed_rhs: str) -> list[tuple[str, str]]:
    """Normalize the ``random`` argument into grouping levels.

    Parameters
    ----------
    random : str, mapping, sequence or None
        Random-effects specification accepted by :func:`lme`.
    groups : str or None
        Grouping factor used when ``random`` has no ``|`` term.
    fixed_rhs : str
        Fixed-effects right-hand side used when ``random`` is ``None``.

    Returns
    -------
    list[tuple[str, str]]
        ``(grouping variable, random-effects right-hand side)`` pairs ordered
        from the outer level to the innermost nested level.
    """
    if isinstance(random, Mapping):
        return [(g, parse_form(f).rhs if "~" in f or "|" in f else f) for g, f in random.items()]
    if isinstance(random, (list, tuple)):
        return [(g, parse_form(f).rhs) for g, f in random]
    if random is None:
        random = "~" + fixed_rhs
    fp = parse_form(random)
    chain = fp.groups
    if not chain:
        if groups is None:
            raise ValueError("specify the groups: random='~ x | g' or groups='g'")
        chain = parse_form(groups if "~" in groups or "|" in groups else "~1|" + groups).groups or [groups.strip("~ ")]
    return [(g, fp.rhs) for g in chain]


def _fit_lme(fixed: str, data, random=None, groups: Optional[str] = None, correlation: Optional[CorStruct] = None,
             weights: Optional[VarStruct] = None, method: str = "REML", na_action: str = "omit",
             control: Optional[Mapping] = None, pd_struct=None, cor_prefix: bool = False) -> LmeFit:
    """Fit a linear mixed-effects model (R ``nlme::lme`` equivalent).

    Parameters
    ----------
    fixed : str
        Two-sided formula for the response and the fixed effects, for example
        ``"distance ~ age + Sex"``.
    data : pandas.DataFrame or array-like
        Data containing every column referenced by the formulas.
    random : str, mapping or sequence, optional
        Random-effects formula.  Accepted forms are ``"~ age | Subject"``,
        ``"~ 1 | Dog/Side"`` (nested), a ``{group: formula}`` mapping such as
        ``{"Dog": "~ day", "Side": "~ 1"}``, or a formula without groups
        (``"~ age"``) combined with ``groups``.
    groups : str, optional
        Grouping factor used when ``random`` does not contain ``|``.
    correlation : CorStruct, optional
        Within-group residual correlation structure (see
        :mod:`statcraft.nlme.structs`), for example
        ``corAR1(form="~ 1 | Subject")``.
    weights : VarStruct, optional
        Variance function describing heteroscedasticity, for example
        ``varIdent(form="~ 1 | g")`` or ``varPower(form="~ x")``.
    method : {"REML", "ML"}, optional
        Estimation criterion; ``"REML"`` by default.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied to the columns used by the model.
    control : mapping, optional
        Engine controls such as ``em_iter``, ``optimizer``, ``max_iter``,
        ``gtol``, ``sigma``, ``fixed_theta`` or ``analytic_grad``.
    pd_struct : PdStruct, sequence or mapping, optional
        Parametrization of the random-effects covariance matrix ``D``.  A
        single structure applies to every level; a sequence or mapping can be
        used to specify one structure per grouping level.
    cor_prefix : bool, optional
        If ``True``, keep a correlation grouping that is coarser than the
        random-effects grouping instead of replacing it with the full chain
        (the R ``lme`` behavior).

    Returns
    -------
    LmeFit
        Fitted model with fixed effects, variance components, random effects
        and prediction/inference helpers.
    """
    if method not in ("REML", "ML"):
        raise ValueError("method must be 'REML' or 'ML'")
    df = _as_df(data)
    fp = parse_form(fixed)
    if fp.lhs is None:
        raise ValueError("fixed needs a response: 'y ~ x'")
    levels = _parse_random(random, groups, fp.rhs)
    chain = [g for g, _ in levels]
    forms = [fp.lhs, fp.rhs] + [r for _, r in levels] + chain + [(correlation.form if correlation else None)] + _var_forms(weights)
    cols = collect_columns(forms, df)
    df2, keep = handle_na(df, cols, na_action)
    y = eval_expr(fp.lhs, df2)
    X, names, fspec = design_spec(fp.rhs, df2)
    labels, lev0 = factor_codes(df2[chain[0]])
    Z0, z0names, zspec0 = design_spec(levels[0][1], df2)
    zspecs = [zspec0]
    inner_Z, inner_sub, inner_names = [], [], []
    for l in range(1, len(levels)):
        Zl, zn, zsp = design_spec(levels[l][1], df2); inner_Z.append(_F(Zl)); inner_names.append(zn); zspecs.append(zsp)
        inner_sub.append(_I(nested_codes(df2, chain[:l + 1])))
    cspec, time, coords, cor_sub, _ = _cor_inputs(correlation, df2, chain, cor_prefix)
    vb = _var_spec(weights, df2, labels)
    spec = core.LmeSpec(); spec.reml = method == "REML"; spec.cor = cspec; spec.var = vb.spec
    if pd_struct is not None:
        lst = pd_struct if isinstance(pd_struct, (list, tuple)) else ([pd_struct.get(g, "logchol") for g in chain] if isinstance(pd_struct, Mapping) else [pd_struct] * len(levels))
        spec.pd_specs = [_pd_spec(as_pd(p)) for p in lst]
    _apply_control(spec, control, "lme")
    raw, tl, model = core.fit_lme(_V(y), _F(X), _F(Z0), _I(labels), spec, None if time is None else _V(time), coords,
                                  vb.strata, vb.cov, inner_Z, inner_sub, cor_sub)
    fit = LmeFit(raw, tl, names, df2.index, y, "lme", method,
                 dict(sigma_fixed=spec.sigma_fixed > 0, n_fixed_theta=len(spec.fixed), formula=fixed, random_spec=levels,
                      cor_spec=correlation, var_spec=weights, var_gcodes=labels))
    fit._ddf_model = model
    fit._reml_refit = lambda: _fit_lme(fixed, data, random=random, groups=groups, correlation=correlation, weights=weights,
                                       method="REML", na_action=na_action, control=control, pd_struct=pd_struct,
                                       cor_prefix=cor_prefix)
    # random effects
    gl = lev0[np.array(raw.labels)]
    perm = np.argsort(np.array(raw.labels))            # the engine orders groups by appearance; here, by factor level
    r0 = pd.DataFrame(np.vstack([np.atleast_1d(b) for b in raw.blup])[perm], index=pd.Index(gl[perm], name=chain[0]), columns=z0names)
    if len(levels) == 1:
        fit._ranef = r0
    else:
        out = {chain[0]: r0}
        for l in range(1, len(levels)):
            rows, idx = [], []
            for gi in perm:
                lab = raw.labels[gi]; in_g = np.flatnonzero(labels == lab)
                subs = pd.unique(inner_sub[l - 1][in_g])          # the engine numbers units by first appearance within the group
                for si, sid in enumerate(subs):
                    first = in_g[np.flatnonzero(inner_sub[l - 1][in_g] == sid)[0]]
                    idx.append(tuple(df2[c].iloc[first] for c in chain[:l + 1])); rows.append(np.atleast_1d(raw.blup_inner[gi][l - 1][si]))
            out["/".join(chain[:l + 1])] = pd.DataFrame(np.vstack(rows), index=pd.MultiIndex.from_tuples(idx, names=chain[:l + 1]), columns=inner_names[l - 1])
        fit._ranef = out
    fit.var_params = fit.varcorr
    # prediction and containment degrees of freedom
    fit._fixed_spec, fit._beta_vec, fit._data = fspec, np.array(raw.beta), df2
    maps = [dict(zip(r0.index, r0.to_numpy()))]
    if len(levels) > 1:
        for l in range(1, len(levels)):
            fr = fit._ranef["/".join(chain[:l + 1])]
            maps.append(dict(zip(fr.index, fr.to_numpy())))
    fit._pred_levels = [dict(spec=zspecs[l], chain=chain[:l + 1], map=maps[l]) for l in range(len(levels))]
    fit._fixdf = _get_fix_df(X, [nested_codes(df2, chain[:k + 1]) for k in range(len(levels))])
    return fit


# ----------------------------------------------------------------------------------------------------------------------
# gls
# ----------------------------------------------------------------------------------------------------------------------
def _fit_gls(model: str, data, correlation: Optional[CorStruct] = None, weights: Optional[VarStruct] = None, method: str = "REML",
             na_action: str = "omit", control: Optional[Mapping] = None) -> LmeFit:
    """Fit a generalized least-squares model (R ``nlme::gls`` equivalent).

    Parameters
    ----------
    model : str
        Two-sided formula for the response and the fixed effects, for example
        ``"y ~ x + factor(g)"``.
    data : pandas.DataFrame or array-like
        Data containing every column referenced by the formulas.
    correlation : CorStruct, optional
        Residual correlation structure.  Its grouping factor (the part after
        ``|``, as in ``"~ time | subject"``) defines the independent groups;
        without a grouping factor the model has a single group.
    weights : VarStruct, optional
        Variance function describing heteroscedasticity.
    method : {"REML", "ML"}, optional
        Estimation criterion; ``"REML"`` by default.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied to the columns used by the model.
    control : mapping, optional
        Engine controls such as ``optimizer``, ``max_iter``, ``gtol``,
        ``sigma`` or ``fixed_theta``.

    Returns
    -------
    LmeFit
        Fitted model with ``kind == "gls"``.
    """
    df = _as_df(data)
    fp = parse_form(model)
    if fp.lhs is None:
        raise ValueError("model needs a response: 'y ~ x'")
    cols = collect_columns([fp.lhs, fp.rhs, correlation.form if correlation else None] + _var_forms(weights), df)
    df2, _ = handle_na(df, cols, na_action)
    y = eval_expr(fp.lhs, df2)
    X, names, fspec = design_spec(fp.rhs, df2)
    cspec, time, coords, _, groups = _cor_inputs(correlation, df2, [])
    labels = nested_codes(df2, groups) if groups else np.zeros(len(df2), np.int32)
    vb = _var_spec(weights, df2, labels)
    spec = core.GlsSpec(); spec.reml = method == "REML"; spec.cor = cspec; spec.var = vb.spec
    _apply_control(spec, control, "gls")
    raw, tl, ddmodel = core.fit_gls(_V(y), _F(X), _I(labels), spec, None if time is None else _V(time), coords, vb.strata, vb.cov)
    fit = LmeFit(raw, tl, names, df2.index, y, "gls", method,
                 dict(sigma_fixed=spec.sigma_fixed > 0, n_fixed_theta=len(spec.fixed), formula=model,
                      cor_spec=correlation, var_spec=weights, var_gcodes=labels))
    fit._ddf_model = ddmodel
    fit._reml_refit = lambda: _fit_gls(model, data, correlation=correlation, weights=weights, method="REML",
                                       na_action=na_action, control=control)
    fit._fixed_spec, fit._beta_vec, fit._data, fit._pred_levels = fspec, np.array(raw.beta), df2, []
    fit._fixdf = np.full(X.shape[1], float(len(y) - X.shape[1]))      # summary.gls: N - p
    return fit


# ----------------------------------------------------------------------------------------------------------------------
# nonlinear model: parameters, function and Jacobian
# ----------------------------------------------------------------------------------------------------------------------
def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""; continue
        depth += (ch in "([") - (ch in ")]"); cur += ch
    out.append(cur.strip()); return out


def _parse_param_forms(spec, what: str) -> dict[str, str]:
    """Normalize nonlinear-model parameter formulas.

    Parameters
    ----------
    spec : str, sequence or mapping
        Accepted forms are ``"a + b ~ 1"``, ``["a ~ 1", "b ~ trt"]`` and
        ``{"a": "1", "b": "trt"}``.
    what : str
        Role of the formulas (``"fixed"`` or ``"random"``), used to build
        precise error messages.

    Returns
    -------
    dict[str, str]
        Mapping from parameter name to its right-hand-side formula.

    Raises
    ------
    ValueError
        If a formula does not declare its parameters on the left-hand side.
    """
    out: dict[str, str] = {}
    if isinstance(spec, Mapping):
        for k, v in spec.items():
            out[k] = parse_form(v).rhs if ("~" in v or "|" in v) else v
        return out
    for item in ([spec] if isinstance(spec, str) else list(spec)):
        fp = parse_form(item)
        if fp.lhs is None:
            raise ValueError(f"{what}: each formula needs the parameters on the left-hand side ('a + b ~ 1')")
        for n in fp.lhs.split("+"):
            out[n.strip()] = fp.rhs
    return out


def _resolve_params(params, start):
    """Fill in optional parameter formulas from the order of ``start``.

    When ``params`` is omitted, the keys of a mapping ``start`` are used as
    parameter names in insertion order and each one gets the fixed-effects
    formula ``"1"``. A covariate formula still needs an explicit ``params``.
    """
    if params is not None:
        return params
    if isinstance(start, Mapping):
        names = list(start)
        if not all(isinstance(n, str) for n in names):
            raise TypeError("start keys must be parameter names when params is omitted")
        return {n: "1" for n in names}
    raise TypeError(
        "params is required unless start is a mapping; pass "
        "params={'a': '1', 'b': '1'} or start={'a': ..., 'b': ...} "
        "(the key order defines the parameter order)"
    )


class _NlModel:
    """Parsed nonlinear model used by :func:`nlme` and :func:`gnls`.

    The class resolves a ``selfStart`` model (``SSlogis``, ``SSasymp``, ...) or
    a user expression, normalizes the parameter formulas and exposes the
    callables required by the engine (``f``, plus an optional analytic
    Jacobian when SymPy can differentiate the expression).
    """

    def __init__(self, model: str, params, df: pd.DataFrame):
        self.formula = model
        fp = parse_form(model)
        if fp.lhs is None:
            raise ValueError("the model needs a response: 'y ~ f(...)'")
        self.lhs, self.rhs = fp.lhs, fp.rhs
        m = re.fullmatch(r"\s*(SS\w+)\s*\((.*)\)\s*", self.rhs, re.S)
        self.ss = None
        self.fixed_rhs = _parse_param_forms(params, "params")
        if m:
            nm = m.group(1)
            if nm not in _SS:
                raise ValueError(f"unsupported selfStart: {nm} (available: {sorted(_SS)})")
            args = _split_top(m.group(2))
            self.ss = _SS[nm]
            self.xexpr = args[0]
            self.params = args[1:]
            if len(self.params) != len(core.selfstart_params(self.ss)):
                raise ValueError(f"{nm} expects {len(core.selfstart_params(self.ss))} parameters")
        else:
            self.params = list(self.fixed_rhs)
            self.expr = self.rhs
        for p in self.params:
            if p not in self.fixed_rhs:
                raise ValueError(f"missing the fixed-effects formula for parameter '{p}'")
        if not self.ss:
            toks = {t for t in _IDENT.findall(self.rhs) if t not in _FUNCS}
            self.covs = [t for t in toks if t in df.columns and t not in self.params]
        else:
            self.covs = []

    def exprs(self):
        return [self.lhs] + ([self.xexpr] if self.ss else self.covs) + list(self.fixed_rhs.values())

    def make(self, df: pd.DataFrame, rows_by_group: list[np.ndarray]):
        """-> (f, jac) for the engine."""
        if self.ss:
            x = eval_expr(self.xexpr, df)
            self.x_all = x
            fns = core.make_selfstart_fns(self.ss, [_V(x[r]) for r in rows_by_group])
            return fns, None
        covs = {c: df[c].to_numpy(dtype=float) for c in self.covs}
        cbg = [{c: v[r] for c, v in covs.items()} for r in rows_by_group]
        code = compile(r_to_py(self.expr), "<model>", "eval")
        params = self.params
        ns = {"__builtins__": {}, **_FUNCS}

        def f(Phi, g):
            env = {p: Phi[:, i] for i, p in enumerate(params)}; env.update(cbg[g])
            return np.broadcast_to(np.asarray(eval(code, ns, env), dtype=float), (Phi.shape[0],)).copy()  # noqa: S307
        jac = None
        try:
            import sympy as sp
            syms = {n: sp.Symbol(n) for n in params + self.covs}
            loc = dict(syms); loc.update({"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
                                         "abs": sp.Abs, "pi": sp.pi})
            e = sp.sympify(r_to_py(self.expr), locals=loc)
            args = [syms[n] for n in params + self.covs]
            dfs = [sp.lambdify(args, sp.diff(e, syms[p]), "numpy") for p in params]

            def jac(Phi, g):  # noqa: F811
                vals = [Phi[:, i] for i in range(len(params))] + [cbg[g][c] for c in self.covs]
                return np.column_stack([np.broadcast_to(np.asarray(d(*vals), float), (Phi.shape[0],)) for d in dfs])
        except Exception:
            jac = None   # no sympy (or non-differentiable expression): numeric Jacobian in the engine
        return f, jac


class _PyModel:
    """Nonlinear model defined by a plain Python callable.

    Mirror of :class:`_NlModel` for models written as a plain function instead
    of a formula. The portable form is ``def f(x, p): ...``, where ``x`` and
    ``p`` are named views that can be unpacked directly or indexed by name.
    Any argument names work when there is more than one parameter; with one
    parameter, a second argument named ``p``, ``params``, ``parameters`` or
    ``theta`` also selects the vector form. A single-entry view can be used
    directly (``a = params``) or unpacked (``a, = params``). With a single
    predictor, ``x`` can be used directly as the predictor values. The named form
    ``def f(x, a, b): ...`` is still accepted and invoked as
    ``func(*exogenous, *parameters)``. In both cases the model must return one
    fitted value per row.

    Parameters
    ----------
    func : callable
        Model function, for example ``def growth(x, p)`` or
        ``def growth(x, Asym, xmid, scal)``.
    response : str
        Name of the response column in the data.
    params : str, sequence or mapping, optional
        Fixed-effects parameter formulas.  Their order defines the order in
        which the parameters are passed to ``func``.  When omitted, the keys
        of ``start`` are used in insertion order with formula ``"1"``.
    df : pandas.DataFrame
        Data used to validate the exogenous columns.
    predictors : str or list of str, optional
        Exogenous columns. For a vector-form model they are inferred from the
        first argument when it names a data column, otherwise they are
        required. For the named form they are inferred from the signature as
        the arguments that are not parameters.
    jac : callable, optional
        Analytical Jacobian with the same signature as ``func``.  It must
        return an ``(n, k)`` array; when omitted the engine differentiates
        numerically.
    """

    # Interface shared with _NlModel: a callable model is never selfStarting
    # and has no expression to evaluate for newdata.
    ss = None
    xexpr = None

    def __init__(self, func: Callable, response: Optional[str], params, df: pd.DataFrame,
                 predictors=None, jac: Optional[Callable] = None):
        if not callable(func):
            raise TypeError("model must be a formula string or a callable (def/lambda).")
        if response is None:
            raise ValueError("response is required when model is a callable, e.g. response='circumference'.")
        if jac is not None and not callable(jac):
            raise TypeError("jac must be a callable with the same signature as the model function.")

        name = getattr(func, "__name__", "model_func")
        if name == "<lambda>":
            name = "model_func"
        self.func, self.jac_func = func, jac
        self.lhs = self.response = response
        self.fixed_rhs = _parse_param_forms(params, "params")
        self.params = list(self.fixed_rhs)
        if not self.params:
            raise ValueError("params must declare at least one parameter, e.g. params={'a': '1', 'b': '1'}, "
                             "or provide start={'a': 1, 'b': 0.1}.")

        sig = inspect.signature(func)
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values()):
            raise TypeError(f"'{name}' cannot use *args/**kwargs: every argument must be named so that "
                            "exogenous variables and parameters can be told apart.")
        names = [p.name for p in sig.parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        # A two-argument callable with more than one parameter cannot be the
        # named form: that needs one predictor argument plus one argument per
        # parameter. With one parameter, a reserved parameter-view name keeps
        # ``def f(x, params)`` distinct from ``def f(time, a)``.
        reserved_p = (len(names) == 2 and names[1] in PARAM_VIEW_NAMES
                      and names[1] not in self.params)
        self.vector_form = (names == ["x", "p"]
                            or (len(names) == 2 and (len(self.params) > 1 or reserved_p)))
        if self.vector_form and predictors is None and names[0] in df.columns:
            predictors = names[0]
        if self.vector_form and predictors is None:
            raise ValueError(
                f"'{name}' uses the two-argument vector convention; predictors must say which "
                f"columns go in the first argument (or name that column '{names[0]}'), e.g. "
                "predictors=['Time', 'Dose']."
            )

        if predictors is None:
            self.covs = [n for n in names if n not in self.fixed_rhs]
        elif isinstance(predictors, str):
            self.covs = [predictors]
        else:
            self.covs = list(predictors)
        if not self.covs:
            raise ValueError(f"'{name}' has no exogenous argument: the model needs at least one column "
                             "(an argument that is not listed in 'params').")

        n_exog = len(self.covs)
        if not self.vector_form:
            if names[n_exog:] != self.params:
                raise ValueError(
                    f"'{name}' must take {n_exog} exogenous argument(s) followed by the parameters "
                    f"{self.params}, in that order; got arguments {names}. The exogenous arguments are "
                    f"mapped positionally to the columns {self.covs}."
                )

        missing = [c for c in [response, *self.covs] if c not in df.columns]
        if missing:
            raise KeyError(f"the data does not contain the column(s) {missing}. The model function "
                           f"takes {names[:n_exog]} as exogenous argument(s), mapped to the columns "
                           f"{self.covs}; rename the arguments to match the data or pass predictors.")
        bad = [c for c in self.covs if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise TypeError(f"exogenous variable(s) {bad} must be numeric to be used in a nonlinear model.")

        shown = self.covs + self.params if self.vector_form else names
        self.formula = f"{response} ~ {name}({', '.join(shown)})"

    def exprs(self):
        return [self.lhs, *self.covs, *self.fixed_rhs.values()]

    def make(self, df: pd.DataFrame, rows_by_group: list[np.ndarray]):
        """-> (f, jac) for the engine, mirroring :meth:`_NlModel.make`."""
        covs = {c: df[c].to_numpy(dtype=float) for c in self.covs}
        cbg = [{c: v[r] for c, v in covs.items()} for r in rows_by_group]
        func, params, names = self.func, self.params, self.covs
        k = len(params)
        n_name = getattr(func, "__name__", "model_func")

        def call(fn, Phi, g):
            env = cbg[g]
            if self.vector_form:
                xv = ModelView(self.covs, [env[c] for c in self.covs], label="x", columnar=True)
                pv = ModelView(params, [Phi[:, i] for i in range(k)], label="p")
                return fn(xv, pv)
            return fn(*[env[c] for c in names], *[Phi[:, i] for i in range(k)])

        def f(Phi, g):
            out = np.asarray(call(func, Phi, g), dtype=float).reshape(-1)
            if out.size not in (1, Phi.shape[0]):
                raise ValueError(f"'{n_name}' returned {out.size} value(s) for {Phi.shape[0]} row(s).")
            return np.broadcast_to(out, (Phi.shape[0],)).copy()

        jac = None
        if self.jac_func is not None:
            user_jac = self.jac_func

            def jac(Phi, g):  # noqa: F811
                out = np.asarray(call(user_jac, Phi, g), dtype=float)
                if out.size != Phi.shape[0] * k:
                    raise ValueError(f"jac returned {out.size} value(s); expected {Phi.shape[0]} x {k}.")
                return out.reshape(Phi.shape[0], k)

        return f, jac


def _nl_model(model, params, df: pd.DataFrame, response=None, predictors=None, jac=None):
    """Build the parsed-model object used by :func:`_fit_nlme`/:func:`_fit_gnls`.

    ``model`` may be a formula string (:class:`_NlModel`) or a Python callable
    (:class:`_PyModel`).  ``response``, ``predictors`` and ``jac`` are only valid
    for callables.
    """
    if callable(model):
        return _PyModel(model, response, params, df, predictors=predictors, jac=jac)
    if not isinstance(model, str):
        raise TypeError("model must be a formula string or a callable (def/lambda).")
    if response is not None or predictors is not None:
        raise ValueError("response and predictors are only used when model is a callable; "
                         "the formula already declares the response and the covariates.")
    if jac is not None:
        raise ValueError("jac is only supported when model is a callable.")
    return _NlModel(model, params, df)


def _stack_rows(blocks_by_param, rows, k, offs, P):
    """Stack the per-row parameter designs of one group.

    Parameters
    ----------
    blocks_by_param : sequence of ``(parameter index, design)``
        Design matrices for each model parameter.
    rows : numpy.ndarray
        Row indices belonging to the group.
    k : int
        Number of model parameters.
    offs : numpy.ndarray
        Column offsets of each parameter block in the combined design.
    P : int
        Total number of fixed-effect coefficients.

    Returns
    -------
    numpy.ndarray
        Fortran-ordered ``(n * k, P)`` matrix whose row block ``j`` contains
        the parameter designs for the ``j``-th observation.
    """
    arr = np.zeros((len(rows), k, P))
    for j, b in blocks_by_param:
        arr[:, j, offs[j]:offs[j + 1]] = b[rows]
    return _F(arr.reshape(len(rows) * k, P))


def _design_params(nm: _NlModel, df: pd.DataFrame, codes: np.ndarray, order: np.ndarray):
    """Build the fixed-effect designs of the nonlinear parameters.

    Parameters
    ----------
    nm : _NlModel
        Parsed nonlinear model.
    df : pandas.DataFrame
        Data used in the fit.
    codes : numpy.ndarray
        Integer group codes for the top-level grouping factor.
    order : numpy.ndarray
        Group codes in the order used by the engine.

    Returns
    -------
    A_list : list[numpy.ndarray]
        ``k x p`` parameter design of each group (constant within the group).
    A_rows : list[numpy.ndarray]
        When a parameter formula varies within a group (for example
        ``Asym ~ x``), the per-row design ``(n_g * k) x p`` of each group;
        otherwise an empty list.
    beta_names : list[str]
        Names of the fixed-effect coefficients, prefixed by parameter name.
    offs : numpy.ndarray
        Column offsets of each parameter in ``A_list``.
    specs : list[formulaic.ModelSpec]
        Formulaic specifications used to rebuild the designs on new data.
    """
    blocks, bn, specs = [], [], []
    for p in nm.params:
        Xp, cn, sp = design_spec(nm.fixed_rhs[p], df)
        blocks.append(Xp); specs.append(sp); bn += [p if c in ("(Intercept)", "Intercept") else f"{p}.{c}" for c in cn]
    k = len(nm.params); widths = [b.shape[1] for b in blocks]; offs = np.concatenate([[0], np.cumsum(widths)]); P = int(offs[-1])
    rows_g = [np.flatnonzero(codes == c) for c in order]
    varies = any(not np.allclose(b[r], b[r[0]]) for b in blocks for r in rows_g)
    A_list, A_rows = [], []
    for rows in rows_g:
        A = np.zeros((k, P))
        for j, b in enumerate(blocks):
            A[j, offs[j]:offs[j + 1]] = b[rows[0]]
        A_list.append(_F(A))
        if varies:
            A_rows.append(_stack_rows(list(enumerate(blocks)), rows, k, offs, P))
    return A_list, A_rows, bn, offs, specs


def _start_vector(nm: _NlModel, start, bn, offs, y, df):
    """Build the initial fixed-effect vector for a nonlinear fit.

    Parameters
    ----------
    nm : _NlModel
        Parsed nonlinear model.
    start : mapping, sequence or None
        User initial values.  ``None`` is allowed only for ``selfStart``
        models, whose starting values are computed from the data.
    bn : list[str]
        Coefficient names, used for mapping-based initialization.
    offs : numpy.ndarray
        Column offsets of each parameter in the coefficient vector.
    y : numpy.ndarray
        Response values used to compute ``selfStart`` initials.
    df : pandas.DataFrame
        Data used in the fit.

    Returns
    -------
    numpy.ndarray
        Initial coefficient vector with one entry per fixed-effect parameter.
    """
    p = int(offs[-1]); b0 = np.zeros(p)
    if start is None:
        if not nm.ss:
            raise ValueError("start is required for models that are not selfStart (e.g. start={'a': 1, 'b': 0.1})")
        init = np.asarray(core.selfstart_initial(nm.ss, _V(nm.x_all), _V(y)))
        for j in range(len(nm.params)): b0[offs[j]] = init[j]
        return b0
    if isinstance(start, Mapping):
        for k, v in start.items():
            if k in bn: b0[bn.index(k)] = v
            elif k in nm.params: b0[offs[nm.params.index(k)]] = v
            elif k.endswith(".(Intercept)") and k[: -len(".(Intercept)")] in nm.params:
                # Backwards compatibility with the old ``Asym.(Intercept)`` labels.
                b0[offs[nm.params.index(k[: -len(".(Intercept)")])]] = v
            else: raise ValueError(f"start: unknown name {k!r} (parameters {nm.params}, coefficients {bn})")
        return b0
    arr = np.asarray(start, float)
    if arr.size == len(nm.params):
        for j in range(len(nm.params)): b0[offs[j]] = arr[j]
        return b0
    if arr.size == p: return arr
    raise ValueError("start: incorrect length")


def _random_design(nm: _NlModel, random, df: pd.DataFrame):
    """Normalize the random-effects specification of a nonlinear model.

    Parameters
    ----------
    nm : _NlModel
        Parsed nonlinear model (provides the parameter names).
    random : str or mapping
        ``"a + b ~ 1 | g"``, ``"a ~ 1 | Dog/Side"`` or a mapping such as
        ``{"Dog": "a ~ 1", "Side": "b ~ 1"}``.
    df : pandas.DataFrame
        Data providing the grouping columns.

    Returns
    -------
    list[tuple[str, dict[str, str]]]
        ``(group, {parameter: right-hand side})`` pairs ordered from the outer
        level to the innermost nested level.
    """
    if isinstance(random, Mapping):
        lv = [(g, _parse_param_forms(f, "random")) for g, f in random.items()]
    else:
        fp = parse_form(random)
        if not fp.groups: raise ValueError("random needs groups: 'a + b ~ 1 | g' (or a dict {group: 'a ~ 1'})")
        names = [n.strip() for n in fp.lhs.split("+")] if fp.lhs else []
        lv = [(g, {n: fp.rhs for n in names}) for g in fp.groups]
    return lv


def _na_use(na_pattern, df: pd.DataFrame, y: np.ndarray):
    if na_pattern is None or na_pattern is False:
        return None
    if na_pattern is True:
        return ~np.isnan(y)
    if callable(na_pattern):
        return np.asarray(na_pattern(df), bool)
    return np.asarray(na_pattern, bool)


def _fit_nlme(model, data, params=None, random=None, start=None, correlation: Optional[CorStruct] = None, weights: Optional[VarStruct] = None,
              method: str = "ML", na_action: str = "omit", na_pattern=None, control: Optional[Mapping] = None, pd_struct=None,
              cor_prefix: bool = False, response: Optional[str] = None, predictors=None, jac: Optional[Callable] = None) -> NlmeFit:
    """Fit a nonlinear mixed-effects model (R ``nlme::nlme`` equivalent).

    Parameters
    ----------
    model : str or callable
        Two-sided formula for the response and the nonlinear model, either a
        self-starting model such as ``"height ~ SSasymp(age, Asym, R0, lrc)"``
        or an expression such as ``"y ~ a * exp(-b * x)"``.  Alternatively a
        plain Python function ``def f(x, a, b)``; in that case ``response``
        must name the response column and the argument names are, in order,
        the exogenous variables followed by the parameters in ``params``.
    data : pandas.DataFrame or array-like
        Data containing every column referenced by the formulas.
    params : str, sequence or mapping, optional
        Fixed-effects parameter formulas.  Accepted forms are
        ``"Asym + R0 + lrc ~ 1"``, a sequence such as
        ``["a ~ 1", "b ~ trt"]`` or a mapping ``{"a": "1", "b": "trt"}``.
        When omitted, the keys of a mapping ``start`` are used in insertion
        order with formula ``"1"``. Covariates must be constant within each
        top-level group.
    random : str or mapping
        Random-effects specification.  Accepted forms are
        ``"Asym ~ 1 | Seed"``, ``"a + b ~ 1 | Dog/Side"`` (nested) and a
        mapping such as ``{"Dog": "a ~ 1", "Side": "b ~ 1"}``.
    start : mapping or sequence, optional
        Initial fixed-effect values, for example ``{"Asym": 103, ...}`` or a
        flat vector.  Optional for ``selfStart`` models, whose starting values
        are computed automatically.  Always required for a callable model;
        its key order supplies ``params`` when that argument is omitted.
    correlation : CorStruct, optional
        Within-group residual correlation structure.
    weights : VarStruct, optional
        Variance function describing heteroscedasticity.
    method : {"REML", "ML"}, optional
        Estimation criterion; ``"ML"`` by default.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied to the columns used by the model.
    na_pattern : bool, mask or callable, optional
        R ``naPattern`` equivalent.  ``True`` evaluates only the model
        function for missing responses; a boolean mask or ``callable(df)``
        selects the rows used in the objective.
    control : mapping, optional
        Engine controls.  ``{"auto_start": False}`` disables the initial
        ``nlsList`` stage used to initialize the fixed effects and ``D``.
    pd_struct : PdStruct, sequence or mapping, optional
        Parametrization of the random-effects covariance matrix ``D``.
    cor_prefix : bool, optional
        Keep a correlation grouping coarser than the random-effects grouping
        instead of replacing it with the full chain.
    response : str, optional
        Response column name.  Required when ``model`` is a callable and
        ignored for formulas.
    predictors : str or list of str, optional
        Exogenous column(s) of a callable model.  When omitted they are
        inferred from the signature as the arguments that are not parameters.
    jac : callable, optional
        Analytical Jacobian for a callable model, with the same signature as
        ``model`` and returning an ``(n, k)`` array.  By default the engine
        uses a numerical Jacobian.

    Returns
    -------
    NlmeFit
        Fitted nonlinear mixed-effects model.
    """
    df = _as_df(data)
    params = _resolve_params(params, start)
    nm = _nl_model(model, params, df, response=response, predictors=predictors, jac=jac)
    lv = _random_design(nm, random, df)
    chain = [g for g, _ in lv]
    other = [*nm.exprs(), *[r for _, d in lv for r in d.values()], *chain,
             correlation.form if correlation else None, *_var_forms(weights)]
    cols = collect_columns(other, df)
    if na_pattern is not None and na_pattern is not False:
        resp = set(names_in(nm.lhs, df.columns)); cols_ = [c for c in cols if c not in resp]
        df2, keep = handle_na(df, cols_, na_action)
    else:
        df2, keep = handle_na(df, cols, na_action)
    y = eval_expr(nm.lhs, df2)
    use = _na_use(na_pattern, df2, y)
    if use is not None:
        y = np.where(use, y, 0.0)
    codes, lev0 = factor_codes(df2[chain[0]])
    gorder = pd.unique(codes); G = len(gorder)          # the engine numbers groups by first appearance
    rows_by_group = [np.flatnonzero(codes == c) for c in gorder]
    f, jac = nm.make(df2, rows_by_group)
    A_list, A_rows, bn, offs, pspecs = _design_params(nm, df2, codes, gorder)
    k, p = len(nm.params), int(offs[-1])
    # Random-effect design B per grouping level (constant within the level or row-wise)
    B_list, B_rows, inner_B, inner_B_rows, inner_sub, qnames = [], [], [], [], [], []
    rblocks = []
    for li, (g, d) in enumerate(lv):
        cols_B, names_B = [], []; rblocks_lv = []
        for pn, rhs in d.items():
            if pn not in nm.params: raise ValueError(f"random: '{pn}' is not a model parameter")
            Xr, cn, rsp = design_spec(rhs, df2); j = nm.params.index(pn); cols_B.append((j, Xr)); names_B += [pn if c in ("(Intercept)", "Intercept") else f"{pn}.{c}" for c in cn]
            rblocks_lv.append((j, rsp, Xr.shape[1], sum(b[2] for b in rblocks_lv)))
        q = sum(x.shape[1] for _, x in cols_B); qnames.append(names_B); rblocks.append(rblocks_lv)
        boffs = np.concatenate([[0], np.cumsum([x.shape[1] for _, x in cols_B])]).astype(int)
        bblocks = [(j, Xr) for j, Xr in cols_B]
        def _bmat(row):   # k x q containing row `row` of the design
            B = np.zeros((k, q))
            for ii, (j, Xr) in enumerate(bblocks): B[j, boffs[ii]:boffs[ii + 1]] = Xr[row]
            return _F(B)
        def _brows(rows):  # (n*k) x q with the design of each row
            arr = np.zeros((len(rows), k, q))
            for ii, (j, Xr) in enumerate(bblocks): arr[:, j, boffs[ii]:boffs[ii + 1]] = Xr[rows]
            return _F(arr.reshape(len(rows) * k, q))
        if li == 0:
            varies = any(not np.allclose(Xr[r], Xr[r[0]]) for _, Xr in bblocks for r in rows_by_group)
            for gi in range(G):
                B_list.append(_bmat(rows_by_group[gi][0]))
                if varies: B_rows.append(_brows(rows_by_group[gi]))
        else:
            varies = any(not np.allclose(Xr, Xr[0]) for _, Xr in bblocks)
            inner_B.append(_bmat(0)); inner_sub.append(_I(nested_codes(df2, chain[:li + 1])))
            inner_B_rows.append([_brows(r) for r in rows_by_group] if varies else [])
    cspec, time, coords, cor_sub, _ = _cor_inputs(correlation, df2, chain, cor_prefix)
    vb = _var_spec(weights, df2, codes)
    spec = core.NlmeSpec(); spec.reml = method == "REML"; spec.cor = cspec; spec.var = vb.spec; spec.var_use_fitted = vb.fitted
    if pd_struct is not None:
        lst = pd_struct if isinstance(pd_struct, (list, tuple)) else ([pd_struct.get(g, "logchol") for g in chain] if isinstance(pd_struct, Mapping) else [pd_struct] * len(lv))
        spec.pd_specs = [_pd_spec(as_pd(p)) for p in lst]
    extra = _apply_control(spec, control, "nlme")
    b0 = _start_vector(nm, start, bn, offs, y if use is None else y[use], df2) if (start is not None or nm.ss) else _start_vector(nm, start, bn, offs, y, df2)
    if nm.ss and start is None and use is not None:
        nm.x_all = nm.x_all  # the starting values use only the observed rows
        init = np.asarray(core.selfstart_initial(nm.ss, _V(nm.x_all[use]), _V(y[use])))
        b0 = np.zeros(p);
        for j in range(k): b0[offs[j]] = init[j]
    auto = bool(extra.get("auto_start", True))
    args = (_V(y), _I(codes), A_list, B_list, f, jac, spec, _V(b0))
    kw = dict(time=None if time is None else _V(time), coords=coords, strata_slots=vb.strata, cov_slots=vb.cov, inner_B=inner_B,
              inner_sub=inner_sub, cor_sub=cor_sub, use=None if use is None else _I(use.astype(np.int32)),
              A_rows=A_rows, B_rows=B_rows, inner_B_rows=inner_B_rows)
    try:
        raw, b_start, D0, s0, ddmodel = core.fit_nlme(*args, auto_start=auto, **kw)
    except RuntimeError:
        if not auto: raise
        raw, b_start, D0, s0, ddmodel = core.fit_nlme(*args, auto_start=False, **kw)
    fit = NlmeFit(raw, bn, df2.index, y, "nlme", method, dict(formula=nm.formula, params=nm.params, random_spec=lv, start_beta=np.array(b_start),
                                                             cor_spec=correlation, var_spec=weights, var_gcodes=codes,
                                                             sigma_fixed=spec.sigma_fixed > 0))
    fit._ddf_model = ddmodel
    fit._reml_refit = lambda: _fit_nlme(
        model, data, params, random, start=start, correlation=correlation, weights=weights, method="REML",
        na_action=na_action, na_pattern=na_pattern, control=control, pd_struct=pd_struct, cor_prefix=cor_prefix,
        response=response, predictors=predictors, jac=jac,
    )
    fit.sigma_b = pd.DataFrame(np.array(raw.Sigma_b), index=qnames[0], columns=qnames[0])
    # random effects: [b_0 | level-1 units | ...] per group
    q0 = len(qnames[0]); rows0 = np.vstack([np.array(b)[:q0] for b in raw.b])
    perm = np.argsort(gorder)                          # rows in the order of the factor levels
    r0 = pd.DataFrame(rows0[perm], index=pd.Index(lev0[gorder[perm]], name=chain[0]), columns=qnames[0])
    if len(lv) == 1:
        fit._ranef = r0
    else:
        out = {chain[0]: r0}; pos = [q0] * G
        for li in range(1, len(lv)):
            rows, idx, ql = [], [], len(qnames[li])
            for gi in perm:
                in_g = np.flatnonzero(codes == gorder[gi]); subs = pd.unique(inner_sub[li - 1][in_g])
                for sid in subs:
                    first = in_g[np.flatnonzero(inner_sub[li - 1][in_g] == sid)[0]]
                    idx.append(tuple(df2[c].iloc[first] for c in chain[:li + 1]))
                    rows.append(np.array(raw.b[gi])[pos[gi]:pos[gi] + ql]); pos[gi] += ql
            out["/".join(chain[:li + 1])] = pd.DataFrame(np.vstack(rows), index=pd.MultiIndex.from_tuples(idx, names=chain[:li + 1]), columns=qnames[li])
        fit._ranef = out
    # prediction and degrees of freedom (containment on the linearized design)
    fit._nm, fit._data = nm, df2
    fit._param_specs = [(pspecs[j], int(offs[j + 1] - offs[j]), int(offs[j])) for j in range(len(nm.params))]
    maps = [dict(zip(r0.index, r0.to_numpy()))]
    for li in range(1, len(lv)):
        fr = fit._ranef["/".join(chain[:li + 1])]
        maps.append(dict(zip(fr.index, fr.to_numpy())))
    fit._pred_levels = [dict(chain=chain[:li + 1], map=maps[li], q=len(qnames[li]), blocks=[(j, sp, w, c0) for (j, sp, w, c0) in rblocks[li]]) for li in range(len(lv))]
    used = np.array(raw.used, bool); ordr = np.array(raw.order)[used]          # rows used, in the stacked order of X_lin
    Xlin = np.array(raw.X_lin)
    fit._fixdf = _get_fix_df(Xlin, [nested_codes(df2.iloc[ordr], chain[:k + 1]) for k in range(len(lv))])
    return fit


def _fit_gnls(model, data, params=None, start=None, correlation: Optional[CorStruct] = None, weights: Optional[VarStruct] = None,
              na_action: str = "omit", na_pattern=None, control: Optional[Mapping] = None, method: str = "ML",
              response: Optional[str] = None, predictors=None, jac: Optional[Callable] = None) -> NlmeFit:
    """Fit a generalized nonlinear least-squares model (R ``nlme::gnls``).

    Parameters
    ----------
    model : str or callable
        Two-sided formula for the response and the nonlinear model, either a
        ``selfStart`` model or a user expression.  Alternatively a plain
        Python function ``def f(x, a, b)``; in that case ``response`` must
        name the response column and the argument names are, in order, the
        exogenous variables followed by the parameters in ``params``.
    data : pandas.DataFrame or array-like
        Data containing every column referenced by the formulas.
    params : str, sequence or mapping, optional
        Parameter formulas such as ``"a + b ~ 1"``, ``["a ~ 1", "b ~ trt"]``
        or ``{"a": "1", "b": "trt"}``. When omitted, the keys of a mapping
        ``start`` are used in insertion order with formula ``"1"``.
    start : mapping or sequence, optional
        Initial parameter values; optional for ``selfStart`` models and
        required for a callable model. Its key order supplies ``params`` when
        that argument is omitted.
    correlation : CorStruct, optional
        Residual correlation structure.  Its grouping factor defines the
        independent groups.
    weights : VarStruct, optional
        Variance function describing heteroscedasticity.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied to the columns used by the model.
    na_pattern : bool, mask or callable, optional
        R ``naPattern`` equivalent (see :func:`_fit_nlme`).
    control : mapping, optional
        Engine controls such as ``optimizer``, ``max_iter``, ``gtol`` or
        ``sigma``.
    method : {"REML", "ML"}, optional
        Estimation criterion; ``"ML"`` by default.
    response : str, optional
        Response column name.  Required when ``model`` is a callable.
    predictors : str or list of str, optional
        Exogenous column(s) of a callable model; inferred from the signature
        when omitted.
    jac : callable, optional
        Analytical Jacobian for a callable model, with the same signature as
        ``model`` and returning an ``(n, k)`` array.

    Returns
    -------
    NlmeFit
        Fitted nonlinear least-squares model with ``kind == "gnls"``.
    """
    df = _as_df(data)
    params = _resolve_params(params, start)
    nm = _nl_model(model, params, df, response=response, predictors=predictors, jac=jac)
    cols = collect_columns([*nm.exprs(), correlation.form if correlation else None, *_var_forms(weights)], df)
    if na_pattern:
        cols = [c for c in cols if c not in set(names_in(nm.lhs, df.columns))]
    df2, _ = handle_na(df, cols, na_action)
    y = eval_expr(nm.lhs, df2)
    use = _na_use(na_pattern, df2, y)
    if use is not None: y = np.where(use, y, 0.0)
    cspec, time, coords, _, groups = _cor_inputs(correlation, df2, [])
    codes = nested_codes(df2, groups) if groups else np.zeros(len(df2), np.int32)
    gorder = pd.unique(codes)
    rows_by_group = [np.flatnonzero(codes == c) for c in gorder]
    f, jac = nm.make(df2, rows_by_group)
    A_list, A_rows, bn, offs, pspecs = _design_params(nm, df2, codes, gorder)
    b0 = _start_vector(nm, start, bn, offs, y if use is None else y[use], df2)
    vb = _var_spec(weights, df2, codes)
    spec = core.GnlsSpec(); spec.reml = method == "REML"; spec.cor = cspec; spec.var = vb.spec; spec.var_use_fitted = vb.fitted
    _apply_control(spec, control, "gnls")
    raw, ddmodel = core.fit_gnls(_V(y), _I(codes), A_list, f, jac, spec, _V(b0), None if time is None else _V(time), coords, vb.strata, vb.cov,
                                 None if use is None else _I(use.astype(np.int32)), A_rows)
    fit = NlmeFit(raw, bn, df2.index, y, "gnls", method, dict(formula=nm.formula, params=nm.params,
                                                              cor_spec=correlation, var_spec=weights, var_gcodes=codes,
                                                              sigma_fixed=spec.sigma_fixed > 0))
    fit._ddf_model = ddmodel
    fit._reml_refit = lambda: _fit_gnls(
        model, data, params, start=start, correlation=correlation, weights=weights, method="REML",
        na_action=na_action, na_pattern=na_pattern, control=control, response=response,
        predictors=predictors, jac=jac,
    )
    fit._ranef = None
    fit._nm, fit._data, fit._pred_levels = nm, df2, []
    fit._param_specs = [(pspecs[j], int(offs[j + 1] - offs[j]), int(offs[j])) for j in range(len(nm.params))]
    fit._fixdf = np.full(len(bn), float(len(y) - len(bn)))   # summary.gnls: N - p
    fit.sigma_r = math.sqrt(raw.sigma2 * len(y) / max(len(y) - len(bn), 1))   # sigma with N-p, as reported by R
    return fit


# ----------------------------------------------------------------------------------------------------------------------
# models (chainable API: lme(...).fit())
# ----------------------------------------------------------------------------------------------------------------------
class _ModelBase:
    """Unfitted model. ``.fit()`` runs the engine and returns the result.

    Models are constructed unfitted; result attributes (``fixef``, ``se``,
    ``loglik``, ``summary()``, ...) live on the object returned by
    :meth:`fit`, as in the rest of the package::

        res = lme("distance ~ age", data, random="~ 1 | Subject").fit()
        print(res.summary())
    """

    def fit(self):
        """Fit the model (once) and return the result object."""
        if self.__dict__.get("_result") is None:
            self.__dict__["_result"] = self._fit()
        return self.__dict__["_result"]

    def _fit(self):  # pragma: no cover - implemented by each subclass
        raise NotImplementedError

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"{type(self).__name__} is not fitted: {name!r} is only available on the result. "
            f"Call .fit() first, e.g. `result = model.fit()` and then `result.{name}`."
        )

    def __repr__(self):
        res = self.__dict__.get("_result")
        if res is not None:
            return repr(res)
        return f"{type(self).__name__}({getattr(self, 'formula', '')!r}, fitted=False)"


class LME(_ModelBase):
    """Unfitted linear mixed-effects model (R ``nlme::lme``).

    The constructor stores the specification; call :meth:`~_ModelBase.fit` to
    run the engine and obtain an :class:`LmeFit`.  For example::

        res = LME("distance ~ age", data, random="~ 1 | Subject").fit()

    See :func:`_fit_lme` for the meaning of ``random``, ``correlation``,
    ``weights``, ``pd_struct`` and ``control``.
    """

    def __init__(self, fixed: str, data, random=None, groups: Optional[str] = None, correlation: Optional[CorStruct] = None,
                 weights: Optional[VarStruct] = None, method: str = "REML", na_action: str = "omit",
                 control: Optional[Mapping] = None, pd_struct=None, cor_prefix: bool = False):
        self.formula = fixed
        self.data, self.random, self.groups = data, random, groups
        self.correlation, self.weights, self.method = correlation, weights, method
        self.na_action, self.control, self.pd_struct, self.cor_prefix = na_action, control, pd_struct, cor_prefix

    def _fit(self):
        return _fit_lme(self.formula, self.data, random=self.random, groups=self.groups, correlation=self.correlation,
                        weights=self.weights, method=self.method, na_action=self.na_action, control=self.control,
                        pd_struct=self.pd_struct, cor_prefix=self.cor_prefix)


class GLS(_ModelBase):
    """Unfitted generalized least-squares model (R ``nlme::gls``).

    Groups are defined by the correlation structure (for example
    ``corAR1(form="~ time | subject")``); without a grouping factor the model
    has a single group.  Call :meth:`~_ModelBase.fit` to obtain an
    :class:`LmeFit` with ``kind == "gls"``.
    """

    def __init__(self, model: str, data, correlation: Optional[CorStruct] = None, weights: Optional[VarStruct] = None,
                 method: str = "REML", na_action: str = "omit", control: Optional[Mapping] = None):
        self.formula = model
        self.data, self.correlation, self.weights = data, correlation, weights
        self.method, self.na_action, self.control = method, na_action, control

    def _fit(self):
        return _fit_gls(self.formula, self.data, correlation=self.correlation, weights=self.weights,
                        method=self.method, na_action=self.na_action, control=self.control)


class NLME(_ModelBase):
    """Unfitted nonlinear mixed-effects model (R ``nlme::nlme``).

    ``model`` is a self-starting formula such as
    ``"height ~ SSasymp(age, Asym, R0, lrc)"``, a user expression such as
    ``"y ~ a * exp(-b * x)"``, or a plain Python function
    ``def f(x, a, b)`` (then pass ``response=`` and, optionally,
    ``predictors=``). ``params`` gives the parameter formulas, or it can be
    omitted when ``start`` is a mapping: its keys then name the parameters in
    order with formula ``"1"``. ``random`` gives the grouped random effects.
    Call :meth:`~_ModelBase.fit` to obtain an :class:`NlmeFit`.

    See :func:`_fit_nlme` for the meaning of ``start``, ``correlation``,
    ``weights``, ``na_pattern``, ``pd_struct``, ``control`` and ``jac``.
    """

    def __init__(self, model, data, params=None, random=None, start=None, correlation: Optional[CorStruct] = None,
                 weights: Optional[VarStruct] = None, method: str = "ML", na_action: str = "omit", na_pattern=None,
                 control: Optional[Mapping] = None, pd_struct=None, cor_prefix: bool = False,
                 response: Optional[str] = None, predictors=None, jac: Optional[Callable] = None):
        if random is None:
            raise TypeError("random is required for nlme models, e.g. random='a ~ 1 | subject'")

        self.formula = model
        self.data, self.params, self.random, self.start = data, params, random, start
        self.correlation, self.weights, self.method = correlation, weights, method
        self.na_action, self.na_pattern, self.control = na_action, na_pattern, control
        self.pd_struct, self.cor_prefix = pd_struct, cor_prefix
        self.response, self.predictors, self.jac = response, predictors, jac

    def _fit(self):
        return _fit_nlme(self.formula, self.data, self.params, self.random, start=self.start,
                         correlation=self.correlation, weights=self.weights, method=self.method,
                         na_action=self.na_action, na_pattern=self.na_pattern, control=self.control,
                         pd_struct=self.pd_struct, cor_prefix=self.cor_prefix,
                         response=self.response, predictors=self.predictors, jac=self.jac)


class GNLS(_ModelBase):
    """Unfitted generalized nonlinear least-squares model (R ``nlme::gnls``).

    This is a nonlinear model with optional correlation and variance
    structures but without random effects.  ``model`` may be a formula, a
    self-starting model or a plain Python function ``def f(x, a, b)`` (pass
    ``response=`` and, optionally, ``predictors=``). ``params`` may be omitted
    when ``start`` is a mapping; its keys then name the parameters in order.
    Call :meth:`~_ModelBase.fit` to obtain an :class:`NlmeFit` with
    ``kind == "gnls"``.  See :func:`_fit_gnls` for the parameter meanings.

    Notes
    -----
    ``gnls`` is the structured counterpart of :class:`statcraft.nls.nls`.
    It adds ``correlation`` and ``weights`` but offers no parameter bounds,
    robust losses, ``fit_multistart``, HC0-HC3 covariance, cross-validation,
    prediction intervals or JAX backend.  For a single curve with any of
    those, use :class:`statcraft.nls.nls` or :class:`statcraft.nls.NLS`.
    """

    def __init__(self, model, data, params=None, start=None, correlation: Optional[CorStruct] = None,
                 weights: Optional[VarStruct] = None, na_action: str = "omit", na_pattern=None,
                 control: Optional[Mapping] = None, method: str = "ML",
                 response: Optional[str] = None, predictors=None, jac: Optional[Callable] = None):
        self.formula = model
        self.data, self.params, self.start = data, params, start
        self.correlation, self.weights, self.method = correlation, weights, method
        self.na_action, self.na_pattern, self.control = na_action, na_pattern, control
        self.response, self.predictors, self.jac = response, predictors, jac

    def _fit(self):
        return _fit_gnls(self.formula, self.data, self.params, start=self.start, correlation=self.correlation,
                         weights=self.weights, na_action=self.na_action, na_pattern=self.na_pattern,
                         control=self.control, method=self.method,
                         response=self.response, predictors=self.predictors, jac=self.jac)


#: Lowercase alias for the R/nlme API (they return a model; call .fit()).
lme = LME
gls = GLS
nlme = NLME
gnls = GNLS


# ----------------------------------------------------------------------------------------------------------------------
# lmList / nlsList
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ListFit(PlottingMixin):
    """Result of :func:`lm_list` / :func:`nls_list` (one fit per group).

    Attributes
    ----------
    coef : pandas.DataFrame
        Estimated coefficients, one row per group.
    se : pandas.DataFrame
        Standard errors with the same shape and labels as ``coef``.
    sigma : pandas.Series
        Residual standard deviation of each group fit.
    df : pandas.Series
        Degrees of freedom of each group fit.
    pooled_sigma : float
        Pooled residual standard deviation across groups.
    pooled_df : int
        Pooled degrees of freedom.
    converged : pandas.Series, optional
        Per-group convergence flags (``nls_list`` only).
    raw
        The underlying engine result object.
    """

    coef: pd.DataFrame
    se: pd.DataFrame
    sigma: pd.Series
    df: pd.Series
    pooled_sigma: float
    pooled_df: int
    converged: Optional[pd.Series] = None
    raw: Any = None
    formula: Optional[str] = None
    kind: Optional[str] = None

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this list fit.

        See :mod:`statcraft.plots` for the supported ``kind`` names; the
        default here is the per-group coefficient panel.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind or "list_coefficients", ax=ax, **kwargs)

    def summary(self, alpha: float = 0.05):
        """statsmodels-style summary of a per-group ``lmList``/``nlsList`` fit."""
        kind = self.kind or ("nls_list" if self.converged is not None else "lm_list")
        is_nl = kind in ("nls_list", "nls")
        formula = str(self.formula or "")
        lhs, _, rhs = formula.partition("~")
        lhs = lhs.strip() or "y"
        grouping = rhs.partition("|")[2].strip() or "Group"
        n_groups = len(self.coef)
        n_alias = int(self.coef.isna().to_numpy().sum())

        sigmas = np.asarray(self.sigma, dtype=float)
        sigmas = sigmas[np.isfinite(sigmas)]
        left = [
            ("Dep. Variable:", lhs),
            ("Model:", "nlsList" if is_nl else "lmList"),
            ("Method:", "NLS by group" if is_nl else "OLS by group"),
            ("Grouping:", grouping),
            ("No. Groups:", str(n_groups)),
        ]
        right = [
            ("Pooled sigma:", _fmt_value(self.pooled_sigma)),
            ("Pooled df:", str(int(self.pooled_df))),
            ("Mean sigma:", _fmt_value(np.mean(sigmas)) if sigmas.size else "NA"),
            ("Min sigma:", _fmt_value(np.min(sigmas)) if sigmas.size else "NA"),
            ("Max sigma:", _fmt_value(np.max(sigmas)) if sigmas.size else "NA"),
        ]
        if self.converged is not None:
            n_ok = int(np.sum(np.asarray(self.converged, dtype=bool)))
            right.append(("Converged groups:", f"{n_ok} / {n_groups}"))
        left_extra = [
            ("Coefficients/Group:", str(len(self.coef.columns))),
            ("Aliased coefs:", str(n_alias)),
        ]
        right_extra = [
            ("Total residual df:", str(int(np.nansum(np.asarray(self.df, dtype=float))))),
        ]
        left, right = balanced_info(left, right, left_extra, right_extra)

        index = [str(i) for i in self.coef.index]
        names = [str(c) for c in self.coef.columns]
        widths = ["right"] * len(names)
        formats = [".4f"] * len(names)
        tables = [
            TableBlock(
                heading="Coefficients by group",
                columns=names,
                rows=self.coef.to_numpy(dtype=float),
                index=index,
                align=widths,
                formats=formats,
            ),
            TableBlock(
                heading="Standard errors by group",
                columns=names,
                rows=self.se.to_numpy(dtype=float),
                index=index,
                align=widths,
                formats=formats,
            ),
        ]
        group_rows = [
            [float(s), float(d)] for s, d in zip(self.sigma.to_numpy(), self.df.to_numpy())
        ]
        group_columns = ["Sigma", "DF"]
        group_formats = [".4f", ".0f"]
        if self.converged is not None:
            group_columns.append("Converged")
            group_rows = [row + [str(bool(c))] for row, c in zip(group_rows, self.converged.to_numpy())]
            group_formats.append(".4f")
        tables.append(
            TableBlock(
                heading="Group fit summary",
                columns=group_columns,
                rows=group_rows,
                index=index,
                align=["right"] * len(group_columns),
                formats=group_formats,
            )
        )

        notes = []
        if n_alias:
            notes.append(f"{n_alias} coefficient(s) are aliased (NaN) in at least one group fit.")
        if self.converged is not None and not bool(np.all(np.asarray(self.converged, dtype=bool))):
            notes.append("At least one group fit did not converge.")
        return build_summary(
            title="Nonlinear Model List Results" if is_nl else "Linear Model List Results",
            subtitle=f"Formula: {formula}" if formula else None,
            left=left,
            right=right,
            tables=tables,
            dist="normal",
            alpha=alpha,
            notes=notes,
        )

    def __repr__(self):
        return _compact_fit_repr(self)


def lm_list(formula: str, data, na_action: str = "omit") -> ListFit:
    """Fit one linear model per group (R ``nlme::lmList``).

    Parameters
    ----------
    formula : str
        Formula with a grouping term, for example ``"y ~ x | g"``.  Nested
        grouping chains such as ``"y ~ x | a/b"`` are supported.
    data : pandas.DataFrame or array-like
        Data containing the response, the predictors and the grouping columns.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied to the columns used by the model.

    Returns
    -------
    ListFit
        Per-group coefficients, standard errors, residual standard deviations
        and degrees of freedom.  Aliased coefficients are reported as ``NaN``.

    Raises
    ------
    ValueError
        If the formula has no grouping term.
    """
    df = _as_df(data); fp = parse_form(formula)
    if not fp.groups: raise ValueError("formula must include groups: 'y ~ x | g'")
    cols = collect_columns([fp.lhs, fp.rhs] + fp.groups, df)
    df2, _ = handle_na(df, cols, na_action)
    y = eval_expr(fp.lhs, df2); X, names = design(fp.rhs, df2)
    codes, lev = factor_codes(df2[fp.groups[0]]) if len(fp.groups) == 1 else (nested_codes(df2, fp.groups), None)
    raw = core.lm_list(_V(y), _F(X), _I(codes))
    if lev is None:
        idx = pd.MultiIndex.from_frame(df2[fp.groups].drop_duplicates().sort_values(fp.groups))
    else:
        idx = pd.Index(lev[np.array(raw.labels)], name=fp.groups[0])
    return ListFit(pd.DataFrame(np.vstack(raw.coef), index=idx, columns=names), pd.DataFrame(np.vstack(raw.se), index=idx, columns=names),
                   pd.Series(raw.sigma, index=idx), pd.Series(raw.df, index=idx), raw.pooled_sigma, raw.pooled_df, None, raw,
                   formula=formula, kind="lm_list")


def nls_list(model: str, data, start, params=None, na_action: str = "omit", tol: float = 1e-8, max_iter: int = 200) -> ListFit:
    """Fit one nonlinear model per group (R ``nlme::nlsList``).

    Parameters
    ----------
    model : str
        Formula with a grouping term, for example
        ``"y ~ SSlogis(x, Asym, xmid, scal) | g"``.
    data : pandas.DataFrame or array-like
        Data containing the response, the covariates and the grouping columns.
    start : mapping or sequence
        Initial parameter values, either a mapping such as
        ``{"Asym": 1.0, "xmid": 2.0, "scal": 3.0}`` or a flat vector in the
        order of the model parameters.
    params : str, optional
        Parameter formulas for non-``selfStart`` models (for example
        ``"a + b ~ 1"``).  Inferred automatically for ``SS*`` models.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied to the columns used by the model.
    tol : float, optional
        Convergence tolerance for the per-group optimizations.
    max_iter : int, optional
        Maximum number of iterations per group.

    Returns
    -------
    ListFit
        Per-group coefficients, standard errors, residual standard deviations,
        degrees of freedom and convergence flags.

    Raises
    ------
    ValueError
        If the formula has no grouping term, or if ``params`` is missing for a
        model that is not self-starting.
    """
    df = _as_df(data); fp = parse_form(model)
    if not fp.groups: raise ValueError("model must include groups: 'y ~ f(...) | g'")
    mdl = f"{fp.lhs} ~ {fp.rhs}"
    ps = params or "+".join(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", ",".join(_split_top(re.fullmatch(r"\s*SS\w+\s*\((.*)\)\s*", fp.rhs, re.S).group(1))[1:]))) + " ~ 1" if re.match(r"\s*SS", fp.rhs) else params
    if ps is None: raise ValueError("params is required for models that are not selfStart")
    nm = _NlModel(mdl, ps, df)
    cols = collect_columns([mdl, *nm.fixed_rhs.values()] + fp.groups, df)
    df2, _ = handle_na(df, cols, na_action)
    y = eval_expr(nm.lhs, df2)
    codes, lev = factor_codes(df2[fp.groups[0]]); gorder = pd.unique(codes); G = len(gorder)
    rows_by_group = [np.flatnonzero(codes == c) for c in gorder]
    f, jac = nm.make(df2, rows_by_group)
    k = len(nm.params); A_list = [_F(np.eye(k))] * G
    b0 = np.array([start[p] for p in nm.params], float) if isinstance(start, Mapping) else np.asarray(start, float)
    raw = core.nls_list(_V(y), _I(codes), A_list, f, jac, _V(b0), None, tol, max_iter)
    idx = pd.Index(lev[np.array(raw.labels)], name=fp.groups[0])
    return ListFit(pd.DataFrame(np.vstack(raw.coef), index=idx, columns=nm.params), pd.DataFrame(np.vstack(raw.se), index=idx, columns=nm.params),
                   pd.Series(raw.sigma, index=idx), pd.Series(raw.df, index=idx), raw.pooled_sigma, raw.pooled_df,
                   pd.Series([bool(c) for c in raw.converged], index=idx), raw,
                   formula=model, kind="nls_list")


from typing import Any, Union
import numpy as np
import pandas as pd


def corr_fixef(model: Any, values: bool = False) -> Union[pd.DataFrame, np.ndarray]:
    """Compute the correlation matrix of fixed effects for a fitted or unfitted model.

    Parameters
    ----------
    model : Lme or LmeFit (or Nlme / NlmeFit)
        A fitted or unfitted model object.
    values : bool, optional
        If True, return the correlation matrix as a numpy array.
        If False, return a pandas DataFrame. Default is False.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The correlation matrix of fixed effects.
    """
    fit = model if hasattr(model, "cor_fixef") else model.fit()

    corr_matrix = fit.cor_fixef()

    if values:
        return corr_matrix.to_numpy() if isinstance(corr_matrix, pd.DataFrame) else np.asarray(corr_matrix)

    return corr_matrix


def fix_df(
    model_or_formula: Any,
    data: Optional[pd.DataFrame] = None,
    random: Optional[Any] = None,
    groups: Optional[str] = None,
    na_action: str = "omit",
) -> pd.Series:
    """Compute the denominator degrees of freedom for fixed effects.

    Accepts either a model object (fitted or unfitted: LME, GLS, NLME, GNLS)
    or a fixed-effects formula along with its data and grouping structure.
    Uses the containment method (Pinheiro & Bates) when grouping levels are
    present, and ``N - p`` otherwise.

    Parameters
    ----------
    model_or_formula : _ModelBase, _FitBase, or str
        A fitted or unfitted model object, or a fixed-effects formula
        (for example ``"y ~ x + trt"``).
    data : pandas.DataFrame, optional
        Data containing the variables referenced in the formula. Required
        when ``model_or_formula`` is a string.
    random : str, mapping, or sequence, optional
        Random-effects specification (for example ``"~ 1 | subject"`` or
        ``"~ 1 | hospital/patient"``) used when passing a formula.
    groups : str, optional
        Grouping column name used when ``random`` does not include ``|``.
    na_action : {"omit", "fail"}, optional
        Missing-value policy applied when passing a formula and data.
        Defaults to ``"omit"``.

    Returns
    -------
    pandas.Series
        Degrees of freedom indexed by fixed-effect coefficient name.

    Raises
    ------
    ValueError
        If ``model_or_formula`` is a string and ``data`` is not provided, or
        if the formula or grouping specification is invalid.

    Examples
    --------
    Using an unfitted or fitted model:

    >>> mod = lme("distance ~ age + Sex", data=df, random="~ 1 | Subject")
    >>> fix_df(mod)

    Using a formula and DataFrame directly without fitting:

    >>> fix_df("y ~ trt + time", data=df, random="~ 1 | subject")
    >>> fix_df("y ~ trt + time", data=df, random="~ 1 | hospital/patient")
    """

    if isinstance(model_or_formula, _ModelBase):
        fit = model_or_formula.fit()
        return pd.Series(fit._fixdf, index=fit.names, name="DF")

    if isinstance(model_or_formula, _FitBase):
        return pd.Series(model_or_formula._fixdf, index=model_or_formula.names, name="DF")

    if not isinstance(model_or_formula, str) or data is None:
        raise ValueError("You must provide a model object or a formula along with 'data'.")

    df = _as_df(data)
    fp = parse_form(model_or_formula)

    if random is None and groups is None:
        cols = collect_columns([fp.lhs, fp.rhs], df)
        df2, _ = handle_na(df, cols, na_action)
        X, names, _ = design_spec(fp.rhs, df2)
        df_vals = np.full(X.shape[1], float(len(df2) - X.shape[1]))
        return pd.Series(df_vals, index=names, name="DF")

    levels = _parse_random(random, groups, fp.rhs)
    chain = [g for g, _ in levels]
    cols = collect_columns([fp.lhs, fp.rhs] + chain, df)
    df2, _ = handle_na(df, cols, na_action)

    X, names, _ = design_spec(fp.rhs, df2)
    gcodes = [nested_codes(df2, chain[:k + 1]) for k in range(len(chain))]
    df_vals = _get_fix_df(X, gcodes)

    return pd.Series(df_vals, index=names, name="DF")
