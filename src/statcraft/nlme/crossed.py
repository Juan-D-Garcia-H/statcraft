"""``lmer``: linear mixed-effects models with crossed / nested random effects.

This module implements the lme4-style interface on top of the sparse C++ engine.
The accepted formula syntax is the one used by ``lme4::lmer``:

* ``(1 | g)`` -- random intercept per level of ``g``;
* ``(1 + x | g)`` -- correlated random intercept and slope;
* ``(x || g)`` -- uncorrelated (diagonal covariance) random effects;
* ``(1 | a/b)`` -- nested effects, expanded to ``(1 | a) + (1 | a:b)``;
* ``(1 | a:b)`` -- interaction grouping.

Construct a model with :func:`lmer` (or :class:`LMER`) and call ``.fit()``,
as with the rest of the project::

    fit = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy).fit()

The engine uses sparse algebra with a Cholesky factorization of
``A = Z~'Z~ + I``, so models with tens of thousands of random effects are
feasible.  Results are exposed through :class:`LmerFit`, which mirrors the
lme4 accessors (``fixef``, ``ranef``, ``fitted``, ``predict``, ``tTable``,
``summary``) and adds the lmerTest Satterthwaite degrees of freedom.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import _nlmefit as core
from ..plots.methods import PlottingMixin
from ..summary import build_summary
from .formula import (collect_columns, design_new, design_spec, eval_expr, handle_na, nested_codes, _IDENT)
from .models import _ModelBase, _compact_fit_repr, _fmt_value
from ._summary import (
    balanced_info,
    crossed_random_tables,
    fixed_block,
    grouping_counts,
    grouping_text,
    r2_icc,
    residual_summary_table,
)
from .structs import PdStruct, as_pd


def _split_top(s: str, sep: str) -> list[str]:
    """Split ``s`` on ``sep`` at parenthesis depth zero.

    Parentheses are tracked so separators inside ``(...)`` (for example the
    ``+`` in ``(1 + x | g)``) do not split the string.

    Parameters
    ----------
    s:
        String to split.
    sep:
        Single-character separator.

    Returns
    -------
    list[str]
        Non-empty, stripped pieces.
    """
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return [t.strip() for t in out if t.strip()]


def parse_lmer(formula: str):
    """Parse an lme4-style formula.

    Parameters
    ----------
    formula:
        Formula such as ``'y ~ x + (1 + x | g) + (1 | a/b) + (x || h)'``.

    Returns
    -------
    lhs : str
        Left-hand side (the response).
    fixed_rhs : str
        Fixed-effects right-hand side.
    terms : list[tuple[str, list[str], bool]]
        Random-effect terms as ``(expression, group chain, uncorrelated)``.
        ``a/b`` is expanded into the chains ``[a]`` and ``[a, b]`` and ``||``
        marks an uncorrelated (diagonal) term.

    Raises
    ------
    ValueError
        If the formula has no ``~``.
    """
    if "~" not in formula:
        raise ValueError("the formula must contain '~'")
    lhs, rhs = formula.split("~", 1)
    fixed, terms = [], []
    for tok in _split_top(rhs, "+"):
        if tok.startswith("(") and tok.endswith(")") and "|" in tok:
            inner = tok[1:-1]
            unc = "||" in inner
            expr, grp = inner.split("||" if unc else "|", 1)
            expr = expr.strip() or "1"
            grp = grp.strip()
            if "/" in grp:                                  # a/b -> (a) + (a:b)
                parts = [p.strip() for p in grp.split("/")]
                chains = [parts[:k + 1] for k in range(len(parts))]
            else:
                chains = [[p.strip() for p in grp.split(":")]]
            for ch in chains:
                terms.append((expr, ch, unc))
        else:
            fixed.append(tok)
    return lhs.strip(), (" + ".join(fixed) if fixed else "1"), terms


class LmerFit(PlottingMixin):
    """Result of :meth:`LMER.fit` (``lmer(...).fit()``).

    ``fixef``/``se``/``vcov`` cover only the estimable columns (as in lme4);
    ``aliased`` lists the dropped coefficients.  Additional attributes include
    ``sigma``, ``loglik``, ``deviance``, ``nobs``, ``theta``, ``npar``,
    ``singular``, ``converged``, ``warning``, ``varcorr`` and ``ranef``.
    """

    def __init__(self, raw, model, names, keep, index, y, formula, method, terms_info, fixed_spec, data_cols):
        """Wrap an engine result.

        Parameters
        ----------
        raw:
            Object returned by the C++ ``CrossedModel.fit()``.
        model:
            The fitted engine model (used for profiling in
            :meth:`satterthwaite_df`).
        names:
            All candidate coefficient names.
        keep:
            Indices of the estimable coefficients.
        index:
            Row index of the data used in the fit.
        y:
            Response vector.
        formula:
            Original formula string.
        method:
            ``"REML"`` or ``"ML"``.
        terms_info:
            Per-term information built by :func:`build_re_terms`.
        fixed_spec:
            Formulaic specification of the fixed-effects design.
        data_cols:
            Columns used by the fit.
        """
        self.raw, self._model, self.index, self._y, self.formula, self.method = raw, model, index, y, formula, method
        self.terms_info, self._fixed_spec = terms_info, fixed_spec
        self.aliased = [names[j] for j in range(len(names)) if j not in keep]
        kn = [names[j] for j in keep]
        self._names, self._keep = kn, list(keep)
        self.fixef = pd.Series(np.array(raw.beta)[keep], index=kn, name="Estimate")
        self.se = pd.Series(np.array(raw.se)[keep], index=kn, name="Std. Error")
        self.vcov = pd.DataFrame(np.array(raw.vcov_beta)[np.ix_(keep, keep)], index=kn, columns=kn)
        self.sigma = math.sqrt(raw.sigma2)
        self.loglik = raw.loglik
        self.deviance = raw.deviance
        self.nobs = len(y)
        self.theta = np.array(raw.theta)
        self.npar = int(raw.rank) + len(raw.theta) + 1
        self.singular = bool(raw.singular)
        self.converged = bool(raw.opt.converged)
        self.warning = raw.warning
        self._build_varcorr()
        self._build_ranef()

    # ---- random-effects summary ------------------------------------------------------------------------------
    def _build_varcorr(self):
        """Build the ``varcorr`` table of variances and correlations.

        One row is produced per random-effect variance (``std_dev`` filled) and
        per pairwise correlation (``corr`` filled), plus a ``Residual`` row.
        Columns: ``group``, ``name``, ``variance``, ``std_dev`` and ``corr``.
        """
        rows = []
        for t, info in enumerate(self.terms_info):
            S = np.array(self.raw.Sigma_terms[t]); sd = np.sqrt(np.diag(S))
            for c, cn in enumerate(info["cols"]):
                rows.append((info["name"], cn, S[c, c], sd[c], np.nan))
            for c in range(1, len(info["cols"])):
                for d in range(c):
                    cor = S[c, d] / (sd[c] * sd[d]) if sd[c] > 0 and sd[d] > 0 else np.nan
                    rows.append((info["name"], f"corr({info['cols'][d]},{info['cols'][c]})", S[c, d], np.nan, cor))
        rows.append(("Residual", "", self.raw.sigma2, self.sigma, np.nan))
        self.varcorr = pd.DataFrame(rows, columns=["group", "name", "variance", "std_dev", "corr"])

    def _build_ranef(self):
        """Build ``ranef``: a dict ``group -> DataFrame`` of BLUPs.

        Terms sharing a grouping factor are concatenated column-wise (as in
        lme4's ``ranef`` list element).
        """
        by: dict[str, list[pd.DataFrame]] = {}
        for t, info in enumerate(self.terms_info):
            b = np.array(self.raw.blup[t])
            df = pd.DataFrame(b, index=info["levels"], columns=info["cols"])
            by.setdefault(info["name"], []).append(df)
        self.ranef = {g: (l[0] if len(l) == 1 else pd.concat(l, axis=1)) for g, l in by.items()}

    # ---- fitted values and predictions ----------------------------------------------------------------------
    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this fitted model.

        See :mod:`statcraft.plots` for the supported ``kind`` names.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def fitted(self, re_form: str | bool | None = "all") -> pd.Series:
        """Fitted values.

        Parameters
        ----------
        re_form:
            ``"all"``/``True`` (default) includes every random effect;
            ``False``/``None`` returns the population level (fixed effects only).

        Returns
        -------
        pandas.Series
        """
        v = self.raw.fitted_cond if (re_form == "all" or re_form is True) else self.raw.fitted_pop
        return pd.Series(np.array(v), index=self.index)

    def residuals(self, re_form: str | bool | None = "all") -> pd.Series:
        """Residuals ``y - fitted(re_form)``.

        Parameters
        ----------
        re_form:
            Passed to :meth:`fitted`; the default uses all random effects.

        Returns
        -------
        pandas.Series
        """
        return pd.Series(self._y - self.fitted(re_form).to_numpy(), index=self.index)

    def predict(self, newdata: Optional[pd.DataFrame] = None, re_form: str | bool | Sequence[str] | None = "all",
                allow_new_levels: bool = False) -> pd.Series:
        """Like predict.merMod: re_form='all' (default) uses every random effect, False/None uses only
        the fixed effects (re.form=NA), and a list of group names uses only those groups.
        allow_new_levels=True assigns effect 0 to levels not seen in the fit."""
        if newdata is None:
            return self.fitted("all" if re_form not in (False, None) else False)
        nd = pd.DataFrame(newdata).reset_index(drop=True)
        X = design_new(self._fixed_spec, nd)[:, self._keep]
        pred = np.nan_to_num(X, nan=np.nan) @ self.fixef.to_numpy()
        use = None if re_form in ("all", True) else ([] if re_form in (False, None) else list(re_form))
        for t, info in enumerate(self.terms_info):
            if use is not None and info["name"] not in use:
                continue
            Z = design_new(info["spec"], nd)
            key = list(zip(*[nd[c] for c in info["vars"]])) if len(info["vars"]) > 1 else list(nd[info["vars"][0]])
            b = np.array(self.raw.blup[t]); idx = []
            for k in key:
                pos = info["level_map"].get(k, -1)
                idx.append(pos)
            idx = np.array(idx)
            if (idx < 0).any() and not allow_new_levels:
                bad = [key[i] for i in np.flatnonzero(idx < 0)][:3]
                raise ValueError(f"newdata has levels of '{info['name']}' not seen in the fit {bad}; use allow_new_levels=True")
            bb = np.where((idx >= 0)[:, None], b[np.maximum(idx, 0)], 0.0)
            pred = pred + (Z * bb).sum(axis=1)
        return pd.Series(pred, index=nd.index if newdata is not None else self.index)

    # ---- inference -------------------------------------------------------------------------------------------
    def aic(self) -> float:
        """Akaike information criterion, ``-2 logLik + 2 npar``."""
        return -2.0 * self.loglik + 2.0 * self.npar

    def bic(self) -> float:
        """Bayesian information criterion, ``-2 logLik + log(n) npar``."""
        return -2.0 * self.loglik + math.log(self.nobs) * self.npar

    def _satterthwaite_state(self):
        """Build (and cache) the profiled covariance needed by Satterthwaite df."""
        if getattr(self, "_satt_state", None) is not None:
            return self._satt_state
        model, reml = self._model, bool(self.raw.reml)
        th = np.array(self.raw.theta)
        vp0 = np.append(th, self.sigma)
        nth = len(th)

        def dev(vp):
            P = model.profile(vp[:nth])
            if not P.ok:
                return np.inf
            s2 = vp[nth] ** 2
            dof = P.N - P.p if reml else P.N
            return dof * math.log(s2) + P.logdetA + (P.logdetS if reml else 0.0) + P.pwrss / s2

        def cov_beta(vp):
            P = model.profile(vp[:nth])
            return vp[nth] ** 2 * np.asarray(P.Sinv, dtype=float)

        H = _hessian(dev, vp0)
        try:
            ev = np.linalg.eigvalsh(H)
            if ev.min() <= 1e-8 * max(1.0, ev.max()):
                raise np.linalg.LinAlgError
            cov_theta = 2.0 * np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov_theta = np.full((vp0.size, vp0.size), np.nan)
        self._satt_state = (vp0, cov_theta, cov_beta)
        return self._satt_state

    def satterthwaite_contrast_df(self, L_list) -> np.ndarray:
        """Satterthwaite denominator df for arbitrary fixed-effect contrasts."""
        from ._ddf import require_reml, satterthwaite_df as _satt_df

        fit = require_reml(self, "Satterthwaite degrees of freedom")
        state_fit = fit if isinstance(fit, LmerFit) else self
        vp0, cov_theta, cov_beta = state_fit._satterthwaite_state()
        L_list = [np.atleast_2d(np.asarray(L, dtype=float)) for L in L_list]
        return _satt_df(vp0, cov_theta, cov_beta, L_list)

    def satterthwaite_df(self) -> pd.Series:
        """Satterthwaite degrees of freedom for each coefficient (lmerTest method).

        Uses the profiled covariance model and the REML Hessian in
        ``(theta, sigma)``; the result is cached.
        """
        if getattr(self, "_satt", None) is not None:
            return self._satt
        df = self.satterthwaite_contrast_df(np.eye(len(self._names)))
        self._satt = pd.Series(df, index=self._names, name="df")
        return self._satt

    def tTable(self, df: str = "satterthwaite", *, ddf: str | None = None) -> pd.DataFrame:
        """Estimate, Std. Error, df, t value and Pr(>|t|).

        ``df='satterthwaite'`` (default, lmerTest), ``'residual'`` (N - rank) or
        ``'none'`` omits df and p.  ``ddf`` is an alias for ``df``.
        """
        from scipy import stats

        if ddf is not None:
            df = ddf
        method = str(df).strip().lower().replace("-", "_").replace(" ", "_")
        if method in ("kr", "kenward", "kenward_roger"):
            raise NotImplementedError("Kenward-Roger df are not implemented yet")
        t = pd.DataFrame({"Estimate": self.fixef, "Std. Error": self.se})
        t["t value"] = t["Estimate"] / t["Std. Error"]
        if method == "none":
            return t
        if method in ("satterthwaite", "satt"):
            d = self.satterthwaite_df()
        elif method in ("residual", "containment"):
            d = pd.Series(self.nobs - self.raw.rank, index=self._names, dtype=float)
        else:
            raise ValueError("df must be 'satterthwaite', 'residual' or 'none'")
        t.insert(2, "df", d)
        t["Pr(>|t|)"] = 2.0 * stats.t.sf(np.abs(t["t value"]), d)
        return t

    def anova(self, other, ddf: str = "satterthwaite"):
        """Wald F test comparing nested fixed effects (Satterthwaite denominator df)."""
        from ._ddf import anova_f_test, require_reml

        big, small = (self, other) if len(self.fixef) >= len(other.fixef) else (other, self)
        big = require_reml(big, "Satterthwaite ANOVA")
        small = require_reml(small, "Satterthwaite ANOVA")
        method = str(ddf).strip().lower().replace("-", "_").replace(" ", "_")
        if method in ("kr", "kenward", "kenward_roger"):
            raise NotImplementedError("Kenward-Roger df are not implemented yet")
        if method not in ("satterthwaite", "residual"):
            raise ValueError("anova: ddf must be 'satterthwaite' or 'residual'")
        res = anova_f_test(small, big, method=method)
        q = int(res["num_df"])
        return pd.DataFrame(
            {
                "Df": [len(small.fixef), len(big.fixef)],
                "Df diff": [np.nan, q],
                "F": [np.nan, res["statistic"]],
                "Den Df": [np.nan, res["den_df"]],
                "Pr(>F)": [np.nan, res["pvalue"]],
            },
            index=["small", "big"],
        )

    def fit(self):
        """Return the already-fitted result itself (chainable API: lmer(...).fit())."""
        return self

    def summary(self, alpha: float = 0.05):
        """statsmodels-style summary (statcraft.summary) of the lmer fit."""
        lhs = str(self.formula).split("~", 1)[0].strip() or "y"
        counts = grouping_counts(self.terms_info)
        groups = grouping_text(self.terms_info)
        left = [
            ("Dep. Variable:", lhs),
            ("Model:", "LME (lme4)"),
            ("Method:", self.method),
            ("No. Observations:", str(self.nobs)),
            ("No. Groups:", str(sum(counts.values()))),
        ]
        right = [
            ("Log-Likelihood:", f"{self.loglik:.6f}"),
            ("AIC:", f"{self.aic():.4f}"),
            ("BIC:", f"{self.bic():.4f}"),
            ("Residual SE:", _fmt_value(self.sigma)),
        ]
        stats = r2_icc(self)
        if stats and "r2c" in stats:
            right.append(("R2 Marginal / Cond.:", f"{stats['r2m']:.3f} / {stats['r2c']:.3f}"))
            icc = stats.get("icc")
            if icc is not None and np.isfinite(icc):
                right.append(("ICC:", f"{icc:.3f}"))
        right.append(("Converged:", str(self.converged)))
        left_extra = [
            ("Grouping:", groups or "none"),
            ("Fixed effects:", str(len(self.fixef))),
            ("Variance params:", str(len(self.theta))),
            ("Free parameters:", str(self.npar)),
            ("Scale (variance):", _fmt_value(self.raw.sigma2)),
        ]
        right_extra = [("Singular fit:", str(self.singular))]
        left, right = balanced_info(left, right, left_extra, right_extra)

        df = max(float(self.nobs - int(self.raw.rank)), 1.0)
        blocks = [fixed_block(self.fixef, self.se, "t", df)]
        tables = crossed_random_tables(self, residual=True)
        resid_table = residual_summary_table(
            self, heading="Standardized conditional residuals", standardized=True
        )
        if resid_table is not None:
            tables.append(resid_table)
        notes = [self.warning] if self.warning else []
        if self.singular:
            notes.append("The random-effects covariance is singular (some variance components are near zero).")
        return build_summary(
            title="Linear Mixed-Effects Model Results",
            subtitle=f"Formula: {self.formula}",
            left=left,
            right=right,
            blocks=blocks,
            tables=tables,
            dist="t",
            df=df,
            alpha=alpha,
            notes=notes,
        )

    def __repr__(self):
        return _compact_fit_repr(self)


def _richardson_step(x: float, d: float = 2e-3) -> float:
    """Initial step of the Richardson extrapolation: relative with an absolute floor
    (log-Cholesky parameters can be near 0, where a purely relative step leaves the
    Hessian dominated by round-off noise)."""
    return d * max(abs(x), 1.0)


def _hessian(f, x: np.ndarray, r: int = 4) -> np.ndarray:
    """Numeric Hessian with Richardson extrapolation (in the style of numDeriv::hessian)."""
    n = len(x); H = np.zeros((n, n)); f0 = f(x)
    h0 = np.array([_richardson_step(v) for v in x])
    def d2(i, j, h):
        if i == j:
            e = np.zeros(n); e[i] = h
            return (f(x + e) - 2 * f0 + f(x - e)) / h ** 2
        ei = np.zeros(n); ei[i] = h; ej = np.zeros(n); ej[j] = h
        return (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    for i in range(n):
        for j in range(i, n):
            a = [d2(i, j, h0[i] / 2.0 ** k if i == j else min(h0[i], h0[j]) / 2.0 ** k) for k in range(r)]
            for m in range(1, r):                      # Richardson (quadratic in h: factor 4^m)
                a = [(4 ** m * a[k + 1] - a[k]) / (4 ** m - 1) for k in range(len(a) - 1)]
            H[i, j] = H[j, i] = a[0]
    return H


def _jacobian(f, x: np.ndarray, r: int = 4) -> np.ndarray:
    """Numeric Jacobian with Richardson extrapolation (in the style of numDeriv).

    Parameters
    ----------
    f:
        Vector-valued function of ``x``.
    x:
        Point at which the Jacobian is evaluated.
    r:
        Number of Richardson refinement steps.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(f(x)), len(x))``.
    """
    n = len(x); f0 = np.atleast_1d(f(x)); J = np.zeros((len(f0), n))
    h0 = np.array([_richardson_step(v) for v in x])
    for j in range(n):
        a = []
        for k in range(r):
            h = h0[j] / 2.0 ** k; e = np.zeros(n); e[j] = h
            a.append((np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * h))
        for m in range(1, r):
            a = [(4 ** m * a[k + 1] - a[k]) / (4 ** m - 1) for k in range(len(a) - 1)]
        J[:, j] = a[0]
    return J


def build_re_terms(df2: pd.DataFrame, terms, pd_struct=None):
    """Turn lme4 random-effect terms into engine terms and prediction metadata.

    Parameters
    ----------
    df2:
        Complete data (after NA handling).
    terms:
        Output of :func:`parse_lmer`: ``(expr, chain, uncorrelated)`` items.
    pd_struct:
        Optional ``PdStruct`` (or name) for the covariance parametrization;
        uncorrelated terms always use ``pdDiag``.

    Returns
    -------
    rt : list
        Engine ``ReTerm`` objects.
    terms_info : list[dict]
        Per-term information (name, group variables, design spec, levels and a
        level-to-row map) used by ``ranef`` and ``predict``.
    """
    from .models import _pd_spec
    rt, terms_info = [], []
    for expr, chain, unc in terms:
        Z, cn, zspec = design_spec(expr, df2)
        codes = nested_codes(df2, chain)
        uniq, inv = np.unique(codes, return_inverse=True)
        first = np.array([np.flatnonzero(codes == u)[0] for u in uniq])          # levels (original values) in code order
        keys = (list(zip(*[df2[c].iloc[first] for c in chain])) if len(chain) > 1 else list(df2[chain[0]].iloc[first]))
        name = ":".join(chain)
        pdk = as_pd("diag") if unc else (as_pd(pd_struct) if pd_struct is not None else as_pd("logchol"))
        rt.append(core.make_re_term(name, np.ascontiguousarray(inv, np.int32), len(uniq), np.asfortranarray(Z), _pd_spec(pdk)))
        terms_info.append(dict(name=name, vars=list(chain), cols=cn, spec=zspec, levels=pd.Index(keys if len(chain) == 1 else pd.MultiIndex.from_tuples(keys)),
                               level_map={k: i for i, k in enumerate(keys)}))
    return rt, terms_info


def _fit_lmer(formula: str, data, REML: bool = True, weights=None, offset=None, na_action: str = "omit",
              control: Optional[Mapping] = None, pd_struct=None) -> LmerFit:
    """Run the crossed-effects engine and return an :class:`LmerFit`.

    Construct models with :class:`LMER` (alias :func:`lmer`) and call
    ``.fit()``; this helper is the internal implementation used by that step.

    Examples
    --------
    ::

        lmer("Reaction ~ Days + (Days | Subject)", sleepstudy).fit()
        lmer("diameter ~ 1 + (1 | plate) + (1 | sample)", penicillin).fit()         # crossed
        lmer("strength ~ 1 + (1 | batch/cask)", pastes).fit()                         # nested
        lmer("y ~ x + (x || g)", d).fit()                                              # diagonal covariance

    Parameters
    ----------
    formula:
        lme4-style formula.
    data:
        Data frame or anything accepted by :class:`pandas.DataFrame`.
    REML:
        Use restricted maximum likelihood (default ``True``); otherwise ML.
    weights:
        Column name or vector of prior weights.
    offset:
        Column name or vector added to the linear predictor.
    na_action:
        ``"omit"`` (default) or ``"fail"``.
    control:
        Optional dict with ``optimizer``, ``max_iter``, ``gtol``, ``ftol``,
        ``fd_step`` or ``theta_bound``.
    pd_struct:
        Optional covariance parametrization (see :mod:`statcraft.nlme.structs`).

    Returns
    -------
    LmerFit

    Raises
    ------
    ValueError
        If the formula has no random-effect terms or an unknown control option
        is supplied.

    Notes
    -----
    Sparse algebra with a Cholesky factorization of ``A = Z~'Z~ + I`` (symbolic
    analysis only once) scales to tens of thousands of effects.
    """
    df = pd.DataFrame(data)
    lhs, fixed_rhs, terms = parse_lmer(formula)
    if not terms:
        raise ValueError("the formula has no random-effect terms '(... | g)'")
    wcol = weights if isinstance(weights, str) else None
    ocol = offset if isinstance(offset, str) else None
    exprs = [lhs, fixed_rhs] + [t[0] for t in terms] + [c for t in terms for c in t[1]] + [wcol, ocol]
    cols = collect_columns(exprs, df)
    if weights is not None and wcol is None:
        df = df.assign(__w=np.asarray(weights, float)); cols.append("__w"); wcol = "__w"
    if offset is not None and ocol is None:
        df = df.assign(__o=np.asarray(offset, float)); cols.append("__o"); ocol = "__o"
    df2, keep_rows = handle_na(df, cols, na_action)
    y = eval_expr(lhs, df2)
    X, names, fspec = design_spec(fixed_rhs, df2)
    rt, terms_info = build_re_terms(df2, terms, pd_struct)
    spec = core.CrossedSpec(); spec.reml = bool(REML)
    if control:
        for k, v in dict(control).items():
            if k == "optimizer":
                spec.optimizer = getattr(core.Optimizer, {"internal": "Internal", "trust": "TrustRegion", "lbfgsb": "LBFGSB", "scipy": "ScipyBFGS"}[str(v).lower()])
            elif k in ("max_iter", "gtol", "ftol", "fd_step"):
                setattr(spec.opt, k, v)
            elif k == "theta_bound":
                spec.theta_bound = float(v)
            else:
                raise ValueError(f"unknown control option for lmer: {k!r}")
    w = np.ascontiguousarray(df2[wcol].to_numpy(float)) if wcol else np.zeros(0)
    o = np.ascontiguousarray(df2[ocol].to_numpy(float)) if ocol else np.zeros(0)
    model = core.CrossedModel(np.ascontiguousarray(y), np.asfortranarray(X), rt, spec, w, o)
    raw = model.fit()
    keepc = [j for j in range(X.shape[1]) if j not in set(model.aliased)]
    fit = LmerFit(raw, model, names, keepc, df2.index, y, formula, "REML" if REML else "ML", terms_info, fspec, cols)
    return fit


class LMER(_ModelBase):
    """Unfitted linear mixed model with crossed/nested random effects.

    The constructor stores the lme4-style specification; call :meth:`fit` to
    obtain an :class:`LmerFit`, as with ``LME``/``GLS``/``NLME``/``GNLS``::

        res = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy).fit()
        print(res.summary())

    ``LMER`` and its lowercase alias :func:`lmer` never fit at construction
    time.  See :func:`_fit_lmer` for the meaning of the arguments.
    """

    def __init__(self, formula: str, data, REML: bool = True, weights=None, offset=None,
                 na_action: str = "omit", control: Optional[Mapping] = None, pd_struct=None):
        self.formula = formula
        self.data = data
        self.REML = bool(REML)
        self.weights, self.offset = weights, offset
        self.na_action, self.control, self.pd_struct = na_action, control, pd_struct

    def _fit(self):
        fit = _fit_lmer(self.formula, self.data, REML=self.REML, weights=self.weights,
                        offset=self.offset, na_action=self.na_action, control=self.control,
                        pd_struct=self.pd_struct)
        fit._reml_refit = lambda: LMER(self.formula, self.data, REML=True, weights=self.weights,
                                       offset=self.offset, na_action=self.na_action,
                                       control=self.control, pd_struct=self.pd_struct).fit()
        return fit


#: Lowercase alias for the R/lme4 API (returns an unfitted model; call .fit()).
lmer = LMER
