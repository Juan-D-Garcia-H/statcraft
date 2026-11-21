"""``glmer``: generalized linear mixed-effects models with lme4 syntax.

Supported families are binomial, Poisson, negative binomial, Gaussian, Gamma
and inverse Gaussian; the link is selected per family (``logit``/``probit``/
``cloglog``/``cauchit`` for binomial, ``log``/``sqrt``/``identity`` for the
count families, ``identity``/``log``/``inverse``/``sqrt`` for Gaussian,
``inverse``/``log``/``identity``/``sqrt`` for Gamma and ``invmu2``/``log``/
``identity``/``inverse`` for inverse Gaussian).  Continuous families estimate
(or fix via ``dispersion=``) a dispersion parameter.  ``family`` accepts the
same family objects as :mod:`statcraft.glm`, for example
``glmer(..., family=Poisson("log")).fit()`` or
``glmer(..., family=Gamma("log")).fit()``.  Family strings such as
``family="poisson"`` are rejected: pass a family object or class instead.
Models are constructed unfitted, so ``.fit()`` is required.  The likelihood is
approximated by the Laplace method (``nAGQ=1``, the default) or by adaptive
Gauss-Hermite quadrature (``nAGQ>1``, currently restricted to a single scalar
random effect).

The module also provides the NumPy implementations of the inverse link, the
derivative ``dmu/deta``, the variance function and the unit deviance that are
used to build residuals and predictions, consistent with the C++ engine.
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
from .crossed import build_re_terms, parse_lmer
from .formula import collect_columns, design_new, design_spec, eval_expr, handle_na
from .models import _ModelBase, _compact_fit_repr, _fmt_value
from ._summary import (
    balanced_info,
    crossed_random_tables,
    fixed_block,
    glmer_fit_table,
    grouping_counts,
    grouping_text,
    residual_summary_table,
)

_LINKS = {"logit": "Logit", "probit": "Probit", "cloglog": "Cloglog", "cauchit": "Cauchit", "log": "Log", "sqrt": "Sqrt", "identity": "Identity", "inverse": "Inverse", "invmu2": "Invmu2"}
_DEFAULT_LINK = {"binomial": "logit", "poisson": "log", "negbin": "log", "gaussian": "identity", "gamma": "inverse", "invgaussian": "invmu2"}
_ALLOWED = {
    "binomial": {"logit", "probit", "cloglog", "cauchit"},
    "poisson": {"log", "sqrt", "identity"},
    "negbin": {"log", "sqrt", "identity"},
    "gaussian": {"identity", "log", "inverse", "sqrt"},
    "gamma": {"inverse", "log", "identity", "sqrt"},
    "invgaussian": {"invmu2", "log", "identity", "inverse"},
}
_FAMILY_CLASSES = {"Binomial": "binomial", "Poisson": "poisson", "NegativeBinomial": "negbin", "Gaussian": "gaussian", "Gamma": "gamma", "InverseGaussian": "invgaussian"}
_LINK_ALIASES = {"logit_log": "logit"}  # accepted by glm.Binomial as an alias of logit
_CONTINUOUS = frozenset({"gaussian", "gamma", "invgaussian"})


_FAMILY_HINT = (
    "family must be a GLM family object or class such as Binomial('logit'), "
    "Poisson('log'), NegativeBinomial(theta=2.0), Gaussian(), Gamma('log') or "
    "InverseGaussian(); family strings such as 'binomial' are not accepted"
)


def _normalize_family_link(family, link: Optional[str], theta: Optional[float], dispersion: Optional[float] = None):
    """Normalize a GLM-style family object to ``(family, link, theta, dispersion)``.

    ``glmer`` accepts the same family objects as :mod:`statcraft.glm`, e.g.
    ``Poisson("log")``, ``Binomial("probit")``,
    ``NegativeBinomial(theta=2.0, link="log")``, ``Gamma("log")`` and
    ``InverseGaussian("invmu2")``, or the corresponding classes.  Passing a
    class instantiates it with its defaults.  Strings are deliberately
    rejected.  The explicit ``link=`` argument overrides the link carried by
    the family object.  ``theta`` belongs to the negative binomial and
    ``dispersion`` to the continuous families (Gaussian, Gamma, inverse
    Gaussian).
    """
    if family is None or isinstance(family, str):
        raise ValueError(_FAMILY_HINT)

    if isinstance(family, type):
        if family.__name__ not in _FAMILY_CLASSES:
            raise ValueError(_FAMILY_HINT)
        fam_obj = family()
        fam = _FAMILY_CLASSES[family.__name__]
    else:
        cls = type(family).__name__
        if cls not in _FAMILY_CLASSES:
            raise ValueError(_FAMILY_HINT)
        fam_obj = family
        fam = _FAMILY_CLASSES[cls]

    obj_link = getattr(fam_obj, "link", None)
    lk = link if link is not None else (obj_link or _DEFAULT_LINK[fam])
    if fam == "negbin" and theta is None and getattr(fam_obj, "theta_is_fixed", False):
        theta = float(getattr(fam_obj.params, "theta", 1.0))
    elif theta is not None and fam != "negbin":
        raise ValueError("theta is only supported by the negative-binomial family")

    if dispersion is not None and fam not in _CONTINUOUS:
        raise ValueError("dispersion is only supported by the gaussian, gamma and invgaussian families")
    if fam in _CONTINUOUS and theta is not None:
        raise ValueError("theta is only supported by the negative-binomial family")
    if dispersion is not None:
        if not np.isfinite(dispersion) or float(dispersion) <= 0.0:
            raise ValueError("dispersion must be a finite positive number")

    lk = str(lk).lower().replace(" ", "")
    lk = _LINK_ALIASES.get(lk, lk)
    if lk not in _ALLOWED[fam]:
        raise ValueError(f"link {lk!r} not supported for {fam} (use {sorted(_ALLOWED[fam])})")
    return fam, lk, theta, (None if dispersion is None else float(dispersion))


# ---- link, variance and deviance functions (NumPy), consistent with the engine's glm_detail ----
def _linkinv(link: str, eta):
    """Inverse link ``mu = g^{-1}(eta)``.

    Parameters
    ----------
    link:
        One of ``"logit"``, ``"probit"``, ``"cloglog"``, ``"cauchit"``,
        ``"log"``, ``"sqrt"``, ``"identity"``, ``"inverse"`` or ``"invmu2"``.
    eta:
        Linear predictor (scalar or array).

    Returns
    -------
    numpy.ndarray
        Mean response, clipped to a numerically safe range.
    """
    from scipy import special
    eps = np.finfo(float).eps
    if link == "logit":
        return np.clip(special.expit(np.clip(eta, -30, 30)), eps, 1 - eps)
    if link == "probit":
        return np.clip(special.ndtr(np.clip(eta, -8.2, 8.2)), eps, 1 - eps)
    if link == "cloglog":
        return np.clip(-np.expm1(-np.exp(np.minimum(eta, 30))), eps, 1 - eps)
    if link == "cauchit":
        return np.clip(0.5 + np.arctan(eta) / np.pi, eps, 1 - eps)
    if link == "log":
        return np.maximum(np.exp(np.minimum(eta, 700)), eps)
    if link == "sqrt":
        return eta ** 2
    if link == "inverse":
        return 1.0 / np.maximum(eta, 1e-6)
    if link == "invmu2":
        return 1.0 / np.sqrt(np.maximum(eta, 1e-6))
    return eta


def _mu_eta(link: str, eta):
    """Derivative of the inverse link ``dmu/deta``.

    Parameters
    ----------
    link:
        Link name (see :func:`_linkinv`).
    eta:
        Linear predictor (scalar or array).

    Returns
    -------
    numpy.ndarray
        Derivative, clipped away from zero for numerical stability.
    """
    eps = np.finfo(float).eps
    if link == "logit":
        e = np.exp(-np.abs(np.clip(eta, -30, 30))); return np.maximum(e / (1 + e) ** 2, eps)
    if link == "probit":
        return np.maximum(np.exp(-0.5 * eta ** 2) / math.sqrt(2 * math.pi), eps)
    if link == "cloglog":
        e = np.minimum(eta, 30); return np.maximum(np.exp(e) * np.exp(-np.exp(e)), eps)
    if link == "cauchit":
        return np.maximum(1 / (np.pi * (1 + eta ** 2)), eps)
    if link == "log":
        return np.maximum(np.exp(np.minimum(eta, 700)), eps)
    if link == "sqrt":
        return 2 * eta
    if link == "inverse":
        e = np.maximum(eta, 1e-6)
        return -1.0 / (e * e)
    if link == "invmu2":
        e = np.maximum(eta, 1e-6)
        return -0.5 * np.power(e, -1.5)
    return np.ones_like(eta)


def _variance(family: str, mu, theta_nb, disp=1.0):
    """Variance function ``V(mu)`` of a GLM family.

    Parameters
    ----------
    family:
        ``"binomial"`` (``mu (1 - mu)``), ``"poisson"`` (``mu``),
        ``"negbin"`` (``mu + mu^2 / theta``), ``"gaussian"`` (``1``),
        ``"gamma"`` (``mu^2``) or ``"invgaussian"`` (``mu^3``).
    mu:
        Mean response.
    theta_nb:
        Negative-binomial dispersion (ignored by the other families).
    disp:
        Dispersion of the continuous families (multiplies ``V(mu)``); ignored
        by the discrete families.

    Returns
    -------
    numpy.ndarray
    """
    if family == "binomial":
        return mu * (1 - mu)
    if family == "poisson":
        return mu
    if family == "negbin":
        return mu + mu ** 2 / theta_nb
    if family == "gaussian":
        return np.full_like(mu, disp, dtype=float)
    if family == "gamma":
        return disp * mu ** 2
    return disp * mu ** 3


def _unit_deviance(family: str, y, mu, theta_nb):
    """Unit deviance ``d(y, mu)`` of a GLM family.

    Parameters
    ----------
    family:
        ``"binomial"``, ``"poisson"``, ``"negbin"``, ``"gaussian"``,
        ``"gamma"`` or ``"invgaussian"``.
    y:
        Observed response.
    mu:
        Mean response.
    theta_nb:
        Negative-binomial dispersion (ignored by the other families).

    Returns
    -------
    numpy.ndarray
    """
    def ylogy(a, b): return np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0) / b), 0.0)
    if family == "binomial":
        return 2 * (ylogy(y, mu) + ylogy(1 - y, 1 - mu))
    if family == "poisson":
        return 2 * (ylogy(y, mu) - (y - mu))
    if family == "negbin":
        return 2 * (ylogy(y, mu) - (y + theta_nb) * np.log((y + theta_nb) / (mu + theta_nb)))
    if family == "gaussian":
        return (y - mu) ** 2
    if family == "gamma":
        return 2 * ((y - mu) / mu - np.log(y / np.maximum(mu, 1e-12)))
    return (y - mu) ** 2 / (y * mu ** 2)


class GlmerFit(PlottingMixin):
    """Result of :meth:`GLMER.fit` (``glmer(...).fit()``).

    ``fixef``/``se``/``vcov`` cover only the estimable columns; ``aliased``
    lists the dropped ones.  Additional attributes include ``dispersion``
    (``sigma^2`` for Gaussian, ``1/shape`` for Gamma / inverse Gaussian),
    ``loglik``, ``deviance``, ``loglik_lme4``, ``theta_nb`` (negative binomial),
    ``theta``, ``nobs``, ``npar``, ``singular``, ``converged``, ``warning``,
    ``vcov_from_hessian``, ``varcorr`` and ``ranef``.
    """

    def __init__(self, raw, model, names, keep, index, y, wts, offset, formula, family, link, terms_info, fixed_spec, nAGQ, theta_fixed,
                 dispersion_fixed=False, data=None):
        """Wrap an engine result.

        Parameters
        ----------
        raw:
            Object returned by the C++ ``GlmmModel.fit()``.
        model:
            The fitted engine model.
        names:
            All candidate coefficient names.
        keep:
            Indices of the estimable coefficients.
        index:
            Row index of the data used in the fit.
        y:
            Response vector (proportion of successes for binomial data).
        wts:
            Prior weights (number of trials for binomial data).
        offset:
            Offset vector or ``None``.
        formula:
            Original formula string.
        family:
            Normalized family name.
        link:
            Link name.
        terms_info:
            Per-term information built by :func:`build_re_terms`.
        fixed_spec:
            Formulaic specification of the fixed-effects design.
        nAGQ:
            Number of quadrature points (1 means the Laplace approximation).
        theta_fixed:
            Fixed negative-binomial ``theta`` or ``None`` when estimated.
        dispersion_fixed:
            Whether the continuous-family dispersion was fixed by the user.
        data:
            Data frame actually used by the fit, kept for simulation-based
            residual diagnostics and conditional BLUP intervals.
        """
        self.raw, self._model, self.index, self._y, self._w, self._off = raw, model, index, y, wts, offset
        self._data = data
        self.formula, self.family, self.link, self.terms_info, self._fixed_spec, self.nAGQ = formula, family, link, terms_info, fixed_spec, nAGQ
        self.aliased = [names[j] for j in range(len(names)) if j not in keep]
        kn = [names[j] for j in keep]
        self._names, self._keep = kn, list(keep)
        self.fixef = pd.Series(np.array(raw.beta)[keep], index=kn, name="Estimate")
        self.se = pd.Series(np.array(raw.se)[keep], index=kn, name="Std. Error")
        self.vcov = pd.DataFrame(np.array(raw.vcov_beta)[np.ix_(keep, keep)], index=kn, columns=kn)
        self.loglik = raw.loglik
        self.deviance = -2.0 * raw.loglik
        self.loglik_lme4 = raw.loglik - raw.ll_sat if nAGQ > 1 else raw.loglik   # lme4 reports logLik this way when nAGQ > 1
        self.theta_nb = raw.theta_nb if family == "negbin" else None
        self.dispersion = float(raw.dispersion) if family in _CONTINUOUS else None
        self.dispersion_fixed = bool(dispersion_fixed)
        self.sigma = math.sqrt(self.dispersion) if family == "gaussian" and self.dispersion is not None else None
        self.theta = np.array(raw.theta)
        self.nobs = len(y)
        # npar as in lme4: coefficients + covariance parameters, plus one for
        # an estimated scale parameter (negative binomial theta, or the
        # dispersion of a continuous family).
        scale = (family == "negbin") or (family in _CONTINUOUS and not self.dispersion_fixed)
        self.npar = int(raw.rank) + len(raw.theta) + (1 if scale else 0)
        self.singular = bool(raw.singular)
        self.converged = bool(raw.opt.converged)
        self.warning = raw.warning
        self.vcov_from_hessian = bool(raw.vcov_hessian)
        rows = []
        for t, info in enumerate(terms_info):
            S = np.array(raw.D_terms[t]); sd = np.sqrt(np.diag(S))
            for c, cn in enumerate(info["cols"]):
                rows.append((info["name"], cn, S[c, c], sd[c], np.nan))
            for c in range(1, len(info["cols"])):
                for d in range(c):
                    rows.append((info["name"], f"corr({info['cols'][d]},{info['cols'][c]})", S[c, d], np.nan, S[c, d] / (sd[c] * sd[d]) if sd[c] > 0 and sd[d] > 0 else np.nan))
        self.varcorr = pd.DataFrame(rows, columns=["group", "name", "variance", "std_dev", "corr"])
        by: dict[str, list[pd.DataFrame]] = {}
        for t, info in enumerate(terms_info):
            by.setdefault(info["name"], []).append(pd.DataFrame(np.array(raw.blup[t]), index=info["levels"], columns=info["cols"]))
        self.ranef = {g: (l[0] if len(l) == 1 else pd.concat(l, axis=1)) for g, l in by.items()}

    # ---- fitted values, residuals, prediction ----------------------------------------------------------------
    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this fitted model.

        See :mod:`statcraft.plots` for the supported ``kind`` names.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def fitted(self, type: str = "response", re_form: str | bool | None = "all") -> pd.Series:
        """Fitted values on the link or response scale.

        Parameters
        ----------
        type:
            ``"response"`` (default, ``mu``) or ``"link"`` (``eta``).
        re_form:
            ``"all"``/``True`` includes the random effects; ``False``/``None``
            returns the population level.

        Returns
        -------
        pandas.Series

        Raises
        ------
        ValueError
            If ``type`` is not ``"response"`` or ``"link"``.
        """
        full = re_form in ("all", True)
        if type == "link":
            v = self.raw.eta if full else self.raw.eta_pop
        elif type == "response":
            v = self.raw.mu if full else self.raw.mu_pop
        else:
            raise ValueError("type must be 'response' or 'link'")
        return pd.Series(np.array(v), index=self.index)

    def residuals(self, type: str = "deviance") -> pd.Series:
        """residuals.glmerMod: 'deviance' (default), 'pearson', 'response' or 'working'."""
        mu = np.array(self.raw.mu); y, w = self._y, self._w
        tnb, disp = self.theta_nb, (self.dispersion or 1.0)
        if type == "response":
            r = y - mu
        elif type == "pearson":
            r = (y - mu) * np.sqrt(w) / np.sqrt(_variance(self.family, mu, tnb, disp))
        elif type == "working":
            r = (y - mu) / _mu_eta(self.link, np.array(self.raw.eta))
        elif type == "deviance":
            r = np.sign(y - mu) * np.sqrt(np.maximum(w * _unit_deviance(self.family, y, mu, tnb) / disp, 0.0))
        else:
            raise ValueError("type must be 'deviance', 'pearson', 'response' or 'working'")
        return pd.Series(r, index=self.index)

    def resid_quantile(self, seed: int = 42, n_sim: int = 500) -> pd.Series:
        """Simulated quantile residuals on the normal-score scale.

        A replicate of the response is simulated from each row's fitted
        conditional distribution (binomial with the fitted trials, Poisson or
        negative binomial with the fitted mean).  The scaled residual is the
        mid-p tail probability ``P(Y_sim < y_obs) + U P(Y_sim == y_obs)``
        divided by ``n_sim``, converted to a normal score.  This is the
        DHARMa construction: it is uniform under a correct discrete model for
        Bernoulli, grouped binomial ``k/n``, Poisson and negative-binomial
        responses, and it never produces the horizontal bands of raw or
        deviance residuals.

        The simulation uses the fitted conditional means, not a refit of each
        replicate, so it is a fast diagnostic rather than a parametric
        bootstrap test.
        """
        if self.family not in ("binomial", "poisson", "negbin"):
            raise ValueError(
                "simulated quantile residuals are available for binomial, "
                f"Poisson and negative-binomial GLMER fits, not {self.family!r}"
            )
        from scipy import stats

        rng = np.random.default_rng(int(seed))
        n_sim = max(50, int(n_sim))
        mu = np.asarray(self.raw.mu, dtype=float).reshape(-1)
        y = np.asarray(self._y, dtype=float).reshape(-1)
        if mu.size != y.size:
            raise ValueError("fitted means and observed response have different lengths")
        w = np.ones(y.size, dtype=float) if self._w is None else np.asarray(self._w, dtype=float).reshape(-1)
        if w.size != y.size:
            w = np.ones(y.size, dtype=float)

        if self.family == "binomial":
            trials = np.maximum(np.rint(w).astype(int), 1)
            target = np.clip(np.rint(y * trials).astype(int), 0, trials)
            sim = rng.binomial(trials, np.clip(mu, 0.0, 1.0), size=(n_sim, y.size))
        elif self.family == "poisson":
            target = np.rint(np.maximum(y, 0.0)).astype(int)
            sim = rng.poisson(np.maximum(mu, 0.0), size=(n_sim, y.size))
        else:
            theta = float(self.theta_nb or 1.0)
            if not np.isfinite(theta) or theta <= 0:
                theta = 1.0
            prob = theta / np.clip(theta + np.maximum(mu, 0.0), 1e-12, None)
            target = np.rint(np.maximum(y, 0.0)).astype(int)
            sim = rng.negative_binomial(theta, prob, size=(n_sim, y.size))

        lower = np.count_nonzero(sim < target[None, :], axis=0)
        tied = np.count_nonzero(sim == target[None, :], axis=0)
        u = (lower + rng.random(y.size) * tied) / n_sim
        u = np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
        return pd.Series(stats.norm.ppf(u), index=self.index)

    def blup_conditional_se(self) -> dict[str, pd.DataFrame]:
        """Posterior SDs of the BLUPs from the Laplace conditional covariance.

        For one random-effect term with covariance ``D``, design ``Z`` and
        GLM working weights ``W = w (dμ/dη)² / V(μ)``, the conditional covariance
        of the mode is ``(D⁻¹ + Zᵀ W Z)⁻¹``.  The returned tables have the same
        group names, levels and columns as :attr:`ranef`.
        """
        data = getattr(self, "_data", None)
        if data is None:
            return {}
        raw = self.raw
        eta = np.asarray(raw.eta, dtype=float).reshape(-1)
        mu = np.asarray(raw.mu, dtype=float).reshape(-1)
        w = np.ones(mu.size, dtype=float) if self._w is None else np.asarray(self._w, dtype=float).reshape(-1)
        if w.size != mu.size:
            w = np.ones(mu.size, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = _variance(self.family, mu, self.theta_nb, self.dispersion or 1.0)
            weights = w * _mu_eta(self.link, eta) ** 2 / np.maximum(variance, 1e-12)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)

        out: dict[str, pd.DataFrame] = {}
        for term, info in enumerate(self.terms_info):
            name = str(info["name"])
            frame = self.ranef.get(name)
            if frame is None:
                continue
            D = np.asarray(raw.D_terms[term], dtype=float)
            Z = np.asarray(design_new(info["spec"], data), dtype=float)
            if Z.ndim == 1:
                Z = Z[:, None]
            if Z.shape[0] != mu.size or D.ndim != 2 or Z.shape[1] != D.shape[0]:
                continue
            vars_ = list(info["vars"])
            if len(vars_) > 1:
                keys = [tuple(row) for row in data[vars_].to_numpy()]
            else:
                keys = list(data[vars_[0]]) if vars_ else [0] * mu.size
            ses = np.full((len(frame), Z.shape[1]), np.nan, dtype=float)
            D_inv = np.linalg.pinv(D)
            for row, level in enumerate(frame.index):
                mask = np.asarray([key == level for key in keys], dtype=bool)
                Zi = Z[mask]
                if Zi.shape[0] == 0:
                    continue
                cov = np.linalg.pinv(D_inv + Zi.T @ (weights[mask][:, None] * Zi))
                ses[row] = np.sqrt(np.maximum(np.diag(cov), 0.0))
            se_frame = pd.DataFrame(ses, index=frame.index, columns=frame.columns)
            if np.isfinite(se_frame.to_numpy()).all():
                out[name] = se_frame
        return out

    def predict(self, newdata: Optional[pd.DataFrame] = None, type: str = "link", re_form: str | bool | Sequence[str] | None = "all",
                allow_new_levels: bool = False, offset=None) -> pd.Series:
        """Like predict.merMod: type='link' (default) or 'response'; re_form='all', False/None
        (fixed effects only, re.form=NA) or a list of groups; allow_new_levels=True assigns
        effect 0 to levels not seen in the fit. If the model has an offset, newdata requires offset=."""
        if type not in ("link", "response"):
            raise ValueError("type must be 'link' or 'response'")
        if newdata is None:
            return self.fitted(type, "all" if re_form not in (False, None) else False)
        nd = pd.DataFrame(newdata).reset_index(drop=True)
        X = design_new(self._fixed_spec, nd)[:, self._keep]
        eta = X @ self.fixef.to_numpy()
        if self._off is not None and np.any(self._off != 0):
            if offset is None:
                raise ValueError("the model was fitted with an offset: pass offset= for newdata")
        if offset is not None:
            eta = eta + np.asarray(offset, float)
        use = None if re_form in ("all", True) else ([] if re_form in (False, None) else list(re_form))
        for t, info in enumerate(self.terms_info):
            if use is not None and info["name"] not in use:
                continue
            Z = design_new(info["spec"], nd)
            key = list(zip(*[nd[c] for c in info["vars"]])) if len(info["vars"]) > 1 else list(nd[info["vars"][0]])
            idx = np.array([info["level_map"].get(k, -1) for k in key])
            if (idx < 0).any() and not allow_new_levels:
                raise ValueError(f"newdata has levels of '{info['name']}' not seen in the fit {[key[i] for i in np.flatnonzero(idx < 0)][:3]}; use allow_new_levels=True")
            b = np.array(self.raw.blup[t])
            eta = eta + (Z * np.where((idx >= 0)[:, None], b[np.maximum(idx, 0)], 0.0)).sum(axis=1)
        return pd.Series(_linkinv(self.link, eta) if type == "response" else eta, index=nd.index)

    # ---- inference -------------------------------------------------------------------------------------------
    def aic(self) -> float:
        """Akaike information criterion, ``-2 logLik + 2 npar``."""
        return -2.0 * self.loglik + 2.0 * self.npar

    def bic(self) -> float:
        """Bayesian information criterion, ``-2 logLik + log(n) npar``."""
        return -2.0 * self.loglik + math.log(self.nobs) * self.npar

    def tTable(self) -> pd.DataFrame:
        """Estimate, Std. Error, z value and Pr(>|z|) (Wald test with the lme4 vcov: Hessian of the deviance)."""
        from scipy import stats
        t = pd.DataFrame({"Estimate": self.fixef, "Std. Error": self.se})
        t["z value"] = t["Estimate"] / t["Std. Error"]
        t["Pr(>|z|)"] = 2.0 * stats.norm.sf(np.abs(t["z value"]))
        return t

    def fit(self):
        """Return the already-fitted result itself (chainable API: glmer(...).fit())."""
        return self

    def summary(self, alpha: float = 0.05):
        """statsmodels-style summary (statcraft.summary) of the glmer fit."""
        lhs = str(self.formula).split("~", 1)[0].strip() or "y"
        method = "Laplace" if self.nAGQ == 1 else f"AGQ ({self.nAGQ} nodes)"
        counts = grouping_counts(self.terms_info)
        groups = grouping_text(self.terms_info)
        left = [
            ("Dep. Variable:", lhs),
            ("Model:", "GLMM"),
            ("Family:", self.family),
            ("Link:", self.link),
            ("Method:", method),
            ("No. Observations:", str(self.nobs)),
            ("No. Groups:", str(sum(counts.values()))),
        ]
        right = [
            ("Log-Likelihood:", f"{self.loglik:.6f}"),
            ("AIC:", f"{self.aic():.4f}"),
            ("BIC:", f"{self.bic():.4f}"),
        ]
        if self.theta_nb is not None:
            right.append(("Theta (NB):", _fmt_value(self.theta_nb)))
        if self.dispersion is not None:
            label = "Sigma:" if self.family == "gaussian" else "Dispersion:"
            value = self.sigma if self.family == "gaussian" else self.dispersion
            right.append((label, _fmt_value(value)))
        right.append(("Converged:", str(self.converged)))
        left_extra = [
            ("Grouping:", groups or "none"),
            ("Fixed effects:", str(len(self.fixef))),
            ("Variance params:", str(len(self.theta))),
            ("Free parameters:", str(self.npar)),
        ]
        right_extra = [
            ("Singular fit:", str(self.singular)),
            ("Residual df:", str(max(self.nobs - int(len(self.fixef)), 1))),
            ("Fixed effects:", str(len(self.fixef))),
            ("Variance params:", str(len(self.theta))),
        ]
        left, right = balanced_info(left, right, left_extra, right_extra)

        blocks = [fixed_block(self.fixef, self.se, "normal")]
        tables = crossed_random_tables(self, residual=False, scale="link")
        resid_table = residual_summary_table(
            self, heading="Deviance residuals", residual_type="deviance"
        )
        if resid_table is not None:
            tables.append(resid_table)
        fit_table = glmer_fit_table(self)
        if fit_table is not None:
            tables.append(fit_table)
        notes = [self.warning] if self.warning else []
        if self.singular:
            notes.append("The random-effects covariance is singular (some variance components are near zero).")
        if any(str(getattr(t, "heading", "")).startswith("Random effects") for t in tables):
            notes.append("Random-effects variances are reported on the linear predictor scale.")
        return build_summary(
            title="Generalized Linear Mixed-Effects Model Results",
            subtitle=f"Formula: {self.formula}",
            left=left,
            right=right,
            blocks=blocks,
            tables=tables,
            dist="normal",
            alpha=alpha,
            notes=notes,
        )

    def __repr__(self):
        return _compact_fit_repr(self)


def _column_or_array(x, df: pd.DataFrame, name: str):
    """Resolve a ``weights``/``offset`` argument to a column name or an array.

    Parameters
    ----------
    x:
        ``None``, a column name (``str``) or an array-like.
    df:
        Data frame (reserved for consistency; not modified).
    name:
        Argument name, used for error messages.

    Returns
    -------
    column : str | None
        The column name when ``x`` is a string.
    array : numpy.ndarray | None
        The array when ``x`` is array-like.
    """
    if x is None:
        return None, None
    if isinstance(x, str):
        return x, None
    return None, np.asarray(x, float)


def _fit_glmer(formula: str, data, family: str, link: Optional[str] = None, nAGQ: int = 1, weights=None, offset=None,
               theta: Optional[float] = None, dispersion: Optional[float] = None, na_action: str = "omit", control: Optional[Mapping] = None,
               pd_struct=None) -> GlmerFit:
    """Run the GLMM engine and return a :class:`GlmerFit`.

    Construct models with :class:`GLMER` (alias :func:`glmer`) and call
    ``.fit()``; this helper is the internal implementation used by that step.
    ``family``/``link`` are expected to be normalized by
    :func:`_normalize_family_link`.

    Examples
    --------
    ::

        glmer("cbind(incidence, size - incidence) ~ period + (1 | herd)", cbpp, family=Binomial("logit")).fit()
        glmer("TICKS ~ YEAR + (1 | BROOD)", grouse, family=Poisson("log")).fit()
        glmer("y ~ x + (x | g)", d, family=Binomial("probit"), nAGQ=10).fit()
        glmer("y ~ x + (1 | g)", d, family=NegativeBinomial(theta=2.0)).fit()
        glmer("y ~ x + (1 | g)", d, family=Gaussian()).fit()
        glmer("y ~ x + (1 | g)", d, family=Gamma("log")).fit()
        glmer("y ~ x + (1 | g)", d, family=InverseGaussian()).fit()

    Parameters
    ----------
    formula:
        lme4-style formula.  For binomial data the response may be a
        proportion (with ``weights`` = number of trials) or
        ``cbind(successes, failures)``.
    data:
        Data frame or anything accepted by :class:`pandas.DataFrame`.
    family, link:
        Normalized family name (``'binomial'``, ``'poisson'``, ``'negbin'``,
        ``'gaussian'``, ``'gamma'``, ``'invgaussian'``) and link.  Callers
        normally pass GLM-style family objects such as ``Poisson("log")`` or
        ``Gamma("log")`` to :class:`GLMER`.
    nAGQ:
        Number of quadrature points; ``1`` (default) uses the Laplace
        approximation, values ``> 1`` adaptive Gauss-Hermite quadrature.
    weights, offset:
        Column name or vector.
    theta:
        Fixed negative-binomial dispersion; when ``None`` it is estimated.
    dispersion:
        Fixed dispersion of a continuous family (``sigma^2`` for Gaussian,
        ``1/shape`` for Gamma and inverse Gaussian); when ``None`` it is
        estimated.  Only used by the continuous families.
    na_action:
        ``"omit"`` (default) or ``"fail"``.
    control:
        Optional dict with ``optimizer``, ``max_iter``, ``gtol``, ``ftol``,
        ``fd_step``, ``theta_bound``, ``init_stage``, ``pirls_tol``,
        ``pirls_step_tol`` or ``max_pirls``.
    pd_struct:
        Optional covariance parametrization (see :mod:`statcraft.nlme.structs`).

    Returns
    -------
    GlmerFit

    Raises
    ------
    ValueError
        For a formula without random effects, invalid ``theta`` or an unknown
        control option.

    Notes
    -----
    Standard errors use, as in glmer, the numeric Hessian of the deviance in
    ``(theta, beta)``.
    """
    fam = family
    if fam not in _DEFAULT_LINK:
        raise ValueError(_FAMILY_HINT)
    lk = str(link or _DEFAULT_LINK[fam]).lower()
    if lk not in _ALLOWED[fam]:
        raise ValueError(f"link {lk!r} not supported for {fam} (use {sorted(_ALLOWED[fam])})")
    df = pd.DataFrame(data)
    lhs, fixed_rhs, terms = parse_lmer(formula)
    if not terms:
        raise ValueError("the formula has no random-effect terms '(... | g)'")
    cm = re.fullmatch(r"\s*cbind\((.+),(.+)\)\s*", lhs, re.S) if fam == "binomial" else None
    wcol, warr = _column_or_array(weights, df, "weights")
    ocol, oarr = _column_or_array(offset, df, "offset")
    exprs = ([cm.group(1).strip(), cm.group(2).strip()] if cm else [lhs]) + [fixed_rhs] + [t[0] for t in terms] + [c for t in terms for c in t[1]] + [wcol, ocol]
    cols = collect_columns(exprs, df)
    if warr is not None:
        df = df.assign(__w=warr); cols.append("__w"); wcol = "__w"
    if oarr is not None:
        df = df.assign(__o=oarr); cols.append("__o"); ocol = "__o"
    df2, _ = handle_na(df, cols, na_action)
    if cm:
        a, b = eval_expr(cm.group(1).strip(), df2), eval_expr(cm.group(2).strip(), df2)
        tot = a + b
        if (tot <= 0).any():
            raise ValueError("cbind(successes, failures) has rows with no trials")
        y, w = a / tot, tot
    else:
        y, w = eval_expr(lhs, df2), np.ones(len(df2))
    y = np.asarray(y, dtype=float)
    if fam in ("gamma", "invgaussian") and np.any(y <= 0):
        raise ValueError("the response must be > 0 for the gamma and invgaussian families")
    if fam in ("poisson", "negbin") and np.any(y < 0):
        raise ValueError("the response must be >= 0 for count families")
    if fam == "gaussian":
        if lk in ("log", "inverse") and np.any(y <= 0):
            raise ValueError(f"the response must be > 0 for the gaussian family with link={lk!r}")
        if lk == "sqrt" and np.any(y < 0):
            raise ValueError("the response must be >= 0 for the gaussian family with link='sqrt'")
    if wcol:
        w = w * df2[wcol].to_numpy(float)
    off = df2[ocol].to_numpy(float) if ocol else None
    X, names, fspec = design_spec(fixed_rhs, df2)
    rt, terms_info = build_re_terms(df2, terms, pd_struct)
    spec = core.GlmmSpec()
    spec.family = {
        "binomial": core.GlmFamily.Binomial, "poisson": core.GlmFamily.Poisson, "negbin": core.GlmFamily.NegBin,
        "gaussian": core.GlmFamily.Gaussian, "gamma": core.GlmFamily.Gamma, "invgaussian": core.GlmFamily.InvGauss,
    }[fam]
    spec.link = getattr(core.GlmLink, _LINKS[lk])
    spec.nAGQ = int(nAGQ)
    if fam == "negbin":
        spec.estimate_theta = theta is None
        spec.theta_nb = 1.0 if theta is None else float(theta)
    elif theta is not None:
        raise ValueError("theta is only used with family='negbin'")
    if fam in _CONTINUOUS:
        spec.estimate_dispersion = dispersion is None
        spec.dispersion = 1.0 if dispersion is None else float(dispersion)
    elif dispersion is not None:
        raise ValueError("dispersion is only used with the gaussian, gamma and invgaussian families")
    if control:
        for k, v in dict(control).items():
            if k == "optimizer":
                spec.optimizer = getattr(core.Optimizer, {"internal": "Internal", "trust": "TrustRegion", "lbfgsb": "LBFGSB", "scipy": "ScipyBFGS"}[str(v).lower()])
            elif k in ("max_iter", "gtol", "ftol", "fd_step"):
                setattr(spec.opt, k, v)
            elif k in ("theta_bound", "init_stage", "pirls_tol", "pirls_step_tol", "max_pirls"):
                setattr(spec, k, v)
            else:
                raise ValueError(f"unknown control option for glmer: {k!r}")
    model = core.GlmmModel(np.ascontiguousarray(y), np.asfortranarray(X), rt, spec, np.ascontiguousarray(w) if (cm or wcol) else np.zeros(0),
                           np.ascontiguousarray(off) if off is not None else np.zeros(0))
    raw = model.fit()
    keep = [j for j in range(X.shape[1]) if j not in set(model.aliased)]
    return GlmerFit(raw, model, names, keep, df2.index, y, w, off, formula, fam, lk, terms_info, fspec, int(nAGQ), theta,
                    dispersion_fixed=(fam in _CONTINUOUS and dispersion is not None), data=df2)


class GLMER(_ModelBase):
    """Unfitted generalized linear mixed model with lme4 syntax.

    Like ``LME``/``GLS``/``NLME``/``GNLS``, construction only stores the
    specification; call :meth:`fit` to obtain a :class:`GlmerFit`::

        from statcraft.glm import Poisson, Gamma
        res = glmer("y ~ x + (1 | g)", data, family=Poisson("log")).fit()
        res = glmer("y ~ x + (1 | g)", data, family=Gamma("log")).fit()

    ``family`` is required and must be a GLM family object (``Binomial("logit")``,
    ``Poisson("log")``, ``NegativeBinomial(theta=2.0)``, ``Gaussian()``,
    ``Gamma()``, ``InverseGaussian()``) or the corresponding class.  Family
    strings such as ``'binomial'`` are rejected to avoid ambiguity.  Continuous
    families estimate the dispersion unless ``dispersion=`` is supplied.
    ``GLMER`` and its lowercase alias :func:`glmer` never fit at construction
    time.
    """

    def __init__(self, formula: str, data, family, link: Optional[str] = None,
                 nAGQ: int = 1, weights=None, offset=None, theta: Optional[float] = None,
                 dispersion: Optional[float] = None, na_action: str = "omit",
                 control: Optional[Mapping] = None, pd_struct=None):
        self.formula = formula
        self.data = data
        self.family, self.link, self.theta, self.dispersion = _normalize_family_link(family, link, theta, dispersion)
        self.nAGQ = int(nAGQ)
        self.weights, self.offset = weights, offset
        self.na_action, self.control, self.pd_struct = na_action, control, pd_struct

    def _fit(self):
        return _fit_glmer(self.formula, self.data, family=self.family, link=self.link,
                          nAGQ=self.nAGQ, weights=self.weights, offset=self.offset,
                          theta=self.theta, dispersion=self.dispersion, na_action=self.na_action,
                          control=self.control, pd_struct=self.pd_struct)


#: Lowercase alias for the R/lme4 API (returns an unfitted model; call .fit()).
glmer = GLMER
