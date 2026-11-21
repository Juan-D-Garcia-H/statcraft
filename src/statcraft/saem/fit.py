"""SAEM fitting engine and results.

The result object :class:`SaemFit` exposes R-style methods (``fixef``, ``ranef``, ``coef``,
``vcov``, ``fitted``, ``residuals``, ``npde``, ``predict``, ``simulate``, ``summary``, ``anova``, ``conf_int``).
Model construction and fitting live in :mod:`statcraft.saem.api`:
``SAEM(...).fit()`` / ``saem(...).fit()``.
"""
from __future__ import annotations

import math
import warnings
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import _saemfit as _core
from ..plots.methods import PlottingMixin
from ..summary import Summary, build_summary
from ._utils import ERROR_PARS, apply_transform
from .control import SaemControl
from .data import DataInfo, as_frame, prepare
from .model import NLModel


class SaemWarning(UserWarning):
    """Fitting warnings (variance collapse, singular FIM, ...)."""


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------
def _norm_sf(z: float) -> float:
    """Two-sided standard-normal tail probability ``P(|Z| > |z|)`` (NaN for non-finite ``z``)."""
    return math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else float("nan")


def _gammaq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:                       # series
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1.0 - a, 1e300, 1.0 / (x + 1.0 - a)    # continued fraction
    h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2.0
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_sf(x: float, k: int) -> float:
    """Upper-tail probability of a chi-square distribution with ``k`` degrees of freedom."""
    return _gammaq(0.5 * k, 0.5 * x) if x > 0 and k > 0 else float("nan")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _fmt_num(x, p: int = 6) -> str:
    """Compact numeric formatting for the summary header (``n/a`` when non-finite)."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.{p}g}" if np.isfinite(v) else "n/a"


#: Compact labels for the verbose standard-error descriptions produced by the engine.
#: The first matching entry wins, so the more specific patterns come first.
_SE_METHOD_LABELS = (
    ("numerical Hessian", "observed FIM"),
    ("Louis' formula per cluster", "observed FIM (Louis, cluster)"),
    ("Louis' formula", "observed FIM (Louis)"),
    ("adaptive Gauss-Hermite", "observed FIM (Gauss-Hermite)"),
    ("hierarchical linearized", "linearized FIM (cluster)"),
    ("linearized FIM", "linearized FIM"),
)

#: Compact labels for the log-likelihood method (already short except for the
#: cluster-level importance-sampling variants).
_LL_METHOD_LABELS = (
    ("joint importance sampling", "importance sampling (cluster)"),
    ("adaptive nested Gauss-Hermite", "nested Gauss-Hermite"),
)


def _method_label(name: str, labels) -> str:
    """Short, table-friendly label for an engine method description.

    The engine reports long, self-documenting names such as
    ``"observed FIM (numerical Hessian of the Laplace logLik, deterministic)"``.
    Those names widen the whole summary header without adding information, so
    the printed summary uses a compact label while the raw string stays
    available as ``fit.res.se_method`` / ``fit.res.loglik_method``.
    """
    text = str(name)
    for needle, label in labels:
        if needle in text:
            return label
    return text


def _block(heading: str, names, coefs, se=None) -> dict:
    """One coefficient block in the shape ``statcraft.summary.build_summary`` expects."""
    return {
        "heading": heading,
        "names": [str(n) for n in names],
        "coefs": np.asarray(coefs, dtype=float),
        "se": None if se is None else np.asarray(se, dtype=float),
    }


class SaemSummary(Summary):
    """Result of ``fit.summary()``.

    It is a regular :class:`statcraft.summary.Summary` -- ``print(summary)`` gives the report,
    ``summary.tables[i].data`` gives each table as a pandas object, and ``save_html`` / ``save_latex`` /
    ``save_excel`` export it -- with the fitted object and its main tables also attached as attributes
    (``fit``, ``fixed``, ``random``, ``corr``, ``error``, ``stats``).
    """

    def __init__(self, built: Summary, fit: "SaemFit", tables: dict):
        super().__init__(built.tables, built.extra_txt, built.as_text(), built.as_html())
        self.fit = fit
        self.fixed = tables["fixed"]
        self.random = tables["random"]
        self.corr = tables["corr"]
        self.error = tables["error"]
        self.stats = tables["stats"]


# ---------------------------------------------------------------------------
# SaemFit
# ---------------------------------------------------------------------------
class SaemFit(PlottingMixin):
    """Fit result. Main methods (R names in snake_case):

    ``summary()``, ``fixef()``, ``pop_params()``, ``ranef()``, ``coef()``, ``vcov()``, ``omega()``,
    ``sigma()``, ``fitted()``, ``residuals()``, ``npde()``, ``predict()``, ``simulate()``, ``conf_int()``,
    ``anova()``, ``to_frame()``.  Properties: ``loglik``, ``aic``, ``bic``, ``df``,
    ``n_obs``, ``nobs``, ``n_subjects``, ``n_params``, ``warnings``, ``res`` (raw result of the
    C++ core).

    ``fit()`` returns the object itself, so the API is also chainable after fitting
    (``saem(...).fit().summary()``).
    """

    def __init__(self, res: "_core.Results", model: NLModel, info: DataInfo, control: SaemControl):
        """Wrap a low-level fit result.

        Parameters
        ----------
        res : _core.Results
            Raw result returned by the C++ core.
        model : NLModel
            Model that was fitted.
        info : DataInfo
            Metadata of the prepared data.
        control : SaemControl
            Control actually used for the fit.

        Any warning reported by the engine is re-emitted as a :class:`SaemWarning`.
        """
        self.res = res
        self.model = model
        self.info = info
        self.control = control
        self.params = list(model.params)
        self.warnings = list(res.warnings)
        for w in self.warnings:
            warnings.warn(w, SaemWarning, stacklevel=3)
        self._nb = len(res.coef)

    def fit(self) -> "SaemFit":
        """The model is already fitted: returns ``self`` (chainable API)."""
        return self

    # ------------------------------------------------------------------ properties
    @property
    def n_obs(self) -> int:
        """Number of observations used in the fit."""
        return int(self.res.n_obs)

    @property
    def n_subjects(self) -> int:
        """Number of subjects used in the fit."""
        return int(self.res.n_subjects)

    @property
    def n_params(self) -> int:
        """Number of structural-model parameters."""
        return len(self.params)

    @property
    def loglik(self) -> float:
        """Marginal log-likelihood (method in ``summary().stats['loglik_method']``); NaN if not computed."""
        return float(self.res.loglik) if self.res.loglik_computed else float("nan")

    @property
    def df(self) -> int:
        """Number of free estimated parameters."""
        if self.res.se_computed and self.res.cov_theta.size:
            # fixed parameters do not enter cov_theta; count their non-zero diagonal
            return int(np.sum(np.abs(np.diag(self.res.cov_theta)) > 0))
        return int(len(self.res.theta))

    @property
    def nobs(self) -> int:
        """Alias of :attr:`n_obs` (statsmodels convention)."""
        return self.n_obs

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``-2 logLik + 2 df``."""
        return -2.0 * self.loglik + 2.0 * self.df

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``-2 logLik + log(n_obs) df``."""
        return -2.0 * self.loglik + math.log(max(self.n_obs, 1)) * self.df

    @property
    def _ids(self):
        """Index with the original subject labels (named after the subject column)."""
        return pd.Index(self.info.ids, name=self.info.id_col)

    def _stack(self, lst) -> np.ndarray:
        """Stack a list of per-subject arrays into a matrix, with an empty fallback."""
        return np.vstack([np.asarray(v) for v in lst]) if len(lst) else np.empty((0, len(self.params)))

    # ------------------------------------------------------------------ fixed effects
    def fixef(self, scale: str = "estimation") -> pd.Series:
        """Fixed coefficients. ``scale='estimation'`` (phi: the scale in which they are estimated) or
        ``'natural'`` (intercepts on the psi scale = h(phi); covariate coefficients remain in phi)."""
        v = np.asarray(self.res.coef, float).copy()
        if scale == "natural":
            v[:len(self.params)] = np.asarray(self.res.psi_pop)
        elif scale != "estimation":
            raise ValueError("scale: 'estimation' or 'natural'")
        return pd.Series(v, index=list(self.res.coef_names), name="estimate")

    def pop_params(self) -> pd.Series:
        """Typical population values on the natural scale, psi = h(beta) (subject with covariates at 0)."""
        return pd.Series(np.asarray(self.res.psi_pop), index=self.params, name="psi_pop")

    def vcov(self) -> pd.DataFrame:
        """(Asymptotic) covariance of the fixed coefficients."""
        if not self.res.se_computed:
            raise RuntimeError("standard errors were not computed (control: se != 'none')")
        C = np.asarray(self.res.cov_theta)[:self._nb, :self._nb]
        n = list(self.res.coef_names)
        return pd.DataFrame(C, index=n, columns=n)

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this SAEM fit.

        See :mod:`statcraft.plots` for the supported ``kind`` names; the
        default is the adaptive diagnostics panel.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def conf_int(self, alpha: float = 0.05, scale: str = "estimation") -> pd.DataFrame:
        """Wald confidence intervals for the fixed coefficients.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level in (0, 1), i.e. confidence level ``1 - alpha``.
        scale : {"estimation", "natural"}, default "estimation"
            ``"estimation"`` returns intervals on the estimation (phi) scale.
            ``"natural"`` transforms the structural-parameter bounds via the model
            parameter transforms (``apply_transform``), while covariate-effect
            coefficients remain on the estimation scale.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        level = 1.0 - alpha
        from statistics import NormalDist
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        est = self.fixef(scale="estimation")
        se_arr = np.asarray(self.res.se_coef, float)
        if len(se_arr) < len(est):
            se_arr = np.pad(se_arr, (0, len(est) - len(se_arr)), constant_values=np.nan)
        se = pd.Series(se_arr[:len(est)], index=est.index, dtype=float)
        lower = est - z * se
        upper = est + z * se
        if scale == "natural":
            est = self.fixef(scale="natural")
            p = len(self.params)
            lo = lower.to_numpy(dtype=float).copy()
            hi = upper.to_numpy(dtype=float).copy()
            for j, tr in enumerate(self.model.transform):
                lo[j] = apply_transform(lo[j], tr)
                hi[j] = apply_transform(hi[j], tr)
                # ensure bounds ordered (monotone transforms preserve order, but be safe)
                if np.isfinite(lo[j]) and np.isfinite(hi[j]) and lo[j] > hi[j]:
                    lo[j], hi[j] = hi[j], lo[j]
            lower = pd.Series(lo, index=est.index)
            upper = pd.Series(hi, index=est.index)
        elif scale != "estimation":
            raise ValueError("scale: 'estimation' or 'natural'")
        return pd.DataFrame({"estimate": est, "lower": lower, "upper": upper})

    # ------------------------------------------------------------------ random effects
    def omega(self) -> pd.DataFrame:
        """Covariance matrix of the random effects (phi scale)."""
        return pd.DataFrame(np.asarray(self.res.omega), index=self.params, columns=self.params)

    def sigma(self) -> pd.DataFrame:
        """Error-model parameters per response (a, b, c)."""
        rows = []
        for r, ep in enumerate(self.res.error_pars):
            nm = self.model.errors[min(r, len(self.model.errors) - 1)]
            rows.append({"response": self._resp_label(r), "model": nm, "a": ep.a, "b": ep.b, "c": ep.c})
        return pd.DataFrame(rows).set_index("response") if rows else pd.DataFrame()

    def _resp_label(self, r: int):
        """User-facing label of response index ``r`` (the integer index if unlabelled)."""
        lab = self.info.ytype_labels
        return lab[r] if lab is not None and r < len(lab) else r

    def _pop_phi(self) -> np.ndarray:
        """Population phi of each subject: beta + covariate effects (ns x p)."""
        p, ns = len(self.params), self.n_subjects
        coef = np.asarray(self.res.coef, float)
        P = np.tile(coef[:p], (ns, 1))
        for t, nm in enumerate(list(self.res.coef_names)[p:]):
            pname, cov = nm.split("~", 1)
            j = self.params.index(pname)
            if cov in self.info.cov_values:
                col = self.info.cov_values[cov]
            else:
                base, _, lev = cov.partition("=")
                col = (self.info.cat_labels[base] == lev).astype(float)
            P[:, j] += coef[p + t] * col
        return P

    def _ebe(self) -> np.ndarray:
        """Per-subject conditional means on the phi scale (falls back to the sampled ``phi``)."""
        return self._stack(self.res.ebe if self.res.cond_computed and len(self.res.ebe) else self.res.phi)

    def ranef(self, kind: str = "eta") -> pd.DataFrame:
        """Individual random effects (conditional mean): ``kind='eta'`` (deviations on the phi scale,
        only parameters with a random effect), ``'phi'`` or ``'psi'`` (individual parameters)."""
        if kind == "psi":
            return pd.DataFrame(self._stack(self.res.psi_ebe), index=self._ids, columns=self.params)
        phi = self._ebe()
        if kind == "phi":
            return pd.DataFrame(phi, index=self._ids, columns=self.params)
        if kind != "eta":
            raise ValueError("kind: 'eta', 'phi' or 'psi'")
        eta = phi - self._pop_phi()
        cols = [j for j, n in enumerate(self.params) if n in self.model.random]
        return pd.DataFrame(eta[:, cols], index=self._ids, columns=[self.params[j] for j in cols])

    def coef(self) -> pd.DataFrame:
        """Individual parameters on the natural scale (like nlme's ``coef()``)."""
        return self.ranef("psi")

    def cluster_ranef(self) -> pd.DataFrame:
        """(hierarchical) conditional mean of the cluster effects."""
        if not self.res.hierarchical:
            raise RuntimeError("the model is not hierarchical (use group=...)")
        X = self._stack(self.res.xi_cond_mean) if len(self.res.xi_cond_mean) else self._stack(self.res.xi)
        idx = pd.Index(self.info.group_labels, name="cluster")
        return pd.DataFrame(X, index=idx[:len(X)], columns=self.params)

    # ------------------------------------------------------------------ fit / residuals
    def _check_gaussian(self, what: str):
        """Raise ``NotImplementedError`` unless the family is Gaussian.

        ``what`` names the caller in the error message.
        """
        if self.model.family != "gaussian":
            raise NotImplementedError(f"{what}: only for Gaussian families")

    def fitted(self, level: str = "individual") -> pd.Series:
        """Predictions: ``level='individual'`` (with EBE) or ``'population'`` (random effects = 0)."""
        self._check_gaussian("fitted")
        if level == "individual":
            v = np.asarray(self.res.fitted_ind)
        elif level == "population":
            v = np.asarray(self.res.fitted_pop)
        else:
            raise ValueError("level: 'individual' or 'population'")
        return pd.Series(v, index=self.info.row_index, name=f"fitted_{level[:3]}")

    def _err_sd(self, f: np.ndarray) -> np.ndarray:
        """Residual standard deviation of each observation given prediction ``f``."""
        yt = np.asarray(self.res.data.ytype, int) if len(self.res.data.ytype) else np.zeros(len(f), int)
        sd = np.empty(len(f))
        for r, ep in enumerate(self.res.error_pars):
            nm = self.model.errors[min(r, len(self.model.errors) - 1)]
            m = yt == r
            a, b, c = ep.a, ep.b, ep.c
            ff = np.abs(f[m])
            if nm in ("constant", "exponential"): sd[m] = a
            elif nm == "proportional": sd[m] = b * ff
            elif nm == "combined": sd[m] = a + b * ff
            elif nm == "combined2": sd[m] = np.sqrt(a * a + (b * ff) ** 2)
            else: sd[m] = a + b * ff ** c
        return sd

    def residuals(self, level: str = "individual", type: str = "raw", nsim: int = 1000, seed: int = 1) -> pd.Series:
        """Residuals.

        Parameters
        ----------
        level : {"individual", "population"}, default "individual"
            Prediction level for raw/pearson residuals.
        type : {"raw", "pearson", "npde", "quantile"}, default "raw"
            Residual type. ``"raw"`` and ``"pearson"`` are available for
            Gaussian-family models (censored observations are set to NaN).
            ``"npde"``/``"quantile"`` compute simulation-based normalized
            prediction distribution / quantile residuals using the C++ simulation
            engine and are supported for censored and non-Gaussian models.
        nsim : int, default 1000
            Number of simulations for ``type='npde'``/``'quantile'``.
        seed : int, default 1
            Random seed for simulation-based residuals.
        """
        type = str(type).lower()
        if type in ("npde", "pd", "quantile", "quantile_residuals"):
            return self.npde(nsim=nsim, seed=seed)
        self._check_gaussian("residuals")
        f = self.fitted(level).to_numpy()
        y = np.asarray(self.res.data.y)
        yt = np.asarray(self.res.data.ytype, int) if len(self.res.data.ytype) else np.zeros(len(y), int)
        r = y - f
        for k, nm in enumerate(self.model.errors):
            m = yt == k
            if nm == "exponential":
                with np.errstate(all="ignore"):
                    r[m] = np.log(y[m]) - np.log(f[m])
        if type == "pearson":
            r = r / self._err_sd(f)
        elif type not in ("raw", "response"):
            raise ValueError("type: 'raw', 'pearson', 'npde' or 'quantile'")
        cens = np.asarray(self.res.data.cens)
        if len(cens):
            r = np.where(cens != 0, np.nan, r)
        return pd.Series(r, index=self.info.row_index, name=f"resid_{type if type != 'response' else 'raw'}")

    def _simulated_y(self, nsim: int = 1000, seed: int = 1, new_random_effects: bool = True):
        """Simulated response matrix with shape (n_obs, nsim)."""
        nan = float("nan")
        sims = _core.simulate(self.res, int(seed) & 0xFFFFFFFF, int(nsim), nan, nan, bool(new_random_effects),
                              self.model.is_compiled)
        if not sims:
            return np.empty((self.n_obs, 0), dtype=float)
        return np.column_stack([np.asarray(d.y, dtype=float) for d in sims])

    def npde(self, nsim: int = 1000, seed: int = 1, decorrelate: bool = False) -> pd.Series:
        """Simulation-based normalized prediction distribution errors / quantile residuals.

        Uses the C++ simulation engine and is applicable to censored data and
        non-Gaussian families where :meth:`fitted`/:meth:`residuals` may be unavailable.

        Parameters
        ----------
        nsim : int, default 1000
            Number of Monte Carlo simulations.
        seed : int, default 1
            Random seed.
        decorrelate : bool, default False
            If True, apply per-subject empirical decorrelation of simulated
            predictions (Brendel-style NPDE: whiten ``y`` using the empirical mean
            and covariance of simulations, then compute prediction discrepancies in
            the decorrelated space). If False, return per-observation normal
            quantile residuals from simulated predictive distributions.

        Returns
        -------
        pandas.Series
            Residuals aligned with observations (NaN when not computable).
        """
        nsim = int(nsim)
        if nsim <= 0:
            raise ValueError(f"nsim must be positive, got {nsim!r}")
        ysim = self._simulated_y(nsim=nsim, seed=seed, new_random_effects=True)
        y = np.asarray(self.res.data.y, dtype=float)
        if ysim.shape[0] != y.shape[0]:
            raise RuntimeError("npde: simulated response length does not match observed data")
        cens = np.asarray(self.res.data.cens)
        if cens.size != y.size:
            cens = np.zeros_like(y, dtype=int)
        y_obs = y.copy()
        # res.data.y stores the limit/value used by the engine for censored observations
        # (cens: 1 left-censored <= limit, -1 right-censored >= limit, 0 observed)
        pdv = np.full(y.shape[0], np.nan, dtype=float)
        eps = 0.5 / float(nsim)

        if decorrelate:
            n = y.shape[0]
            subj = None
            # preferred: zero-based subject index per observation
            info_subj = getattr(self.info, "subj", None)
            if info_subj is not None:
                subj = np.asarray(info_subj)
                if subj.shape[0] != n:
                    subj = None
            if subj is None:
                # fall back to data frame subject column (preserve core/row order)
                try:
                    id_col = getattr(self.info, "id_col", None)
                    frame = getattr(self.info, "frame", None)
                    if id_col is not None and frame is not None and id_col in frame.columns:
                        vals = frame[id_col].to_numpy()
                        if vals.shape[0] == n:
                            _, subj = np.unique(vals, return_inverse=True)
                except Exception:
                    subj = None
            if subj is None:
                # fall back to core data subject ids if aligned
                try:
                    core_id = None
                    data_obj = getattr(self.res, "data", None)
                    for attr in ("id", "subject_ids"):
                        if data_obj is not None and hasattr(data_obj, attr):
                            core_id = np.asarray(getattr(data_obj, attr))
                            break
                    if core_id is not None and core_id.shape[0] == n:
                        _, subj = np.unique(core_id, return_inverse=True)
                except Exception:
                    subj = None
            if subj is None or np.asarray(subj).shape[0] != n:
                subj = np.arange(n, dtype=int)
            subj = np.asarray(subj, dtype=int)
            for sid in np.unique(subj):
                idx = np.flatnonzero(subj == sid)
                Yi = np.asarray(ysim[idx, :], dtype=float)          # (ni, nsim)
                yi = np.asarray(y_obs[idx], dtype=float)
                ni = Yi.shape[0]
                if ni == 0:
                    continue
                mu = Yi.mean(axis=1)
                if ni == 1 or nsim <= 1:
                    Yt, yt = Yi, yi.reshape(-1)
                else:
                    Yc = Yi - mu[:, None]
                    S = (Yc @ Yc.T) / float(max(nsim - 1, 1))
                    # regularize for numerical stability / singular within-subject covariance
                    scale = np.trace(S) / float(ni) if np.isfinite(np.trace(S)) and np.trace(S) > 0 else 1.0
                    jitter = 1e-8 * scale
                    for _ in range(6):
                        try:
                            L = np.linalg.cholesky(S + jitter * np.eye(ni))
                            Yt = np.linalg.solve(L, Yc)
                            yt = np.linalg.solve(L, yi - mu)
                            break
                        except np.linalg.LinAlgError:
                            jitter *= 100.0
                    else:
                        # fallback: no decorrelation for this subject
                        Yt, yt = Yc, yi - mu
                with np.errstate(all="ignore"):
                    le = (Yt <= yt[:, None]).sum(axis=1)
                    pd_i = le / float(nsim)
                pdv[idx] = pd_i
        else:
            with np.errstate(all="ignore"):
                le = (ysim <= y_obs[:, None]).sum(axis=1)
                pd_i = le / float(nsim)
            pdv = pd_i

        pdv = np.asarray(pdv, dtype=float)
        # censored observations: randomized prediction discrepancies
        # cens==-1: left-censored (y <= limit) => U(0, P(Y_sim <= limit))
        # cens==1: right-censored (y >= limit) => U(P(Y_sim <= limit), 1)
        if cens.size == y.size and np.any(cens != 0):
            rng = np.random.default_rng((int(seed) & 0xFFFFFFFF) ^ 0x6E706465)
            for i, c in enumerate(np.asarray(cens, dtype=int)):
                if not np.isfinite(pdv[i]):
                    continue
                u = rng.random()
                if c == -1:
                    pdv[i] = u * pdv[i]
                elif c == 1:
                    pdv[i] = pdv[i] + u * (1.0 - pdv[i])
        pdv = np.clip(pdv, eps, 1.0 - eps)

        try:
            from scipy.stats import norm
            q = norm.ppf(pdv)
        except Exception:
            from statistics import NormalDist
            nd = NormalDist()
            q = np.array([nd.inv_cdf(float(v)) if np.isfinite(v) else np.nan for v in pdv], dtype=float)
        return pd.Series(q, index=self.info.row_index, name="npde")

    def to_frame(self) -> pd.DataFrame:
        """Data used + predictions and residuals per observation."""
        df = self.info.frame.copy()
        if self.model.family == "gaussian":
            df["pred_pop"] = self.fitted("population")
            df["pred_ind"] = self.fitted("individual")
            df["res_pop"] = self.residuals("population")
            df["res_ind"] = self.residuals("individual")
            df["pres_ind"] = self.residuals("individual", "pearson")
        return df

    # ------------------------------------------------------------------ prediction with new data
    def predict(self, newdata=None, level: str = "population", events=None, evid: Optional[str] = None) -> pd.Series:
        """Predict with ``newdata`` (same time/predictor/covariate columns; ``y`` is not needed).
        ``level='population'``: typical parameters (with the covariate effect); ``'individual'``: uses the subjects'
        EBEs for the same id. Without ``newdata`` returns ``fitted(level)``."""
        self._check_gaussian("predict")
        if newdata is None:
            return self.fitted("individual" if level == "individual" else "population")
        if level not in ("population", "individual"):
            raise ValueError("level: 'population' or 'individual'")
        info, p = self.info, len(self.params)
        df = as_frame(newdata)
        idc = info.id_col
        if idc not in df.columns:
            df = df.assign(**{idc: 0})
        if info.time_synthetic and info.time_col not in df.columns:      # static model: observation index
            df = df.assign(**{info.time_col: df.groupby(idc, sort=False).cumcount().astype(float)})
        for c in info.x_names + info.cov_names + info.cat_names:
            if c not in df.columns:
                raise KeyError(f"predict: missing column {c!r} in newdata")
        ev_df = None
        if evid is not None:
            e = df[evid].fillna(0).astype(int); ev_df = df[e == 1]; df = df[e == 0]
        if events is not None:
            extra = as_frame(events); ev_df = extra if ev_df is None else pd.concat([ev_df, extra], ignore_index=True)
        ec = info.event_cols
        yt_col = None
        coef = np.asarray(self.res.coef, float)
        names = list(self.res.coef_names)
        out = pd.Series(np.nan, index=df.index, name="pred")
        fe = np.asarray(self.res.psi_pop)
        ebe_psi = None
        if level == "individual":
            ebe_psi = self._stack(self.res.psi_ebe)
            id_pos = {v: k for k, v in enumerate(info.ids.tolist())}
        tr = self.model.transform
        cm = self.res.model
        for sid, g in df.groupby(idc, sort=False):
            X = np.column_stack([g[c].to_numpy(float) for c in info.x_names] + [np.zeros(len(g))])
            cov = np.array([float(g[c].iloc[0]) for c in info.cov_names])
            if level == "individual":
                if sid not in id_pos:
                    raise KeyError(f"predict(level='individual'): id {sid!r} was not in the fit")
                psi = ebe_psi[id_pos[sid]]
            else:
                phi = coef[:p].copy()
                for t, nm in enumerate(names[p:]):
                    pn, cv = nm.split("~", 1)
                    j = self.params.index(pn)
                    if cv in info.cov_names:
                        phi[j] += coef[p + t] * float(g[cv].iloc[0])
                    else:
                        base, _, lev = cv.partition("=")
                        phi[j] += coef[p + t] * float(str(g[base].iloc[0]) == lev)
                psi = np.array([apply_transform(phi[j], tr[j]) for j in range(p)], float)
            doses = None
            if ev_df is not None:
                e = ev_df[ev_df[idc] == sid]
                if len(e):
                    amt = e[ec["amt"]].to_numpy(float)
                    n = len(e)
                    cmt = e[ec["cmt"]].to_numpy(float) if ec["cmt"] in e else np.ones(n)
                    rate = e[ec["rate"]].to_numpy(float) if ec["rate"] in e else np.zeros(n)
                    addl = e[ec["addl"]].to_numpy(int) if ec["addl"] in e else np.zeros(n, int)
                    ii = e[ec["ii"]].to_numpy(float) if ec["ii"] in e else np.zeros(n)
                    ss = e[ec["ss"]].to_numpy(float) if ec["ss"] in e else np.zeros(n)
                    rows = []
                    for k in range(n):
                        dur = amt[k] / rate[k] if rate[k] > 0 else 0.0
                        for a in range(int(addl[k]) + 1):
                            rows.append([e[info.time_col].iloc[k] + a * ii[k], cmt[k], amt[k], dur, ii[k] if (a == 0 and ss[k]) else 0.0,
                                         ss[k] if a == 0 else 0.0, -1.0 if rate[k] == -1 else (-2.0 if rate[k] == -2 else 0.0)])
                    doses = np.asarray(sorted(rows, key=lambda r: r[0]), float)
            f = cm.eval(np.ascontiguousarray(psi, float), np.ascontiguousarray(X), np.ascontiguousarray(cov, float), doses)
            out.loc[g.index] = f
        return out

    # ------------------------------------------------------------------ simulation
    def simulate(self, nsim: int = 1, seed: int = 1, *, new_random_effects: bool = True,
                 lloq: Optional[float] = None, uloq: Optional[float] = None) -> pd.DataFrame:
        """Simulate ``nsim`` response sets with the estimated parameters (predictive check).
        Returns a DataFrame aligned with the data and columns ``sim_1 ... sim_n`` (and ``cens_k`` if there is lloq/uloq)."""
        nan = float("nan")
        sims = _core.simulate(self.res, int(seed) & 0xFFFFFFFF, int(nsim), nan if lloq is None else float(lloq),
                              nan if uloq is None else float(uloq), bool(new_random_effects),
                              self.model.is_compiled)
        cols = {}
        for k, d in enumerate(sims, 1):
            cols[f"sim_{k}"] = np.asarray(d.y)
            if lloq is not None or uloq is not None:
                cols[f"cens_{k}"] = np.asarray(d.cens)
        return pd.DataFrame(cols, index=self.info.row_index)

    def simulate_events(self, tmax: float, repeated: bool = False, nsim: int = 1, seed: int = 1,
                        new_random_effects: bool = True) -> pd.DataFrame:
        """(survival families) simulate time-to-event data; returns a long DataFrame with ``rep``, id, time, y (1 = event, 0 = censoring)."""
        sims = _core.simulate_events(self.res, float(tmax), bool(repeated), int(seed) & 0xFFFFFFFF, int(nsim),
                                     bool(new_random_effects))
        frames = []
        for k, d in enumerate(sims, 1):
            df = pd.DataFrame({"rep": k, self.info.id_col: self.info.ids[np.asarray(d.id) - 1],
                               self.info.time_col: np.asarray(d.time), self.info.y_col: np.asarray(d.y)})
            for nme in d.predictor_names:
                df[nme] = d.predictor(nme)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ tables / summary
    def _fixed_table(self) -> pd.DataFrame:
        """Fixed-effect table with estimate, SE, RSE, z/p values and natural-scale intercepts."""
        est = np.asarray(self.res.coef, float)
        n = len(est)
        se = np.asarray(self.res.se_coef, float) if self.res.se_computed and len(self.res.se_coef) >= n else np.full(n, np.nan)
        z = est / se
        df = pd.DataFrame({"Estimate": est, "SE": se, "RSE(%)": 100 * np.abs(se / est),
                           "z": z, "p": [_norm_sf(v) for v in z]}, index=list(self.res.coef_names))
        nat = np.full(n, np.nan)
        nat[:len(self.params)] = np.asarray(self.res.psi_pop)
        df["natural"] = nat
        return df

    def _random_table(self) -> pd.DataFrame:
        """Random-effect standard deviations, their SEs and the CV on the natural scale."""
        rows = []
        sd, se = np.asarray(self.res.sd), np.asarray(self.res.se_sd, float)
        for j, nme in enumerate(self.params):
            if nme not in self.model.random:
                continue
            cv = 100 * math.sqrt(math.expm1(sd[j] ** 2)) if self.model.transform[j] == "log" else float("nan")
            rows.append({"param": nme, "sd": sd[j], "SE": se[j] if j < len(se) else float("nan"), "CV(%)": cv})
        return pd.DataFrame(rows).set_index("param") if rows else pd.DataFrame()

    def _corr_table(self):
        """Lower-triangular random-effect correlation matrix, or ``None`` if there is nothing to show."""
        idx = [j for j, n in enumerate(self.params) if n in self.model.random]
        if len(idx) < 2:
            return None
        C = np.asarray(self.res.corr, float)
        if C.size == 0 or np.allclose(C[np.ix_(idx, idx)] - np.eye(len(idx)), 0):
            return None
        nm = [self.params[j] for j in idx]
        M = pd.DataFrame(C[np.ix_(idx, idx)], index=nm, columns=nm)
        return M.where(np.tril(np.ones(M.shape, bool), -1))

    def _cluster_table(self) -> pd.DataFrame:
        """Cluster-level random-effect standard deviations and their SEs."""
        sd, se = np.asarray(self.res.sd_cluster), np.asarray(self.res.se_sd_cluster, float)
        return pd.DataFrame({"sd": sd, "SE": se if len(se) == len(sd) else np.nan}, index=self.params)

    def _error_table(self) -> pd.DataFrame:
        """Residual-error parameters (and extra parameters) with their standard errors."""
        rows, flat = [], 0
        se = list(self.res.se_err)
        for r, ep in enumerate(self.res.error_pars):
            nm = self.model.errors[min(r, len(self.model.errors) - 1)]
            for pn in ERROR_PARS[nm]:
                rows.append({"response": self._resp_label(r), "model": nm, "param": pn,
                             "estimate": getattr(ep, pn), "SE": se[flat] if flat < len(se) else float("nan")})
                flat += 1
        for k, v in enumerate(np.asarray(self.res.extra)):
            rows.append({"response": "-", "model": "extra", "param": f"extra{k}", "estimate": v,
                         "SE": float(np.asarray(self.res.se_extra)[k]) if len(self.res.se_extra) > k else float("nan")})
        return pd.DataFrame(rows).set_index(["response", "param"]) if rows else pd.DataFrame()

    def _stats(self) -> dict:
        """Scalar fit statistics used by the summary and comparison tables."""
        return {"loglik": self.loglik, "loglik_method": self.res.loglik_method if self.res.loglik_computed else "",
                "loglik_lin": float(self.res.loglik_lin), "aic": self.aic, "bic": self.bic, "df": self.df,
                "n_obs": self.n_obs, "n_subjects": self.n_subjects,
                "se_method": self.res.se_method if self.res.se_computed else ""}

    def summary(
        self,
        *,
        alpha: float = 0.05,
        stars: bool | dict = False,
        diagnostics: bool = True,
        title: str = "Nonlinear Mixed-Effects Model Fit by SAEM",
    ) -> SaemSummary:
        """Summarize the fitted model with statcraft's shared summary engine.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the confidence intervals, in (0, 1).
        stars : bool or dict, default False
            Add significance stars next to the coefficients.
        diagnostics : bool, default True
            Include the secondary algorithm/control block (iterations, drift, MCMC settings, EBE checks).
        title : str
            Report title.

        Returns
        -------
        SaemSummary
            A :class:`statcraft.summary.Summary` with one table per block, structured pandas data in
            ``.tables[i].data``, and ``save_html`` / ``save_latex`` / ``save_excel`` exporters.
        """
        fixed = self._fixed_table()
        random = self._random_table()
        corr = self._corr_table()
        error = self._error_table()

        blocks = []
        if len(fixed):
            blocks.append(
                _block(
                    "Fixed effects (estimation scale)",
                    fixed.index,
                    fixed["Estimate"],
                    fixed["SE"],
                )
            )
        if len(random):
            se = random["SE"].to_numpy(float) if "SE" in random.columns else None
            blocks.append(_block("Random effects (SD)", random.index, random["sd"], se))
        if corr is not None and len(corr):
            cnames, cvals = [], []
            cols = list(corr.columns)
            for i, row in enumerate(corr.index):
                for col in cols[:i]:
                    val = corr.loc[row, col]
                    if np.isfinite(val):
                        cnames.append(f"{row} ~ {col}")
                        cvals.append(float(val))
            if cnames:
                blocks.append(_block("Random-effects correlations", cnames, cvals))
        if self.res.hierarchical:
            cl = self._cluster_table()
            if len(cl):
                se = (
                    cl["SE"].to_numpy(float)
                    if "SE" in cl.columns and cl["SE"].notna().any()
                    else None
                )
                blocks.append(_block("Cluster random effects (SD)", cl.index, cl["sd"], se))
        if len(error):
            enames = [f"{a}: {b}" for a, b in error.index]
            se = error["SE"].to_numpy(float) if "SE" in error.columns else None
            blocks.append(_block("Residual error", enames, error["estimate"], se))

        left = [
            ("Model:", self.model.name),
            ("Family:", self.model.family),
            ("Error model:", ", ".join(self.model.errors)),
            ("No. Observations:", str(self.n_obs)),
            ("No. Subjects:", str(self.n_subjects)),
        ]
        if self.res.hierarchical:
            left.append(("No. Clusters:", str(int(self.res.n_clusters))))
        left += [("Parameters:", str(len(self.params))), ("Df:", str(self.df))]
        right = [
            ("Log-Likelihood:", _fmt_num(self.loglik, 8)),
            ("AIC:", _fmt_num(self.aic, 8)),
            ("BIC:", _fmt_num(self.bic, 8)),
            ("LogLik method:", _method_label(self.res.loglik_method, _LL_METHOD_LABELS)
             if self.res.loglik_computed else "n/a"),
            ("SE method:", _method_label(self.res.se_method, _SE_METHOD_LABELS)
             if self.res.se_computed else "n/a"),
        ]

        # Algorithm/control details live in the secondary diagnostics block, keeping the header
        # focused on the model and the fit criteria.
        diag_left = diag_right = None
        if diagnostics:
            c = self.res.convergence
            diag_left = [
                ("Iterations:", str(int(c.iterations))),
                ("Max drift (last 25%):", _fmt_num(c.max_drift, 4)),
                ("Stopped early:", "Yes" if c.stopped_early else "No"),
                ("Algorithm:", str(self.control.mstep)),
            ]
            diag_right = [
                ("Burn-in:", str(self.control.n_burn)),
                ("MCMC sweeps:", str(self.control.n_mcmc)),
                ("Chains:", str(self.control.n_chains)),
                ("Seed:", str(self.control.seed)),
                ("Threads:", str(self.control.n_threads)),
            ]
            if getattr(self.res, "cond_computed", False):
                diag_left.append(("EBE ESS (min):", _fmt_num(self.res.cond_ess_min, 4)))
                diag_right.append(("EBE R-hat (max):", _fmt_num(self.res.cond_rhat_max, 4)))

        notes = list(self.warnings)
        notes.append(
            "Fixed effects are reported on the estimation (phi) scale; use fixef(scale='natural') "
            "or pop_params() for the natural scale."
        )
        if not self.res.se_computed:
            notes.append("Standard errors were not computed (control: se_method='none').")
        if not self.res.loglik_computed:
            notes.append("The marginal log-likelihood was not computed (control: ll_method='none').")

        extra_lines = []
        natural = self.pop_params()
        if len(natural):
            extra_lines.append(
                (
                    "Population (natural scale)",
                    ", ".join(f"{k}={_fmt_num(v, 4)}" for k, v in natural.items()),
                )
            )

        built = build_summary(
            title=title,
            left=left,
            right=right,
            blocks=blocks,
            diag_left=diag_left,
            diag_right=diag_right,
            dist="normal",
            alpha=alpha,
            notes=notes,
            extra_lines=extra_lines or None,
            stars=stars,
        )
        tables = {
            "fixed": fixed,
            "random": random,
            "corr": corr,
            "error": error,
            "stats": self._stats(),
        }
        return SaemSummary(built, self, tables)

    def __repr__(self):
        return (f"<SaemFit {self.model.name}: {self.n_subjects} subjects, {self.n_obs} obs, "
                f"logLik={self.loglik:.4g}>")

    # ------------------------------------------------------------------ model comparison
    def anova(self, *others: "SaemFit") -> pd.DataFrame:
        """Comparison table (AIC, BIC, logLik) and likelihood-ratio test between nested models
        ordered by number of parameters."""
        fits = sorted([self, *others], key=lambda f: f.df)
        rows, prev = [], None
        for k, f in enumerate(fits, 1):
            r = {"model": f.model.name + (f"#{k}" if len({g.model.name for g in fits}) < len(fits) else ""),
                 "df": f.df, "AIC": f.aic, "BIC": f.bic, "logLik": f.loglik}
            if prev is not None:
                lr = 2.0 * (f.loglik - prev.loglik)
                r.update({"LR": lr, "ddf": f.df - prev.df, "p": _chi2_sf(lr, f.df - prev.df) if f.df > prev.df else np.nan})
            rows.append(r); prev = f
        return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# fitting engine (used by SAEM.fit())
# ---------------------------------------------------------------------------
def _vec_by_name(spec, names: Sequence[str], what: str, *, allow_partial=False, fill=None):
    """Convert a mapping or sequence of initial values to a vector aligned with ``names``.

    ``spec`` may be a mapping ``{name: value}`` (then ``allow_partial`` controls whether every
    name is required and ``fill`` is used for the missing ones) or a plain sequence whose length
    must match ``names``. ``what`` names the option in error messages.
    """
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        bad = set(spec) - set(names)
        if bad:
            raise ValueError(f"{what}: unknown names {sorted(bad)} (valid: {list(names)})")
        miss = [n for n in names if n not in spec]
        if miss and not allow_partial:
            raise ValueError(f"{what}: missing {miss}. Specify all parameters (or none for automatic start)")
        return np.array([float(spec[n]) if n in spec else fill for n in names], float)
    arr = np.asarray(spec, float).reshape(-1)
    if len(arr) != len(names):
        raise ValueError(f"{what}: expected {len(names)} values, received {len(arr)}")
    return arr


def _run_saem(model: NLModel, data, *, id: Optional[str] = None, time: Optional[str] = None,
              response: Optional[str] = None,
              ytype: Optional[str] = None, cens: Optional[str] = None, limit: Optional[str] = None,
              group: Optional[str] = None, events=None, evid: Optional[str] = None,
              event_cols: Optional[Mapping[str, str]] = None, predictors: Optional[Sequence[str]] = None,
              reference: Optional[Mapping[str, str]] = None, na_action: str = "omit",
              start=None, start_sd=None, start_error=None, start_cov: Optional[Mapping[str, float]] = None,
              fix: Optional[Sequence[str]] = None, fix_sd: Optional[Sequence[str]] = None,
              fix_error: Optional[Sequence[str]] = None,
              control: Optional[SaemControl] = None) -> SaemFit:
    """Run the fit with an already-built ``NLModel`` and ``SaemControl``.

    It is the internal engine of :meth:`statcraft.saem.SAEM.fit`; it is not used directly.
    """
    control = control or SaemControl()
    cdata, info = prepare(model, data, id=id, time=time, response=response, ytype=ytype, cens=cens, limit=limit, group=group,
                          events=events, evid=evid, event_cols=event_cols, predictors=predictors,
                          reference=reference, na_action=na_action)
    if info.id_synthetic:
        warnings.warn(
            "No subject column (id): each row is treated as a distinct subject. With one observation per subject "
            "the random effects are not identifiable (their variance is confounded with the residual error) and the engine "
            "does not yet support models without random effects: the estimates are NOT equivalent to those of a nonlinear "
            "regression (the fixed effects may differ appreciably) and the random SDs and the residual error cannot be "
            "interpreted separately. If the data have groups, specify id=<column>.", SaemWarning, stacklevel=3)
    n_resp = len(info.ytype_labels) if info.ytype_labels is not None else 1
    if len(model.errors) not in (1, n_resp):
        raise ValueError(f"error: {len(model.errors)} error models for {n_resp} responses (ytype)")
    cmodel = model._to_core(info.x_names, info.cov_names, info.cat_names)
    if group and control.ll_method == "auto":
        # the nested GH quadrature of the hierarchical model is prohibitive (minutes): joint IS per cluster
        control = control.copy(ll_method="importance_sampling")
    c = control._to_core()
    names = model.params

    # ---- initial values ----
    b0 = _vec_by_name(start, names, "start")
    if b0 is not None:
        c.beta0 = np.ascontiguousarray(b0)
    sd0 = _vec_by_name(start_sd, names, "start_sd", allow_partial=True, fill=1.0)
    if sd0 is not None:
        c.sd0 = np.ascontiguousarray(sd0)
    if start_cov:
        if info.cat_names:
            raise NotImplementedError("start_cov with categorical covariates is not supported")
        tnames = [f"{p}~{cv}" for p, cv in model.covariates]
        bad = set(start_cov) - set(tnames)
        if bad:
            raise ValueError(f"start_cov: unknown terms {sorted(bad)} (valid: {tnames})")
        c.beta_cov0 = np.array([float(start_cov.get(t, 0.0)) for t in tnames], float)
    if start_error is not None:
        lst = [start_error] if not isinstance(start_error, (list, tuple)) else list(start_error)
        eps = []
        for r, e in enumerate(lst):
            nm = model.errors[min(r, len(model.errors) - 1)]
            if not isinstance(e, Mapping):
                e = {ERROR_PARS[nm][0]: float(e)}
            eps.append(_core.ErrorPars(float(e.get("a", 0.0)), float(e.get("b", 0.0)), float(e.get("c", 1.0))))
        c.error_init = eps
    # ---- fixed parameters ----
    if fix or fix_sd or fix_error:
        if (fix or fix_sd) and b0 is None:
            raise ValueError("fix/fix_sd require start (the value they are fixed at)")
        if fix:
            bad = set(fix) - set(names)
            if bad:
                raise ValueError(f"fix: unknown parameters {sorted(bad)}")
            c.fixed_beta = [1 if n in set(fix) else 0 for n in names]
        if fix_sd:
            bad = set(fix_sd) - set(names)
            if bad:
                raise ValueError(f"fix_sd: unknown parameters {sorted(bad)}")
            c.fixed_sd = [1 if n in set(fix_sd) else 0 for n in names]
        if fix_error:
            if len(model.errors) != 1:
                raise NotImplementedError("fix_error with several responses: use the low-level control")
            ep = ERROR_PARS[model.errors[0]]
            bad = set(fix_error) - set(ep)
            if bad:
                raise ValueError(f"fix_error: {sorted(bad)} do not belong to {ep}")
            c.fixed_error = [1 if n in set(fix_error) else 0 for n in ep]

    res = _core.fit(cdata, cmodel, c, model.is_compiled)
    return SaemFit(res, model, info, control)


def __getattr__(name):
    """Compatibility: ``from statcraft.saem.fit import saem`` still works."""
    if name == "saem":
        from .api import saem
        return saem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
