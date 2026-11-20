# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
glm.py -- GLM(formula, data, family) on top of _glmfit, with R-style
formulas (via formulaic) and all the control options of the IRLS engine.

Usage:
    from statcraft import glm, Gaussian, Binomial, Poisson, ...

    fit = glm("y ~ x1 + x2", data, family=Binomial(link="logit"))
    fit.summary()              # statsmodels-style text/HTML summary
    fit.coefficients           # -> fit.beta
    fit.fittedvalues

GLM.fit() dispatches on `family.family_type` (set by the Family classes in
family.py -- see that module's docstring for the five family groups) to one
of several internal engines, but the public surface is identical regardless
of which engine actually runs:
    GLM(X, y, family=...)          -- construct from raw arrays
    GLM.from_formula(formula, data, family=...)  -- construct from an
                                       R-style formula (this is what the
                                       module-level `glm(...)` shortcut
                                       calls)
    .fit(...)                      -- returns self, sets .beta/.se/.vcov/
                                       .deviance/.log_likelihood/... (which
                                       of these end up populated depends on
                                       family_type; see family.py)
    .summary(...)                  -- statsmodels-style results table
                                       (.as_text()/.as_html()/repr/
                                       _repr_html_)
    .predict(...), .fittedvalues, .resid_*, .aic, .bic, ...

Zero-inflated/hurdle/beta-regression families take a two-part formula
"count_formula | zero_formula" (R's Formula package convention, as used by
pscl::zeroinfl/hurdle) -- see _split_zeroinfl_formula(). GeneralizedGamma
instead takes separate mu/sigma/Q formulas via GLM.from_formula_gengamma()
(exposed as generalized_gamma(formula, data, sigma_formula=..., Q_formula=...)),
since it has three submodels rather than one count + one auxiliary model.

Module-level shortcuts (gaussian, binomial, poisson, ..., zero_inflated_poisson,
hurdle_gamma, generalized_gamma, ...) are thin wrappers around
GLM.from_formula() that build the right Family object for you; see __all__
at the bottom of this file for the full list.
"""

import inspect
import re
from types import SimpleNamespace
from typing import Any

import formulaic
import numpy as np
import pandas as pd
import scipy.linalg

from .. import _glmfit
from ..plots.methods import PlottingMixin
from ..summary import Summary, build_summary
from . import glm_diag as _gdiag
from . import mle_diag as _mdiag
from . import mle_dist as _dist
from .family import (
    Beta,
    Binomial,
    CustomFamily,
    Exponential,
    Family,
    FamilyCode,
    Gamma,
    Gaussian,
    GeneralizedGamma,
    Gumbel,
    HurdleBeta,
    HurdleGamma,
    HurdleGeometric,
    HurdleLogNormal,
    HurdleNegativeBinomial,
    HurdlePoisson,
    InverseGamma,
    InverseGaussian,
    LogisticDistribution,
    LogLogistic,
    LogNormal,
    MleFamily,
    Multinomial,
    NegativeBinomial,
    Ordinal,
    Pareto,
    Poisson,
    QuasiBinomial,
    QuasiCustom,
    QuasiFamily,
    QuasiGamma,
    QuasiGaussian,
    QuasiInverseGaussian,
    QuasiPoisson,
    Simplex,
    SkewNormal,
    SkewT,
    StudentT,
    TruncatedGaussian,
    TruncatedGeometric,
    TruncatedNegativeBinomial,
    TruncatedPoisson,
    Tweedie,
    ValueInflatedNegativeBinomial,
    ValueInflatedPoisson,
    Weibull,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


def _qnorm_acklam(p):
    """Standard normal quantile, Acklam's rational approximation."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1.0 - 0.02425

    low = p < p_low
    high = p > p_high
    mid = ~low & ~high

    q = np.sqrt(-2.0 * np.log(np.where(low, p, 1.0)))
    num_low = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    den_low = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    out[low] = (num_low / den_low)[low]

    qm = p[mid] - 0.5
    r = qm * qm
    out[mid] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * qm
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )

    qh = np.sqrt(-2.0 * np.log(np.where(high, 1.0 - p, 1.0)))
    num_high = ((((c[0] * qh + c[1]) * qh + c[2]) * qh + c[3]) * qh + c[4]) * qh + c[5]
    den_high = (((d[0] * qh + d[1]) * qh + d[2]) * qh + d[3]) * qh + 1.0
    out[high] = (-num_high / den_high)[high]
    return out


_FORWARD_LINK = {
    "identity": lambda mu: mu,
    "log": lambda mu: np.log(mu),
    "inverse": lambda mu: 1.0 / mu,
    "logit": lambda mu: np.log(mu / (1.0 - mu)),
    "probit": _qnorm_acklam,
    "cloglog": lambda mu: np.log(-np.log(1.0 - mu)),
    "sqrt": lambda mu: np.sqrt(mu),
    "invmu2": lambda mu: 1.0 / (mu**2),
}

_LINK_TRANSFORMS = {
    "identity": {
        "inv": lambda x: x,
        "deriv": lambda x: np.ones_like(x) if isinstance(x, np.ndarray) else 1.0,
        "label": "identity",
    },
    "log": {"inv": lambda x: np.exp(x), "deriv": lambda x: np.exp(x), "label": "exp(log)"},
    "inverse": {"inv": lambda x: 1.0 / x, "deriv": lambda x: -1.0 / (x**2), "label": "1/x"},
    "logit": {
        "inv": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "deriv": lambda x: np.exp(-x) / ((1.0 + np.exp(-x)) ** 2),
        "label": "logistic",
    },
    "sqrt": {"inv": lambda x: x**2, "deriv": lambda x: 2.0 * x, "label": "squared"},
    "invmu2": {
        "inv": lambda x: 1.0 / np.sqrt(x),
        "deriv": lambda x: -0.5 * (x ** (-1.5)),
        "label": "1/sqrt(x)",
    },
}

_METHOD_SOLVE = {
    "qr": _glmfit.TYPE_QR,
    "colpiv_qr": _glmfit.TYPE_COLPIV_QR,
    "colpivqr": _glmfit.TYPE_COLPIV_QR,
    "fullpiv_qr": _glmfit.TYPE_FULLPIV_QR,
    "fullpivqr": _glmfit.TYPE_FULLPIV_QR,
    "cholesky": _glmfit.TYPE_CHOLESKY,
    "llt": _glmfit.TYPE_CHOLESKY,
    "chol": _glmfit.TYPE_CHOLESKY,
    "ldlt": _glmfit.TYPE_LDLT,
    "svd": _glmfit.TYPE_SVD,
}

_Minimizer = getattr(_glmfit, "Minimizer", None)
_MINIMIZER_MAP = {
    "lbfgsb": _Minimizer.LBFGSB if _Minimizer is not None else None,
    "L-BFGS-B": _Minimizer.LBFGSB if _Minimizer is not None else None,
    "l-bfgs-b": _Minimizer.LBFGSB if _Minimizer is not None else None,
    "bfgs": _Minimizer.BFGS if _Minimizer is not None else None,
    "BFGS": _Minimizer.BFGS if _Minimizer is not None else None,
    "neldermead": _Minimizer.NelderMead if _Minimizer is not None else None,
    "nelder-mead": _Minimizer.NelderMead if _Minimizer is not None else None,
    "nelder_mead": _Minimizer.NelderMead if _Minimizer is not None else None,
    "Nelder-Mead": _Minimizer.NelderMead if _Minimizer is not None else None,
}


def _resolve_col(spec, data, n, name):
    if spec is None:
        return None
    if isinstance(spec, str):
        if data is None or spec not in data:
            raise ValueError(f"GLM: {name}='{spec}' is not a column of `data`.")
        arr = np.asarray(data[spec], dtype=float)
    else:
        arr = np.asarray(spec, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"GLM: {name} has size {arr.shape[0]}, expected {n}.")
    return arr


def _resolve_array(spec, n, name):
    if spec is None:
        return None
    arr = np.asarray(spec, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"GLM: {name} has size {arr.shape[0]}, expected {n}.")
    return arr


def _split_zeroinfl_formula(formula):
    if "|" not in formula:
        return formula, None
    parts = formula.split("|")
    if len(parts) > 2:
        raise ValueError(
            f"GLM: formula has {len(parts) - 1} '|' separators, only one is supported: {formula!r}."
        )
    count_formula, zero_rhs = parts[0].strip(), parts[1].strip()
    if "~" in zero_rhs:
        raise ValueError(f"GLM: the part after '|' must be a bare RHS (no '~'), got {zero_rhs!r}.")
    zero_formula = f"~ {zero_rhs}"
    return count_formula, zero_formula


def _check_finite(arr, name):
    bad = ~np.isfinite(arr)
    if not bad.any():
        return
    n_bad = int(bad.sum())
    if arr.ndim == 1:
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"GLM: {name} has {n_bad} non-finite value(s) (NaN/Inf); "
            f"the first one is at position {first}. Check or remove those "
            f"observations before fitting the model."
        )
    rows_bad = np.flatnonzero(bad.any(axis=1))
    cols_bad = np.flatnonzero(bad.any(axis=0))
    raise ValueError(
        f"GLM: {name} has {n_bad} non-finite value(s) (NaN/Inf) across "
        f"{rows_bad.size} row(s) (first: {int(rows_bad[0])}) and "
        f"{cols_bad.size} column(s) (first: {int(cols_bad[0])}). Check "
        f"or remove those observations/columns before fitting the model."
    )


def _detect_rank_deficiency(X, feature_names=None):
    n, p = X.shape
    if p == 0:
        return np.arange(0, dtype=int), np.arange(0, dtype=int)

    _, R, piv = scipy.linalg.qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))

    if diag.size == 0 or diag[0] <= 0.0:
        raise ValueError(
            "GLM: X has rank 0 (all columns are constant zero or "
            "degenerate); a model cannot be fit."
        )

    threshold = diag[0] * np.finfo(float).eps * max(n, p)
    rank = int(np.sum(diag > threshold))

    active = np.sort(piv[:rank])
    aliased = np.sort(piv[rank:])

    if aliased.size > 0:
        names = feature_names if feature_names is not None else [f"x{j}" for j in range(p)]
        aliased_names = ", ".join(f"'{names[j]}'" for j in aliased)
        import warnings

        warnings.warn(
            f"GLM: X is rank-deficient (rank {rank} of {p} columns). "
            f"The following columns are a linear combination of the "
            f"others and are excluded from the fit (their coefficients "
            f"are reported as NaN): {aliased_names}.",
            stacklevel=3,
        )

    return active.astype(int), aliased.astype(int)


def _clean_level_label(label):
    """Strip formulaic's wrapper: 'apply[unlikely]' -> 'unlikely'.
    If the label does not have that format, it is returned as is (as str)."""
    label = str(label)
    m = re.match(r"^[^\[]*\[(.*)\]$", label, flags=re.S)
    return m.group(1) if m else label


def _ordinal_level_names(y_categories, n_thresholds):
    """Names of the K = n_thresholds + 1 levels of an ordinal response, or None if
    there is no consistent information (the caller then falls back to cut0|cut1, ...)."""
    if not y_categories or len(y_categories) != n_thresholds + 1:
        return None
    return [_clean_level_label(c) for c in y_categories]


def _ordinal_threshold_names(y_categories, n_thresholds):
    """Threshold labels: 'unlikely|somewhat likely', ... or the generic scheme."""
    levels = _ordinal_level_names(y_categories, n_thresholds)
    if levels is None:
        return [f"cut{k}|cut{k + 1}" for k in range(n_thresholds)]
    return [f"{levels[k]}|{levels[k + 1]}" for k in range(n_thresholds)]


def _response_levels_from_data(formula, data):
    """Levels of the response taken from `data` when the response was NOT expanded to dummies
    (e.g. coded 1,2,3 or with a dtype formulaic treats as numeric).
    Returns (list_of_levels | None, ordered: bool | None)."""
    try:
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data:
            return None, None
        col = data[lhs]
        dtype = getattr(col, "dtype", None)
        if hasattr(dtype, "categories"):
            return [str(c) for c in dtype.categories], bool(getattr(dtype, "ordered", False))
        vals = np.unique(np.asarray(col)[~_isnull(col)])
        return [str(v) for v in vals.tolist()], None
    except Exception:
        return None, None


# Common ordinal scales (from lowest to highest). Used to automatically order a
# text response when all its levels belong to one of these scales.
_ORDINAL_SCALES = [
    [
        "very unlikely",
        "unlikely",
        "somewhat unlikely",
        "neither",
        "neutral",
        "undecided",
        "somewhat likely",
        "likely",
        "very likely",
    ],
    [
        "strongly disagree",
        "disagree",
        "somewhat disagree",
        "neither",
        "neutral",
        "undecided",
        "somewhat agree",
        "agree",
        "strongly agree",
    ],
    [
        "very low",
        "low",
        "lower",
        "medium low",
        "medium",
        "moderate",
        "mid",
        "middle",
        "average",
        "medium high",
        "high",
        "higher",
        "very high",
    ],
    [
        "never",
        "almost never",
        "rarely",
        "seldom",
        "occasionally",
        "sometimes",
        "often",
        "frequently",
        "usually",
        "very often",
        "almost always",
        "always",
    ],
    [
        "very poor",
        "poor",
        "bad",
        "fair",
        "average",
        "ok",
        "good",
        "very good",
        "great",
        "excellent",
    ],
    ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    ["none", "no", "minimal", "mild", "slight", "moderate", "severe", "extreme"],
    ["tiny", "small", "medium", "large", "huge"],
    ["bajo", "baja", "medio", "media", "alto", "alta"],
    ["nunca", "casi nunca", "raramente", "a veces", "frecuentemente", "casi siempre", "siempre"],
    ["muy improbable", "improbable", "poco probable", "probable", "muy probable"],
    ["muy malo", "malo", "regular", "bueno", "muy bueno", "excelente"],
    ["ninguno", "leve", "moderado", "grave", "severo"],
]


def _norm_level(x):
    return re.sub(r"[\s_\-]+", " ", str(x).strip().lower())


def _order_ordinal_levels(labels, user_levels=None, data_levels=None, data_ordered=False):
    """Automatically orders the levels of an ordinal response.

    labels       : clean labels in the current order (alphabetical, formulaic's).
    user_levels  : order given explicitly (`y_levels=`); overrides everything else.
    data_levels  : categories of the column if it is a pandas Categorical.
    data_ordered : True if that Categorical has ordered=True.

    Returns (perm, source): `perm` is the list of indices such that [labels[i] for i in perm]
    is in increasing ordinal order; `source` indicates how the order was decided:
    'user' | 'data' | 'numeric' | 'scale' | 'fallback'. With 'fallback' there is no way to
    know the order and the current one is kept (the caller warns).
    """
    labels = [str(l) for l in labels]
    pos = {l: i for i, l in enumerate(labels)}

    if user_levels is not None:
        ul = [str(l) for l in user_levels]
        missing = [l for l in labels if l not in ul]
        if missing:
            raise ValueError(f"y_levels does not include all response levels: {missing}")
        return [pos[l] for l in ul if l in pos], "user"

    if data_ordered and data_levels is not None:
        dl = [str(l) for l in data_levels]
        if all(l in dl for l in labels):
            return [pos[l] for l in dl if l in pos], "data"

    try:  # "1","2","10" -> numeric order, not alphabetical
        nums = [float(l) for l in labels]
        return sorted(range(len(labels)), key=lambda i: nums[i]), "numeric"
    except ValueError:
        pass

    norm = [_norm_level(l) for l in labels]
    for scale in _ORDINAL_SCALES:
        if all(n in scale for n in norm) and len(set(norm)) == len(norm):
            return sorted(range(len(labels)), key=lambda i: scale.index(norm[i])), "scale"

    return list(range(len(labels))), "fallback"


def _isnull(col):
    arr = np.asarray(col)
    try:
        return np.asarray(
            [v is None or (isinstance(v, float) and np.isnan(v)) for v in arr.tolist()]
        )
    except Exception:
        return np.zeros(arr.shape, dtype=bool)


def _gengamma_inverse_link(eta, link):
    """Inverse link for a GeneralizedGamma submodel, mirroring `apply_inverse_link` in
    include/glm/mle_utils.hpp: ``Log`` -> exp(eta), ``Inverse`` -> 1/eta, otherwise identity."""
    if link == "log":
        return np.exp(eta)
    if link == "inverse":
        return 1.0 / eta
    return eta


class GLMPredictionResults:
    """Class to hold prediction results."""

    def __init__(
        self, predicted_mean, se_mean, conf_int_mean, se_obs, conf_int_obs, index=None, alpha=0.05
    ):
        self.predicted_mean = predicted_mean
        self.se_mean = se_mean
        self.conf_int_mean = conf_int_mean
        self.se_obs = se_obs
        self.conf_int_obs = conf_int_obs
        self.index = index
        self.alpha = alpha

    def summary_frame(self):
        """Predictions and mean confidence intervals as a DataFrame."""
        import pandas as pd

        col_lower = f"mean_ci_lower_{(1 - self.alpha) * 100:.1f}%"
        col_upper = f"mean_ci_upper_{(1 - self.alpha) * 100:.1f}%"
        data = {
            "mean": self.predicted_mean,
            "mean_se": self.se_mean,
            col_lower: self.conf_int_mean[:, 0],
            col_upper: self.conf_int_mean[:, 1],
        }
        return pd.DataFrame(data, index=self.index)


class GLMInfluence:
    """Class to compute leverage, Cook's distance, and standardized residuals."""

    def __init__(self, model):
        self.model = model
        self.hat_matrix_diag = model.hat_matrix_diag
        self.cooks_distance = model.cooks_distance()
        self.resid_sspearson = model.resid_sspearson
        self.resid_sdeviance = model.resid_sdeviance
        self.dffits = model.dffits  # R's dffits (uses sigma_(i))
        self.dffits_internal = model.dffits_internal
        self.resid_rstudent = model.resid_rstudent
        self.dfbeta = model.dfbeta
        self.dfbetas = model.dfbetas

    def summary_frame(self):
        """Influence diagnostics (leverage, Cook's D, residuals) as a DataFrame."""
        import pandas as pd

        data = {
            "hat_diag": self.hat_matrix_diag,
            "cooks_d": self.cooks_distance,
            "std_resid_pearson": self.resid_sspearson,
            "std_resid_deviance": self.resid_sdeviance,
            "student_resid": self.resid_rstudent,
            "dffits": self.dffits,
        }
        return pd.DataFrame(data, index=getattr(self.model.data, "index", None))


class GLM(PlottingMixin):
    """Generalized linear model, fit via one of several engines depending on
    `family.family_type` (see the module docstring above and family.py's
    docstring for the five family groups). Construct directly from arrays
    with GLM(X, y, family=...), or from an R-style formula with
    GLM.from_formula(formula, data, family=...) -- the latter is what the
    module-level `glm(...)` shortcut calls.

    Parameters mirror R's glm(): `weights` are prior/case weights, `offset`
    is added to the linear predictor before the link, `solver` selects the
    linear-algebra backend for the IRLS engine (QR/Cholesky/LDLT/SVD
    variants), and `minimizer` selects the optimizer for MLE/QuasiCustom
    families (L-BFGS-B by default). See .fit()'s docstring for the extra
    arguments some families need (`d`, `Z`, `left_limit`/`right_limit`,
    `entry`).
    """

    def __init__(
        self,
        X,
        y,
        family=None,
        weights=None,
        offset=None,
        tol=1e-8,
        maxit=1000,
        solver="colpivqr",
        big_matrix=False,
        firth=False,
        feature_names=None,
        ftol=1e-08,
        gtol=1e-6,
        maxcor=10,
        verbose=0,
        compute_se=True,
        minimizer=None,
        use_t=None,
        standardize=True,
        y_levels=None,
    ):

        self.formula = None
        self.data = None
        self._model_spec = None
        self._Z_from_formula = None
        self._zero_formula = None
        self._zero_model_spec = None
        self._zero_feature_names = None
        self._cens_from_formula = None
        self._entry_from_formula = None

        self._y_categories = list(y_levels) if y_levels is not None else None

        # Each GLM owns an independent copy of its family (incl. FamilyParams): the NB two-step fit
        # stores the estimated theta in `family.params`, and a shared object let one model's fit (e.g.
        # the null-model refit inside `llnull`) silently overwrite another's theta.
        _fam = family if family is not None else Gaussian()
        self.family = _fam.copy() if hasattr(_fam, "copy") else _fam
        # use_t: reference distribution of the Wald statistics. None = automatic (t with df_resid for
        # families whose dispersion is estimated -- Gaussian, Gamma, inverse Gaussian, quasi* -- and
        # normal otherwise, as R's summary.glm); True/False force it.
        self._use_t_arg = use_t
        self._null_cache = {}

        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float).flatten()
        if self._X.ndim != 2:
            raise ValueError("GLM: X must be a 2D matrix (n, p).")
        n, p = self._X.shape
        if self._y.shape[0] != n:
            raise ValueError(
                f"GLM: X and y have a different number of rows ({n} vs {self._y.shape[0]})."
            )

        w = _resolve_array(weights, n, "weights")
        self._weights = w if w is not None else np.ones(n)
        off = _resolve_array(offset, n, "offset")
        self._offset = off if off is not None else np.zeros(n)

        if feature_names is not None:
            feature_names = list(feature_names)
            if len(feature_names) != p:
                raise ValueError(
                    f"GLM: feature_names has {len(feature_names)} names, expected {p}."
                )
            self._feature_names = feature_names
        else:
            self._feature_names = [f"x{j}" for j in range(p)]

        _check_finite(self._X, "X")
        _check_finite(self._y, "y")
        _check_finite(self._weights, "weights")
        _check_finite(self._offset, "offset")

        # standardize=True fits on a rescaled version of X (z-score for each continuous
        # column; binary/constant columns left untouched) to improve the conditioning of
        # the Hessian when covariates live on very different scales (e.g. an interaction
        # like math*science, of order 1e3-1e4, alongside 0/1 dummies). It is an EXACT and
        # invertible reparametrization -- see `_begin_standardization` /
        # `_backtransform_standardization` -- not an approximation: beta/se/vcov are
        # returned on the ORIGINAL scale, and predict() and summary() need not know this
        # happened.
        self.standardize = bool(standardize)
        if self.standardize:
            self._x_center, self._x_scale, self._x_intercept_idx = self._compute_standardization(
                self._X, self._feature_names
            )
        else:
            self._x_center = self._x_scale = self._x_intercept_idx = None

        family_type = getattr(self.family, "family_type", "irls")
        if family_type in ("irls", "quasi", "nb_two_step", "quasi_custom"):
            self._active_cols, self._aliased_cols = _detect_rank_deficiency(
                self._X, self._feature_names
            )
        else:
            self._active_cols = np.arange(p, dtype=int)
            self._aliased_cols = np.array([], dtype=int)
        self.rank_deficient = self._aliased_cols.size > 0
        self._X_active = self._X if not self.rank_deficient else self._X[:, self._active_cols]

        self.tol = tol
        self.maxit = maxit
        self.solver = self._resolve_solver(solver)
        self.minimizer = self._resolve_minimizer(minimizer)
        if minimizer is not None and family_type not in (
            "mle",
            "mle_inflated",
            "mle_gengamma",
            "quasi_custom",
        ):
            import warnings

            warnings.warn(
                f"GLM: minimizer='{minimizer}' was specified but family "
                f"'{self.family.__class__.__name__}' uses the "
                f"'{family_type}' engine, which doesn't go through "
                f"MleSolver/QuasiSolver -- the chosen minimizer has no "
                f"effect here (it only applies to MLE and QuasiCustom "
                f"families: Beta, Weibull, LogNormal, ZeroInflated*, "
                f"GeneralizedGamma, QuasiCustom, etc.).",
                stacklevel=2,
            )
        self.big_matrix = big_matrix
        self.firth = firth
        self.ftol = ftol
        self.gtol = gtol
        self.maxcor = maxcor
        self.verbose = verbose
        self.compute_se = compute_se

        self.beta = None
        self.se = None
        self.vcov = None
        self._deviance = None
        self.rank = None
        self.iterations = None
        self._converged = False
        self._engine = None
        self._fitted = False
        self.dispersion = None
        self.gamma = None
        self.extra_param = None
        self.extra_param2 = None
        self.log_likelihood = None
        self._fit_ctx = None  # d / Z / limits / entry used by the MLE fit (null model, prediction)
        self._moments_cache = None  # (mean, var) of y from the distribution layer (MLE families)
        self._infl_cache = None  # information-based influence measures (MLE families)

    @staticmethod
    def _compute_standardization(X, feature_names):
        n, p = X.shape
        center = np.zeros(p)
        scale = np.ones(p)
        intercept_idx = None

        for j in range(p):
            col = X[:, j]
            uniq = np.unique(col)

            if uniq.shape[0] <= 1:
                if uniq.shape[0] == 1 and intercept_idx is None and np.isclose(uniq[0], 1.0):
                    intercept_idx = j
                continue

            if uniq.shape[0] <= 2 and np.all(np.isin(uniq, [0.0, 1.0])):
                continue

            q75, q25 = np.percentile(col, [75, 25])
            col_iqr = float(q75 - q25)
            col_median = float(np.median(col))

            if col_iqr > 1e-12:
                scale[j] = col_iqr
            elif col_median > 1e-12:
                scale[j] = col_median
            else:
                p95 = float(np.percentile(col, 95))
                if p95 > 1e-12:
                    scale[j] = p95
                else:
                    max_abs = float(np.max(np.abs(col)))
                    scale[j] = max_abs if max_abs > 1e-12 else 1.0

        return center, scale, intercept_idx

    def _begin_standardization(self):
        """Swap self._X (and self._X_active) for the standardized version for the
        duration of one fit() call. `_end_standardization` restores the originals
        (always, even on failure) and `_backtransform_standardization` maps the
        fitted beta/se/vcov back to the original coordinates."""
        self.__x_orig_bak = self._X
        self.__x_active_orig_bak = self._X_active
        self._X = (self._X - self._x_center) / self._x_scale
        self._X_active = self._X if not self.rank_deficient else self._X[:, self._active_cols]

    def _end_standardization(self):
        self._X = self.__x_orig_bak
        self._X_active = self.__x_active_orig_bak
        del self.__x_orig_bak
        del self.__x_active_orig_bak

    def _backtransform_standardization(self):
        """Maps self.beta/self.se/self.vcov from standardized to original-scale
        coordinates. Exact (not a delta-method approximation): standardizing is a
        linear reparametrization beta_orig = D @ beta_scaled with

            D = diag(1/scale), plus D[intercept_row, j] -= center[j]/scale[j]

        (the intercept row absorbs the constant shift introduced by centering; see
        `_compute_standardization`). Applied block-wise so it also covers models
        whose beta is several repeated blocks over the same design columns (e.g.
        Multinomial's K-1 category blocks); any parameters beyond the beta block(s)
        in self.vcov (e.g. Ordinal's thresholds, a secondary submodel's gamma) are
        left as-is -- correctly, since standardizing X doesn't change their scale --
        while their covariance WITH beta still transforms consistently through the
        congruence transform below.
        """
        if getattr(self.family, "family_type", "irls") == "mle_gengamma":
            self._backtransform_standardization_gengamma()
            return

        p = self._x_center.shape[0]
        D = np.diag(1.0 / self._x_scale)
        if self._x_intercept_idx is not None:
            j0 = self._x_intercept_idx
            D[j0, :] += -self._x_center / self._x_scale
            D[j0, j0] = 1.0 / self._x_scale[j0]

        nb = len(self.beta) if self.beta is not None else 0
        if nb == 0 or nb % p != 0:
            import warnings

            warnings.warn(
                "GLM: standardize=True could not be reverted for this fit result "
                f"(len(beta)={nb} is not a multiple of the number of design columns "
                f"p={p}); beta/se/vcov are left in STANDARDIZED coordinates.",
                stacklevel=2,
            )
            return

        n_blocks = nb // p
        self.beta = np.concatenate([D @ self.beta[k * p : (k + 1) * p] for k in range(n_blocks)])

        if self.vcov is not None:
            P_total = self.vcov.shape[0]
            T = np.eye(P_total)
            for k in range(n_blocks):
                sl = slice(k * p, (k + 1) * p)
                T[sl, sl] = D
            self.vcov = T @ self.vcov @ T.T
            diag = np.diag(self.vcov)
            with np.errstate(invalid="ignore"):
                self.se = np.where(
                    np.isfinite(diag) & (diag > 0), np.sqrt(np.maximum(diag, 0.0)), np.nan
                )
        elif self.se is not None:
            # Without vcov, re-derive se (should not happen for the supported family
            # types) by rescaling each se with the same diagonal factor, which ignores
            # the cross-covariance of the centering shift.
            diagD = np.abs(np.diag(D))
            self.se = np.concatenate(
                [diagD * self.se[k * p : (k + 1) * p] for k in range(n_blocks)]
            )

    def _backtransform_standardization_gengamma(self):
        """Original-scale reparametrization for GeneralizedGamma.

        Its design matrix holds covariates only (the three intercepts live in
        ``self.gamma``): column group ``g`` (mu/sigma/Q) enters linear predictor
        ``g`` as ``gamma[g] + X_g @ beta_g``. Fitting on
        ``(X_g - center_g) / scale_g`` therefore maps

            beta_orig  = beta_scaled / scale
            gamma_orig = gamma_scaled - sum_j (center_j / scale_j) * beta_scaled_j

        with the sum restricted to the columns of each group. The covariance is
        transformed with the same block-triangular map acting on
        ``theta = [gamma(3), beta]``:

            T = [[I3, -G], [0, diag(1/scale)]],   G[g, j] = center_j / scale_j
        """
        import warnings

        fam = self.family
        p = self._x_center.shape[0]
        nb = len(self.beta) if self.beta is not None else 0
        ng = len(self.gamma) if self.gamma is not None else 0
        groups = [
            np.asarray(fam.mx_mu_cols, dtype=int),
            np.asarray(fam.mx_sigma_cols, dtype=int),
            np.asarray(fam.mx_Q_cols, dtype=int),
        ]
        covered = np.sort(np.concatenate(groups))
        valid_groups = ng == 3 and nb == p and np.array_equal(covered, np.arange(p))
        valid_vcov = self.vcov is None or self.vcov.shape[0] == ng + p
        if not (valid_groups and valid_vcov):
            warnings.warn(
                "GLM: standardize=True could not be reverted for this GeneralizedGamma "
                f"fit result (len(beta)={nb}, design columns p={p}, len(gamma)={ng}, "
                f"grouped columns={covered.size}); beta/gamma/se/vcov are left in "
                "STANDARDIZED coordinates.",
                stacklevel=2,
            )
            return

        scale = self._x_scale
        D = np.diag(1.0 / scale)
        G = np.zeros((3, p))
        gamma_orig = np.asarray(self.gamma, dtype=float).copy()
        for g, cols in enumerate(groups):
            G[g, cols] = self._x_center[cols] / scale[cols]
            gamma_orig[g] -= float(G[g, cols] @ self.beta[cols])

        self.beta = D @ self.beta
        self.gamma = gamma_orig

        if self.vcov is not None:
            T = np.eye(ng + p)
            T[:ng, ng:] = -G
            T[ng:, ng:] = D
            self.vcov = T @ self.vcov @ T.T
            diag = np.diag(self.vcov)
            with np.errstate(invalid="ignore"):
                self.se = np.where(
                    np.isfinite(diag) & (diag > 0), np.sqrt(np.maximum(diag, 0.0)), np.nan
                )
        elif self.se is not None:
            # Without vcov, rescale the diagonal only (ignoring the cross-covariance
            # of the centering shift), exactly as in the generic branch above.
            self.se = np.concatenate([np.abs(self.se[:ng]), np.abs(1.0 / scale) * self.se[ng:]])

    @property
    def log_likelihood(self):
        """Log-likelihood at the fit. For IRLS families it is evaluated lazily on first access (the
        Tweedie series can be costly), at the dispersion given by `family.ll_dispersion`."""
        if self._log_likelihood is None and self._ll_lazy is not None:
            thunk, self._ll_lazy = self._ll_lazy, None
            self._log_likelihood = thunk()
        return self._log_likelihood

    @log_likelihood.setter
    def log_likelihood(self, value):
        self._log_likelihood = value
        self._ll_lazy = None

    @staticmethod
    def _resolve_solver(solver):
        if isinstance(solver, str):
            key = solver.strip().lower()
            if key in _METHOD_SOLVE:
                return _METHOD_SOLVE[key]
            valid_keys = ", ".join(f"'{k}'" for k in sorted(set(_METHOD_SOLVE.keys())))
            raise ValueError(f"GLM: solver '{solver}' is not valid. Allowed options: {valid_keys}.")
        elif isinstance(solver, (int, np.integer)):
            valid_codes = {
                _glmfit.TYPE_COLPIV_QR,
                _glmfit.TYPE_QR,
                _glmfit.TYPE_CHOLESKY,
                _glmfit.TYPE_LDLT,
                _glmfit.TYPE_FULLPIV_QR,
                _glmfit.TYPE_SVD,
            }
            if solver not in valid_codes:
                raise ValueError(f"GLM: numerical solve method {solver} is not valid.")
            return int(solver)
        else:
            raise TypeError(f"GLM: solver must be str or int, got {type(solver).__name__}.")

    @staticmethod
    def _resolve_minimizer(minimizer):
        if minimizer is None:
            return _Minimizer.BFGS if _Minimizer is not None else None
        if _Minimizer is not None and isinstance(minimizer, _Minimizer):
            return minimizer
        if isinstance(minimizer, str):
            key = minimizer.strip().lower()
            if key in _MINIMIZER_MAP and _MINIMIZER_MAP[key] is not None:
                return _MINIMIZER_MAP[key]
            valid_keys = ", ".join(f"'{k}'" for k in sorted(set(_MINIMIZER_MAP.keys())))
            raise ValueError(
                f"GLM: minimizer '{minimizer}' is not valid. Allowed options: {valid_keys}."
            )
        raise TypeError(
            f"GLM: minimizer must be str or _glmfit.Minimizer, got {type(minimizer).__name__}."
        )

    def _minimizer_label(self):
        """Human-readable name of the current minimizer, for summary()/__repr__.
        With the old binding (no _glmfit.Minimizer) or self.minimizer=None,
        falls back to 'L-BFGS-B' since that's still the only/default option
        in that case."""
        if self.minimizer is None or _Minimizer is None:
            return "L-BFGS-B"
        labels = {
            _Minimizer.LBFGSB: "L-BFGS-B",
            _Minimizer.BFGS: "BFGS",
            _Minimizer.NelderMead: "Nelder-Mead",
        }
        return labels.get(self.minimizer, str(self.minimizer))

    def _make_mle_solver(self):
        """Builds the native MleSolver, passing `minimizer` only if the
        compiled binding already supports it -- avoids a pybind TypeError
        when running against an older _glmfit.so, compiled before
        MleSolver accepted that parameter."""
        kwargs = {"fit_intercept": False}
        if _Minimizer is not None:
            kwargs["minimizer"] = self.minimizer
        return _glmfit.MleSolver(**kwargs)

    def _make_quasi_solver(self):
        """Analogous to _make_mle_solver but for QuasiSolver (only used by
        family_type == 'quasi_custom' -- the rest of the 'quasi' families
        go through fit_quasi()/irls_two_step.hpp, which has no minimizer)."""
        kwargs = {"fit_intercept": False}
        if _Minimizer is not None:
            kwargs["minimizer"] = self.minimizer
        return _glmfit.QuasiSolver(**kwargs)

    def _resolve_fit_column(self, spec, name):
        """Resolve a fit-time data argument that may be a column name.

        Formula-built models carry their ``data``, so ``fit(cens="status")`` can look the
        column up directly; matrix-built models have no ``data`` and must pass values.
        Arrays/Series are returned untouched (their validation happens in the solver).
        """
        if isinstance(spec, str):
            if self.data is None:
                raise ValueError(
                    f"GLM: {name}='{spec}' was given as a column name, but this model has no "
                    "`data`; pass the values directly or build the model with a formula."
                )
            return _resolve_col(spec, self.data, self._X.shape[0], name)
        return spec

    def _expand_beta_se(self, vec_active):
        if not self.rank_deficient:
            return vec_active
        p = self._X.shape[1]
        out = np.full(p, np.nan, dtype=float)
        out[self._active_cols] = vec_active
        return out

    def _expand_vcov(self, vcov_active):
        if not self.rank_deficient:
            return vcov_active
        p = self._X.shape[1]
        out = np.full((p, p), np.nan, dtype=float)
        out[np.ix_(self._active_cols, self._active_cols)] = vcov_active
        return out

    def _mustart(self):
        y, w = self._y, self._weights
        support = getattr(self.family, "SUPPORT", "positive")
        if support == "unit":
            return (w * y + 0.5) / (w + 1.0)
        if support == "real":
            return y.copy()
        return np.maximum(y, 0.1)

    @staticmethod
    def _winsorize(v, lo_pct=0.05, hi_pct=0.95):
        lo, hi = np.quantile(v, [lo_pct, hi_pct])
        return np.clip(v, lo, hi)

    def _find_intercept_column(self):
        X = self._X
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.ptp(col) < 1e-9 * max(1.0, abs(col[0])) and abs(col[0]) > 1e-12:
                return j
        return -1

    def _is_valid_start(self, eta):
        mu = self.family.linkinv(eta)
        return bool(self.family.valid_eta(eta)) and bool(self.family.valid_mu(mu))

    def _ensure_safe_start(self, beta0):
        beta0 = beta0.copy()
        eta = self._X @ beta0 + self._offset
        if self._is_valid_start(eta):
            return beta0

        icol = self._find_intercept_column()
        if icol >= 0:
            base = beta0[icol]
            for attempt in range(1, 61):
                shift = 0.1 * attempt * (1.0 + abs(eta.mean()))
                beta0[icol] = base + shift
                eta = self._X @ beta0 + self._offset
                if self._is_valid_start(eta):
                    return beta0
                beta0[icol] = base - shift
                eta = self._X @ beta0 + self._offset
                if self._is_valid_start(eta):
                    return beta0

        link_desc = ""
        if not isinstance(self.family, CustomFamily):
            link_desc = f", link='{self.family.link}'"
        raise RuntimeError(
            f"GLM: could not find a valid starting point for "
            f"family={self.family.__class__.__name__}{link_desc}."
        )

    def initialize_parameters(self):
        """Build starting values ``(beta0, mu0, eta0)`` for the fitting loop.

        Returns
        -------
        tuple of numpy.ndarray
            Initial coefficients, mean response and linear predictor.
        """
        n, p = self._X.shape
        if isinstance(self.family, CustomFamily):
            beta0 = np.zeros(p)
        else:
            mu0_target = self._mustart()
            link = self.family.link

            if link == "power":
                p_power = getattr(self.family.params, "link_power", 1.0)
                if p_power == 0:
                    eta0_target = np.log(mu0_target)
                else:
                    eta0_target = mu0_target**p_power
            elif link in _FORWARD_LINK:
                eta0_target = _FORWARD_LINK[link](mu0_target)
            else:
                raise ValueError(f"GLM: no direct link implemented for '{link}'.")

            eta0_target = self._winsorize(eta0_target)
            beta0, *_ = np.linalg.lstsq(self._X, eta0_target - self._offset, rcond=None)

        beta0 = self._ensure_safe_start(beta0)
        eta0 = self._X @ beta0 + self._offset
        mu0 = self.family.linkinv(eta0)
        return beta0, mu0, eta0

    def fit(
        self,
        start: Any = None,
        cens: Any = None,
        Z: Any = None,
        left_limit: float | None = None,
        right_limit: float | None = None,
        entry: Any = None,
    ) -> "GLM":
        """Fit the model and return ``self``.

        Parameters
        ----------
        start : array-like, optional
            Initial coefficient values for IRLS/quasi families. Not
            supported for MLE families, whose solver builds its own starts.
        cens : array-like or str, optional
            Event/censoring indicator for the survival families
            (``LogNormal``, ``Weibull``, ``LogLogistic``) and status for
            ``GeneralizedGamma``: 1 = event, 0 = right-censored. ``None``
            treats every observation as an event. A string names a column of
            the model's ``data`` (models built from a formula); it can also
            be declared when building the model and then omitted here.
        Z : array-like, optional
            Design matrix for the secondary submodel (zero-inflation,
            hurdle probability, Beta precision or Simplex dispersion).
            For two-part models it is normally supplied through the
            ``"count_part | zero_part"`` formula instead.
        left_limit, right_limit : float, optional
            Truncation limits for the ``Truncated*`` families. ``None``
            means no additional truncation on that side.
        entry : array-like or str, optional
            Entry (left-truncation) times for ``GeneralizedGamma``. A string
            names a column of the model's ``data``.

        Returns
        -------
        GLM
            The fitted model (``self``), with ``beta``, ``se``, ``vcov``,
            ``log_likelihood`` and the other result attributes populated.
        """
        family_type = getattr(self.family, "family_type", "irls")
        self._moments_cache = None
        self._infl_cache = None
        self._fit_ctx = None
        self._null_cache = {}
        if Z is None:
            Z = self._Z_from_formula

        # `cens`/`entry` can be given as column names for formula-built models; explicit
        # fit() arguments override the defaults recorded at construction time.
        if cens is None:
            cens = self._cens_from_formula
        if entry is None:
            entry = self._entry_from_formula
        cens = self._resolve_fit_column(cens, "cens")
        entry = self._resolve_fit_column(entry, "entry")

        if not self.standardize:
            return self._dispatch_fit(start, cens, Z, left_limit, right_limit, entry, family_type)

        if self.rank_deficient:
            raise NotImplementedError(
                "GLM: standardize=True is not currently supported together with a "
                "rank-deficient design matrix (aliased columns detected). Fit with "
                "standardize=False."
            )

        self._begin_standardization()
        try:
            result = self._dispatch_fit(start, cens, Z, left_limit, right_limit, entry, family_type)
        finally:
            self._end_standardization()
        if self._fitted:
            self._backtransform_standardization()
        return result

    def _dispatch_fit(self, start, cens, Z, left_limit, right_limit, entry, family_type):
        if family_type == "nb_two_step":
            return self._fit_nb_two_step(start)
        if family_type == "mle_gengamma":
            return self._fit_mle_gengamma(start, cens, entry)
        if family_type == "mle_inflated":
            return self._fit_mle_inflated(start, Z)
        if family_type == "mle":
            return self._fit_mle(start, cens, Z, left_limit, right_limit)
        if family_type == "quasi":
            return self._fit_quasi(start)
        if family_type == "quasi_custom":
            return self._fit_quasi_custom(start)
        return self._fit_irls(start)

    def _fit_irls(self, start):
        n, p = self._X.shape
        if start is not None:
            beta0 = np.asarray(start, dtype=float)
            if beta0.shape[0] != p:
                raise ValueError(f"GLM.fit: start has size {beta0.shape[0]}, expected {p}.")
            eta0 = self._X @ beta0 + self._offset
            mu0 = self.family.linkinv(eta0)
        else:
            beta0, mu0, eta0 = self.initialize_parameters()

        if hasattr(self.family, "make_engine"):
            self._engine = self.family.make_engine(
                self._X_active,
                self._y,
                self._weights,
                self._offset,
                tol=self.tol,
                maxit=self.maxit,
                type=self.solver,
                is_big_matrix=self.big_matrix,
                firth=self.firth,
            )
        else:
            self._engine = _glmfit.IrlsEngine(
                self._X_active,
                self._y,
                self._weights,
                self._offset,
                int(self.family.code),
                self.family.params,
                self.tol,
                self.maxit,
                self.solver,
                self.big_matrix,
                self.firth,
            )

        self._engine.init_parms(beta0[self._active_cols], mu0, eta0)
        self.iterations = self._engine.solve(self.maxit)

        self.beta = self._expand_beta_se(self._engine.get_beta())
        self.se = self._expand_beta_se(self._engine.get_se())
        self.vcov = self._expand_vcov(self._engine.get_vcov())
        # `deviance` is the ordinary deviance sum(w_i d_i). With Firth the engine's own objective also
        # subtracts log|X'WX| (penalized deviance): keep that one separately.
        self._deviance = self._engine.get_std_dev()
        self._penalized_deviance = self._engine.get_dev() if self.firth else None
        self.rank = self._engine.get_rank()
        self._converged = self._engine.get_converged()

        mu = self._engine.get_mu()
        if getattr(self.family, "DISPERSION", "fixed") == "estimate" and not isinstance(
            self.family, CustomFamily
        ):
            V = _glmfit.variance(int(self.family.code), self.family.params, mu)
            resid2 = (self._y - mu) ** 2
            pearson_chi2 = float(np.sum(self._weights * resid2 / V))
            # residual df = (observations with positive prior weight) - rank, as R's glm.fit
            n_ok = int(np.sum(self._weights > 0))
            phi = pearson_chi2 / max(n_ok - self.rank, 1)
            self.dispersion = phi
            self.vcov = self.vcov * phi
            self.se = self.se * np.sqrt(phi)

        # logLik / AIC / BIC are evaluated lazily, at the maximizing dispersion (see Family.ll_dispersion)
        fam, y, w, dev = self.family, self._y, self._weights, self._deviance
        est_disp = bool(getattr(fam, "estimates_dispersion", False))

        def _loglik():
            phi_ll = fam.ll_dispersion(y, mu, w, dev) if est_disp else 1.0
            ll_obs = fam.loglike_obs(y, mu, w, phi_ll)
            if ll_obs is None:
                return None
            total = float(np.sum(ll_obs))
            return total if np.isfinite(total) else None

        self._log_likelihood = None
        self._ll_lazy = _loglik
        self._fitted = True
        return self

    def _fit_nb_two_step(self, start):
        n, p = self._X.shape
        if start is not None:
            beta0 = np.asarray(start, dtype=float)
            if beta0.shape[0] != p:
                raise ValueError(f"GLM.fit: start has size {beta0.shape[0]}, expected {p}.")
        else:
            beta0, _, _ = self.initialize_parameters()

        theta_start = self.family.params.theta
        estimate_theta = bool(getattr(self.family, "estimates_theta", True))
        result = _glmfit.fit_nb_two_step(
            self._X_active,
            self._y,
            self._weights,
            self._offset,
            int(self.family.code),
            beta0[self._active_cols],
            theta_start=theta_start,
            outer_maxit=self.maxit,
            outer_tol=self.tol,
            type=self.solver,
            is_big_matrix=self.big_matrix,
            link_power=self.family.params.link_power,
            estimate_theta=estimate_theta,  # NegativeBinomial(theta=...) means KNOWN theta
        )

        self._engine = None
        self.beta = self._expand_beta_se(result.beta)
        self.se = self._expand_beta_se(result.se)
        self.vcov = self._expand_vcov(result.vcov)
        self.theta = result.theta
        # extra_param only when theta was ESTIMATED: it is what AIC/BIC count as a parameter
        self.extra_param = result.theta if estimate_theta else None
        self.family.params.theta = result.theta  # own copy of the family (see __init__)
        # dispersion is theta for NegativeBinomial
        self.dispersion = result.theta
        self.rank = result.rank
        self.iterations = result.outer_iters
        self._converged = result.converged
        # ordinary GLM deviance at the final theta (-2*logLik is a log-likelihood quantity, not a deviance)
        self._deviance = result.deviance

        eta = self._X_active @ self.beta[self._active_cols] + self._offset
        mu = np.asarray(self.family.linkinv(eta))
        fam, y, w = self.family, self._y, self._weights

        def _loglik():
            ll_obs = fam.loglike_obs(y, mu, w, 1.0)
            return float(np.sum(ll_obs)) if ll_obs is not None else None

        self._log_likelihood = None
        self._ll_lazy = _loglik
        self._fitted = True
        return self

    def _fit_quasi(self, start):
        n, p = self._X.shape
        if start is not None:
            beta0 = np.asarray(start, dtype=float)
            if beta0.shape[0] != p:
                raise ValueError(f"GLM.fit: start has size {beta0.shape[0]}, expected {p}.")
        else:
            beta0, _, _ = self.initialize_parameters()

        result = _glmfit.fit_quasi(
            self._X_active,
            self._y,
            self._weights,
            self._offset,
            int(self.family.code),
            beta0[self._active_cols],
            maxit=self.maxit,
            tol=self.tol,
            type=self.solver,
            is_big_matrix=self.big_matrix,
        )

        self._engine = None
        self.beta = self._expand_beta_se(result.beta)
        self.se = self._expand_beta_se(result.se)
        self.vcov = self._expand_vcov(result.vcov)
        self.dispersion = result.dispersion
        # quasi-deviance = deviance of the base family with the same V(mu): defined even though there
        # is no likelihood (R reports it for quasipoisson etc.). log_likelihood stays None.
        self._deviance = result.deviance
        self.rank = result.rank
        self.iterations = result.outer_iters
        self._converged = result.converged
        self.log_likelihood = None
        self._fitted = True
        return self

    def _fit_quasi_custom(self, start):
        fam = self.family
        n, p = self._X.shape
        if start is not None:
            beta0 = np.asarray(start, dtype=float)
            if beta0.shape[0] != p:
                raise ValueError(f"GLM.fit: start has size {beta0.shape[0]}, expected {p}.")
        else:
            beta0, _, _ = self.initialize_parameters()

        solver = self._make_quasi_solver()
        result = solver.fit_custom(
            self._X_active,
            self._y,
            fam.variance_fn,
            weights=self._weights,
            link=fam.quasi_link_type,
            maxiter=self.maxit,
            ftol=self.tol,
            gtol=self.tol,
            offset=self._offset,  # forwarded to the custom-variance fit_custom solver
        )

        self._engine = None
        self.beta = self._expand_beta_se(result.beta)
        self.se = self._expand_beta_se(result.se) if result.se is not None else None
        self.vcov = self._expand_vcov(result.vcov) if result.vcov is not None else None
        self.dispersion = result.dispersion
        self._deviance = 2.0 * result.quasi_deviance
        self.rank = len(self._active_cols)
        self.iterations = result.iterations
        self._converged = result.success
        self._fit_message = result.message
        self.log_likelihood = None  # quasi-likelihood: there is no real log-likelihood
        self._fitted = True
        return self

    def _fit_mle(self, start, d, Z, left_limit, right_limit):
        if start is not None:
            raise ValueError("GLM.fit: `start` does not apply to MLE families.")

        solver = self._make_mle_solver()

        # Origin of Z (used to name the secondary-submodel coefficients in summary()):
        # "formula" (y ~ x | z), "array" (fit(Z=...)) or "default" (the solver picks its Z).
        self._z_source = (
            "formula"
            if Z is not None and Z is self._Z_from_formula
            else "array"
            if Z is not None
            else "default"
        )

        # 1. Get the secondary-link string (the parameter modelled by Z):
        #    gamma_link (Beta phi_link, Simplex sigma_link, Hurdle* zero_link) or
        #    infl_link (ZeroInflated*). None -> the solver uses its default link.
        secondary_link_str = getattr(self.family, "secondary_link", None)

        # 2. Map it to the C++ LinkType if defined
        secondary_link = (
            self._resolve__glmfit_linktype(secondary_link_str)
            if secondary_link_str is not None
            else None
        )

        use_left = left_limit is not None
        use_right = right_limit is not None
        left_val = left_limit if left_limit is not None else 0
        right_val = right_limit if right_limit is not None else -1

        # 3. Call the solver (secondary_link = the secondary-parameter link in MleSolver.fit,
        #    applies to ZI*, Hurdle*, Beta and Simplex; C++ raises ValueError if invalid)
        result = solver.fit(
            int(self.family.code),
            self._X,
            self._y,
            d=d,
            Z=Z,
            left_limit=left_val,
            right_limit=right_val,
            maxiter=self.maxit,
            ftol=self.ftol,
            gtol=self.gtol,
            maxcor=self.maxcor,
            verbose=self.verbose,
            compute_se=self.compute_se,
            secondary_link=secondary_link,
            use_left=use_left,
            use_right=use_right,
        )

        self._warn_unused_weights_offset()
        self._fit_ctx = {
            "d": d,
            "Z": self._Z_used(Z),
            "left": left_limit,
            "right": right_limit,
            "entry": None,
        }

        self._engine = None
        self.beta = result.beta
        self.gamma = result.gamma
        self.extra_param = result.extra_param
        self.dispersion = (
            1 / result.extra_param
            if (result.extra_param is not None and result.extra_param != 0)
            else None
        )
        self.extra_param2 = result.extra_param2
        self.log_likelihood = result.log_likelihood
        if result.success and type(self.family).__name__ in _dist.LL_OMITS_LOG_T:
            # log_likelihood.hpp's Weibull/LogLogistic NLL is written in terms of Z = (log Y - mu)/sigma,
            # not Y: the -log(y) Jacobian per event (d_i = 1) is missing. d=None == all events (see
            # mle_solver.hpp: cens_d = d.value_or(Ones(n))), so that default is replicated here.
            d_eff = np.ones_like(self._y) if d is None else np.asarray(d, dtype=float)
            self.log_likelihood += float(np.sum(d_eff * np.log(np.maximum(self._y, 1e-300))))
        self.se = result.se
        self.vcov = result.vcov
        self._deviance = (-2.0 * self.log_likelihood) if result.success else None

        if isinstance(self.family, Ordinal) and self.gamma is not None and self.vcov is not None:
            # `res.gamma` (thresholds) is on the NATURAL scale (reconstructed from the raw optimizer
            # vector: alpha[0] raw, alpha[m] = alpha[m-1] + exp(gamma_raw[m]) for m >= 1), but the
            # solver's vcov/se are for the RAW (partly log-transformed) vector -- so se[nb+1:] used to be
            # the standard error of log(alpha[m] - alpha[m-1]), not of alpha[m] itself. Delta method:
            # reconstruct_thresholds: alpha[0] = g_raw[0]; alpha[m] = alpha[m-1] + exp(g_raw[m]), m >= 1, so
            # d alpha[m] / d g_raw[j] = 1 for j = 0, = exp(g_raw[j]) = gap[j] for 1 <= j <= m, = 0 for j > m,
            # where gap[j] = alpha[j] - alpha[j-1] is the fitted gap (the SAME value for every row m >= j).
            nb = len(self.beta)
            nt = len(self.gamma)
            J = np.zeros((nt, nt))
            J[:, 0] = 1.0
            for j in range(1, nt):
                J[j:, j] = self.gamma[j] - self.gamma[j - 1]
            V_thr = J @ self.vcov[nb : nb + nt, nb : nb + nt] @ J.T
            self.vcov = self.vcov.copy()
            self.vcov[nb : nb + nt, nb : nb + nt] = V_thr
            self.se = self.se.copy()
            self.se[nb : nb + nt] = np.sqrt(np.maximum(np.diag(V_thr), 0.0))
        self.rank = self._X.shape[1]
        self.iterations = result.iterations
        self._converged = result.success
        self._fit_message = result.message
        self._fitted = True
        return self

    _LINK_STR_TO__glmfit_ENUM = {
        "identity": "Identity",
        "log": "Log",
        "sqrt": "Sqrt",
        "logit": "Logit",
        "probit": "Probit",
        "cloglog": "Cloglog",
        "cauchit": "Cauchit",
        "loglog": "Loglog",
        "inverse": "Inverse",
    }

    def _resolve__glmfit_linktype(self, link_str):
        enum_name = self._LINK_STR_TO__glmfit_ENUM.get(link_str)
        if enum_name is None:
            valid = ", ".join(f'"{k}"' for k in self._LINK_STR_TO__glmfit_ENUM)
            raise ValueError(f"GLM: link '{link_str}' is not available. Valid: {valid}.")
        return getattr(_glmfit.LinkType, enum_name)

    def _fit_mle_inflated(self, start, Z=None):
        if start is not None:
            raise ValueError("GLM.fit: `start` does not apply to Inflated families.")

        fam = self.family
        solver = self._make_mle_solver()
        base_link = self._resolve__glmfit_linktype(fam.base_link)
        infl_link = self._resolve__glmfit_linktype(fam.infl_link)

        if isinstance(fam, ValueInflatedPoisson):
            result = solver.fit_poisson_inflated(
                self._X,
                self._y,
                fam.target_value,
                base_link=base_link,
                infl_link=infl_link,
                maxiter=self.maxit,
                ftol=self.ftol,
                gtol=self.gtol,
                maxcor=self.maxcor,
                verbose=self.verbose,
                compute_se=self.compute_se,
                Z=Z,
            )
        elif isinstance(fam, ValueInflatedNegativeBinomial):
            var_form = getattr(_glmfit.VarianceForm, fam.var_form.capitalize())
            result = solver.fit_negative_binomial_inflated(
                self._X,
                self._y,
                fam.target_value,
                var_form=var_form,
                base_link=base_link,
                infl_link=infl_link,
                maxiter=self.maxit,
                ftol=self.ftol,
                gtol=self.gtol,
                maxcor=self.maxcor,
                verbose=self.verbose,
                compute_se=self.compute_se,
                Z=Z,
            )
        else:
            raise TypeError(f"GLM: invalid family for mle_inflated: {fam.__class__.__name__}")

        self._warn_unused_weights_offset()
        self._fit_ctx = {
            "d": None,
            "Z": self._Z_used(Z),
            "left": None,
            "right": None,
            "entry": None,
        }

        self._engine = None
        self.beta = result.beta
        self.gamma = result.gamma
        if isinstance(fam, ValueInflatedNegativeBinomial):
            self.extra_param = result.extra_param
            self.extra_param2 = result.extra_param2
        else:
            self.extra_param = None
            self.extra_param2 = None

        self.log_likelihood = result.log_likelihood
        self.dispersion = (
            1 / result.extra_param
            if (result.extra_param is not None and result.extra_param != 0)
            else None
        )
        self.se = result.se
        self.vcov = result.vcov
        self._deviance = (-2.0 * result.log_likelihood) if result.success else None
        self.rank = len(self.beta) + len(self.gamma)
        self.iterations = result.iterations
        self._converged = result.success
        self._fit_message = result.message
        self._fitted = True
        return self

    def _fit_mle_gengamma(self, start, d, entry):
        if start is not None:
            raise ValueError("GLM.fit: `start` does not apply to GeneralizedGamma.")

        fam = self.family
        if not isinstance(fam, GeneralizedGamma):
            raise TypeError(f"GLM: family_type=='mle_gengamma' but got {fam.__class__.__name__}.")
        if d is None:
            raise ValueError(
                "GLM.fit: GeneralizedGamma needs `d` (status: 1=event, 0=right-censored)."
            )

        n = self._X.shape[0]
        status = np.asarray(d, dtype=float).flatten()
        if status.shape[0] != n:
            raise ValueError(f"GLM.fit: d has size {status.shape[0]}, expected {n}.")

        if entry is None:
            start_time = np.zeros(n)
        else:
            start_time = np.asarray(entry, dtype=float).flatten()
            if start_time.shape[0] != n:
                raise ValueError(f"GLM.fit: entry has size {start_time.shape[0]}, expected {n}.")

        Y = np.column_stack([start_time, self._y, np.zeros(n), status])

        solver = self._make_mle_solver()
        link_mu = self._resolve__glmfit_linktype(fam.link_mu)
        link_sigma = self._resolve__glmfit_linktype(fam.link_sigma)
        link_Q = self._resolve__glmfit_linktype(fam.link_Q)

        result = solver.fit_generalized_gamma(
            self._X,
            Y,
            fam.mx_mu_cols,
            fam.mx_sigma_cols,
            fam.mx_Q_cols,
            weights=self._weights,
            link_mu=link_mu,
            link_sigma=link_sigma,
            link_Q=link_Q,
            maxiter=self.maxit,
            ftol=self.ftol,
            gtol=self.gtol,
            maxcor=self.maxcor,
            verbose=self.verbose,
            compute_se=self.compute_se,
        )

        self._fit_ctx = {"d": status, "Z": None, "left": None, "right": None, "entry": start_time}

        self._engine = None
        self.beta = result.beta
        self.gamma = result.gamma
        self.extra_param = result.extra_param
        self.extra_param2 = result.extra_param2
        self.log_likelihood = result.log_likelihood
        self.se = result.se
        self.vcov = result.vcov
        self._deviance = (-2.0 * result.log_likelihood) if result.success else None

        n_beta = len(self.beta) if self.beta is not None else 0
        n_gamma = len(self.gamma) if self.gamma is not None else 0
        self.rank = n_beta + n_gamma

        self.iterations = result.iterations
        self._converged = result.success
        self._fit_message = result.message
        self._fitted = True
        return self

    # MLE: parameter layout, inference on all parameters, distribution layer
    # An MLE fit estimates more than beta: a secondary submodel (gamma: pi of
    # ZeroInflated*/Hurdle*, phi of Beta, sigma^2 of Simplex, the scalars of
    # StudentT/Skew*) and, for some families, one independent auxiliary scalar
    # (alpha, sigma, shape, phi -- Family.EXTRA_FREE). Everything below is built
    # so that residuals, fitted values, predictions, information criteria and
    # tests use ALL of them, in the same order as the optimizer / vcov:
    #     [beta, gamma, log(extra)]        (GeneralizedGamma: [gamma(3), beta])

    def _is_mle(self):
        return getattr(self.family, "family_type", "irls") in (
            "mle",
            "mle_inflated",
            "mle_gengamma",
        )

    def _warn_unused_weights_offset(self):
        if np.any(self._weights != 1.0) or np.any(self._offset != 0.0):
            import warnings

            warnings.warn(
                f"GLM: MleSolver does not use `weights`/`offset` -- they are ignored by the "
                f"'{self.family.__class__.__name__}' fit, and therefore by its fitted values, "
                f"residuals and predictions.",
                stacklevel=3,
            )

    def _Z_used(self, Z):
        """The Z matrix the MLE solver actually used (its own default when Z is None):
        X for probability-type secondary parameters (ZI*/Hurdle*), intercept-only for
        positive ones (Beta precision / Simplex dispersion), None if there is no Z."""
        if Z is not None:
            return np.asarray(Z, dtype=float)
        kind = getattr(self.family, "SECONDARY_KIND", None)
        if kind == "positive":
            return np.ones((self._X.shape[0], 1))
        if kind == "probability":
            return self._X
        return None

    def _n_gamma(self):
        return 0 if self.gamma is None else len(self.gamma)

    def _extra_free_names(self):
        return tuple(getattr(self.family, "EXTRA_FREE", ()))

    def _extra_free_values(self):
        """Independent auxiliary scalars on the estimation (log) scale."""
        if not self._extra_free_names():
            return np.zeros(0)
        ep = self.extra_param
        if ep is None or not np.isfinite(ep) or ep <= 0:
            return np.zeros(0)
        return np.array([np.log(ep)])

    def _param_layout(self):
        """Slices of beta / gamma / extra inside `all_params` (= vcov order)."""
        nb, ng, ne = len(self.beta), self._n_gamma(), len(self._extra_free_values())
        if getattr(self.family, "family_type", "irls") == "mle_gengamma":
            return {"beta": slice(ng, ng + nb), "gamma": slice(0, ng), "extra": slice(0, 0)}
        # Ordinal thresholds (self.gamma) are on the natural scale and, since GLM._fit_mle applies the
        # delta-method correction to their vcov block, inference now covers them like any other family.
        return {
            "beta": slice(0, nb),
            "gamma": slice(nb, nb + ng),
            "extra": slice(nb + ng, nb + ng + ne),
        }

    @property
    def all_params(self):
        """All free parameters, in the order of `vcov`/`bse`: [beta, gamma, log(extra)]
        (estimation scale). For non-MLE families this is just `beta`."""
        self._require_fitted()
        if not self._is_mle():
            return self.beta
        lay = self._param_layout()
        parts = [(lay["beta"], np.asarray(self.beta, dtype=float))]
        if self._n_gamma() and lay["gamma"].stop > lay["gamma"].start:
            parts.append((lay["gamma"], np.asarray(self.gamma, dtype=float)))
        ex = self._extra_free_values()
        if ex.size and lay["extra"].stop > lay["extra"].start:
            parts.append((lay["extra"], ex))
        total = max(sl.stop for sl, _ in parts)
        theta = np.zeros(total)
        for sl, v in parts:
            theta[sl] = v
        return theta

    def _gamma_param_names(self, n_gamma):
        fam = self.family
        if isinstance(fam, (Beta, Simplex)):
            prefix = "precision" if isinstance(fam, Beta) else "dispersion"
            return [f"{prefix}:{nm}" for nm in self._secondary_submodel_names(n_gamma)]
        feats = (
            self._zero_feature_names
            if self._zero_feature_names is not None
            else self._feature_names
        )
        names = fam.gamma_names(self._y, n_gamma, feats) if hasattr(fam, "gamma_names") else None
        if names is None or len(names) != n_gamma:
            names = [f"gamma[{j}]" for j in range(n_gamma)]
        return list(names)

    @property
    def all_param_names(self):
        """Names matching `all_params`: beta names, then the secondary submodel
        (e.g. 'infl:age', 'precision:Intercept'), then the auxiliary scalars ('log_alpha')."""
        self._require_fitted()
        if not self._is_mle():
            return self.param_names
        lay = self._param_layout()
        total = self.all_params.shape[0]
        names = [None] * total
        for j, nm in zip(range(lay["beta"].start, lay["beta"].stop), self.param_names):
            names[j] = nm
        if lay["gamma"].stop > lay["gamma"].start:
            for j, nm in zip(
                range(lay["gamma"].start, lay["gamma"].stop),
                self._gamma_param_names(self._n_gamma()),
            ):
                names[j] = nm
        if lay["extra"].stop > lay["extra"].start:
            for j, nm in zip(
                range(lay["extra"].start, lay["extra"].stop), self._extra_free_names()
            ):
                names[j] = nm
        return names

    @property
    def _k_free(self):
        """Number of free parameters actually estimated (excluding an IRLS-estimated dispersion)."""
        if self._is_mle():
            V = self.vcov
            if V is not None:
                V = np.asarray(V)
                if V.ndim == 2 and V.shape[0] == V.shape[1]:
                    return int(V.shape[0])  # the vcov is over exactly the optimized vector
            if getattr(self.family, "family_type", "irls") == "mle_gengamma":
                return len(self.beta) + self._n_gamma()
            return len(self.beta) + self._n_gamma() + len(self._extra_free_values())
        n_extra = (1 if self.extra_param is not None else 0) + (
            1 if self.extra_param2 is not None else 0
        )
        # estimated coefficients = rank: aliased (collinear) columns are not parameters
        n_coef = int(self.rank) if self.rank is not None else len(self.beta)
        return n_coef + self._n_gamma() + n_extra

    @property
    def n_params(self):
        """Total number of estimated parameters (beta + gamma + independent auxiliary scalars
        [+ dispersion for IRLS families with estimated dispersion]); used by AIC, BIC,
        df_resid and the adjusted pseudo R-squared."""
        self._require_fitted()
        return self._k_free + (
            1 if getattr(self.family, "DISPERSION", "fixed") == "estimate" else 0
        )

    def _inference_block(self):
        """(theta, names, se, vcov) over ALL parameters (beta only if the layout can't be trusted)."""
        theta = np.asarray(self.all_params, dtype=float)
        names = list(self.all_param_names)
        se = None if self.se is None else np.asarray(self.se, dtype=float)
        V = None if self.vcov is None else np.asarray(self.vcov, dtype=float)
        n = theta.shape[0]
        if se is not None and se.shape[0] >= n and (V is None or V.shape[0] >= n):
            return theta, names, se[:n], (None if V is None else V[:n, :n])
        nb = len(self.beta)
        se_b = None if se is None else se[:nb]
        V_b = None if V is None else V[:nb, :nb]
        return np.asarray(self.beta, dtype=float), list(self.param_names), se_b, V_b

    def _loglike_obs_for_inference(self, theta=None):
        """Per-observation log-likelihood for the inference parameter vector, when available."""
        if theta is None:
            theta = np.asarray(self.all_params, dtype=float)
        theta = np.asarray(theta, dtype=float)
        if self._gengamma_supported():
            return np.asarray(self._gengamma_loglik_obs(theta), dtype=float)
        if self._categorical_supported():
            return np.asarray(self._categorical_loglik_obs(theta), dtype=float)
        if self._mle_supported():
            ctx = self._fit_ctx or {}
            comp = self._mle_components(theta, self._X, ctx.get("Z"))
            return np.asarray(_dist.logpdf(self.family, comp, self._y, ctx.get("d")), dtype=float)
        if hasattr(self.family, "loglike_obs"):
            phi = self._phi() if getattr(self.family, "estimates_dispersion", False) else 1.0
            try:
                ll = self.family.loglike_obs(self._y, self.fittedvalues, self._weights, phi)
            except Exception:
                ll = None
            if ll is not None:
                return np.asarray(ll, dtype=float)
        return None

    def _score_obs(self):
        """Per-observation score contributions for inference parameters, when computable."""
        theta, _, _, V = self._inference_block()
        n = self._y.shape[0]
        k = theta.shape[0]
        # Prefer numerical gradient of per-observation log-likelihood when the likelihood
        # depends on the full inference parameter vector.
        ll_obs = self._loglike_obs_for_inference(theta)
        if ll_obs is not None and ll_obs.shape[0] == n and (
            self._is_mle() or self._gengamma_supported() or self._categorical_supported()
        ):
            scores = np.empty((n, k), dtype=float)
            for j in range(k):
                h = 1e-6 * max(1.0, abs(float(theta[j])))
                tp = theta.copy()
                tm = theta.copy()
                tp[j] += h
                tm[j] -= h
                lp = self._loglike_obs_for_inference(tp)
                lm = self._loglike_obs_for_inference(tm)
                scores[:, j] = (np.asarray(lp, dtype=float) - np.asarray(lm, dtype=float)) / (
                    2.0 * h
                )
            if np.all(np.isfinite(scores)):
                return scores

        # IRLS/quasi families: score for beta is X_i * W_i * (z_i - eta_i) (scaled).
        family_type = getattr(self.family, "family_type", "irls")
        if family_type in ("irls", "quasi", "nb_two_step", "quasi_custom") and k == len(self.beta):
            W = self._working_weights()
            rw = self.resid_working
            if W is not None and rw is not None:
                Xb = self._X_active
                scores_act = Xb * (W * rw)[:, None]
                # Drop dispersion scaling from scores for families with estimated dispersion
                # by keeping them in the same scale as the unscaled information X'WX.
                if not self.rank_deficient:
                    return np.asarray(scores_act, dtype=float)
                scores = np.zeros((n, len(self.beta)), dtype=float)
                scores[:, self._active_cols] = scores_act
                return scores
        return None

    def _robust_cov(self, cov_type="HC0", cov_kwds=None):
        """Robust (sandwich) covariance matrix for inference parameters."""
        cov_kwds = cov_kwds or {}
        cov_type = str(cov_type).upper()
        theta, names, se, V_model = self._inference_block()
        if V_model is None:
            raise RuntimeError(
                f"GLM: cannot compute {cov_type} covariance because the model covariance is unavailable."
            )
        scores = self._score_obs()
        if scores is None:
            raise NotImplementedError(
                f"GLM: robust covariance '{cov_type}' is not available for this model/family."
            )
        n, k = scores.shape
        if scores.shape[1] != V_model.shape[0]:
            raise NotImplementedError(
                f"GLM: robust covariance '{cov_type}' is not available for this parameter layout."
            )

        # Bread: inverse observed information. For IRLS/quasi with estimated dispersion,
        # model vcov = phi * (X'WX)^-1 while scores above correspond to unscaled equations,
        # so use vcov/phi as bread.
        bread = np.asarray(V_model, dtype=float).copy()
        family_type = getattr(self.family, "family_type", "irls")
        if family_type in ("irls", "quasi", "nb_two_step", "quasi_custom") and getattr(
            self.family, "estimates_dispersion", False
        ):
            phi = self._phi()
            if phi and np.isfinite(phi) and phi > 0:
                bread = bread / phi

        groups = cov_kwds.get("groups", cov_kwds.get("cluster"))
        if cov_type == "CLUSTER":
            if groups is None:
                raise ValueError("GLM: cov_type='cluster' requires cov_kwds={'groups': ...}.")
            groups = np.asarray(groups)
            if groups.shape[0] != n:
                raise ValueError(
                    f"GLM: groups has length {groups.shape[0]}, expected {n}."
                )
            meat = np.zeros((k, k), dtype=float)
            for g in pd.unique(groups):
                sg = scores[groups == g].sum(axis=0)
                meat += np.outer(sg, sg)
            G = int(pd.unique(groups).size)
            if G > 1:
                adj = (G / (G - 1.0)) * ((n - 1.0) / max(n - k, 1))
            else:
                adj = 1.0
            cov = bread @ meat @ bread
            return cov * adj

        if cov_type in {"HC0", "HC1", "HC2", "HC3"}:
            h = self.hat_matrix_diag
            if cov_type == "HC0":
                scale = np.ones(n)
                adj = 1.0
            elif cov_type == "HC1":
                scale = np.ones(n)
                adj = n / max(n - k, 1)
            elif cov_type == "HC2":
                if h is None:
                    raise NotImplementedError("GLM: HC2 requires leverage values.")
                scale = 1.0 / np.maximum(1.0 - np.asarray(h, dtype=float), 1e-12)
                adj = 1.0
            else:  # HC3
                if h is None:
                    raise NotImplementedError("GLM: HC3 requires leverage values.")
                scale = 1.0 / np.maximum(1.0 - np.asarray(h, dtype=float), 1e-12) ** 2
                adj = 1.0
            meat = (scores * scale[:, None]).T @ scores
            cov = bread @ meat @ bread
            return cov * adj

        raise ValueError(
            f"GLM: cov_type '{cov_type}' is not supported. Use HC0, HC1, HC2, HC3 or cluster."
        )

    def _inference_params(self, cov_type=None, cov_kwds=None):
        """Inference parameters with optional robust covariance."""
        theta, names, se, V = self._inference_block()
        if cov_type is None:
            return theta, names, se, V
        V_r = self._robust_cov(cov_type=cov_type, cov_kwds=cov_kwds)
        diag = np.diag(V_r)
        with np.errstate(invalid="ignore"):
            se_r = np.where(np.isfinite(diag) & (diag > 0), np.sqrt(np.maximum(diag, 0.0)), np.nan)
        if se_r.shape[0] != theta.shape[0]:
            # fall back if layout mismatch
            return theta, names, se, V_r
        return theta, names, se_r, V_r

    def get_robustcov_results(self, cov_type="HC0", **cov_kwds):
        """Return robust covariance, standard errors and parameter names.

        Parameters
        ----------
        cov_type : {"HC0", "HC1", "HC2", "HC3", "cluster"}, default "HC0"
            Robust covariance type.
        **cov_kwds
            Extra arguments (for example ``groups`` for clustered covariance).

        Returns
        -------
        dict
            Dictionary with ``cov_type``, ``cov_params``, ``bse``, ``params`` and
            ``param_names``.
        """
        theta, names, se, V = self._inference_params(cov_type=cov_type, cov_kwds=cov_kwds)
        return {
            "cov_type": cov_type,
            "cov_params": V,
            "bse": se,
            "params": theta,
            "param_names": names,
        }

    def _beta_positions(self):
        theta_len = len(self._inference_block()[0])
        if theta_len == len(self.beta):
            return np.arange(theta_len)
        lay = self._param_layout()
        return np.arange(lay["beta"].start, lay["beta"].stop)

    # Distribution layer (mean / variance using every parameter)
    def _mle_supported(self):
        return (
            self._is_mle()
            and self.beta is not None
            and self._fit_ctx is not None
            and _dist.supports(self.family)
        )

    # Multinomial / Ordinal: category probabilities, predicted class, deviance-type residuals
    def _multinomial_probs(self, X=None):
        """(n, K) fitted category probabilities (softmax for logit, sequential stick-breaking otherwise,
        mirroring MultinomialMultiLinkLoss::evaluate_inverse_link exactly)."""
        X = self._X if X is None else X
        fam = self.family
        n_vars = len(self._feature_names)
        K = len(self.beta) // n_vars + 1
        B = self.beta.reshape(n_vars, K - 1, order="F")  # column j = coefficients of category j
        eta = X @ B  # (n, K-1)
        link = getattr(fam, "link", "logit")
        if link == "logit":
            m = eta.max(axis=1, keepdims=True)
            ex = np.exp(eta - m)
            last = np.exp(-m)
            allc = np.concatenate([ex, last], axis=1)
            return allc / allc.sum(axis=1, keepdims=True)
        F = np.clip(_dist.inv_link(link, eta), 1e-12, 1 - 1e-12)
        probs = np.empty((X.shape[0], K))
        rem = np.ones(X.shape[0])
        for j in range(K - 1):
            probs[:, j] = rem * F[:, j]
            rem = rem * (1.0 - F[:, j])
        probs[:, K - 1] = rem
        return probs

    def _ordinal_probs(self, X=None):
        """(n, M) fitted category probabilities P(Y=m) = G(alpha_m - eta) - G(alpha_{m-1} - eta),
        mirroring OrdinalLoss::operator() (eta = X beta with NO intercept column; alpha increasing)."""
        X = self._X if X is None else X
        link = self.family.link
        eta = X @ self.beta
        cuts = np.r_[-np.inf, self.gamma, np.inf]
        G = lambda z: _dist.inv_link(link, z)
        edges = np.array(
            [np.where(np.isinf(c), (c > 0) * 1.0 + (c < 0) * 0.0, G(c - eta)) for c in cuts]
        ).T
        return np.diff(edges, axis=1)

    def _categorical_supported(self):
        return isinstance(self.family, (Multinomial, Ordinal)) and self.beta is not None

    def _categorical_loglik_obs(self, theta):
        """Per-observation log P(Y = y_i | x_i, theta), theta = all_params (== beta for Multinomial;
        [beta, gamma] for Ordinal, gamma on the NATURAL threshold scale -- matching what `all_params`
        already reports, since the SE fix applies the same natural-scale covariance)."""
        fam = self.family
        nb = len(self.beta)
        if isinstance(fam, Multinomial):
            beta = theta
            n_vars = len(self._feature_names)
            K = len(beta) // n_vars + 1
            B = beta.reshape(n_vars, K - 1, order="F")
            eta = self._X @ B
            link = getattr(fam, "link", "logit")
            if link == "logit":
                m = eta.max(axis=1, keepdims=True)
                ex = np.exp(eta - m)
                allc = np.concatenate([ex, np.exp(-m)], axis=1)
                P = allc / allc.sum(axis=1, keepdims=True)
            else:
                F = np.clip(_dist.inv_link(link, eta), 1e-12, 1 - 1e-12)
                P = np.empty((self._X.shape[0], K))
                rem = np.ones(self._X.shape[0])
                for j in range(K - 1):
                    P[:, j] = rem * F[:, j]
                    rem = rem * (1.0 - F[:, j])
                P[:, K - 1] = rem
        else:
            beta = theta[:nb]
            alpha = theta[nb:]
            eta = self._X @ beta
            link = fam.link
            cuts = np.r_[-np.inf, alpha, np.inf]
            G = lambda z: _dist.inv_link(link, z)
            edges = np.array(
                [np.where(np.isinf(c), (c > 0) * 1.0 + (c < 0) * 0.0, G(c - eta)) for c in cuts]
            ).T
            P = np.diff(edges, axis=1)
        idx = np.clip(np.rint(self._y).astype(int), 0, P.shape[1] - 1)
        return np.log(np.maximum(P[np.arange(P.shape[0]), idx], 1e-300))

    def predict_proba(self, exog=None, transform=True):
        """(n, K) fitted category probabilities for Multinomial / Ordinal models."""
        self._require_fitted()
        fam = self.family
        if not isinstance(fam, (Multinomial, Ordinal)):
            raise TypeError("GLM.predict_proba: only defined for Multinomial and Ordinal families.")
        X_new, _, idx = self._mle_design(exog, transform)
        P = (
            self._multinomial_probs(X_new)
            if isinstance(fam, Multinomial)
            else self._ordinal_probs(X_new)
        )
        if exog is not None and idx is not None:
            import pandas as pd

            names = [f"cat{k}" for k in range(P.shape[1])]
            return pd.DataFrame(P, index=idx, columns=names)
        return P

    def predict_category(self, exog=None, transform=True):
        """Predicted category (argmax of the fitted probabilities) for Multinomial / Ordinal models."""
        P = self.predict_proba(exog, transform)
        vals = P.values if hasattr(P, "values") else P
        cat = np.argmax(vals, axis=1)
        if hasattr(P, "index"):
            import pandas as pd

            return pd.Series(cat, index=P.index, name="category")
        return cat

    def _categorical_fitted_obs_prob(self):
        """Fitted probability of the OBSERVED category, P(Y = y_i | x_i) (used as `fittedvalues` and to build
        the deviance-type residual for Multinomial / Ordinal)."""
        P = (
            self._multinomial_probs()
            if isinstance(self.family, Multinomial)
            else self._ordinal_probs()
        )
        idx = np.rint(self._y).astype(int)
        return P[np.arange(P.shape[0]), np.clip(idx, 0, P.shape[1] - 1)]

    # GeneralizedGamma: 3 own linear predictors (mu, sigma, Q), each on its own X columns
    def _gengamma_supported(self):
        return (
            getattr(self.family, "family_type", "irls") == "mle_gengamma"
            and self.gamma is not None
            and self.beta is not None
        )

    def _gengamma_row_params(self, theta, X):
        """(mu_i, sigma_i, Q_i) from theta = [mu0, sigma0, Q0, beta...] (== all_params order) and design X,
        mirroring GeneralizedGammaLoss::operator() exactly, including the |Q_i| < 1e-4 clamp."""
        fam = self.family
        eta_mu = np.full(X.shape[0], theta[0])
        eta_sg = np.full(X.shape[0], theta[1])
        eta_Q = np.full(X.shape[0], theta[2])
        beta = theta[3:]
        j = 0
        for c in fam.mx_mu_cols:
            eta_mu = eta_mu + X[:, c] * beta[j]
            j += 1
        for c in fam.mx_sigma_cols:
            eta_sg = eta_sg + X[:, c] * beta[j]
            j += 1
        for c in fam.mx_Q_cols:
            eta_Q = eta_Q + X[:, c] * beta[j]
            j += 1
        mu_i = _gengamma_inverse_link(eta_mu, fam.link_mu)
        sg_i = _gengamma_inverse_link(eta_sg, fam.link_sigma)
        Q_i = _gengamma_inverse_link(eta_Q, fam.link_Q)
        Q_i = np.where(np.abs(Q_i) < 1e-4, np.where(Q_i >= 0, 1e-4, -1e-4), Q_i)
        return mu_i, sg_i, Q_i

    def _gengamma_mu_linear_predictor(self):
        """Linear predictor of the μ (log-time location) submodel, before the link."""
        theta = np.asarray(self.all_params, dtype=float)
        eta = np.full(self._X.shape[0], theta[0])
        beta = theta[3:]
        j = 0
        for col in self.family.mx_mu_cols:
            eta = eta + self._X[:, col] * beta[j]
            j += 1
        return eta

    def _gengamma_row_loglik(self, mu_cand, sigma_i, Q_i, t, d, ctx):
        """Per-row log-likelihood at a CANDIDATE mu (sigma_i, Q_i fixed), same truncation logic as
        `_gengamma_loglik_obs`; used for the saturated log-likelihood (golden-section search over mu)."""
        entry = ctx.get("entry")
        entry = np.zeros(t.shape[0]) if entry is None else np.asarray(entry, dtype=float)
        ll = np.where(
            d > 0.5,
            _dist.gengamma_logpdf(t, mu_cand, sigma_i, Q_i),
            _dist.gengamma_logsurv(t, mu_cand, sigma_i, Q_i),
        )
        has_entry = entry > 1e-8
        if np.any(has_entry):
            ll = ll - np.where(
                has_entry,
                _dist.gengamma_logsurv(np.where(has_entry, entry, 1.0), mu_cand, sigma_i, Q_i),
                0.0,
            )
        return np.where(entry >= t - 1e-5, 0.0, ll)

    def _gengamma_moments(self):
        """(E[T], Var[T]) at the fit, CONDITIONAL on T > entry for rows with left truncation (entry > 0) --
        the unconditional moments understate the mean whenever there is delayed entry, since the model only
        ever observed T > entry for those rows. Rows with entry <= 0 (or no entry at all) use the plain
        (unconditional) moments; both are exact under the same Prentice/flexsurv parametrization as the fit."""
        theta = np.asarray(self.all_params, dtype=float)
        mu_i, sg_i, Q_i = self._gengamma_row_params(theta, self._X)
        ctx = self._fit_ctx or {}
        entry = ctx.get("entry")
        if entry is None:
            return _dist.gengamma_mean_var(mu_i, sg_i, Q_i)
        entry = np.asarray(entry, dtype=float)
        has_entry = entry > 1e-8
        if not np.any(has_entry):
            return _dist.gengamma_mean_var(mu_i, sg_i, Q_i)
        m = np.empty(mu_i.shape[0])
        v = np.empty(mu_i.shape[0])
        if np.any(~has_entry):
            m[~has_entry], v[~has_entry] = _dist.gengamma_mean_var(
                mu_i[~has_entry], sg_i[~has_entry], Q_i[~has_entry]
            )
        m[has_entry], v[has_entry] = _dist.gengamma_mean_var_truncated(
            mu_i[has_entry], sg_i[has_entry], Q_i[has_entry], entry[has_entry]
        )
        return m, v

    def _gengamma_loglik_obs(self, theta):
        """Per-observation log-likelihood, mirroring GeneralizedGammaLoss::operator() EXACTLY, including
        left truncation (entry) and right censoring (d). Rows with entry >= time (invalid interval, silently
        skipped by the C++ objective) contribute 0."""
        ctx = self._fit_ctx or {}
        entry = ctx.get("entry")
        entry = np.zeros(self._y.shape[0]) if entry is None else np.asarray(entry, dtype=float)
        d = ctx.get("d")
        d = np.ones(self._y.shape[0]) if d is None else np.asarray(d, dtype=float)
        t = self._y
        mu_i, sg_i, Q_i = self._gengamma_row_params(theta, self._X)
        ll = np.where(
            d > 0.5,
            _dist.gengamma_logpdf(t, mu_i, sg_i, Q_i),
            _dist.gengamma_logsurv(t, mu_i, sg_i, Q_i),
        )
        has_entry = entry > 1e-8
        if np.any(has_entry):
            ll = ll - np.where(
                has_entry,
                _dist.gengamma_logsurv(np.where(has_entry, entry, 1.0), mu_i, sg_i, Q_i),
                0.0,
            )
        return np.where(entry >= t - 1e-5, 0.0, ll)

    def _mle_components(self, theta, X, Z):
        fam = self.family
        ctx = self._fit_ctx or {}
        nb, ng = len(self.beta), self._n_gamma()
        beta = theta[:nb]
        gamma = theta[nb : nb + ng]
        ex = theta[nb + ng :]
        eta = np.ascontiguousarray(X @ beta, dtype=float)
        mu = np.asarray(fam.linkinv(eta), dtype=float)
        sec = None
        if getattr(fam, "SECONDARY_KIND", None):
            if Z is None:
                raise ValueError("GLM: this model needs the secondary-submodel design matrix Z.")
            sec = _dist.inv_link(fam.secondary_link, Z @ gamma)
        extra = float(np.exp(ex[0])) if len(ex) else None
        return _dist.make_components(
            mu=mu, sec=sec, gamma=gamma, extra=extra, left=ctx.get("left"), right=ctx.get("right")
        )

    def _mle_moments(self):
        """(mean, variance) of y at the fitted parameters, from the distribution layer."""
        if self._moments_cache is None:
            ctx = self._fit_ctx or {}
            comp = self._mle_components(
                np.asarray(self.all_params, dtype=float), self._X, ctx.get("Z")
            )
            self._moments_cache = _dist.mean_var(self.family, comp)
        return self._moments_cache

    def _mle_design(self, exog, transform=True, exog_z=None):
        """Design matrices (X, Z) and index for `exog` (or the training data if None)."""
        import pandas as pd

        ctx = self._fit_ctx or {}
        if exog is None:
            idx = getattr(self.data, "index", None) if self.data is not None else None
            return self._X, ctx.get("Z"), idx
        if self._model_spec is not None and transform and isinstance(exog, pd.DataFrame):
            X_new = np.asarray(self._model_spec.get_model_matrix(exog), dtype=float)
        else:
            X_new = np.asarray(exog, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        idx = getattr(exog, "index", None) if hasattr(exog, "index") else None
        n_new = X_new.shape[0]
        kind = getattr(self.family, "SECONDARY_KIND", None)
        Z_new = None
        if kind is not None:
            source = getattr(self, "_z_source", None)
            if exog_z is not None:
                if (
                    self._zero_model_spec is not None
                    and transform
                    and isinstance(exog_z, pd.DataFrame)
                ):
                    Z_new = np.asarray(self._zero_model_spec.get_model_matrix(exog_z), dtype=float)
                else:
                    Z_new = np.asarray(exog_z, dtype=float)
                if Z_new.ndim == 1:
                    Z_new = (
                        Z_new.reshape(n_new, -1)
                        if Z_new.size % n_new == 0
                        else Z_new.reshape(1, -1)
                    )
            elif (
                source == "formula"
                and self._zero_model_spec is not None
                and isinstance(exog, pd.DataFrame)
            ):
                Z_new = np.asarray(self._zero_model_spec.get_model_matrix(exog), dtype=float)
            elif source in (None, "default"):
                Z_new = X_new if kind == "probability" else np.ones((n_new, 1))
            else:
                raise ValueError(
                    "GLM.predict: the secondary submodel was fitted with its own Z; pass `exog_z` "
                    "(new Z rows, or a DataFrame if it came from the `| z-formula`)."
                )
            if Z_new.shape != (n_new, self._n_gamma()):
                raise ValueError(
                    f"GLM.predict: exog_z has shape {Z_new.shape}, expected ({n_new}, {self._n_gamma()})."
                )
        return X_new, Z_new, idx

    # Reference distribution (z or t) and dispersion helpers
    @property
    def use_t(self):
        """True if Wald statistics are referred to t(df_resid) (estimated dispersion) instead of N(0,1)."""
        if self._use_t_arg is not None:
            return bool(self._use_t_arg)
        return bool(getattr(self.family, "estimates_dispersion", False)) and not self._is_mle()

    def _phi(self):
        """Dispersion that scales Pearson/deviance-based diagnostics: the estimated phi for families
        with estimated dispersion, 1 for the rest (Poisson, Binomial, NB, Exponential, ...)."""
        if getattr(self.family, "estimates_dispersion", False) and self.dispersion is not None:
            return float(self.dispersion)
        return 1.0

    def _ref_quantile(self, prob):
        if self.use_t:
            from scipy.stats import t as _t

            return float(_t.ppf(prob, max(self.df_resid, 1)))
        from scipy.stats import norm as _norm

        return float(_norm.ppf(prob))

    def _ref_two_sided_p(self, stat):
        from scipy.stats import norm as _norm
        from scipy.stats import t as _t

        stat = np.abs(np.asarray(stat, dtype=float))
        if self.use_t:
            return 2.0 * _t.sf(stat, max(self.df_resid, 1))
        return 2.0 * _norm.sf(stat)

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("GLM: must call fit() before querying results.")

    # Diagnostic and inference properties / metrics

    @property
    def converged(self):
        """
        Return whether the estimation algorithm successfully converged.

        Returns
        -------
        bool
            True if optimization converged within tolerance limits, False otherwise.
        """
        self._require_fitted()
        return bool(self._converged)

    @property
    def params(self):
        """All estimated parameters (see `all_params`): beta for IRLS families;
        [beta, gamma, log(extra)] for MLE families. Names: `all_param_names`."""
        self._require_fitted()
        return self.all_params

    @property
    def bse(self):
        """Standard errors of the fitted coefficients."""
        self._require_fitted()
        return self.se

    @property
    def mu(self):
        """Fitted mean values as a pandas Series."""
        import pandas as pd

        self._require_fitted()
        idx = getattr(self.data, "index", None) if self.data is not None else None
        return pd.Series(self.fittedvalues, index=idx, name="mu")

    @property
    def aic(self):
        """Akaike Information Criterion (``None`` without a log-likelihood)."""
        self._require_fitted()
        if self.log_likelihood is None:
            return None
        return -2.0 * self.log_likelihood + 2.0 * self.n_params

    @property
    def bic(self):
        """Bayesian Information Criterion (``None`` without a log-likelihood)."""
        self._require_fitted()
        if self.log_likelihood is None:
            return None
        return -2.0 * self.log_likelihood + self.n_params * np.log(max(self.nobs, 1))

    @property
    def deviance(self):
        """Deviance of the fitted GLM model."""
        self._require_fitted()
        return self._deviance

    def _null_model(self):
        """Intercept-only refit (same family, weights, offset) used for the IRLS/NB/quasi null
        deviance and null log-likelihood. Cached until the next fit()."""
        if "glm" not in self._null_cache:
            extra = (
                {"minimizer": self.minimizer}
                if getattr(self.family, "family_type", "irls") == "quasi_custom"
                else {}
            )
            null_glm = GLM(
                np.ones((self.nobs, 1), dtype=float),
                self._y,
                family=self.family,  # copied by GLM.__init__
                weights=self._weights,
                offset=self._offset,
                tol=self.tol,
                maxit=self.maxit,
                solver=self.solver,
                big_matrix=self.big_matrix,
                **extra,
            )
            null_glm.fit()
            self._null_cache["glm"] = null_glm
        return self._null_cache["glm"]

    @property
    def null_deviance(self):
        """Deviance of the null model (intercept-only, same offset/weights).

        For IRLS, NB and quasi families this is the GLM deviance of the null fit -- the same scale as
        `deviance`, so `1 - deviance/null_deviance` and `null_deviance - deviance` are meaningful (it
        used to be -2*llnull, which is not a deviance and lives on a different scale). For MLE families
        the fitted `deviance` is -2*logLik, so the null one is -2*llnull."""
        self._require_fitted()
        family_type = getattr(self.family, "family_type", "irls")
        if family_type in ("mle", "mle_inflated", "mle_gengamma"):
            ll_null = self.llnull
            return -2.0 * ll_null if ll_null is not None else None
        return self._null_model().deviance

    @property
    def llf(self):
        """Log-likelihood of the fitted GLM model."""
        self._require_fitted()
        return self.log_likelihood

    @property
    def llnull(self):
        """Log-likelihood of the null model (intercept-only)."""
        self._require_fitted()
        family_type = getattr(self.family, "family_type", "irls")
        if family_type in ("quasi", "quasi_custom") or self.log_likelihood is None:
            return None

        X_null = np.ones((self.nobs, 1), dtype=float)

        if family_type in ("mle", "mle_inflated", "mle_gengamma"):
            # Null model = intercept-only in EVERY submodel, refitted with the SAME data context as
            # the full model: censoring indicator `d`, truncation limits, late entry. (Dropping them
            # made the null likelihood answer a different question -- e.g. ignoring censoring.)
            import warnings

            ctx = self._fit_ctx or {}
            fam0 = self.family
            if family_type == "mle_gengamma":
                fam0 = GeneralizedGamma(
                    (),
                    (),
                    (),
                    link_mu=self.family.link_mu,
                    link_sigma=self.family.link_sigma,
                    link_Q=self.family.link_Q,
                )
            Z_null = np.ones((self.nobs, 1), dtype=float) if ctx.get("Z") is not None else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                null_glm = GLM(
                    X_null,
                    self._y,
                    family=fam0,
                    weights=self._weights,
                    offset=self._offset,
                    tol=self.tol,
                    maxit=self.maxit,
                    minimizer=self.minimizer,
                    compute_se=False,
                )
                if family_type == "mle_gengamma":
                    null_glm.fit(cens=ctx.get("d"), entry=ctx.get("entry"))
                else:
                    null_glm.fit(
                        d=ctx.get("d"),
                        Z=Z_null,
                        left_limit=ctx.get("left"),
                        right_limit=ctx.get("right"),
                    )
            return null_glm.log_likelihood

        return self._null_model().log_likelihood

    @property
    def pearson_chi2(self):
        """Pearson chi-squared statistic (sum of squared Pearson residuals)."""
        self._require_fitted()
        rp = self.resid_pearson
        return float(np.sum(rp**2)) if rp is not None else None

    def pseudo_rsquared(self, kind="mcfadden"):
        """McFadden pseudo R-squared (handling both negative and positive log-likelihoods)."""
        self._require_fitted()
        if self.llf is None or self.llnull is None:
            return None

        # Adjust the sign when a continuous density yields LL > 0
        if self.llf > 0 and self.llnull > 0:
            mcfadden = 1.0 - (self.llnull / self.llf)
        else:
            mcfadden = 1.0 - (self.llf / self.llnull)

        if kind == "mcfadden":
            return mcfadden
        if kind == "mcfadden_adj":
            k = self.n_params  # every estimated parameter (beta + gamma + auxiliary scalars)
            return mcfadden - (k / abs(self.llnull))
        raise ValueError(f"Unknown pseudo R-squared kind: '{kind}'")

    @property
    def resid_pearson(self):
        """Pearson residuals."""
        self._require_fitted()
        mu = self.fittedvalues
        y = self._y
        w = self._weights
        if isinstance(self.family, (Multinomial, Ordinal)):
            # No standard scalar Pearson residual for a categorical (unordered / ordered) response: use
            # `resid_deviance` (well defined: -2 log P(Y=y_i)) or `predict_proba` for per-category diagnostics.
            return None

        if self._gengamma_supported():
            m_, v_ = self._gengamma_moments()
            with np.errstate(divide="ignore", invalid="ignore"):
                return (y - m_) / np.sqrt(v_)

        if self._mle_supported():
            # (y - E[y]) / sd(y) with E[y] and Var[y] built from ALL fitted parameters (pi, alpha,
            # sigma, nu, per-observation phi ...). For censored AFT data (LogNormal/Weibull/
            # LogLogistic with `d`) the recorded time is used as is.
            m, v = self._mle_moments()
            with np.errstate(divide="ignore", invalid="ignore"):
                return (y - m) / np.sqrt(v)

        family_type = getattr(self.family, "family_type", "irls")

        if family_type == "quasi_custom":
            V = np.array([self.family.variance_fn(m) for m in mu], dtype=float)
            return np.sqrt(w) * (y - mu) / np.sqrt(V)

        if hasattr(self.family, "code"):
            try:
                V = _glmfit.variance(int(self.family.code), self.family.params, mu)
                return np.sqrt(w) * (y - mu) / np.sqrt(V)
            except Exception:
                pass

        if hasattr(self.family, "variance"):
            V = self.family.variance(mu)
            return np.sqrt(w) * (y - mu) / np.sqrt(V)

        var_y = np.var(y - mu)
        if var_y > 0:
            return np.sqrt(w) * (y - mu) / np.sqrt(var_y)

        return y - mu

    @property
    def resid_deviance(self):
        """
        Deviance residuals, as in R's residuals(fit, type="deviance").

            r_D_i = sign(y_i - mu_i) * sqrt(w_i * d(y_i, mu_i))

        with d the family's UNIT deviance (C++ `dev_resids`, defined for every IRLS and quasi family --
        Poisson/NB/Binomial zeros, Gaussian, Tweedie, Pareto, InverseGamma, quasi* -- with no
        log-likelihood needed). It is not divided by the dispersion; use `resid_sdeviance` for the
        standardized version. (Earlier versions derived it from log-likelihood differences and gave
        NaN for every y = 0 in Poisson/NB and for all Gaussian rows, applied the prior weights twice,
        and divided by sqrt(phi) for Gamma/inverse Gaussian.)

        MLE families with a distribution layer use the likelihood: sign * sqrt(2 (l_sat - l)) with l_sat the maximum over the
        primary parameter (see mle_diag); the rest fall back to Pearson residuals.
        """
        self._require_fitted()
        mu = self.fittedvalues
        y = self._y
        w = self._weights

        d_i = self.family.dev_resids(y, mu, w)
        if d_i is not None:
            return np.sign(y - mu) * np.sqrt(np.maximum(d_i, 0.0))

        if isinstance(self.family, (Multinomial, Ordinal)):
            # No natural sign (the response is a category, not an ordered real number for Multinomial): this is
            # the (unsigned, non-negative) per-observation deviance contribution sqrt(-2 log P(Y=y_i | x_i)),
            # exactly what -2*loglik sums to (the "saturated" model trivially gives P = 1 for the true category).
            p_obs = np.clip(self._categorical_fitted_obs_prob(), 1e-300, 1.0)
            return np.sqrt(np.maximum(-2.0 * np.log(p_obs), 0.0))

        if self._gengamma_supported():
            theta = np.asarray(self.all_params, dtype=float)
            mu_i, sg_i, Q_i = self._gengamma_row_params(theta, self._X)
            ctx = self._fit_ctx or {}
            d = ctx.get("d")
            d = np.ones(y.shape[0]) if d is None else np.asarray(d, dtype=float)
            l = self._gengamma_loglik_obs(theta)
            # saturated: golden-section max over mu_i (sigma_i, Q_i, entry, d held fixed) of the same per-row objective
            lo, hi = mu_i - 20.0 * sg_i - 5.0, mu_i + 20.0 * sg_i + 5.0
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            a_, b_ = lo.copy(), hi.copy()
            f = lambda mm: self._gengamma_row_loglik(mm, sg_i, Q_i, y, d, ctx)
            x1, x2 = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
            f1, f2 = f(x1), f(x2)
            for _ in range(70):
                left = f1 > f2
                b_ = np.where(left, x2, b_)
                a_ = np.where(left, a_, x1)
                x1, x2 = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
                f1, f2 = f(x1), f(x2)
            sat = np.maximum(f(0.5 * (a_ + b_)), l)
            mean_ = self._gengamma_moments()[0]
            sign_ = np.sign(y - mean_)
            sign_ = np.where(sign_ == 0, 1.0, sign_)
            return sign_ * np.sqrt(np.maximum(2.0 * (sat - l), 0.0))

        if self._mle_supported():
            # deviance residual from the per-observation log-likelihood: sign * sqrt(2 (l_sat - l)), with l_sat the
            # maximum over the primary parameter and the others held at their fitted values. For AFT models with
            # censoring the sign is that of the martingale residual (as R's survreg).
            comp, d = self._mle_comp_full()
            fam = self.family
            sign_ref = None
            if type(fam).__name__ in _dist.SURVIVAL_FAMILIES and d is not None:
                sign_ref = _mdiag.martingale(fam, comp, y, d)
            return _mdiag.deviance_residuals(fam, comp, y, self._mle_moments()[0], d, sign_ref)

        rp = self.resid_pearson
        return rp if rp is not None else (y - mu)

    # MLE families: components, likelihood, quantile / survival residuals, influence
    def _mle_comp_full(self):
        """(components at the fitted parameters, event indicator d or None)."""
        ctx = self._fit_ctx or {}
        comp = self._mle_components(np.asarray(self.all_params, dtype=float), self._X, ctx.get("Z"))
        return comp, ctx.get("d")

    def _mle_loglik_obs_fn(self):
        """theta -> per-observation log-likelihood l_i(theta), in the order of `all_params`."""
        ctx = self._fit_ctx or {}
        X, Z, d, y, fam = self._X, ctx.get("Z"), ctx.get("d"), self._y, self.family
        return lambda th: _dist.logpdf(
            fam, self._mle_components(np.asarray(th, dtype=float), X, Z), y, d
        )

    @property
    def loglik_obs(self):
        """Per-observation log-likelihood at the fit (MLE families with a distribution layer): the proper
        log-density (log-survival for censored AFT rows). NOTE: the solver's Weibull / LogLogistic likelihood omits
        the Jacobian term -log t, so `log_likelihood` = sum(loglik_obs) + sum(d log t) for those two families
        (see mle_dist.LL_OMITS_LOG_T)."""
        self._require_fitted()
        if self._gengamma_supported():
            return self._gengamma_loglik_obs(np.asarray(self.all_params, dtype=float))
        if not self._mle_supported():
            return None
        return self._mle_loglik_obs_fn()(np.asarray(self.all_params, dtype=float))

    def resid_quantile(self, seed=None):
        """Randomized quantile residuals (Dunn & Smyth 1996): N(0,1) if the model is right, for ANY family with a
        computable distribution -- the standard IRLS families (Gaussian, Binomial, Poisson, Exponential, Gamma,
        InverseGaussian, NegativeBinomial), Tweedie and the MLE families. Continuous laws are deterministic;
        counts, grouped binomial data and Tweedie zeros are randomized within the jump of the distribution
        function (`seed` makes them reproducible). For censored AFT rows the residual is undefined (NaN): use the
        Cox-Snell / martingale residuals there. None for families without a computable distribution function
        (Pareto, InverseGamma, quasi, CustomFamily)."""
        self._require_fitted()
        if self._mle_supported():
            comp, d = self._mle_comp_full()
            r = _mdiag.quantile_residuals(
                self.family, comp, self._y, rng=np.random.default_rng(seed)
            )
            if d is not None and type(self.family).__name__ in _dist.SURVIVAL_FAMILIES:
                r = np.where(np.asarray(d) > 0.5, r, np.nan)
            return r
        from . import quantile as _qr

        return _qr.quantile_residuals(
            self.family,
            self._y,
            self.fittedvalues,
            self._phi(),
            self._weights,
            rng=np.random.default_rng(seed),
        )

    @property
    def has_quantile_residuals(self):
        """Whether `resid_quantile` returns values: every MLE family, the standard IRLS families with a known
        conditional distribution, and Tweedie. False for NLS-like models, Pareto / InverseGamma, the quasi
        families, and grouped binomial data with non-integer trial counts."""
        self._require_fitted()
        if self._mle_supported():
            return True
        from . import quantile as _qr

        return bool(_qr._is_available(self.family, self._weights))

    def _survival_comp(self):
        """(components, event indicator, entry) for the censored survival families.

        Includes GeneralizedGamma: its parameters live in three submodels rather
        than the MLE component namespace, so they are packed into a small
        namespace that ``mle_diag.log_survival`` understands.
        """
        self._require_fitted()
        ctx = self._fit_ctx or {}
        n = self._y.shape[0]
        raw_d = ctx.get("d")
        d = np.ones(n) if raw_d is None else np.asarray(raw_d, dtype=float).reshape(-1)
        raw_entry = ctx.get("entry")
        entry = None if raw_entry is None else np.asarray(raw_entry, dtype=float).reshape(-1)
        if self._gengamma_supported():
            mu, sigma, Q = self._gengamma_row_params(
                np.asarray(self.all_params, dtype=float), self._X
            )
            comp = SimpleNamespace(mu=mu, sigma=sigma, Q=Q)
            return comp, d, entry
        if self._mle_supported() and type(self.family).__name__ in _dist.SURVIVAL_FAMILIES:
            comp, _ = self._mle_comp_full()
            return comp, d, entry
        return None

    @property
    def resid_cox_snell(self):
        """Cox-Snell residuals -log S(t_i | T > entry_i), unit exponential under a correct model.

        Available for LogNormal / Weibull / LogLogistic and GeneralizedGamma;
        ``None`` for other families.  Censored rows are censored draws, not NaN.
        """
        self._require_fitted()
        sc = self._survival_comp()
        return None if sc is None else _mdiag.cox_snell(self.family, sc[0], self._y, sc[2])

    @property
    def resid_martingale(self):
        """Martingale residuals d_i - r_CS,i for every censored-survival row."""
        self._require_fitted()
        sc = self._survival_comp()
        return None if sc is None else _mdiag.martingale(self.family, sc[0], self._y, sc[1], sc[2])

    @property
    def resid_deviance_martingale(self):
        """Deviance-martingale residuals sign(m) sqrt(-2 [m + d log(d - m)]) (symmetrizes the martingale residual)."""
        self._require_fitted()
        sc = self._survival_comp()
        return (
            None
            if sc is None
            else _mdiag.deviance_martingale(self.family, sc[0], self._y, sc[1], sc[2])
        )

    def _mle_influence(self):
        """Score / information based leverage, leave-one-out change and Cook's distance (see mle_diag)."""
        if self._infl_cache is None:
            if self._gengamma_supported():
                fn = self._gengamma_loglik_obs
            elif self._categorical_supported():
                fn = self._categorical_loglik_obs
            else:
                fn = self._mle_loglik_obs_fn()
            self._infl_cache = _mdiag.information_measures(
                fn, np.asarray(self.all_params, dtype=float)
            )
        return self._infl_cache

    @property
    def dfbeta_all(self):
        """One-step leave-one-out change of ALL parameters, theta_hat - theta_hat_(i) (n x len(all_params)), in the
        order of `all_param_names` (MLE families with a distribution layer)."""
        self._require_fitted()
        supported = (
            self._mle_supported() or self._gengamma_supported() or self._categorical_supported()
        )
        return self._mle_influence()["dfbeta"] if supported else None

    def _loo_sigma(self):
        """sigma_(i): leave-one-out estimate of the scale, exactly R's influence(fit)$sigma (lm.influence on the
        final weighted-least-squares fit):  sigma_i^2 = (chi2_P - r_P,i^2 / (1 - h_i)) / (n_ok - p - 1)."""
        rp = self.resid_pearson
        h = self.hat_matrix_diag
        if rp is None or h is None:
            return None
        ok = self._weights > 0
        df = int(ok.sum()) - int(self.rank) - 1
        if df <= 0:
            return None
        chi2 = float(np.sum(rp[ok] ** 2))
        return np.sqrt(np.maximum(chi2 - rp**2 / np.maximum(1.0 - h, 1e-12), 0.0) / df)

    @property
    def resid_rstudent(self):
        """Externally studentized residuals, as R's rstudent.glm:

            r*_i = sign(r_D,i) sqrt( r_D,i^2 + h_i r_P,i^2 / (1 - h_i) ),   divided by sigma_(i) when the
            dispersion is estimated (Gaussian, Gamma, inverse Gaussian, quasi*).

        Exact leave-one-out t residual for a Gaussian model (verified against refits). R divides by sigma_(i)
        for every family except binomial and poisson; here it is done exactly when the dispersion is
        estimated, so NB (fixed dispersion 1) is not divided. None for MLE families."""
        self._require_fitted()
        rd, rp, h = self.resid_deviance, self.resid_pearson, self.hat_matrix_diag
        if h is None or self._is_mle():
            return None
        r = np.sign(rd) * np.sqrt(rd**2 + h * rp**2 / np.maximum(1.0 - h, 1e-12))
        if getattr(self.family, "estimates_dispersion", False):
            sig = self._loo_sigma()
            if sig is None:
                return None
            r = r / sig
        return r

    @property
    def dffits(self):
        """DFFITS as R's dffits(): r_D sqrt(h) / (sigma_(i) (1 - h)). Exact for a Gaussian model. None for MLE."""
        self._require_fitted()
        h, sig, rd = self.hat_matrix_diag, self._loo_sigma(), self.resid_deviance
        if h is None or sig is None or self._is_mle():
            return None
        return rd * np.sqrt(h) / (sig * np.maximum(1.0 - h, 1e-12))

    @property
    def dfbetas(self):
        """Scaled dfbeta as R's dfbetas(): dfbeta_ij / (sigma_(i) sqrt(((X'WX)^-1)_jj)). Aliased columns are NaN."""
        self._require_fitted()
        D, sig = self.dfbeta, self._loo_sigma()
        if D is None or sig is None or self._is_mle():
            return None
        W = self._working_weights()
        X = self._X_active
        xxi = np.linalg.pinv(X.T @ (W[:, None] * X))
        scale = np.sqrt(np.diag(xxi))
        if not self.rank_deficient:
            return D / (sig[:, None] * scale[None, :])
        out = np.full_like(D, np.nan)
        out[:, self._active_cols] = D[:, self._active_cols] / (sig[:, None] * scale[None, :])
        return out

    # Quasi-likelihood based criteria
    def _eql(self):
        """Extended quasi-likelihood (Nelder & Pregibon 1987) for families with a variance function but no
        likelihood: quasi*, Pareto, InverseGamma.
            Q+ = -1/2 sum_i [ log(2 pi phi V(y~_i) / w_i) + w_i d(y_i, mu_i) / phi ],   phi_hat = sum w d / n_ok,
        with y~ = y (+1/6 where y = 0; the [0,1] response of quasi-binomial is shrunk to (y+1/6)/(4/3)) inside V."""
        fam = self.family
        ft = getattr(fam, "family_type", "irls")
        if not (ft in ("quasi", "quasi_custom") or isinstance(fam, (Pareto, InverseGamma))):
            return None
        y, w, mu = self._y, self._weights, self.fittedvalues
        wd = fam.dev_resids(y, mu, w)
        if wd is None:
            return None
        ok = w > 0
        n_ok = int(ok.sum())
        phi = float(np.sum(wd[ok])) / max(n_ok, 1)
        if n_ok == 0 or not phi > 0:
            return None
        if isinstance(fam, QuasiBinomial):
            yt = (y + 1.0 / 6.0) / (4.0 / 3.0)
        else:
            yt = np.where(y > 0, y, 1.0 / 6.0)
        if ft == "quasi_custom":
            V = np.array([max(float(fam.variance_fn(v)), 1e-12) for v in yt])
        else:
            V = np.asarray(
                _glmfit.variance(
                    int(fam.code), fam.params, np.ascontiguousarray(yt, dtype=float)
                )
            )
        val = -0.5 * np.sum(np.log(2.0 * np.pi * phi * V[ok] / w[ok]) + wd[ok] / phi)
        return float(val) if np.isfinite(val) else None

    @property
    def eql_loglik(self):
        """Extended quasi-log-likelihood (Nelder-Pregibon) of quasi*, Pareto and InverseGamma models, which have no
        true likelihood (`log_likelihood` is None). A pseudo-likelihood: use it to compare models of the SAME
        family and variance function, not against true likelihoods."""
        self._require_fitted()
        return self._eql()

    @property
    def aic_eql(self):
        """-2 EQL + 2 (k + 1): AIC computed from the extended quasi-likelihood (dispersion counted as a parameter)."""
        ll = self.eql_loglik
        return None if ll is None else -2.0 * ll + 2.0 * (self._k_free + 1)

    @property
    def qaic(self):
        """QAIC = -2 logL / c_hat + 2 (k + 1) (Burnham & Anderson) for QuasiPoisson / QuasiBinomial, with logL the
        Poisson / Binomial log-likelihood at mu_hat and c_hat the Pearson dispersion. None for other families."""
        self._require_fitted()
        fam = self.family
        if isinstance(fam, QuasiPoisson):
            ll_obs = Poisson().loglike_obs(self._y, self.fittedvalues, self._weights, 1.0)
        elif isinstance(fam, QuasiBinomial):
            ll_obs = Binomial().loglike_obs(self._y, self.fittedvalues, self._weights, 1.0)
        else:
            return None
        c = self.dispersion
        if ll_obs is None or c is None or not c > 0 or not np.all(np.isfinite(ll_obs)):
            return None
        return float(-2.0 * np.sum(ll_obs) / c + 2.0 * (self._k_free + 1))

    @property
    def resid_sdeviance(self):
        """Standardized deviance residuals r_D / sqrt(phi (1 - h)) (R's rstandard(type="deviance"))."""
        self._require_fitted()
        rd = self.resid_deviance
        h = self.hat_matrix_diag
        if rd is None or h is None:
            return None
        return rd / np.sqrt(self._phi() * np.maximum(1.0 - h, 1e-12))

    @property
    def resid_response(self):
        """Raw response residuals (y - mu)."""
        self._require_fitted()
        return self._y - self.fittedvalues

    @property
    def resid_working(self):
        """Working residuals."""
        self._require_fitted()
        if getattr(self.family, "family_type", "irls") not in (
            "irls",
            "quasi",
            "nb_two_step",
            "quasi_custom",
        ):
            return None
        mu = self.fittedvalues
        eta = self.linear_predictors
        h = 1e-6 * np.maximum(1.0, np.abs(eta))
        dmu_deta = (self.family.linkinv(eta + h) - self.family.linkinv(eta - h)) / (2.0 * h)
        dmu_deta = np.where(np.abs(dmu_deta) < 1e-12, 1e-12, dmu_deta)
        return (self._y - mu) / dmu_deta

    @property
    def resid_anscombe(self):
        """
        Anscombe residuals  sqrt(w) (A(y) - A(mu)) / V(mu)^(1/6),  A(x) = int V(t)^(-1/3) dt.

        Closed forms for every built-in variance function -- Gaussian, Poisson, Binomial, Gamma / Exponential /
        Pareto / InverseGamma (V = mu^2), inverse Gaussian, NEGATIVE BINOMIAL (V = mu + mu^2/theta) and TWEEDIE
        (V = mu^p) -- and adaptive quadrature for user-supplied variances (QuasiCustom, CustomFamily).
        (NB and Tweedie used to fall back to Pearson residuals.) Checked against int_mu^y V^(-1/3) dt.
        MLE families have no variance function of this kind: they return Pearson residuals, which use all
        the fitted parameters.
        """
        self._require_fitted()
        mu = self.fittedvalues
        y = self._y
        w = self._weights
        fam = self.family
        kw = {}
        if isinstance(fam, (Gaussian, QuasiGaussian)):
            kind = "gaussian"
        elif isinstance(fam, (Binomial, QuasiBinomial)):
            kind = "binomial"
        elif isinstance(fam, (Poisson, QuasiPoisson)):
            kind = "poisson"
        elif isinstance(fam, (Gamma, QuasiGamma, Exponential, Pareto, InverseGamma)):
            kind = "gamma"
        elif isinstance(fam, (InverseGaussian, QuasiInverseGaussian)):
            kind = "invgauss"
        elif isinstance(fam, NegativeBinomial):
            kind, kw = "nb", {"theta": float(fam.params.theta)}
        elif isinstance(fam, Tweedie):
            kind, kw = "tweedie", {"p": float(fam.params.var_power)}
        elif (
            isinstance(fam, (QuasiCustom, CustomFamily))
            and getattr(fam, "variance_fn", None) is not None
        ):
            return _gdiag.anscombe_residuals_numeric(fam.variance_fn, y, mu, w)
        else:
            return self.resid_pearson
        return _gdiag.anscombe_residuals(kind, y, mu, w, **kw)

    @property
    def resid_sspearson(self):
        """Standardized Pearson residuals."""
        self._require_fitted()
        rp = self.resid_pearson
        h = self.hat_matrix_diag
        if rp is None or h is None:
            return None
        # r_P / sqrt(phi (1 - h)): for families with estimated dispersion (Gaussian/Gamma/
        # inverse Gaussian/quasi*) phi is not 1, so it must divide the standardized residuals
        return rp / np.sqrt(self._phi() * np.maximum(1.0 - h, 1e-12))

    @property
    def scale(self):
        """
        Return the estimated dispersion or scale parameter of the model.

        Calculates Pearson's chi-squared statistic divided by residual degrees
        of freedom for families with estimated dispersion under IRLS, or retrieves
        the dispersion parameter computed by the specific engine.
        """
        self._require_fitted()
        family_type = getattr(self.family, "family_type", "irls")

        if family_type in ("quasi", "quasi_custom"):
            return self.dispersion

        if self._mle_supported() or self._gengamma_supported():
            # Pearson chi2 / residual df with the family's full variance (~1 when well specified);
            # `dispersion` (= 1/extra_param) is not a scale for most MLE families.
            rp = self.resid_pearson
            rp = rp[np.isfinite(rp)]
            return float(np.sum(rp**2) / max(self.df_resid, 1)) if rp.size else 1.0

        if family_type in ("mle", "mle_inflated", "mle_gengamma", "nb_two_step"):
            return self.dispersion if self.dispersion is not None else 1.0

        if getattr(self.family, "DISPERSION", "fixed") == "estimate":
            if self.dispersion is not None:
                return self.dispersion
            rp = self.resid_pearson
            if rp is not None:
                chisq = np.sum(rp**2)
                df_res = max(self.df_resid, 1)
                return chisq / df_res

        return 1.0

    @property
    def fittedvalues(self):
        """E[y | x, z] at the fitted parameters. For MLE families this uses ALL parameters
        (pi, alpha, sigma, phi_i, truncation limits ...), e.g. (1-pi)*lambda for ZeroInflated*;
        `predict(which="location")` gives the old g^-1(X beta) (count mean / location)."""
        self._require_fitted()
        if isinstance(self.family, (Multinomial, Ordinal)):
            # P(Y = y_i | x_i) at the fit: the discrete analogue of "how well is THIS observation fitted";
            # use `predict_proba` for the full (n, K) probability matrix.
            return self._categorical_fitted_obs_prob()
        if self._gengamma_supported():
            return self._gengamma_moments()[0]
        if self._mle_supported():
            return self._mle_moments()[0]
        if self._engine is not None:
            return self._engine.get_mu()
        return self.family.linkinv(self.linear_predictors)

    @property
    def linear_predictors(self):
        """Linear predictor ``eta = X @ beta + offset`` for the fitted data."""
        self._require_fitted()
        if self._engine is not None:
            return self._engine.get_eta()
        if self.rank_deficient:
            return self._X_active @ self.beta[self._active_cols] + self._offset
        return self._X @ self.beta + self._offset

    @property
    def param_names(self):
        """Coefficient labels, including secondary-submodel parameters."""
        self._require_fitted()
        names = self.family.beta_names(self._feature_names, len(self.beta))
        if names is None:
            return list(self._feature_names)
        if len(names) != len(self.beta):
            return [f"x{j}" for j in range(len(self.beta))]
        return names

    @property
    def nobs(self):
        """Number of observations used in the fit."""
        self._require_fitted()
        return self._y.shape[0]

    @property
    def df_resid(self):
        """Residual degrees of freedom of the fitted model."""
        self._require_fitted()
        if self._is_mle():
            return self.nobs - self._k_free  # beta + gamma + auxiliary scalars
        return self.nobs - (self.rank if self.rank is not None else 0)

    @property
    def exog(self):
        """Design matrix used in the fit."""
        return self._X

    @property
    def endog(self):
        """Response vector used in the fit."""
        return self._y

    @property
    def exog_names(self):
        """Names of the design-matrix columns."""
        return list(self._feature_names)

    @property
    def endog_names(self):
        """Name of the response (left-hand side of the formula)."""
        if self.formula:
            return self.formula.split("~", 1)[0].strip()
        return "y"

    def _working_weights(self):
        """IRLS working weights W_i = w_i (dmu/deta)^2 / V(mu_i) at the fit (None for MLE families)."""
        self._require_fitted()
        ft = getattr(self.family, "family_type", "irls")
        if self._engine is not None:
            return np.asarray(self._engine.get_w())
        if ft in ("quasi", "nb_two_step", "quasi_custom"):
            mu = self.fittedvalues
            if ft == "quasi_custom":
                V = np.array([self.family.variance_fn(m) for m in mu])
            else:
                V = _glmfit.variance(int(self.family.code), self.family.params, mu)
            eta = self.linear_predictors
            h = 1e-6 * np.maximum(1.0, np.abs(eta))
            dmu_deta = (self.family.linkinv(eta + h) - self.family.linkinv(eta - h)) / (2.0 * h)
            return self._weights * dmu_deta**2 / V
        return None

    @property
    def hat_matrix_diag(self):
        """Leverage. IRLS / quasi: the hat value w_i x_i'(X'WX)^-1 x_i. MLE families with a distribution layer:
        the information leverage tr(I^-1 I_i) over ALL parameters (beta, secondary submodel, auxiliary scalars),
        which sums to the number of parameters and equals the hat value for a GLM. Because it uses the OBSERVED
        per-observation information, families where that matrix can be indefinite (e.g. the zeros of a zero-inflated
        model) may give leverages outside [0, 1]; the sum is always p."""
        self._require_fitted()
        if self._mle_supported() or self._gengamma_supported() or self._categorical_supported():
            return self._mle_influence()["hat"]
        w = self._working_weights()
        if w is None:
            return None
        X = self._X_active
        XtWX = X.T @ (w[:, None] * X)
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        return w * np.einsum("ij,jk,ik->i", X, XtWX_inv, X)

    @property
    def dfbeta(self):
        """One-step approximation of beta_hat - beta_hat_(i), the change in the coefficients when
        observation i is deleted (n x p; R's dfbeta / influence()$coefficients):

            (X'WX)^-1 x_i W_i r^W_i / (1 - h_i)      with r^W_i = (y_i - mu_i) / (dmu/deta)_i.

        Exact for a Gaussian identity model; a very close approximation for other GLMs. Aliased
        (collinear) coefficients get NaN. None for MLE families."""
        self._require_fitted()
        if self._mle_supported() or self._gengamma_supported() or self._categorical_supported():
            return self._mle_influence()["dfbeta"][:, self._beta_positions()]  # the beta columns
        W = self._working_weights()
        if W is None:
            return None
        h = self.hat_matrix_diag
        rw = self.resid_working
        X = self._X_active
        try:
            inv = np.linalg.inv(X.T @ (W[:, None] * X))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(X.T @ (W[:, None] * X))
        factor = W * rw / np.maximum(1.0 - h, 1e-12)
        D_act = (X * factor[:, None]) @ inv  # rows: (X'WX)^-1 x_i  (symmetric) times the scalar
        if not self.rank_deficient:
            return D_act
        D = np.full((D_act.shape[0], self._X.shape[1]), np.nan)
        D[:, self._active_cols] = D_act
        return D

    @property
    def dffits_internal(self):
        """DFFITS with the internally standardized Pearson residual: r_i sqrt(h_i / (1 - h_i)).
        (Uses the full-sample dispersion, like statsmodels' dffits_internal.) None for MLE families."""
        self._require_fitted()
        r = self.resid_sspearson
        h = self.hat_matrix_diag
        if r is None or h is None:
            return None
        return r * np.sqrt(h / np.maximum(1.0 - h, 1e-12))

    @property
    def tvalues(self):
        """
        Compute z-stat (or t-stat) values for testing parameter significance.

        Calculates z = theta / se(theta) over ALL estimated parameters: for MLE families
        that is beta, the secondary submodel (gamma) and the auxiliary scalars
        (estimation/log scale, e.g. 'log_alpha' tests log(alpha) = 0), in `all_param_names`
        order. For IRLS families it is beta, as before.

        Returns
        -------
        pandas.Series
            Series of z-values indexed by parameter names.
        """
        import pandas as pd

        self._require_fitted()
        if self.se is None:
            raise RuntimeError("GLM.tvalues: Standard errors were not computed for this model.")
        theta, names, se, _ = self._inference_block()
        return pd.Series(theta / se, index=names, name="tvalues")

    @property
    def pvalues(self):
        """
        Compute two-tailed p-values of the Wald statistics.

        Reference distribution: t with `df_resid` degrees of freedom when the dispersion is estimated
        (Gaussian, Gamma, inverse Gaussian, every quasi* family -- see `use_t`), N(0,1) otherwise, as in
        R's summary.glm. (It was always normal, which is anti-conservative for small samples.)

        Returns
        -------
        pandas.Series
            Series of p-values indexed by parameter names.
        """
        import pandas as pd

        self._require_fitted()
        t_vals = self.tvalues
        p_vals = self._ref_two_sided_p(t_vals.values)
        return pd.Series(p_vals, index=t_vals.index, name="pvalues")

    def cooks_distance(self):
        """Cook's distance. IRLS / quasi: r_P^2 h / (phi p (1-h)^2). MLE families with a distribution layer: the
        likelihood-distance version  Delta_i' I Delta_i / p  with Delta_i the one-step change of all parameters."""
        self._require_fitted()
        if self._mle_supported() or self._gengamma_supported() or self._categorical_supported():
            return self._mle_influence()["cooks"]
        rp = self.resid_pearson
        h = self.hat_matrix_diag
        if rp is None or h is None:
            return None
        return (rp**2 * h) / (self._phi() * self.rank * (1.0 - h) ** 2)

    def get_influence(self):
        """Return diagnostic measures for influence and leverage points."""
        self._require_fitted()
        return GLMInfluence(self)

    def cov_params(self):
        """Variance-covariance matrix of parameters."""
        self._require_fitted()
        return self.vcov

    def wald_test(self, r_matrix, q_matrix=None):
        """
        Wald test for the linear hypothesis R * theta = q, where theta = `all_params`
        (beta, then the secondary submodel gamma, then the auxiliary scalars on their
        estimation scale). `r_matrix` may have `len(all_params)` columns (any parameter
        can be tested) or only `len(beta)` columns (then it refers to beta, as before).
        The covariance used is the block of `vcov` matching those parameters -- previously a
        beta-only R was multiplied against the full vcov, which fails for MLE families.
        """
        self._require_fitted()
        theta, _, _, V = self._inference_block()
        if V is None:
            raise RuntimeError(
                "GLM.wald_test: the covariance matrix was not computed for this model."
            )
        R = np.atleast_2d(np.asarray(r_matrix, dtype=float))
        n_full, n_beta = theta.shape[0], len(self.beta)
        if R.shape[1] == n_full:
            R_full = R
        elif R.shape[1] == n_beta:
            R_full = np.zeros((R.shape[0], n_full))
            R_full[:, self._beta_positions()] = R
        else:
            raise ValueError(
                f"GLM.wald_test: r_matrix has {R.shape[1]} columns; expected {n_full} "
                f"(all parameters: {', '.join(self.all_param_names)}) or {n_beta} (beta only)."
            )
        q = np.asarray(q_matrix, dtype=float) if q_matrix is not None else np.zeros(R_full.shape[0])
        R = R_full
        diff = R @ theta - q
        cov_r = R @ V @ R.T
        stat = float(diff.T @ np.linalg.solve(cov_r, diff))
        df = R.shape[0]
        if self.use_t:
            # estimated dispersion: W / q ~ F(q, df_resid) (as statsmodels' use_t Wald and R's anova F test)
            from scipy.stats import f as f_dist

            df_den = max(self.df_resid, 1)
            f_stat = stat / df
            return {
                "statistic": f_stat,
                "df": df,
                "df_denom": df_den,
                "pvalue": float(f_dist.sf(f_stat, df, df_den)),
                "distribution": "F",
                "chi2": stat,
            }
        from scipy.stats import chi2

        pvalue = float(chi2.sf(stat, df))
        return {"statistic": stat, "df": df, "pvalue": pvalue, "distribution": "chi2"}

    def anova(self, other, test="auto"):
        """Compare this model with a NESTED one (R's anova(fit0, fit1, test=...)).

        The model with more parameters is taken as the full one (the caller is responsible for the
        models being nested and fitted to the same data). Deviance-based for IRLS / NB / quasi families,
        -2 logLik for MLE families.

        test : {"auto", "Chisq", "F", "LRT"}
            "auto" -> "F" when the dispersion is estimated (Gaussian, Gamma, inverse Gaussian, quasi*: the
            deviance difference is divided by the full model's Pearson dispersion and referred to
            F(df, df_resid_full), as R does), otherwise "Chisq" (likelihood-ratio / deviance test).
            "LRT" is an alias of "Chisq".

        Returns a pandas DataFrame with the R layout: Resid. Df, Resid. Dev, Df, Deviance, statistic and p.
        """
        import pandas as pd
        from scipy.stats import chi2
        from scipy.stats import f as f_dist

        self._require_fitted()
        other._require_fitted()
        if self._y.shape != other._y.shape or not np.allclose(self._y, other._y):
            raise ValueError("GLM.anova: the two models must be fitted to the same response.")
        if self._is_mle() != other._is_mle():
            raise ValueError(
                "GLM.anova: cannot compare an MLE-family model with an IRLS/quasi one."
            )

        def _resid_dev(m):
            if m._is_mle():
                return -2.0 * m.log_likelihood if m.log_likelihood is not None else None
            return m.deviance

        big, small = (self, other) if self.df_resid <= other.df_resid else (other, self)
        d_big, d_small = _resid_dev(big), _resid_dev(small)
        if d_big is None or d_small is None:
            raise ValueError(
                "GLM.anova: a model has no deviance / log-likelihood (e.g. Tweedie with p > 2)."
            )
        ddf = small.df_resid - big.df_resid
        ddev = d_small - d_big
        if ddf <= 0:
            raise ValueError(
                "GLM.anova: the models have the same number of parameters; nothing to test."
            )

        if test == "auto":
            test = (
                "F"
                if (not big._is_mle() and getattr(big.family, "estimates_dispersion", False))
                else "Chisq"
            )
        if test == "LRT":
            test = "Chisq"
        if test == "F":
            phi = big._phi()
            stat = ddev / ddf / phi
            pval = float(f_dist.sf(stat, ddf, max(big.df_resid, 1)))
            stat_name, p_name = "F", "Pr(>F)"
        elif test == "Chisq":
            phi = big._phi() if not big._is_mle() else 1.0
            stat = ddev / phi
            pval = float(chi2.sf(stat, ddf))
            stat_name, p_name = "Chisq", "Pr(>Chi)"
        else:
            raise ValueError("GLM.anova: test must be 'auto', 'Chisq', 'F' or 'LRT'.")

        table = pd.DataFrame(
            {
                "Resid. Df": [small.df_resid, big.df_resid],
                "Resid. Dev": [d_small, d_big],
                "Df": [np.nan, ddf],
                "Deviance": [np.nan, ddev],
                stat_name: [np.nan, stat],
                p_name: [np.nan, pval],
            },
            index=["small", "big"],
        )
        return table

    def conf_int(self, alpha=0.05, cols=None, cov_type=None, cov_kwds=None):
        """
        Calculate Wald confidence intervals for ALL estimated parameters (beta, secondary
        submodel gamma and auxiliary scalars on the estimation scale for MLE families; beta
        for IRLS families).

        Parameters
        ----------
        alpha : float, optional
            Significance level (default 0.05 for 95% confidence intervals).
        cols : array-like or slice, optional
            Columns/indices of coefficients to include.
        cov_type : str, optional
            Covariance estimator. ``None`` uses the model-based covariance. Robust
            options include ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"`` and
            ``"cluster"`` (requires ``cov_kwds={"groups": ...}``).
        cov_kwds : dict, optional
            Extra arguments for robust covariance estimation.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing lower and upper confidence limits.
        """
        import pandas as pd

        self._require_fitted()
        theta, names, se_theta, _ = self._inference_params(cov_type=cov_type, cov_kwds=cov_kwds)
        if se_theta is None:
            raise RuntimeError("GLM.conf_int: Standard errors were not computed for this model.")
        q = self._ref_quantile(1.0 - alpha / 2.0)
        lower = theta - q * se_theta
        upper = theta + q * se_theta
        col_lower = f"{alpha / 2.0 * 100:.1f}%"
        col_upper = f"{(1.0 - alpha / 2.0) * 100:.1f}%"
        df = pd.DataFrame(
            np.column_stack([lower, upper]), index=names, columns=[col_lower, col_upper]
        )
        if cols is not None:
            return (
                df.iloc[cols] if isinstance(cols, (int, slice, list, np.ndarray)) else df.loc[cols]
            )
        return df

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a diagnostic plot for this fitted GLM.

        See :mod:`statcraft.plots` for the supported ``kind`` names; the
        default is the adaptive 2x2 diagnostics panel.
        """
        from ..plots import plot as _plot

        return _plot(self, kind=kind, ax=ax, **kwargs)

    def coef_table(self, alpha=0.05, exponentiate=False, cov_type=None, cov_kwds=None):
        """Coefficients, standard errors, test statistics, p-values and confidence intervals.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for the confidence interval.
        exponentiate : bool, default False
            If ``True``, report ``exp(coef)`` and exponentiate the confidence
            interval bounds. Standard errors are transformed by the delta
            method (``exp(coef) * se``); the test statistic and p-value remain
            on the original scale.
        cov_type : str, optional
            Covariance estimator. ``None`` uses the model-based covariance.
            Robust options include ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"``
            and ``"cluster"`` (requires ``cov_kwds={"groups": ...}``).
        cov_kwds : dict, optional
            Extra arguments for robust covariance estimation.

        Returns
        -------
        pandas.DataFrame
            Table indexed by parameter name with columns ``coef``/``std err``/stat/
            p-value and confidence interval bounds.
        """
        import pandas as pd

        self._require_fitted()
        theta, names, se_theta, _ = self._inference_params(cov_type=cov_type, cov_kwds=cov_kwds)
        if se_theta is None:
            raise RuntimeError("GLM.coef_table: Standard errors were not computed for this model.")
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = theta / se_theta
        pvals = self._ref_two_sided_p(stat)
        q = self._ref_quantile(1.0 - alpha / 2.0)
        lower = theta - q * se_theta
        upper = theta + q * se_theta
        stat_name = "t" if self.use_t else "z"
        p_name = "P>|t|" if self.use_t else "P>|z|"
        coef_name = "exp(coef)" if exponentiate else "coef"
        col_lower = f"[{alpha / 2.0:.3f}"
        col_upper = f"{1.0 - alpha / 2.0:.3f}]"
        coef = np.asarray(theta, dtype=float)
        se_out = np.asarray(se_theta, dtype=float)
        if exponentiate:
            ecoef = np.exp(coef)
            se_out = ecoef * se_out
            lower, upper = np.exp(lower), np.exp(upper)
            coef = ecoef
        table = pd.DataFrame(
            {
                coef_name: coef,
                "std err": se_out,
                stat_name: stat,
                p_name: pvals,
                col_lower: lower,
                col_upper: upper,
            },
            index=names,
        )
        return table

    def summary_frame(self, alpha=0.05, exponentiate=False, cov_type=None, cov_kwds=None):
        """Alias of :meth:`coef_table` returning the coefficient table as a DataFrame."""
        return self.coef_table(
            alpha=alpha, exponentiate=exponentiate, cov_type=cov_type, cov_kwds=cov_kwds
        )

    def to_dataframe(self, alpha=0.05, exponentiate=False, cov_type=None, cov_kwds=None):
        """Alias of :meth:`coef_table` returning the coefficient table as a DataFrame."""
        return self.coef_table(
            alpha=alpha, exponentiate=exponentiate, cov_type=cov_type, cov_kwds=cov_kwds
        )

    def odds_ratios(self, alpha=0.05, cov_type=None, cov_kwds=None):
        """Exponentiated coefficients (odds ratios) with confidence intervals.

        Intended for logit-linked models, but applies ``exp`` to any model's
        parameters. Test statistics and p-values are reported on the original scale.
        """
        return self.coef_table(
            alpha=alpha, exponentiate=True, cov_type=cov_type, cov_kwds=cov_kwds
        )

    def irr(self, alpha=0.05, cov_type=None, cov_kwds=None):
        """Exponentiated coefficients (incidence rate ratios) with confidence intervals.

        Intended for log-linked count models, but applies ``exp`` to any model's
        parameters. Test statistics and p-values are reported on the original scale.
        """
        return self.coef_table(
            alpha=alpha, exponentiate=True, cov_type=cov_type, cov_kwds=cov_kwds
        )

    def predict(
        self, exog=None, transform=True, linear=False, which="mean", exog_z=None, offset=None
    ):
        """
        Generate predictions for new exogenous data.

        For MLE families the default prediction is E[y] computed from ALL fitted
        parameters (e.g. ``(1-pi)*lambda`` for ZeroInflated*, ``pi*E[Y|Y>0]`` for
        Hurdle*, ``exp(m + sigma^2/2)`` for LogNormal), not just ``g^-1(X beta)``.

        Parameters
        ----------
        exog : array-like or pandas.DataFrame, optional
            New covariates for prediction. If None, predicts on training data.
        transform : bool, default True
            If True and model was created from formula, parses exog through
            the formulaic spec.
        linear : bool, default False
            If True, returns linear predictor (eta). If False, returns
            link-inverted fitted values (mu).
        which : {"mean", "location", "variance", "secondary"}, default "mean"
            ``"mean"`` gives ``E[y | x, z]`` (all parameters); ``"location"``
            gives ``g^-1(X beta)``, the count mean / location parameter (the old
            behavior); ``"variance"`` gives ``Var[y | x, z]`` (all parameters);
            ``"secondary"`` gives the secondary parameter per row -- pi for
            ZI*/Hurdle*, phi for Beta or sigma^2 for Simplex. ``linear=True``
            still returns the primary linear predictor.
        exog_z : array-like or DataFrame, optional
            New rows of the secondary-submodel matrix Z. Needed only when Z was
            passed to ``fit`` as an array; with a ``| z-formula`` it is built
            from ``exog``, and with the solver's default Z (X, or intercept-only
            for Beta/Simplex) it is implied.
        offset : array-like, optional
            Offset for the new rows (IRLS / NB / quasi families), added to the
            linear predictor as in the fit. With ``exog=None`` the training
            offset is used automatically. Predictions for new data used to
            ignore the offset even for models fitted with one.

        Returns
        -------
        ndarray or pandas.Series
            Predicted values.
        """
        import pandas as pd

        self._require_fitted()

        if isinstance(self.family, (Multinomial, Ordinal)):
            if linear or which != "mean":
                raise NotImplementedError(
                    "GLM.predict: for Multinomial/Ordinal, `which` and `linear` are not applicable; "
                    "use predict_proba() (category probabilities) or predict_category() (argmax)."
                )
            if exog is None:
                return self.fittedvalues  # fitted probability of the OBSERVED category (no exog to compare against for new data)
            return self.predict_proba(
                exog, transform
            )  # new data: the (n, K) probability matrix (no observed category to index into)
        if self._gengamma_supported() and (exog is not None or which != "mean" or linear):
            raise NotImplementedError(
                "GLM.predict: for GeneralizedGamma only predict() with no arguments (E[T] at the fitted data) "
                "is implemented; new-data prediction, which='location'/'variance'/'secondary' and linear=True are not."
            )
        if self._gengamma_supported():
            # Guard above guarantees this is the no-argument, which="mean", linear=False case:
            # GeneralizedGamma has no single beta linear predictor (its parameters live in gamma),
            # so return E[T] at the fitted data directly.
            return self._gengamma_moments()[0]

        if which not in ("mean", "location", "variance", "secondary"):
            raise ValueError(
                "GLM.predict: `which` must be 'mean', 'location', 'variance' or 'secondary'."
            )

        if not linear and self._mle_supported() and which != "location":
            X_new, Z_new, idx = self._mle_design(exog, transform, exog_z)
            comp = self._mle_components(np.asarray(self.all_params, dtype=float), X_new, Z_new)
            if which == "secondary":
                if comp.sec is None:
                    raise ValueError(
                        f"GLM.predict: {self.family.__class__.__name__} has no secondary parameter."
                    )
                out, label = comp.sec, "secondary"
            else:
                mean, var = _dist.mean_var(self.family, comp)
                out, label = (mean, "predicted") if which == "mean" else (var, "variance")
            return (
                pd.Series(out, index=idx, name=label)
                if (exog is not None and idx is not None)
                else out
            )
        if which == "secondary":
            raise ValueError(
                f"GLM.predict: {self.family.__class__.__name__} has no secondary parameter."
            )
        if which == "variance":
            raise ValueError(
                f"GLM.predict(which='variance') is only available for MLE families with a distribution "
                f"layer; {self.family.__class__.__name__} doesn't have one."
            )

        if exog is None:
            eta = self.linear_predictors
            if linear:
                return eta
            if which == "location" and self._mle_supported():
                return np.asarray(
                    self.family.linkinv(np.ascontiguousarray(self._X @ self.beta, dtype=float))
                )
            return self.fittedvalues

        if self._model_spec is not None and transform:
            if isinstance(exog, pd.DataFrame):
                X_new = np.asarray(self._model_spec.get_model_matrix(exog), dtype=float)
            else:
                X_new = np.asarray(exog, dtype=float)
        else:
            X_new = np.asarray(exog, dtype=float)

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        if self.rank_deficient:
            X_active = X_new[:, self._active_cols]
            beta_active = self.beta[self._active_cols]
        else:
            X_active = X_new
            beta_active = self.beta

        eta = X_active @ beta_active
        if offset is not None:
            off = np.asarray(offset, dtype=float).reshape(-1)
            if off.size == 1:
                off = np.full(eta.shape[0], float(off[0]))
            if off.shape[0] != eta.shape[0]:
                raise ValueError(
                    f"GLM.predict: offset has {off.shape[0]} rows, exog has {eta.shape[0]}."
                )
            eta = eta + off

        if linear:
            return pd.Series(eta, index=exog.index) if isinstance(exog, pd.DataFrame) else eta

        mu_pred = self.family.linkinv(eta)
        if isinstance(exog, pd.DataFrame) and hasattr(exog, "index"):
            return pd.Series(mu_pred, index=exog.index, name="predicted")
        return mu_pred

    def get_prediction(self, exog=None, transform=True, alpha=0.05, exog_z=None, offset=None):
        """
        Compute predictions along with standard errors and confidence intervals.

        For MLE families the predicted mean is E[y] from ALL parameters, and its standard
        error uses the delta method over the FULL parameter vector and its covariance
        (beta, gamma and auxiliary scalars together, including their cross-covariances) --
        so the interval reflects the uncertainty of pi, alpha, sigma, phi ... as well as beta.
        `se_obs`/`conf_int_obs` remain those of the primary linear predictor.
        """
        self._require_fitted()

        if isinstance(self.family, (Multinomial, Ordinal, GeneralizedGamma)):
            raise NotImplementedError(
                f"GLM.get_prediction: not implemented for {self.family.__class__.__name__}; use "
                f"{'predict_proba' if isinstance(self.family, (Multinomial, Ordinal)) else 'predict'}."
            )

        if self._mle_supported():
            if self.se is None or self.vcov is None:
                raise RuntimeError(
                    "GLM.get_prediction: the covariance matrix was not computed for this model."
                )
            theta, _, _, V = self._inference_block()
            if V is None or V.shape[0] != theta.shape[0]:
                raise RuntimeError(
                    "GLM.get_prediction: unexpected covariance layout for this model."
                )
            X_new, Z_new, idx = self._mle_design(exog, transform, exog_z)

            def _mean_at(th):
                return _dist.mean_var(self.family, self._mle_components(th, X_new, Z_new))[0]

            m0 = _mean_at(theta)
            J = np.empty((m0.shape[0], theta.shape[0]))
            for j in range(theta.shape[0]):
                h = 1e-6 * max(1.0, abs(theta[j]))
                tp, tm = theta.copy(), theta.copy()
                tp[j] += h
                tm[j] -= h
                J[:, j] = (_mean_at(tp) - _mean_at(tm)) / (2.0 * h)
            se_mean = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, V, J), 0.0))

            q = self._ref_quantile(1.0 - alpha / 2.0)
            lo, hi = m0 - q * se_mean, m0 + q * se_mean
            support = getattr(self.family, "SUPPORT", "real")
            if support in ("positive", "unit"):
                lo = np.maximum(lo, 0.0)
            if support == "unit":
                hi = np.minimum(hi, 1.0)

            pos = self._beta_positions()
            eta = X_new @ self.beta
            V_bb = V[np.ix_(pos, pos)]
            se_eta = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X_new, V_bb, X_new), 0.0))
            return GLMPredictionResults(
                predicted_mean=m0,
                se_mean=se_mean,
                conf_int_mean=np.column_stack([lo, hi]),
                se_obs=se_eta,
                conf_int_obs=np.column_stack([eta - q * se_eta, eta + q * se_eta]),
                index=idx,
                alpha=alpha,
            )

        # 1. Get the design matrix X_mat
        if exog is None:
            X_mat = self._X_active
            eta = self.linear_predictors
            idx = getattr(self.data, "index", None) if self.data is not None else None
        else:
            if self._model_spec is not None and transform:
                import pandas as pd

                if isinstance(exog, pd.DataFrame):
                    X_mat = np.asarray(self._model_spec.get_model_matrix(exog), dtype=float)
                else:
                    X_mat = np.asarray(exog, dtype=float)
            else:
                X_mat = np.asarray(exog, dtype=float)

            if X_mat.ndim == 1:
                X_mat = X_mat.reshape(1, -1)

            if self.rank_deficient:
                X_mat = X_mat[:, self._active_cols]

            beta_active = self.beta[self._active_cols] if self.rank_deficient else self.beta
            eta = X_mat @ beta_active
            if offset is not None:
                off = np.asarray(offset, dtype=float).reshape(-1)
                if off.size == 1:
                    off = np.full(eta.shape[0], float(off[0]))
                if off.shape[0] != eta.shape[0]:
                    raise ValueError(
                        f"GLM.get_prediction: offset has {off.shape[0]} rows, exog has {eta.shape[0]}."
                    )
                eta = eta + off
            idx = getattr(exog, "index", None) if hasattr(exog, "index") else None

        # 2. vcov block of the ACTIVE coefficients. With collinear columns the expanded vcov has NaN
        #    rows at the aliased positions, so the leading n x n block would pick the wrong entries
        #    whenever an aliased column is not the last one.
        act = np.asarray(self._active_cols, dtype=int)
        vcov_beta = np.asarray(self.vcov)[np.ix_(act, act)]

        # 3. Standard error of the linear predictor: se_eta = sqrt(diag(X * vcov_beta * X'))
        var_eta = np.einsum("ij,jk,ik->i", X_mat, vcov_beta, X_mat)
        se_eta = np.sqrt(np.maximum(var_eta, 0.0))

        # 4. Confidence intervals on the linear and response scales
        q = self._ref_quantile(1.0 - alpha / 2.0)
        eta_lower = eta - q * se_eta
        eta_upper = eta + q * se_eta

        predicted_mean = self.family.linkinv(eta)
        mean_a = self.family.linkinv(eta_lower)
        mean_b = self.family.linkinv(eta_upper)
        # a decreasing inverse link (inverse, Binomial loglog) maps eta_lower to the UPPER mean bound
        mean_lower = np.minimum(mean_a, mean_b)
        mean_upper = np.maximum(mean_a, mean_b)

        # Delta method for se_mean
        h = 1e-6 * np.maximum(1.0, np.abs(eta))
        dmu_deta = (self.family.linkinv(eta + h) - self.family.linkinv(eta - h)) / (2.0 * h)
        se_mean = np.abs(dmu_deta) * se_eta

        return GLMPredictionResults(
            predicted_mean=predicted_mean,
            se_mean=se_mean,
            conf_int_mean=np.column_stack([mean_lower, mean_upper]),
            se_obs=se_eta,
            conf_int_obs=np.column_stack([eta_lower, eta_upper]),
            index=idx,
            alpha=alpha,
        )

    def _secondary_submodel_names(self, n_gamma):
        """Names of the covariates of the secondary submodel (the one built from Z) of
        Beta/Simplex. Unlike ZeroInflated*/Hurdle* (whose default Z is X), their default
        Z is intercept-only, so the names of X don't apply there.
        """
        source = getattr(self, "_z_source", None)
        z_names = self._zero_feature_names
        if source == "formula" and z_names is not None and len(z_names) == n_gamma:
            return [str(nm) for nm in z_names]
        if source in (None, "default") and n_gamma == 1:
            return ["Intercept"]
        return [f"z{j}" for j in range(n_gamma)]

    def summary(
        self,
        yname: str | None = None,
        xname: list[str] | None = None,
        title: str | None = None,
        alpha: float = 0.05,
        cov_type: str | None = None,
        cov_kwds: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Summary:
        """Render the fitted model as a statsmodels-style summary.

        Parameters
        ----------
        yname : str, optional
            Name of the dependent variable shown in the header. Defaults
            to the left-hand side of the fitted formula.
        xname : list of str, optional
            Coefficient labels. Defaults to ``self.param_names``.
        title : str, optional
            Report title. Defaults to ``"<Family> Regression Results"``.
        alpha : float, default 0.05
            Significance level for the confidence-interval columns.
        cov_type : str, optional
            Covariance estimator for inference (e.g. ``"HC0"``–``"HC3"`` or
            ``"cluster"``). When provided, robust standard errors are used in
            the coefficient tables where supported. Use :meth:`coef_table` or
            :meth:`get_robustcov_results` for a structured robust report.
        cov_kwds : dict, optional
            Extra arguments for robust covariance estimation (for example
            ``{"groups": ...}`` for clustered covariance).
        **kwargs
            Accepted for compatibility with statsmodels-style summaries.

        Returns
        -------
        Summary
            Object whose text/HTML representations are the report; its
            ``.tables`` attribute exposes each rendered table separately.
        """
        self._require_fitted()

        names = list(xname) if xname is not None else self.param_names
        if len(names) != len(self.beta):
            names = [f"x{j}" for j in range(len(self.beta))]
        if yname is None:
            yname = self.formula.split("~", 1)[0].strip() if self.formula else "y"

        n, p = self._X.shape
        n_gamma = len(self.gamma) if self.gamma is not None else 0
        k_total = self._k_free  # beta + gamma + independent auxiliary scalars (exact for MLE)

        method_by_type = {
            "irls": "IRLS",
            "quasi": "Quasi-likelihood (IRLS)",
            "mle": f"{self._minimizer_label()} (MLE)",
            "mle_inflated": f"{self._minimizer_label()} (MLE)",
            "mle_gengamma": f"{self._minimizer_label()} (MLE)",
            "quasi_custom": f"{self._minimizer_label()} (Custom Variance)",
        }
        family_type = getattr(self.family, "family_type", "irls")
        if family_type == "nb_two_step":
            method_by_type["nb_two_step"] = "IRLS"

        top_left = [
            ("Dep. Variable:", yname),
            ("Model:", self.family.__class__.__name__),
            ("Link Function:", self.family.link_description()),
            ("Method:", method_by_type.get(family_type, family_type)),
            ("No. Observations:", str(n)),
            ("Df Residuals:", str(self.df_resid)),
            (
                "Df Model:",
                str(
                    k_total if self._is_mle() else (self.rank if self.rank is not None else k_total)
                ),
            ),
        ]
        dev_str = f"{self.deviance:8.4g}" if self.deviance is not None else "n/a"
        disp_str = f"{self.dispersion:8.4g}" if self.dispersion is not None else "n/a"
        ll_str = f"{self.log_likelihood:8.4g}" if self.log_likelihood is not None else "n/a"
        aic = bic = None
        if self.log_likelihood is not None:
            k_aic = self.n_params
            aic = -2.0 * self.log_likelihood + 2.0 * k_aic
            bic = -2.0 * self.log_likelihood + k_aic * np.log(max(n, 1))
        top_right = [
            ("Deviance:", dev_str),
            ("Dispersion:", disp_str),
            ("Log-Likelihood:", ll_str),
            ("AIC:", f"{aic:8.4g}" if aic is not None else "n/a"),
            ("BIC:", f"{bic:8.4g}" if bic is not None else "n/a"),
            ("Converged:", "True" if self.converged else "False"),
            ("Iterations:", str(self.iterations)),
        ]

        if isinstance(
            self.family,
            (
                TruncatedPoisson,
                TruncatedNegativeBinomial,
                TruncatedGeometric,
                TruncatedGaussian,
            ),
        ):
            ctx = self._fit_ctx or {}
            lo = ctx.get("left")
            hi = ctx.get("right")
            if lo is None and hi is None:
                bounds = "none"
            else:
                lo_str = "-inf" if lo is None else f"{lo:g}"
                hi_str = "inf" if hi is None else f"{hi:g}"
                bounds = f"[{lo_str}, {hi_str}]"
            top_left.append(("Truncation:", bounds))

        has_se = self.se is not None
        n_beta = len(self.beta)
        se_beta = self.se[:n_beta] if has_se else None

        def _se_slice(start, stop):
            return self.se[start:stop] if has_se else None

        def _block(heading, coef_names, coefs, se):
            return {"heading": heading, "names": coef_names, "coefs": coefs, "se": se}

        blocks = []

        if isinstance(self.family, Multinomial):
            n_vars = len(self._feature_names)
            n_cats = len(self.beta) // n_vars if n_vars > 0 else 0

            # If the model came from a formula with a categorical response, use the
            # real category names (e.g. "prog2[academic]") instead of the generic
            # "catK"; the generic scheme is used when no such information is available
            # (a model built by hand from X, y).
            # The multinomial fitting engine uses the LAST category (in the order of
            # _y_categories, which matches the order of the class codes 0..K-1) as the
            # reference/base category, with implicit zero coefficients (see
            # _multinomial_probs: only K-1 beta blocks are estimated, and the missing
            # one is the last column's). Hence self._y_categories carries K names but
            # only n_cats == K-1 blocks need labels.
            if self._y_categories and len(self._y_categories) == n_cats + 1:
                cat_labels = self._y_categories[:-1]
            elif self._y_categories and len(self._y_categories) == n_cats:
                cat_labels = self._y_categories
            else:
                cat_labels = [f"cat{k}" for k in range(n_cats)]

            for k in range(n_cats):
                idx_start = k * n_vars
                idx_end = (k + 1) * n_vars
                blocks.append(
                    _block(
                        f"Category {cat_labels[k]}",
                        self._feature_names,
                        self.beta[idx_start:idx_end],
                        _se_slice(idx_start, idx_end),
                    )
                )

        elif isinstance(self.family, Ordinal):
            n_thresholds = len(self.gamma) if self.gamma is not None else 0
            blocks.append(
                _block(
                    self.family.beta_heading(),
                    names,
                    self.beta,
                    _se_slice(0, n_beta),
                )
            )
            if n_thresholds > 0:
                # Labels based on the real response levels, e.g. "unlikely|somewhat likely";
                # if no levels consistent with the number of thresholds exist, the generic
                # "cut0|cut1" scheme is used.
                thresh_names = _ordinal_threshold_names(self._y_categories, n_thresholds)
                blocks.append(
                    _block(
                        "Thresholds | Cutpoints",
                        thresh_names,
                        self.gamma,
                        _se_slice(n_beta, n_beta + n_thresholds),
                    )
                )

        elif isinstance(self.family, GeneralizedGamma):
            transform_sigma = kwargs.get("transform_sigma", True)

            info = getattr(self, "_gengamma_submodel_info", None)
            if info is not None:
                p_mu_covs = info["p_mu_covs"]
                p_sigma_covs = info["p_sigma_covs"]

                names_mu = info["mu_names"]
                names_sigma = info["sigma_names"]
                names_Q = info["Q_names"]
            else:
                p_mu_covs = len(self.family.mx_mu_cols)
                p_sigma_covs = len(self.family.mx_sigma_cols)

                names_mu = ["Intercept"] + [self._feature_names[i] for i in self.family.mx_mu_cols]
                names_sigma = ["Intercept"] + [
                    self._feature_names[i] for i in self.family.mx_sigma_cols
                ]
                names_Q = ["Intercept"] + [self._feature_names[i] for i in self.family.mx_Q_cols]

            int_mu = self.gamma[0] if len(self.gamma) > 0 else 0.0
            int_sigma = self.gamma[1] if len(self.gamma) > 1 else 0.0
            int_Q = self.gamma[2] if len(self.gamma) > 2 else 0.0

            b_mu = self.beta[:p_mu_covs]
            b_sigma = self.beta[p_mu_covs : p_mu_covs + p_sigma_covs]
            b_Q = self.beta[p_mu_covs + p_sigma_covs :]

            coefs_mu = np.concatenate([[int_mu], b_mu])
            coefs_sigma = np.concatenate([[int_sigma], b_sigma])
            coefs_Q = np.concatenate([[int_Q], b_Q])

            if has_se and len(self.se) == (len(self.beta) + len(self.gamma)):
                se_int_mu = self.se[0]
                se_int_sigma = self.se[1]
                se_int_Q = self.se[2]

                se_betas = self.se[3:]

                se_b_mu = se_betas[:p_mu_covs]
                se_b_sigma = se_betas[p_mu_covs : p_mu_covs + p_sigma_covs]
                se_b_Q = se_betas[p_mu_covs + p_sigma_covs :]

                se_mu = np.concatenate([[se_int_mu], se_b_mu])
                se_sigma = np.concatenate([[se_int_sigma], se_b_sigma])
                se_Q = np.concatenate([[se_int_Q], se_b_Q])
            else:
                se_mu = se_sigma = se_Q = None

            sigma_link_name = self.family.link_sigma
            if transform_sigma and sigma_link_name in _LINK_TRANSFORMS:
                link_spec = _LINK_TRANSFORMS[sigma_link_name]
                inv_func = link_spec["inv"]
                deriv_func = link_spec["deriv"]

                int_sigma_trans = inv_func(coefs_sigma[0])

                if se_sigma is not None:
                    se_int_sigma_trans = np.abs(deriv_func(coefs_sigma[0])) * se_sigma[0]
                    se_sigma = np.concatenate([[se_int_sigma_trans], se_sigma[1:]])

                coefs_sigma = np.concatenate([[int_sigma_trans], coefs_sigma[1:]])
                sigma_heading_scale = f"sigma (natural scale - {link_spec['label']})"
            else:
                sigma_heading_scale = f"sigma (scale parameter - {sigma_link_name} link)"

            blocks.append(
                _block(
                    f"mu (location parameter - {self.family.link_mu} link)",
                    names_mu,
                    coefs_mu,
                    se_mu,
                )
            )
            blocks.append(_block(sigma_heading_scale, names_sigma, coefs_sigma, se_sigma))
            blocks.append(
                _block(f"Q (shape parameter - {self.family.link_Q} link)", names_Q, coefs_Q, se_Q)
            )

        elif isinstance(
            self.family,
            (
                ZeroInflatedPoisson,
                ZeroInflatedNegativeBinomial,
                ValueInflatedPoisson,
                ValueInflatedNegativeBinomial,
            ),
        ):
            n_gamma = len(self.gamma) if self.gamma is not None else 0
            gamma_feature_names = (
                self._zero_feature_names
                if self._zero_feature_names is not None
                else self._feature_names
            )
            g_names = (
                self.family.gamma_names(self._y, n_gamma, gamma_feature_names)
                if hasattr(self.family, "gamma_names")
                else gamma_feature_names
            )
            g_names = [str(name).split(":", 1)[-1] for name in g_names]

            blocks.append(
                _block(self.family.beta_heading(), names, self.beta, _se_slice(0, n_beta))
            )
            if n_gamma > 0:
                blocks.append(
                    _block(
                        self.family.gamma_heading(),
                        g_names,
                        self.gamma,
                        _se_slice(n_beta, n_beta + n_gamma),
                    )
                )

        elif isinstance(
            self.family,
            (
                HurdlePoisson,
                HurdleNegativeBinomial,
                HurdleGamma,
                HurdleLogNormal,
                HurdleBeta,
                HurdleGeometric,
            ),
        ):
            n_gamma = len(self.gamma) if self.gamma is not None else 0
            gamma_feature_names = (
                self._zero_feature_names
                if self._zero_feature_names is not None
                else self._feature_names
            )
            g_names = (
                self.family.gamma_names(self._y, n_gamma, gamma_feature_names)
                if hasattr(self.family, "gamma_names")
                else gamma_feature_names
            )
            g_names = [str(name).split(":", 1)[-1] for name in g_names]

            blocks.append(
                _block(self.family.beta_heading(), names, self.beta, _se_slice(0, n_beta))
            )
            if n_gamma > 0:
                blocks.append(
                    _block(
                        self.family.gamma_heading(),
                        g_names,
                        self.gamma,
                        _se_slice(n_beta, n_beta + n_gamma),
                    )
                )

        elif isinstance(self.family, (Beta, Simplex)):
            n_gamma = len(self.gamma) if self.gamma is not None else 0

            blocks.append(
                _block(self.family.beta_heading(), names, self.beta, _se_slice(0, n_beta))
            )
            if n_gamma > 0:
                blocks.append(
                    _block(
                        self.family.gamma_heading(),
                        self._secondary_submodel_names(n_gamma),
                        self.gamma,
                        _se_slice(n_beta, n_beta + n_gamma),
                    )
                )

        elif self.gamma is not None and len(self.gamma) > 0:
            n_gamma = len(self.gamma)
            gamma_feature_names = (
                self._zero_feature_names
                if self._zero_feature_names is not None
                else self._feature_names
            )
            g_names = (
                self.family.gamma_names(self._y, n_gamma, gamma_feature_names)
                if hasattr(self.family, "gamma_names")
                else [f"gamma[{j}]" for j in range(n_gamma)]
            )

            blocks.append(
                _block(
                    self.family.beta_heading(),
                    names,
                    self.beta,
                    _se_slice(0, n_beta),
                )
            )
            blocks.append(
                _block(
                    getattr(self.family, "gamma_heading", lambda: "Auxiliary Parameters")(),
                    g_names,
                    self.gamma,
                    _se_slice(n_beta, n_beta + n_gamma),
                )
            )

        # Independent auxiliary scalars (alpha / sigma / shape / phi ...) estimated by MLE,
        # reported on the estimation (log) scale like log_sigma / log_nu of StudentT/Skew*;
        # their SE comes from the same vcov.
        ex_names = list(self._extra_free_names())
        ex_vals = self._extra_free_values()
        if (
            self._is_mle()
            and ex_names
            and ex_vals.size == len(ex_names)
            and family_type in ("mle", "mle_inflated")
        ):
            if not blocks:
                blocks.append(
                    _block(self.family.beta_heading(), names, self.beta, _se_slice(0, n_beta))
                )
            s0 = n_beta + (len(self.gamma) if self.gamma is not None else 0)
            blocks.append(
                _block(
                    "Auxiliary Parameters (log scale)",
                    ex_names,
                    ex_vals,
                    _se_slice(s0, s0 + len(ex_names)),
                )
            )

        extra_lines = []
        notes = []
        if (
            isinstance(self.family, Multinomial)
            and self._y_categories
            and len(self._y_categories) >= 2
        ):
            notes.append(
                f"Reference category (implicit zero coefficients): {self._y_categories[-1]}."
            )
        if not self.converged:
            msg = getattr(self, "_fit_message", None)
            notes.append(
                f"The optimizer/solver did NOT converge{f' (message={msg!r})' if msg else ''}."
            )
        if not has_se:
            notes.append("No standard errors computed.")
        elif np.any(~np.isfinite(np.asarray(self.se, dtype=float))):
            # The C++ engine (mle_vcov.hpp) marks parameters whose variance could not be
            # estimated reliably (singular Hessian in that direction) with NaN, instead of
            # an arbitrary large number: a sentinel like 1e6 would collapse z=coef/se to 0
            # and the p-value to 1, reading as "clearly not significant" when the correct
            # reading is "significance unknown". Such rows are shown as n/a in the table.
            n_bad = int(np.sum(~np.isfinite(np.asarray(self.se, dtype=float))))
            notes.append(
                f"Standard error could not be reliably estimated for {n_bad} "
                f"parameter{'s' if n_bad != 1 else ''} (appears as NaN in the table "
                "above). This reflects a singular or ill-conditioned Hessian in that "
                "direction, not a computed value of zero uncertainty."
            )
        if self.use_t and has_se:
            notes.append(
                f"Dispersion is estimated: the z / p-value / interval columns above use a normal reference; "
                f"the t(df_resid={self.df_resid}) based ones are in .tvalues, .pvalues and .conf_int()."
            )

        return build_summary(
            title=title or f"{self.family.__class__.__name__} Regression Results",
            subtitle=f"Formula: {self.formula}" if self.formula else f"GLM(X, y)  -- X: ({n}, {p})",
            left=top_left,
            right=top_right,
            names=names,
            coefs=self.beta,
            se=se_beta,
            blocks=blocks,
            extra_lines=extra_lines,
            alpha=alpha,
            notes=notes,
        )

    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        if self.formula is not None:
            return f"GLM('{self.formula}', family={self.family!r}, {status})"
        n, p = self._X.shape
        return f"GLM(X=({n},{p}), family={self.family!r}, {status})"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        family: Family | None = None,
        weights: Any = None,
        offset: Any = None,
        cens: Any = None,
        entry: Any = None,
        **kwargs: Any,
    ) -> "GLM":
        """Build an unfitted GLM from an R-style formula.

        Parameters
        ----------
        formula : str
            Model formula, e.g. ``"y ~ x1 + x2"``. Two-part models
            (zero-inflated/hurdle) use the R ``Formula`` convention
            ``"count_part | zero_part"``.
        data : pandas.DataFrame
            Data containing the variables referenced by ``formula``.
        family : Family, optional
            Family instance such as ``Poisson()``, ``Binomial()`` or
            ``HurdleGeometric()``. Defaults to ``Gaussian()``.
        weights, offset : str, array-like or None, optional
            Prior weights and offset. A string names a column of ``data``;
            an array-like value is used directly.
        cens : str, array-like or None, optional
            Event/censoring indicator (1 = event, 0 = right-censored) for the
            survival families and ``GeneralizedGamma``. A string names a column
            of ``data``; it is stored on the model and used by ``.fit()``
            unless overridden there.
        entry : str, array-like or None, optional
            Entry (left-truncation) times for ``GeneralizedGamma``. A string
            names a column of ``data``.
        **kwargs
            Further ``GLM`` constructor options (``solver``, ``maxit``,
            ``ftol``, ``gtol``, ``maxcor``, ``compute_se``, ``minimizer``,
            ``use_t``, ``standardize``, ``y_levels``, ...).

        Returns
        -------
        GLM
            Unfitted model; call ``.fit()`` to estimate it.
        """
        count_formula, zero_formula = _split_zeroinfl_formula(formula)

        if isinstance(family, Ordinal):
            f_str = (
                count_formula
                if "- 1" in count_formula or "+ 0" in count_formula
                else f"{count_formula} - 1"
            )
            y_mat, X_mat = formulaic.model_matrix(f_str, data)
        else:
            y_mat, X_mat = formulaic.model_matrix(count_formula, data)

        y_raw = np.asarray(y_mat)
        y_categories = None
        if y_raw.ndim == 2 and y_raw.shape[1] > 1:
            y_arr = y_raw.argmax(axis=1).astype(float)
            y_categories = list(y_mat.columns)
        else:
            y_arr = y_raw.reshape(-1).astype(float)
            if isinstance(family, Ordinal):
                # Response not expanded to dummies (e.g. coded 1,2,3): the levels are the
                # observed values, in increasing order (= the order of the codes).
                lv, _ = _response_levels_from_data(count_formula, data)
                if lv and kwargs.get("y_levels") is None:
                    y_categories = lv

        if isinstance(family, Ordinal) and y_raw.ndim == 2 and y_raw.shape[1] > 1:
            # Automatic level ordering (no pd.Categorical(..., ordered=True) needed):
            #   1. y_levels=[...] if given; 2. order of an ordered Categorical;
            #   3. numeric order if the levels are numbers; 4. known scales
            #   (unlikely < somewhat likely < very likely, low < medium < high, ...);
            #   5. if none applies, warn and use the alphabetical order.
            labels = [_clean_level_label(c) for c in y_categories]
            lv, ordered = _response_levels_from_data(count_formula, data)
            perm, how = _order_ordinal_levels(
                labels,
                user_levels=kwargs.get("y_levels"),
                data_levels=lv,
                data_ordered=bool(ordered),
            )
            if how == "fallback":
                import warnings

                warnings.warn(
                    "Ordinal: could not infer the order of the response levels; the alphabetical "
                    f"order {labels} was used. Specify the correct order with y_levels=[...] "
                    "(from lowest to highest).",
                    stacklevel=2,
                )
            y_raw = y_raw[:, perm]
            y_arr = y_raw.argmax(axis=1).astype(float)
            y_categories = [y_categories[i] for i in perm]
            kwargs.pop("y_levels", None)

        X_arr = np.asarray(X_mat).astype(float)
        feature_names = list(X_mat.columns)
        n = X_arr.shape[0]

        w = _resolve_col(weights, data, n, "weights")
        off = _resolve_col(offset, data, n, "offset")

        model = cls(
            X_arr,
            y_arr,
            family=family,
            weights=w,
            offset=off,
            feature_names=feature_names,
            **kwargs,
        )
        model.formula = formula
        model.data = data
        model._cens_from_formula = cens
        model._entry_from_formula = entry
        model._model_spec = getattr(X_mat, "model_spec", None)
        if y_categories is not None or model._y_categories is None:
            model._y_categories = y_categories

        if zero_formula is not None:
            Z_mat = formulaic.model_matrix(zero_formula, data)
            Z_arr = np.asarray(Z_mat).astype(float)
            if Z_arr.shape[0] != n:
                raise ValueError("GLM: subformula rows do not match.")
            model._Z_from_formula = Z_arr
            model._zero_formula = zero_formula
            model._zero_model_spec = getattr(Z_mat, "model_spec", None)
            model._zero_feature_names = list(Z_mat.columns)

        return model

    @classmethod
    def from_formula_gengamma(
        cls,
        formula: str,
        data: pd.DataFrame,
        sigma_formula: str = "~ 1",
        Q_formula: str = "~ 1",
        weights: Any = None,
        offset: Any = None,
        cens: Any = None,
        entry: Any = None,
        link_mu: str = "identity",
        link_sigma: str = "log",
        link_Q: str = "identity",
        **kwargs: Any,
    ) -> "GLM":
        """Build an unfitted GeneralizedGamma model from R-style formulas.

        GeneralizedGamma has three linear predictors (location ``mu``,
        scale ``sigma`` and shape ``Q``), so the main formula models
        ``mu`` while ``sigma_formula`` and ``Q_formula`` model the other two.

        Parameters
        ----------
        formula : str
            Formula for the location submodel, e.g. ``"y ~ x"``.
        data : pandas.DataFrame
            Data containing all variables referenced by the formulas.
        sigma_formula, Q_formula : str, optional
            Formulas for the scale and shape submodels, e.g. ``"~ x"``.
            They default to intercept-only (``"~ 1"``); use ``"~ 0"`` or
            ``"~ - 1"`` to drop the intercept.
        weights, offset : str, array-like or None, optional
            Prior weights and offset. A string names a column of ``data``.
        cens, entry : str, array-like or None, optional
            Event/censoring status (1 = event, 0 = right-censored) and entry
            (left-truncation) times. A string names a column of ``data``; the
            values are stored on the model and used by ``.fit()`` unless
            overridden there.
        link_mu, link_sigma, link_Q : str
            Link functions for the ``mu``, ``sigma`` and ``Q``
            submodels. Defaults: ``"identity"``, ``"log"``, ``"identity"``.
        **kwargs
            Further ``GLM`` constructor options.

        Returns
        -------
        GLM
            Unfitted model; call ``.fit()`` to estimate it.
        """
        mu_formula = formula

        def _strip_intercept(f_str):
            if "- 1" not in f_str and "+ 0" not in f_str:
                if "~" in f_str:
                    lhs, rhs = f_str.split("~", 1)
                    return f"{lhs.strip()} ~ {rhs.strip()} - 1"
                return f"{f_str} - 1"
            return f_str

        f_mu_clean = _strip_intercept(mu_formula)
        f_sigma_clean = _strip_intercept(sigma_formula)
        f_Q_clean = _strip_intercept(Q_formula)

        y_mat, X_mu_mat = formulaic.model_matrix(f_mu_clean, data)
        X_sigma_mat = (
            formulaic.model_matrix(f_sigma_clean, data)
            if f_sigma_clean.strip() != "~ - 1"
            else None
        )
        X_Q_mat = formulaic.model_matrix(f_Q_clean, data) if f_Q_clean.strip() != "~ - 1" else None

        y_raw = np.asarray(y_mat)
        # Same as in from_formula: with a categorical response, formulaic returns one dummy
        # column per level; the class code (argmax) is used instead of flattening, so the row
        # count is preserved and each observation keeps its category.
        if y_raw.ndim == 2 and y_raw.shape[1] > 1:
            y_arr = y_raw.argmax(axis=1).astype(float)
        else:
            y_arr = y_raw.reshape(-1).astype(float)

        names_mu_covs = list(X_mu_mat.columns) if X_mu_mat is not None else []
        names_sigma_covs = list(X_sigma_mat.columns) if X_sigma_mat is not None else []
        names_Q_covs = list(X_Q_mat.columns) if X_Q_mat is not None else []

        X_mu = (
            np.asarray(X_mu_mat, dtype=float) if X_mu_mat is not None else np.empty((len(y_arr), 0))
        )
        X_sigma = (
            np.asarray(X_sigma_mat, dtype=float)
            if X_sigma_mat is not None
            else np.empty((len(y_arr), 0))
        )
        X_Q = np.asarray(X_Q_mat, dtype=float) if X_Q_mat is not None else np.empty((len(y_arr), 0))

        matrices = [m for m in [X_mu, X_sigma, X_Q] if m.shape[1] > 0]
        X_full = np.hstack(matrices) if len(matrices) > 0 else np.empty((len(y_arr), 0))

        p_mu = X_mu.shape[1]
        p_sigma = X_sigma.shape[1]
        p_Q = X_Q.shape[1]

        mx_mu_cols = (
            np.arange(0, p_mu, dtype=np.int32) if p_mu > 0 else np.array([], dtype=np.int32)
        )
        mx_sigma_cols = (
            np.arange(p_mu, p_mu + p_sigma, dtype=np.int32)
            if p_sigma > 0
            else np.array([], dtype=np.int32)
        )
        mx_Q_cols = (
            np.arange(p_mu + p_sigma, p_mu + p_sigma + p_Q, dtype=np.int32)
            if p_Q > 0
            else np.array([], dtype=np.int32)
        )

        family = GeneralizedGamma(
            mx_mu_cols=mx_mu_cols,
            mx_sigma_cols=mx_sigma_cols,
            mx_Q_cols=mx_Q_cols,
            link_mu=link_mu,
            link_sigma=link_sigma,
            link_Q=link_Q,
        )

        n = len(y_arr)
        w = _resolve_col(weights, data, n, "weights")
        off = _resolve_col(offset, data, n, "offset")

        feature_names = names_mu_covs + names_sigma_covs + names_Q_covs

        model = cls(
            X_full,
            y_arr,
            family=family,
            weights=w,
            offset=off,
            feature_names=feature_names,
            **kwargs,
        )

        model.formula = f"mu: {mu_formula} | sigma: {sigma_formula} | Q: {Q_formula}"
        model.data = data
        model._cens_from_formula = cens
        model._entry_from_formula = entry
        model._gengamma_submodel_info = {
            "mu_names": ["Intercept"] + names_mu_covs,
            "sigma_names": ["Intercept"] + names_sigma_covs,
            "Q_names": ["Intercept"] + names_Q_covs,
            "p_mu_covs": p_mu,
            "p_sigma_covs": p_sigma,
            "p_Q_covs": p_Q,
        }

        return model


# Analysis helper and sampling functions


def aic(model: GLM) -> float | None:
    """Akaike Information Criterion of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    float or None
        ``-2 * log_likelihood + 2 * n_params``, or ``None`` when the
        family does not provide a log-likelihood.
    """
    return model.aic


def bic(model: GLM) -> float | None:
    """Bayesian Information Criterion of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    float or None
        ``-2 * log_likelihood + n_params * log(nobs)``, or ``None`` when
        the family does not provide a log-likelihood.
    """
    return model.bic


def deviance(model: GLM) -> float:
    """Deviance of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    float
        Residual deviance of the fit.
    """
    return model.deviance


def llf(model: GLM) -> float | None:
    """Log-likelihood of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    float or None
        Maximized log-likelihood, or ``None`` when the family does not
        provide one.
    """
    return model.llf


def llnull(model: GLM) -> float:
    """Log-likelihood of the intercept-only null model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    float
        Null-model log-likelihood, obtained by an intercept-only refit
        with the same family, weights and offset.
    """
    return model.llnull


def scale(model: GLM) -> float:
    """Scale (dispersion) parameter of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    float
        Estimated dispersion; ``1.0`` for families with fixed scale.
    """
    return model.scale


def resid_pearson(model: GLM) -> pd.Series:
    """Pearson residuals of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        ``(y - mu) / sqrt(V(mu))`` for each observation.
    """
    return model.resid_pearson


def resid_deviance(model: GLM) -> pd.Series:
    """Deviance residuals of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        Signed square roots of the per-observation deviance contributions.
    """
    return model.resid_deviance


def resid_response(model: GLM) -> pd.Series:
    """Raw response residuals of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        ``y - mu`` for each observation.
    """
    return model.resid_response


def resid_anscombe(model: GLM) -> pd.Series:
    """Anscombe residuals of a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        Residuals on the Anscombe variance-stabilizing scale.
    """
    return model.resid_anscombe


def tvalues(model: GLM) -> pd.Series:
    """Wald z/t statistics of the fitted coefficients.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        ``beta / se`` for each coefficient.
    """
    return model.tvalues


def pvalues(model: GLM) -> pd.Series:
    """Two-sided p-values of the fitted coefficients.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        p-value for each coefficient, using the model's reference
        distribution (normal or t).
    """
    return model.pvalues


def mu(model: GLM) -> pd.Series:
    """Fitted mean response of a model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    pandas.Series
        Fitted mean ``μ`` for each observation in the training data.
    """
    return model.mu


def conf_int(model: GLM, alpha: float = 0.05, cols: Any = None) -> pd.DataFrame:
    """Confidence intervals for the model coefficients.

    Parameters
    ----------
    model : GLM
        A fitted model.
    alpha : float, default 0.05
        Significance level; intervals are computed at ``1 - alpha``.
    cols : sequence of str, optional
        Subset of coefficient names to include. Defaults to all.

    Returns
    -------
    pandas.DataFrame
        Lower and upper confidence bounds for each coefficient.
    """
    return model.conf_int(alpha=alpha, cols=cols)


def predict(
    model: GLM,
    exog: Any = None,
    transform: bool = True,
    linear: bool = False,
    **kwargs: Any,
) -> Any:
    """Predict from a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.
    exog : array-like or pandas.DataFrame, optional
        New design matrix. Defaults to the training data.
    transform : bool, default True
        Apply the inverse link so predictions are on the response scale.
    linear : bool, default False
        Return the linear predictor instead of the mean.
    **kwargs
        Forwarded to :meth:`GLM.predict` (e.g. ``which``, ``exog_z``,
        ``offset``).

    Returns
    -------
    pandas.Series or numpy.ndarray
        Predicted means, or linear predictors when ``linear=True``.
    """
    return model.predict(exog=exog, transform=transform, linear=linear, **kwargs)


def get_prediction(
    model: GLM,
    exog: Any = None,
    transform: bool = True,
    alpha: float = 0.05,
    exog_z: Any = None,
    **kwargs: Any,
) -> "GLMPredictionResults":
    """Predictions with confidence intervals and a result frame.

    Parameters
    ----------
    model : GLM
        A fitted model.
    exog : array-like or pandas.DataFrame, optional
        New design matrix. Defaults to the training data.
    transform : bool, default True
        Apply the inverse link so predictions are on the response scale.
    alpha : float, default 0.05
        Significance level for the confidence intervals.
    exog_z : array-like or pandas.DataFrame, optional
        Design matrix for the secondary submodel (zero-inflation,
        hurdle probability, Beta precision or Simplex dispersion).
    **kwargs
        Forwarded to :meth:`GLM.get_prediction` (for example ``offset``).

    Returns
    -------
    GLMPredictionResults
        Object exposing ``predicted_mean``, ``se_mean``, ``conf_int``
        and a ``summary_frame()`` table.
    """
    return model.get_prediction(
        exog=exog, transform=transform, alpha=alpha, exog_z=exog_z, **kwargs
    )


def converged(model: GLM) -> bool:
    """Whether the fitting algorithm converged.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    bool
        ``True`` when the solver met its convergence criterion.
    """
    return model.converged


def wald_test(model: GLM, r_matrix: Any, q_matrix: Any = None) -> dict:
    """Wald test for a linear restriction on the model parameters.

    Parameters
    ----------
    model : GLM
        A fitted model.
    r_matrix : array-like
        Restriction matrix ``R`` in ``R * theta = q``.
    q_matrix : array-like, optional
        Right-hand side ``q``. Defaults to zeros.

    Returns
    -------
    dict
        Test statistic, degrees of freedom and p-value.
    """
    return model.wald_test(r_matrix, q_matrix=q_matrix)


def anova(model: GLM, other: GLM, test: str = "auto") -> pd.DataFrame:
    """Compare two nested models with an analysis-of-deviance table.

    Parameters
    ----------
    model : GLM
        A fitted model.
    other : GLM
        Another fitted model, nested with ``model`` and fit to the same data.
    test : {"auto", "Chisq", "F", "LRT"}, default "auto"
        Test statistic to report.

    Returns
    -------
    pandas.DataFrame
        R-style analysis-of-deviance table.
    """
    return model.anova(other, test=test)


def pseudo_rsquared(model: GLM, kind: str = "mcfadden") -> float | None:
    """Pseudo R-squared for a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.
    kind : {"mcfadden", "mcfadden_adj"}, default "mcfadden"
        Pseudo R-squared variant.

    Returns
    -------
    float or None
        The requested pseudo R-squared, or ``None`` if unavailable.
    """
    return model.pseudo_rsquared(kind=kind)


def resid_quantile(model: GLM, seed: Any = None):
    """Randomized quantile residuals for a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.
    seed : int or numpy.random.Generator, optional
        Seed for randomized residuals on discrete responses.

    Returns
    -------
    numpy.ndarray or None
        Quantile residuals, or ``None`` when unavailable.
    """
    return model.resid_quantile(seed=seed)


def get_influence(model: GLM) -> "GLMInfluence":
    """Influence and leverage diagnostics for a fitted model.

    Parameters
    ----------
    model : GLM
        A fitted model.

    Returns
    -------
    GLMInfluence
        Object with leverage, Cook's distance and residual diagnostics.
    """
    return model.get_influence()


glm = GLM.from_formula


_SECONDARY_LINK_DESCRIPTIONS = {
    "zero_link": "Link for the zero-hurdle probability submodel (P(y > 0)).",
    "infl_link": "Link for the zero-inflation probability submodel (P(y = 0)).",
    "phi_link": "Link for the precision parameter (phi).",
    "sigma_link": "Link for the dispersion parameter (sigma).",
}


def _family_default(family_cls, argument):
    """Default value of `argument` in ``family_cls.__init__`` (or None)."""
    try:
        param = inspect.signature(family_cls.__init__).parameters.get(argument)
    except (TypeError, ValueError):
        return None
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return param.default


def _shortcut_docstring(family_cls, family_kwargs, alias, summary=None):
    """Shared NumPy-style docstring for the generated family shortcuts."""
    family_name = family_cls.__name__
    link_default = _family_default(family_cls, "link")
    if link_default is None:
        link_default = _family_default(family_cls, "mu_link")
    if link_default is None:
        link_text = "Link function for the main linear predictor."
    else:
        link_text = (
            "Link function for the main linear predictor. "
            f"Default: ``{link_default!r}``."
        )

    extra_params = []
    for arg in family_kwargs:
        desc = _SECONDARY_LINK_DESCRIPTIONS.get(
            arg, f"Value passed to the ``{arg}`` argument of ``{family_name}``."
        )
        default = _family_default(family_cls, arg)
        if default is not None:
            desc += f" Default: ``{default!r}``."
        extra_params.append(f"    {arg} : str, optional\n        {desc}")
    extra_block = ("\n".join(extra_params) + "\n") if extra_params else ""

    two_part = family_name.startswith(("ZeroInflated", "Hurdle"))
    example_formula = "y ~ x | z" if two_part else "y ~ x"
    example_name = alias or family_name

    return (
        f"{summary or f'Build a {family_name} GLM from an R-style formula.'}\n"
        "\n"
        f"Shortcut for ``GLM.from_formula(formula, data, family={family_name}(...))``.\n"
        "It returns an **unfitted** :class:`GLM`; call ``.fit()`` to estimate it.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "formula : str\n"
        f"    R-style model formula, e.g. ``\"{example_formula}\"``.\n"
        "    Two-part models (zero-inflated/hurdle) use the R ``Formula``\n"
        "    convention ``\"count_part | zero_part\"``.\n"
        "data : pandas.DataFrame\n"
        "    Data frame with the variables referenced by ``formula``.\n"
        "link : str, optional\n"
        f"    {link_text}\n"
        f"{extra_block}"
        "**kwargs\n"
        "    Extra keyword arguments forwarded to :meth:`GLM.from_formula`\n"
        "    (e.g. ``solver``, ``maxit``, ``compute_se``, ``y_levels``).\n"
        "\n"
        "Returns\n"
        "-------\n"
        "GLM\n"
        "    Unfitted model instance; call ``.fit()`` before ``.summary()``.\n"
        "\n"
        "Examples\n"
        "--------\n"
        f">>> from statcraft.glm import {example_name}\n"
        f">>> model = {example_name}(\"{example_formula}\", df)\n"
        ">>> fit = model.fit()\n"
    )


def _formula_shortcut(family_cls, family_kwargs=(), alias=None, summary=None):
    """Build a formula-based shortcut for `family_cls`.

    `family_kwargs` names extra Family-constructor arguments the shortcut
    accepts (the secondary link: "phi_link", "sigma_link", "zero_link",
    "infl_link"); they are taken out of **kwargs before the rest goes to
    GLM. `alias` is the public function name, used in the generated
    docstring (e.g. "poisson"). `summary` optionally replaces the first
    line of that docstring.
    """
    init_params = inspect.signature(family_cls.__init__).parameters
    # Beta/Simplex call their main link `mu_link` instead of `link`.
    link_keyword = "link" if "link" in init_params else "mu_link"

    def _shortcut(
        formula: str,
        data: pd.DataFrame,
        link: str | None = None,
        weights: Any = None,
        offset: Any = None,
        **kwargs: Any,
    ) -> GLM:
        fam_kw = {k: kwargs.pop(k) for k in family_kwargs if k in kwargs}
        fam = (
            family_cls(**fam_kw)
            if link is None
            else family_cls(**{link_keyword: link}, **fam_kw)
        )
        return GLM.from_formula(formula, data, family=fam, weights=weights, offset=offset, **kwargs)

    _shortcut.__name__ = alias or family_cls.__name__
    _shortcut.__qualname__ = _shortcut.__name__
    _shortcut.__doc__ = _shortcut_docstring(family_cls, family_kwargs, alias, summary=summary)
    return _shortcut


gaussian = _formula_shortcut(Gaussian, alias="gaussian")
ols = _formula_shortcut(
    Gaussian,
    alias="ols",
    summary="Ordinary least squares (Gaussian GLM) from an R-style formula.",
)


def OLS(X, y, **kwargs):
    """Ordinary least squares (Gaussian GLM) from raw arrays.

    Matrix counterpart of :func:`ols`: it builds the same Gaussian model
    but takes the design matrix and response directly, without a formula.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix. Add a column of ones yourself if you want an
        intercept (for example with
        ``statcraft.utils.add_constant(X)``).
    y : array-like of shape (n,)
        Response vector.
    **kwargs
        Further :class:`GLM` constructor options, such as ``weights``,
        ``offset``, ``solver``, ``maxit``, ``compute_se`` or
        ``feature_names``.

    Returns
    -------
    GLM
        Unfitted model; call ``.fit()`` to estimate it.

    Examples
    --------
    >>> import numpy as np
    >>> from statcraft.glm import OLS
    >>> X = np.column_stack([np.ones(len(x)), x])
    >>> fit = OLS(X, y).fit()
    """
    return GLM(X, y, family=Gaussian(), **kwargs)


binomial = _formula_shortcut(Binomial, alias="binomial")
poisson = _formula_shortcut(Poisson, alias="poisson")
gamma = _formula_shortcut(Gamma, alias="gamma")
inverse_gaussian = _formula_shortcut(InverseGaussian, alias="inverse_gaussian")
exponential = _formula_shortcut(Exponential, alias="exponential")
pareto = _formula_shortcut(Pareto, alias="pareto")
inverse_gamma = _formula_shortcut(InverseGamma, alias="inverse_gamma")


def negative_binomial(
    formula: str,
    data: pd.DataFrame,
    theta: float,
    link: str = "log",
    weights: Any = None,
    offset: Any = None,
    **kwargs: Any,
) -> GLM:
    """Build a NegativeBinomial GLM from an R-style formula.

    Parameters
    ----------
    formula : str
        Model formula, e.g. ``"y ~ x1 + x2"``.
    data : pandas.DataFrame
        Data containing the variables referenced by ``formula``.
    theta : float
        Dispersion parameter of the negative binomial distribution.
    link : str, default "log"
        Link function for the mean.
    weights, offset : str, array-like or None, optional
        Prior weights and offset; a string names a column of ``data``.
    **kwargs
        Further ``GLM`` options forwarded to :meth:`GLM.from_formula`.

    Returns
    -------
    GLM
        Unfitted model; call ``.fit()`` to estimate it.
    """
    fam = NegativeBinomial(theta=theta, link=link)
    return GLM.from_formula(formula, data, family=fam, weights=weights, offset=offset, **kwargs)


def tweedie(
    formula: str,
    data: pd.DataFrame,
    var_power: float,
    link: str = "log",
    weights: Any = None,
    offset: Any = None,
    **kwargs: Any,
) -> GLM:
    """Build a Tweedie GLM from an R-style formula.

    Parameters
    ----------
    formula : str
        Model formula, e.g. ``"y ~ x1 + x2"``.
    data : pandas.DataFrame
        Data containing the variables referenced by ``formula``.
    var_power : float
        Tweedie variance power (1 < ``var_power`` < 2 for compound
        Poisson-Gamma).
    link : str, default "log"
        Link function for the mean.
    weights, offset : str, array-like or None, optional
        Prior weights and offset; a string names a column of ``data``.
    **kwargs
        Further ``GLM`` options forwarded to :meth:`GLM.from_formula`.

    Returns
    -------
    GLM
        Unfitted model; call ``.fit()`` to estimate it.
    """
    fam = Tweedie(var_power=var_power, link=link)
    return GLM.from_formula(formula, data, family=fam, weights=weights, offset=offset, **kwargs)


quasi_poisson = _formula_shortcut(QuasiPoisson, alias="quasi_poisson")
quasi_binomial = _formula_shortcut(QuasiBinomial, alias="quasi_binomial")
quasi_gamma = _formula_shortcut(QuasiGamma, alias="quasi_gamma")
quasi_inverse_gaussian = _formula_shortcut(
    QuasiInverseGaussian, alias="quasi_inverse_gaussian"
)
quasi_gaussian = _formula_shortcut(QuasiGaussian, alias="quasi_gaussian")

beta = _formula_shortcut(Beta, ("phi_link",), alias="beta")
lognormal = _formula_shortcut(LogNormal, alias="lognormal")
weibull = _formula_shortcut(Weibull, alias="weibull")
gumbel = _formula_shortcut(Gumbel, alias="gumbel")
loglogistic = _formula_shortcut(LogLogistic, alias="loglogistic")
student_t = _formula_shortcut(StudentT, alias="student_t")
skew_normal = _formula_shortcut(SkewNormal, alias="skew_normal")
skew_t = _formula_shortcut(SkewT, alias="skew_t")
logistic_distribution = _formula_shortcut(
    LogisticDistribution, alias="logistic_distribution"
)
simplex = _formula_shortcut(Simplex, ("sigma_link",), alias="simplex")

multinomial = _formula_shortcut(Multinomial, alias="multinomial")
ordinal = _formula_shortcut(Ordinal, alias="ordinal")

zero_inflated_poisson = _formula_shortcut(
    ZeroInflatedPoisson, ("infl_link",), alias="zero_inflated_poisson"
)
zero_inflated_negative_binomial = _formula_shortcut(
    ZeroInflatedNegativeBinomial,
    ("infl_link",),
    alias="zero_inflated_negative_binomial",
)


def value_inflated_poisson(
    formula: str,
    data: pd.DataFrame,
    target_value: float,
    base_link: str = "log",
    infl_link: str = "logit",
    weights: Any = None,
    offset: Any = None,
    **kwargs: Any,
) -> GLM:
    """Build a Poisson GLM with inflation at an arbitrary value.

    Parameters
    ----------
    formula : str
        Model formula. The inflation submodel can be given with the
        two-part syntax ``"count_part | infl_part"``.
    data : pandas.DataFrame
        Data containing the variables referenced by ``formula``.
    target_value : float
        Value at which the distribution is inflated.
    base_link : str, default "log"
        Link function for the Poisson mean.
    infl_link : str, default "logit"
        Link function for the inflation probability.
    weights, offset : str, array-like or None, optional
        Prior weights and offset; a string names a column of ``data``.
    **kwargs
        Further ``GLM`` options forwarded to :meth:`GLM.from_formula`.

    Returns
    -------
    GLM
        Unfitted model; call ``.fit()`` to estimate it.
    """
    fam = ValueInflatedPoisson(target_value, base_link=base_link, infl_link=infl_link)
    return GLM.from_formula(formula, data, family=fam, weights=weights, offset=offset, **kwargs)


def value_inflated_negative_binomial(
    formula: str,
    data: pd.DataFrame,
    target_value: float,
    var_form: str = "quadratic",
    base_link: str = "log",
    infl_link: str = "logit",
    weights: Any = None,
    offset: Any = None,
    **kwargs: Any,
) -> GLM:
    """Build a NegativeBinomial GLM with inflation at an arbitrary value.

    Parameters
    ----------
    formula : str
        Model formula. The inflation submodel can be given with the
        two-part syntax ``"count_part | infl_part"``.
    data : pandas.DataFrame
        Data containing the variables referenced by ``formula``.
    target_value : float
        Value at which the distribution is inflated.
    var_form : {"quadratic", "linear"}, default "quadratic"
        NegativeBinomial variance form: ``"quadratic"`` is NB2 and
        ``"linear"`` is NB1.
    base_link : str, default "log"
        Link function for the negative binomial mean.
    infl_link : str, default "logit"
        Link function for the inflation probability.
    weights, offset : str, array-like or None, optional
        Prior weights and offset; a string names a column of ``data``.
    **kwargs
        Further ``GLM`` options forwarded to :meth:`GLM.from_formula`.

    Returns
    -------
    GLM
        Unfitted model; call ``.fit()`` to estimate it.
    """
    fam = ValueInflatedNegativeBinomial(
        target_value, var_form=var_form, base_link=base_link, infl_link=infl_link
    )
    return GLM.from_formula(formula, data, family=fam, weights=weights, offset=offset, **kwargs)


hurdle_poisson = _formula_shortcut(HurdlePoisson, ("zero_link",), alias="hurdle_poisson")
hurdle_negative_binomial = _formula_shortcut(
    HurdleNegativeBinomial, ("zero_link",), alias="hurdle_negative_binomial"
)
hurdle_gamma = _formula_shortcut(HurdleGamma, ("zero_link",), alias="hurdle_gamma")
hurdle_lognormal = _formula_shortcut(
    HurdleLogNormal, ("zero_link",), alias="hurdle_lognormal"
)
hurdle_beta = _formula_shortcut(HurdleBeta, ("zero_link",), alias="hurdle_beta")
hurdle_geometric = _formula_shortcut(
    HurdleGeometric, ("zero_link",), alias="hurdle_geometric"
)

truncated_poisson = _formula_shortcut(TruncatedPoisson, alias="truncated_poisson")
truncated_negative_binomial = _formula_shortcut(
    TruncatedNegativeBinomial, alias="truncated_negative_binomial"
)
truncated_geometric = _formula_shortcut(
    TruncatedGeometric, alias="truncated_geometric"
)
truncated_gaussian = _formula_shortcut(TruncatedGaussian, alias="truncated_gaussian")


def generalized_gamma(
    formula: str,
    data: pd.DataFrame,
    sigma_formula: str = "~ 1",
    Q_formula: str = "~ 1",
    link_mu: str = "identity",
    link_sigma: str = "log",
    link_Q: str = "identity",
    weights: Any = None,
    offset: Any = None,
    cens: Any = None,
    entry: Any = None,
    **kwargs: Any,
) -> GLM:
    """Build a GeneralizedGamma GLM from R-style formulas.

    GeneralizedGamma has three linear predictors (location ``mu``, scale
    ``sigma`` and shape ``Q``); the main formula models ``mu`` while
    ``sigma_formula`` and ``Q_formula`` model the other two.

    Parameters
    ----------
    formula : str
        Formula for the location submodel, e.g. ``"y ~ x"``.
    data : pandas.DataFrame
        Data containing all variables referenced by the formulas.
    sigma_formula, Q_formula : str, optional
        Formulas for the scale and shape submodels, e.g. ``"~ x"``.
        They default to intercept-only (``"~ 1"``); use ``"~ 0"`` or
        ``"~ - 1"`` to drop the intercept.
    link_mu, link_sigma, link_Q : str
        Link functions for the ``mu``, ``sigma`` and ``Q`` submodels.
        Defaults: ``"identity"``, ``"log"``, ``"identity"``.
    weights, offset : str, array-like or None, optional
        Prior weights and offset; a string names a column of ``data``.
    cens, entry : str, array-like or None, optional
        Event/censoring status (1 = event, 0 = right-censored) and entry
        (left-truncation) times. A string names a column of ``data``; the
        values are stored on the model and used by ``.fit()`` unless
        overridden there.
    **kwargs
        Further ``GLM`` options forwarded to
        :meth:`GLM.from_formula_gengamma`.

    Returns
    -------
    GLM
        Unfitted model; call ``.fit()`` to estimate it.
    """
    return GLM.from_formula_gengamma(
        formula,
        data,
        sigma_formula=sigma_formula,
        Q_formula=Q_formula,
        link_mu=link_mu,
        link_sigma=link_sigma,
        link_Q=link_Q,
        weights=weights,
        offset=offset,
        cens=cens,
        entry=entry,
        **kwargs,
    )


__all__ = [
    "glm",
    "gaussian",
    "ols",
    "OLS",
    "binomial",
    "poisson",
    "gamma",
    "inverse_gaussian",
    "negative_binomial",
    "tweedie",
    "exponential",
    "pareto",
    "inverse_gamma",
    "quasi_poisson",
    "quasi_binomial",
    "quasi_gamma",
    "quasi_inverse_gaussian",
    "quasi_gaussian",
    "beta",
    "lognormal",
    "weibull",
    "gumbel",
    "loglogistic",
    "student_t",
    "skew_normal",
    "skew_t",
    "logistic_distribution",
    "simplex",
    "multinomial",
    "ordinal",
    "zero_inflated_poisson",
    "zero_inflated_negative_binomial",
    "value_inflated_poisson",
    "value_inflated_negative_binomial",
    "hurdle_poisson",
    "hurdle_negative_binomial",
    "hurdle_gamma",
    "hurdle_lognormal",
    "hurdle_beta",
    "hurdle_geometric",
    "truncated_poisson",
    "truncated_negative_binomial",
    "truncated_geometric",
    "truncated_gaussian",
    "generalized_gamma",
    "GLM",
    "GLMPredictionResults",
    "GLMInfluence",
    "aic",
    "bic",
    "deviance",
    "llf",
    "llnull",
    "scale",
    "resid_pearson",
    "resid_deviance",
    "resid_response",
    "resid_anscombe",
    "resid_quantile",
    "tvalues",
    "pvalues",
    "mu",
    "conf_int",
    "predict",
    "get_prediction",
    "wald_test",
    "anova",
    "pseudo_rsquared",
    "get_influence",
    "converged",
    "FamilyCode",
    "Family",
    "CustomFamily",
    "Gaussian",
    "Binomial",
    "Poisson",
    "Gamma",
    "InverseGaussian",
    "NegativeBinomial",
    "Tweedie",
    "Exponential",
    "Pareto",
    "InverseGamma",
    "QuasiFamily",
    "QuasiPoisson",
    "QuasiBinomial",
    "QuasiGamma",
    "QuasiInverseGaussian",
    "QuasiGaussian",
    "QuasiCustom",
    "MleFamily",
    "Beta",
    "LogNormal",
    "Weibull",
    "Gumbel",
    "LogLogistic",
    "StudentT",
    "SkewNormal",
    "SkewT",
    "LogisticDistribution",
    "Simplex",
    "Multinomial",
    "Ordinal",
    "ZeroInflatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ValueInflatedPoisson",
    "ValueInflatedNegativeBinomial",
    "GeneralizedGamma",
    "HurdlePoisson",
    "HurdleNegativeBinomial",
    "HurdleGamma",
    "HurdleLogNormal",
    "HurdleBeta",
    "HurdleGeometric",
    "TruncatedPoisson",
    "TruncatedNegativeBinomial",
    "TruncatedGeometric",
    "TruncatedGaussian",
]
