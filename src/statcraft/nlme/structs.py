"""Correlation, variance and random-effects covariance structures with nlme names.

This module provides lightweight, engine-agnostic descriptors for the three
kinds of structure accepted by :mod:`statcraft.nlme.models`:

* :class:`CorStruct` -- within-group correlation of the residuals
  (``corAR1``, ``corCAR1``, ``corCompSymm``, ``corSymm``, ``corARMA``,
  the spatial structures ``corExp``/``corGaus``/``corLin``/``corRatio``/
  ``corSpher`` and :func:`corCustom`);
* :class:`VarStruct` -- variance function / heteroscedasticity
  (``varIdent``, ``varFixed``, ``varPower``, ``varExp``, ``varConstPower``,
  ``varConstProp``, ``varComb`` and :func:`varCustom`);
* :class:`PdStruct` -- parametrization of the random-effects covariance
  matrix ``D`` (``pdLogChol``, ``pdSymm``, ``pdChol``, ``pdNatural``,
  ``pdCompSymm``, ``pdDiag``, ``pdIdent``, :func:`pdBlocked` and
  :func:`pdCustom`).

The factories mirror the R ``nlme`` constructors and only build descriptors;
the actual matrix/optimization work happens in the C++ engine.  ``form``
strings use the same syntax as R, e.g. ``"~ 1 | Subject"`` (grouping only),
``"~ age | Subject"`` (position/time covariate) or ``"~ x + y | g"``
(spatial coordinates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class CorStruct:
    """Descriptor of a residual correlation structure.

    Attributes
    ----------
    kind:
        Internal identifier of the structure (``"AR1"``, ``"CAR1"``,
        ``"CompoundSymmetry"``, ``"Symm"``, ``"ARMA"``, ``"Exp"``, ``"Gaus"``,
        ``"Lin"``, ``"Ratio"``, ``"Spher"`` or ``"Custom"``).
    form:
        R-style formula such as ``"~ age | Subject"``.  The left-hand side
        names the position/time (or the spatial coordinates) and the part
        after ``|`` names the grouping factor(s).
    p, q:
        Orders of an ARMA structure (``corARMA`` only).
    nugget:
        Whether a nugget effect is added (spatial structures only).
    metric:
        Distance metric used by the spatial structures: ``"euclidean"`` or
        ``"maximum"``.
    custom:
        For ``kind == "Custom"``: dictionary with the user-supplied
        ``matrix``/``deriv`` callables and the initial values.
    """

    kind: str
    form: Optional[str] = None
    p: int = 0
    q: int = 0
    nugget: bool = False
    metric: str = "euclidean"
    custom: Optional[dict] = None


def corAR1(form=None):
    """First-order autoregressive correlation ``corAR1`` of R ``nlme``.

    Parameters
    ----------
    form:
        ``"~ time | group"`` (or ``"~ 1 | group"``); the position covariate is
        used to order observations within each group.

    Returns
    -------
    CorStruct
    """
    return CorStruct("AR1", form)


def corCAR1(form=None):
    """Continuous-time first-order autoregressive correlation ``corCAR1``.

    Parameters
    ----------
    form:
        ``"~ time | group"``; the correlation decays exponentially with the
        time distance between observations in the same group.

    Returns
    -------
    CorStruct
    """
    return CorStruct("CAR1", form)


def corCompSymm(form=None):
    """Compound-symmetry (exchangeable) correlation ``corCompSymm``.

    All pairs of observations within a group share the same correlation.

    Parameters
    ----------
    form:
        ``"~ 1 | group"``.

    Returns
    -------
    CorStruct
    """
    return CorStruct("CompoundSymmetry", form)


def corSymm(form=None):
    """General (unstructured) correlation ``corSymm``.

    Parameters
    ----------
    form:
        ``"~ 1 | group"``; a full correlation matrix is estimated for each
        group.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Symm", form)


def corARMA(form=None, p=0, q=0):
    """Autoregressive moving-average correlation ``corARMA(p, q)``.

    Parameters
    ----------
    form:
        ``"~ time | group"``.
    p, q:
        Autoregressive and moving-average orders.

    Returns
    -------
    CorStruct
    """
    return CorStruct("ARMA", form, p=p, q=q)


def corExp(form=None, nugget=False, metric="euclidean"):
    """Exponential spatial correlation ``corExp``.

    Parameters
    ----------
    form:
        ``"~ x + y | group"`` (one or more spatial coordinates).
    nugget:
        Add a nugget effect.
    metric:
        ``"euclidean"`` or ``"maximum"`` distance.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Exp", form, nugget=nugget, metric=metric)


def corGaus(form=None, nugget=False, metric="euclidean"):
    """Gaussian spatial correlation ``corGaus``.

    Parameters
    ----------
    form:
        ``"~ x + y | group"``.
    nugget:
        Add a nugget effect.
    metric:
        ``"euclidean"`` or ``"maximum"`` distance.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Gaus", form, nugget=nugget, metric=metric)


def corLin(form=None, nugget=False, metric="euclidean"):
    """Linear spatial correlation ``corLin``.

    Parameters
    ----------
    form:
        ``"~ x + y | group"``.
    nugget:
        Add a nugget effect.
    metric:
        ``"euclidean"`` or ``"maximum"`` distance.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Lin", form, nugget=nugget, metric=metric)


def corRatio(form=None, nugget=False, metric="euclidean"):
    """Rational-quadratic spatial correlation ``corRatio``.

    Parameters
    ----------
    form:
        ``"~ x + y | group"``.
    nugget:
        Add a nugget effect.
    metric:
        ``"euclidean"`` or ``"maximum"`` distance.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Ratio", form, nugget=nugget, metric=metric)


def corSpher(form=None, nugget=False, metric="euclidean"):
    """Spherical spatial correlation ``corSpher``.

    Parameters
    ----------
    form:
        ``"~ x + y | group"``.
    nugget:
        Add a nugget effect.
    metric:
        ``"euclidean"`` or ``"maximum"`` distance.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Spher", form, nugget=nugget, metric=metric)


def corCustom(matrix: Callable, n_params: int, form: Optional[str] = None, start: Optional[Sequence[float]] = None, name: str = "custom",
              deriv: Optional[Callable] = None) -> CorStruct:
    """User-defined correlation structure.

    Parameters
    ----------
    matrix:
        ``matrix(theta, t, coords)`` must return the ``(n, n)`` correlation
        matrix of a single block of observations.  ``theta`` holds the
        unconstrained parameters, ``t`` the positions/times and ``coords`` an
        ``(n, d)`` matrix or an empty array.
    n_params:
        Number of correlation parameters in ``theta``.
    form:
        Optional formula supplying the time/spatial covariates and the
        grouping factor (``"~ t | g"``).
    start:
        Optional initial values for the parameters.
    name:
        Label used in tables and summaries.
    deriv:
        Optional analytic derivative
        ``deriv(theta, t, coords, j) -> d matrix / d theta_j`` (``(n, n)``).
        When omitted, the gradient is computed by finite differences.

    Returns
    -------
    CorStruct
    """
    return CorStruct("Custom", form, custom=dict(name=name, n_params=n_params, matrix=matrix, deriv=deriv,
                                                 start=None if start is None else np.asarray(start, float)))


@dataclass
class VarStruct:
    """Descriptor of a variance function (heteroscedasticity model).

    Attributes
    ----------
    kind:
        Internal identifier (``"Ident"``, ``"Fixed"``, ``"Power"``, ``"Exp"``,
        ``"ConstPower"``, ``"ConstProp"``, ``"Comb"`` or ``"Custom"``).
    form:
        R-style formula such as ``"~ x"``, ``"~ fitted(.)"`` or
        ``"~ 1 | g"``.
    parts:
        Components of a ``varComb`` structure, in order.
    custom:
        For ``kind == "Custom"``: dictionary with the user-supplied
        ``scales``/``dscales`` callables and the initial values.
    """

    kind: str
    form: Optional[str] = None
    parts: list = field(default_factory=list)
    custom: Optional[dict] = None


def varIdent(form="~1|1"):
    """Group-wise variance ratios ``varIdent`` (one variance per stratum).

    Parameters
    ----------
    form:
        ``"~ 1 | g"``; the first stratum is the reference (ratio 1) and the
        remaining strata get one ratio parameter each.

    Returns
    -------
    VarStruct
    """
    return VarStruct("Ident", form)


def varFixed(form):
    """Fixed (known) variance weights ``varFixed``.

    Parameters
    ----------
    form:
        ``"~ x"``; the residual variance is proportional to ``x``.

    Returns
    -------
    VarStruct
    """
    return VarStruct("Fixed", form)


def varPower(form="~fitted(.)"):
    """Power-of-the-mean variance ``varPower``.

    Parameters
    ----------
    form:
        ``"~ x"`` for a covariate or ``"~ fitted(.)"`` for the fitted values;
        ``Var = sigma^2 * |x|^(2 * delta)``.

    Returns
    -------
    VarStruct
    """
    return VarStruct("Power", form)


def varExp(form="~fitted(.)"):
    """Exponential-of-the-mean variance ``varExp``.

    Parameters
    ----------
    form:
        ``"~ x"`` or ``"~ fitted(.)"``; ``Var = sigma^2 * exp(2 * delta * x)``.

    Returns
    -------
    VarStruct
    """
    return VarStruct("Exp", form)


def varConstPower(form="~fitted(.)"):
    """Constant-plus-power variance ``varConstPower``.

    Parameters
    ----------
    form:
        ``"~ x"`` or ``"~ fitted(.)"``.  Two parameters are estimated: the
        constant term and the power ``delta``.

    Returns
    -------
    VarStruct
    """
    return VarStruct("ConstPower", form)


def varConstProp(form="~fitted(.)"):
    """Constant-plus-proportional variance ``varConstProp``.

    Parameters
    ----------
    form:
        ``"~ x"`` or ``"~ fitted(.)"``.  Two parameters are estimated: the
        constant term and the proportionality coefficient.

    Returns
    -------
    VarStruct
    """
    return VarStruct("ConstProp", form)


def varComb(*parts):
    """Combine several variance functions multiplicatively ``varComb``.

    Parameters
    ----------
    *parts:
        :class:`VarStruct` instances whose relative standard deviations are
        multiplied together.

    Returns
    -------
    VarStruct
    """
    return VarStruct("Comb", parts=list(parts))


def varCustom(scales: Callable, n_params: int, form: Optional[str] = None, start: Optional[Sequence[float]] = None, name: str = "custom",
              dscales: Optional[Callable] = None) -> VarStruct:
    """User-defined variance function.

    Parameters
    ----------
    scales:
        ``scales(theta, cov, strata, n) -> relative standard deviations (n,) > 0``.
    n_params:
        Number of variance parameters in ``theta``.
    form:
        Supplies the covariate (``'~x'``) and, optionally, the strata
        (``'~x|g'``); ``cov`` is empty when ``form`` is ``None``.
    start:
        Optional initial values for the parameters.
    name:
        Label used in tables and summaries.
    dscales:
        Optional analytic derivative
        ``dscales(theta, cov, strata, n, j) -> d scales / d theta_j`` (``n,``).

    Returns
    -------
    VarStruct
    """
    return VarStruct("Custom", form, custom=dict(name=name, n_params=n_params, scales=scales, needs_cov=form is not None, dscales=dscales,
                                                 start=None if start is None else np.asarray(start, float)))


@dataclass
class PdStruct:
    """Descriptor of the random-effects covariance parametrization ``D``.

    Attributes
    ----------
    kind:
        Internal identifier: ``"LogCholesky"``, ``"MatrixLog"``,
        ``"Cholesky"``, ``"Natural"``, ``"CompoundSymmetry"``, ``"Diag"``,
        ``"Ident"``, ``"Blocked"`` or ``"Custom"``.
    sizes, kinds:
        For ``pdBlocked``: sizes and per-block parametrizations.
    custom:
        For ``kind == "Custom"``: dictionary with the user-supplied
        ``nparams``/``factor``/``dfactor`` callables.
    """

    kind: str                       # LogCholesky, MatrixLog, Cholesky, Natural, CompoundSymmetry, Diag, Ident, Blocked, Custom
    sizes: Optional[list] = None
    kinds: Optional[list] = None
    custom: Optional[dict] = None


def pdLogChol():
    """Log-Cholesky parametrization of ``D`` (the R ``nlme`` default).

    Returns
    -------
    PdStruct
    """
    return PdStruct("LogCholesky")


def pdSymm():
    """General positive-definite matrix ``D`` via matrix logarithm.

    Returns
    -------
    PdStruct
    """
    return PdStruct("MatrixLog")


def pdChol():
    """Cholesky parametrization of a general ``D``.

    Returns
    -------
    PdStruct
    """
    return PdStruct("Cholesky")


def pdNatural():
    """Natural (standard-deviation/correlation) parametrization of ``D``.

    Returns
    -------
    PdStruct
    """
    return PdStruct("Natural")


def pdCompSymm():
    """Compound-symmetry parametrization of ``D``.

    Returns
    -------
    PdStruct
    """
    return PdStruct("CompoundSymmetry")


def pdDiag():
    """Diagonal ``D`` (uncorrelated random effects).

    Returns
    -------
    PdStruct
    """
    return PdStruct("Diag")


def pdIdent():
    """Scaled identity ``D`` (a single variance, as in ``(1 | g)``).

    Returns
    -------
    PdStruct
    """
    return PdStruct("Ident")


def pdBlocked(sizes: Sequence[int], kinds: Sequence[str | PdStruct]) -> PdStruct:
    """Block-diagonal ``D`` with one parametrization per block.

    Parameters
    ----------
    sizes:
        Number of random effects in each block.
    kinds:
        Parametrization of each block (a :class:`PdStruct` or its name).

    Returns
    -------
    PdStruct
    """
    return PdStruct("Blocked", sizes=list(sizes), kinds=[k.kind if isinstance(k, PdStruct) else k for k in kinds])


def pdCustom(nparams: Callable[[int], int], factor: Callable, start: Optional[Callable] = None, from_D: Optional[Callable] = None,
             name: str = "custom", dfactor: Optional[Callable] = None) -> PdStruct:
    """User-defined parametrization of ``D``.

    Parameters
    ----------
    nparams:
        ``nparams(q) -> number of parameters`` for a ``q``-dimensional effect.
    factor:
        ``factor(q, theta) -> lower-triangular q x q L`` with ``D = L L'``.
    start:
        Optional function returning initial values.
    from_D:
        Optional function mapping a covariance matrix ``D`` to ``theta``.
    name:
        Label used in tables and summaries.
    dfactor:
        Optional analytic derivative ``dfactor(q, theta, j) -> d L / d theta_j``
        (``q x q``).  When omitted, finite differences are used.

    Returns
    -------
    PdStruct
    """
    return PdStruct("Custom", custom=dict(name=name, nparams=nparams, factor=factor, start=start, from_D=from_D, dfactor=dfactor))


_PD_ALIAS = {"logchol": "LogCholesky", "symm": "MatrixLog", "matrixlog": "MatrixLog", "chol": "Cholesky", "cholesky": "Cholesky",
             "natural": "Natural", "compsymm": "CompoundSymmetry", "diag": "Diag", "ident": "Ident"}


def as_pd(x) -> PdStruct:
    """Coerce ``x`` to a :class:`PdStruct`.

    Parameters
    ----------
    x:
        A :class:`PdStruct` (returned unchanged) or a case-insensitive name
        such as ``"logchol"``, ``"pdSymm"``, ``"diag"`` or ``"ident"``.

    Returns
    -------
    PdStruct

    Raises
    ------
    ValueError
        If the name is unknown.
    TypeError
        If ``x`` is neither a :class:`PdStruct` nor a string.
    """
    if isinstance(x, PdStruct):
        return x
    if isinstance(x, str):
        k = _PD_ALIAS.get(x.lower().replace("pd", "", 1) if x.lower().startswith("pd") else x.lower())
        if k is None:
            raise ValueError(f"unknown pd: {x!r} (use {sorted(_PD_ALIAS)})")
        return PdStruct(k)
    raise TypeError("pd must be a PdStruct or a name ('logchol', 'symm', 'diag', ...)")
