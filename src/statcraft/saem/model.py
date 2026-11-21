"""Model definition: ``NLModel`` (structural function + statistical part) and compiled models."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

from .. import _saemfit as _core
from ..utils.modelview import PARAM_VIEW_NAMES, ModelView
from ._families import DEFAULT_FAMILY, SAEM_FAMILIES, normalize_family
from ._utils import (error_enum, error_name, parse_covariates, transform_name)
from ._trace import (NS, OPS as _TRACE_OPS, Roles, SeqNS, SinglePred, TraceError, kernel_argnames,
                     trace_program)


def _trace_ops():
    """Operation table (name -> code) shared with the C++ expression evaluator."""
    return _TRACE_OPS

# ---------------------------------------------------------------------------
# Registry of compiled models (C++, no GIL, multithreaded)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuiltinSpec:
    """Specification of a model compiled in C++.

    Attributes
    ----------
    params : tuple[str, ...]
        Parameter names, in model order.
    transform : tuple[str, ...]
        Default transformation of each parameter.
    description : str
        Human-readable formula or description.
    kind : {"algebraic", "pk"}
        Whether the model is a closed-form algebraic curve or a linear compartmental PK model.
    """

    params: tuple
    transform: tuple
    description: str
    kind: str          # 'algebraic' | 'pk'


BUILTINS: dict[str, BuiltinSpec] = {
    "exp_decay": BuiltinSpec(("A", "k"), ("log", "log"), "y = A exp(-k x)", "algebraic"),
    "biexp": BuiltinSpec(("A1", "k1", "A2", "k2"), ("log",) * 4, "y = A1 exp(-k1 x) + A2 exp(-k2 x)", "algebraic"),
    "emax": BuiltinSpec(("E0", "Emax", "EC50"), ("identity", "identity", "log"), "y = E0 + Emax x/(EC50 + x)", "algebraic"),
    "sigmoid_emax": BuiltinSpec(("E0", "Emax", "EC50", "gamma"), ("identity", "identity", "log", "log"),
                                "y = E0 + Emax x^g/(EC50^g + x^g)", "algebraic"),
    "logistic": BuiltinSpec(("Asym", "xmid", "scal"), ("log", "identity", "log"), "y = Asym/(1 + exp((xmid - x)/scal))", "algebraic"),
    "gompertz": BuiltinSpec(("Asym", "b2", "b3"), ("log", "log", "logit"), "y = Asym exp(-b2 b3^x)", "algebraic"),
    "michaelis_menten": BuiltinSpec(("Vmax", "Km"), ("log", "log"), "y = Vmax x/(Km + x)", "algebraic"),
    "pk1_iv": BuiltinSpec(("CL", "V"), ("log", "log"), "1 compartment, IV (bolus/infusion); cmt 1 = central; y = A1/V", "pk"),
    "pk1_oral": BuiltinSpec(("ka", "CL", "V"), ("log",) * 3, "1 compartment, oral first-order; cmt 1 = depot; y = A2/V", "pk"),
    "pk2_iv": BuiltinSpec(("CL", "V1", "Q", "V2"), ("log",) * 4, "2 compartments, IV; cmt 1 = central", "pk"),
    "pk2_oral": BuiltinSpec(("ka", "CL", "V1", "Q", "V2"), ("log",) * 5, "2 compartments, oral; cmt 1 = depot", "pk"),
}


def builtin_models() -> dict[str, str]:
    """Available compiled models: ``{name: description}``."""
    return {k: f"{v.description}  params={list(v.params)}" for k, v in BUILTINS.items()}


class Doses:
    """Subject dose record (column arrays). ``None`` in the function => no doses."""
    __slots__ = ("time", "cmt", "amt", "dur", "ii", "ss", "modeled")

    def __init__(self, arr: np.ndarray):
        self.time, self.cmt, self.amt = arr[:, 0], arr[:, 1].astype(int), arr[:, 2]
        self.dur, self.ii = arr[:, 3], arr[:, 4]
        self.ss, self.modeled = arr[:, 5].astype(int), arr[:, 6].astype(int)

    def __len__(self):
        return len(self.time)

    def __repr__(self):
        return f"Doses(n={len(self)}, time={self.time}, amt={self.amt})"

    def superpose(self, time, kernel):
        """Linear superposition: ``sum_d kernel(dt, amt, dur, cmt)`` over doses with ``time >= t_d``.

        ``kernel`` is an ELEMENTWISE function of (a subset of) ``dt`` (= time - dose_time), ``amt``, ``dur``
        (infusion duration, 0 = bolus) and ``cmt`` (1-based); it may use the parameters by closure::

            def pk(ka, CL, V, time, doses):
                k = CL / V
                return doses.superpose(time, lambda dt, amt: amt / V * ka / (ka - k) * (np.exp(-k * dt) - np.exp(-ka * dt)))

        This is the ONLY way to use doses that compiles to C++ (no GIL, multithreaded). It does not support SS
        or modeled RATE."""
        names = kernel_argnames(kernel)
        t = np.asarray(time, float)
        scalar = t.ndim == 0
        t = t.reshape(-1)
        if np.any(self.ss != 0):
            raise ValueError("doses.superpose does not support steady state (SS): use compile=False or a built-in model")
        if np.any(self.modeled != 0):
            raise ValueError("doses.superpose does not support RATE = -1/-2 (modeled rate or duration)")
        if len(self) == 0:
            out = np.zeros(t.shape[0])
            return float(out[0]) if scalar else out
        dt = t[:, None] - self.time[None, :]
        cols = {"dt": dt, "amt": self.amt[None, :], "dur": self.dur[None, :], "cmt": self.cmt[None, :]}
        with np.errstate(all="ignore"):
            k = np.broadcast_to(np.asarray(kernel(**{n: cols[n] for n in names}), float), dt.shape)
            out = np.where(dt >= 0.0, k, 0.0).sum(axis=1)
        return float(out[0]) if scalar else out


_EMPTY_DOSES = Doses(np.zeros((0, 7)))

# roles of the user function arguments; _XV is the portable ``x`` view
_PSI, _X, _COV, _DOSES, _YTYPE, _EXTRA, _Y, _PNS, _DNS, _PRD, _XV = range(11)
# reserved names: ``params`` and ``predictors`` are named sequences (Emax, ED50 = params; params.Emax),
# ``data`` gives access to columns and covariates by name; ``doses`` / ``ytype``
_RESERVED = {"doses": _DOSES, "ytype": _YTYPE, "params": _PNS, "data": _DNS, "predictors": _PRD}


@dataclass
class _Plan:
    """How the arguments of a callback map to parameters, columns, covariates and reserved names.

    Attributes
    ----------
    names : list[str]
        Argument names, in signature order.
    kinds : list[int]
        Role code of each argument (``_PSI``, ``_X``, ``_COV``, ...).
    idx : list[int]
        Index associated with each role (parameter, predictor, covariate, ...).
    """

    names: list
    kinds: list
    idx: list


def _signature_args(fn: Callable) -> list[tuple[str, bool]]:
    """[(name, has_default)] of the named arguments of ``fn``."""
    out = []
    for n, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append((n, p.default is not p.empty))
    return out


def _is_vector_signature(fn: Callable, param_names: Optional[Sequence[str]] = None) -> bool:
    """True for the portable two-argument form ``f(x, p)``.

    A first argument named ``x``/``X`` and a reserved parameter-view name are
    recognized always. More than one parameter has no valid named-form
    interpretation, so any two names then denote the vector form. With one
    parameter, a reserved second name keeps ``f(x, params)`` distinct from
    ``f(time, a)`` unless ``a`` itself is that reserved name.
    """
    args = [n for n, _ in _signature_args(fn)]
    if len(args) != 2:
        return False
    if args[0] in ("x", "X") and args[1] in PARAM_VIEW_NAMES:
        return True
    if param_names is None:
        return False
    if len(param_names) > 1:
        return True
    return args[1] in PARAM_VIEW_NAMES and args[1] not in param_names


# ---------------------------------------------------------------------------
# NLModel
# ---------------------------------------------------------------------------
class NLModel:
    """Nonlinear mixed-effects model.

    Parameters
    ----------
    fn : callable
        Structural function. The portable two-argument form is the recommended
        standard, shared with NLS and NLME::

            def model(x, p):
                Time, Dose = x              # x["Time"], x.Time, x[0] also work
                ka, CL, V = p               # p["CL"], p.CL, p[0] also work
                return ...

        ``x`` contains the observed exogenous columns (the order is
        ``predictors=[...]`` when declared, otherwise every data column with
        time first, then the covariates); ``x.doses`` exposes the dose record
        when one is present. Both arguments can be unpacked, indexed by name
        or position, and converted with ``len()`` / ``np.asarray()``.
        Arguments are otherwise resolved **by name**:

        * names in ``params``            -> individual parameter value (float)
        * time column (``time``/``t``)   -> array with the subject's times
        * other ``data`` columns         -> array (if it varies within subject) or float (constant)
        * ``doses``                      -> :class:`Doses` with the subject's dose record
        * ``ytype``                      -> array of responses (multi-response models)
        * ``params``                     -> ALL parameters, as a sequence (SciPy style)
        * ``predictors``                 -> the columns in ``predictors=[...]``, in that order; with a
          single column it behaves directly as that column (``x``), and ``x[0]`` also still works
        * ``data``                       -> access to columns and covariates by name (``data.dose``)

        The two-argument form uses ``predictors=[...]`` (or the model's own
        ``predictors``) to define the contents and order of ``x``; declare the
        time column there as well if the model needs it. The argument names need
        not be ``x``/``p`` when more than one parameter is declared; with one,
        ``p``/``params``/``parameters``/``theta`` also mark the parameter view,
        so ``a = params`` works for a single-parameter model. The older styles
        remain available and can be mixed. The **explicit** one
        (``def f(Emax, ED50, dose, time)``) is readable with few parameters.
        The **positional** one, as in SciPy, scales with many::

            def emax(params, predictors):
                Emax, ED50, Hill = params            # order of params=[...]
                dosis = predictors[0]                # order of predictors=[...]
                return Emax * dosis**Hill / (ED50**Hill + dosis**Hill)

            NLModel(emax, params=["Emax", "ED50", "Hill"], predictors=["dosis"])

        ``params`` and ``predictors`` are unpacked, indexed (``params[1]``, ``params[-1]``) and ``len()`` works;
        they also accept names (``params.ED50``). ``predictors`` is REQUIRED to use that argument and can be given
        in ``NLModel`` or in ``SAEM(..., predictors=[...])``; only the listed columns enter (include the time
        column if you want it: ``predictors=["time", "dose"]``). There is deliberately no ``**kwargs``: it would
        hide which names exist and could not be validated.

        The names in ``params=[...]`` are free: only the order matters, so
        ``params=["Vm", "K"]`` works the same as ``params=["Vmax", "Km"]``. The names
        ``params``, ``predictors``, ``data``, ``doses`` and ``ytype`` are reserved, but they can also be
        renamed: the first unrecognized argument (neither a parameter nor a data column) receives the full
        parameter sequence and the next one the predictor sequence, in signature order. For example::

            def michaelis_menten(p, x):          # p = parameters, x = predictors
                Vmax, Km = p
                return Vmax * x[0] / (Km + x[0])

            NLModel(michaelis_menten, params=["Vmax", "Km"], predictors=["dose"])

        If the renamed name matches a data column, it is resolved as a column (the column has priority); in
        that case use the reserved names for the sequences.

        With ``vectorized=True`` (default) it is called **once per subject** with numpy arrays and must
        return an array with one prediction per observation (the fast path). With
        ``vectorized=False`` it is called per observation with scalars.
    params : sequence[str]
        Names of the individual parameters (in model order).
    transform : str | sequence | dict
        Per-parameter transformation: ``"identity"``, ``"log"`` (log-normal), ``"logit"``, ``"probit"``.
        A single str applies to all; a dict ``{"cl": "log"}`` (unmentioned ones: identity).
    random : sequence[str] | None
        Parameters with a random effect (default: all).
    omega : "diagonal" | "full" | list[list[str]] | array
        Random-effect covariance structure: ``"diagonal"``, ``"full"``, blocks
        ``[["ka", "cl"], ["v"]]`` or a boolean (p x p) matrix of free covariances.
    error : str | sequence[str]
        ``"constant"``, ``"proportional"``, ``"exponential"``, ``"combined"``, ``"combined2"``, ``"power"``
        (one per response if there is a ``ytype``).
    covariates : list[str] | dict
        Covariate model, ``["cl ~ wt + sex", "v ~ wt"]``. Covariates are constant within
        subject; numeric ones enter on the phi scale (pre-transform the column: ``log(wt/70)``) and
        text/categorical ones are encoded with 0/1 variables (``reference={"sex": "F"}`` in ``saem``).
    family : Family object | type[Family]
        GLM-style family object or class; strings are rejected. Default:
        ``Gaussian``. Objects are translated to the SAEM likelihood:
        ``Binomial("logit")`` -> Bernoulli with an internal logistic link (``fn``
        returns eta), other binomial links -> probability-scale Bernoulli (``fn``
        returns ``P(y=1)``), ``Poisson`` -> Poisson, ``NegativeBinomial`` ->
        negative binomial, ``ZeroInflatedPoisson`` -> ZIP, ``Multinomial`` ->
        categorical, ``Ordinal`` -> ordinal, ``Weibull``/``LogNormal``/
        ``LogLogistic``/``GeneralizedGamma`` -> time-to-event (requires
        ``cumhazard=``), and ``CustomFamily`` -> user log-likelihood (requires
        ``loglik=``). In the non-Gaussian families ``fn`` is called **per
        observation** with scalars and returns: P(y=1) / logit / lambda / mean
        (negbin; uses 1 extra parameter log r) / ``(lambda, pi)`` (zip) / eta
        (ordinal, with ``cuts=g(params)``) / probability vector (categorical) /
        hazard (tte, with ``cumhazard=H(t, ...)``).
    loglik, sample : callable
        For ``family=CustomFamily``: ``loglik(y, <params>, <x>, extra=...)`` -> log p(y|...).
    compile : "auto" | True | False
        Compilation of ``fn`` (and ``loglik``) to C++: Gaussian prediction, ``family=CustomFamily`` (``loglik``) and all
        families (bernoulli, poisson, negbin, zip, ordinal, categorical, tte). With ``"auto"`` (default) the function is
        *traced* with objects that record numpy operations (``exp``, ``log``, ``where``,
        ``maximum``, ``**``, ``erf``...) and runs in the core without the GIL and on several threads (``n_threads``):
        ~10x faster than the callback. The result is validated numerically against the Python function itself
        before being accepted. Only **elementwise** functions of parameters, columns and covariates are
        compiled: ``if`` on model values, ``sum``/``cumsum``, indexing (``time[0]``), ``len``,
        ``float(...)`` or ``math.*`` cause the Python callback to be used (``compile_info`` explains the
        reason). ``True`` turns that failure into an error; ``False`` disables compilation.
        The function is traced ONCE: it must not depend on global state that changes during the fit.

        **Doses**: use ``doses.superpose(time, kernel)`` (see :meth:`Doses.superpose`): it sums ``kernel(dt, amt, dur,
        cmt)`` over the doses before each time and compiles (iterating ``doses.time``/``doses.amt`` by hand
        is not compilable). It does not support SS or modeled RATE. ``ytype`` is compiled (``np.where(ytype == 1, ...)``);
        in ``family=CustomFamily`` so are ``y`` and ``extra[i]``. The ``sample`` and ``cuts`` functions may remain in Python.
    """

    def __init__(self, fn: Optional[Callable] = None, *, params: Sequence[str],
                 transform: Union[str, Sequence, Mapping, None] = "identity",
                 random: Optional[Sequence[str]] = None,
                 omega: Union[str, Sequence, np.ndarray] = "diagonal",
                 error: Union[str, Sequence[str]] = "proportional",
                 covariates=None, predictors: Optional[Sequence[str]] = None,
                 family: object = DEFAULT_FAMILY, vectorized: bool = True,
                 cumhazard: Optional[Callable] = None, cuts: Optional[Callable] = None,
                 loglik: Optional[Callable] = None, sample: Optional[Callable] = None,
                 n_extra: int = 0, tte_start: Optional[str] = None,
                 cluster_omega: Union[str, Sequence] = "diagonal",
                 name: Optional[str] = None, _builtin: Optional[tuple] = None,
                 compile: Union[bool, str] = "auto"):
        self.params = [str(p) for p in params]
        if len(set(self.params)) != len(self.params):
            raise ValueError("params with duplicate names")
        self.fn = fn
        self.name = name or getattr(fn, "__name__", None) or (_builtin[0] if _builtin else "model")
        self.family, family_updates = normalize_family(
            family, n_extra=n_extra, cuts=cuts, cumhazard=cumhazard, loglik=loglik
        )
        self.vectorized = bool(vectorized)
        self.predictors = [str(c) for c in predictors] if predictors else None   # order of the `predictors` argument
        self._pred_order = None             # order resolved when preparing the data (see data.prepare)
        self._time_synthetic = False        # True if the data have no time (see data.prepare)
        self.cumhazard, self.cuts, self.loglik_fn, self.sample_fn = cumhazard, cuts, loglik, sample
        self.n_extra = int(family_updates.get("n_extra", n_extra))
        self.tte_start = tte_start
        self._builtin = _builtin            # (name, x, dose)
        if compile not in (True, False, "auto"):
            raise ValueError("compile: True, False or 'auto'")
        self.compile = compile
        self._expr_ok = False               # True if the function was compiled to C++ (decided in _to_core)
        self._compile_msg = ""

        p = len(self.params)
        # --- transformations ---
        if isinstance(transform, Mapping):
            bad = set(transform) - set(self.params)
            if bad:
                raise ValueError(f"transform: unknown parameters {sorted(bad)}")
            self.transform = [transform_name(transform.get(n, "identity")) for n in self.params]
        elif isinstance(transform, (str, type(None))):
            self.transform = [transform_name(transform)] * p
        else:
            if len(transform) != p:
                raise ValueError(f"transform: expected {p} values")
            self.transform = [transform_name(t) for t in transform]
        # --- random effects ---
        if random is None:
            self.random = list(self.params)
        else:
            bad = set(random) - set(self.params)
            if bad:
                raise ValueError(f"random: unknown parameters {sorted(bad)}")
            self.random = [n for n in self.params if n in set(random)]
        self.omega = omega
        self.cluster_omega = cluster_omega
        # --- error ---
        self.errors = [error_name(e) for e in ([error] if isinstance(error, str) else error)]
        self.covariates = parse_covariates(covariates, self.params)
        self._validate()

    # ------------------------------------------------------------------ constructors
    @classmethod
    def builtin(cls, name: str, *, x: Optional[str] = None, dose: Optional[str] = None,
                transform=None, **kw) -> "NLModel":
        """Model compiled in C++ (fast, multithreaded, no GIL). See :func:`builtin_models`.

        ``x``: regressor column in the algebraic models (default: time).
        ``dose``: column (constant within subject) with the single dose at t=0 if there is no event
        record (``events=``) in the PK models.
        """
        if name not in BUILTINS:
            raise ValueError(f"unknown compiled model {name!r}. Options: {sorted(BUILTINS)}")
        spec = BUILTINS[name]
        params = kw.pop("params", list(spec.params))
        if len(params) != len(spec.params):
            raise ValueError(f"{name}: expected {len(spec.params)} parameters")
        if transform is None:
            transform = list(spec.transform)
        return cls(None, params=params, transform=transform, name=name, _builtin=(name, x, dose), **kw)

    # ------------------------------------------------------------------ validation
    def _validate(self):
        """Check that the family and the required callbacks are consistent."""
        if self._builtin is None and self.family != "custom" and self.fn is None:
            raise ValueError("missing model function (fn)")
        if self.family == "tte" and self.cumhazard is None:
            raise ValueError("survival families require cumhazard=<cumulative hazard function>")
        if self.family == "ordinal" and self.cuts is None:
            raise ValueError("Ordinal() requires cuts=<cut-point function>")
        if self.family == "custom" and self.loglik_fn is None:
            raise ValueError("CustomFamily requires loglik=<function>")
        if self.family not in SAEM_FAMILIES:
            raise ValueError(f"unknown family: {self.family!r}")
        if self.family == "negbin" and self.n_extra == 0:
            self.n_extra = 1
        if self._builtin is not None and self.family != "gaussian":
            raise ValueError("compiled models are Gaussian")

    @property
    def is_compiled(self) -> bool:
        """True if there are no Python callbacks (the GIL is released and threads are allowed): built-in
        model, or user function compiled to C++ (see ``compile``)."""
        return (self._builtin is not None and self.family == "gaussian") or self._expr_ok

    @property
    def compile_info(self) -> str:
        """Compilation status of ``fn`` to C++ (filled in when fitting): compiled, or the reason
        why the Python callback is used."""
        return self._compile_msg or "not evaluated (decided when fitting the model)"

    @property
    def n_params(self) -> int:
        """Number of structural-model parameters."""
        return len(self.params)

    # ------------------------------------------------------------------ signature analysis
    def arg_names(self) -> set[str]:
        """Names (not parameters) that the function asks for: candidates for data columns.

        A name that is not a data column can be used as a positional alias of
        ``params``/``predictors`` (see :class:`NLModel`); in that case it is not required as a column.
        """
        names: set[str] = set()
        for fn in (self.fn, self.cumhazard, self.loglik_fn, self.sample_fn):
            if fn is None or _is_vector_signature(fn, self.params):
                continue
            for n, _ in _signature_args(fn):
                if n not in self.params and n not in _RESERVED and n not in ("y", "extra", "seed"):
                    names.add(n)
        return names

    def _plan(self, fn: Callable, x_names: Sequence[str], cov_names: Sequence[str], *,
              allow_y=False, allow_extra=False, allow_seed=False, allow_seq_aliases=False,
              tvar: Optional[str] = None) -> _Plan:
        """Map the arguments of ``fn`` to the roles understood by the core.

        Parameters are matched by name, declared predictors/covariates by name, the reserved
        ``params``/``predictors``/``data`` arguments by identity, and the optional ``y``,
        ``extra`` and ``seed`` arguments are enabled explicitly. Unmatched positional arguments
        can be bound to ``params`` and ``predictors`` in order when ``allow_seq_aliases`` is set.

        Returns
        -------
        _Plan
            Argument names together with their role code and index.
        """
        names, kinds, idx = [], [], []
        pending = []
        has_psi = has_pns = has_prd = False
        sig_args = _signature_args(fn)
        # Portable two-argument convention, shared with NLS and NLME: x is a
        # named view of the exogenous data and p a view of all parameters.
        if allow_seq_aliases and _is_vector_signature(fn, self.params):
            return _Plan([n for n, _ in sig_args], [_XV, _PNS], [0, 0])
        for n, has_default in sig_args:
            if allow_y and n == "y":
                kind, i = _Y, 0
            elif allow_extra and n == "extra":
                kind, i = _EXTRA, 0
            elif allow_seed and n == "seed":
                kind, i = _EXTRA, -1
            elif n in self.params:
                kind, i = _PSI, self.params.index(n)
                has_psi = True
            elif n in _RESERVED:
                kind, i = _RESERVED[n], 0
                has_pns = has_pns or kind == _PNS
                has_prd = has_prd or kind == _PRD
            elif n in x_names:
                kind, i = _X, list(x_names).index(n)
            elif n in ("t", "time"):
                if self._time_synthetic:
                    raise ValueError(f"the function uses '{n}' but the data have no time column: "
                                     "specify time=<column> (without it the model is treated as static)")
                kind, i = _X, 0
            elif n in cov_names:
                kind, i = _COV, list(cov_names).index(n)
            elif has_default:
                continue
            else:
                pending.append(n)
                continue
            names.append(n); kinds.append(kind); idx.append(i)
        if pending:
            if not allow_seq_aliases or has_psi:
                n = pending[0]
                raise ValueError(f"{self.name}: the function argument {n!r} is neither a parameter "
                                 f"({self.params}) nor a data column")
            # Renamed positional arguments: the first unresolved one is the full parameter
            # sequence and the next is the predictor sequence (SciPy style, bound by order).
            for n in pending:
                if not has_pns:
                    kind, i = _PNS, 0
                    has_pns = True
                elif not has_prd:
                    kind, i = _PRD, 0
                    has_prd = True
                else:
                    raise ValueError(f"{self.name}: the function argument {n!r} is neither a parameter "
                                     f"({self.params}) nor a data column")
                names.append(n); kinds.append(kind); idx.append(i)
        return _Plan(names, kinds, idx)

    # ------------------------------------------------------------------ callbacks for the core
    def _params_ns(self, psi):
        """Expose the parameter vector as a named view (``params.ka``, unpacking, ``np.asarray``)."""
        return ModelView(self.params, psi, label="params")

    def _pred_spec(self, x_names, cov_names):
        """[(name, "x"|"cov", index)] of the columns of the ``predictors`` argument, in declared order."""
        order = self.predictors or self._pred_order
        if not order:
            raise ValueError("the function uses the 'predictors' argument: declare the columns in order with "
                             "predictors=[...] (in NLModel or in SAEM)")
        spec = []
        for nm in order:
            if nm in x_names:
                spec.append((nm, "x", list(x_names).index(nm)))
            elif nm in cov_names:
                spec.append((nm, "cov", list(cov_names).index(nm)))
            else:
                raise ValueError(f"predictors: column {nm!r} is not among the data predictors/covariates")
        return spec

    @staticmethod
    def _pred_ns(spec, getx, cov):
        """Build the value of the ``predictors`` argument from a column spec.

        A single predictor is returned as a scalar-like :class:`SinglePred`; several predictors
        are returned as a :class:`SeqNS` that also supports iteration and indexing.
        """
        vals = [getx(j) if src == "x" else cov[j] for _, src, j in spec]
        if len(spec) == 1:
            return SinglePred(vals[0], spec[0][0])
        return SeqNS([nm for nm, _, _ in spec], vals, "predictors")

    def _data_ns(self, x_names, cov_names, col, cov):
        """``data``: x columns (and the t/time aliases), and covariates. ``col(j)`` gives column j.
        Columns other than ``time`` must be declared in ``predictors=[...]`` (or be model covariates)."""
        d = {nm: col(j) for j, nm in enumerate(x_names)}
        if not self._time_synthetic:
            for alias in ("t", "time"):
                d.setdefault(alias, col(0))
        for j, nm in enumerate(cov_names):
            d[nm] = cov[j]
        return NS(d, "data", "Columns must be declared in predictors=[...] (or be model covariates)")

    def _xv_spec(self, x_names, cov_names):
        """[(name, 'x'|'cov', index)] for the portable ``x`` view, in unpacking order."""
        order = list(self.predictors or self._pred_order or [])
        if order:
            out = []
            for nm in order:
                if nm in x_names:
                    out.append((nm, "x", list(x_names).index(nm)))
                elif nm in cov_names:
                    out.append((nm, "cov", list(cov_names).index(nm)))
                else:
                    raise ValueError(f"predictors: column {nm!r} is not among the data predictors/covariates")
            return out
        return ([(nm, "x", j) for j, nm in enumerate(x_names)] +
                [(nm, "cov", j) for j, nm in enumerate(cov_names)])

    def _x_view(self, spec, getx, cov, dose=None):
        """Build the portable ``x`` view for ``def model(x, p)``.

        A dose record, when present, is available as ``x.doses`` without
        changing the unpacking order.
        """
        names = [nm for nm, _, _ in spec]
        values = [getx(j) if src == "x" else cov[j] for _, src, j in spec]
        extras = {"doses": dose} if dose is not None else None
        return ModelView(names, values, label="x", columnar=True, extras=extras)

    def _make_subject_fn(self, x_names, cov_names):
        """f(psi, x[n,stride], cov, doses|None) -> array(n)   (per-subject prediction)."""
        plan = self._plan(self.fn, x_names, cov_names, allow_seq_aliases=True)
        fn, vect = self.fn, self.vectorized
        names, kinds, idx = plan.names, plan.kinds, plan.idx
        k_psi, k_x, k_cov, k_dos, k_yt, k_xv = _PSI, _X, _COV, _DOSES, _YTYPE, _XV
        k_pns, k_dns, k_prd = _PNS, _DNS, _PRD
        spec = self._pred_spec(x_names, cov_names) if _PRD in kinds else None
        xv_spec = self._xv_spec(x_names, cov_names) if k_xv in kinds else None
        pred_ns, x_view = self._pred_ns, self._x_view
        params_ns, data_ns = self._params_ns, self._data_ns

        if vect:
            psi_items = tuple((n, i) for n, k, i in zip(names, kinds, idx) if k == k_psi)
            dos_names = tuple(n for n, k in zip(names, kinds) if k == k_dos)
            pns_names = tuple(n for n, k in zip(names, kinds) if k == k_pns)
            # Per subject, x and cov do not change between calls: the C++ bridge returns the SAME numpy object
            # as long as its contents do not change (if they change, it delivers a new one). The x columns,
            # the cov scalars and the observation type are cached; only psi is rebuilt each time.
            cache = {}

            def call(psi, x, cov, doses):
                """Vectorized per-subject prediction adapter used by the core."""
                ent = cache.get(id(x))
                if ent is None or ent[0] is not x or ent[1] is not cov:
                    if len(cache) > 4096:
                        cache.clear()
                    base = {}
                    for n, k, i in zip(names, kinds, idx):
                        if k == k_x: base[n] = x[:, i]
                        elif k == k_cov: base[n] = cov[i]
                        elif k == k_yt: base[n] = x[:, -1].astype(int)
                        elif k == k_dns: base[n] = data_ns(x_names, cov_names, lambda j: x[:, j], cov)
                        elif k == k_prd: base[n] = pred_ns(spec, lambda j: x[:, j], cov)
                        elif k == k_xv: base[n] = x_view(
                            xv_spec, lambda j: x[:, j], cov,
                            Doses(doses) if doses is not None else None,
                        )
                    ent = (x, cov, base)
                    cache[id(x)] = ent
                kw = ent[2].copy()
                for n, i in psi_items:
                    kw[n] = psi[i]
                for n in dos_names:
                    kw[n] = _EMPTY_DOSES if doses is None else Doses(doses)
                for n in pns_names:
                    kw[n] = params_ns(psi)
                r = np.asarray(fn(**kw), dtype=float)
                if r.ndim == 0:
                    r = np.full(x.shape[0], float(r))
                return np.ascontiguousarray(r.reshape(-1))
        else:
            def call(psi, x, cov, doses):
                """Row-by-row per-subject prediction adapter for non-vectorized callbacks."""
                n_obs = x.shape[0]
                out = np.empty(n_obs)
                dos = _EMPTY_DOSES if doses is None else Doses(doses)
                for j in range(n_obs):
                    kw = {}
                    for n, k, i in zip(names, kinds, idx):
                        if k == k_psi: kw[n] = psi[i]
                        elif k == k_x: kw[n] = x[j, i]
                        elif k == k_cov: kw[n] = cov[i]
                        elif k == k_dos: kw[n] = dos
                        elif k == k_yt: kw[n] = int(x[j, -1])
                        elif k == k_pns: kw[n] = params_ns(psi)
                        elif k == k_dns: kw[n] = data_ns(x_names, cov_names, lambda c, _j=j: x[_j, c], cov)
                        elif k == k_prd: kw[n] = pred_ns(spec, lambda c, _j=j: x[_j, c], cov)
                        elif k == k_xv: kw[n] = x_view(xv_spec, lambda c, _j=j: x[_j, c], cov,
                                                       dos if doses is not None else None)
                    out[j] = fn(**kw)
                return out
        return call

    def _row_fn(self, fn, x_names, cov_names, *, with_t=False, tname=None, extra_args=False,
                allow_seq_aliases=False):
        """Per-observation adapter: C++ calls (psi, x_row, cov[, t])."""
        plan = self._plan(fn, x_names, cov_names, allow_seq_aliases=allow_seq_aliases, tvar=tname)
        names, kinds, idx = plan.names, plan.kinds, plan.idx
        t_i = 0
        spec = self._pred_spec(x_names, cov_names) if _PRD in kinds else None
        xv_spec = self._xv_spec(x_names, cov_names) if _XV in kinds else None

        def call(psi, *rest):
            """Per-observation adapter used by the core for family callbacks."""
            if with_t:
                t, x, cov = rest
            else:
                x, cov = rest
                t = None
            kw = {}
            for n, k, i in zip(names, kinds, idx):
                if k == _PSI: kw[n] = psi[i]
                elif k == _X: kw[n] = t if (with_t and i == t_i) else x[i]
                elif k == _COV: kw[n] = cov[i]
                elif k == _YTYPE: kw[n] = int(x[-1])
                elif k == _PNS: kw[n] = self._params_ns(psi)
                elif k == _DNS: kw[n] = self._data_ns(x_names, cov_names, lambda c: t if (with_t and c == t_i) else x[c], cov)
                elif k == _PRD: kw[n] = self._pred_ns(spec, lambda c: t if (with_t and c == t_i) else x[c], cov)
                elif k == _XV: kw[n] = self._x_view(xv_spec, lambda c: x[c], cov)
            return fn(**kw)
        return call

    def _custom_loglik(self, x_names, cov_names):
        """Adapter for a user log-likelihood, called as ``call(psi, x_row, cov, y, extra)``."""
        plan = self._plan(self.loglik_fn, x_names, cov_names, allow_y=True, allow_extra=True)
        fn = self.loglik_fn
        spec = self._pred_spec(x_names, cov_names) if _PRD in plan.kinds else None

        def call(psi, x, cov, y, extra):
            """Per-observation log-likelihood adapter used by the core."""
            kw = {}
            for n, k, i in zip(plan.names, plan.kinds, plan.idx):
                if k == _PSI: kw[n] = psi[i]
                elif k == _X: kw[n] = x[i]
                elif k == _COV: kw[n] = cov[i]
                elif k == _Y: kw[n] = y
                elif k == _EXTRA: kw[n] = extra
                elif k == _YTYPE: kw[n] = int(x[-1])
                elif k == _PNS: kw[n] = self._params_ns(psi)
                elif k == _DNS: kw[n] = self._data_ns(x_names, cov_names, lambda c: x[c], cov)
                elif k == _PRD: kw[n] = self._pred_ns(spec, lambda c: x[c], cov)
            return float(fn(**kw))
        return call

    def _custom_sample(self, x_names, cov_names):
        """Adapter for a user sampler, called as ``call(psi, x_row, cov, extra, seed)``."""
        plan = self._plan(self.sample_fn, x_names, cov_names, allow_extra=True, allow_seed=True)
        fn = self.sample_fn
        spec = self._pred_spec(x_names, cov_names) if _PRD in plan.kinds else None

        def call(psi, x, cov, extra, seed):
            """Per-observation sampler adapter used by the core."""
            kw = {}
            for n, k, i in zip(plan.names, plan.kinds, plan.idx):
                if k == _PSI: kw[n] = psi[i]
                elif k == _X: kw[n] = x[i]
                elif k == _COV: kw[n] = cov[i]
                elif k == _EXTRA: kw[n] = extra if i == 0 else int(seed)
                elif k == _PNS: kw[n] = self._params_ns(psi)
                elif k == _DNS: kw[n] = self._data_ns(x_names, cov_names, lambda c: x[c], cov)
                elif k == _PRD: kw[n] = self._pred_ns(spec, lambda c: x[c], cov)
            return float(fn(**kw))
        return call

    # ------------------------------------------------------------------ statistical structure
    def _omega_blocks(self) -> tuple[list, list]:
        """(omega_block per parameter, covariance_pattern) for the core."""
        p, names = len(self.params), self.params
        om = self.omega
        if isinstance(om, str):
            key = om.lower()
            if key == "diagonal":
                return [], []
            if key == "full":
                grp = [n for n in names if n in self.random]
                return self._blocks_from_groups([grp]), []
            raise ValueError("omega: 'diagonal', 'full', blocks [[...]] or a pattern matrix")
        if isinstance(om, np.ndarray):
            pat = om.astype(bool)
        elif isinstance(om, (list, tuple)) and om and all(isinstance(g, (list, tuple)) and g and
                                                          all(isinstance(n, str) for n in g) for g in om):
            return self._blocks_from_groups([list(g) for g in om]), []
        else:
            pat = np.asarray(om).astype(bool)
        if pat.shape != (p, p):
            raise ValueError(f"omega: pattern {pat.shape} != ({p}, {p})")
        return [], pat.astype(int).tolist()

    def _blocks_from_groups(self, groups):
        """Map covariance groups to a per-parameter block index.

        Parameters that do not appear in any group are placed in their own single-parameter
        block, which is how the core represents independent random effects.
        """
        names = self.params
        ids = {}
        for gi, g in enumerate(groups):
            for n in g:
                if n not in names:
                    raise ValueError(f"omega: unknown parameter {n!r}")
                if n not in self.random:
                    raise ValueError(f"omega: {n!r} has no random effect (random=...)")
                if n in ids:
                    raise ValueError(f"omega: {n!r} appears in more than one block")
                ids[n] = gi
        nxt = len(groups)
        out = []
        for n in names:
            if n in ids:
                out.append(ids[n])
            else:
                out.append(nxt); nxt += 1
        return out

    def _cluster_blocks(self):
        """Block index for the cluster-level random effects (``cluster_omega``)."""
        cm = self.cluster_omega
        if isinstance(cm, str):
            if cm == "diagonal":
                return []
            if cm == "full":
                return self._blocks_from_groups([[n for n in self.params if n in self.random]])
            raise ValueError("cluster_omega: 'diagonal', 'full' or blocks")
        return self._blocks_from_groups([list(g) for g in cm])

    # ------------------------------------------------------------------ compilation to C++
    def _roles(self, x_names, cov_names) -> Roles:
        """Role table used by the expression tracer to resolve callback arguments."""
        R = Roles(_PSI, _X, _COV, _DOSES, _YTYPE, _EXTRA, _Y, _PNS, _DNS, _PRD,
                  self.params, x_names, cov_names,
                  lambda: self._pred_spec(x_names, cov_names),
                  _XV, lambda: self._xv_spec(x_names, cov_names))
        R.time_alias = not self._time_synthetic
        return R

    def _compile_gate(self, what: str) -> Optional[str]:
        """None if compilation can be attempted; otherwise the reason (recorded in compile_info)."""
        if self.compile is False:
            return "compile=False: Python callback"
        if getattr(_core, "EXPR_NOPS", None) != len(_trace_ops()):
            return "operation table out of sync with the core"
        return None

    def _finish_compile(self, why: Optional[str], ok_msg: str) -> bool:
        """Record the compilation outcome.

        Returns ``True`` when compilation succeeded. On failure, ``compile=True`` raises an error
        while the default ``compile`` setting silently keeps the Python callback.
        """
        if why is None:
            self._expr_ok = True
            self._compile_msg = ok_msg
            return True
        self._compile_msg = f"Python callback ({why})"
        if self.compile is True:
            raise ValueError(f"{self.name}: could not compile to C++: {why}")
        return False

    def _compile_expr(self, m, x_names: Sequence[str], cov_names: Sequence[str]) -> bool:
        """Compile the Gaussian prediction ``fn`` (supports ``doses.superpose``, ``ytype``, ``params``/``data``)."""
        self._expr_ok = False
        if self.fn is None or self.family != "gaussian":
            self._compile_msg = "not applicable (no Gaussian fn)"
            return False
        gate = self._compile_gate("fn")
        if gate is not None:
            self._compile_msg = gate
            return False
        why = None
        try:
            plan = self._plan(self.fn, x_names, cov_names, allow_seq_aliases=True)
            prog = trace_program(self.fn, plan.names, plan.kinds, plan.idx, self._roles(x_names, cov_names), 1)
            expr = _core.Expr(*prog.arrays())
            why = self._validate_expr(expr, x_names, cov_names, _DOSES in plan.kinds)
            if why is None:
                m.set_expr(expr)
        except TraceError as e:
            why = str(e)
        except Exception as e:                       # e.g. core failure while loading the graph
            why = f"{type(e).__name__}: {e}"
        return self._finish_compile(why, f"compiled to C++ ({len(prog)} nodes): no GIL and multithreaded"
                                    if why is None else "")

    def _rand_inputs(self, rng, x_names, cov_names, n=7):
        """Random ``(psi, x, cov)`` triple used to validate a compiled program."""
        stride = len(x_names) + 1
        x = rng.uniform(0.5, 5.0, size=(n, stride))
        x[:, 0] = np.sort(rng.uniform(0.25, 24.0, size=n))
        x[:, -1] = 0.0
        return (rng.uniform(0.5, 5.0, size=self.n_params), np.ascontiguousarray(x),
                rng.uniform(0.5, 5.0, size=len(cov_names)))

    @staticmethod
    def _random_doses(rng):
        """(time, 1-based cmt, amt, dur, ii, ss, modeled): boluses and infusions, no SS."""
        d = np.zeros((4, 7))
        d[:, 0] = np.sort(rng.uniform(0.0, 8.0, size=4))
        d[:, 1] = rng.integers(1, 3, size=4)
        d[:, 2] = rng.uniform(50.0, 150.0, size=4)
        d[:, 3] = np.array([0.0, 2.0, 0.0, 1.0])
        return np.ascontiguousarray(d)

    @staticmethod
    def _close(a, b, rtol=1e-9) -> Optional[str]:
        """Compare two numeric outputs; return a failure description or ``None``."""
        a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
        if a.shape != b.shape:
            return "the output shape differs"
        if np.any(np.isfinite(a) != np.isfinite(b)):
            return "different finiteness between Python and C++"
        ok = np.isfinite(a)
        if ok.any() and np.max(np.abs(a[ok] - b[ok])) > rtol * (1.0 + np.max(np.abs(a[ok]))):
            return "numerical validation against the Python function failed"
        return None

    def _validate_expr(self, expr, x_names, cov_names, has_doses: bool) -> Optional[str]:
        """Compare the compiled prediction with the Python function on random inputs."""
        py = self._make_subject_fn(x_names, cov_names)
        rng = np.random.default_rng(20240607)
        for _ in range(4):
            psi, x, cov = self._rand_inputs(rng, x_names, cov_names)
            doses = self._random_doses(rng) if has_doses else None
            with np.errstate(all="ignore"):
                try:
                    a = py(psi, x, cov, doses)
                except Exception as e:
                    return f"the function failed on test data ({type(e).__name__}: {e})"
                b = expr.eval(psi, x, cov, doses)
            bad = self._close(a, b)
            if bad:
                return bad
        return None

    def _validate_rows(self, pairs, x_names, cov_names, *, with_y=False, n_extra=0, with_t=False,
                       psi_only=False) -> Optional[str]:
        """Validate single-row programs: ``pairs`` = [(python_callable, expr, expected_output_is_vector)]."""
        rng = np.random.default_rng(20240608)
        for _ in range(6):
            psi, x, cov = self._rand_inputs(rng, x_names, cov_names, n=1)
            row = np.ascontiguousarray(x[0])
            y = float(rng.uniform(0.0, 3.0)); extra = rng.uniform(0.5, 2.0, size=n_extra)
            t = float(rng.uniform(0.5, 10.0)) if with_t else None
            for call, expr in pairs:
                with np.errstate(all="ignore"):
                    try:
                        if psi_only:
                            a = call(psi)
                        elif with_y:
                            a = call(psi, row, cov, y, extra)
                        elif with_t:
                            a = call(psi, t, row, cov)
                        else:
                            a = call(psi, row, cov)
                    except Exception as e:
                        return f"the function failed on test data ({type(e).__name__}: {e})"
                    b = expr.eval_row(psi, np.zeros(0) if psi_only else row, np.zeros(0) if psi_only else cov,
                                      y, extra, t)
                bad = self._close(np.atleast_1d(np.asarray(a, dtype=float)), b)
                if bad:
                    return bad
        return None

    def _compile_loglik(self, m, x_names, cov_names) -> bool:
        """Compile the custom likelihood ``loglik(y, <params>, <x>, extra=...)``."""
        self._expr_ok = False
        gate = self._compile_gate("loglik")
        if gate is not None:
            self._compile_msg = gate
            return False
        why, prog = None, None
        try:
            plan = self._plan(self.loglik_fn, x_names, cov_names, allow_y=True, allow_extra=True)
            prog = trace_program(self.loglik_fn, plan.names, plan.kinds, plan.idx, self._roles(x_names, cov_names), 1)
            expr = _core.Expr(*prog.arrays())
            why = self._validate_rows([(self._custom_loglik(x_names, cov_names), expr)], x_names, cov_names,
                                      with_y=True, n_extra=self.n_extra)
            if why is None:
                m.set_expr_loglik(expr, self._custom_sample(x_names, cov_names) if self.sample_fn else None, self.n_extra)
        except TraceError as e:
            why = str(e)
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
        return self._finish_compile(why, f"loglik compiled to C++ ({len(prog)} nodes)" if why is None else "")

    def _compile_family(self, m, x_names, cov_names) -> bool:
        """Compile the non-Gaussian families (bernoulli, poisson, negbin, zip, ordinal, categorical, tte)."""
        self._expr_ok = False
        gate = self._compile_gate("fn")
        if gate is not None:
            self._compile_msg = gate
            return False
        fam, why, nn = self.family, None, 0
        try:
            R = self._roles(x_names, cov_names)
            tname = x_names[0]
            with_t = fam == "tte"
            plan = self._plan(self.fn, x_names, cov_names, allow_seq_aliases=True, tvar=tname)
            n_out = {"zip": 2, "categorical": None}.get(fam, 1)
            pf = trace_program(self.fn, plan.names, plan.kinds, plan.idx, R, n_out)
            ef = _core.Expr(*pf.arrays()); nn = len(pf)
            pairs, eg, sc = [], None, -1
            f_py = self._row_fn(self.fn, x_names, cov_names, allow_seq_aliases=True, with_t=with_t, tname=tname)
            pairs.append((f_py, ef))
            if fam == "tte":
                gplan = self._plan(self.cumhazard, x_names, cov_names, tvar=tname)
                pg = trace_program(self.cumhazard, gplan.names, gplan.kinds, gplan.idx, R, 1)
                eg = _core.Expr(*pg.arrays()); nn += len(pg)
                pairs.append((self._row_fn(self.cumhazard, x_names, cov_names, with_t=True, tname=tname), eg))
                if self.tte_start is not None:
                    if self.tte_start not in x_names:
                        raise ValueError(f"tte_start={self.tte_start!r} is not a predictor")
                    sc = list(x_names).index(self.tte_start)
            why = self._validate_rows(pairs, x_names, cov_names, with_t=with_t)
            if why is None and fam == "ordinal":
                cplan = self._plan(self.cuts, x_names, cov_names)
                if any(k not in (_PSI, _PNS) for k in cplan.kinds):
                    raise TraceError("cuts may only depend on the parameters")
                pc = trace_program(self.cuts, cplan.names, cplan.kinds, cplan.idx, R, None)
                eg = _core.Expr(*pc.arrays()); nn += len(pc)

                def cuts_py(psi, _c=self.cuts, _cp=cplan):
                    """Python cut-point function used to validate the compiled ordinal program."""
                    kw = {n: (psi[i] if k == _PSI else self._params_ns(psi)) for n, k, i in zip(_cp.names, _cp.kinds, _cp.idx)}
                    return np.asarray(_c(**kw), float).reshape(-1)
                why = self._validate_rows([(cuts_py, eg)], x_names, cov_names, psi_only=True)
            if why is None:
                m.set_expr_family(fam, ef, eg, sc)
        except TraceError as e:
            why = str(e)
        except ValueError:
            raise
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
        return self._finish_compile(why, f"family {fam!r} compiled to C++ ({nn} nodes)" if why is None else "")

    # ------------------------------------------------------------------ core
    def _to_core(self, x_names: Sequence[str], cov_names: Sequence[str], cat_covs: Sequence[str]):
        """Build the ``_core.Model``. ``x_names`` = [time, predictors...] ; ``cov_names`` = numeric
        per-subject covariates (core order)."""
        m = _core.Model()
        self._expr_ok = False
        p = self.n_params
        m.n_phi = p
        m.param_names = list(self.params)
        m.transform = [getattr(_core.Transform, t) for t in self.transform]
        if len(self.random) != p:
            m.random_effect = [1 if n in self.random else 0 for n in self.params]
        blocks, pattern = self._omega_blocks()
        if blocks:
            m.omega_block = blocks
        if len(pattern):
            m.covariance_pattern = pattern
        cb = self._cluster_blocks()
        if cb:
            m.psi_block = cb
        if len(self.errors) == 1:
            m.error = error_enum(self.errors[0])
        else:
            m.error = error_enum(self.errors[0])
            m.errors = [error_enum(e) for e in self.errors]
        m.covariate_model = [(self.params.index(pn), c) for pn, c in self.covariates]

        if self._builtin is not None:
            name, xcol, dose = self._builtin
            xi = 0
            if xcol is not None and xcol != x_names[0]:
                if xcol not in x_names:
                    raise ValueError(f"x={xcol!r} is not a data column")
                xi = list(x_names).index(xcol)
            di = -1
            if dose is not None:
                if dose not in cov_names:
                    raise ValueError(f"dose={dose!r} must be a per-subject (constant) covariate of the data")
                di = list(cov_names).index(dose)
            m.set_builtin(name, xi, di)
        elif self.family == "gaussian":
            if not self._compile_expr(m, x_names, cov_names):
                m.set_predict_subject(self._make_subject_fn(x_names, cov_names))
        elif self.family == "custom":
            if not self._compile_loglik(m, x_names, cov_names):
                m.set_loglik(self._custom_loglik(x_names, cov_names),
                             self._custom_sample(x_names, cov_names) if self.sample_fn else None, self.n_extra)
            m.set_predict_subject(lambda psi, x, cov, doses: np.zeros(x.shape[0]))
        else:
            tname = x_names[0]
            if self._compile_family(m, x_names, cov_names):
                pass
            elif self.family == "tte":
                f = self._row_fn(self.fn, x_names, cov_names, allow_seq_aliases=True, with_t=True, tname=tname)
                g = self._row_fn(self.cumhazard, x_names, cov_names, with_t=True, tname=tname)
                sc = -1
                if self.tte_start is not None:
                    if self.tte_start not in x_names:
                        raise ValueError(f"tte_start={self.tte_start!r} is not a predictor")
                    sc = list(x_names).index(self.tte_start)
                m.set_family("tte", lambda psi, t, x, cov: f(psi, t, x, cov),
                             lambda psi, t, x, cov: g(psi, t, x, cov), sc)
            elif self.family == "ordinal":
                f = self._row_fn(self.fn, x_names, cov_names, allow_seq_aliases=True)
                cuts_plan = self._plan(self.cuts, x_names, cov_names)
                cn, ck, ci = cuts_plan.names, cuts_plan.kinds, cuts_plan.idx
                cuts = self.cuts

                def g(psi):
                    """Cut-point callback exposed to the core (``g(psi) -> cut points``)."""
                    kw = {n: (psi[i] if k == _PSI else self._params_ns(psi)) for n, k, i in zip(cn, ck, ci) if k in (_PSI, _PNS)}
                    return np.ascontiguousarray(np.asarray(cuts(**kw), dtype=float).reshape(-1))
                m.set_family("ordinal", f, g, -1)
            elif self.family == "zip":
                f = self._row_fn(self.fn, x_names, cov_names, allow_seq_aliases=True)
                m.set_family("zip", lambda psi, x, cov: tuple(float(v) for v in f(psi, x, cov)), None, -1)
            elif self.family == "categorical":
                f = self._row_fn(self.fn, x_names, cov_names, allow_seq_aliases=True)
                m.set_family("categorical",
                             lambda psi, x, cov: np.ascontiguousarray(np.asarray(f(psi, x, cov), float).reshape(-1)), None, -1)
            else:
                f = self._row_fn(self.fn, x_names, cov_names, allow_seq_aliases=True)
                m.set_family(self.family, lambda psi, x, cov: float(f(psi, x, cov)), None, -1)
            # the engine predictor is not used with a custom likelihood: null prediction
            m.set_predict_subject(lambda psi, x, cov, doses: np.zeros(x.shape[0]))
        return m

    def __repr__(self):
        return (f"NLModel({self.name!r}, params={self.params}, transform={self.transform}, "
                f"random={self.random}, error={self.errors}, family={self.family!r})")


class _Models:
    """Shortcuts: ``sf.models.pk1_oral(dose="dose")``."""

    def __getattr__(self, name):
        if name not in BUILTINS:
            raise AttributeError(name)

        def make(**kw):
            """Build a compiled model of this kind with the given options."""
            return NLModel.builtin(name, **kw)
        make.__name__ = name
        make.__doc__ = f"{BUILTINS[name].description}\nparams = {list(BUILTINS[name].params)}"
        return make

    def __dir__(self):
        return sorted(BUILTINS)


models = _Models()
