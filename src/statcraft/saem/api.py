"""High-level API: ``SAEM(...)`` / ``saem(...)`` build the model and ``.fit()`` fits it.

The style is the usual one in Python (and in the rest of statcraft): first create the object with the
data and the specification, then run the fit explicitly with ``.fit()``::

    import statcraft as sf
    from statcraft.saem import models

    model = sf.SAEM(models.pk1_oral(dose="dose"), df, id="id", time="time", response="conc",
                    start={"ka": 1.5, "CL": 3.0, "V": 30.0})
    fit = model.fit()                      # -> SaemFit
    print(fit.summary())

    # the fit can also be configured at fitting time
    fit = model.fit(n_iter=500, seed=1, se_method="louis")

Model options (``params``, ``transform``, ``epsilon``...) are passed to the constructor; control
options (``n_iter``, ``seed``...) can be passed to the constructor or to ``.fit()``. Result
attributes (``summary()``, ``fixef``, ``ranef``, ...) live on the :class:`SaemFit` returned by
``.fit()``.
"""
from __future__ import annotations

import inspect
from typing import Mapping, Optional, Sequence

from .control import SaemControl, _CONTROL_FIELDS, _normalize_control_kwargs
from .fit import SaemFit, _run_saem
from .model import NLModel

__all__ = ["SAEM", "saem"]

#: Options that belong to ``NLModel`` (not to the control).
_MODEL_KW = set(inspect.signature(NLModel.__init__).parameters) - {"self", "fn", "_builtin"}


class SAEM:
    """Nonlinear mixed-effects model fitted by SAEM, ready to fit with :meth:`fit`.

    Parameters
    ----------
    model : NLModel | callable
        ``NLModel`` or the structural function directly. In the second case the model arguments are
        passed as ``kwargs``: ``params``, ``transform``, ``random``, ``omega``, ``error``,
        ``covariates``, ``family``, ... ``family`` accepts only a GLM family object
        or class (e.g. ``Binomial("logit")``, ``Poisson("log")``); these are
        re-exported from :mod:`statcraft.saem`. Family strings are rejected. If
        ``params`` is missing and ``start`` is a dict, its keys are used.
    data : DataFrame | dict
        Data in long format: one row per observation.
    id, time, response : str
        Names of the subject, time and response-variable columns (defaults ``"id"``, ``"time"`` and
        ``"y"``).
        ``id`` is OPTIONAL: if omitted, the ``"id"`` column is used when it exists; if there is none, each
        row is treated as a distinct subject (id 1, 2, 3, ...) and a warning is issued (SaemWarning): with
        one observation per subject the random effects are not identifiable and the engine does not yet
        support models without random effects.
        ``time`` is OPTIONAL: if omitted, the ``"time"`` column is used when it exists; if the data have no
        time (e.g. dose-response) the model is treated as static and each subject's observation index is
        used as the ``obs`` column (no "dummy" column is needed). That is not possible with dose events
        (``events``/``evid``), built-in models or survival families such as ``Weibull()``, which need real time, nor if the
        function uses a ``time`` argument. When present, time is one more predictor: to receive it in
        ``predictors`` just list it (``predictors=["time", "dosis"]``).
    ytype, cens, limit, group : str, optional
        Columns for response type, censoring (-1 left, 0 observed, +1 right), censoring limit and
        cluster (two-level hierarchical model).
    events, evid, event_cols : optional
        NONMEM dose record: an ``events`` DataFrame with ``id,time,amt[,cmt,rate,addl,ii,ss]`` or
        ``evid`` = an EVID column inside ``data`` (1 = dose, 0 = observation).
    predictors : sequence[str], optional
        Columns that must be treated as predictors (even if constant within subject).
    reference : mapping[str, str], optional
        Reference level of the categorical covariates, ``{"sex": "F"}``.
    na_action : {"omit", "fail"}
        What to do with rows that contain missing values.
    start, start_sd, start_error, start_cov : optional
        Initial population values (natural scale), random-effect SDs (phi scale), error model and
        covariate coefficients.
    fix, fix_sd, fix_error : sequence[str], optional
        Parameters kept fixed at their initial value.
    control : SaemControl, optional
        Algorithm configuration. Its individual options are also accepted in ``kwargs``
        (``n_iter=500, seed=1, ...``).
    **kwargs
        ``NLModel`` or ``SaemControl`` options (separated automatically).

    Notes
    -----
    ``.fit()`` can be called several times; with no arguments it returns the already computed result
    (``fit()`` is idempotent). With new control options it fits again.
    """

    def __init__(self, model, data, *, id: Optional[str] = None, time: Optional[str] = None,
                 response: Optional[str] = None,
                 ytype: Optional[str] = None, cens: Optional[str] = None, limit: Optional[str] = None,
                 group: Optional[str] = None, events=None, evid: Optional[str] = None,
                 event_cols: Optional[Mapping[str, str]] = None,
                 predictors: Optional[Sequence[str]] = None,
                 reference: Optional[Mapping[str, str]] = None, na_action: str = "omit",
                 start=None, start_sd=None, start_error=None,
                 start_cov: Optional[Mapping[str, float]] = None,
                 fix: Optional[Sequence[str]] = None, fix_sd: Optional[Sequence[str]] = None,
                 fix_error: Optional[Sequence[str]] = None,
                 control: Optional[SaemControl] = None, **kwargs):
        # ---- separate model and control options (same logic as the old saem()) ----
        kwargs = _normalize_control_kwargs(kwargs)
        model_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _MODEL_KW}
        ctrl_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _CONTROL_FIELDS}
        if kwargs:
            unknown = sorted(kwargs)
            raise TypeError(f"SAEM() unrecognized arguments: {unknown}")

        if not isinstance(model, NLModel):
            if not callable(model):
                raise TypeError("model must be an NLModel or a function")
            if "params" not in model_kw:
                if isinstance(start, Mapping):
                    model_kw["params"] = list(start)
                else:
                    raise TypeError("missing params=[...] (or pass start={name: value, ...})")
            model = NLModel(model, **model_kw)
        elif model_kw:
            raise TypeError(f"an already-built NLModel does not accept model options: {sorted(model_kw)}")

        self.model = model
        self.data = data
        if control is not None and not isinstance(control, SaemControl):
            raise TypeError("control must be a SaemControl (normally from saem_control(...))")
        self.control = control or SaemControl()
        if ctrl_kw:
            self.control = self.control.copy(**ctrl_kw)

        # data-preparation and initial-value options, inspectable as attributes
        self._prep = {
            "id": id, "time": time, "response": response, "ytype": ytype, "cens": cens, "limit": limit,
            "group": group, "events": events, "evid": evid, "event_cols": event_cols,
            "predictors": predictors, "reference": reference, "na_action": na_action,
        }
        self._init = {
            "start": start, "start_sd": start_sd, "start_error": start_error,
            "start_cov": start_cov, "fix": fix, "fix_sd": fix_sd, "fix_error": fix_error,
        }
        self.__dict__.update(self._prep)
        self.__dict__.update(self._init)

        self._result: Optional[SaemFit] = None

    # ------------------------------------------------------------------ fitting
    def fit(self, *, control: Optional[SaemControl] = None, **control_kwargs) -> SaemFit:
        """Fit the model by SAEM and return a :class:`SaemFit`.

        Parameters
        ----------
        control : SaemControl, optional
            Alternative control for this fit (cannot be combined with ``control_kwargs``).
        **control_kwargs
            Individual ``SaemControl`` options for this fit, e.g. ``n_iter=500``, ``seed=1``
            or ``se_method="louis"``. If none are passed and the model was already fitted, the
            previous result is returned.
        """
        if control is not None and control_kwargs:
            raise TypeError("fit(): use control=... or the individual options, not both")
        if control is not None and not isinstance(control, SaemControl):
            raise TypeError("fit(): control must be a SaemControl (normally from saem_control(...))")
        if control is not None:
            ctrl = control
        elif control_kwargs:
            control_kwargs = _normalize_control_kwargs(control_kwargs)
            unknown = set(control_kwargs) - _CONTROL_FIELDS
            if unknown:
                raise TypeError(f"fit(): unknown control options {sorted(unknown)}. "
                                f"Valid: {sorted(_CONTROL_FIELDS)}")
            ctrl = self.control.copy(**control_kwargs)
        elif self._result is not None:
            return self._result
        else:
            ctrl = self.control

        result = _run_saem(self.model, self.data, **self._prep, **self._init, control=ctrl)
        self.control = ctrl
        self._result = result
        return result

    def refit(self, **control_kwargs) -> SaemFit:
        """Fit again even if a result already exists (equivalent to ``fit()`` without caching)."""
        self._result = None
        return self.fit(**control_kwargs)

    # ------------------------------------------------------------------ state
    @property
    def is_fitted(self) -> bool:
        """``True`` if the model has already been fitted at least once."""
        return self._result is not None

    @property
    def result(self) -> Optional[SaemFit]:
        """Result of the last fit (``None`` if ``fit()`` has not been called yet)."""
        return self._result

    @property
    def params(self) -> list[str]:
        """Names of the structural-model parameters."""
        return list(self.model.params)

    def __getattr__(self, name):
        """Reject result attributes until the model has been fitted."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"SAEM is not fitted: {name!r} is only available on the result. "
            f"Call .fit() first, e.g. `fit = model.fit()` and then `fit.{name}`."
        )

    def __repr__(self):
        result = self.__dict__.get("_result")
        if result is not None:
            return repr(result)
        return f"SAEM({self.model.name!r}, params={self.model.params!r}, fitted=False)"


#: Lowercase alias for the ``saem(...).fit()`` style (like ``lme(...)`` in ``nlme``).
saem = SAEM
