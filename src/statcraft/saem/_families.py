"""GLM-style family objects as the only public ``family=`` API for SAEM.

SAEM keeps canonical likelihood strings internally, but callers specify the
family with the same objects/classes used by :mod:`statcraft.glm` and
:mod:`statcraft.nlme`.  Strings are deliberately rejected so all engines share
one spelling.
"""
from __future__ import annotations

from ..glm.family import (
    Binomial,
    CustomFamily,
    Family,
    Gaussian,
    GeneralizedGamma,
    LogLogistic,
    LogNormal,
    Multinomial,
    NegativeBinomial,
    Ordinal,
    Poisson,
    Weibull,
    ZeroInflatedPoisson,
)

#: Likelihood strings understood by the C++ engine (internal, single source of truth).
SAEM_FAMILIES = (
    "gaussian",
    "bernoulli",
    "bernoulli_logit",
    "poisson",
    "negbin",
    "zip",
    "ordinal",
    "categorical",
    "tte",
    "custom",
)

#: Default ``NLModel`` family: passing the class instantiates it with defaults.
DEFAULT_FAMILY = Gaussian

_OBJECT_FAMILIES = {
    "Gaussian": "gaussian",
    "Binomial": "bernoulli",
    "Poisson": "poisson",
    "NegativeBinomial": "negbin",
    "ZeroInflatedPoisson": "zip",
    "Ordinal": "ordinal",
    "Multinomial": "categorical",
    # Parametric survival families use the hazard likelihood.
    "Weibull": "tte",
    "LogNormal": "tte",
    "LogLogistic": "tte",
    "GeneralizedGamma": "tte",
    # Class token: the log-likelihood itself is supplied via loglik=.
    "CustomFamily": "custom",
}

_LINK_ALIASES = {"logit_log": "logit"}
_LOGIT_LINKS = frozenset({"logit"})

_FAMILY_HINT = (
    "family must be a GLM family object or class, such as Binomial('logit'), "
    "Poisson('log'), NegativeBinomial(), ZeroInflatedPoisson(), Multinomial(), "
    "Ordinal(), Weibull() (with cumhazard=) or CustomFamily (with loglik=); "
    "family strings are no longer accepted"
)


def normalize_family(family, *, n_extra=None, cuts=None, cumhazard=None, loglik=None):
    """Return ``(saem_family_string, updates)`` for a GLM family object/class.

    ``updates`` contains model options that the family object fixes, currently
    only ``n_extra=1`` for the negative binomial.  Final validation of required
    callbacks stays in :meth:`statcraft.saem.model.NLModel._validate`.
    """
    updates = {}

    if isinstance(family, str):
        raise ValueError(f"family strings are no longer accepted: {family!r}. {_FAMILY_HINT}")

    if isinstance(family, type):
        cls_name = family.__name__
        if cls_name not in _OBJECT_FAMILIES:
            raise ValueError(f"{cls_name} is not supported by SAEM. {_FAMILY_HINT}")
        if cls_name == "CustomFamily":
            # CustomFamily needs callbacks in GLM; here the class is a token and
            # the SAEM log-likelihood comes from loglik=.
            if loglik is None:
                raise ValueError("CustomFamily requires loglik=<function>")
            return "custom", updates
        else:
            try:
                family = family()
            except TypeError as exc:  # e.g. Tweedie(var_power) without its argument
                raise ValueError(
                    f"{cls_name} needs constructor arguments; pass an instance instead"
                ) from exc

    if not isinstance(family, Family):
        raise ValueError(_FAMILY_HINT)

    cls_name = type(family).__name__
    if cls_name not in _OBJECT_FAMILIES:
        raise ValueError(f"{cls_name} is not supported by SAEM. {_FAMILY_HINT}")

    name = _OBJECT_FAMILIES[cls_name]
    link = str(getattr(family, "link", "") or "").strip().lower().replace(" ", "")
    link = _LINK_ALIASES.get(link, link)

    if cls_name == "Gaussian" and link and link != "identity":
        raise ValueError(
            "Gaussian: SAEM expects the prediction on the mean scale (no link is "
            "applied); use Gaussian('identity') and apply the inverse link in fn "
            "if you need another scale"
        )
    if cls_name == "Ordinal" and link and link not in _LOGIT_LINKS:
        raise ValueError(
            f"Ordinal: SAEM implements the logit proportional-odds likelihood; "
            f"link={link!r} is not available (use Ordinal('logit'))"
        )
    if cls_name == "Binomial":
        # SAEM applies the logistic link internally only for "bernoulli_logit".
        # Every other binomial link maps to the probability-scale likelihood.
        name = "bernoulli_logit" if link in _LOGIT_LINKS else "bernoulli"
    if cls_name == "NegativeBinomial":
        if n_extra not in (None, 0, 1):
            raise ValueError(
                "NegativeBinomial: SAEM uses exactly one extra parameter "
                "(log r); leave n_extra unset or equal to 1"
            )
        updates["n_extra"] = 1
    if name == "tte" and cumhazard is None:
        raise ValueError(f"{cls_name}: SAEM time-to-event families require cumhazard=<function>")
    if name == "custom" and loglik is None:
        raise ValueError("CustomFamily: SAEM requires loglik=<function>")

    return name, updates
