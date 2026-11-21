"""SAEM for nonlinear mixed-effects models (C++20 + Eigen, nanobind).

The model is built first and fitted explicitly with ``.fit()``, like the rest of
statcraft::

    import statcraft as sf
    from statcraft.saem import models

    model = sf.SAEM(
        models.pk1_oral(dose="dose"), df, id="id", time="time", response="conc",
        start={"ka": 1, "CL": 3, "V": 30}, n_iter=500,
    )
    fit = model.fit()
    print(fit.summary())

From the submodule, ``saem`` is an alias of `SAEM`::

    from statcraft.saem import saem
    fit = saem(model, df, id="id", time="time", response="conc").fit()

Result attributes such as ``summary()``, ``fixef`` and ``ranef`` live on the
`SaemFit` returned by ``.fit()``, as in the rest of statcraft.
"""
from ..glm.family import *  # noqa: F401,F403 -- same family objects as GLM/GLMER
from ..glm.family import __all__ as _glm_family_all
from .api import SAEM, saem
from .control import SaemControl, saem_control
from .data import DataInfo
from .fit import SaemFit, SaemSummary, SaemWarning
from .model import BUILTINS, Doses, NLModel, builtin_models, models
from . import datasets

__version__ = "0.1.0"
__all__ = ["SAEM", "saem", "saem_control", "SaemControl", "NLModel", "models", "builtin_models",
           "Doses", "SaemFit", "SaemSummary", "SaemWarning", "DataInfo", "datasets", "__version__",
           *_glm_family_all]
