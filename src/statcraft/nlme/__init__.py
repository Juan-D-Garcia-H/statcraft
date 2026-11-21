"""statcraft.nlme -- linear and nonlinear mixed-effects models (C++/Eigen) with R/nlme-style formulas.

The API is chainable: ``lme``, ``gls``, ``nlme``, ``gnls``, ``lmer`` and
``glmer`` (and their uppercase aliases) construct the model; ``.fit()`` runs
the engine and returns the result, exposing ``fixef``, ``se``, ``summary()``,
...

    from statcraft.nlme import LME
    res = LME("distance ~ age", data, random="~ 1 | Subject").fit()
    print(res.summary())

``nlme`` and ``gnls`` also accept a plain Python function as the model, e.g.
``def f(x, p)`` or ``def f(x, params)`` for one parameter. ``predictors=`` is
inferred when the first argument names a data column. ``params=`` is optional:
when omitted, the keys of ``start`` name the parameters in order with formula
``"1"``. Use ``params=`` for covariate formulas.

For a single curve with parameter bounds, robust losses, multistart or
HC0-HC3 covariance, use :mod:`statcraft.nls` instead.
"""
from .. import _nlmefit as core
from ..glm.family import *  # noqa: F401,F403 -- same family objects as GLM/GLMER
from .crossed import LMER, LmerFit, lmer
from .glmm import GLMER, GlmerFit, glmer
from .models import LmeFit, ListFit, NlmeFit, gls, gnls, lm_list, lme, nlme, nls_list, corr_fixef, fix_df
from .structs import (CorStruct, PdStruct, VarStruct, corAR1, corARMA, corCAR1, corCompSymm, corCustom, corExp, corGaus, corLin,
                      corRatio, corSpher, corSymm, pdBlocked, pdChol, pdCompSymm, pdCustom, pdDiag, pdIdent, pdLogChol, pdNatural,
                      pdSymm, varComb, varConstPower, varConstProp, varCustom, varExp, varFixed, varIdent, varPower)

#: Uppercase aliases used by the top-level API (R ``nlme::*``).
LME = lme
NLME = nlme
GLS = gls
GNLS = gnls
LMER = lmer
GLMER = glmer

__version__ = "0.5.0"
__all__ = [n for n in dir() if not n.startswith("_")]
