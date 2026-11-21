# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Nonlinear least squares (:class:`nls`, :class:`NLS`) with formula parsing and optional JAX backend.

Scope: this subpackage is the standalone front end for fitting **a single
nonlinear curve** with :func:`scipy.optimize.least_squares`.  It owns the
features that only make sense without grouping: parameter bounds, robust
losses, ``fit_multistart``, HC0-HC3 covariance, cross-validation, prediction
intervals, the JSON/pickle persistence layer and the optional JAX backend.

The module is feature-frozen: anything that needs groups, random effects,
residual correlation or variance structures belongs to :mod:`statcraft.nlme`
(``nlme``/``gnls``/``nls_list``), not here.

``statcraft.nls`` is the module, so import the estimators explicitly::

    from statcraft.nls import nls, NLS
"""
from ._nls import nls
from .base import NLS

__all__ = ["nls", "NLS"]
