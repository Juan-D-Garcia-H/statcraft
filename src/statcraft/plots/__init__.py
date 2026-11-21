# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Plotting for fitted statcraft models.

Every public plotting entry point uses the word ``plot``.  The generic entry
point is :func:`plot`; convenience functions and fitted-result methods are
generated from the same registry::

    fit.plot()                              # adaptive 2x2 diagnostic panel
    fit.plot_diagnostics()
    fit.plot_qq()
    fit.qqplot()
    statcraft.plots.plot_diagnostics(fit)
    statcraft.plots.qqplot(fit)
    statcraft.plots.plot(fit, kind="forest", save="forest.png")

The plot implementations live in private submodules and are deliberately not
re-exported here.  Matplotlib is optional: importing this package does not
import it; the first call to a plotting function does, and a clear error
message points to the ``viz`` extra when it is missing.
"""

from __future__ import annotations

from .adapters import (
    PlotData,
    coefficient_table,
    default_exponentiate,
    has_coefficients,
    has_random_effects,
    is_fitted,
    model_name,
    random_effects,
    resolve,
    response_name,
)
from .classification import ClassificationEvaluator
from .foundation import PlotResult
from .methods import PLOT_FUNCTIONS, PUBLIC_KINDS, PlottingMixin
from .registry import (
    CANONICAL_PLOTS,
    KIND_ALIASES,
    available_kinds,
    empirical_variogram,
    npde_values,
    plot,
    rootogram_data,
    suggest_plots,
)

# Publish the generated plotting facade in this module's namespace.
globals().update(PLOT_FUNCTIONS)

__all__ = list(dict.fromkeys([
    # entry points
    "plot",
    "PlotResult",
    "PlottingMixin",
    "PUBLIC_KINDS",
    "PLOT_FUNCTIONS",
    "available_kinds",
    "suggest_plots",
    "KIND_ALIASES",
    "CANONICAL_PLOTS",
    # data helpers
    "rootogram_data",
    "npde_values",
    "empirical_variogram",
    # adapters
    "PlotData",
    "resolve",
    "is_fitted",
    "model_name",
    "response_name",
    "coefficient_table",
    "random_effects",
    "has_coefficients",
    "has_random_effects",
    "default_exponentiate",
    # evaluator with plot_* methods
    "ClassificationEvaluator",
    # every generated facade function
    *PLOT_FUNCTIONS,
]))
