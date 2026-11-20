# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Utility functions and helpers for statcraft.

Dataset helpers are canonical in :mod:`statcraft.datasets`; the loader names
here are compatibility aliases.
"""

from .data import DatasetLoader, add_constant, data_loader
from .exceptions import ConvergenceError, DataValidationError, FitError, FormulaError, NLSError, NotFittedError
from .FormulaParser import FormulaNamespaceError, FormulaParser
from .Initializer import Initializer
from .modelview import ModelView

__all__ = [
    "DatasetLoader",
    "data_loader",
    "add_constant",
    "FormulaParser",
    "Initializer",
    "ModelView",
    "ConvergenceError",
    "DataValidationError",
    "FitError",
    "FormulaError",
    "FormulaNamespaceError",
    "NLSError",
    "NotFittedError",
]
