# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""statcraft -- engines for statistical model fitting (GLM, NLME, MCGLM, SAEM).

A plain ``import statcraft as sf`` exposes the full public API: the engine
namespaces plus every public name exported by them (families, formula
shortcuts, variance/correlation structures, plotting helpers, ...).

The engine submodules keep their names: ``sf.glm`` is the *module*, not the
estimator; use ``import statcraft.glm as scg`` and ``scg.glm(...)`` for formula
entry points, or ``sf.GLM(...)`` for the matrix interface. Likewise
``sf.nls`` is the subpackage, while ``sf.NLS`` is the estimator class.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
import time
import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

try:
    __version__ = version("statcraft")
except PackageNotFoundError:
    # Source-tree fallback: pyproject.toml remains the single source of truth.
    try:
        import tomllib

        _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        __version__ = tomllib.loads(_pyproject.read_text(encoding="utf-8"))["project"]["version"]
    except Exception:
        __version__ = "0.0.0"

#: Engine namespaces. Names equal to a submodule stay bound to the module.
_SUBMODULES = (
    "glm", "nlme", "mcglm", "nls", "plots", "datasets", "summary", "saem", "utils",
)

# Import every engine so a bare ``import statcraft`` really does expose everything.
_MODULES: Dict[str, ModuleType] = {}
for _submodule in _SUBMODULES:
    _module = import_module(f"{__name__}.{_submodule}")
    globals()[_submodule] = _module
    _MODULES[_submodule] = _module

#: Historical top-level aliases, bound first so their identity is stable.
_TOPLEVEL_ENTRY = {
    "GLM": ("glm", "GLM"),
    "LME": ("nlme", "LME"),
    "NLME": ("nlme", "NLME"),
    "GLS": ("nlme", "GLS"),
    "GNLS": ("nlme", "GNLS"),
    "LMER": ("nlme", "LMER"),
    "GLMER": ("nlme", "GLMER"),
    "MCGLM": ("mcglm", "MCGLM"),
    # No "nls" entry: ``statcraft.nls`` is the subpackage above, so the
    # top-level name must stay bound to the module. Use
    # ``from statcraft.nls import nls`` or ``sf.NLS`` for the estimator.
    "NLS": ("nls", "NLS"),
    "lmer": ("nlme", "lmer"),
    "glmer": ("nlme", "glmer"),
    "SAEM": ("saem", "SAEM"),
    "saem_control": ("saem", "saem_control"),
    "list_datasets": ("datasets", "list_datasets"),
    "dataset_info": ("datasets", "dataset_info"),
    "data_loader": ("datasets", "data_loader"),
    "DatasetLoader": ("datasets", "DatasetLoader"),
}

__all__ = list(_SUBMODULES)
_seen = set(__all__)

for _name, (_submodule, _attribute) in _TOPLEVEL_ENTRY.items():
    globals()[_name] = getattr(_MODULES[_submodule], _attribute)
    if _name not in _seen:
        __all__.append(_name)
        _seen.add(_name)

# Publish every other public name from the engines. Internal module aliases
# (e.g. nlme.core) are skipped: only genuine public objects reach the top level.
for _submodule, _module in _MODULES.items():
    for _name in getattr(_module, "__all__", ()):
        if _name in _seen or _name in _SUBMODULES:
            continue
        _value = getattr(_module, _name)
        if isinstance(_value, ModuleType):
            continue
        globals()[_name] = _value
        __all__.append(_name)
        _seen.add(_name)

__all__.sort()


def __getattr__(name: str):
    """Fallback for names added to a submodule after this package was imported."""
    for _module in _MODULES.values():
        if name in getattr(_module, "__all__", ()) and name not in _SUBMODULES:
            value = getattr(_module, name)
            if isinstance(value, ModuleType):
                continue
            globals()[name] = value
            if name not in __all__:
                __all__.append(name)
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
