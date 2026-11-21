# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Uniform plotting facade and the mixin used by fitted-result classes.

The plot implementations live in the submodules of :mod:`statcraft.plots` and
are intentionally not re-exported as the public API.  Every public plotting
entry point is either :func:`plot` or a generated facade function/method that
delegates to it, so ``kind`` aliases stay in one place.  Facade names use the
systematic ``plot_<kind>`` spelling except when ``kind`` already ends in
``plot`` (``fit.forest_plot()``) or is a joined density name (``fit.qqplot()``,
``fit.ppplot()``).
"""

from __future__ import annotations

import re
from typing import Any

from .registry import KIND_ALIASES, plot


def _make_plot_function(kind: str | None, name: str):
    def plot_kind(result: Any, ax=None, **kwargs):
        return plot(result, kind=kind, ax=ax, **kwargs)

    plot_kind.__name__ = name
    plot_kind.__qualname__ = name
    plot_kind.__module__ = __name__
    plot_kind.__doc__ = (
        f"Render the {kind!r} plot for a fitted statcraft result."
        if kind is not None
        else "Generic plotting entry point; pass ``kind=`` to select the panel."
    )
    return plot_kind


def _make_plot_method(kind: str | None, name: str):
    def plot_method(self, ax=None, **kwargs):
        selected = kwargs.pop("kind", kind)
        return plot(self, kind=selected, ax=ax, **kwargs)

    plot_method.__name__ = name
    plot_method.__qualname__ = f"PlottingMixin.{name}"
    plot_method.__module__ = __name__
    plot_method.__doc__ = (
        f"Render the {kind!r} plot for this fitted result."
        if kind is not None
        else "Generic plotting method; pass ``kind=`` to select the panel."
    )
    return plot_method


class PlottingMixin:
    """Shared plotting API injected into fitted result classes."""

    def plot(self, kind=None, ax=None, **kwargs):
        """Render a plot by ``kind`` (defaults to the adaptive diagnostics panel)."""
        return plot(self, kind=kind, ax=ax, **kwargs)


#: Public facade name -> ``kind`` understood by :func:`plot`.
PUBLIC_KINDS: dict[str, str | None] = {}
#: Public facade callables generated from the registry.
PLOT_FUNCTIONS: dict[str, Any] = {}


def _public_plot_name(kind: str) -> str:
    """Avoid ``plot_qq_plot``-style duplication in the generated facade.

    Kinds that already end in ``plot`` keep that suffix; ``qqplot`` and
    ``ppplot`` use the conventional joined spelling of their siblings.  Every
    other kind keeps the systematic ``plot_<kind>`` spelling.
    """
    key = re.sub(r"\W+", "_", kind).strip("_")
    if key in {"qq_plot", "qqplot"}:
        return "qqplot"
    if key == "pp_plot":
        return "ppplot"
    if key.endswith("plot"):
        return key
    return "plot_" + key


for _alias in sorted(KIND_ALIASES):
    _name = _public_plot_name(_alias)
    PUBLIC_KINDS.setdefault(_name, _alias)
    PLOT_FUNCTIONS[_name] = _make_plot_function(_alias, _name)

# Common spellings that are not registry aliases.
PUBLIC_KINDS.setdefault("qqplot", "qq")
PLOT_FUNCTIONS["qqplot"] = _make_plot_function("qq", "qqplot")
PUBLIC_KINDS.setdefault("plot_grafica", None)
PLOT_FUNCTIONS["plot_grafica"] = _make_plot_function(None, "plot_grafica")

for _name, _kind in PUBLIC_KINDS.items():
    setattr(PlottingMixin, _name, _make_plot_method(_kind, _name))

__all__ = [
    "PlottingMixin",
    "PUBLIC_KINDS",
    "PLOT_FUNCTIONS",
    "plot",
    *PLOT_FUNCTIONS,
]
