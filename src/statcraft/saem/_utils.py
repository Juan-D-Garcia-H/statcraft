"""Internal utilities: enums by name and covariate-formula parsing."""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from .. import _saemfit as _core

_ERROR_ALIASES = {
    "constant": "constant", "const": "constant", "additive": "constant", "add": "constant", "a": "constant",
    "proportional": "proportional", "prop": "proportional", "b": "proportional",
    "exponential": "exponential", "exp": "exponential", "lognormal": "exponential",
    "combined": "combined", "combined1": "combined", "comb": "combined", "comb1": "combined",
    "combined2": "combined2", "comb2": "combined2",
    "power": "power", "pow": "power",
}
_TRANSFORM_ALIASES = {
    "identity": "identity", "id": "identity", "none": "identity", "normal": "identity", "linear": "identity",
    "log": "log", "lognormal": "log",
    "logit": "logit",
    "probit": "probit",
}
# number of error parameters per model (a, b, c), and which ones are estimated
ERROR_PARS = {
    "constant": ("a",), "proportional": ("b",), "exponential": ("a",),
    "combined": ("a", "b"), "combined2": ("a", "b"), "power": ("a", "b", "c"),
}


def _norm(s) -> str:
    """Normalize an option name: lower-case, with ``-``/spaces converted to ``_``."""
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def error_name(value) -> str:
    """Canonical error-model name for a user-provided alias (e.g. ``"prop"`` -> ``"proportional"``)."""
    key = _norm(value)
    if key not in _ERROR_ALIASES:
        raise ValueError(f"unknown error model: {value!r}. Options: "
                         f"{sorted(set(_ERROR_ALIASES.values()))}")
    return _ERROR_ALIASES[key]


def error_enum(value):
    """Low-level ``ErrorModel`` enum value for a user-provided error-model name."""
    return getattr(_core.ErrorModel, error_name(value))


def transform_name(value) -> str:
    """Canonical parameter-transform name (``None`` becomes ``"identity"``)."""
    key = _norm(value if value is not None else "identity")
    if key not in _TRANSFORM_ALIASES:
        raise ValueError(f"unknown transform: {value!r}. Options: identity, log, logit, probit")
    return _TRANSFORM_ALIASES[key]


def transform_enum(value):
    """Low-level ``Transform`` enum value for a user-provided transform name."""
    return getattr(_core.Transform, transform_name(value))


def enum_from(enum_cls, value, what: str):
    """Resolve ``value`` against ``enum_cls``, accepting an enum instance or a normalized name.

    ``what`` names the option in the error message.
    """
    if isinstance(value, enum_cls):
        return value
    key = _norm(value)
    try:
        return getattr(enum_cls, key)
    except AttributeError:
        opts = [n for n in dir(enum_cls) if not n.startswith("_") and n not in ("name", "value")]
        raise ValueError(f"{what}: {value!r} is not valid. Options: {opts}") from None


def apply_transform(x, name: str):
    """h(phi): phi -> psi (numpy)."""
    import numpy as np
    from math import sqrt
    if name == "identity":
        return x
    if name == "log":
        return np.exp(x)
    if name == "logit":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "probit":
        from math import erf
        return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / sqrt(2.0)))
    raise ValueError(name)


_SPLIT = re.compile(r"\s*\+\s*")


def _terms(rhs: str) -> list[str]:
    """Split a formula right-hand side into validated column names.

    Terms are separated by ``+``; a bare ``1`` (the intercept) is ignored. Only plain column
    names are accepted (``name`` or ``table.name``); derived terms must be computed beforehand.
    """
    out = [t.strip() for t in _SPLIT.split(rhs.strip()) if t.strip() and t.strip() not in ("1",)]
    for t in out:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", t):
            raise ValueError(f"invalid covariate term: {t!r} (column names only; "
                             "transform the column beforehand: df['lwt'] = np.log(df.wt / 70))")
    return out


def parse_covariates(spec, params: Sequence[str]) -> list[tuple[str, str]]:
    """Accepts  ["cl ~ wt + sex", "v ~ wt"]  |  {"cl": ["wt", "sex"], "v": "wt"}  |  "cl + v ~ wt".
    Returns [(parameter, covariate), ...] without duplicates."""
    if spec is None:
        return []
    pairs: list[tuple[str, str]] = []

    def add(lhs_list: Iterable[str], covs: Iterable[str]):
        """Register every ``(parameter, covariate)`` pair, validating the parameter names."""
        for p in lhs_list:
            if p not in params:
                raise ValueError(f"covariates: parameter {p!r} is not in params={list(params)}")
            for c in covs:
                if (p, c) not in pairs:
                    pairs.append((p, c))

    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, Mapping):
        for p, v in spec.items():
            covs = _terms(v) if isinstance(v, str) else [str(c) for c in v]
            add([p], covs)
    else:
        for s in spec:
            if "~" not in s:
                raise ValueError(f"covariate formula without '~': {s!r}  (e.g. 'cl ~ wt + sex')")
            lhs, rhs = s.split("~", 1)
            add(_terms(lhs), _terms(rhs))
    return pairs
