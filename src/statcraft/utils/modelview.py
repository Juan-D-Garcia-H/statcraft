# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Lightweight named views for the ``x`` and ``p`` arguments of model functions.

A :class:`ModelView` is a read-only sequence whose entries also have names. It
is what makes the portable two-argument convention work::

    def model(x, p):
        Time, Dose = x          # unpack in the declared order
        ka, CL, V = p           # same for the parameters
        return ...              # x["Time"], x.Time, p["CL"], p.V also work

Views are used by NLS, NLME and SAEM. The same object works during SAEM's
compilation pass, where entries are tracer objects instead of numbers. A view
with one entry also behaves directly as that entry, so both ``a = p`` and
``a, = p`` work for a single-parameter model.
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping, Optional, Sequence

import numpy as np


# Reserved names for the parameter-sequence argument of a two-argument model.
# With more than one parameter any name is accepted; with one parameter these
# names disambiguate the vector form from ``def f(predictor, parameter)``.
PARAM_VIEW_NAMES = frozenset({"p", "params", "parameters", "theta"})


def _is_jax(value: Any) -> bool:
    """True for JAX arrays/tracers, which must be stacked with ``jax.numpy``."""
    return type(value).__module__.split(".")[0] in {"jax", "jaxlib"} or hasattr(value, "aval")


def _unwrap(value: Any) -> Any:
    """Return the underlying array of a view, leaving other objects untouched."""
    return value._array() if isinstance(value, ModelView) else value


def _jnp():
    """Import ``jax.numpy`` lazily: JAX is an optional dependency."""
    import jax.numpy as jnp

    return jnp


class ModelView:
    """Read-only, named, unpackable view over exogenous values or parameters.

    Parameters
    ----------
    names : sequence of str
        Names in declaration order. They define the unpacking order.
    values : sequence
        One value per name (a scalar, a NumPy/JAX array, or a SAEM tracer).
    label : str, default "x"
        Name used in error messages and ``repr``.
    columnar : bool, default False
        When True, ``np.asarray(view)`` stacks the values as columns (the
        convention for the exogenous matrix). When False it returns a 1-D
        parameter vector.

    Notes
    -----
    ``x[0]``/``p[0]`` use position, ``x["Time"]``/``p.ka`` use names. Integer
    slicing returns a tuple of values, and NumPy-style two-index access such
    as ``x[:, 0]`` selects columns before slicing rows. A view with a single
    entry behaves directly as that entry: ``np.asarray(x)`` is 1-D for one
    predictor and ``np.asarray(p)`` is 1-D for one parameter, while
    ``time, = x`` and ``a, = p`` still work.
    """

    __slots__ = ("_names", "_values", "_label", "_columnar", "_extras")
    __array_priority__ = 1000

    def __init__(self, names: Sequence[str], values: Sequence[Any], *, label: str = "x",
                 columnar: bool = False, extras: Optional[Mapping[str, Any]] = None) -> None:
        names = tuple(str(n) for n in names)
        values = tuple(values)
        if len(names) != len(values):
            raise ValueError(f"{label}: {len(names)} name(s) but {len(values)} value(s)")
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_label", str(label))
        object.__setattr__(self, "_columnar", bool(columnar))
        object.__setattr__(self, "_extras", dict(extras or {}))

    # -- basic sequence protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, (int, np.integer)):
            return -len(self._values) <= int(key) < len(self._values)
        return key in self._names or key in self._extras

    def _single(self) -> Any:
        """Return the sole value of a one-entry view, or ``None``."""
        return self._values[0] if len(self._values) == 1 else None

    def __repr__(self) -> str:
        return f"{self._label}({', '.join(self._names)})"

    def keys(self):
        """Names in declaration order."""
        return self._names

    @property
    def values(self):
        """Values in declaration order."""
        return self._values

    # -- name and index lookup ---------------------------------------------------
    def _get_name(self, name: Any) -> Any:
        if name in self._names:
            return self._values[self._names.index(name)]
        if name in self._extras:
            return self._extras[name]
        hint = difflib.get_close_matches(str(name), list(self._names) + list(self._extras), n=1)
        msg = f"{self._label}: {name!r} does not exist. Available: {list(self._names)}"
        if hint:
            msg += f". Did you mean {hint[0]!r}?"
        raise AttributeError(msg)

    def _get_index(self, key: Any) -> Any:
        if isinstance(key, (int, np.integer)):
            return self._values[int(key)]
        if isinstance(key, slice):
            return tuple(self._values[key])
        if isinstance(key, (list, tuple)):
            names, values = [], []
            for k in key:
                if isinstance(k, (int, np.integer)):
                    i = int(k)
                    names.append(self._names[i]); values.append(self._values[i])
                else:
                    names.append(str(k)); values.append(self._get_name(k))
            return ModelView(names, values, label=self._label, columnar=self._columnar,
                             extras=self._extras)
        return self._get_name(key)

    def _column_key(self, key: Any) -> Any:
        """Translate a column key (position or name) into an array column index."""
        if isinstance(key, (int, np.integer)):
            return int(key)
        if isinstance(key, slice):
            return key
        if isinstance(key, (list, tuple)):
            return [self._column_key(k) for k in key]
        return self._names.index(key)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"{self._label}: expected view[rows, columns], got {len(key)} indices")
            rows, cols = key
            matrix = self._array(columnar=True, force_2d=True)
            return matrix[rows, self._column_key(cols)]
        return self._get_index(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self._names:                      # names win over helper attributes
            return self._values[self._names.index(name)]
        if name in self._extras:
            return self._extras[name]
        if name in ("shape", "ndim", "size", "dtype", "T"):
            arr = self._array(columnar=True)
            return getattr(arr, name)
        raise AttributeError(f"{self._label}: no value named {name!r}. Available: {list(self._names)}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{self._label} is read-only")

    # -- array conversion --------------------------------------------------------
    @staticmethod
    def _stack(values: Sequence[Any], *, columnar: bool, force_2d: bool = False):
        values = tuple(values)
        if not values:
            return np.asarray([])
        if any(type(v).__name__ == "Tr" and type(v).__module__.endswith("saem._trace") for v in values):
            raise TypeError("converting a traced x/p view to an array is not compilable; unpack or index it")
        if not columnar:
            return _jnp().asarray(values) if _is_jax(values[0]) else np.asarray(values)
        xp = _jnp() if _is_jax(values[0]) else np
        arrays = [xp.asarray(v) for v in values]
        if any(a.ndim > 0 for a in arrays):
            arrays = xp.broadcast_arrays(*arrays)
            out = xp.stack(arrays, axis=-1)
        else:
            out = xp.asarray(arrays)
        if force_2d and getattr(out, "ndim", 0) == 1:
            out = xp.reshape(out, (-1, 1))
        return out

    def _array(self, *, dtype=None, copy=None, columnar: Optional[bool] = None,
               force_2d: bool = False):
        columnar = self._columnar if columnar is None else columnar
        single = self._single() if not force_2d else None
        if single is not None:
            arr = np.asarray(single, dtype=dtype) if dtype is not None or copy is not None else single
            return arr.copy() if copy else arr
        arr = self._stack(self._values, columnar=columnar, force_2d=force_2d)
        if dtype is not None or copy is not None:
            kwargs = {} if copy is None else {"copy": copy}
            return np.asarray(arr, dtype=dtype, **kwargs)
        return arr

    def __array__(self, dtype=None, copy=None):
        single = self._single()
        if single is not None:
            arr = np.asarray(single, dtype=dtype)
            return arr.copy() if copy else arr
        return self._array(dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(_unwrap(v) for v in inputs)
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return getattr(ufunc, method)(*inputs, **kwargs)

    # -- operators: delegate to the wrapped value(s) -----------------------------
    def _binary(self, other, op):
        single = self._single()
        return op(single if single is not None else self._array(), _unwrap(other))

    def _rbinary(self, other, op):
        single = self._single()
        return op(_unwrap(other), single if single is not None else self._array())

    def __add__(self, other): return self._binary(other, lambda a, b: a + b)
    def __radd__(self, other): return self._rbinary(other, lambda a, b: a + b)
    def __sub__(self, other): return self._binary(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._rbinary(other, lambda a, b: a - b)
    def __mul__(self, other): return self._binary(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._rbinary(other, lambda a, b: a * b)
    def __truediv__(self, other): return self._binary(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._rbinary(other, lambda a, b: a / b)
    def __floordiv__(self, other): return self._binary(other, lambda a, b: a // b)
    def __rfloordiv__(self, other): return self._rbinary(other, lambda a, b: a // b)
    def __mod__(self, other): return self._binary(other, lambda a, b: a % b)
    def __rmod__(self, other): return self._rbinary(other, lambda a, b: a % b)
    def __pow__(self, other): return self._binary(other, lambda a, b: a ** b)
    def __rpow__(self, other): return self._rbinary(other, lambda a, b: a ** b)
    def __matmul__(self, other): return self._binary(other, lambda a, b: a @ b)
    def __rmatmul__(self, other): return self._rbinary(other, lambda a, b: a @ b)
    def __neg__(self): return -self._array()
    def __pos__(self): return +self._array()
    def __abs__(self): return abs(self._array())
    def __lt__(self, other): return self._binary(other, lambda a, b: a < b)
    def __le__(self, other): return self._binary(other, lambda a, b: a <= b)
    def __gt__(self, other): return self._binary(other, lambda a, b: a > b)
    def __ge__(self, other): return self._binary(other, lambda a, b: a >= b)
    def __eq__(self, other): return self._binary(other, lambda a, b: a == b)
    def __ne__(self, other): return self._binary(other, lambda a, b: a != b)
    __hash__ = None
