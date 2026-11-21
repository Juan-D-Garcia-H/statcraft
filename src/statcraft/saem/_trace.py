"""Compile the user's structural function to an expression graph (evaluated in C++).

The function is *traced*: it is called once with :class:`Tr` objects instead of numbers/arrays; each
numpy operation (``exp``, ``log``, ``where``, arithmetic...) is recorded as a node. The resulting graph
runs in the C++ core without the GIL and on several threads (see ``include/saemfit/expr.hpp``).

Only **elementwise** functions of parameters, data columns and covariates are compiled. Anything that
cannot be recorded (``if`` on a traced value, ``sum``, indexing, ``doses``...) raises
:class:`TraceError` and the model keeps using the Python callback, as before.
"""
from __future__ import annotations

import difflib
import inspect
import math
from typing import Callable, Optional, Sequence

import numpy as np

from ..utils.modelview import ModelView

try:  # optional: erf, erfc, expit, gammaln
    from scipy import special as _sp
except Exception:  # pragma: no cover
    _sp = None


class TraceError(Exception):
    """The function cannot be compiled (the reason is in the message)."""


def _np_expit(x):
    """NumPy implementation of the logistic sigmoid, used to fold constant ``expit`` calls."""
    return 1.0 / (1.0 + np.exp(-x))


def _np_gammaln(x):
    """NumPy implementation of the log-gamma function for constant folding."""
    return np.vectorize(math.lgamma, otypes=[float])(x)


# Operation table: (name, nargs, numpy function used to fold constants). The ORDER must match
# `enum ExprOp` in include/saemfit/expr.hpp (checked with _core.EXPR_NOPS).
OPS = [
    ("CONST", 0, None), ("PSI", 0, None), ("X", 0, None), ("COV", 0, None),
    ("ADD", 2, np.add), ("SUB", 2, np.subtract), ("MUL", 2, np.multiply), ("DIV", 2, np.divide),
    ("POW", 2, np.power), ("NEG", 1, np.negative),
    ("EXP", 1, np.exp), ("LOG", 1, np.log), ("LOG10", 1, np.log10), ("LOG2", 1, np.log2),
    ("LOG1P", 1, np.log1p), ("EXPM1", 1, np.expm1), ("EXP2", 1, np.exp2),
    ("SQRT", 1, np.sqrt), ("CBRT", 1, np.cbrt), ("ABS", 1, np.abs), ("SIGN", 1, np.sign),
    ("SIN", 1, np.sin), ("COS", 1, np.cos), ("TAN", 1, np.tan),
    ("ASIN", 1, np.arcsin), ("ACOS", 1, np.arccos), ("ATAN", 1, np.arctan),
    ("SINH", 1, np.sinh), ("COSH", 1, np.cosh), ("TANH", 1, np.tanh),
    ("ASINH", 1, np.arcsinh), ("ACOSH", 1, np.arccosh), ("ATANH", 1, np.arctanh),
    ("FLOOR", 1, np.floor), ("CEIL", 1, np.ceil), ("SQUARE", 1, np.square), ("RECIP", 1, np.reciprocal),
    ("MAX", 2, np.maximum), ("MIN", 2, np.minimum), ("ATAN2", 2, np.arctan2), ("MOD", 2, np.remainder),
    ("LT", 2, np.less), ("LE", 2, np.less_equal), ("GT", 2, np.greater), ("GE", 2, np.greater_equal),
    ("EQ", 2, np.equal), ("NE", 2, np.not_equal),
    ("AND", 2, np.logical_and), ("OR", 2, np.logical_or), ("NOT", 1, np.logical_not),
    ("WHERE", 3, np.where),
    ("ERF", 1, getattr(_sp, "erf", None)), ("ERFC", 1, getattr(_sp, "erfc", None)),
    ("EXPIT", 1, _np_expit), ("GAMMALN", 1, _np_gammaln),
    # per-row / per-dose inputs and reduction over doses (no numpy function: they are not folded)
    ("YTYPE", 0, None), ("Y", 0, None), ("EXTRA", 0, None),
    ("DT", 0, None), ("DAMT", 0, None), ("DDUR", 0, None), ("DCMT", 0, None),
    ("SUMDOSE", 2, None),
]
_INDEX_OPS = ("PSI", "X", "COV", "EXTRA")          # ops whose `a` is a column index, not a node
_CODE = {name: i for i, (name, _, _) in enumerate(OPS)}
_NARGS = [n for _, n, _ in OPS]
_FN = [f for _, _, f in OPS]

# numpy/scipy ufunc name -> operation
_UFUNC = {
    "add": "ADD", "subtract": "SUB", "multiply": "MUL", "divide": "DIV", "true_divide": "DIV",
    "power": "POW", "float_power": "POW", "negative": "NEG", "positive": None,
    "exp": "EXP", "log": "LOG", "log10": "LOG10", "log2": "LOG2", "log1p": "LOG1P", "expm1": "EXPM1",
    "exp2": "EXP2", "sqrt": "SQRT", "cbrt": "CBRT", "absolute": "ABS", "fabs": "ABS", "sign": "SIGN",
    "sin": "SIN", "cos": "COS", "tan": "TAN", "arcsin": "ASIN", "arccos": "ACOS", "arctan": "ATAN",
    "sinh": "SINH", "cosh": "COSH", "tanh": "TANH", "arcsinh": "ASINH", "arccosh": "ACOSH",
    "arctanh": "ATANH", "floor": "FLOOR", "ceil": "CEIL", "square": "SQUARE", "reciprocal": "RECIP",
    "maximum": "MAX", "minimum": "MIN", "fmax": None, "fmin": None, "arctan2": "ATAN2",
    "remainder": "MOD", "mod": "MOD",
    "less": "LT", "less_equal": "LE", "greater": "GT", "greater_equal": "GE", "equal": "EQ", "not_equal": "NE",
    "logical_and": "AND", "logical_or": "OR", "logical_not": "NOT",
    "erf": "ERF", "erfc": "ERFC", "expit": "EXPIT", "gammaln": "GAMMALN",
}


class NS:
    """Read-only namespace: ``params.Emax`` / ``params["Emax"]`` / ``Emax, ED50 = params("Emax", "ED50")``.

    It is used both on the compiled path (traced values) and in the Python callback (numeric values), and
    a missing name gives an error with suggestions instead of an opaque KeyError."""
    __slots__ = ("_d", "_what", "_hint")

    def __init__(self, d: dict, what: str, hint: str = ""):
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_what", what)
        object.__setattr__(self, "_hint", hint)

    def _get(self, n):
        """Return entry ``n``, raising a helpful error (with suggestions) when it is missing."""
        try:
            return self._d[n]
        except KeyError:
            hint = difflib.get_close_matches(str(n), list(self._d), n=1)
            msg = f"{self._what}: '{n}' does not exist. Available: {list(self._d)}"
            if hint:
                msg += f". Did you mean '{hint[0]}'?"
            if self._hint:
                msg += f". {self._hint}"
            raise AttributeError(msg) from None

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return self._get(n)

    def __getitem__(self, n):
        return self._get(n)

    def __call__(self, *names):
        vals = tuple(self._get(n) for n in names)
        return vals[0] if len(vals) == 1 else vals

    def __setattr__(self, n, v):
        raise AttributeError(f"{self._what} is read-only")

    def __contains__(self, n):
        return n in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def keys(self):
        """Names available in the namespace."""
        return self._d.keys()

    def __repr__(self):
        return f"{self._what}({', '.join(self._d)})"


class SeqNS(NS):
    """Named sequence: unpacked/indexed by POSITION like a SciPy array and also accessed
    by name. ``Emax, ED50, Hill = params``; ``params[1]``; ``params.ED50``; ``len(params)``."""
    __slots__ = ("_order",)

    def __init__(self, names, values, what: str, hint: str = ""):
        names = tuple(names)
        super().__init__(dict(zip(names, values)), what, hint)
        object.__setattr__(self, "_order", names)

    def __iter__(self):                      # values, in order (unpacking)
        return iter([self._d[n] for n in self._order])

    def __len__(self):
        return len(self._order)

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            return self._d[self._order[k]]
        if isinstance(k, slice):
            return tuple(self._d[n] for n in self._order[k])
        return self._get(k)

    def keys(self):
        """Names of the entries, in declared order."""
        return self._order

    def __repr__(self):
        return f"{self._what}({', '.join(self._order)})"


class SinglePred:
    """A single predictor: used directly (``x``) and still accepting ``x[0]``/``x["name"]``.

    Avoids writing ``x[0]`` when ``predictors=[...]`` has a single column. Arithmetic and
    ufuncs are delegated to the wrapped value (an array, a scalar or a :class:`Tr` during tracing), so
    it works both in the Python callback and in the compilation to C++.
    """
    __slots__ = ("_v", "_name")
    __array_priority__ = 1000

    def __init__(self, value, name=None):
        self._v, self._name = value, name

    # -- as a one-element sequence (compatibility with predictors[0]) --
    def __len__(self):
        return 1

    def __iter__(self):
        return iter((self._v,))

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            if k in (0, -1):
                return self._v
            raise IndexError(f"predictors has 1 element; index {k} out of range")
        if isinstance(k, slice):
            return (self._v,)[k]
        if k == self._name:
            return self._v
        raise AttributeError(f"predictors: {k!r} does not exist. Available: {[self._name]}")

    def keys(self):
        """Name of the single predictor (empty when unnamed)."""
        return (self._name,) if self._name is not None else ()

    def __contains__(self, k):
        return k == self._name or k in (0, -1)

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        if n == self._name:
            return self._v
        return getattr(self._v, n)

    def __repr__(self):
        return f"predictors({self._name}={self._v!r})"

    # -- arithmetic delegated to the wrapped value --
    def __add__(s, o): return s._v + _unwrap_single(o)
    def __radd__(s, o): return _unwrap_single(o) + s._v
    def __sub__(s, o): return s._v - _unwrap_single(o)
    def __rsub__(s, o): return _unwrap_single(o) - s._v
    def __mul__(s, o): return s._v * _unwrap_single(o)
    def __rmul__(s, o): return _unwrap_single(o) * s._v
    def __truediv__(s, o): return s._v / _unwrap_single(o)
    def __rtruediv__(s, o): return _unwrap_single(o) / s._v
    def __pow__(s, o): return s._v ** _unwrap_single(o)
    def __rpow__(s, o): return _unwrap_single(o) ** s._v
    def __mod__(s, o): return s._v % _unwrap_single(o)
    def __rmod__(s, o): return _unwrap_single(o) % s._v
    def __neg__(s): return -s._v
    def __pos__(s): return +s._v
    def __abs__(s): return abs(s._v)
    def __lt__(s, o): return s._v < _unwrap_single(o)
    def __le__(s, o): return s._v <= _unwrap_single(o)
    def __gt__(s, o): return s._v > _unwrap_single(o)
    def __ge__(s, o): return s._v >= _unwrap_single(o)
    def __eq__(s, o): return s._v == _unwrap_single(o)
    def __ne__(s, o): return s._v != _unwrap_single(o)
    def __and__(s, o): return s._v & _unwrap_single(o)
    def __rand__(s, o): return _unwrap_single(o) & s._v
    def __or__(s, o): return s._v | _unwrap_single(o)
    def __ror__(s, o): return _unwrap_single(o) | s._v
    def __invert__(s): return ~s._v
    def __bool__(s): return bool(s._v)
    def __float__(s): return float(s._v)
    def __int__(s): return int(s._v)

    # -- numpy integration --
    def __array__(self, dtype=None, copy=None):
        a = np.asarray(self._v, dtype=dtype)
        return a.copy() if copy else a

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(_unwrap_single(v) for v in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return func(*_unwrap_single(args), **_unwrap_single(kwargs))


def _unwrap_single(v):
    """Unwrap ``SinglePred`` (and numpy containers holding one)."""
    if isinstance(v, SinglePred):
        return v._v
    if isinstance(v, tuple):
        return tuple(_unwrap_single(x) for x in v)
    if isinstance(v, list):
        return [_unwrap_single(x) for x in v]
    if isinstance(v, dict):
        return {k: _unwrap_single(x) for k, x in v.items()}
    return v


DOSE_KERNEL_ARGS = ("dt", "amt", "dur", "cmt")


def kernel_argnames(kernel) -> list:
    """Arguments of the ``doses.superpose`` kernel (a subset of dt, amt, dur, cmt), validated."""
    names = []
    for n, p in inspect.signature(kernel).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError("doses.superpose: the kernel does not accept *args/**kwargs")
        if n not in DOSE_KERNEL_ARGS:
            raise TypeError(f"doses.superpose: the kernel receives {DOSE_KERNEL_ARGS}, not '{n}'")
        names.append(n)
    return names


class _Graph:
    """Expression graph under construction.

    Nodes are stored in topological order; ``op`` holds the operation code and ``a``/``b``/``c``
    the argument nodes (or column indices for the index operations). Constant folding and node
    deduplication happen on insertion, so repeated sub-expressions share a single node.
    """

    def __init__(self):
        self.op, self.a, self.b, self.c, self.cst = [], [], [], [], []
        self._cache = {}

    def add(self, code, a=-1, b=-1, c=-1, cst=0.0):
        """Add a node (or reuse an identical one) and return its id.

        ``code`` is the operation code, ``a``/``b``/``c`` are the argument node ids and ``cst``
        is the payload for constant nodes (``code == 0``).
        """
        key = (code, a, b, c, repr(float(cst)) if code == 0 else "")
        k = self._cache.get(key)
        if k is None:
            k = len(self.op)
            self.op.append(code); self.a.append(a); self.b.append(b); self.c.append(c); self.cst.append(float(cst))
            self._cache[key] = k
        return k

    def const(self, v):
        """Return the id of a constant node holding ``v``."""
        return self.add(0, cst=float(v))

    def is_const(self, k):
        """Return ``True`` when node ``k`` is a constant."""
        return self.op[k] == 0

    def mk(self, name, *ids):
        """Node `name` with arguments `ids`; folds constants (if all arguments are constants)."""
        code = _CODE[name]
        if all(self.is_const(i) for i in ids):
            fn = _FN[code]
            if fn is not None:
                with np.errstate(all="ignore"):
                    v = fn(*[self.cst[i] for i in ids])
                v = np.asarray(v)
                if v.ndim == 0:
                    return self.const(float(v))
        ids = list(ids) + [-1] * (3 - len(ids))
        return self.add(code, ids[0], ids[1], ids[2])


def _id(g: _Graph, v) -> int:
    """Return the graph node id of ``v``, adding a constant node when it is a plain number."""
    if isinstance(v, SinglePred):
        return _id(g, v._v)
    if isinstance(v, ModelView):
        single = v._single()
        if single is not None:
            return _id(g, single)
    if isinstance(v, Tr):
        if v._g is not g:
            raise TraceError("mixing different traces")
        return v._id
    if isinstance(v, (bool, int, float, np.number, np.bool_)):
        return g.const(float(v))
    if isinstance(v, np.ndarray) and v.ndim == 0:
        if v.dtype == object:                       # np.asarray(Tr) -> 0-d object array
            return _id(g, v.item())
        return g.const(float(v))
    raise TraceError(f"value of type {type(v).__name__} cannot be compiled (scalars and data columns only)")


class Tr:
    """Traced value: a graph node. Supports arithmetic and numpy ufuncs."""
    __slots__ = ("_g", "_id")
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(self, g, i):
        self._g, self._id = g, i

    # -- arithmetic --
    def _b(self, name, other, swap=False):
        """Build a binary node ``name`` from ``self`` and ``other``.

        ``swap`` puts ``other`` first, which is needed by the reflected operators (``__radd__``,
        ``__rsub__``, ...).
        """
        a, b = _id(self._g, self), _id(self._g, other)
        return Tr(self._g, self._g.mk(name, b, a) if swap else self._g.mk(name, a, b))

    def __add__(s, o): return s._b("ADD", o)
    def __radd__(s, o): return s._b("ADD", o, True)
    def __sub__(s, o): return s._b("SUB", o)
    def __rsub__(s, o): return s._b("SUB", o, True)
    def __mul__(s, o): return s._b("MUL", o)
    def __rmul__(s, o): return s._b("MUL", o, True)
    def __truediv__(s, o): return s._b("DIV", o)
    def __rtruediv__(s, o): return s._b("DIV", o, True)
    def __mod__(s, o): return s._b("MOD", o)
    def __rmod__(s, o): return s._b("MOD", o, True)
    def __lt__(s, o): return s._b("LT", o)
    def __le__(s, o): return s._b("LE", o)
    def __gt__(s, o): return s._b("GT", o)
    def __ge__(s, o): return s._b("GE", o)
    def __eq__(s, o): return s._b("EQ", o)
    def __ne__(s, o): return s._b("NE", o)
    def __and__(s, o): return s._b("AND", o)
    def __rand__(s, o): return s._b("AND", o, True)
    def __or__(s, o): return s._b("OR", o)
    def __ror__(s, o): return s._b("OR", o, True)
    def __invert__(s): return Tr(s._g, s._g.mk("NOT", s._id))
    def __neg__(s): return Tr(s._g, s._g.mk("NEG", s._id))
    def __pos__(s): return s
    def __abs__(s): return Tr(s._g, s._g.mk("ABS", s._id))

    def __pow__(s, o):
        return Tr(s._g, _pow(s._g, s._id, _id(s._g, o)))

    def __rpow__(s, o):
        return Tr(s._g, s._g.mk("POW", _id(s._g, o), s._id))

    # -- anything that depends on the concrete value / shape is not compilable --
    def __bool__(self): raise TraceError("control flow (Python if/and/or/max/min) on a model value")
    def __float__(self): raise TraceError("conversion to float of a model value (math.*, float())")
    __int__ = __float__
    __index__ = __float__
    def __len__(self): raise TraceError("len() of a model value")
    def __iter__(self): raise TraceError("iteration over a model value")
    def __getitem__(self, k): raise TraceError("indexing a model value")

    def __getattr__(self, name):
        if name.startswith("__"):                   # python/numpy protocols (np.asarray, copy...)
            raise AttributeError(name)
        raise TraceError(f"attribute/method '{name}' is not supported on a model value")

    # -- numpy integration --
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None or kwargs.get("where", True) is not True:
            raise TraceError(f"ufunc {ufunc.__name__}.{method} is not supported")
        name = ufunc.__name__
        if name not in _UFUNC:
            raise TraceError(f"numpy function '{name}' is not supported")
        g = self._g
        op = _UFUNC[name]
        ids = [_id(g, v) for v in inputs]
        if name in ("positive",):
            return Tr(g, ids[0])
        if name in ("fmax", "fmin"):
            raise TraceError(f"'{name}' ignores NaN differently from maximum/minimum: use np.maximum/np.minimum")
        if op is None or len(ids) != _NARGS[_CODE[op]]:
            raise TraceError(f"numpy function '{name}' with unsupported arguments")
        if op == "POW":
            return Tr(g, _pow(g, ids[0], ids[1]))
        if op in ("ERF", "ERFC") and _FN[_CODE[op]] is None:
            raise TraceError("scipy is not available")
        return Tr(g, g.mk(op, *ids))

    def __array_function__(self, func, types, args, kwargs):
        g = self._g
        if func is np.where and len(args) == 3 and not kwargs:
            return Tr(g, g.mk("WHERE", *[_id(g, v) for v in args]))
        if func is np.clip and 2 <= len(args) <= 3:
            lo = args[1]
            hi = args[2] if len(args) == 3 else kwargs.get("a_max")
            if len(args) == 2:
                lo = kwargs.get("a_min", args[1])
            x = _id(g, args[0])
            if lo is not None:
                x = g.mk("MAX", x, _id(g, lo))
            if hi is not None:
                x = g.mk("MIN", x, _id(g, hi))
            return Tr(g, x)
        raise TraceError(f"numpy function '{getattr(func, '__name__', func)}' is not supported")


def _pow(g: _Graph, a: int, b: int) -> int:
    """a ** b with numpy's fast paths for constant exponents (2, 0.5, -1, 1)."""
    if g.is_const(b):
        e = g.cst[b]
        if e == 2.0: return g.mk("SQUARE", a)
        if e == 0.5: return g.mk("SQRT", a)
        if e == -1.0: return g.mk("RECIP", a)
        if e == 1.0: return a
    return g.mk("POW", a, b)


class TrDoses:
    """``doses`` during tracing: only ``superpose`` is compilable."""

    def __init__(self, g: _Graph):
        self.__dict__["_g"] = g

    def superpose(self, time, kernel):
        """Sum of ``kernel(dt, amt, dur, cmt)`` over doses with ``time >= dose_time`` (``dt = time - dose_time``)."""
        g = self._g
        t_id = _id(g, time)
        kw = {}
        for n in kernel_argnames(kernel):
            kw[n] = Tr(g, g.add(_CODE[{"dt": "DT", "amt": "DAMT", "dur": "DDUR", "cmt": "DCMT"}[n]]))
        core = _id(g, kernel(**kw))
        return Tr(g, g.mk("SUMDOSE", core, t_id))

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        raise TraceError(f"doses.{n} is not compilable (use doses.superpose(time, kernel))")

    def __len__(self): raise TraceError("len(doses) is not compilable")
    def __iter__(self): raise TraceError("iterating doses is not compilable (use doses.superpose)")
    def __bool__(self): raise TraceError("doses in a conditional is not compilable")


class TrExtra:
    """``extra`` (custom likelihood) during tracing: only ``extra[i]`` with a constant integer i."""

    def __init__(self, g: _Graph):
        self._g = g

    def __getitem__(self, i):
        if not isinstance(i, (int, np.integer)):
            raise TraceError("extra only accepts integer indices (extra[0])")
        return Tr(self._g, self._g.add(_CODE["EXTRA"], a=int(i)))

    def __len__(self): raise TraceError("len(extra) is not compilable")
    def __iter__(self): raise TraceError("iterating extra is not compilable")


class Program:
    """Pruned graph, ready for the core."""

    def __init__(self, op, a, b, c, cst, outs):
        self.op, self.a, self.b, self.c, self.cst, self.outs = op, a, b, c, cst, list(outs)

    def arrays(self):
        """Flat arrays ``(op, a, b, c, cst, outs)`` consumed by the C++ core."""
        return self.op, self.a, self.b, self.c, self.cst, self.outs

    def __len__(self):
        return len(self.op)


def _prune(g: _Graph, outs: Sequence[int]) -> Program:
    """Remove nodes that do not reach any output and renumber them preserving topological order."""
    n = len(g.op)
    live = [False] * n
    stack = list(outs)
    while stack:
        k = stack.pop()
        if live[k]:
            continue
        live[k] = True
        for t in range(_NARGS[g.op[k]]):
            stack.append((g.a[k], g.b[k], g.c[k])[t])
    idx_codes = {_CODE[nm] for nm in _INDEX_OPS}
    new = {}
    op, a, b, c, cst = [], [], [], [], []
    for k in range(n):
        if not live[k]:
            continue
        new[k] = len(op)
        na = _NARGS[g.op[k]]
        op.append(g.op[k]); cst.append(g.cst[k])
        if g.op[k] in idx_codes:                 # `a` is a column/extra index, not a node
            a.append(g.a[k]); b.append(-1); c.append(-1)
        else:
            args = [g.a[k], g.b[k], g.c[k]]
            a.append(new[args[0]] if na > 0 else -1)
            b.append(new[args[1]] if na > 1 else -1)
            c.append(new[args[2]] if na > 2 else -1)
    return Program(op, a, b, c, cst, [new[o] for o in outs])


class Roles:
    """Roles of the function arguments (the same codes as model.py) and data names."""

    def __init__(self, psi, x, cov, doses, ytype, extra, y, pns, dns, prd, params,
                 x_names, cov_names, pred_spec, xv=None, xv_spec=None):
        self.psi, self.x, self.cov, self.doses, self.ytype = psi, x, cov, doses, ytype
        self.extra, self.y, self.pns, self.dns, self.prd, self.xv = extra, y, pns, dns, prd, xv
        self.params, self.x_names, self.cov_names = list(params), list(x_names), list(cov_names)
        self.time_alias = True                     # expose data.t / data.time as aliases of column 0
        self.pred_spec = pred_spec                 # callable -> [(name, "x"|"cov", index)] (lazy: may raise)
        self.xv_spec = xv_spec                     # same, but for the portable full ``x`` view


def trace_program(fn: Callable, names, kinds, idx, R: Roles, n_out: Optional[int] = 1) -> Program:
    """Trace ``fn(**args)``. ``names/kinds/idx`` come from NLModel._plan. Raises TraceError if not compilable.

    ``n_out``: expected number of outputs (None = any; the function returns a value or a sequence)."""
    g = _Graph()
    kw = {}
    for n, k, i in zip(names, kinds, idx):
        if k == R.psi:
            kw[n] = Tr(g, g.add(_CODE["PSI"], a=int(i)))
        elif k == R.x:
            kw[n] = Tr(g, g.add(_CODE["X"], a=int(i)))
        elif k == R.cov:
            kw[n] = Tr(g, g.add(_CODE["COV"], a=int(i)))
        elif k == R.ytype:
            kw[n] = Tr(g, g.add(_CODE["YTYPE"]))
        elif k == R.y:
            kw[n] = Tr(g, g.add(_CODE["Y"]))
        elif k == R.extra:
            kw[n] = TrExtra(g)
        elif k == R.doses:
            kw[n] = TrDoses(g)
        elif k == R.pns:
            kw[n] = ModelView(R.params, [Tr(g, g.add(_CODE["PSI"], a=j)) for j in range(len(R.params))],
                              label="params")
        elif k == R.xv:
            spec = R.xv_spec()
            names = [nm for nm, _, _ in spec]
            vals = [Tr(g, g.add(_CODE["X" if src == "x" else "COV"], a=int(j))) for _, src, j in spec]
            kw[n] = ModelView(names, vals, label="x", columnar=True,
                              extras={"doses": TrDoses(g)})
        elif k == R.prd:
            spec = R.pred_spec()
            vals = [Tr(g, g.add(_CODE["X" if src == "x" else "COV"], a=int(j))) for _, src, j in spec]
            if len(spec) == 1:
                kw[n] = SinglePred(vals[0], spec[0][0])
            else:
                kw[n] = SeqNS([nm for nm, _, _ in spec], vals, "predictors")
        elif k == R.dns:
            d = {nm: Tr(g, g.add(_CODE["X"], a=j)) for j, nm in enumerate(R.x_names)}
            if R.time_alias:
                for alias in ("t", "time"):
                    d.setdefault(alias, Tr(g, g.add(_CODE["X"], a=0)))
            for j, nm in enumerate(R.cov_names):
                d[nm] = Tr(g, g.add(_CODE["COV"], a=j))
            kw[n] = NS(d, "data", "Columns must be declared in predictors=[...] (or be model covariates)")
        else:
            raise TraceError(f"argument '{n}' is not compilable")
    try:
        with np.errstate(all="ignore"):
            r = fn(**kw)
    except TraceError:
        raise
    except Exception as e:  # anything else (AttributeError, TypeError...) = not compilable
        raise TraceError(f"{type(e).__name__}: {e}") from e
    if isinstance(r, np.ndarray) and r.dtype == object and r.ndim == 0:
        r = r.item()
    if isinstance(r, (tuple, list)) or (isinstance(r, np.ndarray) and r.ndim == 1):
        outs = [_id(g, v) for v in r]
    else:
        outs = [_id(g, r)]
    if not outs:
        raise TraceError("the function returned an empty sequence")
    if n_out is not None and len(outs) != n_out:
        raise TraceError(f"the function returns {len(outs)} values but {n_out} were expected")
    return _prune(g, outs)
