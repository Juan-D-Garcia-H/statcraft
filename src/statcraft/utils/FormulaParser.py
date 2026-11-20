# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""
FormulaParser
=============
Parses nonlinear regression formulas of the form:

    y ~ a * exp(b * x)
    y ~ a * np.exp(b * x)        # np. prefix also accepted
    y ~ SSlogis(x, Asym, xmid, scal)

Function namespace
------------------
Evaluation uses a **closed, explicit namespace** — only functions listed in
``FORMULA_NAMESPACE`` are available.  Any function not in the namespace raises
a clear ``FormulaNamespaceError`` at parse time, explaining exactly what the
function is and what to use instead.

This replaces the old ``UNSUPPORTED_SCIPY_FUNCTIONS`` blacklist approach:
instead of warning about things that are prohibited, the parser simply only
knows about things that are permitted.

Namespace categories
~~~~~~~~~~~~~~~~~~~~
- **Elementary**  : exp, log, log2, log10, log1p, expm1, exp2, sqrt, cbrt,
                    power, abs, sign, hypot
- **Trigonometric**: sin, cos, tan, arcsin, arccos, arctan, arctan2,
                     sinh, cosh, tanh, arcsinh, arccosh, arctanh
- **Rounding**    : floor, ceil, round
- **Special**     : gamma, lgamma, erf, erfc, beta  (via mpmath → numpy arrays)
- **Constants**   : pi, e, inf, nan

Special self-starting functions (SS*)
--------------------------------------
SSexp, SSlogis, SSmicmen, SSgompertz, SSweibull, SSasymp, SSfpl

Restrictions
-------------
- No transformations on the LHS  (e.g. log(y) ~ ... is forbidden)
- No dots in variable names      (use y_1 instead of y.1)
- Parameter names cannot clash with namespace functions or constants
  (e.g. 'e', 'pi', 'exp', 'sin' are reserved — use 'a', 'b', 'k' etc.)

Numeric evaluation backends
----------------------------
By default, numeric evaluation uses numpy (``backend="numpy"``, the
historical behavior). Passing ``backend="jax"`` to the constructor
translates the same SymPy expression to JAX instead of numpy: this
enables automatic differentiation (``FormulaParser.jacobian``) and JIT
compilation (``FormulaParser.to_jax_callable``), useful for HMC/NUTS and
for gradient-based optimization. Parsing, namespace validation and all
restrictions stay exactly the same — the backend only changes which
numeric engine executes the already-parsed expression. JAX is an
optional dependency: it is only imported when ``backend="jax"`` is
requested.
"""

import re
import warnings
from itertools import groupby
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sympy as sp
from scipy import special as _sp_special

from .exceptions import DataValidationError, FormulaError

# JAX is an OPTIONAL dependency. It is not imported at module level so it
# isn't forced on anyone who only uses backend="numpy" (the default). It
# is lazily resolved the first time backend="jax" is requested.
_jax_module: Any = None
_jax_import_error: Optional[Exception] = None


def _get_jax() -> Any:
    """Lazily imports and caches the jax module. Raises a clear error if
    jax isn't installed — only triggered when backend="jax" is requested."""
    global _jax_module, _jax_import_error
    if _jax_module is None and _jax_import_error is None:
        try:
            import jax
            import jax.numpy  # noqa: F401  (ensures jax.numpy is bound)
            import jax.scipy.special  # noqa: F401

            _jax_module = jax
        except ImportError as exc:
            _jax_import_error = exc
    if _jax_module is None:
        raise ImportError(
            "backend='jax' requires the 'jax' package, which is not installed. "
            "Install it with `pip install jax` (CPU) or the appropriate "
            "jax[cuda]/jax[tpu] extra for your hardware."
        ) from _jax_import_error
    return _jax_module


# ---------------------------------------------------------------------------
# FormulaNamespaceError
# ---------------------------------------------------------------------------


class FormulaNamespaceError(FormulaError):
    """
    Raised when a formula references a function not in FORMULA_NAMESPACE.

    A specialization of ``FormulaError`` (still a ``ValueError``, so
    existing ``except ValueError``/``except FormulaError`` code keeps
    working unchanged).

    Attributes
    ----------
    func_name  : str — offending function name
    suggestion : str — what to use instead (may be empty)
    """

    def __init__(self, func_name: str, suggestion: str = "") -> None:
        self.func_name = func_name
        self.suggestion = suggestion
        msg = (
            f"'{func_name}' is not available in the formula namespace.\n"
            "  Call FormulaParser.list_namespace() to see all available functions.\n"
        )
        if suggestion:
            msg += f"  Suggestion: {suggestion}\n"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# _NS — master function registry
# ---------------------------------------------------------------------------
# Each entry:  name → {numpy, sympy, cat, desc}

_NS: Dict[str, Dict[str, Any]] = {
    # ── Elementary ────────────────────────────────────────────────────────
    "exp": {"numpy": np.exp, "sympy": sp.exp, "cat": "elementary", "desc": "e^x"},
    "exp2": {"numpy": np.exp2, "sympy": sp.Function("exp2"), "cat": "elementary", "desc": "2^x"},
    "expm1": {
        "numpy": np.expm1,
        "sympy": sp.Function("expm1"),
        "cat": "elementary",
        "desc": "e^x - 1",
    },
    "log": {"numpy": np.log, "sympy": sp.log, "cat": "elementary", "desc": "natural logarithm"},
    "log2": {
        "numpy": np.log2,
        "sympy": sp.Function("log2"),
        "cat": "elementary",
        "desc": "log base 2",
    },
    "log10": {
        "numpy": np.log10,
        "sympy": sp.Function("log10"),
        "cat": "elementary",
        "desc": "log base 10",
    },
    "log1p": {
        "numpy": np.log1p,
        "sympy": sp.Function("log1p"),
        "cat": "elementary",
        "desc": "log(1 + x)",
    },
    "sqrt": {"numpy": np.sqrt, "sympy": sp.sqrt, "cat": "elementary", "desc": "square root"},
    "cbrt": {
        "numpy": np.cbrt,
        "sympy": sp.Function("cbrt"),
        "cat": "elementary",
        "desc": "cube root",
    },
    "power": {"numpy": np.power, "sympy": sp.Pow, "cat": "elementary", "desc": "x ** y"},
    "abs": {"numpy": np.abs, "sympy": sp.Abs, "cat": "elementary", "desc": "absolute value"},
    "absolute": {
        "numpy": np.abs,
        "sympy": sp.Abs,
        "cat": "elementary",
        "desc": "absolute value (alias)",
    },
    "sign": {"numpy": np.sign, "sympy": sp.sign, "cat": "elementary", "desc": "sign of x"},
    "hypot": {
        "numpy": np.hypot,
        "sympy": sp.Function("hypot"),
        "cat": "elementary",
        "desc": "sqrt(x² + y²)",
    },
    # ── Trigonometric ─────────────────────────────────────────────────────
    "sin": {"numpy": np.sin, "sympy": sp.sin, "cat": "trigonometric", "desc": "sine"},
    "cos": {"numpy": np.cos, "sympy": sp.cos, "cat": "trigonometric", "desc": "cosine"},
    "tan": {"numpy": np.tan, "sympy": sp.tan, "cat": "trigonometric", "desc": "tangent"},
    "arcsin": {
        "numpy": np.arcsin,
        "sympy": sp.asin,
        "cat": "trigonometric",
        "desc": "inverse sine",
    },
    "arccos": {
        "numpy": np.arccos,
        "sympy": sp.acos,
        "cat": "trigonometric",
        "desc": "inverse cosine",
    },
    "arctan": {
        "numpy": np.arctan,
        "sympy": sp.atan,
        "cat": "trigonometric",
        "desc": "inverse tangent",
    },
    "arctan2": {
        "numpy": np.arctan2,
        "sympy": sp.atan2,
        "cat": "trigonometric",
        "desc": "arctan(y / x)",
    },
    "sinh": {"numpy": np.sinh, "sympy": sp.sinh, "cat": "trigonometric", "desc": "hyperbolic sine"},
    "cosh": {
        "numpy": np.cosh,
        "sympy": sp.cosh,
        "cat": "trigonometric",
        "desc": "hyperbolic cosine",
    },
    "tanh": {
        "numpy": np.tanh,
        "sympy": sp.tanh,
        "cat": "trigonometric",
        "desc": "hyperbolic tangent",
    },
    "arcsinh": {
        "numpy": np.arcsinh,
        "sympy": sp.asinh,
        "cat": "trigonometric",
        "desc": "inverse hyperbolic sine",
    },
    "arccosh": {
        "numpy": np.arccosh,
        "sympy": sp.acosh,
        "cat": "trigonometric",
        "desc": "inverse hyperbolic cosine",
    },
    "arctanh": {
        "numpy": np.arctanh,
        "sympy": sp.atanh,
        "cat": "trigonometric",
        "desc": "inverse hyperbolic tangent",
    },
    # ── Rounding ──────────────────────────────────────────────────────────
    "floor": {"numpy": np.floor, "sympy": sp.floor, "cat": "rounding", "desc": "floor(x)"},
    "ceil": {"numpy": np.ceil, "sympy": sp.ceiling, "cat": "rounding", "desc": "ceil(x)"},
    "round": {
        "numpy": np.round,
        "sympy": sp.Function("round"),
        "cat": "rounding",
        "desc": "round(x)",
    },
    # ── Special functions (scipy.special — vectorised in C) ───────────────
    "gamma": {
        "numpy": _sp_special.gamma,
        "sympy": sp.gamma,
        "cat": "special",
        "desc": "Γ(x) — gamma function",
    },
    "lgamma": {
        "numpy": _sp_special.gammaln,
        "sympy": sp.loggamma,
        "cat": "special",
        "desc": "log Γ(x) — log-gamma",
    },
    "erf": {"numpy": _sp_special.erf, "sympy": sp.erf, "cat": "special", "desc": "error function"},
    "erfc": {
        "numpy": _sp_special.erfc,
        "sympy": sp.erfc,
        "cat": "special",
        "desc": "complementary error function",
    },
    "beta": {
        "numpy": _sp_special.beta,
        "sympy": sp.beta,
        "cat": "special",
        "desc": "B(a, b) — beta function",
    },
}

# Constants
_NS_CONSTANTS: Dict[str, float] = {
    "pi": float(np.pi),
    "e": float(np.e),
    "inf": float(np.inf),
    "nan": float(np.nan),
}

# Flat evaluation namespace: bare name AND np.<name> both work
FORMULA_NAMESPACE: Dict[str, Any] = {
    **{name: info["numpy"] for name, info in _NS.items()},
    **{f"np.{name}": info["numpy"] for name, info in _NS.items()},
    **_NS_CONSTANTS,
}

# ---------------------------------------------------------------------------
# Optional JAX-backed numeric namespace — mirrors _NS 1:1 by key, built
# lazily on first use of backend="jax" so jax stays an optional dependency.
# ---------------------------------------------------------------------------

_NS_JAX_CACHE: Optional[Dict[str, Any]] = None


def _build_jax_namespace() -> Dict[str, Any]:
    """Builds (and caches) the jax.numpy / jax.scipy.special equivalent of
    FORMULA_NAMESPACE. Same keys as FORMULA_NAMESPACE — the sympy symbols
    in _SYMPY_LOCALS keep working unchanged for either backend; only the
    numeric callable that lambdify binds to each name changes."""
    global _NS_JAX_CACHE
    if _NS_JAX_CACHE is not None:
        return _NS_JAX_CACHE

    jax = _get_jax()
    jnp = jax.numpy
    jsp = jax.scipy.special

    numpy_to_jax: Dict[str, Any] = {
        # Elementary
        "exp": jnp.exp,
        "exp2": jnp.exp2,
        "expm1": jnp.expm1,
        "log": jnp.log,
        "log2": jnp.log2,
        "log10": jnp.log10,
        "log1p": jnp.log1p,
        "sqrt": jnp.sqrt,
        "cbrt": jnp.cbrt,
        "power": jnp.power,
        "abs": jnp.abs,
        "absolute": jnp.abs,
        "sign": jnp.sign,
        "hypot": jnp.hypot,
        # Trigonometric
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "arcsin": jnp.arcsin,
        "arccos": jnp.arccos,
        "arctan": jnp.arctan,
        "arctan2": jnp.arctan2,
        "sinh": jnp.sinh,
        "cosh": jnp.cosh,
        "tanh": jnp.tanh,
        "arcsinh": jnp.arcsinh,
        "arccosh": jnp.arccosh,
        "arctanh": jnp.arctanh,
        # Rounding — zero gradient a.e., same caveat as the numpy backend
        "floor": jnp.floor,
        "ceil": jnp.ceil,
        "round": jnp.round,
        # Special functions — jax.scipy.special, not scipy.special (that
        # one isn't traceable/differentiable)
        "gamma": jsp.gamma,
        "lgamma": jsp.gammaln,
        "erf": jsp.erf,
        "erfc": jsp.erfc,
        "beta": jsp.beta,
    }
    missing = set(_NS) - set(numpy_to_jax)
    if missing:
        raise RuntimeError(
            f"Internal error: no jax mapping for namespace function(s): {sorted(missing)}"
        )

    ns_jax: Dict[str, Any] = {
        **numpy_to_jax,
        **{f"np.{name}": func for name, func in numpy_to_jax.items()},
        **_NS_CONSTANTS,  # plain Python floats — fine as jax leaves/constants
    }
    _NS_JAX_CACHE = ns_jax
    return ns_jax


# SymPy locals for sympify
_SYMPY_LOCALS: Dict[str, Any] = {name: info["sympy"] for name, info in _NS.items()}

# Regex
_RE_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_RE_IDENTIFIERS = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.]*")
_RE_ANY_FUNC = re.compile(r"\b(?:np\.)?(\w+)\s*\(")
_RE_NP_PREFIX = re.compile(r"\bnp\.(\w+)\s*\(")
_RE_SPECIAL_FUNC = re.compile(r"^SS\w+\([^)]+\)$")


# ---------------------------------------------------------------------------
# Special self-starting function registry
# ---------------------------------------------------------------------------

_SPECIAL_FUNCTIONS: Dict[str, Dict] = {
    "SSexp": {
        "min_args": 3,
        "expr": lambda a: f"exp({a[2]} * {a[0]}) * {a[1]}",
        "exogenous_idx": [0],
        "params_idx": [1, 2],
        "description": "y = a * exp(b * x)",
    },
    "SSlogis": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} / (1 + exp(({a[2]} - {a[0]}) / {a[3]}))",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = Asym / (1 + exp((xmid - x) / scal))  [logistic]",
    },
    "SSmicmen": {
        "min_args": 3,
        "expr": lambda a: f"{a[1]} * {a[0]} / ({a[2]} + {a[0]})",
        "exogenous_idx": [0],
        "params_idx": [1, 2],
        "description": "y = Vm * x / (K + x)  [Michaelis-Menten]",
    },
    "SSgompertz": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} * exp(-{a[2]} * {a[3]} ** {a[0]})",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = a * exp(-b * c^x)  [Gompertz]",
    },
    "SSweibull": {
        "min_args": 5,
        "expr": lambda a: f"{a[1]} - {a[2]} * exp(-exp({a[3]}) * {a[0]} ** {a[4]})",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3, 4],
        "description": "y = a - b * exp(-exp(c) * x^d)  [Weibull]",
    },
    "SSasymp": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} + ({a[2]} - {a[1]}) * exp(-exp({a[3]}) * {a[0]})",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = Asym + (R0 - Asym) * exp(-exp(lrc) * x)  [asymptotic]",
    },
    "SSfpl": {
        "min_args": 5,
        "expr": lambda a: f"{a[1]} + ({a[2]} - {a[1]}) / (1 + exp(({a[3]} - {a[0]}) / {a[4]}))",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3, 4],
        "description": "y = A + (B-A) / (1 + exp((xmid-x)/scal))  [4-param logistic]",
    },
    "SSbiexp": {
        "min_args": 5,
        "expr": lambda a: (
            f"{a[1]} * exp(-exp({a[2]}) * {a[0]}) + {a[3]} * exp(-exp({a[4]}) * {a[0]})"
        ),
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3, 4],
        "description": "y = A1 * exp(-exp(lrc1) * x) + A2 * exp(-exp(lrc2) * x)  [biexponencial]",
    },
    "SSlogis5": {
        "min_args": 6,
        "expr": lambda a: (
            f"{a[1]} + ({a[2]} - {a[1]}) / ((1 + exp(({a[3]} - {a[0]}) / {a[4]})) ** {a[5]})"
        ),
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3, 4, 5],
        "description": "y = A + (B-A) / (1 + exp((xmid-x)/scal))^g  [5-param logistic]",
    },
    "SSlin": {
        "min_args": 3,
        "expr": lambda a: f"{a[1]} + {a[2]} * {a[0]}",
        "exogenous_idx": [0],
        "params_idx": [1, 2],
        "description": "y = a + b * x  [simple linear]",
    },
    "SSquad": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} + {a[2]} * {a[0]} + {a[3]} * {a[0]} ** 2",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = a + b * x + c * x^2  [quadratic]",
    },
    "SSpower": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} * {a[0]} ** {a[2]} + {a[3]}",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = a * x^b + c  [power model with offset]",
    },
    "SShill": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} * {a[0]} ** {a[3]} / ({a[2]} ** {a[3]} + {a[0]} ** {a[3]})",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = Vm * x^n / (K^n + x^n)  [Hill equation]",
    },
    "SSgauss": {
        "min_args": 4,
        "expr": lambda a: f"{a[1]} * exp(-(({a[0]} - {a[2]}) ** 2) / (2 * {a[3]} ** 2))",
        "exogenous_idx": [0],
        "params_idx": [1, 2, 3],
        "description": "y = a * exp(-(x - b)^2 / (2 * c^2))  [Gaussian bell]",
    },
}

# ---------------------------------------------------------------------------
# Suggestions for common unknown functions (shown in FormulaNamespaceError)
# Purpose: better error messages only — NOT a blacklist
# ---------------------------------------------------------------------------
_UNKNOWN_FUNC_SUGGESTIONS: Dict[str, str] = {
    # scipy.optimize
    "curve_fit": "Use the NLS solver in this package.",
    "least_squares": "Use the NLS solver in this package.",
    "minimize": "Use the NLS solver in this package.",
    "fsolve": "Use the NLS solver in this package.",
    # scipy.special — some moved into namespace via mpmath (gamma, erf, erfc, beta)
    "bessel": "Bessel functions are not in the namespace.",
    "jv": "Bessel functions are not in the namespace.",
    "yv": "Bessel functions are not in the namespace.",
    "iv": "Bessel functions are not in the namespace.",
    "kv": "Bessel functions are not in the namespace.",
    "hyp2f1": "Hypergeometric functions are not in the namespace.",
    "expit": "Use '1 / (1 + exp(-x))' instead.",
    "logit": "Use 'log(x / (1 - x))' instead.",
    # scipy.stats
    "norm": "Statistical distributions are not formula functions.",
    "chi2": "Statistical distributions are not formula functions.",
    # scipy.integrate
    "quad": "Integration is not supported inside formulas.",
    "odeint": "ODE solvers are not supported inside formulas.",
    # numpy — not in namespace
    "convolve": "Signal processing functions are not in the namespace.",
    "fft": "FFT is not in the formula namespace.",
    "polyval": "Use an explicit polynomial expression instead.",
}


# ---------------------------------------------------------------------------
# Reserved names — built automatically from the namespace so it never drifts
# ---------------------------------------------------------------------------
# Any symbol that appears in FORMULA_NAMESPACE or as a constant is reserved
# and cannot be used as a parameter name.  This prevents silent bugs like
# writing  y ~ e * exp(x)  where 'e' is Euler's number, not a free parameter.

_RESERVED_NAMES: frozenset = frozenset(list(_NS.keys()) + list(_NS_CONSTANTS.keys()) + ["np"])

# Single compiled alternation over all reserved names (longest-first so e.g.
# 'log10' matches before 'log'), built once at import time instead of
# recompiling one regex per name on every parse.
_RESERVED_PATTERN: re.Pattern = re.compile(
    r"\b("
    + "|".join(re.escape(n) for n in sorted(_RESERVED_NAMES - {"np"}, key=len, reverse=True))
    + r")\b(?!\s*\()"
)

# Formula-size guardrails against pathological / adversarial input.
_MAX_FORMULA_LENGTH = 2000
_MAX_EXPONENT_MAGNITUDE = 1000


# ---------------------------------------------------------------------------
# Module helper
# ---------------------------------------------------------------------------


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# FormulaParser
# ---------------------------------------------------------------------------


class FormulaParser:
    """
    Parses nonlinear regression formulas of the form ``y ~ f(x, params...)``.

    Uses a **closed namespace** for evaluation: only functions in
    ``FORMULA_NAMESPACE`` are available.  Unknown functions raise
    ``FormulaNamespaceError`` at parse time with a clear message.

    Parameters
    ----------
    formula : str
        e.g. ``"y ~ a * exp(b * x)"`` or ``"y ~ a * np.exp(b * x)"``.
    data : pd.DataFrame
        Columns define which symbols are exogenous vs. parameters.
    extra_functions : dict, optional
        User-defined functions to make available in this formula only.
        Each entry maps a name to either:

        - a plain callable  ``{"myfunc": np.sin}``
        - a tuple           ``{"myfunc": (np_callable, sympy_func)}``

        Functions registered here are **instance-scoped** — they do not
        affect other ``FormulaParser`` instances or the global namespace.
        With ``backend="jax"`` the callable must itself be jax-traceable
        (built from ``jax.numpy``/``jax.scipy.special``, no plain numpy).
    backend : {"numpy", "jax"}, default "numpy"
        Numeric evaluation engine. "numpy" is the historical default —
        no extra dependency. "jax" translates the same parsed SymPy
        expression through JAX instead, enabling ``jacobian()`` (autodiff)
        and ``to_jax_callable()`` (JIT-compiled). Requires the optional
        ``jax`` package; only imported when this is set to "jax".

    Examples
    --------
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2.7, 7.4, 20.1]})
    >>> fp = FormulaParser("y ~ a * exp(b * x)", df)
    >>> fp.parameters
    ['a', 'b']

    Using a custom function:

    >>> import numpy as np, sympy as sp
    >>> def softplus(x): return np.log1p(np.exp(x))
    >>> fp2 = FormulaParser(
    ...     "y ~ a * softplus(b * x)",
    ...     df,
    ...     extra_functions={"softplus": (softplus, sp.Function("softplus"))},
    ... )
    >>> fp2.evaluate({"a": 1.0, "b": 0.5})
    array([...])

    Using the JAX backend for autodiff:

    >>> fp3 = FormulaParser("y ~ a * exp(b * x)", df, backend="jax")
    >>> fp3.jacobian({"a": 1.0, "b": 0.5})   # d(pred_i) / d(a, b)
    array([...])
    """

    _VALID_BACKENDS = ("numpy", "jax")

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        extra_functions: Optional[Dict[str, Any]] = None,
        backend: str = "numpy",
    ) -> None:
        if not isinstance(formula, str):
            raise TypeError("The formula must be a string.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if backend not in self._VALID_BACKENDS:
            raise FormulaError(f"backend must be one of {self._VALID_BACKENDS}, got '{backend}'.")

        self.formula_original: str = formula.strip()
        self._data: pd.DataFrame = data
        self._data_columns: set = set(data.columns.tolist())
        self.backend: str = backend

        self.lhs: str = ""
        self.rhs: str = ""
        self.expression: str = ""
        self.exogenous: List[str] = []
        self.params: List[str] = []
        self._is_special: bool = False
        self._func_name_special: Optional[str] = None
        self._sympy_expr: Optional[sp.Expr] = None
        self._eval_func: Optional[Any] = None
        self._raw_func: Optional[Any] = None  # positional, unwrapped
        # — only set for backend="jax",
        # used by jacobian()/to_jax_callable()
        self._jac_fn: Optional[Any] = None  # jit-compiled jax.jacfwd, backend="jax" only

        # Build instance-scoped namespaces (copies of globals + user extensions)
        base_ns: Dict[str, Any] = _build_jax_namespace() if backend == "jax" else FORMULA_NAMESPACE
        self._ns_eval: Dict[str, Any] = dict(base_ns)  # for evaluate()
        self._ns_sympy: Dict[str, Any] = dict(_SYMPY_LOCALS)  # for sympify()
        self._extra_func_names: set = set()

        if extra_functions:
            self._register_extra_functions(extra_functions)

        self._validate_syntax()
        self._parse_formula()
        self._validate_parameter_names()
        self._eval_func = self._build_lambdify()

    # ------------------------------------------------------------------
    # Extra-function registration (instance-scoped)
    # ------------------------------------------------------------------

    def _register_extra_functions(self, extra_functions: Dict[str, Any]) -> None:
        """
        Validates and registers user-supplied functions into the
        instance-scoped namespaces.

        Accepts two formats per entry:

        - ``"name": callable``
            The callable is used for numerical evaluation.
            SymPy will treat the function as an opaque ``Function("name")``.

        - ``"name": (numpy_callable, sympy_func)``
            Provides both a numerical and a symbolic representation.
            Useful when you want SymPy to differentiate the expression.

        Raises
        ------
        TypeError  if a value is neither callable nor a (callable, sympy) tuple.
        ValueError if a name shadows a built-in namespace function.
        """
        for name, value in extra_functions.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise FormulaError(
                    f"extra_functions key '{name}' is not a valid Python identifier."
                )
            if name in self._ns_eval:
                raise FormulaError(
                    f"extra_functions name '{name}' shadows a built-in namespace "
                    f"function. Choose a different name."
                )

            if callable(value):
                np_func = value
                sympy_func = sp.Function(name)
            elif isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
                np_func, sympy_func = value
            else:
                raise TypeError(
                    f"extra_functions['{name}'] must be a callable or a "
                    f"(numpy_callable, sympy_func) tuple; got {type(value).__name__}."
                )

            self._ns_eval[name] = np_func
            self._ns_sympy[name] = sympy_func
            self._extra_func_names.add(name)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_syntax(self) -> None:
        if "~" not in self.formula_original:
            raise FormulaError("The formula must contain '~'.")

        parts = self.formula_original.split("~", maxsplit=1)
        lhs = parts[0].strip()
        rhs = parts[1].strip() if len(parts) > 1 else ""

        # ── LHS ──────────────────────────────────────────────────────────
        if not lhs:
            raise FormulaError("The left-hand side of '~' cannot be empty.")
        if "." in lhs:
            raise FormulaError(
                f"Invalid response variable: '{lhs}'. "
                "Dots are not allowed in variable names. "
                f"Use '{lhs.replace('.', '_')}' instead."
            )
        if not _RE_IDENTIFIER.match(lhs):
            raise FormulaError(
                f"Invalid response variable: '{lhs}'. Use only letters, digits, and underscores."
            )
        if any(op in lhs for op in ("+", "-", "*", "/", "(", ")", "**", "^")):
            raise FormulaError(
                f"Transformations are not allowed on the LHS: '{lhs}'. "
                "Pre-transform the column and use the new column name."
            )

        # ── RHS ──────────────────────────────────────────────────────────
        if not rhs:
            raise FormulaError("The right-hand side of '~' cannot be empty.")

        self._validate_rhs_names(rhs)

        # Run reserved-name check HERE (before sympify) so that
        # 'nan' is caught even though SymPy would swallow it entirely.
        self._validate_reserved_in_string(rhs)

        # Hard length guard against pathological input before any parsing
        # (sympify time grows badly with deeply nested/huge expressions).
        if len(rhs) > _MAX_FORMULA_LENGTH:
            raise FormulaError(
                f"Expression too long ({len(rhs)} chars, max {_MAX_FORMULA_LENGTH})."
            )
        self._validate_exponent_magnitude(rhs)

        # Namespace check for ALL formulas, including SS*. SS* arguments
        # used to be exempt from this check (only a dot-check ran in
        # _parse_special_function), which meant any bare call like
        # 'foo(...)' inside an SS* argument reached sympify() unvalidated.
        self._validate_namespace(rhs)

    def _validate_exponent_magnitude(self, expression: str) -> None:
        """
        Rejects literal exponents above ``_MAX_EXPONENT_MAGNITUDE``.

        SymPy will attempt to symbolically expand/represent expressions
        like ``a**99999999999``, which can hang or exhaust memory even
        though the input contains no disallowed function calls — this is
        a plain denial-of-service vector, not a namespace violation, so
        ``_validate_namespace`` doesn't catch it.
        """
        for m in re.finditer(r"\*\*\s*(-?\d+(?:\.\d+)?)", expression):
            if abs(float(m.group(1))) > _MAX_EXPONENT_MAGNITUDE:
                raise FormulaError(
                    f"Exponent magnitude too large: '{m.group(0).strip()}' "
                    f"(max {_MAX_EXPONENT_MAGNITUDE})."
                )

    def _validate_rhs_names(self, expression: str) -> None:
        """Rejects identifiers with dots, except decimal literals and np.*."""
        for identifier in _RE_IDENTIFIERS.findall(expression):
            if "." not in identifier:
                continue
            parts = identifier.split(".")
            if all(p.isdigit() for p in parts):
                continue  # decimal literal e.g. 3.14
            if identifier.startswith("np."):
                continue  # handled by namespace check
            raise FormulaError(
                f"Invalid name in formula: '{identifier}'. "
                "Dots are not allowed in variable/parameter names. "
                f"Suggestion: use '{identifier.replace('.', '_')}' instead."
            )

    def _validate_reserved_in_string(self, rhs: str) -> None:
        """
        Scans the raw RHS string for reserved names used as bare
        identifiers.  Called once from ``_validate_syntax`` *before* SymPy
        processes the expression, so that names like ``nan`` are caught
        even when SymPy would silently discard them.

        Uses a single precompiled alternation (``_RESERVED_PATTERN``)
        instead of recompiling one regex per reserved name on every call.
        """
        hints_map = {
            "e": "'e' = Euler's number 2.718… — use 'a', 'k', or another name",
            "pi": "'pi' = 3.14159… — use 'a' or another name",
            "inf": "'inf' is not a finite parameter — choose a different name",
            "nan": "'nan' is not a valid parameter value — choose a different name",
        }
        rhs_stripped = _RE_NP_PREFIX.sub("__NPFUNC__(", rhs)
        data_cols = self._data_columns  # available even at syntax-check time

        confirmed = sorted(
            {m for m in _RESERVED_PATTERN.findall(rhs_stripped) if m not in data_cols}
        )

        if confirmed:
            detail = "; ".join(
                f"'{c}' → {hints_map.get(c, 'choose a different name')}" for c in confirmed
            )
            raise FormulaError(
                f"Parameter name(s) clash with reserved namespace names: {confirmed}.\n"
                f"  {detail}.\n"
                f"  Call FormulaParser.list_namespace() to see all reserved names."
            )

    def _validate_namespace(self, expression: str) -> None:
        """
        Raises FormulaNamespaceError for any function call not in the
        instance namespace (global FORMULA_NAMESPACE + extra_functions).
        SS* functions and data/parameter identifiers are ignored.
        """
        for m in _RE_ANY_FUNC.finditer(expression):
            bare_name = m.group(1)
            full_call = m.group(0).rstrip("( ")

            if bare_name.startswith("SS"):
                continue  # validated in _parse_special_function

            in_ns = (
                bare_name in self._ns_eval
                or full_call in self._ns_eval
                or bare_name in self._extra_func_names
            )
            if not in_ns:
                suggestion = _UNKNOWN_FUNC_SUGGESTIONS.get(bare_name, "")
                raise FormulaNamespaceError(bare_name, suggestion)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_formula(self) -> None:
        parts = [p.strip() for p in self.formula_original.split("~", maxsplit=1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise FormulaError("The formula must have exactly one '~' with non-empty sides.")

        self.lhs, self.rhs = parts

        if self.lhs not in self._data_columns:
            raise FormulaError(
                f"Response variable '{self.lhs}' not found in the DataFrame. "
                f"Available columns: {sorted(self._data_columns)}"
            )

        self._is_special = bool(_RE_SPECIAL_FUNC.match(self.rhs))
        if self._is_special:
            self._parse_special_function()
        else:
            self._parse_regular_formula()

    def _parse_special_function(self) -> None:
        match = re.match(r"^(\w+)\(([^)]+)\)$", self.rhs)
        if not match:
            raise FormulaError(f"Could not parse special function: '{self.rhs}'.")

        func_name = match.group(1)
        args = [a.strip() for a in match.group(2).split(",")]

        for arg in args:
            # Each SS* argument must be a plain identifier (column/parameter
            # name) or a numeric literal — nothing else. Previously only a
            # dot-check ran here, which let arbitrary non-dotted text (e.g.
            # stray function-call fragments) through to be spliced verbatim
            # into the expression template and handed to sympify().
            if not (_RE_IDENTIFIER.match(arg) or _is_numeric(arg)):
                raise FormulaError(
                    f"Invalid argument '{arg}' for '{match.group(1)}': "
                    "SS* arguments must be plain identifiers or numeric literals."
                )

        if func_name not in _SPECIAL_FUNCTIONS:
            supported = ", ".join(sorted(_SPECIAL_FUNCTIONS))
            raise FormulaError(
                f"Special function '{func_name}' is not supported. "
                f"Supported SS* functions: {supported}"
            )

        info = _SPECIAL_FUNCTIONS[func_name]
        if len(args) < info["min_args"]:
            raise FormulaError(
                f"'{func_name}' requires at least {info['min_args']} arguments; got {len(args)}."
            )

        self._func_name_special = func_name
        self.expression = info["expr"](args)
        self.exogenous = [args[i] for i in info["exogenous_idx"]]
        self.params = [args[i] for i in info["params_idx"]]

        if self.lhs in self.exogenous:
            raise FormulaError(f"Response variable '{self.lhs}' cannot appear on the RHS.")

        self._validate_exogenous_vars()
        self._build_sympy_expr(self.expression)

    def _parse_regular_formula(self) -> None:
        # Strip np. prefix before handing to SymPy
        clean_rhs = _RE_NP_PREFIX.sub(lambda m: m.group(1) + "(", self.rhs)
        self.expression = self.rhs
        self._build_sympy_expr(clean_rhs)

        # `free_symbols` is a Python set — iterating it directly makes the
        # resulting order depend on Python's hash-randomized set iteration
        # (different PYTHONHASHSEED per process), so the exact same formula
        # string could produce self.params in a different order between
        # runs. Order deterministically by first appearance in the formula
        # text instead (also matches how SS* functions preserve the
        # user-written argument order).
        expr = self._sympy_expr
        free_names = {str(s) for s in expr.free_symbols} if expr is not None else set()
        all_names: List[str] = []
        for tok in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", self.rhs):
            name = tok.group(0)
            if name in free_names and name not in all_names:
                all_names.append(name)

        self.exogenous = []
        self.params = []
        for name in all_names:
            if name in self._data_columns:
                self.exogenous.append(name)
            else:
                self.params.append(name)

        if self.lhs in self.exogenous:
            raise FormulaError(f"Response variable '{self.lhs}' cannot appear on the RHS.")
        self._validate_exogenous_vars()
        self._warn_on_likely_column_typos()

    def _warn_on_likely_column_typos(self) -> None:
        """
        Any free symbol not matching a DataFrame column is silently
        treated as a parameter to estimate — a typo'd column name (e.g.
        'xx' instead of 'x') becomes a phantom parameter instead of a
        clear error, and only surfaces later as a confusing shape
        mismatch during fitting. Warn (don't fail — a param genuinely
        named close to a column is legal) when a parameter name is a
        near-miss for an existing column.
        """
        import difflib

        for p in self.params:
            close = difflib.get_close_matches(p, self._data_columns, n=1, cutoff=0.6)
            if close:
                warnings.warn(
                    f"Parameter '{p}' will be estimated as a free parameter, but it "
                    f"closely resembles DataFrame column '{close[0]}'. If this was "
                    f"meant to reference the column, check for a typo.",
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # SymPy
    # ------------------------------------------------------------------

    def _build_sympy_expr(self, expression: str) -> None:
        try:
            self._sympy_expr = sp.sympify(expression, locals=self._ns_sympy)
        except Exception as exc:
            raise FormulaError(
                f"Could not parse expression with SymPy: '{expression}'\nDetails: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _validate_exogenous_vars(self) -> None:
        if not self.exogenous:
            raise FormulaError(
                "No exogenous variables found in the formula. "
                "Check that column names match the DataFrame."
            )
        missing = [v for v in self.exogenous if v not in self._data_columns]
        if missing:
            raise DataValidationError(
                f"Exogenous variable(s) not found in DataFrame: {missing}. "
                f"Available: {sorted(self._data_columns)}"
            )

    def _validate_parameter_names(self) -> None:
        """
        Second-pass reserved-name check, driven by the already-extracted
        ``self.params`` list instead of re-scanning the raw RHS string
        (the full-string scan already ran once in ``_validate_syntax`` via
        ``_validate_reserved_in_string``). O(n_params) instead of a second
        O(len(rhs)) regex pass.
        """
        clashes = sorted(p for p in self.params if p in _RESERVED_NAMES)
        if clashes:
            raise FormulaError(
                f"Parameter name(s) clash with reserved namespace names: {clashes}.\n"
                f"  Call FormulaParser.list_namespace() to see all reserved names."
            )

    def _build_lambdify(self) -> Any:
        """
        Builds and caches a callable from the SymPy expression, backed by
        either numpy or jax depending on ``self.backend``.

        Uses the instance-scoped ``_ns_eval`` so that extra_functions
        registered at construction time are included automatically.
        """
        exog_syms = [sp.Symbol(v) for v in self.exogenous]
        param_syms = [sp.Symbol(p) for p in self.params]
        all_syms = exog_syms + param_syms

        exog_names = list(self.exogenous)
        param_names = list(self.params)

        if self.backend == "jax":
            return self._build_lambdify_jax(all_syms, exog_names, param_names)
        return self._build_lambdify_numpy(all_syms, exog_names, param_names)

    def _build_lambdify_numpy(self, all_syms, exog_names, param_names) -> Any:
        # Instance namespace first (includes extra_functions), then numpy fallback
        lm_modules = [self._ns_eval, "numpy"]
        raw_func = sp.lambdify(all_syms, self.sympy_expression, modules=lm_modules)

        def _caller(*exog_arrays, **param_scalars):
            param_vals = [float(param_scalars[p]) for p in param_names]
            return np.asarray(raw_func(*exog_arrays, *param_vals), dtype=float)

        _caller.__doc__ = (
            f"Compiled evaluator (numpy) for '{self.formula_original}'.\n"
            f"Positional args : {exog_names}\n"
            f"Keyword args    : {param_names}"
        )
        return _caller

    def _build_lambdify_jax(self, all_syms, exog_names, param_names) -> Any:
        jax = _get_jax()
        jnp = jax.numpy

        # Instance namespace first (includes extra_functions), then jax fallback
        lm_modules = [self._ns_eval, "jax"]
        raw_func = sp.lambdify(all_syms, self.sympy_expression, modules=lm_modules)
        # Kept unwrapped/positional for jacobian() and to_jax_callable() — no
        # float()/np.asarray() casts here, those would break jax's tracing.
        self._raw_func = raw_func

        @jax.jit
        def _jit_func(exog_arrays, param_vals):
            return raw_func(*exog_arrays, *param_vals)

        # Stored as an attribute (not just a local variable) so other
        # consumers (e.g. nls._build_residual_and_jac, via
        # predictions_fn()) can reuse this same jit-compiled function
        # instead of compiling an equivalent one on their own.
        self._pred_fn = _jit_func

        # Jacobian of predictions w.r.t. params, compiled ONCE here instead
        # of inside jacobian(): jax.jacfwd(f) builds a *new* function object
        # every time it's called, so calling it fresh on each jacobian()
        # invocation forced JAX to retrace/recompile every single time —
        # exactly the case that matters most (an optimiser calling this
        # every iteration). jit(jacfwd(...)) compiled once here is reused
        # across calls as long as input shapes/dtypes match.
        @jax.jit
        def _jit_jacfwd(exog_arrays, param_vals):
            def _f(pv):
                return raw_func(*exog_arrays, *[pv[i] for i in range(len(param_vals))])

            return jax.jacfwd(_f)(param_vals)

        self._jac_fn = _jit_jacfwd

        def _caller(*exog_arrays, **param_scalars):
            exog_jax = tuple(jnp.asarray(a) for a in exog_arrays)
            param_jax = tuple(jnp.asarray(param_scalars[p]) for p in param_names)
            return np.asarray(_jit_func(exog_jax, param_jax))

        _caller.__doc__ = (
            f"Compiled evaluator (jax, jit-compiled) for '{self.formula_original}'.\n"
            f"Positional args : {exog_names}\n"
            f"Keyword args    : {param_names}\n"
            f"See also: FormulaParser.jacobian(), FormulaParser.to_jax_callable()."
        )
        return _caller

    def _require_jax_backend(self, method_name: str) -> Any:
        if self.backend != "jax":
            raise FormulaError(
                f"{method_name}() requires backend='jax' "
                f"(this instance was built with backend='{self.backend}')."
            )
        return _get_jax()

    def jacobian(
        self,
        param_values: Dict[str, float],
        data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Jacobian of the predictions w.r.t. the parameters, via JAX autodiff.
        Requires ``backend="jax"``.

        Parameters
        ----------
        param_values : dict  ``{param_name: value}`` — evaluation point.
        data : DataFrame, optional — overrides the dataset used at construction.

        Returns
        -------
        np.ndarray of shape (n_obs, n_params): entry [i, j] is
        d(prediction_i) / d(parameter_j).
        """
        jax = self._require_jax_backend("jacobian")
        jnp = jax.numpy

        missing = [p for p in self.parameters if p not in param_values]
        if missing:
            raise DataValidationError(f"Missing parameter value(s): {missing}")
        extra = set(param_values) - set(self.parameters)
        if extra:
            raise DataValidationError(
                f"Unexpected parameter name(s): {sorted(extra)}. Expected: {self.parameters}"
            )

        df = data if data is not None else self._data
        exog = tuple(jnp.asarray(df[col].values) for col in self.exogenous)
        param_names = list(self.parameters)

        p0 = jnp.array([float(param_values[p]) for p in param_names])
        jac_fn = self._jac_fn
        if jac_fn is None:
            raise RuntimeError("Jacobian is not available (internal error)")
        try:
            jac = jac_fn(exog, p0)
        except Exception as exc:
            raise RuntimeError(
                f"Error computing jacobian for '{self.formula_original}' "
                f"with params={param_values}: {exc}"
            ) from exc
        return np.asarray(jac)

    def predictions_fn(self) -> Any:
        """
        Returns the jit-compiled predictions function, exactly as it was
        built once in the constructor (``_build_lambdify_jax``). Requires
        ``backend="jax"``.

        Signature: ``f(exog_tuple, param_array) -> jnp.ndarray``, where
        ``exog_tuple`` is a tuple of jax arrays (one per exogenous
        variable, in the order of ``self.exogenous``) and ``param_array``
        is a 1-D jax array in the order of ``self.parameters``.

        Meant for internal consumers (like ``nls``) that need to evaluate
        the model inside an optimization loop without paying the cost of
        compiling an equivalent jax function on their own — reuses the
        same jit-compiled object that ``evaluate()`` and ``to_callable()``
        use internally.
        """
        self._require_jax_backend("predictions_fn")
        return self._pred_fn

    def predictions_jacobian_fn(self) -> Any:
        """
        Returns the jacobian of the predictions with respect to the
        parameters, already jit-compiled (``jax.jacfwd``) once in the
        constructor — the same object ``jacobian()`` uses internally.
        Requires ``backend="jax"``.

        Signature: ``f(exog_tuple, param_array) -> jnp.ndarray`` of shape
        ``(n_obs, n_params)``, with the same conventions as
        ``predictions_fn()``.

        Meant for internal consumers that need this jacobian (e.g. ``nls``
        for the analytical jacobian passed to ``least_squares``) without
        re-deriving/recompiling an equivalent one with
        ``jax.jacfwd(...)`` on their own.
        """
        self._require_jax_backend("predictions_jacobian_fn")
        return self._jac_fn

    def to_jax_callable(self) -> Any:
        """
        Returns a JIT-compiled, jax-native callable ``f(X, params) -> jnp.ndarray``
        for composing into a larger JAX model (e.g. a log-posterior). Requires
        ``backend="jax"``. Unlike ``to_callable()``, inputs/outputs stay as
        jax arrays (no conversion to/from numpy), so it's differentiable and
        composable with ``jax.grad``/``jax.jit`` in the caller.

        ``X`` has the same shape convention as ``to_callable()``: 1-D when
        there's a single exogenous variable, or shape (n_obs, n_exog)
        otherwise. ``params`` is a 1-D array in ``self.parameters`` order.
        """
        jax = self._require_jax_backend("to_jax_callable")
        jnp = jax.numpy
        exog_names = list(self.exogenous)
        param_names = list(self.parameters)
        raw_func = self._raw_func

        @jax.jit
        def _f(X, params):
            X = jnp.asarray(X)
            if len(exog_names) == 1:
                exog_arrays = [X.ravel()]
            else:
                exog_arrays = [X[:, i] for i in range(len(exog_names))]
            param_vals = [params[i] for i in range(len(param_names))]
            return raw_func(*exog_arrays, *param_vals)

        return _f

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def endogenous(self) -> str:
        """Response variable — LHS of ~."""
        return self.lhs

    @property
    def exogenous_vars(self) -> List[str]:
        """Exogenous (data) variables found in the formula."""
        return list(self.exogenous)

    @property
    def parameters(self) -> List[str]:
        """Parameters to be estimated."""
        return list(self.params)

    @property
    def sympy_expression(self) -> sp.Expr:
        """Parsed SymPy expression object."""
        if self._sympy_expr is None:
            raise AttributeError("SymPy expression not built — this is a bug.")
        return self._sympy_expr

    @property
    def expression_string(self) -> str:
        """Original RHS expression string as provided by the user."""
        return self.expression

    @property
    def func_name_special(self) -> Optional[str]:
        """SS* function name, or None for regular formulas."""
        return self._func_name_special

    @property
    def is_special(self) -> bool:
        """True if the formula uses an SS* self-starting function."""
        return self._is_special

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_info(self) -> Dict:
        """Returns a summary dict of the parsed formula."""
        return {
            "endogenous": self.lhs,
            "exogenous": self.exogenous_vars,
            "parameters": self.parameters,
            "expression": self.expression_string,
            "is_special_function": self._is_special,
            "func_name_special": self._func_name_special,
            "sympy_expression": str(self.sympy_expression),
        }

    def evaluate(
        self,
        param_values: Dict[str, float],
        data: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Numerically evaluates the formula for given parameter values.

        Uses the cached ``lambdify`` callable built at parse time — safe to
        call thousands of times inside an optimiser loop with no overhead from
        expression recompilation.

        Parameters
        ----------
        param_values : dict  ``{param_name: value}``
        data : DataFrame, optional
            Overrides the dataset used at construction (for prediction on
            new observations).

        Returns
        -------
        np.ndarray of predicted values.
        """
        missing = [p for p in self.parameters if p not in param_values]
        if missing:
            raise DataValidationError(f"Missing parameter value(s): {missing}")

        extra = set(param_values) - set(self.parameters)
        if extra:
            raise DataValidationError(
                f"Unexpected parameter name(s): {sorted(extra)}. Expected: {self.parameters}"
            )

        df = data if data is not None else self._data
        exog = [df[col].values for col in self.exogenous]

        try:
            eval_func = self._eval_func
            if eval_func is None:
                raise RuntimeError("Model function is not available (internal error)")
            return eval_func(*exog, **param_values)
        except Exception as exc:
            raise RuntimeError(
                f"Error evaluating '{self.formula_original}' with params={param_values}: {exc}"
            ) from exc

    def to_callable(self) -> Any:
        """
        Returns a function compatible with ``scipy.optimize.curve_fit``.

        The returned callable has signature ``f(X, *params) -> np.ndarray``
        where ``X`` can be:
        - 1-D array when there is only one exogenous variable
        - 2-D array with shape (n_obs, n_exog) for multiple variables

        Examples
        --------
        >>> # Single exogenous variable
        >>> f = fp.to_callable()
        >>> popt, _ = curve_fit(f, df['x'].values, df['y'].values, p0=[1,1])

        >>> # Multiple exogenous variables
        >>> X = np.column_stack([df['x1'].values, df['x2'].values])
        >>> popt, _ = curve_fit(f, X, df['y'].values, p0=[1,1,1])
        """
        exog_names = list(self.exogenous)
        param_names = list(self.parameters)
        eval_func = self._eval_func

        def _curve_fit_callable(X, *params):
            # Convert X to an array if needed
            X = np.asarray(X)

            # Decide how to split X based on the number of exogenous variables
            if len(exog_names) == 1:
                # Single variable: X may be 1D
                exog_arrays = [X.ravel()]
            else:
                # Multiple variables: X must be 2D with shape (n_obs, n_exog)
                if X.ndim == 1:
                    raise DataValidationError(
                        f"Expected 2D array with {len(exog_names)} columns, "
                        f"got 1D array. Use X.reshape(-1, {len(exog_names)}) or "
                        f"np.column_stack([x1, x2, ...])"
                    )
                if X.shape[1] != len(exog_names):
                    raise DataValidationError(
                        f"Expected {len(exog_names)} exogenous variables, "
                        f"but X has {X.shape[1]} columns. Variables: {exog_names}"
                    )
                # Split by column
                exog_arrays = [X[:, i] for i in range(len(exog_names))]

            # Build the parameter dict
            if len(params) != len(param_names):
                raise DataValidationError(
                    f"Expected {len(param_names)} parameters ({param_names}), got {len(params)}"
                )

            kw = dict(zip(param_names, params))
            return eval_func(*exog_arrays, **kw)

        # Enriched docstring
        _curve_fit_callable.__doc__ = (
            f"curve_fit-compatible callable for '{self.formula_original}'\n"
            f"Exogenous variables ({len(exog_names)}): {exog_names}\n"
            f"Parameters ({len(param_names)}): {param_names}\n"
            f"\nUsage:\n"
            f"  # Single variable\n"
            f"  popt, _ = curve_fit(f, x_data, y_data, p0=initial_params)\n"
            f"  # Multiple variables\n"
            f"  X = np.column_stack([x1_data, x2_data])\n"
            f"  popt, _ = curve_fit(f, X, y_data, p0=initial_params)"
        )

        return _curve_fit_callable

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """
        Makes ``FormulaParser`` instances picklable (``pickle.dump``,
        ``copy.deepcopy``, ``joblib.dump``, multiprocessing, etc).

        ``_eval_func``/``_raw_func``/``_pred_fn``/``_jac_fn`` are closures
        built inside instance methods (``_build_lambdify*``) — plain
        ``pickle`` cannot serialize a locally-defined function, so they
        are dropped here and rebuilt by ``__setstate__`` on unpickling via
        ``_build_lambdify()``. This is cheap for ``backend="numpy"``; for
        ``backend="jax"`` it re-pays JAX's one-time jit-compilation cost.
        """
        state = self.__dict__.copy()
        for key in ("_eval_func", "_raw_func", "_pred_fn", "_jac_fn"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._raw_func = None
        self._pred_fn = None
        self._jac_fn = None
        self._eval_func = self._build_lambdify()

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def list_namespace() -> None:
        """Prints all functions and constants available in the formula namespace."""
        entries = sorted(_NS.items(), key=lambda kv: (kv[1]["cat"], kv[0]))
        print("Formula namespace — available functions")
        print("=" * 56)
        for cat, group in groupby(entries, key=lambda kv: kv[1]["cat"]):
            print(f"\n  [{cat}]")
            for name, info in group:
                np_alias = f"  /  np.{name}"
                print(f"    {name:12s}{np_alias:18s}  {info['desc']}")
        print("\n  [constants]")
        for name, val in _NS_CONSTANTS.items():
            print(f"    {name:12s}  {val}")
        print()

    @staticmethod
    def list_special_functions() -> None:
        """Prints all supported SS* self-starting functions."""
        print("Supported self-starting (SS*) functions")
        print("=" * 50)
        for name, info in _SPECIAL_FUNCTIONS.items():
            print(f"  {name:15s}  {info['description']}")
        print()

    @staticmethod
    def register_special_function(
        name: str,
        expr: Any,
        exogenous_idx: List[int],
        params_idx: List[int],
        min_args: Optional[int] = None,
        description: str = "",
        overwrite: bool = False,
    ) -> None:
        """
        Registers a new ``SS*`` self-starting model, so it can be used in
        a formula the same way as the built-in ones (``SSlogis``,
        ``SSasymp``, etc — see ``list_special_functions()``).

        This makes the ``SS*`` catalog extensible without modifying this
        module: any project can add its own self-starting models at
        import time.

        Parameters
        ----------
        name : str
            The function name as it will appear in formulas, e.g.
            ``"SSmymodel"``. Must start with ``"SS"`` (that's exactly what
            the formula parser's self-starting-function detector matches
            on) and be a valid Python identifier.
        expr : callable
            ``expr(args) -> str``, where ``args`` is the list of argument
            names as written in the formula (in order — exogenous
            variables and parameters interleaved however the model needs
            them). Must return a valid expression string in terms of
            those names, e.g. for ``SSlinear``:
            ``lambda a: f"{a[1]} + {a[2]} * {a[0]}"``.
        exogenous_idx : list[int]
            Positions (into ``args``) that are exogenous (data) variables.
        params_idx : list[int]
            Positions (into ``args``) that are parameters to estimate.
        min_args : int, optional
            Minimum number of arguments the formula call must supply.
            Defaults to ``max(exogenous_idx + params_idx) + 1``.
        description : str, optional
            One-line description, shown by ``list_special_functions()``.
        overwrite : bool, default False
            If a function with this ``name`` is already registered (a
            built-in one or one registered earlier), raises
            ``FormulaError`` unless ``overwrite=True``.

        Note
        ----
        This only makes the formula *usable* — with a fallback initial
        value of 1.0 for every parameter, same as any formula with no
        special initialization heuristic. To give it a proper initial
        value guess (equivalent to what R calls a "self-starting"
        function), also call
        ``Initializer.register_initializer(name, init_fn)``.

        Example
        -------
        >>> FormulaParser.register_special_function(
        ...     "SSmymodel",
        ...     expr=lambda a: f"{a[1]} * {a[0]} ** {a[2]}",
        ...     exogenous_idx=[0],
        ...     params_idx=[1, 2],
        ...     description="y = a * x^b  [power model, no offset]",
        ... )
        >>> model = nls("y ~ SSmymodel(x, a, b)", df)
        """
        if not isinstance(name, str) or not name.isidentifier():
            raise FormulaError(f"'{name}' is not a valid Python identifier.")
        if not name.startswith("SS"):
            raise FormulaError(
                f"'{name}' must start with 'SS' — that's what the formula "
                "parser's self-starting-function detector matches on "
                "(a formula like 'y ~ SSfoo(...)' is only recognized as a "
                "special function when the name starts with 'SS')."
            )
        if not _RE_SPECIAL_FUNC.match(f"{name}(x)"):
            raise FormulaError(
                f"'{name}' does not match the required pattern for "
                "self-starting function names (letters/digits/underscores "
                "only after 'SS')."
            )
        if name in _SPECIAL_FUNCTIONS and not overwrite:
            raise FormulaError(
                f"'{name}' is already registered as a special function. "
                "Pass overwrite=True to replace it."
            )
        if not callable(expr):
            raise FormulaError("expr must be a callable: expr(args) -> str.")

        all_idx = list(exogenous_idx) + list(params_idx)
        if not all_idx:
            raise FormulaError("exogenous_idx and params_idx cannot both be empty.")
        if len(set(all_idx)) != len(all_idx):
            raise FormulaError(
                "exogenous_idx and params_idx must not overlap: "
                f"exogenous_idx={exogenous_idx}, params_idx={params_idx}."
            )
        resolved_min_args = min_args if min_args is not None else max(all_idx) + 1
        if resolved_min_args <= max(all_idx):
            raise FormulaError(
                f"min_args ({resolved_min_args}) must be greater than the "
                f"largest index used in exogenous_idx/params_idx ({max(all_idx)})."
            )

        _SPECIAL_FUNCTIONS[name] = {
            "min_args": resolved_min_args,
            "expr": expr,
            "exogenous_idx": list(exogenous_idx),
            "params_idx": list(params_idx),
            "description": description or f"{name} (user-registered)",
        }
