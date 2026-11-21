"""R/nlme-style formula parsing and design-matrix construction.

Design matrices are built with `formulaic` (treatment contrasts, interactions,
``I()``, ``poly()``, etc.); this module adapts R syntax (``^`` inside ``I()``,
grouping with ``|``, nested ``a/b``) and handles missing values and factors.

The helpers here are shared by every high-level model in
:mod:`statcraft.nlme.models`, :mod:`statcraft.nlme.crossed` and
:mod:`statcraft.nlme.glmm`:

* :func:`parse_form` splits ``"lhs ~ rhs | g1/g2"`` into its parts;
* :func:`eval_expr` evaluates numeric expressions (with R's ``^`` as
  exponentiation);
* :func:`design` / :func:`design_spec` build the fixed-effects design matrix,
  the latter also keeping formulaic's model specification so the matrix can be
  rebuilt on new data with :func:`design_new`;
* :func:`factor_codes` / :func:`nested_codes` give R-compatible integer codes
  for factors and nested grouping chains;
* :func:`collect_columns` / :func:`handle_na` implement ``na.omit`` /
  ``na.fail`` over the columns actually used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # formulaic is required; raise a clear error if it is missing
    import formulaic
except ImportError as e:  # pragma: no cover
    raise ImportError("statcraft.nlme requires 'formulaic' (pip install formulaic)") from e

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_FUNCS = {n: getattr(np, n) for n in ("exp", "log", "log10", "log2", "sqrt", "sin", "cos", "tan", "tanh", "sinh", "cosh",
                                       "arctan", "abs", "floor", "ceil", "maximum", "minimum", "where")}
_FUNCS.update({"pi": np.pi, "np": np, "ifelse": np.where, "pnorm": None})
del _FUNCS["pnorm"]


@dataclass
class FormParts:
    """Parsed parts of an R-style formula.

    Attributes
    ----------
    lhs:
        Left-hand side (the response), or ``None`` when the formula has no
        response.
    rhs:
        Right-hand side (the model terms), ``"1"`` when empty.
    groups:
        Grouping variables from the part after ``|``; ``["g1", "g2"]`` for
        ``"g1/g2"``.
    """

    lhs: str | None
    rhs: str
    groups: list[str] = field(default_factory=list)   # ["g1", "g2"] for g1/g2


def r_to_py(expr: str) -> str:
    """Translate `^` to `**` (in R `^` is exponentiation; in Python/formulaic it would be XOR)."""
    return expr.replace("^", "**")


def _fix_I(expr: str) -> str:
    """Translate `^` to `**` only inside I(...) (formulaic treats `^` outside I() as interaction power)."""
    out, i = [], 0
    while i < len(expr):
        j = expr.find("I(", i)
        if j < 0 or (j > 0 and (expr[j - 1].isalnum() or expr[j - 1] == "_")):
            if j < 0:
                out.append(expr[i:]); break
            out.append(expr[i:j + 2]); i = j + 2; continue
        out.append(expr[i:j + 2]); k, depth = j + 2, 1
        while k < len(expr) and depth:
            depth += (expr[k] == "(") - (expr[k] == ")"); k += 1
        out.append(expr[j + 2:k].replace("^", "**")); i = k
    return "".join(out)


def parse_form(form: str) -> FormParts:
    """Parse ``'lhs ~ rhs | g1/g2'`` into :class:`FormParts`.

    Formulas without a left-hand side (``'~ x | g'``) and without a tilde
    (``'x | g'``) are accepted.

    Parameters
    ----------
    form:
        R-style formula string.

    Returns
    -------
    FormParts
    """
    s = form.strip()
    lhs = None
    if "~" in s:
        l, s = s.split("~", 1)
        lhs = l.strip() or None
    rhs, groups = s, []
    if "|" in s:
        rhs, g = s.rsplit("|", 1)
        groups = [t.strip() for t in g.split("/") if t.strip()]
    return FormParts(lhs, rhs.strip() or "1", groups)


def names_in(expr: str, columns: Iterable[str]) -> set[str]:
    """Return the data-frame columns referenced by an expression.

    Parameters
    ----------
    expr:
        Expression string (e.g. ``"age + I(age^2) + Sex"``).
    columns:
        Available column names.

    Returns
    -------
    set[str]
        Names that are both valid identifiers in ``expr`` and present in
        ``columns``.
    """
    cols = set(columns)
    return {t for t in _IDENT.findall(expr) if t in cols}


def eval_expr(expr: str, df: pd.DataFrame) -> np.ndarray:
    """Evaluate a numeric expression against a data frame.

    Data-frame columns are available as variables, the NumPy functions listed
    in ``_FUNCS`` are available, and R's ``^`` is translated to ``**``.

    Parameters
    ----------
    expr:
        Numeric expression such as ``"age + I(age^2)"``.
    df:
        Data frame providing the columns.

    Returns
    -------
    numpy.ndarray
        A float array with one value per row (a scalar result is broadcast).

    Raises
    ------
    ValueError
        If the expression cannot be evaluated.
    """
    env = {c: df[c].to_numpy() for c in df.columns if _IDENT.fullmatch(str(c))}
    try:
        v = eval(compile(r_to_py(expr), "<expr>", "eval"), {"__builtins__": {}, **_FUNCS}, env)  # noqa: S307
    except Exception as e:
        raise ValueError(f"could not evaluate '{expr}': {e}") from e
    v = np.asarray(v, dtype=float)
    if v.ndim == 0:
        v = np.full(len(df), float(v))
    return v


def design(rhs: str, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the design matrix of one right-hand side with formulaic.

    Parameters
    ----------
    rhs:
        Model terms such as ``'age + Sex'``, ``'1'``, ``'0 + x'`` or
        ``'I(age^2)'``.
    df:
        Data frame with the columns referenced by ``rhs``.

    Returns
    -------
    X : numpy.ndarray
        The ``(n, p)`` design matrix, including the intercept when requested.
    names : list[str]
        Column names; the intercept is reported as ``"(Intercept)"``.

    Raises
    ------
    ValueError
        If the resulting matrix has a different number of rows than ``df``.
    """
    rhs = _fix_I(rhs.strip() or "1")
    mm = formulaic.model_matrix(rhs, df, na_action="raise")
    names = ["(Intercept)" if c == "Intercept" else str(c) for c in mm.columns]
    X = np.asarray(mm, dtype=float)
    if X.shape[0] != len(df):
        raise ValueError(f"the design matrix for '{rhs}' has {X.shape[0]} rows but the data has {len(df)}")
    return X, names


def factor_codes(values: pd.Series | np.ndarray | Sequence) -> tuple[np.ndarray, pd.Index]:
    """Encode values as R-style factor codes ``0..G-1``.

    Levels are sorted (or follow the order of a categorical dtype) so the
    result is stable and matches R's factor coding.

    Parameters
    ----------
    values:
        Series, array or sequence of labels.

    Returns
    -------
    codes : numpy.ndarray
        Integer codes with dtype ``int32``.
    levels : pandas.Index
        The levels in coding order.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        codes = s.cat.codes.to_numpy().astype(np.int32)
        return codes, pd.Index(s.cat.categories)
    codes, uniq = pd.factorize(s, sort=True)
    return codes.astype(np.int32), pd.Index(uniq)


def nested_codes(df: pd.DataFrame, chain: Sequence[str]) -> np.ndarray:
    """Integer id of the subgroup defined by a grouping chain.

    For a single variable this is the ordinary factor code.  For a chain
    ``(g1, ..., gk)`` each distinct combination of the ``k`` variables gets a
    new integer, implementing R's ``g1/g2/...`` nesting.

    Parameters
    ----------
    df:
        Data frame with the grouping columns.
    chain:
        Grouping variables from outer to inner.

    Returns
    -------
    numpy.ndarray
        Integer codes with dtype ``int32``.
    """
    if len(chain) == 1:
        return factor_codes(df[chain[0]])[0]
    key = pd.MultiIndex.from_frame(df[list(chain)])
    codes, _ = pd.factorize(key, sort=True)
    return codes.astype(np.int32)


def collect_columns(exprs: Iterable[str | None], df: pd.DataFrame) -> list[str]:
    """Collect the data-frame columns referenced by several expressions.

    Order of first appearance is preserved and duplicates are removed; this is
    the set of columns that must be complete for ``na.omit``.

    Parameters
    ----------
    exprs:
        Expressions (``None`` entries are ignored).
    df:
        Data frame providing the column names.

    Returns
    -------
    list[str]
    """
    seen: list[str] = []
    for e in exprs:
        if not e:
            continue
        for n in names_in(e, df.columns):
            if n not in seen:
                seen.append(n)
    return seen


def handle_na(df: pd.DataFrame, cols: Sequence[str], na_action: str = "omit") -> tuple[pd.DataFrame, np.ndarray]:
    """Apply R's ``na.omit`` / ``na.fail`` semantics over the used columns.

    Infinite values are treated as missing, like R's ``na.omit`` on model
    frames.

    Parameters
    ----------
    df:
        Input data frame.
    cols:
        Columns that must be complete.
    na_action:
        ``"omit"`` drops incomplete rows, ``"fail"`` raises an error.

    Returns
    -------
    data : pandas.DataFrame
        The data with the incomplete rows removed.
    keep : numpy.ndarray
        Boolean mask of the rows kept.

    Raises
    ------
    ValueError
        If ``na_action`` is invalid or ``"fail"`` finds missing values.
    """
    if na_action not in ("omit", "fail"):
        raise ValueError("na_action must be 'omit' or 'fail'")
    sub = df[list(cols)] if cols else df.iloc[:, :0]
    bad = np.array(sub.isna().any(axis=1).to_numpy(), dtype=bool, copy=True) if len(cols) else np.zeros(len(df), bool)
    for c in cols:   # infinities are also treated as missing
        if pd.api.types.is_float_dtype(df[c]):
            bad |= np.isinf(df[c].to_numpy())
    if bad.any() and na_action == "fail":
        raise ValueError(f"{int(bad.sum())} rows have missing values (na_action='fail')")
    keep = ~bad
    return df.loc[keep], keep


def design_spec(rhs: str, df: pd.DataFrame):
    """Build a design matrix and keep formulaic's model specification.

    Unlike :func:`design`, the returned model spec records factor coding and
    stateful transformations so the matrix can be rebuilt on new data with
    :func:`design_new` (used by ``predict``).

    Parameters
    ----------
    rhs:
        Model terms.
    df:
        Data frame with the columns referenced by ``rhs``.

    Returns
    -------
    X : numpy.ndarray
        The ``(n, p)`` design matrix.
    names : list[str]
        Column names, with ``"(Intercept)"`` for the intercept.
    spec : formulaic.ModelSpec
        The fitted specification.

    Raises
    ------
    ValueError
        If the resulting matrix has a different number of rows than ``df``.
    """
    rhs = _fix_I(rhs.strip() or "1")
    mm = formulaic.model_matrix(rhs, df, na_action="raise")
    names = ["(Intercept)" if c == "Intercept" else str(c) for c in mm.columns]
    X = np.asarray(mm, dtype=float)
    if X.shape[0] != len(df):
        raise ValueError(f"the design matrix for '{rhs}' has {X.shape[0]} rows but the data has {len(df)}")
    return X, names, mm.model_spec


def design_new(spec, newdata: pd.DataFrame) -> np.ndarray:
    """Rebuild a design matrix on new data using the coding learned in the fit.

    Rows with missing values yield ``NaN`` (``na.pass``); a factor level not
    seen in the fit is an error, as in R.

    Parameters
    ----------
    spec:
        The ``ModelSpec`` returned by :func:`design_spec`.
    newdata:
        New data frame.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If ``newdata`` contains factor levels that were not present in the fit.
    """
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mm = spec.update(na_action="ignore").get_model_matrix(newdata)
    for x in w:
        if "outside of the nominated levels" in str(x.message) or "not seen in original" in str(x.message):
            raise ValueError(f"newdata has factor levels not seen in the fit: {x.message}")
    return np.asarray(mm, dtype=float)
