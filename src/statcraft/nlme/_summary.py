"""Variance and correlation parameter tables for ``summary()``.

The R/nlme summary organizes these parameters into named blocks
(``Random effects``, ``Correlation structure``, ``Variance function``,
``Residual``) instead of a flat list such as ``est(cor.phi)``. This module
translates the engine's internal names (``re0.sd[0]``, ``cor.phi``,
``var0.ratio[1]``, ``sigma``, ...) into coefficient blocks understood by
:func:`statcraft.summary.build_summary`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .formula import factor_codes, nested_codes, parse_form
from ..summary import TableBlock

__all__ = [
    "fixed_block",
    "lm_varcorr_blocks",
    "nlme_varcorr_blocks",
    "crossed_varcorr_blocks",
    "lm_random_tables",
    "nlme_random_tables",
    "crossed_random_tables",
    "standardized_residuals_table",
    "residual_summary_table",
    "glmer_fit_table",
    "grouping_counts",
    "grouping_text",
    "structure_text",
    "structure_details",
    "balanced_info",
    "r2_icc",
]

_COR_KIND = {
    "AR1": "AR(1)",
    "CAR1": "CAR(1)",
    "CompoundSymmetry": "Compound symmetry",
    "Symm": "General correlation",
    "ARMA": "ARMA",
    "Exp": "Exponential",
    "Gaus": "Gaussian",
    "Lin": "Linear",
    "Ratio": "Rational quadratic",
    "Spher": "Spherical",
    "Custom": "Custom",
}

_COR_PARAM = {"phi": "Phi", "rho": "rho", "range": "range", "nugget": "nugget"}

_VAR_KIND = {
    "Ident": "varIdent",
    "Power": "varPower",
    "Exp": "varExp",
    "ConstPower": "varConstPower",
    "ConstProp": "varConstProp",
    "Fixed": "varFixed",
    "Custom": "varCustom",
}


def _fmt_value(v: Any, digits: int = 5) -> str:
    """Format a number with ``digits`` significant digits for summary tables."""
    return f"{float(v):.{digits}g}"


def _default_display(name: Any) -> str:
    """Name of a random component without the double parentheses of ``sd((Intercept))``."""
    text = str(name)
    return "Intercept" if text in ("Intercept", "(Intercept)") else text


def _cor_display(label: str) -> str:
    """lmer/glmer ``corr(...)`` label -> ``cor(...)`` without a nested ``(Intercept)``."""
    return label.replace("corr(", "cor(", 1).replace("(Intercept)", "Intercept")


def _info_label(pair) -> str:
    """Comparison key for a (label, value) header row."""
    return str(pair[0]).rstrip(":").strip()


def balanced_info(
    left: Sequence[tuple[str, str]],
    right: Sequence[tuple[str, str]],
    left_extra: Sequence[tuple[str, str]] = (),
    right_extra: Sequence[tuple[str, str]] = (),
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Give both sides of the summary header the same number of cells.

    The short side is first topped up with the real, caller-supplied
    ``*_extra`` metrics, in order.  Only if those run out are truly empty
    rows added, so the two columns line up without inventing values.
    """
    left_rows = [(str(k), str(v)) for k, v in left]
    right_rows = [(str(k), str(v)) for k, v in right]
    target = max(len(left_rows), len(right_rows))

    for rows, extra in ((left_rows, left_extra), (right_rows, right_extra)):
        seen = {_info_label(p) for p in rows}
        for pair in extra:
            if len(rows) >= target:
                break
            label = _info_label(pair)
            if label and label not in seen and str(pair[1]).strip():
                rows.append((str(pair[0]), str(pair[1])))
                seen.add(label)

    pad = ("", "")
    left_rows.extend([pad] * (target - len(left_rows)))
    right_rows.extend([pad] * (target - len(right_rows)))
    return left_rows, right_rows


def grouping_counts(terms_info: Optional[Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    """Number of observed levels per grouping factor in a crossed fit."""
    by: dict[str, set] = {}
    for info in terms_info or []:
        name = str(info.get("name", "Group"))
        by.setdefault(name, set()).update(str(v) for v in info.get("levels", []))
    return {name: len(levels) for name, levels in by.items()}


def grouping_text(terms_info: Optional[Sequence[Mapping[str, Any]]]) -> str:
    """Compact ``group (n levels)`` description of the random-effects terms."""
    return ", ".join(f"{g} ({n})" for g, n in grouping_counts(terms_info).items())


def structure_text(fit) -> str:
    """Compact description of the correlation/variance structures of a fit."""
    pieces = []
    cor = getattr(fit, "cor_spec", None)
    var = getattr(fit, "var_spec", None)
    if cor is not None:
        kind = getattr(cor, "kind", "")
        if kind == "Custom":
            name = str((cor.custom or {}).get("name", "corCustom"))
        else:
            name = _COR_KIND.get(kind, str(kind))
            if kind == "ARMA":
                name += f"({cor.p},{cor.q})"
        pieces.append(name)
    if var is not None:
        pieces.append(" * ".join(_var_name(p) for p in _var_parts(var)))
    return ", ".join(pieces) if pieces else "none"


def structure_details(fit) -> str:
    """Full structure specification, kept for a footer note instead of widening the header."""
    pieces = []
    cor = getattr(fit, "cor_spec", None)
    var = getattr(fit, "var_spec", None)
    if cor is not None:
        pieces.append(_cor_heading(cor))
    if var is not None:
        pieces.append(_var_heading(var))
    return "; ".join(pieces)


def _block(heading: str, names: Sequence[str], coefs: Sequence[float]) -> dict:
    """Estimates-only block (like the R nlme/lme4 summary)."""
    return {
        "heading": heading,
        "names": [str(n) for n in names],
        "coefs": np.asarray(coefs, dtype=float),
        "se": None,
    }


def fixed_block(fixef: pd.Series, se: pd.Series, dist: str = "t", df: Optional[float] = None) -> dict:
    """Fixed-effects table with standard errors and inference."""
    return {
        "heading": "Fixed effects",
        "names": list(fixef.index),
        "coefs": fixef.to_numpy(),
        "se": se.to_numpy(),
        "dist": dist,
        "df": df,
    }


# ---------------------------------------------------------------------------------------------------------------
# parameter names
# ---------------------------------------------------------------------------------------------------------------
def _var_parts(var) -> list:
    """Simple components of a VarStruct (unwraps ``varComb`` in order)."""
    if var is None:
        return []
    if getattr(var, "kind", None) == "Comb":
        out = []
        for part in var.parts:
            out.extend(_var_parts(part))
        return out
    return [var]


def _var_name(comp) -> str:
    """Display name of one variance-function component (``varPower`` etc.)."""
    if getattr(comp, "kind", None) == "Custom":
        return str((comp.custom or {}).get("name", "varCustom"))
    return _VAR_KIND.get(getattr(comp, "kind", ""), f"var{getattr(comp, 'kind', '')}")


def _cor_heading(cor) -> str:
    """Heading of the correlation-structure block, with kind and formula."""
    if cor is None:
        return "Correlation structure"
    kind = getattr(cor, "kind", "")
    if kind == "Custom":
        name = str((cor.custom or {}).get("name", "corCustom"))
    else:
        name = _COR_KIND.get(kind, str(kind))
        if kind == "ARMA":
            name += f"({cor.p},{cor.q})"
    form = getattr(cor, "form", None)
    return f"Correlation structure: {name}" + (f" ({form})" if form else "")


def _var_heading(var) -> str:
    """Heading of the variance-function block, with components and formulas."""
    if var is None:
        return "Variance function"
    parts = _var_parts(var)
    name = " * ".join(_var_name(p) for p in parts)
    forms = [p.form for p in parts if getattr(p, "form", None)]
    if forms:
        name += f" ({'; '.join(forms)})"
    return f"Variance function: {name}"


def _r_strata(codes: np.ndarray, gcodes: Optional[np.ndarray]) -> np.ndarray:
    """Copy of the strata recoding from :mod:`models` (reference = first occurrence)."""
    idx = np.arange(len(codes)) if gcodes is None else np.argsort(gcodes, kind="stable")
    seen: dict = {}
    for c in codes[idx]:
        if c not in seen:
            seen[c] = len(seen)
    return np.array([seen[c] for c in codes], dtype=np.int32)


def _strata_labels(form: Optional[str], df: Optional[pd.DataFrame], gcodes: Optional[np.ndarray]) -> Optional[list[str]]:
    """Strata labels in the engine's internal order, or None if they cannot be deduced."""
    if df is None or not form:
        return None
    fp = parse_form(form)
    groups = fp.groups
    if not groups:
        return None
    try:
        codes = nested_codes(df, groups)
        rec = _r_strata(codes, gcodes)
        order = np.argsort(gcodes, kind="stable") if gcodes is not None else np.arange(len(codes))
        by_rec: dict[int, int] = {}
        for i in order:
            by_rec.setdefault(int(rec[i]), int(codes[i]))
        if len(groups) == 1:
            _, uniq = factor_codes(df[groups[0]])
            labels = [str(v) for v in uniq]
        else:
            key = pd.MultiIndex.from_frame(df[list(groups)])
            uniq = key.unique().sort_values()
            labels = ["/".join(str(v) for v in tup) for tup in uniq]
        prefix = "/".join(groups)
        return [f"{prefix}={labels[by_rec[k]]}" for k in sorted(by_rec)]
    except Exception:  # incomplete data to rebuild the levels: fall back to the generic name
        return None


def _var_param_labels(var, df: Optional[pd.DataFrame], gcodes: Optional[np.ndarray]) -> list[str]:
    """Readable names of the variance parameters, in the order emitted by the engine."""
    out: list[str] = []
    for comp in _var_parts(var):
        kind = comp.kind
        levels = _strata_labels(comp.form, df, gcodes)
        if kind == "Fixed":
            continue
        if kind == "Ident":
            if levels:
                out.extend(levels[1:])          # the first stratum is the reference (ratio 1)
        elif kind in ("Power", "Exp"):
            if levels:
                out.extend(levels)
            else:
                out.append("power" if kind == "Power" else "exp")
        elif kind in ("ConstPower", "ConstProp"):
            first, second = ("const", "power") if kind == "ConstPower" else ("const", "prop")
            if levels:
                out.extend([f"{first} ({lev})" for lev in levels])
                out.extend([f"{second} ({lev})" for lev in levels])
            else:
                out.extend([first, second])
        elif kind == "Custom":
            cname = _var_name(comp)
            n = int((comp.custom or {}).get("n_params", 0))
            out.extend([f"{cname}[{i}]" for i in range(n)])
    return out


def _fallback_var_name(name: str) -> str:
    """Turn an engine name such as ``var0.Power[1]`` into ``var0:Power[1]``.

    Used when the readable labels cannot be rebuilt (for example because the
    original data are unavailable).
    """
    m = re.fullmatch(r"var(\d+)\.(\w+)\[(\d+)\]", name)
    if m:
        return f"var{m.group(1)}:{m.group(2)}[{m.group(3)}]"
    return name


# ---------------------------------------------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------------------------------------------
def _lm_re_heading(random_spec, chain, level: int) -> str:
    """Heading for the random-effects block of grouping ``level`` in lme/gls."""
    if random_spec and level < len(random_spec):
        group, rhs = random_spec[level]
        group_name = "/".join(chain[: level + 1]) if chain else str(group)
        return f"Random effects: ~{rhs} | {group_name}"
    if chain and level < len(chain):
        return f"Random effects: {chain[level]}"
    return "Random effects"


def lm_varcorr_blocks(
    varcorr: pd.DataFrame,
    *,
    random_spec=None,
    chain: Optional[Sequence[str]] = None,
    level_names: Optional[Sequence[Sequence[str]]] = None,
    coef_display: Callable[[Any], str] = _default_display,
    cor_spec=None,
    var_spec=None,
    data: Optional[pd.DataFrame] = None,
    var_gcodes: Optional[np.ndarray] = None,
    random_blocks: bool = True,
    residual: bool = True,
) -> list[dict]:
    """``Random effects`` / ``Correlation structure`` / ``Variance function`` / ``Residual``
    blocks from the ``varcorr`` table of :class:`LmeFit` (lme/gls)."""
    rows = {str(idx): row for idx, row in varcorr.iterrows()}
    chain = list(chain or [])
    blocks: list[dict] = []

    # random effects, one block per grouping level
    re_buckets: dict[int, list[tuple[str, str]]] = {}
    for name in rows:
        m = re.fullmatch(r"re(\d+)\.(sd|cor)\[(.*?)\]", name)
        if m:
            re_buckets.setdefault(int(m.group(1)), []).append((m.group(2), m.group(3)))
    for level in sorted(re_buckets) if random_blocks else []:
        names, coefs = [], []
        lv = level_names[level] if level_names and level < len(level_names) else None
        for kind, idx in re_buckets[level]:
            if kind == "sd":
                k = int(idx)
                comp = coef_display(lv[k]) if lv and k < len(lv) else f"[{k}]"
                names.append(f"sd({comp})")
            else:
                i, j = (int(x) for x in idx.split(","))
                ci = coef_display(lv[i]) if lv and i < len(lv) else f"[{i}]"
                cj = coef_display(lv[j]) if lv and j < len(lv) else f"[{j}]"
                names.append(f"cor({ci},{cj})")
            coefs.append(float(rows[f"re{level}.{kind}[{idx}]"]["estimate"]))
        if names:
            blocks.append(_block(_lm_re_heading(random_spec, chain, level), names, coefs))

    # correlation structure
    cor_names, cor_coefs = [], []
    for name, row in rows.items():
        m = re.fullmatch(r"cor\.(phi|rho|range|nugget)", name)
        if m:
            cor_names.append(_COR_PARAM[m.group(1)])
            cor_coefs.append(float(row["estimate"]))
            continue
        m = re.fullmatch(r"cor\.(ar|ma)\[(\d+)\]", name)
        if m:
            cor_names.append(f"{m.group(1)}({m.group(2)})")
            cor_coefs.append(float(row["estimate"]))
            continue
        if name.startswith("cor["):
            cor_names.append(name.replace(" (theta)", ""))
            cor_coefs.append(float(row["estimate"]))
    if cor_names:
        blocks.append(_block(_cor_heading(cor_spec), cor_names, cor_coefs))

    # variance function
    var_names, var_coefs = [], []
    labels = _var_param_labels(var_spec, data, var_gcodes)
    k = 0
    for name, row in rows.items():
        if not name.startswith("var"):
            continue
        var_names.append(labels[k] if k < len(labels) else _fallback_var_name(name))
        var_coefs.append(float(row["estimate"]))
        k += 1
    if var_names:
        blocks.append(_block(_var_heading(var_spec), var_names, var_coefs))

    # residual
    if residual and "sigma" in rows:
        blocks.append(_block("Residual", ["sigma"], [float(rows["sigma"]["estimate"])]))

    return blocks


def _nlme_display(name: Any) -> str:
    """nlme random-effect name without the redundant ``.(Intercept)`` suffix."""
    n = str(name)
    if n.endswith(".(Intercept)"):
        return n[: -len(".(Intercept)")]
    return "Intercept" if n == "(Intercept)" else n


def _nlme_re_heading(random_spec, chain, level: int) -> str:
    if random_spec and level < len(random_spec):
        group, params = random_spec[level]
        group_name = "/".join(chain[: level + 1]) if chain else str(group)
        if isinstance(params, dict):
            left = " + ".join(params)
            rhs = " + ".join(dict.fromkeys(str(v) for v in params.values()))
            return f"Random effects: {left} ~ {rhs} | {group_name}"
        return f"Random effects: ~{params} | {group_name}"
    if chain and level < len(chain):
        return f"Random effects: {chain[level]}"
    return "Random effects"


def _cov_block(heading: str, cov: np.ndarray, cols: Sequence[str], display: Callable[[Any], str]) -> dict:
    """Standard deviations and correlations of a random-effects covariance matrix."""
    S = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.clip(np.diag(S), 0.0, None))
    labels = [display(c) for c in cols]
    names, coefs = [], []
    for i, lab in enumerate(labels):
        names.append(f"sd({lab})")
        coefs.append(float(sd[i]))
    for i in range(1, len(labels)):
        for j in range(i):
            den = sd[i] * sd[j]
            names.append(f"cor({labels[j]},{labels[i]})")
            coefs.append(float(S[i, j] / den) if den > 0 else float("nan"))
    return _block(heading, names, coefs)


def nlme_varcorr_blocks(fit) -> list[dict]:
    """Variance blocks for ``nlme`` (from ``Sigma_b``/``Sigma_levels``) or ``gnls`` (residual only)."""
    blocks: list[dict] = []
    if getattr(fit, "kind", None) == "nlme":
        random_spec = list(getattr(fit, "random_spec", None) or [])
        chain = [g for g, _ in random_spec]
        ranef = getattr(fit, "_ranef", None)
        for level in range(len(random_spec)):
            key = "/".join(chain[: level + 1])
            if isinstance(ranef, dict):
                fr = ranef.get(key)
            else:
                fr = ranef if level == 0 else None
            if fr is None:
                continue
            if level == 0:
                cov = np.array(fit.raw.Sigma_b)
            else:
                levels = list(getattr(fit.raw, "Sigma_levels", []))
                cov = np.array(levels[level]) if level < len(levels) else None
            if cov is None or cov.shape[0] != len(fr.columns):
                continue
            blocks.append(_cov_block(_nlme_re_heading(random_spec, chain, level), cov, list(fr.columns), _nlme_display))
    blocks.append(_block("Residual", ["sigma"], [float(fit.sigma)]))
    return blocks


def crossed_varcorr_blocks(varcorr: pd.DataFrame) -> list[dict]:
    """Random-effect blocks for ``lmer``/``glmer`` (group/name/std_dev/corr columns)."""
    blocks: list[dict] = []
    for group, part in varcorr.groupby("group", sort=False):
        sigma_names, sigma_coefs = [], []
        cor_names, cor_coefs = [], []
        for _, row in part.iterrows():
            label = str(row.get("name", ""))
            if pd.notna(row.get("std_dev")):
                sigma_names.append("sigma" if group == "Residual" else f"sd({_default_display(label)})")
                sigma_coefs.append(float(row["std_dev"]))
            elif pd.notna(row.get("corr")):
                cor_names.append(_cor_display(label))
                cor_coefs.append(float(row["corr"]))
        if group == "Residual":
            if sigma_names:
                blocks.append(_block("Residual", sigma_names, sigma_coefs))
            continue
        if sigma_names or cor_names:
            heading = f"Random effects: {group}"
            blocks.append(_block(heading, sigma_names + cor_names, sigma_coefs + cor_coefs))
    return blocks


def crossed_random_tables(
    fit,
    *,
    residual: bool = True,
    scale: Optional[str] = None,
) -> list[TableBlock]:
    """Full variance/correlation tables for ``lmer``/``glmer`` fits.

    The old summary squeezed ``sd(...)`` and ``cor(...)`` estimates into a
    coefficient-style block.  This rebuilds the same information as the rich
    ``lme``/``nlme`` tables, keeping the correlations in one compact, aligned
    ``Corr`` column instead of adding one column per pair.  ``residual=False``
    omits the residual row for GLMMs, whose scale is fixed by the family rather
    than estimated as a Gaussian variance.
    """
    levels: list[tuple[str, str, list[list[Any]], dict[tuple[int, int], float]]] = []
    for group, part in fit.varcorr.groupby("group", sort=False):
        if group == "Residual":
            continue
        sd_rows = part[part["std_dev"].notna()]
        if sd_rows.empty:
            continue
        labels = [_default_display(str(name)) for name in sd_rows["name"]]
        pos: dict[str, int] = {}
        for i, label in enumerate(labels):
            pos.setdefault(label, i)
        pairs: dict[tuple[int, int], float] = {}
        for _, row in part[part["corr"].notna()].iterrows():
            m = re.fullmatch(r"corr\((.*),(.*)\)", str(row["name"]))
            if not m:
                continue
            a = _default_display(m.group(1).strip())
            b = _default_display(m.group(2).strip())
            if a in pos and b in pos:
                upper, lower = max(pos[a], pos[b]), min(pos[a], pos[b])
                pairs[(upper, lower)] = float(row["corr"])
        rows = [
            [label, float(variance), float(sd)]
            for label, variance, sd in zip(labels, sd_rows["variance"], sd_rows["std_dev"])
        ]
        heading = f"Random effects: {group}" + (f" ({scale} scale)" if scale else "")
        levels.append((heading, str(group), rows, pairs))

    sigma = float(fit.sigma) if residual and getattr(fit, "sigma", None) is not None else None
    return _variance_tables(levels, sigma, include_residual=residual)


# ---------------------------------------------------------------------------------------------------------------
# richer mixed-model pieces: variance-components tables, residuals, R2/ICC
# ---------------------------------------------------------------------------------------------------------------
def _corr_column(
    pairs: Mapping[tuple[int, int], float], labels: Sequence[str]
) -> list[str]:
    """Compact ``Corr`` cells with fixed-width, comma-free subcolumns.

    ``pairs`` uses ``(upper index, lower index)`` keys in the order of
    ``labels``.  A row packs its correlations with the preceding components
    into one cell, but each estimate occupies the same fixed width, so the
    values line up across rows without adding one table column per pair.
    """
    raw: list[list[str]] = []
    for upper in range(len(labels)):
        values = []
        for lower in range(upper):
            value = pairs.get((upper, lower))
            if value is not None and np.isfinite(value):
                values.append(f"{float(value):7.4f}")
        raw.append(values)
    return [" ".join(values) for values in raw]


def _variance_tables(
    levels: list[tuple[str, str, list[list[Any]], dict[tuple[int, int], float]]],
    sigma: Optional[float] = None,
    *,
    include_residual: bool = True,
) -> list[TableBlock]:
    """Build the ``Group / Name / Variance / Std.Dev. / Corr`` tables.

    ``levels`` holds ``(heading, group label, rows, correlations)`` entries
    ordered from the outer grouping level to the innermost one.  Each row has
    ``[Name, Variance, Std.Dev.]``; ``correlations`` maps a pair of row indices
    to its estimate.  The correlations are packed into one compact, right-
    aligned ``Corr`` cell per row instead of one column per pair, so the table
    does not explode with many random parameters.  The residual row is appended
    to the last table (or becomes a standalone table when there are no random
    effects, as in ``gnls``), unless ``sigma`` is None (GLMMs have no residual
    variance on the response scale).
    """
    if not levels:
        if sigma is None or not include_residual:
            return []
        return [
            TableBlock(
                heading="Residual",
                columns=["Variance", "Std.Dev."],
                rows=[[sigma ** 2, sigma]],
            )
        ]

    out: list[TableBlock] = []
    for i, (heading, group, rows, pairs) in enumerate(levels):
        rows = list(rows)
        labels = [str(row[0]) for row in rows]
        corr_cells = _corr_column(pairs, labels)
        full_rows = [list(row) + [cell] for row, cell in zip(rows, corr_cells)]
        idx = [group] + [""] * (len(rows) - 1)

        add_residual = include_residual and sigma is not None and i == len(levels) - 1
        if add_residual:
            full_rows.append(["Residual", float(sigma) ** 2, float(sigma), ""])
            idx.append("")

        out.append(
            TableBlock(
                heading=heading,
                columns=["Name", "Variance", "Std.Dev.", "Corr"],
                rows=full_rows,
                index=idx,
                index_name="Group",
                align=["left", "right", "right", "right"],
            )
        )
    return out


def lm_random_tables(fit) -> list[TableBlock]:
    """Variance-components tables for ``lme`` fits.

    Reads the ``re<level>.sd[k]`` / ``re<level>.cor[i,j]`` entries of
    ``fit.varcorr`` and rebuilds the familiar R layout: one table per grouping
    level with ``Group``, ``Name``, ``Variance``, ``Std.Dev.`` and the
    correlations with the preceding components, followed by the residual row.
    """
    rows = {str(idx): row for idx, row in fit.varcorr.iterrows()}
    random_spec = list(getattr(fit, "random_spec", None) or [])
    chain = [g for g, _ in random_spec]
    level_names = [
        list(getattr(pl.get("spec"), "column_names", []) or [])
        for pl in getattr(fit, "_pred_levels", [])
    ]

    levels: list[tuple[str, str, list[list[Any]], dict[tuple[int, int], float]]] = []
    for level in range(max(len(level_names), len(chain))):
        sds: dict[int, float] = {}
        cors: dict[tuple[int, int], float] = {}
        for name, row in rows.items():
            m = re.fullmatch(rf"re{level}\.sd\[(\d+)\]", name)
            if m:
                sds[int(m.group(1))] = float(row["estimate"])
                continue
            m = re.fullmatch(rf"re{level}\.cor\[(\d+),(\d+)\]", name)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                cors[(max(a, b), min(a, b))] = float(row["estimate"])
        if not sds:
            continue
        q = max(sds) + 1
        labels = level_names[level] if level < len(level_names) else []
        table_rows = []
        pairs: dict[tuple[int, int], float] = {}
        for k in range(q):
            sd = sds.get(k, float("nan"))
            for j in range(k):
                if (k, j) in cors:
                    pairs[(k, j)] = float(cors[(k, j)])
            table_rows.append(
                [
                    str(labels[k]) if k < len(labels) else f"[{k}]",
                    sd * sd,
                    sd,
                ]
            )
        group = chain[level] if level < len(chain) else "Group"
        levels.append((_lm_re_heading(random_spec, chain, level), group, table_rows, pairs))

    return _variance_tables(levels, float(fit.sigma))


def nlme_random_tables(fit) -> list[TableBlock]:
    """Variance-components tables for ``nlme`` (random effects) or ``gnls`` (residual only)."""
    levels: list[tuple[str, str, list[list[Any]], dict[tuple[int, int], float]]] = []
    if getattr(fit, "kind", None) == "nlme":
        random_spec = list(getattr(fit, "random_spec", None) or [])
        chain = [g for g, _ in random_spec]
        ranef = getattr(fit, "_ranef", None)
        for level in range(len(random_spec)):
            key = "/".join(chain[: level + 1])
            fr = ranef.get(key) if isinstance(ranef, dict) else (ranef if level == 0 else None)
            if fr is None:
                continue
            if level == 0:
                cov = np.array(fit.raw.Sigma_b, dtype=float)
            else:
                sig_levels = list(getattr(fit.raw, "Sigma_levels", []))
                cov = np.array(sig_levels[level], dtype=float) if level < len(sig_levels) else None
            if cov is None or cov.shape[0] != len(fr.columns):
                continue
            sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            table_rows = []
            pairs: dict[tuple[int, int], float] = {}
            for k, col in enumerate(fr.columns):
                for j in range(k):
                    if sd[k] > 0 and sd[j] > 0:
                        pairs[(k, j)] = float(cov[k, j] / (sd[k] * sd[j]))
                table_rows.append([_nlme_display(col), cov[k, k], sd[k]])
            levels.append((_nlme_re_heading(random_spec, chain, level), chain[level], table_rows, pairs))

    return _variance_tables(levels, _residual_sigma(fit))


def _residual_sigma(fit) -> float:
    """Residual standard deviation reported by the summary (R uses N-p for gnls)."""
    if getattr(fit, "kind", None) == "gnls":
        return float(getattr(fit, "sigma_r", fit.sigma))
    return float(fit.sigma)


def residual_summary_table(
    fit,
    *,
    heading: str = "Residuals",
    standardized: bool = False,
    residual_type: Optional[str] = None,
) -> Optional[TableBlock]:
    """``Min / Q1 / Med / Q3 / Max`` of one residual series.

    ``standardized=True`` divides by the fitted scale (Gaussian residual SE);
    ``residual_type`` selects a GLMM residual type such as ``"deviance"``.
    """
    try:
        resid = fit.residuals() if residual_type is None else fit.residuals(type=residual_type)
        resid = np.asarray(getattr(resid, "to_numpy", lambda: resid)(), dtype=float)
        if standardized:
            scale = _residual_sigma(fit) if getattr(fit, "kind", None) in ("lme", "nlme") else getattr(fit, "sigma", None)
            if scale is not None and np.isfinite(scale) and scale > 0:
                resid = resid / float(scale)
    except Exception:
        return None
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return None
    q = np.quantile(resid, [0.0, 0.25, 0.5, 0.75, 1.0])
    return TableBlock(
        heading=heading,
        columns=["Min", "Q1", "Med", "Q3", "Max"],
        rows=[q],
        formats=[".3f"] * 5,
    )


def standardized_residuals_table(fit) -> Optional[TableBlock]:
    """``Min / Q1 / Med / Q3 / Max`` of the standardized residuals.

    Residuals use all random-effect levels for ``nlme`` and the fixed-effects
    fit for ``gnls``; they are divided by the residual standard deviation
    (the same standardization R's summary methods display).
    """
    heading = (
        "Standardized within-group residuals"
        if getattr(fit, "kind", None) in ("lme", "nlme")
        else "Standardized residuals"
    )
    return residual_summary_table(fit, heading=heading, standardized=True)


def glmer_fit_table(fit) -> Optional[TableBlock]:
    """Compact goodness-of-fit table for a GLMM (deviance and Pearson chi-square)."""
    try:
        pearson = np.asarray(fit.residuals(type="pearson").to_numpy(), dtype=float)
        chi2 = float(np.sum(pearson ** 2))
        df = max(int(fit.nobs - len(fit.fixef)), 1)
        rows = [
            ["Deviance", float(fit.deviance)],
            ["Pearson chi-square", chi2],
            ["Pearson chi-square / df", chi2 / df],
            ["Residual df", df],
        ]
    except Exception:
        return None
    return TableBlock(
        heading="Goodness of fit",
        columns=["Statistic", "Value"],
        rows=rows,
        formats=[".4f", ".4f"],
        align=["left", "right"],
    )


def r2_icc(fit) -> Optional[dict[str, float]]:
    """Marginal/conditional R2 (Nakagawa-style) and the ICC of a mixed model.

    ``R2 marginal`` uses only the fixed-effects variance; ``R2 conditional``
    adds the variance explained by the random effects.  ``ICC`` is the share
    of the random-effects plus residual variance due to the random effects.
    For ``gnls`` (no random effects) only the ordinary R2 is returned.
    """
    try:
        y = np.asarray(fit._y, dtype=float)
        f0 = np.asarray(fit.fitted(0).to_numpy(), dtype=float)
        sigma = _residual_sigma(fit) if hasattr(fit, "kind") else getattr(fit, "sigma", None)
        ve = float(sigma) ** 2 if sigma is not None else float("nan")
        # ``lme``/``nlme`` expose ``kind``; ``lmer`` exposes ``terms_info`` and
        # the same fitted(0)/fitted() pair for population vs conditional means.
        has_re = getattr(fit, "kind", None) in ("lme", "nlme") or (
            hasattr(fit, "terms_info") and sigma is not None
        )
        if has_re:
            f1 = np.asarray(fit.fitted().to_numpy(), dtype=float)
            vf = float(np.var(f0, ddof=1))
            vr = float(np.var(f1 - f0, ddof=1))
            total = vf + vr + ve
            return {
                "r2m": vf / total if total > 0 else float("nan"),
                "r2c": (vf + vr) / total if total > 0 else float("nan"),
                "icc": vr / (vr + ve) if (vr + ve) > 0 else float("nan"),
            }
        sse = float(np.sum((y - f0) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        return {"r2m": 1.0 - sse / sst if sst > 0 else float("nan")}
    except Exception:
        return None
