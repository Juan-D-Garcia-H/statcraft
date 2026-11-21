"""Data preparation: DataFrame (long format) -> ``_core.Data`` + metadata."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .. import _saemfit as _core

_EVENT_DEFAULTS = {"amt": "amt", "cmt": "cmt", "rate": "rate", "addl": "addl", "ii": "ii", "ss": "ss"}


@dataclass
class DataInfo:
    """Metadata about the prepared data, shared between the fitting engine and the result object.

    It records the column names that were used, how subjects and rows were mapped to the
    core representation, and the covariates/categories extracted from the long-format frame.

    Attributes
    ----------
    id_col, time_col, y_col : str
        Names of the subject, time and response columns in the original data. ``time_col`` is a
        synthetic column (``"obs"``, ``"obs_"``, ...) when the data had no time column.
    ids : numpy.ndarray
        Original subject labels in core order (one entry per subject).
    row_index : pandas.Index
        Original index of the rows that were used (after NA/event filtering), in core order.
    subj : numpy.ndarray
        Zero-based subject index of every used row.
    x_names : list[str]
        Names of the within-subject predictors; the first element is always the time column.
    cov_names : list[str]
        Numeric per-subject covariates in core order.
    cov_values : dict[str, numpy.ndarray]
        Mapping ``name -> array(n_subjects)`` with the covariate value of each subject.
    cat_names : list[str]
        Categorical covariates (constant within subject).
    cat_labels : dict[str, numpy.ndarray]
        Mapping ``name -> array(n_subjects)`` with the string label of each subject.
    cat_reference : dict[str, str]
        Reference level used for each categorical covariate.
    ytype_labels : numpy.ndarray, optional
        Response labels when the data have multiple responses; ``None`` for a single response.
    group_labels : numpy.ndarray, optional
        Cluster labels for the hierarchical (two-level) model; ``None`` otherwise.
    n_dropped : int
        Number of rows removed because of missing values.
    frame : pandas.DataFrame
        Rows that were used, kept for ``predict``/``plot`` and ``to_frame``.
    events : pandas.DataFrame, optional
        Dose-event records that were supplied separately or extracted through ``evid``.
    event_cols : dict[str, str]
        Column mapping for the dose-event record.
    id_synthetic : bool
        ``True`` when no subject column was supplied and every row was treated as a subject.
    time_synthetic : bool
        ``True`` when no time column was supplied and ``time_col`` holds the within-subject
        observation index instead.
    """

    id_col: str
    time_col: str
    y_col: str
    ids: np.ndarray                      # original label of each subject (core order)
    row_index: pd.Index                  # original index of the rows used
    subj: np.ndarray                     # subject (0-based) of each row used
    x_names: list                        # [time, predictors...]
    cov_names: list                      # numeric per-subject covariates (core order)
    cov_values: dict                     # name -> array(ns)
    cat_names: list
    cat_labels: dict                     # name -> array(ns) of str
    cat_reference: dict
    ytype_labels: Optional[np.ndarray] = None
    group_labels: Optional[np.ndarray] = None
    n_dropped: int = 0
    frame: pd.DataFrame = field(default=None, repr=False)   # rows used (for predict/plot)
    events: Optional[pd.DataFrame] = field(default=None, repr=False)
    event_cols: dict = field(default_factory=dict)
    cens_col: Optional[str] = None                         # column with -1/0/+1 censoring codes
    limit_col: Optional[str] = None                        # column with the censoring limit
    id_synthetic: bool = False                             # no subject column: each row is a subject (id 1..N)
    time_synthetic: bool = False                           # no time column: time_col is the observation index

    @property
    def n_subjects(self) -> int:
        """Number of subjects left after data preparation."""
        return len(self.ids)


def as_frame(data) -> pd.DataFrame:
    """Coerce ``data`` to a :class:`pandas.DataFrame`.

    Accepts a DataFrame as-is, a mapping of column name to values, or any sequence accepted by
    ``pandas.DataFrame``. Raises :class:`TypeError` if the conversion is not possible.
    """
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, Mapping):
        return pd.DataFrame({k: np.asarray(v) for k, v in data.items()})
    try:
        return pd.DataFrame(data)
    except Exception as e:  # pragma: no cover
        raise TypeError("data must be a pandas DataFrame or a dict of columns") from e


def _need(df: pd.DataFrame, col: str, what: str):
    """Validate that column ``col`` exists, with an error that lists the available columns.

    ``what`` is the user-facing role of the column (``"id"``, ``"response"``, ...) and is included
    in the message to make the failing argument obvious.
    """
    if col not in df.columns:
        raise KeyError(f"column {what}={col!r} does not exist in data. Columns: {list(df.columns)}")


def _is_categorical(s: pd.Series) -> bool:
    """Return ``True`` for categorical, object/string or boolean columns."""
    return (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s.dtype)
            or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype))


def _event_arrays(ev: pd.DataFrame, id_map: dict, id_col: str, time_col: str, cols: dict):
    """Convert a dose-event frame to the eight core arrays, dropping unknown subjects.

    Parameters
    ----------
    ev : pandas.DataFrame
        Dose records with one row per administration.
    id_map : dict
        Mapping from the original subject label to its 1-based core id.
    id_col, time_col : str
        Names of the subject and time columns.
    cols : dict
        Mapping from the semantic event fields (``amt``, ``cmt``, ``rate``, ``addl``,
        ``ii``, ``ss``) to the actual column names.

    Returns
    -------
    tuple
        ``(ids, time, amt, cmt, rate, addl, ii, ss)`` as NumPy arrays. Events that belong to
        subjects without observations are ignored with a warning.
    """
    ev = ev.copy()
    ids, keep = [], []
    for k, v in enumerate(ev[id_col].tolist()):
        if v in id_map:
            ids.append(id_map[v]); keep.append(k)
    n_unknown = len(ev) - len(keep)
    if n_unknown:
        warnings.warn(f"{n_unknown} doses from subjects without observations: ignored", stacklevel=3)
    ev = ev.iloc[keep]
    n = len(ev)

    def col(name, default, dtype):
        """Read an event column (falling back to ``default`` when absent)."""
        c = cols.get(name, name)
        if c in ev.columns:
            return ev[c].fillna(default).to_numpy(dtype)
        return np.full(n, default, dtype)
    _need(ev, cols.get("amt", "amt"), "events.amt")
    return (np.asarray(ids, np.int64), ev[time_col].to_numpy(float), col("amt", 0.0, float),
            col("cmt", 1, np.int64), col("rate", 0.0, float), col("addl", 0, np.int64),
            col("ii", 0.0, float), col("ss", 0, np.int64))


def prepare(model, data, *, id: Optional[str] = None, time: Optional[str] = None,
            response: Optional[str] = None,
            ytype: Optional[str] = None, cens: Optional[str] = None, limit: Optional[str] = None,
            group: Optional[str] = None, events=None, evid: Optional[str] = None,
            event_cols: Optional[Mapping[str, str]] = None, predictors: Optional[Sequence[str]] = None,
            reference: Optional[Mapping[str, str]] = None, na_action: str = "omit"):
    """Turn long-format data into the low-level core data plus :class:`DataInfo` metadata.

    The function resolves the subject/time/response columns, validates the columns required by the
    model, handles missing values, extracts dose events, and classifies every extra column as a
    within-subject predictor, a numeric covariate or a categorical covariate.

    Parameters
    ----------
    model : NLModel
        Model whose ``predictors``, ``covariates`` and built-in specification are inspected.
    data : pandas.DataFrame or mapping
        One row per observation (long format).
    id, time, response : str, optional
        Subject, time and response columns. When omitted, ``"id"``/``"time"`` are used if present
        and ``response`` defaults to ``"y"``; a synthetic subject column or a synthetic time index
        is created otherwise.
    ytype, cens, limit, group : str, optional
        Multi-response, censoring, censoring-limit and cluster columns.
    events, evid, event_cols : optional
        Separate dose records, an EVID column inside ``data``, and the event column mapping.
    predictors : sequence[str], optional
        Columns that must be treated as within-subject predictors even if constant.
    reference : mapping[str, str], optional
        Reference level of each categorical covariate.
    na_action : {"omit", "fail"}, default "omit"
        Whether to drop rows with missing values or raise an error.

    Returns
    -------
    tuple
        ``(core_data, info)`` where ``core_data`` is a ``_core.Data`` and ``info`` is a
        :class:`DataInfo`.

    Raises
    ------
    KeyError
        If a requested column is missing from the data.
    ValueError
        On invalid event records, invalid ``na_action``, categorical predictors, covariates that
        vary within a subject, or when no observations remain.
    """
    df = as_frame(data)
    response_col = response or "y"
    _need(df, response_col, "response")           # before creating internal columns (obs, id)
    # id: subject column. If omitted, the "id" column is used when it exists; if there is none, there are no groups and
    # each row is treated as a distinct subject (id 1, 2, 3, ...).
    id_synth = False
    if id is None:
        if "id" in df.columns:
            id = "id"
        else:
            if events is not None or evid is not None:
                raise ValueError("events/evid need a subject column: specify id=<column>")
            id = "id"
            df = df.assign(id=np.arange(1, len(df) + 1))
            id_synth = True
    # time: time column. If omitted, the "time" column is used when it exists; if there is none, the model is
    # static (e.g. dose-response) and the observation index within each subject is used.
    time_synth = False
    if time is None:
        if "time" in df.columns:
            time = "time"
        else:
            if events is not None or evid is not None:
                raise ValueError("events/evid need a time column: specify time=<column>")
            if model._builtin is not None or model.family == "tte" or model.tte_start:
                raise ValueError("this model needs a time column: specify time=<column>")
            _need(df, id, "id")
            time = "obs"
            while time in df.columns:
                time += "_"
            df = df.assign(**{time: df.groupby(id, sort=False).cumcount().astype(float)})
            time_synth = True
    model._time_synthetic = time_synth
    # predictors: those declared in the model (NLModel(predictors=...)) and/or in the call. The ORDER of the function's
    # `predictors` argument (SciPy-style unpacking) is the model's and, if absent, the call's.
    _mp, _cp = list(getattr(model, "predictors", None) or []), list(predictors or [])
    model._pred_order = _mp or _cp or None
    predictors = list(dict.fromkeys(_mp + _cp)) or None
    ecols = dict(_EVENT_DEFAULTS)
    ecols.update(event_cols or {})
    for col, what in ((id, "id"), (time, "time")):
        _need(df, col, what)
    if na_action not in ("omit", "fail"):
        raise ValueError("na_action: 'omit' or 'fail'")

    # --- combined NONMEM record (evid) / separate events ---
    ev_df = None
    if evid is not None:
        _need(df, evid, "evid")
        e = df[evid].fillna(0).astype(int)
        bad = ~e.isin([0, 1])
        if bad.any():
            raise ValueError(f"evid: only 0 (observation) and 1 (dose) are allowed; found {sorted(set(e[bad]))}")
        ev_df = df[e == 1]
        df = df[e == 0]
    if events is not None:
        extra = as_frame(events)
        _need(extra, id, "events[id]"); _need(extra, time, "events[time]")
        ev_df = extra if ev_df is None else pd.concat([ev_df, extra], ignore_index=True)
    _need(df, response_col, "response")

    # --- columns required by the model ---
    cov_model_cols = []
    for _, c in model.covariates:
        if c not in cov_model_cols:
            cov_model_cols.append(c)
    builtin_x = model._builtin[1] if model._builtin else None
    builtin_dose = model._builtin[2] if model._builtin else None
    sig = [n for n in model.arg_names() if n not in ("t", "time") or n in df.columns]
    sig = [n for n in sig if n in df.columns and n not in (id, time, response_col)]
    wanted = list(dict.fromkeys(list(predictors or []) + cov_model_cols + sig +
                                [c for c in (builtin_x, builtin_dose) if c]))
    if model.tte_start and model.tte_start not in wanted:
        wanted.append(model.tte_start)
    for c in wanted:
        if c == time:
            continue
        _need(df, c, "covariate/predictor")
    opt = {k: v for k, v in (("ytype", ytype), ("cens", cens), ("limit", limit), ("group", group)) if v}
    for k, v in opt.items():
        _need(df, v, k)

    # --- NA ---
    must = [id, time, response_col] + [c for c in (ytype, group, cens) if c]
    num_wanted = [c for c in wanted if c != time and not _is_categorical(df[c])]
    na_mask = df[must].isna().any(axis=1)
    na_cov = df[num_wanted].isna().any(axis=1) if num_wanted else pd.Series(False, index=df.index)
    na_cat = df[[c for c in wanted if c != time and _is_categorical(df[c])]].isna().any(axis=1) \
        if any(_is_categorical(df[c]) for c in wanted if c != time) else pd.Series(False, index=df.index)
    drop = na_mask | na_cov | na_cat
    n_dropped = int(drop.sum())
    if n_dropped:
        if na_action == "fail":
            raise ValueError(f"{n_dropped} rows with missing values (na_action='fail')")
        warnings.warn(f"{n_dropped} rows with missing values omitted", stacklevel=3)
        df = df[~drop]
    if len(df) == 0:
        raise ValueError("no observations left")

    codes, uniques = pd.factorize(df[id], sort=False)
    ns = len(uniques)
    id_map = {v: k + 1 for k, v in enumerate(uniques.tolist())}
    subj = codes.astype(np.int64)

    # --- column classification: predictor (varies within subject) vs covariate (constant) ---
    def varies(col: str) -> bool:
        """Return ``True`` if ``col`` changes within at least one subject."""
        return bool(df.groupby(codes)[col].nunique(dropna=False).max() > 1)

    x_names = [time]
    cov_names: list = []
    cat_names: list = []
    forced_pred = set(predictors or [])
    if builtin_x and builtin_x != time:
        forced_pred.add(builtin_x)
    if model.tte_start:
        forced_pred.add(model.tte_start)
    for c in wanted:
        if c == time:
            continue
        if _is_categorical(df[c]):
            if c in forced_pred:
                raise ValueError(f"{c!r}: a predictor cannot be categorical")
            if varies(c):
                raise ValueError(f"categorical covariate {c!r}: must be constant within each subject")
            cat_names.append(c)
        elif c in forced_pred:
            x_names.append(c)
        elif c in cov_model_cols or c == builtin_dose or not varies(c):
            if varies(c):
                raise ValueError(f"covariate {c!r}: must be constant within each subject "
                                 "(if it is a time-varying predictor, do not use it in 'covariates')")
            cov_names.append(c)
        else:
            x_names.append(c)

    # --- core ---
    cd = _core.Data(subj + 1, df[time].to_numpy(float), df[response_col].to_numpy(float))
    ytype_labels = group_labels = None
    if ytype:
        yc, yu = pd.factorize(df[ytype], sort=True)
        cd.set_ytype(yc.astype(np.int64)); ytype_labels = np.asarray(yu)
    if cens:
        cc = df[cens].astype(int).to_numpy(np.int64)
        if not np.isin(cc, (-1, 0, 1)).all():
            raise ValueError("cens: allowed values are -1 (left), 0 (observed), +1 (right)")
        cd.set_cens(cc)
    if limit:
        cd.set_limit(df[limit].to_numpy(float))
    if group:
        gc, gu = pd.factorize(df[group], sort=True)
        cd.set_group((gc + 1).astype(np.int64)); group_labels = np.asarray(gu)
    for c in x_names[1:]:
        cd.add_predictor(c, df[c].to_numpy(float))
    cov_values, cat_labels = {}, {}
    first = df.groupby(codes, sort=True)
    for c in cov_names:
        v = first[c].first().to_numpy(float)
        cov_values[c] = v
        cd.add_covariate(c, v)
    ref = dict(reference or {})
    for c in cat_names:
        lab = first[c].first().astype(str).to_numpy()
        cat_labels[c] = lab
        cd.add_categorical(c, [str(s) for s in lab], ref.get(c))
    if ev_df is not None and len(ev_df):
        cd.add_events(*_event_arrays(ev_df, id_map, id, time, ecols))
    cd.check()

    info = DataInfo(id_col=id, time_col=time, y_col=response_col, ids=np.asarray(uniques), row_index=df.index,
                    subj=subj, x_names=x_names, cov_names=cov_names, cov_values=cov_values,
                    cat_names=cat_names, cat_labels=cat_labels, cat_reference=ref,
                    ytype_labels=ytype_labels, group_labels=group_labels, n_dropped=n_dropped,
                    frame=df, events=ev_df, event_cols=ecols, cens_col=cens, limit_col=limit,
                    time_synthetic=time_synth, id_synthetic=id_synth)
    return cd, info
