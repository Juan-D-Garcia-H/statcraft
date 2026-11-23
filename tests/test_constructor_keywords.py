# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Regression tests for the single set of callable constructor keywords."""

import numpy as np
import pandas as pd
import pytest

import statcraft as sf
import statcraft.nlme as nl
from statcraft.nls import NLS


def _nls_frame():
    t = np.linspace(0.0, 4.0, 30)
    return pd.DataFrame({"t": t, "y": 1.0 + 2.0 * np.exp(-0.7 * t)})


def _exponential(x, p):
    time, = x
    a, b = p
    return a + 2.0 * np.exp(-b * time)


def _nlme_frame():
    t = np.linspace(0.0, 3.0, 6)
    rows = []
    for group in range(3):
        rows.append(pd.DataFrame({
            "t": t,
            "y": 4.0 * (1.0 - np.exp(-0.8 * t)) + np.array([0.0, 0.1, -0.1, 0.0, 0.05, -0.05]),
            "group": f"g{group}",
        }))
    return pd.concat(rows, ignore_index=True)


def _growth(x, p):
    time, = x
    a, b = p
    return a * (1.0 - np.exp(-b * time))


def _saem_frame():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    rows = []
    for subject in range(3):
        rows.append(pd.DataFrame({
            "id": subject + 1,
            "time": t,
            "y": 1.0 + 0.5 * t + np.array([0.0, 0.1, -0.1, 0.0]),
        }))
    return pd.concat(rows, ignore_index=True)


def _linear(x, p):
    time, = x
    a, b = p
    return a + b * time


def test_nls_uses_response_and_predictors():
    fit = NLS(
        _exponential, _nls_frame(),
        response="y", predictors="t", start={"a": 0.5, "b": 0.5},
    ).fit()

    assert np.allclose(fit.params, [1.0, 0.7], atol=0.05)


def test_nls_rejects_removed_keywords():
    df = _nls_frame()

    with pytest.raises(TypeError, match="y"):
        NLS(_exponential, df, y="y", predictors="t", start={"a": 1.0, "b": 1.0})
    with pytest.raises(TypeError, match="exog_var"):
        NLS(_exponential, df, response="y", exog_var="t", start={"a": 1.0, "b": 1.0})


def test_nlme_uses_params_predictors_and_response():
    fit = nl.NLME(
        _growth, _nlme_frame(),
        params={"a": "1", "b": "1"},
        predictors="t",
        response="y",
        random="a ~ 1 | group",
        start={"a": 4.0, "b": 1.0},
    ).fit()

    assert len(fit.fixef) == 2
    assert np.all(np.isfinite(np.asarray(fit.fixef, dtype=float)))


def test_nlme_rejects_removed_keywords():
    df = _nlme_frame()

    with pytest.raises(TypeError, match="fixed"):
        nl.NLME(_growth, df, fixed={"a": "1", "b": "1"}, random="a ~ 1 | group")
    with pytest.raises(TypeError, match="y"):
        nl.NLME(_growth, df, params={"a": "1", "b": "1"}, y="y", random="a ~ 1 | group")
    with pytest.raises(TypeError, match="exog_var"):
        nl.NLME(_growth, df, params={"a": "1", "b": "1"}, exog_var="t", random="a ~ 1 | group")


def test_lme_keeps_fixed_as_the_full_formula():
    model = nl.lme("y ~ t", _nlme_frame(), random="~ 1 | group")

    assert model.formula == "y ~ t"


def test_saem_uses_response_and_predictors():
    fit = sf.SAEM(
        _linear, _saem_frame(),
        id="id", time="time", response="y",
        params=["a", "b"], predictors=["time"],
        start={"a": 0.5, "b": 0.1},
    ).fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)

    assert np.all(np.isfinite(np.asarray(fit.fixef(), dtype=float)))


def test_saem_rejects_removed_keywords():
    df = _saem_frame()

    with pytest.raises(TypeError, match="y"):
        sf.SAEM(_linear, df, id="id", time="time", y="y", params=["a", "b"])
    with pytest.raises(TypeError, match="exog_var"):
        sf.SAEM(_linear, df, id="id", time="time", response="y",
                params=["a", "b"], exog_var=["time"])
