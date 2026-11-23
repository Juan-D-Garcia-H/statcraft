# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Regression tests for the shared ``def model(x, p)`` calling convention."""

import numpy as np
import pandas as pd

import statcraft as sf
import statcraft.nlme as nl
from statcraft.nls import NLS
from statcraft.utils.modelview import ModelView


def test_model_view_supports_unpacking_indexing_and_arrays():
    x = ModelView(["Time", "Dose"], [np.array([1.0, 2.0]), np.array([4.0, 5.0])],
                  label="x", columnar=True)
    p = ModelView(["ka", "CL"], [0.5, 2.0], label="p")

    time, dose = x
    ka, cl = p
    assert np.allclose(time, [1.0, 2.0])
    assert np.allclose(dose, [4.0, 5.0])
    assert (ka, cl) == (0.5, 2.0)
    assert x.Time is time
    assert x["Dose"] is dose
    assert p[-1] == 2.0
    assert np.asarray(p).shape == (2,)
    assert np.asarray(x).shape == (2, 2)
    assert np.allclose(x[:, 0], [1.0, 2.0])


def test_nls_accepts_canonical_two_argument_function():
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 5.0, 40)
    df = pd.DataFrame({"t": t, "y": 2.0 + 3.0 * np.exp(-0.5 * t) + rng.normal(0, 0.01, t.size)})

    def model(x, p):
        time, = x
        a, b = p
        return a + 3.0 * np.exp(-b * time)

    fit = NLS(model, df, response="y", predictors="t", start={"a": 1.0, "b": 1.0}).fit()
    assert np.allclose(fit.params, [2.0, 0.5], atol=0.05)


def test_nlme_accepts_canonical_two_argument_function():
    rng = np.random.default_rng(12)
    t = np.linspace(0.0, 4.0, 8)
    rows = []
    for group in range(4):
        amplitude = 5.0 + rng.normal(0, 0.2)
        rows.append(pd.DataFrame({
            "t": t,
            "y": amplitude * (1.0 - np.exp(-0.8 * t)) + rng.normal(0, 0.05, t.size),
            "group": [f"g{group}"] * t.size,
        }))
    df = pd.concat(rows, ignore_index=True)

    def model(x, p):
        time, = x
        a, b = p
        return a * (1.0 - np.exp(-b * time))

    fit = nl.nlme(
        model, df, response="y", predictors="t",
        params={"a": "1", "b": "1"}, random="a ~ 1 | group",
        start={"a": 4.0, "b": 1.0},
    ).fit()
    assert len(fit.fixef) == 2
    assert np.all(np.isfinite(np.asarray(fit.fixef, dtype=float)))


def test_saem_accepts_canonical_two_argument_function():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    rows = []
    for subject in range(3):
        rows.append(pd.DataFrame({
            "id": [subject + 1] * t.size,
            "time": t,
            "y": 1.0 + 0.5 * t + np.array([0.0, 0.1, -0.1, 0.0]),
        }))
    df = pd.concat(rows, ignore_index=True)

    def model(x, p):
        time, = x
        a, b = p
        return a + b * time

    fit = sf.SAEM(
        model, df, id="id", time="time", response="y",
        params=["a", "b"], predictors=["time"], omega="diagonal",
        start={"a": 0.5, "b": 0.1},
    ).fit(n_iter=5, n_burn=2, se_method="none", ll_method="none", seed=1)
    assert np.all(np.isfinite(np.asarray(fit.fixef(), dtype=float)))
