# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- ordinal mixed model with estimated cut points.

The ``knee`` data record an ordinal pain score (0-4) over time under two
treatments.  A latent linear predictor is mapped through ordered cut points;
``c1`` is the first cut, and ``d2``-``d4`` are positive increments between the
remaining cuts (log-transformed so they stay ordered).  A random effect on
``c1`` lets the cut points shift between patients.

Run from the repository root:

    python examples/saem/knee_ordinal.py
"""

from pathlib import Path

import numpy as np

import statcraft as sf
from statcraft.datasets import data_loader
from statcraft.saem import Ordinal

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

knee = data_loader("knee", data_dir_override=str(DATA_DIR)).load().copy()
if knee["y"].min() == 1:                       # ensure categories 0..4
    knee["y"] = knee["y"] - 1


def eta_ordinal(b_time, b_treat, time, treatment):
    """Latent linear predictor."""
    return b_time * time + b_treat * treatment


def cortes_ordinales(c1, d2, d3, d4):
    """Ordered cut points built from the first cut and positive increments."""
    c2 = c1 + d2
    c3 = c2 + d3
    c4 = c3 + d4
    return (c1, c2, c3, c4)


fit = sf.SAEM(
    model=eta_ordinal,
    cuts=cortes_ordinales,
    data=knee,
    id="id",
    time="time",
    response="y",
    params=["b_time", "b_treat", "c1", "d2", "d3", "d4"],
    transform={"b_time": "identity", "b_treat": "identity",
               "c1": "identity", "d2": "log", "d3": "log", "d4": "log"},
    random=["c1"],
    family=Ordinal(),
    start={"b_time": -0.3, "b_treat": -0.5, "c1": -1.5, "d2": 1.0, "d3": 1.0, "d4": 1.0},
    start_sd={"c1": 1.5},
    seed=632545,
).fit(
    n_burn=300,
    n_iter=200,
    n_chains=1,
    se_method="linearization",
    ll_method="importance_sampling",
    is_samples=5000,
    n_threads=2,
    auto_stop=True,
)

print("Compilation:", fit.model.compile_info)
print(fit.summary())
