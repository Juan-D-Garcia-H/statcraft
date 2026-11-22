# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""LogNormal GLM -- accelerated failure-time model.

Models survival time in the bundled ``lung`` dataset. The ``status``
column is recoded to the engine's convention: 1 = event, 0 = censored.

Run from the repository root:

    python examples/glm/lognormal.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

lung = data_loader("lung.csv", data_dir_override=str(DATA_DIR)).load()
data = lung[["time", "age", "sex", "status"]].copy()
data["status"] = data["status"].map({1: 0, 2: 1})

fit = scg.glm(
    "time ~ age + sex",
    data=data,
    family=scg.LogNormal(),
).fit(cens=data["status"].to_numpy())

print(fit.summary())
