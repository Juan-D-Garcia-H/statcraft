# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""LogLogistic GLM -- accelerated failure-time model.

Log-logistic baseline for the lung-cancer survival data. ``sex`` is
declared categorical so the formula uses a factor contrast.

Run from the repository root:

    python examples/glm/loglogistic.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

lung = data_loader("lung.csv", data_dir_override=str(DATA_DIR)).load()
data = lung[["time", "age", "sex", "status"]].copy()
data["sex"] = data["sex"].astype("category")
data["status"] = data["status"].map({1: 0, 2: 1})

fit = scg.glm(
    "time ~ age + sex",
    data=data,
    family=scg.LogLogistic("identity"),
).fit(cens=data["status"].to_numpy())

print(fit.summary())
