# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Weibull GLM -- accelerated failure-time model.

Same lung-cancer survival data as ``lognormal.py``, with a Weibull
baseline and an identity link.

Run from the repository root:

    python examples/glm/weibull.py
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
    family=scg.Weibull("identity"),
).fit(cens=data["status"].to_numpy())

print(fit.summary())
