# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Hurdle Beta GLM -- two-part bounded continuous model.

The positive part is a Beta model for a share in (0, 1) and the hurdle
part models whether the share is positive. Here the expenditure share is
built by dividing ``med`` by ``exp(linc)``.

Run from the repository root:

    python examples/glm/hurdle_beta.py
"""

from pathlib import Path

import numpy as np

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

medexp = data_loader("MedExp.csv", data_dir_override=str(DATA_DIR)).load()

# Map expenditure to a valid share in [0, 1).
medexp["share"] = medexp["med"] / np.exp(medexp["linc"])
medexp = medexp[medexp["share"] < 1.0].copy()

fit = scg.glm(
    "share ~ health + linc + age",
    data=medexp,
    family=scg.HurdleBeta(),
).fit()

print(fit.summary())
