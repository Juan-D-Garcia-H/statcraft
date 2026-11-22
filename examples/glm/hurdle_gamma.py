# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Hurdle Gamma GLM -- two-part positive continuous model.

The hurdle part models whether medical expenditure is positive and the
second part is a Gamma model for the positive amounts.

Run from the repository root:

    python examples/glm/hurdle_gamma.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

medexp = data_loader("MedExp.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "med ~ health + linc + age",
    data=medexp,
    family=scg.HurdleGamma(),
).fit()

print(fit.summary())
