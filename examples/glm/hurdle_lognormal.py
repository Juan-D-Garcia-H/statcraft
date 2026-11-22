# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Hurdle LogNormal GLM -- two-part log-normal model.

The positive part is log-normal with an identity link; the hurdle
probability uses a logit link. ``standardize=False`` keeps the supplied
scale of the predictors.

Run from the repository root:

    python examples/glm/hurdle_lognormal.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

medexp = data_loader("MedExp.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "med ~ health + linc + age",
    data=medexp,
    family=scg.HurdleLogNormal("identity", zero_link="logit"),
    minimizer="BFGS",
    standardize=False,
).fit()

print(fit.summary())
