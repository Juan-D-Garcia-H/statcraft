# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Gaussian GLM -- ordinary linear regression.

Fits state life expectancy on population, murder rate, high-school
graduation and frost, using the bundled ``state_x77`` dataset.

Run from the repository root:

    python examples/glm/gaussian.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

state = data_loader("state_x77.csv", data_dir_override=str(DATA_DIR)).load()
state["Density_Population"] = state["Population"] / state["Area"]

# family=None means Gaussian(link="identity").
fit = scg.glm(
    "Life_Exp ~ Population + Murder + HS_Grad + Frost",
    data=state,
).fit()

print(fit.summary())
