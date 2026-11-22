# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Tweedie GLM -- compound Poisson-Gamma regression.

Tweedie models are useful for semi-continuous responses: a point mass at
zero plus a positive, right-skewed continuous part (insurance claims,
health-care spending, ...). ``var_power=1.25`` sits between Poisson (1)
and Gamma (2).

Run from the repository root:

    python examples/glm/tweedie.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

claims = data_loader("AutoClaim.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "CLM_AMT5 ~ MVR_PTS + TRAVTIME + BLUEBOOK",
    data=claims,
    family=scg.Tweedie(var_power=1.25),
).fit()

print(fit.summary())
