# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SkewNormal GLM -- linear regression with skewed errors.

Models athletes' BMI from body fat and sex in the bundled ``ais``
dataset, allowing the residuals to be asymmetric.

Run from the repository root:

    python examples/glm/skew_normal.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ais = data_loader("ais.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "BMI ~ Bfat + sex",
    data=ais,
    family=scg.SkewNormal(),
    minimizer="BFGS",
).fit()

print(fit.summary())
