# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SkewT GLM -- regression with skewness and heavy tails.

Combines the skew-normal and Student-t error distributions, applied to
athletes' BMI in the bundled ``ais`` dataset.

Run from the repository root:

    python examples/glm/skew_t.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ais = data_loader("ais.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "BMI ~ Bfat + sex",
    data=ais,
    family=scg.SkewT(),
).fit()

print(fit.summary())
