# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Value-inflated negative binomial GLM.

Combines an arbitrary inflation point with an over-dispersed negative
binomial count model.

Run from the repository root:

    python examples/glm/value_inflated_negative_binomial.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

bio = data_loader("bioChemists.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "art ~ fem + kid5 + ment | ment",
    data=bio,
    family=scg.ValueInflatedNegativeBinomial(target_value=0),
).fit()

print(fit.summary())
