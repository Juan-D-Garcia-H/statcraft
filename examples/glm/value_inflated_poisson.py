# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Value-inflated Poisson GLM.

Like the zero-inflated model, but the extra probability mass can sit at
any value (here ``target_value=0``), not only at zero.

Run from the repository root:

    python examples/glm/value_inflated_poisson.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

bio = data_loader("bioChemists.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "art ~ fem + kid5 + ment | ment",
    data=bio,
    family=scg.ValueInflatedPoisson(target_value=0),
).fit()

print(fit.summary())
