# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Zero-inflated Poisson GLM.

Models publication counts in the bundled ``bioChemists`` dataset. The
formula's two-part syntax ``count_part | zero_part`` gives the count
submodel and the zero-inflation submodel separately.

Run from the repository root:

    python examples/glm/zero_inflated_poisson.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

bio = data_loader("bioChemists.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "art ~ fem + kid5 + ment | ment",
    data=bio,
    family=scg.ZeroInflatedPoisson(),
).fit()

print(fit.summary())
