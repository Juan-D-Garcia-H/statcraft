# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Zero-inflated negative binomial GLM.

Same publication-count model as ``zero_inflated_poisson.py``, but the
count part allows extra over-dispersion through the negative binomial.

Run from the repository root:

    python examples/glm/zero_inflated_negative_binomial.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

bio = data_loader("bioChemists.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "art ~ fem + kid5 + ment | ment",
    data=bio,
    family=scg.ZeroInflatedNegativeBinomial(),
).fit()

print(fit.summary())
