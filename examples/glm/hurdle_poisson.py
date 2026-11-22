# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Hurdle Poisson GLM -- two-part count model.

The hurdle part models ``P(y > 0)`` and the count part is a zero-truncated
Poisson for the positive counts. The formula's ``count | hurdle`` syntax
assigns a separate linear predictor to each part.

Run from the repository root:

    python examples/glm/hurdle_poisson.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

rec = data_loader("RecreationDemand.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "trips ~ ski + income + userfee + costS + costH | quality + costS + costH",
    data=rec,
    family=scg.HurdlePoisson(),
).fit()

print(fit.summary())
