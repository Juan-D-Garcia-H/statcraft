# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Hurdle negative binomial GLM -- two-part over-dispersed count model.

Adds a negative binomial count part to the hurdle structure, so the
positive counts can be over-dispersed.

Run from the repository root:

    python examples/glm/hurdle_negative_binomial.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

rec = data_loader("RecreationDemand.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "trips ~ ski + income + userfee + costS + costH | quality + costS + costH",
    data=rec,
    family=scg.HurdleNegativeBinomial(),
).fit()

print(fit.summary())
