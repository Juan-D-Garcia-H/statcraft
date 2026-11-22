# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""NegativeBinomial GLM -- over-dispersed count regression.

Uses the ``breslow`` epilepsy data. With ``NegativeBinomial()`` (no
``theta=``), the dispersion is estimated from the data in a two-step
fit; pass ``theta=<value>`` to keep it fixed instead.

Run from the repository root:

    python examples/glm/negative_binomial.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

breslow = data_loader("breslow.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "sumY ~ Base + Age + Trt",
    data=breslow,
    family=scg.NegativeBinomial(),
).fit()

print(fit.summary())
