# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Poisson GLM -- log-linear count regression.

Models the number of extramarital affairs from age, years married,
religiousness and marriage rating, using the bundled ``Affairs`` dataset.

Run from the repository root:

    python examples/glm/poisson.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

affairs = data_loader("Affairs.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "affairs ~ age + yearsmarried + religiousness + rating",
    data=affairs,
    family=scg.Poisson(),
).fit()

print(fit.summary())
