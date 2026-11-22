# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Quasi-Poisson GLM.

Count models often show over-dispersion relative to the Poisson
variance. ``QuasiPoisson`` keeps the Poisson mean structure but rescales
the standard errors by the estimated Pearson dispersion.

Run from the repository root:

    python examples/glm/quasi_poisson.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

affairs = data_loader("Affairs.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "affairs ~ yearsmarried + religiousness + rating",
    data=affairs,
    family=scg.QuasiPoisson(link="log"),
).fit()

print(fit.summary())
