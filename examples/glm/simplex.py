# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Simplex GLM -- regression for proportions in (0, 1).

The simplex distribution is an alternative to the Beta for bounded
responses: it is parameterized by the mean and a dispersion ``sigma^2``
(``sigma_link="log"`` by default).

Run from the repository root:

    python examples/glm/simplex.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

gasoline = data_loader("GasolineYield.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "yield ~ gravity + pressure + temp10 + temp + batch",
    data=gasoline,
    family=scg.Simplex(),
).fit()

print(fit.summary())
