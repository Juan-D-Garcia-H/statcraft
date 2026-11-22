# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Beta GLM -- regression for proportions in (0, 1).

Models gasoline yield as a proportion of the input, using the bundled
``GasolineYield`` dataset. The precision parameter ``phi`` is modelled
through its own linear predictor (``phi_link="log"`` by default).

Run from the repository root:

    python examples/glm/beta.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

gasoline = data_loader("GasolineYield.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "yield ~ gravity + pressure + temp10 + temp + batch",
    data=gasoline,
    family=scg.Beta(),
).fit()

print(fit.summary())
