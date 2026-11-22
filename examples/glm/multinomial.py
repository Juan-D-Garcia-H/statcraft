# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Multinomial GLM -- baseline-category logit model.

Models the wine cultivar from alcohol and colour in the bundled
``wines`` dataset. The last level of the response is the reference
category (``-1`` removes the intercept for identification).

Run from the repository root:

    python examples/glm/multinomial.py
"""

from pathlib import Path

import pandas as pd

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

wines = data_loader("wines.csv", data_dir_override=str(DATA_DIR)).load()

# Put 'Barbera' last so it becomes the reference category.
category_order = ["Barolo", "Grignolino", "Barbera"]
wines["wine"] = pd.Categorical(
    wines["wine"], categories=category_order, ordered=False
)

fit = scg.glm(
    "wine ~ alcohol + colour - 1",
    data=wines,
    family=scg.Multinomial(),
).fit()

print(fit.summary())
