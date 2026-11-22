# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Nonlinear mixed-effects model (``nlme``) -- logistic growth curves.

The ``Orange`` dataset records the trunk circumference of five orange trees
as they grow.  A self-starting logistic curve ``SSlogis`` is fitted with a
tree-level random effect on the asymptote, so every tree gets its own growth
curve while the shape parameters are shared.  This is the canonical
``nlme::nlme`` example.

Run from the repository root:

    python examples/nlme/nlme.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()

model_nlme = nl.nlme(
    "circumference ~ SSlogis(age, Asym, xmid, scal)",
    orange,
    params="Asym + xmid + scal ~ 1",
    random="Asym ~ 1 | Tree",
)
result = model_nlme.fit()

print(result.summary())
