# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Linear mixed-effects model (``lme``) -- random intercept and slope.

The ``Orthodont`` dataset measures the distance from the pituitary to the
pterygomaxillary fissure for 27 children at four ages.  Growth is modelled
with a ``Sex`` by ``age`` interaction and a random intercept plus random
slope per child, the classic R ``nlme::lme`` example.

Run from the repository root:

    python examples/nlme/lme.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orth = data_loader("Orthodont.csv", data_dir_override=str(DATA_DIR)).load()

model_lme = nl.lme(
    "distance ~ age * Sex",
    orth,
    random="~ age | Subject",
)
result = model_lme.fit()

print(result.summary())
