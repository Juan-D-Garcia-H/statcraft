# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``lmer`` -- lme4-style formula syntax for crossed/nested random effects.

The ``sleepstudy`` data record reaction time for 18 subjects over ten days
of sleep deprivation.  The formula ``Reaction ~ Days + (Days | Subject)``
gives every subject a random intercept and a random slope, with the two
random effects allowed to be correlated (use ``(Days || Subject)`` for an
uncorrelated, diagonal covariance).

Run from the repository root:

    python examples/nlme/lmer.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

sleepstudy = data_loader("sleepstudy.csv", data_dir_override=str(DATA_DIR)).load()

result = nl.lmer(
    "Reaction ~ Days + (Days | Subject)",
    sleepstudy,
).fit()

print(result.summary())
