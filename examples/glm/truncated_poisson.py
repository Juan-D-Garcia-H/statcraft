# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Truncated Poisson GLM.

When the data only contain positive counts (zero visits are unobserved),
the likelihood must be conditional on ``y >= 1``. ``left_limit=1``
declares that truncation point.

Run from the repository root:

    python examples/glm/truncated_poisson.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

doctor = data_loader("DoctorVisits.csv", data_dir_override=str(DATA_DIR)).load()
positive = doctor[doctor["visits"] > 0].copy()

fit = scg.glm(
    "visits ~ gender + age + income + illness",
    data=positive,
    family=scg.TruncatedPoisson(),
).fit(left_limit=1)

print(fit.summary())
