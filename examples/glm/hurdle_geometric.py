# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Hurdle Geometric GLM -- two-part count model with a geometric positive part.

Models doctor visits: the hurdle part separates zero visits from any
visit, and the positive counts follow a geometric distribution.

Run from the repository root:

    python examples/glm/hurdle_geometric.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

doctor = data_loader("DoctorVisits.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "visits ~ gender + age + income + illness",
    data=doctor,
    family=scg.HurdleGeometric(),
).fit()

print(fit.summary())
