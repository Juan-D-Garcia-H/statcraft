# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Ordinal GLM -- proportional-odds model.

Models the ordered college-application rating from parental education,
school type and GPA in the bundled ``ologit`` dataset.

Run from the repository root:

    python examples/glm/ordinal.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ologit = data_loader("ologit.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "apply ~ pared + public + gpa",
    data=ologit,
    family=scg.Ordinal(),
).fit()

print(fit.summary())
