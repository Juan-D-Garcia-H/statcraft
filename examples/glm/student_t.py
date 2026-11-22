# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""StudentT GLM -- robust linear regression.

Heavy-tailed errors make the fit less sensitive to outliers. Uses the
``longley`` macroeconomic dataset.

Run from the repository root:

    python examples/glm/student_t.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

longley = data_loader("longley.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "Employed ~ GNP",
    data=longley,
    family=scg.StudentT(),
).fit()

print(fit.summary())
