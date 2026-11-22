# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Truncated Gaussian GLM.

Only respondents with a positive number of affairs are observed here, so
the Gaussian model is left-truncated at zero (``left_limit=0``).

Run from the repository root:

    python examples/glm/truncated_gaussian.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

affairs = data_loader("Affairs.csv", data_dir_override=str(DATA_DIR)).load()
positive = affairs[affairs["affairs"] > 0]

fit = scg.glm(
    "affairs ~ yearsmarried + rating",
    data=positive,
    family=scg.TruncatedGaussian(),
).fit(left_limit=0)

print(fit.summary())
