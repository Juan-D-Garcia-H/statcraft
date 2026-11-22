# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Multivariate GLM for four soil chemistry responses.

``PHWATER``, ``CA``, ``MG`` and ``K`` are modelled jointly from the
spatial coordinates and the sand/clay content. Each response is given
its own identity block in ``z``.

Run from the repository root:

    python examples/mcglm/soil.py
"""

from pathlib import Path

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

soil = data_loader("soil.csv", data_dir_override=str(DATA_DIR)).load()

n = soil.shape[0]

fit = scm.mcglm(
    [
        "PHWATER ~ COORD_X + SAND + CLAY",
        "CA ~ COORD_X + SAND + CLAY",
        "MG ~ COORD_X",
        "K ~ COORD_X + CLAY",
    ],
    data=soil,
    z=[
        [scm.mc_id(n)],
        [scm.mc_id(n)],
        [scm.mc_id(n)],
        [scm.mc_id(n)],
    ],
    link=["identity", "log", "log", "log"],
    variance=["constant"] * 4,
).fit()

print(fit.summary())
