# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Bivariate GLM with a random hunter effect and an offset (Hunting).

Both hunting outcomes (``OT`` and ``BD``) use log links and
Poisson-Tweedie variances, with ``HUNTER`` entering as a random
intercept and the log of the sampling effort as an offset.

Run from the repository root:

    python examples/mcglm/hunting.py
"""

from pathlib import Path

import numpy as np

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

hunting = data_loader("Hunting.csv", data_dir_override=str(DATA_DIR)).load().astype(
    {
        "ALT": "category",
        "SEX": "category",
        "METHOD": "category",
        "HUNTER": "category",
        "HUNTER_MONTH": "category",
    }
)

n = hunting.shape[0]
log_offset = np.log(hunting["OFFSET"].values)

z = [scm.mc_id(n)] + scm.mc_mixed(formula="~ 1 | HUNTER", data=hunting)

fit = scm.mcglm(
    [
        "OT ~ ALT + SEX",
        "BD ~ ALT + SEX + METHOD",
    ],
    data=hunting,
    z=z,
    link=["log", "log"],
    variance=["poisson_tweedie", "poisson_tweedie"],
    power_fixed=[False, False],
    offset=[log_offset, log_offset],
).fit(tuning=0.2, retry=(0.5, 0.2, 0.1), max_iter=150)

print(fit.summary())
