# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Linear mixed model via MCGLM (sleepstudy).

Reproduces a random-intercept plus random-slope model for reaction time
on days of sleep deprivation, with ``Subject`` as the grouping factor.
The ``z`` list combines an identity block with the two ``mc_mixed``
random-effect blocks.

Run from the repository root:

    python examples/mcglm/sleepstudy.py
"""

from pathlib import Path

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

sleepstudy = data_loader("sleepstudy.csv", data_dir_override=str(DATA_DIR)).load().astype(
    {"Subject": "category"}
)

n = sleepstudy.shape[0]

z = (
    [scm.mc_id(n)]
    + scm.mc_mixed(formula="~ 1 | Subject", data=sleepstudy)
    + scm.mc_mixed(formula="~ 0 + Days | Subject", data=sleepstudy)
)

fit = scm.mcglm(
    ["Reaction ~ Days"],
    data=sleepstudy,
    z=z,
    link="identity",
    variance="constant",
).fit()

print(fit.summary())
