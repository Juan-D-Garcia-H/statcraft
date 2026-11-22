# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Multivariate GLM for five health-care utilisation counts (AHS).

The Australian Health Survey responses (doctor and non-doctor visits,
admissions, hospital stays and medications) are fitted jointly with
log links and Poisson-Tweedie variance functions. ``power_fixed=False``
lets each response estimate its own Tweedie power.

Run from the repository root:

    python examples/mcglm/ahs.py
"""

from pathlib import Path

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ahs = data_loader("ahs.csv", data_dir_override=str(DATA_DIR)).load().astype(
    {
        "sex": "category",
        "chcond": "category",
    }
)

n = ahs.shape[0]

fit = scm.mcglm(
    [
        "Ndoc ~ sex + freepoor + illness + actdays + hscore",
        "Nndoc ~ age + freerepa + illness + actdays + hscore + chcond",
        "Nadm ~ income + illness + actdays + hscore + chcond",
        "Nhosp ~ age + income + actdays + chcond",
        "Nmed ~ sex + age + levyplus + illness + actdays + hscore + chcond",
    ],
    data=ahs,
    z=[scm.mc_id(n)],
    link=["log"] * 5,
    variance=["poisson_tweedie"] * 5,
    power_fixed=[False] * 5,
).fit()

print(fit.summary())
