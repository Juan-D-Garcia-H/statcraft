# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Quasi-Gaussian GLM.

A Gaussian mean model with the variance treated as an unknown dispersion
estimated from the Pearson residuals. The tiny constant added to the
zero expenditures avoids problems with the identity link on a
non-negative response.

Run from the repository root:

    python examples/glm/quasi_gaussian.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

medexp = data_loader("MedExp.csv", data_dir_override=str(DATA_DIR)).load()
medexp.loc[medexp["med"] == 0, "med"] += 1e-5

fit = scg.glm(
    "med ~ physlim + ndisease + linc + age + child",
    data=medexp,
    family=scg.QuasiGaussian(),
).fit()

print(fit.summary())
