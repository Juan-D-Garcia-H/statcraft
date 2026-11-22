# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Quasi-InverseGaussian GLM.

Inverse-Gaussian mean structure for medical expenditure, with a free
dispersion estimated from the Pearson residuals.

Run from the repository root:

    python examples/glm/quasi_inverse_gaussian.py
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
    family=scg.QuasiInverseGaussian(),
).fit()

print(fit.summary())
