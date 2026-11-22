# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- robust loss functions against contaminated observations.

Five early Orange measurements are replaced by implausible values.  With the
default ``loss="linear"`` the outliers pull the inflection point below zero;
the soft-L1 and Huber losses down-weight them and recover parameters close to
the clean fit.  R² is still computed on all observations, so it looks worse for
the robust fits: the point here is parameter recovery, not the raw R².

Run from the repository root:

    python examples/nls/robust_loss.py
"""

from pathlib import Path

import pandas as pd

from statcraft.nls import nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()
orange["circumference"] = orange["circumference"].astype(float)

formula = "circumference ~ SSlogis(age, Asym, xmid, scal)"

clean = nls(formula, orange).fit()

contaminated = orange.copy()
contaminated.loc[[0, 1, 2, 3, 4], "circumference"] = [500.0, 480.0, 520.0, 490.0, 510.0]

rows = []
rows.append({
    "data": "clean",
    "loss": "linear",
    **{name: value for name, value in clean.coefficients.items()},
    "R2": clean.rsquared,
})

for loss in ("linear", "soft_l1", "huber"):
    fit = nls(formula, contaminated).fit(loss=loss)
    rows.append({
        "data": "contaminated",
        "loss": loss,
        **{name: value for name, value in fit.coefficients.items()},
        "R2": fit.rsquared,
    })

table = pd.DataFrame(rows)[["data", "loss", "Asym", "xmid", "scal", "R2"]]
print("Parameter estimates under different loss functions")
print(table.round(3).to_string(index=False))
print("\nThe clean fit is the reference: Asym ~ 192.7, xmid ~ 728.8, scal ~ 353.5")
