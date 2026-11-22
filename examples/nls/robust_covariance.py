# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- heteroscedasticity-robust standard errors (HC0-HC3).

The classical NLS covariance assumes a constant error variance.  When that
assumption is doubtful, ``cov_type="HC0"..."HC3"`` computes a sandwich
covariance using the jacobian at the optimum as the design matrix (the same
idea as R's ``sandwich::vcovHC``).  HC3 gives each residual a stronger
leverage correction and is the most conservative choice.

Run from the repository root:

    python examples/nls/robust_covariance.py
"""

from pathlib import Path

import pandas as pd

from statcraft.nls import NLS
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

muscle = data_loader("Muscle.csv", data_dir_override=str(DATA_DIR)).load()


def michaelis_menten(conc, Vm, K):
    return Vm * conc / (K + conc)


rows = []
for cov_type in ("nonrobust", "HC0", "HC1", "HC2", "HC3"):
    fit = NLS(
        michaelis_menten,
        muscle,
        response="length",
        predictors="conc",
        start={"Vm": 200.0, "K": 0.1},
    ).fit(cov_type=cov_type)

    ci = fit.conf_int()
    rows.append({
        "cov_type": cov_type,
        "Vm": fit.coefficients["Vm"],
        "SE(Vm)": fit.std_errors[0],
        "Vm lower": ci.loc["Vm"].iloc[0],
        "Vm upper": ci.loc["Vm"].iloc[1],
        "K": fit.coefficients["K"],
        "SE(K)": fit.std_errors[1],
        "K lower": ci.loc["K"].iloc[0],
        "K upper": ci.loc["K"].iloc[1],
    })

table = pd.DataFrame(rows)
print("Same point estimates, different standard errors")
print(table.round(4).to_string(index=False))
print("\nHC3 is the most conservative; use it with small samples or when")
print("the residual variance grows with the fitted value.")
