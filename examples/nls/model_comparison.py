# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- comparing self-starting growth models with AIC/BIC and CV.

Three built-in self-starting functions are fitted to the same soybean growth
data: logistic (``SSlogis``), Gompertz (``SSgompertz``) and Weibull
(``SSweibull``).  AIC/BIC rank the models on the training data and
``cross_validate`` refits each one on five folds to estimate out-of-sample
performance.

Run from the repository root:

    python examples/nls/model_comparison.py
"""

from pathlib import Path

import pandas as pd

from statcraft.nls import nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

soybean = data_loader("Soybean.csv", data_dir_override=str(DATA_DIR)).load()

models = {
    "logistic": "weight ~ SSlogis(Time, Asym, xmid, scal)",
    "gompertz": "weight ~ SSgompertz(Time, Asym, b2, b3)",
    "weibull": "weight ~ SSweibull(Time, Asym, Drop, lrc, pwr)",
}

rows = []
for name, formula in models.items():
    fit = nls(formula, soybean).fit()
    cv = fit.cross_validate(k=5, random_state=42)
    rows.append({
        "model": name,
        "params": len(fit.coefficients),
        "R2": fit.rsquared,
        "RMSE": fit.rmse,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "CV R2": cv["r2_mean"],
        "CV RMSE": cv["rmse_mean"],
    })

table = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
print("Model comparison on the soybean growth data")
print(table.round(4).to_string(index=False))
print("\nLower AIC/BIC and higher CV R2 indicate a better trade-off.")
