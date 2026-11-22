# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- self-starting logistic growth curve for the Orange trees.

``SSlogis`` is one of the built-in self-starting functions: the formula is
translated to ``Asym / (1 + exp((xmid - age) / scal))`` and the initial values
are derived from the data, so no ``start`` dict is needed.  The fitted curve is
then used to build confidence and prediction bands for new ages.

Run from the repository root:

    python examples/nls/logistic_growth.py
"""

from pathlib import Path

import pandas as pd

from statcraft.nls import nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()

model = nls("circumference ~ SSlogis(age, Asym, xmid, scal)", orange)
print("Starting values chosen by Initializer:", model.start)

fit = model.fit()
print(fit.summary())
print("\nCoefficient table")
print(fit.coef_table().round(4))

print("\nGoodness of fit")
print(f"R2 = {fit.rsquared:.4f}  adj-R2 = {fit.adj_rsquared:.4f}")
print(
    f"RMSE = {fit.rmse:.2f}  MAE = {fit.mae:.2f}  "
    f"AIC = {fit.aic:.2f}  BIC = {fit.bic:.2f}"
)

new_ages = pd.DataFrame({"age": [118.0, 484.0, 728.8, 1180.0, 1580.0]})
print("\nPredicted circumference with 95% intervals")
print("confidence:\n", fit.predict(new_ages, interval="confidence").round(2))
print("prediction:\n", fit.predict(new_ages, interval="prediction").round(2))
