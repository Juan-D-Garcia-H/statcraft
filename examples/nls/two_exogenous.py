# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``NLS`` -- two exogenous variables with the matrix/vector signature.

When the model function has the shape ``def f(X, p)``, ``X`` is the matrix of
exogenous columns and ``p`` the parameter vector.  With more than one column
the wrapper stacks them in the order given by ``predictors``; with two columns
the function receives an ``(n, 2)`` matrix.

Here a Cobb-Douglas-like yield surface is fitted to the ``Wheat`` experiment:

    DryMatter = a * (1 - exp(-b * Moisture)) * fertilizer ** c

Run from the repository root:

    python examples/nls/two_exogenous.py
"""

from pathlib import Path

import numpy as np

from statcraft.nls import NLS
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

wheat = data_loader("Wheat.csv", data_dir_override=str(DATA_DIR)).load()


def wheat_surface(X, p):
    """Yield from a two-column design matrix ``X = [Moisture, fertilizer]``."""
    a, b, c = p
    moisture = X[:, 0]
    fertilizer = X[:, 1]
    return a * (1.0 - np.exp(-b * moisture)) * fertilizer ** c


model = NLS(
    wheat_surface,
    wheat,
    response="DryMatter",
    predictors=["Moisture", "fertilizer"],
    start={"a": 10.0, "b": 0.1, "c": 0.3},
)
fit = model.fit()

print(fit.summary())
print("\nCoefficient table")
print(fit.coef_table().round(4))

new_data = wheat.iloc[[0, 10, 20, 30, 40]][["Moisture", "fertilizer"]]
print("\nPredicted vs observed dry matter")
print(
    new_data.assign(
        observed=wheat.iloc[[0, 10, 20, 30, 40]]["DryMatter"].to_numpy(),
        predicted=fit.predict(new_data).round(3),
    ).to_string(index=False)
)
