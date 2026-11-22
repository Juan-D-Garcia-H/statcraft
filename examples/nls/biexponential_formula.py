# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- explicit biexponential formula with starting values and bounds.

Self-starting functions cover the common shapes, but any expression can be
written directly.  The ``Cefamandole`` data are a two-compartment concentration
curve, modelled here as the sum of two exponentials.  Rates are parameterised
in log scale (``exp(lrc)``) so they stay positive, and the amplitudes are
bounded below at zero.

Run from the repository root:

    python examples/nls/biexponential_formula.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from statcraft.nls import nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

cef = data_loader("Cefamandole.csv", data_dir_override=str(DATA_DIR)).load()

formula = "conc ~ A1 * exp(-exp(lrc1) * Time) + A2 * exp(-exp(lrc2) * Time)"

model = nls(
    formula,
    cef,
    start={"A1": 300.0, "lrc1": -2.0, "A2": 90.0, "lrc2": -4.0},
)

# Amplitudes in (0, 1000); log-rates in (-10, 2).
fit = model.fit(bounds=([0.0, -10.0, 0.0, -10.0], [1000.0, 2.0, 1000.0, 2.0]))

print(fit.summary())
print("\nCoefficient table")
print(fit.coef_table().round(4))

print("\nNatural-scale estimates")
print(f"fast phase: A1 = {fit.coefficients['A1']:.2f}"
      f"  rate = {np.exp(fit.coefficients['lrc1']):.4f}")
print(f"slow phase: A2 = {fit.coefficients['A2']:.2f}"
      f"  rate = {np.exp(fit.coefficients['lrc2']):.4f}")

new_times = pd.DataFrame({"Time": [1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 240.0]})
print("\nPredicted concentrations (95% confidence intervals)")
print(fit.predict(new_times, interval="confidence").round(2))
