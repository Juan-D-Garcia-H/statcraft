# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``NLS`` -- Michaelis-Menten curve from a plain Python function.

``NLS`` is the statsmodels-style wrapper: instead of writing a formula, the
model is a normal ``def``. The portable ``f(x, p)`` signature keeps the same
shape as the number of columns and parameters grows. The same fit is also built
with the formula interface, which shows that both entry points are equivalent.

Run from the repository root:

    python examples/nls/michaelis_menten.py
"""

from pathlib import Path

import numpy as np

from statcraft.nls import NLS, nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

muscle = data_loader("Muscle.csv", data_dir_override=str(DATA_DIR)).load()


def michaelis_menten(x, p):
    """Length as a function of concentration: ``Vm * conc / (K + conc)``."""
    conc, = x
    Vm, K = p
    return Vm * conc / (K + conc)


model = NLS(
    michaelis_menten,
    muscle,
    response="length",
    predictors="conc",
    start={"Vm": 200.0, "K": 0.1},
)
fit = model.fit()

print(fit.summary())
print("\nCoefficient table")
print(fit.coef_table().round(4))

formula_fit = nls(
    "length ~ Vm * conc / (K + conc)",
    muscle,
    start={"Vm": 200.0, "K": 0.1},
).fit()

print("\nFormula interface gives the same estimates:",
      np.allclose(fit.params, formula_fit.params))
print(f"max |difference| = {np.max(np.abs(fit.params - formula_fit.params)):.3e}")
