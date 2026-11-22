# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- saving and restoring fitted models.

Formula-based models can be stored as plain JSON with ``save()``/``load()``
(or ``to_dict()``/``from_dict()``).  Models built from arbitrary Python
callables (the ``NLS`` wrapper, ``extra_functions``) cannot be represented as
JSON, but they are picklable, so ``pickle``/``joblib`` can be used instead.

Run from the repository root:

    python examples/nls/persistence.py
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np

from statcraft.nls import NLS, nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def michaelis_menten(conc, Vm, K):
    return Vm * conc / (K + conc)


orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()
muscle = data_loader("Muscle.csv", data_dir_override=str(DATA_DIR)).load()

formula_fit = nls(
    "circumference ~ SSlogis(age, Asym, xmid, scal)",
    orange,
).fit()

wrapped_fit = NLS(
    michaelis_menten,
    muscle,
    response="length",
    predictors="conc",
    start={"Vm": 200.0, "K": 0.1},
).fit()

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "orange_nls.json"
    formula_fit.save(str(path))
    restored = nls.load(str(path))

    print("JSON round trip (formula-based model)")
    print(f"  saved to        : {path}")
    print(f"  same formula    : {restored.formula == formula_fit.formula}")
    print(f"  same estimates  : {np.allclose(restored.params, formula_fit.params)}")
    print(f"  same R2         : {np.isclose(restored.rsquared, formula_fit.rsquared)}")
    print(f"  summary works   : {type(restored.summary()).__name__}")

blob = pickle.dumps(wrapped_fit)
unpickled = pickle.loads(blob)

print("\nPickle round trip (NLS wrapper)")
print(f"  same estimates  : {np.allclose(unpickled.params, wrapped_fit.params)}")
print(f"  parameters      : { {k: round(float(v), 4) for k, v in unpickled.coefficients.items()} }")
