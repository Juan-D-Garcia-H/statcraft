# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nlme``/``gnls`` -- nonlinear models written as Python functions.

``nlme`` and ``gnls`` accept a plain Python function instead of a formula.
The recommended portable form is ``f(x, p)``: ``x`` holds the columns named by
``predictors`` and ``p`` holds the parameter names from ``start`` (or from an
explicit ``params``). The older ``func(*exogenous, *parameters)`` form is still
supported (the soybean fit below uses it to show an analytical Jacobian).

The Orange fit below is the same model as ``examples/nlme/nlme.py`` (logistic
curve, tree-level random effect on ``Asym``), so the estimates must match the
``SSlogis`` formula fit exactly.  The soybean fit shows a callable ``gnls``
with a fixed-effect covariate and a user-supplied analytical Jacobian.

Run from the repository root:

    python examples/nlme/callable_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()
soybean = data_loader("Soybean.csv", data_dir_override=str(DATA_DIR)).load()


# ---------------------------------------------------------------------------
# nlme: logistic growth curve as a Python function
# ---------------------------------------------------------------------------
def logistic(x, p):
    """``Asym / (1 + exp((xmid - age) / scal))`` with the portable f(x, p) form."""
    age, = x
    Asym, xmid, scal = p
    return Asym / (1.0 + np.exp((xmid - age) / scal))


formula_fit = nl.nlme(
    "circumference ~ SSlogis(age, Asym, xmid, scal)",
    orange,
    params="Asym + xmid + scal ~ 1",
    random="Asym ~ 1 | Tree",
).fit()

callable_fit = nl.nlme(
    logistic,
    orange,
    response="circumference",
    predictors="age",
    random="Asym ~ 1 | Tree",
    start={"Asym": 190.0, "xmid": 730.0, "scal": 350.0},
).fit()

print(callable_fit.summary())

comparison = pd.DataFrame(
    {
        "SSlogis formula": formula_fit.fixef,
        "Python function": callable_fit.fixef,
    }
)
comparison["difference"] = comparison.iloc[:, 0] - comparison.iloc[:, 1]
print("\nFormula vs callable estimates")
print(comparison.round(6).to_string())
print(f"\nsame log-likelihood: {np.isclose(formula_fit.loglik, callable_fit.loglik)}")
print(f"callable model label: {callable_fit.formula}")

new_ages = pd.DataFrame({"age": [118.0, 484.0, 728.8, 1180.0, 1580.0]})
print("\nPredicted circumference (population level)")
print(callable_fit.predict(new_ages, level=0).round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# gnls: callable model with a fixed-effect covariate and analytical Jacobian
# ---------------------------------------------------------------------------
def logistic_soy(Time, Asym, xmid, scal):
    return Asym / (1.0 + np.exp((xmid - Time) / scal))


def logistic_soy_jac(Time, Asym, xmid, scal):
    """Derivatives with respect to (Asym, xmid, scal), one row per observation."""
    e = np.exp((xmid - Time) / scal)
    d_asym = 1.0 / (1.0 + e)
    d_xmid = -Asym * e / (scal * (1.0 + e) ** 2)
    d_scal = Asym * e * (xmid - Time) / (scal ** 2 * (1.0 + e) ** 2)
    return np.column_stack([d_asym, d_xmid, d_scal])


gnls_fit = nl.gnls(
    logistic_soy,
    soybean,
    response="weight",
    params={"Asym": "Variety", "xmid": "1", "scal": "1"},
    start={"Asym": 18.0, "xmid": 54.0, "scal": 8.0},
    jac=logistic_soy_jac,
).fit()

print("\n\nCallable gnls with Asym ~ Variety (analytical Jacobian)")
print(gnls_fit.summary())

print("\nPredicted final weight per variety (population level, Time = 70)")
per_variety = pd.DataFrame({"Time": [70.0, 70.0], "Variety": sorted(soybean["Variety"].unique())})
per_variety["predicted"] = gnls_fit.predict(per_variety).to_numpy()
print(per_variety.round(3).to_string(index=False))
