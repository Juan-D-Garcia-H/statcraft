# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- Emax dose-response with left-censored observations.

The ``PD1`` data are a dose-response study.  Responses at or below the limit
of quantification (LLOQ = 25) are marked with ``cens = -1``; the engine then
treats the recorded response as the censoring threshold.  An Emax model is
fitted with a log scale for ``EC50``.

Run from the repository root:

    python examples/saem/emax_censored.py
"""

from pathlib import Path

import numpy as np

import statcraft as sf
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

pd1 = data_loader("PD1", data_dir_override=str(DATA_DIR)).load()
LLOQ = 25.0
pd1["cens"] = np.where(pd1["response"] <= LLOQ, -1, 0)
pd1["y_cens"] = np.where(pd1["cens"] == -1, LLOQ, pd1["response"])


def model_emax(E0, Emax, EC50, dose):
    return E0 + Emax * dose / np.maximum(EC50 + dose, 1e-12)


fit = sf.SAEM(
    model=model_emax,
    data=pd1,
    id="subject",
    time="dose",
    response="y_cens",
    cens="cens",                               # -1 = left-censored at y_cens
    params=["E0", "Emax", "EC50"],
    transform={"E0": "identity", "Emax": "identity", "EC50": "log"},
    error="constant",
    start={"E0": 20.0, "Emax": 80.0, "EC50": 15.0},
    start_sd={"E0": 5.0, "Emax": 15.0, "EC50": 0.5},
    start_error=5.0,
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
