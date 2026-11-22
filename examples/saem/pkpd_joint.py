# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- simultaneous PK/PD model with two response types.

A single structural function returns either the concentration (PK) or the
drug effect (PD) depending on a response-type column.  ``ytype="ytype"``
tells the engine to split the likelihood, and ``error=["constant",
"proportional"]`` assigns a different residual model to each response.
Responses are ordered alphabetically: ``PD`` -> 0, ``PK`` -> 1.

Run from the repository root:

    python examples/saem/pkpd_joint.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statcraft as sf

rng = np.random.default_rng(632545)
n_subj = 25
times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
dose_val = 100.0

rows = []
for i in range(1, n_subj + 1):
    ka_i = 1.5 * np.exp(0.25 * rng.standard_normal())
    cl_i = 3.0 * np.exp(0.20 * rng.standard_normal())
    v_i = 30.0 * np.exp(0.20 * rng.standard_normal())
    e0_i = 100.0 * np.exp(0.10 * rng.standard_normal())
    ic50_i = 1.5 * np.exp(0.25 * rng.standard_normal())
    imax = 0.85
    k_i = cl_i / v_i

    for t in times:
        conc = dose_val / v_i * ka_i / (ka_i - k_i) * (np.exp(-k_i * t) - np.exp(-ka_i * t))
        eff = e0_i * (1.0 - imax * conc / (ic50_i + conc))
        rows.append({"id": i, "time": t, "dose": dose_val, "ytype": "PK",
                     "y": max(conc * (1.0 + 0.10 * rng.standard_normal()), 1e-4)})
        rows.append({"id": i, "time": t, "dose": dose_val, "ytype": "PD",
                     "y": eff + 3.0 * rng.standard_normal()})

df_pkpd = pd.DataFrame(rows)


def model_pkpd(ka, CL, V, E0, Imax, IC50, time, dose, ytype):
    k = CL / V
    conc = dose * ka / (V * (ka - k)) * (np.exp(-k * time) - np.exp(-ka * time))
    eff = E0 * (1.0 - Imax * conc / (IC50 + conc))
    # ytype == 0 -> "PD", ytype == 1 -> "PK"
    return np.where(ytype == 0, eff, conc)


fit = sf.SAEM(
    model=model_pkpd,
    data=df_pkpd,
    id="id",
    time="time",
    response="y",
    ytype="ytype",
    params=["ka", "CL", "V", "E0", "Imax", "IC50"],
    transform={"ka": "log", "CL": "log", "V": "log",
               "E0": "log", "Imax": "logit", "IC50": "log"},
    random=["ka", "CL", "V", "E0", "IC50"],    # Imax has no random effect
    error=["constant", "proportional"],        # Error[0] for PD, Error[1] for PK
    start={"ka": 1.2, "CL": 2.5, "V": 25.0, "E0": 95.0, "Imax": 0.8, "IC50": 1.2},
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
