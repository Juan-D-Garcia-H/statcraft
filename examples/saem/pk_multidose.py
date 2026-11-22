# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- repeated-dose PK with a separate dosing event table.

Concentrations after a loading dose plus maintenance doses are simulated, and
the dosing history is passed through ``events=`` as its own
``id, time, amt`` table.  ``doses.superpose`` evaluates the contribution of
every previous dose inside the structural model.

Run from the repository root:

    python examples/saem/pk_multidose.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statcraft as sf

rng = np.random.default_rng(632545)
n_subj = 20
obs_times = np.array([1.0, 4.0, 11.5, 13.0, 23.5, 37.0, 47.5, 61.0, 64.0, 72.0])
dose_times = np.array([0.0, 12.0, 24.0, 36.0, 48.0, 60.0])
dose_amts = np.array([200.0, 100.0, 100.0, 100.0, 100.0, 100.0])   # loading + maintenance

events_rows, obs_rows = [], []
for i in range(1, n_subj + 1):
    ka_i = 1.2 * np.exp(0.30 * rng.standard_normal())
    cl_i = 3.0 * np.exp(0.25 * rng.standard_normal())
    v_i = 30.0 * np.exp(0.20 * rng.standard_normal())
    k_i = cl_i / v_i

    for td, amt in zip(dose_times, dose_amts):
        events_rows.append({"id": i, "time": td, "amt": amt})

    for t in obs_times:
        past = dose_times <= t
        dt = t - dose_times[past]
        amts = dose_amts[past]
        conc_true = np.sum(amts / v_i * ka_i / (ka_i - k_i) * (np.exp(-k_i * dt) - np.exp(-ka_i * dt)))
        conc = max(conc_true * (1.0 + 0.10 * rng.standard_normal()), 1e-4)
        obs_rows.append({"id": i, "time": t, "conc": conc})

df_events = pd.DataFrame(events_rows)
df_obs = pd.DataFrame(obs_rows)


def pk1_multidosis(ka, CL, V, time, doses):
    k = CL / V
    return doses.superpose(
        time,
        lambda dt, amt: amt / V * ka / (ka - k) * (np.exp(-k * dt) - np.exp(-ka * dt)),
    )


fit = sf.SAEM(
    model=pk1_multidosis,
    data=df_obs,
    events=df_events,
    id="id",
    time="time",
    response="conc",
    params=["ka", "CL", "V"],
    transform="log",
    error="proportional",
    start={"ka": 1.0, "CL": 2.5, "V": 25.0},
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
