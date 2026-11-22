# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- two-level hierarchical model with hospital clusters.

Eight hospitals each contribute ten patients.  Each patient has an
exponential-decay response, and the model separates the variability between
hospitals (``cluster_omega``) from the variability between patients within a
hospital (``omega``).  ``group="hospital"`` declares the second level.

Run from the repository root:

    python examples/saem/hierarchical.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statcraft as sf

rng = np.random.default_rng(632545)
n_clusters = 8
subj_per_cluster = 10
times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])

A_pop, k_pop = 100.0, 0.25
sd_cluster_A, sd_cluster_k = 0.20, 0.15    # between hospitals
sd_subj_A, sd_subj_k = 0.15, 0.10          # between patients within a hospital
sigma_err = 2.0

rows = []
subj_id = 1
for c in range(1, n_clusters + 1):
    xi_A = sd_cluster_A * rng.standard_normal()
    xi_k = sd_cluster_k * rng.standard_normal()

    for _ in range(subj_per_cluster):
        A_i = A_pop * np.exp(xi_A + sd_subj_A * rng.standard_normal())
        k_i = k_pop * np.exp(xi_k + sd_subj_k * rng.standard_normal())

        for t in times:
            y_val = A_i * np.exp(-k_i * t) + sigma_err * rng.standard_normal()
            rows.append({"hospital": f"Hosp_{c}", "id": subj_id, "time": t, "y": y_val})
        subj_id += 1

df_hier = pd.DataFrame(rows)


def exp_decay(A, k, time):
    return A * np.exp(-k * time)


fit = sf.SAEM(
    model=exp_decay,
    data=df_hier,
    group="hospital",                          # level 2: hospital cluster
    id="id",                                   # level 1: patient
    time="time",
    response="y",
    params=["A", "k"],
    transform="log",
    error="constant",
    cluster_omega="diagonal",                  # hospital-level covariance
    omega="diagonal",                          # patient-level covariance
    start={"A": 90.0, "k": 0.2},
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())

print("\nRandom effects by hospital (xi):")
print(fit.cluster_ranef())
