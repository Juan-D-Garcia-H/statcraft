# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- one-compartment oral PK model with centred covariates.

The ``theo`` data are the classic Theophylline concentration measurements.
Weight is centred at 70 kg so the typical values correspond to a 70 kg
patient; clearance additionally depends on weight and sex. The model uses the
portable ``f(x, p)`` signature, with ``predictors`` defining the order of the
``x`` view. This is the standard first example for ``statcraft.SAEM``.

Run from the repository root:

    python examples/saem/theo_covariates.py
"""

from pathlib import Path

import numpy as np

import statcraft as sf
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

theo = data_loader("theo", data_dir_override=str(DATA_DIR)).load()
theo["cWeight"] = theo["Weight"] - 70.0          # centred at 70 kg
# Optional allometric version: theo["lWeight"] = np.log(theo["Weight"] / 70.0)


def model1cpt(x, p):
    """One-compartment oral absorption model with the portable f(x, p) form."""
    Time, Dose = x
    ka, CL, V = p
    k = CL / V
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))


fit = sf.SAEM(
    model=model1cpt,
    data=theo,
    id="Id",
    time="Time",
    response="Concentration",
    params=["ka", "CL", "V"],
    predictors=["Time", "Dose"],
    transform="log",
    error="constant",
    covariates=["ka ~ cWeight", "CL ~ cWeight + Sex"],
    omega="diagonal",
    start={"ka": 1.5, "CL": 2.8, "V": 32.0},     # typical values at 70 kg
    start_sd={"ka": 0.5, "CL": 0.5, "V": 0.5},
    seed=632545,
).fit()

print(fit.summary())
