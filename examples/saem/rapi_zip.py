# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- zero-inflated Poisson counts with a covariate.

The ``rapi`` data are counts measured over time.  A ZIP model combines a
Poisson mean with a zero-inflation probability, both depending on time; the
effect of ``gender`` enters only through a covariate equation on ``b0``.
``random=["b0", "p0"]`` gives patient-level variability on both parts.

Run from the repository root:

    python examples/saem/rapi_zip.py
"""

from pathlib import Path

import numpy as np

import statcraft as sf
from statcraft.datasets import data_loader
from statcraft.saem import ZeroInflatedPoisson

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

rapi = data_loader("rapi", data_dir_override=str(DATA_DIR)).load()
print(rapi.head())


def model_zip(b0, b1, p0, p1, time):
    """Return (Poisson mean, zero-inflation probability)."""
    lam = np.exp(b0 + b1 * time)
    pi = 1.0 / (1.0 + np.exp(-(p0 + p1 * time)))
    return lam, pi


fit = sf.SAEM(
    model=model_zip,
    data=rapi,
    id="id",
    time="time",
    response="rapi",
    params=["b0", "b1", "p0", "p1"],
    transform="identity",
    random=["b0", "p0"],
    omega="full",
    covariates=["b0 ~ gender"],
    family=ZeroInflatedPoisson(),
    start={"b0": 1.8, "b1": -0.1, "p0": -1.5, "p1": 0.2},
    start_sd={"b0": 0.8, "p0": 1.5},
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
