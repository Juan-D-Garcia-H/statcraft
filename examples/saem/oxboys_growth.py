# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- linear growth curves with a full random-effect covariance.

The ``oxboys`` data record the height of 26 boys in Oxford at nine occasions.
A linear growth model ``height = base + slope * age`` is fitted with
``omega="full"``, so the between-subject variability of the intercept and the
slope is estimated together with their correlation.

Run from the repository root:

    python examples/saem/oxboys_growth.py
"""

from pathlib import Path

import statcraft as sf
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

oxboys = data_loader("oxboys", data_dir_override=str(DATA_DIR)).load()
print(oxboys.head())


def linear_growth(base, slope, age):
    """Linear growth: height = base + slope * age."""
    return base + slope * age


fit = sf.SAEM(
    model=linear_growth,
    data=oxboys,
    id="Subject",
    time="age",
    response="height",
    params=["base", "slope"],
    transform="identity",
    omega="full",                                 # estimate the base/slope correlation
    error="constant",
    start={"base": 150.0, "slope": 6.0},
    start_sd={"base": 5.0, "slope": 1.0},
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
