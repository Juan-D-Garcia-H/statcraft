# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- Weibull time-to-event model (``family=Weibull()``).

The ``lung`` data come from a lung-cancer trial.  ``status`` is recoded to
0 = censored / 1 = event and ``sex`` to 0 = male / 1 = female, then a Weibull
hazard is fitted with a subject-level random effect on the scale parameter
and a sex effect on ``lam``.

Run from the repository root:

    python examples/saem/lung_weibull.py
"""

from pathlib import Path

import statcraft as sf
from statcraft.datasets import data_loader
from statcraft.saem import Weibull

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

lung = data_loader("lung", data_dir_override=str(DATA_DIR)).load()

# 1 = censored, 2 = dead -> 0 = censored, 1 = event
if lung["status"].max() == 2:
    lung["status"] = lung["status"] - 1
# 1 = male, 2 = female -> 0 = male (reference), 1 = female
if lung["sex"].max() == 2:
    lung["sex"] = lung["sex"] - 1


def weibull_hazard(lam, k, time):
    return (k / lam) * ((time / lam) ** (k - 1.0))


def weibull_cumhazard(lam, k, time):
    return (time / lam) ** k


fit = sf.SAEM(
    model=weibull_hazard,
    cumhazard=weibull_cumhazard,
    data=lung,
    time="time",
    response="status",
    params=["lam", "k"],
    transform="log",
    random=["lam"],
    covariates=["lam ~ sex"],
    family=Weibull(),
    start={"lam": 400.0, "k": 1.2},
    start_sd={"lam": 0.2},
    seed=632545,
).fit()

print(fit.summary())
