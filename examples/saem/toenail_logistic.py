# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- logistic mixed model for a binary toenail outcome.

The ``toenail`` trial follows patients over time with a binary response
(onychomycosis severity).  The structural function is the logit linear
predictor; ``family=Binomial("logit")`` maps it to the Bernoulli likelihood.
In discrete models only ``beta0`` carries a random effect.

Run from the repository root:

    python examples/saem/toenail_logistic.py
"""

from pathlib import Path

import statcraft as sf
from statcraft.datasets import data_loader
from statcraft.saem import Binomial

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

toenail = data_loader("toenail", data_dir_override=str(DATA_DIR)).load()
print(toenail.head())


def model_logistic(beta0, beta1, beta2, time, treatment):
    """Logit-scale linear predictor (no inverse link here)."""
    return beta0 + beta1 * time + beta2 * treatment * time


fit = sf.SAEM(
    model=model_logistic,
    data=toenail,
    id="id",
    time="time",
    response="y",
    params=["beta0", "beta1", "beta2"],
    transform="identity",                         # parameters live on the logit scale
    random=["beta0"],                             # random intercept per patient
    family=Binomial("logit"),
    start={"beta0": -1.0, "beta1": -0.3, "beta2": -0.1},
    start_sd={"beta0": 3.0},
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
