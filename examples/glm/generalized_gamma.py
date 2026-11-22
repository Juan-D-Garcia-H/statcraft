# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""GeneralizedGamma GLM -- flexible accelerated failure-time model.

Models survival time in the bundled ``lung`` dataset with three linear
predictors: location ``mu``, scale ``sigma`` and shape ``Q``. The main
formula models ``mu``; ``sigma_formula`` and ``Q_formula`` model the other
two, so the scale and shape can depend on different covariates.

The ``lung`` dataset codes ``status`` as 1 = censored and 2 = dead, while
the engine expects 1 = event and 0 = right-censored. After the recode,
``cens="status"`` names the column directly (no ``to_numpy()`` needed).

Run from the repository root:

    python examples/glm/generalized_gamma.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

lung = data_loader("lung.csv", data_dir_override=str(DATA_DIR)).load()
lung_clean = lung.dropna(subset=["time", "age", "sex", "ph_ecog", "status"]).copy()
# 1 = censored, 2 = dead -> 0 = censored, 1 = event.
lung_clean["status"] = lung_clean["status"] - 1

model_gengamma = scg.generalized_gamma(
    "time ~ sex + ph_ecog",
    sigma_formula="~ 1",
    Q_formula="~ age",
    data=lung_clean,
    minimizer="BFGS",
).fit(cens="status")

print(model_gengamma.summary())
