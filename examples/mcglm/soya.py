# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Multivariate GLM for three associated responses (soya beans).

``grain``, ``seeds`` and the viable-pea proportion are modelled jointly.
Each response gets its own link and variance function, and the binomial
proportion receives ``ntrial`` (the number of peas per sample).

Run from the repository root:

    python examples/mcglm/soya.py
"""

from pathlib import Path

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

soya = data_loader("soya.csv", data_dir_override=str(DATA_DIR)).load().astype(
    {
        "block": "category",
        "water": "category",
        "pot": "category",
    }
)

soya["viable_peas"] = soya["viablepeas"] / soya["totalpeas"]
n = soya.shape[0]

fit = scm.mcglm(
    [
        "grain ~ block + water * pot",
        "seeds ~ block + water * pot",
        "viable_peas ~ block + water * pot",
    ],
    data=soya,
    z=[scm.mc_id(n)],
    link=["identity", "log", "logit"],
    variance=["constant", "tweedie", "binomialP"],
    ntrial=[None, None, soya["totalpeas"].values],
).fit()

print(fit.summary())
