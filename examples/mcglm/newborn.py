# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Longitudinal bivariate GLM with moving-average dependence (NewBorn).

Three neonatal responses (respiratory rate, heart rate and oxygen
saturation) are modelled jointly. ``RR`` and ``HR`` include a
first-order moving-average block per baby (``NBI``) to capture serial
dependence, while ``SPO2`` uses only an identity block.

Run from the repository root:

    python examples/mcglm/newborn.py
"""

from pathlib import Path

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

newborn = data_loader("NewBorn.csv", data_dir_override=str(DATA_DIR)).load().astype(
    {
        "Sex": "category",
        "PRE": "category",
        "HD": "category",
        "SUR": "category",
        "JAU": "category",
        "PNE": "category",
        "PDA": "category",
        "PPI": "category",
        "OTHERS": "category",
        "AUX": "category",
        "TREAT": "category",
        "NBI": "category",
    }
)

newborn = newborn.sort_values(["NBI", "TIME", "TREAT"]).reset_index(drop=True)
n = len(newborn)

z = [
    [scm.mc_id(n)] + scm.mc_ma(n, order=1, groups=newborn["NBI"]),
    [scm.mc_id(n)] + scm.mc_ma(n, order=1, groups=newborn["NBI"]),
    [scm.mc_id(n)],
]

fit = scm.mcglm(
    [
        "RR ~ PRE + SUR + PDA + OTHERS + AUX + TREAT",
        "HR ~ GA + BW + APGAR1M + APGAR5M + PRE + SUR + JAU + PNE + PDA + PPI + AUX",
        "SPO2 ~ GA + BW + APGAR1M + HD + PNE + PDA + PPI + TREAT",
    ],
    data=newborn,
    z=z,
    link=["identity", "identity", "logit"],
    variance=["constant", "constant", "binomialP"],
).fit()

print(fit.summary())
