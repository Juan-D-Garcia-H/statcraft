# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Quasi-binomial GLM.

Same spam model as ``binomial.py``, but the dispersion is estimated from
the Pearson residuals instead of being fixed at 1. Use it when the
binary response shows over- or under-dispersion.

Run from the repository root:

    python examples/glm/quasi_binomial.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

email = data_loader("email.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "spam ~ to_multiple + format + attach + dollar + winner + inherit + password + re_subj",
    data=email,
    family=scg.QuasiBinomial(),
).fit()

print(fit.summary())
