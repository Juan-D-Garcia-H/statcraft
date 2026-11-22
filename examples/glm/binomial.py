# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Binomial GLM -- logistic regression.

Models the probability that an email is spam from its headers and body
features, using the bundled ``email`` dataset.

Run from the repository root:

    python examples/glm/binomial.py
"""

from pathlib import Path

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

email = data_loader("email.csv", data_dir_override=str(DATA_DIR)).load()

fit = scg.glm(
    "spam ~ to_multiple + format + attach + dollar + winner + inherit + password + re_subj",
    data=email,
    family=scg.Binomial(),
).fit()

print(fit.summary())
