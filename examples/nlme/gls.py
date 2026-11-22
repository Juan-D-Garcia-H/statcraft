# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Generalized least squares (``gls``) with an AR(1) error structure.

The ``Ovary`` dataset counts ovarian follicles over a scaled time index in 11
mares.  Follicle counts follow a daily cycle, so the mean is modelled with a
sine/cosine pair (one harmonic of a 2*pi period) and the serial dependence
between successive observations of the same mare is modelled with a
first-order autoregressive correlation structure.  This is the classic
``nlme::gls`` example; unlike ``lme`` there are no random effects.

``result.plot()`` draws the GLS diagnostic mosaic: observed vs fitted with a
LOESS calibration curve, a normalised Q-Q with 95% envelope, normalised
residuals and `√|r̂|` against `ŷ` on independent data-scaled axes, and an ACF
accumulated within mares. The individual kinds are `observed_vs_fitted`,
`qq_plot_with_envelope`, `normalized_residuals_and_spread` and `grouped_acf`.

Run from the repository root:

    python examples/nlme/gls.py
"""

from pathlib import Path

import numpy as np

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

ovary = data_loader("Ovary.csv", data_dir_override=str(DATA_DIR)).load()
# One harmonic of the daily cycle (the formula parser does not evaluate
# sin/cos directly, so the two columns are precomputed).
ovary["sin"] = np.sin(2 * np.pi * ovary["Time"])
ovary["cos"] = np.cos(2 * np.pi * ovary["Time"])

model_gls = nl.gls(
    "follicles ~ sin + cos",
    ovary,
    correlation=nl.corAR1(form="~ 1 | Mare"),
    method="REML",
)
result = model_gls.fit()

print(result.summary())
