# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Generalized nonlinear least squares (``gnls``) -- no random effects.

The ``Soybean`` dataset follows the weight of 48 soybean plots over time.
Here a single logistic growth curve is fitted to all plots together, which is
the canonical ``nlme::gnls`` example.  ``gnls`` has no random effects, but it
can still model heteroscedasticity and correlated errors through ``weights``
and ``correlation``.

``result.plot()`` draws the GNLS diagnostic mosaic: observed vs fitted with an
identity line and LOESS calibration curve, a Q-Q with 95% envelope, normalised
residuals and `√|r̂|` against ``Time`` (the primary structural variable) on
independent data-scaled axes, and a residual ACF computed within ``Plot``.
Normalised residuals use the full observation scale ``σ̂ · g(μ̂ᵢ, δ)``, so
variance functions such as ``varPower`` leave the Q-Q on the N(0, 1) scale. The
individual kinds are `observed_vs_fitted`, `qq_plot_with_envelope`,
`normalized_residuals_and_spread` and `grouped_acf`.

Run from the repository root:

    python examples/nlme/gnls.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

soybean = data_loader("Soybean.csv", data_dir_override=str(DATA_DIR)).load()

model_gnls = nl.gnls(
    "weight ~ SSlogis(Time, Asym, xmid, scal)",
    soybean,
    params="Asym + xmid + scal ~ 1",
)
result = model_gnls.fit()

print(result.summary())
