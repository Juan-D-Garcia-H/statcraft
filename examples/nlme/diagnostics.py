# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostic plots for mixed-effects and GLS fits (``statcraft.nlme``).

Two fits are drawn:

* ``lme`` on ``Orthodont`` -- the four-panel hierarchy mosaic (population vs.
  conditional, normalised residuals vs. conditional fit with a LOESS band,
  Q-Q with a 95% envelope, BLUP caterpillar with conditional intervals), plus
  the standalone binned-residual check;
* ``gls`` on ``Ovary`` with ``corAR1`` -- the four-panel GLS mosaic (observed
  vs fitted with LOESS, normalised Q-Q envelope, residual/spread against
  ``ŷ`` and within-group ACF) plus the empirical semi-variogram, which shows
  whether the AR(1) structure absorbs the serial dependence.

The figures are written to ``examples/nlme/figures/``.

Run from the repository root:

    python examples/nlme/diagnostics.py
"""

import os
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        'This example needs matplotlib. Install it with: pip install "statcraft[viz]"'
    ) from exc

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")

import numpy as np

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# lme: random intercept and slope per child
# ---------------------------------------------------------------------------
orth = data_loader("Orthodont.csv", data_dir_override=str(DATA_DIR)).load()
lme_fit = nl.lme("distance ~ age * Sex", orth, random="~ age | Subject").fit()

lme_fit.plot_diagnostics(save=OUT / "lme_diagnostics.png")
lme_fit.plot_binned_standardized_residuals(
    save=OUT / "lme_binned_standardized_residuals.png"
)
print(f"lme: wrote {OUT / 'lme_diagnostics.png'}")
print(f"lme: wrote {OUT / 'lme_binned_standardized_residuals.png'}")

# ---------------------------------------------------------------------------
# gls with AR(1) errors: the empirical variogram should flatten after lag 1
# ---------------------------------------------------------------------------
ovary = data_loader("Ovary.csv", data_dir_override=str(DATA_DIR)).load()
ovary["sin"] = np.sin(2 * np.pi * ovary["Time"])
ovary["cos"] = np.cos(2 * np.pi * ovary["Time"])
gls_fit = nl.gls(
    "follicles ~ sin + cos",
    ovary,
    correlation=nl.corAR1(form="~ 1 | Mare"),
    method="REML",
).fit()

gls_fit.plot_diagnostics(save=OUT / "gls_diagnostics.png")
gls_fit.plot_variogram(save=OUT / "gls_variogram.png")
print(f"gls: wrote {OUT / 'gls_diagnostics.png'}")
print(f"gls: wrote {OUT / 'gls_variogram.png'}")
