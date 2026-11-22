# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostic plots for a single nonlinear curve (``statcraft.nls``).

Fits the self-starting logistic growth curve to the ``Orange`` trees and draws
the diagnostics that matter for a nonlinear least-squares fit: the fitted
curve with delta-method confidence and prediction bands, residuals against the
predictor, Cook's distance with the influential observations labelled, and a
normal Q-Q plot.

The figures are written to ``examples/nls/figures/``.

Run from the repository root:

    python examples/nls/diagnostics.py
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

from statcraft.nls import nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()
fit = nls("circumference ~ SSlogis(age, Asym, xmid, scal)", orange).fit()

fit.plot_diagnostics(save=OUT / "nls_diagnostics.png")
fit.plot_curve(save=OUT / "nls_curve.png")
fit.plot_cooks(save=OUT / "nls_cooks.png")
print(f"wrote {OUT / 'nls_diagnostics.png'}")
print(f"wrote {OUT / 'nls_curve.png'}")
print(f"wrote {OUT / 'nls_cooks.png'}")
