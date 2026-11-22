# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nlsList`` -- one nonlinear least-squares fit per group.

``nls_list`` fits the logistic growth curve separately to each soybean plot,
which shows how much the asymptote and inflection point vary between plots
before those differences are modelled as random effects in ``nlme.py``.

Run from the repository root:

    python examples/nlme/nls_list.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

soybean = data_loader("Soybean.csv", data_dir_override=str(DATA_DIR)).load()

fits = nl.nls_list(
    "weight ~ SSlogis(Time, Asym, xmid, scal) | Plot",
    soybean,
    start={"Asym": 19.0, "xmid": 55.0, "scal": 8.5},
)

print("Per-plot growth-curve parameters")
print(fits.coef)
print()
print(f"Converged plots: {int(fits.converged.sum())} / {len(fits.converged)}")
print(f"Pooled residual SD: {fits.pooled_sigma:.4f}")
