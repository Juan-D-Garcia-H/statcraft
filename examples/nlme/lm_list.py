# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``lmList`` -- one ordinary least-squares fit per group.

Before fitting a mixed model it is useful to look at the separate fits.
``lm_list`` runs ``distance ~ age`` independently for each child in the
``Orthodont`` data and collects the coefficients in one table.  The spread of
the per-child slopes is the empirical evidence for a random slope.

Run from the repository root:

    python examples/nlme/lm_list.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orth = data_loader("Orthodont.csv", data_dir_override=str(DATA_DIR)).load()

fits = nl.lm_list("distance ~ age | Subject", orth)

print("Per-subject coefficients")
print(fits.coef)
print()
print("Per-subject residual standard deviations")
print(fits.sigma)
print()
print(f"Pooled residual SD: {fits.pooled_sigma:.4f} (df = {fits.pooled_df})")
