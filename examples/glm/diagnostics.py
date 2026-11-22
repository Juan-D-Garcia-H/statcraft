# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostic plots for GLM fits.

Shows the randomized-quantile-residual diagnostics (residuals vs fitted,
simulated-band normal Q-Q) and the count rootogram, including the
zero-inflated case where the excess of zeros is the whole point of the model.

The figures are written to ``examples/glm/figures/``; call ``fit.plot()``
without ``save=`` to display them interactively instead.

Run from the repository root:

    python examples/glm/diagnostics.py
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

import statcraft.glm as scg
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Poisson counts: 2x2 diagnostics + rootogram
# ---------------------------------------------------------------------------
affairs = data_loader("Affairs.csv", data_dir_override=str(DATA_DIR)).load()
poisson = scg.glm(
    "affairs ~ age + yearsmarried + religiousness + rating",
    data=affairs,
    family=scg.Poisson(),
).fit()

poisson.plot_diagnostics(save=OUT / "poisson_diagnostics.png")
poisson.plot_rootogram(save=OUT / "poisson_rootogram.png")
print(f"Poisson: wrote {OUT / 'poisson_diagnostics.png'}")
print(f"Poisson: wrote {OUT / 'poisson_rootogram.png'}")

# ---------------------------------------------------------------------------
# Zero-inflated Poisson: the rootogram shows the zero class explicitly
# ---------------------------------------------------------------------------
bio = data_loader("bioChemists.csv", data_dir_override=str(DATA_DIR)).load()
zip_fit = scg.glm(
    "art ~ fem + kid5 + ment | ment",
    data=bio,
    family=scg.ZeroInflatedPoisson(),
).fit()

zip_fit.plot_diagnostics(save=OUT / "zip_diagnostics.png")
zip_fit.plot_rootogram(save=OUT / "zip_rootogram.png")
print(f"ZIP: wrote {OUT / 'zip_diagnostics.png'}")
print(f"ZIP: wrote {OUT / 'zip_rootogram.png'}")
