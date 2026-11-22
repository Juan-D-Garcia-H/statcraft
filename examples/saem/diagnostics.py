# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostic plots for a SAEM fit.

The Emax dose-response model with left-censored responses is fitted once and
then used for the pharmacometric diagnostics:

* the SAEM diagnostic mosaic (predictions, NPDE, Q-Q envelope, within-subject
  ACF);
* the SAEM ``theta`` convergence trace;
* a visual predictive check (VPC) with simulated P10/P50/P90 envelopes;
* normalized prediction distribution errors (NPDE), including the censored
  observations.

The figures are written to ``examples/saem/figures/``.  The simulation-based
plots accept ``nsim=`` to trade accuracy for speed.

Run from the repository root:

    python examples/saem/diagnostics.py
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

import statcraft as sf
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

pd1 = data_loader("PD1", data_dir_override=str(DATA_DIR)).load()
LLOQ = 25.0
pd1["cens"] = np.where(pd1["response"] <= LLOQ, -1, 0)
pd1["y_cens"] = np.where(pd1["cens"] == -1, LLOQ, pd1["response"])


def model_emax(E0, Emax, EC50, dose):
    return E0 + Emax * dose / np.maximum(EC50 + dose, 1e-12)


fit = sf.SAEM(
    model=model_emax,
    data=pd1,
    id="subject",
    time="dose",
    response="y_cens",
    cens="cens",                               # -1 = left-censored at y_cens
    params=["E0", "Emax", "EC50"],
    transform={"E0": "identity", "Emax": "identity", "EC50": "log"},
    error="constant",
    start={"E0": 20.0, "Emax": 80.0, "EC50": 15.0},
    start_sd={"E0": 5.0, "Emax": 15.0, "EC50": 0.5},
    start_error=5.0,
    seed=632545,
).fit()

fit.plot_diagnostics(save=OUT / "saem_diagnostics.png")
fit.plot_trace(save=OUT / "saem_trace.png")
fit.plot_vpc(nsim=100, save=OUT / "saem_vpc.png")
fit.plot_npde(nsim=200, save=OUT / "saem_npde.png")
fit.plot_ipred(save=OUT / "saem_ipred.png")
print(f"wrote {OUT / 'saem_diagnostics.png'}")
print(f"wrote {OUT / 'saem_trace.png'}")
print(f"wrote {OUT / 'saem_vpc.png'}")
print(f"wrote {OUT / 'saem_npde.png'}")
print(f"wrote {OUT / 'saem_ipred.png'}")
