# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostic plots for a multivariate covariance GLM (``statcraft.mcglm``).

Three associated responses are fitted jointly.  Each response gets its own
Pearson-residual-vs-fitted panel, and the between-response correlation
observed in the data is compared with the correlation implied by the fitted
joint covariance.

The figures are written to ``examples/mcglm/figures/``.

Run from the repository root:

    python examples/mcglm/diagnostics.py
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

import statcraft.mcglm as scm
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

soya = data_loader("soya.csv", data_dir_override=str(DATA_DIR)).load().astype(
    {"block": "category", "water": "category", "pot": "category"}
)
soya["viable_peas"] = soya["viablepeas"] / soya["totalpeas"]
n = soya.shape[0]

fit = scm.mcglm(
    [
        "grain ~ block + water * pot",
        "seeds ~ block + water * pot",
        "viable_peas ~ block + water * pot",
    ],
    data=soya,
    z=[scm.mc_id(n)],
    link=["identity", "log", "logit"],
    variance=["constant", "tweedie", "binomialP"],
    ntrial=[None, None, soya["totalpeas"].values],
).fit()

fit.plot_pearson(save=OUT / "mcglm_pearson.png")
fit.plot_correlation(save=OUT / "mcglm_correlation.png")
fit.plot_diagnostics(save=OUT / "mcglm_diagnostics.png")
print(f"wrote {OUT / 'mcglm_pearson.png'}")
print(f"wrote {OUT / 'mcglm_correlation.png'}")
print(f"wrote {OUT / 'mcglm_diagnostics.png'}")
