# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- escaping a local minimum with ``fit_multistart``.

Nonlinear least squares is sensitive to the starting values.  The Weibull
growth curve below is deliberately started far from the data range: the plain
``fit()`` converges to a local minimum (a flat curve, R² = 0), while
``fit_multistart()`` perturbs the start multiplicatively and keeps the
lowest-cost converged solution.

Run from the repository root:

    python examples/nls/multistart_local_minimum.py
"""

from pathlib import Path

from statcraft.nls import nls
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

soybean = data_loader("Soybean.csv", data_dir_override=str(DATA_DIR)).load()

formula = "weight ~ SSweibull(Time, Asym, Drop, lrc, pwr)"
bad_start = {"Asym": 1.0, "Drop": 1.0, "lrc": 0.0, "pwr": 0.1}

plain = nls(formula, soybean, start=bad_start).fit()
print("Plain fit from the bad start")
print(f"  converged : {plain.result.success}")
print(f"  R2        : {plain.rsquared:.4f}")
print(f"  cost      : {plain.result.cost:.2f}")
print(f"  params    : { {k: round(float(v), 3) for k, v in plain.coefficients.items()} }")

multi = nls(formula, soybean, start=bad_start).fit_multistart(
    n_starts=25,
    perturbation=1.0,
    random_state=2026,
)
print("\nfit_multistart")
print(f"  converged : {multi.result.success}")
print(f"  R2        : {multi.rsquared:.4f}")
print(f"  cost      : {multi.result.cost:.2f}")
print(f"  params    : { {k: round(float(v), 3) for k, v in multi.coefficients.items()} }")
print(f"  attempts  : {multi.multistart_info['n_converged']} / "
      f"{multi.multistart_info['n_starts']} converged")
print(f"  costs     : {[round(float(c), 1) for c in multi.multistart_info['costs'][:8]]} ...")
