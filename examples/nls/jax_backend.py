# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``nls`` -- optional JAX backend with an analytical jacobian.

``backend="jax"`` compiles the model with JAX, so ``fit()`` can use an
analytical jacobian instead of the finite-difference approximation used by the
NumPy backend.  JAX is an optional dependency: if it is not installed this
script explains how to get it and exits without an error.

Run from the repository root:

    python examples/nls/jax_backend.py
"""

import time
from pathlib import Path

import numpy as np

from statcraft.nls import nls
from statcraft.datasets import data_loader

try:
    import jax  # noqa: F401
except ImportError:
    print("JAX is not installed, so this optional example is skipped.")
    print("Install it with: pip install jax jaxlib")
    raise SystemExit(0)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

orange = data_loader("Orange.csv", data_dir_override=str(DATA_DIR)).load()
formula = "circumference ~ SSlogis(age, Asym, xmid, scal)"

start = time.perf_counter()
numpy_fit = nls(formula, orange, backend="numpy").fit()
numpy_time = time.perf_counter() - start

start = time.perf_counter()
jax_fit = nls(formula, orange, backend="jax").fit(use_analytical_jacobian=True)
jax_time = time.perf_counter() - start

print(f"NumPy backend : {numpy_time:.3f}s")
print(f"JAX backend   : {jax_time:.3f}s (includes compilation)")
print(f"same estimates: {np.allclose(numpy_fit.params, jax_fit.params, rtol=1e-4)}")
print(f"NumPy params  : {numpy_fit.params}")
print(f"JAX params    : {jax_fit.params}")
print(f"JAX R2        : {jax_fit.rsquared:.6f}")
