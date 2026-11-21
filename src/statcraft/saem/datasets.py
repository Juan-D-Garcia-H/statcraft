"""Simulated datasets for examples and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_pk_oral(n_subjects: int = 40, seed: int = 1, *, dose: float = 100.0,
                     ka: float = 1.5, cl: float = 3.0, v: float = 30.0,
                     omega=(0.3, 0.25, 0.2), b: float = 0.1, wt_effect: float = 0.0,
                     times=(0.25, 0.5, 1, 2, 4, 6, 8, 12, 24)) -> pd.DataFrame:
    """Simulate a one-compartment oral PK dataset with proportional residual error.

    The generated data have one row per observation and the columns
    ``id, time, conc, dose, wt, lwt`` where ``lwt = log(wt / 70)``. If ``wt_effect != 0``,
    clearance depends on ``lwt`` with that coefficient.

    Parameters
    ----------
    n_subjects : int, default 40
        Number of simulated subjects.
    seed : int, default 1
        Random seed.
    dose : float, default 100.0
        Dose given to every subject at ``t = 0``.
    ka, cl, v : float
        Typical absorption rate, clearance and volume.
    omega : tuple[float, float, float], default (0.3, 0.25, 0.2)
        Log-normal random-effect standard deviations for ``ka``, ``CL`` and ``V``.
    b : float, default 0.1
        Proportional residual-error standard deviation.
    wt_effect : float, default 0.0
        Coefficient of ``log(wt / 70)`` in the clearance model.
    times : sequence[float], optional
        Observation times.

    Returns
    -------
    pandas.DataFrame
        Long-format simulated data.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(times, float)
    wt = rng.normal(70, 12, n_subjects).clip(40, 120)
    lwt = np.log(wt / 70.0)
    rows = []
    for i in range(n_subjects):
        ka_i = ka * np.exp(omega[0] * rng.standard_normal())
        cl_i = cl * np.exp(omega[1] * rng.standard_normal() + wt_effect * lwt[i])
        v_i = v * np.exp(omega[2] * rng.standard_normal())
        k = cl_i / v_i
        f = dose / v_i * ka_i / (ka_i - k) * (np.exp(-k * t) - np.exp(-ka_i * t))
        yv = f * (1 + b * rng.standard_normal(len(t)))
        for tj, yj in zip(t, yv):
            rows.append((i + 1, tj, max(yj, 1e-6), dose, wt[i], lwt[i]))
    return pd.DataFrame(rows, columns=["id", "time", "conc", "dose", "wt", "lwt"])
