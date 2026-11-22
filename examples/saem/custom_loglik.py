# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""SAEM -- user-supplied log-likelihood (robust Student-t regression).

Standard SAEM models define a structural function and a built-in error model.
When the likelihood is not of that form, pass ``loglik=`` to :class:`NLModel`
(with ``fn=None``) and SAEM optimises it directly.  Here a Student-t
likelihood with 4 degrees of freedom makes the linear growth fit robust to
four deliberately injected outliers.

Run from the repository root:

    python examples/saem/custom_loglik.py
"""

from pathlib import Path

import numpy as np
from scipy import special

import statcraft as sf
from statcraft.saem import CustomFamily, NLModel
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

oxboys = data_loader("oxboys", data_dir_override=str(DATA_DIR)).load().copy()
oxboys.loc[[5, 42, 110, 195], "height"] += np.array([15.0, -18.0, 20.0, -15.0])


def loglik_student_t(y, base, slope, sigma, age):
    """Student-t log-likelihood with nu = 4."""
    nu = 4.0
    mu = base + slope * age
    z2 = ((y - mu) / sigma) ** 2
    const = special.gammaln(0.5 * (nu + 1.0)) - special.gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    return const - np.log(sigma) - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)


model_custom = NLModel(
    fn=None,
    loglik=loglik_student_t,
    params=["base", "slope", "sigma"],
    transform={"base": "identity", "slope": "identity", "sigma": "log"},
    random=["base", "slope"],
    omega="full",
    family=CustomFamily,
)

fit = sf.SAEM(
    model=model_custom,
    data=oxboys,
    id="Subject",
    time="age",
    response="height",
    start={"base": 150.0, "slope": 6.5, "sigma": 0.7},
    start_sd={"base": 7.0, "slope": 1.5},
    seed=632545,
).fit()

print("Compilation:", fit.model.compile_info)
print(fit.summary())
