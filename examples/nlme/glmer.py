# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""``glmer`` -- generalized linear mixed models with lme4 syntax.

The ``bdf`` data contain the arithmetic test results of Dutch pupils nested
in schools.  ``repeatgr`` counts grade repetitions; it is recoded to a binary
indicator so the model is a logistic regression with a school-level random
intercept, fitted by the Laplace approximation.

For this Bernoulli fit, ``result.plot()`` draws the discrete GLMER mosaic:
population vs conditional predictions, a DHARMa-style Q-Q of simulated
quantile residuals, binned quantile residuals against ``rank(ŷ)`` and a BLUP
caterpillar with 95% conditional intervals. The same construction supports
grouped binomial ``k/n``, Poisson and negative-binomial responses; continuous
families keep the population/conditional, normal Q-Q, binned residual and BLUP
layout.

The second part fits the same one-way random-intercept structure to the
continuous ``Machines`` scores with the Gaussian, Gamma and inverse Gaussian
families, showing both estimated and fixed dispersion.

Run from the repository root:

    python examples/nlme/glmer.py
"""

from pathlib import Path

import statcraft.nlme as nl
from statcraft.glm import Binomial, Gaussian, Gamma, InverseGaussian
from statcraft.datasets import data_loader

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

bdf = data_loader("bdf.csv", data_dir_override=str(DATA_DIR)).load()
bdf = bdf.dropna(subset=["repeatgr", "aritPRET", "Minority", "schoolNR"]).copy()
bdf["repeatgr"] = (bdf["repeatgr"] > 0).astype(int)

result = nl.glmer(
    "repeatgr ~ aritPRET + Minority + (1 | schoolNR)",
    bdf,
    family=Binomial("logit"),  # same GLM-style family objects as glm(); strings are rejected
).fit()

print(result.summary())

# ---- continuous responses: Gaussian, Gamma and inverse Gaussian ------------
machines = data_loader("Machines.csv", data_dir_override=str(DATA_DIR)).load()
for family in (Gaussian(), Gamma("log"), InverseGaussian()):
    fit = nl.glmer(
        "score ~ 1 + (1 | Machine)",
        machines,
        family=family,
    ).fit()
    print(f"\n{family}: dispersion = {fit.dispersion:.6g}")
    print(fit.summary())

# The dispersion can also be held fixed (sigma^2 for Gaussian, 1/shape for the
# other two families) instead of being estimated.
fixed = nl.glmer(
    "score ~ 1 + (1 | Machine)",
    machines,
    family=Gaussian(),
    dispersion=30.0,
).fit()
print(f"\nGaussian with fixed dispersion: {fixed.dispersion}")

