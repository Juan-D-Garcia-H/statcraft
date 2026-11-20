# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Diagnostics for IRLS / quasi families that need the variance function V(mu) itself.

Anscombe residuals
------------------
    r_A = ( A(y) - A(mu) ) / V(mu)^(1/6),      A(x) = int V(t)^(-1/3) dt          (up to a constant)

so that A(Y) is as close to normal as a variance-stabilizing transform can make it. Closed forms are used
for every built-in variance function (Gaussian, Poisson, Binomial, Gamma/Exponential/Pareto/InverseGamma,
inverse Gaussian, Negative Binomial, Tweedie) and adaptive quadrature for user-supplied variances
(QuasiCustom). Verified against int_mu^y V^(-1/3) dt by numerical integration.
"""

import numpy as np
from scipy import special as sp
from scipy.integrate import quad


def anscombe_A(kind, x, theta=None, p=None):
    """A(x) for a named variance function. kind in {"gaussian","poisson","binomial","gamma","invgauss","nb","tweedie"}."""
    x = np.asarray(x, dtype=float)
    if kind == "gaussian":  # V = 1
        return x
    if kind == "poisson":  # V = mu
        return 1.5 * np.maximum(x, 0.0) ** (2.0 / 3.0)
    if kind == "binomial":  # V = mu (1 - mu)
        z = np.clip(x, 0.0, 1.0)
        return 1.5 * z ** (2.0 / 3.0) * sp.hyp2f1(1.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0, z)
    if kind == "gamma":  # V = mu^2
        return 3.0 * np.maximum(x, 0.0) ** (1.0 / 3.0)
    if kind == "invgauss":  # V = mu^3
        return np.log(np.maximum(x, 1e-300))
    if kind == "nb":  # V = mu + mu^2 / theta
        z = np.maximum(x, 0.0)
        return 1.5 * z ** (2.0 / 3.0) * sp.hyp2f1(1.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0, -z / theta)
    if kind == "tweedie":  # V = mu^p
        if abs(p - 3.0) < 1e-10:
            return np.log(np.maximum(x, 1e-300))
        e = 1.0 - p / 3.0
        return np.maximum(x, 0.0) ** e / e
    raise ValueError(f"anscombe_A: unknown variance kind '{kind}'")


def variance_of(kind, mu, theta=None, p=None):
    """Variance function V(mu) for a named exponential-family distribution.
    ``kind`` in {"gaussian", "poisson", "binomial", "gamma", "invgauss",
    "nb", "tweedie"}; ``theta`` is the negative-binomial dispersion and
    ``p`` the Tweedie power."""
    mu = np.asarray(mu, dtype=float)
    return {
        "gaussian": lambda: np.ones_like(mu),
        "poisson": lambda: mu,
        "binomial": lambda: mu * (1.0 - mu),
        "gamma": lambda: mu**2,
        "invgauss": lambda: mu**3,
        "nb": lambda: mu + mu**2 / theta,
        "tweedie": lambda: mu**p,
    }[kind]()


def anscombe_residuals(kind, y, mu, weights, theta=None, p=None):
    """sqrt(w) (A(y) - A(mu)) / V(mu)^(1/6)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(weights, dtype=float)
    V = variance_of(kind, mu, theta, p)
    return (
        np.sqrt(w)
        * (anscombe_A(kind, y, theta, p) - anscombe_A(kind, mu, theta, p))
        / V ** (1.0 / 6.0)
    )


def anscombe_residuals_numeric(variance_fn, y, mu, weights):
    """Same residual for an arbitrary variance function: A(y) - A(mu) = int_mu^y V(t)^(-1/3) dt."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = np.zeros(y.shape[0])
    f = lambda t: max(float(variance_fn(t)), 1e-12) ** (-1.0 / 3.0)
    for i in range(y.shape[0]):
        if abs(y[i] - mu[i]) < 1e-14:
            continue
        integral = quad(f, mu[i], y[i], epsabs=1e-12, epsrel=1e-10, limit=200)[0]
        out[i] = np.sqrt(w[i]) * integral / max(float(variance_fn(mu[i])), 1e-12) ** (1.0 / 6.0)
    return out
