"""Degrees-of-freedom helpers for mixed-model inference.

The routines here implement the Satterthwaite approximation used by
``lmerTest``/``nlme`` for fixed effects.  They deliberately work on the
profiled covariance model exposed by the C++ engine: given a vector of
variance parameters, ``profile(theta)`` returns the unscaled covariance
``(X' V(theta)^-1 X)^-1`` of the fixed effects.  The delta-method gradient of
that covariance, combined with the asymptotic covariance of the variance
parameters, produces denominator degrees of freedom that respect weights,
residual correlation and crossed random effects.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .crossed import _hessian


# ---------------------------------------------------------------------------
# generic Satterthwaite algebra
# ---------------------------------------------------------------------------
def satterthwaite_df(
    theta_hat: Sequence[float],
    cov_theta: np.ndarray,
    cov_beta_func: Callable[[np.ndarray], np.ndarray],
    L_list: Iterable[np.ndarray],
    eps: float = 1e-5,
) -> np.ndarray:
    """Satterthwaite denominator df for a list of fixed-effect contrasts.

    Parameters
    ----------
    theta_hat
        Estimated variance parameters (REML), including ``log(sigma)`` when the
        residual variance is estimated.
    cov_theta
        Asymptotic covariance of ``theta_hat`` (inverse Hessian of the negative
        REML log-likelihood).
    cov_beta_func
        Callable mapping a variance-parameter vector to the covariance matrix
        of the fixed effects at that point.
    L_list
        Contrasts ``L`` with as many columns as fixed effects.  A one-row
        contrast yields a t df; a multi-row contrast uses Fai & Cornelius
        (1996) to combine the eigen-component dfs into one F denominator df.

    Returns
    -------
    numpy.ndarray
        One denominator df per contrast (``NaN`` when it cannot be computed).
    """
    th = np.asarray(theta_hat, dtype=float).reshape(-1)
    C_theta = np.asarray(cov_theta, dtype=float)
    if C_theta.ndim != 2 or C_theta.shape[0] != C_theta.shape[1]:
        raise ValueError("cov_theta must be a square matrix")
    if C_theta.shape[0] != th.size:
        raise ValueError(
            f"cov_theta has shape {C_theta.shape}, expected ({th.size}, {th.size})"
        )

    phi_hat = np.atleast_2d(np.asarray(cov_beta_func(th), dtype=float))
    m, p = th.size, phi_hat.shape[0]
    dPhi = np.zeros((m, p, p), dtype=float)
    for k in range(m):
        step = eps * max(1.0, abs(th[k]))
        tp, tm = th.copy(), th.copy()
        tp[k] += step
        tm[k] -= step
        try:
            dPhi[k] = (
                np.atleast_2d(np.asarray(cov_beta_func(tp), dtype=float))
                - np.atleast_2d(np.asarray(cov_beta_func(tm), dtype=float))
            ) / (2.0 * step)
        except Exception:
            dPhi[k] = np.nan

    out = []
    for L in L_list:
        L = np.atleast_2d(np.asarray(L, dtype=float))
        if L.shape[1] != p:
            raise ValueError(f"contrast has {L.shape[1]} columns, expected {p}")
        q = L.shape[0]
        C = L @ phi_hat @ L.T
        if q == 1:
            v = float(C[0, 0])
            g = np.array(
                [float((L @ dPhi[k] @ L.T)[0, 0]) for k in range(m)], dtype=float
            )
            denom = float(g @ C_theta @ g)
            out.append(2.0 * v * v / denom if denom > 0 and np.isfinite(denom) else np.nan)
            continue

        # Fai & Cornelius: combine the eigenvalues of the multivariate contrast.
        eigvals, eigvecs = np.linalg.eigh(C)
        nu_i = []
        for i in range(q):
            p_i = eigvecs[:, i : i + 1]
            l_i = p_i.T @ L
            v_i = float(eigvals[i])
            g_i = np.array(
                [float((l_i @ dPhi[k] @ l_i.T)[0, 0]) for k in range(m)], dtype=float
            )
            denom_i = float(g_i @ C_theta @ g_i)
            nu_i.append(
                2.0 * v_i * v_i / denom_i if denom_i > 0 and np.isfinite(denom_i) else np.nan
            )
        nu_i = np.asarray(nu_i, dtype=float)
        if np.all(np.isfinite(nu_i)) and np.all(nu_i > 2.0):
            E = float(np.sum(nu_i / (nu_i - 2.0)))
            out.append((2.0 * E) / (E - q) if E > q else np.nan)
        else:
            out.append(float(np.nanmin(nu_i)) if np.isfinite(nu_i).any() else np.nan)
    return np.asarray(out, dtype=float)


def cov_theta_from_profile(
    model: Any,
    theta_hat: Sequence[float],
    *,
    reml: bool,
    sigma_fixed: bool,
    sigma2: float,
    free: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Numerical asymptotic covariance of the variance parameters.

    Returns ``(theta_ext, cov_theta, cov_beta_func)`` where ``theta_ext`` holds
    the free covariance parameters followed by ``log(sigma)`` (unless the
    residual variance is fixed).  ``cov_beta_func`` evaluates the fixed-effect
    covariance at any such extended parameter vector.
    """
    theta_hat = np.asarray(theta_hat, dtype=float).reshape(-1)
    if free is None:
        free = list(getattr(model, "free_theta")())
    free = [int(k) for k in free]
    theta_full = theta_hat.copy()

    def expand(ext: np.ndarray) -> np.ndarray:
        th = theta_full.copy()
        th[free] = ext[: len(free)]
        return th

    if sigma_fixed:
        ext0 = theta_hat[free].astype(float)
    else:
        ext0 = np.append(theta_hat[free], 0.5 * math.log(float(sigma2)))

    def cov_beta_func(ext: np.ndarray) -> np.ndarray:
        det = model.profile(expand(ext))
        C = np.asarray(det.cov_unscaled, dtype=float)
        s2 = float(sigma2) if sigma_fixed else math.exp(2.0 * float(ext[-1]))
        return s2 * C

    def dev(ext: np.ndarray) -> float:
        det = model.profile(expand(ext))
        nobs = int(model.n_obs())
        dof = nobs - int(det.rank) if reml else nobs
        s2 = float(sigma2) if sigma_fixed else math.exp(2.0 * float(ext[-1]))
        C = np.asarray(det.cov_unscaled, dtype=float)
        val = dof * math.log(s2) + det.logdetW + det.rss / s2
        if reml:
            # 2 log|R| = -log|X' W^-1 X| = logdet((X' W^-1 X)^-1).
            with np.errstate(all="ignore"):
                sign, ld = np.linalg.slogdet(C)
            if sign > 0 and np.isfinite(ld):
                val += ld
        return float(val)

    if ext0.size == 0:
        return ext0, np.zeros((0, 0)), cov_beta_func
    H = _hessian(dev, ext0)
    try:
        cov = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((ext0.size, ext0.size), np.nan)
    return ext0, cov, cov_beta_func


# ---------------------------------------------------------------------------
# REML handling
# ---------------------------------------------------------------------------
def require_reml(fit: Any, what: str = "Satterthwaite degrees of freedom") -> Any:
    """Return a REML fit, warning (and refitting when possible) for ML fits."""
    method = str(getattr(fit, "method", "")).upper()
    if method == "REML":
        return fit
    warnings.warn(
        f"{what} require REML estimates; {method} fit detected. "
        "Refitting in REML when possible.",
        UserWarning,
        stacklevel=3,
    )
    refit = getattr(fit, "_reml_refit", None)
    if callable(refit):
        try:
            new = refit()
            if str(getattr(new, "method", "")).upper() == "REML":
                return new
        except Exception as exc:  # pragma: no cover - depends on user model
            warnings.warn(f"automatic REML refit failed ({exc}); using current fit.", UserWarning, stacklevel=3)
    return fit


def ddf_for_fit(fit: Any, L_list: Iterable[np.ndarray]) -> np.ndarray:
    """Satterthwaite dfs for arbitrary contrasts on a supported mixed fit."""
    L_list = [np.atleast_2d(np.asarray(L, dtype=float)) for L in L_list]
    name = type(fit).__name__
    if name == "LmerFit":
        return fit.satterthwaite_contrast_df(L_list)
    if name not in ("LmeFit", "NlmeFit"):
        raise NotImplementedError(f"Satterthwaite df not implemented for {name}")

    fit = require_reml(fit)
    model = getattr(fit, "_ddf_model", None)
    if model is None:
        return np.full(len(L_list), np.nan)
    raw = fit.raw
    sigma2 = float(raw.sigma2)
    sigma_fixed = bool(getattr(fit, "sigma_fixed", False))
    reml = str(getattr(fit, "method", "")).upper() == "REML"

    if name == "LmeFit":
        free = list(model.free_theta())
        cov = np.asarray(getattr(raw, "vcov_theta", np.empty((0, 0))), dtype=float)
        if cov.size == 0:
            ext, cov, cov_beta = cov_theta_from_profile(
                model, np.asarray(raw.theta), reml=reml, sigma_fixed=sigma_fixed,
                sigma2=sigma2, free=free,
            )
        else:
            ext = np.append(np.asarray(raw.theta)[free], 0.5 * np.log(sigma2)) if not sigma_fixed else np.asarray(raw.theta)[free]

            def cov_beta(x):
                det = model.profile(_expand_theta(np.asarray(raw.theta), free, x))
                s2 = sigma2 if sigma_fixed else math.exp(2.0 * float(x[-1]))
                return s2 * np.asarray(det.cov_unscaled, dtype=float)
    else:  # NlmeFit: use the working LME model linearized at the optimum
        try:
            model = model.linearized_lme(np.asarray(fit.fixef, dtype=float),
                                         [np.asarray(b, dtype=float) for b in raw.b])
        except Exception:
            return np.full(len(L_list), np.nan)
        sigma_fixed = sigma_fixed or bool(model.sigma_is_fixed())
        free = list(model.free_theta())
        ext, cov, cov_beta = cov_theta_from_profile(
            model, np.asarray(raw.theta), reml=reml, sigma_fixed=sigma_fixed,
            sigma2=sigma2, free=free,
        )
    if not np.all(np.isfinite(cov)):
        return np.full(len(L_list), np.nan)
    return satterthwaite_df(ext, cov, cov_beta, L_list)


def _expand_theta(theta_full: np.ndarray, free: Sequence[int], ext: np.ndarray) -> np.ndarray:
    th = np.asarray(theta_full, dtype=float).copy()
    th[list(free)] = np.asarray(ext, dtype=float)[: len(free)]
    return th


# ---------------------------------------------------------------------------
# model-comparison ANOVA with Satterthwaite F tests
# ---------------------------------------------------------------------------
def anova_f_test(small: Any, big: Any, method: str = "satterthwaite") -> dict[str, float]:
    """Wald F test for the extra fixed effects of ``big`` over ``small``."""
    s_names = [str(n) for n in getattr(small, "fixef", pd.Series(dtype=float)).index]
    b_names = [str(n) for n in getattr(big, "fixef", pd.Series(dtype=float)).index]
    if not set(s_names).issubset(set(b_names)):
        raise ValueError("anova with ddf requires nested fixed-effects structures")
    extra = [n for n in b_names if n not in set(s_names)]
    if not extra:
        raise ValueError("the models have the same fixed effects; nothing to test")
    beta = np.asarray(big.fixef, dtype=float)
    V = np.asarray(big.vcov, dtype=float)
    R = np.zeros((len(extra), len(b_names)))
    for i, name in enumerate(extra):
        R[i, b_names.index(name)] = 1.0
    diff = R @ beta
    C = R @ V @ R.T
    try:
        stat = float(diff @ np.linalg.solve(C, diff) / len(extra))
    except np.linalg.LinAlgError:
        stat = np.nan
    if method == "satterthwaite":
        den_df = float(np.nanmean(ddf_for_fit(big, [R]))) if type(big).__name__ != "LmerFit" else float(
            big.satterthwaite_contrast_df([R])[0]
        )
    else:
        den_df = float(big.nobs - int(np.asarray(V).shape[0]))
    return {
        "statistic": stat,
        "num_df": float(len(extra)),
        "den_df": den_df,
        "pvalue": float(stats.f.sf(stat, len(extra), den_df)) if np.isfinite(stat) and np.isfinite(den_df) and den_df > 0 else np.nan,
    }
