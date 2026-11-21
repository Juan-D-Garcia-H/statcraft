"""Algorithm control: ``saem_control()`` (in the style of ``nlmeControl`` / ``saemixControl``)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Optional
import numpy as np

from .. import _saemfit as _core
from ._utils import enum_from

def _default_seed():
    """Draw a fresh 32-bit seed for a new :class:`SaemControl`."""
    return int(np.random.default_rng().integers(0, 2**32))

@dataclass
class SaemControl:
    """Fitting options for the SAEM engine. Every option has a sensible default.

    Algorithm
    ---------
    n_iter, n_burn : total SAEM iterations and initial MCMC burn-in iterations.
    n_mcmc         : MCMC sweeps per iteration.
    n_chains       : MCMC chains per subject (2 avoids the variance collapse of classical SAEM).
    seed           : seed (the result is reproducible with the same seed and INDEPENDENT of n_threads:
                     each subject has its own generator).
    n_threads      : 0 = all hardware threads (default); 1 = serial. Only applies to compiled models
                     (with Python functions the fit always runs on 1 thread because of the GIL).
    mstep          : 'auto' | 'sufficient_stats' | 'gradient'.
    simulated_annealing, map_kernel, map_every, auto_stop, stop_tol, adapt_rw.
    step_k1, step_alpha : classical SAEM steps (gamma_k = 1 up to k1, then k^-alpha).
    verbose, print_every : progress reporting to ``sys.stdout``.

    Inference
    ---------
    se_method      : 'auto' | 'linearization' | 'gauss_hermite' | 'louis' | 'laplace' | 'none'.
    ll_method      : 'auto' | 'gauss_hermite' | 'importance_sampling' | 'linearization' | 'laplace' | 'none'.
    cond_mean      : compute the conditional mean (EBE) with SD, ESS and R-hat.
    n_cond, cond_burn, cond_thin : conditional-sampling size, burn-in and thinning.
    gq_nodes, gq_inner_nodes : Gauss-Hermite nodes for the outer/inner quadrature.
    is_samples     : importance-sampling draws for the marginal likelihood.
    louis_draws, louis_is_samples : draws used by Louis' method and its importance-sampling variant.

    Attributes
    ----------
    n_iter : int, default 1000
        Number of SAEM iterations after burn-in.
    n_burn : int, default 200
        Number of initial burn-in iterations.
    n_mcmc : int, default 5
        MCMC sweeps performed in every SAEM iteration.
    n_chains : int, default 2
        Independent MCMC chains per subject.
    seed : int
        Random seed; a random one is drawn when the control is created.
    n_threads : int, default 0
        Worker threads (0 = all hardware threads, 1 = serial).
    mstep : str, default "auto"
        M-step strategy.
    simulated_annealing : bool, default True
        Enable simulated annealing of the MCMC kernel.
    map_kernel : bool, default True
        Use a MAP-type kernel during sampling.
    map_every : int, default 25
        Interval (in iterations) for MAP updates.
    auto_stop : bool, default False
        Stop early when the parameter drift falls below ``stop_tol``.
    stop_tol : float, default 1e-3
        Tolerance used by ``auto_stop``.
    adapt_rw : bool, default True
        Adapt the random-walk proposal.
    step_k1, step_alpha : int, float
        Classical SAEM step schedule: ``gamma_k = 1`` for ``k <= step_k1`` and
        ``k ** -step_alpha`` afterwards.
    verbose : bool, default False
        Print progress information.
    print_every : int, default 100
        Progress-printing interval.
    cond_mean : bool, default True
        Compute the conditional mean (EBE) and its diagnostics.
    n_cond : int, default 1000
        Conditional samples per subject.
    cond_burn : int, default 200
        Conditional-sampling burn-in.
    cond_thin : int, default 2
        Conditional-sampling thinning.
    se_method : str, default "auto"
        Standard-error method.
    ll_method : str, default "auto"
        Marginal log-likelihood method.
    gq_nodes : int, default 15
        Outer Gauss-Hermite nodes.
    gq_inner_nodes : int, default 16
        Inner Gauss-Hermite nodes.
    is_samples : int, default 2000
        Importance-sampling draws for the likelihood.
    louis_draws : int, default 150
        Draws used by Louis' method.
    louis_is_samples : int, default 4000
        Importance-sampling draws used by Louis' method.
    rm : mapping, optional
        Overrides for the Robbins-Monro kernel (``gamma0``, ``alpha``, ``schedule``, ...).
    sa : mapping, optional
        Overrides for the simulated-annealing kernel (``beta0``, ``rate``, ``schedule``, ...).
    """
    n_iter: int = 500
    n_burn: int = 300
    n_mcmc: int = 4
    n_chains: int = 2
    seed: int = field(default_factory=_default_seed)
    n_threads: int = 0
    mstep: str = "auto"
    simulated_annealing: bool = True
    map_kernel: bool = True
    map_every: int = 25
    auto_stop: bool = False
    stop_tol: float = 1e-3
    adapt_rw: bool = True
    step_k1: int = -1
    step_alpha: float = 1.0
    verbose: bool = False
    print_every: int = 100

    cond_mean: bool = True
    n_cond: int = 1000
    cond_burn: int = 200
    cond_thin: int = 2

    se_method: str = "auto"
    ll_method: str = "auto"
    gq_nodes: int = 15
    gq_inner_nodes: int = 16
    is_samples: int = 5000
    louis_draws: int = 150
    louis_is_samples: int = 4000


    rm: Optional[Mapping[str, Any]] = field(default=None, repr=False)   # e.g. {"gamma0": 2.0, "alpha": 0.6}
    sa: Optional[Mapping[str, Any]] = field(default=None, repr=False)   # e.g. {"beta0": 10.0, "rate": 0.97}

    def copy(self, **changes) -> "SaemControl":
        """Return a copy of the control with the given fields replaced."""
        return replace(self, **changes)

    def _to_core(self) -> "_core.Control":
        """Translate and validate the options for the low-level ``_core.Control``."""
        if self.n_iter < 1 or self.n_burn < 0:
            raise ValueError("n_iter >= 1 and n_burn >= 0")
        if self.n_chains < 1:
            raise ValueError("n_chains >= 1")
        c = _core.Control()
        c.n_iterations = int(self.n_iter)
        c.n_burn_in = int(self.n_burn)
        c.n_mcmc_sweeps = int(self.n_mcmc)
        c.n_chains = int(self.n_chains)
        c.seed = int(self.seed) & 0xFFFFFFFF
        c.n_threads = int(self.n_threads)
        c.mstep = enum_from(_core.MStep, self.mstep, "mstep")
        c.simulated_annealing = bool(self.simulated_annealing)
        c.map_kernel = bool(self.map_kernel)
        c.map_every = int(self.map_every)
        c.auto_stop = bool(self.auto_stop)
        c.stop_tol = float(self.stop_tol)
        c.adapt_rw = bool(self.adapt_rw)
        c.step.k1 = int(self.step_k1)
        c.step.alpha = float(self.step_alpha)
        c.verbose = bool(self.verbose)
        c.print_every = int(self.print_every)
        c.compute_cond = bool(self.cond_mean)
        c.n_cond_samples = int(self.n_cond)
        c.cond_burn_in = int(self.cond_burn)
        c.cond_thin = int(self.cond_thin)
        c.se_method = enum_from(_core.SEMethod, self.se_method, "se_method")
        c.compute_se = c.se_method != _core.SEMethod.none
        c.ll_method = enum_from(_core.LLMethod, self.ll_method, "ll_method")
        c.gq_nodes = int(self.gq_nodes)
        c.gq_inner_nodes = int(self.gq_inner_nodes)
        c.is_samples = int(self.is_samples)
        c.louis_draws = int(self.louis_draws)
        c.louis_is_samples = int(self.louis_is_samples)
        for name, target in (("rm", c.rm), ("sa", c.sa)):
            opts = getattr(self, name)
            for k, v in (opts or {}).items():
                if not hasattr(target, k):
                    raise ValueError(f"{name}: unknown option {k!r}")
                if k == "schedule":
                    v = enum_from(_core.RMSchedule if name == "rm" else _core.SASchedule, v, f"{name}.schedule")
                setattr(target, k, v)
        return c


_CONTROL_FIELDS = {f.name for f in fields(SaemControl)}

#: Short aliases (handy in notebooks) -> real field name.
_CONTROL_ALIASES = {
    "se": "se_method",
    "ll": "ll_method",
    "iterations": "n_iter",
    "burn": "n_burn",
    "threads": "n_threads",
    "chains": "n_chains",
    "sweeps": "n_mcmc",
}


def _normalize_control_kwargs(kwargs: Mapping[str, Any]) -> dict:
    """Translate short aliases to the ``SaemControl`` field names."""
    out = dict(kwargs)
    for alias, real in _CONTROL_ALIASES.items():
        if alias in out:
            if real in out:
                raise TypeError(f"{alias!r} and {real!r} are the same control option; use only one")
            out[real] = out.pop(alias)
    return out


def saem_control(**kwargs) -> SaemControl:
    """Build a :class:`SaemControl`. E.g. ``saem_control(n_iter=500, seed=1, se_method="louis")``.

    Short aliases are also accepted: ``se`` (= ``se_method``), ``ll`` (= ``ll_method``),
    ``iterations`` (= ``n_iter``), ``burn`` (= ``n_burn``), ``sweeps`` (= ``n_mcmc``),
    ``threads`` (= ``n_threads``) and ``chains`` (= ``n_chains``).
    """
    kwargs = _normalize_control_kwargs(kwargs)
    bad = set(kwargs) - _CONTROL_FIELDS
    if bad:
        raise TypeError(f"saem_control: unknown options {sorted(bad)}. Valid: {sorted(_CONTROL_FIELDS)}")
    return SaemControl(**kwargs)
