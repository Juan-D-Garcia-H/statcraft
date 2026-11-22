# SAEM (nonlinear mixed effects)

`statcraft.SAEM` fits nonlinear mixed-effects models by stochastic approximation
EM. It supports built-in and user-defined structural models, covariates,
censoring, dose events, non-Gaussian responses and two-level hierarchical data.

## Entry points

| Import | Object | Purpose |
|---|---|---|
| `import statcraft as sf` | `sf.SAEM` | Model constructor (recommended) |
| `import statcraft as sf` | `sf.saem_control` | Build a `SaemControl` |
| `from statcraft.saem import SAEM, saem` | `saem` | Alias of `SAEM` |
| `import statcraft as sf` | `sf.saem` | The **submodule**, not an estimator |

```python
import numpy as np
import statcraft as sf
from statcraft.datasets import data_loader
```

The constructor returns an **unfitted** `SAEM` object. `.fit()` returns a
`SaemFit`. Calling `.fit()` again with no arguments returns the cached result;
passing controls or calling `.refit()` forces a new fit.

## A first model

```python
theo = data_loader("theo").load()
theo["cWeight"] = theo["Weight"] - 70.0

def model1cpt(x, p):
    Time, Dose = x
    ka, CL, V = p
    k = CL / V
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))

fit = sf.SAEM(
    model=model1cpt,
    data=theo,
    id="Id",
    time="Time",
    response="Concentration",
    params=["ka", "CL", "V"],
    predictors=["Time", "Dose"],
    transform="log",
    error="constant",
    covariates=["ka ~ cWeight", "CL ~ cWeight + Sex"],
    omega="diagonal",
    start={"ka": 1.5, "CL": 2.8, "V": 32.0},
    start_sd={"ka": 0.5, "CL": 0.5, "V": 0.5},
    seed=632545,
).fit()

print(fit.summary())
```

The data are in long format: one row per observation. The recommended
[`f(x, p)` signature](model-functions.md) is shared with NLS and NLME;
`predictors=[...]` defines what `x` contains and in which order it unpacks.
`transform="log"` enforces positivity, and `start_sd` gives initial
random-effect standard deviations on the estimation scale.

## `SAEM` constructor

```python
SAEM(
    model,
    data,
    *,
    id=None,
    time=None,
    response=None,
    ytype=None,
    cens=None,
    limit=None,
    group=None,
    events=None,
    evid=None,
    event_cols=None,
    predictors=None,
    reference=None,
    na_action="omit",
    start=None,
    start_sd=None,
    start_error=None,
    start_cov=None,
    fix=None,
    fix_sd=None,
    fix_error=None,
    control=None,
    **kwargs,
)
```

### Data-mapping options

| Argument | Default | Meaning |
|---|---|---|
| `model` | required | An `NLModel` or a structural callable. With a callable, model options are accepted in `**kwargs`. |
| `data` | required | Long-format `pandas.DataFrame` or mapping, one row per observation. |
| `id` | `None` | Subject column. If omitted, `id` is used when present; otherwise a synthetic per-row id is created with a warning. |
| `time` | `None` | Time column. If omitted, `time` is used when present; otherwise a synthetic within-subject observation index is created. |
| `response` | `None` | Response column; defaults to `y`. |
| `ytype` | `None` | Column identifying multiple response types. |
| `cens` | `None` | Censoring column: `-1` left, `0` observed, `+1` right. |
| `limit` | `None` | Optional column holding censoring limits. |
| `group` | `None` | Cluster column for a two-level hierarchical model. |
| `events` | `None` | Separate dose-event table with `id`, `time`, `amt` and optional `cmt`, `rate`, `addl`, `ii`, `ss`. |
| `evid` | `None` | Event-indicator column in `data`: `1` = dose, `0` = observation. |
| `event_cols` | `None` | Mapping from canonical event field names to custom column names. |
| `predictors` | `None` | Ordered columns exposed as `x` / `predictors` to the model. |
| `reference` | `None` | Mapping such as `{"Sex": "F"}` choosing categorical reference levels. |
| `na_action` | `"omit"` | `"omit"` drops rows with missing required values and warns; `"fail"` raises instead. |

### Initialization and fixing options

| Argument | Default | Meaning |
|---|---|---|
| `start` | `None` | Fixed-effect starting values. A mapping is keyed by parameter name; its key order also supplies parameter order when `params=` is omitted. |
| `start_sd` | `None` | Starting random-effect standard deviations; missing names default to `1.0`. |
| `start_error` | `None` | Starting error-model parameters (`a`, `b`, `c`). Missing fields default to `a=0`, `b=0`, `c=1`. |
| `start_cov` | `None` | Starting covariate effects keyed by terms such as `"CL~wt"`; missing terms default to `0.0`. |
| `fix` | `None` | Structural parameters held at their `start` values. |
| `fix_sd` | `None` | Random-effect SDs held at their `start_sd` values. |
| `fix_error` | `None` | Error-model parameters held fixed; requires a single response. |
| `control` | `None` | A `SaemControl` object. Individual control options may also be passed in `**kwargs`. |

`fix` and `fix_sd` require `start`. `start_cov` does not support categorical
covariates. If `params=` is omitted and `start` is not a mapping, pass it
explicitly:

```python
sf.SAEM(model1cpt, theo, params=["ka", "CL", "V"], ...)
```

An already-built `NLModel` cannot be combined with model options such as
`params=`, `transform=` or `error=`. Unknown constructor keywords raise
`TypeError`.

### Model options accepted through `**kwargs`

When `model` is a callable, these options configure the underlying `NLModel`:
`params`, `transform`, `random`, `omega`, `error`, `covariates`, `predictors`,
`family`, `vectorized`, `n_extra`, `cumhazard`, `cuts`, `loglik`, `sample`,
`tte_start`, `cluster_omega`, `name`, `compile` and `_builtin`. Control options
are separated automatically; see the control table below.

### Object state

The constructor stores the data mapping, initialization options, `model`,
`data` and `control` as public attributes. It also exposes:

| Property | Meaning |
|---|---|
| `is_fitted` | `True` once a result is cached. |
| `result` | The cached `SaemFit`, or `None`. |
| `params` | Structural parameter names from `model.params`. |

Result-only attributes such as `fixef` are guarded before fitting with an
informative `AttributeError`.

## Structural models (`NLModel`)

```python
NLModel(
    fn=None,
    *,
    params,
    transform="identity",
    random=None,
    omega="diagonal",
    error="proportional",
    covariates=None,
    predictors=None,
    family=Gaussian,
    vectorized=True,
    cumhazard=None,
    cuts=None,
    loglik=None,
    sample=None,
    n_extra=0,
    tte_start=None,
    cluster_omega="diagonal",
    name=None,
    _builtin=None,
    compile="auto",
)
```

| Option | Default | Meaning |
|---|---|---|
| `fn` | `None` | Structural callable, except for built-ins or `family=CustomFamily`. |
| `params` | required | Parameter names; their order defines the parameter vector. Duplicates are rejected. |
| `transform` | `"identity"` | `"identity"`, `"log"`, `"logit"`, `"probit"`; one value, a list aligned with `params`, or a dict such as `{"CL": "log"}`. |
| `random` | `None` | Random-effect parameter names; `None` uses all structural parameters. |
| `omega` | `"diagonal"` | Random-effect covariance: `"diagonal"`, `"full"`, a block list such as `[["ka", "CL"], ["V"]]`, or a 0/1 pattern matrix. |
| `error` | `"proportional"` | Error model; one value or one per response when `ytype=` is used. |
| `covariates` | `None` | Formulas such as `"CL ~ wt + sex"`, or an equivalent list/dict. |
| `predictors` | `None` | Ordered columns exposed as predictors. |
| `family` | `Gaussian` | Response family: a GLM family object or class (e.g. `Binomial("logit")`); strings are rejected; see the table below. |
| `vectorized` | `True` | Gaussian callables are called once per subject with arrays by default. |
| `cumhazard` | `None` | Cumulative-hazard callable for survival families such as `Weibull()`. |
| `cuts` | `None` | Cut-point callable for `Ordinal()`. |
| `loglik` | `None` | Log-likelihood callable for `family=CustomFamily`. |
| `sample` | `None` | Optional simulation callable for `family=CustomFamily`. |
| `n_extra` | `0` | Extra model parameters; automatically `1` for `negbin`. |
| `tte_start` | `None` | Initial time-to-event state. |
| `cluster_omega` | `"diagonal"` | Cluster covariance for two-level models. |
| `name` | `None` | Model label used in reports. |
| `compile` | `"auto"` | `"auto"` compiles when possible and falls back silently; `True` raises on failure; `False` always uses the Python callback. |

`NLModel` also provides `is_compiled`, `compile_info`, `n_params`,
`arg_names()` and the `NLModel.builtin(...)` constructor.

### Callable forms

All of these are supported:

```python
# Recommended: portable views
def f(x, p):
    Time, Dose = x
    ka, CL, V = p
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))

# Explicit arguments, resolved by name
def explicit(ka, CL, V, Time, Dose):
    ...

# SciPy-like positional sequences (requires predictors=)
def scipy_style(params, predictors):
    ka, CL, V = params
    Time, Dose = predictors
    ...
```

With more than one declared parameter, a two-argument callable may use any
argument names. With one parameter, the second argument must be named `p`,
`params`, `parameters` or `theta`; a single-entry view behaves as the value
itself (`a = params`), while `a, = params` also works. Views can be unpacked,
indexed by position or name, and converted with `np.asarray`. Explicit callbacks
are resolved **by argument name** against `params=`, data columns and reserved
names, so their order is irrelevant. The portable two-argument form `f(x, p)`
uses the order declared by `predictors=`/`params=`. Reserved callback names are
`params`, `predictors`, `data`, `doses` and `ytype`; `t`/`time` resolve to the
time column, and `y`/`extra` are available to custom log-likelihoods.

### Family contracts

`family=` accepts only the same family **objects/classes** used by `glm` and
`glmer`; likelihood strings are rejected. These classes are re-exported from
`statcraft.saem`, so no import from `statcraft.glm` is needed:

```python
from statcraft.saem import Binomial  # also Gaussian, Poisson, ... (all GLM families)

model = NLModel(
    fn=logit, params=["a", "b"], family=Binomial("logit"),
)
```

| Family object | Internal likelihood | Callable return | Extra requirement |
|---|---|---|---|
| `Gaussian("identity")` | gaussian | Prediction vector, one value per observation | Uses `error=`; other Gaussian links are rejected (apply the inverse link in `fn`) |
| `Binomial("logit")` | bernoulli with logistic link | Logit/linear predictor `eta` | SAEM applies the logistic link internally |
| `Binomial("probit")`, `Binomial("cloglog")`, ... | bernoulli | `P(y=1)` already on the probability scale | — |
| `Poisson("log")` | poisson | Mean `lambda` | — |
| `NegativeBinomial()` | negative binomial | Mean | Adds one `log r` parameter; a fixed `theta` is not used by SAEM |
| `ZeroInflatedPoisson()` | ZIP | `(lambda, pi)` | — |
| `Ordinal("logit")` | ordinal | `eta` | `cuts=`; only the logit link is implemented |
| `Multinomial()` | categorical | Probability vector | — |
| `Weibull()`, `LogNormal()`, `LogLogistic()`, `GeneralizedGamma()` | time-to-event | Hazard | `cumhazard=`; optional `tte_start=` |
| `CustomFamily` (class token) | user log-likelihood | — | `loglik=`, optional `sample=` |

Passing a class instead of an instance uses its defaults (`family=Binomial` is
equivalent to `family=Binomial()`). The internal canonical likelihood remains
available as `fit.model.family` for inspection, but it is not a constructor
input. GLM families without a direct SAEM likelihood — `Gamma`, `Tweedie`,
truncated/hurdle families, `ZeroInflatedNegativeBinomial`, etc. — raise a clear
error; use `family=CustomFamily` with `loglik=` for those distributions. The
object sets the likelihood and validates the link, but it does not change the
callable return contract shown above: SAEM does not invert arbitrary GLM links
(the exception is the internal logistic link of `Binomial("logit")`).

For `family=CustomFamily`, set `fn=None` and provide `loglik=`. It is called once
per observation as `loglik(y, <params>, <x>, extra=...)` and must return one
scalar log density. The optional `sample=<params>, <x>, extra=..., seed=...)`
returns one scalar draw. Arguments are matched by name, so these signatures are
valid:

```python
from statcraft.saem import CustomFamily, NLModel

def loglik(y, base, slope, sigma, age, extra=None):
    mu = base + slope * age
    return -0.5 * ((y - mu) / sigma) ** 2 - np.log(sigma)

def sample(base, slope, sigma, age, extra=None, seed=None):
    return base + slope * age + sigma * np.random.default_rng(seed).normal()

model = NLModel(
    fn=None,
    loglik=loglik,
    sample=sample,
    params=["base", "slope", "sigma"],
    family=CustomFamily,
)
```

`loglik=` can be compiled to C++; `sample=` may remain in Python. With more than
one extra parameter, `extra[i]` supplies the requested entry.

Error aliases are `constant`/`const`/`additive`/`add`/`a`,
`proportional`/`prop`/`b`, `exponential`/`exp`/`lognormal`,
`combined`/`combined1`/`comb`/`comb1`, `combined2`/`comb2` and `power`/`pow`.
Transform aliases are `identity`/`id`/`none`/`normal`/`linear`, `log`/`lognormal`,
`logit` and `probit`.

Covariates may be given as `"CL ~ wt + sex"`, `{"CL": ["wt", "sex"], "V": "wt"}`
or `"CL + V ~ wt"`. Only column names (optionally `table.column`) are accepted;
precompute derived terms in the data frame. Covariates must be constant within
a subject.

### Compilation and built-ins

`compile="auto"` traces elementwise NumPy expressions and numerically checks
them against the Python callback before accepting the compiled form.
`compile_info` reports success or the fallback reason. Conditional statements on
model values, `sum`, `cumsum`, positional indexing, `len()`, `float()`,
`math.*` and manual dose loops prevent compilation. `np.asarray(x)` cannot be
traced either; use the views directly when compilation matters.

Built-in structural models are available through `models` and
`NLModel.builtin()`:

```python
from statcraft.saem import models, builtin_models

model = models.pk1_oral(dose="dose")
print(builtin_models())
```

Available names are `exp_decay`, `biexp`, `emax`, `sigmoid_emax`, `logistic`,
`gompertz`, `michaelis_menten`, `pk1_iv`, `pk1_oral`, `pk2_iv` and `pk2_oral`.

For manually supplied doses, callbacks receive a `Doses` object with arrays
`time`, `cmt`, `amt`, `dur`, `ii`, `ss` and `modeled`. `doses.superpose(time,
kernel)` is the only dose path that compiles; it does not support steady-state
records (`SS`) or modeled rate/duration (`RATE = -1/-2`).

### Dose events

Use `events=` for a separate `id, time, amt` dose table:

```python
import pandas as pd

obs = pd.DataFrame({"id": [1, 1], "time": [1.0, 4.0],
                    "conc": [0.4, 0.8]})
events = pd.DataFrame({"id": [1, 1], "time": [0.0, 12.0],
                       "amt": [200.0, 100.0]})

fit = sf.SAEM(
    models.pk1_oral(), obs,
    id="id", time="time", response="conc",
    events=events,
    start={"ka": 1.5, "CL": 3.0, "V": 30.0},
).fit()
```

If doses are mixed into `data`, pass `evid=<column>` instead: `1` marks a dose
row (`amt` is required) and `0` marks an observation. Without either `events=`
or `evid=`, `models.pk1_oral(dose="dose")` uses a single dose at `t = 0` from a
constant-per-subject column.

A user-defined model receives `doses`; `superpose` sums every dose before the
current time and is the compilable path:

```python
def pk1_oral_multi(ka, CL, V, time, doses):
    k = CL / V
    return doses.superpose(
        time,
        lambda dt, amt: amt / V * ka / (ka - k)
        * (np.exp(-k * dt) - np.exp(-ka * dt)),
    )

fit = sf.SAEM(
    pk1_oral_multi, obs,
    id="id", time="time", response="conc",
    events=events,
    params=["ka", "CL", "V"],
    transform="log",
    start={"ka": 1.5, "CL": 3.0, "V": 30.0},
).fit()
```

## Fitting and control options

```python
fit = model.fit()                         # cached result if already fitted
fit = model.fit(control=saem_control(n_iter=800))
fit = model.fit(n_iter=800, verbose=True)  # copy and update the stored control
fit = model.refit(n_iter=800)              # always run a new fit
```

`fit(control=..., **control_kwargs)` cannot combine an explicit `control=` object
with individual control keywords. Without arguments it is idempotent and
returns the existing `SaemFit`. `refit()` clears the cache and returns a new
result. `SaemFit.fit()` returns itself.

### `SaemControl`

```python
from statcraft.saem import saem_control

control = saem_control(
    n_iter=800, n_burn=300, n_chains=2, seed=1,
    n_threads=0, verbose=True,
)
```

| Option | Default | Meaning |
|---|---|---|
| `n_iter` | `500` | Total SAEM iterations. |
| `n_burn` | `300` | Extra MCMC sweeps at iteration 0. |
| `n_mcmc` | `4` | MCMC sweeps per iteration after iteration 0. |
| `n_chains` | `2` | Independent MCMC chains per subject. |
| `seed` | random | 32-bit seed; results are reproducible and independent of `n_threads`. |
| `n_threads` | `0` | `0` = all hardware threads, `1` = serial. Applies only to compiled models. |
| `mstep` | `"auto"` | `"auto"`, `"sufficient_stats"` or `"gradient"`. |
| `simulated_annealing` | `True` | Use the simulated-annealing variance floor/kernel. |
| `map_kernel` | `True` | Use the MAP/Laplace-type kernel. |
| `map_every` | `25` | MAP refresh interval. |
| `auto_stop` | `False` | Stop early on parameter drift. |
| `stop_tol` | `1e-3` | Early-stop tolerance. |
| `adapt_rw` | `True` | Adapt the random-walk proposal. |
| `step_k1` | `-1` | Exploration length; negative means automatic. |
| `step_alpha` | `1.0` | Robbins–Monro smoothing exponent, in `(0.5, 1]`. |
| `verbose` | `False` | Print progress. |
| `print_every` | `100` | Progress interval. |
| `cond_mean` | `True` | Compute conditional means/EBEs with ESS and R-hat. |
| `n_cond` | `1000` | Conditional samples per subject. |
| `cond_burn` | `200` | Conditional-sampling burn-in. |
| `cond_thin` | `2` | Conditional-sampling thinning. |
| `se_method` | `"auto"` | Standard-error method. |
| `ll_method` | `"auto"` | Log-likelihood method. |
| `gq_nodes` | `15` | Outer Gauss–Hermite nodes. |
| `gq_inner_nodes` | `16` | Inner Gauss–Hermite nodes. |
| `is_samples` | `5000` | Importance-sampling draws for the likelihood. |
| `louis_draws` | `150` | Draws per subject for Louis' method. |
| `louis_is_samples` | `4000` | Importance samples for hierarchical Louis. |
| `rm` | `None` | Optional low-level Robbins–Monro override mapping. |
| `sa` | `None` | Optional low-level simulated-annealing override mapping. |

`SaemControl.copy(**changes)` returns a modified copy without touching the
original. Short aliases are accepted by `saem_control()` and `fit()`: `se` →
`se_method`, `ll` → `ll_method`, `iterations` → `n_iter`, `burn` → `n_burn`,
`threads` → `n_threads`, `chains` → `n_chains` and `sweeps` → `n_mcmc`. Supplying
both an alias and its real name is an error.

| Option | Valid values |
|---|---|
| `mstep` | `auto`, `gradient`, `sufficient_stats` |
| `se_method` | `auto`, `linearization`, `gauss_hermite`, `louis`, `laplace`, `none` |
| `ll_method` | `auto`, `gauss_hermite`, `importance_sampling`, `linearization`, `laplace`, `none` |
| `rm["schedule"]` | `constant`, `one_over_t`, `power_decay`, `kesten` |
| `sa["schedule"]` | `geometric`, `exponential`, `power`, `inverse` |

With `se_method="auto"`, linearization is used for uncensored, non-custom
models and Laplace otherwise. With `ll_method="auto"`, Gauss–Hermite quadrature
is used for up to three random effects and importance sampling beyond that;
hierarchical fits always switch to importance sampling.

The step size is `gamma_k = 1` for `k < K1` and
`(k - K1 + 1) ** -step_alpha` afterwards, where `K1 = step_k1` when positive,
otherwise `min(int(0.6 * n_iter), 300)` (shortened during annealing).

`rm` accepts `schedule`, `gamma0`, `alpha`, `t0`, `clip_gradient`, `grad_clip`,
`truncate`, `lower`, `upper`, `diagonal_precond`, `precond_decay`, `precond_eps`,
`averaging`, `avg_start`, `conv_window` and `conv_tol`. `sa` accepts `schedule`,
`beta0`, `rate`, `T`, `K`, `power`, `shrink`, `inflate_proposal` and
`diagonal_only`.

!!! note "Python defaults are authoritative"
    The C++ engine has different raw defaults; the Python `SaemControl` always
    supplies its translated settings. Older docstrings in the source still list
    the stale values, so use this table.

## Data preparation

- `id` defaults to a column named `id`; without one, each row becomes a distinct
  subject and a `SaemWarning` is issued. Random effects are not identifiable
  with one observation per subject.
- `time` defaults to a column named `time`; without one, a synthetic within-subject
  observation index named `obs` is created. Synthetic time is not allowed with
  events, built-in models, time-to-event models or callbacks that use `time`.
- `x_names` always starts with the time column. `predictors=` forces otherwise
  constant columns to be within-subject predictors; other numeric/categorical
  covariates must be constant within a subject.
- `evid=1` rows become dose events and `evid=0` rows remain observations.
  `events=` accepts a separate table. Missing event fields default to
  `cmt=1`, `rate=0`, `addl=0`, `ii=0` and `ss=0`; `amt` is required.
- Censoring codes are `-1` (left), `0` (observed) and `+1` (right).
- `ytype=` and `group=` are factorized with sorted labels. `group=` enables
  two-level hierarchical fitting.

## Results

`.fit()` returns `SaemFit`. `repr(fit)` is compact; `fit.summary()` renders the
full report.

### Properties and information criteria

| Property | Meaning |
|---|---|
| `res` | Raw `statcraft._saemfit.Results` object. |
| `model`, `info`, `control` | Structural model, prepared `DataInfo` and control actually used. |
| `params`, `warnings` | Parameter names and engine warnings. |
| `n_obs`, `nobs`, `n_subjects`, `n_params`, `df` | Sample, subject, parameter and free-parameter counts. |
| `loglik`, `aic`, `bic` | Log-likelihood and information criteria. `loglik` is `NaN` when not computed. |
| `fit()` | Return this result object; mainly useful for chained calls. |

### Fixed effects and inference

| Method | Returns | Purpose |
|---|---|---|
| `fixef(scale="estimation")` | `Series` | Fixed coefficients on the `"estimation"` (phi) or `"natural"` scale. Covariate coefficients stay on the estimation scale. |
| `pop_params()` | `Series` | Typical population parameters on the natural scale. |
| `vcov()` | `DataFrame` | Asymptotic covariance of fixed coefficients; raises when SEs were not computed. |
| `conf_int(alpha=0.05, scale="estimation")` | `DataFrame` | Wald intervals with columns `estimate`, `lower` and `upper`. |
| `anova(*others)` | `DataFrame` | AIC/BIC/log-likelihood comparison and likelihood-ratio tests, ordered by `df`. |

### Random effects, error model and fitted data

| Method | Returns | Purpose |
|---|---|---|
| `omega()` | `DataFrame` | Random-effect covariance on the phi scale. |
| `sigma()` | `DataFrame` | Error-model parameters per response with columns `model`, `a`, `b`, `c`. |
| `ranef(kind="eta")` | `DataFrame` | `"eta"` deviations, `"phi"` conditional parameters or `"psi"` natural-scale conditional parameters. |
| `coef()` | `DataFrame` | Alias of `ranef("psi")`. |
| `cluster_ranef()` | `DataFrame` | Cluster effects for hierarchical fits. |
| `fitted(level="individual")` | `Series` | Gaussian fitted values at the `"individual"` or `"population"` level. |
| `residuals(level="individual", type="raw", nsim=1000, seed=1)` | `Series` | `"raw"` or `"pearson"` Gaussian residuals; `type="npde"`/`"pd"`/`"quantile"` routes to `npde()`. |
| `predict(newdata=None, level="population", events=None, evid=None)` | `Series` | Gaussian population or individual predictions; requires matching ids for individual level. |
| `to_frame()` | `DataFrame` | Prepared data plus Gaussian `pred_pop`, `pred_ind`, `res_pop`, `res_ind` and `pres_ind`. |

### Simulation-based diagnostics

| Method | Returns | Purpose |
|---|---|---|
| `npde(nsim=1000, seed=1, decorrelate=False)` | `Series` | Normalized prediction-distribution errors for censored and non-Gaussian fits. `decorrelate=True` whitens subject trajectories empirically. |
| `simulate(nsim=1, seed=1, *, new_random_effects=True, lloq=None, uloq=None)` | `DataFrame` | Columns `sim_1...` and, with limits, `cens_1...`. |
| `simulate_events(tmax, repeated=False, nsim=1, seed=1, new_random_effects=True)` | `DataFrame` | Long-format event times for survival families such as `Weibull()`; `y=1` is an event. |

### Summary object

```python
summary = fit.summary(alpha=0.05, stars=False,
                      diagnostics=True,
                      title="Nonlinear Mixed-Effects Model Fit by SAEM")
```

`SaemSummary` inherits from `statcraft.summary.Summary`. It exposes `fit`,
`fixed`, `random`, `corr`, `error` and `stats`, the individual `tables[i].data`
frames, and `as_text()` / `as_html()` / `as_latex()` plus `save_text()`,
`save_html()`, `save_latex()` and `save_excel()` exporters. The printed report
includes fixed effects, random-effect SDs and correlations, cluster effects,
residual error, fit criteria and algorithm diagnostics.

!!! note "Fixed-effects scale"
    `summary()` reports fixed effects on the estimation (phi) scale. Use
    `fit.fixef(scale="natural")` or `fit.pop_params()` for natural-scale values.

## Plots

```python
fit.plot()                   # SAEM diagnostic mosaic (default)
fit.plot(kind="trace")       # SAEM convergence trace
fit.plot(kind="forest")
fit.plot(kind="npde")
fit.plot(kind="vpc")
fit.plot_vpc(nsim=200)       # generated alias for the same kind
```

Useful kinds are `diagnostics`, `saem_diagnostics`, `trace`, `saem_trace`,
`forest`, `residuals_vs_predictions`, `qq_plot_with_envelope`, `grouped_acf`,
`caterpillar`, `ranef_distribution`, `gof`, `individuals`,
`population_vs_conditional`, `npde` and `vpc`.

| Plot | Notable defaults |
|---|---|
| `diagnostics` / `saem_diagnostics` | Four-panel mosaic: population vs conditional predictions, Pearson residuals vs IPRED (Gaussian) or NPDE vs time (censored/non-Gaussian), Q–Q with a 95% simulation envelope, and an ACF accumulated within subjects. Aliases: `diagnose_saem`, `saem_panel`, `saem_mosaic`. |
| `saem_trace` | `max_params=12`; falls back to `forest` when no trace exists. |
| `npde` | `nsim=500`, `seed=42`; NPDE versus time plus a normal Q–Q panel. |
| `vpc` | `nsim=200`, `n_bins=8`, `seed=42`; P10/P50/P90 envelopes and a 90% CI for the simulated median. |
| `grouped_acf` | Within-subject ACF with a `max(n_i) - 1` lag cap and a `±1.96/√N` white-noise band. |

For censored fits the mosaic uses NPDE rather than Pearson residuals: the
engine stores the censoring limit in `res.data.y`, so Pearson residuals are NaN
for those rows. The prediction panel labels its observed points as censored
limits. The simulation-based panels accept `nsim=` and `seed=` (the mosaic's
Q–Q uses `n_sim=400` by default); all panels share one `seed=` so they show the
same NPDE draws.

`statcraft.plots.suggest_plots(fit)` filters simulation-based plots for custom
likelihood models and hierarchical fits. There is no `kind="individual"` alias;
use `individuals`, `ipred` or `individual_fits`. See
[Plotting and diagnostics](plotting.md) for the full registry and `PlotResult`.

## Gotchas

- Construction is unfitted; keep the object returned by `.fit()` (or use the
  original model, which caches the result).
- `fit(control=...)` cannot be combined with individual control keywords.
- `n_iter` is the total iteration count and `n_burn` is extra iteration-0 MCMC
  sweeps, not a count of burn-in SAEM iterations.
- `fitted()`, raw/Pearson `residuals()` and `predict()` are Gaussian-only. For
  censored or non-Gaussian families use `npde()`, `residuals(type="npde"|
  "quantile")` and the `npde` / `vpc` plots.
- Censoring codes are `-1` left, `0` observed and `+1` right; do not reverse
  them.
- Synthetic subject ids produce one observation per subject, so random effects
  are not identifiable. Supply a real `id` column.
- A callable that cannot be traced falls back to Python; `compile=True` turns
  that fallback into an error. `compile_info` explains the choice.
- `doses.superpose()` does not support steady-state or modeled rate/duration
  records.
- The option is `error=`, not `epsilon=`.
- `fitted()`/`predict()` for non-Gaussian models are intentionally unavailable;
  plotting adapters derive IPRED/PRED from simulation instead.

## See also

- [Model functions](model-functions.md) — shared `f(x, p)` conventions.
- [Mixed effects (NLME)](nlme.md) — formula-based mixed models.
- [Nonlinear least squares](nls.md) — single-curve fitting with bounds and
  multistart.
- [Data loading](data-loading.md), [Plotting](plotting.md) and
  [Troubleshooting](troubleshooting.md).
