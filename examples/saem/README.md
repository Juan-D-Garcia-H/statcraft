# statcraft.saem examples

Runnable examples for the stochastic approximation EM (SAEM) estimator of
nonlinear mixed-effects models. Each script is self-contained: it either loads
a dataset bundled in `statcraft/data/` or simulates one with a fixed seed, fits
a single model and prints the summary.

Run any example from the repository root:

```bash
python examples/saem/theo_covariates.py
```

All scripts resolve the data directory relative to their own location, so they
also work from another working directory as long as the repository layout is
intact.

As in the rest of `statcraft`, a model is just a specification until `.fit()`
is called. Result attributes (`fixef`, `ranef`, `summary()`, `plot()`, ...)
live on the object returned by `.fit()`:

```python
model = sf.SAEM(
    model=model1cpt, data=theo, id="Id", time="Time", response="Concentration",
    params=["ka", "CL", "V"], start={"ka": 1.5, "CL": 2.8, "V": 32.0},
)
fit = model.fit()
print(fit.summary())
```

## Continuous models

| Example | Data | Description |
| --- | --- | --- |
| `theo_covariates.py` | `theo.csv` | One-compartment oral PK model; weight centred at 70 kg and a sex effect on `CL` |
| `oxboys_growth.py` | `oxboys.csv` | Linear growth with a full intercept/slope random-effect covariance |
| `emax_censored.py` | `PD1.csv` | Emax dose-response with left-censored responses at the LLOQ |
| `custom_loglik.py` | `oxboys.csv` | User-supplied Student-t log-likelihood (robust regression with injected outliers) |

## Simulated multi-response and multi-level models

| Example | Description |
| --- | --- |
| `pk_multidose.py` | Repeated dosing through a separate `events=` table and `doses.superpose` |
| `hierarchical.py` | Two levels (patients within hospitals) with a `cluster_omega` |
| `pkpd_joint.py` | Simultaneous PK/PD fit driven by a `ytype` column and one error model per response |

## Discrete and time-to-event models

| Example | Data | Description |
| --- | --- | --- |
| `toenail_logistic.py` | `toenail.csv` | Bernoulli-logit model with a patient-level random intercept |
| `lung_weibull.py` | `lung.csv` | Weibull survival model with a random effect on the scale and a sex effect |
| `rapi_zip.py` | `rapi.csv` | Zero-inflated Poisson counts with a covariate equation on `b0` |
| `knee_ordinal.py` | `knee.csv` | Ordinal mixed model with estimated cut points and a random shift |

## Diagnostic plots

`fit.plot()` renders a dedicated 2x2 diagnostic mosaic. Every panel is also
available as `fit.plot_<kind>()`, for example `fit.plot_vpc(nsim=100)`;
`fit.plot(kind=...)` remains supported. The `kind` names are:

| `kind` | Figure |
| --- | --- |
| `diagnostics`, `saem_diagnostics` | Default mosaic: population vs conditional predictions, Pearson/IPRED or NPDE/time, a Q-Q with 95% envelope and a within-subject ACF |
| `trace`, `saem_trace` | SAEM convergence of every `theta` coordinate (small multiples) |
| `vpc` | Visual predictive check with simulated P10/P50/P90 envelopes |
| `npde` | NPDE vs time and NPDE Q-Q (censoring aware) |
| `ipred`, `individuals` | Per-subject observed, IPRED and PRED curves against time |
| `gof` | Observed response against PRED and IPRED |
| `residuals`, `qq` | Classic residual panels |

The simulation-based plots accept `nsim=` and `seed=`. Matplotlib is optional:

```bash
pip install "statcraft[viz]"
python examples/saem/diagnostics.py
```

## Reading the summary

Every SAEM summary has the same layout:

* a header with the model and family, the log-likelihood, AIC/BIC, the number of
  observations, subjects (and clusters, when `group=` is used), parameters and
  degrees of freedom, plus the method used for the log-likelihood and the
  standard errors;
* the fixed-effects table on the estimation scale with standard errors,
  z-values, p-values and confidence intervals;
* the random-effects table (`Random effects (SD)`), with a correlation table
  when `omega="full"`;
* the residual-error table(s), one per response when `ytype=` is used.

The header reports compact method labels (for example `observed FIM` or
`linearized FIM`). The full engine description is still available as
`fit.res.se_method` and `fit.res.loglik_method`.

Control options such as `n_iter`, `n_burn`, `n_chains`, `se_method` or
`ll_method` can be passed to the constructor or to `.fit()`:

```python
fit = model.fit(n_burn=300, n_iter=200, n_chains=1, se_method="linearization")
```
