# statcraft examples

Runnable examples for every GLM family in `statcraft`. Each script is
self-contained: it loads a dataset bundled in `statcraft/data/`, fits one
model and prints the summary.

Run any example from the repository root:

```bash
python examples/glm/poisson.py
```

All scripts resolve the data directory relative to their own location, so
they also work from another working directory as long as the repository
layout is intact.

## Linear and Gaussian models

| Example | Family |
| --- | --- |
| `gaussian.py` | `Gaussian` |
| `student_t.py` | `StudentT` |
| `skew_normal.py` | `SkewNormal` |
| `skew_t.py` | `SkewT` |
| `truncated_gaussian.py` | `TruncatedGaussian` |

## Binary and categorical models

| Example | Family |
| --- | --- |
| `binomial.py` | `Binomial` |
| `multinomial.py` | `Multinomial` |
| `ordinal.py` | `Ordinal` |

## Count models

| Example | Family |
| --- | --- |
| `poisson.py` | `Poisson` |
| `negative_binomial.py` | `NegativeBinomial` |
| `hurdle_poisson.py` | `HurdlePoisson` |
| `hurdle_negative_binomial.py` | `HurdleNegativeBinomial` |
| `hurdle_geometric.py` | `HurdleGeometric` |
| `truncated_poisson.py` | `TruncatedPoisson` |
| `truncated_negative_binomial.py` | `TruncatedNegativeBinomial` |
| `truncated_geometric.py` | `TruncatedGeometric` |

## Continuous positive and bounded models

| Example | Family |
| --- | --- |
| `beta.py` | `Beta` |
| `simplex.py` | `Simplex` |
| `tweedie.py` | `Tweedie` |
| `hurdle_gamma.py` | `HurdleGamma` |
| `hurdle_lognormal.py` | `HurdleLogNormal` |
| `hurdle_beta.py` | `HurdleBeta` |

## Survival models (accelerated failure time)

| Example | Family |
| --- | --- |
| `lognormal.py` | `LogNormal` |
| `weibull.py` | `Weibull` |
| `gumbel.py` | `Gumbel` |
| `loglogistic.py` | `LogLogistic` |
| `generalized_gamma.py` | `GeneralizedGamma` |

## Zero- and value-inflated models

| Example | Family |
| --- | --- |
| `zero_inflated_poisson.py` | `ZeroInflatedPoisson` |
| `zero_inflated_negative_binomial.py` | `ZeroInflatedNegativeBinomial` |
| `value_inflated_poisson.py` | `ValueInflatedPoisson` |
| `value_inflated_negative_binomial.py` | `ValueInflatedNegativeBinomial` |

## Quasi-likelihood models

| Example | Family |
| --- | --- |
| `quasi_binomial.py` | `QuasiBinomial` |
| `quasi_poisson.py` | `QuasiPoisson` |
| `quasi_gaussian.py` | `QuasiGaussian` |
| `quasi_gamma.py` | `QuasiGamma` |
| `quasi_inverse_gaussian.py` | `QuasiInverseGaussian` |

## Diagnostic plots

`fit.plot()` (from `statcraft.plots`, optional `matplotlib`) draws a 2x2
diagnostic panel from randomized quantile residuals: residuals vs fitted,
a simulated-band normal Q-Q, a rootogram for count families and Cook's
distance. Individual panels are available through `fit.plot_<kind>()`, for
example `fit.plot_rootogram()`; `fit.plot(kind=...)` remains supported.
Names: `residuals`, `qq`, `pp`, `half-normal`, `worm`, `scale-location`, `rootogram`,
`cooks`, `dffits`, `leverage`, `observed-vs-predicted` and `coefficients`.

```bash
pip install "statcraft[viz]"
python examples/glm/diagnostics.py
```

`diagnostics.py` writes the Poisson and zero-inflated Poisson panels, including
the rootogram that shows whether the zero class is being captured.

## Notes

- Two-part models use the formula syntax `count_part | second_part`.
- `Truncated*` families receive their limits through `.fit(left_limit=..., right_limit=...)`.
- Survival families receive the event indicator through `.fit(cens=...)`. A string
  names a column of the model data (`cens="status"`); it can also be declared when
  building the model.
- `GeneralizedGamma` takes `sigma_formula` and `Q_formula` for its scale and shape
  submodels, e.g. `generalized_gamma("y ~ x", data, sigma_formula="~ 1", Q_formula="~ z")`.

## Multivariate covariance GLMs (`statcraft.mcglm`)

See [`mcglm/`](mcglm/) for runnable examples of multivariate and
mixed-effects models fitted with `statcraft.mcglm`.
