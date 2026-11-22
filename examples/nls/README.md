# statcraft.nls examples

Runnable examples for nonlinear least squares in `statcraft.nls`. Each script
is self-contained: it loads a dataset bundled in `statcraft/data/`, fits one or
more models and prints the results.

Run any example from the repository root:

```bash
python examples/nls/logistic_growth.py
```

All scripts resolve the data directory relative to their own location, so they
also work from another working directory as long as the repository layout is
intact.

`nls` and `NLS` return the fitted model itself, so `.fit()` can be chained and
the result is the same object with all the result attributes (`coefficients`,
`std_errors`, `summary()`, `predict()`, ...). Call `.fit()` first: accessing
a result attribute on an unfitted model raises `NotFittedError`.

```python
from statcraft.nls import nls

fit = nls("circumference ~ SSlogis(age, Asym, xmid, scal)", orange).fit()
print(fit.summary())
print(fit.predict(new_data, interval="prediction"))
```

## Which module should I use?

`statcraft` has two nonlinear-regression front ends. They share the formula
syntax and the self-starting `SS*` functions, but target different problems.

| | `statcraft.nls` | `statcraft.nlme` (`nlme`/`gnls`/`nls_list`) |
| --- | --- | --- |
| Unit of analysis | **one curve**, no groups | grouped data |
| Random effects | no | yes (`nlme`, `lmer`-style too) |
| Correlation / variance structures | no | yes (`corStruct`, `varStruct`) |
| Parameter bounds | yes | no |
| Robust losses | yes (`soft_l1`, `huber`, `cauchy`) | no |
| `fit_multistart` | yes | no |
| HC0-HC3 covariance | yes | no |
| `cross_validate` | yes | no |
| Prediction/confidence intervals | yes | point predictions only |
| Backend | NumPy, optional JAX | C++/Eigen |

Use `statcraft.nls` when you fit a single curve: a dose-response experiment, a
growth curve per treatment, a quick model comparison with AIC/BIC, or a fit
that must resist outliers. Use `statcraft.nlme` when the data are grouped
(repeated measures, plots within blocks, subjects) or when the residuals need
a correlation or variance function.

Both front ends accept a model written as a Python function, so that is not
what distinguishes them: `nlme`/`gnls` call it with `response=`, `predictors=`
and `params=`, while `nls`/`NLS` use `response=` and `predictors=` and return
the fitted model directly.

```python
# single curve, no groups
from statcraft.nls import nls

# grouped data, random effects
from statcraft.nlme import nlme
```

`nls` is feature-frozen: new capabilities for grouped data belong in
`statcraft.nlme`, and `nls` will not reimplement them.

## Model specification

| Example | Model | Data | Description |
| --- | --- | --- | --- |
| `logistic_growth.py` | `nls` + `SSlogis` | `Orange.csv` | Self-starting logistic growth, coefficient table and prediction bands |
| `michaelis_menten.py` | `NLS` | `Muscle.csv` | Michaelis-Menten curve from a plain Python function; compared with the formula interface |
| `two_exogenous.py` | `NLS` | `Wheat.csv` | `def f(X, p)` matrix signature with two exogenous columns |
| `biexponential_formula.py` | `nls` | `Cefamandole.csv` | Explicit biexponential formula, manual start, parameter bounds and confidence intervals |

## Fitting strategies

| Example | Description |
| --- | --- |
| `multistart_local_minimum.py` | A bad start lands in a local minimum; `fit_multistart()` recovers the global fit |
| `robust_loss.py` | `loss="soft_l1"`/`"huber"` resist injected outliers better than the default linear loss |
| `model_comparison.py` | `SSlogis`, `SSgompertz` and `SSweibull` compared with AIC/BIC and 5-fold cross-validation |
| `robust_covariance.py` | Classical vs HC0-HC3 heteroscedasticity-robust standard errors |

## Diagnostic plots

`fit.plot()` draws a 2x2 diagnostic panel (fitted curve with confidence and
prediction bands, residuals vs predictor, Cook's distance with influential
observations labelled, normal Q-Q). Individual panels are available through
`fit.plot_<kind>()`, for example `fit.plot_curve()`; `fit.plot(kind=...)`
remains supported. Names: `curve`, `residuals`, `qq`, `pp`, `half-normal`, `worm`,
`scale-location`, `cooks`, `dffits`, `leverage`, `observed-vs-predicted` and
`coefficients`. Matplotlib is optional:

```bash
pip install "statcraft[viz]"
python examples/nls/diagnostics.py
```

## Backend and persistence

| Example | Description |
| --- | --- |
| `persistence.py` | JSON `save()`/`load()` for formula models and `pickle` for `NLS` models |
| `jax_backend.py` | Optional `backend="jax"` with the analytical jacobian (skips if JAX is absent) |

## Self-starting functions

The formula interface recognises these `SS*` functions without a `start` dict:

| Function | Model |
| --- | --- |
| `SSlogis` | `Asym / (1 + exp((xmid - x) / scal))` |
| `SSlogis5` | 5-parameter logistic |
| `SSfpl` | 4-parameter logistic |
| `SSgompertz` | `a * exp(-b * c^x)` |
| `SSweibull` | `a - b * exp(-exp(c) * x^d)` |
| `SSasymp` | `Asym + (R0 - Asym) * exp(-exp(lrc) * x)` |
| `SSmicmen` | `Vm * x / (K + x)` |
| `SSexp` | `a * exp(b * x)` |
| `SSbiexp` | Sum of two exponentials |
| `SShill` | Hill equation |
| `SSgauss` | Gaussian bell |
| `SSlin`, `SSquad`, `SSpower` | Linear, quadratic and power models |

Use `FormulaParser.list_special_functions()` to print the registry at runtime:

```python
from statcraft.utils.FormulaParser import FormulaParser

FormulaParser.list_special_functions()
```

## Reading the summary

`summary()` follows the statsmodels layout:

* model information (formula, backend, covariance type, observations,
  residual degrees of freedom and number of parameters);
* fit criteria (R², adjusted R², F-statistic and p-value, log-likelihood,
  AIC and BIC);
* the coefficient table with standard errors, t-statistics, p-values and
  confidence intervals;
* residual diagnostics (Omnibus/Jarque-Bera normality tests, skewness,
  kurtosis, Durbin-Watson, and the condition number of the jacobian at the
  optimum).

`coef_table()` returns the coefficient table as a normal `pandas.DataFrame`
instead of the pre-rendered table inside `summary()`.
