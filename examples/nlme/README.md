# statcraft.nlme examples

Runnable examples for the linear and nonlinear mixed-effects models in
`statcraft.nlme`. Each script is self-contained: it loads a dataset bundled in
`statcraft/data/`, fits one model and prints the summary.

Run any example from the repository root:

```bash
python examples/nlme/lme.py
```

All scripts resolve the data directory relative to their own location, so they
also work from another working directory as long as the repository layout is
intact.

As in the rest of `statcraft`, a model is just a specification until `.fit()`
is called. Result attributes (`fixef`, `se`, `summary()`, ...) live on the
object returned by `.fit()`, not on the model itself:

```python
model = nl.lme("distance ~ age * Sex", orth, random="~ age | Subject")
result = model.fit()
print(result.summary())
```

## Mixed-effects models

| Example | Model | Data | Description |
| --- | --- | --- | --- |
| `lme.py` | `lme` | `Orthodont.csv` | Random intercept and slope per child, `age * Sex` fixed effects |
| `gls.py` | `gls` | `Ovary.csv` | Harmonic mean model with AR(1) errors within `Mare`, no random effects |
| `nlme.py` | `nlme` | `Orange.csv` | Logistic growth curve with a tree-level random `Asym` |
| `gnls.py` | `gnls` | `Soybean.csv` | Single logistic growth curve for all plots, no random effects |
| `callable_model.py` | `nlme`, `gnls` | `Orange.csv`, `Soybean.csv` | Model written as a Python function instead of a formula, with an analytical Jacobian |

For a **single curve** with parameter bounds, robust losses, multistart,
HC0-HC3 covariance, cross-validation or prediction intervals, use
`statcraft.nls` (see `examples/nls/README.md`). `gnls` is the right choice when
the residuals need a `correlation` or `weights` structure.

### Models written as Python functions

`nlme` and `gnls` accept a callable in place of the formula. The parameters are
those declared in `params`, or the keys of `start` when `params` is omitted;
the other positional arguments are the exogenous columns, in the same order,
and the function is called as `func(*exogenous, *parameters)`:

```python
def logistic(age, Asym, xmid, scal):
    return Asym / (1.0 + np.exp((xmid - age) / scal))

fit = nl.nlme(
    logistic,
    orange,
    response="circumference",
    random="Asym ~ 1 | Tree",
    start={"Asym": 190.0, "xmid": 730.0, "scal": 350.0},
).fit()
```

Use `predictors` when the argument names differ from the data columns, and
`jac=` to supply an analytical Jacobian. A `start` dict is always required for
callable models (there is no self-start heuristic without a formula).
`nls_list` still takes a formula with a `| group` term.

## Diagnostic plots

`fit.plot()` draws a 2x2 diagnostic panel: binned standardized residuals
against fitted values, a random-effects Q-Q (or residuals vs fitted for
`gls`/`gnls`), a residual Q-Q, and the empirical semi-variogram of the
residuals when a `correlation` structure is present. Individual panels are
available through `fit.plot_<kind>()`, for example
`fit.plot_binned_standardized_residuals()`; `fit.plot(kind=...)` remains
supported. Names: `binned-standardized-residuals`,
`qq-random`, `qq-residuals`, `variogram`, `residuals` and
`observed-vs-predicted`. Matplotlib is optional:

```bash
pip install "statcraft[viz]"
python examples/nlme/diagnostics.py
```

The example combines an `lme` random-slope fit on `Orthodont` with the `gls`
AR(1) fit on `Ovary`, whose variogram shows the serial dependence absorbed by
`corAR1`.

## Per-group fits

| Example | Model | Data | Description |
| --- | --- | --- | --- |
| `lm_list.py` | `lm_list` | `Orthodont.csv` | One linear fit per child (`distance ~ age`) |
| `nls_list.py` | `nls_list` | `Soybean.csv` | One logistic fit per plot |

## lme4-style syntax

| Example | Model | Data | Description |
| --- | --- | --- | --- |
| `lmer.py` | `lmer` | `sleepstudy.csv` | `Reaction ~ Days + (Days \| Subject)` |
| `glmer.py` | `glmer` | `bdf.csv` | Logistic model with a school-level random intercept |

## Reading the summary

The `lme`, `gls`, `nlme` and `gnls` summaries follow the R `nlme` layout:

* a header with the response, method, log-likelihood, AIC/BIC, residual
  standard error and, for mixed models, the number of groups, marginal and
  conditional R² and the intraclass correlation (ICC);
* the fixed-effects table with standard errors, test statistics, p-values and
  confidence intervals;
* one variance-components table per grouping level with `Variance`,
  `Std.Dev.` and the correlations with the preceding components, followed by
  the residual row;
* a summary of the standardized (within-group) residuals: `Min`, `Q1`,
  `Med`, `Q3` and `Max`.

The header lines, coefficient tables and extra tables are also available as
pandas objects through `result.summary().tables[i].data`, so they can be
exported with `to_csv()` or manipulated directly.

`lmer` and `glmer` keep the lme4-style summary, with the random-effect
standard deviations and correlations collected in per-group blocks. Both are
unfitted constructors: call `.fit()` to obtain `LmerFit`/`GlmerFit`. `glmer`
requires GLM-style families such as `Binomial("logit")` / `Poisson("log")`;
family strings are rejected.
