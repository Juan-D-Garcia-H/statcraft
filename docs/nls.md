# Nonlinear least squares (NLS)

`statcraft.nls` is a subpackage. Import the two estimators explicitly:

```python
from statcraft.nls import nls, NLS
```

- `nls` takes an R-style **formula**.
- `NLS` takes a plain Python **callable**.

Both constructors return an **unfitted** model. `.fit()` estimates the
parameters and returns the same object, now fitted. Result properties raise
`NotFittedError` before fitting.

## Entry points and model inputs

| Entry point | Signature | State |
|---|---|---|
| `nls` | `nls(formula, data, start=None, backend="numpy", extra_functions=None)` | Formula interface; unfitted |
| `NLS` | `NLS(func, data, response, predictors=None, start=None, n_params=None, backend="numpy")` | Callable interface; unfitted |

### Constructor options

| Argument | Default | Applies to | Meaning |
|---|---|---|---|
| `formula` | required | `nls` | `"y ~ f(x, a, b)"`. Free right-hand-side symbols that are not data columns become parameters. |
| `data` | required | both | `pandas.DataFrame` containing the response and exogenous columns. |
| `start` | `None` | both | Initial values. For `nls` a dict keyed by parameter name; when omitted, `Initializer` derives heuristic values. For `NLS` a dict, list or array; its key order supplies parameter names for two-argument callables. |
| `backend` | `"numpy"` | both | `"numpy"` (finite-difference Jacobian) or `"jax"` (analytical/autodiff Jacobian). JAX is optional and must be installed separately. |
| `extra_functions` | `None` | `nls` | Extra callables made available to the formula namespace, for example custom transforms. |
| `func` | required | `NLS` | Callable model function. See [Model function signatures](model-functions.md). |
| `response` | required | `NLS` | Name of the response column. |
| `predictors` | `None` | `NLS` | Name(s) of the exogenous column(s). Required when a named-argument function has more than two arguments. |
| `n_params` | `None` | `NLS` | Number of parameters for `f(x, p)` when `start` is not supplied; initial values default to `1.0`. |

### Formula interface

```python
from statcraft.nls import nls
from statcraft.datasets import data_loader

muscle = data_loader("Muscle.csv").load()

fit = nls(
    "length ~ Vm * conc / (K + conc)",
    muscle,
    start={"Vm": 200.0, "K": 0.1},
).fit()
```

With `start=None`, `Initializer` uses data-derived heuristics only for
registered self-start `SS*` model names such as `SSlogis`. Other expressions
start every free parameter at `1.0`; supply `start=` when that is not a useful
guess.

`extra_functions` is scoped to one formula. A plain callable is enough for
numerical evaluation; a `(numpy_callable, sympy_func)` tuple also gives SymPy a
symbolic representation:

```python
import sympy as sp

def richards(x, A, K, x0, r, nu):
    return A / (1.0 + np.exp(-r * (x - x0))) ** (1.0 / nu)

fit = nls(
    "y ~ my_richards(x, A, K, x0, r, nu)",
    data,
    extra_functions={"my_richards": (richards, sp.Function("my_richards"))},
    start={"A": 10.0, "K": 1.0, "x0": 5.0, "r": 0.5, "nu": 1.0},
).fit()
```

`extra_functions` cannot shadow a built-in namespace function.

### Callable interface

The recommended signature is `f(x, p)`:

```python
from statcraft.nls import NLS

def michaelis_menten(x, p):
    conc, = x
    Vm, K = p
    return Vm * conc / (K + conc)

fit = NLS(
    michaelis_menten,
    muscle,
    response="length",
    predictors="conc",
    start={"Vm": 200.0, "K": 0.1},
).fit()
```

`predictors` chooses the columns in `x`; `start` names the entries of `p`. Both
views can be unpacked, indexed by name (`x["conc"]`, `p["Vm"]`), indexed by
position, or converted with `np.asarray`. With one predictor, `x` can be used
directly; a single-entry parameter view behaves as its value, so `a = p` and
`a, = p` are both valid. For a two-argument callable with several parameters,
the names of `p` are arbitrary: only their order matters.

The older named form `def f(conc, Vm, K)` is called positionally as
`func(*exogenous, *parameters)`; `predictors` is then required. See
[Model function signatures](model-functions.md) for all supported forms.

## Fitting options

The complete fitting contract is:

```python
fit = model.fit(
    method="trf",
    loss="linear",
    use_analytical_jacobian=None,
    cov_type="nonrobust",
    raise_on_non_convergence=False,
    **kwargs,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `method` | `"trf"` | Algorithm passed to `scipy.optimize.least_squares`: `"trf"`, `"dogbox"` or `"lm"`. `"lm"` only supports `loss="linear"`. |
| `loss` | `"linear"` | Robust loss: `"linear"`, `"soft_l1"`, `"huber"`, `"cauchy"` or `"arctan"`. |
| `use_analytical_jacobian` | `None` | `None` uses JAX autodiff with `backend="jax"` and finite differences otherwise. `True`/`False` force the choice; `True` requires JAX. |
| `cov_type` | `"nonrobust"` | `"nonrobust"`, `"HC0"`, `"HC1"`, `"HC2"` or `"HC3"`. HC variants use heteroscedasticity-consistent sandwich covariance; HC3 is conservative in small samples. |
| `raise_on_non_convergence` | `False` | `False` warns and keeps the (inspectable) non-converged fit; `True` raises `ConvergenceError`. |
| `**kwargs` | — | Forwarded to `scipy.optimize.least_squares`: commonly `bounds`, `x_scale`, `f_scale`, `max_nfev`, `ftol`, `xtol` and `gtol`. |

Pass bounds as a pair of lower/upper sequences, in parameter order (a
`scipy.optimize.Bounds` object is also accepted):

```python
fit = model.fit(bounds=([0.0, 0.0], [np.inf, np.inf]))
```

`method="lm"` does not support bounds; use `"trf"` or `"dogbox"`. The same
`bounds=` argument on `fit_multistart()` also clips perturbed starting values.

For a compiled analytical Jacobian, install JAX and build with
`backend="jax"`:

```bash
python -m pip install jax
```

```python
fit = NLS(michaelis_menten, muscle, response="length",
          predictors="conc", backend="jax").fit()
```

!!! note "JAX is optional"
    It is not declared as a dependency. With `backend="numpy"` the Jacobian is
    computed by finite differences and everything works without JAX.

### Multistart and cross-validation

```python
best = model.fit_multistart(n_starts=20, random_state=1)
cv = model.cross_validate(k=5, random_state=1)
```

`fit_multistart(n_starts=10, perturbation=0.5, method="trf", loss="linear",
use_analytical_jacobian=None, random_state=None, bounds=None,
cov_type="nonrobust", raise_on_non_convergence=False, **kwargs)` tries
`n_starts` perturbed initial values and keeps the lowest-cost converged fit. It
returns the fitted model itself. Starting values are perturbed log-normally,
`start * exp(N(0, perturbation))`, with an additive fallback for zeros;
`bounds` restricts the search. If no candidate converges, the lowest-cost
non-converged fit is applied with a warning, or raises when
`raise_on_non_convergence=True`.

`cross_validate(k=5, method="trf", loss="linear",
use_analytical_jacobian=None, random_state=None, **kwargs)` performs real
K-fold refits. It returns a dict with per-fold `rmse`, `mae` and `r2` lists,
their means (`rmse_mean`, `mae_mean`, `r2_mean`) and `n_folds_failed`. It does
not fit the model permanently. If every fold fails, it warns and returns NaN
for the mean metrics.

## Results

The fitted object is the same `nls` / `NLS` instance returned by `.fit()`.
`repr(fit)` is compact; `fit.summary()` renders the full report.

### Tables, inference and prediction

| Method | Returns | Purpose |
|---|---|---|
| `summary(yname=None, xname=None, title=None, alpha=0.05)` | `Summary` | Full statsmodels-style report with visual text/HTML output and `.tables`. |
| `coef_table(alpha=0.05)` | `DataFrame` | Coefficients, standard errors, t statistics, p-values and confidence bounds. |
| `conf_int(alpha=0.05)` | `DataFrame` | Parameter confidence intervals using the t distribution with `resid_df` degrees of freedom. |
| `predict(newdata=None, interval=None, alpha=0.05)` | array / `DataFrame` | Point predictions, or a `fit`/`se`/`lower`/`upper` frame when `interval` is `"confidence"` (mean) or `"prediction"` (new observation). |
| `save(path)` | `None` | Save `to_dict()` as JSON. Only available when the model has no `extra_functions`. |
| `to_dict()` / `nls.from_dict(d)` | `dict` / fitted model | Serialize or restore a formula model without a file. JSON is invalid for `NLS` and for any `nls(..., extra_functions=...)`; use Python `pickle` for those models. |

### Fit objects and arrays set by `.fit()`

| Attribute | Meaning |
|---|---|
| `result` | Full `scipy.optimize.OptimizeResult`. |
| `coefficients` | Dict of estimated parameters keyed by name. |
| `params` | Estimated parameter values in model order. |
| `fitted_values`, `residuals` | Arrays of fitted values and `y - fitted`. |
| `covariance_matrix` | Parameter covariance matrix (classical or HC0–HC3). |
| `std_errors` | Square roots of the covariance diagonal. |
| `cov_type` | Covariance estimator used in the last fit. |

### Scalar diagnostics

| Property | Meaning |
|---|---|
| `rss`, `tss`, `resid_df` | Residual sum of squares, total sum of squares, residual degrees of freedom (`n - k`). |
| `rmse`, `mae`, `nrmse` | RMSE, mean absolute error and RMSE normalized by the response range. |
| `rsquared`, `adj_rsquared` | R² and degrees-of-freedom-adjusted R². Without an intercept, R² is a heuristic and may be negative. |
| `press`, `rsquared_pred` | Leverage-based leave-one-out PRESS and predictive R². For actual refits use `cross_validate()`. |
| `logLik`, `aic`, `bic`, `aicc` | Gaussian log-likelihood and information criteria using `k + 1` degrees of freedom. `aicc` is preferred when `n / K` is small. |
| `fvalue`, `f_pvalue` | F statistic against the mean-only null model and its p-value. |

### Influence and plots

`leverage` (hat values from the Jacobian SVD), `cooks_distance` and
`predict(newdata, interval=...)` provide influence and uncertainty diagnostics.
`fit.plot(kind=None, ax=None, **kwargs)` returns a `PlotResult`; call
`statcraft.plots.suggest_plots(fit)` for recommended kinds. See
[Plotting and diagnostics](plotting.md).

## Gotchas

- Construction is unfitted; always keep the return value of `.fit()` (or use
  `model.fit()` on the same object).
- `method="lm"` rejects any loss other than `"linear"`.
- HC0–HC3 change standard errors, intervals and the coefficient table, not the
  parameter estimates.
- `rsquared`, `press`, `leverage` and `cooks_distance` use a local linearization
  at the optimum; they are approximations for nonlinear models.
- `cross_validate()` rebuilds an equivalent model for each fold (preserving the
  backend and any `extra_functions`); it does not mutate the fitted model.

## See also

- [Model functions](model-functions.md) — shared callable conventions.
- [Data loading](data-loading.md) — locate bundled example data.
- [Mixed effects (NLME)](nlme.md) — grouped data, random effects, correlation
  and variance structures.
- [Plotting and diagnostics](plotting.md) and [Troubleshooting](troubleshooting.md).
