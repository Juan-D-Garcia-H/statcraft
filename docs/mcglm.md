# Multivariate GLM (MCGLM)

MCGLM fits several correlated responses jointly, with a regression and variance
function per response and a covariance structure across responses or
observations. Model construction only stores the specification; `.fit()` runs
the engine and returns an `MCGLMResults` object.

## Entry points and model inputs

| Entry point | Signature | Returns |
|---|---|---|
| `mcglm` | `mcglm(formulas, data, z, **kwargs)` | Alias of `MCGLM.from_formula`; unfitted `MCGLM` |
| `MCGLM.from_formula` | `MCGLM.from_formula(formulas, data, z, **kwargs)` | Unfitted `MCGLM` |
| `MCGLM` | `MCGLM(endog, exog, z, *, link="identity", variance="constant", covariance="identity", power_fixed=True, ntrial=None, offset=None, weights=None, endog_names=None, exog_names=None, detect_diagonal=True)` | Matrix interface; unfitted `MCGLM` |

`**kwargs` on `from_formula` are the keyword options of `MCGLM`. Formula terms
fill `endog_names` and `exog_names` automatically; `endog_names` may be
overridden, but `exog_names` cannot be passed because the formula interface
always supplies it.

### Constructor options

| Argument | Default | Meaning |
|---|---|---|
| `endog` / `formulas` | required | Responses: a vector, list of vectors or 2-D array (one response per column) for `MCGLM`; one formula per response for `MCGLM.from_formula`. NaN responses receive weight 0. |
| `exog` / `data` | required | Design matrix (shared by all responses when a single matrix is given), or the data frame used to evaluate the formulas. |
| `z` | required | List of known symmetric matrices for the matrix linear predictor. A flat list is shared by all responses; a nested list gives each response its own matrices. Use `mc_id(n)` for independence. |
| `link` | `"identity"` | Link per response: `identity`, `log`, `logit`, `probit`, `cauchit`, `cloglog`, `loglog`, `sqrt`, `1/mu^2` or `inverse`. |
| `variance` | `"constant"` | Variance function per response: `constant`, `tweedie` (alias `power`), `binomialP`, `binomialPQ`, `poisson_tweedie` or `geom_tweedie`. |
| `covariance` | `"identity"` | Covariance link: `identity`, `inverse` or `expm`. |
| `power_fixed` | `True` | Hold variance exponents at their starting values instead of estimating them. May be a scalar or one value per response. |
| `ntrial` | `None` | Number of trials for binomial variance functions (scalar or vector; defaults to 1). |
| `offset` | `None` | Offset added to the linear predictor; one array or one per response. |
| `weights` | `None` | Non-negative weights of length `n` (reused for all responses) or `K * n` (stacked). |
| `endog_names`, `exog_names` | `None` | Labels for the summary. `exog_names` is a list of names for `K = 1`, and a list of lists otherwise. |
| `detect_diagonal` | `True` | Store a dense all-off-diagonal-zero `Z` as a diagonal matrix. |

Scalar settings may be given once and are repeated for every response; lists
supply per-response values.

## Formula interface

`statcraft.mcglm.mcglm` is an alias of `MCGLM.from_formula`:

```python
import statcraft.mcglm as scm
from statcraft.datasets import data_loader

soya = data_loader("soya.csv").load().astype(
    {"block": "category", "water": "category", "pot": "category"}
)
soya["viable_peas"] = soya["viablepeas"] / soya["totalpeas"]
n = len(soya)

fit = scm.mcglm(
    [
        "grain ~ block + water * pot",
        "seeds ~ block + water * pot",
        "viable_peas ~ block + water * pot",
    ],
    data=soya,
    z=[scm.mc_id(n)],
    link=["identity", "log", "logit"],
    variance=["constant", "tweedie", "binomialP"],
    ntrial=[None, None, soya["totalpeas"].values],
).fit()

print(fit.summary())
```

One formula is supplied per response. `z` is a list of covariance matrices;
`scm.mc_id(n)` gives an independence (identity blocks) structure.

### Direct matrix interface

```python
from statcraft.mcglm import MCGLM

fit = MCGLM(
    endog=[y1, y2],
    exog=[X1, X2],
    z=[scm.mc_id(n)],
    link=["identity", "log"],
    variance=["constant", "tweedie"],
    covariance="identity",
    power_fixed=True,
).fit()
```

`endog` may be a 2-D array (one response per column) or a list of vectors.
`exog` is reused for every response when a single matrix is given.

## Covariance matrices

Helpers are available directly from the `statcraft.mcglm` namespace:

| Helper | Signature | Structure |
|---|---|---|
| `mc_id` | `mc_id(n)` | Independent observations |
| `mc_diag` | `mc_diag(d)` | Diagonal weights |
| `mc_group` | `mc_group(codes, sparse=False)` | Group/block indicator |
| `mc_ma` | `mc_ma(n, order=1, groups=None, sparse=False)` | Moving average |
| `mc_dist` | `mc_dist(coords)` | Distance matrix |
| `mc_exp` / `mc_matern32` / `mc_wendland` | `(coords, range_)` | Spatial correlation |
| `mc_rw` / `mc_car` | `mc_rw(n, order=1, sparse=False)` / `mc_car(n, adj=None, sparse=False)` | Random walk and conditional autoregressive |
| `mc_ns` | `mc_ns(x, df=None, knots=None, degree=3)` | Natural spline basis |
| `mc_mixed` | `mc_mixed(formula=None, data=None, sparse=False, **kwargs)` | Build a matrix from a formula |
| `mc_dglm` | `mc_dglm(variance_structure=None, **kwargs)` | Dispersed GLM structure |
| `mc_twin`, `mc_twin_bio` | `mc_twin(n_pairs=None, twin_type="MZ", **kwargs)` | Twin structures; `mc_twin_bio` is an alias of `mc_twin` |
| `mc_twin_full` | `mc_twin_full(n_pairs=None, **kwargs)` | Full twin structure |

With `twin_type="MZ"`, `mc_twin` returns two identical `G` matrices, so its two
variance components are not identifiable without further constraints.

Set `sparse=True` on the helpers that support it when the matrices are large.
`model.structure(block_target=32, sparse_dense_max_n=4000)` reports the chosen
sparse/dense blocking before fitting; its result has `kind`, `n_blocks`,
`max_block`, `n_components`, `n_singletons` and `max_component`.

A flat `z` list is shared by every response; a nested list gives each response
its own variance matrices:

```python
# Two matrices shared by all responses
z_shared = [scm.mc_id(n), scm.mc_group(group)]

# Response 0: independence + group; response 1: independence + spatial decay
z_by_response = [
    [scm.mc_id(n), scm.mc_group(group)],
    [scm.mc_id(n), scm.mc_exp(coords, range_=1.0)],
]
```

## Model inspection

| Method | Returns | Purpose |
|---|---|---|
| `structure(block_target=32, sparse_dense_max_n=4000)` | `dict` | Report the chosen sparse/dense covariance organization. |
| `initial_values()` | `Params` | Return the automatic starting values for regression and covariance parameters. |

## Fitting options

The complete fitting contract is:

```python
fit = model.fit(
    method="chaser",
    max_iter=20,
    tol=1e-4,
    tuning=1.0,
    correct=True,
    compute_vcov=True,
    cross_block="r",
    r_diag_variability=True,
    start=None,
    callback=None,
    verbose=False,
    retry=(),
    y=None,
    weights=None,
    structure="auto",
    block_target=32,
    block_threads=0,
    sparse_dense_max_n=4000,
    n_probes=32,
    n_probes_final=128,
    probe_seed=12345,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `method` | `"chaser"` | Covariance update rule: `"chaser"`, `"rc"` or `"gradient"`. |
| `max_iter` | `20` | Maximum iterations; the starting point counts as the first. |
| `tol` | `1e-4` | Convergence tolerance on the largest absolute parameter change. |
| `tuning` | `1.0` | Step scaling of the covariance update; reduce for difficult problems. |
| `correct` | `True` | Include the bias-correction term in the Pearson estimating function. |
| `compute_vcov` | `True` | Compute the sandwich covariance needed for standard errors and tests. |
| `cross_block` | `"r"` | Cross-block derivative convention: `"r"` (R compatible) or `"exact"` (usually converges better). |
| `r_diag_variability` | `True` | Zero off-diagonal Pearson-variability terms, as the R package does. |
| `start` | `None` | `Params` object or dict with `beta` and `tau`, plus optional `power` and `rho`. Automatic when omitted. |
| `callback` | `None` | `callback(iteration, beta, cov)` after each iteration; return `True` to stop (sets `result.stopped`). |
| `verbose` | `False` | Print covariance parameters at every iteration. |
| `retry` | `()` | On numerical error, refit with `tuning` multiplied by each factor in turn. |
| `y` | `None` | Replace responses for this fit (stacked length `K * n`). |
| `weights` | `None` | Replace weights for this fit; also used for automatic starting values. |
| `structure` | `"auto"` | `"auto"`, `"dense"`, `"blocks"` or `"sparse"` covariance organization. |
| `block_target` | `32` | Merge small independent groups into batches of about this many units. |
| `block_threads` | `0` | Threads for block processing (OpenMP builds); `0` uses the OpenMP default. |
| `sparse_dense_max_n` | `4000` | Under `"auto"`, use the sparse path above this connected-block size with a sparse `Z`. |
| `n_probes` | `32` | Sparse method: random probes during iterations (minimum 2). |
| `n_probes_final` | `128` | Sparse method: probes for final sensitivity/covariance (minimum 8). |
| `probe_seed` | `12345` | Probe seed; fits are deterministic for a fixed seed. |

The sparse method applies to one response with an identity covariance link and
a sparse `Z`. Its estimates are exact; only covariance-parameter standard
errors use stochastic trace estimates, whose relative Monte Carlo error is
reported by `result.mc_error`.

## Results

`.fit()` returns `MCGLMResults`. Arrays are read-only views of engine memory.

### Tables, inference and prediction

| Method | Returns | Purpose |
|---|---|---|
| `summary(*, alpha=0.05, stars=False, title="MCGLM Results", diagnostics=True)` | `Summary` | Full formatted report. |
| `to_frame(response=None)` | `DataFrame` | Coefficient table (`coef`, and `std err`, `z`, `P>|z|` when available), optionally limited to one response. |
| `conf_int(alpha=0.05)` | array | Wald intervals, shape `(len(params), 2)`. |
| `wald_test(r_matrix, q_matrix=None)` | `dict` | Test `R * params = q`; returns `statistic`, `df` and `pvalue`. |
| `anova(other)` | `DataFrame` | Pseudo-likelihood-ratio comparison of nested fits on the same data. |
| `predict(exog=None, *, response=0, offset=None)` | array | Mean `g⁻¹(Xβ + offset)` for one response. |
| `get_prediction(exog=None, response=0, alpha=0.05)` | `DataFrame` | Mean with `mean_se`, `mean_ci_lower` and `mean_ci_upper`. |
| `covariance_matrix(full=False)` | `CovarianceOut` | `mu`, `diag_C` and, with `full=True`, dense `C`, `Sigma` and `Sigma_between`. |
| `cov_params()` | array | Joint sandwich covariance of all estimates. |
| `refit(**kw)` | `MCGLMResults` | Refit from these estimates; accepts any `MCGLM.fit()` keyword. |
| `plot(kind=None, ax=None, **kwargs)` | `PlotResult` | Diagnostic plot. |

The default `fit.plot()` mosaic puts the standardised residual Q-Q and binned
residuals in the top row. The bottom-left panel draws standardised residuals
against **within-response fitted quantiles**, with the same blue, orange and
green series colours used above, so responses on very different scales remain
comparable. The bottom-right `covariance` panel splits correlations (ρ) from
dispersion/power parameters (τ): ρ is shown on a fixed `[-1, 1]` axis above τ on
its natural scale. Both layouts are also available as individual kinds,
`standardized_residuals_vs_fitted` and `covariance`.

`predict(exog=..., response=k)` expects an already built **numeric** `(m, p)`
design matrix, not a formula or a data frame, with `p == len(beta[k])`. If you
fit from formulas, build the new design matrix with your formula tool first.
`offset=`, when supplied, must have one value per new row.

`MCGLMResults.to_frame()` is the coefficient table. GLM's `coef_table()` and its
`summary_frame()`/`to_dataframe()` aliases do not exist here; likewise, MCGLM's
`to_frame()` also includes variance/covariance parameters when present.

### Parameter and covariance arrays

| Property | Meaning |
|---|---|
| `params`, `param_names`, `beta`, `cov` | All estimates (`beta`, `rho`, power, `tau`), their labels, regression coefficients alone, and covariance parameters alone. |
| `bse`, `zvalues`, `pvalues` | Standard errors, Wald statistics and two-sided p-values. |
| `rho`, `power`, `tau` | Between-response correlations, variance exponents and dispersions. |
| `params_struct` | Estimates as a `Params` object usable as `start` for another fit. |
| `cov_params()` | Joint covariance matrix (method). |

### Fit state, diagnostics and residuals

| Property | Meaning |
|---|---|
| `converged`, `stopped`, `iterations`, `history` | Convergence flag, callback-stop flag, iteration count and per-iteration `beta`, `cov`, `score_beta`, `score_cov`. |
| `nobs`, `df`, `pseudo_loglik`, `pAIC`, `pBIC` | Sample size, number of estimated parameters, pseudo log-likelihood and pseudo information criteria. |
| `mc_error` | Largest relative Monte Carlo standard error of sparse traces (`0` for exact methods). |
| `structure` | Chosen covariance organization: `kind`, `n_blocks`, `max_block`. |
| `variability`, `sensitivity` | Final `V_beta`, `V_cov` and `inv_S_beta`, `S_cov` matrices. |
| `fitted`, `fitted_by_response` | Fitted means, stacked and split by response. |
| `resid`, `resid_by_response` | Raw residuals, stacked and split by response. |
| `resid_pearson`, `resid_standardized` | Pearson and Cholesky/Mahalanobis residuals. |
| `model`, `raw` | The fitted `MCGLM` and the low-level engine result. |

`params` is stacked as `[all beta, rho, then for each response (power, tau)]`.
For more than one response, `rho` follows the upper-triangle order `(1, 2),
(1, 3), ..., (K-1, K)`. Within response `k`, `tau[k][j]` belongs to `z[k][j]`
(or to entry `j` of the shared flat `z`). `param_names` is the authoritative
order for `params`, `bse`, `conf_int()` and `wald_test()`.

`pAIC` and `pBIC` are pseudo-information criteria based on the quasi-likelihood;
they are useful for comparing covariance structures, not Gaussian likelihoods.

## Gotchas

- `fitted`, `resid`, `history`, `sensitivity` and `variability` are **properties**,
  not methods: use `fit.fitted`, not `fit.fitted()`.
- `compute_vcov=False` makes standard errors, `conf_int()`, `wald_test()` and
  prediction intervals unavailable.
- With `structure="sparse"`, only one response, `covariance="identity"` and a
  sparse `Z` are supported.
- `start` is a dict with at least `beta` and `tau`; `rho` is ignored for a single
  response.
- Calling `.fit()` again on the same model reuses the stored design matrices;
  pass `y=` or `weights=` to fit a different response vector.

## See also

- [Data loading](data-loading.md) — locate bundled example data.
- [Plotting and diagnostics](plotting.md) — plot kinds and `PlotResult`.
- [Generalized linear models](glm.md) — single-response GLMs.
- [Troubleshooting](troubleshooting.md).
