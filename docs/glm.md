# Generalized linear models (GLM)

## Two entry points

For R-style formulas, import the **module** and use `scg.glm`:

```python
import statcraft.glm as scg
from statcraft.datasets import data_loader

affairs = data_loader("Affairs.csv").load()

model = scg.glm(
    "affairs ~ age + yearsmarried + religiousness + rating",
    data=affairs,
    family=scg.Poisson(),
)          # unfitted
fit = model.fit()
print(fit.summary())
```

`scg.glm` is an alias of `GLM.from_formula`. Family-specific shortcut functions
such as `scg.poisson("y ~ x", df)` also exist.

For pre-built design matrices, import the class directly:

```python
from statcraft.glm import GLM, Binomial

fit = GLM(X, y, family=Binomial()).fit()
```

!!! warning "`statcraft.glm` is a module"
    `import statcraft as sf; sf.glm(...)` does not work. Use `import statcraft.glm
    as scg` for formulas or `sf.GLM(...)` / `from statcraft.glm import GLM` for
    matrices. Family classes such as `sf.Binomial` and shortcuts such as
    `sf.poisson` are available directly from `import statcraft as sf`.

## Entry points and model inputs

| Entry point | Signature | State |
|---|---|---|
| `scg.glm` | `glm(formula, data, family=None, weights=None, offset=None, cens=None, entry=None, **kwargs)` | Alias of `GLM.from_formula`; unfitted |
| `GLM` | `GLM(X, y, family=None, weights=None, offset=None, tol=1e-8, maxit=1000, solver="colpivqr", big_matrix=False, firth=False, feature_names=None, ftol=1e-8, gtol=1e-6, maxcor=10, verbose=0, compute_se=True, minimizer=None, use_t=None, standardize=True, y_levels=None)` | Matrix interface; unfitted |
| `GLM.from_formula` | `GLM.from_formula(formula, data, family=None, weights=None, offset=None, cens=None, entry=None, **kwargs)` | Formula interface; unfitted |
| `scg.<shortcut>` | `shortcut(formula, data, link=None, weights=None, offset=None, **kwargs)` | Builds a family and calls `GLM.from_formula`; unfitted |
| `scg.generalized_gamma` | `generalized_gamma(formula, data, sigma_formula="~ 1", Q_formula="~ 1", link_mu="identity", link_sigma="log", link_Q="identity", weights=None, offset=None, cens=None, entry=None, **kwargs)` | Three-submodel formula interface; unfitted |
| `OLS` | `OLS(X, y, **kwargs)` | `GLM(X, y, family=Gaussian(), **kwargs)`; unfitted |

`**kwargs` on the formula and shortcut interfaces are forwarded to
`GLM.__init__`. All models are unfitted until `.fit()` is called.

### Constructor options

| Argument | Default | Meaning |
|---|---|---|
| `X`, `y` | required | Design matrix and response for the matrix interface. |
| `formula`, `data` | required | R-style formula and data frame for `scg.glm` / `GLM.from_formula`. |
| `family` | `None` (`Gaussian()`) | A family instance such as `scg.Poisson()`. The model keeps its own copy. |
| `weights` | `None` | Prior/case weights; a column name or array of length `n`. |
| `offset` | `None` | Added to the linear predictor before the link; column name or array. |
| `tol` | `1e-8` | Convergence tolerance for IRLS, negative binomial and quasi engines. |
| `maxit` | `1000` | Maximum iterations for every engine. |
| `solver` | `"colpivqr"` | Linear-algebra backend: `"qr"`, `"colpiv_qr"`, `"colpivqr"`, `"fullpiv_qr"`, `"fullpivqr"`, `"cholesky"`, `"llt"`, `"chol"`, `"ldlt"` or `"svd"`. |
| `big_matrix` | `False` | Use the engine path intended for large matrices. |
| `firth` | `False` | Firth penalization for IRLS models. |
| `feature_names` | `None` | Labels for `X` columns; defaults to `x0`, `x1`, ... |
| `ftol`, `gtol` | `1e-8`, `1e-6` | Function/gradient tolerances for MLE and `QuasiCustom` optimizers. |
| `maxcor` | `10` | Quasi-Newton memory used by the MLE optimizer. |
| `verbose` | `0` | Optimizer progress level. |
| `compute_se` | `True` | Compute standard errors and covariance when supported. |
| `minimizer` | `None` | `"lbfgsb"`, `"bfgs"` or `"neldermead"`; `None` resolves to BFGS. Ignored by IRLS families. |
| `use_t` | `None` | `None` uses t statistics with estimated dispersion and normal statistics otherwise; `True`/`False` force the choice. |
| `standardize` | `True` | Internally standardize continuous columns for conditioning; coefficients and errors are returned on the original scale. |
| `y_levels` | `None` | Explicit response-level order for `Ordinal`, lowest to highest. |
| `cens` | `None` | Formula-only; event indicator (1 = event, 0 = right-censored) or data-column name. |
| `entry` | `None` | Formula-only; delayed-entry/left-truncation times for `GeneralizedGamma`. |

Shortcuts accept `link=` (mapped to `mu_link` for `Beta`/`Simplex`) and the
secondary-link keywords `phi_link=`, `sigma_link=`, `zero_link=` and
`infl_link=`. The shortcut `negative_binomial(..., theta)` requires `theta`,
while the `NegativeBinomial(theta=None)` family estimates it; `tweedie(...,
var_power)` and the `ValueInflated*` shortcuts require `var_power` and
`target_value` respectively. `sigma_formula=` and `Q_formula=` belong to
`generalized_gamma(...)` / `GLM.from_formula_gengamma(...)`, not to `.fit()`.

## Families

| Group | Classes |
|---|---|
| Core | `Gaussian`, `Binomial`, `Poisson`, `Gamma`, `InverseGaussian`, `NegativeBinomial`, `Tweedie`, `Exponential`, `Pareto`, `InverseGamma` |
| Robust / asymmetric / bounded | `Beta`, `Simplex`, `StudentT`, `SkewNormal`, `SkewT`, `LogisticDistribution`, `Gumbel`, `GeneralizedGamma` |
| Zero-inflated and hurdle | `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomial`, `HurdlePoisson`, `HurdleNegativeBinomial`, `HurdleGeometric`, `HurdleGamma`, `HurdleBeta`, `HurdleLogNormal` |
| Value-inflated | `ValueInflatedPoisson`, `ValueInflatedNegativeBinomial` |
| Truncated | `TruncatedPoisson`, `TruncatedGeometric`, `TruncatedNegativeBinomial`, `TruncatedGaussian` |
| Survival | `Weibull`, `LogNormal`, `LogLogistic`, `GeneralizedGamma` |
| Categorical | `Multinomial`, `Ordinal` |
| Quasi / custom | `QuasiPoisson`, `QuasiBinomial`, `QuasiGamma`, `QuasiGaussian`, `QuasiInverseGaussian`, `QuasiCustom`, `CustomFamily` |

### Family constructor arguments

Every family is constructed before fitting, for example
`family=scg.TruncatedPoisson(link="log")`. Links are normally case-insensitive
strings and an invalid link raises `ValueError` listing the valid ones.
Exceptions: `TruncatedGaussian` validates its link before lowercasing, so pass
lowercase; `ValueInflated*` `base_link` and `GeneralizedGamma` submodel links
are only checked when the model is fitted. The tables below use two shorthand
sets for secondary links:

- `PROB`: `logit`, `probit`, `cloglog`, `loglog`, `cauchit`
- `POS`: `log`, `identity`, `sqrt`, `inverse`

`link_power` is only accepted with `link="power"` and is required there
(`mu = eta**(1 / link_power)`, with `0` meaning `log`).

#### IRLS families

| Family | Constructor arguments | Main link values |
|---|---|---|
| `Gaussian` / `OLS` | `link="identity"` | `identity`, `log`, `inverse`, `sqrt` |
| `Binomial` | `link="logit"` | `logit`, `probit`, `cloglog`, `log`, `loglog`, `cauchit`, `logit_log` |
| `Poisson` | `link="log"` | `log`, `identity`, `sqrt`, `inverse` |
| `Gamma` | `link="inverse"`, `link_power=None` | `log`, `inverse`, `identity`, `sqrt`, `power` |
| `InverseGaussian` | `link="invmu2"`, `link_power=None` | `invmu2`, `log`, `identity`, `inverse`, `power` |
| `NegativeBinomial` | `theta=None`, `link="log"`, `link_power=None` | `log`, `sqrt`, `identity`, `inverse`, `power` |
| `Tweedie` | `var_power`, `link="log"`, `link_power=None` | `log`, `identity`, `inverse`, `sqrt`, `power` |
| `Exponential` | `link="log"` | `log`, `identity`, `inverse`, `sqrt` |
| `Pareto` | `link="log"` | `log`, `inverse` |
| `InverseGamma` | `link="inverse"` | `log`, `inverse`, `identity` |
| `CustomFamily` | `variance_fn`, `mu_eta_fn`, `linkinv_fn`, `dev_resids_fn`, `valideta_fn`, `validmu_fn` | supplied by the callables |

`NegativeBinomial(theta=None)` estimates the dispersion; pass `theta=` to fix
it. `Tweedie(var_power=...)` requires `var_power` (typically `1 < p < 2`).
`Binomial(link="logit_log")` is accepted for backward compatibility as an exact
alias of `logit`; prefer `logit` in new code.

#### Quasi families

| Family | Constructor arguments | Main link values |
|---|---|---|
| `QuasiPoisson` | `link="log"` | `log`, `identity` |
| `QuasiBinomial` | `link="logit"` | `logit`, `probit` |
| `QuasiGamma` | `link="log"` | `log`, `identity`, `inverse` |
| `QuasiGaussian` | `link="identity"` | `identity`, `log` |
| `QuasiInverseGaussian` | `link="log"` | `log`, `identity`, `inverse` |
| `QuasiCustom` | `variance_fn`, `link="log"` | `log`, `identity` |

#### MLE families with one linear predictor

| Family | Constructor arguments | Main link values |
|---|---|---|
| `Beta` | `mu_link="logit"`, `phi_link="log"` | `logit`, `probit`, `cloglog`, `loglog`, `cauchit` |
| `Simplex` | `mu_link="logit"`, `sigma_link="log"` | `logit`, `probit`, `loglog`, `cloglog`, `cauchit` |
| `LogNormal` | `link="identity"` | `identity`, `log`, `inverse` |
| `Weibull` | `link="log"` | `log`, `identity` |
| `LogLogistic` | `link="identity"` | `identity`, `log` |
| `Gumbel` | `link="identity"` | `identity`, `log` |
| `StudentT` | `link="identity"` | `identity`, `log` |
| `SkewNormal` | `link="identity"` | `identity`, `log` |
| `SkewT` | `link="identity"` | `identity`, `log` |
| `LogisticDistribution` | `link="identity"` | `identity`, `log` |
| `Multinomial` | `link="logit"` | `logit`, `probit`, `cloglog`, `loglog`, `cauchit` |
| `Ordinal` | `link="logit"` | `logit`, `probit`, `cloglog`, `loglog`, `cauchit` |
| `TruncatedPoisson` | `link="log"` | `log`, `identity` |
| `TruncatedNegativeBinomial` | `link="log"` | `log`, `identity` |
| `TruncatedGeometric` | `link="log"` | `log` |
| `TruncatedGaussian` | `link="identity"` | `identity`, `log`, `inverse` |

`Beta` models precision `phi` through `phi_link`; `Simplex` models dispersion
`sigma` through `sigma_link`. Both secondary links accept `POS`.

#### MLE families with a second linear predictor

| Family | Constructor arguments | Main link values | Secondary link |
|---|---|---|---|
| `ZeroInflatedPoisson` | `link="log"`, `infl_link="logit"` | `log`, `identity`, `sqrt`, `inverse` | `infl_link`: `PROB` |
| `ZeroInflatedNegativeBinomial` | `link="log"`, `infl_link="logit"` | `log`, `sqrt`, `identity`, `inverse` | `infl_link`: `PROB` |
| `HurdlePoisson` | `link="log"`, `zero_link="logit"` | `log`, `sqrt` | `zero_link`: `PROB` |
| `HurdleNegativeBinomial` | `link="log"`, `zero_link="logit"` | `log`, `sqrt` | `zero_link`: `PROB` |
| `HurdleGamma` | `link="log"`, `zero_link="logit"` | `log`, `inverse` | `zero_link`: `PROB` |
| `HurdleLogNormal` | `link="log"`, `zero_link="logit"` | `log`, `identity` | `zero_link`: `PROB` |
| `HurdleBeta` | `link="logit"`, `zero_link="logit"` | `logit`, `probit` | `zero_link`: `PROB` |
| `HurdleGeometric` | `link="log"`, `zero_link="logit"` | `log`, `identity`, `sqrt` | `zero_link`: `PROB` |
| `ValueInflatedPoisson` | `target_value`, `base_link="log"`, `infl_link="logit"` | `base_link`: `log`, `identity`, `sqrt`, `inverse` | `infl_link`: `PROB` |
| `ValueInflatedNegativeBinomial` | `target_value`, `var_form="quadratic"`, `base_link="log"`, `infl_link="logit"` | `base_link`: `log`, `identity`, `sqrt`, `inverse` | `infl_link`: `PROB` |
| `GeneralizedGamma` | `mx_mu_cols=()`, `mx_sigma_cols=()`, `mx_Q_cols=()`, `link_mu="identity"`, `link_sigma="log"`, `link_Q="identity"` | MLE links for `mu`, `sigma`, `Q` | three separate predictors |

`target_value` is required and may be any value to inflate, not only zero.
`var_form` is `quadratic` (NB2) or `linear` (NB1). `ValueInflated*` models use
`infl_link` for the probability of an inflated observation. For
`GeneralizedGamma`, the `mx_*_cols` arguments are design-matrix column indices;
with formulas, prefer `scg.generalized_gamma(..., sigma_formula=..., Q_formula=...)`.

### Three-predictor and inflated models

`GeneralizedGamma` has separate predictors for location `mu`, scale `sigma` and
shape `Q`:

```python
fit = scg.generalized_gamma(
    "time ~ treatment + age",
    data=patients,
    sigma_formula="~ treatment",
    Q_formula="~ 1",
    link_mu="log",
    cens="status",
).fit()
```

`ValueInflated*` models take `target_value`; the part after `|` models the
probability of an inflated observation:

```python
vi_pois = scg.value_inflated_poisson("count ~ x | z", df, target_value=0).fit()
vi_nb = scg.value_inflated_negative_binomial(
    "count ~ x | z", df, target_value=10, var_form="quadratic"
).fit()
```

`CustomFamily` supplies all six IRLS callbacks. The vectorized functions
receive and return arrays; `dev_resids_fn` returns the total deviance:

```python
import numpy as np

def variance(mu):
    return np.asarray(mu, dtype=float)

def dmu_deta(eta):
    return np.exp(eta)

def linkinv(eta):
    return np.exp(eta)

def dev_resids(y, mu, w):
    y, mu, w = np.broadcast_arrays(y, mu, w)
    log_term = np.log(np.where(y > 0, y, 1.0) / mu)
    term = np.where(y > 0, y * log_term - (y - mu), mu - y)
    return float(np.sum(2 * w * term))

def valid_eta(eta):
    return bool(np.all(np.isfinite(eta)))

def valid_mu(mu):
    return bool(np.all(np.asarray(mu) > 0))

family = scg.CustomFamily(
    variance, dmu_deta, linkinv, dev_resids, valid_eta, valid_mu
)
fit = scg.glm("y ~ x", df, family=family).fit()
```

## Fitting options

The complete fitting contract is:

```python
fit = model.fit(
    start=None,
    cens=None,
    Z=None,
    left_limit=None,
    right_limit=None,
    entry=None,
)
```

`.fit()` returns the same `GLM` object; calling it again refits that object.

| Argument | Default | Applies to | Meaning |
|---|---|---|---|
| `start` | `None` | IRLS, NB two-step, quasi and `QuasiCustom` | Initial coefficient vector. MLE families raise `ValueError` if it is supplied. |
| `cens` | `None` | Weibull, LogNormal, LogLogistic, GeneralizedGamma | Event indicator: `1` = event, `0` = right-censored. `None` treats all observations as events. A string names a column when the model was built from a formula. Also accepted by `from_formula` / `glm(...)`. |
| `Z` | `None` | Zero-inflated, hurdle and other second-predictor families | Secondary design matrix. When omitted, the matrix built by a two-part formula is used; an explicit array overrides it. |
| `left_limit` | `None` | `Truncated*` | Lower truncation bound; `None` means no lower bound. |
| `right_limit` | `None` | `Truncated*` | Upper truncation bound; `None` means no upper bound. |
| `entry` | `None` | GeneralizedGamma | Delayed-entry / left-truncation times; `None` means zeros. A string names a data column for formula-built models. |

All numerical controls (`tol`, `maxit`, `solver`, `minimizer`, `ftol`, `gtol`,
`maxcor`, `verbose`, `compute_se`, `standardize`, `use_t`, ...) are constructor
arguments, not `.fit()` arguments. `weights` and `offset` are also supplied at
construction. `sigma_formula=`, `Q_formula=`, `link_mu=`, `link_sigma=` and
`link_Q=` belong to `generalized_gamma(...)` / `GLM.from_formula_gengamma(...)`.

### Two-part models

Use the R `Formula` convention `"mean_part | secondary_rhs"`: the text after
`|` is a bare right-hand side and must not contain `~`.

```python
fit = scg.glm(
    "affairs ~ age + rating | age + yearsmarried",
    data=affairs,
    family=scg.ZeroInflatedPoisson(),
).fit()
```

Alternatively, pass an explicit secondary design matrix with `Z=`.

### Censored survival models

```python
fit = scg.glm(
    "time ~ age + treatment",
    data=patients,
    family=scg.Weibull(),
    cens="status",
).fit()
```

`cens=` is the public argument. `d` is an internal variable and is not accepted
by `.fit()`.

### Truncated families

```python
fit = scg.glm(
    "count ~ x",
    data=df,
    family=scg.TruncatedPoisson(link="log"),
).fit(left_limit=1, right_limit=50)
```

For continuous truncation, `TruncatedGaussian` uses a dedicated 2×2 diagnostic
mosaic as its default `fit.plot()` / `kind="diagnostics"` view:

```python
panel = fit.plot()                              # default for TruncatedGaussian
panel = fit.plot(kind="truncated_diagnostics")  # explicit equivalent
```

It shows a normal Q-Q of quantile residuals with a 95% simulation envelope,
quantile residuals and their spread against `η`, the observed density against
the average per-row truncated predictive density with dashed truncation
limits, and observed responses against the fitted truncated mean with a
smoothed trend and 95% interval. The older `kind="truncated"` remains a single
observed-versus-fitted panel and can still be used for a quick look.

## Working with the result

`.fit()` returns the fitted `GLM` itself. `repr(fit)` / `print(fit)` are compact;
call `fit.summary()` for the full report. Result properties and methods raise
`RuntimeError` before fitting. If the model was built with `compute_se=False`,
standard errors, covariance matrices and inference procedures that need them
are unavailable.

### Summaries, tables and inference

| Method | Returns | Purpose |
|---|---|---|
| `summary(yname=None, xname=None, title=None, alpha=0.05, cov_type=None, cov_kwds=None, **kwargs)` | `Summary` | Full statsmodels-style report; use `print()` or render as HTML. `.tables` exposes individual tables. `cov_type` may be `HC0`–`HC3` or `cluster` with `cov_kwds={"groups": ...}`. |
| `coef_table(alpha=0.05, exponentiate=False, cov_type=None, cov_kwds=None)` | `DataFrame` | Coefficients, standard errors, test statistic, p-values and confidence bounds. |
| `summary_frame(...)`, `to_dataframe(...)` | `DataFrame` | Aliases of `coef_table()` with the same arguments. |
| `odds_ratios(alpha=0.05, cov_type=None, cov_kwds=None)`, `irr(...)` | `DataFrame` | Exponentiated coefficient tables; conventionally used for logit and log-link models respectively. |
| `conf_int(alpha=0.05, cols=None, cov_type=None, cov_kwds=None)` | `DataFrame` | Wald intervals for all estimated parameters, optionally selecting columns. |
| `wald_test(r_matrix, q_matrix=None)` | `dict` | Test `R * all_params = q`; supports beta-only matrices with `len(beta)` columns. Returns statistic, df, p-value and distribution. |
| `anova(other, test="auto")` | `DataFrame` | Compare nested fits on the same response; `test` is `auto`, `Chisq`, `F` or `LRT`. |
| `cov_params()` | array | Parameter covariance matrix (`vcov`). |
| `get_robustcov_results(cov_type="HC0", **cov_kwds)` | `dict` | Robust covariance with keys `cov_type`, `cov_params`, `bse`, `params` and `param_names`. It is **not** a fitted result object; pass `cov_type=` to `summary()` / `coef_table()` for a table. |
| `pseudo_rsquared(kind="mcfadden")` | `float \| None` | McFadden pseudo R²; `kind` is `mcfadden` or `mcfadden_adj`. |

### Prediction and categorical responses

| Method | Returns | Purpose |
|---|---|---|
| `predict(exog=None, transform=True, linear=False, which="mean", exog_z=None, offset=None)` | array / `Series` | Predict on new rows or, with no arguments, on the fitted data. `linear=True` returns `eta`; otherwise `which` selects `mean` (`E[y]`), `location` (`g⁻¹(Xβ)`), `variance` or `secondary`. `exog_z` supplies the secondary matrix when `Z` was an array. For `Multinomial` / `Ordinal`, new data returns the probability matrix. |
| `get_prediction(exog=None, transform=True, alpha=0.05, exog_z=None, offset=None)` | `GLMPredictionResults` | Mean prediction with delta-method standard errors and confidence intervals. Not implemented for `Multinomial`, `Ordinal` or `GeneralizedGamma`. |
| `predict_proba(exog=None, transform=True)` | array / `DataFrame` | Category probabilities for `Multinomial` / `Ordinal`. |
| `predict_category(exog=None, transform=True)` | array / `Series` | Argmax category indices for `Multinomial` / `Ordinal`. |

`GLMPredictionResults` has `predicted_mean`, `se_mean`, `conf_int_mean`,
`se_obs`, `conf_int_obs`, `index`, `alpha`, and `summary_frame()`, which returns
mean, mean standard error and mean confidence bounds as a `DataFrame`:

```python
pred = fit.get_prediction(exog=new_x, alpha=0.05)
pred_frame = pred.summary_frame()
```

### Residuals, influence and diagnostics

| Method | Returns | Purpose |
|---|---|---|
| `cooks_distance()` | array / `None` | Case influence (IRLS/Pearson form or likelihood-distance form for supported MLE families). |
| `get_influence()` | `GLMInfluence` | Diagnostic container with `hat_matrix_diag`, `cooks_distance`, `resid_sspearson`, `resid_sdeviance`, `dffits`, `dffits_internal`, `resid_rstudent`, `dfbeta`, `dfbetas` and `summary_frame()`. |
| `resid_quantile(seed=None)` | array / `None` | Randomized Dunn–Smyth quantile residuals where supported. |

Extract influence diagnostics from the container; `dfbeta`/`dfbetas` are arrays
and are not included in `summary_frame()`:

```python
infl = fit.get_influence()
infl_frame = infl.summary_frame()  # hat_diag, cooks_d, residuals, dffits
dfbetas = infl.dfbetas
```
| `initialize_parameters()` | tuple | Starting `(beta, mu, eta)` values used by the solver; also callable before fitting. |

The fitted object exposes the following public properties. Values marked
"sometimes" are `None` for families or engines that do not support them.

| Category | Properties |
|---|---|
| Model state | `family`, `formula`, `data`, `exog`, `endog`, `exog_names`, `endog_names`, `nobs`, `df_resid`, `rank`, `iterations`, `converged`, `use_t` |
| Coefficients | `beta`, `params`, `all_params`, `param_names`, `all_param_names`, `n_params`, `se`, `bse`, `vcov`, `dispersion`, `gamma`, `extra_param`, `extra_param2`, `tvalues`, `pvalues` |
| Fit criteria | `log_likelihood`, `llf`, `llnull`, `loglik_obs`, `deviance`, `null_deviance`, `pearson_chi2`, `aic`, `bic`, `eql_loglik`, `aic_eql`, `qaic`, `scale` |
| Fitted values | `mu`, `fittedvalues`, `linear_predictors` |
| Residuals | `resid_response`, `resid_working`, `resid_pearson`, `resid_deviance`, `resid_anscombe`, `resid_sspearson`, `resid_sdeviance`, `resid_rstudent`, `resid_martingale`, `resid_deviance_martingale`, `resid_cox_snell` (sometimes), `has_quantile_residuals` |
| Influence | `hat_matrix_diag` (sometimes), `dffits`, `dffits_internal`, `dfbeta`, `dfbeta_all`, `dfbetas` (sometimes) |

### Auxiliary parameter storage

The auxiliary attributes are family-specific; `all_param_names` gives the exact
labels used by `summary()`, `coef_table()` and `conf_int()`.

| Attribute | Families | Stored values |
|---|---|---|
| `gamma` | Zero-inflated, hurdle and `ValueInflated*` | Coefficients of the inflation/hurdle submodel on its link scale. |
| `gamma` | `Ordinal` | Threshold coefficients on the natural response scale. |
| `gamma` | `StudentT`, `SkewNormal`, `SkewT` | Auxiliary distribution parameters: scale plus skewness and/or degrees of freedom. |
| `gamma` | `GeneralizedGamma` | `[mu_intercept, sigma_intercept, Q_intercept]`; covariate slopes are in `beta`. |
| `extra_param` | `NegativeBinomial(theta=None)` | Estimated `theta`. With fixed `theta` this is `None` and the value is in `family.params.theta`. |
| `extra_param` | Other MLE families | Natural-scale dispersion or shape (`sigma`, `alpha`, Gamma `shape`, Beta `phi`, Simplex `sigma`, ...). For `GeneralizedGamma` it duplicates `gamma[2]` (`Q`). |
| `extra_param2` | `ValueInflatedNegativeBinomial` only | `log(alpha)` on the estimation scale; `extra_param` is `alpha` on the natural scale. |

Module-level wrappers such as `statcraft.glm.glm.aic(model)` and
`statcraft.glm.glm.predict(model, ...)` also exist, but they are not exported
from `statcraft.glm`; prefer the fitted-object methods above.

### Plots

```python
from statcraft.plots import available_kinds, suggest_plots

fit.plot()                         # adaptive 2x2 diagnostics panel
fit.plot("rootogram")              # one named kind
print(suggest_plots(fit))          # recommended kinds for this family
print(available_kinds(fit))        # same filtered list as suggest_plots(fit)
print(available_kinds())           # every registered kind and alias
```

`plot(kind=None, ax=None, **kwargs)` returns a `PlotResult` with `.fig`, `.axes`,
`.ax`, `.show()`, `.save()`, `.savefig()` and `.close()`. Common options include
`seed`, `nsim`, `alpha`, `bins`, `n_bins` and `figsize`; exact choices depend on
the selected kind. Percentile/QQ, rootogram, ACF, calibration, influence and
family-specific panels are registered for GLM results. See
[Plotting and diagnostics](plotting.md) for the full registry contract.

## Gotchas

- `start=` is not supported for MLE-based families; their solver chooses its own
  starting values.
- `standardize=True` (the default) cannot be combined with a rank-deficient
  design matrix in the engines that detect aliasing. Remove duplicate columns or
  fit with `standardize=False`.
- `d` is not a `.fit()` keyword: supply `cens=`. `sigma_formula=` and
  `Q_formula=` are constructor/shortcut arguments.
- Two-part formulas put `|` after the mean-model right-hand side; the text
  after `|` must not contain `~`.
- `get_robustcov_results()` returns a `dict`; use
  `fit.summary(cov_type="HC3")` or `fit.coef_table(cov_type="HC3")` for a table.
- Truncation bounds are passed to `.fit()`, not to the family constructor.
- `predict()` for `GeneralizedGamma` only supports no-argument fitted-data
  means; `get_prediction()` is unavailable for that family.
- Plotting needs `matplotlib`: `python -m pip install "statcraft[viz]"`.

## See also

- [Model functions](model-functions.md) — shared callable conventions.
- [Data loading](data-loading.md) — locate bundled example data.
- [Plotting and diagnostics](plotting.md) — plot kinds and `PlotResult`.
- [Troubleshooting](troubleshooting.md) — installation and fitting errors.
