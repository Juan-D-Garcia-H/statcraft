# Linear and nonlinear mixed effects (NLME)

`statcraft.nlme` exposes R `nlme`-style constructors. All constructors return an
**unfitted** model; `.fit()` takes no user arguments, caches the result and
returns a fitted result object. Repeated calls to the model's `.fit()` return
the same cached object without refitting. Result-only attributes accessed before
fitting raise an `AttributeError` that points to `.fit()`.

| Constructor | Model | Result class |
|---|---|---|
| `lme` / `LME` | Linear mixed effects | `LmeFit` |
| `gls` / `GLS` | Generalized least squares without random effects | `LmeFit` |
| `nlme` / `NLME` | Nonlinear mixed effects | `NlmeFit` |
| `gnls` / `GNLS` | Generalized nonlinear least squares | `NlmeFit` |
| `lmer` / `LMER` | Crossed random effects (lme4-style) | `LmerFit` |
| `glmer` / `GLMER` | Crossed random effects with a non-Gaussian response | `GlmerFit` |

```python
import statcraft.nlme as nl
from statcraft.datasets import data_loader
```

`nl.lme is nl.LME`, and likewise for the other five pairs: the lowercase names
are aliases of the same class objects. Top-level `statcraft` exports the
uppercase classes plus lowercase `lmer` and `glmer`; lowercase `lme`, `gls`,
`nlme` and `gnls` live in the namespace. `statcraft.nlme` is a subpackage, not a
callable estimator.

## Constructors

### `LME` / `lme`

```python
LME(
    fixed,
    data,
    random=None,
    groups=None,
    correlation=None,
    weights=None,
    method="REML",
    na_action="omit",
    control=None,
    pd_struct=None,
    cor_prefix=False,
)
```

`fixed` is a two-sided formula such as `"distance ~ age * Sex"`. `random` is a
one-sided grouping formula such as `"~ age | Subject"`; use `"~ 1 | Subject"`
for a random intercept only. `groups` selects the grouping column when it is not
encoded in `random`. `method` is `"REML"` (default) or `"ML"`.

### `GLS` / `gls`

```python
GLS(
    model,
    data,
    correlation=None,
    weights=None,
    method="REML",
    na_action="omit",
    control=None,
)
```

`model` is a two-sided formula. GLS has no random-effects term.

### `NLME` / `nlme`

```python
NLME(
    model,
    data,
    params=None,
    random=None,
    start=None,
    correlation=None,
    weights=None,
    method="ML",
    na_action="omit",
    na_pattern=None,
    control=None,
    pd_struct=None,
    cor_prefix=False,
    response=None,
    predictors=None,
    jac=None,
)
```

`model` may be a formula or a callable. `params` names the nonlinear parameters;
`start` supplies their initial values. With a two-argument callable such as
`f(x, p)`, `params` is optional when `start` is an ordered mapping. Pass
`params=` explicitly when parameters enter through covariate formulas.
`response` and `predictors` select the data columns when the callable does not
use the shared `f(x, p)` convention. `jac` optionally supplies an analytical
Jacobian. `method` defaults to `"ML"`.

### `GNLS` / `gnls`

```python
GNLS(
    model,
    data,
    params=None,
    start=None,
    correlation=None,
    weights=None,
    na_action="omit",
    na_pattern=None,
    control=None,
    method="ML",
    response=None,
    predictors=None,
    jac=None,
)
```

GNLS has the same callable contract as NLME without random effects.

### `LMER` / `lmer`

```python
LMER(
    formula,
    data,
    REML=True,
    weights=None,
    offset=None,
    na_action="omit",
    control=None,
    pd_struct=None,
)
```

`formula` is an lme4-style formula, for example
`"Reaction ~ Days + (Days | Subject)"`. `REML=True` requests restricted
maximum likelihood.

### `GLMER` / `glmer`

```python
GLMER(
    formula,
    data,
    family,
    link=None,
    nAGQ=1,
    weights=None,
    offset=None,
    theta=None,
    dispersion=None,
    na_action="omit",
    control=None,
    pd_struct=None,
)
```

`family` is required and must be a GLM family object or class such as
`Binomial("logit")`, `Poisson("log")`, `NegativeBinomial(theta=2.0)`,
`Gaussian()`, `Gamma("log")` or `InverseGaussian()`; family strings are
rejected. `link` overrides the family's default link. `nAGQ=1` uses the Laplace
approximation and `nAGQ > 1` uses adaptive Gauss–Hermite quadrature, which is
restricted to a single scalar random effect.

`theta` is the **negative-binomial dispersion**, not a covariance starting
value. For `NegativeBinomial`, `theta=None` means estimate it; passing `theta`
with another family is an error. `dispersion` may fix the scale for Gaussian,
Gamma and inverse-Gaussian families. Family **objects** are required; strings
are rejected. The GLM family classes are re-exported from `statcraft.nlme`, so
no separate `statcraft.glm` import is needed:

```python
from statcraft.nlme import Binomial, NegativeBinomial

binary_fit = nl.glmer("cured ~ dose + (1 | clinic)", data,
                      family=Binomial("logit")).fit()
count_fit = nl.glmer("count ~ dose + (1 | clinic)", data,
                     family=NegativeBinomial(theta=2.0)).fit()
```

### Common arguments

| Argument | Applies to | Meaning |
|---|---|---|
| `random` | `LME`, `NLME` | Random-effects formula, including grouping after `|`. |
| `groups` | `LME` | Explicit grouping column. |
| `correlation` | first four constructors | A `CorStruct` object; the R-style formula belongs in its `form=`, for example `corAR1(form="~ 1 | Subject")`. |
| `weights` | all six | A `VarStruct` for the first four constructors; a column name or numeric vector for `LMER`/`GLMER`. |
| `method` | first four | `"REML"` or `"ML"`; defaults differ by constructor. Only `LME` validates the value; otherwise anything but `"REML"` behaves as ML. |
| `REML` | `LMER` | `True` for REML, `False` for ML. |
| `na_action` | all six | `"omit"` (default) or `"fail"`. |
| `na_pattern` | `NLME`, `GNLS` | `None`/`False` for none, `True` to use non-missing response rows, or a callable/array yielding a boolean mask. |
| `control` | all six | Mapping of algorithm options; see below. |
| `pd_struct` | `LME`, `NLME`, `LMER`, `GLMER` | Positive-definite covariance structure override. |
| `cor_prefix` | `LME`, `NLME` | Allow a correlation grouping that is a strict prefix of the random-effects chain. |
| `offset` | `LMER`, `GLMER` | Model offset. |
| `family`, `link`, `nAGQ`, `theta`, `dispersion` | `GLMER` | Generalized-linear family and its controls. |

### Control options

`control` is a mapping. The accepted keys depend on the constructor family.

For `LME`, `GLS`, `NLME` and `GNLS`:

| Key | Meaning |
|---|---|
| `optimizer` | One of `internal`, `trust`, `lbfgsb` or `scipy`. |
| `max_iter` | Maximum optimizer iterations. |
| `gtol`, `ftol` | Gradient and function-change tolerances. |
| `fd_step` | Finite-difference step. |
| `max_step` | Maximum parameter step. |
| `sigma` | Fix residual sigma at this value; not a starting value. |
| `fixed_theta` | Mapping of covariance labels to fixed values. |
| `auto_start` | Derive NLME starting values automatically; only used by `NLME`. |
| `ident` | Accepted for compatibility, but currently inert. |

Keys that are already attributes of the engine specification, such as `reml`,
`diagnose`, `em_iter`, `var_max_ratio`, `analytic_grad`, `tol`, `max_outer`,
`max_pnls`, `theta_bound` and `sigma_fixed`, are also accepted. Unknown keys
raise `ValueError`.

`LMER` accepts only `optimizer`, `max_iter`, `gtol`, `ftol`, `fd_step` and
`theta_bound`. `GLMER` accepts those plus `init_stage`, `pirls_tol`,
`pirls_step_tol` and `max_pirls`. For both crossed-effects constructors unknown
keys raise `ValueError`; `max_step`, `sigma`, `fixed_theta`, `auto_start` and
`ident` are not available.

### Correlation, variance and covariance structures

Pass structure factories to `correlation=`, `weights=` or `pd_struct=` as
appropriate. The correlation factories are `corAR1(form=None)`,
`corCAR1(form=None)`, `corCompSymm(form=None)`, `corSymm(form=None)`,
`corARMA(form=None, p=0, q=0)`, and the spatial families
`corExp`/`corGaus`/`corLin`/`corRatio`/`corSpher`
`(form=None, nugget=False, metric="euclidean")`. `corCustom(matrix, n_params,
form=None, start=None, name="custom", deriv=None)` accepts callbacks returning
a correlation matrix and optional derivatives.

Variance factories are `varIdent(form="~1|1")`, `varFixed(form)`,
`varPower(form="~fitted(.)")`, `varExp(form="~fitted(.)")`,
`varConstPower(form="~fitted(.)")`, `varConstProp(form="~fitted(.)")`,
`varComb(*parts)` and `varCustom(scales, n_params, form=None, start=None,
name="custom", dscales=None)`. Positive-definite factories are `pdLogChol()`,
`pdSymm()`, `pdChol()`, `pdNatural()`, `pdCompSymm()`, `pdDiag()`,
`pdIdent()`, `pdBlocked(sizes, kinds)` and `pdCustom(nparams, factor,
start=None, from_D=None, name="custom", dfactor=None)`.

```python
model = nl.lme(
    "distance ~ age",
    data,
    random="~ 1 | Subject",
    correlation=nl.corAR1(form="~ 1 | Subject"),
)
```

`pdBlocked(sizes, kinds)` splits the random-effects covariance into independent
blocks, and `varComb(*parts)` multiplies relative standard deviations from
several variance functions:

```python
pd = nl.pdBlocked([2, 2], [nl.pdDiag(), nl.pdCompSymm()])
weights = nl.varComb(
    nl.varIdent(form="~1|Sex"),
    nl.varPower(form="~fitted(.)"),
)

fit = nl.lme(
    "distance ~ age",
    orth,
    random="~ age + Sex + age:Sex | Subject",  # four random-effect columns
    pd_struct=pd,
    weights=weights,
)
```

## A first model

```python
orth = data_loader("Orthodont.csv").load()

model = nl.lme(
    "distance ~ age * Sex",
    orth,
    random="~ age | Subject",   # random intercept and slope per child
)
fit = model.fit()
print(fit.summary())
```

`method="REML"` is the default for `lme`, `gls` and `lmer`; use `method="ML"`
(or `REML=False` for `lmer`) when comparing likelihoods across different fixed
effects. REML likelihoods with different fixed effects are not comparable.

## Nonlinear model functions

The shared [`f(x, p)` convention](model-functions.md) works in `NLME` and
`GNLS`:

```python
def model(x, p):
    Time, Dose = x
    ka, CL, V = p
    k = CL / V
    return Dose * ka / (V * (ka - k)) * (np.exp(-k * Time) - np.exp(-ka * Time))
```

An older named-argument form is also accepted: parameters and predictors are
resolved against data-column names. A two-argument callable may use any argument
names when more than one parameter is declared; with one parameter, use `p`,
`params`, `parameters` or `theta` as the second name. `params` is optional when
the key order of `start=` supplies the names; explicit `params=` is required for
covariate formulas. Formula models may use the selfStart functions `SSlogis`,
`SSfpl`, `SSasymp`, `SSasympOff`, `SSasympOrig`, `SSmicmen`, `SSgompertz` and
`SSweibull`; other expressions require `start=` or derivable parameter
formulas. `response=`, `predictors=` and `jac=` apply only to callable models.

`params=` may be a two-sided formula (one formula for all parameters) or a
mapping of parameter-specific formulas. The key order defines the order in which
the parameters are passed to the model function:

```python
def logistic_soy(Time, Asym, xmid, scal):
    return Asym / (1.0 + np.exp((xmid - Time) / scal))

fit = nl.gnls(
    logistic_soy,
    soybean,
    response="weight",
    params={"Asym": "Variety", "xmid": "1", "scal": "1"},
    start={"Asym": 18.0, "xmid": 54.0, "scal": 8.0},
).fit()
```

Here `Asym` has a covariate formula while `xmid` and `scal` are intercepts.

## Results

`LmeFit`, `NlmeFit`, `LmerFit` and `GlmerFit` all provide a compact `repr` and
`plot()`. Full tables are printed by `summary()`.

| Result class | `fitted(...)` | `tTable(...)` | `conf_int()` | `wald_test()` | `anova()` | `simulate()` |
|---|---|---|---|---|---|---|
| `LmeFit` | `level=` | `ddf=`, `df=` | yes | yes | yes | no |
| `NlmeFit` | `level=` | `ddf=`, `df=` | yes | yes | yes | no |
| `LmerFit` | `re_form=` | `df=`, `ddf=` | no | no | yes (REML refit) | no |
| `GlmerFit` | `type=`, `re_form=` | no arguments | no | no | no | no |

### `LmeFit` and `NlmeFit`

```python
fit.fixef          # fixed-effect estimates
fit.se             # standard errors
fit.vcov           # covariance of fixed effects
fit.sigma          # residual standard deviation
fit.loglik         # log-likelihood
fit.nobs           # number of observations
fit.theta          # covariance parameters
fit.varcorr        # variance-correlation structure
fit.ranef          # random effects (property)
fit.kind           # model kind
fit.method         # "REML" or "ML"
fit.warning        # fitting warning, if any
fit.converged      # convergence flag
fit.raw            # raw engine result
```

| Method | Purpose |
|---|---|
| `fit()` | Return this fitted object. |
| `plot(kind=None, ax=None, **kwargs)` | Render a plot by kind. |
| `fitted(level=None)` | Fitted values; `level=0` is population level and higher levels include random effects. |
| `residuals(type="response", level=None)` | `response`/`raw`, `pearson` and `normalized` residuals. |
| `predict(newdata=None, level=None)` | Predictions for the original or new data; unseen groups give `NaN`. |
| `tTable(ddf="containment", *, df=None)` | Coefficient table; supports `containment` and Satterthwaite degrees of freedom. |
| `conf_int(alpha=0.05)` | Wald confidence intervals. |
| `cor_fixef()` | Correlation matrix of fixed effects. |
| `wald_test(r_matrix, q_matrix=None)` | General linear hypothesis test. |
| `anova(other, ddf="chi2")` | Compare two fits; `ddf` may be `chi2`, `lrt`, `satterthwaite` or `residual`. |
| `aic()`, `bic()` | Information criteria. |
| `summary(alpha=0.05)` | Printable summary. |

For likelihood-ratio comparisons of different fixed effects, fit both models
with ML. `tTable(ddf="satterthwaite")` refits to REML where possible.

`residuals(type="pearson")` and `residuals(type="normalized")` divide by the
observation SD `σ̂ · g(μ̂ᵢ, δ)`, so a variance function (`varPower`, `varExp`, ...)
produces unit-variance residuals on the N(0, 1) Q-Q scale. For `LME`, the
covariance-normalized form additionally whitens random effects when no residual
correlation is present. A `GLS` result has no BLUPs, so `fit.ranef` raises
`AttributeError`; a `GNLS` result has `fit.ranef is None`.
If the engine omits the nonlinear fixed-effect covariance, `NlmeFit.vcov` is an
all-`NaN` frame of the correct shape.

### `LmerFit`

Attributes include `fixef`, `se`, `vcov`, `sigma`, `loglik`, `deviance`,
`nobs`, `theta`, `npar`, `singular`, `converged`, `warning`, `varcorr`,
`ranef`, `aliased`, `formula`, `method`, `raw` and `terms_info`. Fixed effects,
standard errors and the covariance matrix contain only estimable columns;
`aliased` records dropped coefficients.

| Method | Purpose |
|---|---|
| `fit()`, `plot(...)`, `summary(alpha=0.05)` | Common fitted-object API. |
| `fitted(re_form="all")` | Fitted values; `"all"`/`True` includes random effects. |
| `residuals(re_form="all")` | Residuals at the requested random-effect level. |
| `predict(newdata=None, re_form="all", allow_new_levels=False)` | Predictions; a sequence of group names selects random effects, and new levels need `allow_new_levels=True`. |
| `tTable(df="satterthwaite", *, ddf=None)` | Coefficient table; also accepts `residual`, `containment` and `none`. |
| `satterthwaite_contrast_df(L_list)` | Degrees of freedom for custom contrasts. |
| `satterthwaite_df()` | Per-coefficient Satterthwaite degrees of freedom. |
| `anova(other, ddf="satterthwaite")` | Refits both models to REML and compares them; also accepts `residual`. |
| `aic()`, `bic()` | Information criteria. |

`LmerFit` has no `conf_int()`, `cor_fixef()` or `wald_test()`.

### `GlmerFit`

Attributes include `fixef`, `se`, `vcov`, `sigma`, `loglik`, `deviance`,
`loglik_lme4`, `theta_nb`, `dispersion`, `dispersion_fixed`, `theta`, `nobs`,
`npar`, `singular`, `converged`, `warning`, `vcov_from_hessian`, `varcorr`,
`ranef`, `aliased`, `formula`, `family`, `link`, `nAGQ`, `raw` and
`terms_info`.

| Method | Purpose |
|---|---|
| `fit()`, `plot(...)`, `summary(alpha=0.05)` | Common fitted-object API. |
| `fitted(type="response", re_form="all")` | Fitted values; `type` may be `response` or `link`. |
| `residuals(type="deviance")` | Residuals; also accepts `pearson`, `response` and `working`. |
| `resid_quantile(seed=42, n_sim=500)` | Mid-p simulated quantile residuals on the normal-score scale for binomial (Bernoulli or grouped `k/n`), Poisson and negative-binomial fits. |
| `blup_conditional_se()` | Posterior SDs of the BLUPs from `(D⁻¹ + ZᵀWZ)⁻¹`, returned as one table per random-effect term; drives the 95% caterpillar intervals. |
| `predict(newdata=None, type="link", re_form="all", allow_new_levels=False, offset=None)` | Predictions with an explicit scale; pass `offset=` when the model was fitted with one. |
| `tTable()` | Coefficient table with normal `z` values; takes no arguments. |
| `aic()`, `bic()` | Information criteria. |

`tTable()` for `GlmerFit` does not accept `ddf=` or `df=`. `GlmerFit` has no
`anova()`, `conf_int()`, `cor_fixef()` or `wald_test()`. Note the different
default scales: `fitted(type="response")` is on the mean scale, while
`predict(...)` defaults to `type="link"`. Pass `type="response"` to `predict()`
when comparing predictions with `fitted()`; both accept `re_form=`.

Simulated quantile residuals reuse the fitted conditional means instead of
refitting each replicate. They are a fast diagnostic for discrete responses,
not a parametric bootstrap test.

## Helpers

| Helper | Purpose |
|---|---|
| `lm_list(formula, data, na_action="omit")` | Fit an OLS model per grouping level; the formula must contain `| groups`. |
| `nls_list(model, data, start, params=None, na_action="omit", tol=1e-8, max_iter=200)` | Fit a nonlinear model per grouping level; the formula must contain `| groups`. |
| `corr_fixef(model, values=False)` | Correlation matrix of fixed effects as a `DataFrame`, or an array with `values=True`. |
| `fix_df(model_or_formula, data=None, random=None, groups=None, na_action="omit")` | Containment degrees of freedom for fixed effects. |
| `ListFit` | Grouped-fit result with `coef`, `se`, `sigma`, `df`, `pooled_sigma`, `pooled_df`, optional `converged`, `summary(alpha=0.05)` and `plot()` methods. |

```python
by_subject = nl.lm_list("distance ~ age | Subject", orth)
print(by_subject.summary())
```

## Plotting

`fit.plot(kind=...)` uses the shared [plot registry](plotting.md). Common kinds
for mixed models are `diagnostics`, `residuals`, `qq_residuals`, `gof`,
`population_vs_conditional`, `individuals`, `caterpillar`,
`ranef_distribution`, `qq_random`, `variance_components` and `variogram`.

The default `fit.plot()` mosaic checks both hierarchy levels. For `LME` it
shows population vs. conditional predictions, normalised residuals against the
conditional fit with a LOESS trend and 95% band, a Q-Q with a 95% simulation
envelope, and a caterpillar of the BLUPs with conditional 95% intervals and the
random-effect SD in the title. For `NLME` the residual panel uses the primary
structural variable (time/dose) instead of the fitted value, because opposite
phases of a nonlinear curve can share the same `ŷ`; the caterpillar selects the
term with the largest relative variability so a collapsed random effect is not
plotted as noise. The same panels are available individually as
`normalized_residuals_vs_fitted`, `normalized_residuals_vs_time` and
`qq_plot_with_envelope`.

`GLS` and `GNLS` have no random effects, so their default mosaics never show a
population/conditional pair. `GLS` shows observed vs fitted with an identity
line and a LOESS calibration curve, a normalised Q-Q with 95% envelope,
normalised residuals together with `√|r̂|` against `ŷ` on independent dynamic Y
scales, and an ACF accumulated within the correlation grouping factor. The ACF
never extends beyond `max(n_i) - 1` and draws the `±1.96/√N` white-noise band.
`GNLS` uses the same residual and ACF panels, but plots the residual/spread
panel against the primary structural variable (time/dose) when available;
otherwise it falls back to `ŷ`. Its observed-vs-fitted panel also carries the
LOESS calibration curve. Normalised residuals use the full observation scale
`σ̂ · g(μ̂ᵢ, δ)`, and the ACF is computed within plots/subjects inferred from the
data when no correlation structure supplies groups. These checks are also available individually as
`observed_vs_fitted`, `qq_plot_with_envelope`,
`normalized_residuals_and_spread` and `grouped_acf`, or through
`gls_diagnostics` / `gnls_diagnostics`.

Discrete `GLMER` fits (Bernoulli, grouped binomial `k/n`, Poisson and negative
binomial) use a second mosaic: population vs conditional predictions, a DHARMa-
style Q-Q of simulated quantile residuals with a 95% KS envelope, mean quantile
residuals in bins of `rank(ŷ)` on `[0, 1]`, and a caterpillar of the BLUPs with
95% conditional intervals. `n_sim=` controls the simulation size. Continuous
`GLMER` families keep the population/conditional, normal Q-Q, binned
standardised residuals and BLUP caterpillar layout. The individual kinds are
`randomized_quantile_qq` and `binned_quantile_residuals_by_rank`, plus
`glmer_diagnostics` for the whole mosaic. Binary `GLMER` fits also suggest
`roc`, `pr` and `calibration`. NLME result
classes do not expose `simulate()`, so `vpc` is not currently recommended for
them. Generated `plot_<kind>()` methods and module-level
`statcraft.plots.plot_<kind>()` functions are also available.

## Gotchas

- All constructors are unfitted; the object returned by `.fit()` is different
  from the model for the LME/NLME family.
- `NLME`/`GNLS` default to `method="ML"`, while `LME`/`GLS` default to REML.
- `LmeFit`/`NlmeFit` use `fitted(level=...)`, `LmerFit` uses `fitted(re_form=...)`
  and `GlmerFit` uses `fitted(type=..., re_form=...)`. The signatures are not
  interchangeable. `GlmerFit.fitted()` defaults to the response scale, whereas
  `GlmerFit.predict()` defaults to the link scale; pass `type="response"` to
  `predict()` to compare them.
- `tTable` signatures differ: `LmeFit`/`NlmeFit` use
  `tTable(ddf="containment", *, df=None)`, `LmerFit` uses
  `tTable(df="satterthwaite", *, ddf=None)`, and `GlmerFit` uses `tTable()`.
- ML likelihoods are required for comparing different fixed effects;
  Satterthwaite tables refit to REML where possible.
- `cor_prefix=True` is available for `LME` and `NLME` when the correlation
  grouping is a strict prefix of the random-effects chain.
- Only `LME` validates `method`; for `GLS`, `NLME` and `GNLS`, any value other
  than `"REML"` behaves as ML.
- `control["sigma"]` fixes residual sigma rather than supplying a starting
  value. `ident` is accepted but inert, and `auto_start` is used only by `NLME`.
- GLMER `theta` is the negative-binomial dispersion, not a covariance start;
  GLMER families must be family objects, not strings. The GLM family classes
  are available directly from `statcraft.nlme` and `statcraft.saem`.
- All NA handling treats both `NaN` and infinities as missing.

## See also

- [Model functions](model-functions.md) — shared callable conventions.
- [NLS](nls.md) — single-curve nonlinear least squares.
- [SAEM](saem.md) — simulation-based nonlinear mixed effects.
- [Plotting](plotting.md), [Data loading](data-loading.md) and
  [Troubleshooting](troubleshooting.md).
