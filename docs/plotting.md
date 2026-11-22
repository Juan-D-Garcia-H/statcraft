# Plotting and diagnostics

Plotting requires matplotlib:

```bash
python -m pip install "statcraft[viz]"
```

## Three equivalent ways to call a plot

Every fitted result exposes `fit.plot()`, and `statcraft.plots.plot()` is the
module-level equivalent. Both dispatch through the same registry of plot kinds.

```python
fit.plot()                                      # adaptive 'diagnostics' panel
fit.plot(kind="qq")
fit.plot(kind="residuals_vs_fitted", save="residuals.png", dpi=150)

from statcraft.plots import plot
plot(fit, kind="forest", show=True)
```

There is also one preferred generated facade name per kind. The tables below
list it in the **Call** column; use it either as a result method or as a module
function:

```python
fit.qqplot()                              # same as fit.plot(kind="qq")
fit.plot_residuals_vs_fitted(ax=ax)       # same as fit.plot(kind="residuals_vs_fitted")
fit.forest_plot()                         # same as fit.plot(kind="forest")

from statcraft.plots import qqplot, plot_residuals_vs_fitted, forest_plot
qqplot(fit)
plot_residuals_vs_fitted(fit, ax=ax)
forest_plot(fit)
```

The special facade `plot_grafica()` leaves `kind` unspecified, so it renders the
adaptive panel unless you pass one:

```python
fit.plot_grafica(kind="residuals")
```

Facade names follow one rule: `plot_<kind>` normally, but a kind that already
ends in `plot` keeps that suffix (`forest_plot`, `calibration_plot`,
`cooks_distance_plot`, ...) and the joined density names are `qqplot` and
`ppplot`. The old `plot_qqplot`, `plot_qq_plot`, `plot_pp_plot`,
`plot_forest_plot`, ... spellings no longer exist.

`plot()` and every facade forward `show=`, `save=`, `dpi=`, `close=` and
`block=` to the renderer. They return a `PlotResult` with `fig`, `axes`, `ax`,
`show()`, `save()` / `savefig()` and `close()`. Extra keyword arguments go to
the selected plot (`seed`, `nsim`, `alpha`, `response`, `exponentiate`, ...).
Hyphens and spaces in `kind` are normalized to underscores, so `scale-location`
is accepted as `scale_location`.

Runtime discovery:

```python
from statcraft.plots import available_kinds, suggest_plots, KIND_ALIASES

available_kinds()                # every accepted canonical name and alias
available_kinds(fit)             # same as suggest_plots(fit)
suggest_plots(fit)               # kinds that make sense for this result
sorted(KIND_ALIASES)             # authoritative alias list
```

`suggest_plots(fit)` returns **kinds**, not facade names. A kind marked as
unsupported below can still be called directly if the result happens to expose
the required data; it is simply not advertised by the adaptive panel.

### Model codes

| Code | Meaning |
|---|---|
| `all` | Any fitted result with the required data |
| `Resid` | Fits with usable residuals |
| `GLM` | Any `GLM` fit |
| `NLS` | `nls` / `NLS` fits |
| `Mix` | `LmeFit`, `NlmeFit`, `LmerFit`, `GlmerFit` |
| `SAEM` | `SaemFit` results |
| `MCGLM` | `MCGLMResults` |
| `List` | `ListFit` (`lm_list` / `nls_list`) |
| `hat` | GLM/NLS exposing the hat diagonal (`influence`, `cooks`, ...) |
| `RE` | Fit with random effects |
| `sim` | Fit with a usable `simulate()` method |
| `binary`, `count`, `quasi`, ... | Family or data-dependent GLM variants |

## Residual diagnostics and model checks

All rows in this table are also reachable with `fit.plot(kind="<Kind>")`.

| Kind | Call | Models | Description |
|---|---|---|---|
| `diagnostics` | `plot_diagnostics()` | all | Adaptive 2×2 panel chosen for the result family: continuous and Tweedie GLMs get quantile scale-location, Q-Q with 95% envelope, quantile residuals vs `η` and residuals vs leverage; count GLMs get DHARMa Q-Q, rootogram, quantile residuals vs `μ` and leverage; Beta/simplex GLMs get observed-vs-fitted, an annotated quantile-residual Q-Q, quantile residuals vs `η` and either influence or modelled dispersion vs `η`; censored survival GLMs get censoring-aware observed-vs-fitted, Cox-Snell with a Greenwood band, deviance-martingale residuals vs `η` and influence (GeneralizedGamma keeps its submodel panel). GLS/GNLS/GLMER/SAEM delegate to their dedicated mosaics described below. |
| `residuals` | `plot_residuals()` | Resid | Residuals vs fitted plus a normal Q-Q plot. |
| `qq` | `qqplot()` | Resid | Normal Q-Q of the model's preferred residuals. |
| `qq_residuals` | `plot_qq_residuals()` | Resid, Mix | Explicit residual Q-Q spelling used by mixed fits. |
| `pp` | `ppplot()` | Resid | Probability-probability plot of residuals against N(0, 1). |
| `half_normal` | `half_normal_plot()` | Resid | Half-normal absolute residuals with a simulated envelope. |
| `worm` | `worm_plot()` | Resid | Detrended Q-Q showing local deviations from the identity line. |
| `hist` | `plot_hist()` | Resid | Residual histogram with a matched normal density. |
| `residuals_vs_fitted` | `plot_residuals_vs_fitted()` | Resid | Residuals against the linear predictor or fitted mean. |
| `scale_location` | `plot_scale_location()` | Resid | `√|r|` against the fitted mean. |
| `observed_vs_fitted` | `plot_observed_vs_fitted()` | Resid | Observed response against fitted mean with the identity line. Axes use their own data ranges and the line is drawn only on the observed/fitted overlap; `symmetric_log=True` applies matching symlog axes for skewed nonnegative responses; `loess=True` adds a local-linear calibration curve (used by the GLS mosaic); `include_zero=False` replaces the zero baseline with `0.95 × min` to `1.03 × max` per axis, which the GLS mosaic uses when the response lives far from zero. |
| `binned` | `plot_binned()` | Resid | Binned residual means (Gelman & Hill). |
| `grouped_residuals` | `plot_grouped_residuals()` | Resid + group | Residual boxplots by group with observations overlaid; explicit-only. |
| `binned_standardized_residuals` | `plot_binned_standardized_residuals()` | Mix, SAEM | Binned unit-scale residuals against fitted, time or index. |
| `qq_plot_with_envelope` | `plot_qq_plot_with_envelope()` | LME, NLME, GLS, GNLS, SAEM | Normal Q-Q of the preferred residuals with a 95% simulation envelope. SAEM labels NPDE draws as `NPDE`. Aliases: `qq_with_envelope`, `qq_envelope`, `normal_qq_envelope`. |
| `normalized_residuals_vs_fitted` | `plot_normalized_residuals_vs_fitted()` | LME | Normalised residuals against conditional fitted values, with a LOESS trend and 95% band. Aliases: `residuals_vs_conditional`, `lme_residual_trend`. |
| `normalized_residuals_vs_time` | `plot_normalized_residuals_vs_time()` | NLME | Normalised residuals against the structural variable (time/dose), with a LOESS trend and 95% band. Aliases: `nlme_residuals_vs_time`, `nlme_residual_trend`. |
| `normalized_residuals_and_spread` | `plot_normalized_residuals_and_spread()` | GLS, GNLS | Normalised residuals `(y - ŷ) / (σ̂ · g(ŷ, δ))` (left axis) and `√|r̂|` (right axis) against `ŷ`, or against the primary structural variable with `use_time=True`. Both Y limits come from the data, so a fan shape is never clipped to `[-2, 2]`; each series has its own LOESS trend and the spread has a 95% band. The shared legend sits in the upper-left corner. Aliases: `normalised_residuals_and_spread`, `residuals_and_spread`, `gls_residual_spread`, `gnls_residual_spread`. |
| `acf` | `plot_acf()` | Mix / GLS | Autocorrelation function of residuals in supplied order (mixed-model panel). |
| `grouped_acf` | `plot_grouped_acf()` | GLS, GNLS, SAEM | Autocorrelation of normalised residuals accumulated within grouping levels. GLS takes groups from the correlation structure; unstructured GNLS infers the repeated-measures factor from columns such as `Plot` or `Subject`; SAEM uses the `id=` column. Without groups it falls back to a single-series ACF. The lag is capped at `max(n_i) - 1` and the white-noise band is `±1.96/√N`. `max_lag=` lowers the cap. Aliases: `grouped_acf_plot`, `acf_by_group`, `gls_residual_acf`, `gnls_residual_acf`. |
| `gls_diagnostics` | `plot_gls_diagnostics()` | GLS | Four-panel GLS mosaic: observed vs fitted with identity and LOESS, normalised Q-Q with 95% envelope, normalised residuals and `√|r̂|` vs `ŷ` with dynamic axes, and within-group ACF. Aliases: `diagnose_gls`, `gls_panel`. |
| `gnls_diagnostics` | `plot_gnls_diagnostics()` | GNLS | Four-panel GNLS mosaic: observed vs fitted with identity and LOESS, Q-Q with 95% envelope, normalised residuals and spread against the primary structural variable (falling back to `ŷ`), and within-group ACF. Aliases: `diagnose_gnls`, `gnls_panel`. |
| `glmer_diagnostics` | `plot_glmer_diagnostics()` | GLMER | Four-panel GLMER mosaic. Discrete families use population vs conditional predictions, a simulated quantile Q-Q, binned quantile residuals vs `rank(ŷ)` and a BLUP caterpillar with 95% conditional intervals; continuous families keep the normal Q-Q and binned standardised residuals. `n_sim=` controls the quantile simulation. Aliases: `diagnose_glmer`, `glmer_panel`. |
| `saem_diagnostics` | `plot_saem_diagnostics()` | SAEM | Four-panel SAEM mosaic: population vs conditional predictions, Pearson residuals vs IPRED (Gaussian) or NPDE vs time (censored/non-Gaussian), a Q-Q with 95% simulation envelope, and a within-subject ACF. Censored rows are kept through NPDE instead of being dropped as NaN Pearson residuals. `n_sim=` controls the Q-Q envelope. Aliases: `diagnose_saem`, `saem_panel`, `saem_mosaic`. |
| `rootogram` | `plot_rootogram()` | GLM count | Hanging rootogram of observed vs expected counts. |
| `randomized_quantile_qq` | `dharma_plot()` | GLM count/continuous/Tweedie, discrete GLMER | Q-Q of randomised quantile residuals with a 95% KS envelope. GLMER accepts `n_sim=` for the conditional simulation. |
| `quantile_residuals_vs_fitted` | `plot_quantile_residuals_vs_fitted()` | GLM count/continuous/Tweedie | Quantile residuals against `μ`, or against `η` with `use_linear_predictor=True`. |
| `binned_quantile_residuals_by_rank` | `plot_binned_quantile_residuals_by_rank()` | discrete GLMER | Mean quantile residuals in equal bins of `rank(ŷ)` on `[0, 1]` with ±SE bars. Fitted values of Bernoulli, grouped-binomial and count responses pile up near zero, so ranking spreads the bins evenly. Aliases: `binned_quantile_residuals`, `quantile_residuals_by_rank`, `glmer_binned_quantile_residuals`. |
| `quantile_scale_location` | `plot_quantile_scale_location()` | GLM continuous/Tweedie | `√|quantile residual|` against `η` with the `√(2/π)` reference and a local-linear trend. |
| `standardized_pearson_vs_fitted` | `plot_standardized_pearson_vs_fitted()` | GLM quasi | Standardised Pearson residuals against `μ`. |
| `pearson_scale_location` | `plot_pearson_scale_location()` | GLM quasi | Pearson-based scale-location check. |
| `residuals_vs_predictions` | `plot_residuals_vs_predictions()` | SAEM | Residuals against conditional predictions, or against time. The reference lines are `±2` for Pearson residuals and `±1.96` for NPDE. |

## Fit and family-specific panels

| Kind | Call | Models | Description |
|---|---|---|---|
| `curve` | `plot_curve()` | NLS, 1 predictor | Fitted curve with delta-method confidence and prediction bands. |
| `gof` | `plot_gof()` | Mix, SAEM | Observed response vs population and individual predictions. |
| `calibration` | `calibration_plot()` | GLM binary | Reliability diagram and Brier score. |
| `cox_snell` | `plot_cox_snell()` | GLM censored/AFT/GeneralizedGamma | Nelson–Aalen cumulative hazard of Cox–Snell residuals against the unit-exponential line, with a 95% Greenwood band and a rug of censored residuals. |
| `deviance_martingale_vs_eta` | `plot_deviance_martingale_vs_eta()` | GLM censored/AFT/GeneralizedGamma | Deviance-martingale residuals against `η`, using events and censored rows with a local-linear trend. |
| `truncated` | `plot_truncated()` | GLM truncated | Observed vs fitted response with truncation limits. |
| `truncated_diagnostics` | `plot_truncated_diagnostics()` | GLM `TruncatedGaussian` | 2×2 truncation mosaic: Q-Q with a 95% simulation envelope, quantile residuals and spread vs `η`, observed density vs the mean predictive density with truncation lines, and observed vs fitted truncated mean with a smoothed 95% interval. This is the default `diagnostics` panel for the family. Aliases: `truncated_mosaic`, `truncated_panel`, `truncated_qc`. |
| `hurdle` | `plot_hurdle()` | GLM hurdle | Conditional positive component of a continuous hurdle model, with matching symlog axes, data-driven limits and the identity line restricted to the observed/predicted overlap. |
| `secondary` | `plot_secondary()` | GLM hurdle/beta | Distribution of the secondary submodel parameter (`π`, `φ` or `σ²`). |
| `secondary_vs_eta` | `plot_secondary_vs_eta()` | GLM Beta/simplex with modelled dispersion | Row-wise precision `φᵢ` (Beta) or dispersion `σ²ᵢ` (simplex) against `η`, with a local-linear trend. |
| `category_probs` | `plot_category_probs()` | GLM multinomial/ordinal | Boxplots of the probability assigned to the observed category. |
| `multiclass_binned_residuals` | `plot_multiclass_binned_residuals()` | GLM multinomial | Binned mean of the cell residuals `y_ik − p̂_ik` over all `N × K` cells, with ±2 SE bars. |
| `multiclass_calibration` | `plot_multiclass_calibration()` | GLM multinomial | One-vs-rest reliability curves, one per category, with Brier scores in the legend. |
| `ordinal` | `plot_ordinal()` | GLM ordinal | Category-probability curves along the linear predictor. |
| `lorenz` | `plot_lorenz()` | GLM Lorenz families | Lorenz curve and Gini index ordered by fitted value. |
| `gengamma` | `plot_gengamma()` | GLM GeneralizedGamma | Boxplots of row-wise location, scale and shape submodels. |

The censored-survival quartet is shared by `Weibull`, `LogNormal` and `LogLogistic`; `GeneralizedGamma` uses the same first three panels and keeps its submodels as the fourth. Its observed-vs-fitted panel scales each axis from its own data and draws the identity line only over the observed/fitted overlap, which keeps the cloud centred instead of pushing it to one side. Cox–Snell keeps one numeric scale on both axes but uses the full rectangular panel.

## Coefficients, covariance and grouped fits

| Kind | Call | Models | Description |
|---|---|---|---|
| `forest` | `forest_plot()` | fits with coefficients | Coefficient forest plot with Wald confidence intervals. |
| `cor_fixef` | `plot_cor_fixef()` | fits with coefficients | Heatmap of the fixed-effect correlation matrix. |
| `variance_components` | `plot_variance_components()` | Mix + `varcorr` | Standard deviations of the random effects. |
| `covariance` | `plot_covariance()` | MCGLM | Covariance parameters with confidence intervals. Correlations (ρ) use a fixed `[-1, 1]` axis above dispersion/power parameters (τ) on their natural scale; fits with only one group use a single axis. |
| `correlation` | `plot_correlation()` | MCGLM, `K > 1` | Observed vs model-implied between-response correlation. |
| `list_coefs` | `plot_list_coefs()` | List | Per-group coefficient panels for `lm_list` / `nls_list`. |

## Influence diagnostics

| Kind | Call | Models | Description |
|---|---|---|---|
| `influence` | `plot_influence()` | GLM/NLS + hat | Leverage against Cook's distance with annotated rows. |
| `residuals_vs_leverage` | `plot_residuals_vs_leverage()` | GLM + hat | Residuals against leverage with Cook's-distance contours. |
| `cooks` | `cooks_distance_plot()` | GLM/NLS + hat | Cook's distance by observation index. |
| `dffits` | `dffits_plot()` | GLM/NLS + hat | DFFITS with the `2√(p/n)` threshold. |
| `dfbetas` | `plot_dfbetas()` | GLM/NLS + hat | Standardised leave-one-out coefficient changes as a heatmap. |
| `leverage` | `leverage_plot()` | GLM/NLS + hat | Hat diagonal with the `2p/n` reference line. |

## Longitudinal, mixed-effects and SAEM panels

| Kind | Call | Models | Description |
|---|---|---|---|
| `caterpillar` | `plot_caterpillar()` | Mix, SAEM + RE | Caterpillar plot of estimated random effects (BLUPs/EBEs). |
| `ranef_distribution` | `plot_ranef_distribution()` | Mix, SAEM + RE | Random-effect histograms with a matched normal density. |
| `qq_random` | `plot_qq_random()` | Mix + RE | Normal Q-Q plots of the estimated random effects. |
| `variogram` | `plot_variogram()` | Mix + correlation | Empirical semi-variogram of residuals over time. |
| `individuals` | `plot_individuals()` | Mix, SAEM + groups | Per-subject observed, IPRED and PRED curves. |
| `population_vs_conditional` | `plot_population_vs_conditional()` | Mix, SAEM | Observed response against population and conditional predictions. |
| `vpc` | `plot_vpc()` | Mix, SAEM + sim | Visual predictive check with Monte Carlo envelopes. |
| `npde` | `plot_npde()` | SAEM + sim | NPDE values against time plus a normal Q-Q plot. |

## MCGLM panels

| Kind | Call | Models | Description |
|---|---|---|---|
| `pearson` | `plot_pearson()` | MCGLM | Pearson residuals against fitted mean, one panel per response. |
| `standardized_residual_qq` | `plot_standardized_residual_qq()` | MCGLM | Standardised residual distribution, one series per response. |
| `standardized_residual_density` | `plot_standardized_residual_density()` | MCGLM | Kernel density of standardised residuals per response. |
| `standardized_residuals_vs_fitted` | `plot_standardized_residuals_vs_fitted()` | MCGLM | Standardised residuals against within-response fitted quantiles, coloured by response. Aliases: `standardised_residuals_vs_fitted`, `mcglm_residuals_vs_fitted`, `residuals_vs_fitted_by_response`. |
| `binned_by_response` | `plot_binned_by_response()` | MCGLM | Binned standardised residuals for every response on a common scale. |

## Classification panels

`calibration` is listed above because it is the binary-GLM reliability panel.
`classification_calibration` below is the equivalent for a
`ClassificationEvaluator`.

| Kind | Call | Models | Description |
|---|---|---|---|
| `roc` | `plot_roc()` | GLM binary, evaluator | ROC curve. |
| `pr` | `plot_pr()` | GLM binary, evaluator | Precision-recall curve. |
| `confusion` | `confusion_matrix_plot()` | GLM binary, evaluator | Confusion matrix at a decision threshold. |
| `classification_calibration` | `plot_classification_calibration()` | evaluator | Reliability diagram for a probabilistic classifier. |
| `gains` | `plot_gains()` | GLM binary, evaluator | Cumulative gains chart. |
| `lift` | `plot_lift()` | GLM binary, evaluator | Lift chart. |
| `threshold_metrics` | `plot_threshold_metrics()` | GLM binary, evaluator | Accuracy/precision/recall/F1 against the decision threshold. |

## Convergence traces

| Kind | Call | Models | Description |
|---|---|---|---|
| `convergence` | `plot_convergence()` | iterative fits | Dispatches to the convergence view matching the result. Alias: `trace`. |
| `saem_trace` | `plot_saem_trace()` | SAEM | SAEM parameter traces with real labels and the `K1` phase marker. |
| `mcglm_convergence` | `plot_mcglm_convergence()` | MCGLM | Parameter trajectories and score norms of the MCGLM algorithm. |

## Typical suggestions by result class

The exact list is data- and family-dependent. `suggest_plots(fit)` is always
the authoritative answer; the table below summarizes the common core.

| Result | Usually suggested |
|---|---|
| `GLM` | `diagnostics`, residual checks, `observed_vs_fitted`, `forest`, `cor_fixef`; plus count (`rootogram`, `randomized_quantile_qq`, `quantile_residuals_vs_fitted`, `residuals_vs_leverage`), continuous and Tweedie (`quantile_scale_location`, Q-Q with 95% band, quantile residuals vs `η`), binary (`calibration`, Q-Q with 95% band, `roc`, ...), multinomial (`category_probs`, `multiclass_binned_residuals`, `multiclass_calibration`), quasi (Pearson checks on `η`), ordinal, truncated (`truncated_diagnostics` is the default diagnostics panel for continuous `TruncatedGaussian` fits), hurdle, Lorenz and influence panels when applicable. Beta/simplex fits get observed-vs-fitted, a quantile-residual Q-Q (annotated with fixed `φ`/`σ²`), quantile residuals vs `η`, and either `influence` (constant dispersion) or `secondary_vs_eta` (modelled dispersion). Censored AFT and GeneralizedGamma fits get `cox_snell`, `deviance_martingale_vs_eta` and a censoring-aware observed-vs-fitted; AFT fits add `influence`, while GeneralizedGamma keeps `gengamma`. |
| `nls` / `NLS` | Residual checks, `observed_vs_fitted`, `forest`, `curve` for one predictor, and `influence`, `cooks`, `leverage`. |
| `LmeFit` / `NlmeFit` / `LmerFit` / `GlmerFit` | Residual checks, `gof`, `forest`, `cor_fixef`; plus `qq_residuals` for continuous fits, `individuals`, `vpc`, `variogram`, random-effect panels and variance components when available. LME defaults to a four-panel mosaic of population vs. conditional predictions, normalised residuals vs. the conditional fit with a LOESS band, a Q-Q with 95% envelope and a BLUP caterpillar with conditional 95% intervals. NLME uses the same shape but plots residuals against the structural variable and automatically selects the dominant random-effect term. GLS adds `gls_diagnostics`, `observed_vs_fitted`, `normalized_residuals_and_spread` and `grouped_acf`; GNLS adds the analogous `gnls_diagnostics` panels. Discrete GLMER adds `glmer_diagnostics`, `randomized_quantile_qq` and `binned_quantile_residuals_by_rank`; continuous GLMER keeps `binned_standardized_residuals`. Binary fits also get ROC/PR/calibration. |
| `SaemFit` | `diagnostics` / `saem_diagnostics`, `trace`, `forest`; plus `residuals_vs_predictions`, `qq_plot_with_envelope`, `grouped_acf`, random-effect panels, `gof`, `individuals`, `population_vs_conditional`, `npde` and `vpc` when the data and simulator allow. |
| `MCGLMResults` | Residual checks plus `pearson`, `convergence`, `covariance`, standardised-residual panels, `standardized_residuals_vs_fitted` (within-response fitted quantiles, coloured by response) and `binned_by_response`; `correlation` when there is more than one response. |
| `ListFit` | `list_coefficients` (facade `plot_list_coefs()`). |

## Data helpers

Three helpers expose the plotted data as DataFrames or arrays for custom figures:

```python
from statcraft.plots import rootogram_data, npde_values, empirical_variogram

counts = rootogram_data(fit)                       # count, observed, expected
npde = npde_values(fit, nsim=500, seed=42)
semi = empirical_variogram(fit, n_lags=20)         # lag, semivariance, n_pairs
```

## See also

- [GLM](glm.md), [NLME](nlme.md), [NLS](nls.md), [MCGLM](mcglm.md) and
  [SAEM](saem.md) — engine-specific result APIs and plot options.
- [Troubleshooting](troubleshooting.md) — optional matplotlib installation and
  unknown-kind errors.
