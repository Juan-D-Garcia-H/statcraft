# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `statcraft.datasets` is now the single public module for datasets: directory
  resolution, `data_loader`/`DatasetLoader`, the generated `data/datasets.json`
  catalogue, `list_datasets()`/`dataset_info()` and a `python -m
  statcraft.datasets` CLI. Top-level shortcuts `statcraft.data_loader`,
  `statcraft.list_datasets` and `statcraft.dataset_info` are available too.
- Added `kind="truncated_diagnostics"` for continuous truncated GLM fits: a
  2×2 mosaic with a Q-Q envelope, residual and spread trends against `η`, the
  observed density against the average truncated predictive density, and a
  calibration curve for the fitted truncated mean. It is the default
  `fit.plot()` / `kind="diagnostics"` view for that family; the existing
  `truncated` plot is unchanged.
- MCGLM's diagnostics mosaic now includes `standardized_residuals_vs_fitted`:
  residuals against within-response fitted quantiles, coloured by response.
  The `covariance` panel separates correlations (ρ, fixed `[-1, 1]` axis) from
  dispersion/power parameters (τ, natural scale).
- LME and NLME now use a four-panel hierarchy mosaic in `fit.plot()`:
  population vs. conditional predictions, a level-1 residual trend with a
  LOESS band and Q-Q envelope, and a level-2 BLUP caterpillar with conditional
  95% intervals and the random-effect SD in the title. NLME plots residuals
  against the structural variable and selects the dominant random-effect term.
- GLS and GNLS now use dedicated four-panel mosaics in `fit.plot()`: observed
  vs fitted (with a LOESS curve for GLS), a normalised Q-Q with 95% envelope,
  normalised residuals and `√|r̂|` on independent data-scaled Y axes (against
  `ŷ` for GLS and the primary structural variable for GNLS), and an ACF
  accumulated within correlation groups with a `max(n_i) - 1` lag cap.
- Discrete GLMER fits now use a diagnostic mosaic built for counts and
  proportions: population vs conditional predictions, a DHARMa-style Q-Q of
  simulated quantile residuals, binned quantile residuals against `rank(ŷ)`,
  and a BLUP caterpillar with 95% conditional intervals. Bernoulli, grouped
  binomial `k/n`, Poisson and negative-binomial families are supported;
  `GlmerFit.resid_quantile()` and `GlmerFit.blup_conditional_se()` expose the
  underlying calculations. Continuous GLMER families keep their previous
  prediction/residual/BLUP layout.
- SAEM fits now use a dedicated four-panel diagnostic mosaic in `fit.plot()`:
  population vs conditional predictions, Pearson residuals vs IPRED (Gaussian)
  or censoring-aware NPDE vs time, a Q-Q with a 95% simulation envelope and a
  within-subject ACF. It is available as `kind="saem_diagnostics"` with the
  aliases `diagnose_saem`, `saem_panel` and `saem_mosaic`.
- `family=` in SAEM now accepts **only** GLM family objects and classes (for
  example `Binomial("logit")`, `Poisson("log")`, `NegativeBinomial()`), matching
  GLM and GLMER. Likelihood strings are rejected with an actionable error;
  `fit.model.family` still exposes the canonical likelihood for inspection.
  Survival families (`Weibull()`, `LogNormal()`, `LogLogistic()`,
  `GeneralizedGamma()`) map to the time-to-event likelihood with `cumhazard=`,
  and `CustomFamily` is the class token for `loglik=`. All GLM family classes
  are re-exported from `statcraft.nlme` and `statcraft.saem`, so callers no
  longer need to import them from `statcraft.glm`.

### Changed

- `import statcraft as sf` now exposes the complete public API, not a curated
  lazy subset: every name in each engine's `__all__` (families, shortcuts,
  structures, plotting and summary helpers) is bound at import time. Names that
  are submodules (`sf.glm`, `sf.nlme`, `sf.nls`, `sf.saem`, ...) remain modules;
  `sf.Binomial`, `sf.poisson`, `sf.varIdent` and `sf.plot` are now valid.
- **Breaking:** SAEM `family=` no longer accepts likelihood strings such as
  `"bernoulli_logit"` or `"custom"`; pass the equivalent family object/class
  (`Binomial("logit")`, `CustomFamily`, ...) instead.
- Callable estimators now share one set of constructor keywords: `response=`,
  `predictors=` and `params=`. The experimental `y=`, `exog_var=` and nonlinear
  `fixed=` spellings are no longer accepted.
- The two-argument callable form now accepts any argument names in NLME, GNLS
  and SAEM when more than one parameter is declared; `x`/`p` remain the
  conventional names.
- `params=` is now optional in NLME and GNLS callable models: the key order of
  `start=` supplies the parameter names and order. Pass `params=` explicitly for
  covariate formulas.
- Single-parameter callables now accept `params`, `parameters` or `theta` as
  the second argument name; `a = params` behaves as the parameter value while
  `a, = params` keeps working. NLME and GNLS also infer the predictor from the
  first argument when it names a data column.
- A single predictor is now available directly as `x` in `f(x, p)` models;
  `time, = x` remains supported.
- The GLM documentation now lists constructor arguments, defaults, valid links
  and secondary links for every public family.
- The engine manuals are now self-contained: every constructor, fitting or
  control option and every public result method/property is listed for GLM,
  NLME, NLS, MCGLM and SAEM. Shared pages cross-link each engine.
- The manuals now include worked examples for complex GLM families, MCGLM
  covariance matrices, mixed-model structures, dose events and custom SAEM
  likelihoods, plus a comparison of NLME result-class capabilities.
- Fitted objects use a compact Python representation. Full tables are only
  rendered by `fit.summary()`, including in notebooks.
- Dataset helpers moved from `statcraft.utils` to `statcraft.datasets`;
  `statcraft.utils.data_loader` remains as a compatibility alias.
- The generated plotting facade no longer repeats `plot` in names that already
  end with it: `fit.forest_plot()` replaces `fit.plot_forest_plot()`, and the
  joined spellings `fit.qqplot()` / `fit.ppplot()` replace `fit.plot_qqplot()`,
  `fit.plot_qq_plot()` and `fit.plot_pp_plot()`. The same names are available as
  `statcraft.plots.<name>(fit)`; the old forms are no longer exported.
- `suggest_plots()` for SAEM now advertises the diagnostics mosaic, the Q-Q with
  envelope, the within-subject ACF and the fixed-effect correlation panel when
  standard errors are available, instead of only the trace and forest plots.

### Fixed

- GNLS/NLME normalized and Pearson residuals now divide by the full
  observation SD `σ̂ · g(μ̂ᵢ, δ)` instead of `σ̂` alone. With `varPower` or
  `varExp`, the Q-Q slope now matches the N(0, 1) reference and residuals stay
  on the standard normal scale. LME applies the same variance-function factor
  to its marginal normalization.
- The GNLS mosaic now adds the LOESS calibration curve to observed vs. fitted,
  places the residual/spread legend in the upper-left corner, and computes the
  residual ACF within plots/subjects inferred from the data (falling back to a
  single series when no grouping factor exists).
- The GLS, GNLS and GLMER mosaic titles no longer repeat the acronym when a
  caller supplies a legacy ``"<Engine> diagnostics — <Engine>"`` title; the
  figure suptitle stays simply ``"<Engine> diagnostics"``.
- The GLS observed-vs-fitted panel now uses padded data limits per axis instead
  of a zero baseline, the ACF lag ticks are integers, the Q-Q legend says
  “Normalised residuals”, and the mosaic title no longer repeats “GLS”.
- The MCGLM covariance panel now leaves enough vertical room between the
  correlations and dispersion sub-panels, so the top x tick labels no longer
  collide with the bottom sub-title, and lifts the overall panel title clear
  of the correlations title.
- SAEM NPDE now randomizes censored predictive discrepancies with the
  documented convention: `-1` is left-censored and `+1` is right-censored.
- Censored SAEM plots now use NPDE instead of Pearson residuals, so censored
  rows are no longer dropped from residual, Q-Q and ACF panels. The residual
  reference lines use `±1.96` for NPDE and `±2` for Pearson residuals.
- SAEM plot axes now use the response column name from `DataInfo`, and random
  effects with no numeric columns no longer produce empty caterpillar panels.
- The SAEM trace plot now labels variance and error trajectories with their
  real names (`sd(ka)`, `omega(CL, ka)`, `sigma.a`, ...), uses the model function
  name in the title, marks the end of exploration with `K1` instead of the MCMC
  burn-in, and repeats the X-axis title only on the bottom row of panels.

## [0.1.0] - 2026-11-20

### Added

- First public release on PyPI.
- GLM, NLME, MCGLM, NLS and SAEM engines backed by four C++23 nanobind
  extension modules.
- Lazy, namespaced Python API with curated top-level exports.
- Bundled example datasets under `statcraft/data`, available through
  `statcraft.utils.data_loader`.
- Cross-platform wheels built with `cibuildwheel` for Linux (x86_64/aarch64),
  macOS (universal2) and Windows (AMD64), Python 3.11-3.14.
- Minimal pytest smoke suite and GitHub Actions CI.
- Shared `f(x, p)` model-function convention for NLS, NLME and SAEM, with
  direct unpacking plus name- and position-based access to exogenous values
  and parameters; older signatures remain supported.
- Shared constructor keywords across callable estimators: `response=`,
  `predictors=` and `params=`.

[0.1.0]: https://pypi.org/project/statcraft/0.1.0/
