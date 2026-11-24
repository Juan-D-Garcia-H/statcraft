# statcraft

`statcraft` is a Python package for statistical model fitting with compiled C++/Eigen engines (nanobind bindings). It targets applied statistics and biostatistics/pharmacometrics workflows and provides high-level APIs for:

- **GLM**: generalized linear and extended mean/dispersion/censored/hurdle/zero-inflated/quasi/MLE families
- **MCGLM**: multivariate covariance generalized linear models
- **NLME**: linear/nonlinear mixed-effects, GLS/GNLS, and mixed-model helpers (`lmer`, `glmer`, structures)
- **NLS**: standalone nonlinear least squares (single curve)
- **SAEM**: nonlinear mixed-effects models estimated by SAEM (including censoring, non-Gaussian families, simulation-based diagnostics)

The package also includes shared summary/reporting utilities, data helpers, example scripts, and example datasets.

> **Status**: research/engineering codebase under active development. APIs and implementation details may evolve; consult module docstrings and `examples/` for current behavior.

---

## Table of contents

- [Key capabilities](#key-capabilities)
- [Requirements](#requirements)
- [Installation](#installation)
- [Import conventions](#import-conventions)
- [Quick start](#quick-start)
- [Module guide](#module-guide)
  - [GLM (`statcraft.glm`)](#glm-statcraftglm)
  - [MCGLM (`statcraft.mcglm`)](#mcglm-statcraftmcglm)
  - [NLME (`statcraft.nlme`)](#nlme-statcraftnlme)
  - [NLS (`statcraft.nls`)](#nls-statcraftnls)
  - [SAEM (`statcraft.saem`)](#saem-statcraftsaem)
  - [Summaries, plots and utilities](#summaries-plots-and-utilities)
- [Data and examples](#data-and-examples)
- [Project layout](#project-layout)
- [Build and development notes](#build-and-development-notes)
- [Limitations and caveats](#limitations-and-caveats)
- [License and authors](#license-and-authors)

---

## Key capabilities

### GLM
- Formula interface (`import statcraft.glm as scg`; `scg.glm(...)`, family shortcuts such as `scg.poisson(...)`, `scg.binomial(...)`, etc.) and matrix interface (`from statcraft.glm import GLM, ...`; `GLM(X, y, family=...)`).
- Broad family support in `statcraft.glm` (exponential/quasi/MLE/extended families), including common links and specialized families (e.g. count, binary/multinomial/ordinal, gamma/beta/inverse-Gaussian, truncated/hurdle/zero- or value-inflated, generalized gamma, and more; see `statcraft.glm` exports).
- `GLM.fit(...)` dispatches to the appropriate internal engine by `family_type` while exposing a consistent result surface where available.
- Inference and reporting:
  - `summary()` (text/HTML-capable summary objects), `conf_int(...)`, predictions/fitted values and residual/diagnostic helpers.
  - Structured coefficient/result export as DataFrames (e.g. `coef_table(...)`, `summary_frame(...)`, `to_dataframe()`).
  - Exponentiated effect measures with transformed intervals where applicable (e.g. odds ratios / IRR helpers).
  - Robust (sandwich) covariance options where supported (e.g. HC variants and cluster-robust covariance via `cov_type` / `cov_kwds` on relevant methods/results).
  - Model/inference helpers such as Wald tests, ANOVA/comparison and influence/diagnostic wrappers (availability depends on family/model form).

### MCGLM
- `MCGLM`, `MCGLMResults`, and covariance-structure constructors (`mc_*`: identity/diagonal/group/mixed, moving-average/random-walk/CAR, distance/kernel structures, DGLM, twin structures, etc.).
- Models mean via link functions and covariance as linear combinations of known matrices with multivariate responses/correlation structures.
- Result utilities:
  - Fitted and residual access, including by-response views and Pearson/standardized residuals.
  - `wald_test(...)` and `anova(...)` helpers for hypothesis testing/model comparison.
  - `get_prediction(...)` and `to_frame(...)` for prediction/export workflows.

### NLME
- `LME`, `GLS`, `NLME`, `GNLS`, `LMER`, `GLMER` (and lowercase aliases `lme`, `gls`, `nlme`, `gnls`, `lmer`, `glmer`), `lm_list`, `nls_list`.
- Variance/correlation/random-effects structures: `var*`, `cor*`, `pd*` (e.g. `varIdent`, `corAR1`, `pdDiag`, etc.).
- Chainable model construction + fitting: `Model(...).fit() -> fit/result object`. All mixed-model constructors, including `lmer`/`glmer`, are unfitted until `.fit()` is called.
- `glmer` accepts the same GLM-style family objects as `glm` (`Binomial("logit")`, `Poisson("log")`, `NegativeBinomial(theta=2.0)`, `Gaussian()`, `Gamma()`, `InverseGaussian()`); these classes are re-exported from `statcraft.nlme`, and family strings are rejected.
- Inference and diagnostics:
  - `conf_int(alpha=...)` for fixed effects and variance components (as available).
  - `wald_test(...)` for linear hypotheses on fixed effects.
  - `tTable(ddf="satterthwaite")` for lmerTest-style denominator df (works with weights/correlation and crossed `lmer`; ML fits are warned and refitted in REML when possible).
  - `anova(other, ddf="satterthwaite")` for a Wald F test of extra fixed effects; `anova(other)` keeps the usual chi-square LRT (compare fixed-effects structures under ML when using LRT).
  - `residuals(type="response"|"pearson"|"normalized", level=...)` with backward-compatible positional level usage; normalized/marginal residuals are reconstructed when supported and otherwise fall back (e.g. to Pearson scaling) depending on model/structure.

### NLS
- Standalone nonlinear least squares for a **single curve** (`statcraft.nls.nls` / `NLS`) using SciPy least-squares machinery.
- Focused on curve-fitting features (bounds, robust losses, multistart, covariance/prediction utilities, persistence/options depending on implementation).
- Feature-frozen for grouped/random-effects/correlated/heterogeneous-structure nonlinear modeling; use `statcraft.nlme` (`nlme`/`gnls`/`nls_list`) for those cases.

### SAEM
- `SAEM`/`saem` API with `NLModel`, model/control configuration, data preparation, and `SaemFit` results.
- Supports subject/time/response data, covariates, censoring, dose/event-style inputs and built-in model helpers (module-dependent; see `statcraft.saem`).
- Result capabilities (subject to model/family/method settings): fixed/random effects (`fixef`, `ranef`, `coef`), `pop_params`, `vcov`, `omega`, `sigma`, fitted/residuals/predict/simulate/summary/anova/to_frame and plotting helpers where implemented.
- `conf_int(alpha=..., scale="estimation"|"natural")`:
  - estimation/phi scale intervals, and
  - natural-scale intervals for structural parameters via parameter transforms (covariate-effect coefficients remain on estimation scale).
- Simulation-based diagnostics:
  - `npde(nsim=..., seed=..., decorrelate=False|True)`
  - `residuals(type="npde"|"quantile", nsim=..., seed=...)`
  - Uses the C++ simulation engine; applicable to censored data and non-Gaussian families where standard `fitted()`/`residuals()` may be unavailable.
  - `decorrelate=True` performs per-subject empirical whitening/decorrelation for NPDE-style discrepancies; censored observations are handled via randomized prediction discrepancies from the simulated CDF at the censoring bound.

### Shared components
- `statcraft.summary`: summary tables/rendering/export infrastructure used by model results.
- `statcraft.datasets`: dataset directory resolution, `data_loader`, catalogue/registry and CLI.
- `statcraft.utils`: formula/util helpers plus a compatibility alias for `data_loader`.
- `statcraft.plots` (if available in the installed build): plotting helpers.
- Bundled example datasets are installed under `statcraft/data` and available through `data_loader("filename.csv")` without any path configuration. The repository checkout also keeps the original `data/` directory and `examples/` scripts.

---

## Requirements

- **Python**: >= 3.11
- **Build system**: `scikit-build-core>=0.11`, CMake >= 3.19, `nanobind>=2.0`
- **Build/runtime numeric deps**: `numpy>=1.22`
- **Runtime dependencies** (from `pyproject.toml`): `numpy`, `pandas`, `scipy`, `sympy`, `formulaic`
- **Optional**: `matplotlib>=3.6` (extra: `viz`), `pyarrow>=10` (extra: `data`, fast CSV/Parquet engine), `pytest>=7` (extra: `test`)
- **System/toolchain**: a C++ compiler/toolchain supported by your platform and the project’s CMake configuration; linear algebra/backend libraries as required by the build environment (platform-specific)

---

## Installation

### From source (repository checkout)

```bash
git clone <repository-url>
cd statcraft
pip install .
```

Editable/development install:

```bash
pip install -e .
```

With optional visualization dependencies:

```bash
pip install ".[viz]"
```

### Build notes

- This project builds compiled extensions during installation (`build-backend = scikit_build_core.build`).
- If the build fails, verify: Python/CMake versions, a working C++ compiler, and that `nanobind`/`numpy` build requirements are available in the build environment.
- Build artifacts may be placed under `build/` (configured by `scikit-build`).

---

## Import conventions

`import statcraft as sf` exposes the full public API: every engine namespace plus all names exported by them (family classes and shortcuts, model classes, variance/correlation structures, plotting helpers, summary/data utilities). Names that are also submodules stay bound to the module.

Common patterns:

```python
import statcraft as sf
import statcraft.glm as scg

# formula-style GLM entry points live in the glm module
scg.glm("y ~ x", data=df, family=scg.Binomial())
scg.poisson("y ~ x", data=df)

# everything is top-level too: families, shortcuts, structures, plot helpers
sf.Binomial("logit"), sf.poisson("y ~ x", data=df)
sf.varIdent, sf.corAR1, sf.plot, sf.build_summary
sf.SAEM(...)
sf.nlme, sf.mcglm, sf.saem, sf.glm  # modules, not callable estimators
```

Direct imports are also used in several modules:

```python
from statcraft.glm import GLM, Binomial
from statcraft.nls import nls, NLS
from statcraft.nlme import LME, lmer, varIdent, corAR1
from statcraft.mcglm import MCGLM, mc_id, mc_ma
```

Notes:
- `statcraft.glm` is the module containing the formula entry points and family helpers; it is not itself callable.
- `statcraft.nls` is a subpackage/module; standalone estimators are imported explicitly as `from statcraft.nls import nls, NLS`.
- Some APIs follow a “construct model, then `.fit()`” pattern (notably NLME and SAEM high-level interfaces).
- `print(fit)` uses a compact Python representation; call `fit.summary()` for the full tables.

---

## Quick start

### GLM (logistic regression, formula)

```python
import pandas as pd
import statcraft.glm as scg

df = pd.read_csv("data.csv")
fit = scg.glm(
    "y ~ x1 + x2 + x3",
    data=df,
    family=scg.Binomial(link="logit"),
).fit()

print(fit.summary())
print(fit.conf_int())
print(fit.coef_table())  # structured coefficients/results table (DataFrame)

# Robust covariance (when supported for the fitted model/family):
# print(fit.coef_table(cov_type="HC3"))
# print(fit.get_robustcov_results(cov_type="cluster", cov_kwds={"groups": groups}))

# Exponentiated effects for logit/log models where applicable:
# print(fit.odds_ratios())
# print(fit.irr())
```

### GLM (matrix API)

```python
import numpy as np
from statcraft.glm import GLM, Poisson

X = np.column_stack([np.ones(n), x1, x2])
y = ...
fit = GLM(X, y, family=Poisson(link="log")).fit()
print(fit.summary())
```

### MCGLM

```python
import numpy as np
import statcraft.mcglm as mc

n = 500
x = np.random.default_rng(0).random(n)
X = np.column_stack([np.ones(n), x])
y = np.random.default_rng(1).poisson(np.exp(0.5 + 0.8 * x)).astype(float)

fit = mc.MCGLM(y, X, [mc.mc_id(n)], link="log", variance="tweedie").fit()
print(fit.summary())

# Additional result utilities (depending on model setup):
# fit.resid_pearson
# fit.resid_standardized
# fit.wald_test(r_matrix)
# fit.anova(other_fit)
# fit.get_prediction(exog=X_new, response=0)
# fit.to_frame()
```

### NLME (LME)

```python
from statcraft.datasets import data_loader
from statcraft.nlme import LME

# Uses the datasets bundled in the installed package.
orth = data_loader("Orthodont.csv").load()

fit = LME("distance ~ age * Sex", orth, random="~ age | Subject").fit()
print(fit.summary())
print(fit.conf_int(alpha=0.05))

# Linear hypothesis on fixed effects (example contrast)
# print(fit.wald_test([[0, 1, 0, 0]]))

# Residuals
# raw/pearson/normalized (normalized availability depends on model/structure)
print(fit.residuals(type="pearson").head())

# Model comparison (use ML when comparing fixed-effects specifications)
# fit_full_ml.anova(fit_reduced_ml)
```

### NLS (standalone single-curve fitting)

```python
from statcraft.nls import nls

# API depends on model specification (formula/function/data); consult statcraft.nls docs/examples.
fit = nls(...)
print(fit.summary())
```

### SAEM

```python
import statcraft as sf
from statcraft.saem import models

# Example structure; adapt columns/start/control to your data and model
model = sf.SAEM(
    models.pk1_oral(dose="dose"),
    data=df,
    id="id",
    time="time",
    response="conc",
    start={"ka": 1.5, "CL": 3.0, "V": 30.0},
)

fit = model.fit()
print(fit.summary())

# Confidence intervals (estimation/phi scale, or natural scale for structural parameters)
print(fit.conf_int(alpha=0.05))
print(fit.conf_int(alpha=0.05, scale="natural"))

# Simulation-based residuals / NPDE (useful for censored/non-Gaussian cases)
npde = fit.npde(nsim=1000, seed=1, decorrelate=True)
qres = fit.residuals(type="quantile", nsim=1000, seed=1)
```

---

## Module guide

### GLM (`statcraft.glm`)

**Entrypoints and classes**
- `glm(...)` / family shortcuts (e.g. `gaussian`, `binomial`, `poisson`, `gamma`, `negative_binomial`, quasi/MLE/extended shortcuts) for formula-based construction.
- `GLM` for array/matrix-based construction; `GLM.from_formula(...)` for formula parsing.
- Extensive family classes in `statcraft.glm.family` / exported from `statcraft.glm` (e.g. `Gaussian`, `Binomial`, `Poisson`, `Gamma`, `InverseGaussian`, `NegativeBinomial`, `Tweedie`, `Beta`, `Multinomial`, `Ordinal`, truncated/hurdle/zero-inflated/value-inflated families, `GeneralizedGamma`, quasi families, custom families, etc.).

**Model specification notes**
- Standard formulas follow R-style syntax (via `formulaic`).
- Some extended models use specialized formula forms (e.g. two-part/hurdle/zero-inflated formulas and multi-parameter specifications such as generalized-gamma submodels); check `glm.py`/family docs for the exact syntax.

**Fitting and results**
- `.fit(...)` returns a fitted result/model object (engine dispatch depends on family type).
- Typical result capabilities include parameter estimates, standard errors/covariance (as available), log-likelihood/deviance/information criteria (model-dependent), fitted values, predictions, and summaries.
- **Inference/export helpers** (availability may vary by family/engine):
  - `conf_int(...)`
  - `coef_table(...)`, `summary_frame(...)`, `to_dataframe(...)`
  - exponentiated coefficient/CI helpers (e.g. OR/IRR methods where meaningful)
  - robust covariance results and `cov_type`/`cov_kwds` support on relevant methods
  - tests/comparisons and diagnostics (`wald_test`, `anova`, influence/residual diagnostics wrappers)

**Example (binomial/logit)**

```python
import statcraft.glm as scg

fit = scg.glm(
    "spam ~ to_multiple + format + attach + dollar + winner",
    data=email_df,
    family=scg.Binomial(),
).fit()
print(fit.summary())
```

---

### MCGLM (`statcraft.mcglm`)

**Entrypoints and building blocks**
- `MCGLM`, `MCGLMResults`, `mcglm`.
- Link/variance/covariance/method/structure enums and options exposed from the compiled engine (`_mcglmfit`) and high-level wrappers.
- Covariance/matrix structure constructors (`mc_*`): e.g. `mc_id`, `mc_diag`, `mc_group`, `mc_mixed`, `mc_ma`, `mc_rw`, `mc_car`, `mc_dist`, `mc_exp`, `mc_matern32`, `mc_wendland`, `mc_ns`, `mc_dglm`, `mc_twin`, `mc_twin_bio`, `mc_twin_full`.

**Modeling concept**
- Mean modeled via link functions per response and covariance modeled via linear combinations of known matrices (`Z`) with response/dispersion structure and cross-response correlation (per Bonat & Jørgensen framework implemented by the engine).

**Result utilities (commonly used)**
- `summary()`, fitted values and residuals (including by-response views)
- `resid_pearson`, `resid_standardized`
- `wald_test(r_matrix, q_matrix=None)` and `anova(other)`
- `get_prediction(...)` (delta-method-based prediction/interval workflow where supported)
- `to_frame(...)` for exporting parameters/results

---

### NLME (`statcraft.nlme`)

**Entrypoints**
- `LME`/`lme`, `GLS`/`gls`, `NLME`/`nlme`, `GNLS`/`gnls`, `LMER`/`lmer`, `GLMER`/`glmer`, `lm_list`, `nls_list`
- Structure constructors: `varComb`, `varConstPower`, `varConstProp`, `varExp`, `varFixed`, `varIdent`, `varPower`, `corAR1`, `corARMA`, `corCAR1`, `corCompSymm`, `corExp`, `corGaus`, `corLin`, `corRatio`, `corSpher`, `corSymm`, `pdBlocked`, `pdDiag`, `pdIdent`, `pdSymm`, etc.

**Usage pattern**
- Define the model and call `.fit()`:

```python
from statcraft.nlme import LME
fit = LME("distance ~ age * Sex", data=orth, random="~ age | Subject").fit()

from statcraft.nlme import Poisson, Gaussian, Gamma, InverseGaussian, lmer, glmer
lmer_fit = lmer("Reaction ~ Days + (Days | Subject)", sleepstudy).fit()
glmer_fit = glmer("TICKS ~ YEAR + (1 | BROOD)", grouse, family=Poisson("log")).fit()

# Continuous responses (dispersion estimated unless dispersion= is given)
gauss_fit = glmer("score ~ 1 + (1 | Machine)", machines, family=Gaussian()).fit()
gamma_fit = glmer("score ~ 1 + (1 | Machine)", machines, family=Gamma("log")).fit()
ig_fit = glmer("score ~ 1 + (1 | Machine)", machines, family=InverseGaussian()).fit()
fixed_fit = glmer("score ~ 1 + (1 | Machine)", machines, family=Gaussian(), dispersion=30.0).fit()
```

- `lmer` and `glmer` follow the same rule: construction returns an unfitted model and `.fit()` returns `LmerFit`/`GlmerFit`.
- `glmer` requires a GLM family object or class, such as `family=Poisson("log")`; family strings are rejected.
- Continuous families support `identity`/`log`/`inverse`/`sqrt` (Gaussian), `inverse`/`log`/`identity`/`sqrt` (Gamma) and `invmu2`/`log`/`identity`/`inverse` (inverse Gaussian). `dispersion` is `sigma^2` for Gaussian and `1/shape` (`1/lambda`) for Gamma/inverse Gaussian; the fit result exposes it as `.dispersion` (plus `.sigma` for Gaussian). The Gamma and inverse Gaussian families require a strictly positive response.

- `nlme`/`gnls` also accept Python-callable nonlinear model specifications (with parameter/start/response definitions as required by the API).
- `summary()` uses balanced two-column headers, a compact aligned `Corr` column for random-effect correlations (no commas, no column per pair), R²/ICC for mixed fits, residual-quantile tables, and GLMM goodness-of-fit panels. `lm_list`/`nls_list` results also provide per-group summary tables.

**Inference and diagnostics**
- `conf_int(alpha=0.05)` -> confidence intervals (DataFrame) for fixed effects and available variance components.
- `wald_test(r_matrix, q_matrix=None)` -> Wald test for linear hypotheses on fixed effects.
- `tTable(ddf="containment"|"satterthwaite")` -> fixed-effect table; `"satterthwaite"` uses the REML covariance model (auto-refit from ML when the original model object is available). `ddf="kr"` is reserved for a future Kenward-Roger implementation.
- `anova(other, ddf="chi2"|"satterthwaite")` -> chi-square LRT by default (fit/compare fixed effects under ML), or a Wald F test with Satterthwaite denominator df for nested fixed-effect structures under REML.
- `residuals(type="response"|"pearson"|"normalized", level=None)`
  - `response`: raw residuals
  - `pearson`: scaled by residual SD
  - `normalized`: marginal covariance-standardized residuals \(V^{-1/2}(y-\mu)\) when reconstructable (important with weights/correlation structures); falls back to Pearson scaling when full marginal covariance is unavailable from the current fit/engine representation

---

### NLS (`statcraft.nls`)

**Purpose**
- Standalone nonlinear least squares for a single response/curve.

**API**
- `from statcraft.nls import nls, NLS`

**Scope and limitations**
- Supports single-curve fitting features provided by the subpackage; **not** intended for random effects, grouped hierarchical structures, residual correlation models, or variance-function modeling. Use `statcraft.nlme` (`nlme`, `gnls`, `nls_list`) for those scenarios.

---

### SAEM (`statcraft.saem`)

**Main components**
- `SAEM` / `saem` (model specification + fitting workflow)
- `NLModel`, `models`/`builtin_models`, `Doses`
- `SaemControl`, `saem_control`
- `DataInfo`
- `SaemFit`, `SaemSummary`, `SaemWarning`
- GLM family classes (`Binomial`, `Poisson`, `Gaussian`, ...) are re-exported for `family=`
- `datasets`

**Typical workflow**

```python
import statcraft as sf
from statcraft.saem import models, saem_control

model = sf.SAEM(
    models.pk1_oral(dose="dose"),
    data=df,
    id="id",           # optional in some cases; see data preparation behavior
    time="time",       # optional for static models without events/TTE/builtin time needs
    response="conc",
    start={"ka": 1.5, "CL": 3.0, "V": 30.0},
)
fit = model.fit(
    # control options may be passed here or via control=saem_control(...)
)
```

`family=` accepts only GLM family objects/classes re-exported from
`statcraft.saem` (e.g. `family=Binomial("logit")`, `family=Poisson()`,
`family=Weibull()` with `cumhazard=`, or `family=CustomFamily` with `loglik=`).
Likelihood strings are rejected; `fit.model.family` still records the canonical
likelihood for inspection.

**Data and specification notes** (high level)
- Data are typically in long format (one row per observation).
- Supports optional `id`, `time`, `response`/`y`, response typing/censoring/limits/grouping, dose/event inputs (`events`, `evid`, `event_cols`), predictors/covariates/references and NA handling depending on API options.
- If no subject id is provided and none is inferred, rows may be treated as separate subjects (with a warning) and random effects may not be identifiable; provide `id` when grouped data are intended.
- Static/time-free models can be handled in specific cases; event-based/TTE/builtin time-dependent models require proper time information.

**Result methods/properties (selected)**
- Identification/fit info: `n_obs`/`nobs`, `n_subjects`, `n_params`, `df`, `loglik`, `aic`, `bic`, warnings/raw result access
- Parameters/effects: `fixef(scale="estimation"|"natural")`, `pop_params()`, `ranef(...)`, `coef()`, `vcov()`, `omega()`, `sigma()`
- Intervals: `conf_int(alpha=0.05, scale="estimation"|"natural")`
- Predictions/simulation: `fitted(...)`, `residuals(...)`, `predict(...)`, `simulate(...)`, `simulate_events(...)` (availability depends on family/model)
- Diagnostics/comparison/export: `npde(...)`, `summary()`, `anova(...)`, `to_frame()`

**Confidence intervals**
- `conf_int(alpha=0.05, scale="estimation")`: Wald intervals on estimation (phi) scale.
- `conf_int(alpha=0.05, scale="natural")`: transforms structural-parameter bounds via model parameter transforms (useful for PK/PD reporting); covariate-effect coefficients remain on estimation scale.

**Simulation-based residuals / NPDE**
- `npde(nsim=1000, seed=1, decorrelate=False)` and `residuals(type="npde"|"quantile", nsim=..., seed=...)`.
- Uses `_core.simulate` from the compiled SAEM engine.
- Works for censored data and non-Gaussian families where Gaussian-only `fitted()`/`residuals()` may raise `NotImplementedError`.
- `decorrelate=True`: per-subject empirical covariance whitening of simulated predictions before computing prediction discrepancies (NPDE-style).
- Censored observations are incorporated via randomized prediction discrepancies based on the simulated CDF at the censoring bound (left/right censoring handling as applicable).

---

### Summaries, plots and utilities

- **`statcraft.summary`**: summary/table construction and rendering/export helpers used across result objects (text/HTML/DataFrame-oriented representations depending on object).
- **`statcraft.datasets`**: dataset loading, directory resolution and the generated `data/datasets.json` catalogue. Use `data_loader("file.csv")`, `list_datasets()` and `dataset_info()` from this single module (also available at top level as `statcraft.data_loader`).
- **`statcraft.utils`**: formula-related support and a compatibility alias for `data_loader`.
- **`statcraft.plots`**: unified plotting API. Every public entry point uses the word `plot`: `fit.plot()` / `fit.plot_diagnostics()` / `fit.plot_qq()` / `fit.qqplot()`, or `statcraft.plots.plot_diagnostics(fit)` / `statcraft.plots.qqplot(fit)`. Plot implementations stay private and are selected through `kind` aliases; matplotlib is optional (`viz` extra).

---

## Data and examples

- `data/`: source datasets used by examples; the wheel installs a copy as `statcraft/data`. `data_loader` finds either location automatically, regardless of the current working directory, unless `STATCRAFT_DATA_DIR` is set. `data/datasets.json` is the generated catalogue (name, format, rows, columns, size, modification time); it is refreshed automatically by `statcraft.list_datasets()` / `statcraft.dataset_info()` or `statcraft.datasets.*` whenever the directory changes, or on demand with `python -m statcraft.datasets --refresh`.
- `examples/`: runnable example scripts organized by module, e.g.:
  - `examples/glm/` (many family/use-case examples)
  - `examples/mcglm/`
  - `examples/nlme/`
  - `examples/nls/`
  - `examples/saem/`

Examples are a good reference for current API usage and expected inputs.

---

## Project layout

```text
.
├── bindings/              # nanobind/Python binding sources
├── include/               # C++ headers
├── src/
│   └── statcraft/
│       ├── glm/           # GLM families, fitting, diagnostics
│       ├── mcglm/         # MCGLM engine interface, structures, results
│       ├── nlme/          # LME/GLS/NLME/GNLS/lmer/glmer/structures
│       ├── nls/           # standalone nonlinear least squares
│       ├── saem/          # SAEM model/control/data/fit/results
│       ├── summary/       # summary/reporting framework
│       ├── utils/         # shared utilities
│       └── __init__.py    # eager full public API (namespaces + all exports)
├── examples/              # usage examples
├── data/                  # example datasets (installed as statcraft/data in the wheel)
├── CMakeLists.txt
├── pyproject.toml
└── README.md
```

---

## Build and development notes

- Packaging/build: `scikit-build-core` with nanobind extensions (`pyproject.toml`).
- Local development typically uses an editable install from the repo root (`pip install -e .`) in an environment with build tools available.
- Because compiled extensions are involved, code changes in C++/bindings require rebuilding/reinstalling the extension modules as appropriate.
- Build parallelism is controlled by the build tool: set `CMAKE_BUILD_PARALLEL_LEVEL=4` or pass `-Cbuild.tool-args=-j4`. CMake defaults to sequential module builds only on low-memory machines (`STATCRAFT_SEQUENTIAL_MODULES`); OpenMP is used automatically when available.
- The repository may contain local build artifacts (`build/`) and cached bytecode in working environments; avoid treating generated artifacts as source.

### Quick verification ideas

- Import core modules:
  ```bash
  python -c "import statcraft; import statcraft.glm, statcraft.mcglm, statcraft.nlme, statcraft.nls, statcraft.saem"
  ```
- Run selected example scripts from the repository root. They pass an explicit `data_dir_override` to the checkout's `data/`; installed users can load the same datasets directly with `data_loader("filename.csv")`.

---

## Support, contributing and changelog

See [SUPPORT.md](SUPPORT.md) for supported Python versions, platforms, compiler
requirements and compatibility policy. Development instructions are in
[CONTRIBUTING.md](CONTRIBUTING.md), and release notes are in
[CHANGELOG.md](CHANGELOG.md).

The full [user guide](docs/index.md) covers installation, data loading, each
engine, plotting and troubleshooting. The shared
[`f(x, p)` model-function convention](docs/model-functions.md) is the recommended
signature for NLS, NLME and SAEM callables. Preview the guide locally with
`mkdocs serve` after installing `mkdocs` and `mkdocs-material`.

---

## Limitations and caveats

- **Evolving API/behavior**: some method names, argument conventions, and engine behaviors may change as the project matures.
- **Family/model-specific availability**: not every result method is meaningful or implemented for every family, link, covariance structure, or fitting mode (GLM/MCGLM/NLME/SAEM).
- **Robust covariance (GLM)**: support depends on family/parameterization/engine path; treat robust options as model-dependent and validate for your use case.
- **NLME residuals**: `"normalized"` residuals require reconstructable marginal covariance; when unavailable, implementations may fall back to Pearson-scaled residuals.
- **NLME model comparison**: for LRT comparisons involving different fixed-effects structures, fit models under ML (REML likelihoods are not directly comparable in that setting).
- **SAEM Gaussian-only methods**: `fitted()`/standard `residuals()` may be restricted for certain families; simulation-based `npde()`/quantile residuals are provided as an alternative for censored/non-Gaussian cases.
- **SAEM NPDE**: prediction discrepancies for censored observations are randomized from the simulated CDF at the censoring bound; decorrelation is empirical per subject and depends on simulation size (`nsim`) and model fit quality.
- **NLS scope**: standalone NLS is intended for single-curve problems; use NLME for grouped/random-effects/correlated/structured-variance nonlinear modeling.
- **Performance/scalability**: depends on model size/structure and compiled backend characteristics; dense covariance/pseudo-criteria computations can be expensive for large problems.

Always validate statistical assumptions, numerical convergence, and inference choices for your specific model and dataset.

---

## License and authors

- **License**: Apache-2.0 (see `LICENSE`)
- **Author**: Juan Garcia (<juandavidgh2011@gmail.com>)
