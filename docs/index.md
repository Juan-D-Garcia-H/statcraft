# statcraft

`statcraft` fits statistical models in Python using compiled C++23 engines. It
provides five model families behind a consistent, statsmodels-style API, with
R-style formulas, bundled example datasets, summary tables and diagnostic plots.

| Engine | Use it for | Main entry point | Manual |
|---|---|---|---|
| GLM | Generalized linear models, including zero-inflated, hurdle, truncated and survival families | `statcraft.glm.glm` / `statcraft.glm.GLM` | [GLM](glm.md) |
| NLME | Linear and nonlinear mixed-effects models, correlated errors, variance structures | `statcraft.nlme.lme`, `nlme`, `gls`, `gnls`, `lmer`, `glmer` | [NLME](nlme.md) |
| NLS | Single-curve nonlinear least squares with robust losses and multistart | `statcraft.nls.nls` (formula) / `statcraft.nls.NLS` (callable) | [NLS](nls.md) |
| MCGLM | Multivariate GLM with covariance and power structures | `statcraft.mcglm.mcglm` / `MCGLM.from_formula` | [MCGLM](mcglm.md) |
| SAEM | Nonlinear mixed effects via stochastic approximation EM, including PK/PD and censored data | `statcraft.SAEM` | [SAEM](saem.md) |

## 60-second example

```python
import statcraft.glm as scg
from statcraft.datasets import data_loader

affairs = data_loader("Affairs.csv").load()

fit = scg.glm(
    "affairs ~ age + yearsmarried + religiousness + rating",
    data=affairs,
    family=scg.Poisson(),
).fit()

print(fit.summary())
```

## Four conventions to remember

1. **Import by namespace for formulas.** `sf.glm` is the *module*, not the
   estimator. Use `import statcraft.glm as scg` and `scg.glm(...)`, or the
   top-level matrix class `sf.GLM(...)`. `import statcraft as sf` already
   exposes every public family and helper (`sf.Binomial`, `sf.poisson`,
   `sf.varIdent`, `sf.plot`, ...). Likewise, `statcraft.nls` is the subpackage;
   the estimator class is `sf.NLS` and the lowercase function is
   `from statcraft.nls import nls`.

2. **Models start unfitted.** Every constructor returns an unfitted model; call
   `.fit()` to estimate it. GLM and NLS return the same object, now fitted;
   NLME, MCGLM and SAEM return a dedicated fitted-result object.

3. **Data loading is location-independent.** `data_loader("soya.csv").load()`
   finds the bundled datasets from an installed wheel, a source checkout or an
   editable install, regardless of the current working directory.

4. **Summaries are explicit.** `repr(fit)` and `print(fit)` show a compact
   description; call `fit.summary()` when you want the full tables.

```python
import statcraft as sf

print(sf.__version__)
print(sf.SAEM, sf.MCGLM, sf.NLS, sf.Binomial, sf.plot)  # all top-level
```

## Where to go next

- [Installation](installation.md) — packages, extras and building from source.
- [Data loading](data-loading.md) — dataset search order and overrides.
- [Model functions](model-functions.md) — the shared `f(x, p)` callable convention.
- Engine manuals: [GLM](glm.md), [NLME](nlme.md), [NLS](nls.md),
  [MCGLM](mcglm.md) and [SAEM](saem.md). Each one lists every constructor
  option, fitting option and result object.
- [Plotting and diagnostics](plotting.md) — `fit.plot(kind=...)` and the kind registry.
- [Troubleshooting](troubleshooting.md) — common import, data and fitting issues.

Additional runnable scripts live in `examples/` in the repository; they are the
de facto extended smoke tests for each engine.
