# Troubleshooting

## Import and API shape

The complete import and constructor contracts are in the engine manuals:
[GLM](glm.md), [NLME](nlme.md), [NLS](nls.md), [MCGLM](mcglm.md) and
[SAEM](saem.md).

**`TypeError: 'module' object is not callable` for `sf.glm(...)`**

`statcraft.glm` is the module, not the estimation function. Use:

```python
import statcraft.glm as scg
fit = scg.glm("y ~ x", data=df, family=scg.Poisson()).fit()
```

or import the class directly:

```python
from statcraft.glm import GLM, Poisson
fit = GLM(X, y, family=Poisson()).fit()
```

`sf.poisson` and `sf.Binomial` **are** top-level exports now (families and
formula shortcuts come from every engine). `sf.glm` itself stays the module, so
call `scg.glm(...)` for formula entry points or `sf.GLM(...)` for matrices.

**`sf.nls(...)` does not work**

`statcraft.nls` is the subpackage. Import the estimator explicitly:

```python
from statcraft.nls import nls, NLS
```

**`ImportError` for `_glmfit`, `_nlmefit`, `_mcglmfit` or `_saemfit`**

The compiled extensions are missing or were built for another interpreter.
Reinstall the wheel for the active Python version, or rebuild the editable
install after changing C++/bindings:

```bash
python -m pip install --force-reinstall statcraft
# or, in a checkout
python -m pip install -e ".[viz,data,test]"
```

## Data loading

**`NotADirectoryError: Invalid data directory: .../statcraft/data`**

In an editable checkout the data live in the repository `data/` directory, not
under `src/statcraft/`. Modern versions resolve it relative to the module and
work from any CWD. If you use a custom location, set:

```bash
export STATCRAFT_DATA_DIR=/path/to/datasets
```

or pass `data_dir_override=` to `data_loader`.

**`PyArrow engine is not installed or failed. Falling back to the standard pandas parsing engine.`**

Informational: install `statcraft[data]` if you need Parquet or the faster CSV
path. CSV loading still works without pyarrow.

## Fitting

**GLM: `standardize=True is not currently supported together with a rank-deficient design matrix`**

Remove aliased/duplicate columns or fit with `standardize=False`.

**GLM: `start=` seems to be ignored for an MLE family**

Expected: MLE-based families build their own starting values. `start=` applies
to IRLS and quasi families.

**NLME: likelihood-ratio test looks wrong**

REML likelihoods are not comparable across different fixed effects. Fit both
models with `method="ML"`. Call `tTable(ddf="satterthwaite")` afterwards; it
refits to REML internally where possible.

**SAEM: `fitted()` or `residuals()` fails for a censored/non-Gaussian model**

Those methods may be unavailable. Use `fit.npde(...)`,
`fit.residuals(type="npde"|"quantile", ...)`, and the `npde` / `vpc` plots.

**NLS: fit lands in a local minimum**

Use `fit_multistart(n_starts=20, random_state=1)`, provide tighter `bounds=`, or
improve `start=`. For noisy data, try `loss="soft_l1"` or `loss="huber"`.

## Plots and optional backends

**`ModuleNotFoundError: matplotlib`**

Install the plotting extra: `python -m pip install "statcraft[viz]"`.

**JAX backend not available**

JAX is optional and undeclared. Install it with `python -m pip install jax`, or
keep `backend="numpy"` (the default), which uses finite differences.

**`ValueError: unknown plot kind ...`**

Call `statcraft.plots.available_kinds()` to see accepted names and aliases, and
`suggest_plots(fit)` for the ones that apply to your result.

## Building from source

**Compilation runs out of RAM**

Reduce build parallelism with `CMAKE_BUILD_PARALLEL_LEVEL=1` or force
`-DSTATCRAFT_SEQUENTIAL_MODULES=ON`.

**OpenMP message on macOS**

Stock AppleClang does not ship OpenMP. This is handled: the engines build
single-threaded and the message is informational. Install libomp and configure
it if you need multithreading.

**Network needed during a clean build**

Eigen, Boost.Math and nanobind are fetched with FetchContent when they are not
installed. Use a machine with network access, or provide system copies, for the
first build.

## See also

- [Model function signatures](model-functions.md) — callable forms and keyword
  names.
- [Data loading](data-loading.md) — dataset search order and overrides.
- [Plotting and diagnostics](plotting.md) — plot kinds and aliases.
