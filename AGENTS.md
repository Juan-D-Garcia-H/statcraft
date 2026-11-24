# AGENTS.md

## Build and verification

- Python >=3.11 and a C++23 toolchain. From the repo root: `pip install -e ".[viz,data,test]"` (extras: `viz`=matplotlib, `data`=pyarrow, `test`=pytest). Four nanobind extensions back the engines: `_glmfit`, `_nlmefit`, `_mcglmfit`, `_saemfit`. Build is `scikit-build-core` + CMake >=3.19; Eigen, Boost.Math and nanobind are fetched at build time if absent (clean builds need network or system copies).
- Editable install picks up pure-Python changes under `src/statcraft/` automatically; C++ headers or `bindings/` require a rebuild/reinstall (stale extensions fail with `ImportError` for `_glmfit` etc.).
- Build parallelism comes from the build tool: `CMAKE_BUILD_PARALLEL_LEVEL` (packaging default `-j2`) or `-Cbuild.tool-args=-j4`. `STATCRAFT_SEQUENTIAL_MODULES` defaults ON below ~6 GiB RAM. Keep `-DSTATCRAFT_NATIVE=ON` off for distributable wheels; OpenMP is optional (stock macOS Clang builds single-threaded).
- `pytest -q` runs the whole suite; focused runs: `pytest -q tests/test_model_signatures.py` or `pytest -q tests/test_smoke.py::test_glm_poisson_fit`. `tests/test_smoke.py` has many plotting/diagnostic assertions behind `pytest.importorskip("matplotlib")`, so without the `viz` extra they skip; the other three files pin callable signatures, constructor keywords, and compact reprs/top-level exports. Examples are de facto smoke scripts: `python examples/glm/poisson.py` and `python examples/saem/theo_covariates.py`.
- No lint/typecheck/formatter config exists; verification is pytest plus examples. CI: `tests.yml` builds a wheel per Python 3.11–3.14, installs it, then runs pytest; `wheels.yml` uses `cibuildwheel` and publishes on a `v*` tag via PyPI trusted publishing. Local `python -m build --wheel` (or `uv build --wheel`) only builds the current platform.
- Docs use `mkdocs serve` / `mkdocs build --strict` with `mkdocs` + `mkdocs-material` (undeclared); `site/` is generated output.

## Python API layout (trust code over README quick starts)

- `src/statcraft/__init__.py` eagerly exposes every public name from all engines plus the engine namespaces. `sf.glm`/`sf.nlme`/`sf.mcglm`/`sf.nls`/`sf.saem` remain **modules**, not estimators; families and shortcuts are top-level (`sf.Binomial`, `sf.poisson`, `sf.varIdent`, `sf.plot`, ...). Formula GLM entry points live in `import statcraft.glm as scg; scg.glm(...)`; the matrix class is `sf.GLM`. README's `nls(...)` quick start is a placeholder, not runnable code.
- The NLS estimator class is `sf.NLS`; the lowercase estimator is `from statcraft.nls import nls`, while `sf.nls` stays the subpackage.
- GLM/NLME/MCGLM/SAEM constructors are unfitted until `.fit()` (results live on the returned fit); `nls`/`NLS` return the fitted model itself. `repr(fit)`/`print(fit)` are intentionally compact (no tables); use `fit.summary()` for full tables.

## Shared callable-model convention (regression-tested)

- NLS/NLME/GNLS/SAEM Python callables share `def model(x, p)`: `x` = exogenous values from `predictors=`, `p` = parameters from `params=` (or key order of `start=` when omitted). `ModelView` supports unpacking, name/position indexing, `np.asarray` and slicing. With a single predictor/parameter, `x`/`p` is the value directly (`t, = x` and `a, = p` still work); argument names are conventional, but a single parameter's second argument must be `p`/`params`/`parameters`/`theta` to select the vector form. See `docs/model-functions.md`.
- SAEM compiles callables into a C++ expression; `np.asarray(x)` defeats tracing and falls back to the Python callback. Keep `x` direct/unpacked/indexed when compilation matters.
- Shared constructor keywords are `response=`, `predictors=`, `params=`. `y=`, `exog_var=` and `fixed=` (outside `LME`, whose `fixed=` is its full formula) were removed and raise `TypeError`; MCGLM uses `endog=`/`exog=`. Fast SAEM test fits: `.fit(n_iter=3, n_burn=1, se_method="none", ll_method="none", seed=1)`.

## Data, plots and artifacts

- `data_loader` resolves datasets independently of CWD: `STATCRAFT_DATA_DIR` first, then packaged `statcraft/data`, then repo `data/`, then CWD fallbacks. `data/datasets.json` is a generated catalogue refreshed by `list_datasets()`/`dataset_info()` or `python -m statcraft.datasets --refresh`. pyarrow is optional (`data` extra); pandas is the fallback.
- Plotting goes through `fit.plot()` / `plot(fit, kind=...)`; implementations are private and selected by `kind` aliases in `src/statcraft/plots/registry.py` (first alias is canonical; hyphens/spaces are normalized). `suggest_plots(fit)` and `available_kinds()` list valid names. matplotlib is optional (`viz`); `backend="jax"` for NLS needs `pip install jax` (undeclared extra).
- `build/`, `include/glmfit/.cr-build/` and `__pycache__/` are generated artifacts, not source. Version is declared only in `pyproject.toml` (CMake reads it at configure time; Python falls back to it). Docs and dev/release notes: `docs/`, `CONTRIBUTING.md`, `SUPPORT.md`, `CHANGELOG.md`.

## Modeling conventions that are easy to miss

- GLM family arguments go on `.fit()`: `cens="status"` (survival families; a string names a data column), `left_limit=`/`right_limit=` (`Truncated*`), `entry`/`Z`/`start`; two-part formulas look like `"count_part | second_part"`. `d` is internal, not a `.fit()` keyword. `GeneralizedGamma` takes `sigma_formula=`/`Q_formula=` on the constructor/shortcut (`generalized_gamma(...)` or `GLM.from_formula_gengamma(...)`), **not** `.fit()`. `start=` only affects IRLS/quasi fits (MLE families pick their own starts); default `standardize=True` rejects rank-deficient designs — fit with `standardize=False`.
- NLME: `NLME`/`GNLS` default to `method="ML"`, `LME`/`GLS` to REML; only `LME` validates `method` (otherwise anything but `"REML"` behaves as ML). For likelihood-ratio comparisons across different fixed effects, fit under ML (REML likelihoods are not comparable). `tTable(ddf="satterthwaite")` refits to REML when possible; `ddf="kr"` is unimplemented. `tTable`/`fitted` signatures differ across `LmeFit`/`NlmeFit`/`LmerFit`/`GlmerFit`.
- SAEM `family=` accepts family classes/objects, never strings. `fitted()`/standard `residuals()` may be unavailable for censored/non-Gaussian families. Use `fit.npde(...)`, `fit.residuals(type="npde"|"quantile", ...)` and the `npde`/`vpc` plot kinds instead.
