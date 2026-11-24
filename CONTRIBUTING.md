# Contributing

Thanks for helping improve statcraft.

## Development setup

Python 3.11 or newer and a C++23-capable compiler are required. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[viz,data,test]"
```

The editable install builds the four nanobind extensions with
`scikit-build-core`. Eigen, Boost.Math and nanobind are downloaded
automatically if they are not available on the system.

## Before opening a pull request

- Keep public docstrings, comments and user-facing messages in English.
- If you change C++ headers or `bindings/`, rebuild/reinstall; pure-Python
  changes under `src/statcraft/` are picked up by the editable install.
- Run the smoke suite:

  ```bash
  pytest -q
  ```

- Run the de facto smoke examples from the repository root:

  ```bash
  python examples/glm/poisson.py
  python examples/saem/theo_covariates.py
  ```

- Leave `-DSTATCRAFT_NATIVE=ON` off for distributable wheels. Build parallelism
  comes from the build tool: set `CMAKE_BUILD_PARALLEL_LEVEL=4` or pass
  `-Cbuild.tool-args=-j4`. On machines with less than roughly 6 GiB of RAM,
  `STATCRAFT_SEQUENTIAL_MODULES` defaults to ON to cap peak memory; override it
  explicitly if you know your environment can handle parallel module builds.

## Tests and CI

`tests/test_smoke.py` checks imports, packaged data loading and one small fit
for GLM and NLS. CI builds a wheel for each supported Python version and runs
that suite; `cibuildwheel` repeats it on Linux, macOS and Windows.

## Releases

The version lives only in `pyproject.toml`; CMake reads it at configure time.
To publish, push a `vX.Y.Z` tag after updating `CHANGELOG.md`. The
`wheels.yml` workflow builds all artifacts and publishes them through PyPI
trusted publishing.
