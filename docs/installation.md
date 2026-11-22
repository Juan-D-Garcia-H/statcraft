# Installation

## Supported environments

Wheels are published for Python **3.11–3.14** on Linux (x86_64, aarch64), macOS
(universal2) and Windows (AMD64). Building from a source distribution requires
Python 3.11+, CMake 3.19+ and a C++23-capable compiler; see `SUPPORT.md` in the
repository for details.

## Install from PyPI

```bash
python -m pip install statcraft
```

### Optional extras

| Extra | Installs | Use it when |
|---|---|---|
| `viz` | `matplotlib>=3.6` | You want `fit.plot()` or any diagnostic figure |
| `data` | `pyarrow>=10` | You want the fast CSV engine or Parquet inputs |
| `test` | `pytest>=7` | You want to run the bundled smoke suite |

```bash
python -m pip install "statcraft[viz,data]"
```

`pyarrow` is optional: without it, `data_loader` falls back to pandas for CSV
files with a log message.

## Install from source

Eigen, Boost.Math and nanobind are fetched automatically when they are not
already available, so a source build needs network access on a clean machine.

```bash
git clone https://github.com/Juan-D-Garcia-H/statcraft.git
cd statcraft
python -m pip install -e ".[viz,data,test]"
```

Four nanobind extensions are compiled: `_glmfit`, `_nlmefit`, `_mcglmfit` and
`_saemfit`. Pure-Python changes under `src/statcraft/` are picked up by an
editable install; changes to C++ headers or `bindings/` require a rebuild.

### Build parallelism and memory

The build tool controls parallelism, not CMake itself:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=4 python -m pip install -e .
# or
python -m pip install -e . -Cbuild.tool-args=-j4
```

The default in the packaging configuration is `-j2`. On machines with less than
roughly 6 GiB of RAM, CMake turns on `STATCRAFT_SEQUENTIAL_MODULES` so the four
heavy modules compile one at a time. Override it explicitly when you know your
environment can handle parallel module builds.

OpenMP is detected automatically. If it is unavailable (for example with stock
AppleClang on macOS), the engines build single-threaded instead of failing.
Leave `-DSTATCRAFT_NATIVE=ON` off for distributable wheels: it emits CPU-specific
instructions.

## Verify the installation

```python
import statcraft as sf
import statcraft.glm as scg
import statcraft.mcglm
import statcraft.nlme
import statcraft.nls
import statcraft.saem
from statcraft.datasets import data_loader

print(sf.__version__)
print(data_loader("Orthodont.csv").load().shape)
```

## Building the user guide locally

The manual sources are plain Markdown in `docs/`, configured by `mkdocs.yml`:

```bash
python -m pip install mkdocs mkdocs-material
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # verify all pages and links
```
