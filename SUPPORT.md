# Support and compatibility

## Supported platforms

Wheels are built and tested in CI for:

- **Python**: 3.11, 3.12, 3.13 and 3.14.
- **Linux**: x86_64 and aarch64 (`manylinux_2_28`).
- **macOS**: universal2 (Intel and Apple Silicon).
- **Windows**: AMD64.

Other platforms may work when building from the source distribution, but they
are not covered by the published wheels.

## Compiler requirements

The core is C++23. A recent C++23-capable toolchain is required when building
from source. CI uses the toolchains shipped in the `manylinux_2_28` images on
Linux, AppleClang on macOS and MSVC on Windows. If your compiler defaults to an
older standard, configure it to use C++23 before building.

CMake 3.19 or newer is required; `scikit-build-core` installs a suitable CMake
and Ninja in isolated builds when they are not present.

## Python API compatibility

This project is at `0.x`: public APIs may change between minor releases. Within
a patch release, bug fixes will not intentionally break existing code. When a
breaking change is needed, it will be called out in `CHANGELOG.md`.

## Dependencies

Runtime dependencies are declared in `pyproject.toml`. Optional extras:

- `viz`: matplotlib for plotting helpers.
- `data`: pyarrow for the fast CSV engine and Parquet inputs; `data_loader`
  falls back to pandas when it is not installed.
- `test`: pytest for the smoke suite.

## Getting help

Open an issue in the project repository with the platform, Python version,
statcraft version and a minimal reproducer.
