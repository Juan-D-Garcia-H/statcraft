# Datasets and data loading

Everything related to datasets lives in one module:
[`statcraft.datasets`][statcraft.datasets]. You can list files, inspect their
metadata and load one without importing from another module.

```python
from statcraft.datasets import data_loader, list_datasets, dataset_info

orth = data_loader("Orthodont.csv").load()
names = list_datasets()
info = dataset_info()
```

The same three helpers are available as top-level shortcuts:

```python
import statcraft

statcraft.data_loader("theo").load()
statcraft.list_datasets()
statcraft.dataset_info()
```

Old `from statcraft.utils import data_loader` imports still work as a
compatibility alias, but `statcraft.datasets` is the canonical location.

## Loading a file

`data_loader(...)` is an alias of `DatasetLoader`; it creates a loader. Call
`.load()` to get a `pandas.DataFrame`:

```python
from statcraft.datasets import data_loader

orth = data_loader("Orthodont.csv").load()
print(orth.head())
```

The extension may be omitted. If the exact name does not exist, the loader
tries all extensions with the same stem first:

```python
soya = data_loader("soya").load()
```

Nested names from the catalogue are accepted too:

```python
frame = data_loader("subdir/measurements.csv").load()
```

If the file cannot be found, the error includes close matches and a preview of
available datasets.

### Chunked CSV and context manager

`chunksize=` applies to CSV files and returns a pandas `TextFileReader` instead
of a `DataFrame`. Use the loader as a context manager so the reader is closed:

```python
with data_loader("large_table.csv", chunksize=50_000) as reader:
    total = 0
    for chunk in reader:
        total += len(chunk)
print(total)
```

## Where datasets are searched

Resolution does **not** depend on the current working directory. The order is:

1. `STATCRAFT_DATA_DIR` — explicit process-wide override.
2. `statcraft/data` — datasets packaged inside an installed wheel.
3. The repository `data/` directory, located relative to `src/statcraft` in a
   source checkout or editable install.
4. `statcraft/data` or `data` relative to the current directory, as a legacy
   fallback.

If none of these directories exists, `NotADirectoryError` is raised with the
path that was tried.

### Override for one call

```python
from pathlib import Path

data = data_loader(
    "theo",
    data_dir_override=str(Path.home() / "datasets"),
).load()
```

### Override for a whole process

```bash
export STATCRAFT_DATA_DIR=/path/to/my/datasets
```

```python
fit_data = data_loader("my_table.csv").load()
```

The runnable examples pass their own checkout directory explicitly so they
always use the files shipped with the repository.

## File formats

| Extension | Reader | Extra dependency |
|---|---|---|
| `.csv` | `pd.read_csv` with pyarrow when available, otherwise the pandas C engine | optional: `statcraft[data]` |
| `.parquet`, `.pq` | `pd.read_parquet` | pyarrow (`statcraft[data]`) |
| `.feather`, `.ftr` | `pd.read_feather` | pyarrow (`statcraft[data]`) |
| `.xlsx`, `.xls` | `pd.read_excel` | pandas Excel engine (`openpyxl` / `xlrd`) |
| `.json` | `pd.read_json` | pandas only |

CSV and JSON also accept gzip (`.gz`, `.tgz`) and zip (`.zip`) compression:

```python
frame = data_loader("measurements.csv.gz").load()
```

Install the optional data dependency for Parquet, Feather and faster CSV
parsing:

```bash
python -m pip install "statcraft[data]"
```

Engines themselves take a `pandas.DataFrame`; the loader is a convenience for
bundled or external files.

## Dataset catalogue

The catalogue is generated from the data directory. It records each dataset's
relative name, file name, format, row/column counts, size in bytes and
modification time.

```python
from statcraft.datasets import (
    available_datasets,
    dataset_info,
    dataset_registry_path,
    list_datasets,
)

list_datasets()                  # catalogue names, e.g. "Orthodont.csv"
available_datasets()             # names accepted by data_loader(...)
dataset_info().head()            # DataFrame with metadata
dataset_registry_path()          # Path to datasets.json
```

`list_datasets()` mirrors the catalogue and keeps the `.csv` suffix, matching
`dataset_info()["name"]`. Pass `strip_ext=True` to get the shorter,
loader-friendly spelling; `available_datasets()` already defaults to that form.

`dataset_info()` returns these columns:

| Column | Meaning |
|---|---|
| `name` | Relative POSIX path accepted by `data_loader()` |
| `file` | File name |
| `format` | Format suffix, including the inner suffix for compressed files |
| `rows` | Number of rows, or `NaN` when the file could not be inspected |
| `columns` | Number of columns, or `NaN` when the file could not be inspected |
| `size_bytes` | File size |
| `modified` | ISO modification timestamp |

All catalogue functions accept `data_dir=` and `refresh=`:

```python
list_datasets(data_dir="/path/to/my/datasets", refresh=True)
dataset_info(data_dir="/path/to/my/datasets")
```

### Generated registry

`data/datasets.json` (or `statcraft/data/datasets.json` in a wheel) is
generated, not meant to be edited by hand. The constant `REGISTRY_FILENAME`
holds its file name. The JSON stores the directory fingerprint, dataset count
and records. A fingerprint change — added, removed or modified file — makes the
next catalogue call regenerate it automatically. On a read-only installation
the fresh records are still returned without rewriting the file.

Force a rescan explicitly with:

```python
from statcraft.datasets import refresh_dataset_registry

refresh_dataset_registry()                 # rescan and write datasets.json
refresh_dataset_registry(write=False)       # return records without writing
```

Or from the command line:

```bash
python -m statcraft.datasets
python -m statcraft.datasets --refresh
python -m statcraft.datasets --data-dir /path/to/my/datasets
```

The CLI prints the full metadata table and the registry path.

## Errors

| Exception | Typical cause |
|---|---|
| `NotADirectoryError` | The resolved or overridden data directory does not exist. |
| `FileNotFoundError` | No exact filename, extension variant or close match exists. |
| `ValueError` | The file extension is not a supported dataset format. |
| `IOError` | The file exists but pandas cannot parse it. |
| `MemoryError` | The file is too large for available RAM. |

## See also

- [Installation](installation.md) — optional extras, including `data`.
- [Troubleshooting](troubleshooting.md) — data-directory and Parquet errors.
- [GLM](glm.md), [NLME](nlme.md), [NLS](nls.md), [MCGLM](mcglm.md) and
  [SAEM](saem.md) — engine manuals using bundled example datasets.
