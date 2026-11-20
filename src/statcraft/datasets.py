# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

"""Dataset discovery and the generated catalogue for statcraft.

This module owns everything related to datasets: where they are found, the
generated ``data/datasets.json`` index, and the loader used to read one file.
:mod:`statcraft.utils` keeps a compatibility alias for ``data_loader``.

    from statcraft import datasets

    datasets.list_datasets()
    datasets.dataset_info()
    datasets.refresh_dataset_registry()

Command line equivalent:

    python -m statcraft.datasets --refresh
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Union

import pandas as pd

logger = logging.getLogger(__name__)

#: Generated catalogue of bundled datasets.  It lives next to the CSV files so
#: it is included in the wheel by the ``data`` force-include in pyproject.toml.
REGISTRY_FILENAME = "datasets.json"

_DATASET_SUFFIXES = {
    ".csv", ".parquet", ".pq", ".feather", ".ftr", ".xlsx", ".xls", ".json",
}
_COMPRESSED_SUFFIXES = {".gz", ".tgz", ".zip"}
_REGISTRY_COLUMNS = ["name", "file", "format", "rows", "columns", "size_bytes", "modified"]
_REGISTRY_CACHE: dict[tuple[str, str], list[dict]] = {}
_REGISTRY_UNWRITABLE: set[str] = set()

__all__ = [
    "REGISTRY_FILENAME",
    "DatasetLoader",
    "data_loader",
    "available_datasets",
    "list_datasets",
    "dataset_info",
    "dataset_registry_path",
    "refresh_dataset_registry",
]


def _default_data_dir() -> Path:
    """Resolve the dataset directory without depending on the current CWD.

    Search order:

    1. ``STATCRAFT_DATA_DIR`` (explicit user override).
    2. ``statcraft/data`` inside an installed wheel.
    3. The repository ``data/`` directory next to ``src/statcraft`` in a
       source checkout or editable install.
    4. ``statcraft/data`` / ``data`` relative to the CWD (legacy fallback).
    """
    env_dir = os.environ.get("STATCRAFT_DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    module_path = Path(__file__).resolve()
    # In a wheel this file is ``statcraft/datasets.py`` -> ``statcraft/data``.
    packaged = module_path.parents[0] / "data"
    # In a checkout it is ``src/statcraft/datasets.py`` -> repo-root ``data``.
    checkout = (
        module_path.parents[2] / "data"
        if len(module_path.parents) > 2
        else packaged
    )

    candidates = (packaged, checkout, Path("statcraft/data"), Path("data"))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved

    # Return the packaged location so the resulting error message is stable.
    return packaged


def _resolve_data_dir(data_dir: str | os.PathLike | None = None) -> Path:
    return Path(data_dir).resolve() if data_dir is not None else _default_data_dir()


def _dataset_suffixes(path: Path) -> list[str]:
    return [suffix.lower() for suffix in path.suffixes]


def _is_dataset_file(path: Path) -> bool:
    suffixes = _dataset_suffixes(path)
    if not suffixes:
        return False
    if suffixes[-1] in _COMPRESSED_SUFFIXES:
        return len(suffixes) > 1 and suffixes[-2] in _DATASET_SUFFIXES
    return suffixes[-1] in _DATASET_SUFFIXES


def _compression_for(path: Path) -> str | None:
    suffixes = _dataset_suffixes(path)
    if not suffixes:
        return None
    if suffixes[-1] == ".zip":
        return "zip"
    if suffixes[-1] in {".gz", ".tgz"}:
        return "gzip"
    return None


def _format_for(path: Path) -> str:
    suffixes = _dataset_suffixes(path)
    if _compression_for(path) and len(suffixes) > 1:
        return suffixes[-2]
    return suffixes[-1] if suffixes else ""


def _iter_dataset_paths(data_dir: Path):
    """Yield dataset files in stable order, ignoring the generated registry."""
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name == REGISTRY_FILENAME:
            continue
        if path.name.startswith((".", "_")) or not _is_dataset_file(path):
            continue
        yield path


def _fingerprint(paths: list[Path], data_dir: Path) -> str:
    payload = []
    for path in paths:
        try:
            stat = path.stat()
            payload.append(
                f"{path.relative_to(data_dir).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}"
            )
        except OSError:
            payload.append(path.relative_to(data_dir).as_posix())
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _read_shape(path: Path) -> tuple[int | None, int | None]:
    suffix = _format_for(path)
    compression = _compression_for(path)
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path, compression=compression, engine="c")
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".feather", ".ftr"}:
            frame = pd.read_feather(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif suffix == ".json":
            frame = pd.read_json(path, compression=compression)
        else:  # pragma: no cover - guarded by _is_dataset_file
            return None, None
        return len(frame.index), len(frame.columns)
    except Exception as exc:  # optional engines, malformed files, ...
        logger.debug("Could not inspect dataset %s: %s", path.name, exc)
        return None, None


def _dataset_record(path: Path, data_dir: Path) -> dict:
    stat = path.stat()
    rows, columns = _read_shape(path)
    return {
        "name": path.relative_to(data_dir).as_posix(),
        "file": path.name,
        "format": _format_for(path),
        "rows": rows,
        "columns": columns,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _registry_path(data_dir: Path) -> Path:
    return data_dir / REGISTRY_FILENAME


def _read_registry(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def refresh_dataset_registry(
    data_dir: str | os.PathLike | None = None, *, write: bool = True
) -> list[dict]:
    """Rescan ``data_dir`` and (by default) rewrite ``datasets.json``.

    The file is written atomically. A read-only installation is supported:
    when writing fails the freshly scanned records are still returned.
    """
    base = _resolve_data_dir(data_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"Invalid data directory: {base}")

    paths = list(_iter_dataset_paths(base))
    fingerprint = _fingerprint(paths, base)
    records = [_dataset_record(path, base) for path in paths]

    if write:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": base.name,
            "fingerprint": fingerprint,
            "dataset_count": len(records),
            "datasets": records,
        }
        registry_path = _registry_path(base)
        temporary = registry_path.with_name(registry_path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(registry_path)
        except OSError as exc:
            logger.warning("Could not update dataset registry %s: %s", registry_path, exc)
            _REGISTRY_UNWRITABLE.add(str(base))
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    _REGISTRY_CACHE[(str(base), fingerprint)] = records
    return records


def _current_dataset_records(
    data_dir: str | os.PathLike | None = None, *, refresh: bool = False
) -> list[dict]:
    base = _resolve_data_dir(data_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"Invalid data directory: {base}")
    if refresh:
        return refresh_dataset_registry(base)

    paths = list(_iter_dataset_paths(base))
    fingerprint = _fingerprint(paths, base)
    key = (str(base), fingerprint)

    # Always check the file itself: a directory can return to a fingerprint
    # that is still cached while the on-disk registry was written for a
    # different intermediate state.
    if str(base) in _REGISTRY_UNWRITABLE:
        records = [_dataset_record(path, base) for path in paths]
        _REGISTRY_CACHE[key] = records
        return records

    registry = _read_registry(_registry_path(base))
    if registry is not None and registry.get("fingerprint") == fingerprint:
        if key in _REGISTRY_CACHE:
            return _REGISTRY_CACHE[key]
        records = registry.get("datasets")
        if isinstance(records, list) and all(
            isinstance(item, dict) and "name" in item for item in records
        ):
            _REGISTRY_CACHE[key] = records
            return records

    # New/removed/edited file: regenerate so the registry never goes stale.
    return refresh_dataset_registry(base)


def available_datasets(
    data_dir: str | os.PathLike | None = None,
    *,
    refresh: bool = False,
    strip_ext: bool = True,
) -> list[str]:
    """Names accepted by :func:`data_loader`.

    Parameters
    ----------
    data_dir : str or PathLike, optional
        Custom directory containing datasets.
    refresh : bool, default False
        Force regeneration of ``datasets.json``.
    strip_ext : bool, default True
        Remove the ``.csv`` extension from the returned names for cleaner display.
        Set to ``False`` to inspect the raw filenames.
    """
    records = _current_dataset_records(data_dir, refresh=refresh)
    names = []
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("name"), str):
            name = record["name"]
            if strip_ext and name.lower().endswith(".csv"):
                name = name[:-4]
            names.append(name)
    return names


def list_datasets(
    data_dir: str | os.PathLike | None = None,
    *,
    refresh: bool = False,
    strip_ext: bool = True,
) -> list[str]:
    """Friendly alias of :func:`available_datasets`."""
    return available_datasets(data_dir, refresh=refresh, strip_ext=strip_ext)


def dataset_info(
    data_dir: str | os.PathLike | None = None, *, refresh: bool = False
) -> pd.DataFrame:
    """DataFrame with name, format, shape, size and modification time."""
    records = _current_dataset_records(data_dir, refresh=refresh)
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=_REGISTRY_COLUMNS)
    return frame.reindex(columns=_REGISTRY_COLUMNS)


def dataset_registry_path(data_dir: str | os.PathLike | None = None) -> Path:
    """Path of the generated catalogue for the currently resolved data dir."""
    return _registry_path(_resolve_data_dir(data_dir))


class DatasetLoader:
    """Dual-mode dataset loader: direct ``.load()`` or context manager.

    This is the canonical loader for bundled or external data files. The
    catalogue helpers in this module tell you which names are available.
    """

    def __init__(self, filename: str, data_dir_override: str | None = None, chunksize: int | None = None):
        self.filename = filename
        self.chunksize = chunksize
        self.start_time = None
        self._reader = None

        self.base_dir = Path(data_dir_override).resolve() if data_dir_override else _default_data_dir()

        if not self.base_dir.is_dir():
            logger.critical(f"Base directory does not exist or is inaccessible: {self.base_dir}")
            raise NotADirectoryError(f"Invalid data directory: {self.base_dir}")

        self.file_path = (self.base_dir / filename).resolve()
        if not str(self.file_path).startswith(str(self.base_dir) + os.sep) and self.file_path != self.base_dir:
            try:
                if not self.file_path.is_relative_to(self.base_dir):
                    logger.warning(f"Unauthorized access attempt outside the base directory: {filename}")
                    raise PermissionError("The requested file is outside the permitted directory.")
            except Exception:
                pass

        if not self.file_path.exists():
            logger.info(f"Exact file not found: '{filename}'. Searching for alternatives...")
            safe_name = self.file_path.name
            possible_files = sorted(list(self.base_dir.glob(f"{Path(safe_name).stem}.*")))
            if not possible_files:
                possible_files = sorted(list(self.base_dir.rglob(safe_name)))
            if possible_files:
                self.file_path = possible_files[0]
                logger.info(f"Alternative variant automatically found: '{self.file_path.name}'")
            else:
                known = available_datasets(self.base_dir)
                suggestions = difflib.get_close_matches(
                    Path(filename).name, known, n=5, cutoff=0.45
                )
                error_msg = f"Could not find '{filename}' or any extension variants in: {self.base_dir}"
                if suggestions:
                    error_msg += f" Did you mean: {', '.join(suggestions)}?"
                elif known:
                    preview = ", ".join(known[:8])
                    error_msg += (
                        f" Available datasets include: {preview}"
                        f"{' ...' if len(known) > 8 else ''}. "
                        "Use statcraft.datasets.list_datasets() for the full list."
                    )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

        self.suffix = self.file_path.suffix.lower()
        self.compression = None

        if self.suffix == ".zip":
            self.compression = "zip"
            inner_suffixes = [s.lower() for s in self.file_path.suffixes]
            self.suffix = inner_suffixes[-2] if len(inner_suffixes) > 1 else ".csv"
        elif self.suffix in [".gz", ".tgz"] or (
            len(self.file_path.suffixes) > 1 and self.file_path.suffixes[-1].lower() == ".gz"
        ):
            self.compression = "gzip"
            inner_suffixes = [s.lower() for s in self.file_path.suffixes]
            self.suffix = inner_suffixes[-2] if len(inner_suffixes) > 1 else ".csv"

    def _read_csv_safely(self, path: Path) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        kwargs = {"chunksize": self.chunksize, "compression": self.compression}
        try:
            return pd.read_csv(path, engine="pyarrow", **kwargs)
        except Exception:
            try:
                logger.warning("PyArrow engine is not installed or failed. Falling back to the standard pandas parsing engine.")
            except Exception:
                pass
            return pd.read_csv(path, engine="c", **kwargs)

    def load(self) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        self.start_time = time.perf_counter()

        loaders: Dict[str, Callable[[Path], Any]] = {
            ".csv": self._read_csv_safely,
            ".parquet": lambda p: pd.read_parquet(p),
            ".pq": lambda p: pd.read_parquet(p),
            ".feather": lambda p: pd.read_feather(p),
            ".ftr": lambda p: pd.read_feather(p),
            ".xlsx": lambda p: pd.read_excel(p),
            ".xls": lambda p: pd.read_excel(p),
            ".json": lambda p: pd.read_json(p, compression=self.compression),
        }

        if self.suffix not in loaders:
            error_msg = f"Unsupported file format '{self.suffix}' for file: {self.file_path.name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.info(f"Starting file execution for: {self.file_path.name} (Format: {self.suffix})")
            data_source = loaders[self.suffix](self.file_path)

            if isinstance(data_source, pd.io.parsers.TextFileReader):
                self._reader = data_source
            else:
                try:
                    mem_usage_mb = data_source.memory_usage(deep=True).sum() / (1024 ** 2)
                    logger.info(f"Dataset loaded completely into RAM. Memory usage: {mem_usage_mb:.2f} MB")
                except Exception:
                    pass

            return data_source
        except MemoryError as me:
            logger.critical(f"Out of memory error while loading {self.file_path.name}: {me}")
            raise MemoryError("The file is too large for the available system memory.") from me
        except Exception as e:
            logger.error(f"Critical error processing the internal structure of {self.file_path.name}: {e}")
            raise IOError(f"Could not parse the file '{self.file_path.name}'. Verify it is not corrupted.") from e

    def __enter__(self) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if getattr(self, "_reader", None) is not None:
            try:
                self._reader.close()
                logger.info(f"Successfully closed file stream for: {self.file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to close file reader smoothly: {e}")
        if getattr(self, "start_time", None) is not None:
            try:
                execution_time = time.perf_counter() - self.start_time
                logger.info(f"Context session finished for '{self.file_path.name}' after {execution_time:.4f} seconds.")
            except Exception:
                pass
        return False


# Backwards-compatible short name, now defined next to the catalogue.
data_loader = DatasetLoader


def _main() -> None:  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description="List the datasets bundled with statcraft.")
    parser.add_argument(
        "--refresh", action="store_true", help="rescan the data directory before listing"
    )
    parser.add_argument("--data-dir", default=None, help="override the dataset directory")
    args = parser.parse_args()

    table = dataset_info(args.data_dir, refresh=args.refresh)
    if table.empty:
        print(f"No datasets found in {_resolve_data_dir(args.data_dir)}")
        return
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(table.to_string(index=False))
    print(f"\n{len(table)} datasets. Registry: {dataset_registry_path(args.data_dir)}")


if __name__ == "__main__":  # pragma: no cover
    _main()
