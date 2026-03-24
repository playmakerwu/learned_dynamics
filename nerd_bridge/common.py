"""Common helpers for local NeRD integration without modifying existing code paths."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_NERD_ROOT = PROJECT_ROOT / "external" / "neural-robot-dynamics"
USER_DECLARED_BASE_DATASET = Path("/recording/base.hdfs")
FALLBACK_BASE_DATASET = PROJECT_ROOT / "recordings" / "base.hdf5"
DEFAULT_CONVERTED_DATASET = PROJECT_ROOT / "recordings" / "nerd_base_converted.hdf5"
DEFAULT_TRAIN_INDICES = PROJECT_ROOT / "recordings" / "nerd_base_train_indices.npy"
DEFAULT_TEST_INDICES = PROJECT_ROOT / "recordings" / "nerd_base_test_indices.npy"
DEFAULT_SPLIT_SUMMARY = PROJECT_ROOT / "recordings" / "nerd_base_split_summary.json"
DEFAULT_INSPECTION_SUMMARY = PROJECT_ROOT / "recordings" / "base_hdf5_summary.json"


def ensure_directory(path: Path) -> None:
    """Create the parent directory for a file, or the directory itself."""

    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    """Convert nested objects into JSON-serializable values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def save_json(path: Path, payload: Any) -> None:
    """Write a JSON file with stable formatting."""

    ensure_directory(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(jsonable(payload), file, indent=2, sort_keys=True)


def discover_isaac_warp_root(python_executable: str | Path | None = None) -> Path | None:
    """Find Isaac Sim's bundled Warp package that still exposes `warp.sim`."""

    executable = Path(python_executable or sys.executable).resolve()
    env_root = executable.parents[1]
    site_packages = env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    extcache_root = site_packages / "isaacsim" / "extscache"
    if not extcache_root.exists():
        return None

    for candidate in sorted(extcache_root.glob("omni.warp.core-*"), reverse=True):
        if (candidate / "warp" / "sim").exists():
            return candidate
    return None


def prepend_python_path(path: Path) -> None:
    """Add a path to the front of `sys.path` if it is not already present."""

    as_str = str(path)
    if as_str not in sys.path:
        sys.path.insert(0, as_str)


def configure_nerd_imports() -> dict[str, str | None]:
    """Expose the upstream NeRD repo and Isaac's Warp-Sim build to Python imports."""

    warp_root = discover_isaac_warp_root()
    if warp_root is not None:
        prepend_python_path(warp_root)
    if UPSTREAM_NERD_ROOT.exists():
        prepend_python_path(UPSTREAM_NERD_ROOT)
    return {
        "project_root": str(PROJECT_ROOT),
        "upstream_nerd_root": str(UPSTREAM_NERD_ROOT) if UPSTREAM_NERD_ROOT.exists() else None,
        "isaac_warp_root": str(warp_root) if warp_root is not None else None,
    }


def resolve_source_dataset(preferred_path: str | Path | None = None) -> tuple[Path, list[str]]:
    """Resolve the intended base trajectory file while making fallbacks explicit."""

    notes: list[str] = []
    candidates: list[Path] = []
    resolved_preferred: Path | None = None

    if preferred_path is not None:
        resolved_preferred = Path(preferred_path).expanduser()
        candidates.append(resolved_preferred)
    candidates.append(USER_DECLARED_BASE_DATASET)
    candidates.append(FALLBACK_BASE_DATASET)

    seen: set[Path] = set()
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if candidate.is_file():
            if resolved_preferred is not None:
                preferred_candidate = (
                    resolved_preferred.resolve()
                    if resolved_preferred.is_absolute()
                    else (PROJECT_ROOT / resolved_preferred).resolve()
                )
                if candidate != preferred_candidate:
                    notes.append(
                        f"Requested input path '{preferred_candidate}' was not found. "
                        f"Falling back to '{candidate}'."
                    )
            elif candidate != USER_DECLARED_BASE_DATASET.resolve() and not USER_DECLARED_BASE_DATASET.exists():
                notes.append(
                    "Inferred the source dataset path from the local workspace because "
                    f"'{USER_DECLARED_BASE_DATASET}' does not exist here. "
                    f"Using '{candidate}'."
                )
            return candidate, notes

    checked_msg = "\n".join(f" - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find the requested base dataset file. Checked:\n"
        f"{checked_msg}"
    )


def default_device() -> str:
    """Pick a reasonable device string without importing project-specific code."""

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def safe_hdf5_attr_value(value: Any) -> Any:
    """Convert metadata into an HDF5-safe scalar or string."""

    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(jsonable(value))
    return value


def format_count(value: int) -> str:
    """Pretty-print a count with separators."""

    return f"{int(value):,}"


def env_python() -> Path:
    """Return the Python executable that should be used for new helper scripts."""

    return Path(sys.executable).resolve()
