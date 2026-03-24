"""HDF5 inspection, conversion, and split helpers for NeRD training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .common import ensure_directory, safe_hdf5_attr_value


@dataclass(slots=True)
class DatasetSummary:
    """Programmatic summary of an HDF5 trajectory dataset."""

    path: Path
    format_name: str
    dataset_keys: list[str]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]
    num_trajectories: int
    horizon: int | None
    traj_length_min: int | None
    traj_length_max: int | None
    traj_length_mean: float | None
    total_transitions: int | None
    attrs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format_name": self.format_name,
            "dataset_keys": self.dataset_keys,
            "shapes": self.shapes,
            "dtypes": self.dtypes,
            "num_trajectories": self.num_trajectories,
            "horizon": self.horizon,
            "traj_length_min": self.traj_length_min,
            "traj_length_max": self.traj_length_max,
            "traj_length_mean": self.traj_length_mean,
            "total_transitions": self.total_transitions,
            "attrs": self.attrs,
        }


def inspect_hdf5(path: str | Path) -> DatasetSummary:
    """Inspect either the collector output schema or the upstream NeRD schema."""

    dataset_path = Path(path).expanduser().resolve()
    with h5py.File(dataset_path, "r") as file:
        if "data" in file and isinstance(file["data"], h5py.Group):
            group = file["data"]
            format_name = "nerd_trajectory_group"
            dataset_keys = sorted(list(group.keys()))
            shapes = {key: list(group[key].shape) for key in dataset_keys}
            dtypes = {key: str(group[key].dtype) for key in dataset_keys}
            states_shape = group["states"].shape
            horizon = int(states_shape[0])
            num_trajectories = int(states_shape[1])
            traj_lengths = group["traj_lengths"][:] if "traj_lengths" in group else np.full(num_trajectories, horizon)
            attrs = {key: file["data"].attrs[key] for key in file["data"].attrs.keys()}
        else:
            group = file
            format_name = "collector_trajectory_major"
            dataset_keys = sorted(list(group.keys()))
            shapes = {key: list(group[key].shape) for key in dataset_keys}
            dtypes = {key: str(group[key].dtype) for key in dataset_keys}
            states_shape = group["states"].shape
            num_trajectories = int(states_shape[0])
            horizon = int(states_shape[1])
            traj_lengths = group["traj_lengths"][:] if "traj_lengths" in group else np.full(num_trajectories, horizon)
            attrs = {key: file.attrs[key] for key in file.attrs.keys()}

    total_transitions = int(np.asarray(traj_lengths).sum()) if traj_lengths is not None else None
    return DatasetSummary(
        path=dataset_path,
        format_name=format_name,
        dataset_keys=dataset_keys,
        shapes=shapes,
        dtypes=dtypes,
        num_trajectories=num_trajectories,
        horizon=horizon,
        traj_length_min=int(np.min(traj_lengths)) if traj_lengths is not None else None,
        traj_length_max=int(np.max(traj_lengths)) if traj_lengths is not None else None,
        traj_length_mean=float(np.mean(traj_lengths)) if traj_lengths is not None else None,
        total_transitions=total_transitions,
        attrs={key: safe_hdf5_attr_value(val) for key, val in attrs.items()},
    )


def _copy_attrs(source_attrs: Any, target_attrs: Any) -> None:
    for key in source_attrs.keys():
        target_attrs[key] = safe_hdf5_attr_value(source_attrs[key])


def convert_collector_to_nerd(
    source_path: str | Path,
    output_path: str | Path,
    *,
    env_name: str = "PegInsertIsaacLab",
    chunk_trajectories: int = 32,
) -> dict[str, Any]:
    """Convert collector output `(B, T, ...)` into NeRD's expected `(T, B, ...)` dataset layout."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    ensure_directory(output)

    with h5py.File(source, "r") as src, h5py.File(output, "w") as dst:
        data_group = dst.create_group("data")
        _copy_attrs(src.attrs, data_group.attrs)
        data_group.attrs["env"] = env_name
        data_group.attrs["mode"] = "trajectory"
        data_group.attrs["source_path"] = str(source)
        data_group.attrs["source_format"] = "collector_trajectory_major"

        num_trajectories, horizon = src["states"].shape[:2]
        traj_lengths = src["traj_lengths"][:]
        total_transitions = int(traj_lengths.sum())
        data_group.attrs["total_trajectories"] = int(num_trajectories)
        data_group.attrs["total_transitions"] = total_transitions
        data_group.attrs["state_dim"] = int(src["states"].shape[-1])
        data_group.attrs["next_state_dim"] = int(src["next_states"].shape[-1])
        data_group.attrs["joint_act_dim"] = int(src["joint_acts"].shape[-1])
        if "contact_depths" in src:
            data_group.attrs["num_contacts_per_env"] = int(src["contact_depths"].shape[-1])
        data_group.attrs["has_net_contact_force"] = "net_contact_force" in src
        data_group.attrs["converted_shape_convention"] = "data[name] has shape [T, B, ...]"

        for key in sorted(src.keys()):
            dataset = src[key]
            if dataset.ndim >= 2 and dataset.shape[0] == num_trajectories and dataset.shape[1] == horizon:
                out_shape = (horizon, num_trajectories, *dataset.shape[2:])
                out_chunks = (horizon, min(num_trajectories, chunk_trajectories), *dataset.shape[2:])
                out_dataset = data_group.create_dataset(
                    key,
                    shape=out_shape,
                    dtype=dataset.dtype,
                    chunks=out_chunks,
                )
                for start in range(0, num_trajectories, chunk_trajectories):
                    stop = min(num_trajectories, start + chunk_trajectories)
                    chunk = dataset[start:stop, ...]
                    out_dataset[:, start:stop, ...] = np.swapaxes(chunk, 0, 1)
            elif dataset.ndim >= 1 and dataset.shape[0] == num_trajectories:
                out_dataset = data_group.create_dataset(
                    key,
                    data=dataset[...],
                    dtype=dataset.dtype,
                )
            else:
                raise ValueError(
                    f"Unsupported dataset shape for conversion: key='{key}', shape={dataset.shape}"
                )

    return {
        "source_path": str(source),
        "output_path": str(output),
        "num_trajectories": int(num_trajectories),
        "horizon": int(horizon),
        "total_transitions": total_transitions,
    }


def split_trajectory_indices(
    dataset_path: str | Path,
    train_indices_path: str | Path,
    test_indices_path: str | Path,
    summary_path: str | Path,
    *,
    train_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Create deterministic train/test trajectory index splits without copying the dataset."""

    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    dataset = Path(dataset_path).expanduser().resolve()
    train_indices_path = Path(train_indices_path).expanduser().resolve()
    test_indices_path = Path(test_indices_path).expanduser().resolve()
    summary_path = Path(summary_path).expanduser().resolve()
    ensure_directory(train_indices_path)
    ensure_directory(test_indices_path)
    ensure_directory(summary_path)

    with h5py.File(dataset, "r") as file:
        data_group = file["data"]
        num_trajectories = int(data_group["states"].shape[1])
        traj_lengths = data_group["traj_lengths"][:]

    rng = np.random.default_rng(seed=seed)
    all_indices = np.arange(num_trajectories, dtype=np.int32)
    shuffled = rng.permutation(all_indices)

    train_count = int(num_trajectories * train_ratio)
    test_count = num_trajectories - train_count
    train_indices = np.sort(shuffled[:train_count]).astype(np.int32)
    test_indices = np.sort(shuffled[train_count:]).astype(np.int32)

    np.save(train_indices_path, train_indices)
    np.save(test_indices_path, test_indices)

    split_summary = {
        "dataset_path": str(dataset),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "num_trajectories": int(num_trajectories),
        "train_trajectories": int(train_count),
        "test_trajectories": int(test_count),
        "train_total_transitions": int(traj_lengths[train_indices].sum()),
        "test_total_transitions": int(traj_lengths[test_indices].sum()),
        "train_indices_path": str(train_indices_path),
        "test_indices_path": str(test_indices_path),
    }

    from .common import save_json

    save_json(summary_path, split_summary)
    return split_summary

