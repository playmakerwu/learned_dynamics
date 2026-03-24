"""Trajectory-major HDF5 writer utilities for NeRD datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def _to_hdf5_attr_value(value: Any) -> Any:
    """Convert metadata into an HDF5-attribute-safe scalar or string."""

    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bytes, bool, int, float, np.bool_, np.integer, np.floating)):
        return value
    return str(value)


def _to_numpy(value: torch.Tensor | np.ndarray | Any) -> np.ndarray:
    """Detach torch tensors and convert them to NumPy arrays."""

    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TrajectoryHDF5Writer:
    """Write fixed-horizon trajectories into a trajectory-major HDF5 file."""

    def __init__(
        self,
        *,
        path: Path,
        num_trajectories: int,
        horizon: int,
        field_specs: dict[str, tuple[tuple[int, ...], np.dtype | str]],
        metadata: dict[str, Any],
        hdf5_cfg: Any,
    ) -> None:
        self.path = Path(path)
        self.num_trajectories = int(num_trajectories)
        self.horizon = int(horizon)
        self.field_specs = field_specs
        self._write_index = 0
        self._total_transitions = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._datasets: dict[str, h5py.Dataset] = {}

        compression = getattr(hdf5_cfg, "compression", None)
        compression_level = getattr(hdf5_cfg, "compression_level", None)
        self._flush_every = max(1, int(getattr(hdf5_cfg, "flush_every_trajectories", 1)))
        self._chunk_trajectories = max(1, int(getattr(hdf5_cfg, "chunk_trajectories", 1)))

        for name, (per_step_shape, dtype) in field_specs.items():
            full_shape = (self.num_trajectories, self.horizon, *per_step_shape)
            chunks = self._build_chunks(full_shape)
            fill_value = False if np.dtype(dtype) == np.dtype(np.bool_) else 0
            dataset_kwargs: dict[str, Any] = {
                "shape": full_shape,
                "dtype": dtype,
                "chunks": chunks,
                "fillvalue": fill_value,
            }
            if compression is not None:
                dataset_kwargs["compression"] = compression
                if compression_level is not None:
                    dataset_kwargs["compression_opts"] = compression_level
            self._datasets[name] = self._file.create_dataset(name, **dataset_kwargs)

        self._traj_lengths = self._file.create_dataset(
            "traj_lengths",
            shape=(self.num_trajectories,),
            dtype=np.int32,
            chunks=(min(self.num_trajectories, self._chunk_trajectories),),
            fillvalue=0,
        )
        self._source_env_ids = self._file.create_dataset(
            "source_env_ids",
            shape=(self.num_trajectories,),
            dtype=np.int32,
            chunks=(min(self.num_trajectories, self._chunk_trajectories),),
            fillvalue=-1,
        )
        self._episode_returns = self._file.create_dataset(
            "episode_returns",
            shape=(self.num_trajectories,),
            dtype=np.float32,
            chunks=(min(self.num_trajectories, self._chunk_trajectories),),
            fillvalue=0.0,
        )

        for key, value in metadata.items():
            self._file.attrs[key] = _to_hdf5_attr_value(value)
        self._file.attrs["shape_convention"] = "trajectory_major[num_trajectories, horizon, ...]"
        self._file.attrs["total_trajectories"] = 0
        self._file.attrs["total_transitions"] = 0

    @property
    def num_written(self) -> int:
        return self._write_index

    @property
    def remaining_capacity(self) -> int:
        return self.num_trajectories - self._write_index

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    def append_trajectory(
        self,
        trajectory: dict[str, torch.Tensor | np.ndarray],
        *,
        length: int,
        env_id: int,
        episode_return: float,
    ) -> int:
        """Append one trajectory and return its trajectory index."""

        if self._write_index >= self.num_trajectories:
            raise RuntimeError("HDF5 writer is already full.")
        if length < 1 or length > self.horizon:
            raise ValueError(f"Invalid trajectory length {length}; expected 1 <= length <= {self.horizon}.")

        index = self._write_index
        for name, dataset in self._datasets.items():
            if name not in trajectory:
                raise KeyError(f"Trajectory is missing required field: {name}")
            array = _to_numpy(trajectory[name])
            if array.shape[0] != length:
                raise ValueError(
                    f"Field '{name}' has length {array.shape[0]} but expected trajectory length {length}."
                )
            dataset[index, :length] = array

        self._traj_lengths[index] = length
        self._source_env_ids[index] = int(env_id)
        self._episode_returns[index] = float(episode_return)

        self._write_index += 1
        self._total_transitions += int(length)
        self._file.attrs["total_trajectories"] = self._write_index
        self._file.attrs["total_transitions"] = self._total_transitions

        if self._write_index % self._flush_every == 0:
            self._file.flush()
        return index

    def close(self) -> None:
        """Flush and close the underlying HDF5 file."""

        if getattr(self, "_file", None) is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _build_chunks(self, full_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Chunk full trajectories together for efficient sequential trajectory reads."""

        if len(full_shape) == 0:
            return ()
        if len(full_shape) == 1:
            return (min(full_shape[0], self._chunk_trajectories),)
        return (min(full_shape[0], self._chunk_trajectories), full_shape[1], *full_shape[2:])
