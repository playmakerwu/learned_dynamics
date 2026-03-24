"""Shared utilities for solver24-vs-NeRD evaluation without touching old code paths."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from nerd_bridge.common import configure_nerd_imports, ensure_directory, jsonable
from nerd_bridge.training import load_normalization_stats

configure_nerd_imports()

from models.models import ModelMixedInput

from .config import EvalConfig


@dataclass(slots=True)
class CollectorDataset:
    """In-memory view of a collector-format HDF5 trajectory file."""

    path: Path
    attrs: dict[str, Any]
    data: dict[str, np.ndarray]

    @property
    def num_trajectories(self) -> int:
        self.require_keys("states")
        return int(self.data["states"].shape[0])

    @property
    def horizon(self) -> int:
        self.require_keys("states")
        return int(self.data["states"].shape[1])

    @property
    def state_dim(self) -> int:
        self.require_keys("states")
        return int(self.data["states"].shape[2])

    @property
    def traj_lengths(self) -> np.ndarray:
        self.require_keys("traj_lengths")
        return self.data["traj_lengths"].astype(np.int32, copy=False)

    @property
    def source_env_ids(self) -> np.ndarray:
        self.require_keys("source_env_ids")
        return self.data["source_env_ids"].astype(np.int32, copy=False)

    def has_key(self, key: str) -> bool:
        return key in self.data

    def require_keys(self, *keys: str) -> None:
        missing = [key for key in keys if key not in self.data]
        if missing:
            raise KeyError(
                f"Dataset {self.path} is missing required keys: {missing}. "
                f"Available keys: {sorted(self.data.keys())}"
            )


def task_name_for_solver(solver_position_iterations: int, cfg: EvalConfig) -> str:
    """Map a solver iteration count to the matching Isaac Lab peg-insert env ID."""

    if solver_position_iterations == 24:
        return cfg.solver24_task_name
    if solver_position_iterations == 192:
        return cfg.solver192_task_name
    raise ValueError(f"Unsupported peg-insert solver position iteration count: {solver_position_iterations}")


def default_output_for_solver(solver_position_iterations: int, cfg: EvalConfig) -> Path:
    """Pick the default raw trajectory output path for the requested solver."""

    if solver_position_iterations == 24:
        return cfg.solver24_real_path
    if solver_position_iterations == 192:
        return cfg.solver192_real_path
    raise ValueError(f"Unsupported solver: {solver_position_iterations}")


def build_collector_command(
    cfg: EvalConfig,
    *,
    solver_position_iterations: int,
    output_path: Path,
) -> list[str]:
    """Assemble the collector CLI for a specific solver variant."""

    task_name = task_name_for_solver(solver_position_iterations, cfg)
    command = [
        str(Path(sys.executable).resolve()),
        str((Path(__file__).resolve().parents[1] / "nerd_collector" / "collector.py").resolve()),
        "--task",
        task_name,
        "--checkpoint",
        str(cfg.policy_checkpoint.resolve()),
        "--output_path",
        str(output_path.resolve()),
        "--num_trajectories",
        str(cfg.num_trajectories),
        "--num_envs",
        str(cfg.num_envs),
        "--seed",
        str(cfg.seed),
        "--device",
        str(cfg.device),
        "--policy_device",
        str(cfg.resolved_policy_device()),
        "--log_every_steps",
        str(cfg.log_every_steps),
        "--headless",
    ]
    if cfg.horizon_steps is not None:
        command.extend(["--horizon_steps", str(cfg.horizon_steps)])
    if cfg.episode_length_steps is not None:
        command.extend(["--episode_length_steps", str(cfg.episode_length_steps)])
    if not cfg.deterministic_policy:
        command.append("--stochastic_policy")
    return command


def annotate_evaluation_hdf5(
    path: str | Path,
    *,
    solver_position_iterations: int,
    eval_role: str,
    cfg: EvalConfig,
) -> None:
    """Attach evaluation-only metadata to a newly collected HDF5 file."""

    dataset_path = Path(path).expanduser().resolve()
    with h5py.File(dataset_path, "a") as file:
        file.attrs["evaluation_role"] = eval_role
        file.attrs["solver_position_iteration_count"] = int(solver_position_iterations)
        file.attrs["evaluation_seed"] = int(cfg.seed)
        file.attrs["evaluation_num_trajectories"] = int(cfg.num_trajectories)
        file.attrs["evaluation_num_envs"] = int(cfg.num_envs)
        file.attrs["evaluation_policy_checkpoint"] = str(cfg.policy_checkpoint.resolve())
        file.attrs["evaluation_expected_alignment"] = "trajectory index and source_env_ids"


def run_real_collection(
    cfg: EvalConfig,
    *,
    solver_position_iterations: int,
    output_path: Path | None = None,
) -> Path:
    """Run the existing collector as a subprocess and tag the new output as evaluation data."""

    target_path = (output_path or default_output_for_solver(solver_position_iterations, cfg)).expanduser().resolve()
    ensure_directory(target_path)
    command = build_collector_command(
        cfg,
        solver_position_iterations=solver_position_iterations,
        output_path=target_path,
    )
    print("Running collector command:", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    annotate_evaluation_hdf5(
        target_path,
        solver_position_iterations=solver_position_iterations,
        eval_role="solver24_baseline" if solver_position_iterations == 24 else "solver192_reference",
        cfg=cfg,
    )
    print(f"Saved evaluation trajectories to: {target_path}", flush=True)
    return target_path


def load_collector_dataset(path: str | Path) -> CollectorDataset:
    """Read the full collector-format HDF5 file into memory for lightweight evaluation."""

    dataset_path = Path(path).expanduser().resolve()
    with h5py.File(dataset_path, "r") as file:
        attrs = {key: file.attrs[key] for key in file.attrs.keys()}
        data = {key: file[key][...] for key in file.keys()}
    dataset = CollectorDataset(path=dataset_path, attrs=attrs, data=data)

    if not dataset.data:
        raise RuntimeError(
            f"Dataset {dataset_path} contains no HDF5 datasets.\n"
            "This usually means the collection process failed after creating the file header/attrs but before "
            "writing trajectory arrays. Re-run the corresponding collection step."
        )

    required = {"states", "next_states", "joint_acts", "traj_lengths"}
    missing_required = sorted(required - set(dataset.data.keys()))
    if missing_required:
        raise RuntimeError(
            f"Dataset {dataset_path} is missing required trajectory datasets: {missing_required}\n"
            f"Available keys: {sorted(dataset.data.keys())}\n"
            "Please re-run collection for this file."
        )
    return dataset


def _parse_jsonish(value: Any) -> Any:
    """Decode HDF5 string attributes that may hold JSON payloads."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def parse_state_layout(dataset: CollectorDataset) -> list[dict[str, Any]]:
    """Recover the collector's state layout metadata if it was saved."""

    raw = dataset.attrs.get("state_layout")
    parsed = _parse_jsonish(raw)
    if not isinstance(parsed, list):
        raise KeyError(
            f"Dataset {dataset.path} does not expose a usable 'state_layout' attribute. "
            "The evaluation scripts need it to decode peg/socket pose slices."
        )
    return parsed


def find_state_slice(layout: list[dict[str, Any]], field_name: str) -> slice:
    """Find one named slice inside the flat simulator state vector."""

    for item in layout:
        if item.get("name") == field_name:
            return slice(int(item["start"]), int(item["end"]))
    raise KeyError(f"State field '{field_name}' not found in state layout.")


def root_body_q_from_states(states: np.ndarray, layout: list[dict[str, Any]]) -> np.ndarray:
    """Reconstruct `[x, y, z, qw, qx, qy, qz]` from the flat state tensor."""

    pos_slice = find_state_slice(layout, "held_root_pos_local")
    quat_slice = find_state_slice(layout, "held_root_quat_wxyz")
    return np.concatenate([states[..., pos_slice], states[..., quat_slice]], axis=-1).astype(np.float32, copy=False)


def fixed_root_q_from_states(states: np.ndarray, layout: list[dict[str, Any]]) -> np.ndarray:
    """Reconstruct the fixed-asset pose from the flat state tensor."""

    pos_slice = find_state_slice(layout, "fixed_root_pos_local")
    quat_slice = find_state_slice(layout, "fixed_root_quat_wxyz")
    return np.concatenate([states[..., pos_slice], states[..., quat_slice]], axis=-1).astype(np.float32, copy=False)


def align_real_datasets(
    solver24: CollectorDataset,
    solver192: CollectorDataset,
) -> tuple[CollectorDataset, CollectorDataset, dict[str, Any]]:
    """Align the two real datasets by source env id when possible."""

    alignment_notes: list[str] = []

    if solver24.has_key("source_env_ids") and solver192.has_key("source_env_ids"):
        rough_ids = solver24.source_env_ids
        ref_ids = solver192.source_env_ids
    else:
        rough_ids = None
        ref_ids = None

    if rough_ids is not None and ref_ids is not None and len(np.unique(rough_ids)) == solver24.num_trajectories and len(np.unique(ref_ids)) == solver192.num_trajectories:
        if set(rough_ids.tolist()) != set(ref_ids.tolist()):
            raise ValueError(
                "solver24 and solver192 evaluation files do not expose the same source_env_ids, "
                "so fair alignment failed."
            )
        rough_order = np.argsort(rough_ids)
        ref_order = np.argsort(ref_ids)
        alignment_notes.append("Aligned trajectories by unique source_env_ids.")
    else:
        if solver24.num_trajectories != solver192.num_trajectories:
            raise ValueError(
                "Cannot fall back to index alignment because the two datasets have different numbers of trajectories."
            )
        rough_order = np.arange(solver24.num_trajectories, dtype=np.int32)
        ref_order = np.arange(solver192.num_trajectories, dtype=np.int32)
        if rough_ids is None or ref_ids is None:
            alignment_notes.append("At least one dataset has no source_env_ids; fell back to trajectory index alignment.")
        else:
            alignment_notes.append("Source env ids were not unique; fell back to trajectory index alignment.")

    def reindex(dataset: CollectorDataset, order: np.ndarray) -> CollectorDataset:
        return CollectorDataset(
            path=dataset.path,
            attrs=dict(dataset.attrs),
            data={
                key: value[order, ...] if value.ndim >= 1 and value.shape[0] == dataset.num_trajectories else value
                for key, value in dataset.data.items()
            },
        )

    aligned_rough = reindex(solver24, rough_order)
    aligned_ref = reindex(solver192, ref_order)

    common_trajectories = min(aligned_rough.num_trajectories, aligned_ref.num_trajectories)
    common_horizon = min(aligned_rough.horizon, aligned_ref.horizon)
    alignment_info = {
        "common_trajectories": int(common_trajectories),
        "common_horizon": int(common_horizon),
        "notes": alignment_notes,
    }
    return truncate_dataset(aligned_rough, common_trajectories, common_horizon), truncate_dataset(
        aligned_ref, common_trajectories, common_horizon
    ), alignment_info


def truncate_dataset(dataset: CollectorDataset, num_trajectories: int, horizon: int) -> CollectorDataset:
    """Slice a collector dataset to a shared `(B, T, ...)` prefix."""

    return CollectorDataset(
        path=dataset.path,
        attrs=dict(dataset.attrs),
        data={
            key: _truncate_array(value, num_trajectories=num_trajectories, horizon=horizon, full_trajectories=dataset.num_trajectories)
            for key, value in dataset.data.items()
        },
    )


def _truncate_array(value: np.ndarray, *, num_trajectories: int, horizon: int, full_trajectories: int) -> np.ndarray:
    if value.ndim >= 2 and value.shape[0] == full_trajectories:
        return value[:num_trajectories, :horizon, ...]
    if value.ndim >= 1 and value.shape[0] == full_trajectories:
        return value[:num_trajectories, ...]
    return value


def make_dummy_sample_inputs(dataset: CollectorDataset, input_keys: list[str], *, state_layout: list[dict[str, Any]], device: str) -> dict[str, torch.Tensor]:
    """Build one-sample tensors with the flattened feature shapes expected by ModelMixedInput."""

    sample: dict[str, torch.Tensor] = {}
    for key in input_keys:
        if key == "root_body_q":
            if "root_body_q" in dataset.data:
                raw = dataset.data["root_body_q"][:1, :1, ...]
            else:
                raw = root_body_q_from_states(dataset.data["states"][:1, :1, ...], state_layout)
        else:
            raw = dataset.data[key][:1, :1, ...]
        flat = raw.reshape(1, 1, -1).astype(np.float32, copy=False)
        sample[key] = torch.from_numpy(flat).to(device)
    return sample


def load_nerd_model(
    checkpoint_path: str | Path,
    *,
    sample_inputs: dict[str, torch.Tensor],
    output_dim: int,
    device: str,
) -> tuple[ModelMixedInput, dict[str, Any]]:
    """Restore the trained NeRD model and its normalization statistics."""

    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=device, weights_only=False)
    model = ModelMixedInput(
        input_sample=sample_inputs,
        output_dim=int(output_dim),
        input_cfg=checkpoint["input_cfg"],
        network_cfg=checkpoint["network_cfg"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    load_normalization_stats(model, checkpoint["normalization_state"], device=device)
    model.to(device)
    model.eval()
    return model, checkpoint


def quaternion_geodesic_distance_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute quaternion angular error in degrees with the double-cover handled safely."""

    q1 = q1.astype(np.float64, copy=False)
    q2 = q2.astype(np.float64, copy=False)
    q1 = q1 / np.clip(np.linalg.norm(q1, axis=-1, keepdims=True), 1.0e-12, None)
    q2 = q2 / np.clip(np.linalg.norm(q2, axis=-1, keepdims=True), 1.0e-12, None)
    dots = np.abs(np.sum(q1 * q2, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots)).astype(np.float32)


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON file with stable formatting."""

    target = Path(path).expanduser().resolve()
    ensure_directory(target)
    with target.open("w", encoding="utf-8") as file:
        json.dump(jsonable(payload), file, indent=2, sort_keys=True)
