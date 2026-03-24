"""Create a trajectory-level train/test split stratified by rollout difficulty.

This keeps the existing random split script untouched and adds a new, more robust split
path for datasets where some trajectories are much harder than others.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from nerd_bridge.common import ensure_directory, save_json


@dataclass(slots=True)
class SplitStats:
    """Summary statistics for one side of the split."""

    num_trajectories: int
    total_transitions: int
    difficulty_mean: float
    difficulty_std: float
    difficulty_min: float
    difficulty_max: float
    return_mean: float
    return_std: float
    contact_active_frac_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_trajectories": int(self.num_trajectories),
            "total_transitions": int(self.total_transitions),
            "difficulty_mean": float(self.difficulty_mean),
            "difficulty_std": float(self.difficulty_std),
            "difficulty_min": float(self.difficulty_min),
            "difficulty_max": float(self.difficulty_max),
            "return_mean": float(self.return_mean),
            "return_std": float(self.return_std),
            "contact_active_frac_mean": float(self.contact_active_frac_mean),
        }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a difficulty-stratified train/test split for NeRD.")
    parser.add_argument("--input", type=Path, required=True, help="Converted NeRD dataset path with a /data group.")
    parser.add_argument("--train_indices", type=Path, required=True)
    parser.add_argument("--test_indices", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_bins",
        type=int,
        default=20,
        help="Number of difficulty quantile bins used for stratification.",
    )
    return parser


def _load_converted_group(path: Path) -> h5py.Group:
    file = h5py.File(path, "r")
    if "data" not in file:
        file.close()
        raise KeyError(f"Expected converted NeRD dataset with /data group, got: {path}")
    return file["data"]


def compute_per_trajectory_difficulty(data_group: h5py.Group) -> dict[str, np.ndarray]:
    """Compute per-trajectory difficulty and supporting stats from a converted dataset.

    The converted dataset uses shape convention ``[T, B, ...]``.
    Difficulty is the mean valid-step next-state delta MSE per trajectory.
    """

    states = data_group["states"][:]
    next_states = data_group["next_states"][:]
    traj_lengths = data_group["traj_lengths"][:].astype(np.int32)
    returns = data_group["episode_returns"][:].astype(np.float32)
    contact_counts = data_group["contact_counts"][:] if "contact_counts" in data_group else None

    horizon, num_trajectories = states.shape[:2]
    if next_states.shape[:2] != (horizon, num_trajectories):
        raise ValueError("states and next_states shapes are inconsistent.")

    sq = ((next_states - states) ** 2).mean(axis=2)  # [T, B]
    difficulty = np.zeros((num_trajectories,), dtype=np.float32)
    contact_active_frac = np.zeros((num_trajectories,), dtype=np.float32)

    for traj_idx in range(num_trajectories):
        valid_steps = int(traj_lengths[traj_idx])
        if valid_steps <= 0:
            raise ValueError(f"Trajectory {traj_idx} has non-positive length: {valid_steps}")
        difficulty[traj_idx] = float(np.mean(sq[:valid_steps, traj_idx]))
        if contact_counts is not None:
            contact_active_frac[traj_idx] = float(np.mean(contact_counts[:valid_steps, traj_idx] > 0))

    return {
        "difficulty": difficulty,
        "returns": returns,
        "traj_lengths": traj_lengths,
        "contact_active_frac": contact_active_frac,
    }


def assign_quantile_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Assign each value to a quantile bin, collapsing duplicate edges safely."""

    if num_bins < 2:
        return np.zeros_like(values, dtype=np.int32)

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros_like(values, dtype=np.int32)

    # Digitize against interior edges so bins are [edge_i, edge_{i+1}).
    return np.digitize(values, bins=edges[1:-1], right=False).astype(np.int32)


def stratified_split(indices: np.ndarray, strata: np.ndarray, *, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split trajectory indices within each stratum using a deterministic RNG."""

    rng = np.random.default_rng(seed=seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for stratum in np.unique(strata):
        stratum_indices = indices[strata == stratum].copy()
        rng.shuffle(stratum_indices)

        if len(stratum_indices) == 1:
            train_count = 1 if train_ratio >= 0.5 else 0
        else:
            train_count = int(round(len(stratum_indices) * train_ratio))
            train_count = max(1, min(len(stratum_indices) - 1, train_count))

        train_parts.append(np.sort(stratum_indices[:train_count]))
        test_parts.append(np.sort(stratum_indices[train_count:]))

    train_indices = np.sort(np.concatenate(train_parts).astype(np.int32))
    test_indices = np.sort(np.concatenate(test_parts).astype(np.int32))
    return train_indices, test_indices


def summarize_split(
    indices: np.ndarray,
    *,
    difficulty: np.ndarray,
    returns: np.ndarray,
    traj_lengths: np.ndarray,
    contact_active_frac: np.ndarray,
) -> SplitStats:
    """Build one-side summary statistics."""

    subset_difficulty = difficulty[indices]
    subset_returns = returns[indices]
    subset_lengths = traj_lengths[indices]
    subset_contact_active = contact_active_frac[indices]
    return SplitStats(
        num_trajectories=int(len(indices)),
        total_transitions=int(subset_lengths.sum()),
        difficulty_mean=float(subset_difficulty.mean()),
        difficulty_std=float(subset_difficulty.std()),
        difficulty_min=float(subset_difficulty.min()),
        difficulty_max=float(subset_difficulty.max()),
        return_mean=float(subset_returns.mean()),
        return_std=float(subset_returns.std()),
        contact_active_frac_mean=float(subset_contact_active.mean()),
    )


def main() -> None:
    args = make_parser().parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {args.train_ratio}")

    dataset_path = args.input.expanduser().resolve()
    ensure_directory(args.train_indices)
    ensure_directory(args.test_indices)
    ensure_directory(args.summary)

    file = h5py.File(dataset_path, "r")
    try:
        if "data" not in file:
            raise KeyError(f"Expected converted NeRD dataset with /data group, got: {dataset_path}")
        data_group = file["data"]
        metrics = compute_per_trajectory_difficulty(data_group)
    finally:
        file.close()

    num_trajectories = int(metrics["difficulty"].shape[0])
    all_indices = np.arange(num_trajectories, dtype=np.int32)
    strata = assign_quantile_bins(metrics["difficulty"], int(args.num_bins))
    train_indices, test_indices = stratified_split(
        all_indices,
        strata,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )

    np.save(args.train_indices.expanduser().resolve(), train_indices)
    np.save(args.test_indices.expanduser().resolve(), test_indices)

    train_stats = summarize_split(
        train_indices,
        difficulty=metrics["difficulty"],
        returns=metrics["returns"],
        traj_lengths=metrics["traj_lengths"],
        contact_active_frac=metrics["contact_active_frac"],
    )
    test_stats = summarize_split(
        test_indices,
        difficulty=metrics["difficulty"],
        returns=metrics["returns"],
        traj_lengths=metrics["traj_lengths"],
        contact_active_frac=metrics["contact_active_frac"],
    )

    summary = {
        "dataset_path": str(dataset_path),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "num_bins": int(args.num_bins),
        "num_trajectories": int(num_trajectories),
        "train_indices_path": str(args.train_indices.expanduser().resolve()),
        "test_indices_path": str(args.test_indices.expanduser().resolve()),
        "train": train_stats.to_dict(),
        "test": test_stats.to_dict(),
        "difficulty_mean_gap": float(test_stats.difficulty_mean - train_stats.difficulty_mean),
        "difficulty_mean_ratio": float(test_stats.difficulty_mean / max(train_stats.difficulty_mean, 1.0e-12)),
    }
    save_json(args.summary, summary)

    print(f"Input dataset: {dataset_path}", flush=True)
    print(f"Train trajectories: {train_stats.num_trajectories}", flush=True)
    print(f"Test trajectories: {test_stats.num_trajectories}", flush=True)
    print(
        f"Train difficulty mean={train_stats.difficulty_mean:.6f}, "
        f"Test difficulty mean={test_stats.difficulty_mean:.6f}, "
        f"Ratio={summary['difficulty_mean_ratio']:.3f}",
        flush=True,
    )
    print(f"Train indices: {args.train_indices.expanduser().resolve()}", flush=True)
    print(f"Test indices: {args.test_indices.expanduser().resolve()}", flush=True)
    print(f"Split summary: {args.summary.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
