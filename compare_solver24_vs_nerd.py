"""Compare solver24 and NeRD rollouts against solver192 reference trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from nerd_bridge.common import ensure_directory
from nerd_eval.config import EvalConfig
from nerd_eval.utils import (
    align_real_datasets,
    find_state_slice,
    load_collector_dataset,
    parse_state_layout,
    quaternion_geodesic_distance_deg,
    write_json,
)

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERROR_HORIZONS = [1, 5, 10, 20, 50, 100, 150]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare solver24 and NeRD rollouts against the solver192 reference.")
    parser.add_argument("--solver24_dataset", type=str, default=None)
    parser.add_argument("--solver192_dataset", type=str, default=None)
    parser.add_argument("--nerd_rollout", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser


def masked_mean(values: np.ndarray, valid_mask: np.ndarray) -> float:
    """Average only over valid trajectory-time entries."""

    selected = values[valid_mask]
    if selected.size == 0:
        raise ValueError("Masked metric received zero valid elements.")
    return float(np.mean(selected))


def masked_curve(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Average over trajectories for each time step while respecting valid lengths."""

    curve = np.zeros(values.shape[1], dtype=np.float64)
    for step in range(values.shape[1]):
        step_mask = valid_mask[:, step]
        if np.any(step_mask):
            curve[step] = float(np.mean(values[step_mask, step]))
    return curve.astype(np.float32)


def final_step_gather(values: np.ndarray, traj_lengths: np.ndarray) -> np.ndarray:
    """Gather the last valid time step from a `[B, T, ...]` array."""

    indices = np.clip(traj_lengths - 1, 0, values.shape[1] - 1)
    return values[np.arange(values.shape[0]), indices, ...]


def save_plot(path: Path, *, title: str, ylabel: str, baseline_curve: np.ndarray, nerd_curve: np.ndarray) -> None:
    """Write a simple comparison plot to disk."""

    ensure_directory(path)
    plt.figure(figsize=(8, 4.5))
    plt.plot(baseline_curve, label="solver24 vs solver192", linewidth=2.0)
    plt.plot(nerd_curve, label="NeRD vs solver192", linewidth=2.0)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_error_vs_horizon(
    gt_states: np.ndarray,
    solver24_states: np.ndarray,
    nerd_states: np.ndarray,
    horizons: list[int],
) -> tuple[list[float], list[float]]:
    """Compute rollout-horizon MAE against the solver192 ground truth."""

    if gt_states.shape != solver24_states.shape or gt_states.shape != nerd_states.shape:
        raise ValueError(
            "Expected gt_states, solver24_states, and nerd_states to share the same shape, "
            f"got {gt_states.shape}, {solver24_states.shape}, and {nerd_states.shape}."
        )
    if gt_states.ndim != 3:
        raise ValueError(f"Expected [B, T, D] state arrays, got shape {gt_states.shape}.")

    solver24_errors: list[float] = []
    nerd_errors: list[float] = []
    max_horizon = gt_states.shape[1]

    for horizon in horizons:
        if horizon < 0:
            raise ValueError(f"Horizons must be non-negative, got {horizon}.")
        if horizon >= max_horizon:
            solver24_errors.append(float("nan"))
            nerd_errors.append(float("nan"))
            continue

        gt_step = gt_states[:, horizon, :]
        solver24_step = solver24_states[:, horizon, :]
        nerd_step = nerd_states[:, horizon, :]

        solver24_valid = np.all(np.isfinite(gt_step), axis=-1) & np.all(np.isfinite(solver24_step), axis=-1)
        nerd_valid = np.all(np.isfinite(gt_step), axis=-1) & np.all(np.isfinite(nerd_step), axis=-1)

        if np.any(solver24_valid):
            solver24_errors.append(float(np.mean(np.abs(solver24_step[solver24_valid] - gt_step[solver24_valid]))))
        else:
            solver24_errors.append(float("nan"))

        if np.any(nerd_valid):
            nerd_errors.append(float(np.mean(np.abs(nerd_step[nerd_valid] - gt_step[nerd_valid]))))
        else:
            nerd_errors.append(float("nan"))

    return solver24_errors, nerd_errors


def plot_error_vs_horizon(horizons: list[int], solver24_errors: list[float], nerd_errors: list[float]) -> None:
    """Plot state MAE against prediction horizon for solver24 and NeRD."""

    plt.figure(figsize=(8, 4.5))
    plt.plot(horizons, solver24_errors, marker="o", label="solver=24", linewidth=2.0)
    plt.plot(horizons, nerd_errors, marker="o", label="NeRD", linewidth=2.0)
    plt.xlabel("Prediction Horizon (steps)")
    plt.ylabel("Mean Absolute Error")
    plt.title("Error vs Horizon: solver24 vs NeRD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main() -> None:
    args = build_parser().parse_args()
    cfg = EvalConfig()
    solver24_path = Path(args.solver24_dataset) if args.solver24_dataset is not None else cfg.solver24_real_path
    solver192_path = Path(args.solver192_dataset) if args.solver192_dataset is not None else cfg.solver192_real_path
    rollout_path = Path(args.nerd_rollout) if args.nerd_rollout is not None else cfg.nerd_rollout_path
    output_dir = Path(args.output_dir) if args.output_dir is not None else cfg.results_dir
    ensure_directory(output_dir)

    solver24 = load_collector_dataset(solver24_path)
    solver192 = load_collector_dataset(solver192_path)
    solver24, solver192, alignment_info = align_real_datasets(solver24, solver192)
    state_layout = parse_state_layout(solver192)

    with h5py.File(rollout_path.expanduser().resolve(), "r") as file:
        predicted_states = file["predicted_states"][...].astype(np.float32, copy=False)
        predicted_next_states = file["predicted_next_states"][...].astype(np.float32, copy=False)
        predicted_root_body_q = file["predicted_root_body_q"][...].astype(np.float32, copy=False)
        predicted_fixed_root_q = file["predicted_fixed_root_q"][...].astype(np.float32, copy=False)
        traj_lengths = file["traj_lengths"][...].astype(np.int32, copy=False)

    traj_lengths = np.minimum(traj_lengths, np.minimum(solver24.traj_lengths, solver192.traj_lengths))
    valid_mask = np.arange(solver192.horizon, dtype=np.int32)[None, :] < traj_lengths[:, None]

    solver24_state_diff = solver24.data["states"] - solver192.data["states"]
    nerd_state_diff = predicted_states - solver192.data["states"]
    solver24_state_se = np.mean(np.square(solver24_state_diff), axis=-1)
    nerd_state_se = np.mean(np.square(nerd_state_diff), axis=-1)
    solver24_state_ae = np.mean(np.abs(solver24_state_diff), axis=-1)
    nerd_state_ae = np.mean(np.abs(nerd_state_diff), axis=-1)

    solver24_final_state = final_step_gather(solver24.data["states"], traj_lengths)
    solver192_final_state = final_step_gather(solver192.data["states"], traj_lengths)
    nerd_final_state = final_step_gather(predicted_states, traj_lengths)

    solver24_final_next_state = final_step_gather(solver24.data["next_states"], traj_lengths)
    solver192_final_next_state = final_step_gather(solver192.data["next_states"], traj_lengths)
    nerd_final_next_state = final_step_gather(predicted_next_states, traj_lengths)

    ref_root_body_q = solver192.data["root_body_q"].astype(np.float32, copy=False)
    rough_root_body_q = solver24.data["root_body_q"].astype(np.float32, copy=False)
    solver24_peg_pos_err = np.linalg.norm(rough_root_body_q[..., :3] - ref_root_body_q[..., :3], axis=-1)
    nerd_peg_pos_err = np.linalg.norm(predicted_root_body_q[..., :3] - ref_root_body_q[..., :3], axis=-1)
    solver24_peg_quat_err_deg = quaternion_geodesic_distance_deg(rough_root_body_q[..., 3:], ref_root_body_q[..., 3:])
    nerd_peg_quat_err_deg = quaternion_geodesic_distance_deg(predicted_root_body_q[..., 3:], ref_root_body_q[..., 3:])

    held_pos_slice = find_state_slice(state_layout, "held_root_pos_local")
    fixed_pos_slice = find_state_slice(state_layout, "fixed_root_pos_local")
    solver24_rel = solver24.data["states"][..., held_pos_slice] - solver24.data["states"][..., fixed_pos_slice]
    solver192_rel = solver192.data["states"][..., held_pos_slice] - solver192.data["states"][..., fixed_pos_slice]
    nerd_rel = predicted_states[..., held_pos_slice] - predicted_fixed_root_q[..., :3]
    solver24_rel_err = np.linalg.norm(solver24_rel - solver192_rel, axis=-1)
    nerd_rel_err = np.linalg.norm(nerd_rel - solver192_rel, axis=-1)

    gt_states_for_horizon = solver192.data["states"].astype(np.float32, copy=True)
    solver24_states_for_horizon = solver24.data["states"].astype(np.float32, copy=True)
    nerd_states_for_horizon = predicted_states.astype(np.float32, copy=True)
    gt_states_for_horizon[~valid_mask] = np.nan
    solver24_states_for_horizon[~valid_mask] = np.nan
    nerd_states_for_horizon[~valid_mask] = np.nan

    solver24_horizon_errors, nerd_horizon_errors = compute_error_vs_horizon(
        gt_states_for_horizon,
        solver24_states_for_horizon,
        nerd_states_for_horizon,
        ERROR_HORIZONS,
    )

    metrics = {
        "alignment_info": alignment_info,
        "num_trajectories": int(traj_lengths.shape[0]),
        "horizon": int(solver192.horizon),
        "traj_length_mean": float(np.mean(traj_lengths)),
        "solver24_vs_solver192": {
            "state_mse": masked_mean(solver24_state_se, valid_mask),
            "state_mae": masked_mean(solver24_state_ae, valid_mask),
            "final_state_mse": float(np.mean(np.square(solver24_final_state - solver192_final_state))),
            "final_state_mae": float(np.mean(np.abs(solver24_final_state - solver192_final_state))),
            "final_next_state_mse": float(np.mean(np.square(solver24_final_next_state - solver192_final_next_state))),
            "peg_position_error": masked_mean(solver24_peg_pos_err, valid_mask),
            "peg_orientation_error_deg": masked_mean(solver24_peg_quat_err_deg, valid_mask),
            "peg_socket_relative_position_error": masked_mean(solver24_rel_err, valid_mask),
        },
        "nerd_vs_solver192": {
            "state_mse": masked_mean(nerd_state_se, valid_mask),
            "state_mae": masked_mean(nerd_state_ae, valid_mask),
            "final_state_mse": float(np.mean(np.square(nerd_final_state - solver192_final_state))),
            "final_state_mae": float(np.mean(np.abs(nerd_final_state - solver192_final_state))),
            "final_next_state_mse": float(np.mean(np.square(nerd_final_next_state - solver192_final_next_state))),
            "peg_position_error": masked_mean(nerd_peg_pos_err, valid_mask),
            "peg_orientation_error_deg": masked_mean(nerd_peg_quat_err_deg, valid_mask),
            "peg_socket_relative_position_error": masked_mean(nerd_rel_err, valid_mask),
        },
    }

    improvement = {}
    verdict_details = {}
    for metric_name, rough_value in metrics["solver24_vs_solver192"].items():
        nerd_value = metrics["nerd_vs_solver192"][metric_name]
        improvement_ratio = (rough_value - nerd_value) / max(abs(rough_value), 1.0e-12)
        improvement[metric_name] = {
            "relative_improvement": float(improvement_ratio),
            "nerd_beats_solver24": bool(nerd_value < rough_value),
        }
        verdict_details[metric_name] = bool(nerd_value < rough_value)

    verdict = {
        "nerd_beats_solver24_on_all_primary_metrics": bool(all(verdict_details.values())),
        "nerd_beats_solver24_on_state_mse": verdict_details["state_mse"],
        "nerd_beats_solver24_on_state_mae": verdict_details["state_mae"],
        "nerd_beats_solver24_on_final_state_mse": verdict_details["final_state_mse"],
        "nerd_beats_solver24_on_peg_position_error": verdict_details["peg_position_error"],
        "primary_question_answer": (
            "yes"
            if verdict_details["state_mse"] and verdict_details["state_mae"]
            else "no"
        ),
    }

    state_mse_curve_solver24 = masked_curve(solver24_state_se, valid_mask)
    state_mse_curve_nerd = masked_curve(nerd_state_se, valid_mask)
    peg_pos_curve_solver24 = masked_curve(solver24_peg_pos_err, valid_mask)
    peg_pos_curve_nerd = masked_curve(nerd_peg_pos_err, valid_mask)

    save_plot(
        output_dir / "state_mse_over_time.png",
        title="State MSE vs solver192 reference",
        ylabel="Mean Squared Error",
        baseline_curve=state_mse_curve_solver24,
        nerd_curve=state_mse_curve_nerd,
    )
    save_plot(
        output_dir / "peg_position_error_over_time.png",
        title="Peg Position Error vs solver192 reference",
        ylabel="Position Error",
        baseline_curve=peg_pos_curve_solver24,
        nerd_curve=peg_pos_curve_nerd,
    )
    error_vs_horizon_path = output_dir / "error_vs_horizon.png"
    ensure_directory(error_vs_horizon_path)
    plot_error_vs_horizon(ERROR_HORIZONS, solver24_horizon_errors, nerd_horizon_errors)
    plt.savefig(error_vs_horizon_path)
    plt.close()

    np.savez(
        output_dir / "comparison_curves.npz",
        state_mse_solver24=state_mse_curve_solver24,
        state_mse_nerd=state_mse_curve_nerd,
        peg_pos_solver24=peg_pos_curve_solver24,
        peg_pos_nerd=peg_pos_curve_nerd,
        error_horizons=np.asarray(ERROR_HORIZONS, dtype=np.int32),
        error_vs_horizon_solver24=np.asarray(solver24_horizon_errors, dtype=np.float32),
        error_vs_horizon_nerd=np.asarray(nerd_horizon_errors, dtype=np.float32),
        traj_lengths=traj_lengths,
    )

    summary_payload = {
        "solver24_dataset": str(solver24_path.expanduser().resolve()),
        "solver192_dataset": str(solver192_path.expanduser().resolve()),
        "nerd_rollout": str(rollout_path.expanduser().resolve()),
        "error_vs_horizon": {
            "horizons": ERROR_HORIZONS,
            "solver24_state_mae": [float(value) if np.isfinite(value) else None for value in solver24_horizon_errors],
            "nerd_state_mae": [float(value) if np.isfinite(value) else None for value in nerd_horizon_errors],
        },
        "metrics": metrics,
        "improvement": improvement,
        "verdict": verdict,
    }
    write_json(output_dir / "comparison_metrics.json", summary_payload)

    summary_text = "\n".join(
        [
            "NeRD vs solver24 evaluation against solver192 reference",
            f"Trajectories compared: {traj_lengths.shape[0]}",
            f"Average valid rollout length: {np.mean(traj_lengths):.2f}",
            f"State MSE: solver24={metrics['solver24_vs_solver192']['state_mse']:.6f}, NeRD={metrics['nerd_vs_solver192']['state_mse']:.6f}",
            f"State MAE: solver24={metrics['solver24_vs_solver192']['state_mae']:.6f}, NeRD={metrics['nerd_vs_solver192']['state_mae']:.6f}",
            f"Final-state MSE: solver24={metrics['solver24_vs_solver192']['final_state_mse']:.6f}, NeRD={metrics['nerd_vs_solver192']['final_state_mse']:.6f}",
            f"Peg position error: solver24={metrics['solver24_vs_solver192']['peg_position_error']:.6f}, NeRD={metrics['nerd_vs_solver192']['peg_position_error']:.6f}",
            f"Peg orientation error (deg): solver24={metrics['solver24_vs_solver192']['peg_orientation_error_deg']:.6f}, NeRD={metrics['nerd_vs_solver192']['peg_orientation_error_deg']:.6f}",
            f"Primary answer: {verdict['primary_question_answer']}",
        ]
    )
    summary_txt = output_dir / "comparison_summary.txt"
    ensure_directory(summary_txt)
    summary_txt.write_text(summary_text, encoding="utf-8")

    print("Horizon | solver24 | NeRD", flush=True)
    print("--------------------------------", flush=True)
    for horizon, solver24_error, nerd_error in zip(ERROR_HORIZONS, solver24_horizon_errors, nerd_horizon_errors):
        solver24_text = f"{solver24_error:.6f}" if np.isfinite(solver24_error) else "nan"
        nerd_text = f"{nerd_error:.6f}" if np.isfinite(nerd_error) else "nan"
        print(f"{horizon:<7} | {solver24_text:<8} | {nerd_text}", flush=True)

    print(summary_text, flush=True)
    print(f"Saved comparison metrics to: {output_dir / 'comparison_metrics.json'}", flush=True)
    print(f"Saved error-vs-horizon plot to: {error_vs_horizon_path}", flush=True)


if __name__ == "__main__":
    main()
