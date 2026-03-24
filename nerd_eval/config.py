"""Configuration for NeRD vs solver24 evaluation on Isaac Lab peg-insert."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common import DEFAULT_TASK

from nerd_bridge.common import PROJECT_ROOT, default_device


DEFAULT_POLICY_CHECKPOINT = (
    PROJECT_ROOT
    / "logs"
    / "peg_insert_rlgames"
    / "2026-03-13_02-25-23"
    / "nn"
    / "last_peginsert_parallel_1000_ep_800_rew_379.98492.pth"
)


@dataclass(slots=True)
class EvalConfig:
    """User-adjustable knobs for the final solver24-vs-NeRD evaluation."""

    solver24_task_name: str = "Isaac-Factory-PegInsert-Rough24-Direct-v0"
    solver192_task_name: str = DEFAULT_TASK
    seed: int = 42
    num_envs: int = 32
    num_trajectories: int = 32
    device: str = default_device()
    policy_device: str | None = None
    headless: bool = True
    horizon_steps: int | None = None
    episode_length_steps: int | None = None
    deterministic_policy: bool = True
    log_every_steps: int = 50

    policy_checkpoint: Path = DEFAULT_POLICY_CHECKPOINT
    nerd_checkpoint: Path = PROJECT_ROOT / "outputs" / "nerd_peg_insert_run1" / "best_checkpoint.pt"

    recordings_dir: Path = PROJECT_ROOT / "recordings"
    results_dir: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192"

    solver24_real_path: Path = PROJECT_ROOT / "recordings" / "eval_solver24_real.hdf5"
    solver192_real_path: Path = PROJECT_ROOT / "recordings" / "eval_solver192_real.hdf5"

    nerd_rollout_path: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192" / "nerd_rollout_from_solver24.hdf5"
    metrics_json_path: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192" / "comparison_metrics.json"
    metrics_npz_path: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192" / "comparison_curves.npz"
    summary_txt_path: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192" / "comparison_summary.txt"
    state_mse_plot_path: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192" / "state_mse_over_time.png"
    peg_pos_plot_path: Path = PROJECT_ROOT / "outputs" / "nerd_eval_solver24_vs_192" / "peg_pos_error_over_time.png"

    # Rollout policy for NeRD inference.
    rollout_input_source: str = "solver24"

    def resolved_policy_device(self) -> str:
        return self.policy_device or self.device

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

