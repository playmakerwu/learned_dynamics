"""Developer note:
This collector is meant to work with the RL-Games policy trained in
`/home/yiru-wu/Documents/learned_dynamics`.

Use `/home/yiru-wu/Documents/learned_dynamics/play.py` as the primary reference
for how the trained policy should be loaded and stepped for inference.

The goal of this collector is to save NeRD-friendly simulator trajectories,
not standard RL rollout logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from common import DEFAULT_TASK


@dataclass(slots=True)
class HDF5Config:
    """Parameters that control HDF5 layout and write behavior."""

    chunk_trajectories: int = 8
    flush_every_trajectories: int = 8
    compression: str | None = None
    compression_level: int | None = None


@dataclass(slots=True)
class CollectorConfig:
    """Single place for all trajectory collection settings."""

    # Environment / simulation setup.
    task_name: str = DEFAULT_TASK
    num_envs: int = 32
    device: str = "cuda:0"
    policy_device: str | None = None
    headless: bool = True
    enable_cameras: bool = False
    disable_fabric: bool = False
    seed: int = 42

    # RL-Games loading. These mirror the local play.py setup.
    logdir: Path = Path("logs")
    experiment_name: str = "peg_insert_rlgames"
    run_name: str | None = None
    checkpoint_path: Path | None = None
    clip_obs: float = 10.0
    clip_actions: float = 1.0

    # Simulation overrides. Leave as None to keep the task defaults from Isaac Lab.
    sim_dt: float | None = None
    decimation: int | None = None
    episode_length_steps: int | None = None
    horizon_steps: int | None = None

    # Collection targets.
    num_trajectories_to_save: int = 256
    output_path: Path = Path("recordings/nerd_peg_insert_trajectories.hdf5")
    save_on_gpu_first: bool = True
    deterministic_policy: bool = True
    action_noise_std: float = 0.0
    log_every_steps: int = 100

    # Contact processing.
    contact_slot_count_k: int = 64
    max_depth_clamp: float = 0.02
    contact_thickness: float = 0.0

    # Which simulator assets to read.
    robot_asset_name: str = "robot"
    contact_source_asset_name: str = "held_asset"
    contact_target_asset_name: str = "fixed_asset"
    root_body_asset_name: str = "held_asset"
    ee_body_name: str = "panda_fingertip_centered"
    robot_joint_count: int | None = 7

    # Optional saved fields.
    save_root_body_q: bool = True
    save_contact_points_0: bool = True
    save_contact_points_1: bool = True
    save_contact_impulses: bool = True
    save_contact_impulse_vectors: bool = True
    save_contact_identities: bool = True

    # HDF5 writing.
    hdf5: HDF5Config = field(default_factory=HDF5Config)

    def resolved_policy_device(self) -> str:
        return self.policy_device or self.device

    def resolved_output_path(self) -> Path:
        return self.output_path.expanduser().resolve()

    def resolved_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path is None:
            return None
        return self.checkpoint_path.expanduser().resolve()

    def to_metadata_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["logdir"] = str(self.logdir.expanduser())
        data["output_path"] = str(self.output_path.expanduser())
        data["checkpoint_path"] = (
            str(self.checkpoint_path.expanduser()) if self.checkpoint_path is not None else None
        )
        return data


CONFIG = CollectorConfig()
