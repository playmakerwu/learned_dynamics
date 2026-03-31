"""Developer note:
This collector is intended to consume the RL-Games policy trained under
`/home/yiru-wu/Documents/learned_dynamics`.

Treat `/home/yiru-wu/Documents/learned_dynamics/play.py` as the primary
reference for how the policy should be created, restored, and stepped for
inference.

Unlike a standard RL rollout script, this collector writes NeRD-friendly
trajectory datasets with simulator-level state and contact features.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from nerd_collector.config import CONFIG, CollectorConfig
    from nerd_collector.contact_utils import FixedSlotContacts
    from nerd_collector.hdf5_utils import TrajectoryHDF5Writer
else:
    from .config import CONFIG, CollectorConfig
    from .contact_utils import FixedSlotContacts
    from .hdf5_utils import TrajectoryHDF5Writer


def make_parser() -> argparse.ArgumentParser:
    """Create the CLI parser while keeping config.py as the main source of defaults."""

    parser = argparse.ArgumentParser(description="Collect NeRD-friendly peg-insert trajectories from an RL-Games policy.")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--policy_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_trajectories", type=int, default=None)
    parser.add_argument("--episode_length_steps", type=int, default=None)
    parser.add_argument("--horizon_steps", type=int, default=None)
    parser.add_argument("--action_noise_std", type=float, default=None)
    parser.add_argument("--log_every_steps", type=int, default=None)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--stochastic_policy", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _flag_was_provided(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def resolve_runtime_config(base_cfg: CollectorConfig, args: argparse.Namespace, argv: list[str]) -> CollectorConfig:
    """Merge CLI overrides into the dataclass config."""

    runtime_cfg = base_cfg

    if args.task is not None:
        runtime_cfg = replace(runtime_cfg, task_name=args.task)
    if args.num_envs is not None:
        runtime_cfg = replace(runtime_cfg, num_envs=args.num_envs)
    if args.policy_device is not None:
        runtime_cfg = replace(runtime_cfg, policy_device=args.policy_device)
    if args.seed is not None:
        runtime_cfg = replace(runtime_cfg, seed=args.seed)
    if args.checkpoint is not None:
        runtime_cfg = replace(runtime_cfg, checkpoint_path=Path(args.checkpoint))
    if args.output_path is not None:
        runtime_cfg = replace(runtime_cfg, output_path=Path(args.output_path))
    if args.num_trajectories is not None:
        runtime_cfg = replace(runtime_cfg, num_trajectories_to_save=args.num_trajectories)
    if args.episode_length_steps is not None:
        runtime_cfg = replace(runtime_cfg, episode_length_steps=args.episode_length_steps)
    if args.horizon_steps is not None:
        runtime_cfg = replace(runtime_cfg, horizon_steps=args.horizon_steps)
    if args.action_noise_std is not None:
        runtime_cfg = replace(runtime_cfg, action_noise_std=args.action_noise_std)
    if args.log_every_steps is not None:
        runtime_cfg = replace(runtime_cfg, log_every_steps=args.log_every_steps)
    if args.disable_fabric:
        runtime_cfg = replace(runtime_cfg, disable_fabric=True)
    if args.stochastic_policy:
        runtime_cfg = replace(runtime_cfg, deterministic_policy=False)

    if _flag_was_provided(argv, "--device"):
        runtime_cfg = replace(runtime_cfg, device=args.device)
    if _flag_was_provided(argv, "--headless"):
        runtime_cfg = replace(runtime_cfg, headless=args.headless)
    if _flag_was_provided(argv, "--enable_cameras"):
        runtime_cfg = replace(runtime_cfg, enable_cameras=args.enable_cameras)

    # Keep AppLauncher aligned with the config even when the user relies on config.py only.
    args.device = runtime_cfg.device
    args.headless = runtime_cfg.headless
    args.enable_cameras = runtime_cfg.enable_cameras

    return runtime_cfg


@dataclass(slots=True, frozen=True)
class StateField:
    """One contiguous slice inside the flat NeRD state vector."""

    name: str
    width: int
    source: str
    description: str


@dataclass(slots=True)
class StateSnapshot:
    """State-aligned tensors for one environment step."""

    state: Any
    gravity_dir: Any
    root_body_q: Any | None


class StateAssembler:
    """Assemble a simulator-level state vector suitable for NeRD dynamics modeling."""

    def __init__(self, env: Any, cfg: CollectorConfig):
        self.env = env
        self.cfg = cfg
        self.robot = env.scene.articulations[cfg.robot_asset_name]
        self.contact_source_asset = env.scene.articulations[cfg.contact_source_asset_name]
        self.contact_target_asset = env.scene.articulations[cfg.contact_target_asset_name]
        self.root_body_asset = env.scene.articulations[cfg.root_body_asset_name]

        self.ee_body_index = self.robot.body_names.index(cfg.ee_body_name)
        inferred_joint_count = self.robot.num_joints if cfg.robot_joint_count is None else cfg.robot_joint_count
        if inferred_joint_count > self.robot.num_joints:
            raise ValueError(
                f"Requested robot_joint_count={inferred_joint_count}, but the robot only has {self.robot.num_joints} joints."
            )
        self.robot_joint_count = inferred_joint_count

        self.fields = [
            StateField(
                name="robot_joint_pos",
                width=self.robot_joint_count,
                source="direct",
                description="Robot joint positions read from the articulation state.",
            ),
            StateField(
                name="robot_joint_vel",
                width=self.robot_joint_count,
                source="direct",
                description="Robot joint velocities read from the articulation state.",
            ),
            StateField(
                name="ee_pos_local",
                width=3,
                source="derived",
                description="End-effector position in the per-env local frame.",
            ),
            StateField(
                name="ee_quat_wxyz",
                width=4,
                source="direct",
                description="End-effector orientation quaternion in world frame.",
            ),
            StateField(
                name="ee_lin_vel_w",
                width=3,
                source="direct",
                description="End-effector linear velocity in world frame.",
            ),
            StateField(
                name="ee_ang_vel_w",
                width=3,
                source="direct",
                description="End-effector angular velocity in world frame.",
            ),
            StateField(
                name="held_root_pos_local",
                width=3,
                source="derived",
                description="Peg root position in the per-env local frame.",
            ),
            StateField(
                name="held_root_quat_wxyz",
                width=4,
                source="direct",
                description="Peg root orientation quaternion in world frame.",
            ),
            StateField(
                name="held_root_lin_vel_w",
                width=3,
                source="direct",
                description="Peg root linear velocity in world frame.",
            ),
            StateField(
                name="held_root_ang_vel_w",
                width=3,
                source="direct",
                description="Peg root angular velocity in world frame.",
            ),
            StateField(
                name="fixed_root_pos_local",
                width=3,
                source="derived",
                description="Socket root position in the per-env local frame.",
            ),
            StateField(
                name="fixed_root_quat_wxyz",
                width=4,
                source="direct",
                description="Socket root orientation quaternion in world frame.",
            ),
        ]

        self.state_dim = sum(field.width for field in self.fields)
        self.layout_metadata = self._build_layout_metadata()

    def capture(self) -> StateSnapshot:
        """Capture the current NeRD state from simulator tensors."""

        import torch

        env_origins = self.env.scene.env_origins

        # Direct simulator reads from Isaac Lab tensors.
        robot_joint_pos = self.robot.data.joint_pos[:, : self.robot_joint_count]
        robot_joint_vel = self.robot.data.joint_vel[:, : self.robot_joint_count]
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_index]
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_index]
        ee_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.ee_body_index]
        ee_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.ee_body_index]

        held_root_pose_w = self.contact_source_asset.data.root_link_pose_w
        held_root_vel_w = self.contact_source_asset.data.root_link_vel_w
        fixed_root_pose_w = self.contact_target_asset.data.root_link_pose_w
        root_body_pose_w = self.root_body_asset.data.root_link_pose_w if self.cfg.save_root_body_q else None

        # Derived quantities reconstructed from direct simulator tensors.
        ee_pos_local = ee_pos_w - env_origins
        held_root_pos_local = held_root_pose_w[:, :3] - env_origins
        fixed_root_pos_local = fixed_root_pose_w[:, :3] - env_origins
        root_body_q = None
        if root_body_pose_w is not None:
            root_body_q = torch.cat((root_body_pose_w[:, :3] - env_origins, root_body_pose_w[:, 3:7]), dim=-1)

        state = torch.cat(
            [
                robot_joint_pos,
                robot_joint_vel,
                ee_pos_local,
                ee_quat_w,
                ee_lin_vel_w,
                ee_ang_vel_w,
                held_root_pos_local,
                held_root_pose_w[:, 3:7],
                held_root_vel_w[:, :3],
                held_root_vel_w[:, 3:6],
                fixed_root_pos_local,
                fixed_root_pose_w[:, 3:7],
            ],
            dim=-1,
        )
        gravity_dir = self.robot.data.GRAVITY_VEC_W[: self.env.num_envs]
        return StateSnapshot(state=state, gravity_dir=gravity_dir, root_body_q=root_body_q)

    def _build_layout_metadata(self) -> list[dict[str, Any]]:
        start = 0
        layout: list[dict[str, Any]] = []
        for field in self.fields:
            end = start + field.width
            layout.append(
                {
                    "name": field.name,
                    "start": start,
                    "end": end,
                    "source": field.source,
                    "description": field.description,
                }
            )
            start = end
        return layout


@dataclass(slots=True)
class StepResult:
    next_obs: Any
    rewards: Any
    dones: Any
    terminated: Any
    truncated: Any
    extras: dict[str, Any]
    next_snapshot: StateSnapshot
    applied_joint_torque: Any | None = None  # [N, num_joints] from last substep


class EpisodeStorage:
    """GPU/CPU staging buffers for one in-flight episode per environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        horizon: int,
        state_dim: int,
        torque_dim: int,
        contact_slots: int,
        storage_device: str,
        save_root_body_q: bool,
        save_contact_points_0: bool,
        save_contact_points_1: bool,
        save_contact_impulses: bool = True,
        save_contact_impulse_vectors: bool = True,
        save_contact_identities: bool = True,
    ) -> None:
        import torch

        self.num_envs = num_envs
        self.horizon = horizon
        self.storage_device = storage_device
        self._env_ids = torch.arange(num_envs, device=storage_device, dtype=torch.long)
        self._ptr = torch.zeros(num_envs, device=storage_device, dtype=torch.long)
        self._episode_returns = torch.zeros(num_envs, device=storage_device, dtype=torch.float32)

        self.buffers: dict[str, Any] = {
            "states": torch.zeros((num_envs, horizon, state_dim), device=storage_device, dtype=torch.float32),
            "next_states": torch.zeros((num_envs, horizon, state_dim), device=storage_device, dtype=torch.float32),
            "applied_joint_torque": torch.zeros((num_envs, horizon, torque_dim), device=storage_device, dtype=torch.float32),
            "gravity_dir": torch.zeros((num_envs, horizon, 3), device=storage_device, dtype=torch.float32),
            "contact_normals": torch.zeros(
                (num_envs, horizon, contact_slots, 3), device=storage_device, dtype=torch.float32
            ),
            "contact_depths": torch.zeros((num_envs, horizon, contact_slots), device=storage_device, dtype=torch.float32),
            "contact_thicknesses": torch.zeros(
                (num_envs, horizon, contact_slots), device=storage_device, dtype=torch.float32
            ),
            "contact_counts": torch.zeros((num_envs, horizon), device=storage_device, dtype=torch.int32),
            "dones": torch.zeros((num_envs, horizon), device=storage_device, dtype=torch.bool),
            "terminated": torch.zeros((num_envs, horizon), device=storage_device, dtype=torch.bool),
            "truncated": torch.zeros((num_envs, horizon), device=storage_device, dtype=torch.bool),
        }
        if save_root_body_q:
            self.buffers["root_body_q"] = torch.zeros((num_envs, horizon, 7), device=storage_device, dtype=torch.float32)
        if save_contact_points_0:
            self.buffers["contact_points_0"] = torch.zeros(
                (num_envs, horizon, contact_slots, 3),
                device=storage_device,
                dtype=torch.float32,
            )
        if save_contact_points_1:
            self.buffers["contact_points_1"] = torch.zeros(
                (num_envs, horizon, contact_slots, 3),
                device=storage_device,
                dtype=torch.float32,
            )
        if save_contact_impulses:
            self.buffers["contact_impulses"] = torch.zeros(
                (num_envs, horizon, contact_slots),
                device=storage_device,
                dtype=torch.float32,
            )
        if save_contact_impulse_vectors:
            self.buffers["contact_impulse_vectors"] = torch.zeros(
                (num_envs, horizon, contact_slots, 3),
                device=storage_device,
                dtype=torch.float32,
            )
        if save_contact_identities:
            self.buffers["contact_identities"] = torch.zeros(
                (num_envs, horizon, contact_slots),
                device=storage_device,
                dtype=torch.int32,
            )

    def append(
        self,
        *,
        transition: dict[str, Any],
        rewards: Any,
    ) -> None:
        """Append one batched environment step into the per-env episode buffers."""

        import torch

        step_ids = self._ptr.clone()
        if bool(torch.any(step_ids >= self.horizon)):
            raise RuntimeError(
                f"Episode storage overflowed horizon={self.horizon}. Increase horizon_steps or shorten the episode."
            )

        for name, batch in transition.items():
            if name not in self.buffers:
                continue
            target_buffer = self.buffers[name]
            cast_batch = batch.to(device=self.storage_device, dtype=target_buffer.dtype)
            target_buffer[self._env_ids, step_ids] = cast_batch

        self._episode_returns += rewards.to(device=self.storage_device, dtype=torch.float32)
        self._ptr += 1

    def finalize_done(self, done_mask: Any) -> list[tuple[int, int, float, dict[str, Any]]]:
        """Extract finished trajectories and reset their write pointers."""

        done_ids = done_mask.to(device=self.storage_device, dtype=self.buffers["dones"].dtype).nonzero(as_tuple=False)
        done_ids = done_ids.squeeze(-1)
        completed: list[tuple[int, int, float, dict[str, Any]]] = []
        for env_id in done_ids.tolist():
            length = int(self._ptr[env_id].item())
            trajectory = {name: buffer[env_id, :length].clone() for name, buffer in self.buffers.items()}
            episode_return = float(self._episode_returns[env_id].item())
            completed.append((env_id, length, episode_return, trajectory))
            self._ptr[env_id] = 0
            self._episode_returns[env_id] = 0.0
        return completed


def resolve_checkpoint_path(cfg: CollectorConfig) -> Path:
    """Resolve an explicit checkpoint path or fall back to the latest RL-Games run."""

    from common import checkpoint_is_valid, latest_checkpoint

    checkpoint = cfg.resolved_checkpoint_path()
    if checkpoint is None:
        experiment_root = cfg.logdir.expanduser().resolve() / cfg.experiment_name
        return latest_checkpoint(experiment_root, run_name=cfg.run_name, require_valid=True)

    is_valid, reason = checkpoint_is_valid(checkpoint)
    if not is_valid:
        raise RuntimeError(f"Checkpoint is not readable: {checkpoint}\nReason: {reason}")
    return checkpoint


def build_env_cfg(cfg: CollectorConfig) -> Any:
    """Construct the Isaac Lab environment config and apply local overrides."""

    from common import build_env_cfg as build_local_env_cfg

    env_cfg = build_local_env_cfg(
        task=cfg.task_name,
        device=cfg.device,
        num_envs=cfg.num_envs,
        disable_fabric=cfg.disable_fabric,
        seed=cfg.seed,
    )

    if cfg.sim_dt is not None:
        env_cfg.sim.dt = cfg.sim_dt
    if cfg.decimation is not None:
        env_cfg.decimation = cfg.decimation
    if cfg.episode_length_steps is not None:
        env_cfg.episode_length_s = cfg.episode_length_steps * env_cfg.sim.dt * env_cfg.decimation
    return env_cfg


def resolve_horizon(cfg: CollectorConfig, env: Any) -> int:
    """Pick the stored horizon and make sure it can hold a full episode."""

    requested = cfg.horizon_steps or env.max_episode_length
    if requested < env.max_episode_length:
        raise ValueError(
            "Configured horizon_steps is shorter than the environment episode length. "
            "Use a horizon large enough to hold the full episode, or lower episode_length_steps."
        )
    return requested


def validate_collection_config(cfg: CollectorConfig) -> None:
    """Catch obviously unsafe collection settings before Isaac Sim starts doing heavy work."""

    if cfg.num_envs < 1:
        raise ValueError("num_envs must be at least 1.")
    if cfg.num_trajectories_to_save < 1:
        raise ValueError("num_trajectories_to_save must be at least 1.")

    if cfg.task_name.startswith("Isaac-Factory-") and str(cfg.device).startswith("cuda") and cfg.num_envs > 512:
        raise ValueError(
            "num_envs is the number of parallel simulators, not the total number of trajectories.\n"
            f"You requested num_envs={cfg.num_envs}, which is far too large for this Factory task on GPU and can "
            "cause the process to die before the HDF5 file is flushed.\n"
            "Use a much smaller num_envs such as 32, 64, or 128, and keep num_trajectories large."
        )


def build_agent_cfg(cfg: CollectorConfig, checkpoint: Path) -> dict[str, Any]:
    """Patch the rl_games config the same way the local play.py script does."""

    from common import load_rl_games_cfg, patch_rl_games_cfg

    experiment_root = cfg.logdir.expanduser().resolve() / cfg.experiment_name
    run_name = cfg.run_name or checkpoint.parent.parent.name
    agent_cfg = load_rl_games_cfg(cfg.task_name)
    return patch_rl_games_cfg(
        agent_cfg,
        device=cfg.resolved_policy_device(),
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        experiment_name=cfg.experiment_name,
        experiment_root=experiment_root,
        run_name=run_name,
        clip_obs=cfg.clip_obs,
        clip_actions=cfg.clip_actions,
        games_num=cfg.num_envs,
        deterministic=cfg.deterministic_policy,
        render=not cfg.headless,
        render_sleep=0.0,
        print_stats=False,
    )


def load_player(cfg: CollectorConfig, checkpoint: Path) -> tuple[Any, Any]:
    """Create the RL-Games player and restore the trained policy weights."""

    import torch
    from rl_games.torch_runner import Runner

    from common import configure_rl_games_checkpoint_loading

    configure_rl_games_checkpoint_loading(cfg.resolved_policy_device())

    runner = Runner()
    runner.load(build_agent_cfg(cfg, checkpoint))
    runner.reset()
    player = runner.create_player()

    # ===== POLICY-LOADING PLACEHOLDER ======================================
    # This block mirrors `/home/yiru-wu/Documents/learned_dynamics/play.py`.
    # Replace it if your local rl_games setup restores checkpoints differently.
    checkpoint_data = torch.load(str(checkpoint), map_location=cfg.resolved_policy_device(), weights_only=False)

    load_result = player.model.load_state_dict(checkpoint_data["model"], strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch detected while loading the policy.\n"
            f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
        )

    if player.normalize_input and "running_mean_std" in checkpoint_data:
        player.model.running_mean_std.load_state_dict(checkpoint_data["running_mean_std"])

    env_state = checkpoint_data.get("env_state")
    if player.env is not None and env_state is not None:
        player.env.set_env_state(env_state)
    # =======================================================================

    player.reset()
    player.model.eval()
    return runner, player


def build_transition_batch(
    *,
    current_snapshot: StateSnapshot,
    next_snapshot: StateSnapshot,
    applied_joint_torque: Any,
    contacts: FixedSlotContacts,
    dones: Any,
    terminated: Any,
    truncated: Any,
    cfg: CollectorConfig,
) -> dict[str, Any]:
    """Package one batched simulator step into per-field tensors."""

    transition = {
        "states": current_snapshot.state,  # [N, state_dim] current simulator state for NeRD.
        "next_states": next_snapshot.state,  # [N, state_dim] next simulator state before any auto-reset.
        "applied_joint_torque": applied_joint_torque,  # [N, torque_dim] actual joint torque sent to PhysX.
        "gravity_dir": current_snapshot.gravity_dir,  # [N, 3] gravity direction in world frame.
        "contact_normals": contacts.contact_normals,  # [N, K, 3] normalized contact normals.
        "contact_depths": contacts.contact_depths,  # [N, K] clamped penetration depths.
        "contact_thicknesses": contacts.contact_thicknesses,  # [N, K] constant or simulator thickness.
        "contact_counts": contacts.contact_counts,  # [N] number of populated contact slots.
        "dones": dones,  # [N] episode boundary marker for downstream slicing.
        "terminated": terminated,  # [N] true environment termination flag.
        "truncated": truncated,  # [N] time-limit / truncation flag.
    }
    if cfg.save_root_body_q and current_snapshot.root_body_q is not None:
        transition["root_body_q"] = current_snapshot.root_body_q  # [N, 7] local root pose of the tracked body.
    if cfg.save_contact_points_0:
        transition["contact_points_0"] = contacts.contact_points_0  # [N, K, 3] source-body contact points.
    if cfg.save_contact_points_1:
        transition["contact_points_1"] = contacts.contact_points_1  # [N, K, 3] reconstructed target-body points.
    if cfg.save_contact_impulses:
        transition["contact_impulses"] = contacts.contact_impulses  # [N, K] impulse magnitude per slot.
    if cfg.save_contact_impulse_vectors:
        transition["contact_impulse_vectors"] = contacts.contact_impulse_vectors  # [N, K, 3] impulse vector per slot.
    if cfg.save_contact_identities:
        transition["contact_identities"] = contacts.contact_identities  # [N, K] int32: 0=hole/env, 1=robot.
    return transition


def build_writer(
    *,
    cfg: CollectorConfig,
    horizon: int,
    state_dim: int,
    torque_dim: int,
    state_layout: list[dict[str, Any]],
    step_dt: float,
) -> TrajectoryHDF5Writer:
    """Create the HDF5 writer and describe every dataset up front."""

    import numpy as np

    field_specs: dict[str, tuple[tuple[int, ...], np.dtype | str]] = {
        "states": ((state_dim,), np.float32),
        "next_states": ((state_dim,), np.float32),
        "applied_joint_torque": ((torque_dim,), np.float32),
        "gravity_dir": ((3,), np.float32),
        "contact_normals": ((cfg.contact_slot_count_k, 3), np.float32),
        "contact_depths": ((cfg.contact_slot_count_k,), np.float32),
        "contact_thicknesses": ((cfg.contact_slot_count_k,), np.float32),
        "contact_counts": ((), np.int32),
        "dones": ((), np.bool_),
        "terminated": ((), np.bool_),
        "truncated": ((), np.bool_),
    }
    if cfg.save_root_body_q:
        field_specs["root_body_q"] = ((7,), np.float32)
    if cfg.save_contact_points_0:
        field_specs["contact_points_0"] = ((cfg.contact_slot_count_k, 3), np.float32)
    if cfg.save_contact_points_1:
        field_specs["contact_points_1"] = ((cfg.contact_slot_count_k, 3), np.float32)
    if cfg.save_contact_impulses:
        field_specs["contact_impulses"] = ((cfg.contact_slot_count_k,), np.float32)
    if cfg.save_contact_impulse_vectors:
        field_specs["contact_impulse_vectors"] = ((cfg.contact_slot_count_k, 3), np.float32)
    if cfg.save_contact_identities:
        field_specs["contact_identities"] = ((cfg.contact_slot_count_k,), np.int32)

    metadata = cfg.to_metadata_dict()
    metadata.update(
        {
            "task_name": cfg.task_name,
            "state_dim": state_dim,
            "torque_dim": torque_dim,
            "num_contacts_per_env": cfg.contact_slot_count_k,
            "collection_horizon": horizon,
            "step_dt": float(step_dt),
            "state_layout": state_layout,
        }
    )
    return TrajectoryHDF5Writer(
        path=cfg.resolved_output_path(),
        num_trajectories=cfg.num_trajectories_to_save,
        horizon=horizon,
        field_specs=field_specs,
        metadata=metadata,
        hdf5_cfg=cfg.hdf5,
    )


