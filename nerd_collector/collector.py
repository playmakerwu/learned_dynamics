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
import gc
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from isaaclab.app import AppLauncher

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from nerd_collector.config import CONFIG, CollectorConfig
    from nerd_collector.contact_utils import FixedSlotContacts, assign_contact_slots, empty_fixed_slot_contacts
    from nerd_collector.hdf5_utils import TrajectoryHDF5Writer
else:
    from .config import CONFIG, CollectorConfig
    from .contact_utils import FixedSlotContacts, assign_contact_slots, empty_fixed_slot_contacts
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


class PhysXContactExtractor:
    """Read raw PhysX contacts and map them into fixed K contact slots per env."""

    def __init__(self, env: Any, cfg: CollectorConfig):
        self.env = env
        self.cfg = cfg
        self.source_asset = env.scene.articulations[cfg.contact_source_asset_name]
        self.target_asset = env.scene.articulations[cfg.contact_target_asset_name]
        self.available = False
        self._warned_runtime = False
        self.initialization_error: Exception | None = None

        if not cfg.enable_raw_contact_extraction:
            self.initialization_error = RuntimeError(
                "Raw contact extraction is disabled by config; contact tensors will be zero-filled."
            )
            return

        if (
            str(env.device).startswith("cuda")
            and cfg.task_name.startswith("Isaac-Factory-")
            and cfg.contact_target_asset_name == "fixed_asset"
            and not cfg.allow_unsupported_gpu_contact_filter
        ):
            self.initialization_error = RuntimeError(
                "Skipping raw filtered PhysX contact extraction on GPU for the Factory FixedAsset collider. "
                "This task emits the native warning "
                "'GPU contact filter for collider ... is not supported' and may terminate before writing data. "
                "The collector will keep the contact slot tensors zero-filled instead. "
                "If you need raw contact slots, try simulating on CPU with policy on GPU."
            )
            return

        try:
            import carb
            from isaacsim.core.simulation_manager import SimulationManager

            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
            physics_sim_view = SimulationManager.get_physics_sim_view()
            source_glob = _articulation_contact_glob(self.source_asset)
            target_glob = _articulation_contact_glob(self.target_asset)
            self.contact_view = physics_sim_view.create_rigid_contact_view(
                source_glob,
                filter_patterns=[target_glob],
                max_contact_data_count=(
                    self.cfg.max_contact_data_count_per_prim * max(1, len(self.source_asset.body_names)) * env.num_envs
                ),
            )
            if getattr(self.contact_view, "_backend", None) is None:
                raise RuntimeError("PhysX failed to create the rigid contact view backend.")

            self.num_source_bodies = len(self.source_asset.body_names)
            self.num_filter_bodies = int(self.contact_view.filter_count)
            self.available = True
        except Exception as exc:
            self.initialization_error = exc
            if self.cfg.strict_contact_extraction:
                raise

    def capture(self) -> FixedSlotContacts:
        """Capture current raw contacts and project them into fixed K slots."""

        if not self.available:
            return empty_fixed_slot_contacts(
                num_envs=self.env.num_envs,
                k=self.cfg.contact_slot_count_k,
                device=self.env.device,
                dtype=self.source_asset.data.root_link_pose_w.dtype,
                contact_thickness=self.cfg.contact_thickness,
            )

        try:
            force_buffer, point_buffer, normal_buffer, separation_buffer, count_buffer, start_buffer = (
                self.contact_view.get_contact_data(dt=self.env.physics_dt)
            )
            return assign_contact_slots(
                force_magnitudes=force_buffer,
                contact_points_0=point_buffer,
                contact_normals=normal_buffer,
                separations=separation_buffer,
                buffer_count=count_buffer,
                buffer_start_indices=start_buffer,
                num_envs=self.env.num_envs,
                num_source_bodies=self.num_source_bodies,
                num_filter_bodies=max(1, self.num_filter_bodies),
                k=self.cfg.contact_slot_count_k,
                max_depth=self.cfg.max_depth_clamp,
                contact_thickness=self.cfg.contact_thickness,
            )
        except Exception:
            if self.cfg.strict_contact_extraction:
                raise
            if not self._warned_runtime:
                print("Warning: raw PhysX contact extraction failed; contact tensors will be zero-filled.", flush=True)
                traceback.print_exc()
                self._warned_runtime = True
            return empty_fixed_slot_contacts(
                num_envs=self.env.num_envs,
                k=self.cfg.contact_slot_count_k,
                device=self.env.device,
                dtype=self.source_asset.data.root_link_pose_w.dtype,
                contact_thickness=self.cfg.contact_thickness,
            )


def _articulation_contact_glob(articulation: Any) -> str:
    if not articulation.body_names:
        raise RuntimeError(f"Articulation at '{articulation.cfg.prim_path}' has no bodies to build a contact view from.")
    body_names_regex = r"(" + "|".join(articulation.body_names) + r")"
    return f"{articulation.cfg.prim_path}/{body_names_regex}".replace(".*", "*")


@dataclass(slots=True)
class StepResult:
    next_obs: Any
    rewards: Any
    dones: Any
    terminated: Any
    truncated: Any
    extras: dict[str, Any]
    next_snapshot: StateSnapshot


class EpisodeStorage:
    """GPU/CPU staging buffers for one in-flight episode per environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        horizon: int,
        state_dim: int,
        action_dim: int,
        contact_slots: int,
        storage_device: str,
        save_root_body_q: bool,
        save_contact_points_0: bool,
        save_contact_points_1: bool,
        save_contact_impulses: bool = True,
        save_contact_impulse_vectors: bool = True,
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
            "joint_acts": torch.zeros((num_envs, horizon, action_dim), device=storage_device, dtype=torch.float32),
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


def step_direct_env_for_collection(
    *,
    direct_env: Any,
    wrapped_env: Any,
    action: Any,
    snapshot_fn: Callable[[], StateSnapshot],
    obs_to_torch_fn: Callable[[Any], Any],
) -> StepResult:
    """Step a DirectRLEnv while capturing terminal next_states before auto-reset."""

    action = action.to(direct_env.device)
    direct_env._pre_physics_step(action)
    is_rendering = direct_env.sim.has_gui() or direct_env.sim.has_rtx_sensors()

    for _ in range(direct_env.cfg.decimation):
        direct_env._sim_step_counter += 1
        direct_env._apply_action()
        direct_env.scene.write_data_to_sim()
        direct_env.sim.step(render=False)
        if direct_env._sim_step_counter % direct_env.cfg.sim.render_interval == 0 and is_rendering:
            direct_env.sim.render()
        direct_env.scene.update(dt=direct_env.physics_dt)

    direct_env.episode_length_buf += 1
    direct_env.common_step_counter += 1
    direct_env.reset_terminated[:], direct_env.reset_time_outs[:] = direct_env._get_dones()
    direct_env.reset_buf = direct_env.reset_terminated | direct_env.reset_time_outs
    direct_env.reward_buf = direct_env._get_rewards()

    # Snapshot before reset so terminal transitions see the actual terminal state.
    next_snapshot = snapshot_fn()

    reset_env_ids = direct_env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        direct_env._reset_idx(reset_env_ids)
        if direct_env.sim.has_rtx_sensors() and direct_env.cfg.num_rerenders_on_reset > 0:
            for _ in range(direct_env.cfg.num_rerenders_on_reset):
                direct_env.sim.render()

    if direct_env.cfg.events and "interval" in direct_env.event_manager.available_modes:
        direct_env.event_manager.apply(mode="interval", dt=direct_env.step_dt)

    direct_env.obs_buf = direct_env._get_observations()
    if direct_env.cfg.observation_noise_model:
        direct_env.obs_buf["policy"] = direct_env._observation_noise_model(direct_env.obs_buf["policy"])

    # Match RL-Games' BasePlayer.env_step(): convert wrapper output back into the
    # tensor format expected by player.get_action().
    next_obs = obs_to_torch_fn(wrapped_env._process_obs(direct_env.obs_buf))
    rewards = direct_env.reward_buf.to(device=wrapped_env._rl_device)
    terminated = direct_env.reset_terminated.to(device=wrapped_env._rl_device)
    truncated = direct_env.reset_time_outs.to(device=wrapped_env._rl_device)
    dones = (direct_env.reset_terminated | direct_env.reset_time_outs).to(device=wrapped_env._rl_device)
    extras = {
        key: value.to(device=wrapped_env._rl_device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in direct_env.extras.items()
    }
    if "log" in extras:
        extras["episode"] = extras.pop("log")
    return StepResult(
        next_obs=next_obs,
        rewards=rewards,
        dones=dones,
        terminated=terminated,
        truncated=truncated,
        extras=extras,
        next_snapshot=next_snapshot,
    )


def build_transition_batch(
    *,
    current_snapshot: StateSnapshot,
    next_snapshot: StateSnapshot,
    actions: Any,
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
        "joint_acts": actions,  # [N, act_dim] action actually applied to Isaac Lab.
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
    return transition


def build_writer(
    *,
    cfg: CollectorConfig,
    horizon: int,
    state_dim: int,
    action_dim: int,
    state_layout: list[dict[str, Any]],
    step_dt: float,
) -> TrajectoryHDF5Writer:
    """Create the HDF5 writer and describe every dataset up front."""

    import numpy as np

    field_specs: dict[str, tuple[tuple[int, ...], np.dtype | str]] = {
        "states": ((state_dim,), np.float32),
        "next_states": ((state_dim,), np.float32),
        "joint_acts": ((action_dim,), np.float32),
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

    metadata = cfg.to_metadata_dict()
    metadata.update(
        {
            "task_name": cfg.task_name,
            "state_dim": state_dim,
            "act_dim": action_dim,
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


def main() -> None:
    parser = make_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    cfg = resolve_runtime_config(CONFIG, args, argv)
    validate_collection_config(cfg)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    created_envs: list[Any] = []
    runner = None
    player = None
    writer = None

    try:
        import isaaclab_tasks  # noqa: F401
        import torch
        from isaaclab.envs import DirectRLEnv

        from common import ensure_task_assets_available, register_rl_games_env

        checkpoint = resolve_checkpoint_path(cfg)
        env_cfg = build_env_cfg(cfg)
        ensure_task_assets_available(cfg.task_name, env_cfg)

        created_envs = register_rl_games_env(
            cfg.task_name,
            env_cfg,
            rl_device=cfg.resolved_policy_device(),
            clip_obs=cfg.clip_obs,
            clip_actions=cfg.clip_actions,
            render_mode="human" if not cfg.headless else None,
        )

        runner, player = load_player(cfg, checkpoint)
        wrapped_env = getattr(player.env, "env", created_envs[0] if created_envs else None)
        if wrapped_env is None:
            raise RuntimeError("Unable to recover the RL-Games wrapped environment from the player.")
        direct_env = wrapped_env.unwrapped

        obses = player.env_reset(player.env)
        batch_size = player.get_batch_size(obses, batch_size=1)
        print(f"Detected policy batch size: {batch_size}", flush=True)
        if player.is_rnn:
            player.init_rnn()

        assembler = StateAssembler(direct_env, cfg)
        contact_extractor = PhysXContactExtractor(direct_env, cfg)
        if not contact_extractor.available and contact_extractor.initialization_error is not None:
            print(
                "Warning: raw PhysX contact view could not be initialized; contact tensors will be zero-filled.\n"
                f"Reason: {contact_extractor.initialization_error}",
                flush=True,
            )

        horizon = resolve_horizon(cfg, direct_env)
        action_dim = int(wrapped_env.action_space.shape[0])
        storage_device = direct_env.device if (cfg.save_on_gpu_first and "cuda" in str(direct_env.device)) else "cpu"
        writer = build_writer(
            cfg=cfg,
            horizon=horizon,
            state_dim=assembler.state_dim,
            action_dim=action_dim,
            state_layout=assembler.layout_metadata,
            step_dt=float(direct_env.step_dt),
        )
        print("HDF5 writer initialized.", flush=True)
        episode_storage = EpisodeStorage(
            num_envs=direct_env.num_envs,
            horizon=horizon,
            state_dim=assembler.state_dim,
            action_dim=action_dim,
            contact_slots=cfg.contact_slot_count_k,
            storage_device=storage_device,
            save_root_body_q=cfg.save_root_body_q,
            save_contact_points_0=cfg.save_contact_points_0,
            save_contact_points_1=cfg.save_contact_points_1,
            save_contact_impulses=cfg.save_contact_impulses,
            save_contact_impulse_vectors=cfg.save_contact_impulse_vectors,
        )

        total_env_steps = 0
        trajectories_written = 0
        print(f"Task: {cfg.task_name}", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"Num envs: {cfg.num_envs}", flush=True)
        print(f"Simulation device: {cfg.device}", flush=True)
        print(f"Policy device: {cfg.resolved_policy_device()}", flush=True)
        print(f"Output path: {cfg.resolved_output_path()}", flush=True)
        print(f"Horizon: {horizon}", flush=True)
        print(f"Contact slots K: {cfg.contact_slot_count_k}", flush=True)

        with torch.inference_mode():
            while writer.remaining_capacity > 0:
                current_snapshot = assembler.capture()
                contacts = contact_extractor.capture()

                policy_action = player.get_action(obses, is_deterministic=cfg.deterministic_policy)
                applied_action = policy_action.detach()
                if cfg.action_noise_std > 0.0:
                    applied_action = applied_action + cfg.action_noise_std * torch.randn_like(applied_action)
                applied_action = torch.clamp(applied_action, -cfg.clip_actions, cfg.clip_actions)

                if cfg.use_manual_direct_step and isinstance(direct_env, DirectRLEnv):
                    step_result = step_direct_env_for_collection(
                        direct_env=direct_env,
                        wrapped_env=wrapped_env,
                        action=applied_action,
                        snapshot_fn=assembler.capture,
                        obs_to_torch_fn=player.obs_to_torch,
                    )
                else:
                    next_obs, rewards, dones, extras = player.env_step(player.env, applied_action)
                    next_snapshot = assembler.capture()
                    terminated = direct_env.reset_terminated.to(device=wrapped_env._rl_device)
                    truncated = direct_env.reset_time_outs.to(device=wrapped_env._rl_device)
                    step_result = StepResult(
                        next_obs=next_obs,
                        rewards=rewards,
                        dones=dones,
                        terminated=terminated,
                        truncated=truncated,
                        extras=extras,
                        next_snapshot=next_snapshot,
                    )

                transition = build_transition_batch(
                    current_snapshot=current_snapshot,
                    next_snapshot=step_result.next_snapshot,
                    actions=applied_action,
                    contacts=contacts,
                    dones=step_result.dones,
                    terminated=step_result.terminated,
                    truncated=step_result.truncated,
                    cfg=cfg,
                )
                episode_storage.append(transition=transition, rewards=step_result.rewards)

                if player.is_rnn and player.states is not None:
                    done_indices = step_result.dones.nonzero(as_tuple=False).squeeze(-1)
                    for state in player.states:
                        state[:, done_indices, :] = 0.0

                completed = episode_storage.finalize_done(step_result.dones)
                for env_id, length, episode_return, trajectory in completed:
                    if writer.remaining_capacity <= 0:
                        break
                    writer.append_trajectory(
                        trajectory,
                        length=length,
                        env_id=env_id,
                        episode_return=episode_return,
                    )
                    trajectories_written += 1
                    print(
                        f"trajectory={trajectories_written:04d} | env={env_id} | steps={length} | return={episode_return:.3f}",
                        flush=True,
                    )

                obses = step_result.next_obs
                total_env_steps += 1
                if cfg.log_every_steps > 0 and total_env_steps % cfg.log_every_steps == 0:
                    print(
                        f"env_step={total_env_steps:06d} | "
                        f"trajectories_written={writer.num_written}/{cfg.num_trajectories_to_save}",
                        flush=True,
                    )

        print(
            f"Finished writing {writer.num_written} trajectories and {writer.total_transitions} transitions.",
            flush=True,
        )
    except BaseException:
        print("Collector failed with the following exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        player = None
        runner = None
        gc.collect()
        for env in created_envs:
            try:
                env.close()
            except Exception:
                pass
        created_envs.clear()
        gc.collect()
        simulation_app.close()


if __name__ == "__main__":
    main()
