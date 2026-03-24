"""Developer note:
This collector is intended to consume the RL-Games policy trained under
`/home/yiru-wu/Documents/learned_dynamics`.

Treat `/home/yiru-wu/Documents/learned_dynamics/play.py` as the primary
reference for how the policy should be created, restored, and stepped for
inference.

Unlike the existing collector path, this script uses the lower-level PhysX
contact report immediate API to gather nonzero per-contact geometry for
NeRD-friendly trajectory datasets.
"""

from __future__ import annotations

import gc
import sys
import traceback
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from nerd_collector.collector import (  # noqa: E402
    CONFIG,
    EpisodeStorage,
    StateAssembler,
    StepResult,
    build_env_cfg,
    build_transition_batch,
    build_writer,
    load_player,
    make_parser,
    resolve_checkpoint_path,
    resolve_horizon,
    resolve_runtime_config,
    validate_collection_config,
)
from nerd_collector.physx_contact_report import PhysXContactReportExtractor  # noqa: E402


def run_contact_preflight(
    *,
    direct_env: Any,
    contact_extractor: PhysXContactReportExtractor,
    lateral_x_offset: float = 0.004,
    physics_steps: int = 4,
) -> None:
    """Force a short peg-vs-socket overlap and fail fast if low-level contacts stay zero."""

    import torch

    held_asset = contact_extractor.source_asset
    fixed_asset = contact_extractor.target_asset
    env_ids = torch.arange(direct_env.num_envs, device=direct_env.device, dtype=torch.long)
    held_pose = fixed_asset.data.root_link_pose_w.clone()
    held_pose[:, 0] += float(lateral_x_offset)
    held_vel = torch.zeros_like(held_asset.data.root_link_vel_w)

    held_asset.write_root_pose_to_sim(held_pose, env_ids=env_ids)
    held_asset.write_root_velocity_to_sim(held_vel, env_ids=env_ids)
    held_asset.reset()
    direct_env.scene.write_data_to_sim()

    contact_extractor.begin_step()
    for _ in range(max(1, physics_steps)):
        direct_env.sim.step(render=False)
        contact_extractor.capture_substep_reports()
        direct_env.scene.update(dt=direct_env.physics_dt)
    slots = contact_extractor.end_step()

    if int(slots.contact_counts.sum().item()) <= 0 or contact_extractor.last_debug_stats.matching_contact_count <= 0:
        raise RuntimeError(
            "Low-level PhysX contact-report preflight failed: even an explicit peg-vs-socket forced overlap produced "
            "zero matching contacts. For this task, the validated working path is CPU simulation. "
            "Please rerun with '--device cpu' and optionally keep '--policy_device cuda:0'."
        )


def step_direct_env_with_contact_reports(
    *,
    direct_env: Any,
    wrapped_env: Any,
    action: Any,
    snapshot_fn: Any,
    obs_to_torch_fn: Any,
    contact_extractor: PhysXContactReportExtractor,
) -> tuple[StepResult, Any]:
    """Step a DirectRLEnv while collecting low-level PhysX contact reports.

    Direct simulator reads:
    - PhysX contact report headers and per-contact records during each physics substep

    Derived values:
    - fixed-slot NeRD contact tensors created from the accumulated raw reports
    """

    action = action.to(direct_env.device)
    contact_extractor.begin_step()
    direct_env._pre_physics_step(action)
    is_rendering = direct_env.sim.has_gui() or direct_env.sim.has_rtx_sensors()

    for _ in range(direct_env.cfg.decimation):
        direct_env._sim_step_counter += 1
        direct_env._apply_action()
        direct_env.scene.write_data_to_sim()
        direct_env.sim.step(render=False)
        # Direct low-level PhysX report read from the just-finished substep.
        contact_extractor.capture_substep_reports()
        if direct_env._sim_step_counter % direct_env.cfg.sim.render_interval == 0 and is_rendering:
            direct_env.sim.render()
        direct_env.scene.update(dt=direct_env.physics_dt)

    contacts = contact_extractor.end_step()

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
    return (
        StepResult(
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
            terminated=terminated,
            truncated=truncated,
            extras=extras,
            next_snapshot=next_snapshot,
        ),
        contacts,
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
        if not isinstance(direct_env, DirectRLEnv):
            raise RuntimeError(f"Expected a DirectRLEnv for low-level contact collection, got: {type(direct_env)!r}")

        obses = player.env_reset(player.env)
        batch_size = player.get_batch_size(obses, batch_size=1)
        print(f"Detected policy batch size: {batch_size}", flush=True)
        if player.is_rnn:
            player.init_rnn()

        assembler = StateAssembler(direct_env, cfg)
        contact_extractor = PhysXContactReportExtractor(direct_env, cfg)
        run_contact_preflight(direct_env=direct_env, contact_extractor=contact_extractor)
        contact_extractor.reset_statistics()
        obses = player.env_reset(player.env)
        if player.is_rnn:
            player.init_rnn()

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
        if getattr(writer, "_file", None) is not None:
            writer._file.attrs["contact_api_path"] = "physx_contact_report_immediate_api"
            writer._file.attrs["contact_points_0_semantics"] = "direct PhysX contact_report position"
            writer._file.attrs["contact_points_1_semantics"] = "reconstructed as point0 + normal * depth"
            writer._file.attrs["contact_normals_semantics"] = "force direction on the configured source asset"
            writer._file.attrs["contact_depth_semantics"] = "clamp(max(0, -separation), max=max_depth_clamp)"
            writer._file.attrs["contact_extractor_note"] = (
                "Contacts are aggregated from all physics substeps inside one Isaac Lab env step."
            )
            writer._file.attrs["contact_impulses_semantics"] = (
                "impulse magnitude (norm of PhysX impulse vector) per contact slot"
            )
            writer._file.attrs["contact_impulse_vectors_semantics"] = (
                "3D impulse vector from PhysX contact report, direction follows contact_normals convention"
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
        print(f"Enabled source contact report prims: {contact_extractor.enabled_source_contact_report_prims}", flush=True)
        print(f"Enabled target contact report prims: {contact_extractor.enabled_target_contact_report_prims}", flush=True)

        with torch.inference_mode():
            while writer.remaining_capacity > 0:
                current_snapshot = assembler.capture()
                policy_action = player.get_action(obses, is_deterministic=cfg.deterministic_policy)
                applied_action = policy_action.detach()
                if cfg.action_noise_std > 0.0:
                    applied_action = applied_action + cfg.action_noise_std * torch.randn_like(applied_action)
                applied_action = torch.clamp(applied_action, -cfg.clip_actions, cfg.clip_actions)

                step_result, contacts = step_direct_env_with_contact_reports(
                    direct_env=direct_env,
                    wrapped_env=wrapped_env,
                    action=applied_action,
                    snapshot_fn=assembler.capture,
                    obs_to_torch_fn=player.obs_to_torch,
                    contact_extractor=contact_extractor,
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
                    stats = contact_extractor.last_debug_stats
                    print(
                        f"env_step={total_env_steps:06d} | trajectories_written={writer.num_written}/{cfg.num_trajectories_to_save} "
                        f"| matching_contacts={stats.matching_contact_count} slot_contacts={stats.slot_contact_count} "
                        f"| max_depth={stats.max_depth:.6f}",
                        flush=True,
                    )

        if contact_extractor.total_matching_contacts_seen <= 0:
            raise RuntimeError(
                "The low-level PhysX contact report collector finished without seeing any matching peg-vs-socket "
                "contacts. The saved dataset would be invalid for NeRD."
            )

        print(
            f"Finished writing {writer.num_written} trajectories and {writer.total_transitions} transitions.",
            flush=True,
        )
        print(
            f"Observed {contact_extractor.total_matching_contacts_seen} matching raw PhysX contacts across "
            f"{contact_extractor.total_env_steps_with_contacts} env steps.",
            flush=True,
        )
    except BaseException:
        print("Collector failed with the following exception:", flush=True)
        print(traceback.format_exc(), flush=True)
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
