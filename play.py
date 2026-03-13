import argparse
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from common import DEFAULT_TASK


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained Isaac Lab Peg Insert RL-Games policy.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument(
        "--policy_device",
        type=str,
        default=None,
        help="Device for policy inference. Defaults to the same value as --device. Use cpu to reduce GPU memory usage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="peg_insert_rlgames")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run folder to constrain checkpoint lookup.")
    parser.add_argument("--games_num", type=int, default=32, help="How many episodes RL-Games should play before exit.")
    parser.add_argument("--loop_forever", action="store_true", help="Ignore games_num and keep playing until Ctrl+C.")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional cap on total environment steps. 0 means no cap.")
    parser.add_argument("--print_every", type=int, default=100, help="Print a short progress line every N environment steps.")
    parser.add_argument(
        "--print_action_stats",
        action="store_true",
        help="Print action magnitude statistics together with the periodic progress line.",
    )
    parser.add_argument(
        "--camera_eye",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional viewer camera eye position for a closer manual view.",
    )
    parser.add_argument(
        "--camera_lookat",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional viewer camera look-at target. Must be used together with --camera_eye.",
    )
    parser.add_argument("--render_sleep", type=float, default=0.02, help="Sleep between rendered frames in play mode.")
    parser.add_argument(
        "--pause_before_play",
        type=float,
        default=0.0,
        help="Optional pause after the env is created and before policy stepping starts.",
    )
    parser.add_argument(
        "--hold_on_exit",
        type=float,
        default=0.0,
        help="Optional pause before closing the simulation app so the last frame stays visible.",
    )
    parser.add_argument("--clip_obs", type=float, default=10.0)
    parser.add_argument("--clip_actions", type=float, default=1.0)
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab_tasks  # noqa: F401
    import torch
    from rl_games.torch_runner import Runner

    from common import (
        build_env_cfg,
        checkpoint_is_valid,
        configure_rl_games_checkpoint_loading,
        list_checkpoints,
        latest_checkpoint,
        load_rl_games_cfg,
        patch_rl_games_cfg,
        register_rl_games_env,
    )

    experiment_root = Path(args.logdir).expanduser().resolve() / args.experiment_name
    policy_device = args.policy_device or args.device

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        is_valid, reason = checkpoint_is_valid(checkpoint)
        if not is_valid:
            sibling_candidates = []
            if checkpoint.parent.exists():
                for candidate in list_checkpoints(checkpoint.parent):
                    ok, _ = checkpoint_is_valid(candidate)
                    if ok:
                        sibling_candidates.append(candidate)
                    if len(sibling_candidates) >= 5:
                        break

            message_lines = [
                f"Checkpoint is not readable: {checkpoint}",
                f"Reason: {reason}",
            ]
            if sibling_candidates:
                message_lines.append("Try one of these valid checkpoints instead:")
                message_lines.extend(str(candidate) for candidate in sibling_candidates)
            raise RuntimeError("\n".join(message_lines))
    else:
        checkpoint = latest_checkpoint(experiment_root, run_name=args.run_name, require_valid=True)

    configure_rl_games_checkpoint_loading(policy_device)

    env_cfg = build_env_cfg(
        task=args.task,
        device=args.device,
        num_envs=args.num_envs,
        disable_fabric=args.disable_fabric,
        seed=args.seed,
    )

    agent_cfg = load_rl_games_cfg(args.task)
    agent_cfg = patch_rl_games_cfg(
        agent_cfg,
        device=policy_device,
        num_envs=args.num_envs,
        seed=args.seed,
        experiment_name=args.experiment_name,
        experiment_root=experiment_root,
        run_name=args.run_name or checkpoint.parent.parent.name,
        clip_obs=args.clip_obs,
        clip_actions=args.clip_actions,
        games_num=args.games_num,
        deterministic=True,
        render=not args.headless,
        render_sleep=args.render_sleep,
        print_stats=True,
    )

    created_envs = register_rl_games_env(
        args.task,
        env_cfg,
        rl_device=policy_device,
        clip_obs=args.clip_obs,
        clip_actions=args.clip_actions,
        render_mode="human" if not args.headless else None,
    )

    try:
        if (args.camera_eye is None) != (args.camera_lookat is None):
            raise ValueError("Please provide both --camera_eye and --camera_lookat together.")

        print(f"Playing task: {args.task}", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"Num envs: {args.num_envs}", flush=True)
        print(f"Simulation device: {args.device}", flush=True)
        print(f"Policy device: {policy_device}", flush=True)
        print(f"Games to play: {args.games_num}", flush=True)
        print(f"Rendering enabled: {not args.headless}", flush=True)
        print(f"Loop forever: {args.loop_forever}", flush=True)

        runner = Runner()
        print("Runner created.", flush=True)
        runner.load(agent_cfg)
        print("Runner config loaded.", flush=True)
        runner.reset()
        print("Runner reset complete.", flush=True)
        player = runner.create_player()
        print("Player created.", flush=True)

        print(f"Loading checkpoint dictionary with torch.load(map_location={policy_device})...", flush=True)
        checkpoint_data = torch.load(str(checkpoint), map_location=policy_device, weights_only=False)
        print(f"Checkpoint keys: {sorted(checkpoint_data.keys())}", flush=True)

        print("Loading model state dict...", flush=True)
        load_result = player.model.load_state_dict(checkpoint_data["model"], strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        if missing_keys or unexpected_keys:
            print(f"Missing keys: {missing_keys}", flush=True)
            print(f"Unexpected keys: {unexpected_keys}", flush=True)
            raise RuntimeError("Checkpoint/model mismatch detected while loading the policy.")

        if player.normalize_input and "running_mean_std" in checkpoint_data:
            print("Loading standalone running_mean_std state...", flush=True)
            player.model.running_mean_std.load_state_dict(checkpoint_data["running_mean_std"])

        env_state = checkpoint_data.get("env_state", None)
        if player.env is not None and env_state is not None:
            print("Restoring env_state from checkpoint...", flush=True)
            player.env.set_env_state(env_state)

        print("Checkpoint restore complete.", flush=True)
        player.reset()
        print("Player reset complete.", flush=True)
        player.model.eval()
        print("Player model switched to eval mode.", flush=True)
        render_env = created_envs[0].env if (created_envs and not args.headless) else None

        obses = player.env_reset(player.env)
        print("Environment reset through player wrapper complete.", flush=True)
        if not args.headless and args.camera_eye is not None and created_envs:
            try:
                created_envs[0].env.unwrapped.sim.set_camera_view(
                    eye=tuple(args.camera_eye),
                    target=tuple(args.camera_lookat),
                )
                print(
                    f"Viewer camera set to eye={tuple(args.camera_eye)} target={tuple(args.camera_lookat)}.",
                    flush=True,
                )
            except Exception:
                print("Failed to set viewer camera explicitly.", flush=True)
                traceback.print_exc()
        if render_env is not None:
            try:
                render_env.render()
            except TypeError:
                render_env.render(mode="human")
        batch_size = player.get_batch_size(obses, batch_size=1)
        print(f"Detected batch size: {batch_size}", flush=True)
        if player.is_rnn:
            player.init_rnn()
            print("RNN state initialized.", flush=True)

        episode_rewards = torch.zeros(batch_size, dtype=torch.float32)
        episode_steps = torch.zeros(batch_size, dtype=torch.float32)
        episodes_finished = 0
        total_steps = 0

        def summarize_episode_info(infos, env_id: int) -> str:
            if not isinstance(infos, dict):
                return ""
            episode_info = infos.get("episode")
            if not isinstance(episode_info, dict):
                return ""

            parts = []
            for key in ("successes", "success", "reward", "length"):
                if key not in episode_info:
                    continue
                value = episode_info[key]
                if hasattr(value, "detach"):
                    value = value.detach().cpu()
                if hasattr(value, "numel"):
                    if value.numel() == 1:
                        parts.append(f"{key}={float(value.item()):.3f}")
                    elif env_id < int(value.shape[0]):
                        parts.append(f"{key}={float(value[env_id].item()):.3f}")
                else:
                    parts.append(f"{key}={value}")
            return " | " + " ".join(parts) if parts else ""

        if args.pause_before_play > 0.0:
            print(
                f"Pausing for {args.pause_before_play:.2f}s before stepping so you can inspect the scene...",
                flush=True,
            )
            time.sleep(args.pause_before_play)

        print("Starting manual player loop. Press Ctrl+C to stop.", flush=True)
        try:
            while True:
                action = player.get_action(obses, is_deterministic=True)
                obses, rewards, dones, infos = player.env_step(player.env, action)

                rewards = rewards.to(dtype=torch.float32)
                dones = dones.to(dtype=torch.bool)
                episode_rewards += rewards
                episode_steps += 1
                total_steps += 1

                if render_env is not None:
                    try:
                        render_env.render()
                    except TypeError:
                        render_env.render(mode="human")
                    if args.render_sleep > 0.0:
                        time.sleep(args.render_sleep)

                if args.print_every > 0 and total_steps % args.print_every == 0:
                    message = (
                        f"step={total_steps:06d} | "
                        f"live_reward_mean={episode_rewards.mean().item():.3f} | "
                        f"done_count={int(dones.sum().item())}"
                    )
                    if args.print_action_stats:
                        action_t = action.detach().float().cpu()
                        message += (
                            f" | action_mean={action_t.mean().item():.4f}"
                            f" | action_abs_max={action_t.abs().max().item():.4f}"
                            f" | action_l2={action_t.norm().item():.4f}"
                        )
                    print(message, flush=True)

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.numel() > 0:
                    if player.is_rnn and player.states is not None:
                        for state in player.states:
                            state[:, done_indices, :] = 0.0

                    for env_id in done_indices.tolist():
                        episodes_finished += 1
                        info_suffix = summarize_episode_info(infos, env_id)
                        print(
                            f"episode={episodes_finished:04d} | "
                            f"env={env_id} | "
                            f"reward={episode_rewards[env_id].item():.3f} | "
                            f"steps={int(episode_steps[env_id].item())}"
                            f"{info_suffix}",
                            flush=True,
                        )

                    episode_rewards[done_indices] = 0.0
                    episode_steps[done_indices] = 0.0

                if not args.loop_forever and args.games_num > 0 and episodes_finished >= args.games_num:
                    print(f"Stopping after reaching requested episode count: {episodes_finished}", flush=True)
                    break
                if args.max_steps > 0 and total_steps >= args.max_steps:
                    print(f"Stopping after reaching requested step count: {total_steps}", flush=True)
                    break
        except KeyboardInterrupt:
            print("Interrupted by user. Closing player loop.", flush=True)
        except Exception:
            print("Exception inside manual player loop:", flush=True)
            traceback.print_exc()
            raise
    except Exception:
        print("Exception while setting up play mode:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if args.hold_on_exit > 0.0:
            print(
                f"Holding final frame for {args.hold_on_exit:.2f}s before shutdown...",
                flush=True,
            )
            time.sleep(args.hold_on_exit)
        for env in created_envs:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
