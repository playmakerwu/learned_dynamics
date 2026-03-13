import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from common import DEFAULT_TASK, timestamp


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Isaac Lab Peg Insert with RL-Games PPO.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="peg_insert_rlgames")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a .pth checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--horizon_length", type=int, default=None)
    parser.add_argument("--minibatch_size", type=int, default=None)
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
    from rl_games.torch_runner import Runner

    from common import (
        build_env_cfg,
        configure_rl_games_checkpoint_loading,
        latest_checkpoint,
        load_rl_games_cfg,
        mirror_checkpoints,
        patch_rl_games_cfg,
        register_rl_games_env,
        save_yaml,
    )

    run_name = args.run_name or timestamp()
    experiment_root = Path(args.logdir).expanduser().resolve() / args.experiment_name
    run_root = experiment_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

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
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        experiment_name=args.experiment_name,
        experiment_root=experiment_root,
        run_name=run_name,
        clip_obs=args.clip_obs,
        clip_actions=args.clip_actions,
        max_iterations=args.max_iterations,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        horizon_length=args.horizon_length,
        minibatch_size=args.minibatch_size,
        games_num=args.num_envs,
        deterministic=True,
    )

    save_yaml(run_root / "resolved_env_cfg.yaml", env_cfg)
    save_yaml(run_root / "resolved_agent_cfg.yaml", agent_cfg)
    save_yaml(run_root / "cli_args.yaml", vars(args))

    if args.checkpoint:
        configure_rl_games_checkpoint_loading(args.device)

    created_envs = register_rl_games_env(
        args.task,
        env_cfg,
        rl_device=args.device,
        clip_obs=args.clip_obs,
        clip_actions=args.clip_actions,
        render_mode=None,
    )

    runner = None
    try:
        print(f"Training task: {args.task}")
        print(f"Run directory: {run_root}")
        print(f"TensorBoard root: {experiment_root}")
        if args.checkpoint:
            print(f"Resuming from checkpoint: {Path(args.checkpoint).expanduser().resolve()}")

        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        runner.run(
            {
                "train": True,
                "play": False,
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else "",
                "sigma": None,
            }
        )

        try:
            ckpt = latest_checkpoint(run_root)
            print(f"Latest checkpoint: {ckpt}")
        except FileNotFoundError:
            print("Training finished, but no checkpoint was found under the expected run directory.")

        if args.checkpoint_dir:
            dst = Path(args.checkpoint_dir).expanduser().resolve() / args.experiment_name / run_name
            copied = mirror_checkpoints(run_root, dst)
            if copied:
                print(f"Mirrored {len(copied)} checkpoint(s) to: {dst}")
    finally:
        for env in created_envs:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
