import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from common import DEFAULT_TASK


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained Isaac Lab Peg Insert RL-Games policy.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="peg_insert_rlgames")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run folder to constrain checkpoint lookup.")
    parser.add_argument("--games_num", type=int, default=32, help="How many episodes RL-Games should play before exit.")
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
        latest_checkpoint,
        load_rl_games_cfg,
        patch_rl_games_cfg,
        register_rl_games_env,
    )

    experiment_root = Path(args.logdir).expanduser().resolve() / args.experiment_name

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint = latest_checkpoint(experiment_root, run_name=args.run_name)

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
        run_name=args.run_name or checkpoint.parent.parent.name,
        clip_obs=args.clip_obs,
        clip_actions=args.clip_actions,
        games_num=args.games_num,
        deterministic=True,
    )

    created_envs = register_rl_games_env(
        args.task,
        env_cfg,
        rl_device=args.device,
        clip_obs=args.clip_obs,
        clip_actions=args.clip_actions,
        render_mode=None,
    )

    try:
        print(f"Playing task: {args.task}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Num envs: {args.num_envs}")
        print("RL-Games will print episode statistics during evaluation.")

        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        runner.run(
            {
                "train": False,
                "play": True,
                "checkpoint": str(checkpoint),
                "sigma": None,
            }
        )
    finally:
        for env in created_envs:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
