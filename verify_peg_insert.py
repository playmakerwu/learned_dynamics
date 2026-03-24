import argparse
import gc

from isaaclab.app import AppLauncher

from common import DEFAULT_TASK


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Isaac Lab Peg Insert with batched stepping.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def summarize_obs(obs):
    import torch

    if torch.is_tensor(obs):
        return {
            "shape": list(obs.shape),
            "dtype": str(obs.dtype),
            "device": str(obs.device),
        }
    if isinstance(obs, dict):
        return {k: summarize_obs(v) for k, v in obs.items()}
    return str(type(obs))


def infer_action_spec(action_space, num_envs, device):
    import torch

    if not hasattr(action_space, "shape") or action_space.shape is None:
        raise RuntimeError(f"Expected a Box-like action space, got: {action_space}")

    full_shape = tuple(action_space.shape)
    is_batched = len(full_shape) >= 1 and full_shape[0] == num_envs
    per_env_shape = tuple(full_shape[1:]) if is_batched else full_shape
    if not per_env_shape:
        raise RuntimeError(f"Unable to infer per-env action shape from: {full_shape}")

    low = torch.as_tensor(action_space.low, device=device, dtype=torch.float32)
    high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32)
    if is_batched:
        low = low[0]
        high = high[0]

    return {
        "full_shape": full_shape,
        "per_env_shape": per_env_shape,
        "is_batched": is_batched,
        "low": low,
        "high": high,
    }


def main():
    parser = make_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from common import build_env_cfg, ensure_task_assets_available

    env = None
    try:
        env_cfg = build_env_cfg(
            task=args.task,
            device=args.device,
            num_envs=args.num_envs,
            disable_fabric=args.disable_fabric,
            seed=args.seed,
        )
        ensure_task_assets_available(args.task, env_cfg)

        env = gym.make(args.task, cfg=env_cfg)

        try:
            obs, info = env.reset(seed=args.seed)
        except TypeError:
            obs, info = env.reset()

        device = torch.device(str(env.unwrapped.device))
        num_envs = int(env.unwrapped.num_envs)
        action_space = env.action_space
        action_spec = infer_action_spec(action_space, num_envs, device)

        print(f"Task: {args.task}")
        print(f"Num envs: {num_envs}")
        print(f"Device: {device}")
        print(f"Action space: {action_space}")
        print(
            "Per-env action shape: "
            f"{action_spec['per_env_shape']} "
            f"(batched_action_space={action_spec['is_batched']})"
        )
        print(f"Observation summary after reset: {summarize_obs(obs)}")

        for step in range(args.num_steps):
            if step < 20:
                actions = torch.zeros(
                    (num_envs, *action_spec["per_env_shape"]),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                actions = 0.05 * torch.randn(
                    (num_envs, *action_spec["per_env_shape"]),
                    device=device,
                    dtype=torch.float32,
                )
                actions = torch.max(torch.min(actions, action_spec["high"]), action_spec["low"])

            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated | truncated

            if step % args.print_every == 0 or step == args.num_steps - 1:
                rewards_t = torch.as_tensor(rewards)
                done_t = torch.as_tensor(done)
                print(
                    f"step={step:04d} | "
                    f"reward_mean={rewards_t.mean().item(): .5f} | "
                    f"reward_min={rewards_t.min().item(): .5f} | "
                    f"reward_max={rewards_t.max().item(): .5f} | "
                    f"done_count={int(done_t.sum().item())}"
                )

        print("Verification passed: Peg Insert reset and stepped successfully with batched actions.")
    finally:
        if env is not None:
            env.close()
            env = None
        gc.collect()
        simulation_app.close()


if __name__ == "__main__":
    main()
