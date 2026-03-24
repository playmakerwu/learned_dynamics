"""Run the full solver24-vs-NeRD evaluation workflow end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nerd_eval.config import EvalConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run solver24 collection, solver192 collection, NeRD rollout, and comparison.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = EvalConfig()
    if args.device is not None:
        cfg.device = args.device
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.num_trajectories is not None:
        cfg.num_trajectories = args.num_trajectories
    if args.seed is not None:
        cfg.seed = args.seed

    steps = [
        [
            str(Path(sys.executable).resolve()),
            str((Path(__file__).resolve().parent / "collect_eval_solver24.py").resolve()),
            "--num_envs",
            str(cfg.num_envs),
            "--num_trajectories",
            str(cfg.num_trajectories),
            "--seed",
            str(cfg.seed),
            "--device",
            str(cfg.device),
        ],
        [
            str(Path(sys.executable).resolve()),
            str((Path(__file__).resolve().parent / "collect_eval_solver192.py").resolve()),
            "--num_envs",
            str(cfg.num_envs),
            "--num_trajectories",
            str(cfg.num_trajectories),
            "--seed",
            str(cfg.seed),
            "--device",
            str(cfg.device),
        ],
        [
            str(Path(sys.executable).resolve()),
            str((Path(__file__).resolve().parent / "rollout_nerd_eval.py").resolve()),
            "--device",
            str(cfg.device),
        ],
        [
            str(Path(sys.executable).resolve()),
            str((Path(__file__).resolve().parent / "compare_solver24_vs_nerd.py").resolve()),
        ],
    ]

    for command in steps:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
