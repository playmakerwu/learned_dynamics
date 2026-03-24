"""Collect new peg-insert evaluation trajectories for a chosen solver setting.

This is a thin evaluation-only wrapper around the existing `nerd_collector/collector.py`.
It keeps all old collection behavior untouched and only adds evaluation-specific metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nerd_eval.config import EvalConfig
from nerd_eval.utils import default_output_for_solver, run_real_collection


def build_parser(*, solver_required: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect new peg-insert evaluation trajectories for solver24 or solver192.")
    parser.add_argument("--solver", type=int, required=solver_required, choices=(24, 192))
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--nerd_checkpoint", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_trajectories", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--policy_device", type=str, default=None)
    parser.add_argument("--horizon_steps", type=int, default=None)
    parser.add_argument("--episode_length_steps", type=int, default=None)
    return parser


def main(default_solver: int | None = None, default_output_path: Path | None = None) -> None:
    parser = build_parser(solver_required=default_solver is None)
    args = parser.parse_args()
    cfg = EvalConfig()
    solver = default_solver if default_solver is not None else args.solver

    if args.checkpoint is not None:
        cfg.policy_checkpoint = Path(args.checkpoint)
    if args.nerd_checkpoint is not None:
        cfg.nerd_checkpoint = Path(args.nerd_checkpoint)
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.num_trajectories is not None:
        cfg.num_trajectories = args.num_trajectories
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.policy_device is not None:
        cfg.policy_device = args.policy_device
    if args.horizon_steps is not None:
        cfg.horizon_steps = args.horizon_steps
    if args.episode_length_steps is not None:
        cfg.episode_length_steps = args.episode_length_steps

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else default_output_path
        if default_output_path is not None
        else default_output_for_solver(solver, cfg)
    )
    run_real_collection(cfg, solver_position_iterations=solver, output_path=output_path)


if __name__ == "__main__":
    main()
