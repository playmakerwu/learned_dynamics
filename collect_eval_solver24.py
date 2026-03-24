"""Collect new solver=24 rough-simulator evaluation trajectories."""

from __future__ import annotations

from nerd_eval.config import EvalConfig

from collect_eval_real import main


if __name__ == "__main__":
    main(default_solver=24, default_output_path=EvalConfig().solver24_real_path)

