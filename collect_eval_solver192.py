"""Collect new solver=192 reference evaluation trajectories."""

from __future__ import annotations

from nerd_eval.config import EvalConfig

from collect_eval_real import main


if __name__ == "__main__":
    main(default_solver=192, default_output_path=EvalConfig().solver192_real_path)

