"""Train a data-driven NeRD model from the converted peg-insert trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from nerd_bridge.common import (
    DEFAULT_CONVERTED_DATASET,
    DEFAULT_TEST_INDICES,
    DEFAULT_TRAIN_INDICES,
    default_device,
)
from nerd_bridge.training import TrainConfig, run_training


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NeRD model from the converted peg-insert dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CONVERTED_DATASET, help="Converted dataset path.")
    parser.add_argument("--train_indices", type=Path, default=DEFAULT_TRAIN_INDICES)
    parser.add_argument("--test_indices", type=Path, default=DEFAULT_TEST_INDICES)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/nerd_peg_insert"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--evaluation_frequency", type=int, default=1)
    parser.add_argument(
        "--max_train_batches_per_epoch",
        type=int,
        default=0,
        help="Maximum train batches per epoch. Use 0 to consume the full train loader.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Maximum eval batches per epoch. Use 0 to consume the full eval loader.",
    )
    parser.add_argument("--normalization_batches", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Transformer dropout rate.")
    parser.add_argument(
        "--no_normalized_loss",
        action="store_true",
        help="Disable normalized loss (use raw MSE instead).",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config = TrainConfig(
        converted_dataset_path=args.dataset,
        train_indices_path=args.train_indices,
        test_indices_path=args.test_indices,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        history_length=args.history_length,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_path=args.checkpoint,
        evaluation_frequency=args.evaluation_frequency,
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
        max_eval_batches=args.max_eval_batches,
        normalization_batches=args.normalization_batches,
        weight_decay=args.weight_decay,
        transformer_dropout=args.transformer_dropout,
        normalized_loss=not args.no_normalized_loss,
    )
    summary = run_training(config)

    print(f"Training finished. Output dir: {args.output_dir.resolve()}", flush=True)
    print(f"Latest checkpoint: {summary['latest_checkpoint']}", flush=True)
    if summary["best_checkpoint"] is not None:
        print(f"Best checkpoint: {summary['best_checkpoint']}", flush=True)
    print(f"Train windows: {summary['train_windows']}", flush=True)
    print(f"Test windows: {summary['test_windows']}", flush=True)


if __name__ == "__main__":
    main()
