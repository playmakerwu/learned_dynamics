"""Convert the collector output into an upstream NeRD-friendly trajectory HDF5 format."""

from __future__ import annotations

import argparse
from pathlib import Path

from nerd_bridge.common import DEFAULT_CONVERTED_DATASET, resolve_source_dataset, save_json
from nerd_bridge.dataset_utils import convert_collector_to_nerd, inspect_hdf5


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert the local collector dataset into NeRD trajectory format.")
    parser.add_argument("--input", type=str, default=None, help="Path to the source collector dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONVERTED_DATASET,
        help="Path for the converted NeRD-style dataset.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="PegInsertIsaacLab",
        help="Metadata tag stored in the converted dataset.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("recordings/nerd_base_converted_summary.json"),
        help="Where to save the converted dataset summary JSON.",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    source_path, notes = resolve_source_dataset(args.input)
    result = convert_collector_to_nerd(source_path, args.output, env_name=args.env_name)
    summary = inspect_hdf5(args.output).to_dict()
    summary["conversion"] = result
    summary["notes"] = notes
    save_json(args.summary, summary)

    print(f"Converted source dataset: {source_path}", flush=True)
    for note in notes:
        print(f"Note: {note}", flush=True)
    print(f"Converted dataset written to: {args.output.resolve()}", flush=True)
    print(f"Num trajectories: {summary['num_trajectories']}", flush=True)
    print(f"Horizon: {summary['horizon']}", flush=True)
    print(f"Saved summary to: {args.summary.resolve()}", flush=True)


if __name__ == "__main__":
    main()

