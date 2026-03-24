"""Verify that a collected HDF5 dataset contains meaningful contact impulse data.

This script inspects the contact fields in a trajectory dataset and prints
statistics to confirm that impulse data is present and nonzero. It also
compares impulse quality against the old depth-only contact representation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


CONTACT_FIELDS = [
    "net_contact_force",
    "contact_normals",
    "contact_depths",
    "contact_thicknesses",
    "contact_points_0",
    "contact_points_1",
    "contact_counts",
    "contact_impulses",
    "contact_impulse_vectors",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify contact impulse data quality in an HDF5 dataset.")
    parser.add_argument("input", type=str, help="Path to the HDF5 file (collector or NeRD format).")
    parser.add_argument("--summary_path", type=str, default=None, help="Optional JSON summary output.")
    return parser


def summarize_contact_field(name: str, arr: np.ndarray) -> dict:
    """Compute summary statistics for a contact array."""
    flat = arr.reshape(-1)
    nz = int(np.count_nonzero(flat))
    result = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nonzero": nz,
        "total": int(flat.size),
        "nonzero_pct": f"{nz / flat.size * 100:.2f}%",
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "mean_abs": float(np.mean(np.abs(flat))),
        "std": float(np.std(flat)),
    }
    if nz > 0:
        nz_vals = flat[flat != 0]
        result["nz_min"] = float(np.min(np.abs(nz_vals)))
        result["nz_max"] = float(np.max(np.abs(nz_vals)))
        result["nz_mean"] = float(np.mean(np.abs(nz_vals)))
        result["nz_median"] = float(np.median(np.abs(nz_vals)))
    return result


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.input).expanduser().resolve()
    if not path.is_file():
        print(f"ERROR: File not found: {path}", flush=True)
        sys.exit(1)

    summary = {"path": str(path), "fields": {}, "verdict": ""}

    with h5py.File(path, "r") as f:
        # Detect format (collector vs NeRD)
        if "data" in f:
            grp = f["data"]
            fmt = "nerd_converted"
        else:
            grp = f
            fmt = "collector_raw"
        print(f"File: {path}", flush=True)
        print(f"Format: {fmt}", flush=True)

        all_keys = sorted(grp.keys())
        print(f"All keys: {all_keys}\n", flush=True)

        # Check all contact fields
        has_impulses = False
        has_nonzero_impulses = False

        for field_name in CONTACT_FIELDS:
            if field_name not in grp:
                print(f"  {field_name}: MISSING", flush=True)
                summary["fields"][field_name] = {"missing": True}
                continue

            arr = grp[field_name][:]
            stats = summarize_contact_field(field_name, arr)
            summary["fields"][field_name] = stats

            if field_name == "contact_impulses":
                has_impulses = True
                if stats["nonzero"] > 0:
                    has_nonzero_impulses = True

            print(f"  {field_name}:", flush=True)
            print(f"    shape={stats['shape']} dtype={stats['dtype']}", flush=True)
            print(f"    nonzero={stats['nonzero']}/{stats['total']} ({stats['nonzero_pct']})", flush=True)
            print(f"    min={stats['min']:.8f}  max={stats['max']:.8f}  mean_abs={stats['mean_abs']:.8f}", flush=True)
            if stats["nonzero"] > 0:
                print(f"    nonzero: min_abs={stats['nz_min']:.8f}  max_abs={stats['nz_max']:.8f}  "
                      f"mean_abs={stats['nz_mean']:.8f}  median_abs={stats['nz_median']:.8f}", flush=True)
            print(flush=True)

        # Comparison: impulse vs depth informativeness
        if "contact_impulses" in grp and "contact_depths" in grp:
            impulses = grp["contact_impulses"][:]
            depths = grp["contact_depths"][:]
            print("=== Impulse vs Depth comparison ===", flush=True)
            imp_nz = np.count_nonzero(impulses)
            dep_nz = np.count_nonzero(depths)
            print(f"  Impulse nonzero: {imp_nz}/{impulses.size} ({imp_nz/impulses.size*100:.2f}%)", flush=True)
            print(f"  Depth nonzero:   {dep_nz}/{depths.size} ({dep_nz/depths.size*100:.2f}%)", flush=True)
            if imp_nz > 0:
                imp_nz_vals = np.abs(impulses[impulses != 0])
                print(f"  Impulse dynamic range: {imp_nz_vals.min():.6f} to {imp_nz_vals.max():.6f} "
                      f"(ratio: {imp_nz_vals.max()/imp_nz_vals.min():.1f}x)", flush=True)
            if dep_nz > 0:
                dep_nz_vals = np.abs(depths[depths != 0])
                print(f"  Depth dynamic range:   {dep_nz_vals.min():.8f} to {dep_nz_vals.max():.8f} "
                      f"(ratio: {dep_nz_vals.max()/dep_nz_vals.min():.1f}x)", flush=True)
            print(flush=True)

        # Per-timestep contact activity
        if "contact_counts" in grp:
            counts = grp["contact_counts"][:]
            print("=== Per-timestep contact activity ===", flush=True)
            has_contact = counts > 0
            print(f"  Steps with any contact: {has_contact.sum()}/{has_contact.size} "
                  f"({has_contact.sum()/has_contact.size*100:.1f}%)", flush=True)
            if has_contact.any():
                print(f"  Contact count when active: mean={counts[has_contact].mean():.1f}, "
                      f"max={counts[has_contact].max()}", flush=True)
            print(flush=True)

        # Check net_contact_force (the GPU-compatible path)
        has_net_force = "net_contact_force" in summary["fields"] and not summary["fields"]["net_contact_force"].get("missing")
        has_nonzero_net_force = has_net_force and summary["fields"]["net_contact_force"]["nonzero"] > 0

        # Verdict
        if has_nonzero_net_force:
            nf_stats = summary["fields"]["net_contact_force"]
            nf_pct = nf_stats["nonzero"] / nf_stats["total"]
            if nf_pct < 0.01:
                verdict = "WARN: Very few nonzero net_contact_force entries (<1%)."
            else:
                verdict = f"PASS: Dataset contains meaningful net_contact_force data ({nf_pct*100:.1f}% nonzero)."
        elif has_impulses and has_nonzero_impulses:
            impulse_stats = summary["fields"]["contact_impulses"]
            if impulse_stats["nonzero"] / impulse_stats["total"] < 0.01:
                verdict = "WARN: Very few nonzero impulses (<1%). Contact detection may be unreliable."
            else:
                verdict = "PASS: Dataset contains meaningful contact impulse data."
        elif has_impulses:
            verdict = "FAIL: contact_impulses exists but is all-zero. Check collection device."
        else:
            verdict = "FAIL: No contact force data found. Recollect with collect_with_net_contact_force.py."

        summary["verdict"] = verdict
        print(f"VERDICT: {verdict}", flush=True)

    if args.summary_path is not None:
        out_path = Path(args.summary_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Summary saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
