"""Run autoregressive NeRD rollout evaluation using the solver24 real
trajectories as exogenous input.

Frame handling
--------------
The rollout now mirrors the training pipeline exactly:

1. At each step *t*, extract ``root_body_q`` from the current (world-frame)
   predicted state.
2. Convert the state window to body frame.
3. Convert exogenous inputs (contacts, gravity, actions) to body frame.
4. Run the NeRD model → prediction in body frame.
5. Reconstruct the next body-frame state (quaternion-aware).
6. Convert the next state back to world frame using the *current* anchor.

All quaternions use the **[w, x, y, z]** convention (Isaac Lab).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from nerd_bridge.common import ensure_directory
from nerd_bridge.frame_utils import (
    positions_body_to_world,
    positions_world_to_body,
    quat_apply_delta,
    quat_normalize,
    quat_positive_w,
    quats_body_to_world,
    quats_world_to_body,
    vectors_body_to_world,
    vectors_world_to_body,
)
from nerd_bridge.preprocessing import (
    StateLayout,
    apply_contact_mask,
    build_contact_mask,
    convert_contacts_to_body_frame,
    convert_gravity_to_body_frame,
    convert_states_to_body_frame,
    convert_states_to_world_frame,
    extract_root_body_q,
    reconstruct_next_state,
)
from nerd_eval.config import EvalConfig
from nerd_eval.utils import (
    align_real_datasets,
    find_state_slice,
    fixed_root_q_from_states,
    load_collector_dataset,
    load_nerd_model,
    make_dummy_sample_inputs,
    parse_state_layout,
    root_body_q_from_states,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Roll out the trained NeRD model from solver24 evaluation trajectories."
    )
    parser.add_argument("--solver24_dataset", type=str, default=None)
    parser.add_argument("--solver192_dataset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def flatten_sequence(raw: np.ndarray) -> np.ndarray:
    """Flatten feature dimensions into ``[B, T, D]`` for the NeRD model."""
    return raw.reshape(raw.shape[0], raw.shape[1], -1).astype(np.float32, copy=False)


def _build_raw_input_window(
    *,
    current_states: np.ndarray,
    dataset: Any,
    start: int,
    stop: int,
    input_keys: list[str],
    state_layout: list[dict[str, Any]],
    contact_mask_slots: int = 16,
) -> dict[str, np.ndarray]:
    """Assemble one raw (world-frame) input window for the NeRD model.

    ``states`` and ``root_body_q`` come from the predicted trajectory.
    All other keys come from the real solver24 dataset.
    """
    inputs: dict[str, np.ndarray] = {}

    # Always include states and root_body_q (derived from states).
    inputs["states"] = flatten_sequence(current_states[:, start:stop, :])
    inputs["root_body_q"] = flatten_sequence(
        root_body_q_from_states(current_states[:, start:stop, :], state_layout)
    )

    # Load exogenous inputs from the real dataset.
    for key in input_keys:
        if key in ("states", "root_body_q"):
            continue  # already handled above
        if key in dataset.data:
            inputs[key] = flatten_sequence(dataset.data[key][:, start:stop, ...])

    # Also load contact_counts for masking if available.
    if "contact_counts" in dataset.data:
        inputs["contact_counts"] = dataset.data["contact_counts"][:, start:stop].astype(
            np.int32, copy=False
        )

    return inputs


def _preprocess_window_torch(
    raw_np: dict[str, np.ndarray],
    layout: StateLayout,
    device: str,
    *,
    use_body_frame: bool,
    use_contact_masking: bool,
    num_contact_slots: int,
) -> dict[str, torch.Tensor]:
    """Convert a numpy input window to torch and apply body-frame preprocessing."""
    batch = {k: torch.from_numpy(v).to(device) for k, v in raw_np.items()}

    if use_body_frame:
        states = batch["states"]
        root_pos, root_quat = extract_root_body_q(states, layout)

        batch["states"] = convert_states_to_body_frame(states, layout, root_pos, root_quat)

        if "gravity_dir" in batch:
            batch["gravity_dir"] = convert_gravity_to_body_frame(batch["gravity_dir"], root_quat)

        if "root_body_q" in batch:
            rbq = batch["root_body_q"]
            rbq_pos, rbq_quat = rbq[..., :3], rbq[..., 3:7]
            batch["root_body_q"] = torch.cat(
                [positions_world_to_body(rbq_pos, root_pos, root_quat),
                 quats_world_to_body(rbq_quat, root_quat)],
                dim=-1,
            )

        batch = convert_contacts_to_body_frame(
            batch, root_pos, root_quat, num_contact_slots=num_contact_slots,
        )

    if use_contact_masking and "contact_counts" in batch:
        mask = build_contact_mask(batch["contact_counts"].int(), num_slots=num_contact_slots)
        batch = apply_contact_mask(batch, mask, num_contact_slots=num_contact_slots)

    return batch


def _reconstruct_world_state(
    state_body: torch.Tensor,
    prediction: torch.Tensor,
    layout: StateLayout,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    *,
    use_quat_targets: bool,
) -> np.ndarray:
    """Reconstruct the next world-frame state from a body-frame prediction."""
    if use_quat_targets:
        next_body = reconstruct_next_state(state_body, prediction, layout)
    else:
        next_body = state_body + prediction
    next_world = convert_states_to_world_frame(next_body, layout, root_pos, root_quat)
    return next_world.detach().cpu().numpy().astype(np.float32, copy=False)


def save_rollout_hdf5(
    path: Path,
    *,
    solver24_dataset_path: Path,
    solver192_dataset_path: Path,
    nerd_checkpoint_path: Path,
    alignment_info: dict[str, Any],
    predicted_states: np.ndarray,
    predicted_next_states: np.ndarray,
    predicted_root_body_q: np.ndarray,
    predicted_fixed_root_q: np.ndarray,
    traj_lengths: np.ndarray,
    source_env_ids: np.ndarray,
    actions: np.ndarray,
    state_layout: list[dict[str, Any]],
    use_body_frame: bool = True,
    use_quat_targets: bool = True,
) -> None:
    """Save the NeRD rollout for the later comparison step."""
    ensure_directory(path)
    with h5py.File(path, "w") as file:
        file.attrs["format"] = "nerd_eval_rollout"
        file.attrs["source_solver24_dataset"] = str(solver24_dataset_path)
        file.attrs["source_solver192_dataset"] = str(solver192_dataset_path)
        file.attrs["nerd_checkpoint_path"] = str(nerd_checkpoint_path)
        file.attrs["trajectory_alignment"] = "source_env_ids"
        file.attrs["state_layout"] = str(state_layout)
        file.attrs["alignment_info"] = str(alignment_info)
        file.attrs["rollout_input_source"] = "solver24"
        file.attrs["use_body_frame"] = use_body_frame
        file.attrs["use_quat_targets"] = use_quat_targets

        file.create_dataset("predicted_states", data=predicted_states, dtype=np.float32)
        file.create_dataset("predicted_next_states", data=predicted_next_states, dtype=np.float32)
        file.create_dataset("predicted_root_body_q", data=predicted_root_body_q, dtype=np.float32)
        file.create_dataset("predicted_fixed_root_q", data=predicted_fixed_root_q, dtype=np.float32)
        file.create_dataset("joint_acts", data=actions, dtype=np.float32)
        file.create_dataset("traj_lengths", data=traj_lengths.astype(np.int32), dtype=np.int32)
        file.create_dataset("source_env_ids", data=source_env_ids.astype(np.int32), dtype=np.int32)


def main() -> None:
    args = build_parser().parse_args()
    cfg = EvalConfig()
    solver24_path = Path(args.solver24_dataset) if args.solver24_dataset else cfg.solver24_real_path
    solver192_path = Path(args.solver192_dataset) if args.solver192_dataset else cfg.solver192_real_path
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.nerd_checkpoint
    output_path = Path(args.output_path) if args.output_path else cfg.nerd_rollout_path
    device = args.device or cfg.device

    solver24 = load_collector_dataset(solver24_path)
    solver192 = load_collector_dataset(solver192_path)
    aligned_solver24, aligned_solver192, alignment_info = align_real_datasets(solver24, solver192)
    state_layout_list = parse_state_layout(aligned_solver24)

    # Load checkpoint and extract preprocessing settings.
    checkpoint_peek = torch.load(
        Path(checkpoint_path).expanduser().resolve(), map_location=device, weights_only=False,
    )
    input_keys = list(checkpoint_peek["input_keys"])
    use_body_frame = checkpoint_peek.get("use_body_frame", False)
    use_quat_targets = checkpoint_peek.get("use_quat_targets", False)
    use_contact_masking = checkpoint_peek.get("use_contact_masking", False)
    num_contact_slots = checkpoint_peek.get("num_contact_slots", 16)

    # Build state layout from checkpoint (preferred) or dataset.
    ckpt_layout_raw = checkpoint_peek.get("state_layout")
    if ckpt_layout_raw is not None:
        layout = StateLayout.from_list(ckpt_layout_raw)
    else:
        layout = StateLayout.from_list(state_layout_list)

    print(f"Checkpoint input_keys: {input_keys}", flush=True)
    print(f"Preprocessing: body_frame={use_body_frame}, quat_targets={use_quat_targets}, "
          f"contact_masking={use_contact_masking}", flush=True)

    sample_inputs = make_dummy_sample_inputs(
        aligned_solver24,
        input_keys=input_keys,
        state_layout=state_layout_list,
        device=device,
    )
    model, checkpoint = load_nerd_model(
        checkpoint_path,
        sample_inputs=sample_inputs,
        output_dim=aligned_solver24.state_dim,
        device=device,
    )
    history_length = int(checkpoint["config"]["history_length"])

    num_trajectories = aligned_solver24.num_trajectories
    horizon = aligned_solver24.horizon
    traj_lengths = np.minimum(
        aligned_solver24.traj_lengths, aligned_solver192.traj_lengths,
    ).astype(np.int32)

    # All predicted states are stored in WORLD frame for downstream comparison.
    predicted_states = np.zeros_like(aligned_solver24.data["states"], dtype=np.float32)
    predicted_next_states = np.zeros_like(aligned_solver24.data["next_states"], dtype=np.float32)

    # Warm-start: copy the initial states from solver24.
    predicted_states[:, 0, :] = aligned_solver24.data["states"][:, 0, :].astype(np.float32, copy=False)

    with torch.inference_mode():
        for step in range(horizon):
            active_mask = traj_lengths > step
            if not np.any(active_mask):
                break

            active_count = int(active_mask.sum())
            window_start = max(0, step - history_length + 1)

            # 1. Assemble raw (world-frame) input window.
            raw_np = _build_raw_input_window(
                current_states=predicted_states[active_mask, ...],
                dataset=aligned_solver24,
                start=window_start,
                stop=step + 1,
                input_keys=input_keys,
                state_layout=state_layout_list,
            )

            # Also need the world-frame state at the LAST timestep in the window
            # so we can convert the prediction back to world frame.
            world_state_t = torch.from_numpy(
                predicted_states[active_mask, step, :].copy()
            ).to(device)
            root_pos_t, root_quat_t = extract_root_body_q(world_state_t, layout)

            # 2. Preprocess (body-frame conversion + contact masking).
            batch = _preprocess_window_torch(
                raw_np, layout, device,
                use_body_frame=use_body_frame,
                use_contact_masking=use_contact_masking,
                num_contact_slots=num_contact_slots,
            )

            # 3. Run the model – take the last timestep's prediction.
            model_inputs = {key: batch[key] for key in input_keys if key in batch}
            prediction = model(model_inputs)[:, -1, :]

            # 4. Reconstruct next state in world frame.
            if use_body_frame:
                state_body_t = batch["states"][:, -1, :]
                next_world = _reconstruct_world_state(
                    state_body_t, prediction, layout,
                    root_pos_t, root_quat_t,
                    use_quat_targets=use_quat_targets,
                )
            elif use_quat_targets:
                state_t = world_state_t
                next_body = reconstruct_next_state(
                    state_t, prediction.detach(), layout,
                )
                next_world = next_body.cpu().numpy().astype(np.float32, copy=False)
            else:
                # Legacy: naive addition.
                delta = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
                next_world = predicted_states[active_mask, step, :] + delta

            predicted_next_states[active_mask, step, :] = next_world
            if step + 1 < horizon:
                predicted_states[active_mask, step + 1, :] = next_world

            if step == 0:
                print(
                    f"NeRD rollout warm-started from solver24 initial states "
                    f"for {active_count} trajectories.",
                    flush=True,
                )

    predicted_root_body_q = root_body_q_from_states(predicted_states, state_layout_list)
    predicted_fixed_root_q = fixed_root_q_from_states(predicted_states, state_layout_list)

    save_rollout_hdf5(
        output_path.expanduser().resolve(),
        solver24_dataset_path=solver24_path.expanduser().resolve(),
        solver192_dataset_path=solver192_path.expanduser().resolve(),
        nerd_checkpoint_path=checkpoint_path.expanduser().resolve(),
        alignment_info=alignment_info,
        predicted_states=predicted_states,
        predicted_next_states=predicted_next_states,
        predicted_root_body_q=predicted_root_body_q,
        predicted_fixed_root_q=predicted_fixed_root_q,
        traj_lengths=traj_lengths,
        source_env_ids=aligned_solver24.source_env_ids,
        actions=aligned_solver24.data["joint_acts"].astype(np.float32, copy=False),
        state_layout=state_layout_list,
        use_body_frame=use_body_frame,
        use_quat_targets=use_quat_targets,
    )

    summary = {
        "solver24_dataset": str(solver24_path.expanduser().resolve()),
        "solver192_dataset": str(solver192_path.expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "output_path": str(output_path.expanduser().resolve()),
        "num_trajectories": int(num_trajectories),
        "horizon": int(horizon),
        "history_length": int(history_length),
        "alignment_info": alignment_info,
        "input_keys": input_keys,
        "use_body_frame": use_body_frame,
        "use_quat_targets": use_quat_targets,
        "use_contact_masking": use_contact_masking,
    }
    write_json(output_path.expanduser().resolve().with_suffix(".json"), summary)
    print(f"Saved NeRD rollout to: {output_path.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
