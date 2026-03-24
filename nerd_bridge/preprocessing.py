"""Body-frame conversion, quaternion-aware target construction, and contact masking.

This module sits between the raw HDF5 dataset and the NeRD model, implementing
the preprocessing steps described in the official NeRD paper:

1. **Body-frame conversion** – all positions, orientations, velocities,
   contact normals/points, and gravity are expressed in a body-centric frame
   anchored at ``root_body_q`` (the held peg).
2. **Quaternion-aware target** – orientation deltas use proper rotation
   composition (``delta = q_to ⊗ q_from^{-1}``) instead of naive subtraction.
3. **Contact masking** – inactive contact slots (beyond ``contact_counts``)
   are zeroed out so the model receives a clean signal.

Frame conventions
-----------------
- Input (dataset) frame: *local* frame, i.e. world – env_origin for positions;
  world frame for orientations and velocities.
- Output (body) frame: anchored at ``root_body_q`` extracted from each
  timestep's state vector.  The peg's own position becomes [0,0,0] and its
  orientation becomes identity [1,0,0,0].

All quaternions are **[w, x, y, z]** (scalar-first, Isaac Lab convention).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .frame_utils import (
    positions_body_to_world,
    positions_world_to_body,
    quat_apply_delta,
    quat_delta,
    quat_normalize,
    quat_positive_w,
    quats_body_to_world,
    quats_world_to_body,
    vectors_body_to_world,
    vectors_world_to_body,
)


# ---------------------------------------------------------------------------
# State-layout descriptor
# ---------------------------------------------------------------------------

# These field names define the semantic category for body-frame transforms.
_POSITION_FIELDS = {"ee_pos_local", "held_root_pos_local", "fixed_root_pos_local"}
_QUATERNION_FIELDS = {"ee_quat_wxyz", "held_root_quat_wxyz", "fixed_root_quat_wxyz"}
_VELOCITY_FIELDS = {"ee_lin_vel_w", "ee_ang_vel_w", "held_root_lin_vel_w", "held_root_ang_vel_w"}
_JOINT_FIELDS = {"robot_joint_pos", "robot_joint_vel"}  # joint-local; no transform

# root_body_q is extracted from these two fields of the state vector:
_ROOT_POS_FIELD = "held_root_pos_local"
_ROOT_QUAT_FIELD = "held_root_quat_wxyz"


@dataclass(frozen=True, slots=True)
class StateFieldInfo:
    """Metadata for one named slice of the 47-dim state vector."""

    name: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Parsed state-vector layout loaded from the HDF5 metadata.

    Provides named slices and semantic categories used by the preprocessing
    pipeline.  The layout is immutable and hashable.
    """

    fields: tuple[StateFieldInfo, ...]
    state_dim: int

    # Pre-computed category indices for fast batch transforms.
    position_slices: tuple[slice, ...]
    quaternion_slices: tuple[slice, ...]
    velocity_slices: tuple[slice, ...]
    root_pos_slice: slice
    root_quat_slice: slice

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "StateLayout":
        """Build from the JSON list stored in the HDF5 ``state_layout`` attr."""
        fields = tuple(
            StateFieldInfo(name=item["name"], start=int(item["start"]), end=int(item["end"]))
            for item in items
        )
        state_dim = max(f.end for f in fields) if fields else 0

        pos_slices = tuple(f.slice for f in fields if f.name in _POSITION_FIELDS)
        quat_slices = tuple(f.slice for f in fields if f.name in _QUATERNION_FIELDS)
        vel_slices = tuple(f.slice for f in fields if f.name in _VELOCITY_FIELDS)

        root_pos = next(f for f in fields if f.name == _ROOT_POS_FIELD)
        root_quat = next(f for f in fields if f.name == _ROOT_QUAT_FIELD)

        return cls(
            fields=fields,
            state_dim=state_dim,
            position_slices=pos_slices,
            quaternion_slices=quat_slices,
            velocity_slices=vel_slices,
            root_pos_slice=root_pos.slice,
            root_quat_slice=root_quat.slice,
        )

    @classmethod
    def from_hdf5_attr(cls, raw: str | list) -> "StateLayout":
        """Parse from the raw HDF5 attribute (either a JSON string or list)."""
        if isinstance(raw, str):
            raw = json.loads(raw)
        return cls.from_list(raw)

    def field_by_name(self, name: str) -> StateFieldInfo:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"State field '{name}' not found in layout.")


# ---------------------------------------------------------------------------
# Body-frame conversion for the full state vector
# ---------------------------------------------------------------------------


def extract_root_body_q(states: Tensor, layout: StateLayout) -> tuple[Tensor, Tensor]:
    """Extract anchor position and quaternion from the state vector.

    Parameters
    ----------
    states : (..., D)
    layout : StateLayout

    Returns
    -------
    root_pos : (..., 3)  – anchor position in local/world frame
    root_quat : (..., 4) – anchor orientation [w,x,y,z] in world frame
    """
    root_pos = states[..., layout.root_pos_slice]
    root_quat = states[..., layout.root_quat_slice]
    return root_pos, root_quat


def convert_states_to_body_frame(
    states: Tensor,
    layout: StateLayout,
    root_pos: Tensor | None = None,
    root_quat: Tensor | None = None,
) -> Tensor:
    """Convert an entire state vector from local/world frame to body frame.

    Transformation per field category:
    - **joint fields** (``robot_joint_pos/vel``): unchanged (joint-local).
    - **position fields** (``*_pos_local``): ``R^{-1}(p - anchor_pos)``.
    - **quaternion fields** (``*_quat_wxyz``): ``conj(anchor_q) ⊗ q``.
    - **velocity fields** (``*_vel_w``): ``R^{-1} v``.

    After conversion the peg (held) position is ~0 and its quaternion is
    ~identity, which is expected: it IS the body-frame origin.

    Parameters
    ----------
    states : (..., D)   – state in local/world frame.
    layout : StateLayout
    root_pos, root_quat : optional overrides; extracted from *states* if None.

    Returns
    -------
    Tensor (..., D) – state in body frame.
    """
    if root_pos is None or root_quat is None:
        root_pos, root_quat = extract_root_body_q(states, layout)

    out = states.clone()

    for sl in layout.position_slices:
        out[..., sl] = positions_world_to_body(states[..., sl], root_pos, root_quat)

    for sl in layout.quaternion_slices:
        out[..., sl] = quats_world_to_body(states[..., sl], root_quat)

    for sl in layout.velocity_slices:
        out[..., sl] = vectors_world_to_body(states[..., sl], root_quat)

    return out


def convert_states_to_world_frame(
    states_body: Tensor,
    layout: StateLayout,
    root_pos: Tensor,
    root_quat: Tensor,
) -> Tensor:
    """Inverse of :func:`convert_states_to_body_frame`.

    The caller must supply the original world-frame ``root_pos`` and
    ``root_quat`` because they cannot be recovered from the body-frame state
    (the peg's own fields are trivial there).
    """
    out = states_body.clone()

    for sl in layout.position_slices:
        out[..., sl] = positions_body_to_world(states_body[..., sl], root_pos, root_quat)

    for sl in layout.quaternion_slices:
        out[..., sl] = quats_body_to_world(states_body[..., sl], root_quat)

    for sl in layout.velocity_slices:
        out[..., sl] = vectors_body_to_world(states_body[..., sl], root_quat)

    return out


# ---------------------------------------------------------------------------
# Body-frame conversion for contact and exogenous tensors
# ---------------------------------------------------------------------------


def convert_contacts_to_body_frame(
    batch: dict[str, Tensor],
    root_pos: Tensor,
    root_quat: Tensor,
    *,
    num_contact_slots: int = 16,
) -> dict[str, Tensor]:
    """In-place-ish conversion of contact tensors to body frame.

    Contact normals and impulse vectors are free vectors → rotate only.
    Contact points are position vectors → translate + rotate.

    Parameters
    ----------
    batch : dict with potential keys ``contact_normals``, ``contact_points_0``,
        ``contact_points_1``, ``contact_impulse_vectors``.
        Shapes are (..., K*3) flattened or (..., K, 3).
    root_pos, root_quat : (..., 3) and (..., 4) – body-frame anchor.
    num_contact_slots : K (default 16).

    Returns
    -------
    The same dict with transformed tensors (new allocations, not truly in-place).
    """
    K = num_contact_slots
    out = dict(batch)

    def _reshape_3(t: Tensor) -> Tensor:
        """Reshape (..., K*3) → (..., K, 3) if needed."""
        if t.shape[-1] == K * 3:
            return t.view(*t.shape[:-1], K, 3)
        return t

    def _flatten_3(t: Tensor, ref_shape: torch.Size) -> Tensor:
        """Collapse back to (..., K*3) if the reference was flat."""
        if ref_shape[-1] == K * 3:
            return t.view(*t.shape[:-2], K * 3)
        return t

    # Expand anchor to match contact-slot dimension.
    # root_pos/root_quat are (..., 3/4); we need (..., K, 3/4).
    rp = root_pos.unsqueeze(-2).expand(*root_pos.shape[:-1], K, 3)
    rq = root_quat.unsqueeze(-2).expand(*root_quat.shape[:-1], K, 4)

    for key in ("contact_normals", "contact_impulse_vectors"):
        if key in out:
            ref = out[key]
            v = _reshape_3(ref)
            out[key] = _flatten_3(vectors_world_to_body(v, rq), ref.shape)

    for key in ("contact_points_0", "contact_points_1"):
        if key in out:
            ref = out[key]
            p = _reshape_3(ref)
            out[key] = _flatten_3(positions_world_to_body(p, rp, rq), ref.shape)

    return out


def convert_gravity_to_body_frame(
    gravity_dir: Tensor,
    root_quat: Tensor,
) -> Tensor:
    """Rotate gravity direction vector to body frame.

    ``g_body = R^{-1}(anchor) g_world``
    """
    return vectors_world_to_body(gravity_dir, root_quat)


# ---------------------------------------------------------------------------
# Contact masking
# ---------------------------------------------------------------------------


def build_contact_mask(
    contact_counts: Tensor,
    num_slots: int = 16,
) -> Tensor:
    """Build a boolean mask from per-sample contact counts.

    Parameters
    ----------
    contact_counts : (...,)  int tensor – number of active slots per sample.
    num_slots : K.

    Returns
    -------
    mask : (..., K)  bool – True for active slots, False for empty ones.
    """
    slot_idx = torch.arange(num_slots, device=contact_counts.device)
    # Expand contact_counts to compare with each slot index.
    return slot_idx.expand(*contact_counts.shape, num_slots) < contact_counts.unsqueeze(-1)


def apply_contact_mask(
    batch: dict[str, Tensor],
    mask: Tensor,
    *,
    num_contact_slots: int = 16,
) -> dict[str, Tensor]:
    """Zero-out inactive contact slots in all ``contact_*`` tensors.

    Parameters
    ----------
    batch : dict with contact tensors.
    mask : (..., K) bool – True = active.
    num_contact_slots : K.

    Returns
    -------
    New dict with masked tensors.
    """
    K = num_contact_slots
    out = dict(batch)

    for key in list(out.keys()):
        if not key.startswith("contact_") or key == "contact_counts":
            continue
        t = out[key]
        last_dim = t.shape[-1]

        if last_dim == K:
            # Scalar-per-slot: (..., K) – e.g. contact_depths, contact_impulses.
            out[key] = t * mask.float()
        elif last_dim == K * 3:
            # Vector-per-slot flattened: (..., K*3).
            mask_3 = mask.unsqueeze(-1).expand(*mask.shape, 3).reshape(*mask.shape[:-1], K * 3)
            out[key] = t * mask_3.float()
        # Other shapes (e.g. already (..., K, 3)) – expand mask.
        elif t.ndim >= 2 and t.shape[-2] == K and t.shape[-1] == 3:
            out[key] = t * mask.unsqueeze(-1).float()

    return out


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def compute_target(
    states_body: Tensor,
    next_states_body: Tensor,
    layout: StateLayout,
) -> Tensor:
    """Quaternion-aware prediction target in body frame.

    For **non-quaternion** dimensions:
        ``target = next_state - state``  (standard state delta)

    For **quaternion** dimensions:
        ``target = delta_quat(state_quat, next_state_quat)``
        where ``next_quat ≈ target_quat ⊗ state_quat``.

    Both states must already be in body frame.

    Parameters
    ----------
    states_body, next_states_body : (..., D) – body-frame states.
    layout : StateLayout

    Returns
    -------
    target : (..., D) – prediction target.
    """
    target = next_states_body - states_body

    for sl in layout.quaternion_slices:
        q_from = states_body[..., sl]
        q_to = next_states_body[..., sl]
        target[..., sl] = quat_delta(q_from, q_to)

    return target


def reconstruct_next_state(
    states_body: Tensor,
    prediction: Tensor,
    layout: StateLayout,
) -> Tensor:
    """Reconstruct next-state from current state + model prediction.

    Inverse of :func:`compute_target`:
    - Non-quaternion: ``next = state + prediction``
    - Quaternion: ``next_quat = normalize(prediction_quat ⊗ state_quat)``

    Parameters
    ----------
    states_body : (..., D) – current body-frame state.
    prediction : (..., D) – model output (delta / rotation delta).
    layout : StateLayout

    Returns
    -------
    next_states_body : (..., D) – predicted next state in body frame.
    """
    next_states = states_body + prediction

    for sl in layout.quaternion_slices:
        q_from = states_body[..., sl]
        delta = prediction[..., sl]
        next_states[..., sl] = quat_apply_delta(q_from, delta)

    return next_states


# ---------------------------------------------------------------------------
# Full batch preprocessing  (called before forward pass)
# ---------------------------------------------------------------------------


def preprocess_batch(
    batch: dict[str, Tensor],
    layout: StateLayout,
    *,
    use_body_frame: bool = True,
    apply_contact_masking: bool = True,
    num_contact_slots: int = 16,
) -> dict[str, Tensor]:
    """Apply the full NeRD-style preprocessing to a training batch.

    Steps (in order):
    1. Extract ``root_body_q`` (anchor) from ``states``.
    2. Convert states and next_states to body frame.
    3. Convert contact fields and gravity to body frame.
    4. Build contact mask from ``contact_counts`` and zero inactive slots.
    5. Replace ``root_body_q`` with body-frame version (becomes [0,..,0,1,0,0,0]).

    The batch is returned as a **new dict** with transformed tensors.
    The original tensors are not modified.

    Parameters
    ----------
    batch : dict of (B, T, D) tensors from the dataloader.
    layout : StateLayout
    use_body_frame : apply body-frame conversion (True by default).
    apply_contact_masking : zero out inactive contact slots (True by default).
    num_contact_slots : K (default 16).

    Returns
    -------
    Preprocessed batch dict.
    """
    out = dict(batch)

    states = out["states"]
    next_states = out["next_states"]

    if use_body_frame:
        # 1. Extract per-timestep anchor from the (world-frame) state.
        root_pos, root_quat = extract_root_body_q(states, layout)

        # 2. Convert states and next_states.
        #    "every" anchoring: each timestep uses its own root_body_q.
        #    next_states at time t are converted using root_body_q at time t.
        out["states"] = convert_states_to_body_frame(states, layout, root_pos, root_quat)
        out["next_states"] = convert_states_to_body_frame(next_states, layout, root_pos, root_quat)

        # 3. Convert exogenous inputs.
        if "gravity_dir" in out:
            out["gravity_dir"] = convert_gravity_to_body_frame(out["gravity_dir"], root_quat)

        # Convert root_body_q itself to body frame.
        # In "every" anchoring, this becomes [0,0,0, 1,0,0,0] everywhere.
        if "root_body_q" in out:
            rbq = out["root_body_q"]
            rbq_pos = rbq[..., :3]
            rbq_quat = rbq[..., 3:7]
            body_pos = positions_world_to_body(rbq_pos, root_pos, root_quat)
            body_quat = quats_world_to_body(rbq_quat, root_quat)
            out["root_body_q"] = torch.cat([body_pos, body_quat], dim=-1)

        out = convert_contacts_to_body_frame(
            out, root_pos, root_quat, num_contact_slots=num_contact_slots,
        )

    if apply_contact_masking and "contact_counts" in out:
        counts = out["contact_counts"].int()
        # contact_counts may have a trailing feature dim of 1 from the dataset
        # loader's flatten step.  Squeeze it so the mask is (..., K).
        if counts.ndim >= 1 and counts.shape[-1] == 1:
            counts = counts.squeeze(-1)
        mask = build_contact_mask(counts, num_slots=num_contact_slots)
        out = apply_contact_mask(out, mask, num_contact_slots=num_contact_slots)

    return out
