"""Utilities for projecting raw PhysX contacts into fixed NeRD contact slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class FixedSlotContacts:
    """Fixed-width contact tensors for one environment step."""

    contact_normals: torch.Tensor
    contact_depths: torch.Tensor
    contact_thicknesses: torch.Tensor
    contact_points_0: torch.Tensor
    contact_points_1: torch.Tensor
    contact_counts: torch.Tensor
    contact_impulses: torch.Tensor  # [num_envs, K] impulse magnitude per slot
    contact_impulse_vectors: torch.Tensor  # [num_envs, K, 3] impulse direction per slot
    contact_identities: torch.Tensor  # [num_envs, K] int32: 0=hole/env, 1=robot


def safe_normalize(vectors: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize vectors while keeping zero vectors stable."""

    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    safe_norms = torch.clamp(norms, min=eps)
    normalized = vectors / safe_norms
    return torch.where(norms > eps, normalized, torch.zeros_like(vectors))


def empty_fixed_slot_contacts(
    *,
    num_envs: int,
    k: int,
    device: torch.device | str,
    dtype: torch.dtype,
    contact_thickness: float,
) -> FixedSlotContacts:
    """Return an all-zero fixed-slot contact batch."""

    return FixedSlotContacts(
        contact_normals=torch.zeros((num_envs, k, 3), device=device, dtype=dtype),
        contact_depths=torch.zeros((num_envs, k), device=device, dtype=dtype),
        contact_thicknesses=torch.full((num_envs, k), float(contact_thickness), device=device, dtype=dtype),
        contact_points_0=torch.zeros((num_envs, k, 3), device=device, dtype=dtype),
        contact_points_1=torch.zeros((num_envs, k, 3), device=device, dtype=dtype),
        contact_counts=torch.zeros((num_envs,), device=device, dtype=torch.int32),
        contact_impulses=torch.zeros((num_envs, k), device=device, dtype=dtype),
        contact_impulse_vectors=torch.zeros((num_envs, k, 3), device=device, dtype=dtype),
        contact_identities=torch.zeros((num_envs, k), device=device, dtype=torch.int32),
    )


def assign_contact_slots(
    *,
    force_magnitudes: torch.Tensor,
    contact_points_0: torch.Tensor,
    contact_normals: torch.Tensor,
    separations: torch.Tensor,
    buffer_count: torch.Tensor,
    buffer_start_indices: torch.Tensor,
    num_envs: int,
    num_source_bodies: int,
    num_filter_bodies: int,
    k: int,
    max_depth: float,
    contact_thickness: float,
    force_vectors: torch.Tensor | None = None,
) -> FixedSlotContacts:
    """Project variable-length PhysX contact data into K fixed slots per environment.

    Slot assignment keeps the strongest contacts per environment, ranked by impulse
    magnitude (primary) with depth as tiebreaker. The convention used here is:

    * `contact_points_0`: raw PhysX point on the source/sensor body.
    * `contact_normals`: unit normals pointing in the direction of the force on the
      source/sensor body.
    * `contact_depths`: `clamp(max(0, -separation), max=max_depth)`.
    * `contact_points_1`: reconstructed point on the target body using
      `contact_points_0 + contact_normals * contact_depth`.
    * `contact_impulses`: impulse magnitude per contact slot.
    * `contact_impulse_vectors`: 3D impulse vector per contact slot.
    """

    device = contact_points_0.device
    dtype = contact_points_0.dtype
    slots = empty_fixed_slot_contacts(
        num_envs=num_envs,
        k=k,
        device=device,
        dtype=dtype,
        contact_thickness=contact_thickness,
    )

    if force_magnitudes.numel() == 0 or buffer_count.numel() == 0:
        return slots

    force_magnitudes = force_magnitudes.reshape(-1).to(dtype=dtype)
    separations = separations.reshape(-1).to(dtype=dtype)
    flat_counts = buffer_count.reshape(-1).to(device=device, dtype=torch.long)
    flat_starts = buffer_start_indices.reshape(-1).to(device=device, dtype=torch.long)
    active_groups = flat_counts > 0

    if not torch.any(active_groups):
        return slots

    group_ids = torch.nonzero(active_groups, as_tuple=False).squeeze(-1)
    group_counts = flat_counts.index_select(0, group_ids)
    group_starts = flat_starts.index_select(0, group_ids)

    flat_row_ids = torch.div(group_ids, num_filter_bodies, rounding_mode="floor")
    env_ids_per_group = torch.div(flat_row_ids, num_source_bodies, rounding_mode="floor")

    repeated_group_ids = torch.repeat_interleave(
        torch.arange(group_ids.numel(), device=device, dtype=torch.long),
        group_counts,
    )
    group_offsets = torch.cumsum(group_counts, dim=0) - group_counts
    deltas = torch.arange(int(group_counts.sum().item()), device=device, dtype=torch.long)
    deltas = deltas - group_offsets.repeat_interleave(group_counts)
    raw_contact_ids = group_starts.index_select(0, repeated_group_ids) + deltas
    env_ids = env_ids_per_group.index_select(0, repeated_group_ids)

    normals = safe_normalize(contact_normals.index_select(0, raw_contact_ids))
    depths = torch.clamp(-separations.index_select(0, raw_contact_ids), min=0.0, max=max_depth)
    points_0 = contact_points_0.index_select(0, raw_contact_ids)
    points_1 = points_0 + normals * depths.unsqueeze(-1)
    per_contact_impulse = force_magnitudes.index_select(0, raw_contact_ids).abs()

    # Impulse magnitude is the primary ranking signal. Depth is a small tiebreaker
    # since the PhysX solver resolves penetrations to near-zero separation.
    scores = per_contact_impulse + 1.0e-6 * depths

    # Resolve per-contact impulse vectors if available.
    has_force_vectors = force_vectors is not None and force_vectors.numel() > 0
    if has_force_vectors:
        flat_force_vectors = force_vectors.reshape(-1, 3).to(dtype=dtype)
        per_contact_impulse_vec = flat_force_vectors.index_select(0, raw_contact_ids)
    else:
        per_contact_impulse_vec = normals * per_contact_impulse.unsqueeze(-1)

    for env_id in range(num_envs):
        env_mask = env_ids == env_id
        if not torch.any(env_mask):
            continue

        env_contact_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        order = torch.argsort(scores.index_select(0, env_contact_ids), descending=True)
        chosen = env_contact_ids.index_select(0, order[:k])
        slot_count = int(chosen.numel())

        slots.contact_counts[env_id] = slot_count
        slots.contact_normals[env_id, :slot_count] = normals.index_select(0, chosen)
        slots.contact_depths[env_id, :slot_count] = depths.index_select(0, chosen)
        slots.contact_points_0[env_id, :slot_count] = points_0.index_select(0, chosen)
        slots.contact_points_1[env_id, :slot_count] = points_1.index_select(0, chosen)
        slots.contact_impulses[env_id, :slot_count] = per_contact_impulse.index_select(0, chosen)
        slots.contact_impulse_vectors[env_id, :slot_count] = per_contact_impulse_vec.index_select(0, chosen)

    return slots

