"""Quaternion arithmetic and rigid-body frame conversions for NeRD.

Convention
----------
All quaternions are **[w, x, y, z]** (scalar-first), matching Isaac Lab.
Hamilton product order: ``quat_multiply(a, b)`` returns the quaternion
equivalent to *first rotate by b, then rotate by a*.

Frame definitions
-----------------
- **world frame**: Isaac Lab world frame (z-up).  Positions in the dataset
  are already local (world - env_origin).
- **body frame**: Anchored at ``root_body_q`` (the held peg's pose).
  Positions and free vectors are rotated by the inverse of the anchor
  orientation; positions are additionally translated so the anchor position
  becomes the origin.

All public functions accept arbitrary leading batch dimensions unless noted.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Quaternion primitives  (convention: [w, x, y, z])
# ---------------------------------------------------------------------------


def quat_conjugate(q: Tensor) -> Tensor:
    """Conjugate (inverse for unit quaternions).  ``[w, -x, -y, -z]``."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product  a ⊗ b  with [w,x,y,z] layout.

    Rotation semantics: ``quat_rotate(quat_multiply(a, b), v)`` rotates
    *v* first by *b*, then by *a*.
    """
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.cat([w, x, y, z], dim=-1)


def quat_normalize(q: Tensor) -> Tensor:
    """Project back to the unit sphere."""
    return q / (q.norm(dim=-1, keepdim=True).clamp(min=1e-8))


def quat_positive_w(q: Tensor) -> Tensor:
    """Canonical form: flip sign so that w >= 0 (avoids double-cover issues)."""
    return torch.where(q[..., :1] < 0.0, -q, q)


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3-vector *v* by unit quaternion *q*.

    Input frame  → output frame:  world → body  when q = q_body.
    Uses the Rodrigues-like formula avoiding two full quat products.
    """
    u = q[..., 1:4]  # imaginary part
    w = q[..., :1]    # scalar part
    t = 2.0 * torch.linalg.cross(u, v, dim=-1)
    return v + w * t + torch.linalg.cross(u, t, dim=-1)


def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3-vector *v* by the inverse of unit quaternion *q*.

    Equivalent to ``quat_rotate(quat_conjugate(q), v)``.
    """
    u = q[..., 1:4]
    w = q[..., :1]
    t = 2.0 * torch.linalg.cross(u, v, dim=-1)
    return v - w * t + torch.linalg.cross(u, t, dim=-1)


# ---------------------------------------------------------------------------
# Quaternion delta  (orientation regression targets)
# ---------------------------------------------------------------------------


def quat_delta(q_from: Tensor, q_to: Tensor) -> Tensor:
    """Compute the delta quaternion *d* such that ``q_to ≈ d ⊗ q_from``.

    Returns a unit quaternion in canonical (w >= 0) form.

    This is the correct rotation-regression target for NeRD:  the model
    predicts *d*, and reconstruction applies ``quat_multiply(d, q_from)``.
    """
    d = quat_multiply(q_to, quat_conjugate(q_from))
    return quat_positive_w(quat_normalize(d))


def quat_apply_delta(q_from: Tensor, delta: Tensor) -> Tensor:
    """Reconstruct ``q_to`` from a base quaternion and a predicted delta.

    ``q_to = normalize(positive_w(delta ⊗ q_from))``
    """
    return quat_positive_w(quat_normalize(quat_multiply(delta, q_from)))


def quat_geodesic_distance(q1: Tensor, q2: Tensor) -> Tensor:
    """Geodesic angular distance in **radians** between two unit quaternions.

    Handles double-cover (q ≡ -q) via absolute dot product.
    Numerically stable: clamps the dot product away from exactly 1.0 to
    avoid NaN/Inf from ``acos(1.0)`` in float32.
    """
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0 - 1e-7)
    return 2.0 * torch.acos(dot)


# ---------------------------------------------------------------------------
# Rigid-body frame transforms
# ---------------------------------------------------------------------------


def positions_world_to_body(
    positions: Tensor,
    anchor_pos: Tensor,
    anchor_quat: Tensor,
) -> Tensor:
    """Transform position vectors from world/local frame to body frame.

    ``p_body = R(anchor_quat)^{-1} (p_world - anchor_pos)``

    Parameters
    ----------
    positions : (..., 3)
        Positions in world/local frame.
    anchor_pos : (..., 3)
        Body-frame origin in world/local frame.
    anchor_quat : (..., 4)
        Body-frame orientation in [w,x,y,z].

    Returns
    -------
    Tensor (..., 3) in body frame.
    """
    return quat_rotate_inverse(anchor_quat, positions - anchor_pos)


def positions_body_to_world(
    positions: Tensor,
    anchor_pos: Tensor,
    anchor_quat: Tensor,
) -> Tensor:
    """Inverse of :func:`positions_world_to_body`.

    ``p_world = R(anchor_quat) p_body + anchor_pos``
    """
    return quat_rotate(anchor_quat, positions) + anchor_pos


def vectors_world_to_body(vectors: Tensor, anchor_quat: Tensor) -> Tensor:
    """Rotate free 3-vectors (velocities, normals, gravity) to body frame.

    ``v_body = R(anchor_quat)^{-1} v_world``
    """
    return quat_rotate_inverse(anchor_quat, vectors)


def vectors_body_to_world(vectors: Tensor, anchor_quat: Tensor) -> Tensor:
    """Inverse of :func:`vectors_world_to_body`."""
    return quat_rotate(anchor_quat, vectors)


def quats_world_to_body(quats: Tensor, anchor_quat: Tensor) -> Tensor:
    """Express orientation quaternions in body frame.

    ``q_body = conj(anchor_quat) ⊗ q_world``

    When the object IS the anchor, the result is identity [1,0,0,0].
    """
    return quat_positive_w(quat_normalize(quat_multiply(quat_conjugate(anchor_quat), quats)))


def quats_body_to_world(quats: Tensor, anchor_quat: Tensor) -> Tensor:
    """Inverse of :func:`quats_world_to_body`.

    ``q_world = anchor_quat ⊗ q_body``
    """
    return quat_positive_w(quat_normalize(quat_multiply(anchor_quat, quats)))
