"""Low-level PhysX contact-report extraction for NeRD-ready contact slots.

This module provides an alternative contact path to the existing raw tensor contact view.
It uses Isaac Sim / PhysX's lower-level contact report immediate API:

* direct simulator read: ``get_physx_simulation_interface().get_contact_report()``
* direct simulator read: per-contact ``position``, ``normal``, ``impulse``, ``separation``

It then converts the variable-length contact stream from one environment step into the
fixed ``K`` contact slots expected by the NeRD pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from .config import CollectorConfig
from .contact_utils import FixedSlotContacts, empty_fixed_slot_contacts, safe_normalize


_ENV_PATH_RE = re.compile(r"^/World/envs/env_(\d+)(/.*)$")


@dataclass(slots=True)
class ContactReportDebugStats:
    """Summary stats for the latest environment step."""

    raw_header_count: int = 0
    matching_header_count: int = 0
    matching_contact_count: int = 0
    slot_env_count: int = 0
    slot_contact_count: int = 0
    max_depth: float = 0.0
    sample_position: list[float] | None = None
    sample_normal: list[float] | None = None
    sample_separation: float | None = None


def _iter_rigid_body_prims_under(root_prim_path: str) -> list[str]:
    """Return all rigid-body prim paths under a root prim."""

    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Invalid root prim for rigid-body discovery: {root_prim_path}")

    rigid_body_paths: list[str] = []
    prim_queue = [root_prim]
    while prim_queue:
        prim = prim_queue.pop(0)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_paths.append(prim.GetPath().pathString)
        prim_queue.extend(list(prim.GetChildren()))
    return rigid_body_paths


def ensure_contact_report_api(root_prim_path: str, *, threshold: float = 0.0) -> list[str]:
    """Force-enable PhysX contact reporting on rigid bodies under a prim root."""

    import omni.usd
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    rigid_body_paths = _iter_rigid_body_prims_under(root_prim_path)
    if not rigid_body_paths:
        raise RuntimeError(f"No rigid bodies found under {root_prim_path} for contact-report enabling.")

    enabled_paths: list[str] = []
    for prim_path in rigid_body_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        else:
            cr_api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim.GetPrimPath())
        cr_api.CreateThresholdAttr().Set(float(threshold))

        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPrimPath())
            rb_api.CreateSleepThresholdAttr().Set(0.0)
        enabled_paths.append(prim_path)
    return enabled_paths


def _split_env_path(path: str) -> tuple[int | None, str]:
    """Parse a stage path into ``(env_id, local_path_after_env_root)``."""

    match = _ENV_PATH_RE.match(path)
    if match is None:
        return None, path
    return int(match.group(1)), match.group(2)


def _match_source_target_pair(
    *,
    actor0: str,
    actor1: str,
    collider0: str,
    collider1: str,
    source_local_root: str,
    target_local_root: str,
) -> tuple[int | None, bool] | None:
    """Return ``(env_id, source_is_first)`` when a pair matches the desired assets."""

    collider0_env, collider0_local = _split_env_path(collider0)
    collider1_env, collider1_local = _split_env_path(collider1)
    actor0_env, actor0_local = _split_env_path(actor0)
    actor1_env, actor1_local = _split_env_path(actor1)

    first_is_source = (
        (collider0_env is not None and collider0_local.startswith(source_local_root))
        or (actor0_env is not None and actor0_local.startswith(source_local_root))
    )
    second_is_target = (
        (collider1_env is not None and collider1_local.startswith(target_local_root))
        or (actor1_env is not None and actor1_local.startswith(target_local_root))
    )
    if first_is_source and second_is_target:
        env_id = collider0_env if collider0_env is not None else actor0_env
        return env_id, True

    first_is_target = (
        (collider0_env is not None and collider0_local.startswith(target_local_root))
        or (actor0_env is not None and actor0_local.startswith(target_local_root))
    )
    second_is_source = (
        (collider1_env is not None and collider1_local.startswith(source_local_root))
        or (actor1_env is not None and actor1_local.startswith(source_local_root))
    )
    if first_is_target and second_is_source:
        env_id = collider1_env if collider1_env is not None else actor1_env
        return env_id, False

    return None


class PhysXContactReportExtractor:
    """Collect low-level PhysX contact reports across one Isaac Lab environment step.

    The environment step consists of ``decimation`` physics substeps. This extractor reads
    the contact report after each substep, keeps the contacts involving the configured
    source and target assets, and maps the variable-length raw contacts into fixed NeRD
    slots at the end of the environment step.
    """

    def __init__(self, env: Any, cfg: CollectorConfig):
        import carb
        from omni.physx import get_physx_simulation_interface

        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.dtype = torch.float32
        self.source_asset = env.scene.articulations[cfg.contact_source_asset_name]
        self.target_asset = env.scene.articulations[cfg.contact_target_asset_name]
        self.source_root = self.source_asset.cfg.prim_path.replace(".*", "0")
        self.target_root = self.target_asset.cfg.prim_path.replace(".*", "0")
        self.source_local_root = self.source_root.replace("/World/envs/env_0", "")
        self.target_local_root = self.target_root.replace("/World/envs/env_0", "")
        self.sim_interface = get_physx_simulation_interface()

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.enabled_source_contact_report_prims = ensure_contact_report_api(self.source_root, threshold=0.0)
        self.enabled_target_contact_report_prims = ensure_contact_report_api(self.target_root, threshold=0.0)

        self._records_per_env: list[list[dict[str, Any]]] = [[] for _ in range(env.num_envs)]
        self.total_matching_contacts_seen = 0
        self.total_env_steps_with_contacts = 0
        self.last_debug_stats = ContactReportDebugStats()

    def reset_statistics(self) -> None:
        """Clear aggregate counters so preflight probes do not pollute rollout stats."""

        self.total_matching_contacts_seen = 0
        self.total_env_steps_with_contacts = 0
        self.last_debug_stats = ContactReportDebugStats()
        self._records_per_env = [[] for _ in range(self.env.num_envs)]

    def begin_step(self) -> None:
        """Clear the current step buffer before starting substep simulation."""

        self._records_per_env = [[] for _ in range(self.env.num_envs)]
        self.last_debug_stats = ContactReportDebugStats()

    def capture_substep_reports(self) -> None:
        """Read the latest PhysX contact report for the just-finished physics substep."""

        from pxr import PhysicsSchemaTools

        contact_headers, contact_data = self.sim_interface.get_contact_report()
        self.last_debug_stats.raw_header_count += len(contact_headers)

        for header in contact_headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
            collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))

            match = _match_source_target_pair(
                actor0=actor0,
                actor1=actor1,
                collider0=collider0,
                collider1=collider1,
                source_local_root=self.source_local_root,
                target_local_root=self.target_local_root,
            )
            if match is None:
                continue

            env_id, source_is_first = match
            if env_id is None or not (0 <= env_id < self.env.num_envs):
                continue

            self.last_debug_stats.matching_header_count += 1
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            self.last_debug_stats.matching_contact_count += int(header.num_contact_data)

            for index in range(start, stop):
                raw_normal = torch.tensor(contact_data[index].normal, device=self.device, dtype=self.dtype)
                if not source_is_first:
                    raw_normal = -raw_normal
                raw_position = torch.tensor(contact_data[index].position, device=self.device, dtype=self.dtype)
                raw_impulse = torch.tensor(contact_data[index].impulse, device=self.device, dtype=self.dtype)
                if not source_is_first:
                    raw_impulse = -raw_impulse
                raw_separation = float(contact_data[index].separation)
                self._records_per_env[env_id].append(
                    {
                        # Direct simulator reads from PhysX contact reports.
                        "position": raw_position,
                        "normal": raw_normal,
                        "impulse_norm": torch.linalg.norm(raw_impulse),
                        "impulse_vector": raw_impulse,
                        "separation": raw_separation,
                    }
                )

                if self.last_debug_stats.sample_position is None:
                    self.last_debug_stats.sample_position = raw_position.detach().cpu().tolist()
                    self.last_debug_stats.sample_normal = raw_normal.detach().cpu().tolist()
                    self.last_debug_stats.sample_separation = raw_separation

    def end_step(self) -> FixedSlotContacts:
        """Project this step's raw contact reports into fixed NeRD slots."""

        slots = empty_fixed_slot_contacts(
            num_envs=self.env.num_envs,
            k=self.cfg.contact_slot_count_k,
            device=self.device,
            dtype=self.dtype,
            contact_thickness=self.cfg.contact_thickness,
        )

        total_slot_contacts = 0
        active_envs = 0
        max_depth = 0.0

        for env_id, env_records in enumerate(self._records_per_env):
            if not env_records:
                continue

            positions = torch.stack([record["position"] for record in env_records], dim=0)
            normals = safe_normalize(torch.stack([record["normal"] for record in env_records], dim=0))
            impulse_norms = torch.stack([record["impulse_norm"] for record in env_records], dim=0).to(dtype=self.dtype)
            impulse_vectors = torch.stack([record["impulse_vector"] for record in env_records], dim=0).to(dtype=self.dtype)
            separations = torch.tensor(
                [record["separation"] for record in env_records], device=self.device, dtype=self.dtype
            )

            # Derived quantities for NeRD representation.
            depths = torch.clamp(-separations, min=0.0, max=self.cfg.max_depth_clamp)
            points_0 = positions
            points_1 = points_0 + normals * depths.unsqueeze(-1)
            # Impulse magnitude is the primary ranking signal. Depth is a small tiebreaker
            # since the PhysX solver resolves penetrations to near-zero separation.
            scores = impulse_norms + 1.0e-6 * depths

            chosen = torch.argsort(scores, descending=True)[: self.cfg.contact_slot_count_k]
            slot_count = int(chosen.numel())
            if slot_count == 0:
                continue

            slots.contact_counts[env_id] = slot_count
            slots.contact_normals[env_id, :slot_count] = normals.index_select(0, chosen)
            slots.contact_depths[env_id, :slot_count] = depths.index_select(0, chosen)
            slots.contact_points_0[env_id, :slot_count] = points_0.index_select(0, chosen)
            slots.contact_points_1[env_id, :slot_count] = points_1.index_select(0, chosen)
            slots.contact_impulses[env_id, :slot_count] = impulse_norms.index_select(0, chosen)
            slots.contact_impulse_vectors[env_id, :slot_count] = impulse_vectors.index_select(0, chosen)

            active_envs += 1
            total_slot_contacts += slot_count
            max_depth = max(max_depth, float(torch.max(depths).item()))

        self.last_debug_stats.slot_env_count = active_envs
        self.last_debug_stats.slot_contact_count = total_slot_contacts
        self.last_debug_stats.max_depth = max_depth
        self.total_matching_contacts_seen += self.last_debug_stats.matching_contact_count
        if total_slot_contacts > 0:
            self.total_env_steps_with_contacts += 1
        return slots
