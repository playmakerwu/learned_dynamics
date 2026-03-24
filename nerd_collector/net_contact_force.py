"""GPU-compatible net contact force extractor for NeRD trajectory collection.

Uses ``RigidContactView.get_net_contact_forces(dt)`` — the only PhysX contact API
confirmed to work on GPU for the Factory peg-insert task. Returns the aggregate
net contact force vector (3D) per environment rather than per-contact geometry.

The per-contact APIs (``get_contact_report()``, ``get_contact_data()``) produce
zero contacts on GPU for this task due to unsupported GPU contact filters on the
Factory FixedAsset collider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class NetContactForceDebugStats:
    """Summary statistics for the latest environment step."""

    num_envs_with_force: int = 0
    max_force_magnitude: float = 0.0
    mean_force_magnitude: float = 0.0
    sample_force: list[float] | None = None


class NetContactForceExtractor:
    """Extract GPU-resident net contact forces for the peg body each environment step.

    This extractor creates an unfiltered ``RigidContactView`` for the held (peg)
    asset. Unlike the filtered per-contact APIs, this path does NOT trigger the
    "GPU contact filter for collider ... is not supported" warning because it does
    not use filter patterns.

    The returned force is the sum of all contact forces acting on the peg body,
    including contacts with the hole, the table, and the robot gripper fingers.
    """

    def __init__(self, env: Any, cfg: Any):
        from isaacsim.core.simulation_manager import SimulationManager

        import carb

        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.dtype = torch.float32

        self.source_asset = env.scene.articulations[cfg.contact_source_asset_name]

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)

        body_names_regex = r"(" + "|".join(self.source_asset.body_names) + r")"
        sensor_glob = f"{self.source_asset.cfg.prim_path}/{body_names_regex}".replace(".*", "*")

        physics_sim_view = SimulationManager.get_physics_sim_view()
        # No filter patterns — avoids the unsupported GPU contact filter issue.
        self._contact_view = physics_sim_view.create_rigid_contact_view(sensor_glob)

        sensor_count = int(self._contact_view.sensor_count)
        if sensor_count <= 0:
            raise RuntimeError(
                f"Net contact force view initialized with zero sensors for '{sensor_glob}'."
            )
        if sensor_count % env.num_envs != 0:
            raise RuntimeError(
                f"Sensor count ({sensor_count}) does not divide evenly across "
                f"environments ({env.num_envs})."
            )
        self._bodies_per_env = sensor_count // env.num_envs
        self.physics_dt = float(env.physics_dt)

        self.total_steps_with_force = 0
        self.total_steps = 0
        self.last_debug_stats = NetContactForceDebugStats()

    def capture(self) -> torch.Tensor:
        """Read net contact forces from GPU and return ``[num_envs, 3]`` tensor.

        Forces are aggregated across all rigid bodies of the peg asset per env.
        Zero forces indicate no contact for that environment.
        """

        raw_forces = self._contact_view.get_net_contact_forces(dt=self.physics_dt)
        if raw_forces.numel() == 0:
            force = torch.zeros((self.env.num_envs, 3), device=self.device, dtype=self.dtype)
        else:
            # raw_forces: [num_envs * bodies_per_env, 3] → [num_envs, bodies_per_env, 3]
            force = raw_forces.view(self.env.num_envs, self._bodies_per_env, 3).sum(dim=1)

        # Update debug stats.
        magnitudes = torch.norm(force, dim=-1)
        active_mask = magnitudes > 0.0
        n_active = int(active_mask.sum().item())

        self.last_debug_stats = NetContactForceDebugStats(
            num_envs_with_force=n_active,
            max_force_magnitude=float(magnitudes.max().item()),
            mean_force_magnitude=float(magnitudes.mean().item()),
            sample_force=force[active_mask][0].tolist() if n_active > 0 else None,
        )

        self.total_steps += 1
        if n_active > 0:
            self.total_steps_with_force += 1

        return force

    def reset_statistics(self) -> None:
        """Clear aggregate counters for a fresh rollout."""

        self.total_steps_with_force = 0
        self.total_steps = 0
        self.last_debug_stats = NetContactForceDebugStats()
