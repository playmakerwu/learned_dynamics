# Learned Dynamics — NeRD Trajectory Collector

Data-driven dynamics modeling for the Isaac Lab Factory PegInsert task.
This repository collects simulator trajectories and trains NeRD (Neural Robot Dynamics) models.

## Trajectory Collector

The primary collector is `collect_trajectories_with_physx_contacts.py`.
It replays a trained RL-Games PPO policy in Isaac Lab and saves NeRD-friendly
HDF5 datasets with per-contact geometry and applied joint torques.

### What the collector saves

Each timestep in the saved trajectory contains:

| Field | Shape | Description |
|---|---|---|
| `states` | `[47]` | Simulator state vector (joint pos/vel, EE pose/vel, peg pose/vel, socket pose) |
| `next_states` | `[47]` | Next simulator state before any auto-reset |
| `applied_joint_torque` | `[7]` | **Actual joint torque sent to PhysX** via Operational Space Control, read from `robot.data.applied_torque` after the last physics substep. This is NOT the raw 6-dim RL policy output. |
| `gravity_dir` | `[3]` | Gravity direction in world frame |
| `root_body_q` | `[7]` | Peg root pose (pos + quat) in local frame |
| `contact_normals` | `[K, 3]` | Unit contact normals (force direction on peg) |
| `contact_depths` | `[K]` | Clamped penetration depths |
| `contact_thicknesses` | `[K]` | Contact thickness (constant) |
| `contact_points_0` | `[K, 3]` | Contact position from PhysX |
| `contact_points_1` | `[K, 3]` | Reconstructed point on the other body |
| `contact_impulses` | `[K]` | Impulse magnitude per contact slot |
| `contact_impulse_vectors` | `[K, 3]` | 3D impulse vector per contact slot |
| `contact_identities` | `[K]` | **Peg-centric binary identity**: `0` = hole/environment, `1` = robot |
| `contact_counts` | scalar | Number of valid (populated) contact slots |
| `dones` | bool | Episode boundary marker |
| `terminated` | bool | True environment termination |
| `truncated` | bool | Time-limit truncation |

Where `K = 64` (default number of contact slots).

### Key design decisions

**Applied joint torque (not raw RL action)**:
The saved control signal is the 7-dim arm joint torque that is actually applied to
the Franka robot in PhysX. The RL policy outputs a 6-dim action, which passes through
EMA smoothing, Operational Space Control (Jacobian transpose + nullspace projection),
and ImplicitActuator effort-limit clipping before becoming the final torque. The
collector reads `robot.data.applied_torque[:, :7]` after the last decimation substep,
**before** any episode reset can overwrite it.

**Peg-centric contacts**:
Only contacts involving the peg (held_asset) are saved. Each contact is classified
by the identity of the **other** body:
- `contact_identity = 0`: the other body is the hole/socket (fixed_asset)
- `contact_identity = 1`: the other body is the robot (manipulator links/fingers)

Contacts not involving the peg are discarded. The identity label is slot-aligned
with all other contact fields (normals, points, impulses, depths).

**Top-64 contact slots**:
Contacts are ranked by impulse magnitude (with depth as tiebreaker) and the top 64
are kept per environment per step. Contacts from both peg-vs-hole and peg-vs-robot
are ranked together in the same pool.

**Contact report API**:
The collector uses `get_physx_simulation_interface().get_contact_report()` — the
low-level PhysX immediate contact report API. This requires CPU simulation
(GPU simulation does not reliably report contacts for Factory tasks). The policy
can still run on GPU via `--policy_device cuda:0`.

### How to run

```bash
# Basic usage (CPU simulation, GPU policy inference)
python collect_trajectories_with_physx_contacts.py \
    --device cpu \
    --policy_device cuda:0 \
    --num_envs 32 \
    --num_trajectories 512 \
    --output_path recordings/nerd_peg_insert_trajectories.hdf5 \
    --headless

# With explicit checkpoint
python collect_trajectories_with_physx_contacts.py \
    --device cpu \
    --policy_device cuda:0 \
    --checkpoint logs/peg_insert_rlgames/run1/nn/last_peg_insert_rlgames_ep_2000_rew_39.93.pth \
    --num_envs 64 \
    --num_trajectories 1024 \
    --output_path recordings/my_dataset.hdf5 \
    --headless
```

**Important**: Use `--device cpu` for reliable contact extraction. The PhysX contact
report API does not work reliably on GPU for Factory tasks. Use `--policy_device cuda:0`
to keep policy inference fast.

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--device` | `cuda:0` | Simulation device (use `cpu` for contact reports) |
| `--policy_device` | same as device | Policy inference device |
| `--num_envs` | 32 | Number of parallel environments (max 512 for Factory on GPU) |
| `--num_trajectories` | 256 | Total trajectories to collect |
| `--output_path` | `recordings/nerd_peg_insert_trajectories.hdf5` | Output HDF5 path |
| `--checkpoint` | latest | RL-Games checkpoint to replay |
| `--task` | `Isaac-Factory-PegInsert-Direct-v0` | Isaac Lab task name |
| `--episode_length_steps` | task default | Override episode length |
| `--horizon_steps` | episode length | Storage horizon per trajectory |
| `--action_noise_std` | 0.0 | Noise added to policy output before stepping |
| `--seed` | 42 | Random seed |
| `--headless` | flag | Run without rendering |
| `--stochastic_policy` | flag | Use stochastic (not deterministic) policy |
| `--disable_fabric` | flag | Disable Isaac Lab Fabric |

### Configuration

All defaults are in `nerd_collector/config.py` (`CollectorConfig` dataclass).
Key settings:

- `contact_slot_count_k = 64` — number of contact slots per environment per step
- `contact_source_asset_name = "held_asset"` — the peg (reference object for contacts)
- `contact_target_asset_name = "fixed_asset"` — the hole/socket
- `robot_asset_name = "robot"` — the Franka manipulator
- `robot_joint_count = 7` — number of arm joints (excluding gripper)
- `max_depth_clamp = 0.02` — maximum penetration depth (meters)

### HDF5 output format

The output file contains:
- One HDF5 group per trajectory (`traj_000000`, `traj_000001`, ...)
- Each group has datasets for all fields listed above, with shape `[T, ...]`
- Root-level metadata attributes: task name, state layout, torque dim, contact slot count, step dt, etc.

### Downstream compatibility

The new collector saves `applied_joint_torque` instead of the old `joint_acts` field,
and adds `contact_identities`. Downstream modules that expect the old format will need
updates:

- `nerd_bridge/training.py` — `DEFAULT_INPUT_KEYS` references `"joint_acts"`
- `nerd_bridge/dataset_utils.py` — conversion code references `"joint_acts"`
- `nerd_eval/utils.py` — dataset loading expects `"joint_acts"`
- `rollout_nerd_eval.py` — rollout code references `"joint_acts"`
- `tests/test_preprocessing.py` — test fixtures use `"joint_acts"`

These are **not** updated in this version. When adapting downstream code, change
`"joint_acts"` to `"applied_joint_torque"` and update dimension from 6 to 7.

## Project Structure

```
collect_trajectories_with_physx_contacts.py  # Primary trajectory collector
nerd_collector/
    config.py              # CollectorConfig dataclass
    collector.py           # Shared library: StateAssembler, EpisodeStorage, build_writer, etc.
    physx_contact_report.py  # PhysXContactReportExtractor (peg-centric contact extraction)
    contact_utils.py       # FixedSlotContacts dataclass, slot assignment utilities
    hdf5_utils.py          # TrajectoryHDF5Writer
    net_contact_force.py   # NetContactForceExtractor (GPU fallback, aggregate force only)
nerd_bridge/
    training.py            # NeRD model training loop
    preprocessing.py       # Body-frame conversion, contact masking, target computation
    frame_utils.py         # Quaternion arithmetic and rigid-body transforms
    dataset_utils.py       # HDF5 format conversion and dataset splitting
    common.py              # Path constants, import configuration
nerd_eval/                 # Evaluation utilities
train_nerd_from_base.py    # CLI entrypoint for NeRD training
rollout_nerd_eval.py       # Autoregressive NeRD rollout evaluation
compare_solver24_vs_nerd.py  # NeRD vs solver24 comparison metrics
common.py                 # Shared Isaac Lab + RL-Games infrastructure
train.py                  # RL policy training (PPO via RL-Games)
play.py                   # Policy replay / visualization
```
