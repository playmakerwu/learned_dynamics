"""Data-driven NeRD training helpers with body-frame preprocessing and
quaternion-aware targets.

Changes from the original implementation
-----------------------------------------
1. **Body-frame conversion** – all positions, orientations, velocities,
   contact geometry, and gravity are expressed in a body frame anchored at
   the held peg's pose (``root_body_q``).  This makes the learning problem
   position/rotation-invariant.
2. **Quaternion-aware targets** – orientation deltas are computed via proper
   rotation composition (``delta = q_to ⊗ conj(q_from)``) instead of naive
   subtraction.  Reconstruction uses ``next_q = delta ⊗ q_from``.
3. **Contact masking** – inactive contact slots (beyond ``contact_counts``)
   are zeroed out before the model sees them.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .common import (
    DEFAULT_CONVERTED_DATASET,
    DEFAULT_TEST_INDICES,
    DEFAULT_TRAIN_INDICES,
    PROJECT_ROOT,
    configure_nerd_imports,
    default_device,
    ensure_directory,
    save_json,
)
from .preprocessing import (
    StateLayout,
    compute_target as _compute_target_body,
    preprocess_batch,
    reconstruct_next_state,
)

configure_nerd_imports()

from models.models import ModelMixedInput
from utils.running_mean_std import RunningMeanStd


DEFAULT_INPUT_KEYS: tuple[str, ...] = (
    "states",
    "joint_acts",
    "gravity_dir",
    "root_body_q",
    "contact_normals",
    "contact_points_0",
    "contact_impulses",
)
# Per-contact slot inputs from CPU collection via get_contact_report().
# contact_impulses is the impulse magnitude per slot — the most informative
# contact signal since depths are near-zero (192 solver iterations).
# Missing keys are auto-skipped with a warning for backward compatibility.

# Keys that the preprocessing pipeline reads but are NOT model inputs.
# They must be loaded from the dataset so that preprocessing can use them.
_PREPROCESSING_EXTRA_KEYS: frozenset[str] = frozenset({"contact_counts"})


@dataclass(slots=True)
class TrainConfig:
    """User-adjustable parameters for NeRD training from the converted peg-insert dataset."""

    converted_dataset_path: Path = DEFAULT_CONVERTED_DATASET
    train_indices_path: Path = DEFAULT_TRAIN_INDICES
    test_indices_path: Path = DEFAULT_TEST_INDICES
    output_dir: Path = PROJECT_ROOT / "outputs" / "nerd_peg_insert"
    seed: int = 42
    batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 1.0e-4
    history_length: int = 10
    num_workers: int = 0
    device: str = field(default_factory=default_device)
    checkpoint_path: Path | None = None
    evaluation_frequency: int = 1
    max_train_batches_per_epoch: int | None = None
    max_eval_batches: int | None = None
    normalization_batches: int | None = 20
    grad_clip_norm: float = 1.0
    input_keys: tuple[str, ...] = DEFAULT_INPUT_KEYS
    weight_decay: float = 1e-4
    normalized_loss: bool = True

    # Body-frame and preprocessing options (new, default=on for NeRD correctness).
    use_body_frame: bool = True
    use_quat_targets: bool = True
    use_contact_masking: bool = True
    num_contact_slots: int = 16

    # Official NeRD model hyperparameters, adapted for the peg-insert dataset.
    encoder_layer_sizes: tuple[int, ...] = (256, 256)
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_embedding_dim: int = 128
    transformer_dropout: float = 0.1
    model_head_layer_sizes: tuple[int, ...] = (128, 64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IndexedTrajectoryWindowDataset(Dataset):
    """Lazy trajectory-window dataset backed by a converted NeRD-style HDF5 file."""

    def __init__(
        self,
        hdf5_path: str | Path,
        trajectory_indices: Sequence[int] | str | Path,
        *,
        sequence_length: int,
        keys: Sequence[str],
        max_windows: int | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser().resolve()
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        if isinstance(trajectory_indices, (str, Path)):
            self.trajectory_indices = np.load(Path(trajectory_indices).expanduser().resolve()).astype(np.int32)
        else:
            self.trajectory_indices = np.asarray(list(trajectory_indices), dtype=np.int32)

        self.keys = list(keys)
        self._file: h5py.File | None = None
        self._data_group: h5py.Group | None = None

        with h5py.File(self.hdf5_path, "r") as file:
            data_group = file["data"]
            available_keys = set(data_group.keys())
            missing = [key for key in self.keys if key not in available_keys]
            if missing:
                raise KeyError(f"Converted dataset is missing requested keys: {missing}")

            self.traj_lengths = data_group["traj_lengths"][self.trajectory_indices].astype(np.int32)
            self.feature_dims = {
                key: int(np.prod(data_group[key].shape[2:])) if data_group[key].ndim > 2 else 1
                for key in self.keys
            }

        mapping: list[tuple[int, int]] = []
        for relative_index, traj_length in enumerate(self.traj_lengths.tolist()):
            for start in range(max(0, int(traj_length) - self.sequence_length + 1)):
                mapping.append((relative_index, start))
        if max_windows is not None:
            mapping = mapping[: max_windows]
        self.mapping = np.asarray(mapping, dtype=np.int32)

    def _ensure_open(self) -> h5py.Group:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r", swmr=True, libver="latest")
            self._data_group = self._file["data"]
        assert self._data_group is not None
        return self._data_group

    def __len__(self) -> int:
        return int(self.mapping.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data_group = self._ensure_open()
        relative_traj_index, start = self.mapping[index].tolist()
        trajectory_index = int(self.trajectory_indices[relative_traj_index])

        sample: dict[str, torch.Tensor] = {}
        for key in self.keys:
            raw = data_group[key][start : start + self.sequence_length, trajectory_index, ...]
            flattened = raw.reshape(self.sequence_length, -1).astype(np.float32, copy=False)
            sample[key] = torch.from_numpy(flattened)
        return sample


def make_model_network_cfg(config: TrainConfig) -> dict[str, Any]:
    """Construct the upstream NeRD network configuration dictionary."""

    return {
        "normalize_input": True,
        "normalize_output": True,
        "encoder": {
            "low_dim": {
                "layer_sizes": list(config.encoder_layer_sizes),
                "activation": "relu",
                "layernorm": False,
            }
        },
        "transformer": {
            "block_size": max(32, config.history_length),
            "n_layer": int(config.transformer_layers),
            "n_head": int(config.transformer_heads),
            "n_embd": int(config.transformer_embedding_dim),
            "dropout": float(config.transformer_dropout),
            "bias": False,
        },
        "model": {
            "mlp": {
                "layer_sizes": list(config.model_head_layer_sizes),
                "activation": "relu",
                "layernorm": False,
            }
        },
    }


def make_input_cfg(input_keys: Sequence[str]) -> dict[str, Any]:
    """Construct the upstream NeRD input configuration dictionary."""

    return {"low_dim": list(input_keys)}


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch for deterministic dataset splits and training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Move a batch of tensors to the target device."""

    return {key: value.to(device) for key, value in batch.items()}


def compute_target(
    batch: dict[str, torch.Tensor],
    layout: StateLayout | None = None,
    *,
    use_quat_targets: bool = True,
) -> torch.Tensor:
    """Compute the NeRD prediction target from a (preprocessed) batch.

    When *layout* is provided and *use_quat_targets* is True, quaternion
    dimensions use proper rotation-delta targets.  Otherwise, falls back to
    naive ``next_states - states`` for backward compatibility.
    """
    if layout is not None and use_quat_targets:
        return _compute_target_body(batch["states"], batch["next_states"], layout)
    return batch["next_states"] - batch["states"]


def load_state_layout(hdf5_path: Path) -> StateLayout:
    """Read the state-layout metadata from a converted NeRD HDF5 dataset."""
    with h5py.File(hdf5_path, "r") as f:
        raw = f["data"].attrs.get("state_layout")
        if raw is None:
            raise KeyError(f"Dataset {hdf5_path} has no 'state_layout' attribute.")
    return StateLayout.from_hdf5_attr(raw)


def build_model_from_sample(
    sample_batch: dict[str, torch.Tensor],
    *,
    config: TrainConfig,
) -> ModelMixedInput:
    """Build the official NeRD mixed-input model from one sample batch."""

    input_sample = {key: sample_batch[key].to(config.device) for key in config.input_keys}
    output_dim = int(sample_batch["states"].shape[-1])
    model = ModelMixedInput(
        input_sample=input_sample,
        output_dim=output_dim,
        input_cfg=make_input_cfg(config.input_keys),
        network_cfg=make_model_network_cfg(config),
        device=config.device,
    )
    model.to(config.device)
    return model


def rms_to_state(rms: RunningMeanStd) -> dict[str, Any]:
    """Serialize a RunningMeanStd object into a checkpoint-friendly dictionary."""

    return {
        "mean": rms.mean.detach().cpu(),
        "var": rms.var.detach().cpu(),
        "count": float(rms.count),
    }


def rms_from_state(state: dict[str, Any], *, device: str) -> RunningMeanStd:
    """Restore RunningMeanStd statistics from a saved state dictionary."""

    rms = RunningMeanStd(shape=tuple(state["mean"].shape), device=device)
    rms.mean = state["mean"].to(device)
    rms.var = state["var"].to(device)
    rms.count = float(state["count"])
    return rms


def attach_normalization_stats(
    model: ModelMixedInput,
    dataloader: DataLoader,
    *,
    config: TrainConfig,
    layout: StateLayout | None = None,
) -> dict[str, Any]:
    """Compute and attach input/output normalization statistics.

    Statistics are computed **after** preprocessing (body-frame conversion,
    contact masking) so the model's running-mean-std tracks the distribution
    the model actually sees.
    """

    input_rms: dict[str, RunningMeanStd] = {}
    output_rms: RunningMeanStd | None = None

    normalization_limit = normalize_batch_limit(config.normalization_batches)
    for step, batch in enumerate(dataloader):
        if normalization_limit is not None and step >= normalization_limit:
            break
        batch = move_batch_to_device(batch, config.device)

        # Apply the same preprocessing the training loop uses.
        if layout is not None:
            batch = preprocess_batch(
                batch,
                layout,
                use_body_frame=config.use_body_frame,
                apply_contact_masking=config.use_contact_masking,
                num_contact_slots=config.num_contact_slots,
            )

        target = compute_target(
            batch,
            layout=layout,
            use_quat_targets=config.use_quat_targets,
        )

        for key in config.input_keys:
            if key not in input_rms:
                input_rms[key] = RunningMeanStd(shape=batch[key].shape[2:], device=config.device)
            input_rms[key].update(batch[key], batch_dim=True, time_dim=True)

        if output_rms is None:
            output_rms = RunningMeanStd(shape=target.shape[2:], device=config.device)
        output_rms.update(target, batch_dim=True, time_dim=True)

    if output_rms is None:
        raise RuntimeError("Normalization statistics could not be computed because the dataloader is empty.")

    model.set_input_rms(input_rms)
    model.set_output_rms(output_rms)

    return {
        "input_rms": {key: rms_to_state(rms) for key, rms in input_rms.items()},
        "output_rms": rms_to_state(output_rms),
    }


def load_normalization_stats(model: ModelMixedInput, stats_state: dict[str, Any], *, device: str) -> None:
    """Restore previously saved normalization statistics onto a model."""

    input_rms = {key: rms_from_state(value, device=device) for key, value in stats_state["input_rms"].items()}
    output_rms = rms_from_state(stats_state["output_rms"], device=device)
    model.set_input_rms(input_rms)
    model.set_output_rms(output_rms)


def train_or_eval_epoch(
    model: ModelMixedInput,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
    max_batches: int | None,
    layout: StateLayout | None = None,
) -> dict[str, float]:
    """Run one training or evaluation epoch.

    When *layout* is provided, each batch is preprocessed before the forward
    pass (body-frame conversion, contact masking, quaternion targets).
    """

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_raw_loss = 0.0
    total_next_state_mse = 0.0
    batches = 0

    use_normalized_loss = (
        config.normalized_loss
        and hasattr(model, "output_rms")
        and model.output_rms is not None
    )

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, config.device)

        # --- Preprocessing: body-frame conversion + contact masking ---
        if layout is not None:
            batch = preprocess_batch(
                batch,
                layout,
                use_body_frame=config.use_body_frame,
                apply_contact_masking=config.use_contact_masking,
                num_contact_slots=config.num_contact_slots,
            )

        inputs = {key: batch[key] for key in config.input_keys}
        target = compute_target(
            batch,
            layout=layout,
            use_quat_targets=config.use_quat_targets,
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(inputs)
            raw_loss = torch.nn.functional.mse_loss(prediction, target)

            if use_normalized_loss:
                norm_pred = model.output_rms.normalize(prediction)
                norm_target = model.output_rms.normalize(target)
                loss = torch.nn.functional.mse_loss(norm_pred, norm_target)
            else:
                loss = raw_loss

            # Reconstruct next state using the correct update rule.
            if layout is not None and config.use_quat_targets:
                predicted_next_states = reconstruct_next_state(
                    batch["states"], prediction, layout,
                )
            else:
                predicted_next_states = batch["states"] + prediction
            next_state_mse = torch.nn.functional.mse_loss(
                predicted_next_states, batch["next_states"],
            )

            if is_training:
                loss.backward()
                if config.grad_clip_norm > 0.0:
                    clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.item())
        total_raw_loss += float(raw_loss.item())
        total_next_state_mse += float(next_state_mse.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Dataloader produced zero batches. Check the split indices and history length.")

    return {
        "loss": total_loss / batches,
        "raw_loss": total_raw_loss / batches,
        "next_state_mse": total_next_state_mse / batches,
        "num_batches": float(batches),
    }


def normalize_batch_limit(value: int | None) -> int | None:
    """Interpret non-positive batch limits as 'use the full dataloader'."""

    if value is None:
        return None
    if int(value) <= 0:
        return None
    return int(value)


def save_training_checkpoint(
    path: Path,
    *,
    model: ModelMixedInput,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    normalization_state: dict[str, Any],
    metrics: dict[str, Any],
    state_layout_list: list[dict[str, Any]] | None = None,
) -> None:
    """Save a checkpoint for later continuation or evaluation.

    The checkpoint now also stores:
    - ``state_layout``: the parsed state-layout list, so evaluation scripts
      can reconstruct the preprocessing pipeline without the dataset file.
    - ``use_body_frame``, ``use_quat_targets``, ``use_contact_masking``:
      preprocessing flags for consistent eval-time behaviour.
    """

    ensure_directory(path)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "input_keys": list(config.input_keys),
            "network_cfg": make_model_network_cfg(config),
            "input_cfg": make_input_cfg(config.input_keys),
            "normalization_state": normalization_state,
            "metrics": metrics,
            "state_layout": state_layout_list,
            "use_body_frame": config.use_body_frame,
            "use_quat_targets": config.use_quat_targets,
            "use_contact_masking": config.use_contact_masking,
            "num_contact_slots": config.num_contact_slots,
        },
        path,
    )


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Train a data-driven NeRD model on the converted peg-insert dataset."""

    ensure_directory(config.output_dir)
    set_random_seeds(config.seed)

    train_batch_limit = normalize_batch_limit(config.max_train_batches_per_epoch)
    eval_batch_limit = normalize_batch_limit(config.max_eval_batches)

    # Load state layout for preprocessing (body frame, quat targets).
    layout: StateLayout | None = None
    state_layout_list: list[dict[str, Any]] | None = None
    if config.use_body_frame or config.use_quat_targets:
        layout = load_state_layout(config.converted_dataset_path)
        # Keep the raw list for checkpoint serialization.
        state_layout_list = [
            {"name": f.name, "start": f.start, "end": f.end}
            for f in layout.fields
        ]
        print(
            f"State layout loaded: {layout.state_dim}-dim, "
            f"{len(layout.quaternion_slices)} quaternion fields, "
            f"body_frame={config.use_body_frame}, quat_targets={config.use_quat_targets}",
            flush=True,
        )

    # Validate input keys against the dataset. Drop unavailable keys with a warning
    # so that old datasets (e.g. without contact_impulses) still work.
    with h5py.File(config.converted_dataset_path, "r") as _check_f:
        available_keys = set(_check_f["data"].keys())
    resolved_input_keys = []
    for key in config.input_keys:
        if key in available_keys:
            resolved_input_keys.append(key)
        else:
            print(
                f"WARNING: input key '{key}' not found in dataset "
                f"({config.converted_dataset_path}). Skipping.",
                flush=True,
            )
    config = replace(config, input_keys=tuple(resolved_input_keys))

    # Build the set of keys to load from HDF5.  Includes model input keys,
    # always-required keys, and any extra keys needed by preprocessing.
    required_keys = set(config.input_keys) | {"states", "next_states"}
    if config.use_contact_masking:
        required_keys |= _PREPROCESSING_EXTRA_KEYS & available_keys

    train_dataset = IndexedTrajectoryWindowDataset(
        config.converted_dataset_path,
        config.train_indices_path,
        sequence_length=config.history_length,
        keys=sorted(required_keys),
    )
    test_dataset = IndexedTrajectoryWindowDataset(
        config.converted_dataset_path,
        config.test_indices_path,
        sequence_length=config.history_length,
        keys=sorted(required_keys),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=eval_batch_limit is not None,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        generator=(torch.Generator().manual_seed(config.seed + 1) if eval_batch_limit is not None else None),
    )

    sample_batch = next(iter(train_loader))
    model = build_model_from_sample(sample_batch, config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    normalization_state: dict[str, Any]
    if config.checkpoint_path is not None:
        checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        normalization_state = checkpoint["normalization_state"]
        load_normalization_stats(model, normalization_state, device=config.device)
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        normalization_state = attach_normalization_stats(
            model, train_loader, config=config, layout=layout,
        )
        start_epoch = 1

    config_path = config.output_dir / "train_config.json"
    save_json(config_path, config.to_dict())

    history: list[dict[str, Any]] = []
    best_eval_loss = math.inf
    latest_checkpoint_path = config.output_dir / "latest_checkpoint.pt"
    best_checkpoint_path = config.output_dir / "best_checkpoint.pt"

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        train_metrics = train_or_eval_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            config=config,
            max_batches=train_batch_limit,
            layout=layout,
        )

        result = {
            "epoch": int(epoch),
            "train": train_metrics,
        }

        if epoch % config.evaluation_frequency == 0:
            eval_metrics = train_or_eval_epoch(
                model,
                test_loader,
                optimizer=None,
                config=config,
                max_batches=eval_batch_limit,
                layout=layout,
            )
            result["eval"] = eval_metrics

            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                save_training_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    config=config,
                    normalization_state=normalization_state,
                    metrics=result,
                    state_layout_list=state_layout_list,
                )

        save_training_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            normalization_state=normalization_state,
            metrics=result,
            state_layout_list=state_layout_list,
        )

        history.append(result)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_raw_mse={train_metrics['raw_loss']:.6f}",
            flush=True,
        )
        if "eval" in result:
            print(
                f"epoch={epoch:03d} "
                f"eval_loss={result['eval']['loss']:.6f} "
                f"eval_raw_mse={result['eval']['raw_loss']:.6f}",
                flush=True,
            )

    metrics_path = config.output_dir / "training_metrics.json"
    summary = {
        "config_path": str(config_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "history": history,
        "train_windows": len(train_dataset),
        "test_windows": len(test_dataset),
    }
    save_json(metrics_path, summary)
    return summary
