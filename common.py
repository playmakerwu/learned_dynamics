from __future__ import annotations

import copy
import dataclasses
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import yaml

DEFAULT_TASK = "Isaac-Factory-PegInsert-Direct-v0"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def to_builtin(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return to_builtin(dataclasses.asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return to_builtin(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: to_builtin(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def save_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(data), f, sort_keys=False)


def safe_set(mapping: dict, path: list[str], value: Any) -> None:
    node = mapping
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def _get_task_utils():
    # Isaac Lab task utilities must be imported only after the simulation app starts.
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    return load_cfg_from_registry, parse_env_cfg


def build_env_cfg(task: str, device: str, num_envs: int, disable_fabric: bool, seed: int | None):
    _, parse_env_cfg = _get_task_utils()
    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=num_envs,
        use_fabric=not disable_fabric,
    )
    if seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = num_envs
        # Fabric cloning is unnecessary for a single env and can be brittle on some setups.
        if num_envs <= 1 and hasattr(env_cfg.scene, "clone_in_fabric"):
            env_cfg.scene.clone_in_fabric = False
    if disable_fabric and hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "clone_in_fabric"):
        env_cfg.scene.clone_in_fabric = False
    return env_cfg


def _collect_usd_paths(obj: Any, *, _seen: set[int] | None = None, _paths: list[str] | None = None) -> list[str]:
    if _seen is None:
        _seen = set()
    if _paths is None:
        _paths = []

    if obj is None:
        return _paths

    obj_id = id(obj)
    if obj_id in _seen:
        return _paths
    _seen.add(obj_id)

    if isinstance(obj, str):
        if obj.endswith(".usd") and obj not in _paths:
            _paths.append(obj)
        return _paths

    if isinstance(obj, dict):
        for value in obj.values():
            _collect_usd_paths(value, _seen=_seen, _paths=_paths)
        return _paths

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            _collect_usd_paths(value, _seen=_seen, _paths=_paths)
        return _paths

    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            _collect_usd_paths(getattr(obj, field.name), _seen=_seen, _paths=_paths)
        return _paths

    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue
            _collect_usd_paths(value, _seen=_seen, _paths=_paths)

    return _paths


def _extra_usd_paths_for_task(task: str) -> list[str]:
    if not task.startswith("Isaac-Factory-"):
        return []

    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    return [
        f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",
        f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    ]


def ensure_task_assets_available(task: str, env_cfg) -> None:
    from isaaclab.utils.assets import check_file_path

    usd_paths = _collect_usd_paths(env_cfg)
    usd_paths.extend(path for path in _extra_usd_paths_for_task(task) if path not in usd_paths)

    missing_paths: list[str] = []
    for path in usd_paths:
        try:
            status = check_file_path(path)
        except Exception:
            status = 0
        if status == 0:
            missing_paths.append(path)

    if not missing_paths:
        return

    details = "\n".join(f" - {path}" for path in missing_paths[:8])
    if len(missing_paths) > 8:
        details += f"\n - ... and {len(missing_paths) - 8} more"

    raise FileNotFoundError(
        "Required USD assets are not reachable for this task.\n"
        f"Task: {task}\n"
        "Missing assets:\n"
        f"{details}\n\n"
        "Isaac Lab 5.1 assets are hosted on AWS S3 by default. This usually means either:\n"
        "  1. the machine cannot reach the cloud asset server right now, or\n"
        "  2. the assets are not cached locally yet.\n\n"
        "Recommended fix:\n"
        "  - Launch Isaac Lab once with GUI via `./isaaclab.sh -s` and enable CACHE/Hub.\n"
        "  - Ensure the machine can access the cloud asset root.\n"
        "  - Retry the command after the assets are cached.\n"
    )


def load_rl_games_cfg(task: str) -> dict:
    load_cfg_from_registry, _ = _get_task_utils()
    cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    return copy.deepcopy(cfg)


def patch_rl_games_cfg(
    agent_cfg: dict,
    *,
    device: str,
    num_envs: int,
    seed: int,
    experiment_name: str,
    experiment_root: Path,
    run_name: str,
    clip_obs: float,
    clip_actions: float,
    max_iterations: int | None = None,
    checkpoint_interval: int | None = None,
    learning_rate: float | None = None,
    horizon_length: int | None = None,
    minibatch_size: int | None = None,
    games_num: int | None = None,
    deterministic: bool = True,
    render: bool | None = None,
    render_sleep: float | None = None,
    print_stats: bool | None = None,
) -> dict:
    safe_set(agent_cfg, ["params", "seed"], seed)
    safe_set(agent_cfg, ["params", "config", "env_name"], "rlgpu")
    safe_set(agent_cfg, ["params", "config", "device"], device)
    safe_set(agent_cfg, ["params", "config", "device_name"], device)
    safe_set(agent_cfg, ["params", "config", "num_actors"], num_envs)
    safe_set(agent_cfg, ["params", "config", "name"], experiment_name)
    safe_set(agent_cfg, ["params", "config", "train_dir"], str(experiment_root))
    safe_set(agent_cfg, ["params", "config", "full_experiment_name"], run_name)
    safe_set(agent_cfg, ["params", "config", "clip_observations"], clip_obs)
    safe_set(agent_cfg, ["params", "config", "clip_actions"], clip_actions)

    if max_iterations is not None:
        safe_set(agent_cfg, ["params", "config", "max_epochs"], max_iterations)
    if checkpoint_interval is not None:
        safe_set(agent_cfg, ["params", "config", "save_frequency"], checkpoint_interval)
        safe_set(agent_cfg, ["params", "config", "save_best_after"], checkpoint_interval)
    if learning_rate is not None:
        safe_set(agent_cfg, ["params", "config", "learning_rate"], learning_rate)
    if horizon_length is not None:
        safe_set(agent_cfg, ["params", "config", "horizon_length"], horizon_length)
    if minibatch_size is not None:
        safe_set(agent_cfg, ["params", "config", "minibatch_size"], minibatch_size)

    safe_set(agent_cfg, ["params", "config", "player", "use_vecenv"], True)
    safe_set(agent_cfg, ["params", "config", "player", "games_num"], games_num or num_envs)
    safe_set(agent_cfg, ["params", "config", "player", "deterministic"], deterministic)
    if render is not None:
        safe_set(agent_cfg, ["params", "config", "player", "render"], render)
    if render_sleep is not None:
        safe_set(agent_cfg, ["params", "config", "player", "render_sleep"], render_sleep)
    if print_stats is not None:
        safe_set(agent_cfg, ["params", "config", "player", "print_stats"], print_stats)

    return agent_cfg


def register_rl_games_env(
    task: str,
    env_cfg,
    *,
    rl_device: str,
    clip_obs: float,
    clip_actions: float,
    render_mode: str | None = None,
):
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    created_envs = []

    def env_creator(**kwargs):
        env = gym.make(task, cfg=copy.deepcopy(env_cfg), render_mode=render_mode)
        env = RlGamesVecEnvWrapper(
            env,
            rl_device=rl_device,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
        )
        created_envs.append(env)
        return env

    try:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
    except Exception:
        pass

    registration = {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **kwargs: env_creator(**kwargs),
    }

    try:
        env_configurations.register("rlgpu", registration)
    except Exception:
        if hasattr(env_configurations, "configurations"):
            env_configurations.configurations["rlgpu"] = registration

    return created_envs


def checkpoint_is_valid(path: Path) -> tuple[bool, str | None]:
    path = Path(path)
    if not path.is_file():
        return False, "file does not exist"
    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                return False, f"zip member is corrupted: {bad_member}"
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def list_checkpoints(search_root: Path, run_name: str | None = None) -> list[Path]:
    search_root = Path(search_root)
    candidates: list[Path] = []
    for path in search_root.rglob("*.pth"):
        if run_name is not None and run_name not in path.parts:
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_checkpoint(search_root: Path, run_name: str | None = None, require_valid: bool = False) -> Path:
    candidates = list_checkpoints(search_root, run_name=run_name)
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoint found under: {search_root}")
    if not require_valid:
        return candidates[0]

    invalid_reasons = []
    for candidate in candidates:
        is_valid, reason = checkpoint_is_valid(candidate)
        if is_valid:
            return candidate
        invalid_reasons.append(f"{candidate}: {reason}")

    details = "\n".join(invalid_reasons[:5])
    raise FileNotFoundError(
        "No valid .pth checkpoint found under "
        f"{search_root}. Invalid candidates:\n{details}"
    )


def configure_rl_games_checkpoint_loading(map_location: str | None) -> None:
    from rl_games.algos_torch import torch_ext

    if not map_location:
        return

    def safe_load(filename):
        return torch_ext.safe_filesystem_op(
            torch.load,
            filename,
            map_location=map_location,
            weights_only=False,
        )

    def load_checkpoint(filename):
        print(f"=> loading checkpoint '{filename}' (map_location={map_location})")
        return safe_load(filename)

    torch_ext.safe_load = safe_load
    torch_ext.load_checkpoint = load_checkpoint


def mirror_checkpoints(src_root: Path, dst_root: Path) -> list[Path]:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    checkpoints = sorted(src_root.rglob("*.pth"), key=lambda p: p.stat().st_mtime)
    copied = []
    if not checkpoints:
        return copied
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in checkpoints:
        dst = dst_root / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied
