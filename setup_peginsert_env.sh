#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-peginsert_lab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-2.3.2.post1}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0}"
RAY_VERSION="${RAY_VERSION:-2.45.0}"
WANDB_VERSION="${WANDB_VERSION:-0.25.1}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-6.31.1}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found in PATH. Install Miniconda or Anaconda first."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Creating conda env: ${ENV_NAME}"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "Activating env: ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing Isaac Lab + Isaac Sim"
pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" --extra-index-url https://pypi.nvidia.com

echo "Installing PyTorch CUDA wheels"
pip install -U \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu128

echo "Restoring Isaac Sim compatible shared pins"
pip install \
    "packaging==23.0" \
    "wheel<0.46" \
    "protobuf==${PROTOBUF_VERSION}"

echo "Installing RL-Games stack without re-resolving Isaac Sim core packages"
pip install --no-deps \
    "ray==${RAY_VERSION}" \
    "wandb==${WANDB_VERSION}"

pip install --no-deps git+https://github.com/isaac-sim/rl_games.git@python3.11

echo "Installing leaf dependencies required by ray / rl_games / wandb"
pip install --no-deps \
    platformdirs==4.9.4 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    referencing==0.37.0 \
    rpds-py==0.27.1 \
    msgpack==1.1.2 \
    gym==0.23.1 \
    gym_notices==0.1.0 \
    setproctitle==1.3.7 \
    tensorboardX==2.6.4

echo "Running pip consistency check"
pip check

echo
echo "Environment install completed successfully."
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  isaacsim"
echo "  python verify_peg_insert.py --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 32 --num_steps 200 --headless --device cuda:0"
