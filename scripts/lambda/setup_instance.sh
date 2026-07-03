#!/usr/bin/env bash
# Provision a fresh Lambda Cloud instance (Ubuntu 22.04, x86_64) for CrawlerSumo training.
# Usage: bash setup_instance.sh [repo_url] [branch]
set -euo pipefail

REPO_URL="${1:-https://github.com/robathanh/ml-agents-unity.git}"
BRANCH="${2:-crawler-sumo}"

# mlagents pins python_requires >=3.10.1,<=3.10.12 -- Ubuntu 22.04 ships 3.10.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYVER" != "3.10" ]; then
    echo "ERROR: python3 is $PYVER but mlagents requires 3.10.x."
    echo "Pick an Ubuntu 22.04 image, or install python3.10 (deadsnakes) and rerun."
    exit 1
fi

sudo apt-get update -y
# xvfb: Unity Linux players crash (SIGSEGV) on hosts with no display server,
# even with -nographics -- training runs under xvfb-run.
sudo apt-get install -y python3-venv python3-pip unzip tmux htop xvfb

if [ ! -d "$HOME/ml-agents" ]; then
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$HOME/ml-agents"
else
    git -C "$HOME/ml-agents" pull
fi

if [ ! -d "$HOME/venv" ]; then
    python3 -m venv "$HOME/venv"
fi
# shellcheck disable=SC1091
source "$HOME/venv/bin/activate"
pip install --upgrade pip
pip install -e "$HOME/ml-agents/ml-agents-envs" -e "$HOME/ml-agents/ml-agents"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available -- check instance type/drivers"
print(f"OK: torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
EOF

mkdir -p "$HOME/ml-agents/envs"
echo ""
echo "Setup complete."
echo "Next: upload the Linux build into ~/ml-agents/envs/CrawlerSumoEGNN_linux/"
echo "then: bash ~/ml-agents/scripts/lambda/launch_training.sh <run-id>"
