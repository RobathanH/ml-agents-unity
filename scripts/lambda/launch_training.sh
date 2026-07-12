#!/usr/bin/env bash
# Launch (or resume) CrawlerSumoEGNN training in a detached tmux session,
# with TensorBoard in a second tmux window.
# Usage: bash launch_training.sh <run-id> [num-envs] [extra mlagents-learn args...]
#   e.g. bash launch_training.sh CrawlerSumoEGNN_005 12
#        bash launch_training.sh CrawlerSumoEGNN_005 12 --resume
set -euo pipefail

RUN_ID="${1:?usage: launch_training.sh <run-id> [num-envs] [extra args...]}"
NUM_ENVS="${2:-8}"
shift $(( $# >= 2 ? 2 : 1 ))
EXTRA_ARGS=("$@")

REPO="$HOME/ml-agents"
# Multi-arena build by default (12 arenas/process -> batched policy inference).
# Override with ENV_BIN=... for the single-arena build.
ENV_BIN="${ENV_BIN:-$REPO/envs/CrawlerSumoEGNN_Multi_linux/CrawlerSumoEGNN.x86_64}"

if [ ! -f "$ENV_BIN" ]; then
    echo "ERROR: $ENV_BIN not found. Upload the Linux build first."
    exit 1
fi
chmod +x "$ENV_BIN"

if tmux has-session -t train 2>/dev/null; then
    echo "ERROR: tmux session 'train' already exists. Attach with: tmux attach -t train"
    echo "or kill it first: tmux kill-session -t train"
    exit 1
fi

mkdir -p "$REPO/results"
# xvfb-run: the Unity Linux player SIGSEGVs on headless hosts without a display
# server (NULL strcasecmp in display probing), even with -nographics. A virtual
# X display fixes it; all env worker subprocesses inherit it from the trainer.
CMD="source $HOME/venv/bin/activate && cd $REPO && xvfb-run -a mlagents-learn config/ppo/CrawlerSumoEGNN.yaml \
 --env $ENV_BIN --run-id $RUN_ID --num-envs $NUM_ENVS --no-graphics --torch-device cuda \
 ${EXTRA_ARGS[*]:-} 2>&1 | tee -a $REPO/results/${RUN_ID}_console.log"

tmux new-session -d -s train -n learn "bash -lc '$CMD'"
tmux new-window -t train -n tb \
    "bash -lc 'source $HOME/venv/bin/activate && tensorboard --logdir $REPO/results --port 6006'"

echo "Started run '$RUN_ID' with $NUM_ENVS envs in tmux session 'train'."
echo "  watch logs:   tmux attach -t train      (detach: Ctrl+b then d)"
echo "  tensorboard:  ssh -L 6006:localhost:6006 ubuntu@<this-ip>  ->  http://localhost:6006"
