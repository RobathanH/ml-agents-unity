#!/usr/bin/env bash
# Time-limit watchdog: runs ON the Lambda instance (in tmux window 'watchdog').
# At (limit - grace): gracefully SIGINT training so the final checkpoint exports.
# During the grace window the local machine can pull results.
# At limit: terminates THIS instance via the Lambda Cloud API (billing stops).
# Also terminates early (after grace) if training exits/crashes on its own.
#
# Usage: watchdog.sh <instance_id> <limit_minutes> [grace_minutes=15]
# Requires: ~/.lambda_api_key (chmod 600)
set -u

INSTANCE_ID="${1:?usage: watchdog.sh <instance_id> <limit_minutes> [grace_minutes]}"
LIMIT_MIN="${2:?usage: watchdog.sh <instance_id> <limit_minutes> [grace_minutes]}"
GRACE_MIN="${3:-15}"
API_KEY=$(cat "$HOME/.lambda_api_key")
API="https://cloud.lambdalabs.com/api/v1"

STOP_AT=$(( (LIMIT_MIN - GRACE_MIN) * 60 ))
[ "$STOP_AT" -lt 60 ] && STOP_AT=60
KILL_AT=$(( LIMIT_MIN * 60 ))

log() { echo "[watchdog $(date -u +%H:%M:%S)] $*"; }
log "limit=${LIMIT_MIN}min grace=${GRACE_MIN}min instance=$INSTANCE_ID"

# Anchored on the venv path, NOT on "mlagents-lear[n]" alone. tmux forks its
# SERVER from the first client, and the server inherits that client's argv --
# which is the whole launch_training.sh command line, mlagents-learn included.
# So the bare pattern matches a process that outlives training by definition,
# the "exited early" branch below can never fire, and a run that crashes in
# hour 2 bills for the full limit doing nothing. Observed on CrawlerParkour_004:
# pgrep still returned the tmux server minutes after the trainer was gone.
# The real trainer is $HOME/venv/bin/python3 $HOME/venv/bin/mlagents-learn; the
# server's argv has "venv/bin/activate" and "mlagents-learn" far apart, so
# requiring them adjacent tells the two apart. The [n] still keeps pgrep from
# matching the shell that is running pgrep.
TRAINER="venv/bin/mlagents-lear[n]"

# Phase 1: wait until stop time, or until training exits on its own
while [ "$SECONDS" -lt "$STOP_AT" ]; do
    if ! pgrep -f "$TRAINER" >/dev/null; then
        log "training process gone; entering grace window early"
        break
    fi
    sleep 60
done

# Phase 2: graceful stop -> final checkpoint export
if pgrep -f "$TRAINER" >/dev/null; then
    log "sending SIGINT to training for final checkpoint export"
    tmux send-keys -t train:learn C-c 2>/dev/null
    for i in $(seq 1 30); do
        pgrep -f "$TRAINER" >/dev/null || break
        sleep 10
    done
    log "training stopped"
fi

# Phase 3: grace window for the local machine to pull results
REMAIN=$(( KILL_AT - SECONDS ))
if [ "$REMAIN" -gt 0 ]; then
    log "grace window: terminating in $(( REMAIN / 60 )) min -- pull checkpoints now"
    sleep "$REMAIN"
fi

log "terminating instance $INSTANCE_ID"
curl -s -X POST "$API/instance-operations/terminate" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}"
