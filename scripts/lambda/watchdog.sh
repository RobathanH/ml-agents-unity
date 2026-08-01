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

# Phase 0: wait for the trainer to APPEAR before treating its absence as "finished".
#
# The Unity env takes ~1 minute to boot, so a watchdog armed alongside the launch
# finds nothing on its first pgrep, breaks out of Phase 1 immediately, and skips
# Phase 2 for the ENTIRE run -- the instance is then destroyed mid-step at the limit
# instead of being stopped gracefully with a final checkpoint export. The log says
# "training process gone" one second after arming, which reads like a crash report
# rather than a race, so it is easy to look straight past. Cost this project the
# graceful stop twice: at run 008's launch and again at its 12h extension.
#
# $SECONDS keeps running here on purpose -- the limit is measured from when the
# instance started billing, not from when training happened to come up.
STARTUP_WAIT="${STARTUP_WAIT:-600}"
waited=0
while [ "$waited" -lt "$STARTUP_WAIT" ]; do
    pgrep -f "$TRAINER" >/dev/null && break
    sleep 10
    waited=$(( waited + 10 ))
done
if pgrep -f "$TRAINER" >/dev/null; then
    log "trainer detected after ${waited}s; monitoring"
else
    # Deliberately NOT an error exit: a launch that never started still leaves a
    # billing instance, and terminating it at the limit is this script's whole job.
    log "trainer never appeared within ${STARTUP_WAIT}s -- treating as a failed launch"
fi

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
