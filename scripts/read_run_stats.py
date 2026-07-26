"""
Compact training-stats summary from staged TensorBoard events.

Usage (unityrl conda env):
  python scripts/read_run_stats.py                      # newest staged run
  python scripts/read_run_stats.py CrawlerSumoEGNN_007
Prints first->last for key scalars plus a recent-window ELO slope.
"""
import glob
import os
import sys

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

# Resolved from this file, not hardcoded: with one worktree per project each
# has its own staging root, and an absolute path would read the wrong one.
# Override with UNITYRL_STAGING.
STAGING = os.environ.get("UNITYRL_STAGING") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "cloud_staging")

COMMON_TAGS = [
    "Environment/Cumulative Reward",
    "Environment/Episode Length",
    "Policy/Entropy",
    "Losses/Value Loss",
]

# Per-behavior tags. The behavior name doubles as the events subdirectory.
BEHAVIOR_TAGS = {
    "CrawlerSumo": [
        "Self-play/ELO",
        "CrawlerSumoLearner/PushingReward",
        "CrawlerSumoLearner/WinReward",
        "CrawlerSumoOpponent/WinReward",
        "CrawlerSumo/FlipKnockdownEnd",  # run 011+: matches decided by flip
        # Run 014+: the primary decisiveness signal. ELO no longer carries it --
        # timeouts are flagged interrupted, and ghost/trainer.py skips ELO
        # accounting for interrupted trajectories, so ELO now scores decisive
        # games only. Read DrawRate first, ELO second.
        "CrawlerSumo/DrawRate",
        "CrawlerSumo/TimeoutRate",
    ],
    "CrawlerParkour": [
        "CrawlerParkour/ProgressFraction",
        "CrawlerParkour/Finished",
        "CrawlerParkour/Respawns",
        "CrawlerParkour/Difficulty",   # which curriculum lesson is live
        "CrawlerParkour/FinishSteps",
        "CrawlerParkour/EnergyCost",
        "CrawlerParkour/ActionRateCost",
        # Should sit at ~0. Sustained non-zero means a segment pattern is
        # emitting geometry its own feasibility check rejects, so the track is
        # quietly rebuilding itself as flat ground and the run is far easier
        # than the difficulty parameter claims.
        "CrawlerParkour/TrackRepairs",
    ],
}


def find_behavior(run_dir):
    for name in BEHAVIOR_TAGS:
        if os.path.isdir(os.path.join(run_dir, name)):
            return name
    return None


def main():
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        runs = [d for d in os.listdir(STAGING)
                if find_behavior(os.path.join(STAGING, d))]
        if not runs:
            sys.exit(f"no staged runs under {STAGING}")
        run_id = max(runs, key=lambda d: os.path.getmtime(os.path.join(STAGING, d)))

    run_dir = os.path.join(STAGING, run_id)
    behavior = find_behavior(run_dir)
    if behavior is None:
        sys.exit(f"no known behavior directory under {run_dir}")
    key_tags = COMMON_TAGS + BEHAVIOR_TAGS[behavior]

    files = sorted(glob.glob(
        os.path.join(run_dir, behavior, "events.out.tfevents.*")))
    if not files:
        sys.exit(f"no events for {run_id}")

    series = {t: [] for t in key_tags}
    for f in files:
        for ev in EventFileLoader(f).Load():
            if not getattr(ev, "summary", None):
                continue
            for v in ev.summary.value:
                if v.tag in series and v.tensor.float_val:
                    series[v.tag].append((ev.step, v.tensor.float_val[0]))

    print(f"run: {run_id}  ({behavior})")
    for tag, pts in series.items():
        if not pts:
            continue
        s0, v0 = pts[0]
        s1, v1 = pts[-1]
        print(f"  {tag}: {v0:.4g} @ {s0/1e6:.1f}M -> {v1:.4g} @ {s1/1e6:.1f}M")

    elo = series.get("Self-play/ELO", [])
    if len(elo) > 10:
        recent = [p for p in elo if p[0] >= elo[-1][0] - 3_000_000] or elo[-10:]
        ds = (recent[-1][0] - recent[0][0]) / 1e6
        dv = recent[-1][1] - recent[0][1]
        if ds > 0:
            print(f"  ELO slope (last ~3M steps): {dv/ds:+.1f} per 1M")


if __name__ == "__main__":
    main()
