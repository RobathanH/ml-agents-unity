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

STAGING = r"D:\UnityRL\ml-agents\results\cloud_staging"
KEY_TAGS = [
    "Self-play/ELO",
    "Policy/Entropy",
    "Environment/Episode Length",
    "CrawlerSumoLearner/PushingReward",
    "CrawlerSumoLearner/WinReward",
    "CrawlerSumoOpponent/WinReward",
    "CrawlerSumo/FlipKnockdownEnd",  # run 011+: proportion of matches decided by flip
    # Run 014+: the primary decisiveness signal. ELO no longer carries it —
    # timeouts are flagged interrupted, and ghost/trainer.py skips ELO
    # accounting for interrupted trajectories, so ELO now scores decisive games
    # only. Read DrawRate first, ELO second.
    "CrawlerSumo/DrawRate",
    "CrawlerSumo/TimeoutRate",
    "Losses/Value Loss",
]


def main():
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        runs = [d for d in os.listdir(STAGING)
                if os.path.isdir(os.path.join(STAGING, d, "CrawlerSumo"))]
        run_id = max(runs, key=lambda d: os.path.getmtime(os.path.join(STAGING, d)))

    files = sorted(glob.glob(
        os.path.join(STAGING, run_id, "CrawlerSumo", "events.out.tfevents.*")))
    if not files:
        sys.exit(f"no events for {run_id}")

    series = {t: [] for t in KEY_TAGS}
    for f in files:
        for ev in EventFileLoader(f).Load():
            if not getattr(ev, "summary", None):
                continue
            for v in ev.summary.value:
                if v.tag in series and v.tensor.float_val:
                    series[v.tag].append((ev.step, v.tensor.float_val[0]))

    print(f"run: {run_id}")
    for tag, pts in series.items():
        if not pts:
            continue
        s0, v0 = pts[0]
        s1, v1 = pts[-1]
        print(f"  {tag}: {v0:.4g} @ {s0/1e6:.1f}M -> {v1:.4g} @ {s1/1e6:.1f}M")

    elo = series["Self-play/ELO"]
    if len(elo) > 10:
        recent = [p for p in elo if p[0] >= elo[-1][0] - 3_000_000] or elo[-10:]
        ds = (recent[-1][0] - recent[0][0]) / 1e6
        dv = recent[-1][1] - recent[0][1]
        if ds > 0:
            print(f"  ELO slope (last ~3M steps): {dv/ds:+.1f} per 1M")


if __name__ == "__main__":
    main()
