"""
Banded analysis of a staged CrawlerParkour run, plus the reward<->finish-rate fit
that the curriculum thresholds should be derived from.

Usage (unityrl conda env):
  python scripts/analyze_parkour_run.py                       # newest staged run
  python scripts/analyze_parkour_run.py CrawlerParkour_005
  python scripts/analyze_parkour_run.py CrawlerParkour_005 --fit-from 4000000

Three things here are easy to get wrong and were all got wrong at least once:

1. ml-agents writes scalars as rank-0 TENSORS. Reading `v.simple_value` returns
   0.0 for every tag, silently, and every number downstream is then a confident
   zero. Values live in `v.tensor.float_val[0]`.

2. Tags do NOT share a step grid. `Environment/*` and `CrawlerParkour/*` are
   written by different code paths at different times, so intersecting step sets
   across tags matches nothing and yields NaN. Every tag is banded on its own.

3. `CrawlerParkour/FinishSteps` is added once per finishing episode and
   aggregated by AVERAGE, so ml-agents writes one point per summary window. Its
   sample count is the number of windows containing at least one finish, NOT the
   number of finishes. The finish count has to come from `Finished` (a windowed
   mean of a 0/1 per episode) times the episodes closed in that window, which is
   itself derived from `Environment/Episode Length`.

And one thing about reading the output: a single reporting window is noise. At a
120 s episode a 10k-step window closes only ~8 episodes against a per-episode
reward std near 1.0. Everything printed below is a banded mean and the band is
printed with it.
"""
import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

# Resolved from this file, not hardcoded: with one worktree per project each has
# its own staging root, and an absolute path would read the wrong one.
STAGING = os.environ.get("UNITYRL_STAGING") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "cloud_staging")
BEHAVIOR = "CrawlerParkour"
WINDOW_STEPS = 10_000          # summary_freq
PHYS_DT = 0.02
LESSONS = ["flat", "gentle", "moderate", "hard", "max"]

TAGS = [
    ("Environment/Cumulative Reward",   "reward",  3, 1.0),
    ("CrawlerParkour/Finished",         "finish%", 2, 100.0),
    ("CrawlerParkour/MeanForwardSpeed", "m/s",     3, 1.0),
    ("CrawlerParkour/ProgressFraction", "prog",    3, 1.0),
    ("CrawlerParkour/Respawns",         "resp",    3, 1.0),
    ("Environment/Episode Length",      "eplen",   0, 1.0),
    ("CrawlerParkour/EnergyCost",       "energy",  3, 1.0),
    ("Policy/Entropy",                  "entropy", 3, 1.0),
]


def load(run):
    d = os.path.join(STAGING, run, BEHAVIOR)
    files = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
    if not files:
        sys.exit(f"no events files under {d}")
    series = {}
    for path in files:
        for ev in EventFileLoader(path).Load():
            for v in ev.summary.value:
                if v.tensor.float_val:
                    series.setdefault(v.tag, []).append(
                        (ev.step, float(v.tensor.float_val[0])))
    for t in series:
        series[t].sort()
    return series


def band(series, tag, lo, hi):
    """Mean of one tag over [lo, hi], on that tag's OWN step grid."""
    pts = series.get(tag)
    if not pts:
        return float("nan"), 0
    vals = [v for s, v in pts if lo <= s <= hi]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def episode_counts(series):
    """(episodes, finishes) summed window by window -- see note 3 in the docstring."""
    el = dict(series.get("Environment/Episode Length", []))
    fin = dict(series.get("CrawlerParkour/Finished", []))
    ep = fi = 0.0
    for step, length in el.items():
        if length > 0:
            n = WINDOW_STEPS / length
            ep += n
            fi += n * fin.get(step, 0.0)
    return ep, fi


def fit_reward_on_finish(series, start, end, width=2_000_000):
    """Least squares of banded reward on banded finish rate.

    Bands before `start` are excluded on purpose: while the curriculum is still
    moving the difficulty is changing under the agent, so those windows are a
    different environment and would corrupt a fit that is about one lesson.
    """
    pts = []
    lo = start
    while lo < end:
        hi = min(lo + width, end)
        f, nf = band(series, "CrawlerParkour/Finished", lo, hi)
        r, nr = band(series, "Environment/Cumulative Reward", lo, hi)
        if nf and nr:
            pts.append((f, r))
        lo += width
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    a = my - b * mx
    sst = sum((p[1] - my) ** 2 for p in pts)
    ssr = sum((p[1] - (a + b * p[0])) ** 2 for p in pts)
    return a, b, (1 - ssr / sst if sst else float("nan")), n, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="")
    ap.add_argument("--bands", type=int, default=4_000_000)
    ap.add_argument("--fit-from", type=int, default=None,
                    help="first step of the fit (default: where the last lesson began)")
    args = ap.parse_args()

    run = args.run
    if not run:
        cands = [d for d in glob.glob(os.path.join(STAGING, "*"))
                 if os.path.isdir(os.path.join(d, BEHAVIOR))]
        if not cands:
            sys.exit(f"no staged {BEHAVIOR} runs under {STAGING}")
        run = os.path.basename(max(cands, key=os.path.getmtime))

    s = load(run)
    end = s["Environment/Cumulative Reward"][-1][0]
    print(f"{run}: {end:,} steps\n")

    print("--- curriculum ---")
    cur, changes = None, []
    for step, v in s.get("Environment/Lesson Number/difficulty", []):
        if v != cur:
            changes.append((step, int(v)))
            cur = v
    for step, num in changes:
        name = LESSONS[num] if num < len(LESSONS) else str(num)
        print(f"  {step:>12,}  lesson {num} = {name}")
    last_start = changes[-1][0] if changes else 0

    ep, fi = episode_counts(s)
    nfs = len(s.get("CrawlerParkour/FinishSteps", []))
    nfin = len(s.get("CrawlerParkour/Finished", []))
    print(f"\n--- episodes ---")
    print(f"  episodes  ~{ep:,.0f}")
    print(f"  finishes  ~{fi:,.0f}  ({100*fi/ep:.2f}%)" if ep else "  finishes  n/a")
    print(f"  windows containing >=1 finish: {nfs:,} of {nfin:,}"
          f"   <- NOT the finish count")

    print(f"\n--- banded means, {args.bands/1e6:.0f}M-step bands ---")
    print(f"{'band (M)':>14} " + " ".join(f"{lbl:>8}" for _, lbl, _, _ in TAGS))
    lo = 0
    while lo < end:
        hi = min(lo + args.bands, end)
        cells = []
        for tag, _lbl, dp, scale in TAGS:
            m, _ = band(s, tag, lo, hi)
            cells.append(f"{m*scale:8.{dp}f}")
        print(f"{lo/1e6:6.0f}-{hi/1e6:<7.0f} " + " ".join(cells))
        lo += args.bands

    fs = s.get("CrawlerParkour/FinishSteps")
    if fs:
        print("\n--- finish time (mean over windows that contained a finish) ---")
        lo = 0
        while lo < end:
            hi = min(lo + 2 * args.bands, end)
            m, n = band(s, "CrawlerParkour/FinishSteps", lo, hi)
            if n:
                print(f"  {lo/1e6:5.0f}-{hi/1e6:<5.0f}M  {m:7.0f} env steps"
                      f" = {m*PHYS_DT:6.1f} s   (n={n} windows)")
            lo += 2 * args.bands

    start = args.fit_from if args.fit_from is not None else last_start
    got = fit_reward_on_finish(s, start, end)
    if got:
        a, b, r2, n, _pts = got
        print(f"\n--- reward vs finish rate, {n} bands from {start/1e6:.0f}M ---")
        print(f"  R = {a:.3f} + {b:.3f} * finish_rate      R^2 = {r2:.4f}")
        print(f"  stalling episode  ~{a:.2f}")
        print(f"  finishing episode ~{a+b:.2f}  (EXTRAPOLATED past the data --"
              f" cross-check against the reward model at the observed finish time)")
        print("\n  a reward threshold is really a finish-rate demand:")
        for thr in (8.0, 9.5, 12.0, 17.0, 18.0):
            need = (thr - a) / b
            flag = "  <-- above 100%, unreachable" if need > 1 else ""
            print(f"    R >= {thr:5.1f}  ->  finish rate {100*need:6.1f}%{flag}")
        f_now, _ = band(s, "CrawlerParkour/Finished", end - 4_000_000, end)
        print(f"\n  this run reached {100*f_now:.1f}% over its last 4M steps.")


if __name__ == "__main__":
    main()
