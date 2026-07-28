"""
Banded analysis of a CrawlerParkour run.

Usage (unityrl conda env):
  python scripts/analyze_parkour_run.py                       # newest staged run
  python scripts/analyze_parkour_run.py CrawlerParkour_006
  python scripts/analyze_parkour_run.py CrawlerParkour_006 --bands 4000000

RUN 006 CHANGED WHAT THIS READS. The environment no longer has a finish line, so
`Finished`, `FinishSteps` and `ProgressFraction` are gone, and with them the
reward-vs-finish-rate fit this script used to compute. Runs 002-005 are still
readable with `--legacy`, which restores the old tag set.

The headline is now `DistanceRate` (m/s of new ground). It is scale-free, so it is
comparable across runs 005 and 006 despite the reward re-parameterisation, and
across training (30s episodes) and eval (140s rollouts).

Four things here are easy to get wrong and were all got wrong at least once:

1. ml-agents writes scalars as rank-0 TENSORS. Reading `v.simple_value` returns
   0.0 for every tag, silently, and every number downstream is then a confident
   zero. Values live in `v.tensor.float_val[0]`.

2. Tags do NOT share a step grid. `Environment/*` and `CrawlerParkour/*` are
   written by different code paths at different times, so intersecting step sets
   across tags matches nothing and yields NaN. Every tag is banded on its own.

3. A single reporting window is noise. Everything printed below is a banded mean
   and the band is printed with it.

4. NEW IN 006, and the one most likely to produce a confidently wrong story:
   every tag is a windowed mean over a population that the per-arena curriculum
   deliberately SPREADS ACROSS DIFFICULTY LEVELS. So marginal `DistanceRate` moves
   when the level distribution moves, whether or not the policy improved -- a run
   whose arenas all get promoted will show FALLING mean speed while getting
   strictly better. Read `RateAtLevel/*` for anything causal; the marginal is a
   summary, not evidence. This is DESIGN.md 5.2's corollary made permanent.
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
LEVELS = 10

TAGS = [
    ("Environment/Cumulative Reward",     "reward",  3, 1.0),
    ("CrawlerParkour/DistanceRate",       "m/s",     3, 1.0),
    ("CrawlerParkour/DistanceCovered",    "metres",  1, 1.0),
    ("CrawlerParkour/TerrainLevel",       "level",   2, 1.0),
    ("CrawlerParkour/Difficulty",         "diff",    3, 1.0),
    ("CrawlerParkour/RespawnsPer100m",    "fall/100m", 2, 1.0),
    ("CrawlerParkour/MeanForwardSpeed",   "raw m/s", 3, 1.0),
    ("CrawlerParkour/EnergyCost",         "energy",  3, 1.0),
    ("Policy/Entropy",                    "entropy", 3, 1.0),
]

LEGACY_TAGS = [
    ("Environment/Cumulative Reward",   "reward",  3, 1.0),
    ("CrawlerParkour/Finished",         "finish%", 2, 100.0),
    ("CrawlerParkour/MeanForwardSpeed", "m/s",     3, 1.0),
    ("CrawlerParkour/ProgressFraction", "prog",    3, 1.0),
    ("CrawlerParkour/Respawns",         "resp",    3, 1.0),
    ("Environment/Episode Length",      "eplen",   0, 1.0),
    ("CrawlerParkour/EnergyCost",       "energy",  3, 1.0),
    ("Policy/Entropy",                  "entropy", 3, 1.0),
]

PATTERNS = ["Flat", "StepField", "Hurdle", "Slalom", "Gap",
            "Ramp", "Pebbles", "Overhang", "Squeeze", "Beam"]


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


def table(s, tags, end, width):
    print(f"{'band (M)':>14} " + " ".join(f"{lbl:>9}" for _, lbl, _, _ in tags))
    lo = 0
    while lo < end:
        hi = min(lo + width, end)
        cells = []
        for tag, _lbl, dp, scale in tags:
            m, _ = band(s, tag, lo, hi)
            cells.append(f"{m * scale:9.{dp}f}")
        print(f"{lo/1e6:6.0f}-{hi/1e6:<7.0f} " + " ".join(cells))
        lo += width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="")
    ap.add_argument("--bands", type=int, default=4_000_000)
    ap.add_argument("--legacy", action="store_true",
                    help="read the runs 002-005 tag set (finish rate, progress fraction)")
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
    legacy = args.legacy or "CrawlerParkour/DistanceRate" not in s
    print(f"{run}: {end:,} steps"
          f"{'   [legacy run 002-005 tag set]' if legacy else ''}\n")

    if legacy:
        table(s, LEGACY_TAGS, end, args.bands)
        print("\nThis run predates the run 006 reframe. Its reward scale, curriculum"
              "\nand finish metric are not comparable with 006 -- only DistanceRate"
              "\nis, and this run does not record it. Derive it as"
              "\n  ProgressFraction x 128m / (Episode Length x 0.1s)")
        return

    print(f"--- banded means, {args.bands/1e6:.0f}M-step bands ---")
    table(s, TAGS, end, args.bands)

    # The conditional view. Everything above is marginal over a population the
    # curriculum deliberately spreads across levels, so a level that gains arenas
    # can drag the marginal down while every arena in it got faster.
    print(f"\n--- distance rate BY TERRAIN LEVEL (m/s), {args.bands/1e6:.0f}M bands ---")
    print("    a level with no samples in a band had no arena on it that window")
    header = " ".join(f"{('L%d' % i):>7}" for i in range(LEVELS))
    print(f"{'band (M)':>14} {header}")
    lo = 0
    while lo < end:
        hi = min(lo + args.bands, end)
        cells = []
        for i in range(LEVELS):
            m, n = band(s, f"CrawlerParkour/RateAtLevel/{i}", lo, hi)
            cells.append("      ." if n == 0 else f"{m:7.3f}")
        print(f"{lo/1e6:6.0f}-{hi/1e6:<7.0f} " + " ".join(cells))
        lo += args.bands

    # Per-pattern clear rate: the measure that says whether this is parkour or
    # walking the flat stretches. Ordered by final clear rate, hardest last.
    print(f"\n--- segment clear rate by pattern (%), first vs last {args.bands/1e6:.0f}M ---")
    print(f"{'pattern':>12} {'first':>8} {'last':>8} {'delta':>8}   {'falls/entry last':>16}")
    rows = []
    for p in PATTERNS:
        f0, n0 = band(s, f"CrawlerParkour/Cleared/{p}", 0, args.bands)
        f1, n1 = band(s, f"CrawlerParkour/Cleared/{p}", end - args.bands, end)
        fl, _ = band(s, f"CrawlerParkour/Falls/{p}", end - args.bands, end)
        if n0 == 0 and n1 == 0:
            continue
        rows.append((f1, p, f0, f1, fl))
    for _, p, f0, f1, fl in sorted(rows, reverse=True):
        print(f"{p:>12} {100*f0:8.1f} {100*f1:8.1f} {100*(f1-f0):+8.1f}   {fl:16.3f}")

    lvl, _ = band(s, "CrawlerParkour/TerrainLevel", end - args.bands, end)
    rate, _ = band(s, "CrawlerParkour/DistanceRate", end - args.bands, end)
    print(f"\nlast {args.bands/1e6:.0f}M: mean level {lvl:.2f} of {LEVELS-1},"
          f" {rate:.3f} m/s marginal")


if __name__ == "__main__":
    main()
