"""What do the run 008 promote/demote gates actually do to the population?

This is DESIGN 5.2's check -- "a threshold is a demand on the DISTRIBUTION, not on
one gait" -- applied to a per-rung gate, and it is the check whose absence lost run
005 (gate above anything the policy could reach) and runs 006 and 007 (gate below
what the policy already did, so a one-way ratchet to the ceiling).

A promote/demote pair defines a random walk on the terrain ladder. The population is
stationary only when, at the rung the policy is competent at,

    P(rate >= promote)  ~=  P(rate < demote)

If p(up) > p(down) at every rung the ladder ratchets to the ceiling no matter how
carefully the number was chosen -- which is exactly what both previous runs did. A
mean-vs-threshold comparison cannot see this; it needs the whole per-episode spread,
which is what CrawlerParkour/RateAtLevel/<L> records.

Run with no arguments to check the fractions currently in the yaml against run 007's
measured distributions. Pass a promote and demote fraction to try others.
"""
import glob
import os
import re
import sys

import yaml
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = os.path.join(ROOT, "config", "ppo", "CrawlerParkour.yaml")
CTRL = os.path.join(ROOT, "Project", "Assets", "CrawlerParkour", "Scripts",
                    "CrawlerParkourEnvController.cs")
EVENTS = os.path.join(ROOT, "results", "cloud_staging", "CrawlerParkour_007",
                      "CrawlerParkour")
TAIL = 4_000_000       # the band the reference curve was measured over
LEVELS = 10


def reference_curve():
    """Read ReferenceRate straight out of the C#, so this cannot drift from it."""
    src = open(CTRL, encoding="utf-8").read()
    m = re.search(r"ReferenceRate\s*=\s*\{(.*?)\}", src, re.S)
    if not m:
        sys.exit("could not find ReferenceRate in the environment controller")
    vals = [float(x) for x in re.findall(r"([0-9.]+)f", m.group(1))]
    if len(vals) != LEVELS:
        sys.exit(f"ReferenceRate has {len(vals)} entries, expected {LEVELS}")
    return vals


def rate_samples():
    """Per-window distance rate at each level, over run 007's last 4M steps."""
    files = sorted(glob.glob(os.path.join(EVENTS, "events.out.tfevents.*")))
    if not files:
        sys.exit(f"no run 007 events file under {EVENTS}")
    # Never merge events files; take the longest-running one.
    best, best_span, series = None, -1, {}
    for f in files:
        s, w0, w1 = {}, None, None
        for ev in EventFileLoader(f).Load():
            if ev.wall_time:
                w0 = w0 or ev.wall_time
                w1 = ev.wall_time
            for v in ev.summary.value:
                if v.tensor.float_val and v.tag.startswith("CrawlerParkour/RateAtLevel/"):
                    s.setdefault(v.tag, []).append((ev.step, float(v.tensor.float_val[0])))
        span = (w1 - w0) if w0 and w1 else 0
        if span > best_span:
            best, best_span, series = f, span, s
    end = max(st for pts in series.values() for st, _ in pts)
    out = {}
    for L in range(LEVELS):
        pts = series.get(f"CrawlerParkour/RateAtLevel/{L}", [])
        out[L] = sorted(v for st, v in pts if st >= end - TAIL)
    return out, end


cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
ep = cfg["environment_parameters"]
pf = float(sys.argv[1]) if len(sys.argv) > 1 else float(ep["promote_fraction"])
df = float(sys.argv[2]) if len(sys.argv) > 2 else float(ep["demote_fraction"])

ref = reference_curve()
samples, end = rate_samples()
print(f"run 007 final {TAIL/1e6:.0f}M steps (to {end:,}), "
      f"promote_fraction {pf}, demote_fraction {df}\n")
print(f"{'level':6}{'ref':>8}{'promote':>9}{'demote':>8}{'n':>7}"
      f"{'p(up)':>8}{'p(down)':>9}{'stay':>7}   verdict")
print("-" * 78)

unmeasured, bad = [], []
for L in range(LEVELS):
    v = samples[L]
    promote, demote = pf * ref[L], df * ref[L]
    if len(v) < 30:
        unmeasured.append(L)
        print(f"L{L:<5}{ref[L]:8.3f}{promote:9.3f}{demote:8.3f}{len(v):7d}"
              f"{'--':>8}{'--':>9}{'--':>7}   no final-policy data")
        continue
    up = sum(1 for x in v if x >= promote) / len(v)
    down = sum(1 for x in v if x < demote) / len(v)
    stay = 1 - up - down
    # A rung is one-way when promotion is several times likelier than demotion.
    # L9 is the ceiling, so an upward bias there is harmless.
    ratio = up / down if down > 0 else float("inf")
    if L == LEVELS - 1:
        verdict = "ceiling (bias is harmless)"
    elif ratio > 3:
        verdict = f"RATCHETS UP ({ratio:.1f}x)"
        bad.append(L)
    elif ratio < 1 / 3:
        verdict = f"STRANDS DOWN ({1/ratio:.1f}x)"
        bad.append(L)
    else:
        verdict = f"balanced ({ratio:.2f}x)"
    print(f"L{L:<5}{ref[L]:8.3f}{promote:9.3f}{demote:8.3f}{len(v):7d}"
          f"{up:8.2f}{down:9.2f}{stay:7.2f}   {verdict}")

print()
if unmeasured:
    print(f"note  levels {unmeasured} carry no final-policy measurement -- once the "
          f"ratchet\n      turned in run 007 no arena went back down there. Their "
          f"reference values are\n      extrapolated, and run 008 is what measures "
          f"them. This is why the fractions\n      matter more than the curve.")
if bad:
    print(f"\nFAIL  rungs {bad} are one-way. The population will not hold a spread "
          f"there.")
    sys.exit(1)
print("\nGATES OK -- every measured rung below the ceiling is two-way.")
