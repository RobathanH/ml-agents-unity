"""
Pre-launch gate for a CrawlerParkour run. Run this before booking an instance.

  python scripts/preflight_parkour.py

Checks three things, each of which has its own way of failing SILENTLY -- the run
launches, trains for a day, and only the report shows something was wrong:

 1. The config parses and ml-agents' own settings parser accepts it. A rejected
    config otherwise surfaces on the instance, after the build has been uploaded.

 2. The config and the C# agree on every environment parameter. A yaml key the
    environment never reads is a setting that does nothing; a GetParam name with no
    yaml entry silently falls back to the inspector default baked into the prefab,
    which is NOT what the config says. Run 005 had no such mismatch and it was
    still worth checking, because the cost of one is a wasted 24 hours.

 3. The built Linux binary is actually the current environment. This is the one
    that bit CrawlerSumo: a C# change that never made it into the shipped player
    trains happily against the old environment. Checked in BOTH directions --
    new symbols present, retired ones gone.

Exits non-zero if anything fails, so it can gate a launch script.
"""
import io
import os
import re
import sys

import yaml

# register_trainer_plugins() is what populates the trainer-type registry that
# settings.py validates `trainer_type: ppo` against. learn.py calls it before
# parsing; without it RunOptions.from_dict rejects EVERY config in this repo with
# "Invalid trainer type ppo", which reads as a config error and is not one.
from mlagents.plugins.trainer_type import register_trainer_plugins
from mlagents.trainers.settings import RunOptions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "ppo", "CrawlerParkour.yaml")
CTRL = os.path.join(ROOT, "Project", "Assets", "CrawlerParkour", "Scripts",
                    "CrawlerParkourEnvController.cs")
DLL = os.path.join(ROOT, "envs", "CrawlerParkour_Multi_linux",
                   "CrawlerParkour_Data", "Managed", "Assembly-CSharp.dll")

# Deliberately absent from the training config: an eval-only override
# (--env-param fixed_seed=N puts two checkpoints on the SAME track), whose prefab
# default of -1 is the correct training value.
EVAL_ONLY = {"fixed_seed"}

# Run 006 symbols. The per-pattern stat keys are built at runtime by string
# interpolation, so the joined key never appears as a literal -- look for the parts.
PRESENT = [
    "EpisodeInterrupted",
    # The run 006 reset bug: spawn and respawn wrote every nested part's world
    # transform in a loop and tore the rig apart. MoveRig transforms only the root.
    # Absent here means the binary still has the broken reset.
    "MoveRig", "is not a descendant of",
    "TryPickSpawn", "SpawnYawDegrees", "spawn_reserve_segments", "spawn_yaw_jitter",
    "progress_per_meter", "velocity_per_second",
    "difficulty_pin", "initial_level", "promote_speed", "demote_speed",
    "DistanceRate", "DistanceCovered", "TerrainLevel", "RespawnsPer100m",
    "RateAtLevel", "CrawlerParkour/", "Cleared", "Falls", "Overhang", "Squeeze",
    "segment_count", "max_boxes",
]
ABSENT = [
    "finish_bonus", "progress_weight", "velocity_weight",
    "CrawlerParkour/Finished", "CrawlerParkour/FinishSteps",
    "CrawlerParkour/ProgressFraction",
]

DECISION_PERIOD = 5
PHYS_DT = 0.02
SEGMENT_LENGTH = 8.0

fails = []


def check(ok, message):
    print(f"  {'ok  ' if ok else 'FAIL'}  {message}")
    if not ok:
        fails.append(message)


print("1. config schema")
cfg = yaml.safe_load(io.open(CFG, encoding="utf-8").read())
register_trainer_plugins()
try:
    RunOptions.from_dict(cfg)
    check(True, "ml-agents RunOptions accepts the config")
except Exception as exc:
    check(False, f"ml-agents rejected the config: {exc}")

ep = cfg["environment_parameters"]
beh = cfg["behaviors"]["CrawlerParkour"]

print("\n2. config <-> environment agreement")
params = set(ep)
read = set(re.findall(r'GetParam\("([a-z_]+)"', io.open(CTRL, encoding="utf-8").read()))
missing = sorted(read - params - EVAL_ONLY)
unused = sorted(params - read)
check(not missing, f"every GetParam name has a yaml entry (missing: {missing})")
check(not unused, f"every yaml key is read by the environment (unused: {unused})")
check(not [k for k, v in ep.items() if isinstance(v, dict)],
      "no stale curriculum block (run 006 uses per-arena levels)")

decisions = ep["max_episode_steps"] // DECISION_PERIOD
seconds = ep["max_episode_steps"] * PHYS_DT
gamma = beh["reward_signals"]["extrinsic"]["gamma"]
check(beh["time_horizon"] >= decisions,
      f"time_horizon {beh['time_horizon']} >= {decisions} decisions/episode"
      " (one trajectory per episode, bootstrapped at the truncation)")

# Sized against target_speed, not measured speed: the reward points at
# target_speed, so a reserve that only covers today's gait becomes a silent
# ceiling the moment the run starts working.
reserve_m = ep["spawn_reserve_segments"] * SEGMENT_LENGTH
reach_m = ep["target_speed"] * seconds
check(reserve_m >= reach_m,
      f"runway {reserve_m:.0f}m >= {ep['target_speed']} m/s x {seconds:.0f}s"
      f" = {reach_m:.0f}m (episode cannot reach the end of the track)")
boundaries = ep["segment_count"] - ep["spawn_reserve_segments"]
check(boundaries >= 8,
      f"{boundaries} usable spawn boundaries of {ep['segment_count']} segments"
      " (start distribution stays spread)")

print(f"\n   episode {ep['max_episode_steps']} steps = {seconds:.0f}s"
      f" = {decisions} decisions;"
      f" discount horizon {1/(1-gamma)*DECISION_PERIOD*PHYS_DT:.0f}s")

print("\n3. built binary matches the environment")
if not os.path.exists(DLL):
    check(False, f"no built assembly at {DLL} -- run BuildMultiLinux")
else:
    blob = open(DLL, "rb").read()

    def has(needle):
        return needle.encode("utf-16-le") in blob or needle.encode("utf-8") in blob

    miss = [s for s in PRESENT if not has(s)]
    still = [s for s in ABSENT if has(s)]
    check(not miss, f"run 006 symbols present (missing: {miss})")
    check(not still, f"run 005 symbols gone (still there: {still})")
    mtime = os.path.getmtime(DLL)
    newer = [p for p in (CTRL, os.path.join(os.path.dirname(CTRL), "CrawlerParkourAgent.cs"),
                         os.path.join(os.path.dirname(CTRL), "ParkourTrackGenerator.cs"))
             if os.path.getmtime(p) > mtime]
    check(not newer,
          f"binary is newer than the sources ({[os.path.basename(p) for p in newer]} changed since)")

print()
if fails:
    print(f"PREFLIGHT FAILED -- {len(fails)} problem(s). Do not launch.")
    sys.exit(1)
print("PREFLIGHT OK -- config, environment and binary agree.")
