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
    # Run 008: the gate is a fraction of a per-rung reference curve. Absent here
    # means the binary still has one flat gate for all ten rungs, and the run will
    # ratchet to the ceiling again exactly as 006 and 007 did.
    "promote_fraction", "demote_fraction", "PromoteGate", "RateMinusGate",
    # The assertion that would have caught the run 006 reset bug at launch.
    "RESET TORE THE RIG",
    "DistanceRate", "DistanceCovered", "TerrainLevel", "RespawnsPer100m",
    "RateAtLevel", "CrawlerParkour/", "Cleared", "Falls", "Overhang", "Squeeze",
    "segment_count", "max_boxes",
    # ---- Run 009 ----
    # Per-foot friction. The materials are built at run time and named by
    # interpolation, so these are the constant parts. Absent means the shipped
    # player still has a crawler carrying no physics material at all -- every
    # contact at the flat 1.1 average, and four policy outputs wired to nothing.
    "CrawlerFoot", "CrawlerBody", "foot collider(s) found",
    # The air-phase price. Absent means the binary trains runs 006-008's reward
    # while the config claims otherwise.
    "airborne_per_second", "AirborneFraction", "AirborneReward",
    "FootFriction", "FootRelease",
    # Obstacle stats conditioned on terrain level: the run 008 report's first
    # recommendation, and the thing that makes the marginal series safe to read.
    "ClearedLevel",
    # The correction knob for a reference curve that cannot be measured before
    # launch, because there is no policy for this action space yet (11.4).
    "reference_rate_scale",
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

print("\n3. the rig in the prefab is the rig the code expects")
# A SYMBOL CHECK CANNOT SEE ANY OF THIS. The action count, the observation width
# and every joint limit live in serialised asset data, not in the assembly, so the
# check below -- "these C# strings are in the DLL" -- would pass with a prefab that
# still declares 20 actions and hinges that only travel one way. Run 009 changes
# the rig more than it changes the code, which makes this the half that matters,
# and BuildAll is the only thing that writes it. Forgetting to re-run the builder
# after editing its constants is a completely silent way to train the old animal.
PREFAB = os.path.join(ROOT, "Project", "Assets", "CrawlerParkour", "Prefabs",
                      "CrawlerParkourEnv.prefab")
AGENT = os.path.join(os.path.dirname(CTRL), "CrawlerParkourAgent.cs")
BUILDER = os.path.join(ROOT, "Project", "Assets", "CrawlerParkour", "Editor",
                       "CrawlerParkourBuilder.cs")

prefab = io.open(PREFAB, encoding="utf-8").read()
builder = io.open(BUILDER, encoding="utf-8").read()
agent_cs = io.open(AGENT, encoding="utf-8").read()


def cs_const(source, name):
    m = re.search(rf"const\s+\w+\s+{name}\s*=\s*(-?[\d.]+)f?;", source)
    return float(m.group(1)) if m else None


def prefab_int(key):
    m = re.search(rf"{key}:\s*(\d+)", prefab)
    return int(m.group(1)) if m else None


want_actions = int(sum(cs_const(agent_cs, n) or 0 for n in
                       ("NumJointTargetActions", "NumStrengthActions", "NumFrictionActions")))
check(prefab_int("m_NumContinuousActions") == want_actions,
      f"prefab declares {prefab_int('m_NumContinuousActions')} continuous actions,"
      f" agent constants sum to {want_actions}")

# 6 orientation + 6 velocity + 1 height + 4 contacts + 3 track + N prev actions,
# plus two rays per height-field cell. Recomputed here rather than read from the
# prefab twice, because the failure being guarded against is the two disagreeing.
grid_f = int(re.search(r"GridForward\s*=\s*(\d+)", agent_cs).group(1))
grid_l = int(re.search(r"GridLateral\s*=\s*(\d+)", agent_cs).group(1))
want_obs = 6 + 6 + 1 + 4 + 3 + want_actions + 2 * grid_f * grid_l
check(prefab_int("VectorObservationSize") == want_obs,
      f"prefab declares {prefab_int('VectorObservationSize')} observations,"
      f" CollectObservations writes {want_obs} (grid {grid_f}x{grid_l})")

# Joint limits, per body part, against the builder's own constants. Parsed by
# walking the prefab's YAML documents so a limit can be attributed to the leg it
# belongs to -- there are eight ConfigurableJoints and the hips and knees want
# different answers.
docs = re.split(r"\n(?=--- !u!)", prefab)
names = {}
for d in docs:
    fid = re.search(r"--- !u!\d+ &(\d+)", d)
    nm = re.search(r"\n  m_Name: (\S+)", d)
    if fid and nm:
        names[fid.group(1)] = nm.group(2) if nm.lastindex == 2 else nm.group(1)


def limit(doc, key):
    m = re.search(rf"{key}:\n(?:    \w+: [^\n]*\n)*?    limit: (-?[\d.e+]+)", doc)
    return float(m.group(1)) if m else None


want = {
    "leg": (cs_const(builder, "HipLowXLimit"), cs_const(builder, "HipHighXLimit"),
            cs_const(builder, "HipYLimit")),
    "foreleg": (cs_const(builder, "KneeLowXLimit"), cs_const(builder, "KneeHighXLimit"), None),
}
seen = {"leg": 0, "foreleg": 0}
bad = []
for d in docs:
    if "\nConfigurableJoint:" not in d:
        continue
    go = re.search(r"m_GameObject: \{fileID: (\d+)\}", d)
    who = names.get(go.group(1), "?") if go else "?"
    kind = "foreleg" if who.startswith("foreleg") else "leg" if who.startswith("leg") else None
    if kind is None:
        continue
    seen[kind] += 1
    lo, hi, y = want[kind]
    got = (limit(d, "m_LowAngularXLimit"), limit(d, "m_HighAngularXLimit"),
           limit(d, "m_AngularYLimit"))
    for value, target, axis in zip(got, (lo, hi, y), ("lowX", "highX", "Y")):
        if target is not None and abs((value if value is not None else 1e9) - target) > 1e-3:
            bad.append(f"{who}.{axis} is {value}, builder says {target}")

check(seen == {"leg": 4, "foreleg": 4},
      f"found {seen['leg']} hip and {seen['foreleg']} knee joints (expected 4 and 4)")
check(not bad, "every leg joint carries the builder's limits" + (f" ({bad})" if bad else ""))

# Symmetric about the authored pose is the whole point of widening them -- an
# upside-down crawler has the workspace an upright one has only if the range is.
hip_lo, hip_hi = cs_const(builder, "HipLowXLimit"), cs_const(builder, "HipHighXLimit")
knee_lo, knee_hi = cs_const(builder, "KneeLowXLimit"), cs_const(builder, "KneeHighXLimit")
check(abs(hip_lo + hip_hi) < 1e-6 and abs(knee_lo + knee_hi) < 1e-6,
      f"ranges are symmetric about zero: hip [{hip_lo}, {hip_hi}], knee [{knee_lo}, {knee_hi}]")

for field, want_value in (("maxJointSpring", 80000), ("jointDampen", 10000),
                          ("maxJointForceLimit", 40000)):
    got = prefab_int(field)
    check(got == want_value,
          f"{field} is {got} (run 008 set this by hand and a rebuild used to revert it)")

check(os.path.getmtime(PREFAB) > os.path.getmtime(BUILDER),
      "prefab is newer than CrawlerParkourBuilder.cs (BuildAll has been re-run since"
      " its constants last moved)")

print("\n4. built binary matches the environment")
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
