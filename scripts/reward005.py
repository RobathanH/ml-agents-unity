"""Episode reward as a function of steady gait speed, for run 005's geometry.

Doubling max_episode_steps 3000 -> 6000 does NOT scale the reward uniformly, so
the run 003 curriculum thresholds cannot simply be carried over:

  progress   caps at 10.0 either way, but is reached at HALF the speed
  velocity   whole-episode budget (divided by MaxDecisions) -- unchanged...
             ...except an agent that FINISHES early collects only the fraction
             of the budget it was alive for, so it falls with speed
  cost       charged PER DECISION and never normalised -> DOUBLES
  respawns   roughly double for the same terrain, being time-exposure
  finish     unreachable at 60 s below 2.13 m/s; reachable at 120 s from 1.06

This reproduces the run 003 table first as a check on the model, then re-derives
the run 005 ladder. Run by hand once; the numbers it prints are the ones that go
in the config, and the config comment cites this file.
"""

TRACK = 128.0          # SegmentCount 16 x SegmentLength 8
FINISH_AT = TRACK - 1  # ApplyProgressReward: MaxProgress >= TrackLength - 1
DT = 0.02
DECISION_PERIOD = 5

PROGRESS_W = 10.0
VELOCITY_W = 2.0
TARGET_SPEED = 2.5
RESPAWN_PENALTY = 0.5
# A vigorous gait's combined energy + action-rate cost per decision. The run 003
# table's "1.2 per 600-decision episode" in exactly these units; keeping the
# per-decision form is the whole point, since that is what does not normalise.
COST_PER_DECISION = 1.2 / 600
CONTROL_COST_FLOOR = 0.1


def velocity_shape(s):
    """ApplyVelocityReward: 1 at target, 0 at standstill and at twice target."""
    d = min(abs(s - TARGET_SPEED), TARGET_SPEED) / TARGET_SPEED
    return (1.0 - d * d) ** 2


def episode(s, difficulty, respawns, max_episode_steps, finish_bonus):
    """Reward for holding speed `s` for a whole episode. Heading assumed forward."""
    T = max_episode_steps * DT
    decisions = max_episode_steps // DECISION_PERIOD

    finishes = s * T >= FINISH_AT
    t_used = FINISH_AT / s if finishes else T
    frac = t_used / T
    distance = FINISH_AT if finishes else s * T

    progress = PROGRESS_W * distance / TRACK
    # Accrues per decision, so an early finish truncates the collection.
    velocity = VELOCITY_W * velocity_shape(s) * frac
    # 0.5 + 0.5 * timeLeft, timeLeft = 1 - StepCount/MaxDecisions.
    finish = finish_bonus * (1.0 - 0.5 * frac) if finishes else 0.0
    lerp = CONTROL_COST_FLOOR + (1.0 - CONTROL_COST_FLOOR) * difficulty
    cost = COST_PER_DECISION * decisions * frac * lerp
    penalty = RESPAWN_PENALTY * respawns

    return {
        "s": s, "d": difficulty, "P": distance / TRACK, "V": velocity_shape(s),
        "finishes": finishes, "t": t_used,
        "progress": progress, "velocity": velocity, "finish": finish,
        "cost": cost, "penalty": penalty,
        "R": progress + velocity + finish - cost - penalty,
    }


def table(title, rows, max_episode_steps, finish_bonus):
    print(f"\n=== {title}  (max_episode_steps={max_episode_steps}, "
          f"finish_bonus={finish_bonus}, T={max_episode_steps*DT:.0f}s) ===")
    print(f"  {'lesson':9s} {'s':>5s} {'d':>5s} {'P':>5s} {'V':>5s} {'fin':>4s} "
          f"{'prog':>6s} {'vel':>6s} {'bonus':>6s} {'cost':>6s} {'resp':>6s} {'R':>7s}")
    out = []
    for name, s, d, resp in rows:
        e = episode(s, d, resp, max_episode_steps, finish_bonus)
        print(f"  {name:9s} {e['s']:5.2f} {e['d']:5.2f} {e['P']:5.2f} {e['V']:5.2f} "
              f"{'yes' if e['finishes'] else ' no':>4s} "
              f"{e['progress']:6.2f} {e['velocity']:6.2f} {e['finish']:6.2f} "
              f"{e['cost']:6.2f} {e['penalty']:6.2f} {e['R']:7.2f}")
        out.append((name, e["R"]))
    return out


# --- 1. Model check: reproduce the run 003 config comment exactly -------------
# Its table reads flat 3.13 / gentle 5.27 / moderate 7.24 / hard 8.53.
r003 = table("run 003 as shipped -- MODEL CHECK", [
    ("flat", 0.64, 0.00, 0.3),
    ("gentle", 1.07, 0.25, 0.5),
    ("moderate", 1.49, 0.50, 1.0),
    ("hard", 1.81, 0.75, 1.5),
], 3000, 5.0)
expected = [3.13, 5.27, 7.24, 8.53]
worst = max(abs(a - b) for (_, a), b in zip(r003, expected))
print(f"  max deviation from the shipped table: {worst:.3f}  "
      f"{'OK' if worst < 0.02 else 'MODEL DOES NOT MATCH -- STOP'}")

# --- 2. The same gaits at 120 s, finish_bonus unchanged ----------------------
# Every lesson above flat now finishes, and once you finish, progress saturates
# and only the bonus varies -- so the ladder collapses.
table("120 s, finish_bonus 5.0 -- WHY IT CANNOT STAY", [
    ("flat", 0.64, 0.00, 0.6),
    ("gentle", 1.07, 0.25, 1.0),
    ("moderate", 1.49, 0.50, 2.0),
    ("hard", 1.81, 0.75, 3.0),
], 6000, 5.0)

# --- 3. Where run 004's policy actually sits at 120 s -------------------------
# 0.638 m/s measured over the trailing band; respawns 0.11/episode at 60 s.
print("\n=== run 004's CURRENT gait (0.64 m/s) at 120 s, per lesson ===")
for d, resp in ((0.00, 0.2), (0.25, 0.3), (0.50, 0.4), (0.75, 0.6), (1.00, 0.8)):
    e = episode(0.638, d, resp, 6000, 15.0)
    print(f"  d={d:4.2f}  R={e['R']:6.2f}   (prog {e['progress']:.2f} "
          f"vel {e['velocity']:.2f} cost {e['cost']:.2f} resp {e['penalty']:.2f})")

# --- 4. The run 005 ladder ----------------------------------------------------
# Each lesson's target gait is chosen to be a real step past what the warm-start
# already earns AT THAT LESSON, not past what it earns on flat.
prop = table("run 005 PROPOSED ladder", [
    ("flat", 0.85, 0.00, 0.2),
    ("gentle", 1.05, 0.25, 0.6),
    ("moderate", 1.20, 0.50, 1.6),
    ("hard", 1.60, 0.75, 3.0),
], 6000, 15.0)
print("\n  thresholds (rounded DOWN, lenient as in run 003):")
for name, r in prop:
    print(f"    {name:9s} target R {r:6.2f}")

# --- 5. Does speed still pay once finishing is easy? -------------------------
print("\n=== reward vs speed at d=1.00, finish_bonus 15.0 "
      "(the gradient run 005 is betting on) ===")
for s in (0.64, 0.90, 1.00, 1.06, 1.20, 1.50, 2.13, 2.50, 3.00):
    e = episode(s, 1.00, 0.8 + s, 6000, 15.0)
    print(f"  s={s:4.2f}  R={e['R']:6.2f}  {'finish @%5.1fs' % e['t'] if e['finishes'] else 'times out'}")
