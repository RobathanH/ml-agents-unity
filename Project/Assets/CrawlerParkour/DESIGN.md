# CrawlerParkour — design spec

A crawler traverses a procedurally generated obstacle track as fast as possible.
Single agent, no self-play. The design goal carried over from the CrawlerSumo
post-mortem is **non-single-mode behaviour**: the track must keep asking
different physical questions so that one memorised gait cannot score well.

---

## 1. Measured agent envelope

Everything below is calibrated against the actual prefab, not guessed.

| Quantity | Value | Source |
|---|---|---|
| Body collider | sphere, r = 0.5 (1.0 m across) | `Crawler.prefab` Body |
| Body mass | 20 kg | |
| Upper leg (`legN`) | capsule r = 0.15, len 0.95, mass 2 | ×4 |
| Lower leg (`forelegN`) | capsule r = 0.15, len 1.5, mass 1 | ×4 |
| Total mass | 32 kg | |
| `maxJointSpring` / `jointDampen` / `maxJointForceLimit` | 40000 / 5000 / 20000 | `CrawlerSumoEGNNEnv.prefab` |

Derived:

- **Leg reach** ≈ 0.95 + 1.5 = **2.45 m** fully extended.
- **Nominal stance height** (body centre) ≈ **1.0 m**; can flatten to ≈ **0.55 m**
  (body resting on its own collider) and rear up to ≈ **1.8 m**.
- **Splayed stance span** ≈ **3.5 m** tip-to-tip; legs can tuck to ≈ **1.4 m**.
- **Force-to-weight** ≈ 20000 N / 314 N ≈ **64:1**.

The force-to-weight ratio is the important one: **the crawler is hugely
over-actuated.** Jumping is not torque-limited. Every difficulty ceiling below is
therefore set by *geometry and control*, not by strength — which is what we want,
because it means "harder" always means "needs a smarter movement", never "needs
more watts".

`BODY = 1.0 m` (body diameter) is the unit for all obstacle bounds, so
re-tuning the morphology rescales the track automatically.

---

## 2. One primitive, many roles

Every obstacle and every piece of floor is an **oriented box**: centre, rotation,
half-extents. Nothing else. Roles are emergent from the parameters:

| Role | How the box is parameterised |
|---|---|
| Floor tile | wide, flat, at track level |
| Raised/sunken floor | same, y offset ±0..1.5 |
| Step / ledge | flat box, top face 0.3–1.5 above local floor |
| Wall | tall, thin, spanning part of the track width |
| Ramp | box pitched 10–40° about the lateral axis |
| Off-camber | box rolled 5–25° about the forward axis |
| Pebble / boulder | small box, 0.15–0.5, random yaw+pitch |
| Hurdle / bar | long thin box, bottom face raised |
| Overhang / ceiling | wide box, bottom face 1.1–2.0 above floor |
| Beam | narrow floor tile with void either side |
| Gap | *absence* of a floor tile — not an object |
| Stepping stones | array of small floor tiles with voids between |

This is the "small vocab, many roles" requirement taken literally: there is
exactly **one** obstacle geometry in the whole environment.

### Node typing follows the primitive, not the role

Per the requirement to share EGNN node types rather than minting a one-hot per
semantic role, the type vocabulary is **geometric and kinematic only**:

- `type` ∈ {`self`, `obstacle`} — 2 slots
- `subtype` ∈ {`body`, `upper`, `lower`, `static`, `dynamic`} — 5 slots

(The crawler's lower legs are its feet; there is no separate foot segment.)

A wall, a ramp, a ledge, a pebble and the floor are all `obstacle/static`. They
are distinguished by their **geometry**, which the sensor transmits losslessly
(§4). `dynamic` exists only because "can this be pushed" is a mass property that
is genuinely *not* inferable from shape — it earns its slot. Nothing else does.

This matters beyond tidiness: if "wall" were a one-hot, the policy could learn a
wall-reflex keyed on the label and never look at the geometry. Forcing role
inference through shape is what makes the behaviour generalise to box
configurations never seen in training.

---

## 3. Difficulty bounds — "hard but possible"

Each bound is stated as the value at difficulty `d = 0` (curriculum start) and
`d = 1` (ceiling), with the reason the ceiling is where it is.

| Parameter | d=0 | d=1 | Ceiling rationale |
|---|---|---|---|
| Track width | 12 m | 5 m | 5 m ≈ 1.4× splayed span: lateral error is punished, gait still fits |
| Step-up height | 0.2 | 1.4 | 1.4 < 2.45 leg reach: reachable without a jump, but needs a real weight shift |
| Wall height (must mount) | — | 2.2 | Needs rear-up (1.8) **plus** a hop; above this it is not reliably solvable |
| Gap width | 0.4 | 2.5 | 2.5 ≈ 0.7× splayed span: a static straddle fails, needs a run-up |
| Overhang clearance | 2.4 | 1.15 | Body is 1.0 across — 1.15 is a 0.15 margin, forces a full belly-crawl |
| Lateral corridor | 6 m | 1.7 m | Forces legs tucked from 3.5 to ≈1.4: a genuinely distinct gait. Was 1.35 in the first draft, which is *narrower than the tucked stance itself* — impossible rather than hard |
| Ramp pitch | 5° | 40° | ≈ arctan(µ) for the physics material; steeper simply slides |
| Off-camber roll | 0° | 25° | Beyond this the body slides off sideways regardless of gait |
| Pebble field density | 0 | 2.5 /m² | Dense enough that a fixed-period gait trips |
| Obstacles per 8 m segment | 1 | 4 | Compound obstacles (ramp→gap→overhang) |

Two bounds deserve emphasis because they are the ones that force the
*qualitatively new* movements the environment exists to produce:

- **Overhang at 1.15 m** cannot be cleared by the nominal 1.0 m stance plus leg
  swing — the body must go down and stay down while the legs still generate
  thrust. That is not a scaled-down walk; it is a different gait.
- **Lateral corridor at 1.7 m** cannot be entered with legs splayed at 3.5 m.
  The agent must tuck, align its yaw, and push through. Also not a scaled walk.

Both bounds have a second, easily-missed requirement: the obstacle must not be
*avoidable by a different move*. An overhang thin enough to climb on top of is
not a crawl obstacle, it is a vault — so the slab is thick enough that its top
stays above the mountable height at every difficulty. The feasibility harness
checks this explicitly, because it was wrong in the first implementation.

These are exactly the cases where "navigating around would require a constrained
gait" — and because the track has fall-off edges (§3.1), going around is not an
escape hatch.

### 3.1 Fall-off edges are what make the bounds bind

The track is a raised platform with **no railing**. Leaving it laterally drops
the agent into a void. Without this, every "squeeze under" reduces to "walk
around", and the whole vocabulary collapses to one question. With it, the
generator can *force* an interaction by spanning the free width.

**The two ends are aprons, not edges.** `StartApron` and `FinishApron` (4 m each)
extend solid full-width ground behind segment 0 and past the last segment. They
are not part of `TrackLength`, so they change neither `ProgressFraction` nor the
finish threshold — they are ground to stand on, not track to cover.

They exist because the edges above are deliberate and the ends were not. The
crawler is ~3.9 m long but every placement in this project is a single point at
the body centre: the controller spawned it at `z = 1` and a checkpoint respawn
lands at `z = 0.5`, against a floor that began at `z = 0`. That put **25% of the
animal over the void at spawn and 38% after every fall**, hind legs first, so its
opening move was always a scramble not to fall off backwards — a start condition
being trained as if it were a skill. The same defect sat at the far end, where
`Finished` triggers at `TrackLength − 1` and the final metre had to be walked
with the front feet over nothing.

**Segment 0's lane is not randomised.** Lane centres drift laterally per segment
(§3.2), and the spawn and first checkpoint respawn both sit on segment 0's. A
drifted start lane threw the crawler off the *side* exactly as the missing apron
threw it off the *back*: at d = 1 the track is 5 m wide against a 3.94 m splayed
rest pose, so any lane offset over 0.53 m leaves part of the animal over the void
before it has taken an action. Segment 0 is therefore pinned to the centre line,
where the crawler fits at every difficulty; segments 1+ drift as before, so track
variety is unchanged and only the start is deterministic.

Both defects survived into runs 001–003 because the build check tested
`GroundHeightAt` at the body centre, which was a comfortable metre clear of the
edge the whole time. A point test cannot see a footprint. The check now
rasterises the crawler's actual collider bounds at spawn, first respawn and the
finish line, across d ∈ {0, 0.25, 1}, and separates ground missing *in Z* (never
intended: the world ran out) from footprint *past the lateral edge* (intended
mid-track at high difficulty, where the narrow track is the challenge). Only the
first fails the build.

### 3.2 Feasibility invariant (the part that must not be hand-waved)

"Difficult but possible" is only real if it is **checked**, not hoped for. The
generator therefore enforces, per segment, before committing any box:

Rasterise the track width into lateral bins (0.25 m). For each bin record
`floorTop` (highest walkable surface) and `ceiling` (lowest obstacle bottom face
above it). A bin is **passable** iff `ceiling − floorTop ≥ minClearance(d)` and
the longitudinal void ahead of it is `≤ maxGap(d)` and the step from the previous
segment's `floorTop` is `≤ maxStep(d)`.

**Invariant: every segment must contain a contiguous run of passable bins at
least `minCorridor(d)` wide, connected to a passable run in the previous
segment.** If a candidate layout violates it, the offending box is shrunk or
dropped and the check re-runs (bounded retries, then fall back to a flat
segment). This is a genuine guarantee that a traversable path exists — it is not
a guarantee that the agent can execute it, which is the difficulty we want.

Connectivity between consecutive segments is checked on bin overlap, so the
generator cannot produce a passable corridor on the far left followed by one on
the far right with a wall between.

---

## 4. EGNN sensing — the contact-point problem

**The problem.** A centre-of-mass node is a good summary of a *small* object and
a nearly useless summary of a *large* one. A 12 m × 0.5 m wall's centroid can be
6 m from the agent while its face is 0.3 m away. Every large obstacle in §2 —
floor tiles, walls, overhangs, ramps — is in the useless regime. The pebbles are
the only things a COM node describes well.

**Is the information even sufficient?** Yes, and this is worth stating precisely:
for a box, the triple **(centre, orientation, half-extents) is a lossless
encoding**. The distance from any body part to the box, the contact normal, the
top surface height at any (x,z) — all are closed-form functions of that triple.
So an EGNN given those attributes has, in principle, everything needed for the
physical reasoning in §3. Nothing is hidden from it.

**But sufficient ≠ learnable.** The EGNN's inductive bias is radial: kNN
selection and the message MLP both operate on `‖x_i − x_j‖`. If `x_j` is the
centroid, that distance is uncorrelated with the quantity that actually matters
(surface distance), so the architecture's strongest prior is pointed at the wrong
scalar and the network has to learn a box SDF from scratch to undo it.

**Fix: move the node to where the interaction is.** Each obstacle emits one node
whose position is the **closest point on its surface to the agent's body**, plus:

| Field | Kind | Purpose |
|---|---|---|
| `pos` (3) | equivariant position | the contact point — now `‖x_i − x_j‖` *is* surface distance |
| `quat` (4) | → 3 equivariant axes | surface/box orientation |
| `linvel`, `angvel` (3+3) | equivariant | motion of movable boxes |
| `to_center` (3) | **new** equivariant vector | contact point → box centre |
| `extent` (3) | **new** invariant scalars | half-extents |

`pos + to_center` recovers the centre, so **the lossless triple is preserved** —
we have re-parameterised the encoding without discarding anything, while making
the geometrically meaningful distance the one the architecture is built around.
The outward contact normal comes for free as the edge direction from the contact
node to the agent's body node, which the EGNN already forms.

Half-extents are lengths in the box's own frame, hence rotation-invariant, so
they slot into the existing scalar block with no new machinery. `to_center` is a
displacement, so it becomes a 7th equivariant vector channel alongside quat-axes,
linvel, angvel and the constant gravity/up channel.

Cost: vector channels 6 → 7. Node invariants grow 21 → 28, edge extras 48 → 63
(the pairwise-dot term is quadratic in channel count — 7 is comfortable, but this
is the reason not to add vector channels casually).

### 4.1 Closest point is computed analytically, not via the physics engine

`Collider.ClosestPoint` is a physics query and would run K×arenas times per
decision. Since every obstacle is a box, the closest point is a clamp in the
box's frame:

```
d      = p − centre
local  = (d·right, d·up, d·forward)
local  = clamp(local, −halfExtents, +halfExtents)
closest = centre + right·local.x + up·local.y + forward·local.z
```

No physics call, no allocation, exact for boxes. When `p` is inside the box the
clamp is a no-op and `closest = p`, which correctly reads as "penetrating" and
still recovers the centre through `to_center`.

### 4.2 Dynamic entity membership

The existing sensor freezes its entity list from a static hierarchy at
`CreateSensors`. Procedural obstacles change every episode, so the sensor gains
an **entity source** interface: a source declares its type/subtype vocabulary and
a maximum entity count up front (keeping the frozen category plan and the
allocated row width valid), then supplies rows each step. The parkour source
selects the K nearest obstacles within a look-ahead window; unused slots pad with
zeros — which is exactly the padding path the kNN masking fix made sound.

### 4.3 What the EGNN still cannot do, and the height field

Object-level reasoning is the EGNN's job. **Foot-level surface reasoning is
not** — "how high is the ground 0.4 m ahead of my front-left foot" is a query
against the *union* of boxes, and answering it through message passing means
learning a max-reduction over SDFs.

So the agent also gets a **height field**: a body-local, yaw-aligned grid of
downward raycasts returning floor clearance, plus a matching upward set for
ceiling clearance. This is standard in legged-parkour work and it is the right
tool for foot placement, gap edges and overhang profiles. The division of labour:

- **Height field** — where the surfaces are, right now, near my feet.
- **EGNN** — what the objects are, how they are oriented, how big they are, which
  way they are moving, and what is coming that the height field cannot see yet.

---

## 5. Reward

Speed is the whole objective; everything else is a cost or a correction.

- **Progress**: `w_prog · max(0, z − z_max_so_far)`. Paying only for *new* ground
  means oscillating back and forth earns nothing and a respawn costs exactly the
  time to re-cover the ground — no double-payment, no farming.
- **Dense velocity** (added for run 003): `w_vel · speed(v_z) · heading / decisions`,
  the stock Crawler's shape — 1 at `target_speed`, 0 at standstill and at twice
  target. It is unsigned and zero going backwards, so it does *not* reintroduce
  the oscillation farming the ratchet exists to prevent. `w_vel` is the
  whole-episode budget, and must stay under `finish_bonus / 2` or dawdling to the
  finish line out-earns the early-finish bonus it gives up.
- **Finish bonus** scaled by remaining time, so finishing faster is strictly better.
- **Control costs**: energy + action-rate, carried over from run 013, but scaled
  by `lerp(control_cost_floor, 1, difficulty)` so the flat lesson barely charges
  them. They make a working gait efficient; they cannot make a gait exist.
- **No alive bonus** — it would pay for stalling.

### Risk pricing (the lesson from sumo)

Falling off does **not** end the episode. The agent respawns at the last
checkpoint having lost the traversal time. Failure is cheap and recoverable, so
the optimal policy is aggressive; a terminal fall would price failure so high
that the optimum is a slow, conservative shuffle — which is the single-mode
outcome this environment exists to avoid.

### Lazy-policy test

*What is the laziest policy that scores 80% of maximum?* A cautious walk that
never leaves the ground scores well below a policy that leaps gaps and belly-
crawls overhangs, because progress reward is linear in distance and the finish
bonus is time-scaled. Obstacles that *can* be walked around are prevented by the
width-spanning rule (§3.1). This is the check every future difficulty change must
re-pass.

**Run 002 showed this test was asking the wrong question.** It compares policies
that all make progress, and the laziest policy is not a lazy *walk* — it is not
walking at all. Standing still scored better than random flailing, because the
only positive terms were conditional on net displacement while the costs were
charged every step, so the reward-maximising policy before any gait exists is to
stop moving. Reward rose by a full point over 5.5M steps while ProgressFraction
*fell*. Ask the question at both ends: what does the best policy score, and what
does the policy that does **nothing** score — and is there a smooth path of
improving reward between them? The dense velocity term is what supplies that
path.

### 5.1 Episode length is a reward parameter, not a budget knob

`max_episode_steps` looks like a scheduling detail. It is not: **it rescales
four of the five reward terms, each by a different factor.** Changing it without
re-deriving the curriculum silently changes what every lesson is asking for.

| term | on doubling the episode |
|---|---|
| progress | ceiling unchanged at `w_prog`, but reached at **half** the speed |
| dense velocity | unchanged — `ApplyVelocityReward` divides by `MaxDecisions` |
| control costs | **doubled** — charged per decision, never normalised |
| respawn penalty | roughly doubled — it is time-exposure to the same terrain |
| finish bonus | unchanged in size, but its *reachability* moves by 2× in speed |

The last row is the one that bites. Below the finishing speed the reward is
dominated by progress and rises smoothly with pace. At and above it, progress
saturates and **the finish bonus becomes the only term that still varies with
speed** — while the dense velocity term actually *falls*, because an agent that
finishes early collects a smaller share of a budget sized for a whole episode.

So a `finish_bonus` chosen when finishing was out of reach will be far too small
once it is not, and the reward goes flat exactly where the run is trying to
climb. Run 005 measured this: at 120 s with the run 003 value of 5.0, the target
gaits for gentle / moderate / hard score 12.07 / 12.20 / 11.87 — not even
monotone, so no threshold could separate those lessons. Raising `finish_bonus`
to 15.0 restored a 4.3-point spread between a bare finish and a 2.5 m/s one.

**Any change to `max_episode_steps` must re-run the derivation and re-fit every
curriculum threshold.** `scripts/reward005.py` is that derivation; it reproduces
the shipped run 003 table to within 0.02 before computing anything new, which is
what makes its new numbers trustworthy.

### 5.2 A threshold is a demand on the DISTRIBUTION, not on one gait

Run 005 stalled on lesson `moderate` for 67.9M of its 68.2M steps, and the cause
was in the threshold, not the policy. `moderate`'s 17.0 was derived by computing
what **one** episode finishing at 1.2 m/s would score (17.28) and setting the bar
just under it. That silently assumes essentially every episode finishes.

What the trainer actually averages is a mixture. Once finishing is reachable but
not reliable, the episode population is bimodal — some episodes complete, the
rest stall — and the mean reward is

    R = R_stall + (R_finish - R_stall) * f

for finish rate `f`. Run 005 measured both ends directly: regressing banded
reward on banded finish rate over 33 x 2M bands at fixed difficulty gives
`R = 3.37 + 19.58 f` with **R^2 = 0.95**. Reward at a fixed lesson is the finish
rate and very little else.

Inverted, that says what a threshold really demands. 17.0 was asking for **two
thirds to four fifths of all episodes to finish**; run 005 reached 10.6% and had
nowhere to advance to. The rung below it, 9.5, demanded only ~30% — so the ladder
jumped from "cover ground without finishing" straight to "finish almost always",
with nothing in between.

**Derive thresholds through the mixture, and state each one as the finish rate it
demands before committing to it.** `scripts/analyze_parkour_run.py` fits the two
coefficients from a finished run and prints that conversion for a ladder of
candidate thresholds; a threshold whose demanded finish rate is not reachable
from where the run starts is a wall, not a lesson.

Two corollaries worth stating separately:

- **Calibrate each lesson's baseline AT that lesson.** Run 005's `flat` threshold
  came from run 004's 0.638 m/s, measured at d=0.5 on obstacle terrain. On
  near-flat ground the same policy is far faster, so `flat` and `gentle` were both
  cleared on the `min_lesson_length` episode counter without gating anything.
- **Mean forward speed is not a progress measure once the population splits.**
  Run 005's fell from run 004's 0.653 to 0.529 m/s while progress rose from 0.334
  to 0.496 and finishes went from 1 to ~4,390: the mean is dominated by the
  stalled majority. Track the finish rate.

---

## 6. Curriculum

Difficulty `d` is a single environment parameter driving every bound in §3 by
interpolation, exposed through the existing `GetParam` plumbing and advanced by
ML-Agents lessons on mean episode progress. Secondary parameters
(`track_width`, `pebble_density`, …) can be pinned individually for ablations.

Starting at `d = 0` is close to flat ground, which gets locomotion working before
any obstacle reasoning is required.

---

## 7. Training setup

- Single behaviour `CrawlerParkour`, **no self-play** — removes the ghost-trainer
  non-stationarity that made sumo runs hard to read.
- EGNN + RSA + LSTM retained; memory matters here because an obstacle leaves the
  sensor window before it is cleared and the agent must commit to a jump.
- `gamma` 0.995 (long episodes), 32 arenas per env process.
- Cloud: one Lambda `gpu_1x_a10` at a time, same cadence and budget checkpoint as
  the sumo runs.
