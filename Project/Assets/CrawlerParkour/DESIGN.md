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

The cheapest check is blunter than any of that, and it would have caught run 005
before launch: **what does the threshold score on the lesson BELOW it?** Run 005
spent its first 140k steps on `flat` -- difficulty 0.0, the easiest lesson that
exists -- finishing 83.6% of episodes for a mean reward of **16.04**. The gate
guarding `moderate` was 17.0. The threshold was above what that policy scored
while finishing five episodes in six on flat ground, so no amount of training at
d=0.5 could ever have cleared it. A gate higher than the rung below it is a
ceiling, and the run is lost at launch.

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

> **Superseded for run 006 by 8.4.** The ML-Agents lesson ladder described here
> was replaced by a per-arena terrain level that adapts on measured distance rate.
> The difficulty scalar and its bounds are unchanged; only what advances it moved.

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

---

## 8. Run 006 — no goal, no finish line, no terminal state

Runs 002–005 trained *"cover a 128 m track and cross the line."* Run 006 trains
*"keep moving forward indefinitely, from any point in any track."*

This is not a reweighting. Five things move together and each is load-bearing for
the others; changing any one alone makes the environment worse, not better. The
order below is the dependency order, not the order of importance.

### 8.1 There is no terminal state, so every episode is a truncation

Falling respawns at a checkpoint. The clock truncates. Nothing else ends an
episode. So `CrawlerParkourEnvController` always calls `Agent.EpisodeInterrupted()`
and the trainer bootstraps `V(s_T)` — `ppo/trainer.py` L98 gates on
`done_reached and not interrupted`.

Run 005 called `EndEpisode()` on timeout. That told the trainer a timeout was a
true terminal, forcing a value target of **0** for roughly **89% of its 55,832
episodes**, at a step nothing in the observation identifies, from states
indistinguishable from ones where the agent keeps running. This is the same bug
CrawlerSumo fixed in `19d19a0ca`, and it is worse here than it was there: sumo's
mean episode ended at 1306 of 1500 by squeeze, so its cap was rarely reached.

This is Pardo et al.'s case (ii) — the time limit is a training convenience, not
part of the task — so the fix is to bootstrap, **not** to add remaining-time to the
observation. The corollary is that no reward term may depend on the clock. Removing
the finish bonus (§8.3) is the other half of the same fix: its `timeLeft` factor was
the only place time entered the reward.

### 8.2 Every reward term is a rate

| term | run 005 | run 006 | value |
|---|---|---|---|
| progress | `progress_weight × Δz / TrackLength` | `progress_per_meter × Δz` | `10/128 = 0.078125` |
| velocity | `velocity_weight × shape / MaxDecisions` | `velocity_per_second × shape × dt` | `2.0/120 = 0.0166667` |
| costs | per decision | per decision | unchanged |

Both old denominators were "finish the track" concepts, and they coupled the reward
scale to two numbers that are now free parameters. §5.1 documents what that cost:
doubling the episode length for run 005 moved every term by a different factor, and
that is what made its curriculum thresholds underivable.

As rates, **episode length and track length can both change without rescaling
anything** — which is what lets a 30 s training episode and a 140 s eval rollout be
scored on the same axis. The values above are run 005's numbers to full precision,
so the re-parameterisation is a numerical identity at launch. Writing `0.08` instead
of `0.078125` would have been a silent +2.4% bump to the dominant term.

**The ratchet stays primary and velocity stays shaping.** This is a deliberate
departure from legged-gym-style setups, and the reason is the one thing this
environment wants that they do not. A velocity-tracking reward charges the agent for
every step it is stopped, which prices a deliberate pause to set up a leap as a
loss. The max-so-far displacement ratchet is indifferent to *when* ground gets
covered, so it tolerates the tactical slowdown a hard obstacle needs while still
paying nothing for oscillation. Velocity tracking is primary in that literature
because it needs a controller that *follows a commanded velocity*; we want "as far
as possible".

### 8.3 Short episodes and a mid-track spawn

`max_episode_steps` **6000 → 1500** (120 s → 30 s), and the agent starts at a random
segment boundary rather than at z = 1.

Shorter is the right direction once the goal is gone, and the reason is a ratio.
Run 005 covered 63.5 m of a 128 m track every episode — but always the **same first
63.5 m**, including 16 m of guaranteed-flat `LeadInSegments`. So about a quarter of
every trajectory was flat by construction, and the back half of the track was never
seen at all. What matters is **obstacles per collected timestep, not per episode**,
and 4× the starts at a uniform sample of the track is what fixes that ratio. Two
obstacles per episode is also roughly what legged-gym collects (20 s at ~1 m/s over
8 m tiles).

30 s also sits just past the discount horizon — γ = 0.995 at 0.1 s/decision gives
1/(1−γ) = 200 decisions = 20 s — which is the regime where bootstrapping at the
truncation does real work without carrying the whole value estimate.

Spawn placement is `ParkourTrackGenerator.TryPickSpawn`. Segment **boundaries**, not
uniform z: obstacles are inset 15% of a segment from each end, so a boundary is the
one longitudinal position the generator already guarantees clear. Gap and Beam are
excluded on either side. Three lateral candidates are tried (the two lanes and their
midpoint) because the lane can drift by 0.6 × SegmentLength between the two segments
the body straddles — the midpoint alone is outside both corridors 12% of the time.
Each candidate is then re-derived from the placed geometry at ±2 m: ground present,
step within `MaxStep`, and **headroom ≥ 1.5 m**. That last bound is the one that
bites — the body centre sits 1.2 m above the surface while `MinClearance` falls to
1.15 m at d = 1, so a spawn under an Overhang would start the episode with the body
inside the slab.

Two consequences worth stating explicitly:

- **`MaxProgress` initialises to the spawn z, not 0.** Left at 0 the ratchet pays the
  whole distance from the start line to the spawn on the first decision — 0.078 ×
  40 m ≈ 3.1, several times a real episode's total. This existed in miniature
  before: spawning at z = 1 against `MaxProgress = 0` paid 0.078 every episode.
- **The spawn segment counts as neither entered nor cleared** in the per-pattern
  stats (§8.5). It is met part-way through, and crediting a clear for the metre of a
  Squeeze that happened to be left ahead of the spawn would inflate exactly the
  statistic those counters exist to measure honestly.

The track grows to **24 segments (192 m)**, forced by the runway requirement. The
reserve ahead of every spawn is sized against `target_speed`, not against measured
speed: 2.5 m/s × 30 s = 75 m, so the reserve is 10 segments = 80 m. The policy has
never exceeded ~0.7 m/s, so 48 m would have been ample in practice — and would have
become a silent ceiling exactly when the run started working. 16 segments minus a
10-segment reserve leaves only 6 usable spawn boundaries, which would re-concentrate
the start distribution near the beginning of the track and undo most of the point;
24 leaves 14, spread over the first 112 m. Total extent (192 m + two 4 m aprons) sits
inside the 220 m `ArenaSpacingZ` the 32-arena scene is laid out on.

### 8.4 Per-arena terrain levels replace the reward-threshold ladder

Each arena carries its own integer level in `[0, 10)`, mapping to the same 0..1
difficulty scalar every bound in §3 is interpolated from. At the end of each episode
that arena promotes if its measured distance rate ≥ `promote_speed`, demotes if
< `demote_speed`.

Run 005 is why. It gated the `moderate` lesson at reward 17.0 while the policy scored
16.04 on **flat**, the easiest terrain in the game — so the gate was above anything it
could ever reach and the run was lost at launch (§5.2). That failure is structural,
not a bad number: one global scalar threshold, against a reward whose scale moves
whenever episode length, track length or any weight moves.

A per-arena level has no threshold to mis-calibrate. It is measured in m/s, a
physical quantity that does not move when the reward is re-weighted; each arena finds
its own rung; and the population spreads across the ladder, so the policy keeps
seeing easy terrain instead of having it taken away. This is Rudin et al.'s
game-inspired curriculum with distance **rate** in place of raw distance, so the rule
survives a change of episode length.

Arenas randomise their starting level uniformly up to `initial_level` rather than all
starting on it. A population in lockstep is what lets one bad promotion strand
everything at once — precisely what a single global threshold did to run 005 — and a
spread means speed-vs-level is readable from the first summary window instead of only
after the ladder has been climbed.

`difficulty_pin ≥ 0` freezes the ladder. **Any rollout used to judge a checkpoint must
pass `--env-param difficulty_pin=<d>`**, or it is being scored on terrain the training
run never ran.

### 8.5 Metrics: what replaces finish rate

`Finished`, `FinishSteps` and `ProgressFraction` are gone. A fraction of a track
length only means something when there is a track to finish.

| stat | why |
|---|---|
| `DistanceRate` (m/s) | **headline.** The whole task as one number. Scale-free, so it is comparable across runs 005 and 006 despite the reward change, and across training and eval. |
| `TerrainLevel` | the curriculum's own progress curve, replacing `Lesson Number`. |
| `RespawnsPer100m` | the raw count conflates "falls a lot" with "survived long enough to reach anything worth falling off". |
| `Cleared/<Pattern>`, `Falls/<Pattern>` | per-pattern traversal, 10 of each. |

The per-pattern rates are the measure that says whether the policy is doing parkour
or walking the flat stretches between obstacles. Nothing before run 006 distinguished
crossing a `Gap` from crossing a `Flat`; a single finish rate had to stand in for all
ten. A segment counts as cleared once the agent is past its **far** boundary — standing
on top of a hurdle is not clearing it — and a fall is charged to the segment the agent
fell out of, not the checkpoint it is returned to.

Aggregation detail that matters: each stat emits `entered` samples worth
`count / entered`, so the window mean is the **pooled** rate across every episode and
arena in the window, rather than an average of per-episode ratios that would weight a
one-segment episode as heavily as a ten-segment one.

**Report speed conditioned on level, never marginal.** With per-arena levels the
population always splits by design, so §5.2's corollary about mean forward speed
applies permanently rather than only after a policy bifurcates.

### 8.6 Considered and rejected

- **Treadmill / infinite recycling track.** Behaviourally identical to a random
  mid-track spawn once episodes are 30 s — at 2.5 m/s an episode covers 75 m against
  80 m of reserve, so it cannot reach the end. Segment recycling, seam lane
  continuity, unbounded-z bookkeeping and box pooling would all be new places for a
  silent geometry bug, bought for nothing. Revisit only if episodes get long.
- **Snapshot replay / reference state initialisation.** RSI (DeepMimic) and reverse
  curricula (Florensa) exist for tasks with a hard exploration bottleneck — a state
  unreachable by chance. Parkour locomotion is not that; the terrain curriculum
  already provides graded difficulty. What it would add is a state distribution that
  is non-stationary *and* coupled to the policy, fighting the `normalize: true`
  running observation statistics, plus the question of what LSTM hidden state a
  replayed snapshot has (zeroing it is off-distribution; storing it is real
  machinery), plus restoring 9 rigidbodies and the exact track seed.
- **Velocity-command conditioning.** We want maximum distance, not a controller that
  follows a commanded velocity. See §8.2.
- **Terminate on fall.** It would make "do not fall" a value-function property rather
  than a tuned penalty, but it also prices risk so high that the optimal policy is a
  slow shuffle — the single-mode outcome this environment exists to avoid (§5,
  risk pricing). The respawn mechanism is not what was broken.
- **Physics domain randomisation.** Standard for sim-to-real; there is no real robot,
  so it buys generalisation robustness only. Not now.
- **Deleting the two goal observations.** `Clamp01(z/len)` and
  `Clamp01(MaxProgress/len)` are goal-conditioning on a goal that no longer exists,
  and with a mid-track spawn they let the policy key on absolute track position — so
  they are **held at constant zero**. Deleting them outright would take
  `NonGridObservations` from 42 to 40 and make run 005's 68M-step checkpoint
  incompatible on both the vector encoder's first layer and the running observation
  normaliser. A constant input is a bias shift PPO absorbs in a few thousand steps.
  Delete them properly at the next architecture change.

### 8.7 Two traps found while building this

Both were caught by the offline harness (`Tests~`), and both would have been
invisible in training.

**The three-column spawn check is the whole check, not belt-and-braces.** Every
placement in this project is a single point at the body centre, while the animal is
3.9 m long. Testing only the centre column accepted 12% of spawns that put a quarter
of the crawler inside a Slalom wall or under an Overhang.

**A test grid must be incommensurate with the geometry it samples.** The harness's
traversability rasteriser used `DZ = 0.4`, so slice centres were `0.4n + 0.2` — while
a `Gap` at d = 0 has edges at `8k + 3.8` and `8k + 4.2`, both exactly slice centres,
for every k. Whether each edge read as floor or void came down to float rounding that
varies with the magnitude of `iz`, so a legal 0.4 m gap rasterised as a 1.2 m one in
*some* segments, deterministically by position. Lengthening the track from 16 to 24
segments turned that from invisible into 842 false failures out of 7500. The
generator was never wrong — its own `VerifyCorridor` reported 0 repairs at d = 0
throughout. This is exactly the value of §3.2's rule that the two checks be
independent: when they disagree, one of them is wrong, and which one is a question
with an answer.

### 8.8 References

- Rudin, Hoeller, Reist, Hutter, *Learning to Walk in Minutes Using Massively
  Parallel Deep RL*, CoRL 2021 — game-inspired terrain curriculum, 8 m tiles,
  promote/demote on distance travelled.
- Pardo, Tavakoli, Levdik, Kormushev, *Time Limits in Reinforcement Learning*,
  ICML 2018 — partial-episode bootstrapping; case (i) vs case (ii).
- Zhuang et al., *Robot Parkour Learning*, CoRL 2023 — simple forward-motion reward,
  no reference motion.
- Cheng, Shi, Agarwal, Pathak, *Extreme Parkour with Legged Robots*, 2023.

### 8.9 What run 006 measured, and the check that was still missing

Run 006 delivered every mechanism in §8 and still produced no learning. The
mechanisms were not the problem; the curriculum's calibration was.

**The result.** Terrain level went from 2.71 to 8.91 of 9 within 1.1M steps — 4% of
the run — and stayed there. After that, distance rate at level 9 was flat for 23M
steps: least-squares slope +0.0015 m/s per 1M steps at R² = 0.0008, a total change
of +0.036 m/s against a window-to-window standard deviation of 0.366. Per-pattern
clear rates did not move, and `Squeeze`, `Overhang` and `Hurdle` went backwards.

**The cause.** `promote_speed` was 0.50 m/s, calibrated against run 005's final
policy, which managed 0.548 m/s at difficulty 0.5. But run 006 was warm-started, and
the warm-started policy did **1.085 m/s** in its first 200k steps. A promote gate
below the starting policy's rate is not a curriculum — it is a one-way ratchet to the
ceiling. `demote_speed` at 0.15 was never approached, so nothing came back down.

**§5.2's check has a mirror image, and it is the one that was missing.** §5.2 records
the cheap pre-launch question that would have caught run 005: *what does this
threshold score on the lesson below it?* — a guard against a gate set too **high**. It
has an opposite that guards against a gate set too **low**:

> **What does the policy you are starting from already score?** If it clears the
> promote gate at every rung, the curriculum is a no-op by construction, and the run
> is lost at launch just as surely as run 005 was. One eval rollout of the starting
> checkpoint answers it.

Both questions are the same question — *is the gate inside the reachable band?* — asked
from the two sides. Run 005 missed the top edge, run 006 the bottom.

**And a structural point beneath both.** A single global `promote_speed` assumes one
distance rate is equally demanding at every difficulty. It is not: 0.5 m/s is trivial
on flat and hard at 0.99. So no single value can hold a population in the middle of
the ladder — whatever rate is chosen is either below the policy everywhere (ratchets
up) or above it everywhere (strands). The gate has to scale per rung, set from a
*measured* rate-vs-difficulty curve of the starting policy rather than from one
number. This is the same lesson as §5.2's "a threshold is a demand on the
DISTRIBUTION", one level up: with an adaptive per-arena curriculum the threshold is a
demand on the *rung*, and it has to be expressed per rung.

**Two operational findings from the same run.**

`Beam` draws 2.72 falls per segment entered against a median of 0.55 for every other
pattern — roughly 5× worse. It has floor only under a 1.7 m lane with void either
side, against a body that splays to 3.9 m. At difficulty 0.99 the agent falls off it
repeatedly inside one segment, and it is a candidate for having consumed the whole
run's error budget. Either fix it or drop it from the mix until the rest works.

Throughput measured **686 steps/s**, not the ~300 estimated from the first few
reporting windows — env startup sits inside that first window and makes it useless as
a rate estimate. Worth remembering before re-deriving `max_steps` from early output.

**Data loss.** The scheduled artifact sync stopped 11.06 h into a 23.75 h run, so only
the first 27.31M steps of an estimated 58.6M survive, and there is no final
checkpoint — the instance terminated on schedule and took the rest with it. Checkpoints
exist only on the instance until synced (§ management README), and a sync that fails
silently is indistinguishable from one that is up to date. The report covers what
survived and says so.

> **Superseded in part by §9.** The cause given above — the promote gate sitting below
> the starting policy's rate — did happen and §9.3 keeps it. It is **not** why run 006
> failed. Run 006's episode reset was tearing the crawler's rig apart, and the 1.085 m/s
> "starting policy rate" that this section reasons from was itself a product of that bug.
> §9 replaces the causal claim; the structural argument at the end of this section is
> what survives, and §9.4 acts on it.

## 9. Run 007 — the reset fix, and what the environment actually measures

Run 007 re-ran run 006 with one thing changed. Same config, same warm start from
`CrawlerParkour_005`, same instance type. The diff between the two launch commits is
two files: `CrawlerParkourAgent.cs`, and `preflight_parkour.py`, which is a check script
and is not shipped. `CrawlerParkourEnvController.cs` — which computes every statistic
either run reports — is byte-identical across the two. That is what makes the comparison
below a controlled one rather than a pair of anecdotes.

66.75M steps in 21.5 h at 862 steps/s, stopped with SIGINT so the trainer exported its
final ONNX. $27.73.

### 9.1 The bug: a rigid move that was not rigid

The prefab hierarchy is `Body → legN → forelegN`. Every one of the other eight parts is
a **descendant** of `body`, and `transform.position`/`rotation` are world values derived
through the parent chain — so writing Body's already moves all eight with it.

Both the spawn path (`PlaceAt`) and the respawn path (`TeleportTo`) looped over every
part writing `tr.position = target + rot * (tr.position - pivot)`. That reads a value a
previous write in the same loop has already displaced, so the offset landed twice on a
leg and three times on a foreleg:

```
Body      P + delta
legN      P + 2*delta
forelegN  P + 3*delta
```

`delta` is the distance from the recorded rest pose to the placement. At a mid-track
spawn on a 24-segment track that is up to ~112 m. The ConfigurableJoints were violated
by that much on every reset and the solver answered with an enormous corrective impulse:
the crawler was launched off the track spinning, respawned, and launched again.

**Why it hid for five runs.** `TeleportTo` carried the defect from the day the
environment was written, but a respawn moves the agent a few metres, and a rig stretched
by a few metres looks like ordinary contact noise. §8.3's mid-track spawn multiplied the
same defect by roughly fifty. The bug did not appear in run 006 — run 006 is where it
became large enough to see.

It was found by a human watching the rollout videos and asking whether the odd starting
angle was a capture artefact. Nothing in 27M steps of metrics said "broken reset"; the
metrics said "bad policy", which is what a broken reset looks like from the outside.

### 9.2 Run 006's headline number was not locomotion

Over the 4M steps both runs cover, run 006 reports **0.844 m/s** of progress and run 007
**0.469** — so on the face of it the broken build was better. It was not, and the reason
matters more than the number.

`DistanceCovered` is `MaxProgress - StartZ`: a monotone high-water mark of z. Any impulse
that throws the agent **down** the track is banked as progress and never given back. Four
tests, all over the shared band:

| test | run 006 | run 007 |
|---|---|---|
| distance rate ÷ mean forward speed | **5.97×** | **1.09×** |
| corr(respawns, distance rate) | **+0.364** | −0.033 |
| metres of progress per respawn | 13.3 | 170.1 |
| clear-rate spread across the 10 patterns | 14.4 pts | 93.9 pts |

The first is decisive on its own. Distance rate and mean forward speed are measured
independently — net displacement over episode seconds, versus the average of the body's
actual forward velocity — and for honest locomotion they track each other. Run 006 banked
six metres of progress for every metre it walked.

The second is a **sign flip**, not a magnitude difference: within run 006, windows with
more respawns show *more* distance. Falling paid. A difference in policy skill cannot
produce that.

The fourth needs no statistics. Run 006 cleared every obstacle type at roughly the same
rate — 57.9% to 72.3% across ten patterns ranging from bare flat ground to a hole in the
floor. Run 007's span 0.0% to 93.9%. A policy that finds a Gap exactly as easy as flat
ground is not solving either one.

**The general lesson: a monotone high-water-mark metric cannot distinguish locomotion
from displacement.** It was chosen (§8.2) because it is scale-free and immune to
backtracking, and those properties still hold. But it needs a companion that no impulse
can fake. `MeanForwardSpeed` was already being recorded and already contained the answer;
nobody had thought to compare the two. **Any run whose rate/speed ratio is far from 1 is
reporting something other than walking.**

### 9.3 What run 007 measured

The marginal distance rate barely moves — 0.463 m/s over the first 4M against 0.563 over
the last — because the terrain gets harder at the same time. Conditioned on rung, it
moves a lot:

| quarter | rate @ L7 | rate @ L8 | rate @ L9 | entropy |
|---|---|---|---|---|
| Q1 | 0.467 | 0.378 | 0.297 | 0.974 |
| Q2 | 0.573 | 0.501 | 0.395 | 0.717 |
| Q3 | 0.642 | 0.585 | 0.484 | 0.479 |
| Q4 | 0.651 | 0.611 | 0.526 | 0.312 |

**+77% at maximum difficulty on terrain that did not change.** Least squares past 4M:
+0.0045 m/s per 1M at R² 0.47 over 6,233 windows; run 006's same fit was +0.0015 at
R² 0.0008. Entropy fell monotonically at R² 0.985 where run 006's rose. The critic's
value estimate went from −0.185 (against a *positive* mean reward, which is only
consistent with returns it could not predict) to +0.141.

It is flattening: the tail-20M fit is +0.0027 at R² 0.04, and Q4 added only 0.042 m/s.
Entropy at 0.31 means exploration is nearly spent. This configuration has given most of
what it has.

Run 007 is the first run of this environment whose numbers describe the policy rather
than the physics solver. Nothing before it should be read as a measurement of a reward
or a curriculum.

### 9.4 The gate has to be a fraction, and now is

§8.9 argued that a single global `promote_speed` cannot be right at every rung. Run 007
proved it a second way. Both runs used 0.5 m/s; both pinned to the ceiling. What changed
was only *how fast*:

| terrain level first reached | run 006 | run 007 |
|---|---|---|
| 8.0 | 1.0M steps | 16.4M steps |
| 8.5 | 1.2M steps | 33.8M steps |

**Sixteen times slower** — because the fix removed the free ballistic distance that was
pushing every arena past the gate on its first episodes. The curriculum in run 006 was
not responding to skill at all. But it still saturated in run 007, because 0.5 m/s is
below what the policy does at every rung it reached (L6 0.834, L7 0.667, L8 0.613,
L9 0.537 over the last 4M).

Run 008 replaces the absolute gate with a fraction of a per-rung reference curve:

```
promote = promote_fraction * ReferenceRate[level]
demote  = demote_fraction  * ReferenceRate[level]
```

`ReferenceRate` lives in `CrawlerParkourEnvController` and is a property of the
**terrain**, not of a checkpoint: it says what this rung costs. Levels 6–9 are run 007's
measurement; 0–5 are the least-squares extrapolation of those four points, because once
the ratchet turned no arena went back down there and no final-policy measurement exists.

**The fractions were nearly wrong in a new way, and the check caught it.** The obvious
pair is `promote_fraction: 1.0` — beat what the last policy did on this terrain — with a
wide demote band, per §8.4's warning about oscillation. Run against run 007's actual
per-episode spread, that pair gives, at level 8:

```
p(promote) 0.46    p(demote) 0.03    -> 18x upward bias
```

Which is a ratchet. Slower and better-motivated than 0.5 m/s, but the same failure.
**A gate at the median of a distribution promotes half the time and demotes almost
never.** Mean-versus-threshold reasoning cannot see this; only the spread can.

`scripts/check_curriculum_gates.py` computes p(up), p(down) and p(stay) per rung from
`RateAtLevel`, and fails when any rung below the ceiling is more than 3:1 one-way. It
selected **1.15 / 0.85**:

| rung | p(up) | p(down) | stay |
|---|---|---|---|
| L6 | 0.35 | 0.49 | 0.16 |
| L7 | 0.33 | 0.42 | 0.25 |
| L8 | 0.26 | 0.28 | 0.47 |
| L9 | 0.16 | 0.16 | 0.68 |

An arena holds its rung most episodes, the population spreads instead of collecting at
one end, and climbing requires the policy to genuinely get 15% better. §8.4's oscillation
worry was about noise-driven thrash; what it should have said is that thrash at the
*individual* arena is how a *population* holds a spread, and the thing to control is the
bias of the walk, not its step rate.

**Run this before every launch.** It is the cheap check §5.2 has now been re-learned
three times for: run 005's gate was above anything reachable, runs 006 and 007's below
anything the policy could fail, and run 008's first draft was a ratchet in disguise.

That extrapolation is the weakest part of this, and it is **self-correcting**: with
per-rung gates the population spreads across the ladder, so run 008 produces
`RateAtLevel` data at every rung and the curve can be replaced with measurement. The
per-rung gate does not just fix the curriculum — it fixes the reason the curriculum could
not be calibrated.

`promote_speed`/`demote_speed` are kept and used whenever `promote_fraction` is not
positive, so runs 006 and 007 remain reproducible from their own configs. A config that
silently means something different than it did when it ran destroys the comparison the
run existed to make.

`CrawlerParkour/PromoteGate` and `RateMinusGate` are now recorded, so "did the curriculum
have anything to say" is one chart rather than an archaeology exercise against the config.

### 9.5 Two obstacles the policy cannot do

**Gap: 0.0%.** Not poor — never, across 66.75M steps, and it was already 0.4% in the
first 4M. It does not fall in either (0.010 falls per entry): it stops at the edge and
the episode times out. The crawler has no jump and nothing in a rate-based reward pushes
a quadruped to leave the ground. Either add an explicit air-phase term or drop the
pattern; leaving it in means paying for episodes that stall.

**Beam: 6.2% cleared, 0.61 falls per entry** — eighteen times the next worst pattern
(Hurdle, 0.034), and one of only two that got worse over the run. §8.9 flagged it and it
survived the reset fix, so it is not a symptom of the bug. Before treating it as a
learning problem, check whether the beam the generator produces at high difficulty is
traversable by this rig at all: the offline harness proves a *corridor* exists, it does
not check that a body which splays to 3.9 m can balance along a 1.7 m lane with void
either side.

Everything else improved: Overhang 54.3 → 76.1%, Slalom 32.5 → 49.2%, Squeeze 18.9 →
26.3%, StepField 68.6 → 75.4%, Flat 91.1 → 92.4%.

### 9.6 The check that was missing, and now exists

Every pre-launch check run 006 had was on **geometry** — that generated tracks are
traversable, that spawn points have clear ground and headroom — and none was on the
**agent arriving there intact**. Preflight even verified that the reset code's symbols
were present in the shipped binary; it could not verify that the code was right.

`CrawlerParkourAgent.AssertRigIntact` now snapshots every part's **body-local** position
immediately before a rigid move and compares immediately after. A rigid move must not
change any part's position relative to the body; if one moves more than 2 cm, it logs
once and says in the message that no distance or fall metric from that run can be
trusted.

Body-local rather than world offsets is what makes this correct for `TeleportTo` too: a
respawn deliberately preserves whatever pose the fall left, so the offsets differ episode
to episode, and only their **invariance across the move** is the actual contract.

The general form of this, worth applying beyond the reset: *every invariant that is
enforced by construction should be asserted anyway, at the place it is constructed.* The
comment above `MoveRig` correctly explains why moving only the root is exact. Comments do
not fail a build.

### 9.7 Infrastructure

Run 007's artifact sync ran cleanly for the whole run, but it was checked by hand
mid-run — nothing would have alerted us if it had stopped, and §8.9 records what that
cost run 006. Two consecutive runs have now had their result depend on a silent
infrastructure failure not happening. The sync should fail loudly.

Throughput measured **862 steps/s** for run 007 against run 006's 686 on the same
hardware and config; the difference is the broken build's respawn storm. `max_steps` for
a 24 h run is therefore ~74.5M, and run 008 books 74M.

## 10. Run 008 — stronger joints, more grip, and what that costs the curriculum

Run 008 doubles the actuation and the grip, warm-starting from run 007's final
checkpoint:

| | run 007 | run 008 |
|---|---|---|
| `maxJointSpring` | 40000 | 80000 |
| `jointDampen` | 5000 | 10000 |
| `maxJointForceLimit` | 20000 | 40000 |
| ground friction | 0.8 | 1.6 |

Both target §9.5: `Beam` is a balance failure and `Gap` needs power the rig may not
have. More authority and more grip is the cheapest thing to try before redesigning
either obstacle.

**The friction number is not the contact number.** The crawler's own colliders carry no
physics material, so Unity pairs `ParkourGround` with the project default (0.6), and
`FrictionCombine` is Average — the contact coefficient goes 0.7 → **1.1**, not 0.8 →
1.6. Doubling the material doubles the material. Assign the material to the foot
colliders, or switch the combine mode, if the full factor is ever wanted.

Friction also lived in **two** hardcoded places — the checked-in `.physicMaterial` asset
and `CrawlerParkourBuilder`, which silently overwrites the asset on any rebuild. It is
now one named constant, `GroundFriction`.

### 10.1 A physics change invalidates the curriculum calibration

This is the trap, and it is not obvious: `ReferenceRate` (§9.4) is a measured curve, and
what it measures is the terrain **and the physics together**. Doubling the joints made
run 007's curve wrong by ~11%, which would have put every gate below what the policy
does and re-created the run 006/007 ratchet — through a change that had nothing to do
with the curriculum at all.

So the curve was re-measured rather than reasoned about. `ParkourRolloutViewer` gained a
`--measure-seconds` mode that runs the environment headless at a pinned difficulty and
reports the mean distance per completed episode; run 007's final checkpoint was run at
all ten rungs under run 008 physics, 9 episodes each, 90 total.

The result is a near-uniform **0.90× of the run 007 curve**. The warm-started policy is
*slower* under stiffer joints, because it was trained against the old gains and its
learned actions produce different torques at double the stiffness. It re-adapts during
training — which is precisely why the gate is a fraction of a curve rather than an
absolute speed, and why the first few million steps of run 008 are expected to look
worse before they look better.

The ten raw points are **fitted, not used raw**. Nine episodes per rung is thin, and the
raw L3 came back above the raw L2 — impossible, since difficulty is one scalar every
obstacle bound interpolates from. A non-monotone reference curve puts a wall at one rung
and a hole at the next and arenas pile up under the wall. The linear fit pools all 90
episodes into two parameters (`1.2419 - 0.0870*L`, R² 0.736) and is clamped monotone.

**Rule: re-run the calibration whenever the physics, the rig or the generator changes.**
A curve is not a constant.

### 10.2 checkpoint.pt can be stale, and nothing about it says so

`-InitializeFrom` seeds a run from `results/cloud_staging/<run>/<behavior>/checkpoint.pt`.
Run 007's staged copy held **499,800** steps while the numbered checkpoints beside it
reached **66,759,900** — the trainer rewrites `checkpoint.pt` in place, and a sync that
copies on size-and-mtime can skip it indefinitely.

Warm starting run 008 from it would have seeded from a policy 133× less trained than
intended, silently, and the physics change under test would have taken the blame. Name,
size and date all look correct; only the tensor inside says otherwise.

`cloud_train.ps1` now reads the seed's `global_step`, prints it, and **refuses to launch**
if it is far behind the newest numbered checkpoint sitting in the same directory. Same
shape as §9.6: the invariant was being trusted, and now it is checked.

This is a guard, not a cure. The staging sync still skips the file, and it did it again
during run 008 — the local copy sat at 499,800 for the entire 36 hours. The fix (compare
mtime as well as size, and `scp -p` so the local stamp is the remote one) exists on this
branch, but the scheduled task runs the copy in the **crawler-sumo** worktree, which
predates it. Two checkouts of one script, one of them stale, is the actual bug.

### 10.3 A moving curriculum confounds every aggregated per-obstacle statistic

`Cleared/<pattern>` and `Falls/<pattern>` are summed over all arenas at whatever rung each
has reached. While `TerrainLevel` climbs, the *same number describes harder terrain*, so
the raw first-quarter-to-last-quarter delta measures difficulty and policy at once and
cannot separate them. Run 008 read, at face value:

| | Q1 | Q4 |
|---|---|---|
| `Beam` cleared | 58.1% | **36.9%** |
| `Squeeze` cleared | 66.6% | 57.9% |
| `Hurdle` cleared | 66.1% | 57.6% |
| mean `TerrainLevel` | 4.66 | 6.92 |

which reads as a policy falling apart. Bin the summary windows by `TerrainLevel` and
compare early against late *inside the same bin* and the collapse disappears: `Beam` goes
from −21 points raw to **+1.4 and +2.5** in the two bins with enough windows on both
sides, and most patterns move up. The residual negatives are fractions of a point
(`Ramp` −0.7, `Pebbles` −1.0) rather than the double-digit falls the raw series showed.
Restricted to the 12 h extension, where the terrain moved less, **nine of ten obstacles
improved in both well-populated bins** — `StepField` at −0.5 is the only exception.

Being precise about this matters more than the headline: "nothing regressed" would be an
overclaim, and a section arguing that the raw reading is wrong is worth nothing if the
corrected reading is itself sanded smooth.

**Rule: a statistic aggregated over a population whose difficulty is itself moving must be
conditioned on that difficulty before it is read.** This is why `RateAtLevel/<L>` exists
and is trustworthy while the per-pattern series are not. The per-pattern stats should be
emitted per rung too; until they are, the binning has to be done offline, and any claim
made from the raw series is unsupported.

### 10.4 What run 008 measured

**The ladder holds a spread.** This is the §9.4 fix working, and it is the run's main
result:

| | run 007 | run 008 |
|---|---|---|
| rungs carrying data | 6 of 10 | **10 of 10** |
| largest single rung's share | 26.5% | **13.0%** |
| mean `TerrainLevel`, final 4M | 8.65 | 7.42 |
| `RateMinusGate`, final 4M | — | +0.11 |

Run 007 had no measurement at all below L4 because once its ratchet turned, no arena ever
went back down. Run 008's population sits just above its own promote gate across the whole
ladder, which is the marginal-competence regime the curriculum is supposed to hold.

**Rate at terrain level 9** — matched difficulty, and the only fair cross-run number:

| | m/s |
|---|---|
| run 007, final 4M | 0.537 |
| run 008 at ~32M | **0.509** |
| run 008 pre-extension | 0.568 |
| run 008 final | **0.603** |

The dip is §10.1 measured rather than predicted: the warm-started policy really is slower
under stiffer joints before it is faster. Final is **+12.2%** over run 007 — with the
caveat that run 008's L9 sample is *selected* (only arenas that earned their way up are
there) while run 007's was its entire pinned population, so that is the optimistic end.

The 12 h extension bought **+6.2%** on that number, and at matched difficulty: `Beam`
+5.5 to +5.9 points, `Squeeze` +3.9 to +5.1, respawns per 100 m 0.932 → 0.807. Rate over
speed finished at **1.055**, so the §9.1 reset fix is still holding and the distance being
reported is still distance that was walked.

### 10.5 `Gap` is not a training-duration problem, and now that is settled

0.7–3.1% cleared at every rung, moving +0.1 to +0.8 across **33M additional steps**, with
falls/entry of 0.02 — the agent stops at the edge rather than falling in. Doubling the
joint drive did not buy a jump either, and entropy ended at −0.04, far below where it
started and still falling, so there is no exploration left to find one with.

Three hypotheses are now eliminated: not training duration, not actuation authority, not
exploration noise. What remains is the reward. Nothing in it pays for the ballistic phase:
the progress ratchet pays on landing, and the seconds in the air are seconds not covering
ground. A deliberate leap is *dominated* by shuffling up to the edge and stopping. Either
add a term that prices the air phase, or drop the pattern — but stop spending compute on
it as-is.

### 10.6 Throughput is a function of terrain level, so `max_steps` sized on the start of a run overshoots

`max_steps` was sized on run 007's measured 862 steps/s. Run 008's first half delivered
**756** and its second half **646** — a 14.6% drop between two halves of the *same run on
the same hardware*. Higher rungs carry more obstacle boxes, so the environment gets more
expensive exactly as the curriculum succeeds.

**Rule: size `max_steps` from the throughput expected at the terrain the run will END on,
not the one it starts on.** Undershooting remains the cheap error — the run finishes early
and the watchdog stops it — while overshooting leaves the learning rate high at the end.

### 10.7 Extending a run rewinds its schedules unless you stop it

Every decayed hyperparameter in ML-Agents is anchored at step 0:

```
value = (v0 - vmin) * (1 - step/max_steps) + vmin        # ModelUtils.polynomial_decay
```

so raising `max_steps` to extend a run does not extend the anneal, it **rewinds** it. At
run 008's resume step, going 74M → 97M alone would have taken the learning rate from
5.99e-5 to 1.17e-4 — **+95%**, dropped on a policy 59M steps into convergence whose Adam
state `--resume` restores along with the weights.

The fix is to re-solve the initial value so the schedule passes through the value already
in force and still reaches its floor at the new end:

```
v0' = (v_now - vmin) / (1 - s/M_new) + vmin
```

which gave +2.7% instead of +95%. Three details that matter: read `v_now` from the run's
own TensorBoard rather than predicting it, validate the formula against those logged values
*before* inverting it, and note that the floors differ per parameter — learning rate floors
at 1e-10, `beta` at 1e-5, and using one for both is wrong at the tail.

### 10.8 The watchdog has to wait for the trainer to appear

Armed alongside a launch, the watchdog's first `pgrep` ran while the Unity environment was
still booting, found nothing, broke out of its monitoring loop and **skipped the
graceful-stop phase for the entire run** — so the instance would have been destroyed
mid-step instead of SIGINTed into a final checkpoint export.

Billing stays bounded either way, which is exactly why it survived two occurrences. The
only symptom is a `training process gone` line one second after arming, which reads like a
crash report rather than a race. `watchdog.sh` now polls for the trainer to appear first,
and a launch that genuinely never starts still falls through to the terminate path.

### 10.9 A restart does not preserve the terrain ladder

Per-arena levels live in the Unity environment, not in the checkpoint. On `--resume` every
arena restarts at `initial_level`, so run 008's population dropped from 6.92 back to ~2.5
and spent ~20 minutes re-climbing.

Cheap here — 3% of a 14 h extension — but it would dominate a short continuation, and it
means **the first summary windows after any restart describe easier terrain**. Reward jumps
(run 008's went 1.84 → 3.35) and the jump means nothing. Any before/after comparison across
a restart has to discard the re-climb, which is why §10.4's extension figures skip the
first 2M steps after the boundary.
