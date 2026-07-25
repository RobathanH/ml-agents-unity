# CrawlerParkour — Unity scene setup

The scene, prefabs, materials and layer are **generated from code**, not wired by
hand:

```
"C:\Program Files\Unity\Hub\Editor\6000.0.40f1\Editor\Unity.exe" `
  -batchmode -nographics -projectPath D:\UnityRL\ml-agents\Project `
  -executeMethod CrawlerParkourBuilder.BuildAll -logFile D:\UnityRL\parkour_build.log
```

or `Training ▸ CrawlerParkour ▸ Build Assets + Scenes` from the editor menu.
[Editor/CrawlerParkourBuilder.cs](Editor/CrawlerParkourBuilder.cs) is the source
of truth; this file explains what it produces and why each choice matters.

Rebuild whenever the agent's observation layout, the sensor toggles or the
obstacle budget change. The builder is idempotent — it overwrites its own assets
and reuses the layer it already reserved.

## What it produces

| Asset | Notes |
|---|---|
| Layer `ParkourTrack` (9) | the agent's `GroundMask`, so height-field rays cannot hit its own legs |
| `Materials/ParkourGround.physicMaterial` | friction 0.8 |
| `Materials/ParkourFloor.mat`, `ParkourObstacle.mat` | visual only |
| `Prefabs/ParkourBox.prefab` | unit cube, tagged `ground`, on the track layer |
| `Prefabs/CrawlerParkourEnv.prefab` | one arena |
| `Scenes/CrawlerParkour.unity` | 1 arena, for viewing and eval |
| `Scenes/CrawlerParkour_Multi.unity` | 32 arenas, for training |

```
CrawlerParkourEnv                (CrawlerParkourEnvController)
├── Track                        (ParkourTrackGenerator, ParkourObstacleSource)
│   └── Boxes                    (BoxParent; obstacles spawn here at runtime)
└── Crawler                      (CrawlerParkourAgent, BehaviorParameters,
    │                             DecisionRequester, JointDriveController,
    │                             EGNNSensorComponent, ModelOverrider)
    └── Body                     ← EGNN root group
        ├── leg0 … leg3          (subtype `upper`)
        │   └── foreleg0 … foreleg3  (subtype `lower`)
        └── Sweatband            (excluded)
```

The ML-Agents components sit on the **Crawler** root, not on `Body`: sensors are
collected from the agent's own GameObject, and `JointDriveController` wires each
`GroundContact.agent` from *its* GameObject's `Agent`, so the two must coincide.

## Things that are quietly load-bearing

**The box prefab must be exactly 1×1×1.** `ParkourBox.Place` writes
`localScale = halfExtents * 2`. If the prefab is any other size every obstacle in
the environment is wrong by that factor — and the feasibility harness, which
works from half-extents rather than transforms, still reports the track
traversable.

**Friction 0.8 is part of the feasibility guarantee.** The steepest generated
ramp is 40°, and the ceiling for a walkable ramp is `arctan(µ)`. A slippier
material makes the hardest ramps unclimbable and the generator cannot detect it.

**The box prefab is tagged `ground`.** `GroundContact.touchingGround` only flips
for colliders carrying that tag. Without it the four foot-contact observations
are permanently zero — four dead inputs and no contact sense at all.

**The example Crawler ends its episode on ground contact.** `Body` and the four
upper legs ship with `agentDoneOnGroundContact` and `penalizeGroundContact` set.
Parkour needs the opposite: bellying under an overhang and scrambling over a step
are the behaviours the environment exists to produce. The builder clears all five,
and logs how many it cleared. Left set, they call `EndEpisode()` behind the
controller's back, so its step counter and stats describe episodes that no longer
exist.

**The track root must be an unrotated, unscaled translation.** Obstacles are
placed with `localPosition` in the track's own frame so that 32 arenas are
translations of one layout. `ParkourTrackGenerator.Generate` checks this and logs
an error if the track or its `BoxParent` is rotated, scaled or displaced.

## EGNN sensor

One root group on `Body`, type `self`, root subtype `body`. Children are typed
`upper` / `lower` — **not** `leg0`…`leg3`. A blank `SubType` falls back to the
GameObject's name, which would mint eight one-hot slots, widen every row, and let
the policy identify a specific leg by its label rather than by where it is;
nothing learned about one leg would transfer to the others. The sweatband has a
collider, so the sensor would otherwise discover it — it is explicitly excluded.

`ParkourObstacleSource` is registered in **Entity Sources**. It declares
`obstacle` / {`static`, `dynamic`} and a budget of 16.

Vocabulary: types {`self`, `obstacle`} = 2, subtypes {`body`, `upper`, `lower`,
`static`, `dynamic`} = 5.

### Toggles, and the config keys they must match

| Toggle | Value | Config key |
|---|---|---|
| Include Rotation | ✅ | `has_quaternion: true` |
| Include Linear Velocity | ✅ | `has_linear_velocity: true` |
| Include Angular Velocity | ✅ | `has_angular_velocity: true` |
| Include Center Offset | ✅ | `has_center_offset: true` |
| Include Extent | ✅ | *(no key — invariant, lands in the scalar block)* |
| Virtual Root | the arena root | — |

Row width: `3 + 4 + 3 + 3 + 3 + 3 + 2 + 5 = 26`.
Entities: `9` self + `16` obstacles = **25**.

A mismatch between toggles and config keys does not fail loudly — it
reinterprets geometry columns as one-hots and trains anyway. The encoder logs its
layout at startup; check it reads

```
pos[0:3] | quat[3:7]->3 axes | linvel[7:10] | angvel[10:13] | center_offset[13:16] | up(const) | scalars[16:26]
```

with `vec_channels=7`.

## Behavior Parameters

| Field | Value |
|---|---|
| Behavior Name | `CrawlerParkour` |
| Continuous Actions | 20 |
| Vector Observation Size | 168 |
| Stacked Vectors | 1 |
| Model | none |

168 comes from `CrawlerParkourAgent.ObservationCount`, which the builder reads
rather than recomputing — 42 fixed floats (6 orientation + 6 velocity + 1 height
+ 4 foot contacts + 5 track-relative + 20 previous actions) plus
`2 × GridForward × GridLateral` for the height field. Duplicating that arithmetic
in the builder is how it was briefly 174: declaring more floats than the agent
writes pads the tail with zeros and trains happily.

**DecisionRequester**: period 5, `TakeActionsBetweenDecisions = false`. The
action-rate cost assumes `OnActionReceived` fires once per decision.

## Arena replication

32 arenas in a 4×8 grid, 60 apart on X and 220 on Z. Track length is
`SegmentCount × SegmentLength` = 128 and width reaches 12. No arena has floor
outside its own track, so a crawler that falls off falls into open space rather
than onto a neighbour.

## Verify before launching

```
# geometry bounds, from the real generator, no Unity involved
cd Project/Assets/CrawlerParkour/Tests~ && dotnet run -c Release

# arena replication, sensor round-trip, spawn ground
Unity.exe -batchmode -nographics -projectPath Project `
  -executeMethod CrawlerParkourBuilder.VerifyMultiArena -logFile verify.log
```

Expect `ALL TRAVERSABLE` with a repair count near zero, and `VERIFY OK`.

A repair count in the thousands means a pattern is emitting geometry its own
check rejects and the track is quietly flattening — training would still work,
and would silently be far easier than intended.
