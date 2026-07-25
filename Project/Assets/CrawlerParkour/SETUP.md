# CrawlerParkour — Unity scene setup

The C# is complete and validated; the scene and prefab still have to be built in
the editor. This is the exact wiring the training config assumes.

## 1. Arena prefab

```
CrawlerParkourEnv                (CrawlerParkourEnvController)
├── Track                        (ParkourTrackGenerator, ParkourObstacleSource)
│   └── Boxes                    (empty; BoxParent)
└── Crawler                      (copy of ML-Agents Crawler prefab)
    ├── Body                     + EGNNSensorComponent, BehaviorParameters,
    │                              DecisionRequester, CrawlerParkourAgent,
    │                              JointDriveController
    ├── leg0Upper … leg3Upper
    └── leg0Lower … leg3Lower
```

**BoxPrefab**: a unit cube (1×1×1) with `BoxCollider` and `MeshRenderer`, no
Rigidbody. `ParkourBox.Place` sets `localScale = halfExtents * 2`, so the prefab
must be exactly 1 unit per side or every obstacle is mis-sized.

Give it a physics material with friction ≈ 0.8. The 40° ramp ceiling is
`arctan(µ)`; a slippier material makes the steepest ramps unclimbable and the
generator has no way to know.

## 2. EGNN sensor — the subtype labels matter

Configure one root group:

| Field | Value |
|---|---|
| Root | `Crawler` |
| Type | `self` |
| IncludeChildren | true |
| RootSubType | `body` |

Then **explicitly set the `SubType` of every child override**:

- `leg0Upper` … `leg3Upper` → `upper`
- `leg0Lower` … `leg3Lower` → `lower`

This is not cosmetic. When `SubType` is blank the sensor falls back to the
GameObject's name, which would mint eight distinct one-hot slots — `leg0Upper`,
`leg1Upper`, … — instead of one shared `upper`. That both widens the row and
destroys the point of sharing node types: the policy could then identify a
specific leg by its label rather than by where it is, and nothing would
generalise across legs.

Add `ParkourObstacleSource` to the sensor's **Entity Sources** list. It declares
`obstacle` / {`static`, `dynamic`} and a budget of `MaxObstacles`.

Resulting vocabulary: types {`self`, `obstacle`} = 2, subtypes {`body`, `upper`,
`lower`, `static`, `dynamic`} = 5.

### Sensor toggles (these MUST match the training config)

| Toggle | Value | Config key |
|---|---|---|
| Include Rotation | ✅ | `has_quaternion: true` |
| Include Linear Velocity | ✅ | `has_linear_velocity: true` |
| Include Angular Velocity | ✅ | `has_angular_velocity: true` |
| Include Center Offset | ✅ | `has_center_offset: true` |
| Include Extent | ✅ | *(no key — invariant, lands in the scalar block)* |
| Virtual Root | the arena root | — |

Row width: `3 + 4 + 3 + 3 + 3 + 3 + 2 + 5 = 26`.
Entities: `9` self + `MaxObstacles` (16) = **25**.

A mismatch between the toggles and the config keys silently reinterprets
geometry columns as one-hots. The encoder logs its layout at startup — check the
`[EGNN] Sensor 'EGNNSensor' entity layout:` line reads
`pos[0:3] | quat[3:7]->3 axes | linvel[7:10] | angvel[10:13] | center_offset[13:16] | up(const) | scalars[16:26]`
and that `vec_channels=7`.

## 3. Behavior Parameters

| Field | Value |
|---|---|
| Behavior Name | `CrawlerParkour` |
| Continuous Actions | 20 |
| Vector Observation Size | see below |
| Stacked Vectors | 1 |

Vector observations, in `CollectObservations` order:

| Block | Count |
|---|---|
| body up, forward (yaw frame) | 6 |
| body linear + angular velocity | 6 |
| height above ground | 1 |
| foot ground contacts | 4 |
| lane offset, two edge distances, z fraction, progress fraction | 5 |
| height field (9 × 7 × {floor, headroom}) | 126 |
| previous actions | 20 |
| **total** | **168** |

Changing `GridForward` / `GridLateral` changes this — update the size to
`48 + 2 * GridForward * GridLateral`.

**DecisionRequester**: period 5, `TakeActionsBetweenDecisions = false`. The
action-rate cost assumes `OnActionReceived` fires once per decision.

**Ground mask**: set the agent's `GroundMask` to exclude the crawler's own
colliders, or the downward rays hit its own legs and the height field becomes
noise.

## 4. Arena replication

32 arenas per env process, spaced ≥ 40 on X (track width reaches 12 and boxes
sit inside that) and ≥ 200 on Z (track length is `SegmentCount * SegmentLength`
= 128, plus margin). Arenas must not share physics space — a crawler that falls
off must not land on a neighbour's track.

## 5. Verify before launching

Run the feasibility harness after any change to the generator or its bounds:

```
cd Project/Assets/CrawlerParkour/Tests~
dotnet run -c Release
```

Expect `ALL TRAVERSABLE` and a repair count near zero. A repair count in the
thousands means a pattern is emitting geometry its own check rejects and the
track is quietly flattening — the training run would still work, and would
silently be much easier than intended.
