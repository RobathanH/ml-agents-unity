using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;
using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// Crawler that runs a procedurally generated obstacle track for time.
    /// Action space matches CrawlerSumo (20 continuous: 12 joint targets, 8
    /// strengths) so run 013's control-cost findings carry over unchanged.
    /// </summary>
    [RequireComponent(typeof(JointDriveController))]
    public class CrawlerParkourAgent : Agent
    {
        [Header("Body Parts")]
        public Transform body;
        public Transform leg0Upper, leg0Lower;
        public Transform leg1Upper, leg1Lower;
        public Transform leg2Upper, leg2Lower;
        public Transform leg3Upper, leg3Lower;

        [Header("Track (set by controller)")]
        public ParkourTrackGenerator track;

        [Header("Height field")]
        [Tooltip("Longitudinal samples, from BackExtent to ForwardExtent.")]
        public int GridForward = 9;
        [Tooltip("Lateral samples, spanning +/- SideExtent.")]
        public int GridLateral = 7;
        public float ForwardExtent = 3.5f;
        public float BackExtent = -1f;
        public float SideExtent = 1.5f;
        public LayerMask GroundMask = ~0;

        // Reward weights, pushed in from environment_parameters by the controller.
        //
        // Run 006: every positive term is a RATE -- reward per metre of new ground,
        // reward per second of good locomotion -- instead of a per-episode budget
        // divided by TrackLength or MaxDecisions. Both of those denominators were
        // "finish the track" concepts, and they coupled the reward scale to two
        // numbers that are now free parameters. Doubling the episode length for run
        // 005 moved every term by a different factor, which is what made that run's
        // curriculum thresholds underivable (DESIGN.md 5.2 and 6). As rates,
        // episode length and track length can both change without rescaling
        // anything, so a 30s training episode and a 140s eval rollout score on the
        // same axis.
        //
        // Run 005's settings are recovered exactly at progressPerMeter = 10/128 =
        // 0.078 and velocityPerSecond = 2.0 / 1200 decisions / 0.1 s = 0.0167, so
        // this is numerically an identity at launch and the warm start is not
        // shocked by it.
        [System.NonSerialized] public float progressPerMeter = 0.08f;
        [System.NonSerialized] public float energyCostWeight;
        [System.NonSerialized] public float actionRateCostWeight;
        [System.NonSerialized] public float respawnPenalty;
        [System.NonSerialized] public float velocityPerSecond = 0.0167f;
        [System.NonSerialized] public float targetSpeed = 2.5f;

        /// <summary>
        /// Everything CollectObservations writes outside the height field:
        /// 6 orientation (up, forward in the yaw frame) + 6 velocity + 1 height
        /// above ground + 4 foot contacts + 5 track-relative + 20 previous actions.
        /// </summary>
        /// <remarks>
        /// Here rather than in the scene builder because it has to track the body
        /// of CollectObservations, and the failure it guards against is quiet:
        /// BehaviorParameters declaring more floats than the agent writes pads the
        /// tail with zeros and trains anyway.
        ///
        /// Run 006 retires two of the five track-relative slots (see
        /// CollectObservations) but keeps WRITING them, as constant zero. Deleting
        /// them would take this from 42 to 40, change ObservationCount, and make run
        /// 005's 68M-step checkpoint incompatible on the first layer of the vector
        /// encoder AND on the running observation normaliser. A constant input is
        /// just a bias shift that PPO absorbs in a few thousand steps, so the cheap
        /// version buys the same behaviour for none of the risk. Delete them
        /// properly at the next architecture change.
        /// </remarks>
        public const int NonGridObservations = 6 + 6 + 1 + 4 + 5 + 20;

        /// <summary>Total vector observation size, for BehaviorParameters.</summary>
        public int ObservationCount => NonGridObservations + 2 * GridForward * GridLateral;

        /// <summary>Number of distinct <see cref="SegmentPattern"/> values, for the
        /// per-pattern traversal counters.</summary>
        public const int PatternCount = 10;

        public float MaxProgress { get; private set; }
        public int RespawnCount { get; private set; }
        public float EpisodeEnergyCost { get; private set; }
        public float EpisodeActionRateCost { get; private set; }
        public float EpisodeVelocityReward { get; private set; }

        /// <summary>Track-space z the episode started at. Run 006 spawns in the
        /// middle of the track, so this is no longer ~0 and every distance measure
        /// has to be taken relative to it.</summary>
        public float StartZ { get; private set; }

        /// <summary>New ground covered this episode, metres. Replaces
        /// ProgressFraction: a fraction of a track length only means something when
        /// there is a track to finish.</summary>
        public float DistanceCovered => MaxProgress - StartZ;

        /// <summary>Segments the agent entered / fully cleared / fell inside, by
        /// pattern. Indexed by (int)SegmentPattern. This is the measure that says
        /// whether the policy is doing parkour or just walking the flat stretches --
        /// nothing before run 006 distinguished crossing a Gap from crossing a
        /// Flat, so a single finish rate had to stand in for all ten.</summary>
        public readonly int[] SegmentsEntered = new int[PatternCount];
        public readonly int[] SegmentsCleared = new int[PatternCount];
        public readonly int[] SegmentFalls = new int[PatternCount];

        /// <summary>
        /// Mean down-track speed over the episode, m/s. The one number that says
        /// whether the crawler walks: run 002 improved its reward by a full point
        /// while this sat at ~0.02, so reward alone cannot be read as progress.
        /// </summary>
        public float MeanForwardSpeed => m_Decisions > 0 ? m_ForwardSpeedSum / m_Decisions : 0f;

        private const int k_NumStrengthActions = 8;
        private JointDriveController m_Jd;
        private float[] m_PrevActions;
        private bool m_HasPrevActions;
        private float m_CheckpointZ;
        private Transform[] m_Feet;
        private int m_DecisionPeriod = 1;
        private float m_DecisionSeconds = 0.02f;
        private float m_ForwardSpeedSum;
        private int m_Decisions;
        // Highest segment index the agent has ENTERED and the highest it has fully
        // cleared, so each segment is counted exactly once however many times the
        // agent crosses back over the boundary after a fall.
        private int m_EnteredThrough;
        private int m_ClearedThrough;

        public override void Initialize()
        {
            m_Jd = GetComponent<JointDriveController>();
            m_Jd.bodyPartsDict.Clear();
            m_Jd.bodyPartsList.Clear();
            foreach (var t in new[] { body, leg0Upper, leg0Lower, leg1Upper, leg1Lower,
                                      leg2Upper, leg2Lower, leg3Upper, leg3Lower })
            {
                m_Jd.SetupBodyPart(t);
            }
            m_Feet = new[] { leg0Lower, leg1Lower, leg2Lower, leg3Lower };
            MaxStep = 0;   // controller owns episode length

            // StepCount counts DECISIONS, while the controller's episode cap counts
            // physics steps -- they differ by this factor. Anything comparing the
            // two has to divide, which the run 005 finish bonus previously did not.
            var requester = GetComponent<DecisionRequester>();
            m_DecisionPeriod = requester != null ? Mathf.Max(1, requester.DecisionPeriod) : 1;
            // Wall-clock seconds one decision covers. This is what turns
            // velocityPerSecond into a per-decision charge, and it is the only place
            // the physics tick enters the reward.
            m_DecisionSeconds = m_DecisionPeriod * Time.fixedDeltaTime;
        }

        /// <summary>
        /// Where the controller wants the agent to start. Applied here rather than
        /// by the controller because BodyPart.Reset restores recorded WORLD
        /// transforms, and EndEpisode calls OnEpisodeBegin synchronously -- so any
        /// placement the controller does before EndEpisode is immediately undone.
        /// Teleporting after the reset, inside the same call, is the only ordering
        /// that holds regardless of who triggers the episode.
        /// </summary>
        [System.NonSerialized] public Vector3 SpawnPoint;

        /// <summary>Heading offset applied at spawn, degrees about world up.</summary>
        /// <remarks>
        /// Run 006. BodyPart.Reset restores an identical recorded pose every
        /// episode, so before this the policy had never seen a start that was not
        /// perfectly square to the track -- the only off-axis states it ever
        /// experienced were the ones it had already fallen into. A policy that is
        /// supposed to pick up and keep moving from an arbitrary point in an
        /// arbitrary track needs arbitrary headings in its start distribution.
        ///
        /// Deliberately a RIGID rotation of the whole assembly, not per-joint
        /// jitter. The ConfigurableJoint anchors are computed once in
        /// SetupBodyPart, and perturbing individual body-part transforms after
        /// Reset risks popping a joint into a state the solver has to fight out of
        /// -- which would show up as a mystery spike in early-episode reward rather
        /// than as an error. A rigid transform of every part is exactly what
        /// TeleportTo already does for translation and is safe for the same reason.
        /// </remarks>
        [System.NonSerialized] public float SpawnYawDegrees;

        /// <summary>Body velocity the episode starts with, world space. Small, and
        /// nonzero for the same reason as the yaw: standing perfectly still is one
        /// state out of many the policy has to be able to continue from.</summary>
        [System.NonSerialized] public Vector3 SpawnVelocity;

        public override void OnEpisodeBegin()
        {
            foreach (var bp in m_Jd.bodyPartsDict.Values) bp.Reset(bp);
            PlaceAt(SpawnPoint, SpawnYawDegrees, SpawnVelocity);

            // Run 006 spawns mid-track, so MaxProgress must start at the spawn
            // rather than at zero. Left at 0 the progress ratchet would pay the
            // whole distance from the start line to the spawn point on the first
            // decision -- 0.08 x 40m = 3.2 of free reward, several times a real
            // episode's total. (This existed in miniature before: spawning at z=1
            // against MaxProgress 0 paid 0.078 every episode.)
            StartZ = track != null ? track.ToTrack(SpawnPoint).z : 0f;
            MaxProgress = StartZ;
            RespawnCount = 0;
            m_CheckpointZ = track != null
                ? Mathf.Floor(StartZ / track.SegmentLength) * track.SegmentLength : 0f;
            // The spawn segment is entered part-way through, so it is counted as
            // NEITHER entered nor cleared -- crediting a clear for the metre of a
            // Squeeze that happened to be left ahead of the spawn would inflate
            // exactly the statistic these counters exist to measure honestly. Only
            // segments met at their own start boundary count.
            m_EnteredThrough = track != null ? track.SegmentIndexAt(StartZ) : 0;
            m_ClearedThrough = m_EnteredThrough;
            System.Array.Clear(SegmentsEntered, 0, PatternCount);
            System.Array.Clear(SegmentsCleared, 0, PatternCount);
            System.Array.Clear(SegmentFalls, 0, PatternCount);

            m_HasPrevActions = false;
            EpisodeEnergyCost = 0f;
            EpisodeActionRateCost = 0f;
            EpisodeVelocityReward = 0f;
            m_ForwardSpeedSum = 0f;
            m_Decisions = 0;
        }

        /// <summary>
        /// Yaw-only frame: heading without pitch or roll. Observations expressed in
        /// it stay egocentric while remaining well defined when the body tumbles,
        /// which the full body frame is not.
        /// </summary>
        private Quaternion YawFrame()
        {
            var fwd = body.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            return Quaternion.LookRotation(fwd.normalized, Vector3.up);
        }

        /// <summary>
        /// Body position in the track's frame. The generator places geometry
        /// relative to its own root so that arenas are translations of one another,
        /// so every comparison against track coordinates has to go through here.
        /// Rigidbody positions, raycasts and the EGNN sensor all stay in world space.
        /// </summary>
        private Vector3 TrackPos => track != null ? track.ToTrack(body.position) : body.position;

        public override void CollectObservations(VectorSensor sensor)
        {
            var inv = Quaternion.Inverse(YawFrame());
            var rb = m_Jd.bodyPartsDict[body].rb;

            sensor.AddObservation(inv * body.up);
            sensor.AddObservation(inv * body.forward);
            sensor.AddObservation(inv * rb.linearVelocity * 0.1f);
            sensor.AddObservation(inv * rb.angularVelocity * 0.1f);

            var tp = TrackPos;
            float groundY = 0f;
            bool haveGround = track != null && track.GroundHeightAt(tp.x, tp.z, out groundY);
            sensor.AddObservation(haveGround ? Mathf.Clamp((tp.y - groundY) / 3f, -2f, 2f) : -2f);

            foreach (var foot in m_Feet)
            {
                var gc = m_Jd.bodyPartsDict[foot].groundContact;
                sensor.AddObservation(gc != null && gc.touchingGround);
            }

            // Track-relative situation: LATERAL only.
            //
            // The last two slots used to be Clamp01(z / TrackLength) and
            // Clamp01(MaxProgress / TrackLength) -- "how far through the track am
            // I". That is goal-conditioning on a goal run 006 removes, and with a
            // mid-track spawn it is worse than useless: it is a feature the policy
            // can key absolute track position on, which is exactly the dependence
            // "keep moving at any point in any track" is meant to break. Held at
            // zero rather than deleted so the observation vector keeps its shape and
            // run 005's checkpoint stays loadable -- see NonGridObservations.
            float halfW = track != null ? track.TrackWidth * 0.5f : 1f;
            float laneX = track != null ? track.LaneCenterAt(tp.z) : 0f;
            sensor.AddObservation(Mathf.Clamp((tp.x - laneX) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp((halfW - tp.x) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp((tp.x + halfW) / 5f, -2f, 2f));
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);

            AddHeightField(sensor);

            if (m_PrevActions != null)
            {
                foreach (var a in m_PrevActions) sensor.AddObservation(a);
            }
            else
            {
                for (int i = 0; i < 20; i++) sensor.AddObservation(0f);
            }
        }

        /// <summary>
        /// Downward and upward raycasts on a yaw-aligned grid around the body.
        /// </summary>
        /// <remarks>
        /// This is the half of perception the EGNN should not be asked to do.
        /// "How high is the ground just ahead of my front-left foot" is a query
        /// against the union of all boxes; answering it through message passing
        /// means learning a max-reduction over box SDFs. The entity graph handles
        /// object-level reasoning -- what a thing is, how it is oriented, how big,
        /// where it is going -- and this handles surface-level reasoning.
        /// </remarks>
        private void AddHeightField(VectorSensor sensor)
        {
            var frame = YawFrame();
            var origin = body.position;
            for (int i = 0; i < GridForward; i++)
            {
                float fz = Mathf.Lerp(BackExtent, ForwardExtent, GridForward == 1 ? 0.5f : i / (float)(GridForward - 1));
                for (int j = 0; j < GridLateral; j++)
                {
                    float fx = Mathf.Lerp(-SideExtent, SideExtent, GridLateral == 1 ? 0.5f : j / (float)(GridLateral - 1));
                    var p = origin + frame * new Vector3(fx, 0f, fz);

                    float floor = -1f;
                    if (Physics.Raycast(p + Vector3.up * 3f, Vector3.down, out var hit, 9f, GroundMask))
                    {
                        floor = Mathf.Clamp((origin.y - hit.point.y) / 3f, -1f, 2f);
                    }
                    sensor.AddObservation(floor);

                    float head = 1f;
                    if (Physics.Raycast(p + Vector3.up * 0.05f, Vector3.up, out var up, 4f, GroundMask))
                    {
                        head = Mathf.Clamp01(up.distance / 4f);
                    }
                    sensor.AddObservation(head);
                }
            }
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            var bp = m_Jd.bodyPartsDict;
            var c = actionBuffers.ContinuousActions;
            int i = -1;

            bp[leg0Upper].SetJointTargetRotation(c[++i], c[++i], 0);
            bp[leg1Upper].SetJointTargetRotation(c[++i], c[++i], 0);
            bp[leg2Upper].SetJointTargetRotation(c[++i], c[++i], 0);
            bp[leg3Upper].SetJointTargetRotation(c[++i], c[++i], 0);
            bp[leg0Lower].SetJointTargetRotation(c[++i], 0, 0);
            bp[leg1Lower].SetJointTargetRotation(c[++i], 0, 0);
            bp[leg2Lower].SetJointTargetRotation(c[++i], 0, 0);
            bp[leg3Lower].SetJointTargetRotation(c[++i], 0, 0);

            bp[leg0Upper].SetJointStrength(c[++i]);
            bp[leg1Upper].SetJointStrength(c[++i]);
            bp[leg2Upper].SetJointStrength(c[++i]);
            bp[leg3Upper].SetJointStrength(c[++i]);
            bp[leg0Lower].SetJointStrength(c[++i]);
            bp[leg1Lower].SetJointStrength(c[++i]);
            bp[leg2Lower].SetJointStrength(c[++i]);
            bp[leg3Lower].SetJointStrength(c[++i]);

            ApplyVelocityReward();
            ApplyProgressReward();
            ApplyControlCosts(c);
        }

        /// <summary>
        /// Pays every decision for moving down-track at roughly target speed while
        /// facing that way. This is the term run 002 did not have, and its absence
        /// is why that run converged on standing still.
        /// </summary>
        /// <remarks>
        /// The progress ratchet is the only other positive term and it is
        /// conditional on NET forward displacement, while the control costs are
        /// charged every step. Before a gait exists that pairing makes stillness
        /// strictly optimal -- the agent pays continuously and earns nothing -- and
        /// PPO finds that optimum quickly and never leaves.
        ///
        /// Shape is the stock ML-Agents Crawler's: 1 at target speed, falling to 0
        /// at standstill AND at twice target, so there is nothing to gain by
        /// sprinting and nothing to gain by reversing. Because it is unsigned and
        /// zero for backward motion, oscillating over the same stretch earns
        /// nothing, which is the exploit the ratchet was chosen to avoid -- so this
        /// term can be added without giving that exploit back.
        ///
        /// SECONDARY by design in run 006, and the reason is the one thing this
        /// environment wants that stock legged-locomotion setups do not. A
        /// velocity-tracking term charges the agent for every step it is stopped,
        /// which prices a deliberate pause to set up a leap as a loss. The
        /// displacement ratchet is indifferent to WHEN the ground gets covered, so
        /// it tolerates the tactical slowdown a hard obstacle needs while still
        /// paying nothing for oscillation. Velocity tracking is primary in
        /// legged-gym because that work needs a controller that FOLLOWS a commanded
        /// velocity; we want "as far as possible" instead, and for that the ratchet
        /// is the better primitive. This term stays as shaping and as the thing that
        /// pays for facing down-track (the `heading` factor).
        /// </remarks>
        private void ApplyVelocityReward()
        {
            // Track space differs from world space by a translation only
            // (ToTrack/ToWorld), so a world-space direction is already track-space.
            float vz = AvgVelocity().z;
            m_ForwardSpeedSum += vz;
            m_Decisions++;

            // Speed is accounted above this guard, not below it: MeanForwardSpeed is
            // the stat that tells run 002's failure apart from real progress, and an
            // ablation setting velocity_per_second to 0 is exactly when it is needed.
            if (velocityPerSecond <= 0f || targetSpeed <= 0f) return;

            float d = Mathf.Clamp(Mathf.Abs(vz - targetSpeed), 0f, targetSpeed) / targetSpeed;
            float speed = (1f - d * d) * (1f - d * d);
            float heading = (body.forward.z + 1f) * 0.5f;

            // Per SECOND, charged over the seconds this decision covers -- so the
            // term is independent of both episode length and decision period. The
            // old form divided a whole-episode budget by MaxDecisions, which meant
            // changing the episode cap silently rescaled it.
            float r = velocityPerSecond * m_DecisionSeconds * speed * heading;
            AddReward(r);
            EpisodeVelocityReward += r;
        }

        /// <summary>
        /// Mean velocity over every body part rather than the torso alone: torso-only
        /// velocity is satisfied by throwing the limbs around, which reads as speed
        /// without moving the animal. Same reasoning as the stock Crawler.
        /// </summary>
        private Vector3 AvgVelocity()
        {
            var parts = m_Jd.bodyPartsList;
            if (parts.Count == 0) return Vector3.zero;
            var sum = Vector3.zero;
            for (int i = 0; i < parts.Count; i++) sum += parts[i].rb.linearVelocity;
            return sum / parts.Count;
        }

        /// <summary>
        /// Pays only for ground never covered before.
        /// </summary>
        /// <remarks>
        /// Rewarding the signed delta would let an agent farm reward by oscillating
        /// across the same stretch, and would charge a huge one-off negative the
        /// moment a respawn teleports it backwards. Paying on max-so-far makes a
        /// respawn cost exactly what it should -- the time to re-cover the ground --
        /// and makes going backwards worth nothing rather than worth punishing.
        /// </remarks>
        private void ApplyProgressReward()
        {
            if (track == null) return;
            float z = TrackPos.z;
            if (z <= MaxProgress) return;

            // Per METRE. There is no TrackLength denominator any more, because
            // there is no track to finish: a fraction-of-track reward would mean
            // that generating a longer track for a long eval rollout silently
            // divided the reward, and would make training and eval score on
            // different axes.
            AddReward(progressPerMeter * (z - MaxProgress));
            MaxProgress = z;

            // Checkpoint at each segment boundary cleared.
            float seg = track.SegmentLength;
            m_CheckpointZ = Mathf.Floor(MaxProgress / seg) * seg;

            int reached = track.SegmentIndexAt(MaxProgress);
            while (m_EnteredThrough < reached)
            {
                m_EnteredThrough++;
                SegmentsEntered[(int)track.PatternAt(m_EnteredThrough)]++;
            }
            // A segment counts as cleared once the agent is past its FAR boundary,
            // which is the same thing as having entered the next one. Standing on
            // top of a hurdle is not clearing it.
            while (m_ClearedThrough < reached - 1)
            {
                m_ClearedThrough++;
                SegmentsCleared[(int)track.PatternAt(m_ClearedThrough)]++;
            }
        }

        /// <summary>Episode length the controller is enforcing, in PHYSICS steps.
        /// Kept for reporting only -- run 006 has no reward term that depends on
        /// it, which is what makes the policy episode-length invariant and lets a
        /// 30s training episode be evaluated over a 140s rollout.</summary>
        [System.NonSerialized] public int MaxStepOverride = 1500;

        private void ApplyControlCosts(ActionSegment<float> c)
        {
            int n = c.Length;
            if (m_PrevActions == null || m_PrevActions.Length != n)
            {
                m_PrevActions = new float[n];
                m_HasPrevActions = false;
            }

            if (energyCostWeight > 0f || actionRateCostWeight > 0f)
            {
                float energy = 0f;
                int start = Mathf.Max(0, n - k_NumStrengthActions);
                for (int a = start; a < n; a++)
                {
                    float s = (c[a] + 1f) * 0.5f;
                    energy += s * s;
                }
                energy = (n - start) > 0 ? energy / (n - start) : 0f;

                float rate = 0f;
                if (m_HasPrevActions)
                {
                    for (int a = 0; a < n; a++)
                    {
                        float d = c[a] - m_PrevActions[a];
                        rate += d * d;
                    }
                    rate /= n;
                }

                AddReward(-energyCostWeight * energy - actionRateCostWeight * rate);
                EpisodeEnergyCost += energyCostWeight * energy;
                EpisodeActionRateCost += actionRateCostWeight * rate;
            }

            for (int a = 0; a < n; a++) m_PrevActions[a] = c[a];
            m_HasPrevActions = true;
        }

        /// <summary>
        /// True when the agent has fallen off the track and needs respawning.
        /// </summary>
        public bool HasFallen(float fallDepth)
        {
            if (track == null) return false;
            var tp = TrackPos;
            if (!track.GroundHeightAt(tp.x, tp.z, out float g))
            {
                // No ground under it at all: over the void.
                return tp.y < -fallDepth;
            }
            return tp.y < g - fallDepth;
        }

        /// <summary>
        /// Returns to the last cleared checkpoint. Failure is recoverable and
        /// costs time, not the episode: a terminal fall would price risk so high
        /// that the optimal policy is a slow shuffle, which is precisely the
        /// single-mode outcome this environment exists to avoid.
        /// </summary>
        public void RespawnAtCheckpoint()
        {
            RespawnCount++;
            AddReward(-respawnPenalty);
            // Attributed to the segment the agent fell OUT of, which is where it is
            // now, not the checkpoint it is about to be returned to. Falls charged
            // to the checkpoint segment would credit every fall to whatever pattern
            // preceded the hard one.
            if (track != null) SegmentFalls[(int)track.PatternAt(track.SegmentIndexAt(TrackPos.z))]++;

            TeleportTo(CheckpointSpawn());
        }

        /// <summary>Spawn point for the last cleared checkpoint, on the lane so the
        /// agent restarts on the traversable route rather than beside it. Returned
        /// in world space, ready for <see cref="TeleportTo"/>.</summary>
        public Vector3 CheckpointSpawn()
        {
            if (track == null) return new Vector3(0f, 1.2f, m_CheckpointZ + 0.5f);
            float x = track.LaneCenterAt(m_CheckpointZ);
            float z = m_CheckpointZ + 0.5f;
            float y = track.GroundHeightAt(x, z, out float g) ? g + 1.2f : 1.2f;
            return track.ToWorld(new Vector3(x, y, z));
        }

        /// <summary>
        /// Rigid translation of the whole body, preserving the current pose. A
        /// respawn should not also re-fold the legs -- the agent has to recover
        /// from wherever the fall left it.
        /// </summary>
        private void TeleportTo(Vector3 target)
        {
            var delta = target - body.position;
            foreach (var bp in m_Jd.bodyPartsDict.Values)
            {
                bp.rb.linearVelocity = Vector3.zero;
                bp.rb.angularVelocity = Vector3.zero;
                bp.rb.transform.position += delta;
            }
        }

        /// <summary>
        /// Rigid placement of the whole body at a point and heading, with a given
        /// starting velocity. Used at episode start; respawns use
        /// <see cref="TeleportTo"/> instead, because a fall should not also re-square
        /// the animal to the track.
        /// </summary>
        /// <remarks>
        /// The pivot is read BEFORE the loop: <c>body</c> is itself one of the body
        /// parts, so reading it inside would rotate the remaining parts about an
        /// already-moved origin and tear the rig apart.
        /// </remarks>
        private void PlaceAt(Vector3 target, float yawDegrees, Vector3 velocity)
        {
            var rot = Quaternion.AngleAxis(yawDegrees, Vector3.up);
            var pivot = body.position;
            foreach (var bp in m_Jd.bodyPartsDict.Values)
            {
                var tr = bp.rb.transform;
                tr.position = target + rot * (tr.position - pivot);
                tr.rotation = rot * tr.rotation;
                bp.rb.linearVelocity = velocity;
                bp.rb.angularVelocity = Vector3.zero;
            }
        }
    }
}
