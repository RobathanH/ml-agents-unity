using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;
using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// Crawler that runs a procedurally generated obstacle track for time.
    /// </summary>
    /// <remarks>
    /// RUN 009 widens the action space from 20 to 24. The first twenty are
    /// unchanged in meaning and order -- 12 joint targets then 8 strengths, which
    /// is CrawlerSumo's layout, so run 013's control-cost findings still carry --
    /// and the four new ones command the friction of each foot independently. See
    /// <see cref="FootFrictionMax"/> for the arithmetic and DESIGN.md 11.
    /// </remarks>
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

        /// <summary>Reward per second with every foot clear of the ground. Run 009's
        /// answer to the last surviving Gap hypothesis; see
        /// <see cref="ApplyAirborneReward"/>.</summary>
        [System.NonSerialized] public float airbornePerSecond;

        // ------------------------------------------------------------- actions
        //
        // The first twenty are run 008's, unchanged in order and meaning. Friction
        // is APPENDED rather than interleaved so that expanding a run 008 checkpoint
        // onto this action space is a pure append on the output head, with the
        // existing rows left where they are.

        /// <summary>8 upper-leg (x, y) + 4 lower-leg (x) joint targets.</summary>
        public const int NumJointTargetActions = 12;
        /// <summary>Per-joint drive strengths, one per actuated joint.</summary>
        public const int NumStrengthActions = 8;
        /// <summary>Per-foot friction commands. Run 009.</summary>
        public const int NumFrictionActions = 4;

        public const int FirstStrengthAction = NumJointTargetActions;
        public const int FirstFrictionAction = NumJointTargetActions + NumStrengthActions;
        public const int ActionCount =
            NumJointTargetActions + NumStrengthActions + NumFrictionActions;

        // ----------------------------------------------------------- friction
        //
        // WHAT THE NUMBER 1.375 IS, because it looks arbitrary and is not.
        //
        // Run 008's crawler carried no physics material at all, so every contact
        // paired the track's ParkourGround (1.6) with Unity's implicit default
        // (0.6) under Average combine: a contact coefficient of 1.1. The foot
        // materials created here use MULTIPLY combine, which wins over Average
        // because Unity resolves a pair by the higher combine enum -- so the
        // contact coefficient becomes GroundFriction * (what the agent commands).
        //
        // The brief was "0 to twice the current friction", i.e. contact in
        // [0, 2.2], so the commanded multiplier spans [0, 2.2 / 1.6] = [0, 1.375].
        // Which makes the midpoint 0.6875, and 1.6 * 0.6875 = 1.1 EXACTLY -- the
        // run 008 contact. A zero action is therefore not a neutral-ish default,
        // it is bit-for-bit the old physics, which is what makes it meaningful to
        // initialise the four new policy outputs at zero and claim the warm start
        // is behaviourally identical at launch.
        //
        // Kept next to the runtime that consumes it rather than in the builder,
        // because the materials are created per agent at run time -- see
        // SetupFrictionMaterials for why they cannot be shared assets.
        public const float GroundFriction = 1.6f;
        private const float k_UnityDefaultFriction = 0.6f;
        /// <summary>Contact coefficient run 008 actually ran, Average combine.</summary>
        public const float Run008ContactFriction =
            (GroundFriction + k_UnityDefaultFriction) * 0.5f;
        /// <summary>Largest multiplier a foot may command: twice run 008's contact.</summary>
        public const float FootFrictionMax = 2f * Run008ContactFriction / GroundFriction;

        /// <summary>
        /// Fixed multiplier for every collider that is NOT a foot -- the body, its
        /// decoration, and the four upper legs.
        /// </summary>
        /// <remarks>
        /// These surfaces are not meant to be pushed against; when they touch
        /// terrain the crawler is scraping, and grip there is what turns a scrape
        /// into a snag. 0.15 x 1.6 = 0.24 contact, against 1.1 in run 008, so a
        /// belly or a thigh resting on an edge now slides off it.
        ///
        /// Deliberately not zero. A frictionless body cannot brace against a wall
        /// at all, and bellying over a hurdle -- one of the behaviours this
        /// environment exists to produce -- needs the torso to purchase something.
        /// </remarks>
        public const float BodyFriction = 0.15f;

        /// <summary>
        /// Everything CollectObservations writes outside the height field:
        /// 6 orientation (up, forward in the yaw frame) + 6 velocity + 1 height
        /// above ground + 4 foot contacts + 3 track-relative + 24 previous actions.
        /// </summary>
        /// <remarks>
        /// Here rather than in the scene builder because it has to track the body
        /// of CollectObservations, and the failure it guards against is quiet:
        /// BehaviorParameters declaring more floats than the agent writes pads the
        /// tail with zeros and trains anyway.
        ///
        /// Run 006 retired two of the five track-relative slots but kept WRITING
        /// them as constant zero, because deleting them would have changed this
        /// count and broken run 005's checkpoint on both the first layer of the
        /// vector encoder and the running observation normaliser. The note left
        /// there said to delete them properly at the next architecture change.
        ///
        /// RUN 009 IS THAT CHANGE. The action space widens to 24, which moves the
        /// previous-action block and the encoder's input width anyway, so the two
        /// dead slots cost something to keep and nothing to remove. 42 -> 44.
        /// </remarks>
        public const int NonGridObservations = 6 + 6 + 1 + 4 + 3 + ActionCount;

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
        public float EpisodeAirborneReward { get; private set; }

        /// <summary>Fraction of this episode's decisions with all four feet clear of
        /// the ground. The measure that says whether run 009's air-phase price
        /// bought any air; without it the term is unfalsifiable.</summary>
        public float AirborneFraction => m_Decisions > 0 ? m_AirborneDecisions / (float)m_Decisions : 0f;

        /// <summary>Mean commanded friction multiplier over every foot and decision.
        /// A policy that ignores the new actuator sits at
        /// <see cref="FootFrictionMax"/>/2; one that has learned to release its feet
        /// sits below it.</summary>
        public float MeanFootFriction => m_Decisions > 0 ? m_FrictionSum / (m_Decisions * (float)NumFrictionActions) : 0f;

        /// <summary>Fraction of foot-decisions commanding near-zero grip. Separated
        /// from the mean because "releases two feet hard while gripping with two" and
        /// "holds everything at half" are the same average and completely different
        /// behaviours.</summary>
        public float FootReleaseFraction => m_Decisions > 0 ? m_ReleaseCount / (m_Decisions * (float)NumFrictionActions) : 0f;

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

        private JointDriveController m_Jd;
        private float[] m_PrevActions;
        private bool m_HasPrevActions;
        private float m_CheckpointZ;
        private Transform[] m_Feet;
        private int m_DecisionPeriod = 1;
        private float m_DecisionSeconds = 0.02f;
        private float m_ForwardSpeedSum;
        private int m_Decisions;
        private int m_AirborneDecisions;
        private float m_FrictionSum;
        private int m_ReleaseCount;
        // One material per foot, owned by THIS agent. See SetupFrictionMaterials.
        private PhysicsMaterial[] m_FootMaterials;
        private PhysicsMaterial m_BodyMaterial;
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

            // MoveRig moves the rig by transforming ONLY `body`, which is exact
            // precisely because every other part hangs off it in the prefab
            // hierarchy. If that stops being true, the legs silently stop following
            // the body on every spawn and respawn -- the agent would be placed with
            // its limbs left behind at the previous position, which reads as a
            // mysteriously terrible policy rather than as a broken reset. Checked
            // once here rather than trusted.
            foreach (var t in new[] { leg0Upper, leg0Lower, leg1Upper, leg1Lower,
                                      leg2Upper, leg2Lower, leg3Upper, leg3Lower })
            {
                if (t != null && !t.IsChildOf(body))
                {
                    Debug.LogError(
                        $"{name}: body part '{t.name}' is not a descendant of '{body.name}'. "
                        + "MoveRig transforms only the root and relies on the hierarchy to "
                        + "carry the rest; re-parenting breaks every spawn and respawn.");
                }
            }

            SetupFrictionMaterials();
        }

        /// <summary>
        /// Gives this agent its own mutable friction on each foot, and pins every
        /// other collider it owns to <see cref="BodyFriction"/>.
        /// </summary>
        /// <remarks>
        /// PER AGENT, AT RUN TIME, and neither half of that is a style choice.
        ///
        /// Per agent: 32 arenas share one prefab, so a material assigned in the
        /// builder is ONE asset behind 32 crawlers. Writing to it from
        /// OnActionReceived would make every arena's feet carry the last friction
        /// any arena commanded -- 31 agents acting on another agent's policy
        /// output, which trains perfectly happily and means nothing. Assigning
        /// `sharedMaterial` a freshly constructed instance is what keeps the
        /// actuator private.
        ///
        /// At run time: a .physicMaterial asset per foot per arena is 128 assets
        /// to keep in step with two constants, and the builder would have to
        /// rewrite them all on every rebuild. The constants live here, next to the
        /// code that reads them, for the same reason ObservationCount does.
        ///
        /// MULTIPLY combine is what makes the command mean something absolute.
        /// Unity resolves a contact with the HIGHER of the two materials' combine
        /// modes, and ParkourGround is Average (0) while this is Multiply (2), so
        /// this side wins every contact against the track: the coefficient is
        /// GroundFriction x the multiplier, on the floor, on a hurdle and on a
        /// beam alike. Under Average the same command would have meant a different
        /// grip on every surface, and under Minimum the top of the range would have
        /// been clamped to the track's own 1.6 and the agent could never grip
        /// harder than the ground it stands on.
        /// </remarks>
        private void SetupFrictionMaterials()
        {
            m_BodyMaterial = new PhysicsMaterial("CrawlerBody")
            {
                dynamicFriction = BodyFriction,
                staticFriction = BodyFriction,
                bounciness = 0f,
                frictionCombine = PhysicsMaterialCombine.Multiply,
                bounceCombine = PhysicsMaterialCombine.Average,
                hideFlags = HideFlags.HideAndDontSave,
            };

            m_FootMaterials = new PhysicsMaterial[m_Feet.Length];
            for (int i = 0; i < m_Feet.Length; i++)
            {
                m_FootMaterials[i] = new PhysicsMaterial($"CrawlerFoot{i}")
                {
                    // Midpoint = run 008's contact exactly, so an agent that has not
                    // learned to use the actuator yet is running the old physics.
                    dynamicFriction = FootFrictionMax * 0.5f,
                    staticFriction = FootFrictionMax * 0.5f,
                    bounciness = 0f,
                    frictionCombine = PhysicsMaterialCombine.Multiply,
                    bounceCombine = PhysicsMaterialCombine.Average,
                    hideFlags = HideFlags.HideAndDontSave,
                };
            }

            // Every collider under the agent gets one or the other. Walking the
            // whole hierarchy rather than the nine body parts on purpose: the
            // example crawler carries decoration with a collider (the sweatband,
            // welded to the torso and therefore part of what scrapes along an
            // overhang), and anything else added later would otherwise silently
            // keep Unity's 0.6 default and be the grippiest surface on the animal.
            int feet = 0, others = 0;
            foreach (var col in GetComponentsInChildren<Collider>(true))
            {
                if (col.isTrigger) continue;
                int footIndex = System.Array.FindIndex(
                    m_Feet, f => f != null && col.transform.IsChildOf(f));
                if (footIndex >= 0)
                {
                    col.sharedMaterial = m_FootMaterials[footIndex];
                    feet++;
                }
                else
                {
                    col.sharedMaterial = m_BodyMaterial;
                    others++;
                }
            }

            // A foot with no collider is a foot whose friction command does nothing,
            // and the run would look like the actuator simply did not help.
            if (feet < m_Feet.Length)
            {
                Debug.LogError(
                    $"{name}: only {feet} foot collider(s) found for {m_Feet.Length} feet. "
                    + "The friction actions on the missing ones are inert -- the policy "
                    + "will be paying an action-rate cost for an actuator that is not "
                    + "connected to anything.");
            }
            Debug.Log($"{name}: friction materials -- {feet} foot collider(s) controllable "
                      + $"in [0, {FootFrictionMax:F3}] x ground, {others} pinned at {BodyFriction}");
        }

        private void OnDestroy()
        {
            if (m_FootMaterials != null)
            {
                foreach (var m in m_FootMaterials)
                {
                    if (m != null) Destroy(m);
                }
            }
            if (m_BodyMaterial != null) Destroy(m_BodyMaterial);
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
            EpisodeAirborneReward = 0f;
            m_ForwardSpeedSum = 0f;
            m_Decisions = 0;
            m_AirborneDecisions = 0;
            m_FrictionSum = 0f;
            m_ReleaseCount = 0;
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
            // Two more slots used to sit here, Clamp01(z / TrackLength) and
            // Clamp01(MaxProgress / TrackLength) -- "how far through the track am
            // I". That is goal-conditioning on a goal run 006 removed, and with a
            // mid-track spawn it is worse than useless: it is a feature the policy
            // can key absolute track position on, which is exactly the dependence
            // "keep moving at any point in any track" is meant to break. Run 006
            // held them at zero rather than deleting them so run 005's checkpoint
            // stayed loadable. Run 009 changes the width of this vector anyway, so
            // they are gone -- see NonGridObservations.
            float halfW = track != null ? track.TrackWidth * 0.5f : 1f;
            float laneX = track != null ? track.LaneCenterAt(tp.z) : 0f;
            sensor.AddObservation(Mathf.Clamp((tp.x - laneX) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp((halfW - tp.x) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp((tp.x + halfW) / 5f, -2f, 2f));

            AddHeightField(sensor);

            // Previous actions, INCLUDING the four friction commands. Those four are
            // the only place the policy can read what its feet are currently doing:
            // friction is a property of a material, not a measurable state, and
            // nothing else in the observation reflects it. Without them the policy
            // would be charged an action-rate cost for changing a quantity it cannot
            // observe.
            if (m_PrevActions != null)
            {
                foreach (var a in m_PrevActions) sensor.AddObservation(a);
            }
            else
            {
                for (int i = 0; i < ActionCount; i++) sensor.AddObservation(0f);
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

            // ONE place, at the top. Four per-episode fractions divide by this
            // (MeanForwardSpeed, AirborneFraction, MeanFootFriction,
            // FootReleaseFraction) and their numerators are accumulated by three
            // different methods further down; incrementing it inside one of those,
            // as the velocity reward used to, makes every other fraction depend on
            // the order the rewards happen to be applied in.
            m_Decisions++;

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

            ApplyFootFriction(c);

            ApplyVelocityReward();
            ApplyAirborneReward();
            ApplyProgressReward();
            ApplyControlCosts(c);
        }

        /// <summary>
        /// Sets each foot's friction from its own action, in [0, FootFrictionMax].
        /// </summary>
        /// <remarks>
        /// This is run 009's answer to the failure mode the rollouts kept showing:
        /// a foot wedged behind a ledge or dropped into a gap, with the policy
        /// unable to retract it because the only way out was to drag it sideways
        /// against 1.1 of grip. Grip is not a property of the terrain to be endured;
        /// it is now a thing the animal decides, per foot, ten times a second. The
        /// same knob is what a leap needs at the other end -- push against the
        /// ground hard, then let go of it.
        ///
        /// Writing dynamicFriction is cheap but not free (PhysX re-reads the
        /// material for existing contacts), so a command that has not moved is not
        /// re-applied. The deadband is a thousandth of the range: far below
        /// anything the policy can act on deliberately, far above float noise.
        /// </remarks>
        private void ApplyFootFriction(ActionSegment<float> c)
        {
            if (m_FootMaterials == null) return;
            const float deadband = FootFrictionMax * 0.001f;

            for (int f = 0; f < m_FootMaterials.Length; f++)
            {
                float a = Mathf.Clamp(c[FirstFrictionAction + f], -1f, 1f);
                float mu = (a + 1f) * 0.5f * FootFrictionMax;

                m_FrictionSum += mu;
                // "Released" = under a tenth of the range. At 0.1375 x 1.6 = 0.22
                // contact the foot slides out of anything it is caught in.
                if (mu < FootFrictionMax * 0.1f) m_ReleaseCount++;

                var mat = m_FootMaterials[f];
                if (mat == null || Mathf.Abs(mat.dynamicFriction - mu) < deadband) continue;
                mat.dynamicFriction = mu;
                mat.staticFriction = mu;
            }
        }

        /// <summary>
        /// Pays per second with every foot clear of the ground, at the same speed
        /// and heading shaping as the velocity term.
        /// </summary>
        /// <remarks>
        /// RUN 009. Run 008 eliminated three explanations for `Gap` staying at
        /// 0.7-3.1% cleared -- not training duration (+0.1 to +0.8 over 33M extra
        /// steps), not actuation (doubling the joint drive moved it by nothing),
        /// not exploration in the crude sense (falls per entry 0.02: the crawler
        /// walks to the edge and stops). What was left is structural. Every
        /// positive term in this environment is a rate against ground covered or
        /// seconds of good locomotion, and a leap is neither: it is a second in
        /// which no foot can push, no new metre is guaranteed, and a failure costs
        /// a respawn plus the walk back over ground the ratchet has already paid
        /// for. Stopping at the edge is not the policy being timid, it is the
        /// policy being right about the reward it was given.
        ///
        /// So the air phase gets a price. Deliberately shaped by the SAME speed
        /// and heading factors as ApplyVelocityReward rather than paid flat, and
        /// that is what closes the obvious exploits:
        ///
        ///   - hopping on the spot earns nothing, because the speed factor is zero
        ///     at zero down-track velocity;
        ///   - so does being launched backwards off a hurdle, for the same reason;
        ///   - and throwing itself off the track to farm airtime earns nothing
        ///     either, because of the guard below: once the body is under the local
        ///     ground line it is falling, not flying, and falling is already priced
        ///     by respawnPenalty.
        ///
        /// What is left that this pays for is a foot-free second spent travelling
        /// forwards -- which is a leap, or the flight phase of a bound. Both are
        /// what the environment exists to produce.
        /// </remarks>
        private void ApplyAirborneReward()
        {
            bool airborne = true;
            for (int i = 0; i < m_Feet.Length; i++)
            {
                var gc = m_Jd.bodyPartsDict[m_Feet[i]].groundContact;
                if (gc != null && gc.touchingGround) { airborne = false; break; }
            }

            // Below the local ground line is a fall in progress, not a leap. Checked
            // against the same height field HasFallen uses, so the two agree about
            // what "under the track" means.
            if (airborne && track != null)
            {
                var tp = TrackPos;
                if (track.GroundHeightAt(tp.x, tp.z, out float g) && tp.y < g) airborne = false;
            }

            if (!airborne) return;
            m_AirborneDecisions++;
            if (airbornePerSecond <= 0f || targetSpeed <= 0f) return;

            float vz = AvgVelocity().z;
            float d = Mathf.Clamp(Mathf.Abs(vz - targetSpeed), 0f, targetSpeed) / targetSpeed;
            float speed = (1f - d * d) * (1f - d * d);
            float heading = (body.forward.z + 1f) * 0.5f;

            float r = airbornePerSecond * m_DecisionSeconds * speed * heading;
            AddReward(r);
            EpisodeAirborneReward += r;
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
                // The strength block, by INDEX, not by "the last eight".
                //
                // It was `n - k_NumStrengthActions` while the strengths happened to
                // be the tail of the vector. Run 009 appends four friction commands
                // after them, so that expression would have averaged four strengths
                // and four frictions and called the result energy -- charging the
                // agent for gripping and refunding it for letting go, at the same
                // weight as joint torque, with nothing in any metric to show for it.
                int start = Mathf.Clamp(FirstStrengthAction, 0, n);
                int end = Mathf.Clamp(FirstStrengthAction + NumStrengthActions, start, n);
                float energy = 0f;
                for (int a = start; a < end; a++)
                {
                    float s = (c[a] + 1f) * 0.5f;
                    energy += s * s;
                }
                energy = (end - start) > 0 ? energy / (end - start) : 0f;

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
            MoveRig(target, 0f, Vector3.zero);
        }

        /// <summary>
        /// Rigid placement of the whole body at a point and heading, with a given
        /// starting velocity. Used at episode start; respawns use
        /// <see cref="TeleportTo"/> instead, because a fall should not also re-square
        /// the animal to the track.
        /// </summary>
        private void PlaceAt(Vector3 target, float yawDegrees, Vector3 velocity)
        {
            MoveRig(target, yawDegrees, velocity);
        }

        /// <summary>
        /// Moves the whole crawler to a world position and heading, rigidly, by
        /// transforming ONLY the root body.
        /// </summary>
        /// <remarks>
        /// THE PARTS ARE NESTED, and this is the whole reason this method exists.
        /// The prefab hierarchy is Body -> legN -> forelegN, so every one of the
        /// other eight parts is a descendant of <c>body</c>. `transform.position`
        /// and `transform.rotation` are WORLD values derived through the parent
        /// chain, so writing Body's already moves all eight descendants with it.
        ///
        /// Both this and the respawn path used to loop over every part writing
        /// `tr.position = target + rot * (tr.position - pivot)`. That reads a value
        /// a previous write in the same loop had already displaced, so the offset
        /// landed twice on a leg and three times on a foreleg:
        ///
        ///     Body      P + delta
        ///     legN      P + 2*delta
        ///     forelegN  P + 3*delta
        ///
        /// At a mid-track spawn `delta` is the distance from the recorded rest pose
        /// to the spawn point -- up to ~112m on a 24-segment track. The rig was
        /// therefore torn tens of metres apart on every episode reset, the
        /// ConfigurableJoints were violated by that much, and the solver answered
        /// with an enormous corrective impulse: the crawler was launched off the
        /// track spinning, respawned, and launched again. The yaw rotation
        /// compounded the same way.
        ///
        /// Moving only the root is exact rather than approximately right: every
        /// relative transform, and therefore every joint constraint, is preserved
        /// by construction, and there is no read-after-write to get wrong. It also
        /// makes TeleportTo actually do what its comment always claimed -- preserve
        /// the pose a fall left the agent in.
        /// </remarks>
        private void MoveRig(Vector3 target, float yawDegrees, Vector3 velocity)
        {
            SnapshotRig();

            if (yawDegrees != 0f)
            {
                // About Body's own pivot, so this changes heading without moving it.
                body.rotation = Quaternion.AngleAxis(yawDegrees, Vector3.up) * body.rotation;
            }
            body.position = target;

            foreach (var bp in m_Jd.bodyPartsDict.Values)
            {
                bp.rb.linearVelocity = velocity;
                bp.rb.angularVelocity = Vector3.zero;
            }

            AssertRigIntact();
        }

        // Body-local positions of every part, taken immediately before a MoveRig and
        // compared immediately after. See AssertRigIntact.
        private Vector3[] m_RigSnapshot;
        private bool m_RigBreakReported;

        private void SnapshotRig()
        {
            var parts = m_Jd.bodyPartsDict.Values;
            if (m_RigSnapshot == null || m_RigSnapshot.Length != parts.Count)
            {
                m_RigSnapshot = new Vector3[parts.Count];
            }
            int i = 0;
            foreach (var bp in parts)
            {
                m_RigSnapshot[i++] = body.InverseTransformPoint(bp.rb.position);
            }
        }

        /// <summary>
        /// A rigid move must not change any part's position RELATIVE TO THE BODY.
        /// Checked, not assumed.
        /// </summary>
        /// <remarks>
        /// This is the assertion whose absence cost run 006. Every pre-launch check
        /// that run had was on GEOMETRY -- that generated tracks are traversable, that
        /// spawn points have clear ground and headroom -- and none was on the AGENT
        /// arriving there in one piece. The reset had been tearing the rig apart since
        /// the environment was written; it stayed invisible because respawns move the
        /// agent a few metres and a rig stretched by a few metres reads as ordinary
        /// contact noise. Mid-track spawning multiplied the same defect by fifty and
        /// burned a full 24 h run before anyone watched a video.
        ///
        /// Body-local rather than world offsets is what makes this correct for
        /// TeleportTo as well: a respawn deliberately preserves whatever pose the fall
        /// left, so the offsets differ from episode to episode and only their
        /// invariance ACROSS THE MOVE is the actual contract. The tolerance is loose
        /// enough that float error in a transform round-trip cannot trip it and tight
        /// enough that the run 006 bug -- which displaced parts by metres -- could not
        /// have hidden under it.
        ///
        /// Reported once per agent. A torn rig recurs every episode, and thousands of
        /// identical errors would bury the first one.
        /// </remarks>
        private void AssertRigIntact()
        {
            const float tolerance = 0.02f;
            if (m_RigBreakReported || m_RigSnapshot == null) { return; }

            int i = 0;
            float worst = 0f;
            string worstName = null;
            foreach (var bp in m_Jd.bodyPartsDict.Values)
            {
                float d = Vector3.Distance(body.InverseTransformPoint(bp.rb.position),
                                           m_RigSnapshot[i++]);
                if (d > worst) { worst = d; worstName = bp.rb.name; }
            }
            if (worst > tolerance)
            {
                m_RigBreakReported = true;
                Debug.LogError(
                    $"{name}: RESET TORE THE RIG. Part '{worstName}' moved {worst:F2} m "
                    + "relative to the body during a rigid move, which must be zero. "
                    + "The joints are now violated by that much and the solver will "
                    + "answer with a corrective impulse. Do not trust any distance or "
                    + "fall metric from this run.");
            }
        }
    }
}
