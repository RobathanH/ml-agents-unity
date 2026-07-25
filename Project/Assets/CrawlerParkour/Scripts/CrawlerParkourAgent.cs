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
        [System.NonSerialized] public float progressWeight = 1f;
        [System.NonSerialized] public float energyCostWeight;
        [System.NonSerialized] public float actionRateCostWeight;
        [System.NonSerialized] public float respawnPenalty;
        [System.NonSerialized] public float finishBonus = 5f;

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
        /// </remarks>
        public const int NonGridObservations = 6 + 6 + 1 + 4 + 5 + 20;

        /// <summary>Total vector observation size, for BehaviorParameters.</summary>
        public int ObservationCount => NonGridObservations + 2 * GridForward * GridLateral;

        public float MaxProgress { get; private set; }
        public int RespawnCount { get; private set; }
        public bool Finished { get; private set; }
        public float EpisodeEnergyCost { get; private set; }
        public float EpisodeActionRateCost { get; private set; }

        private const int k_NumStrengthActions = 8;
        private JointDriveController m_Jd;
        private float[] m_PrevActions;
        private bool m_HasPrevActions;
        private float m_CheckpointZ;
        private Transform[] m_Feet;

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

        public override void OnEpisodeBegin()
        {
            foreach (var bp in m_Jd.bodyPartsDict.Values) bp.Reset(bp);
            TeleportTo(SpawnPoint);
            MaxProgress = 0f;
            RespawnCount = 0;
            Finished = false;
            m_CheckpointZ = 0f;
            m_HasPrevActions = false;
            EpisodeEnergyCost = 0f;
            EpisodeActionRateCost = 0f;
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

            // Track-relative situation.
            float len = track != null ? track.TrackLength : 1f;
            float halfW = track != null ? track.TrackWidth * 0.5f : 1f;
            float laneX = track != null ? track.LaneCenterAt(tp.z) : 0f;
            sensor.AddObservation(Mathf.Clamp((tp.x - laneX) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp((halfW - tp.x) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp((tp.x + halfW) / 5f, -2f, 2f));
            sensor.AddObservation(Mathf.Clamp01(tp.z / len));
            sensor.AddObservation(Mathf.Clamp01(MaxProgress / len));

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

            ApplyProgressReward();
            ApplyControlCosts(c);
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
            if (Finished || track == null) return;
            float z = TrackPos.z;
            if (z > MaxProgress)
            {
                AddReward(progressWeight * (z - MaxProgress) / track.TrackLength);
                MaxProgress = z;
                // Checkpoint at each segment boundary cleared.
                float seg = track.SegmentLength;
                m_CheckpointZ = Mathf.Floor(MaxProgress / seg) * seg;
            }
            if (MaxProgress >= track.TrackLength - 1f)
            {
                Finished = true;
                float timeLeft = StepCount > 0 ? 1f - Mathf.Clamp01(StepCount / (float)Mathf.Max(1, MaxStepOverride)) : 0f;
                AddReward(finishBonus * (0.5f + 0.5f * timeLeft));
            }
        }

        /// <summary>Episode length the controller is enforcing; used to scale the
        /// finish bonus so finishing sooner is strictly better.</summary>
        [System.NonSerialized] public int MaxStepOverride = 3000;

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
    }
}
