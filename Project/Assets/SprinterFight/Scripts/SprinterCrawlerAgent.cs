using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;

/// <summary>
/// Physics-based crawler agent adapted for head-to-head racing.
/// Observations are concatenated: own features first, opponent features second.
/// Action space matches the original CrawlerAgent (16 continuous actions).
/// </summary>
[RequireComponent(typeof(JointDriveController))]
public class SprinterCrawlerAgent : Agent
{
    [Header("Body Parts")] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    [Header("Foot Grounded Visualization")] public bool useFootGroundedVisualization;
    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    [Header("Opponent Reference (set by controller)")]
    public SprinterCrawlerAgent opponent;

    [Header("Track Settings (set by controller)")]
    public float trackStartX = 0f;
    public float trackFinishX = 105f;
    public float trackZHalfWidth = 14.5f;

    private JointDriveController m_JdController;

    public override void Initialize()
    {
        m_JdController = GetComponent<JointDriveController>();

        // CLEAR ANY EXISTING BODY PARTS FIRST - handles weird error with env replicator
        m_JdController.bodyPartsDict.Clear();
        m_JdController.bodyPartsList.Clear();

        // Setup each body part with the joint drive controller
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Lower);
        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Lower);
        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Lower);
        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Lower);

        // Disable per-agent max step; environment controller will manage episode length
        MaxStep = 0;
    }

    public void SetOpponent(SprinterCrawlerAgent other)
    {
        opponent = other;
    }

    public void SetTrack(float startX, float finishX, float zHalfWidth)
    {
        trackStartX = startX;
        trackFinishX = finishX;
        trackZHalfWidth = zHalfWidth;
    }

    public override void OnEpisodeBegin()
    {
        // Reset all body parts to their recorded initial conditions
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
        // Face roughly along +X to reduce exploration burden
        var euler = body.rotation.eulerAngles;
        body.rotation = Quaternion.Euler(0f, 90f, 0f); // +X forward
    }

    /// <summary>
    /// Add own features first, then opponent summary features.
    /// </summary>
    /// <param name="sensor"></param>
    public override void CollectObservations(VectorSensor sensor)
    {
        // Own features
        var avgVel = GetAvgVelocity();
        var forwardAxis = Vector3.right; // racing down +X

        // Progress along track (normalized 0..1)
        float progressNorm = GetNormalizedProgressX();
        sensor.AddObservation(progressNorm);

        // Lateral offset within track (-1..1 when within bounds)
        float lateralNorm = Mathf.Clamp(body.position.z / trackZHalfWidth, -1f, 1f);
        sensor.AddObservation(lateralNorm);

        // Average velocity (in body/stable frame not required; use world for simplicity)
        sensor.AddObservation(Vector3.Dot(avgVel, forwardAxis)); // speed along +X
        sensor.AddObservation(avgVel.y);
        sensor.AddObservation(Vector3.Dot(avgVel, Vector3.forward)); // lateral speed

        // Heading alignment with +X
        sensor.AddObservation(Vector3.Dot(body.forward, forwardAxis));

        // Raycast down for ground proximity
        float maxRaycastDist = 10f;
        if (Physics.Raycast(body.position, Vector3.down, out var hit, maxRaycastDist))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
        {
            sensor.AddObservation(1f);
        }

        // Per-limb ground contact and normalized strength
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            sensor.AddObservation(bodyPart.groundContact.touchingGround);
            if (bodyPart.rb.transform != body)
            {
                sensor.AddObservation(bodyPart.currentStrength / m_JdController.maxJointForceLimit);
            }
        }

        // Opponent summary features (if present)
        if (opponent != null)
        {
            var opponentAvgVel = opponent.GetAvgVelocity();
            var relPos = opponent.body.position - body.position;

            // Relative position normalized
            sensor.AddObservation(Mathf.Clamp(relPos.x / Mathf.Max(1f, (trackFinishX - trackStartX)), -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(relPos.z / Mathf.Max(1f, trackZHalfWidth), -1f, 1f));

            // Opponent speed along +X and lateral
            sensor.AddObservation(Vector3.Dot(opponentAvgVel, forwardAxis));
            sensor.AddObservation(Vector3.Dot(opponentAvgVel, Vector3.forward));

            // Opponent heading alignment
            sensor.AddObservation(Vector3.Dot(opponent.body.forward, forwardAxis));

            // Opponent progress
            sensor.AddObservation(opponent.GetNormalizedProgressX());
        }
        else
        {
            // Fill with zeros to keep observation size constant
            sensor.AddObservation(0f); // rel x
            sensor.AddObservation(0f); // rel z
            sensor.AddObservation(0f); // opp speed x
            sensor.AddObservation(0f); // opp speed z
            sensor.AddObservation(0f); // opp heading
            sensor.AddObservation(0f); // opp progress
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var continuous = actionBuffers.ContinuousActions;
        int i = -1;

        // Target rotations
        bpDict[leg0Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuous[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuous[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuous[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuous[++i], 0, 0);

        // Strengths
        bpDict[leg0Upper].SetJointStrength(continuous[++i]);
        bpDict[leg1Upper].SetJointStrength(continuous[++i]);
        bpDict[leg2Upper].SetJointStrength(continuous[++i]);
        bpDict[leg3Upper].SetJointStrength(continuous[++i]);
        bpDict[leg0Lower].SetJointStrength(continuous[++i]);
        bpDict[leg1Lower].SetJointStrength(continuous[++i]);
        bpDict[leg2Lower].SetJointStrength(continuous[++i]);
        bpDict[leg3Lower].SetJointStrength(continuous[++i]);
    }

    private void FixedUpdate()
    {
        if (!useFootGroundedVisualization) return;

        // Optional visualization of grounded feet
        foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
        foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
        foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
        foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround ? groundedMaterial : unGroundedMaterial;
    }

    public Vector3 GetAvgVelocity()
    {
        Vector3 velocitySum = Vector3.zero;
        int numBodies = 0;
        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            numBodies++;
            velocitySum += bodyPart.rb.linearVelocity;
        }
        if (numBodies == 0) return Vector3.zero;
        return velocitySum / numBodies;
    }

    public float GetNormalizedProgressX()
    {
        float len = Mathf.Max(1f, (trackFinishX - trackStartX));
        return Mathf.Clamp01((body.position.x - trackStartX) / len);
    }

    public bool IsBodyTouchingGround()
    {
        // True if the main body segment is in contact with the ground
        if (m_JdController == null) return false;
        if (!m_JdController.bodyPartsDict.ContainsKey(body)) return false;
        return m_JdController.bodyPartsDict[body].groundContact.touchingGround;
    }
}


