using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgentsExamples;

/// <summary>
/// Physics-based crawler agent adapted for sumo-style competition on a circular platform.
/// Observations include own state, opponent state, and platform-relative positioning.
/// Action space matches the original CrawlerAgent (16 continuous actions).
/// </summary>
[RequireComponent(typeof(JointDriveController))]
public class CrawlerSumoAgent : Agent
{
    [Header("Body Parts")] 
    public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    [Header("Foot Grounded Visualization")] 
    public bool useFootGroundedVisualization;
    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    [Header("Opponent Reference (set by controller)")]
    public CrawlerSumoAgent opponent;

    [Header("Platform Settings (set by controller)")]
    public Vector3 platformCenter = Vector3.zero;
    public float platformRadius = 15f;

    // Run 013 control costs. Weights are pushed in by the env controller from
    // environment_parameters; 0 disables (runs 006-012 behaviour exactly).
    [System.NonSerialized] public float energyCostWeight;
    [System.NonSerialized] public float actionRateCostWeight;

    // Per-episode totals, read by the env controller for the reward breakdown
    // stats just before EndEpisode(). Reported as positive magnitudes.
    // Disc = weighted by gamma^(decision index), i.e. the term's contribution to
    // V(s_0) rather than its raw mass. Control costs accrue once per decision,
    // so the agent tracks its own discount exactly.
    public float EpisodeEnergyCost { get; private set; }
    public float EpisodeActionRateCost { get; private set; }
    public float EpisodeEnergyCostDisc { get; private set; }
    public float EpisodeActionRateCostDisc { get; private set; }

    // Diagnostics only; mirrors reward_signals.extrinsic.gamma. Set by controller.
    [System.NonSerialized] public float statsGamma = 1f;
    private float m_Discount = 1f;

    // Actions 12-19 are per-joint strength (see OnActionReceived).
    private const int k_NumStrengthActions = 8;
    private float[] m_PrevActions;
    private bool m_HasPrevActions;

    private JointDriveController m_JdController;

    public override void Initialize()
    {
        m_JdController = GetComponent<JointDriveController>();

        // Clear any existing body parts first
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

    public void SetOpponent(CrawlerSumoAgent other)
    {
        opponent = other;
    }

    public void SetPlatform(Vector3 center, float radius)
    {
        platformCenter = center;
        platformRadius = radius;
    }

    public override void OnEpisodeBegin()
    {
        // Reset all body parts to their recorded initial conditions
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        // No action-rate cost across an episode boundary: the first decision of
        // a new episode has no meaningful predecessor.
        m_HasPrevActions = false;
        EpisodeEnergyCost = 0f;
        EpisodeActionRateCost = 0f;
        EpisodeEnergyCostDisc = 0f;
        EpisodeActionRateCostDisc = 0f;
        m_Discount = 1f;
    }

    /// <summary>
    /// Collect observations for sumo competition:
    /// - Own state relative to platform
    /// - Opponent state relative to platform and self
    /// - Platform geometry information
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        // === OWN STATE ===
        var avgVel = GetAvgVelocity();
        Vector3 bodyPos = body.position;
        
        // Distance from platform center (normalized by radius)
        Vector2 bodyPos2D = new Vector2(bodyPos.x, bodyPos.z);
        Vector2 platformCenter2D = new Vector2(platformCenter.x, platformCenter.z);
        float distFromCenter = Vector2.Distance(bodyPos2D, platformCenter2D);
        float normalizedDistFromCenter = Mathf.Clamp01(distFromCenter / platformRadius);
        sensor.AddObservation(normalizedDistFromCenter);
        
        // Direction to platform center (normalized)
        Vector3 dirToCenter = (platformCenter - bodyPos).normalized;
        sensor.AddObservation(dirToCenter.x);
        sensor.AddObservation(dirToCenter.z);
        
        // Whether on platform (binary)
        float onPlatform = distFromCenter <= platformRadius ? 1f : 0f;
        sensor.AddObservation(onPlatform);
        
        // Body orientation relative to platform center
        Vector3 bodyForward = body.forward;
        float headingToCenter = Vector3.Dot(bodyForward, dirToCenter);
        sensor.AddObservation(headingToCenter);
        
        // Body stability (how upright)
        float bodyUpright = Vector3.Dot(body.up, Vector3.up);
        sensor.AddObservation(bodyUpright);
        
        // Average velocity components
        sensor.AddObservation(avgVel.x);
        sensor.AddObservation(avgVel.y);
        sensor.AddObservation(avgVel.z);
        
        // Velocity toward/away from center
        float velTowardCenter = Vector3.Dot(avgVel, dirToCenter);
        sensor.AddObservation(velTowardCenter);
        
        // Raycast down for ground proximity
        float maxRaycastDist = 10f;
        if (Physics.Raycast(bodyPos, Vector3.down, out var hit, maxRaycastDist))
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

        // === OPPONENT STATE ===
        if (opponent != null)
        {
            var opponentAvgVel = opponent.GetAvgVelocity();
            Vector3 opponentPos = opponent.body.position;
            
            // Relative position to opponent
            Vector3 relPos = opponentPos - bodyPos;
            float relDistance = relPos.magnitude;
            Vector3 relPosNorm = relDistance > 0.01f ? relPos / relDistance : Vector3.zero;
            
            sensor.AddObservation(relPosNorm.x);
            sensor.AddObservation(relPosNorm.z);
            sensor.AddObservation(Mathf.Clamp01(relDistance / (platformRadius * 2f))); // normalized distance
            
            // Opponent distance from center
            Vector2 opponentPos2D = new Vector2(opponentPos.x, opponentPos.z);
            float opponentDistFromCenter = Vector2.Distance(opponentPos2D, platformCenter2D);
            float opponentNormalizedDistFromCenter = Mathf.Clamp01(opponentDistFromCenter / platformRadius);
            sensor.AddObservation(opponentNormalizedDistFromCenter);
            
            // Opponent on platform
            float opponentOnPlatform = opponentDistFromCenter <= platformRadius ? 1f : 0f;
            sensor.AddObservation(opponentOnPlatform);
            
            // Opponent velocity components
            sensor.AddObservation(opponentAvgVel.x);
            sensor.AddObservation(opponentAvgVel.y);
            sensor.AddObservation(opponentAvgVel.z);
            
            // Opponent velocity toward/away from center
            Vector3 opponentDirToCenter = (platformCenter - opponentPos).normalized;
            float opponentVelTowardCenter = Vector3.Dot(opponentAvgVel, opponentDirToCenter);
            sensor.AddObservation(opponentVelTowardCenter);
            
            // Opponent body orientation
            float opponentUpright = Vector3.Dot(opponent.body.up, Vector3.up);
            sensor.AddObservation(opponentUpright);
            
            // Relative velocity (opponent velocity relative to self)
            Vector3 relVel = opponentAvgVel - avgVel;
            sensor.AddObservation(relVel.x);
            sensor.AddObservation(relVel.z);
        }
        else
        {
            // Fill with zeros to keep observation size constant (11 opponent observations)
            for (int i = 0; i < 11; i++)
            {
                sensor.AddObservation(0f);
            }
        }
        
        // === PLATFORM GEOMETRY ===
        // Platform radius (normalized, though constant in this case)
        sensor.AddObservation(platformRadius / 20f); // assuming max radius of 20
        
        // Edge proximity - how close to falling off
        float edgeDistance = platformRadius - distFromCenter;
        float normalizedEdgeDistance = Mathf.Clamp01(edgeDistance / platformRadius);
        sensor.AddObservation(normalizedEdgeDistance);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var bpDict = m_JdController.bodyPartsDict;
        var continuous = actionBuffers.ContinuousActions;
        int i = -1;

        // Target rotations (same as original crawler)
        bpDict[leg0Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuous[++i], continuous[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuous[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuous[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuous[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuous[++i], 0, 0);

        // Strengths (same as original crawler)
        bpDict[leg0Upper].SetJointStrength(continuous[++i]);
        bpDict[leg1Upper].SetJointStrength(continuous[++i]);
        bpDict[leg2Upper].SetJointStrength(continuous[++i]);
        bpDict[leg3Upper].SetJointStrength(continuous[++i]);
        bpDict[leg0Lower].SetJointStrength(continuous[++i]);
        bpDict[leg1Lower].SetJointStrength(continuous[++i]);
        bpDict[leg2Lower].SetJointStrength(continuous[++i]);
        bpDict[leg3Lower].SetJointStrength(continuous[++i]);

        ApplyControlCosts(continuous);
    }

    /// <summary>
    /// Run 013: metabolic + action-rate costs.
    ///
    /// Deliberately NOT zero-sum, unlike every other term in this environment.
    /// A relative cost cancels when both agents share a habit, which is exactly
    /// the equilibrium we are trying to break -- mutual jitter would go unpriced.
    /// Both crawlers pay the identical cost function, so competitive fairness
    /// (the reason the other terms are zero-sum) is preserved.
    ///
    /// Why energy matters here: actions 12-19 feed SetJointStrength, which maps
    /// a -> (a+1)/2 of maxJointForceLimit. With no cost, maximum stiffness is
    /// strictly dominant, so those 8 dims saturate and become dead -- control
    /// collapses to bang-bang target rotations against rigid PD joints, which is
    /// what produced runs 006-012's jerking. Pricing strength makes stiffness a
    /// real decision (stiffen to drive, relax to swing a limb).
    /// </summary>
    private void ApplyControlCosts(ActionSegment<float> continuous)
    {
        int n = continuous.Length;
        if (m_PrevActions == null || m_PrevActions.Length != n)
        {
            m_PrevActions = new float[n];
            m_HasPrevActions = false;
        }

        if (energyCostWeight > 0f || actionRateCostWeight > 0f)
        {
            // Energy: mean squared normalized joint strength over the strength dims.
            float energy = 0f;
            int strengthStart = Mathf.Max(0, n - k_NumStrengthActions);
            int strengthCount = n - strengthStart;
            for (int a = strengthStart; a < n; a++)
            {
                float normStrength = (continuous[a] + 1f) * 0.5f; // -> [0, 1]
                energy += normStrength * normStrength;
            }
            energy = strengthCount > 0 ? energy / strengthCount : 0f;

            // Action rate: mean squared change since the previous decision.
            // DecisionRequester has TakeActionsBetweenDecisions = 0, so
            // OnActionReceived fires once per decision -- these are genuine
            // consecutive decisions, not repeated frames of the same action.
            float rate = 0f;
            if (m_HasPrevActions)
            {
                for (int a = 0; a < n; a++)
                {
                    float d = continuous[a] - m_PrevActions[a];
                    rate += d * d;
                }
                rate /= n;
            }

            float energyCost = energyCostWeight * energy;
            float rateCost = actionRateCostWeight * rate;
            EpisodeEnergyCost += energyCost;
            EpisodeActionRateCost += rateCost;
            EpisodeEnergyCostDisc += energyCost * m_Discount;
            EpisodeActionRateCostDisc += rateCost * m_Discount;
            AddReward(-(energyCost + rateCost));
        }

        // Advance after applying: the first decision of an episode is index 0
        // and must carry discount 1.
        m_Discount *= statsGamma;

        for (int a = 0; a < n; a++)
        {
            m_PrevActions[a] = continuous[a];
        }
        m_HasPrevActions = true;
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

    public float GetDistanceFromPlatformCenter()
    {
        Vector2 bodyPos2D = new Vector2(body.position.x, body.position.z);
        Vector2 platformCenter2D = new Vector2(platformCenter.x, platformCenter.z);
        return Vector2.Distance(bodyPos2D, platformCenter2D);
    }

    public bool IsOnPlatform()
    {
        return GetDistanceFromPlatformCenter() <= platformRadius;
    }

    public bool IsBodyTouchingGround()
    {
        // True if the main body segment is in contact with the ground
        //if (m_JdController == null) return false;
        //if (!m_JdController.bodyPartsDict.ContainsKey(body)) return false;
        return m_JdController.bodyPartsDict[body].groundContact.touchingGround;
    }
}
