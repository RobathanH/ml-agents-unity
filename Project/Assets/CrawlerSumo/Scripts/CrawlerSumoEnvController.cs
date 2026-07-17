using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgentsExamples;

/// <summary>
/// Coordinates two CrawlerSumoAgents in a sumo-style competition on a circular platform.
/// Agents spawn at random positions and compete to push each other off the platform.
/// All rewards are zero-sum to ensure competitive balance.
/// </summary>
public class CrawlerSumoEnvController : MonoBehaviour
{
    [Header("Agents")] 
    public CrawlerSumoAgent crawler1;
    public CrawlerSumoAgent crawler2;

    [Header("Platform Settings (Scene Configuration)")] 
    public Vector3 platformCenter = Vector3.zero;
    public float platformRadius = 15f;
    public float fallY = -5f;

    [Header("Environment Parameters (configured via YAML)")]
    [Tooltip("These values are overridden by environment_parameters in the config file")]
    private float minSpawnDistanceProportion = 0.1f; // Proportion of platform radius (0.2 = 20% of radius)
    private float maxSpawnDistanceProportion = 0.3f; // Proportion of platform radius (0.53 = 53% of radius)
    // public so RolloutViewer's eval mode can match the training config's value
    public int maxEpisodeSteps = 3000;

    // Shrinking ring: from ringShrinkStartStep the effective platform radius
    // lerps toward platformRadius * ringShrinkEndProportion by episode end.
    // An agent outside the effective radius counts as fallen even if standing.
    // Forces decisive outcomes and breaks mutual-turtling equilibria (runs
    // 006-008 all converged to draw-camping). 1.0 disables the shrink.
    private int ringShrinkStartStep = 0;
    private float ringShrinkEndProportion = 1f;

    // Actuator strength: per-episode multiplier sampled from [min, max], applied
    // identically to BOTH crawlers (asymmetric strength would break zero-sum
    // fairness and poison self-play ELO). 1.0/1.0 disables.
    private float jointStrengthMultMin = 1f;
    private float jointStrengthMultMax = 1f;

    // Platform friction: per-episode static+dynamic friction sampled from
    // [min, max] on an instanced physic material (default combine mode averages
    // with the crawlers' default 0.6). min < 0 disables.
    private float platformFrictionMin = -1f;
    private float platformFrictionMax = -1f;

    // Flip knockdown: a crawler whose body.up dot world-up stays below
    // flipKnockdownDot for flipKnockdownSteps consecutive physics steps counts
    // as fallen (loses). Makes flipping the opponent a direct win condition.
    // 0 steps disables.
    private int flipKnockdownSteps = 0;
    private float flipKnockdownDot = -0.2f;

    private float survivalReward = 0.01f;
    private float centerControlReward = 0.005f;
    private float pushingReward = 0.01f;
    private float winReward = 5.0f;
    private float bodyGroundPenalty = 0.005f;
    private float stabilityReward = 0.002f;

    private int m_StepCount;
    private float m_C1StartY;
    private float m_C2StartY;
    private Vector3 m_C1LastPosition;
    private Vector3 m_C2LastPosition;

    private JointDriveController m_C1Jd;
    private JointDriveController m_C2Jd;
    private float m_BaseMaxJointSpring;
    private float m_BaseMaxJointForceLimit;
    private PhysicsMaterial m_PlatformMaterial;
    private int m_C1FlipSteps;
    private int m_C2FlipSteps;
    private bool m_LastEndWasFlip;

    private StatsRecorder m_Recorder;
    private int m_CurrentLearningTeam = -1;
    private bool m_LearningTeamInitialized = false;

    private void Awake()
    {
        // Load environment parameters from config
        LoadEnvironmentParameters();
        
        // Wire opponents and platform settings
        crawler1.SetOpponent(crawler2);
        crawler2.SetOpponent(crawler1);
        crawler1.SetPlatform(platformCenter, platformRadius);
        crawler2.SetPlatform(platformCenter, platformRadius);

        m_C1StartY = crawler1.body.position.y;
        m_C2StartY = crawler2.body.position.y;

        m_Recorder = Academy.Instance.StatsRecorder;
        
        // Initialize learning team detection
        InitializeLearningTeamDetection();
    }

    private void LoadEnvironmentParameters()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        
        // Spawn settings (proportional to platform radius)
        minSpawnDistanceProportion = envParams.GetWithDefault("min_spawn_distance_proportion", minSpawnDistanceProportion);
        maxSpawnDistanceProportion = envParams.GetWithDefault("max_spawn_distance_proportion", maxSpawnDistanceProportion);
        
        // Episode settings
        maxEpisodeSteps = Mathf.RoundToInt(envParams.GetWithDefault("max_episode_steps", maxEpisodeSteps));
        
        // Reward weights
        survivalReward = envParams.GetWithDefault("survival_reward", survivalReward);
        centerControlReward = envParams.GetWithDefault("center_control_reward", centerControlReward);
        pushingReward = envParams.GetWithDefault("pushing_reward", pushingReward);
        winReward = envParams.GetWithDefault("win_reward", winReward);
        bodyGroundPenalty = envParams.GetWithDefault("body_ground_penalty", bodyGroundPenalty);
        stabilityReward = envParams.GetWithDefault("stability_reward", stabilityReward);

        // Shrinking ring
        ringShrinkStartStep = Mathf.RoundToInt(envParams.GetWithDefault("ring_shrink_start_step", ringShrinkStartStep));
        ringShrinkEndProportion = envParams.GetWithDefault("ring_shrink_end_proportion", ringShrinkEndProportion);

        // Actuator strength randomization
        jointStrengthMultMin = envParams.GetWithDefault("joint_strength_multiplier_min", jointStrengthMultMin);
        jointStrengthMultMax = envParams.GetWithDefault("joint_strength_multiplier_max", jointStrengthMultMax);

        // Platform friction randomization
        platformFrictionMin = envParams.GetWithDefault("platform_friction_min", platformFrictionMin);
        platformFrictionMax = envParams.GetWithDefault("platform_friction_max", platformFrictionMax);

        // Flip knockdown
        flipKnockdownSteps = Mathf.RoundToInt(envParams.GetWithDefault("flip_knockdown_steps", flipKnockdownSteps));
        flipKnockdownDot = envParams.GetWithDefault("flip_knockdown_dot", flipKnockdownDot);
        
        Debug.Log($"CrawlerSumo: Loaded environment parameters - Platform Radius: {platformRadius} (scene), " +
                  $"Spawn Range: {minSpawnDistanceProportion * platformRadius:F1}-{maxSpawnDistanceProportion * platformRadius:F1}, " +
                  $"Win Reward: {winReward}");
    }

    private void Start()
    {
        InitPhysicsRandomization();
        ResetSumo();
    }

    /// <summary>
    /// Cache joint-drive baselines and locate this arena's platform collider.
    /// Runs in Start (after all Awakes) so the physics scene is fully populated.
    /// </summary>
    private void InitPhysicsRandomization()
    {
        m_C1Jd = crawler1.GetComponent<JointDriveController>();
        m_C2Jd = crawler2.GetComponent<JointDriveController>();
        m_BaseMaxJointSpring = m_C1Jd.maxJointSpring;
        m_BaseMaxJointForceLimit = m_C1Jd.maxJointForceLimit;

        if (platformFrictionMin < 0f)
        {
            return;
        }
        // Find the platform collider by raycasting down at the arena center,
        // ignoring anything that is part of a crawler. Avoids prefab rewiring
        // and works for multi-arena clones (each arena hits its own platform).
        var hits = Physics.RaycastAll(platformCenter + Vector3.up * 5f, Vector3.down, 30f);
        Collider platform = null;
        float best = float.MaxValue;
        foreach (var hit in hits)
        {
            if (hit.collider.GetComponentInParent<CrawlerSumoAgent>() != null) continue;
            if (hit.distance < best)
            {
                best = hit.distance;
                platform = hit.collider;
            }
        }
        if (platform == null)
        {
            Debug.LogWarning("CrawlerSumo: platform collider not found; friction randomization disabled");
            platformFrictionMin = -1f;
            return;
        }
        m_PlatformMaterial = new PhysicsMaterial("SumoPlatformRandomized");
        platform.sharedMaterial = m_PlatformMaterial;
    }

    private void FixedUpdate()
    {
        m_StepCount++;

        // Early termination checks
        if (CheckFallConditions())
        {
            return;
        }

        // Update learning team detection
        UpdateLearningTeam();
        
        // Apply zero-sum rewards
        ApplyZeroSumRewards();

        // Max steps termination
        if (maxEpisodeSteps > 0 && m_StepCount >= maxEpisodeSteps)
        {
            EndBothEpisodes(0f, 0f); // Draw - no terminal rewards
            ResetSumo();
        }
    }

    private void ApplyZeroSumRewards()
    {
        Vector3 c1Pos = crawler1.body.position;
        Vector3 c2Pos = crawler2.body.position;
        
        // Distance from platform center (2D distance, ignoring Y)
        float c1DistFromCenter = Vector2.Distance(new Vector2(c1Pos.x, c1Pos.z), 
                                                 new Vector2(platformCenter.x, platformCenter.z));
        float c2DistFromCenter = Vector2.Distance(new Vector2(c2Pos.x, c2Pos.z), 
                                                 new Vector2(platformCenter.x, platformCenter.z));

        // Survival reward - zero-sum based on relative platform position
        float c1OnPlatform = c1DistFromCenter <= platformRadius ? 1f : 0f;
        float c2OnPlatform = c2DistFromCenter <= platformRadius ? 1f : 0f;
        
        if (c1OnPlatform > 0f && c2OnPlatform > 0f)
        {
            // Both on platform - no survival bonus
            // This prevents reward farming by both agents staying safe
        }
        else if (c1OnPlatform > 0f && c2OnPlatform == 0f)
        {
            crawler1.AddReward(survivalReward);
            crawler2.AddReward(-survivalReward);
        }
        else if (c2OnPlatform > 0f && c1OnPlatform == 0f)
        {
            crawler2.AddReward(survivalReward);
            crawler1.AddReward(-survivalReward);
        }

        // Center control reward - zero-sum based on who's closer to center
        if (c1OnPlatform > 0f && c2OnPlatform > 0f)
        {
            float centerAdvantage = (c2DistFromCenter - c1DistFromCenter) / platformRadius;
            centerAdvantage = Mathf.Clamp(centerAdvantage, -1f, 1f);
            
            float c1CenterReward = centerControlReward * centerAdvantage;
            float c2CenterReward = -c1CenterReward;
            
            crawler1.AddReward(c1CenterReward);
            crawler2.AddReward(c2CenterReward);
            
            RecordStatForLearningAgent("CenterControlReward", c1CenterReward, crawler1);
            RecordStatForLearningAgent("CenterControlReward", c2CenterReward, crawler2);
        }

        // Pushing reward - zero-sum based on pushing opponent away from center
        Vector3 c1Movement = c1Pos - m_C1LastPosition;
        Vector3 c2Movement = c2Pos - m_C2LastPosition;
        
        // Calculate if agents are moving opponent away from center
        Vector3 c1ToCenter = (platformCenter - c1Pos).normalized;
        Vector3 c2ToCenter = (platformCenter - c2Pos).normalized;
        
        // Reward for opponent moving away from center when you're close
        float c1PushEffect = Vector3.Dot(c2Movement, -c2ToCenter);
        float c2PushEffect = Vector3.Dot(c1Movement, -c1ToCenter);
        
        float c1PushReward = pushingReward * Mathf.Max(0f, c1PushEffect);
        float c2PushReward = pushingReward * Mathf.Max(0f, c2PushEffect);
        
        crawler1.AddReward(c1PushReward - c2PushReward);
        crawler2.AddReward(c2PushReward - c1PushReward);
        
        RecordStatForLearningAgent("PushingReward", c1PushReward - c2PushReward, crawler1);
        RecordStatForLearningAgent("PushingReward", c2PushReward - c1PushReward, crawler2);

        // Body stability reward - zero-sum based on relative body orientation
        float c1Stability = Mathf.Clamp01(Vector3.Dot(crawler1.body.up, Vector3.up));
        float c2Stability = Mathf.Clamp01(Vector3.Dot(crawler2.body.up, Vector3.up));
        
        float stabilityDiff = c1Stability - c2Stability;
        float c1StabilityReward = stabilityReward * stabilityDiff;
        float c2StabilityReward = -c1StabilityReward;
        
        crawler1.AddReward(c1StabilityReward);
        crawler2.AddReward(c2StabilityReward);
        
        RecordStatForLearningAgent("StabilityReward", c1StabilityReward, crawler1);
        RecordStatForLearningAgent("StabilityReward", c2StabilityReward, crawler2);

        // Body-ground contact penalty - zero-sum
        bool c1BodyTouching = crawler1.IsBodyTouchingGround();
        bool c2BodyTouching = crawler2.IsBodyTouchingGround();
        
        float c1BodyPenalty = c1BodyTouching ? -bodyGroundPenalty : 0f;
        float c2BodyPenalty = c2BodyTouching ? -bodyGroundPenalty : 0f;
        
        // Make it zero-sum by giving opponent a small bonus when you touch ground
        if (c1BodyTouching && !c2BodyTouching)
        {
            crawler1.AddReward(c1BodyPenalty);
            crawler2.AddReward(-c1BodyPenalty);
        }
        else if (c2BodyTouching && !c1BodyTouching)
        {
            crawler2.AddReward(c2BodyPenalty);
            crawler1.AddReward(-c2BodyPenalty);
        }
        else if (c1BodyTouching && c2BodyTouching)
        {
            // Both touching - no penalty to maintain zero-sum
        }
        
        RecordStatForLearningAgent("BodyGroundPenalty", c1BodyPenalty, crawler1);
        RecordStatForLearningAgent("BodyGroundPenalty", c2BodyPenalty, crawler2);

        // Update last positions for next frame
        m_C1LastPosition = c1Pos;
        m_C2LastPosition = c2Pos;
    }

    private float EffectiveRadius()
    {
        if (ringShrinkEndProportion >= 1f || maxEpisodeSteps <= 0)
        {
            return platformRadius;
        }
        float t = Mathf.Clamp01(
            (m_StepCount - ringShrinkStartStep)
            / (float)Mathf.Max(1, maxEpisodeSteps - ringShrinkStartStep));
        return platformRadius * Mathf.Lerp(1f, ringShrinkEndProportion, t);
    }

    private bool CheckFallConditions()
    {
        float effRadius = EffectiveRadius();
        Vector3 p1 = crawler1.body.position;
        Vector3 p2 = crawler2.body.position;
        float d1 = Vector2.Distance(new Vector2(p1.x, p1.z), new Vector2(platformCenter.x, platformCenter.z));
        float d2 = Vector2.Distance(new Vector2(p2.x, p2.z), new Vector2(platformCenter.x, platformCenter.z));

        bool c1Flipped = false;
        bool c2Flipped = false;
        if (flipKnockdownSteps > 0)
        {
            m_C1FlipSteps = Vector3.Dot(crawler1.body.up, Vector3.up) < flipKnockdownDot ? m_C1FlipSteps + 1 : 0;
            m_C2FlipSteps = Vector3.Dot(crawler2.body.up, Vector3.up) < flipKnockdownDot ? m_C2FlipSteps + 1 : 0;
            c1Flipped = m_C1FlipSteps >= flipKnockdownSteps;
            c2Flipped = m_C2FlipSteps >= flipKnockdownSteps;
        }

        bool c1Fell = p1.y <= fallY || d1 > effRadius || c1Flipped;
        bool c2Fell = p2.y <= fallY || d2 > effRadius || c2Flipped;
        if (c1Fell || c2Fell)
        {
            m_LastEndWasFlip = c1Flipped || c2Flipped;
        }

        if (c1Fell && !c2Fell)
        {
            // Crawler1 fell, Crawler2 wins
            crawler1.AddReward(-winReward);
            crawler2.AddReward(winReward);
            EndBothEpisodesWithWinInfo(-winReward, winReward);
            ResetSumo();
            return true;
        }
        if (c2Fell && !c1Fell)
        {
            // Crawler2 fell, Crawler1 wins
            crawler2.AddReward(-winReward);
            crawler1.AddReward(winReward);
            EndBothEpisodesWithWinInfo(winReward, -winReward);
            ResetSumo();
            return true;
        }
        if (c1Fell && c2Fell)
        {
            // Both fell - draw
            EndBothEpisodesWithWinInfo(0f, 0f);
            ResetSumo();
            return true;
        }

        return false;
    }

    /// <summary>
    /// Fired whenever a match ends, with crawler1's terminal reward
    /// (positive = crawler1 won, negative = crawler2 won, 0 = draw).
    /// Used by RolloutViewer's evaluation mode; no gameplay effect.
    /// </summary>
    public static event System.Action<CrawlerSumoEnvController, float> MatchEnded;

    private void EndBothEpisodesWithWinInfo(float c1TerminalReward, float c2TerminalReward)
    {
        if (m_Recorder != null && flipKnockdownSteps > 0)
        {
            // Proportion of matches decided by flip knockdown (vs ring-out/timeout)
            m_Recorder.Add("CrawlerSumo/FlipKnockdownEnd", m_LastEndWasFlip ? 1f : 0f);
        }
        m_LastEndWasFlip = false;
        MatchEnded?.Invoke(this, c1TerminalReward);
        // Record win/loss statistics
        float c1WinReward = (c1TerminalReward > 0f) ? c1TerminalReward : 0f;
        float c2WinReward = (c2TerminalReward > 0f) ? c2TerminalReward : 0f;
        RecordStatForLearningAgent("WinReward", c1WinReward, crawler1);
        RecordStatForLearningAgent("WinReward", c2WinReward, crawler2);
        
        float c1LossPenalty = (c1TerminalReward < 0f) ? -c1TerminalReward : 0f;
        float c2LossPenalty = (c2TerminalReward < 0f) ? -c2TerminalReward : 0f;
        RecordStatForLearningAgent("LossPenalty", c1LossPenalty, crawler1);
        RecordStatForLearningAgent("LossPenalty", c2LossPenalty, crawler2);
        
        crawler1.EndEpisode();
        crawler2.EndEpisode();
    }
    
    private void EndBothEpisodes(float c1TerminalReward, float c2TerminalReward)
    {
        EndBothEpisodesWithWinInfo(c1TerminalReward, c2TerminalReward);
    }

    private void ResetSumo()
    {
        m_StepCount = 0;
        m_C1FlipSteps = 0;
        m_C2FlipSteps = 0;

        // Per-episode physics randomization (same values for both crawlers)
        if (m_C1Jd != null && (jointStrengthMultMin != 1f || jointStrengthMultMax != 1f))
        {
            float mult = Random.Range(jointStrengthMultMin, jointStrengthMultMax);
            m_C1Jd.maxJointSpring = m_BaseMaxJointSpring * mult;
            m_C1Jd.maxJointForceLimit = m_BaseMaxJointForceLimit * mult;
            m_C2Jd.maxJointSpring = m_BaseMaxJointSpring * mult;
            m_C2Jd.maxJointForceLimit = m_BaseMaxJointForceLimit * mult;
        }
        if (m_PlatformMaterial != null)
        {
            float friction = Random.Range(platformFrictionMin, platformFrictionMax);
            m_PlatformMaterial.staticFriction = friction;
            m_PlatformMaterial.dynamicFriction = friction;
        }

        // Generate random spawn positions on the platform
        Vector2 spawn1, spawn2;
        GenerateSpawnPositions(out spawn1, out spawn2);

        // Convert to 3D positions maintaining original Y
        Vector3 c1Start = new Vector3(spawn1.x, m_C1StartY, spawn1.y);
        Vector3 c2Start = new Vector3(spawn2.x, m_C2StartY, spawn2.y);

        // Position agents
        crawler1.transform.position = c1Start;
        crawler2.transform.position = c2Start;
        
        // Face each other initially
        Vector3 c1ToC2 = (c2Start - c1Start).normalized;
        Vector3 c2ToC1 = (c1Start - c2Start).normalized;
        
        crawler1.transform.rotation = Quaternion.LookRotation(c1ToC2, Vector3.up);
        crawler2.transform.rotation = Quaternion.LookRotation(c2ToC1, Vector3.up);

        // Zero velocities on all Rigidbodies
        ZeroRigidbodies(crawler1);
        ZeroRigidbodies(crawler2);

        // Initialize position tracking
        m_C1LastPosition = c1Start;
        m_C2LastPosition = c2Start;
    }

    private void GenerateSpawnPositions(out Vector2 spawn1, out Vector2 spawn2)
    {
        // Generate two random positions within the platform that maintain minimum distance
        // All distances are proportional to platform radius for scale independence
        int maxAttempts = 50;
        int attempts = 0;
        
        float maxSpawnDistance = maxSpawnDistanceProportion * platformRadius;
        float minSpawnDistance = minSpawnDistanceProportion * platformRadius;
        
        do
        {
            // Random angle and distance from center
            float angle1 = Random.Range(0f, 2f * Mathf.PI);
            float angle2 = Random.Range(0f, 2f * Mathf.PI);
            
            float dist1 = Random.Range(0f, maxSpawnDistance);
            float dist2 = Random.Range(0f, maxSpawnDistance);
            
            spawn1 = new Vector2(
                platformCenter.x + dist1 * Mathf.Cos(angle1),
                platformCenter.z + dist1 * Mathf.Sin(angle1)
            );
            
            spawn2 = new Vector2(
                platformCenter.x + dist2 * Mathf.Cos(angle2),
                platformCenter.z + dist2 * Mathf.Sin(angle2)
            );
            
            attempts++;
        }
        while (Vector2.Distance(spawn1, spawn2) < minSpawnDistance && attempts < maxAttempts);
        
        // Fallback if we couldn't find good positions (proportional to platform size)
        if (attempts >= maxAttempts)
        {
            float fallbackDistance = 0.2f * platformRadius; // 20% of radius separation
            spawn1 = new Vector2(platformCenter.x - fallbackDistance, platformCenter.z);
            spawn2 = new Vector2(platformCenter.x + fallbackDistance, platformCenter.z);
        }
    }

    private void ZeroRigidbodies(CrawlerSumoAgent agent)
    {
        foreach (var bodyPart in agent.GetComponent<JointDriveController>().bodyPartsList)
        {
            bodyPart.rb.linearVelocity = Vector3.zero;
            bodyPart.rb.angularVelocity = Vector3.zero;
        }
    }

    private void InitializeLearningTeamDetection()
    {
        m_CurrentLearningTeam = Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault("learning_team", -1f));
        
        if (m_CurrentLearningTeam == -1)
        {
            m_CurrentLearningTeam = 0;
            Debug.Log("CrawlerSumo: Learning team parameter not set, defaulting to team 0");
        }
        else
        {
            Debug.Log($"CrawlerSumo: Initial learning team detected as {m_CurrentLearningTeam}");
        }
    }
    
    private void UpdateLearningTeam()
    {
        int newLearningTeam = Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault("learning_team", m_CurrentLearningTeam));
        
        if (newLearningTeam != m_CurrentLearningTeam)
        {
            Debug.Log($"CrawlerSumo: Learning team changed from {m_CurrentLearningTeam} to {newLearningTeam}");
            m_CurrentLearningTeam = newLearningTeam;
            m_LearningTeamInitialized = true;
        }
    }
    
    private CrawlerSumoAgent GetLearningAgent()
    {
        var crawler1TeamId = crawler1.GetComponent<BehaviorParameters>().TeamId;
        var crawler2TeamId = crawler2.GetComponent<BehaviorParameters>().TeamId;
        
        if (m_CurrentLearningTeam == crawler1TeamId) return crawler1;
        if (m_CurrentLearningTeam == crawler2TeamId) return crawler2;
        
        return null;
    }
    
    private void RecordStatForLearningAgent(string key, float value, CrawlerSumoAgent agent)
    {
        if (m_Recorder == null) return;
        
        CrawlerSumoAgent learningAgent = GetLearningAgent();
        
        bool isLearningAgent = (learningAgent != null && learningAgent == agent);
        bool isOpponentAgent = (learningAgent != null && learningAgent != agent);
        
        if (learningAgent == null)
        {
            m_Recorder.Add($"CrawlerSumo/{key}", value);
            return;
        }
        
        if (isLearningAgent)
        {
            m_Recorder.Add($"CrawlerSumoLearner/{key}", value);
            
            if (m_LearningTeamInitialized)
            {
                var agentTeamId = agent.GetComponent<BehaviorParameters>().TeamId;
                m_Recorder.Add($"CrawlerSumoLearner/TeamId", agentTeamId);
            }
        }
        else if (isOpponentAgent)
        {
            m_Recorder.Add($"CrawlerSumoOpponent/{key}", value);
            
            if (m_LearningTeamInitialized)
            {
                var agentTeamId = agent.GetComponent<BehaviorParameters>().TeamId;
                m_Recorder.Add($"CrawlerSumoOpponent/TeamId", agentTeamId);
            }
        }
    }
    
    public int GetCurrentLearningTeam()
    {
        return m_CurrentLearningTeam;
    }
    
    public CrawlerSumoAgent GetCurrentLearningAgent()
    {
        return GetLearningAgent();
    }
}
