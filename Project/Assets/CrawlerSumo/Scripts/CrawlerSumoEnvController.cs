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
    private int maxEpisodeSteps = 3000;
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
        
        Debug.Log($"CrawlerSumo: Loaded environment parameters - Platform Radius: {platformRadius} (scene), " +
                  $"Spawn Range: {minSpawnDistanceProportion * platformRadius:F1}-{maxSpawnDistanceProportion * platformRadius:F1}, " +
                  $"Win Reward: {winReward}");
    }

    private void Start()
    {
        ResetSumo();
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

    private bool CheckFallConditions()
    {
        bool c1Fell = crawler1.body.position.y <= fallY;
        bool c2Fell = crawler2.body.position.y <= fallY;

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

    private void EndBothEpisodesWithWinInfo(float c1TerminalReward, float c2TerminalReward)
    {
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
