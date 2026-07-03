using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgentsExamples;

/// <summary>
/// Coordinates two SprinterCrawlerAgents in a head-to-head race along +X.
/// Handles episode resets, win/loss conditions, and dense shaping rewards.
/// </summary>
public class SprinterFightEnvController : MonoBehaviour
{
    [Header("Agents")] public SprinterCrawlerAgent crawler1;
    public SprinterCrawlerAgent crawler2;

    [Header("Track Settings")] public float startX = 0f;
    public float finishX = 105f;
    public float zHalfWidth = 14.5f;
    public float fallY = -10f;

    [Header("Episode Settings")] public int maxEpisodeSteps = 5000;

    [Header("Reward Weights")] public float progressRewardWeight = 1.0f;
    public float relativeLeadRewardWeight = 0.5f;
    public float headingAlignWeight = 0.05f;
    public float stepPenalty = 0.0f;
    public float winReward = 2.0f;
    public float losePenalty = -2.0f;
    public float opponentFallsBonus = 0.0f;
    public float fallPenalty = -2.0f;
    public float forwardVelocityRewardWeight = 0.05f;
    public float bodyGroundPenalty = -0.1f;

    private float m_C1LastProgress;
    private float m_C2LastProgress;
    private int m_StepCount;
    private float m_C1StartY;
    private float m_C2StartY;

    private StatsRecorder m_Recorder;
    private int m_CurrentLearningTeam = -1;
    private bool m_LearningTeamInitialized = false;

    private void Awake()
    {
        // Wire opponents and track settings
        crawler1.SetOpponent(crawler2);
        crawler2.SetOpponent(crawler1);
        crawler1.SetTrack(startX, finishX, zHalfWidth);
        crawler2.SetTrack(startX, finishX, zHalfWidth);

        m_C1StartY = crawler1.body.position.y;
        m_C2StartY = crawler2.body.position.y;

        m_Recorder = Academy.Instance.StatsRecorder;
        
        // Initialize learning team detection
        InitializeLearningTeamDetection();
    }

    private void Start()
    {
        ResetRace();
    }

    private void FixedUpdate()
    {
        m_StepCount++;

        // Early termination checks
        if (CheckFinishOrFall())
        {
            return;
        }

        // Update learning team detection
        UpdateLearningTeam();
        
        // Dense shaping
        float c1Prog = crawler1.GetNormalizedProgressX();
        float c2Prog = crawler2.GetNormalizedProgressX();

        float c1Delta = Mathf.Max(0f, c1Prog - m_C1LastProgress);
        float c2Delta = Mathf.Max(0f, c2Prog - m_C2LastProgress);

        // Absolute forward progress (positive-only to avoid oscillation exploits)
        crawler1.AddReward(progressRewardWeight * c1Delta);
        crawler2.AddReward(progressRewardWeight * c2Delta);
        RecordStatForLearningAgent("ProgressReward", progressRewardWeight * c1Delta, crawler1);
        RecordStatForLearningAgent("ProgressReward", progressRewardWeight * c2Delta, crawler2);

        // Relative lead shaping
        float lead1 = c1Prog - c2Prog;
        float lead2 = c2Prog - c1Prog;
        crawler1.AddReward(relativeLeadRewardWeight * lead1);
        crawler2.AddReward(relativeLeadRewardWeight * lead2);
        RecordStatForLearningAgent("RelativeLeadReward", relativeLeadRewardWeight * lead1, crawler1);
        RecordStatForLearningAgent("RelativeLeadReward", relativeLeadRewardWeight * lead2, crawler2);

        // Heading alignment with +X
        float c1Heading = Mathf.Clamp01(Vector3.Dot(crawler1.body.forward, Vector3.right));
        float c2Heading = Mathf.Clamp01(Vector3.Dot(crawler2.body.forward, Vector3.right));
        crawler1.AddReward(headingAlignWeight * c1Heading);
        crawler2.AddReward(headingAlignWeight * c2Heading);
        RecordStatForLearningAgent("HeadingAlignReward", headingAlignWeight * c1Heading, crawler1);
        RecordStatForLearningAgent("HeadingAlignReward", headingAlignWeight * c2Heading, crawler2);

        // Forward velocity shaping (positive-only along +X)
        if (forwardVelocityRewardWeight != 0f)
        {
            float c1SpeedX = Mathf.Max(0f, Vector3.Dot(crawler1.GetAvgVelocity(), Vector3.right));
            float c2SpeedX = Mathf.Max(0f, Vector3.Dot(crawler2.GetAvgVelocity(), Vector3.right));
            float c1VelReward = forwardVelocityRewardWeight * c1SpeedX;
            float c2VelReward = forwardVelocityRewardWeight * c2SpeedX;
            crawler1.AddReward(c1VelReward);
            crawler2.AddReward(c2VelReward);
            RecordStatForLearningAgent("ForwardVelocityReward", c1VelReward, crawler1);
            RecordStatForLearningAgent("ForwardVelocityReward", c2VelReward, crawler2);
        }

        // Body-ground contact penalty (applied per step when touching)
        if (bodyGroundPenalty != 0f)
        {
            bool c1BodyTouching = crawler1.IsBodyTouchingGround();
            bool c2BodyTouching = crawler2.IsBodyTouchingGround();
            float c1Penalty = c1BodyTouching ? bodyGroundPenalty : 0f;
            float c2Penalty = c2BodyTouching ? bodyGroundPenalty : 0f;
            if (c1Penalty != 0f) crawler1.AddReward(c1Penalty);
            if (c2Penalty != 0f) crawler2.AddReward(c2Penalty);
            // Record zero as well so the average reflects per-step mean
            RecordStatForLearningAgent("BodyGroundPenalty", c1Penalty, crawler1);
            RecordStatForLearningAgent("BodyGroundPenalty", c2Penalty, crawler2);
        }

        // Optional step penalty
        if (stepPenalty != 0f)
        {
            crawler1.AddReward(stepPenalty);
            crawler2.AddReward(stepPenalty);
            RecordStatForLearningAgent("StepPenalty", stepPenalty, crawler1);
            RecordStatForLearningAgent("StepPenalty", stepPenalty, crawler2);
        }

        m_C1LastProgress = c1Prog;
        m_C2LastProgress = c2Prog;

        // Max steps termination
        if (maxEpisodeSteps > 0 && m_StepCount >= maxEpisodeSteps)
        {
            EndBothEpisodesWithFallInfo(0f, 0f, 0f, 0f, 0f, 0f); // draw - no terminal rewards
            ResetRace();
        }
    }

    private bool CheckFinishOrFall()
    {
        // Finish line
        bool c1Finished = crawler1.body.position.x >= finishX;
        bool c2Finished = crawler2.body.position.x >= finishX;

        if (c1Finished && !c2Finished)
        {
            EndBothEpisodesWithFallInfo(winReward, losePenalty, 0f, 0f, 0f, 0f);
            ResetRace();
            return true;
        }
        if (c2Finished && !c1Finished)
        {
            EndBothEpisodesWithFallInfo(losePenalty, winReward, 0f, 0f, 0f, 0f);
            ResetRace();
            return true;
        }
        if (c1Finished && c2Finished)
        {
            EndBothEpisodesWithFallInfo(0.5f * winReward, 0.5f * winReward, 0f, 0f, 0f, 0f);
            ResetRace();
            return true;
        }

        // Fall off
        bool c1Fell = crawler1.body.position.y <= fallY;
        bool c2Fell = crawler2.body.position.y <= fallY;

        if (c1Fell && !c2Fell)
        {
            crawler1.AddReward(fallPenalty);
            crawler2.AddReward(opponentFallsBonus);
            EndBothEpisodesWithFallInfo(0f, 0f, fallPenalty, 0f, 0f, opponentFallsBonus);
            ResetRace();
            return true;
        }
        if (c2Fell && !c1Fell)
        {
            crawler2.AddReward(fallPenalty);
            crawler1.AddReward(opponentFallsBonus);
            EndBothEpisodesWithFallInfo(0f, 0f, 0f, fallPenalty, opponentFallsBonus, 0f);
            ResetRace();
            return true;
        }
        if (c1Fell && c2Fell)
        {
            crawler1.AddReward(fallPenalty * 0.5f);
            crawler2.AddReward(fallPenalty * 0.5f);
            EndBothEpisodesWithFallInfo(0f, 0f, fallPenalty * 0.5f, fallPenalty * 0.5f, 0f, 0f);
            ResetRace();
            return true;
        }

        return false;
    }

    private void EndBothEpisodesWithFallInfo(float c1TerminalBonus, float c2TerminalBonus, 
        float c1FallPenalty, float c2FallPenalty, float c1OpponentFallsBonus, float c2OpponentFallsBonus)
    {
        // Apply actual rewards
        if (c1TerminalBonus != 0f)
        {
            crawler1.AddReward(c1TerminalBonus);
        }
        if (c2TerminalBonus != 0f)
        {
            crawler2.AddReward(c2TerminalBonus);
        }
        
        // Always record all terminal reward components to ensure proper averaging
        // Record WinReward (positive values) or 0
        float c1WinReward = (c1TerminalBonus > 0f) ? c1TerminalBonus : 0f;
        float c2WinReward = (c2TerminalBonus > 0f) ? c2TerminalBonus : 0f;
        RecordStatForLearningAgent("WinReward", c1WinReward, crawler1);
        RecordStatForLearningAgent("WinReward", c2WinReward, crawler2);
        
        // Record LosePenalty (negative values converted to positive) or 0
        float c1LosePenalty = (c1TerminalBonus < 0f) ? -c1TerminalBonus : 0f;
        float c2LosePenalty = (c2TerminalBonus < 0f) ? -c2TerminalBonus : 0f;
        RecordStatForLearningAgent("LosePenalty", c1LosePenalty, crawler1);
        RecordStatForLearningAgent("LosePenalty", c2LosePenalty, crawler2);
        
        // Always record FallPenalty (convert to positive value for stats) 
        float c1FallPenaltyPositive = (c1FallPenalty != 0f) ? -c1FallPenalty : 0f;
        float c2FallPenaltyPositive = (c2FallPenalty != 0f) ? -c2FallPenalty : 0f;
        RecordStatForLearningAgent("FallPenalty", c1FallPenaltyPositive, crawler1);
        RecordStatForLearningAgent("FallPenalty", c2FallPenaltyPositive, crawler2);
        
        // Always record OpponentFallsBonus or 0
        RecordStatForLearningAgent("OpponentFallsBonus", c1OpponentFallsBonus, crawler1);
        RecordStatForLearningAgent("OpponentFallsBonus", c2OpponentFallsBonus, crawler2);
        
        crawler1.EndEpisode();
        crawler2.EndEpisode();
    }
    
    private void EndBothEpisodes(float c1TerminalBonus, float c2TerminalBonus)
    {
        // Legacy method - redirects to comprehensive version with no fall rewards
        EndBothEpisodesWithFallInfo(c1TerminalBonus, c2TerminalBonus, 0f, 0f, 0f, 0f);
    }

    private void ResetRace()
    {
        m_StepCount = 0;

        // Random lateral start within track; small separation
        float z1 = Random.Range(-zHalfWidth * 0.6f, -zHalfWidth * 0.1f);
        float z2 = Random.Range(zHalfWidth * 0.1f, zHalfWidth * 0.6f);

        // Maintain initial Y to avoid sinking or popping
        Vector3 c1Start = new Vector3(startX, m_C1StartY, z1);
        Vector3 c2Start = new Vector3(startX, m_C2StartY, z2);

        // Provide desired starts to be applied on OnEpisodeBegin
        crawler1.transform.position = c1Start;
        crawler2.transform.position = c2Start;
        crawler1.transform.rotation = Quaternion.Euler(0f, 90f, 0f);
        crawler2.transform.rotation = Quaternion.Euler(0f, 90f, 0f);

        // Zero velocities on all Rigidbodies
        ZeroRigidbodies(crawler1);
        ZeroRigidbodies(crawler2);

        // Reset internal progress trackers
        m_C1LastProgress = crawler1.GetNormalizedProgressX();
        m_C2LastProgress = crawler2.GetNormalizedProgressX();
    }

    private void ZeroRigidbodies(SprinterCrawlerAgent agent)
    {
        foreach (var bodyPart in agent.GetComponent<JointDriveController>().bodyPartsList)
        {
            bodyPart.rb.linearVelocity = Vector3.zero;
            bodyPart.rb.angularVelocity = Vector3.zero;
        }
    }

    private void InitializeLearningTeamDetection()
    {
        // Get initial learning team from environment parameters
        // Default to -1 if not set yet
        m_CurrentLearningTeam = Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault("learning_team", -1f));
        
        // If still not set, try to detect from team IDs (fallback for initial episodes)
        if (m_CurrentLearningTeam == -1)
        {
            // Default to team 0 as initial learning team based on typical self-play setups
            m_CurrentLearningTeam = 0;
            Debug.Log("SprinterFight: Learning team parameter not set, defaulting to team 0");
        }
        else
        {
            Debug.Log($"SprinterFight: Initial learning team detected as {m_CurrentLearningTeam}");
        }
    }
    
    private void UpdateLearningTeam()
    {
        int newLearningTeam = Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault("learning_team", m_CurrentLearningTeam));
        
        if (newLearningTeam != m_CurrentLearningTeam)
        {
            Debug.Log($"SprinterFight: Learning team changed from {m_CurrentLearningTeam} to {newLearningTeam}");
            m_CurrentLearningTeam = newLearningTeam;
            m_LearningTeamInitialized = true;
        }
    }
    
    private SprinterCrawlerAgent GetLearningAgent()
    {
        var crawler1TeamId = crawler1.GetComponent<BehaviorParameters>().TeamId;
        var crawler2TeamId = crawler2.GetComponent<BehaviorParameters>().TeamId;
        
        if (m_CurrentLearningTeam == crawler1TeamId) return crawler1;
        if (m_CurrentLearningTeam == crawler2TeamId) return crawler2;
        
        // Fallback: if learning team not properly detected, record for both agents
        // This ensures we don't lose stats during initialization
        return null;
    }
    
    private void RecordStatForLearningAgent(string key, float value, SprinterCrawlerAgent agent)
    {
        if (m_Recorder == null) return;
        
        SprinterCrawlerAgent learningAgent = GetLearningAgent();
        
        // Determine if this agent is the learning agent or opponent
        bool isLearningAgent = (learningAgent != null && learningAgent == agent);
        bool isOpponentAgent = (learningAgent != null && learningAgent != agent);
        
        // If learning agent not detected yet, record for both with fallback prefix
        if (learningAgent == null)
        {
            m_Recorder.Add($"SprinterFight/{key}", value);
            return;
        }
        
        // Record with appropriate prefix based on agent role
        if (isLearningAgent)
        {
            m_Recorder.Add($"SprinterFightLearner/{key}", value);
            
            // Add debug info for learning team detection
            if (m_LearningTeamInitialized)
            {
                var agentTeamId = agent.GetComponent<BehaviorParameters>().TeamId;
                m_Recorder.Add($"SprinterFightLearner/TeamId", agentTeamId);
            }
        }
        else if (isOpponentAgent)
        {
            m_Recorder.Add($"SprinterFightOpponent/{key}", value);
            
            // Add debug info for opponent team
            if (m_LearningTeamInitialized)
            {
                var agentTeamId = agent.GetComponent<BehaviorParameters>().TeamId;
                m_Recorder.Add($"SprinterFightOpponent/TeamId", agentTeamId);
            }
        }
    }
    
    private void RecordStat(string key, float value)
    {
        if (m_Recorder != null)
        {
            m_Recorder.Add(key, value);
        }
    }
    
    /// <summary>
    /// Get the current learning team ID for external access
    /// </summary>
    public int GetCurrentLearningTeam()
    {
        return m_CurrentLearningTeam;
    }
    
    /// <summary>
    /// Get the currently learning agent for external access
    /// </summary>
    public SprinterCrawlerAgent GetCurrentLearningAgent()
    {
        return GetLearningAgent();
    }
}


