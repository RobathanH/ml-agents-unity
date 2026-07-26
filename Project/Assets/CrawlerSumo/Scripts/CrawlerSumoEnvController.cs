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

    // The ring radius is FIXED at platformRadius for the whole episode. Runs
    // 009-013 shrank it (15u -> 1.5u from step 300) to break the 006-008
    // draw-camping equilibria; removed 2026-07-26 because the shrink was
    // invisible to the policy and the observations actively contradicted it:
    // CollectObservations reports platformRadius/20 as a constant and
    // normalizedEdgeDistance/onPlatform against the FULL radius, so a crawler
    // was told it had 11u of margin in the step before a ring-out killed it.
    // An unobservable, within-episode-varying kill boundary is a second,
    // implicit shaping channel -- it taught centre-hugging and a low crouch
    // (brace for the floor vanishing) and it inflates value loss, which is the
    // same lesson as the run-007 unobserved physics randomization. All
    // behaviour shaping now lives in one place: the reward terms below.

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

    // Run 013 control costs, applied per-agent inside CrawlerSumoAgent. 0 disables.
    private float energyCostWeight = 0f;
    private float actionRateCostWeight = 0f;

    /// <summary>
    /// One reward component's episode total, tracked two ways.
    ///
    /// Raw is the undiscounted sum: the right lens for "is the reward function
    /// well-posed", i.e. does a shaping term carry enough mass to move the
    /// argmax (the runs 007/008 failure).
    ///
    /// Disc weights each contribution by gamma^(decision index), making it the
    /// component's share of V(s_0) -- what the policy actually optimises. The
    /// two differ a lot here, and not uniformly: dense shaping accrues
    /// throughout the episode and keeps ~52% of its mass at gamma 0.995 over
    /// 300 decisions, while terminal reward sits entirely at T and keeps only
    /// ~22%. Reading Raw alone overstates the win condition's pull by ~2.3x.
    /// (Second-order, not captured here: with GAE lambda 0.95 the advantage's
    /// effective horizon is ~18 decisions, so terminal reward reaches early
    /// decisions almost entirely through the value function -- when value loss
    /// is high, its real influence is weaker still than Disc suggests.)
    /// </summary>
    private struct RewardTerm
    {
        public float Raw;
        public float Disc;

        public void Add(float v, float discount)
        {
            Raw += v;
            Disc += v * discount;
        }

        public void Clear()
        {
            Raw = 0f;
            Disc = 0f;
        }
    }

    /// <summary>
    /// Per-episode reward totals, one instance per crawler. Recorded once at
    /// episode end, in the same units as winReward. Per-step means cannot show
    /// shaping-vs-terminal imbalance; these can.
    /// </summary>
    private struct RewardBreakdown
    {
        public RewardTerm Survival;
        public RewardTerm CenterControl;
        public RewardTerm Pushing;
        public RewardTerm Stability;
        public RewardTerm BodyGround;
        public RewardTerm Terminal;

        public float ShapingRaw =>
            Survival.Raw + CenterControl.Raw + Pushing.Raw + Stability.Raw + BodyGround.Raw;

        public float ShapingDisc =>
            Survival.Disc + CenterControl.Disc + Pushing.Disc + Stability.Disc + BodyGround.Disc;

        public void Clear()
        {
            Survival.Clear();
            CenterControl.Clear();
            Pushing.Clear();
            Stability.Clear();
            BodyGround.Clear();
            Terminal.Clear();
        }
    }

    private RewardBreakdown m_C1Rewards;
    private RewardBreakdown m_C2Rewards;

    // gamma^(current decision index), for the Disc totals above. Diagnostics
    // only -- never touches the reward actually sent to the trainer.
    private float m_StatsGamma = 0.995f;
    private int m_DecisionPeriod = 1;
    private float m_Discount = 1f;
    private int m_DiscountDecisionIndex;

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

    // CLI overrides for standalone eval/viewer builds, which have no python
    // side channel and would otherwise run DEFAULT physics -- a policy trained
    // under modified constants (e.g. run 011's 1.5x strength) is meaningless
    // to evaluate under different physics. Repeatable arg: --env-param name=value.
    // Lazily parsed: no script-execution-order dependency. Training never
    // passes the flag, so trainer-driven runs are unaffected.
    private static System.Collections.Generic.Dictionary<string, float> s_CliOverrides;

    private static System.Collections.Generic.Dictionary<string, float> CliOverrides
    {
        get
        {
            if (s_CliOverrides == null)
            {
                s_CliOverrides = new System.Collections.Generic.Dictionary<string, float>();
                var args = System.Environment.GetCommandLineArgs();
                for (int i = 0; i < args.Length - 1; i++)
                {
                    if (args[i] != "--env-param") continue;
                    var kv = args[i + 1].Split('=');
                    if (kv.Length == 2 && float.TryParse(
                            kv[1], System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture, out var f))
                    {
                        s_CliOverrides[kv[0]] = f;
                        Debug.Log($"CrawlerSumo: CLI env-param override {kv[0]}={f}");
                    }
                }
            }
            return s_CliOverrides;
        }
    }

    private float GetParam(string name, float def)
    {
        if (CliOverrides.TryGetValue(name, out var v)) return v;
        return Academy.Instance.EnvironmentParameters.GetWithDefault(name, def);
    }

    private void LoadEnvironmentParameters()
    {
        // Spawn settings (proportional to platform radius)
        minSpawnDistanceProportion = GetParam("min_spawn_distance_proportion", minSpawnDistanceProportion);
        maxSpawnDistanceProportion = GetParam("max_spawn_distance_proportion", maxSpawnDistanceProportion);

        // Episode settings
        maxEpisodeSteps = Mathf.RoundToInt(GetParam("max_episode_steps", maxEpisodeSteps));

        // Reward weights
        survivalReward = GetParam("survival_reward", survivalReward);
        centerControlReward = GetParam("center_control_reward", centerControlReward);
        pushingReward = GetParam("pushing_reward", pushingReward);
        winReward = GetParam("win_reward", winReward);
        bodyGroundPenalty = GetParam("body_ground_penalty", bodyGroundPenalty);
        stabilityReward = GetParam("stability_reward", stabilityReward);

        // Control costs (run 013). Pushed down to the agents, which apply them
        // in OnActionReceived where the action vector is available.
        energyCostWeight = GetParam("energy_cost_weight", energyCostWeight);
        actionRateCostWeight = GetParam("action_rate_cost_weight", actionRateCostWeight);
        crawler1.energyCostWeight = energyCostWeight;
        crawler2.energyCostWeight = energyCostWeight;
        crawler1.actionRateCostWeight = actionRateCostWeight;
        crawler2.actionRateCostWeight = actionRateCostWeight;

        // Diagnostics only: must mirror reward_signals.extrinsic.gamma, which
        // is trainer-side and not visible to the environment. If they drift,
        // the EpRewardDisc/* stats are wrong but training is unaffected.
        m_StatsGamma = GetParam("stats_gamma", m_StatsGamma);
        crawler1.statsGamma = m_StatsGamma;
        crawler2.statsGamma = m_StatsGamma;

        // gamma is per DECISION, not per physics step: AddReward accumulates
        // across the intermediate FixedUpdates and is delivered at the next
        // decision, so all 5 physics steps share one discount exponent.
        var dr = crawler1.GetComponent<DecisionRequester>();
        m_DecisionPeriod = dr != null ? Mathf.Max(1, dr.DecisionPeriod) : 1;

        // Actuator strength randomization
        jointStrengthMultMin = GetParam("joint_strength_multiplier_min", jointStrengthMultMin);
        jointStrengthMultMax = GetParam("joint_strength_multiplier_max", jointStrengthMultMax);

        // Platform friction randomization
        platformFrictionMin = GetParam("platform_friction_min", platformFrictionMin);
        platformFrictionMax = GetParam("platform_friction_max", platformFrictionMax);

        // Flip knockdown
        flipKnockdownSteps = Mathf.RoundToInt(GetParam("flip_knockdown_steps", flipKnockdownSteps));
        flipKnockdownDot = GetParam("flip_knockdown_dot", flipKnockdownDot);
        
        Debug.Log($"CrawlerSumo: Loaded environment parameters - Platform Radius: {platformRadius} (scene), " +
                  $"Spawn Range: {minSpawnDistanceProportion * platformRadius:F1}-{maxSpawnDistanceProportion * platformRadius:F1}, " +
                  $"Win Reward: {winReward}");
    }

    private void Start()
    {
        InitPhysicsRandomization();
        ApplyTeamColors();
        ResetSumo();
    }

    // Team tints so rollouts are unambiguous about which agent is which:
    // team 0 = blue, team 1 = orange. In viewer/eval builds --team0-model is
    // always the "current" checkpoint, so blue = current, orange = opponent.
    // MaterialPropertyBlock only: no material instances, no effect on physics,
    // and -nographics training runs never draw it.
    private static readonly Color k_Team0Color = new Color(0.25f, 0.55f, 1f);
    private static readonly Color k_Team1Color = new Color(1f, 0.5f, 0.1f);

    private void ApplyTeamColors()
    {
        TintCrawler(crawler1);
        TintCrawler(crawler2);
    }

    private void TintCrawler(CrawlerSumoAgent crawler)
    {
        var bp = crawler.GetComponent<BehaviorParameters>();
        var teamId = bp != null ? bp.TeamId : 0;
        var mpb = new MaterialPropertyBlock();
        mpb.SetColor("_Color", teamId == 0 ? k_Team0Color : k_Team1Color);
        foreach (var r in crawler.GetComponentsInChildren<Renderer>(true))
        {
            var m = r.sharedMaterial;
            // Black joint segments keep their material for pose contrast
            if (m != null && m.name.StartsWith("Black")) continue;
            r.SetPropertyBlock(mpb);
        }
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
            // Draw, no terminal rewards -- and flagged as a TRUNCATION, not a
            // terminal state. See EndBothEpisodesWithWinInfo.
            EndBothEpisodesWithWinInfo(0f, 0f, interrupted: true);
            ResetSumo();
        }
    }

    /// <summary>
    /// Advance gamma^(decision index) to match the current physics step.
    /// m_StepCount is incremented at the top of FixedUpdate, so the episode's
    /// first physics step is decision index 0 (discount 1).
    /// </summary>
    private void UpdateDiscount()
    {
        int decisionIndex = Mathf.Max(0, m_StepCount - 1) / m_DecisionPeriod;
        while (m_DiscountDecisionIndex < decisionIndex)
        {
            m_Discount *= m_StatsGamma;
            m_DiscountDecisionIndex++;
        }
    }

    private void ApplyZeroSumRewards()
    {
        UpdateDiscount();

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
            m_C1Rewards.Survival.Add(survivalReward, m_Discount);
            m_C2Rewards.Survival.Add(-survivalReward, m_Discount);
        }
        else if (c2OnPlatform > 0f && c1OnPlatform == 0f)
        {
            crawler2.AddReward(survivalReward);
            crawler1.AddReward(-survivalReward);
            m_C2Rewards.Survival.Add(survivalReward, m_Discount);
            m_C1Rewards.Survival.Add(-survivalReward, m_Discount);
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
            m_C1Rewards.CenterControl.Add(c1CenterReward, m_Discount);
            m_C2Rewards.CenterControl.Add(c2CenterReward, m_Discount);

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
        m_C1Rewards.Pushing.Add(c1PushReward - c2PushReward, m_Discount);
        m_C2Rewards.Pushing.Add(c2PushReward - c1PushReward, m_Discount);

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
        m_C1Rewards.Stability.Add(c1StabilityReward, m_Discount);
        m_C2Rewards.Stability.Add(c2StabilityReward, m_Discount);

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
            // Accumulate what was actually applied, not the notional penalty:
            // in the both-touching branch nothing is added, so the per-step
            // stat below over-reports while these episode totals do not.
            m_C1Rewards.BodyGround.Add(c1BodyPenalty, m_Discount);
            m_C2Rewards.BodyGround.Add(-c1BodyPenalty, m_Discount);
        }
        else if (c2BodyTouching && !c1BodyTouching)
        {
            crawler2.AddReward(c2BodyPenalty);
            crawler1.AddReward(-c2BodyPenalty);
            m_C2Rewards.BodyGround.Add(c2BodyPenalty, m_Discount);
            m_C1Rewards.BodyGround.Add(-c2BodyPenalty, m_Discount);
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

        bool c1Fell = p1.y <= fallY || d1 > platformRadius || c1Flipped;
        bool c2Fell = p2.y <= fallY || d2 > platformRadius || c2Flipped;
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

    /// <summary>
    /// Ends the match. <paramref name="interrupted"/> distinguishes the two
    /// kinds of ending, which look identical to the environment but not to the
    /// trainer.
    ///
    /// A ring-out is a real terminal: nothing follows it, so a value target of
    /// 0 is correct. Hitting maxEpisodeSteps is a TRUNCATION -- the fight was
    /// still going and we cut it off -- so Agent.EpisodeInterrupted is the
    /// right call. It sets DoneReason.MaxStepReached, which makes the trainer
    /// bootstrap V(s_T) instead of forcing 0 (ppo/trainer.py L98 passes
    /// `done_reached and not interrupted` into the value estimate).
    /// EndEpisode here would train the critic to predict 0 at a step the agent
    /// cannot identify -- there is no clock in the observation -- which is the
    /// same unobservable-mechanic bug as the shrinking ring, just on the value
    /// target instead of the termination boundary.
    ///
    /// KNOWN SIDE EFFECT: ghost/trainer.py L201-205 skips ELO accounting for
    /// interrupted trajectories, so from run 014 on, Self-play/ELO is computed
    /// from DECISIVE games only and draws no longer damp it toward the
    /// opponent's rating. ELO numbers are therefore NOT comparable with runs
    /// 006-013, and a draw-camping meta would no longer show up as a flat ELO
    /// (it did not reliably show up before either -- run 008 climbed 1201->1634
    /// while drawing 98% of tournament games). The DrawRate/TimeoutRate stats
    /// below exist to monitor that directly instead of inferring it from ELO.
    /// </summary>
    private void EndBothEpisodesWithWinInfo(
        float c1TerminalReward, float c2TerminalReward, bool interrupted = false)
    {
        if (m_Recorder != null)
        {
            if (flipKnockdownSteps > 0)
            {
                // Proportion of matches decided by flip knockdown (vs ring-out/timeout)
                m_Recorder.Add("CrawlerSumo/FlipKnockdownEnd", m_LastEndWasFlip ? 1f : 0f);
            }
            // Match outcome mix. DrawRate counts both timeouts and the rare
            // both-fell-together case; TimeoutRate isolates the cap. Match
            // properties, not agent properties, so they are recorded unsplit.
            m_Recorder.Add("CrawlerSumo/DrawRate", c1TerminalReward == 0f ? 1f : 0f);
            m_Recorder.Add("CrawlerSumo/TimeoutRate", interrupted ? 1f : 0f);
        }
        m_LastEndWasFlip = false;
        MatchEnded?.Invoke(this, c1TerminalReward);

        // Reward breakdown must be recorded BEFORE EndEpisode(), which triggers
        // OnEpisodeBegin() and clears the agents' control-cost accumulators.
        // CheckFallConditions can end the episode before ApplyZeroSumRewards
        // runs this step, so refresh the discount before stamping the terminal.
        UpdateDiscount();
        m_C1Rewards.Terminal.Add(c1TerminalReward, m_Discount);
        m_C2Rewards.Terminal.Add(c2TerminalReward, m_Discount);
        RecordRewardBreakdown(crawler1, m_C1Rewards);
        RecordRewardBreakdown(crawler2, m_C2Rewards);
        // Record win/loss statistics
        float c1WinReward = (c1TerminalReward > 0f) ? c1TerminalReward : 0f;
        float c2WinReward = (c2TerminalReward > 0f) ? c2TerminalReward : 0f;
        RecordStatForLearningAgent("WinReward", c1WinReward, crawler1);
        RecordStatForLearningAgent("WinReward", c2WinReward, crawler2);
        
        float c1LossPenalty = (c1TerminalReward < 0f) ? -c1TerminalReward : 0f;
        float c2LossPenalty = (c2TerminalReward < 0f) ? -c2TerminalReward : 0f;
        RecordStatForLearningAgent("LossPenalty", c1LossPenalty, crawler1);
        RecordStatForLearningAgent("LossPenalty", c2LossPenalty, crawler2);
        
        if (interrupted)
        {
            crawler1.EpisodeInterrupted();
            crawler2.EpisodeInterrupted();
        }
        else
        {
            crawler1.EndEpisode();
            crawler2.EndEpisode();
        }
    }

    /// <summary>
    /// Emit every reward component as an episode total, in the same units as
    /// winReward. Read these together: if ShapingTotal rivals or exceeds
    /// Terminal, shaping is defining the objective rather than guiding it, and
    /// the policy will optimise the shaping term (runs 007/008). EnergyCost and
    /// ActionRateCost are recorded as their signed contribution to return, so
    /// every key sums to Total.
    /// </summary>
    private void RecordRewardBreakdown(CrawlerSumoAgent agent, RewardBreakdown r)
    {
        if (m_Recorder == null) return;

        float energyCost = agent.EpisodeEnergyCost;
        float rateCost = agent.EpisodeActionRateCost;
        float energyCostDisc = agent.EpisodeEnergyCostDisc;
        float rateCostDisc = agent.EpisodeActionRateCostDisc;

        // Undiscounted mass: is the reward function well-posed?
        RecordStatForLearningAgent("EpReward/Survival", r.Survival.Raw, agent);
        RecordStatForLearningAgent("EpReward/CenterControl", r.CenterControl.Raw, agent);
        RecordStatForLearningAgent("EpReward/Pushing", r.Pushing.Raw, agent);
        RecordStatForLearningAgent("EpReward/Stability", r.Stability.Raw, agent);
        RecordStatForLearningAgent("EpReward/BodyGround", r.BodyGround.Raw, agent);
        RecordStatForLearningAgent("EpReward/EnergyCost", -energyCost, agent);
        RecordStatForLearningAgent("EpReward/ActionRateCost", -rateCost, agent);
        RecordStatForLearningAgent("EpReward/ShapingTotal", r.ShapingRaw, agent);
        RecordStatForLearningAgent("EpReward/Terminal", r.Terminal.Raw, agent);
        RecordStatForLearningAgent(
            "EpReward/Total", r.ShapingRaw + r.Terminal.Raw - energyCost - rateCost, agent);
        RecordStatForLearningAgent("EpReward/EpisodeSteps", m_StepCount, agent);

        // gamma-weighted: what the policy actually optimises. Compare
        // EpRewardDisc/ShapingTotal against EpRewardDisc/Terminal -- the raw
        // pair above flatters the terminal by ~2.3x at gamma 0.995 over a
        // full-length episode, because shaping accrues throughout while the
        // terminal sits entirely at T.
        RecordStatForLearningAgent("EpRewardDisc/Survival", r.Survival.Disc, agent);
        RecordStatForLearningAgent("EpRewardDisc/CenterControl", r.CenterControl.Disc, agent);
        RecordStatForLearningAgent("EpRewardDisc/Pushing", r.Pushing.Disc, agent);
        RecordStatForLearningAgent("EpRewardDisc/Stability", r.Stability.Disc, agent);
        RecordStatForLearningAgent("EpRewardDisc/BodyGround", r.BodyGround.Disc, agent);
        RecordStatForLearningAgent("EpRewardDisc/EnergyCost", -energyCostDisc, agent);
        RecordStatForLearningAgent("EpRewardDisc/ActionRateCost", -rateCostDisc, agent);
        RecordStatForLearningAgent("EpRewardDisc/ShapingTotal", r.ShapingDisc, agent);
        RecordStatForLearningAgent("EpRewardDisc/Terminal", r.Terminal.Disc, agent);
        RecordStatForLearningAgent(
            "EpRewardDisc/Total",
            r.ShapingDisc + r.Terminal.Disc - energyCostDisc - rateCostDisc, agent);
        // Sanity check that this env's gamma matches the trainer's: this is
        // V(s_0) under the recorded gamma, and should track the trainer's own
        // value estimate. A persistent gap means stats_gamma has drifted from
        // reward_signals.extrinsic.gamma.
        RecordStatForLearningAgent("EpRewardDisc/FinalDiscount", m_Discount, agent);
    }

    private void ResetSumo()
    {
        m_StepCount = 0;
        m_C1FlipSteps = 0;
        m_C2FlipSteps = 0;
        m_C1Rewards.Clear();
        m_C2Rewards.Clear();
        m_Discount = 1f;
        m_DiscountDecisionIndex = 0;

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
