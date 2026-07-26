using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// Owns one parkour arena: regenerates the track each episode, spawns the
    /// agent, respawns it when it falls, and ends the episode on finish or timeout.
    /// </summary>
    public class CrawlerParkourEnvController : MonoBehaviour
    {
        public CrawlerParkourAgent agent;
        public ParkourTrackGenerator track;
        public ParkourObstacleSource obstacleSource;

        [Header("Episode")]
        public int maxEpisodeSteps = 3000;
        [Tooltip("Depth below local ground that counts as having fallen off.")]
        public float fallDepth = 3f;

        [Header("Difficulty")]
        [Range(0f, 1f)] public float difficulty = 0f;
        [Tooltip("Fixed seed for reproducible eval. Negative means random per episode.")]
        public int fixedSeed = -1;

        [Header("Rewards")]
        public float progressWeight = 10f;
        public float respawnPenalty = 0.5f;
        public float finishBonus = 5f;
        public float energyCostWeight = 0f;
        public float actionRateCostWeight = 0f;
        [Tooltip("Whole-episode reward budget for perfect down-track locomotion.")]
        public float velocityWeight = 2f;
        [Tooltip("Down-track speed the dense reward peaks at, m/s.")]
        public float targetSpeed = 2.5f;
        [Tooltip("Fraction of the control costs charged at difficulty 0.")]
        [Range(0f, 1f)] public float controlCostFloor = 0.1f;

        private int m_Steps;
        private int m_EpisodeIndex;
        private StatsRecorder m_Stats;

        // Standalone eval builds have no python side channel, so a run whose
        // physics or difficulty was changed can only be evaluated faithfully if the
        // same values can be forced from the command line. Repeatable arg:
        // --env-param name=value. Training never passes it.
        private static Dictionary<string, float> s_CliOverrides;

        private static Dictionary<string, float> CliOverrides
        {
            get
            {
                if (s_CliOverrides == null)
                {
                    s_CliOverrides = new Dictionary<string, float>();
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
                            Debug.Log($"CrawlerParkour: CLI env-param override {kv[0]}={f}");
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

        private void Start()
        {
            m_Stats = Academy.Instance.StatsRecorder;
            if (obstacleSource != null)
            {
                obstacleSource.Track = track;
                if (agent != null) obstacleSource.Reference = agent.body;
            }
            ResetEpisode();
        }

        private void LoadEnvironmentParameters()
        {
            difficulty = Mathf.Clamp01(GetParam("difficulty", difficulty));
            maxEpisodeSteps = Mathf.RoundToInt(GetParam("max_episode_steps", maxEpisodeSteps));
            progressWeight = GetParam("progress_weight", progressWeight);
            respawnPenalty = GetParam("respawn_penalty", respawnPenalty);
            finishBonus = GetParam("finish_bonus", finishBonus);
            energyCostWeight = GetParam("energy_cost_weight", energyCostWeight);
            actionRateCostWeight = GetParam("action_rate_cost_weight", actionRateCostWeight);
            fallDepth = GetParam("fall_depth", fallDepth);
            velocityWeight = GetParam("velocity_weight", velocityWeight);
            targetSpeed = GetParam("target_speed", targetSpeed);
            controlCostFloor = Mathf.Clamp01(GetParam("control_cost_floor", controlCostFloor));
        }

        private void ResetEpisode()
        {
            LoadEnvironmentParameters();

            int seed = fixedSeed >= 0 ? fixedSeed : Random.Range(0, int.MaxValue);
            track.Generate(seed, difficulty);

            agent.track = track;
            agent.progressWeight = progressWeight;
            agent.respawnPenalty = respawnPenalty;
            agent.finishBonus = finishBonus;
            agent.velocityWeight = velocityWeight;
            agent.targetSpeed = targetSpeed;
            agent.MaxStepOverride = maxEpisodeSteps;

            // Control costs ramp in with the curriculum instead of being charged in
            // full from step 1. They are what made stillness pay in run 002: before
            // a gait exists they are the only term with a reliable sign, so the
            // cheapest policy is to stop moving. Deferring them until there is a
            // gait to make efficient is the point -- lesson `flat` charges
            // controlCostFloor of the full price, lesson `max` charges all of it.
            //
            // Derived from `difficulty` rather than given its own curriculum block:
            // two curricula on the same measure advance independently and can
            // desync, and this way the ramp is visible in the Difficulty stat we
            // already log.
            float costScale = Mathf.Lerp(controlCostFloor, 1f, difficulty);
            agent.energyCostWeight = energyCostWeight * costScale;
            agent.actionRateCostWeight = actionRateCostWeight * costScale;

            // Track space: the generator lays everything out relative to its own
            // root, so a replicated arena is a pure translation of this one.
            float laneX = track.LaneCenterAt(1f);
            float y = track.GroundHeightAt(laneX, 1f, out float g) ? g + 1.2f : 1.2f;
            // The agent applies this inside OnEpisodeBegin, after BodyPart.Reset
            // has restored its recorded world transforms -- placing it from here
            // would just be undone.
            agent.SpawnPoint = track.ToWorld(new Vector3(laneX, y, 1f));

            m_Steps = 0;
            m_EpisodeIndex++;
        }

        private void FixedUpdate()
        {
            m_Steps++;

            if (agent.HasFallen(fallDepth))
            {
                agent.RespawnAtCheckpoint();
            }

            if (agent.Finished || m_Steps >= maxEpisodeSteps)
            {
                // Stats first: EndEpisode runs OnEpisodeBegin synchronously and
                // clears the per-episode accumulators.
                RecordStats();
                // Then set up the next track, because EndEpisode's OnEpisodeBegin
                // is what places the agent -- it needs the new spawn point to
                // already be there.
                ResetEpisode();
                agent.EndEpisode();
            }
        }

        private void RecordStats()
        {
            if (m_Stats == null) return;
            float len = Mathf.Max(1f, track.TrackLength);
            m_Stats.Add("CrawlerParkour/ProgressFraction", agent.MaxProgress / len);
            m_Stats.Add("CrawlerParkour/Finished", agent.Finished ? 1f : 0f);
            m_Stats.Add("CrawlerParkour/Respawns", agent.RespawnCount);
            m_Stats.Add("CrawlerParkour/Difficulty", difficulty);
            m_Stats.Add("CrawlerParkour/EnergyCost", agent.EpisodeEnergyCost);
            m_Stats.Add("CrawlerParkour/ActionRateCost", agent.EpisodeActionRateCost);
            m_Stats.Add("CrawlerParkour/VelocityReward", agent.EpisodeVelocityReward);
            // Read this before reward. Run 002's reward rose by a full point while
            // the crawler was motionless, so mean speed is the term that says
            // whether anything is actually happening.
            m_Stats.Add("CrawlerParkour/MeanForwardSpeed", agent.MeanForwardSpeed);
            // Steady non-zero repairs mean a pattern is emitting geometry its own
            // feasibility check rejects, and the track is quietly flattening.
            m_Stats.Add("CrawlerParkour/TrackRepairs", track.RepairCount);
            if (agent.Finished)
            {
                m_Stats.Add("CrawlerParkour/FinishSteps", m_Steps);
            }
        }
    }
}
