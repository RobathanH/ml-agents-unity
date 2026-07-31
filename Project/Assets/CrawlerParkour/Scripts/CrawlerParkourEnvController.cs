using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// Owns one parkour arena: regenerates the track each episode, drops the agent
    /// somewhere in the middle of it, respawns it when it falls, truncates the
    /// episode on the clock, and carries this arena's own terrain difficulty level.
    /// </summary>
    /// <remarks>
    /// Run 006 reframed this environment from "finish a 128m track" to "keep moving
    /// forward indefinitely, from any point in any track". Three things follow, and
    /// they are load-bearing for each other rather than independent tweaks:
    ///
    ///  1. There is no terminal state. Falling respawns; the clock truncates. So
    ///     the episode always ends with Agent.EpisodeInterrupted() and the trainer
    ///     always bootstraps V(s_T). Under run 005 this was EndEpisode(), which
    ///     forced a value target of 0 at a step nothing in the observation
    ///     identifies, for ~89% of episodes.
    ///  2. The episode is SHORT (30s, not 120s) and the spawn is mid-track. Run 005
    ///     covered 63.5m of a 128m track every episode -- but always the same first
    ///     63.5m, including 16m of guaranteed-flat lead-in, so a quarter of every
    ///     trajectory was flat by construction and the back half of the track was
    ///     never seen. What matters is obstacles per collected timestep, not per
    ///     episode, and a random mid-track spawn is what fixes that ratio.
    ///  3. Difficulty is PER ARENA and adapts on measured distance, replacing the
    ///     global reward-threshold ladder. See UpdateTerrainLevel.
    /// </remarks>
    public class CrawlerParkourEnvController : MonoBehaviour
    {
        public CrawlerParkourAgent agent;
        public ParkourTrackGenerator track;
        public ParkourObstacleSource obstacleSource;

        [Header("Episode")]
        public int maxEpisodeSteps = 1500;
        [Tooltip("Depth below local ground that counts as having fallen off.")]
        public float fallDepth = 3f;

        [Header("Spawn")]
        [Tooltip("Segments of runway that must remain ahead of the spawn point. The "
            + "episode must not be able to reach the end of the track.")]
        public int spawnReserveSegments = 6;
        [Tooltip("Longitudinal jitter around the chosen segment boundary, metres.")]
        public float spawnZJitter = 0.5f;
        [Tooltip("Heading randomisation at spawn, +/- degrees.")]
        public float spawnYawJitter = 30f;
        [Tooltip("Down-track speed the agent may start with, m/s.")]
        public float spawnSpeedJitter = 0.5f;

        [Header("Difficulty")]
        [Range(0f, 1f)] public float difficulty = 0f;
        [Tooltip("Level this arena starts at, 0..Levels-1. Arenas randomise up to "
            + "this so the population covers a spread from step 1.")]
        public int initialLevel = 5;
        [Tooltip("Distance rate (m/s) at or above which this arena moves up a level. "
            + "Only used when promoteFraction is not positive; see UpdateTerrainLevel.")]
        public float promoteSpeed = 0.5f;
        [Tooltip("Distance rate (m/s) below which this arena moves down a level. "
            + "Only used when promoteFraction is not positive.")]
        public float demoteSpeed = 0.15f;
        [Tooltip("Promote when the arena beats this fraction of ReferenceRate at its "
            + "CURRENT level. Positive enables per-rung gates; zero or below falls "
            + "back to the flat promoteSpeed/demoteSpeed pair.")]
        public float promoteFraction = -1f;
        [Tooltip("Demote below this fraction of ReferenceRate at the current level. "
            + "Set from the measured spread by scripts/check_curriculum_gates.py, not "
            + "by eye -- a gate at the median promotes half the time and demotes "
            + "almost never, which is a ratchet however it was derived.")]
        public float demoteFraction = 0.85f;
        [Tooltip("Negative: adapt. Zero or above: pin difficulty here and freeze the "
            + "curriculum. Eval builds use this.")]
        public float difficultyPin = -1f;
        [Tooltip("Fixed seed for reproducible eval. Negative means random per episode.")]
        public int fixedSeed = -1;

        [Header("Rewards")]
        [Tooltip("Reward per metre of new ground covered.")]
        public float progressPerMeter = 0.08f;
        public float respawnPenalty = 0.5f;
        public float energyCostWeight = 0f;
        public float actionRateCostWeight = 0f;
        [Tooltip("Reward per second of perfect down-track locomotion.")]
        public float velocityPerSecond = 0.0167f;
        [Tooltip("Down-track speed the dense reward peaks at, m/s.")]
        public float targetSpeed = 2.5f;
        [Tooltip("Fraction of the control costs charged at difficulty 0.")]
        [Range(0f, 1f)] public float controlCostFloor = 0.1f;

        /// <summary>
        /// Rungs on the per-arena difficulty ladder. Level L maps to difficulty
        /// L/(Levels-1), so the ladder spans the same 0..1 scalar every obstacle
        /// bound is already interpolated from -- this is a finer-grained version of
        /// run 005's five named lessons, not a different axis.
        /// </summary>
        public const int Levels = 10;

        /// <summary>
        /// Distance rate (m/s) a competent policy achieves at each rung. The promote
        /// and demote gates are FRACTIONS of this, not absolute speeds.
        /// </summary>
        /// <remarks>
        /// Run 006 and run 007 both used one global promote_speed of 0.5 m/s, and both
        /// ratcheted every arena to the ceiling, because 0.5 m/s is trivial on flat and
        /// hard at difficulty 1.0. No single number can hold a population in the middle
        /// of a ladder: whatever value is chosen is either below the policy at every rung
        /// (ratchets up) or above it at every rung (strands). The gate has to be
        /// expressed per rung -- DESIGN 8.9.
        ///
        /// These come from run 007's final policy, measured over its last 4M steps:
        /// L6 0.834, L7 0.667, L8 0.613, L9 0.537 m/s. Levels 0-5 have no final-policy
        /// measurement, because once the ratchet turned no arena went back down there,
        /// so the curve below L6 is the least-squares extrapolation of those four points
        /// (1.3715 - 0.0945*L). That extrapolation is the weakest part of this and it is
        /// self-correcting: with per-rung gates the population spreads, so the next run
        /// produces RateAtLevel data at every rung and the curve can be replaced with
        /// measurement.
        ///
        /// The curve is deliberately a property of the ENVIRONMENT, not of a checkpoint:
        /// it says "this is what this terrain costs", so the same numbers stay meaningful
        /// when the policy changes. Re-derive it only when the terrain generator changes.
        /// </remarks>
        public static readonly float[] ReferenceRate =
        {
            1.372f, 1.277f, 1.183f, 1.088f, 0.994f,
            0.899f, 0.834f, 0.667f, 0.613f, 0.537f,
        };

        private int m_Steps;
        private int m_EpisodeIndex;
        private int m_Level;
        private StatsRecorder m_Stats;
        private System.Random m_SpawnRng;

        // Precomputed so RecordStats does no per-episode string work.
        private static readonly string[] k_ClearedKeys = BuildPatternKeys("Cleared");
        private static readonly string[] k_FallKeys = BuildPatternKeys("Falls");

        /// <summary>
        /// Distance rate broken out per terrain level.
        /// </summary>
        /// <remarks>
        /// Not a nice-to-have. Every tag ml-agents writes is a windowed MEAN over
        /// whatever the population was doing, and a per-arena curriculum guarantees
        /// the population is spread across levels -- so a marginal DistanceRate mixes
        /// arenas on flat ground with arenas on level 9 and moves whenever the level
        /// distribution moves, independently of whether the policy got better. That
        /// is DESIGN.md 5.2's corollary (mean forward speed stops being a progress
        /// measure once the population splits), made permanent by construction.
        /// Conditioning on the level is the only way to read this run honestly, and
        /// it cannot be recovered after the fact from the marginal tags.
        /// </remarks>
        private static readonly string[] k_RateAtLevelKeys = BuildLevelKeys();

        private static string[] BuildLevelKeys()
        {
            var keys = new string[Levels];
            for (int i = 0; i < Levels; i++) keys[i] = $"CrawlerParkour/RateAtLevel/{i}";
            return keys;
        }

        private static string[] BuildPatternKeys(string group)
        {
            var names = System.Enum.GetNames(typeof(SegmentPattern));
            // The agent sizes its counter arrays from a hardcoded PatternCount, and
            // both it and this index by (int)SegmentPattern. Adding a pattern
            // without bumping that constant would silently drop the new pattern's
            // stats and, worse, leave the counters one short of the cast -- so this
            // is checked rather than assumed. Enum.GetNames orders by value, which
            // for a 0..N-1 sequential enum is the same as (int) order.
            if (names.Length != CrawlerParkourAgent.PatternCount)
            {
                Debug.LogError(
                    $"SegmentPattern has {names.Length} values but "
                    + $"CrawlerParkourAgent.PatternCount is {CrawlerParkourAgent.PatternCount}. "
                    + "Per-pattern traversal stats will be wrong until they agree.");
            }
            var keys = new string[names.Length];
            for (int i = 0; i < names.Length; i++) keys[i] = $"CrawlerParkour/{group}/{names[i]}";
            return keys;
        }

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
            LoadEnvironmentParameters();
            // Arenas start SPREAD across the ladder rather than all on level 0.
            // Every arena starting at the same rung makes the whole population move
            // in lockstep, which is the failure mode the per-arena curriculum exists
            // to avoid: one bad promotion then strands all 32 at once, which is
            // exactly what a single global threshold did to run 005. A spread also
            // means there is data at every level from the first summary window, so
            // speed-vs-level is readable immediately instead of after the ladder has
            // been climbed.
            m_Level = Random.Range(0, Mathf.Clamp(initialLevel, 0, Levels - 1) + 1);
            ResetEpisode();
        }

        private void LoadEnvironmentParameters()
        {
            maxEpisodeSteps = Mathf.RoundToInt(GetParam("max_episode_steps", maxEpisodeSteps));
            // Reachable from the CLI so two checkpoints can be compared on the SAME
            // track. Without it a side-by-side is two policies on two different
            // random layouts, which reads as a difference in skill and is not one.
            // Negative (the default) keeps the per-episode randomisation training
            // needs, so this is inert unless an eval asks for it.
            fixedSeed = Mathf.RoundToInt(GetParam("fixed_seed", fixedSeed));
            progressPerMeter = GetParam("progress_per_meter", progressPerMeter);
            respawnPenalty = GetParam("respawn_penalty", respawnPenalty);
            energyCostWeight = GetParam("energy_cost_weight", energyCostWeight);
            actionRateCostWeight = GetParam("action_rate_cost_weight", actionRateCostWeight);
            fallDepth = GetParam("fall_depth", fallDepth);
            velocityPerSecond = GetParam("velocity_per_second", velocityPerSecond);
            targetSpeed = GetParam("target_speed", targetSpeed);
            controlCostFloor = Mathf.Clamp01(GetParam("control_cost_floor", controlCostFloor));

            initialLevel = Mathf.RoundToInt(GetParam("initial_level", initialLevel));
            promoteSpeed = GetParam("promote_speed", promoteSpeed);
            demoteSpeed = GetParam("demote_speed", demoteSpeed);
            promoteFraction = GetParam("promote_fraction", promoteFraction);
            demoteFraction = GetParam("demote_fraction", demoteFraction);
            difficultyPin = GetParam("difficulty_pin", difficultyPin);
            spawnReserveSegments = Mathf.RoundToInt(
                GetParam("spawn_reserve_segments", spawnReserveSegments));
            spawnYawJitter = GetParam("spawn_yaw_jitter", spawnYawJitter);
            spawnSpeedJitter = GetParam("spawn_speed_jitter", spawnSpeedJitter);
            // Long eval rollouts need more track than training does -- at 2.5 m/s a
            // 140s clip covers 350m against a 128m training track. Reachable from
            // the CLI so the eval build can ask for it without a rebuild.
            track.SegmentCount = Mathf.RoundToInt(GetParam("segment_count", track.SegmentCount));
            track.MaxBoxes = Mathf.RoundToInt(GetParam("max_boxes", track.MaxBoxes));
        }

        /// <summary>Difficulty this arena is currently running, from its level.</summary>
        private float LevelDifficulty =>
            difficultyPin >= 0f ? Mathf.Clamp01(difficultyPin)
                                : m_Level / (float)(Levels - 1);

        private float EpisodeSeconds => Mathf.Max(1e-3f, maxEpisodeSteps * Time.fixedDeltaTime);

        /// <summary>
        /// Moves THIS ARENA up or down one rung on the terrain ladder, on the
        /// distance rate it just measured.
        /// </summary>
        /// <remarks>
        /// This replaces the global reward-threshold curriculum, and the reason is
        /// run 005. That run set a threshold of 17.0 to leave the `moderate` lesson;
        /// the policy scored 16.04 on FLAT, the easiest terrain in the game, so the
        /// gate was above anything it could ever reach and the run was lost at
        /// launch. That failure is structural, not a bad number: one global scalar
        /// gate, against a reward whose scale moves whenever episode length, track
        /// length or any weight moves.
        ///
        /// A per-arena level has no threshold to mis-calibrate. It is measured in
        /// m/s, which is a physical quantity that does not move when the reward is
        /// re-weighted; each arena finds its own rung; and the population spreads
        /// across the ladder, so the policy keeps seeing easy terrain instead of
        /// having it taken away. This is Rudin et al.'s game-inspired curriculum
        /// (promote on distance travelled, demote on failing to cover the commanded
        /// distance), with distance RATE in place of raw distance so the rule
        /// survives a change of episode length.
        ///
        /// The deadband between promote and demote is wide on purpose: adjacent
        /// rungs differ by 1/9 of the difficulty scalar, and a narrow band would
        /// oscillate an arena every episode on nothing but per-episode variance.
        /// </remarks>
        private void UpdateTerrainLevel()
        {
            if (difficultyPin >= 0f) return;
            float rate = agent.DistanceCovered / EpisodeSeconds;
            GetGates(m_Level, out float promote, out float demote);
            if (rate >= promote) m_Level = Mathf.Min(m_Level + 1, Levels - 1);
            else if (rate < demote) m_Level = Mathf.Max(m_Level - 1, 0);
        }

        /// <summary>
        /// Promote and demote rates for a given rung. Per-rung when promoteFraction is
        /// positive, otherwise the flat pair runs 006 and 007 used.
        /// </summary>
        /// <remarks>
        /// The flat path is kept so those two runs stay reproducible from their own
        /// configs -- a config that silently means something different than it did when
        /// it ran destroys the comparison the whole run existed to make.
        /// </remarks>
        public void GetGates(int level, out float promote, out float demote)
        {
            if (promoteFraction <= 0f)
            {
                promote = promoteSpeed;
                demote = demoteSpeed;
                return;
            }
            float reference = ReferenceRate[Mathf.Clamp(level, 0, Levels - 1)];
            promote = promoteFraction * reference;
            demote = demoteFraction * reference;
        }

        private void ResetEpisode()
        {
            LoadEnvironmentParameters();
            difficulty = LevelDifficulty;

            int seed = fixedSeed >= 0 ? fixedSeed : Random.Range(0, int.MaxValue);
            track.Generate(seed, difficulty);
            // Derived from the track seed so a fixed_seed eval reproduces the spawn
            // as well as the layout, but XORed off it so the spawn is not correlated
            // with the generator's own stream.
            m_SpawnRng = new System.Random(seed ^ 0x5f3759df);

            agent.track = track;
            agent.progressPerMeter = progressPerMeter;
            agent.respawnPenalty = respawnPenalty;
            agent.velocityPerSecond = velocityPerSecond;
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
            //
            // Mid-track by default, falling back to the old start-line spawn only if
            // the generator can find nowhere safe (a very short track, or one whose
            // box budget ran out). The fallback is deliberately loud in its
            // consequences rather than silent: it is the one case where run 006
            // degrades to run 005's start distribution.
            Vector3 ground;
            if (!track.TryPickSpawn(m_SpawnRng, spawnReserveSegments, spawnZJitter, out ground))
            {
                float laneX = track.LaneCenterAt(1f);
                float g0 = track.GroundHeightAt(laneX, 1f, out float g) ? g : 0f;
                ground = new Vector3(laneX, g0, 1f);
            }

            // The agent applies these inside OnEpisodeBegin, after BodyPart.Reset
            // has restored its recorded world transforms -- placing it from here
            // would just be undone.
            agent.SpawnPoint = track.ToWorld(ground + new Vector3(0f, 1.2f, 0f));
            agent.SpawnYawDegrees = (float)(m_SpawnRng.NextDouble() * 2.0 - 1.0) * spawnYawJitter;
            agent.SpawnVelocity = new Vector3(
                0f, 0f, (float)m_SpawnRng.NextDouble() * spawnSpeedJitter);

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

            if (m_Steps >= maxEpisodeSteps)
            {
                // Stats first: ending the episode runs OnEpisodeBegin synchronously
                // and clears the per-episode accumulators.
                RecordStats();
                // Level before the reset, because the reset reads it to build the
                // next track.
                UpdateTerrainLevel();
                // Then set up the next track, because OnEpisodeBegin is what places
                // the agent -- it needs the new spawn point to already be there.
                ResetEpisode();

                // EpisodeInterrupted, NOT EndEpisode. Run 006 has no terminal state
                // at all: falling respawns and the clock truncates, so every episode
                // ends here. EndEpisode would tell ppo/trainer.py (L98,
                // `done_reached and not interrupted`) that this is a true terminal
                // and force a value target of 0 -- at a step nothing in the
                // observation identifies, from a state indistinguishable from one
                // where the agent keeps running. Interrupting bootstraps V(s_T)
                // instead, which is what makes the value function look ahead
                // indefinitely rather than to the end of the clock.
                //
                // This is Pardo et al.'s case (ii): the time limit is a training
                // convenience, not part of the task. The corollary is that remaining
                // time must NOT be in the observation -- it is not, and removing the
                // finish bonus took the clock out of the reward too, so the two
                // halves of that fix agree.
                agent.EpisodeInterrupted();
            }
        }

        private void RecordStats()
        {
            if (m_Stats == null) return;

            // Headline. Metres per second of NEW ground, which is the whole task
            // stated as one number. Scale-free, so it is comparable across runs 005
            // and 006 despite the reward re-parameterisation, and across training
            // (30s episodes) and eval (140s rollouts).
            float dist = agent.DistanceCovered;
            float rate = dist / EpisodeSeconds;
            m_Stats.Add("CrawlerParkour/DistanceRate", rate);
            m_Stats.Add("CrawlerParkour/DistanceCovered", dist);
            // The curriculum's own progress curve, replacing Lesson Number.
            m_Stats.Add("CrawlerParkour/TerrainLevel", m_Level);
            // The gate actually in force this episode. Recorded because runs 006 and
            // 007 both had a promote gate below the policy's rate at every rung and
            // neither run could show it without going back to the config; with this
            // logged, "did the curriculum have anything to say" is one chart.
            GetGates(m_Level, out float promote, out float demote);
            m_Stats.Add("CrawlerParkour/PromoteGate", promote);
            m_Stats.Add("CrawlerParkour/RateMinusGate", rate - promote);
            // ...and the same rate conditioned on the level it was earned at, which
            // is the only form of it that is not confounded by the level
            // distribution. See k_RateAtLevelKeys.
            if (m_Level >= 0 && m_Level < k_RateAtLevelKeys.Length)
            {
                m_Stats.Add(k_RateAtLevelKeys[m_Level], rate);
            }
            m_Stats.Add("CrawlerParkour/Difficulty", difficulty);

            m_Stats.Add("CrawlerParkour/Respawns", agent.RespawnCount);
            // Per distance, not per episode: the raw count conflates "falls a lot"
            // with "survived long enough to reach anything worth falling off".
            if (dist > 1f) m_Stats.Add("CrawlerParkour/RespawnsPer100m", 100f * agent.RespawnCount / dist);

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

            // Per-pattern traversal. This is the measure that says whether the
            // policy is doing parkour or walking the flat stretches between
            // obstacles -- a single finish rate had to stand in for all ten patterns
            // and could not tell those apart.
            //
            // Emitting `entered` samples each worth `count/entered` makes the
            // WINDOW MEAN the pooled rate across every episode and arena in the
            // window (sum of counts / sum of entries), rather than an average of
            // per-episode ratios that would weight a one-segment episode as heavily
            // as a ten-segment one.
            int patterns = Mathf.Min(CrawlerParkourAgent.PatternCount, k_ClearedKeys.Length);
            for (int p = 0; p < patterns; p++)
            {
                int entered = agent.SegmentsEntered[p];
                if (entered <= 0) continue;
                float clearedShare = agent.SegmentsCleared[p] / (float)entered;
                float fallShare = agent.SegmentFalls[p] / (float)entered;
                for (int i = 0; i < entered; i++)
                {
                    m_Stats.Add(k_ClearedKeys[p], clearedShare);
                    m_Stats.Add(k_FallKeys[p], fallShare);
                }
            }
        }
    }
}
