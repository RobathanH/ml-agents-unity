using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CrawlerParkour;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Builds every CrawlerParkour asset -- layer, physics material, box prefab,
/// arena prefab, single- and multi-arena scenes -- from code.
///
/// Invoke from CLI:
///   Unity.exe -projectPath &lt;Project&gt; -batchmode -nographics
///            -executeMethod CrawlerParkourBuilder.BuildAll -logFile &lt;log&gt;
/// </summary>
/// <remarks>
/// Written as a script rather than done by hand in the inspector because the
/// wiring has several failure modes that are invisible once they are wrong: an
/// EGNN subtype string that silently mints extra one-hot slots, an observation
/// count that disagrees with the config by a few floats, a ground mask that
/// reads the crawler's own legs. Each of those trains happily and produces
/// nonsense. Here they are derived from the same constants the runtime uses, so
/// they cannot drift apart, and <see cref="Report"/> prints the numbers the
/// training config has to match.
/// </remarks>
public static class CrawlerParkourBuilder
{
    const string Root = "Assets/CrawlerParkour";
    const string PrefabDir = Root + "/Prefabs";
    const string SceneDir = Root + "/Scenes";
    const string MaterialDir = Root + "/Materials";
    const string BoxPrefabPath = PrefabDir + "/ParkourBox.prefab";
    const string EnvPrefabPath = PrefabDir + "/CrawlerParkourEnv.prefab";
    const string ScenePath = SceneDir + "/CrawlerParkour.unity";
    const string MultiScenePath = SceneDir + "/CrawlerParkour_Multi.unity";
    const string CrawlerPrefabPath = "Assets/ML-Agents/Examples/Crawler/Prefabs/Crawler.prefab";

    const string TrackLayer = "ParkourTrack";
    // GroundContact only flips touchingGround for colliders carrying this tag, so
    // without it the four foot-contact observations are permanently zero.
    const string GroundTag = "ground";
    const string BehaviorName = "CrawlerParkour";

    const int MaxObstacles = 16;
    // Mirrors spawn_reserve_segments in config/ppo/CrawlerParkour.yaml, so the
    // verifier samples the same spawn distribution training will.
    const int SpawnReserveSegments = 10;
    const int MultiArenaCount = 32;
    const int ArenaGridWidth = 4;
    // Track is 128 long (16 segments x 8) and at most 12 wide. Spacing has to
    // clear that plus overrun; a crawler that falls off falls into open space
    // rather than onto a neighbour, because no arena has a floor outside its own
    // track.
    const float ArenaSpacingX = 60f;
    const float ArenaSpacingZ = 220f;

    [MenuItem("Training/CrawlerParkour/Build Assets + Scenes")]
    public static void BuildAll()
    {
        var ok = false;
        try
        {
            Directory.CreateDirectory(PrefabDir);
            Directory.CreateDirectory(SceneDir);
            Directory.CreateDirectory(MaterialDir);
            AssetDatabase.Refresh();

            var layer = EnsureLayer(TrackLayer);
            var physics = EnsurePhysicsMaterial();
            var floorMat = EnsureMaterial("ParkourFloor", new Color(0.35f, 0.38f, 0.42f));
            var obstacleMat = EnsureMaterial("ParkourObstacle", new Color(0.72f, 0.45f, 0.20f));
            var boxPrefab = BuildBoxPrefab(layer, physics);
            var envPrefab = BuildEnvPrefab(boxPrefab, floorMat, obstacleMat, layer);

            BuildScene(envPrefab, ScenePath, 1);
            BuildScene(envPrefab, MultiScenePath, MultiArenaCount);

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Report(envPrefab);
            ok = true;
        }
        catch (Exception e)
        {
            Debug.LogError($"CrawlerParkourBuilder failed: {e}");
        }
        ExitIfBatch(ok);
    }

    [MenuItem("Training/CrawlerParkour/Build Multi-Arena (Linux x86_64)")]
    public static void BuildMultiLinux()
    {
        ExitIfBatch(BuildPlayer(
            MultiScenePath, "CrawlerParkour_Multi_linux", "CrawlerParkour.x86_64",
            BuildTarget.StandaloneLinux64));
    }

    [MenuItem("Training/CrawlerParkour/Build Viewer (Windows)")]
    public static void BuildViewerWindows()
    {
        ExitIfBatch(BuildPlayer(
            ScenePath, "CrawlerParkour_viewer_win", "CrawlerParkour.exe",
            BuildTarget.StandaloneWindows64));
    }

    // ---------------------------------------------------------------- project

    /// <summary>
    /// Reserves a layer for track geometry. The height-field raycasts mask to it
    /// so they cannot return the crawler's own legs, which would turn the whole
    /// 126-float terrain observation into a reading of its current pose.
    /// </summary>
    static int EnsureLayer(string name)
    {
        var existing = LayerMask.NameToLayer(name);
        if (existing >= 0)
        {
            return existing;
        }

        var assets = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset");
        if (assets == null || assets.Length == 0)
        {
            throw new Exception("TagManager.asset not readable");
        }
        var so = new SerializedObject(assets[0]);
        var layers = so.FindProperty("layers");
        for (var i = 8; i < layers.arraySize; i++)
        {
            var slot = layers.GetArrayElementAtIndex(i);
            if (string.IsNullOrEmpty(slot.stringValue))
            {
                slot.stringValue = name;
                so.ApplyModifiedProperties();
                AssetDatabase.SaveAssets();
                Debug.Log($"Reserved layer {i} for '{name}'");
                return i;
            }
        }
        throw new Exception("No free user layer available");
    }

    /// <summary>
    /// Ground and obstacle friction. Run 008 doubled this from 0.8; the
    /// generator's steepest ramp is 40 degrees against a walkable ceiling of
    /// arctan(mu), so it is part of the feasibility guarantee rather than a
    /// cosmetic choice.
    /// </summary>
    /// <remarks>
    /// Read from the AGENT rather than declared here. Run 008's note said this
    /// number lives in two places -- this builder and the checked-in
    /// .physicMaterial asset, which a rebuild silently overwrites -- and run 009
    /// gives it a third reader: the foot friction range is expressed as a
    /// multiplier on it (CrawlerParkourAgent.FootFrictionMax), so the two must
    /// agree or the "0 to twice run 008's contact" arithmetic is simply wrong.
    /// One constant, and the asset is derived from it.
    /// </remarks>
    const float GroundFriction = CrawlerParkourAgent.GroundFriction;

    // -------------------------------------------------------------- rig physics
    //
    // RUN 008 SET THESE BY HAND ON THE PREFAB AND THE BUILDER DID NOT KNOW.
    // BuildEnvPrefab re-instantiates the example Crawler on every run, and the
    // example ships 40000/5000/20000 -- so `BuildAll` would have quietly halved the
    // joint drive of the run under test and rewritten the prefab with it, and the
    // only evidence would have been a policy that suddenly could not climb. The
    // values are here now, applied explicitly, for the same reason GroundFriction is.
    const float MaxJointSpring = 80000f;
    const float JointDampen = 10000f;
    const float MaxJointForceLimit = 40000f;

    // ----------------------------------------------------------- joint ranges
    //
    // RUN 009. The example crawler's legs are hinged for walking on a floor, the
    // right way up, and the limits say so: the hip swings [-60, 0] about X and the
    // knee bends [0, 150]. Both ranges sit entirely on ONE SIDE of the joint's
    // zero -- and a ConfigurableJoint's zero is the pose the rig was authored in --
    // so the animal is mechanically incapable of doing anything on its back that it
    // can do on its feet, and incapable of folding a leg in under its body at all.
    // Every recovery from a tumble had to be a re-flip.
    //
    // Making each range symmetric about zero is what buys both: the leg can be
    // driven as far one way as the other, so an inverted crawler has exactly the
    // workspace an upright one has, and a hip that can reach +90 as well as -90 can
    // tuck the whole limb underneath the torso.
    //
    // THE COST IS RESOLUTION, and it is worth stating plainly rather than
    // discovering. SetJointTargetRotation lerps the action across the limit range,
    // so tripling the hip's X range divides the angular precision of a unit of
    // policy output by three. Foot placement on a Beam is exactly the skill that
    // needs precision. Y is widened less than X for that reason -- the splay only
    // has to reach under the body, not around it.
    const float HipLowXLimit = -90f;
    const float HipHighXLimit = 90f;
    const float HipYLimit = 45f;
    const float KneeLowXLimit = -150f;
    const float KneeHighXLimit = 150f;

    static PhysicsMaterial EnsurePhysicsMaterial()
    {
        // The class was renamed in Unity 6 but the asset extension this editor
        // actually imports is still .physicMaterial, which is also what the rest of
        // the project uses. The others are fallbacks for a future editor that
        // changes its mind; a rejected extension is deleted rather than left behind
        // as a duplicate asset.
        foreach (var ext in new[] { ".physicMaterial", ".physicsMaterial", ".asset" })
        {
            var path = MaterialDir + "/ParkourGround" + ext;
            var existing = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
            if (existing != null)
            {
                existing.dynamicFriction = GroundFriction;
                existing.staticFriction = GroundFriction;
                existing.bounciness = 0f;
                EditorUtility.SetDirty(existing);
                return existing;
            }

            var mat = new PhysicsMaterial("ParkourGround")
            {
                dynamicFriction = GroundFriction,
                staticFriction = GroundFriction,
                bounciness = 0f,
            };
            try
            {
                AssetDatabase.CreateAsset(mat, path);
                AssetDatabase.SaveAssets();
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport);
                var loaded = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(path);
                if (loaded != null)
                {
                    Debug.Log($"Created physics material {path}");
                    return loaded;
                }
                AssetDatabase.DeleteAsset(path);
            }
            catch (Exception e)
            {
                Debug.Log($"Physics material extension '{ext}' rejected ({e.Message}); trying next");
                AssetDatabase.DeleteAsset(path);
            }
        }
        throw new Exception("Could not create the track physics material");
    }

    static Material EnsureMaterial(string name, Color color)
    {
        var path = $"{MaterialDir}/{name}.mat";
        var existing = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (existing != null)
        {
            existing.color = color;
            EditorUtility.SetDirty(existing);
            return existing;
        }
        var shader = Shader.Find("Standard") ?? Shader.Find("Diffuse");
        var mat = new Material(shader) { color = color };
        AssetDatabase.CreateAsset(mat, path);
        return mat;
    }

    // ----------------------------------------------------------------- assets

    /// <summary>
    /// The single obstacle primitive. ParkourBox.Place writes
    /// localScale = halfExtents * 2, so this must be exactly one unit per side or
    /// every obstacle in the environment is the wrong size -- and the feasibility
    /// harness, which works from the half-extents rather than the transform,
    /// would still report the track as traversable.
    /// </summary>
    static GameObject BuildBoxPrefab(int layer, PhysicsMaterial physics)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        try
        {
            go.name = "ParkourBox";
            go.layer = layer;
            go.tag = GroundTag;
            go.transform.localScale = Vector3.one;
            var col = go.GetComponent<BoxCollider>();
            col.size = Vector3.one;
            col.center = Vector3.zero;
            col.sharedMaterial = physics;
            var saved = PrefabUtility.SaveAsPrefabAsset(go, BoxPrefabPath);
            Debug.Log($"Box prefab -> {BoxPrefabPath}");
            return saved;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    static GameObject BuildEnvPrefab(
        GameObject boxPrefab, Material floorMat, Material obstacleMat, int layer)
    {
        var root = new GameObject("CrawlerParkourEnv");
        try
        {
            var track = new GameObject("Track");
            track.transform.SetParent(root.transform, false);
            var boxes = new GameObject("Boxes");
            boxes.transform.SetParent(track.transform, false);

            var crawler = InstantiateCrawler();
            crawler.transform.SetParent(root.transform, false);
            crawler.transform.localPosition = new Vector3(0f, 1.2f, 1f);

            var agent = ConfigureCrawler(crawler, layer);
            var body = agent.body.gameObject;

            var generator = track.AddComponent<ParkourTrackGenerator>();
            generator.BoxPrefab = boxPrefab;
            generator.BoxParent = boxes.transform;
            generator.FloorMaterial = floorMat;
            generator.ObstacleMaterial = obstacleMat;
            // Set explicitly rather than left to the field initialiser, so the
            // value is visible in the prefab YAML instead of depending on what
            // Unity does with a field the serialised data predates. 4m against a
            // deepest start footprint edge of z=-1.47 (the first checkpoint
            // respawn) leaves ~2.5m to stagger backwards into.
            generator.StartApron = 4f;
            generator.FinishApron = 4f;

            var source = track.AddComponent<ParkourObstacleSource>();
            source.Track = generator;
            source.Reference = agent.body;
            source.MaxObstacles = MaxObstacles;

            agent.track = generator;
            ConfigureSensor(crawler, root, body, source);

            var controller = root.AddComponent<CrawlerParkourEnvController>();
            controller.agent = agent;
            controller.track = generator;
            controller.obstacleSource = source;
            // Defaults mirror config/ppo/CrawlerParkour.yaml. environment_parameters
            // override them during training; these are what a standalone eval or
            // viewer build runs with.
            controller.maxEpisodeSteps = 1500;
            controller.fallDepth = 3f;
            controller.difficulty = 0f;
            controller.fixedSeed = -1;
            controller.progressPerMeter = 0.08f;
            controller.velocityPerSecond = 0.0167f;
            controller.targetSpeed = 2.5f;
            controller.airbornePerSecond = 0.05f;
            controller.respawnPenalty = 0.5f;
            controller.energyCostWeight = 0.002f;
            controller.actionRateCostWeight = 0.002f;
            controller.controlCostFloor = 0.1f;
            controller.initialLevel = 5;
            controller.promoteSpeed = 0.5f;
            controller.demoteSpeed = 0.15f;
            // Set here as well as in the config from run 009 on. Left at the field
            // initialiser (-1) a standalone build silently falls back to the flat
            // promote/demote pair that ratcheted runs 006 and 007 to the ceiling, so
            // any eval that let the curriculum run -- rather than pinning it -- would
            // be watching a different curriculum than the one that trained.
            controller.promoteFraction = 1.15f;
            controller.demoteFraction = 0.85f;
            controller.referenceRateScale = 1f;
            // Adaptive by default. An eval or viewer build pins the terrain with
            // --env-param difficulty_pin=<d>, which is the only way a standalone
            // build (no python side channel) can be made to run the terrain the
            // training run was actually on.
            controller.difficultyPin = -1f;
            controller.spawnReserveSegments = 6;
            controller.spawnZJitter = 0.5f;
            controller.spawnYawJitter = 30f;
            controller.spawnSpeedJitter = 0.5f;

            var saved = PrefabUtility.SaveAsPrefabAsset(root, EnvPrefabPath);
            Debug.Log($"Arena prefab -> {EnvPrefabPath}");
            return saved;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }
    }

    static GameObject InstantiateCrawler()
    {
        var source = AssetDatabase.LoadAssetAtPath<GameObject>(CrawlerPrefabPath);
        if (source == null)
        {
            throw new Exception($"Crawler prefab not found at {CrawlerPrefabPath}");
        }
        var instance = (GameObject)PrefabUtility.InstantiatePrefab(source);
        // Unpacked rather than nested: the parkour crawler strips components and
        // rewrites GroundContact flags, and as prefab overrides those would be
        // silently reverted by any future upstream edit to the example.
        PrefabUtility.UnpackPrefabInstance(
            instance, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
        instance.name = "Crawler";
        return instance;
    }

    static CrawlerParkourAgent ConfigureCrawler(GameObject crawler, int layer)
    {
        // DecisionRequester first, and it is not optional to do so.
        // [RequireComponent(typeof(Agent))] makes Unity REFUSE to destroy the
        // example CrawlerAgent while a DecisionRequester is present -- it logs
        // "Can't remove CrawlerAgent (Script) because DecisionRequester (Script)
        // depends on it" and carries on, so the build still "succeeds" with two
        // Agent components on one GameObject. That is illegal in ML-Agents and
        // shows up only at runtime, as a NullReferenceException in
        // AgentInfo.CopyActions, after the trainer has already been paid for.
        // The requester is re-added further down, once our agent exists.
        foreach (var stale in crawler.GetComponents<DecisionRequester>())
        {
            UnityEngine.Object.DestroyImmediate(stale, true);
        }

        foreach (var component in crawler.GetComponents<Component>().ToArray())
        {
            // The example agent, and a RigidBodySensorComponent that would append
            // an entire observation block the training config knows nothing about.
            // ModelOverrider stays: it is what lets a standalone viewer build load
            // a checkpoint from the command line.
            if (component is Agent || component.GetType().Name == "RigidBodySensorComponent")
            {
                UnityEngine.Object.DestroyImmediate(component, true);
            }
        }

        // Assert it actually happened. DestroyImmediate reports refusal to the
        // log and returns void, so without this the failure is silent.
        var strays = crawler.GetComponents<Agent>();
        if (strays.Length > 0)
        {
            throw new Exception(
                $"{strays.Length} Agent component(s) survived removal on '{crawler.name}': "
                + string.Join(", ", strays.Select(a => a.GetType().Name))
                + ". Something still depends on them; a build with two Agents on one "
                + "GameObject fails at runtime in AgentInfo.CopyActions.");
        }

        var parts = crawler.GetComponentsInChildren<Transform>(true);
        Transform Find(string name)
        {
            var t = parts.FirstOrDefault(x => x.name == name);
            if (t == null)
            {
                throw new Exception($"Crawler body part '{name}' not found");
            }
            return t;
        }

        var agent = crawler.AddComponent<CrawlerParkourAgent>();
        agent.body = Find("Body");
        // The example names the upper segments leg0..3 and the lower ones
        // foreleg0..3; the agent's fields say Upper/Lower.
        agent.leg0Upper = Find("leg0");
        agent.leg1Upper = Find("leg1");
        agent.leg2Upper = Find("leg2");
        agent.leg3Upper = Find("leg3");
        agent.leg0Lower = Find("foreleg0");
        agent.leg1Lower = Find("foreleg1");
        agent.leg2Lower = Find("foreleg2");
        agent.leg3Lower = Find("foreleg3");
        agent.GroundMask = 1 << layer;

        // The example Crawler ends its episode when the body or an upper leg
        // touches the ground. Parkour needs the opposite: bellying under an
        // overhang and scrambling over a step are the behaviours the environment
        // exists to produce. Left alone this fires EndEpisode() behind the
        // controller's back, so its step counter and stats would describe
        // episodes that no longer exist.
        var cleared = 0;
        foreach (var contact in crawler.GetComponentsInChildren<Unity.MLAgentsExamples.GroundContact>(true))
        {
            if (contact.agentDoneOnGroundContact || contact.penalizeGroundContact)
            {
                cleared++;
            }
            contact.agentDoneOnGroundContact = false;
            contact.penalizeGroundContact = false;
            contact.groundContactPenalty = 0f;
        }
        Debug.Log($"Cleared terminating/penalising ground contact on {cleared} body part(s)");

        ConfigureJointDrive(crawler);
        WidenLegJoints(agent);

        var behavior = crawler.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = BehaviorName;
        behavior.BrainParameters.VectorObservationSize = VectorObservationSize(agent);
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        // From the agent's own constant, not a literal. A BehaviorParameters that
        // declares more actions than OnActionReceived reads is not an error at any
        // level -- the extra dimensions are simply sampled, costed by the action-rate
        // term, and ignored.
        behavior.BrainParameters.ActionSpec =
            ActionSpec.MakeContinuous(CrawlerParkourAgent.ActionCount);
        behavior.Model = null;
        behavior.BehaviorType = BehaviorType.Default;

        var requester = crawler.GetComponent<DecisionRequester>()
            ?? crawler.AddComponent<DecisionRequester>();
        requester.DecisionPeriod = 5;
        // The action-rate control cost assumes OnActionReceived fires exactly once
        // per decision; repeating actions between decisions would divide it by the
        // decision period without changing anything the agent did.
        requester.TakeActionsBetweenDecisions = false;

        return agent;
    }

    /// <summary>
    /// Deliberately not recomputed here. The count belongs next to the code that
    /// writes the observations; duplicating the arithmetic in the builder is what
    /// produced a six-float discrepancy in the first place.
    /// </summary>
    static int VectorObservationSize(CrawlerParkourAgent agent) => agent.ObservationCount;

    /// <summary>
    /// Applies run 008's joint gains, which until now only existed as a hand edit
    /// on the prefab this builder overwrites.
    /// </summary>
    static void ConfigureJointDrive(GameObject crawler)
    {
        var jd = crawler.GetComponent<Unity.MLAgentsExamples.JointDriveController>();
        if (jd == null)
        {
            throw new Exception(
                $"{crawler.name} has no JointDriveController; the example Crawler prefab "
                + "is expected to carry one and the agent requires it.");
        }
        jd.maxJointSpring = MaxJointSpring;
        jd.jointDampen = JointDampen;
        jd.maxJointForceLimit = MaxJointForceLimit;
        Debug.Log($"Joint drive: spring {MaxJointSpring}, dampen {JointDampen}, "
                  + $"force limit {MaxJointForceLimit}");
    }

    /// <summary>
    /// Makes every leg joint's angular range symmetric about its authored pose.
    /// </summary>
    /// <remarks>
    /// Only `limit` is written. `bounciness` and `contactDistance` are read back out
    /// of the existing SoftJointLimit and put back unchanged, because SoftJointLimit
    /// is a struct: assigning a fresh one would silently zero both, and a zeroed
    /// contactDistance means the solver only notices the limit once it has already
    /// been violated, which reads as a leg that jitters at full extension.
    ///
    /// Y motion is asserted rather than assumed to be free on the hips. The action
    /// space drives (x, y) on each upper leg, and if a future rig locks Y the policy
    /// would keep paying an action-rate cost for a dimension that moves nothing --
    /// the same class of silent no-op as an observation the config over-declares.
    /// </remarks>
    static void WidenLegJoints(CrawlerParkourAgent agent)
    {
        var hips = new[] { agent.leg0Upper, agent.leg1Upper, agent.leg2Upper, agent.leg3Upper };
        var knees = new[] { agent.leg0Lower, agent.leg1Lower, agent.leg2Lower, agent.leg3Lower };

        static SoftJointLimit With(SoftJointLimit existing, float limit)
        {
            existing.limit = limit;
            return existing;
        }

        foreach (var t in hips)
        {
            var j = t.GetComponent<ConfigurableJoint>();
            if (j == null) throw new Exception($"Hip '{t.name}' has no ConfigurableJoint");
            j.lowAngularXLimit = With(j.lowAngularXLimit, HipLowXLimit);
            j.highAngularXLimit = With(j.highAngularXLimit, HipHighXLimit);
            j.angularYLimit = With(j.angularYLimit, HipYLimit);
            if (j.angularYMotion != ConfigurableJointMotion.Limited)
            {
                throw new Exception(
                    $"Hip '{t.name}' has angularYMotion = {j.angularYMotion}, not Limited. "
                    + "The policy spends an action on this axis every decision; locked, "
                    + "that action is charged for and does nothing.");
            }
        }

        foreach (var t in knees)
        {
            var j = t.GetComponent<ConfigurableJoint>();
            if (j == null) throw new Exception($"Knee '{t.name}' has no ConfigurableJoint");
            j.lowAngularXLimit = With(j.lowAngularXLimit, KneeLowXLimit);
            j.highAngularXLimit = With(j.highAngularXLimit, KneeHighXLimit);
        }

        Debug.Log(
            $"Leg ranges: hip X [{HipLowXLimit}, {HipHighXLimit}] Y +/-{HipYLimit}, "
            + $"knee X [{KneeLowXLimit}, {KneeHighXLimit}] -- symmetric about the authored pose");
    }

    /// <summary>
    /// Wires the EGNN sensor. Its fields are private [SerializeField], so this
    /// goes through SerializedObject rather than reaching in with reflection.
    /// </summary>
    static void ConfigureSensor(
        GameObject crawler, GameObject arenaRoot, GameObject body, ParkourObstacleSource source)
    {
        var sensor = crawler.AddComponent<EGNNSensorComponent>();
        var so = new SerializedObject(sensor);
        so.FindProperty("m_SensorName").stringValue = "EGNNSensor";
        so.FindProperty("m_MaxEntities").intValue = SelfEntityCount(body) + MaxObstacles;
        so.FindProperty("m_IncludeRotation").boolValue = true;
        so.FindProperty("m_IncludeLinearVelocity").boolValue = true;
        so.FindProperty("m_IncludeAngularVelocity").boolValue = true;
        // Obstacle nodes sit on the box surface nearest the agent, and this is the
        // displacement back to the centre. Without it a surface node is ambiguous
        // about the box behind it; with it the encoding is lossless. The training
        // config must set has_center_offset to match, or the encoder reads the
        // three offset floats as something else entirely.
        so.FindProperty("m_IncludeCenterOffset").boolValue = true;
        so.FindProperty("m_IncludeExtent").boolValue = true;
        // Arena-relative coordinates, so a replicated arena produces observations
        // identical to arena zero's.
        so.FindProperty("m_VirtualRoot").objectReferenceValue = arenaRoot;

        var groups = so.FindProperty("m_RootGroups");
        groups.arraySize = 1;
        var group = groups.GetArrayElementAtIndex(0);
        group.FindPropertyRelative("Root").objectReferenceValue = body;
        group.FindPropertyRelative("Type").stringValue = "self";
        group.FindPropertyRelative("IncludeChildren").boolValue = true;
        group.FindPropertyRelative("RootSubType").stringValue = "body";

        var children = group.FindPropertyRelative("Children");
        var overrides = ChildOverrides(body);
        children.arraySize = overrides.Count;
        for (var i = 0; i < overrides.Count; i++)
        {
            var element = children.GetArrayElementAtIndex(i);
            element.FindPropertyRelative("Child").objectReferenceValue = overrides[i].Child;
            element.FindPropertyRelative("Include").boolValue = overrides[i].Include;
            element.FindPropertyRelative("SubType").stringValue = overrides[i].SubType;
        }

        var sources = so.FindProperty("m_EntitySources");
        sources.arraySize = 1;
        sources.GetArrayElementAtIndex(0).objectReferenceValue = source;
        so.ApplyModifiedProperties();
    }

    struct ChildSpec
    {
        public Transform Child;
        public bool Include;
        public string SubType;
    }

    /// <summary>
    /// One entry per child the sensor will discover, matching its own rule:
    /// anything under the root carrying a Rigidbody or a Collider.
    /// </summary>
    /// <remarks>
    /// The subtypes are deliberately shared -- four legs are "upper", not
    /// leg0/leg1/leg2/leg3. A blank SubType falls back to the GameObject name,
    /// which would mint eight one-hot slots, widen every row, and let the policy
    /// identify a specific leg by its label instead of by where it is. Nothing
    /// learned about one leg would transfer to the others.
    /// </remarks>
    static List<ChildSpec> ChildOverrides(GameObject body)
    {
        var specs = new List<ChildSpec>();
        foreach (var t in body.GetComponentsInChildren<Transform>(true))
        {
            if (t == body.transform || !IsSensed(t))
            {
                continue;
            }
            string subType;
            var include = true;
            if (t.name.StartsWith("foreleg", StringComparison.Ordinal))
            {
                subType = "lower";
            }
            else if (t.name.StartsWith("leg", StringComparison.Ordinal))
            {
                subType = "upper";
            }
            else
            {
                // Decoration with a collider (the example's sweatband). Excluded
                // rather than typed: an entity that is not a body part would take
                // both a row and a one-hot slot to say nothing.
                subType = t.name;
                include = false;
            }
            specs.Add(new ChildSpec { Child = t, Include = include, SubType = subType });
        }
        return specs;
    }

    static bool IsSensed(Transform t)
        => t.GetComponent<Rigidbody>() != null || t.GetComponent<Collider>() != null;

    static int SelfEntityCount(GameObject body)
        => 1 + ChildOverrides(body).Count(s => s.Include);

    // ----------------------------------------------------------------- scenes

    static void BuildScene(GameObject envPrefab, string path, int arenas)
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        for (var i = 0; i < arenas; i++)
        {
            var arena = (GameObject)PrefabUtility.InstantiatePrefab(envPrefab, scene);
            arena.name = arenas == 1 ? envPrefab.name : $"{envPrefab.name}_{i}";
            arena.transform.position = new Vector3(
                (i % ArenaGridWidth) * ArenaSpacingX,
                0f,
                (i / ArenaGridWidth) * ArenaSpacingZ);
        }
        if (!EditorSceneManager.SaveScene(scene, path))
        {
            throw new Exception($"Failed to save scene {path}");
        }
        Debug.Log($"Scene ({arenas} arena{(arenas == 1 ? "" : "s")}) -> {path}");
    }

    // ------------------------------------------------------------ diagnostics

    /// <summary>
    /// Prints the numbers that have to agree with config/ppo/CrawlerParkour.yaml
    /// and with SETUP.md. A mismatch between the sensor toggles and the encoder
    /// keys does not fail loudly -- it reinterprets geometry columns as one-hots
    /// and trains anyway -- so these are worth having in the build log.
    /// </summary>
    static void Report(GameObject envPrefab)
    {
        var agent = envPrefab.GetComponentInChildren<CrawlerParkourAgent>(true);
        var behavior = agent.GetComponent<BehaviorParameters>();
        var self = SelfEntityCount(agent.body.gameObject);

        // types {self, obstacle}; subtypes {body, upper, lower} + {static, dynamic}
        const int types = 2;
        const int subTypes = 5;
        var row = 3 + 4 + 3 + 3 + 3 + 3 + types + subTypes;

        var jd = agent.GetComponent<Unity.MLAgentsExamples.JointDriveController>();
        var hip = agent.leg0Upper.GetComponent<ConfigurableJoint>();
        var knee = agent.leg0Lower.GetComponent<ConfigurableJoint>();

        Debug.Log(
            "CrawlerParkour build report\n"
            + $"  behavior name        : {behavior.BehaviorName}\n"
            + $"  continuous actions   : {behavior.BrainParameters.ActionSpec.NumContinuousActions}"
            + $" ({CrawlerParkourAgent.NumJointTargetActions} targets,"
            + $" {CrawlerParkourAgent.NumStrengthActions} strengths,"
            + $" {CrawlerParkourAgent.NumFrictionActions} friction)\n"
            + $"  vector observations  : {behavior.BrainParameters.VectorObservationSize}"
            + $" (grid {agent.GridForward}x{agent.GridLateral})\n"
            + $"  joint drive          : spring {jd.maxJointSpring}, dampen {jd.jointDampen},"
            + $" force {jd.maxJointForceLimit}\n"
            + $"  hip range            : X [{hip.lowAngularXLimit.limit},"
            + $" {hip.highAngularXLimit.limit}] Y +/-{hip.angularYLimit.limit}\n"
            + $"  knee range           : X [{knee.lowAngularXLimit.limit},"
            + $" {knee.highAngularXLimit.limit}]\n"
            + $"  friction             : ground {GroundFriction}, foot command"
            + $" [0, {CrawlerParkourAgent.FootFrictionMax:F4}] x ground"
            + $" = contact [0, {GroundFriction * CrawlerParkourAgent.FootFrictionMax:F3}],"
            + $" body {CrawlerParkourAgent.BodyFriction} (run 008 contact was"
            + $" {CrawlerParkourAgent.Run008ContactFriction:F3} everywhere)\n"
            + $"  EGNN entities        : {self} self + {MaxObstacles} obstacles = {self + MaxObstacles}\n"
            + $"  EGNN row width       : {row} (pos 3, quat 4, linvel 3, angvel 3,"
            + $" offset 3, extent 3, type {types}, subtype {subTypes})\n"
            + $"  ground mask          : layer {LayerMask.NameToLayer(TrackLayer)} '{TrackLayer}'"
            + $", agent mask = {agent.GroundMask.value}\n"
            + "  config must set      : has_quaternion, has_linear_velocity,"
            + " has_angular_velocity, has_center_offset = true");

        if (behavior.BrainParameters.VectorObservationSize != VectorObservationSize(agent))
        {
            Debug.LogError("Vector observation size disagrees with CollectObservations");
        }
    }

    /// <summary>
    /// Steps real PhysX and checks that the friction actuator run 009 adds is
    /// connected to anything at all.
    /// </summary>
    /// <remarks>
    /// EVERY OTHER PART OF THIS CHANGE CAN BE READ AND BELIEVED. This one cannot.
    /// It rests on two claims about an engine, not about our code:
    ///
    ///   1. that a contact between a Multiply material and an Average one resolves
    ///      as Multiply, so a foot's command means GroundFriction x itself rather
    ///      than some average of the two;
    ///   2. that writing `dynamicFriction` on a material a collider is ALREADY
    ///      RESTING ON changes the force on the next step, rather than on the next
    ///      time the contact is created.
    ///
    /// If (2) is false the whole feature still compiles, still logs, still shows a
    /// policy learning to move the four new outputs -- and does nothing, because a
    /// foot only ever gets the friction it happened to have when it landed. That is
    /// a 24-hour run and roughly $31 to discover from a reward curve, and it would
    /// most likely be read as "the actuator did not help".
    ///
    /// So it is measured against closed-form predictions instead. A block sliding on
    /// a plane decelerates at mu*g, so distance to rest from v0 is v0^2 / (2*mu*g) --
    /// arithmetic with no free parameters, which makes a wrong result unmistakable
    /// rather than merely surprising. The last case is the one that matters: slide
    /// frictionless for half a second, then write the material mid-slide. If the
    /// write is ignored the block travels the full 5 m and the check fails loudly.
    /// </remarks>
    [MenuItem("Training/CrawlerParkour/Verify Friction Control")]
    public static void VerifyFrictionControl()
    {
        var ok = true;
        var previousMode = Physics.simulationMode;
        var trash = new List<UnityEngine.Object>();
        try
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            Physics.simulationMode = SimulationMode.Script;

            const float dt = 0.02f;
            const float v0 = 5f;
            float g = Mathf.Abs(Physics.gravity.y);

            // The track's own material, exactly as the box prefab carries it.
            var ground = new PhysicsMaterial("ProbeGround")
            {
                dynamicFriction = GroundFriction,
                staticFriction = GroundFriction,
                bounciness = 0f,
                frictionCombine = PhysicsMaterialCombine.Average,
            };
            trash.Add(ground);

            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            trash.Add(floor);
            floor.name = "ProbeFloor";
            floor.transform.localScale = new Vector3(200f, 1f, 200f);
            floor.transform.position = new Vector3(0f, -0.5f, 0f);
            floor.GetComponent<BoxCollider>().sharedMaterial = ground;

            const int maxSteps = 400;

            // Slides a body and returns (metres travelled, mean deceleration while a
            // constant friction was in force). `switchAt` < 0 holds mu0 throughout;
            // otherwise mu1 is written at that time and the deceleration reported is
            // the one AFTER the write, which is what makes the write observable.
            (float travelled, float decel) Slide(
                PrimitiveType shape, bool freezeRotation, float mu0, float mu1, float switchAt,
                bool onSide = false)
            {
                var block = GameObject.CreatePrimitive(shape);
                trash.Add(block);
                block.name = "ProbeBlock";
                // Resting height is per shape, and getting it wrong is not a small
                // error: Unity's capsule primitive is 2 tall, so dropping it in at the
                // box's 0.5 would bury half of it in the floor and measure a
                // depenetration impulse rather than friction.
                var halfHeight = shape == PrimitiveType.Capsule && !onSide ? 1f : 0.5f;
                block.transform.position = new Vector3(0f, halfHeight, 0f);
                if (onSide) block.transform.rotation = Quaternion.Euler(0f, 0f, 90f);
                var rb = block.AddComponent<Rigidbody>();
                if (freezeRotation) rb.constraints = RigidbodyConstraints.FreezeRotation;
                rb.linearVelocity = new Vector3(0f, 0f, v0);

                var mat = new PhysicsMaterial("ProbeFoot")
                {
                    dynamicFriction = mu0,
                    staticFriction = mu0,
                    bounciness = 0f,
                    frictionCombine = PhysicsMaterialCombine.Multiply,
                };
                trash.Add(mat);
                block.GetComponent<Collider>().sharedMaterial = mat;

                float t = 0f, measureFrom = 0f, vAtMeasure = v0, vLast = v0, tLast = 0f;
                var switched = switchAt < 0f;
                for (var step = 0; step < maxSteps; step++)
                {
                    if (!switched && t >= switchAt)
                    {
                        mat.dynamicFriction = mu1;
                        mat.staticFriction = mu1;
                        switched = true;
                        measureFrom = t;
                        vAtMeasure = rb.linearVelocity.z;
                    }
                    Physics.Simulate(dt);
                    t += dt;
                    if (rb.linearVelocity.z > 0.02f) { vLast = rb.linearVelocity.z; tLast = t; }
                    else break;
                }
                var travelled = block.transform.position.z;
                var span = tLast - measureFrom;
                var decel = span > 1e-4f ? (vAtMeasure - vLast) / span : 0f;
                UnityEngine.Object.DestroyImmediate(block);
                return (travelled, decel);
            }

            Debug.Log(
                "Friction probe: a body launched at 5 m/s along a ParkourGround plane.\n"
                + "  Coulomb says it decelerates at mu*g with mu = ground x command under "
                + "Multiply combine,\n  and at (ground + command)/2 * g under Average -- which "
                + "is what this is here to tell apart.");

            var neutral = CrawlerParkourAgent.FootFrictionMax * 0.5f;
            var full = CrawlerParkourAgent.FootFrictionMax;

            // What PhysX actually charges, per unit of Coulomb, measured rather than
            // assumed. See the block comment below the results for why it is not 1.
            float Effective(string label, PrimitiveType shape, bool freeze, float mu,
                            bool onSide = false)
            {
                var (d, a) = Slide(shape, freeze, mu, mu, -1f, onSide);
                var coulomb = GroundFriction * mu * g;
                var ratio = coulomb > 0f ? a / coulomb : 0f;
                Debug.Log($"  {label}: command {mu:F4}, contact mu {GroundFriction * mu:F3}"
                          + $" -> slid {d:F3} m, decelerating at {a:F2} m/s^2"
                          + $" against a Coulomb {coulomb:F2} ({ratio:F2}x)");
                return ratio;
            }

            float EffectiveTipped(string label, float mu)
                => Effective(label, PrimitiveType.Capsule, true, mu, true);

            // --- 1. Is the pair resolved as Multiply, or as Average? ---
            //
            // Asked as a RATIO, which is the form that does not depend on the constant
            // measured below. Doubling the command doubles the contact under Multiply
            // and multiplies it by only 1.30 under Average ((1.6+1.375)/2 over
            // (1.6+0.6875)/2), so the two answers are far apart and nothing in between
            // is a near miss.
            var aNeutral = Slide(PrimitiveType.Cube, true, neutral, neutral, -1f).decel;
            var aFull = Slide(PrimitiveType.Cube, true, full, full, -1f).decel;
            var ratio = aNeutral > 0f ? aFull / aNeutral : 0f;
            var multiplyOk = Mathf.Abs(ratio - 2f) <= 0.1f;
            Debug.Log($"  doubling the command changed the deceleration by {ratio:F3}x"
                      + $" (Multiply predicts 2.00, Average predicts 1.30)"
                      + $" {(multiplyOk ? "ok" : "FAIL")}");
            if (!multiplyOk)
            {
                ok = false;
                Debug.LogError(
                    $"The foot material is not winning the combine: doubling the command "
                    + $"moved the contact by {ratio:F2}x, not 2x. Every friction number in "
                    + "CrawlerParkourAgent is stated as GroundFriction x the command, and "
                    + "under any other combine mode that arithmetic is simply wrong.");
            }

            // --- 2. Does a zero command really mean no grip? ---
            //
            // Under Average a released foot would still carry (1.6 + 0)/2 = 0.8 and
            // stop inside two metres. The whole point of the actuator is that it can
            // let go, so this is the case that has to be unambiguous.
            var free = Slide(PrimitiveType.Cube, true, 0f, 0f, -1f).travelled;
            var freeMax = v0 * maxSteps * dt;
            var freeOk = free > 0.98f * freeMax;
            Debug.Log($"  released (action -1): slid {free:F3} m of a frictionless "
                      + $"{freeMax:F1} m {(freeOk ? "ok" : "FAIL")}");
            if (!freeOk) ok = false;

            // --- 3. THE ONE THAT MATTERS: written onto a material already in contact ---
            //
            // Frictionless for 0.5 s, then full grip written while the body is sliding.
            // If PhysX only picks a material up when a contact is created, this
            // deceleration is zero and the body runs to the end of the plane.
            var (midD, midA) = Slide(PrimitiveType.Cube, true, 0f, full, 0.5f);
            var midOk = aFull > 0f && Mathf.Abs(midA - aFull) / aFull <= 0.1f;
            Debug.Log($"  written mid-slide: slid {midD:F3} m, decelerating at {midA:F2} m/s^2"
                      + $" after the write against {aFull:F2} for a foot that had that grip "
                      + $"all along {(midOk ? "ok" : "FAIL")}");
            if (!midOk)
            {
                ok = false;
                Debug.LogError(
                    $"Writing dynamicFriction on a material ALREADY IN CONTACT did not take "
                    + $"effect: {midA:F2} m/s^2 against {aFull:F2} for the same grip applied "
                    + "before the slide began. The four friction actions would be decorative "
                    + "-- a foot would only ever get the grip it happened to have when it "
                    + "landed, and the run would read as 'the actuator did not help'.");
            }

            // --- 4. How much of Coulomb the solver actually charges, BY SHAPE ---
            //
            // The cases above all use a box, and a box lands at exactly 2.00x the
            // textbook mu*g. That is not the engine being wrong and it is not our
            // numbers being wrong -- it is PxFrictionType patch putting two friction
            // anchors under a face-down contact and allowing each the full mu*N. A
            // sphere, which touches at a point, comes back at exactly 1.00x.
            //
            // WHICH MEANS THE BOX FIGURE DOES NOT APPLY TO THE CRAWLER, and reporting
            // it as if it did would have been the friction version of run 008's
            // "doubling the material doubles the contact". Every collider on this
            // animal is a capsule or a sphere; none of them is a face. So the capsule
            // is measured too, in both orientations it actually meets terrain in --
            // on its rounded end, which is a foot tip, and on its side, which is a leg
            // lying flat -- and that is the number the design arithmetic rests on.
            //
            // It matters beyond bookkeeping: the generator's feasibility guarantee
            // says a ramp is climbable below arctan(mu), and that bound is only true
            // of the effective coefficient.
            Effective("cube,    face down    ", PrimitiveType.Cube, true, neutral);
            Effective("cube,    free to spin ", PrimitiveType.Cube, false, neutral);
            Effective("sphere,  point contact", PrimitiveType.Sphere, true, neutral);
            var footRatio = Effective("capsule, on its end  ", PrimitiveType.Capsule, true, neutral);
            var legRatio = EffectiveTipped("capsule, on its side ", neutral);
            Debug.Log(
                $"  A capsule standing on its rounded end is charged {footRatio:F2}x Coulomb "
                + $"and the same capsule lying on its side {legRatio:F2}x -- one anchor "
                + "against two.\n"
                + "  So contact mu as stated in CrawlerParkourAgent is exactly right for a "
                + "foot placed on its tip, and DOUBLE for any limb lying along the ground.\n"
                + $"  That asymmetry is the leg-snag mechanism measured: under run 008 a "
                + $"foot got {CrawlerParkourAgent.Run008ContactFriction:F2} and a leg fallen "
                + $"flat across an edge got {2f * CrawlerParkourAgent.Run008ContactFriction:F2}"
                + ", the grippiest contact on the animal, on the part with no actuator to "
                + "release it.\n"
                + $"  Run 009 puts that limb at {CrawlerParkourAgent.BodyFriction:F2} x "
                + $"{GroundFriction} x 2 = "
                + $"{2f * CrawlerParkourAgent.BodyFriction * GroundFriction:F2} effective, a "
                + $"{2f * CrawlerParkourAgent.Run008ContactFriction / (2f * CrawlerParkourAgent.BodyFriction * GroundFriction):F1}x "
                + "reduction in what a scraping leg has to drag itself out of.");

            Debug.Log(ok ? "FRICTION OK" : "FRICTION FAILED");
        }
        catch (Exception e)
        {
            Debug.LogError($"Friction probe threw: {e}");
            ok = false;
        }
        finally
        {
            Physics.simulationMode = previousMode;
            foreach (var o in trash)
            {
                if (o != null) UnityEngine.Object.DestroyImmediate(o);
            }
        }
        ExitIfBatch(ok);
    }

    /// <summary>
    /// Drives the real generator inside the built multi-arena scene and checks that
    /// a replicated arena lays its geometry out where its own root is.
    /// </summary>
    /// <remarks>
    /// The generator works in the track's own frame, so every arena computes the
    /// same layout and the transform does the separating. That is invisible to the
    /// offline feasibility harness, which has no transforms at all, and invisible in
    /// the prefab YAML, which contains no obstacles -- they are spawned at runtime.
    /// If it were wrong, all 32 arenas would generate one pile of geometry at the
    /// origin and 31 crawlers would spend the run falling through empty space, while
    /// the feasibility harness kept reporting every layout traversable.
    /// </remarks>
    [MenuItem("Training/CrawlerParkour/Verify Multi-Arena Scene")]
    public static void VerifyMultiArena()
    {
        var ok = true;
        try
        {
            var scene = EditorSceneManager.OpenScene(MultiScenePath, OpenSceneMode.Single);
            var arenas = scene.GetRootGameObjects()
                .Select(g => g.GetComponent<CrawlerParkourEnvController>())
                .Where(c => c != null)
                .ToList();
            Debug.Log($"Verify: {arenas.Count} arenas in {MultiScenePath}");

            // Exactly one Agent per crawler. Two is legal to SERIALIZE but not to
            // run: ML-Agents allocates action buffers per Agent from the single
            // shared BehaviorParameters, and the second one dies in
            // AgentInfo.CopyActions. Run 001 was launched, provisioned and paid
            // for before this surfaced -- geometry checks cannot see it, because
            // nothing here steps the Academy.
            foreach (var controller in arenas)
            {
                foreach (var agent in controller.GetComponentsInChildren<Agent>(true))
                {
                    var all = agent.GetComponents<Agent>();
                    if (all.Length == 1 && all[0] is CrawlerParkourAgent) continue;
                    Debug.LogError(
                        $"{agent.gameObject.name}: expected exactly one CrawlerParkourAgent, found "
                        + $"{all.Length} Agent component(s): {string.Join(", ", all.Select(a => a.GetType().Name))}");
                    ok = false;
                    break;
                }
                if (!ok) break;
            }

            foreach (var index in new[] { 0, arenas.Count - 1 })
            {
                var controller = arenas[index];
                var track = controller.track;
                var origin = track.Origin;
                track.Generate(12345, 1f);

                var boxes = track.ActiveBoxes;
                if (boxes.Count == 0)
                {
                    Debug.LogError($"arena {index}: generated no geometry");
                    ok = false;
                    continue;
                }

                var worstPlacement = 0f;
                foreach (var box in boxes)
                {
                    worstPlacement = Mathf.Max(
                        worstPlacement,
                        Vector3.Distance(box.Tr.position, origin + box.Center));
                }

                // And the sensor has to see those boxes from where the crawler is,
                // in world space -- this is the conversion back out of track space.
                controller.obstacleSource.Reference = controller.agent.body;
                var entities = new List<EGNNEntity>();
                controller.obstacleSource.CollectEntities(entities, MaxObstacles);
                var nearest = entities.Count == 0
                    ? float.NaN
                    : entities.Min(e => Vector3.Distance(e.Position, controller.agent.body.position));

                Debug.Log(
                    $"arena {index}: origin {origin}, {boxes.Count} boxes, "
                    + $"worst placement error {worstPlacement:F4}m, "
                    + $"{entities.Count} sensed, nearest {nearest:F2}m");

                if (worstPlacement > 1e-3f)
                {
                    Debug.LogError($"arena {index}: boxes are not where the maths says");
                    ok = false;
                }
                if (entities.Count == 0 || float.IsNaN(nearest) || nearest > 25f)
                {
                    Debug.LogError(
                        $"arena {index}: obstacle nodes are not near the crawler "
                        + "-- the track-to-world conversion is wrong");
                    ok = false;
                }

                // Every place an episode can START has to be solid ground under the
                // whole animal, not just under its navel.
                //
                // This was one GroundHeightAt at the body centre, and it passed
                // while 25% of the crawler hung over the back edge at every spawn
                // and 38% after every fall -- the crawler is ~3.9m long and its
                // centre was always a metre clear of the edge. Measure what the body
                // occupies, not where it is.
                //
                // Only ground missing in Z is a defect. Lateral overhang at high
                // difficulty is the environment working as designed: the track
                // narrows to 5m and the corridor to 1.7m against a 3.94m splayed
                // rest pose, so the crawler is MEANT to have to tuck. Running out of
                // world in Z is never intended, so the two are counted separately
                // and only the first fails the build.
                // Run 006 samples this distribution rather than three fixed poses.
                // The old version checked spawn (z=1), respawn (z=0.5) and finish
                // (z = TrackLength - 1) on ONE seed. Two of those are now dead --
                // there is no finish line, and the spawn is mid-track -- and one
                // seed was always thin: the finish pose only ever passed because
                // seed 12345's last segment happened not to be a Gap. Lengthening
                // the track to 24 segments moved that segment and the check failed
                // on geometry that was never a defect.
                //
                // So: draw real spawns from TryPickSpawn, and real checkpoints from
                // along the track, over several seeds.
                foreach (var d in new[] { 0f, 0.25f, 1f })
                {
                    float worstSpawnVoid = 0f, worstSpawnSide = 0f;
                    float worstRespawnVoid = 0f;
                    int spawns = 0, noSpawn = 0;

                    for (int seed = 0; seed < 8; seed++)
                    {
                        track.Generate(12345 + seed, d);
                        var rng = new System.Random(999 + seed);

                        for (int k = 0; k < 8; k++)
                        {
                            if (!track.TryPickSpawn(rng, SpawnReserveSegments, 0.5f, out var p))
                            {
                                noSpawn++;
                                continue;
                            }
                            spawns++;
                            var (v, s) = Footprint(track, controller.agent, new Vector2(p.x, p.z));
                            worstSpawnVoid = Mathf.Max(worstSpawnVoid, v);
                            worstSpawnSide = Mathf.Max(worstSpawnSide, s);
                        }

                        // Checkpoint respawns land on a segment boundary + 0.5 on the
                        // lane, anywhere the agent has reached.
                        for (int seg = 0; seg < track.BuiltSegments; seg++)
                        {
                            float z = seg * track.SegmentLength + 0.5f;
                            var (v, _) = Footprint(track, controller.agent,
                                new Vector2(track.LaneCenterAt(z), z));
                            worstRespawnVoid = Mathf.Max(worstRespawnVoid, v);
                        }
                    }

                    Debug.Log(
                        $"arena {index} d={d:F2}: {spawns} spawns drawn ({noSpawn} unfound), "
                        + $"worst spawn {worstSpawnVoid * 100f:F0}% over void / "
                        + $"{worstSpawnSide * 100f:F0}% past the edge; "
                        + $"worst respawn {worstRespawnVoid * 100f:F0}% over void");

                    if (noSpawn > 0)
                    {
                        Debug.LogError($"arena {index} d={d:F2}: {noSpawn} tracks had no usable "
                            + "spawn -- TryPickSpawn is rejecting everything");
                        ok = false;
                    }
                    if (worstSpawnVoid > 1e-3f)
                    {
                        Debug.LogError(
                            $"arena {index} d={d:F2}: the crawler spawns over the end of the "
                            + $"world -- {worstSpawnVoid * 100f:F0}% of its footprint has no "
                            + "ground under it");
                        ok = false;
                    }
                    // A checkpoint respawn is allowed to land somewhere partly
                    // unsupported -- it returns the agent to ground it has already
                    // crossed, which may be a beam. A SPAWN is not: the episode
                    // should not open with a fall the policy had no chance to avoid.
                    if (worstRespawnVoid > 0.5f)
                    {
                        Debug.LogError(
                            $"arena {index} d={d:F2}: a checkpoint respawn puts "
                            + $"{worstRespawnVoid * 100f:F0}% of the crawler over a void");
                        ok = false;
                    }
                }
            }
            Debug.Log(ok ? "VERIFY OK" : "VERIFY FAILED");
        }
        catch (Exception e)
        {
            Debug.LogError($"Verify threw: {e}");
            ok = false;
        }
        ExitIfBatch(ok);
    }

    /// <summary>
    /// Rasterises the crawler's footprint with its body centre at a track-space
    /// (x, z), and splits what is unsupported into two causes: over a void inside
    /// the track's own width, and past the track edge entirely.
    /// </summary>
    /// <remarks>
    /// The split is the whole point. Lateral overhang is a design property at high
    /// difficulty -- the track narrows below the splayed rest pose on purpose --
    /// while a void in Z means the track simply ran out, which is never intended.
    /// A single "unsupported" number conflates the two and either fails on correct
    /// geometry or passes on a crawler spawned off the end.
    ///
    /// Measures the authored rest pose, which is exactly what an episode starts
    /// from: OnEpisodeBegin calls BodyPart.Reset to restore the recorded world
    /// transforms, then TeleportTo translates the whole animal rigidly to the spawn
    /// point. So the footprint at spawn is the prefab's own footprint, offset.
    /// </remarks>
    static (float overVoid, float overSide) Footprint(
        ParkourTrackGenerator track, CrawlerParkourAgent agent, Vector2 centre)
    {
        var cols = agent.GetComponentsInChildren<Collider>();
        if (cols.Length == 0) return (0f, 0f);
        var b = cols[0].bounds;
        foreach (var c in cols) b.Encapsulate(c.bounds);

        // Relative to the body centre, since that is what the spawn point sets.
        // Not assumed symmetric -- the offset is measured, not halved.
        var lo = b.min - agent.body.position;
        var hi = b.max - agent.body.position;
        var halfW = track.TrackWidth * 0.5f;

        const float step = 0.1f;
        var voids = 0;
        var sides = 0;
        var total = 0;
        for (var dx = lo.x; dx <= hi.x + 1e-4f; dx += step)
        {
            for (var dz = lo.z; dz <= hi.z + 1e-4f; dz += step)
            {
                total++;
                var x = centre.x + dx;
                if (Mathf.Abs(x) > halfW) { sides++; }
                else if (!track.GroundHeightAt(x, centre.y + dz, out _)) { voids++; }
            }
        }
        if (total == 0) return (0f, 0f);
        return (voids / (float)total, sides / (float)total);
    }

    static bool BuildPlayer(string scenePath, string outDirName, string exeName, BuildTarget target)
    {
        try
        {
            var outDir = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", "envs", outDirName));
            Directory.CreateDirectory(outDir);

            EditorUserBuildSettings.standaloneBuildSubtarget = StandaloneBuildSubtarget.Player;
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = Path.Combine(outDir, exeName),
                target = target,
                subtarget = (int)StandaloneBuildSubtarget.Player,
                options = BuildOptions.None,
            });

            if (report.summary.result != BuildResult.Succeeded)
            {
                Debug.LogError(
                    $"Build FAILED: {report.summary.result}, errors={report.summary.totalErrors}");
                return false;
            }
            Debug.Log(
                $"Build succeeded: {report.summary.outputPath} "
                + $"({report.summary.totalSize / (1024 * 1024)} MB)");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"Build threw: {e}");
            return false;
        }
    }

    static void ExitIfBatch(bool success)
    {
        if (Application.isBatchMode)
        {
            EditorApplication.Exit(success ? 0 : 1);
        }
    }
}
