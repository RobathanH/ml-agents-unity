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
    /// Friction 0.8. The generator's steepest ramp is 40 degrees and the ceiling
    /// for a ramp that can be walked up is arctan(mu) = 38.7 degrees at mu=0.8,
    /// so this material is part of the feasibility guarantee rather than a
    /// cosmetic choice -- a slippier one makes the hardest ramps unclimbable and
    /// the generator has no way to detect it.
    /// </summary>
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
                existing.dynamicFriction = 0.8f;
                existing.staticFriction = 0.8f;
                existing.bounciness = 0f;
                EditorUtility.SetDirty(existing);
                return existing;
            }

            var mat = new PhysicsMaterial("ParkourGround")
            {
                dynamicFriction = 0.8f,
                staticFriction = 0.8f,
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
            controller.maxEpisodeSteps = 3000;
            controller.fallDepth = 3f;
            controller.difficulty = 0f;
            controller.fixedSeed = -1;
            controller.progressWeight = 10f;
            controller.respawnPenalty = 0.5f;
            controller.finishBonus = 5f;
            controller.energyCostWeight = 0.002f;
            controller.actionRateCostWeight = 0.002f;

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

        var behavior = crawler.GetComponent<BehaviorParameters>();
        behavior.BehaviorName = BehaviorName;
        behavior.BrainParameters.VectorObservationSize = VectorObservationSize(agent);
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(20);
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

        Debug.Log(
            "CrawlerParkour build report\n"
            + $"  behavior name        : {behavior.BehaviorName}\n"
            + $"  continuous actions   : {behavior.BrainParameters.ActionSpec.NumContinuousActions}\n"
            + $"  vector observations  : {behavior.BrainParameters.VectorObservationSize}"
            + $" (grid {agent.GridForward}x{agent.GridLateral})\n"
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
                foreach (var d in new[] { 0f, 0.25f, 1f })
                {
                    track.Generate(12345, d);
                    var finishZ = track.TrackLength - 1f;
                    var cases = new[]
                    {
                        ("spawn", new Vector2(track.LaneCenterAt(1f), 1f)),
                        ("respawn", new Vector2(track.LaneCenterAt(0f), 0.5f)),
                        ("finish", new Vector2(track.LaneCenterAt(finishZ), finishZ)),
                    };
                    foreach (var (name, centre) in cases)
                    {
                        var (overVoid, overSide) = Footprint(track, controller.agent, centre);
                        Debug.Log(
                            $"arena {index} d={d:F2} {name}: {overVoid * 100f:F0}% of the "
                            + $"footprint over a void, {overSide * 100f:F0}% past the track edge");
                        if (overVoid > 1e-3f)
                        {
                            Debug.LogError(
                                $"arena {index} d={d:F2}: the crawler {name}s over the end of "
                                + $"the world -- {overVoid * 100f:F0}% of its footprint has no "
                                + "ground under it. Extend StartApron/FinishApron.");
                            ok = false;
                        }
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
