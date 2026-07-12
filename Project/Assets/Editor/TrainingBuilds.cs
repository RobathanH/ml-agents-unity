using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Command-line friendly builds of training environments.
/// Invoke from CLI:
///   Unity.exe -projectPath <Project> -batchmode -nographics
///            -executeMethod TrainingBuilds.<Method> -logFile <log path>
/// Output goes to <repo>/envs/<name>/ next to the existing training builds.
/// </summary>
public static class TrainingBuilds
{
    const string EgnnScenePath = "Assets/CrawlerSumo/Scenes/CrawlerSumoEGNN.unity";
    const string EgnnMultiScenePath = "Assets/CrawlerSumo/Scenes/CrawlerSumoEGNN_Multi.unity";
    const string EgnnPrefabPath = "Assets/CrawlerSumo/Prefabs/CrawlerSumoEGNNEnv.prefab";
    const int MultiArenaCount = 32;
    const float ArenaSpacing = 250f;

    [MenuItem("Training/Build CrawlerSumoEGNN (Linux x86_64)")]
    public static void BuildCrawlerSumoEGNNLinux()
    {
        ExitIfBatch(BuildTo(
            EgnnScenePath, "CrawlerSumoEGNN_linux", "CrawlerSumoEGNN.x86_64",
            BuildTarget.StandaloneLinux64, StandaloneBuildSubtarget.Player));
    }

    [MenuItem("Training/Build CrawlerSumoEGNN (Linux Dedicated Server)")]
    public static void BuildCrawlerSumoEGNNLinuxServer()
    {
        ExitIfBatch(BuildTo(
            EgnnScenePath, "CrawlerSumoEGNN_linux_server", "CrawlerSumoEGNN.x86_64",
            BuildTarget.StandaloneLinux64, StandaloneBuildSubtarget.Server));
    }

    /// <summary>
    /// Generates the multi-arena variant of the EGNN scene (saved as a new scene
    /// asset, original untouched) and builds it for Windows for local validation.
    /// </summary>
    [MenuItem("Training/Create Multi-Arena Scene + Build Windows")]
    public static void CreateMultiArenaAndBuildWindows()
    {
        bool ok = CreateMultiArenaScene()
            && BuildTo(
                EgnnMultiScenePath, "CrawlerSumoEGNN_Multi_win", "UnityEnvironment.exe",
                BuildTarget.StandaloneWindows64, StandaloneBuildSubtarget.Player);
        ExitIfBatch(ok);
    }

    /// <summary>Regenerates the multi-arena scene and builds it for Linux in one pass.</summary>
    public static void CreateMultiArenaAndBuildLinux()
    {
        bool ok = CreateMultiArenaScene()
            && BuildTo(
                EgnnMultiScenePath, "CrawlerSumoEGNN_Multi_linux", "CrawlerSumoEGNN.x86_64",
                BuildTarget.StandaloneLinux64, StandaloneBuildSubtarget.Player);
        ExitIfBatch(ok);
    }

    [MenuItem("Training/Build Multi-Arena EGNN (Linux x86_64)")]
    public static void BuildCrawlerSumoEGNNMultiLinux()
    {
        ExitIfBatch(BuildTo(
            EgnnMultiScenePath, "CrawlerSumoEGNN_Multi_linux", "CrawlerSumoEGNN.x86_64",
            BuildTarget.StandaloneLinux64, StandaloneBuildSubtarget.Player));
    }

    static bool CreateMultiArenaScene()
    {
        try
        {
            var scene = EditorSceneManager.OpenScene(EgnnScenePath, OpenSceneMode.Single);
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(EgnnPrefabPath);
            if (prefab == null)
            {
                Debug.LogError($"Prefab not found at {EgnnPrefabPath}");
                return false;
            }

            GameObject original = null;
            foreach (var root in scene.GetRootGameObjects())
            {
                if (PrefabUtility.GetCorrespondingObjectFromSource(root) == prefab)
                {
                    original = root;
                    break;
                }
            }
            if (original == null)
            {
                Debug.LogError("No CrawlerSumoEGNNEnv prefab instance found in scene");
                return false;
            }

            var baseCtrl = original.GetComponentInChildren<CrawlerSumoEnvController>(true);
            if (baseCtrl == null)
            {
                Debug.LogError("No CrawlerSumoEnvController found on arena instance");
                return false;
            }

            for (int i = 1; i < MultiArenaCount; i++)
            {
                // 4-wide grid, 250 units apart -- platforms (r=15) can never interact
                var offset = new Vector3((i % 4) * ArenaSpacing, 0f, (i / 4) * ArenaSpacing);
                var clone = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
                clone.name = $"{original.name}_{i}";
                clone.transform.position = original.transform.position + offset;

                var ctrl = clone.GetComponentInChildren<CrawlerSumoEnvController>(true);
                // platformCenter is an absolute world position -- shift per arena
                ctrl.platformCenter = baseCtrl.platformCenter + offset;
                ctrl.platformRadius = baseCtrl.platformRadius;
                ctrl.fallY = baseCtrl.fallY;
            }

            bool saved = EditorSceneManager.SaveScene(scene, EgnnMultiScenePath);
            Debug.Log($"Multi-arena scene ({MultiArenaCount} arenas) saved: {saved} -> {EgnnMultiScenePath}");
            return saved;
        }
        catch (Exception e)
        {
            Debug.LogError($"CreateMultiArenaScene threw: {e}");
            return false;
        }
    }

    static bool BuildTo(
        string scenePath,
        string outDirName,
        string executableName,
        BuildTarget target,
        StandaloneBuildSubtarget subtarget)
    {
        try
        {
            var outDir = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", "envs", outDirName));
            Directory.CreateDirectory(outDir);

            EditorUserBuildSettings.standaloneBuildSubtarget = subtarget;
            var options = new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = Path.Combine(outDir, executableName),
                target = target,
                subtarget = (int)subtarget,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(options);
            var summary = report.summary;
            if (summary.result != BuildResult.Succeeded)
            {
                Debug.LogError($"Build FAILED: {summary.result}, errors={summary.totalErrors}");
                return false;
            }

            Debug.Log(
                $"Build succeeded: {summary.outputPath} " +
                $"({summary.totalSize / (1024 * 1024)} MB, {summary.totalTime.TotalMinutes:F1} min)");
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
