using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

/// <summary>
/// Command-line friendly builds of training environments.
/// Invoke from CLI:
///   Unity.exe -projectPath <Project> -batchmode -nographics
///            -executeMethod TrainingBuilds.BuildCrawlerSumoEGNNLinux
///            -logFile <log path>
/// Output goes to <repo>/envs/<name>_linux/ next to the existing Windows training builds.
/// </summary>
public static class TrainingBuilds
{
    [MenuItem("Training/Build CrawlerSumoEGNN (Linux x86_64)")]
    public static void BuildCrawlerSumoEGNNLinux()
    {
        BuildLinux(
            "Assets/CrawlerSumo/Scenes/CrawlerSumoEGNN.unity",
            "CrawlerSumoEGNN_linux",
            "CrawlerSumoEGNN.x86_64");
    }

    [MenuItem("Training/Build CrawlerSumo (Linux x86_64)")]
    public static void BuildCrawlerSumoLinux()
    {
        BuildLinux(
            "Assets/CrawlerSumo/Scenes/CrawlerSumo.unity",
            "CrawlerSumo_linux",
            "CrawlerSumo.x86_64");
    }

    [MenuItem("Training/Build CrawlerSumoEGNN (Linux Dedicated Server)")]
    public static void BuildCrawlerSumoEGNNLinuxServer()
    {
        BuildLinux(
            "Assets/CrawlerSumo/Scenes/CrawlerSumoEGNN.unity",
            "CrawlerSumoEGNN_linux_server",
            "CrawlerSumoEGNN.x86_64",
            StandaloneBuildSubtarget.Server);
    }

    static void BuildLinux(
        string scenePath,
        string outDirName,
        string executableName,
        StandaloneBuildSubtarget subtarget = StandaloneBuildSubtarget.Player)
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
                target = BuildTarget.StandaloneLinux64,
                subtarget = (int)subtarget,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(options);
            var summary = report.summary;
            if (summary.result != BuildResult.Succeeded)
            {
                Debug.LogError(
                    $"Build FAILED: {summary.result}, errors={summary.totalErrors}");
                if (Application.isBatchMode)
                {
                    EditorApplication.Exit(1);
                }
                return;
            }

            Debug.Log(
                $"Build succeeded: {summary.outputPath} " +
                $"({summary.totalSize / (1024 * 1024)} MB, {summary.totalTime.TotalMinutes:F1} min)");
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(0);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Build threw: {e}");
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(1);
            }
        }
    }
}
