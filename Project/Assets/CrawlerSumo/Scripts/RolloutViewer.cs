using System;
using System.Collections;
using System.IO;
using System.Reflection;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

/// <summary>
/// Headless-driven rollout viewer: loads a .sentis model per team from disk,
/// runs the match in inference mode, and captures frames as PNGs for GIF/video
/// encoding by an external script.
///
/// Only activates when launched with model args, so training builds that
/// include this script are unaffected:
///   CrawlerSumoViewer.exe --team0-model a.sentis --team1-model b.sentis
///       --capture-dir C:\frames [--capture-seconds 25] [--capture-fps 10]
///
/// .sentis files are produced from checkpoint .onnx by
/// TrainingBuilds.ConvertOnnxToSentis (the ONNX importer is editor-only;
/// runtime can only load serialized sentis models).
/// </summary>
public class RolloutViewer : MonoBehaviour
{
    string m_Team0Path;
    string m_Team1Path;
    string m_CaptureDir;
    float m_CaptureSeconds = 25f;
    float m_CaptureFps = 10f;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (GetArg("--team0-model") == null)
        {
            return;
        }
        var go = new GameObject("RolloutViewer");
        DontDestroyOnLoad(go);
        go.AddComponent<RolloutViewer>();
    }

    static string GetArg(string name)
    {
        var args = Environment.GetCommandLineArgs();
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                return args[i + 1];
            }
        }
        return null;
    }

    void Start()
    {
        m_Team0Path = GetArg("--team0-model");
        m_Team1Path = GetArg("--team1-model");
        m_CaptureDir = GetArg("--capture-dir");
        var secs = GetArg("--capture-seconds");
        if (secs != null) { m_CaptureSeconds = float.Parse(secs); }
        var fps = GetArg("--capture-fps");
        if (fps != null) { m_CaptureFps = float.Parse(fps); }

        try
        {
            var team0Asset = LoadModelAssetFromSentis(m_Team0Path);
            var team1Asset = LoadModelAssetFromSentis(m_Team1Path);
            int applied = 0;
            foreach (var agent in FindObjectsByType<Agent>(FindObjectsSortMode.None))
            {
                var bp = agent.GetComponent<BehaviorParameters>();
                if (bp == null) { continue; }
                var asset = bp.TeamId == 0 ? team0Asset : team1Asset;
                agent.SetModel(bp.BehaviorName, asset);
                bp.BehaviorType = BehaviorType.InferenceOnly;
                applied++;
            }
            Debug.Log($"RolloutViewer: applied models to {applied} agents " +
                $"(team0={Path.GetFileName(m_Team0Path)}, team1={Path.GetFileName(m_Team1Path)})");
        }
        catch (Exception e)
        {
            Debug.LogError($"RolloutViewer: model load failed: {e}");
            Application.Quit(2);
            return;
        }

        FrameCamera();

        if (!string.IsNullOrEmpty(m_CaptureDir))
        {
            Directory.CreateDirectory(m_CaptureDir);
            StartCoroutine(CaptureLoop());
        }
    }

    /// <summary>
    /// Frame the arena tightly: disable any fly-cam controller and place the
    /// main camera on a fixed shot of the platform (radius 15 at the origin).
    /// </summary>
    static void FrameCamera()
    {
        foreach (var mb in FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
        {
            if (mb.GetType().Name == "FlyCamera")
            {
                mb.enabled = false;
            }
        }
        var cam = Camera.main;
        if (cam == null)
        {
            return;
        }
        var center = Vector3.zero;
        var ctrl = FindObjectsByType<CrawlerSumoEnvController>(FindObjectsSortMode.None);
        if (ctrl.Length > 0)
        {
            center = ctrl[0].platformCenter;
        }
        cam.fieldOfView = 55f;
        cam.transform.position = center + new Vector3(0f, 9f, -17f);
        cam.transform.LookAt(center + Vector3.up * 0.5f);
    }

    /// <summary>
    /// Builds a ModelAsset at runtime from a serialized .sentis file.
    /// ModelAsset's data fields are internal, and the ONNX/asset importers are
    /// editor-only, so this mirrors InferenceEngineModelImporter via reflection.
    /// </summary>
    static ModelAsset LoadModelAssetFromSentis(string path)
    {
        var model = ModelLoader.Load(path);
        var asm = typeof(ModelAsset).Assembly;

        // internal static ModelWriter.SaveModel(Model, out byte[], out byte[][])
        var save = typeof(ModelWriter).GetMethod(
            "SaveModel",
            BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public,
            null,
            new[] { typeof(Model), typeof(byte[]).MakeByRefType(), typeof(byte[][]).MakeByRefType() },
            null);
        var args = new object[] { model, null, null };
        save.Invoke(null, args);
        var desc = (byte[])args[1];
        var weights = (byte[][])args[2];

        const BindingFlags kField = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        var dataType = asm.GetType("Unity.InferenceEngine.ModelAssetData");
        var weightsType = asm.GetType("Unity.InferenceEngine.ModelAssetWeightsData");

        var asset = ScriptableObject.CreateInstance<ModelAsset>();
        var data = ScriptableObject.CreateInstance(dataType);
        dataType.GetField("value", kField).SetValue(data, desc);
        typeof(ModelAsset).GetField("modelAssetData", kField).SetValue(asset, data);

        var chunks = Array.CreateInstance(weightsType, weights.Length);
        for (var i = 0; i < weights.Length; i++)
        {
            var chunk = ScriptableObject.CreateInstance(weightsType);
            weightsType.GetField("value", kField).SetValue(chunk, weights[i]);
            chunks.SetValue(chunk, i);
        }
        typeof(ModelAsset).GetField("modelWeightsChunks", kField).SetValue(asset, chunks);
        asset.name = Path.GetFileNameWithoutExtension(path);
        return asset;
    }

    IEnumerator CaptureLoop()
    {
        // Let models apply and the first episode begin before recording
        yield return new WaitForSeconds(1.0f);
        var interval = 1f / m_CaptureFps;
        var endTime = Time.time + m_CaptureSeconds;
        var frame = 0;
        while (Time.time < endTime)
        {
            yield return new WaitForEndOfFrame();
            var tex = ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(
                Path.Combine(m_CaptureDir, $"f{frame:D5}.png"), tex.EncodeToPNG());
            Destroy(tex);
            frame++;
            yield return new WaitForSeconds(interval);
        }
        Debug.Log($"RolloutViewer: captured {frame} frames -> {m_CaptureDir}");
        Application.Quit(0);
    }
}
