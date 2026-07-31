using System;
using System.Collections;
using System.IO;
using System.Reflection;
using CrawlerParkour;
using Unity.InferenceEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

/// <summary>
/// Headless-driven rollout viewer: loads one .sentis policy from disk, runs the
/// single-arena parkour scene in inference mode, and captures frames as PNGs for
/// GIF encoding by an external script.
///
/// Only activates when launched with --model, so training builds that include
/// this script are unaffected:
///   CrawlerParkour.exe --model ckpt.sentis --capture-dir C:\frames
///       [--capture-seconds 20] [--capture-fps 15] [--env-param difficulty=0.25]
///
/// .sentis files are produced from checkpoint .onnx by
/// TrainingBuilds.ConvertOnnxToSentis (the ONNX importer is editor-only;
/// runtime can only load serialized sentis models).
///
/// Unlike the sumo viewer this cannot use a fixed camera: the track is 128 m
/// long, so a static shot would show the crawler leaving frame in the first few
/// seconds. The camera trails the body instead, which is also the only framing
/// in which "is it walking or twitching in place" is legible.
/// </summary>
public class ParkourRolloutViewer : MonoBehaviour
{
    string m_ModelPath;
    string m_CaptureDir;
    float m_CaptureSeconds = 20f;
    float m_CaptureFps = 15f;

    Transform m_Follow;
    Camera m_Cam;
    Vector3 m_CamVel;
    // Trailing shot: back along the track, up, and off to one side so the gait
    // reads in silhouette rather than being hidden behind the crawler's own body.
    static readonly Vector3 kOffset = new Vector3(3.5f, 2.6f, -5.5f);

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
    {
        if (GetArg("--model") == null)
        {
            return;
        }
        var go = new GameObject("ParkourRolloutViewer");
        DontDestroyOnLoad(go);
        go.AddComponent<ParkourRolloutViewer>();
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
        m_ModelPath = GetArg("--model");
        m_CaptureDir = GetArg("--capture-dir");
        var secs = GetArg("--capture-seconds");
        if (secs != null) { m_CaptureSeconds = float.Parse(secs); }
        var fps = GetArg("--capture-fps");
        if (fps != null) { m_CaptureFps = float.Parse(fps); }

        try
        {
            var asset = LoadModelAssetFromSentis(m_ModelPath);
            var applied = 0;
            foreach (var agent in FindObjectsByType<Agent>(FindObjectsSortMode.None))
            {
                var bp = agent.GetComponent<BehaviorParameters>();
                if (bp == null) { continue; }
                agent.SetModel(bp.BehaviorName, asset);
                bp.BehaviorType = BehaviorType.InferenceOnly;
                applied++;
            }
            Debug.Log($"ParkourRolloutViewer: applied {Path.GetFileName(m_ModelPath)} " +
                $"to {applied} agent(s)");
        }
        catch (Exception e)
        {
            Debug.LogError($"ParkourRolloutViewer: model load failed: {e}");
            Application.Quit(2);
            return;
        }

        SetUpCamera();

        var measure = GetArg("--measure-seconds");
        if (measure != null)
        {
            // Calibration is CPU-bound, not wall-clock-bound: fixedDeltaTime is
            // untouched, so raising the time scale runs the same physics steps sooner
            // and changes nothing about the rate being measured. If the CPU cannot
            // keep up Unity simply falls behind, which costs time and not accuracy.
            var ts = GetArg("--time-scale");
            if (ts != null) { Time.timeScale = float.Parse(ts); }
            StartCoroutine(MeasureLoop(float.Parse(measure)));
            return;
        }

        if (!string.IsNullOrEmpty(m_CaptureDir))
        {
            Directory.CreateDirectory(m_CaptureDir);
            StartCoroutine(CaptureLoop());
        }
    }

    /// <summary>
    /// Run without capturing and report the distance rate this policy achieves on the
    /// pinned terrain. Exists to calibrate CrawlerParkourEnvController.ReferenceRate.
    /// </summary>
    /// <remarks>
    /// The promote/demote gates are fractions of a reference rate-vs-difficulty curve,
    /// and that curve is a property of the PHYSICS as much as of the terrain. Run 008
    /// doubled joint strength and ground friction, which invalidates a curve measured
    /// under run 007's physics: a policy that now covers ground faster clears a stale
    /// gate every episode, and the ladder ratchets to the ceiling exactly as it did in
    /// runs 006 and 007.
    ///
    /// Capturing frames is not needed to measure a rate and costs ~50x the wall clock,
    /// so this path skips the camera work entirely. Timescale is left alone: the
    /// physics step is what the rate is defined against.
    ///
    /// Reports the mean over whole episodes only. A partial episode at the cutoff has a
    /// distance but not its full clock, and averaging it in biases the rate upward.
    /// </remarks>
    System.Collections.IEnumerator MeasureLoop(float seconds)
    {
        yield return new WaitForSeconds(1.0f);
        var agents = FindObjectsByType<CrawlerParkourAgent>(FindObjectsSortMode.None);
        if (agents.Length == 0)
        {
            Debug.LogError("ParkourRolloutViewer: no CrawlerParkourAgent to measure");
            Application.Quit(2);
            yield break;
        }

        var lastEpisode = new int[agents.Length];
        var lastDistance = new float[agents.Length];
        double sum = 0;
        int episodes = 0;
        float t0 = Time.time;

        while (Time.time - t0 < seconds)
        {
            for (int i = 0; i < agents.Length; i++)
            {
                int ep = agents[i].CompletedEpisodes;
                if (ep != lastEpisode[i])
                {
                    // The episode that just ended covered whatever distance was
                    // standing before the reset zeroed it.
                    if (lastEpisode[i] > 0)
                    {
                        sum += lastDistance[i];
                        episodes++;
                    }
                    lastEpisode[i] = ep;
                }
                lastDistance[i] = agents[i].DistanceCovered;
            }
            yield return new WaitForSeconds(0.1f);
        }

        float mean = episodes > 0 ? (float)(sum / episodes) : 0f;
        Debug.Log($"ParkourRolloutViewer: MEASURE episodes={episodes} " +
                  $"meanDistance={mean:F4} agents={agents.Length}");
        Application.Quit(0);
    }

    /// <summary>
    /// Disables any fly-cam controller and latches the main camera onto the
    /// crawler's body for the trailing shot.
    /// </summary>
    void SetUpCamera()
    {
        foreach (var mb in FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
        {
            if (mb.GetType().Name == "FlyCamera")
            {
                mb.enabled = false;
            }
        }
        m_Cam = Camera.main;
        if (m_Cam == null)
        {
            return;
        }
        var ctrl = FindObjectsByType<CrawlerParkourEnvController>(FindObjectsSortMode.None);
        if (ctrl.Length > 0 && ctrl[0].agent != null)
        {
            m_Follow = ctrl[0].agent.body;
        }
        m_Cam.fieldOfView = 60f;
        if (m_Follow != null)
        {
            m_Cam.transform.position = m_Follow.position + kOffset;
            m_Cam.transform.LookAt(m_Follow.position);
        }
    }

    void LateUpdate()
    {
        if (m_Cam == null || m_Follow == null)
        {
            return;
        }
        // Track-space offset, not body-space: the body yaws and rolls constantly
        // while walking, and a camera parented to that rotation makes a steady
        // gait look like the world is lurching.
        var want = m_Follow.position + kOffset;
        m_Cam.transform.position = Vector3.SmoothDamp(
            m_Cam.transform.position, want, ref m_CamVel, 0.25f);
        m_Cam.transform.LookAt(m_Follow.position + Vector3.up * 0.3f);
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
        // Let the model apply and the first episode begin before recording
        yield return new WaitForSeconds(1.0f);
        // --capture-fps is the wait BETWEEN frames, not the achieved rate: the
        // screenshot and PNG encode below cost real time on top of it, so frames
        // land ~10% further apart than nominal. Encode playback at the measured
        // rate (frame count / --capture-seconds, logged on exit), never at this
        // value, or every gait in the clip is shown faster than it ran.
        var interval = 1f / m_CaptureFps;
        var startTime = Time.time;
        var endTime = startTime + m_CaptureSeconds;
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
        var elapsed = Time.time - startTime;
        Debug.Log($"ParkourRolloutViewer: captured {frame} frames in {elapsed:F2}s " +
            $"({frame / elapsed:F2} fps measured) -> {m_CaptureDir}");
        Application.Quit(0);
    }
}
