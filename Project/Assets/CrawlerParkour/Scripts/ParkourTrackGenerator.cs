using System.Collections.Generic;
using UnityEngine;

namespace CrawlerParkour
{
    /// <summary>
    /// Every obstacle bound, interpolated from a single difficulty scalar.
    /// </summary>
    /// <remarks>
    /// Values are calibrated against the measured crawler prefab (see DESIGN.md),
    /// not guessed. The crawler is roughly 64:1 force-to-weight, so nothing here is
    /// torque-limited -- every ceiling is geometric, which is what keeps "harder"
    /// meaning "needs a smarter movement" rather than "needs more power".
    ///
    /// Reference envelope: body 1.0 across, nominal stance height 1.0, flattens to
    /// ~0.55, rears to ~1.8, leg reach 2.45, splayed span ~3.5, tucked span ~1.4.
    /// </remarks>
    public struct ParkourDifficulty
    {
        public float TrackWidth;
        public float MaxStep;
        public float WallHeight;
        public float MaxGap;
        public float MinClearance;
        public float MinCorridor;
        public float RampPitch;
        public float Camber;
        public float PebbleDensity;
        public int ObstaclesPerSegment;

        public static ParkourDifficulty From(float d)
        {
            d = Mathf.Clamp01(d);
            return new ParkourDifficulty
            {
                // 5m is ~1.4x the splayed stance: lateral error is punished, but a
                // normal gait still fits.
                TrackWidth = Mathf.Lerp(12f, 5f, d),
                // 1.4 < 2.45 leg reach, so reachable without a jump -- but it needs
                // a real weight shift rather than a swing-through.
                MaxStep = Mathf.Lerp(0.2f, 1.4f, d),
                // Rearing reaches ~1.8, so 2.2 needs rear-up plus a hop. Above this
                // it stops being reliably solvable.
                WallHeight = Mathf.Lerp(0.6f, 2.2f, d),
                // 2.5 is ~0.7x the splayed span: a static straddle fails, so it
                // forces an actual run-up and leap.
                MaxGap = Mathf.Lerp(0.4f, 2.5f, d),
                // Body is 1.0 across. 1.15 leaves 0.15 of margin, which cannot be
                // cleared by the 1.0 nominal stance -- it forces a belly-crawl that
                // still has to generate thrust. This is the bound that produces a
                // qualitatively different gait rather than a scaled-down walk.
                MinClearance = Mathf.Lerp(2.4f, 1.15f, d),
                // Legs splay to ~3.5 and tuck to ~1.4, so this cannot be entered
                // without tucking and yaw-aligning first. The other gait-changing
                // bound. Floored at 1.7 rather than the 1.35 first tried: 1.35 is
                // narrower than the tucked stance itself, i.e. flatly impossible
                // rather than hard. 1.7 leaves 0.7 over the body and 0.3 over the
                // tuck -- enough to fit, tight enough that the legs cannot splay.
                MinCorridor = Mathf.Lerp(6f, 1.7f, d),
                // Near arctan(mu) for the surface material; steeper just slides.
                RampPitch = Mathf.Lerp(5f, 40f, d),
                Camber = Mathf.Lerp(0f, 25f, d),
                PebbleDensity = Mathf.Lerp(0f, 2.5f, d),
                ObstaclesPerSegment = Mathf.RoundToInt(Mathf.Lerp(1f, 4f, d)),
            };
        }
    }

    public enum SegmentPattern
    {
        Flat,
        StepField,
        Hurdle,
        Slalom,
        Gap,
        Ramp,
        Pebbles,
        Overhang,
        Squeeze,
        Beam,
    }

    /// <summary>
    /// Procedurally builds a parkour track out of oriented boxes.
    /// </summary>
    /// <remarks>
    /// Feasibility is constructive, not rejection-sampled: each segment first
    /// picks a lane (a lateral band the agent is guaranteed to be able to occupy),
    /// then places obstacles relative to it. Patterns that are meant to be
    /// traversed *around* leave the lane clear; patterns meant to be traversed
    /// *through* span the lane but are bounded to stay within the agent's
    /// envelope. <see cref="VerifyCorridor"/> then re-derives passability from the
    /// placed geometry as an independent check, so a bug in a pattern surfaces as
    /// a regenerate rather than as an unsolvable track.
    ///
    /// Lane centres drift by at most MaxLaneDrift between segments and obstacles
    /// are inset from segment boundaries, so there is always clear ground in which
    /// to make the lateral move.
    /// </remarks>
    public class ParkourTrackGenerator : MonoBehaviour
    {
        [Header("Track")]
        public int SegmentCount = 16;
        public float SegmentLength = 8f;
        public float FloorThickness = 2f;
        [Tooltip("Flat run-in before the first patterned segment.")]
        public int LeadInSegments = 2;

        [Header("Prefabs")]
        public GameObject BoxPrefab;
        public Transform BoxParent;
        [Tooltip("Hard ceiling on pooled boxes. Generation stops early if hit.")]
        public int MaxBoxes = 220;

        [Header("Materials (visual only)")]
        public Material FloorMaterial;
        public Material ObstacleMaterial;

        // Lateral resolution of the feasibility rasteriser. Counting whole bins
        // under-measures a free run by up to one bin, so the corridor test carries
        // a one-bin tolerance and the effective guarantee is
        // "corridor >= MinCorridor - k_BinWidth". At 0.1 that slack is negligible
        // against a 1.7-6.0m corridor. It is not cosmetic: at 0.25 the bias
        // rejected essentially every Slalom segment, whose corridor sits exactly
        // on MinCorridor by construction, and silently rebuilt ~12% of the track
        // as flat ground.
        private const float k_BinWidth = 0.1f;
        // How far the guaranteed lane may move sideways between segments. The
        // agent covers a segment in ~2s and strafes at ~1-2 m/s, so ~0.6x the
        // segment length is a move it can actually make.
        private const float k_MaxLaneDriftFrac = 0.6f;
        // Obstacles stay inside the middle of a segment, leaving clear ground at
        // each end for the lateral reposition the lane drift assumes.
        private const float k_SegmentInset = 0.15f;
        // Walls that must be routed around rather than climbed. Kept above the
        // WallHeight ceiling (2.2) at every difficulty, so a Slalom or Squeeze
        // never degenerates into "just hop over it" at low d -- those patterns
        // exist to force lateral routing and a tucked gait.
        private const float k_ImpassableWallHeight = 2.6f;
        // See the Overhang case: keeps the slab top above floor + WallHeight at
        // every difficulty so the only route is underneath it.
        private const float k_OverhangSlabHalfThickness = 1.2f;

        private readonly List<ParkourBox> m_Pool = new List<ParkourBox>();
        private readonly List<ParkourBox> m_Active = new List<ParkourBox>();
        private readonly List<float> m_SegmentFloorY = new List<float>();
        private readonly List<float> m_LaneCenter = new List<float>();
        private readonly List<ParkourBox> m_SliceCandidates = new List<ParkourBox>();
        private System.Random m_Rng;
        private ParkourDifficulty m_Diff;

        public IReadOnlyList<ParkourBox> ActiveBoxes => m_Active;
        public float TrackLength => SegmentCount * SegmentLength;
        public float TrackWidth => m_Diff.TrackWidth;
        public ParkourDifficulty Difficulty => m_Diff;

        /// <summary>Number of layouts that failed the corridor check and were
        /// rebuilt. Exposed so training can alarm if it is ever non-trivial --
        /// steady non-zero values mean a pattern is generating impossible geometry.
        /// </summary>
        public int RepairCount { get; private set; }

        public void Generate(int seed, float difficulty)
        {
            m_Rng = new System.Random(seed);
            m_Diff = ParkourDifficulty.From(difficulty);
            RepairCount = 0;

            foreach (var b in m_Active)
            {
                if (b.Tr != null) b.Tr.gameObject.SetActive(false);
            }
            m_Active.Clear();
            m_SegmentFloorY.Clear();
            m_LaneCenter.Clear();

            float halfW = m_Diff.TrackWidth * 0.5f;
            float y = 0f;
            float lane = 0f;
            var prevPattern = SegmentPattern.Flat;

            for (int i = 0; i < SegmentCount; i++)
            {
                var pattern = i < LeadInSegments ? SegmentPattern.Flat : PickPattern();

                // Lane first: everything else is placed relative to it.
                float maxDrift = SegmentLength * k_MaxLaneDriftFrac;
                // The drift budget assumes there is clear ground to drift ON. A
                // beam segment has floor only under the lane, so two consecutive
                // beams whose lanes differ by more than the beam width are
                // separated by pure void with no way across -- the one layout in
                // 2000 that the traversal check caught. Consecutive beams must
                // therefore overlap.
                if (pattern == SegmentPattern.Beam && prevPattern == SegmentPattern.Beam)
                {
                    maxDrift = Mathf.Min(maxDrift, BeamWidth() * 0.5f);
                }
                float laneHalf = m_Diff.MinCorridor * 0.5f;
                float laneMin = -halfW + laneHalf;
                float laneMax = halfW - laneHalf;
                if (laneMin > laneMax)
                {
                    // Track narrower than the corridor requirement: the whole width
                    // is the lane.
                    lane = 0f;
                }
                else
                {
                    lane = Mathf.Clamp(
                        lane + (float)(m_Rng.NextDouble() * 2.0 - 1.0) * maxDrift,
                        laneMin, laneMax);
                }

                float nextY = y;
                if (pattern == SegmentPattern.StepField || pattern == SegmentPattern.Ramp)
                {
                    nextY = y + (float)(m_Rng.NextDouble() * 2.0 - 1.0) * m_Diff.MaxStep;
                    nextY = Mathf.Max(nextY, -3f);
                }

                int budget = MaxBoxes - m_Active.Count;
                if (budget < 8) break;

                int firstBox = m_Active.Count;
                BuildSegment(i, pattern, y, nextY, lane, halfW);

                // Independent re-derivation from the geometry that actually got
                // placed. A pattern bug shows up here as a rebuild, not as a track
                // the agent cannot finish.
                if (!VerifyCorridor(i, y, nextY))
                {
                    RepairCount++;
                    for (int k = m_Active.Count - 1; k >= firstBox; k--)
                    {
                        if (m_Active[k].Tr != null) m_Active[k].Tr.gameObject.SetActive(false);
                        m_Active.RemoveAt(k);
                    }
                    nextY = y;
                    BuildSegment(i, SegmentPattern.Flat, y, nextY, lane, halfW);
                }

                m_SegmentFloorY.Add(y);
                m_LaneCenter.Add(lane);
                prevPattern = pattern;
                y = nextY;
            }
        }

        private SegmentPattern PickPattern()
        {
            // Weighted so that the gait-changing patterns (Overhang, Squeeze) stay
            // common enough to be learned rather than treated as noise, and Flat
            // stays present so there is somewhere to build up speed.
            float r = (float)m_Rng.NextDouble();
            if (r < 0.12f) return SegmentPattern.Flat;
            if (r < 0.24f) return SegmentPattern.StepField;
            if (r < 0.36f) return SegmentPattern.Hurdle;
            if (r < 0.48f) return SegmentPattern.Slalom;
            if (r < 0.58f) return SegmentPattern.Gap;
            if (r < 0.68f) return SegmentPattern.Ramp;
            if (r < 0.76f) return SegmentPattern.Pebbles;
            if (r < 0.86f) return SegmentPattern.Overhang;
            if (r < 0.95f) return SegmentPattern.Squeeze;
            return SegmentPattern.Beam;
        }

        private void BuildSegment(int index, SegmentPattern pattern, float y, float nextY, float lane, float halfW)
        {
            float z0 = index * SegmentLength;
            float zMid = z0 + SegmentLength * 0.5f;
            float inset = SegmentLength * k_SegmentInset;
            float usableLen = SegmentLength - 2f * inset;

            switch (pattern)
            {
                case SegmentPattern.Gap:
                {
                    // Void across the full width: no way round, must be jumped.
                    float gap = Mathf.Min(m_Diff.MaxGap, usableLen * 0.6f);
                    float frontLen = (SegmentLength - gap) * 0.5f;
                    AddFloor(z0 + frontLen * 0.5f, 0f, frontLen, m_Diff.TrackWidth, y);
                    AddFloor(z0 + SegmentLength - frontLen * 0.5f, 0f, frontLen, m_Diff.TrackWidth, y);
                    break;
                }
                case SegmentPattern.Beam:
                {
                    // Narrow floor with void either side. Never narrower than the
                    // corridor bound, so it is a balance problem, not a coin flip.
                    float w = BeamWidth();
                    // Camber lives here rather than on full-width ramps: on a beam
                    // this narrow the cross-slope is w*sin(camber), which stays
                    // small enough to be a balance problem instead of a cliff.
                    float roll = RandRange(-m_Diff.Camber, m_Diff.Camber);
                    var beamRot = Quaternion.Euler(0f, 0f, roll);
                    AddBox(new Vector3(lane, y - FloorThickness * 0.5f, zMid), beamRot,
                        new Vector3(w * 0.5f, FloorThickness * 0.5f, SegmentLength * 0.5f),
                        true, true);
                    break;
                }
                case SegmentPattern.Ramp:
                {
                    // A slab whose TOP SURFACE runs from y at the start of the ramp
                    // to nextY at its end, with flat ground either side. Placing a
                    // pitched box on top of a flat floor (the first attempt) makes
                    // the ramp a step rather than a slope, because the floor stays
                    // at y underneath it.
                    //
                    // Camber is deliberately not applied here. Rolling a full-width
                    // slab tilts the whole track sideways: at 8.5m wide and 22 deg
                    // that is a 3.2m cross-slope, which is a cliff, not a camber.
                    // Off-camber belongs on narrow features -- see Beam.
                    float rise = nextY - y;
                    float pitch = Mathf.Max(5f, m_Diff.RampPitch);
                    float run = Mathf.Abs(rise) / Mathf.Tan(pitch * Mathf.Deg2Rad);
                    run = Mathf.Clamp(run, 1f, usableLen);

                    float rampZ0 = zMid - run * 0.5f;
                    float rampZ1 = zMid + run * 0.5f;
                    float leadLen = rampZ0 - z0;
                    float tailLen = (z0 + SegmentLength) - rampZ1;
                    if (leadLen > 0.05f) AddFloor(z0 + leadLen * 0.5f, 0f, leadLen, m_Diff.TrackWidth, y);
                    if (tailLen > 0.05f) AddFloor(rampZ1 + tailLen * 0.5f, 0f, tailLen, m_Diff.TrackWidth, nextY);

                    float theta = Mathf.Atan2(rise, run);
                    float slabLen = Mathf.Sqrt(run * run + rise * rise);
                    const float slabThick = 0.5f;
                    // Negative X rotation tilts +Z upward in Unity's convention.
                    var rot = Quaternion.Euler(-theta / Mathf.Deg2Rad, 0f, 0f);
                    // Drop the centre half a thickness along the slab's own down
                    // axis so the TOP face, not the centre, hits the two heights.
                    var surfaceMid = new Vector3(0f, (y + nextY) * 0.5f, zMid);
                    var center = surfaceMid + (rot * Vector3.up) * (-slabThick * 0.5f);
                    AddBox(center, rot,
                        new Vector3(m_Diff.TrackWidth * 0.5f, slabThick * 0.5f, slabLen * 0.5f),
                        false, true);
                    break;
                }
                case SegmentPattern.StepField:
                {
                    AddFloor(z0 + SegmentLength * 0.25f, 0f, SegmentLength * 0.5f, m_Diff.TrackWidth, y);
                    AddFloor(z0 + SegmentLength * 0.75f, 0f, SegmentLength * 0.5f, m_Diff.TrackWidth, nextY);
                    break;
                }
                default:
                {
                    AddFloor(zMid, 0f, SegmentLength, m_Diff.TrackWidth, y);
                    break;
                }
            }

            switch (pattern)
            {
                case SegmentPattern.Hurdle:
                {
                    // Spans the full width, so it cannot be walked around -- the
                    // fall-off edges are what make that true. Bounded by the
                    // mountable wall height.
                    float h = Mathf.Lerp(0.3f, m_Diff.WallHeight, (float)m_Rng.NextDouble());
                    AddBox(
                        new Vector3(0f, y + h * 0.5f, zMid),
                        Quaternion.Euler(0f, RandRange(-12f, 12f), 0f),
                        new Vector3(m_Diff.TrackWidth * 0.6f, h * 0.5f, RandRange(0.2f, 0.5f)),
                        false, false);
                    break;
                }
                case SegmentPattern.Slalom:
                {
                    // Partial walls from alternating sides, staggered along z. Each
                    // leaves the lane clear, so the route exists but weaves.
                    int count = Mathf.Clamp(m_Diff.ObstaclesPerSegment, 1, 3);
                    for (int i = 0; i < count; i++)
                    {
                        float t = (i + 0.5f) / count;
                        float z = z0 + inset + usableLen * t;
                        bool fromLeft = (i % 2) == 0;
                        // Wall runs from one edge up to the lane boundary.
                        float laneEdge = lane + (fromLeft ? -m_Diff.MinCorridor * 0.5f : m_Diff.MinCorridor * 0.5f);
                        float edge = fromLeft ? -halfW : halfW;
                        float w = Mathf.Abs(laneEdge - edge);
                        if (w < 0.4f) continue;
                        AddBox(
                            new Vector3((laneEdge + edge) * 0.5f, y + k_ImpassableWallHeight * 0.5f, z),
                            Quaternion.identity,
                            new Vector3(w * 0.5f, k_ImpassableWallHeight * 0.5f, RandRange(0.2f, 0.4f)),
                            false, false);
                    }
                    break;
                }
                case SegmentPattern.Squeeze:
                {
                    // Two tall walls leaving exactly the corridor bound between
                    // them. At high difficulty that is 1.35 against a 3.5 splayed
                    // stance, so the legs must tuck and the body must yaw-align.
                    float half = m_Diff.MinCorridor * 0.5f;
                    float wallH = k_ImpassableWallHeight;
                    float leftEdge = lane - half;
                    float rightEdge = lane + half;
                    float wl = Mathf.Abs(leftEdge - (-halfW));
                    float wr = Mathf.Abs(halfW - rightEdge);
                    float depth = Mathf.Min(usableLen * 0.5f, 2.5f);
                    if (wl > 0.4f)
                    {
                        AddBox(new Vector3((leftEdge - halfW) * 0.5f, y + wallH * 0.5f, zMid),
                            Quaternion.identity,
                            new Vector3(wl * 0.5f, wallH * 0.5f, depth * 0.5f), false, false);
                    }
                    if (wr > 0.4f)
                    {
                        AddBox(new Vector3((rightEdge + halfW) * 0.5f, y + wallH * 0.5f, zMid),
                            Quaternion.identity,
                            new Vector3(wr * 0.5f, wallH * 0.5f, depth * 0.5f), false, false);
                    }
                    break;
                }
                case SegmentPattern.Overhang:
                {
                    // Floating slab spanning the full width. Its underside sits at
                    // the clearance bound, and its z depth is long enough that the
                    // agent has to stay low rather than dip under and pop up.
                    float clear = m_Diff.MinClearance;
                    float depth = Mathf.Min(usableLen, Mathf.Lerp(1.5f, 4.5f, (float)m_Rng.NextDouble()));
                    // The slab has to be thick enough that going OVER it is not an
                    // option, or the pattern silently degenerates into a vault and
                    // never forces the low gait it exists to force. Worst case is
                    // d=1: clearance 1.15 with a mountable height of 2.2, so the
                    // top must clear 3.35 and the slab must be at least 2.2 thick.
                    float slabHalf = k_OverhangSlabHalfThickness;
                    AddBox(
                        new Vector3(0f, y + clear + slabHalf, zMid),
                        Quaternion.identity,
                        new Vector3(m_Diff.TrackWidth * 0.6f, slabHalf, depth * 0.5f),
                        false, false);
                    break;
                }
                case SegmentPattern.Pebbles:
                {
                    // Trip hazards. Small enough that a centre-of-mass node
                    // describes them well, which is exactly the regime where the
                    // surface-point re-parameterisation is a no-op.
                    float area = usableLen * m_Diff.TrackWidth;
                    int n = Mathf.Clamp(Mathf.RoundToInt(area * m_Diff.PebbleDensity * 0.05f), 0, 12);
                    for (int i = 0; i < n; i++)
                    {
                        float s = RandRange(0.15f, 0.5f);
                        AddBox(
                            new Vector3(RandRange(-halfW, halfW), y + s * 0.5f, RandRange(z0 + inset, z0 + SegmentLength - inset)),
                            Quaternion.Euler(RandRange(0f, 360f), RandRange(0f, 360f), RandRange(0f, 360f)),
                            new Vector3(s * 0.5f, s * 0.5f, s * 0.5f),
                            false, false);
                    }
                    break;
                }
            }
        }

        private float BeamWidth() => Mathf.Max(m_Diff.MinCorridor, 1.4f);

        private float RandRange(float a, float b) => a + (float)m_Rng.NextDouble() * (b - a);

        private void AddFloor(float z, float x, float lenZ, float width, float topY)
        {
            AddBox(
                new Vector3(x, topY - FloorThickness * 0.5f, z),
                Quaternion.identity,
                new Vector3(width * 0.5f, FloorThickness * 0.5f, lenZ * 0.5f),
                true, true);
        }

        private ParkourBox AddBox(Vector3 center, Quaternion rot, Vector3 half, bool isFloor, bool walkable)
        {
            if (m_Active.Count >= MaxBoxes) return null;
            ParkourBox box;
            if (m_Active.Count < m_Pool.Count)
            {
                box = m_Pool[m_Active.Count];
            }
            else
            {
                box = new ParkourBox();
                if (BoxPrefab != null)
                {
                    var go = Instantiate(BoxPrefab, BoxParent != null ? BoxParent : transform);
                    box.Tr = go.transform;
                }
                m_Pool.Add(box);
            }
            box.IsFloor = isFloor || walkable;
            box.IsDynamic = false;
            box.Place(center, rot, half);
            if (box.Tr != null)
            {
                box.Tr.gameObject.SetActive(true);
                var mr = box.Tr.GetComponent<MeshRenderer>();
                if (mr != null)
                {
                    var mat = box.IsFloor ? FloorMaterial : ObstacleMaterial;
                    if (mat != null) mr.sharedMaterial = mat;
                }
            }
            m_Active.Add(box);
            return box;
        }

        /// <summary>
        /// Re-derives passability from the placed geometry: rasterise the width
        /// into bins, find the walkable surface and the lowest thing above it in
        /// each, and require a contiguous passable run at least MinCorridor wide.
        /// </summary>
        /// <remarks>
        /// This is deliberately independent of the placement logic. If it agreed
        /// with the generator by construction it would only be restating the
        /// generator's assumptions and could not catch a pattern bug.
        /// </remarks>
        private bool VerifyCorridor(int index, float y, float nextY)
        {
            if (Mathf.Abs(nextY - y) > m_Diff.MaxStep + 1e-3f) return false;

            float halfW = m_Diff.TrackWidth * 0.5f;
            int bins = Mathf.Max(2, Mathf.CeilToInt(m_Diff.TrackWidth / k_BinWidth));
            float binW = m_Diff.TrackWidth / bins;
            float z0 = index * SegmentLength;

            const int kSlices = 5;
            for (int s = 0; s < kSlices; s++)
            {
                float z = z0 + SegmentLength * (s + 0.5f) / kSlices;
                int run = 0, best = 0, withFloor = 0;

                // Broad-phase: only boxes whose world AABB spans this slice can
                // matter. Without it the finer bin resolution would multiply the
                // per-bin scan over every box on the track.
                m_SliceCandidates.Clear();
                for (int i = 0; i < m_Active.Count; i++)
                {
                    var box = m_Active[i];
                    if (z < box.WorldMin.z || z > box.WorldMax.z) continue;
                    m_SliceCandidates.Add(box);
                }

                for (int b = 0; b < bins; b++)
                {
                    float x = -halfW + (b + 0.5f) * binW;

                    // Highest walkable surface in this column.
                    float floorTop = float.NegativeInfinity;
                    for (int i = 0; i < m_SliceCandidates.Count; i++)
                    {
                        var box = m_SliceCandidates[i];
                        if (!box.IsFloor) continue;
                        if (x < box.WorldMin.x || x > box.WorldMax.x) continue;
                        if (!box.VerticalSpanAt(x, z, out _, out float to)) continue;
                        if (to > floorTop) floorTop = to;
                    }
                    if (float.IsNegativeInfinity(floorTop))
                    {
                        run = 0;   // void column
                        continue;
                    }
                    withFloor++;

                    // An obstacle resting on the ground is a barrier the agent has
                    // to get *over*; one that floats is a ceiling it has to get
                    // *under*. Collapsing the two into a single "clearance" test
                    // was wrong: it made every jumpable hurdle look impassable.
                    bool blocked = false;
                    for (int i = 0; i < m_SliceCandidates.Count; i++)
                    {
                        var box = m_SliceCandidates[i];
                        if (box.IsFloor) continue;
                        if (x < box.WorldMin.x || x > box.WorldMax.x) continue;
                        if (!box.VerticalSpanAt(x, z, out float bo, out float to)) continue;
                        if (to <= floorTop + 1e-3f) continue;      // buried in the floor

                        if (bo <= floorTop + 0.05f)
                        {
                            // Rests on the ground: surmountable only up to the
                            // mountable wall height.
                            if (to - floorTop > m_Diff.WallHeight) { blocked = true; break; }
                        }
                        else if (bo - floorTop < m_Diff.MinClearance - 1e-3f)
                        {
                            // Floats too low to fit under.
                            blocked = true;
                            break;
                        }
                    }

                    run = blocked ? 0 : run + 1;
                    if (run > best) best = run;
                }

                // A slice with no ground anywhere is a gap, which is legal by
                // construction (bounded by MaxGap at placement) and is crossed
                // longitudinally rather than laterally.
                if (withFloor == 0) continue;
                // +1 bin corrects the rasteriser's systematic under-measurement of
                // a free run (partial bins at each end are dropped).
                if ((best + 1) * binW < m_Diff.MinCorridor) return false;
            }
            return true;
        }

        /// <summary>Highest walkable surface at a world column, for spawning and
        /// respawning. Returns false where there is no ground.</summary>
        public bool GroundHeightAt(float x, float z, out float top)
        {
            top = float.NegativeInfinity;
            for (int i = 0; i < m_Active.Count; i++)
            {
                var b = m_Active[i];
                if (!b.IsFloor) continue;
                if (!b.VerticalSpanAt(x, z, out float bo, out float to)) continue;
                if (to > top) top = to;
            }
            return !float.IsNegativeInfinity(top);
        }

        /// <summary>Lane centre for the segment containing <paramref name="z"/>.
        /// Respawns use it so the agent restarts on the traversable route.</summary>
        public float LaneCenterAt(float z)
        {
            if (m_LaneCenter.Count == 0) return 0f;
            int i = Mathf.Clamp(Mathf.FloorToInt(z / SegmentLength), 0, m_LaneCenter.Count - 1);
            return m_LaneCenter[i];
        }
    }
}
