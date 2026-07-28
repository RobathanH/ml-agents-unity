// Validates the REAL ParkourTrackGenerator (compiled against a Unity shim) with a
// feasibility test that is deliberately NOT the generator's own check: a forward
// path search over the placed geometry, rather than a per-segment corridor width.
// If both agree across thousands of layouts, the guarantee is worth something.
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using CrawlerParkour;

static class Program
{
    // The sampling grid must be INCOMMENSURATE with the geometry, and finer than
    // the narrowest feature it has to resolve.
    //
    // This bit once, and cost an hour. At DZ = 0.4 the slice centres are 0.4n + 0.2,
    // while a Gap at d=0 has its edges at 8k + 3.8 and 8k + 4.2 -- both exactly
    // slice centres, for every k. Whether each edge then read as floor or as void
    // came down to float rounding of (iz + 0.5f) * 0.4f against the box bound, which
    // varies with the magnitude of iz. So a perfectly legal 0.4m gap rasterised as a
    // 1.2m one, but only in some segments, deterministically by position.
    // Lengthening the track from 16 to 24 segments turned that from invisible into
    // 842 false failures out of 7500 -- and the generator was never wrong, its own
    // VerifyCorridor reported 0 repairs at d=0 throughout.
    //
    // 0.17 shares no factor with the 8m segment length or any feature offset, and is
    // well under the 0.4m narrowest gap. It costs ~2.4x the slices, paid for below
    // by fewer seeds.
    const float DZ = 0.17f;
    const float DX = 0.25f;
    // Lateral movement allowed per forward step. 45 degrees is well inside what a
    // crawler moving forward can strafe, so the search is conservative.
    const float LATERAL_PER_STEP = 1.0f;

    enum Col { Void, Blocked, Ok }

    static Col Classify(IReadOnlyList<ParkourBox> boxes, ParkourDifficulty d,
                        float x, float z, out float surface)
    {
        surface = float.NegativeInfinity;
        float floorTop = float.NegativeInfinity;
        foreach (var b in boxes)
        {
            if (!b.IsFloor) continue;
            if (!b.VerticalSpanAt(x, z, out _, out float to)) continue;
            if (to > floorTop) floorTop = to;
        }
        if (float.IsNegativeInfinity(floorTop)) return Col.Void;

        surface = floorTop;
        foreach (var b in boxes)
        {
            if (b.IsFloor) continue;
            if (!b.VerticalSpanAt(x, z, out float bo, out float to)) continue;
            if (to <= floorTop + 1e-3f) continue;

            bool resting = bo <= floorTop + 0.05f;
            bool fitsUnder = !resting && (bo - floorTop) >= d.MinClearance - 1e-3f;
            if (fitsUnder) continue;
            // Cannot fit under: the only other option is to get on top of it.
            if (to - floorTop > d.WallHeight) return Col.Blocked;
            if (to > surface) surface = to;
        }
        return Col.Ok;
    }

    // Reachability from the start line to the finish, stepping forward in DZ and
    // allowing a bounded lateral move, a bounded step up, and a jump across a void
    // no wider than MaxGap.
    static bool Traversable(ParkourTrackGenerator gen, out float reachedZ)
    {
        var boxes = gen.ActiveBoxes;
        var d = gen.Difficulty;
        float halfW = d.TrackWidth * 0.5f;
        int nx = Math.Max(2, (int)Math.Ceiling(d.TrackWidth / DX));
        int nz = (int)(gen.TrackLength / DZ);
        int lat = Math.Max(1, (int)(LATERAL_PER_STEP / DX));
        int maxJumpSteps = Math.Max(1, (int)Math.Ceiling(d.MaxGap / DZ)) + 1;

        var cur = new bool[nx];
        var surf = new float[nx];
        var state = new Col[nz][];
        var height = new float[nz][];
        for (int iz = 0; iz < nz; iz++)
        {
            state[iz] = new Col[nx];
            height[iz] = new float[nx];
            float z = (iz + 0.5f) * DZ;
            for (int ix = 0; ix < nx; ix++)
            {
                float x = -halfW + (ix + 0.5f) * (d.TrackWidth / nx);
                state[iz][ix] = Classify(boxes, d, x, z, out float s);
                height[iz][ix] = s;
            }
        }

        var reach = new bool[nz][];
        for (int iz = 0; iz < nz; iz++) reach[iz] = new bool[nx];
        for (int ix = 0; ix < nx; ix++)
            if (state[0][ix] == Col.Ok) reach[0][ix] = true;

        reachedZ = 0f;
        for (int iz = 0; iz < nz; iz++)
        {
            bool any = false;
            for (int ix = 0; ix < nx; ix++) if (reach[iz][ix]) { any = true; break; }
            if (!any) continue;
            reachedZ = (iz + 1) * DZ;

            for (int ix = 0; ix < nx; ix++)
            {
                if (!reach[iz][ix]) continue;
                // Walk / climb to the next slice, or jump a void up to MaxGap.
                for (int step = 1; step <= maxJumpSteps; step++)
                {
                    int jz = iz + step;
                    if (jz >= nz) break;
                    for (int jx = Math.Max(0, ix - lat); jx <= Math.Min(nx - 1, ix + lat); jx++)
                    {
                        if (state[jz][jx] != Col.Ok) continue;
                        float rise = height[jz][jx] - height[iz][ix];
                        // Dropping is always allowed; climbing is bounded by the
                        // mountable height. NOT by MaxStep -- that is a generation
                        // constraint on how much the *floor* may change between
                        // segments, not a limit on what the agent can climb.
                        if (rise > d.WallHeight + 1e-3f) continue;
                        reach[jz][jx] = true;
                    }
                    // Only keep extending the jump while the intervening slices
                    // really are void -- a jump does not pass through solid walls.
                    bool allVoid = true;
                    for (int jx = 0; jx < nx; jx++)
                        if (state[iz + step][jx] != Col.Void) { allVoid = false; break; }
                    if (!allVoid) break;
                }
            }
        }

        for (int ix = 0; ix < nx; ix++) if (reach[nz - 1][ix]) return true;
        return false;
    }

    static void Dump(ParkourTrackGenerator gen, float zBlock)
    {
        var d = gen.Difficulty;
        float halfW = d.TrackWidth * 0.5f;
        Console.WriteLine($"     segment {(int)(zBlock / gen.SegmentLength)}, boxes near z={zBlock:F1}:");
        foreach (var b in gen.ActiveBoxes)
        {
            if (Math.Abs(b.Center.z - zBlock) > 5f) continue;
            Console.WriteLine($"       {(b.IsFloor ? "floor" : "obst ")} c={b.Center} half={b.HalfExtents}");
        }
        for (float z = zBlock - 1.2f; z <= zBlock + 2.4f; z += 0.4f)
        {
            var row = new System.Text.StringBuilder($"     z={z,6:F1} ");
            for (int ix = 0; ix < 40; ix++)
            {
                float x = -halfW + (ix + 0.5f) * (d.TrackWidth / 40);
                var c = Classify(gen.ActiveBoxes, d, x, z, out _);
                row.Append(c == Col.Ok ? '.' : c == Col.Void ? ' ' : '#');
            }
            Console.WriteLine(row.ToString());
        }
    }

    /// Vertical gap at a column between the walkable surface and the lowest thing
    /// hanging over it. Negative infinity where there is no ground at all, 0 where
    /// something rests on the surface and blocks the column outright.
    static float Headroom(IReadOnlyList<ParkourBox> boxes, float x, float z)
    {
        float floorTop = float.NegativeInfinity;
        foreach (var b in boxes)
        {
            if (!b.IsFloor) continue;
            if (!b.VerticalSpanAt(x, z, out _, out float to)) continue;
            if (to > floorTop) floorTop = to;
        }
        if (float.IsNegativeInfinity(floorTop)) return float.NegativeInfinity;

        float ceiling = float.PositiveInfinity;
        foreach (var b in boxes)
        {
            if (b.IsFloor) continue;
            if (!b.VerticalSpanAt(x, z, out float bo, out float to)) continue;
            if (to <= floorTop + 1e-3f) continue;               // buried in the floor
            if (bo <= floorTop + 0.05f) return 0f;              // a wall, not a ceiling
            if (bo < ceiling) ceiling = bo;
        }
        return ceiling - floorTop;
    }

    /// Run 006's mid-track spawn. The generator's claim is that a segment boundary
    /// is clear ground; this re-derives that from the geometry that actually got
    /// placed, at the three columns the ~3.9m body occupies.
    ///
    /// The number that matters is 1.2 -- the controller puts the body centre that
    /// far above the surface, while MinClearance falls to 1.15 at d=1. An Overhang
    /// over a spawn column would embed the crawler inside the slab on frame one,
    /// which would read as a mysteriously bad policy rather than as a placement bug.
    const float kZJitter = 0.5f;

    static void CheckSpawns(ParkourTrackGenerator gen, int reserve, int[] fails,
                            ref float worstHeadroom, ref float worstRunway)
    {
        const float kRideHeight = 1.2f;
        const float kBodyHalf = 2f;
        var boxes = gen.ActiveBoxes;

        for (int t = 0; t < 8; t++)
        {
            var rng = new Random(t * 7919 + 13);
            if (!gen.TryPickSpawn(rng, reserve, kZJitter, out var p)) { fails[0]++; continue; }

            // Runway: an episode must not be able to reach the end of the track.
            // The jitter is allowed to eat into the reserve -- it is +/-0.5m against
            // a reserve measured in 8m segments -- so it is credited here rather
            // than counted as a failure.
            float runway = gen.TrackLength - p.z;
            if (runway < worstRunway) worstRunway = runway;
            if (runway < reserve * gen.SegmentLength - kZJitter) fails[1]++;

            foreach (var dz in new[] { -kBodyHalf, 0f, kBodyHalf })
            {
                float h = Headroom(boxes, p.x, p.z + dz);
                if (float.IsNegativeInfinity(h)) { fails[2]++; break; }   // over the void
                if (h < worstHeadroom) worstHeadroom = h;
                if (h < kRideHeight + 0.3f) { fails[3]++; break; }
            }
        }
    }

    static int Main()
    {
        var diffs = new[] { 0f, 0.25f, 0.5f, 0.75f, 1f };
        const int seeds = 600;   // see DZ: finer slices, fewer layouts, same guarantee
        const int spawnReserve = 10;
        int failures = 0, totalRepairs = 0;
        var spawnFails = new int[4];      // none / runway / void / headroom
        float spawnWorstHeadroom = float.PositiveInfinity;
        float spawnWorstRunway = float.PositiveInfinity;

        Console.WriteLine("difficulty bounds:");
        Console.WriteLine($"{"d",5} {"width",7} {"step",6} {"wall",6} {"gap",6} {"clear",7} {"corr",6} {"pitch",6} {"peb",5}");
        foreach (var dv in diffs)
        {
            var p = ParkourDifficulty.From(dv);
            Console.WriteLine($"{dv,5:F2} {p.TrackWidth,7:F2} {p.MaxStep,6:F2} {p.WallHeight,6:F2} {p.MaxGap,6:F2} " +
                              $"{p.MinClearance,7:F2} {p.MinCorridor,6:F2} {p.RampPitch,6:F1} {p.PebbleDensity,5:F2}");
        }
        Console.WriteLine();

        foreach (var dv in diffs)
        {
            int fail = 0, repairs = 0;
            var boxCounts = new List<int>();
            for (int s = 0; s < seeds; s++)
            {
                var gen = new ParkourTrackGenerator();
                gen.Generate(1000 + s, dv);
                repairs += gen.RepairCount;
                boxCounts.Add(gen.ActiveBoxes.Count);
                CheckSpawns(gen, spawnReserve, spawnFails, ref spawnWorstHeadroom, ref spawnWorstRunway);
                if (!Traversable(gen, out float reached))
                {
                    fail++;
                    if (fail <= 1)
                    {
                        Console.WriteLine($"  !! d={dv:F2} seed={1000 + s} unreachable past z={reached:F1} " +
                                          $"of {gen.TrackLength:F0} ({gen.ActiveBoxes.Count} boxes)");
                        Dump(gen, reached);
                    }
                }
            }
            failures += fail;
            totalRepairs += repairs;
            Console.WriteLine($"d={dv:F2}: {seeds - fail}/{seeds} traversable, repairs={repairs}, " +
                              $"boxes avg={boxCounts.Average():F0} max={boxCounts.Max()}");
        }

        Console.WriteLine();
        int spawnTotal = diffs.Length * seeds * 8;
        Console.WriteLine($"mid-track spawn ({spawnTotal:N0} picks, reserve {spawnReserve} segments):");
        Console.WriteLine($"  no candidate found   {spawnFails[0]}");
        Console.WriteLine($"  runway too short     {spawnFails[1]}");
        Console.WriteLine($"  spawned over void    {spawnFails[2]}");
        Console.WriteLine($"  headroom too low     {spawnFails[3]}");
        Console.WriteLine($"  worst headroom       {spawnWorstHeadroom:F2} m  (need >= 1.50 for a 1.20 ride height)");
        Console.WriteLine($"  worst runway ahead   {spawnWorstRunway:F1} m  (reserve = {spawnReserve * 8} m)");

        Console.WriteLine();
        // The tightest cases at maximum difficulty, against the measured envelope.
        var hard = ParkourDifficulty.From(1f);
        Console.WriteLine("worst-case margins at d=1 (body 1.00 across, splay 3.5, tuck 1.4, reach 2.45):");
        Console.WriteLine($"  overhang clearance {hard.MinClearance:F2} vs body 1.00 -> margin {hard.MinClearance - 1.0f:F2}");
        Console.WriteLine($"  corridor {hard.MinCorridor:F2} vs tucked span 1.40 -> margin {hard.MinCorridor - 1.40f:F2}");
        Console.WriteLine($"  step {hard.MaxStep:F2} vs leg reach 2.45 -> headroom {2.45f - hard.MaxStep:F2}");
        Console.WriteLine($"  wall {hard.WallHeight:F2} vs rear-up 1.80 -> must jump {hard.WallHeight - 1.80f:F2}");
        Console.WriteLine($"  gap {hard.MaxGap:F2} vs splayed span 3.50 -> {hard.MaxGap / 3.5f:P0} of stance");
        Console.WriteLine($"  overhang slab top = {hard.MinClearance + 2.4f:F2} vs mountable {hard.WallHeight:F2} -> " +
                          (hard.MinClearance + 2.4f > hard.WallHeight ? "cannot be vaulted (correct)" : "VAULTABLE (bug)"));

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? $"ALL TRAVERSABLE ({diffs.Length * seeds} layouts, {totalRepairs} repairs)"
            : $"FAILURES: {failures}/{diffs.Length * seeds}");
        int spawnBad = spawnFails.Sum();
        Console.WriteLine(spawnBad == 0
            ? $"ALL SPAWNS USABLE ({spawnTotal:N0} picks)"
            : $"SPAWN FAILURES: {spawnBad}/{spawnTotal} [none/runway/void/headroom = "
              + string.Join("/", spawnFails) + "]");
        return failures == 0 && spawnBad == 0 ? 0 : 1;
    }
}
