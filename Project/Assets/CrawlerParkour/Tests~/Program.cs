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
    const float DZ = 0.4f;
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

    static int Main()
    {
        var diffs = new[] { 0f, 0.25f, 0.5f, 0.75f, 1f };
        const int seeds = 1500;
        int failures = 0, totalRepairs = 0;

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
        return failures == 0 ? 0 : 1;
    }
}
