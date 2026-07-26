"""
Fixed-anchor tournament evaluator for CrawlerSumo self-play checkpoints.

Matches run NATIVELY inside a Unity player (32 arenas, in-process Sentis
inference, timescale 20) -- the python low-level API stalls on Windows
headless builds, so python only orchestrates:
  staged .onnx -> .sentis (editor batch, cached) -> eval player run
  (per side-swap) -> JSON outcomes -> aggregate CSV + summary.

Unlike the trainer's Self-play/ELO (measured against a moving snapshot
window), this gives absolute skill comparisons between fixed checkpoints.

Usage (any python 3; no ML deps needed):
  python scripts/evaluate_checkpoints.py                    # newest vs earliest
  python scripts/evaluate_checkpoints.py --model-a 10499757 --model-b 5499779
  python scripts/evaluate_checkpoints.py --curve            # spread vs anchor
Results append to <staging>/eval/results.csv.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# Repo-relative, not absolute: with one worktree per project each has its own
# staging root, builds and Unity project, and absolute paths would silently
# read another project's data. Unity itself is machine-wide.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.environ.get("UNITYRL_STAGING") or os.path.join(REPO, "results", "cloud_staging")
EVAL_EXE = os.path.join(REPO, "envs", "CrawlerSumoEGNN_Multi_win", "UnityEnvironment.exe")
UNITY_EXE = r"C:\Program Files\Unity\Hub\Editor\6000.0.40f1\Editor\Unity.exe"
PROJECT = os.path.join(REPO, "Project")


def step_of(path):
    return int(re.search(r"-(\d+)\.(onnx|sentis)$", path).group(1))


def find_checkpoints(run_id):
    ckpts = sorted(
        glob.glob(os.path.join(STAGING, run_id, "CrawlerSumo", "CrawlerSumo-*.onnx")),
        key=step_of,
    )
    if not ckpts:
        sys.exit(f"No checkpoints found for {run_id}")
    return ckpts


def newest_run():
    runs = [
        d for d in os.listdir(STAGING)
        if os.path.isdir(os.path.join(STAGING, d, "CrawlerSumo"))
    ]
    if not runs:
        sys.exit("No runs in staging")
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(STAGING, d)))


def ensure_sentis(run_id, onnx_paths):
    """Convert any missing .onnx -> .sentis via a Unity editor batch call."""
    sentis_dir = os.path.join(STAGING, "_sentis", run_id)
    os.makedirs(sentis_dir, exist_ok=True)
    missing = [
        p for p in onnx_paths
        if not os.path.exists(os.path.join(
            sentis_dir, os.path.splitext(os.path.basename(p))[0] + ".sentis"))
    ]
    if missing:
        print(f"converting {len(missing)} onnx -> sentis (editor batch, ~1 min)...")
        env = dict(os.environ)
        env["ONNX_CONVERT_LIST"] = ";".join(missing)
        env["SENTIS_OUT_DIR"] = sentis_dir
        rc = subprocess.run(
            [UNITY_EXE, "-projectPath", PROJECT, "-batchmode", "-nographics",
             "-executeMethod", "TrainingBuilds.ConvertOnnxToSentis",
             "-logFile", os.path.join(sentis_dir, "convert.log")],
            env=env, timeout=600,
        ).returncode
        if rc != 0:
            sys.exit(f"sentis conversion failed (exit {rc}); see {sentis_dir}\\convert.log")
    return {
        step_of(p): os.path.join(
            sentis_dir, os.path.splitext(os.path.basename(p))[0] + ".sentis")
        for p in onnx_paths
    }


def run_eval(team0_sentis, team1_sentis, episodes, time_scale, timeout, max_steps,
             env_params=()):
    result_file = os.path.join(tempfile.gettempdir(), f"eval_{os.getpid()}_{time.time_ns()}.json")
    cmd = [
        EVAL_EXE,
        "--team0-model", team0_sentis, "--team1-model", team1_sentis,
        "--eval-episodes", str(episodes), "--result-json", result_file,
        "--time-scale", str(time_scale), "--eval-timeout", str(timeout),
        "--max-episode-steps", str(max_steps),
        "-screen-width", "320", "-screen-height", "180", "-screen-fullscreen", "0",
    ]
    # Physics/env overrides so eval can match a run's training physics
    # (standalone builds have no python side channel and default otherwise)
    for p in env_params:
        cmd += ["--env-param", p]
    subprocess.run(cmd, timeout=timeout + 120)
    with open(result_file) as f:
        result = json.load(f)
    os.remove(result_file)
    return result


def play_matchup(sentis_a, sentis_b, episodes, time_scale, timeout, max_steps,
                 env_params=()):
    """Side-swapped matchup. Returns (a_wins, b_wins, draws, timed_out)."""
    half = episodes // 2
    r1 = run_eval(sentis_a, sentis_b, half, time_scale, timeout, max_steps, env_params)
    r2 = run_eval(sentis_b, sentis_a, episodes - half, time_scale, timeout, max_steps, env_params)
    a = r1["team0_wins"] + r2["team1_wins"]
    b = r1["team1_wins"] + r2["team0_wins"]
    d = r1["draws"] + r2["draws"]
    return a, b, d, r1["timed_out"] or r2["timed_out"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--model-a", type=int, default=None, help="challenger step (default newest)")
    ap.add_argument("--model-b", type=int, default=None, help="opponent step (default earliest)")
    ap.add_argument("--curve", action="store_true",
                    help="evaluate ~4 spread checkpoints against the earliest anchor")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-side eval timeout (s)")
    ap.add_argument("--max-episode-steps", type=int, default=3000,
                    help="eval standard: 3000 (2x training cap) so matches resolve; "
                         "evaluation play stalls longer than noisy training rollouts")
    ap.add_argument("--env-param", action="append", default=[], metavar="NAME=VALUE",
                    help="repeatable env-param override forwarded to the eval build "
                         "(match the run's training physics, e.g. "
                         "joint_strength_multiplier_min=1.5)")
    args = ap.parse_args()

    run_id = args.run_id or newest_run()
    ckpts = find_checkpoints(run_id)
    steps_sorted = [step_of(p) for p in ckpts]

    if args.curve:
        anchor = steps_sorted[0]
        n = len(steps_sorted)
        challengers = sorted({steps_sorted[max(0, round(i * (n - 1) / 4))] for i in range(1, 5)})
        matchups = [(c, anchor) for c in challengers if c != anchor]
    else:
        def nearest(step):
            return min(steps_sorted, key=lambda s: abs(s - step))
        a = nearest(args.model_a) if args.model_a else steps_sorted[-1]
        b = nearest(args.model_b) if args.model_b else steps_sorted[0]
        matchups = [(a, b)]

    involved = sorted({s for m in matchups for s in m})
    by_step_onnx = {step_of(p): p for p in ckpts}
    sentis = ensure_sentis(run_id, [by_step_onnx[s] for s in involved])

    out_csv = os.path.join(STAGING, "eval", "results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    print(f"run {run_id}: {len(matchups)} matchup(s), {args.episodes} episodes each")
    for a_step, b_step in matchups:
        t0 = time.time()
        wa, wb, dr, to = play_matchup(
            sentis[a_step], sentis[b_step], args.episodes, args.time_scale,
            args.timeout, args.max_episode_steps, args.env_param)
        n = max(1, wa + wb + dr)
        flag = "  [TIMED OUT — partial]" if to else ""
        print(f"  {a_step/1e6:.1f}M vs {b_step/1e6:.1f}M:  "
              f"W {wa} ({100*wa/n:.0f}%)  L {wb} ({100*wb/n:.0f}%)  D {dr} ({100*dr/n:.0f}%)"
              f"   [{time.time()-t0:.0f}s]{flag}")
        new = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "run", "model_a_step", "model_b_step",
                            "episodes", "a_wins", "b_wins", "draws", "timed_out"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), run_id, a_step, b_step,
                        n, wa, wb, dr, to])


if __name__ == "__main__":
    main()
