"""Prove that an expanded run 009 checkpoint is run 008's policy, exactly.

expand_checkpoint.py claims the transformed network emits the same JOINT ANGLES as
the original for every input. That claim is checkable without building either
network or launching Unity, and it is worth checking rather than trusting, because
a wrong column offset produces a checkpoint that loads cleanly, trains happily, and
is a scrambled policy -- which reads as "the warm start did not help".

The proof rests on the transform touching only the two ends of the network:

    [ vector obs ] -> normalizer -> x_self_encoder ---.
                                                       >-- (128 wide, IDENTICAL) --.
    [ EGNN entities ] -> egnn -> rsa ------------------'                            |
                                                                                    v
    [ normalized vector obs | attention embedding ] -> _body_endoder -> lstm -> mu head

Everything between the first layer and the last is bit-for-bit unchanged by the
expansion, so if the first layers produce identical outputs for corresponding
inputs, and the last layer produces identical angles for identical hidden states,
the whole network is equivalent. Both halves are checked here on the actual
tensors.

    python scripts/verify_expanded_checkpoint.py \
        --old results/cloud_staging/CrawlerParkour_008/CrawlerParkour/checkpoint.pt \
        --new results/cloud_staging/CrawlerParkour_008_r009/CrawlerParkour/checkpoint.pt
"""
import argparse
import sys

import torch

from expand_checkpoint import (ACTION_CLIP_SCALE, ATTENTION_EMBEDDING, HIP_X_ROWS,
                               HIP_Y_ROWS, KNEE_X_ROWS, NEW_ACT, NEW_LIMITS,
                               OLD_ACT, OLD_LIMITS, STRENGTH_ROWS, build_column_map)

TOL = 2e-5
fails = []


def check(ok, msg):
    print(f"  {'ok  ' if ok else 'FAIL'}  {msg}")
    if not ok:
        fails.append(msg)


def normalize(x, mean, var, steps):
    """Normalizer.forward, verbatim (torch_entities/encoders.py)."""
    return torch.clamp((x - mean) / torch.sqrt(var / steps), -5.0, 5.0)


def env_action(x):
    """What Unity receives, given the head's sampled output x.

    Identical on both paths: AgentAction.to_action_tuple(clip=True) during
    training, ActionModel.forward before ONNX export. Everything downstream --
    including the whole claim this script is checking -- is stated in terms of
    this, not of x.
    """
    return torch.clamp(x, -ACTION_CLIP_SCALE, ACTION_CLIP_SCALE) / ACTION_CLIP_SCALE


def angles(actions, limits, rows):
    """BodyPart.SetJointTargetRotation: angle = lerp(lo, hi, (a + 1) / 2).

    Mathf.Lerp CLAMPS its interpolant, and so does this -- the equivalence has to
    hold under the function Unity actually applies, not the unclamped ideal.
    """
    lo, hi = limits
    t = torch.clamp((actions[:, rows] + 1.0) * 0.5, 0.0, 1.0)
    return lo + (hi - lo) * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    torch.manual_seed(20090)
    old = torch.load(args.old, map_location="cpu", weights_only=False)
    new = torch.load(args.new, map_location="cpu", weights_only=False)
    mapping, n_old, n_new = build_column_map()
    B = args.batch

    print("1. shapes")
    check(new["Policy"]["action_model._continuous_distribution.mu.weight"].shape[0] == NEW_ACT,
          f"action head is {NEW_ACT} wide")
    check(int(new["Policy"]["continuous_act_size_vector"][0]) == NEW_ACT,
          f"continuous_act_size_vector says {int(new['Policy']['continuous_act_size_vector'][0])}"
          " (the ONNX exporter reads this, not the head)")

    # An observation the ENVIRONMENT could actually produce: the two retired
    # track-relative slots are constant zero, which is the fact the deletion rests
    # on. The four new columns are filled with noise rather than zeros on purpose --
    # if the expansion left them any weight at all, that noise moves the output and
    # this test fails.
    o_old = torch.randn(B, n_old)
    o_old[:, 20] = 0.0
    o_old[:, 21] = 0.0
    o_new = torch.zeros(B, n_new)
    for j, i in enumerate(mapping):
        if i >= 0:
            o_new[:, j] = o_old[:, i]
        else:
            o_new[:, j] = torch.rand(B) * 2.0 - 1.0

    print("\n2. the retired observations are constant, but NOT constant zero")
    for module in ("Policy", "Optimizer:critic"):
        sd_o = old[module]
        key = next(k for k in sd_o if k.endswith("processors.1.normalizer.running_mean"))
        base = key[: -len("running_mean")]
        mean, var = sd_o[base + "running_mean"], sd_o[base + "running_variance"]
        steps = sd_o[base + "normalization_steps"].float()
        z = normalize(torch.zeros(1, n_old), mean, var, steps)[0]
        # The naive reading -- "run 006 writes 0.0, so the column is zero" -- is
        # wrong by a third of a standard deviation, because the running mean was
        # inherited from run 005 through four warm starts and decays by only
        # (x - mean)/total_steps per sample. Nonzero here is the CORRECT result;
        # what would be a defect is the value not being constant, and it is
        # constant because the input is.
        check(abs(float(z[20])) > 1e-3,
              f"{module}: columns 20,21 normalise to {float(z[20]):+.4f}, "
              f"{float(z[21]):+.4f} -- a fixed contribution, so folding it into the "
              "bias is required and dropping it silently would not have been free")

    print("\n3. first-layer outputs are identical")
    for module in ("Policy", "Optimizer:critic"):
        sd_o, sd_n = old[module], new[module]
        base = next(k for k in sd_o if k.endswith("processors.1.normalizer.running_mean"))
        base = base[: -len("running_mean")]
        z_old = normalize(o_old, sd_o[base + "running_mean"], sd_o[base + "running_variance"],
                          sd_o[base + "normalization_steps"].float())
        z_new = normalize(o_new, sd_n[base + "running_mean"], sd_n[base + "running_variance"],
                          sd_n[base + "normalization_steps"].float())

        for name in sd_o:
            if not name.endswith(".weight"):
                continue
            w_old, w_new = sd_o[name], sd_n[name]
            if w_old.shape == w_new.shape:
                continue
            if w_old.shape[1] == n_old:                       # x_self_encoder
                bias = sd_o.get(name[: -len("weight")] + "bias")
                y_old = z_old @ w_old.T + bias
                y_new = z_new @ w_new.T + sd_n[name[: -len("weight")] + "bias"]
            elif w_old.shape[1] == n_old + ATTENTION_EMBEDDING:   # _body_endoder
                att = torch.randn(B, ATTENTION_EMBEDDING)
                bias = sd_o.get(name[: -len("weight")] + "bias")
                y_old = torch.cat([z_old, att], 1) @ w_old.T + bias
                y_new = torch.cat([z_new, att], 1) @ w_new.T + sd_n[name[: -len("weight")] + "bias"]
            else:
                continue
            d = float((y_old - y_new).abs().max())
            check(d < TOL, f"{module}: {name.split('.')[-3]} "
                           f"{tuple(w_old.shape)} -> {tuple(w_new.shape)}, "
                           f"max |delta| over {B} observations = {d:.2e}")

    print("\n4. the action head drives the same joint ANGLES")
    p_o, p_n = old["Policy"], new["Policy"]
    w_o = p_o["action_model._continuous_distribution.mu.weight"]
    w_n = p_n["action_model._continuous_distribution.mu.weight"]

    # Hidden states scaled so the head's output lands inside the +/-3 clip, which is
    # where a converged policy lives and where equivalence is claimed. Saturation is
    # tested separately below, because there the claim is different.
    h = torch.randn(B, w_o.shape[1])
    h = h / max(1.0, float((h @ w_o.T).abs().max()) / 2.0)
    mu_o = h @ w_o.T + p_o["action_model._continuous_distribution.mu.bias"]
    mu_n = h @ w_n.T + p_n["action_model._continuous_distribution.mu.bias"]
    a_o, a_n = env_action(mu_o), env_action(mu_n)
    unsaturated = float(mu_o.abs().max()) < ACTION_CLIP_SCALE
    check(unsaturated, f"test inputs stay inside the clip (max |x| = "
                       f"{float(mu_o.abs().max()):.2f} of {ACTION_CLIP_SCALE})")

    for rows, joint in ((HIP_X_ROWS, "hip_x"), (HIP_Y_ROWS, "hip_y"), (KNEE_X_ROWS, "knee_x")):
        d = float((angles(a_o, OLD_LIMITS[joint], rows)
                   - angles(a_n, NEW_LIMITS[joint], rows)).abs().max())
        check(d < 1e-3, f"{joint}: max angle difference {d:.2e} degrees "
                        f"over {B * len(rows)} commands "
                        f"(range {OLD_LIMITS[joint]} -> {NEW_LIMITS[joint]})")

    d = float((a_o[:, STRENGTH_ROWS] - a_n[:, STRENGTH_ROWS]).abs().max())
    check(d < TOL, f"strengths: max |delta| {d:.2e} -- not angles, so not remapped")

    print("\n4b. where run 008 was pinned against a stop, run 009 keeps going")
    # A head output beyond +/-3 meant "further than this joint can travel". The old
    # rig answered with the limit; the new one has travel left. That is not a
    # violation of the remap, it is the widened joint doing its job -- but it must
    # move in the direction the old policy was pushing and never the other way,
    # which is what this checks.
    for rows, joint in ((HIP_X_ROWS, "hip_x"), (HIP_Y_ROWS, "hip_y"), (KNEE_X_ROWS, "knee_x")):
        s = torch.tensor([[-9.0], [-3.0], [3.0], [9.0]]).repeat(1, len(rows))
        lo_o, hi_o = OLD_LIMITS[joint]
        sc = (hi_o - lo_o) / (NEW_LIMITS[joint][1] - NEW_LIMITS[joint][0])
        off = 2.0 * (lo_o - NEW_LIMITS[joint][0]) \
            / (NEW_LIMITS[joint][1] - NEW_LIMITS[joint][0]) + sc - 1.0
        a_old = torch.clamp(s, -3, 3) / 3
        a_new = torch.clamp(s * sc + off * ACTION_CLIP_SCALE, -3, 3) / 3
        ang_o = angles(a_old, OLD_LIMITS[joint], list(range(len(rows))))
        ang_n = angles(a_new, NEW_LIMITS[joint], list(range(len(rows))))
        at_limit = torch.tensor([True, False, False, True])
        agree = float((ang_o - ang_n)[~at_limit].abs().max())
        beyond_lo = float(ang_n[0, 0] - ang_o[0, 0]) <= 1e-4     # further negative
        beyond_hi = float(ang_n[3, 0] - ang_o[3, 0]) >= -1e-4    # further positive
        check(agree < 1e-3 and beyond_lo and beyond_hi,
              f"{joint}: exact at the clip edge ({agree:.1e} deg), and past it goes to "
              f"{float(ang_n[0, 0]):+.1f}/{float(ang_n[3, 0]):+.1f} where run 008 "
              f"stopped at {float(ang_o[0, 0]):+.1f}/{float(ang_o[3, 0]):+.1f}")

    print("\n5. the four new actions start as run 008's physics")
    fr = mu_n[:, OLD_ACT:NEW_ACT]
    check(float(fr.abs().max()) < 1e-9,
          f"friction mean output is {float(fr.abs().max()):.2e} for every input "
          "-- a zero action is exactly the 1.1 contact run 008 ran")
    ls_n = p_n["action_model._continuous_distribution.log_sigma"][0]
    sig = float(ls_n[OLD_ACT:].exp().mean())
    check(0.05 < sig < 1.0, f"friction sigma {sig:.3f} -- exploring, not deterministic")

    print("\n6. exploration is preserved in ANGLE space, not action space")
    ls_o = p_o["action_model._continuous_distribution.log_sigma"][0]
    for rows, joint in ((HIP_X_ROWS, "hip_x"), (HIP_Y_ROWS, "hip_y"), (KNEE_X_ROWS, "knee_x")):
        ro = OLD_LIMITS[joint][1] - OLD_LIMITS[joint][0]
        rn = NEW_LIMITS[joint][1] - NEW_LIMITS[joint][0]
        # An action sigma maps to an angle sigma of (range / 2) * sigma.
        spread_o = float((ls_o[rows].exp() * ro * 0.5).mean())
        spread_n = float((ls_n[rows].exp() * rn * 0.5).mean())
        rel = abs(spread_o - spread_n) / max(spread_o, 1e-12)
        check(rel < 1e-4,
              f"{joint}: {spread_o:.4f} deg of exploration before, {spread_n:.4f} after "
              f"({rel * 100:.4f}% apart) -- the dial got longer, the wobble did not")

    print()
    if fails:
        print(f"{len(fails)} PROBLEM(S) -- do not launch from this checkpoint")
        return 1
    print("EXPANDED CHECKPOINT VERIFIED: behaviourally identical to run 008 at launch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
