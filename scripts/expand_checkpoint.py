"""Expand a run 008 CrawlerParkour checkpoint onto run 009's rig.

Run 009 changes the shape of the policy in three ways at once:

  * the vector observation goes 168 -> 170 (two dead track-relative slots deleted,
    four previous-action slots added for the friction commands);
  * the action space goes 20 -> 24 (four per-foot friction commands appended);
  * every leg joint's angular limits widen, so THE SAME ACTION NUMBER NOW MEANS A
    DIFFERENT ANGLE.

Any one of those makes `--initialize-from` refuse the checkpoint, and the third
would be worse if it did not: the weights would load and the policy would drive
its legs to angles it never chose. Run 008 spent 36 hours and $46 reaching 0.603
m/s at terrain level 9; starting run 009 from noise throws that away and spends
most of a 24 h booking re-learning to walk instead of testing the four things run
009 is actually about.

So the tensors are transformed rather than discarded, and the transform is chosen
so that THE POLICY IS BEHAVIOURALLY IDENTICAL AT LAUNCH -- not approximately, not
after a settling period. Three facts make that exactly true rather than merely
plausible:

  1. The two deleted observations are CONSTANT -- run 006 retired them and has
     written a literal 0.0 into both on every decision since. They are NOT
     constant zero after normalisation, which is the trap: the running mean for
     those slots was inherited through the run 005 -> 006 -> 007 -> 008 warm-start
     chain from when they carried real values, and a mean only decays by
     (x - mean)/total_steps per sample, so after 184M samples it is still 0.0302
     and a zero input normalises to -0.3253. So each column contributes a fixed
     VECTOR to every layer it feeds, not nothing. That contribution is folded into
     the receiving layer's bias, which makes the deletion exact instead of
     approximately harmless.

  2. The four added observation columns and the four added action rows are
     initialised to zero weight, so they contribute exactly 0 as well, and a zero
     friction action is -- by the arithmetic in CrawlerParkourAgent -- run 008's
     contact coefficient of 1.1 exactly. The new actuator starts by reproducing
     the physics it replaces.

  3. The joint limits are absorbed into an AFFINE REMAP of the action head. If a'
     = s*a + o is chosen so that lerp(new_limits, a') == lerp(old_limits, a), then
     rescaling the head makes the new network emit the old network's joint ANGLES
     for every input. log_sigma shifts by log(s), which keeps the exploration
     distribution the same in ANGLE space too -- the policy does not start flailing
     merely because its dial got longer.

     The head does not emit `a` directly. Both the training path
     (networks.py: to_action_tuple(clip=True)) and the ONNX export path
     (action_model.py) send the environment clamp(x, -3, 3) / 3, where x is the
     sampled pre-action. So the bias offset carries a FACTOR OF THREE -- x' =
     s*x + 3*o is what produces a' = s*a + o downstream of that divide, and
     writing `o` instead would land every joint a third of the way to the wrong
     angle while looking perfectly reasonable in the tensor.

     Equivalence is exact wherever the old head did not saturate, |x| <= 3, which
     is where a converged policy spends nearly all of its time. Where it DID
     saturate, the new angle continues past the old limit in the same direction
     the old policy was already pushing -- run 008 pinned against a mechanical
     stop now gets the travel it was asking for, which is the point of widening
     the joint rather than a defect in the remap.

What is NOT preserved: the non-foot colliders are slipperier (0.24 contact against
run 008's 1.1), which is a real physical change and the point of the exercise. A
walking gait rarely rests a thigh on the ground, so the launch behaviour should be
close; if it is not, that IS the finding.

Adam's moments are dropped rather than remapped. They are a running estimate of a
gradient field that has just changed shape, and rebuilding them costs a few
thousand steps against getting a subtle rescaling wrong for the whole run.

    python scripts/expand_checkpoint.py \
        --src results/cloud_staging/CrawlerParkour_008/CrawlerParkour/checkpoint.pt \
        --dst results/cloud_staging/CrawlerParkour_008_r009/CrawlerParkour/checkpoint.pt

Then launch with -InitializeFrom CrawlerParkour_008_r009.

THE LAUNCH CHECK, which is the one that matters and costs two minutes: run 009's
first summary window should report DistanceRate near run 008's final, not near
zero. If the surgery is wrong the policy is scrambled, and that is visible in the
first window rather than at the end of the booking.
"""
import argparse
import math
import os
import sys

import torch

# ---------------------------------------------------------------- the layout
#
# Written as blocks rather than indices so the mapping is derived from the same
# description CollectObservations follows, and so a mistake shows up as a total
# that does not equal 168 or 170 instead of as a silently shifted column.
OLD_BLOCKS = [
    ("orientation", 6),   # up, forward in the yaw frame
    ("velocity", 6),      # linear, angular
    ("height", 1),
    ("contacts", 4),
    ("track", 5),         # 3 lateral + 2 DEAD (constant zero since run 006)
    ("grid", 126),        # 9 x 7 height field, two rays each
    ("prev", 20),
]
NEW_BLOCKS = [
    ("orientation", 6),
    ("velocity", 6),
    ("height", 1),
    ("contacts", 4),
    ("track", 3),         # the two dead slots deleted
    ("grid", 126),
    ("prev", 24),         # + 4 friction commands
]

OLD_ACT, NEW_ACT = 20, 24
ATTENTION_EMBEDDING = 128   # ObservationEncoder.ATTENTION_EMBEDDING_SIZE

# The environment receives clamp(x, -3, 3) / this, where x is the head's sampled
# output -- applied identically on the training path (AgentAction.to_action_tuple,
# clip=True) and in the exported ONNX (ActionModel.forward). Every statement about
# what an "action" is has to go through it.
ACTION_CLIP_SCALE = 3.0

# Joint limits, (low, high) degrees, before and after. Must match
# CrawlerParkourBuilder's Hip*/Knee* constants, and the action indices must match
# the order OnActionReceived reads them in: (x, y) per upper leg, then x per lower.
HIP_X_ROWS = [0, 2, 4, 6]
HIP_Y_ROWS = [1, 3, 5, 7]
KNEE_X_ROWS = [8, 9, 10, 11]
STRENGTH_ROWS = list(range(12, 20))

OLD_LIMITS = {"hip_x": (-60.0, 0.0), "hip_y": (-20.0, 20.0), "knee_x": (0.0, 150.0)}
NEW_LIMITS = {"hip_x": (-90.0, 90.0), "hip_y": (-45.0, 45.0), "knee_x": (-150.0, 150.0)}


def remap(old, new):
    """(scale, offset) so that lerp(new, s*a + o) == lerp(old, a) for every a.

    BodyPart.SetJointTargetRotation computes angle = lerp(lo, hi, (a + 1) / 2), so
    with Ro = hi_o - lo_o and Rn = hi_n - lo_n,

        a' = (Ro/Rn) * a + [2*(lo_o - lo_n)/Rn + Ro/Rn - 1]

    which is linear in a, which is why the whole change can be pushed into the
    output layer instead of being learned.
    """
    lo_o, hi_o = old
    lo_n, hi_n = new
    ro, rn = hi_o - lo_o, hi_n - lo_n
    if rn <= 0:
        raise ValueError(f"degenerate new range {new}")
    scale = ro / rn
    offset = 2.0 * (lo_o - lo_n) / rn + scale - 1.0
    return scale, offset


def build_column_map():
    """new_column -> old_column, or -1 for a column that did not exist."""
    old_off, new_off = {}, {}
    at = 0
    for name, n in OLD_BLOCKS:
        old_off[name] = (at, n)
        at += n
    old_total = at
    at = 0
    for name, n in NEW_BLOCKS:
        new_off[name] = (at, n)
        at += n
    new_total = at

    if old_total != 168 or new_total != 170:
        raise SystemExit(
            f"block table does not describe the runs it claims to: "
            f"old sums to {old_total} (expected 168), new to {new_total} (expected 170)")

    mapping = [-1] * new_total
    for name, (n_start, n_len) in new_off.items():
        o_start, o_len = old_off[name]
        # Both shrinking blocks keep their LEADING entries: `track` drops the two
        # trailing dead slots, `prev` appends four new actions after the old ones.
        keep = min(n_len, o_len)
        for i in range(keep):
            mapping[n_start + i] = o_start + i
    return mapping, old_total, new_total


def remap_columns(tensor, mapping, old_width, offset=0, total_width=None):
    """Rebuild a weight matrix's input columns under `mapping`.

    `offset`/`total_width` handle the body encoder, whose input is the vector
    observation concatenated with the 128-wide attention embedding: only the first
    `old_width` columns are remapped and the rest are carried across unchanged.
    """
    if total_width is None:
        total_width = old_width
    if tensor.shape[1] != total_width:
        raise ValueError(f"expected {total_width} columns, found {tensor.shape[1]}")
    tail = total_width - old_width - offset
    out = torch.zeros(tensor.shape[0], len(mapping) + offset + tail, dtype=tensor.dtype)
    if offset:
        out[:, :offset] = tensor[:, :offset]
    for new_i, old_i in enumerate(mapping):
        if old_i >= 0:
            out[:, offset + new_i] = tensor[:, offset + old_i]
    if tail:
        out[:, offset + len(mapping):] = tensor[:, offset + old_width:]
    return out


def remap_vector(tensor, mapping, fill):
    out = torch.full((len(mapping),), float(fill), dtype=tensor.dtype)
    for new_i, old_i in enumerate(mapping):
        if old_i >= 0:
            out[new_i] = tensor[old_i]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--friction-sigma", type=float, default=0.5,
                    help="Standard deviation the four new dimensions start exploring "
                         "at, IN ACTION UNITS -- the [-1, 1] the environment sees, not "
                         "the head's own scale. The default spreads the commanded "
                         "contact over roughly 0.55-1.65 against run 008's flat 1.1: "
                         "wide enough to discover that letting go helps, narrow enough "
                         "that the feet are not chattering from step 1. Inherited "
                         "log_sigma would be near-deterministic, which is the one thing "
                         "a brand new actuator must not be.")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    mapping, old_obs, new_obs = build_column_map()
    added = [i for i, o in enumerate(mapping) if o < 0]
    dropped = sorted(set(range(old_obs)) - {o for o in mapping if o >= 0})
    print(f"observations {old_obs} -> {new_obs}: dropped old columns {dropped}, "
          f"new columns {added} start at zero weight")

    body_in_old = old_obs + ATTENTION_EMBEDDING
    body_in_new = new_obs + ATTENTION_EMBEDDING

    # ------------------------------------------------------------ audit first
    #
    # Every tensor whose shape depends on a width this script changes must be
    # handled. Rather than trusting a list written by reading the state dict once,
    # collect them by shape and fail if any is left over -- a tensor that silently
    # keeps its old width is a policy that loads and is wrong.
    watched = {old_obs, body_in_old, OLD_ACT}
    handled, touched = set(), 0

    def key_of(module, name):
        return f"{module}::{name}"

    dropped_cols = sorted(set(range(old_obs)) - {o for o in mapping if o >= 0})

    for module in ("Policy", "Optimizer:critic"):
        sd = ck.get(module)
        if sd is None:
            continue

        # What the retired columns actually feed the network, computed from THIS
        # module's own normalizer and before that normalizer is rewritten. Deleting
        # a constant input is only free if its contribution is carried across.
        norm = next((k[: -len("running_mean")] for k in sd
                     if k.endswith("normalizer.running_mean")), None)
        z_dropped = {}
        if norm is not None:
            mean = sd[norm + "running_mean"]
            var = sd[norm + "running_variance"]
            steps = sd[norm + "normalization_steps"].float()
            zeros = torch.zeros(old_obs, dtype=mean.dtype)
            z = torch.clamp((zeros - mean) / torch.sqrt(var / steps), -5.0, 5.0)
            z_dropped = {c: float(z[c]) for c in dropped_cols}
            print(f"{module}: retired columns normalise to "
                  + ", ".join(f"[{c}] {v:+.4f}" for c, v in z_dropped.items())
                  + " -- folded into the receiving biases")

        for name in list(sd):
            t = sd[name]
            if not torch.is_tensor(t):
                continue
            shape = tuple(t.shape)

            if name.endswith("normalizer.running_mean") and shape == (old_obs,):
                sd[name] = remap_vector(t, mapping, 0.0)
                handled.add(key_of(module, name)); touched += 1
            elif name.endswith("normalizer.running_variance") and shape == (old_obs,):
                # A fresh Normalizer starts at variance 1 with 1 step, and normalises
                # by sqrt(running_variance / normalization_steps). By now the step
                # count is ~184M, so writing 1.0 into a new slot would divide it by
                # 7.4e-5 and hand the encoder a saturated +/-5 clamp on its first
                # observation. The new slots are previous ACTIONS, which is exactly
                # what the twenty slots beside them are, so they inherit the mean of
                # those -- calibrated by construction rather than by a guess.
                prev_start = sum(n for _, n in OLD_BLOCKS[:-1])
                inherited = float(t[prev_start:old_obs].mean())
                sd[name] = remap_vector(t, mapping, inherited)
                handled.add(key_of(module, name)); touched += 1
            elif shape == (128, old_obs) or (len(shape) == 2 and shape[1] == body_in_old):
                total = old_obs if shape[1] == old_obs else body_in_old
                bias_key = name[: -len("weight")] + "bias"
                if bias_key in sd and z_dropped:
                    delta = sum(t[:, c] * v for c, v in z_dropped.items())
                    sd[bias_key] = sd[bias_key] + delta
                    handled.add(key_of(module, bias_key))
                sd[name] = remap_columns(t, mapping, old_obs, total_width=total)
                want = new_obs if total == old_obs else body_in_new
                if sd[name].shape[1] != want:
                    raise SystemExit(f"{name}: came out {sd[name].shape[1]} wide, "
                                     f"expected {want}")
                handled.add(key_of(module, name)); touched += 1

    # ------------------------------------------------------- the action head
    policy = ck["Policy"]
    mu_w = policy["action_model._continuous_distribution.mu.weight"]
    mu_b = policy["action_model._continuous_distribution.mu.bias"]
    log_sigma = policy["action_model._continuous_distribution.log_sigma"]
    if mu_w.shape[0] != OLD_ACT or log_sigma.shape != (1, OLD_ACT):
        raise SystemExit(f"action head is {tuple(mu_w.shape)} / {tuple(log_sigma.shape)}, "
                         f"not the {OLD_ACT}-action head this script expects")

    new_w = torch.zeros(NEW_ACT, mu_w.shape[1], dtype=mu_w.dtype)
    new_b = torch.zeros(NEW_ACT, dtype=mu_b.dtype)
    new_ls = torch.zeros(1, NEW_ACT, dtype=log_sigma.dtype)
    # ACTION_CLIP_SCALE again, and it caught us once already. --friction-sigma is
    # quoted in action units because that is what every other number here is quoted
    # in, but log_sigma lives on the HEAD's scale and the environment sees x/3. A
    # sigma written straight in makes the feet explore a third as far as intended:
    # at the intended 0.5 a genuine release is 1.6 sigma out and gets sampled ~5% of
    # the time, at 0.5/3 it is 4.8 sigma out and is sampled never. That is the
    # difference between an actuator the policy can discover and one it cannot, and
    # it is invisible in every metric except FootRelease sitting at exactly zero.
    head_sigma = args.friction_sigma * ACTION_CLIP_SCALE
    new_ls[0, OLD_ACT:] = math.log(head_sigma)
    print(f"  friction sigma {args.friction_sigma} in action units "
          f"-> log_sigma {math.log(head_sigma):+.4f} on the head's scale")

    groups = [(HIP_X_ROWS, "hip_x"), (HIP_Y_ROWS, "hip_y"), (KNEE_X_ROWS, "knee_x")]
    for rows, joint in groups:
        s, o = remap(OLD_LIMITS[joint], NEW_LIMITS[joint])
        # ACTION_CLIP_SCALE, not 1. The head's output x reaches the environment as
        # clamp(x, -3, 3) / 3, so an offset of `o` in action space is an offset of
        # 3*o in head space. Omitting it is a silent third-of-the-range error.
        head_offset = o * ACTION_CLIP_SCALE
        print(f"  {joint:6s} {OLD_LIMITS[joint]} -> {NEW_LIMITS[joint]}: "
              f"action a' = {s:.6f} a {o:+.6f}, head x' = {s:.6f} x {head_offset:+.6f}"
              f"   rows {rows}")
        for r in rows:
            new_w[r] = mu_w[r] * s
            new_b[r] = mu_b[r] * s + head_offset
            new_ls[0, r] = log_sigma[0, r] + math.log(s)
    for r in STRENGTH_ROWS:      # untouched: strengths are not angles
        new_w[r] = mu_w[r]
        new_b[r] = mu_b[r]
        new_ls[0, r] = log_sigma[0, r]

    policy["action_model._continuous_distribution.mu.weight"] = new_w
    policy["action_model._continuous_distribution.mu.bias"] = new_b
    policy["action_model._continuous_distribution.log_sigma"] = new_ls
    for k in ("Policy::action_model._continuous_distribution.mu.weight",
              "Policy::action_model._continuous_distribution.mu.bias",
              "Policy::action_model._continuous_distribution.log_sigma"):
        handled.add(k)
    touched += 3

    # The exported ONNX reads its action count from these, not from the head, so a
    # stale value here produces a model file that declares 20 actions and a Unity
    # build that silently drives four of them with zeros.
    for k in ("continuous_act_size_vector", "act_size_vector_deprecated"):
        if k in policy:
            policy[k] = torch.tensor([float(NEW_ACT)])
            handled.add(key_of("Policy", k)); touched += 1

    # ------------------------------------------------------------ leftovers
    leftover = []
    for module in ("Policy", "Optimizer:critic"):
        for name, t in (ck.get(module) or {}).items():
            if not torch.is_tensor(t) or key_of(module, name) in handled:
                continue
            if any(d in watched for d in t.shape):
                leftover.append(f"{module}::{name} {tuple(t.shape)}")
    if leftover:
        raise SystemExit(
            "these tensors still carry a width this script changes and were not "
            "transformed:\n  " + "\n  ".join(leftover)
            + "\nThe checkpoint would load into some layers and not others.")

    # --------------------------------------------------------- Adam's moments
    opt = ck.get("Optimizer:value_optimizer")
    if isinstance(opt, dict) and "state" in opt:
        n = len(opt["state"])
        opt["state"] = {}
        print(f"dropped {n} Adam moment pairs; they estimate a gradient field that "
              f"has just changed shape")

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(ck, args.dst)
    print(f"\n{touched} tensors transformed -> {args.dst}")
    print(f"  {os.path.getsize(args.dst) / 1e6:.1f} MB, "
          f"global_step {int(ck['global_step']['_GlobalSteps__global_step'][0])} "
          f"(reset to 0 by --initialize-from)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
