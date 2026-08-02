"""The four-arm comparison: dense, random skip, fixed exit, AdaptiveGPT.

    python scripts/compare.py --ckpt runs/s3_joint/final.pt --data-dir data/wikitext103

Writes `results/compare.json` and prints the table that goes in the writeup.

Why all four arms come out of ONE checkpoint
--------------------------------------------
The baselines are not separately trained models. They are the *same weights* evaluated
under a different routing rule: `exit_mode` is a config field, the routing decision is
the only thing it changes, and `evaluate.load_checkpoint` rebuilds the model with it
swapped. So the four rows differ in the routing rule and in nothing else -- not the
seed, not the token budget, not the data order.

That matters most for the two arms that exist to rule something out:

* **random** exits at random, at a per-layer rate tuned to reach the same average
  depth as the adaptive arm. If the adaptive arm does not beat it, the router has
  learned to hit a budget and nothing more.
* **fixed** exits every token at the same layer, again chosen to match the average
  depth. If the adaptive arm does not beat it, per-token adaptivity is buying nothing
  over simply using a shallower model -- which is the single most likely way for this
  project's claim to be wrong, and the reason this arm is not optional.

Matching the depth is what makes them controls. An unmatched baseline compares two
things at once and settles neither.

A caveat to state in the writeup rather than hide
-------------------------------------------------
The adaptive arm was fine-tuned (Stage 3) to work under its own routing; `fixed` and
`random` are evaluated on those same weights without a matching fine-tune, which
favours the adaptive arm. Pass `--finetuned-fixed` / `--finetuned-random` with
checkpoints from `agpt.train --variant fixed --stage joint` to close that gap, and say
which comparison the numbers came from.
"""

import os
import sys

# `python scripts/x.py` puts scripts/ on sys.path, not the repo root, so `agpt` is not
# importable without this. Two lines here beats requiring `pip install -e .` before the
# first run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math

import torch

from agpt.data import SegmentLoader
from agpt.evaluate import arm_metrics, load_checkpoint, throughput


def match_random_p(target_depth, cfg):
    """Per-layer continue probability whose expected depth matches the adaptive arm.

    Under an independent per-layer coin with probability p, a token entering the
    routable region survives k more layers with probability p^k, so the expected depth
    is n_min + sum_{k=1..R} p^k. Solved by bisection because that sum does not invert
    in closed form.
    """
    n_min, R = cfg.n_min_layers, cfg.n_layer - cfg.n_min_layers
    if target_depth <= n_min:
        return 0.0

    def depth_at(p):
        return n_min + sum(p ** k for k in range(1, R + 1))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if depth_at(mid) < target_depth:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 6)


def pareto_verdict(results):
    """Does the adaptive arm sit below the fixed-depth frontier at its own cost?

    This is the whole claim, and stating it correctly is fiddly enough to be worth
    doing in one place.

    A fixed depth is an integer, so no single fixed arm lands at the adaptive arm's
    exact cost. Comparing against the nearest one is wrong in both directions: against
    the cheaper neighbour the adaptive arm wins for free (it is simply spending more),
    and against the dearer neighbour it can lose while still being the better trade.
    Neither says anything about allocation.

    The right comparison is the *frontier*: the piecewise-linear curve through every
    uniform-depth model, evaluated at the adaptive arm's FLOP cost. Beating that means
    per-token allocation buys quality that no constant depth of the same cost can.
    """
    ad = results["adaptive"]
    x, y = ad["flops_frac_layers"], ad["ppl"]

    # The fixed-depth frontier, dense included -- it is just uniform depth n_layer.
    pts = sorted((results[n]["flops_frac_layers"], results[n]["ppl"], n)
                 for n in ("fixed", "fixed_hi", "dense") if n in results)

    ref, how = None, ""
    for (x0, y0, n0), (x1, y1, n1) in zip(pts, pts[1:]):
        if x0 - 1e-9 <= x <= x1 + 1e-9 and x1 > x0:
            ref = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            how = f"interpolating {n0}@{x0:.1%} and {n1}@{x1:.1%}"
            break
    if ref is None:                       # outside the bracket: fall back to nearest
        x0, y0, n0 = min(pts, key=lambda p: abs(p[0] - x))
        ref, how = y0, f"nearest fixed arm {n0}@{x0:.1%} (NOT bracketed -- weak)"

    beats_fixed = y < ref
    beats_random = y < results["random"]["ppl"]
    lines = [
        f"fixed-depth frontier at {x:.1%} layer FLOPs: {ref:.2f} ppl  ({how})",
        f"adaptive                                    : {y:.2f} ppl",
    ]
    if beats_fixed:
        lines.append(f"VERDICT   adaptive is {ref - y:.2f} ppl BELOW the fixed-depth "
                     "frontier at equal cost.")
        lines.append("          Per-token allocation buys something constant depth "
                     "cannot.")
    else:
        lines.append(f"VERDICT   adaptive is {y - ref:.2f} ppl ABOVE the fixed-depth "
                     "frontier at equal cost.")
        lines.append("          Per-token adaptivity is not paying here -- report "
                     "that, do not tune it away.")
    if not beats_random:
        lines.append("WARNING   adaptive does not beat random skipping at matched "
                     "depth: the router")
        lines.append("          has learned to hit a budget, not to allocate.")
    return {"frontier_ppl": ref, "adaptive_ppl": y, "margin": ref - y,
            "beats_fixed_frontier": bool(beats_fixed),
            "beats_random": bool(beats_random), "basis": how, "lines": lines}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the trained adaptive checkpoint")
    ap.add_argument("--data-dir", default="data/wikitext103")
    ap.add_argument("--out", default="results/compare.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--bench-steps", type=int, default=12)
    ap.add_argument("--finetuned-fixed", default=None,
                    help="checkpoint for a separately fine-tuned fixed-exit arm")
    ap.add_argument("--finetuned-random", default=None,
                    help="checkpoint for a separately fine-tuned random-skip arm")
    ap.add_argument("--no-bench", action="store_true",
                    help="skip the wall-clock measurements")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, _ = load_checkpoint(args.ckpt, device)
    if cfg.exit_mode != "adaptive":
        raise SystemExit(
            f"--ckpt is an {cfg.exit_mode!r} model. The baselines are matched to the "
            "adaptive arm's measured depth, so this needs the adaptive checkpoint.")

    B, T = args.batch_size, cfg.block_size
    loader = SegmentLoader(args.data_dir, B, T, split="val")

    print("evaluating adaptive ...")
    results = {"adaptive": arm_metrics(model, loader, device, args.eval_steps)}
    target = results["adaptive"]["avg_depth"]
    print(f"  avg depth {target:.2f} of {cfg.n_layer}")
    del model

    # A fixed depth is an integer and the adaptive arm's mean is not, so no single
    # fixed arm is iso-cost with it. Rounding to the nearest layer quietly compares
    # against a CHEAPER model and flatters the adaptive arm: at mean depth 10.26,
    # round() gives 10, which is 83.3% of dense layer FLOPs against the adaptive arm's
    # 87.6%. Beating a cheaper baseline on quality is not a Pareto claim.
    #
    # So both neighbours are evaluated and the adaptive arm has to sit below the line
    # joining them. `fixed_lo` is cheaper, `fixed_hi` dearer; if adaptive beats only
    # `fixed_lo`, it bought its quality with FLOPs rather than with allocation.
    lo = max(cfg.n_min_layers, min(cfg.n_layer, int(math.floor(target))))
    hi = max(cfg.n_min_layers, min(cfg.n_layer, int(math.ceil(target))))
    random_p = match_random_p(target, cfg)

    arms = {
        "dense": (args.ckpt, dict(exit_mode="dense")),
        "random": (args.finetuned_random or args.ckpt,
                   dict(exit_mode="random", random_continue_p=random_p)),
        "fixed": (args.finetuned_fixed or args.ckpt,
                  dict(exit_mode="fixed", fixed_exit_layer=lo)),
    }
    if hi != lo:
        arms["fixed_hi"] = (args.finetuned_fixed or args.ckpt,
                            dict(exit_mode="fixed", fixed_exit_layer=hi))
    print(f"  matched: fixed_exit_layer={lo}"
          + (f" and {hi} (bracketing depth {target:.2f})" if hi != lo else "")
          + f", random_continue_p={random_p}")

    for name, (path, overrides) in arms.items():
        print(f"evaluating {name} ...")
        arm, _, _ = load_checkpoint(path, device, **overrides)
        results[name] = arm_metrics(arm, loader, device, args.eval_steps)
        results[name]["overrides"] = overrides
        results[name]["ckpt"] = path
        del arm

    # Wall clock is a different question from FLOPs and often a different answer --
    # see agpt/evaluate.py. Each arm is rebuilt for its own timing run so no arm is
    # measured on a GPU still holding another one's allocations.
    if not args.no_bench:
        for name in list(results):
            path = results[name].get("ckpt", args.ckpt)
            arm, _, _ = load_checkpoint(path, device,
                                        **results[name].get("overrides", {}))
            results[name]["throughput"] = throughput(arm, B, T, device,
                                                     steps=args.bench_steps)
            del arm

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ckpt": args.ckpt, "config": cfg.to_dict(),
                   "target_depth": target, "arms": results}, f, indent=2)

    print(f"\n{'arm':10s} {'ppl':>8s} {'depth':>7s} {'FLOPs/tok':>11s} "
          f"{'layer':>7s} {'total':>7s} {'tok/s':>10s} {'speedup':>8s}")
    base_tps = results["dense"].get("throughput", {}).get("tokens_per_sec")
    for name in [n for n in ("dense", "random", "fixed", "fixed_hi", "adaptive")
                 if n in results]:
        r = results[name]
        tps = r.get("throughput", {}).get("tokens_per_sec", float("nan"))
        speed = tps / base_tps if base_tps else float("nan")
        print(f"{name:10s} {r['ppl']:8.2f} {r['avg_depth']:7.2f} "
              f"{r['flops']['total']/1e6:10.1f}M {r['flops_frac_layers']:7.1%} "
              f"{r['flops_frac_total']:7.1%} {tps:10,.0f} {speed:7.2f}x")

    print()
    verdict = pareto_verdict(results)
    for line in verdict["lines"]:
        print(line)
    results["_verdict"] = {k: v for k, v in verdict.items() if k != "lines"}

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
