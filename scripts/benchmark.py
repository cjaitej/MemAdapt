"""Throughput, latency, VRAM, and the FLOP-model validation.

    python scripts/benchmark.py --compile
    python scripts/benchmark.py --ckpt runs/s3_joint/final.pt --compile

With no checkpoint this benchmarks freshly initialised models, which is enough for the
speed numbers (throughput does not depend on the weights) and is how you check whether
routing is worth anything on this hardware before spending a training run on it.

Read this before quoting a speedup
----------------------------------
FLOPs saved and time saved are different numbers, and on a small model they differ a
lot. At d=384 on a laptop GPU the model is launch-latency bound, not compute bound:
skipping a layer removes its arithmetic but not its kernel launches, and the compact
path adds a gather, a scatter and a mask on top. Eager mode has repeatedly measured
*slower* than dense while using materially fewer FLOPs.

`torch.compile` is what closes the gap, by fusing the bookkeeping into the kernels it
sits between. On Windows it needs `pip install triton-windows` -- without it
`torch.compile` raises and this script falls back to eager, where every efficiency
claim in the project inverts. Never make an efficiency claim from an eager run.
"""

import os
import sys

# `python scripts/x.py` puts scripts/ on sys.path, not the repo root, so `agpt` is not
# importable without this. Two lines here beats requiring `pip install -e .` before the
# first run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import torch

from agpt.evaluate import load_checkpoint, throughput
from agpt.model import AdaptiveGPT, FlopModel, measured_flops, variant
from agpt.precision import describe_device


def build(args, exit_mode, device, **overrides):
    if args.ckpt:
        model, cfg, _ = load_checkpoint(args.ckpt, device, exit_mode=exit_mode,
                                        **overrides)
        return model, cfg
    cfg = variant(exit_mode, n_layer=args.n_layer, n_head=args.n_head,
                  n_embd=args.n_embd, block_size=args.block_size, **overrides)
    return AdaptiveGPT(cfg).to(device).eval(), cfg


def validate_flops(model, cfg, device, depth=None):
    """Analytic vs profiler, on the axis the profiler can actually see.

    `FlopCounterMode` has no formula registered for the fused attention kernels SDPA
    dispatches to, so it reports zero for them. The analytic model is therefore
    compared with its attention term switched off (`ctx_len=0`); comparing against the
    full number would credit the analytic model with an error the size of the whole
    attention cost. See agpt/model/flops.py.
    """
    x = torch.randint(0, cfg.vocab_size, (2, min(cfg.block_size, 256)), device=device)
    fm = FlopModel(cfg)
    if cfg.exit_mode == "dense":
        analytic = fm.dense_breakdown(ctx_len=0).total
    elif cfg.exit_mode == "fixed":
        analytic = fm.at_uniform_depth(cfg.fixed_exit_layer, ctx_len=0).total
    else:
        depths = model.token_depths(x)
        from agpt.model import active_fractions
        analytic = fm.breakdown(active_fractions(depths, cfg.n_layer), ctx_len=0).total
    profiled = measured_flops(model, x, compact=cfg.exit_mode != "dense")
    return analytic, profiled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--fixed-depth", type=int, default=6,
                    help="depth for the fixed-exit arm when no checkpoint is given")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device      : {describe_device()}")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    if not args.compile:
        print("WARNING     : eager mode. Routing is usually SLOWER than dense here; "
              "do not quote these numbers as an efficiency result.")

    arms = {
        "dense": {},
        "fixed": dict(fixed_exit_layer=args.fixed_depth),
        "adaptive": {},
    }

    results = {}
    for name, over in arms.items():
        model, cfg = build(args, name, device, **over)
        if args.compile:
            try:
                model = torch.compile(model)
            except Exception as e:                        # noqa: BLE001
                print(f"  torch.compile failed for {name}: {e}\n"
                      "  (on Windows: pip install triton-windows)")
        rows = []
        for B in args.batch_sizes:
            try:
                rows.append(throughput(model, B, cfg.block_size, device,
                                       compact=name != "dense", steps=args.steps))
            except torch.cuda.OutOfMemoryError:
                print(f"  {name} B={B}: OOM, skipped")
                torch.cuda.empty_cache()
        raw = getattr(model, "_orig_mod", model)
        analytic, profiled = validate_flops(raw, cfg, device)
        results[name] = {"throughput": rows,
                         "flops_analytic_matmul": analytic,
                         "flops_profiled": profiled}
        del model, raw
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'arm':10s} {'B':>4s} {'tok/s':>12s} {'ms/batch':>10s} "
          f"{'peak MB':>9s} {'vs dense':>9s}")
    base = {r["batch_size"]: r["tokens_per_sec"]
            for r in results["dense"]["throughput"]}
    for name, res in results.items():
        for r in res["throughput"]:
            ref = base.get(r["batch_size"])
            rel = f"{r['tokens_per_sec']/ref:.2f}x" if ref else "-"
            print(f"{name:10s} {r['batch_size']:4d} {r['tokens_per_sec']:12,.0f} "
                  f"{r['latency_ms']:10.1f} {r['peak_mb']:9.0f} {rel:>9s}")

    print(f"\n{'arm':10s} {'analytic (matmul)':>20s} {'profiler':>12s} {'ratio':>8s}")
    for name, res in results.items():
        a, p = res["flops_analytic_matmul"], res["flops_profiled"]
        print(f"{name:10s} {a/1e6:19.2f}M {p/1e6:11.2f}M {p/a:8.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": describe_device(), "compiled": args.compile,
                   "args": vars(args), "arms": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
