"""Steady-state throughput, VRAM, and FLOP-model validation.

Answers the two questions that decide whether the project plan is realistic:

1. Does routing actually make it *faster*? Fewer FLOPs is easy; fewer FLOPs and less
   wall-clock is the claim. If the routed variants are not faster here, the
   gather/scatter overhead is eating the saving and the architecture needs rethinking
   before any training run is launched (risk R3).
2. Is the analytic FLOP model honest? Every efficiency number in the writeup comes
   from `amt/model/flops.py`; this checks it against torch's profiler.

    python scripts/benchmark.py
    python scripts/benchmark.py --variants b1_dense b6_amt_joint --steps 30
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from amt.model import AMT, FlopModel, variant  # noqa: E402
from amt.model.amt import stats_to_floats  # noqa: E402
from amt.precision import describe_device, select_precision  # noqa: E402


def measure(name, B, T, steps, warmup, device, compile_model=False):
    cfg = variant(name, block_size=T)
    model = AMT(cfg).to(device)
    model.train()
    if compile_model:
        model = torch.compile(model)

    amp_dtype, _, _ = select_precision(device, verbose=False)
    bank = model.make_bank(B, device, dtype=amp_dtype) if cfg.use_memory else None
    opt = model.configure_optimizers(0.1, 6e-4, "cuda" if device == "cuda" else "cpu",
                                     verbose=False)

    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    autocast = torch.autocast(device_type=device, dtype=amp_dtype,
                              enabled=device == "cuda")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def one_step():
        opt.zero_grad(set_to_none=True)
        with autocast:
            _, losses, stats, kv = model(x, targets=y, bank=bank)
        model.write_memory(bank, kv)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return stats

    for _ in range(warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        stats = one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    fm = FlopModel(cfg)
    return {
        "variant": name,
        "tokens_per_sec": steps * B * T / dt,
        "ms_per_step": dt / steps * 1000,
        "peak_vram_gb": peak,
        "layers_per_token": stats_to_floats(stats)["layers_per_token"],
        "analytic_layer_mflops": fm.breakdown().layers / 1e6,
        "analytic_total_mflops": fm.breakdown().total / 1e6,
        "params_m": (model._orig_mod if compile_model else model).num_params() / 1e6,
    }


def validate_flop_model(name, B, T, device):
    """Compare the analytic forward FLOPs against torch's profiler.

    The profiler counts only matmul/conv, which is exactly the convention flops.py
    uses, so the two should agree closely. A large gap means the analytic model is
    wrong and every reported efficiency number inherits the error.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    cfg = variant(name, block_size=T)
    model = AMT(cfg).to(device).eval()
    amp_dtype, _, _ = select_precision(device, verbose=False)
    bank = model.make_bank(B, device, dtype=amp_dtype) if cfg.use_memory else None
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    if bank is not None:  # warm the bank so retrieval is actually exercised
        with torch.no_grad():
            _, _, _, kv = model(x, bank=bank)
        model.write_memory(bank, kv)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x, bank=bank, return_logits=True)
    measured = counter.get_total_flops() / (B * T)
    analytic = FlopModel(cfg).breakdown().total
    return {"variant": name, "measured_mflops": measured / 1e6,
            "analytic_mflops": analytic / 1e6,
            "ratio": measured / analytic}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+",
                    default=["b1_dense", "b2_memory_only", "b3_depth_only",
                             "b6_amt_joint"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {describe_device()}")
    print(f"shape : B={a.batch_size} T={a.block_size}\n")

    rows = []
    for name in a.variants:
        try:
            rows.append(measure(name, a.batch_size, a.block_size, a.steps,
                                a.warmup, device, a.compile))
        except torch.cuda.OutOfMemoryError:
            print(f"{name}: OOM at B={a.batch_size} T={a.block_size}")
            torch.cuda.empty_cache()
        torch.cuda.empty_cache() if device == "cuda" else None

    print(f"{'variant':<16} {'tok/s':>9} {'ms/step':>9} {'VRAM GB':>8} "
          f"{'l/tok':>6} {'MFLOP/tok':>10} {'speedup':>8}")
    print("-" * 74)
    base = next((r["tokens_per_sec"] for r in rows if r["variant"] == "b1_dense"), None)
    for r in rows:
        sp = f"{r['tokens_per_sec'] / base:.2f}x" if base else "-"
        print(f"{r['variant']:<16} {r['tokens_per_sec']:>9,.0f} {r['ms_per_step']:>9.1f} "
              f"{r['peak_vram_gb']:>8.2f} {r['layers_per_token']:>6.2f} "
              f"{r['analytic_total_mflops']:>10.1f} {sp:>8}")

    if base and rows:
        hours = 1e9 / base / 3600
        print(f"\nat b1_dense throughput, 1B tokens takes {hours:.1f} h")

    print("\nFLOP model validation (forward, analytic vs profiler)")
    print(f"{'variant':<16} {'measured':>10} {'analytic':>10} {'ratio':>7}")
    print("-" * 46)
    for name in a.variants:
        v = validate_flop_model(name, a.batch_size, a.block_size, device)
        if v:
            flag = "" if 0.9 < v["ratio"] < 1.1 else "   <-- CHECK flops.py"
            print(f"{v['variant']:<16} {v['measured_mflops']:>10.1f} "
                  f"{v['analytic_mflops']:>10.1f} {v['ratio']:>7.3f}{flag}")
        torch.cuda.empty_cache() if device == "cuda" else None


if __name__ == "__main__":
    main()
