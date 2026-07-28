"""Where the time inside KVMemoryBank.read actually goes.

`benchmark.py` measures the memory path as a black box -- b2 minus b1 -- and that
delta sits at ~15% of dense step time on both a T4 and an RTX 3050. It did not move
on the T4 when the kNN search switched from fp32 to fp16, which rules the search
matmul out as the bottleneck on that card and leaves the rest of `read()`
unaccounted for. Reasoning from tensor shapes produced two wrong answers about this;
this script profiles the kernels instead.

    python scripts/profile_memory.py --batch-size 32            # sweep + profile
    python scripts/profile_memory.py --batch-size 32 --no-sweep

The sweep is the actionable half: it varies the two knobs that scale the retrieved
neighbour tensors -- n_neighbors and the query-chunk size -- at both the b2 query
count (all tokens) and the b6 one (mem_capacity=0.25), so the cost model comes from
measurement rather than inference.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")

from amt.model import variant  # noqa: E402
from amt.model.memory import KVMemoryBank  # noqa: E402
from amt.precision import describe_device, select_precision  # noqa: E402


def build_bank(B, cfg, device, dtype, fill_frac=1.0):
    """A bank filled with random keys/values, so retrieval does real work."""
    bank = KVMemoryBank(B, cfg.n_head, cfg.head_dim, cfg.mem_size, device, dtype)
    n = max(1, int(cfg.mem_size * fill_frac))
    k = torch.randn(B, cfg.n_head, n, cfg.head_dim, device=device, dtype=dtype)
    v = torch.randn(B, cfg.n_head, n, cfg.head_dim, device=device, dtype=dtype)
    bank.write(k, v)
    return bank


def make_query(B, cfg, Tq, device, dtype, backward):
    # In the real model the query is produced by a Linear under autocast, so it is
    # already in the autocast dtype. Building it as fp32 here would make the gather
    # in read() upcast and silently profile a path the model never takes.
    return torch.randn(B, cfg.n_head, Tq, cfg.head_dim, device=device, dtype=dtype,
                       requires_grad=backward)


def time_read(bank, q, k, chunk, steps, warmup, backward, device):
    def once():
        y, _ = bank.read(q, n_neighbors=k, chunk_size=chunk)
        if backward:
            y.float().sum().backward()
            q.grad = None

    for _ in range(warmup):
        once()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        once()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def sweep(B, cfg, device, dtype, steps, warmup, backward):
    print(f"\nread() cost in ms  (bank M={cfg.mem_size}, H={cfg.n_head}, "
          f"D={cfg.head_dim}, dtype={str(dtype).replace('torch.', '')}, "
          f"backward={backward})")
    print(f"{'queries':>8} {'k':>4} {'chunk':>6} {'ms':>8} {'neighbour tensor':>18}")
    print("-" * 50)
    for Tq, label in ((cfg.block_size, "b2: all tokens"),
                      (max(1, int(cfg.block_size * cfg.mem_capacity)), "b6: routed")):
        q = make_query(B, cfg, Tq, device, dtype, backward)
        for k in (8, 16, 32):
            for chunk in (128, 512):
                ms = time_read(bank_for(B, cfg, device, dtype), q, k, chunk,
                               steps, warmup, backward, device)
                # The two (B, H, Tq, k, D) tensors gathered per chunk and held for
                # backward: the term that scales with k.
                mb = 2 * B * cfg.n_head * Tq * k * cfg.head_dim * dtype.itemsize / 1e6
                print(f"{Tq:>8} {k:>4} {chunk:>6} {ms:>8.2f} {mb:>15.0f} MB")
        print(f"{'':>8} {label}")


_BANK_CACHE = {}


def bank_for(B, cfg, device, dtype):
    """One bank per shape; rebuilding it per sweep cell would dominate the timing."""
    key = (B, cfg.mem_size, cfg.n_head, cfg.head_dim, dtype)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = build_bank(B, cfg, device, dtype)
    return _BANK_CACHE[key]


def profile_read(bank, q, k, chunk, steps, backward, device):
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if device == "cuda":
        acts.append(ProfilerActivity.CUDA)

    def once():
        y, _ = bank.read(q, n_neighbors=k, chunk_size=chunk)
        if backward:
            y.float().sum().backward()
            q.grad = None

    once()  # warm
    if device == "cuda":
        torch.cuda.synchronize()

    with profile(activities=acts) as prof:
        for _ in range(steps):
            once()
        if device == "cuda":
            torch.cuda.synchronize()

    # PyTorch renamed the CUDA columns to "device" in 2.x; try new name first.
    for key in ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total"):
        try:
            return prof.key_averages().table(sort_by=key, row_limit=18)
        except (KeyError, AssertionError, RuntimeError):
            continue
    return prof.key_averages().table(row_limit=18)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="b6_amt_joint")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--neighbors", type=int, default=None,
                    help="k for the profiler pass; defaults to the config's n_neighbors")
    ap.add_argument("--chunk", type=int, default=None,
                    help="query chunk for the profiler pass; defaults to the config's")
    ap.add_argument("--forward-only", action="store_true",
                    help="skip backward; the gathered tensors are held for backward, "
                         "so the two numbers bracket the real cost")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = variant(a.variant)
    dtype, _, label = select_precision(device, verbose=False)
    backward = not a.forward_only

    print(f"device: {describe_device()}")
    print(f"config: {a.variant}  B={a.batch_size}  M={cfg.mem_size}  "
          f"k={cfg.n_neighbors}  chunk={cfg.mem_query_chunk}  precision={label}")

    if not a.no_sweep:
        sweep(a.batch_size, cfg, device, dtype, a.steps, a.warmup, backward)

    if not a.no_profile:
        k = a.neighbors or cfg.n_neighbors
        chunk = a.chunk or cfg.mem_query_chunk
        Tq = cfg.block_size
        print(f"\nkernel breakdown  (Tq={Tq}, k={k}, chunk={chunk}, "
              f"backward={backward})")
        q = make_query(a.batch_size, cfg, Tq, device, dtype, backward)
        print(profile_read(bank_for(a.batch_size, cfg, device, dtype), q, k, chunk,
                           a.steps, backward, device))


if __name__ == "__main__":
    main()
