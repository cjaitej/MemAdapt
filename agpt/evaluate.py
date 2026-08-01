"""Measurement library: everything the scripts report, in one place.

`scripts/evaluate.py`, `scripts/compare.py`, `scripts/benchmark.py` and
`scripts/figures.py` all read from here, so a metric is defined once and every figure
and table in the writeup agrees by construction.

The measurement that decides the project
----------------------------------------
`arm_metrics` reports quality and cost side by side, and `throughput` reports what the
hardware actually does about it. Those are different numbers and the gap between them
is the honest finding: a token that skips six layers stops paying for their arithmetic
but still costs a kernel launch, a gather and a scatter, and at d=384 on a laptop GPU
the model is launch-latency bound rather than compute bound. Quote the FLOP saving and
the wall-clock saving together; never let one stand in for the other.
"""

import math

import torch

from .model import AdaptiveGPT, FlopModel, active_fractions
from .model.adaptive_gpt import strip_compile_prefix
from .model.config import AdaptiveGPTConfig


def load_checkpoint(path, device="cpu", **overrides):
    """Rebuild the model a checkpoint was saved from, with optional config changes.

    `overrides` is how the matched baselines are built: load the trained backbone,
    swap `exit_mode` to "fixed" or "random", and you have an arm that differs from the
    adaptive one in the routing rule and in nothing else -- same weights, same
    training, same seed. Routers are allowed to be missing or surplus for exactly this
    reason; nothing else is.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    if not isinstance(cfg, AdaptiveGPTConfig):
        cfg = AdaptiveGPTConfig(**cfg)
    if overrides:
        cfg = AdaptiveGPTConfig(**{**cfg.__dict__, **overrides})

    model = AdaptiveGPT(cfg)
    missing, unexpected = model.load_state_dict(
        strip_compile_prefix(ck["model"]), strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith("routers.")]
    if bad:
        raise ValueError(f"{path}: checkpoint does not match the config outside the "
                         f"routers: {bad[:6]}")
    return model.to(device).eval(), cfg, ck


@torch.no_grad()
def arm_metrics(model, loader, device, steps=50, compact=True, amp_dtype=None):
    """Quality and cost for one arm, on one validation stream.

    Returns a dict with the loss, the perplexity, the depth distribution, the measured
    per-layer activity and the FLOP breakdown implied by it.

    `compact=True` runs the fast path. It computes the same function as the gated path
    (`tests/test_exit.py::test_compact_matches_dense`), so this is a free choice for
    quality and the right one for reporting, because it is also what `throughput`
    times.
    """
    cfg = model.config
    model.eval()
    loader.reset()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    forward = model.forward_compact if compact else model.forward

    total = 0.0
    hist = torch.zeros(cfg.n_layer + 1, dtype=torch.float64)
    depth_sum, n_tok = 0.0, 0
    frac_sum = [0.0] * cfg.n_layer

    for _ in range(steps):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type,
                            dtype=amp_dtype or torch.bfloat16,
                            enabled=device_type == "cuda"):
            _, losses, _ = forward(x, targets=y)
        total += losses["lm"].item()

        depths = model.token_depths(x)
        hist += torch.bincount(depths.flatten().long(),
                               minlength=cfg.n_layer + 1).double().cpu()
        depth_sum += depths.sum().item()
        n_tok += depths.numel()
        for i, f in enumerate(active_fractions(depths, cfg.n_layer)):
            frac_sum[i] += f

    loss = total / steps
    fracs = [f / steps for f in frac_sum]
    fm = FlopModel(cfg)
    b, dense = fm.breakdown(fracs), fm.dense_breakdown()

    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20)),
        "avg_depth": depth_sum / max(n_tok, 1),
        "depth_hist": (hist / hist.sum()).tolist(),
        "active_fracs": fracs,
        "flops": b.as_dict(),
        "flops_dense": dense.as_dict(),
        "flops_frac_layers": b.layers / dense.layers,
        "flops_frac_total": b.total / dense.total,
        "exit_mode": cfg.exit_mode,
    }


@torch.no_grad()
def throughput(model, batch_size, block_size, device, compact=True, steps=12,
               warmup=3, amp_dtype=torch.bfloat16):
    """Tokens/second, per-batch latency and peak VRAM for one configuration.

    Warms up first: the first calls allocate workspaces, pick cuDNN algorithms and --
    under torch.compile -- trigger the compile itself, all of which would otherwise be
    charged to the model being measured.
    """
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    x = torch.randint(0, model.config.vocab_size, (batch_size, block_size),
                      device=device)
    forward = model.forward_compact if compact else model.forward

    def run():
        with torch.autocast(device_type=device_type, dtype=amp_dtype,
                            enabled=device_type == "cuda"):
            forward(x)

    for _ in range(warmup):
        run()
    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    import time
    t0 = time.time()
    for _ in range(steps):
        run()
    if device_type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / steps

    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device_type == "cuda" else float("nan"))
    return {"tokens_per_sec": batch_size * block_size / dt,
            "latency_ms": dt * 1000, "peak_mb": peak,
            "batch_size": batch_size, "block_size": block_size}


@torch.no_grad()
def generation_latency(model, prompt, n_tokens, device, compact=True):
    """Wall-clock seconds to generate `n_tokens`, and the mean depth spent on them.

    Generation is the setting where early exit should look best -- one token at a
    time, so the compacted batch is a single row and there is no ragged-batch penalty
    at all. It is also the setting with the most overhead per unit of arithmetic,
    which is the counter-argument. Measured rather than argued.
    """
    import time
    gen = torch.Generator(device=device).manual_seed(0)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out, depths = model.generate(prompt, n_tokens, generator=gen, compact=compact,
                                 return_depth=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    mean_depth = (torch.stack([d.float() for d in depths]).mean().item()
                  if depths else float("nan"))
    return {"seconds": dt, "tokens": n_tokens, "tokens_per_sec": n_tokens / dt,
            "mean_depth": mean_depth, "output": out}
