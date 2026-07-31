"""Training loop for the Adaptive Memory Transformer.

Derived from `train_gpt2.py`'s loop, with three additions: auxiliary router losses, a
capacity warmup schedule, and routing instrumentation.

    python -m amt.train --variant b6_amt_joint --data-dir data/fineweb_edu_docs
    python -m amt.train --variant b1_dense --max-steps 100 --synthetic

Instrumentation is not optional. When a routing run goes wrong the loss curve looks
almost normal -- what actually tells you is the routing rate, the gate value and the
aux-predictor agreement. Every one of those is logged every step from step 0.
"""

import argparse
import glob
import json
import math
import os
import time
from contextlib import nullcontext

import torch

from amt.data import DocSegmentLoader, make_random_shard
from amt.model import AMT, FlopModel, TopKTokenRouter, variant
from amt.model.amt import stats_to_floats, strip_compile_prefix
from amt.model.blocks import MemoryBlock
from amt.model.config import VARIANTS
from amt.model.retrofit import (describe, freeze_backbone, from_gpt2, gpt2_config)
from amt.precision import describe_device, make_scaler, select_precision


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def lr_at(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def capacity_at(step, target, max_steps, warmup_frac=0.1, anneal_frac=0.4, levels=4):
    """Anneal capacity from dense down to the target, in a few discrete steps.

    Routing on barely-contextualised embeddings is uninformed, so the stack runs dense
    for the first `warmup_frac` of training and then tightens.

    Quantised rather than continuous because k = ceil(capacity * T) determines tensor
    shapes: a smoothly-varying capacity would change k almost every step and make
    torch.compile recompile the whole graph each time, which costs far more than the
    smoother schedule is worth.
    """
    warm, anneal = warmup_frac * max_steps, anneal_frac * max_steps
    if step < warm:
        return 1.0
    if step >= anneal:
        return target
    frac = (step - warm) / max(anneal - warm, 1)
    level = math.floor(frac * levels) / levels
    return 1.0 + (target - 1.0) * level


def gate_floor_at(step, max_steps, warmup_frac=0.1, init=0.5):
    """Hold the memory gate open early, then release it (risk R2).

    The gate is initialised near-shut so the model does not lean on noise from an
    untrained memory. But a shut gate gets no gradient, so it can stay shut forever.
    Forcing a floor during warmup guarantees the retrieval path is exercised while
    it is still worth learning from.
    """
    warm = warmup_frac * max_steps
    if step >= warm:
        return 0.0
    return init * (1.0 - step / max(warm, 1))


# ---------------------------------------------------------------------------
# Model surgery for the schedules
# ---------------------------------------------------------------------------

def set_capacity(model, depth_capacity, mem_capacity):
    for m in model.modules():
        if isinstance(m, TopKTokenRouter):
            m.capacity = mem_capacity if m.name == "mem_router" else depth_capacity


def set_gate_floor(model, floor):
    for m in model.modules():
        if isinstance(m, MemoryBlock):
            m.gate_floor.fill_(floor)      # in place: see MemoryBlock.gate_floor


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, bank, device, device_type, steps=20, raw_model=None,
             amp_dtype=torch.bfloat16):
    """Validation loss plus routing telemetry.

    Calibrates the router thresholds first. Without it `agree/*` compares the causal
    predictor against an uncalibrated threshold and reports a number near the routing
    rate regardless of how good the predictor is -- i.e. it looks like a real metric
    and measures nothing.
    """
    model.eval()
    loader.reset()
    if bank is not None:
        bank.clear()

    if raw_model is not None:
        calib = [loader.next_batch()[0] for _ in range(3)]
        # The bank must be passed: without it the memory layer's read is skipped
        # entirely, so mem_router never runs, never gets calibrated, and its
        # agreement silently reports the routing rate instead of an accuracy.
        raw_model.calibrate_routers(calib, device=device, bank=bank)
        loader.reset()
        if bank is not None:
            bank.clear()
    total, agg = 0.0, {}
    for _ in range(steps):
        x, y, reset = loader.next_batch()
        x, y = x.to(device), y.to(device)
        if bank is not None:
            bank.clear(reset)                      # BEFORE forward -- see write_memory
        with torch.autocast(device_type=device_type, dtype=amp_dtype,
                            enabled=device_type == "cuda"):
            _, losses, stats, kv = model(x, targets=y, bank=bank)
        model.write_memory(bank, kv)
        total += losses["lm"].item()
        for k, v in stats_to_floats(stats).items():
            agg[k] = agg.get(k, 0.0) + v
    model.train()
    n = steps
    return total / n, {k: v / n for k, v in agg.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _toks(n):
    """Token counts at a readable scale -- synthetic runs are thousands, real ones
    hundreds of millions, and a fixed unit prints '0M' for one of them."""
    if n >= 1e9:
        return f"{n/1e9:,.2f}B"
    if n >= 1e6:
        return f"{n/1e6:,.1f}M"
    return f"{n/1e3:,.1f}K"


def prune_checkpoints(run_dir, keep):
    """Delete all but the `keep` most recent checkpoints in a run directory.

    Called only after a successful save, so a crash mid-write can never leave the
    directory empty: the file being replaced still exists until the new one is on
    disk. `keep=0` disables pruning entirely.

    Sorted by the step number in the filename rather than mtime -- a resumed run
    rewrites earlier steps' files, and mtime would then rank them newest.
    """
    if keep <= 0:
        return
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    for stale in ckpts[:-keep]:
        try:
            os.remove(stale)
        except OSError:
            pass          # a locked or already-removed file is not worth failing on


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="b6_amt_joint")
    ap.add_argument("--init-from", default="scratch",
                    help="'scratch', or a GPT-2 checkpoint (gpt2, gpt2-medium, "
                         "gpt2-large, gpt2-xl) to retrofit. The pretrained backbone "
                         "is frozen and only the routers and memory modules train, "
                         "so 'base vs ours' compares identical weights")
    ap.add_argument("--train-layernorms", action="store_true",
                    help="retrofit only: also train the LayerNorms. Reach for this "
                         "if the routers cannot learn on frozen features -- and "
                         "report that you did")
    ap.add_argument("--train-memory-block", action="store_true",
                    help="retrofit only: also train the memory layer's attn/MLP")
    ap.add_argument("--data-dir", default="data/fineweb_edu_docs")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on generated random shards; for plumbing checks only")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default=None)

    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--total-batch-tokens", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=19073)
    ap.add_argument("--epochs", type=float, default=None,
                    help="passes over the corpus, as an alternative to --max-steps; "
                         "overrides it when given. Resolved to a step count before "
                         "training starts rather than replacing it, because the LR "
                         "cosine, capacity anneal, gate floor and entropy anneal all "
                         "need a fixed horizon to schedule against")
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--max-lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--keep-ckpts", type=int, default=2,
                    help="how many recent checkpoints to retain; older ones are "
                         "deleted after a successful save. 0 keeps everything, which "
                         "at ~0.26 GB each will exhaust Kaggle's 20 GB working dir "
                         "over a full run. The final step's checkpoint is always kept")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"],
                    default="auto",
                    help="auto picks bf16 on Ampere+ and fp16+GradScaler on T4/P100")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint, or 'latest' to pick the newest in the "
                         "run dir; needed on Kaggle/Colab where sessions are capped")
    ap.add_argument("--capacity-warmup-frac", type=float, default=0.1,
                    help="fraction of steps run dense before routing tightens")
    ap.add_argument("--capacity-anneal-frac", type=float, default=0.4,
                    help="fraction of steps by which target capacity is reached")
    ap.add_argument("--capacity-levels", type=int, default=4,
                    help="anneal granularity; EACH level costs one torch.compile "
                         "recompile, so use 0 warmup/anneal for short debug runs")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"device      : {describe_device()}")
    if device_type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    run_name = args.run_name or f"{args.variant}_{int(time.time())}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # -- data -------------------------------------------------------------
    data_dir = args.data_dir
    if args.synthetic:
        data_dir = os.path.join(run_dir, "synthetic_data")
        make_random_shard(data_dir, "train", 0, n_docs=64,
                          doc_len=8 * args.block_size, vocab_size=50257, seed=0)
        make_random_shard(data_dir, "val", 0, n_docs=16,
                          doc_len=8 * args.block_size, vocab_size=50257, seed=1)
        print(f"synthetic data in {data_dir} -- loss will not drop below chance")

    B, T = args.batch_size, args.block_size
    train_loader = DocSegmentLoader(data_dir, B, T, split="train")
    val_loader = DocSegmentLoader(data_dir, B, T, split="val", shuffle=False)

    assert args.total_batch_tokens % (B * T) == 0, \
        "total-batch-tokens must be divisible by batch-size * block-size"
    grad_accum = args.total_batch_tokens // (B * T)

    # Resolve --epochs into a step count here, before anything schedules against it.
    # Written back onto args so config.json and the checkpoints record the number
    # actually used -- a resume reads max_steps, not epochs, and the two must agree
    # or the LR and capacity schedules would shift underneath the resumed run.
    epoch_tokens = train_loader.tokens_per_epoch()
    if args.epochs is not None:
        args.max_steps = max(1, round(args.epochs * epoch_tokens
                                      / args.total_batch_tokens))
    planned = args.max_steps * args.total_batch_tokens
    print(f"corpus      : {_toks(epoch_tokens)} trainable tokens/epoch at T={T}")
    print(f"budget      : {args.max_steps:,} steps = {_toks(planned)} tokens "
          f"= {planned/epoch_tokens:.2f} epochs"
          + (f"  (from --epochs {args.epochs})" if args.epochs is not None else ""))

    # -- model ------------------------------------------------------------
    # Two ways in, one variant table. `--init-from gpt2` applies the same routing
    # flags to a pretrained backbone instead of a fresh one, so every baseline in
    # RESEARCH_PLAN.md §7.1 has a retrofit twin under the same name.
    if args.init_from == "scratch":
        cfg = variant(args.variant, block_size=T)
        model = AMT(cfg).to(device)
    else:
        if args.variant not in VARIANTS:
            raise KeyError(f"unknown variant {args.variant!r}; "
                           f"known: {sorted(VARIANTS)}")
        cfg = gpt2_config(args.init_from, block_size=T, **VARIANTS[args.variant])
        model, report = from_gpt2(args.init_from, cfg, device=device)
        stats = freeze_backbone(model, train_layernorms=args.train_layernorms,
                                train_memory_block=args.train_memory_block)
        print(describe(report, stats))
        if stats["trainable"] == 0:
            raise SystemExit(
                f"--variant {args.variant} adds no modules to {args.init_from}, so a "
                "retrofit run would train nothing. That arm is the frozen base model: "
                "evaluate it with scripts/infer.py instead of training it.")
    fm = FlopModel(cfg)

    print(f"\nvariant     : {args.variant}")
    print(f"params      : {model.num_params():,} non-embedding")
    print(f"layout      : trunk={cfg.n_trunk} mem_layer={cfg.mem_layer} "
          f"adaptive={cfg.adaptive_layers} tail={cfg.n_dense_tail}")
    print(f"grad accum  : {grad_accum} micro-steps of {B}x{T} = {args.total_batch_tokens} tokens/step")
    print(fm.summary())
    print()

    raw_model = model                  # buffer writes must target the real module
    if args.compile:
        model = torch.compile(model)

    amp_dtype, use_scaler, _ = select_precision(
        device_type, None if args.precision == "auto" else args.precision)
    scaler = make_scaler(use_scaler)

    # The bank stores raw keys/values, so it must match the autocast dtype or every
    # read pays a cast and fp16 runs silently store fp16 into a bf16 buffer.
    bank_dtype = amp_dtype if amp_dtype != torch.float32 else torch.float32
    bank = model.make_bank(B, device, dtype=bank_dtype) if cfg.use_memory else None
    optimizer = model.configure_optimizers(args.weight_decay, args.max_lr, device_type)
    autocast = (torch.autocast(device_type=device_type, dtype=amp_dtype)
                if device_type == "cuda" else nullcontext())

    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == "latest":
            found = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_"))
            if not found:
                raise FileNotFoundError(f"--resume latest: no ckpt_*.pt in {run_dir}")
            ckpt_path = os.path.join(run_dir, found[-1])
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(strip_compile_prefix(ck["model"]))
        optimizer.load_state_dict(ck["optimizer"])
        train_loader.load_state_dict(ck["loader"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"] + 1
        # Carry the best-so-far across the resume. Restarting it at infinity would
        # let the first eval after a resume overwrite best.pt with a worse model.
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {ckpt_path} at step {start_step}"
              + (f" (best val {best_val:.4f})" if best_val < float("inf") else ""))

    log_path = os.path.join(run_dir, "log.jsonl")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "config": cfg.to_dict(),
                   "flops": fm.breakdown().as_dict()}, f, indent=2)

    # -- loop -------------------------------------------------------------
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last = step == args.max_steps - 1

        dc = capacity_at(step, cfg.depth_capacity, args.max_steps,
                         args.capacity_warmup_frac, args.capacity_anneal_frac,
                         args.capacity_levels)
        mc = capacity_at(step, cfg.mem_capacity, args.max_steps,
                         args.capacity_warmup_frac, args.capacity_anneal_frac,
                         args.capacity_levels)
        set_capacity(raw_model, dc, mc)
        set_gate_floor(raw_model, gate_floor_at(step, args.max_steps,
                                                args.capacity_warmup_frac))
        ent_w = cfg.lambda_entropy * max(0.0, 1.0 - step / (0.4 * args.max_steps))
        raw_model.set_entropy_weight(ent_w)

        if step % args.eval_every == 0 or last:
            val_loss, val_stats = evaluate(model, val_loader, bank, device,
                                           device_type, args.eval_steps,
                                           raw_model=raw_model, amp_dtype=amp_dtype)
            print(f"  [eval] step {step} val_loss {val_loss:.4f} "
                  f"ppl {math.exp(min(val_loss, 20)):.2f} "
                  f"layers/tok {val_stats.get('layers_per_token', 0):.2f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "split": "val",
                                    "loss": val_loss, **val_stats}) + "\n")

            if val_loss < best_val:
                best_val = val_loss
                # Named best.pt, NOT ckpt_best.pt: `--resume latest` picks the
                # lexicographically last ckpt_*.pt, and "best" sorts after every
                # zero-padded step, so a ckpt_ name would silently rewind training
                # to the best checkpoint instead of the newest one. It would also
                # occupy a slot in prune_checkpoints' retention window.
                #
                # No optimizer or loader state either: this file is for evaluation
                # and inference, and carrying them would triple its size for
                # something resume already gets from the step checkpoints.
                torch.save({"model": raw_model.state_dict(), "config": cfg,
                            "step": step, "val_loss": val_loss,
                            "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
                print(f"  [best] val_loss {val_loss:.4f} -> best.pt")

        optimizer.zero_grad(set_to_none=True)
        acc = {"lm": 0.0, "aux": 0.0, "entropy": 0.0}
        last_stats = {}
        for micro in range(grad_accum):
            x, y, reset = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Clear BEFORE the forward: a flagged stream begins a new document with
            # THIS batch, so clearing afterwards would let it retrieve from the
            # previous one. See AMT.write_memory.
            if bank is not None:
                bank.clear(reset)
            with autocast:
                _, losses, stats, kv = model(x, targets=y, bank=bank)
            model.write_memory(bank, kv)
            scaler.scale(losses["total"] / grad_accum).backward()
            for k in acc:
                acc[k] += losses[k].item() / grad_accum
            last_stats = stats  # tensors; converted once below, not per micro-step

        # Gradients must be unscaled before clipping, or the clip threshold is
        # applied to scaled values and effectively does nothing.
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = lr_at(step, args.max_lr, args.max_lr * args.min_lr_frac,
                   args.warmup_steps, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr * (0.1 if g.get("is_router") else 1.0)
        scaler.step(optimizer)
        scaler.update()

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tps = args.total_batch_tokens / dt

        if step % args.log_every == 0:
            last_stats = stats_to_floats(last_stats)
            row = {"step": step, "split": "train", "loss": acc["lm"],
                   "aux": acc["aux"], "entropy": acc["entropy"], "lr": lr,
                   "grad_norm": float(norm), "depth_capacity": dc, "mem_capacity": mc,
                   "dt_ms": dt * 1000, "tokens_per_sec": tps,
                   # Fractional position in the corpus. train_loader.epoch counts
                   # completed wraps; the fraction within one comes from the token
                   # budget, since streams advance at different rates per document.
                   "epoch": (step + 1) * args.total_batch_tokens / epoch_tokens,
                   **last_stats}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if step % max(args.log_every, 10) == 0:
                print(f"step {step:5d} | ep {row['epoch']:5.2f} "
                      f"| loss {acc['lm']:.4f} | aux {acc['aux']:.3f} "
                      f"| lr {lr:.2e} | norm {float(norm):.2f} "
                      f"| cap {dc:.2f} | l/tok {last_stats.get('layers_per_token', 0):.2f} "
                      f"| {dt*1000:.0f}ms | {tps:,.0f} tok/s")

        if step > 0 and (step % args.ckpt_every == 0 or last):
            torch.save({
                # raw_model, not model: under --compile the latter is an
                # OptimizedModule whose state_dict prefixes every key with
                # `_orig_mod.`, which nothing downstream can load into a plain AMT.
                "model": raw_model.state_dict(), "config": cfg,
                "optimizer": optimizer.state_dict(),
                "loader": train_loader.state_dict(), "step": step, "args": vars(args),
                "scaler": scaler.state_dict() if use_scaler else None,
                "best_val": best_val,
            }, os.path.join(run_dir, f"ckpt_{step:06d}.pt"))
            print(f"  [ckpt] step {step}")
            prune_checkpoints(run_dir, args.keep_ckpts)

    print(f"\ndone -> {run_dir}")


if __name__ == "__main__":
    main()
