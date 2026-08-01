"""Training loop for AdaptiveGPT.

Three stages, run as three separate invocations so each one's checkpoint is a thing
you can inspect, evaluate and roll back to:

    # 1. the language model, dense. This checkpoint is ALSO the dense baseline.
    python -m agpt.train --stage dense --run-name s1_dense --max-steps 12000

    # 2. freeze it, teach the routers where the stack stops mattering
    python -m agpt.train --stage routers --init-from runs/s1_dense/best.pt \
        --run-name s2_routers --max-steps 1500

    # 3. unfreeze, let the model adapt to being interrupted
    python -m agpt.train --stage joint --init-from runs/s2_routers/best.pt \
        --run-name s3_joint --max-steps 3000

Why three stages and not one
----------------------------
Routing and language modelling learned simultaneously is a race between two things
that need each other's output. The router cannot tell an easy token from a hard one
until the representations mean something, and the representations cannot settle while
half of them are being cut off at a randomly chosen depth. Staged, each phase has a
stationary target: Stage 2 learns to route a model that is no longer moving, and Stage
3 adapts a model to routing decisions that are already roughly right.

Stage 1 is the dense baseline
-----------------------------
There is no separate control run. `--stage dense` builds the full adaptive model and
pins every gate open, so the weights it produces are exactly what a dense-only model
would have produced -- and are then used both as the baseline to beat and as the
initialisation for Stages 2 and 3. The comparison is therefore against the identical
backbone rather than against a separately-trained model that happened to see the same
tokens, which removes seed variance from the headline number entirely.

Instrumentation is not optional. When a routing run goes wrong the loss curve looks
almost normal -- what tells you is the depth, the exit fraction and the per-layer
continue probabilities. All of those are logged every step from step 0.
"""

import argparse
import glob
import json
import math
import os
import time
from contextlib import nullcontext

import torch

from agpt.data import SegmentLoader, make_random_shard
from agpt.model import AdaptiveGPT, FlopModel, variant
from agpt.model.adaptive_gpt import stats_to_floats, strip_compile_prefix
from agpt.model.config import VARIANTS
from agpt.model.retrofit import describe, freeze_backbone, from_gpt2, gpt2_config
from agpt.model.targets import exit_targets
from agpt.precision import describe_device, make_scaler, select_precision

STAGES = ("dense", "routers", "joint")


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


def depth_weight_at(step, target, max_steps, warmup_frac=0.2):
    """Ramp the depth penalty in rather than applying it from step 0.

    The penalty is the only term that actively wants tokens to exit, and at step 0 the
    routers have no idea which tokens should. Applied immediately it is answered the
    only way an untrained router can answer it -- by exiting everything at the earliest
    opportunity -- and a stack that has collapsed to minimum depth gives the BCE
    nothing to learn from, because no token ever reaches the later routers. Ramping it
    lets the convergence labels shape the routing first and the budget squeeze it
    afterwards.
    """
    warm = warmup_frac * max_steps
    if step >= warm:
        return target
    return target * (step / max(warm, 1))


# ---------------------------------------------------------------------------
# Stage setup
# ---------------------------------------------------------------------------

def apply_stage(model, stage, args):
    """Configure gating, loss weights and which parameters train.

    Returns a short description for the log.
    """
    raw = model
    if stage == "dense":
        raw.set_routing_enabled(False)
        raw.set_lambda_router(0.0)
        raw.set_lambda_depth(0.0)
        for p in raw.parameters():
            p.requires_grad = True
        return "gates pinned open; every parameter trains; router losses off"

    # Read the weights off the config, not off `args`: the CLI flags default to None
    # so that "unset" can be told apart from "set to the default", and
    # `config_overrides` has already folded any that were given into the config.
    raw.set_routing_enabled(True)
    raw.set_lambda_router(raw.config.lambda_router)
    raw.set_lambda_depth(0.0)          # ramped in by depth_weight_at

    if stage == "routers":
        n_router = 0
        for name, p in raw.named_parameters():
            train = name.startswith("routers.")
            p.requires_grad = train
            n_router += p.numel() if train else 0
        if n_router == 0:
            raise SystemExit(
                f"--stage routers on --variant {args.variant} trains nothing: only "
                "the adaptive arm has routers. The random and fixed baselines have no "
                "parameters to fit, so they go straight from --stage dense to "
                "--stage joint.")
        return f"backbone frozen; {n_router:,} router params train"

    for p in raw.parameters():
        p.requires_grad = True
    return "everything trains"


def load_init(model, path, device):
    """Load weights from a previous stage, tolerating a config change.

    A Stage 3 run may swap `exit_mode` (to build the matched `fixed` and `random`
    baselines off the same backbone), which changes whether `routers.*` exists. That
    is a legitimate load, not a corrupt checkpoint, so the router tensors are allowed
    to be missing in either direction -- but nothing else is.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    state = strip_compile_prefix(ck["model"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if not k.startswith("routers.")]
    if bad:
        raise ValueError(f"{path}: checkpoint does not match the model outside the "
                         f"routers: {bad[:6]}")
    note = ""
    if missing:
        note = f" ({len(missing)} router tensors left at init)"
    elif unexpected:
        note = f" ({len(unexpected)} router tensors in the checkpoint dropped)"
    print(f"init from   : {path} at step {ck.get('step', '?')}{note}")
    return ck


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, raw_model, loader, device, device_type, steps=20,
             amp_dtype=torch.bfloat16, labels=True):
    """Validation loss plus depth telemetry.

    No calibration step: an exit router reads one token's own state, so the decision
    it makes here is the decision it made in training. (The per-layer top-k design
    this replaced needed a calibrated causal predictor and a reported agreement number
    before any of these figures meant anything.)
    """
    model.eval()
    loader.reset()
    total, agg = 0.0, {}
    for _ in range(steps):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=amp_dtype,
                            enabled=device_type == "cuda"):
            lab = derive_labels(raw_model, x) if labels else None
            _, losses, stats = model(x, targets=y, exit_labels=lab)
        total += losses["lm"].item()
        for k, v in stats_to_floats(stats).items():
            agg[k] = agg.get(k, 0.0) + v
        agg["router_bce"] = agg.get("router_bce", 0.0) + float(losses["router"])
    model.train()
    return total / steps, {k: v / steps for k, v in agg.items()}


def derive_labels(raw_model, x):
    """Run the stack densely and turn what the later layers did into exit labels.

    Costs one extra forward pass per micro-step, roughly a third again on top of
    forward+backward. That is the price of supervised routing targets, and it buys the
    thing that makes Stage 2 converge in ~1500 steps instead of not converging: the
    router is told what the right answer was, rather than having to infer it from a
    language-modelling gradient that reaches it through a straight-through estimator.
    """
    cfg = raw_model.config
    if not cfg.router_layers:
        return None
    hiddens = raw_model.dense_hidden_states(x)
    labels, _ = exit_targets(hiddens, cfg,
                             ln_f=raw_model.transformer.ln_f,
                             lm_head=raw_model.lm_head)
    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _toks(n):
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
    for stale in sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))[:-keep]:
        try:
            os.remove(stale)
        except OSError:
            pass          # a locked or already-removed file is not worth failing on


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="adaptive", choices=sorted(VARIANTS))
    ap.add_argument("--stage", default="dense", choices=STAGES,
                    help="dense: train the LM with gates pinned open (this is also "
                         "the dense baseline). routers: freeze the backbone and fit "
                         "the exit routers. joint: unfreeze and adapt")
    ap.add_argument("--init-from", default=None,
                    help="a checkpoint from the previous stage")
    ap.add_argument("--from-gpt2", default=None,
                    help="retrofit a pretrained GPT-2 (gpt2, gpt2-medium, ...) "
                         "instead of training from scratch")
    ap.add_argument("--train-head", action="store_true",
                    help="retrofit only: also train ln_f and the tied lm_head. A "
                         "frozen head has never seen a mid-stack representation, so "
                         "you almost certainly want this -- see retrofit.py")
    ap.add_argument("--train-layernorms", action="store_true",
                    help="retrofit only: also train the per-block LayerNorms")

    ap.add_argument("--data-dir", default="data/wikitext103")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on random shards; for plumbing checks only")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default=None)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--total-batch-tokens", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--max-lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--router-lr-mult", type=float, default=1.0)

    # -- routing knobs, all mirrored onto the config ------------------------
    ap.add_argument("--n-min-layers", type=int, default=None)
    ap.add_argument("--target-type", choices=["delta", "kl"], default=None)
    ap.add_argument("--target-tau", type=float, default=None)
    ap.add_argument("--lambda-router", type=float, default=None)
    ap.add_argument("--lambda-depth", type=float, default=None,
                    help="weight on the expected-depth penalty. This is the knob "
                         "that trades perplexity for speed; sweep it to draw the "
                         "Pareto curve")
    ap.add_argument("--depth-warmup-frac", type=float, default=0.2)
    ap.add_argument("--fixed-exit-layer", type=int, default=None)
    ap.add_argument("--random-continue-p", type=float, default=None)
    ap.add_argument("--exited-as-keys", choices=["stale", "drop"], default=None)
    ap.add_argument("--dropout", type=float, default=None,
                    help="residual + attention dropout. 0.0 suits a single pass over "
                         "a huge corpus; WikiText-103 is small enough that training "
                         "is multi-epoch by construction, so use ~0.1")
    ap.add_argument("--soft-gate", action="store_true",
                    help="use the soft cumulative probability instead of the "
                         "straight-through hard gate. Trains more smoothly and "
                         "benchmarks a model you did not train")

    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--keep-ckpts", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"],
                    default="auto")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint, or 'latest' for the newest in the "
                         "run dir; needed on Kaggle/Colab where sessions are capped")
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def config_overrides(args):
    """CLI routing flags -> AdaptiveGPTConfig kwargs, omitting anything unset."""
    named = dict(dropout=args.dropout,
                 n_min_layers=args.n_min_layers, target_type=args.target_type,
                 target_tau=args.target_tau, lambda_router=args.lambda_router,
                 lambda_depth=args.lambda_depth,
                 fixed_exit_layer=args.fixed_exit_layer,
                 random_continue_p=args.random_continue_p,
                 exited_as_keys=args.exited_as_keys)
    out = {k: v for k, v in named.items() if v is not None}
    if args.soft_gate:
        out["straight_through"] = False
    return out


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"device      : {describe_device()}")
    if device_type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")

    run_name = args.run_name or f"{args.variant}_{args.stage}_{int(time.time())}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # -- data -------------------------------------------------------------
    data_dir = args.data_dir
    if args.synthetic:
        data_dir = os.path.join(run_dir, "synthetic_data")
        make_random_shard(data_dir, "train", 0, n_tokens=1 << 20, seed=0)
        make_random_shard(data_dir, "val", 0, n_tokens=1 << 18, seed=1)
        print(f"synthetic data in {data_dir} -- loss will not drop below chance")

    B, T = args.batch_size, args.block_size
    train_loader = SegmentLoader(data_dir, B, T, split="train")
    val_loader = SegmentLoader(data_dir, B, T, split="val")

    assert args.total_batch_tokens % (B * T) == 0, \
        "total-batch-tokens must be divisible by batch-size * block-size"
    grad_accum = args.total_batch_tokens // (B * T)

    epoch_tokens = train_loader.tokens_per_epoch()
    planned = args.max_steps * args.total_batch_tokens
    print(f"corpus      : {_toks(epoch_tokens)} tokens/epoch at T={T}")
    print(f"budget      : {args.max_steps:,} steps = {_toks(planned)} tokens "
          f"= {planned/max(epoch_tokens, 1):.2f} epochs")

    # -- model ------------------------------------------------------------
    overrides = config_overrides(args)
    if args.from_gpt2:
        cfg = gpt2_config(args.from_gpt2, block_size=T,
                          **{**VARIANTS[args.variant], **overrides})
        model, report = from_gpt2(args.from_gpt2, cfg, device=device)
        frozen = freeze_backbone(model, train_head=args.train_head,
                                 train_layernorms=args.train_layernorms)
        print(describe(report, frozen))
    else:
        cfg = variant(args.variant, block_size=T, **overrides)
        model = AdaptiveGPT(cfg).to(device)

    start_step, best_val = 0, float("inf")
    if args.init_from:
        load_init(model, args.init_from, device)

    stage_note = apply_stage(model, args.stage, args)
    if args.from_gpt2 and args.stage != "dense":
        # The retrofit's own freeze is the stricter of the two and must win: Stage 3
        # would otherwise unfreeze 124M pretrained parameters that the whole point of
        # a retrofit is to hold fixed.
        frozen = freeze_backbone(model, train_head=args.train_head,
                                 train_layernorms=args.train_layernorms)
        stage_note = (f"retrofit freeze re-applied: {frozen['trainable']:,} of "
                      f"{frozen['trainable'] + frozen['frozen']:,} train")

    fm = FlopModel(cfg)
    print(f"\nvariant     : {args.variant}   stage: {args.stage}")
    print(f"stage       : {stage_note}")
    print(f"params      : {model.num_params():,} non-embedding, "
          f"{model.router_params():,} in routers")
    print(f"layout      : always-on 0..{cfg.n_min_layers - 1}, "
          f"routers after {cfg.router_layers or 'none'}")
    print(f"grad accum  : {grad_accum} micro-steps of {B}x{T} "
          f"= {args.total_batch_tokens} tokens/step")
    print(fm.summary())
    print()

    raw_model = model                  # buffer writes must target the real module
    if args.compile:
        model = torch.compile(model)

    amp_dtype, use_scaler, _ = select_precision(
        device_type, None if args.precision == "auto" else args.precision)
    scaler = make_scaler(use_scaler)
    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.max_lr, device_type,
        router_lr_mult=args.router_lr_mult)
    autocast = (torch.autocast(device_type=device_type, dtype=amp_dtype)
                if device_type == "cuda" else nullcontext())

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
        # Carry the best-so-far across the resume. Restarting it at infinity would let
        # the first eval after a resume overwrite best.pt with a worse model.
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {ckpt_path} at step {start_step}"
              + (f" (best val {best_val:.4f})" if best_val < float("inf") else ""))

    want_labels = args.stage != "dense" and bool(cfg.router_layers)
    log_path = os.path.join(run_dir, "log.jsonl")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "config": cfg.to_dict(),
                   "flops": fm.dense_breakdown().as_dict()}, f, indent=2)

    # -- loop -------------------------------------------------------------
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last = step == args.max_steps - 1

        lam_depth = (0.0 if args.stage == "dense" else
                     depth_weight_at(step, cfg.lambda_depth, args.max_steps,
                                     args.depth_warmup_frac))
        raw_model.set_lambda_depth(lam_depth)

        if step % args.eval_every == 0 or last:
            val_loss, val_stats = evaluate(model, raw_model, val_loader, device,
                                           device_type, args.eval_steps,
                                           amp_dtype=amp_dtype, labels=want_labels)
            print(f"  [eval] step {step} val_loss {val_loss:.4f} "
                  f"ppl {math.exp(min(val_loss, 20)):.2f} "
                  f"depth {val_stats.get('depth', 0):.2f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "split": "val",
                                    "loss": val_loss, **val_stats}) + "\n")

            if val_loss < best_val:
                best_val = val_loss
                # Named best.pt, NOT ckpt_best.pt: `--resume latest` picks the
                # lexicographically last ckpt_*.pt, and "best" sorts after every
                # zero-padded step, so a ckpt_ name would silently rewind training to
                # the best checkpoint instead of the newest one.
                torch.save({"model": raw_model.state_dict(), "config": cfg,
                            "step": step, "val_loss": val_loss, "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
                print(f"  [best] val_loss {val_loss:.4f} -> best.pt")

        optimizer.zero_grad(set_to_none=True)
        acc = {"lm": 0.0, "router": 0.0, "depth": 0.0}
        last_stats = {}
        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast:
                labels = derive_labels(raw_model, x) if want_labels else None
                _, losses, stats = model(x, targets=y, exit_labels=labels)
            scaler.scale(losses["total"] / grad_accum).backward()
            for k in acc:
                acc[k] += losses[k].item() / grad_accum
            last_stats = stats  # tensors; converted once below, not per micro-step

        # Gradients must be unscaled before clipping, or the threshold is applied to
        # scaled values and effectively does nothing.
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = lr_at(step, args.max_lr, args.max_lr * args.min_lr_frac,
                   args.warmup_steps, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr * (args.router_lr_mult if g.get("is_router") else 1.0)
        scaler.step(optimizer)
        scaler.update()

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tps = args.total_batch_tokens / dt

        if step % args.log_every == 0:
            last_stats = stats_to_floats(last_stats)
            row = {"step": step, "split": "train", "loss": acc["lm"],
                   "router": acc["router"], "depth_loss": acc["depth"], "lr": lr,
                   "grad_norm": float(norm), "lambda_depth": lam_depth,
                   "dt_ms": dt * 1000, "tokens_per_sec": tps,
                   "epoch": (step + 1) * args.total_batch_tokens / max(epoch_tokens, 1),
                   **last_stats}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if step % max(args.log_every, 10) == 0:
                print(f"step {step:5d} | ep {row['epoch']:5.2f} "
                      f"| loss {acc['lm']:.4f} | bce {acc['router']:.3f} "
                      f"| lr {lr:.2e} | norm {float(norm):.2f} "
                      f"| depth {last_stats.get('depth', 0):5.2f} "
                      f"| exit {last_stats.get('exit_frac', 0):.2f} "
                      f"| {dt*1000:.0f}ms | {tps:,.0f} tok/s")

        if step > 0 and (step % args.ckpt_every == 0 or last):
            torch.save({
                # raw_model, not model: under --compile the latter is an
                # OptimizedModule whose state_dict prefixes every key with
                # `_orig_mod.`, which nothing downstream can load.
                "model": raw_model.state_dict(), "config": cfg,
                "optimizer": optimizer.state_dict(),
                "loader": train_loader.state_dict(), "step": step,
                "args": vars(args), "best_val": best_val,
                "scaler": scaler.state_dict() if use_scaler else None,
            }, os.path.join(run_dir, f"ckpt_{step:06d}.pt"))
            print(f"  [ckpt] step {step}")
            prune_checkpoints(run_dir, args.keep_ckpts)

    print(f"\ndone -> {run_dir}")


if __name__ == "__main__":
    main()
