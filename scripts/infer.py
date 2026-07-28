"""Load a trained checkpoint and see what it does.

    python scripts/infer.py eval     runs/r1_b6_amt_joint --data-dir DATA
    python scripts/infer.py generate runs/r1_b6_amt_joint --prompt "The key idea is"

Three things about this model make naive inference quietly wrong, so they are handled
here rather than left to the caller.

1. ROUTER THRESHOLDS MUST BE CALIBRATED. Training selects tokens with a top-k over
   the whole sequence, which peeks at the future and cannot be used to generate. The
   causal path thresholds the auxiliary predictor instead, and that threshold is
   meaningless until calibrated against real data (routers.py::calibrate). An
   uncalibrated router still runs and still emits fluent-looking text -- it just
   routes at whatever rate the aux head happens to produce, which is not the model
   that was trained. The threshold buffer is persistent, so a checkpoint saved after
   an eval carries a calibrated one; this script checks and warns when it does not.

2. GENERATION NEVER WRITES TO THE MEMORY BANK. `AMT.generate` reads from the bank but
   does not write to it, so for a memory variant an empty bank means the retrieval
   path contributes exactly nothing and you are looking at a depth-routed model with
   extra steps. `--context` fills the bank from a passage first, which is what makes
   the memory half observable at all.

3. THE VOCAB IS PADDED. vocab_size is 50304 (a multiple of 128) but GPT-2 BPE only
   defines 50257, so ids 50257-50303 are trainable-but-never-observed and cannot be
   decoded. They are filtered on decode and reported if they appear.
"""

import argparse
import glob
import math
import os
import sys

import torch

sys.path.insert(0, ".")

from amt.data import DocSegmentLoader  # noqa: E402
from amt.model import AMT  # noqa: E402
from amt.model.amt import strip_compile_prefix  # noqa: E402
from amt.model.routers import TopKTokenRouter  # noqa: E402
from amt.precision import describe_device, select_precision  # noqa: E402
from amt.train import evaluate  # noqa: E402

GPT2_VOCAB = 50257          # real BPE size; the config pads this to 50304


def resolve_ckpt(path):
    """Accept a run directory or a checkpoint file."""
    if os.path.isfile(path):
        return path
    ckpts = sorted(glob.glob(os.path.join(path, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {path}")
    return ckpts[-1]


def load_model(path, device):
    ckpt_path = resolve_ckpt(path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = AMT(cfg).to(device).eval()
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    print(f"loaded      : {os.path.basename(ckpt_path)} (step {ck.get('step', '?')})")
    print(f"variant     : {ck.get('args', {}).get('variant', '?')}  "
          f"params {model.num_params()/1e6:.2f}M non-embedding")
    return model, cfg, ck


def routers_of(model):
    return [m for m in model.modules() if isinstance(m, TopKTokenRouter)]


def check_calibration(model, data_dir, cfg, device, bank, B, recalibrate=True):
    """Warn on, and optionally fix, uncalibrated router thresholds.

    A threshold still sitting at its init of exactly 0.0 means sigmoid(logit) > 0.0 is
    true for every token, so the router selects everything and the causal path routes
    at rate 1.0 -- dense, silently.
    """
    routers = routers_of(model)
    if not routers:
        return
    stale = [r.name for r in routers if float(r.threshold) == 0.0]
    if stale:
        print(f"WARNING     : uncalibrated thresholds on {', '.join(stale)}")

    if not (recalibrate and data_dir):
        if stale:
            print("              pass --data-dir to calibrate; routing is not "
                  "trustworthy until you do")
        return

    loader = DocSegmentLoader(data_dir, B, cfg.block_size, split="val", shuffle=False)
    batches = [loader.next_batch()[0] for _ in range(3)]
    if bank is not None:
        bank.clear()
    rates = model.calibrate_routers(batches, device=device, bank=bank)
    if bank is not None:
        bank.clear()
    print("calibrated  : " + "  ".join(f"{k}={float(v):.3f}" for k, v in rates.items()))


def warm_bank(model, bank, ids, cfg, device):
    """Fill the memory bank from a context passage, segment by segment.

    Mirrors training: forward a segment, then write it. Never write before the
    forward that consumes it, or the model retrieves the tokens it is predicting.
    """
    if bank is None:
        return 0
    bank.clear()
    T = cfg.block_size
    written = 0
    with torch.no_grad():
        for s in range(0, len(ids), T):
            seg = ids[s:s + T]
            if len(seg) < 8:                     # a scrap of a segment is not worth a write
                break
            x = torch.tensor(seg, dtype=torch.long, device=device).unsqueeze(0)
            x = x.expand(bank.B, -1).contiguous()
            _, _, _, kv = model(x, bank=bank)
            model.write_memory(bank, kv)
            written += len(seg)
    print(f"bank warmed : {written} tokens, fill={float(bank.fill.float().mean()):.0f}"
          f"/{bank.capacity}")
    return written


def generate(model, cfg, enc, args, device, bank):
    """Sample, collecting the routing decision made for each generated token."""
    per_token = {r.name: [] for r in routers_of(model)}

    def hook(mod, inp, out):
        # The decision that matters for generation is the one at the last position:
        # that is the token about to be emitted.
        per_token[mod.name].append(float(out["mask"][:, -1].float().mean()))

    handles = [r.register_forward_hook(hook) for r in routers_of(model)]
    ids = enc.encode(args.prompt)
    n_bad, n_new = 0, 0
    try:
        for i in range(args.num_samples):
            idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            # Distinct seed per sample: reusing one makes every sample identical and
            # makes a degenerate model look deceptively consistent.
            gen = torch.Generator(device=device).manual_seed(args.seed + i)
            out = model.generate(idx, args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 bank=bank, generator=gen)

            new = out[0, len(ids):].tolist()
            n_bad += sum(1 for t in new if t >= GPT2_VOCAB)
            n_new += len(new)
            text = enc.decode([t for t in new if t < GPT2_VOCAB])
            label = (f" sample {i + 1}/{args.num_samples} "
                     if args.num_samples > 1 else " ")
            print(f"\n{('-' * 8) + label:-<70}\n{args.prompt}{text}")
    finally:
        for h in handles:
            h.remove()
    print("-" * 70)

    if n_bad:
        print(f"note: {n_bad}/{n_new} sampled ids were in the padded vocab "
              f"({GPT2_VOCAB}-{cfg.vocab_size-1}) and were dropped on decode. "
              f"Expected early in training; persistent means the head has mass on "
              f"tokens that never occur.")

    if per_token and any(per_token.values()):
        print("\nrouting during generation (fraction of generated tokens taking "
              "each path)")
        for name, vals in per_token.items():
            if not vals:
                continue
            target = (cfg.mem_capacity if name == "mem_router" else cfg.depth_capacity)
            print(f"  {name:<20} {sum(vals)/len(vals):.3f}   (trained capacity "
                  f"{target:.2f})")
        print("A rate far from the trained capacity means the causal threshold does "
              "not\nreproduce the top-k selection -- check agree/* from training.")


def run_eval(model, cfg, args, device, bank):
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype, _, _ = select_precision(device, verbose=False)
    loader = DocSegmentLoader(args.data_dir, args.batch_size, cfg.block_size,
                              split="val", shuffle=False)
    loss, stats = evaluate(model, loader, bank, device, device_type,
                           steps=args.eval_steps, raw_model=model, amp_dtype=amp_dtype)
    print(f"\nval loss    : {loss:.4f}   ppl {math.exp(min(loss, 20)):.2f}   "
          f"({args.eval_steps} x {args.batch_size} x {cfg.block_size} tokens)")
    for k in sorted(stats):
        print(f"  {k:<24} {stats[k]:.4f}")
    if any(k.startswith("agree/") for k in stats):
        print("\nagree/* below ~0.85 means the model you would generate with is not "
              "the model\nthat was trained (routers.py). Read it before trusting any "
              "sample above.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("run", help="run directory or a ckpt_*.pt path")
    common.add_argument("--data-dir", default="data/fineweb_edu_docs")
    common.add_argument("--batch-size", type=int, default=4)
    common.add_argument("--no-calibrate", action="store_true",
                        help="trust the checkpoint's saved thresholds")

    e = sub.add_parser("eval", parents=[common], help="val loss and routing telemetry")
    e.add_argument("--eval-steps", type=int, default=40)

    g = sub.add_parser("generate", parents=[common], help="sample text")
    g.add_argument("--prompt", default="The most important idea in this paper is")
    g.add_argument("--context", default=None,
                   help="passage (or a .txt path) used to fill the memory bank "
                        "before generating. Without it a memory variant retrieves "
                        "from an empty bank and the memory half does nothing")
    g.add_argument("--max-new-tokens", type=int, default=200)
    g.add_argument("--num-samples", type=int, default=4,
                   help="one sample says little about a model; each uses seed+i")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device      : {describe_device()}")
    model, cfg, _ = load_model(args.run, device)

    amp_dtype, _, _ = select_precision(device, verbose=False)
    # Generation runs one sequence; eval runs a batch. The bank is allocated for a
    # fixed batch size and asserts on it, so calibration must use the same one.
    bank_B = args.batch_size if args.mode == "eval" else 1
    bank = model.make_bank(bank_B, device, dtype=amp_dtype) if cfg.use_memory else None

    check_calibration(model, args.data_dir, cfg, device, bank,
                      B=bank_B, recalibrate=not args.no_calibrate)

    if args.mode == "eval":
        run_eval(model, cfg, args, device, bank)
        return

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    if args.context:
        text = (open(args.context).read() if os.path.exists(args.context)
                else args.context)
        warm_bank(model, bank, enc.encode(text), cfg, device)
    elif cfg.use_memory:
        print("note        : no --context, so the bank is empty and retrieval "
              "contributes nothing")
    generate(model, cfg, enc, args, device, bank)


if __name__ == "__main__":
    main()
