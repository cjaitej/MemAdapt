"""Perplexity by position within the document -- base model vs retrofit.

    # the base: frozen GPT-2, no modules, nothing trained
    python scripts/eval_retrofit.py --init-from gpt2 --variant b1_dense

    # a trained arm
    python scripts/eval_retrofit.py --run runs/pilot_rf_b6_amt_joint

Why bucket by segment index
---------------------------
The overall perplexity of a memory model is a blend of two very different regimes and
hides the effect being measured. On a document's first segment the bank is empty, so
the retrofit and the base are running the same computation and must score the same;
by the fourth segment the bank holds ~1500 tokens the base cannot see at all. If
retrieval is doing anything, the gap has to widen with depth into the document -- and
a single averaged number cannot show that, while a gap that appears at segment 0 is
evidence of a bug rather than of memory.

This is the comparison the application rests on: GPT-2's window is 1024 tokens and the
corpus is a median 3133 (`data/fineweb_edu_docs`), so most of every document is
outside what the base model can condition on.

Both arms are evaluated teacher-forced, with the same top-k routing used in training.
That measures the model that was trained; the separate question of whether the causal
path reproduces it is what `agree/*` in the run log is for.
"""

import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from amt.data import DocSegmentLoader  # noqa: E402
from amt.model import AMT  # noqa: E402
from amt.model.amt import strip_compile_prefix  # noqa: E402
from amt.model.config import VARIANTS  # noqa: E402
from amt.model.routers import TopKTokenRouter  # noqa: E402
from amt.precision import describe_device, select_precision  # noqa: E402


def load_from_run(path, device):
    """A trained checkpoint: prefer best.pt, fall back to the newest periodic one."""
    if os.path.isfile(path):
        ckpt = path
    else:
        best = os.path.join(path, "best.pt")
        found = sorted(glob.glob(os.path.join(path, "ckpt_*.pt")))
        if not (os.path.exists(best) or found):
            raise FileNotFoundError(f"no best.pt or ckpt_*.pt in {path}")
        ckpt = best if os.path.exists(best) else found[-1]
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = AMT(ck["config"]).to(device).eval()
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    label = f"{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}"
    return model, ck["config"], label


def load_base(init_from, variant_name, block_size, device):
    """An untrained retrofit. With --variant b1_dense that is exactly stock GPT-2.

    Evaluating the base needs its own path because there is no checkpoint to point at:
    the control in this experiment is a model nobody trained, which is the whole
    reason the comparison is clean.
    """
    from amt.model.retrofit import from_gpt2, gpt2_config
    cfg = gpt2_config(init_from, block_size=block_size, **VARIANTS[variant_name])
    model, _ = from_gpt2(init_from, cfg, device=device)
    return model.eval(), cfg, f"{init_from}:{variant_name} (untrained)"


class DepthProbe:
    """Per-token routing decisions, collected with hooks.

    Hooks rather than a model change, for the same reason compare.py uses them: the
    masks are already exposed, and threading another return value through forward
    would touch the compiled path for an offline measurement.
    """

    def __init__(self, model, cfg):
        self.routers = [m for m in model.modules() if isinstance(m, TopKTokenRouter)]
        # With route_memory=False (baseline B2) there is no memory router, because
        # every token retrieves. Reporting 0.000 there would read as "retrieval never
        # happened" when it happened for everything: the absence of a decision is
        # itself the decision.
        self.mem_always = cfg.use_memory and not cfg.route_memory
        self.depth, self.mem = [], []
        self._handles = []

    def __enter__(self):
        self._handles = [r.register_forward_hook(self._hook) for r in self.routers]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        return False

    def _hook(self, mod, inp, out):
        (self.mem if mod.name == "mem_router" else self.depth).append(
            out["mask"].detach().float())

    def take(self, shape, n_always_on, device):
        """(depth, mem) as (B, T), then reset for the next batch."""
        depth = (torch.stack(self.depth).sum(0) if self.depth
                 else torch.zeros(shape, device=device))
        if self.mem:
            mem = torch.stack(self.mem).sum(0)
        elif self.mem_always:
            mem = torch.ones(shape, device=device)
        else:
            mem = torch.zeros(shape, device=device)
        self.depth, self.mem = [], []
        return depth + n_always_on, mem


def per_token_loss(logits, targets, chunks=8):
    """Cross-entropy per token, chunked so the fp32 upcast never exists all at once.

    (B, T, 50257) in fp32 is 800MB at B=8 -- the same reason amt.py chunks its
    training loss, and the same fix.
    """
    B, T, V = logits.shape
    flat_logits = logits.view(-1, V)
    flat_t = targets.reshape(-1)
    out = [F.cross_entropy(lc.float(), tc, reduction="none")
           for lc, tc in zip(flat_logits.chunk(chunks), flat_t.chunk(chunks))]
    return torch.cat(out).view(B, T)


@torch.no_grad()
def run(model, cfg, args, device):
    amp_dtype, _, _ = select_precision(device, verbose=False)
    loader = DocSegmentLoader(args.data_dir, args.batch_size, cfg.block_size,
                              split="val", shuffle=False)
    bank = (model.make_bank(args.batch_size, device, dtype=amp_dtype)
            if cfg.use_memory else None)
    n_always_on = cfg.n_layer - len(cfg.adaptive_layers)
    cap = args.max_segment

    # bucket -> [loss_sum, tokens, depth_sum, mem_sum]
    buckets = {}
    seg = torch.zeros(args.batch_size, dtype=torch.long)
    autocast = torch.autocast(device_type="cuda", dtype=amp_dtype,
                              enabled=device == "cuda")

    with DepthProbe(model, cfg) as probe:
        for _ in range(args.batches):
            x, y, reset = loader.next_batch()
            x, y = x.to(device), y.to(device)
            reset_cpu = torch.as_tensor(reset).cpu().bool()
            seg[reset_cpu] = 0
            if bank is not None:
                bank.clear(reset)          # BEFORE the forward -- see write_memory

            with autocast:
                logits, _, _, kv = model(x, bank=bank, return_logits=True)
            losses = per_token_loss(logits, y, args.ce_chunks)
            depth, mem = probe.take(x.shape, n_always_on, device)
            if bank is not None:
                model.write_memory(bank, kv)

            for b in range(args.batch_size):
                key = min(int(seg[b]), cap)
                acc = buckets.setdefault(key, [0.0, 0, 0.0, 0.0])
                acc[0] += float(losses[b].sum())
                acc[1] += losses.shape[1]
                acc[2] += float(depth[b].sum())
                acc[3] += float(mem[b].sum())
            seg += 1

    rows = []
    for key in sorted(buckets):
        loss_sum, n, depth_sum, mem_sum = buckets[key]
        rows.append({
            "segment": key, "tokens": n,
            "loss": loss_sum / n, "ppl": math.exp(min(loss_sum / n, 20)),
            "layers_per_token": depth_sum / n, "mem_rate": mem_sum / n,
            "capped": key == cap,
        })
    total_loss = sum(v[0] for v in buckets.values())
    total_n = sum(v[1] for v in buckets.values())
    return rows, total_loss / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="a trained run directory or ckpt_*.pt")
    src.add_argument("--init-from", help="evaluate an UNTRAINED retrofit, e.g. gpt2")
    ap.add_argument("--variant", default="b1_dense",
                    help="with --init-from: which arm to build. b1_dense is the "
                         "frozen base with no added modules")
    ap.add_argument("--data-dir", default="data/fineweb_edu_docs")
    ap.add_argument("--block-size", type=int, default=512,
                    help="with --init-from only; a run carries its own")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=60,
                    help="more batches means deeper segment buckets get filled")
    ap.add_argument("--max-segment", type=int, default=8,
                    help="segments at or beyond this are pooled into one bucket")
    ap.add_argument("--ce-chunks", type=int, default=8)
    ap.add_argument("--json", help="write the rows here for plotting")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device      : {describe_device()}")
    if args.run:
        model, cfg, label = load_from_run(args.run, device)
    else:
        model, cfg, label = load_base(args.init_from, args.variant,
                                      args.block_size, device)
    print(f"model       : {label}")
    print(f"layout      : {cfg.n_layer} layers, block_size {cfg.block_size}, "
          f"memory {'on' if cfg.use_memory else 'off'}, "
          f"depth routing {'on' if cfg.route_depth else 'off'}")

    rows, overall = run(model, cfg, args, device)

    print(f"\noverall     : loss {overall:.4f}  ppl {math.exp(min(overall, 20)):.2f}")
    print(f"\n{'segment':>8} {'tokens':>10} {'ppl':>9} {'layers/tok':>11} {'mem rate':>9}")
    print("-" * 52)
    for r in rows:
        name = f"{r['segment']}+" if r["capped"] else str(r["segment"])
        print(f"{name:>8} {r['tokens']:>10,} {r['ppl']:>9.2f} "
              f"{r['layers_per_token']:>11.2f} {r['mem_rate']:>9.3f}")

    print("\nSegment 0 is the empty-bank case: a memory model must match the base "
          "there.\nA gap that only opens at later segments is retrieval doing work; a "
          "gap at\nsegment 0 is a bug. Run the base with --init-from gpt2 --variant "
          "b1_dense to\nget the row to subtract.")
    print(f"\nCompare ONLY against a base run with the same --batch-size "
          f"({args.batch_size}) and\n--batches ({args.batches}): the loader is "
          "deterministic, so those two settings fix\nwhich documents and which "
          "segments were scored. They also fix the token counts\nabove -- deep buckets "
          "are thin, and the ppl differences BETWEEN buckets are\ndominated by which "
          "documents were long enough to reach them, not by depth.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": label, "overall_loss": overall, "rows": rows}, f,
                      indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
