"""Compare routed variants against the dense baseline.

Two questions, two modes, because they need different evidence.

`table` -- is the routing worth its cost?
    Perplexity alone cannot answer this. A routed variant spends fewer FLOPs per
    token than the dense control, so at equal training tokens it *should* be
    slightly worse; reporting that as "routing hurts" is as wrong as reporting the
    FLOP saving without the perplexity as "routing wins". The table puts quality and
    both cost axes side by side so the trade is visible.

    Both cost axes, because they disagree. matmul FLOPs is the convention the
    literature reports and the only axis FlopCounterMode can validate; bytes moved
    is where the memory path's cost actually lives (see amt/model/flops.py). A
    variant can look efficient on one and expensive on the other.

`align` -- is the routing *adaptive*, or merely sparse?
    This is the question a dense baseline cannot answer at all, and the one the
    project's contribution rests on. A router that hits its capacity target by
    picking tokens arbitrarily produces exactly the same FLOP saving as one that
    picks the tokens that need the compute. Perplexity conflates them; b4_random
    exists as the control precisely because of this.

    So: measure whether depth goes where difficulty is. Difficulty comes from an
    independent dense reference model's per-token loss, never from the routed model
    itself -- a token that received more compute has lower loss almost by
    construction, and correlating those would produce a confident number that means
    nothing.

    python scripts/compare.py table runs/r1_b1_dense runs/r1_b6_amt_joint
    python scripts/compare.py align runs/r1_b6_amt_joint --reference runs/r1_b1_dense
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from amt.data import DocSegmentLoader  # noqa: E402
from amt.model import AMT, FlopModel  # noqa: E402
from amt.model.amt import strip_compile_prefix  # noqa: E402
from amt.model.routers import TopKTokenRouter  # noqa: E402
from amt.precision import select_precision  # noqa: E402


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------

def latest_ckpt(run_dir):
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_*.pt in {run_dir}")
    return ckpts[-1]


def load_run(run_dir):
    """Log rows plus the config, which only the checkpoint carries."""
    log_path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"no log.jsonl in {run_dir}")
    rows = [json.loads(line) for line in open(log_path) if line.strip()]

    ck = torch.load(latest_ckpt(run_dir), map_location="cpu", weights_only=False)
    return {
        "name": os.path.basename(os.path.normpath(run_dir)),
        "config": ck["config"],
        "args": ck.get("args", {}),
        "step": ck.get("step", 0),
        "train": [r for r in rows if r.get("split") == "train"],
        "val": [r for r in rows if r.get("split") == "val"],
    }


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def summarise(run):
    """One row of the comparison table."""
    cfg, fm = run["config"], FlopModel(run["config"])
    val = run["val"]
    train = run["train"]

    # Skip the first few train rows: step 0 carries the torch.compile cost and would
    # drag the median down by more than the effect being measured.
    tps = median([r["tokens_per_sec"] for r in train[3:] if "tokens_per_sec" in r])
    hours = sum(r.get("dt_ms", 0) for r in train) / 3.6e6

    traffic = fm.retrieval_traffic()
    mc = cfg.effective_mem_capacity
    final = val[-1] if val else {}
    return {
        "name": run["name"],
        "step": run["step"],
        "val_loss": final.get("loss", float("nan")),
        "ppl": _ppl(final.get("loss")),
        "best_ppl": _ppl(min((r["loss"] for r in val), default=None)),
        "mflops": fm.breakdown().total / 1e6,
        "kb_tok": mc * traffic.total / 1e3,
        "l_tok": final.get("layers_per_token", float("nan")),
        "tok_s": tps,
        "hours": hours,
        "agree": _mean_agreement(final),
    }


def _ppl(loss):
    import math
    if loss is None:
        return float("nan")
    return math.exp(min(loss, 20))


def _mean_agreement(row):
    vals = [v for k, v in row.items() if k.startswith("agree/")]
    return sum(vals) / len(vals) if vals else float("nan")


def print_table(runs):
    rows = [summarise(r) for r in runs]
    # The dense control is the denominator for every relative number below.
    dense = next((r for r in rows if "dense" in r["name"]), rows[0])

    print(f"{'run':<20} {'step':>6} {'ppl':>9} {'MFLOP/tok':>10} {'KB/tok':>8} "
          f"{'l/tok':>6} {'tok/s':>8} {'GPU-h':>6} {'agree':>6}")
    print("-" * 90)
    for r in rows:
        print(f"{r['name']:<20} {r['step']:>6} {r['ppl']:>9.2f} {r['mflops']:>10.1f} "
              f"{r['kb_tok']:>8.1f} {r['l_tok']:>6.2f} {r['tok_s']:>8,.0f} "
              f"{r['hours']:>6.2f} {r['agree']:>6.3f}")

    print(f"\nrelative to {dense['name']} (quality cost vs budget saved)")
    print(f"{'run':<20} {'ppl delta':>10} {'FLOP saved':>11} {'bytes added':>12} "
          f"{'speedup':>8}")
    print("-" * 66)
    for r in rows:
        if r["name"] == dense["name"]:
            continue
        d_ppl = r["ppl"] - dense["ppl"]
        d_flop = 1 - r["mflops"] / dense["mflops"]
        d_bytes = r["kb_tok"] - dense["kb_tok"]
        speed = r["tok_s"] / dense["tok_s"] if dense["tok_s"] else float("nan")
        print(f"{r['name']:<20} {d_ppl:>+10.3f} {d_flop:>10.1%} "
              f"{d_bytes:>+11.1f}K {speed:>7.2f}x")

    print("\nA FLOP saving that costs perplexity is a trade, not a win: read the two "
          "columns together.\nSpeedup is measured, FLOP saving is analytic -- they "
          "diverge because ~80% of the\nretrieval path is not matmul (amt/model/flops.py).")


# ---------------------------------------------------------------------------
# Alignment: does compute go where difficulty is?
# ---------------------------------------------------------------------------

def build(run_dir, device):
    ck = torch.load(latest_ckpt(run_dir), map_location=device, weights_only=False)
    model = AMT(ck["config"]).to(device).eval()
    model.load_state_dict(strip_compile_prefix(ck["model"]))
    return model, ck["config"]


@torch.no_grad()
def per_token_difficulty(model, x, y, chunks=8):
    """Per-token cross-entropy from the dense reference: the difficulty proxy.

    Chunked over rows because the logits tensor is (B, T, 50304) and materialising
    the whole thing alongside the routed model's activations does not fit on a small
    card.
    """
    logits, _, _, _ = model(x, bank=None, return_logits=True)
    B, T, V = logits.shape
    flat_logits, flat_y = logits.view(-1, V), y.reshape(-1)
    out = []
    for part_logits, part_y in zip(flat_logits.chunk(chunks), flat_y.chunk(chunks)):
        out.append(F.cross_entropy(part_logits.float(), part_y, reduction="none"))
    return torch.cat(out).view(B, T)


@torch.no_grad()
def per_token_depth(model, x, bank):
    """Adaptive layers each token actually ran, collected via router hooks.

    Hooks rather than a model change: the routers already expose their selection
    mask, and threading an extra return value through forward would touch the
    compiled path for the sake of an offline analysis.
    """
    depth_masks, mem_masks = [], []

    def hook(mod, inp, out):
        (mem_masks if mod.name == "mem_router" else depth_masks).append(
            out["mask"].detach().float())

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, TopKTokenRouter)]
    try:
        _, _, _, kv = model(x, bank=bank)
    finally:
        for h in handles:
            h.remove()
    if bank is not None:
        model.write_memory(bank, kv)

    depth = (torch.stack(depth_masks).sum(0) if depth_masks
             else torch.zeros_like(x, dtype=torch.float))
    mem = (torch.stack(mem_masks).sum(0) if mem_masks
           else torch.zeros_like(x, dtype=torch.float))
    return depth, mem


def spearman(a, b):
    """Rank correlation. Ties get arbitrary ranks, which matters here.

    Depth is an integer in [0, n_adaptive], so it is heavily tied and this number is
    attenuated. It is reported as a secondary signal; the decile gap below is the
    robust one.
    """
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return (ra * rb).mean().item()


def decile_gap(difficulty, depth):
    """Mean depth on the hardest 10% of tokens minus the easiest 10%.

    Robust to the heavy tying in depth, and reported in layers, which is a unit that
    means something: "+0.8 layers" says the router spends most of one extra layer on
    hard tokens. Zero means compute is allocated independently of difficulty --
    which is exactly what b4_random should produce.
    """
    n = difficulty.numel()
    order = difficulty.argsort()
    cut = max(1, n // 10)
    easy, hard = order[:cut], order[-cut:]
    return depth[hard].mean().item() - depth[easy].mean().item()


def run_alignment(routed_dir, reference_dir, data_dir, B, T, batches, device):
    routed, rcfg = build(routed_dir, device)
    reference, _ = build(reference_dir, device)

    amp_dtype, _, _ = select_precision(device, verbose=False)
    loader = DocSegmentLoader(data_dir, B, T, split="val", shuffle=False)
    bank = (routed.make_bank(B, device, dtype=amp_dtype)
            if rcfg.use_memory else None)

    # Calibration matters even though this is top-k mode: an uncalibrated mem_router
    # threshold makes the memory mask meaningless. See train.evaluate.
    calib = [loader.next_batch()[0].to(device) for _ in range(3)]
    routed.calibrate_routers(calib, device=device, bank=bank)
    loader.reset()
    if bank is not None:
        bank.clear()

    diffs, depths, mems = [], [], []
    autocast = torch.autocast(device_type=device, dtype=amp_dtype,
                              enabled=device == "cuda")
    for _ in range(batches):
        x, y, reset = loader.next_batch()
        x, y = x.to(device), y.to(device)
        if bank is not None:
            bank.clear(reset)
        with autocast:
            d = per_token_difficulty(reference, x, y)
            depth, mem = per_token_depth(routed, x, bank)
        diffs.append(d.flatten().float().cpu())
        depths.append(depth.flatten().cpu())
        mems.append(mem.flatten().cpu())

    difficulty = torch.cat(diffs)
    depth = torch.cat(depths)
    mem = torch.cat(mems)

    n_adaptive = len(rcfg.adaptive_layers)
    print(f"\ncompute-difficulty alignment: {os.path.basename(routed_dir)}")
    print(f"difficulty from {os.path.basename(reference_dir)} per-token loss, "
          f"{difficulty.numel():,} tokens\n")
    print(f"{'':<26} {'depth':>10} {'memory':>10}")
    print("-" * 48)
    print(f"{'mean rate':<26} {depth.mean():>10.3f} {mem.mean():>10.3f}")
    print(f"{'decile gap (hard-easy)':<26} {decile_gap(difficulty, depth):>10.3f} "
          f"{decile_gap(difficulty, mem):>10.3f}")
    print(f"{'spearman rho':<26} {spearman(difficulty, depth):>10.3f} "
          f"{spearman(difficulty, mem):>10.3f}")
    print(f"\ndepth is out of {n_adaptive} adaptive layers; memory out of 1 "
          f"retrieval decision.")
    print("A decile gap near zero means compute is allocated independently of "
          "difficulty --\nthe same FLOP saving a random router achieves. Compare "
          "against b4_random, which\nshould score ~0 by construction: that "
          "difference is the adaptivity claim.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("table", help="quality and both cost axes, across runs")
    t.add_argument("runs", nargs="+")

    a = sub.add_parser("align", help="does depth go where difficulty is?")
    a.add_argument("run")
    a.add_argument("--reference", required=True,
                   help="a DENSE run's directory; its per-token loss is the "
                        "difficulty proxy. Using the routed model itself would be "
                        "circular")
    a.add_argument("--data-dir", default="data/fineweb_edu_docs")
    a.add_argument("--batch-size", type=int, default=4)
    a.add_argument("--block-size", type=int, default=512)
    a.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "table":
        print_table([load_run(r) for r in args.runs])
    else:
        run_alignment(args.run, args.reference, args.data_dir, args.batch_size,
                      args.block_size, args.batches, device)


if __name__ == "__main__":
    main()
