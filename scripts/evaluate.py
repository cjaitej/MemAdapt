"""Evaluate one checkpoint: perplexity, depth, FLOPs, and router health.

    python scripts/evaluate.py --ckpt runs/s3_joint/final.pt
    python scripts/evaluate.py --ckpt runs/s3_joint/final.pt --confidence
    python scripts/evaluate.py --ckpt runs/s3_joint/final.pt --oracle

`--oracle` is the diagnostic worth running before concluding the method failed. It
reports the depth the Delta rule would have chosen if it could see the future, and how
often the router agrees with it. Three readings, three different problems:

* router agrees, quality is bad  -> the *rule* is wrong; try `--target-type kl`
* router disagrees, oracle is shallow -> the router failed to learn a learnable signal
* oracle itself is deep -> there is no easy-token structure here to exploit, and no
  router can invent it. That is a real result about the model and the corpus, not a
  bug, and it is worth reporting rather than tuning around.
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

from agpt.data import SegmentLoader
from agpt.evaluate import arm_metrics, generation_latency, load_checkpoint
from agpt.model.targets import exit_targets, oracle_depth


@torch.no_grad()
def router_confidence(model, loader, device, steps=20, bins=40):
    """Histogram of every router's continue probability over a validation stream."""
    cfg = model.config
    edges = torch.linspace(0, 1, bins + 1)
    counts = torch.zeros(bins, dtype=torch.float64)
    loader.reset()
    for _ in range(steps):
        x, _ = loader.next_batch()
        x = x.to(device)
        h = model._embed(x)
        for l, block in enumerate(model.transformer.h):
            h = block(h)
            key = str(l)
            if key in model.routers:
                p = torch.sigmoid(model.routers[key](h)).flatten().float().cpu()
                counts += torch.histc(p, bins=bins, min=0.0, max=1.0).double()
    return {"edges": edges.tolist(), "counts": counts.tolist(),
            "n_routers": len(cfg.router_layers)}


@torch.no_grad()
def oracle_agreement(model, loader, device, steps=20):
    """Compare the learned routing against the label it was trained on."""
    cfg = model.config
    if not cfg.router_layers:
        return None
    loader.reset()
    agree, n, oracle_sum, learned_sum = 0.0, 0, 0.0, 0.0
    for _ in range(steps):
        x, _ = loader.next_batch()
        x = x.to(device)
        hiddens = model.dense_hidden_states(x)
        labels, _ = exit_targets(hiddens, cfg, model.transformer.ln_f, model.lm_head)
        oracle_sum += oracle_depth(labels, cfg.n_min_layers).float().mean().item()
        learned_sum += model.token_depths(x).mean().item()

        for l in cfg.router_layers:
            p = torch.sigmoid(model.routers[str(l)](hiddens[l + 1]))
            pred = (p > cfg.exit_threshold).float()
            agree += (pred == labels[l]).float().mean().item()
            n += 1
    return {"router_vs_oracle_agreement": agree / max(n, 1),
            "oracle_depth": oracle_sum / steps,
            "learned_depth": learned_sum / steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="data/wikitext103")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--exit-mode", default=None,
                    help="evaluate the checkpoint under a different routing rule")
    ap.add_argument("--confidence", action="store_true",
                    help="also write results/confidence.json for figure 4")
    ap.add_argument("--oracle", action="store_true",
                    help="also compare the router against the derived labels")
    ap.add_argument("--gen-tokens", type=int, nargs="*", default=[],
                    help="also time generation at these lengths, e.g. 100 500 1000")
    ap.add_argument("--dense-path", action="store_true",
                    help="evaluate on the gated path instead of the compact one; "
                         "they compute the same function, so this only differs in "
                         "speed (and in what a bug would look like)")
    ap.add_argument("--out", default="results/evaluate.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    over = {"exit_mode": args.exit_mode} if args.exit_mode else {}
    model, cfg, ck = load_checkpoint(args.ckpt, device, **over)
    loader = SegmentLoader(args.data_dir, args.batch_size, cfg.block_size, split="val")

    out = {"ckpt": args.ckpt, "step": ck.get("step"), "config": cfg.to_dict()}
    out["metrics"] = arm_metrics(model, loader, device, args.eval_steps,
                                 compact=not args.dense_path)
    m = out["metrics"]
    print(f"\nloss {m['loss']:.4f}   ppl {m['ppl']:.2f}   "
          f"depth {m['avg_depth']:.2f}/{cfg.n_layer}")
    print(f"layer FLOPs {m['flops_frac_layers']:.1%} of dense   "
          f"total FLOPs {m['flops_frac_total']:.1%} of dense")

    if args.oracle:
        out["oracle"] = oracle_agreement(model, loader, device,
                                         min(args.eval_steps, 20))
        if out["oracle"]:
            o = out["oracle"]
            print(f"oracle depth {o['oracle_depth']:.2f}   "
                  f"learned depth {o['learned_depth']:.2f}   "
                  f"agreement {o['router_vs_oracle_agreement']:.1%}")

    if args.gen_tokens:
        out["generation"] = []
        prompt = torch.randint(0, min(cfg.vocab_size, 50257), (1, 8), device=device)
        for n in args.gen_tokens:
            g = generation_latency(model, prompt, n, device)
            g.pop("output")
            out["generation"].append(g)
            print(f"generate {n:5d} tokens: {g['seconds']:.2f}s  "
                  f"{g['tokens_per_sec']:.1f} tok/s  mean depth {g['mean_depth']:.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    if args.confidence:
        if not cfg.router_layers:
            print("no routers on this checkpoint; skipping the confidence histogram")
        else:
            conf = router_confidence(model, loader, device, min(args.eval_steps, 20))
            path = os.path.join(os.path.dirname(args.out) or ".", "confidence.json")
            with open(path, "w") as f:
                json.dump(conf, f, indent=2)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
