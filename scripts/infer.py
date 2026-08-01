"""Sample from a checkpoint and show how deep each token went.

    python scripts/infer.py --ckpt runs/s3_joint/best.pt --prompt "The capital of"
    python scripts/infer.py --ckpt runs/s3_joint/best.pt --show-depth

An aggregate depth of 6.2 layers/token tells you the model is cheaper. It does not
tell you it is cheaper *for the right tokens*, which is the actual claim. Printing the
depth beside the text does: function words, punctuation and the second half of a
predictable word should be shallow, and content words at the start of a clause should
not. If the depth pattern looks like noise, the average is a budget the router hit
rather than a decision it made.

One vocabulary trap: `vocab_size` is padded to a multiple of 128 for tensor-core
alignment, so the model can sample an id that GPT-2's BPE has no token for. Those are
replaced rather than crashed on, and counted -- a nonzero count on a trained model
means the head is putting mass on tokens that do not exist.
"""

import os
import sys

# `python scripts/x.py` puts scripts/ on sys.path, not the repo root, so `agpt` is not
# importable without this. Two lines here beats requiring `pip install -e .` before the
# first run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch

from agpt.evaluate import load_checkpoint

# 24-step blue ramp reused as a terminal background: deeper = darker.
DEPTH_BG = [17, 18, 19, 20, 21, 25, 26, 27, 32, 33, 39, 45]


def color_for(depth, n_layer):
    """xterm-256 background for a depth, light (shallow) to dark (deep)."""
    i = min(int(depth / max(n_layer, 1) * len(DEPTH_BG)), len(DEPTH_BG) - 1)
    return DEPTH_BG[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--show-depth", action="store_true",
                    help="colour each generated token by the depth it was given")
    ap.add_argument("--exit-mode", default=None)
    args = ap.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    over = {"exit_mode": args.exit_mode} if args.exit_mode else {}
    model, cfg, _ = load_checkpoint(args.ckpt, device, **over)

    ids = enc.encode_ordinary(args.prompt)
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"model: {cfg.exit_mode}, {cfg.n_layer} layers, "
          f"min depth {cfg.min_depth}\n")

    undecodable = 0
    for s in range(args.samples):
        gen = torch.Generator(device=device).manual_seed(args.seed + s)
        # One token at a time so each forward's reported depth belongs to exactly one
        # generated token. `generate` already re-runs the whole context every step --
        # there is no KV cache in this model -- so this costs nothing extra.
        out, depths = model.generate(prompt, args.tokens, temperature=args.temperature,
                                     top_k=args.top_k, generator=gen,
                                     return_depth=True)
        new_ids = out[0, len(ids):].tolist()

        print(f"--- sample {s + 1} " + "-" * 50)
        print(f"\033[1m{args.prompt}\033[0m", end="")
        for tok, depth in zip(new_ids, [float(d) for d in depths]):
            try:
                piece = enc.decode([tok])
            except Exception:                              # noqa: BLE001
                undecodable += 1
                piece = "�"
            if args.show_depth:
                bg = color_for(depth, cfg.n_layer)
                print(f"\033[48;5;{bg}m\033[38;5;255m{piece}\033[0m", end="")
            else:
                print(piece, end="")
        mean_depth = sum(float(d) for d in depths) / max(len(depths), 1)
        print(f"\n\n[mean depth {mean_depth:.2f} of {cfg.n_layer}]\n")

    if args.show_depth:
        print("depth scale: ", end="")
        for i in range(cfg.n_layer + 1):
            print(f"\033[48;5;{color_for(i, cfg.n_layer)}m\033[38;5;255m{i:3d}\033[0m",
                  end="")
        print("  (layers computed)")
    if undecodable:
        print(f"\n{undecodable} sampled ids had no GPT-2 token (vocab is padded to "
              f"{cfg.vocab_size}); shown as �")


if __name__ == "__main__":
    main()
