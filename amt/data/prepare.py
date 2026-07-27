"""Tokenise a corpus into document-aware shards.

Differs from the repo's `fineweb.py` in one way that matters: alongside the token
stream it writes a `_offsets.npy` index of document boundaries, so the loader can
keep each memory stream inside a single document.

Documents shorter than `--min-doc-tokens` are dropped. Most of FineWeb is a few
hundred tokens long, and a document that fits in one segment can never exercise
memory -- keeping them would dilute every retrieval metric with cases where the bank
is empty by construction.

    python -m amt.data.prepare --dataset fineweb --shards 3
    python -m amt.data.prepare --dataset local --input-dir ./my_text
"""

import argparse
import os

import numpy as np
import tiktoken
from tqdm import tqdm


def get_tokenizer(name="gpt2"):
    """GPT-2 BPE by default: keeps `hellaswag.py` usable and leaves the door open to
    the GPT-2 retrofit demo, which needs a matching vocabulary."""
    enc = tiktoken.get_encoding(name)
    return enc, enc._special_tokens["<|endoftext|>"]


def write_shard(out_dir, split, index, docs):
    """Write one (tokens, offsets) pair.

    `offsets` has len(docs)+1 entries -- starts plus a final sentinel -- so document i
    is exactly [offsets[i], offsets[i+1]).
    """
    tokens = np.concatenate(docs).astype(np.uint16)
    lengths = np.array([len(d) for d in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{split}_{index:05d}")
    np.save(f"{stem}_tokens.npy", tokens)
    np.save(f"{stem}_offsets.npy", offsets)
    print(f"  wrote {stem}: {len(docs):,} docs, {len(tokens):,} tokens")
    return len(tokens)


def iter_fineweb(name="sample-10BT"):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "the 'datasets' package is required for --dataset fineweb:\n"
            "    pip install datasets"
        ) from e
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=name, split="train",
                      streaming=True)
    for row in ds:
        yield row["text"]


def iter_local(input_dir):
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith((".txt", ".md", ".py", ".rst")):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    yield fh.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["fineweb", "local"], default="fineweb")
    ap.add_argument("--input-dir", help="source directory for --dataset local")
    ap.add_argument("--out-dir", default="data/fineweb_edu_docs")
    ap.add_argument("--shards", type=int, default=3,
                    help="number of TRAIN shards; ~7h of training per shard on a 3050")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--min-doc-tokens", type=int, default=2048,
                    help="drop shorter docs; must exceed block_size for memory to matter")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    if args.dataset == "local" and not args.input_dir:
        ap.error("--dataset local requires --input-dir")

    enc, eot = get_tokenizer(args.tokenizer)
    source = iter_local(args.input_dir) if args.dataset == "local" else iter_fineweb()

    # The first shard is validation, the rest training -- matching fineweb.py.
    budgets = [("val", args.val_tokens)] + [("train", args.shard_tokens)] * args.shards

    it = iter(source)
    kept = dropped = 0
    for shard_i, (split, budget) in enumerate(budgets):
        docs, n = [], 0
        pbar = tqdm(total=budget, unit="tok", desc=f"{split} shard {shard_i}")
        while n < budget:
            try:
                text = next(it)
            except StopIteration:
                break
            ids = [eot] + enc.encode_ordinary(text)
            if len(ids) < args.min_doc_tokens:
                dropped += 1
                continue
            kept += 1
            docs.append(np.array(ids, dtype=np.uint16))
            n += len(ids)
            pbar.update(len(ids))
        pbar.close()
        if not docs:
            print(f"  no documents left for shard {shard_i}; stopping early")
            break
        idx = 0 if split == "val" else shard_i - 1
        write_shard(args.out_dir, split, idx, docs)

    total = kept + dropped
    if total:
        print(f"\nkept {kept:,} docs, dropped {dropped:,} shorter than "
              f"{args.min_doc_tokens} tokens ({dropped / total:.1%} of the stream)")


if __name__ == "__main__":
    main()
