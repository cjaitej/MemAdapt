"""Tokenise a corpus into flat `*_tokens.npy` shards.

    python -m agpt.data.prepare --dataset wikitext103
    python -m agpt.data.prepare --dataset fineweb --shards 3
    python -m agpt.data.prepare --dataset local --input-dir ./my_text

WikiText-103 is the default corpus for this project: ~103M training tokens, which is
roughly one epoch at the planned budget, small enough to tokenise in a few minutes and
standard enough that the perplexities mean something to a reader. It also *fits* --
the whole tokenised corpus is about 240 MB on disk, against 1.7 GB for a comparable
slice of FineWeb-Edu.

GPT-2 BPE throughout, which is not a neutral choice: it keeps `hellaswag.py` usable
and it is what the GPT-2 retrofit needs, since a retrofit with a different vocabulary
is not a retrofit. The cost is that the 50257-row output head is ~44% of this model's
forward FLOPs at d=384 (see model/flops.py), which shortens the efficiency axis this
whole project is trying to move along. Kept anyway, and reported on both axes.
"""

import argparse
import os

import numpy as np
import tiktoken
from tqdm import tqdm

# Bytes of text handed to the tokeniser at once. Large enough that the per-call
# overhead vanishes, small enough that the intermediate list of ids stays small.
CHUNK_BYTES = 1 << 20


def get_tokenizer(name="gpt2"):
    enc = tiktoken.get_encoding(name)
    return enc, enc._special_tokens["<|endoftext|>"]


def write_shard(out_dir, split, index, tokens):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}_{index:05d}_tokens.npy")
    np.save(path, np.asarray(tokens, dtype=np.uint16))
    print(f"  wrote {path}: {len(tokens):,} tokens")
    return len(tokens)


def _load_hf(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("the 'datasets' package is required for this source:\n"
                         "    pip install datasets") from e
    return load_dataset(*args, **kwargs)


def iter_wikitext103(split):
    """Text blocks from WikiText-103-raw.

    The dataset is stored one line per row and the articles run together into a single
    continuous stream, which is the standard setup for it -- so the lines are simply
    concatenated in order rather than treated as documents. `--dataset local` and
    `--dataset fineweb` insert an <|endoftext|> between documents; this one does not,
    because there are no document boundaries to mark.
    """
    hf_split = {"train": "train", "val": "validation"}[split]
    ds = _load_hf("Salesforce/wikitext", "wikitext-103-raw-v1", split=hf_split)
    buf = []
    size = 0
    for row in ds:
        line = row["text"]
        if not line:
            continue
        buf.append(line)
        size += len(line)
        if size >= CHUNK_BYTES:
            yield "".join(buf)
            buf, size = [], 0
    if buf:
        yield "".join(buf)


def iter_fineweb(name="sample-10BT"):
    ds = _load_hf("HuggingFaceFW/fineweb-edu", name=name, split="train",
                  streaming=True)
    for row in ds:
        yield row["text"]


def iter_local(input_dir):
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith((".txt", ".md", ".py", ".rst")):
                with open(os.path.join(root, f), "r", encoding="utf-8",
                          errors="ignore") as fh:
                    yield fh.read()


def tokenise_stream(blocks, enc, eot, budget, separate_docs, desc):
    """Encode text blocks until `budget` tokens are collected.

    Returns (tokens, exhausted). `separate_docs` prepends <|endoftext|> to each block,
    which is right when the blocks are whole documents and wrong when they are
    arbitrary slices of one continuous stream.
    """
    out = []
    pbar = tqdm(total=budget, unit="tok", desc=desc)
    exhausted = True
    for text in blocks:
        ids = enc.encode_ordinary(text)
        if separate_docs:
            ids = [eot] + ids
        out.extend(ids)
        pbar.update(len(ids))
        if len(out) >= budget:
            exhausted = False
            break
    pbar.close()
    return out[:budget], exhausted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["wikitext103", "fineweb", "local"],
                    default="wikitext103")
    ap.add_argument("--input-dir", help="source directory for --dataset local")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to data/<dataset>")
    ap.add_argument("--shards", type=int, default=2,
                    help="number of TRAIN shards to write")
    ap.add_argument("--shard-tokens", type=int, default=60_000_000)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--fineweb-config", default="sample-10BT")
    args = ap.parse_args()

    if args.dataset == "local" and not args.input_dir:
        ap.error("--dataset local requires --input-dir")
    out_dir = args.out_dir or os.path.join("data", args.dataset)

    enc, eot = get_tokenizer(args.tokenizer)
    # WikiText-103 ships its own validation split, so val comes from there rather
    # than from the head of train. Holding out a slice of train would work too, but
    # then the number is not comparable to any published WikiText-103 perplexity,
    # which is half the reason for using this corpus.
    separate_docs = args.dataset != "wikitext103"

    if args.dataset == "wikitext103":
        val_iter, train_iter = iter_wikitext103("val"), iter_wikitext103("train")
    else:
        # No held-out split to draw on, so validation comes off the head of the
        # stream and training continues from where it stopped. Same iterator, which
        # is what keeps the two disjoint.
        train_iter = (iter_fineweb(args.fineweb_config) if args.dataset == "fineweb"
                      else iter_local(args.input_dir))
        val_iter = train_iter

    total = 0
    tokens, _ = tokenise_stream(val_iter, enc, eot, args.val_tokens, separate_docs,
                                "val shard")
    total += write_shard(out_dir, "val", 0, tokens)

    for i in range(args.shards):
        tokens, exhausted = tokenise_stream(train_iter, enc, eot, args.shard_tokens,
                                            separate_docs, f"train shard {i}")
        if not tokens:
            print(f"  source exhausted before shard {i}; stopping")
            break
        total += write_shard(out_dir, "train", i, tokens)
        if exhausted:
            print("  source exhausted; no further shards")
            break

    print(f"\n{total:,} tokens in {out_dir}")


if __name__ == "__main__":
    main()
