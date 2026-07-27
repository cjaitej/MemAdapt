"""Synthetic corpora: shard fixtures for tests, and the key-value recall probe.

The recall probe exists because perplexity is an indirect way to ask "did retrieval
work". It plants a fact early in a long document and queries it thousands of tokens
later, so retrieval accuracy is read off directly instead of inferred from a
perplexity delta of a few hundredths.

Note the probe measures *in-context* recall, which a 40M model will not do zero-shot.
Mix a small fraction of these documents into training (`--mix-rate` in train.py) and
evaluate on held-out key/value pairs at held-out distances.
"""

import os

import numpy as np


def write_shard(out_dir, split, index, docs, answer_mask=None):
    """Write a (tokens, offsets) shard in the layout DocSegmentLoader expects."""
    tokens = np.concatenate(docs).astype(np.uint16)
    lengths = np.array([len(d) for d in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{split}_{index:05d}")
    np.save(f"{stem}_tokens.npy", tokens)
    np.save(f"{stem}_offsets.npy", offsets)
    if answer_mask is not None:
        np.save(f"{stem}_answers.npy", np.concatenate(answer_mask).astype(np.uint8))
    return f"{stem}_tokens.npy"


def make_random_shard(out_dir, split="train", index=0, n_docs=8, doc_len=4096,
                      vocab_size=128, seed=0):
    """Structureless documents. For exercising loader plumbing, not for learning."""
    rng = np.random.default_rng(seed)
    docs = [rng.integers(0, vocab_size, size=doc_len, dtype=np.uint16)
            for _ in range(n_docs)]
    return write_shard(out_dir, split, index, docs)


def make_recall_shard(out_dir, split="val", index=0, n_docs=64, doc_len=4096,
                      n_facts=32, vocab_size=1024, seed=0):
    """Key-value recall documents.

    Layout per document, in a toy token vocabulary:

        [KEY k][VAL v]  x n_facts        <- the facts, stated up front
        [filler ...]                     <- pushes the facts out of local attention
        [QUERY][KEY k][VAL v]            <- the query; the model must emit VAL v

    Reserved ids: 0 = filler-safe padding, 1 = QUERY marker. Keys occupy
    [2, 2+n_keys), values [2+n_keys, vocab_size).

    Returns (path, metadata) where metadata carries the answer positions so recall
    accuracy can be computed exactly rather than approximated by loss.
    """
    rng = np.random.default_rng(seed)
    QUERY = 1
    n_keys = max(n_facts * 4, 64)
    key_lo, key_hi = 2, 2 + n_keys
    val_lo, val_hi = key_hi, vocab_size
    assert val_hi > val_lo, "vocab_size too small for the requested n_facts"

    docs, masks, meta = [], [], []
    for d in range(n_docs):
        keys = rng.choice(np.arange(key_lo, key_hi), size=n_facts, replace=False)
        vals = rng.integers(val_lo, val_hi, size=n_facts)

        doc = []
        for k, v in zip(keys, vals):
            doc.extend([int(k), int(v)])
        n_fact_tokens = len(doc)

        # Filler drawn only from value ids, so it can never be mistaken for a key.
        target = rng.integers(0, n_facts)
        query_len = 3
        n_filler = doc_len - n_fact_tokens - query_len
        if n_filler < 1:
            raise ValueError("doc_len too small for n_facts; raise doc_len")
        doc.extend(rng.integers(val_lo, val_hi, size=n_filler).tolist())
        doc.extend([QUERY, int(keys[target]), int(vals[target])])

        arr = np.array(doc, dtype=np.uint16)
        mask = np.zeros(len(arr), dtype=np.uint8)
        mask[-1] = 1                      # the single token that must be recalled
        docs.append(arr)
        masks.append(mask)
        meta.append({
            "answer_token": int(vals[target]),
            "distance": int(len(arr) - 1 - (2 * target + 1)),
        })

    path = write_shard(out_dir, split, index, docs, answer_mask=masks)
    return path, meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/recall_probe")
    ap.add_argument("--n-docs", type=int, default=256)
    ap.add_argument("--doc-len", type=int, default=4096)
    ap.add_argument("--n-facts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    path, meta = make_recall_shard(a.out_dir, n_docs=a.n_docs, doc_len=a.doc_len,
                                   n_facts=a.n_facts, seed=a.seed)
    d = [m["distance"] for m in meta]
    print(f"wrote {path}")
    print(f"recall distances: min={min(d)} median={int(np.median(d))} max={max(d)}")
