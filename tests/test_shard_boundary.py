"""Crossing a shard boundary must not corrupt in-flight streams.

Only one shard's tokens are resident, so advancing to the next one replaces
`tokens`, `doc_starts`, `doc_ends` and `doc_order` wholesale. Reassigning only the
stream that triggered the roll left the other B-1 holding indices and offsets into
the shard that had just been discarded.

The failure had two faces. When the incoming shard held fewer documents it raised
`IndexError: index 22004 is out of bounds for axis 0 with size 21837`. When it held
more, the stale index was silently *valid* -- a different document entirely, read at
an offset computed for the old shard -- and training continued on wrong text with no
error at all. Both are covered here, the silent one first, because it is the one that
quietly invalidates results.
"""

import numpy as np
import pytest
import torch

from amt.data import DocSegmentLoader


def write_shard(d, split, index, n_docs, doc_len, lo):
    """A shard whose token values identify it: shard N uses [lo, lo+50)."""
    d.mkdir(parents=True, exist_ok=True)
    docs = [np.full(doc_len, lo + (i % 50), dtype=np.uint16) for i in range(n_docs)]
    tokens = np.concatenate(docs)
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in docs])]).astype(np.int64)
    np.save(d / f"{split}_{index:05d}_tokens.npy", tokens)
    np.save(d / f"{split}_{index:05d}_offsets.npy", offsets)


@pytest.fixture
def shards(tmp_path):
    """Shard 1 is deliberately SMALLER than shard 0 -- that is the IndexError case."""
    d = tmp_path / "shards"
    write_shard(d, "train", 0, n_docs=12, doc_len=600, lo=1000)
    write_shard(d, "train", 1, n_docs=6, doc_len=600, lo=2000)
    return str(d)


def band(t):
    """Which shard a token value came from."""
    return 1000 if t < 2000 else 2000


def test_no_sequence_mixes_tokens_from_two_shards(shards):
    """The silent corruption: a stale index reads another shard's tokens.

    Every row of every batch must come from exactly one document, and therefore from
    exactly one shard. Rows within a batch may differ -- streams roll at different
    times -- but a single row must never straddle.
    """
    loader = DocSegmentLoader(shards, B=4, T=128, split="train", shuffle=False)
    for _ in range(60):
        x, y, _ = loader.next_batch()
        for row in torch.cat([x, y], dim=1):
            bands = {band(int(t)) for t in row}
            assert len(bands) == 1, f"row spans shards {bands}"


def test_shrinking_shard_does_not_raise(shards):
    """The loud case, reproduced: 12 documents then 6, with 4 live streams."""
    loader = DocSegmentLoader(shards, B=4, T=128, split="train", shuffle=False)
    for _ in range(80):
        loader.next_batch()                 # IndexError before the fix
    assert loader.epoch >= 1, "80 batches should have wrapped the whole corpus"


def test_every_stream_resets_when_the_shard_rolls(shards):
    """A stale bank is the R6 leakage risk: streams get new documents, so the caller
    must be told to clear memory for all of them."""
    loader = DocSegmentLoader(shards, B=4, T=128, split="train", shuffle=False)
    seen_full_reset = False
    start_shard = loader.shard_idx
    for _ in range(60):
        _, _, reset = loader.next_batch()
        if loader.shard_idx != start_shard:
            # The batch that rolled the shard re-parked every stream.
            seen_full_reset = seen_full_reset or bool(reset.all())
            start_shard = loader.shard_idx
    assert seen_full_reset, "a shard roll must reset every stream"


def test_dropped_tails_are_counted(shards):
    """Re-parking discards partially-read documents; telemetry should admit it."""
    loader = DocSegmentLoader(shards, B=4, T=128, split="train", shuffle=False)
    for _ in range(60):
        loader.next_batch()
    assert loader.tokens_dropped > 0
