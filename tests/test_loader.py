"""Loader correctness.

The loader is the quiet failure point of the whole project: if segments are not
consecutive within a document, or `reset_mask` is wrong, the memory bank fills with
unrelated text and every retrieval number is noise -- while the loss curve looks
completely normal.
"""

import numpy as np
import pytest
import torch

from amt.data import DocSegmentLoader, make_random_shard


@pytest.fixture
def shard_dir(tmp_path):
    make_random_shard(tmp_path, split="train", index=0, n_docs=8, doc_len=1024,
                      vocab_size=128, seed=0)
    return str(tmp_path)


def test_shapes_and_targets_are_shifted(shard_dir):
    ld = DocSegmentLoader(shard_dir, B=2, T=64, split="train", shuffle=False)
    x, y, reset = ld.next_batch()
    assert x.shape == (2, 64) and y.shape == (2, 64)
    assert reset.shape == (2,) and reset.dtype == torch.bool
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted by one"


def test_segments_are_consecutive_within_a_document(shard_dir):
    """The property the memory bank depends on: batch n+1 continues batch n."""
    ld = DocSegmentLoader(shard_dir, B=1, T=64, split="train", shuffle=False)
    x1, y1, r1 = ld.next_batch()
    x2, y2, r2 = ld.next_batch()
    assert not r2[0], "should still be inside the first document"
    assert x2[0, 0].item() == y1[0, -1].item(), "segment 2 must continue segment 1"


def test_reset_fires_exactly_on_document_change(shard_dir):
    """doc_len=1024, T=64 -> 15 full segments per doc (the 16th needs T+1 tokens)."""
    ld = DocSegmentLoader(shard_dir, B=1, T=64, split="train", shuffle=False)
    resets = [ld.next_batch()[2][0].item() for _ in range(31)]
    assert resets[0] is True, "first batch of a fresh stream starts a document"
    fired = [i for i, r in enumerate(resets) if r]
    assert fired == [0, 15, 30], f"resets at unexpected positions: {fired}"


def test_no_segment_ever_straddles_two_documents(shard_dir):
    """Documents are random ids, so a straddle is not directly visible in the data --
    check the invariant on the loader's own bookkeeping instead."""
    ld = DocSegmentLoader(shard_dir, B=4, T=64, split="train", shuffle=False)
    for _ in range(40):
        ld.next_batch()
        for b in range(4):
            doc = ld.stream_doc[b]
            assert ld.doc_starts[doc] <= ld.stream_pos[b] <= ld.doc_ends[doc]


def test_streams_are_independent(shard_dir):
    ld = DocSegmentLoader(shard_dir, B=4, T=64, split="train", shuffle=False)
    ld.next_batch()
    assert len(set(ld.stream_doc.tolist())) == 4, "streams must sit on distinct docs"


def test_short_documents_are_rejected_loudly(tmp_path):
    """Silently yielding nothing here would look like a hang, not a config error."""
    make_random_shard(tmp_path, split="train", index=0, n_docs=4, doc_len=32,
                      vocab_size=64, seed=0)
    with pytest.raises(ValueError, match="no document is"):
        DocSegmentLoader(str(tmp_path), B=2, T=128, split="train")


def test_missing_shards_are_rejected_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="no 'train' shards"):
        DocSegmentLoader(str(tmp_path), B=2, T=64, split="train")


def test_too_few_documents_for_batch_size(shard_dir):
    with pytest.raises(ValueError, match="documents for rank"):
        DocSegmentLoader(shard_dir, B=64, T=64, split="train")


def test_ddp_ranks_get_disjoint_documents(shard_dir):
    a = DocSegmentLoader(shard_dir, B=2, T=64, process_rank=0, num_processes=2,
                         shuffle=False)
    b = DocSegmentLoader(shard_dir, B=2, T=64, process_rank=1, num_processes=2,
                         shuffle=False)
    assert not (set(a.doc_order.tolist()) & set(b.doc_order.tolist()))


def test_state_round_trips_for_resume(shard_dir):
    ld = DocSegmentLoader(shard_dir, B=2, T=64, split="train", shuffle=False)
    for _ in range(5):
        ld.next_batch()
    state = ld.state_dict()
    expected = ld.next_batch()

    other = DocSegmentLoader(shard_dir, B=2, T=64, split="train", shuffle=False)
    other.load_state_dict(state)
    got = other.next_batch()

    assert torch.equal(expected[0], got[0])
    assert torch.equal(expected[1], got[1])
    assert torch.equal(expected[2], got[2])
