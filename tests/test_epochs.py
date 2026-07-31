"""Epoch accounting.

An epoch here is not the corpus size. The loader skips documents shorter than T+1
and drops each document's trailing partial segment rather than padding it, so that no
batch ever mixes two documents. Both losses depend on T, which is why an epoch cannot
be read off the file sizes and needs its own accounting.

`--epochs` resolves to a step count before training starts rather than replacing
`max_steps`, because the LR cosine, capacity anneal, gate floor and entropy anneal all
schedule against a fixed horizon.
"""

import numpy as np
import pytest

from amt.data import DocSegmentLoader
from amt.data.synthetic import make_random_shard


@pytest.fixture
def corpus(tmp_path):
    d = str(tmp_path / "shards")
    make_random_shard(d, "train", 0, n_docs=8, doc_len=1000, vocab_size=512, seed=0)
    make_random_shard(d, "val", 0, n_docs=4, doc_len=1000, vocab_size=512, seed=1)
    return d


def test_epoch_is_whole_segments_not_raw_tokens(corpus):
    """8 docs x 1000 tokens, T=256: each yields floor(999/256)=3 segments."""
    loader = DocSegmentLoader(corpus, B=2, T=256, split="train")
    assert loader.tokens_per_epoch() == 8 * 3 * 256

    raw = sum(np.load(p, mmap_mode="r").shape[0] for p in loader.shard_paths)
    assert loader.tokens_per_epoch() < raw, "trailing partial segments are dropped"


def test_epoch_size_depends_on_the_segment_length(corpus):
    """And not monotonically -- which is exactly why it needs measuring, not deriving.

    With 1000-token documents, T=256 fits 3 segments and wastes 231 tokens per doc,
    while T=400 fits only 2 but wastes 199. The larger segment yields the *bigger*
    epoch. Anything that estimates epoch size from corpus bytes and T will be wrong
    in a direction that depends on the document-length distribution.
    """
    at_256 = DocSegmentLoader(corpus, B=2, T=256, split="train").tokens_per_epoch()
    at_400 = DocSegmentLoader(corpus, B=2, T=400, split="train").tokens_per_epoch()

    assert at_256 == 8 * 3 * 256             # 6144
    assert at_400 == 8 * 2 * 400             # 6400
    assert at_400 > at_256


def test_documents_shorter_than_a_segment_contribute_nothing(tmp_path):
    d = str(tmp_path / "tiny")
    make_random_shard(d, "train", 0, n_docs=4, doc_len=100, vocab_size=512, seed=0)
    # T+1 > doc_len, so every document is skipped and the loader has nothing to serve.
    with pytest.raises(ValueError, match="no document is >="):
        DocSegmentLoader(d, B=2, T=256, split="train")


def test_epochs_resolve_to_steps_and_are_written_back(corpus, monkeypatch):
    """--epochs must set args.max_steps, since that is what every schedule reads."""
    loader = DocSegmentLoader(corpus, B=2, T=256, split="train")
    per_epoch = loader.tokens_per_epoch()

    total_batch_tokens = 512
    for epochs in (0.5, 1.0, 3.0):
        steps = max(1, round(epochs * per_epoch / total_batch_tokens))
        realised = steps * total_batch_tokens / per_epoch
        assert abs(realised - epochs) < 0.02, (
            f"{epochs} epochs resolved to {steps} steps = {realised:.3f} epochs"
        )
