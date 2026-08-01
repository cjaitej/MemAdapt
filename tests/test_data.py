"""The loader: coverage, shard rotation and resumability."""

import numpy as np
import pytest
import torch

from agpt.data import SegmentLoader, make_random_shard
from agpt.data import loaders
from agpt.data.loaders import resolve_data_dir


@pytest.fixture
def shards(tmp_path):
    d = tmp_path / "shards"
    make_random_shard(str(d), "train", 0, n_tokens=4096, vocab_size=100, seed=0)
    make_random_shard(str(d), "train", 1, n_tokens=4096, vocab_size=100, seed=1)
    make_random_shard(str(d), "val", 0, n_tokens=2048, vocab_size=100, seed=2)
    return str(d)


def test_x_and_y_are_offset_by_one(shards):
    loader = SegmentLoader(shards, B=2, T=16, split="train")
    x, y = loader.next_batch()
    assert x.shape == y.shape == (2, 16)
    assert torch.equal(x[0, 1:], y[0, :-1]), "y must be x shifted by one"


def test_batches_are_contiguous_and_do_not_repeat(shards):
    loader = SegmentLoader(shards, B=1, T=16, split="train")
    first, _ = loader.next_batch()
    second, _ = loader.next_batch()
    tokens = np.load(f"{shards}/train_00000_tokens.npy").astype(np.int64)
    assert torch.equal(first[0], torch.from_numpy(tokens[:16]))
    assert torch.equal(second[0], torch.from_numpy(tokens[16:32]))


def test_shard_rotation_and_epoch_counting(shards):
    loader = SegmentLoader(shards, B=2, T=16, split="train")
    seen = set()
    for _ in range(400):                     # far more than one shard holds
        loader.next_batch()
        seen.add(loader.shard_idx)
    assert seen == {0, 1}, "the loader never moved off the first shard"
    assert loader.epoch >= 1


def test_resume_continues_where_it_stopped(shards):
    a = SegmentLoader(shards, B=2, T=16, split="train")
    for _ in range(37):
        a.next_batch()
    state = a.state_dict()
    expected, _ = a.next_batch()

    b = SegmentLoader(shards, B=2, T=16, split="train")
    b.load_state_dict(state)
    got, _ = b.next_batch()
    assert torch.equal(expected, got), "a resumed loader served a different batch"


def test_reset_returns_to_the_start(shards):
    loader = SegmentLoader(shards, B=2, T=16, split="val")
    first, _ = loader.next_batch()
    for _ in range(5):
        loader.next_batch()
    loader.reset()
    again, _ = loader.next_batch()
    assert torch.equal(first, again)


def test_tokens_per_epoch_is_whole_batches(shards):
    loader = SegmentLoader(shards, B=2, T=16, split="train")
    n = loader.tokens_per_epoch()
    assert n % (2 * 16) == 0
    assert 0 < n <= 8192


def test_missing_split_says_so(shards):
    with pytest.raises(FileNotFoundError, match="no 'test' shards"):
        SegmentLoader(shards, B=2, T=16, split="test")


def test_batch_too_big_for_the_shard_says_so(shards):
    with pytest.raises(ValueError, match="less than one batch"):
        SegmentLoader(shards, B=64, T=512, split="train")


def test_resolve_data_dir_passes_through_an_existing_dir(shards):
    assert resolve_data_dir(shards) == shards


# The three branches below are parametrised on `find_shard_dirs` rather than on the
# real filesystem. The earlier version of this test just pointed at a missing path and
# expected a raise, which passed on a laptop and failed on Kaggle -- where the attached
# dataset IS a discoverable shard directory, so `resolve_data_dir` correctly redirected
# to it instead of raising. A test that depends on the machine having no corpus on it
# is testing the machine.

def test_resolve_data_dir_raises_when_nothing_is_findable(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "find_shard_dirs", lambda *a, **k: [])
    with pytest.raises(FileNotFoundError, match="no token shards"):
        resolve_data_dir(str(tmp_path / "nope"))


def test_resolve_data_dir_redirects_to_the_only_candidate(tmp_path, monkeypatch,
                                                          shards):
    """One usable corpus on the machine: take it, and say so loudly.

    This is the branch that saves a Kaggle run whose dataset mounted under a slug-based
    path nobody could have typed from memory.
    """
    monkeypatch.setattr(loaders, "find_shard_dirs", lambda *a, **k: [shards])
    assert resolve_data_dir(str(tmp_path / "nope")) == shards


def test_resolve_data_dir_refuses_to_guess_between_several(tmp_path, monkeypatch):
    """Several candidates: refuse and list them.

    Silently picking one would change what an experiment trained on, which is the kind
    of thing that surfaces as an unreproducible result rather than as an error.
    """
    monkeypatch.setattr(loaders, "find_shard_dirs",
                        lambda *a, **k: ["/corpus/a", "/corpus/b"])
    with pytest.raises(FileNotFoundError, match="/corpus/a"):
        resolve_data_dir(str(tmp_path / "nope"))
