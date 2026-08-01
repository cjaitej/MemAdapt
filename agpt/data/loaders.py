"""Flat segment loader over tokenised shards.

Consecutive `T`-token segments from a concatenated token stream, `B` of them per
batch, rolling through shards and wrapping at the end of the corpus.

This used to be a document-aware loader that parked each batch row on its own long
document and reported when a row crossed a document boundary. All of that existed to
serve an external memory bank, which had to be cleared whenever a stream changed
document. There is no memory bank any more, and an early-exit decision depends only on
the token in front of it, so the machinery has no reader left. A flat stream is what
nanoGPT uses and it is enough.
"""

import os

import numpy as np
import torch


def find_shard_dirs(roots=("data", "/kaggle/input", "/kaggle/working"), max_depth=7):
    """Directories that actually contain `*_tokens.npy`, searched a few levels deep.

    Kaggle mounts a dataset at a path built from the dataset's slug and its internal
    folder layout, which almost never matches what you typed from memory. The result
    is a run that dies two seconds in, after the session has already been started.
    """
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        root_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count(os.sep) - root_depth >= max_depth:
                dirnames[:] = []
                continue
            if any(f.endswith("_tokens.npy") for f in filenames):
                found.append(dirpath)
                dirnames[:] = []          # do not descend into a shard directory
    return sorted(set(found))


def resolve_data_dir(data_dir):
    """Return a directory holding token shards, or raise saying where to look.

    A wrong `--data-dir` used to surface as `FileNotFoundError` from `os.listdir`
    inside the loader -- accurate, and useless, because it neither says what was
    expected nor what is available. When exactly one usable directory exists, this
    takes it and says so loudly; when several do, it refuses and lists them rather
    than guessing which corpus an experiment was supposed to use.

    Only a directory that does not EXIST is redirected. One that exists but holds no
    shards is a different failure -- prep that did not finish, or the wrong split --
    and substituting some other corpus there would quietly change what an experiment
    trained on. That case is left to the caller's own check.
    """
    if os.path.isdir(data_dir):
        return data_dir

    candidates = find_shard_dirs()
    if len(candidates) == 1:
        print(f"data dir    : {data_dir!r} not found; using the only shard directory "
              f"on this machine:\n              {candidates[0]}", flush=True)
        return candidates[0]

    detail = ("\n".join(f"  {c}" for c in candidates) if candidates
              else "  (none found under data/, /kaggle/input, /kaggle/working)")
    raise FileNotFoundError(
        f"no token shards in {data_dir!r}.\n"
        f"Directories that do contain *_tokens.npy:\n{detail}\n"
        "Pass one of those as --data-dir, or run `python -m agpt.data.prepare`."
    )


class SegmentLoader:
    """Yields (x, y) pairs of consecutive T-token segments.

    Parameters
    ----------
    data_dir : directory of `*_tokens.npy` shards written by `agpt/data/prepare.py`
    B, T : batch rows and segment length
    split : "train" or "val"
    process_rank, num_processes : DDP sharding, by stride within a shard
    """

    def __init__(self, data_dir, B, T, split="train", process_rank=0,
                 num_processes=1):
        data_dir = resolve_data_dir(data_dir)
        self.data_dir = data_dir
        self.B, self.T = B, T
        self.split = split
        self.rank, self.world = process_rank, num_processes

        shards = sorted(f for f in os.listdir(data_dir)
                        if f.endswith("_tokens.npy") and split in f)
        if not shards:
            raise FileNotFoundError(
                f"no '{split}' shards in {data_dir}. Run agpt/data/prepare.py first.")
        self.shard_paths = [os.path.join(data_dir, s) for s in shards]
        self.epoch = 0
        self.reset()

    def _load_shard(self, i):
        self.tokens = torch.from_numpy(
            np.load(self.shard_paths[i]).astype(np.int64))
        stride = self.B * self.T * self.world
        if len(self.tokens) < stride + 1:
            raise ValueError(
                f"{self.shard_paths[i]}: {len(self.tokens):,} tokens is less than one "
                f"batch of {stride:,}. Use a bigger shard or a smaller B*T.")

    def reset(self):
        self.shard_idx = 0
        self._load_shard(0)
        self.pos = self.B * self.T * self.rank

    def tokens_per_epoch(self):
        """Tokens in one pass over every shard, rounded down to whole batches.

        Counts the whole corpus, not this rank's share; under DDP each rank sees
        roughly 1/world of it. Reads only the array headers, not the arrays.
        """
        stride = self.B * self.T * self.world
        total = 0
        for p in self.shard_paths:
            n = int(np.load(p, mmap_mode="r").shape[0])
            total += ((n - 1) // stride) * stride
        return total

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.pos:self.pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.pos += B * T * self.world
        if self.pos + (B * T * self.world + 1) > len(self.tokens):
            self.shard_idx += 1
            if self.shard_idx >= len(self.shard_paths):
                self.shard_idx = 0
                self.epoch += 1
            self._load_shard(self.shard_idx)
            self.pos = B * T * self.rank
        return x, y

    # -- checkpointing -----------------------------------------------------

    def state_dict(self):
        return {"shard_idx": self.shard_idx, "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, s):
        self.epoch = s["epoch"]
        self.shard_idx = s["shard_idx"]
        self._load_shard(self.shard_idx)
        self.pos = s["pos"]

    def stats(self):
        return {"epoch": self.epoch, "shard": self.shard_idx, "pos": self.pos}
