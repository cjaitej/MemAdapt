"""Document-aware segment loader.

nanoGPT's `DataLoaderLite` chops a flat concatenated token stream at arbitrary
offsets. That is fine for a plain LM and useless here: external memory only means
something if consecutive batches are consecutive *within a document*, and if the bank
is wiped when a stream crosses into a new one. A flat loader silently mixes documents
into the same memory bank, which makes every retrieval metric meaningless while
looking perfectly healthy on the loss curve.

So: B independent streams, each parked on one long document, each yielding the next
consecutive segment, and a `reset_mask` telling the training loop which banks to
clear.
"""

import os

import numpy as np
import torch


class DocSegmentLoader:
    """Yields consecutive segments of long documents, one stream per batch row.

    Parameters
    ----------
    data_dir : directory of `*_tokens.npy` / `*_offsets.npy` shard pairs written by
        `amt/data/prepare.py`
    B, T : batch rows (independent streams) and segment length
    split : "train" or "val"
    process_rank, num_processes : DDP sharding over documents
    shuffle : permute document order (off for val so eval is reproducible)
    """

    def __init__(self, data_dir, B, T, split="train", process_rank=0,
                 num_processes=1, shuffle=True, seed=1337):
        self.data_dir = data_dir
        self.B, self.T = B, T
        self.split = split
        self.rank, self.world = process_rank, num_processes
        self.shuffle = shuffle
        self.seed = seed

        shards = sorted(f for f in os.listdir(data_dir)
                        if f.endswith("_tokens.npy") and split in f)
        if not shards:
            raise FileNotFoundError(
                f"no '{split}' shards in {data_dir}. Run amt/data/prepare.py first."
            )
        self.shard_paths = [os.path.join(data_dir, s) for s in shards]
        self.epoch = 0
        self.reset()

    # -- shard handling ----------------------------------------------------

    def _load_shard(self, i):
        tok_path = self.shard_paths[i]
        off_path = tok_path.replace("_tokens.npy", "_offsets.npy")
        self.tokens = torch.from_numpy(np.load(tok_path).astype(np.int64))
        offsets = np.load(off_path).astype(np.int64)

        # offsets holds document starts plus a final sentinel end, so doc i is
        # [offsets[i], offsets[i+1]).
        starts, ends = offsets[:-1], offsets[1:]
        keep = (ends - starts) >= (self.T + 1)   # need at least one full segment
        self.doc_starts = starts[keep]
        self.doc_ends = ends[keep]
        if len(self.doc_starts) == 0:
            raise ValueError(
                f"{tok_path}: no document is >= T+1 = {self.T + 1} tokens. "
                "Lower T or raise min_doc_tokens in prepare.py."
            )

        order = np.arange(len(self.doc_starts))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch * 1000 + i).shuffle(order)
        # DDP: each rank takes a disjoint slice of documents.
        self.doc_order = order[self.rank::self.world]
        if len(self.doc_order) < self.B:
            raise ValueError(
                f"{tok_path}: {len(self.doc_order)} documents for rank {self.rank} "
                f"but B={self.B} streams. Use a bigger shard or a smaller batch."
            )

    def reset(self):
        self.shard_idx = 0
        self._load_shard(0)
        self.cursor = 0                       # next unassigned doc in doc_order
        self.stream_doc = np.zeros(self.B, dtype=np.int64)
        self.stream_pos = np.zeros(self.B, dtype=np.int64)
        # Streams start "fresh" so the very first batch also reports reset=True. The
        # bank happens to be empty then, but the caller's contract stays exactly one
        # rule -- clear where reset is True -- which stays correct when a loader and
        # bank are reused across evaluation passes.
        self.stream_fresh = np.zeros(self.B, dtype=bool)
        self.docs_consumed = 0
        self.tokens_dropped = 0
        for b in range(self.B):               # _assign updates the state above
            self._assign(b)

    def _next_shard(self):
        self.shard_idx += 1
        if self.shard_idx >= len(self.shard_paths):
            self.shard_idx = 0
            self.epoch += 1
        self._load_shard(self.shard_idx)
        self.cursor = 0

    def _assign(self, b):
        """Park stream b on the next document."""
        if self.cursor >= len(self.doc_order):
            self._next_shard()
        d = self.doc_order[self.cursor]
        self.cursor += 1
        self.stream_doc[b] = d
        self.stream_pos[b] = self.doc_starts[d]
        self.stream_fresh[b] = True
        self.docs_consumed += 1

    # -- iteration ---------------------------------------------------------

    def next_batch(self):
        """Return (x, y, reset_mask).

        reset_mask[b] is True when stream b starts a *new document* with this batch,
        meaning the caller must clear that stream's memory bank BEFORE writing the
        segment. Getting this wrong is silent: the model simply retrieves from an
        unrelated document and the loss barely moves.
        """
        B, T = self.B, self.T
        x = torch.empty(B, T, dtype=torch.long)
        y = torch.empty(B, T, dtype=torch.long)
        reset = torch.zeros(B, dtype=torch.bool)

        for b in range(B):
            end = self.doc_ends[self.stream_doc[b]]
            if self.stream_pos[b] + T + 1 > end:
                # Not enough left for a full segment; the tail is dropped rather than
                # padded, so no batch ever contains cross-document tokens.
                self.tokens_dropped += int(end - self.stream_pos[b])
                self._assign(b)

            reset[b] = bool(self.stream_fresh[b])
            self.stream_fresh[b] = False

            p = int(self.stream_pos[b])
            buf = self.tokens[p:p + T + 1]
            x[b], y[b] = buf[:-1], buf[1:]
            self.stream_pos[b] = p + T

        return x, y, reset

    # -- checkpointing -----------------------------------------------------

    def state_dict(self):
        return {
            "shard_idx": self.shard_idx, "cursor": self.cursor, "epoch": self.epoch,
            "stream_doc": self.stream_doc.copy(), "stream_pos": self.stream_pos.copy(),
            "stream_fresh": self.stream_fresh.copy(),
            "docs_consumed": self.docs_consumed, "tokens_dropped": self.tokens_dropped,
        }

    def load_state_dict(self, s):
        self.epoch = s["epoch"]
        self.shard_idx = s["shard_idx"]
        self._load_shard(self.shard_idx)
        self.cursor = s["cursor"]
        self.stream_doc = np.asarray(s["stream_doc"]).copy()
        self.stream_pos = np.asarray(s["stream_pos"]).copy()
        self.stream_fresh = np.asarray(s["stream_fresh"]).copy()
        self.docs_consumed = s["docs_consumed"]
        self.tokens_dropped = s["tokens_dropped"]

    def stats(self):
        return {"docs_consumed": self.docs_consumed,
                "tokens_dropped": self.tokens_dropped,
                "epoch": self.epoch, "shard": self.shard_idx}
