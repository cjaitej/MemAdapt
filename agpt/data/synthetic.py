"""Random token shards, for checking plumbing without downloading a corpus.

    python -m agpt.train --variant adaptive --synthetic --max-steps 20

The loss will not drop below chance and is not supposed to. What this exercises is
everything around the loss: shard loading, the gating arithmetic, the three-stage
schedule, checkpoint round-trips and the compact inference path. Those are where the
bugs live, and none of them need real text to surface.
"""

import os

import numpy as np


def make_random_shard(out_dir, split="train", index=0, n_tokens=1 << 18,
                      vocab_size=50257, seed=0):
    """Write one shard of uniformly random token ids. Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    path = os.path.join(out_dir, f"{split}_{index:05d}_tokens.npy")
    np.save(path, tokens)
    return path
