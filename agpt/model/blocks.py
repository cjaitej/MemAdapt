"""Transformer blocks.

`CausalSelfAttention` and `MLP` keep nanoGPT's parameter names -- which are also
HuggingFace's names for GPT-2 -- so `retrofit.py` is a rename and not a port.

Two things are added on top of a stock block:

* `delta(x)` returns the block's contribution to the residual stream with `x`
  excluded, so the model can gate it. Gating the whole stream instead would attenuate
  the residual path and destabilise training.
* `delta_compact(...)` computes that same contribution for a *subset* of positions
  while still attending over the full context. This is where the wall-clock saving
  actually comes from, and the reason it is a separate method is spelled out below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def _split_heads(self, t):
        B, T, C = t.size()
        return t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    def qkv(self, x):
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        return self._split_heads(q), self._split_heads(k), self._split_heads(v)

    def kv_only(self, x):
        """Keys and values without paying for the query projection.

        `c_attn` is one fused Linear producing [q | k | v], so computing all three
        and throwing q away costs 6d^2 per token when 4d^2 would do. In the compact
        path every token needs k and v (an exited token is still attended TO) but only
        the survivors need q, so the fused projection is exactly the wrong shape and
        the weight is sliced instead. It moves the layer's floor from 6d^2 to 4d^2 --
        a sixth of what routing can save at all.
        """
        d = self.n_embd
        w = self.c_attn.weight[d:]                      # (2d, d): the k and v rows
        b = self.c_attn.bias[d:] if self.c_attn.bias is not None else None
        k, v = F.linear(x, w, b).split(d, dim=2)
        return self._split_heads(k), self._split_heads(v)

    def q_only(self, x):
        d = self.n_embd
        w = self.c_attn.weight[:d]
        b = self.c_attn.bias[:d] if self.c_attn.bias is not None else None
        return self._split_heads(F.linear(x, w, b))

    def merge_heads(self, y):
        B, nh, T, hs = y.size()
        return y.transpose(1, 2).contiguous().view(B, T, nh * hs)

    def forward(self, x):
        q, k, v = self.qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(self.merge_heads(y))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def delta(self, x):
        """The block's contribution to the residual stream, x excluded."""
        h = x + self.attn(self.ln_1(x))
        h = h + self.mlp(self.ln_2(h))
        return h - x

    # -- the compact path --------------------------------------------------

    def delta_compact(self, x_full, idx, valid, drop_exited=False):
        """`delta` for the positions in `idx` only, attending over the full context.

        Parameters
        ----------
        x_full : (B, T, C) the whole residual stream. Rows belonging to tokens that
            have already exited still hold their frozen state -- that is the point:
            a token that stopped computing is still part of the context every later
            token reads.
        idx : (B, k) positions to compute, ascending. `k` is bucketed, so the tail of
            this may be padding.
        valid : (B, k) bool, False on the padded tail.
        drop_exited : remove exited tokens from the attention entirely instead of
            re-projecting their frozen state (AdaptiveGPTConfig.exited_as_keys).

        Returns (B, k, C).

        Why this is exact
        -----------------
        With `drop_exited=False` this computes, for each selected position, precisely
        what `delta(x_full)` computes for that row -- the keys and values are derived
        from the same `x_full`, and the causal mask is built from the tokens' original
        positions rather than their positions within the gathered subset. So the fast
        path and the reference path are the same function, and
        `tests/test_exit.py::test_compact_matches_dense` holds them to it. A fast path
        that is merely *similar* to the reference is not a speedup, it is a second
        model that nobody evaluated.
        """
        B, T, C = x_full.shape
        k = idx.size(1)

        h_full = self.ln_1(x_full)
        h_sel = gather_tokens(h_full, idx)                       # (B, k, C)
        q = self.attn.q_only(h_sel)                              # (B, nh, k, hd)

        pos_q = idx.unsqueeze(-1)                                # (B, k, 1)
        if drop_exited:
            kk, vv = self.attn.kv_only(h_sel)                    # keys from survivors
            pos_k = idx.unsqueeze(1)                             # (B, 1, k)
            mask = (pos_k <= pos_q) & valid.unsqueeze(1)
            # A padded query row would otherwise be all-masked, and softmax over an
            # all-masked row is NaN -- which then survives being multiplied by zero
            # and poisons the whole batch from a value that was going to be discarded.
            eye = torch.eye(k, dtype=torch.bool, device=idx.device).unsqueeze(0)
            mask = mask | eye
        else:
            kk, vv = self.attn.kv_only(h_full)                   # every token is a key
            pos_k = torch.arange(T, device=idx.device).view(1, 1, T)
            mask = pos_k <= pos_q                                # (B, k, T)

        y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask.unsqueeze(1))
        y = self.attn.c_proj(self.attn.merge_heads(y))           # (B, k, C)

        x_sel = gather_tokens(x_full, idx)
        h = x_sel + y
        h = h + self.mlp(self.ln_2(h))
        return h - x_sel


def gather_tokens(x, idx):
    """(B, T, C) -> (B, k, C) at the given positions."""
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def scatter_add_tokens(x, idx, y):
    """Residual-add (B, k, C) back into (B, T, C) at the given positions.

    `scatter_add`, not `scatter`: the padded tail of `idx` repeats a real position
    (position 0), so an overwriting scatter would clobber that row with the garbage
    computed for a padding slot. The caller zeroes the padded rows of `y`, and adding
    zero to a duplicated index is exactly a no-op -- which makes the padding invisible
    rather than merely unlikely to matter.
    """
    return x.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, y.size(-1)), y)
