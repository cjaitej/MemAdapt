"""Transformer blocks: dense, memory-augmented, and adaptive.

`CausalSelfAttention` and `MLP` are lifted from `train_gpt2.py` with two changes:
attention can hand back its keys/values (so the memory layer can bank them), and
blocks can report their residual *delta* (so the adaptive wrapper can scale it by the
router weight before scattering it back).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routers import TopKTokenRouter, gather_tokens, scatter_add_tokens


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def qkv(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        return (q.view(shape).transpose(1, 2),
                k.view(shape).transpose(1, 2),
                v.view(shape).transpose(1, 2))          # each (B, nh, T, hs)

    def merge_heads(self, y):
        B, nh, T, hs = y.size()
        return y.transpose(1, 2).contiguous().view(B, T, nh * hs)

    def forward(self, x, return_kv=False, key_mask=None):
        q, k, v = self.qkv(x)
        y = attend(q, k, v, key_mask)
        y = self.c_proj(self.merge_heads(y))
        return (y, k, v) if return_kv else y


def attend(q, k, v, key_mask=None):
    """Causal attention, optionally restricted to a subset of keys.

    `key_mask` is (B, T) bool: which positions may be attended *to*. The routed path
    needs it so that causal-mode evaluation reproduces training semantics, where a
    block only ever sees the tokens its router selected.

    The diagonal is forced visible. Without it a token whose own position is masked
    out has an all-masked row, softmax returns NaN, and the NaN survives being
    multiplied by zero later -- poisoning the whole batch from a value that was
    supposed to be discarded.
    """
    if key_mask is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    B, nh, T, _ = q.shape
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    mask = causal.view(1, 1, T, T) & key_mask.view(B, 1, 1, T)
    eye = torch.eye(T, dtype=torch.bool, device=q.device).view(1, 1, T, T)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask | eye)


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

    def delta(self, x, key_mask=None):
        """The block's contribution to the residual stream, x excluded.

        The adaptive wrapper needs this so it can scale by the router weight before
        adding back -- scaling the *whole* stream would attenuate the residual path
        and destabilise training.
        """
        h = x + self.attn(self.ln_1(x), key_mask=key_mask)
        h = h + self.mlp(self.ln_2(h))
        return h - x


class MemoryBlock(nn.Module):
    """The memory layer: local causal attention fused with a gated kNN read.

    Runs for every token (it sits in the always-on part of the stack), but only the
    tokens chosen by `mem_router` actually query the bank -- that routing is what
    makes retrieval affordable, since the kNN scan is linear in bank size.

    Emits `conf`, the top-1 retrieval similarity, which is the wire that couples the
    memory decision to the depth decisions downstream (contribution C1).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        self.mem_router = (
            TopKTokenRouter(config.n_embd, config.mem_capacity, name="mem_router")
            if config.route_memory else None
        )

        # Per-head fusion gate, conditioned on the token and on retrieval quality.
        self.w_gate = nn.Linear(config.n_embd + 1, config.n_head, bias=True)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.constant_(self.w_gate.bias, config.mem_gate_init)

        # Learned softmax temperature over retrieved similarities (cosine sims live in
        # [-1, 1], so an unscaled softmax over them is nearly uniform).
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        # Lower bound on the gate, held above zero during warmup and released by the
        # trainer (risk R2). The gate starts near-shut so the model does not lean on
        # an untrained memory -- but a shut gate receives almost no gradient, so
        # without a floor it can stay shut permanently and memory is never learned.
        #
        # A buffer, not a float: torch.compile specialises on Python scalars read
        # inside forward, so an annealed float here triggers a full recompile EVERY
        # step it changes. At ~4 minutes per compile on a laptop GPU that made short
        # runs impossible. Written in place with .fill_() so the tensor identity is
        # stable and no guard is invalidated.
        self.register_buffer("gate_floor", torch.zeros(()), persistent=False)

    def forward(self, x, bank=None, causal=False):
        B, T, C = x.shape
        h = self.ln_1(x)
        q, k, v = self.attn.qkv(h)
        y_local = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,nh,T,hs)

        conf_full = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        route_out = None

        if bank is not None and self.config.use_memory:
            if self.mem_router is not None:
                route_out = self.mem_router(h, causal=causal)
                idx = route_out["idx"]                                    # (B, kq) or None
                # In causal mode the selected count varies, so every token reads and
                # the gate is forced shut on the ones the router rejected. Same
                # result as gathering, at dense retrieval cost -- see
                # AdaptiveBlock._masked_forward for why that trade is deliberate.
                q_sel = self._gather_heads(q, idx) if idx is not None else q
            else:
                idx = None
                q_sel = q

            y_mem, conf = bank.read(
                q_sel,
                n_neighbors=self.config.n_neighbors,
                chunk_size=self.config.mem_query_chunk,
                temperature=self.log_temp.exp().clamp(0.02, 10.0),
            )

            h_sel = gather_tokens(h, idx) if idx is not None else h       # (B, kq, C)
            gate = torch.sigmoid(
                self.w_gate(torch.cat([h_sel, conf.unsqueeze(-1).to(h.dtype)], dim=-1))
            )                                                             # (B, kq, nh)
            # Lift rather than clamp: keeps the gate differentiable everywhere, where
            # clamp would zero the gradient for every gate under the floor. Applied
            # unconditionally -- a `if floor > 0` branch on a tensor value is a graph
            # break, and at floor=0 the expression is exactly the identity anyway.
            floor = self.gate_floor.to(gate.dtype)
            gate = floor + (1.0 - floor) * gate
            if route_out is not None and idx is None:
                gate = gate * route_out["mask"].unsqueeze(-1).to(gate.dtype)
            gate = gate.transpose(1, 2).unsqueeze(-1)                     # (B, nh, kq, 1)

            y_local_sel = self._gather_heads(y_local, idx) if idx is not None else y_local
            fused = gate * y_mem.to(y_local.dtype) + (1 - gate) * y_local_sel
            self._last_gate = gate.mean().detach()

            if idx is not None:
                y_local = self._scatter_heads(y_local, idx, fused)
                conf_full = conf_full.scatter(1, idx, conf.to(conf_full.dtype))
            else:
                y_local = fused
                conf_full = conf.to(conf_full.dtype)
                if route_out is not None:
                    conf_full = conf_full * route_out["mask"].to(conf_full.dtype)

        y = self.attn.c_proj(self.attn.merge_heads(y_local))
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, conf_full, route_out, (k, v)

    # -- head-major gather/scatter helpers (B, nh, T, hs) -------------------

    @staticmethod
    def _gather_heads(t, idx):
        B, nh, T, hs = t.shape
        i = idx.view(B, 1, -1, 1).expand(B, nh, idx.size(1), hs)
        return t.gather(2, i)

    @staticmethod
    def _scatter_heads(t, idx, src):
        B, nh, T, hs = t.shape
        i = idx.view(B, 1, -1, 1).expand(B, nh, idx.size(1), hs)
        return t.scatter(2, i, src.to(t.dtype))


class AdaptiveBlock(nn.Module):
    """A Block that only runs on the tokens its router selects.

    gather -> dense block on k tokens -> scale by router weight -> scatter-add back.
    Because k = ceil(capacity * T) is fixed, every tensor here has a static shape.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block = Block(config)
        cond_dim = 1 if config.couple else 0    # the retrieval-confidence wire
        self.router = TopKTokenRouter(
            config.n_embd,
            capacity=config.depth_capacity,
            cond_dim=cond_dim,
            random_route=config.random_route,
            name=f"depth_router_{layer_idx}",
            # Only the depth routers take this. The memory router's `weight` never
            # scales anything -- MemoryBlock uses its idx/mask and a separate learned
            # gate -- so saturating its scores would cost discrimination in the
            # selection and buy nothing back.
            bias_init=config.route_bias_init,
        )

    def forward(self, x, conf=None, causal=False):
        cond = conf.unsqueeze(-1).to(x.dtype) if self.config.couple else None
        out = self.router(x, cond=cond, causal=causal)

        if not self.config.route_attention:
            return self._dense_attention_forward(x, out), out

        if out["idx"] is None:
            return self._masked_forward(x, out), out

        idx, weight = out["idx"], out["weight"]
        xs = gather_tokens(x, idx)                       # (B, k, C)
        delta = self.block.delta(xs)                     # (B, k, C)
        # Scaling by the router weight is what puts the router on the backward path.
        # Drop this and the router gets no gradient and never leaves its init.
        delta = delta * weight.unsqueeze(-1).to(delta.dtype)
        return scatter_add_tokens(x, idx, delta), out

    def _dense_attention_forward(self, x, out):
        """Route the block's contribution, not its attention (route_attention=False).

        The block runs over the whole sequence -- so every token still attends to
        every earlier token, exactly as the pretrained model expects -- and the delta
        is added back only where the router selected. Identical in both modes: `mask`
        is the top-k membership under teacher forcing and the thresholded causal
        prediction during generation, so the same expression serves both and there is
        no train/generate asymmetry to reconcile here.

        The router still receives gradient through `scores`, which is what keeps this
        trainable at all (see TopKTokenRouter.forward).
        """
        delta = self.block.delta(x)
        keep = out["mask"].to(delta.dtype) * out["scores"].to(delta.dtype)
        return x + delta * keep.unsqueeze(-1)

    def _masked_forward(self, x, out):
        """Causal-mode path: same semantics as the gather/scatter path, no speedup.

        The selected count varies per sequence here, so there is no fixed-shape
        gather. We instead run the block over every token while restricting attention
        to the selected keys, then add the delta back only where the mask says so --
        numerically what the routed model computes, at dense cost.

        That is deliberate: this path exists to measure the *quality* of causal
        routing. Speed is measured on the generation path, where tokens arrive one at
        a time and the threshold decision is naturally causal and genuinely cheap.
        """
        mask = out["mask"]                                # (B, T) bool
        delta = self.block.delta(x, key_mask=mask)        # (B, T, C)
        gate = (mask.to(delta.dtype) * out["weight"].to(delta.dtype)).unsqueeze(-1)
        return x + delta * gate
