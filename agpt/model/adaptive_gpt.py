"""AdaptiveGPT: a GPT that decides, per token, how many layers to spend.

    embeddings
        |
    blocks [0, n_min_layers)          always run
        |
    +-> block l ---> router l ---> continue? --+
    |                                  |       |
    |                                 exit     |
    |                                  |       |
    +----------------------------------+-------+
        |
    ln_f -> lm_head

A token that exits keeps its hidden state frozen for the rest of the stack and is
unembedded from wherever it stopped. It remains part of the context: later tokens
still attend to it (see `AdaptiveGPTConfig.exited_as_keys`).

Two forward paths, and both matter
----------------------------------
`forward` runs every block over every token and *gates* each block's contribution.
Static shapes, full gradient, torch.compile-friendly -- but no faster than dense,
because the arithmetic still happens and is then multiplied by zero. This is the
training and teacher-forced-eval path.

`forward_compact` gathers the surviving tokens at each layer and runs the block on
that subset only. Dynamic shapes, inference only, and genuinely faster. This is what
`scripts/benchmark.py` measures and what `generate` uses.

They compute the same function. `tests/test_exit.py::test_compact_matches_dense`
holds them to it, and that test is the reason any efficiency number here can be
believed: it is the only thing standing between "the fast path is faster" and "the
fast path is faster because it is doing something else".
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import Block, scatter_add_tokens
from .config import AdaptiveGPTConfig
from .router import ExitRouter, hard_gate, router_bce


def strip_compile_prefix(state_dict):
    """Drop the `_orig_mod.` prefix torch.compile adds to every key.

    `torch.compile` returns an OptimizedModule wrapper whose `state_dict()` namespaces
    the real module underneath it, so a checkpoint saved from the compiled handle
    cannot be loaded into a plain AdaptiveGPT -- which is every consumer: resuming
    training, inference, analysis.
    """
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def stats_to_floats(stats):
    """Convert the tensor telemetry from forward() into plain floats.

    Call this OUTSIDE the compiled region -- once per logged step, not per forward.
    See `_stats` for why the conversion cannot live inside the model.
    """
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in stats.items()}


class AdaptiveGPT(nn.Module):

    def __init__(self, config: AdaptiveGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying

        # Keyed by the layer the router sits AFTER. A ModuleDict rather than a list
        # so the key survives a config change: adding a layer to `router_layers`
        # must not silently renumber the routers a checkpoint was saved with.
        self.routers = nn.ModuleDict({
            str(l): ExitRouter(config.n_embd, config.router_hidden,
                               config.router_bias_init, name=f"exit_{l}")
            for l in config.router_layers
        })

        # Loss weights live as buffers, written in place by the trainer. A Python
        # float read inside forward() is a value torch.compile specialises on, so
        # annealing one recompiles the whole graph every step it changes -- ~4 min
        # each on the target GPU, which makes a run impossible rather than slow.
        self.register_buffer("lambda_router",
                             torch.tensor(float(config.lambda_router)),
                             persistent=False)
        self.register_buffer("lambda_depth",
                             torch.tensor(float(config.lambda_depth)),
                             persistent=False)
        # Set by the trainer during Stage 1 to hold the stack dense while the
        # language model itself is still being learned. A buffer for the same reason.
        self.register_buffer("routing_enabled", torch.tensor(1.0), persistent=False)

        self.apply(self._init_weights)
        # The routers carry a deliberate non-default bias (start dense, learn to
        # exit). The blanket init above would have flattened it.
        for m in self.routers.values():
            m._init_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- schedule hooks (in place; see the buffer comment above) -------------

    def set_lambda_depth(self, w):
        self.lambda_depth.fill_(float(w))

    def set_lambda_router(self, w):
        self.lambda_router.fill_(float(w))

    def set_routing_enabled(self, on):
        """0.0 pins every gate open, so the model is exactly its dense self."""
        self.routing_enabled.fill_(1.0 if on else 0.0)

    # -- embedding ---------------------------------------------------------

    def _embed(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        return self.transformer.wte(idx) + self.transformer.wpe(pos)

    # -- the exit decision --------------------------------------------------

    def _continue_prob(self, layer, x, generator=None):
        """P(a token continues past block `layer`), one entry per token.

        Every arm answers the same question; only `adaptive` uses parameters to do
        it. Keeping the three baselines here rather than in separate model classes is
        what makes them honest comparisons -- they run the identical stack, the
        identical gating arithmetic and the identical loss, and differ only in this
        one expression.
        """
        c = self.config
        key = str(layer)
        if c.exit_mode == "adaptive" and key in self.routers:
            logit = self.routers[key](x)
            return torch.sigmoid(logit), logit
        if c.exit_mode == "random":
            r = torch.rand(x.shape[:2], device=x.device, dtype=x.dtype,
                           generator=generator)
            # Hard 0/1 rather than a constant probability: the baseline has to make
            # a *decision* per token, or it is just a uniformly shallower model and
            # the comparison against `fixed` collapses.
            return (r < c.random_continue_p).to(x.dtype), None
        if c.exit_mode == "fixed":
            keep = 1.0 if (layer + 1) < c.fixed_exit_layer else 0.0
            return x.new_full(x.shape[:2], keep), None
        return None, None                                  # dense: nothing to decide

    def _routes_at(self, layer):
        """Does an exit decision happen after this block?"""
        c = self.config
        if c.exit_mode == "dense":
            return False
        return c.n_min_layers - 1 <= layer < c.n_layer - 1

    # -- forward (gated, static shapes) -------------------------------------

    def forward(self, idx, targets=None, exit_labels=None, return_logits=False,
                generator=None):
        """
        Parameters
        ----------
        exit_labels : (n_layer, B, T) float from `targets.exit_targets`, or None.
            When given, each router is trained by BCE against its row. This is the
            Stage 2 supervision; Stage 3 usually passes it too, at a lower weight.

        Returns (logits_or_None, losses, stats).
        """
        c = self.config
        x = self._embed(idx)
        B, T = idx.size()

        alive_p = x.new_ones(B, T)        # cumulative P(continue), differentiable
        gate = x.new_ones(B, T)           # what multiplies this block's contribution
        depth = x.new_zeros(B, T)         # hard layer count, telemetry
        expected_depth = x.new_zeros(B, T)  # soft layer count, the depth penalty
        logits_by_layer, alive_by_layer = {}, {}

        for l, block in enumerate(self.transformer.h):
            if l < c.n_min_layers or c.exit_mode == "dense":
                x = block(x)
                depth = depth + 1.0
                expected_depth = expected_depth + 1.0
            else:
                # `routing_enabled` pins the gate open during Stage 1 without
                # changing the graph, so switching stages costs no recompile. The
                # depth counters are blended the same way rather than left to track
                # the decisions the gate is currently ignoring -- a Stage 1 run that
                # logs 6.2 layers/token while computing all 12 is a metric that
                # invites exactly the wrong conclusion.
                on = self.routing_enabled
                g = gate * on + (1.0 - on)
                x = x + g.unsqueeze(-1) * block.delta(x)
                hard = (alive_p > c.exit_threshold).to(depth.dtype)
                depth = depth + hard * on + (1.0 - on)
                expected_depth = expected_depth + alive_p * on + (1.0 - on)

            if self._routes_at(l):
                alive_by_layer[l] = (alive_p > c.exit_threshold).to(x.dtype)
                p, logit = self._continue_prob(l, x, generator=generator)
                alive_p = alive_p * p
                gate = hard_gate(alive_p, c.exit_threshold, c.straight_through)
                if logit is not None:
                    logits_by_layer[l] = logit

        x = self.transformer.ln_f(x)

        losses, logits = {}, None
        if targets is not None:
            losses["lm"] = self._lm_loss(x, targets)
        if return_logits or targets is None:
            logits = self.lm_head(x)

        losses["router"] = self._router_loss(logits_by_layer, alive_by_layer,
                                             exit_labels, x)
        losses["depth"] = expected_depth.mean() / c.n_layer
        if targets is not None:
            losses["total"] = (losses["lm"]
                               + self.lambda_router * losses["router"]
                               + self.lambda_depth * losses["depth"])

        return logits, losses, self._stats(depth, expected_depth, logits_by_layer)

    # -- forward (compact, dynamic shapes, inference only) ------------------

    @torch.no_grad()
    def forward_compact(self, idx, targets=None, return_logits=False,
                        generator=None):
        """The same function, computed only where it matters.

        At each routable layer the surviving tokens are gathered into a contiguous
        block and the layer runs on that alone. Because an exit is final the survivor
        set only ever shrinks, so this never has to grow the batch back.

        The count is rounded up to `config.compact_bucket`, which is the difference
        between torch.compile seeing a handful of shapes and seeing a new one every
        step. Uncompiled, the bucketing costs a little wasted work and nothing else.
        """
        c = self.config
        x = self._embed(idx)
        B, T = idx.size()

        alive = torch.ones(B, T, dtype=torch.bool, device=idx.device)
        alive_p = x.new_ones(B, T)
        depth = x.new_zeros(B, T)
        drop = c.exited_as_keys == "drop"

        for l, block in enumerate(self.transformer.h):
            if l < c.n_min_layers or c.exit_mode == "dense":
                x = block(x)
                depth = depth + 1.0
            elif alive.any():
                sel, valid = self._compact_index(alive)
                delta = block.delta_compact(x, sel, valid, drop_exited=drop)
                # Zero the padded rows BEFORE the scatter-add: their positions repeat
                # a real one, so a nonzero value there would be added to a token that
                # was not supposed to be updated.
                delta = delta * valid.unsqueeze(-1).to(delta.dtype)
                x = scatter_add_tokens(x, sel, delta)
                depth = depth + alive.to(depth.dtype)
            else:
                break     # everyone has exited; the rest of the stack is the identity

            if self._routes_at(l):
                p, _ = self._continue_prob(l, x, generator=generator)
                alive_p = alive_p * p
                alive = alive_p > c.exit_threshold

        x = self.transformer.ln_f(x)
        losses, logits = {}, None
        if targets is not None:
            losses["lm"] = self._lm_loss(x, targets)
        if return_logits or targets is None:
            logits = self.lm_head(x)
        return logits, losses, self._stats(depth, depth, {})

    def _compact_index(self, alive):
        """(B, T) bool -> (idx, valid), both (B, k) with k bucketed.

        A stable descending sort of the boolean puts the surviving positions first,
        still in ascending position order, and the dead ones after. Padded slots are
        pointed at position 0 and marked invalid; `delta_compact` masks them out of
        the attention and `scatter_add_tokens` adds zero to them.
        """
        c = self.config
        B, T = alive.shape
        k = int(alive.sum(dim=1).max().item())
        if c.compact_bucket > 0:
            k = min(T, math.ceil(k / c.compact_bucket) * c.compact_bucket)
        k = max(k, 1)

        order = torch.argsort(alive.to(torch.uint8), dim=1, descending=True,
                              stable=True)
        idx = order[:, :k]
        valid = alive.gather(1, idx)
        return torch.where(valid, idx, torch.zeros_like(idx)), valid

    # -- losses -------------------------------------------------------------

    def _lm_loss(self, x, targets):
        """Cross-entropy, computed in chunks so the (B, T, vocab) logits never all exist.

        At B=4, T=512, vocab=50304 the logits are ~200 MB in bf16 and cross_entropy
        upcasts to fp32 on top of that -- on a 4 GB card this single tensor is the
        difference between training and OOM. Chunking under checkpoint() recomputes
        each slice in the backward pass instead of holding all of them.
        """
        B, T, C = x.shape
        n = self.config.ce_chunks
        flat_x = x.view(B * T, C)
        flat_t = targets.reshape(B * T)

        if n <= 1:
            return F.cross_entropy(self.lm_head(flat_x), flat_t)

        def chunk_loss(xc, tc):
            return F.cross_entropy(self.lm_head(xc), tc, reduction="sum")

        total = flat_x.new_zeros((), dtype=torch.float32)
        for xc, tc in zip(flat_x.chunk(n, dim=0), flat_t.chunk(n, dim=0)):
            if self.training and torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, xc, tc, use_reentrant=False).float()
            else:
                total = total + chunk_loss(xc, tc).float()
        return total / flat_t.numel()

    def _router_loss(self, logits_by_layer, alive_by_layer, exit_labels, ref):
        """Mean BCE over the routers, against the derived convergence labels.

        Each router is scored only on the tokens that were still alive when it ran.
        A token that exited three layers ago never reaches this decision, and
        training the head on it teaches it to score states it will never see.
        """
        if not logits_by_layer or exit_labels is None:
            return ref.new_zeros(())
        total = ref.new_zeros(())
        for l, logit in logits_by_layer.items():
            total = total + router_bce(logit.float(),
                                       exit_labels[l].to(logit.device).float(),
                                       weight=alive_by_layer[l].float())
        return total / len(logits_by_layer)

    # -- telemetry ----------------------------------------------------------

    @torch.no_grad()
    def _stats(self, depth, expected_depth, logits_by_layer):
        """Routing telemetry, returned as TENSORS.

        Nothing here may call `.item()`. Doing so inside the forward pass is a
        torch.compile graph break plus a GPU->CPU sync on every step -- measured at
        several minutes of extra compile time and a visible throughput loss, worst on
        exactly the variants being evaluated. Callers convert once, outside the
        compiled region, with `stats_to_floats`.
        """
        stats = {
            "depth": depth.mean(),
            "depth_expected": expected_depth.mean(),
            "depth_frac": depth.mean() / self.config.n_layer,
            "exit_frac": (depth < self.config.n_layer).to(depth.dtype).mean(),
        }
        for l, logit in logits_by_layer.items():
            stats[f"p_continue/{l}"] = torch.sigmoid(logit).mean()
        return stats

    @torch.no_grad()
    def token_depths(self, idx, generator=None):
        """(B, T) layers spent on each token.

        `forward` reports the mean and `forward_compact` cannot report anything else
        without undoing the compaction, but the *distribution* of depth is the whole
        story -- an average of 6.2 means one thing if it is every token at 6 and quite
        another if it is half the tokens at 3 and half at 10. This walks the gated
        path to recover it, at dense cost, for figures and eval only.
        """
        c = self.config
        x = self._embed(idx)
        alive_p = x.new_ones(idx.shape)
        alive_h = x.new_ones(idx.shape)
        depth = x.new_zeros(idx.shape)

        for l, block in enumerate(self.transformer.h):
            if l < c.n_min_layers or c.exit_mode == "dense":
                x = block(x)
                depth = depth + 1.0
            else:
                x = x + alive_h.unsqueeze(-1) * block.delta(x)
                depth = depth + alive_h
            if self._routes_at(l):
                p, _ = self._continue_prob(l, x, generator=generator)
                alive_p = alive_p * p
                alive_h = (alive_p > c.exit_threshold).to(alive_h.dtype)
        return depth

    # -- dense probe (Stage 2 supervision) ----------------------------------

    @torch.no_grad()
    def dense_hidden_states(self, idx):
        """Every layer's residual stream, routing ignored.

        Returns a list of length n_layer + 1: the embedding, then the state after
        each block. `targets.exit_targets` turns this into the labels the routers are
        trained on. Deliberately not a flag on `forward` -- the labels have to come
        from what the model does when it is NOT allowed to exit, and a shared code
        path would make it far too easy to derive them from a model that already had.
        """
        x = self._embed(idx)
        hiddens = [x]
        for block in self.transformer.h:
            x = block(x)
            hiddens.append(x)
        return hiddens

    # -- optimiser -----------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, device_type,
                             router_lr_mult=1.0, verbose=True):
        """AdamW with the routers in their own group.

        The routers get no weight decay: decay drags their output toward a constant,
        which makes every token exit at the same depth and quietly turns the adaptive
        arm into the fixed-depth baseline it is supposed to beat.

        Unlike the top-k router this replaced, the LR multiplier defaults to 1.0. A
        top-k router's score has to stay calibrated against the whole sequence's
        scores, so it is destabilised by moving faster than the representations it
        ranks; an exit router only has to threshold its own token and does not have
        that problem. Lower it if Stage 3 shows the depth collapsing.
        """
        router_ids = {id(p) for p in self.routers.parameters()}
        decay, nodecay, routers = [], [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in router_ids:
                routers.append(p)
            elif p.dim() >= 2:
                decay.append(p)
            else:
                nodecay.append(p)

        groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": learning_rate},
            {"params": nodecay, "weight_decay": 0.0, "lr": learning_rate},
            {"params": routers, "weight_decay": 0.0,
             "lr": learning_rate * router_lr_mult, "is_router": True},
        ]
        groups = [g for g in groups if g["params"]]
        if verbose:
            print(f"optim groups: decay={sum(p.numel() for p in decay):,} "
                  f"nodecay={sum(p.numel() for p in nodecay):,} "
                  f"router={sum(p.numel() for p in routers):,}")

        fused = ("fused" in inspect.signature(torch.optim.AdamW).parameters
                 and device_type == "cuda")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95),
                                 eps=1e-8, fused=fused)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
            n -= self.transformer.wte.weight.numel()   # tied with lm_head
        return n

    def router_params(self):
        return sum(p.numel() for p in self.routers.parameters())

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50,
                 generator=None, compact=True, return_depth=False):
        """Autoregressive sampling.

        No calibration step and no separate causal routing mode: an exit decision
        reads one token's own hidden state, so the decision made here is exactly the
        decision made during training. That equivalence is the main practical
        argument for early exit over per-layer top-k routing -- see router.py.
        """
        self.eval()
        forward = self.forward_compact if compact else self.forward
        depths = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, stats = forward(idx_cond, return_logits=True,
                                       generator=generator)
            if return_depth:
                depths.append(stats["depth"])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat((idx, nxt), dim=1)
        return (idx, depths) if return_depth else idx
