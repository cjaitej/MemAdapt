"""The Adaptive Memory Transformer.

Assembles the stack described in RESEARCH_PLAN.md §2.1:

    dense trunk -> memory layer -> adaptive stack -> dense tail -> head

The trunk is dense because routing on barely-contextualised embeddings is uninformed;
the tail is dense because skipping the last layer wrecks the output calibration.
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..precision import select_precision
from .blocks import AdaptiveBlock, Block, MemoryBlock
from .config import AMTConfig
from .memory import KVMemoryBank
from .routers import TopKTokenRouter


def strip_compile_prefix(state_dict):
    """Drop the `_orig_mod.` prefix torch.compile adds to every key.

    `torch.compile` returns an OptimizedModule wrapper, and its `state_dict()`
    namespaces the real module underneath it. A checkpoint saved from the compiled
    handle therefore cannot be loaded into a plain AMT -- which is every consumer:
    resuming training, inference, analysis.

    Kept tolerant rather than strict because checkpoints written before this was
    fixed still carry the prefix, and a run that was interrupted mid-experiment is
    exactly when reloading matters most.
    """
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def stats_to_floats(stats):
    """Convert the tensor telemetry from AMT.forward into plain floats.

    Call this OUTSIDE the compiled region -- once per logged step, not per forward.
    See AMT._stats for why the conversion cannot live inside the model.
    """
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in stats.items()}


class AMT(nn.Module):

    def __init__(self, config: AMTConfig):
        super().__init__()
        self.config = config

        blocks = []
        for i in range(config.n_layer):
            if i == config.mem_layer and config.use_memory:
                blocks.append(MemoryBlock(config))
            elif i in config.adaptive_layers:
                blocks.append(AdaptiveBlock(config, layer_idx=i))
            else:
                blocks.append(Block(config))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList(blocks),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying

        # Annealed by the trainer. A buffer rather than a forward argument for the
        # same reason as MemoryBlock.gate_floor: a Python float that changes every
        # step forces torch.compile to recompile every step. Set with
        # `set_entropy_weight`, which writes in place.
        self.register_buffer("entropy_weight",
                             torch.tensor(float(config.lambda_entropy)),
                             persistent=False)

        self.apply(self._init_weights)
        # Routers and gates carry deliberate non-default inits (target routing rate,
        # closed memory gate). The blanket init above would have flattened them.
        for m in self.modules():
            if isinstance(m, TopKTokenRouter):
                m._init_weights()
            elif isinstance(m, MemoryBlock):
                nn.init.constant_(m.w_gate.bias, config.mem_gate_init)

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

    # -- memory plumbing ---------------------------------------------------

    def make_bank(self, batch_size, device, dtype=None) -> KVMemoryBank:
        """Allocate the KV bank, defaulting to this device's autocast dtype.

        Not a hardcoded bf16 default any more. The bank's normalised keys are one
        operand of the kNN similarity matmul, so a bank dtype that disagrees with
        the autocast dtype moves that matmul off the tensor-core path silently --
        and bf16 on a pre-Ampere card is emulated, which is the worst case of all.
        """
        c = self.config
        if dtype is None:
            dev = str(device)
            dtype, _, _ = select_precision(
                "cuda" if dev.startswith("cuda") else "cpu", verbose=False)
        return KVMemoryBank(batch_size, c.n_head, c.head_dim, c.mem_size, device, dtype)

    # -- forward -----------------------------------------------------------

    def set_entropy_weight(self, w):
        """Anneal the entropy bonus. In-place so no compile guard is invalidated."""
        self.entropy_weight.fill_(float(w))

    def forward(self, idx, targets=None, bank=None, causal=False,
                return_logits=False):
        """
        Parameters
        ----------
        bank : KVMemoryBank or None. When given, the memory layer reads from it and
            returns the keys/values to be written *after* this call -- see
            `write_memory`. Never written to inside forward (invariant R6).
        causal : route with the auxiliary causal predictor instead of top-k. Required
            for autoregressive generation; optional at eval to measure the gap.

        Returns (logits_or_None, losses, stats, kv_to_write)
        """
        B, T = idx.size()
        c = self.config
        assert T <= c.block_size, f"sequence length {T} exceeds block_size {c.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        conf = torch.zeros(B, T, device=idx.device, dtype=x.dtype)
        route_outs, kv = [], None

        for block in self.transformer.h:
            if isinstance(block, MemoryBlock):
                x, conf, r_out, kv = block(x, bank=bank, causal=causal)
                if r_out is not None:
                    route_outs.append(("mem", block.mem_router, r_out))
            elif isinstance(block, AdaptiveBlock):
                x, r_out = block(x, conf=conf, causal=causal)
                route_outs.append((f"depth_{block.layer_idx}", block.router, r_out))
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        losses, logits = {}, None
        if targets is not None:
            losses["lm"] = self._lm_loss(x, targets)
        if return_logits or targets is None:
            logits = self.lm_head(x)

        losses.update(self._router_losses(route_outs))
        if targets is not None:
            losses["total"] = (
                losses["lm"]
                + c.lambda_aux * losses["aux"]
                - self.entropy_weight * losses["entropy"]
            )

        return logits, losses, self._stats(route_outs, conf, T), kv

    def _lm_loss(self, x, targets):
        """Cross-entropy, computed in chunks so the (B, T, vocab) logits never all exist.

        At B=4, T=512, vocab=50304 the logits are ~200MB in bf16 and cross_entropy
        upcasts to fp32 on top of that -- on a 4GB card this single tensor is the
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

    def _router_losses(self, route_outs):
        if not route_outs:
            zero = torch.zeros((), device=self.lm_head.weight.device)
            return {"aux": zero, "entropy": zero}
        aux = sum(r.aux_loss(o) for _, r, o in route_outs) / len(route_outs)
        ent = sum(r.entropy_bonus(o) for _, r, o in route_outs) / len(route_outs)
        return {"aux": aux, "entropy": ent}

    @torch.no_grad()
    def _stats(self, route_outs, conf, T):
        """Routing telemetry, returned as TENSORS.

        Nothing here may call `.item()`. Doing so inside the forward pass is a
        torch.compile graph break plus a GPU->CPU sync on every step -- measured as
        several minutes of extra compile time and a visible throughput loss, worst on
        the variants with the most routers, i.e. exactly the ones being evaluated.
        Callers convert with `stats_to_floats` once, outside the compiled region.
        """
        c = self.config
        n_always_on = c.n_layer - len(c.adaptive_layers)
        layers = conf.new_tensor(float(n_always_on))
        stats = {
            "mem_conf_mean": conf.mean(),
            "mem_rate": (conf != 0).float().mean(),
        }
        for name, router, out in route_outs:
            rate = out["label"].mean()
            stats[f"rate/{name}"] = rate
            stats[f"agree/{name}"] = router.agreement(out)
            stats[f"score/{name}"] = out["scores"].mean()
            if name.startswith("depth"):
                layers = layers + rate
        stats["layers_per_token"] = layers
        return stats

    def write_memory(self, bank, kv):
        """Commit this segment's keys/values to the bank.

        Call AFTER forward, never during: the bank may only ever contain tokens
        strictly earlier than the segment being predicted (invariant R6).

        The per-step order is therefore three separate calls, and the order matters:

            bank.clear(reset_mask)        # BEFORE forward -- see below
            ... = model(x, targets=y, bank=bank)
            model.write_memory(bank, kv)  # AFTER forward

        Clearing has to precede the forward pass. A stream flagged by the loader is
        starting a *new document* with this very batch, so if the clear happens after
        the forward, that batch has already retrieved from the previous document. The
        loss barely moves when this is wrong, which is exactly what makes it
        dangerous -- hence no `reset_mask` argument here to tempt the mistake.
        """
        if bank is None or kv is None:
            return
        k, v = kv
        bank.write(k, v)
        bank.detach_()

    # -- optimiser ---------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, device_type,
                             router_lr_mult=0.1, verbose=True):
        """AdamW with routers in their own group.

        Router parameters are tiny (one row each) and touchy: at the full LR they
        outrun the representations they route on, and weight decay drags their scores
        toward a constant, which makes the selection degenerate. Separate group, 10x
        lower LR, no decay.
        """
        router_names = {
            n for n, _ in self.named_parameters()
            if ".router." in n or ".mem_router." in n or n.endswith("w_gate.bias")
        }
        decay, nodecay, routers = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n in router_names:
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
        if verbose:
            print(f"optim groups: decay={sum(p.numel() for p in decay):,} "
                  f"nodecay={sum(p.numel() for p in nodecay):,} "
                  f"router={sum(p.numel() for p in routers):,}")

        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95),
                                 eps=1e-8, fused=fused)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
            n -= self.transformer.wte.weight.numel()   # tied with lm_head
        return n

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50,
                 bank=None, generator=None):
        """Autoregressive sampling.

        Always routes with `causal=True`: the top-k selection used in training peeks
        at the whole sequence, which does not exist yet when generating.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _, _ = self(idx_cond, bank=bank, causal=True, return_logits=True)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    @torch.no_grad()
    def calibrate_routers(self, batches, device, bank=None):
        """Set every router's causal threshold so its realised rate hits its capacity.

        Run on held-out batches before any generation or causal-mode eval, otherwise
        the thresholds sit at their init and the routing rate at inference is whatever
        the aux head happens to produce.
        """
        routers = [m for m in self.modules() if isinstance(m, TopKTokenRouter)]
        collected = {id(m): [] for m in routers}
        hooks = [
            m.register_forward_hook(
                lambda mod, inp, out, c=collected: c[id(mod)].append(out["aux_logits"].detach())
            )
            for m in routers
        ]
        try:
            for x in batches:
                self(x.to(device), targets=None, bank=bank)
        finally:
            for h in hooks:
                h.remove()

        return {
            m.name: m.calibrate(torch.cat(collected[id(m)], dim=0))
            for m in routers if collected[id(m)]
        }
