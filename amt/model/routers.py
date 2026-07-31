"""Token routers: fixed-capacity top-k selection over the sequence.

Why fixed capacity (and not a learned per-token budget)
------------------------------------------------------
The naive reading of "each token gets the compute it needs" is a per-token halting
decision, which produces ragged batches, dynamic shapes, and a model that uses fewer
FLOPs while running *slower* than dense (risk R3). Mixture-of-Depths' fix is to fix
the capacity k = ceil(c*T) and let the router decide only *which* tokens fill it. The
shapes stay static, torch.compile stays happy, and the saving is real wall-clock.

The research claim is unaffected: contribution C1 is about *which* tokens the router
picks (does a token with a good memory hit get dropped?), not about the total.

The training/inference asymmetry -- and why the aux head exists
--------------------------------------------------------------
top-k over the sequence dimension is not causal: whether token 5 is selected depends
on token 900's score. That is fine under teacher forcing but impossible during
autoregressive generation, where token 900 does not exist yet. So every router also
trains a small auxiliary head to predict "would I have been in the top-k?" from the
token alone, and generation thresholds that instead. The gap between the two is a
reported metric (`route_agreement`), not an implementation detail to hide: if it is
low, the generating model is not the model that was trained.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKTokenRouter(nn.Module):
    """Selects ceil(capacity * T) tokens per sequence.

    Parameters
    ----------
    n_embd : width of the token representation being routed on
    capacity : fraction of tokens to select, in (0, 1]
    cond_dim : extra conditioning features concatenated to the token (the coupling
        wire feeds retrieval confidence in here; 0 disables it)
    random_route : baseline B4 -- score tokens at random at the same capacity, to
        show the learned router is doing more than hitting a FLOP target
    """

    def __init__(self, n_embd, capacity, cond_dim=0, random_route=False, name="router"):
        super().__init__()
        self.n_embd = n_embd
        self.capacity = capacity
        self.cond_dim = cond_dim
        self.random_route = random_route
        self.name = name

        self.w_route = nn.Linear(n_embd + cond_dim, 1, bias=True)
        # Auxiliary causal predictor. Trained on detached features so it can never
        # distort the language model to make its own job easier.
        self.w_aux = nn.Linear(n_embd + cond_dim, 1, bias=True)

        # Calibrated at eval time so the realised rate matches `capacity`.
        self.register_buffer("threshold", torch.tensor(0.0), persistent=True)

        self._init_weights()

    def _init_weights(self):
        for lin in (self.w_route, self.w_aux):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.zeros_(lin.bias)
        # Start the aux head near the target rate so early BCE gradients are sane.
        with torch.no_grad():
            p = min(max(self.capacity, 1e-3), 1 - 1e-3)
            self.w_aux.bias.fill_(math.log(p / (1 - p)))

    def k_for(self, T: int) -> int:
        return max(1, math.ceil(self.capacity * T))

    def _features(self, x, cond):
        if self.cond_dim == 0:
            return x
        assert cond is not None, f"{self.name}: cond_dim={self.cond_dim} but no cond given"
        return torch.cat([x, cond], dim=-1)

    def forward(self, x, cond=None, causal=False):
        """Route tokens.

        Two selection modes, and the difference is not cosmetic:

        * `causal=False` (training, teacher-forced eval): top-k over the sequence.
          Fast and exactly on budget, but NOT causal -- whether token 5 is selected
          depends on token 900's score. Returns `idx` (B, k) for the gather/scatter
          fast path.
        * `causal=True` (generation, causal-mode eval): threshold the auxiliary
          predictor per token, which depends on that token alone. Genuinely causal,
          but the count varies, so there is no fixed-shape gather -- returns `mask`
          only and the caller falls back to a masked dense path.

        Returns a dict with
          idx        : (B, k) selected positions ascending, or None in causal mode
          mask       : (B, T) bool, selected-or-not; defined in both modes
          weight     : (B, k) in top-k mode / (B, T) in causal mode. The caller MUST
                       multiply the block output by this -- it is the only path by
                       which the router receives gradient. Omit it and the router
                       never leaves its initialisation.
          scores     : (B, T) sigmoid router scores
          aux_logits : (B, T) causal predictor logits
          label      : (B, T) float, 1 where the token made the top-k (the BCE target)
        """
        B, T, _ = x.shape
        feat = self._features(x, cond)
        k = self.k_for(T)

        if self.random_route:
            scores = torch.rand(B, T, device=x.device, dtype=x.dtype)
        else:
            scores = torch.sigmoid(self.w_route(feat).squeeze(-1))       # (B, T)

        aux_logits = self.w_aux(feat.detach()).squeeze(-1)               # (B, T)

        # The BCE target is always the top-k membership, even in causal mode: the aux
        # head's job is to imitate the selection training actually used.
        topk_idx = scores.topk(k, dim=1).indices.sort(dim=1).values
        label = torch.zeros(B, T, device=x.device, dtype=scores.dtype)
        label.scatter_(1, topk_idx, 1.0)

        if causal:
            mask = torch.sigmoid(aux_logits) > self.threshold
            return {"idx": None, "mask": mask, "weight": scores, "scores": scores,
                    "aux_logits": aux_logits, "label": label}

        return {"idx": topk_idx, "mask": label.bool(), "weight": scores.gather(1, topk_idx),
                "scores": scores, "aux_logits": aux_logits, "label": label}

    # -- losses ------------------------------------------------------------

    def aux_loss(self, out):
        """BCE teaching the causal predictor to imitate the non-causal top-k."""
        return F.binary_cross_entropy_with_logits(out["aux_logits"], out["label"])

    def entropy_bonus(self, out):
        """Mean Bernoulli entropy of the router scores.

        Fixed capacity already rules out the classic collapse-to-all-or-nothing, but
        scores saturating at 0/1 still kills the gradient that reaches the router
        through `weight`. Keeping some entropy early keeps that path alive; annealed
        to zero so the final routing decisions can be confident.

        Computed in fp32, and that cast is load-bearing. Under bf16 autocast `scores`
        is bf16, whose 8-bit mantissa cannot represent 1 - 1e-5: it rounds to exactly
        1.0, the clamp meant to hold the log away from zero does nothing, (1 - p) is
        0, and the whole loss becomes NaN. Saturated scores are the precise case this
        term exists to prevent, so it has to stay finite through them -- and a
        pretrained backbone (amt/model/retrofit.py) saturates these routers on the
        first step, because the depth routers score the raw residual stream and a
        trained one carries far larger activations than a freshly initialised one.
        """
        p = out["scores"].float().clamp(1e-5, 1 - 1e-5)
        return -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    @torch.no_grad()
    def agreement(self, out):
        """Fraction of tokens where the causal predictor matches the top-k choice.

        Reported per RESEARCH_PLAN.md §2.2. Below ~0.85 and the generation-time model
        diverges from the trained one.
        """
        pred = (torch.sigmoid(out["aux_logits"]) > self.threshold).to(out["label"].dtype)
        return (pred == out["label"]).float().mean()

    @torch.no_grad()
    def calibrate(self, aux_logits):
        """Set the decision threshold so the realised routing rate matches capacity.

        Call on a held-out batch before evaluating generation.
        """
        p = torch.sigmoid(aux_logits).flatten()
        q = torch.quantile(p.float(), 1.0 - self.capacity)
        self.threshold.fill_(q.item())
        return self.threshold.item()

    def extra_repr(self):
        return (f"capacity={self.capacity}, cond_dim={self.cond_dim}, "
                f"random_route={self.random_route}")


def gather_tokens(x, idx):
    """(B, T, C) -> (B, k, C) for the selected positions."""
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def scatter_add_tokens(x, idx, y):
    """Residual-add (B, k, C) back into (B, T, C) at the selected positions."""
    return x.scatter_add(1, idx.unsqueeze(-1).expand(-1, -1, y.size(-1)), y)
