"""The exit router: one small head per layer that decides whether a token continues.

    h_l  ->  Linear(d, 64)  ->  ReLU  ->  Linear(64, 1)  ->  sigmoid  ->  P(continue)

Why early exit is *simpler* than per-layer skipping
---------------------------------------------------
The obvious alternative -- let each layer independently choose which tokens to
process, Mixture-of-Depths style -- has to pick those tokens with a top-k over the
sequence, and top-k is not causal: whether token 5 is selected depends on token 900's
score. That is fine under teacher forcing and impossible during generation, so an MoD
model needs a second, causal predictor head, a BCE loss teaching it to imitate the
top-k, a calibrated decision threshold, and a reported train/generate agreement number
to prove the two have not drifted apart.

An exit decision needs none of that. `P(continue)` is a function of that token's own
hidden state and nothing else, so it is causal by construction: the decision made
during generation is bit-for-bit the decision made during training. There is no second
head, no threshold to calibrate, and no agreement metric to caveat every result with.

The cost of that simplicity is that the realised depth is no longer fixed in advance.
A top-k router hits its FLOP budget exactly; this one is steered toward a budget by
`lambda_depth` and then *measured*. `scripts/compare.py` matches the baseline arms to
whatever depth the adaptive arm actually reached, for exactly this reason.

Monotonicity
------------
Exit is final. Once a token stops, it stops for every remaining layer -- which is what
makes the saving real, since the model can compact the batch down to the survivors and
never has to grow it back. The model enforces this by carrying a cumulative
`alive = prod_j p_j` rather than consulting each router independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExitRouter(nn.Module):
    """Scores P(this token continues past this layer).

    Parameters
    ----------
    n_embd : width of the residual stream being scored
    hidden : width of the router's single hidden layer
    bias_init : initial bias on the output unit. Positive means "continue", so a
        large value makes the stack start out dense -- see
        AdaptiveGPTConfig.router_bias_init for why a retrofit needs that.
    """

    def __init__(self, n_embd, hidden=64, bias_init=4.0, name="router"):
        super().__init__()
        self.n_embd = n_embd
        self.bias_init = bias_init
        self.name = name

        self.fc = nn.Linear(n_embd, hidden)
        self.proj = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for lin in (self.fc, self.proj):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            nn.init.zeros_(lin.bias)
        with torch.no_grad():
            self.proj.bias.fill_(self.bias_init)

    def forward(self, x):
        """(B, T, C) -> (B, T) logits. Positive logit = continue.

        The router reads the residual stream directly rather than a normalised copy.
        That is deliberate for the retrofit case: the *scale* of the residual stream
        grows with depth in a trained transformer, and that growth is itself signal
        about how much the layer stack is still doing to this token.
        """
        return self.proj(F.relu(self.fc(x))).squeeze(-1)

    def extra_repr(self):
        return f"n_embd={self.n_embd}, bias_init={self.bias_init}"


def hard_gate(alive_p, threshold, straight_through=True):
    """Turn a cumulative continue-probability into the gate the block actually sees.

    `alive_p` is prod_j P(continue at layer j), so it decreases monotonically down the
    stack -- which is what makes the exit final.

    With `straight_through`, the forward value is the hard {0, 1} decision that
    inference makes, while the backward pass sees d/d(alive_p). The model that trains
    is therefore the model that gets benchmarked, and the router still gets gradient.
    Drop the straight-through term and the routers receive nothing: a comparison
    against a constant has zero derivative, training proceeds normally, the loss goes
    down, and the exit decisions stay frozen at their initialisation forever.
    """
    hard = (alive_p > threshold).to(alive_p.dtype)
    if not straight_through:
        return alive_p
    return hard + alive_p - alive_p.detach()


def router_bce(logits, targets, weight=None):
    """BCE teaching a router to predict the derived exit label.

    `weight` is the per-token mask of tokens that were still alive at this layer. A
    token that has already exited has no decision to make here, and training the
    router on it teaches the head to score representations that will never reach it.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weight is None:
        return loss.mean()
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom
