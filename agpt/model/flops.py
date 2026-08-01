"""Analytic FLOP model.

Every efficiency claim in this project routes through here, so the conventions are
stated explicitly and the model is validated against `torch.utils.flop_counter` in
`tests/test_flops.py`. If the analytic number and the profiler disagree by more than a
few percent, the analytic model is wrong -- fix it here rather than fudging the
reported numbers.

Conventions
-----------
* A multiply-accumulate counts as 2 FLOPs (the usual convention; matches
  FlopCounterMode and Karpathy's 6*N*D rule of thumb).
* Numbers are FORWARD FLOPs per token. Training cost is ~3x forward.
* Softmax / LayerNorm / GELU are elementwise and O(d) or O(T); they are excluded, as
  is standard, and are <1% of the total at these shapes.

Why an exited token is not free
-------------------------------
The tempting arithmetic is "half the layers, half the FLOPs". It is wrong, and by
enough to matter.

Under `exited_as_keys="stale"` a token that has stopped computing is still part of the
context: every later token attends to it, so its key and value have to exist at every
later layer. That costs the 4d^2 k/v projection for *every* token at *every* layer,
whether it exited or not. What routing actually saves is the query projection (2d^2), the
output projection (2d^2) and the MLP (16d^2) -- 20d^2 of the layer's 24d^2 -- plus
that token's own attention row.

So the honest per-layer cost at active fraction `a` is

    4d^2  +  a * (20d^2 + 4d*ctx)

not `a * (24d^2 + 4d*ctx)`. At a = 0.5 that is a 42% layer saving, not 50%.
`exited_as_keys="drop"` recovers the difference by removing exited tokens from the
attention entirely, and pays for it in quality -- see AdaptiveGPTConfig.

The output head is the other half of the story
----------------------------------------------
`lm_head` runs for every token at full width no matter how early it exited, and at
d=384 with the GPT-2 vocabulary it is ~44% of dense forward FLOPs -- about 11x one
transformer layer. Nothing routing does touches it, so it sets a hard floor on the
total-FLOP saving. Report both axes: `layers` (the standard "non-embedding" number
used in the adaptive-compute literature) and `total`. The gap between them is itself a
finding worth stating -- at small d, adaptive-depth methods have far less headroom
than the layer-only figures in the literature suggest.
"""

from dataclasses import dataclass

import torch


@dataclass
class FlopBreakdown:
    """Forward FLOPs per token, split so the head can be excluded from claims."""
    min_layers: float = 0.0    # the always-on prefix
    routable: float = 0.0      # the early-exit region, already scaled by activity
    routing: float = 0.0       # the exit routers themselves
    head: float = 0.0          # lm_head (+ tied embedding read)

    @property
    def layers(self) -> float:
        """Everything except the output head -- the 'non-embedding' FLOPs."""
        return self.min_layers + self.routable + self.routing

    @property
    def total(self) -> float:
        return self.layers + self.head

    def as_dict(self) -> dict:
        return dict(min_layers=self.min_layers, routable=self.routable,
                    routing=self.routing, head=self.head,
                    layers=self.layers, total=self.total)


def active_fractions(depths, n_layer):
    """Per-layer fraction of tokens still computing, from measured exit depths.

    `depths` is a tensor of per-token layer counts. A token of depth D entered blocks
    0..D-1, so the fraction entering block l is P(depth > l). Returns a list of length
    n_layer.

    This is the right primitive to feed `breakdown`: the average depth alone does not
    determine the cost, because the layers are not equally expensive to skip once the
    always-on prefix and the k/v floor are accounted for.
    """
    d = depths.detach().float().flatten()
    return [(d > l).float().mean().item() for l in range(n_layer)]


class FlopModel:
    """Analytic cost model for an AdaptiveGPTConfig."""

    def __init__(self, config):
        self.c = config

    # -- primitives --------------------------------------------------------

    def layer_matmul(self) -> float:
        """Projection FLOPs per token for one dense transformer layer.

        qkv 6d^2 + out-proj 2d^2 + mlp (d->4d->d) 16d^2 = 24 d^2
        """
        d = self.c.n_embd
        return 24.0 * d * d

    def layer_attention(self, ctx_len=None) -> float:
        """Attention score+value FLOPs per token: 4 * d * ctx_len.

        For causal attention over a segment of length T the average query attends to
        (T+1)/2 keys, so that is the default context length.
        """
        d = self.c.n_embd
        if ctx_len is None:
            ctx_len = (self.c.block_size + 1) / 2.0
        return 4.0 * d * ctx_len

    def layer(self, ctx_len=None) -> float:
        """Forward FLOPs per token for one dense transformer layer."""
        return self.layer_matmul() + self.layer_attention(ctx_len)

    def adaptive_layer(self, active_frac, ctx_len=None) -> float:
        """Forward FLOPs per token for one routable layer at a given activity.

        See the module docstring for why the k/v projections do not scale with
        `active_frac` under the default `exited_as_keys="stale"`.
        """
        d, a = self.c.n_embd, active_frac
        if a <= 0.0:
            # Nobody is left to query, so nothing is attended to and the k/v carry is
            # not paid either -- `forward_compact` stops walking the stack entirely.
            # Without this the `fixed` baseline and `budget_range` both charge a k/v
            # projection for layers that never execute.
            return 0.0
        attn = self.layer_attention(ctx_len)
        if self.c.exited_as_keys == "drop":
            # Exited tokens vanish, so the k/v projections go with them and the
            # attention shrinks on both axes: fewer queries AND fewer keys.
            return a * 24.0 * d * d + a * a * attn
        return 4.0 * d * d + a * (20.0 * d * d + attn)

    def routing(self, active_frac=1.0) -> float:
        """Router projections per token, summed over every router in the model.

        Two Linears each: Linear(d, hidden) then Linear(hidden, 1). Tiny -- ~0.1% of
        the total -- but they are matmuls, so FlopCounterMode counts them and leaving
        them out biases the validation comparison rather than merely rounding it.

        Scaled by activity: a router only runs on tokens that are still alive.
        """
        c = self.c
        if not c.router_layers:
            return 0.0
        per_router = 2.0 * c.n_embd * c.router_hidden + 2.0 * c.router_hidden
        return len(c.router_layers) * per_router * active_frac

    def head(self) -> float:
        """lm_head projection: d -> vocab_size. Paid by every token, always."""
        return 2.0 * self.c.n_embd * self.c.vocab_size

    # -- assembled ---------------------------------------------------------

    def breakdown(self, active_fracs=None, ctx_len=None) -> FlopBreakdown:
        """Forward FLOPs per token given per-layer activity.

        `active_fracs` is a list of length n_layer -- the fraction of tokens entering
        each block, as produced by `active_fractions`. Defaults to fully dense.
        """
        c = self.c
        if active_fracs is None:
            active_fracs = [1.0] * c.n_layer
        if len(active_fracs) != c.n_layer:
            raise ValueError(f"active_fracs has {len(active_fracs)} entries, "
                             f"expected n_layer={c.n_layer}")

        per_layer = self.layer(ctx_len)
        tail = active_fracs[c.n_min_layers:]
        routable = sum(self.adaptive_layer(a, ctx_len) for a in tail)
        # The routers run on whatever was alive when each of them fired; the mean
        # activity over the routable region is a close enough stand-in for a term
        # this small.
        mean_active = sum(tail) / len(tail) if tail else 1.0

        return FlopBreakdown(
            min_layers=c.n_min_layers * per_layer,
            routable=routable,
            routing=self.routing(mean_active),
            head=self.head(),
        )

    def dense_breakdown(self, ctx_len=None) -> FlopBreakdown:
        """The same model with every token at full depth -- the denominator."""
        c = self.c
        per_layer = self.layer(ctx_len)
        return FlopBreakdown(
            min_layers=c.n_min_layers * per_layer,
            routable=(c.n_layer - c.n_min_layers) * per_layer,
            routing=self.routing(1.0),
            head=self.head(),
        )

    def from_depths(self, depths, ctx_len=None) -> FlopBreakdown:
        """Convenience: measured per-token depths -> a breakdown."""
        return self.breakdown(active_fractions(depths, self.c.n_layer), ctx_len)

    def at_uniform_depth(self, depth, ctx_len=None) -> FlopBreakdown:
        """The breakdown for a fixed-depth model, for the `fixed` baseline arm."""
        fracs = [1.0 if l < depth else 0.0 for l in range(self.c.n_layer)]
        return self.breakdown(fracs, ctx_len)

    # -- reporting ---------------------------------------------------------

    def budget_range(self, ctx_len=None):
        """(floor, ceiling, ratio) of achievable forward FLOPs per token.

        The floor is what the model costs if every token exits at the first
        opportunity: the always-on prefix, the routers, and the output head. (The k/v
        carry vanishes at that extreme -- with nobody left to query, there is nothing
        to be a key for.) A narrow range here means a short x-axis on the Pareto plot,
        i.e. the headline figure has little room to show anything. Check it before
        spending a training run.
        """
        c = self.c
        floor = self.breakdown([1.0] * c.n_min_layers
                               + [0.0] * (c.n_layer - c.n_min_layers), ctx_len).total
        ceil = self.dense_breakdown(ctx_len).total
        return floor, ceil, floor / ceil

    def summary(self, active_fracs=None) -> str:
        b = self.breakdown(active_fracs)
        d = self.dense_breakdown()
        floor, _, ratio = self.budget_range()
        return "\n".join([
            f"forward FLOPs/token   layers={b.layers/1e6:8.2f}M  "
            f"head={b.head/1e6:8.2f}M  total={b.total/1e6:8.2f}M",
            f"dense equivalent      layers={d.layers/1e6:8.2f}M  "
            f"head={d.head/1e6:8.2f}M  total={d.total/1e6:8.2f}M",
            f"savings               layers={1 - b.layers/d.layers:7.1%}   "
            f"total={1 - b.total/d.total:7.1%}",
            f"head share of dense   {d.head/d.total:7.1%}   "
            "<- caps the achievable total saving",
            f"floor at zero depth   {floor/1e6:8.2f}M = {ratio:.2f}x dense   "
            f"(always-on prefix + head)",
            f"exited_as_keys        {self.c.exited_as_keys!r}",
        ])


def measured_flops(model, idx, compact=False):
    """Forward FLOPs per token for one real batch, straight from PyTorch's profiler.

    The ground truth the analytic model above is checked against. Kept here rather
    than only in the test so `scripts/benchmark.py` can print the comparison too -- an
    efficiency claim that has never been held up against a profiler is a claim about a
    spreadsheet.

    IMPORTANT: on the builds this project runs on, `FlopCounterMode` reports **zero**
    FLOPs for `scaled_dot_product_attention` -- it has no formula registered for the
    fused kernels SDPA dispatches to. So this counts projections and the output head
    and nothing else. Compare it against `breakdown(..., ctx_len=0)`, which switches
    the analytic attention term off for exactly this reason; comparing it against the
    full analytic number silently credits the analytic model with an error the size of
    the entire attention cost.
    """
    from torch.utils.flop_counter import FlopCounterMode

    forward = model.forward_compact if compact else model.forward
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        forward(idx)
    return counter.get_total_flops() / idx.numel()


def report(config) -> str:
    return FlopModel(config).summary()
