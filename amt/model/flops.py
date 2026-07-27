"""Analytic FLOP model.

Every efficiency claim in this project routes through here, so the conventions are
stated explicitly and the model is validated against `torch.utils.flop_counter` in
`tests/test_flops.py`. If the analytic number and the profiler disagree by more than
a few percent, the analytic model is wrong -- fix it here rather than fudging the
reported numbers.

Conventions
-----------
* A multiply-accumulate counts as 2 FLOPs (the usual convention; matches
  FlopCounterMode and Karpathy's 6*N*D rule of thumb).
* Numbers are FORWARD FLOPs per token. Training cost is ~3x forward
  (1x forward + ~2x backward).
* Softmax / LayerNorm / GELU are elementwise and O(d) or O(T); they are excluded,
  as is standard. They are <1% of the total at these shapes.

A note on the design change vs RESEARCH_PLAN.md §2.4
----------------------------------------------------
The plan proposed a differentiable budget loss `(F/F_target - 1)^2` to *learn* the
compute/memory split. With fixed-capacity top-k routing (the choice that makes the
model actually fast -- see routers.py) the realised FLOPs are a deterministic
function of the config, so that loss has zero gradient and is meaningless.

The budget is therefore enforced *architecturally* and the compute<->memory exchange
rate is measured by sweeping configs along an iso-FLOP line (`iso_flop_configs`
below) instead of by a loss term. This is strictly more rigorous -- the exchange rate
becomes a measured quantity rather than an optimisation artefact -- and it removes
risk R1 (router collapse) almost entirely, since capacity can no longer drift.
Contribution C1 (retrieval-conditioned depth allocation) is unaffected: the router
still decides *which* tokens get depth, which is what the coupling claim is about.
"""

from dataclasses import dataclass


@dataclass
class FlopBreakdown:
    """Forward FLOPs per token, split so the head can be excluded from claims."""
    trunk: float = 0.0        # dense trunk layers
    memory_layer: float = 0.0  # the memory layer's own transformer compute
    retrieval: float = 0.0    # kNN scan + attend over neighbours
    adaptive: float = 0.0     # adaptive stack (already scaled by capacity)
    tail: float = 0.0         # dense tail layers
    head: float = 0.0         # lm_head (+ tied embedding read)

    @property
    def layers(self) -> float:
        """Everything except the output head -- the 'non-embedding' FLOPs."""
        return self.trunk + self.memory_layer + self.retrieval + self.adaptive + self.tail

    @property
    def total(self) -> float:
        return self.layers + self.head

    def as_dict(self) -> dict:
        return dict(
            trunk=self.trunk, memory_layer=self.memory_layer, retrieval=self.retrieval,
            adaptive=self.adaptive, tail=self.tail, head=self.head,
            layers=self.layers, total=self.total,
        )


class FlopModel:
    """Analytic forward-FLOPs-per-token model for an AMTConfig."""

    def __init__(self, config):
        self.c = config

    # -- primitives --------------------------------------------------------

    def layer_matmul(self) -> float:
        """Projection FLOPs per token for one transformer layer.

        qkv 6d^2 + out-proj 2d^2 + mlp (d->4d->d) 16d^2 = 24 d^2
        """
        d = self.c.n_embd
        return 24.0 * d * d

    def layer_attention(self, ctx_len: int = None) -> float:
        """Attention score+value FLOPs per token: 4 * d * ctx_len.

        For causal attention over a segment of length T the average query attends to
        (T+1)/2 keys, so that is the default context length.
        """
        d = self.c.n_embd
        if ctx_len is None:
            ctx_len = (self.c.block_size + 1) / 2.0
        return 4.0 * d * ctx_len

    def layer(self, ctx_len: int = None) -> float:
        """Total forward FLOPs per token for one dense transformer layer."""
        return self.layer_matmul() + self.layer_attention(ctx_len)

    def retrieval(self) -> float:
        """FLOPs per *retrieving* token: exact kNN scan + attend over neighbours.

        The scan dominates: every query head is compared against all M bank entries.
        This is why `mem_capacity` matters -- routing out 75% of tokens cuts it 4x.
        """
        d, M, k = self.c.n_embd, self.c.mem_size, self.c.n_neighbors
        scan = 2.0 * d * M
        attend = 2.0 * d * k
        return scan + attend

    def head(self) -> float:
        """lm_head projection: d -> vocab_size."""
        return 2.0 * self.c.n_embd * self.c.vocab_size

    # -- assembled ---------------------------------------------------------

    def breakdown(self, depth_capacity: float = None,
                  mem_capacity: float = None) -> FlopBreakdown:
        """Forward FLOPs per token. Capacities default to the config's."""
        c = self.c
        dc = c.effective_depth_capacity if depth_capacity is None else depth_capacity
        mc = c.effective_mem_capacity if mem_capacity is None else mem_capacity

        per_layer = self.layer()
        n_adaptive = len(c.adaptive_layers)
        # Layers that are neither trunk, memory, nor adaptive run dense.
        n_dense_other = c.n_layer - c.n_trunk - 1 - n_adaptive

        return FlopBreakdown(
            trunk=c.n_trunk * per_layer,
            memory_layer=per_layer,
            retrieval=mc * self.retrieval() if c.use_memory else 0.0,
            adaptive=n_adaptive * dc * per_layer,
            tail=n_dense_other * per_layer,
            head=self.head(),
        )

    def dense_breakdown(self) -> FlopBreakdown:
        """The same model with all routing disabled -- the denominator for savings."""
        c = self.c
        per_layer = self.layer()
        return FlopBreakdown(
            trunk=c.n_trunk * per_layer,
            memory_layer=per_layer,
            retrieval=self.retrieval() if c.use_memory else 0.0,
            adaptive=len(c.adaptive_layers) * per_layer,
            tail=(c.n_layer - c.n_trunk - 1 - len(c.adaptive_layers)) * per_layer,
            head=self.head(),
        )

    # -- the exchange rate (contribution C3) -------------------------------

    def exchange_rate(self) -> float:
        """rho = cost of one retrieval / cost of one layer, in layer-equivalents.

        This is the quantity the whole project is trying to measure empirically.
        The analytic value here is what the *hardware* charges; the experiments ask
        what the *loss* is willing to pay.
        """
        return self.retrieval() / self.layer()

    def budget_range(self) -> tuple:
        """(min, max) achievable forward FLOPs/token, and the same as a fraction of dense.

        The floor is the *unroutable* cost: dense trunk, memory layer, dense tail and
        the output head all run for every token no matter what the routers decide.
        A narrow range here means a short x-axis on the Pareto plot -- i.e. the
        headline figure has little room to show anything. Check this before training.
        """
        floor = self.breakdown(depth_capacity=0.0, mem_capacity=0.0).total
        ceil = self.dense_breakdown().total
        return floor, ceil, floor / ceil

    def iso_flop_configs(self, target_flops: float, n_points: int = 5,
                         max_depth_capacity: float = 1.0) -> list:
        """(depth_capacity, mem_capacity) pairs costing `target_flops` per token.

        Sweeping these and comparing perplexity is how the compute<->memory trade-off
        is measured (RESEARCH_PLAN.md §7.5 figure 1). Pairs needing a capacity outside
        (0, 1] are dropped, so the returned list may be shorter than n_points -- and
        empty if the target is below the unroutable floor, which raises rather than
        silently returning nothing.
        """
        c = self.c
        n_adaptive = len(c.adaptive_layers)
        if n_adaptive == 0 or not c.use_memory:
            raise ValueError("iso-FLOP sweep needs both an adaptive stack and memory")

        floor, ceil, _ = self.budget_range()
        if target_flops < floor:
            raise ValueError(
                f"target {target_flops/1e6:.2f}M FLOPs/token is below the unroutable "
                f"floor of {floor/1e6:.2f}M ({floor/ceil:.2f}x dense). Nothing can be "
                f"routed away to reach it. Reduce n_trunk/n_dense_tail, shrink "
                f"vocab_size (the head alone is {self.head()/1e6:.2f}M), or raise the target."
            )

        per_layer = self.layer()
        out = []
        for i in range(n_points):
            dc = max_depth_capacity * (i + 1) / n_points
            budget_left = target_flops - floor - n_adaptive * dc * per_layer
            mc = budget_left / self.retrieval()
            if 0.0 < mc <= 1.0:
                out.append((round(dc, 4), round(mc, 4)))
        return out

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        b = self.breakdown()
        d = self.dense_breakdown()
        lines = [
            f"forward FLOPs/token   layers={b.layers/1e6:8.2f}M  head={b.head/1e6:8.2f}M  total={b.total/1e6:8.2f}M",
            f"dense equivalent      layers={d.layers/1e6:8.2f}M  head={d.head/1e6:8.2f}M  total={d.total/1e6:8.2f}M",
            f"savings               layers={1 - b.layers/d.layers:7.1%}   total={1 - b.total/d.total:7.1%}",
            f"head share of dense   {d.head/d.total:7.1%}   <- caps achievable total savings",
            f"exchange rate rho     {self.exchange_rate():.3f} layer-equivalents per retrieval",
        ]
        return "\n".join(lines)


def report(config) -> str:
    return FlopModel(config).summary()
