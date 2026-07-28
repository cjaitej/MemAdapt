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

WHAT THE FLOP NUMBERS DO NOT INCLUDE (read this before quoting an efficiency figure)
------------------------------------------------------------------------------------
Everything above counts matmul FLOPs, because that is the convention
`FlopCounterMode` uses and comparing against it is the only way to validate the
model. For the dense transformer path that convention is fine -- matmuls really are
the work.

For the *retrieval* path it is not. Profiling `KVMemoryBank.read` on both a T4 and
an RTX 3050 (`scripts/profile_memory.py`) puts only ~20% of its GPU time in matmul
kernels. The rest is topk (~35%), the neighbour gathers (~23-28%), and the
masked_fill over the similarity tensor (~8%) -- operations that move a lot of memory
and do almost no arithmetic. A FLOP count cannot see them, and no amount of care in
counting FLOPs will make it see them.

So this module models retrieval on two axes: `retrieval()` for matmul FLOPs, and
`retrieval_traffic()` for bytes moved. Quoting the FLOP number alone overstates how
much wall-clock routing actually buys, because the memory path's real cost is on the
axis the FLOP number omits. `benchmark.py` prints both, and labels the FLOP-model
validation table as matmul-only so the agreement there is not mistaken for a
statement about total cost.

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
    retrieval: float = 0.0    # kNN scan + attend over neighbours (MATMUL ONLY)
    adaptive: float = 0.0     # adaptive stack (already scaled by capacity)
    tail: float = 0.0         # dense tail layers
    routing: float = 0.0      # router + aux-predictor projections, all tokens
    head: float = 0.0         # lm_head (+ tied embedding read)

    @property
    def layers(self) -> float:
        """Everything except the output head -- the 'non-embedding' FLOPs."""
        return (self.trunk + self.memory_layer + self.retrieval + self.adaptive
                + self.tail + self.routing)

    @property
    def total(self) -> float:
        return self.layers + self.head

    def as_dict(self) -> dict:
        return dict(
            trunk=self.trunk, memory_layer=self.memory_layer, retrieval=self.retrieval,
            adaptive=self.adaptive, tail=self.tail, routing=self.routing,
            head=self.head, layers=self.layers, total=self.total,
        )


@dataclass
class TrafficBreakdown:
    """Bytes moved per *retrieving* token by the retrieval path, forward pass.

    A separate type from FlopBreakdown on purpose. These operations have no
    meaningful FLOP count -- topk does comparisons, gather does address arithmetic --
    so folding them into a FLOP total would be inventing arithmetic that never
    happens. Bytes is the unit that predicts their cost, because they are all
    bandwidth-bound.

    Fields follow the order of operations in `KVMemoryBank.read`.
    """
    scan_write: float = 0.0    # writing the (n_head, M) similarity row
    mask: float = 0.0          # masked_fill over that row
    select: float = 0.0        # topk's radix-select passes over that row
    gather: float = 0.0        # materialising the two (n_head, k, head_dim) neighbours
    attend: float = 0.0        # re-reading those neighbours for sim_live and the output
    keys_stream: float = 0.0   # amortised share of streaming the normalised bank

    @property
    def per_row(self) -> float:
        """Terms that scale with bank size M and are independent of k."""
        return self.scan_write + self.mask + self.select

    @property
    def per_neighbour(self) -> float:
        """Terms that scale with the neighbour count k."""
        return self.gather + self.attend

    @property
    def total(self) -> float:
        return self.per_row + self.per_neighbour + self.keys_stream

    def as_dict(self) -> dict:
        return dict(
            scan_write=self.scan_write, mask=self.mask, select=self.select,
            gather=self.gather, attend=self.attend, keys_stream=self.keys_stream,
            per_row=self.per_row, per_neighbour=self.per_neighbour, total=self.total,
        )


class FlopModel:
    """Analytic cost model for an AMTConfig: matmul FLOPs, plus retrieval bytes.

    See the module docstring for why retrieval needs the second axis.
    """

    # Passes over the (n_head, M) similarity row, in units of that row's size.
    # Calibrated against scripts/profile_memory.py rather than derived: PyTorch's
    # multi-block topk runs digit-count, within-k-count, cumsum and gather kernels,
    # which is ~3 effective passes, and masked_fill is out-of-place so it reads and
    # writes (an in-place masked_fill_ would make _MASK_PASSES 1.0).
    _MASK_PASSES = 2.0
    _TOPK_PASSES = 3.0
    # Passes over the (n_head, k, head_dim) neighbour tensors: two gathers, each a
    # scattered read plus a contiguous write, then one re-read each for sim_live and
    # the output einsum.
    _GATHER_PASSES = 4.0
    _ATTEND_PASSES = 2.0

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
        """MATMUL FLOPs per *retrieving* token: kNN scan + attend over neighbours.

        Matmul only, and for this path that is roughly a fifth of the real cost --
        see `retrieval_traffic` and the module docstring. Do not quote this as the
        cost of retrieval.

        The scan dominates what *is* here: every query head is compared against all
        M bank entries. This is why `mem_capacity` matters -- routing out 75% of
        tokens cuts it 4x.
        """
        d, M, k = self.c.n_embd, self.c.mem_size, self.c.n_neighbors
        scan = 2.0 * d * M
        # Two einsums over the neighbours, not one: `sim_live` recomputes the top-k
        # similarities so the query gets a gradient, and the output weights the
        # values. Both are (n_head, k, head_dim) contractions costing 2*d*k, and both
        # dispatch to bmm, so FlopCounterMode sees both. Counting one was an
        # undercount of 2*d*k -- ~1.5% of retrieval at the default k.
        attend = 4.0 * d * k
        return scan + attend

    def retrieval_traffic(self, dtype_bytes: int = 2,
                          chunk_size: int = None) -> TrafficBreakdown:
        """Bytes moved per *retrieving* token by the retrieval path, forward pass.

        This is the cost `retrieval()` cannot express. `dtype_bytes` defaults to 2
        because the bank is allocated in the autocast dtype (fp16 or bf16); pass 4
        to model an fp32 bank.

        First-order only: it tracks the tensors big enough to matter and agrees with
        measured `read()` time to roughly 20% on a T4, which is enough to show that
        the bandwidth axis dominates and enough to rank k and mem_size against each
        other. It is not a wall-clock predictor -- use `scripts/profile_memory.py`
        for that.

        Backward roughly doubles these figures: the gathered neighbours, the
        attention weights and the live similarities are all held for backward, while
        the scan itself runs under no_grad and is not recomputed.
        """
        c = self.c
        if not c.use_memory:
            return TrafficBreakdown()

        chunk = chunk_size or c.mem_query_chunk
        row = c.n_head * c.mem_size * dtype_bytes            # the (H, M) similarity row
        nbr = c.n_head * c.n_neighbors * c.head_dim * dtype_bytes  # the (H, k, D) tensors

        return TrafficBreakdown(
            scan_write=row,
            mask=self._MASK_PASSES * row,
            select=self._TOPK_PASSES * row,
            gather=self._GATHER_PASSES * nbr,
            attend=self._ATTEND_PASSES * nbr,
            # The normalised bank is streamed once per chunk, so its cost per token
            # falls as the chunk grows. This is the only term mem_query_chunk moves,
            # which is why raising it measured as a ~3% win and nothing more.
            keys_stream=c.n_head * c.mem_size * c.head_dim * dtype_bytes / chunk,
        )

    def retrieval_intensity(self, dtype_bytes: int = 2) -> float:
        """Arithmetic intensity of the retrieval path: matmul FLOPs per byte moved.

        Low intensity means bandwidth-bound, which is the whole point: compare this
        against a device's ridge point (peak FLOP/s divided by peak bytes/s, order
        200 FLOP/byte on a T4) to see how far the path sits from being compute-bound.
        """
        traffic = self.retrieval_traffic(dtype_bytes).total
        return self.retrieval() / traffic if traffic else float("inf")

    def routing(self) -> float:
        """Router projections per token, for every router in the model.

        Small -- ~0.25% of the total -- but matmuls, so FlopCounterMode counts them
        and leaving them out biased the validation comparison rather than merely
        rounding it. Each TopKTokenRouter carries two Linear(d + cond, 1) heads: the
        router itself and the auxiliary causal predictor.

        Not scaled by capacity: a router has to score every token in order to decide
        which ones to drop.
        """
        c = self.c
        if not c.route_depth and not (c.use_memory and c.route_memory):
            return 0.0

        per_head = lambda width: 2.0 * width * 1.0      # noqa: E731  Linear(width, 1)
        total = 0.0
        if c.use_memory and c.route_memory:             # the memory router, uncoupled
            total += 2 * per_head(c.n_embd)             # w_route + w_aux
        if c.route_depth:
            cond = 1 if c.couple else 0                 # the retrieval-confidence wire
            total += len(c.adaptive_layers) * 2 * per_head(c.n_embd + cond)
        return total

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
            routing=self.routing(),
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
            routing=self.routing(),
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

    def axis_spans(self) -> tuple:
        """(depth_span, memory_span, ratio): FLOPs/token each axis can move.

        The iso-FLOP sweep trades depth capacity against memory capacity at a fixed
        budget, which only works if the two axes can move comparable amounts of
        budget. Sweeping mem_capacity across its whole (0, 1] range shifts exactly
        `retrieval()` FLOPs; sweeping depth_capacity shifts that times the number of
        adaptive layers.

        At the default config the ratio is ~17x, which means the memory axis cannot
        pay for even one grid step on the depth axis and the sweep has almost no
        feasible points. Widening it means a much larger mem_size -- and `retrieval`
        FLOPs are the *cheap* axis of the memory path, so the bank size needed to
        make this figure work would make retrieval's bandwidth cost enormous. That
        tension is a result worth reporting, not a bug to tune away.
        """
        depth = len(self.c.adaptive_layers) * self.layer()
        memory = self.retrieval() if self.c.use_memory else 0.0
        return depth, memory, (depth / memory if memory else float("inf"))

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

        if not out:
            # An empty list here used to propagate silently into an empty figure.
            # It almost always means the two axes are too unequal to trade, not that
            # the caller picked a bad target, so say which.
            depth_span, mem_span, ratio = self.axis_spans()
            raise ValueError(
                f"no iso-FLOP pair at {target_flops/1e6:.2f}M FLOPs/token on a "
                f"{n_points}-point grid. The depth axis spans "
                f"{depth_span/1e6:.2f}M FLOPs/token and the memory axis only "
                f"{mem_span/1e6:.2f}M -- a {ratio:.1f}x asymmetry, so one grid step "
                f"in depth capacity ({depth_span/n_points/1e6:.2f}M) already exceeds "
                f"everything mem_capacity can pay for. Raise mem_size to widen the "
                f"memory axis (and see retrieval_traffic() for what that costs in "
                f"bandwidth), reduce n_points, or narrow max_depth_capacity."
            )
        return out

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        b = self.breakdown()
        d = self.dense_breakdown()
        lines = [
            "matmul FLOPs (the axis FlopCounterMode validates)",
            f"forward FLOPs/token   layers={b.layers/1e6:8.2f}M  head={b.head/1e6:8.2f}M  total={b.total/1e6:8.2f}M",
            f"dense equivalent      layers={d.layers/1e6:8.2f}M  head={d.head/1e6:8.2f}M  total={d.total/1e6:8.2f}M",
            f"savings               layers={1 - b.layers/d.layers:7.1%}   total={1 - b.total/d.total:7.1%}",
            f"head share of dense   {d.head/d.total:7.1%}   <- caps achievable total savings",
            f"exchange rate rho     {self.exchange_rate():.3f} layer-equivalents per retrieval",
        ]
        if self.c.use_memory and self.c.adaptive_layers:
            dspan, mspan, ratio = self.axis_spans()
            warn = "   <- too unequal to trade; see axis_spans()" if ratio > 4 else ""
            lines.append(
                f"iso-FLOP axis spans   depth={dspan/1e6:6.2f}M  memory={mspan/1e6:6.2f}M"
                f"  ratio={ratio:5.1f}x{warn}"
            )
        if self.c.use_memory:
            t = self.retrieval_traffic()
            mc = self.c.effective_mem_capacity
            lines += [
                "",
                "retrieval bytes (the axis FLOPs cannot see -- ~80% of read()'s GPU time)",
                f"bytes/retrieving tok  {t.total/1e3:8.1f}K   "
                f"row(M-scaled)={t.per_row/t.total:5.1%}  "
                f"neighbours(k-scaled)={t.per_neighbour/t.total:5.1%}",
                f"bytes/model tok       {mc * t.total/1e3:8.1f}K   at mem_capacity={mc:.2f}",
                f"arithmetic intensity  {self.retrieval_intensity():8.1f} FLOP/byte   "
                f"<- bandwidth-bound well below a T4's ~200 ridge point",
            ]
        return "\n".join(lines)


def report(config) -> str:
    return FlopModel(config).summary()
