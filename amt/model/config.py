"""Model configuration.

One dataclass drives every variant in the experiment matrix (B1-B7 in
RESEARCH_PLAN.md §7.1), so a baseline is a config file, never a code branch.
"""

from dataclasses import dataclass, asdict, field


@dataclass
class AMTConfig:
    # ---- transformer shape ----------------------------------------------
    block_size: int = 512      # segment length; documents span many segments
    vocab_size: int = 50304    # GPT-2 BPE padded to a multiple of 128
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    bias: bool = True          # bias in Linears/LayerNorms (GPT-2 uses True)

    # ---- layer roles -----------------------------------------------------
    # layers [0, n_trunk)                       -> dense trunk
    # layer  mem_layer                          -> memory layer (always runs)
    # layers (mem_layer, n_layer - n_dense_tail) -> adaptive stack
    # layers [n_layer - n_dense_tail, n_layer)  -> dense tail
    n_trunk: int = 3
    n_dense_tail: int = 1

    # ---- routing ---------------------------------------------------------
    route_depth: bool = True      # False -> every layer is dense (B1, B2)
    use_memory: bool = True       # False -> no memory at all      (B1, B3)
    route_memory: bool = True     # False -> every token retrieves (B2)
    couple: bool = True           # C1 wire: conf -> depth router  (B7 sets False)
    random_route: bool = False    # B4: random scores at matched capacity

    depth_capacity: float = 0.5   # c_l: fraction of tokens through each adaptive layer
    mem_capacity: float = 0.25    # fraction of tokens that query the memory bank

    # ---- memory ----------------------------------------------------------
    mem_size: int = 2048          # M: FIFO bank entries per batch element
    n_neighbors: int = 32         # k in kNN
    # Query-chunk size for the kNN scan, trading VRAM for bandwidth: every chunk
    # re-streams the whole normalised bank (B*H*M*head_dim), so 4 chunks read it 4
    # times. 512 covers a full segment in one pass; the resulting (B, H, T, M)
    # similarity tensor is ~200 MB at B=16 in fp16, which is affordable anywhere the
    # model itself fits. Lower it if a larger mem_size pushes that out of VRAM.
    mem_query_chunk: int = 512
    mem_gate_init: float = -2.0   # gate bias init; sigmoid(-2) ~ 0.12 (risk R2)

    # ---- loss weights ----------------------------------------------------
    lambda_aux: float = 0.05      # causal routing predictor BCE
    lambda_entropy: float = 0.01  # anti-degeneracy bonus, annealed to 0

    # ---- misc ------------------------------------------------------------
    dropout: float = 0.0
    ce_chunks: int = 4            # chunked cross-entropy (see amt.py); 1 = off

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert 0 < self.depth_capacity <= 1.0
        assert 0 < self.mem_capacity <= 1.0
        assert self.n_trunk + self.n_dense_tail < self.n_layer, (
            "no adaptive layers left: n_trunk + n_dense_tail must be < n_layer"
        )
        assert self.mem_layer < self.n_layer - self.n_dense_tail, (
            "memory layer must sit before the dense tail"
        )

    # ---- derived ---------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mem_layer(self) -> int:
        """The memory layer is the first layer after the dense trunk."""
        return self.n_trunk

    @property
    def adaptive_layers(self) -> list:
        """Indices of layers that route. Empty when route_depth is False."""
        if not self.route_depth:
            return []
        return list(range(self.mem_layer + 1, self.n_layer - self.n_dense_tail))

    @property
    def effective_depth_capacity(self) -> float:
        return self.depth_capacity if self.route_depth else 1.0

    @property
    def effective_mem_capacity(self) -> float:
        if not self.use_memory:
            return 0.0
        return self.mem_capacity if self.route_memory else 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            mem_layer=self.mem_layer,
            adaptive_layers=self.adaptive_layers,
            head_dim=self.head_dim,
        )
        return d


# ---------------------------------------------------------------------------
# Named variants: the experiment matrix from RESEARCH_PLAN.md §7.1.
# ---------------------------------------------------------------------------

def _base(**kw) -> AMTConfig:
    return AMTConfig(**kw)


VARIANTS = {
    # B1: dense control -- no routing, no memory. This is nanoGPT at 40M.
    "b1_dense": dict(route_depth=False, use_memory=False),
    # B2: dense + always-on memory (~ Memorizing Transformer).
    "b2_memory_only": dict(route_depth=False, use_memory=True, route_memory=False),
    # B3: depth routing only (~ Mixture-of-Depths).
    "b3_depth_only": dict(route_depth=True, use_memory=False),
    # B4: random routing at matched capacity -- proves the router learned something.
    "b4_random": dict(route_depth=True, use_memory=True, random_route=True),
    # B6: the claim -- joint allocation with the coupling wire live.
    "b6_amt_joint": dict(route_depth=True, use_memory=True, couple=True),
    # B7: coupling wire cut -- isolates contribution C1.
    "b7_uncoupled": dict(route_depth=True, use_memory=True, couple=False),
}


def variant(name: str, **overrides) -> AMTConfig:
    """Build a config for one of the named baselines, with optional overrides."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    kw = dict(VARIANTS[name])
    kw.update(overrides)
    return AMTConfig(**kw)
