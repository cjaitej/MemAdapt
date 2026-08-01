"""Model configuration.

One dataclass drives every arm of the experiment (see VARIANTS at the bottom), so a
baseline is a config file, never a code branch. The four arms the project compares are
`dense`, `random`, `fixed` and `adaptive`; everything else here is a knob those four
share.
"""

from dataclasses import dataclass, asdict


@dataclass
class AdaptiveGPTConfig:
    # ---- transformer shape ----------------------------------------------
    block_size: int = 512      # context length
    vocab_size: int = 50304    # GPT-2 BPE padded to a multiple of 128
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    bias: bool = True          # bias in Linears/LayerNorms (GPT-2 uses True)
    dropout: float = 0.0

    # ---- early exit -------------------------------------------------------
    # "dense"    : every token runs every layer. The control arm, and Stage 1.
    # "adaptive" : learned per-token exit. The method.
    # "random"   : exit at random, matched to `random_continue_p`. Baseline that
    #              proves the router learned something beyond hitting a depth target.
    # "fixed"    : every token exits after `fixed_exit_layer`. Baseline that proves
    #              adaptive depth beats the best constant depth.
    exit_mode: str = "adaptive"

    # Layers [0, n_min_layers) always run. Routing on barely-contextualised
    # embeddings is uninformed -- the first few layers have not yet produced a
    # representation that says anything about how hard the token is. This also keeps
    # the minimum depth off zero, which matters because a token that exits at layer 0
    # is a bag-of-words prediction.
    n_min_layers: int = 3

    # Hard-gate threshold on the cumulative continue probability. 0.5 is the
    # calibration-free choice and is what `depth_per_token` reports against.
    exit_threshold: float = 0.5

    # ---- router -----------------------------------------------------------
    router_hidden: int = 64        # Linear(d, 64) -> ReLU -> Linear(64, 1)
    # Bias init on the router's output layer, i.e. how likely a token is to CONTINUE
    # at step 0. sigmoid(4) = 0.982, so a 12-layer stack starts at ~11.8 expected
    # layers and the model begins life as (almost exactly) the dense function it was
    # initialised from. This matters enormously for the retrofit: a pretrained
    # backbone that starts by dropping half its depth is a damaged model long before
    # any router has learned anything, and the run never recovers. Lower it if the
    # routers refuse to move -- a saturated sigmoid passes little gradient.
    router_bias_init: float = 4.0

    # Straight-through hard gating vs. the soft relaxation.
    #   True  : forward uses the hard {0,1} gate (exactly what inference does),
    #           backward uses the smooth product of sigmoids. No train/test gap.
    #   False : forward uses the soft cumulative probability. Smoother early on,
    #           but then the trained model is not the model that gets benchmarked.
    straight_through: bool = True

    # ---- exit targets (Stage 2 supervision) --------------------------------
    # How the "this token has converged" label is derived from a dense forward pass.
    #   "delta" : relative remaining change in the residual stream,
    #             r_l = ||h_L - h_l|| / ||h_L||. Cheap. This is the plan's Delta rule.
    #   "kl"    : KL( p(.|h_L) || p(.|h_l) ) through the shared head. Measures what
    #             actually matters -- whether the prediction would change -- at the
    #             cost of an extra vocab-sized projection per layer.
    target_type: str = "delta"
    target_tau: float = 0.05       # exit where the remaining change is below this

    # ---- baseline arms ----------------------------------------------------
    # exit_mode="fixed": run layers [0, fixed_exit_layer). 0 resolves to two thirds of
    # the stack, so `--exit-mode fixed` is usable on any depth without a second flag;
    # scripts/compare.py overrides it with whatever depth the adaptive arm reached,
    # which is the only setting that makes it a control rather than a curiosity.
    fixed_exit_layer: int = 0
    random_continue_p: float = 0.9  # exit_mode="random": per-layer continue prob

    # ---- loss weights -----------------------------------------------------
    lambda_router: float = 0.1     # BCE against the derived exit targets
    lambda_depth: float = 0.01     # penalty on expected depth, normalised to [0, 1]

    # ---- inference --------------------------------------------------------
    # What an exited token contributes to LATER layers' attention.
    #   "stale" : its frozen hidden state is re-projected to k/v at every later
    #             layer, so a still-active token attends over the full context.
    #             Exact -- the compact path reproduces the dense path to numerical
    #             tolerance, which is what `test_compact_matches_dense` pins.
    #   "drop"  : it disappears from later layers entirely. Cheaper (the k/v
    #             projections go too) but a different function, and a pretrained
    #             backbone has never seen a subsampled context.
    exited_as_keys: str = "stale"
    # Active-token counts are rounded up to a multiple of this in the compact path.
    # The count varies per batch, and a fresh shape means a fresh torch.compile
    # (~4 min on the target GPU). Bucketing caps the number of distinct shapes at a
    # handful instead of one per step. 0 disables it.
    compact_bucket: int = 64

    # ---- misc -------------------------------------------------------------
    ce_chunks: int = 4             # chunked cross-entropy (see adaptive_gpt.py)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.exit_mode in ("dense", "adaptive", "random", "fixed"), \
            f"unknown exit_mode {self.exit_mode!r}"
        assert self.target_type in ("delta", "kl"), \
            f"unknown target_type {self.target_type!r}"
        assert self.exited_as_keys in ("stale", "drop"), \
            f"unknown exited_as_keys {self.exited_as_keys!r}"
        assert 0 < self.n_min_layers < self.n_layer, \
            "n_min_layers must be positive and leave at least one routable layer"
        if self.fixed_exit_layer <= 0:
            # Resolved here rather than at the use site so `to_dict` and the saved
            # config record the depth the run actually used.
            self.fixed_exit_layer = max(self.n_min_layers,
                                        round(2 * self.n_layer / 3))
        if self.exit_mode == "fixed":
            assert self.n_min_layers <= self.fixed_exit_layer <= self.n_layer, (
                f"fixed_exit_layer {self.fixed_exit_layer} outside "
                f"[{self.n_min_layers}, {self.n_layer}]"
            )

    # ---- derived ---------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def router_layers(self) -> list:
        """Layers that carry an exit router.

        A router sits AFTER block l and decides whether the token enters block l+1,
        so the first decision is made after block n_min_layers-1 and the last after
        block n_layer-2. Routers exist only in "adaptive" mode -- the other arms make
        the same decision without parameters.
        """
        if self.exit_mode != "adaptive":
            return []
        return list(range(self.n_min_layers - 1, self.n_layer - 1))

    @property
    def min_depth(self) -> int:
        """Layers every token pays for, whatever the routing decides."""
        if self.exit_mode == "dense":
            return self.n_layer
        if self.exit_mode == "fixed":
            return self.fixed_exit_layer
        return self.n_min_layers

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(head_dim=self.head_dim, router_layers=self.router_layers,
                 min_depth=self.min_depth)
        return d


# ---------------------------------------------------------------------------
# The four arms. `fixed_exit_layer` and `random_continue_p` are set at run time to
# match the depth the adaptive arm actually reaches -- scripts/compare.py reads
# avg_depth off the adaptive run and matches the baselines to it. Matching depth is
# the entire point of those two arms; against an unmatched baseline they measure
# nothing.
# ---------------------------------------------------------------------------

VARIANTS = {
    "dense":    dict(exit_mode="dense"),
    "random":   dict(exit_mode="random"),
    "fixed":    dict(exit_mode="fixed"),
    "adaptive": dict(exit_mode="adaptive"),
}


def variant(name: str, **overrides) -> AdaptiveGPTConfig:
    """Build a config for one of the four arms, with optional overrides."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    kw = dict(VARIANTS[name])
    kw.update(overrides)
    return AdaptiveGPTConfig(**kw)
