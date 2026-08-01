"""The analytic FLOP model, held against PyTorch's profiler.

Every efficiency claim the project makes is computed by `agpt/model/flops.py`, so it
is checked against `torch.utils.flop_counter` rather than trusted.

One thing this comparison cannot cover: `FlopCounterMode` has no formula registered
for the fused kernels `scaled_dot_product_attention` dispatches to, and reports zero
for them. So the comparison runs with the analytic attention term switched off
(`ctx_len=0`) and validates the projections and the output head -- which is where all
the routing-dependent arithmetic lives anyway. Comparing against the full analytic
number would credit the model with an error the size of the entire attention cost.
"""

import pytest
import torch

from agpt.model import AdaptiveGPT, FlopModel, active_fractions, measured_flops, variant
from conftest import randomise_routers

WIDE = dict(n_layer=12, n_head=6, n_embd=384, block_size=256, vocab_size=50304,
            ce_chunks=1, n_min_layers=3)


def wide(exit_mode, **over):
    cfg = variant(exit_mode, **{**WIDE, **over})
    return AdaptiveGPT(cfg).eval(), cfg


@pytest.mark.parametrize("exit_mode,over,depth", [
    ("dense", {}, None),
    ("fixed", dict(fixed_exit_layer=4), 4),
    ("fixed", dict(fixed_exit_layer=6), 6),
    ("fixed", dict(fixed_exit_layer=11), 11),
])
def test_analytic_matches_profiler(exit_mode, over, depth):
    model, cfg = wide(exit_mode, **over)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    fm = FlopModel(cfg)
    analytic = (fm.dense_breakdown(ctx_len=0) if depth is None
                else fm.at_uniform_depth(depth, ctx_len=0)).total
    profiled = measured_flops(model, x, compact=depth is not None)
    assert profiled == pytest.approx(analytic, rel=0.01), \
        f"{exit_mode}: analytic {analytic/1e6:.2f}M vs profiler {profiled/1e6:.2f}M"


def test_analytic_matches_profiler_when_routing(tokens):
    """The adaptive arm, with the activity read off a real forward pass."""
    model, cfg = wide("adaptive")
    randomise_routers(model)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    fracs = active_fractions(model.token_depths(x), cfg.n_layer)
    analytic = FlopModel(cfg).breakdown(fracs, ctx_len=0).total
    profiled = measured_flops(model, x, compact=True)
    # Looser than the fixed-depth cases: the compact path rounds the active count up
    # to `compact_bucket`, so it really does compute a little more than the fractions
    # imply. That overhead is a property of the implementation, not of the model, and
    # it is bounded by the bucket size.
    assert profiled == pytest.approx(analytic, rel=0.10)
    assert profiled >= analytic * 0.98, "profiler came in below the analytic floor"


# ---------------------------------------------------------------------------
# Properties the model must have
# ---------------------------------------------------------------------------

def test_a_layer_nobody_enters_is_free():
    _, cfg = wide("adaptive")
    assert FlopModel(cfg).adaptive_layer(0.0) == 0.0


def test_full_activity_equals_a_dense_layer():
    """The routed decomposition (4d^2 k/v + 20d^2 routed) must re-sum to 24d^2."""
    _, cfg = wide("adaptive")
    fm = FlopModel(cfg)
    assert fm.adaptive_layer(1.0) == pytest.approx(fm.layer())


def test_cost_is_monotone_in_depth():
    _, cfg = wide("adaptive")
    fm = FlopModel(cfg)
    costs = [fm.at_uniform_depth(d).total for d in range(cfg.n_min_layers,
                                                         cfg.n_layer + 1)]
    assert costs == sorted(costs)


def test_drop_is_cheaper_than_stale():
    """The two exit semantics differ in cost, and in the direction claimed."""
    _, stale = wide("adaptive", exited_as_keys="stale")
    _, drop = wide("adaptive", exited_as_keys="drop")
    a = FlopModel(stale).adaptive_layer(0.5)
    b = FlopModel(drop).adaptive_layer(0.5)
    assert b < a


def test_head_dominates_at_this_width():
    """The finding that shortens the whole efficiency axis, pinned as a test.

    At d=384 with the GPT-2 vocabulary the output head is a large fraction of forward
    FLOPs and routing cannot touch it. If a config change ever makes this false, the
    total-FLOP claims get much stronger and the writeup needs updating -- so failing
    here is informative, not annoying.
    """
    _, cfg = wide("dense")
    d = FlopModel(cfg).dense_breakdown()
    assert d.head / d.total > 0.4


def test_active_fractions_from_depths():
    depths = torch.tensor([[3.0, 6.0, 6.0, 12.0]])
    fracs = active_fractions(depths, 12)
    assert fracs[0] == 1.0                      # everyone enters layer 0
    assert fracs[3] == pytest.approx(0.75)      # the depth-3 token has left
    assert fracs[6] == pytest.approx(0.25)      # only the depth-12 token remains
    assert fracs[11] == pytest.approx(0.25)


def test_breakdown_rejects_a_wrong_length():
    _, cfg = wide("adaptive")
    with pytest.raises(ValueError):
        FlopModel(cfg).breakdown([1.0] * (cfg.n_layer - 1))


def test_budget_range_is_a_real_range():
    _, cfg = wide("adaptive")
    floor, ceil, ratio = FlopModel(cfg).budget_range()
    assert 0 < floor < ceil
    assert ratio == pytest.approx(floor / ceil)
    # The head alone keeps the floor well off zero; that is the point of reporting it.
    assert ratio > 0.3
