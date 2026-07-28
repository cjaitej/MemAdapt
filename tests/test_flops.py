"""The cost model, on both of its axes.

`flops.py` has claimed since it was written that it is validated against
`torch.utils.flop_counter` in this file. It was not -- this file did not exist, and
the only check was a print-out in `scripts/benchmark.py` that nobody asserts on.

The tests here pin two separate things:

1. The matmul FLOP count agrees with FlopCounterMode. That is the claim the module
   docstring makes, and it is worth pinning because every efficiency figure in the
   writeup divides by it.
2. The matmul count is *not* the cost of retrieval. Profiling puts ~80% of
   `KVMemoryBank.read`'s GPU time in topk, gather and masked_fill, none of which a
   FLOP counter can see. `test_flop_model_is_blind_to_retrievals_real_cost` exists
   so that nobody re-derives an efficiency claim from FLOPs alone without tripping
   over the reason they should not.
"""

import pytest
import torch

from amt.model import AMT, FlopModel, variant


def tiny(**kw):
    base = dict(n_layer=6, n_embd=64, n_head=4, block_size=32, vocab_size=128,
                n_trunk=1, n_dense_tail=1, mem_size=64, n_neighbors=8,
                mem_query_chunk=16, ce_chunks=2)
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# Axis 1: matmul FLOPs
# --------------------------------------------------------------------------

def count_flops(cfg, B=2):
    """Per-token matmul FLOPs from FlopCounterMode, split by aten op."""
    from torch.utils.flop_counter import FlopCounterMode

    model = AMT(cfg).eval()
    T = cfg.block_size
    x = torch.randint(0, cfg.vocab_size, (B, T))

    bank = model.make_bank(B, "cpu", dtype=torch.float32) if cfg.use_memory else None
    if bank is not None:            # warm the bank so retrieval actually runs
        with torch.no_grad():
            _, _, _, kv = model(x, bank=bank)
        model.write_memory(bank, kv)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x, bank=bank, return_logits=True)
    by_op = {str(op).split(".")[-1]: f / (B * T)
             for op, f in counter.get_flop_counts()["Global"].items()}
    return counter.get_total_flops() / (B * T), by_op


def test_retrieval_matmuls_match_the_profiler_exactly():
    """The retrieval path's matmuls are the part of the model this project claims.

    They land in bmm and nothing else does, so this is an exact check rather than a
    tolerance band -- and it is what caught `attend` counting one einsum instead of
    the two that `read()` actually performs.
    """
    cfg = variant("b6_amt_joint", **tiny())
    _, by_op = count_flops(cfg)
    analytic = FlopModel(cfg).breakdown().retrieval
    assert by_op["bmm"] == pytest.approx(analytic, rel=1e-6)


def test_projection_flops_match_the_profiler():
    """Linear projections dominate, and they are counted as addmm.

    Deliberately not a check on the *total*: FlopCounterMode does not attribute any
    FLOPs to attention on the CPU math backend, so a total-vs-total comparison drifts
    with the backend rather than with the model. Comparing the projection term keeps
    the test measuring flops.py.
    """
    cfg = variant("b6_amt_joint", **tiny())
    _, by_op = count_flops(cfg)
    fm = FlopModel(cfg)
    b = fm.breakdown()

    # Every layer's projections, scaled the way breakdown() scales the layers, plus
    # the routers. Attention and the head are excluded: the head is aten.mm.
    layer_share = (b.trunk + b.memory_layer + b.adaptive + b.tail) / fm.layer()
    expected = layer_share * fm.layer_matmul() + fm.routing()
    assert by_op["addmm"] == pytest.approx(expected, rel=0.01)


def test_head_flops_match_the_profiler():
    cfg = variant("b6_amt_joint", **tiny())
    _, by_op = count_flops(cfg)
    assert by_op["mm"] == pytest.approx(FlopModel(cfg).head(), rel=1e-6)


def test_router_flops_are_counted():
    """Routers are small but they are matmuls, so omitting them biased validation."""
    cfg = variant("b6_amt_joint", **tiny())
    assert FlopModel(cfg).routing() > 0
    assert FlopModel(variant("b1_dense", **tiny())).routing() == 0.0


def test_routing_reduces_matmul_flops():
    dense = FlopModel(variant("b1_dense", **tiny())).breakdown().total
    routed = FlopModel(variant("b3_depth_only", **tiny())).breakdown().total
    assert routed < dense


def test_retrieval_counts_both_neighbour_einsums():
    """read() contracts over the neighbours twice -- sim_live and the output.

    Counting one was a real undercount; this pins the fix.
    """
    cfg = variant("b6_amt_joint", **tiny())
    fm = FlopModel(cfg)
    d, M, k = cfg.n_embd, cfg.mem_size, cfg.n_neighbors
    assert fm.retrieval() == pytest.approx(2.0 * d * M + 4.0 * d * k)


def test_no_memory_means_no_retrieval_cost():
    cfg = variant("b3_depth_only", **tiny())
    assert FlopModel(cfg).breakdown().retrieval == 0.0
    assert FlopModel(cfg).retrieval_traffic().total == 0.0


# --------------------------------------------------------------------------
# Axis 2: bytes moved
# --------------------------------------------------------------------------

def test_flop_model_is_blind_to_retrievals_real_cost():
    """The point of the whole two-axis split.

    Arithmetic intensity far below a GPU's ridge point (order 200 FLOP/byte on a
    T4) means the path is bandwidth-bound, so a FLOP count cannot describe its cost.
    If this ever stops holding, the second axis has become unnecessary -- but check
    the profiler before believing that.
    """
    fm = FlopModel(variant("b6_amt_joint"))       # real config, not tiny
    assert fm.retrieval_intensity() < 20.0, (
        "retrieval looks compute-bound, which contradicts the profile in "
        "scripts/profile_memory.py -- re-measure before trusting FLOPs alone"
    )


def test_neighbour_traffic_scales_with_k_and_row_traffic_does_not():
    """k is the lever on the neighbour tensors; it does not touch the topk scan."""
    small = FlopModel(variant("b6_amt_joint", **tiny(n_neighbors=8))).retrieval_traffic()
    large = FlopModel(variant("b6_amt_joint", **tiny(n_neighbors=32))).retrieval_traffic()

    assert large.per_neighbour == pytest.approx(4 * small.per_neighbour)
    assert large.per_row == small.per_row, "topk scans the bank regardless of k"


def test_row_traffic_scales_with_mem_size():
    small = FlopModel(variant("b6_amt_joint", **tiny(mem_size=64))).retrieval_traffic()
    large = FlopModel(variant("b6_amt_joint", **tiny(mem_size=128))).retrieval_traffic()
    assert large.per_row == pytest.approx(2 * small.per_row)
    assert large.per_neighbour == small.per_neighbour


def test_query_chunk_only_moves_the_streaming_term():
    """Why raising mem_query_chunk measured as ~3% and nothing more.

    It amortises one re-read of the normalised bank. It cannot touch topk, the
    gathers, or the masked_fill, which is where the time actually is.
    """
    a = FlopModel(variant("b6_amt_joint", **tiny(mem_query_chunk=16))).retrieval_traffic()
    b = FlopModel(variant("b6_amt_joint", **tiny(mem_query_chunk=32))).retrieval_traffic()

    assert b.keys_stream == pytest.approx(a.keys_stream / 2)
    assert (b.per_row, b.per_neighbour) == (a.per_row, a.per_neighbour)
    assert b.total < a.total


def test_traffic_scales_with_dtype_width():
    cfg = variant("b6_amt_joint", **tiny())
    half = FlopModel(cfg).retrieval_traffic(dtype_bytes=2).total
    full = FlopModel(cfg).retrieval_traffic(dtype_bytes=4).total
    assert full == pytest.approx(2 * half)


# --------------------------------------------------------------------------
# Budget plumbing
# --------------------------------------------------------------------------

def test_iso_flop_targets_below_the_floor_raise():
    fm = FlopModel(variant("b6_amt_joint", **tiny()))
    floor, _, _ = fm.budget_range()
    with pytest.raises(ValueError, match="below the unroutable floor"):
        fm.iso_flop_configs(floor * 0.5)


def test_iso_flop_pairs_that_exist_hit_their_target():
    """When the sweep does find pairs, they must actually cost the target."""
    fm = FlopModel(variant("b6_amt_joint", **tiny()))
    floor, ceil, _ = fm.budget_range()
    # Near the ceiling both capacities are high, which is where pairs are feasible.
    pairs = fm.iso_flop_configs(floor + 0.92 * (ceil - floor), n_points=20)
    assert pairs
    for dc, mc in pairs:
        got = fm.breakdown(depth_capacity=dc, mem_capacity=mc).total
        assert got == pytest.approx(floor + 0.92 * (ceil - floor), rel=1e-3)


def test_the_iso_flop_axes_are_too_unequal_to_trade():
    """The finding that makes RESEARCH_PLAN figure 1 undeliverable as designed.

    Sweeping mem_capacity over its entire range moves ~1.6M FLOPs/token; sweeping
    depth_capacity moves ~27.5M, because there are 7 adaptive layers and only one
    memory layer. So a fixed-FLOP trade between them has almost no feasible points.

    Widening the memory axis means a much larger mem_size, and mem_size is what
    drives the topk scan -- the dominant *bandwidth* cost. The two axes therefore
    cannot both be made comparable and affordable, which is the substantive result.
    """
    fm = FlopModel(variant("b6_amt_joint"))         # real config
    depth_span, mem_span, ratio = fm.axis_spans()
    assert ratio > 10, (
        f"depth spans {depth_span/1e6:.1f}M and memory {mem_span/1e6:.1f}M "
        f"({ratio:.1f}x) -- if this has narrowed, revisit the iso-FLOP figure"
    )


def test_infeasible_iso_flop_sweep_explains_itself():
    """An empty sweep used to propagate silently into an empty figure."""
    fm = FlopModel(variant("b6_amt_joint"))
    floor, ceil, _ = fm.budget_range()
    with pytest.raises(ValueError, match="asymmetry"):
        fm.iso_flop_configs((floor + ceil) / 2, n_points=5)
