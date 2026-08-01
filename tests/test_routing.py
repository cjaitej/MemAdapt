"""Router correctness.

Two silent-failure modes are pinned here:

* `test_router_receives_gradient` -- if the block output is not scaled by the router
  score, the router has no path to the loss. Training proceeds, loss goes down, and
  the routing decisions are frozen at their random initialisation forever. Nothing
  else in the system reports this.
* `test_causal_mode_is_prefix_invariant` -- the top-k selection used in training peeks
  at the whole sequence. The causal path must not, or generation is running a
  different model than the one that was trained.
* `test_entropy_bonus_survives_saturated_scores` -- the anti-saturation term must not
  itself blow up on saturated scores, which is exactly when it is needed.
"""

import pytest
import torch

from amt.model import AMT, TopKTokenRouter, variant
from amt.model.blocks import AdaptiveBlock


def tiny_config(**kw):
    base = dict(n_layer=6, n_embd=64, n_head=4, block_size=32, vocab_size=128,
                n_trunk=1, n_dense_tail=1, mem_size=64, n_neighbors=8,
                mem_query_chunk=16, ce_chunks=1)
    base.update(kw)
    return variant("b6_amt_joint", **base)


# --------------------------------------------------------------------------
# Capacity and shapes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("capacity,T,expected_k", [
    (0.5, 32, 16), (0.25, 32, 8), (1.0, 32, 32), (0.3, 10, 3), (0.01, 8, 1),
])
def test_capacity_is_exact_and_at_least_one(capacity, T, expected_k):
    r = TopKTokenRouter(n_embd=16, capacity=capacity)
    assert r.k_for(T) == expected_k
    out = r(torch.randn(2, T, 16))
    assert out["idx"].shape == (2, expected_k)
    assert out["label"].sum(dim=1).tolist() == [expected_k] * 2


def test_indices_are_sorted():
    """Ascending order keeps token order intact so causal attention inside the routed
    block still means what it says."""
    r = TopKTokenRouter(n_embd=16, capacity=0.5)
    idx = r(torch.randn(4, 32, 16))["idx"]
    assert (idx.diff(dim=1) > 0).all()


def test_selection_follows_the_scores():
    r = TopKTokenRouter(n_embd=8, capacity=0.25)
    out = r(torch.randn(1, 16, 8))
    chosen = out["scores"][0, out["idx"][0]]
    dropped = out["scores"][0][~out["label"][0].bool()]
    assert chosen.min() >= dropped.max()


def test_random_router_still_respects_capacity():
    """Baseline B4: routing must be random but the FLOP budget identical, or the
    comparison against the learned router is not matched."""
    r = TopKTokenRouter(n_embd=8, capacity=0.5, random_route=True)
    out = r(torch.randn(4, 32, 8))
    assert out["label"].sum(dim=1).tolist() == [16] * 4


def test_conditioning_changes_the_decision():
    """The coupling wire (C1) must actually reach the routing decision.

    The conditioning has to *vary across tokens* to matter: top-k compares tokens
    against each other, so a constant conf shifts every score equally and cannot
    change the ranking. Per-token conf is also the real case -- retrieval confidence
    differs token by token.
    """
    torch.manual_seed(0)
    r = TopKTokenRouter(n_embd=8, capacity=0.5, cond_dim=1)
    torch.nn.init.normal_(r.w_route.weight, std=1.0)
    x = torch.randn(2, 32, 8)
    a = r(x, cond=torch.zeros(2, 32, 1))["label"]
    b = r(x, cond=torch.randn(2, 32, 1) * 5.0)["label"]
    assert not torch.equal(a, b), "conf does not influence routing; C1 is not wired up"


def test_missing_cond_is_an_error_not_a_silent_zero():
    r = TopKTokenRouter(n_embd=8, capacity=0.5, cond_dim=1)
    with pytest.raises(AssertionError):
        r(torch.randn(2, 8, 8))


# --------------------------------------------------------------------------
# Gradient flow -- the silent killer
# --------------------------------------------------------------------------

def test_router_receives_gradient():
    torch.manual_seed(0)
    cfg = tiny_config()
    block = AdaptiveBlock(cfg, layer_idx=2)
    x = torch.randn(2, 32, cfg.n_embd)
    out, _ = block(x, conf=torch.randn(2, 32))
    out.sum().backward()
    g = block.router.w_route.weight.grad
    assert g is not None and g.abs().sum() > 0, (
        "router got no gradient -- the block output is not being scaled by the "
        "router score, and routing will never learn"
    )


def test_every_router_in_the_model_learns():
    torch.manual_seed(0)
    model = AMT(tiny_config())
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(0, 128, (2, 32))
    bank = model.make_bank(2, "cpu", dtype=torch.float32)
    _, _, _, kv = model(x, bank=bank)
    model.write_memory(bank, kv)
    _, losses, _, _ = model(x, targets=y, bank=bank)
    losses["total"].backward()

    dead = [n for n, p in model.named_parameters()
            if "w_route" in n and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f"routers with no gradient: {dead}"


def test_aux_head_does_not_backprop_into_the_model():
    """The aux predictor trains on detached features; if it leaked gradient it could
    reshape the representations to make its own prediction task easier."""
    torch.manual_seed(0)
    r = TopKTokenRouter(n_embd=8, capacity=0.5)
    x = torch.randn(2, 16, 8, requires_grad=True)
    out = r(x)
    r.aux_loss(out).backward()
    assert x.grad is None or x.grad.abs().sum() == 0


# --------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------

def test_training_mode_is_admittedly_non_causal():
    """Documents the known gap rather than pretending it away: under top-k, changing
    a late token can change an early token's routing. This is why the aux head and
    the causal path exist.

    Swept over perturbations rather than asserting on one: any single perturbation
    may happen to leave the early ranking intact, so a fixed seed would make this
    test flaky in whichever direction the seed fell.
    """
    torch.manual_seed(0)
    r = TopKTokenRouter(n_embd=8, capacity=0.5)
    torch.nn.init.normal_(r.w_route.weight, std=1.0)
    x = torch.randn(1, 16, 8)
    baseline = r(x)["label"][0, :8]

    changed = False
    for _ in range(20):
        x2 = x.clone()
        x2[0, 8:] = torch.randn(8, 8) * 10
        if not torch.equal(r(x2)["label"][0, :8], baseline):
            changed = True
            break
    assert changed, "expected top-k routing to be non-causal"


def test_causal_mode_is_prefix_invariant():
    """The real test: in causal mode, logits at position t must not move when tokens
    after t change. Covers attention masking, routing, and memory in one assertion."""
    torch.manual_seed(0)
    model = AMT(tiny_config()).eval()
    for m in model.modules():
        if isinstance(m, TopKTokenRouter):
            m.threshold.fill_(0.5)

    B, T, split = 1, 32, 16
    x = torch.randint(0, 128, (B, T))
    x2 = x.clone()
    x2[:, split:] = torch.randint(0, 128, (B, T - split))

    bank = model.make_bank(B, "cpu", dtype=torch.float32)
    with torch.no_grad():
        prev = torch.randint(0, 128, (B, T))
        _, _, _, kv = model(prev, bank=bank, causal=True)
        model.write_memory(bank, kv)

        a = model(x, bank=bank, causal=True, return_logits=True)[0]
        b = model(x2, bank=bank, causal=True, return_logits=True)[0]

    assert torch.allclose(a[:, :split], b[:, :split], atol=1e-4), (
        "future tokens changed past logits -- the causal path leaks"
    )


def test_causal_mode_matches_capacity_after_calibration():
    torch.manual_seed(0)
    model = AMT(tiny_config()).eval()
    batches = [torch.randint(0, 128, (2, 32)) for _ in range(4)]
    thresholds = model.calibrate_routers(batches, device="cpu")
    assert thresholds, "no routers were calibrated"

    _, _, stats, _ = model(batches[0], causal=True)
    for key, rate in stats.items():
        if key.startswith("rate/depth"):
            assert 0.2 < rate < 0.8, f"{key} rate {rate:.2f} far from capacity 0.5"


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["b1_dense", "b2_memory_only", "b3_depth_only",
                                  "b4_random", "b6_amt_joint", "b7_uncoupled"])
def test_every_variant_runs_forward_and_backward(name):
    torch.manual_seed(0)
    cfg = variant(name, n_layer=6, n_embd=64, n_head=4, block_size=32,
                  vocab_size=128, n_trunk=1, n_dense_tail=1, mem_size=64,
                  n_neighbors=8, mem_query_chunk=16, ce_chunks=2)
    model = AMT(cfg)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(0, 128, (2, 32))
    bank = model.make_bank(2, "cpu", dtype=torch.float32) if cfg.use_memory else None
    _, losses, stats, kv = model(x, targets=y, bank=bank)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    expected = cfg.n_layer - len(cfg.adaptive_layers) + len(cfg.adaptive_layers) * cfg.depth_capacity
    assert stats["layers_per_token"] == pytest.approx(expected, abs=0.01)


def test_uncoupled_variant_has_no_conf_wire():
    """B7 must genuinely lack the coupling input, not merely be fed zeros."""
    cfg = variant("b7_uncoupled", n_layer=6, n_embd=64, n_head=4, block_size=32,
                  vocab_size=128, n_trunk=1, n_dense_tail=1)
    model = AMT(cfg)
    for m in model.modules():
        if isinstance(m, AdaptiveBlock):
            assert m.router.cond_dim == 0
            assert m.router.w_route.in_features == cfg.n_embd

    coupled = AMT(variant("b6_amt_joint", n_layer=6, n_embd=64, n_head=4,
                          block_size=32, vocab_size=128, n_trunk=1, n_dense_tail=1))
    for m in coupled.modules():
        if isinstance(m, AdaptiveBlock):
            assert m.router.w_route.in_features == cfg.n_embd + 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_entropy_bonus_survives_saturated_scores(dtype):
    """Saturated scores are the case the entropy bonus exists to prevent.

    bf16 cannot represent 1 - 1e-5 (it rounds to 1.0), so a clamp performed in the
    score dtype leaves log(1 - p) = log(0) and the loss becomes NaN. A pretrained
    backbone saturates these routers on the very first step, so this is not a
    late-training corner case -- see amt/model/retrofit.py.
    """
    router = TopKTokenRouter(n_embd=8, capacity=0.5)
    out = {"scores": torch.tensor([[1.0, 0.0, 0.5, 1.0]], dtype=dtype)}
    bonus = router.entropy_bonus(out)
    assert torch.isfinite(bonus), f"entropy bonus is {bonus} in {dtype}"
    assert bonus >= 0.0


def test_entropy_bonus_still_rewards_uncertainty():
    """Guard the fix: the fp32 cast must not flatten the term into a constant."""
    router = TopKTokenRouter(n_embd=8, capacity=0.5)
    uncertain = router.entropy_bonus({"scores": torch.full((1, 4), 0.5)})
    confident = router.entropy_bonus({"scores": torch.full((1, 4), 0.99)})
    assert uncertain > confident


# --------------------------------------------------------------------------
# route_attention=False: route the contribution, keep the context
# --------------------------------------------------------------------------

def test_dense_attention_mode_still_trains_the_router():
    """The router must stay on the backward path when attention is not routed."""
    cfg = tiny_config(route_attention=False)
    block = AdaptiveBlock(cfg, layer_idx=2)
    x = torch.randn(2, 32, cfg.n_embd)
    out, _ = block(x, conf=torch.randn(2, 32))
    out.sum().backward()
    g = block.router.w_route.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_dense_attention_leaves_dropped_tokens_untouched():
    """A token the router rejects must come out of the block exactly as it went in."""
    torch.manual_seed(0)
    cfg = tiny_config(route_attention=False, depth_capacity=0.5)
    block = AdaptiveBlock(cfg, layer_idx=2).eval()
    x = torch.randn(1, 32, cfg.n_embd)
    with torch.no_grad():
        y, out = block(x, conf=torch.zeros(1, 32))
    dropped = ~out["mask"][0]
    assert dropped.any(), "fixture must drop something"
    assert torch.allclose(y[0][dropped], x[0][dropped], atol=1e-6)


def test_dense_attention_sees_the_whole_context():
    """The distinguishing property: a dropped token still shapes what others read.

    Under the gather path a rejected token vanishes from the block's attention, so
    perturbing it cannot move any other token's output. Keeping attention dense is
    precisely the choice to let it -- which is what a pretrained backbone needs,
    since it learned its attention over the full context.

    The routing is hand-made rather than taken from the router: perturbing a token
    also changes its own score, which can flip the top-k membership, and a selection
    change would move the outputs for reasons that have nothing to do with attention.
    """
    from amt.model.routers import gather_tokens, scatter_add_tokens

    torch.manual_seed(0)
    cfg = tiny_config(depth_capacity=0.5)
    block = AdaptiveBlock(cfg, layer_idx=2).eval()
    T, keep_at, drop_at, look_at = 32, 0, 1, 2      # keep evens, drop odds

    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[0, ::2] = True
    idx = mask[0].nonzero().flatten().unsqueeze(0)
    scores = torch.full((1, T), 0.9)
    assert mask[0, look_at] and not mask[0, drop_at]

    x = torch.randn(1, T, cfg.n_embd)
    x2 = x.clone()
    x2[0, drop_at] += 5.0

    def gather_path(inp):
        xs = gather_tokens(inp, idx)
        delta = block.block.delta(xs) * scores.gather(1, idx).unsqueeze(-1)
        return scatter_add_tokens(inp, idx, delta)

    def dense_path(inp):
        return block._dense_attention_forward(
            inp, {"mask": mask, "scores": scores})

    with torch.no_grad():
        gathered = (gather_path(x2) - gather_path(x))[0, look_at].abs().max()
        dense = (dense_path(x2) - dense_path(x))[0, look_at].abs().max()

    assert gathered == 0, "gather path should not see the dropped token"
    assert dense > 0, "dense-attention path must still attend to it"
