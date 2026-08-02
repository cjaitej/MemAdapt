"""The early-exit mechanism: what it computes, and what it must never do."""

import pytest
import torch

from conftest import SMALL, build, randomise_routers


# ---------------------------------------------------------------------------
# The invariant the whole project rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bucket", [0, 4, 8, 64])
@pytest.mark.parametrize("keys", ["stale", "drop"])
def test_compact_matches_dense(tokens, bucket, keys):
    """The fast path and the reference path must be the same function.

    This is the most important test in the suite. Every efficiency number the project
    reports is measured on `forward_compact`; every quality number could be measured
    on either. If they diverge, the speedup is not a speedup -- it is a second model
    that nobody evaluated, being credited with the first model's perplexity.

    Run across bucket sizes because bucketing is where the padding lives, and padding
    is where an off-by-one silently corrupts one row in a batch.
    """
    model, cfg = build("adaptive", compact_bucket=bucket, exited_as_keys=keys)
    randomise_routers(model)

    with torch.no_grad():
        dense_logits, _, dense_stats = model(tokens, return_logits=True)
        model.config.exited_as_keys = keys
        compact_logits, _, compact_stats = model.forward_compact(tokens,
                                                                 return_logits=True)

    if keys == "stale":
        assert torch.allclose(dense_logits, compact_logits, atol=1e-5), \
            "compact path diverged from the gated path"
    assert dense_stats["depth"] == pytest.approx(float(compact_stats["depth"]))


def test_compact_actually_exits(tokens):
    """The equivalence test is worthless if nothing ever exits."""
    model, cfg = build("adaptive")
    randomise_routers(model)
    depths = model.token_depths(tokens)
    assert depths.min() < cfg.n_layer, "no token exited early; the test is vacuous"
    assert depths.float().std() > 0.3, "every token got the same depth"


# ---------------------------------------------------------------------------
# Exit semantics
# ---------------------------------------------------------------------------

def test_exit_is_monotone(tokens):
    """Once a token stops, it stays stopped.

    Monotonicity is what makes the saving real: the survivor set only ever shrinks, so
    `forward_compact` never has to grow the batch back. A router that could revive an
    exited token would still reduce FLOPs on paper and would be strictly slower in
    practice.
    """
    model, cfg = build("adaptive")
    randomise_routers(model)

    x = model._embed(tokens)
    alive_p = x.new_ones(tokens.shape)
    history = []
    for l, block in enumerate(model.transformer.h):
        alive_h = (alive_p > cfg.exit_threshold)
        x = block(x) if l < cfg.n_min_layers else x + alive_h.unsqueeze(-1) * block.delta(x)
        if model._routes_at(l):
            p, _ = model._continue_prob(l, x)
            alive_p = alive_p * p
            history.append((alive_p > cfg.exit_threshold).clone())

    for earlier, later in zip(history, history[1:]):
        assert not (later & ~earlier).any(), "a token that had exited came back"


def test_depth_respects_bounds(tokens):
    model, cfg = build("adaptive")
    randomise_routers(model, scale=4.0)
    d = model.token_depths(tokens)
    assert d.min() >= cfg.n_min_layers
    assert d.max() <= cfg.n_layer


def test_dense_mode_is_a_plain_transformer(tokens):
    model, cfg = build("dense")
    assert len(model.routers) == 0
    with torch.no_grad():
        _, _, stats = model(tokens)
    assert float(stats["depth"]) == cfg.n_layer
    assert float(stats["exit_frac"]) == 0.0


def test_routing_disabled_reproduces_dense(tokens):
    """Stage 1 must be exactly the dense model, not approximately.

    `--stage dense` builds the full adaptive model with the gates pinned open, and its
    checkpoint is then used as BOTH the dense baseline and the Stage 2 initialisation.
    If pinning the gates is not exact, the baseline is a slightly different model from
    the one the comparison claims it is.
    """
    model, cfg = build("adaptive")
    randomise_routers(model)
    dense, _ = build("dense")
    dense.load_state_dict({k: v for k, v in model.state_dict().items()
                           if not k.startswith("routers.")}, strict=False)

    model.set_routing_enabled(False)
    with torch.no_grad():
        a, _, sa = model(tokens, return_logits=True)
        b, _, sb = dense(tokens, return_logits=True)
    assert torch.allclose(a, b, atol=1e-6)
    assert float(sa["depth"]) == float(sb["depth"]) == cfg.n_layer


def test_fixed_arm_exits_where_told(tokens):
    for layer in (2, 4, 6):
        model, cfg = build("fixed", fixed_exit_layer=layer)
        d = model.token_depths(tokens)
        assert (d == layer).all(), f"fixed@{layer} gave depths {d.unique().tolist()}"


def test_random_arm_is_reproducible(tokens):
    """A baseline that changes between two evaluations cannot be compared against."""
    model, _ = build("random", random_continue_p=0.6)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(model.token_depths(tokens, generator=g1),
                       model.token_depths(tokens, generator=g2))


def test_random_arm_hits_its_rate(tokens):
    shallow, cfg = build("random", random_continue_p=0.2)
    deep, _ = build("random", random_continue_p=0.95)
    g = torch.Generator().manual_seed(0)
    assert shallow.token_depths(tokens, generator=g).mean() < \
        deep.token_depths(tokens, generator=g).mean()


# ---------------------------------------------------------------------------
# Causality -- the property that made early exit worth choosing
# ---------------------------------------------------------------------------

def test_exit_decision_is_prefix_invariant(tokens):
    """A token's exit depth must not depend on tokens that come after it.

    This is the structural advantage of early exit over per-layer top-k routing, and
    it is worth pinning rather than assuming: top-k over the sequence would fail this
    test, which is exactly why that design needed a separate causal predictor head and
    a calibrated threshold before any generation result meant anything.
    """
    model, cfg = build("adaptive")
    randomise_routers(model)
    cut = SMALL["block_size"] // 2

    full = model.token_depths(tokens)
    prefix = model.token_depths(tokens[:, :cut])
    assert torch.equal(full[:, :cut], prefix), \
        "an exit decision changed when later tokens were removed"


# ---------------------------------------------------------------------------
# Gradient -- the silent failure
# ---------------------------------------------------------------------------

def test_router_receives_gradient(tokens):
    """Remove the straight-through term and everything still 'works'.

    Training proceeds, the loss goes down, and the routing decisions stay frozen at
    their initialisation forever. Nothing else in the system reports this -- hence
    this test.
    """
    model, cfg = build("adaptive")
    randomise_routers(model)
    model.train()
    y = torch.randint(0, SMALL["vocab_size"], tokens.shape)

    hiddens = model.dense_hidden_states(tokens)
    from agpt.model.targets import exit_targets
    labels, _ = exit_targets(hiddens, cfg, model.transformer.ln_f, model.lm_head)

    _, losses, _ = model(tokens, targets=y, exit_labels=labels)
    losses["total"].backward()

    for name, p in model.routers.named_parameters():
        assert p.grad is not None, f"routers.{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"routers.{name} got a zero gradient"


def test_lm_loss_alone_reaches_the_router(tokens):
    """The straight-through gate, isolated: no BCE, no depth penalty, just the LM."""
    model, cfg = build("adaptive")
    randomise_routers(model)
    model.train()
    model.set_lambda_router(0.0)
    model.set_lambda_depth(0.0)
    y = torch.randint(0, SMALL["vocab_size"], tokens.shape)

    _, losses, _ = model(tokens, targets=y)
    losses["total"].backward()
    total = sum(p.grad.abs().sum() for p in model.routers.parameters()
                if p.grad is not None)
    assert total > 0, "the LM loss never reached the routers"


def test_hard_gate_forward_value_is_hard(tokens):
    """Straight-through must not leak the soft value into the forward pass."""
    from agpt.model.router import hard_gate
    p = torch.tensor([0.1, 0.4, 0.6, 0.9], requires_grad=True)
    g = hard_gate(p, 0.5, straight_through=True)
    assert torch.equal(g.detach(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    g.sum().backward()
    assert torch.equal(p.grad, torch.ones(4))


def test_exited_tokens_do_not_update_the_backbone(tokens):
    """A block must get no gradient from tokens that skipped it.

    If it does, the model is being trained on computation it will not perform at
    inference -- the classic way a "fast" model turns out to have been trained dense.
    """
    model, cfg = build("adaptive")
    with torch.no_grad():                       # force EVERY token to exit at n_min
        for r in model.routers.values():
            r.proj.bias.fill_(-20.0)
            r.proj.weight.zero_()
            r.fc.weight.zero_()
            r.fc.bias.zero_()
    model.train()
    y = torch.randint(0, SMALL["vocab_size"], tokens.shape)
    _, losses, _ = model(tokens, targets=y)
    losses["lm"].backward()

    last = model.transformer.h[cfg.n_layer - 1]
    grads = [p.grad for p in last.parameters() if p.grad is not None]
    assert grads, "no gradients recorded at all"
    assert all(g.abs().sum() == 0 for g in grads), \
        "the last block got gradient from tokens that never entered it"


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

def test_dropout_is_active_in_train_and_off_in_eval(tokens):
    """A config field that nothing reads is worse than no field at all.

    `dropout` sat in the config unused for the whole of the previous design, so a run
    launched with `--dropout 0.1` would have trained with none and the config.json
    would have said otherwise. This pins that it reaches the model, and that it is off
    at eval -- attention dropout goes to SDPA as a probability rather than a module,
    so it does not consult `self.training` unless something makes it.
    """
    model, _ = build("adaptive", dropout=0.5)

    model.eval()
    with torch.no_grad():
        a, _, _ = model(tokens, return_logits=True)
        b, _, _ = model(tokens, return_logits=True)
    assert torch.equal(a, b), "eval-mode forward was not deterministic"

    model.train()
    torch.manual_seed(0)
    c, _, _ = model(tokens, return_logits=True)
    torch.manual_seed(1)
    d, _, _ = model(tokens, return_logits=True)
    assert not torch.allclose(c, d), "train-mode forward showed no dropout"


def test_zero_dropout_changes_nothing(tokens):
    model, _ = build("adaptive", dropout=0.0)
    model.train()
    with torch.no_grad():
        a, _, _ = model(tokens, return_logits=True)
        b, _, _ = model(tokens, return_logits=True)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# The two fixes from the first real training run
# ---------------------------------------------------------------------------

def test_depth_penalty_measures_the_depth_actually_computed(tokens):
    """The depth loss must not be satisfiable without anyone exiting.

    Charging the soft cumulative probability let a run reach a reported 7.66 layers
    while the gate -- which thresholds at 0.5 -- still ran 11.27: moving a token from
    0.99 to 0.6 pays the penalty and changes nothing. Counting the straight-through
    gate instead makes the reported number the real one.
    """
    model, cfg = build("adaptive")
    randomise_routers(model)
    with torch.no_grad():
        _, losses, stats = model(tokens, targets=tokens)
    assert float(losses["depth"]) * cfg.n_layer == pytest.approx(
        float(stats["depth"]), rel=1e-4), \
        "the depth penalty and the reported depth disagree"


def test_depth_penalty_still_reaches_the_router(tokens):
    """A hard forward value must not cost the penalty its gradient."""
    model, cfg = build("adaptive")
    randomise_routers(model)
    model.train()
    model.set_lambda_router(0.0)
    model.set_lambda_depth(1.0)
    _, losses, _ = model(tokens, targets=tokens)
    losses["depth"].backward()
    total = sum(p.grad.abs().sum() for p in model.routers.parameters()
                if p.grad is not None)
    assert total > 0, "the depth penalty produced no router gradient"


def test_label_mask_restricts_the_bce(tokens):
    """Positions outside the mask must contribute nothing to the router loss."""
    from agpt.model.targets import exit_targets
    model, cfg = build("adaptive")
    randomise_routers(model)
    hiddens = model.dense_hidden_states(tokens)
    labels, _ = exit_targets(hiddens, cfg, model.transformer.ln_f, model.lm_head)

    full = torch.ones(tokens.shape, dtype=torch.bool)
    none_ = torch.zeros(tokens.shape, dtype=torch.bool)
    with torch.no_grad():
        _, a, _ = model(tokens, targets=tokens, exit_labels=labels, label_mask=full)
        _, b, _ = model(tokens, targets=tokens, exit_labels=labels, label_mask=none_)
        _, c, _ = model(tokens, targets=tokens, exit_labels=labels)
    assert float(a["router"]) == pytest.approx(float(c["router"]), rel=1e-5)
    assert float(b["router"]) == 0.0, "an all-false mask still contributed loss"
