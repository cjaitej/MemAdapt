"""The GPT-2 -> AMT weight mapping.

Built against a synthetic state dict rather than a real checkpoint: the mapping is
pure bookkeeping (rename, transpose, slice), and testing it should not cost a 500MB
download or a network round trip in CI. `test_from_gpt2_real` covers the download
path and skips unless the checkpoint is already cached.
"""

import pytest
import torch

from amt.model import AMT
from amt.model.config import AMTConfig
from amt.model.retrofit import (
    GPT2_SHAPES, describe, freeze_backbone, from_gpt2, gpt2_config,
    is_new_module, load_gpt2_state_dict,
)


def fake_hf_state(cfg, vocab=None, positions=None):
    """A state dict with GPT2LMHeadModel's keys and shapes, filled deterministically.

    Values are distinct per tensor so a mis-routed copy cannot pass by coincidence --
    an all-zeros or all-ones fixture would let the adaptive-layer rename land on the
    wrong layer undetected.
    """
    vocab = vocab or cfg.vocab_size
    positions = positions or cfg.block_size
    d = cfg.n_embd
    sd, seed = {}, [0]

    # Scaled like a real checkpoint (std 0.02, LayerNorm gains near 1) rather than
    # unit normal. Unscaled weights make activations blow up through the stack, which
    # saturates the router's sigmoid at random and makes any test of what the router
    # multiplier does to the output meaningless.
    def t(*shape, scale=0.02):
        seed[0] += 1
        g = torch.Generator().manual_seed(seed[0])
        return torch.randn(*shape, generator=g) * scale

    sd["transformer.wte.weight"] = t(vocab, d)
    sd["transformer.wpe.weight"] = t(positions, d)
    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        sd[p + "ln_1.weight"] = 1.0 + t(d)
        sd[p + "ln_1.bias"] = t(d)
        sd[p + "attn.c_attn.weight"] = t(d, 3 * d)      # Conv1D: (in, out)
        sd[p + "attn.c_attn.bias"] = t(3 * d)
        sd[p + "attn.c_proj.weight"] = t(d, d)
        sd[p + "attn.c_proj.bias"] = t(d)
        sd[p + "ln_2.weight"] = 1.0 + t(d)
        sd[p + "ln_2.bias"] = t(d)
        sd[p + "mlp.c_fc.weight"] = t(d, 4 * d)
        sd[p + "mlp.c_fc.bias"] = t(4 * d)
        sd[p + "mlp.c_proj.weight"] = t(4 * d, d)
        sd[p + "mlp.c_proj.bias"] = t(d)
        sd[p + "attn.bias"] = torch.ones(1, 1, positions, positions)   # buffer
    sd["transformer.ln_f.weight"] = 1.0 + t(d)
    sd["transformer.ln_f.bias"] = t(d)
    sd["lm_head.weight"] = sd["transformer.wte.weight"]                # tied
    return sd


def small_cfg(**kw):
    """A GPT-2-shaped config small enough to build in a test."""
    base = dict(n_layer=6, n_head=2, n_embd=32, block_size=16, vocab_size=64,
                n_trunk=2, n_dense_tail=1)
    base.update(kw)
    return AMTConfig(**base)


def test_every_gpt2_tensor_lands_somewhere():
    cfg = small_cfg()
    model = AMT(cfg)
    report = load_gpt2_state_dict(model, fake_hf_state(cfg))
    assert report["unmatched_source"] == [], report["unmatched_source"]
    # Everything not loaded must be a module GPT-2 does not have.
    assert all(is_new_module(n) for n in report["new"]), report["new"]


def test_head_is_tied_so_the_alias_is_safe():
    """lm_head.weight maps onto wte only because they are one tensor. Verify that."""
    model = AMT(small_cfg())
    assert (model.lm_head.weight.data_ptr()
            == model.transformer.wte.weight.data_ptr())


def test_adaptive_layers_get_the_right_weights():
    """The `.block.` rename must not shift weights between layers."""
    cfg = small_cfg()
    assert cfg.adaptive_layers, "fixture must exercise the adaptive path"
    model = AMT(cfg)
    sd = fake_hf_state(cfg)
    load_gpt2_state_dict(model, sd)

    got = dict(model.named_parameters())
    for i in range(cfg.n_layer):
        src = sd[f"transformer.h.{i}.ln_1.weight"]
        if i in cfg.adaptive_layers:
            key = f"transformer.h.{i}.block.ln_1.weight"
        else:
            key = f"transformer.h.{i}.ln_1.weight"
        assert torch.equal(got[key], src), f"layer {i} got the wrong ln_1"


def test_conv1d_weights_are_transposed():
    cfg = small_cfg()
    model = AMT(cfg)
    sd = fake_hf_state(cfg)
    load_gpt2_state_dict(model, sd)
    got = dict(model.named_parameters())

    src = sd["transformer.h.0.attn.c_attn.weight"]        # (d, 3d) on disk
    assert torch.equal(got["transformer.h.0.attn.c_attn.weight"], src.t())
    # ...and the biases are not transposed.
    assert torch.equal(got["transformer.h.0.attn.c_attn.bias"],
                       sd["transformer.h.0.attn.c_attn.bias"])


def test_position_table_is_truncated_not_rejected():
    """A shorter block_size keeps the first N positions."""
    cfg = small_cfg(block_size=8)
    model = AMT(cfg)
    sd = fake_hf_state(cfg, positions=16)
    load_gpt2_state_dict(model, sd)
    got = dict(model.named_parameters())["transformer.wpe.weight"]
    assert got.shape[0] == 8
    assert torch.equal(got, sd["transformer.wpe.weight"][:8])


def test_padded_vocab_keeps_pretrained_rows_first():
    cfg = small_cfg(vocab_size=72)
    model = AMT(cfg)
    sd = fake_hf_state(cfg, vocab=64)
    report = load_gpt2_state_dict(model, sd)
    assert report["padded_vocab"] == 8
    got = dict(model.named_parameters())["transformer.wte.weight"]
    assert torch.equal(got[:64], sd["transformer.wte.weight"])


def test_shape_disagreements_raise():
    cfg = small_cfg()
    model = AMT(cfg)
    with pytest.raises(ValueError, match="width"):
        load_gpt2_state_dict(model, fake_hf_state(small_cfg(n_embd=64)))
    with pytest.raises(ValueError, match="smaller than the checkpoint"):
        load_gpt2_state_dict(model, fake_hf_state(cfg, vocab=128))
    with pytest.raises(ValueError, match="learned positions"):
        load_gpt2_state_dict(model, fake_hf_state(cfg, positions=8))


def test_freeze_leaves_only_the_new_modules():
    cfg = small_cfg()
    model = AMT(cfg)
    stats = freeze_backbone(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "nothing left to train"
    assert all(is_new_module(n) for n in trainable), trainable
    assert stats["fraction"] < 0.01
    # The routers must actually be in there, or the retrofit trains nothing useful.
    assert any(".router." in n for n in trainable)


def test_freeze_ladder_adds_layernorms_and_memory_block():
    cfg = small_cfg()
    model = AMT(cfg)
    base = freeze_backbone(model)["trainable"]
    with_ln = freeze_backbone(model, train_layernorms=True)["trainable"]
    with_mem = freeze_backbone(model, train_memory_block=True)["trainable"]
    assert with_ln > base
    assert with_mem > base
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith(f"transformer.h.{cfg.mem_layer}.") for n in names)


def test_retrofit_model_still_runs_a_forward():
    """The point of the exercise: a loaded model has to train and generate."""
    cfg = small_cfg()
    model = AMT(cfg)
    load_gpt2_state_dict(model, fake_hf_state(cfg))
    freeze_backbone(model)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    bank = model.make_bank(2, "cpu", dtype=torch.float32)
    _, losses, _, kv = model(x, targets=x, bank=bank)
    losses["total"].backward()

    grads = [n for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    assert any(".router." in n for n in grads), \
        "routers got no gradient -- the router weight is not on the backward path"
    frozen_with_grad = [n for n, p in model.named_parameters()
                        if not p.requires_grad and p.grad is not None]
    assert not frozen_with_grad, frozen_with_grad


def test_gpt2_config_matches_the_checkpoint_shapes():
    for name, shape in GPT2_SHAPES.items():
        cfg = gpt2_config(name)
        assert (cfg.n_layer, cfg.n_head, cfg.n_embd) == (
            shape["n_layer"], shape["n_head"], shape["n_embd"])
        assert cfg.vocab_size == 50257, "padding the head hurts the base comparison"
        assert cfg.block_size == 1024


def test_describe_is_printable():
    cfg = small_cfg()
    model = AMT(cfg)
    report = load_gpt2_state_dict(model, fake_hf_state(cfg))
    text = describe(report, freeze_backbone(model))
    assert "new modules" in text and "trainable" in text


@pytest.mark.skipif(
    not __import__("os").path.isdir(
        __import__("os").path.expanduser("~/.cache/huggingface/hub/models--gpt2")),
    reason="gpt2 not in the HF cache; skip rather than download in a test run")
def test_from_gpt2_real():
    model, report = from_gpt2("gpt2")
    assert report["unmatched_source"] == []
    assert report["n_new"] == 21_559, report["n_new"]
    assert all(is_new_module(n) for n in report["new"])


def _routed_vs_dense(bias_init):
    """Max |logit| difference between a capacity-1.0 routed model and a dense one
    built from identical weights."""
    shared = dict(n_layer=6, n_head=2, n_embd=32, block_size=16, vocab_size=64,
                  n_trunk=2, n_dense_tail=1, use_memory=False)
    sd = fake_hf_state(AMTConfig(**shared))

    dense = AMT(AMTConfig(**shared, route_depth=False))
    routed = AMT(AMTConfig(**shared, route_depth=True, depth_capacity=1.0,
                           route_bias_init=bias_init))
    load_gpt2_state_dict(dense, sd)
    load_gpt2_state_dict(routed, sd)
    assert routed.config.adaptive_layers, "fixture must have adaptive layers"

    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        a = dense(x, return_logits=True)[0]
        b = routed(x, return_logits=True)[0]
    return (a - b).abs().max().item()


def test_full_capacity_retrofit_reproduces_the_dense_model():
    """At capacity 1.0 nothing is routed away, so the model must BE the original.

    It is not, by default: every adaptive block's delta is scaled by
    sigmoid(w_route(x)), which at bias 0 is 0.5 -- seven of twelve layers contribute
    half of themselves. Training from scratch grows into that; a pretrained backbone
    just starts damaged, measured at 27.9 -> 51.3 ppl on GPT-2 before a single token
    is routed away. This is the regression test for that.
    """
    damaged = _routed_vs_dense(0.0)
    restored = _routed_vs_dense(4.0)
    assert restored < damaged / 5, (
        f"a high route_bias_init should nearly reproduce the dense model: "
        f"max|dlogit| {restored:.4f} with bias 4.0 vs {damaged:.4f} with bias 0.0")


def test_gpt2_config_defaults_to_a_neutral_router():
    cfg = gpt2_config("gpt2")
    assert cfg.route_bias_init >= 4.0, (
        "retrofits must start as the pretrained function, not a halved one")
    assert AMTConfig().route_bias_init == 0.0, "from-scratch behaviour must not change"
