"""Loading pretrained GPT-2 weights into the AdaptiveGPT stack.

Built against a synthetic HuggingFace-shaped state dict rather than a download, so the
key mapping, the Conv1D transposes and the freeze logic are all testable offline. The
one thing that needs the network -- that OpenAI's actual checkpoint has exactly these
keys -- is checked by `test_real_gpt2_keys`, which skips when `transformers` is absent.
"""

import pytest
import torch

from agpt.model import AdaptiveGPT
from agpt.model.retrofit import (GPT2_SHAPES, describe, freeze_backbone, gpt2_config,
                                 is_new_module, load_gpt2_state_dict)

TINY = dict(n_layer=4, n_head=4, n_embd=32, vocab_size=50257, block_size=64,
            bias=True, n_min_layers=2)


def fake_hf_state(n_layer=4, n_embd=32, vocab=50257, n_pos=64, seed=0):
    """A state dict shaped exactly like GPT2LMHeadModel's, filled with noise.

    The four projection weights are stored transposed, because OpenAI's checkpoints
    hold them as `Conv1D` -- a Linear with its axes swapped. Getting that wrong loads
    cleanly and produces a model that is quietly wrong, which is why it is the first
    thing this file checks.
    """
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)          # noqa: E731
    st = {
        "transformer.wte.weight": r(vocab, n_embd),
        "transformer.wpe.weight": r(n_pos, n_embd),
        "transformer.ln_f.weight": r(n_embd),
        "transformer.ln_f.bias": r(n_embd),
    }
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        st.update({
            p + "ln_1.weight": r(n_embd), p + "ln_1.bias": r(n_embd),
            p + "ln_2.weight": r(n_embd), p + "ln_2.bias": r(n_embd),
            p + "attn.c_attn.weight": r(n_embd, 3 * n_embd),   # Conv1D: (in, out)
            p + "attn.c_attn.bias": r(3 * n_embd),
            p + "attn.c_proj.weight": r(n_embd, n_embd),
            p + "attn.c_proj.bias": r(n_embd),
            p + "mlp.c_fc.weight": r(n_embd, 4 * n_embd),
            p + "mlp.c_fc.bias": r(4 * n_embd),
            p + "mlp.c_proj.weight": r(4 * n_embd, n_embd),
            p + "mlp.c_proj.bias": r(n_embd),
            p + "attn.bias": torch.ones(1, 1, n_pos, n_pos),   # buffer, must be skipped
        })
    st["lm_head.weight"] = st["transformer.wte.weight"]        # tied
    return st


def build(**over):
    from agpt.model import AdaptiveGPTConfig
    return AdaptiveGPT(AdaptiveGPTConfig(**{**TINY, **over}))


def test_every_source_tensor_finds_a_home():
    model = build(exit_mode="adaptive")
    report = load_gpt2_state_dict(model, fake_hf_state())
    assert report["unmatched_source"] == [], \
        f"unmatched: {report['unmatched_source']}"
    assert report["padded_vocab"] == 0


def test_conv1d_weights_are_transposed():
    model = build(exit_mode="adaptive")
    state = fake_hf_state()
    load_gpt2_state_dict(model, state)
    got = model.transformer.h[0].attn.c_attn.weight
    assert torch.equal(got, state["transformer.h.0.attn.c_attn.weight"].t())


def test_only_the_routers_are_new():
    model = build(exit_mode="adaptive")
    report = load_gpt2_state_dict(model, fake_hf_state())
    assert all(n.startswith("routers.") for n in report["new"]), report["new"]
    assert report["n_new"] == model.router_params()


def test_a_dense_arm_has_nothing_new():
    model = build(exit_mode="dense")
    report = load_gpt2_state_dict(model, fake_hf_state())
    assert report["new"] == []


def test_padded_vocab_keeps_the_pretrained_rows():
    model = build(exit_mode="adaptive", vocab_size=50304)
    state = fake_hf_state()
    report = load_gpt2_state_dict(model, state)
    assert report["padded_vocab"] == 50304 - 50257
    assert torch.equal(model.transformer.wte.weight[:50257],
                       state["transformer.wte.weight"])


def test_a_shorter_context_takes_the_head_of_the_position_table():
    model = build(exit_mode="adaptive", block_size=32)
    state = fake_hf_state(n_pos=64)
    load_gpt2_state_dict(model, state)
    assert torch.equal(model.transformer.wpe.weight,
                       state["transformer.wpe.weight"][:32])


def test_mismatches_are_refused_not_absorbed():
    with pytest.raises(ValueError, match="width"):
        load_gpt2_state_dict(build(n_embd=64, n_head=4), fake_hf_state(n_embd=32))
    with pytest.raises(ValueError, match="smaller than"):
        load_gpt2_state_dict(build(vocab_size=1000), fake_hf_state())
    with pytest.raises(ValueError, match="learned positions"):
        load_gpt2_state_dict(build(block_size=128), fake_hf_state(n_pos=64))


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def test_freeze_leaves_only_the_routers():
    model = build(exit_mode="adaptive")
    stats = freeze_backbone(model)
    assert stats["trainable"] == model.router_params()
    for name, p in model.named_parameters():
        assert p.requires_grad == is_new_module(name), name


def test_train_head_releases_ln_f_and_the_tied_head():
    model = build(exit_mode="adaptive")
    base = freeze_backbone(model)["trainable"]
    with_head = freeze_backbone(model, train_head=True)["trainable"]
    assert with_head > base
    assert model.lm_head.weight.requires_grad
    # Tying means the input embedding comes along; the docstring says so and the
    # trainable count is meaningless if a reader does not know it.
    assert model.transformer.wte.weight.requires_grad


def test_train_layernorms_releases_the_blocks_norms():
    model = build(exit_mode="adaptive")
    freeze_backbone(model, train_layernorms=True)
    assert model.transformer.h[0].ln_1.weight.requires_grad
    assert not model.transformer.h[0].attn.c_attn.weight.requires_grad


def test_a_dense_retrofit_would_train_nothing():
    """The trainer turns this into a clear error rather than a silent no-op run."""
    model = build(exit_mode="dense")
    assert freeze_backbone(model)["trainable"] == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_gpt2_config_shapes_match_the_checkpoints():
    for name, shape in GPT2_SHAPES.items():
        cfg = gpt2_config(name)
        for k, v in shape.items():
            assert getattr(cfg, k) == v
        assert cfg.vocab_size == 50257, "padding the vocab breaks the comparison"
        assert 0 < cfg.n_min_layers < cfg.n_layer


def test_gpt2_config_scales_the_always_on_prefix():
    assert gpt2_config("gpt2").n_min_layers == 3
    assert gpt2_config("gpt2-large").n_min_layers == 9


def test_unknown_checkpoint_name_is_refused():
    with pytest.raises(KeyError):
        gpt2_config("gpt2-enormous")


def test_describe_mentions_the_freeze():
    model = build(exit_mode="adaptive")
    report = load_gpt2_state_dict(model, fake_hf_state())
    text = describe(report, freeze_backbone(model, train_head=True))
    assert "trainable" in text and "new modules" in text


@pytest.mark.slow
def test_real_gpt2_keys():
    """The one claim the synthetic dict cannot make: these are GPT-2's actual keys."""
    transformers = pytest.importorskip("transformers")
    hf = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    cfg = gpt2_config("gpt2", block_size=64)
    model = AdaptiveGPT(cfg)
    report = load_gpt2_state_dict(model, hf.state_dict())
    assert report["unmatched_source"] == []
    assert all(n.startswith("routers.") for n in report["new"])
