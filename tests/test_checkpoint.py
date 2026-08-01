"""Checkpoint round-trips, including the ones the three-stage pipeline depends on."""

import pytest
import torch

from agpt.evaluate import load_checkpoint
from agpt.model.adaptive_gpt import strip_compile_prefix
from conftest import build, randomise_routers


def save(model, cfg, path, step=0):
    torch.save({"model": model.state_dict(), "config": cfg, "step": step,
                "val_loss": 1.0, "args": {}}, path)
    return str(path)


def test_round_trip_preserves_the_function(tokens, tmp_path):
    model, cfg = build("adaptive")
    randomise_routers(model)
    path = save(model, cfg, tmp_path / "ck.pt")

    loaded, _, _ = load_checkpoint(path)
    with torch.no_grad():
        a, _, _ = model(tokens, return_logits=True)
        b, _, _ = loaded(tokens, return_logits=True)
    assert torch.equal(a, b)


def test_compile_prefix_is_stripped():
    """A checkpoint saved from a compiled handle must still load.

    `torch.compile` returns a wrapper whose state_dict namespaces every key under
    `_orig_mod.`, and nothing downstream -- resume, eval, inference -- can read that.
    """
    model, _ = build("adaptive")
    prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    clean = strip_compile_prefix(prefixed)
    assert set(clean) == set(model.state_dict())
    model.load_state_dict(clean)
    # Idempotent: an already-clean dict must pass through untouched.
    assert strip_compile_prefix(clean) is clean


def test_stage1_checkpoint_loads_into_every_arm(tokens, tmp_path):
    """The pipeline's central move: one backbone, four routing rules.

    Stage 1 saves an adaptive model with routers at their initialisation. `compare.py`
    then rebuilds that same checkpoint as `dense`, `random` and `fixed`, which means
    dropping the router tensors -- and rebuilds it back as `adaptive`, which means
    finding them again. Both directions have to work, and nothing else may go missing.
    """
    model, cfg = build("adaptive")
    randomise_routers(model)
    path = save(model, cfg, tmp_path / "s1.pt")

    for mode in ("dense", "random", "fixed"):
        arm, arm_cfg, _ = load_checkpoint(path, exit_mode=mode)
        assert arm_cfg.exit_mode == mode
        assert len(arm.routers) == 0
        with torch.no_grad():
            arm(tokens)                      # must run, not just build

    back, back_cfg, _ = load_checkpoint(path, exit_mode="adaptive")
    assert len(back.routers) == len(cfg.router_layers)
    with torch.no_grad():
        a, _, _ = model(tokens, return_logits=True)
        b, _, _ = back(tokens, return_logits=True)
    assert torch.equal(a, b), "reloading as adaptive lost the routers"


def test_dense_checkpoint_gains_fresh_routers(tokens, tmp_path):
    """Loading a router-free checkpoint as adaptive leaves the routers at init.

    That is the Stage 1 -> Stage 2 handoff when Stage 1 was run as the `dense` variant
    rather than as adaptive-with-gates-pinned. It has to work, and it has to start
    from the deliberate start-dense initialisation rather than from noise.
    """
    model, cfg = build("dense")
    path = save(model, cfg, tmp_path / "d.pt")
    arm, arm_cfg, _ = load_checkpoint(path, exit_mode="adaptive")
    assert len(arm.routers) == len(arm_cfg.router_layers)
    for r in arm.routers.values():
        assert r.proj.bias.item() == pytest.approx(arm_cfg.router_bias_init)


def test_a_genuinely_wrong_checkpoint_is_rejected(tmp_path):
    """Only router tensors may be missing. A width mismatch must not be tolerated."""
    model, cfg = build("adaptive")
    path = save(model, cfg, tmp_path / "ck.pt")
    with pytest.raises(Exception):
        load_checkpoint(path, n_embd=128, n_head=4)


def test_config_overrides_survive_the_load(tmp_path):
    model, cfg = build("adaptive")
    path = save(model, cfg, tmp_path / "ck.pt")
    _, out, _ = load_checkpoint(path, exit_mode="fixed", fixed_exit_layer=4)
    assert out.exit_mode == "fixed" and out.fixed_exit_layer == 4
    assert out.n_embd == cfg.n_embd, "an override leaked into the architecture"


def test_config_stored_as_a_dict_still_loads(tokens, tmp_path):
    """Older checkpoints and hand-written ones store the config as a plain dict."""
    model, cfg = build("adaptive")
    path = str(tmp_path / "ck.pt")
    torch.save({"model": model.state_dict(),
                "config": {k: v for k, v in cfg.__dict__.items()},
                "step": 0}, path)
    loaded, out, _ = load_checkpoint(path)
    assert out.exit_mode == "adaptive"
    with torch.no_grad():
        loaded(tokens)
