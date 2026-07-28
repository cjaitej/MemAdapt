"""Checkpoints must load into a plain AMT, whatever wrapper wrote them.

`torch.compile` returns an OptimizedModule, and its `state_dict()` prefixes every
key with `_orig_mod.`. Saving that handle produced checkpoints that nothing could
load -- not `--resume latest`, not inference, not the analysis scripts. The failure
only appears under `--compile`, so it survived every uncompiled test run and only
surfaced when a real run needed reloading.

These tests pin both halves: the prefix is stripped on load, and training saves the
unwrapped module in the first place.
"""

import torch

from amt.model import AMT, variant
from amt.model.amt import strip_compile_prefix


def tiny():
    return variant("b6_amt_joint", n_layer=4, n_embd=64, n_head=4, block_size=32,
                   vocab_size=128, n_trunk=1, n_dense_tail=1, mem_size=32,
                   n_neighbors=4, mem_query_chunk=16, ce_chunks=1)


def test_clean_state_dict_passes_through_unchanged():
    sd = AMT(tiny()).state_dict()
    assert strip_compile_prefix(sd) is sd, "no prefix means no copy"


def test_compiled_prefix_is_stripped():
    sd = AMT(tiny()).state_dict()
    wrapped = {f"_orig_mod.{k}": v for k, v in sd.items()}
    assert set(strip_compile_prefix(wrapped)) == set(sd)


def test_a_compiled_checkpoint_loads_into_a_plain_model():
    """The end-to-end failure: a --compile run's checkpoint reloaded for inference."""
    cfg = tiny()
    trained = AMT(cfg)
    with torch.no_grad():                      # make the weights distinguishable
        for p in trained.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    # Exactly what torch.compile's OptimizedModule.state_dict() emits.
    as_saved = {f"_orig_mod.{k}": v for k, v in trained.state_dict().items()}

    fresh = AMT(cfg)
    fresh.load_state_dict(strip_compile_prefix(as_saved))
    for a, b in zip(trained.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)


def test_training_saves_the_unwrapped_module():
    """train.py must save raw_model, not the compiled handle.

    Stripping on load covers checkpoints already written, but new ones should not
    need the workaround at all.
    """
    import inspect

    from amt import train
    src = inspect.getsource(train.main)
    assert '"model": raw_model.state_dict()' in src, (
        "checkpoint save must use raw_model.state_dict(); saving the compiled "
        "handle writes _orig_mod.-prefixed keys"
    )
