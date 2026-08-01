"""Shared fixtures.

Models here are deliberately tiny -- 6 layers at width 64 -- because every test in
this suite is about control flow (which tokens ran which layers) rather than about
learned behaviour, and control flow is width-independent. The suite runs on CPU in
seconds, which is the only way it gets run often enough to be worth having.
"""

import pytest
import torch

from agpt.model import AdaptiveGPT, variant

SMALL = dict(n_layer=6, n_head=4, n_embd=64, block_size=32, vocab_size=128,
             n_min_layers=2, ce_chunks=1, compact_bucket=8)


def build(exit_mode="adaptive", **overrides):
    cfg = variant(exit_mode, **{**SMALL, **overrides})
    return AdaptiveGPT(cfg).eval(), cfg


def randomise_routers(model, seed=0, scale=1.0, bias=1.0):
    """Push the routers off their start-dense initialisation.

    Every router is initialised to say "continue" with probability 0.98, so a freshly
    built model never exits and every test about exiting passes vacuously.

    The gate thresholds the *cumulative* product of continue probabilities, which is
    what makes these numbers fussier than they look: a per-router bias of 2.0 leaves
    every token alive through the whole stack (0.88^4 = 0.6 > 0.5), while 0.0 exits
    everything at the first router. The defaults here sit in the narrow band that
    produces a genuine spread of exit depths -- `test_compact_actually_exits` fails if
    they drift out of it, which is the point of having that test.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for r in model.routers.values():
            r.fc.weight.normal_(0, 0.5, generator=g)
            r.proj.weight.normal_(0, scale, generator=g)
            r.proj.bias.fill_(bias)
    return model


@pytest.fixture
def tokens():
    torch.manual_seed(0)
    return torch.randint(0, SMALL["vocab_size"], (3, SMALL["block_size"]))
