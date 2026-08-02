"""Deriving exit labels from a dense forward pass."""

import pytest
import torch

from agpt.model.targets import (_monotone, delta_targets, exit_targets, kl_targets,
                                oracle_depth)
from conftest import build


def make_hiddens(n_layer, B=2, T=8, C=16, converge_at=None, seed=0):
    """A synthetic residual-stream trajectory.

    `converge_at` freezes the stream from that layer on, so the correct label is known
    in advance: continue up to `converge_at`, exit after.
    """
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, C, generator=g)
    out = [h]
    for l in range(n_layer):
        if converge_at is not None and l >= converge_at:
            out.append(out[-1].clone())
        else:
            out.append(out[-1] + torch.randn(B, T, C, generator=g))
    return out


def test_monotone_makes_exit_final():
    flags = torch.tensor([[1], [0], [1], [1]], dtype=torch.bool).view(4, 1, 1)
    out = _monotone(flags).view(-1)
    assert out.tolist() == [True, False, False, False], \
        "a token revived after being told to exit"


def test_delta_labels_a_converged_stream_as_exit():
    hiddens = make_hiddens(n_layer=8, converge_at=4)
    targets, ratios = delta_targets(hiddens, tau=1e-4)
    # Layers 4.. leave the stream untouched, so the remaining change from layer 4 on
    # is exactly zero and every one of those should say exit.
    assert targets[4:].sum() == 0, "a frozen stream was still told to continue"
    assert targets[:3].all(), "a moving stream was told to exit"
    assert ratios[-1].abs().max() < 1e-6


def test_delta_ratio_decreases_toward_the_end():
    hiddens = make_hiddens(n_layer=8)
    _, ratios = delta_targets(hiddens, tau=0.05)
    assert ratios[0].mean() > ratios[-2].mean(), \
        "the remaining change should shrink as the stack progresses"


def test_tau_controls_how_much_exits():
    hiddens = make_hiddens(n_layer=8)
    strict, _ = delta_targets(hiddens, tau=0.001)   # almost nothing has converged
    loose, _ = delta_targets(hiddens, tau=10.0)     # everything has
    assert strict.mean() > loose.mean()


def test_kl_targets_agree_with_a_frozen_stream():
    """A stream that stops moving cannot change the prediction either."""
    torch.manual_seed(0)
    hiddens = make_hiddens(n_layer=6, C=16, converge_at=3)
    ln_f = torch.nn.LayerNorm(16)
    lm_head = torch.nn.Linear(16, 32, bias=False)
    targets, ratios = kl_targets(hiddens, tau=1e-6, ln_f=ln_f, lm_head=lm_head,
                                 chunks=2)
    assert targets[3:].sum() == 0
    assert ratios[-1].abs().max() < 1e-5


def test_exit_targets_dispatches(tokens):
    model, cfg = build("adaptive")
    hiddens = model.dense_hidden_states(tokens)

    cfg.target_type = "delta"
    a, _ = exit_targets(hiddens, cfg)
    cfg.target_type = "kl"
    b, _ = exit_targets(hiddens, cfg, model.transformer.ln_f, model.lm_head)
    assert a.shape == b.shape == (cfg.n_layer,) + tokens.shape

    cfg.target_type = "kl"
    with pytest.raises(ValueError):
        exit_targets(hiddens, cfg)          # kl without a head is a caller error


def test_oracle_depth_is_bounded():
    n_layer, n_min = 8, 3
    hiddens = make_hiddens(n_layer=n_layer)
    targets, _ = delta_targets(hiddens, tau=0.05)
    d = oracle_depth(targets, n_min)
    assert d.min() >= n_min
    assert d.max() <= n_layer


def test_oracle_depth_at_the_extremes():
    n_layer, n_min = 8, 3
    hiddens = make_hiddens(n_layer=n_layer)
    shallow = oracle_depth(delta_targets(hiddens, tau=1e9)[0], n_min)
    deep = oracle_depth(delta_targets(hiddens, tau=0.0)[0], n_min)
    assert (shallow == n_min).all(), "tau=inf should exit everything immediately"
    assert (deep == n_layer).all(), "tau=0 should keep everything to the end"


# ---------------------------------------------------------------------------
# Subsampled labelling -- the thing that makes the KL rule affordable
# ---------------------------------------------------------------------------

def test_sample_positions_hits_its_rate():
    from agpt.model.targets import sample_positions
    g = torch.Generator().manual_seed(0)
    m = sample_positions((64, 256), 0.125, "cpu", generator=g)
    assert m.dtype == torch.bool
    assert 0.10 < m.float().mean() < 0.15
    assert sample_positions((8, 8), 1.0, "cpu").all()


def test_masked_kl_equals_unmasked_where_labelled():
    """Subsampling must change WHICH positions are labelled, never the labels.

    If these diverged, the cheap path would be training the router on a different
    target from the one the diagnostics report.
    """
    torch.manual_seed(0)
    hiddens = make_hiddens(n_layer=6, B=2, T=16, C=16)
    ln_f = torch.nn.LayerNorm(16)
    lm_head = torch.nn.Linear(16, 32, bias=False)

    full, full_r = kl_targets(hiddens, 0.5, ln_f, lm_head, chunks=2)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, ::3] = True
    part, part_r = kl_targets(hiddens, 0.5, ln_f, lm_head, chunks=2, mask=mask)

    m = mask.unsqueeze(0).expand_as(full)
    assert torch.allclose(full_r[m], part_r[m], atol=1e-5)
    assert torch.equal(full[m], part[m])


def test_unlabelled_positions_default_to_exit():
    """The trap the label_mask exists to close.

    Unlabelled positions come back with ratio 0, which reads as "converged" and
    monotonises to exit-everywhere. Training on them unweighted would collapse the
    stack to minimum depth, so this pins the behaviour that makes the mask mandatory.
    """
    torch.manual_seed(0)
    hiddens = make_hiddens(n_layer=6, B=2, T=16, C=16)
    ln_f, lm_head = torch.nn.LayerNorm(16), torch.nn.Linear(16, 32, bias=False)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 0] = True
    labels, _ = kl_targets(hiddens, 0.5, ln_f, lm_head, chunks=1, mask=mask)
    assert labels[:, :, 1:].sum() == 0, "an unlabelled position was not left at 'exit'"


def test_empty_mask_is_not_a_crash():
    torch.manual_seed(0)
    hiddens = make_hiddens(n_layer=4, B=2, T=8, C=16)
    ln_f, lm_head = torch.nn.LayerNorm(16), torch.nn.Linear(16, 32, bias=False)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    labels, ratios = kl_targets(hiddens, 0.5, ln_f, lm_head, mask=mask)
    assert labels.shape == (4, 2, 8) and ratios.abs().sum() == 0
