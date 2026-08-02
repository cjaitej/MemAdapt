"""Deriving "this token has converged" labels from a dense forward pass.

The router has to be told what a good exit looks like, and hand-labelling tokens as
easy or hard is not available. So the labels are *derived*: run the stack densely,
watch what the remaining layers actually do to each token, and call a token converged
at layer l if the layers after l were going to leave it essentially alone.

Two rules, and the difference between them is the whole question of what "converged"
should mean.

`delta` -- the residual stream stops moving
------------------------------------------
    r_l = || h_L - h_l ||  /  || h_L ||

Cheap, and it is the rule the project plan specifies. It measures convergence of the
*representation*.

`kl` -- the prediction stops moving
-----------------------------------
    r_l = KL( p(. | h_L)  ||  p(. | h_l) )

Costs a vocab-sized projection per layer, but it measures convergence of the thing the
model is actually judged on. These two disagree more than you would expect: the last
few layers of a trained transformer move the residual stream substantially while
barely changing the argmax, because much of that movement is along directions the
unembedding is close to blind to. `delta` therefore tends to label fewer tokens as
converged than the loss would justify, which shows up as a conservative router. Report
which rule was used; it is a modelling choice, not a hyperparameter.

Monotonicity
------------
`r_l` is not monotone in l -- a token can look settled at layer 6 and get revised at
layer 9. But an exit is final, so the label has to be: the target is 1 (continue) only
while *no* earlier layer has already said stop. `_monotone` applies that cumulative
AND, which is why the labels are derived here in one place rather than thresholded
independently at each router.
"""

import torch
import torch.nn.functional as F


def _monotone(continue_flags):
    """(L, B, T) bool -> the same with an exit made final.

    Once a layer says stop, every later layer says stop.
    """
    return torch.cumprod(continue_flags.to(torch.uint8), dim=0).bool()


@torch.no_grad()
def delta_targets(hiddens, tau):
    """Relative remaining change in the residual stream.

    Parameters
    ----------
    hiddens : list of (B, T, C), length n_layer + 1. `hiddens[0]` is the embedding
        and `hiddens[l+1]` is the state after block l.
    tau : exit where the remaining change falls below this fraction.

    Returns (targets, ratios), both (n_layer, B, T). `targets` is float in {0, 1}
    where 1 means continue; `ratios` is the raw r_l, kept because the histogram of it
    is the figure that justifies the choice of tau.
    """
    h_final = hiddens[-1]
    scale = h_final.norm(dim=-1).clamp_min(1e-6)                # (B, T)

    ratios = torch.stack([
        (h_final - h).norm(dim=-1) / scale for h in hiddens[1:]
    ])                                                          # (n_layer, B, T)
    return _monotone(ratios > tau).to(h_final.dtype), ratios


def sample_positions(shape, fraction, device, generator=None):
    """A random (B, T) bool mask selecting `fraction` of the positions.

    Which positions get labelled is resampled every step, so over a run every position
    is labelled many times -- this trades a little gradient noise for a large constant
    factor, it does not permanently hide part of the corpus from the routers.
    """
    if fraction >= 1.0:
        return torch.ones(shape, dtype=torch.bool, device=device)
    return torch.rand(shape, device=device, generator=generator) < fraction


@torch.no_grad()
def kl_targets(hiddens, tau, ln_f, lm_head, chunks=8, mask=None):
    """Remaining change in the predictive distribution, in nats.

    Chunked over the flattened token axis: the (B*T, vocab) logits are ~400 MB in fp32
    at B=4, T=512, and this needs them once per layer. Materialising all of them is an
    instant OOM on a 4 GB card, which is the card this project targets.

    `mask` restricts the work to a subset of positions. This is not an optimisation
    detail -- measured on the target GPU this function costs 580 ms per B=4 batch
    against 38 ms for the forward pass it rides on, so at full coverage it dominates
    training by more than an order of magnitude. Unmasked positions come back with a
    ratio of 0, which `_monotone` turns into an all-exit label; the caller MUST weight
    them out of the loss, which is what `label_mask` does in `AdaptiveGPT.forward`.
    Feeding these labels in unweighted would teach every unlabelled token to exit
    immediately.
    """
    n_layer = len(hiddens) - 1
    B, T, _ = hiddens[0].shape
    ratios = hiddens[0].new_zeros(n_layer, B, T)
    flat_ratios = ratios.view(n_layer, -1)

    flat_final = ln_f(hiddens[-1]).view(B * T, -1)
    flat_layers = [ln_f(h).view(B * T, -1) for h in hiddens[1:]]

    index = None
    if mask is not None:
        index = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            return _monotone(ratios > tau).to(hiddens[-1].dtype), ratios
        flat_final = flat_final[index]
        flat_layers = [f[index] for f in flat_layers]

    start = 0
    for xf in flat_final.chunk(max(chunks, 1), dim=0):
        n = xf.size(0)
        log_p_final = F.log_softmax(lm_head(xf).float(), dim=-1)
        p_final = log_p_final.exp()
        for l, flat in enumerate(flat_layers):
            log_p_l = F.log_softmax(lm_head(flat[start:start + n]).float(), dim=-1)
            kl = (p_final * (log_p_final - log_p_l)).sum(-1).to(ratios.dtype)
            if index is None:
                flat_ratios[l, start:start + n] = kl
            else:
                flat_ratios[l, index[start:start + n]] = kl
        start += n

    return _monotone(ratios > tau).to(hiddens[-1].dtype), ratios


@torch.no_grad()
def exit_targets(hiddens, config, ln_f=None, lm_head=None, mask=None):
    """Dispatch on `config.target_type`. Returns (targets, ratios).

    `targets[l]` is the label for the router that sits after block l. Only the entries
    for `config.router_layers` are used; the rest are computed anyway because the full
    (n_layer, B, T) tensor is what the oracle-depth figure plots.

    `mask` is honoured only by the KL rule -- the delta rule is cheap enough that
    subsampling it would trade accuracy for nothing.
    """
    if config.target_type == "kl":
        if ln_f is None or lm_head is None:
            raise ValueError("target_type='kl' needs ln_f and lm_head")
        return kl_targets(hiddens, config.target_tau, ln_f, lm_head,
                          chunks=max(config.ce_chunks, 1) * 2, mask=mask)
    return delta_targets(hiddens, config.target_tau)


def oracle_depth(targets, n_min_layers):
    """(n_layer, B, T) labels -> (B, T) the depth the oracle rule would spend.

    A token that continues through every router runs all n_layer blocks; one that is
    told to stop after block l runs l+1. The floor is `n_min_layers`, since the first
    few layers are not routable.
    """
    n_layer = targets.size(0)
    # targets[l] == 1 means "continue past block l". Summing the routable ones counts
    # the blocks entered beyond the minimum.
    routable = targets[n_min_layers - 1:n_layer - 1]
    return routable.sum(0) + n_min_layers
