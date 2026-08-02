"""The Pareto verdict: the function that decides the project's headline claim.

Worth testing precisely because getting it wrong is silent -- it prints a confident
sentence either way. The first version was too strict: it demanded the adaptive arm
beat every fixed arm of equal-or-greater cost, which fails whenever the dearer
neighbour is better *and* dearer. That is a different point on the curve, not a
refutation, and it produced a false negative on the first real result.
"""

import pytest

from compare import match_random_p, pareto_verdict


def arms(adaptive_ppl, adaptive_flops, fixed, random_ppl=999.0):
    """`fixed` is {name: (flops_frac, ppl)}."""
    out = {"adaptive": {"ppl": adaptive_ppl, "flops_frac_layers": adaptive_flops},
           "random": {"ppl": random_ppl, "flops_frac_layers": adaptive_flops}}
    for name, (f, p) in fixed.items():
        out[name] = {"ppl": p, "flops_frac_layers": f}
    return out


def test_interpolates_between_the_bracketing_fixed_arms():
    """The real measured case: adaptive loses to the dearer arm and still wins.

    fixed@10 = 49.45 ppl at 83.3%, fixed@11 = 41.78 at 91.7%, adaptive = 42.94 at
    87.6%. Adaptive is worse than fixed@11 in isolation -- but fixed@11 costs 4 points
    more FLOPs, and the frontier at 87.6% is 45.52. Adaptive wins by 2.57.
    """
    v = pareto_verdict(arms(42.94, 0.876,
                            {"fixed": (0.833, 49.45), "fixed_hi": (0.917, 41.78),
                             "dense": (1.0, 41.79)}))
    assert v["frontier_ppl"] == pytest.approx(45.52, abs=0.02)
    assert v["beats_fixed_frontier"]
    assert v["margin"] == pytest.approx(2.57, abs=0.02)


def test_reports_a_loss_when_adaptive_is_above_the_frontier():
    v = pareto_verdict(arms(47.0, 0.876,
                            {"fixed": (0.833, 49.45), "fixed_hi": (0.917, 41.78),
                             "dense": (1.0, 41.79)}))
    assert not v["beats_fixed_frontier"]
    assert v["margin"] < 0
    assert any("not paying" in line for line in v["lines"])


def test_beating_only_a_cheaper_fixed_arm_is_not_a_win():
    """Adaptive spending more FLOPs than every fixed arm must not read as a victory."""
    v = pareto_verdict(arms(45.0, 0.95,
                            {"fixed": (0.833, 49.45), "fixed_hi": (0.917, 47.0),
                             "dense": (1.0, 41.79)}))
    # 0.95 sits between fixed_hi (91.7%) and dense (100%): frontier ~44.6, so a 45.0
    # adaptive is ABOVE it despite beating both fixed arms outright.
    assert not v["beats_fixed_frontier"]


def test_losing_to_random_is_flagged_loudly():
    v = pareto_verdict(arms(50.0, 0.876,
                            {"fixed": (0.833, 49.45), "fixed_hi": (0.917, 41.78),
                             "dense": (1.0, 41.79)}, random_ppl=45.0))
    assert not v["beats_random"]
    assert any("WARNING" in line for line in v["lines"])


def test_unbracketed_falls_back_and_says_so():
    v = pareto_verdict(arms(42.0, 0.40, {"fixed": (0.833, 49.45),
                                         "dense": (1.0, 41.79)}))
    assert "NOT bracketed" in v["basis"]


def test_match_random_p_hits_the_target_depth():
    class Cfg:
        n_min_layers, n_layer = 3, 12
    for target in (4.0, 6.5, 9.0, 11.5):
        p = match_random_p(target, Cfg())
        got = 3 + sum(p ** k for k in range(1, 10))
        assert got == pytest.approx(target, abs=0.05)
