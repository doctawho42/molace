"""The pieces a continuous label informativeness would need, and the one that breaks.

LI = I(y_xi, y_eta) / H(y_xi). For a continuous label the numerator is fine: mutual information is
defined and is invariant under a smooth invertible change of variables. The denominator is not: the
differential entropy of a rescaled variable is h(cY) = h(Y) + log|c|, so the ratio depends on the
units the label is measured in, and there is always a scale at which the denominator is zero.

These tests pin the estimator against closed forms, so the demonstration built on it means something.
"""
from __future__ import annotations

import numpy as np
import pytest

from molace.measures.continuous_li import differential_entropy, mutual_information


def test_differential_entropy_of_a_gaussian_matches_the_closed_form():
    rng = np.random.default_rng(0)
    for sigma in (0.5, 1.0, 3.0):
        y = rng.normal(0.0, sigma, size=40000)
        want = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
        assert differential_entropy(y) == pytest.approx(want, abs=0.05)


def test_differential_entropy_of_a_uniform_matches_the_closed_form():
    rng = np.random.default_rng(1)
    y = rng.uniform(0.0, 4.0, size=40000)
    assert differential_entropy(y) == pytest.approx(np.log(4.0), abs=0.05)


def test_rescaling_shifts_differential_entropy_by_log_c():
    """The whole problem, in one assertion."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=20000)
    for c in (0.1, 2.0, 1000.0):
        assert differential_entropy(c * y) == pytest.approx(
            differential_entropy(y) + np.log(c), abs=0.02
        )


def test_differential_entropy_goes_negative_for_a_concentrated_variable():
    rng = np.random.default_rng(3)
    assert differential_entropy(rng.normal(0.0, 0.01, size=20000)) < 0.0


def test_mutual_information_of_independent_variables_is_near_zero():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=8000), rng.normal(size=8000)
    assert abs(mutual_information(a, b)) < 0.05


def test_mutual_information_rises_with_dependence():
    rng = np.random.default_rng(5)
    a = rng.normal(size=8000)
    weak = 0.3 * a + rng.normal(size=8000)
    strong = 0.95 * a + 0.1 * rng.normal(size=8000)
    assert mutual_information(a, weak) < mutual_information(a, strong)


def test_scaling_both_variables_together_leaves_the_estimate_alone():
    """A real property of the estimator, not an imposed one: scaling the joint space is free.

    KSG's max-norm distances all scale by the same factor, so every neighbour comparison is
    unchanged. This is the case the units demonstration actually uses -- one label at two edge
    endpoints, rescaled together -- so it is the one that has to hold without help.
    """
    rng = np.random.default_rng(6)
    a = rng.normal(size=4000)
    b = 0.7 * a + rng.normal(size=4000)
    base = mutual_information(a, b)
    for c in (1e-3, 2.0, 1000.0):
        # approximate, not exact: the neighbour counts come from floating-point comparisons at the
        # distance boundary, so a few points cross it. Measured at 4e-5 here; on a heavily tied
        # sample the same mechanism moves it by 17%, which is why the units table below cannot
        # present numerator invariance as something it observed.
        assert mutual_information(c * a, c * b) == pytest.approx(base, abs=1e-3)


def test_scaling_one_variable_moves_the_estimate_unless_you_standardise():
    """What standardising actually buys, and why its invariance is not evidence of anything.

    True mutual information is invariant under rescaling either argument alone. This estimator is
    not: the max-norm mixes the two axes. Standardising removes it BY CONSTRUCTION, which is why a
    demonstration run with standardise=True proves nothing about mutual information -- a stub
    returning a constant would pass identically.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(size=4000)
    b = 0.7 * a + rng.normal(size=4000)

    raw = mutual_information(a, b)
    skewed = mutual_information(1000.0 * a, b)
    assert abs(skewed - raw) > 0.15, (
        f"the unstandardised estimator must move when one axis is stretched; "
        f"got {raw:.4f} against {skewed:.4f}"
    )

    std_raw = mutual_information(a, b, standardise=True)
    std_skewed = mutual_information(1000.0 * a, b, standardise=True)
    assert std_skewed == pytest.approx(std_raw, abs=1e-5)


def test_heavy_ties_break_the_estimator_in_both_directions():
    """Documented limitation, not a bug to fix -- and the reason a continuous LI is hard in practice.

    LI's marginal is a degree-weighted edge-endpoint sample, so each molecule's label enters deg(v)
    times. Two things then happen to a nearest-neighbour mutual information, and they pull opposite
    ways, so neither can be corrected for by a constant:

      * plain duplication INFLATES it by orders of magnitude -- the k nearest neighbours of a point
        become its own copies, at distance zero, which the estimator reads as perfect dependence;
      * rounding to an assay's reporting grid drives it NEGATIVE, which no mutual information is.

    Measured below. The real data has both at once: multiplicity 20 to 50 on a label already rounded.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(size=400)
    b = 0.6 * a + rng.normal(size=400)

    clean = mutual_information(a, b)
    duplicated = mutual_information(np.repeat(a, 5), np.repeat(b, 5))
    rounded = mutual_information(np.round(a, 1), np.round(b, 1))

    assert duplicated > 10 * abs(clean), (
        f"degree weighting must inflate the estimate by an order of magnitude; "
        f"got {duplicated:.4f} against a clean {clean:.4f}"
    )
    assert rounded < 0.0, (
        f"rounding to a reporting grid must push the estimate below zero, where no true mutual "
        f"information can be; got {rounded:.4f}"
    )


def test_differential_entropy_refuses_a_tied_sample_by_default():
    """The default must not quietly invent a number for a sample that has no estimate."""
    with pytest.raises(ValueError, match="zero .* neighbour distance"):
        differential_entropy(np.repeat(np.array([1.0, 2.0, 3.0]), 40))


def test_the_nudge_is_opt_in_and_the_level_it_gives_depends_on_it():
    """Quoting a level off a nudged estimate is quoting the constant. Pinned so nobody forgets."""
    # mostly distinct values with a few repeated groups: only the repeats hit the tie branch
    y = np.concatenate([np.linspace(0.0, 1.0, 200), np.repeat([0.3, 0.7], 6)])
    levels = [differential_entropy(y, tie_policy="nudge", nudge=u) for u in (1e-2, 1e-6, 1e-12)]
    assert levels[0] > levels[1] > levels[2]
    assert levels[0] - levels[2] > 1.0, f"the nudge must move the level materially; got {levels}"


def test_but_the_difference_between_two_scales_survives_the_nudge():
    """What the units argument actually rests on: h(cY) - h(Y) = log c, whatever the nudge."""
    y = np.concatenate([np.linspace(1.0, 2.0, 200), np.repeat([1.3, 1.7], 6)])
    for u in (1e-2, 1e-6, 1e-12):
        a = differential_entropy(y, tie_policy="nudge", nudge=u)
        b = differential_entropy(60.0 * y, tie_policy="nudge", nudge=u)
        assert b - a == pytest.approx(np.log(60.0), abs=1e-9)


def test_a_negative_estimate_is_reported_as_degenerate():
    from molace.measures.continuous_li import is_degenerate

    assert is_degenerate(-0.37) is True
    assert is_degenerate(float("nan")) is True
    assert is_degenerate(0.12) is False
