"""Evaluation maths: confidence intervals, kappa and balanced accuracy."""
import math

from ablation_gate import kappa, stats, wilson


def test_wilson_interval_matches_the_reported_clean_test_result():
    lo, hi = wilson(42, 48)               # reported as 88%, 95% CI 75 to 94%
    assert 75 < lo < 76 and 94 < hi < 95


def test_wilson_with_no_data():
    assert wilson(0, 0) == (0.0, 0.0)


def test_perfect_agreement_has_kappa_one():
    assert kappa([("yes", "yes"), ("no", "no")]) == 1.0


def test_kappa_is_undefined_when_everything_is_one_class():
    assert math.isnan(kappa([("yes", "yes"), ("yes", "yes")]))


def test_review_and_cant_tell_are_handled():
    st = stats([("yes", "yes"), ("yes", "review"), ("no", "no"), ("no", "yes"), ("cant_tell", "yes")])
    assert st["decided"] == 4
    assert (st["agree"], st["n_dec"]) == (2, 3)
    assert st["balanced"] == 75.0
