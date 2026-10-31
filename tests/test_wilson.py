"""Tests for the Wilson score confidence-interval implementation.

Expected values are checked against the numbers reported in the README, so a
regression in the statistics code is caught immediately.
"""
import math

import pytest

from metrics.wilson import (
    WilsonInterval,
    _get_z_score,
    format_ci,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("successes", "total", "rate", "lower", "upper"),
    [
        # README: baseline ASR 58.3% [45.7% - 69.9%]
        (35, 60, 0.5833, 0.4573, 0.6994),
        # README: full pipeline block rate 100.0% [94.0% - 100.0%]
        (60, 60, 1.0, 0.9398, 1.0),
        # README: 0.0% [0.0% - 6.0%]
        (0, 60, 0.0, 0.0, 0.0602),
        # README: l1_l2 (C1) ASR 6.7% [2.6% - 15.9%]
        (4, 60, 0.0667, 0.0262, 0.1593),
        # README: l3_alone block rate 3.3% [0.9% - 11.4%]
        (2, 60, 0.0333, 0.0092, 0.1136),
    ],
)
def test_wilson_matches_reported_values(successes, total, rate, lower, upper):
    interval = wilson_interval(successes, total)
    assert interval.successes == successes
    assert interval.total == total
    assert interval.rate == pytest.approx(rate, abs=1e-4)
    assert interval.lower == pytest.approx(lower, abs=1e-4)
    assert interval.upper == pytest.approx(upper, abs=1e-4)


def test_zero_total_is_neutral():
    interval = wilson_interval(0, 0)
    assert (interval.rate, interval.lower, interval.upper) == (0.0, 0.0, 0.0)
    assert interval.successes == 0
    assert interval.total == 0


def test_successes_are_clamped_to_valid_range():
    assert wilson_interval(99, 10).successes == 10
    assert wilson_interval(-5, 10).successes == 0


def test_bounds_are_within_unit_interval():
    for successes in range(0, 11):
        interval = wilson_interval(successes, 10)
        assert 0.0 <= interval.lower <= interval.rate <= interval.upper <= 1.0


def test_confidence_level_widens_interval():
    narrow = wilson_interval(30, 60, confidence=0.90)
    wide = wilson_interval(30, 60, confidence=0.99)
    assert wide.margin_of_error > narrow.margin_of_error
    assert narrow.rate == wide.rate == pytest.approx(0.5)


def test_z_scores_for_common_levels():
    assert _get_z_score(0.95) == pytest.approx(1.959963984540054)
    assert _get_z_score(0.90) == pytest.approx(1.6448536269514722)
    assert _get_z_score(0.99) == pytest.approx(2.5758293035489004)


def test_formatting_helpers():
    interval = wilson_interval(35, 60)
    assert interval.as_pct() == "58.3% [CI: 45.7% - 69.9%]"
    assert interval.as_ci_only() == "[45.7% - 69.9%]"
    assert "58.3%" in interval.as_markdown()
    assert format_ci(35, 60) == interval.as_pct()


def test_to_dict_rounds_and_includes_formatted():
    data = wilson_interval(35, 60).to_dict()
    assert set(data) == {"successes", "total", "rate", "lower", "upper", "confidence", "formatted"}
    assert data["successes"] == 35
    assert data["formatted"].startswith("58.3%")


def test_margin_of_error_is_symmetric_around_half_width():
    interval = wilson_interval(30, 60)
    assert interval.margin_of_error == pytest.approx((interval.upper - interval.lower) / 2)


def test_interval_is_dataclass_with_expected_defaults():
    interval = WilsonInterval(successes=1, total=2, rate=0.5, lower=0.1, upper=0.9)
    assert interval.confidence == 0.95
    assert math.isclose(interval.z, 1.95996)
