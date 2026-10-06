"""Lab 4 — unit tests for the drift statistics in monitoring/drift.py.

Unit tests in the Task 1 sense: they test our code, need no data file and no network, and
fail when someone changes the arithmetic — not when the data changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from monitoring.drift import ks_statistic, psi, verdict_for

RNG = np.random.default_rng(0)
REFERENCE = RNG.normal(80.0, 5.0, 5000)


def test_psi_is_near_zero_for_the_same_distribution():
    assert psi(REFERENCE, RNG.normal(80.0, 5.0, 5000)) < 0.02


def test_psi_grows_with_a_mean_shift():
    small = psi(REFERENCE, REFERENCE + 1.0)
    large = psi(REFERENCE, REFERENCE + 6.0)
    assert 0 < small < large


def test_psi_sees_a_change_of_spread_that_leaves_the_mean_alone():
    """The case a mean-based alert misses entirely: same average, wider spread."""
    centre = REFERENCE.mean()
    wider = centre + (REFERENCE - centre) * 1.5
    assert abs(wider.mean() - REFERENCE.mean()) < 0.01
    assert psi(REFERENCE, wider) > 0.1


def test_psi_is_finite_when_a_bin_is_empty():
    """Laplace smoothing: without it, an empty bin makes the log term infinite."""
    assert np.isfinite(psi(REFERENCE, np.full(500, 200.0)))


def test_ks_is_the_largest_gap_between_the_cdfs():
    assert ks_statistic(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert ks_statistic(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(1.0)


def test_verdict_boundaries():
    assert verdict_for(0.0) == "stable"
    assert verdict_for(10.0) == "significant"
