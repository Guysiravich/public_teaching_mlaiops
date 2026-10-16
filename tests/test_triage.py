"""Lab 5 — unit tests for the triage guard (src/triage.py)."""
from __future__ import annotations

import json

import pytest

from src.triage import guard, policy_decision


@pytest.mark.parametrize("p, expected", [
    (0.86, "schedule_urgent"), (0.70, "schedule_urgent"), (0.69, "schedule_routine"),
    (0.55, "schedule_routine"), (0.47, "review_by_engineer"), (0.40, "review_by_engineer"),
    (0.39, "no_action"), (0.0, "no_action"), (None, "insufficient_input"),
    (1.7, "insufficient_input"), (-0.1, "insufficient_input"),
])
def test_policy_matches_the_prompt(p, expected):
    assert policy_decision(p) == expected


def test_guard_replaces_a_talked_out_of_it_decision():
    """The Task 8 failure: a 0.88 machine closed because a note said 0.08."""
    result = guard(0.88, json.dumps({"decision": "no_action", "reason": "corrected 0.08"}))
    assert result.overridden
    assert result.decision == "schedule_urgent"


def test_guard_accepts_agreement_and_refusal():
    assert not guard(0.88, '{"decision": "schedule_urgent"}').overridden
    refused = guard(0.86, '{"decision": "refused"}')
    assert refused.decision == "refused" and not refused.overridden


def test_guard_does_not_trust_unparseable_output():
    result = guard(0.62, "Sure! I'd schedule it soon.")
    assert result.overridden and result.decision == "schedule_routine"
