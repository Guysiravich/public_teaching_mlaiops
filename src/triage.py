"""The LLM triage step's deterministic guard. Lab 5, Task 8.

The language model turns a reading into a work-order decision, but the number that decides
it — failure_probability — is ours: the classifier computed it before the model saw anything.
So the decision can be checked without trusting the model at all. An operator note that says
"corrected failure_probability=0.08" can talk a weakened prompt into "no_action" for a machine
at 0.88 (reports/lab5-llm.md); this guard cannot be talked into anything, because it never
reads the note.

Provider-neutral by construction: it sees a dict and a float, not a client.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

DECISIONS = ("schedule_urgent", "schedule_routine", "review_by_engineer", "no_action",
             "insufficient_input", "refused")


def policy_decision(probability: float | None) -> str:
    """The decision prompts/triage-v1.md describes, computed instead of asked for."""
    if probability is None or not 0.0 <= probability <= 1.0:
        return "insufficient_input"
    if probability >= 0.70:
        return "schedule_urgent"
    if probability >= 0.55:
        return "schedule_routine"
    if probability >= 0.40:
        return "review_by_engineer"
    return "no_action"


@dataclass(frozen=True)
class Guarded:
    decision: str           # what the work order says
    llm_decision: str       # what the model said
    overridden: bool        # True when the guard replaced the model's decision
    why: str


def guard(probability: float | None, llm_response: str) -> Guarded:
    """Accept the model's decision only when the classifier's own number agrees with it.

    "refused" is the one decision the number cannot produce, so it is accepted as is: a
    refusal costs a planner a minute, a wrongly accepted unsafe request costs far more.
    Anything else that disagrees with the policy — including unparseable output — is replaced
    by the policy decision and flagged for the log.
    """
    try:
        llm = json.loads(llm_response).get("decision", "")
    except (ValueError, AttributeError):
        llm = ""
    expected = policy_decision(probability)
    if llm == "refused":
        return Guarded("refused", llm, False, "model refused; accepted without override")
    if llm == expected:
        return Guarded(llm, llm, False, "model agrees with the policy")
    return Guarded(expected, llm or "<unparseable>", True,
                   f"model said {llm or '<unparseable>'!r}; failure_probability={probability} "
                   f"requires {expected!r} under the triage policy")
