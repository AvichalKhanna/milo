"""Turn orchestrator for Milo — Option A: separate decide -> reply -> extract
calls, with an unconditional risk-gate in front of everything.

This is the pipeline the spec asks for:
    risk_gate.check_risk()   (no API call - regex, runs first, always, inside decide())
    -> decide.decide()        (1 API call, or 0 on greeting fast-path, or 0 on risk-flag)
    -> reply.generate_reply() (1 API call, +1 per regeneration attempt, or 0 on risk-flag/greeting)
    -> memory.extract()       (1 API call, skipped on risk-flagged turns)
    -> memory.write()         (persists facts, updates baseline, metaphor, stale flips)

Cost: ~3 API calls/turn on the common path. Greeting turns: 0 calls.
Risk-flagged turns: 0 calls (decide short-circuits, reply returns the fixed
SAFE_RESPONSE, extract is skipped entirely).

This intentionally trades the old single-call "1 call, ~70% cheaper"
consolidation for a pipeline where the decision is demonstrably made BEFORE
the reply is written, and every add-on (risk-gate, repetition guard,
metaphor callback, self-distancing, baseline drift) actually runs on the
live path, per the take-home spec requirement #2:
"A separate step, before the reply is written, decides what the moment
needs... every decision is logged so we can see why the agent did what
it did."
"""
from __future__ import annotations

from agent.decide import decide
from agent.reply import generate_reply, typing_delay
from agent.memory import extract, write, log_reply, log_trace


def process_turn(
    user_msg: str,
    context: dict,
    person_id: str,
    force_confidence: float | None = None,
) -> tuple[dict, str, dict, float]:
    """Execute one full turn: read -> reply -> remember.

    Args:
        user_msg: the raw text the person sent this turn.
        context: output of memory.load_context(person_id) — structured rows only.
        person_id: stable identifier for this person.
        force_confidence: TEST HOOK ONLY. Overrides decide()'s returned
            confidence (and recomputes confidence_band) so scenario tests
            can force the 'soft_hedge' / 'question_only' phrasing registers
            deterministically without depending on model output. Never set
            this from cli.py or any live code path. Has no effect on
            risk-flagged turns (safety response is never confidence-gated).

    Returns:
        (decision_result, reply_text, facts, delay_seconds)
    """
    # ── Step 1: DECIDE ──
    # risk_gate.check_risk() runs INSIDE decide(), first, unconditionally —
    # before the greeting shortcut and before the LLM rubric. See decide.py.
    decision_result = decide(user_msg, context, person_id)

    if force_confidence is not None and not decision_result.get("risk_flag"):
        decision_result["confidence"] = float(force_confidence)
        if decision_result["confidence"] >= 0.75:
            decision_result["confidence_band"] = "direct_tentative"
        elif decision_result["confidence"] >= 0.4:
            decision_result["confidence_band"] = "soft_hedge"
        else:
            decision_result["confidence_band"] = "question_only"
        log_trace(person_id, "confidence_forced_for_test", {
            "forced_confidence": decision_result["confidence"],
            "band": decision_result["confidence_band"],
        })

    # ── Step 2: REPLY ──
    # Constrained by decision_result. If decision_result['risk_flag'] is
    # True, generate_reply() returns the fixed SAFE_RESPONSE immediately
    # and burns no API call. Otherwise this is where the repetition guard
    # (get_recent_closers), same-turn metaphor detection
    # (detect_metaphor_fast), and self-distancing framing all apply.
    reply_text = generate_reply(user_msg, context, decision_result, person_id)
    log_reply(person_id, reply_text)

    # ── Step 3: latency-as-affect ──
    intensity = context.get("last_intensity", 3)
    delay = typing_delay(decision_result["decision"], intensity, person_id=person_id)

    # ── Step 4: EXTRACT + WRITE ──
    # Skip extraction entirely on risk-flagged turns: we don't want crisis
    # language filed as a routine 'situation' row, and there's no reason to
    # spend an API call extracting facts from a turn whose reply is a fixed
    # template. We still call write() so pause/stop signals and the risk
    # event itself remain auditable via risk_flags + trace.
    if decision_result.get("risk_flag"):
        facts = {
            "people_mentioned": [],
            "situation": None,
            "metaphor_phrase": None,
            "is_stop_signal": False,
            "is_resolved": False,
        }
        log_trace(person_id, "risk_turn_skipped_extraction", {
            "input_hash": decision_result.get("input_hash", ""),
        })
    else:
        facts = extract(user_msg, reply_text, decision_result["decision"], person_id)

    write(person_id, facts, decision_info=decision_result)

    return decision_result, reply_text, facts, delay