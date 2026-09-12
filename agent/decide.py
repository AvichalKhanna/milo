"""Read-step: classify moment -> hold | explore | move_forward, outputting confidence.

Pipeline order (all unconditional, all logged):
    1. risk_gate.check_risk()   — crisis-adjacent language, always checked first
    2. is_pure_greeting()       — "hey", "hello", etc. — zero API cost
    3. is_pure_gratitude()      — "thanks", "thank you" — zero API cost
    4. LLM rubric               — everything else

Every path logs to the decisions table and the trace table, so every
decision the agent makes is auditable after the fact — this is the
"read before replying, and show your work" requirement from the spec.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from agent.llm import call_llm, MODEL
from agent.risk_gate import check_risk

GREETINGS_SET = frozenset({
    "hello", "hi", "hey", "hey there", "hi there", "hello there",
    "good morning", "good afternoon", "good evening", "howdy",
    "sup", "whats up", "what's up", "how are you", "how are you doing", "yo",
    "milo", "hello milo", "hi milo", "hey milo", "greetings",
})

GRATITUDE_SET = frozenset({
    "thanks", "thank you", "thanks a lot", "thank u", "ty", "thx",
    "appreciate it", "i appreciate it", "appreciate that", "thanks so much",
    "thank you so much", "means a lot", "that means a lot",
})


def is_pure_greeting(msg: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", msg.lower()).strip()
    return clean in GREETINGS_SET


def is_pure_gratitude(msg: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", msg.lower()).strip()
    return clean in GRATITUDE_SET


_RUBRIC = """
You are the read-step classifier for Milo, a supportive listening companion.
Given a user message and structured context, classify the moment into exactly one category:
hold | explore | move_forward.

GREETINGS / CASUAL CHECK-INS — choose EXPLORE:
- Casual greetings: "hello", "hi", "hey", "good morning", "how are you"
- Neutral small-talk with no distress or heavy disclosure
- NEVER choose HOLD for simple greetings or casual openings!

HOLD — choose when:
- Fresh emotional disclosure of distress, hurt, grief, or pain
- Intensity is high — person sounds overwhelmed, sad, shocked, hurt, scared
- Person is venting — even if phrased as a question
- Rhetorical pain-questions like "why does everyone always leave me?",
  "what is wrong with me?" are ACUTE DISTRESS, not reflection — HOLD.
- CRITICAL MISCLASSIFICATION TO AVOID: "Help me please, my [pet/person] is
  very ill" and similar phrases LOOK like a request for action (move_forward
  surface pattern: "help me") but are actually FRESH EMOTIONAL DISCLOSURE.
  The surface phrase "help me" does NOT make something move_forward if it
  is paired with fear, grief, or a just-happened crisis. Classify HOLD.
- "Answer me" / "tell me" following unanswered distress is agitation, not
  a calm practical request — stays HOLD, not move_forward.
- Default choice for ambiguous expressions of pain

EXPLORE — choose when:
- Casual greetings, openings, or checking in
- Simple acknowledgments or gratitude ("thanks", "that means a lot") when
  no new distress is present — this is closure/connection, not fresh pain
- Person explicitly invites genuine reflection ("why do I keep doing this?")
  AND the tone is calm, not desperate
- Person is mid-processing, thinking out loud in a stable emotional state
- They want understanding more than action

MOVE_FORWARD — choose ONLY when:
- Person explicitly asks for concrete logistical/practical steps AND the
  emotional charge is already processed or clearly secondary
  (e.g. "what's the best way to word this email", "should I call the vet
  or the emergency clinic" asked in a level tone)
- Situation stated as resolved or clearly in the past
- Do NOT choose move_forward just because the word "help" appears — check
  whether the person is emotionally activated right now. If yes, it's HOLD.

ADD-ON 3 (SELECTIVE FORGETTING):
If context contains 'stale_threads', they are inactive/carried-forward
situations that need to be VERIFIED, not assumed still true. If the current
message doesn't clearly reference one, don't treat it as live context.

CONFIDENCE SCORING (0.0 to 1.0):
>= 0.75: Very clear indicator present.
0.4 - 0.75: Mixed cues or moderate ambiguity.
< 0.4: Highly ambiguous, cryptic, or conflicting signals.

Return ONLY valid JSON (no markdown fences):
{
  "decision": "hold"|"explore"|"move_forward",
  "confidence": 0.0-1.0,
  "reason": "1 sentence explanation",
  "is_resolved": false
}
"""

VALID_DECISIONS = frozenset({"hold", "explore", "move_forward"})


def decide(user_msg: str, context: dict, person_id: str) -> dict:
    """Classify message; log to decisions and trace tables. Returns decision dict.

    Risk-gate runs first and unconditionally (Add-on 5). If flagged, decide()
    short-circuits with decision='hold', confidence=1.0, and risk_flag=True —
    reply.py checks risk_flag and overrides to the fixed safe response
    regardless of decision content.
    """
    input_hash = hashlib.sha256(user_msg.encode()).hexdigest()[:16]

    # ── ADD-ON 5: Independent risk gate — runs before EVERYTHING else ──
    risk_result = check_risk(user_msg)
    if risk_result["flagged"]:
        result = {
            "decision": "hold",
            "confidence": 1.0,
            "reason": f"risk_gate triggered: {risk_result['reason']}",
            "is_resolved": False,
            "input_hash": input_hash,
            "model_version": "risk_gate_v1",
            "confidence_band": "direct_tentative",
            "risk_flag": True,
            "risk_reason": risk_result["reason"],
            "logged_to_db": True,
        }
        _log_decision(person_id, input_hash, result)
        from agent.memory import log_trace, log_risk_flag
        log_risk_flag(person_id, input_hash, risk_result["reason"])
        log_trace(person_id, "risk_gate_triggered", {"reason": risk_result["reason"]})
        return result

    # ── Fast path: pure greeting, 0 tokens ──
    if is_pure_greeting(user_msg):
        result = {
            "decision": "explore",
            "confidence": 0.90,
            "reason": "Casual greeting / check-in",
            "is_resolved": False,
            "input_hash": input_hash,
            "model_version": MODEL,
            "confidence_band": "direct_tentative",
            "risk_flag": False,
            "logged_to_db": True,
        }
        _log_decision(person_id, input_hash, result)
        from agent.memory import log_trace
        log_trace(person_id, "confidence_band", {"confidence": 0.90, "band": "direct_tentative"})
        return result

    # ── Fast path: pure gratitude/acknowledgment, 0 tokens ──
    if is_pure_gratitude(user_msg):
        result = {
            "decision": "explore",
            "confidence": 0.85,
            "reason": "Simple acknowledgment or gratitude — not fresh distress",
            "is_resolved": False,
            "input_hash": input_hash,
            "model_version": MODEL,
            "confidence_band": "direct_tentative",
            "risk_flag": False,
            "logged_to_db": True,
        }
        _log_decision(person_id, input_hash, result)
        from agent.memory import log_trace
        log_trace(person_id, "confidence_band", {"confidence": 0.85, "band": "direct_tentative"})
        return result

    context_str = _format_context(context)

    prompt = (
        f"Structured context (do not invent facts beyond these rows):\n{context_str}\n\n"
        f'Current user message: "{user_msg}"\n\nClassify this moment.'
    )

    try:
        raw = call_llm(prompt, system=_RUBRIC, json_mode=True, temperature=0.15)
        result = json.loads(raw)
    except Exception:
        result = {
            "decision": "hold",
            "confidence": 0.35,
            "reason": "parse/API error — defaulting to hold with low confidence",
            "is_resolved": False,
        }

    if result.get("decision") not in VALID_DECISIONS:
        result["decision"] = "hold"

    try:
        confidence = float(result.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5
    result["confidence"] = confidence

    result["input_hash"] = input_hash
    result["model_version"] = MODEL
    result["risk_flag"] = False

    if confidence >= 0.75:
        band = "direct_tentative"
    elif confidence >= 0.4:
        band = "soft_hedge"
    else:
        band = "question_only"
    result["confidence_band"] = band

    _log_decision(person_id, input_hash, result)
    result["logged_to_db"] = True

    from agent.memory import log_trace
    log_trace(person_id, "confidence_band", {"confidence": confidence, "band": band})

    return result


def _log_decision(person_id: str, input_hash: str, result: dict) -> None:
    from agent.memory import get_db
    conn = get_db()
    conn.execute(
        "INSERT INTO decisions"
        " (timestamp, person_id, input_hash, decision, confidence, reason, model_version)"
        " VALUES (?,?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            person_id,
            input_hash,
            result["decision"],
            result.get("confidence", 0.5),
            result.get("reason", ""),
            result.get("model_version", ""),
        ),
    )
    conn.commit()
    conn.close()


def _format_context(context: dict) -> str:
    parts: list[str] = []
    if context.get("last_session_summary"):
        parts.append(f"Last session summary: {context['last_session_summary']['summary']}")
    if context.get("situations"):
        for s in context["situations"][:3]:
            parts.append(
                f"Open situation ({s['area_of_life']}): {s['description'][:80]}"
                f" [intensity {s.get('intensity', '?')}/5]"
            )
    if context.get("stale_threads"):
        for st in context["stale_threads"][:2]:
            parts.append(f"Stale thread (CHECK ONLY, DO NOT ASSUME): ({st.get('area', 'other')}) {st.get('description', '')[:80]}")
    if context.get("escalating"):
        parts.append("NOTE: recent intensity readings are trending above this person's baseline.")
    if not parts:
        parts.append("No prior context — new person or first message.")
    return "\n".join(parts)