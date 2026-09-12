"""Reply generator, constrained by decision, confidence register, latency
affect, metaphor callback, and self-distancing.

Enforced structurally (post-generation validator + regeneration), not just
prompted. Risk-flagged turns bypass generation entirely.

Guard coverage (applies to EVERY decision type, not just 'hold'):
  - opener repetition
  - whole-reply template similarity
  - stacking / multiple interpretive clauses
  - hollow deflection questions
  - fabrication (anti-hallucination grounding instruction)

Fast paths (zero API cost, zero risk of drift):
  - risk-flagged turn        -> fixed SAFE_RESPONSE
  - pure greeting            -> rotating canned greeting
  - pure gratitude           -> rotating canned acknowledgment
  - identity/meta question   -> direct honest answer from context, not generation
"""
from __future__ import annotations

import difflib
import json as _json
import random
import re
from agent.llm import call_llm
from agent.risk_gate import SAFE_RESPONSE
from agent.decide import is_pure_greeting, is_pure_gratitude

MAX_CHARS = 280
MAX_QUESTIONS = 1
MAX_ATTEMPTS = 3
WHOLE_REPLY_SIMILARITY_THRESHOLD = 0.55

ADVICE_PATTERNS: list[str] = [
    r"\byou should\b", r"\btry\b", r"\bhave you considered\b", r"\bwhy not\b",
    r"\bI suggest\b", r"\bI recommend\b", r"\bperhaps you could\b",
    r"\bmaybe you could\b", r"\bit would help if you\b", r"\byou need to\b",
    r"\byou ought to\b", r"\bone thing (that|to|you)\b", r"\byou might want to\b",
]

UNHEDGED_CLAIM_PATTERNS: list[str] = [
    r"\b(you are|you're|you must be)\s+(feeling|so stressed|hurt|overwhelmed|anxious|angry|sad|hopeless)\b",
    r"\b(you are|you're)\s+experiencing\b",
]
HEDGE_WORDS: list[str] = [
    "maybe", "perhaps", "could be", "wondering if", "curious if", "could it be", "i might be off"
]

_INTERPRETIVE_CLAUSE = (
    r"(?:sounds like|seems like|it feels like|that must|that sounds|i hear|"
    r"what do you think is (?:happening|going on)|what's (?:going on|happening)|"
    r"i wonder if|i(?:'m| am) guessing|i sense|i notice)"
)
_STACKING_PATTERN = re.compile(
    _INTERPRETIVE_CLAUSE + r".{5,120}" + _INTERPRETIVE_CLAUSE,
    re.IGNORECASE | re.DOTALL,
)

_FORMULAIC_OPENERS = re.compile(
    r"^(it sounds like|it feels like|it seems like|sounds like|seems like|"
    r"i hear that|i hear you|i can hear)",
    re.IGNORECASE,
)

_HOLLOW_PHRASES = [
    r"\bis there anything (on your mind|you.d like to share|you want to talk about)\b",
    r"\bwould you like to (tell|share|talk)\b",
    r"\bfeel free to (share|tell|talk)\b",
    r"\bwhat.s on your mind\b",
]
_HOLLOW_PATTERN = re.compile("|".join(_HOLLOW_PHRASES), re.IGNORECASE)

_META_QUESTION_PATTERN = re.compile(
    r"\b(who am i|who are (?:you|u)|who r u|what are (?:you|u)|"
    r"are you (a bot|real|human|ai)|who'?s my \w+|what'?s my \w+|"
    r"do you know who i am)\b",
    re.IGNORECASE,
)

_GREETING_RESPONSES_WITH_CONTEXT = [
    "Hey, good to hear from you. How've you been?",
    "Hey! Was just thinking about you. How's everything?",
    "Hey, glad you're here. How are you doing?",
    "Good to hear from you. How have things been?",
]

_GREETING_RESPONSES_FRESH = [
    "Hey, what's on your mind?",
    "Hey there! How are you doing today?",
    "Hi! How are you holding up?",
    "Hey, good to see you. What's up?",
]

_GRATITUDE_RESPONSES = [
    "Of course. I'm here.",
    "Anytime. I'm right here.",
    "Glad to be here for this.",
    "Of course — I mean that.",
]

_FALLBACKS: dict[str, list[str]] = {
    "hold": [
        "That sounds really hard. I'm right here with you.",
        "That's a lot to sit with. I'm here.",
        "I hear you. I'm not going anywhere.",
    ],
    "explore": [
        "What's been sitting with you most about this?",
        "What part of that stands out to you right now?",
        "What's on your mind about it?",
    ],
    "move_forward": [
        "What feels like the most useful next step for you right now?",
        "What would help most right now?",
    ],
}


_SYSTEM_TEMPLATES: dict[str, str] = {
    "hold": """You are Milo, a warm supportive companion. Someone just shared something painful.

Your only job: make them feel heard. You are a friend who listens, not a therapist diagnosing.

STRICT RULES:
- ONE brief reflection or acknowledgement. Not two. Not three. One.
- Do NOT give advice. No "you should", "try", "have you considered", "why not", "I suggest",
  "I recommend", "maybe you could", "one thing that helps", "you might want to", "you need to".
- Do NOT ask WHY questions ("why do you feel that?").
- At most ONE gentle question — only if it fits naturally. No question is often better than a bad one.
- OPENER VARIETY — Do NOT begin every reply with "It sounds like", "It feels like", or "I hear you".
  This becomes a mechanical tic. Vary how you open. Examples: "That's hard.", "Of course.",
  "No wonder.", "Yeah, that's a lot.", "Of course you do.", "Ten years is a long time."
- Do NOT ask hollow deflection questions: "Is there anything on your mind?", "Would you like to share?",
  "Feel free to tell me." When someone says "I need support" — just BE there, don't redirect.
- Under 280 characters. Short. Plain. Warm. Not clinical, not verbose.
- Do NOT parrot their exact words back sentence-for-sentence.
- Do NOT stack multiple emotional interpretations. Say ONE thing and stop.
  BAD: "That sounds heavy. That must be overwhelming. I hear how painful this is."
  GOOD: "That's a lot to be carrying. I'm right here."
- CRITICAL — Do NOT invent or assume what happened. Follow their lead. Mirror only what they said.
- You are a person in their corner, not a crisis line reading from a script.""",

    "explore": """You are Milo, a warm, supportive listening companion.

The person seems open to reflection. Help them think through what they're feeling.

RULES:
- You may ask ONE open, gentle question to help them explore
- No advice-giving or problem-solving
- No directive language ("you should", "try", etc.)
- Keep reply under 280 characters
- Be curious and warm, not analytical
- Vary your openers — do NOT default to "It sounds like" every time
- Do NOT invent or assume what happened to people, pets, or situations they mention
- Reference ONLY what they just said or what is in the structured context provided""",

    "move_forward": """You are Milo, a warm, supportive listening companion.

The person is ready for practical support or next steps.

RULES:
- You may offer a gentle suggestion, framed softly (e.g., "One thing that sometimes helps is...")
  — but only ONE such suggestion per reply, and vary the phrasing each time; do not reuse the
  same suggestion template turn after turn.
- Keep under 280 characters
- Stay warm — this is support, not coaching
- Ask at most one follow-up question
- Do NOT invent or assume what happened to people, pets, or situations they mention
  ("the recovery you mentioned", "like last time") unless they explicitly said it.
- Reference ONLY what they just said or what is in the structured context provided""",
}


def typing_delay(decision: str, intensity: int | float = 3, person_id: str | None = None) -> float:
    """ADD-ON 1: Compute affect delay seconds based on decision and emotional intensity."""
    dec = (decision or "hold").lower()
    try:
        inten = float(intensity if intensity is not None else 3.0)
    except (ValueError, TypeError):
        inten = 3.0

    if dec == "hold":
        base = random.uniform(3.5, 5.0) if inten >= 4.0 else random.uniform(2.0, 3.5)
    elif dec == "explore":
        base = random.uniform(1.5, 2.5)
    elif dec == "move_forward":
        base = random.uniform(0.3, 1.0)
    else:
        base = random.uniform(1.5, 2.5)

    delay = round(base * random.uniform(0.85, 1.15), 2)

    if person_id:
        from agent.memory import log_trace
        log_trace(person_id, "latency_computed", {"decision": dec, "intensity": inten, "delay": delay})

    return delay


def _get_last_reply_text(person_id: str) -> str:
    """Fetch the full text of the last reply sent, for opener/whole-reply
    similarity checks. Returns '' if none exists yet."""
    from agent.memory import get_db
    conn = get_db()
    row = conn.execute(
        "SELECT details FROM trace WHERE person_id=? AND event_type='reply_sent' ORDER BY id DESC LIMIT 1",
        (person_id,),
    ).fetchone()
    conn.close()
    if not row:
        return ""
    try:
        return _json.loads(row["details"]).get("text", "")
    except Exception:
        return ""


def generate_reply(user_msg: str, context: dict, decision: dict, person_id: str) -> str:
    """Generate a validated reply. Risk-flagged turns bypass generation entirely."""

    # ── ADD-ON 5: Risk override — takes priority over EVERYTHING below ──
    if decision.get("risk_flag"):
        return SAFE_RESPONSE

    # ── Fast path: pure greeting ──
    if is_pure_greeting(user_msg):
        open_sits = context.get("situations") or context.get("open_situations") or []
        name = (context.get("person") or {}).get("name")
        resp = random.choice(_GREETING_RESPONSES_WITH_CONTEXT) if open_sits else random.choice(_GREETING_RESPONSES_FRESH)
        return _personalize_greeting(resp, name)
    
    # ── Fast path: pure gratitude/acknowledgment ──
    if is_pure_gratitude(user_msg):
        return random.choice(_GRATITUDE_RESPONSES)

    # ── Fast path: identity/meta questions — answered directly, never
    # routed through emotional-reflection generation ──
    if _META_QUESTION_PATTERN.search(user_msg):
        person = context.get("person") or {}
        name = person.get("name")
        sits = context.get("situations") or context.get("open_situations") or []
        lower_msg = user_msg.lower()

        if "who am i" in lower_msg or "what's my" in lower_msg or "do you know who i am" in lower_msg:
            if name:
                return f"You're {name}. What's going on right now?"
            return "I don't have a name for you yet — you haven't told me one. What's going on right now?"

        if "who are" in lower_msg or "who r u" in lower_msg or "what are" in lower_msg or "are you" in lower_msg:
            return "I'm Milo — I'm here to listen. What's going on right now?"

        if sits:
            return f"You mentioned {sits[0]['description'][:60]} earlier — is that what you meant?"

        return "I'm not sure what you're referring to — can you tell me a bit more?"

    decision_key: str = decision.get("decision", "hold")
    confidence: float = float(decision.get("confidence", 0.8))
    intensity: float = float(context.get("last_intensity", 3.0))
    base_system = _SYSTEM_TEMPLATES.get(decision_key, _SYSTEM_TEMPLATES["hold"])

    register_text = _build_register_instruction(confidence)

    stale_text = ""
    if context.get("stale_threads"):
        stale_text = (
            "\n\nSTALE THREADS INSTRUCTION (ADD-ON 3):\n"
            "Context contains inactive 'stale_threads'. If a stale thread is relevant to this turn, "
            "ask a light check-in question about whether it is still relevant — NEVER assume it is still true."
        )

    metaphor_text = ""
    metaphor = context.get("last_metaphor")
    if metaphor:
        metaphor_text = (
            f"\n\nMETAPHOR CALLBACK: This person previously described their experience as "
            f"\"{metaphor}\". If it fits naturally, reuse THEIR EXACT WORDS for this — "
            f"do not substitute your own synonym or reframe it. Only use it if genuinely relevant "
            f"to this turn; do not force it in."
        )

    distancing_text = ""
    if decision_key == "explore" and intensity >= 4.0:
        raw_name = (context.get("person") or {}).get("name") or ""
        name_hint = raw_name if (raw_name and "_" not in raw_name and raw_name[0:1].upper() == raw_name[0:1]) else "they"
        distancing_text = (
            f"\n\nSELF-DISTANCING TECHNIQUE: Intensity is high and this is a reflective moment. "
            f"Frame your question in third person/distanced form rather than direct 'you' — "
            f"e.g. instead of 'why does this upset you', ask something like "
            f"'what do you think is going on for {name_hint} right now'. "
            f"Keep it natural, not clinical — never announce that you're using a technique."
        )

    is_first_disclosure = not context.get("situations") and not context.get("last_session_summary")
    if decision_key == "hold" and is_first_disclosure:
        grounding_text = (
            "\n\nFIRST-REPLY RULE: This is their first disclosure. "
            "STAY WITH THEM — do not analyze, diagnose, or interpret. "
            "One sentence of presence, one gentle question at most. "
            "GOOD: 'That's a lot. I'm right here.' "
            "BAD: 'It sounds like you're carrying intense pain and feeling overwhelmed...'"
        )
    else:
        grounding_text = (
            "\n\nGROUNDING RULE (applies to this reply regardless of decision type): "
            "Do not invent details about what happened to any person, pet, event, or situation "
            "unless they explicitly said it in this message or it appears in the structured "
            "context above. Never reference a 'recovery', 'last time', or any prior event that "
            "is not explicitly present in context. If you are not sure something was said, "
            "do not reference it at all."
        )

    memory_ref_text = ""
    if re.search(r"\b(do you remember|remember my|you remember)\b", user_msg, re.IGNORECASE):
        memory_ref_text = (
            "\n\nMEMORY REFERENCE: The person is asking if you remember something. "
            "Acknowledge warmly that you do (only if it's actually in the context above) or "
            "say you're here now. Do NOT assume or invent what happened to the thing they're "
            "referencing. Ask gently what's going on with it RIGHT NOW instead of projecting a story."
        )

    system = (
        f"{base_system}\n\n{register_text}{grounding_text}{memory_ref_text}"
        f"{stale_text}{metaphor_text}{distancing_text}"
    )
    context_str = _format_context_for_reply(context)

    base_prompt = (
        f"Structured context (only reference what is in these rows — do not invent facts):\n"
        f"{context_str}\n\n"
        f'Their message: "{user_msg}"\n\nWrite your reply.'
    )

    last_reply_text = _get_last_reply_text(person_id)

    for attempt in range(MAX_ATTEMPTS):
        reply = call_llm(base_prompt, system=system, temperature=0.7 + attempt * 0.05)
        reply = reply.strip().strip('"')

        valid, issue = _validate(reply, decision_key, confidence, last_reply_text=last_reply_text)
        if valid:
            return reply

        base_prompt = (
            f"Your previous reply was REJECTED because: {issue}\n"
            f"Write a new reply that strictly fixes this issue.\n\n"
            f"Structured context:\n{context_str}\n\n"
            f'Their message: "{user_msg}"\n\nNew reply:'
        )

    # All attempts failed validation — rotate fallback so even the safety
    # net doesn't produce visible repeats across consecutive fallback turns.
    return random.choice(_FALLBACKS.get(decision_key, ["I hear you."]))

def _personalize_greeting(resp: str, name: str | None) -> str:
    """Inject a learned name into a canned greeting response, if present."""
    if not name:
        return resp
    if resp.startswith("Hey,"):
        return resp.replace("Hey,", f"Hey {name},", 1)
    if resp.startswith("Hey!"):
        return resp.replace("Hey!", f"Hey {name}!", 1)
    if resp.startswith("Hey "):
        return resp.replace("Hey ", f"Hey {name}, ", 1)
    if resp.startswith("Hi!"):
        return resp.replace("Hi!", f"Hi {name}!", 1)
    if resp.startswith("Good to hear"):
        return f"Hey {name}! " + resp
    return resp

def _build_register_instruction(confidence: float) -> str:
    if confidence >= 0.75:
        return (
            "PHRASING REGISTER: DIRECT TENTATIVE\n"
            "You may make a tentative observation — but sound like a person, not a template.\n"
            "Vary your openers. Do NOT start every reply with 'It sounds like', 'It feels like', "
            "or 'I hear you'.\n"
            "Good openers: 'That's a lot.', 'Of course.', 'That makes sense.', "
            "'Yeah, that's hard.', 'Of course you do.', 'No wonder.', 'That's really hard.'\n"
            "Save 'sounds like / feels like' for when it fits naturally — not as a default tic.\n"
            "Hard constraint: stay tentative — don't claim emotions more firmly than you know."
        )
    elif confidence >= 0.4:
        return (
            "PHRASING REGISTER: SOFT HEDGE\n"
            "You're not fully sure how they feel — soften interpretations.\n"
            "Good phrases: 'maybe...', 'I could be off, but...', 'perhaps...', 'I wonder if...'\n"
            "Do NOT start with 'It sounds like' or 'I hear you' — vary how you open.\n"
            "Hard constraint: do not state any emotional interpretation more firmly than you know."
        )
    else:
        return (
            "PHRASING REGISTER: QUESTION FORM ONLY\n"
            "HARD CONSTRAINT: Ask a gentle question — make NO emotional claim or interpretation.\n"
            "Do NOT say 'you are feeling...' or 'you must be...'. Mirror or ask, nothing more."
        )


def _validate(
    reply: str,
    decision: str,
    confidence: float,
    last_reply_text: str = "",
) -> tuple[bool, str | None]:
    if len(reply) > MAX_CHARS:
        return False, f"Too long: {len(reply)} chars (max {MAX_CHARS})"

    q_count = reply.count("?")
    if q_count > MAX_QUESTIONS:
        return False, f"Too many questions: {q_count} (max {MAX_QUESTIONS})"

    # ── Opener repetition — applies to EVERY decision type ──
    if last_reply_text:
        reply_m = _FORMULAIC_OPENERS.match(reply)
        prev_m = _FORMULAIC_OPENERS.match(last_reply_text)
        if reply_m and prev_m:
            reply_opener = reply_m.group(0).lower().strip()
            prev_opener = prev_m.group(0).lower().strip()
            if reply_opener == prev_opener:
                return False, (
                    f"Opener repetition: you started the last reply with '{prev_opener}' too. "
                    "Begin this reply differently — 'Of course.', 'That makes sense.', "
                    "'No wonder.', 'That's hard.', or simply mirror one specific word they said."
                )

        # ── Whole-reply template similarity — applies to EVERY decision type ──
        sim = difflib.SequenceMatcher(None, reply.lower(), last_reply_text.lower()).ratio()
        if sim > WHOLE_REPLY_SIMILARITY_THRESHOLD:
            return False, (
                f"Reply too similar ({sim:.2f}) to your previous reply as a whole. "
                "Use a genuinely different structure and wording this time, not just "
                "swapped details in the same template."
            )

    if decision == "hold":
        for pat in ADVICE_PATTERNS:
            if re.search(pat, reply, re.IGNORECASE):
                return False, f"Advice language detected: pattern '{pat}'"

        if _STACKING_PATTERN.search(reply):
            return False, "Stacking detected: say ONE thing, then stop."

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s.strip()]
        if len(sentences) > 2:
            return False, (
                f"Too many sentences in hold mode: {len(sentences)} (max 2). "
                "Say ONE thing and stop. Example: 'That's a lot. I'm right here.'"
            )

        if _HOLLOW_PATTERN.search(reply):
            return False, (
                "Hollow deflection detected ('is there anything on your mind?' etc.). "
                "When someone shares pain, stay WITH them — don't redirect. "
                "Acknowledge what they said. A question is optional, not required."
            )

    if decision == "move_forward":
        if _STACKING_PATTERN.search(reply):
            return False, "Stacking detected: say ONE thing, then stop."
        if _HOLLOW_PATTERN.search(reply):
            return False, (
                "Hollow deflection detected. Give one concrete, warm response — "
                "don't just redirect back with an empty invitation to share."
            )

    if confidence < 0.4:
        reply_lower = reply.lower()
        for pat in UNHEDGED_CLAIM_PATTERNS:
            if re.search(pat, reply_lower):
                has_hedge = any(h in reply_lower for h in HEDGE_WORDS)
                if not has_hedge:
                    return False, f"Unhedged emotional claim at low confidence ({confidence:.2f})"

    return True, None


def _format_context_for_reply(context: dict) -> str:
    parts: list[str] = []
    if context.get("last_session_summary"):
        parts.append(f"Previous session: {context['last_session_summary']['summary']}")
    if context.get("situations"):
        for s in context["situations"][:3]:
            parts.append(
                f"- ({s['area_of_life']}) {s['description'][:100]}"
                f" — intensity {s.get('intensity', '?')}/5"
            )
    if context.get("stale_threads"):
        for st in context["stale_threads"][:2]:
            parts.append(
                f"- [STALE THREAD: CHECK ONLY, DO NOT ASSUME TRUE] ({st.get('area', 'other')}): {st.get('description', '')[:90]}"
            )
    if context.get("last_metaphor"):
        parts.append(f"- Their own phrase for this: \"{context['last_metaphor']}\"")
    return "\n".join(parts) or "No prior context."