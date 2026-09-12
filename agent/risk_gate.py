"""ADD-ON 5: Independent risk gate.

Runs separately from decide.py's rubric classifier. Purpose is narrow:
catch crisis-adjacent language and force a fixed, non-generated safe
response + resources, regardless of what the main pipeline would have said.

Deliberately conservative (favors false positives over false negatives) and
deliberately simple — pattern-level detection only, not exhaustive keyword
enumeration. This is a safety floor, not a clever classifier: it should be
easy for a human reviewer to audit every line of why it fires.
"""
from __future__ import annotations

import re

# Pattern-level signals only — not an exhaustive phrase list.
# Each pattern is a general shape of crisis-adjacent language, not a
# lookup table of specific wording, so it generalizes rather than being
# gamed by rephrasing.
_RISK_PATTERNS: list[str] = [
    r"\b(kill|hurt|harm)\s+(myself|me)\b",
    r"\bwant(ed)?\s+to\s+die\b",
    r"\bend(ing)?\s+(it|my life|everything)\b",
    r"\bno\s+reason\s+to\s+(live|go on|keep going)\b",
    r"\bsuicid\w*\b",
    r"\bself[\s-]?harm\w*\b",
    r"\bbetter\s+off\s+(without me|dead|gone)\b",
    r"\bcan'?t\s+(go on|do this anymore|keep living)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _RISK_PATTERNS]

SAFE_RESPONSE: str = (
    "I'm really glad you told me. What you're carrying sounds heavy, and I want "
    "you to have real support right now, not just me. If you're in the US, you "
    "can reach the 988 Suicide & Crisis Lifeline by call or text, anytime. "
    "Are you safe right now?"
)


def check_risk(user_msg: str) -> dict:
    """Return {flagged: bool, reason: str|None, matched_pattern: str|None}.

    Runs BEFORE decide.py's rubric and before any greeting shortcut.
    High recall by design — a false positive here costs one templated
    reply; a false negative is the failure mode that actually matters.
    """
    for pattern in _COMPILED:
        m = pattern.search(user_msg)
        if m:
            return {
                "flagged": True,
                "reason": "crisis-adjacent language pattern matched",
                "matched_pattern": pattern.pattern,
            }
    return {"flagged": False, "reason": None, "matched_pattern": None}