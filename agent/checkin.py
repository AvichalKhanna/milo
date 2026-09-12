"""Self-initiated follow-up generator with Check-in Variance (Add-on 6).

Usage:
    python -m agent.checkin --now +2d       # simulate 2 days forward
    python -m agent.checkin --now +1w       # simulate 1 week forward
    python -m agent.checkin                 # real-time scan
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import datetime, timedelta

from agent.llm import call_llm
from agent.memory import log_trace

_CHECKIN_SYSTEM = """
You are Milo, a warm supportive companion sending a brief check-in.

RULES:
• Under 140 characters
• One sentence only
• Reference at least one specific word from the situation description provided
• Warm and non-intrusive — not alarming, not presumptuous
• No advice
• No questions that demand a full response — gentle, optional
"""

_STOP_WORDS = {
    "the", "a", "an", "is", "was", "are", "were", "i", "me", "my",
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "that", "this", "have", "has", "had", "be", "been", "do", "did",
    "not", "no", "can", "could", "will", "would", "just", "very", "really",
    "they", "them", "their", "there", "here", "what", "when", "where",
    "some", "about", "from", "also",
}


def parse_advance(s: str) -> timedelta:
    """Parse '+2d', '+1w', '+3h' into timedelta."""
    m = re.match(r"\+(\d+)([dhw])", s.strip())
    if not m:
        return timedelta(0)
    n, unit = int(m.group(1)), m.group(2)
    return {"d": timedelta(days=n), "h": timedelta(hours=n), "w": timedelta(weeks=n)}.get(unit, timedelta(0))


def base_gap_hours(intensity: int | float) -> float:
    """ADD-ON 6: Higher intensity -> shorter base gap."""
    try:
        val = float(intensity if intensity is not None else 3)
    except (ValueError, TypeError):
        val = 3.0

    if val >= 4.0:
        return 24.0  # 1 day for high intensity
    elif val >= 3.0:
        return 48.0  # 2 days for moderate intensity
    else:
        return 72.0  # 3 days for mild intensity


def run_checkins(
    simulated_now: datetime | None = None,
    threshold_days: int = 2,
    seed: int | None = None,
) -> list[dict]:
    """Scan situations and generate check-ins using propensity and jitter variance."""
    from agent.memory import get_db

    if seed is not None:
        random.seed(seed)

    now = simulated_now or datetime.utcnow()
    # Broad candidate threshold (e.g. 1 day minimum)
    candidate_threshold = (now - timedelta(hours=20)).isoformat()

    conn = get_db()
    c = conn.cursor()

    situations = c.execute(
        """
        SELECT s.*, p.paused, p.checkin_propensity, p.name AS person_name
        FROM   situations s
        JOIN   people p ON s.person_id = p.person_id
        WHERE  s.status    IN ('open', 'stale')
          AND  s.updated_at < ?
          AND  p.paused    = 0
        """,
        (candidate_threshold,),
    ).fetchall()

    generated: list[dict] = []

    for sit in situations:
        person_id = sit["person_id"]
        sit_id = sit["id"]
        intensity = sit["intensity"] or 3

        # HARD GUARD 1: Person paused
        if sit["paused"]:
            continue

        # HARD GUARD 2: Skip if unreplied checkin already exists
        unreplied = c.execute(
            "SELECT id FROM checkins"
            " WHERE person_id=? AND situation_id=? AND replied=0 AND stopped=0",
            (person_id, sit_id),
        ).fetchone()
        if unreplied:
            log_trace(person_id, "checkin_evaluated", {
                "situation_id": sit_id,
                "sent": False,
                "reason": "unreplied_exists",
            }, conn=conn)
            continue

        # ADD-ON 6: Variance calculation
        scheduled_gap = base_gap_hours(intensity)
        jitter = scheduled_gap * random.uniform(-0.3, 0.3)
        fire_gap_hours = scheduled_gap + jitter

        # Check elapsed time since updated_at
        try:
            last_touch = datetime.fromisoformat(sit["updated_at"])
            elapsed_hours = (now - last_touch).total_seconds() / 3600.0
        except Exception:
            elapsed_hours = 999.0

        if elapsed_hours < fire_gap_hours:
            log_trace(person_id, "checkin_evaluated", {
                "situation_id": sit_id,
                "scheduled_gap": scheduled_gap,
                "jitter": jitter,
                "fire_gap_hours": fire_gap_hours,
                "elapsed_hours": elapsed_hours,
                "sent": False,
                "reason": "gap_not_reached",
            }, conn=conn)
            continue

        # ADD-ON 6: Propensity skip roll
        propensity = sit["checkin_propensity"]
        if propensity is None:
            propensity = 0.6
        else:
            propensity = float(propensity)

        skip_roll = random.random() > propensity
        if skip_roll:
            log_trace(person_id, "checkin_evaluated", {
                "situation_id": sit_id,
                "scheduled_gap": scheduled_gap,
                "jitter": jitter,
                "propensity": propensity,
                "skip_roll": True,
                "sent": False,
                "reason": "propensity_skip",
            }, conn=conn)
            continue

        # Passes all guards & variance rolls — generate checkin
        msg = _generate_checkin(sit)
        keywords = _extract_keywords(sit["description"])

        # Validator: message must cite at least one keyword from situation
        if not any(kw.lower() in msg.lower() for kw in keywords):
            msg = _fallback_checkin(sit, keywords)

        c.execute(
            "INSERT INTO checkins (person_id, situation_id, sent_at, message, replied, stopped)"
            " VALUES (?,?,?,?,0,0)",
            (person_id, sit_id, now.isoformat(), msg),
        )
        generated.append({"person_id": person_id, "situation_id": sit_id, "message": msg})

        log_trace(person_id, "checkin_evaluated", {
            "situation_id": sit_id,
            "scheduled_gap": scheduled_gap,
            "jitter": jitter,
            "fire_gap_hours": fire_gap_hours,
            "skip_roll": False,
            "sent": True,
            "message": msg,
        }, conn=conn)

    conn.commit()
    conn.close()
    return generated


def mark_replied(person_id: str, situation_id: int) -> None:
    from agent.memory import get_db
    conn = get_db()
    conn.execute(
        "UPDATE checkins SET replied=1 WHERE person_id=? AND situation_id=? AND replied=0",
        (person_id, situation_id),
    )
    conn.commit()
    conn.close()


def _generate_checkin(sit) -> str:
    prompt = (
        f"Area of life: {sit['area_of_life']}\n"
        f"Situation: {sit['description']}\n\n"
        "Write one gentle check-in sentence:"
    )
    return call_llm(prompt, system=_CHECKIN_SYSTEM, temperature=0.7).strip()


def _extract_keywords(description: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", description)
    return [w for w in words if w.lower() not in _STOP_WORDS][:6]


def _fallback_checkin(sit, keywords: list[str]) -> str:
    kw = keywords[0] if keywords else sit["area_of_life"]
    return f"Hey, just wanted to check in — how are things going with the {kw} situation?"


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.memory import init_db

    parser = argparse.ArgumentParser(description="Milo check-in runner with variance")
    parser.add_argument("--now", default=None, help="Simulate time advance, e.g. +2d")
    parser.add_argument("--threshold", type=int, default=2, help="Inactivity threshold in days")
    parser.add_argument("--seed", type=int, default=None, help="Fixed random seed for testing")
    args = parser.parse_args()

    init_db()

    sim_now = None
    if args.now:
        delta = parse_advance(args.now)
        sim_now = datetime.utcnow() + delta
        print(f"Simulated time: {sim_now.isoformat()}")

    results = run_checkins(simulated_now=sim_now, threshold_days=args.threshold, seed=args.seed)
    if not results:
        print("No check-ins needed (or skipped via variance/propensity).")
    else:
        for ci in results:
            print(f"\n→ {ci['person_id']}: {ci['message']}")