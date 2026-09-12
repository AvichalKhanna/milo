"""Memory store: SQLite read/write/extract with Selective Forgetting,
Metaphor Callback (same-turn + persisted), Baseline Drift Detection,
Repetition Tracking, and Trace Logging.

Rule: reply/checkin generation is given only structured rows, never raw
transcript — prevents 'reciting a file' and claiming things unsaid.

Session-boundary fix: a situation not touched during the most recently
CLOSED session is flipped to 'stale' the next time load_context() runs —
independent of STALE_DAYS. This stops a situation from a prior conversation
(e.g. a cat mentioned last week) being asserted as live fact just because
it isn't old enough yet under the day-based threshold alone.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Any

DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "store", "memory.db")
STALE_DAYS: int = 5
ESCALATION_SD_THRESHOLD: float = 1.5


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS people (
    person_id          TEXT PRIMARY KEY,
    name               TEXT,
    relation           TEXT,
    first_mentioned    TEXT,
    last_mentioned     TEXT,
    paused             INTEGER DEFAULT 0,
    checkin_propensity REAL DEFAULT 0.6,
    baseline_mean       REAL DEFAULT 3.0,
    baseline_var        REAL DEFAULT 0.0,
    baseline_n           INTEGER DEFAULT 0,
    last_metaphor        TEXT
);

CREATE TABLE IF NOT EXISTS situations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    area_of_life  TEXT,
    description   TEXT,
    intensity     INTEGER,
    status        TEXT DEFAULT 'open',
    created_at    TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    situation_id  INTEGER,
    need          TEXT,
    created_at    TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    started_at    TEXT,
    summary       TEXT
);

CREATE TABLE IF NOT EXISTS checkins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT,
    situation_id  INTEGER,
    sent_at       TEXT,
    message       TEXT,
    replied       INTEGER DEFAULT 0,
    stopped       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT,
    person_id     TEXT,
    input_hash    TEXT,
    decision      TEXT,
    confidence    REAL,
    reason        TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS trace (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT,
    person_id     TEXT,
    event_type    TEXT,
    details       TEXT
);

CREATE TABLE IF NOT EXISTS risk_flags (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT,
    person_id     TEXT,
    input_hash    TEXT,
    reason        TEXT
);
"""

_MIGRATIONS: list[tuple[str, str, str]] = [
    ("people", "checkin_propensity", "ALTER TABLE people ADD COLUMN checkin_propensity REAL DEFAULT 0.6"),
    ("people", "baseline_mean", "ALTER TABLE people ADD COLUMN baseline_mean REAL DEFAULT 3.0"),
    ("people", "baseline_var", "ALTER TABLE people ADD COLUMN baseline_var REAL DEFAULT 0.0"),
    ("people", "baseline_n", "ALTER TABLE people ADD COLUMN baseline_n INTEGER DEFAULT 0"),
    ("people", "last_metaphor", "ALTER TABLE people ADD COLUMN last_metaphor TEXT"),
]


def init_db() -> None:
    conn = get_db()
    conn.executescript(CREATE_SQL)
    for table, col, ddl in _MIGRATIONS:
        cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in cols:
            conn.execute(ddl)
    conn.commit()
    conn.close()


def log_trace(person_id: str, event_type: str, details: dict, conn: sqlite3.Connection | None = None) -> None:
    should_close = False
    if conn is None:
        conn = get_db()
        should_close = True
    conn.execute(
        "INSERT INTO trace (timestamp, person_id, event_type, details) VALUES (?,?,?,?)",
        (datetime.utcnow().isoformat(), person_id, event_type, json.dumps(details)),
    )
    if should_close:
        conn.commit()
        conn.close()


def log_risk_flag(person_id: str, input_hash: str, reason: str) -> None:
    conn = get_db()
    conn.execute(
        "INSERT INTO risk_flags (timestamp, person_id, input_hash, reason) VALUES (?,?,?,?)",
        (datetime.utcnow().isoformat(), person_id, input_hash, reason),
    )
    conn.commit()
    conn.close()


def log_reply(person_id: str, reply_text: str) -> None:
    """Log every sent reply so reply.py can detect cross-turn repetition."""
    log_trace(person_id, "reply_sent", {"text": reply_text})


def get_recent_closers(person_id: str, n: int = 3) -> list[str]:
    """Return the trailing question/closer fragment of the last N replies."""
    conn = get_db()
    rows = conn.execute(
        "SELECT details FROM trace WHERE person_id=? AND event_type='reply_sent'"
        " ORDER BY id DESC LIMIT ?",
        (person_id, n),
    ).fetchall()
    conn.close()

    closers: list[str] = []
    for r in rows:
        text = json.loads(r["details"]).get("text", "")
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        question_parts = [p for p in parts if "?" in p]
        if question_parts:
            closers.append(question_parts[-1])
    return closers


def list_known_people(limit: int = 20) -> list[dict]:
    """Return primary identities (not secondary 'mentioned' people, which use
    the 'parent::name' id format) for the session-picker menu, most recently
    active first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT person_id, name, last_mentioned FROM people"
        " WHERE person_id NOT LIKE '%::%'"
        " ORDER BY last_mentioned DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Same-turn metaphor detection (fast, regex-based, no LLM round-trip) ──
_METAPHOR_PATTERNS: list[str] = [
    r"\b(like a|like an|like i'?m|like i am)\s+([a-z][a-z\s]{3,40}?)(?:[.,!?]|$)",
    r"\b(feels? like)\s+([a-z][a-z\s]{3,40}?)(?:[.,!?]|$)",
    r"\b(carrying|dragging|hauling)\s+([a-z][a-z\s]{3,60}?)(?:[.,!?]|$)",
    r"\b(drowning in|buried under|trapped in|stuck in)\s+([a-z][a-z\s]{3,40}?)(?:[.,!?]|$)",
    r"\b(a (?:huge|big|heavy|massive)\s+(?:weight|burden|wall|fog|storm))\b",
]
_COMPILED_METAPHOR = [re.compile(p, re.IGNORECASE) for p in _METAPHOR_PATTERNS]


def detect_metaphor_fast(user_msg: str) -> str | None:
    """Lightweight same-turn metaphor detector. Returns the user's own phrase
    verbatim (trimmed), or None."""
    for pattern in _COMPILED_METAPHOR:
        m = pattern.search(user_msg)
        if m:
            start, end = m.span()
            phrase = user_msg[start:end].strip().rstrip(".,!?")
            if len(phrase) >= 6:
                return phrase
    return None


def check_and_flip_stale(person_id: str | None = None, now_dt: datetime | None = None) -> list[int]:
    """Flip open situations to 'stale' if EITHER:
      (a) untouched for STALE_DAYS (existing day-based rule), OR
      (b) untouched since before the start of the most recently CLOSED
          session (NEW — session-boundary rule). This is what stops a
          situation from a prior conversation being asserted as current
          just because it isn't old enough yet by day-count alone.
    """
    now = now_dt or datetime.utcnow()
    day_threshold = (now - timedelta(days=STALE_DAYS)).isoformat()

    conn = get_db()
    c = conn.cursor()

    session_boundary: str | None = None
    if person_id:
        last_sess = c.execute(
            "SELECT started_at FROM sessions WHERE person_id=? ORDER BY started_at DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        if last_sess:
            session_boundary = last_sess["started_at"]

    if person_id:
        if session_boundary:
            rows = c.execute(
                "SELECT * FROM situations WHERE person_id=? AND status='open'"
                " AND (updated_at < ? OR updated_at < ?)",
                (person_id, day_threshold, session_boundary),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM situations WHERE person_id=? AND status='open' AND updated_at < ?",
                (person_id, day_threshold),
            ).fetchall()
    else:
        rows = c.execute(
            "SELECT * FROM situations WHERE status='open' AND updated_at < ?", (day_threshold,),
        ).fetchall()

    flipped_ids: list[int] = []
    for r in rows:
        sit_id = r["id"]
        c.execute("UPDATE situations SET status='stale' WHERE id=?", (sit_id,))
        flipped_ids.append(sit_id)
        log_trace(r["person_id"], "stale_flip", {
            "situation_id": sit_id, "area_of_life": r["area_of_life"],
            "description": r["description"], "updated_at": r["updated_at"],
            "reason": "session_boundary" if session_boundary and r["updated_at"] < session_boundary else "day_threshold",
        }, conn=conn)

    conn.commit()
    conn.close()
    return flipped_ids


def _update_baseline(conn: sqlite3.Connection, person_id: str, new_intensity: float) -> bool:
    """Welford's online algorithm for running mean/variance. Returns True if escalating."""
    row = conn.execute(
        "SELECT baseline_mean, baseline_var, baseline_n FROM people WHERE person_id=?", (person_id,)
    ).fetchone()

    mean = row["baseline_mean"] if row and row["baseline_mean"] is not None else 3.0
    n = (row["baseline_n"] if row and row["baseline_n"] is not None else 0)
    prev_var = (row["baseline_var"] if row and row["baseline_var"] is not None else 0.0)
    var_sum = prev_var * max(n - 1, 1) if n > 1 else 0.0

    escalating = False
    if n >= 2:
        sd = math.sqrt(max(var_sum / max(n - 1, 1), 0.0))
        if sd > 0 and (new_intensity - mean) > ESCALATION_SD_THRESHOLD * sd:
            escalating = True

    n += 1
    delta = new_intensity - mean
    mean += delta / n
    delta2 = new_intensity - mean
    var_sum += delta * delta2
    var = var_sum / max(n - 1, 1) if n > 1 else 0.0

    conn.execute(
        "UPDATE people SET baseline_mean=?, baseline_var=?, baseline_n=? WHERE person_id=?",
        (mean, var, n, person_id),
    )
    return escalating


def load_context(person_id: str, now_dt: datetime | None = None) -> dict:
    """Return unified structured context. Never passes raw transcript."""
    check_and_flip_stale(person_id, now_dt)

    conn = get_db()
    c = conn.cursor()

    person = c.execute("SELECT * FROM people WHERE person_id=?", (person_id,)).fetchone()

    open_situations = c.execute(
        "SELECT * FROM situations WHERE person_id=? AND status='open' ORDER BY updated_at DESC LIMIT 5",
        (person_id,),
    ).fetchall()

    stale_situations = c.execute(
        "SELECT * FROM situations WHERE person_id=? AND status='stale' ORDER BY updated_at DESC LIMIT 5",
        (person_id,),
    ).fetchall()

    recent_signals = c.execute(
        "SELECT * FROM signals WHERE person_id=? ORDER BY created_at DESC LIMIT 10", (person_id,),
    ).fetchall()

    last_session = c.execute(
        "SELECT * FROM sessions WHERE person_id=? ORDER BY started_at DESC LIMIT 1", (person_id,),
    ).fetchone()

    last_intensity = 3.0
    if open_situations:
        last_intensity = float(open_situations[0]["intensity"] or 3.0)
    elif stale_situations:
        last_intensity = float(stale_situations[0]["intensity"] or 3.0)
    else:
        most_recent_any = c.execute(
            "SELECT intensity FROM situations WHERE person_id=? ORDER BY updated_at DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        if most_recent_any and most_recent_any["intensity"] is not None:
            last_intensity = float(most_recent_any["intensity"])
        elif person and person["baseline_mean"] is not None:
            last_intensity = float(person["baseline_mean"])

    checkin_propensity = 0.6
    if person and person["checkin_propensity"] is not None:
        checkin_propensity = float(person["checkin_propensity"])

    stale_threads = [
        {"area": s["area_of_life"], "description": s["description"], "id": s["id"]}
        for s in stale_situations
    ]

    escalating = False
    if person and person["baseline_n"] and person["baseline_n"] >= 2 and person["baseline_var"] is not None:
        sd = math.sqrt(max(person["baseline_var"], 0.0))
        if sd > 0 and (last_intensity - person["baseline_mean"]) > ESCALATION_SD_THRESHOLD * sd:
            escalating = True

    conn.close()

    return {
        "person": dict(person) if person else None,
        "situations": [dict(s) for s in open_situations],
        "stale_threads": stale_threads,
        "last_intensity": last_intensity,
        "checkin_propensity": checkin_propensity,
        "last_metaphor": (person["last_metaphor"] if person else None),
        "escalating": escalating,
        "open_situations": [dict(s) for s in open_situations],
        "recent_signals": [dict(s) for s in recent_signals],
        "last_session_summary": dict(last_session) if last_session else None,
    }


_EXTRACT_SYSTEM = """
You extract structured facts from a conversation turn for a supportive AI.
Return ONLY valid JSON — no markdown fences.

JSON schema:
{
  "self_name": str | null,
  "people_mentioned": [{"name": str, "relation": str}],
  "situation": {
    "area_of_life": "work|relationships|health|family|academic|financial|personal_growth|other",
    "description": str,
    "intensity": int,
    "apparent_need": "hold|explore|move_forward"
  } | null,
  "metaphor_phrase": str | null,
  "is_stop_signal": bool,
  "is_resolved": bool
}

self_name: if the user explicitly states THEIR OWN name (e.g. "my name is Avi",
"call me Avi", "I'm Avi"), extract it here as a properly capitalised string.
Otherwise null. Do NOT extract names of other people the user mentions here.

metaphor_phrase: if the person used a distinctive image or metaphor for how
they feel (e.g. "drowning", "the fog", "hit a wall"), extract it VERBATIM in
their own words. Otherwise null. Do not invent or paraphrase one.

Only include what is clearly stated. Set situation to null if no meaningful
situation is described. If user says the situation is resolved or over, set is_resolved to true.
"""


def extract(user_msg: str, reply_text: str, decision: str, person_id: str) -> dict:
    """Use LLM to extract structured facts from one conversation turn."""
    from agent.llm import call_llm

    prompt = (
        f"Person: {person_id}\n"
        f"User message: {user_msg}\n"
        f"Assistant reply: {reply_text}\n"
        f"Decision taken: {decision}\n\n"
        "Extract facts."
    )
    try:
        raw = call_llm(prompt, system=_EXTRACT_SYSTEM, json_mode=True, temperature=0.1)
        data = json.loads(raw)
    except Exception:
        data = {
            "people_mentioned": [], "situation": None,
            "metaphor_phrase": None, "is_stop_signal": False, "is_resolved": False,
        }

    if not data.get("metaphor_phrase"):
        fast = detect_metaphor_fast(user_msg)
        if fast:
            data["metaphor_phrase"] = fast

    return data


_STOP_WORDS = {
    "the", "a", "an", "is", "was", "are", "were", "i", "me", "my", "it",
    "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "that",
    "this", "have", "has", "had", "be", "been", "being", "do", "did", "does",
    "not", "no", "can", "could", "will", "would", "should", "about", "so",
    "very", "just", "really", "they", "them", "their", "there", "here",
    "what", "when", "where", "which", "who", "how", "from",
}


def write(person_id: str, facts: dict, decision_info: dict | None = None) -> int | None:
    """Persist extracted facts; dedup/merge situations; handle stale/resolved/metaphor/baseline."""
    conn = get_db()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()

    existing = c.execute("SELECT person_id FROM people WHERE person_id=?", (person_id,)).fetchone()
    if not existing:
        c.execute(
            "INSERT INTO people (person_id, name, first_mentioned, last_mentioned, paused, checkin_propensity, baseline_mean, baseline_var, baseline_n)"
            " VALUES (?,?,?,?,0,0.6,3.0,0.0,0)",
            (person_id, None, now, now),
        )
    else:
        c.execute("UPDATE people SET last_mentioned=? WHERE person_id=?", (now, person_id))

    if facts.get("is_stop_signal"):
        c.execute("UPDATE people SET paused=1 WHERE person_id=?", (person_id,))

    if facts.get("is_resolved") or (decision_info and decision_info.get("is_resolved")):
        c.execute(
            "UPDATE situations SET status='resolved', updated_at=? WHERE person_id=? AND status IN ('open', 'stale')",
            (now, person_id),
        )
        log_trace(person_id, "resolution_detected", {"status": "resolved", "timestamp": now}, conn=conn)

    metaphor = facts.get("metaphor_phrase")
    if metaphor:
        c.execute("UPDATE people SET last_metaphor=? WHERE person_id=?", (metaphor.strip(), person_id))
        log_trace(person_id, "metaphor_captured", {"phrase": metaphor.strip()}, conn=conn)

    self_name = (facts.get("self_name") or "").strip()
    if self_name:
        c.execute("UPDATE people SET name=? WHERE person_id=?", (self_name, person_id))
        log_trace(person_id, "self_name_learned", {"name": self_name}, conn=conn)

    for p in facts.get("people_mentioned") or []:
        sec_name = p.get("name", "").strip()
        if not sec_name:
            continue
        sec_id = f"{person_id}::{sec_name.lower().replace(' ', '_')}"
        existing_sec = c.execute("SELECT person_id, relation FROM people WHERE person_id=?", (sec_id,)).fetchone()
        if not existing_sec:
            c.execute(
                "INSERT INTO people (person_id, name, relation, first_mentioned, last_mentioned, paused, checkin_propensity, baseline_mean, baseline_var, baseline_n)"
                " VALUES (?,?,?,?,?,0,0.6,3.0,0.0,0)",
                (sec_id, sec_name, p.get("relation", ""), now, now),
            )
        else:
            new_rel = p.get("relation") or (existing_sec["relation"] if existing_sec else "")
            c.execute("UPDATE people SET last_mentioned=?, relation=? WHERE person_id=?", (now, new_rel, sec_id))

    situation_id: int | None = None
    situation = facts.get("situation")
    if situation and situation.get("description"):
        existing_sits = c.execute(
            "SELECT * FROM situations WHERE person_id=? AND status IN ('open', 'stale')", (person_id,),
        ).fetchall()

        desc_words = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", situation["description"].lower())
            if w not in _STOP_WORDS
        ][:4]

        matched: sqlite3.Row | None = None
        for s in existing_sits:
            if s["area_of_life"] == situation.get("area_of_life") and any(w in s["description"].lower() for w in desc_words):
                matched = s
                break

        intensity_val = situation.get("intensity") or (matched["intensity"] if matched else 3)

        if matched:
            c.execute(
                "UPDATE situations SET description=?, intensity=?, status='open', updated_at=? WHERE id=?",
                (situation["description"], intensity_val, now, matched["id"]),
            )
            situation_id = matched["id"]
        else:
            c.execute(
                "INSERT INTO situations (person_id, area_of_life, description, intensity, status, created_at, updated_at)"
                " VALUES (?,?,?,?,'open',?,?)",
                (person_id, situation.get("area_of_life", "other"), situation["description"], intensity_val, now, now),
            )
            situation_id = c.lastrowid

        c.execute(
            "INSERT INTO signals (person_id, situation_id, need, created_at) VALUES (?,?,?,?)",
            (person_id, situation_id, situation.get("apparent_need", "hold"), now),
        )

        escalating = _update_baseline(conn, person_id, float(intensity_val))
        if escalating:
            log_trace(person_id, "escalation_flagged", {
                "situation_id": situation_id, "intensity": intensity_val,
            }, conn=conn)

    if decision_info and decision_info.get("logged_to_db") is not True:
        c.execute(
            "INSERT INTO decisions (timestamp, person_id, input_hash, decision, confidence, reason, model_version)"
            " VALUES (?,?,?,?,?,?,?)",
            (now, person_id, decision_info.get("input_hash", ""), decision_info.get("decision", ""),
             decision_info.get("confidence", 0.0), decision_info.get("reason", ""), decision_info.get("model_version", "")),
        )

    conn.commit()
    conn.close()

    check_and_flip_stale(person_id)

    return situation_id


def is_paused(person_id: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT paused FROM people WHERE person_id=?", (person_id,)).fetchone()
    conn.close()
    return bool(row and row["paused"])


def close_session(person_id: str, summary: str) -> None:
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (person_id, started_at, summary) VALUES (?,?,?)",
        (person_id, datetime.utcnow().isoformat(), summary),
    )
    conn.commit()
    conn.close()