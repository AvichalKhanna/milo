#!/usr/bin/env python3
"""Milo interactive CLI — premium feel, zero fake latency.

TEST BUILD ONLY. Name/session and API key prompts are for local testing.

Usage:
    python cli.py                        # interactive: pick/create identity, enter key
    python cli.py alice                  # skip prompts, use 'alice' directly
    python cli.py alice --now +2d        # simulate time forward, run check-ins, then chat
    python cli.py alice --debug
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import os
import threading
import time
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Dependency check: verify required packages, install if missing ──────────
def _ensure_dependencies() -> None:
    """Check for required third-party packages. Attempt to install any that
    are missing via pip. If installation still fails after attempting,
    alert the user clearly and exit — rather than crashing later with a
    confusing ImportError buried deep in a stack trace."""
    # (import_name, pip_install_name) — these can differ (e.g. python-dotenv -> dotenv)
    required = [
        ("dotenv", "python-dotenv"),
        ("openai", "openai"),
        ("yaml", "pyyaml"),
    ]

    missing: list[str] = []
    for import_name, pip_name in required:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return

    print(f"Missing required package(s): {', '.join(missing)}")
    print("Attempting to install automatically...\n")

    for pip_name in missing:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  Installed {pip_name}")
        except subprocess.CalledProcessError:
            print(
                f"\nERROR: Could not automatically install '{pip_name}'.\n"
                f"Please install it manually and try again:\n\n"
                f"    pip install {pip_name}\n\n"
                f"Or install everything at once:\n\n"
                f"    pip install -r requirements.txt\n"
            )
            sys.exit(1)

    # Re-verify after install attempts, in case something silently failed
    still_missing = []
    for import_name, pip_name in required:
        try:
            importlib.import_module(import_name)
        except ImportError:
            still_missing.append(pip_name)

    if still_missing:
        print(
            f"\nERROR: The following package(s) are still missing after "
            f"installation attempts: {', '.join(still_missing)}\n\n"
            f"Please install manually:\n\n"
            f"    pip install {' '.join(still_missing)}\n\n"
            f"Or:\n\n"
            f"    pip install -r requirements.txt\n"
        )
        sys.exit(1)

    print()  # blank line before banner for a clean startup


_ensure_dependencies()

from agent.memory import init_db, load_context, close_session, list_known_people
from agent.pipeline import process_turn
from agent.checkin import run_checkins, parse_advance


class C:
    RESET        = "\033[0m"
    DIM          = "\033[2m"
    CYAN         = "\033[36m"
    SOFT_MAGENTA = "\033[35m"
    GRAY         = "\033[90m"
    BOLD         = "\033[1m"
    YELLOW       = "\033[33m"


BANNER = f"""{C.DIM}
+------------------------------------------+
|  {C.RESET}{C.BOLD}Milo{C.RESET}{C.DIM}  ·  a space to be heard            |
|  Type  quit / exit  to end the session   |
+------------------------------------------+{C.RESET}
"""

TEST_BUILD_NOTICE = (
    f"{C.YELLOW}This is a TEST BUILD. The name and API key entered here are "
    f"for local testing only — do not share this build or enter a production key.{C.RESET}\n"
)

_MIN_INDICATOR_SECS = 0.6
_TICK = 0.18


def _typing_indicator_until(stop_event: threading.Event, label: str = "Milo") -> None:
    frames = ["·", "··", "···", "··", "·", ""]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{C.GRAY}{label} is typing{frames[i % len(frames)]}   {C.RESET}")
        sys.stdout.flush()
        stop_event.wait(timeout=_TICK)
        i += 1
    sys.stdout.write("\r" + " " * 42 + "\r")
    sys.stdout.flush()


def _stream_reply(reply: str, label: str = "Milo", delay: float = 0.0) -> None:
    base_cps = max(0.008, min(0.032, delay * 0.006))
    prefix = f"{C.CYAN}{label}:{C.RESET} "
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for ch in reply:
        sys.stdout.write(ch)
        sys.stdout.flush()
        pause = base_cps * (2.0 if ch in ".,!?\n" else 1.0)
        time.sleep(pause)
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def _select_identity() -> str:
    known = list_known_people()
    if not known:
        name = input(f"{C.SOFT_MAGENTA}No past conversations found. Enter a name to start one: {C.RESET}").strip()
        return name or "default_user"

    print(f"{C.DIM}Past conversations:{C.RESET}")
    for i, p in enumerate(known, 1):
        label = p.get("name") or p["person_id"]
        last_active = (p.get("last_mentioned") or "")[:10]
        print(f"  {C.CYAN}{i}.{C.RESET} {label}  {C.DIM}(last active {last_active}){C.RESET}")
    new_option = len(known) + 1
    print(f"  {C.CYAN}{new_option}.{C.RESET} Start a new conversation")

    choice = input(f"{C.SOFT_MAGENTA}Select a number: {C.RESET}").strip()
    try:
        idx = int(choice)
    except ValueError:
        idx = new_option

    if 1 <= idx <= len(known):
        return known[idx - 1]["person_id"]

    name = input(f"{C.SOFT_MAGENTA}Enter a name for the new conversation: {C.RESET}").strip()
    return name or "default_user"


def _select_api_key() -> None:
    entered = input(f"{C.SOFT_MAGENTA}Groq API key (or type 'default' to use the configured test key): {C.RESET}").strip()

    normalized = entered.lower().replace(" ", "")
    default_typos = {"default", "deafult", "defualt", "defalut", "dfault", ""}

    if normalized in default_typos:
        print(f"{C.DIM}Using the default configured key.{C.RESET}")
    else:
        os.environ["GROQ_API_KEY"] = entered
        print(f"{C.DIM}Using the key you entered for this session only.{C.RESET}")

    from agent.llm import _get_client
    _, provider = _get_client()
    if provider == "Local Heuristic Simulator":
        print(
            f"{C.YELLOW}⚠ No valid API key detected — running in OFFLINE SIMULATOR mode.\n"
            f"  Replies will be template-based, not model-generated.{C.RESET}"
        )
    else:
        print(f"{C.DIM}Connected: {provider}.{C.RESET}")
    print()


def _run_time_simulation(now_arg: str) -> None:
    """Wire --now into the same entry point as the chat session: simulate
    time passing, run the check-in scanner once, print any generated
    check-ins before the interactive loop starts."""
    delta = parse_advance(now_arg)
    sim_now = datetime.utcnow() + delta
    print(f"{C.DIM}Simulating time forward to {sim_now.isoformat()} ...{C.RESET}")

    generated = run_checkins(simulated_now=sim_now, threshold_days=2)
    if not generated:
        print(f"{C.DIM}No check-ins were due.{C.RESET}\n")
        return

    print(f"{C.DIM}Check-ins sent:{C.RESET}")
    for ci in generated:
        print(f"  {C.CYAN}[{ci['person_id']}]{C.RESET} {ci['message']}")
    print()


def chat(person_id: str, skip_prompts: bool = False) -> None:
    init_db()
    print(BANNER)

    if not skip_prompts:
        print(TEST_BUILD_NOTICE)
        person_id = _select_identity()
        _select_api_key()
        print()

    session_turns: list[dict] = []

    try:
        while True:
            try:
                user_msg = input(f"{C.SOFT_MAGENTA}You:{C.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_msg:
                continue
            if user_msg.lower() in ("quit", "exit", "bye", "q"):
                break

            result: list = [None]
            exc: list    = [None]
            stop_event   = threading.Event()

            def _llm_worker() -> None:
                try:
                    ctx = load_context(person_id)
                    result[0] = process_turn(user_msg, ctx, person_id)
                except Exception as e:  # noqa: BLE001
                    exc[0] = e
                finally:
                    stop_event.set()

            worker = threading.Thread(target=_llm_worker, daemon=True)
            t_start = time.monotonic()
            worker.start()

            _typing_indicator_until(stop_event)

            elapsed = time.monotonic() - t_start
            if elapsed < _MIN_INDICATOR_SECS:
                time.sleep(_MIN_INDICATOR_SECS - elapsed)

            worker.join()

            if exc[0] is not None:
                print(f"{C.DIM}[Something went wrong — please try again.]{C.RESET}\n")
                continue

            decision, reply, facts, delay = result[0]
            _stream_reply(reply, delay=delay)

            session_turns.append({"user": user_msg, "decision": decision["decision"], "reply": reply})

    finally:
        if session_turns:
            summary = _summarise_session(session_turns)
            close_session(person_id, summary)
            print(f"{C.DIM}[Session saved. Take care.]{C.RESET}")


def _summarise_session(turns: list[dict]) -> str:
    topics = [t["user"][:50] for t in turns if len(t["user"]) > 4 and t["user"].lower() not in ("hi", "hello", "hey")]
    if topics:
        return f"Session covered: {'; '.join(topics[:2])}."
    return "Session completed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milo — a space to be heard")
    parser.add_argument("person", nargs="?", default=None, help="Person ID. If given, skips identity/key prompts.")
    parser.add_argument("--debug", action="store_true", help="Show LLM debug output")
    parser.add_argument("--now", default=None, help="Simulate time forward before chatting, e.g. +2d")
    args = parser.parse_args()

    if args.debug:
        os.environ["MILO_DEBUG"] = "1"

    init_db()

    if args.now:
        _run_time_simulation(args.now)

    if args.person:
        chat(args.person, skip_prompts=True)
    else:
        chat("default_user", skip_prompts=False)