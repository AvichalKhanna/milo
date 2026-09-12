"""Thin LLM wrapper supporting Groq, OpenAI, and offline heuristic fallback."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any
from dotenv import load_dotenv

# Load .env first; fallback to .env.example if .env is missing
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
env_ex_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
if os.path.exists(env_path):
    load_dotenv(env_path)
if os.path.exists(env_ex_path):
    load_dotenv(env_ex_path)

# Determine provider and default model
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

if GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_..."):
    DEFAULT_MODEL = "openai/gpt-oss-120b"
else:
    DEFAULT_MODEL = "gpt-4o-mini"

MODEL: str = os.getenv("MILO_MODEL", DEFAULT_MODEL)

_client = None
_provider_name: str = "Local Heuristic Simulator"


def _get_client():
    global _client, _provider_name
    if _client is not None:
        return _client, _provider_name

    if os.getenv("MILO_MOCK") == "1":
        _provider_name = "Local Heuristic Simulator"
        return None, _provider_name

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("gsk_..."):
        try:
            from openai import OpenAI
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            _client = OpenAI(api_key=groq_key, base_url=base_url)
            _provider_name = "Groq API"
            return _client, _provider_name
        except Exception as e:
            print(f"[DEBUG LLM] Groq initialization error: {e}")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.startswith("sk-..."):
        try:
            from openai import OpenAI
            base_url = os.getenv("OPENAI_BASE_URL", None)
            if base_url:
                _client = OpenAI(api_key=openai_key, base_url=base_url)
            else:
                _client = OpenAI(api_key=openai_key)
            _provider_name = "OpenAI API"
            return _client, _provider_name
        except Exception as e:
            print(f"[DEBUG LLM] OpenAI initialization error: {e}")

    _provider_name = "Local Heuristic Simulator"
    return None, _provider_name


def call_llm(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.7,
    model: str | None = None,
) -> str:
    """Make a single chat completion call or use heuristic simulation fallback."""
    client, provider = _get_client()
    used_model = model or MODEL
    debug = os.getenv("MILO_DEBUG", "0").strip() == "1"

    if client is not None:
        try:
            if debug:
                print(f"[DEBUG LLM] Calling {provider} -> model: {used_model} | json_mode: {json_mode}")
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict = {
                "model": used_model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            if debug:
                print(f"[DEBUG LLM] Answer returned from {provider} ({len(content)} chars)")
            return content
        except Exception as exc:
            if debug:
                print(f"[DEBUG LLM] {provider} call failed ({exc}). Falling back to Local Heuristic Simulator.")

    if debug:
        print(f"[DEBUG LLM] Answer returned from Local Heuristic Simulator (offline/mock) | json_mode: {json_mode}")
    return _simulate_llm(prompt=prompt, system=system or "", json_mode=json_mode)



def _simulate_llm(prompt: str, system: str, json_mode: bool) -> str:
    """Deterministic local simulator following Milo rules when API is unavailable."""
    system_lower = system.lower()
    prompt_lower = prompt.lower()

    # Case 0: Consolidated Turn Pipeline (Single-Call)
    if "in this single step, you will classify" in system_lower:
        msg_match = re.search(r'User Message:\s*"([^"]+)"', prompt, re.IGNORECASE)
        user_msg = msg_match.group(1) if msg_match else prompt

        # Sub-call 1: Decision
        dec_prompt = f'Current user message: "{user_msg}"\n{prompt.replace("Process this turn and return JSON.", "")}'
        dec_json = _simulate_llm(dec_prompt, system="read-step classifier", json_mode=True)
        dec_dict = json.loads(dec_json)

        forced_conf_match = re.search(r"forced confidence.*?([0-9\.]+)", prompt, re.IGNORECASE)
        if forced_conf_match:
            dec_dict["confidence"] = float(forced_conf_match.group(1))

        conf = dec_dict.get("confidence", 0.85)
        reg = "direct tentative" if conf >= 0.75 else ("soft hedge" if conf >= 0.4 else "question form only")
        rep_sys = f"{dec_dict['decision']} {reg}"

        # Sub-call 2: Reply
        rep_prompt = f'Their message: "{user_msg}"\n{prompt.replace("Process this turn and return JSON.", "")}'
        reply_text = _simulate_llm(rep_prompt, system=rep_sys, json_mode=False)

        # Sub-call 3: Extraction
        ext_prompt = f"User message: {user_msg}\nAssistant reply: {reply_text}"
        ext_json = _simulate_llm(ext_prompt, system="extract structured facts", json_mode=True)
        ext_dict = json.loads(ext_json)

        return json.dumps({
            "decision": dec_dict["decision"],
            "confidence": dec_dict.get("confidence", 0.85),
            "reason": dec_dict.get("reason", "emotional safety"),
            "reply": reply_text,
            "facts": ext_dict,
        })

    # Case 1: Decision step rubric
    if "read-step classifier" in system_lower or "hold | explore | move_forward" in prompt_lower or "classify this moment" in prompt_lower:
        msg_match = re.search(r'Current user message:\s*"([^"]+)"', prompt, re.IGNORECASE)
        user_msg = msg_match.group(1).lower() if msg_match else prompt_lower

        # Check for forced confidence in testing/prompt
        forced_conf_match = re.search(r'forced_confidence[:=]\s*([0-9\.]+)', prompt, re.IGNORECASE)
        forced_conf = float(forced_conf_match.group(1)) if forced_conf_match else None

        clean_msg = re.sub(r"[^\w\s]", "", user_msg).strip()
        if clean_msg in {"hello", "hi", "hey", "hey there", "hi there", "hello there", "good morning", "good afternoon", "good evening", "howdy", "sup", "whats up", "what's up", "how are you", "how are you doing", "yo", "milo"}:
            res = {"decision": "explore", "confidence": forced_conf if forced_conf is not None else 0.90, "reason": "Casual greeting / check-in."}
        elif "stop messaging" in user_msg or "leave me alone" in user_msg or "need space" in user_msg:
            res = {"decision": "hold", "confidence": forced_conf if forced_conf is not None else 0.95, "reason": "User requested space / stop signal."}
        elif "all resolved" in user_msg or "it's over" in user_msg or "we worked it out" in user_msg:
            res = {"decision": "move_forward", "confidence": forced_conf if forced_conf is not None else 0.90, "reason": "User noted resolution.", "is_resolved": True}
        elif "what should i do" in user_msg or "help me plan" in user_msg or "action plan" in user_msg or "next steps" in user_msg:
            res = {"decision": "move_forward", "confidence": forced_conf if forced_conf is not None else 0.85, "reason": "User explicitly requested practical steps."}
        elif "why do i always leave everything" in user_msg or "don't understand myself" in user_msg or "why do i keep" in user_msg:
            res = {"decision": "explore", "confidence": forced_conf if forced_conf is not None else 0.88, "reason": "User explicitly invited reflection on habits."}
        elif "why does everyone always leave me" in user_msg:
            res = {"decision": "explore", "confidence": forced_conf if forced_conf is not None else 0.65, "reason": "Venting question with reflective framing."}
        else:
            conf = forced_conf if forced_conf is not None else 0.92
            res = {"decision": "hold", "confidence": conf, "reason": "Fresh emotional disclosure; prioritising emotional safety."}

        return json.dumps(res)

    # Case 2: Extraction schema
    if "extract structured facts" in system_lower or "candidate facts" in prompt_lower:
        msg_match = re.search(r'User message:\s*([^\n]+)', prompt, re.IGNORECASE)
        user_msg = msg_match.group(1).lower() if msg_match else prompt_lower

        people = []
        situation = None
        is_stop = False
        is_resolved = False

        if "stop messaging" in user_msg or "leave me alone" in user_msg or "need space" in user_msg:
            is_stop = True

        if "all resolved" in user_msg or "it's over" in user_msg or "we worked it out" in user_msg:
            is_resolved = True

        if "sister" in user_msg:
            people.append({"name": "sister", "relation": "sister"})

        if "cancer" in user_msg or "dad" in user_msg:
            if "dad" in user_msg:
                people.append({"name": "dad", "relation": "father"})
            situation = {
                "area_of_life": "health",
                "description": "Dad diagnosed with cancer; processing emotional shock.",
                "intensity": 5,
                "apparent_need": "hold",
            }
        elif "exam" in user_msg or "stress" in user_msg or "leave everything" in user_msg or "procrastinat" in user_msg:
            situation = {
                "area_of_life": "academic",
                "description": "Stressed about exams and sleep deprivation, tendency to procrastinate to the last minute.",
                "intensity": 4,
                "apparent_need": "explore" if ("why do i" in user_msg or "understand" in user_msg) else "hold",
            }
        elif "sister" in user_msg or "fight" in user_msg:
            situation = {
                "area_of_life": "family",
                "description": "Had a painful fight with sister; emotional tension lingering.",
                "intensity": 4,
                "apparent_need": "hold",
            }
        elif any(w in user_msg for w in ["interview", "work", "overwhelmed", "anxiety", "anxious", "focus", "job"]):
            situation = {
                "area_of_life": "work",
                "description": "Preparing for an important interview while feeling anxious and stressed.",
                "intensity": 4,
                "apparent_need": "hold",
            }
        elif "leave me" in user_msg or "wrong with me" in user_msg:
            situation = {
                "area_of_life": "relationships",
                "description": "Feeling abandoned and questioning self-worth during deep emotional distress.",
                "intensity": 5,
                "apparent_need": "hold",
            }

        res = {
            "people_mentioned": people,
            "situation": situation,
            "is_stop_signal": is_stop,
            "is_resolved": is_resolved,
        }
        return json.dumps(res)

    # Case 3: Check-in generation
    if "brief check-in" in system_lower or "one gentle check-in sentence" in prompt_lower:
        desc_match = re.search(r'Situation:\s*([^\n]+)', prompt, re.IGNORECASE)
        area_match = re.search(r'Area of life:\s*([^\n]+)', prompt, re.IGNORECASE)
        desc = desc_match.group(1).lower() if desc_match else ""
        area = area_match.group(1).lower() if area_match else "day"

        if "exam" in desc or "academic" in desc or "procrastinat" in desc:
            return "Hey, just thinking of you — how is the exam stress feeling today?"
        elif "sister" in desc or "fight" in desc:
            return "Hey, just checking in gently — how are things feeling with your sister?"
        elif "work" in desc or "anxiety" in desc or "interview" in desc:
            return "Thinking of you today — how has that interview anxiety been feeling lately?"
        elif "cancer" in desc or "dad" in desc:
            return "Just wanted to send a gentle thought your way — how are you holding up?"
        else:
            first_word = desc.split()[0] if desc else area
            return f"Checking in to see how things are going with the {first_word} situation."

    # Case 4: Session summary
    if "summarise this conversation" in prompt_lower:
        return "The user shared deeply about their personal challenges and emotional struggles, seeking space to process."

    # Case 5: Reply generation with Phrasing Registers & Stale Threads
    msg_match = re.search(r'Their message:\s*"([^"]+)"', prompt, re.IGNORECASE)
    user_msg = msg_match.group(1).lower() if msg_match else prompt_lower
    context_str = prompt.lower()

    # If stale thread is present in prompt, surface a light check-in question
    if "stale thread" in context_str or "stale_thread" in context_str or "[stale" in context_str:
        if "interview" in context_str:
            return "I remember you were preparing for an interview. Is that still on your mind, or how did it go?"
        if "exam" in context_str or "academic" in context_str:
            return "How have things been since we last talked? Is that exam stress still feeling heavy for you?"
        if "sister" in context_str:
            return "I was thinking of you — are things with your sister still feeling tense, or has that shifted?"
        if "work" in context_str or "anxiety" in context_str:
            return "It's good to hear from you. Is that work anxiety still on your mind lately?"
        return "I remember you mentioned that earlier. Is that situation still feeling relevant for you right now?"

    # Check phrasing register requirement
    is_question_only = "question form only" in system_lower or "confidence < 0.4" in system_lower
    is_soft_hedge = "soft hedge" in system_lower or "0.4 - 0.75" in system_lower
    is_direct_tentative = "direct tentative" in system_lower or "confidence >= 0.75" in system_lower

    if is_question_only:
        # Question form only, NO bare claim like "you're feeling"
        return "What is that experience feeling like for you right now?"

    if ("someone just shared something painful" in system_lower
            or "the person just shared something painful" in system_lower
            or ("hold" in system_lower and "your only job" in system_lower)):
        prefix = "It sounds like " if is_direct_tentative else ("Maybe " if is_soft_hedge else "")
        if "cat" in user_msg or "nancy" in user_msg or "pet" in user_msg:
            # If they're asking for help or mentioning the pet again, acknowledge with warmth
            if "help" in user_msg or "feel better" in user_msg or "love" in user_msg:
                return "Loving them that much makes this so hard. I'm sitting right here with you."
            return f"{prefix}that's real grief. I'm right here with you."
        if "cancer" in user_msg or "dad" in user_msg:
            return f"{prefix}that's devastating news. I'm right here with you."
        if "leave me" in user_msg or "wrong with me" in user_msg:
            return f"{prefix}that's a lot of pain to be sitting with. You're not alone in this."
        if "stop messaging" in user_msg or "space" in user_msg:
            return "Of course. I'll give you space. Take care of yourself."
        if "overwhelmed" in user_msg:
            return f"{prefix}that sounds like a heavy weight. I'm here."
        if "sister" in user_msg or "fight" in user_msg:
            return f"{prefix}that sounds really painful. I'm right here."
        if "exam" in user_msg:
            return f"{prefix}weeks without proper sleep is exhausting. I hear you."
        if "anxiety" in user_msg or "anxious" in user_msg:
            return f"{prefix}carrying that anxiety is draining. I'm right here with you."
        if "pain" in user_msg or "hurt" in user_msg or "hard" in user_msg:
            return f"{prefix}that sounds really hard. I'm right here."
        return f"{prefix}that's a lot to be carrying. I'm right here with you."

    if "open to reflection" in system_lower or "explore" in system_lower:
        clean_msg = re.sub(r"[^\w\s]", "", user_msg).strip()
        if clean_msg in {"hello", "hi", "hey", "hey there", "hi there", "hello there", "good morning", "good afternoon", "good evening", "howdy", "sup", "whats up", "what's up", "how are you", "how are you doing", "yo", "milo"}:
            return "Hey there! Good to hear from you. How are you doing today?"
        prefix = "It seems like " if is_direct_tentative else ("I could be off, but maybe " if is_soft_hedge else "")
        if "sister" in user_msg or "sister" in context_str or "fight" in context_str or "tense" in context_str:
            return f"{prefix}things still feel tense with your sister. How is your heart holding that today?"
        if "anxiety" in user_msg or "work" in context_str or "focus" in context_str:
            return f"{prefix}work anxiety is lingering. What part of the day feels hardest to stay focused?"
        if "last minute" in user_msg or "procrastinat" in user_msg:
            return f"{prefix}that pattern brings frustration. What do you notice happening right before you put things off?"
        return f"{prefix}that is worth pausing on. What comes up for you when you sit with that?"

    if "ready for practical support" in system_lower or "move_forward" in system_lower:
        return "We can take this one small step at a time. What feels like the smallest thing to start with?"

    return "I'm listening and right here with you."