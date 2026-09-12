# Milo — Architectural Design Note
Time Taken: 6 hours

## 1. Executive Summary & Core Philosophy

**Milo** is a supportive listening companion engineered around emotional safety, auditable classification, and structural constraints. Unlike typical conversational assistants that default to rapid problem-solving, unprompted advice, or probing interrogations, Milo prioritises:
1. **Holding space** for painful disclosures without jumping to fixes.
2. **Auditable separation** between classification (`decide.py`) and text generation (`reply.py`).
3. **Strict memory grounding** from structured relational state rather than raw transcripts, avoiding hallucinated recollections or file recitation.
4. **Non-intrusive proactive care** with strict guardrails against badgering or ignoring boundaries.

---

## 2. Decoupled Pipeline: Read-Step vs. Reply-Step

### 2.1 The Problem with Single-Step LLM Responses
In conventional LLM architectures, classification, reasoning, and response generation happen in a single generation step. This causes two major failure modes in supportive contexts:
- **Tone bleeding**: If the model determines that a problem exists, its generation weights immediately drift toward advice, solutions, or toxic positivity ("Have you tried meditation?").
- **Unauditable decisions**: If the model behaves inappropriately, it is impossible to cleanly decouple whether the failure was a misclassification of the user's emotional state or a failure of conversational phrasing.

### 2.2 The Two-Call Architecture
Milo enforces a strict two-stage pipeline per turn:
```
User Message
    │
    ▼
[memory.load_context(person_id)]  ──> Structured rows (situations, signals, summary)
    │
    ▼
[decide.py]                       ──> {decision: hold|explore|move_forward, confidence, reason}
    │                                  └─> Logged to SQLite `decisions` table ALWAYS
    ▼
[reply.py]                        ──> Tailored system prompt + structural post-validator
    │                                  └─> Loop: validate (advice regex, len, '?') -> regenerate
    ▼
[memory.extract]                  ──> Extract structured facts (entities, situation, intensity)
    │
    ▼
[memory.write]                    ──> Upsert SQLite rows (people, situations, signals, pause flag)
```

**Key Benefit**: The decision is logged to the `decisions` table *before* reply generation or validation occurs. Even if the reply generator fails, crashes, or is retried, the auditor can verify how the model classified the moment.

---

## 3. Decision Rubric: Hold, Explore, Move Forward

| Mode | Heuristics | System Prompt Guidance | Constraints |
| :--- | :--- | :--- | :--- |
| **HOLD** | Fresh disclosure, high emotional intensity (1-5 scale >= 4), grief, trauma, shock, venting. Default when uncertain. | Deep presence, empathy, zero advice, zero coaching. | No advice language, max 1 gentle check-in question, < 280 chars. |
| **EXPLORE** | User explicitly invites reflection ("Why do I keep doing this?", "I don't understand myself"), thinking out loud. | Gentle curiosity, helping person examine patterns without diagnosing. | Max 1 open question, no advice, < 280 chars. |
| **MOVE_FORWARD**| Explicit request for concrete help ("What should I do?", "Help me plan") or explicitly resolved situation. | Soft, low-pressure framing ("One small thing that might help..."). | Constructive suggestions, max 1 question, < 280 chars. |

---

## 4. Reply Constraints: Structural Enforcement vs. Prompting

Relying on system prompts alone to prevent advice-giving during emotional distress is unreliable. Milo implements a multi-tier defense:

1. **Prompt Constraint**: System prompts explicitly forbid directive phrases ("you should", "try", "have you considered", "why not", "I suggest").
2. **Post-Generation Validator**:
   - **Advice Filter**: Regex scanning against banned advice phrases (`ADVICE_PATTERNS`).
   - **Question Cap**: Rejection if string contains more than one `?`.
   - **Length Cap**: Strict limit of 280 characters.
3. **Regeneration Loop**:
   If validation fails, the validator feeds the rejection reason back into a targeted mutation prompt and regenerates (up to 3 attempts). If all attempts fail, a hardcoded safe fallback response is returned.

---

## 5. Memory Schema & Transcript Isolation Rule

### 5.1 SQLite Relational Schema
- `people(person_id, name, relation, first_mentioned, last_mentioned, paused)`
- `situations(id, person_id, area_of_life, description, intensity, status, created_at, updated_at)`
- `signals(id, person_id, situation_id, need, created_at)`
- `sessions(id, person_id, started_at, summary)`
- `checkins(id, person_id, situation_id, sent_at, message, replied, stopped)`
- `decisions(id, timestamp, person_id, input_hash, decision, confidence, reason, model_version)`

### 5.2 The Transcript Isolation Rule
> **Rule**: Reply generation and check-in generation are supplied *only* structured relational fields (situation description, area of life, intensity, session summary), never the raw verbatim chat history.

**Why?**
1. **Prevents "reciting the file"**: Assistants with full transcript windows often quote past user statements unnaturally or regurgitate earlier phrasing.
2. **Prevents false claims**: LLMs frequently hallucinate facts from unsaid conversational nuances. By passing only verified, extracted facts, the context boundary is crisp.
3. **Data minimization**: Old messages can be pruned or archived without breaking continuity.

---

## 6. Proactive Check-Ins & Safety Guards

The `checkin.py` module supports clock simulation (`--now "+2d"`) and autonomous background scanning.

### 6.1 Guardrails
1. **Keyword Citation Check**: The check-in generator must cite at least one concrete keyword from the stored situation description. If it fails, a templated fallback is used.
2. **Single-Nudge Guard**: If `checkins` has an existing entry for that situation where `replied == 0` and `stopped == 0`, no second check-in is sent.
3. **The Stop / Pause Guard**: If the user sends a boundary or stop signal ("stop messaging me", "leave me alone", "I need space"), `memory.py` immediately flags `people.paused = 1`. All proactive check-ins are completely halted until the user affirmatively initiates conversation again.

---

## 7. Known Limitations & Scenario 6 Discussion

### 7.1 The "Venting Disguised as a Question" Dilemma
In `scenarios/s6_known_failure.yaml`, the user asks:
> *"Why does everyone always leave me? What is wrong with me?"*

**The Challenge**:
Semantically and syntactically, this turn looks like an exploratory question ("Why does...?", "What is...?"). However, pragmatically and emotionally, this is acute, high-intensity distress (HOLD).

Current generation LLMs frequently misclassify this turn as `explore` (or in worst cases `move_forward`) because attention weights strongly attend to interrogative words ("Why", "What") rather than emotional desperation.

3. In future iterations, fine-tuned distress classifiers or acoustic/sentiment thresholds can further safeguard against this nuance.

---

## 8. Groq API & Open Models (`openai/gpt-oss-120b`)

Milo natively supports Groq's high-throughput LPU inference via OpenAI-compatible endpoints:
- **Base URL**: `https://api.groq.com/openai/v1`
- **Default Model**: `openai/gpt-oss-120b` (an open-weight 120B MoE model offering high-reasoning agentic capabilities).
- **Fallback**: Automatically falls back to internal heuristic simulation if no key is configured, allowing offline deterministic test execution.

---

## 9. Add-on 1: Latency as Affect

Rather than responding at uniform machine speed, Milo modulates output pacing to reflect emotional presence:
- **Delay Formula**:
  - `hold + intensity >= 4` -> 3.5–5.0s (deliberate pause holding space)
  - `hold + intensity 2-3` -> 2.0–3.5s
  - `explore` -> 1.5–2.5s (thoughtful reflection)
  - `move_forward` -> 0.3–1.0s (energetic, practical)
- **Jitter**: `delay *= random.uniform(0.85, 1.15)`
- **Execution**: In interactive CLI (`cli.py`), introduces conversational pacing before printing replies. In the headless runner (`runner.py`), delays are logged and asserted without stalling test execution.
- **Auditing**: Logged as `latency_computed` in the `trace` table.

---

## 10. Add-on 3: Selective Forgetting

Human memory does not maintain all past disclosures as perpetually active facts:
- **Stale Threshold (`STALE_DAYS = 5`)**: If an open situation has not been touched in over 5 simulated days, its status flips to `'stale'`.
- **Asking Over Assuming**: Stale situations are stripped from active declarative context and placed in `stale_threads: [{area, description}]`.
- **System Prompt Constraint**: Prompt forces the model to ask a light check-in question rather than asserting that the situation is still ongoing.
- **Resolution**: Explicit user resolution flags the situation as `'resolved'`, excluding it from active context while retaining it for audit.
- **Auditing**: Logged as `stale_flip` in the `trace` table.

---

## 11. Add-on 4: Uncertainty-Calibrated Language

Milo matches phrasing firmness directly to classifier confidence:
- **Classification Output**: `decide.py` outputs `confidence: float (0.0 to 1.0)`.
- **Phrasing Registers**:
  - **$\ge 0.75$ (Direct Tentative)**: *"Sounds like...", "Seems like..."*
  - **$0.4 - 0.75$ (Soft Hedge)**: *"Maybe...", "I could be off, but..."*
  - **$< 0.4$ (Question Form Only)**: Hard constraint requiring question form only; no bare emotional claims permitted.
- **Structural Validator**: If confidence $< 0.4$, regex checks reject bare emotional claims (`"you are feeling..."`, `"you must be..."`) lacking explicit hedge words.
- **Auditing**: Logged as `confidence_band` in the `trace` table.

---

## 12. Add-on 6: Check-In Variance

Periodic check-ins must not feel like automated cron pings:
- **Base Gaps by Intensity**:
  - Intensity $\ge 4$: 24 hours
  - Intensity $= 3$: 48 hours
  - Intensity $\le 2$: 72 hours
- **Temporal Jitter**: `jitter = scheduled_gap * random.uniform(-0.3, 0.3)`
- **Check-In Propensity**: Stored in `people.checkin_propensity` (default 0.6). A probabilistic roll skips cycles if `random.random() > propensity`, rescheduling for subsequent cycles.
- **Auditing**: Every evaluation is logged as `checkin_evaluated` in the `trace` table.