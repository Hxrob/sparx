"""
Direction Engine.

Takes a transcript string from Parakeet, runs it through the same OpenAI-compatible
chat LLM as FormFinder (e.g. llama.cpp on port 8081), and returns a DirectionResult —
categorized, uncategorized, needs_more_info, or irrelevant.

If categorized or uncategorized, appends one line to complaints.log.
Does not call OpenClaw — that is someone else's responsibility.

Architecture
------------
No separate inference server required beyond the LLM HTTP API. This module is
imported directly by server.py and called after Parakeet returns a transcription.
One chat completion call, one result.

Session context
---------------
Pass an optional session_id to process(). The engine accumulates all clips
for that session and sends the full history to the LLM, so each follow-up
clip is understood in context. Sessions expire after SESSION_TTL seconds
of inactivity.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# SparX root (voice_node -> nodes -> nemoclaw -> sparx) for form_finder imports
_SPARX_ROOT = Path(__file__).resolve().parents[3]
if str(_SPARX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARX_ROOT))

import httpx

from form_finder.form_finder import _extract_json
from schemas import DirectionResult, NYCBenefitCategory, RoutingDecision


LOGGER = logging.getLogger("direction_engine")

# Same defaults as FormFinder — one llama-server (or vLLM, etc.) for everything
LLM_CHAT_URL = os.environ.get(
    "SPARX_LLM_CHAT_URL", "http://localhost:8081/v1/chat/completions"
)
LOG_PATH = os.path.join(os.path.dirname(__file__), "complaints.log")

SESSION_TTL = 300  # seconds until an idle session is expired

_CATEGORIES = "\n  ".join(f'"{c.value}"' for c in NYCBenefitCategory)
_DECISIONS = ", ".join(d.value for d in RoutingDecision)

_SYSTEM = f"""\
You are a NYC social services intake classifier.

Given a transcript (which may be a single clip or multiple clips from one session),
you must:
1. Identify the core question or complaint.
2. Decide if it falls into one of the NYC Benefits and Programs categories,
   is a genuine complaint/question that doesn't fit those categories,
   needs more information, or is completely irrelevant.
3. Write a 1-2 sentence summary suitable for a case worker.

Routing decisions: {_DECISIONS}
  - categorized:     one or more clear category matches — send to OpenClaw
  - uncategorized:   genuine complaint or question but NO matching category — still send to OpenClaw
  - needs_more_info: unclear intent, specify what is missing
  - irrelevant:      small talk, greetings, test phrases, off-topic — do NOT send to OpenClaw

NYC Benefits and Programs categories (use EXACT strings, only for "categorized"):
  {_CATEGORIES}

RULES:
- A single transcript can match MULTIPLE categories — list all that apply.
- Only populate "categories" when decision is "categorized".
- "uncategorized" means you are confident it is a real complaint but it does not fit
  any of the 19 categories above. Do NOT use "uncategorized" just because you are unsure.
- missing_info must be non-empty when decision is "needs_more_info".
- Be honest with confidence. Uncertainty is better than a wrong category.
- "response" is always required — it is what gets spoken back to the user:
    - categorized/uncategorized: confirm what you heard and what help is being looked up
    - needs_more_info: ask a natural follow-up question for the missing info
    - irrelevant:      respond naturally as a helpful assistant would
- Output ONLY one valid JSON object. The first character of your reply MUST be {{
  and the last MUST be }}. No markdown fences, no prose before or after the object.

Output format (copy this shape; use valid JSON with double-quoted keys):
{{
  "decision": "<value>",
  "categories": ["<exact category>", "<exact category>"],
  "question": "<what the person is asking>",
  "summary": "<1-2 sentences for OpenClaw>",
  "response": "<what to say back to the user>",
  "confidence": 0.0,
  "missing_info": []
}}
"""

_USER = "TRANSCRIPT:\n{transcript}\n\nReply with the JSON object only."


def _strip_reasoning_blocks(text: str) -> str:
    """Remove common chain-of-thought wrappers so the JSON parser can see output."""
    t = text
    # Hex escapes avoid typos in long XML-like tag names.
    think_o, think_c = "\x3c\x74\x68\x69\x6e\x6b\x3e", "\x3c\x2f\x74\x68\x69\x6e\x6b\x3e"
    patterns = (
        r"``",
        r"<reasoning>.*?</reasoning>",
        re.escape(think_o) + r".*?" + re.escape(think_c),
    )
    for pat in patterns:
        t = re.sub(pat, "", t, flags=re.DOTALL | re.IGNORECASE)
    return t.strip()


def _decode_first_json_object(text: str) -> dict | None:
    """Parse the first top-level JSON object (handles nested braces inside strings)."""
    s = text.strip()
    i = s.find("{")
    if i < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(s[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_model_json(raw: str) -> dict | None:
    """Try FormFinder fenced/greedy extract, then reasoning-strip + raw_decode."""
    cleaned = _strip_reasoning_blocks(raw)
    try:
        out = _extract_json(cleaned)
        if isinstance(out, dict):
            return out
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        out = _extract_json(raw)
        if isinstance(out, dict):
            return out
    except (ValueError, json.JSONDecodeError):
        pass
    return _decode_first_json_object(cleaned) or _decode_first_json_object(raw)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Persistent httpx client — reused across calls, avoids TCP handshake overhead
_client: httpx.AsyncClient | None = None

# In-memory session store: session_id -> {clips: list[str], last_seen: float}
_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _accumulate(session_id: str | None, new_clip: str) -> str:
    """
    Append new_clip to the session's history and return the full transcript
    formatted for the LLM. Expires idle sessions on every call.
    If session_id is None, returns just the new clip (stateless).
    """
    now = time.monotonic()

    # Expire stale sessions
    expired = [sid for sid, s in _sessions.items() if now - s["last_seen"] > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
        LOGGER.debug("Session %s expired", sid)

    if not session_id:
        return new_clip

    if session_id not in _sessions:
        _sessions[session_id] = {"clips": [], "last_seen": now}

    session = _sessions[session_id]
    session["clips"].append(new_clip)
    session["last_seen"] = now

    if len(session["clips"]) == 1:
        return new_clip  # single clip — no labeling needed

    # Format as numbered clips so the model understands sequence
    lines = [f"[clip {i+1}] {clip}" for i, clip in enumerate(session["clips"])]
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    """Remove a session explicitly (e.g., user presses End / navigates away)."""
    _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# OpenAI-compatible chat client (same stack as FormFinder)
# ---------------------------------------------------------------------------


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


async def _call_llm(transcript: str) -> str:
    """POST to /v1/chat/completions (llama.cpp, vLLM, etc.)."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(transcript=transcript)},
    ]
    payload: dict = {
        "messages": messages,
        "temperature": float(os.environ.get("SPARX_DIRECTION_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("SPARX_DIRECTION_MAX_TOKENS", "1024")),
    }
    model = os.environ.get("SPARX_LLM_MODEL")
    if model:
        payload["model"] = model
    # llama.cpp / some servers: forces JSON-only output when supported
    if os.environ.get("SPARX_DIRECTION_JSON_MODE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        payload["response_format"] = {"type": "json_object"}

    client = _get_client()
    resp = await client.post(LLM_CHAT_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM returned no choices: " + repr(data)[:500])

    first = choices[0]
    msg = first.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = None
    if not content:
        content = first.get("text") or first.get("content")

    text = str(content or "").strip()
    if not text:
        raise ValueError("LLM returned empty content: " + repr(data)[:500])
    return text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse(raw: str, transcript: str) -> DirectionResult:
    data = _parse_model_json(raw)
    if data is None:
        m = _JSON_RE.search(_strip_reasoning_blocks(raw))
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                data = None

    if data is None:
        snippet = raw[:1200].replace("\r", " ")
        LOGGER.warning(
            "No JSON found in LLM response (showing start of raw output): %s",
            snippet,
        )
        return DirectionResult(
            decision=RoutingDecision.needs_more_info,
            question="Unknown",
            summary="Could not parse model response.",
            response="I didn't quite catch that. Could you tell me more about what you need help with?",
            confidence=0.0,
            missing_info=["model returned unparseable output"],
            transcript=transcript,
        )

    try:
        decision = RoutingDecision(data.get("decision", "needs_more_info"))
    except ValueError:
        decision = RoutingDecision.needs_more_info

    # Parse categories — only meaningful for "categorized"
    categories: list[NYCBenefitCategory] = []
    if decision == RoutingDecision.categorized:
        raw_cats = data.get("categories") or []
        if isinstance(raw_cats, str):
            raw_cats = [raw_cats]
        for raw_cat in raw_cats:
            try:
                categories.append(NYCBenefitCategory(raw_cat))
            except ValueError:
                LOGGER.warning("Unknown category %r — skipped", raw_cat)
        if not categories:
            # Model said categorized but gave no valid categories — treat as uncategorized
            decision = RoutingDecision.uncategorized

    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

    return DirectionResult(
        decision=decision,
        categories=categories,
        question=str(data.get("question", "")).strip(),
        summary=str(data.get("summary", "")).strip(),
        response=str(data.get("response", "")).strip(),
        confidence=confidence,
        missing_info=list(data.get("missing_info") or []),
        transcript=transcript,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_complaint(result: DirectionResult) -> None:
    """Append one line to complaints.log — called for categorized and uncategorized results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cats = ", ".join(c.value for c in result.categories) if result.categories else "none"
    line = (
        f"[{timestamp}] "
        f"DECISION: {result.decision.value} | "
        f"CATEGORIES: {cats} | "
        f"CONFIDENCE: {result.confidence:.0%} | "
        f"{result.question}\n"
    )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
    LOGGER.info("Logged complaint: %s", line.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def process(transcript: str, session_id: str | None = None) -> DirectionResult:
    """
    Main entry point.
    Takes a transcript string from Parakeet and an optional session_id.
    Accumulates session context if session_id is provided.
    Returns a DirectionResult. Appends to complaints.log if categorized or uncategorized.
    """
    if not transcript or not transcript.strip():
        return DirectionResult(
            decision=RoutingDecision.irrelevant,
            question="",
            summary="Empty transcript.",
            response="I didn't hear anything. Could you try again?",
            confidence=1.0,
            transcript=transcript,
        )

    full_transcript = _accumulate(session_id, transcript.strip())

    try:
        raw = await _call_llm(full_transcript)
    except Exception as exc:
        LOGGER.error("Direction LLM error: %s", exc)
        return DirectionResult(
            decision=RoutingDecision.needs_more_info,
            question="Unknown",
            summary=f"LLM unreachable: {exc}",
            response="I'm having trouble processing that right now. Please try again in a moment.",
            confidence=0.0,
            missing_info=["llm unavailable"],
            transcript=full_transcript,
        )

    result = _parse(raw, full_transcript)

    if result.decision in (RoutingDecision.categorized, RoutingDecision.uncategorized):
        _log_complaint(result)

    LOGGER.info(
        "Direction: %s | categories=%s | confidence=%.0f%%",
        result.decision.value,
        ", ".join(c.value for c in result.categories) if result.categories else "none",
        result.confidence * 100,
    )

    return result
