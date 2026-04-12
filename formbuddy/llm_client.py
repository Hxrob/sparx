import json
import time

import httpx

from config import LLM_BASE_URL, LLM_MODEL, LLM_OPTIONS
from models import LLMDebugInfo, SuggestResponse


def _strip_fences(content: str) -> str:
    """Strip markdown ```json ... ``` fences if present."""
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
        content = content.strip()
    return content


def _extract_usage(data: dict) -> dict[str, int]:
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


async def chat_completion(
    messages: list[dict],
) -> tuple[SuggestResponse, LLMDebugInfo]:
    debug = LLMDebugInfo(request_messages=[m.copy() for m in messages])

    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": messages,
                **LLM_OPTIONS,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()
    debug.raw_response = content
    tokens = _extract_usage(data)
    debug.prompt_tokens = tokens["prompt_tokens"]
    debug.completion_tokens = tokens["completion_tokens"]
    debug.total_tokens = tokens["total_tokens"]

    content = _strip_fences(content)

    try:
        parsed = json.loads(content)
        debug.duration_ms = (time.monotonic() - t0) * 1000
        return SuggestResponse(**parsed), debug
    except (json.JSONDecodeError, Exception):
        debug.parsed_ok = False
        debug.retried = True

        # One retry asking for valid JSON
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": "That was not valid JSON. Return ONLY valid JSON matching the schema. No extra text.",
            }
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp2 = await client.post(
                f"{LLM_BASE_URL}/v1/chat/completions",
                json={"model": LLM_MODEL, "messages": messages, **LLM_OPTIONS},
            )
        resp2.raise_for_status()
        data2 = resp2.json()
        content2 = data2["choices"][0]["message"]["content"].strip()
        debug.retry_raw_response = content2

        retry_tokens = _extract_usage(data2)
        debug.prompt_tokens += retry_tokens["prompt_tokens"]
        debug.completion_tokens += retry_tokens["completion_tokens"]
        debug.total_tokens += retry_tokens["total_tokens"]

        content2 = _strip_fences(content2)
        parsed2 = json.loads(content2)
        debug.duration_ms = (time.monotonic() - t0) * 1000
        return SuggestResponse(**parsed2), debug
