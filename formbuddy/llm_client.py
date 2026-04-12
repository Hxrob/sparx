import json

import httpx

from config import LLM_BASE_URL, LLM_MODEL, LLM_OPTIONS
from models import SuggestResponse


async def chat_completion(messages: list[dict]) -> SuggestResponse:
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
    # Strip markdown fences if the model wraps in ```json ... ```
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
        content = content.strip()

    try:
        parsed = json.loads(content)
        return SuggestResponse(**parsed)
    except (json.JSONDecodeError, Exception):
        # One retry asking for valid JSON
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": "That was not valid JSON. Return ONLY valid JSON matching the schema. No extra text.",
            }
        )
        resp2 = await httpx.AsyncClient(timeout=120.0).post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json={"model": LLM_MODEL, "messages": messages, **LLM_OPTIONS},
        )
        resp2.raise_for_status()
        content2 = resp2.json()["choices"][0]["message"]["content"].strip()
        if content2.startswith("```"):
            content2 = content2.split("\n", 1)[1]
            if content2.endswith("```"):
                content2 = content2[: content2.rfind("```")]
            content2 = content2.strip()
        parsed2 = json.loads(content2)
        return SuggestResponse(**parsed2)
