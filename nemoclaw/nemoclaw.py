"""
NemoClaw — Orchestrator agent for SparX.

Sits between the Direction Engine and the user-facing response.
For categorized/uncategorized complaints, NemoClaw:
  1. Tries to answer from NYC Open Data (data_lookup module)
  2. If Open Data can't help, falls back to FormFinder for a 311 complaint form

Usage:
    from nemoclaw import NemoClaw

    claw = NemoClaw()
    result = await claw.handle(direction_result_dict)
    # result is a NemoClawResult with .source, .form_finder / .open_data, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Allow imports from sibling packages
_SPARX_ROOT = Path(__file__).resolve().parent.parent
if str(_SPARX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARX_ROOT))

from form_finder.form_finder import FormFinder, FormFinderError
import data_lookup

LOGGER = logging.getLogger("nemoclaw")

LLM_CHAT_URL = os.environ.get(
    "SPARX_LLM_CHAT_URL", "http://localhost:8081/v1/chat/completions"
)

_SYNTHESIZE_SYSTEM = """\
You are a NYC civic assistant. You have been given REAL DATA ROWS from \
NYC Open Data datasets that are relevant to a resident's question.

Your job: write a clear, helpful, conversational answer that uses SPECIFIC \
facts from the data rows. Reference actual numbers, dates, locations, names, \
or statuses from the rows when relevant. Keep it concise (3-5 sentences).

If the rows don't contain enough info to fully answer the question, say what \
you CAN tell from the data and mention which dataset(s) the user can explore \
for more (include the dataset title).

Do NOT make up data. Only cite what appears in the rows below.
"""

_SYNTHESIZE_USER = """\
USER'S QUESTION: {question}

DATASETS FOUND:
{datasets}

DATA ROWS (from those datasets):
{rows_json}

Write a helpful answer grounded in the data above.
"""


@dataclass
class NemoClawResult:
    """Final result handed back to the server / UI layer."""

    source: str                       # "open_data" | "form_finder" | "none"
    decision: str                     # original routing decision
    categories: list[str]             # original categories
    transcript: str

    # Populated when source == "open_data"
    open_data: dict | None = None     # {answer, source, records, rows}

    # Populated when source == "form_finder"
    form_finder: dict | None = None   # {intent, form_name, form_url, summary, next_steps}

    # Plain-language response to speak back to the user
    response: str = ""


@dataclass
class NemoClaw:
    """Orchestrator that routes categorized complaints through Open Data
    then FormFinder."""

    _finder: FormFinder = field(default=None, repr=False, init=False)
    _http: httpx.AsyncClient = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._finder = FormFinder()
        self._http = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )

    async def _synthesize_from_rows(
        self, question: str, od_result: dict
    ) -> str | None:
        """Use the LLM to produce a tailored answer from actual Open Data rows."""
        rows = od_result.get("rows") or []
        records = od_result.get("records") or []
        if not rows:
            return None

        # Build dataset summary for the prompt
        datasets_text = "\n".join(
            f"- {r['title']}: {r.get('description', '')[:200]} ({r['url']})"
            for r in records[:4]
        )

        # Trim rows to avoid blowing up context — keep first 10 rows, truncate
        rows_for_prompt = rows[:10]
        rows_json = json.dumps(rows_for_prompt, indent=1, default=str)
        # Cap total size to ~6k chars
        if len(rows_json) > 6000:
            rows_json = rows_json[:6000] + "\n... (truncated)"

        user_msg = _SYNTHESIZE_USER.format(
            question=question,
            datasets=datasets_text,
            rows_json=rows_json,
        )

        payload: dict = {
            "messages": [
                {"role": "system", "content": _SYNTHESIZE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }
        model = os.environ.get("SPARX_LLM_MODEL")
        if model:
            payload["model"] = model

        try:
            resp = await self._http.post(LLM_CHAT_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if content:
                LOGGER.info("LLM synthesized answer from %d rows", len(rows_for_prompt))
                return content
        except Exception as exc:
            LOGGER.error("LLM synthesis failed: %s", exc)

        return None

    async def handle(self, direction: dict) -> NemoClawResult:
        """Process a DirectionResult dict from the direction engine.

        Expected keys in `direction`:
            decision, categories, question, summary, response,
            confidence, missing_info, transcript (optional)

        Only runs the lookup pipeline for categorized / uncategorized.
        For needs_more_info / irrelevant, passes through as-is.
        """
        decision   = direction.get("decision", "irrelevant")
        categories = direction.get("categories", [])
        question   = direction.get("question", "")
        summary    = direction.get("summary", "")
        transcript = direction.get("transcript", question)
        response   = direction.get("response", "")

        if decision not in ("categorized", "uncategorized"):
            return NemoClawResult(
                source="none",
                decision=decision,
                categories=categories,
                transcript=transcript,
                response=response,
            )

        query_text = question or summary or transcript

        # Run both lookups in parallel — Open Data for context, FormFinder for 311 forms
        LOGGER.info("NemoClaw lookup for: %s", query_text[:80])

        od_result = None
        ff_result = None

        async def _open_data():
            return await asyncio.to_thread(data_lookup.search, query_text, categories)

        async def _form_finder():
            return await asyncio.to_thread(self._finder.classify, query_text)

        od_task = asyncio.create_task(_open_data())
        ff_task = asyncio.create_task(_form_finder())

        try:
            od_result = await od_task
        except Exception as exc:
            LOGGER.error("Open Data lookup failed: %s", exc)

        try:
            ff_result = await ff_task
        except Exception as exc:
            LOGGER.error("FormFinder failed: %s", exc)

        # If Open Data returned actual rows, synthesize a tailored answer via LLM
        synthesized = None
        if od_result and od_result.get("rows"):
            try:
                synthesized = await self._synthesize_from_rows(query_text, od_result)
            except Exception as exc:
                LOGGER.error("Synthesis step failed: %s", exc)

        # Build response from whichever sources returned data
        if ff_result and od_result:
            source = "form_finder+open_data"
            od_answer = synthesized or od_result.get("answer", "")
            resp = (
                f"I found a matching NYC 311 form: {ff_result['form_name']}. "
                f"{ff_result['summary']} "
                f"Additionally, from NYC Open Data: {od_answer}"
            )
        elif ff_result:
            source = "form_finder"
            resp = (
                f"I found a matching NYC 311 form: {ff_result['form_name']}. "
                f"{ff_result['summary']}"
            )
        elif od_result:
            source = "open_data"
            resp = synthesized or od_result.get("answer", response)
        else:
            return NemoClawResult(
                source="none",
                decision=decision,
                categories=categories,
                transcript=transcript,
                response=(
                    "I understand your concern but I'm having trouble looking up "
                    "the right form right now. Please try calling 311 directly."
                ),
            )

        return NemoClawResult(
            source=source,
            decision=decision,
            categories=categories,
            transcript=transcript,
            open_data=od_result,
            form_finder=ff_result,
            response=resp,
        )

    def handle_sync(self, direction: dict) -> NemoClawResult:
        """Synchronous wrapper for callers that aren't in an async context."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.handle(direction)).result()
        return asyncio.run(self.handle(direction))
