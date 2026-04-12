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

_DATASET_PICKER_SYSTEM = """\
You are a NYC Open Data expert. The user asked a question that does NOT \
relate to 311 complaints. You must pick the BEST dataset from the list \
below and write a Socrata SODA $where query to answer their question.

RULES:
- Pick ONE dataset that is most relevant.
- Write a valid SODA $where clause using the dataset's columns.
- Use single quotes for string literals in $where.
- For text matching use: upper(column) LIKE '%VALUE%'
- For dates use: column > '2025-01-01T00:00:00'
- Keep queries simple and targeted.
- If no dataset fits, set dataset_id to null.

Output ONLY a JSON object with these keys:
- "dataset_id": the 4-4 Socrata ID (e.g. "sejx-2gn3") or null
- "title": dataset title you picked
- "where": the $where clause string (can be empty string for no filter)
- "order": optional $order clause (e.g. "date DESC"), empty string if none
- "reason": one sentence why you picked this dataset

The first character of your reply MUST be { and the last MUST be }.
"""

_DATASET_PICKER_USER = """\
USER'S QUESTION: {question}

AVAILABLE DATASETS:
{datasets}

Pick the best dataset and write a $where query. JSON only.
"""

_SYNTHESIZE_SYSTEM = """\
You are a NYC civic assistant. You have been given REAL 311 COMPLAINT DATA \
and service records that are relevant to a resident's question.

Your job: write a clear, helpful, conversational answer that cites SPECIFIC \
facts from the data — complaint types, addresses, dates, statuses, boroughs, \
agencies, and resolution descriptions. Keep it concise (3-5 sentences).

RULES:
- NEVER output URLs or dataset links. The user is on a phone — links are useless.
- NEVER say "check this dataset" or "explore this link".
- DO cite specific addresses, dates, complaint counts, and statuses from the rows.
- If the data shows patterns (e.g. many noise complaints in an area), mention that.
- If data is insufficient, say what you found and suggest calling 311 for more help.
- Follow the response template: What I found / Best next step / If this does not work.

Do NOT make up data. Only cite what appears in the rows below.
"""

_SYNTHESIZE_USER = """\
USER'S QUESTION: {question}

DATA SOURCE: {source}

DATA ROWS:
{rows_json}

Write a helpful answer grounded in the data above. No URLs or links.
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

    async def _call_llm(self, system: str, user: str,
                        temperature: float = 0.2, max_tokens: int = 512) -> str | None:
        """Generic LLM call helper. Returns content string or None."""
        payload: dict = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        model = os.environ.get("SPARX_LLM_MODEL")
        if model:
            payload["model"] = model
        try:
            resp = await self._http.post(LLM_CHAT_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            ) or None
        except Exception as exc:
            LOGGER.error("LLM call failed: %s", exc)
            return None

    async def _discover_and_query(self, question: str) -> dict | None:
        """When 311 doesn't fit, ask the LLM to pick a dataset and build a query.

        1. Search the NYC Open Data catalog for candidate datasets
        2. Send dataset schemas to LLM → it picks one + writes a $where clause
        3. Execute the pandas query and return rows
        """
        # Step 1: discover candidate datasets
        keywords = data_lookup._short_keywords(question)
        if not keywords:
            return None

        candidates = await asyncio.to_thread(data_lookup.discover_datasets, keywords, 5)
        if not candidates:
            LOGGER.info("No catalog datasets found for: %s", keywords)
            return None

        # Step 2: ask LLM to pick the best dataset + generate $where
        datasets_text = ""
        for c in candidates:
            cols = ", ".join(c["columns"][:20]) if c["columns"] else "(columns unknown)"
            datasets_text += (
                f"- {c['dataset_id']} | {c['title']}\n"
                f"  Description: {c['description']}\n"
                f"  Columns: {cols}\n\n"
            )

        user_msg = _DATASET_PICKER_USER.format(
            question=question,
            datasets=datasets_text,
        )

        raw = await self._call_llm(
            _DATASET_PICKER_SYSTEM, user_msg,
            temperature=0.1, max_tokens=300,
        )
        if not raw:
            return None

        # Strip thinking tags and fences
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:raw.rfind("```")]
            raw = raw.strip()

        try:
            picked = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("LLM dataset picker returned invalid JSON: %s", raw[:200])
            return None

        dataset_id = picked.get("dataset_id")
        if not dataset_id:
            LOGGER.info("LLM decided no dataset fits the question")
            return None

        title = picked.get("title", dataset_id)
        where = picked.get("where", "")
        order = picked.get("order", "")
        reason = picked.get("reason", "")
        LOGGER.info("LLM picked dataset %s (%s): %s", dataset_id, title, reason)

        # Step 3: execute the pandas query
        rows = await asyncio.to_thread(
            data_lookup.query_dynamic, dataset_id,
            where=where, order=order, limit=15,
        )

        if not rows:
            # Try without $where as fallback (just full-text)
            rows = await asyncio.to_thread(
                data_lookup.query_dynamic, dataset_id,
                q=keywords, limit=10,
            )

        if not rows:
            LOGGER.info("Dynamic query returned no rows for %s", dataset_id)
            return None

        for row in rows:
            row["_dataset_title"] = title
            row["_dataset_id"] = dataset_id

        return {
            "answer": f"Found {len(rows)} records from {title}.",
            "source": f"{title} ({dataset_id})",
            "records": [{"title": title, "dataset_id": dataset_id, "description": reason}],
            "rows": rows,
        }

    async def _synthesize_from_rows(
        self, question: str, od_result: dict
    ) -> str | None:
        """Use the LLM to produce a tailored answer from actual Open Data rows."""
        rows = od_result.get("rows") or []
        if not rows:
            return None

        source = od_result.get("source", "NYC Open Data")
        rows_for_prompt = rows[:12]
        rows_json = json.dumps(rows_for_prompt, indent=1, default=str)
        if len(rows_json) > 6000:
            rows_json = rows_json[:6000] + "\n... (truncated)"

        user_msg = _SYNTHESIZE_USER.format(
            question=question,
            source=source,
            rows_json=rows_json,
        )
        content = await self._call_llm(_SYNTHESIZE_SYSTEM, user_msg, temperature=0.3)
        if content:
            LOGGER.info("LLM synthesized answer from %d rows", len(rows_for_prompt))
        return content

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

        # If 311/priority returned no rows, try dynamic dataset discovery
        if not od_result or not od_result.get("rows"):
            LOGGER.info("311 returned no rows — trying dynamic dataset discovery")
            try:
                od_result = await self._discover_and_query(query_text)
            except Exception as exc:
                LOGGER.error("Dynamic dataset discovery failed: %s", exc)

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
                f"Based on recent data: {od_answer}"
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
                    "the right information right now. Please try calling 311 directly."
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
