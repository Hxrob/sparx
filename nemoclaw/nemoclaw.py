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

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Allow imports from sibling packages
_SPARX_ROOT = Path(__file__).resolve().parent.parent
if str(_SPARX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARX_ROOT))

from form_finder.form_finder import FormFinder, FormFinderError
import data_lookup

LOGGER = logging.getLogger("nemoclaw")


@dataclass
class NemoClawResult:
    """Final result handed back to the server / UI layer."""

    source: str                       # "open_data" | "form_finder" | "none"
    decision: str                     # original routing decision
    categories: list[str]             # original categories
    transcript: str

    # Populated when source == "open_data"
    open_data: dict | None = None     # {answer, source, records}

    # Populated when source == "form_finder"
    form_finder: dict | None = None   # {intent, form_name, form_url, summary, next_steps}

    # Plain-language response to speak back to the user
    response: str = ""


@dataclass
class NemoClaw:
    """Orchestrator that routes categorized complaints through Open Data
    then FormFinder."""

    _finder: FormFinder = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._finder = FormFinder()

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

        # Step 1 — try NYC Open Data
        LOGGER.info("Trying Open Data lookup for: %s", query_text[:80])
        od_result = data_lookup.search(query_text, categories=categories)

        if od_result:
            LOGGER.info("Open Data hit: %s", od_result.get("source", ""))
            return NemoClawResult(
                source="open_data",
                decision=decision,
                categories=categories,
                transcript=transcript,
                open_data=od_result,
                response=od_result.get("answer", response),
            )

        # Step 2 — fall back to FormFinder (311 complaint form lookup)
        LOGGER.info("Open Data miss — falling back to FormFinder")
        try:
            ff_result = self._finder.classify(query_text)
            return NemoClawResult(
                source="form_finder",
                decision=decision,
                categories=categories,
                transcript=transcript,
                form_finder=ff_result,
                response=(
                    f"I found a matching NYC 311 form: {ff_result['form_name']}. "
                    f"{ff_result['summary']}"
                ),
            )
        except FormFinderError as exc:
            LOGGER.error("FormFinder failed: %s", exc)
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
