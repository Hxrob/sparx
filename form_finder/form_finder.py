"""
FormFinder – NYC 311 complaint classifier module.

Uses the same OpenAI-compatible chat endpoint as the voice direction engine.
Override with SPARX_LLM_CHAT_URL; optional SPARX_LLM_MODEL for the JSON body.

Usage by NemoClaw or any other caller:

    from form_finder import FormFinder

    finder = FormFinder()
    result = finder.classify("my landlord hasn't turned on the heat")

    print(result["form_name"])   # "Heat or Hot Water Complaint in a Residential Building"
    print(result["form_url"])    # "https://portal.311.nyc.gov/article/?kanumber=KA-01036"
    print(result["intent"])      # "heat"
    print(result["summary"])     # "..."
    print(result["next_steps"])  # ["...", "...", "..."]
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_DATA_PATH = Path(__file__).parent / "KA.json"


def _default_llm_chat_url() -> str:
    """Same OpenAI-compatible endpoint as the direction engine (llama.cpp, vLLM, etc.)."""
    return os.environ.get(
        "SPARX_LLM_CHAT_URL", "http://localhost:8081/v1/chat/completions"
    )


@dataclass
class FormFinder:
    llm_url: str = field(default_factory=_default_llm_chat_url)
    data_path: Path = _DEFAULT_DATA_PATH
    timeout: int = 120

    _items: list[dict] = field(default_factory=list, repr=False, init=False)
    _by_name: dict[str, dict] = field(default_factory=dict, repr=False, init=False)
    _system_prompt: str = field(default="", repr=False, init=False)

    def __post_init__(self) -> None:
        raw = json.loads(Path(self.data_path).read_text())
        self._items = raw["items"]
        self._by_name = {item["name"]: item for item in self._items}

        catalogue = "\n".join(
            f"  - {item['name']}: {item['description']}" for item in self._items
        )
        self._system_prompt = (
            "You are a NYC 311 assistant.\n\n"
            "You have access to the complete official NYC 311 complaint directory below. "
            "Given the user's complaint or question, pick the SINGLE best matching complaint "
            "name from the directory (copy it exactly as written) and respond with ONLY valid JSON:\n\n"
            "{\n"
            '  "intent": "short category label, e.g. heat, noise, rodent, parking",\n'
            '  "form_name": "<exact complaint name from the directory below>",\n'
            '  "summary": "1-2 sentence plain English summary of what to do",\n'
            '  "next_steps": ["step 1", "step 2", "step 3"]\n'
            "}\n\n"
            "Do NOT include a form_url field — it will be added automatically.\n"
            'If no entry fits, set form_name to "General 311 Complaint".\n\n'
            f"NYC 311 Complaint Directory:\n{catalogue}"
        )

    @property
    def entry_count(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    # Public API — this is what NemoClaw should call
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict:
        """Classify a user complaint and return the matching 311 form info.

        Returns a dict with keys:
            intent, form_name, form_url, summary, next_steps
        Raises FormFinderError on failure.
        """
        import requests

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": text},
        ]

        payload: dict = {"messages": messages}
        model = os.environ.get("SPARX_LLM_MODEL")
        if model:
            payload["model"] = model

        try:
            resp = requests.post(
                self.llm_url, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise FormFinderError(f"LLM request failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise FormFinderError(f"LLM response is not valid JSON: {exc}") from exc
        choices = data.get("choices", [])
        if not choices or "message" not in choices[0]:
            raise FormFinderError(f"Unexpected LLM response: {data}")

        raw_content = choices[0]["message"].get("content", "")
        try:
            parsed = _extract_json(raw_content)
        except ValueError as exc:
            raise FormFinderError(f"Failed to parse LLM output: {exc}\nRaw: {raw_content}") from exc

        form_name = parsed.get("form_name", "General 311 Complaint")
        canonical, url = self._resolve(form_name)

        return {
            "intent": parsed.get("intent", ""),
            "form_name": canonical,
            "form_url": url,
            "summary": parsed.get("summary", ""),
            "next_steps": parsed.get("next_steps", []),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, form_name: str) -> tuple[str, str]:
        """Map an LLM-produced form name to its canonical name and verified URL."""
        item = self._by_name.get(form_name)
        if item:
            return item["name"], _ka_url(item["ka_number"])

        lower = form_name.lower()
        for name, it in self._by_name.items():
            if name.lower() == lower:
                return it["name"], _ka_url(it["ka_number"])

        best, best_score = None, 0
        for name, it in self._by_name.items():
            overlap = len(set(name.lower().split()) & set(lower.split()))
            if overlap > best_score:
                best, best_score = it, overlap
        if best and best_score >= 2:
            return best["name"], _ka_url(best["ka_number"])

        return form_name, "https://portal.311.nyc.gov/report-problems/"


class FormFinderError(Exception):
    """Raised when FormFinder classification fails."""


def _ka_url(ka_number: str) -> str:
    return f"https://portal.311.nyc.gov/article/?kanumber={ka_number}"


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise ValueError("No JSON object found in LLM response")
