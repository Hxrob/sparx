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

    # Common synonyms the LLM might use instead of the exact catalog name
    _SYNONYMS = {
        "rodent": "rat or mouse",
        "pest": "rat or mouse",
        "vermin": "rat or mouse",
        "cockroach": "residential pest",
        "roach": "residential pest",
        "insect": "residential pest",
        "garbage": "dirty condition",
        "trash": "dirty condition",
        "litter": "dirty condition",
        "pothole": "pothole or cave-in",
        "double parking": "illegal parking",
        "street light": "street light complaint",
        "lamp post": "street light complaint",
        "sidewalk": "sidewalk condition",
        "elevator": "elevator complaint",
        "mold": "mold complaint",
        "asbestos": "asbestos complaint",
        "lead paint": "lead complaint",
    }

    def _expand_synonyms(self, text: str) -> str:
        """Replace common synonym words so they match catalog names."""
        lower = text.lower()
        for syn, replacement in self._SYNONYMS.items():
            if syn in lower:
                lower = lower.replace(syn, replacement)
        return lower

    def _resolve(self, form_name: str) -> tuple[str, str]:
        """Map an LLM-produced form name to its canonical name and verified URL.

        Uses multiple matching strategies to ensure we almost always resolve
        to a real KA entry instead of falling back to the generic URL.
        """
        # 1. Exact match
        item = self._by_name.get(form_name)
        if item:
            return item["name"], _ka_url(item["ka_number"])

        # 2. Case-insensitive exact match
        lower = form_name.lower()
        for name, it in self._by_name.items():
            if name.lower() == lower:
                return it["name"], _ka_url(it["ka_number"])

        # 3. Substring match — LLM name contained in a catalog entry or vice versa
        for name, it in self._by_name.items():
            nl = name.lower()
            if lower in nl or nl in lower:
                return it["name"], _ka_url(it["ka_number"])

        # Expand synonyms so "rodent" matches "rat or mouse", etc.
        expanded = self._expand_synonyms(lower)

        # Skip generic filler words when scoring
        _NOISE = {"complaint", "report", "a", "an", "the", "in", "on", "of",
                   "for", "and", "or", "with", "from", "to", "my", "about",
                   "general", "311", "nyc", "city", "new", "york"}
        query_words = set(expanded.split()) - _NOISE

        if not query_words:
            return form_name, "https://portal.311.nyc.gov/report-problems/"

        # 4. Word overlap scoring — ignore filler, rank by meaningful overlap
        best, best_score = None, 0.0
        for name, it in self._by_name.items():
            name_words = set(name.lower().split()) - _NOISE
            if not name_words:
                continue
            overlap = query_words & name_words
            if not overlap:
                continue
            # Score: fraction of query words matched + fraction of name words matched
            score = (len(overlap) / len(query_words) + len(overlap) / len(name_words)) / 2
            if score > best_score:
                best, best_score = it, score
        if best and best_score >= 0.3:
            return best["name"], _ka_url(best["ka_number"])

        # 5. Description keyword match — check if meaningful words appear in descriptions
        best_desc, best_desc_score = None, 0.0
        for name, it in self._by_name.items():
            desc = it.get("description", "").lower()
            hits = sum(1 for w in query_words if w in desc)
            if hits > best_desc_score:
                best_desc, best_desc_score = it, hits
        if best_desc and best_desc_score >= 2:
            return best_desc["name"], _ka_url(best_desc["ka_number"])

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
