"""
Document and Image Intake for SparX.

Accepts uploaded files and extracts structured information relevant to
NYC civic assistance — eviction notices, lease violations, utility shutoffs,
court summons, HPD violation notices, benefits denial letters.

Pipeline:
  Image (JPG/PNG/WebP) → vision LLM → structured JSON extraction
  PDF                  → pypdf text  → text LLM → structured JSON extraction

The extracted "situation" field feeds directly into the direction engine
and NemoClaw, exactly like a voice transcript would.

All processing is local — no cloud, no third-party OCR services.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

LOGGER = logging.getLogger("document_intake")

LLM_CHAT_URL = os.environ.get(
    "SPARX_LLM_CHAT_URL", "http://localhost:8081/v1/chat/completions"
)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
SUPPORTED_DOC_TYPES   = {"application/pdf"}

_EXTRACTION_SYSTEM = """\
You are a NYC social services document analyst. A resident has uploaded a document
and needs help understanding it. Many of these residents speak Spanish or other
languages and may not fully understand legal or government documents.

Extract the following as a single JSON object:
{
  "document_type": "eviction_notice | court_summons | utility_shutoff | lease | hpd_violation | benefits_denial | pay_stub | id_document | other",
  "urgency": "low | medium | high | critical",
  "key_dates": [{"label": "string describing the date", "date": "YYYY-MM-DD or plain description"}],
  "amounts_owed": [{"label": "what the amount is for", "amount": "dollar amount as string"}],
  "address": "property address mentioned in the document, or empty string",
  "case_number": "case, docket, or reference number if present, or empty string",
  "issuing_agency": "name of the agency or company that sent this",
  "required_actions": ["list of things the person must do"],
  "deadline_days": null,
  "summary": "2-3 sentence plain English summary of what this document means for the resident. Write as if explaining to someone who has never seen this type of document.",
  "situation": "Describe this person's situation in 1-2 natural sentences as if they were telling a social worker. This will be used to route them to the right resources. Example: My landlord sent me an eviction notice saying I have 14 days to pay $2,400 or leave my apartment at 456 Grand Street Brooklyn."
}

deadline_days should be an integer (number of days from today until the most urgent deadline) or null if no deadline is clear.
urgency should be critical if deadline_days <= 7, high if <= 30, medium if <= 90, low otherwise.

Output ONLY the JSON object. No markdown, no prose before or after.
"""

_EXTRACTION_USER_TEXT = "Please analyze this document and extract the key information."

_EXTRACTION_USER_IMAGE = "Please analyze this document image and extract the key information."


@dataclass
class DocumentResult:
    document_type:    str
    urgency:          str                      # low | medium | high | critical
    summary:          str
    situation:        str                      # feeds into direction engine
    key_dates:        list[dict] = field(default_factory=list)
    amounts_owed:     list[dict] = field(default_factory=list)
    address:          str = ""
    case_number:      str = ""
    issuing_agency:   str = ""
    required_actions: list[str] = field(default_factory=list)
    deadline_days:    int | None = None
    raw_text:         str = ""                 # extracted text (PDFs) or empty
    error:            str = ""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )
    return _client


def _parse_json(raw: str) -> dict | None:
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip("` \n")
    # Find first {...}
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _build_result(data: dict, raw_text: str = "") -> DocumentResult:
    return DocumentResult(
        document_type=    str(data.get("document_type", "other")),
        urgency=          str(data.get("urgency", "medium")),
        summary=          str(data.get("summary", "")),
        situation=        str(data.get("situation", "")),
        key_dates=        list(data.get("key_dates") or []),
        amounts_owed=     list(data.get("amounts_owed") or []),
        address=          str(data.get("address") or ""),
        case_number=      str(data.get("case_number") or ""),
        issuing_agency=   str(data.get("issuing_agency") or ""),
        required_actions= list(data.get("required_actions") or []),
        deadline_days=    data.get("deadline_days"),
        raw_text=         raw_text[:4000],  # cap stored text
    )


# ---------------------------------------------------------------------------
# Image analysis — vision LLM
# ---------------------------------------------------------------------------

async def analyze_image(image_bytes: bytes, mime_type: str) -> DocumentResult:
    """
    Send image to the local vision-capable LLM via OpenAI vision format.
    Falls back gracefully if the model doesn't support vision.
    """
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime_type};base64,{b64}"

    payload: dict = {
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": _EXTRACTION_USER_IMAGE},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens":  1024,
    }
    model = os.environ.get("SPARX_LLM_MODEL")
    if model:
        payload["model"] = model

    try:
        client = _get_client()
        resp = await client.post(LLM_CHAT_URL, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        data = _parse_json(content)
        if data:
            LOGGER.info("Image analysis complete: type=%s urgency=%s",
                        data.get("document_type"), data.get("urgency"))
            return _build_result(data)
        raise ValueError("No JSON in LLM response")
    except Exception as exc:
        LOGGER.error("Image analysis failed: %s", exc)
        return DocumentResult(
            document_type="unknown",
            urgency="medium",
            summary="Could not analyze this image automatically.",
            situation="A resident uploaded a document image that could not be processed.",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# PDF analysis — text extraction + text LLM
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as exc:
        LOGGER.error("PDF text extraction failed: %s", exc)
        return ""


async def analyze_pdf(pdf_bytes: bytes) -> DocumentResult:
    """Extract text from PDF then send to LLM for structured analysis."""
    raw_text = _extract_pdf_text(pdf_bytes)

    if not raw_text.strip():
        # Scanned PDF — no extractable text
        LOGGER.warning("PDF has no extractable text (scanned?)")
        return DocumentResult(
            document_type="unknown",
            urgency="medium",
            summary="This PDF appears to be a scanned image. Text could not be extracted automatically.",
            situation="A resident uploaded a scanned PDF document that could not be read.",
            error="no_extractable_text",
        )

    # Trim to avoid blowing context — keep first 6000 chars
    trimmed = raw_text[:6000]
    if len(raw_text) > 6000:
        trimmed += "\n... [document truncated]"

    payload: dict = {
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{_EXTRACTION_USER_TEXT}\n\n"
                    f"DOCUMENT TEXT:\n{trimmed}"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens":  1024,
    }
    model = os.environ.get("SPARX_LLM_MODEL")
    if model:
        payload["model"] = model

    try:
        client = _get_client()
        resp = await client.post(LLM_CHAT_URL, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        data = _parse_json(content)
        if data:
            LOGGER.info("PDF analysis complete: type=%s urgency=%s",
                        data.get("document_type"), data.get("urgency"))
            return _build_result(data, raw_text=raw_text)
        raise ValueError("No JSON in LLM response")
    except Exception as exc:
        LOGGER.error("PDF analysis failed: %s", exc)
        return DocumentResult(
            document_type="unknown",
            urgency="medium",
            summary="Could not analyze this PDF automatically.",
            situation="A resident uploaded a PDF document that could not be processed.",
            raw_text=raw_text,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

async def analyze(file_bytes: bytes, filename: str, content_type: str) -> DocumentResult:
    """
    Route to the correct analyzer based on MIME type.
    Called by the /upload endpoint in server.py.
    """
    ct = content_type.lower().split(";")[0].strip()

    if ct in SUPPORTED_IMAGE_TYPES:
        return await analyze_image(file_bytes, ct)

    if ct in SUPPORTED_DOC_TYPES or filename.lower().endswith(".pdf"):
        return await analyze_pdf(file_bytes)

    # Unknown type — try to detect from content
    if file_bytes[:4] == b"%PDF":
        return await analyze_pdf(file_bytes)

    LOGGER.warning("Unsupported file type: %s (%s)", filename, ct)
    return DocumentResult(
        document_type="unknown",
        urgency="low",
        summary="",
        situation="",
        error=f"Unsupported file type: {ct}. Supported: JPG, PNG, WebP, PDF.",
    )
