"""
NYC Open Data lookup for NemoClaw.

Uses the public Socrata **Catalog API** to find relevant datasets on
`data.cityofnewyork.us`, then queries actual rows from those datasets
via the SODA API to return real data that NemoClaw's LLM can use to
synthesize a tailored answer.

Optional: `NYC_OPEN_DATA_APP_TOKEN` — create a free app token at
https://data.cityofnewyork.us/profile/edit (raises rate limits when set).

NemoClaw calls `search()` first; if it returns None, NemoClaw falls back to
FormFinder.

Return shape:
    {"answer": str, "source": str, "records": list[dict],
     "portal_url": str, "rows": list[dict]}
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger("data_lookup")
DATA_DIR = Path(__file__).parent / "data"

_CATALOG = "https://api.us.socrata.com/api/catalog/v1"
_NYC_DOMAIN = "data.cityofnewyork.us"
_PORTAL = "https://opendata.cityofnewyork.us"
_DATA_SITE = f"https://{_NYC_DOMAIN}"
# 311 Service Requests from 2010 to Present — optional sample rows for context
_RESOURCE_311 = "erm2-nwe9"


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "User-Agent": "SparX-NemoClaw/1.0 (NYC Open Data catalog)",
    }
    token = os.environ.get("NYC_OPEN_DATA_APP_TOKEN", "").strip()
    if token:
        h["X-App-Token"] = token
    return h


def _get_json(url: str, timeout: float = 20.0) -> object | None:
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        LOGGER.warning("Open Data request failed (%s): %s", url[:100], exc)
        return None


def _catalog_hits(search_text: str, limit: int = 6) -> list[dict]:
    """Return catalog rows: title, url, description, dataset_id."""
    params = urllib.parse.urlencode(
        {
            "domains": _NYC_DOMAIN,
            "only": "dataset",
            "limit": str(limit),
            "q": search_text[:500],
        }
    )
    data = _get_json(f"{_CATALOG}?{params}")
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for item in data.get("results") or []:
        res = item.get("resource") or {}
        rid = (res.get("id") or "").strip()
        name = (res.get("name") or "").strip() or "Untitled dataset"
        desc = (res.get("description") or "").strip()
        if len(desc) > 280:
            desc = desc[:277] + "..."

        link = (item.get("link") or "").strip()
        if not link:
            link = (res.get("permalink") or "").strip()
        if not link and rid:
            link = f"{_DATA_SITE}/d/{rid}"

        if link:
            out.append(
                {
                    "title": name,
                    "url": link,
                    "description": desc,
                    "dataset_id": rid,
                }
            )
    return out


def _short_query(query: str) -> str:
    """Extract a short keyword query from a natural-language sentence.

    SODA $q does full-text search — long sentences cause slow scans.
    Keep the most meaningful words (skip filler) and cap at ~60 chars.
    """
    _FILLER = {
        "i", "im", "i'm", "me", "my", "a", "an", "the", "is", "are", "was",
        "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "that",
        "this", "it", "its", "and", "or", "but", "not", "no", "so", "if",
        "about", "can", "could", "would", "should", "will", "just", "also",
        "very", "really", "looking", "need", "want", "help", "please", "know",
        "there", "where", "how", "what", "which", "who", "when",
    }
    words = query.lower().split()
    keywords = [w for w in words if w not in _FILLER and len(w) > 1]
    if not keywords:
        keywords = words[:5]
    return " ".join(keywords[:8])[:60]


def _query_dataset_rows(
    dataset_id: str, query: str, limit: int = 5
) -> list[dict]:
    """Query actual rows from a dataset via the SODA API using full-text search."""
    if not dataset_id or not query:
        return []
    q = _short_query(query)
    if not q:
        return []
    params = urllib.parse.urlencode({"$q": q, "$limit": str(limit)})
    url = f"{_DATA_SITE}/resource/{dataset_id}.json?{params}"
    data = _get_json(url, timeout=25.0)
    if not isinstance(data, list):
        return []
    # Trim overly large field values to keep payloads manageable
    trimmed: list[dict] = []
    for row in data:
        clean = {}
        for k, v in row.items():
            if k.startswith(":"):  # skip Socrata meta-columns like :id
                continue
            if isinstance(v, str) and len(v) > 400:
                v = v[:397] + "..."
            clean[k] = v
        if clean:
            trimmed.append(clean)
    return trimmed


def _query_top_datasets(
    hits: list[dict], query: str, max_datasets: int = 3, rows_per: int = 5
) -> list[dict]:
    """Query actual rows from the top catalog hits *in parallel*.

    Returns a flat list of dicts, each tagged with the source dataset title/id.
    """
    # Collect the datasets we'll query
    targets: list[dict] = []
    for hit in hits:
        did = hit.get("dataset_id", "").strip()
        if did:
            targets.append(hit)
        if len(targets) >= max_datasets:
            break

    if not targets:
        return []

    def _fetch(hit: dict) -> list[dict]:
        did = hit["dataset_id"]
        rows = _query_dataset_rows(did, query, limit=rows_per)
        for row in rows:
            row["_dataset_title"] = hit.get("title", did)
            row["_dataset_id"] = did
        return rows

    all_rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_datasets) as pool:
        futures = {pool.submit(_fetch, t): t for t in targets}
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                hit = futures[fut]
                LOGGER.warning("SODA query failed for %s: %s", hit.get("dataset_id"), exc)

    return all_rows


def _sample_311_lines(query: str, limit: int = 2) -> list[str]:
    """Pull a couple of 311 rows via SODA `$q` for extra civic context."""
    q = query.strip()[:200]
    if not q:
        return []
    params = urllib.parse.urlencode({"$q": q, "$limit": str(limit)})
    data = _get_json(f"{_DATA_SITE}/resource/{_RESOURCE_311}.json?{params}")
    if not isinstance(data, list):
        return []
    lines: list[str] = []
    for row in data:
        ct = (row.get("complaint_type") or "").strip()
        if not ct:
            continue
        boro = (row.get("borough") or "").strip()
        suf = f" ({boro})" if boro else ""
        lines.append(f"311 example: {ct}{suf}")
    return lines


def search(
    query: str,
    categories: list[str] | None = None,
) -> dict | None:
    """Search NYC Open Data (catalog + optional 311 samples).

    Args:
        query: The user's question / complaint in plain text.
        categories: Optional NYC benefit categories to widen the catalog query.

    Returns:
        {answer, source, records, portal_url} or None if the catalog has no hits
        and we cannot offer a useful Open Data response.
    """
    raw = (query or "").strip()
    if not raw:
        return None

    q_parts = [raw]
    if categories:
        for c in categories:
            if isinstance(c, str) and c.strip():
                q_parts.append(c.strip())
    catalog_q = " ".join(q_parts)[:500]

    hits = _catalog_hits(catalog_q)
    if not hits:
        short = " ".join(raw.split())[:100]
        if short and short != catalog_q:
            hits = _catalog_hits(short)

    if not hits:
        LOGGER.info("NYC Open Data catalog returned no datasets")
        return None

    records: list[dict] = []
    for h in hits:
        t, url, desc = h["title"], h["url"], h.get("description") or ""
        records.append(
            {
                "title": t,
                "url": url,
                "description": desc,
                "dataset_id": h.get("dataset_id", ""),
            }
        )

    # Query actual rows from the top datasets via SODA
    rows = _query_top_datasets(hits, raw, max_datasets=3, rows_per=5)
    LOGGER.info(
        "Queried %d rows from %d datasets",
        len(rows),
        len({r.get("_dataset_id") for r in rows}),
    )

    # Build a basic answer — NemoClaw's LLM will synthesize a tailored one
    # from the rows, but we include a fallback summary here
    body_lines: list[str] = []
    for h in hits:
        t, url = h["title"], h["url"]
        body_lines.append(f"{t}. Link: {url}")

    extra = _sample_311_lines(raw, limit=2)
    if extra:
        body_lines.append(
            "For neighborhood context, recent 311 categories matching "
            "your words include:"
        )
        body_lines.extend(extra)

    answer = (
        "I found related datasets on NYC Open Data. "
        + " | ".join(body_lines[:4])
    )

    return {
        "answer": answer,
        "source": f"NYC Open Data ({_NYC_DOMAIN} catalog)",
        "records": records,
        "rows": rows,
        "portal_url": _PORTAL,
    }
