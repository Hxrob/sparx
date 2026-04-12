"""
NYC Open Data lookup for NemoClaw.

Uses the public Socrata **Catalog API** to search datasets on
`data.cityofnewyork.us`, then returns human-readable answers with links to the
official portal (https://opendata.cityofnewyork.us/).

Optional: `NYC_OPEN_DATA_APP_TOKEN` — create a free app token at
https://data.cityofnewyork.us/profile/edit (raises rate limits when set).

NemoClaw calls `search()` first; if it returns None, NemoClaw falls back to
FormFinder.

Return shape:
    {"answer": str, "source": str, "records": list[dict], "portal_url": str}
"""

from __future__ import annotations

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

    intro = (
        "I found related datasets on NYC Open Data — the city's public data portal. "
        f"You can open each link to explore or download data. More datasets: {_PORTAL}/ ."
    )
    body_lines: list[str] = []
    records: list[dict] = []

    for h in hits:
        t, url, desc = h["title"], h["url"], h.get("description") or ""
        body_lines.append(f"{t}. Link: {url}")
        if desc:
            body_lines.append(desc)
        records.append(
            {
                "title": t,
                "url": url,
                "description": desc,
                "dataset_id": h.get("dataset_id", ""),
            }
        )

    extra = _sample_311_lines(raw, limit=2)
    if extra:
        body_lines.append("For neighborhood context, recent 311 categories matching your words include:")
        body_lines.extend(extra)

    answer = intro + "\n\n" + "\n\n".join(body_lines)

    return {
        "answer": answer,
        "source": f"NYC Open Data ({_NYC_DOMAIN} catalog)",
        "records": records,
        "portal_url": _PORTAL,
    }
