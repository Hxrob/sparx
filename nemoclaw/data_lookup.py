"""
NYC Open Data lookup for NemoClaw.

Queries actual row data from key NYC datasets via the SODA API using
pandas ``pd.read_json()``.  The primary dataset is 311 Service Requests
(erm2-nwe9) — the single largest source of complaint/service data in NYC.
Priority datasets from SKILL.md are checked when the query category matches.

The Socrata **Catalog API** is used only as a last-resort fallback when
the primary datasets return no rows.

Return shape:
    {"answer": str, "source": str, "records": list[dict],
     "rows": list[dict]}
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("data_lookup")

_NYC_DOMAIN = "data.cityofnewyork.us"
_DATA_SITE = f"https://{_NYC_DOMAIN}"

# ── Primary dataset: 311 Service Requests from 2010 to Present ──
_311_DATASET = "erm2-nwe9"

# ── Priority datasets from SKILL.md ──
_PRIORITY_DATASETS = {
    "sejx-2gn3": {
        "title": "Citywide Public Computer Centers",
        "keywords": ["computer", "internet", "wifi", "digital", "library", "tech"],
    },
    "tc6u-8rnp": {
        "title": "Directory of SNAP Centers",
        "keywords": ["snap", "food", "ebt", "benefit", "hunger", "food stamp"],
    },
    "qafz-7myz": {
        "title": "Bus Stop Shelters",
        "keywords": ["bus", "shelter", "transit", "stop", "accessible", "wheelchair"],
    },
}

# Complaint-type mappings — maps common spoken words to 311 complaint_type values
_COMPLAINT_MAPPINGS = {
    "noise":        "Noise - Residential",
    "loud":         "Noise - Residential",
    "music":        "Noise - Residential",
    "party":        "Noise - Residential",
    "pothole":      "Pothole",
    "heat":         "HEAT/HOT WATER",
    "hot water":    "HEAT/HOT WATER",
    "heating":      "HEAT/HOT WATER",
    "rat":          "Rodent",
    "rats":         "Rodent",
    "mouse":        "Rodent",
    "mice":         "Rodent",
    "roach":        "Unsanitary Condition",
    "cockroach":    "Unsanitary Condition",
    "trash":        "Dirty Conditions",
    "garbage":      "Dirty Conditions",
    "litter":       "Dirty Conditions",
    "dirty":        "Dirty Conditions",
    "graffiti":     "Graffiti",
    "parking":      "Illegal Parking",
    "double park":  "Illegal Parking",
    "hydrant":      "Illegal Parking",
    "tree":         "Damaged Tree",
    "branch":       "Damaged Tree",
    "sidewalk":     "Damaged Tree",
    "water leak":   "Water System",
    "pipe":         "Water System",
    "flood":        "Water System",
    "sewer":        "Sewer",
    "drain":        "Sewer",
    "street light": "Street Light Condition",
    "light out":    "Street Light Condition",
    "lamp":         "Street Light Condition",
    "homeless":     "Homeless Encampment",
    "encampment":   "Homeless Encampment",
    "elevator":     "Elevator",
    "smoke":        "Air Quality",
    "air quality":  "Air Quality",
    "asbestos":     "Asbestos",
    "lead":         "Lead",
    "paint":        "Lead",
    "mold":         "Mold",
    "bed bug":      "Bed Bugs",
    "bedbug":       "Bed Bugs",
    "construction": "Construction",
    "building":     "Building/Use",
    "fire escape":  "Fire Safety Director - Loss of Certificate",
    "rent":         "HPD Literature Request",
    "landlord":     "HPD Literature Request",
    "eviction":     "HPD Literature Request",
    "tenant":       "HPD Literature Request",
}

# Boroughs recognized in speech
_BOROUGH_MAP = {
    "manhattan":     "MANHATTAN",
    "bronx":         "BRONX",
    "brooklyn":      "BROOKLYN",
    "queens":        "QUEENS",
    "staten island": "STATEN ISLAND",
    "staten":        "STATEN ISLAND",
}


def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "User-Agent": "SparX-NemoClaw/1.0",
    }
    token = os.environ.get("NYC_OPEN_DATA_APP_TOKEN", "").strip()
    if token:
        h["X-App-Token"] = token
    return h


def _soda_url(dataset_id: str, where: str = "", q: str = "",
              order: str = "", limit: int = 10) -> str:
    """Build a SODA API URL with query parameters."""
    params: dict[str, str] = {"$limit": str(limit)}
    if where:
        params["$where"] = where
    if q:
        params["$q"] = q
    if order:
        params["$order"] = order
    qs = urllib.parse.urlencode(params)
    return f"{_DATA_SITE}/resource/{dataset_id}.json?{qs}"


def _query_pandas(dataset_id: str, where: str = "", q: str = "",
                  order: str = "", limit: int = 10) -> pd.DataFrame:
    """Query a SODA dataset and return a DataFrame via pd.read_json()."""
    url = _soda_url(dataset_id, where=where, q=q, order=order, limit=limit)
    # Attach app token via header-aware approach
    token = os.environ.get("NYC_OPEN_DATA_APP_TOKEN", "").strip()
    storage_options = {"User-Agent": "SparX-NemoClaw/1.0"}
    if token:
        storage_options["X-App-Token"] = token

    try:
        df = pd.read_json(url, storage_options=storage_options)
        return df
    except Exception as exc:
        LOGGER.warning("pd.read_json failed for %s: %s", dataset_id, exc)
        return pd.DataFrame()


def _df_to_rows(df: pd.DataFrame, max_rows: int = 10) -> list[dict]:
    """Convert a DataFrame to a list of cleaned dicts (drop Socrata meta cols)."""
    if df.empty:
        return []
    # Drop Socrata internal columns
    drop_cols = [c for c in df.columns if c.startswith(":")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    rows = df.head(max_rows).to_dict(orient="records")
    # Trim long string values
    cleaned: list[dict] = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if pd.isna(v):
                continue
            if isinstance(v, str) and len(v) > 400:
                v = v[:397] + "..."
            clean[k] = v
        if clean:
            cleaned.append(clean)
    return cleaned


# ── Extractors: pull complaint type and borough from transcript ──

def _extract_complaint_type(query: str) -> str | None:
    """Match spoken words to a known 311 complaint_type."""
    q = query.lower()
    for phrase, ctype in _COMPLAINT_MAPPINGS.items():
        if phrase in q:
            return ctype
    return None


def _extract_borough(query: str) -> str | None:
    """Extract a borough name from the query."""
    q = query.lower()
    for phrase, boro in _BOROUGH_MAP.items():
        if phrase in q:
            return boro
    return None


def _extract_address(query: str) -> str | None:
    """Try to extract a street address from the query."""
    # Match patterns like "123 Main St" or "45-67 Broadway"
    m = re.search(r'\b(\d{1,5}(?:-\d{1,5})?\s+[A-Za-z][\w\s]{2,30}(?:st|street|ave|avenue|blvd|boulevard|rd|road|pl|place|dr|drive|way|ln|lane|ct|court))\b', query, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ── 311 Query Builder ──

def _build_311_where(query: str) -> tuple[str, str]:
    """Build a $where clause and description for querying 311 data.

    Returns (where_clause, description_of_filter).
    """
    clauses: list[str] = []
    desc_parts: list[str] = []

    # Complaint type
    ctype = _extract_complaint_type(query)
    if ctype:
        clauses.append(f"complaint_type='{ctype}'")
        desc_parts.append(f"complaint type '{ctype}'")

    # Borough
    boro = _extract_borough(query)
    if boro:
        clauses.append(f"borough='{boro}'")
        desc_parts.append(f"in {boro}")

    # Address (partial match via LIKE)
    addr = _extract_address(query)
    if addr:
        # Use upper() for SODA case-insensitive match
        clauses.append(f"upper(incident_address) LIKE '%{addr}%'")
        desc_parts.append(f"near {addr}")

    # Always limit to recent complaints (last 12 months)
    clauses.append("created_date > '2025-04-01T00:00:00'")

    where = " AND ".join(clauses)
    desc = ", ".join(desc_parts) if desc_parts else "general complaints"

    return where, desc


def _query_311(query: str) -> tuple[list[dict], str]:
    """Query the 311 Service Requests dataset with targeted filters.

    Returns (rows, filter_description).
    """
    where, desc = _build_311_where(query)

    # Primary query: specific $where filters, ordered by most recent
    df = _query_pandas(
        _311_DATASET,
        where=where,
        order="created_date DESC",
        limit=15,
    )

    rows = _df_to_rows(df)

    # If specific filters returned nothing, try full-text search as fallback
    if not rows:
        short_q = _short_keywords(query)
        if short_q:
            df2 = _query_pandas(
                _311_DATASET,
                q=short_q,
                order="created_date DESC",
                limit=10,
            )
            rows = _df_to_rows(df2)
            desc = f"full-text search for '{short_q}'"

    return rows, desc


def _short_keywords(query: str) -> str:
    """Extract short keywords for SODA $q full-text search."""
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
    return " ".join(keywords[:6])[:60]


# ── Priority dataset queries ──

def _match_priority_datasets(query: str) -> list[str]:
    """Return dataset IDs from SKILL.md priority list that match the query."""
    q = query.lower()
    matched: list[str] = []
    for did, info in _PRIORITY_DATASETS.items():
        for kw in info["keywords"]:
            if kw in q:
                matched.append(did)
                break
    return matched


def _query_priority_dataset(dataset_id: str, query: str) -> tuple[list[dict], str]:
    """Query a priority dataset with full-text search."""
    title = _PRIORITY_DATASETS.get(dataset_id, {}).get("title", dataset_id)
    short_q = _short_keywords(query)
    if not short_q:
        return [], title

    df = _query_pandas(dataset_id, q=short_q, limit=10)
    return _df_to_rows(df), title


# ── Main search entry point ──

def search(
    query: str,
    categories: list[str] | None = None,
) -> dict | None:
    """Search NYC Open Data for actual complaint/service data.

    Queries the 311 Service Requests dataset first with targeted $where
    filters, then checks priority SKILL.md datasets if relevant.

    Args:
        query: The user's question / complaint in plain text.
        categories: Optional NYC benefit categories from direction engine.

    Returns:
        {answer, source, records, rows} or None if no data found.
    """
    raw = (query or "").strip()
    if not raw:
        return None

    all_rows: list[dict] = []
    sources: list[str] = []
    records: list[dict] = []

    # 1. Always query 311 Service Requests — the primary complaint dataset
    LOGGER.info("Querying 311 dataset for: %s", raw[:80])
    rows_311, filter_desc = _query_311(raw)

    if rows_311:
        for row in rows_311:
            row["_dataset_title"] = "311 Service Requests"
            row["_dataset_id"] = _311_DATASET
        all_rows.extend(rows_311)
        sources.append(f"311 Service Requests ({filter_desc})")
        records.append({
            "title": "311 Service Requests",
            "description": f"Filtered by {filter_desc}",
            "dataset_id": _311_DATASET,
        })

    # 2. Check priority datasets from SKILL.md if query matches their keywords
    priority_ids = _match_priority_datasets(raw)
    if priority_ids:
        def _fetch_priority(did: str) -> tuple[list[dict], str]:
            return _query_priority_dataset(did, raw)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_priority, did): did for did in priority_ids}
            for fut in concurrent.futures.as_completed(futures):
                did = futures[fut]
                try:
                    p_rows, p_title = fut.result()
                    if p_rows:
                        for row in p_rows:
                            row["_dataset_title"] = p_title
                            row["_dataset_id"] = did
                        all_rows.extend(p_rows)
                        sources.append(p_title)
                        records.append({
                            "title": p_title,
                            "description": f"Priority dataset ({did})",
                            "dataset_id": did,
                        })
                except Exception as exc:
                    LOGGER.warning("Priority dataset %s query failed: %s", did, exc)

    if not all_rows:
        LOGGER.info("No rows found from 311 or priority datasets")
        return None

    LOGGER.info(
        "Found %d total rows from: %s",
        len(all_rows), ", ".join(sources),
    )

    # Build a fallback answer from actual row data (LLM synthesis in NemoClaw
    # will override this, but we provide it as backup)
    answer = _build_fallback_answer(rows_311)

    return {
        "answer": answer,
        "source": "; ".join(sources),
        "records": records,
        "rows": all_rows,
    }


def _build_fallback_answer(rows_311: list[dict]) -> str:
    """Build a plain-text fallback answer from 311 rows (no links)."""
    if not rows_311:
        return "No matching 311 complaints found for your query."

    # Summarize the top complaints found
    complaint_types: dict[str, int] = {}
    boroughs: set[str] = set()
    recent_status: list[str] = []

    for row in rows_311[:15]:
        ct = row.get("complaint_type", "")
        if ct:
            complaint_types[ct] = complaint_types.get(ct, 0) + 1
        boro = row.get("borough", "")
        if boro and boro != "Unspecified":
            boroughs.add(boro)
        status = row.get("status", "")
        if status and len(recent_status) < 3:
            addr = row.get("incident_address", "")
            date = str(row.get("created_date", ""))[:10]
            recent_status.append(
                f"{ct} at {addr} ({date}) — {status}" if addr
                else f"{ct} ({date}) — {status}"
            )

    parts: list[str] = []
    if complaint_types:
        top = sorted(complaint_types.items(), key=lambda x: -x[1])[:3]
        type_str = ", ".join(f"{ct} ({n})" for ct, n in top)
        parts.append(f"Found recent 311 complaints: {type_str}.")
    if boroughs:
        parts.append(f"Areas: {', '.join(sorted(boroughs))}.")
    if recent_status:
        parts.append("Recent reports: " + "; ".join(recent_status[:2]) + ".")

    return " ".join(parts) if parts else "Found 311 complaint data matching your query."
