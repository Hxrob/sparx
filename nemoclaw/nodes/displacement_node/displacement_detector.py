"""
Displacement Detector — predictive tenant displacement risk scoring.

Pulls three live signals from NYC Open Data in parallel for any address:

  1. HPD Violations (wvxf-dwi5)    — harassment pattern: heat, hot water, roaches
  2. Eviction Filings (6z8x-wfk4)  — active displacement in housing court
  3. HPD Registrations (tesw-yqqr) — recent ownership change, corporate owner flag

Scores 0.0–1.0. Anything above 0.55 triggers a proactive alert to the caller.
All queries are public NYC Open Data — no cloud, no privacy risk.

Usage:
    result = score("123 Main Street Brooklyn")
    if result.risk_level in ("high", "critical"):
        # alert tenant, route to Legal Aid
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta

LOGGER = logging.getLogger("displacement_detector")

_DATA_SITE   = "https://data.cityofnewyork.us"
_HPD_VIOLATIONS   = "wvxf-dwi5"   # HPD housing maintenance violations
_EVICTIONS        = "6z8x-wfk4"   # NYC marshal eviction executions
_HPD_REGISTRATIONS = "tesw-yqqr"  # HPD multiple dwelling registrations (ownership)

# Scoring weights
_W_VIOLATIONS  = 0.40
_W_EVICTIONS   = 0.40
_W_OWNERSHIP   = 0.20

# Risk thresholds
MODERATE_THRESHOLD = 0.30
HIGH_THRESHOLD     = 0.55
CRITICAL_THRESHOLD = 0.75

# Street type normalization
_STREET_ABBREVS = {
    r"\bSt\b":      "Street",
    r"\bAve\b":     "Avenue",
    r"\bBlvd\b":    "Boulevard",
    r"\bRd\b":      "Road",
    r"\bDr\b":      "Drive",
    r"\bLn\b":      "Lane",
    r"\bPl\b":      "Place",
    r"\bCt\b":      "Court",
    r"\bPkwy\b":    "Parkway",
}

_ADDRESS_RE = re.compile(
    r"\b(\d+(?:-\w+)?)\s+"
    r"((?:[A-Za-z]+\s+){0,3}"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
    r"Lane|Ln|Place|Pl|Court|Ct|Way|Parkway|Pkwy|Broadway|"
    r"Concourse|Terrace|Expressway|Heights))\b",
    re.IGNORECASE,
)

_BOROUGH_MAP = {
    "manhattan":    "MANHATTAN",
    "brooklyn":     "BROOKLYN",
    "bronx":        "BRONX",
    "the bronx":    "BRONX",
    "queens":       "QUEENS",
    "staten island":"STATEN ISLAND",
    "bronx":        "BRONX",
}

# Corporate ownership patterns that flag predatory equity
_LLC_RE = re.compile(r"\b(LLC|L\.L\.C|LP|L\.P|CORP|INC|HOLDINGS|EQUITIES|REALTY|PROPERTIES|PARTNERS|ASSOCIATES|CAPITAL|MGMT|MANAGEMENT)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DisplacementResult:
    address: str
    risk_score: float                       # 0.0 – 1.0
    risk_level: str                         # low | moderate | high | critical
    signals: dict = field(default_factory=dict)
    alert_message: str = ""
    found: bool = False                     # False if address not found in any dataset

    @property
    def should_alert(self) -> bool:
        return self.risk_score >= HIGH_THRESHOLD


# ---------------------------------------------------------------------------
# Address extraction
# ---------------------------------------------------------------------------

def extract_address(text: str) -> tuple[str, str] | None:
    """
    Extract (street_address, borough) from a transcript string.
    Returns None if no address pattern is found.
    """
    text_lower = text.lower()

    borough = ""
    for key, val in _BOROUGH_MAP.items():
        if key in text_lower:
            borough = val
            break

    match = _ADDRESS_RE.search(text)
    if not match:
        return None

    number = match.group(1)
    street = match.group(2).strip()

    # Normalize abbreviations
    for abbrev, full in _STREET_ABBREVS.items():
        street = re.sub(abbrev, full, street, flags=re.IGNORECASE)

    address = f"{number} {street}"
    return address, borough


# ---------------------------------------------------------------------------
# HTTP helper (same pattern as data_lookup.py — no extra dependencies)
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "User-Agent": "SparX-Displacement/1.0",
    }
    token = os.environ.get("NYC_OPEN_DATA_APP_TOKEN", "").strip()
    if token:
        h["X-App-Token"] = token
    return h


def _get_json(url: str, timeout: float = 15.0) -> list | dict | None:
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        LOGGER.warning("Open Data request failed (%s): %s", url[:120], exc)
        return None


def _soda_url(dataset: str, params: dict) -> str:
    qs = urllib.parse.urlencode(params)
    return f"{_DATA_SITE}/resource/{dataset}.json?{qs}"


# ---------------------------------------------------------------------------
# Signal queries
# ---------------------------------------------------------------------------

def _query_violations(address: str, borough: str) -> dict:
    """
    HPD housing maintenance violations at the address.
    Counts recent violations (18 months) and classifies severity.
    Class C = immediately hazardous (heat, gas, lead).
    """
    cutoff = (datetime.utcnow() - timedelta(days=548)).strftime("%Y-%m-%dT00:00:00.000")

    q = address
    if borough:
        q = f"{address} {borough}"

    url = _soda_url(_HPD_VIOLATIONS, {
        "$q":     q,
        "$where": f"novissueddate >= '{cutoff}'",
        "$limit": "100",
        "$order": "novissueddate DESC",
    })

    rows = _get_json(url) or []
    if not isinstance(rows, list):
        return {"count": 0, "class_c": 0, "score": 0.0, "sample": []}

    class_c = sum(1 for r in rows if str(r.get("class", "")).upper() == "C")
    total   = len(rows)

    # Score: ramps up with volume, class C violations weight extra
    base = min(total / 15, 1.0) * 0.6 + min(class_c / 6, 1.0) * 0.4

    sample = [
        r.get("novdescription", "")
        for r in rows[:3]
        if r.get("novdescription")
    ]

    LOGGER.info("HPD violations at '%s': %d total, %d class-C", address, total, class_c)
    return {"count": total, "class_c": class_c, "score": float(base), "sample": sample}


def _query_evictions(address: str, borough: str) -> dict:
    """
    Marshal eviction executions at the address (last 12 months).
    Even 1 eviction filing is a serious signal.
    """
    cutoff = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%dT00:00:00.000")

    q = address
    if borough:
        q = f"{address} {borough}"

    url = _soda_url(_EVICTIONS, {
        "$q":     q,
        "$where": f"executed_date >= '{cutoff}'",
        "$limit": "20",
        "$order": "executed_date DESC",
    })

    rows = _get_json(url) or []
    if not isinstance(rows, list):
        return {"count": 0, "score": 0.0, "latest": None}

    count = len(rows)
    # 1 eviction = 0.5 score; 3+ = 1.0
    eviction_score = min(count / 3, 1.0)

    latest = rows[0].get("executed_date", "") if rows else None

    LOGGER.info("Eviction filings at '%s': %d", address, count)
    return {"count": count, "score": float(eviction_score), "latest": latest}


def _query_ownership(address: str, borough: str) -> dict:
    """
    HPD Multiple Dwelling Registration for ownership signals:
    - Corporate owner (LLC, Corp) → predatory equity flag
    - Recent registration change → potential new owner
    """
    q = address
    if borough:
        q = f"{address} {borough}"

    url = _soda_url(_HPD_REGISTRATIONS, {
        "$q":     q,
        "$limit": "5",
        "$order": "lastregistrationdate DESC",
    })

    rows = _get_json(url) or []
    if not isinstance(rows, list) or not rows:
        return {"corporate": False, "recent_change": False, "score": 0.0, "owner": ""}

    row = rows[0]
    owner_parts = [
        row.get("ownerfirstname", ""),
        row.get("ownerlastname", ""),
        row.get("corporationname", ""),
    ]
    owner_name = " ".join(p for p in owner_parts if p).strip()

    corporate = bool(_LLC_RE.search(owner_name)) or row.get("ownertype", "").upper() == "C"

    # Recent registration (within 2 years) = potential ownership change
    last_reg = row.get("lastregistrationdate", "")
    recent_change = False
    if last_reg:
        try:
            reg_date = datetime.fromisoformat(last_reg[:10])
            recent_change = (datetime.utcnow() - reg_date).days < 730
        except ValueError:
            pass

    ownership_score = 0.0
    if corporate:
        ownership_score += 0.6
    if recent_change:
        ownership_score += 0.4

    LOGGER.info(
        "Ownership at '%s': owner=%s corporate=%s recent_change=%s",
        address, owner_name or "unknown", corporate, recent_change,
    )
    return {
        "corporate":     corporate,
        "recent_change": recent_change,
        "score":         float(min(ownership_score, 1.0)),
        "owner":         owner_name,
    }


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score(address_text: str) -> DisplacementResult:
    """
    Score displacement risk for an address extracted from a transcript.

    Runs all three NYC Open Data queries in parallel.
    Returns a DisplacementResult with risk_level and alert_message.
    """
    parsed = extract_address(address_text) if address_text else None

    if not parsed:
        return DisplacementResult(
            address=address_text or "",
            risk_score=0.0,
            risk_level="unknown",
            alert_message="",
            found=False,
        )

    address, borough = parsed
    LOGGER.info("Scoring displacement risk for: %s (%s)", address, borough or "borough unknown")

    # Run all three queries in parallel — Blackwell has the compute, use it
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        v_fut = pool.submit(_query_violations, address, borough)
        e_fut = pool.submit(_query_evictions,  address, borough)
        o_fut = pool.submit(_query_ownership,  address, borough)

        violations = v_fut.result()
        evictions  = e_fut.result()
        ownership  = o_fut.result()

    # Weighted composite score
    risk_score = (
        violations["score"] * _W_VIOLATIONS +
        evictions["score"]  * _W_EVICTIONS  +
        ownership["score"]  * _W_OWNERSHIP
    )
    risk_score = float(min(risk_score, 1.0))

    # Risk level
    if risk_score >= CRITICAL_THRESHOLD:
        risk_level = "critical"
    elif risk_score >= HIGH_THRESHOLD:
        risk_level = "high"
    elif risk_score >= MODERATE_THRESHOLD:
        risk_level = "moderate"
    else:
        risk_level = "low"

    found = (
        violations["count"] > 0
        or evictions["count"] > 0
        or bool(ownership.get("owner"))
    )

    # Human-readable alert
    alert_message = _build_alert(address, risk_level, violations, evictions, ownership)

    LOGGER.info(
        "Displacement risk for '%s': score=%.2f level=%s",
        address, risk_score, risk_level,
    )

    return DisplacementResult(
        address=address,
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        signals={
            "violations": violations,
            "evictions":  evictions,
            "ownership":  ownership,
        },
        alert_message=alert_message,
        found=found,
    )


def _build_alert(
    address: str,
    risk_level: str,
    violations: dict,
    evictions: dict,
    ownership: dict,
) -> str:
    if risk_level == "low":
        return ""

    parts = []

    v_count = violations.get("count", 0)
    c_count = violations.get("class_c", 0)
    if v_count > 0:
        parts.append(
            f"{v_count} housing violation{'s' if v_count != 1 else ''} filed at this address "
            f"in the last 18 months"
            + (f", including {c_count} immediately hazardous" if c_count else "")
            + "."
        )

    e_count = evictions.get("count", 0)
    if e_count > 0:
        parts.append(
            f"{e_count} eviction filing{'s' if e_count != 1 else ''} recorded here "
            f"in the past year."
        )

    owner = ownership.get("owner", "")
    if ownership.get("corporate") and owner:
        parts.append(f"The building is registered to a corporate entity ({owner}).")
    if ownership.get("recent_change"):
        parts.append("Ownership appears to have changed in the last two years.")

    if not parts:
        return ""

    prefix = {
        "moderate": "Heads up:",
        "high":     "Warning:",
        "critical": "This building shows a strong displacement pattern.",
    }.get(risk_level, "")

    return f"{prefix} " + " ".join(parts)
