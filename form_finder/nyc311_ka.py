"""
Fetch NYC 311 knowledge articles by KA number and extract section headings plus
service-request create URLs embedded in the page.

Article URLs look like: https://portal.311.nyc.gov/article/?kanumber=KA-02130
SR create URLs use query params caid and kasid as in the portal's JavaScript.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup

PORTAL_BASE = "https://portal.311.nyc.gov"
DEFAULT_UA = (
    "Mozilla/5.0 (compatible; NYC311-KA-scraper/1.0; +https://portal.311.nyc.gov/)"
)

_CREATE_SR_RE = re.compile(
    r"createServiceRequest\s*\(\s*"
    r"'([^']*)'\s*,\s*"
    r"'([^']*)'\s*,\s*"
    r"'([^']*)'\s*,\s*"
    r"'([^']*)'\s*,\s*"
    r"'([^']*)'\s*,\s*"
    r"'([^']*)'\s*"
    r"\)",
    re.IGNORECASE,
)


def normalize_ka_number(ka: str) -> str:
    """Accept 'KA-02130', 'ka-02130', '02130' -> 'KA-02130'."""
    s = ka.strip().upper()
    if re.fullmatch(r"\d+", s):
        return f"KA-{s}"
    if re.fullmatch(r"KA-?\d+", s.replace(" ", "")):
        num = re.sub(r"^KA-?", "", s.replace(" ", ""))
        return f"KA-{num}"
    if re.fullmatch(r"KA-\d+", s):
        return s
    raise ValueError(f"Unrecognized KA number format: {ka!r}")


def article_url(ka_number: str, base: str = PORTAL_BASE) -> str:
    n = normalize_ka_number(ka_number)
    return urljoin(base.rstrip("/") + "/", f"article/?kanumber={n}")


def service_request_create_url(caid: str, kasid: str, base: str = PORTAL_BASE) -> str:
    q = urlencode({"caid": caid, "kasid": kasid})
    return urljoin(base.rstrip("/") + "/", f"servicerequest-create/?{q}")


def fetch_article_html(
    ka_number: str,
    *,
    base: str = PORTAL_BASE,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_UA,
) -> str:
    url = article_url(ka_number, base=base)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(
            resp.headers.get_content_charset() or "utf-8", errors="replace"
        )


@dataclass
class ServiceRequestLink:
    """A 'Report …' (or similar) action that opens the SR create flow."""

    label: str
    url: str
    section_heading: str | None
    content_action_id: str
    ka_section_id: str
    agency: str
    service_request_type_id: str = ""
    knowledge_article_id: str = ""
    step_id: str = ""


@dataclass
class KAArticleOutline:
    ka_number: str
    title: str
    article_url: str
    headings: list[str] = field(default_factory=list)
    service_request_links: list[ServiceRequestLink] = field(default_factory=list)


def _card_header_text(card) -> str | None:
    h = card.select_one(".card-header h4")
    if not h:
        return None
    btn = h.select_one("button .col-md-11, button .col-lg-11")
    if btn:
        return btn.get_text(" ", strip=True)
    return h.get_text(" ", strip=True) or None


def _parse_create_service_request_calls(
    onclick: str,
) -> list[tuple[str, str, str, str, str, str]]:
    return _CREATE_SR_RE.findall(onclick or "")


def parse_ka_article_html(
    html: str,
    *,
    ka_number: str,
    base: str = PORTAL_BASE,
) -> KAArticleOutline:
    soup = BeautifulSoup(html, "html.parser")
    n = normalize_ka_number(ka_number)

    h1 = soup.select_one("h1.entry-title.page-title")
    title = h1.get_text(strip=True) if h1 else ""

    container = soup.select_one("#knowledgearticle .ka-container") or soup.select_one(
        ".ka-container"
    )
    headings: list[str] = []
    links: list[ServiceRequestLink] = []

    if not container:
        return KAArticleOutline(
            ka_number=n,
            title=title,
            article_url=article_url(n, base=base),
            headings=headings,
            service_request_links=links,
        )

    for card in container.select(".accordion .card"):
        section = _card_header_text(card)
        if section and section not in headings:
            headings.append(section)

        body = card.select_one(".card-body")
        if not body:
            continue
        for a in body.select("a.contentaction[onclick]"):
            oc = a.get("onclick") or ""
            for match in _parse_create_service_request_calls(oc):
                sr_type, caid, kaid, step_id, agency, kasid = match
                label = a.get_text(" ", strip=True)
                if not caid or not kasid:
                    continue
                url = service_request_create_url(caid, kasid, base=base)
                links.append(
                    ServiceRequestLink(
                        label=label,
                        url=url,
                        section_heading=section,
                        content_action_id=caid,
                        ka_section_id=kasid,
                        agency=agency,
                        service_request_type_id=sr_type,
                        knowledge_article_id=kaid,
                        step_id=step_id,
                    )
                )

    return KAArticleOutline(
        ka_number=n,
        title=title,
        article_url=article_url(n, base=base),
        headings=headings,
        service_request_links=links,
    )


def fetch_ka_outline(
    ka_number: str,
    *,
    base: str = PORTAL_BASE,
    timeout: float = 30.0,
) -> KAArticleOutline:
    html = fetch_article_html(ka_number, base=base, timeout=timeout)
    return parse_ka_article_html(html, ka_number=ka_number, base=base)


def outline_to_dict(outline: KAArticleOutline) -> dict:
    return {
        "ka_number": outline.ka_number,
        "title": outline.title,
        "article_url": outline.article_url,
        "headings": outline.headings,
        "service_request_links": [
            {
                "label": L.label,
                "url": L.url,
                "section_heading": L.section_heading,
                "agency": L.agency,
                "content_action_id": L.content_action_id,
                "ka_section_id": L.ka_section_id,
            }
            for L in outline.service_request_links
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fetch an NYC 311 knowledge article by KA number and print headings "
        "and servicerequest-create URLs parsed from the page."
    )
    p.add_argument(
        "ka",
        help="Knowledge article id, e.g. KA-02130 or 02130",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of plain text",
    )
    args = p.parse_args(argv)

    try:
        outline = fetch_ka_outline(args.ka)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e.reason}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(outline_to_dict(outline), indent=2))
        return 0

    print(f"KA:        {outline.ka_number}")
    print(f"Title:     {outline.title}")
    print(f"Article:   {outline.article_url}")
    print()
    print("Headings (accordion sections):")
    for h in outline.headings:
        print(f"  - {h}")
    print()
    print("Service request links:")
    if not outline.service_request_links:
        print("  (none found on this page)")
    for L in outline.service_request_links:
        where = f" [{L.section_heading}]" if L.section_heading else ""
        print(f"  - {L.label}{where}")
        print(f"      {L.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())