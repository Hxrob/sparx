import re
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import Response

from config import ALLOWED_PROXY_HOST
from session_store import Session

PROXY_TARGET = f"https://{ALLOWED_PROXY_HOST}"

# Headers to strip from proxied responses (prevent frame-blocking)
STRIP_RESPONSE_HEADERS = {
    "x-frame-options",
    "content-security-policy",
    "content-security-policy-report-only",
}

# Headers to not forward from browser to target
SKIP_REQUEST_HEADERS = {
    "host",
    "origin",
    "referer",
    "cookie",
    "connection",
    "transfer-encoding",
    "content-length",
}

# Hosts that should NOT be proxied (external CDNs, etc.)
EXTERNAL_HOSTS = {
    "translate.google.com",
    "content.powerapps.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdn.jsdelivr.net",
    "cdnjs.cloudflare.com",
    "ajax.googleapis.com",
    "code.jquery.com",
    "us-mobile.events.data.microsoft.com",
}


def proxy_prefix(session_id: str) -> str:
    return f"/s/{session_id}/proxy"


def _is_external_url(url: str) -> bool:
    """Check if a URL points to a known external host that shouldn't be proxied."""
    try:
        parsed = urlparse(url)
        if parsed.hostname and parsed.hostname != ALLOWED_PROXY_HOST:
            return True
    except Exception:
        pass
    return False


def rewrite_url_in_content(content: str, session_id: str) -> str:
    """Rewrite URLs in HTML/CSS content to go through the proxy."""
    prefix = proxy_prefix(session_id)

    # Protect protocol-relative external URLs BEFORE blanket host replacement.
    # Only replace //host when NOT preceded by a colon (i.e. not already https://host).
    for host in EXTERNAL_HOSTS:
        content = re.sub(rf'(?<!:)//{re.escape(host)}', f'https://{host}', content)

    # Full URLs to the target host only
    content = content.replace(f"https://{ALLOWED_PROXY_HOST}", prefix)
    content = content.replace(f"http://{ALLOWED_PROXY_HOST}", prefix)
    # Protocol-relative for the target host
    content = content.replace(f"//{ALLOWED_PROXY_HOST}", prefix)

    # Rewrite absolute paths in href, src, action, data-url attributes
    def rewrite_attr(m: re.Match) -> str:
        attr = m.group(1)
        path = m.group(2)
        if path.startswith(prefix):
            return m.group(0)
        # Don't rewrite paths to our own app routes
        if path.startswith("/s/") or path.startswith("/api/") or path.startswith("/static/") or path.startswith("/buddy/"):
            return m.group(0)
        return f'{attr}="{prefix}{path}"'

    content = re.sub(
        r'(href|src|action|data-url)="(/[^"]*)"',
        rewrite_attr,
        content,
    )

    # Also handle single-quoted attributes
    def rewrite_attr_sq(m: re.Match) -> str:
        attr = m.group(1)
        path = m.group(2)
        if path.startswith(prefix):
            return m.group(0)
        if path.startswith("/s/") or path.startswith("/api/") or path.startswith("/static/") or path.startswith("/buddy/"):
            return m.group(0)
        return f"{attr}='{prefix}{path}'"

    content = re.sub(
        r"(href|src|action|data-url)='(/[^']*)'",
        rewrite_attr_sq,
        content,
    )

    # Rewrite CSS url() references with absolute paths
    def rewrite_css_url(m: re.Match) -> str:
        path = m.group(1)
        if path.startswith(prefix) or path.startswith("data:") or path.startswith("http"):
            return m.group(0)
        if path.startswith("/"):
            return f"url({prefix}{path})"
        return m.group(0)

    content = re.sub(r"url\(([^)]+)\)", rewrite_css_url, content)

    # Rewrite ASP.NET/PowerPages ~/path patterns to absolute proxy paths
    content = re.sub(
        r'(["\'])~/',
        lambda m: f"{m.group(1)}{prefix}/",
        content,
    )

    return content


def make_intercept_script(session_id: str) -> str:
    """JS to inject into proxied pages to intercept fetch/XHR/navigation."""
    prefix = proxy_prefix(session_id)
    return f"""<script data-formbuddy="intercept">
(function() {{
    var P = '{prefix}';
    var H = '{ALLOWED_PROXY_HOST}';
    var EXT = {{}};
    {';'.join(f'EXT["{h}"]=1' for h in EXTERNAL_HOSTS)};
    function isExt(h) {{ return !!EXT[h]; }}
    function rw(u) {{
        if (!u || typeof u !== 'string') return u;
        // Handle ~/ ASP.NET paths
        if (u.startsWith('~/')) return P + u.slice(1);
        try {{
            var a = new URL(u, location.origin);
            // Don't proxy external hosts
            if (a.hostname !== H && a.hostname !== location.hostname) return u;
            if (a.hostname === H) return P + a.pathname + a.search + a.hash;
            if (a.origin === location.origin && a.pathname.startsWith('/') &&
                !a.pathname.startsWith(P) && !a.pathname.startsWith('/s/') &&
                !a.pathname.startsWith('/api/') && !a.pathname.startsWith('/static/') &&
                !a.pathname.startsWith('/buddy/'))
                return P + a.pathname + a.search + a.hash;
        }} catch(e) {{
            if (u.startsWith('/') && !u.startsWith(P) && !u.startsWith('/s/') &&
                !u.startsWith('/api/') && !u.startsWith('/static/'))
                return P + u;
        }}
        return u;
    }}
    // fetch
    var _f = window.fetch;
    window.fetch = function(i, o) {{
        if (typeof i === 'string') i = rw(i);
        else if (i && i.url) i = new Request(rw(i.url), i);
        return _f.call(this, i, o);
    }};
    // XHR
    var _o = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(m, u) {{
        arguments[1] = rw(u);
        return _o.apply(this, arguments);
    }};
    // location.assign / location.replace
    try {{
        var _a = location.assign.bind(location);
        location.assign = function(u) {{ _a(rw(u)); }};
        var _r = location.replace.bind(location);
        location.replace = function(u) {{ _r(rw(u)); }};
    }} catch(e) {{}}
    // Intercept window.location.href setter via navigation
    var origPushState = history.pushState;
    history.pushState = function() {{
        if (typeof arguments[2] === 'string') arguments[2] = rw(arguments[2]);
        return origPushState.apply(this, arguments);
    }};
    var origReplaceState = history.replaceState;
    history.replaceState = function() {{
        if (typeof arguments[2] === 'string') arguments[2] = rw(arguments[2]);
        return origReplaceState.apply(this, arguments);
    }};
    // form submit action rewrite
    document.addEventListener('submit', function(e) {{
        var f = e.target;
        if (f && f.action) f.action = rw(f.action);
    }}, true);
    // click handler for links
    document.addEventListener('click', function(e) {{
        var a = e.target.closest('a[href]');
        if (a && a.href) {{
            var h = a.getAttribute('href');
            var rh = rw(h);
            if (rh !== h) a.setAttribute('href', rh);
        }}
    }}, true);
}})();
</script>"""


async def proxy_request(
    request: Request,
    session: Session,
    path: str,
) -> Response:
    """Forward a request to the target host through the session's HTTP client."""
    query = str(request.url.query)
    target_url = f"{PROXY_TARGET}/{path}"
    if query:
        target_url += f"?{query}"

    # Build headers to forward
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in SKIP_REQUEST_HEADERS:
            headers[key] = value
    headers["host"] = ALLOWED_PROXY_HOST
    headers["referer"] = f"{PROXY_TARGET}/"
    headers["accept-encoding"] = "identity"

    body = await request.body()

    resp = await session.http_client.request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body if body else None,
    )

    # Build response headers
    response_headers = {}
    for key, value in resp.headers.multi_items():
        if key.lower() in STRIP_RESPONSE_HEADERS:
            continue
        if key.lower() == "location":
            value = rewrite_redirect(value, session.id)
        if key.lower() in ("transfer-encoding", "content-encoding", "content-length"):
            continue
        response_headers[key] = value

    content_type = resp.headers.get("content-type", "")
    body_bytes = resp.content

    if "text/html" in content_type:
        text = body_bytes.decode("utf-8", errors="replace")
        text = rewrite_url_in_content(text, session.id)
        script = make_intercept_script(session.id)
        # Inject intercept script as early as possible in <head>
        head_pattern = re.compile(r"(<head[^>]*>)", re.IGNORECASE)
        m = head_pattern.search(text)
        if m:
            text = text[: m.end()] + script + text[m.end() :]
        else:
            text = script + text
        body_bytes = text.encode("utf-8")
    elif "text/css" in content_type:
        text = body_bytes.decode("utf-8", errors="replace")
        text = rewrite_url_in_content(text, session.id)
        body_bytes = text.encode("utf-8")
    elif "javascript" in content_type or "application/x-javascript" in content_type:
        # Rewrite JS string literals containing the target host or ~/ paths
        text = body_bytes.decode("utf-8", errors="replace")
        prefix = proxy_prefix(session.id)
        text = text.replace(f"https://{ALLOWED_PROXY_HOST}", prefix)
        text = text.replace(f"http://{ALLOWED_PROXY_HOST}", prefix)
        # Rewrite ~/ patterns in JS string contexts
        text = re.sub(r'(["\'])~/', lambda m: f"{m.group(1)}{prefix}/", text)
        body_bytes = text.encode("utf-8")

    return Response(
        content=body_bytes,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=content_type.split(";")[0].strip() if content_type else None,
    )


def rewrite_redirect(location: str, session_id: str) -> str:
    """Rewrite a redirect Location header."""
    prefix = proxy_prefix(session_id)
    if location.startswith(f"https://{ALLOWED_PROXY_HOST}"):
        return prefix + location[len(f"https://{ALLOWED_PROXY_HOST}"):]
    if location.startswith(f"http://{ALLOWED_PROXY_HOST}"):
        return prefix + location[len(f"http://{ALLOWED_PROXY_HOST}"):]
    if location.startswith("/") and not location.startswith(prefix):
        return prefix + location
    return location
