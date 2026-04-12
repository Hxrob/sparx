import uuid

import httpx

from config import SYSTEM_PROMPT


class Session:
    def __init__(self, transcript: str, report_url: str):
        self.id = str(uuid.uuid4())
        self.transcript = transcript
        self.report_url = report_url
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.known_facts: dict[str, str] = {}
        self.last_suggestion: dict | None = None
        self.http_client = httpx.AsyncClient(
            follow_redirects=False,
            timeout=30.0,
            cookies=httpx.Cookies(),
        )

    def proxy_path(self) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(self.report_url)
        path = parsed.path
        if parsed.query:
            path += f"?{parsed.query}"
        return path


_sessions: dict[str, "Session"] = {}


def create_session(transcript: str, report_url: str) -> Session:
    session = Session(transcript, report_url)
    _sessions[session.id] = session
    return session


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)
