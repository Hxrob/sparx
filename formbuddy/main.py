from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from config import ALLOWED_PROXY_HOST
from llm_client import chat_completion
from models import SuggestRequest
from proxy import proxy_prefix, proxy_request
from session_store import create_session, get_session

app = FastAPI(title="FormBuddy")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Page routes ---


@app.get("/", response_class=HTMLResponse)
async def start_page():
    return FileResponse(STATIC_DIR / "start.html")


@app.post("/start")
async def handle_start(transcript: str = Form(...), report_url: str = Form(...)):
    # Validate URL host
    parsed = urlparse(report_url)
    if parsed.hostname != ALLOWED_PROXY_HOST:
        raise HTTPException(400, f"URL must be on {ALLOWED_PROXY_HOST}")

    session = create_session(transcript, report_url)
    return RedirectResponse(f"/buddy/{session.id}", status_code=303)


@app.get("/buddy/{session_id}", response_class=HTMLResponse)
async def buddy_page(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return FileResponse(STATIC_DIR / "buddy.html")


# --- API routes ---


@app.get("/api/sessions/{session_id}")
async def session_info(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session.id,
        "proxy_url": f"{proxy_prefix(session.id)}{session.proxy_path()}",
        "has_last_suggestion": session.last_suggestion is not None,
        "known_facts": session.known_facts,
    }


@app.post("/api/sessions/{session_id}/suggest")
async def suggest(session_id: str, req: SuggestRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Build the user message with current context
    fields_desc = []
    for f in req.fields:
        desc = f"- {f.field_id} (label: {f.label!r}, type: {f.type}"
        if f.required:
            desc += ", required"
        if f.options:
            desc += f", options: {f.options}"
        if f.current_value:
            desc += f", current: {f.current_value!r}"
        if f.placeholder:
            desc += f", placeholder: {f.placeholder!r}"
        desc += ")"
        fields_desc.append(desc)

    user_content = f"""Page: {req.page_context.title or 'Unknown'}
URL: {req.page_context.url or 'Unknown'}

Transcript:
{session.transcript}

Known facts from previous steps:
{session.known_facts or 'None yet'}

Currently visible form fields:
{chr(10).join(fields_desc) or 'No fields found'}

Fill in the fields based on the transcript and known facts. Return JSON only."""

    session.messages.append({"role": "user", "content": user_content})

    result = await chat_completion(session.messages)

    # Update session state
    session.messages.append(
        {"role": "assistant", "content": result.model_dump_json()}
    )
    session.known_facts.update(result.known_facts)
    session.last_suggestion = result.model_dump()

    return result


# --- Proxy route ---


@app.api_route(
    "/s/{session_id}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(session_id: str, path: str, request: Request):
    session = get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return await proxy_request(request, session, path)
