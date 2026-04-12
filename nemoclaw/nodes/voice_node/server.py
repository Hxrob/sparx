"""
Voice Node Server — Parakeet ASR + Direction Engine + NemoClaw.

Flow:
  Phone records audio
    -> POST /transcribe
    -> Parakeet transcribes to text
    -> Direction Engine categorizes via OpenAI-compatible LLM (same as FormFinder)
    -> NemoClaw resolves: Open Data first, then FormFinder fallback
    -> Returns transcription + direction + resolution
    -> Appends to complaints.log if categorized

One server, port 8443, HTTPS (required for phone mic access).
"""
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from typing import Optional
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import uvicorn

from asr_engine import VoiceNode
import direction_engine
import emotion_detector
import document_intake
from storage import Storage
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "displacement_node")))
import displacement_detector

LOGGER = logging.getLogger("server")

# Add nemoclaw parent to path so we can import it
_NEMOCLAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NEMOCLAW_DIR not in sys.path:
    sys.path.insert(0, _NEMOCLAW_DIR)
from nemoclaw import NemoClaw

# Add formbuddy to path for proxy + session imports
_FORMBUDDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "formbuddy"))
if _FORMBUDDY_DIR not in sys.path:
    sys.path.insert(0, _FORMBUDDY_DIR)
from session_store import create_session as fb_create_session, get_session as fb_get_session
from proxy import proxy_request as fb_proxy_request
from models import SuggestRequest as FBSuggestRequest, SuggestDebugResponse as FBSuggestDebugResponse
from llm_client import chat_completion as fb_chat_completion

voice_node: VoiceNode = None
nemoclaw: NemoClaw = None
db: Storage = None

NON_ENGLISH_THRESHOLD = 0.25  # 25% non-English words triggers translation

_DETECT_SYSTEM = """\
You are a language analyst. Given a transcript, determine what percentage of the words \
are in a language other than English and identify that language.

Output ONLY a JSON object with these keys:
- "non_english_ratio": a float between 0.0 and 1.0 representing the fraction of non-English words
- "language": the ISO 639-1 code of the dominant non-English language (e.g. "es" for Spanish, "zh" for Chinese), or "en" if the text is entirely English
- "language_name": the full name of that language (e.g. "Spanish", "Chinese")

The first character of your reply MUST be { and the last MUST be }. No markdown, no prose.
"""

_TRANSLATE_SYSTEM = """\
You are a professional translator. Translate the following text into {language_name}. \
Preserve the original meaning, tone, and any proper nouns (like form names or URLs). \
Output ONLY the translated text, nothing else.
"""


async def _detect_language(transcript: str) -> dict:
    """Use the LLM to detect what fraction of the transcript is non-English."""
    client = direction_engine._get_client()
    payload = {
        "messages": [
            {"role": "system", "content": _DETECT_SYSTEM},
            {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"},
        ],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    model = os.environ.get("SPARX_LLM_MODEL")
    if model:
        payload["model"] = model

    try:
        resp = await client.post(direction_engine.LLM_CHAT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Parse JSON from response
        parsed = direction_engine._parse_model_json(content)
        if parsed and "non_english_ratio" in parsed:
            return parsed
    except Exception as exc:
        LOGGER.error("Language detection failed: %s", exc)

    return {"non_english_ratio": 0.0, "language": "en", "language_name": "English"}


async def _translate_text(text: str, language_name: str) -> str:
    """Translate text into the target language using the LLM."""
    if not text or (isinstance(text, str) and not text.strip()):
        return text

    client = direction_engine._get_client()
    payload = {
        "messages": [
            {"role": "system", "content": _TRANSLATE_SYSTEM.format(language_name=language_name)},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    model = os.environ.get("SPARX_LLM_MODEL")
    if model:
        payload["model"] = model

    try:
        resp = await client.post(direction_engine.LLM_CHAT_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        translated = data["choices"][0]["message"]["content"].strip()
        if translated:
            return translated
    except Exception as exc:
        LOGGER.error("Translation to %s failed: %s", language_name, exc)

    return text  # fallback to original on failure


@asynccontextmanager
async def lifespan(app: FastAPI):
    global voice_node, nemoclaw, db
    print("Loading Parakeet model onto GPU...")
    voice_node = VoiceNode()
    print("Parakeet ready.")
    nemoclaw = NemoClaw()
    print(f"NemoClaw ready ({nemoclaw._finder.entry_count} KA entries loaded).")
    db = Storage()
    print("Storage ready.")
    yield
    await direction_engine.close_client()


app = FastAPI(lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# FormBuddy static files (buddy.html, buddy.js, styles.css)
_fb_static = os.path.join(_FORMBUDDY_DIR, "static")
if os.path.isdir(_fb_static):
    app.mount("/fb-static", StaticFiles(directory=_fb_static), name="fb-static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(static_dir, "index.html"), "r") as f:
        return f.read()


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    print(f"Received audio: {file.filename} | session: {session_id}")

    # Save and convert to 16kHz mono WAV (Parakeet requirement)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        in_path = tmp.name

    out_path = in_path.replace(".webm", ".wav")
    transcript = ""

    emotion_result = emotion_detector.EmotionResult(distress_score=0.0, emotion="neutral")

    try:
        audio = AudioSegment.from_file(in_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format="wav")

        # Analyze emotion from raw audio BEFORE transcription — catches what words miss
        emotion_result = emotion_detector.analyze(out_path)
        if emotion_result.is_distressed:
            print(f"Emotion: {emotion_result.emotion} (score={emotion_result.distress_score:.2f}) "
                  f"markers={emotion_result.markers}")

        transcript = voice_node.transcribe_wav(out_path)
    finally:
        if os.path.exists(in_path):  os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)

    print(f"Transcript: {transcript}")

    # Run direction engine — categorize and summarize, informed by vocal emotion
    result = await direction_engine.process(
        transcript, session_id=session_id, emotion=emotion_result
    )

    print(f"Direction: decision={result.decision.value} | "
          f"categories={[c.value for c in result.categories]} | "
          f"confidence={result.confidence:.0%}")

    direction = {
        "decision":     result.decision.value,
        "categories":   [c.value for c in result.categories],
        "question":     result.question,
        "summary":      result.summary,
        "response":     result.response,
        "confidence":   result.confidence,
        "missing_info": result.missing_info,
        "transcript":   transcript,
    }

    # Run NemoClaw and displacement detector in parallel — zero added latency
    resolution_task    = asyncio.create_task(nemoclaw.handle(direction))
    displacement_task  = asyncio.create_task(
        asyncio.to_thread(displacement_detector.score, transcript)
    )
    resolution   = await resolution_task
    displacement = await displacement_task

    if displacement.should_alert:
        print(f"Displacement alert [{displacement.risk_level}] at '{displacement.address}': "
              f"score={displacement.risk_score}")
    print(f"NemoClaw: source={resolution.source} | "
          f"open_data={'yes' if resolution.open_data else 'no'} | "
          f"form_finder={'yes' if resolution.form_finder else 'no'}")

    # Detect language — if ≥25% non-English, translate user-facing text
    lang_info = await _detect_language(transcript)
    detected_lang = lang_info.get("language", "en")
    detected_name = lang_info.get("language_name", "English")
    ratio = lang_info.get("non_english_ratio", 0.0)
    translated = False

    if detected_lang != "en" and ratio >= NON_ENGLISH_THRESHOLD:
        print(f"Language: {detected_name} ({ratio:.0%}) — translating responses")

        # Translate the direction engine's spoken response
        direction["response"] = await _translate_text(direction["response"], detected_name)

        # Translate NemoClaw's resolution response
        resolution.response = await _translate_text(resolution.response, detected_name)

        # Translate form_finder next_steps if present (it's a list of strings)
        if resolution.form_finder and resolution.form_finder.get("next_steps"):
            steps = resolution.form_finder["next_steps"]
            if isinstance(steps, list):
                resolution.form_finder["next_steps"] = [
                    await _translate_text(s, detected_name) for s in steps if s
                ]
            else:
                resolution.form_finder["next_steps"] = await _translate_text(
                    steps, detected_name
                )

        # Translate open_data answer if present
        if resolution.open_data and resolution.open_data.get("answer"):
            resolution.open_data["answer"] = await _translate_text(
                resolution.open_data["answer"], detected_name
            )

        translated = True

    # If NemoClaw found a 311 form, auto-create a FormBuddy session
    buddy_url = None
    if resolution.form_finder and resolution.form_finder.get("form_url"):
        try:
            fb_session = fb_create_session(transcript, resolution.form_finder["form_url"])
            buddy_url = f"/buddy/{fb_session.id}"
            LOGGER.info("FormBuddy session created: %s -> %s", fb_session.id, resolution.form_finder["form_url"])
        except Exception as exc:
            LOGGER.error("FormBuddy session creation failed: %s", exc)

    response_payload = {
        "transcription": transcript,
        "direction": direction,
        "resolution": {
            "source":       resolution.source,
            "response":     resolution.response,
            "open_data":    resolution.open_data,
            "form_finder":  resolution.form_finder,
            "buddy_url":    buddy_url,
        },
        "language": {
            "detected":   detected_name,
            "code":       detected_lang,
            "ratio":      round(ratio, 2),
            "translated": translated,
        },
        "emotion": {
            "emotion":        emotion_result.emotion,
            "distress_score": round(emotion_result.distress_score, 2),
            "markers":        emotion_result.markers,
            "arousal":        round(emotion_result.arousal, 2),
            "is_crisis":      emotion_result.is_crisis,
        },
        "displacement": {
            "address":       displacement.address,
            "risk_level":    displacement.risk_level,
            "risk_score":    displacement.risk_score,
            "alert":         displacement.alert_message,
            "signals": {
                "violations": displacement.signals.get("violations", {}).get("count", 0),
                "class_c":    displacement.signals.get("violations", {}).get("class_c", 0),
                "evictions":  displacement.signals.get("evictions", {}).get("count", 0),
                "corporate_owner": displacement.signals.get("ownership", {}).get("corporate", False),
                "recent_sale":     displacement.signals.get("ownership", {}).get("recent_change", False),
                "owner":           displacement.signals.get("ownership", {}).get("owner", ""),
            },
        },
    }

    # Persist — runs in background thread so it doesn't block the response
    asyncio.create_task(asyncio.to_thread(db.save_call, session_id, response_payload))

    return response_payload


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Accept an image (JPG/PNG/WebP) or PDF, extract structured information,
    route through direction engine + NemoClaw, persist to storage.
    """
    print(f"Received document: {file.filename} ({file.content_type}) | session: {session_id}")

    file_bytes = await file.read()

    # Extract structured info from the document
    doc = await document_intake.analyze(file_bytes, file.filename or "", file.content_type or "")

    if doc.error and not doc.situation:
        return {
            "error": doc.error,
            "document": {
                "document_type": doc.document_type,
                "urgency":       doc.urgency,
                "summary":       doc.summary,
            },
        }

    print(f"Document: type={doc.document_type} urgency={doc.urgency} "
          f"deadline_days={doc.deadline_days}")

    # Use the extracted situation as the transcript into direction engine + NemoClaw
    situation_text = doc.situation or doc.summary

    result = await direction_engine.process(situation_text, session_id=session_id)

    direction = {
        "decision":     result.decision.value,
        "categories":   [c.value for c in result.categories],
        "question":     result.question,
        "summary":      result.summary,
        "response":     result.response,
        "confidence":   result.confidence,
        "missing_info": result.missing_info,
        "transcript":   situation_text,
    }

    # Run NemoClaw and displacement detector in parallel
    resolution_task   = asyncio.create_task(nemoclaw.handle(direction))
    displacement_task = asyncio.create_task(
        asyncio.to_thread(displacement_detector.score, doc.address or situation_text)
    )
    resolution   = await resolution_task
    displacement = await displacement_task

    # Auto-create FormBuddy session for document uploads too
    doc_buddy_url = None
    if resolution.form_finder and resolution.form_finder.get("form_url"):
        try:
            fb_sess = fb_create_session(situation_text, resolution.form_finder["form_url"])
            doc_buddy_url = f"/buddy/{fb_sess.id}"
        except Exception as exc:
            LOGGER.error("FormBuddy session creation failed (upload): %s", exc)

    response_payload = {
        "document": {
            "document_type":    doc.document_type,
            "urgency":          doc.urgency,
            "summary":          doc.summary,
            "situation":        doc.situation,
            "key_dates":        doc.key_dates,
            "amounts_owed":     doc.amounts_owed,
            "address":          doc.address,
            "case_number":      doc.case_number,
            "issuing_agency":   doc.issuing_agency,
            "required_actions": doc.required_actions,
            "deadline_days":    doc.deadline_days,
        },
        "direction": direction,
        "resolution": {
            "source":      resolution.source,
            "response":    resolution.response,
            "open_data":   resolution.open_data,
            "form_finder": resolution.form_finder,
            "buddy_url":   doc_buddy_url,
        },
        "displacement": {
            "address":    displacement.address,
            "risk_level": displacement.risk_level,
            "risk_score": displacement.risk_score,
            "alert":      displacement.alert_message,
        },
    }

    asyncio.create_task(asyncio.to_thread(db.save_call, session_id, {
        "transcription": situation_text,
        "direction":     direction,
        "resolution":    response_payload["resolution"],
        "language":      {"code": "en", "detected": "English", "ratio": 0.0, "translated": False},
        "emotion":       {"emotion": "neutral", "distress_score": 0.0, "markers": [], "is_crisis": False},
        "displacement":  response_payload["displacement"],
    }))

    return response_payload


@app.get("/history/{session_id}")
async def get_session_history(session_id: str):
    return {"session_id": session_id, "calls": db.session_history(session_id)}


@app.get("/address/{address}")
async def get_address_history(address: str):
    return {"address": address, "calls": db.address_history(address)}


@app.get("/stats")
async def get_stats():
    return db.stats()


@app.get("/crises")
async def get_recent_crises(hours: int = 24):
    return {"hours": hours, "crises": db.recent_crises(hours)}


# ---------------------------------------------------------------------------
# FormBuddy integration — serve buddy page, proxy, and suggest API
# ---------------------------------------------------------------------------


@app.get("/buddy/{session_id}", response_class=HTMLResponse)
async def buddy_page(session_id: str):
    session = fb_get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    # Serve buddy.html with paths rewritten to /fb-static/
    with open(os.path.join(_fb_static, "buddy.html"), "r") as f:
        html = f.read()
    html = html.replace('"/static/', '"/fb-static/')
    html = html.replace("'/static/", "'/fb-static/")
    return HTMLResponse(html)


@app.get("/api/sessions/{session_id}")
async def fb_session_info(session_id: str):
    from proxy import proxy_prefix
    session = fb_get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session.id,
        "proxy_url": f"{proxy_prefix(session.id)}{session.proxy_path()}",
        "has_last_suggestion": session.last_suggestion is not None,
        "known_facts": session.known_facts,
    }


@app.post("/api/sessions/{session_id}/suggest")
async def fb_suggest(session_id: str, req: FBSuggestRequest):
    session = fb_get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

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
    result, debug = await fb_chat_completion(session.messages)

    session.messages.append(
        {"role": "assistant", "content": result.model_dump_json()}
    )
    session.known_facts.update(result.known_facts)
    session.last_suggestion = result.model_dump()

    return FBSuggestDebugResponse(**result.model_dump(), debug=debug)


@app.api_route(
    "/s/{session_id}/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def fb_proxy(session_id: str, path: str, request: Request):
    session = fb_get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return await fb_proxy_request(request, session, path)


import re as _re

@app.api_route(
    "/_portal/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def fb_portal_catchall(path: str, request: Request):
    referer = request.headers.get("referer", "")
    m = _re.search(r"/s/([^/]+)/proxy/", referer)
    if not m:
        raise HTTPException(404, "Session not found")
    session_id = m.group(1)
    session = fb_get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return await fb_proxy_request(request, session, f"_portal/{path}")


if __name__ == "__main__":
    key_file  = "key.pem"
    cert_file = "cert.pem"

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("Generating self-signed SSL certificates...")
        os.system(
            f'openssl req -x509 -nodes -days 365 -newkey rsa:2048 '
            f'-keyout {key_file} -out {cert_file} '
            f'-subj "/C=US/ST=NY/L=NYC/O=SparX/CN=localhost"'
        )

    print("\n" + "=" * 50)
    print("SparX Voice Node live!")
    print(f"Phone:  https://<DGX_IP>:8443")
    print(f"Logs:   {direction_engine.LOG_PATH}")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8443, ssl_keyfile=key_file, ssl_certfile=cert_file)
