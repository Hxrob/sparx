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
import os
import sys
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import uvicorn

from asr_engine import VoiceNode
import direction_engine

# Add nemoclaw parent to path so we can import it
_NEMOCLAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NEMOCLAW_DIR not in sys.path:
    sys.path.insert(0, _NEMOCLAW_DIR)
from nemoclaw import NemoClaw

voice_node: VoiceNode = None
nemoclaw: NemoClaw = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global voice_node, nemoclaw
    print("Loading Parakeet model onto GPU...")
    voice_node = VoiceNode()
    print("Parakeet ready.")
    nemoclaw = NemoClaw()
    print(f"NemoClaw ready ({nemoclaw._finder.entry_count} KA entries loaded).")
    yield
    await direction_engine.close_client()


app = FastAPI(lifespan=lifespan)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


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

    try:
        audio = AudioSegment.from_file(in_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out_path, format="wav")
        transcript = voice_node.transcribe_wav(out_path)
    finally:
        if os.path.exists(in_path):  os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)

    print(f"Transcript: {transcript}")

    # Run direction engine — categorize and summarize
    result = await direction_engine.process(transcript, session_id=session_id)

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

    # Run NemoClaw — Open Data lookup, then FormFinder fallback
    resolution = await nemoclaw.handle(direction)

    return {
        "transcription": transcript,
        "direction": direction,
        "resolution": {
            "source":       resolution.source,
            "response":     resolution.response,
            "open_data":    resolution.open_data,
            "form_finder":  resolution.form_finder,
        },
    }


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
