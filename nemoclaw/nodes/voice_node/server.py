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
import json
import logging
import os
import re
import sys
import tempfile
from contextlib import asynccontextmanager
import base64

from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional
from fastapi.responses import HTMLResponse
from TTS.api import TTS
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import uvicorn

from asr_engine import VoiceNode
import direction_engine

LOGGER = logging.getLogger("server")

# Add nemoclaw parent to path so we can import it
_NEMOCLAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NEMOCLAW_DIR not in sys.path:
    sys.path.insert(0, _NEMOCLAW_DIR)
from nemoclaw import NemoClaw

voice_node: VoiceNode = None
nemoclaw: NemoClaw = None
tts_node: TTS = None

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
    global voice_node, nemoclaw, tts_node
    print("Loading Parakeet model onto GPU...")
    voice_node = VoiceNode()
    print("Parakeet ready.")
    print("Loading Coqui XTTS v2 onto GPU...")
    tts_node = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print("XTTS ready.")
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
        # Ensure we keep out_path so Coqui can use it to clone the speaker's voice!

    print(f"Transcript: {transcript}")

    # Run direction engine — categorize and summarize
    result = await direction_engine.process(transcript, session_id=session_id)

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

    # Run NemoClaw — Open Data lookup, then FormFinder fallback
    resolution = await nemoclaw.handle(direction)
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

    # ---- XTTS ZERO-SHOT CLONING ----
    audio_b64 = ""
    reply_text = resolution.response
    
    xtts_lang = detected_lang
    if xtts_lang == "zh":
        xtts_lang = "zh-cn"
    supported_xtts = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]
    if xtts_lang not in supported_xtts:
        xtts_lang = "en"

    if reply_text and os.path.exists(out_path):
        try:
            print(f"Synthesizing XTTS audio in '{xtts_lang}' using {out_path} as speaker reference...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_tts:
                tmp_tts_path = tmp_tts.name
            
            # Make the bot speak back in the user's cloned voice!
            tts_node.tts_to_file(
                text=reply_text, 
                speaker_wav=out_path, 
                language=xtts_lang, 
                file_path=tmp_tts_path
            )
            
            with open(tmp_tts_path, "rb") as audio_file:
                audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")
                
            os.remove(tmp_tts_path)
        except Exception as e:
            print(f"XTTS synthesis failed: {e}")

    # Finally cleanup the original user wav file that we passed to XTTS
    if os.path.exists(out_path): os.remove(out_path)

    return {
        "transcription": transcript,
        "direction": direction,
        "resolution": {
            "source":       resolution.source,
            "response":     resolution.response,
            "open_data":    resolution.open_data,
            "form_finder":  resolution.form_finder,
        },
        "language": {
            "detected":   detected_name,
            "code":       detected_lang,
            "ratio":      round(ratio, 2),
            "translated": translated,
        },
        "audio_base64": audio_b64,
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
