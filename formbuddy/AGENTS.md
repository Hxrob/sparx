# FormBuddy

Local dev tool that helps fill NYC 311 complaint forms using a local LLM. A user pastes a transcript and a 311 form URL, then iterates through a multi-step form with LLM-assisted autofill.

## How it works

1. User submits a transcript + 311 URL on the start page
2. The buddy page loads the 311 form inside an iframe via a **same-origin reverse proxy**
3. User clicks "Send Context" → JS extracts visible form fields from the iframe DOM, sends them + transcript to the LLM
4. User clicks "Autofill Form" → JS fills the iframe's fields with the LLM's structured response
5. User advances to the next form step and repeats (conversation history carries context across steps)

## File layout

### Backend (Python/FastAPI)
- `main.py` — FastAPI app, page routes (`/`, `/buddy/{id}`), API routes (`/api/sessions/...`), proxy route (`/s/{id}/proxy/...`)
- `config.py` — LLM endpoint URL, model name, parameters, system prompt, allowed proxy host
- `models.py` — Pydantic models for API request/response (fields, suggestions, fills)
- `session_store.py` — In-memory session dict (transcript, message history, known_facts, per-session httpx client for proxy cookies)
- `llm_client.py` — Calls `/v1/chat/completions` (OpenAI-compatible), parses structured JSON response, one retry on invalid JSON
- `proxy.py` — Reverse proxy for `portal.311.nyc.gov`: URL rewriting in HTML/CSS/JS, intercept script injection, redirect rewriting, frame-blocking header stripping

### Frontend (static HTML/CSS/JS)
- `static/start.html` — Transcript + URL input form
- `static/buddy.html` — iframe + collapsible buddy panel (bottom sheet on mobile, sidebar on desktop)
- `static/buddy.js` — Field extraction from iframe DOM, Send Context / Autofill Form logic
- `static/styles.css` — Mobile-first responsive styles

## Key design decisions

- **Reverse proxy** (`proxy.py`) makes the 311 iframe same-origin so JS can read/write form fields. This is the most complex part of the codebase.
- **URL rewriting** happens at three layers: server-side HTML/CSS/JS rewriting, injected JS intercept script (fetch/XHR/location/form submit), and redirect Location header rewriting.
- **External hosts** (powerapps CDN, google translate, etc.) are explicitly preserved — only `portal.311.nyc.gov` URLs get proxied.
- **ASP.NET `~/` paths** are rewritten both server-side and client-side (the 311 portal is a PowerApps/Dynamics site).
- **Session state** is in-memory only — fine for local single-user use. Each session holds its own httpx client with a cookie jar for the proxied 311 session.
- **LLM conversation** is maintained as a growing messages array per session. Known facts are extracted and re-injected on each step so the model doesn't need to re-derive basics.

## Running

```
uv run fastapi dev main.py
```

## LLM endpoint

Expects an OpenAI-compatible `/v1/chat/completions` endpoint (e.g. llama.cpp). Configure in `config.py`.
Currently using Gemma 4 26B (`unsloth/gemma-4-26B-A4B-it-GGUF`) via llama.cpp.

## LLM response parsing (`llm_client.py`)

`llm_client.py` handles several model output quirks before JSON parsing:
1. **Thinking tags** — strips `<think>...</think>` blocks (and unclosed `<think>` from truncation). Models with reasoning (Gemma 4-it, QwQ, DeepSeek-R1) emit these.
2. **Markdown fences** — strips `` ```json ... ``` `` wrappers.
3. **Retry** — if JSON parse fails after stripping, appends a correction prompt and retries once.

When swapping models, the main things to adjust in `config.py`:
- `LLM_MODEL` — model identifier
- `max_tokens` — thinking models need headroom (currently 4096); increase if the debug panel shows parse failures from truncated output
- Sampling params (`temperature`, `top_p`, `top_k`) — tuned per model family
