# FormBuddy

Local dev tool that helps fill NYC 311 complaint forms using a local LLM. Paste a transcript and a 311 form URL, then iterate through the multi-step form with LLM-assisted autofill.

## How it works

1. Submit a transcript + 311 URL on the start page
2. The buddy page loads the 311 form in an iframe via a same-origin reverse proxy
3. Click **Send Context** → JS extracts visible form fields from the iframe, sends them + transcript to the LLM
4. Click **Autofill Form** → JS fills the iframe's fields with the LLM's structured response
5. Advance to the next form step and repeat (conversation history carries context across steps)

## Setup

### LLM server

FormBuddy expects an OpenAI-compatible `/v1/chat/completions` endpoint. We use [llama.cpp](https://github.com/ggerganov/llama.cpp) with Gemma 4 26B:

```bash
export LLAMA_CACHE="unsloth/gemma-4-26B-A4B-it-GGUF"
llama-server \
    -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64
```

The server runs on `http://localhost:8080` by default. Configure the endpoint and model in `config.py`.

### Swapping models

Any OpenAI-compatible local model should work, but watch for these quirks:

- **Thinking tags**: Instruction-tuned models with thinking/reasoning (Gemma 4, QwQ, DeepSeek-R1) wrap output in `<think>...</think>` blocks. `llm_client.py` strips these automatically before parsing JSON.
- **Markdown fences**: Many models wrap JSON in `` ```json ... ``` `` fences even when told not to. Also stripped automatically.
- **`max_tokens`**: Models with thinking burn tokens on reasoning before producing output. If the model hits the limit mid-thought, the response will contain no JSON at all. The default (4096 in `config.py`) leaves room for ~2k thinking + ~2k output. Increase if you see parse failures in the debug panel.
- **Sampling params**: `temperature`, `top_p`, `top_k` in `config.py` are tuned for Gemma 4. Other models may need different values (e.g. Qwen3 works better with `temperature: 0.1` and no `top_k`).
- **Retry**: If the first response isn't valid JSON, `llm_client.py` appends a correction prompt and retries once. The debug panel shows whether a retry occurred.

### App server

```bash
uv run fastapi dev main.py
```

Then open `http://127.0.0.1:8000` in your browser.

## Debug panel

After each LLM request, a collapsible debug panel appears in the buddy sidebar showing:

- Request duration and tokens/sec
- Prompt, completion, and total token counts
- Whether the response parsed cleanly or required a retry
- Raw LLM response text
- The full message history sent to the model
