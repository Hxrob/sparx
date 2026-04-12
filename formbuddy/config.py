import os

# Base URL only (no /v1/...). Must match llama-server --port and http vs https.
# Override if your server differs: export FORMBUDDY_LLM_BASE_URL=http://127.0.0.1:8081
LLM_BASE_URL = os.environ.get("FORMBUDDY_LLM_BASE_URL", "http://localhost:8081").rstrip("/")
LLM_MODEL = os.environ.get("FORMBUDDY_LLM_MODEL", "unsloth/gemma-4-26B-A4B-it-GGUF")
LLM_OPTIONS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_tokens": 4096,
}

ALLOWED_PROXY_HOST = "portal.311.nyc.gov"

SYSTEM_PROMPT = """\
You are FormBuddy. Your job is to help fill NYC 311 complaint forms.
Use the transcript and previously learned facts to suggest values only for the currently visible form fields.
Return JSON only, matching this schema exactly:
{
  "assistant_message": "short explanation of what you filled",
  "known_facts": {"key": "value"},
  "fills": [
    {
      "field_id": "the field id or name",
      "value": "the value to fill",
      "confidence": 0.0 to 1.0,
      "reason": "why this value"
    }
  ],
  "needs_user_input": ["field_ids you cannot determine"]
}
Rules:
- Return ONLY valid JSON, no markdown fences, no extra text.
- Only use field_ids from the provided list of visible fields.
- For selects/radios, choose ONLY from the provided options (use exact option values).
- Never invent facts not grounded in the transcript or prior known facts.
- If uncertain about a field, omit it from fills and add to needs_user_input.
"""
