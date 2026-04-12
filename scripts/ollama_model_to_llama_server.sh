#!/usr/bin/env bash
# Resolve an Ollama library model tag to the on-disk GGUF blob and print a llama-server example.
#
# Usage:
#   ./ollama_model_to_llama_server.sh nemotron-3-nano 30b
#   ./ollama_model_to_llama_server.sh nemotron-4-nano 80b    # if that tag exists after ollama pull
#
# Prerequisite:  ollama pull <name>:<tag>
#
# Override models root if not under ~/.ollama or /usr/share/ollama:
#   export OLLAMA_MODELS=/path/to/.ollama/models

set -euo pipefail

LIB="${1:?library name e.g. nemotron-3-nano}"
TAG="${2:?tag e.g. 30b or 80b}"

resolve_root() {
  local d
  for d in "${OLLAMA_MODELS:-}" "${HOME}/.ollama/models" "/usr/share/ollama/.ollama/models"; do
    [[ -z "$d" ]] && continue
    if [[ -f "$d/manifests/registry.ollama.ai/library/$LIB/$TAG" ]]; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

ROOT="$(resolve_root)" || {
  echo "Manifest not found for registry.ollama.ai/library/$LIB/$TAG" >&2
  echo "Tried OLLAMA_MODELS, ~/.ollama/models, /usr/share/ollama/.ollama/models" >&2
  echo "Run: ollama pull $LIB:$TAG" >&2
  exit 1
}

MAN="$ROOT/manifests/registry.ollama.ai/library/$LIB/$TAG"
DIGEST="$(python3 - <<'PY' "$MAN"
import json, sys, pathlib
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
for lay in m.get("layers", []):
    if lay.get("mediaType") == "application/vnd.ollama.image.model":
        d = lay["digest"]  # sha256:hex
        if d.startswith("sha256:"):
            print("sha256-" + d.split(":", 1)[1])
        else:
            print(d)
        raise SystemExit(0)
raise SystemExit("no model layer in manifest")
PY
)"

BLOB="$ROOT/blobs/$DIGEST"
if [[ ! -f "$BLOB" ]]; then
  echo "Blob missing: $BLOB" >&2
  exit 1
fi

echo "# Model blob (GGUF inside Ollama store):"
echo "$BLOB"
echo ""
echo "# Example llama-server (adjust -ngl and llama-server path):"
echo "llama-server -m \"$BLOB\" --host 0.0.0.0 --port 8081 -ngl 99"
echo ""
echo "# SparX / FormBuddy (OpenAI shim — set model id if your server requires it):"
echo "export SPARX_LLM_CHAT_URL=http://127.0.0.1:8081/v1/chat/completions"
echo "export SPARX_LLM_MODEL=${LIB}:${TAG}"
echo "# or FORMBUDDY_LLM_BASE_URL=http://127.0.0.1:8081  FORMBUDDY_LLM_MODEL=${LIB}:${TAG}"
