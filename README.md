# SparX

Voice node (Parakeet ASR), direction engine, NemoClaw (NYC Open Data + FormFinder), and related tooling.

## Prerequisites

- **Python 3.12** (or the version you use for this repo’s venv)
- **NVIDIA GPU** with a driver compatible with your chosen PyTorch CUDA wheels
- **ffmpeg** — audio conversion (`pydub`) for ASR
- **openssl** — generates self-signed TLS certs for the voice server if `cert.pem` / `key.pem` are missing

On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg openssl
```

## Python environment

From the repository root (the directory that contains `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

## Install dependencies (order matters)

1. Install **PyTorch + torchaudio** from the file that matches your CUDA/driver (see [PyTorch — Get Started](https://pytorch.org/get-started/locally/) if unsure):

   ```bash
   pip install -r requirements-torch-cu130.txt
   ```

   or

   ```bash
   pip install -r requirements-torch-cu124.txt
   ```

2. Then install the rest:

   ```bash
   pip install -r requirements.txt
   ```

If `import torchaudio` fails with FFT-related symbol errors, your `torch` and `torchaudio` builds do not match. Fix:

```bash
pip uninstall -y torch torchaudio
pip install -r requirements-torch-cu124.txt   # or cu130
```

### CUDA runtime libraries in the venv

If you see errors such as `OSError: libcudart.so.12: cannot open shared object file`, with the venv activated run:

```bash
source scripts/source_cuda_libs.sh
```

That prepends NVIDIA wheel libraries under `site-packages` to `LD_LIBRARY_PATH`. The project also lists `nvidia-cuda-runtime-cu12` in `requirements.txt` for many setups.

## LLM (required at runtime)

The direction engine and FormFinder call an **OpenAI-compatible** HTTP API.

- Default URL: `http://localhost:8081/v1/chat/completions`
- Override with:

  ```bash
  export SPARX_LLM_CHAT_URL="http://HOST:PORT/v1/chat/completions"
  ```

Optional:

```bash
export SPARX_LLM_MODEL="your-model-id"
```

Run **llama.cpp** (`llama-server`), **vLLM**, or any compatible server before using voice features that need classification or FormFinder.

## Run the voice application

The server loads Parakeet on startup and serves **HTTPS** on port **8443** (needed for microphone access from the browser). If `cert.pem` and `key.pem` are missing in `nemoclaw/nodes/voice_node/`, the app generates self-signed certificates with `openssl`.

Recommended (keeps `key.pem` / `cert.pem` next to the server; cwd matters for OpenSSL output):

```bash
cd nemoclaw/nodes/voice_node
source ../../../.venv/bin/activate
# optional if you hit libcudart loader errors:
# source ../../../scripts/source_cuda_libs.sh
python server.py
```

From the repository root you can run `python nemoclaw/nodes/voice_node/server.py`; self-signed certs are then created in the **current directory** (repo root), not under `voice_node/`.

Open **`https://<host-ip>:8443`** in a browser and accept the certificate warning for local/self-signed TLS.

## Repository layout (high level)

| Path | Role |
|------|------|
| `nemoclaw/nodes/voice_node/server.py` | FastAPI HTTPS app: upload audio → ASR → direction → NemoClaw |
| `nemoclaw/nodes/voice_node/asr_engine.py` | Parakeet / NeMo voice ASR |
| `nemoclaw/nodes/voice_node/direction_engine.py` | LLM-based routing; logs to `complaints.log` |
| `nemoclaw/nemoclaw.py` | NemoClaw orchestrator (Open Data, then FormFinder) |
| `form_finder/` | 311 form classifier; uses `KA.json` |
| `scripts/source_cuda_libs.sh` | Prepends venv NVIDIA libs to `LD_LIBRARY_PATH` |

## Git: nested repositories

If `git add` fails with **`'some_dir/' does not have a commit checked out`**, that directory likely contains its own `.git/` with no commits yet. Either remove that nested `.git` to track normal files in this repo, or commit inside the subdirectory and use a proper submodule workflow.
