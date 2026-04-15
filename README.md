# SparX

> **NVIDIA Spark Hack Series: NYC Hackathon**.  
> Built by **Yahil, Hirab, Alejandro, and Aditya**

---

## What is SparX?

SparX is a **privacy-first, locally-run AI social worker** designed to give underserved communities in New York City secure, equitable access to government resources and services — without ever sending their data to the cloud.

The system runs entirely on local hardware (an **Acer Veriton N100** with **128 GB RAM** and an **NVIDIA GB10 Grace Blackwell Superchip** — capable of up to **1 petaFLOP of FP4 AI performance**), meaning no user profile is built, no conversation is stored on a remote server, and no third party can access what is being said. For communities that have historically had reason to distrust surveillance or data collection, this matters.


## The Problem

NYC is one of the most linguistically diverse cities in the world. Millions of residents speak English as a second language, or not at all. Navigating city services — 311, benefits enrollment, housing forms, emergency resources — is already complex in English. For Spanish, Mandarin, Bengali, Haitian Creole, or Arabic speakers, it can be nearly impossible.

Existing AI assistants (ChatGPT, Google Assistant, Alexa) require an internet connection and build persistent profiles on users. Underserved populations — immigrants, low-income residents, undocumented individuals — are often the most hesitant to use these tools precisely because of privacy concerns.


## The Solution

SparX puts the AI **on the edge, not in the cloud.**

Using NVIDIA's **Parakeet ASR** model for speech transcription, SparX supports **25 European languages** with the ability to seamlessly switch between them mid-sentence — including mixed-language speech like Spanglish. A user doesn't have to choose a language before speaking; the system understands them naturally.

From there, a direction engine routes the transcribed query to **NemoClaw**, which queries **NYC Open Data** and a 311 **FormFinder** to surface the most relevant city resources and forms — all in real time, all locally.


## Key Features

- 🎙️ **Multilingual ASR** — Parakeet supports 25 European languages; switch languages mid-sentence
- 🔒 **Fully local** — runs on-device; no cloud calls, no data retention, no user profiling
- 🗽 **NYC-focused** — integrated with NYC Open Data and 311 services via NemoClaw
- 🧭 **Smart routing** — LLM-based direction engine classifies intent and surfaces the right resource
- 🌐 **Browser-accessible** — served over HTTPS for microphone access from any local browser


## Hardware

| Component | Spec |
|-----------|------|
| Device    | Acer Veriton N100 |
| RAM       | 128 GB |
| Accelerator | NVIDIA GB10 Grace Blackwell Superchip |
| AI Performance | Up to 1 petaFLOP (FP4) |
| OS        | Debian/Ubuntu Linux |


---

## Prerequisites

- **Python 3.12** (or the version you use for this repo's venv)
- **NVIDIA GPU** with a driver compatible with your chosen PyTorch CUDA wheels
- **ffmpeg** — audio conversion (`pydub`) for ASR
- **openssl** — generates self-signed TLS certs for the voice server if `cert.pem` / `key.pem` are missing

On Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg openssl
```


## Python Environment

From the repository root (the directory that contains `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```


## Install Dependencies (order matters)

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

### CUDA Runtime Libraries in the venv

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

### Ollama-pulled Nemotron → `llama-server`

NVIDIA's public **Nano** line on [Ollama](https://ollama.com/library/nemotron-3-nano) is **`nemotron-3-nano:4b`** and **`nemotron-3-nano:30b`** (about **30B total** MoE weights in the larger tag—not a separate widely advertised ‘80B Nano”). For **~120B** MoE use **`nemotron-3-super:120b`**. If you have another tag (e.g. a future **`:80b`**), the same steps apply after `ollama pull`.

1. `ollama pull \<library\>:\<tag\>`
2. Resolve the GGUF blob and print a ready-made `llama-server` line:

   ```bash
   ./scripts/ollama_model_to_llama_server.sh nemotron-3-nano 30b
   ```

3. Run the printed `llama-server -m /.../blobs/sha256-...` command, then set `SPARX_LLM_CHAT_URL` / `SPARX_LLM_MODEL` as shown.

Use a **recent** `llama-server` build (Nemotron/Minitron support landed in upstream [llama.cpp](https://github.com/ggml-org/llama.cpp)).

## Run the Voice Application

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


## Repository Layout

| Path | Role |
|------|------|
| `nemoclaw/nodes/voice_node/server.py` | FastAPI HTTPS app: upload audio → ASR → direction → NemoClaw |
| `nemoclaw/nodes/voice_node/asr_engine.py` | Parakeet / NeMo voice ASR |
| `nemoclaw/nodes/voice_node/direction_engine.py` | LLM-based routing; logs to `complaints.log` |
| `nemoclaw/nemoclaw.py` | NemoClaw orchestrator (Open Data, then FormFinder) |
| `form_finder/` | 311 form classifier; uses `KA.json` |
| `scripts/source_cuda_libs.sh` | Prepends venv NVIDIA libs to `LD_LIBRARY_PATH` |


## Git: Nested Repositories

If `git add` fails with **`'some_dir/' does not have a commit checked out`**, that directory likely contains its own `.git/` with no commits yet. Either remove that nested `.git` to track normal files in this repo, or commit inside the subdirectory and use a proper submodule workflow.
