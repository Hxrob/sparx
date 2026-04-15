"""
ASR Engine — Parakeet voice transcription for SparX.

Wraps NVIDIA's Parakeet TDT model (via NeMo) for two use cases:
  - transcribe_wav(): offline WAV file transcription (used by server.py)
  - transcribe_stream(): live mic stream (dev/testing ONLY)

Model is loaded once at startup and stays on the GPU.
"""
import argparse
import queue
import torch
import sounddevice as sd
import soundfile as sf
from nemo.collections.asr.models import ASRModel


class VoiceNode:
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
        self.model_name = model_name
        self.sample_rate = 16000
        self.chunk_duration_ms = 500
        self.chunk_samples = int(self.sample_rate * (self.chunk_duration_ms / 1000.0))

        self.audio_queue = queue.Queue()

        print(f"Loading ASR model {self.model_name}...")
        self.model = ASRModel.from_pretrained(model_name=self.model_name)

        # enable eval mode and push to GPU
        print("Moving model to CUDA...")
        self.model = self.model.to("cuda")
        self.model.eval()

        print("Model loaded successfully.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called automatically by sounddevice for each chunk of audio stream."""
        if status:
            print(f"Status in _audio_callback: {status}", flush=True)

        # select mono channel (index 0) and push to the processing queue
        audio_chunk = torch.tensor(indata[:, 0], dtype=torch.float32)
        self.audio_queue.put(audio_chunk)

    def transcribe_stream(self):
        """
        Stream audio from mic and transcribe in real time.
        Dev/testing utility ONLY — production uses transcribe_wav() via /transcribe endpoint.
        """
        print(f"Starting audio stream... (Sample Rate: {self.sample_rate}Hz, Chunk: {self.chunk_duration_ms}ms)")

        device_context = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_samples,
            callback=self._audio_callback
        )

        with device_context, torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            print("Listening... Press Ctrl+C to stop.")
            try:
                while True:
                    audio_chunk = self.audio_queue.get()
                    audio_array = audio_chunk.cpu().numpy()

                    transcription_outputs = self.model.transcribe(audio=audio_array)

                    if isinstance(transcription_outputs, tuple):
                        text = transcription_outputs[0][0]
                    elif isinstance(transcription_outputs, list):
                        text = transcription_outputs[0]
                    else:
                        text = transcription_outputs

                    if hasattr(text, 'text'):
                        text = text.text
                    elif not isinstance(text, str):
                        text = str(text)

                    text = text.strip()
                    if text:
                        print(f"[ASR] {text}")

            except KeyboardInterrupt:
                print("Stopping audio stream gracefully.")

    def transcribe_wav(self, file_path: str) -> str:
        """
        Transcribe a pre-recorded 16kHz mono WAV file.
        Returns the transcription string, or empty string if no speech detected.
        """
        print(f"Transcribing WAV file: {file_path}")

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            transcription_outputs = self.model.transcribe(audio=[file_path])

            if isinstance(transcription_outputs, tuple):
                text = transcription_outputs[0][0]
            elif isinstance(transcription_outputs, list):
                text = transcription_outputs[0]
            else:
                text = transcription_outputs

            if hasattr(text, 'text'):
                text = text.text
            elif not isinstance(text, str):
                text = str(text)

            text = text.strip()
            if text:
                print(f"[ASR] {text}")
                return text
            else:
                print("[ASR] No speech detected in WAV file.")
                return ""


if __name__ == "__main__":
    # dev util: run to test mic or WAV transcription
    parser = argparse.ArgumentParser(description="SparX ASR Engine - dev test entrypoint")
    parser.add_argument("--wav", type=str, help="Path to a WAV file to transcribe (omit for mic stream)")
    args = parser.parse_args()

    node = VoiceNode()
    if args.wav:
        node.transcribe_wav(args.wav)
    else:
        node.transcribe_stream()
