import time
import argparse
import requests
import queue
import torch
import sounddevice as sd
import soundfile as sf
from nemo.collections.asr.models import ASRModel

class VoiceNode:
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3", vllm_url: str = "http://localhost:8000/v1/chat/completions"):
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.sample_rate = 16000
        self.chunk_duration_ms = 500  # Listen in 500ms audio chunks
        self.chunk_samples = int(self.sample_rate * (self.chunk_duration_ms / 1000.0))
        
        # Audio queue for processing chunks asynchronously 
        self.audio_queue = queue.Queue()
        
        print(f"Loading ASR model {self.model_name}...")
        self.model = ASRModel.from_pretrained(model_name=self.model_name)
        
        # Blackwell Optimization: Enable `.eval()` mode and push to `cuda`
        print("Moving model to CUDA (Blackwell hardware)...")
        self.model = self.model.to("cuda")
        self.model.eval()
        
        print("Model loaded successfully.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called automatically by sounddevice for each chunk of audio stream."""
        if status:
            print(f"Status in _audio_callback: {status}", flush=True)
            
        # Select Mono channel (index 0) and push to the processing queue
        audio_chunk = torch.tensor(indata[:, 0], dtype=torch.float32)
        self.audio_queue.put(audio_chunk)

    def transcribe_stream(self):
        """
        Stream audio from microphone, push it natively to the DGX GPU, and transcribe.
        """
        print(f"Starting audio stream... (Sample Rate: {self.sample_rate}Hz, Chunk: {self.chunk_duration_ms}ms)")
        
        device_context = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            blocksize=self.chunk_samples,
            callback=self._audio_callback
        )
        
        # Enable torch.inference_mode and bfloat16 tensor core optimization for maximum speed profiling on Blackwell
        with device_context, torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            print("Listening for Spanglish natively... Press Ctrl+C to stop.")
            try:
                while True:
                    audio_chunk = self.audio_queue.get()
                    
                    # Ensure shape is appropriate for numpy ingestion 
                    audio_array = audio_chunk.cpu().numpy()
                    
                    # Process directly through the rnnt network using the 'audio' argument
                    transcription_outputs = self.model.transcribe(
                        audio=audio_array
                    )
                    
                    # Parse transcription result
                    # Model behavior varies based on batching, typically text is tuple 0 or string index 0
                    if isinstance(transcription_outputs, tuple):
                        text = transcription_outputs[0][0]
                    elif isinstance(transcription_outputs, list):
                        text = transcription_outputs[0]
                    else:
                        text = transcription_outputs

                    # Unwrap from NeMo Hypothesis object if present
                    if hasattr(text, 'text'):
                        text = text.text
                    elif not isinstance(text, str):
                        text = str(text)

                    text = text.strip()
                    if text:
                        print(f"[ASR] Transcription: {text}")
                        # self.send_to_vllm(text)
                        
            except KeyboardInterrupt:
                print("Stopping audio stream gracefully.")

    def transcribe_wav(self, file_path: str):
        """
        Test the ASR and vLLM bridge with a pre-recorded WAV file.
        """
        print(f"Transcribing WAV file: {file_path}")
        
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Based on the signature, we can just pass the path directly into 'audio'
            transcription_outputs = self.model.transcribe(
                audio=[file_path]
            )
            
            if isinstance(transcription_outputs, tuple):
                text = transcription_outputs[0][0]
            elif isinstance(transcription_outputs, list):
                text = transcription_outputs[0]
            else:
                text = transcription_outputs
                
            # Unwrap from NeMo Hypothesis object if present
            if hasattr(text, 'text'):
                text = text.text
            elif not isinstance(text, str):
                text = str(text)

            text = text.strip()
            if text:
                print(f"[ASR] Transcription: {text}")
                # self.send_to_vllm(text)
                return text
            else:
                print("[ASR] No speech detected in WAV file.")
                return ""

    def send_to_vllm(self, text: str):
        """
        Pipes the transcribed text to the local vLLM reasoning engine, with zero external calls.
        """
        # Note: Set "model" to whatever you invoked your local vLLM docker image with. 
        # (Often "meta-llama/Llama-2-7b-chat-hf" or similar huggingface name).
        payload = {
            "model": "vllm-local-model",
            "messages": [
                {"role": "system", "content": "You are a bilingual Social Help Agent. You must provide useful advice given the user's transcript in Spanglish."},
                {"role": "user", "content": text}
            ],
            "max_tokens": 150
        }
        
        try:
            # Low latency, blocking POST using the bridge
            response = requests.post(self.vllm_url, json=payload, timeout=5.0)
            if response.status_code == 200:
                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                print(f"[vLLM Reasoning Bridge]: \033[92m{reply}\033[0m")
            else:
                print(f"[vLLM Error] Code: {response.status_code}, Body: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[vLLM Exception]: {e} (Is vllm running on {self.vllm_url}?)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Help Agent Voice Node")
    parser.add_argument("--wav", type=str, help="Path to a WAV file to transcribe instead of using microphone stream")
    args = parser.parse_args()
    
    node = VoiceNode()
    if args.wav:
        node.transcribe_wav(args.wav)
    else:
        node.transcribe_stream()
