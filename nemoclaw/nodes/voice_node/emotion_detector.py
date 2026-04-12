"""
Emotion Detector — paralinguistic distress analysis from raw audio.

Analyzes a 16kHz mono WAV file for acoustic markers of emotional distress
before the transcript ever hits the LLM. Runs entirely on-device — no cloud.

Markers detected:
  - Elevated / unstable pitch (fear, anxiety)
  - Voice tremor in the 4-8Hz range (distress, crying)
  - Rapid speech rate (anxiety, panic)
  - Frequent long pauses (overwhelmed, dissociated, depressed)
  - High energy variance (emotional intensity)

Returns a distress score (0.0–1.0) and a human-readable emotion label
that the direction engine uses to adjust its response tone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger("emotion_detector")

DISTRESS_THRESHOLD = 0.35   # above this → adjust direction engine tone
CRISIS_THRESHOLD   = 0.65   # above this → crisis protocol


@dataclass
class EmotionResult:
    distress_score: float           # 0.0 – 1.0
    emotion: str                    # neutral | mild_distress | distress | crisis
    markers: list[str] = field(default_factory=list)
    arousal: float = 0.5            # energy level (0 low – 1 high)
    valence: float = 0.5            # positivity proxy (0 negative – 1 positive)

    @property
    def is_distressed(self) -> bool:
        return self.distress_score >= DISTRESS_THRESHOLD

    @property
    def is_crisis(self) -> bool:
        return self.distress_score >= CRISIS_THRESHOLD


def analyze(wav_path: str) -> EmotionResult:
    """
    Analyze a WAV file and return an EmotionResult.
    Falls back gracefully to neutral on any error.
    """
    try:
        return _analyze(wav_path)
    except Exception as exc:
        LOGGER.warning("Emotion analysis failed (%s): %s", wav_path, exc)
        return EmotionResult(distress_score=0.0, emotion="neutral")


def _analyze(wav_path: str) -> EmotionResult:
    import librosa
    import librosa.feature
    import librosa.onset

    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # Need at least 1 second of audio to be meaningful
    if len(y) < sr:
        return EmotionResult(distress_score=0.0, emotion="neutral")

    markers: list[str] = []
    distress = 0.0

    # ------------------------------------------------------------------
    # 1. Pitch (F0) analysis — fear/anxiety raise and destabilize pitch
    # ------------------------------------------------------------------
    try:
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
        )
        f0_voiced = f0[voiced & ~np.isnan(f0)] if voiced is not None else np.array([])
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) > 10:
        pitch_mean = float(np.mean(f0_voiced))
        pitch_std  = float(np.std(f0_voiced))

        # Elevated pitch → fear / high stress
        if pitch_mean > 280:
            distress += 0.15
            markers.append("elevated pitch")
        elif pitch_mean > 220:
            distress += 0.08

        # High pitch variance → emotional instability
        if pitch_std > 70:
            distress += 0.20
            markers.append("pitch instability")
        elif pitch_std > 45:
            distress += 0.10

        # Abrupt pitch jumps → voice breaks (crying, panic)
        diffs = np.abs(np.diff(f0_voiced))
        break_ratio = float(np.mean(diffs > 60))
        if break_ratio > 0.15:
            distress += 0.15
            markers.append("voice breaks")

    # ------------------------------------------------------------------
    # 2. Voice tremor — 4–8 Hz modulation of amplitude envelope
    # ------------------------------------------------------------------
    try:
        from scipy import signal as scipy_signal

        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        hop_rate = sr / 256  # frames per second

        if len(rms) >= 32:
            freqs, psd = scipy_signal.welch(rms, fs=hop_rate, nperseg=min(len(rms), 128))
            tremor_band = (freqs >= 4) & (freqs <= 8)
            baseline     = float(np.mean(psd))

            if tremor_band.any() and baseline > 0:
                tremor_power = float(np.max(psd[tremor_band]))
                if tremor_power > baseline * 4:
                    distress += 0.25
                    markers.append("voice tremor")
                elif tremor_power > baseline * 2.5:
                    distress += 0.12
    except Exception:
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]

    # ------------------------------------------------------------------
    # 3. Speaking rate — too fast (anxiety/panic) or too slow (depressed)
    # ------------------------------------------------------------------
    try:
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        duration    = len(y) / sr
        rate        = len(onset_times) / duration if duration > 0 else 0.0

        if rate > 6.0:
            distress += 0.10
            markers.append("rapid speech")
        elif rate < 1.2 and duration > 3.0:
            distress += 0.08
            markers.append("very slow speech")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4. Silence / pause ratio — long pauses signal being overwhelmed
    # ------------------------------------------------------------------
    try:
        energy_mean = float(np.mean(rms))
        silence_threshold = energy_mean * 0.08
        silent_ratio = float(np.mean(rms < silence_threshold))

        if silent_ratio > 0.55:
            distress += 0.10
            markers.append("long pauses")
    except Exception:
        energy_mean = 0.05

    # ------------------------------------------------------------------
    # 5. Energy variance — high variance = emotional intensity
    # ------------------------------------------------------------------
    try:
        energy_std = float(np.std(rms))
        if energy_std > 0.08:
            distress += 0.08
            markers.append("high energy variance")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Arousal and valence proxies
    # ------------------------------------------------------------------
    try:
        arousal = float(min(np.mean(rms) * 25, 1.0))
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        valence = float(min(np.mean(spectral_centroid) / (sr / 2) * 3, 1.0))
    except Exception:
        arousal, valence = 0.5, 0.5

    # ------------------------------------------------------------------
    # Final score and label
    # ------------------------------------------------------------------
    distress = float(min(distress, 1.0))

    if distress < 0.20:
        emotion = "neutral"
    elif distress < DISTRESS_THRESHOLD:
        emotion = "mild_distress"
    elif distress < CRISIS_THRESHOLD:
        emotion = "distress"
    else:
        emotion = "crisis"

    LOGGER.info(
        "Emotion: %s (score=%.2f) markers=%s",
        emotion, distress, markers or ["none"],
    )

    return EmotionResult(
        distress_score=distress,
        emotion=emotion,
        markers=markers,
        arousal=arousal,
        valence=valence,
    )
