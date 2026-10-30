"""Estimación local de tempo y pulsos con librosa."""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 512
MIN_AUDIO_SECONDS = 4


def analyze(y: np.ndarray, sr: int = SR) -> dict:
    """Devuelve el BPM global y los tiempos de pulso; BPM=0 si no hay señal útil."""
    if len(y) < sr * MIN_AUDIO_SECONDS or not np.any(y):
        return {"bpm": 0.0, "beats": np.zeros(0, dtype=np.float64)}

    import librosa

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempos, beat_times = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sr,
        hop_length=HOP,
        units="time",
        trim=False,
    )
    bpm = float(np.asarray(tempos).reshape(-1)[0])
    beats = np.asarray(beat_times, dtype=np.float64)
    if not np.isfinite(bpm) or bpm <= 0:
        return {"bpm": 0.0, "beats": np.zeros(0, dtype=np.float64)}
    return {"bpm": round(bpm, 1), "beats": beats}
