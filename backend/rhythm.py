"""
Tempo y pulsos (beats) usando solo numpy.

  audio -> fuerza de ataques (spectral flux) -> tempo por autocorrelación
        -> seguimiento de pulsos por programación dinámica (Ellis, 2007)
"""
from __future__ import annotations

import numpy as np

SR = 22050
N_FFT = 2048
HOP = 512  # ~23 ms


def onset_envelope(y: np.ndarray, sr: int = SR) -> tuple[np.ndarray, float]:
    """Fuerza de ataques por cuadro. Devuelve (env, cuadros_por_segundo)."""
    if len(y) < N_FFT * 2:
        return np.zeros(1, dtype=np.float32), sr / HOP
    n = 1 + (len(y) - N_FFT) // HOP
    win = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    keep = (freqs >= 40) & (freqs <= 8000)
    offs = np.arange(N_FFT)[None, :]

    spec = np.empty((n, int(keep.sum())), dtype=np.float32)
    for s in range(0, n, 512):
        e = min(s + 512, n)
        frames = y[(np.arange(s, e) * HOP)[:, None] + offs] * win
        spec[s:e] = np.abs(np.fft.rfft(frames, axis=1))[:, keep]

    spec = np.log1p(30.0 * spec / (spec.max() + 1e-9))        # compresión logarítmica
    flux = np.maximum(0.0, np.diff(spec, axis=0)).mean(axis=1)  # solo aumentos de energía
    env = np.concatenate([[0.0], flux]).astype(np.float64)

    fps = sr / HOP
    w = max(int(0.5 * fps), 3)                                # quita la tendencia lenta
    env = np.maximum(0.0, env - np.convolve(env, np.ones(w) / w, mode="same"))
    return env / (env.std() + 1e-9), fps


def estimate_tempo(env: np.ndarray, fps: float, lo: float = 60.0, hi: float = 180.0) -> float:
    """BPM por autocorrelación, con preferencia por tempos "humanos" (~110 BPM)."""
    n = len(env)
    if n < int(4 * fps):
        return 0.0
    f = np.fft.rfft(env, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-9)

    bpms = np.arange(lo, hi + 0.05, 0.1)
    lags = 60.0 * fps / bpms
    grid = np.arange(n)

    def at(l):  # autocorrelación interpolada en lags fraccionarios
        return np.interp(l, grid, ac, right=0.0)

    # Un tempo real se repite también al doble y cuádruple de su periodo (compás)
    score = at(lags) + 0.5 * at(2 * lags) + 0.25 * at(4 * lags)
    score = score * np.exp(-0.5 * (np.log2(bpms / 110.0) / 0.9) ** 2)
    return float(bpms[int(np.argmax(score))])


def track_beats(env: np.ndarray, fps: float, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Tiempos (s) de los pulsos, por programación dinámica."""
    if bpm <= 0 or len(env) < 8:
        return np.zeros(0)
    period = 60.0 * fps / bpm
    g = np.exp(-0.5 * (np.arange(-int(period / 4) - 1, int(period / 4) + 2) / max(period / 16, 1.0)) ** 2)
    local = np.convolve(env, g / g.sum(), mode="same")

    n = len(local)
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    lo, hi = max(int(round(period / 2)), 1), int(round(2 * period))
    for t in range(n):
        a, b = max(t - hi, 0), t - lo
        if b < 0:
            cum[t] = local[t]
            continue
        prev = np.arange(a, b + 1)
        cand = cum[prev] - tightness * np.log((t - prev) / period) ** 2
        j = int(np.argmax(cand))
        cum[t] = local[t] + cand[j]
        back[t] = prev[j]

    tail = max(int(period), 1)
    t = n - tail + int(np.argmax(cum[-tail:]))
    idx = []
    while t >= 0:
        idx.append(t)
        t = back[t]
    idx = np.array(idx[::-1])

    # centro de la ventana del STFT = instante del ataque
    return (idx * HOP + N_FFT / 2) / (fps * HOP)


def analyze(y: np.ndarray, sr: int = SR) -> dict:
    """{'bpm': float, 'beats': ndarray de segundos}. bpm=0 si no se pudo estimar."""
    env, fps = onset_envelope(y, sr)
    bpm = estimate_tempo(env, fps)
    beats = track_beats(env, fps, bpm)
    if len(beats) >= 8:
        idx = np.arange(len(beats))
        slope, inter = np.polyfit(idx, beats, 1)           # tiempo = inicio + n * periodo
        resid = np.abs(beats - (slope * idx + inter))
        if np.median(resid) < 0.08 * slope:                # tempo estable: usa el ajuste global
            bpm = float(60.0 / slope)
        else:                                              # tempo variable: usa la mediana
            bpm = float(60.0 / np.median(np.diff(beats)))
    elif len(beats) >= 3:
        bpm = float(60.0 / np.median(np.diff(beats)))
    return {"bpm": round(bpm, 1), "beats": beats}
