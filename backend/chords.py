"""
Detección de acordes mayores, menores y séptimas usando numpy + ffmpeg.

  audio -> STFT -> chroma (12 notas) -> plantillas de acordes -> Viterbi

Las séptimas se aceptan solo cuando superan claramente la evidencia de su tríada.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import rhythm

SR = 22050
# Ajustes del decodificador (calibrados con un banco de canciones sintéticas difíciles):
#   bass  = peso de la evidencia del bajo para decidir la raíz (su ventana cubre más de un pulso)
#   tol7  = tolerancia a séptimas/novenas en las plantillas
#   stay  = probabilidad de seguir en el mismo acorde de un pulso al siguiente
CFG = dict(bass=0.4, tol7=0.4, stay=0.1, b3=0.5, b5=0.4)
N_FFT = 8192   # ventana de ~0.37 s: resolución suficiente para notas graves de guitarra
HOP = 2048     # un cuadro cada ~0.093 s
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TRIAD_COUNT = 24
SEVENTH_COUNT = 36
CHORD_COUNT = TRIAD_COUNT + SEVENTH_COUNT
NO_CHORD_STATE = TRIAD_COUNT
SEVENTH_STATE_START = NO_CHORD_STATE + 1
SEVENTH_MIN_MARGIN = 0.035
SEVENTH_MIN_SEPARATION = 0.02

# Perfiles de tonalidad de Krumhansl-Kessler
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def decode(path: Path, sr: int = SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("No se encontró ffmpeg. Instálalo con: winget install ffmpeg") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg no pudo leer el audio: " + proc.stderr.decode(errors="replace")[-300:])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _median_filter(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Mediana deslizante de ventana k a lo largo de `axis` (solo numpy, por bloques)."""
    a = np.moveaxis(a, axis, 0)                    # (n, resto)
    n = a.shape[0]
    pad = k // 2
    ap = np.pad(a, ((pad, pad),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    out = np.empty_like(a)
    step = max(1, 2_000_000 // (k * max(1, int(np.prod(a.shape[1:])))))
    for s0 in range(0, n, step):
        e0 = min(s0 + step, n)
        win = np.lib.stride_tricks.sliding_window_view(ap[s0:e0 + 2 * pad], k, axis=0)
        out[s0:e0] = np.median(win, axis=-1)
    return np.moveaxis(out, 0, axis)


def _harmonic_part(spec: np.ndarray, k_time: int = 9, k_freq: int = 13) -> np.ndarray:
    """Separación armónico/percusivo: conserva lo sostenido (notas) y atenúa golpes (batería)."""
    h = _median_filter(spec, k_time, axis=0)       # suave en el tiempo -> armónico
    p = _median_filter(spec, k_freq, axis=1)       # suave en frecuencia -> percusivo
    return spec * (h ** 2 / (h ** 2 + p ** 2 + 1e-12))


def _nonlocal_smooth(agg: np.ndarray, w: int = 4, k: int = 4, thr: float = 0.85) -> np.ndarray:
    """Las canciones repiten sus progresiones, la melodía no siempre. Promedia cada pulso con los
    pulsos de otras partes de la canción que tienen el mismo contexto armónico."""
    n = len(agg)
    if n < 3 * w:
        return agg
    a = agg / (np.linalg.norm(agg, axis=1, keepdims=True) + 1e-9)
    ctx = np.hstack([np.roll(a, -o, axis=0) for o in range(w)])      # contexto de w pulsos
    ctx /= np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9
    sim = ctx @ ctx.T
    idx = np.arange(n)
    sim[np.abs(idx[:, None] - idx[None, :]) < w] = -1.0              # no compararse con vecinos
    sim[n - w + 1:, :] = -1.0                                        # contexto incompleto al final
    sim[:, n - w + 1:] = -1.0
    out = agg.copy()
    top = np.argsort(-sim, axis=1)[:, :k]
    for i in range(n):
        js = [j for j in top[i] if sim[i, j] >= thr]
        if js:
            out[i] = (agg[i] + agg[js].sum(axis=0)) / (1 + len(js))
    return out


def chroma(y: np.ndarray, sr: int = SR, hpss: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (chroma[T,12] sin normalizar, energía[T])."""
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    n_frames = 1 + (len(y) - N_FFT) // HOP
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    idx = np.nonzero((freqs >= 130.0) & (freqs <= 2000.0))[0]
    win = np.hanning(N_FFT).astype(np.float32)
    offs = np.arange(N_FFT)[None, :]

    spec = np.empty((n_frames, len(idx)), dtype=np.float32)
    for s in range(0, n_frames, 256):
        e = min(s + 256, n_frames)
        frames = y[(np.arange(s, e) * HOP)[:, None] + offs] * win
        spec[s:e] = np.abs(np.fft.rfft(frames, axis=1))[:, idx]

    energy = spec.sum(axis=1)
    if hpss:
        spec = _harmonic_part(spec)

    # Solo cuentan los picos espectrales (máximos locales) que sobresalen del ruido del cuadro:
    # así el ruido de batería/fondo no "ensucia" las 12 notas.
    pad = np.pad(spec, ((0, 0), (3, 3)), mode="edge")
    local_max = np.max(
        np.stack([pad[:, o:o + spec.shape[1]] for o in range(7) if o != 3]), axis=0
    )
    floor = 2.5 * np.median(spec, axis=1, keepdims=True)
    spec = np.where((spec >= local_max) & (spec > floor), spec, 0.0)
    spec = np.sqrt(spec / (spec.max() + 1e-9))        # compresión suave

    # Afinación: desviación media de los picos respecto a las notas temperadas
    midi = 69.0 + 12.0 * np.log2(freqs[idx] / 440.0)
    w = spec.mean(axis=0)
    tuning = np.angle(np.sum(w * np.exp(2j * np.pi * (midi - np.round(midi))))) / (2 * np.pi)
    midi = midi - tuning

    # Cada bin aporta a la nota más cercana, con más peso cuanto más centrado esté
    nearest = np.round(midi)
    weight = np.exp(-0.5 * ((midi - nearest) / 0.3) ** 2)
    pcs = (nearest.astype(int)) % 12
    M = np.zeros((len(idx), 12), dtype=np.float32)
    M[np.arange(len(idx)), pcs] = weight
    return spec @ M, energy


def bass_chroma(y: np.ndarray, n_frames: int, sr: int = SR) -> np.ndarray:
    """Chroma de la banda grave (41-250 Hz): el bajo suele tocar la raíz del acorde."""
    nf = 16384
    pad = np.pad(y, ((nf - N_FFT) // 2, nf))
    freqs = np.fft.rfftfreq(nf, 1.0 / sr)
    idx = np.nonzero((freqs >= 41.0) & (freqs <= 250.0))[0]
    win = np.hanning(nf).astype(np.float32)
    offs = np.arange(nf)[None, :]
    spec = np.empty((n_frames, len(idx)), dtype=np.float32)
    for s0 in range(0, n_frames, 128):
        e0 = min(s0 + 128, n_frames)
        fr = pad[(np.arange(s0, e0) * HOP)[:, None] + offs] * win
        spec[s0:e0] = np.abs(np.fft.rfft(fr, axis=1))[:, idx]
    pad_ = np.pad(spec, ((0, 0), (2, 2)), mode="edge")
    lm = np.max(np.stack([pad_[:, o:o + spec.shape[1]] for o in range(5) if o != 2]), axis=0)
    spec = np.where(spec >= lm, spec, 0.0)
    spec = np.sqrt(spec / (spec.max() + 1e-9))
    midi = 69.0 + 12.0 * np.log2(freqs[idx] / 440.0)
    nearest = np.round(midi)
    weight = np.exp(-0.5 * ((midi - nearest) / 0.3) ** 2)
    M = np.zeros((len(idx), 12), dtype=np.float32)
    M[np.arange(len(idx)), nearest.astype(int) % 12] = weight
    return spec @ M


def _templates() -> np.ndarray:
    """Plantillas de 24 tríadas y 36 séptimas, normalizadas."""
    T = np.zeros((CHORD_COUNT, 12), dtype=np.float32)
    for r in range(12):
        T[r, [r, (r + 4) % 12, (r + 7) % 12]] = [1.0, 0.85, 0.95]
        T[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = [1.0, 0.85, 0.95]
        if CFG["tol7"]:
            for tpl, third in ((r, 4), (12 + r, 3)):
                for off, w in ((10, 1), (11, 1), (2, 0.8)):   # 7ª menor, 7ª mayor, 9ª
                    if (off - third) % 12 != 0:
                        T[tpl, (r + off) % 12] += CFG["tol7"] * w
        for tpl, third, seventh in (
            (24 + r, 4, 10),  # dominante
            (36 + r, 4, 11),  # mayor
            (48 + r, 3, 10),  # menor
        ):
            T[tpl, [r, (r + third) % 12, (r + 7) % 12, (r + seventh) % 12]] = [
                1.0, 0.85, 0.95, 1.5,
            ]
    return T / np.linalg.norm(T, axis=1, keepdims=True)


def estimate_key(mean_chroma: np.ndarray) -> tuple[int, bool]:
    """(tónica 0-11, es_menor)."""
    best, best_score = (0, False), -2.0
    for r in range(12):
        for minor, prof in ((False, _KK_MAJOR), (True, _KK_MINOR)):
            sc = np.corrcoef(mean_chroma, np.roll(prof, r))[0, 1]
            if sc > best_score:
                best, best_score = (r, minor), sc
    return best


def _diatonic(root: int, minor: bool) -> set[int]:
    """Índices de plantilla (0-11 mayores, 12-23 menores) que pertenecen a la tonalidad."""
    rel = (root + 3) % 12 if minor else root  # tónica de la mayor relativa
    s = {rel, (rel + 5) % 12, (rel + 7) % 12,
         12 + (rel + 2) % 12, 12 + (rel + 4) % 12, 12 + (rel + 9) % 12}
    if minor:
        s.add((root + 7) % 12)  # V mayor (dominante armónica)
    return s


def _viterbi(emit: np.ndarray, stay: float = 0.93, extra: np.ndarray | None = None) -> np.ndarray:
    """emit[T,S] = log-probabilidades. Cambiar de estado cuesta más que quedarse.
    `extra[t]` (>=0) es un coste adicional por cambiar justo antes del intervalo t."""
    T, S = emit.shape
    ls, lw = np.log(stay), np.log((1 - stay) / (S - 1))
    delta = emit[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        k = int(np.argmax(delta))
        switch = delta[k] + lw - (0.0 if extra is None else extra[t])
        keep = delta + ls
        use_keep = keep >= switch
        back[t] = np.where(use_keep, np.arange(S), k)
        delta = np.where(use_keep, keep, switch) + emit[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _name(i: int) -> str:
    if i < 12:
        return NOTES[i]
    if i < TRIAD_COUNT:
        return NOTES[i - 12] + "m"
    if i < 36:
        return NOTES[i - TRIAD_COUNT] + "7"
    if i < 48:
        return NOTES[i - 36] + "maj7"
    return NOTES[i - 48] + "m7"


def detect(audio: Path, min_seconds: float | None = None) -> dict:
    return detect_signal(decode(audio), min_seconds)


def _extend(beats: np.ndarray, dur: float) -> np.ndarray:
    """Prolonga la rejilla de pulsos hacia atrás (hasta 0) y hacia adelante (hasta el final)."""
    period = float(np.median(np.diff(beats)))
    before = beats[0] - period * np.arange(int(beats[0] // period), 0, -1)
    after = beats[-1] + period * np.arange(1, int((dur - beats[-1]) // period) + 1)
    return np.concatenate([before, beats, after])


def _decode_grid(beats: np.ndarray, dur: float) -> np.ndarray:
    """Rejilla para decidir acordes: pulsos con periodo entre 0.35 y 0.8 s (evita octavas del tempo)."""
    grid = beats
    while len(grid) > 1 and np.median(np.diff(grid)) > 0.8:      # tempo lento: subdivide
        mids = (grid[:-1] + grid[1:]) / 2
        grid = np.sort(np.concatenate([grid, mids]))
    while len(grid) > 3 and np.median(np.diff(grid)) < 0.35:     # tempo muy rápido: junta
        grid = grid[::2]
    return _extend(grid, dur) if len(grid) > 1 else grid


def detect_signal(
    y: np.ndarray,
    min_seconds: float | None = None,
    sr: int = SR,
    smooth: int = 3,
    stay_frame: float = 0.95,
    stay_beat: float = 0.8,
    use_beats: bool = True,
    hpss: bool = True,
    repeat: bool = False,
) -> dict:
    dur = len(y) / sr
    ch, energy = chroma(y, sr, hpss)
    k = np.hanning(smooth + 2)[1:-1].astype(np.float32)
    k /= k.sum()
    ch = np.stack([np.convolve(ch[:, c], k, mode="same") for c in range(12)], axis=1)

    hop_s = HOP / sr
    center = (np.arange(len(ch)) * HOP + N_FFT / 2) / sr

    # ---- ritmo
    rh = rhythm.analyze(y, sr) if use_beats else {"bpm": 0.0, "beats": np.zeros(0)}
    beats = rh["beats"]
    beat_mode = len(beats) >= 8 and rh["bpm"] > 0

    # ---- intervalos de decisión: un acorde por pulso (o por cuadro si no hay pulsos fiables)
    if beat_mode:
        bounds = np.concatenate([_decode_grid(beats, dur), [dur]])
        es = np.concatenate([[0.0], np.cumsum(energy)])
        lo = np.searchsorted(center, bounds[:-1])
        hi = np.maximum(np.searchsorted(center, bounds[1:]), lo + 1)
        hi = np.minimum(hi, len(ch))
        lo = np.minimum(lo, hi - 1)
        cnt = (hi - lo)[:, None]

        def aggregate(m):  # promedio de los cuadros que caen en cada intervalo
            cs = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(m, axis=0)])
            return (cs[hi] - cs[lo]) / cnt

        agg = aggregate(ch)
        agg_energy = (es[hi] - es[lo]) / cnt[:, 0]
        if repeat:
            agg = _nonlocal_smooth(agg)
        stay = CFG["stay"] if CFG["stay"] else stay_beat
        if min_seconds is None:
            # La cuadrícula ya limita los cambios; no borrar acordes de un solo pulso.
            min_seconds = 0.0
    else:
        bounds = np.concatenate([[0.0], center[:-1] + hop_s / 2, [dur]]) if len(ch) else np.array([0.0, dur])
        aggregate = lambda m: m  # noqa: E731
        agg, agg_energy, stay = ch, energy, stay_frame
        if min_seconds is None:
            min_seconds = 0.2

    # ---- decodificación de acordes
    key_root, key_minor = estimate_key(ch.sum(axis=0))
    norm = np.linalg.norm(agg, axis=1, keepdims=True)
    normalized = agg / (norm + 1e-9)
    templates = _templates()
    scores = normalized @ templates[:TRIAD_COUNT].T
    seventh_scores = normalized @ templates[TRIAD_COUNT:].T
    for i in _diatonic(key_root, key_minor):               # pequeño empuje a los acordes de la tonalidad
        scores[:, i] += 0.04
        root = i % 12
        seventh_indices = (root, 12 + root) if i < 12 else (24 + root,)
        seventh_scores[:, seventh_indices] += 0.04
    if CFG["bass"]:                                        # evidencia del bajo: ¿la nota grave es la raíz?
        bn = aggregate(bass_chroma(y, len(ch), sr))
        bn = bn / (bn.sum(axis=1, keepdims=True) + 1e-9)
        B = np.zeros((TRIAD_COUNT, 12), dtype=np.float32)  # apoyo de las notas del bajo al acorde
        seventh_B = np.zeros((SEVENTH_COUNT, 12), dtype=np.float32)
        for r in range(12):
            for tpl, third in ((r, 4), (12 + r, 3)):
                B[tpl, r] = 1.0                        # raíz
                B[tpl, (r + third) % 12] = CFG["b3"]   # tercera (acorde invertido, ej. C/E)
                B[tpl, (r + 7) % 12] = CFG["b5"]       # quinta (ej. C/G)
            for tpl, third in (
                (r, 4), (12 + r, 4), (24 + r, 3),
            ):
                seventh_B[tpl, r] = 1.0
                seventh_B[tpl, (r + third) % 12] = CFG["b3"]
                seventh_B[tpl, (r + 7) % 12] = CFG["b5"]
        scores += CFG["bass"] * bn @ B.T
        seventh_scores += CFG["bass"] * bn @ seventh_B.T
    silent = agg_energy < 0.15 * np.median(agg_energy)
    scores = np.concatenate([scores, np.zeros((len(scores), 1), dtype=np.float32)], axis=1)
    scores[silent, :TRIAD_COUNT] = 0.0
    scores[silent, NO_CHORD_STATE] = 1.0                    # estado "sin acorde"

    logits = 10.0 * scores
    mx = logits.max(axis=1, keepdims=True)
    emit = logits - (mx + np.log(np.exp(logits - mx).sum(axis=1, keepdims=True)))  # log-softmax
    path = _viterbi(emit, stay)
    classified_path = path.copy()
    for t, triad in enumerate(path):
        if triad >= TRIAD_COUNT:
            continue
        root = int(triad) % 12
        candidates = (root, 12 + root) if triad < 12 else (24 + root,)
        values = seventh_scores[t, candidates]
        order = np.argsort(values)[::-1]
        best_index = int(order[0])
        best_score = float(values[best_index])
        second_score = float(values[order[1]]) if len(order) > 1 else float("-inf")
        if (
            best_score - float(scores[t, triad]) >= SEVENTH_MIN_MARGIN
            and best_score - second_score >= SEVENTH_MIN_SEPARATION
        ):
            classified_path[t] = SEVENTH_STATE_START + candidates[best_index]
    full = _extend(beats, dur) if beat_mode else None

    segs: list[list] = []  # [inicio, fin, estado, intervalo_ini, intervalo_fin]
    for i, s in enumerate(classified_path):
        a, b = float(bounds[i]), float(bounds[i + 1])
        if segs and segs[-1][2] == s:
            segs[-1][1], segs[-1][4] = b, i + 1
        else:
            segs.append([a, b, int(s), i, i + 1])

    # Los tramos muy cortos se absorben en el vecino (evita parpadeos)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, sg in enumerate(segs):
            if sg[1] - sg[0] < min_seconds:
                j = i - 1 if i > 0 else i + 1
                if j == i - 1:
                    segs[j][1], segs[j][4] = sg[1], sg[4]
                else:
                    segs[j][0], segs[j][3] = sg[0], sg[3]
                del segs[i]
                changed = True
                break
        merged: list[list] = []
        for sg in segs:
            if merged and merged[-1][2] == sg[2]:
                merged[-1][1], merged[-1][4] = sg[1], sg[4]
            else:
                merged.append(sg)
        segs = merged

    out = []
    for a, b, s, i0, i1 in segs:
        if s == NO_CHORD_STATE:
            continue
        is_seventh = s >= SEVENTH_STATE_START
        chord_index = s - 1 if is_seventh else s
        triad_index = chord_index
        if is_seventh:
            seventh_index = chord_index - TRIAD_COUNT
            root = seventh_index % 12
            triad_index = root if seventh_index < 24 else 12 + root
        mean = scores[i0:i1, :TRIAD_COUNT].mean(axis=0)
        alts = [_name(int(k)) for k in np.argsort(mean)[::-1] if int(k) != triad_index][:3]
        sc = scores[i0:i1, :TRIAD_COUNT]
        if is_seventh:
            all_scores = np.concatenate(
                [sc, seventh_scores[i0:i1]], axis=1,
            )
            chosen = chord_index
        else:
            all_scores = sc
            chosen = s
        margin = float(
            np.mean(all_scores[:, chosen] - np.max(np.delete(all_scores, chosen, axis=1), axis=1))
        )
        conf = float(np.clip((margin + 0.1) / 0.6, 0.0, 1.0))                          # 0 = dudoso, 1 = seguro
        out.append({"start": round(a, 2), "end": round(b, 2), "chord": _name(chord_index), "alts": alts,
                    "conf": round(conf, 2)})

    # ---- tempo, pulsos y primer tiempo del compás (4/4 estimado por dónde cambian los acordes)
    result = {"key": _name(key_root + (12 if key_minor else 0)), "chords": out,
              "bpm": 0.0, "beats": [], "downbeat": 0}
    if beat_mode:
        counts = np.zeros(4)
        for c in out[1:]:
            counts[int(np.argmin(np.abs(full - c["start"]))) % 4] += 1
        result.update(
            bpm=rh["bpm"],
            beats=[round(float(b), 3) for b in full],
            downbeat=int(np.argmax(counts)) if counts.sum() else 0,
        )
    return result
