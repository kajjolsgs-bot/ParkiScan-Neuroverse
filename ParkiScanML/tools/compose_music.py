"""
ParkiScan Neuroverse — procedural music & SFX composer (offline, deterministic).

Generates four loopable instrumental tracks and a small gameplay SFX library as 16-bit WAV files.
This is the documented FALLBACK used because no music-generation API/model was available in the build
environment (see README "Audio"). Everything here is real composition: chord progressions, melodies with
motif development, bass lines, rhythm parts, and a light stereo reverb — no noise beds, no random beeps.

Usage:
    python tools/compose_music.py --out <dir_for_music> --sfx <dir_for_sfx>
"""
import argparse
import os
import wave

import numpy as np

SR = 44100
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------------------------- helpers
def mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def env_adsr(n, a=0.01, d=0.1, s=0.7, r=0.2, sr=SR):
    a_n, d_n, r_n = int(a * sr), int(d * sr), int(r * sr)
    s_n = max(0, n - a_n - d_n - r_n)
    e = np.concatenate([
        np.linspace(0, 1, max(a_n, 1), endpoint=False),
        np.linspace(1, s, max(d_n, 1), endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0, max(r_n, 1)),
    ])
    if len(e) < n:
        e = np.pad(e, (0, n - len(e)))
    return e[:n]


def lowpass(x, cutoff):
    """One-pole low-pass (gentle) applied forward/backward (zero phase)."""
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):  # small signals only (single notes)
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def lp_fast(x, cutoff):
    # vectorised one-pole via scipy-free cumulative filter (good enough for pads/whole tracks)
    from scipy.signal import lfilter
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def additive(freq, dur, harmonics, decay=None, vibrato=0.0, vib_rate=5.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    phase_mod = vibrato * np.sin(2 * np.pi * vib_rate * t) * np.clip(t / 0.25, 0, 1)
    out = np.zeros(n)
    for k, amp in enumerate(harmonics, start=1):
        f = freq * k
        if f > SR / 2.2:
            break
        e = np.exp(-t * decay * k) if decay else 1.0
        out += amp * e * np.sin(2 * np.pi * f * t + k * phase_mod)
    return out


def pluck(freq, dur, bright=0.5, buzz=0.0):
    """Karplus-Strong string (sitar-like 'buzz' adds a soft nonlinearity: jawari)."""
    n = int(dur * SR)
    period = max(2, int(SR / freq))
    buf = RNG.uniform(-1, 1, period)
    buf = lp_fast(buf, 2000 + 6000 * bright)  # shape the excitation (no audible noise after a few ms)
    out = np.empty(n)
    idx = 0
    damp = 0.996
    for i in range(n):
        v = buf[idx]
        nxt = buf[(idx + 1) % period]
        nv = damp * 0.5 * (v + nxt)
        if buzz:
            nv = nv + buzz * 0.02 * np.tanh(6 * nv)
        buf[idx] = nv
        out[i] = v
        idx = (idx + 1) % period
    out *= env_adsr(n, 0.002, 0.05, 1.0, 0.05)
    return out


def bell(freq, dur, bright=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    mod = np.sin(2 * np.pi * freq * 3.5 * t) * 1.2 * bright * np.exp(-t * 6)
    return np.sin(2 * np.pi * freq * t + mod) * np.exp(-t * 2.2) * np.clip(t / 0.003, 0, 1)


def marimba(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) * np.exp(-t * 5) + 0.35 * np.sin(2 * np.pi * freq * 4 * t) * np.exp(-t * 18)
    return s * np.clip(t / 0.002, 0, 1)


def pad(freq, dur, bright=0.35):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (-0.12, 0.0, 0.11):  # gentle chorus
        f = freq * 2 ** (det / 12)
        for k in range(1, 7):
            out += (bright ** (k - 1)) / k * np.sin(2 * np.pi * f * k * t + k)
    return out / 3 * env_adsr(n, min(0.6, dur * 0.3), 0.3, 0.85, min(0.8, dur * 0.3))


def flute(freq, dur, glide_from=None, glide_t=0.18):
    """Bansuri-like: sine + soft 2nd/3rd partials, delayed vibrato, optional meend (glide) into the note."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.full(n, freq)
    if glide_from:
        g = np.clip(t / glide_t, 0, 1)
        f = glide_from * (freq / glide_from) ** (g * g * (3 - 2 * g))
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip((t - 0.25) / 0.4, 0, 1)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    s = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    return s * env_adsr(n, 0.06, 0.1, 0.85, min(0.25, dur * 0.4))


def membrane(freq, dur, bend=1.6, decay=9.0, click=0.15):
    """Tabla/tom-like pitched drum: decaying sine with a downward pitch bend and a tiny attack transient."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = freq * (1 + (bend - 1) * np.exp(-t * 30))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * decay)
    s += click * np.sin(2 * np.pi * freq * 5.1 * t) * np.exp(-t * 120)
    return s


def kick(dur=0.35):
    # soft, rounded "heartbeat" kick (higher, shorter and quieter than a cinematic boom)
    return membrane(75, min(dur, 0.22), bend=1.6, decay=16, click=0.05) * 0.6


def place(buf, sig, start_s, gain=1.0, pan=0.0):
    i = int(start_s * SR)
    if i >= buf.shape[1]:
        return
    sig = sig[: buf.shape[1] - i]
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + len(sig)] += sig * gain * l
    buf[1, i:i + len(sig)] += sig * gain * r


def reverb(stereo, mix=0.22, room=0.82):
    """Small Schroeder reverb (4 combs + 2 allpasses per channel)."""
    from scipy.signal import lfilter
    out = np.zeros_like(stereo)
    for ch in range(2):
        x = stereo[ch]
        y = np.zeros_like(x)
        for j, d in enumerate((1557, 1617, 1491, 1422)):
            d = d + ch * 23
            b = np.zeros(d + 1); b[0] = 1
            a = np.zeros(d + 1); a[0] = 1; a[d] = -room
            y += lfilter(b, a, x)
        y /= 4
        for d in (225, 556):
            b = np.zeros(d + 1); b[0] = -0.5; b[d] = 1
            a = np.zeros(d + 1); a[0] = 1; a[d] = -0.5
            y = lfilter(b, a, y)
        out[ch] = lp_fast(y, 6000)
    return stereo * (1 - mix) + out * mix


def finish(stereo, peak=0.82, loop_xfade=1.5):
    """Make the loop seamless: fold the reverb tail back onto the start, then normalise."""
    tail = int(loop_xfade * SR)
    body = stereo[:, :-tail].copy()
    body[:, :tail] += stereo[:, -tail:] * np.linspace(1, 0, tail)
    # gentle 2nd-order high-pass at 70 Hz: keeps warmth, removes sub rumble that reads as a "boom" on speakers
    from scipy.signal import butter, sosfiltfilt
    body = sosfiltfilt(butter(2, 70, btype="high", fs=SR, output="sos"), body, axis=1)
    m = np.max(np.abs(body)) + 1e-9
    return body / m * peak


def write_wav(path, stereo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.clip(stereo.T, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2 if stereo.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def new_buf(seconds):
    return np.zeros((2, int(seconds * SR)))


# ----------------------------------------------------------------------------------------------- tracks
def melody_from_motif(motif, scale, bars, variation, seed):
    """Develop a motif (scale-degree, length-in-beats) over bars: repeat, transpose, invert tails."""
    rng = np.random.default_rng(seed)
    notes = []
    for b in range(bars):
        m = list(motif)
        kind = variation[b % len(variation)]
        if kind == "up":
            m = [(d + 2, l) for d, l in m]
        elif kind == "down":
            m = [(d - 1, l) for d, l in m]
        elif kind == "invert":
            m = m[:2] + [(m[0][0] - (d - m[0][0]), l) for d, l in m[2:]]
        elif kind == "rest_end":
            m = m[:-1] + [(None, m[-1][1])]
        elif kind == "ornament" and len(m) > 1:
            d0, l0 = m[1]
            m = [m[0], (d0, l0 / 2), (d0 + 1, l0 / 2)] + m[2:]
        if rng.random() < 0.15:
            m[-1] = (m[-1][0] if m[-1][0] is None else m[-1][0] + rng.choice([-1, 1]), m[-1][1])
        notes.append(m)
    return notes


def deg_to_midi(deg, scale, root):
    if deg is None:
        return None
    octv, k = divmod(deg, len(scale))
    return root + 12 * octv + scale[k]


def track_entry():
    """Entry / Level 1 — peppy, curious, peaceful futuristic exploration. 104 BPM, C major (Lydian colour)."""
    bpm = 104; beat = 60 / bpm; bars = 28; L = bars * 4 * beat + 1.5
    buf = new_buf(L)
    scale = [0, 2, 4, 6, 7, 9, 11]  # Lydian: bright, wondering
    root = 60
    chords = [[0, 4, 7, 11], [9 - 12, 0, 4, 7], [5 - 12, 9 - 12, 0, 4], [7 - 12, 11 - 12, 2, 6]]  # Cmaj7 Am7 Fmaj7 G(add#11)
    for bar in range(bars):
        t0 = bar * 4 * beat
        ch = chords[bar % 4]
        for n in ch:
            place(buf, pad(mtof(root + n - 12), 4 * beat + 0.4, 0.3), t0, 0.08, pan=-0.2)
        # marimba arpeggio in 8ths (pattern varies every 8 bars)
        patt = [0, 2, 1, 3, 2, 1, 3, 2] if (bar // 8) % 2 == 0 else [0, 1, 2, 3, 2, 3, 1, 2]
        for i, p in enumerate(patt):
            place(buf, marimba(mtof(root + ch[p] + 12), 0.6), t0 + i * beat / 2, 0.16, pan=0.35)
        # plucky bass on 1 and the "and" of 2
        bn = root - 24 + ch[0]
        place(buf, pluck(mtof(bn), 0.8, 0.3), t0, 0.32)
        place(buf, pluck(mtof(bn + 7), 0.5, 0.3), t0 + 1.5 * beat, 0.22)
        place(buf, pluck(mtof(bn), 0.5, 0.3), t0 + 3 * beat, 0.22)
        # light rhythm (from bar 4): soft kick + tonal "tick" on the off-beats
        if bar >= 4:
            for k in (0, 2):
                place(buf, kick(), t0 + k * beat, 0.18)
            for k in range(4):
                place(buf, bell(mtof(96), 0.12, 0.4) * 0.5, t0 + (k + 0.5) * beat, 0.05, pan=0.5)
    # bell melody (enters bar 8)
    motif = [(4, 1), (6, 0.5), (7, 0.5), (8, 1.5), (6, 0.5)]
    mel = melody_from_motif(motif, scale, 16, ["base", "up", "ornament", "rest_end", "base", "invert", "down", "rest_end"], 3)
    for b, m in enumerate(mel):
        t = (8 + b) * 4 * beat
        for d, l in m:
            mm = deg_to_midi(d, scale, root)
            if mm is not None:
                place(buf, bell(mtof(mm + 12), max(0.8, l * beat * 1.6)), t, 0.11, pan=-0.3)
            t += l * beat
    return finish(reverb(buf, 0.25))


def track_reactor():
    """Level 2 Neural Reactor — bright, peppy sci-fi groove, pleasant and uplifting. 112 BPM, D major."""
    bpm = 112; beat = 60 / bpm; bars = 32; L = bars * 4 * beat + 1.5
    buf = new_buf(L)
    root = 62  # D
    scale = [0, 2, 4, 5, 7, 9, 11]  # major
    prog = [[0, 4, 7], [-5, -1, 2], [-3, 0, 4], [-7, -3, 0]]  # D A Bm G (I V vi IV)
    for bar in range(bars):
        t0 = bar * 4 * beat
        ch = prog[(bar // 2) % 4]
        for n in ch:
            place(buf, pad(mtof(root + n - 12), 4 * beat + 0.3, 0.45), t0, 0.07, pan=0.0)
        # driving 8th-note bass (octave pulses)
        for i in range(8):
            nb = root - 24 + ch[0] + (12 if i % 2 else 0)
            place(buf, pluck(mtof(nb), 0.3, 0.45), t0 + i * beat / 2, 0.17)
        # 16th arpeggio synth (from bar 4)
        if bar >= 4:
            arp = [0, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 0, 2]
            for i, a in enumerate(arp):
                place(buf, marimba(mtof(root + ch[a] + 12), 0.25), t0 + i * beat / 4, 0.06, pan=0.45 if i % 2 else -0.45)
        # light groove: soft kick on 1 & 3, bright woodblock-like ticks on 2 & 4 (no toms, no impacts)
        if bar >= 2:
            place(buf, kick(), t0, 0.2); place(buf, kick(), t0 + 2 * beat, 0.16)
            place(buf, membrane(880, 0.08, 1.0, 40, click=0.2), t0 + 1 * beat, 0.05, pan=-0.3)
            place(buf, membrane(880, 0.08, 1.0, 40, click=0.2), t0 + 3 * beat, 0.05, pan=0.3)
    motif = [(7, 1), (6, 0.5), (4, 0.5), (5, 1), (2, 1)]
    mel = melody_from_motif(motif, scale, 16, ["base", "rest_end", "up", "base", "invert", "rest_end", "down", "base"], 5)
    for b, m in enumerate(mel):
        t = (12 + b) * 4 * beat
        for d, l in m:
            mm = deg_to_midi(d, scale, root)
            if mm is not None:
                place(buf, flute(mtof(mm), l * beat * 0.95) * 0.9, t, 0.13, pan=0.15)
            t += l * beat
    return finish(reverb(buf, 0.22))


def track_assessment():
    """Level 3 assessment — calm, reassuring, focused, unobtrusive under speech. 66 BPM, F major."""
    bpm = 66; beat = 60 / bpm; bars = 18; L = bars * 4 * beat + 2.0
    buf = new_buf(L)
    root = 65
    prog = [[0, 4, 7, 11], [-3, 0, 4, 7], [-7, -3, 0, 4], [-5, -1, 2, 7]]  # Fmaj7 Dm7 Bbmaj7 C(sus)
    scale = [0, 2, 4, 5, 7, 9, 11]
    for bar in range(bars):
        t0 = bar * 4 * beat
        ch = prog[bar % 4]
        for n in ch:
            place(buf, pad(mtof(root + n - 12), 4 * beat + 0.8, 0.22), t0, 0.07, pan=-0.1)
        place(buf, additive(mtof(root - 24 + ch[0]), 4 * beat, [1, 0.3, 0.1]) * env_adsr(int(4 * beat * SR), 0.4, 0.4, 0.8, 0.8), t0, 0.12)
        # sparse soft "piano" (bell, low brightness) — two notes per bar
        for k, beat_pos in enumerate((0, 2.5)):
            nn = root + ch[(bar + k) % 4] + 12
            place(buf, bell(mtof(nn), 2.4, 0.35), t0 + beat_pos * beat, 0.06, pan=0.3 if k else -0.3)
    return finish(reverb(buf, 0.32), peak=0.6)


def track_india():
    """Return to India — warm, uplifting, raga-Yaman-inspired: tanpura drone, bansuri melody with meend,
    sitar-like plucks and a restrained tabla-like teentaal groove. 84 BPM, Sa = D."""
    bpm = 84; beat = 60 / bpm; cycles = 8; L = cycles * 16 * beat + 1.5
    buf = new_buf(L)
    sa = 62  # D
    yaman = [0, 2, 4, 6, 7, 9, 11]  # Sa Re Ga Ma(tivra) Pa Dha Ni
    # tanpura: Pa - Sa' - Sa' - Sa (low) cycle, long rich decays
    tan = [sa - 5, sa, sa, sa - 12]
    for c in range(int(L / (4 * beat)) + 1):
        for k, n in enumerate(tan):
            s = additive(mtof(n - 12), 3.2, [1, 0.6, 0.5, 0.35, 0.25, 0.2, 0.12], decay=0.25)
            place(buf, s * env_adsr(len(s), 0.08, 0.3, 0.8, 1.2), c * 4 * beat + k * beat, 0.07, pan=-0.35 + 0.2 * k)
    # sitar-like plucks: alap-style phrases rising Ni Re Ga Ma# Pa ...
    sitar_phr = [[-1, 1, 2], [2, 3, 4, 3], [4, 5, 6, 7], [6, 4, 3, 2, 1, 0]]
    t = 0.0
    for c in range(cycles):
        ph = sitar_phr[c % 4]
        for i, d in enumerate(ph):
            mm = deg_to_midi(d, yaman, sa)
            place(buf, pluck(mtof(mm), 1.4, 0.6, buzz=1.0), c * 16 * beat + i * 2 * beat, 0.14, pan=0.35)
    # bansuri melody from cycle 2, with meend glides into long notes
    phrases = [[(4, 2), (6, 1), (7, 1), (9, 3), (7, 1)], [(9, 1), (8, 1), (7, 2), (6, 1), (4, 3)],
               [(7, 2), (8, 1), (9, 1), (11, 2), (9, 2)], [(9, 1), (7, 1), (6, 1), (4, 1), (2, 2), (0, 2)]]
    for c in range(2, cycles):
        tt = c * 16 * beat
        prev = None
        for d, l in phrases[(c - 2) % 4]:
            mm = deg_to_midi(d, yaman, sa)
            f = mtof(mm)
            place(buf, flute(f, l * beat * 1.05, glide_from=(mtof(prev) if prev and l >= 2 else None)), tt, 0.16, pan=-0.1)
            prev = mm
            tt += l * beat
    # tabla-like teentaal theka (16 beats), restrained: dha dhin dhin dha | dha dhin dhin dha | dha tin tin ta | ta dhin dhin dha
    bols = ["dha", "dhin", "dhin", "dha", "dha", "dhin", "dhin", "dha", "dha", "tin", "tin", "ta", "ta", "dhin", "dhin", "dha"]
    for c in range(1, cycles):
        for i, b in enumerate(bols):
            t0 = c * 16 * beat + i * beat
            if b in ("dha", "dhin"):
                place(buf, membrane(mtof(sa - 24), 0.4, 1.15, 8), t0, 0.13)  # bayan (bass, kept soft)
            if b in ("dha", "ta", "tin", "dhin"):
                place(buf, membrane(mtof(sa + 12), 0.25, 1.05, 18, click=0.25), t0, 0.11 if b != "dha" else 0.14, pan=0.2)  # dayan
            if i % 4 == 3:
                place(buf, membrane(mtof(sa + 12), 0.15, 1.05, 22, click=0.2), t0 + beat / 2, 0.06, pan=0.2)
    return finish(reverb(buf, 0.28))


# ----------------------------------------------------------------------------------------------- SFX
def mono_finish(x, peak=0.7):
    fade = int(0.01 * SR)
    x = x.copy()
    x[-fade:] *= np.linspace(1, 0, fade)
    return np.vstack([x, x]) / (np.max(np.abs(x)) + 1e-9) * peak


def sfx_library():
    out = {}
    c5, e5, g5, c6 = (mtof(m) for m in (72, 76, 79, 84))
    # correct: warm two-note bell (major third up)
    x = np.zeros(int(0.9 * SR)); b1 = bell(c5 * 2, 0.9, 0.6); b2 = bell(e5 * 2, 0.8, 0.6)
    x[:len(b1)] += b1; x[int(0.09 * SR):int(0.09 * SR) + len(b2)] += b2[: len(x) - int(0.09 * SR)]
    out["Correct"] = mono_finish(x, 0.55)
    # incorrect / retry: soft neutral descending pair (no buzzer)
    x = np.zeros(int(0.7 * SR)); a = additive(mtof(67), 0.35, [1, 0.25]) * env_adsr(int(0.35 * SR), 0.01, 0.1, 0.5, 0.2)
    b = additive(mtof(64), 0.4, [1, 0.25]) * env_adsr(int(0.4 * SR), 0.01, 0.1, 0.5, 0.25)
    x[:len(a)] += a; x[int(0.18 * SR):int(0.18 * SR) + len(b)] += b
    out["Incorrect"] = mono_finish(x, 0.4)
    # task complete: short uplifting flourish (arpeggio + sustained chord)
    x = np.zeros(int(1.8 * SR))
    for i, m in enumerate((72, 76, 79, 84)):
        s = bell(mtof(m) * 2, 1.4, 0.7); st = int(i * 0.08 * SR); x[st:st + len(s)] += s[: len(x) - st] * 0.8
    for m in (60, 64, 67):
        s = pad(mtof(m), 1.8, 0.35); x += s[: len(x)] * 0.35
    out["TaskComplete"] = mono_finish(x, 0.6)
    # memory fragment / node connect: crystalline
    x = np.zeros(int(1.0 * SR))
    for i, m in enumerate((88, 95)):
        s = bell(mtof(m), 0.9, 1.2); st = int(i * 0.05 * SR); x[st:st + len(s)] += s[: len(x) - st]
    out["Crystal"] = mono_finish(x, 0.42)
    # repair lock: clean mechanical click-clack + energy rise (tonal)
    x = np.zeros(int(0.8 * SR))
    for i, f in enumerate((900, 600)):
        s = membrane(f, 0.08, 1.0, 60, click=0.4); st = int(i * 0.06 * SR); x[st:st + len(s)] += s
    rise = additive(1, 0.6, [1]) * 0
    t = np.arange(int(0.6 * SR)) / SR
    rise = np.sin(2 * np.pi * np.cumsum(220 + 440 * t / 0.6) / SR) * env_adsr(len(t), 0.05, 0.2, 0.6, 0.2) * 0.4
    x[int(0.15 * SR):int(0.15 * SR) + len(rise)] += rise
    out["RepairLock"] = mono_finish(x, 0.5)
    # core restoration: rich short flourish
    x = np.zeros(int(2.2 * SR))
    for m in (48, 55, 60, 64, 67, 72):
        s = pad(mtof(m), 2.2, 0.5); x += s[: len(x)] * 0.25
    for i, m in enumerate((79, 84, 88, 91)):
        s = bell(mtof(m), 1.5, 0.8); st = int((0.2 + i * 0.1) * SR); x[st:st + len(s)] += s[: len(x) - st] * 0.6
    out["CoreRestore"] = mono_finish(x, 0.6)
    # signal / traffic light cue: brief, clear, non-startling
    out["SignalCue"] = mono_finish(bell(mtof(81), 0.45, 0.5), 0.35)
    # transition: airy high shimmer (no low frequencies, no swell/boom, no noise)
    x = np.zeros(int(1.4 * SR))
    for i, m in enumerate((84, 88, 91, 96)):
        s = bell(mtof(m), 1.1, 0.5) * np.clip(np.arange(int(1.1 * SR)) / (0.03 * SR), 0, 1)  # soft attack
        st = int(i * 0.07 * SR); x[st:st + len(s)] += s[: len(x) - st] * (0.9 - 0.12 * i)
    out["Transition"] = mono_finish(lowpass(x, 7000), 0.22)
    # level start: gentle, happy harp-like rising arpeggio (C major pentatonic, mid-high register)
    x = np.zeros(int(2.0 * SR))
    for i, m in enumerate((72, 74, 76, 79, 81, 84)):
        s = pluck(mtof(m), 1.4, 0.35); st = int(i * 0.09 * SR); x[st:st + len(s)] += s[: len(x) - st] * 0.8
    for m in (60, 67, 76):
        s = pad(mtof(m), 2.0, 0.25); x += s[: len(x)] * 0.18
    out["LevelStart"] = mono_finish(x, 0.3)
    # NIA attention: subtle soft blip
    out["NiaCue"] = mono_finish(bell(mtof(84), 0.3, 0.3), 0.22)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sfx", required=True)
    a = ap.parse_args()
    tracks = {
        "Entry_Level1_Music": track_entry,
        "Level2_NeuralReactor_Music": track_reactor,
        "Level3_Assessment_Music": track_assessment,
        "ReturnToIndia_IndianInspired_Music": track_india,
    }
    for name, fn in tracks.items():
        s = fn()
        write_wav(os.path.join(a.out, name + ".wav"), s)
        print(f"music {name}: {s.shape[1] / SR:.1f}s")
    for name, s in sfx_library().items():
        write_wav(os.path.join(a.sfx, "SFX_" + name + ".wav"), s)
        print(f"sfx {name}: {s.shape[1] / SR:.2f}s")


if __name__ == "__main__":
    main()
