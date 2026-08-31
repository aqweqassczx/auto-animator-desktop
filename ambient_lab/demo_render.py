"""Эскиз движка dark ambient: рендер трека из кода, без DAW и без AI-аудио.

Рецепт слоёв взят из разборов жанра (øneheart / snowfall-стиль):
  - пэд: стек расстроенных пил -> лоупасс -> медленная огибающая, длинные аккорды
  - арп: чистый синус по нотам аккорда -> пинг-понг дилей 3/16 -> большой реверб
  - бас: пила -2 октавы, один голос, глайд, срезан верх (рис-бас из фонка)
  - лид: тихий "расстроенный" синус с вибрато и дрейфом питча
  - текстура: плёночный шум + редкий винильный крэкл (заполняет верх спектра)
  - реверб: свёртка с синтетическим IR ~8 секунд, тёмный хвост

Запуск:  python3 ambient_lab/demo_render.py [путь_к_выходному_файлу.mp3|.wav]
Зависимости: numpy, scipy, soundfile (pip install numpy scipy soundfile)
"""

import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve, sawtooth

SR = 44100
BPM = 72
BEAT = 60.0 / BPM          # 0.833 c
BAR = 4 * BEAT
CHORD_LEN = 2 * BAR        # каждый аккорд живёт 2 такта
SECTION = 4 * CHORD_LEN    # секция = проход прогрессии (8 тактов)

rng = np.random.default_rng(7)


def midi_to_freq(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def lowpass(x, cutoff, order=2):
    sos = butter(order, cutoff / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, x, axis=0) if x.ndim > 1 else sosfilt(sos, x)


def highpass(x, cutoff, order=2):
    sos = butter(order, cutoff / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, x, axis=0) if x.ndim > 1 else sosfilt(sos, x)


def place(buf, start_sec, chunk):
    """Подмешать stereo-фрагмент в общий буфер начиная с start_sec."""
    i0 = int(start_sec * SR)
    i1 = min(i0 + len(chunk), len(buf))
    if i1 > i0:
        buf[i0:i1] += chunk[: i1 - i0]


def env_asr(n, attack, release, sustain_level=1.0):
    """Огибающая attack-sustain-release косинусными скатами."""
    e = np.full(n, sustain_level)
    a = min(int(attack * SR), n)
    r = min(int(release * SR), n - a)
    if a > 0:
        e[:a] = sustain_level * 0.5 * (1 - np.cos(np.linspace(0, np.pi, a)))
    if r > 0:
        e[n - r:] = e[n - r] * 0.5 * (1 + np.cos(np.linspace(0, np.pi, r)))
    return e


# --- Гармония: C minor, прогрессия i9 - VI - III(add9) - v7, open voicing ---
# (верхние ноты раскинуты на октаву, низ не трогаем — как в туториалах)
PROGRESSION = [
    dict(pad=[48, 55, 58, 63, 67, 74], bass=36, arp=[72, 74, 75, 79, 82]),  # Cm9
    dict(pad=[44, 51, 56, 60, 67, 72], bass=32, arp=[72, 75, 79, 80]),      # Abmaj7
    dict(pad=[51, 58, 62, 67, 70, 77], bass=39, arp=[74, 75, 79, 82]),      # Ebmaj9
    dict(pad=[43, 50, 55, 58, 65, 70], bass=31, arp=[74, 77, 79, 82]),      # Gm7
]

# Ритм арпа на 8 долей аккорда: (доля, индекс ноты, велосити)
ARP_PATTERN_B = [(0.0, 0, 0.9), (1.5, 1, 0.7), (3.0, 2, 0.8),
                 (4.0, 3, 1.0), (5.5, 2, 0.6), (7.0, 1, 0.7)]
ARP_PATTERN_C = [(0.5, 1, 0.8), (2.0, 2, 0.7), (3.5, 3, 1.0),
                 (4.5, -1, 0.9), (6.0, 2, 0.6), (7.5, 0, 0.7)]


def render_pad_chord(notes, dur, cutoff):
    n = int((dur + 5.0) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    detunes = [-9, -4.5, 0, 4.5, 9]  # в центах
    for m in notes:
        f0 = midi_to_freq(m)
        for ch in range(2):  # свои фазы на канал -> настоящая стереоширина
            sig = np.zeros(n)
            for cents in detunes:
                f = f0 * 2 ** (cents / 1200)
                sig += sawtooth(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            sig /= len(detunes)
            # синус на октаву ниже — тело аккорда
            sig += 0.22 * np.sin(2 * np.pi * (f0 / 2) * t + rng.uniform(0, 2 * np.pi))
            out[:, ch] += sig
    out /= len(notes)
    out = lowpass(out, cutoff, order=2)
    out *= env_asr(n, attack=2.8, release=5.0)[:, None]
    return out


def render_bass_note(midi, prev_midi, dur):
    n = int((dur + 1.0) * SR)
    t = np.arange(n) / SR
    f_target = float(midi_to_freq(midi))
    f_prev = float(midi_to_freq(prev_midi))
    glide = int(0.10 * SR)
    freq = np.full(n, f_target)
    freq[:glide] = np.linspace(f_prev, f_target, glide)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = 0.55 * sawtooth(phase) + 0.6 * np.sin(phase)
    sig = lowpass(sig, 120, order=4)
    sig *= env_asr(n, attack=0.4, release=0.8)
    return np.column_stack([sig, sig])


def render_arp_note(midi, vel):
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    f = float(midi_to_freq(midi))
    sig = np.sin(2 * np.pi * f * t) + 0.08 * np.sin(4 * np.pi * f * t)
    env = np.exp(-t / 1.1)
    a = int(0.015 * SR)
    env[:a] *= np.linspace(0, 1, a)
    sig *= env * vel
    return np.column_stack([sig, sig]) * 0.5


def render_lead_note(midi, dur):
    """Тихий 'болезненный' лид: синус с вибрато и медленным дрейфом питча."""
    n = int((dur + 3.0) * SR)
    t = np.arange(n) / SR
    f0 = float(midi_to_freq(midi))
    vibrato = 8.0 * np.sin(2 * np.pi * 4.3 * t)            # ±8 центов
    drift = 6.0 * np.sin(2 * np.pi * 0.13 * t + 1.0)       # медленный дрейф
    f = f0 * 2 ** ((vibrato + drift) / 1200)
    phase = 2 * np.pi * np.cumsum(f) / SR
    sig = np.sin(phase) + 0.15 * np.sin(2 * phase)
    sig *= env_asr(n, attack=1.2, release=2.5)
    return np.column_stack([sig, sig]) * 0.5


def ping_pong_delay(x, delay_sec, feedback=0.55, taps=10):
    n = len(x)
    out = x.copy()
    d = int(delay_sec * SR)
    tap = x.copy()
    for k in range(1, taps + 1):
        tap = lowpass(tap, 6500) * feedback
        shifted = np.zeros_like(x)
        if k * d < n:
            shifted[k * d:] = tap[: n - k * d]
        pan = 0.75 if k % 2 else 0.25  # поочерёдно вправо/влево
        out[:, 0] += shifted[:, 0] * (1 - pan) * 2
        out[:, 1] += shifted[:, 1] * pan * 2
    return out


def make_reverb_ir(seconds=8.0, predelay=0.03):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    decay = np.exp(-6.907 * t / seconds)  # хвост до -60 дБ
    ir = rng.standard_normal((n, 2)) * decay[:, None]
    ir = lowpass(highpass(ir, 180), 5000)  # тёмный хвост, без грязи внизу
    pre = np.zeros((int(predelay * SR), 2))
    ir = np.vstack([pre, ir])
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    return ir


def reverb(x, ir, wet):
    w = np.column_stack([
        fftconvolve(x[:, 0], ir[:, 0]),
        fftconvolve(x[:, 1], ir[:, 1]),
    ])[: len(x) + int(10 * SR)]
    w *= wet * 3.0
    out = np.zeros((len(w), 2))
    out[: len(x)] = x
    return out + w


def main(out_path):
    total = 3 * SECTION + 8.0  # три секции + хвост реверба
    n_total = int(total * SR)

    pad = np.zeros((n_total, 2))
    bass = np.zeros((n_total, 2))
    arp = np.zeros((n_total, 2))
    lead = np.zeros((n_total, 2))

    section_cutoffs = [1000, 1400, 1800]  # пэд светлеет от секции к секции
    prev_bass = PROGRESSION[-1]["bass"]

    for s in range(3):
        for ci, chord in enumerate(PROGRESSION):
            t0 = s * SECTION + ci * CHORD_LEN
            place(pad, t0, render_pad_chord(chord["pad"], CHORD_LEN, section_cutoffs[s]))
            place(bass, t0, render_bass_note(chord["bass"], prev_bass, CHORD_LEN))
            prev_bass = chord["bass"]

            if s >= 1:  # арп вступает со второй секции
                pattern = ARP_PATTERN_B if s == 1 else ARP_PATTERN_C
                for beat_pos, idx, vel in pattern:
                    note = chord["arp"][idx if idx >= 0 else len(chord["arp"]) - 1]
                    place(arp, t0 + beat_pos * BEAT, render_arp_note(note, vel))

            if s == 2:  # лид — только в третьей секции, очень тихо
                place(lead, t0, render_lead_note(chord["pad"][-1], CHORD_LEN))

    # Текстура: плёночный шум с медленным дыханием + редкий крэкл
    t = np.arange(n_total) / SR
    hiss = rng.standard_normal((n_total, 2))
    hiss = lowpass(highpass(hiss, 2500), 10000)
    hiss *= (0.55 + 0.35 * np.sin(2 * np.pi * 0.1 * t))[:, None]
    crackle = np.zeros((n_total, 2))
    n_clicks = int(total * 2)
    for _ in range(n_clicks):
        pos = rng.integers(0, n_total - 200)
        amp = rng.uniform(0.2, 1.0)
        click = amp * np.exp(-np.arange(120) / 25.0)
        ch = rng.integers(0, 2)
        crackle[pos:pos + 120, ch] += click
    crackle = lowpass(crackle, 6000)

    # Дилей на арпе — 3/16 пинг-понг, фирменный приём жанра
    arp = ping_pong_delay(arp, delay_sec=0.75 * BEAT, feedback=0.5)

    ir = make_reverb_ir(8.0)
    pad_w = reverb(pad, ir, wet=0.45)
    arp_w = reverb(arp, ir, wet=0.9)
    lead_w = reverb(lead, ir, wet=0.9)

    n_out = len(pad_w)
    mix = np.zeros((n_out, 2))
    for sig, gain in [(pad_w, 0.34), (arp_w, 0.26), (lead_w, 0.10)]:
        mix[: len(sig)] += sig * gain
    for sig, gain in [(bass, 0.40), (hiss, 0.035), (crackle, 0.05)]:
        mix[: len(sig)] += sig * gain

    # Медленное "дыхание" всего микса — автоматизация громкости
    tt = np.arange(n_out) / SR
    mix *= (1.0 + 0.16 * np.sin(2 * np.pi * 0.07 * tt - 1.2))[:, None]

    mix = highpass(mix, 24)
    mix = np.tanh(mix * 1.1) / 1.1  # мягкая сатурация вместо жёсткого клипа

    fade_in = int(4.0 * SR)
    mix[:fade_in] *= (0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_in))))[:, None]
    fade_out = int(10.0 * SR)
    mix[-fade_out:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_out))))[:, None]

    mix *= 0.85 / np.abs(mix).max()

    # Числовая самопроверка вместо ушей: пики, RMS, спектральный баланс
    peak_db = 20 * np.log10(np.abs(mix).max())
    rms_db = 20 * np.log10(np.sqrt((mix ** 2).mean()))
    spec = np.abs(np.fft.rfft(mix[:, 0]))
    freqs = np.fft.rfftfreq(len(mix), 1 / SR)
    bands = [(20, 80, "sub"), (80, 300, "low"), (300, 2000, "mid"),
             (2000, 8000, "high"), (8000, 20000, "air")]
    report = []
    for lo, hi, name in bands:
        e = np.sqrt((spec[(freqs >= lo) & (freqs < hi)] ** 2).mean())
        report.append(f"{name}={20 * np.log10(e + 1e-12):.1f}dB")
    print(f"len={n_out / SR:.1f}s peak={peak_db:.2f}dBFS rms={rms_db:.2f}dBFS")
    print("spectral balance:", " ".join(report))
    assert np.isfinite(mix).all(), "NaN в миксе!"

    sf.write(out_path, mix.astype(np.float32), SR)
    print("written:", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_dark_ambient.mp3")
