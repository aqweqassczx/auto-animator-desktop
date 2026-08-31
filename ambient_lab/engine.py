"""Движок ambient_lab: детерминированный рендер dark ambient из JSON-спеки.

Формат спеки и правила жанра — в SPEC.md рядом.
Запуск: python3 ambient_lab/engine.py spec.json out.mp3 [--midi out.mid]
"""

import copy
import json
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100

ARP_PATTERNS = {
    "A": [[0.0, 0, 0.8], [4.0, 1, 0.6]],
    "B": [[0.0, 0, 0.9], [1.5, 1, 0.7], [3.0, 2, 0.8],
          [4.0, 3, 1.0], [5.5, 2, 0.6], [7.0, 1, 0.7]],
    "C": [[0.5, 1, 0.8], [2.0, 2, 0.7], [3.5, 3, 1.0],
          [4.5, -1, 0.9], [6.0, 2, 0.6], [7.5, 0, 0.7]],
}

DEFAULT_SPEC = {
    "title": "untitled",
    "seed": 1,
    "bpm": 70,
    "beats_per_chord": 8,
    "piano_pattern": [[0.0, 2, 0.55], [3.0, 1, 0.4], [5.5, 3, 0.45]],
    "lead_mode": "top",
    "fx": {"reverb_sec": 9.0, "predelay": 0.04, "delay_beats": 0.75,
           "delay_fb": 0.5, "pad_wet": 0.45, "arp_wet": 0.9,
           "piano_wet": 0.7, "lead_wet": 0.9},
    "mix": {"pad": 0.36, "bass": 0.31, "arp": 0.20, "piano": 0.40,
            "lead": 0.18, "hiss": 0.035, "crackle": 0.05, "wind": 0.10},
    "master": {"target_rms_db": -18.5, "peak_db": -1.2,
               "fade_in": 4.0, "fade_out": 10.0},
}

LAYERS = ["pad", "bass", "arp", "piano", "lead", "hiss", "crackle", "wind"]


def _fail(msg):
    sys.exit(f"spec error: {msg}")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_spec(path):
    try:
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
    except json.JSONDecodeError as e:
        _fail(f"битый JSON — {e}")
    except OSError as e:
        _fail(f"не читается файл — {e}")

    spec = copy.deepcopy(DEFAULT_SPEC)
    for key, val in user.items():
        if isinstance(val, dict) and isinstance(spec.get(key), dict):
            spec[key].update(val)
        else:
            spec[key] = val

    if not isinstance(spec.get("chords"), list) or not spec["chords"]:
        _fail("нужен непустой список chords")
    if not isinstance(spec.get("sections"), list) or not spec["sections"]:
        _fail("нужен непустой список sections")
    if not _num(spec["bpm"]) or not 40 <= spec["bpm"] <= 140:
        _fail("bpm должен быть числом в 40..140")
    if not _num(spec["beats_per_chord"]) or not 1 <= spec["beats_per_chord"] <= 32:
        _fail("beats_per_chord должен быть числом в 1..32")
    if spec["lead_mode"] not in ("top", "ninth"):
        _fail("lead_mode: только 'top' или 'ninth'")

    for ch in spec["chords"]:
        name = ch.get("name", "?")
        for field in ("pad", "arp"):
            if not isinstance(ch.get(field), list):
                _fail(f"{field} аккорда {name} должен быть списком нот")
        if not _num(ch.get("bass")):
            _fail(f"bass аккорда {name} должен быть MIDI-числом")
        if not 3 <= len(ch["pad"]) <= 8:
            _fail(f"pad аккорда {name}: нужно 3-8 нот (лучше 4-6)")
        if not 2 <= len(ch["arp"]) <= 8:
            _fail(f"arp аккорда {name}: нужно 2-8 нот")
        if "piano" in ch and (not isinstance(ch["piano"], list) or not ch["piano"]):
            _fail(f"piano аккорда {name}: непустой список нот или убери поле")
        ch.setdefault("piano", list(ch["pad"])[2:6])
        notes = list(ch["pad"]) + list(ch["arp"]) + [ch["bass"]] + list(ch["piano"])
        if not all(_num(n) and 20 <= n <= 100 for n in notes):
            _fail(f"MIDI-ноты вне диапазона 20..100 в {name}")

    def check_pattern(p, where):
        if isinstance(p, str):
            if p not in ARP_PATTERNS:
                _fail(f"неизвестный arp_pattern '{p}' в {where} (есть A/B/C или свой список)")
            return
        if not isinstance(p, list) or not p:
            _fail(f"arp_pattern в {where}: строка A/B/C или непустой список [доля, индекс, велосити]")
        for ev in p:
            if (not isinstance(ev, list) or len(ev) != 3
                    or not all(_num(x) for x in ev) or not 0 <= ev[2] <= 1.5):
                _fail(f"событие паттерна {ev} в {where}: нужно [доля, индекс, велосити 0..1.5]")

    for i, sec in enumerate(spec["sections"], 1):
        where = f"секции {i}"
        if not _num(sec.get("bars")) or sec["bars"] < 1:
            _fail(f"bars в {where} должен быть числом >= 1")
        sec.setdefault("layers", {})
        if not isinstance(sec["layers"], dict):
            _fail(f"layers в {where} должен быть объектом")
        unknown = set(sec["layers"]) - set(LAYERS)
        if unknown:
            _fail(f"неизвестные слои {sorted(unknown)} в {where} (есть: {', '.join(LAYERS)})")
        for k, v in sec["layers"].items():
            if not _num(v) or not 0 <= v <= 1.5:
                _fail(f"громкость слоя {k} в {where}: число 0..1.5")
        sec.setdefault("pad_cutoff", 1200)
        if not _num(sec["pad_cutoff"]) or not 200 <= sec["pad_cutoff"] <= 8000:
            _fail(f"pad_cutoff в {where}: число 200..8000")
        sec.setdefault("arp_pattern", "B")
        check_pattern(sec["arp_pattern"], where)
    check_pattern(spec["piano_pattern"], "piano_pattern")

    for k in ("reverb_sec", "predelay", "delay_beats", "delay_fb"):
        if not _num(spec["fx"].get(k)):
            _fail(f"fx.{k} должен быть числом")
    if not 2 <= spec["fx"]["reverb_sec"] <= 16:
        _fail("fx.reverb_sec: 2..16 секунд")
    if not 0 <= spec["fx"]["delay_fb"] <= 0.85:
        _fail("fx.delay_fb: 0..0.85")
    return spec


def midi_to_freq(m):
    return 440.0 * 2.0 ** ((float(m) - 69.0) / 12.0)


def lowpass(x, cutoff, order=2):
    sos = butter(order, min(cutoff, SR / 2 - 100) / (SR / 2), "low", output="sos")
    return sosfilt(sos, x, axis=0) if x.ndim > 1 else sosfilt(sos, x)


def highpass(x, cutoff, order=2):
    sos = butter(order, max(cutoff, 5) / (SR / 2), "high", output="sos")
    return sosfilt(sos, x, axis=0) if x.ndim > 1 else sosfilt(sos, x)


def env_asr(n, attack, release, level=1.0):
    e = np.full(n, level)
    a = min(int(attack * SR), n)
    r = min(int(release * SR), n - a)
    if a > 0:
        e[:a] = level * 0.5 * (1 - np.cos(np.linspace(0, np.pi, a)))
    if r > 0:
        e[n - r:] = e[n - r] * 0.5 * (1 + np.cos(np.linspace(0, np.pi, r)))
    return e


def end_fade(sig, ms=25.0):
    """Короткий скат в ноль в конце ноты — против ступеньки при обрезке хвоста."""
    k = min(int(ms / 1000 * SR), len(sig))
    if k > 0:
        sig[-k:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, k)))
    return sig


_SAW_TABLES = {}


def bl_saw(f, n, phase01):
    """Бэндлимитед-пила через вейвтейбл (без алиасинга наивной пилы)."""
    K = max(1, min(int(0.45 * SR / f), 1024))
    table = _SAW_TABLES.get(K)
    if table is None:
        size = 4096
        spec = np.zeros(size // 2 + 1, dtype=complex)
        k = np.arange(1, min(K, size // 2 - 1) + 1)
        spec[1:len(k) + 1] = 1j / k
        table = np.fft.irfft(spec)
        table /= np.abs(table).max()
        _SAW_TABLES[K] = table
    size = len(table)
    idx = (phase01 + f * np.arange(n) / SR) % 1.0 * size
    i0 = idx.astype(np.int64)
    frac = idx - i0
    i1 = (i0 + 1) % size
    return table[i0] * (1 - frac) + table[i1] * frac


class Renderer:
    def __init__(self, spec):
        self.spec = spec
        self.rng = np.random.default_rng(int(spec["seed"]))
        self.beat = 60.0 / spec["bpm"]
        self.bar = 4 * self.beat
        self.bpc = spec["beats_per_chord"]
        self.total_bars = sum(s["bars"] for s in spec["sections"])
        self.total_beats = self.total_bars * 4
        self.tail = max(6.0, spec["fx"]["reverb_sec"])
        self.n_total = int((self.total_beats * self.beat + self.tail) * SR)

        # Слоты аккордов: (стартовая доля, аккорд)
        self.chord_slots = []
        b = 0.0
        i = 0
        while b < self.total_beats:
            self.chord_slots.append((b, spec["chords"][i % len(spec["chords"])]))
            b += self.bpc
            i += 1

        # Границы секций в долях
        self.section_bounds = []
        b = 0.0
        for sec in spec["sections"]:
            self.section_bounds.append((b, b + sec["bars"] * 4, sec))
            b += sec["bars"] * 4

    def section_at(self, beat_pos):
        for start, end, sec in self.section_bounds:
            if start <= beat_pos < end:
                return sec
        return self.section_bounds[-1][2]

    def layer_gain_curve(self, layer):
        """Кусочно-постоянная громкость слоя по секциям, сглаженная 1с-скатами."""
        xp, fp = [0.0], [self.section_bounds[0][2]["layers"].get(layer, 0.0)]
        for start, _end, sec in self.section_bounds:
            g = sec["layers"].get(layer, 0.0)
            t = start * self.beat
            xp.extend([max(t - 0.5, xp[-1] + 1e-4), t + 0.5])
            fp.extend([fp[-1], g])
        xp.append(self.n_total / SR)
        fp.append(fp[-1])
        times = np.arange(self.n_total) / SR
        return np.interp(times, xp, fp)[:, None]

    def place(self, buf, start_sec, chunk):
        i0 = int(start_sec * SR)
        if i0 < 0:
            chunk = chunk[-i0:]
            i0 = 0
        i1 = min(i0 + len(chunk), len(buf))
        if i1 > i0:
            buf[i0:i1] += chunk[: i1 - i0]

    # --- инструменты ---

    def pad_chord(self, notes, dur, cutoff):
        n = int((dur + 5.0) * SR)
        t = np.arange(n) / SR
        out = np.zeros((n, 2))
        detunes = [-9, -4.5, 0, 4.5, 9]
        for idx, m in enumerate(notes):
            f0 = midi_to_freq(m)
            # общая фаза суб-октавы на оба канала: низ остаётся моно-совместимым
            sub_phase = self.rng.uniform(0, 2 * np.pi)
            for ch in range(2):
                sig = np.zeros(n)
                for cents in detunes:
                    f = f0 * 2 ** (cents / 1200)
                    sig += bl_saw(f, n, self.rng.uniform(0, 1))
                sig /= len(detunes)
                if idx == 0:
                    sig += 0.22 * np.sin(2 * np.pi * (f0 / 2) * t + sub_phase)
                out[:, ch] += sig
        out /= len(notes)
        out = lowpass(out, cutoff, order=2)
        out *= env_asr(n, 2.8, 5.0)[:, None]
        return out

    def bass_note(self, midi, prev_midi, dur):
        n = int((dur + 1.0) * SR)
        f_target = midi_to_freq(midi)
        f_prev = midi_to_freq(prev_midi)
        glide = int(0.10 * SR)
        freq = np.full(n, f_target)
        freq[:glide] = np.linspace(f_prev, f_target, glide)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        sig = 0.55 * (2 * ((phase / (2 * np.pi)) % 1.0) - 1.0) + 0.6 * np.sin(phase)
        sig = lowpass(sig, 120, order=4)
        # релиз заканчивается внутри слота: корни аккордов не звучат вдвоём
        dur_n = min(int(dur * SR), n)
        env = np.zeros(n)
        env[:dur_n] = env_asr(dur_n, 0.4, 0.6)
        sig *= env
        return np.column_stack([sig, sig])

    def arp_note(self, midi, vel):
        n = int(3.0 * SR)
        t = np.arange(n) / SR
        f = midi_to_freq(midi)
        # верхние гармоники с быстрым затуханием: дилей становится слышен над пэдом
        sig = np.zeros(n)
        for mult, amp, tau in [(1, 1.0, 1.1), (2, 0.22, 0.55),
                               (3, 0.10, 0.35), (4, 0.05, 0.25)]:
            if f * mult < SR * 0.45:
                sig += amp * np.sin(2 * np.pi * f * mult * t) * np.exp(-t / tau)
        a = int(0.015 * SR)
        sig[:a] *= np.linspace(0, 1, a)
        end_fade(sig)
        sig *= vel
        return np.column_stack([sig, sig]) * 0.5

    def piano_note(self, midi, vel, pan):
        n = int(4.0 * SR)
        t = np.arange(n) / SR
        f = midi_to_freq(midi)
        sig = np.zeros(n)
        for k, (amp, tau) in enumerate([(1.0, 1.8), (0.35, 0.9), (0.12, 0.5)], start=1):
            if f * k < SR * 0.45:
                sig += amp * np.sin(2 * np.pi * f * k * t) * np.exp(-t / tau)
        a = int(0.004 * SR)
        sig[:a] *= np.linspace(0, 1, a)
        sig = lowpass(sig, 3500)
        end_fade(sig)
        sig *= vel * 0.6
        return np.column_stack([sig * (1 - pan), sig * pan]) * 1.4

    def lead_note(self, midi, dur):
        n = int((dur + 3.0) * SR)
        t = np.arange(n) / SR
        f0 = midi_to_freq(midi)
        vibrato = 8.0 * np.sin(2 * np.pi * 4.3 * t)
        drift = 6.0 * np.sin(2 * np.pi * 0.13 * t + 1.0)
        f = f0 * 2 ** ((vibrato + drift) / 1200)
        phase = 2 * np.pi * np.cumsum(f) / SR
        sig = np.sin(phase) + 0.15 * np.sin(2 * phase)
        sig *= env_asr(n, 1.2, 2.5)
        return np.column_stack([sig, sig]) * 0.5

    # --- эффекты ---

    def ping_pong(self, x, delay_sec, feedback, taps=10):
        n = len(x)
        out = x.copy()
        d = max(1, int(delay_sec * SR))
        tap = x.copy()
        first_right = bool(self.rng.integers(0, 2))  # сторона первого эха — не всегда правая
        for k in range(1, taps + 1):
            tap = lowpass(tap, 6500) * feedback
            if k * d >= n:
                break
            shifted = np.zeros_like(x)
            shifted[k * d:] = tap[: n - k * d]
            right = (k % 2 == 1) == first_right
            pan = 0.75 if right else 0.25
            out[:, 0] += shifted[:, 0] * (1 - pan)
            out[:, 1] += shifted[:, 1] * pan
        return out

    def make_ir(self):
        t60 = self.spec["fx"]["reverb_sec"]
        n = int(t60 * SR)
        t = np.arange(n) / SR
        ir = np.zeros((n, 2))
        # Мультибэнд: низ короче, середина длиннее, верх гаснет быстрее всех
        for band_filter, band_t60 in [
            (lambda x: lowpass(x, 400), t60 * 0.65),
            (lambda x: highpass(lowpass(x, 4000), 400), t60),
            (lambda x: highpass(x, 4000), t60 * 0.5),
        ]:
            noise = self.rng.standard_normal((n, 2))
            decay = np.exp(-6.907 * t / band_t60)[:, None]
            ir += band_filter(noise) * decay
        # Каналам — одинаковая АЧХ (своя только фаза): без стерео-лотереи
        # на длинных синусах арпа/лида; затухание сохраняется в фазах
        H = np.fft.rfft(ir, axis=0)
        mag = np.sqrt((np.abs(H) ** 2).mean(axis=1, keepdims=True))
        H = mag * np.exp(1j * np.angle(H))
        ir = np.fft.irfft(H, n=n, axis=0)
        pre = np.zeros((int(self.spec["fx"]["predelay"] * SR), 2))
        ir = np.vstack([pre, ir])
        ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
        return ir

    # --- сборка ---

    def render(self):
        spec = self.spec
        raw = {name: np.zeros((self.n_total, 2)) for name in
               ("pad", "bass", "arp", "piano", "lead")}

        prev_bass = spec["chords"][-1]["bass"]
        chord_dur = self.bpc * self.beat
        for start_beat, chord in self.chord_slots:
            t0 = start_beat * self.beat
            sec = self.section_at(start_beat)
            self.place(raw["pad"], t0,
                       self.pad_chord(chord["pad"], chord_dur, sec["pad_cutoff"]))
            self.place(raw["bass"], t0,
                       self.bass_note(chord["bass"], prev_bass, chord_dur))
            prev_bass = chord["bass"]

            pattern = sec["arp_pattern"]
            if isinstance(pattern, str):
                pattern = ARP_PATTERNS[pattern]
            for beat_pos, idx, vel in pattern:
                notes = chord["arp"]
                idx = int(idx)
                note = notes[idx] if -len(notes) <= idx < len(notes) else notes[-1]
                jitter = self.rng.uniform(-0.015, 0.015)
                v = vel * self.rng.uniform(0.92, 1.08)
                self.place(raw["arp"], t0 + beat_pos * self.beat + jitter,
                           self.arp_note(note, v))

            for beat_pos, idx, vel in spec["piano_pattern"]:
                notes = chord["piano"]
                i = int(idx) % len(notes)
                note = notes[i]
                # пан симметрично вокруг центра, без систематического крена вправо
                spread = (i / max(len(notes) - 1, 1) - 0.5) * 0.3
                pan = 0.5 + spread
                jitter = self.rng.uniform(-0.02, 0.02)
                self.place(raw["piano"], t0 + beat_pos * self.beat + jitter,
                           self.piano_note(note, vel, pan))

            lead_midi = chord["pad"][-1] if spec["lead_mode"] == "top" else chord["pad"][-2]
            self.place(raw["lead"], t0, self.lead_note(lead_midi, chord_dur))

        # Текстуры
        t = np.arange(self.n_total) / SR
        hiss = lowpass(highpass(self.rng.standard_normal((self.n_total, 2)), 2500), 10000)
        hiss *= (0.55 + 0.35 * np.sin(2 * np.pi * 0.1 * t))[:, None]
        raw["hiss"] = hiss

        crackle = np.zeros((self.n_total, 2))
        for _ in range(int(self.n_total / SR * 2)):
            pos = self.rng.integers(0, self.n_total - 200)
            click = self.rng.uniform(0.2, 1.0) * np.exp(-np.arange(120) / 25.0)
            crackle[pos:pos + 120, self.rng.integers(0, 2)] += click
        raw["crackle"] = lowpass(crackle, 6000)

        # Ветер: срез ниже 70 Гц — это текстура, а не второй саб-бас
        wind = np.cumsum(self.rng.standard_normal((self.n_total, 2)), axis=0)
        wind = lowpass(highpass(wind, 70), 500)
        wind /= np.abs(wind).max() + 1e-9
        wind *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 0.7))[:, None] * 1.6
        raw["wind"] = wind

        # Дилей арпа
        raw["arp"] = self.ping_pong(raw["arp"], spec["fx"]["delay_beats"] * self.beat,
                                    spec["fx"]["delay_fb"])

        # Секционные громкости + баланс
        final = {}
        for name in LAYERS:
            final[name] = raw[name] * self.layer_gain_curve(name) * spec["mix"][name]

        # Один общий реверб-бас: отправки с тональных слоёв
        send = sum(final[name] * spec["fx"][f"{name}_wet"]
                   for name in ("pad", "arp", "piano", "lead"))
        ir = self.make_ir()
        wet = np.column_stack([
            fftconvolve(send[:, 0], ir[:, 0])[: self.n_total],
            fftconvolve(send[:, 1], ir[:, 1])[: self.n_total],
        ]) * 2.2

        mix = wet
        for name in LAYERS:
            mix = mix + final[name]

        # Дыхание микса, чистка, громкость
        mix *= (1.0 + 0.13 * np.sin(2 * np.pi * 0.06 * t - 1.2))[:, None]
        mix = highpass(mix, 24)

        # Баланс каналов: выравниваем RMS L/R до лимитера
        ch_rms = np.sqrt((mix ** 2).mean(axis=0))
        if ch_rms.min() > 1e-9:
            target = float(np.sqrt(ch_rms[0] * ch_rms[1]))
            mix[:, 0] *= target / ch_rms[0]
            mix[:, 1] *= target / ch_rms[1]

        rms = np.sqrt((mix ** 2).mean())
        mix *= 10 ** (spec["master"]["target_rms_db"] / 20) / (rms + 1e-12)
        peak_lin = 10 ** (spec["master"]["peak_db"] / 20)
        mix = np.tanh(mix / peak_lin) * peak_lin

        fi = min(int(spec["master"]["fade_in"] * SR), len(mix))
        if fi > 0:
            mix[:fi] *= (0.5 * (1 - np.cos(np.linspace(0, np.pi, fi))))[:, None]
        fo = min(int(spec["master"]["fade_out"] * SR), len(mix))
        if fo > 0:
            mix[-fo:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, fo))))[:, None]

        assert np.isfinite(mix).all(), "NaN в миксе"
        return mix.astype(np.float32)

    # --- MIDI-экспорт для FL Studio ---

    def export_midi(self, path):
        import mido
        ppq = 480
        mid = mido.MidiFile(ticks_per_beat=ppq)
        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage("set_tempo",
                                            tempo=mido.bpm2tempo(self.spec["bpm"])))
        mid.tracks.append(tempo_track)

        def beats_to_ticks(b):
            return int(round(b * ppq))

        def add_track(name, events):
            """events: список (start_beat, dur_beats, midi, vel)."""
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=name))
            msgs = []
            for start, dur, note, vel in events:
                msgs.append((beats_to_ticks(start), "note_on", int(note),
                             max(1, min(127, int(vel * 127)))))
                msgs.append((beats_to_ticks(start + dur), "note_off", int(note), 0))
            msgs.sort(key=lambda m: (m[0], m[1] == "note_on"))
            now = 0
            for tick, kind, note, vel in msgs:
                track.append(mido.Message(kind, note=note, velocity=vel,
                                          time=tick - now))
                now = tick
            mid.tracks.append(track)

        pad_ev, bass_ev, arp_ev, piano_ev, lead_ev = [], [], [], [], []
        for start_beat, chord in self.chord_slots:
            sec = self.section_at(start_beat)
            layers = sec["layers"]
            if layers.get("pad", 0) > 0:
                pad_ev += [(start_beat, self.bpc, n, 0.6) for n in chord["pad"]]
            if layers.get("bass", 0) > 0:
                bass_ev.append((start_beat, self.bpc, chord["bass"], 0.7))
            if layers.get("arp", 0) > 0:
                pattern = sec["arp_pattern"]
                if isinstance(pattern, str):
                    pattern = ARP_PATTERNS[pattern]
                for beat_pos, idx, vel in pattern:
                    notes = chord["arp"]
                    idx = int(idx)
                    note = notes[idx] if -len(notes) <= idx < len(notes) else notes[-1]
                    arp_ev.append((start_beat + beat_pos, 1.0, note, vel))
            if layers.get("piano", 0) > 0:
                for beat_pos, idx, vel in self.spec["piano_pattern"]:
                    note = chord["piano"][int(idx) % len(chord["piano"])]
                    piano_ev.append((start_beat + beat_pos, 2.0, note, vel))
            if layers.get("lead", 0) > 0:
                lead_midi = (chord["pad"][-1] if self.spec["lead_mode"] == "top"
                             else chord["pad"][-2])
                lead_ev.append((start_beat, self.bpc, lead_midi, 0.5))

        for name, events in [("pad", pad_ev), ("bass", bass_ev), ("arp", arp_ev),
                             ("piano", piano_ev), ("lead", lead_ev)]:
            if events:
                add_track(name, events)
        mid.save(path)


def main():
    argv = list(sys.argv[1:])
    midi_out = None
    if "--midi" in argv:
        i = argv.index("--midi")
        midi_out = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) != 2:
        sys.exit("usage: engine.py spec.json out.mp3 [--midi out.mid]")
    spec_path, out_path = argv

    spec = load_spec(spec_path)
    r = Renderer(spec)
    mix = r.render()
    sf.write(out_path, mix, SR)
    if midi_out:
        r.export_midi(midi_out)
    print(json.dumps({
        "title": spec["title"],
        "out": out_path,
        "midi": midi_out,
        "len_sec": round(len(mix) / SR, 1),
        "peak_db": round(float(20 * np.log10(np.abs(mix).max() + 1e-12)), 2),
        "rms_db": round(float(20 * np.log10(np.sqrt((mix ** 2).mean()) + 1e-12)), 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
