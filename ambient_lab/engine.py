"""Движок ambient_lab: детерминированный рендер dark ambient из JSON-спеки.

Формат спеки и правила жанра — в SPEC.md рядом.
Запуск: python3 ambient_lab/engine.py spec.json out.mp3 [--midi out.mid]
"""

import copy
import json
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve, sawtooth

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
    "mix": {"pad": 0.34, "bass": 0.40, "arp": 0.26, "piano": 0.30,
            "lead": 0.10, "hiss": 0.035, "crackle": 0.05, "wind": 0.10},
    "master": {"target_rms_db": -16.0, "peak_db": -1.2,
               "fade_in": 4.0, "fade_out": 10.0},
}

LAYERS = ["pad", "bass", "arp", "piano", "lead", "hiss", "crackle", "wind"]


def load_spec(path):
    with open(path, encoding="utf-8") as fh:
        user = json.load(fh)
    spec = copy.deepcopy(DEFAULT_SPEC)
    for key, val in user.items():
        if isinstance(val, dict) and isinstance(spec.get(key), dict):
            spec[key].update(val)
        else:
            spec[key] = val
    if not spec.get("chords"):
        sys.exit("spec error: нужен непустой список chords")
    if not spec.get("sections"):
        sys.exit("spec error: нужен непустой список sections")
    for ch in spec["chords"]:
        for field in ("pad", "bass", "arp"):
            if field not in ch:
                sys.exit(f"spec error: у аккорда {ch.get('name', '?')} нет поля {field}")
        notes = list(ch["pad"]) + list(ch["arp"]) + [ch["bass"]]
        if not all(isinstance(n, (int, float)) and 20 <= n <= 100 for n in notes):
            sys.exit(f"spec error: MIDI-ноты вне диапазона 20..100 в {ch.get('name', '?')}")
        ch.setdefault("piano", list(ch["pad"])[2:6])
    for sec in spec["sections"]:
        if "bars" not in sec or sec["bars"] < 1:
            sys.exit("spec error: у секции нет bars >= 1")
        sec.setdefault("layers", {})
        sec.setdefault("pad_cutoff", 1200)
        sec.setdefault("arp_pattern", "B")
    if not 40 <= spec["bpm"] <= 140:
        sys.exit("spec error: bpm вне 40..140")
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
            for ch in range(2):
                sig = np.zeros(n)
                for cents in detunes:
                    f = f0 * 2 ** (cents / 1200)
                    sig += sawtooth(2 * np.pi * f * t + self.rng.uniform(0, 2 * np.pi))
                sig /= len(detunes)
                if idx == 0:  # тело — суб-октава только у нижней ноты
                    sig += 0.35 * np.sin(2 * np.pi * (f0 / 2) * t
                                         + self.rng.uniform(0, 2 * np.pi))
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
        sig = 0.55 * sawtooth(phase) + 0.6 * np.sin(phase)
        sig = lowpass(sig, 120, order=4)
        sig *= env_asr(n, 0.4, 0.8)
        return np.column_stack([sig, sig])

    def arp_note(self, midi, vel):
        n = int(3.0 * SR)
        t = np.arange(n) / SR
        f = midi_to_freq(midi)
        sig = np.sin(2 * np.pi * f * t) + 0.08 * np.sin(4 * np.pi * f * t)
        env = np.exp(-t / 1.1)
        a = int(0.015 * SR)
        env[:a] *= np.linspace(0, 1, a)
        sig *= env * vel
        return np.column_stack([sig, sig]) * 0.5

    def piano_note(self, midi, vel, pan):
        n = int(4.0 * SR)
        t = np.arange(n) / SR
        f = midi_to_freq(midi)
        sig = np.zeros(n)
        for k, (amp, tau) in enumerate([(1.0, 1.8), (0.35, 0.9), (0.12, 0.5)], start=1):
            sig += amp * np.sin(2 * np.pi * f * k * t) * np.exp(-t / tau)
        a = int(0.004 * SR)
        sig[:a] *= np.linspace(0, 1, a)
        sig = lowpass(sig, 3500)
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
        d = int(delay_sec * SR)
        tap = x.copy()
        for k in range(1, taps + 1):
            tap = lowpass(tap, 6500) * feedback
            if k * d >= n:
                break
            shifted = np.zeros_like(x)
            shifted[k * d:] = tap[: n - k * d]
            pan = 0.75 if k % 2 else 0.25
            out[:, 0] += shifted[:, 0] * (1 - pan) * 2
            out[:, 1] += shifted[:, 1] * pan * 2
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
                pattern = ARP_PATTERNS.get(pattern, ARP_PATTERNS["B"])
            for beat_pos, idx, vel in pattern:
                notes = chord["arp"]
                note = notes[idx] if -len(notes) <= idx < len(notes) else notes[-1]
                jitter = self.rng.uniform(-0.015, 0.015)
                v = vel * self.rng.uniform(0.92, 1.08)
                self.place(raw["arp"], t0 + beat_pos * self.beat + jitter,
                           self.arp_note(note, v))

            for beat_pos, idx, vel in spec["piano_pattern"]:
                notes = chord["piano"]
                note = notes[idx % len(notes)]
                pan = 0.35 + 0.3 * (idx % len(notes)) / max(len(notes) - 1, 1)
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

        wind = np.cumsum(self.rng.standard_normal((self.n_total, 2)), axis=0)
        wind = lowpass(highpass(wind, 20), 500)
        wind /= np.abs(wind).max() + 1e-9
        wind *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 0.7))[:, None] * 3.0
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
        ]) * 3.0

        mix = wet
        for name in LAYERS:
            mix = mix + final[name]

        # Дыхание микса, чистка, громкость
        mix *= (1.0 + 0.13 * np.sin(2 * np.pi * 0.06 * t - 1.2))[:, None]
        mix = highpass(mix, 24)

        rms = np.sqrt((mix ** 2).mean())
        mix *= 10 ** (spec["master"]["target_rms_db"] / 20) / (rms + 1e-12)
        peak_lin = 10 ** (spec["master"]["peak_db"] / 20)
        mix = np.tanh(mix / peak_lin) * peak_lin

        fi = int(spec["master"]["fade_in"] * SR)
        mix[:fi] *= (0.5 * (1 - np.cos(np.linspace(0, np.pi, fi))))[:, None]
        fo = int(spec["master"]["fade_out"] * SR)
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
                    pattern = ARP_PATTERNS.get(pattern, ARP_PATTERNS["B"])
                for beat_pos, idx, vel in pattern:
                    notes = chord["arp"]
                    note = notes[idx] if -len(notes) <= idx < len(notes) else notes[-1]
                    arp_ev.append((start_beat + beat_pos, 1.0, note, vel))
            if layers.get("piano", 0) > 0:
                for beat_pos, idx, vel in self.spec["piano_pattern"]:
                    note = chord["piano"][idx % len(chord["piano"])]
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
