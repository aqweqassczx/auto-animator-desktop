"""Числовая проверка рендера вместо ушей: громкость, спектр, стерео, артефакты.

Запуск: python3 ambient_lab/analyze.py track.mp3
Выводит JSON с метриками и списком нарушений жанровых целей (verdicts).
"""

import json
import sys

import numpy as np
import soundfile as sf


def main(path):
    x, sr = sf.read(path, always_2d=True)
    mono = x.mean(axis=1)
    n = len(x)

    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    peak_db = 20 * np.log10(peak + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    crest_db = peak_db - rms_db
    dc = float(np.abs(mono.mean()))
    clip_count = int((np.abs(x) >= 0.999).sum())

    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(sr)
        lufs = round(float(meter.integrated_loudness(x)), 2)
    except Exception:
        lufs = None

    # Клики: резкие скачки между соседними сэмплами (вне первых/последних 100мс)
    guard = int(0.1 * sr)
    diffs = np.abs(np.diff(x[guard:-guard], axis=0))
    click_max = float(diffs.max()) if len(diffs) else 0.0

    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] == 2 else 1.0

    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    bands = {}
    for lo, hi, name in [(20, 80, "sub"), (80, 300, "low"), (300, 2000, "mid"),
                         (2000, 8000, "high"), (8000, 20000, "air")]:
        sel = spec[(freqs >= lo) & (freqs < hi)]
        e = np.sqrt((sel ** 2).mean()) if len(sel) else 0.0
        bands[name] = round(float(20 * np.log10(e + 1e-12)), 1)

    # Тишина: доля окон по 0.5с с RMS ниже -60 дБ (не считая крайние фейды)
    win = int(0.5 * sr)
    frames = mono[: n - n % win].reshape(-1, win)
    frame_rms = np.sqrt((frames ** 2).mean(axis=1))
    silent = float((20 * np.log10(frame_rms + 1e-12) < -60)[3:-3].mean()) if len(frames) > 6 else 0.0

    verdicts = []
    if clip_count > 0:
        verdicts.append(f"КЛИППИНГ: {clip_count} сэмплов на потолке")
    if peak_db > -0.5:
        verdicts.append(f"пик слишком горячий: {peak_db:.2f} dBFS (цель <= -1)")
    if not -22 <= rms_db <= -11:
        verdicts.append(f"RMS {rms_db:.1f} dBFS вне жанрового окна -22..-11")
    if lufs is not None and not -26 <= lufs <= -12:
        verdicts.append(f"LUFS {lufs} вне окна -26..-12")
    if dc > 0.01:
        verdicts.append(f"DC-смещение {dc:.4f}")
    if click_max > 0.30:
        verdicts.append(f"вероятные клики: скачок {click_max:.2f} между сэмплами")
    if crest_db < 6:
        verdicts.append(f"crest {crest_db:.1f} дБ — пережато, нет динамики")
    if corr < 0.0:
        verdicts.append(f"стерео-корреляция {corr:.2f} — риск проблем в моно")
    if bands["sub"] - bands["mid"] > 9:
        verdicts.append(f"суб перевешивает середину на {bands['sub'] - bands['mid']:.0f} дБ")
    if bands["mid"] - bands["high"] > 35:
        verdicts.append("верх мёртвый: high на 35+ дБ ниже mid — добавить текстуру/яркость")
    if bands["mid"] - bands["high"] < 8:
        verdicts.append("слишком ярко для жанра: high ближе 8 дБ к mid")
    if silent > 0.05:
        verdicts.append(f"{silent * 100:.0f}% трека — тишина ниже -60 дБ")

    print(json.dumps({
        "file": path,
        "len_sec": round(n / sr, 1),
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "lufs": lufs,
        "crest_db": round(crest_db, 1),
        "dc_offset": round(dc, 5),
        "clip_count": clip_count,
        "click_max": round(click_max, 3),
        "stereo_corr": round(corr, 2),
        "bands_db": bands,
        "silent_ratio": round(silent, 3),
        "verdicts": verdicts or ["OK: жанровые цели соблюдены"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: analyze.py track.mp3")
    main(sys.argv[1])
