"""Числовая проверка рендера вместо ушей: громкость, спектр, стерео, артефакты.

Запуск: python3 ambient_lab/analyze.py track.mp3
Выводит JSON с метриками и списком нарушений жанровых целей (verdicts).
"""

import json
import sys
import warnings

import numpy as np
import soundfile as sf


def _r(v, nd=2):
    """Округлить; не-числа и не-конечные значения -> None (валидный JSON)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if np.isfinite(f) else None


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

    lufs = None
    try:
        import pyloudnorm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lufs = float(pyloudnorm.Meter(sr).integrated_loudness(x))
    except Exception:
        pass

    # Клики: резкие скачки между соседними сэмплами (вне первых/последних 100мс)
    guard = int(0.1 * sr)
    diffs = np.abs(np.diff(x[guard:-guard], axis=0))
    click_max = float(diffs.max()) if len(diffs) else 0.0

    corr = None
    if x.shape[1] == 2 and x[:, 0].std() > 1e-9 and x[:, 1].std() > 1e-9:
        corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])

    # Баланс каналов
    lr_diff_db = None
    if x.shape[1] == 2:
        rms_l = np.sqrt((x[:, 0] ** 2).mean())
        rms_r = np.sqrt((x[:, 1] ** 2).mean())
        if rms_l > 1e-9 and rms_r > 1e-9:
            lr_diff_db = 20 * np.log10(rms_r / rms_l)

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
    is_silent_track = silent > 0.9 or rms_db < -55

    # Арка громкости: где пик кратковременного RMS (окна 3с);
    # жанровая цель — максимум ближе к ~2/3 трека
    arc_peak_frac = None
    win3 = int(3 * sr)
    if n > 6 * win3 and not is_silent_track:
        f3 = mono[: n - n % win3].reshape(-1, win3)
        st = np.sqrt((f3 ** 2).mean(axis=1))
        arc_peak_frac = float((np.argmax(st) + 0.5) / len(st))

    verdicts = []
    if clip_count > 0:
        verdicts.append(f"КЛИППИНГ: {clip_count} сэмплов на потолке")
    if peak_db > -0.5:
        verdicts.append(f"пик слишком горячий: {peak_db:.2f} dBFS (цель <= -1)")
    if not -24 <= rms_db <= -12:
        verdicts.append(f"RMS {rms_db:.1f} dBFS вне жанрового окна -24..-12")
    if lufs is not None and np.isfinite(lufs) and not -22 <= lufs <= -13:
        verdicts.append(f"LUFS {lufs:.1f} вне жанрового окна -22..-13")
    if dc > 0.01:
        verdicts.append(f"DC-смещение {dc:.4f}")
    if click_max > 0.30:
        verdicts.append(f"вероятные клики: скачок {click_max:.2f} между сэмплами")
    if silent > 0.05:
        verdicts.append(f"{silent * 100:.0f}% трека — тишина ниже -60 дБ")
    if not is_silent_track:
        if crest_db < 6:
            verdicts.append(f"crest {crest_db:.1f} дБ — пережато, нет динамики")
        if corr is not None and corr < 0.0:
            verdicts.append(f"стерео-корреляция {corr:.2f} — риск проблем в моно")
        if lr_diff_db is not None and abs(lr_diff_db) > 1.25:
            verdicts.append(f"перекос каналов: R-L = {lr_diff_db:+.1f} дБ (цель |x| <= 1)")
        if bands["sub"] - bands["mid"] > 10:
            verdicts.append(f"суб перевешивает середину на {bands['sub'] - bands['mid']:.0f} дБ")
        if bands["mid"] - bands["high"] > 35:
            verdicts.append("верх мёртвый: high на 35+ дБ ниже mid — добавить текстуру/яркость")
        if bands["mid"] - bands["high"] < 8:
            verdicts.append("слишком ярко для жанра: high ближе 8 дБ к mid")
        if arc_peak_frac is not None and not 0.40 <= arc_peak_frac <= 0.90:
            verdicts.append(f"арка: пик громкости на {arc_peak_frac * 100:.0f}% "
                            "трека (цель 40..90%, лучше ~66%)")

    print(json.dumps({
        "file": path,
        "len_sec": _r(n / sr, 1),
        "peak_db": _r(peak_db),
        "rms_db": _r(rms_db),
        "lufs": _r(lufs),
        "crest_db": _r(crest_db, 1),
        "dc_offset": _r(dc, 5),
        "clip_count": clip_count,
        "click_max": _r(click_max, 3),
        "stereo_corr": _r(corr),
        "lr_diff_db": _r(lr_diff_db),
        "arc_peak_frac": _r(arc_peak_frac),
        "bands_db": bands,
        "silent_ratio": _r(silent, 3),
        "verdicts": verdicts or ["OK: жанровые цели соблюдены"],
    }, ensure_ascii=False, indent=1, allow_nan=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: analyze.py track.mp3")
    main(sys.argv[1])
