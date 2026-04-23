import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import platform
import time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".mkv")
ORDERED_FOLDERS = ["1", "2", "3", "4", "5", "6"]

# Порог предупреждения: |медиа − фразы| больше этого числа.
MEDIA_PHRASE_DIFF_WARN = 100
LOCAL_ALIGN_CHUNK_SENTENCES = 80
LOCAL_ALIGN_WINDOW_BACK = 50
LOCAL_ALIGN_WINDOW_AHEAD_STEPS = [450, 750, 1050, 1400]
LOCAL_ALIGN_MIN_COVERAGE = 0.20
RENDER_RETRY_DELAYS_SEC = [0.5, 1.0, 2.0]
MIN_PHRASE_SEC = 0.40
MAX_PHRASE_SEC = 12.0
ANCHOR_BLOCK_SENTENCES = 10
BLOCK_FORCED_SENTENCES = 10
MAX_INTER_PHRASE_GAP_SEC = 8.0

# block_forced: ограничить размах одного матча (защита от «съела весь хвост» при rescue).
BLOCK_FORCED_MAX_MATCH_AUDIO_SPAN_SEC = 30.0
BLOCK_FORCED_MAX_WHISPER_WORD_SPAN_RATIO = 4.0
BLOCK_FORCED_MAX_WHISPER_WORD_SPAN_ABSOLUTE = 55

@dataclass
class PipelineConfig:
    audio_file: str
    scenario_file: str
    assets_root: str
    output_dir: str
    asset_paths: list[str] | None = None
    whisper_model: str = "large-v3-turbo"
    whisper_language: str = "en"
    tracks: int = 4
    fps: int = 24
    max_parallel_clips: int = 6
    render_video: bool = True
    render_xml: bool = True
    sequence_name: str = "Auto Animator Sequence"
    video_encoder: str = "auto"
    extend_tail: bool = True
    tail_start_percent: float = 70.0
    xml_parts: int = 3
    processing_mode: str = "render"
    align_mode: str = "block_forced"


def natural_sort_key(value: str) -> list[Any]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"([0-9]+)", value)]


def get_audio_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def path_to_premiere_url(path: str) -> str:
    normalized = path.replace("\\", "/")
    if len(normalized) > 1 and normalized[1] == ":":
        drive = normalized[0]
        tail = normalized[2:]
        parts = [p for p in tail.split("/") if p]
        encoded_tail = "/".join(quote(part, safe="") for part in parts)
        return f"file://localhost/{drive}%3a/{encoded_tail}"
    parts = [p for p in normalized.split("/") if p]
    encoded = "/".join(quote(part, safe="") for part in parts)
    return f"file://localhost/{encoded}"


def load_scenario_sentences(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s*", text) if s.strip()]


def tokenize_for_match(text: str) -> list[str]:
    """
    Нормализованная токенизация для сопоставления:
    - lower
    - разделяем дефисы/слэши/подчеркивания пробелом
    - удаляем остальную пунктуацию
    """
    s = text.lower()
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.findall(r"\w+", s)


def build_intervals_for_assets(
    transition_points: list[float],
    assets_count: int,
    full_duration: float,
    extend_tail: bool,
    tail_start_percent: float,
) -> list[tuple[float, float]]:
    intervals = [(transition_points[i], transition_points[i + 1]) for i in range(len(transition_points) - 1)]
    if assets_count <= len(intervals):
        return intervals[:assets_count]
    if not extend_tail or not intervals:
        return intervals

    tail_start_time = full_duration * max(0.0, min(100.0, tail_start_percent)) / 100.0
    pivot = 0
    for i, (_, end_time) in enumerate(intervals):
        if end_time >= tail_start_time:
            pivot = i
            break
    else:
        pivot = len(intervals) - 1

    preserved = intervals[:pivot]
    extra_count = assets_count - len(preserved)
    tail_from = preserved[-1][1] if preserved else 0.0
    if extra_count <= 0 or tail_from >= full_duration:
        return intervals[:assets_count]

    step = (full_duration - tail_from) / extra_count
    redistributed: list[tuple[float, float]] = []
    cur = tail_from
    for _ in range(extra_count):
        nxt = min(full_duration, cur + step)
        redistributed.append((cur, nxt))
        cur = nxt
    return preserved + redistributed


def sanitize_clip_intervals(
    intervals: list[tuple[float, float]],
    full_duration: float,
    min_dur: float = MIN_PHRASE_SEC,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """
    Если из-за схлопнутых raw_bounds часть интервалов получилась нулевой,
    вся длина озвучки оказывается «съедена» первыми клипами, а хвост — нулевой.
    Тогда part2/part3 XML получают клипы ~1 кадр. Поднимаем нули до min_dur и
    масштабируем так, чтобы сумма длительностей ровно уложилась в full_duration.
    """
    meta: dict[str, Any] = {"zero_or_negative_before": 0, "scaled": False, "clip_count": len(intervals)}
    if not intervals or full_duration <= 0:
        return intervals, meta

    durs: list[float] = []
    for a, b in intervals:
        d = float(b) - float(a)
        if d < 1e-6:
            meta["zero_or_negative_before"] += 1
            durs.append(min_dur)
        else:
            durs.append(d)

    total = sum(durs)
    if total > full_duration + 1e-6:
        scale = full_duration / total
        durs = [x * scale for x in durs]
        meta["scaled"] = True

    out: list[tuple[float, float]] = []
    t = 0.0
    for d in durs:
        out.append((t, t + d))
        t += d
    drift = t - full_duration
    if out and abs(drift) > 1e-3:
        last_s, last_e = out[-1]
        out[-1] = (last_s, last_e - drift)
    return out, meta


def collect_assets(assets_root: str) -> list[str]:
    ordered_assets: list[str] = []
    found_ordered = False
    for folder_name in ORDERED_FOLDERS:
        folder_path = os.path.join(assets_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        found_ordered = True
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(EXTS)]
        files.sort(key=natural_sort_key)
        ordered_assets.extend([os.path.join(folder_path, f) for f in files])

    if found_ordered:
        return ordered_assets

    files = [f for f in os.listdir(assets_root) if f.lower().endswith(EXTS)]
    files.sort(key=natural_sort_key)
    return [os.path.join(assets_root, f) for f in files]


def collect_assets_from_paths(asset_paths: list[str]) -> list[str]:
    assets: list[str] = []
    for folder_path in asset_paths:
        if not folder_path or not os.path.isdir(folder_path):
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(EXTS)]
        files.sort(key=natural_sort_key)
        assets.extend([os.path.join(folder_path, f) for f in files])
    return assets


def transcribe_words(audio_file: str, whisper_model: str, whisper_language: str) -> list[dict[str, float | str]]:
    def _resolve_whisper_repo(model_name: str) -> str:
        normalized = (model_name or "").strip()
        if not normalized:
            normalized = "medium"
        if "/" in normalized:
            return normalized
        return f"Systran/faster-whisper-{normalized}"

    def _ensure_whisper_model_ready(model_name: str) -> None:
        repo_id = _resolve_whisper_repo(model_name)
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "local_dir_use_symlinks": False,
            "resume_download": True,
        }
        if token:
            kwargs["token"] = token

        print(f"Проверяем/скачиваем Whisper-модель: {repo_id}", flush=True)
        model_path = snapshot_download(**kwargs)
        model_bin = os.path.join(model_path, "model.bin")
        if not os.path.isfile(model_bin):
            # If snapshot is incomplete, clear and download once more.
            print(f"model.bin не найден после скачивания, очищаем и перекачиваем: {model_path}", flush=True)
            shutil.rmtree(model_path, ignore_errors=True)
            model_path = snapshot_download(**kwargs)
            model_bin = os.path.join(model_path, "model.bin")
            if not os.path.isfile(model_bin):
                raise RuntimeError(f"Whisper model download incomplete: missing model.bin in {model_path}")

    def _run_whisper_once(source_wav: str):
        def _transcribe_with(device: str, compute_type: str):
            model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
            segments_iter, info = model.transcribe(
                source_wav,
                language=whisper_language,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,
            )
            # Materialize here to catch lazy runtime failures (e.g. missing CUDA libs)
            # inside this function so CPU fallback is guaranteed.
            segments = list(segments_iter)
            return segments, info

        try:
            return _transcribe_with("cuda", "float16")
        except Exception:
            print("Whisper GPU-режим недоступен/упал, повтор на CPU...", flush=True)
            return _transcribe_with("cpu", "int8")

    def _cleanup_broken_whisper_snapshot(err: Exception) -> bool:
        text = str(err)
        marker = "Unable to open file 'model.bin' in model '"
        if marker not in text:
            return False
        start = text.find(marker) + len(marker)
        end = text.find("'", start)
        if end <= start:
            return False
        model_dir = text[start:end]
        if not model_dir:
            return False
        try:
            paths_to_remove: list[str] = []
            normalized = os.path.normpath(model_dir)
            if os.path.isdir(normalized):
                paths_to_remove.append(normalized)
            # If snapshot cleanup is not enough, remove the whole model repo cache.
            # Example:
            # ...\hub\models--Systran--faster-whisper-medium\snapshots\08e178...
            snapshots_token = f"{os.sep}snapshots{os.sep}"
            snap_idx = normalized.lower().find(snapshots_token.lower())
            if snap_idx > 0:
                repo_cache_dir = normalized[:snap_idx]
                if os.path.isdir(repo_cache_dir):
                    paths_to_remove.append(repo_cache_dir)
            removed_any = False
            for path in dict.fromkeys(paths_to_remove):
                print(f"Поврежден кэш модели Whisper, удаляем: {path}", flush=True)
                shutil.rmtree(path, ignore_errors=True)
                removed_any = removed_any or (not os.path.exists(path))
            return removed_any
        except Exception:
            return False
        return False

    def _cleanup_hf_model_cache(model_name: str) -> bool:
        """
        Deeper cleanup when snapshot-only cleanup is not enough.
        Removes model repo cache under huggingface hub, e.g.
        .../huggingface/hub/models--Systran--faster-whisper-medium
        """
        repo_id = _resolve_whisper_repo(model_name)
        safe = repo_id.replace("/", "--")
        repo_dir_name = f"models--{safe}"
        candidates: list[str] = []

        hf_home = os.environ.get("HF_HOME", "").strip()
        if hf_home:
            candidates.append(os.path.join(hf_home, "hub", repo_dir_name))

        if os.name == "nt":
            local = os.environ.get("LOCALAPPDATA", "").strip()
            profile = os.environ.get("USERPROFILE", "").strip()
            if local:
                candidates.append(os.path.join(local, "huggingface", "hub", repo_dir_name))
            if profile:
                candidates.append(os.path.join(profile, ".cache", "huggingface", "hub", repo_dir_name))
        else:
            home = os.path.expanduser("~")
            candidates.append(os.path.join(home, ".cache", "huggingface", "hub", repo_dir_name))

        removed_any = False
        for path in dict.fromkeys([os.path.normpath(p) for p in candidates if p]):
            try:
                if os.path.isdir(path):
                    print(f"Глубокая очистка кэша HF для модели: {path}", flush=True)
                    shutil.rmtree(path, ignore_errors=True)
                    removed_any = removed_any or (not os.path.exists(path))
            except Exception:
                continue
        return removed_any

    temp_wav_path: str | None = None
    try:
        _ensure_whisper_model_ready(whisper_model)
        # Нормализация аудио в WAV 16k mono снижает ошибки декодера ffmpeg/whisper.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_wav_path = tmp.name
        convert_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_file,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            temp_wav_path,
        ]
        subprocess.run(convert_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        try:
            segments, _ = _run_whisper_once(temp_wav_path)
        except Exception as exc:
            if _cleanup_broken_whisper_snapshot(exc):
                try:
                    segments, _ = _run_whisper_once(temp_wav_path)
                except Exception as exc_second:
                    if _cleanup_hf_model_cache(whisper_model):
                        segments, _ = _run_whisper_once(temp_wav_path)
                    else:
                        raise exc_second
            else:
                raise
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    words: list[dict[str, float | str]] = []
    for segment in segments:
        for word in segment.words:
            tokens = tokenize_for_match(word.word or "")
            if not tokens:
                continue
            for t in tokens:
                words.append({"text": t, "start": float(word.start), "end": float(word.end)})
    return words


def map_sentence_bounds_standard(sentences: list[str], whisper_words: list[dict[str, float | str]]) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    whisper_tokens = [str(w["text"]) for w in whisper_words]
    raw_bounds: list[dict[str, float]] = [{"start": 0.0, "end": 0.0} for _ in range(len(sentences))]
    diagnostics: list[dict[str, Any]] = []
    cursor = 0
    last_end = 0.0

    for chunk_start in range(0, len(sentences), LOCAL_ALIGN_CHUNK_SENTENCES):
        chunk_end = min(len(sentences), chunk_start + LOCAL_ALIGN_CHUNK_SENTENCES)
        chunk_sentences = sentences[chunk_start:chunk_end]

        chunk_tokens: list[str] = []
        chunk_token_to_sentence: list[int] = []
        for local_idx, sentence in enumerate(chunk_sentences):
            s_idx = chunk_start + local_idx
            tokens = tokenize_for_match(sentence)
            for token in tokens:
                chunk_tokens.append(token)
                chunk_token_to_sentence.append(s_idx)

        if not chunk_tokens:
            for s_idx in range(chunk_start, chunk_end):
                raw_bounds[s_idx] = {"start": last_end, "end": last_end}
                diagnostics.append(
                    {
                        "sentence_index": s_idx,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "coverage": 0.0,
                        "matched": False,
                        "reason": "empty_sentence_chunk",
                        "window_start": cursor,
                        "window_end": cursor,
                        "whisper_cursor_after": cursor,
                    }
                )
            continue

        best = None
        # Пункт 1: уже использованные слова Whisper больше не переиспользуем.
        # Ищем только вперед от текущего курсора.
        start_search = max(0, cursor)
        for ahead in LOCAL_ALIGN_WINDOW_AHEAD_STEPS:
            end_search = min(len(whisper_tokens), max(start_search + 1, cursor + ahead))
            window_tokens = whisper_tokens[start_search:end_search]
            matcher = difflib.SequenceMatcher(None, chunk_tokens, window_tokens)
            matched_pairs: list[tuple[int, int]] = []
            for a_idx, b_idx, size in matcher.get_matching_blocks():
                for i in range(size):
                    matched_pairs.append((a_idx + i, start_search + b_idx + i))
            matched_chunk_pos = {a for a, _ in matched_pairs}
            coverage = len(matched_chunk_pos) / max(1, len(chunk_tokens))
            candidate = {
                "coverage": coverage,
                "pairs": matched_pairs,
                "window_start": start_search,
                "window_end": end_search,
                "ahead": ahead,
            }
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate
            # достаточное совпадение - дальше окно не раздуваем
            if coverage >= 0.35:
                break

        assert best is not None
        sentence_to_whisper_idx: dict[int, list[int]] = {}
        for chunk_pos, whisper_abs in best["pairs"]:
            s_idx = chunk_token_to_sentence[chunk_pos]
            sentence_to_whisper_idx.setdefault(s_idx, []).append(whisper_abs)

        max_matched_whisper_idx = cursor
        for s_idx in range(chunk_start, chunk_end):
            widx = sentence_to_whisper_idx.get(s_idx, [])
            token_count = len(tokenize_for_match(sentences[s_idx]))
            if widx:
                first_idx = min(widx)
                last_idx = max(widx)
                start = max(last_end, float(whisper_words[first_idx]["start"]))
                end = max(start + 0.04, float(whisper_words[last_idx]["end"]))
                # Пункт 2+3: строгая монотонность и ограничение длительности фразы.
                duration = end - start
                if duration < MIN_PHRASE_SEC:
                    end = start + MIN_PHRASE_SEC
                elif duration > MAX_PHRASE_SEC:
                    end = start + MAX_PHRASE_SEC
                raw_bounds[s_idx] = {"start": start, "end": end}
                last_end = end
                max_matched_whisper_idx = max(max_matched_whisper_idx, last_idx + 1)
                diagnostics.append(
                    {
                        "sentence_index": s_idx,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "coverage": best["coverage"],
                        "matched": True,
                        "sentence_token_count": token_count,
                        "window_start": best["window_start"],
                        "window_end": best["window_end"],
                        "window_ahead": best["ahead"],
                        "whisper_cursor_after": max_matched_whisper_idx,
                    }
                )
            else:
                # no-anchor fallback отключен по вашей просьбе:
                # здесь только минимальный безопасный шаг, без интерполяции от "якорей"
                start = last_end
                end = start + MIN_PHRASE_SEC
                raw_bounds[s_idx] = {"start": start, "end": end}
                last_end = end
                diagnostics.append(
                    {
                        "sentence_index": s_idx,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                        "coverage": best["coverage"],
                        "matched": False,
                        "reason": "no_word_match_in_chunk",
                        "sentence_token_count": token_count,
                        "window_start": best["window_start"],
                        "window_end": best["window_end"],
                        "window_ahead": best["ahead"],
                        "whisper_cursor_after": max_matched_whisper_idx,
                    }
                )
        cursor = max_matched_whisper_idx

    # Дополнительный safety-pass для строгой монотонности по всему массиву.
    prev_end = 0.0
    for b in raw_bounds:
        s = max(prev_end, b["start"])
        e = max(s + MIN_PHRASE_SEC, b["end"])
        if e - s > MAX_PHRASE_SEC:
            e = s + MAX_PHRASE_SEC
        b["start"] = s
        b["end"] = e
        prev_end = e

    return raw_bounds, diagnostics


def map_sentence_bounds_anchor(sentences: list[str], whisper_words: list[dict[str, float | str]]) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    whisper_tokens = [str(w["text"]) for w in whisper_words]
    raw_bounds: list[dict[str, float]] = [{"start": 0.0, "end": 0.0} for _ in range(len(sentences))]
    diagnostics: list[dict[str, Any]] = []
    cursor = 0
    last_end = 0.0

    for block_start in range(0, len(sentences), ANCHOR_BLOCK_SENTENCES):
        block_end = min(len(sentences), block_start + ANCHOR_BLOCK_SENTENCES)
        block_sentences = sentences[block_start:block_end]

        block_tokens: list[str] = []
        token_to_sentence: list[int] = []
        for local_idx, sentence in enumerate(block_sentences):
            s_idx = block_start + local_idx
            for token in tokenize_for_match(sentence):
                block_tokens.append(token)
                token_to_sentence.append(s_idx)

        if not block_tokens:
            for s_idx in range(block_start, block_end):
                start = last_end
                end = start + MIN_PHRASE_SEC
                raw_bounds[s_idx] = {"start": start, "end": end}
                last_end = end
                diagnostics.append(
                    {
                        "sentence_index": s_idx,
                        "matched": False,
                        "coverage": 0.0,
                        "reason": "empty_block_tokens",
                        "block_start": block_start,
                        "block_end": block_end,
                        "window_start": cursor,
                        "window_end": cursor,
                        "whisper_cursor_after": cursor,
                    }
                )
            continue

        start_search = max(0, cursor)
        best = None
        for ahead in LOCAL_ALIGN_WINDOW_AHEAD_STEPS:
            end_search = min(len(whisper_tokens), max(start_search + 1, cursor + ahead))
            window_tokens = whisper_tokens[start_search:end_search]
            matcher = difflib.SequenceMatcher(None, block_tokens, window_tokens)
            pairs: list[tuple[int, int]] = []
            for a_idx, b_idx, size in matcher.get_matching_blocks():
                for i in range(size):
                    pairs.append((a_idx + i, start_search + b_idx + i))
            coverage = (len({a for a, _ in pairs}) / max(1, len(block_tokens)))
            candidate = {
                "pairs": pairs,
                "coverage": coverage,
                "window_start": start_search,
                "window_end": end_search,
                "window_ahead": ahead,
            }
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate
            if coverage >= 0.35:
                break

        assert best is not None
        sentence_to_widx: dict[int, list[int]] = {}
        for block_pos, w_abs in best["pairs"]:
            s_idx = token_to_sentence[block_pos]
            sentence_to_widx.setdefault(s_idx, []).append(w_abs)

        max_idx = cursor
        for s_idx in range(block_start, block_end):
            idxs = sentence_to_widx.get(s_idx, [])
            if idxs:
                first_idx = min(idxs)
                last_idx = max(idxs)
                start = max(last_end, float(whisper_words[first_idx]["start"]))
                end = max(start + MIN_PHRASE_SEC, float(whisper_words[last_idx]["end"]))
                if end - start > MAX_PHRASE_SEC:
                    end = start + MAX_PHRASE_SEC
                raw_bounds[s_idx] = {"start": start, "end": end}
                last_end = end
                max_idx = max(max_idx, last_idx + 1)
                diagnostics.append(
                    {
                        "sentence_index": s_idx,
                        "matched": True,
                        "coverage": best["coverage"],
                        "block_start": block_start,
                        "block_end": block_end,
                        "window_start": best["window_start"],
                        "window_end": best["window_end"],
                        "window_ahead": best["window_ahead"],
                        "whisper_cursor_after": max_idx,
                    }
                )
            else:
                start = last_end
                end = start + MIN_PHRASE_SEC
                raw_bounds[s_idx] = {"start": start, "end": end}
                last_end = end
                diagnostics.append(
                    {
                        "sentence_index": s_idx,
                        "matched": False,
                        "coverage": best["coverage"],
                        "reason": "no_word_match_in_anchor_block",
                        "block_start": block_start,
                        "block_end": block_end,
                        "window_start": best["window_start"],
                        "window_end": best["window_end"],
                        "window_ahead": best["window_ahead"],
                        "whisper_cursor_after": max_idx,
                    }
                )
        cursor = max_idx

    # global monotonic safety
    prev_end = 0.0
    for b in raw_bounds:
        s = max(prev_end, b["start"])
        e = max(s + MIN_PHRASE_SEC, b["end"])
        if e - s > MAX_PHRASE_SEC:
            e = s + MAX_PHRASE_SEC
        b["start"] = s
        b["end"] = e
        prev_end = e
    return raw_bounds, diagnostics


def _block_forced_match_span_allowed(
    first_idx: int,
    last_idx: int,
    sentence_token_count: int,
    whisper_words: list[dict[str, float | str]],
) -> bool:
    """Один матч не должен покрывать слишком много слов Whisper или слишком длинный отрезок времени."""
    span_words = last_idx - first_idx + 1
    max_words = min(
        BLOCK_FORCED_MAX_WHISPER_WORD_SPAN_ABSOLUTE,
        max(8, int(BLOCK_FORCED_MAX_WHISPER_WORD_SPAN_RATIO * max(1, sentence_token_count))),
    )
    if span_words > max_words:
        return False
    t0 = float(whisper_words[first_idx]["start"])
    t1 = float(whisper_words[last_idx]["end"])
    if t1 - t0 > BLOCK_FORCED_MAX_MATCH_AUDIO_SPAN_SEC:
        return False
    return True


def _match_sentence_at_cursor(
    sentence: str,
    whisper_tokens: list[str],
    whisper_words: list[dict[str, float | str]],
    cursor: int,
    min_coverage: float,
    max_ahead_words: int,
) -> tuple[int, int, float, int, int] | None:
    """Возвращает (first_idx, last_idx, coverage, window_start, window_end) или None."""
    tokens = tokenize_for_match(sentence)
    if not tokens:
        return None
    start_search = max(0, cursor)
    end_search = min(len(whisper_tokens), max(start_search + 1, cursor + max_ahead_words))
    window_tokens = whisper_tokens[start_search:end_search]
    matcher = difflib.SequenceMatcher(None, tokens, window_tokens)
    pairs: list[tuple[int, int]] = []
    for a_idx, b_idx, size in matcher.get_matching_blocks():
        for j in range(size):
            pairs.append((a_idx + j, start_search + b_idx + j))
    if not pairs:
        return None
    coverage = len({a for a, _ in pairs}) / max(1, len(tokens))
    if coverage < min_coverage:
        return None
    first_idx = min(w for _, w in pairs)
    last_idx = max(w for _, w in pairs)
    if first_idx < cursor:
        return None
    if not _block_forced_match_span_allowed(first_idx, last_idx, len(tokens), whisper_words):
        return None
    return (first_idx, last_idx, coverage, start_search, end_search)


def map_sentence_bounds_block_forced(
    sentences: list[str], whisper_words: list[dict[str, float | str]], full_duration: float
) -> tuple[list[dict[str, float]], list[dict[str, Any]]]:
    """
    Глобальное выравнивание: курсор по Whisper только вперёд, слова не переиспользуются.
    Сначала строгий матч, затем узкий rescue; безматчевые фразы заполняются только
    равномерно между двумя успешными матчами (или от 0 / до full_duration по краям).
    """
    n = len(sentences)
    whisper_tokens = [str(w["text"]) for w in whisper_words]
    raw_bounds: list[dict[str, float]] = [{"start": 0.0, "end": 0.0} for _ in range(n)]
    diagnostics: list[dict[str, Any]] = []
    safe_full = max(0.0, full_duration)

    strict_cov = 0.45
    rescue_cov = 0.28
    rescue_ahead = 400

    w_cursor = 0
    last_time_end = 0.0
    matched_flags = [False] * n

    def diag_base(s_idx: int) -> dict[str, Any]:
        bs = (s_idx // BLOCK_FORCED_SENTENCES) * BLOCK_FORCED_SENTENCES
        be = min(n, bs + BLOCK_FORCED_SENTENCES)
        return {
            "sentence_index": s_idx,
            "block_start": bs,
            "block_end": be,
        }

    for s_idx in range(n):
        tokens = tokenize_for_match(sentences[s_idx])
        if not tokens:
            diagnostics.append(
                {
                    **diag_base(s_idx),
                    "matched": False,
                    "coverage": 0.0,
                    "reason": "empty_sentence_tokens",
                    "match_kind": "empty_tokens",
                    "window_start": w_cursor,
                    "window_end": w_cursor,
                    "window_ahead": 0,
                    "first_whisper_idx": None,
                    "last_whisper_idx": None,
                    "whisper_cursor_after": w_cursor,
                }
            )
            continue

        hit: tuple[int, int, float, int, int] | None = None
        match_kind = "strict"
        for ahead in LOCAL_ALIGN_WINDOW_AHEAD_STEPS:
            hit = _match_sentence_at_cursor(
                sentences[s_idx], whisper_tokens, whisper_words, w_cursor, strict_cov, ahead
            )
            if hit is not None:
                break
        if hit is None:
            hit = _match_sentence_at_cursor(
                sentences[s_idx], whisper_tokens, whisper_words, w_cursor, rescue_cov, rescue_ahead
            )
            match_kind = "rescue" if hit is not None else "none"

        if hit is None:
            diagnostics.append(
                {
                    **diag_base(s_idx),
                    "matched": False,
                    "coverage": 0.0,
                    "reason": "no_sentence_match",
                    "match_kind": "none",
                    "window_start": w_cursor,
                    "window_end": min(len(whisper_tokens), w_cursor + rescue_ahead),
                    "window_ahead": rescue_ahead,
                    "first_whisper_idx": None,
                    "last_whisper_idx": None,
                    "whisper_cursor_after": w_cursor,
                }
            )
            continue

        first_idx, last_idx, coverage, ws, we = hit
        s_w = float(whisper_words[first_idx]["start"])
        e_w = float(whisper_words[last_idx]["end"])
        s = max(last_time_end, s_w)
        e = max(s + MIN_PHRASE_SEC, e_w)
        if e - s > MAX_PHRASE_SEC:
            e = s + MAX_PHRASE_SEC
        raw_bounds[s_idx] = {"start": s, "end": e}
        matched_flags[s_idx] = True
        w_cursor = last_idx + 1
        last_time_end = e
        diagnostics.append(
            {
                **diag_base(s_idx),
                "matched": True,
                "coverage": coverage,
                "reason": "rescue_low_threshold" if match_kind == "rescue" else None,
                "match_kind": match_kind,
                "window_start": ws,
                "window_end": we,
                "window_ahead": we - ws,
                "first_whisper_idx": first_idx,
                "last_whisper_idx": last_idx,
                "whisper_cursor_after": w_cursor,
            }
        )

    matched_idxs = [i for i in range(n) if matched_flags[i]]

    if not matched_idxs:
        step = safe_full / n if n else 0.0
        cur = 0.0
        for i in range(n):
            nxt = cur + step if i < n - 1 else safe_full
            raw_bounds[i] = {"start": cur, "end": nxt}
            cur = nxt
            diagnostics[i] = {
                **diag_base(i),
                "matched": False,
                "coverage": 0.0,
                "reason": "filled_uniform_no_anchor",
                "match_kind": "filled_uniform",
                "window_start": None,
                "window_end": None,
                "window_ahead": None,
                "first_whisper_idx": None,
                "last_whisper_idx": None,
                "whisper_cursor_after": w_cursor,
            }
    else:
        first_m = matched_idxs[0]
        if first_m > 0:
            t0 = 0.0
            t1 = float(raw_bounds[first_m]["start"])
            span = max(MIN_PHRASE_SEC * first_m, t1 - t0)
            step = span / first_m if first_m else 0.0
            cur = t0
            for j in range(first_m):
                nxt = cur + step if j < first_m - 1 else t1
                raw_bounds[j] = {"start": cur, "end": nxt}
                cur = nxt
                diagnostics[j] = {
                    **diag_base(j),
                    "matched": False,
                    "coverage": 0.0,
                    "reason": "filled_leading",
                    "match_kind": "filled_leading",
                    "window_start": None,
                    "window_end": None,
                    "window_ahead": None,
                    "first_whisper_idx": None,
                    "last_whisper_idx": None,
                    "whisper_cursor_after": w_cursor,
                }

        for mi in range(len(matched_idxs) - 1):
            a = matched_idxs[mi]
            b = matched_idxs[mi + 1]
            if b <= a + 1:
                continue
            t0 = float(raw_bounds[a]["end"])
            t1 = float(raw_bounds[b]["start"])
            count = b - a - 1
            if count <= 0:
                continue
            if t1 > t0:
                step = (t1 - t0) / count
                cur = t0
                for k in range(a + 1, b):
                    kk = k - (a + 1)
                    nxt = cur + step if kk < count - 1 else t1
                    raw_bounds[k] = {"start": cur, "end": nxt}
                    cur = nxt
                    diagnostics[k] = {
                        **diag_base(k),
                        "matched": False,
                        "coverage": 0.0,
                        "reason": "filled_between_anchors",
                        "match_kind": "filled_between",
                        "window_start": None,
                        "window_end": None,
                        "window_ahead": None,
                        "first_whisper_idx": None,
                        "last_whisper_idx": None,
                        "whisper_cursor_after": w_cursor,
                    }
            else:
                cur = t0
                for k in range(a + 1, b):
                    nxt = min(safe_full, cur + MIN_PHRASE_SEC)
                    raw_bounds[k] = {"start": cur, "end": nxt}
                    cur = nxt
                    diagnostics[k] = {
                        **diag_base(k),
                        "matched": False,
                        "coverage": 0.0,
                        "reason": "filled_between_anchors_degenerate",
                        "match_kind": "filled_between",
                        "window_start": None,
                        "window_end": None,
                        "window_ahead": None,
                        "first_whisper_idx": None,
                        "last_whisper_idx": None,
                        "whisper_cursor_after": w_cursor,
                    }

        last_m = matched_idxs[-1]
        if last_m < n - 1:
            t0 = float(raw_bounds[last_m]["end"])
            t1 = safe_full
            cnt = n - 1 - last_m
            span = max(MIN_PHRASE_SEC * cnt, t1 - t0)
            step = span / cnt if cnt else 0.0
            cur = t0
            for j in range(last_m + 1, n):
                nxt = cur + step if j < n - 1 else t1
                raw_bounds[j] = {"start": cur, "end": nxt}
                cur = nxt
                diagnostics[j] = {
                    **diag_base(j),
                    "matched": False,
                    "coverage": 0.0,
                    "reason": "filled_trailing",
                    "match_kind": "filled_trailing",
                    "window_start": None,
                    "window_end": None,
                    "window_ahead": None,
                    "first_whisper_idx": None,
                    "last_whisper_idx": None,
                    "whisper_cursor_after": w_cursor,
                }

    prev_end = 0.0
    for i in range(n):
        s = max(prev_end, float(raw_bounds[i]["start"]))
        e = max(s + MIN_PHRASE_SEC, float(raw_bounds[i]["end"]))
        if e - s > MAX_PHRASE_SEC:
            e = s + MAX_PHRASE_SEC
        if e > safe_full:
            e = safe_full
        if e <= s:
            e = min(safe_full, s + MIN_PHRASE_SEC)
        raw_bounds[i] = {"start": s, "end": e}
        prev_end = e

    return raw_bounds, diagnostics


def stabilize_raw_bounds(
    raw_bounds: list[dict[str, float]],
    diagnostics: list[dict[str, Any]],
    full_duration: float,
) -> list[dict[str, float]]:
    """
    Убирает длинные провалы, когда много low_conf подряд:
    распределяет время между ближайшими "надежными" фразами.
    """
    n = len(raw_bounds)
    if n == 0:
        return raw_bounds
    matched = [bool(d.get("matched")) for d in diagnostics]
    stable = [{"start": b["start"], "end": b["end"]} for b in raw_bounds]

    # leading low_conf -> равномерно до первого matched
    first_match = next((i for i, m in enumerate(matched) if m), None)
    if first_match is not None and first_match > 0:
        t0 = 0.0
        t1 = max(t0, stable[first_match]["start"])
        step = (t1 - t0) / first_match if first_match else 0.0
        cur = t0
        for i in range(first_match):
            nxt = cur + step
            stable[i]["start"] = cur
            stable[i]["end"] = nxt
            cur = nxt

    # trailing low_conf -> равномерно от последнего matched до конца аудио
    last_match = next((i for i in range(n - 1, -1, -1) if matched[i]), None)
    if last_match is not None and last_match < n - 1:
        t0 = stable[last_match]["end"]
        t1 = full_duration
        cnt = n - 1 - last_match
        step = (t1 - t0) / cnt if cnt else 0.0
        cur = t0
        for i in range(last_match + 1, n):
            nxt = cur + step
            stable[i]["start"] = cur
            stable[i]["end"] = nxt
            cur = nxt

    # внутренние low_conf сегменты между matched
    i = 0
    while i < n:
        if matched[i]:
            i += 1
            continue
        seg_start = i
        while i < n and not matched[i]:
            i += 1
        seg_end = i - 1
        left = seg_start - 1
        right = i if i < n else None
        if left >= 0 and right is not None and matched[left] and matched[right]:
            t0 = stable[left]["end"]
            t1 = stable[right]["start"]
            count = seg_end - seg_start + 1
            if count > 0 and t1 >= t0:
                step = (t1 - t0) / count
                cur = t0
                for j in range(seg_start, seg_end + 1):
                    nxt = cur + step
                    stable[j]["start"] = cur
                    stable[j]["end"] = nxt
                    cur = nxt

    # safety: ensure monotonic and non-negative durations
    prev_end = 0.0
    for b in stable:
        start = max(prev_end, b["start"])
        end = max(start + 0.04, b["end"])
        b["start"] = start
        b["end"] = end
        prev_end = end
    if stable[-1]["end"] > full_duration and full_duration > 0:
        scale = full_duration / stable[-1]["end"]
        for b in stable:
            b["start"] *= scale
            b["end"] *= scale
    return stable


def build_transition_points(raw_bounds: list[dict[str, float]], full_duration: float) -> list[float]:
    points = [0.0]
    for i in range(len(raw_bounds) - 1):
        points.append((raw_bounds[i]["end"] + raw_bounds[i + 1]["start"]) / 2)
    points.append(full_duration)
    return points


def build_sequential_clip_intervals_from_bounds(
    raw_bounds: list[dict[str, float]], clip_count: int
) -> list[tuple[float, float]]:
    """
    Длительность клипа i = end_i - start_i по raw_bounds; на таймлайне клипы идут подряд от 0 (natural_sum).
    """
    out: list[tuple[float, float]] = []
    t = 0.0
    for i in range(max(0, clip_count)):
        b = raw_bounds[i]
        dur = max(1e-6, float(b["end"]) - float(b["start"]))
        out.append((t, t + dur))
        t += dur
    return out


def build_absolute_clip_intervals_from_bounds(
    raw_bounds: list[dict[str, float]], clip_count: int, full_duration: float
) -> list[tuple[float, float]]:
    """
    Одна шкала с полной дорожкой озвучки: клип i занимает [start_i, start_{i+1}),
    последний клип — до конца файла full_duration (включая хвост без отдельных картинок).
    """
    out: list[tuple[float, float]] = []
    fd = max(0.0, float(full_duration))
    n = max(0, clip_count)
    for i in range(n):
        s = float(raw_bounds[i]["start"])
        if i + 1 < n:
            e = float(raw_bounds[i + 1]["start"])
        else:
            e = fd
        e = min(e, fd)
        s = min(s, fd)
        if e <= s:
            e = min(fd, s + MIN_PHRASE_SEC)
        out.append((s, e))
    return out


def extend_sequential_intervals_for_extra_assets(
    intervals: list[tuple[float, float]],
    assets_count: int,
    full_duration: float,
    extend_tail: bool,
    tail_start_percent: float,
) -> list[tuple[float, float]]:
    """Если картинок больше, чем фраз — дописываем хвостовые интервалы (как extend_tail, но от конца суммы фраз)."""
    if len(intervals) >= assets_count:
        return intervals[:assets_count]
    if not extend_tail or not intervals:
        return intervals[:assets_count]
    last_end = intervals[-1][1]
    extra = assets_count - len(intervals)
    if extra <= 0 or last_end >= full_duration:
        return intervals[:assets_count]
    tail_start_time = full_duration * max(0.0, min(100.0, tail_start_percent)) / 100.0
    if last_end < tail_start_time:
        return intervals[:assets_count]
    step = (full_duration - last_end) / extra
    cur = last_end
    extended = list(intervals)
    for _ in range(extra):
        nxt = min(full_duration, cur + step)
        extended.append((cur, nxt))
        cur = nxt
    return extended[:assets_count]


def clamp_large_inter_phrase_gaps(
    raw_bounds: list[dict[str, float]],
    max_gap_sec: float,
    full_duration: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """
    Ограничивает гигантские разрывы между соседними фразами.
    Если gap > max_gap_sec, следующая фраза сдвигается ближе к предыдущей:
    next.start = prev.end + max_gap_sec, next.end сохраняет исходную длительность (с safety clamp).
    """
    if not raw_bounds:
        return raw_bounds, []
    fixed = [{"start": b["start"], "end": b["end"]} for b in raw_bounds]
    anomalies: list[dict[str, float]] = []
    prev_end = fixed[0]["end"]
    for i in range(1, len(fixed)):
        cur = fixed[i]
        duration = max(MIN_PHRASE_SEC, cur["end"] - cur["start"])
        gap = cur["start"] - prev_end
        if gap > max_gap_sec:
            old_start = cur["start"]
            old_end = cur["end"]
            cur["start"] = prev_end + max_gap_sec
            cur["end"] = min(full_duration, cur["start"] + min(duration, MAX_PHRASE_SEC))
            if cur["end"] - cur["start"] < MIN_PHRASE_SEC:
                cur["end"] = min(full_duration, cur["start"] + MIN_PHRASE_SEC)
            anomalies.append(
                {
                    "sentence_index": i,
                    "gap_before": gap,
                    "old_start": old_start,
                    "old_end": old_end,
                    "new_start": cur["start"],
                    "new_end": cur["end"],
                }
            )
        if cur["start"] < prev_end:
            cur["start"] = prev_end
            cur["end"] = max(cur["start"] + MIN_PHRASE_SEC, min(full_duration, cur["end"]))
        prev_end = cur["end"]
    return fixed, anomalies


def enforce_bounds_invariants(
    raw_bounds: list[dict[str, float]],
    full_duration: float,
) -> tuple[list[dict[str, float]], list[dict[str, float | int | str]]]:
    """
    Жесткая нормализация таймкодов:
    - no negative times
    - no end < start
    - monotonic sequence
    - everything inside [0, full_duration]
    """
    if not raw_bounds:
        return raw_bounds, []

    fixed: list[dict[str, float]] = []
    changes: list[dict[str, float | int | str]] = []
    prev_end = 0.0
    safe_full = max(0.0, full_duration)

    for i, b in enumerate(raw_bounds):
        old_start = float(b["start"])
        old_end = float(b["end"])
        desired = max(MIN_PHRASE_SEC, min(MAX_PHRASE_SEC, old_end - old_start))

        # start must be monotonic and inside audio.
        start = max(prev_end, old_start, 0.0)
        start = min(start, safe_full)

        # try keeping desired duration, but never exceed audio.
        end = start + desired
        if end > safe_full:
            end = safe_full

        # hard safety: end cannot be before start.
        if end < start:
            end = start

        # keep minimal duration only when there is room.
        if end - start < MIN_PHRASE_SEC and (safe_full - start) >= MIN_PHRASE_SEC:
            end = start + MIN_PHRASE_SEC
        if end > safe_full:
            end = safe_full
        if end < start:
            end = start

        if (abs(start - old_start) > 1e-9) or (abs(end - old_end) > 1e-9):
            changes.append(
                {
                    "sentence_index": i,
                    "reason": "bounds_invariant_fix",
                    "old_start": old_start,
                    "old_end": old_end,
                    "new_start": start,
                    "new_end": end,
                }
            )
        fixed.append({"start": start, "end": end})
        prev_end = end

    return fixed, changes


def enforce_bounds_invariants_soft(
    raw_bounds: list[dict[str, float]],
    full_duration: float,
) -> tuple[list[dict[str, float]], list[dict[str, float | int | str]]]:
    """
    Мягкая нормализация для block_forced: монотонность и [0, full], без схлопывания хвоста в нули.
    """
    if not raw_bounds:
        return raw_bounds, []
    safe_full = max(0.0, full_duration)
    fixed: list[dict[str, float]] = []
    changes: list[dict[str, float | int | str]] = []
    prev_end = 0.0
    for i, b in enumerate(raw_bounds):
        old_s, old_e = float(b["start"]), float(b["end"])
        s = max(prev_end, old_s, 0.0)
        e = max(s + MIN_PHRASE_SEC, old_e)
        if e - s > MAX_PHRASE_SEC:
            e = s + MAX_PHRASE_SEC
        if e > safe_full:
            e = safe_full
            s = min(s, max(prev_end, e - MIN_PHRASE_SEC))
        if e <= s:
            e = min(safe_full, s + MIN_PHRASE_SEC)
        if (abs(s - old_s) > 1e-9) or (abs(e - old_e) > 1e-9):
            changes.append(
                {
                    "sentence_index": i,
                    "reason": "bounds_invariant_soft",
                    "old_start": old_s,
                    "old_end": old_e,
                    "new_start": s,
                    "new_end": e,
                }
            )
        fixed.append({"start": s, "end": e})
        prev_end = e
    return fixed, changes


def build_detailed_timing_report(
    sentences: list[str],
    raw_bounds: list[dict[str, float]],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = min(len(sentences), len(raw_bounds), len(diagnostics))
    for i in range(n):
        start = float(raw_bounds[i]["start"])
        end = float(raw_bounds[i]["end"])
        prev_gap = None if i == 0 else (start - float(raw_bounds[i - 1]["end"]))
        next_gap = None if i + 1 >= n else (float(raw_bounds[i + 1]["start"]) - end)
        d = diagnostics[i] if i < len(diagnostics) else {}
        rows.append(
            {
                "sentence_index": i,
                "sentence_text": sentences[i],
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "gap_from_prev_sec": prev_gap,
                "gap_to_next_sec": next_gap,
                "matched": bool(d.get("matched")),
                "coverage": d.get("coverage"),
                "reason": d.get("reason"),
                "window_start": d.get("window_start"),
                "window_end": d.get("window_end"),
                "window_ahead": d.get("window_ahead"),
                "chunk_start": d.get("chunk_start"),
                "chunk_end": d.get("chunk_end"),
                "block_start": d.get("block_start"),
                "block_end": d.get("block_end"),
                "whisper_cursor_after": d.get("whisper_cursor_after"),
            }
        )
    return rows


def get_ffmpeg_encoders() -> str:
    result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, check=False)
    return (result.stdout or "") + "\n" + (result.stderr or "")


def detect_video_encoder(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred

    encoders_dump = get_ffmpeg_encoders().lower()
    system_name = platform.system().lower()
    if "h264_nvenc" in encoders_dump:
        return "h264_nvenc"
    if system_name == "windows" and "h264_amf" in encoders_dump:
        return "h264_amf"
    if system_name == "darwin" and "h264_videotoolbox" in encoders_dump:
        return "h264_videotoolbox"
    return "libx264"


def build_video_codec_args(encoder: str, fps: int) -> list[str]:
    safe_fps = str(max(1, fps))
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p", "-r", safe_fps]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "speed", "-pix_fmt", "yuv420p", "-r", safe_fps]
    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M", "-pix_fmt", "yuv420p", "-r", safe_fps]
    return ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", safe_fps]


def render_clip(asset: str, duration: float, out_name: str, fps: int, encoder: str) -> None:
    def run_once(target_encoder: str) -> None:
        is_video = asset.lower().endswith((".mp4", ".mov", ".mkv"))
        cmd = ["ffmpeg", "-y"]
        if target_encoder == "h264_nvenc":
            cmd += ["-hwaccel", "cuda"]
        if is_video:
            cmd += ["-i", asset, "-t", str(duration)]
        else:
            cmd += ["-loop", "1", "-i", asset, "-t", str(duration)]
        cmd += ["-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1"]
        cmd += build_video_codec_args(target_encoder, fps)
        cmd += ["-an", out_name]
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)

    attempt_errors: list[str] = []
    attempts = len(RENDER_RETRY_DELAYS_SEC) + 1
    for attempt_idx in range(attempts):
        attempt_no = attempt_idx + 1
        try:
            # 1-я попытка на выбранном энкодере, дальше на CPU для устойчивости.
            encoder_for_attempt = encoder if attempt_idx == 0 else "libx264"
            run_once(encoder_for_attempt)
            return
        except subprocess.CalledProcessError as err:
            msg = (err.stderr or err.stdout or "").strip()
            attempt_errors.append(
                f"attempt {attempt_no}/{attempts} encoder={encoder_for_attempt}: {msg[:1200]}"
            )
            if attempt_idx < len(RENDER_RETRY_DELAYS_SEC):
                time.sleep(RENDER_RETRY_DELAYS_SEC[attempt_idx])

    raise RuntimeError(
        "Не удалось отрендерить клип после повторных попыток.\n"
        f"Файл: {asset}\n"
        + "\n".join(attempt_errors)
    )


def concatenate_final_video(output_folder: str, audio_file: str, final_output_path: str, encoder: str, fps: int) -> None:
    def _ffmpeg_concat_escape(path: str) -> str:
        # ffmpeg concat format wraps path in single quotes; escape embedded quotes safely.
        return path.replace("'", r"'\''")

    clips = sorted([f for f in os.listdir(output_folder) if f.endswith(".mp4")], key=natural_sort_key)
    list_file_path = os.path.join(output_folder, "clips_list.txt")
    with open(list_file_path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(f"file '{_ffmpeg_concat_escape(clip)}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file_path,
        "-i",
        audio_file,
    ]
    cmd += build_video_codec_args(encoder, fps)
    cmd += [
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-vsync",
        "cfr",
        final_output_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        if encoder != "libx264":
            fallback_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file_path,
                "-i",
                audio_file,
            ]
            fallback_cmd += build_video_codec_args("libx264", fps)
            fallback_cmd += [
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                "-vsync",
                "cfr",
                final_output_path,
            ]
            subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            raise
    if os.path.exists(list_file_path):
        os.remove(list_file_path)


def build_fcp_xml_from_clips(
    clips: list[str],
    fps: int,
    tracks: int,
    sequence_name: str,
    durations_frames: list[int] | None = None,
    intervals_seconds: list[tuple[float, float]] | None = None,
) -> str:
    """
    intervals_seconds: если задан (длина = числу клипов), <start>/<end> в кадрах по абсолютному времени озвучки;
    иначе клипы идут вплотную от 0 (старое поведение).
    """
    track_chunks: list[list[str]] = [[] for _ in range(max(1, tracks))]
    timeline_cursor = 0
    if durations_frames is None:
        durations: list[int] = []
        for clip_path in clips:
            dur = get_audio_duration(clip_path)
            durations.append(max(1, int(round(dur * fps))))
    else:
        durations = [max(1, int(x)) for x in durations_frames]

    use_absolute = (
        intervals_seconds is not None
        and len(intervals_seconds) == len(clips)
        and len(clips) > 0
    )
    placement_frames: list[tuple[int, int]] = []
    if use_absolute:
        boundary_sec = [float(intervals_seconds[0][0])]
        boundary_sec.extend(float(h[1]) for h in intervals_seconds)
        boundary_fr: list[int] = []
        for x in boundary_sec:
            boundary_fr.append(int(round(x * fps)))
        for j in range(1, len(boundary_fr)):
            if boundary_fr[j] <= boundary_fr[j - 1]:
                boundary_fr[j] = boundary_fr[j - 1] + 1
        placement_frames = [
            (boundary_fr[i], boundary_fr[i + 1]) for i in range(len(intervals_seconds))
        ]

    for i, clip_path in enumerate(clips):
        clip_name = os.path.basename(clip_path)
        clip_frames = durations[i]
        if use_absolute:
            start, end = placement_frames[i]
            clip_frames = max(1, end - start)
        else:
            start = timeline_cursor
            end = start + clip_frames
            timeline_cursor = end
        track_index = i % max(1, tracks)
        path_url = path_to_premiere_url(clip_path)
        clip_name_xml = escape_xml(clip_name)
        clip_xml = "\n".join(
            [
                f'        <clipitem id="clipitem-{i+1}">',
                "          <enabled>TRUE</enabled>",
                "          <rate>",
                f"            <timebase>{fps}</timebase>",
                "            <ntsc>FALSE</ntsc>",
                "          </rate>",
                f"          <name>{clip_name_xml}</name>",
                f"          <duration>{clip_frames}</duration>",
                f"          <start>{start}</start>",
                f"          <end>{end}</end>",
                "          <in>0</in>",
                f"          <out>{clip_frames}</out>",
                f'          <file id="file-{i+1}">',
                f"            <name>{clip_name_xml}</name>",
                "            <rate>",
                f"              <timebase>{fps}</timebase>",
                "              <ntsc>FALSE</ntsc>",
                "            </rate>",
                "            <timecode>",
                "              <rate>",
                f"                <timebase>{fps}</timebase>",
                "                <ntsc>FALSE</ntsc>",
                "              </rate>",
                "              <string>00:00:00:00</string>",
                "              <frame>0</frame>",
                "              <displayformat>NDF</displayformat>",
                "            </timecode>",
                f"            <pathurl>{path_url}</pathurl>",
                "            <media>",
                "              <video>",
                "                <samplecharacteristics>",
                "                  <rate>",
                f"                    <timebase>{fps}</timebase>",
                "                    <ntsc>FALSE</ntsc>",
                "                  </rate>",
                "                  <width>1920</width>",
                "                  <height>1080</height>",
                "                  <anamorphic>FALSE</anamorphic>",
                "                  <pixelaspectratio>square</pixelaspectratio>",
                "                  <fielddominance>none</fielddominance>",
                "                </samplecharacteristics>",
                "              </video>",
                "            </media>",
                "          </file>",
                "        </clipitem>",
            ]
        )
        track_chunks[track_index].append(clip_xml)

    track_xml = "\n".join(["      <track>\n" + "\n".join(chunk) + "\n      </track>" for chunk in track_chunks])
    if use_absolute:
        total_duration = max((pf[1] for pf in placement_frames), default=0)
    else:
        total_duration = sum(durations)
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE xmeml>",
            '<xmeml version="4">',
            '  <sequence id="sequence-1">',
            f"    <name>{escape_xml(sequence_name)}</name>",
            "    <rate>",
            f"      <timebase>{fps}</timebase>",
            "      <ntsc>FALSE</ntsc>",
            "    </rate>",
            f"    <duration>{total_duration}</duration>",
            "    <media>",
            "      <video>",
            "        <format>",
            "          <samplecharacteristics>",
            "            <rate>",
            f"              <timebase>{fps}</timebase>",
            "              <ntsc>FALSE</ntsc>",
            "            </rate>",
            "            <width>1920</width>",
            "            <height>1080</height>",
            "            <anamorphic>FALSE</anamorphic>",
            "            <pixelaspectratio>square</pixelaspectratio>",
            "            <fielddominance>none</fielddominance>",
            "          </samplecharacteristics>",
            "        </format>",
            track_xml,
            "      </video>",
            "      <audio>",
            "        <track/>",
            "      </audio>",
            "    </media>",
            "  </sequence>",
            "</xmeml>",
        ]
    )


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    selected_encoder = detect_video_encoder(config.video_encoder)
    print(f"Видео энкодер: {selected_encoder}", flush=True)
    print("[1/6] Подготовка входных данных...", flush=True)
    os.makedirs(config.output_dir, exist_ok=True)
    if not os.path.isfile(config.audio_file):
        raise RuntimeError(f"Файл аудио не найден: {config.audio_file}")
    if not os.path.isfile(config.scenario_file):
        raise RuntimeError(f"Файл сценария не найден: {config.scenario_file}")

    sentences = load_scenario_sentences(config.scenario_file)
    if not sentences:
        raise RuntimeError("Сценарий пустой или в нем нет фраз с пунктуацией.")
    print(f"Найдено фраз в сценарии: {len(sentences)}", flush=True)

    if config.asset_paths:
        assets = collect_assets_from_paths(config.asset_paths)
    else:
        assets = collect_assets(config.assets_root)
    if not assets:
        raise RuntimeError("Не найдено медиафайлов в assets_root.")
    print(f"Найдено медиафайлов: {len(assets)}", flush=True)

    phrase_count = len(sentences)
    asset_count = len(assets)
    balance_diff = asset_count - phrase_count
    if abs(balance_diff) > MEDIA_PHRASE_DIFF_WARN:
        if balance_diff > 0:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: медиафайлов на {balance_diff} больше, чем фраз. "
                "Хвост озвучки может остаться без картинок или последние клипы не используются. Перепроверьте результат.",
                flush=True,
            )
        else:
            print(
                f"ПРЕДУПРЕЖДЕНИЕ: фраз на {abs(balance_diff)} больше, чем медиафайлов. "
                "Часть фраз не получит отдельный кадр (обрежется по числу картинок). Перепроверьте результат.",
                flush=True,
            )

    print(f"[2/6] Whisper: {config.whisper_model} ({config.whisper_language})...", flush=True)
    whisper_words = transcribe_words(config.audio_file, config.whisper_model, config.whisper_language)
    if not whisper_words:
        raise RuntimeError("Whisper не вернул слов с таймкодами.")
    print(f"Распознано слов: {len(whisper_words)}", flush=True)

    print("[3/6] Синхронизация фраз и аудио...", flush=True)
    full_duration = get_audio_duration(config.audio_file)
    if config.align_mode == "anchor":
        raw_bounds, align_diagnostics = map_sentence_bounds_anchor(sentences, whisper_words)
    elif config.align_mode == "block_forced":
        raw_bounds, align_diagnostics = map_sentence_bounds_block_forced(
            sentences, whisper_words, full_duration
        )
    else:
        raw_bounds, align_diagnostics = map_sentence_bounds_standard(sentences, whisper_words)
    # Для block_forced не применяем накопительный gap-clamp,
    # т.к. он может добавлять дрейф по всей последовательности.
    if config.align_mode == "block_forced":
        gap_anomalies = []
        raw_bounds, invariant_fixes = enforce_bounds_invariants_soft(raw_bounds, full_duration)
    else:
        raw_bounds, gap_anomalies = clamp_large_inter_phrase_gaps(
            raw_bounds=raw_bounds,
            max_gap_sec=MAX_INTER_PHRASE_GAP_SEC,
            full_duration=full_duration,
        )
        raw_bounds, invariant_fixes = enforce_bounds_invariants(raw_bounds, full_duration)
    temp_dir = os.path.join(config.output_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "whisper_words.json"), "w", encoding="utf-8") as f:
        json.dump(whisper_words, f, ensure_ascii=False, indent=2)
    with open(os.path.join(temp_dir, "scenario_sentences.json"), "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    with open(os.path.join(temp_dir, "raw_bounds.json"), "w", encoding="utf-8") as f:
        json.dump(raw_bounds, f, ensure_ascii=False, indent=2)
    with open(os.path.join(temp_dir, "align_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(align_diagnostics, f, ensure_ascii=False, indent=2)
    detailed_rows = build_detailed_timing_report(sentences, raw_bounds, align_diagnostics)
    with open(os.path.join(temp_dir, "timing_diagnostics_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(temp_dir, "timing_gap_fixes.json"), "w", encoding="utf-8") as f:
        json.dump(gap_anomalies, f, ensure_ascii=False, indent=2)
    with open(os.path.join(temp_dir, "timing_invariant_fixes.json"), "w", encoding="utf-8") as f:
        json.dump(invariant_fixes, f, ensure_ascii=False, indent=2)

    gaps_for_summary = [r["gap_to_next_sec"] for r in detailed_rows if r.get("gap_to_next_sec") is not None]
    max_gap = max(gaps_for_summary) if gaps_for_summary else None
    summary: dict[str, Any] = {
        "align_mode": config.align_mode,
        "sentence_count": len(sentences),
        "strict_matched": sum(
            1
            for d in align_diagnostics
            if d.get("matched") and d.get("match_kind") in (None, "strict")
        ),
        "rescued": sum(1 for d in align_diagnostics if d.get("match_kind") == "rescue"),
        "filled_leading": sum(1 for d in align_diagnostics if d.get("match_kind") == "filled_leading"),
        "filled_between": sum(1 for d in align_diagnostics if d.get("match_kind") == "filled_between"),
        "filled_trailing": sum(1 for d in align_diagnostics if d.get("match_kind") == "filled_trailing"),
        "filled_uniform_no_anchor": sum(1 for d in align_diagnostics if d.get("match_kind") == "filled_uniform"),
        "max_gap_to_next_sec": max_gap,
    }
    with open(os.path.join(temp_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    matched_count = sum(1 for d in align_diagnostics if d.get("matched"))
    low_conf_count = len(align_diagnostics) - matched_count
    print(
        f"Локальное выравнивание ({config.align_mode}): strict_matched={matched_count}, "
        f"без_жёсткого_матча={low_conf_count}, "
        f"min_coverage={LOCAL_ALIGN_MIN_COVERAGE}",
        flush=True,
    )
    if gap_anomalies:
        print(
            f"Защита gap: исправлено больших межфразовых разрывов: {len(gap_anomalies)} "
            f"(лимит {MAX_INTER_PHRASE_GAP_SEC:.1f} сек).",
            flush=True,
        )
    if invariant_fixes:
        print(
            f"Нормализация таймингов: исправлено интервалов {len(invariant_fixes)}.",
            flush=True,
        )
    if max_gap is not None:
        print(f"Диагностика: max_gap_to_next_sec={max_gap:.3f} (см. _temp/summary.json).", flush=True)
    clip_sentence_count = min(len(assets), len(sentences))
    if (
        config.align_mode == "block_forced"
        and len(sentences) > clip_sentence_count
        and clip_sentence_count < len(raw_bounds)
    ):
        last_s = float(raw_bounds[clip_sentence_count - 1]["start"])
        last_dur = max(0.0, full_duration - last_s)
        print(
            f"ПРЕДУПРЕЖДЕНИЕ: медиа {clip_sentence_count} шт., фраз {len(sentences)}. "
            f"Последний клип: ~{last_s:.1f}—{full_duration:.1f} с (длительность ~{last_dur:.1f} с, одна картинка на хвост фраз).",
            flush=True,
        )
    if config.align_mode == "block_forced":
        intervals = build_absolute_clip_intervals_from_bounds(
            raw_bounds, clip_sentence_count, full_duration
        )
        intervals = extend_sequential_intervals_for_extra_assets(
            intervals,
            len(assets),
            full_duration,
            config.extend_tail,
            config.tail_start_percent,
        )
        interval_sanitize_meta: dict[str, Any] = {
            "zero_or_negative_before": 0,
            "scaled": False,
            "clip_count": len(intervals),
            "sequential_from_bounds": False,
            "absolute_audio_timeline": True,
        }
    else:
        transition_points = build_transition_points(raw_bounds, full_duration)
        intervals = build_intervals_for_assets(
            transition_points=transition_points,
            assets_count=len(assets),
            full_duration=full_duration,
            extend_tail=config.extend_tail,
            tail_start_percent=config.tail_start_percent,
        )
        intervals, interval_sanitize_meta = sanitize_clip_intervals(intervals, full_duration)
    with open(os.path.join(temp_dir, "interval_sanitize.json"), "w", encoding="utf-8") as f:
        json.dump(interval_sanitize_meta, f, ensure_ascii=False, indent=2)
    if interval_sanitize_meta.get("zero_or_negative_before", 0):
        print(
            f"Интервалы клипов: поднято нулевых/битых: {interval_sanitize_meta['zero_or_negative_before']}, "
            f"масштабирование={'да' if interval_sanitize_meta.get('scaled') else 'нет'}.",
            flush=True,
        )
    if config.extend_tail and len(assets) > len(sentences):
        print(
            f"extend_tail: включен, хвост перераспределен после {config.tail_start_percent:.0f}% таймлайна",
            flush=True,
        )

    clips_dir = os.path.join(config.output_dir, "clips")
    xml_dir = os.path.join(config.output_dir, "xml")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)

    tasks: list[tuple[str, float, str]] = []
    selected_assets: list[str] = []
    durations_frames: list[int] = []
    clip_count = min(len(intervals), len(assets))
    for i in range(clip_count):
        duration = intervals[i][1] - intervals[i][0]
        if duration <= 0:
            duration = 0.05
        selected_assets.append(assets[i])
        durations_frames.append(max(1, int(round(duration * config.fps))))
        out_name = os.path.join(clips_dir, f"clip_{i+1:04d}.mp4")
        tasks.append((assets[i], duration, out_name))
    print(f"Готово клипов к рендеру: {len(tasks)}", flush=True)

    rendered_clips: list[str] = []
    render_errors: list[dict[str, str]] = []
    if config.processing_mode == "render":
        print("[4/6] Рендер клипов...", flush=True)
        rendered_by_index: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, config.max_parallel_clips)) as executor:
            future_map = {}
            for idx, task in enumerate(tasks):
                asset, duration, out_name = task
                fut = executor.submit(render_clip, asset, duration, out_name, config.fps, selected_encoder)
                future_map[fut] = (idx, asset, out_name)
            for fut in as_completed(future_map):
                idx, asset, out_name = future_map[fut]
                try:
                    fut.result()
                    rendered_by_index[idx] = out_name
                except Exception as e:
                    render_errors.append({"index": str(idx + 1), "asset": asset, "error": str(e)})
                    print(f"ПРЕДУПРЕЖДЕНИЕ: клип пропущен #{idx+1}: {asset}", flush=True)

        ok_indices = sorted(rendered_by_index.keys())
        rendered_clips = [rendered_by_index[i] for i in ok_indices]
        durations_frames = [durations_frames[i] for i in ok_indices]
        with open(os.path.join(temp_dir, "render_errors.json"), "w", encoding="utf-8") as f:
            json.dump(render_errors, f, ensure_ascii=False, indent=2)
        if render_errors:
            print(f"ПРЕДУПРЕЖДЕНИЕ: пропущено проблемных клипов: {len(render_errors)}", flush=True)
        if not rendered_clips:
            raise RuntimeError("Рендер не дал ни одного валидного клипа. Смотрите _temp/render_errors.json")
    else:
        print("[4/6] FastXML режим: рендер клипов пропущен.", flush=True)
        rendered_clips = selected_assets
        ok_indices = list(range(len(rendered_clips)))

    xml_intervals_sec: list[tuple[float, float]] | None = None
    if config.align_mode == "block_forced":
        xml_intervals_sec = [intervals[i] for i in ok_indices]

    result: dict[str, Any] = {
        "clipsRendered": len(rendered_clips) if config.processing_mode == "render" else 0,
        "clipsUsedInXml": len(rendered_clips),
        "clipsDir": clips_dir,
        "finalVideoPath": None,
        "xmlPath": None,
    }

    if config.render_video and config.processing_mode == "render":
        print("[5/6] Финальная склейка видео + аудио...", flush=True)
        final_video_path = os.path.join(config.output_dir, "FINAL_RESULT_VIDEO.mp4")
        concatenate_final_video(clips_dir, config.audio_file, final_video_path, selected_encoder, config.fps)
        result["finalVideoPath"] = final_video_path
    elif config.render_video and config.processing_mode != "render":
        print("[5/6] FastXML режим: финальная склейка mp4 пропущена.", flush=True)

    if config.render_xml:
        print("[6/6] Генерация XML...", flush=True)
        parts = max(1, int(config.xml_parts))
        if parts == 1:
            xml = build_fcp_xml_from_clips(
                rendered_clips,
                config.fps,
                config.tracks,
                config.sequence_name,
                durations_frames,
                intervals_seconds=xml_intervals_sec,
            )
            xml_path = os.path.join(xml_dir, "premiere_timeline.xml")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml)
            result["xmlPath"] = xml_path
            result["xmlParts"] = [xml_path]
        else:
            chunk_size = (len(rendered_clips) + parts - 1) // parts
            xml_paths: list[str] = []
            for part_idx in range(parts):
                start_idx = part_idx * chunk_size
                end_idx = min(len(rendered_clips), (part_idx + 1) * chunk_size)
                if start_idx >= end_idx:
                    continue
                part_clips = rendered_clips[start_idx:end_idx]
                part_durations = durations_frames[start_idx:end_idx]
                part_intervals = (
                    xml_intervals_sec[start_idx:end_idx] if xml_intervals_sec else None
                )
                xml = build_fcp_xml_from_clips(
                    part_clips,
                    config.fps,
                    config.tracks,
                    f"{config.sequence_name} Part {part_idx + 1}",
                    part_durations,
                    intervals_seconds=part_intervals,
                )
                xml_path = os.path.join(xml_dir, f"premiere_timeline_part{part_idx + 1}.xml")
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(xml)
                xml_paths.append(xml_path)
            result["xmlPath"] = xml_paths[0] if xml_paths else None
            result["xmlParts"] = xml_paths

    print("Пайплайн завершен успешно.", flush=True)
    return result


def run_pipeline_from_json(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    config = PipelineConfig(
        audio_file=raw["audio_file"],
        scenario_file=raw["scenario_file"],
        assets_root=raw["assets_root"],
        asset_paths=raw.get("asset_paths"),
        output_dir=raw["output_dir"],
        whisper_model=raw.get("whisper_model", "large-v3-turbo"),
        whisper_language=raw.get("whisper_language", "en"),
        tracks=int(raw.get("tracks", 4)),
        fps=int(raw.get("fps", 24)),
        max_parallel_clips=int(raw.get("max_parallel_clips", 6)),
        render_video=bool(raw.get("render_video", True)),
        render_xml=bool(raw.get("render_xml", True)),
        sequence_name=raw.get("sequence_name", "Auto Animator Sequence"),
        video_encoder=raw.get("video_encoder", "auto"),
        extend_tail=bool(raw.get("extend_tail", True)),
        tail_start_percent=float(raw.get("tail_start_percent", 70.0)),
        xml_parts=int(raw.get("xml_parts", 3)),
        processing_mode=raw.get("processing_mode", "render"),
        align_mode=raw.get("align_mode", "block_forced"),
    )
    return run_pipeline(config)
