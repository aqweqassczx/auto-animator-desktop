use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};
use uuid::Uuid;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[derive(Clone)]
struct RunEntry {
    pid: u32,
    child: Arc<Mutex<Option<Child>>>,
}

#[derive(Default)]
struct AppState {
    runs: Arc<Mutex<HashMap<String, RunEntry>>>,
    finished: Arc<Mutex<HashMap<String, PipelineFinishedPayload>>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineRunRequest {
    python_bin: String,
    project_root: String,
    audio_file: String,
    scenario_file: String,
    output_base: String,
    media_folders: Vec<String>,
    processing_mode: String,
    render_video: bool,
    xml_parts: i32,
    max_parallel_clips: i32,
    whisper_model: String,
    whisper_model_cache_path: Option<String>,
    whisper_language: String,
    align_mode: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelineRunHandle {
    run_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiscoverPathsResult {
    media_folders: Vec<String>,
    audio_file: String,
    scenario_file: String,
    output_base: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiscoverMediaFoldersResult {
    media_folders: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct WhisperModelOption {
    id: String,
    label: String,
    installed: bool,
    cache_path: Option<String>,
}

const KNOWN_WHISPER_MODELS: &[(&str, &str)] = &[
    ("tiny", "Whisper Tiny"),
    ("base", "Whisper Base"),
    ("small", "Whisper Small"),
    ("medium", "Whisper Medium"),
    ("large-v2", "Whisper Large v2"),
    ("large-v3", "Whisper Large v3"),
    ("large-v3-turbo", "Whisper Large v3 Turbo"),
    ("distil-large-v2", "Distil Large v2"),
    ("distil-large-v3", "Distil Large v3"),
];

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct PipelineFinishedPayload {
    ok: bool,
    result: Option<serde_json::Value>,
    error: Option<String>,
    traceback: Option<String>,
}

fn find_first_file_with_ext_recursive(dir: &Path, exts: &[&str], max_depth: usize) -> String {
    let mut dirs: Vec<(PathBuf, usize)> = vec![(dir.to_path_buf(), 0)];
    let mut best = String::new();
    while let Some((cur, depth)) = dirs.pop() {
        if let Ok(entries) = fs::read_dir(&cur) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && depth < max_depth {
                    dirs.push((path, depth + 1));
                    continue;
                }
                if !path.is_file() {
                    continue;
                }
                let ext = match path.extension().and_then(|e| e.to_str()) {
                    Some(v) => v.to_ascii_lowercase(),
                    None => continue,
                };
                if exts.iter().any(|x| ext == *x) {
                    let p = path.to_string_lossy().to_string();
                    if best.is_empty() {
                        best = p.clone();
                    }
                    if path
                        .file_name()
                        .and_then(|f| f.to_str())
                        .map(|f| {
                            let low = f.to_ascii_lowercase();
                            low.contains("сценар") || low.contains("scenario")
                        })
                        .unwrap_or(false)
                    {
                        return p;
                    }
                }
            }
        }
    }
    best
}

fn has_media_files(dir: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let ext = match p.extension().and_then(|e| e.to_str()) {
                Some(v) => v.to_ascii_lowercase(),
                None => continue,
            };
            if ["jpg", "jpeg", "png", "webp", "mp4", "mov", "mkv"]
                .iter()
                .any(|x| ext == *x)
            {
                return true;
            }
        }
    }
    false
}

fn split_key(value: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut is_digit = None;
    for ch in value.chars() {
        let d = ch.is_ascii_digit();
        if is_digit == Some(d) || is_digit.is_none() {
            buf.push(ch);
            is_digit = Some(d);
        } else {
            out.push(buf);
            buf = String::new();
            buf.push(ch);
            is_digit = Some(d);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn natural_compare(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let ka = split_key(a);
    let kb = split_key(b);
    let n = ka.len().min(kb.len());
    for i in 0..n {
        let sa = &ka[i];
        let sb = &kb[i];
        let da = sa.chars().all(|c| c.is_ascii_digit());
        let db = sb.chars().all(|c| c.is_ascii_digit());
        let ord = if da && db {
            let ia = sa.parse::<u64>().unwrap_or(0);
            let ib = sb.parse::<u64>().unwrap_or(0);
            ia.cmp(&ib)
        } else {
            sa.to_ascii_lowercase().cmp(&sb.to_ascii_lowercase())
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    ka.len().cmp(&kb.len())
}

fn collect_media_folders_sorted(media_root: &Path) -> Vec<String> {
    let mut rows: Vec<(String, String)> = Vec::new();
    let excluded = ["result", "_temp", "xml", "clips", "output", "outputs"];
    if let Ok(entries) = fs::read_dir(media_root) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            let folder_name_lower = p
                .file_name()
                .and_then(|x| x.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if excluded.iter().any(|x| *x == folder_name_lower) {
                continue;
            }
            if has_media_files(&p) {
                let name = p
                    .file_name()
                    .and_then(|x| x.to_str())
                    .unwrap_or_default()
                    .to_string();
                rows.push((name, p.to_string_lossy().to_string()));
            }
        }
    }
    rows.sort_by(|a, b| natural_compare(&a.0, &b.0));
    rows.into_iter().map(|(_, p)| p).collect()
}

#[tauri::command]
fn discover_paths(project_root: String) -> Result<DiscoverPathsResult, String> {
    let root = PathBuf::from(project_root);
    if !root.is_dir() {
        return Err("Корневая папка не существует".to_string());
    }

    let media_folders = collect_media_folders_sorted(&root);

    let audio_file = find_first_file_with_ext_recursive(&root, &["mp3", "wav", "m4a"], 3);
    let scenario_file = find_first_file_with_ext_recursive(&root, &["txt"], 3);
    let output_base = root.join("result").to_string_lossy().to_string();

    Ok(DiscoverPathsResult {
        media_folders,
        audio_file,
        scenario_file,
        output_base,
    })
}

fn hf_hub_cache_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        let hub = PathBuf::from(hf_home).join("hub");
        if hub.is_dir() {
            dirs.push(hub);
        }
    }
    if let Ok(userprofile) = std::env::var("USERPROFILE") {
        let hub = PathBuf::from(&userprofile)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if hub.is_dir() {
            dirs.push(hub);
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let hub = PathBuf::from(local).join("huggingface").join("hub");
        if hub.is_dir() {
            dirs.push(hub);
        }
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

fn cache_dir_to_model_id(name: &str) -> Option<String> {
    const MARKER: &str = "--faster-whisper-";
    let idx = name.find(MARKER)?;
    Some(name[idx + MARKER.len()..].to_string())
}

fn snapshot_has_model_bin(snapshot_dir: &Path) -> bool {
    snapshot_dir.join("model.bin").is_file()
}

fn find_installed_whisper_models() -> HashMap<String, String> {
    let mut installed: HashMap<String, String> = HashMap::new();
    for hub_dir in hf_hub_cache_dirs() {
        let Ok(entries) = fs::read_dir(&hub_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(model_id) = cache_dir_to_model_id(dir_name) else {
                continue;
            };
            let snapshots_dir = path.join("snapshots");
            if !snapshots_dir.is_dir() {
                continue;
            }
            let Ok(snaps) = fs::read_dir(&snapshots_dir) else {
                continue;
            };
            for snap in snaps.flatten() {
                let snap_path = snap.path();
                if snap_path.is_dir() && snapshot_has_model_bin(&snap_path) {
                    installed
                        .entry(model_id.clone())
                        .or_insert_with(|| snap_path.to_string_lossy().to_string());
                    break;
                }
            }
        }
    }
    installed
}

fn python_supports_cuda_whisper(python_bin: &str) -> bool {
    let script = "import ctranslate2; raise SystemExit(0 if ctranslate2.get_cuda_device_count() > 0 else 1)";
    Command::new(python_bin)
        .args(["-c", script])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn python_supports_mlx_whisper(python_bin: &str) -> bool {
    let script = "import platform; \
import sys; \
ok = platform.system()=='Darwin' and platform.machine().lower() in {'arm64','aarch64'}; \
exec('import mlx_whisper') if ok else None; \
raise SystemExit(0 if ok else 1)";
    Command::new(python_bin)
        .args(["-c", script])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn host_is_apple_silicon() -> bool {
    true
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn host_is_apple_silicon() -> bool {
    false
}

#[tauri::command]
fn list_whisper_models() -> Result<Vec<WhisperModelOption>, String> {
    let installed = find_installed_whisper_models();
    let mut options: Vec<WhisperModelOption> = KNOWN_WHISPER_MODELS
        .iter()
        .map(|(id, label)| {
            let hit = installed.get(*id);
            WhisperModelOption {
                id: (*id).to_string(),
                label: (*label).to_string(),
                installed: hit.is_some(),
                cache_path: hit.cloned(),
            }
        })
        .collect();

    for (id, cache_path) in installed {
        if options.iter().any(|o| o.id == id) {
            continue;
        }
        options.push(WhisperModelOption {
            label: format!("Whisper {}", id),
            id,
            installed: true,
            cache_path: Some(cache_path),
        });
    }

    options.sort_by(|a, b| {
        b.installed
            .cmp(&a.installed)
            .then_with(|| a.label.cmp(&b.label))
    });
    Ok(options)
}

#[tauri::command]
fn discover_media_folders(media_root: String) -> Result<DiscoverMediaFoldersResult, String> {
    let root = PathBuf::from(media_root);
    if !root.is_dir() {
        return Err("Папка с изображениями не существует".to_string());
    }
    Ok(DiscoverMediaFoldersResult {
        media_folders: collect_media_folders_sorted(&root),
    })
}

fn emit_log(app: &AppHandle, run_id: &str, line: String) {
    let _ = app.emit_all(
        "pipeline-log",
        serde_json::json!({
            "runId": run_id,
            "line": line
        }),
    );
}

fn emit_finished(app: &AppHandle, run_id: &str, payload: PipelineFinishedPayload) {
    let _ = app.emit_all(
        "pipeline-finished",
        serde_json::json!({
            "runId": run_id,
            "payload": payload
        }),
    );
}

fn push_recent_line(store: &Arc<Mutex<Vec<String>>>, line: String) {
    if let Ok(mut lines) = store.lock() {
        lines.push(line);
        if lines.len() > 200 {
            let overflow = lines.len() - 200;
            lines.drain(0..overflow);
        }
    }
}

fn parse_pipeline_json_line(line: &str) -> Option<PipelineFinishedPayload> {
    let direct = serde_json::from_str::<serde_json::Value>(line).ok();
    let value = if let Some(v) = direct {
        v
    } else {
        let start = line.find('{')?;
        let end = line.rfind('}')?;
        if end <= start {
            return None;
        }
        let slice = &line[start..=end];
        serde_json::from_str::<serde_json::Value>(slice).ok()?
    };
    value.get("ok")?;
    let base_error = value
        .get("error")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let error_code = value
        .get("error_code")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let hint = value
        .get("hint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let traceback = value
        .get("traceback")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let composed_error = if let Some(code) = error_code {
        let mut chunks: Vec<String> = vec![format!("[{}] {}", code, base_error.unwrap_or_default())];
        if let Some(h) = hint {
            if !h.trim().is_empty() {
                chunks.push(format!("Что делать: {}", h));
            }
        }
        if let Some(tb) = traceback.clone() {
            let short_tb = tb.lines().take(12).collect::<Vec<&str>>().join("\n");
            if !short_tb.trim().is_empty() {
                chunks.push(format!("Трассировка:\n{}", short_tb));
            }
        }
        Some(chunks.join("\n"))
    } else {
        match (base_error, traceback.clone()) {
            (Some(err), Some(tb)) => {
                let short_tb = tb.lines().take(12).collect::<Vec<&str>>().join("\n");
                Some(format!("{}\nТрассировка:\n{}", err, short_tb))
            }
            (Some(err), None) => Some(err),
            (None, Some(tb)) => Some(tb.lines().take(12).collect::<Vec<&str>>().join("\n")),
            (None, None) => None,
        }
    };
    Some(PipelineFinishedPayload {
        ok: value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
        result: value.get("result").cloned(),
        error: composed_error,
        traceback,
    })
}

fn build_human_error(exit_code_label: &str, fallback_tail: &str) -> String {
    let tail_lower = fallback_tail.to_ascii_lowercase();
    if tail_lower.contains("unable to open file 'model.bin'") {
        return format!(
            "Whisper не смог загрузить модель (model.bin в кэше поврежден или недокачан). \
Попробуй запустить еще раз: приложение очистит кэш и скачает модель заново.\n\nКод: {}\nЛог:\n{}",
            exit_code_label, fallback_tail
        );
    }
    if tail_lower.contains("window-close event") || tail_lower.contains("program aborting") {
        return format!(
            "Процесс рендера был прерван внешним событием (окно/процесс закрыт во время работы).\n\nКод: {}\nЛог:\n{}",
            exit_code_label, fallback_tail
        );
    }
    if fallback_tail.trim().is_empty() {
        return format!("Пайплайн завершился с ошибкой. Код: {}", exit_code_label);
    }
    format!(
        "Пайплайн завершился с ошибкой. Код: {}\nПоследние строки лога:\n{}",
        exit_code_label, fallback_tail
    )
}

fn resolve_bundled_runner(app: &AppHandle) -> Option<PathBuf> {
    let resource_dir = app.path_resolver().resource_dir()?;
    let file_name = if cfg!(target_os = "windows") {
        "pipeline_runner.exe"
    } else {
        "pipeline_runner"
    };
    let candidate = resource_dir.join("runtime").join(file_name);
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

fn runner_supports_result_file(runner_path: &Path) -> bool {
    let output = Command::new(runner_path).arg("--help").output();
    match output {
        Ok(o) => {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            combined.contains("--result-file")
        }
        Err(_) => false,
    }
}

fn find_pipeline_cli_script(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start.to_path_buf());
    for _ in 0..8 {
        if let Some(ref dir) = current {
            let candidate = dir.join("run_pipeline_cli.py");
            if candidate.is_file() {
                return Some(candidate);
            }
            current = dir.parent().map(Path::to_path_buf);
        } else {
            break;
        }
    }
    None
}

fn resolve_bundled_pipeline_script(app: &AppHandle) -> Option<PathBuf> {
    let resource_dir = app.path_resolver().resource_dir()?;
    let candidate = resource_dir.join("runtime").join("run_pipeline_cli.py");
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

fn resolve_pipeline_cli_script(app: &AppHandle, project_root: &Path) -> Option<PathBuf> {
    find_pipeline_cli_script(project_root).or_else(|| resolve_bundled_pipeline_script(app))
}

fn resolve_bundled_mac_python(app: &AppHandle) -> Option<PathBuf> {
    if !host_is_apple_silicon() {
        return None;
    }
    let resource_dir = app.path_resolver().resource_dir()?;
    let candidate = resource_dir
        .join("runtime")
        .join("mac-venv")
        .join("bin")
        .join("python3");
    if candidate.is_file() {
        Some(candidate)
    } else {
        None
    }
}

fn resolve_pipeline_python(app: &AppHandle, request_python: &str) -> String {
    if let Some(mac_python) = resolve_bundled_mac_python(app) {
        return mac_python.to_string_lossy().to_string();
    }
    request_python.to_string()
}

#[tauri::command]
fn start_pipeline_run(
    app: AppHandle,
    state: State<AppState>,
    request: PipelineRunRequest,
) -> Result<PipelineRunHandle, String> {
    let run_id = Uuid::new_v4().to_string();
    let config_json = serde_json::json!({
        "audio_file": request.audio_file,
        "scenario_file": request.scenario_file,
        "assets_root": "",
        "asset_paths": request.media_folders,
        "output_dir": PathBuf::from(&request.output_base).join("final_synced_clips").to_string_lossy().to_string(),
        "whisper_model": request.whisper_model,
        "whisper_language": request.whisper_language,
        "whisper_model_cache_path": request.whisper_model_cache_path,
        "tracks": 4,
        "fps": 24,
        "max_parallel_clips": request.max_parallel_clips,
        "render_video": request.render_video,
        "render_xml": true,
        "sequence_name": "Auto Animator Sequence",
        "video_encoder": "auto",
        "extend_tail": true,
        "tail_start_percent": 70.0,
        "xml_parts": request.xml_parts,
        "processing_mode": request.processing_mode,
        "align_mode": request.align_mode
    });

    let temp_config_path = std::env::temp_dir().join(format!("auto_animator_{}.json", run_id));
    let temp_result_path = std::env::temp_dir().join(format!("auto_animator_result_{}.json", run_id));
    fs::write(
        &temp_config_path,
        serde_json::to_vec_pretty(&config_json).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let bundled_runner = resolve_bundled_runner(&app).filter(|path| runner_supports_result_file(path));
    let script_path = resolve_pipeline_cli_script(&app, Path::new(&request.project_root));
    let pipeline_python = resolve_pipeline_python(&app, &request.python_bin);
    let script_project_root = script_path.as_ref().and_then(|p| p.parent().map(Path::to_path_buf));
    let python_has_cuda = python_supports_cuda_whisper(&pipeline_python);
    let python_has_mlx = python_supports_mlx_whisper(&pipeline_python);
    let has_bundled_mac_python = resolve_bundled_mac_python(&app).is_some();
    // Python: CUDA (Win/NVIDIA), MLX (Mac M), или CPU. pipeline_runner — только fallback без скрипта.
    let prefer_python = script_path.is_some()
        || (host_is_apple_silicon() && (has_bundled_mac_python || python_has_mlx));

    let mut cmd = if prefer_python {
        let script_path = script_path.or_else(|| resolve_bundled_pipeline_script(&app))
            .ok_or_else(|| {
                "Не найден run_pipeline_cli.py (ни в проекте, ни во встроенном runtime).".to_string()
            })?;
        if python_has_cuda {
            emit_log(
                &app,
                &run_id,
                format!(
                    "Python + CUDA (NVIDIA GPU): {} — pipeline_runner пропущен (только CPU)",
                    pipeline_python
                ),
            );
        } else if host_is_apple_silicon() {
            if has_bundled_mac_python {
                emit_log(
                    &app,
                    &run_id,
                    format!(
                        "Mac Apple Silicon + MLX (встроенный Python): {}",
                        pipeline_python
                    ),
                );
            } else if python_has_mlx {
                emit_log(
                    &app,
                    &run_id,
                    format!(
                        "Mac Apple Silicon + MLX: {} — pipeline_runner не используется",
                        pipeline_python
                    ),
                );
            } else {
                emit_log(
                    &app,
                    &run_id,
                    format!(
                        "Mac Apple Silicon: {} — при старте задачи установится mlx-whisper (GPU)",
                        pipeline_python
                    ),
                );
            }
        } else {
            emit_log(
                &app,
                &run_id,
                format!(
                    "Python (CPU): {} — CUDA не найдена, pipeline_runner пропущен",
                    pipeline_python
                ),
            );
        }
        if let Some(runner_path) = bundled_runner.as_ref() {
            emit_log(
                &app,
                &run_id,
                format!(
                    "Подсказка: {} = только CPU; для GPU используем Python.",
                    runner_path.to_string_lossy()
                ),
            );
        }
        emit_log(
            &app,
            &run_id,
            format!("Python-скрипт: {}", script_path.to_string_lossy()),
        );
        let mut python_cmd = Command::new(&pipeline_python);
        python_cmd
            .arg(&script_path)
            .arg("--config")
            .arg(temp_config_path.to_string_lossy().to_string())
            .arg("--result-file")
            .arg(temp_result_path.to_string_lossy().to_string());
        python_cmd
    } else if host_is_apple_silicon() {
        return Err(
            "На Mac Apple Silicon нужен встроенный MLX-runtime. \
Пересобери приложение или запусти desktop-app/tools/setup_mac_runtime.sh"
                .to_string(),
        );
    } else if let Some(runner_path) = bundled_runner {
        emit_log(
            &app,
            &run_id,
            format!(
                "Встроенный runtime (CPU, без GPU): {}",
                runner_path.to_string_lossy()
            ),
        );
        if script_path.is_some() && !python_has_cuda {
            emit_log(
                &app,
                &run_id,
                "CUDA для faster-whisper не найдена в Python — используем встроенный CPU runtime."
                    .to_string(),
            );
        }
        let mut bundled = Command::new(runner_path);
        bundled
            .arg("--config")
            .arg(temp_config_path.to_string_lossy().to_string())
            .arg("--result-file")
            .arg(temp_result_path.to_string_lossy().to_string());
        bundled
    } else if let Some(script_path) = script_path {
        if let Some(stale_runner) = resolve_bundled_runner(&app) {
            emit_log(
                &app,
                &run_id,
                format!(
                    "Встроенный runtime устарел (нет --result-file): {}. Пересобери tools/build_pipeline_runner.ps1",
                    stale_runner.to_string_lossy()
                ),
            );
        } else {
            emit_log(
                &app,
                &run_id,
                format!("Встроенный runtime не найден, используем {}", pipeline_python),
            );
        }
        emit_log(
            &app,
            &run_id,
            format!("Python-скрипт: {}", script_path.to_string_lossy()),
        );
        let mut fallback = Command::new(&pipeline_python);
        fallback
            .arg(&script_path)
            .arg("--config")
            .arg(temp_config_path.to_string_lossy().to_string())
            .arg("--result-file")
            .arg(temp_result_path.to_string_lossy().to_string());
        fallback
    } else {
        return Err(
            "Не найден run_pipeline_cli.py (искали в папке проекта и выше). \
Укажи корень репозитория или пересобери pipeline_runner.exe."
                .to_string(),
        );
    };
    let working_dir = script_project_root.unwrap_or_else(|| PathBuf::from(&request.project_root));
    cmd.env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(&working_dir);
    #[cfg(target_os = "windows")]
    {
        // Do not flash a separate console window for the bundled runner.
        cmd.creation_flags(0x08000000);
    }

    let mut child = cmd.spawn().map_err(|e| format!("Ошибка запуска: {}", e))?;
    let pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let child_arc = Arc::new(Mutex::new(Some(child)));
    state
        .runs
        .lock()
        .map_err(|e| e.to_string())?
        .insert(
            run_id.clone(),
            RunEntry {
                pid,
                child: child_arc.clone(),
            },
        );

    let result_payload: Arc<Mutex<Option<PipelineFinishedPayload>>> = Arc::new(Mutex::new(None));
    let recent_logs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let app_for_out = app.clone();
    let run_out = run_id.clone();
    let result_for_out = result_payload.clone();
    let recent_for_out = recent_logs.clone();
    if let Some(stdout) = stdout {
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                push_recent_line(&recent_for_out, line.clone());
                if let Some(parsed) = parse_pipeline_json_line(&line) {
                    if let Ok(mut lock) = result_for_out.lock() {
                        *lock = Some(parsed);
                    }
                }
                emit_log(&app_for_out, &run_out, line);
            }
        });
    }

    let app_for_err = app.clone();
    let run_err = run_id.clone();
    let result_for_err = result_payload.clone();
    let recent_for_err = recent_logs.clone();
    if let Some(stderr) = stderr {
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                push_recent_line(&recent_for_err, line.clone());
                if let Some(parsed) = parse_pipeline_json_line(&line) {
                    if let Ok(mut lock) = result_for_err.lock() {
                        *lock = Some(parsed);
                    }
                }
                emit_log(&app_for_err, &run_err, line);
            }
        });
    }

    let app_for_wait = app.clone();
    let run_wait = run_id.clone();
    let runs_for_wait = state.runs.clone();
    let finished_for_wait = state.finished.clone();
    let result_for_wait = result_payload.clone();
    let recent_for_wait = recent_logs.clone();
    let config_path_for_wait = temp_config_path.clone();
    let result_path_for_wait = temp_result_path.clone();
    std::thread::spawn(move || {
        let exit_status = {
            let child = match child_arc.lock() {
                Ok(mut guard) => guard.take(),
                Err(_) => {
                    let payload = PipelineFinishedPayload {
                        ok: false,
                        result: None,
                        error: Some("Не удалось дождаться процесса".to_string()),
                        traceback: None,
                    };
                    if let Ok(mut done) = finished_for_wait.lock() {
                        done.insert(run_wait.clone(), payload.clone());
                    }
                    emit_finished(
                        &app_for_wait,
                        &run_wait,
                        payload,
                    );
                    return;
                }
            };
            match child {
                Some(mut c) => c.wait().ok(),
                None => None,
            }
        };

        if let Ok(mut runs) = runs_for_wait.lock() {
            runs.remove(&run_wait);
        }

        match exit_status {
            Some(status) => {
                let file_payload = fs::read_to_string(&result_path_for_wait)
                    .ok()
                    .and_then(|content| parse_pipeline_json_line(content.trim()));
                let payload = file_payload
                    .or_else(|| result_for_wait.lock().ok().and_then(|v| v.clone()));
                let fallback_tail = recent_for_wait
                    .lock()
                    .ok()
                    .map(|lines| {
                        lines
                            .iter()
                            .rev()
                            .take(20)
                            .cloned()
                            .collect::<Vec<String>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<String>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let exit_code_label = status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let default_error = if status.success() {
                    None
                } else {
                    Some(build_human_error(&exit_code_label, &fallback_tail))
                };
                let effective_payload = if let Some(mut p) = payload {
                    if !p.ok && p.error.as_deref().unwrap_or("").trim().is_empty() {
                        p.error = default_error.clone();
                    }
                    Some(p)
                } else {
                    None
                };
                let final_payload = effective_payload.unwrap_or_else(|| PipelineFinishedPayload {
                    ok: status.success(),
                    result: None,
                    error: default_error,
                    traceback: None,
                });
                emit_finished(&app_for_wait, &run_wait, final_payload.clone());
                if let Ok(mut done) = finished_for_wait.lock() {
                    done.insert(run_wait.clone(), final_payload);
                }
                let _ = fs::remove_file(&result_path_for_wait);
                let _ = fs::remove_file(&config_path_for_wait);
            }
            None => {
                let payload = PipelineFinishedPayload {
                    ok: false,
                    result: None,
                    error: Some("Процесс остановлен пользователем".to_string()),
                    traceback: None,
                };
                if let Ok(mut done) = finished_for_wait.lock() {
                    done.insert(run_wait.clone(), payload.clone());
                }
                emit_finished(
                    &app_for_wait,
                    &run_wait,
                    payload,
                );
                let _ = fs::remove_file(&result_path_for_wait);
                let _ = fs::remove_file(&config_path_for_wait);
            }
        }
    });

    Ok(PipelineRunHandle { run_id })
}

#[tauri::command]
fn stop_pipeline_run(state: State<AppState>, run_id: String) -> Result<(), String> {
    let entry = {
        let guard = state.runs.lock().map_err(|e| e.to_string())?;
        guard
            .get(&run_id)
            .cloned()
            .ok_or_else(|| "Запуск не найден".to_string())?
    };
    let pid = entry.pid;
    std::thread::spawn(move || {
        #[cfg(target_os = "windows")]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .status();
        }
        if let Ok(mut guard) = entry.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn get_pipeline_result(
    state: State<AppState>,
    run_id: String,
) -> Result<Option<PipelineFinishedPayload>, String> {
    let mut guard = state.finished.lock().map_err(|e| e.to_string())?;
    Ok(guard.remove(&run_id))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PipelineRuntimeInfo {
    platform: String,
    python_bin: String,
    script_found: bool,
    cuda_available: bool,
    mlx_available: bool,
    apple_silicon: bool,
    backend: String,
    hint: String,
}

fn detect_runtime_backend(
    apple_silicon: bool,
    script_found: bool,
    cuda_available: bool,
    mlx_available: bool,
    has_bundled_mac_python: bool,
) -> (String, String) {
    if apple_silicon {
        if mlx_available || has_bundled_mac_python {
            return (
                "mlx".to_string(),
                "Mac Apple Silicon: Whisper через MLX (GPU).".to_string(),
            );
        }
        return (
            "mlx_pending".to_string(),
            "Mac Apple Silicon: MLX установится автоматически при первом запуске.".to_string(),
        );
    }
    if script_found && cuda_available {
        return (
            "cuda".to_string(),
            "Windows/Linux: Whisper через NVIDIA CUDA (GPU).".to_string(),
        );
    }
    if script_found {
        return (
            "cpu".to_string(),
            "Python без GPU: Whisper на CPU (медленнее).".to_string(),
        );
    }
    (
        "bundled".to_string(),
        "Встроенный pipeline_runner (только CPU, без GPU).".to_string(),
    )
}

#[tauri::command]
fn get_pipeline_runtime_info(
    app: AppHandle,
    project_root: String,
    python_bin: String,
) -> Result<PipelineRuntimeInfo, String> {
    let platform = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
    .to_string();
    let apple_silicon = host_is_apple_silicon();
    let pipeline_python = resolve_pipeline_python(&app, &python_bin);
    let script_found = resolve_pipeline_cli_script(&app, Path::new(&project_root)).is_some();
    let cuda_available = python_supports_cuda_whisper(&pipeline_python);
    let mlx_available = python_supports_mlx_whisper(&pipeline_python);
    let has_bundled_mac_python = resolve_bundled_mac_python(&app).is_some();
    let (backend, hint) = detect_runtime_backend(
        apple_silicon,
        script_found,
        cuda_available,
        mlx_available,
        has_bundled_mac_python,
    );
    Ok(PipelineRuntimeInfo {
        platform,
        python_bin: pipeline_python,
        script_found,
        cuda_available,
        mlx_available,
        apple_silicon,
        backend,
        hint,
    })
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            discover_paths,
            discover_media_folders,
            list_whisper_models,
            get_pipeline_runtime_info,
            start_pipeline_run,
            stop_pipeline_run,
            get_pipeline_result
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
