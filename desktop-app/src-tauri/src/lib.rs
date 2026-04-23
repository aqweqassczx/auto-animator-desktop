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

#[derive(Default)]
struct AppState {
    runs: Arc<Mutex<HashMap<String, Arc<Mutex<Child>>>>>,
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
    let value = serde_json::from_str::<serde_json::Value>(line).ok()?;
    value.get("ok")?;
    Some(PipelineFinishedPayload {
        ok: value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
        result: value.get("result").cloned(),
        error: value
            .get("error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        traceback: value
            .get("traceback")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
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
    fs::write(
        &temp_config_path,
        serde_json::to_vec_pretty(&config_json).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let mut cmd = if let Some(runner_path) = resolve_bundled_runner(&app) {
        emit_log(
            &app,
            &run_id,
            format!(
                "Используем встроенный runtime: {}",
                runner_path.to_string_lossy()
            ),
        );
        let mut bundled = Command::new(runner_path);
        bundled
            .arg("--config")
            .arg(temp_config_path.to_string_lossy().to_string());
        bundled
    } else {
        let script_path = PathBuf::from(&request.project_root).join("run_pipeline_cli.py");
        if !script_path.is_file() {
            return Err(format!("Не найден {}", script_path.to_string_lossy()));
        }
        emit_log(
            &app,
            &run_id,
            format!("Встроенный runtime не найден, используем {}", request.python_bin),
        );
        let mut fallback = Command::new(&request.python_bin);
        fallback
            .arg(script_path)
            .arg("--config")
            .arg(temp_config_path.to_string_lossy().to_string());
        fallback
    };
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(&request.project_root);
    #[cfg(target_os = "windows")]
    {
        // Do not flash a separate console window for the bundled runner.
        cmd.creation_flags(0x08000000);
    }

    let mut child = cmd.spawn().map_err(|e| format!("Ошибка запуска: {}", e))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let child_arc = Arc::new(Mutex::new(child));
    state
        .runs
        .lock()
        .map_err(|e| e.to_string())?
        .insert(run_id.clone(), child_arc.clone());

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
    let result_for_wait = result_payload.clone();
    let recent_for_wait = recent_logs.clone();
    std::thread::spawn(move || {
        let exit_status = {
            let mut child_guard = match child_arc.lock() {
                Ok(v) => v,
                Err(_) => {
                    emit_finished(
                        &app_for_wait,
                        &run_wait,
                        PipelineFinishedPayload {
                            ok: false,
                            result: None,
                            error: Some("Не удалось дождаться процесса".to_string()),
                            traceback: None,
                        },
                    );
                    return;
                }
            };
            child_guard.wait()
        };

        if let Ok(mut runs) = runs_for_wait.lock() {
            runs.remove(&run_wait);
        }

        match exit_status {
            Ok(status) => {
                let payload = result_for_wait.lock().ok().and_then(|v| v.clone());
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
                emit_finished(
                    &app_for_wait,
                    &run_wait,
                    effective_payload.unwrap_or_else(|| PipelineFinishedPayload {
                        ok: status.success(),
                        result: None,
                        error: default_error,
                        traceback: None,
                    }),
                );
            }
            Err(err) => {
                emit_finished(
                    &app_for_wait,
                    &run_wait,
                    PipelineFinishedPayload {
                        ok: false,
                        result: None,
                        error: Some(format!("Ошибка ожидания процесса: {}", err)),
                        traceback: None,
                    },
                );
            }
        }
    });

    Ok(PipelineRunHandle { run_id })
}

#[tauri::command]
fn stop_pipeline_run(state: State<AppState>, run_id: String) -> Result<(), String> {
    let guard = state.runs.lock().map_err(|e| e.to_string())?;
    let child = guard
        .get(&run_id)
        .ok_or_else(|| "Запуск не найден".to_string())?
        .clone();
    drop(guard);
    let mut c = child.lock().map_err(|e| e.to_string())?;
    c.kill().map_err(|e| format!("Не удалось остановить: {}", e))
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            discover_paths,
            discover_media_folders,
            start_pipeline_run,
            stop_pipeline_run
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
