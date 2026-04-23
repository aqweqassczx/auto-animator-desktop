import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { open as openDialog } from "@tauri-apps/api/dialog";
import { open as openPath } from "@tauri-apps/api/shell";
import { getVersion } from "@tauri-apps/api/app";
import { checkUpdate, installUpdate } from "@tauri-apps/api/updater";
import {
  discoverMediaFolders,
  discoverPaths,
  getPipelineResult,
  listenPipelineFinished,
  listenPipelineLogs,
  startPipelineRun,
  stopPipelineRun
} from "./lib/api";
import type { PipelineRunRequest, ProcessingMode, UpdateStatus, WhisperLanguage } from "./types";

const STORAGE_CONFIG_KEY = "auto-animator-desktop-last-config";
const STORAGE_JOBS_KEY = "auto-animator-desktop-jobs";

const DEFAULT_PYTHON = "python";
const DEFAULT_WHISPER_MODEL = "medium";
const DEFAULT_MAX_PARALLEL_CLIPS = 6;

type ConfigState = {
  projectRoot: string;
  mediaRoot: string;
  mediaFolders: string[];
  audioFile: string;
  scenarioFile: string;
  outputBase: string;
  whisperLanguage: WhisperLanguage;
  xmlParts: number;
  processingMode: ProcessingMode;
  renderVideo: boolean;
};

type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

type LibraryJob = {
  id: string;
  createdAt: number;
  status: JobStatus;
  progress: number;
  stageLabel: string;
  runId?: string;
  logs: string[];
  request: PipelineRunRequest;
  result?: {
    finalVideoPath: string | null;
    xmlPath: string | null;
    xmlParts?: string[];
    clipsRendered: number;
    clipsUsedInXml: number;
  };
  error?: string;
};

function normalizeErrorText(value?: string): string | undefined {
  if (!value) return value;
  return value.replace(/Some\((\-?\d+)\)/g, "$1");
}

const defaultState: ConfigState = {
  projectRoot: "",
  mediaRoot: "",
  mediaFolders: [],
  audioFile: "",
  scenarioFile: "",
  outputBase: "",
  whisperLanguage: "en",
  xmlParts: 3,
  processingMode: "render",
  renderVideo: false
};

function loadSavedState(): ConfigState {
  const raw = localStorage.getItem(STORAGE_CONFIG_KEY);
  if (!raw) return defaultState;
  try {
    return { ...defaultState, ...(JSON.parse(raw) as Partial<ConfigState>) };
  } catch {
    return defaultState;
  }
}

function loadSavedJobs(): LibraryJob[] {
  const raw = localStorage.getItem(STORAGE_JOBS_KEY);
  if (!raw) return [];
  try {
    const jobs = JSON.parse(raw) as LibraryJob[];
    return jobs.map((j) => ({
      ...j,
      error:
        j.status === "running"
          ? "Приложение было перезапущено"
          : normalizeErrorText(j.error),
      logs: j.logs.map((line) => normalizeErrorText(line) ?? line),
      status: j.status === "running" ? "failed" : j.status
    }));
  } catch {
    return [];
  }
}

function parseStageProgress(line: string): { progress: number; stageLabel: string } | null {
  const m = line.match(/\[(\d+)\/6\]\s*(.+)?/);
  if (!m) return null;
  const stage = Number(m[1]);
  if (!Number.isFinite(stage) || stage < 1) return null;
  const progress = Math.min(99, Math.max(1, Math.round((stage / 6) * 100)));
  const stageLabel = (m[2] || "").trim() || `Этап ${stage}/6`;
  return { progress, stageLabel };
}

function playSoftClick() {
  try {
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "triangle";
    osc.frequency.value = 220;
    gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.08, audioCtx.currentTime + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.05);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.055);
  } catch {
    // ignore audio errors
  }
}

function App() {
  const [config, setConfig] = useState<ConfigState>(() => loadSavedState());
  const [jobs, setJobs] = useState<LibraryJob[]>(() => loadSavedJobs());
  const [runningRunId, setRunningRunId] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("Готово");
  const [appVersion, setAppVersion] = useState<string>("");
  const [isInstallingUpdate, setIsInstallingUpdate] = useState(false);
  const [updateDots, setUpdateDots] = useState(0);
  const [updateInstallError, setUpdateInstallError] = useState<string>("");
  const [updateStatus, setUpdateStatus] = useState<UpdateStatus>({
    checked: false,
    available: false
  });

  const canRun = useMemo(() => {
    return (
      !!config.audioFile &&
      !!config.scenarioFile &&
      !!config.outputBase &&
      config.mediaFolders.length > 0 &&
      !runningRunId
    );
  }, [config, runningRunId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_CONFIG_KEY, JSON.stringify(config));
  }, [config]);

  useEffect(() => {
    localStorage.setItem(STORAGE_JOBS_KEY, JSON.stringify(jobs));
  }, [jobs]);

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("button")) {
        playSoftClick();
      }
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  useEffect(() => {
    void getVersion()
      .then((v) => setAppVersion(v))
      .catch(() => setAppVersion(""));
  }, []);

  useEffect(() => {
    if (!isInstallingUpdate) {
      setUpdateDots(0);
      return;
    }
    const timer = window.setInterval(() => {
      setUpdateDots((prev) => (prev + 1) % 4);
    }, 400);
    return () => window.clearInterval(timer);
  }, [isInstallingUpdate]);

  useEffect(() => {
    void (async () => {
      try {
        const { shouldUpdate, manifest } = await checkUpdate();
        setUpdateStatus({
          checked: true,
          available: shouldUpdate,
          version: manifest?.version,
          body: manifest?.body
        });
      } catch (error) {
        setUpdateStatus({
          checked: true,
          available: false,
          error: String(error)
        });
      }
    })();
  }, []);

  const setField = <K extends keyof ConfigState>(key: K, value: ConfigState[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const updateJob = (jobId: string, updater: (job: LibraryJob) => LibraryJob) => {
    setJobs((prev) => prev.map((j) => (j.id === jobId ? updater(j) : j)));
  };

  const enqueueCurrentConfig = () => {
    if (!canRun) return;
    const request: PipelineRunRequest = {
      pythonBin: DEFAULT_PYTHON,
      projectRoot: config.projectRoot,
      audioFile: config.audioFile,
      scenarioFile: config.scenarioFile,
      outputBase: config.outputBase,
      mediaFolders: config.mediaFolders,
      processingMode: config.processingMode,
      renderVideo: config.processingMode === "render" ? config.renderVideo : false,
      xmlParts: config.xmlParts,
      maxParallelClips: DEFAULT_MAX_PARALLEL_CLIPS,
      whisperModel: DEFAULT_WHISPER_MODEL,
      whisperLanguage: config.whisperLanguage,
      alignMode: "block_forced"
    };
    const id = `job-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
    const job: LibraryJob = {
      id,
      createdAt: Date.now(),
      status: "queued",
      progress: 0,
      stageLabel: "В очереди",
      logs: [],
      request
    };
    setJobs((prev) => [job, ...prev]);
    setActiveJobId(id);
    setStatus("Задача добавлена в очередь");
  };

  const runJob = async (job: LibraryJob) => {
    setStatus(`Старт задачи ${job.id}`);
    updateJob(job.id, (j) => ({ ...j, status: "running", stageLabel: "Запуск...", progress: 1 }));
    try {
      const { runId } = await startPipelineRun(job.request);
      setRunningRunId(runId);
      setActiveJobId(job.id);
      updateJob(job.id, (j) => ({ ...j, runId }));

      const unlistenLog = await listenPipelineLogs(runId, (line) => {
        updateJob(job.id, (j) => {
          const parsed = parseStageProgress(line);
          return {
            ...j,
            progress: parsed ? Math.max(j.progress, parsed.progress) : j.progress,
            stageLabel: parsed ? parsed.stageLabel : j.stageLabel,
            logs: [...j.logs, line].slice(-600)
          };
        });
      });

      let finishedHandled = false;
      const handleFinished = (payload: { ok: boolean; result?: LibraryJob["result"]; error?: string }) => {
        if (finishedHandled) return;
        finishedHandled = true;
        setRunningRunId(null);
        updateJob(job.id, (j) => ({
          ...j,
          status: payload.ok ? "done" : "failed",
          progress: payload.ok ? 100 : j.progress,
          stageLabel: payload.ok ? "Готово" : "Ошибка",
          result: payload.result,
          error: normalizeErrorText(payload.error),
          logs: payload.error ? [...j.logs, normalizeErrorText(payload.error) ?? payload.error] : j.logs
        }));
        setStatus(payload.ok ? "Задача завершена" : "Задача завершилась с ошибкой");
      };

      const unlistenFinished = await listenPipelineFinished(runId, (payload) => {
        handleFinished(payload);
        void unlistenLog();
        void unlistenFinished();
      });

      const immediateResult = await getPipelineResult(runId);
      if (immediateResult) {
        handleFinished(immediateResult);
        void unlistenLog();
        void unlistenFinished();
      }
    } catch (error) {
      setRunningRunId(null);
      updateJob(job.id, (j) => ({
        ...j,
        status: "failed",
        stageLabel: "Ошибка запуска",
        error: normalizeErrorText(String(error))
      }));
      setStatus(`Ошибка запуска: ${normalizeErrorText(String(error))}`);
    }
  };

  useEffect(() => {
    if (runningRunId) return;
    const next = jobs.find((j) => j.status === "queued");
    if (!next) return;
    void runJob(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs, runningRunId]);

  const chooseProjectRoot = async () => {
    const path = await openDialog({ directory: true, multiple: false });
    if (!path || Array.isArray(path)) return;
    setStatus("Сканируем папки проекта...");
    setField("projectRoot", path);
    try {
      const found = await discoverPaths(path);
      setConfig((prev) => ({
        ...prev,
        projectRoot: path,
        audioFile: found.audioFile || prev.audioFile,
        scenarioFile: found.scenarioFile || prev.scenarioFile,
        outputBase: found.outputBase || prev.outputBase
      }));
      setStatus("Пути обновлены автоматически");
    } catch (error) {
      setStatus(`Ошибка авто-поиска: ${String(error)}`);
    }
  };

  const chooseMediaRoot = async () => {
    const path = await openDialog({ directory: true, multiple: false });
    if (!path || Array.isArray(path)) return;
    setStatus("Сканируем папки с картинками...");
    setField("mediaRoot", path);
    try {
      const found = await discoverMediaFolders(path);
      setConfig((prev) => ({
        ...prev,
        mediaRoot: path,
        mediaFolders: found.mediaFolders
      }));
      setStatus(`Найдено папок: ${found.mediaFolders.length}`);
    } catch (error) {
      setStatus(`Ошибка поиска картинок: ${String(error)}`);
    }
  };

  const chooseFile = async (field: "audioFile" | "scenarioFile") => {
    const path = await openDialog({ directory: false, multiple: false });
    if (!path || Array.isArray(path)) return;
    setField(field, path);
  };

  const chooseOutput = async () => {
    const path = await openDialog({ directory: true, multiple: false });
    if (!path || Array.isArray(path)) return;
    setField("outputBase", path);
  };

  const stopRun = async () => {
    if (!runningRunId) return;
    await stopPipelineRun(runningRunId);
    if (activeJobId) {
      updateJob(activeJobId, (j) => ({ ...j, status: "cancelled", stageLabel: "Остановлено" }));
    }
    setStatus("Остановлено пользователем");
    setRunningRunId(null);
  };

  const removeJobFromLibrary = async (job: LibraryJob, e: MouseEvent) => {
    e.stopPropagation();
    if (job.status === "running" && job.runId) {
      if (runningRunId === job.runId) {
        await stopPipelineRun(job.runId);
        setRunningRunId(null);
      }
    }
    setJobs((prev) => {
      const next = prev.filter((j) => j.id !== job.id);
      setActiveJobId((aid) => (aid === job.id ? next[0]?.id ?? null : aid));
      return next;
    });
  };

  const installAvailableUpdate = async () => {
    if (isInstallingUpdate) return;
    setIsInstallingUpdate(true);
    setUpdateInstallError("");
    setStatus("Скачиваем и устанавливаем обновление...");
    let timeoutId: number | undefined;
    try {
      timeoutId = window.setTimeout(() => {
        setStatus("Обновление ставится дольше обычного... дождись завершения или перезапусти приложение.");
      }, 120000);
      await installUpdate();
      setUpdateStatus((prev) => ({ ...prev, available: false }));
      setStatus("Обновление установлено. Перезапусти приложение.");
    } catch (error) {
      const msg = String(error);
      setUpdateInstallError(msg);
      setStatus(`Ошибка обновления: ${msg}`);
    } finally {
      if (typeof timeoutId !== "undefined") {
        window.clearTimeout(timeoutId);
      }
      setIsInstallingUpdate(false);
    }
  };

  const activeJob = jobs.find((j) => j.id === activeJobId) || jobs[0];

  const openResultPath = async (path?: string | null) => {
    if (!path) return;
    await openPath(path);
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Auto Animator Desktop</h1>
          <p>Очередь рендера и XML</p>
          {appVersion && <p className="versionBadge">Версия приложения: v{appVersion}</p>}
        </div>
        <div className="status">{status}</div>
      </header>

      {updateStatus.checked && updateStatus.available && (
        <section className="updateBanner">
          <div>
            Доступно обновление {updateStatus.version ? `v${updateStatus.version}` : ""}.
            {updateStatus.body ? ` ${updateStatus.body}` : ""}
            {isInstallingUpdate ? ` Устанавливаем${".".repeat(updateDots)}` : ""}
            {updateInstallError ? ` Ошибка: ${updateInstallError}` : ""}
          </div>
          <button onClick={installAvailableUpdate} disabled={isInstallingUpdate}>
            {isInstallingUpdate ? "Установка..." : "Обновить"}
          </button>
        </section>
      )}

      {updateStatus.checked && !updateStatus.available && updateStatus.error && (
        <section className="updateBanner updateBannerError">
          <div>
            Не удалось проверить обновления: {updateStatus.error}
          </div>
        </section>
      )}

      <section className="card">
        <div className="row">
          <label>Профиль</label>
          <div className="buttonRow">
            <button onClick={() => setConfig(loadSavedState())}>Последний</button>
            <button onClick={() => setConfig(defaultState)}>Пользовательский</button>
          </div>
        </div>

        <div className="row">
          <label>Корневая папка проекта</label>
          <input value={config.projectRoot} readOnly placeholder="Выбери папку проекта" />
          <button onClick={chooseProjectRoot}>Выбрать сценарий/аудио</button>
        </div>

        <div className="row">
          <label>Папка с подпапками картинок</label>
          <input value={config.mediaRoot} readOnly placeholder="Выбери корень папок с картинками" />
          <button onClick={chooseMediaRoot}>Выбрать + сортировать</button>
        </div>

        <div className="row">
          <label>Папки медиа (найдено: {config.mediaFolders.length})</label>
          <textarea className="mediaFoldersBox" value={config.mediaFolders.join("\n")} readOnly rows={7} />
        </div>

        <div className="row">
          <label>Аудио</label>
          <input value={config.audioFile} onChange={(e) => setField("audioFile", e.target.value)} />
          <button onClick={() => void chooseFile("audioFile")}>Файл</button>
        </div>

        <div className="row">
          <label>Сценарий</label>
          <input
            value={config.scenarioFile}
            onChange={(e) => setField("scenarioFile", e.target.value)}
          />
          <button onClick={() => void chooseFile("scenarioFile")}>Файл</button>
        </div>

        <div className="row">
          <label>Папка результата</label>
          <input value={config.outputBase} onChange={(e) => setField("outputBase", e.target.value)} />
          <button onClick={chooseOutput}>Папка</button>
        </div>

        <div className="grid2">
          <div className="row compact">
            <label>XML-файлов (частей)</label>
            <select
              className="xmlPartsSelect"
              value={config.xmlParts}
              onChange={(e) => setField("xmlParts", Number(e.target.value))}
            >
              {[1, 2, 3, 4, 5, 6, 8, 10, 12].map((partCount) => (
                <option key={partCount} value={partCount}>
                  {partCount}
                </option>
              ))}
            </select>
          </div>
          <div className="row compact">
            <label>Режим обработки</label>
            <select
              value={config.processingMode}
              onChange={(e) => setField("processingMode", e.target.value as ProcessingMode)}
            >
              <option value="render">С рендером клипов</option>
              <option value="fast_xml">Только XML</option>
            </select>
          </div>
          <div className="row compact">
            <label>Язык Whisper</label>
            <select
              value={config.whisperLanguage}
              onChange={(e) => setField("whisperLanguage", e.target.value as WhisperLanguage)}
            >
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="de">German</option>
            </select>
          </div>
          <div className="row compact">
            <label>Финальный mp4</label>
            <select
              value={config.renderVideo ? "yes" : "no"}
              onChange={(e) => setField("renderVideo", e.target.value === "yes")}
            >
              <option value="no">Отключен</option>
              <option value="yes">Включен</option>
            </select>
          </div>
        </div>

        <div className="buttonRow">
          <button disabled={!canRun} onClick={enqueueCurrentConfig}>
            Добавить в очередь
          </button>
          <button disabled={!runningRunId} onClick={() => void stopRun()}>
            Остановить
          </button>
        </div>
      </section>

      <section className="card logs">
        <h3>Библиотека запусков</h3>
        <div className="queueInfo">
          В очереди: {jobs.filter((j) => j.status === "queued").length} | Выполняется:{" "}
          {jobs.some((j) => j.status === "running") ? "да" : "нет"} | Всего: {jobs.length}
        </div>
        <div className="jobsList">
          {jobs.map((job) => (
            <article
              key={job.id}
              className={`jobCard ${job.id === activeJob?.id ? "active" : ""}`}
              onClick={() => setActiveJobId(job.id)}
            >
              <div className="jobTop">
                <strong>{new Date(job.createdAt).toLocaleString()}</strong>
                <div className="jobTopRight">
                  <span className={`jobStatus ${job.status}`}>{job.status}</span>
                  <button
                    type="button"
                    className="jobDeleteBtn"
                    title="Удалить из библиотеки"
                    onClick={(e) => void removeJobFromLibrary(job, e)}
                  >
                    Удалить
                  </button>
                </div>
              </div>
              <div className="jobStage">{job.stageLabel}</div>
              <div className="progressWrap">
                <div className="progressFill" style={{ width: `${job.progress}%` }} />
              </div>
              <div className="jobMeta">{job.progress}%</div>
              <div className="buttonRow">
                <button onClick={() => void openResultPath(job.request.outputBase)}>Открыть result</button>
                <button onClick={() => void openResultPath(job.result?.xmlPath)}>Открыть XML</button>
                <button onClick={() => void openResultPath(job.result?.finalVideoPath)}>Открыть видео</button>
              </div>
              {job.error && <div className="jobError">{job.error}</div>}
            </article>
          ))}
          {!jobs.length && <div className="hintRow">Пока пусто.</div>}
        </div>
        <h4>Логи выбранной задачи</h4>
        <pre>{activeJob?.logs.join("\n") || ""}</pre>
      </section>
    </div>
  );
}

export default App;
