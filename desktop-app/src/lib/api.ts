import { invoke } from "@tauri-apps/api/tauri";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  DiscoverMediaFoldersResult,
  DiscoverPathsResult,
  PipelineResultPayload,
  PipelineRunHandle,
  PipelineRunRequest,
  PipelineRuntimeInfo,
  WhisperModelOption
} from "../types";

export async function listWhisperModels(): Promise<WhisperModelOption[]> {
  return invoke("list_whisper_models");
}

export async function getPipelineRuntimeInfo(
  projectRoot: string,
  pythonBin = "python"
): Promise<PipelineRuntimeInfo> {
  return invoke("get_pipeline_runtime_info", { projectRoot, pythonBin });
}

export async function discoverPaths(projectRoot: string): Promise<DiscoverPathsResult> {
  return invoke("discover_paths", { projectRoot });
}

export async function discoverMediaFolders(mediaRoot: string): Promise<DiscoverMediaFoldersResult> {
  return invoke("discover_media_folders", { mediaRoot });
}

export async function startPipelineRun(request: PipelineRunRequest): Promise<PipelineRunHandle> {
  return invoke("start_pipeline_run", { request });
}

export async function stopPipelineRun(runId: string): Promise<void> {
  return invoke("stop_pipeline_run", { runId });
}

export async function getPipelineResult(runId: string): Promise<PipelineResultPayload | null> {
  return invoke("get_pipeline_result", { runId });
}

export async function listenPipelineLogs(
  runId: string,
  onLog: (line: string) => void
): Promise<UnlistenFn> {
  return listen<{ runId: string; line: string }>("pipeline-log", (event) => {
    if (event.payload.runId === runId) {
      onLog(event.payload.line);
    }
  });
}

export async function listenPipelineFinished(
  runId: string,
  onFinish: (payload: PipelineResultPayload) => void
): Promise<UnlistenFn> {
  return listen<{ runId: string; payload: PipelineResultPayload }>("pipeline-finished", (event) => {
    if (event.payload.runId === runId) {
      onFinish(event.payload.payload);
    }
  });
}
