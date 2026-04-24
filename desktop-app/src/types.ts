export type ProcessingMode = "render" | "fast_xml";
export type WhisperLanguage = "en" | "es" | "de";

export interface PipelineRunRequest {
  pythonBin: string;
  projectRoot: string;
  audioFile: string;
  scenarioFile: string;
  outputBase: string;
  mediaFolders: string[];
  processingMode: ProcessingMode;
  renderVideo: boolean;
  xmlParts: number;
  maxParallelClips: number;
  whisperModel: string;
  whisperLanguage: WhisperLanguage;
  alignMode: "block_forced";
}

export interface PipelineRunHandle {
  runId: string;
}

export interface PipelineResultPayload {
  ok: boolean;
  result?: {
    outputDir: string;
    finalVideoPath: string | null;
    xmlPath: string | null;
    xmlParts?: string[];
    phraseCount: number;
    assetCount: number;
    clipsPlanned: number;
    clipsRendered: number;
    clipsUsedInXml: number;
  };
  error?: string;
  traceback?: string;
}

export interface DiscoverPathsResult {
  mediaFolders: string[];
  audioFile: string;
  scenarioFile: string;
  outputBase: string;
}

export interface DiscoverMediaFoldersResult {
  mediaFolders: string[];
}

export interface UpdateStatus {
  checked: boolean;
  available: boolean;
  version?: string;
  body?: string;
  error?: string;
}
