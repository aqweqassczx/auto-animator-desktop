param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$desktopAppDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectRoot = (Resolve-Path (Join-Path $desktopAppDir "..")).Path
$runtimeDir = Join-Path $desktopAppDir "src-tauri\runtime"
$buildDir = Join-Path $desktopAppDir ".build-pipeline-runner"
$venvDir = Join-Path $buildDir "venv"

Write-Host "Desktop app dir: $desktopAppDir"
Write-Host "Project root: $projectRoot"
Write-Host "Preparing runtime dir: $runtimeDir"

if (Test-Path $buildDir) {
  Remove-Item $buildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $buildDir | Out-Null
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

Write-Host "Creating isolated virtual environment..."
& $PythonExe -m venv $venvDir

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Venv python not found: $venvPython"
}

Write-Host "Installing minimal build dependencies into venv..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install pyinstaller faster-whisper

$entryScript = Join-Path $projectRoot "run_pipeline_cli.py"
if (-not (Test-Path $entryScript)) {
  throw "Entry script not found: $entryScript"
}

Write-Host "Building pipeline_runner.exe..."
& $venvPython -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --exclude-module torch `
  --exclude-module torchvision `
  --exclude-module torchaudio `
  --exclude-module tensorflow `
  --exclude-module sklearn `
  --exclude-module scipy `
  --exclude-module pandas `
  --exclude-module matplotlib `
  --exclude-module spacy `
  --exclude-module nltk `
  --exclude-module moviepy `
  --name pipeline_runner `
  --distpath (Join-Path $buildDir "dist") `
  --workpath (Join-Path $buildDir "work") `
  --specpath $buildDir `
  --paths $projectRoot `
  $entryScript

$builtExe = Join-Path $buildDir "dist\pipeline_runner.exe"
if (-not (Test-Path $builtExe)) {
  throw "Build output not found: $builtExe"
}

Copy-Item $builtExe (Join-Path $runtimeDir "pipeline_runner.exe") -Force
Write-Host "Done: $runtimeDir\pipeline_runner.exe"
