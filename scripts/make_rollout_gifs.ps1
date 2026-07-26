<#
.SYNOPSIS
Generate rollout GIFs of the newest staged checkpoint against a variety of
opponents: itself (mirror), a mid-training checkpoint, and the earliest one.

Pipeline per matchup:
  staged .onnx  --editor batch-->  .sentis  --viewer player-->  PNG frames
  --ffmpeg-->  cloud_staging\gifs\<run>\<name>.gif

.EXAMPLE
.\scripts\make_rollout_gifs.ps1                     # newest run in staging
.\scripts\make_rollout_gifs.ps1 -RunId CrawlerSumoEGNN_006 -CaptureSeconds 30
#>
# Paths default to THIS worktree (see $PSScriptRoot resolution below), not an
# absolute repo: with one worktree per project, an absolute path would read
# another project's staging and builds. Unity itself is machine-wide.
param(
    [string]$RunId = "",
    [string]$StagingRoot = "",
    [string]$BehaviorName = "CrawlerSumo",
    [int]$CaptureSeconds = 25,
    [int]$CaptureFps = 10,
    [string]$UnityExe = "C:\Program Files\Unity\Hub\Editor\6000.0.40f1\Editor\Unity.exe",
    [string]$ViewerExe = "",
    [string]$ProjectPath = "",
    # name=value overrides forwarded to the viewer as --env-param (match the
    # run's training physics, e.g. "joint_strength_multiplier_min=1.5")
    [string[]]$EnvParams = @()
)
$ErrorActionPreference = "Continue"
$repo = Join-Path $PSScriptRoot ".."
if (-not $StagingRoot) { $StagingRoot = Join-Path $repo "results\cloud_staging" }
if (-not $ProjectPath) { $ProjectPath = Join-Path $repo "Project" }
if (-not $ViewerExe) { $ViewerExe = Join-Path $repo "envs\CrawlerSumoEGNN_viewer_win\CrawlerSumoViewer.exe" }
$LogFile = Join-Path $StagingRoot "gif.log"
function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

# ---------- pick run + checkpoints ----------
if (-not $RunId) {
    $runDir = Get-ChildItem $StagingRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName $BehaviorName) } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $runDir) { Log "no runs in staging"; exit 1 }
    $RunId = $runDir.Name
}
$ckptDir = Join-Path $StagingRoot "$RunId\$BehaviorName"
$ckpts = Get-ChildItem $ckptDir -Filter "$BehaviorName-*.onnx" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $step = [long]($_.BaseName -replace "^$BehaviorName-", "")
        [pscustomobject]@{ Step = $step; File = $_.FullName; Base = $_.BaseName }
    } | Sort-Object Step
if (-not $ckpts) { Log "no checkpoints for $RunId"; exit 1 }

$current = $ckpts[-1]
$opponents = [ordered]@{ "mirror" = $current }
if ($ckpts.Count -ge 3) {
    $target = [long]($current.Step / 2)
    $mid = $ckpts | Sort-Object { [math]::Abs($_.Step - $target) } | Select-Object -First 1
    if ($mid.Step -ne $current.Step) { $opponents["mid$([math]::Round($mid.Step/1e6,1))M"] = $mid }
}
if ($ckpts.Count -ge 2 -and $ckpts[0].Step -ne $current.Step) {
    $opponents["early$([math]::Round($ckpts[0].Step/1e6,1))M"] = $ckpts[0]
}
Log "run $RunId : current=$($current.Base), opponents: $($opponents.Keys -join ', ')"

# ---------- ensure .sentis conversions ----------
$sentisDir = Join-Path $StagingRoot "_sentis\$RunId"
New-Item -ItemType Directory -Force $sentisDir | Out-Null
$needed = @($current) + @($opponents.Values) | Sort-Object Step -Unique
$missing = $needed | Where-Object { -not (Test-Path (Join-Path $sentisDir "$($_.Base).sentis")) }
if ($missing) {
    Log "converting $($missing.Count) onnx -> sentis (editor batch)..."
    $env:ONNX_CONVERT_LIST = ($missing | ForEach-Object { $_.File }) -join ";"
    $env:SENTIS_OUT_DIR = $sentisDir
    & $UnityExe -projectPath $ProjectPath -batchmode -nographics `
        -executeMethod TrainingBuilds.ConvertOnnxToSentis `
        -logFile (Join-Path $StagingRoot "_sentis\convert.log") 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Log "WARN: sentis conversion exited $LASTEXITCODE (see _sentis\convert.log)" }
}
$currentSentis = Join-Path $sentisDir "$($current.Base).sentis"
if (-not (Test-Path $currentSentis)) { Log "ERROR: no .sentis for current model - Sentis import may have failed"; exit 1 }

# ---------- run matchups + encode ----------
$gifDir = Join-Path $StagingRoot "gifs\$RunId"
New-Item -ItemType Directory -Force $gifDir | Out-Null
$stamp = Get-Date -Format "MMdd_HHmm"

foreach ($label in $opponents.Keys) {
    $opp = $opponents[$label]
    $oppSentis = Join-Path $sentisDir "$($opp.Base).sentis"
    if (-not (Test-Path $oppSentis)) { Log "skip ${label}: missing sentis"; continue }

    $frames = Join-Path $env:TEMP "rollout_frames_$label"
    Remove-Item $frames -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $frames | Out-Null

    $viewerArgs = @(
        "--team0-model", $currentSentis, "--team1-model", $oppSentis,
        "--capture-dir", $frames, "--capture-seconds", $CaptureSeconds, "--capture-fps", $CaptureFps,
        "-screen-width", "640", "-screen-height", "360", "-screen-fullscreen", "0", "-popupwindow"
    )
    foreach ($ep in $EnvParams) { $viewerArgs += @("--env-param", $ep) }
    $p = Start-Process -FilePath $ViewerExe -PassThru -ArgumentList $viewerArgs
    if (-not $p.WaitForExit(($CaptureSeconds + 45) * 1000)) {
        $p.Kill()
        Log "WARN: viewer timed out for $label"
    }

    $count = (Get-ChildItem $frames -Filter "*.png" -ErrorAction SilentlyContinue).Count
    if ($count -lt 5) { Log "WARN: only $count frames for $label; skipping encode"; continue }

    $gif = Join-Path $gifDir "step$([math]::Round($current.Step/1e6,1))M_vs_${label}_$stamp.gif"
    ffmpeg -y -loglevel error -framerate $CaptureFps -i "$frames\f%05d.png" `
        -vf "fps=$CaptureFps,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" `
        $gif
    if ($LASTEXITCODE -eq 0) {
        Log ("gif: {0} ({1:N1} MB, {2} frames)" -f (Split-Path $gif -Leaf), ((Get-Item $gif).Length / 1MB), $count)
    } else {
        Log "WARN: ffmpeg failed for $label"
    }
    Remove-Item $frames -Recurse -Force -ErrorAction SilentlyContinue
}
Log "done -> $gifDir"
