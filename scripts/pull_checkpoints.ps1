<#
.SYNOPSIS
Pull new ONNX checkpoints (and a training progress summary) from the Lambda instance.

.EXAMPLE
.\scripts\pull_checkpoints.ps1 -InstanceIp 129.146.x.x -RunId CrawlerSumoEGNN_005
Downloads any checkpoints not already present into results\<RunId>\cloud\ and
prints current step count / recent console output.
#>
param(
    [Parameter(Mandatory = $true)][string]$InstanceIp,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$User = "ubuntu",
    [string]$BehaviorName = "CrawlerSumo",
    # Optional: copy the newest checkpoint into the Unity project for viewing
    [switch]$CopyLatestToUnity
)

$remoteDir = "ml-agents/results/$RunId/$BehaviorName"
$localDir = Join-Path $PSScriptRoot "..\results\$RunId\cloud" | Resolve-Path -ErrorAction SilentlyContinue
if (-not $localDir) {
    $localDir = Join-Path $PSScriptRoot "..\results\$RunId\cloud"
    New-Item -ItemType Directory -Force $localDir | Out-Null
}

Write-Host "== Remote checkpoints =="
$remoteFiles = ssh "$User@$InstanceIp" "ls -1 $remoteDir/*.onnx 2>/dev/null"
if (-not $remoteFiles) {
    Write-Host "No checkpoints found yet in $remoteDir"
}

$newest = $null
foreach ($f in $remoteFiles) {
    $name = Split-Path $f -Leaf
    $local = Join-Path $localDir $name
    if (-not (Test-Path $local)) {
        Write-Host "downloading $name ..."
        scp -q "${User}@${InstanceIp}:$f" $local
    }
    $newest = Join-Path $localDir $name
}

if ($CopyLatestToUnity -and $newest) {
    $unityModels = Join-Path $PSScriptRoot "..\Project\Assets\CrawlerSumo\Models"
    New-Item -ItemType Directory -Force $unityModels | Out-Null
    Copy-Item $newest (Join-Path $unityModels "$RunId-latest.onnx") -Force
    Write-Host "Copied newest checkpoint to Project\Assets\CrawlerSumo\Models\$RunId-latest.onnx"
}

Write-Host "`n== Training progress (last console lines with step counts) =="
ssh "$User@$InstanceIp" "grep -E 'Step:|ELO' ml-agents/results/${RunId}_console.log 2>/dev/null | tail -5"

Write-Host "`n== Local copies in $localDir =="
Get-ChildItem $localDir -Filter *.onnx -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
