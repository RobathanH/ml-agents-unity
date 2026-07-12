<#
.SYNOPSIS
Mirror all trained-agent artifacts from active Lambda instances into a local
staging area, capped at a maximum size.

Staging root (default results\cloud_staging) is DELETABLE STAGING SPACE:
anything worth keeping long-term should be copied elsewhere (e.g. a keep\
folder or Project\Assets\CrawlerSumo\Models). The pruner will eventually
delete anything left here.

What is synced per run: checkpoint .onnx/.pt pairs, checkpoint.pt (trainer
state, enables --resume), configuration.yaml, tensorboard events, console and
player logs. Files are re-downloaded only when their remote size differs.

Pruning (when over -MaxGB):
 1. delete unprotected files, oldest first
    (protected per run: 2 newest .onnx, 2 newest .pt, checkpoint.pt,
     configuration.yaml, events files)
 2. if still over: delete oldest whole run folders, newest run always kept

.EXAMPLE
.\scripts\sync_cloud_staging.ps1            # one sync + prune pass
#>
param(
    [string]$StagingRoot = "D:\UnityRL\ml-agents\results\cloud_staging",
    [double]$MaxGB = 32,
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt"
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force $StagingRoot | Out-Null
$LogFile = Join-Path $StagingRoot "sync.log"

function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

# ---------- 1. Sync from every active instance ----------
$Api = "https://cloud.lambdalabs.com/api/v1"
$instances = @()
try {
    $Headers = @{ Authorization = "Bearer $((Get-Content $ApiKeyFile -Raw).Trim())" }
    $instances = @((Invoke-RestMethod "$Api/instances" -Headers $Headers -ErrorAction Stop).data |
        Where-Object { $_.status -eq "active" -and $_.ip })
} catch {
    Log "WARN: Lambda API query failed ($($_.Exception.Message)); prune-only pass"
}

foreach ($inst in $instances) {
    $ip = $inst.ip
    # No parens/pipes in the remote command: Windows ssh + PowerShell mangle the
    # quoting. List everything, filter by extension client-side.
    $listing = ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15 -o BatchMode=yes "ubuntu@$ip" "find /home/ubuntu/ml-agents/results -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null"
    if ($LASTEXITCODE -ne 0 -or -not $listing) { Log "WARN: could not list files on $ip"; continue }

    $wanted = "\.onnx$|\.pt$|\.yaml$|\.json$|\.log$|/events\.out\."
    $pulled = 0
    foreach ($line in $listing) {
        $parts = "$line".Split(" ", 2)
        if ($parts.Count -ne 2) { continue }
        $size = [long]$parts[0]
        $remote = $parts[1]
        if ($remote -notmatch $wanted) { continue }
        $rel = $remote -replace "^/home/ubuntu/ml-agents/results/", "" -replace "/", "\"
        $local = Join-Path $StagingRoot $rel
        $need = $true
        if (Test-Path $local) {
            if ((Get-Item $local).Length -eq $size) { $need = $false }
        }
        if ($need) {
            New-Item -ItemType Directory -Force (Split-Path $local) | Out-Null
            scp -q -i $KeyFile -o LogLevel=ERROR -o BatchMode=yes "ubuntu@${ip}:$remote" $local 2>$null
            if ($LASTEXITCODE -eq 0) { $pulled++ } else { Log "WARN: scp failed for $remote" }
        }
    }
    Log "synced $ip : $pulled file(s) pulled"
}

# ---------- 2. Prune to cap ----------
function Get-StagingSize {
    $s = (Get-ChildItem $StagingRoot -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($null -eq $s) { return 0 } else { return $s }
}

$capBytes = [long]($MaxGB * 1GB)
$total = Get-StagingSize
if ($total -le $capBytes) {
    Log ("size {0:N1} GB / {1} GB cap - no prune needed" -f ($total / 1GB), $MaxGB)
    return
}

# Build protected set: per run dir, keep the newest artifacts
$protected = New-Object System.Collections.Generic.HashSet[string]
$runDirs = Get-ChildItem $StagingRoot -Directory -ErrorAction SilentlyContinue
foreach ($run in $runDirs) {
    $files = Get-ChildItem $run.FullName -Recurse -File -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        if ($f.Name -eq "configuration.yaml" -or $f.Name -like "events.out.*" -or $f.Name -eq "checkpoint.pt") {
            [void]$protected.Add($f.FullName)
        }
    }
    foreach ($ext in @("*.onnx", "*.pt")) {
        $files | Where-Object { $_.Name -like $ext -and $_.Name -ne "checkpoint.pt" } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
            ForEach-Object { [void]$protected.Add($_.FullName) }
    }
}

# Pass 1: unprotected files, oldest first
$candidates = Get-ChildItem $StagingRoot -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { -not $protected.Contains($_.FullName) -and $_.FullName -ne $LogFile } |
    Sort-Object LastWriteTime
foreach ($f in $candidates) {
    if ($total -le $capBytes) { break }
    $total -= $f.Length
    Log ("prune: {0} ({1:N0} MB)" -f $f.FullName.Substring($StagingRoot.Length + 1), ($f.Length / 1MB))
    Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
}

# Pass 2: whole oldest run dirs (never the newest run)
if ($total -gt $capBytes) {
    $byAge = $runDirs | Sort-Object { (Get-ChildItem $_.FullName -Recurse -File | Measure-Object LastWriteTime -Maximum).Maximum }
    foreach ($run in ($byAge | Select-Object -SkipLast 1)) {
        if ($total -le $capBytes) { break }
        $sz = (Get-ChildItem $run.FullName -Recurse -File | Measure-Object Length -Sum).Sum
        $total -= $sz
        Log ("prune run dir: {0} ({1:N1} GB)" -f $run.Name, ($sz / 1GB))
        Remove-Item $run.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}
Log ("size after prune: {0:N1} GB / {1} GB cap" -f ((Get-StagingSize) / 1GB), $MaxGB)
