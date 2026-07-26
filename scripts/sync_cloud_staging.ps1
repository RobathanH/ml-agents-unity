<#
.SYNOPSIS
Mirror all trained-agent artifacts from active Lambda instances into a local
staging area, capped at a maximum size.

Staging root (default results\cloud_staging beside this script) is DELETABLE
STAGING SPACE: anything worth keeping long-term should be copied elsewhere
(e.g. a keep\ folder or Project\Assets\<Project>\Models). The pruner will
eventually delete anything left here.

What is synced per run: checkpoint .onnx/.pt pairs, checkpoint.pt (trainer
state, enables --resume), configuration.yaml, tensorboard events, console and
player logs. Files are re-downloaded only when their remote size differs.

MULTI-PROJECT: several projects may train at once from separate worktrees.
 - -Project restricts the pull to instances named "unity-rl-<Project>", so a
   session-bound sync does not drag another project's data over the wire.
   Omit it (the durable scheduled task does) to cover every active instance.
 - Pruning ALWAYS groups run dirs by project and applies -MaxGB to each group
   INDEPENDENTLY, so one project's growth can never evict another project's
   only copy of a checkpoint. The cap is per project, not per root.

Pruning (per project group, when that group is over -MaxGB):
 1. delete unprotected files, oldest first
    (protected per run: 2 newest .onnx, 2 newest .pt, checkpoint.pt,
     configuration.yaml, events files)
 2. if still over: delete oldest whole run folders, newest run in the group
    always kept. Never applied to the _misc group (gifs\, eval\, loose logs).

.EXAMPLE
.\scripts\sync_cloud_staging.ps1                            # every instance
.\scripts\sync_cloud_staging.ps1 -Project CrawlerSumoEGNN   # just mine
#>
param(
    [string]$StagingRoot = "",
    [double]$MaxGB = 32,
    # Sync only from instances named "unity-rl-<Project>". Empty = all of them.
    [string]$Project = "",
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt"
)
$ErrorActionPreference = "Continue"

# Derive from this script's location, not an absolute path: each worktree must
# stage into its OWN results\cloud_staging.
if (-not $StagingRoot) {
    $StagingRoot = Join-Path $PSScriptRoot "..\results\cloud_staging"
}

New-Item -ItemType Directory -Force $StagingRoot | Out-Null
$LogFile = Join-Path $StagingRoot "sync.log"

# Self-heal: a hung predecessor (dead network, stuck API call) blocks the
# 10-minute scheduled task indefinitely -- kill a sync older than 15 min.
# Scoped by a PID file in THIS staging root rather than by command-line match:
# every worktree's root is named "cloud_staging" and is a default rather than
# an argument, so the command line cannot tell two projects' syncs apart, and
# killing another project's live sync would silently stop its data collection.
$LockFile = Join-Path $StagingRoot "sync.pid"
if (Test-Path $LockFile) {
    $prevPid = 0
    if ([int]::TryParse((Get-Content $LockFile -Raw -ErrorAction SilentlyContinue).Trim(), [ref]$prevPid) -and $prevPid -ne $PID) {
        $prev = Get-Process -Id $prevPid -ErrorAction SilentlyContinue
        if ($prev -and $prev.Name -eq "powershell" -and ((Get-Date) - $prev.StartTime).TotalMinutes -gt 15) {
            Stop-Process -Id $prevPid -Force -ErrorAction SilentlyContinue
        } elseif ($prev -and $prev.Name -eq "powershell") {
            Write-Host "sync already running (pid $prevPid); exiting"
            return
        }
    }
}
Set-Content -Path $LockFile -Value $PID -Encoding ascii

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
    $instances = @((Invoke-RestMethod "$Api/instances" -Headers $Headers -ErrorAction Stop -TimeoutSec 30).data |
        Where-Object { $_.status -eq "active" -and $_.ip })
    if ($Project) {
        $mine = "unity-rl-$Project"
        $instances = @($instances | Where-Object { $_.name -eq $mine })
    }
} catch {
    Log "WARN: Lambda API query failed ($($_.Exception.Message)); prune-only pass"
}

foreach ($inst in $instances) {
    $ip = $inst.ip
    # No parens/pipes in the remote command: Windows ssh + PowerShell mangle the
    # quoting. List everything, filter by extension client-side.
    $listing = ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes "ubuntu@$ip" "find /home/ubuntu/ml-agents/results -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null"
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
            scp -q -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes "ubuntu@${ip}:$remote" $local 2>$null
            if ($LASTEXITCODE -eq 0) { $pulled++ } else { Log "WARN: scp failed for $remote" }
        }
    }
    Log "synced $ip : $pulled file(s) pulled"
}

# ---------- 2. Prune, per project group ----------
# Run dirs are named <Project>_<NNN>. The cap is applied to each project
# INDEPENDENTLY: a shared root must never let one project's growth evict
# another project's only copy of a checkpoint. Everything that is not a run
# dir (gifs\, eval\, _sentis\, loose logs) forms the "_misc" group, which is
# file-pruned but never deleted wholesale.
$capBytes = [long]($MaxGB * 1GB)
$groups = @{}
foreach ($d in (Get-ChildItem $StagingRoot -Directory -ErrorAction SilentlyContinue)) {
    $g = if ($d.Name -match '^(.+)_\d+$') { $Matches[1] } else { "_misc" }
    if (-not $groups.ContainsKey($g)) { $groups[$g] = @() }
    $groups[$g] += $d
}
# Loose files at the root belong to _misc so they stay prunable.
$looseFiles = @(Get-ChildItem $StagingRoot -File -ErrorAction SilentlyContinue)
if ($looseFiles.Count -gt 0 -and -not $groups.ContainsKey("_misc")) { $groups["_misc"] = @() }

foreach ($g in ($groups.Keys | Sort-Object)) {
    $dirs = $groups[$g]
    $files = @($dirs | ForEach-Object { Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue })
    if ($g -eq "_misc") { $files += $looseFiles }
    $total = ($files | Measure-Object Length -Sum).Sum
    if ($null -eq $total) { $total = 0 }
    if ($total -le $capBytes) {
        Log ("{0}: {1:N1} GB / {2} GB cap - no prune needed" -f $g, ($total / 1GB), $MaxGB)
        continue
    }

    # Protected set: per run dir, keep the newest artifacts
    $protected = New-Object System.Collections.Generic.HashSet[string]
    foreach ($run in $dirs) {
        $rf = @(Get-ChildItem $run.FullName -Recurse -File -ErrorAction SilentlyContinue)
        foreach ($f in $rf) {
            if ($f.Name -eq "configuration.yaml" -or $f.Name -like "events.out.*" -or $f.Name -eq "checkpoint.pt") {
                [void]$protected.Add($f.FullName)
            }
        }
        foreach ($ext in @("*.onnx", "*.pt")) {
            $rf | Where-Object { $_.Name -like $ext -and $_.Name -ne "checkpoint.pt" } |
                Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
                ForEach-Object { [void]$protected.Add($_.FullName) }
        }
    }

    # Pass 1: unprotected files, oldest first
    foreach ($f in ($files | Where-Object {
                -not $protected.Contains($_.FullName) -and
                $_.FullName -ne $LogFile -and $_.FullName -ne $LockFile } |
            Sort-Object LastWriteTime)) {
        if ($total -le $capBytes) { break }
        $total -= $f.Length
        Log ("prune: {0} ({1:N0} MB)" -f $f.FullName.Substring($StagingRoot.Length + 1), ($f.Length / 1MB))
        Remove-Item $f.FullName -Force -ErrorAction SilentlyContinue
    }

    # Pass 2: whole oldest run dirs in this group (never the newest, never _misc)
    if ($total -gt $capBytes -and $g -ne "_misc") {
        $byAge = $dirs | Sort-Object { (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
                Measure-Object LastWriteTime -Maximum).Maximum }
        foreach ($run in (@($byAge) | Select-Object -SkipLast 1)) {
            if ($total -le $capBytes) { break }
            $sz = (Get-ChildItem $run.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
            $total -= $sz
            Log ("prune run dir: {0} ({1:N1} GB)" -f $run.Name, ($sz / 1GB))
            Remove-Item $run.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    Log ("{0}: {1:N1} GB after prune / {2} GB cap" -f $g, ($total / 1GB), $MaxGB)
}

Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
