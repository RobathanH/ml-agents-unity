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
 - -Project restricts the pull to instances named "unity-rl-<Project>" AND
   stages everything into THIS worktree's root. That is the right mode for a
   session-bound or manual sync.
 - With NO -Project -- what the durable scheduled task does -- every active
   instance is covered, and each instance's files are ROUTED to the staging
   root of the worktree that owns that project. See "routing" below. One job
   therefore serves every project correctly, so a new project needs no new
   scheduled task (registering those requires elevation) and cannot be missed.
 - Pruning ALWAYS groups run dirs by project and applies -MaxGB to each group
   INDEPENDENTLY, in every root touched, so one project's growth can never
   evict another project's only copy of a checkpoint.

ROUTING: a project's owning worktree is the one holding
results\cloud_staging\management_<Project>.md. That file is already mandatory
per scripts\lambda\README.md section 7 (one per project, in exactly the root
that project stages into), so it is a registry that already exists and cannot
drift out of sync with reality. A project whose journal does not exist yet
falls back to THIS root and logs a NOTE -- nothing is ever dropped, the
fallback is just visible instead of silent.

Pruning (per project group, when that group is over -MaxGB):
 1. delete unprotected files, oldest first
    (protected per run: 2 newest .onnx, 2 newest .pt, checkpoint.pt,
     configuration.yaml, events files)
 2. if still over: delete oldest whole run folders, newest run in the group
    always kept. Never applied to the _misc group (gifs\, eval\, loose logs).

.EXAMPLE
.\scripts\sync_cloud_staging.ps1                            # every instance, routed
.\scripts\sync_cloud_staging.ps1 -Project CrawlerSumoEGNN   # just mine, here
#>
param(
    [string]$StagingRoot = "",
    [double]$MaxGB = 32,
    # Sync only from instances named "unity-rl-<Project>", into THIS root.
    # Empty = every instance, each routed to its owning worktree.
    [string]$Project = "",
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt"
)
$ErrorActionPreference = "Continue"

# Derive from this script's location, not an absolute path: each worktree must
# stage into its OWN results\cloud_staging. GetFullPath collapses the "..", so
# this string compares equal to a root discovered via worktree enumeration.
if (-not $StagingRoot) {
    $StagingRoot = Join-Path $PSScriptRoot "..\results\cloud_staging"
}
$StagingRoot = [System.IO.Path]::GetFullPath($StagingRoot)

New-Item -ItemType Directory -Force $StagingRoot | Out-Null
$LogFile = Join-Path $StagingRoot "sync.log"

# Lines about another project's data are written to that project's own log as
# well, so each agent can read its own sync history in its own worktree.
function Log([string]$msg, [string]$root = "") {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $LogFile -Value $line
    if ($root) {
        $alt = Join-Path $root "sync.log"
        if ($alt -ne $LogFile) { Add-Content -Path $alt -Value $line -ErrorAction SilentlyContinue }
    }
    Write-Host $line
}

# ---------- 0. Locking, one lock per staging root ----------
# Self-heal: a hung predecessor (dead network, stuck API call) blocks the
# 10-minute scheduled task indefinitely -- steal a lock older than 15 min.
# Scoped by a PID file IN EACH ROOT rather than by command-line match: every
# worktree's root is named "cloud_staging" and is a default rather than an
# argument, so the command line cannot tell two projects' syncs apart, and
# killing another project's live sync would silently stop its data collection.
$Held = New-Object System.Collections.Generic.List[string]
function Enter-SyncLock([string]$root) {
    $lf = Join-Path $root "sync.pid"
    if ($Held.Contains($lf)) { return $true }
    if (Test-Path $lf) {
        $prevPid = 0
        if ([int]::TryParse((Get-Content $lf -Raw -ErrorAction SilentlyContinue).Trim(), [ref]$prevPid) -and $prevPid -ne $PID) {
            $prev = Get-Process -Id $prevPid -ErrorAction SilentlyContinue
            if ($prev -and $prev.Name -eq "powershell") {
                if (((Get-Date) - $prev.StartTime).TotalMinutes -gt 15) {
                    Stop-Process -Id $prevPid -Force -ErrorAction SilentlyContinue
                } else {
                    return $false
                }
            }
        }
    }
    New-Item -ItemType Directory -Force $root | Out-Null
    Set-Content -Path $lf -Value $PID -Encoding ascii
    [void]$Held.Add($lf)
    return $true
}

if (-not (Enter-SyncLock $StagingRoot)) {
    Write-Host "sync already running in $StagingRoot; exiting"
    return
}

# ---------- 0b. Which worktree owns which project ----------
# git is on the machine PATH, so this resolves under the scheduled task's S4U
# logon too. If it ever does not, the list comes back empty and every project
# falls back to this root -- degraded to the old behaviour, never broken.
$Worktrees = @()
try {
    $Worktrees = @(git -C (Split-Path $PSScriptRoot -Parent) worktree list --porcelain 2>$null |
        Where-Object { $_ -like "worktree *" } |
        ForEach-Object { ($_.Substring(9).Trim()) -replace "/", "\" })
} catch {
    Log "WARN: could not enumerate worktrees ($($_.Exception.Message)); staging everything here"
}

$RootCache = @{}
function Resolve-StagingRoot([string]$proj) {
    # An explicit -Project means "this worktree, this project": never route away.
    if ($Project -or -not $proj) { return $StagingRoot }
    if ($RootCache.ContainsKey($proj)) { return $RootCache[$proj] }
    # Track "found a marker" separately from the resolved path. A project whose
    # owner IS this worktree resolves to $StagingRoot legitimately, so comparing
    # paths would report every locally-owned project as unregistered.
    $resolved = $null
    foreach ($wt in $Worktrees) {
        $cand = Join-Path $wt "results\cloud_staging"
        if (Test-Path (Join-Path $cand "management_$proj.md")) { $resolved = $cand; break }
    }
    if (-not $resolved) {
        Log "NOTE: no management_$proj.md in any worktree -- staging $proj into $StagingRoot"
        $resolved = $StagingRoot
    }
    $RootCache[$proj] = $resolved
    return $resolved
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
    # The instance name is the ownership key (README section 1), so it -- not
    # the file paths -- decides where this instance's artifacts land.
    $instProject = ""
    if ("$($inst.name)" -match '^unity-rl-(.+)$') { $instProject = $Matches[1] }
    $dest = Resolve-StagingRoot $instProject
    if (-not (Enter-SyncLock $dest)) {
        Log "skip $ip ($instProject): a sync is already live in $dest"
        continue
    }

    # No parens/pipes in the remote command: Windows ssh + PowerShell mangle the
    # quoting. List everything, filter by extension client-side.
    # mtime as well as size, because size alone CANNOT see the file that matters
    # most. checkpoint.pt is rewritten in place every save and is byte-identical
    # in length every time (fixed architecture), so a size comparison declares it
    # unchanged forever and the staging copy silently stays at whatever the first
    # sync of the run happened to catch. Found this holding run 004's 11:50 copy
    # while the instance had the 19:41 final -- and checkpoint.pt is precisely
    # the file --resume and --initialize-from read.
    $listing = ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes "ubuntu@$ip" "find /home/ubuntu/ml-agents/results -maxdepth 3 -type f -printf '%s %T@ %p\n' 2>/dev/null"
    if ($LASTEXITCODE -ne 0 -or -not $listing) { Log "WARN: could not list files on $ip"; continue }

    $epoch = [datetime]"1970-01-01T00:00:00Z"
    $wanted = "\.onnx$|\.pt$|\.yaml$|\.json$|\.log$|/events\.out\."
    $pulled = 0
    foreach ($line in $listing) {
        $parts = "$line".Split(" ", 3)
        if ($parts.Count -ne 3) { continue }
        $size = [long]$parts[0]
        $remoteMtime = $epoch.AddSeconds([double]$parts[1]).ToLocalTime()
        $remote = $parts[2]
        if ($remote -notmatch $wanted) { continue }
        $rel = $remote -replace "^/home/ubuntu/ml-agents/results/", "" -replace "/", "\"
        $local = Join-Path $dest $rel
        $need = $true
        if (Test-Path $local) {
            $li = Get-Item $local
            # scp -p below stamps the local copy with the REMOTE mtime, so this
            # compares like with like. Files pulled by an older revision of this
            # script carry their download time instead, which is later than the
            # remote mtime -- so they read as current and are not re-fetched.
            # 2s of slack absorbs filesystem timestamp granularity.
            if ($li.Length -eq $size -and $li.LastWriteTime -ge $remoteMtime.AddSeconds(-2)) { $need = $false }
        }
        if ($need) {
            New-Item -ItemType Directory -Force (Split-Path $local) | Out-Null
            scp -q -p -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes "ubuntu@${ip}:$remote" $local 2>$null
            if ($LASTEXITCODE -eq 0) { $pulled++ } else { Log "WARN: scp failed for $remote" }
        }
    }
    Log "synced $ip [$instProject] -> $dest : $pulled file(s) pulled" $dest
}

# ---------- 2. Prune, per project group, in every root touched ----------
# Run dirs are named <Project>_<NNN>. The cap is applied to each project
# INDEPENDENTLY: a shared root must never let one project's growth evict
# another project's only copy of a checkpoint. Everything that is not a run
# dir (gifs\, eval\, _sentis\, loose logs) forms the "_misc" group, which is
# file-pruned but never deleted wholesale.
function Invoke-Prune([string]$root) {
    $capBytes = [long]($MaxGB * 1GB)
    $groups = @{}
    foreach ($d in (Get-ChildItem $root -Directory -ErrorAction SilentlyContinue)) {
        $g = if ($d.Name -match '^(.+)_\d+$') { $Matches[1] } else { "_misc" }
        if (-not $groups.ContainsKey($g)) { $groups[$g] = @() }
        $groups[$g] += $d
    }
    # Loose files at the root belong to _misc so they stay prunable. The sync's
    # own log and lock are never candidates -- deleting the live lock file would
    # let a second sync start on top of this one.
    $looseFiles = @(Get-ChildItem $root -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "sync.log" -and $_.Name -ne "sync.pid" })
    if ($looseFiles.Count -gt 0 -and -not $groups.ContainsKey("_misc")) { $groups["_misc"] = @() }

    foreach ($g in ($groups.Keys | Sort-Object)) {
        $dirs = $groups[$g]
        $files = @($dirs | ForEach-Object { Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue })
        if ($g -eq "_misc") { $files += $looseFiles }
        $total = ($files | Measure-Object Length -Sum).Sum
        if ($null -eq $total) { $total = 0 }
        if ($total -le $capBytes) {
            Log ("{0}: {1:N1} GB / {2} GB cap - no prune needed" -f $g, ($total / 1GB), $MaxGB) $root
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
        foreach ($f in ($files | Where-Object { -not $protected.Contains($_.FullName) } |
                Sort-Object LastWriteTime)) {
            if ($total -le $capBytes) { break }
            $total -= $f.Length
            Log ("prune: {0} ({1:N0} MB)" -f $f.FullName.Substring($root.Length + 1), ($f.Length / 1MB)) $root
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
                Log ("prune run dir: {0} ({1:N1} GB)" -f $run.Name, ($sz / 1GB)) $root
                Remove-Item $run.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
        Log ("{0}: {1:N1} GB after prune / {2} GB cap" -f $g, ($total / 1GB), $MaxGB) $root
    }
}

foreach ($lf in @($Held)) { Invoke-Prune (Split-Path $lf -Parent) }
foreach ($lf in @($Held)) { Remove-Item $lf -Force -ErrorAction SilentlyContinue }
