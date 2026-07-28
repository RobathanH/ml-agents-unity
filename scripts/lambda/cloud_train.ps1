<#
.SYNOPSIS
One-command Lambda Cloud training: launch instance -> provision -> upload build ->
start training -> arm a self-terminating time-limit watchdog.

.EXAMPLE
.\scripts\lambda\cloud_train.ps1 -RunId CrawlerSumoEGNN_006 -TimeLimitHours 8

.NOTES
Requires:
 - Lambda API key in C:\Users\rob\ssh_keys\lambda_api_key.txt (dashboard -> API keys)
 - SSH private key (lambda.pem) whose public half is registered in the dashboard
 - The Linux build tgz (created by the build pipeline) at envs\CrawlerSumoEGNN_Multi_linux.tgz
The API key is copied to the instance (chmod 600) so the watchdog can terminate it.
#>
param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [double]$TimeLimitHours = 8,
    [int]$GraceMinutes = 15,
    [string]$InstanceType = "gpu_1x_a10",
    [int]$NumEnvs = 8,
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt",
    [string]$RepoUrl = "https://github.com/robathanh/ml-agents-unity.git",
    [string]$Branch = "crawler-sumo",
    # Trainer config and env binary on the instance. Defaults are empty, which
    # leaves launch_training.sh on its CrawlerSumoEGNN defaults, so existing
    # sumo invocations are unaffected. For another project pass e.g.
    #   -Branch crawler-parkour
    #   -Config config/ppo/CrawlerParkour.yaml
    #   -EnvBin envs/CrawlerParkour_Multi_linux/CrawlerParkour.x86_64
    [string]$Config = "",
    [string]$EnvBin = "",
    [string]$BuildTgz = "",
    # Project tag. Defaults to the run-id with its trailing _NNN stripped
    # (CrawlerSumoEGNN_014 -> CrawlerSumoEGNN). This is the OWNERSHIP KEY for
    # multi-project use: the instance is named "unity-rl-<Project>", and every
    # other script identifies whose instance is whose by that name. Agents must
    # only ever act on instances carrying their own project tag.
    [string]$Project = "",
    [string]$InstanceName = "",
    # Launch even if this project already has a live instance. Off by default:
    # the standing guardrail is one instance PER PROJECT.
    [switch]$AllowConcurrent,
    [switch]$Resume,
    # Warm start: seed this run's weights from a PREVIOUS run's final checkpoint.
    # Takes a staged run-id (e.g. CrawlerParkour_005). The checkpoint is uploaded
    # from results/cloud_staging/<run>/<behavior>/checkpoint.pt to the matching
    # path under ~/ml-agents/results/ on the instance, which is where
    # mlagents-learn's --initialize-from looks (settings.py resolves it as
    # results_dir/<run>).
    #
    # NOT the same as -Resume, and the difference matters. Resume continues the
    # SAME run-id with its optimiser state, step counter and schedules intact.
    # This takes only the weights, resets the step counter to zero, and starts
    # fresh schedules -- which is the right tool when the environment or the
    # reward has changed underneath the policy, as in run 005 -> 006. ml-agents
    # errors if both are set, so they are refused here rather than there.
    #
    # Observation and action shapes must still match or the state dict will not
    # load. Zeroing a retired observation slot instead of deleting it keeps that
    # true; changing hidden_units, the memory size or the EGNN geometry does not.
    [string]$InitializeFrom = "",
    # Behaviour subdirectory holding checkpoint.pt. Defaults to the project tag,
    # which is the behaviour name for every project here.
    [string]$InitBehavior = ""
)
# NOTE: "Continue" not "Stop" -- in PowerShell 5.1 any native-command stderr
# (e.g. ssh host-key notices) becomes a NativeCommandError that would kill the
# script under Stop. Failures are checked explicitly via $LASTEXITCODE and
# -ErrorAction Stop on the REST calls.
$ErrorActionPreference = "Continue"
$Api = "https://cloud.lambdalabs.com/api/v1"
$ApiKey = (Get-Content $ApiKeyFile -Raw).Trim()
$Headers = @{ Authorization = "Bearer $ApiKey" }
if (-not $BuildTgz) { $BuildTgz = Join-Path $PSScriptRoot "..\..\envs\CrawlerSumoEGNN_Multi_linux.tgz" }
if (-not (Test-Path $BuildTgz)) { throw "Build archive not found: $BuildTgz" }

# The ARCHIVE is what uploads, not the build directory. Nothing otherwise ties
# the two together, so a rebuild that skips re-archiving silently trains the
# PREVIOUS run's environment under the current run's config -- and the numbers
# look entirely plausible. Caught exactly this before run 014 (12:48 archive
# vs 21:39 binaries, which would have kept the removed shrinking ring).
$buildDir = $BuildTgz -replace '\.tgz$', ''
if (Test-Path $buildDir) {
    $tgzTime = (Get-Item $BuildTgz).LastWriteTime
    $newestBin = Get-ChildItem $buildDir -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newestBin -and $newestBin.LastWriteTime -gt $tgzTime) {
        $parent = Split-Path $buildDir -Parent
        $leaf = Split-Path $buildDir -Leaf
        throw ("Build archive is STALE and would train the wrong environment.`n" +
               "  archive:  $BuildTgz ($tgzTime)`n" +
               "  binary:   $($newestBin.Name) ($($newestBin.LastWriteTime))`n" +
               "Re-create it, then relaunch:`n" +
               "  tar -czf '$BuildTgz' -C '$parent' '$leaf'")
    }
}
if (-not $Project) {
    $Project = if ($RunId -match '^(.+)_\d+$') { $Matches[1] } else { $RunId }
}
if (-not $InstanceName) { $InstanceName = "unity-rl-$Project" }

# --- 0. Multi-project preflight ---
# Other projects train from their own worktrees against this same Lambda
# account. Block only on a collision with THIS project; report the others so
# the combined burn rate is visible before spending more.
$live = @((Invoke-RestMethod "$Api/instances" -Headers $Headers -ErrorAction Stop).data |
    Where-Object { $_.status -ne "terminated" })
$ours = @($live | Where-Object { $_.name -eq $InstanceName })
if ($ours.Count -gt 0 -and -not $AllowConcurrent) {
    throw ("$Project already has a live instance ($($ours[0].id), status $($ours[0].status), ip $($ours[0].ip)). " +
           "One instance per project is the standing guardrail. Terminate it, or pass -AllowConcurrent deliberately.")
}
$others = @($live | Where-Object { $_.name -ne $InstanceName })
if ($others.Count -gt 0) {
    # PS 5.1 Measure-Object takes a property NAME, not a scriptblock -- project first.
    $otherRate = ($others | ForEach-Object { $_.instance_type.price_cents_per_hour } |
        Measure-Object -Sum).Sum / 100
    Write-Host "NOTE: $($others.Count) instance(s) from other projects already running (`$$otherRate/hr combined):"
    $others | ForEach-Object { Write-Host "  $($_.name) [$($_.id)] $($_.instance_type.name) - NOT YOURS, do not terminate" }
}

function Invoke-Ssh([string]$ip, [string]$cmd) {
    ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=15 "ubuntu@$ip" $cmd 2>&1 |
        ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "ssh command failed (exit $LASTEXITCODE): $cmd" }
}

# --- 1. SSH key name registered with Lambda ---
$sshKeys = (Invoke-RestMethod "$Api/ssh-keys" -Headers $Headers -ErrorAction Stop).data
if (-not $sshKeys) { throw "No SSH keys registered in your Lambda account" }
$sshKeyName = $sshKeys[0].name
Write-Host "Using Lambda SSH key: $sshKeyName"

# --- 2. Pick a region with capacity ---
$types = (Invoke-RestMethod "$Api/instance-types" -Headers $Headers -ErrorAction Stop).data
$entry = $types.$InstanceType
if (-not $entry) { throw "Unknown instance type '$InstanceType'. Available: $($types.PSObject.Properties.Name -join ', ')" }
$regions = @($entry.regions_with_capacity_available)
if ($regions.Count -eq 0) {
    $avail = $types.PSObject.Properties | Where-Object { @($_.Value.regions_with_capacity_available).Count -gt 0 } |
        ForEach-Object { "$($_.Name) [$(@($_.Value.regions_with_capacity_available)[0].name)]" }
    throw "No capacity for $InstanceType right now. Types with capacity: $($avail -join ', ')"
}
$region = $regions[0].name
$price = $entry.instance_type.price_cents_per_hour / 100
Write-Host "Launching $InstanceType in $region (`$$price/hr, limit $TimeLimitHours h -> ~`$$([math]::Round($price * $TimeLimitHours, 2)))"

# --- 3. Launch ---
$body = @{ region_name = $region; instance_type_name = $InstanceType;
           ssh_key_names = @($sshKeyName); name = $InstanceName } | ConvertTo-Json
$launch = Invoke-RestMethod "$Api/instance-operations/launch" -Method Post -Headers $Headers -ContentType "application/json" -Body $body -ErrorAction Stop
$instanceId = $launch.data.instance_ids[0]
Write-Host "Instance launched: $instanceId"

# --- 4. Wait for active + IP (up to ~12 min) ---
$ip = $null
for ($i = 0; $i -lt 48; $i++) {
    Start-Sleep -Seconds 15
    $inst = (Invoke-RestMethod "$Api/instances/$instanceId" -Headers $Headers -ErrorAction Stop).data
    Write-Host "  status: $($inst.status)"
    if ($inst.status -eq "active" -and $inst.ip) { $ip = $inst.ip; break }
}
if (-not $ip) { throw "Instance did not become active. Check the dashboard; terminate manually if stuck." }
Write-Host "Instance active at $ip"

# --- 5. Wait for SSH ---
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes "ubuntu@$ip" "echo ok" 2>$null
    if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    Start-Sleep -Seconds 10
}
if (-not $ok) { throw "SSH never came up on $ip" }

# --- 6. Provision (clone + venv + mlagents editable install) ---
Write-Host "Provisioning (clone + pip install; several minutes)..."
Invoke-Ssh $ip "git clone --depth 1 --branch $Branch $RepoUrl ml-agents 2>/dev/null || git -C ml-agents pull"
Invoke-Ssh $ip "bash ml-agents/scripts/lambda/setup_instance.sh $RepoUrl $Branch"

# --- 7. Upload build + API key (for self-termination) ---
Write-Host "Uploading build..."
$scpOk = $false
for ($try = 1; $try -le 3; $try++) {
    scp -q -i $KeyFile -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $BuildTgz "ubuntu@${ip}:~/build.tgz"
    if ($LASTEXITCODE -eq 0) { $scpOk = $true; break }
    Write-Host "  scp attempt $try failed; retrying..."
    Start-Sleep -Seconds 10
}
if (-not $scpOk) { throw "scp of build failed after 3 attempts (instance is RUNNING: $instanceId at $ip - finish manually or terminate)" }
Invoke-Ssh $ip "tar -xzf ~/build.tgz -C ~/ml-agents/envs/ && rm ~/build.tgz"
# umask scoped in a subshell: leaking it into the session poisons tmux socket perms
Invoke-Ssh $ip "(umask 177 && echo '$ApiKey' > ~/.lambda_api_key)"

# --- 7b. Warm start: upload the seed checkpoint ---
if ($InitializeFrom) {
    if ($Resume) {
        throw "-Resume and -InitializeFrom are mutually exclusive: resume continues the same run, initialize-from seeds a new one from another run's weights."
    }
    $behavior = if ($InitBehavior) { $InitBehavior } else { $Project }
    $ckpt = Join-Path $PSScriptRoot "..\..\results\cloud_staging\$InitializeFrom\$behavior\checkpoint.pt"
    if (-not (Test-Path $ckpt)) {
        throw "-InitializeFrom $InitializeFrom : no checkpoint at $ckpt. Pull it first with scripts\pull_checkpoints.ps1."
    }
    $sizeMB = [math]::Round((Get-Item $ckpt).Length / 1MB, 1)
    Write-Host "Warm start: uploading $InitializeFrom/$behavior/checkpoint.pt ($sizeMB MB)..."
    Invoke-Ssh $ip "mkdir -p ~/ml-agents/results/$InitializeFrom/$behavior"
    scp -q -i $KeyFile -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 `
        $ckpt "ubuntu@${ip}:~/ml-agents/results/$InitializeFrom/$behavior/checkpoint.pt"
    if ($LASTEXITCODE -ne 0) { throw "scp of the warm-start checkpoint failed (instance is RUNNING: $instanceId at $ip)" }
}

# --- 8. Start training + watchdog ---
$extra = if ($Resume) { "--resume" } elseif ($InitializeFrom) { "--initialize-from=$InitializeFrom" } else { "" }
# Paths are relative to the repo on the instance; expand them there.
$envPrefix = ""
if ($Config) { $envPrefix += "CONFIG=`$HOME/ml-agents/$Config " }
if ($EnvBin) { $envPrefix += "ENV_BIN=`$HOME/ml-agents/$EnvBin " }
Invoke-Ssh $ip "$envPrefix bash ml-agents/scripts/lambda/launch_training.sh $RunId $NumEnvs $extra"
$limitMin = [int]([math]::Round($TimeLimitHours * 60))
Invoke-Ssh $ip "tmux new-window -t train -n watchdog 'bash ~/ml-agents/scripts/lambda/watchdog.sh $instanceId $limitMin $GraceMinutes 2>&1 | tee -a ~/watchdog.log'"

$stopTime = (Get-Date).AddMinutes($limitMin - $GraceMinutes)
$killTime = (Get-Date).AddMinutes($limitMin)
Write-Host ""
Write-Host "=== Training started ==="
Write-Host "  run-id:        $RunId  ($NumEnvs env processes x 12 arenas)"
if ($InitializeFrom) { Write-Host "  warm start:    weights from $InitializeFrom (step counter reset to 0)" }
Write-Host "  instance:      $instanceId ($InstanceType, $region, `$$price/hr)"
Write-Host "  ip:            $ip"
Write-Host "  training stops:   $($stopTime.ToString('HH:mm'))  (graceful, final checkpoint exported)"
Write-Host "  auto-terminates:  $($killTime.ToString('HH:mm'))  (instance destroyed, billing ends)"
Write-Host ""
Write-Host "Monitor:      .\scripts\lambda\cloud_status.ps1"
Write-Host "TensorBoard:  ssh -i $KeyFile -L 6006:localhost:6006 ubuntu@$ip   ->  http://localhost:6006"
Write-Host "Pull models:  .\scripts\pull_checkpoints.ps1 -InstanceIp $ip -RunId $RunId   (run before $($killTime.ToString('HH:mm'))!)"
