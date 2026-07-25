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
    # sumo invocations are unaffected. For parkour pass:
    #   -Branch crawler-parkour
    #   -Config config/ppo/CrawlerParkour.yaml
    #   -EnvBin envs/CrawlerParkour_Multi_linux/CrawlerParkour.x86_64
    [string]$Config = "",
    [string]$EnvBin = "",
    [string]$BuildTgz = "",
    [string]$InstanceName = "unity-rl-train",
    [switch]$Resume
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

# --- 8. Start training + watchdog ---
$extra = if ($Resume) { "--resume" } else { "" }
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
Write-Host "  instance:      $instanceId ($InstanceType, $region, `$$price/hr)"
Write-Host "  ip:            $ip"
Write-Host "  training stops:   $($stopTime.ToString('HH:mm'))  (graceful, final checkpoint exported)"
Write-Host "  auto-terminates:  $($killTime.ToString('HH:mm'))  (instance destroyed, billing ends)"
Write-Host ""
Write-Host "Monitor:      .\scripts\lambda\cloud_status.ps1"
Write-Host "TensorBoard:  ssh -i $KeyFile -L 6006:localhost:6006 ubuntu@$ip   ->  http://localhost:6006"
Write-Host "Pull models:  .\scripts\pull_checkpoints.ps1 -InstanceIp $ip -RunId $RunId   (run before $($killTime.ToString('HH:mm'))!)"
