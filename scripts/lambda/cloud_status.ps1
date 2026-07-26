<#
.SYNOPSIS
Show all Lambda instances, and for each active one: training progress (latest
steps/ELO), newest checkpoints, and watchdog state.

Instances are named "unity-rl-<Project>". With several projects training at
once, every instance is tagged MINE or OTHER against -Project; only ever act
on your own. Omit -Project to see everything untagged.

.EXAMPLE
.\scripts\lambda\cloud_status.ps1
.\scripts\lambda\cloud_status.ps1 -Project CrawlerSumoEGNN   # tag + total spend
.\scripts\lambda\cloud_status.ps1 -Project CrawlerSumoEGNN -MineOnly
#>
param(
    [string]$Project = "",
    [switch]$MineOnly,
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt"
)
# Continue, not Stop: ssh stderr under Stop kills PS 5.1 scripts (NativeCommandError)
$ErrorActionPreference = "Continue"
$Api = "https://cloud.lambdalabs.com/api/v1"
$Headers = @{ Authorization = "Bearer $((Get-Content $ApiKeyFile -Raw).Trim())" }

$instances = @((Invoke-RestMethod "$Api/instances" -Headers $Headers -ErrorAction Stop).data)
if ($instances.Count -eq 0) {
    Write-Host "No instances running. (Billing: `$0/hr)"
    return
}

$mineName = if ($Project) { "unity-rl-$Project" } else { "" }
$totalRate = ($instances | ForEach-Object { $_.instance_type.price_cents_per_hour } |
    Measure-Object -Sum).Sum / 100
Write-Host ("Account total: {0} instance(s), `${1}/hr combined" -f $instances.Count, $totalRate)
Write-Host ""

foreach ($inst in $instances) {
    $isMine = $mineName -and $inst.name -eq $mineName
    if ($MineOnly -and -not $isMine) { continue }
    $tag = if (-not $mineName) { "" } elseif ($isMine) { "  [MINE]" } else { "  [OTHER PROJECT - do not touch]" }
    $price = $inst.instance_type.price_cents_per_hour / 100
    Write-Host "=== $($inst.name) [$($inst.id)] ===$tag"
    Write-Host "  $($inst.instance_type.name) in $($inst.region.name)  |  status: $($inst.status)  |  `$$price/hr  |  ip: $($inst.ip)"
    if ($inst.status -ne "active" -or -not $inst.ip) { continue }
    # Don't ssh into instances that aren't ours -- another agent owns that box.
    if ($mineName -and -not $isMine) { Write-Host ""; continue }

    # Behavior name is not hardcoded: results/<run>/<Behavior>/*.onnx
    $remote = ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR -o ConnectTimeout=10 "ubuntu@$($inst.ip)" @'
echo "--- progress ---"
grep -h "Step:" ~/ml-agents/results/*_console.log 2>/dev/null | tail -3
echo "--- checkpoints ---"
ls -t ~/ml-agents/results/*/*/*.onnx 2>/dev/null | head -3
echo "--- watchdog ---"
tail -2 ~/watchdog.log 2>/dev/null || echo "(no watchdog log)"
echo "--- load ---"
uptime
'@
    $remote | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
}
