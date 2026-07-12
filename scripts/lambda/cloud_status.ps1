<#
.SYNOPSIS
Show all Lambda instances, and for each active one: training progress (latest
steps/ELO), newest checkpoints, and watchdog state.

.EXAMPLE
.\scripts\lambda\cloud_status.ps1
#>
param(
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt"
)
$ErrorActionPreference = "Stop"
$Api = "https://cloud.lambdalabs.com/api/v1"
$Headers = @{ Authorization = "Bearer $((Get-Content $ApiKeyFile -Raw).Trim())" }

$instances = @((Invoke-RestMethod "$Api/instances" -Headers $Headers).data)
if ($instances.Count -eq 0) {
    Write-Host "No instances running. (Billing: `$0/hr)"
    return
}

foreach ($inst in $instances) {
    $price = $inst.instance_type.price_cents_per_hour / 100
    Write-Host "=== $($inst.name) [$($inst.id)] ==="
    Write-Host "  $($inst.instance_type.name) in $($inst.region.name)  |  status: $($inst.status)  |  `$$price/hr  |  ip: $($inst.ip)"
    if ($inst.status -ne "active" -or -not $inst.ip) { continue }

    $remote = ssh -i $KeyFile -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$($inst.ip)" @'
echo "--- progress ---"
grep -h "Step:" ~/ml-agents/results/*_console.log 2>/dev/null | tail -3
echo "--- checkpoints ---"
ls -t ~/ml-agents/results/*/CrawlerSumo/*.onnx 2>/dev/null | head -3
echo "--- watchdog ---"
tail -2 ~/watchdog.log 2>/dev/null || echo "(no watchdog log)"
echo "--- load ---"
uptime
'@
    $remote | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
}
