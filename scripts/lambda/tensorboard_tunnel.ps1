<#
.SYNOPSIS
Open a TensorBoard tunnel to the active Lambda training instance.
Looks up the instance IP via the Cloud API so you never need to track IPs.
Leaves the window open while the tunnel is live; Ctrl+C to close.

.EXAMPLE
.\scripts\lambda\tensorboard_tunnel.ps1     # then browse http://localhost:6006
#>
param(
    [string]$KeyFile = "C:\Users\rob\ssh_keys\lambda.pem",
    [string]$ApiKeyFile = "C:\Users\rob\ssh_keys\lambda_api_key.txt",
    [int]$Port = 6006
)
$ErrorActionPreference = "Continue"
$Headers = @{ Authorization = "Bearer $((Get-Content $ApiKeyFile -Raw).Trim())" }
$inst = @((Invoke-RestMethod "https://cloud.lambdalabs.com/api/v1/instances" -Headers $Headers -ErrorAction Stop).data |
    Where-Object { $_.status -eq "active" -and $_.ip }) | Select-Object -First 1
if (-not $inst) { Write-Host "No active instance."; exit 1 }

Write-Host "Tunneling to $($inst.name) ($($inst.ip))"
Write-Host ">>> TensorBoard: http://localhost:$Port <<<  (Ctrl+C to close tunnel)"
ssh -i $KeyFile -o LogLevel=ERROR -o StrictHostKeyChecking=accept-new -N -L "${Port}:localhost:6006" "ubuntu@$($inst.ip)"
