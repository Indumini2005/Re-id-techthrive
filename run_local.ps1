<#
Run the whole system on this laptop: both cameras, recognition, analytics
and the dashboard, in one process. No Jetson needed.

    .\run_local.ps1 -CamA "rtsp://user:pass@192.168.1.35:554/stream1" -CamB "rtsp://..."
    .\run_local.ps1 -Webcam                # laptop webcam(s) instead of IP cameras
    .\run_local.ps1 -Video path\to\clip.mp4  # the same clip as both cameras (a dry run)

Then the dashboard opens at http://localhost:8765/ (also reachable from a
phone on the same network at http://<this-laptop-ip>:8765/).

The laptop must be on the same network as the IP cameras. Camera URLs can
also come from $env:EDGE_RTSP_A / $env:EDGE_RTSP_B, so passwords do not need
to be typed every time and never end up in this file.

Extra arguments go straight to main_pipeline.py, e.g.
    .\run_local.ps1 -CamA ... -CamB ... --visitor_threshold 0.6
#>
# No positional binding, so extra main_pipeline.py arguments are never
# mistaken for -CamA / -CamB.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$CamA = $env:EDGE_RTSP_A,
    [string]$CamB = $env:EDGE_RTSP_B,
    [switch]$Webcam,
    [string]$Video,
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Video) {
    $sources = @("CAM_A=$Video", "CAM_B=$Video")
} elseif ($Webcam) {
    # One webcam is enough to try it; a second USB webcam becomes CAM_B.
    $sources = @("CAM_A=0", "CAM_B=1")
    $Rest += "--allow_single_camera"
} elseif ($CamA -and $CamB) {
    $sources = @("CAM_A=$CamA", "CAM_B=$CamB")
} else {
    Write-Host "Give two cameras: -CamA <rtsp url> -CamB <rtsp url>  (or -Webcam, or -Video <file>)" -ForegroundColor Yellow
    exit 1
}

$pyArgs = @("main_pipeline.py")
foreach ($s in $sources) { $pyArgs += @("--source", $s) }
$pyArgs += @("--dashboard", "$Port", "--headless")
if ($Rest) { $pyArgs += $Rest }

if (-not $NoBrowser) {
    # Open the dashboard once the models have had a few seconds to load.
    Start-Job -ScriptBlock { param($p) Start-Sleep 8; Start-Process "http://localhost:$p/" } -ArgumentList $Port | Out-Null
}

Write-Host "Starting ClearFace Re-ID locally - dashboard on http://localhost:$Port/  (Ctrl+C to stop)" -ForegroundColor Cyan
python @pyArgs
