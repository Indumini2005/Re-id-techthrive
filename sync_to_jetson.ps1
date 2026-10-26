<#
Copy this project to the Jetson Nano and keep it in step.

    .\sync_to_jetson.ps1            # code only (a few hundred KB)
    .\sync_to_jetson.ps1 -Models    # also the ONNX models (~195 MB, once)

Why not VS Code Remote-SSH: the Nano runs Ubuntu 18.04 (glibc 2.27) and
current VS Code needs glibc 2.28 on the remote, so its server cannot run
there. It would also cost the Nano ~300-500 MB of its 4 GB. So: edit here,
sync, and run on the board in a terminal (`ssh jetson`).

What is sent: every file git tracks or would track (respects .gitignore),
so models, vector_db/ and captures/ on the Nano are never overwritten by a
code sync. Documents (.pptx/.pdf/.docx) are skipped.

The USB link to the Nano drops packets, so every network step retries.
Files are packed into one tar first, so one transfer has to survive
instead of hundreds.
#>
param(
    [switch]$Models,
    [string]$SshHost = "jetson",
    [string]$RemoteDir = "Re-id-techthrive",     # relative to the Nano's home
    [int]$Retries = 6
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$git = "C:\Program Files\Git\cmd\git.exe"
if (-not (Test-Path $git)) { $git = "git" }


function Invoke-WithRetry {
    param([string]$What, [scriptblock]$Action)
    for ($i = 1; $i -le $Retries; $i++) {
        & $Action
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "  $What failed (attempt $i of $Retries); retrying..." -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
    throw "$What failed after $Retries attempts. Is the Nano on? Try: ping 192.168.55.1"
}


# --- 1. Collect the files ---------------------------------------------------
Write-Host "Collecting files..."
$files = & $git -C $root ls-files --cached --others --exclude-standard |
    Where-Object { $_ -notmatch '\.(pptx|pdf|docx)$' } |
    Where-Object { Test-Path (Join-Path $root $_) }
if (-not $files) { throw "git listed no files - is this the project folder?" }

$list = Join-Path $env:TEMP "jetson_sync_files.txt"
$tar = Join-Path $env:TEMP "jetson_sync.tar"
# tar reads this list; it must be plain UTF-8 without a byte-order mark.
[IO.File]::WriteAllLines($list, [string[]]$files, (New-Object Text.UTF8Encoding($false)))

# --- 2. Pack them -----------------------------------------------------------
if (Test-Path $tar) { Remove-Item $tar -Force }
& "$env:SystemRoot\System32\tar.exe" -cf $tar -C $root -T $list
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$kb = [math]::Round((Get-Item $tar).Length / 1KB)
Write-Host ("Packed {0} files ({1} KB)" -f $files.Count, $kb)

# --- 3. Send and unpack -----------------------------------------------------
Write-Host "Sending to ${SshHost}:~/$RemoteDir ..."
Invoke-WithRetry "upload" { scp -q -o ConnectTimeout=45 $tar "${SshHost}:/tmp/jetson_sync.tar" }
Invoke-WithRetry "unpack" {
    # -m: stamp files with the Nano's own time. Its clock has no battery and
    # resets when unplugged, so laptop timestamps look years in the future
    # and tar would warn about every file.
    # Then strip the CRLF line endings Windows gives the working tree: the
    # Nano's copy is a git checkout, and CRLF would show every text file as
    # modified and block `git pull`. grep -I skips binary files.
    # Built from single-quoted pieces because PowerShell 5.1 drops embedded
    # double quotes when it passes an argument to ssh.
    $unpack = 'mkdir -p ~/' + $RemoteDir + '/models && tar -xmf /tmp/jetson_sync.tar -C ~/' + $RemoteDir +
              ' && cd ~/' + $RemoteDir +
              ' && tar -tf /tmp/jetson_sync.tar | xargs -d ''\n'' grep -IlP ''\r'' 2>/dev/null | xargs -r -d ''\n'' sed -i ''s/\r$//''' +
              '; rm -f /tmp/jetson_sync.tar; chmod +x *.sh 2>/dev/null; echo unpacked'
    ssh -o ConnectTimeout=45 $SshHost $unpack
}
Remove-Item $tar, $list -Force

# --- 4. Models (optional, only what is missing or changed) ------------------
if ($Models) {
    $local = Get-ChildItem (Join-Path $root "models") -Filter *.onnx -ErrorAction SilentlyContinue
    if (-not $local) {
        Write-Host "No .onnx files in models/ here; nothing to send." -ForegroundColor Yellow
    }
    foreach ($m in $local) {
        $remoteSize = ssh -o ConnectTimeout=45 $SshHost "stat -c %s ~/$RemoteDir/models/$($m.Name) 2>/dev/null || echo 0"
        if ("$remoteSize".Trim() -eq "$($m.Length)") {
            Write-Host ("  {0}: already there" -f $m.Name)
            continue
        }
        Write-Host ("  {0}: sending {1:N1} MB..." -f $m.Name, ($m.Length / 1MB))
        Invoke-WithRetry "copy $($m.Name)" {
            scp -q -o ConnectTimeout=45 $m.FullName "${SshHost}:$RemoteDir/models/$($m.Name)"
        }
    }
}

Write-Host ""
Write-Host "Done. On the Nano:  ssh $SshHost   then   cd ~/$RemoteDir" -ForegroundColor Green
