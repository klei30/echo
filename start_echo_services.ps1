param(
    [switch]$SkipGemma,
    [switch]$SkipLiveKit,
    [switch]$SkipEcho,
    [switch]$WithVoice,
    [switch]$SkipVoiceAgent,
    [switch]$WithTunnel,
    [switch]$RestartGemma,
    [switch]$RestartTunnel,
    [switch]$KeepQwen,
    [int]$GemmaTimeoutSeconds = 600,
    [int]$EchoTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WslDistro = "Ubuntu-24.04"
$Python = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python.exe"
}

function Write-Step($Text) {
    Write-Host ""
    Write-Host $Text -ForegroundColor Cyan
}

function Test-Http($Url, $TimeoutSeconds = 3) {
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSeconds | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-Http($Name, $Url, $TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    Write-Host "Waiting for $Name " -NoNewline
    while ((Get-Date) -lt $deadline) {
        if (Test-Http $Url 3) {
            Write-Host "ready" -ForegroundColor Green
            return $true
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 5
    }
    Write-Host " timeout" -ForegroundColor Red
    return $false
}

function Stop-PortListener($Port, $Name) {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        try {
            Write-Host "Stopping $Name listener on port $Port PID $($listener.OwningProcess)"
            Stop-Process -Id $listener.OwningProcess -Force -ErrorAction Stop
        } catch {
            Write-Host "Could not stop PID $($listener.OwningProcess): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

function Wait-PortFree($Port, $Name, $TimeoutSeconds = 10) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $listeners) {
            return $true
        }
        Start-Sleep -Seconds 1
    }

    $remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($remaining) {
        $pids = ($remaining | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
        throw "$Name is still holding port $Port after stop attempt. Remaining PID(s): $pids. Run PowerShell as Administrator or stop that process manually."
    }
    return $true
}

function Test-WslDistro {
    $result = & wsl.exe -d $WslDistro sh -lc "printf ok" 2>&1
    if ($LASTEXITCODE -ne 0 -or ($result -join "") -notmatch "ok") {
        Write-Host "WSL distro '$WslDistro' is not available for this Windows user." -ForegroundColor Yellow
        Write-Host "Gemma 4 vLLM and Unsloth training need that distro before they can start." -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Stop-EchoBackend {
    try {
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match "(^|[\\\s`"'])main\.py([`"'\s]|$)" } |
            ForEach-Object {
                Write-Host "Stopping old Echo backend PID $($_.ProcessId)"
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
    } catch {
        Write-Host "Could not inspect Python command lines: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    Stop-PortListener 8002 "Echo backend"
    Wait-PortFree 8002 "Echo backend" 10 | Out-Null
}

function Stop-VoiceAgent {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "(^|[\\\s`"'])voice_agent\.py([`"'\s]|$)" } |
        ForEach-Object {
            Write-Host "Stopping old voice agent PID $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

function Stop-LegacyQwen {
    if (-not (Test-WslDistro)) {
        return
    }
    Write-Host "Stopping legacy Qwen vLLM if present..."
    & wsl.exe -d $WslDistro -u root sh -lc "systemctl disable --now vllm.service >/dev/null 2>&1 || true" 2>$null
    & wsl.exe -d $WslDistro bash -lc "pkill -f 'vllm.entrypoints.openai.api_server --model Qwen' || true; pkill -f 'vllm serve Qwen' || true" 2>$null
}

function Stop-Gemma {
    if (-not (Test-WslDistro)) {
        return
    }
    Write-Host "Stopping stale Gemma 4 vLLM if present..."
    & wsl.exe -d $WslDistro bash -lc "pkill -f 'start_gemma4_e2b_vllm' || true; pkill -f 'vllm serve .*gemma-4-E2B-it' || true; pkill -f 'vllm serve .*gemma4_e2b' || true; pkill -f 'served-model-name gemma4_e2b' || true" 2>$null
    Start-Sleep -Seconds 3

    Stop-PortListener 8003 "Gemma 4 vLLM"
    Wait-PortFree 8003 "Gemma 4 vLLM" 10 | Out-Null
}

function Start-LiveKit {
    if (Test-NetConnection -ComputerName localhost -Port 7880 -InformationLevel Quiet) {
        Write-Host "LiveKit already listening on port 7880" -ForegroundColor Green
        return
    }

    $livekitExe = "C:\LiveKit\livekit-server.exe"
    $livekitConfig = "C:\LiveKit\livekit.yaml"
    if (-not (Test-Path $livekitExe)) {
        throw "LiveKit server not found at $livekitExe"
    }
    if (-not (Test-Path $livekitConfig)) {
        throw "LiveKit config not found at $livekitConfig"
    }

    Get-Process livekit-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Process -FilePath $livekitExe `
        -ArgumentList @("--config", $livekitConfig) `
        -WorkingDirectory "C:\LiveKit" `
        -RedirectStandardOutput (Join-Path $Root "livekit_current.out.log") `
        -RedirectStandardError (Join-Path $Root "livekit_current.err.log") `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds(30)
    Write-Host "Waiting for LiveKit :7880 " -NoNewline
    while ((Get-Date) -lt $deadline) {
        if (Test-NetConnection -ComputerName localhost -Port 7880 -InformationLevel Quiet) {
            Write-Host "ready" -ForegroundColor Green
            return
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }
    Write-Host " timeout" -ForegroundColor Red
    Get-Content (Join-Path $Root "livekit_current.err.log") -Tail 80 -ErrorAction SilentlyContinue
    throw "LiveKit failed to start"
}

function Start-Gemma {
    if (-not $RestartGemma -and (Test-Http "http://localhost:8003/v1/models" 3)) {
        Write-Host "Gemma 4 vLLM already ready on port 8003" -ForegroundColor Green
        return
    }

    if (-not (Test-WslDistro)) {
        throw "Cannot start Gemma 4 vLLM because WSL distro '$WslDistro' is unavailable."
    }

    Stop-Gemma

    $out = Join-Path $Root "gemma4_vllm.current.out.log"
    $err = Join-Path $Root "gemma4_vllm.current.err.log"
    $WslRoot = (& wsl.exe -d $WslDistro wslpath -u $Root).Trim()
    if (-not $WslRoot) {
        throw "Could not translate Echo root to a WSL path: $Root"
    }
    $GemmaScript = "$WslRoot/start_gemma4_e2b_vllm.sh"
    Remove-Item $out, $err -Force -ErrorAction SilentlyContinue

    Start-Process -FilePath "wsl.exe" `
        -ArgumentList @("-d", $WslDistro, "bash", $GemmaScript) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -WindowStyle Hidden | Out-Null

    if (-not (Wait-Http "Gemma 4 vLLM :8003" "http://localhost:8003/v1/models" $GemmaTimeoutSeconds)) {
        Write-Host ""
        Write-Host "Gemma did not become ready. Last log lines:" -ForegroundColor Yellow
        Get-Content $err -Tail 80 -ErrorAction SilentlyContinue
        Get-Content $out -Tail 40 -ErrorAction SilentlyContinue
        throw "Gemma 4 vLLM failed to start"
    }
}

function Start-Echo {
    Stop-EchoBackend

    $lock = Join-Path $env:USERPROFILE ".mem0\migrations_qdrant\.lock"
    if (Test-Path $lock) {
        Write-Host "Removing stale mem0 lock"
        Remove-Item -LiteralPath $lock -Force -ErrorAction SilentlyContinue
    }

    $out = Join-Path $Root "echo_current_session.out.log"
    $err = Join-Path $Root "echo_current_session.err.log"
    Remove-Item $out, $err -Force -ErrorAction SilentlyContinue

    Start-Process -FilePath $Python `
        -ArgumentList @("main.py") `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -WindowStyle Hidden

    if (-not (Wait-Http "Echo backend :8002" "http://localhost:8002/health" $EchoTimeoutSeconds)) {
        Write-Host ""
        Write-Host "Echo did not become ready. Last log lines:" -ForegroundColor Yellow
        Get-Content $err -Tail 120 -ErrorAction SilentlyContinue
        throw "Echo backend failed to start"
    }
}

function Start-Voice {
    Stop-VoiceAgent
    Start-Process -FilePath $Python `
        -ArgumentList @("voice_agent.py", "start", "--url", "ws://localhost:7880", "--api-key", "devkey", "--api-secret", "secret") `
        -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $Root "voice_agent_current.out.log") `
        -RedirectStandardError (Join-Path $Root "voice_agent_current.err.log") `
        -WindowStyle Hidden
    Write-Host "Voice agent launched" -ForegroundColor Green
}

function Get-CloudflaredPath {
    $candidates = @(
        "cloudflared",
        "C:\Windows\System32\cloudflared.exe",
        "C:\Program Files (x86)\cloudflared\cloudflared.exe",
        "$env:ProgramFiles\cloudflared\cloudflared.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe"
    )
    foreach ($candidate in $candidates) {
        try {
            if ($candidate -eq "cloudflared") {
                $found = Get-Command cloudflared -ErrorAction SilentlyContinue
                if ($found) { return $found.Source }
            } elseif (Test-Path $candidate) {
                return $candidate
            }
        } catch {
        }
    }
    return $null
}

function Stop-Tunnel {
    Write-Host "Stopping old cloudflared quick tunnels..."
    Get-Process cloudflared -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Start-Tunnel {
    if ($RestartTunnel) {
        Stop-Tunnel
    }

    $cloudflared = Get-CloudflaredPath
    if (-not $cloudflared) {
        throw "cloudflared not found. Install it with: winget install Cloudflare.cloudflared"
    }

    $out = Join-Path $Root "cloudflared_current.out.log"
    $err = Join-Path $Root "cloudflared_current.err.log"
    $urlFile = Join-Path $Root "cloudflared_current.url.txt"
    $metricsUrl = "http://127.0.0.1:20241/metrics"
    Remove-Item $out, $err, $urlFile -Force -ErrorAction SilentlyContinue

    Start-Process -FilePath $cloudflared `
        -ArgumentList @(
            "tunnel",
            "--url", "http://127.0.0.1:8002",
            "--edge-ip-version", "4",
            "--protocol", "http2",
            "--metrics", "127.0.0.1:20241"
        ) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $out `
        -RedirectStandardError $err `
        -WindowStyle Hidden | Out-Null

    $deadline = (Get-Date).AddSeconds(60)
    Write-Host "Waiting for Cloudflare tunnel " -NoNewline
    $url = $null
    while ((Get-Date) -lt $deadline) {
        $lines = @()
        if (Test-Path $out) { $lines += Get-Content $out -ErrorAction SilentlyContinue }
        if (Test-Path $err) { $lines += Get-Content $err -ErrorAction SilentlyContinue }
        foreach ($line in $lines) {
            if ($line -match "https://[a-z0-9-]+\.trycloudflare\.com") {
                $url = $Matches[0]
                break
            }
        }
        if (-not $url) {
            try {
                $metrics = Invoke-WebRequest -Uri $metricsUrl -UseBasicParsing -TimeoutSec 2
                if ($metrics.Content -match 'userHostname="(https://[a-z0-9-]+\.trycloudflare\.com)"') {
                    $url = $Matches[1]
                }
            } catch {
            }
        }
        if ($url) { break }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
    }

    if (-not $url) {
        Write-Host " timeout" -ForegroundColor Red
        Get-Content $err -Tail 80 -ErrorAction SilentlyContinue
        throw "Cloudflare tunnel did not produce a URL"
    }

    Set-Content -LiteralPath $urlFile -Value $url
    Write-Host " ready" -ForegroundColor Green
    Write-Host "Tunnel URL: $url" -ForegroundColor Green
}

Write-Host "Echo Services" -ForegroundColor Cyan
Write-Host "============="
Write-Host "Root: $Root"

if (-not $KeepQwen) {
    Write-Step "1. GPU cleanup"
    Stop-LegacyQwen
}

if (-not $SkipGemma) {
    Write-Step "2. Gemma 4 vLLM"
    Start-Gemma
}

if (-not $SkipLiveKit) {
    Write-Step "3. LiveKit"
    Start-LiveKit
}

if (-not $SkipEcho) {
    Write-Step "4. Echo backend"
    Start-Echo
}

if ($WithVoice -and -not $SkipVoiceAgent) {
    Write-Step "5. Voice agent"
    Start-Voice
}

if ($WithTunnel) {
    Write-Step "6. Secure tunnel"
    Start-Tunnel
}

Write-Step "Health"
$gemmaStatus = if (Test-Http "http://localhost:8003/v1/models" 3) { "ready" } else { "down" }
$liveKitStatus = if (Test-NetConnection -ComputerName localhost -Port 7880 -InformationLevel Quiet) { "ready" } else { "down" }
$echoStatus = if (Test-Http "http://localhost:8002/health" 3) { "ready" } else { "down" }
$tunnelUrl = Get-Content (Join-Path $Root "cloudflared_current.url.txt") -ErrorAction SilentlyContinue | Select-Object -First 1
Write-Host "Gemma 4 vLLM :8003  $gemmaStatus"
Write-Host "LiveKit      :7880  $liveKitStatus"
Write-Host "Echo backend :8002  $echoStatus"
if ($tunnelUrl) {
    Write-Host "Tunnel       :443   $tunnelUrl"
}
Write-Host ""
Write-Host "Mobile emulator base URL: http://10.0.2.2:8002"
Write-Host "Desktop base URL:         http://localhost:8002"
if ($tunnelUrl) {
    Write-Host "Phone tunnel base URL:    $tunnelUrl"
}
