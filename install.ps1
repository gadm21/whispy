# ThothCraft installer for Windows — turns this computer into a Thoth node.
#
# One-liner (once hosted):
#   irm https://get.thothcraft.com/install.ps1 | iex
# From a cloned repo:
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Local .\packages]
#
# Installs thothcraft-sdk + thothcraft-cli[sensors], then registers
# thothcraftd as a logon task so the node stays reachable via
# thothcraft.local("<this-pc>") and heartbeats to Brain once paired.

[CmdletBinding()]
param(
    # Install from a local checkout instead of PyPI (path containing packages/).
    [string]$Local = "",
    # Skip registering the background daemon task.
    [switch]$NoDaemon,
    # Skip sensor extras (opencv, pyserial, psutil).
    [switch]$NoSensors,
    # Skip enabling OpenSSH server.
    [switch]$NoSsh
)

$ErrorActionPreference = "Stop"

function Find-Python {
    foreach ($cmd in @("python", "py", "python3")) {
        $exe = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($exe) {
            try {
                $ver = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($ver -and [version]$ver -ge [version]"3.10") { return $cmd }
            } catch { }
        }
    }
    return $null
}

Write-Host "== ThothCraft installer (Windows) ==" -ForegroundColor Cyan

$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (check 'Add to PATH') and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python: $(& $py --version)"

$packages = if ($NoSensors) { @("thothcraft-sdk", "thothcraft-cli") } else { @("thothcraft-sdk", "thothcraft-cli[sensors]") }

if ($Local) {
    $sdk = Join-Path $Local "thothcraft-sdk"
    $cli = Join-Path $Local "thothcraft-cli"
    if (-not (Test-Path $sdk) -or -not (Test-Path $cli)) {
        Write-Host "-Local must point at the directory containing thothcraft-sdk/ and thothcraft-cli/" -ForegroundColor Red
        exit 1
    }
    $cliSpec = if ($NoSensors) { $cli } else { "$cli[sensors]" }
    Write-Host "Installing from local checkout: $Local"
    & $py -m pip install --upgrade -e $sdk -e $cliSpec
}
else {
    Write-Host "Installing from PyPI (falls back to the git repo if unpublished)..."
    & $py -m pip install --upgrade @packages 2>$null
    if ($LASTEXITCODE -ne 0) {
        $sdkUrl = "git+https://github.com/gadm21/whispy.git#subdirectory=packages/thothcraft-sdk"
        $cliUrl = "git+https://github.com/gadm21/whispy.git#subdirectory=packages/thothcraft-cli"
        $cliSpec = if ($NoSensors) { $cliUrl } else { "$cliUrl[sensors]" }
        & $py -m pip install --upgrade $sdkUrl $cliSpec
    }
}
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed" -ForegroundColor Red; exit 1 }

function Enable-SshServer {
    Write-Host "Ensuring OpenSSH Server is configured..." -ForegroundColor Cyan
    try {
        $sshd = Get-Service -Name "sshd" -ErrorAction SilentlyContinue
        if (-not $sshd) {
            Write-Host "Installing OpenSSH Server Windows capability..."
            Add-WindowsCapability -Online -Name "OpenSSH.Server~~~~0.0.1.0" -ErrorAction SilentlyContinue | Out-Null
            $sshd = Get-Service -Name "sshd" -ErrorAction SilentlyContinue
        }
        if ($sshd) {
            Set-Service -Name "sshd" -StartupType 'Automatic' -ErrorAction SilentlyContinue
            Start-Service -Name "sshd" -ErrorAction SilentlyContinue
            if (Get-Command "New-NetFirewallRule" -ErrorAction SilentlyContinue) {
                if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
                    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -ErrorAction SilentlyContinue | Out-Null
                }
            }
            Write-Host "✓ OpenSSH Server (sshd) enabled and running" -ForegroundColor Green
        } else {
            Write-Host "Note: OpenSSH Server could not be enabled automatically (run PowerShell as Administrator to enable sshd)." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Note: Could not enable OpenSSH Server ($($_.Exception.Message))." -ForegroundColor Yellow
    }
}

$thothcraftd = Get-Command thothcraftd -ErrorAction SilentlyContinue
if (-not $thothcraftd) {
    $userScripts = & $py -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"
    $candidate = Join-Path $userScripts "thothcraftd.exe"
    if (Test-Path $candidate) { $thothcraftd = $candidate }
}
if (-not $thothcraftd) {
    Write-Host "thothcraftd entry point not found on PATH — check pip output above." -ForegroundColor Red
    exit 1
}
$thothcraftdPath = if ($thothcraftd -is [System.Management.Automation.CommandInfo]) { $thothcraftd.Source } else { [string]$thothcraftd }
Write-Host "thothcraftd: $thothcraftdPath"

if (-not $NoDaemon) {
    $taskName = "ThothcraftDaemon"
    $action = New-ScheduledTaskAction -Execute $thothcraftdPath
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Description "ThothCraft device daemon (local sensor API + Brain heartbeat)" -Force | Out-Null
    Start-ScheduledTask -TaskName $taskName
    Write-Host "✓ thothcraftd registered as logon task '$taskName' and started" -ForegroundColor Green
}

if (-not $NoSsh) {
    Enable-SshServer
}

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Cyan
Write-Host "  thothcraft login            # link your thothHUB account"
Write-Host "  thothcraft pair             # claim this computer as a device"
Write-Host "  thothcraft device init      # probe sensors + pair in one step"
Write-Host ""
Write-Host "Local SDK check (no pairing needed):"
Write-Host "  python -c `"import thothcraft; print(thothcraft.local('127.0.0.1').sensors())`""
