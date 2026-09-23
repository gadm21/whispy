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
    $origPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py -m pip install --upgrade -e $sdk -e $cliSpec
    $code = $LASTEXITCODE
    $ErrorActionPreference = $origPref
    if ($code -ne 0) { Write-Host "pip install failed" -ForegroundColor Red; exit 1 }
}
else {
    Write-Host "Installing from PyPI (falls back to the git repo if unpublished)..."
    $origPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py -m pip install --upgrade $packages
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $sdkUrl = "git+https://github.com/gadm21/whispy.git#subdirectory=packages/thothcraft-sdk"
        $cliUrl = "git+https://github.com/gadm21/whispy.git#subdirectory=packages/thothcraft-cli"
        $cliSpec = if ($NoSensors) { $cliUrl } else { "$cliUrl[sensors]" }
        & $py -m pip install --upgrade $sdkUrl $cliSpec
        $code = $LASTEXITCODE
    }
    $ErrorActionPreference = $origPref
    if ($code -ne 0) { Write-Host "pip install failed" -ForegroundColor Red; exit 1 }
}

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

$userScripts = & $py -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"
$sysScripts = & $py -c "import sysconfig; print(sysconfig.get_path('scripts'))"

# Ensure user scripts are in Windows User PATH
try {
    $currentPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userScripts -and ($currentPath -notlike "*$userScripts*")) {
        $newPath = if ($currentPath) { "$userScripts;$currentPath" } else { $userScripts }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        $env:PATH = "$userScripts;$env:PATH"
        Write-Host "✓ Added Python Scripts to Windows User PATH: $userScripts" -ForegroundColor Green
    }
} catch { }

# Ensure Git Bash profiles have the PATH export
try {
    $posixScripts = $userScripts.Replace('\', '/').Replace('C:', '/c')
    $bashExport = "`n# Added by ThothCraft`nexport PATH=`"`$PATH:$posixScripts`"`n"
    foreach ($profileName in @(".bashrc", ".bash_profile")) {
        $pPath = Join-Path $HOME $profileName
        $existing = if (Test-Path $pPath) { Get-Content $pPath -Raw } else { "" }
        if ($existing -notlike "*$posixScripts*") {
            Add-Content -Path $pPath -Value $bashExport -Encoding utf8
            Write-Host "✓ Configured Git Bash profile: $pPath" -ForegroundColor Green
        }
    }
} catch { }

$thothcraft = Get-Command thothcraft -ErrorAction SilentlyContinue
if (-not $thothcraft) {
    foreach ($cand in @((Join-Path $userScripts "thothcraft.exe"), (Join-Path $sysScripts "thothcraft.exe"))) {
        if (Test-Path $cand) { $thothcraft = $cand; break }
    }
}

# Copy thothcraft.exe to WindowsApps for instant PATH availability across all active terminals
$winApps = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
if (Test-Path $winApps) {
    $srcExe = if ($thothcraft -is [System.Management.Automation.CommandInfo]) { $thothcraft.Source } else { [string]$thothcraft }
    if ($srcExe -and (Test-Path $srcExe)) {
        try {
            Copy-Item $srcExe -Destination (Join-Path $winApps "thothcraft.exe") -Force -ErrorAction SilentlyContinue
            Write-Host "✓ Copied thothcraft to $winApps (immediately available in all terminals)" -ForegroundColor Green
        } catch { }
    }
}

if (-not $thothcraft) {
    Write-Host "thothcraft entry point not found on PATH — check pip output above." -ForegroundColor Red
    exit 1
}
$thothcraftPath = if ($thothcraft -is [System.Management.Automation.CommandInfo]) { $thothcraft.Source } else { [string]$thothcraft }
Write-Host "thothcraft: $thothcraftPath"

if (-not $NoDaemon) {
    $taskName = "Thothcraft"
    $action = New-ScheduledTaskAction -Execute $thothcraftPath -Argument "daemon"
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    $registered = $false
    try {
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
            -Settings $settings -Description "ThothCraft device daemon (local sensor API + Brain heartbeat)" -Force -ErrorAction Stop | Out-Null
        Start-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        Write-Host "✓ thothcraft daemon registered as logon task '$taskName' and started" -ForegroundColor Green
        $registered = $true
    } catch {
        try {
            $startupDir = [System.IO.Path]::Combine($env:APPDATA, "Microsoft\Windows\Start Menu\Programs\Startup")
            if (Test-Path $startupDir) {
                $cmdFile = Join-Path $startupDir "thothcraft.cmd"
                "@start `"`" `"$thothcraftPath`" daemon" | Out-File -FilePath $cmdFile -Encoding ascii
                Write-Host "✓ thothcraft daemon added to Startup folder ($cmdFile)" -ForegroundColor Green
                $registered = $true
            }
        } catch { }
        if (-not $registered) {
            Write-Host "Note: To register scheduled logon task, run PowerShell as Administrator. You can run 'thothcraft daemon' directly." -ForegroundColor Yellow
        }
    }
}

if (-not $NoSsh) {
    Enable-SshServer
}

$hostName = try { & $py -c "import sys; sys.path.insert(0, r'packages/thothcraft-cli'); from thothcraft_cli.daemon import _device_uuid, _device_hostname; print(_device_hostname(_device_uuid()))" 2>$null } catch { "thoth-node.local" }
if (-not $hostName) { $hostName = "thoth-node.local" }

Write-Host ""
Write-Host "Supported Terminals:" -ForegroundColor Cyan
Write-Host "  - Windows PowerShell 5.1 / PowerShell 7+"
Write-Host "  - Windows Terminal"
Write-Host "  - Git Bash (C:\Program Files\Git\bin\bash.exe)"
Write-Host "  - Command Prompt (cmd.exe)"
Write-Host ""
Write-Host "Done. Next steps in your terminal (PowerShell, CMD, or Git Bash):" -ForegroundColor Cyan
Write-Host "  thothcraft login            # link your thothHUB account"
Write-Host "  thothcraft pair             # claim this computer as a device"
Write-Host "  thothcraft device init      # probe sensors + pair in one step"
Write-Host ""
Write-Host "Local Dashboard Access:" -ForegroundColor Cyan
Write-Host "  http://$hostName:5000"
Write-Host "  http://localhost:5000"
Write-Host ""
Write-Host "Local SDK check (no pairing needed):"
Write-Host "  python -c `"import thothcraft; print(thothcraft.local('$hostName').sensors())`""
