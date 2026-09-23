#!/usr/bin/env bash
# ThothCraft installer for Linux/macOS — turns this computer into a Thoth node.
#
# One-liner (once hosted):
#   curl -fsSL https://get.thothcraft.com/install.sh | bash
# From a cloned repo:
#   ./install.sh [--local ./packages] [--no-daemon] [--no-sensors]
#
# Installs thothcraft-sdk + thothcraft-cli[sensors], then registers
# thothcraftd as a user service (systemd) or LaunchAgent (macOS) so the node
# stays reachable via thothcraft.local("<host>") and heartbeats to Brain.

set -euo pipefail

LOCAL=""
NO_DAEMON=0
NO_SENSORS=0
NO_SSH=0
while [ $# -gt 0 ]; do
    case "$1" in
        --local) LOCAL="$2"; shift 2 ;;
        --no-daemon) NO_DAEMON=1; shift ;;
        --no-sensors) NO_SENSORS=1; shift ;;
        --no-ssh) NO_SSH=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

echo "== ThothCraft installer ($(uname -s)) =="

PY=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PY="$cmd"; break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.10+ not found — install it and re-run." >&2
    exit 1
fi
echo "Using Python: $($PY --version)"

if [ -n "$LOCAL" ]; then
    EXTRA=""
    [ "$NO_SENSORS" -eq 0 ] && EXTRA="[sensors]"
    echo "Installing from local checkout: $LOCAL"
    "$PY" -m pip install --user --upgrade -e "$LOCAL/thothcraft-sdk" -e "$LOCAL/thothcraft-cli$EXTRA"
else
    if [ "$NO_SENSORS" -eq 0 ]; then
        PKGS="thothcraft-cli[sensors]"
    else
        PKGS="thothcraft-cli"
    fi
    if ! "$PY" -m pip install --user --upgrade thothcraft-sdk "$PKGS" 2>/dev/null; then
        REPO="git+https://github.com/gadm21/whispy.git"
        "$PY" -m pip install --user --upgrade \
            "$REPO#subdirectory=packages/thothcraft-sdk" \
            "$REPO&subdirectory=packages/thothcraft-cli$([ "$NO_SENSORS" -eq 0 ] && echo '[sensors]' || true)" \
            2>/dev/null || {
                # pip can't combine extras with direct refs on older versions
                "$PY" -m pip install --user --upgrade \
                    "$REPO#subdirectory=packages/thothcraft-sdk" \
                    "$REPO#subdirectory=packages/thothcraft-cli"
                [ "$NO_SENSORS" -eq 0 ] && "$PY" -m pip install --user --upgrade opencv-python pyserial psutil
            }
    fi
fi

THOTHCRAFTD="$(command -v thothcraftd || true)"
if [ -z "$THOTHCRAFTD" ]; then
    THOTHCRAFTD="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')/thothcraftd"
fi
[ -x "$THOTHCRAFTD" ] || THOTHCRAFTD="$HOME/.local/bin/thothcraftd"
echo "thothcraftd: $THOTHCRAFTD"

if [ "$NO_DAEMON" -eq 0 ]; then
    if [ "$(uname -s)" = "Linux" ] && command -v systemctl >/dev/null 2>&1; then
        mkdir -p "$HOME/.config/systemd/user"
        cat > "$HOME/.config/systemd/user/thothcraftd.service" <<EOF
[Unit]
Description=ThothCraft device daemon (local sensor API + Brain heartbeat)
After=network-online.target

[Service]
ExecStart=$THOTHCRAFTD
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now thothcraftd
        echo "✓ thothcraftd enabled as a systemd user service"
    elif [ "$(uname -s)" = "Darwin" ]; then
        PLIST="$HOME/Library/LaunchAgents/com.thothcraft.daemon.plist"
        mkdir -p "$(dirname "$PLIST")"
        cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.thothcraft.daemon</string>
  <key>ProgramArguments</key><array><string>$THOTHCRAFTD</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
EOF
        launchctl unload "$PLIST" 2>/dev/null || true
        launchctl load "$PLIST"
        echo "✓ thothcraftd loaded as a LaunchAgent"
    else
        echo "No supported service manager — run 'thothcraftd' manually or via your init system."
    fi
fi

if [ "$NO_SSH" -eq 0 ]; then
    echo "Ensuring SSH server is enabled..."
    if [ "$(uname -s)" = "Linux" ]; then
        if command -v systemctl >/dev/null 2>&1; then
            if ! systemctl is-active --quiet ssh && ! systemctl is-active --quiet sshd; then
                if command -v apt-get >/dev/null 2>&1; then
                    sudo apt-get update -y && sudo apt-get install -y openssh-server || true
                elif command -v yum >/dev/null 2>&1; then
                    sudo yum install -y openssh-server || true
                elif command -v pacman >/dev/null 2>&1; then
                    sudo pacman -S --noconfirm openssh || true
                fi
                sudo systemctl enable --now ssh 2>/dev/null || sudo systemctl enable --now sshd 2>/dev/null || true
            fi
            if systemctl is-active --quiet ssh || systemctl is-active --quiet sshd; then
                echo "✓ SSH server is enabled and active"
            else
                echo "Note: SSH server could not be started automatically. Run 'sudo systemctl enable --now ssh'."
            fi
        fi
    elif [ "$(uname -s)" = "Darwin" ]; then
        sudo systemsetup -setremotelogin on 2>/dev/null || true
        echo "✓ Remote Login (SSH) requested"
    fi
fi

echo ""
echo "Done. Next steps:"
echo "  thothcraft login            # link your thothHUB account"
echo "  thothcraft pair             # claim this computer as a device"
echo "  thothcraft device init      # probe sensors + pair in one step"
echo ""
echo "Local SDK check (no pairing needed):"
echo "  python3 -c \"import thothcraft; print(thothcraft.local('127.0.0.1').sensors())\""
