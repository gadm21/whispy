"""Mosquitto broker configuration generator for global Whispy deployments.

Generates a production-ready Mosquitto config with TLS, authentication,
ACLs, and persistence for a cloud-hosted broker that receives MQTT
messages from Whispy devices around the world.

Usage::

    whispy backend init --domain mqtt.example.com
    # Generates mosquitto.conf, password file, ACL file, and TLS instructions
"""
from __future__ import annotations
import os, secrets, string
from pathlib import Path

MOSQUITTO_CONF = """\
# =========================================================================
# Whispy Global MQTT Broker — Mosquitto Configuration
# =========================================================================
# Deploy on a cloud VPS (e.g. DigitalOcean, AWS EC2, Hetzner)
# Install: sudo apt install mosquitto mosquitto-clients

# ── Listener (TLS on 8883, optional plain on 1883 for LAN) ─────────────
listener 8883
protocol mqtt
certfile /etc/mosquitto/certs/fullchain.pem
keyfile /etc/mosquitto/certs/privkey.pem
tls_version tlsv1.2

# Optional: plain MQTT on localhost only (for backend co-located on same host)
listener 1883 127.0.0.1
protocol mqtt

# ── Authentication ──────────────────────────────────────────────────────
allow_anonymous false
password_file {config_dir}/passwd

# ── ACL (access control) ───────────────────────────────────────────────
acl_file {config_dir}/acl

# ── Persistence ─────────────────────────────────────────────────────────
persistence true
persistence_location /var/lib/mosquitto/
autosave_interval 300

# ── Logging ─────────────────────────────────────────────────────────────
log_dest file /var/log/mosquitto/mosquitto.log
log_type warning
log_type error
log_type notice
connection_messages true

# ── Limits ──────────────────────────────────────────────────────────────
max_connections 200
max_inflight_messages 20
max_queued_messages 1000
message_size_limit 262144
"""

ACL_TEMPLATE = """\
# =========================================================================
# Whispy MQTT ACL — Access Control List
# =========================================================================

# Backend user — full access to everything
user whispy_backend
topic readwrite #

# Device users — each device can only write to its own topics
# Pattern: whispy/<node_id>/# where %u is the username (= node_id)
pattern readwrite whispy/%u/#

# All devices can read commands from backend
pattern read whispy/broadcast/#

# Home Assistant bridge (if local HA is also connected)
user homeassistant
topic readwrite homeassistant/#
topic read whispy/#
"""


def _random_password(length: int = 24) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def generate_broker_config(
    config_dir: str = "./whispy_broker",
    domain: str = "localhost",
    backend_password: str | None = None,
    device_passwords: dict[str, str] | None = None,
) -> dict[str, str]:
    """Generate Mosquitto config, password, and ACL files.

    Parameters
    ----------
    config_dir : directory to write config files to.
    domain : broker domain/IP for TLS certificate.
    backend_password : password for the backend MQTT user. Auto-generated if None.
    device_passwords : dict of {node_id: password}. If None, generates example entries.

    Returns
    -------
    dict with keys: config_path, passwd_path, acl_path, backend_password,
                    device_passwords, tls_instructions
    """
    out = Path(config_dir)
    out.mkdir(parents=True, exist_ok=True)

    backend_pw = backend_password or _random_password()

    # default example devices
    if device_passwords is None:
        device_passwords = {
            "office-node-01": _random_password(),
            "lab-toronto-01": _random_password(),
            "home-sensor-01": _random_password(),
        }

    # write mosquitto.conf
    conf_path = out / "mosquitto.conf"
    with open(conf_path, "w") as f:
        f.write(MOSQUITTO_CONF.format(config_dir=str(out.resolve())))

    # write ACL
    acl_path = out / "acl"
    with open(acl_path, "w") as f:
        f.write(ACL_TEMPLATE)

    # write password file (plaintext — must run mosquitto_passwd -U to hash)
    passwd_path = out / "passwd"
    with open(passwd_path, "w") as f:
        f.write(f"whispy_backend:{backend_pw}\n")
        for node_id, pw in device_passwords.items():
            f.write(f"{node_id}:{pw}\n")

    # write credentials summary
    creds_path = out / "credentials.json"
    import json
    creds = {
        "backend": {"username": "whispy_backend", "password": backend_pw},
        "devices": {nid: {"username": nid, "password": pw}
                    for nid, pw in device_passwords.items()},
        "broker": {"domain": domain, "port_tls": 8883, "port_plain": 1883},
    }
    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)

    tls_instructions = f"""\
TLS Setup Instructions
======================
1. Install certbot:
   sudo apt install certbot

2. Obtain certificate for {domain}:
   sudo certbot certonly --standalone -d {domain}

3. Copy/link certs:
   sudo ln -s /etc/letsencrypt/live/{domain}/fullchain.pem /etc/mosquitto/certs/fullchain.pem
   sudo ln -s /etc/letsencrypt/live/{domain}/privkey.pem /etc/mosquitto/certs/privkey.pem

4. Hash the password file:
   mosquitto_passwd -U {passwd_path.resolve()}

5. Copy config:
   sudo cp {conf_path.resolve()} /etc/mosquitto/conf.d/whispy.conf
   sudo cp {acl_path.resolve()} /etc/mosquitto/acl
   sudo cp {passwd_path.resolve()} /etc/mosquitto/passwd

6. Restart Mosquitto:
   sudo systemctl restart mosquitto

7. Test:
   mosquitto_pub -h {domain} -p 8883 --capath /etc/ssl/certs/ \\
       -u whispy_backend -P {backend_pw} \\
       -t "whispy/test" -m "hello"
"""

    tls_path = out / "TLS_SETUP.md"
    with open(tls_path, "w") as f:
        f.write(tls_instructions)

    print(f"[broker] Config written to {out.resolve()}/")
    print(f"  mosquitto.conf  — main config")
    print(f"  passwd          — user/password file (hash with mosquitto_passwd -U)")
    print(f"  acl             — access control list")
    print(f"  credentials.json — all credentials (KEEP SECRET)")
    print(f"  TLS_SETUP.md    — certificate setup instructions")

    return {
        "config_path": str(conf_path),
        "passwd_path": str(passwd_path),
        "acl_path": str(acl_path),
        "creds_path": str(creds_path),
        "backend_password": backend_pw,
        "device_passwords": device_passwords,
        "tls_instructions": tls_instructions,
    }


def add_device_credentials(
    config_dir: str = "./whispy_broker",
    node_id: str = "",
    password: str | None = None,
) -> dict[str, str]:
    """Add a new device to the password file."""
    pw = password or _random_password()
    passwd_path = Path(config_dir) / "passwd"
    with open(passwd_path, "a") as f:
        f.write(f"{node_id}:{pw}\n")

    # update credentials.json
    import json
    creds_path = Path(config_dir) / "credentials.json"
    if creds_path.exists():
        with open(creds_path) as f:
            creds = json.load(f)
        creds["devices"][node_id] = {"username": node_id, "password": pw}
        with open(creds_path, "w") as f:
            json.dump(creds, f, indent=2)

    print(f"[broker] Added device: {node_id}")
    print(f"  Remember to re-hash: mosquitto_passwd -U {passwd_path.resolve()}")
    print(f"  And restart: sudo systemctl restart mosquitto")
    return {"node_id": node_id, "username": node_id, "password": pw}
