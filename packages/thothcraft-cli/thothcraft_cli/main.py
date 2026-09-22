"""thothcraft — system CLI for ThothCraft.

Controls the thothcraftd device daemon and talks to Brain through
thothcraft-sdk. A Thoth device is any authenticated node implementing
the device protocol.
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import click

from . import __version__, probe
from .daemon import CONFIG_DIR, DEVICE_FILE


def _client():
    from thothcraft.client import Client
    from thothcraft.errors import AuthError
    try:
        return Client.login()
    except AuthError:
        raise click.ClickException("Not logged in — run `thothcraft login`")


# ── top-level group ───────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="thothcraft")
def main():
    """ThothCraft CLI — manage devices, data and this computer as a Thoth device."""


# ── auth ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--username", "-u", prompt=True)
@click.option("--base-url", default=None, help="Brain API URL")
def login(username, base_url):
    """Authenticate with Brain and store a user access token."""
    from thothcraft.client import Client, DEFAULT_BASE_URL
    password = getpass.getpass("Password: ")
    url = base_url or Client.load_base_url() or DEFAULT_BASE_URL
    client = Client(url)
    token = client._login(username, password)
    Client.store_token(token, url)
    click.echo(f"✓ Logged in as {username} ({url})")


@main.command()
def logout():
    """Remove stored credentials."""
    from thothcraft.client import Client
    Client.clear_token()
    click.echo("✓ Logged out")


@main.command()
def whoami():
    """Show the authenticated account and plan."""
    client = _client()
    ent = client.entitlements()
    click.echo(json.dumps(ent, indent=2))


# ── devices ───────────────────────────────────────────────────────────────────

@main.command("devices")
@click.option("--json", "as_json", is_flag=True)
def devices_list(as_json):
    """List your Thoth devices."""
    client = _client()
    devices = client.devices()
    if as_json:
        click.echo(json.dumps([d.info for d in devices], indent=2))
        return
    if not devices:
        click.echo("No devices. Pair one with `thothcraft pair`.")
        return
    for d in devices:
        state = "●" if d.online else "○"
        click.echo(f"{state} {d.name}  ({d.uuid})")


# ── pairing ───────────────────────────────────────────────────────────────────

@main.command()
def pair():
    """Pair this computer (or an attached device) with your account.

    Starts a pairing session, prints a code to claim in thothHUB,
    and stores the resulting device credential for thothcraftd.
    """
    from thothcraft.client import Client
    from .daemon import _device_uuid

    client = _client()
    device_uuid = _device_uuid()
    name = os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "computer")

    resp = client._http.post_json("/api/device/pairing/start", body={
        "device_id": device_uuid,
        "device_name": name,
        "device_type": "computer",
    })
    code = resp.get("code")
    secret = resp.get("pairing_secret")
    click.echo(f"\nPairing code:\n\n        THOTH-{code}\n")
    click.echo("Open: https://hub.thothcraft.com/pair\n")
    click.echo("Waiting...")

    deadline = time.time() + 600
    while time.time() < deadline:
        # Poll pairing status with the pairing secret header
        req = urllib.request.Request(
            f"{client._http.base_url}/api/device/pairing/status",
            headers={"X-Pairing-Secret": secret, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                payload = json.loads(r.read().decode())
        except Exception:
            payload = {}
        if payload.get("status") == "paired":
            token = payload.get("access_token")
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            DEVICE_FILE.write_text(json.dumps({
                "device_uuid": device_uuid,
                "device_token": token,
                "device_name": payload.get("device_name", name),
            }))
            click.echo(f"\n✓ Paired to {payload.get('user', {}).get('email', 'account')}")
            click.echo(f"✓ Device registered as \"{payload.get('device_name', name)}\"")
            return
        time.sleep(2)
    raise click.ClickException("Pairing timed out — code expired")


@main.command()
@click.argument("device_id")
def unpair(device_id):
    """Detach a device from your account."""
    client = _client()
    client._http._request("DELETE", f"/api/device/{device_id}")
    click.echo(f"✓ Device {device_id} detached")


# ── status / doctor ───────────────────────────────────────────────────────────

@main.command()
def status():
    """Show daemon and account status."""
    client = _client()
    ent = client.entitlements()
    usage = client.storage_usage()
    click.echo(f"Plan:      {ent.get('plan')}")
    click.echo(f"Devices:   {ent.get('entitlements', {}).get('device_limit')} max")
    if usage.get("quota_bytes"):
        gb = usage["used_bytes"] / 1024**3
        qg = usage["quota_bytes"] / 1024**3
        click.echo(f"Storage:   {gb:.1f} / {qg:.0f} GB")
    else:
        click.echo(f"Storage:   {usage.get('used_bytes', 0) / 1024**2:.0f} MB "
                   f"(retention: {usage.get('minute_retention')} minutes)")
    if DEVICE_FILE.exists():
        click.echo(f"This node: {json.loads(DEVICE_FILE.read_text()).get('device_name', 'paired')}")
    else:
        click.echo("This node: not paired — run `thothcraft pair`")


@main.command()
def doctor():
    """Diagnose connectivity, credentials and sensor availability."""
    from thothcraft.client import Client
    click.echo("thothcraft doctor\n")
    # credentials
    token = Client.load_token()
    click.echo(f"  User token:      {'✓ present' if token else '✗ missing — thothcraft login'}")
    dev = json.loads(DEVICE_FILE.read_text()) if DEVICE_FILE.exists() else {}
    click.echo(f"  Device cred:     {'✓ paired' if dev.get('device_token') else '✗ not paired'}")
    # connectivity
    url = Client.load_base_url()
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=8) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    click.echo(f"  Brain ({url}): {'✓ reachable' if ok else '✗ unreachable'}")
    # sensors
    click.echo("\n  Sensors:")
    for name, (avail, detail) in probe.scan().items():
        click.echo(f"    {'✓' if avail else '✗'} {name:<14} {detail}")


# ── sensors ───────────────────────────────────────────────────────────────────

@main.group(invoke_without_command=True)
@click.pass_context
def sensors(ctx):
    """Sensor discovery and listing."""
    if ctx.invoked_subcommand is None:
        for name, (avail, detail) in probe.scan().items():
            click.echo(f"{'✓' if avail else '✗'} {name:<14} {detail}")


@sensors.command("list")
def sensors_list():
    """List detected sensors."""
    for name, (avail, detail) in probe.scan().items():
        click.echo(f"{'✓' if avail else '✗'} {name:<14} {detail}")


@sensors.command("drivers")
def sensors_drivers():
    """List installed sensor-driver plugins (entry points)."""
    try:
        from thothcraft.sensors.base import installed_drivers
    except ImportError:
        raise click.ClickException("thothcraft SDK not installed")
    drivers = installed_drivers()
    if not drivers:
        click.echo("No sensor drivers installed.")
        return
    for name, cls in drivers.items():
        try:
            meta = cls().metadata()
            click.echo(f"{name:<20} {meta.version:<10} {','.join(meta.modalities)}")
        except Exception as exc:
            click.echo(f"{name:<20} (metadata failed: {exc})")


@sensors.command("test")
@click.argument("driver")
def sensors_test(driver):
    """Run the conformance suite against an installed driver."""
    try:
        from thothcraft.sensors.base import check_driver, installed_drivers
    except ImportError:
        raise click.ClickException("thothcraft SDK not installed")
    drivers = installed_drivers()
    if driver not in drivers:
        raise click.ClickException(
            f"unknown driver '{driver}'. Installed: {', '.join(drivers) or 'none'}")
    report = check_driver(drivers[driver]())
    for check in report["checks"]:
        mark = "✓" if check["ok"] else "✗"
        click.echo(f"{mark} {check['name']:<12} {check.get('detail', '')}")
    click.echo("PASSED" if report["passed"] else "FAILED")
    if not report["passed"]:
        raise SystemExit(1)


_DRIVER_TEMPLATE = '''"""ThothCraft sensor driver: {name}."""
from thothcraft.sensors.base import (
    HealthReport, SensorDriver, SensorFrame, SensorMeta,
)


class {cls}(SensorDriver):
    def metadata(self):
        return SensorMeta(name="{name}", modalities=("{modality}",))

    def discover(self):
        return []  # TODO: probe hardware, return [{{"id": ...}}]

    def open(self, config=None):
        pass  # TODO: acquire hardware

    def stream(self):
        return
        yield  # TODO: yield SensorFrame.now("{modality}", data)

    def close(self):
        pass  # TODO: release hardware
'''

_PYPROJECT_TEMPLATE = '''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "thothcraft-sensor-{name}"
version = "0.1.0"
description = "ThothCraft sensor driver for {name}"
dependencies = ["thothcraft-sdk>=0.1.0"]

[project.entry-points."thothcraft.sensors"]
{name} = "thothcraft_sensor_{name}:{cls}"
'''


@sensors.command("new")
@click.argument("name")
@click.option("--modality", default="custom",
              help="Sensor modality (radar, csi, camera, env, ...)")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path),
              default=".")
def sensors_new(name, modality, out_dir):
    """Scaffold a new sensor-driver package."""
    safe = name.lower().replace("-", "_")
    cls = "".join(part.title() for part in safe.split("_")) + "Driver"
    pkg = Path(out_dir) / f"thothcraft-sensor-{safe}"
    (pkg / f"thothcraft_sensor_{safe}").mkdir(parents=True, exist_ok=True)
    (pkg / "pyproject.toml").write_text(
        _PYPROJECT_TEMPLATE.format(name=safe, cls=cls), encoding="utf-8")
    (pkg / f"thothcraft_sensor_{safe}" / "__init__.py").write_text(
        _DRIVER_TEMPLATE.format(name=safe, cls=cls, modality=modality),
        encoding="utf-8")
    click.echo(f"✓ scaffolded {pkg}")
    click.echo(f"  install with: pip install -e {pkg}")


# ── data ──────────────────────────────────────────────────────────────────────

@main.group(invoke_without_command=True)
@click.pass_context
def data(ctx):
    """Captured data: list minutes, sync to cloud."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(data_list)


@data.command("list")
@click.argument("device_id", required=False)
def data_list(device_id):
    """List captured minutes for a device."""
    client = _client()
    devices = client.devices()
    if device_id:
        devices = [d for d in devices if d.uuid == device_id or d.name == device_id]
    for d in devices:
        click.echo(f"{d.name}:")
        for m in sorted(d.minutes(), reverse=True)[:50]:
            click.echo(f"  {m}")


@data.command("sync")
def data_sync():
    """Request cloud sync for pending device files."""
    client = _client()
    for d in client.devices():
        try:
            client._http.post_json(f"/api/device/{d.uuid}/sync-files", body={})
            click.echo(f"✓ sync requested: {d.name}")
        except Exception as e:
            click.echo(f"✗ {d.name}: {e}")


# ── models / predictions ─────────────────────────────────────────────────────

@main.group(invoke_without_command=True)
@click.pass_context
def models(ctx):
    """Upload, deploy and manage models."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(models_list)


@models.command('list')
def models_list():
    """List your trained models."""
    client = _client()
    for model in client.models():
        click.echo(json.dumps(model.info))


@models.command('upload')
@click.argument('path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--name', required=True)
@click.option('--classes', required=True, help='Comma-separated class names')
@click.option('--input-spec', required=True, type=click.Path(exists=True, path_type=Path), help='JSON input specification file')
def models_upload(path, name, classes, input_spec):
    try:
        model = _client().upload_model(path, name=name,
            classes=[c.strip() for c in classes.split(',') if c.strip()],
            input_spec=json.loads(input_spec.read_text(encoding='utf-8')))
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(model.info, indent=2))


@models.command('deploy')
@click.argument('model_id', type=int)
@click.argument('device_id')
@click.option('--wait/--no-wait', default=True)
@click.option('--timeout', type=click.FloatRange(min=0), default=180.0)
def models_deploy(model_id, device_id, wait, timeout):
    deployment = _client().deploy_model(model_id, device_id)
    if wait:
        try:
            deployment.wait(timeout)
        except TimeoutError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(deployment.info, indent=2))


@models.command('deployments')
def models_deployments():
    for deployment in _client().deployments():
        click.echo(json.dumps(deployment.info))


@models.command('cancel')
@click.argument('deployment_id')
def models_cancel(deployment_id):
    click.echo(json.dumps(_client().cancel_deployment(deployment_id)))


@models.command('registry')
@click.option('--sensor', default=None, help='Filter by sensor (radar, csi, camera, fusion)')
@click.option('--task', default=None, help='Filter by task (occupancy, har, localization)')
def models_registry(sensor, task):
    """Browse the public processor catalog."""
    for model in _client().registry(sensor=sensor, task=task):
        info = model.info
        click.echo(
            f"{info.get('registry_name') or info.get('name'):<40} "
            f"{info.get('processor_type') or '':<12} "
            f"{info.get('sensor') or '-':<8} {info.get('task') or '-':<14} "
            f"{info.get('visibility') or 'private'}")


@models.command('install')
@click.argument('registry_name')
@click.argument('device_id')
@click.option('--wait/--no-wait', default=True)
@click.option('--timeout', type=click.FloatRange(min=0), default=180.0)
def models_install(registry_name, device_id, wait, timeout):
    """Deploy a registry model by name: thothcraft models install thothcraft/radar-occupancy-v2 <device>."""
    client = _client()
    device = client.device(device_id)
    try:
        deployment = device.deploy(registry_name, wait=wait, timeout=timeout)
    except TimeoutError as exc:
        raise click.ClickException(str(exc)) from exc
    info = deployment.info if hasattr(deployment, 'info') else deployment
    click.echo(json.dumps(info, indent=2))


@models.command('rule')
@click.argument('name')
@click.option('--when', required=True, help='Rule expression, e.g. "snr_mean > snr_threshold"')
@click.option('--label', 'rule_label', required=True, help='Label when the rule fires')
@click.option('--else', 'else_label', default='unknown')
@click.option('--param', multiple=True, help='key=value tunable params')
@click.option('--sensor', default=None)
@click.option('--task', default=None)
@click.option('--registry-name', default=None)
def models_rule(name, when, rule_label, else_label, param, sensor, task, registry_name):
    """Create a config-only rule processor (no artifact)."""
    params = {}
    for p in param:
        key, _, value = p.partition('=')
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    model = _client().create_rule_model(
        name, rules=[{"when": when, "label": rule_label}],
        else_label=else_label, params=params,
        sensor=sensor, task=task, registry_name=registry_name)
    click.echo(json.dumps(model.info, indent=2))


@models.command('publish')
@click.argument('model_id', type=int)
@click.option('--visibility', default='community',
              type=click.Choice(['community', 'official', 'private']))
@click.option('--registry-name', default=None)
def models_publish(model_id, visibility, registry_name):
    """Publish a model to the community registry."""
    model = _client().publish_model(
        model_id, visibility=visibility, registry_name=registry_name)
    click.echo(json.dumps(model.info, indent=2))


# ── spaces ────────────────────────────────────────────────────────────────────

@main.group(invoke_without_command=True)
@click.pass_context
def spaces(ctx):
    """Spatial context: named areas, zones, live occupancy."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(spaces_list)


@spaces.command('list')
def spaces_list():
    """List spaces with live occupancy."""
    client = _client()
    states = {s['name']: s for s in client.spaces_state()}
    for space in client.spaces():
        state = states.get(space.name, {})
        mark = '●' if state.get('occupied') else '○'
        people = state.get('people_count') or 0
        click.echo(f"{mark} {space.name:<20} people={people} zones={len(space.zones)}")


@spaces.command('create')
@click.argument('name')
@click.option('--width', 'width_m', type=float, default=None)
@click.option('--height', 'height_m', type=float, default=None)
def spaces_create(name, width_m, height_m):
    space = _client().create_space(name, width_m=width_m, height_m=height_m)
    click.echo(f"✓ created space '{space.name}' (id={space.id})")


@spaces.command('state')
@click.argument('name', required=False)
def spaces_state(name):
    """Show live spatial state (one space or all)."""
    client = _client()
    if name:
        click.echo(json.dumps(client.space(name).state(), indent=2))
    else:
        click.echo(json.dumps(client.spaces_state(), indent=2))


@spaces.command('zone')
@click.argument('space_name')
@click.argument('zone_name')
@click.argument('polygon')  # JSON polygon, e.g. "[[0,0],[2,0],[2,2],[0,2]]"
def spaces_zone(space_name, zone_name, polygon):
    """Add a zone polygon to a space."""
    try:
        points = json.loads(polygon)
    except ValueError as exc:
        raise click.ClickException(f"invalid polygon JSON: {exc}") from exc
    zone = _client().space(space_name).add_zone(zone_name, points)
    click.echo(f"✓ zone '{zone.name}' added")


@spaces.command('place')
@click.argument('device_id')
@click.argument('space_name')
@click.option('--x', type=float, default=0.0)
@click.option('--y', type=float, default=0.0)
@click.option('--rotation', 'rotation_deg', type=float, default=0.0)
@click.option('--fov', 'fov_deg', type=float, default=90.0)
@click.option('--range', 'range_m', type=float, default=8.0)
def spaces_place(device_id, space_name, x, y, rotation_deg, fov_deg, range_m):
    """Place a device inside a space (plan coordinates in meters)."""
    client = _client()
    space = client.space(space_name)
    placement = client.device(device_id).place(
        space, x=x, y=y, rotation_deg=rotation_deg,
        fov_deg=fov_deg, range_m=range_m)
    click.echo(json.dumps(placement, indent=2))


@main.command()
@click.argument("device_id")
@click.option("--minute", default=None)
def predictions(device_id, minute):
    """Show recent predictions for a device."""
    client = _client()
    click.echo(json.dumps(client.device(device_id).predictions(minute), indent=2))


# ── config ────────────────────────────────────────────────────────────────────

@main.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """CLI configuration."""
    if ctx.invoked_subcommand is None:
        if DEVICE_FILE.exists():
            click.echo(DEVICE_FILE.read_text())
        else:
            click.echo("No device config — run `thothcraft pair`")


# ── update ────────────────────────────────────────────────────────────────────

@main.command()
def update():
    """Update thothcraft-cli to the latest release."""
    installer = shutil_which("apt") and "sudo apt update && sudo apt install --only-upgrade thothcraft-cli" \
        or shutil_which("brew") and "brew upgrade thothcraft-cli" \
        or shutil_which("pipx") and "pipx upgrade thothcraft-cli" \
        or "pip install --upgrade thothcraft-cli"
    click.echo(f"Run: {installer}")


def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)


# ── device ────────────────────────────────────────────────────────────────────

@main.group()
def device():
    """Turn this computer into a Thoth device (thothcraftd)."""


@device.command("init")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def device_init(yes):
    """Detect hardware and register this computer as a Thoth device."""
    click.echo("\nScanning this computer...\n")
    caps = probe.scan()
    for name, (avail, detail) in caps.items():
        click.echo(f"{name}\n{'✓' if avail else '✗'} {detail}\n")
    if not yes and not click.confirm("Create this computer as a Thoth device?", default=True):
        return
    ctx = click.get_current_context()
    ctx.invoke(pair)


@device.command("start")
def device_start():
    """Start the thothcraftd device daemon."""
    if sys.platform.startswith("linux") and Path("/usr/lib/systemd/user/thothcraftd.service").exists():
        subprocess.run(["systemctl", "--user", "start", "thothcraftd"], check=False)
        click.echo("✓ thothcraftd started via systemd")
        return
    from .daemon import run
    click.echo("Starting thothcraftd in foreground (Ctrl+C to stop)...")
    raise SystemExit(run())


@device.command("stop")
def device_stop():
    """Stop the thothcraftd device daemon."""
    if sys.platform.startswith("linux"):
        subprocess.run(["systemctl", "--user", "stop", "thothcraftd"], check=False)
        click.echo("✓ thothcraftd stopped")
    else:
        click.echo("Stop the foreground daemon with Ctrl+C")


@device.command("restart")
def device_restart():
    """Restart the thothcraftd device daemon."""
    ctx = click.get_current_context()
    ctx.invoke(device_stop)
    ctx.invoke(device_start)


@device.command("logs")
@click.option("--lines", default=50)
def device_logs(lines):
    """Show daemon logs."""
    if sys.platform.startswith("linux"):
        subprocess.run(["journalctl", "--user", "-u", "thothcraftd", "-n", str(lines), "--no-pager"], check=False)
    else:
        log = CONFIG_DIR / "thothcraftd.log"
        if log.exists():
            click.echo("".join(log.read_text().splitlines(keepends=True)[-lines:]))
        else:
            click.echo("No daemon log found")
