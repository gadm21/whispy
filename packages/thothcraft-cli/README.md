# thothcraft-cli

System CLI and device runtime for ThothCraft. Turns any computer —
Raspberry Pi, laptop, Jetson, robot — into a Thoth device.

```bash
pipx install thothcraft-cli        # development
sudo apt install thothcraft-cli    # Debian/Ubuntu (see Packaging)
brew install thothcraft-cli        # macOS (planned)
winget install ThothCraft.CLI      # Windows (planned)
```

## Commands

```text
thothcraft login / logout / whoami
thothcraft devices [list]
thothcraft pair / unpair
thothcraft status / doctor
thothcraft sensors [list]
thothcraft data [list|sync]
thothcraft models / predictions
thothcraft config [get|set]
thothcraft update
thothcraft device init|start|stop|restart|logs
```

## `thothcraft device init`

Probes the computer for usable sensors (camera, microphone, Wi-Fi,
Bluetooth, USB-serial receivers, system sensors) and registers it as a
Thoth device through the standard pairing flow.

## `thothcraftd`

The device daemon handles sensor discovery, collection, local storage,
prediction, cloud sync, commands and heartbeats. The CLI controls it;
on Linux it runs under systemd:

```bash
systemctl status thothcraftd
```

## apt packaging (maintainer steps)

1. Build a wheel: `pip wheel . -w dist`
2. Create the Debian layout:

   ```text
   thothcraft-cli_0.1.0/
   ├── DEBIAN/control          # Package: thothcraft-cli, Depends: python3
   ├── DEBIAN/postinst         # systemctl daemon-reload; enable thothcraftd
   ├── usr/lib/thothcraft/     # unpacked wheel + venv or pex binary
   ├── usr/bin/thothcraft      # wrapper → /usr/lib/thothcraft/bin/thothcraft
   ├── usr/bin/thothcraftd
   └── lib/systemd/system/thothcraftd.service
   ```

3. `dpkg-deb --build thothcraft-cli_0.1.0`
4. Host in an APT repo (reprepro/aptly) or ship the `.deb` directly.

Recommended: bundle with `pex` or a venv so the package has no Python
dependency conflicts. The systemd unit should run
`thothcraftd --config /etc/thothcraft/device.json` as a dedicated
`thothcraft` user.
