# thothcraft-sdk

Python SDK for the ThothCraft platform. The PyPI distribution is
`thothcraft-sdk`; the import namespace is `thothcraft`.

```bash
pip install thothcraft-sdk
```

```python
from thothcraft import Client

thoth = Client.login(username="you@example.com", password="secret")

for device in thoth.devices():
    print(device.name, device.online)
    for minute_id in device.minutes():
        minute = device.minute(minute_id)
        print(minute.predictions)
```

The same API works against a cloud Brain deployment or a local device
runtime — only the `base_url` changes.

## Layout

- `thothcraft.client` — authenticated Brain client
- `thothcraft.devices` — device control, live stream, file listing
- `thothcraft.minutes` — captured-minute decoding (radar/CSI/camera/sense)
- `thothcraft.sensors` — per-modality parsing helpers
- `thothcraft.datasets` — dataset construction from minutes
- `thothcraft.annotation` — auto-annotation helpers
- `thothcraft.integrations` — Home Assistant / ROS2 bridges
