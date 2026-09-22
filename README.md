# ThothCraft Python tooling

This repository contains `thothcraft-sdk` and `thothcraft-cli`. The former whispy
package is retired. Python 3.10–3.13 are tested in CI; import `thothcraft` and run
`thothcraft`/`thothcraftd`. The GitHub repository can be renamed later.

```sh
python -m pip install -e packages/thothcraft-sdk -e packages/thothcraft-cli
thothcraft login --base-url https://web-production-d7d37.up.railway.app
thothcraft devices
thothcraft models list
```

The default API URL is `https://api.thothcraft.com`. Until DNS and the Railway
custom domain are ready, use the Railway URL above. Login stores URL and token
in `~/.thothcraft/credentials.json`; `Client.login()` restores both.
Raw downloads require Home or Research; Labs and datasets require Research.

## Train → export → upload → deploy → predict

Install `packages/thothcraft-sdk[dl]` for PyTorch. This example trains on
**synthetic demonstration data**, not validated sensor data. Replace it with
labeled windows, matching the device's thoth-model/v1 preprocessing and shape.

```python
from pathlib import Path
import torch
from thothcraft import Client

torch.manual_seed(7)
x = torch.randn(64, 1, 1, 8)
y = (x.mean(dim=(1, 2, 3)) > 0).long()
net = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(8, 2))
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(100):
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(net(x), y).backward()
    optimizer.step()
net.eval()
path = Path('occupancy.pt')
torch.jit.trace(net, x[:1]).save(str(path))
client = Client.login()
model = client.upload_model(
    path, name='Occupancy demo', classes=['empty', 'occupied'],
    input_spec={'sensor': 'radar', 'representation': 'raw_adc',
                'frames': 1, 'shape': [1, 1, 1, 8],
                'fit': 'left_pad_latest', 'normalization': {'kind': 'none'}},
)
device = client.devices()[0]
deployment = device.deploy(model, timeout=180)
print(deployment.status)  # delivered or declined
print(device.predictions())  # may be empty until the next capture
for chunk in device.stream(max_items=10):
    print(chunk.get('model_predictions', []))
```

`device.deploy(path, name=..., classes=..., input_spec=...)` uploads first.
Use `wait=False` to return immediately. A wait timeout raises `TimeoutError`
without cancelling the request. `client.cancel_deployment(id)` cancels pending
deployments; `client.set_deployment_active(id, enabled)` controls delivered models.

## Data interoperability

```python
from thothcraft import Dataset
from thothcraft.datasets.torch import ThothTorchDataset  # requires [dl]

with device.minute('20260922_0000') as minute:
    values = minute['radar'].to_numpy()
    table = minute['radar'].to_dataframe()  # requires [pandas]
    tensor = minute['radar'].to_torch()     # requires [dl]
    minute.label(activity=0)
    dataset = Dataset('example').add(minute)
    X, y = dataset.to_xy(sensor='radar', label='activity')
    adapter = ThothTorchDataset(dataset, sensor='radar', label='activity')
```

Minute iterates four sensor keys: radar, csi, camera, sense. NumPy conversion
decodes radar ADC bytes and CSI real/imag channels. Numeric samples are stacked;
ragged windows raise ValueError and need explicit padding/windowing. Decode
camera images before ML use. `to_xy` returns one flattened row per minute and
requires a label on every minute. Downloads are ZIP bundles containing capture.npz.

```sh
thothcraft models upload model.pt --name Occupancy --classes empty,occupied --input-spec inputs.json
thothcraft models deploy 7 DEVICE_UUID --timeout 180
thothcraft models deployments
thothcraft models cancel DEPLOYMENT_ID
thothcraft predictions DEVICE_UUID --minute 20260922_0000
```

## Any computer is a Thoth node

`thothcraftd` runs on commodity hardware (Windows laptop, Mac, Linux box,
Jetson) and exposes the same local API as the Pi dashboard on port 5000 —
sensor inventory, live camera frames — plus Brain heartbeats once paired.

```powershell
# Windows
irm https://get.thothcraft.com/install.ps1 | iex
```

```sh
# Linux / macOS
curl -fsSL https://get.thothcraft.com/install.sh | bash
```

From a cloned repo: `install.ps1 -Local .\packages` or
`./install.sh --local ./packages`. The scripts install both packages (with
the `sensors` extra: opencv, pyserial, psutil) and register thothcraftd as a
logon task / systemd user service / LaunchAgent. The daemon serves the local
API even before pairing, so this works immediately:

```python
import thothcraft
node = thothcraft.local("127.0.0.1")      # or any node on the LAN
node.sensors()                            # probed capabilities
frame = node.camera_frame()               # JPEG bytes
for batch in node.csi_stream():           # CSI sample batches
    ...
node.set_matrix(text="hi")                # Sense HAT LED matrix (Pi nodes)
```

`thothcraft pair` links the node to your account for heartbeats/sync.

## Debian / Raspberry Pi OS

Build on Debian/Ubuntu with dpkg-deb:

```sh
bash packaging/build-deb.sh
sudo apt install ./thothcraft-cli_0.1.0_all.deb
thothcraft login
thothcraft device init
systemctl --user daemon-reload
systemctl --user enable --now thothcraftd
```

The architecture-independent package ships sources. Post-install creates
`/opt/thothcraft/venv` on the target machine and installs both packages; network
access is required. This avoids shipping a nonrelocatable virtualenv or x86
NumPy binaries to a Pi. The unit is installed to `/usr/lib/systemd/user`.
Credentials stay in the user's home. Enable user lingering separately if needed
after logout. Stop the user unit before removing the package.

The existing daemon implements pairing and heartbeats; collection, inference,
sync and command execution remain TODOs. The full Pi runtime is in `thoth`.

### Optional signed APT repository (manual)

1. Install reprepro and gnupg on a trusted Linux publishing machine. Generate a
   dedicated signing key and keep its private material out of GitHub.
2. Create `apt/conf/distributions` with `Codename: stable`, `Components: main`,
   `Architectures: amd64 arm64`, and `SignWith: YOUR_KEY_FINGERPRINT`.
3. Run `reprepro -b apt includedeb stable thothcraft-cli_0.1.0_all.deb`.
4. Export the public key with `gpg --armor --export YOUR_KEY_FINGERPRINT > apt/key.asc`
   and publish the apt directory to GitHub Pages over HTTPS.
5. Install the dearmored key at `/etc/apt/keyrings/thothcraft.gpg`. Add
   `deb [signed-by=/etc/apt/keyrings/thothcraft.gpg] https://OWNER.github.io/REPO stable main`
   to `/etc/apt/sources.list.d/thothcraft.list`, then run `sudo apt update`.

## Development

```sh
python -m pip install -e packages/thothcraft-sdk -e packages/thothcraft-cli pytest
python -m pytest packages/thothcraft-sdk/tests packages/thothcraft-cli/tests
```

CI tests both packages on Python 3.10–3.13 and builds/installs the Debian package.
The PyPI tag job remains a stub until trusted publishing is configured.
