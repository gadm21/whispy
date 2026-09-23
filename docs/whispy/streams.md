# Streams & windows

Raw `SensorSample`s become useful when they're grouped into aligned windows.
Whispy provides `SampleStream` for ingestion and `WindowSynchronizer` for
windowing.

## SampleStream

`SampleStream` wraps a sensor's sample iterator and gives you a uniform,
buffered stream you can merge, filter, and synchronize.

```python
import whispy
from whispy import SampleStream

node = whispy.local()
stream = SampleStream(node.sensor("radar").stream())
```

## WindowSynchronizer

`WindowSynchronizer` aligns one or more streams into fixed-duration
`SensorWindow`s — the unit a processor consumes.

```python
from whispy import WindowSynchronizer

windows = WindowSynchronizer(window_s=2.0).sync(stream)
for window in windows:
    ...
```

For multi-sensor fusion, synchronize several streams so each window contains
aligned samples from every modality:

```python
streams = [
    SampleStream(node.sensor("radar").stream()),
    SampleStream(node.sensor("csi").stream()),
    SampleStream(node.sensor("imu").stream()),
]
windows = WindowSynchronizer(window_s=2.0).sync(*streams)
```

## SensorWindow & WindowFeatures

A `SensorWindow` holds the samples for each sensor in the window:

```python
window["radar"]            # samples for the radar sensor
window.to_numpy("radar")   # as a NumPy array
```

`WindowFeatures` derives per-sensor features used by rule processors and
fusion:

- `<sensor>_mean`, `<sensor>_std`, `<sensor>_max`, `<sensor>_min`,
  `<sensor>_energy`
- `snr_mean` — signal-to-noise summary for RF sensors

These feature names are what rule expressions reference:

```python
{"when": "snr_mean > 12 and radar_std < 0.5", "label": "occupied"}
```
