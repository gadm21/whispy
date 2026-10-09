# LLM prompts & responses

Brain calls OpenAI in exactly **three places**. This page documents each one
end-to-end: the verbatim system prompt, the complete user message built from
**real production data**, the tool schema, and the verbatim model response —
captured against the live fleet on 2026-10-08 (`gpt-4o-mini`).

| # | Call site | Endpoint / trigger | In | Out |
|---|-----------|--------------------|----|-----|
| 1 | `context_builder._llm_update` | `POST /v1/context/rebuild` (also runs on the builder schedule) | evidence bundle + current map | `update_context_map` diff proposal |
| 2 | `context_infer._openai_form` | `POST /v1/context/infer` | window + calibration + descriptors | `submit_context_form` form |
| 3 | `aiagent.handler.query.query_openai` | assistant chat | memories + system stats + question | text (+ optional tool calls) |

Source:

- [`server/v1/context_builder.py`](https://github.com/Thothcraft/Brain/blob/main/server/v1/context_builder.py) — `_SYSTEM_PROMPT`, `build_bundle()`, `_map_tool_schema()`, `_llm_update()`
- [`server/v1/context_infer.py`](https://github.com/Thothcraft/Brain/blob/main/server/v1/context_infer.py) — `InferRequest`, `_SYSTEM_PROMPT`, `_ctx_tool_schema()`, `_openai_form()`
- [`aiagent/handler/query.py`](https://github.com/Thothcraft/Brain/blob/main/aiagent/handler/query.py) — `query_openai()`
- Capture script: `acceptance/_llm_prompt_capture.py` in the thothcraft workspace (writes `llm_prompt_samples.json`)

---

## The data the prompts are built from

The LLM never sees raw sensor samples. Everything is reduced on-device or in
Brain to compact, versioned records in `context_evidence`:

### `context.descriptors.v1` — the per-minute scene uplink

Every node pushes one of these per minute. This is a **real row** from
`thoth-chen` (Pi5, camera + ESP32-CSI + mmWave radar), verbatim:

```json
{
 "value": {
  "detail": "descriptors",
  "rate_s": 60.0,
  "predictions": {},
  "estimates": [
   {"key": "occupancy.v1",
    "value": {"occupied": true, "distinct_subjects": 33},
    "confidence": 1.0}
  ],
  "window_s": 2.0,
  "sensors": {
   "system-0": {
    "type": "system", "n": 2, "rate_hz": 1.0, "age_s": 0.55,
    "fields": {
     "cpu_percent":   {"mean": 31.5, "std": 1.8, "min": 29.7, "max": 33.3},
     "mem_percent":   {"mean": 19.8, "std": 0.0, "min": 19.8, "max": 19.8},
     "load_avg_mean": {"mean": 2.4411, "std": 0.0, "min": 2.4411, "max": 2.4411}
    },
    "text": "system: cpu_percent 31.5, mem_percent 19.8, load_avg_mean 2.44"
   },
   "camera-2c7d": {
    "type": "camera", "n": 30, "rate_hz": 15.14, "age_s": 0.08,
    "fields": {"width": {"mean": 640.0}, "height": {"mean": 480.0}},
    "text": "camera: 30 frames; camera: nobody visible",
    "cues": {"people": 0, "confidence": 0.0, "faces": 0}
   },
   "csi-bb8b": {
    "type": "wifi_csi", "n": 120, "rate_hz": 61.68, "age_s": 0.03,
    "fields": {
     "rssi":          {"mean": -75.7815, "std": 16.6573, "min": -99.0, "max": -25.0},
     "iq_mean":       {"mean": -10.2321, "std": 43.9574},
     "n_subcarriers": {"mean": 64.0},
     "channel":       {"mean": 4.6937, "std": 4.0108, "min": 1.0, "max": 11.0}
    },
    "text": "wifi_csi: addr_type 0.875, rssi -75.8, tx_power 9.67"
   },
   "radar-a316": {
    "type": "radar", "n": 20, "rate_hz": 10.0, "age_s": 0.04,
    "fields": {
     "snr_db":            {"mean": 0.9037, "std": 0.0037, "min": 0.9, "max": 0.912},
     "range_profile_mean": {"mean": 1986.0966, "std": 0.0558},
     "energy":            {"mean": 3969588.6183, "std": 220.3523}
    },
    "text": "radar: low motion (SNR std 0.0 dB, mean 0.9 dB)",
    "cues": {"motion": "low", "snr_std_db": 0.0}
   }
  },
  "scene": "system: cpu_percent 31.5, mem_percent 19.8, load_avg_mean 2.44 | camera: 30 frames; camera: nobody visible | wifi_csi: addr_type 0.875, rssi -75.8, tx_power 9.67 | radar: low motion (SNR std 0.0 dB, mean 0.9 dB)"
 },
 "subject": "device:fa81cdda-58b2-5caa-b31b-e6ff3dce19ec"
}
```

This is what `build_bundle()` puts in `scenes[]` — the LLM mostly reads the
`text`/`cues`/`scene` sentences and the on-device `predictions`/`estimates`.

### Other live evidence keys (real rows, prod table)

| `evidence_key` | count | example `value` (verbatim) |
|---|---|---|
| `radio.wifi.v1` | 8,062 | `{"observer":"c1046bd3-…","component_id":"radio-cacd","target":"wifi:be:d5:ed:dc:90:a4","radio":"wifi","measurement":"rssi","rssi_dbm":-72.0,"channel":132}` |
| `radio.ble.v1` | 13,774 | same shape, `radio:"ble"` |
| `radio.csi.v1` | 251 | `{"observer":"fa81cdda-…","component_id":"csi-bb8b","target":"wifi:1a:00:00:00:00:00","measurement":"csi","rssi_dbm":-75.0,"channel":6,"stream":"wifi_csi_unclassified/v1"}` |
| `ble.discovery.v1` | 1,863 | `{"observer":"phone:gadgad","target":"ble:CC:50:D2:0D:0B:87","rssi_dbm":-34.0,"known":false}` |
| `ble.proximity.v1` | 28 | `{"observer":"phone:gadgad","target":"FF:96:7E:5A:69:41","rssi_dbm":-62.0}` |
| `activity.motion.v1` | 15,134 | `{"acc_x":1.532,"acc_y":7.948,"acc_z":6.213}` (phone/watch IMU) |
| `location.geo.v1` | 511 | `{"lat":42.946,"lon":-81.303,"acc_m":7.663,"speed_mps":0.188}` |
| `context.state.v1` | 6,458 | node estimator output: `{"key":"occupancy.v1","entity_id":"c1046bd3-…","value":{"occupied":false,"distinct_subjects":0},"confidence":0.4,"estimator":"ble-presence/1"}` |
| `net.link.v1` | 2 | `{"iface":"wlan0","state":"connected","ssid_hash":"0fdbb63194ca"}` |

### Manually-entered data — room layout

Users draw a room in the thoth-app room editor; the node caches it as a
`room/v1` document (stored server-side in `node_room`, pushed over the node
websocket). Real doc from `thoth-laptop`:

```json
{
 "format": "room/v1", "room_id": "1", "name": "myroom",
 "dims": {"w": 6.0, "d": 4.0, "h": 2.6},
 "walls": [],
 "furniture": [
  {"id": "f-mui5c38l", "type": "bed",   "pos": [-0.469, 0.0,  0.789],  "rot_y": 0.0,   "dims": [2.0, 0.5, 1.6]},
  {"id": "f-mui5cc2m", "type": "shelf", "pos": [-0.177, 0.0, -1.520],  "rot_y": 0.0,   "dims": [1.2, 1.0, 1.0]},
  {"id": "f-mui5d5bx", "type": "sofa",  "pos": [ 2.550, 0.0, -0.545],  "rot_y": 4.712, "dims": [1.8, 0.8, 0.8]},
  {"id": "f-mui5dnfq", "type": "desk",  "pos": [ 1.395, 0.0,  1.238],  "rot_y": 0.0,   "dims": [1.5, 0.75, 1.0]}
 ],
 "devices": []
}
```

A second doc (thoth-april) carries the building anchor:

```json
{"building": {"id": "house-main", "name": "Toronto",
  "anchor": {"latitude": 43.65, "longitude": -79.38}},
 "spatial": {"surveyed": false, "origin_enu_m": null, "heading_deg": null}}
```

The relational mirror is `space` / `zone` / `device_placement`
(building→floor→room hierarchy, polygon zones, per-device pose `x, y,
rotation_deg, fov_deg, range_m`). Prod currently has `space(1) = "gadroom"`.

### Calibration data

Two different calibration mechanisms exist — don't confuse them:

- **Node-side `thoth calibrate`** — a guided three-region (red/yellow/green)
  workflow. `src/backend/calibration.py` derives midpoint thresholds from
  detection-ratio samples, e.g.
  `{"yellow_threshold_percent": 41.2, "green_threshold_percent": 66.8,
  "sample_summaries": {"red": {"sample_count": 5, "median_percent": 33.0, ...}}}`.
- **`POST /v1/context/infer` `calibration` field** — arbitrary per-target-class
  stats the caller supplies: means/stds/thresholds/priors per class, e.g.
  `{"occupancy.v1": {"occupied": {"mean_conf": 0.8}, "empty": {"mean_conf": 0.3}}}`.
  The model measures descriptor-vs-class distance against these numbers.

---

## Call 1 — context builder (`POST /v1/context/rebuild`)

`POST /v1/context/rebuild` → `run_build()` → `build_bundle()` →
`_llm_update()` → stabilizer (hysteresis, alias resolution, TTL) → DB.

### System prompt (verbatim, `context_builder._SYSTEM_PROMPT`)

```text
You maintain a STABLE semantic context map of persons, places, devices,
activities and objects for one household/lab. You never see raw sensor
data. You get:
  * `scenes` — the latest uplink per node with TEXTUAL cues already
    computed on the device: a one-line `scene`, a sentence per sensor
    (`text`) and structured `cues` (speech transcript, people count,
    recognized face identity, motion level, strongest radio emitter),
    plus on-device `predictions`. Prefer these — they are the most
    direct evidence.
  * `descriptors` — compact physical aggregates per (evidence key,
    device): counts, field mean/min/max, latest value, mean confidence,
    age.
Typical keys: occupancy/presence probabilities, radar SNR and range,
CSI amplitude variance, BLE/Wi-Fi RSSI sightings (with decoded beacon
identities), IMU motion variance, face/person detections, audio level,
model predictions (key 'prediction'), device metadata/room placement.

Update the map by calling update_context_map exactly once:
  * Reuse existing entity ids from `map` whenever they match — never
    invent a new id for something already on the map. Put alternative
    names/MACs/face ids in `aliases`.
  * Devices are pre-seeded from the registry (device:<uuid>); relate
    them, don't recreate them.
  * Use located_in (person/device/object -> place) and doing
    (person -> activity:<slug>) for the current situation; one active
    place and one activity per subject.
  * States: occupancy.v1 per place ({"occupied": bool}), presence.v1 per
    person ({"present": bool}), activity.v1 per person ({"activity": str}).
  * Confidence must reflect descriptor support: weak, stale (large
    age_s) or conflicting evidence → low confidence. Omit what the
    descriptors don't support; absence of evidence is not evidence of
    absence unless the relevant sensor is fresh and reports it.
  * Only propose merges for clear duplicates (same physical thing with
    two ids). `retire` removes ghosts.
```

### User message — the real `build_bundle()` output

```json
{
 "now": 1791460302.29, "window_s": 9605.4, "evidence_rows": 20,
 "scenes": [],
 "descriptors": [
  {"key": "ble.discovery.v1", "device": null, "n": 7, "span_s": 551.4,
   "age_s": 9005.4, "mean_confidence": null,
   "fields": {"rssi_dbm": {"mean": -33.714, "min": -52.0, "max": -22.0, "n": 7}},
   "latest": {"observer": "phone:gadgad", "target": "ble:CC:50:D2:0D:0B:87",
              "rssi_dbm": -34.0, "adv_name": null, "known": false}},
  {"key": "activity.motion.v1", "device": null, "n": 7, "span_s": 516.8,
   "age_s": 9056.7,
   "fields": {"acc_x": {"mean": 0.648, "min": -3.435, "max": 1.796},
              "acc_y": {"mean": 6.182, "min": -0.156, "max": 8.81},
              "acc_z": {"mean": 0.833, "min": -9.962, "max": 6.213}},
   "latest": {"acc_x": 1.532, "acc_y": 7.948, "acc_z": 6.213}},
  {"key": "location.geo.v1", "device": null, "n": 6, "span_s": 468.0,
   "age_s": 9010.9,
   "fields": {"lat": {"mean": 42.946}, "lon": {"mean": -81.303},
              "acc_m": {"mean": 7.298}, "speed_mps": {"mean": 0.138}},
   "latest": {"lat": 42.946, "lon": -81.303, "acc_m": 7.663, "speed_mps": 0.188}}
 ],
 "map": {
  "entities": [
   {"id": "device:fa81cdda-58b2-5caa-b31b-e6ff3dce19ec", "kind": "device",
    "name": "thoth-chen",   "aliases": ["fa81cdda-…", "thoth-chen"],   "confidence": 1.0},
   {"id": "device:6ee6922b-91f0-4343-ad3d-97653354e1da", "kind": "device",
    "name": "phone:gadgad", "aliases": ["6ee6922b-…", "phone:gadgad"], "confidence": 1.0},
   {"id": "device:42d6921a-9872-5a5d-a9da-bacc6f908680", "kind": "device",
    "name": "InfiniTime",   "aliases": ["42d6921a-…", "InfiniTime"],   "confidence": 1.0},
   {"id": "device:168477ac-fbb3-53f6-9c9a-3c7d894ff545", "kind": "device",
    "name": "thoth-april",  "aliases": ["168477ac-…", "thoth-april"],  "confidence": 1.0},
   {"id": "device:c1046bd3-dc05-5f34-a502-32d5bbb859a8", "kind": "device",
    "name": "thoth-laptop", "aliases": ["c1046bd3-…", "thoth-laptop"], "confidence": 1.0}
  ],
  "relationships": [],
  "states": [{"key": "prediction",
              "entity_id": "c1046bd3-dc05-5f34-a502-32d5bbb859a8",
              "value": "no_face", "confidence": 0}]
 }
}
```

### Tool schema — `update_context_map` (abridged)

```json
{"type": "function", "function": {
 "name": "update_context_map",
 "description": "Propose updates to the semantic context map. This is the ONLY output channel…",
 "parameters": {"type": "object", "properties": {
   "summary":       {"type": "string"},
   "entities":      [{"id": "<kind>:<slug>", "kind": "person|place|device|activity|object",
                      "name", "aliases", "attributes", "confidence"}],
   "relationships": [{"subject", "predicate": "located_in|doing|uses|carries|observed_by|part_of|near",
                      "object", "confidence"}],
   "states":        [{"key": "occupancy.v1|activity.v1|presence.v1", "entity_id",
                      "value", "confidence"}],
   "merges":        [{"source", "target", "reason"}],
   "retire":        [{"id", "reason"}]
 }, "required": ["summary"]}}}
```

Call: `model=gpt-4o-mini`, `temperature=0.1`, forced
`tool_choice={function: update_context_map}`.

### Real response (tool args, verbatim)

```json
{
 "summary": "Updating context map with BLE discovery, motion activity, and location data.",
 "entities": [
  {"id": "device:6ee6922b-91f0-4343-ad3d-97653354e1da", "kind": "device", "confidence": 1.0},
  {"id": "device:fa81cdda-58b2-5caa-b31b-e6ff3dce19ec", "kind": "device", "confidence": 1.0},
  {"id": "device:42d6921a-9872-5a5d-a9da-bacc6f908680", "kind": "device", "confidence": 1.0},
  {"id": "device:168477ac-fbb3-53f6-9c9a-3c7d894ff545", "kind": "device", "confidence": 1.0},
  {"id": "device:c1046bd3-dc05-5f34-a502-32d5bbb859a8", "kind": "device", "confidence": 1.0}
 ],
 "states": [
  {"key": "location.geo.v1", "entity_id": "device:6ee6922b-91f0-4343-ad3d-97653354e1da",
   "value": {"lat": 42.946, "lon": -81.303}, "confidence": 0.8},
  {"key": "activity.v1", "entity_id": "device:6ee6922b-91f0-4343-ad3d-97653354e1da",
   "value": "motion", "confidence": 0.8}
 ],
 "merges": [], "retire": []
}
```

Usage: 1890 prompt / 319 completion tokens. Note the conservatism — evidence
was ~2.5 h stale (`age_s ≈ 9000`), so it touched only the seeded devices and
proposed nothing new. On fresh builds the same pipeline produced
`person:Gad located_in …`, `presence.v1 {present:true}` and discovered BLE
entities like `device:c3:33:36:32:25:77 (Govee_H6054)`.

---

## Call 2 — context infer (`POST /v1/context/infer`)

`POST /v1/context/infer` → `_openai_form()` → the returned form is validated
by the same pydantic models and applied through the same `apply_*` helpers as
the REST `/v1/context/*` endpoints — so the LLM writes are format-identical
to manual writes. Scope: `context:write`. `dry_run=true` validates without
persisting.

### Request model (`InferRequest`)

```json
{"window":      {"start_ts": 1791450696.897, "end_ts": 1791460302.29, "device_id": null},
 "entity_hint": null,
 "calibration": {"<evidence_key>": {"<class>": {arbitrary stats}, …}, …},
 "descriptors": {"<evidence_key>": {per-key aggregate}, …},
 "dry_run": true}
```

### System prompt (verbatim, `context_infer._SYSTEM_PROMPT`)

```text
You are the context-layer estimator for a sensor-fusion platform. For
each request you receive:
  * `calibration` — statistics of the target classes learned during
    calibration (per-class feature means/stds, thresholds, priors,
    support counts).
  * `descriptors` — physical descriptors of ONE time window from the
    sensors (CSI/radar/BLE/IMU summaries, RSSI values, variances,
    packet counts, spectral features).
  * `window` — the window's bounds and provenance.

Classify the window against the target classes using the calibration
stats, then respond ONLY by calling submit_context_form. Do not emit
plain text. Rules:
  * states[] must contain one entry per target class key the platform
    uses (e.g. occupancy.v1, activity.v1, location.v1); value is the
    predicted label/object, confidence is calibrated by distance to the
    class statistics — not raw probability.
  * evidence[] records what you based the prediction on (the model
    probabilities, the descriptors, the calibration reference).
  * entities[] and relationships[] describe WHO/WHERE/WHAT the window
    implies — create them when confident, omit when unsupported.
  * Confidence must reflect the descriptor-vs-calibration distance:
    a window far from every class centroid yields low confidence, not
    a forced label.
  * All numbers are floats (epoch seconds); no prose inside values.
```

### Real request — calibration + descriptors

```json
{
 "window": {"start_ts": 1791450696.897, "end_ts": 1791460302.2905, "device_id": null},
 "entity_hint": null,
 "calibration": {
  "ble.discovery.v1":    {"occupied": {"mean_conf": 0.8}, "empty": {"mean_conf": 0.3}},
  "activity.motion.v1":  {"occupied": {"mean_conf": 0.8}, "empty": {"mean_conf": 0.3}},
  "location.geo.v1":     {"occupied": {"mean_conf": 0.8}, "empty": {"mean_conf": 0.3}}
 },
 "descriptors": {
  "ble.discovery.v1":   {"n": 7, "age_s": 9005.4,
    "fields": {"rssi_dbm": {"mean": -33.714, "min": -52.0, "max": -22.0}},
    "latest": {"observer": "phone:gadgad", "target": "ble:CC:50:D2:0D:0B:87", "rssi_dbm": -34.0}},
  "activity.motion.v1": {"n": 7, "age_s": 9056.7,
    "fields": {"acc_x": {"mean": 0.648}, "acc_y": {"mean": 6.182}, "acc_z": {"mean": 0.833}},
    "latest": {"acc_x": 1.532, "acc_y": 7.948, "acc_z": 6.213}},
  "location.geo.v1":    {"n": 6, "age_s": 9010.9,
    "fields": {"lat": {"mean": 42.946}, "lon": {"mean": -81.303},
               "acc_m": {"mean": 7.298}, "speed_mps": {"mean": 0.138}},
    "latest": {"lat": 42.946, "lon": -81.303, "acc_m": 7.663}}
 }
}
```

### Tool schema — `submit_context_form` (abridged)

```json
{"type": "function", "function": {
 "name": "submit_context_form",
 "description": "Submit the completed context layer form for this window… validated and applied exactly as the /v1/context REST endpoints would apply it.",
 "parameters": {"type": "object", "properties": {
   "summary", "entities", "relationships",
   "evidence": [{"ref", "key", "value", "timestamp", "confidence",
                 "model_id", "model_version", "source_id", "device_id", "provenance"}],
   "states":   [{"key", "value", "entity_id", "confidence", "since",
                 "valid_until", "transition": "entered|exited|changed",
                 "evidence_refs"}]
 }, "required": ["summary", "states"]}}}
```

Call: `model=gpt-4o-mini` (`CONTEXT_INFER_MODEL` or `MODEL_NAME`),
`temperature=0.2`, forced tool call.

### Real response (tool args, verbatim)

```json
{
 "summary": "The window likely depicts an occupied space with motion detected and a specific geographical location.",
 "entities": [
  {"id": "space:unknown", "kind": "space",
   "attributes": {"location": "42.946,-81.303", "accuracy_m": 7.298, "speed_mps": 0.188}},
  {"id": "person:gadgad", "kind": "person",
   "attributes": {"device": "phone:gadgad"}}
 ],
 "evidence": [
  {"ref": "ev1", "key": "ble.discovery.v1",
   "value": {"mean_rssi_dbm": -33.714, "n": 7}, "confidence": 0.8},
  {"ref": "ev2", "key": "activity.motion.v1",
   "value": {"mean_acc_x": 0.648, "mean_acc_y": 6.182, "mean_acc_z": 0.833, "n": 7},
   "confidence": 0.8},
  {"ref": "ev3", "key": "location.geo.v1",
   "value": {"lat": 42.946, "lon": -81.303, "acc_m": 7.298, "speed_mps": 0.138},
   "confidence": 0.8}
 ],
 "states": [
  {"key": "ble.discovery.v1",   "value": "occupied", "confidence": 0.8, "evidence_refs": ["ev1"]},
  {"key": "activity.motion.v1", "value": "occupied", "confidence": 0.8, "evidence_refs": ["ev2"]},
  {"key": "location.geo.v1",    "value": "occupied", "confidence": 0.8, "evidence_refs": ["ev3"]}
 ]
}
```

Usage: 1396 prompt / 296 completion tokens.

> **Observed quirk.** With evidence keys (`ble.discovery.v1`) used as the
> calibration class keys, the model echoed *evidence keys* into `states[].key`
> instead of platform state keys (`occupancy.v1`). Calibrate with real target
> keys (e.g. `occupancy.v1: {occupied: …, empty: …}`) so `states[]` lands on
> the versioned state namespace the rest of the API uses.

A second live capture — denver desktop node, proper `occupancy.v1`
calibration keys, and the model correctly emitting `occupancy.v1 → absent` —
is in [LLM context inference](brain/context-infer.md#real-captured-round-trip).

---

## Call 3 — aiagent (`query_openai`)

The chat assistant. Not a structured-output call — it assembles a 4-message
prompt and may chain registered tools (`tool_choice="auto"`, max 5
iterations). `temperature=0.7`, `max_tokens=1024`.

### Real message array

```json
[
 {"role": "system", "content":
  "You are ThothCraft AI, an intelligent assistant for the ThothCraft IoT Research Platform.\n\n"
  "You help users with:\n- Managing IoT devices and sensor data\n- Analyzing data and providing insights\n"
  "- Understanding their system status and metrics\n\n"
  "When users ask about their devices, data, or models, use the system_stats context provided to give accurate answers.\n"
  "Be helpful, concise, and accurate. If you don't have specific information, say so."},

 {"role": "system", "content": "User details: {}\n\nRecent conversation history: []"},

 {"role": "system", "content":
  "Current System Status:\n- Devices: [{\"name\": \"thoth-chen\", \"id\": \"fa81cdda-58b2-5caa-b31b-e6ff3dce19ec\", \"online\": true, \"last_seen\": \"2026-10-08T11:51:32.005795Z\", \"sensors\": [{\"id\": \"system-0\", \"type\": \"system\", \"online\": true}, {\"id\": \"camera-2c7d\", \"type\": \"camera\", \"online\": true}, {\"id\": \"csi-bb8b\", \"type\": \"wifi_csi\", \"online\": true}, {\"id\": \"zigbee-c80c\", \"type\": \"zigbee\", \"online\": true}, {\"id\": \"radio-24d8\", \"type\": \"radio_env\", \"online\": true}, {\"id\": \"radar-a316\", \"type\": \"radar\", \"online\": true}]}, {\"name\": \"phone:gadgad\", \"id\": \"6ee6922b-91f0-4343-ad3d-97653354e1da\", \"online\": false, \"sensors\": [{\"id\": \"phone-gps\", \"type\": \"gps\"}, {\"id\": \"phone-imu\", \"type\": \"imu\"}, {\"id\": \"phone-ble\", \"type\": \"ble_scan\"}]}, {\"name\": \"InfiniTime\", \"id\": \"42d6921a-9872-5a5d-a9da-bacc6f908680\", \"online\": false, \"sensors\": [{\"id\": \"pinetime-motion\", \"type\": \"imu\"}, {\"id\": \"pinetime-hr\", \"type\": \"heart_rate\"}, {\"id\": \"pinetime-steps\", \"type\": \"steps\"}, {\"id\": \"pinetime-battery\", \"type\": \"battery\"}, {\"id\": \"pinetime-prox\", \"type\": \"rssi\"}, {\"id\": \"pinetime-gps\", \"type\": \"gps\"}]}, {\"name\": \"thoth-april\", \"id\": \"168477ac-fbb3-53f6-9c9a-3c7d894ff545\", \"online\": true, \"sensors\": [{\"id\": \"system-0\", \"type\": \"system\"}, {\"id\": \"microphone-9e6c\", \"type\": \"microphone\"}, {\"id\": \"csi-8b45\", \"type\": \"wifi_csi\"}, {\"id\": \"radio-809c\", \"type\": \"radio_env\"}]}, {\"name\": \"thoth-laptop\", \"id\": \"c1046bd3-dc05-5f34-a502-32d5bbb859a8\", \"online\": true, \"sensors\": [{\"id\": \"system-0\"}, {\"id\": \"microphone-ff16\"}, {\"id\": \"microphone-9231\"}, {\"id\": \"camera-6d1a\"}, {\"id\": \"radio-cacd\"}]}]\n- Files: No file data\n- Models: No model data\n\nUse this information to answer questions about the user's devices, data, and models."},

 {"role": "user", "content": "Which of my nodes are online right now, and is anyone detected?"}
]
```

The `User details` / `Recent conversation history` message is built from
`LongTermMemoryManager` profile + `ShortTermMemoryManager` history; the
`Current System Status` message is built from `aux_data["system_stats"]`
(`/v1/devices`, files, models).

### Real response

```text
Currently, the following nodes are online:

1. **thoth-chen**
   - Sensors: All online (system, camera, wifi_csi, zigbee, radio_env, radar)

2. **thoth-april**
   - Sensors: All online (system, microphone, wifi_csi, radio_env)

3. **thoth-laptop**
   - Sensors: All online (system, microphone, microphone, camera, radio_env)

The other nodes, **phone:gadgad** and **InfiniTime**, are offline.

As for detection, all sensors on the online nodes are operational, so
detection capabilities are available across these devices.
```

Usage: 1014 prompt / 133 completion tokens. No tool calls — the device list
in `system_stats` was sufficient.

---

## Where the writes land

| Table | Written by | Contents (prod, Oct 8) |
|---|---|---|
| `context_entity` | builder/infer/REST | `person:Gad`, `place:*`, seeded `device:*`, discovered `device:<mac>` (BELL879, Govee_H6054, 17:70:…) |
| `context_relationship` | builder/infer/REST | `person:Gad located_in …`, `… observed_by phone:gadgad` |
| `context_state` | infer/REST + node estimators | `occupancy.v1`, `presence.v1`, `activity.v1`, `location.geo.v1`, `prediction` |
| `context_evidence` | node uplinks + REST | all rows in the key table above |
| `context_event` | apply pipeline | `entered`/`exited`/`changed` transitions |
| `node_room` | node WS (`PUT /v1/nodes/{id}/room` cache-through) | `room/v1` docs from the app room editor |
| `space`/`zone`/`device_placement` | REST/portal floor plan editor | rooms, zones, device poses |

## Reproduce

```bash
# needs OPENAI_API_KEY + WHISPY_API_KEY (scoped key works) in env
python acceptance\_llm_prompt_capture.py     # → acceptance/llm_prompt_samples.json
python acceptance\_docs_data_probe.py        # raw store dump used on this page
```

`llm_prompt_capture.py` imports the real server modules (`server.v1.context_builder`,
`server.v1.context_infer`) and calls the same `_SYSTEM_PROMPT`,
`_map_tool_schema()` / `_ctx_tool_schema()` functions the API uses — the
captured prompts are byte-identical to production.
