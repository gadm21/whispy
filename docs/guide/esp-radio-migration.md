# ESP serial observation migration

The existing `whispy_sensor_csi.CsiSensorAdapter` remains the single owner of
each receiver's serial port. Do not open a second radio adapter on that port.

The adapter continues to emit the existing `payload_type` and CSI payload
(`encoding`, base64 `data`, `iq`, `rssi`, `mac`, `seq`, `n_subcarriers`). It now
adds `payload.observation`, with `format: radio-observation/v1`.

| Record | SensorSample.sensor_type | Observation stream |
| --- | --- | --- |
| BLE_DATA | radio.ble_rssi | ble_rssi/v1 |
| WIFI_DATA | radio.wifi_rssi | wifi_rssi/v1 |
| SELF_DATA | radio.self | radio_self/v1 |
| HEALTH_DATA | radio.health | radio_health/v1 |
| CSI_DATA, configured direct source | radio.wifi_csi_direct | wifi_csi_direct/v1 |
| CSI_DATA, configured AP source | radio.wifi_csi_ap | wifi_csi_ap/v1 |
| CSI_DATA, unknown/ambiguous source | wifi_csi | wifi_csi_unclassified/v1 |

The hardware descriptor still represents one physical receiver and retains
its existing `wifi_csi` modality and stable sensor ID. Mixed record types do
not create account devices. Consumers should use `payload_type == csi_raw`
when decoding CSI and the observation stream when separating RF paths.

## Source attribution

Supply these optional fields in an existing JSON `WHISPY_CSI_SOURCES` source
entry, or in the adapter connection config:

- `node_id`: containing logical node ID.
- `component_id`: this internal receiver's identity.
- `direct_source_mac`: verified direct transmitter MAC.
- `direct_source_component`: transmitter component identity, when known.
- `ap_bssid`: configured AP source MAC/BSSID.

Unconfigured sources remain unclassified. Equal direct/AP addresses are
ambiguous and never classified as either path. Source attribution records
configuration provenance; it is not a physical-path verification result.
The deployed Chen/April adapters have not been given a new source mapping.

## Time, sequence and quality

`host_received_at` is Unix seconds. `firmware_tick` has an explicit `us` or
`ms` unit and remains independent of the host clock. C6's 32-bit receive
timestamp is normalized to unsigned without fabricating an epoch or unwrap.

The inspected C6 firmware fills its legacy `seq` field with the receive
timestamp. That legacy field remains unchanged for compatibility.
`transmit_sequence` is null: packet loss cannot be inferred from that tick.

Known C6/C5 and legacy ESP32 CSV layouts expose channel, firmware timestamp,
declared CSI length and first-word validity. Empty, odd, noninteger or
out-of-int8-range IQ vectors are rejected. Length mismatches and invalid
first words remain explicit quality issues; they are not silently repaired.
The C5 optional packed 12-bit IQ format is not supported by this int8 path.

`quality.status=reported` describes metadata availability, not measurement
accuracy. There is no runtime ground-truth accuracy estimate.

## Firmware output counters

The C6 receiver queue change adds:

`HEALTH_DATA,uptime_ms,queue_depth,output_dropped,csi_enqueued`

`output_dropped` counts queue failures/oversize output across record types.
`csi_enqueued` counts CSI records accepted into the output queue. Neither
counter measures RF packet loss or successful host delivery. Counters reset
on boot and unsigned counters can wrap. The queue is bounded by
`SCAN_QUEUE_LEN`; each record reserves a 256-byte header and 1024-byte IQ
buffer. At the default 48 slots, queue storage is about 62 KiB. Heap headroom,
throughput and drop rates still require a hardware test before deployment.

This change moves C6 CSI serialization out of the Wi-Fi callback into the
existing scan output task. It does not implement the requested coexistence
scheduler, framed command protocol, Zigbee, AP reception, or role assignment.

## Discovery and deployment

Discovery now examines input until its time deadline, retains bounded noise,
and recognizes valid receiver identity/scan records without requiring an
active CSI transmitter. It no longer exhausts an 8 KiB total boot-log budget.

The host changes were deployed with backups to Chen and April after checking
that neither had an active capture. April subsequently exposed its previously
missing CSI receiver through the node API. No firmware was flashed in this
change. Existing direct source traffic was measured; AP CSI, Zigbee and
coexistence acceptance remain unverified.
