# thothHUB

**thothHUB** is the web portal for the Thothcraft platform, at
`https://hub.thothcraft.com`. It's the front end for Brain — where you pair
devices, manage models, browse captures, and watch your fleet.

## What you can do

- **Pair devices** — link a Thoth node to your account (`thoth pair`, then
  confirm in thothHUB).
- **See your fleet** — every device, its sensors, and whether it's online.
- **Manage models** — register `whispy-model/v1` packages and deploy them
  to nodes.
- **Browse captures** — synchronized sensor windows recorded by your nodes.
- **Watch predictions** — the live output of each node's SMA loop.

## Relationship to the API

thothHUB consumes the same [Brain v1 API](/brain/) you can call directly.
Anything you do in the portal — listing devices, starting a capture,
deploying a model — maps to a `/v1` endpoint, so you can automate it with
the Whispy `Client` or any HTTP client.

## Accounts & plans

Your account's plan sets device and storage entitlements, enforced by Brain.
See `GET /v1/account` for your current entitlements.
