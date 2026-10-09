# Thothcraft Docs

**This documentation is published at https://docs.thothcraft.com — start there.**

Rendered with docsify (`index.html` entry point). The homepage source is
`home.md` — the site root `README.md` is intentionally not served by Vercel,
so the docsify `homepage` setting points at `home.md` instead.

| Piece | What it is |
|---|---|
| **[Whispy](whispy/README.md)** | The standalone sensing SDK — sensors, streams, synchronization, windows, processors, datasets, actuators. |
| **[Thoth](thoth/README.md)** | The edge node — a CLI + daemon running the sense → predict → act loop. |
| **[Brain](brain/README.md)** | The cloud control + data plane — a versioned `/v1` API. |
| **[thothHUB](hub.md)** | The web portal for devices, models, captures, and fleet. |

Edit markdown files here and push to `main` — the site redeploys
automatically. See `DEPLOY.md`.
