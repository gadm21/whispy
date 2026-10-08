# Deploying docs.thothcraft.com

This directory is a **self-contained static docs site** built on
[docsify](https://docsify.js.org). There is **no build step** — the markdown
files are rendered client-side. Deploy the folder as-is.

## Vercel

`thothcraft.com` DNS is hosted on Vercel, so the `docs` record is created
automatically once the domain is attached — no manual DNS.

1. **New Project** → import `gadm21/whispy`.
2. Set **Root Directory** to `docs`. `vercel.json` in this folder pins
   framework `null` + output `.` — leave build/install commands empty.
3. **Deploy**, then **Settings → Domains** → add **`docs.thothcraft.com`**.
   Vercel provisions the cert and the DNS record.

Because routing is client-side (`#/path`), no rewrites are needed.

## Any static host

The site is plain static files — `index.html` plus markdown. Serve the
`docs/` directory with any static file host (Netlify, Cloudflare Pages,
GitHub Pages, nginx). `.nojekyll` is included for GitHub Pages.

## Local preview

```bash
cd docs
python -m http.server 8419
# open http://localhost:8419
```

## Editing

- Pages are Markdown; section index pages are `README.md`.
- `_sidebar.md` and `_navbar.md` control navigation.
- `index.html` holds the docsify config (`window.$docsify`) and theme CSS.
- Static assets live in `public/`.
