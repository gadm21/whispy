# Deploying docs.thothcraft.com

This directory is a **self-contained static docs site** built on
[docsify](https://docsify.js.org). There is **no build step** — the markdown
files are rendered client-side. Deploy the folder as-is.

## Vercel

1. Import the `whispy` repo (or a dedicated docs repo containing this
   folder).
2. Set **Root Directory** to `docs`.
3. Framework preset: **Other**. Build command: *none*. Output directory:
   *leave empty* (the folder is served directly).
4. Add the domain **`docs.thothcraft.com`** to the project.

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
