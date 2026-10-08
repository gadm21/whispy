"""Sync per-repo markdown docs into the docsify site sections.

Each source repo keeps its canonical docs under ``<repo>/docs/**.md``.
This script mirrors them into ``whispy/docs/<section>/`` so docsify can
serve everything from one deploy.

Synced files are tracked in ``docs/<section>/.sync-manifest.json`` — files
authored directly in the site (e.g. ``thoth/cli.md``) are never touched,
and synced files deleted upstream are removed here.

Usage (local, sibling checkouts assumed next to this repo)::

    python tools/sync_docs.py            # sync
    python tools/sync_docs.py --check    # exit 1 if drift (CI gate)

Usage (CI, arbitrary checkout locations)::

    python tools/sync_docs.py --source thoth=.docs-src/thoth/docs \
        --source brain=.docs-src/Brain/docs \
        --source portal=.docs-src/ResearchPortal/docs
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
MANIFEST = ".sync-manifest.json"

# section -> (sibling checkout dir name, filenames to skip)
SOURCES: dict[str, tuple[str, set[str]]] = {
    "thoth": ("thoth", set()),
    "brain": (
        "Brain",
        # internal ops docs — keep out of the public site
        {"STRIPE_RAILWAY_SETUP.md", "SUPABASE_DATABASE_REVIEW.md"},
    ),
    "portal": ("ResearchPortal", set()),
}


def _source_dir(section: str, overrides: dict[str, Path], siblings: Path) -> Path:
    if section in overrides:
        return overrides[section]
    repo, _ = SOURCES[section]
    return siblings / repo / "docs"


def _sync_section(section: str, src: Path, check: bool) -> list[str]:
    dst = DOCS_ROOT / section
    _, skip = SOURCES[section]
    manifest_path = dst / MANIFEST
    previous: list[str] = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text()).get("files", [])
        except (ValueError, KeyError):
            previous = []

    incoming: dict[str, Path] = {}
    if src.is_dir():
        for f in sorted(src.rglob("*.md")):
            if f.name in skip or f.name.startswith("_"):
                continue
            incoming[f.relative_to(src).as_posix()] = f
    else:
        print(f"warning: {src} not found, clearing managed files", file=sys.stderr)

    actions: list[str] = []
    for rel, f in incoming.items():
        target = dst / rel
        if not target.exists() or not filecmp.cmp(f, target, shallow=False):
            actions.append(("update" if target.exists() else "add") + f" {section}/{rel}")
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)

    for rel in previous:
        if rel not in incoming and (dst / rel).exists():
            actions.append(f"remove {section}/{rel}")
            if not check:
                (dst / rel).unlink()

    if not check:
        dst.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"files": sorted(incoming)}, indent=2) + "\n")
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    ap.add_argument(
        "--siblings",
        type=Path,
        default=REPO_ROOT.parent,
        help="dir containing the sibling repo checkouts (default: repo parent)",
    )
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="SECTION=PATH",
        help="override a section's source docs dir (repeatable)",
    )
    args = ap.parse_args()

    overrides: dict[str, Path] = {}
    for item in args.source:
        section, _, path = item.partition("=")
        if not path:
            ap.error(f"--source expects SECTION=PATH, got {item!r}")
        overrides[section] = Path(path).resolve()

    drift = False
    for section in SOURCES:
        for action in _sync_section(section, _source_dir(section, overrides, args.siblings.resolve()), args.check):
            drift = True
            print(action)

    if args.check:
        print("docs in sync" if not drift else "docs out of sync")
        return 1 if drift else 0
    print("done" if drift else "already in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
