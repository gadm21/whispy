#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/packages/thothcraft-cli/pyproject.toml")"
stage="$(mktemp -d)"
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/DEBIAN" "$stage/opt/thothcraft/packages" "$stage/usr/bin" "$stage/usr/lib/systemd/user"
cp -R "$root/packages/thothcraft-sdk" "$root/packages/thothcraft-cli" "$stage/opt/thothcraft/packages/"
find "$stage/opt/thothcraft/packages" -type d \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \) -prune -exec rm -rf -- {} +
sed "s/@VERSION@/$version/g" "$root/packaging/debian/control" > "$stage/DEBIAN/control"
install -m 755 "$root/packaging/debian/postinst" "$root/packaging/debian/prerm" "$stage/DEBIAN/"
install -m 644 "$root/packaging/thothcraft.service" "$stage/usr/lib/systemd/user/"
ln -s /opt/thothcraft/venv/bin/thothcraft "$stage/usr/bin/thothcraft"
dpkg-deb --root-owner-group --build "$stage" "$root/thothcraft-cli_${version}_all.deb"
