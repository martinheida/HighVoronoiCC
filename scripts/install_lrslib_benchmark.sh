#!/usr/bin/env bash
set -euo pipefail

# Installs the lrslib command-line program used only by the standalone
# HighVoronoi integration benchmark. It does not add lrslib to the
# HighVoronoi library or its build dependencies.

if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: this installer expects a Debian/Ubuntu/Linux-Mint apt system."
    echo "Install lrslib with your distribution package manager instead."
    exit 1
fi

echo "Installing lrslib benchmark dependency..."
sudo apt-get update
sudo apt-get install -y lrslib

echo
if ! command -v lrs >/dev/null 2>&1; then
    echo "ERROR: package installation completed, but 'lrs' is not on PATH."
    exit 1
fi

echo "lrslib executable: $(command -v lrs)"
if command -v dpkg-query >/dev/null 2>&1; then
    VERSION="$(dpkg-query -W -f='${Version}' lrslib 2>/dev/null || true)"
    if [[ -n "$VERSION" ]]; then
        echo "lrslib package:    $VERSION"
    fi
fi

echo "Ready for ./run_integration_lrslib_benchmark.sh"
