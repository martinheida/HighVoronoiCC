#!/usr/bin/env bash
set -euo pipefail

# Installs Qhull only as an external benchmark dependency.
# HighVoronoiCC itself is not modified and does not link against Qhull.

if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: This installer currently supports Debian/Ubuntu/Linux Mint (apt-get)."
    echo "Install the Qhull development package for your distribution manually."
    exit 1
fi

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    SUDO=()
else
    if ! command -v sudo >/dev/null 2>&1; then
        echo "ERROR: sudo is required when this script is not run as root."
        exit 1
    fi
    SUDO=(sudo)
fi

echo "============================================================"
echo "INSTALLING QHULL BENCHMARK DEPENDENCY"
echo "============================================================"
echo "Packages: libqhull-dev qhull-bin"
echo

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y libqhull-dev qhull-bin

echo
echo "============================================================"
echo "VERIFYING"
echo "============================================================"

if [[ ! -f /usr/include/libqhullcpp/Qhull.h ]]; then
    echo "ERROR: Qhull C++ header was not found after installation."
    exit 1
fi

if command -v pkg-config >/dev/null 2>&1; then
    echo "qhullcpp version: $(pkg-config --modversion qhullcpp 2>/dev/null || echo unknown)"
    echo "qhull_r version:  $(pkg-config --modversion qhull_r 2>/dev/null || echo unknown)"
    echo "link flags:       $(pkg-config --libs qhullcpp qhull_r 2>/dev/null || echo '-lqhullcpp -lqhull_r')"
fi

if command -v qhull >/dev/null 2>&1; then
    qhull -V 2>&1 | head -n 1 || true
fi

echo
echo "Qhull benchmark dependency is installed."
echo "HighVoronoiCC source files were not changed."
