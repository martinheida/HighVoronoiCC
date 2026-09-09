#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${repo_root}/build/docs"
open_after_build=0

usage() {
    cat <<'EOF'
Usage: scripts/build_manual.sh [--open] [--build-dir PATH]

Build the Doxygen HTML manual/API reference without building tests/examples.

Options:
  --open            Open the generated index.html with xdg-open when available.
  --build-dir PATH  Use a custom CMake build directory.
  -h, --help        Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --open)
            open_after_build=1
            shift
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || { echo "--build-dir requires a path" >&2; exit 2; }
            build_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for command in cmake doxygen; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        echo "On Linux Mint/Debian/Ubuntu, install documentation prerequisites with:" >&2
        echo "  sudo apt install doxygen graphviz libeigen3-dev libboost-dev" >&2
        exit 1
    fi
done

cmake \
    -S "$repo_root" \
    -B "$build_dir" \
    -DHIGHVORONOI_BUILD_DOCS=ON \
    -DHIGHVORONOI_BUILD_TESTS=OFF \
    -DHIGHVORONOI_BUILD_EXAMPLES=OFF \
    -DHIGHVORONOI_BUILD_BENCHMARKS=OFF

cmake --build "$build_dir" --target docs

index_file="${build_dir}/manual/html/index.html"
if [[ ! -f "$index_file" ]]; then
    echo "Documentation build completed but index file is missing: $index_file" >&2
    exit 1
fi

echo
echo "HighVoronoi manual generated at:"
echo "  $index_file"

if ((open_after_build)); then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$index_file" >/dev/null 2>&1 &
    else
        echo "xdg-open is not available; open the file above in your browser." >&2
    fi
fi
