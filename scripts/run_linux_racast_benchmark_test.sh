#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HighVoronoiCC - 5D Raycast benchmark
#
# Compares:
#   - ClassicRaycast
#   - InRangeRaycast
#   - CombinedRaycast
#
# Run from the HighVoronoiCC project root:
#   ./run_raycast_benchmark.sh
#
# Optional overrides:
#   CXX=g++ ./run_raycast_benchmark.sh
#   NODES=500,1000 REPEATS=3 ./run_raycast_benchmark.sh
#   DOMAIN=open SEARCH=provider ./run_raycast_benchmark.sh
#   RAYCAST=combined-fast COMBINED_FALLBACK=auto ./run_raycast_benchmark.sh
# ============================================================

ROOT="$(pwd)"

SRC="${SRC:-$ROOT/benchmarks/benchmark_compute_voronoi_5d_three_raycasts.cpp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_benchmark}"
BIN="${BIN:-$BUILD_DIR/benchmark_compute_voronoi_5d_three_raycasts}"

CXX="${CXX:-clang++}"

NODES="${NODES:-1000,2000}"
REPEATS="${REPEATS:-5}"
DOMAIN="${DOMAIN:-both}"
SEARCH="${SEARCH:-copy}"
RAYCAST="${RAYCAST:-all}"
COMBINED_FALLBACK="${COMBINED_FALLBACK:-auto}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CSV="${CSV:-$ROOT/raycast_benchmark_${TIMESTAMP}.csv}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: benchmark source not found:"
    echo "  $SRC"
    echo
    echo "Copy benchmark_compute_voronoi_5d_three_raycasts.cpp into the"
    echo "project root, or specify SRC explicitly:"
    echo "  SRC=/path/to/benchmark_compute_voronoi_5d_three_raycasts.cpp $0"
    exit 1
fi

if [[ ! -d "$ROOT/include/highvoronoi" ]]; then
    echo "ERROR: current directory does not look like the HighVoronoiCC project root."
    echo "Expected:"
    echo "  $ROOT/include/highvoronoi"
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "============================================================"
echo "COMPILING"
echo "============================================================"
echo "Compiler:   $CXX"
echo "Source:     $SRC"
echo "Binary:     $BIN"
echo

"$CXX" \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -march=native \
    -pthread \
    -I"$ROOT/include" \
    -I/usr/include/eigen3 \
    "$SRC" \
    -o "$BIN"

echo
echo "============================================================"
echo "RUNNING BENCHMARK"
echo "============================================================"
echo "Nodes:      $NODES"
echo "Repeats:    $REPEATS"
echo "Domain:     $DOMAIN"
echo "Search:     $SEARCH"
echo "Raycast:    $RAYCAST"
echo "Combined FB:$COMBINED_FALLBACK"
echo "CSV:        $CSV"
echo

"$BIN" \
    --nodes "$NODES" \
    --repeats "$REPEATS" \
    --domain "$DOMAIN" \
    --search "$SEARCH" \
    --raycast "$RAYCAST" \
    --combined-fallback "$COMBINED_FALLBACK" \
    --csv "$CSV"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Results:"
echo "  $CSV"
