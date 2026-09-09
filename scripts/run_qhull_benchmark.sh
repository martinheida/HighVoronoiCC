#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HighVoronoiCC - open-domain 5D comparison against Qhull
#
# Compares:
#   - ClassicRaycast
#   - InRangeRaycast
#   - CombinedRaycast (default Robust)
#   - CombinedRaycast Fast/Lossy
#   - Qhull/libqhullcpp Voronoi (v Qbb Qc Qz)
#
# Run from the HighVoronoiCC project root:
#   ./run_qhull_benchmark.sh
#
# Optional overrides:
#   CXX=g++ ./run_qhull_benchmark.sh
#   NODES=500,1000 REPEATS=3 ./run_qhull_benchmark.sh
#   SEARCH=provider ./run_qhull_benchmark.sh
#   METHOD=qhull ./run_qhull_benchmark.sh
#   METHOD=combined COMBINED_FALLBACK=inrange ./run_qhull_benchmark.sh
# ============================================================

ROOT="$(pwd)"

SRC="${SRC:-$ROOT/benchmarks/benchmark_compute_voronoi_5d_qhull.cpp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_benchmark_qhull}"
BIN="${BIN:-$BUILD_DIR/benchmark_compute_voronoi_5d_qhull}"

CXX="${CXX:-clang++}"

NODES="${NODES:-1000,2000}"
REPEATS="${REPEATS:-5}"
SEARCH="${SEARCH:-copy}"
METHOD="${METHOD:-all}"
COMBINED_FALLBACK="${COMBINED_FALLBACK:-auto}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CSV="${CSV:-$ROOT/qhull_benchmark_${TIMESTAMP}.csv}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: benchmark source not found:"
    echo "  $SRC"
    echo
    echo "Expected the benchmark at:"
    echo "  $ROOT/benchmarks/benchmark_compute_voronoi_5d_qhull.cpp"
    echo "or specify SRC explicitly."
    exit 1
fi

if [[ ! -d "$ROOT/include/highvoronoi" ]]; then
    echo "ERROR: current directory does not look like the HighVoronoiCC project root."
    echo "Expected:"
    echo "  $ROOT/include/highvoronoi"
    exit 1
fi

if [[ ! -f /usr/include/libqhullcpp/Qhull.h ]]; then
    echo "ERROR: Qhull development headers are not installed."
    echo "Run:"
    echo "  ./install_qhull_benchmark.sh"
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "============================================================"
echo "COMPILING"
echo "============================================================"
echo "Compiler:   $CXX"
echo "Source:     $SRC"
echo "Binary:     $BIN"
echo "Qhull libs: -lqhullcpp -lqhull_r"
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
    -lqhullcpp \
    -lqhull_r \
    -o "$BIN"

echo
echo "============================================================"
echo "RUNNING BENCHMARK"
echo "============================================================"
echo "Nodes:      $NODES"
echo "Repeats:    $REPEATS"
echo "Domain:     open"
echo "HV Search:  $SEARCH"
echo "Method:     $METHOD"
echo "Combined FB:$COMBINED_FALLBACK"
echo "Qhull opts: v Qbb Qc Qz"
echo "CSV:        $CSV"
echo

"$BIN" \
    --nodes "$NODES" \
    --repeats "$REPEATS" \
    --search "$SEARCH" \
    --method "$METHOD" \
    --combined-fallback "$COMBINED_FALLBACK" \
    --csv "$CSV"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Results:"
echo "  $CSV"
