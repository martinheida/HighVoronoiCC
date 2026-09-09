#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HighVoronoiCC - 5D bounded integration benchmark vs lrslib
#
# One bounded [0,1]^5 Voronoi mesh is computed before timing.
# Per repeat the benchmark measures:
#   1. Polygon:     volumes + interfaces + readback
#   2. FastPolygon: volumes + interfaces + readback
#   3. lrslib conversion to rationalized V-representation files
#   4. lrslib volume, cell-by-cell, until cumulative lrslib time
#      reaches the Polygon time of that repeat.
#
# lrslib is used as its documented `lrs` executable. HighVoronoi
# itself has no lrslib dependency.
#
# Run from HighVoronoiCC project root:
#   ./run_integration_lrslib_benchmark.sh
#
# Optional overrides:
#   CXX=g++ ./run_integration_lrslib_benchmark.sh
#   NODES=1000 REPEATS=3 ./run_integration_lrslib_benchmark.sh
#   RATIONAL_SCALE=100000000000000 ./run_integration_lrslib_benchmark.sh
#   LRS_TRACE=1 REPEATS=1 ./run_integration_lrslib_benchmark.sh
# ============================================================

ROOT="$(pwd)"

SRC="${SRC:-$ROOT/benchmarks/benchmark_integrate_voronoi_5d_lrslib.cpp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_benchmark_integration_lrslib}"
BIN="${BIN:-$BUILD_DIR/benchmark_integrate_voronoi_5d_lrslib}"

CXX="${CXX:-clang++}"
NODES="${NODES:-1000}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-0x485642454e434835}"
RATIONAL_SCALE="${RATIONAL_SCALE:-100000000000000}"
LRS_BIN="${LRS_BIN:-$(command -v lrs 2>/dev/null || true)}"
LRS_TRACE="${LRS_TRACE:-0}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CSV="${CSV:-$ROOT/integration_lrslib_benchmark_${TIMESTAMP}.csv}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: benchmark source not found:"
    echo "  $SRC"
    echo
    echo "Expected:"
    echo "  $ROOT/benchmarks/benchmark_integrate_voronoi_5d_lrslib.cpp"
    exit 1
fi

if [[ ! -d "$ROOT/include/highvoronoi" ]]; then
    echo "ERROR: current directory does not look like the HighVoronoiCC project root."
    echo "Expected:"
    echo "  $ROOT/include/highvoronoi"
    exit 1
fi

if [[ -z "$LRS_BIN" || ! -x "$LRS_BIN" ]]; then
    echo "ERROR: lrslib executable 'lrs' was not found."
    echo "Run:"
    echo "  ./install_lrslib_benchmark.sh"
    echo "or specify an executable explicitly:"
    echo "  LRS_BIN=/path/to/lrs $0"
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "============================================================"
echo "COMPILING"
echo "============================================================"
echo "Compiler:      $CXX"
echo "Source:        $SRC"
echo "Binary:        $BIN"
echo "lrslib binary: $LRS_BIN"
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
echo "Nodes:             $NODES"
echo "Repeats:           $REPEATS"
echo "Domain:            bounded [0,1]^5"
echo "Seed:              $SEED"
echo "Rational scale:    $RATIONAL_SCALE"
echo "lrslib binary:     $LRS_BIN"
echo "lrslib trace:      $LRS_TRACE"
echo "Cutoff:            cumulative lrslib volume time >= Polygon time"
echo "CSV:               $CSV"
echo

RUN_ARGS=(
    --nodes "$NODES"
    --repeats "$REPEATS"
    --seed "$SEED"
    --rational-scale "$RATIONAL_SCALE"
    --lrs-binary "$LRS_BIN"
    --csv "$CSV"
)
if [[ "$LRS_TRACE" == "1" ]]; then
    RUN_ARGS+=(--lrs-trace)
fi

"$BIN" "${RUN_ARGS[@]}"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Results:"
echo "  $CSV"
