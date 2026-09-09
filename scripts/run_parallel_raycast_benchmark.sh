#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HighVoronoiCC - 5D parallel benchmark
#
# Default matrix:
#   serial:  MeshThreading=SingleThread, CastThreading=SingleThread
#   cast x2: MeshThreading=SingleThread, CastThreading=MultiThread(2)
#   cast x4: MeshThreading=SingleThread, CastThreading=MultiThread(4)
#   mesh x2: MeshThreading=MultiThread(2), CastThreading=SingleThread
#   mesh x4: MeshThreading=MultiThread(4), CastThreading=SingleThread
#
# Combined is Robust by default in the benchmark. Fast/Lossy remains selectable.
#
# Run from the HighVoronoiCC project root:
#   ./run_parallel_raycast_benchmark.sh
#
# Useful overrides:
#   NODES=2000 DOMAIN=open ./run_parallel_raycast_benchmark.sh
#   PARALLEL_AXIS=mesh ./run_parallel_raycast_benchmark.sh
#   PARALLEL_AXIS=cast ./run_parallel_raycast_benchmark.sh
#   RAYCAST=inrange ./run_parallel_raycast_benchmark.sh
#   RAYCAST=both ./run_parallel_raycast_benchmark.sh
#   COMBINED_MODE=fast ./run_parallel_raycast_benchmark.sh
# ============================================================

ROOT="$(pwd)"

SRC="${SRC:-$ROOT/benchmarks/benchmark_compute_voronoi_5d_parallel_raycast.cpp}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_benchmark}"
BIN="${BIN:-$BUILD_DIR/benchmark_compute_voronoi_5d_parallel_raycast}"

CXX="${CXX:-clang++}"

NODES="${NODES:-1000,2000}"
THREADS="${THREADS:-1,2,4}"
PARALLEL_AXIS="${PARALLEL_AXIS:-both}"
REPEATS="${REPEATS:-5}"
DOMAIN="${DOMAIN:-both}"
SEARCH="${SEARCH:-copy}"
RAYCAST="${RAYCAST:-combined}"
COMBINED_MODE="${COMBINED_MODE:-robust}"
COMBINED_FALLBACK="${COMBINED_FALLBACK:-auto}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CSV="${CSV:-$ROOT/parallel_raycast_benchmark_${TIMESTAMP}.csv}"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: benchmark source not found:"
    echo "  $SRC"
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
echo "Compiler:         $CXX"
echo "Source:           $SRC"
echo "Binary:           $BIN"
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
echo "Nodes:            $NODES"
echo "Thread counts:    $THREADS"
echo "Parallel axis:    $PARALLEL_AXIS"
echo "Repeats:          $REPEATS"
echo "Domain:           $DOMAIN"
echo "Search:           $SEARCH"
echo "Raycast:          $RAYCAST"
echo "Combined mode:    $COMBINED_MODE"
echo "Combined fallback:$COMBINED_FALLBACK"
echo "CSV:              $CSV"
echo

"$BIN" \
    --nodes "$NODES" \
    --threads "$THREADS" \
    --parallel-axis "$PARALLEL_AXIS" \
    --repeats "$REPEATS" \
    --domain "$DOMAIN" \
    --search "$SEARCH" \
    --raycast "$RAYCAST" \
    --combined-mode "$COMBINED_MODE" \
    --combined-fallback "$COMBINED_FALLBACK" \
    --csv "$CSV"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Results:"
echo "  $CSV"
