# Examples and tests

The documentation now distinguishes **user-level examples** from **Level-3 implementation/configuration references**.

## Convenience API regression

The new public-facade smoke test is:

```text
tests/public/test_convenience_api.cpp
```

It checks the intended defaults and compiles/runs the basic Level-1/Level-2 workflows for ordinary/HighVoronoi construction and the integration facade:

```text
double + uint32_t
Combined default ray casting
SingleThread + DirectHash
MultiThread + ReadWriteLock + StaticHash<16>
compute / refine / remove
Level-2 alternate ray-caster selection
Integration Level-1 FastPolygon default
Persistent clean re-run -> zero updated cells
Scalar function integration
Integration Level-2 Polygon selection
HighVoronoi integration facade
```

This is the first place to look when changing the convenience API.

## Existing executable examples

The existing examples remain valuable, but many intentionally spell out Level-3 types so that a particular algorithmic/configuration choice is visible and testable.

| use case | reference |
|---|---|
| bounded general-position construction | `examples/general_position.cpp` |
| exact Cartesian degeneracy / InRange path | `examples/cubic_grid.cpp` |
| ordinary incremental refine/remove | `examples/incremental/voronoi_refine_remove.cpp` |
| non-periodic HighVoronoi refine/remove | `examples/incremental/high_voronoi_refine_remove.cpp` |
| periodic HighVoronoi closure/refinement | `examples/incremental/periodic_high_voronoi_refine.cpp` |
| cast-worker parallelism | `examples/parallel/parallel_cast.cpp` |
| mesh-level parallelism | `examples/parallel/parallel_mesh.cpp` |
| both construction axes | `examples/parallel/parallel_combined.cpp` |
| spherical refine/remove | `examples/spherical/spherical_refine_remove.cpp` |
| Polygon vs FastPolygon integration | `examples/integration/fast_polygon.cpp` |

Do not interpret the explicit boilerplate in a Level-3 example as a requirement for a normal application. For ordinary/HighVoronoi use, start with [Getting started](getting_started.md).

## Complete developer test inventory

For a source-level inventory of every registered test executable and the concrete setup/actions/results checked by its test functions, see [Test overview (developers only)](test_overview_developers.md). This is the page to use when deciding whether a behavior already has regression coverage or where a new regression belongs.

## Tests as semantic references

Tests are useful when a behavior is more precise than a tutorial example can reasonably be. Important groups include:

- `tests/algorithm/compute/` — construction, range equivalence, parallelism, degeneracy, completeness;
- `tests/algorithm/raycast/` — ray-caster behavior;
- `tests/workflows/incremental/` — ordinary refine/remove semantics;
- `tests/workflows/high_voronoi/` — visible/reference mapping, periodic closure, incremental equivalence;
- `tests/integration/` — integral storage/algorithms/parallel/incremental behavior;
- `tests/storage/` — database and hash/container semantics.

The CMake labels allow focused runs, e.g.:

```bash
ctest --test-dir build/release -L api --output-on-failure
ctest --test-dir build/release -L compute --output-on-failure
ctest --test-dir build/release -L highvoronoi --output-on-failure
ctest --test-dir build/release -L integrals --output-on-failure
```
