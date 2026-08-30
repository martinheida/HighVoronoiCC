# HighVoronoiCC

HighVoronoiCC is an experimental C++17 implementation of a local algorithm for the construction and incremental update of high-dimensional Voronoi diagrams.

The project is a redesign and C++ port of the ideas implemented in **HighVoronoi.jl** and developed mathematically in:

**Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025.**  
DOI: `10.20347/WIAS.PREPRINT.3197`

The core algorithm constructs Voronoi diagrams locally. Starting from known vertices, it enumerates the incident Voronoi edges and follows them by nearest-neighbour based ray casting. The implementation supports generators both in general and non-general position.

The C++ implementation now contains two related construction layers:

- `VoronoiMesh` + `ComputeVoronoi` for ordinary Euclidean Voronoi diagrams;
- `HighVoronoiMesh` + `ComputeHighVoronoi` for persistent incremental updates and periodic reference-node closure.

## Status

HighVoronoiCC is currently **experimental / alpha software**. APIs may still change.

The currently validated construction path supports:

- bounded Euclidean domains with planar boundaries;
- unbounded Euclidean construction with persistent infinite Voronoi edges;
- paired periodic planar boundaries through `HighVoronoiMesh`;
- partially periodic domains, e.g. periodic in selected coordinate directions and non-periodic in the others;
- external visible-domain output while periodic reference nodes remain internal;
- generators in general position;
- degenerate / non-general configurations;
- serial Voronoi construction;
- incremental insertion with `RefineVoronoi`;
- incremental removal and local closure with `RemoveVoronoi`;
- batched visible insertion/removal through `ComputeHighVoronoi`;
- periodic refinement after further visible-node insertion;
- serial and parallel ordinary construction;
- parallel construction using independent mesh branches;
- parallel ray-casting workers inside one mesh branch;
- combined mesh- and cast-level parallelism;
- KD-tree nearest-neighbour search through the bundled nanoflann backend;
- brute-force nearest-neighbour search as a reference backend;
- fixed-dimensional and runtime-dimensional point/node infrastructure;
- stored, computed and hybrid node access;
- computed/hybrid mesh engines including the current cuboid engine;
- compact configurable index and scalar types;
- configurable queue- and edge-hash containers with synchronization policies;
- geometric consistency verification;
- topological edge-completeness verification including infinite edges;
- visible-cell completeness verification for periodic HighVoronoi output.

The current regression suite includes ordinary bounded, unbounded, degenerate, parallel and incremental cases as well as periodic HighVoronoi comparisons against independently constructed explicit periodic reference meshes.

Not yet implemented as a finished public construction path are spherical Voronoi diagrams, the fast quasi-periodic copy/modify/paste algorithm from the reference paper, and volume/interface/quadrature algorithms.

## Public headers

Prefer the narrowest public entry header that matches the task.

### Ordinary Euclidean Voronoi

```cpp
#include <highvoronoi/voronoi.hpp>
```

This exposes the ordinary `VoronoiMesh` construction path, including `ComputeVoronoi`, `RefineVoronoi`, `RemoveVoronoi`, search/raycast selection and validation.

### Incremental / periodic HighVoronoi

```cpp
#include <highvoronoi/high_voronoi.hpp>
```

This exposes `HighVoronoiMesh`, `ComputeHighVoronoi`, `VisibleFirstMesh` and the validation helpers needed for persistent incremental and periodic workflows.

### Optional computed mesh engines

```cpp
#include <highvoronoi/mesh_engines.hpp>
```

This exposes the optional computed/hybrid mesh-engine infrastructure such as `CuboidMeshEngine`.

### Complete convenience umbrella

```cpp
#include <highvoronoi/highvoronoi.hpp>
```

The umbrella imports all currently supported public modules.

Shared low-level public types are organized behind `core.hpp`, `database.hpp`, `parameters.hpp` and `version.hpp`. Applications normally do not need to include the corresponding physical `detail/` implementation headers.

## Ordinary Voronoi construction

The classical construction stack remains:

```text
VoronoiMesh
    |
SearchTree
    |
RayCaster + EdgeIterator
    |
VoronoiWorker
    |
SystematicVoronoi
    |
ComputeVoronoi
```

`ComputeVoronoi` is the complete exhaustive kernel. It is also reused by the incremental and periodic layers rather than being duplicated into separate geometry algorithms.

### Incremental insertion

`RefineVoronoi` appends a NEW block and computes only the NEW cells. Every newly created finite vertex on a fixed boundary contains at least one NEW generator, so the NEW-cell pass already discovers the new topology. Old vertices that have become invalid are then removed.

### Incremental removal

`RemoveVoronoi` is deliberately two-phase:

```text
remove()
    destroy selected generators and incident vertices
    remember the affected surviving cells

compute()
    close the damaged cells against the then-current mesh
```

The two phases may be separated by a refinement. The convenience `compute()` workflow used by the examples performs the required closure.

## HighVoronoi and periodic boundaries

`HighVoronoiMesh` separates the externally visible mesh from the internal geometry required for incremental and periodic computation.

A visible generator receives one stable internal index. Periodic copies are additional **invisible reference nodes** that keep a reference to the visible original and a shift mask over periodic boundary planes.

Conceptually:

```text
visible node x
    |
    +-- x + periodic shift A
    +-- x + periodic shift B
    +-- x + periodic shift A + periodic shift B
    +-- ...
```

The user-facing boundary is the fixed **external boundary**. A separate **internal boundary** begins as a copy of it and may expand only along periodic planes so that newly generated reference nodes remain inside the computation domain.

`ComputeHighVoronoi` repeatedly reuses ordinary `RefineVoronoi` / `RemoveVoronoi` and therefore the normal `ComputeVoronoi` kernel. A successful call integrates all pending visible changes and all currently required periodic references.

The current high-level sequence is:

```text
pending visible deletion
    -> remove incident topology

pending visible insertion
    -> refine NEW visible cells
    -> generate/refine required periodic references until closure

removal closure
    -> recompute damaged surviving cells
    -> generate/refine any further periodic references exposed by that closure

final sparse periodic-boundary repair
    -> recover OLD-only vertices hidden by the bootstrap periodic boundary

mark the resulting internal state integrated
```

### Why a final periodic repair is necessary

Ordinary fixed-boundary refinement relies on an important invariant:

> every genuinely new finite vertex created by adding generators contains at least one NEW generator.

That is why computing only NEW cells is sufficient for `RefineVoronoi`.

The invariant no longer covers a **moving artificial periodic boundary**. During bootstrap periodization, finite vertices on the periodic internal boundary are intentionally not persisted. Once that artificial boundary is moved outward, a previously hidden finite vertex may become part of the final geometry even though all of its ordinary generators are OLD visible generators.

The C++ implementation handles this explicitly:

1. transient bootstrap periodic-boundary vertices are inspected before the boundary moves;
2. persistent inward countervertex signatures are retained as sparse seeds;
3. ordinary periodic reference closure is completed;
4. unchanged serial `ComputeVoronoi` is rerun only on seeded/dirty visible cells using transient repair address lists.

The repair is intentionally narrow. If it discovers a genuinely new vertex containing an invisible periodic reference, it throws instead of hiding an incomplete normal periodization pass.

### Visible output

The internal reference geometry is not exposed as extra public cells.

`HighVoronoiMesh`:

- exposes only visible nodes through its public interface;
- projects invisible reference generators back onto their visible originals;
- reads vertex records only from the visible cell's own address lists;
- translates stored vertex positions by periodic partner shifts into the external visible domain.

For diagnostics requiring a one-to-one index map over all active nodes, `VisibleFirstMesh` exposes all active nodes with the visible nodes as an exact prefix.

## Numerical robustness

The C++ implementation keeps the mathematical structure of the original algorithm but strengthens several numerically sensitive operations.

### Vertex correction

A small equal-distance variance is useful as a residual diagnostic, but it does not by itself bound the positional error of a badly conditioned Voronoi vertex.

The current corrector therefore improves the RayCaster candidate through the local correction system

```text
(p_i - p_0)^T delta
    = 1/2 ( |r - p_i|^2 - |r - p_0|^2 )
```

with the following safeguards:

- rows are normalized by `|p_i - p_0|`;
- the vertex system uses `ColPivHouseholderQR`;
- the ratio between the smallest and largest absolute diagonal pivot is used as a cheap conditioning proxy;
- the factorization is reused for a bounded iterative correction;
- convergence is measured by the relative correction size
  `|delta| / |r_initial - p_0|`;
- poor conditioning or insufficient convergence triggers a direct software-`Float128` repeat;
- there is intentionally no platform-dependent `long double` intermediate precision level.

The normal edge-direction QR remains separate so that the Julia-compatible orientation convention of the edge normal is preserved.

### Degenerate edge enumeration

The degenerate `EdgeIterator` follows the Julia algorithm with one deliberate correctness correction: a primary generator is rejected only after **all** admissible supporting-face completions have been tried. Rejecting it after one failed greedy completion can miss a valid edge and is weaker than the existence criterion required by the mathematical edge characterization.

## Parallel construction

HighVoronoiCC distinguishes two independent threading policies:

```text
MeshThreading
    controls independent reordered mesh branches

CastThreading
    controls geometry workers inside each branch
```

The four combinations are supported:

| MeshThreading | CastThreading | Meaning |
|---|---|---|
| `SingleThread` | `SingleThread` | fully serial reference path |
| `MultiThread(n)` | `SingleThread` | parallel mesh branches |
| `SingleThread` | `MultiThread(n)` | parallel workers inside one branch |
| `MultiThread(n)` | `MultiThread(m)` | both parallelization levels |

Parallel mesh branches are `ReorderedMeshView`s of one persistent mesh/database. A vertex discovered in one branch is queued locally first, persisted once, and then translated into the public numbering of the other branches. Worker-owned edge iterators never cross a branch boundary.

Shared queue, edge-hash and degenerate-edge-cache state uses the combined Voronoi threading policy. This also protects a branch with one local worker from concurrent registrations originating in other mesh branches.

The final sparse periodic-boundary repair is currently serial. Structural incremental changes are likewise performed at explicit safe points between delegated `ComputeVoronoi` phases.

## Persistent infinite edges and completeness

`RayCaster` distinguishes finite and infinite ray results. Infinite rays are persisted by the mesh as

```text
(full supporting edge, origin, direction)
```

using the complete supporting edge in stable internal numbering as the persistent duplicate key. This is important in degenerate configurations where several minimal edges may represent the same geometric unbounded edge.

The mesh exposes persisted rays through `infinite_edges()`.

`verify_mesh_complete(...)` extends ordinary geometric verification with global edge closure:

- each finite geometric edge must occur at exactly two finite endpoints;
- an unbounded edge contributes one finite endpoint and one persistent infinite-edge record;
- repeated minimal-edge representations at the same degenerate vertex are deduplicated by the full supporting edge;
- a third occurrence is treated as an error.

`verify_mesh_complete_cells(...)` applies the same closure logic to a selected cell range and is used for periodic visible-cell validation.

`verify_mesh(...)` remains the cheaper geometric consistency check. It verifies stored data but does not by itself prove completeness.

## Configuration

`include/highvoronoi/parameters.hpp` contains the public compile-time and runtime configuration types for:

- `SingleThread` and `MultiThread`;
- hash generators;
- direct/static/dynamic hash-container modes;
- persistent database queue hashes;
- temporary edge hashes;
- `ClassicRaycast` and `InRangeRaycast`;
- ray-cast, verification and vertex-correction tolerances.

The project also contains `howto_config.hpp`, an editor-oriented configuration guide for the ordinary `VoronoiMesh` / `ComputeVoronoi` path. Incremental and periodic HighVoronoi workflows are currently better represented by the examples under `examples/incremental/`.

## Requirements

HighVoronoiCC currently requires:

- a C++17 compiler;
- CMake 3.20 or newer;
- Eigen3;
- Boost;
- a standard threading implementation.

nanoflann is included directly in the source tree. The search implementation is developed and tested against this bundled version rather than an arbitrary system installation.

The rare high-precision correction fallback uses Boost.Multiprecision binary128 support.

## Building

A release build including tests and examples can be created with:

```bash
cmake -S . -B build/release \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIGHVORONOI_BUILD_TESTS=ON \
    -DHIGHVORONOI_BUILD_EXAMPLES=ON

cmake --build build/release
```

Run the complete registered test suite with:

```bash
ctest --test-dir build/release --output-on-failure
```

Tests are registered with thematic CTest labels, so focused subsets can also be run with `ctest -L ...`.

For concurrency changes it is additionally useful to maintain sanitizer builds and exercise the parallel smoke cases there.

## Examples

The `examples/` directory contains complete user-facing workflows.

### One-shot ordinary construction

`examples/general_position.cpp`

- bounded four-dimensional general-position construction;
- `ClassicRaycast`;
- geometric verification.

`examples/cubic_grid.cpp`

- strongly degenerate Cartesian grid;
- `InRangeRaycast`;
- exact expected bounded vertex count.

### Incremental workflows

`examples/incremental/voronoi_refine_remove.cpp`

- ordinary `VoronoiMesh`;
- initial compute;
- `RefineVoronoi`;
- `RemoveVoronoi`;
- completeness after every stage.

`examples/incremental/high_voronoi_refine_remove.cpp`

- non-periodic `HighVoronoiMesh`;
- visible insertion;
- visible deletion;
- repeated `ComputeHighVoronoi` integration.

`examples/incremental/periodic_high_voronoi_refine.cpp`

- partially periodic HighVoronoi construction;
- invisible periodic reference-node closure;
- further visible-node insertion near periodic faces;
- visible-cell completeness and visible-domain output.

### Parallel construction

The `examples/parallel/` directory demonstrates the two threading axes:

- `parallel_cast.cpp`: cast-level parallelism;
- `parallel_mesh.cpp`: mesh-branch parallelism;
- `parallel_combined.cpp`: both levels enabled.

The regression suite complements the examples with stronger serial/parallel, incremental and explicit-reference comparisons.

## Validation and regression strategy

The most important diagnostic paths are:

`verify_mesh(...)`
: checks geometric consistency of stored vertex occurrences.

`verify_mesh_complete(...)`
: adds global finite/infinite full-edge closure.

`verify_mesh_complete_cells(...)`
: checks completeness on a selected cell range, notably the visible prefix of a periodic HighVoronoi representation.

`compare_meshes(...)`
: compares compatible mesh results by nodes, signatures and vertex positions.

Important end-to-end regression cases include:

- bounded 4D general position;
- bounded 4D Cartesian degeneracy;
- unbounded 2D construction with persistent infinite rays;
- 3D serial/parallel comparison;
- full versus range-limited computation;
- ordinary refine/remove workflows;
- HighVoronoi incremental versus batched construction;
- periodic HighVoronoi against an independent explicit `3 x 3` reference tiling for two periodic axes;
- periodic HighVoronoi refinement and removal against fresh periodic references.

## Relation to the reference paper

The accompanying preprint contains the mathematical basis of the construction algorithm, including:

- geometric characterization of Voronoi vertices and edges;
- the Raycast Lemma;
- iterative traversal between adjacent Voronoi vertices;
- localized convex-cone edge enumeration for degenerate vertices;
- the exhaustive construction algorithm;
- the Descent algorithm;
- parallelization strategies;
- refinement;
- fast quasi-periodic generation;
- spherical Voronoi diagrams;
- numerical robustness.

The C++ project deliberately separates the mathematical algorithm from implementation-specific ownership and correction mechanisms. In particular, current C++ periodic closure is an incremental reference-node construction and should not be confused with the later fast quasi-periodic copy/modify/paste algorithm described in Section 5 of the paper.

A copy of the preprint is included under `docs/`.

## Development roadmap

### Near term

- improve cast-worker termination/load balancing when a shared queue is temporarily empty while another worker still has in-flight work;
- stabilize the remaining public construction/configuration surface;
- clean and extend user-facing configuration documentation for HighVoronoi;
- benchmark hash-container sharding and both parallelization axes;
- add install/consumer and sanitizer release checks;
- continue performance profiling without changing the validated construction invariants.

### Longer term

- spherical Voronoi diagrams;
- fast quasi-periodic copy/modify/paste workflows;
- optimized fused `direct_cast` search backends;
- additional nearest-neighbour backends where profiling justifies them;
- optional disk-backed persistent storage;
- volume computation;
- interface measures;
- quadrature and integral evaluation;
- further performance optimization and benchmarking.

`SerialMesh` remains an experimental composition component, but the current incremental and periodic HighVoronoi workflows do not depend on it.

## Development process and AI assistance

The C++ implementation has been developed with extensive AI-assisted coding.

The mathematical algorithms, software architecture, API design, numerical requirements, test strategy and code review are directed by the author, based on the existing Julia implementation and the underlying mathematical work.

AI-generated implementation code is reviewed, compiled, tested and modified before inclusion in the project.

This repository is intentionally published with that development process disclosed.

## License

HighVoronoiCC is **source-available for non-commercial use** under the PolyForm Noncommercial License 1.0.0.

Commercial use is not licensed under those terms. Commercial licensing is available under a separate agreement with the author.

Third-party components retain their respective licenses. In particular, the bundled nanoflann source remains subject to its BSD license.

The scientific preprint and other documentation are not automatically covered by the HighVoronoiCC software license unless explicitly stated otherwise.

See `LICENSE` and `THIRD_PARTY_NOTICES.md` for details.

## Disclaimer

HighVoronoiCC is under active development. The current release should be regarded as experimental software and should not yet be relied upon as a production geometry library without independent validation.
