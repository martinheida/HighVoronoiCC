# HighVoronoiCC

HighVoronoiCC is an experimental C++17 implementation of a local algorithm for the construction of high-dimensional Voronoi diagrams.

The project is a redesign and C++ port of the ideas implemented in **HighVoronoi.jl** and developed mathematically in:

**Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025.**  
DOI: `10.20347/WIAS.PREPRINT.3197`

The algorithm constructs Voronoi diagrams locally by identifying the edges emerging from known vertices and following these edges through nearest-neighbour based ray casting. The mathematical algorithm is designed for generators both in general and non-general position.

## Status

HighVoronoiCC is currently **experimental / alpha software**. APIs may still change.

The currently validated construction path supports:

* bounded Euclidean domains with planar boundaries;
* unbounded Euclidean construction with persistent infinite Voronoi edges;
* generators in general position;
* degenerate / non-general configurations;
* serial Voronoi construction;
* parallel construction using independent mesh branches;
* parallel ray-casting workers inside one mesh branch;
* combined mesh- and cast-level parallelism;
* KD-tree nearest-neighbour search through the bundled nanoflann backend;
* brute-force nearest-neighbour search as a reference backend;
* fixed-dimensional point and mesh types;
* stored, computed and hybrid node-access infrastructure;
* compact configurable index and scalar types;
* configurable queue- and edge-hash containers with synchronization policies;
* geometric consistency verification;
* topological edge-completeness verification including infinite edges.

Current regression cases include bounded four-dimensional general-position and Cartesian-degenerate configurations, a three-dimensional 200-generator parallel smoke case, and an unbounded two-dimensional case with persistent infinite rays.

## Current architecture

The implementation separates geometry, persistent storage, indexing, search and construction logic.

Important design principles include:

* stable internal indices separated from mutable public numbering;
* persistent canonical vertex signatures;
* persistent infinite-edge records keyed by complete supporting-edge signatures;
* reusable caller-owned workspaces in performance-critical paths;
* interchangeable nearest-neighbour backends;
* compile-time selection of synchronization policies;
* independent mesh, search-tree, ray-casting and edge-enumeration layers;
* explicit support for degenerate Voronoi vertices;
* two independent parallelization axes: mesh branches and cast workers;
* synchronization attached to shared mutable state rather than to worker objects;
* strict branch-local ownership of worker `EdgeIterator` state;
* separation between global construction management, per-mesh systematic construction and local ray-casting workers.

A more detailed description is available in `docs/ARCHITECTURE.md`.

## Parallel construction

HighVoronoiCC currently distinguishes two independent threading policies:

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

## Persistent infinite edges and completeness

`RayCaster` distinguishes finite and infinite ray results. Infinite rays are persisted by the mesh as

```text
(full supporting edge, origin, direction)
```

using the complete supporting edge in stable internal numbering as the persistent duplicate key. This is important in degenerate configurations where several minimal edges may represent the same geometric unbounded edge.

The mesh exposes persisted rays through `infinite_edges()`.

`verify_mesh_complete(...)` extends ordinary geometric verification with global edge closure:

* each finite geometric edge must occur at exactly two finite endpoints;
* an unbounded edge contributes one finite endpoint and one persistent infinite-edge record;
* repeated minimal-edge representations at the same degenerate vertex are deduplicated by the full supporting edge;
* a third occurrence is treated as an error.

`verify_mesh(...)` remains the cheaper geometric consistency check. It verifies stored data but does not by itself prove completeness.

## Configuration

The project includes `howto_config.hpp`, an editor-friendly C++17 configuration guide intended to be read, copied and modified directly.

It documents the current choices and defaults for:

* `SingleThread` and `MultiThread`;
* queue-hash configuration;
* edge-hash configuration;
* database configuration;
* `VoronoiMesh`;
* search-tree selection;
* ray-cast strategy;
* `ComputeVoronoi`.

The configuration remains ordinary C++ rather than introducing a custom keyword-argument framework. This keeps the library C++17-compatible and makes the complete setup easy to inspect or generate with external tooling.

## Development roadmap

### Near term

* stabilize `SerialMesh` semantics and ownership;
* build refinement/replacement workflows on top of `SerialMesh`;
* complete periodic construction using composite/computed mesh components;
* improve cast-worker load balancing when a shared queue is temporarily empty while another worker is still producing work;
* stabilize the public construction/configuration API;
* extend parallel and completeness regression coverage;
* benchmark queue/edge hash sharding and parallel scaling.

### Longer term

* spherical Voronoi diagrams;
* quasi-periodic mesh generation;
* insertion and removal of generators;
* optimized fused `direct_cast` search backends;
* additional nearest-neighbour backends where profiling justifies them;
* optional disk-backed persistent storage;
* volume computation;
* interface measures;
* quadrature and integral evaluation;
* further performance optimization and benchmarking.

## Requirements

HighVoronoiCC currently requires:

* a C++17 compiler;
* CMake 3.20 or newer;
* Eigen3;
* Boost;
* a standard threading implementation.

nanoflann is included directly in the source tree. The search implementation is developed and tested against this bundled version rather than an arbitrary system installation.

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

For concurrency changes it is additionally useful to maintain a ThreadSanitizer build and run the parallel smoke cases there.

## Examples

The `examples/` directory contains complete construction examples. The test suite additionally contains small regression programs that are intentionally readable enough to serve as implementation examples.

### General-position example

`examples/general_position.cpp` demonstrates the construction of a bounded Voronoi mesh for generators in general position.

The example shows how to:

1. define scalar, index and mesh types;
2. construct the generator set;
3. define the planar boundary;
4. create the Voronoi mesh;
5. run the Voronoi construction;
6. verify the resulting mesh geometrically.

### Degenerate Cartesian-grid example

`examples/cubic_grid.cpp` applies the same construction algorithm to a highly degenerate Cartesian configuration.

This example is intended to demonstrate the non-general-position edge iterator and to provide a reproducible regression case for configurations in which many generators belong to the same Voronoi vertex.

### Parallel construction examples

The `examples/parallel/` directory demonstrates the two independent `ComputeVoronoi` threading axes on the same deterministic 3D configuration:

* `parallel_cast.cpp`: `MeshThreading = SingleThread`, `CastThreading = MultiThread`;
* `parallel_mesh.cpp`: `MeshThreading = MultiThread`, `CastThreading = SingleThread`;
* `parallel_combined.cpp`: both threading axes are enabled.

The examples use a `ReadWriteLock` persistent database and demonstrate `StaticHash<16>` sharding for the persistent database hash and the shared EdgeHash. The cell-local vertex queue remains `DirectHash` because its current outer queue lock is the dominant synchronization point.

`tests/test_compute_voronoi_parallel_smoke.cpp` complements these examples by computing serial, mesh-parallel and cast-parallel results and comparing the resulting meshes directly.

### Infinite-edge storage test

`tests/test_infinite_edge_storage.cpp` demonstrates persistent infinite-edge insertion, concurrent duplicate suppression and public-index translation through a `ReorderedMeshView`.

### Completeness tests

`tests/test_edge_hash_completeness.cpp` checks exact twice-only edge occurrence semantics directly.

`tests/test_compute_voronoi_unbounded_completeness.cpp` constructs an unbounded 2D Voronoi diagram, verifies its three persistent infinite rays, then removes one ray and confirms that `verify_mesh_complete(...)` detects the incomplete mesh.

## Mesh verification

HighVoronoiCC provides three complementary diagnostic paths.

`verify_mesh(...)` checks the geometric consistency of stored vertices, including equal generator distances and nearest-neighbour membership.

`verify_mesh_complete(...)` additionally reconstructs global full-edge incidences and checks finite/infinite edge closure.

`compare_meshes(...)` compares two meshes by node data, vertex signatures and vertex positions and can therefore be used to compare serial, parallel, reference and optimized construction paths.

These tools have different purposes. `verify_mesh(...)` validates existing records, `verify_mesh_complete(...)` checks the current global edge closure, and `compare_meshes(...)` compares two construction results.

## Mathematical background

The accompanying preprint contains the mathematical basis of the construction algorithm, including:

* geometric characterization of Voronoi vertices and edges;
* the Raycast Lemma;
* iterative traversal between adjacent Voronoi vertices;
* localized convex-cone edge enumeration for degenerate vertices;
* the exhaustive construction algorithm;
* the Descent algorithm for finding an initial vertex;
* parallelization strategies;
* refinement;
* quasi-periodic construction;
* spherical Voronoi diagrams;
* numerical robustness considerations.

A copy of the preprint is included under `docs/`.

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
