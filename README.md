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
* generators in general position;
* degenerate / non-general configurations;
* serial Voronoi construction;
* KD-tree nearest-neighbour search through nanoflann;
* brute-force nearest-neighbour search as a reference backend;
* fixed-dimensional point and mesh types;
* stored, computed and hybrid node-access infrastructure;
* compact configurable index and scalar types;
* configurable hash tables and synchronization policies;
* consistency verification of constructed meshes.

Current regression cases include both a bounded four-dimensional configuration in general position and a highly degenerate four-dimensional Cartesian grid.

## Current architecture

The implementation separates geometry, persistent storage, indexing, search and construction logic.

Important design principles include:

* stable internal indices separated from mutable public numbering;
* persistent canonical vertex signatures;
* reusable caller-owned workspaces in performance-critical paths;
* interchangeable nearest-neighbour backends;
* compile-time selection of synchronization policies;
* independent mesh, search-tree, ray-casting and edge-enumeration layers;
* explicit support for degenerate Voronoi vertices;
* separation between global construction management, per-mesh systematic construction and local ray-casting workers.

A more detailed description is available in `docs/ARCHITECTURE.md`.

## Development roadmap

### Near term

* multithreaded ray casting inside one mesh branch;
* parallel mesh views and communication between construction branches;
* completion and stabilization of the public construction API;
* additional numerical and regression tests;
* periodic boundary construction;
* local refinement;
* insertion and removal of generators.

### Longer term

* partially bounded and unbounded domains;
* spherical Voronoi diagrams;
* quasi-periodic mesh generation;
* additional nearest-neighbour backends;
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

nanoflann is currently included directly in the source tree.

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

## Examples

Two complete examples are provided in the `examples/` directory.

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

The resulting mesh is checked with the same independent mesh-verification utilities used by the test suite.

## Mesh verification

HighVoronoiCC provides diagnostic utilities for checking existing meshes.

`verify_mesh(...)` checks the geometric consistency of stored vertices, including equal generator distances and nearest-neighbour membership.

`compare_meshes(...)` compares two meshes by node data, vertex signatures and vertex positions and can therefore be used to compare reference and optimized construction paths.

These checks verify the consistency of stored data. `verify_mesh(...)` alone does not prove that a mesh is complete.

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

