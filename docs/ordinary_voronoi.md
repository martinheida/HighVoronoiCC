# Constructing ordinary Voronoi diagrams

Ordinary Euclidean construction is the foundation used by the incremental, HighVoronoi, periodic-repair, and spherical layers. Most users should start with the Level-1 facade and only expose the underlying pieces when they need to tune them.

For a direct line-by-line comparison of the **same ordinary problem** at Levels 1, 2, and 3, see [Getting started: ordinary Level 1/2/3 comparison](getting_started.md#the-same-ordinary-problem-at-levels-1-2-and-3).

## Level 1: the intended ordinary workflow

```cpp
#include <highvoronoi/voronoi.hpp>

constexpr int Dim = 3;

double points[] = {
    0.20, 0.20, 0.20,
    0.80, 0.25, 0.30,
    0.30, 0.75, 0.40,
    0.75, 0.85, 0.70,
    0.45, 0.42, 0.80
};

using Boundary = highvoronoi::Boundary<Dim>;
using Point = Boundary::Point;

auto boundary = Boundary::cuboid(
    Point::Constant(1.0),
    Point::Zero(),
    std::vector<std::size_t>{});

auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    5,
    boundary);

highvoronoi::compute(mesh);
```

Coordinates are node-major contiguous `double` values. The factory copies them into the library's node storage, so the input array need not remain alive afterwards.

For fixed dimension, the dimension is the template argument. For `highvoronoi::Dynamic`, pass the runtime dimension explicitly as the final construction argument.

## What Level 1 selects

The ordinary facade selects:

```text
Scalar                  double
Index                   uint32_t
Ray caster              CombinedRaycast (robust fallback)
Search                   KDSearch{8,1}
Threading                SingleThread
Serial database hash    FNV64_128HashGenerator + DirectHash
Serial edge hash        FNV64_128HashGenerator + DirectHash
Queue hash               FNV64_128HashGenerator + DirectHash
```

The initial capacities are estimated from point count and dimension; the hash tables retain their normal growth behavior. This means the estimate is a performance choice, not a fixed correctness limit.

### Why mention these choices this early?

You do not need to configure them to get started, but they explain the first places to look when a workload becomes difficult:

- unusually degenerate geometry may justify comparing Combined with explicit InRange behavior;
- very large datasets may benefit from a different initial hash capacity or container layout;
- parallel workloads benefit from sharded persistent/edge state;
- search parameters can matter when nearest-neighbour work dominates.

See [Configuration and tuning](configuration.md) rather than adding these types to Level-1 application code.

## Bounded and unbounded meshes

The Level-1 factory takes a boundary. For an unconstrained ordinary mesh, pass an empty boundary of the same dimension/type:

```cpp
Boundary empty_boundary;
auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    empty_boundary);
```

Unbounded results may contain persistent infinite edges. Iterate them with:

```cpp
for (const auto& edge : mesh.infinite_edges()) {
    // edge.sigma
    // edge.origin
    // edge.direction
}
```

For a cuboid, remember the existing boundary convention: omitting `periodic_axes` means all axes periodic, while an **empty vector** means no periodic axes. Ordinary construction does not perform periodic reference closure; use [HighVoronoi](high_voronoi.md) for that workflow.

## Accessing results

A convenience mesh intentionally exposes the same result vocabulary as the native mesh:

```cpp
for (const auto& vertex : mesh.vertices(cell)) {
    auto const& position = vertex.position;
    auto const& sigma = vertex.sigma;
}
```

The wrapper also exports useful type aliases:

```cpp
using Point = decltype(mesh)::Point;
using Index = decltype(mesh)::Index;
using Sigma = decltype(mesh)::Sigma;
```

For native APIs not forwarded by the facade:

```cpp
auto& native = mesh.level3_mesh();
```

See [Accessing mesh results](accessing_results.md) for primary/secondary ownership, infinite edges, neighbours, and stable internal indices.

## Parallel Level 1

```cpp
auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    highvoronoi::MultiThread{4});

highvoronoi::compute(mesh);
```

The facade interprets `MultiThread` as mesh-level parallelism and automatically chooses a `ReadWriteLock` database plus `StaticHash<16>` for persistent database and edge state. This corresponds to the currently preferred construction-parallel direction in the project benchmarks. The cell-local queue remains `DirectHash` because its outer queue lock is the relevant synchronization point.

Override the count for one computation with `compute(mesh, count)`. Cast-worker threading remains available through Level 3; see [Parallel computation](parallelism.md).

## Level 2: choose ray casting and hash policies

A configured mesh uses the same factory but receives `VoronoiConfig`:

```cpp
using Config = highvoronoi::VoronoiConfig<
    highvoronoi::MultiThread,
    highvoronoi::CombinedRaycast,
    highvoronoi::FNV64_128HashGenerator,
    highvoronoi::StaticHash<32>,
    highvoronoi::StaticHash<32>>;

Config config;
config.thread_count = 8;
config.database_hash = highvoronoi::StaticHash<32>{4096};
config.edge_hash = highvoronoi::StaticHash<32>{4096};
config.queue_hash = highvoronoi::DirectHash{1024};

auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    config);

highvoronoi::compute(mesh);
```

This is the level at which you should reason about **which hash is serving which state**. The persistent database deduplicates stored signatures, the queue hash claims cell-local vertex signatures, and the edge hash tracks edge occurrence during exploration. A more complex generator increases fingerprint/probe work but can reduce problematic overlap; a multi-table container primarily changes partitioning and lock contention. See [Hashing and storage](configuration.md#hashing-and-storage).

`make_voronoi_config(...)` is usually more convenient than filling every runtime field manually because it provides the same automatic initial-capacity heuristic as Level 1.

## Level 3: explicit construction

Level 3 is the existing full-control path:

```text
VoronoiNodes / Boundary
        |
        v
HVDataBase + VoronoiMesh
        |
        v
SearchTree
        |
        v
RayCaster
        |
        v
ComputeVoronoi
```

At this level you choose database ownership, lock type, queue/edge parameter types, search backend, ray caster, mesh threading, cast threading, optional compute range, and every lower-level runtime parameter explicitly.

Use it when:

- you need cast-worker parallelism or combined mesh/cast parallelism;
- an application owns/shares the database itself;
- you need custom hash table implementations or lock policies;
- you need a nonstandard search backend;
- you need explicit `CombinedRaycastOptions` such as diagnostic statistics;
- you are extending the library rather than just consuming it.

The repository's existing detailed examples remain useful Level-3 references, even though a new user no longer needs to reproduce their infrastructure boilerplate.

## Validation

For new configurations, use the completeness validator on the underlying native mesh:

```cpp
const auto report = highvoronoi::verify_mesh_complete(
    mesh.level3_mesh(),
    1e-12,
    true);
```

`verify_mesh()` only checks the consistency of stored occurrences. See [Validation](validation.md).

## What to read next

- Insert/delete nodes: [Incremental ordinary meshes](incremental.md).
- Periodic closure: [HighVoronoi and periodic meshes](high_voronoi.md).
- Raycasting/hash/search details: [Configuration and tuning](configuration.md).
- Throughput: [Parallel computation](parallelism.md).
