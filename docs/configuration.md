# Configuration and tuning

Configuration is deliberately split between API levels:

- **Level 1** chooses a coherent default and keeps infrastructure out of user code.
- **Level 2** exposes the policy choices that commonly matter in applications while still constructing the database, search tree, ray caster, and compute object for you.
- **Level 3** exposes those concrete objects and their compile-time/runtime parameters directly.

If you want to see the *same geometry problem* written at all three levels, start with [Getting started](getting_started.md#the-same-ordinary-problem-at-levels-1-2-and-3) and [the HighVoronoi comparison](getting_started.md#the-same-periodic-highvoronoi-problem-at-levels-1-2-and-3).

## Default policy

The public defaults are:

```text
Scalar              double
Index               std::uint32_t
Ray caster          CombinedRaycast
Combined policy     Robust fallback
Search              KDSearch{8,1}
Threading           SingleThread
Hash generator      FNV64_128HashGenerator
Serial DB hash      DirectHash
Serial edge hash    DirectHash
Queue hash          DirectHash
Parallel DB hash    StaticHash<16>
Parallel edge hash  StaticHash<16>
```

`RaycastParameters<>` means `RaycastParameters<CombinedRaycast,double>`. `DataBaseParams<>` uses `double`, `std::uint32_t`, `FNV64_128HashGenerator`, and `DirectHash` unless you select alternatives explicitly.

## Level 2 complete blueprint

The following is a complete Level-2 ordinary program. It deliberately writes out every runtime field of `VoronoiConfig` so that it can be used as a copy-and-edit template. The numeric values shown are the current defaults or ordinary starting values; writing them explicitly is not required in normal code.

```cpp
#include <highvoronoi/voronoi.hpp>

#include <cstddef>
#include <vector>

int main() {
    constexpr int Dim = 3;
    constexpr std::size_t point_count = 5;

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

    using Config = highvoronoi::VoronoiConfig<
        highvoronoi::SingleThread,
        highvoronoi::CombinedRaycast,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DirectHash,
        highvoronoi::DirectHash>;

    Config config = highvoronoi::make_voronoi_config<
        highvoronoi::SingleThread,
        highvoronoi::CombinedRaycast,
        highvoronoi::FNV64_128HashGenerator,
        highvoronoi::DirectHash,
        highvoronoi::DirectHash>(
            point_count,
            Dim,
            highvoronoi::SingleThread{});

    // General Level-2 runtime settings.
    config.thread_count = 1;
    config.database_block_units = 65536;
    config.search = highvoronoi::geometry::KDSearch{8, 1};
    config.database_hash = highvoronoi::DirectHash{1024};
    config.queue_hash = highvoronoi::DirectHash{256};
    config.edge_hash = highvoronoi::DirectHash{1024};
    config.verbose = false;

    // Complete RaycastParameters runtime surface.
    config.raycast.method = highvoronoi::CombinedRaycast{};
    config.raycast.variance_tolerance = 1e-15;
    config.raycast.break_tolerance = 1e-5;
    config.raycast.boundary_node_tolerance = 1e-7;
    config.raycast.plane_tolerance = 1e-12;
    config.raycast.ray_tolerance = 1e-12;
    config.raycast.rank_tolerance = 1e-12;
    config.raycast.verification_absolute_tolerance = 1e-10;
    config.raycast.verification_relative_tolerance = 1e-8;
    config.raycast.verify_walk_vertices = false;
    config.raycast.vertex_condition_tolerance = 1e-8;
    config.raycast.vertex_correction_relative_tolerance = 1e-12;
    config.raycast.vertex_correction_max_iterations = 3;
    config.raycast.classic_relative_error_trigger = 1e-10;
    config.raycast.classic_absolute_error_trigger = 1e-8;
    config.raycast.inrange_t_slack = 1e-7;

    auto mesh = highvoronoi::voronoi_mesh<Dim>(
        points,
        point_count,
        boundary,
        config);

    highvoronoi::compute(mesh);
}
```

The template arguments choose the compile-time policy types. The fields in `config` choose the runtime values carried by those types.

The persistent database layout is fixed when the mesh is constructed. Changing `config.database_hash` afterwards does not rebuild an existing persistent database.

### Level-2 limitation: Combined options

`VoronoiConfig` exposes `RaycastParameters`, but it does not currently expose a `CombinedRaycastOptions` field. The standard Level-1/2 Combined path therefore uses the robust Combined policy. Direct selection of `FastLossy`, fallback method, or Combined statistics is an ordinary Level-3 ray-caster concern.

## Level 3 complete blueprint

The following program solves the same bounded 3D problem but spells out the native storage, search, ray-caster, queue/edge parameters, threading axes, and compute object explicitly.

```cpp
#include <highvoronoi/voronoi.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

int main() {
    using Scalar = double;
    using Index = std::uint32_t;
    constexpr int Dim = 3;
    constexpr Index point_count = 5;

    using HashGenerator = highvoronoi::FNV64_128HashGenerator;

    using DatabaseParameters = highvoronoi::DataBaseParams<
        Scalar,
        Index,
        HashGenerator,
        highvoronoi::DirectHash>;

    using Database = highvoronoi::HVDataBase<
        highvoronoi::EmptyLock,
        DatabaseParameters,
        Dim>;

    using Mesh = highvoronoi::VoronoiMesh<
        Scalar,
        Dim,
        Database>;

    using Nodes = Mesh::InternalNodes;
    using Point = Mesh::NodePoint;
    using Boundary = Mesh::BoundaryType;

    using QueueParameters = highvoronoi::DataBaseParams<
        Scalar,
        Index,
        HashGenerator,
        highvoronoi::DirectHash>;

    using EdgeParameters = highvoronoi::EdgeBufferParams<
        HashGenerator,
        highvoronoi::DirectHash>;

    using RayParameters = highvoronoi::RaycastParameters<
        highvoronoi::CombinedRaycast,
        Scalar>;

    const double points[][Dim] = {
        {0.20, 0.20, 0.20},
        {0.80, 0.25, 0.30},
        {0.30, 0.75, 0.40},
        {0.75, 0.85, 0.70},
        {0.45, 0.42, 0.80}
    };

    Nodes nodes(point_count);
    for (Index i = 0; i < point_count; ++i) {
        nodes.set(i, points[i]);
    }

    auto boundary = Boundary::cuboid(
        Point::Constant(1.0),
        Point::Zero(),
        std::vector<Index>{});

    DatabaseParameters database_parameters{
        highvoronoi::DirectHash{1024}};

    auto database = std::make_shared<Database>(
        65536,
        database_parameters);

    Mesh mesh(
        std::move(nodes),
        std::move(boundary),
        database);

    const highvoronoi::geometry::KDSearch search{8, 1};

    RayParameters raycast;
    raycast.method = highvoronoi::CombinedRaycast{};
    raycast.variance_tolerance = Scalar{1e-15};
    raycast.break_tolerance = Scalar{1e-5};
    raycast.boundary_node_tolerance = Scalar{1e-7};
    raycast.plane_tolerance = Scalar{1e-12};
    raycast.ray_tolerance = Scalar{1e-12};
    raycast.rank_tolerance = Scalar{1e-12};
    raycast.verification_absolute_tolerance = Scalar{1e-10};
    raycast.verification_relative_tolerance = Scalar{1e-8};
    raycast.verify_walk_vertices = false;
    raycast.vertex_condition_tolerance = Scalar{1e-8};
    raycast.vertex_correction_relative_tolerance = Scalar{1e-12};
    raycast.vertex_correction_max_iterations = 3;
    raycast.classic_relative_error_trigger = Scalar{1e-10};
    raycast.classic_absolute_error_trigger = Scalar{1e-8};
    raycast.inrange_t_slack = Scalar{1e-7};

    highvoronoi::CombinedRaycastOptions combined;
    combined.precision_policy =
        highvoronoi::CombinedPrecisionPolicy::Robust;
    combined.fallback_method =
        highvoronoi::CombinedFallbackMethod::
            ClassicGeneralInRangeDegenerate;
    combined.collect_statistics = false;

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        search);

    auto raycaster = highvoronoi::make_raycaster(
        tree,
        raycast,
        combined);

    QueueParameters queue_parameters{
        highvoronoi::DirectHash{256}};

    EdgeParameters edge_parameters{
        highvoronoi::DirectHash{1024}};

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread, // mesh threading
        highvoronoi::SingleThread, // cast threading
        QueueParameters,
        EdgeParameters>;

    Compute operation(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,              // compute all public cells
        queue_parameters,
        edge_parameters);

    operation.compute(false);
}
```

This is the useful distinction between Levels 2 and 3: Level 2 still lets the library *wire* the objects; Level 3 makes that wiring part of your program.

## Exact configuration recipes

The following snippets are intended as copy/paste substitutions into the complete blueprints above.

### Choose another ray-caster at Level 2

Combined is the default:

```cpp
auto config = highvoronoi::make_voronoi_config<
    highvoronoi::SingleThread,
    highvoronoi::CombinedRaycast>(point_count, Dim);
```

Use InRange explicitly:

```cpp
auto config = highvoronoi::make_voronoi_config<
    highvoronoi::SingleThread,
    highvoronoi::InRangeRaycast>(point_count, Dim);
```

Use Classic explicitly:

```cpp
auto config = highvoronoi::make_voronoi_config<
    highvoronoi::SingleThread,
    highvoronoi::ClassicRaycast>(point_count, Dim);
```

### Choose another ray-caster at Level 3

Change the `RayParameters` type and construct the matching ray caster:

```cpp
using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;

RayParameters raycast;
auto raycaster = highvoronoi::make_raycaster(tree, raycast);
```

For Combined, Level 3 can additionally pass `CombinedRaycastOptions` to `make_raycaster(...)` as shown in the full blueprint.

## Ray casting

The public method tags are:

```cpp
highvoronoi::CombinedRaycast  // default
highvoronoi::ClassicRaycast
highvoronoi::InRangeRaycast
```

### Combined — default

Combined is the current standard for ordinary, incremental, HighVoronoi, and spherical default paths. It uses the specialized fused KD traversal.

The current robust Combined options are conceptually:

```cpp
highvoronoi::CombinedRaycastOptions options;
options.precision_policy =
    highvoronoi::CombinedPrecisionPolicy::Robust;
options.fallback_method =
    highvoronoi::CombinedFallbackMethod::
        ClassicGeneralInRangeDegenerate;
options.collect_statistics = false;
```

The fast Combined path is retained, but a precision-critical cast that would otherwise remain unresolved/infinite can be recomputed through the established Classic/InRange fallback. `FastLossy` remains an explicit ordinary Level-3 opt-in.

### Classic

Classic is the established nearest-neighbour ray-cast path. It remains useful for comparison, regression, and explicitly general-position workloads.

### InRange

InRange explicitly handles potentially degenerate endpoint geometry and remains useful for Cartesian/lattice-like cases and comparison/regression.

### Tolerances

`RaycastParameters` contains variance, plane, ray, rank/verification, and vertex-correction tolerances. Changing them can change topology, not merely speed. The complete Level-2/Level-3 blueprints above show every current runtime field.

After tuning them, rerun completeness checks.

## Search backends

Level 1/2 currently stores a `geometry::KDSearch` keyword:

```cpp
config.search = highvoronoi::geometry::KDSearch{
    8, // leaf_max_size
    1  // KD-tree build threads
};
```

The KD build thread count is independent of Voronoi mesh/cast threading.

At Level 3, the search backend is selected by the keyword passed to `make_search_tree`:

```cpp
auto tree = highvoronoi::geometry::make_search_tree(
    mesh,
    highvoronoi::geometry::KDSearch{8, 1});
```

The exact brute-force reference backend is:

```cpp
auto tree = highvoronoi::geometry::make_search_tree(
    mesh,
    highvoronoi::geometry::BruteForceSearch{});
```

`CopyKDSearch` and specialized/custom factory keywords are Level-3 choices.

## Hashing and storage

Three logically different hash-backed states matter during construction:

1. **persistent database hash** — deduplicates persistent finite/infinite signatures;
2. **cell-local vertex queue hash** — first-claim state for queued signatures;
3. **cell-local edge hash** — occurrence/completeness state during exploration.

They do not share lifetime, reset semantics, or necessarily the same optimal container layout.

### Choose the hash generator at Level 2

The generator is the third `VoronoiConfig` template parameter:

```cpp
using Config = highvoronoi::VoronoiConfig<
    highvoronoi::SingleThread,
    highvoronoi::CombinedRaycast,
    highvoronoi::Murmur128HashGenerator<>,
    highvoronoi::DirectHash,
    highvoronoi::DirectHash>;
```

The current default is:

```cpp
highvoronoi::FNV64_128HashGenerator
```

### Choose the hash generator at Level 3

Bind it into both the database/queue parameters and edge parameters:

```cpp
using HashGenerator = highvoronoi::Murmur128HashGenerator<>;

using DatabaseParameters = highvoronoi::DataBaseParams<
    Scalar, Index, HashGenerator, highvoronoi::DirectHash>;

using QueueParameters = highvoronoi::DataBaseParams<
    Scalar, Index, HashGenerator, highvoronoi::DirectHash>;

using EdgeParameters = highvoronoi::EdgeBufferParams<
    HashGenerator, highvoronoi::DirectHash>;
```

### DirectHash

```cpp
highvoronoi::DirectHash{4096}
```

One table. The number is the **initial capacity**, not a correctness maximum; the current table implementation can grow.

### StaticHash

```cpp
highvoronoi::StaticHash<16>{4096}
```

A fixed array of 16 independent tables. The runtime capacity is the initial capacity **per table**. Keys are routed by the first signature index modulo the number of tables.

For a parallel Level-2 mesh with 32 shards:

```cpp
using Config = highvoronoi::VoronoiConfig<
    highvoronoi::MultiThread,
    highvoronoi::CombinedRaycast,
    highvoronoi::FNV64_128HashGenerator,
    highvoronoi::StaticHash<32>,
    highvoronoi::StaticHash<32>>;

Config config{highvoronoi::MultiThread{8}};
config.database_hash = highvoronoi::StaticHash<32>{4096};
config.edge_hash = highvoronoi::StaticHash<32>{4096};
config.queue_hash = highvoronoi::DirectHash{256};
```

### DynamicHash

```cpp
highvoronoi::DynamicHash<>{
    4,     // initial table count
    1024,  // first-key values per table
    4096   // initial capacity per table
}
```

`DynamicHash<>` routes first-key ranges by `block_size` and can grow the family of tables. The generic `make_voronoi_config` factory cannot infer a useful runtime layout for every custom container, so fill the runtime object explicitly when using such a policy.

### Why a more complex generator or container can help

A larger/stronger fingerprint can reduce accidental hash overlap and long probe sequences at the cost of more hashing work and/or a larger fingerprint. A multi-table container attacks a different problem: distribution and lock contention. These are independent dimensions.

For a large high-dimensional mesh, distinguish the symptom first:

- repeated table growth -> increase initial capacity;
- long collision/probe behavior -> consider generator/fingerprint strategy;
- parallel lock contention -> consider `StaticHash<N>`/`DynamicHash<>`;
- queue contention -> remember that the current queue also has an outer queue lock, so merely sharding its inner duplicate table does not remove that synchronization point.

## Threading

### Level 1

```cpp
highvoronoi::SingleThread{}      // default
highvoronoi::MultiThread{N}      // mesh-level parallelism
```

The wrapper type determines its database lock at compile time:

```text
SingleThread -> EmptyLock
MultiThread  -> ReadWriteLock
```

### Level 2

Choose the threading type as the first `VoronoiConfig` template argument and set the runtime count:

```cpp
auto config = highvoronoi::make_voronoi_config<
    highvoronoi::MultiThread>(
        point_count,
        Dim,
        highvoronoi::MultiThread{8});
```

Level 2 retains the simple **mesh-level** threading model while allowing hash/search/raycast policy tuning around it.

### Level 3

`ComputeVoronoi` exposes two independent compile-time axes:

```cpp
using Compute = highvoronoi::ComputeVoronoi<
    Mesh,
    RayCaster,
    MeshThreading,
    CastThreading,
    QueueParameters,
    EdgeParameters>;
```

For example, four mesh branches and serial ray casting:

```cpp
using Compute = highvoronoi::ComputeVoronoi<
    Mesh,
    RayCaster,
    highvoronoi::MultiThread,
    highvoronoi::SingleThread,
    QueueParameters,
    EdgeParameters>;

Compute operation(
    mesh,
    raycaster,
    highvoronoi::MultiThread{4},
    highvoronoi::SingleThread{},
    std::nullopt,
    queue_parameters,
    edge_parameters);
```

Any native parallel `ComputeVoronoi` requires a persistent database instantiated with `ReadWriteLock`.

## Scalar and index choices

Level 1/2 intentionally fixes:

```text
double
std::uint32_t
```

At Level 3 the choices are explicit in `DataBaseParams` and the mesh type:

```cpp
using DatabaseParameters = highvoronoi::DataBaseParams<
    double,
    std::uint64_t,
    HashGenerator,
    highvoronoi::DirectHash>;
```

Use Level 3 if the stable internal index space/boundary encoding cannot fit in `uint32_t`, or if another scalar type is required.

## Fixed versus runtime dimension

Fixed dimension is the simplest form:

```cpp
auto mesh = highvoronoi::voronoi_mesh<5>(
    points,
    count,
    boundary);
```

`highvoronoi::Dynamic` is supported by the facade, but the runtime dimension must be supplied explicitly. Fixed dimensions remain preferable when known at compile time.

## HighVoronoi uses the same Level-2 configuration

The exact same `VoronoiConfig` object can be passed to `high_voronoi_mesh`:

```cpp
auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    config);
```

At Level 3 the owner changes from `VoronoiMesh` to `HighVoronoiMesh`, and the orchestration changes from `ComputeVoronoi` to `ComputeHighVoronoi`. See the full side-by-side HighVoronoi examples in [Getting started](getting_started.md#the-same-periodic-highvoronoi-problem-at-levels-1-2-and-3).

## Progress output

At Level 2:

```cpp
config.verbose = true;
```

The wrapper forwards this to ordinary or HighVoronoi compute. Level 3 calls the underlying `compute(true)` operation directly.

## A practical tuning order

For a new large workload:

1. establish a Level-1 serial result and validate completeness;
2. move to Level 2 and change **one policy at a time** using the recipes above;
3. compare explicit ray-caster alternatives only if geometry or diagnostics justify it;
4. inspect initial hash growth/collision behavior before changing the hash family;
5. switch to Level-1/2 `MultiThread` and benchmark mesh-level scaling;
6. tune persistent/edge sharding if contention is visible;
7. tune KD search;
8. use Level 3 only when you need the second threading axis, custom locks/table implementations, direct ray-caster ownership/options, a nonstandard search backend, custom scalar/index types, or implementation work;
9. rerun completeness and serial/parallel equivalence checks after every structural change.
