# Getting started

HighVoronoiCC exposes three API levels. Start at Level 1. Move down only when a problem actually requires more control.

```text
Level 1   task API
          points + boundary -> mesh -> compute/refine/remove

Level 2   configured API
          choose ray caster, hash generator/container, search parameters,
          tolerances and the preferred thread count

Level 3   full-control API
          explicit database, locks, search tree, ray caster,
          ComputeVoronoi / RefineVoronoi / RemoveVoronoi
```

The three levels are not different algorithms. Levels 1 and 2 are thin ownership/configuration facades over the Level-3 classes. Every Level-1/2 mesh exposes its underlying native mesh through `level3_mesh()`.

## Requirements

The project requires a C++17 compiler, CMake 3.20 or newer, Eigen3, Boost, and the platform threading library. The nanoflann backend is bundled with the project. Doxygen is only needed when generating this manual.

A normal release build is:

```bash
cmake -S . -B build/release \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIGHVORONOI_BUILD_TESTS=ON \
    -DHIGHVORONOI_BUILD_EXAMPLES=ON

cmake --build build/release
ctest --test-dir build/release --output-on-failure
```

## Level 1: compute a mesh without infrastructure types

A Level-1 program needs only flat `double` coordinates, the point count, the dimension, and a boundary.

```cpp
#include <highvoronoi/voronoi.hpp>

#include <cstddef>
#include <vector>

int main() {
    constexpr int Dim = 2;

    // Flat node-major storage: x0,y0, x1,y1, ...
    double points[] = {
        0.20, 0.20,
        0.80, 0.25,
        0.30, 0.75,
        0.75, 0.85
    };

    using Boundary = highvoronoi::Boundary<Dim>;
    using Point = Boundary::Point;

    // Empty periodic-axis list => ordinary non-periodic box.
    auto boundary = Boundary::cuboid(
        Point::Constant(1.0),
        Point::Zero(),
        std::vector<std::size_t>{});

    auto mesh = highvoronoi::voronoi_mesh<Dim>(
        points,
        4,
        boundary);

    highvoronoi::compute(mesh);

    for (const auto& vertex : mesh.vertices(0)) {
        // vertex.position
        // vertex.sigma
    }
}
```

No lock, database, hash table, search tree, ray caster, or `ComputeVoronoi` type appears in this program.

The Level-1 defaults are deliberately opinionated:

- coordinate/vertex scalar: `double`;
- index type: `std::uint32_t`;
- ray casting: robust `CombinedRaycast`;
- search: nanoflann KD search;
- threading: `SingleThread`;
- serial persistent/edge hashes: `FNV64_128HashGenerator` + `DirectHash`;
- cell-local queue hash: `DirectHash`;
- initial hash capacities: chosen automatically from node count and dimension; the tables can grow when required.

These defaults are a starting point, not a claim that one configuration is fastest for every dataset. The relevant alternatives are introduced below and explained in [Configuration and tuning](configuration.md).

### Reading result types

The input is deliberately just `double*`. Once a mesh exists, use its aliases instead of reconstructing implementation types yourself:

```cpp
using Mesh = decltype(mesh);
using Point = Mesh::Point;
using Index = Mesh::Index;
using Sigma = Mesh::Sigma;
```

The Level-1 defaults make `Mesh::Point` a fixed-dimensional double point and `Mesh::Index` a `std::uint32_t`.

## Level 1: parallel computation

Pass `MultiThread{N}` when constructing the mesh:

```cpp
auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    highvoronoi::MultiThread{4});

highvoronoi::compute(mesh);
```

At Level 1, `MultiThread` means **mesh-level parallelism**, the currently recommended construction-parallel path. The wrapper automatically switches persistent storage to `ReadWriteLock` and uses `StaticHash<16>` for the persistent database and edge hash; the cell-local queue remains `DirectHash`.

You can override the stored thread count for one operation:

```cpp
highvoronoi::compute(mesh, 8);
```

For a `SingleThread` mesh the count is accepted but ignored. Cast-worker parallelism is intentionally a Level-2/3 choice; see [Parallel computation](parallelism.md).

## Level 1: refine and remove

The same mesh remembers the configuration needed for later operations:

```cpp
double added[] = {
    0.42, 0.52,
    0.63, 0.61
};

const auto refine_report = highvoronoi::refine(mesh, added, 2);
const auto remove_report = highvoronoi::remove(mesh, {Mesh::Index{1}});
```

For a multithreaded Level-1 mesh, `refine` and `remove` use the stored thread count; an explicit count can be supplied as the last argument.

See [Incremental ordinary meshes](incremental.md) for the insertion/removal semantics and the distinction between public and stable internal indices.

## Level 1: HighVoronoi

The HighVoronoi facade has the same shape:

```cpp
#include <highvoronoi/high_voronoi.hpp>

auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
    points,
    point_count,
    boundary);

highvoronoi::compute(mesh);
highvoronoi::refine(mesh, added, added_count);
highvoronoi::remove(mesh, {decltype(mesh)::Index{2}});
```

For periodic boundaries, `ComputeHighVoronoi`'s reference-node closure remains internal. You work with visible nodes and visible cells; see [HighVoronoi and periodic meshes](high_voronoi.md).

## What the default is already doing for you

The compact call hides several mechanisms that become relevant on harder workloads.

### Ray casting

`CombinedRaycast` is the default everywhere in the public construction stack. It uses the fused Combined search path and, by default, the robust fallback to Classic/InRange when a precision-critical cast cannot be decided reliably. `ClassicRaycast` and `InRangeRaycast` remain available at Level 2 and Level 3. See [Ray casting](configuration.md#ray-casting).

### Hashing

The mesh contains several logically separate hash-backed states: persistent vertex identity, cell-local queue claims, and edge occurrence tracking. Serial Level 1 starts with a direct table. Parallel Level 1 shards the persistent database and edge hash into 16 independent tables to reduce contention, while keeping the already externally synchronized queue hash direct. See [Hashing and storage](configuration.md#hashing-and-storage).

For very large or unusual datasets, a different hash generator, more/fewer fixed shards, or `DynamicHash<>` may be preferable. That is a Level-2 concern, not a prerequisite for computing your first diagram.

### Search and threading

Level 1 uses KD search and maps `MultiThread` to mesh-level parallelism. The full API also supports different search configuration and a second cast-worker parallelization axis. See [Search backends](configuration.md#search-backends) and [Parallel computation](parallelism.md).

### Validation

Convenience does not change the distinction between consistency and completeness. `verify_mesh()` checks stored geometry; `verify_mesh_complete()` also checks global edge closure. The validators currently operate on the native Level-3 mesh:

```cpp
const auto report = highvoronoi::verify_mesh_complete(
    mesh.level3_mesh(),
    1e-12,
    true);
```

See [Validation](validation.md).

## Level 2: configure policies without wiring the engine

Level 2 keeps the same factories and operations but exposes `VoronoiConfig`. The library still constructs the database, lock, search tree, ray caster, and compute owner.

A short configured call is:

```cpp
auto config = highvoronoi::make_voronoi_config<
    highvoronoi::SingleThread,
    highvoronoi::InRangeRaycast>(point_count, Dim);

config.raycast.variance_tolerance = 9e-14;

auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    config);

highvoronoi::compute(mesh);
```

The complete Level-2 configuration surface is written out below and in [Configuration and tuning](configuration.md).

## Level 3: full control

Level 3 is the native API. Your code constructs the persistent database, mesh, search tree, ray caster, queue/edge parameter objects, and `ComputeVoronoi`/`ComputeHighVoronoi` operation explicitly.

Use Level 3 when you need both construction parallelization axes independently, custom database ownership, nonstandard locks/table implementations, a nonstandard search backend, custom scalar/index types, direct ray-caster ownership/options, or implementation-level access.

The following two sections show exactly how much additional machinery that means in practice.

## The same ordinary problem at Levels 1, 2 and 3

All three programs below compute the **same five-generator bounded 3D Voronoi diagram** in the unit cube. The difference is only how much infrastructure is made explicit.

### Ordinary Level 1

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

    auto mesh = highvoronoi::voronoi_mesh<Dim>(
        points,
        point_count,
        boundary);

    highvoronoi::compute(mesh);
}
```

Level 1 selects the standard scalar/index, robust Combined ray casting, KD search, serial storage/hash layout, and SingleThread automatically.

### Ordinary Level 2

This version writes out every runtime field carried by `VoronoiConfig` and every compile-time policy exposed by Level 2.

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

    config.thread_count = 1;
    config.database_block_units = 65536;
    config.search = highvoronoi::geometry::KDSearch{8, 1};
    config.database_hash = highvoronoi::DirectHash{1024};
    config.queue_hash = highvoronoi::DirectHash{256};
    config.edge_hash = highvoronoi::DirectHash{1024};
    config.verbose = false;

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

The point of Level 2 is not that you *should* write all of this. It is that every one of these policy/runtime settings can be made explicit without taking ownership of the construction machinery.

### Ordinary Level 3

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
        Scalar, Index, HashGenerator, highvoronoi::DirectHash>;
    using Database = highvoronoi::HVDataBase<
        highvoronoi::EmptyLock, DatabaseParameters, Dim>;
    using Mesh = highvoronoi::VoronoiMesh<Scalar, Dim, Database>;
    using Nodes = Mesh::InternalNodes;
    using Point = Mesh::NodePoint;
    using Boundary = Mesh::BoundaryType;
    using QueueParameters = highvoronoi::DataBaseParams<
        Scalar, Index, HashGenerator, highvoronoi::DirectHash>;
    using EdgeParameters = highvoronoi::EdgeBufferParams<
        HashGenerator, highvoronoi::DirectHash>;
    using RayParameters = highvoronoi::RaycastParameters<
        highvoronoi::CombinedRaycast, Scalar>;

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

    auto tree = highvoronoi::geometry::make_search_tree(mesh, search);
    auto raycaster = highvoronoi::make_raycaster(
        tree, raycast, combined);

    QueueParameters queue_parameters{
        highvoronoi::DirectHash{256}};
    EdgeParameters edge_parameters{
        highvoronoi::DirectHash{1024}};

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        QueueParameters,
        EdgeParameters>;

    Compute operation(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        queue_parameters,
        edge_parameters);

    operation.compute(false);
}
```

This is the same algorithm and the same geometry as Levels 1 and 2; Level 3 simply makes ownership and orchestration explicit.

## The same periodic HighVoronoi problem at Levels 1, 2 and 3

The next three programs compute the same four visible generators in a unit cube that is periodic in `x` and `y` and non-periodic in `z`.

### HighVoronoi Level 1

```cpp
#include <highvoronoi/high_voronoi.hpp>

#include <cstddef>
#include <vector>

int main() {
    constexpr int Dim = 3;
    constexpr std::size_t point_count = 4;

    double points[] = {
        0.15, 0.20, 0.30,
        0.75, 0.22, 0.44,
        0.35, 0.76, 0.58,
        0.69, 0.71, 0.81
    };

    using Boundary = highvoronoi::Boundary<Dim>;
    using Point = Boundary::Point;

    auto boundary = Boundary::cuboid(
        Point::Constant(1.0),
        Point::Zero(),
        std::vector<std::size_t>{0, 1});

    auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
        points,
        point_count,
        boundary);

    highvoronoi::compute(mesh);
}
```

### HighVoronoi Level 2

```cpp
#include <highvoronoi/high_voronoi.hpp>

#include <cstddef>
#include <vector>

int main() {
    constexpr int Dim = 3;
    constexpr std::size_t point_count = 4;

    double points[] = {
        0.15, 0.20, 0.30,
        0.75, 0.22, 0.44,
        0.35, 0.76, 0.58,
        0.69, 0.71, 0.81
    };

    using Boundary = highvoronoi::Boundary<Dim>;
    using Point = Boundary::Point;

    auto boundary = Boundary::cuboid(
        Point::Constant(1.0),
        Point::Zero(),
        std::vector<std::size_t>{0, 1});

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

    config.thread_count = 1;
    config.database_block_units = 65536;
    config.search = highvoronoi::geometry::KDSearch{8, 1};
    config.database_hash = highvoronoi::DirectHash{1024};
    config.queue_hash = highvoronoi::DirectHash{256};
    config.edge_hash = highvoronoi::DirectHash{1024};
    config.verbose = false;

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

    auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
        points,
        point_count,
        boundary,
        config);

    highvoronoi::compute(mesh);
}
```

The Level-2 policy object is deliberately the same type used by ordinary construction. HighVoronoi-specific periodic reference generation and closure remain internal to the operation.

### HighVoronoi Level 3

```cpp
#include <highvoronoi/high_voronoi.hpp>

#include <cstdint>
#include <vector>

int main() {
    using Scalar = double;
    using Index = std::uint32_t;
    constexpr int Dim = 3;

    using HashGenerator = highvoronoi::FNV64_128HashGenerator;
    using DatabaseParameters = highvoronoi::DataBaseParams<
        Scalar, Index, HashGenerator, highvoronoi::DirectHash>;
    using Database = highvoronoi::HVDataBase<
        highvoronoi::EmptyLock, DatabaseParameters, Dim>;
    using Mesh = highvoronoi::HighVoronoiMesh<
        Scalar, Dim, Database>;
    using Point = Mesh::NodePoint;
    using Boundary = Mesh::BoundaryType;
    using QueueParameters = highvoronoi::DataBaseParams<
        Scalar, Index, HashGenerator, highvoronoi::DirectHash>;
    using EdgeParameters = highvoronoi::EdgeBufferParams<
        HashGenerator, highvoronoi::DirectHash>;
    using RayParameters = highvoronoi::RaycastParameters<
        highvoronoi::CombinedRaycast, Scalar>;

    auto boundary = Boundary::cuboid(
        Point::Constant(1.0),
        Point::Zero(),
        std::vector<Index>{0, 1});

    DatabaseParameters database_parameters{
        highvoronoi::DirectHash{1024}};

    Mesh mesh(
        Index{Dim},
        std::move(boundary),
        std::in_place,
        65536,
        database_parameters);

    const double points[][Dim] = {
        {0.15, 0.20, 0.30},
        {0.75, 0.22, 0.44},
        {0.35, 0.76, 0.58},
        {0.69, 0.71, 0.81}
    };

    for (const auto& xyz : points) {
        Point point;
        point << xyz[0], xyz[1], xyz[2];
        (void)mesh.append_visible_node(point);
    }

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

    QueueParameters queue_parameters{
        highvoronoi::DirectHash{256}};
    EdgeParameters edge_parameters{
        highvoronoi::DirectHash{1024}};

    using Compute = highvoronoi::ComputeHighVoronoi<
        Mesh,
        highvoronoi::geometry::KDSearch,
        RayParameters,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        QueueParameters,
        EdgeParameters>;

    Compute::Settings settings;
    settings.nearest_tolerance = Scalar{1e-10};
    settings.boundary_tolerance = Scalar{1e-10};
    settings.maximum_rounds = 16;
    settings.maximum_periodic_repair_sweeps = 16;
    settings.debug_new_view_steps = false;

    Compute operation(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        raycast,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        queue_parameters,
        edge_parameters,
        settings);

    const auto report = operation.compute(false);
    (void)report;
}
```

This Level-3 program also exposes the HighVoronoi-specific orchestration settings (`maximum_rounds`, periodic repair sweeps, etc.) that do not exist in `VoronoiConfig`.

`ComputeHighVoronoi` currently accepts `RaycastParameters` but not a separate `CombinedRaycastOptions` object. With `CombinedRaycast` selected here, its internally constructed ray casters therefore use the standard robust Combined options. Direct `CombinedRaycastOptions` injection is available when you construct an ordinary ray caster yourself, as in the ordinary Level-3 example.

For the exact substitutions needed to change a ray caster, hash generator/container, search backend, or threading mode, continue with [Configuration and tuning](configuration.md).

## Next steps

Continue with [Constructing ordinary Voronoi diagrams](ordinary_voronoi.md). If your real application immediately needs insertion/removal, read [Incremental ordinary meshes](incremental.md); for periodic domains, read [HighVoronoi and periodic meshes](high_voronoi.md).
