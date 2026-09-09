# Level 1-3 API comparison

This chapter shows the three HighVoronoiCC API levels side by side by solving the **same problem three times**. The geometry and intended result remain unchanged; only the amount of configuration, ownership, and orchestration made explicit by the application changes.

Use this page when deciding which API level fits your application:

- **Level 1**: specify the geometric problem and let HighVoronoiCC choose the standard configuration;
- **Level 2**: keep the same high-level workflow but explicitly configure ray casting, search, hashing, tolerances, and threading policies;
- **Level 3**: construct and own the native database, mesh, search tree, ray caster, queue/edge parameters, and compute orchestration directly.

The first comparison uses an ordinary bounded Voronoi diagram. The second uses periodic HighVoronoi. For the meaning of individual tuning parameters, see [Configuration and tuning](configuration.md). For a first minimal program, see [Getting started](getting_started.md).

## Ordinary Voronoi: the same problem at Levels 1, 2 and 3

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

## HighVoronoi: the same periodic problem at Levels 1, 2 and 3

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

## Choosing a level

The examples are intentionally redundant: the point is to make the additional machinery visible. Start with Level 1 unless you need a capability shown only in a lower level. Level 2 is normally the right place for application-level performance and robustness tuning. Level 3 is appropriate when the application needs direct ownership/control of the native construction machinery or when developing HighVoronoiCC itself.
