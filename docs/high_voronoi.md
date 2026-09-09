# HighVoronoi and periodic meshes

`HighVoronoiMesh` is the persistent workflow for visible generators, incremental updates, and periodic reference-node closure. The Level-1 facade gives it the same user vocabulary as ordinary construction.

For a direct comparison of the **same periodic HighVoronoi problem** implemented at Levels 1, 2, and 3, see [Getting started: HighVoronoi Level 1/2/3 comparison](getting_started.md#the-same-periodic-highvoronoi-problem-at-levels-1-2-and-3).

## Level 1

```cpp
#include <highvoronoi/high_voronoi.hpp>

constexpr int Dim = 3;

double points[] = {
    0.15, 0.20, 0.30,
    0.75, 0.22, 0.44,
    0.35, 0.76, 0.58,
    0.69, 0.71, 0.81
};

using Boundary = highvoronoi::Boundary<Dim>;
using Point = Boundary::Point;

// x/y periodic, z non-periodic. Axes are zero-based.
auto boundary = Boundary::cuboid(
    Point::Constant(1.0),
    Point::Zero(),
    std::vector<std::size_t>{0, 1});

auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
    points,
    4,
    boundary);

const auto initial = highvoronoi::compute(mesh);
```

The user does not construct `ComputeHighVoronoi`, the database, locks, reference nodes, or the temporary compute mapping. Periodic references and closure are internal.

The Level-1 defaults match ordinary construction: `double`, `uint32_t`, robust Combined ray casting, KD search, SingleThread, and automatically sized FNV-based hashes.

## Refine and remove

```cpp
double added[] = {
    0.41, 0.53, 0.27,
    0.62, 0.38, 0.72
};

const auto refined = highvoronoi::refine(mesh, added, 2);
const auto removed = highvoronoi::remove(
    mesh,
    {decltype(mesh)::Index{1}});
```

Unlike ordinary refinement, HighVoronoi `refine()` appends visible generators and then runs the complete HighVoronoi transaction. `remove()` marks the requested visible nodes (and their invisible copies) deleted and then runs the same transaction boundary.

For a multithreaded convenience mesh, each operation reuses the stored mesh-thread count; an explicit final count overrides it for that call.

## Visible nodes versus periodic references

The persistent owner distinguishes:

```text
visible public nodes
    application generators/cells

stable internal nodes
    visible nodes plus invisible periodic reference copies

compute-public nodes
    temporary dense bijective numbering used by the geometry kernel
```

Invisible reference nodes are real computational generators but are not separate user-visible cells. Public vertex iteration projects reference indices/positions back into the external visible domain.

This is why a periodic HighVoronoi mesh cannot be reduced to “ordinary mesh + periodic labels” at the user level.

## External and internal boundary

The external boundary is the user domain and remains fixed. HighVoronoi may move only periodic planes of its **internal** artificial boundary to contain reference nodes. The final sparse periodic repair handles finite OLD-only vertices that can become visible after this internal-boundary motion.

Visible nodes appended by the user must still lie inside the external boundary (up to the configured tolerance).

## Parallel Level 1

```cpp
auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    highvoronoi::MultiThread{4});

highvoronoi::compute(mesh);
```

As for ordinary Level 1, `MultiThread` selects mesh-level parallelism and a thread-safe/sharded storage configuration. Internal refine/remove/periodic-repair phases inherit the same convenience policy.

See [Parallel computation](parallelism.md).

## Level 2

Pass an explicit `VoronoiConfig` to `high_voronoi_mesh` to choose ray-caster and hash/search policies without exposing the orchestration types:

```cpp
auto config = highvoronoi::make_voronoi_config<
    highvoronoi::MultiThread,
    highvoronoi::CombinedRaycast>(point_count, Dim, highvoronoi::MultiThread{8});

config.raycast.variance_tolerance = 9e-14;

auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
    points,
    point_count,
    boundary,
    config);
```

For large periodic problems, hash/container choices can matter more than the visible node count suggests because periodic reference closure may create substantially more internal nodes. See [Hashing and storage](configuration.md#hashing-and-storage).

## Level 3

The explicit path remains:

```text
HighVoronoiMesh
    + ComputeHighVoronoi
        -> RemoveVoronoi / RefineVoronoi
        -> periodic reference closure
        -> sparse periodic repair
```

Use it when you need direct orchestration reports/settings, custom compute policies, cast-level threading, explicit queue/edge parameter types, or custom database ownership.

Every convenience HighVoronoi wrapper exposes this owner through:

```cpp
auto& native = mesh.level3_mesh();
```

## Integration

The integration subsystem has a dedicated HighVoronoi view that stores persistent results for visible cells while preserving distinct internal reference geometry during computation. The simple integration facade is intentionally a separate follow-up API; current integration usage is documented in [Integration](integration.md).
