# HighVoronoiCC

HighVoronoiCC is an experimental C++17 implementation of local algorithms for constructing, incrementally updating and integrating high-dimensional Voronoi diagrams on Euclidean, periodic and spherical domains.

The project is a redesign and C++ port of the ideas implemented in **HighVoronoi.jl** and developed mathematically in:

**Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025.**  
DOI: `10.20347/WIAS.PREPRINT.3197`

## Documentation

The complete user and developer manual is maintained under `docs/` and built as HTML with Doxygen.

- [Manual source index](docs/index.md)
- [Construction API: Level 1-3 comparison](docs/level_1_3_api_comparison.md)
- [Integration API: Level 1-3 comparison](docs/integration_level_1_3_api_comparison.md)
- [Developer test overview](docs/test_overview_developers.md)

<!--
After the first GitHub Pages deployment, add the public HTML manual here, e.g.
**HTML manual:** https://<github-owner>.github.io/HighVoronoiCC/
Use the actual Pages URL configured for the repository; do not guess the owner.
-->

## Quick start

The normal user-facing API deliberately hides database, lock, hash, search-tree and ray-caster plumbing:

```cpp
#include <highvoronoi/voronoi.hpp>

#include <cstddef>
#include <vector>

constexpr int Dim = 3;
double points[] = { /* node-major coordinates */ };
constexpr std::size_t point_count = /* number of points */;

using Boundary = highvoronoi::Boundary<Dim>;
using Point = Boundary::Point;

auto boundary = Boundary::cuboid(
    Point::Constant(1.0),
    Point::Zero(),
    std::vector<std::size_t>{}); // no periodic axes

auto mesh = highvoronoi::voronoi_mesh<Dim>(
    points,
    point_count,
    boundary);

highvoronoi::compute(mesh);
```

The Level-1 defaults are `double`, `std::uint32_t`, robust `CombinedRaycast`, KD search and `SingleThread`. Pass `MultiThread{N}` to the mesh factory for the recommended mesh-parallel convenience configuration. `refine(mesh, ...)` and `remove(mesh, ...)` reuse the same stored policy.

For persistent Voronoi volumes and interface areas, the integration facade is equally small:

```cpp
#include <highvoronoi/integrals.hpp>

auto integral = highvoronoi::voronoi_integral(mesh);
const auto integration_report = highvoronoi::integrate(integral); // FastPolygon
```

A scalar function can be attached directly:

```cpp
auto integral = highvoronoi::voronoi_integral(
    mesh,
    [](const auto& x) { return x.squaredNorm(); });

highvoronoi::integrate(integral);
```

### Three API levels

Construction and integration use the same three-level design:

| level | construction | integration | intended use |
|:---|:---|:---|:---|
| **Level 1** | `voronoi_mesh`, `high_voronoi_mesh`, `compute`, `refine`, `remove` with standard defaults | `voronoi_integral` + `integrate`; serial `FastPolygon` by default | normal application code |
| **Level 2** | `VoronoiConfig` exposes threading, search, ray casting, tolerances and hash/container policies | `IntegrationConfig` plus `FastPolygon`, `Polygon`, `MonteCarlo` or `HeuristicMC` and serial/parallel execution | applications that need explicit policy choices without owning low-level machinery |
| **Level 3** | native `VoronoiMesh` / `HighVoronoiMesh`, `HVDataBase`, search/raycaster objects and `ComputeVoronoi` / `ComputeHighVoronoi` | native `VoronoiIntegral`, concrete algorithm objects, `Integrator` and integration views | full control, research and library development |

The Level-1/2 wrappers retain the same persistent native mesh/integral state underneath, so incremental dirty tracking and recomputation are not separate implementations. `level3_mesh()` and `level3_integral()` provide explicit escape hatches when a workflow needs native access.

`high_voronoi_mesh<Dim>(...)` provides the corresponding convenience workflow for persistent/periodic HighVoronoi meshes.

See [Getting started](docs/getting_started.md), [Level 1-3 API comparison](docs/level_1_3_api_comparison.md), and [Integration Level 1-3 API comparison](docs/integration_level_1_3_api_comparison.md) for complete side-by-side examples.

## Performance snapshot

The current 5D benchmark suite uses `double`, `uint32_t`, `CopyKDSearch{8,1}`, the fixed seed `0x485642454e434835`, five repetitions, and release compilation with `-O3 -DNDEBUG -march=native`. The values below are median `compute_s` times from the current development machine; absolute times are hardware-dependent.

### Serial open-domain comparison with Qhull

Qhull is used only as an **external benchmark dependency**. HighVoronoiCC itself does not include or link against Qhull.

The comparison uses the same 5D point sets and Qhull's Voronoi/Delaunay mode (`v Qbb Qc Qz`):

| generators | Qhull | HighVoronoi Combined (Robust) | HV / Qhull |
|---:|---:|---:|---:|
| 1,000 | 0.476 s | 0.573 s | 1.21x |
| 2,000 | 1.205 s | 1.324 s | 1.10x |

On this random open-domain general-position workload, Qhull is faster in the serial case. The gap decreases from about **20.5% at 1,000 generators** to about **9.9% at 2,000 generators**.

The explicit `combined-fast` mode does not provide a meaningful speed advantage here: its medians are 0.581 s and 1.325 s respectively. The robust Combined fallback can therefore remain enabled by default without a measurable performance penalty in this benchmark.

### HighVoronoi ray-casting methods

The ordinary serial benchmark also compares the three construction ray-casters on bounded and open domains:

| generators | domain | Classic | InRange | Combined (Robust) |
|---:|:---|---:|---:|---:|
| 1,000 | bounded | 0.517 s | 0.762 s | **0.453 s** |
| 1,000 | open | 0.958 s | 1.156 s | **0.594 s** |
| 2,000 | bounded | 1.174 s | 1.770 s | **1.019 s** |
| 2,000 | open | 2.484 s | 2.938 s | **1.358 s** |

`CombinedRaycast` is therefore the current default performance path while retaining the robust fallback to the established Classic/InRange algorithms for numerically critical casts.

### Mesh-parallel scaling

HighVoronoi distinguishes **mesh-level parallelism** (independent reordered mesh branches) from **cast-level worker parallelism** inside one branch. The current 5D benchmark shows useful scaling for mesh-level parallelism:

| generators | domain | serial | mesh x2 | speedup | mesh x4 | speedup |
|---:|:---|---:|---:|---:|---:|---:|
| 1,000 | bounded | 0.463 s | 0.321 s | 1.44x | 0.213 s | 2.18x |
| 1,000 | open | 0.621 s | 0.446 s | 1.39x | 0.302 s | 2.06x |
| 2,000 | bounded | 1.092 s | 0.740 s | 1.48x | 0.482 s | 2.27x |
| 2,000 | open | 1.428 s | 1.020 s | 1.40x | 0.674 s | 2.12x |

The parallel benchmark deliberately uses the same thread-safe persistent storage and sharded hash configuration for its serial, 2-thread, and 4-thread cases. Its serial baseline is therefore not numerically identical to the lighter serial configuration used by the standalone Qhull/raycast benchmarks.

Cast-level worker parallelism is supported but is **not currently a performance win in this 5D workload**; synchronization and work-distribution overhead dominate. Mesh-level parallelism is the recommended parallel mode at present.

### Voronoi volume and interface integration

HighVoronoiCC also contains a topology-aware integration layer for data derived from an already constructed Voronoi tessellation. The current bounded 5D benchmark constructs a tessellation of 1,000 random generators in the unit hypercube `[0,1]^5` once and then times the integration stage independently of mesh construction.

Both the ordinary `Polygon` path and `FastPolygon` compute all **1,000 cell volumes** and collect the cell-local measures of all **51,018 interface entries**. A representative current release run gives:

| workload | Polygon | FastPolygon | speedup |
|:---|---:|---:|---:|
| 1,000 5D cell volumes + 51,018 interface entries | 29.970 s | **1.903 s** | **15.7x** |

The resulting volumes satisfy the global tessellation check to floating-point precision:

```text
sum(cell volumes), Polygon     = 0.99999999999999944
sum(cell volumes), FastPolygon = 0.99999999999999922
```

The maximum absolute difference between corresponding Polygon and FastPolygon interface measures in this run is approximately `7.2e-15`.

The speedup is not obtained by approximating or simplifying the geometry. It comes from exploiting information that is already encoded in the Voronoi mesh. Generator/index signatures provide canonical identifiers for lower-dimensional faces and subfaces. `FastPolygon` stores integration data for these lower-dimensional structures and reuses it whenever the same structure contributes to multiple higher-dimensional faces or cells.

Conceptually, the ordinary recursive integration path can revisit the same substructure through many different recursion paths, while `FastPolygon` turns this implicit recursion tree into a reusable face-incidence DAG:

```text
ordinary Polygon:
    higher-dimensional face
        -> recursively recompute lower-dimensional subfaces
        -> the same subface may be reached again from other faces/cells

FastPolygon:
    lower-dimensional subface
        -> compute once
        -> cache by Voronoi signature
        -> reuse from all incident higher-dimensional faces and cells
```

This reuse becomes increasingly important with dimension because the number of repeated lower-dimensional contributions grows rapidly. The integration benchmark therefore measures a workload for which the global structure of the tessellation carries substantial algorithmic value beyond the coordinates of each individual convex cell.

For an independent external check, the benchmark can also export the vertices of each Voronoi cell as rationalized V-representations for `lrslib`. With a `10^6` coordinate scale, the first completed lrslib cells agree with Polygon to about `1.3e-8` maximum absolute cell-volume error (`1.1e-5` relative). A higher `10^14` rationalization reduces the observed absolute discrepancy for the sampled cells to approximately `1.3e-16`. lrslib is intentionally treated as an external validation path rather than a like-for-like performance baseline: it reconstructs generic convex-polytope structure from the vertex coordinates using exact rational arithmetic, while HighVoronoi already owns the tessellation incidence structure.

The benchmark used for these measurements is:

```text
benchmarks/benchmark_integrate_voronoi_5d_lrslib.cpp
```

with `run_integration_lrslib_benchmark.sh` in the repository root. lrslib is optional and is required only for the external rational-volume comparison.

### Why the comparison is not only about the open-domain timing

Qhull is an important reference implementation for the ordinary open-domain problem, but HighVoronoi uses a different local construction strategy rather than a global Delaunay/convex-hull construction. The same HighVoronoi kernel is used for:

- open and planar-bounded domains;
- periodic and partially periodic domains through `HighVoronoiMesh`;
- generators in general and non-general position, including vertices with more than `d+1` incident generators;
- incremental insertion/removal and local recomputation.

The Qhull benchmark above is intentionally restricted to the directly comparable open-domain case. No Qhull dependency is required for normal HighVoronoiCC builds.

The benchmark programs used for these measurements are:

```text
benchmarks/benchmark_compute_voronoi_5d_three_raycasts.cpp
benchmarks/benchmark_compute_voronoi_5d_qhull.cpp
benchmarks/benchmark_compute_voronoi_5d_parallel_raycast.cpp
benchmarks/benchmark_integrate_voronoi_5d_lrslib.cpp
```

with the corresponding runner scripts in the repository root. Qhull and lrslib are optional external benchmark dependencies and are not required for normal HighVoronoiCC builds.

The core algorithm constructs Voronoi diagrams locally. Starting from known vertices, it enumerates the incident Voronoi edges and follows them by nearest-neighbour based ray casting. The implementation supports generators both in general and non-general position.

The C++ implementation now contains several cooperating layers:

- `VoronoiMesh` + `ComputeVoronoi` for ordinary Euclidean Voronoi diagrams;
- `HighVoronoiMesh` + `ComputeHighVoronoi` for persistent incremental updates and periodic reference-node closure;
- `SphericalVoronoiMesh` for spherical and antipodal/projective Voronoi geometry through an origin-cell reduction;
- `VoronoiIntegral` plus the integration algorithms for persistent volumes, interface measures and function integrals.

## Status

HighVoronoiCC is currently **experimental / alpha software**. APIs may still change.

The current implementation and regression suite validate:

- bounded Euclidean domains with planar boundaries;
- unbounded Euclidean construction with persistent infinite Voronoi edges;
- generators in general and non-general position, including degenerate vertices and edges;
- incremental insertion with `RefineVoronoi` and incremental removal with `RemoveVoronoi`;
- `HighVoronoiMesh` with visible nodes, invisible periodic reference nodes and complete periodic closure;
- partially periodic domains, e.g. periodic in selected coordinate directions and non-periodic in the others;
- periodic refinement after additional visible-node insertion and periodic closure after deletion;
- external visible-domain output while periodic reference geometry remains internal;
- spherical Voronoi meshes through the Euclidean origin-cell reduction;
- antipodal/projective spherical meshes representing `S^(d-1) / {x ~ -x}`;
- spherical `compute`, `refine` and `remove` workflows;
- serial and parallel ordinary construction;
- parallel construction using independent mesh branches;
- parallel ray-casting workers inside one mesh branch;
- combined mesh- and cast-level parallelism;
- persistent mesh-neighbour storage with per-consumer dirty tracking;
- persistent cell volumes, interface measures, bulk integrals and interface integrals;
- Polygon, FastPolygon, Monte-Carlo, Heuristic and HeuristicMC integration algorithms;
- incremental integration that recomputes only NEW/DIRTY cells and affected interfaces;
- parallel integration first passes, including a shared parallel FastPolygon facet cache;
- serial/parallel Polygon and FastPolygon equivalence tests and periodic HighVoronoi Polygon integration;
- KD-tree nearest-neighbour search through the bundled nanoflann backend;
- brute-force nearest-neighbour search as a reference backend;
- fixed-dimensional and runtime-dimensional point/node infrastructure;
- one unified node-provider contract over stored, computed and composite sources;
- optional computed/hybrid mesh engines including the current cuboid engine;
- compact configurable index and scalar types;
- configurable queue- and edge-hash containers with synchronization policies;
- geometric consistency verification;
- topological edge-completeness verification including infinite edges;
- visible-cell completeness verification for periodic HighVoronoi output.

The current regression suite includes ordinary bounded, unbounded, degenerate, parallel and incremental cases; spherical and antipodal workflows; integration and parallel-integration regressions; and periodic HighVoronoi comparisons against independently constructed explicit periodic reference meshes.

Not yet implemented as finished public workflows are the fast quasi-periodic copy/modify/paste algorithm from the reference paper and a dedicated spherical-integration facade. Construction and integration intentionally use separate narrow public entry headers (`voronoi.hpp` / `high_voronoi.hpp` and `integrals.hpp`).

## Public headers

Prefer the narrowest public entry header that matches the task.

### Ordinary Euclidean Voronoi

```cpp
#include <highvoronoi/voronoi.hpp>
```

This exposes the Level-1/2 `voronoi_mesh` / `compute` / `refine` / `remove` facade and the Level-3 ordinary `VoronoiMesh` construction path, including `ComputeVoronoi`, search/raycast selection and validation.

### Incremental / periodic HighVoronoi

```cpp
#include <highvoronoi/high_voronoi.hpp>
```

This exposes the Level-1/2 `high_voronoi_mesh` facade together with the Level-3 `HighVoronoiMesh`, `ComputeHighVoronoi`, `VisibleFirstMesh` and validation helpers.

### Spherical Voronoi

```cpp
#include <highvoronoi/spherical_voronoi.hpp>
```

This exposes `SphereVoronoiMesh` and `AntipodalSphericalVoronoiMesh`.

### Integration

```cpp
#include <highvoronoi/integrals.hpp>
```

`integrals.hpp` exposes the Level-1/2 integration facade together with the persistent integral data model and native integration driver. The facade provides `IntegrationConfig`, the `FastPolygon`, `Polygon`, `MonteCarlo` and `HeuristicMC` selectors, and `voronoi_integral(...)` / `integrate(...)`.

The source-integral-dependent native `HeuristicAlgorithm` remains a Level-3 facility and can be included explicitly when needed:

```cpp
#include <highvoronoi/integration/heuristic_integrator.hpp>
```

### Optional computed mesh engines

```cpp
#include <highvoronoi/mesh_engines.hpp>
```

This exposes the optional computed/hybrid mesh-engine infrastructure such as `CuboidMeshEngine`.

### Construction convenience umbrella

```cpp
#include <highvoronoi/highvoronoi.hpp>
```

The current umbrella imports ordinary, HighVoronoi, mesh-engine and spherical construction modules. **It does not import the integration module**, so integration code should include `integrals.hpp` explicitly.

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

### Minimal periodic workflow

A periodic `HighVoronoiMesh` starts from the same visible generator set a user would provide for an ordinary bounded problem. Periodic copies are **not** inserted manually; `ComputeHighVoronoi` creates and closes the required invisible references.

For example, the repository example `examples/incremental/periodic_high_voronoi_refine.cpp` uses a unit cube that is periodic in `x` and `y` but non-periodic in `z`:

```cpp
Boundary boundary = Boundary::cuboid(
    point(1.0, 1.0, 1.0),
    point(0.0, 0.0, 0.0),
    std::vector<Index>{Index{0}, Index{1}});

HighMesh mesh(
    Index{Dimension},
    boundary,
    std::in_place,
    DatabaseUnits,
    database_parameters());

for (const Point& p : initial_points) {
    (void)mesh.append_visible_node(p);
}

HighCompute initial_compute(
    mesh,
    highvoronoi::geometry::KDSearch{8, 1},
    ray_parameters(),
    highvoronoi::SingleThread{},
    highvoronoi::SingleThread{},
    database_parameters(),
    edge_parameters());
(void)initial_compute.compute();
```

Refinement uses exactly the same visible-node API followed by another `ComputeHighVoronoi` transaction:

```cpp
for (const Point& p : refinement_points) {
    (void)mesh.append_visible_node(p);
}

HighCompute refine_compute(
    mesh,
    highvoronoi::geometry::KDSearch{8, 1},
    ray_parameters(),
    highvoronoi::SingleThread{},
    highvoronoi::SingleThread{},
    database_parameters(),
    edge_parameters());
(void)refine_compute.compute();
```

**Important:** every user-visible node, including nodes added during later refinement, must lie inside the fixed external/visible boundary. `append_visible_node()` checks this and throws for points outside the external domain. Only automatically generated invisible periodic reference nodes may lie outside the visible domain; the internal periodic boundary expands as needed to contain them.

Deletion is likewise staged through the persistent mesh and consumed by the next `ComputeHighVoronoi` call. The full non-periodic and periodic workflows are in:

```text
examples/incremental/high_voronoi_refine_remove.cpp
examples/incremental/periodic_high_voronoi_refine.cpp
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

## Spherical Voronoi meshes

`SphericalVoronoiMesh` reuses the ordinary Euclidean kernel instead of maintaining a second spherical geometry implementation. Internally it adds an origin generator, computes the Voronoi cell of that origin in `R^d`, removes the origin from the public signatures and radially projects the resulting finite vertices to the unit sphere `S^(d-1)`.

Two compile-time modes are exposed:

- `SphereVoronoiMesh<...>` for ordinary spherical Voronoi diagrams;
- `AntipodalSphericalVoronoiMesh<...>` for the quotient `S^(d-1) / {x ~ -x}`, where every visible generator has an invisible antipodal partner and public nodes/vertices are canonicalized to one hemisphere.

A spherical mesh is computed once and then updated through its own incremental facade:

```cpp
using Sphere = highvoronoi::SphereVoronoiMesh<double, 3, Database>;

auto database = std::make_shared<Database>(
    16384,
    DatabaseParameters{highvoronoi::DirectHash{16384}});

Sphere sphere(initial_nodes, database);
sphere.compute();

const auto refine_report = sphere.refine(more_nodes);
const auto remove_report = sphere.remove({Index{2}, Index{7}});
```

Input nodes are normalized by the spherical facade. Antipodal mode additionally canonicalizes visible nodes and maintains the invisible `-q` references. `vertices()` returns the projected public spherical vertices; `construction_mesh()` exposes the underlying Euclidean origin-cell mesh for diagnostics.

The full spherical regression, including ordinary `S^2`, antipodal `S^3/{+/-}` and refine/remove checks, is:

```text
tests/workflows/spherical/test_spherical_voronoi_mesh.cpp
```

## Integration: volumes, interfaces and function integrals

Integration is a persistent layer attached to an already constructed mesh. The Level-1/2 facade owns one native `VoronoiIntegral`, so repeated calls preserve the same per-consumer dirty tracking used by the Level-3 API. After refinement or removal, only NEW/DIRTY cells and affected interfaces are recomputed.

The three API levels are:

- **Level 1:** create a persistent integral with `voronoi_integral(mesh)` and call `integrate(integral)`; geometry-only storage and serial `FastPolygon` are selected automatically.
- **Level 2:** use `IntegrationConfig` plus an algorithm selector (`FastPolygon`, `Polygon`, `MonteCarlo`, `HeuristicMC`) and an execution policy.
- **Level 3:** construct `VoronoiIntegral`, concrete algorithm objects and `Integrator` directly.

The full side-by-side comparison is in [Integration Level 1-3 API comparison](docs/integration_level_1_3_api_comparison.md).

The current algorithm family is:

| facade/native algorithm | role |
|:---|:---|
| `FastPolygon` / `FastPolygonAlgorithm` | deterministic topology-aware integration with cached lower-dimensional facet/subfacet data; Level-1 default |
| `Polygon` / `PolygonAlgorithm` | deterministic recursive reference path for polytope geometry and function integrals |
| `MonteCarlo` / `MonteCarloAlgorithm` | stochastic ray-based volume/interface/integrand estimation |
| `HeuristicMC` / `HeuristicMCAlgorithm` | Monte-Carlo geometry followed by heuristic function quadrature |
| `HeuristicAlgorithm` | Level-3 function integration from an existing geometry-source integral |

### Level-1 FastPolygon volume/interface example

```cpp
#include <highvoronoi/integrals.hpp>

// mesh has already been computed
auto integral = highvoronoi::voronoi_integral(mesh);
const auto report = highvoronoi::integrate(integral);

decltype(integral)::CellData cell;
if (integral.read_cell(0, cell)) {
    const double volume = cell.volume();
    const auto& neighbours = cell.neighbours();
    const auto& areas = cell.area();
}
```

The Level-1 geometry default stores cell volumes and interface areas, disables bulk/interface function-integral arrays and runs serial `FastPolygon`. Public-to-stable index conversion during readback is hidden by `read_cell(public_cell, ...)`.

For one scalar function, function-integral storage is enabled automatically with one component:

```cpp
auto integral = highvoronoi::voronoi_integral(
    mesh,
    [](const auto& x) { return x.squaredNorm(); });

highvoronoi::integrate(integral);
```

Vector-valued functions use the overload with an explicit component count.

### Level-2 integration choices

```cpp
highvoronoi::IntegrationConfig config;
config.data_options = {true, true, false, false};
config.integral_components = 0;

auto integral = highvoronoi::voronoi_integral(mesh, config);

const auto report = highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{},
    highvoronoi::ParallelIntegrationExecution{8});
```

The algorithm selector can be changed locally without changing persistent result ownership:

```cpp
highvoronoi::integrate(integral, highvoronoi::Polygon{});

highvoronoi::MonteCarlo monte_carlo;
monte_carlo.options.interface_rays = 4000;
highvoronoi::integrate(integral, monte_carlo);
```

`HeuristicMC` is also exposed at Level 2 for function-bearing integrals. The source-integral-dependent `HeuristicAlgorithm` remains Level 3 because its dependency graph is inherently explicit.

### Level-3 native integration

The native storage and algorithm API remains available without wrappers:

```cpp
highvoronoi::IntegralDataOptions options;
options.volume = true;
options.area = true;
options.bulk_integral = false;
options.interface_integral = false;

using Integral = highvoronoi::VoronoiIntegral<Mesh, double, double>;
Integral integral(mesh, 0, options);

auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral);
const auto report = highvoronoi::integrate(integral, algorithm);
```

Level-1/2 owners expose `level3_integral()` for workflows that need to cross into this native API without giving up persistent ownership.

For ordinary unbounded meshes, cells incident to persistent infinite edges are excluded from finite cell integrators. Their geometric volume is stored as `+infinity`; unavailable bulk/interface function-integral components use `NaN`, and finite reciprocal interfaces are recovered from the bounded side where possible. Bounded and periodic meshes remain on the normal finite integration path.

`HighVoronoiMesh` uses the same Level-1/2 calls. Its dedicated integration view integrates visible/public cells while preserving persistent stable identities behind periodic reference geometry.

### Parallel integration

Parallel integration is implemented. The first pass over the NEW/DIRTY prefix can be split into contiguous worker ranges:

```cpp
const auto report = highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{},
    highvoronoi::ParallelIntegrationExecution{4});
```

Workers own private integration views and algorithm scratch. Cleanup/publication remains a serial transactional phase where the algorithm requires global reconciliation. `FastPolygon` uses a shared, sharded recursive facet cache across its parallel workers while still verifying full canonical facet keys after hash lookup.

The regression suite currently checks serial/parallel equivalence for Polygon and FastPolygon, parallel Monte-Carlo first passes with serial cleanup, Heuristic/HeuristicMC execution, incremental dirty updates, and periodic `HighVoronoiMesh` Polygon integration. The principal executable references are:

```text
tests/integration/test_parallel_integration.cpp
tests/integration/test_high_voronoi_polygon_parallel.cpp
tests/integration/test_fast_polygon_integrator.cpp
tests/integration/test_fast_polygon_integrator_incremental.cpp
```

The 5D performance/validation benchmark described above is:

```text
benchmarks/benchmark_integrate_voronoi_5d_lrslib.cpp
```

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
- `ClassicRaycast`, `InRangeRaycast` and `CombinedRaycast`, including the robust Combined fallback policy;
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

Tests are registered with thematic CTest labels, so focused subsets can also be run with `ctest -L ...`, for example:

```bash
ctest --test-dir build/release -L spherical --output-on-failure
ctest --test-dir build/release -L integrals --output-on-failure
ctest --test-dir build/release -L periodic --output-on-failure
```

For concurrency changes it is additionally useful to maintain sanitizer builds and exercise the parallel smoke cases there.

## Examples

The `examples/` directory contains complete user-facing construction workflows.

### One-shot ordinary construction

`examples/general_position.cpp`

- bounded four-dimensional general-position construction;
- `ClassicRaycast`;
- geometric verification.

`examples/cubic_grid.cpp`

- strongly degenerate Cartesian grid;
- `InRangeRaycast`;
- exact expected bounded vertex count.

### Incremental and periodic workflows

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
- repeated `ComputeHighVoronoi` transactions.

`examples/incremental/periodic_high_voronoi_refine.cpp`

- partially periodic HighVoronoi construction;
- invisible periodic reference-node closure;
- further visible-node insertion near periodic faces;
- visible-cell completeness and visible-domain output.

### Parallel construction

The `examples/parallel/` directory demonstrates the two construction threading axes:

- `parallel_cast.cpp`: cast-level parallelism;
- `parallel_mesh.cpp`: mesh-branch parallelism;
- `parallel_combined.cpp`: both levels enabled.

### Spherical and integration executable references

Dedicated small `examples/` programs for the newer spherical and integration APIs have not yet been split out. Their current executable reference cases live in the regression suite:

```text
tests/workflows/spherical/test_spherical_voronoi_mesh.cpp
tests/integration/test_fast_polygon_integrator.cpp
tests/integration/test_parallel_integration.cpp
tests/integration/test_high_voronoi_polygon_parallel.cpp
```

These tests exercise the same public APIs documented above and are built by default when `HIGHVORONOI_BUILD_TESTS=ON`.

## Validation and regression strategy

The most important construction diagnostic paths are:

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
- 3D serial/parallel construction comparison;
- full versus range-limited computation;
- ordinary refine/remove workflows;
- HighVoronoi incremental versus batched construction;
- periodic HighVoronoi against an independent explicit `3 x 3` reference tiling for two periodic axes;
- periodic HighVoronoi refinement and removal against fresh periodic references;
- spherical `S^2` and antipodal/projective `S^3/{+/-}` construction, refinement and removal;
- neighbour storage/dirty propagation and parallel neighbour publication;
- Polygon/FastPolygon geometry and nonlinear convergence tests;
- FastPolygon cache collision, incremental and parallel shared-cache regressions;
- Monte-Carlo, Heuristic and HeuristicMC integration;
- serial/parallel integration equivalence and periodic HighVoronoi integration;
- integration dependency-cycle rejection.

CMake keeps a registry of test source files and warns when a `tests/test_*.cpp` source is present but not registered, reducing the chance that new regressions silently fall outside CTest.

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

- stabilize the remaining public API, install/consumer surface and release packaging;
- add dedicated compact user examples for spherical construction and integration;
- improve cast-worker termination/load balancing when a shared queue is temporarily empty while another worker still has in-flight work;
- benchmark hash-container sharding, construction scaling and parallel integration scaling;
- add install/consumer and sanitizer release checks;
- classify or remove historical/development alternate raycaster/systematic headers;
- continue performance profiling without changing the validated construction, periodization and integration invariants.

### Longer term

- fast quasi-periodic copy/modify/paste workflows from the reference paper;
- optimized fused `direct_cast` search backends;
- additional nearest-neighbour backends where profiling justifies them;
- optional disk-backed persistent storage;
- a dedicated spherical-integration facade if application requirements justify it;
- additional quadrature/integral algorithms beyond the current Polygon/FastPolygon/Monte-Carlo/Heuristic family;
- further performance optimization and benchmarking.

No `SerialMesh` abstraction is part of the current source tree; ordinary incremental, periodic HighVoronoi, spherical and integration workflows operate through their dedicated persistent meshes and temporary compute/integration views.

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
