# Integration: volumes, interfaces, and function integrals

HighVoronoiCC exposes integration through the same three API levels used by mesh construction.

- **Level 1** keeps one persistent integral owner and uses FastPolygon automatically.
- **Level 2** exposes result-storage choices, integration algorithm selection, and serial/parallel execution.
- **Level 3** is the existing native `VoronoiIntegral` / algorithm / `Integrator` API.

For the same integration problem written at all three levels, see [Integration Level 1-3 API comparison](integration_level_1_3_api_comparison.md).

Construction and integration remain deliberately separate. The mesh owns geometry and neighbour state. The persistent integral owns derived result/history storage and its own dirty tracker. A concrete integration algorithm supplies the numerical traversal policy.

## Public header

Normal users need only:

```cpp
#include <highvoronoi/integrals.hpp>
```

This includes the Level-1/Level-2 integration facade as well as the native Level-3 integration API.

## Level 1: geometry with FastPolygon

Given an already computed ordinary or HighVoronoi mesh:

```cpp
auto integral = highvoronoi::voronoi_integral(mesh);
const auto report = highvoronoi::integrate(integral);
```

The default means:

```text
algorithm             FastPolygon
execution             serial
volume storage        enabled
interface area        enabled
bulk function data    disabled
interface function    disabled
```

`integral` is persistent. Reuse the same object after mesh refinement/removal. Its dedicated dirty tracker lets a later `integrate(integral)` update only NEW/DIRTY cells rather than reconstructing every result.

A second call on an unchanged clean mesh can therefore return `report.updated_cells == 0`.

## Level 1: one function

For a scalar-valued function:

```cpp
auto integral = highvoronoi::voronoi_integral(
    mesh,
    [](const auto& x) {
        return x.squaredNorm();
    });

highvoronoi::integrate(integral);
```

This enables one bulk-integral component and one interface-integral component in addition to volume and interface area. FastPolygon remains the default algorithm.

For a vector-valued function, state the component count explicitly:

```cpp
auto integral = highvoronoi::voronoi_integral(
    mesh,
    3,
    [](const auto& x) {
        std::array<double, 3> value{
            x[0],
            x[1],
            x.squaredNorm()
        };
        return value;
    });
```

The component count is explicit so result storage can be allocated without evaluating the function merely to discover its output shape.

## Reading Level-1/Level-2 results

The facade reads by current public cell number:

```cpp
decltype(integral)::CellData cell;

if (integral.read_cell(public_cell, cell)) {
    const double volume = cell.volume();
    const auto& neighbours = cell.neighbours();
    const auto& areas = cell.area();
    const auto& bulk = cell.bulk_integral();
    const auto& interfaces = cell.interface_integral();
}
```

The facade performs public-to-stable conversion internally. The underlying storage remains the native flat `CellData` layout; there is no `optional`/status wrapper around every numerical result.

The central alignment invariant remains:

```text
neighbours[k] <-> area[k] <-> interface_integral[k,*]
```

Order and multiplicity are meaningful. Never independently sort or deduplicate one of these arrays.

## Level 2: storage configuration

`IntegrationConfig` controls persistent result storage and internal database unit lengths:

```cpp
highvoronoi::IntegrationConfig config;
config.data_options.volume = true;
config.data_options.area = true;
config.data_options.bulk_integral = false;
config.data_options.interface_integral = false;
config.integral_components = 0;
config.area_database_unit_length = 65536;
config.integral_database_unit_length = 65536;


auto integral = highvoronoi::voronoi_integral(mesh, config);
```

The facade default is intentionally geometry-only even though the low-level `IntegralDataOptions` aggregate has its own native defaults.

If bulk or interface function storage is enabled, `integral_components` must be greater than zero.

A function can be combined with an explicit config:

```cpp
highvoronoi::IntegrationConfig config;
config.data_options = {true, true, true, true};
config.integral_components = 2;

auto integral = highvoronoi::voronoi_integral(
    mesh,
    config,
    function);
```

## Level 2: algorithm choice

FastPolygon explicitly:

```cpp
highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{});
```

Deterministic reference Polygon:

```cpp
highvoronoi::integrate(
    integral,
    highvoronoi::Polygon{});
```

Monte Carlo:

```cpp
highvoronoi::MonteCarlo choice;
choice.options.interface_rays = 4000;
choice.options.bulk_samples_per_ray = 200;

highvoronoi::integrate(integral, choice);
```

For a function-bearing facade integral, the same stored function is used for bulk and interface Monte-Carlo function integration.

HeuristicMC is also available for a function-bearing facade integral:

```cpp
highvoronoi::HeuristicMC choice;
choice.options.interface_rays = 4000;

highvoronoi::integrate(integral, choice);
```

The source-integral-based `HeuristicAlgorithm` has a more specialized dependency relationship and remains a Level-3 workflow rather than being hidden behind an artificial simple tag.

## Level 2: parallel integration

Execution policy is independent of mesh-construction threading:

```cpp
const auto report = highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{},
    highvoronoi::ParallelIntegrationExecution{4});
```

The first NEW/DIRTY pass may run in parallel. Cleanup/publication remains transactional where the selected algorithm requires global reconciliation.

A mesh may therefore be constructed serially and integrated in parallel, or constructed in parallel and integrated serially.

## Unbounded cells

> **Warning:** ordinary unbounded Voronoi diagrams can contain cells for which ordinary finite-domain integration is not defined. These cells are deliberately removed from the normal integration work list before Polygon, FastPolygon, MonteCarlo, Heuristic or HeuristicMC sees them.

The default stored semantics are:

```text
unbounded cell volume                         +infinity
bulk function integral not computed           NaN per component
provably unbounded interface area              +infinity
function integral on such an interface         NaN per component
finite reciprocal interface available          copied from finite neighbour
remaining interface not computed               NaN
```

This keeps the normal numerical arrays flat. `NaN` means "not computed by the finite-domain integrator"; it is not an `optional` replacement. `+infinity` is used only where the geometric measure itself is known to be infinite.

An unbounded cell never runs through a finite-domain cell integrator merely to produce an infinite volume.

The integration report additionally records:

```cpp
report.unbounded_cells;
report.unbounded_interfaces;
```

`unbounded_interfaces` counts stored interface occurrences, not necessarily unique geometric facets.

The principal HighVoronoi use cases remain bounded and/or periodic domains, where this exceptional path is normally absent.

## Persistent result model

At Level 3, `VoronoiIntegral<Mesh,...>` stores results by stable internal cell identity, not by mutable public ordering. This is what lets historical results survive public renumbering and be updated incrementally after mesh changes.

The Level-1/Level-2 `ApiVoronoiIntegral` owns exactly one such native object. Access it when required through:

```cpp
auto& native = integral.level3_integral();
```

## Level 3: native storage and algorithm objects

Geometry-only storage can be constructed explicitly as:

```cpp
highvoronoi::IntegralDataOptions options;
options.volume = true;
options.area = true;
options.bulk_integral = false;
options.interface_integral = false;

using Integral = highvoronoi::VoronoiIntegral<Mesh, double, double>;
Integral integral(mesh, 0, options);
```

FastPolygon:

```cpp
auto algorithm = highvoronoi::make_fast_polygon_algorithm(integral);
const auto report = highvoronoi::integrate(integral, algorithm);
```

Reference Polygon:

```cpp
auto algorithm = highvoronoi::make_polygon_algorithm(integral);
const auto report = highvoronoi::integrate(integral, algorithm);
```

Explicit execution:

```cpp
const auto report = highvoronoi::integrate(
    integral,
    algorithm,
    highvoronoi::ParallelIntegrationExecution{4});
```

A native algorithm object may also be run against the native integral owned by the facade:

```cpp
auto algorithm = highvoronoi::make_fast_polygon_algorithm(
    integral.level3_integral());

highvoronoi::integrate(integral, algorithm);
```

Use Level 3 for source-dependent heuristic integration, custom algorithm objects/stores, direct `IntegralData` control, algorithm diagnostics, or changes to the integration planner itself.

## Polygon and FastPolygon

`PolygonAlgorithm` is the deterministic recursive reference path.

`FastPolygonAlgorithm` solves the same topology-aware polygon integration problem but caches lower-dimensional facets/subfacets by canonical stable-internal signatures and reuses them across higher-dimensional faces/cells. Its cache verifies complete canonical keys before treating two facets as identical; hash collisions affect probing/diagnostics rather than geometric identity.

FastPolygon is the Level-1 default because it preserves the deterministic polygon formulation while avoiding repeated lower-dimensional work.

## Monte Carlo, Heuristic, and HeuristicMC

`MonteCarloAlgorithm` provides stochastic ray-based estimation of geometry/integrands.

`HeuristicAlgorithm` derives function integration from a separate source integral that provides geometry. It can force source updates and checks integration dependencies for cycles. Because this relationship explicitly involves two persistent integrals and a source algorithm, it remains a Level-3 workflow.

`HeuristicMCAlgorithm` combines Monte-Carlo geometry with heuristic function integration after geometry cleanup and is exposed by the Level-2 `HeuristicMC` tag for the common single-integral case.

## Incremental integration and dirty state

The mesh owns persistent neighbour state plus dirty propagation to independent consumers. Each persistent integral owns a dedicated tracker. After refine/remove or HighVoronoi updates, the next integration pass plans only NEW/DIRTY cells and affected interfaces.

Do not replace this with an unrelated application dirty-bit system unless your application is tracking genuinely different semantics.

Reference tests include:

```text
tests/integration/test_polygon_integrator_incremental.cpp
tests/integration/test_fast_polygon_integrator_incremental.cpp
tests/storage/neighbours/test_neighbour_dirty_tracking.cpp
```

## HighVoronoi integration

The same Level-1/Level-2 factory accepts a HighVoronoi convenience mesh:

```cpp
auto integral = highvoronoi::voronoi_integral(high_mesh);
highvoronoi::integrate(integral);
```

Internally HighVoronoi uses its dedicated integration view:

- persistent output belongs to visible cells;
- invisible periodic references remain distinct shifted generators during geometric traversal;
- those references are not projected away prematurely inside the integrator.

The principal periodic regression is:

```text
tests/integration/test_high_voronoi_polygon_parallel.cpp
```

## Executable reference

The existing detailed example is:

```text
examples/integration/fast_polygon.cpp
```

It intentionally shows more Level-3 machinery than a normal Level-1 application needs.

## Forward references

- Direct API comparison: [Integration Level 1-3 API comparison](integration_level_1_3_api_comparison.md).
- Mesh correctness first: [Validation](validation.md).
- Stable identity and neighbour multiplicity: [Accessing results](accessing_results.md).
- Construction threading: [Parallel computation](parallelism.md).
- Mesh configuration/hashing: [Configuration](configuration.md#hashing-and-storage).

## Current limitation

Spherical integration is not part of the documented integration API at this snapshot. Do not interpret Euclidean `VoronoiIntegral` volumes as spherical surface measures.
