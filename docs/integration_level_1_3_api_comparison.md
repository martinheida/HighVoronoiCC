# Integration Level 1-3 API comparison

This page solves the same integration task at all three API levels. The geometry is assumed to have already been constructed and stored in `mesh`.

The task is deliberately simple:

- store Voronoi-cell volumes;
- store interface areas;
- use deterministic FastPolygon integration;
- run serially;
- read cell 0 afterwards.

The numerical result is intended to be the same. What changes is how much infrastructure the caller spells out.

## Level 1: normal application code

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

This selects automatically:

```text
persistent result owner      VoronoiIntegral<double,double>
volume                        enabled
interface area                enabled
bulk function integral       disabled
interface function integral  disabled
algorithm                     FastPolygon
execution                     serial
internal DB unit lengths      65536
```

The public-cell-to-stable-cell conversion during readback is also hidden.

## Level 2: all facade-level choices explicit

The same problem can be written with every Level-2 integration choice visible:

```cpp
#include <highvoronoi/integrals.hpp>

highvoronoi::IntegrationConfig config;
config.data_options.volume = true;
config.data_options.area = true;
config.data_options.bulk_integral = false;
config.data_options.interface_integral = false;
config.integral_components = 0;
config.area_database_unit_length = 65536;
config.integral_database_unit_length = 65536;

auto integral = highvoronoi::voronoi_integral(mesh, config);

const auto report = highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{},
    highvoronoi::SerialIntegrationExecution{});

decltype(integral)::CellData cell;
if (integral.read_cell(0, cell)) {
    const double volume = cell.volume();
    const auto& neighbours = cell.neighbours();
    const auto& areas = cell.area();
}
```

Level 2 still owns the native persistent integral and constructs the concrete algorithm object for you.

Typical substitutions are local.

Reference Polygon instead of FastPolygon:

```cpp
highvoronoi::integrate(
    integral,
    highvoronoi::Polygon{},
    highvoronoi::SerialIntegrationExecution{});
```

Parallel FastPolygon:

```cpp
highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{},
    highvoronoi::ParallelIntegrationExecution{8});
```

Monte Carlo:

```cpp
highvoronoi::MonteCarlo monte_carlo;
monte_carlo.options.interface_rays = 4000;
monte_carlo.options.bulk_samples_per_ray = 200;

highvoronoi::integrate(
    integral,
    monte_carlo,
    highvoronoi::ParallelIntegrationExecution{8});
```

## Level 3: native persistent storage and algorithm objects

The same geometry-only FastPolygon task written through the native API is:

```cpp
#include <highvoronoi/integrals.hpp>
#include <highvoronoi/integration/fast_polygon_integrator.hpp>

highvoronoi::IntegralDataOptions options;
options.volume = true;
options.area = true;
options.bulk_integral = false;
options.interface_integral = false;

auto& native_mesh = mesh.level3_mesh();
using Mesh = std::remove_reference_t<decltype(native_mesh)>;
using Integral = highvoronoi::VoronoiIntegral<
    Mesh,
    double,
    double>;

Integral integral(
    native_mesh,
    0,       // function-integral components
    options,
    65536,   // area database unit length
    65536);  // integral database unit length

auto algorithm =
    highvoronoi::make_fast_polygon_algorithm(integral);

const auto report = highvoronoi::integrate(
    integral,
    algorithm,
    highvoronoi::SerialIntegrationExecution{});

Integral::Data::CellData cell;
const auto stable =
    native_mesh.index_mapping().public_to_internal(0);

if (integral.data().read_cell(stable, cell)) {
    const double volume = cell.volume();
    const auto& neighbours = cell.neighbours();
    const auto& areas = cell.area();
}
```

At Level 3 the user explicitly owns the native result type, chooses the concrete algorithm object, and performs public-to-stable index conversion during direct persistent-data access.

## The same comparison with one scalar function

For

```cpp
const auto function = [](const auto& x) {
    return x.squaredNorm();
};
```

Level 1 is:

```cpp
auto integral = highvoronoi::voronoi_integral(mesh, function);
highvoronoi::integrate(integral);
```

Level 2 is:

```cpp
highvoronoi::IntegrationConfig config;
config.data_options = {true, true, true, true};
config.integral_components = 1;

auto integral = highvoronoi::voronoi_integral(
    mesh,
    config,
    function);

highvoronoi::integrate(
    integral,
    highvoronoi::FastPolygon{},
    highvoronoi::SerialIntegrationExecution{});
```

Level 3 is:

```cpp
highvoronoi::IntegralDataOptions options;
options.volume = true;
options.area = true;
options.bulk_integral = true;
options.interface_integral = true;

auto& native_mesh = mesh.level3_mesh();
using Mesh = std::remove_reference_t<decltype(native_mesh)>;
using Integral = highvoronoi::VoronoiIntegral<
    Mesh,
    double,
    double>;

Integral integral(native_mesh, 1, options);
auto algorithm = highvoronoi::make_fast_polygon_algorithm(
    integral,
    function);

highvoronoi::integrate(integral, algorithm);
```

## Ordinary versus HighVoronoi

The Level-1 and Level-2 calls are intentionally identical for an ordinary convenience mesh and a HighVoronoi convenience mesh:

```cpp
auto ordinary_integral = highvoronoi::voronoi_integral(ordinary_mesh);
auto high_integral = highvoronoi::voronoi_integral(high_mesh);

highvoronoi::integrate(ordinary_integral);
highvoronoi::integrate(high_integral);
```

The factory unwraps the convenience mesh to its native persistent mesh. The native integration-view selection then chooses ordinary dense mapping or HighVoronoi projected/reference handling.

At Level 3 those distinct native mesh/view details are visible to the caller.

## When to use each level

| Level | Use it when |
|---|---|
| **1** | you want volumes/areas, optionally one function, with FastPolygon defaults |
| **2** | you want to choose storage fields, algorithm, Monte-Carlo options, or integration worker count |
| **3** | you need source-dependent Heuristic integration, custom algorithm/store types, direct persistent storage, planner internals, or algorithm diagnostics |

The levels do not define different integration mathematics. They expose progressively more of the same persistent integration stack.

See [Integration](integration.md) for semantics, incremental updates, unbounded-cell handling, and algorithm details.
