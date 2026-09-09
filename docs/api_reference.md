# API reference map

This page is a hand-written map of the public HighVoronoiCC API. The documentation build also feeds the `include/highvoronoi/*.hpp` headers to Doxygen as explicitly enumerated files, which provides the generated **Classes** and **Files** indexes with declaration/file documentation. `src/`, tests, benchmarks, unrelated repository files, Markdown below `include/`, and AI-agent material remain outside the Doxygen input set.

## Level-1/2 public facade

Ordinary:

```text
include/highvoronoi/voronoi_api.hpp
```

Key symbols:

```text
ApiVoronoiMesh
voronoi_mesh
compute
refine
remove
VoronoiConfig
make_voronoi_config
```

HighVoronoi:

```text
include/highvoronoi/highvoronoi_api.hpp
```

Key symbols:

```text
ApiHighVoronoiMesh
high_voronoi_mesh
compute
refine
remove
```

Shared facade configuration lives in:

```text
include/highvoronoi/api_common.hpp
```

The facade defaults to `double`, `uint32_t`, Combined ray casting and SingleThread. `MultiThread` selects the convenience mesh-parallel configuration. The native object is available through `level3_mesh()`.

## Level-3 ordinary construction

```text
include/highvoronoi/voronoi.hpp
```

Important symbols:

```text
VoronoiMesh
HVDataBase
ComputeVoronoi
RefineVoronoi
RemoveVoronoi
geometry::make_search_tree
make_raycaster
verify_mesh
verify_mesh_complete
verify_mesh_complete_cells
```

## Level-3 HighVoronoi / periodic

```text
include/highvoronoi/high_voronoi.hpp
```

Important symbols:

```text
HighVoronoiMesh
ComputeHighVoronoi
VisibleFirstMesh
```

## Spherical

```text
include/highvoronoi/spherical_voronoi.hpp
```

Important symbols:

```text
SphericalVoronoiMesh
SphereVoronoiMesh
AntipodalSphericalVoronoiMesh
```

The default spherical compute/refine/remove ray parameters now use Combined as well.

## Integration

Public entry point:

```text
include/highvoronoi/integrals.hpp
```

Level-1/2 facade implementation:

```text
include/highvoronoi/integration/integration_api.hpp
```

Key facade symbols:

```text
ApiVoronoiIntegral
voronoi_integral
integrate
IntegrationConfig
FastPolygon
Polygon
MonteCarlo
HeuristicMC
SerialIntegrationExecution
ParallelIntegrationExecution
```

Level 1 uses persistent geometry storage plus serial FastPolygon by default. A scalar function can be attached directly; vector-valued functions use an explicit component count. `read_cell(public_index, buffer)` hides stable-internal index conversion.

Level 2 exposes persistent result-storage configuration, algorithm choice and integration execution policy while retaining the same owner.

Native Level-3 symbols include:

```text
VoronoiIntegral
IntegralDataOptions
Integrator
FastPolygonAlgorithm
PolygonAlgorithm
MonteCarloAlgorithm
HeuristicAlgorithm
HeuristicMCAlgorithm
make_fast_polygon_algorithm
make_polygon_algorithm
make_heuristic_algorithm
make_heuristic_mc_algorithm
```

The complete direct comparison is in [Integration Level 1-3 API comparison](integration_level_1_3_api_comparison.md).

## Configuration symbols worth bookmarking

```text
SingleThread
MultiThread
VoronoiConfig
DataBaseParams
EdgeBufferParams
DirectHash
StaticHash<N>
DynamicHash<>
FNV64_128HashGenerator
ClassicRaycast
InRangeRaycast
CombinedRaycast
RaycastParameters
CombinedRaycastOptions
CombinedPrecisionPolicy
CombinedFallbackMethod
```

## Internal types you normally should not instantiate directly

```text
EdgeIterator
VoronoiWorker
SystematicVoronoi
IncrementalVoronoiComputeMesh
HighVoronoiComputeMesh
```

If your work requires modifying these types, use `ARCHITECTURE.md` and the focused tests rather than treating the Level-1 facade as the implementation architecture.
