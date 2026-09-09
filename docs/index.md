# HighVoronoiCC Manual

HighVoronoiCC is a C++17 library for constructing, incrementally updating, and integrating high-dimensional Voronoi diagrams on ordinary Euclidean, bounded, periodic, and spherical domains.

## Three API levels

HighVoronoiCC now deliberately separates ease of use from full configurability:

| level | intended use | typical surface |
|---|---|---|
| **Level 1** | normal use | `voronoi_mesh`, `high_voronoi_mesh`, `compute`, `refine`, `remove`, `voronoi_integral`, `integrate` |
| **Level 2** | performance/robustness configuration | `VoronoiConfig`, `IntegrationConfig`, ray-caster/hash/search choices, integration algorithm/execution choices |
| **Level 3** | full control and library development | native mesh/database/compute objects plus `VoronoiIntegral`, concrete integration algorithms and `Integrator` |

The levels reuse the same geometry kernel. A Level-1/2 wrapper exposes its native mesh as `level3_mesh()` rather than duplicating storage or geometry.

## Start here

Read [Getting started](getting_started.md) first. It begins with a Level-1 program that does **not** mention database types, locks, hash tables, search trees, or ray-caster classes.

Then open [Level 1-3 API comparison](level_1_3_api_comparison.md) for a direct side-by-side comparison in which the **same ordinary Voronoi problem** and the **same periodic HighVoronoi problem** are each implemented separately at Level 1, Level 2, and Level 3.

Continue with:

- [Constructing ordinary Voronoi diagrams](ordinary_voronoi.md);
- [Accessing mesh results](accessing_results.md);
- [Validation](validation.md).

Then choose the workflow you need:

- [Incremental ordinary meshes](incremental.md);
- [HighVoronoi and periodic meshes](high_voronoi.md);
- [Spherical Voronoi meshes](spherical.md);
- [Integration](integration.md);
- [Integration Level 1-3 API comparison](integration_level_1_3_api_comparison.md).

Performance-sensitive users should also read [Parallel computation](parallelism.md) and [Configuration and tuning](configuration.md).

## Architecture and scientific foundation

For implementation work and the mathematical background:

- [HighVoronoiCC Architecture](<architecture and scientific foundation/ARCHITECTURE.md>) — ownership, data flow, indexing, synchronization, invariants, and construction layers;
- [High-dimensional Voronoi concept paper](<architecture and scientific foundation/HVConceptPaper.pdf>) — mathematical construction and scientific foundation.
- [Test overview (developers only)](test_overview_developers.md) — source-level map of every registered test, the operations it performs, and the concrete outcomes it checks.

The internal AI-agent material is deliberately not part of this manual.

## License

The authoritative repository license is included directly in the generated manual: @ref highvoronoi_license "License".

## Recommended defaults

Level 1 deliberately chooses one coherent configuration:

- `double`;
- `std::uint32_t`;
- robust `CombinedRaycast`;
- KD search;
- `SingleThread` unless `MultiThread{N}` is requested;
- serial: FNV64/128 + `DirectHash`;
- parallel: FNV64/128 + `StaticHash<16>` for persistent database and edge state, `DirectHash` for the cell-local queue.

These are defaults, not hidden restrictions. Level 2 exposes the important policy choices; Level 3 exposes all current infrastructure.

## Public entry headers

```cpp
#include <highvoronoi/voronoi.hpp>           // ordinary + convenience API + refine/remove
#include <highvoronoi/high_voronoi.hpp>      // HighVoronoi + convenience API
#include <highvoronoi/spherical_voronoi.hpp> // sphere / antipodal sphere
#include <highvoronoi/integrals.hpp>         // integration
```

The ordinary entry header includes `voronoi_api.hpp`; the HighVoronoi entry header includes `highvoronoi_api.hpp`.

## Forward references worth knowing early

Even if Level 1 is sufficient initially, four topics become important as the problem grows:

- **Ray casting:** Combined is the default, while Classic and InRange remain explicit alternatives. See [Ray casting](configuration.md#ray-casting).
- **Hashing/storage:** the persistent database, vertex queue, and edge state are separate hash users. Sharding can matter for large/parallel work. See [Hashing and storage](configuration.md#hashing-and-storage).
- **Threading:** Level 1 exposes the recommended mesh-parallel path; Level 3 exposes mesh and cast axes independently. See [Parallel computation](parallelism.md).
- **Validation:** stored geometric consistency is not the same as topological completeness. See [Validation](validation.md).

