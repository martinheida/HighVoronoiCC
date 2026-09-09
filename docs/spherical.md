# Spherical Voronoi meshes

The current spherical facade is implemented and supports initial construction, refinement, and removal. It reuses the ordinary Euclidean geometry kernel rather than maintaining a separate spherical edge/raycast implementation.

Its default `compute()`, `refine()`, and `remove()` ray parameters now use robust `CombinedRaycast`, matching the ordinary and HighVoronoi default. Explicit Classic/InRange parameters remain available when required.

## Public entry point

```cpp
#include <highvoronoi/spherical_voronoi.hpp>
```

Two convenient aliases are exposed:

```cpp
highvoronoi::SphereVoronoiMesh<...>
highvoronoi::AntipodalSphericalVoronoiMesh<...>
```

## Geometric reduction

For a public sphere `S^(d-1)`, the implementation works in ambient `R^d`:

1. an internal origin generator is present;
2. the Euclidean Voronoi cell of that origin is constructed using the ordinary infrastructure;
3. the resulting vertices are radially projected to the unit sphere;
4. the origin generator is removed from public spherical signatures.

This is why the template dimension is the **ambient** Euclidean dimension. For `S^2`, use `Dimension = 3`.

## Construct a spherical mesh

The executable example uses:

```cpp
using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3; // R^3 -> S^2

using DBParams = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DBParams,
    Dimension>;
using Mesh = highvoronoi::SphereVoronoiMesh<
    Scalar,
    Dimension,
    Database>;
```

With initial visible generators:

```cpp
Mesh mesh(std::move(initial_nodes), database);

mesh.compute(
    highvoronoi::geometry::KDSearch{8, 1},
    ray_parameters);
```

The facade normalizes input nodes to the unit sphere.

## Initial append versus later refine

`append_node()` / `append_nodes()` are initial-setup APIs. After `compute()` has been called, use `refine()`:

```cpp
const auto report = mesh.refine(
    std::move(added_nodes),
    highvoronoi::geometry::KDSearch{8, 1},
    ray_parameters);
```

Calling `append_nodes()` after the initial compute is intentionally rejected.

## Removal

Remove current public spherical generators through:

```cpp
const auto report = mesh.remove(
    std::vector<Index>{Index{2}, Index{7}},
    highvoronoi::geometry::KDSearch{8, 1},
    ray_parameters);
```

As in ordinary incremental construction, public numbering can change after deletion while internal construction identities remain stable.

## Access spherical vertices

The facade exposes projected public spherical vertices:

```cpp
const auto vertices = mesh.vertices();
for (const auto& vertex : vertices) {
    // vertex.position should lie on the unit sphere
    // vertex.sigma contains public spherical generators
}
```

For diagnostics you can inspect the underlying Euclidean origin-cell mesh:

```cpp
const auto& internal = mesh.construction_mesh();
```

Treat this as an implementation-facing diagnostic view, not as the normal public spherical result.

## Antipodal/projective mode

`AntipodalSphericalVoronoiMesh` represents the quotient in which `q` and `-q` are identified. The facade maintains invisible antipodal copies internally and canonicalizes public output to one hemisphere.

In antipodal mode, duplicate public neighbour occurrences can be geometrically meaningful after projection. Do not deduplicate neighbour arrays merely because two internal images map to the same public master.

## Ray casting and threading

Spherical construction reuses the ordinary search/raycast stack. The same method selection and numerical controls apply; see [Ray casting](configuration.md#ray-casting).

Likewise, its incremental operations inherit the shared geometry infrastructure. If you introduce parallel spherical configurations, apply the same database-lock/thread-policy constraints described in [Parallel computation](parallelism.md).

## Integration status

The current general integration API does **not** document spherical integration as a supported user workflow. The repository spherical example therefore demonstrates construction/refine/remove only. Do not assume that Euclidean `VoronoiIntegral` volumes are spherical surface measures.

## Executable reference

```text
examples/spherical/spherical_refine_remove.cpp
```

The stronger regression suite is:

```text
tests/workflows/spherical/test_spherical_voronoi_mesh.cpp
```

It covers ordinary and antipodal spherical behavior in addition to refine/remove.

## What to read next

- Ordinary kernel tuning: [Configuration](configuration.md).
- Interpreting neighbour multiplicity and stable indices: [Accessing results](accessing_results.md).
- Correctness checks: [Validation](validation.md).
