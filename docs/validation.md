# Validation and regression checks

HighVoronoiCC is designed to produce high-quality persistent Voronoi geometry, and the current regression suite exercises bounded/unbounded, degenerate, incremental, periodic, spherical, and parallel workflows. In the directly comparable open-domain cases checked during current Qhull comparison work, the robust Combined path has not shown a known result discrepancy.

The numerically hardest cases are not usually ordinary vertices inside the generator cloud, but **far-field vertices** of open diagrams whose coordinates can lie many orders of magnitude away from the generators. Equal-distance and nearest-neighbour decisions then lose absolute resolution as floating-point scale grows. The current robust Combined path contains correction and fallback machinery specifically for numerically critical casts.

Historically, older experimental Combined variants did expose isolated far-field endpoint discrepancies. In the observed diagnostic cases, a finite endpoint could be missed and the corresponding direction represented as an additional infinite edge instead. Such a result can still be internally edge-closed even though it differs from the intended finite geometry. This is why an independent/reference comparison remains valuable for extreme open-domain numerical cases in addition to the internal completeness validator.

The validators in this chapter should therefore be understood primarily as **Level-3/developer and regression tools**. They encode mathematical vertex/edge conditions from the HighVoronoi construction model; they are not a requirement that every production application revalidate every mesh after every operation.

HighVoronoi distinguishes **geometric consistency of stored records** from **topological completeness of the stored result**. This distinction is important enough that the validators should not be treated as interchangeable.

For a Level-1/Level-2 convenience mesh, pass `mesh.level3_mesh()` to the current validators. The wrapper does not change geometry or storage; it only owns the configuration facade.

## verify_mesh

Use:

```cpp
const auto report = highvoronoi::verify_mesh(
    mesh.level3_mesh(), // use mesh directly for a Level-3 native mesh
    tolerance,
    true);
```

This checks stored finite-vertex occurrences for geometric consistency, including the relevant equal-distance and nearest-neighbour conditions implemented by the validator.

It answers approximately:

> "Are the finite vertices that are present geometrically valid?"

It does **not** answer:

> "Did the algorithm find every required endpoint and close every edge?"

A mesh can therefore pass `verify_mesh()` while still missing topology.

## Complete mesh validation

Use:

```cpp
const auto report = highvoronoi::verify_mesh_complete(
    mesh.level3_mesh(), // use mesh directly for a Level-3 native mesh
    tolerance,
    true);

if (!report.complete()) {
    // inspect report.consistency and edge-closure diagnostics
}
```

The complete validator reconstructs global edge incidence using full supporting edges.

For a complete finite edge, exactly two finite endpoints are expected. For a complete unbounded edge, one finite endpoint plus one persistent infinite-edge occurrence is expected. A third occurrence is invalid.

Degenerate minimal edge representations at one persistent vertex are deduplicated by their full supporting edge before global counting.

Use this validator for:

- new ray-caster or tolerance configurations;
- serial/parallel equivalence work;
- incremental regressions;
- unbounded meshes;
- correctness checks before performance benchmarking.

A successful result means that the **stored topology is internally closed under these rules**. It is not by itself an independent proof that an extreme far-field finite endpoint has not been replaced by a numerically plausible infinite continuation; use independent/reference comparison for that stronger question when it matters.

## Selected-cell validation

Periodic HighVoronoi has internal reference cells that exist as computational scaffolding. To validate only the visible cells, use a suitable view and:

```cpp
const auto report = highvoronoi::verify_mesh_complete_cells(
    view,
    begin_cell,
    end_cell,
    tolerance,
    true);
```

The normal periodic pattern is:

```cpp
highvoronoi::VisibleFirstMesh<HighMesh> view(mesh);

const auto report = highvoronoi::verify_mesh_complete_cells(
    view,
    Index{0},
    view.visible_end(),
    Scalar{1e-12},
    true);
```

This checks public visible-cell closure without demanding that invisible reference cells constitute an independent external tessellation.

## Mesh comparison

`compare_meshes(...)` is used by the regression suite to compare compatible results bidirectionally by node data, signatures, and positions. This is useful when validating:

- serial versus parallel construction;
- full versus range-limited construction;
- incremental versus fresh batched reconstruction;
- periodic HighVoronoi versus explicit periodic tilings.

For open-domain numerical investigations, an external implementation such as Qhull can provide an additional independent comparison because it does not share HighVoronoi's local ray-cast traversal.

Consult the Doxygen reference and the relevant tests for the exact overloads used in the current snapshot.

## Strong regression patterns in the test suite

The current repository includes end-to-end references for:

- bounded general position;
- exact Cartesian degeneracy;
- unbounded construction with infinite rays;
- serial/mesh-parallel/cast-parallel/combined construction;
- range consistency;
- ordinary incremental insertion/removal;
- HighVoronoi incremental versus batched construction;
- partially periodic HighVoronoi versus an explicit periodic tiling;
- periodic incremental insertion/deletion;
- spherical and antipodal spherical workflows;
- Polygon/FastPolygon and serial/parallel integration equivalence.

See [Examples and tests](examples_and_tests.md) for concrete file names.

## Numerical verification inside ray casting

Ray casting also contains semantic vertex verification and numerical correction. A tiny equal-distance variance is not treated as a complete forward positional accuracy certificate: conditioning-aware correction can fall back to software Float128.

This is distinct from validating the final global mesh. A locally well-corrected vertex does not prove that every edge was explored, and a globally edge-closed open mesh does not by itself prove agreement with an independent far-field reference.

See [Ray casting](configuration.md#ray-casting) for the public tolerances.

## Recommended validation policy

During development:

1. run the focused component test for the subsystem you changed;
2. run `verify_mesh_complete()` on a small deterministic executable case;
3. compare against an independent/reference construction where one exists;
4. only then run large performance benchmarks;
5. after parallel changes, run the serial/parallel regression and sanitizer builds used by the project.

For production-like application runs where full validation is too expensive, keep a smaller representative validation configuration in CI rather than replacing completeness checks with `verify_mesh()` everywhere.
