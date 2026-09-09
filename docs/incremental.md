# Incremental ordinary meshes

A Level-1 ordinary mesh remembers the construction policy chosen when it was created, so incremental insertion and removal do not require the user to rebuild the search/raycast/database configuration.

## Level 1: refine

```cpp
double added[] = {
    0.31, 0.43, 0.66,
    0.71, 0.52, 0.37
};

const auto report = highvoronoi::refine(mesh, added, 2);
```

The flat input uses the mesh dimension automatically. For a `MultiThread` convenience mesh, the stored mesh-thread count is reused. You may override it for one operation:

```cpp
const auto report = highvoronoi::refine(mesh, added, 2, 8);
```

The returned object is the native `RefineVoronoi::Report`.

Insertion uses the fixed-boundary invariant that every genuinely new finite vertex contains at least one NEW generator. New cells are computed and affected OLD vertices are invalidated; insertion does not run an unnecessary general second compute over all affected OLD cells.

## Level 1: remove

```cpp
const auto report = highvoronoi::remove(
    mesh,
    {decltype(mesh)::Index{2}, decltype(mesh)::Index{9}});
```

Removal is geometrically different from insertion: deleting a generator can expose topology made entirely from surviving OLD generators. The underlying `RemoveVoronoi` therefore destroys selected topology and then performs a closure compute over affected surviving cells.

An optional final thread count has the same meaning as for `compute` and `refine`.

## Stable identities after deletion

Deleting a public node can rebuild the dense public numbering. Persistent signatures use stable internal identities instead. Therefore do not store a public cell number as a permanent external identity across removal unless your application explicitly tracks remapping.

See [Accessing mesh results](accessing_results.md#public-and-stable-internal-indices).

## Level 2

The incremental convenience functions reuse the mesh's `VoronoiConfig`:

- `config.search`;
- `config.raycast`;
- the stored threading policy/count;
- direct queue hash parameters;
- configured edge hash parameters.

Because the persistent database was already constructed, changing `config.database_hash` after mesh creation does not rebuild that database. Treat database layout as a **construction-time** setting. Runtime ray/search/tolerance fields may be adjusted deliberately, but any such change should be followed by completeness regression testing.

See [Configuration and tuning](configuration.md).

## Level 3

The existing explicit classes remain available:

```cpp
highvoronoi::RefineVoronoi<Mesh> refine(...);
const auto refine_report = refine.compute();

highvoronoi::RemoveVoronoi<Mesh> remove(...);
const auto remove_report = remove.compute();
```

Level 3 is required when you need direct access to the remover's two-phase lifecycle (`remove()` now, `compute()` later), custom backend policies, or separate mesh/cast threading axes.

## Ordinary versus HighVoronoi incremental work

Ordinary incremental construction assumes fixed boundary geometry. Periodic HighVoronoi may expand an artificial internal periodic boundary and create invisible reference nodes; its orchestration additionally performs periodic closure and sparse repair. Do not reproduce that logic manually with ordinary `RefineVoronoi`; use [HighVoronoi and periodic meshes](high_voronoi.md).
