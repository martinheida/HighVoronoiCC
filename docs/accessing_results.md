# Accessing mesh results

This chapter covers the user-facing result model: cells, finite vertices, infinite edges, neighbours, public numbering, and the Level-3 concepts behind stable internal identity.

## Iterate finite vertices of a cell

For an ordinary `VoronoiMesh`:

```cpp
for (const auto& vertex : mesh.vertices(cell)) {
    const auto& sigma = vertex.sigma;
    const auto& position = vertex.position;
}
```

`vertices(cell)` combines all vertex occurrences visible from the selected public cell. A stored geometric vertex exists once in the persistent database but can be referenced by every incident cell.

A `VertexRecord` also contains `vertex.address`. This is the internal database address of the persistent record. The field is useful for Level-3 implementation code and diagnostics, but it is **not** an application-level identifier and user code should not persist or interpret it.

### Level-3/developer detail: primary and secondary ownership

The native mesh also exposes:

```cpp
mesh.primary_vertices(cell);
mesh.secondary_vertices(cell);
```

These ranges expose the storage-ownership partition used by the persistent mesh. They are not different geometric classes of Voronoi vertices. Normal application code should use `vertices(cell)` and does not need to know which incident cell owns which address-list entry.

If you are changing storage, ownership, filtering, or database logic, see [HighVoronoiCC Architecture](<architecture and scientific foundation/ARCHITECTURE.md>) before relying on the primary/secondary distinction.

## Vertex signatures

`vertex.sigma` contains the generators incident to the finite Voronoi vertex in the current public numbering of the mesh/view used for iteration.

Degenerate vertices may contain more than `d+1` generators. Do not assume a fixed signature length. A simple example is a Cartesian/cubic grid: at a generic grid vertex in `d` dimensions, `2^d` cells/generators meet, so the Voronoi signature can contain `2^d` generators.

For HighVoronoi public iteration, invisible reference generators are projected back to visible public indices and stored positions are translated to the external domain.

## Infinite edges

Unbounded ordinary meshes persist rays:

```cpp
for (const auto& edge : mesh.infinite_edges()) {
    const auto& support = edge.sigma;
    const auto& origin = edge.origin;
    const auto& direction = edge.direction;
}
```

`edge.origin` is the finite endpoint from which the unbounded edge leaves the finite diagram. `edge.direction` points from that endpoint towards infinity. `edge.sigma` contains the complete public supporting-generator signature of the edge.

For degenerate unbounded edges, persistent identity uses the **full supporting edge**, not an arbitrary minimal subset of generators. The endpoint together with the full support identifies the geometric edge unambiguously, while `direction` gives its unbounded orientation. This full-support rule is part of the Level-3 topology/storage contract; see [HighVoronoiCC Architecture](<architecture and scientific foundation/ARCHITECTURE.md>).

As for finite vertices, `InfiniteEdgeRecord::address` is an internal database address and should normally be ignored by application code.

## Public indices

### Public ordinary index

A public node index is dense and follows the current visible insertion order. Initially, nodes have public indices `0, 1, 2, ...` in the order in which they were added.

After deleting a node, the gap is closed: subsequent public nodes move forward by one or more positions. Public numbering is therefore convenient for current iteration and presentation, but it is not a permanent identity across removals.

Do not store a public cell index in a long-lived external table and assume that it remains unchanged after a removal transaction.

### Boundary indices in public signatures

Suppose a public ordinary mesh currently has `N` visible nodes and `B` boundary planes. In a public vertex/edge signature:

```text
0 ... N-1           ordinary public node indices
N ... N+B-1         boundary-plane generators
```

A signature value `N + k` therefore refers to boundary plane `k`, not to `mesh.nodes()[N + k]`. The public node container contains only the ordinary nodes. User code should interpret boundary values through the boundary, or simply treat them as generator identifiers produced by the geometry.

### Level-3/developer detail: stable internal ordinary index

Persistent storage cannot use mutable public numbering as its identity. The native mesh therefore maintains stable internal ordinary indices. Persistent signatures and long-lived internal data use those stable identities and do not silently renumber them when public cells are filtered, reordered, or deleted.

For a native ordinary mesh:

```cpp
const Index stable =
    mesh.index_mapping().public_to_internal(public_cell);
```

A Level-1/2 wrapper exposes the native mesh as:

```cpp
auto& native = mesh.level3_mesh();
```

Stable identities matter when implementing extensions, comparing persistent records across views, or using detailed subsystems such as the current integration API. They are otherwise Level-3 knowledge; normal geometry iteration should stay in public numbering.

Persistent internal boundary indices use a separate high-end encoding. Do not construct that encoding in normal user code.

## Level-3/developer detail: HighVoronoi visible versus internal indices

`HighVoronoiMesh` must distinguish public visible cells from the additional invisible periodic references used during computation.

The normal user-facing mesh is the visible projection:

```cpp
auto mesh = highvoronoi::high_voronoi_mesh<Dim>(
    points, point_count, boundary);

highvoronoi::compute(mesh);

// Public visible cells only:
for (const auto& vertex : mesh.vertices(0)) {
    // ...
}
```

For Level-3 diagnostics, obtain the native owner and then its complete active internal data view:

```cpp
auto& native = mesh.level3_mesh();
auto data = native.data_mesh();
```

For an already-native `HighVoronoiMesh`, call `mesh.data_mesh()` directly. The resulting data view includes active invisible periodic reference generators. `native.internal_node_count()` additionally reports the complete stable internal node storage, including inactive/deleted slots.

For public periodic output, stay on the normal `HighVoronoiMesh`/wrapper interface unless you explicitly need implementation diagnostics.

## Neighbours

`AbstractMesh` exposes the persistent/current neighbour interface:

```cpp
std::vector<Index> neighbours;
const bool clean = mesh.neighbours(cell, neighbours);
```

The stored list is returned even when stale; the Boolean tells you whether it is current. This is intentional so update/integration code can compare OLD and NEW neighbour state.

To recompute the current facet-neighbour list:

```cpp
mesh.compute_neighbors(cell);
```

and query dirty state with:

```cpp
mesh.dirty(cell);
```

Boundary mirrors can occur as valid neighbour entries.

### Adjacents versus facet neighbours

`compute_adjecents(...)` (spelling retained by the current API) returns generators sharing at least one finite vertex. This is broader than true facet-neighbourhood.

Do not substitute adjacency for facet neighbours in interface integration.

## Neighbour multiplicity in projected geometries

For an ordinary non-periodic `VoronoiMesh`, a public node corresponds directly to one ordinary generator, so projected duplicate neighbour identities are not part of the ordinary result model.

Multiplicity becomes important when distinct internal generators can project to the same public entity, in particular:

- periodic `HighVoronoiMesh`, where invisible shifted references project back to visible generators;
- spherical/projected workflows, especially `AntipodalSphericalVoronoiMesh`, where internal images can map to one public master.

In these workflows, repeated public neighbour values can remain meaningful because they originate from distinct geometric incidences. Integration preserves neighbour-list order and multiplicity.

If you deduplicate a neighbour list for display, keep that display transformation separate from persistent aligned area/interface data.

## Result access and thread safety

Persistent database records can be written concurrently only under the library's selected lock/storage policy. Structural mesh mutation occurs at explicit safe points.

Application code should not mutate node structure while a parallel compute is active. See [Parallel computation](parallelism.md).

## Integration data

`VoronoiIntegral` stores by stable internal identity. The normal detailed-API read pattern is:

```cpp
Integral::Data::CellData cell_data;
const Index stable = mesh.index_mapping().public_to_internal(public_cell);

if (integral.data().read_cell(stable, cell_data)) {
    auto volume = cell_data.volume();
    const auto& neighbours = cell_data.neighbours();
    const auto& areas = cell_data.area();
}
```

See [Integration](integration.md) for the full lifecycle.

## Validation is a separate question

Result access tells you what is stored; it does not establish that the stored mesh is complete. Before using a new algorithmic/tuning configuration as a reference dataset, run the completeness checks described in [Validation](validation.md).
