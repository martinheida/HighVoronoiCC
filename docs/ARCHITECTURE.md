# HighVoronoiCC Architecture
**Status:** current implementation architecture and extension roadmap  
**Scope:** geometry, storage, indexing, search, Voronoi construction, synchronization, validation, and planned extensions  
**Reference algorithm:** Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025
---

## 1. Purpose and current status
HighVoronoiCC is a C++17 implementation and redesign of the local high-dimensional Voronoi construction algorithm developed in the HighVoronoi project.
This document describes the architecture of the **current C++ implementation**. It is intentionally not a second algorithm paper and not a frozen API specification. Mathematical derivations and the complete algorithmic background belong to the reference paper; this document focuses on software responsibilities, ownership, data flow, synchronization, and implementation boundaries.
The current implementation provides:
- compact configurable scalar and index types;
- stable internal node numbering separated from public numbering;
- stored, computed, and hybrid node access;
- planar boundaries represented through active mirror nodes;
- append-only persistent vertex storage;
- configurable hash tables and synchronization policies;
- brute-force and nanoflann search backends;
- general and degenerate edge enumeration;
- ray casting, descent, vertex verification, and numerical correction;
- complete systematic cell exploration for one mesh branch;
- single-threaded or multi-worker ray casting inside that branch;
- geometric mesh validation and regression tests.
The current `MeshThreading = MultiThread` path is deliberately not completed yet. `ComputeVoronoi` already contains the branch-level ownership and vertex-registration boundary required by the future implementation, but constructing and executing several communicating mesh views currently throws an explicit not-implemented error.
All indices in this document are zero-based unless stated otherwise.
---

## 2. Core design principles
### 2.1 Stable storage is separate from public presentation
Public ordinary node indices are dense and may change after filtering, reordering, or composition. Persistent vertex signatures use stable internal indices that remain valid for the lifetime of their owning storage.
Consequences:
- deleting or reordering public nodes does not rewrite stored signatures;
- views may present another public order without changing persistent storage;
- boundary indices remain disjoint from ordinary internal indices;
- persistent duplicate detection operates on canonical internal signatures;
- public numbering, storage numbering, and backend-native search numbering are separate concepts.
### 2.2 Compile-time configuration is used where it changes semantics or layout
Compile-time choices include:
- scalar and index type;
- static dimension where applicable;
- node access mode;
- lock type;
- hash generator and table implementation;
- search-tree implementation;
- mesh and cast threading policy.
Runtime configuration is used for quantities such as:
- thread count;
- hash capacities and block sizes;
- KD-tree leaf size and build thread count;
- database block size;
- geometric tolerances.
### 2.3 Hot paths reuse memory
Repeated search, edge construction, ray casting, signature conversion, and database access should recycle caller-owned or object-owned buffers.
This applies to:
- search backend data;
- vertex queues;
- queue and edge hashes;
- edge-iterator storage;
- ray-cast signatures;
- orthogonalization matrices and vectors;
- boundary candidate buffers;
- public/internal signature conversion buffers.
Convenience APIs may allocate. The construction core should not depend on repeated temporary allocation.
### 2.4 Generic interfaces and backend hot paths are different
Algorithms that must work with stored, computed, and hybrid nodes use the universal copy interface:

```cpp
nodes.copy_node(index, target);
```

Coordinate-intensive backends such as nanoflann use direct coordinate access:

```cpp
nodes.get_data(index, coordinate);
```

A search backend must not construct a temporary point object for every coordinate request.
### 2.5 Backend-native indices never become mesh indices implicitly
A backend may use `std::size_t` or another native index type while the mesh uses `std::uint16_t`, `std::uint32_t`, or another compact type.
The conversion boundary is explicit:

```text
backend-native index
        -> checked conversion
        -> Mesh::Index
```

No component above the concrete backend may assume identical index representations.
### 2.6 Exact reference implementations remain available
The brute-force search tree is the reference implementation for optimized search backends. The same principle applies throughout the code base:
1. preserve a clear correctness path;
2. optimize behind a compatible interface;
3. compare optimized results against the reference path or independent geometric checks.
### 2.7 Synchronization belongs to shared data, not to workers
The current construction architecture does not put a global mutex around the algorithm and does not make `VoronoiWorker` aware of thread policies.
Instead, synchronization is attached to the structures that are actually shared:
- the persistent database;
- the branch vertex queue;
- the edge hash;
- the edge-iterator cache;
- a shared mesh vertex iterator.
Worker-local geometric state remains lock-free.
---

## 3. Architecture overview
The current construction stack is:

```text
User configuration
        |
        v
Mesh / Nodes / Boundary / Database
        |
        +------------------+
        |                  |
        v                  v
SearchTree              EdgeIterator
        |                  |
        +------> RayCaster-+
                   |
                   v
            VoronoiWorker
                   |
                   v
          SystematicVoronoi
                   |
                   v
            ComputeVoronoi
```

The ownership structure of the construction algorithm is:

```text
ComputeVoronoi
    |
    +-- SystematicVoronoi [branch 0]
    |       |
    |       +-- VoronoiWorker 0
    |       +-- VoronoiWorker 1
    |       +-- ...
    |
    +-- SystematicVoronoi [future branch 1]
    +-- ...
```

The dependency direction is deliberate:
- hashing knows nothing about geometry;
- the database knows nothing about ray casting;
- search trees know nothing about construction queues;
- `VoronoiWorker` knows nothing about global branch scheduling;
- `SystematicVoronoi` knows only cast-level threading;
- only `ComputeVoronoi` knows the complete construction topology.
---

## 4. Data, geometry, and indexing model
### 4.1 Points
The point layer uses Eigen-compatible types:
- `StaticPoint<Scalar, Dim>`;
- `DynamicPoint<Scalar>`;
- static and dynamic read-only point views;
- `highvoronoi::Dynamic` as the runtime-dimension marker.
Stored points may be exposed through zero-copy views. Computed points own their returned values.
### 4.2 Node access modes
The node abstraction supports three compile-time modes.
#### Stored
Every ordinary node has stable contiguous storage. This is the natural representation for arbitrary input point clouds and KD-tree input.
#### Computed
A node is generated from its index when requested. Typical future uses include structured grids and translated periodic copies.
#### Hybrid
Some nodes are stored while others are computed. Active boundary mirrors are one current use of this composition model.
Algorithms independent of the access mode use `copy_node()` and `get_data()` rather than assuming stable pointers.
### 4.3 Boundary mirrors
A planar boundary represents an oriented half-space. During construction of one Voronoi cell, each active boundary plane is represented by a virtual mirror of the active cell generator.
This turns the geometric boundary plane into an ordinary Voronoi bisector between the cell generator and its mirror.
The current extended-node layer supports active cell mirrors and a precomputed reflection variant. Inactive mirrors must never enter search results.
### 4.4 Public, internal, and boundary indices
The mesh separates dense public numbering from stable internal numbering:

```text
public ordinary indices   0 ... public_node_count - 1
internal ordinary indices stable zero-based slots
```

Boundary mirrors use a disjoint internal high-end encoding. `max(Index)` remains reserved as an invalid marker.
The exact mapping is owned by the mesh. Persistent code must not infer internal identity from public position.
### 4.5 Vertex representation and ownership
A finite Voronoi vertex is represented by:

```text
sigma : sorted generating-node signature
r     : vertex position
```

Persistent signatures are canonicalized in stable internal numbering.
The smallest ordinary internal generator is the primary storage owner. Other ordinary generators receive secondary address registrations. The vertex record itself is stored only once.
Storage ownership and algorithmic edge ownership are separate concepts.
---

## 5. Persistent storage and hash structures
### 5.1 `HVDataBase`
`HVDataBase<Lock, DataBaseParams>` is the current persistent append-only vertex database.
Important properties:
- data is stored in fixed-size blocks;
- records may cross block boundaries;
- an atomic top counter reserves disjoint write ranges;
- structural block growth is protected by the configured lock;
- duplicate detection uses the configured queue-hash family;
- address `0` denotes duplicate insertion;
- deletion removes the signature from the hash and sets `sigma_length = 0` as a tombstone;
- record storage is not compacted immediately.
Stable addresses allow per-node address lists to remain valid after logical deletion.
### 5.2 Hashing roles
Hashing is deliberately modular because different lifetimes have different requirements.
Current roles include:
- persistent duplicate detection in `HVDataBase`;
- cell-local duplicate claims in the vertex queue;
- cell-local edge occurrence counting.
The code provides multiple hash functions, configurable hash generators, queue hash tables, edge hash tables, and direct/static/dynamic container modes.
These structures must remain separate even when they use the same implementation family. Persistent and cell-local state have different ownership and reset semantics.
### 5.3 Address lists
Primary and secondary vertex addresses are attached to ordinary nodes through thread-policy-aware read/write lists.
The database owns the record; node lists own only references to that record.
---

## 6. Mesh architecture
### 6.1 `AbstractMesh`
`AbstractMesh` defines the common mesh behavior used by algorithms and wrappers:
- public/internal signature conversion;
- boundary index conversion;
- duplicate-safe vertex storage;
- primary and secondary vertex iteration;
- combined cell-vertex iteration;
- node and vertex filtering;
- tombstone-aware record handling.
The public iterator interface deliberately hides the physical database representation.
### 6.2 `VoronoiMesh`
`VoronoiMesh` is the main concrete stored mesh.
It owns:
- stable ordinary node storage;
- dense public/internal mappings;
- extended boundary nodes;
- the persistent database;
- primary and secondary address lists.
Public node deletion changes the public mapping without renumbering stable internal nodes.
### 6.3 `ReorderedMeshView`
`ReorderedMeshView` is a non-owning facade that changes public order while retaining:
- stable node storage;
- persistent internal signatures;
- database records;
- address lists;
- the underlying boundary.
Its main architectural purpose is to let algorithms operate on a selected public ordering without rebasing persistent data.
### 6.4 `SerialMesh`
`SerialMesh` composes compatible child meshes under one top-level numbering and shared storage semantics.
It supports cached or calculated public/internal conversion and establishes the abstraction needed for later composite or computed meshes.
A top-level vertex address should therefore be treated conceptually as an opaque locator owned by the mesh that produced it, even though the current primary iterator path remains database-backed.
### 6.5 Vertex iterator recycling
The current `AbstractMesh::VertexRange::Iterator` reuses its internal vertex, signature, and position buffers while traversing a range.
Consequences for hot paths:
- range-for and pre-increment reuse capacity;
- a first larger signature may still grow capacity;
- post-increment may copy owning buffers and should be avoided in hot loops;
- dereferenced records are iterator-owned and are overwritten on the next increment.
Algorithms that need to retain a vertex must copy it explicitly.
---

## 7. Search and geometric primitives
### 7.1 Search-tree abstraction
The CRTP search-tree layer combines backend ordinary-node candidates with active boundary candidates handled by the common layer.
Current common operations include:

```cpp
nn(data, skip)
knn(data, count, skip)
inrange(data, radius, skip)
rebuild()
```

Each backend exposes reusable `SearchData` containing query, result, node, and backend-specific scratch buffers.
### 7.2 Brute-force search
`BruteForceSearchTree` scans ordinary public nodes directly and is the exact correctness reference for optimized search structures.
### 7.3 nanoflann search
`NanoflannSearchTree` maintains a persistent KD-tree over ordinary public nodes.
Key boundaries are:
- nanoflann native indices are checked before conversion to `Mesh::Index`;
- coordinate access uses `nodes.get_data()`;
- backend arrays are retained in reusable `SearchData`;
- active cell mirrors remain outside the persistent KD-tree because they change with the current cell.
HighVoronoiCC currently vendors its local `nanoflann.hpp`; the architecture assumes that exact bundled interface rather than an arbitrary system version.
### 7.4 `RayCaster`
`RayCaster` owns the geometric nearest-neighbour/range-search logic required to advance a Voronoi ray.
Its responsibilities include:
- cell-local activation of boundary mirrors;
- finite/infinite ray status;
- candidate signature construction;
- vertex verification;
- numerical correction of candidate vertices.
Ray-cast strategy is selected independently from the search backend, currently including `ClassicRaycast` and `InRangeRaycast` policies.
### 7.5 `direct_cast`
The search-tree abstraction contains a future `direct_cast` hook for a fused `RCCombined`-style branch-and-bound traversal. Its current backend contract is still a placeholder.
Correctness therefore does not depend on `direct_cast`; the implemented construction uses the existing search operations through `RayCaster`.
### 7.6 `EdgeIterator`
`EdgeIterator` enumerates candidate Voronoi edges at a vertex.
It supports:
- the general-position case;
- the degenerate case with more than `dimension + 1` generators;
- reusable fast-edge storage;
- an `OnQueueEdges` mode used when a vertex first enters the cell queue;
- an `OnSysVoronoi` mode used while systematically exploring that vertex.
The expensive degenerate iterator state is cached and shared where safe.
### 7.7 Minimal edge versus full supporting edge
Degenerate edge enumeration distinguishes two signatures:
- the **minimal edge**, which determines algorithmic ownership;
- the **full supporting edge**, which may contain additional generators.
A full supporting edge may legitimately contain generators with global indices smaller than the active cell. These generators must not be globally filtered out.
Repeated occurrences of the same geometric edge are required by the edge-counting algorithm. Completion is determined through edge occurrence state, not by requiring the full supporting edge to have the active cell as its smallest generator.
This distinction is essential for highly degenerate configurations such as Cartesian grids.
---

## 8. Voronoi construction
### 8.1 `ComputeVoronoi`
`ComputeVoronoi` is the top-level construction coordinator.
It owns the complete execution topology:

```text
MeshThreading
    one branch or several communicating mesh branches
CastThreading
    one worker or several workers inside each branch
VoronoiThreading
    MultiThread iff MeshThreading or CastThreading is MultiThread
```

`ComputeVoronoi` therefore knows both threading axes. It selects the branch queue lock from `VoronoiThreading` and creates the branch-local `SystematicVoronoi` objects.
The current implementation completes the one-branch path. `MeshThreading = MultiThread` currently throws before constructing parallel `ReorderedMeshView`/switching branches.
### 8.2 Vertex registration boundary
Newly discovered vertices always pass through `ComputeVoronoi::register_vertex()`.
In the current one-branch implementation this operation:
1. converts the worker point into mesh storage form;
2. performs duplicate-safe persistent insertion through the mesh/database;
3. queues the vertex in the active `SystematicVoronoi` branch;
4. counts newly stored vertices.
The same method is the prepared communication boundary for future mesh branches:

```text
source public sigma
        -> stable internal sigma
        -> target public sigma
        -> target branch queue
```

This keeps cross-branch communication out of `VoronoiWorker` and `SystematicVoronoi`.
### 8.3 `SystematicVoronoi`
`SystematicVoronoi` performs the complete cell loop for exactly one mesh branch.
It knows `CastThreading`, but it does **not** know whether the parent has one branch or several branches.
It owns:
- the branch id and local cell range;
- the atomic current cell;
- the cell-local vertex queue;
- the cell-local edge hash;
- a prototype `RayCaster`;
- a prototype `EdgeIterator`;
- the configured `VoronoiWorker` objects;
- active mirror-index scratch storage.
`SystematicVoronoi` itself owns no mutex. Synchronization is contained in the structures that need it.
### 8.4 Cell lifecycle
For each local cell, `SystematicVoronoi::compute()` performs:

```text
1. reset cell-local queue, edge hash, and boundary state
2. iterate already stored vertices of the cell
3. queue those existing vertices
4. if no starting vertex exists, perform descent
5. persist and queue the descent vertex
6. systematically explore queued vertices until the queue is empty
7. advance to the next cell
```

Existing mesh vertices do **not** go through `ComputeVoronoi::register_vertex()`, because they are already persistent. They are inserted only into the current cell's queue/edge state.
New vertices found by descent or ray traversal do go through `ComputeVoronoi` because persistence and future branch communication belong there.
### 8.5 Cell reset
`reset(cell)` clears cell-local queue and edge state, activates the correct boundary mirrors on the prototype and all worker ray casters, and only then publishes the new atomic current-cell value.
No additional generation barrier is required after a cell has been fully explored. Once systematic exploration of that cell has completed, its cell-local structures are no longer needed.
### 8.6 Queue semantics
The branch queue combines two responsibilities under one queue lock:
- first-claim duplicate detection for the current cell;
- storage of reusable queued vertex objects.
Exactly one thread can successfully claim a signature. That first claimant performs the `OnQueueEdges` pass and is therefore responsible for determining whether the vertex still has incomplete edges.
If all edges are already complete, the vertex need not enter the work queue.
### 8.7 Systematic vertex exploration
For a queued vertex, a worker:
1. resets its searching `EdgeIterator` in `OnSysVoronoi` mode;
2. enumerates locally relevant minimal edges;
3. updates the shared cell edge hash;
4. skips edges already complete or not owned by the active cell;
5. follows incomplete edges with `walk_ray()`;
6. verifies or corrects finite candidates as required;
7. sends new candidates through the top-level registration boundary.
The edge hash therefore acts as the cell-local combinatorial completion mechanism.
### 8.8 `VoronoiWorker`
`VoronoiWorker` contains no global threading policy and owns no locks.
Each worker owns reusable geometric state:
- an independent `RayCaster`;
- a queueing `EdgeIterator`;
- a searching `EdgeIterator`;
- current/candidate/descent vertex buffers;
- walk and descent signatures;
- public/internal conversion buffers;
- orthogonalization state;
- temporary points and directions.
The two worker edge iterators share the prototype FEI cache. Separate iterators are required because queue processing may trigger edge enumeration while systematic edge traversal is already active.
### 8.9 Descent
If a cell has no known stored vertex, worker 0 performs descent from the cell generator.
The current implementation repeatedly finds directions orthogonal to the accumulated minimal support, ray casts in those directions, handles opposite-direction retry for infinite casts, projects the final point inside the boundary, performs correction when appropriate, and verifies the resulting vertex before registration.
Descent is a geometric initialization operation, not a separate persistence path.
---

## 9. Threading and synchronization
### 9.1 Two independent threading axes
HighVoronoiCC separates:
- **MeshThreading**: parallel mesh branches;
- **CastThreading**: parallel geometry workers within one branch.
This separation is intentional. Branch communication and local ray-cast parallelism have different shared state and therefore different lock requirements.
### 9.2 Lock selection
The current lock ownership can be summarized as follows:
| Shared structure | Lock selection | Reason |
|---|---|---|
| persistent mesh/database | mesh database configuration | new vertices may be stored concurrently |
| branch vertex queue | `VoronoiThreading::RWLock` | may receive local worker and future cross-branch registrations |
| edge hash | `CastThreading::RWLock` | shared only among workers of one branch |
| FEI cache | `CastThreading::RWLock` | shared only among workers using cached degenerate-edge data |
| shared mesh vertex iterator | `CastThreading::RWLock` | mutable iterator distributed among workers |
| worker scratch data | none | owned by one worker |
| `SystematicVoronoi` object | none | synchronization is encapsulated below it |
Any parallel construction path requires the persistent mesh/database to use `ReadWriteLock` because workers may store vertices concurrently.
### 9.3 Four configurations
The two threading axes produce four architectural cases:
| MeshThreading | CastThreading | Branch queue | Edge/FEI state | Current status |
|---|---|---|---|---|
| Single | Single | `EmptyLock` | `EmptyLock` | implemented |
| Single | Multi | `ReadWriteLock` | `ReadWriteLock` | implemented |
| Multi | Single | `ReadWriteLock` | `EmptyLock` | branch execution pending |
| Multi | Multi | `ReadWriteLock` | `ReadWriteLock` | branch execution pending |
The queue uses the combined `VoronoiThreading` policy because even a branch with one local worker may later receive vertices from another branch.
### 9.4 Worker execution
`SystematicVoronoi::run_workers()` is the only place where branch-local worker threads are created.
With `CastThreading = SingleThread`, worker 0 is called directly. With `CastThreading = MultiThread`, all configured workers execute the requested phase and are joined before the next cell phase begins.
### 9.5 Shared mesh iteration
`AbstractMesh` iterators have mutable traversal state and are not themselves shared concurrently.
`ConcurrentVertexIterator` wraps one iterator and serializes only `next()`:

```text
lock
    copy current mesh vertex into worker-owned storage
    advance iterator
unlock
```

The underlying mesh iterator therefore remains simple and unchanged. Concurrency is introduced only where the algorithm actually shares it.
---

## 10. Numerical robustness
### 10.1 Precision model
Ordinary geometry uses the configured scalar type. Sensitive basis construction and orthogonalization may use the extended-precision helpers in `detail/float.hpp` and `normal_solver.hpp`.
The precision layer provides a higher-precision type based on sufficiently precise `long double` where available and Boost.Multiprecision otherwise.
The intent is to use extended precision selectively rather than storing the complete mesh in an expensive scalar type.
### 10.2 Vertex verification and correction
A finite candidate vertex must satisfy equal-distance geometry for its generators and must be consistent with nearest-neighbour search.
`RayCaster` exposes separate correction and verification operations so that numerical improvement never replaces a combinatorial validity test.
The core rule is:
> a corrected point is accepted only if the corrected candidate still satisfies the intended Voronoi geometry.
### 10.3 Edge-direction correction
Degenerate edge construction monitors cycle/orthogonality error. Directions may be corrected when the accumulated error exceeds the configured tolerance.
This is especially important in higher dimensions and strongly degenerate configurations.
### 10.4 Tolerances
Geometric tolerances belong to explicit algorithm/search/raycast parameter objects rather than hidden literals. Absolute and relative error criteria must not be conflated.
---

## 11. Validation and regression strategy
### 11.1 Component tests
The source tree contains focused readable tests for:
- locks;
- hashes and hash tables;
- append-only database behavior and block crossings;
- point layout and node access;
- boundaries and active mirrors;
- mesh indexing, filtering, and views;
- search-tree backends;
- normal solving;
- edge enumeration;
- ray casting;
- complete Voronoi construction.
The tests intentionally use explicit fixtures and diagnostic output rather than hiding the tested behavior behind a large macro framework.
### 11.2 `verify_mesh()`
`verify_mesh()` checks every currently stored vertex occurrence in the context of its public cell.
For each occurrence it:
- activates the correct boundary mirrors;
- checks vertex variance;
- performs a nearest-neighbour query;
- verifies that a nearest generator belongs to the stored signature.
It is a **consistency test, not a completeness proof**. A mesh with missing vertices may pass if every stored vertex is individually valid.
### 11.3 `compare_meshes()`
`compare_meshes()` compares compatible meshes in both directions using:
- ordinary node identity;
- primary vertex signatures;
- vertex positions with a variance-derived tolerance.
This provides a useful independent comparison path for future optimized or parallel implementations.
### 11.4 Current end-to-end regression fixtures
Two 4D fixtures exercise complementary cases.
#### General position
The deterministic general-position fixture currently produces:

```text
stored primary vertices:      2156
checked vertex occurrences:   9662
verification errors:             0
```

#### Cubic degeneracy
The Cartesian fixture uses four generator positions per axis:

```text
{-3, -1, 1, 3}^4
```

Hence it contains `4^4 = 256` generators, three cell intervals per axis, and exactly

```text
5^4 = 625
```

bounded-grid Voronoi vertices in the box `[-4, 4]^4`.
The current regression result is:

```text
stored primary vertices:       625
expected primary vertices:     625
checked vertex occurrences:   4096
verification errors:             0
```

This fixture is especially important for the minimal-edge/full-support distinction in degenerate edge enumeration.
---

## 12. Current limitations and roadmap
The current architecture intentionally exposes extension boundaries before all planned algorithm variants are implemented.
### Current limitations
- only the one-mesh branch of `ComputeVoronoi` is executable;
- parallel mesh-view construction and cross-branch execution are not implemented;
- infinite rays are detected by ray casting but are not yet persisted by the mesh API;
- the fused search-tree `direct_cast` path is still a placeholder;
- periodic boundary primitives exist, but complete periodic mesh construction is not implemented;
- spherical Voronoi construction is not implemented;
- refinement/replacement workflows are not implemented;
- disk-backed database storage is not implemented;
- public API/umbrella-header organization is still subject to cleanup;
- volume, interface-measure, quadrature, and integration algorithms are outside the current construction core.
### Roadmap
Planned extensions include:
1. parallel mesh branches using the existing `ComputeVoronoi` registration boundary;
2. persistence of unbounded Voronoi edges;
3. optimized fused `direct_cast` backends;
4. local refinement and replacement;
5. periodic/computed/hybrid mesh construction;
6. spherical Voronoi construction;
7. additional nearest-neighbour backends where profiling justifies them;
8. optional disk-backed storage;
9. volume, interface, and quadrature algorithms as separate higher-level components.
Future implementations should reuse the current mesh, worker, registration, and validation contracts rather than introducing parallel algorithm copies.
---

## 13. Architectural invariants
Future changes should preserve the following invariants.
### Indexing and storage
- [ ] Public ordinary indices are dense and zero-based.
- [ ] Persistent ordinary internal indices do not change implicitly.
- [ ] `max(Index)` remains an invalid marker.
- [ ] Boundary internal indices remain disjoint from ordinary internal indices.
- [ ] Persistent vertex signatures are canonical internal signatures.
- [ ] A stored finite vertex contains at least one ordinary generator.
- [ ] Public reordering does not rewrite persistent signatures.
- [ ] Backend-native search indices never escape unchecked.
### Geometry
- [ ] Inactive mirror nodes never participate in search.
- [ ] Computed nodes are never assumed to provide stable pointers.
- [ ] Minimal-edge ownership is not inferred from the minimum of the full degenerate support.
- [ ] Numerical correction is followed by geometric validation.
- [ ] Optimized search/raycast implementations remain comparable to reference behavior.
### Construction
- [ ] Existing stored vertices enter only cell-local queue/edge state.
- [ ] Newly found vertices pass through `ComputeVoronoi` for persistence and communication.
- [ ] Exactly one queue claimant performs the `OnQueueEdges` pass for a signature.
- [ ] Cell-local edge occurrence state is reset only after the cell is completely processed.
- [ ] `SystematicVoronoi` remains independent of `MeshThreading`.
- [ ] `VoronoiWorker` remains independent of all global threading policies.
### Concurrency
- [ ] Synchronization is attached to shared structures rather than worker code.
- [ ] Single-thread configurations reduce shared cell structures to `EmptyLock` where applicable.
- [ ] Parallel construction uses a thread-safe persistent database.
- [ ] Worker-local scratch state is not shared.
- [ ] Structural mesh changes occur only at explicit safe points.
### Performance and ownership
- [ ] Hot loops can recycle buffers.
- [ ] Search backends do not allocate a point per coordinate lookup.
- [ ] Persistent records are stored once and referenced from ownership lists.
- [ ] Mesh wrappers do not silently rebase stable data.
- [ ] Volume and integration logic remains outside the Voronoi construction core.
---

## 14. Source map
| Area | Principal files |
|---|---|
| Public configuration | `include/highvoronoi/parameters.hpp` |
| Umbrella header | `include/highvoronoi/highvoronoi.hpp` |
| Locks and execution policies | `include/highvoronoi/detail/locks.hpp` |
| Precision helpers | `include/highvoronoi/detail/float.hpp` |
| Hash functions/generators | `include/highvoronoi/detail/hash_functions.hpp`, `hash_generators.hpp` |
| Queue/edge hashes | `queue_hash_table.hpp`, `queue_hash_table_2.hpp`, `edge_hash_table.hpp` |
| Hash containers/types | `hash_table_containers.hpp`, `hash_types.hpp` |
| Persistent database | `include/highvoronoi/detail/hvdatabase.hpp` |
| Address lists | `include/highvoronoi/detail/read_write_list.hpp` |
| Index views | `include/highvoronoi/detail/hvview.hpp` |
| Points | `include/highvoronoi/geometry/point.hpp` |
| Nodes | `include/highvoronoi/geometry/voronoi_nodes.hpp` |
| Boundaries | `include/highvoronoi/geometry/boundary.hpp` |
| Mesh base | `include/highvoronoi/geometry/abstract_mesh.hpp` |
| Stored mesh | `include/highvoronoi/geometry/voronoi_mesh.hpp` |
| Mesh views | `include/highvoronoi/geometry/mesh_view.hpp` |
| Composite mesh | `include/highvoronoi/geometry/serial_mesh.hpp` |
| Mesh validation | `include/highvoronoi/geometry/mesh_validation.hpp` |
| Search base | `include/highvoronoi/geometry/abstract_search_tree_crtp.hpp` |
| Brute-force search | `include/highvoronoi/geometry/brute_force_search_tree_crtp.hpp` |
| nanoflann search | `include/highvoronoi/geometry/nanoflann_search_tree_crtp.hpp` |
| Search factory | `include/highvoronoi/geometry/search_tree_factory_crtp.hpp` |
| Local nanoflann dependency | `include/highvoronoi/detail/nanoflann.hpp` |
| Normal solving | `include/highvoronoi/geometry/normal_solver.hpp` |
| Edge enumeration | `include/highvoronoi/geometry/edge_iterator.hpp` |
| Ray casting | `include/highvoronoi/geometry/raycaster.hpp` |
| Geometry worker | `include/highvoronoi/geometry/voronoi_worker.hpp` |
| Cell/branch construction | `include/highvoronoi/geometry/systematic_voronoi.hpp` |
| Top-level construction | `include/highvoronoi/geometry/compute_voronoi.hpp` |
| End-to-end validation tests | `tests/test_compute_voronoi_validation.cpp` |
---

## 15. Reference
Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025.
Relevant mathematical sections include:
- localized convex-cone edge enumeration: Section 2.5;
- data structures and exhaustive construction: Sections 3.1-3.3;
- RayCast and Descent: Section 3.4;
- parallelization: Section 3.6;
- refinement: Section 4;
- quasi-periodic generation: Section 5;
- spherical Voronoi diagrams: Section 6;
- numerical robustness: Section 7.
