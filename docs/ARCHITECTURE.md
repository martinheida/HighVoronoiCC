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
- append-only persistent finite-vertex storage;
- persistent unbounded-edge storage;
- configurable queue and edge hash tables and synchronization policies;
- brute-force and bundled nanoflann search backends;
- general and degenerate edge enumeration;
- ray casting, descent, vertex verification, and numerical correction;
- complete systematic cell exploration for one mesh branch;
- single-threaded or multi-worker ray casting inside each branch;
- parallel execution of several communicating reordered mesh branches;
- combined mesh-level and cast-level parallelism;
- geometric mesh validation;
- topological full-edge completeness validation including infinite edges;
- serial/parallel comparison utilities and regression tests.

All indices in this document are zero-based unless stated otherwise.

---

## 2. Core design principles

### 2.1 Stable storage is separate from public presentation

Public ordinary node indices are dense and may change after filtering, reordering, or composition. Persistent signatures use stable internal indices that remain valid for the lifetime of their owning storage.

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
- direct/static/dynamic hash-container mode;
- search-tree implementation;
- ray-cast implementation;
- mesh and cast threading policy.

Runtime configuration is used for quantities such as:

- thread count;
- hash capacities and block sizes;
- KD-tree leaf size and build thread count;
- database block size;
- geometric tolerances.

The user-facing file `howto_config.hpp` documents the currently intended configuration surface and its defaults as ordinary editable C++17 code.

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

The construction architecture does not put a global mutex around the algorithm and does not make `VoronoiWorker` aware of thread policies.

Synchronization is attached to the mutable structures that may actually be shared:

- the persistent database and address lists;
- the branch vertex queue;
- the branch edge hash;
- the branch FEI cache used by degenerate edge enumeration;
- the branch prototype `EdgeIterator` scratch state;
- a shared mesh vertex iterator while existing vertices are distributed among cast workers.

Worker-local geometric state remains lock-free.

### 2.8 Worker geometry state is branch-local

A worker-owned `EdgeIterator` belongs to exactly one `SystematicVoronoi` branch because it is tied to that branch's `ExtendedNodes`, public numbering, and FEI state.

It must never be sent through `ComputeVoronoi` into another branch.

Cross-branch vertex communication transfers only geometric vertex data. The target branch performs its own queue-edge processing using its own prototype iterator and its own local numbering.

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

The execution topology is:

```text
ComputeVoronoi
    |
    +-- SystematicVoronoi [branch 0]
    |       |
    |       +-- VoronoiWorker 0
    |       +-- VoronoiWorker 1
    |       +-- ...
    |
    +-- SystematicVoronoi [branch 1]
    |       |
    |       +-- VoronoiWorker 0
    |       +-- ...
    |
    +-- ...
```

Parallel branches are non-owning `ReorderedMeshView`s of the same persistent mesh/database.

The dependency direction is deliberate:

- hashing knows nothing about geometry;
- the database knows nothing about ray casting;
- search trees know nothing about construction queues;
- `VoronoiWorker` knows nothing about global branch scheduling;
- `SystematicVoronoi` knows only cast-level worker execution plus the shared lock type supplied by its parent;
- only `ComputeVoronoi` knows the complete branch topology and cross-branch communication.

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

### 4.5 Finite vertex representation and ownership

A finite Voronoi vertex is represented by:

```text
sigma : sorted generating-node signature
r     : vertex position
```

Persistent signatures are canonicalized in stable internal numbering.

The smallest ordinary internal generator is the primary storage owner. Other ordinary generators receive secondary address registrations. The vertex record itself is stored only once.

Storage ownership and algorithmic edge ownership are separate concepts.

### 4.6 Infinite edge representation

A persistent unbounded Voronoi edge is represented by:

```text
sigma     : complete supporting-edge signature
origin    : finite endpoint position
direction : unbounded ray direction
```

The complete supporting edge, not the minimal algorithmic edge, is converted to stable internal numbering and used as the persistent duplicate key.

This distinction is essential for degenerate vertices: several minimal edges may describe the same geometric unbounded ray.

---

## 5. Persistent storage and hash structures

### 5.1 `HVDataBase`

`HVDataBase<Lock, DataBaseParams>` is the current persistent append-oriented database.

Important properties:

- data is stored in fixed-size blocks;
- records may cross block boundaries;
- an atomic top counter reserves disjoint write ranges;
- structural block growth is protected by the configured lock;
- duplicate detection uses the configured queue-hash family;
- address `0` denotes duplicate insertion;
- finite vertices are written/read through the vertex-record API;
- infinite edges reuse the facet record path and store signature, origin, and direction;
- deletion removes the signature from the hash and sets `sigma_length = 0` as a tombstone;
- record storage is not compacted immediately.

Stable addresses allow address lists to remain valid after logical deletion.

### 5.2 Hashing roles

Hashing is deliberately modular because different lifetimes have different requirements.

Current roles include:

- persistent duplicate detection in `HVDataBase`;
- cell-local first-claim duplicate detection in the vertex queue;
- cell-local edge occurrence counting;
- global full-edge occurrence counting in completeness validation.

The code provides multiple hash functions, configurable hash generators, queue hash tables, edge hash tables, and direct/static/dynamic container modes.

These states must remain separate even when they use the same implementation family. Persistent and cell-local state have different ownership and reset semantics.

### 5.3 Hash-container sharding

`DirectHash` creates one underlying table.

`StaticHash<N>` creates `N` independent tables and routes a key to one subtable. Each subtable has its own lock. This reduces both collision concentration and lock contention under parallel access.

`DynamicHash<>` creates tables according to ranges of the first key value and can grow the number of subtables as required.

The same container-mode abstraction is used for queue hashes and edge hashes through the corresponding `StaticQueueHashContainer`, `DynamicQueueHashContainer`, `StaticEdgeHashContainer`, and `DynamicEdgeHashContainer` implementations.

### 5.4 Exact edge occurrence state

`EdgeHashTable` stores first and second occurrences of a geometric edge.

The distinguished values are:

```text
no_second_cell = -1
infinite_cell  = -2
```

The code uses explicit equality with `no_second_cell`; negative values are not generically interpreted as empty.

A third occurrence sets an `overfull` state. Therefore `all_edges_complete()` means **exactly two occurrences per stored edge**, not merely at least two.

### 5.5 Address lists

Primary and secondary finite-vertex addresses are attached to ordinary nodes through thread-policy-aware read/write lists.

Persistent infinite edges use a mesh-level address list because they are not naturally owned by one ordinary node list.

The database owns the records. Address lists own only references to those records.

---

## 6. Mesh architecture

### 6.1 `AbstractMesh`

`AbstractMesh` defines the common mesh behavior used by algorithms and wrappers:

- public/internal signature conversion;
- boundary index conversion;
- duplicate-safe finite-vertex storage;
- duplicate-safe infinite-edge storage;
- primary and secondary finite-vertex iteration;
- combined cell-vertex iteration;
- infinite-edge iteration;
- node and vertex filtering;
- dependent infinite-edge tombstoning when nodes are removed;
- tombstone-aware record handling.

The public iterator interface deliberately hides the physical database representation.

### 6.2 `VoronoiMesh`

`VoronoiMesh` is the main concrete stored mesh.

It owns:

- stable ordinary node storage;
- dense public/internal mappings;
- extended boundary nodes;
- the persistent database;
- primary and secondary finite-vertex address lists;
- the global infinite-edge address list.

Public node deletion changes the public mapping without renumbering stable internal nodes.

### 6.3 `ReorderedMeshView`

`ReorderedMeshView` is a non-owning facade that changes public order while retaining:

- stable node storage;
- persistent internal signatures;
- database records;
- finite-vertex address lists;
- infinite-edge records and their underlying address list;
- the underlying boundary.

It owns no independent persistent infinite-edge list. Registration is forwarded to the wrapped mesh while iteration converts signatures back into the view's public numbering.

Its main architectural purpose is to let algorithms operate on a selected public ordering without rebasing persistent data.

### 6.4 `SwitchView` and parallel branch numbering

Parallel construction uses `SwitchView` inside `ReorderedMeshView`.

`ComputeVoronoi` partitions the selected cell interval into disjoint contiguous blocks. For each block, a `SwitchView` moves those cells to the beginning of that branch's public numbering so that the branch can process a local range starting at zero.

The persistent internal numbering remains unchanged.

### 6.5 `SerialMesh`

`SerialMesh` composes compatible child meshes under one top-level numbering and shared storage semantics.

Unlike a pure view, it is treated as a geometrically independent composite mesh and therefore owns its own top-level infinite-edge address list.

The current class establishes the abstraction required for later workflows such as:

- local refinement and replacement;
- periodic copies;
- computed or translated submeshes;
- mixed stored/computed compositions.

The detailed refinement/periodic semantics are the next architecture step and are not yet considered stabilized.

### 6.6 Vertex iterator recycling

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

HighVoronoiCC vendors its local `nanoflann.hpp`; the architecture assumes that exact bundled interface rather than an arbitrary system version.

### 7.4 `safe_copy()` versus `rebind()`

These operations have different ownership semantics.

`safe_copy()` creates an independent mutable search/raycast object **inside the same mesh branch** while retaining the same immutable/backend search core where possible. For nanoflann this means worker ray casters can share the branch KD-tree while owning independent mutable search and boundary state.

`rebind(new_mesh)` moves the configuration into a **different mesh/view numbering world** and constructs the backend state appropriate for that new branch.

Therefore the intended parallel structure is:

```text
branch 0 -> one KDTree shared by prototype + workers of branch 0
branch 1 -> one different KDTree shared by prototype + workers of branch 1
...
```

A worker does not rebuild its own KD-tree.

### 7.5 `RayCaster`

`RayCaster` owns the geometric nearest-neighbour/range-search logic required to advance a Voronoi ray.

Its responsibilities include:

- cell-local activation of boundary mirrors;
- finite/infinite ray status;
- candidate signature construction;
- vertex verification;
- numerical correction of candidate vertices.

Ray-cast strategy is selected independently from the search backend, currently including `ClassicRaycast` and `InRangeRaycast` policies.

### 7.6 `direct_cast`

The search-tree abstraction contains a future `direct_cast` hook for a fused `RCCombined`-style branch-and-bound traversal. Its current backend contract is still a placeholder.

Correctness therefore does not depend on `direct_cast`; the implemented construction uses the existing search operations through `RayCaster`.

### 7.7 `EdgeIterator`

`EdgeIterator` enumerates candidate Voronoi edges at a vertex.

It supports:

- the general-position case;
- the degenerate case with more than `dimension + 1` generators;
- reusable fast-edge storage;
- an `OnQueueEdges` mode used when a vertex first enters the cell queue;
- an `OnSysVoronoi` mode used while systematically exploring that vertex.

The expensive degenerate iterator state is cached and shared where safe.

### 7.8 Minimal edge versus full supporting edge

Degenerate edge enumeration distinguishes two signatures:

- the **minimal edge**, which determines algorithmic ownership;
- the **full supporting edge**, which identifies the complete geometric support.

A full supporting edge may legitimately contain generators with global indices smaller than the active cell. These generators must not be globally filtered out.

The minimal edge is used for the systematic cell-ownership rule. The full supporting edge is used for persistent infinite-edge identity and global completeness validation.

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

`ComputeVoronoi` therefore knows both threading axes. It selects the shared branch lock from `VoronoiThreading` and creates the branch-local `SystematicVoronoi` objects.

### 8.2 Branch construction

With `MeshThreading = SingleThread`, branch 0 operates directly on the supplied mesh.

With `MeshThreading = MultiThread(n)`, `ComputeVoronoi`:

1. limits the branch count to the number of selected cells;
2. partitions the selected cell interval into contiguous blocks;
3. creates one `ReorderedMeshView<Mesh, SwitchView>` for each block;
4. calls `raycaster_prototype.rebind(branch_mesh)` once per branch;
5. constructs one `SystematicVoronoi` per branch;
6. runs the branches in separate `std::thread`s;
7. joins all branches and rethrows the first captured exception.

### 8.3 Local-first vertex registration

A vertex found by a worker is processed in this order:

```text
worker discovers vertex V
        |
        v
source SystematicVoronoi::register_found_vertex(V, worker queue iterator)
        |
        +--> source queue_vertex(V)
        |       local claim
        |       local OnQueueEdges
        |       source EdgeHash / FEI state
        |
        v
ComputeVoronoi::register_vertex(V, source branch)
        |
        +--> persistent mesh/database insertion
        |
        +--> translate V into each target branch's public numbering
                |
                v
             target.queue_vertex(V)
                |
                +--> target prototype EdgeIterator
                +--> target EdgeHash / FEI state
```

The local queue operation is deliberately independent of whether persistent insertion wins a global duplicate race. A branch may need the vertex for its own current cell even if another branch stored the persistent record first.

### 8.4 Cross-branch communication boundary

Worker `EdgeIterator` objects never leave their source `SystematicVoronoi`.

`ComputeVoronoi` communicates only:

- the vertex signature;
- the vertex position;
- the source branch id.

The source public signature is converted through stable internal/wrapped numbering and then into the target branch's public numbering.

The target branch performs its own queue-edge pass using its own prototype iterator.

### 8.5 `SystematicVoronoi`

`SystematicVoronoi` performs the complete cell loop for exactly one mesh branch.

It knows `CastThreading`, but it does **not** know how many mesh branches exist.

It owns:

- the branch id and local cell range;
- the atomic current cell;
- the cell-local vertex queue;
- the cell-local edge hash;
- a prototype `RayCaster`;
- a prototype `EdgeIterator`;
- the configured `VoronoiWorker` objects;
- active mirror-index scratch storage;
- a lock that protects only mutable prototype-iterator scratch.

### 8.6 Cell lifecycle

For each local cell, `SystematicVoronoi::compute()` performs:

```text
1. reset cell-local queue, edge hash, and boundary state
2. iterate already stored vertices of the cell
3. queue those existing vertices
4. if no starting vertex exists, perform descent
5. locally queue and persist/communicate the descent vertex
6. systematically explore queued vertices until the current worker runs out of work
7. advance to the next cell
```

Existing mesh vertices are already persistent. They are inserted only into the current cell's queue/edge state.

New vertices found by descent or ray traversal pass through the local-first registration path above.

### 8.7 Cell reset

`reset(cell)` clears cell-local queue and edge state, activates the correct boundary mirrors on the prototype and all worker ray casters, and only then publishes the new atomic current-cell value.

The order is deliberate: queue/hash/boundary state belongs to the previously published cell until reset is complete.

### 8.8 Queue semantics

The branch queue separates first-claim from actual enqueue:

```text
claim(signature)
    -> queue hash insertion under queue lock
    -> exactly one claimant succeeds in one cell generation

OnQueueEdges
    -> performed by that claimant without keeping the queue lock

enqueue_claimed(vertex)
    -> append work item under queue lock if incomplete edges remain
```

No additional queue-generation lock is required for this duplicate-claim contract.

### 8.9 Prototype queueing for remote vertices

A target branch may receive a vertex concurrently from another branch.

Because no worker iterator crosses branches, the target uses `prototype_edge_iterator_` for the queue-edge pass. `prototype_lock_` protects only the prototype iterator's mutable scratch.

The target's EdgeHash and FEI cache remain the same shared branch state used by local workers and are protected independently by the combined shared lock.

### 8.10 Systematic vertex exploration

For a queued vertex, a worker:

1. resets its searching `EdgeIterator` in `OnSysVoronoi` mode;
2. enumerates locally relevant minimal edges;
3. updates the shared cell edge hash using the minimal edge occurrence key;
4. skips edges already complete or not owned by the active cell;
5. follows incomplete edges with `walk_ray()`;
6. persists infinite rays by the full supporting edge;
7. verifies or corrects finite candidates as required;
8. sends finite candidates through the local-first registration path.

### 8.11 Infinite-ray registration

When `walk_ray()` returns `RayCastStatus::Infinite`, the worker calls its parent branch directly:

```text
VoronoiWorker
    -> SystematicVoronoi::register_infinite_edge(...)
    -> branch mesh.store_infinite_edge(...)
    -> shared database duplicate detection
    -> mesh-level infinite-edge address registration
```

No `ComputeVoronoi` broadcast is required. All parallel branch views refer to the same persistent storage, and the database key uses stable internal numbering.

### 8.12 `VoronoiWorker`

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

The two worker edge iterators share the branch FEI cache. Separate iterators are required because queue processing may trigger edge enumeration while systematic edge traversal is already active.

### 8.13 Descent

If a cell has no known stored vertex, worker 0 performs descent from the cell generator.

The current implementation repeatedly finds directions orthogonal to the accumulated minimal support, ray casts in those directions, handles opposite-direction retry for infinite casts, projects the final point inside the boundary, performs correction when appropriate, and verifies the resulting vertex before registration.

Descent is a geometric initialization operation, not a separate persistence path.

---

## 9. Threading and synchronization

### 9.1 Two independent threading axes

HighVoronoiCC separates:

- **MeshThreading**: parallel mesh branches;
- **CastThreading**: parallel geometry workers within one branch.

This separation is intentional. Branch communication and local ray-cast parallelism have different shared state and different performance characteristics.

### 9.2 Combined `VoronoiThreading`

`ComputeVoronoi` defines:

```text
VoronoiThreading = MultiThread
    if MeshThreading is MultiThread
    or CastThreading is MultiThread
```

Otherwise it is `SingleThread`.

This combined policy is required for state that may be touched either by local workers or by remote branch communication.

### 9.3 Lock ownership

The current lock ownership can be summarized as follows:

| Shared structure | Lock selection | Reason |
|---|---|---|
| persistent mesh/database | mesh database configuration | finite/infinite records may be stored concurrently |
| mesh address lists | database/thread policy | persistent registrations may be concurrent |
| branch vertex queue | `VoronoiThreading::RWLock` | local workers and remote branches may register vertices |
| branch EdgeHash | `VoronoiThreading::RWLock` | remote target queueing performs `OnQueueEdges` even with one local worker |
| branch FEI cache | `VoronoiThreading::RWLock` | prototype and workers share degenerate-edge cache state |
| prototype `EdgeIterator` scratch | `VoronoiThreading::RWLock` | only remote/prototype queue passes share this mutable iterator |
| shared existing-vertex iterator | `CastThreading::RWLock` | distributed only among workers inside one branch |
| worker scratch data | none | owned by one worker |

Any parallel construction path requires the persistent mesh/database to use `ReadWriteLock` because finite and infinite records may be stored concurrently.

### 9.4 Four configurations

The two threading axes produce four implemented cases:

| MeshThreading | CastThreading | Queue/Edge/FEI | Existing-vertex iterator | Status |
|---|---|---|---|---|
| Single | Single | `EmptyLock` | `EmptyLock` | implemented |
| Single | Multi | `ReadWriteLock` | `ReadWriteLock` | implemented |
| Multi | Single | `ReadWriteLock` | `EmptyLock` | implemented |
| Multi | Multi | `ReadWriteLock` | `ReadWriteLock` | implemented |

### 9.5 Hash sharding and contention

The lock type and hash-container mode are independent choices.

For example, a `StaticEdgeHashContainer<N>` contains `N` separately locked edge tables. Parallel threads contend only if they access the same subtable at the same time.

The same principle applies to static queue-hash containers.

This is expected to be important for scaling and remains a benchmarking/configuration topic rather than a construction-algorithm requirement.

### 9.6 Worker execution

`SystematicVoronoi::run_workers()` is the only place where branch-local worker threads are created.

With `CastThreading = SingleThread`, worker 0 is called directly. With `CastThreading = MultiThread`, all configured workers execute the requested phase and are joined before the next cell phase begins.

### 9.7 Shared mesh iteration

`AbstractMesh` iterators have mutable traversal state and are not themselves shared concurrently.

`ConcurrentVertexIterator` wraps one iterator and serializes only `next()`:

```text
lock
    copy current mesh vertex into worker-owned storage
    advance iterator
unlock
```

The underlying mesh iterator therefore remains simple and unchanged. Concurrency is introduced only where the algorithm actually shares it.

### 9.8 Current cast-worker scheduling limitation

The current shared work queue has no explicit in-flight-work termination protocol.

A worker may observe the queue as temporarily empty while another worker is still computing a ray that will later generate new queue entries. The observing worker then stops its current exploration phase.

This is not currently known to corrupt the result, but it can reduce cast-level load balancing and likely contributes to weaker scaling of `CastThreading` compared with mesh-branch parallelism.

This scheduling issue is intentionally separated from the correctness/ownership architecture and is a later optimization step.

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
- complete Voronoi construction;
- parallel construction;
- persistent infinite edges;
- exact edge-completeness state.

The tests intentionally use explicit fixtures and diagnostic output rather than hiding the tested behavior behind a large macro framework.

### 11.2 `verify_mesh()`

`verify_mesh()` checks every currently stored finite-vertex occurrence in the context of its public cell.

For each occurrence it:

- activates the correct boundary mirrors;
- checks vertex variance;
- performs a nearest-neighbour query;
- verifies that a nearest generator belongs to the stored signature.

It is a **consistency test, not a completeness proof**. A mesh with missing vertices may pass if every stored vertex is individually valid.

### 11.3 `verify_mesh_complete()`

`verify_mesh_complete()` first runs `verify_mesh()` and then reconstructs global geometric edge incidences.

The finite endpoint contribution is keyed by `EdgeIterator::full_indices()`.

At a degenerate persistent vertex, repeated minimal-edge representations of the same full supporting edge are collapsed per `(vertex address, full edge)` before global counting.

Then:

- every finite endpoint contributes one occurrence;
- every persistent infinite edge contributes one occurrence using `infinite_cell`;
- `EdgeHashTable::all_edges_complete()` requires exactly two total occurrences per edge.

This is a topological closure test under the assumptions of the construction representation. It cannot detect an entirely absent edge together with all of its incidence data.

### 11.4 `compare_meshes()`

`compare_meshes()` compares compatible meshes in both directions using:

- ordinary node identity;
- primary vertex signatures;
- vertex positions with a variance-derived tolerance.

This provides an independent comparison path for serial, parallel, reference, and future optimized implementations.

### 11.5 Current end-to-end regression fixtures

#### Four-dimensional general position

The deterministic general-position fixture exercises the ordinary high-dimensional construction path and geometric verification.

#### Four-dimensional Cartesian degeneracy

The Cartesian fixture uses four generator positions per axis:

```text
{-3, -1, 1, 3}^4
```

Hence it contains `4^4 = 256` generators and exactly

```text
5^4 = 625
```

bounded-grid Voronoi vertices in the box `[-4, 4]^4`.

This fixture is especially important for the minimal-edge/full-support distinction in degenerate edge enumeration.

#### Three-dimensional parallel smoke fixture

A deterministic 200-generator 3D fixture compares serial construction with mesh-level and cast-level parallel construction.

The validated runs produce the same persistent vertex count and compare equal within the variance-derived geometric tolerance. The fixture is intended to catch branch-index translation, shared-state, and synchronization regressions.

#### Unbounded two-dimensional completeness fixture

Three generators form one finite Voronoi vertex with three unbounded rays.

The regression verifies:

```text
finite vertex endpoints: 3
persistent infinite rays: 3
edge closure:             complete
```

After removing one persistent infinite-edge record, `verify_mesh_complete()` must report the mesh as incomplete.

### 11.6 Concurrency diagnostics

Parallel construction should additionally be exercised under ThreadSanitizer when shared-state or ownership rules change.

The current shared-state branch/worker architecture has been exercised with GCC ThreadSanitizer without a reported race in the parallel smoke path.

---

## 12. Current limitations and roadmap

The current architecture intentionally exposes extension boundaries before all planned algorithm variants are implemented.

### Current limitations

- `SerialMesh` refinement/composition semantics are not yet stabilized for the planned refinement and periodic workflows;
- complete periodic Voronoi construction is not implemented;
- spherical Voronoi construction is not implemented;
- local refinement/replacement workflows are not implemented;
- cast-worker scheduling does not yet keep idle workers alive while other workers have in-flight ray computations;
- the fused search-tree `direct_cast` path is still a placeholder;
- insertion/removal workflows beyond the current filtering primitives are not implemented;
- disk-backed database storage is not implemented;
- public API/umbrella-header organization is still subject to cleanup;
- volume, interface-measure, quadrature, and integration algorithms are outside the current construction core.

### Near-term roadmap

1. stabilize `SerialMesh` ownership and indexing semantics;
2. implement local refinement/replacement on top of `SerialMesh`;
3. implement periodic/computed/hybrid mesh composition;
4. improve cast-worker termination/load balancing;
5. benchmark hash-container sharding and both parallelization axes;
6. stabilize the public configuration and construction API;
7. extend completeness and concurrency regression coverage.

### Longer-term roadmap

1. optimized fused `direct_cast` backends;
2. spherical Voronoi construction;
3. quasi-periodic generation;
4. insertion and removal of generators;
5. additional nearest-neighbour backends where profiling justifies them;
6. optional disk-backed storage;
7. volume, interface, quadrature, and integral algorithms as higher-level components.

Future implementations should reuse the current mesh, worker, registration, and validation contracts rather than introducing parallel algorithm copies.

---

## 13. Architectural invariants

Future changes should preserve the following invariants.

### Indexing and storage

- [ ] Public ordinary indices are dense and zero-based.
- [ ] Persistent ordinary internal indices do not change implicitly.
- [ ] `max(Index)` remains an invalid marker.
- [ ] Boundary internal indices remain disjoint from ordinary internal indices.
- [ ] Persistent finite-vertex signatures are canonical internal signatures.
- [ ] Persistent infinite-edge signatures are canonical full supporting-edge signatures.
- [ ] A stored finite vertex contains at least one ordinary generator.
- [ ] A stored infinite edge contains at least one ordinary generator.
- [ ] Public reordering does not rewrite persistent signatures.
- [ ] Backend-native search indices never escape unchecked.

### Geometry

- [ ] Inactive mirror nodes never participate in search.
- [ ] Computed nodes are never assumed to provide stable pointers.
- [ ] Minimal-edge ownership is not inferred from the minimum of the full degenerate support.
- [ ] Infinite-edge identity uses the full supporting edge, not the minimal edge.
- [ ] Numerical correction is followed by geometric validation.
- [ ] Optimized search/raycast implementations remain comparable to reference behavior.

### Construction

- [ ] Existing stored vertices enter only cell-local queue/edge state.
- [ ] Newly found finite vertices are queued locally before global persistence/communication.
- [ ] Persistent duplicate loss does not suppress required local queue state.
- [ ] Worker `EdgeIterator`s never cross a `SystematicVoronoi` branch boundary.
- [ ] Remote branches queue communicated vertices using their own prototype iterator.
- [ ] Exactly one queue claimant performs the `OnQueueEdges` pass for a signature in one cell generation.
- [ ] Infinite rays are persisted directly through the branch mesh using full-edge identity.
- [ ] Cell-local edge occurrence state is reset only after the cell has finished its exploration phase.
- [ ] `SystematicVoronoi` remains independent of `MeshThreading` topology.
- [ ] `VoronoiWorker` remains independent of all global threading policies.

### Concurrency

- [ ] Synchronization is attached to shared structures rather than worker code.
- [ ] `VoronoiThreading` is multi-threaded whenever either execution axis is multi-threaded.
- [ ] Branch queue, EdgeHash, FEI cache, and prototype iterator use the combined shared lock where required.
- [ ] Single-thread configurations reduce shared structures to `EmptyLock` where applicable.
- [ ] Parallel construction uses a thread-safe persistent database and address lists.
- [ ] Worker-local scratch state is not shared.
- [ ] Structural mesh changes occur only at explicit safe points.

### Completeness

- [ ] Global geometric edge identity uses the complete supporting edge.
- [ ] Degenerate repeated minimal-edge representations are deduplicated per persistent endpoint.
- [ ] A complete finite edge has exactly two finite endpoint occurrences.
- [ ] A complete unbounded edge has one finite endpoint and one persistent infinite-edge occurrence.
- [ ] A third edge occurrence is considered invalid.

### Performance and ownership

- [ ] Hot loops can recycle buffers.
- [ ] Search backends do not allocate a point per coordinate lookup.
- [ ] Worker copies inside one branch share immutable/search backend core state where safe.
- [ ] `rebind()` creates backend state appropriate for a different mesh/view numbering world.
- [ ] Persistent records are stored once and referenced from address lists.
- [ ] Pure mesh views do not silently create independent persistent ownership.
- [ ] Volume and integration logic remains outside the Voronoi construction core.

---

## 14. Source map

| Area | Principal files |
|---|---|
| User configuration guide | `howto_config.hpp` |
| Public configuration types | `include/highvoronoi/parameters.hpp` |
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
| Mesh validation/completeness | `include/highvoronoi/geometry/mesh_validation.hpp` |
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
| Parallel smoke test | `tests/test_compute_voronoi_parallel_smoke.cpp` |
| Infinite-edge storage test | `tests/test_infinite_edge_storage.cpp` |
| Edge completeness unit test | `tests/test_edge_hash_completeness.cpp` |
| Unbounded completeness test | `tests/test_compute_voronoi_unbounded_completeness.cpp` |

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
