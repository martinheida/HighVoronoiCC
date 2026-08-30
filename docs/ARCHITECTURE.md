# HighVoronoiCC Architecture

**Status:** current implementation architecture and extension roadmap  
**Scope:** public modules, geometry, storage, indexing, search, ordinary and incremental Voronoi construction, periodic HighVoronoi, synchronization, numerical robustness, validation, and planned extensions  
**Reference algorithm:** Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025

---

## 1. Purpose and current status

HighVoronoiCC is a C++17 implementation and redesign of the local high-dimensional Voronoi construction algorithm developed in the HighVoronoi project.

This document describes the architecture of the **current C++ implementation**. It is intentionally not a second algorithm paper and not a frozen API specification. Mathematical derivations belong to the reference paper; this document focuses on software responsibilities, ownership, data flow, indexing, synchronization, implementation-specific correctness conditions, and the relationship between the ordinary and HighVoronoi construction layers.

The current implementation provides:

- compact configurable scalar and index types;
- stable internal node numbering separated from public numbering;
- stored, computed, and hybrid node access;
- planar Dirichlet, Neumann and paired periodic boundaries;
- append-oriented persistent finite-vertex storage;
- persistent unbounded-edge storage;
- configurable queue and edge hash tables and synchronization policies;
- brute-force and bundled nanoflann search backends;
- general and degenerate edge enumeration;
- ray casting, descent, semantic vertex verification, and condition-aware numerical correction;
- complete systematic cell exploration for one mesh branch;
- single-threaded or multi-worker ray casting inside each branch;
- parallel execution of several communicating reordered mesh branches;
- combined mesh-level and cast-level parallelism;
- ordinary incremental insertion through `RefineVoronoi`;
- ordinary incremental removal through `RemoveVoronoi`;
- `HighVoronoiMesh` with visible nodes and invisible periodic reference nodes;
- batched incremental HighVoronoi orchestration through `ComputeHighVoronoi`;
- iterative periodic reference-node closure;
- final sparse repair of finite vertices hidden by the bootstrap periodic boundary;
- public projection of periodic HighVoronoi data back into the external visible domain;
- geometric mesh validation;
- topological full-edge completeness validation including infinite edges;
- selected-cell completeness validation;
- serial/parallel/reference comparison utilities and regression tests.

All indices in this document are zero-based unless stated otherwise.

---

## 2. Public module structure

The public include surface is intentionally split by construction role.

```text
highvoronoi/version.hpp
highvoronoi/database.hpp
highvoronoi/core.hpp
        |
        +-- highvoronoi/voronoi.hpp
        |       ordinary Euclidean construction
        |       RefineVoronoi / RemoveVoronoi
        |
        +-- highvoronoi/high_voronoi.hpp
        |       HighVoronoiMesh
        |       ComputeHighVoronoi
        |       VisibleFirstMesh
        |
        +-- highvoronoi/mesh_engines.hpp
                optional computed/hybrid engines

highvoronoi/highvoronoi.hpp
        complete convenience umbrella
```

`voronoi.hpp` is the narrow entry point for ordinary Euclidean construction. `high_voronoi.hpp` is the narrow entry point for persistent incremental/periodic HighVoronoi. `mesh_engines.hpp` keeps optional computed mesh-engine infrastructure out of translation units that do not need it.

The HighVoronoi implementation still reuses common incremental backend headers that contain the ordinary `VoronoiMesh` specialization, so its transitive include graph is not yet perfectly isolated. This is a compile-time organization issue, not an ownership or algorithmic coupling requirement.

---

## 3. Core design principles

### 3.1 One geometry kernel, several orchestration layers

The project deliberately avoids separate implementations of the local Voronoi geometry.

The core geometry remains:

```text
SearchTree
    |
RayCaster + EdgeIterator
    |
VoronoiWorker
    |
SystematicVoronoi
    |
ComputeVoronoi
```

Ordinary refinement, removal, HighVoronoi periodization, and sparse periodic repair all construct temporary mesh/database facades and reuse this same `ComputeVoronoi` kernel.

This is a primary architectural invariant: correctness fixes in ray casting, descent, edge enumeration, numerical correction, or systematic exploration must propagate automatically to all higher-level workflows.

### 3.2 Stable storage is separate from public presentation

Persistent signatures use stable internal indices. Public numbering may be dense, reordered, filtered, projected, or temporarily recomputed.

Consequences:

- deleting a public node does not renumber persistent records;
- `ReorderedMeshView` can change the current public order without rebasing storage;
- HighVoronoi invisible periodic references retain permanent internal identities;
- HighVoronoi public output may deliberately project several internal nodes onto one visible public node;
- `HighVoronoiComputeMesh` must therefore use its own dense **bijective** computation mapping instead of the non-bijective visible projection;
- backend-native search numbering is another independent index world.

### 3.3 Structural mutation occurs only at explicit safe points

Persistent record insertion can be parallel where the database/mesh policy permits it. Structural changes to node sets, public mappings, internal boundaries, reference-node lists, and atomic-bit-vector sizes are performed between delegated `ComputeVoronoi` phases.

Incremental orchestration is therefore a sequence of geometry phases separated by serialized structural transitions.

### 3.4 Hot paths reuse memory

Repeated search, edge construction, ray casting, signature conversion, QR work, and database access should recycle caller-owned or object-owned buffers.

This applies to:

- search backend data;
- vertex queues;
- queue and edge hashes;
- edge-iterator storage;
- FEI cache state;
- ray-cast signatures;
- normal-solver matrices and vectors;
- boundary candidate buffers;
- public/internal signature conversion buffers.

Convenience APIs may allocate. The construction core should not depend on repeated temporary allocation.

### 3.5 Synchronization belongs to shared data, not to workers

`VoronoiWorker` contains no global threading policy and owns no locks.

Synchronization is attached to shared mutable state:

- persistent database and address lists;
- branch vertex queue;
- branch EdgeHash;
- branch FEI cache;
- branch prototype `EdgeIterator` scratch;
- shared existing-vertex iterator used by cast workers;
- HighVoronoi round-level event flags.

Worker-local geometry remains lock-free.

### 3.6 Reference implementations and independent checks remain available

The brute-force search tree remains the correctness reference for optimized search backends.

At larger scale, regression tests compare:

- full computation against range-limited computation;
- serial against parallel computation;
- incremental against batched computation;
- periodic HighVoronoi against independently constructed explicit periodic tilings.

---

## 4. Data, geometry, and indexing model

### 4.1 Points

The point layer uses Eigen-compatible fixed or dynamic types:

- `StaticPoint<Scalar, Dim>`;
- `DynamicPoint<Scalar>`;
- static and dynamic read-only point views;
- `highvoronoi::Dynamic` as the runtime-dimension marker.

Stored points may be exposed as zero-copy views. Computed points own their returned values.

### 4.2 Node access modes

The node abstraction supports three compile-time access modes.

**Stored**
: every node has stable contiguous storage.

**Computed**
: coordinates are generated from an index when requested.

**Hybrid**
: some nodes are stored and others computed.

Algorithms independent of access mode use the common copy/data interface instead of assuming stable pointers.

### 4.3 Boundary mirrors

A planar boundary is represented by an oriented half-space. During exploration of one cell, active boundary planes are represented by virtual mirror generators of the active cell generator.

This converts the boundary plane into an ordinary Voronoi bisector.

The boundary layer supports Dirichlet, Neumann, and paired periodic planes. Periodic partners define a translation between corresponding faces; the actual periodic-copy construction belongs to `HighVoronoiMesh` / `ComputeHighVoronoi`, not to `Plane::reflect()`.

### 4.4 Ordinary public, internal, and boundary indices

For an ordinary mesh:

```text
public ordinary indices
    dense, current presentation

stable internal ordinary indices
    persistent identity slots

boundary public indices
    public_node_count + plane_index

boundary internal indices
    max(Index) - 1 - plane_index
```

`max(Index)` remains the invalid marker.

Persistent finite and infinite records use canonical stable internal signatures.

### 4.5 Backend-native indices

A search backend may use its own native index type. Conversion into `Mesh::Index` is explicit and checked. Backend-native numbering must never leak into mesh signatures.

---

## 5. Persistent storage and hash structures

### 5.1 `HVDataBase`

`HVDataBase<Lock, DataBaseParams>` is an append-oriented block database.

Finite records store:

```text
sigma length
sigma
position
```

Facet/infinite-edge records store:

```text
sigma length
sigma
origin
direction
```

Important properties:

- storage is split into fixed-size blocks of 16-bit units;
- an atomic top counter reserves disjoint write regions;
- outer block-vector growth uses the configured structural lock;
- persistent duplicate detection uses the configured QueueHash;
- public address `0` denotes duplicate/no new record;
- deletion removes the signature from the hash and tombstones the stored record by setting the stored signature length to zero;
- record storage is not compacted immediately.

### 5.2 Persistent finite-vertex ownership

A finite vertex is represented by:

```text
sigma : canonical sorted generator signature
r     : vertex position
```

The smallest ordinary stable internal generator owns the primary database address. The other ordinary generators receive secondary address registrations.

The record itself is stored only once.

### 5.3 Persistent infinite edges

An unbounded edge is stored as:

```text
full supporting edge
finite origin
direction
```

The **full** supporting edge is the persistent duplicate key. This matters in degenerate geometry because several minimal algorithmic edges may represent the same geometric ray.

### 5.4 Hashing roles

Hashing is modular because several logically independent states exist:

- persistent database identity;
- cell-local queue first-claim state;
- cell-local edge occurrence state;
- global edge-completeness diagnostics.

These states may use the same implementation family but must not share lifetime or reset semantics.

### 5.5 Queue and edge container sharding

`DirectHash`, `StaticHash<N>`, and `DynamicHash<>` configure one table, a fixed set of independent tables, or a dynamically growing table family.

Lock type and sharding strategy are independent choices. Parallel contention therefore depends both on the selected synchronization policy and on the container routing.

---

## 6. Mesh architecture

### 6.1 `AbstractMesh`

`AbstractMesh` supplies common behavior for compatible mesh representations:

- public/internal signature conversion;
- canonical vertex storage;
- finite and infinite record iteration;
- boundary index conversion;
- primary/secondary address ownership;
- tombstone-aware iteration;
- node/vertex filtering primitives;
- index-mapping integration.

Concrete meshes provide storage and mapping hooks.

### 6.2 `VoronoiMesh`

`VoronoiMesh` is the ordinary stored mesh.

It owns:

- stable ordinary node storage;
- dense public/internal mappings;
- extended boundary nodes;
- persistent database;
- primary and secondary vertex-address lists;
- mesh-level infinite-edge addresses.

Public node deletion changes the public mapping without rewriting stable internal signatures.

### 6.3 `ReorderedMeshView`

`ReorderedMeshView` is a non-owning facade over another mesh.

It changes public numbering while sharing:

- node storage;
- persistent database;
- stable signatures;
- address lists;
- infinite-edge storage;
- boundary geometry.

`ComputeVoronoi` uses reordered branch views to make one selected cell block appear at the beginning of a branch-local public numbering.

### 6.4 `SerialMesh`

`SerialMesh` remains an experimental composite-mesh component. It is not currently the foundation of ordinary refinement or periodic HighVoronoi.

The active incremental implementation instead uses dedicated computation facades and backends described below.

### 6.5 `HighVoronoiMesh`

`HighVoronoiMesh` is the central persistent owner for incremental and periodic HighVoronoi state.

It separates several node notions that coincide in an ordinary `VoronoiMesh`:

```text
stable internal nodes
    every node ever appended has one permanent ordinary internal index

active internal nodes
    currently active visible nodes and periodic reference nodes

visible public nodes
    only active nodes intended for external output

compute-public nodes
    temporary dense bijective ordering used by HighVoronoiComputeMesh
```

The mesh also owns two boundaries:

**external boundary**
: fixed geometry visible to the user.

**internal boundary**
: starts equal to the external boundary and may expand only along periodic planes to contain reference nodes.

Plane ordering never changes, so boundary internal indices remain stable.

### 6.6 Invisible periodic reference nodes

Every invisible reference node points to one visible stable internal generator and stores a shift mask over the external boundary planes.

For a visible reference `x` and mask `M`:

```text
x_M = x + sum(periodic_shift(plane), plane in M)
```

Reference nodes are normal active ordinary nodes inside the internal computation geometry. They are not additional public visible cells.

Deleting a visible node also deactivates every invisible copy that refers to it.

### 6.7 Public HighVoronoi projection

`HighVoronoiMesh` itself is intentionally read-oriented as a visible mesh.

Its mapping is non-bijective:

- a visible internal node maps to its visible public index;
- an active invisible reference maps to the public index of its visible reference;
- an inactive node has no public representation.

Public vertex iteration uses **only the visible cell's own address lists**. It does not union vertices stored on invisible reference cells into the visible cell.

For each record:

1. internal generators are projected onto visible generator indices;
2. the stored vertex position is translated through periodic partner shifts into the external visible domain.

This distinction is essential. Invisible reference cells are computational scaffolding, not independent sources of public cell vertices.

### 6.8 `HighVoronoiDataMeshView`

`data_mesh()` returns a non-owning AbstractMesh view over **all active internal nodes** in stable insertion order, with deleted nodes omitted.

It is useful for diagnostics of the complete active internal geometry.

### 6.9 `HighVoronoiComputeMesh`

`HighVoronoiComputeMesh` is the dense one-to-one compute representation used by incremental operations.

It exposes all active internal nodes through a dense bijective mapping. A requested set of NEW or repair cells can be moved into the public prefix:

```text
preferred front
    -> public indices [0, front_size)

remaining active nodes
    -> stable insertion order afterwards
```

`ComputeVoronoi` therefore sees an ordinary dense mesh and remains unaware of HighVoronoi visibility, references, pending deletion, or periodization.

### 6.10 `VisibleFirstMesh`

`VisibleFirstMesh` is another read-through dense bijective view over all active internal nodes.

Its ordering is:

```text
[ visible nodes in visible-public insertion order ]
[ remaining active invisible nodes in stable internal order ]
```

`visible_end()` marks the exact visible prefix.

This view is useful for validation that needs full internal node geometry while checking only visible cells.

---

## 7. Search and geometric primitives

### 7.1 Search-tree abstraction

The CRTP search layer combines ordinary backend candidates with active boundary candidates.

Common operations are:

```cpp
nn(data, skip)
knn(data, count, skip)
inrange(data, radius, skip)
rebuild()
```

Each backend exposes reusable `SearchData`.

### 7.2 Brute-force and nanoflann

`BruteForceSearchTree` is the reference backend.

`NanoflannSearchTree` maintains a persistent KD-tree over ordinary public nodes. Active boundary mirrors remain outside the KD-tree because they change with the active cell.

The project vendors its local `nanoflann.hpp` and is tested against that interface.

### 7.3 `safe_copy()` and `rebind()`

These operations have different semantics.

`safe_copy()`
: independent mutable search/raycast state inside the same mesh branch; immutable backend core state may be shared.

`rebind(new_mesh)`
: construct backend state for a different mesh/view numbering world.

A worker does not rebuild its own branch KD-tree.

### 7.4 `RayCaster`

`RayCaster` owns the nearest-neighbour/range-search logic required to advance a Voronoi ray.

Current strategies are:

- `ClassicRaycast`;
- `InRangeRaycast`.

The cast interface returns either a finite endpoint or an infinite ray.

`RayCaster` also owns vertex correction and semantic verification.

### 7.5 `EdgeIterator`

`EdgeIterator` enumerates incident Voronoi edges.

It supports:

- general-position vertices;
- degenerate vertices with more than `dimension + 1` generators;
- minimal edge signatures;
- full supporting-edge signatures;
- queue-time and systematic-exploration modes;
- shared FEI cache state;
- cycle-error monitoring and normal correction.

### 7.6 Minimal edge versus full support

Two signatures must remain distinct.

**Minimal edge**
: the `d`-generator algorithmic edge used for local ownership and traversal.

**Full supporting edge**
: every vertex generator lying in the edge hyperplane.

The full support may contain generators with global indices below the active cell. It must not be globally filtered by the minimum of the full signature.

Persistent infinite-edge identity and completeness diagnostics use the full supporting edge.

### 7.7 Corrected degenerate-edge existence test

The C++ `scan_for_edge` implementation contains one deliberate correction relative to the Julia implementation.

A primary generator is rejected only after every admissible supporting-face completion has been attempted. A single failed greedy completion is not sufficient to prove that no valid edge exists.

This change aligns the code with the existence criterion underlying the mathematical localized edge characterization.

---

## 8. Ordinary `ComputeVoronoi`

### 8.1 Top-level role

`ComputeVoronoi` owns the complete execution topology:

```text
MeshThreading
    one branch or several communicating reordered branches

CastThreading
    one worker or several geometry workers inside each branch

VoronoiThreading
    multi-threaded iff either axis is multi-threaded
```

### 8.2 Branch construction

With one mesh thread, the supplied mesh is used directly.

With several mesh threads:

1. the selected cell interval is partitioned;
2. one `ReorderedMeshView` is built per block;
3. the raycaster prototype is rebound to that branch;
4. one `SystematicVoronoi` is created per branch;
5. branches run concurrently and communicate discovered vertices.

### 8.3 Local-first vertex registration

A newly found finite vertex follows:

```text
worker
    -> source branch queue_vertex()
           local first-claim / OnQueueEdges
    -> ComputeVoronoi global registration
           persist once
           translate to target branch numbering
           target queue_vertex() using target prototype iterator
```

A branch may require a vertex locally even if another branch won the persistent database race, so local queueing precedes global persistence.

Worker `EdgeIterator` instances never cross branch boundaries.

### 8.4 `SystematicVoronoi`

One `SystematicVoronoi` owns one branch:

- branch-local cell range;
- current cell;
- cell-local vertex queue;
- cell-local EdgeHash;
- shared FEI cache;
- prototype RayCaster and EdgeIterator;
- configured workers;
- active mirror scratch.

Its cell lifecycle is:

```text
reset local state and boundary mirrors
load existing cell vertices
if needed, find one descent vertex
explore queued vertices
follow incomplete edges
persist/communicate finite vertices
persist infinite rays
advance to next cell
```

### 8.5 `VoronoiWorker`

Workers own only reusable geometric scratch:

- independent RayCaster state;
- queue-time EdgeIterator;
- systematic EdgeIterator;
- vertex/signature buffers;
- descent/walk state;
- orthogonalization scratch.

No global threading policy is encoded in the worker.

### 8.6 Descent correctness detail

Before attempting final vertex correction, descent initializes the correction candidate from the projected descent point. Correction therefore acts on the current descent geometry rather than stale scratch state.

---

## 9. Ordinary incremental operations

### 9.1 Shared backend architecture

`RefineVoronoi` and `RemoveVoronoi` are shared geometric operations over a compile-time `IncrementalVoronoiBackend`.

The ordinary `VoronoiMesh` and `HighVoronoiMesh` specializations differ in:

- storage access;
- computation facade;
- node-append/delete semantics;
- optional periodic event collection.

The geometry itself is shared.

### 9.2 `RefineVoronoi`

`RefineVoronoi` inserts one NEW block.

The key fixed-boundary invariant is:

> every genuinely new finite Voronoi vertex produced by adding generators contains at least one NEW generator.

Therefore the algorithm can:

1. append or identify the NEW block;
2. move exactly those NEW cells into a temporary compute prefix;
3. run ordinary `ComputeVoronoi` only on that prefix;
4. collect OLD cells appearing in NEW vertices;
5. scan only the affected OLD cells for obsolete OLD vertices;
6. erase vertices for which a NEW generator is strictly closer.

There is deliberately **no second `ComputeVoronoi(AFFECTED)` pass** for insertion. New vertices attached to OLD cells have already been discovered from the NEW-cell side and registered in the persistent address lists.

The AFFECTED set only restricts old-vertex invalidation work.

### 9.3 `RemoveVoronoi`

Removal is different because deleting generators can open topology that contains only surviving OLD generators.

`RemoveVoronoi` is therefore two-phase:

```text
remove()
    identify selected stable internal nodes
    erase incident vertices
    deactivate/remove generators
    retain affected surviving OLD cells

compute()
    front the affected surviving cells
    run ordinary ComputeVoronoi against the current mesh
```

The delayed form is intentional: a `RefineVoronoi` may execute between `remove()` and `compute()`. The closure then uses the refined current mesh.

---

## 10. HighVoronoi lifecycle

### 10.1 Pending state

`HighVoronoiMesh` owns a persistent integrated-node watermark.

Visible nodes appended after the watermark are pending NEW visible nodes.

`erase_visible_nodes()`:

- interprets all requested indices in the pre-deletion visible numbering;
- deactivates the selected visible stable internal nodes;
- deactivates every periodic reference copy pointing to them;
- rebuilds visible numbering;
- records deleted internal indices;
- leaves incident database records available until `ComputeHighVoronoi` consumes the deletion.

After a successful reconstruction, `mark_integrated()` advances the watermark to the current internal node count and clears pending deleted-node state.

### 10.2 `ComputeHighVoronoi`

`ComputeHighVoronoi` is orchestration, not a second geometry algorithm.

Its current high-level sequence is:

```text
1. pending deletion:
       construct RemoveVoronoi(existing removed nodes)
       remover.remove()

2. pending visible insertion:
       RefineVoronoi(existing NEW visible block)

3. close periodic requests generated by that refine:
       append reference block
       RefineVoronoi(reference block)
       repeat until no new periodic reference is required

4. if a removal exists:
       remover.compute()
       close any periodic requests created by the removal closure

5. run final sparse periodic-boundary repair

6. mark the complete internal state integrated
```

Delete and refine retain independent AFFECTED state and lifetime.

### 10.3 HighVoronoi round state and event database

`HighVoronoiComputeDataBase` wraps the persistent database during one compute phase.

`AbstractMesh::store_vertex()` has already converted the compute-public signature into canonical stable internal numbering before the wrapper sees it.

The wrapper extracts three kinds of information directly from each new sigma:

**affected OLD nodes**
: during the NEW-cell phase, OLD generators belonging to NEW vertices are marked for later invalidation scans.

**bootstrap periodic boundary requests**
: if a sigma contains a periodic internal-boundary generator, every visible ordinary generator in the sigma requests the corresponding periodic mirror/reference.

**periodic closure requests**
: if a visible generator shares a vertex with an invisible reference shifted through periodic plane `p`, that visible generator requests the partner of `p`.

Vertices containing periodic internal-boundary generators are **not persisted**. They remain available to the local queue/edge traversal of the current `SystematicVoronoi`, while database insertion returns address zero.

### 10.4 Periodic request materialization

Periodic requests are collected per visible reference node and plane.

The materializer:

1. ignores inactive/non-visible references and non-periodic planes;
2. deduplicates requested planes;
3. enumerates all admissible non-empty shift combinations;
4. prevents simultaneous selection of both members of one periodic partner pair;
5. skips already existing reference copies;
6. computes the shifted point from the visible reference plus the selected periodic shifts.

The resulting reference nodes form an ordinary NEW refinement block.

### 10.5 Internal boundary expansion

Before new reference points are appended, the internal boundary is expanded as needed.

Only periodic planes may move.

The external boundary is never changed.

Plane order and partner relations remain unchanged, preserving the stable high-end boundary index encoding.

Periodic internal-boundary vertices are never persisted by the compute database, so moving those artificial planes does not require a stale persistent-record pass for those boundary vertices.

### 10.6 Periodic closure termination

Every appended reference block is refined by the ordinary `RefineVoronoi` machinery. New vertices may expose additional periodic neighbours and therefore additional reference requests.

Closure repeats until no materializable request remains or the configured round limit is exceeded.

This is an incremental exact reference-node closure. It is distinct from the fast quasi-periodic bulk-copy algorithm described later in the reference paper.

---

## 11. Hidden periodic-boundary vertices and sparse repair

### 11.1 The missing invariant

For insertion on a fixed boundary:

```text
new finite vertex
    => contains at least one NEW generator
```

This underlies the efficient NEW-cell-only `RefineVoronoi`.

During periodization, however, the algorithm also changes the **artificial internal boundary**. Moving that boundary can reveal a finite vertex whose ordinary generators are all OLD visible generators.

Such a vertex is genuinely new relative to the previous clipped geometry but contains no NEW reference node. Therefore it is unreachable from a pure NEW-cell refinement pass.

This is the key periodic exception to the ordinary refinement invariant.

### 11.2 Bootstrap capture

During the initial geometry, the internal periodic boundary still coincides with the external visible boundary.

Periodic-boundary vertices are suppressed from persistent storage, but selected ones are captured transiently.

For every opening edge that drops a periodic boundary generator, the algorithm walks inward in the bootstrap geometry and retains the signature of the persistent opposite vertex.

Only signatures survive across periodization. Database addresses and old edge objects are deliberately not retained because later refinement can invalidate them.

### 11.3 Final sparse repair

After all periodic reference nodes are present and the final internal boundary is known:

1. seed signatures are resolved against the final persistent geometry;
2. stale seeds are discarded;
3. a sparse repair mesh exposes the complete final active node geometry;
4. only transient vertex lists for currently seeded/dirty visible cells are exposed to the repair traversal;
5. unchanged serial `ComputeVoronoi` explores those cells;
6. newly persisted real vertices are written both to the HighVoronoi persistent address lists and to the transient repair lists;
7. affected visible cells are processed in further sparse sweeps until no genuinely new repair vertex is stored.

The repair is intentionally serial and localized.

### 11.4 Guardrail

The sparse repair exists only for final vertices hidden by the bootstrap boundary and therefore for topology involving OLD visible generators.

If the repair would newly discover a vertex containing an invisible periodic reference generator, it throws `logic_error`.

Such a vertex should have been discovered by the ordinary periodic reference closure; accepting it in the repair would hide a defect in mirror-request propagation.

---

## 12. Threading and synchronization

### 12.1 Two independent axes

HighVoronoiCC separates:

- **MeshThreading**: independent reordered mesh branches;
- **CastThreading**: geometry workers inside one branch.

### 12.2 Combined `VoronoiThreading`

State that can be mutated either by local workers or by remote branch communication uses a combined policy that is multi-threaded whenever either execution axis is multi-threaded.

### 12.3 Shared-state lock ownership

| Shared structure | Lock selection | Reason |
|---|---|---|
| persistent database | database configuration | records may be stored concurrently |
| persistent address lists | database/thread policy | concurrent registrations |
| branch vertex queue | combined Voronoi threading | local workers and remote branches |
| branch EdgeHash | combined Voronoi threading | remote prototype queueing also mutates edge state |
| branch FEI cache | combined Voronoi threading | prototype and local workers share cache |
| prototype EdgeIterator scratch | combined Voronoi threading | remote queue passes share this iterator |
| existing-vertex iterator | CastThreading | shared only by workers in one branch |
| worker scratch | none | one worker owns it |
| HighVoronoi round mirror/affected flags | atomic bit vectors | worker event collection |
| HighVoronoi structural mutation | serialized phases | mappings/node sets/boundary move between compute phases |

### 12.4 Current cast-worker scheduling limitation

The shared cast-worker queue still has no explicit in-flight-work termination protocol.

A worker can observe the queue as temporarily empty while another worker is computing a ray that may later append work. The observing worker then stops its current exploration phase.

This is treated as a load-balancing/scheduling limitation rather than part of the current correctness design, and it remains a later optimization target.

---

## 13. Numerical robustness

### 13.1 Precision model

Normal stored geometry uses the configured scalar type.

The robust fallback layer intentionally provides exactly two relevant precision tiers for the current vertex correction:

```text
normal path:   Float64 / configured double path
fallback:      software Float128
```

`Float128` is `boost::multiprecision::cpp_bin_float_quad` with at least binary128 precision.

`long double` is deliberately not used as an intermediate precision tier because its precision is platform dependent and may equal `double`.

### 13.2 Edge-direction normal solve

The Julia-compatible normal-direction solve uses an ordinary Householder QR and preserves the orientation rule based on the final QR diagonal.

The edge-direction solver is intentionally not replaced wholesale by column-pivoted QR because the orientation semantics depend on this structure.

Degenerate edge construction separately monitors accumulated cycle/orthogonality error and may correct the already constructed basis when necessary.

### 13.3 Vertex correction system

For a support of exactly `d + 1` generators, choose `p_0` and an existing candidate `r`. The correction `delta` satisfies

```text
(p_i - p_0)^T delta
    = 1/2 ( |r - p_i|^2 - |r - p_0|^2 )
```

for the remaining `d` generators.

The implementation:

1. normalizes every row by `|p_i - p_0|`;
2. factorizes the normalized matrix once with `ColPivHouseholderQR`;
3. computes
   `pivot_ratio = min |R_ii| / max |R_ii|`
   as a cheap conditioning proxy;
4. rejects the ordinary-precision path immediately when the proxy is below `vertex_condition_tolerance`;
5. otherwise iterates corrections with the reused factorization;
6. tests convergence through
   `|delta| / |r_initial - p_0|`;
7. rejects non-finite or non-decreasing correction sequences;
8. falls back to the same correction performed in software Float128 when the double path is poorly conditioned or fails to converge.

The corrected output is written only on success.

### 13.4 Why variance alone is insufficient

Equal-distance variance measures a residual-like defect. In an ill-conditioned support configuration, a tiny residual can coexist with a substantially larger forward positional error.

The current implementation therefore does not treat a very small variance as a complete accuracy certificate. Conditioning and correction convergence are checked explicitly.

### 13.5 Semantic vertex verification

`RayCaster::verify_vertex()` checks more than equal distances:

- at least `d + 1` generators are present;
- every supplied generator lies on one sphere within absolute/relative tolerance;
- a nearest neighbour belongs to the stored sigma;
- no foreign generator lies significantly inside the sphere;
- every generator found on the sphere within tolerance is represented in sigma;
- generator differences span the full ambient dimension, checked with column-pivoted QR.

Numerical correction improves a candidate; semantic verification establishes that the candidate still represents the intended Voronoi vertex.

---

## 14. Validation and regression strategy

### 14.1 Component tests

Focused readable tests cover:

- locks and atomic bit vectors;
- hash tables;
- persistent database behavior;
- point layout;
- stored/computed/hybrid node access;
- boundaries;
- mesh mappings and views;
- search trees;
- normal solving and vertex correction;
- edge enumeration;
- ray casting;
- ordinary complete construction;
- infinite-edge storage;
- edge closure;
- computed/hybrid mesh engines;
- incremental refine/remove;
- HighVoronoi indexing/views;
- periodic reference generation;
- periodic incremental workflows.

### 14.2 `verify_mesh()`

`verify_mesh()` checks stored finite-vertex occurrences for geometric consistency.

It is not a completeness proof.

### 14.3 `verify_mesh_complete()`

This additionally reconstructs global edge incidence using **full supporting edges**.

A complete finite edge contributes exactly two finite endpoints.

A complete unbounded edge contributes one finite endpoint and one persistent infinite-edge occurrence.

Repeated minimal representations of the same full edge at one degenerate persistent vertex are deduplicated before global counting.

A third occurrence is invalid.

### 14.4 `verify_mesh_complete_cells()`

This applies the same logic to a selected public cell interval.

Its principal current use is periodic HighVoronoi validation through:

```text
VisibleFirstMesh
cells [0, visible_end())
```

The invisible reference cells may exist only as scaffolding and are not required to form the public visible-cell closure.

### 14.5 `compare_meshes()`

Compatible meshes can be compared bidirectionally by:

- node data;
- vertex signatures;
- vertex positions.

This supports serial/parallel and reference comparisons.

### 14.6 Important end-to-end regressions

**4D general position**
: ordinary bounded construction.

**4D Cartesian degeneracy**
: strongly non-general grid with exact expected vertex count.

**Unbounded 2D**
: persistent infinite rays and exact edge closure.

**3D parallel smoke**
: serial, mesh-parallel, cast-parallel and combined shared-state behavior.

**Range consistency**
: full computation versus selected/reordered computation ranges.

**Ordinary incremental**
: refine and remove workflows.

**HighVoronoi incremental**
: incremental state versus batched construction and compute-view range equivalence.

**Periodic HighVoronoi**
: a partially periodic 3D unit cube with 30 visible generators is compared against an independent ordinary `VoronoiMesh` built from a complete `3 x 3` x/y tiling, i.e. 270 explicit generators. The visible HighVoronoi cells must be geometrically consistent, topologically complete, and cover the projected periodic reference vertices.

**Periodic incremental HighVoronoi**
: initial periodic construction, further visible insertion near periodic faces, and visible deletion are compared against fresh explicit periodic reference states.

These reference tests are deliberately stronger than merely verifying that every stored vertex is individually valid.

---

## 15. Corrections and strengthened invariants relative to the Julia implementation

The C++ implementation is not a literal preservation of every Julia implementation detail. Several issues exposed while building stronger reference tests are now explicit invariants.

### 15.1 Degenerate edge completion

A primary generator may not be rejected after one failed greedy supporting-face completion. All admissible completions must be exhausted first.

### 15.2 Fixed-boundary refinement versus moving periodic boundary

The NEW-generator invariant is valid for generator insertion on fixed geometry.

It is not sufficient when an artificial periodic boundary moves. Final OLD-only vertices exposed by that move require the dedicated sparse repair described in Section 11.

This is why the periodic repair is not folded into generic `RefineVoronoi`: doing so would weaken a useful and correct fixed-boundary invariant.

### 15.3 Residual versus forward positional accuracy

Very small equal-distance variance does not guarantee a small positional error under poor conditioning.

The C++ corrector therefore adds conditioning-aware CPQR and a software-Float128 fallback instead of relying only on the residual-style variance criterion.

### 15.4 Visible cells and reference-cell ownership

Invisible periodic copies are computational references, not alternate public owners of visible-cell vertex lists.

Public visible output uses each visible cell's own persistent address lists and projects generator identities/positions into the external domain.

---

## 16. Current limitations and roadmap

### Current limitations

- spherical Voronoi construction is not yet implemented;
- the fast quasi-periodic copy/modify/paste algorithm from the paper is not yet implemented as a public C++ workflow;
- `SerialMesh` is not part of the validated current incremental/periodic path;
- cast-worker scheduling does not keep idle workers alive while other workers have in-flight ray computations;
- `direct_cast` remains a future fused-search hook;
- disk-backed persistent storage is not implemented;
- volume, interface measure, quadrature and integration layers are not yet implemented;
- the public include surface has been split thematically, but HighVoronoi still has a transitive ordinary-backend include dependency that can be cleaned later;
- configuration documentation outside this Architecture/README still needs to catch up with the HighVoronoi and new vertex-correction layers.

### Near-term roadmap

1. release/consumer/install and sanitizer checks;
2. update remaining configuration/example documentation;
3. improve cast-worker termination/load balancing;
4. benchmark hash sharding and parallel scaling;
5. stabilize remaining public API details;
6. continue profiling the validated incremental/periodic paths.

### Longer-term roadmap

1. spherical Voronoi construction;
2. fast quasi-periodic generation;
3. optimized fused `direct_cast` backends;
4. additional nearest-neighbour backends where profiling justifies them;
5. optional disk-backed storage;
6. volume/interface/quadrature/integral algorithms.

---

## 17. Architectural invariants

Future changes should preserve the following invariants.

### Indexing and storage

- [ ] Public ordinary indices are dense and zero-based.
- [ ] Persistent ordinary internal indices do not change implicitly.
- [ ] `max(Index)` remains an invalid marker.
- [ ] Boundary internal indices remain disjoint from ordinary internal indices.
- [ ] Persistent finite-vertex signatures are canonical stable internal signatures.
- [ ] Persistent infinite-edge signatures are canonical full supporting-edge signatures.
- [ ] Public reordering does not rewrite persistent signatures.
- [ ] Backend-native search indices never escape unchecked.
- [ ] HighVoronoi invisible reference nodes have stable internal identities.
- [ ] Deleting a visible HighVoronoi node also deactivates all references to it.

### Geometry

- [ ] Inactive mirror nodes never participate in search.
- [ ] Computed nodes are never assumed to provide stable pointers.
- [ ] Minimal-edge ownership is not inferred from the minimum of the full degenerate support.
- [ ] Infinite-edge identity uses the full supporting edge.
- [ ] Degenerate edge existence scans exhaust admissible completions before rejecting a primary.
- [ ] Numerical correction never substitutes for semantic vertex verification.
- [ ] A small variance alone is not treated as a forward-error certificate.

### Ordinary construction

- [ ] Existing stored vertices enter only cell-local queue/edge state.
- [ ] Newly found finite vertices are queued locally before global persistence/communication.
- [ ] Persistent duplicate loss does not suppress required local queue state.
- [ ] Worker EdgeIterators never cross a `SystematicVoronoi` branch boundary.
- [ ] Remote branches queue communicated vertices through their own prototype iterator.
- [ ] Exactly one queue claimant performs the `OnQueueEdges` pass for a signature in one cell generation.
- [ ] Infinite rays are persisted using full-edge identity.
- [ ] `SystematicVoronoi` remains independent of global mesh-branch topology.
- [ ] `VoronoiWorker` remains independent of global threading policies.

### Incremental construction

- [ ] On fixed geometry, insertion computes NEW cells and invalidates affected OLD vertices without a second affected-cell compute.
- [ ] Removal retains an affected OLD set until its explicit closure compute.
- [ ] Delayed removal closure may run after an intervening refinement.
- [ ] Structural mutation does not race with delegated ComputeVoronoi phases.

### HighVoronoi and periodization

- [ ] External boundary geometry is fixed.
- [ ] Only periodic internal boundary planes may expand.
- [ ] Periodic internal-boundary vertices are not persisted.
- [ ] HighVoronoi computation uses a dense bijective mapping over all active internal nodes.
- [ ] Public HighVoronoi output exposes only visible cells.
- [ ] Public visible-cell vertex iteration does not union address lists from invisible reference cells.
- [ ] Invisible generators project to their visible originals only for external presentation.
- [ ] Stored vertex positions are translated into the external visible domain for public output.
- [ ] Periodic reference closure iterates until no new admissible reference request remains.
- [ ] Moving the artificial periodic boundary is allowed to expose OLD-only finite vertices.
- [ ] Those OLD-only vertices are recovered only by the final sparse periodic repair.
- [ ] Sparse periodic repair must not hide a missing invisible-reference vertex from normal periodization.
- [ ] `mark_integrated()` is called only after deletion, insertion, periodic closure, removal closure and sparse repair have all succeeded.

### Concurrency

- [ ] Synchronization is attached to shared structures rather than worker code.
- [ ] Combined Voronoi threading is multi-threaded whenever either execution axis is multi-threaded.
- [ ] Branch queue, EdgeHash, FEI cache and prototype iterator use the combined shared lock where required.
- [ ] Parallel construction uses a thread-safe persistent database/address-list policy.
- [ ] Worker-local scratch is not shared.
- [ ] HighVoronoi bit-vector sizes and structural mappings change only between parallel compute phases.

### Numerical robustness

- [ ] Edge-direction orientation retains the intended Householder-QR convention.
- [ ] Vertex correction uses a separate column-pivoted QR system.
- [ ] Vertex correction rows are normalized.
- [ ] Poor conditioning can force direct Float128 fallback.
- [ ] Correction convergence is judged from correction size, not variance alone.
- [ ] Failed correction leaves the original candidate unchanged.
- [ ] Float128 fallback does not depend on platform `long double` precision.

### Completeness

- [ ] Global geometric edge identity uses the full supporting edge.
- [ ] Degenerate repeated minimal-edge representations are deduplicated per persistent endpoint.
- [ ] A complete finite edge has exactly two finite endpoint occurrences.
- [ ] A complete unbounded edge has one finite endpoint and one persistent infinite-edge occurrence.
- [ ] A third occurrence is invalid.
- [ ] Periodic public completeness is checked on visible cells, while invisible reference cells may remain computational scaffolding.

### Performance and ownership

- [ ] Hot loops can recycle buffers.
- [ ] Search backends do not allocate a point per coordinate lookup.
- [ ] Worker safe copies inside one branch share immutable backend core state where safe.
- [ ] `rebind()` creates backend state for a different mesh/view numbering world.
- [ ] Persistent records are stored once and referenced by address lists.
- [ ] Pure views do not silently acquire persistent ownership.
- [ ] Sparse periodic repair remains localized rather than degrading into a full final recomputation.
- [ ] Volume and integration logic remains outside the Voronoi construction core.

---

## 18. Source map

| Area | Principal files |
|---|---|
| Complete public umbrella | `include/highvoronoi/highvoronoi.hpp` |
| Ordinary public entry | `include/highvoronoi/voronoi.hpp` |
| HighVoronoi public entry | `include/highvoronoi/high_voronoi.hpp` |
| Optional mesh-engine entry | `include/highvoronoi/mesh_engines.hpp` |
| Shared public core | `include/highvoronoi/core.hpp` |
| Public database forwarding | `include/highvoronoi/database.hpp` |
| Version interface | `include/highvoronoi/version.hpp` |
| Public configuration | `include/highvoronoi/parameters.hpp` |
| Precision helpers | `include/highvoronoi/detail/float.hpp` |
| Locks / execution policies | `include/highvoronoi/detail/locks.hpp` |
| Atomic structural/event flags | `include/highvoronoi/detail/atomic_bit_vector.hpp` |
| Hash functions/generators | `include/highvoronoi/detail/hash_functions.hpp`, `hash_generators.hpp` |
| Queue/edge hashes | `queue_hash_table.hpp`, `queue_hash_table_2.hpp`, `edge_hash_table.hpp` |
| Hash containers/types | `hash_table_containers.hpp`, `hash_types.hpp` |
| Persistent database | `include/highvoronoi/detail/hvdatabase.hpp` |
| Hybrid database | `include/highvoronoi/detail/hybrid_database.hpp` |
| Address lists | `include/highvoronoi/detail/read_write_list.hpp` |
| Index views | `include/highvoronoi/detail/hvview.hpp` |
| Points | `include/highvoronoi/geometry/point.hpp` |
| Nodes | `include/highvoronoi/geometry/voronoi_nodes.hpp` |
| Boundaries / periodic pairs | `include/highvoronoi/geometry/boundary.hpp` |
| Mesh base | `include/highvoronoi/geometry/abstract_mesh.hpp` |
| Ordinary stored mesh | `include/highvoronoi/geometry/voronoi_mesh.hpp` |
| Mesh views | `include/highvoronoi/geometry/mesh_view.hpp` |
| Experimental composite mesh | `include/highvoronoi/geometry/serial_mesh.hpp` |
| Computed mesh abstraction | `include/highvoronoi/geometry/compute_mesh.hpp` |
| Compute engine base | `include/highvoronoi/geometry/compute_mesh_engine.hpp` |
| Cuboid compute engine | `include/highvoronoi/geometry/cuboid_mesh_engine.hpp` |
| HighVoronoi owner | `include/highvoronoi/geometry/high_voronoi_mesh.hpp` |
| HighVoronoi compute facade/event DB | `include/highvoronoi/geometry/high_voronoi_compute_mesh.hpp` |
| Visible-first HighVoronoi view | `include/highvoronoi/geometry/visible_first_mesh.hpp` |
| Generic incremental compute facade | `include/highvoronoi/geometry/incremental_voronoi_compute_mesh.hpp` |
| Incremental backend dispatch | `include/highvoronoi/geometry/incremental_voronoi_backend.hpp` |
| HighVoronoi incremental backend | `include/highvoronoi/geometry/high_voronoi_incremental_backend.hpp` |
| Incremental insertion | `include/highvoronoi/geometry/refine_voronoi.hpp` |
| Incremental removal | `include/highvoronoi/geometry/remove_voronoi.hpp` |
| HighVoronoi orchestration | `include/highvoronoi/geometry/compute_high_voronoi.hpp` |
| Hidden periodic-boundary repair | `include/highvoronoi/geometry/periodic_boundary_repair.hpp` |
| Search base | `include/highvoronoi/geometry/abstract_search_tree_crtp.hpp` |
| Brute-force search | `include/highvoronoi/geometry/brute_force_search_tree_crtp.hpp` |
| nanoflann search | `include/highvoronoi/geometry/nanoflann_search_tree_crtp.hpp` |
| Search factory | `include/highvoronoi/geometry/search_tree_factory_crtp.hpp` |
| Bundled nanoflann | `include/highvoronoi/detail/nanoflann.hpp` |
| Normal/vertex solving | `include/highvoronoi/geometry/normal_solver.hpp` |
| Edge enumeration | `include/highvoronoi/geometry/edge_iterator.hpp` |
| Ray casting / verification | `include/highvoronoi/geometry/raycaster.hpp` |
| Geometry worker | `include/highvoronoi/geometry/voronoi_worker.hpp` |
| Cell/branch construction | `include/highvoronoi/geometry/systematic_voronoi.hpp` |
| Ordinary top-level construction | `include/highvoronoi/geometry/compute_voronoi.hpp` |
| Mesh validation | `include/highvoronoi/geometry/mesh_validation.hpp` |
| Selected-cell validation | `include/highvoronoi/geometry/mesh_cell_validation.hpp` |
| Ordinary incremental regression | `tests/test_voronoi_incremental_workflows.cpp` |
| HighVoronoi workflow regression | `tests/test_high_voronoi_nonperiodic_workflow.cpp` |
| HighVoronoi incremental regression | `tests/test_high_voronoi_incremental_vs_batched.cpp` |
| HighVoronoi compute mapping regression | `tests/test_high_voronoi_compute_view_range_equivalence.cpp` |
| Full periodic reference regression | `tests/test_high_voronoi_periodic_reference.cpp` |
| Periodic incremental reference regression | `tests/test_high_voronoi_periodic_incremental_reference.cpp` |
| Periodic neighbour/reference regression | `tests/test_voronoi_periodic_refine_edge_neighbours.cpp` |
| Vertex solver regression | `tests/test_qr_normal_solver_20260807.cpp` |
| Parallel smoke regression | `tests/test_compute_voronoi_parallel_smoke.cpp` |
| Unbounded completeness regression | `tests/test_compute_voronoi_unbounded_completeness_20260814.cpp` |

---

## 19. Relation to the reference paper

Martin Heida, *On the parallelized efficient computation of high dimensional Voronoi diagrams on bounded, unbounded, spherical and periodic domains*, WIAS Preprint No. 3197, 2025.

Relevant mathematical sections include:

- localized convex-cone edge enumeration: Section 2.5;
- data structures and exhaustive construction: Sections 3.1-3.3;
- RayCast and Descent: Section 3.4;
- parallelization: Section 3.6;
- refinement: Section 4;
- fast quasi-periodic generation: Section 5;
- spherical Voronoi diagrams: Section 6;
- numerical robustness: Section 7.

The current C++ implementation uses Section 4's local-refinement structure as a foundation but separates fixed-boundary refinement from the additional internal-boundary motion required by exact periodic reference-node closure.

The software-Float128 correction and the explicit sparse periodic-boundary repair are implementation-level robustness/correctness mechanisms of the C++ code and should be read as extensions of the underlying mathematical construction rather than as replacements for the paper's geometry.
