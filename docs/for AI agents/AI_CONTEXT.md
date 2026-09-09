# HighVoronoiCC AI Context

**Purpose:** compact routing and invariant reference for AI-assisted development.  
**Snapshot:** source/header/example/test bundles and CMake inventory generated 2026-09-22 08:30 CEST, git `77aa830da132d2d92ada78d92672bd0b3be19b2d` with local modifications.  
**Use rule:** consult this file first, use the current `ai_index.jsonl` to route a task to the relevant source/tests, and consult `ARCHITECTURE.md` for cross-cutting ownership/data-flow questions. Source code remains authoritative if a later repository change makes any routing metadata stale.

## 1. Hard architectural invariants

### Geometry kernel

- There is one local Voronoi geometry kernel. Higher-level workflows must reuse it rather than reimplement local geometry.
- Core chain: `SearchTree -> RayCaster + EdgeIterator -> VoronoiWorker -> SystematicVoronoi -> ComputeVoronoi`.
- Fixes in edge enumeration, ray casting, descent, numerical correction, semantic verification, or systematic exploration are expected to propagate to ordinary, incremental, HighVoronoi, periodic-repair, and spherical workflows.

### Node interface

- The old compile-time `Stored` / `Computed` / `Hybrid` node-access-mode architecture is gone.
- Do **not** invent or refer to `NodeAccessMode`, `StoredNodeAccess`, `ComputedNodeAccess`, `HybridNodeAccess`, or `NodeHandle` as current abstractions.
- Every node provider implements `AbstractVoronoiNodes<Scalar, Dim, Index>` with one common semantic contract.
- Generic hot paths use `get_data(node, coordinate)` and `copy_node(node, target)`.
- Convenience access returns one owning `PointType<Scalar, Dim>` independent of how coordinates are produced.
- Whether coordinates are contiguous, computed, or routed among sources is a provider implementation detail.
- `VoronoiNodes::stable_node_data()` is a concrete optimization only; generic algorithms must not require it.
- `CompositeVoronoiNodes` creates one stable global node-index space over engine ranges plus explicitly materialized nodes. Its `is_stored()` query is provider-specific routing information, not a node-access mode.
- `detail::MappedVoronoiNodes` is a non-owning facade applying a mapping to another node provider.

### Index worlds

Keep these index worlds distinct:

1. **public ordinary indices** — dense presentation numbering of a mesh/view;
2. **stable internal ordinary indices** — persistent identities used by stored signatures;
3. **boundary public indices** — `public_node_count + plane_index`;
4. **boundary internal indices** — high-end encoding `max(Index)-1-plane_index`; `max(Index)` itself is invalid;
5. **backend-native search indices** — private to the search backend and converted before leaving it;
6. **HighVoronoi visible-public indices** — external visible cells;
7. **HighVoronoi compute-public indices** — temporary dense bijective ordering over all active internal nodes.

Persistent finite and infinite signatures use stable internal encoding. Reordering, filtering, deletion, or visible projection must not rewrite persistent identities.

### Mapping policies

- `DenseIndexMapping` is the ordinary bijective public/internal mapping with stable internal slots and rebuildable dense public numbering.
- `ProjectedIndexMapping` is HighVoronoi's intentionally many-to-one internal-to-public projection: invisible periodic references may project to the same visible public index as their visible original.
- `HighVoronoiComputeMesh` must therefore never use the visible projected mapping for computation; it builds a separate dense bijection over active internal nodes.

### Storage and ownership

- `HVDataBase` is the canonical append-oriented persistent finite-vertex/facet database.
- Finite and infinite/facet payload records use canonical stable-internal signatures.
- Records are referenced by addresses from per-node address lists; a vertex is stored once and may appear through primary/secondary address ownership.
- Deletion tombstones records/signatures rather than renumbering persistent storage.
- Structural changes occur only at explicit safe points between delegated compute phases.
- `ReadWriteList`, hashes, queues, FEI caches, and round event flags own synchronization appropriate to the shared state they protect.
- `VoronoiWorker` must remain independent of global threading policy and global lock ownership.

### Neighbour state and dirty propagation

- Neighbour storage is **active current architecture**. `NeighbourStorage` owns one current neighbour-record address plus one mesh dirty bit per stable internal cell; `NeighbourDatabase` is append-only historical storage.
- `NeighbourTrackingState` owns the registry of consumer trackers and the neighbour version. Each `NeighbourDirtyTracker` is independent.
- Geometry changes propagate mesh neighbour dirtiness into every live consumer tracker at explicit lifecycle boundaries. Updating/clearing one consumer must never clear another consumer's dirtiness.
- Integration depends on this subsystem. Do **not** reintroduce a second ad-hoc dirty-bit/neighbour cache for integrals.

### Integration

- `VoronoiIntegral` owns persistent result/history state only; it does not own traversal policy or an integration algorithm.
- Persistent integral data is indexed by stable internal cell identity. Public/worker reordering must never rebase integral ownership.
- The cardinal alignment invariant is `neighbours[k] <-> area[k] <-> interface_integral[k,*]`, including duplicate neighbour occurrences.
- Ordinary integration views plan `[NEW][DIRTY OLD][CLEAN OLD]`; HighVoronoi integration views plan `[NEW visible][DIRTY OLD visible][CLEAN OLD visible][REFERENCES]`.
- HighVoronoi integration stores results only for visible cells. Invisible periodic references remain distinct shifted generators for geometry and must not be projected away inside the integrator.
- `integrate(...)` computes each prepared cell immediately, then performs cleanup/publication after the complete first pass.
- `ParallelIntegrationExecution` parallelizes the first-pass update prefix with private worker views. Cleanup/publication is a serial transactional phase.
- Algorithms must explicitly provide `make_worker_algorithm()` to participate in parallel execution.
- Polygon is the deterministic reference path. FastPolygon caches lower-dimensional subfacets by complete canonical sorted stable-internal generator keys and reuses them across higher-dimensional faces/cells.
- FastPolygon hash values are lookup fingerprints only; exact stored-key comparison determines identity. Cache contents are pass-local.
- Monte-Carlo supports parallel first-pass sampling plus serial interface cleanup/symmetrization.
- Heuristic consumes geometry from a source integral; forced source updates are allowed, but dependency cycles are rejected. HeuristicMC performs MC geometry first and heuristic function integration only after final geometry cleanup.
- Current tests validate parallel Polygon, FastPolygon, Monte-Carlo, Heuristic, HeuristicMC, and periodic HighVoronoi Polygon integration.

### Degeneracy and edge identity

- `EdgeView::indices()` / `minimal_indices()` is the minimal `d`-generator edge key.
- `EdgeView::full_indices()` is the complete supporting generator set in the edge hyperplane.
- In degenerate geometry, topological/global edge identity uses the **full supporting edge**, not merely one minimal representation.
- `skip()` is a generator index from the current vertex signature and must not occur in `full_indices()`.
- Degenerate `scan_for_edge` must exhaust admissible supporting-face completions before rejecting a primary generator.
- FEI initialization cache is shared within an iterator family/branch according to the lock policy; numerical scratch remains local/reused.

### Ray casting and numerical robustness

- `ClassicRayCaster` is the general-position path; `InRangeRayCaster` is the non-general/degenerate path.
- Vertex correction uses row-normalized equal-distance systems with column-pivoted QR and can fall back directly to software Float128 for poor conditioning/nonconvergence.
- A tiny equal-distance variance is not a forward positional accuracy certificate.
- Failed correction must not silently replace the original candidate.
- Numerical correction does not replace semantic vertex verification.

### Incremental construction

- Ordinary incremental insertion/removal uses dedicated temporary compute facades around the persistent `VoronoiMesh`; it does not require a composite `SerialMesh`.
- `IncrementalVoronoiComputeMesh` creates a temporary dense compute numbering; `IncrementalVoronoiComputeDataBase` collects AFFECTED events from canonical stable-internal signatures.
- Fixed-geometry insertion and removal have different closure semantics; do not merge them casually.
- Structural mutation and affected-bit-vector resizing happen outside active `ComputeVoronoi` phases.

### HighVoronoi / periodization

- `HighVoronoiMesh` is the persistent owner of visible nodes, invisible periodic references, active/deleted state, external/internal boundaries, mappings, address lists, and persistent database access.
- External boundary is fixed user geometry. Every user-visible node appended initially or during refinement must lie inside this external boundary (up to an explicit tolerance); `append_visible_node(s)` throws otherwise.
- Internal boundary may expand only along periodic planes to contain **invisible reference nodes**; plane order remains stable. Do not place user refinement nodes in the expanded internal-only region.
- Invisible periodic reference nodes are active computational ordinary nodes with permanent stable internal identities, but they are not separate visible public cells.
- Public HighVoronoi vertex iteration uses the visible cell's own address lists only; it does not union invisible-reference cell lists into the visible cell.
- Internal generators are projected to visible originals only for public presentation, and stored positions are translated back into the external visible domain.
- Periodic reference closure iterates until no new admissible reference request remains.
- Moving the bootstrap/artificial periodic boundary can expose OLD-only finite vertices. Those are handled by dedicated sparse periodic repair, not by weakening the fixed-boundary refinement invariant.

### Spherical construction

- `SphericalVoronoiMesh` is implemented and current. Do **not** describe spherical construction as future/unimplemented.
- It reduces the problem to the Euclidean Voronoi cell of internal origin node 0 in `R^d` and projects the resulting vertices radially to the unit sphere.
- Public signatures remove the origin generator.
- `AntipodalHemisphere` mode appends invisible antipodes and identifies `q` with `-q`, representing `S^(d-1)/{x~-x}`.
- It exposes one initial `compute()` followed by incremental `refine()` / `remove()` workflows while reusing ordinary geometry infrastructure.
- `SphereVoronoiMesh` is the ordinary spherical alias. `AntipodalSphericalVoronoiMesh` owns invisible antipodes; refine/remove add/delete visible+antipode pairs and public output is canonicalized to one hemisphere.
- Spherical neighbour projection is current functionality. In antipodal mode duplicate public neighbour occurrences may be meaningful and must not be deduplicated blindly.

## 2. Canonical component routing

- **Local geometry:** `normal_solver.hpp`, `edge_iterator.hpp`, `raycaster.hpp`, `voronoi_worker.hpp`, `systematic_voronoi.hpp`, `compute_voronoi.hpp`.
- **Search:** `abstract_search_tree_crtp.hpp`, brute-force reference backend, nanoflann backend, composite backend, `search_tree_factory_crtp.hpp`. `detail/nanoflann.hpp` is third-party vendor code.
- **Mesh/indexing:** `abstract_mesh.hpp`, `mesh_index_mapping.hpp`, `mapped_voronoi_nodes.hpp`, `voronoi_mesh.hpp`, `mesh_view.hpp`, `visible_first_mesh.hpp`.
- **Nodes/boundaries:** `point.hpp`, `voronoi_nodes.hpp`, `boundary.hpp`.
- **Persistent storage:** `hvdatabase.hpp`, `read_write_list.hpp`, hash/container headers, `atomic_bit_vector.hpp`, `locks.hpp`.
- **Ordinary incremental:** `incremental_voronoi_compute_mesh.hpp`, `incremental_voronoi_backend.hpp`, `refine_voronoi.hpp`, `remove_voronoi.hpp`.
- **HighVoronoi:** `high_voronoi_mesh.hpp`, `high_voronoi_compute_mesh.hpp`, `high_voronoi_incremental_backend.hpp`, `compute_high_voronoi.hpp`, `periodic_boundary_repair.hpp`.
- **Spherical:** `spherical_voronoi_mesh.hpp`, reusing ordinary compute/search/raycast infrastructure.
- **Neighbours/dirty tracking:** `neighbour_database.hpp`, `neighbour_storage.hpp`, `neighbour_dirty_tracker.hpp`, plus forwarding in concrete meshes/views.
- **Persistent integration state/views:** `integral_data.hpp`, `voronoi_integral.hpp`, `integration_view.hpp`, `high_voronoi_integration_view.hpp`.
- **Integration driver:** `integrator.hpp`.
- **Integration algorithms:** `polygon_integrator.hpp`, `fast_polygon_integrator.hpp`, `monte_carlo_integrator.hpp`, `heuristic_integrator.hpp`, `heuristic_mc_integrator.hpp`; FastPolygon cache internals in `integration/detail/fast_polygon_cache.hpp`.
- **Validation:** `mesh_validation.hpp`, `mesh_cell_validation.hpp`.

The current `ai_index.jsonl` is regenerated from the same 2026-09-22 source/test/example inventory and CMake registry as this context. Use it for per-file and feature/test routing; use `ARCHITECTURE.md` for the fuller ownership and data-flow explanation.

## 3. Reusable toolbox — check before implementing new infrastructure

Before introducing a new container, lock wrapper, packed flag array, index map/view, search scratch object, or numerical helper, check the existing infrastructure below. Reuse it when the semantics match rather than creating a parallel project-local abstraction.

### Shared storage and compact state

- **`ReadWriteAddressList<DataT, LockT>`** — `storage/read_write_list.hpp`
  - Use for vector-like per-node address lists that need lock-aware `size()`, indexed reads, `push_back()`, `try_back()`, and `erase_if()`.
  - It is already used by persistent mesh address storage. Prefer it over adding a new mutex-protected `std::vector<Address>` when this interface is sufficient.
  - Structural moves transfer only the payload, never lock state; moves/reallocation require external synchronization at a structural phase boundary.
- **`detail::BitVector`** — `core/detail/atomic_bit_vector.hpp`
  - Use for compact Boolean state with no concurrent mutation, e.g. structural `active`/`visible`-style flags.
  - Prefer it over `std::vector<bool>`, byte-per-flag vectors, or new manual bit packing when only the existing bit API is needed.
- **`detail::AtomicBitVector`** — `core/detail/atomic_bit_vector.hpp`
  - Use for compact flags that several workers may set/reset/test concurrently, e.g. affected/event/request flags.
  - Individual bit operations are atomic and use relaxed ordering: this is atomic flag storage, not an inter-thread publication/synchronization primitive.
  - `resize()` is structural and must run only between parallel compute phases; do not resize while workers access the vector.

### Synchronization and threading policies

- **`detail::EmptyLock`** — no-op lock for statically single-threaded structures.
- **`detail::ReadWriteLock`** — FIFO read/write spin lock for shared mutable structures with short critical sections.
- **`detail::BusyFIFOLock`** — exclusive FIFO spin lock for *very short* critical sections only; do not hold it across I/O, blocking work, or long computations.
- **`detail::ReadLockGuard<Lock>` / `WriteLockGuard<Lock>`** — RAII wrappers; prefer these over hand-written lock/unlock pairs.
- **`detail::ReadToWriteLockGuard<Lock>` / `with_write_lock_from_read()`** — temporarily replace an already-held read lock by a write lock. The transition is **not atomic**: revalidate every condition observed before the upgrade after the write lock is acquired.
- **`detail::with_read_lock()` / `with_write_lock()`** — exception-safe callable wrappers for short protected operations.
- **`SingleThread` / `MultiThread`** — project execution policies selecting `EmptyLock` / `ReadWriteLock` and the worker count. Prefer threading-policy-derived lock types over ad-hoc mutex choices in generic code.

Rule: synchronization belongs to the shared mutable object. Do not add locks to `VoronoiWorker` or other worker-local geometry merely because the caller is parallel.

### Index mappings, views, and non-owning adapters

- **`DenseIndexMapping<Index>`** — `mesh/mesh_index_mapping.hpp`; use for stable-internal / dense-public **bijective** mappings.
- **`ProjectedIndexMapping<Index>`** — same file; use only when the mapping is deliberately many-to-one as in visible HighVoronoi projection.
- **`detail::MappedVoronoiNodes<Nodes, Mapping>`** — `mesh/mapped_voronoi_nodes.hpp`; use to expose an existing node provider through an existing mapping instead of writing another forwarding node class.
- **`HVView` family (`SwitchView`, `ShuffleView`, `CombinedView`, ...)** — `detail/hvview.hpp`; use for lightweight reversible temporary index permutations/compositions. Do not replace persistent mesh mappings with an `HVView`, and do not invent a new permutation wrapper before checking these types.
- **`AbstractMesh::CombinedAddressList<List>`** — `mesh/abstract_mesh.hpp`; read-only concatenation of two existing address sources without allocating or copying a combined container.

### Reusable numerical/search infrastructure

- **`QRNormalSolver` / `ExtendedQRNormalSolver`** — `algorithm/normal_solver.hpp`; reuse for edge normals and equal-distance vertex correction instead of introducing a second local QR implementation. The extended solver is the project precision fallback.
- **`ExtendedFloat` / `ExtendedVector` / `ExtendedMatrix`** — `detail/float.hpp`; use the project-defined extended-precision layer rather than introducing platform-dependent `long double` assumptions.
- **Search-tree `SearchData`** — search backends expose reusable caller-owned scratch/result storage. Hot loops should create it once and reuse it instead of allocating temporary search buffers for every query.
- **Hash infrastructure** — `hash_types.hpp`, `hash_table_containers.hpp`, queue/edge hash tables and parameter adapters already provide project-specific duplicate/edge tracking, sharding, and lock-policy integration. Before adding `unordered_set<vector<Index>>` or another bespoke concurrent hash structure, check whether the existing queue/edge semantics fit.

### Neighbour/integration infrastructure

- **`NeighbourStorage`** — `storage/neighbour/neighbour_storage.hpp`
  - Reuse for the mesh's current per-stable-cell neighbour record and dirty flag. It already routes through append-only historical records.
- **`NeighbourDirtyTracker` / `NeighbourTrackingState`** — `storage/neighbour/neighbour_dirty_tracker.hpp`
  - Use when a new consumer needs an independent invalidation stream from mesh geometry. Do not clear mesh dirtiness or another consumer's tracker as a shortcut.
- **`VoronoiIntegral`** — `integration/voronoi_integral.hpp`
  - Persistent result/history owner indexed by stable internal cells; request one per logically independent integral field.
- **`VoronoiIntegrationView` / `HighVoronoiIntegrationView`**
  - Reuse for NEW/DIRTY scheduling, temporary reordering, neighbour/payload permutation, worker views and periodic reference geometry. Do not create a second cell-work planner.
- **`SerialIntegrationExecution` / `ParallelIntegrationExecution`**
  - Use the existing driver execution policies rather than spawning algorithm-specific cell threads.
- **FastPolygon facet stores/cache policy** — `integration/detail/fast_polygon_cache.hpp`
  - Reuse the exact-key verified serial/shared caches when implementing another FastPolygon-compatible payload; do not replace them with hash-only identity.

### Reuse decision rule

When implementing a new feature, explicitly ask in this order:

1. Is the required state already represented by one of the utilities above?
2. Is the new object merely a mapped/viewed form of existing nodes, indices, or address lists?
3. Is concurrency local to the object, and can its lock type be derived from the existing threading/database policy?
4. Is there already reusable scratch or numerical infrastructure for the hot operation?
5. Only introduce a new helper if the required semantics materially differ; document that difference so a later implementation does not create a second competing abstraction.

## 4. Optional / non-core components

### Engine-backed source/import infrastructure

The following components are current source, but they are **not** node access modes and are not the core ordinary/incremental geometry path:

- `mesh/engine/compute_mesh_engine.hpp` — polymorphic engine contract;
- `mesh/engine/compute_mesh.hpp` — read-oriented `AbstractMesh` facade over one engine;
- `mesh/engine/cuboid_mesh_engine.hpp` — analytic Cartesian engine;
- `storage/hybrid_database.hpp` — logical-address router combining ordinary database records with engine address ranges.

`HighVoronoiMesh::append_engine()` and `CompositeVoronoiNodes::append_engine()` can integrate engine ranges into the stable internal node space. This should be described as engine/provider composition, not `Hybrid` node access.

## 5. Negative knowledge / hallucination guards

- **No `SerialMesh` in the supplied current source snapshot.** Do not propose edits to `serial_mesh.hpp`, base current workflows on `SerialMesh`, or list it as a current component unless a newer snapshot explicitly reintroduces it.
- **No active Stored/Computed/Hybrid node-access modes.** Those names may still occur descriptively or inside old test/document text, but not as current generic node types.
- **Neighbour storage is now integrated.** Any older note claiming `NeighbourDatabase`/`NeighbourStorage` is standalone is stale. Concrete meshes expose neighbour computation/storage/dirty tracking and integration consumes it.
- `HybridDataBase` still exists, but its name is storage/engine routing only; it is not evidence that `HybridNodeAccess` exists.
- `ComputeMesh` still exists, but its `computed` wording refers to engine-backed geometry, not a generic node-access category.
- `detail/nanoflann.hpp` is third-party vendor code. Prefer changing HighVoronoi's wrapper unless a vendor-level change is explicitly intended.
- `HighVoronoiMesh` public mapping is non-bijective; do not use it as the computation mapping.
- `verify_mesh()` checks consistency of stored occurrences, not completeness. Use completeness validators for missing/extra-edge questions.
- `highvoronoi/highvoronoi.hpp` currently omits the integration module. `integrals.hpp` is separate, and Polygon/FastPolygon/Heuristic/HeuristicMC use explicit algorithm headers. Do not assume one umbrella include exposes them.
- `raycaster_old.hpp`, `raycaster_new.hpp`, `raycaster_prop1.hpp`, and `systematic_voronoi_old.hpp` are unclassified historical/development variants in the supplied installable tree. Route current kernel work to `raycaster.hpp` / `systematic_voronoi.hpp` unless the task explicitly targets a variant.

## 6. Task-to-source routing shortcuts

- **Wrong/missing Voronoi endpoint:** `raycaster.hpp` -> `normal_solver.hpp` -> `systematic_voronoi.hpp` -> relevant search backend.
- **Wrong degenerate edge / adjacents from edge support:** `edge_iterator.hpp` first; inspect `full_indices`, `skip`, FEI logic; then `systematic_voronoi.hpp` consumers.
- **Vertex duplicate / persistent signature issue:** `abstract_mesh.hpp` -> `hvdatabase.hpp` -> queue hash implementation.
- **Missing/duplicate global edge / completeness:** `edge_iterator.hpp` full support -> `edge_hash_table.hpp` -> `mesh_validation.hpp` / `mesh_cell_validation.hpp`.
- **Public/internal index bug:** `mesh_index_mapping.hpp` -> concrete mesh (`voronoi_mesh.hpp` or `high_voronoi_mesh.hpp`) -> `mapped_voronoi_nodes.hpp` / `mesh_view.hpp`.
- **Search result wrong:** `abstract_search_tree_crtp.hpp` -> selected backend -> `search_tree_factory_crtp.hpp`; use brute force as reference.
- **Ordinary incremental insertion/removal:** `refine_voronoi.hpp` or `remove_voronoi.hpp` -> `incremental_voronoi_backend.hpp` -> `incremental_voronoi_compute_mesh.hpp`.
- **Periodic reference generation/closure:** `compute_high_voronoi.hpp` -> `high_voronoi_mesh.hpp` -> `high_voronoi_compute_mesh.hpp`.
- **OLD-only vertex after periodic boundary expansion:** `periodic_boundary_repair.hpp`; do not weaken ordinary refinement rules.
- **Visible HighVoronoi output mismatch:** `high_voronoi_mesh.hpp` projected mapping + position translation -> `visible_first_mesh.hpp` for internal-vs-visible validation.
- **Spherical/antipodal issue:** `spherical_voronoi_mesh.hpp` -> ordinary compute/search/raycast stack.
- **Wrong/stale neighbour/interface topology:** concrete mesh `compute_neighbors`/forwarding -> `neighbour_storage.hpp` -> `neighbour_database.hpp` -> `neighbour_dirty_tracker.hpp`; inspect `edge_iterator.hpp` if the geometric neighbour set itself is wrong.
- **Integral not recomputed after mesh change:** `voronoi_integral.hpp` dirty tracker -> `integration_view.hpp` / `high_voronoi_integration_view.hpp` planning -> mesh neighbour dirty propagation.
- **Polygon/FastPolygon wrong volume/interface:** selected integrator -> integration view neighbour occurrence ordering -> `polygon_integrator.hpp` geometry; for FastPolygon also inspect canonical cache keys and `fast_polygon_cache.hpp`.
- **Parallel integration race/mismatch:** `integrator.hpp` worker partitioning/serial cleanup -> algorithm `make_worker_algorithm()` -> worker view mapping; for FastPolygon inspect shared cache state/sharding.
- **Heuristic/HeuristicMC stale source or recursion:** `heuristic_integrator.hpp` / `heuristic_mc_integrator.hpp` -> source integral/algorithm -> integration dependency cycle guard.
- **Engine-backed/import mesh issue:** `compute_mesh.hpp` -> concrete engine -> `compute_mesh_engine.hpp`; if mixed logical addresses are involved, inspect `hybrid_database.hpp`.
- **Threading race:** identify the actually shared structure first (`locks.hpp`, `read_write_list.hpp`, hash/queue/cache/round flags). Do not add global locks to `VoronoiWorker` by default.

## 7. Snapshot coverage caveat

The 2026-09-22 upload includes maintained source bundles, examples, the current CMake test registry, broad foundation/integration/workflow tests, README, AI context/index, architecture documentation and the reference preprint. This is sufficient to treat the implementation/test bundles as current ground truth for the documented features above.

Two cautions remain:

1. `ai_index.jsonl` is routing metadata for the 2026-09-22 snapshot, not a substitute for newer source code; regenerate it when the repository structure materially changes;
2. the unclassified installable headers `raycaster_old.hpp`, `raycaster_new.hpp`, `raycaster_prop1.hpp`, and `systematic_voronoi_old.hpp` are present but are not registered as the maintained kernel path by the current CMake/tests. Treat them as historical/development variants unless newer evidence says otherwise.
