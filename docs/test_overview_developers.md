# Test overview (developers only)

This page is a source-level inventory of the registered CTest suite. Its purpose is not to say merely which class a test “covers”, but to record the concrete experiment performed by each executable/test function and the result that is expected. Use it when deciding where a new regression belongs, when checking whether an invariant is already protected, or when investigating how a bug escaped the suite.

## Scope and maintenance rule

- The inventory follows the tests registered through `highvoronoi_add_test(...)` in the current `CMakeLists.txt` snapshot: **76 test executables**.
- Helper headers are mentioned when the actual test functions live there; examples and benchmarks are not counted as tests.
- Descriptions below are derived from the test source itself. If a test changes, update this page in the same patch.
- “Expects” means the executable has a machine-checked condition (`check`, explicit failure return, `assert`, etc.). Where a file only prints diagnostics, that is called out explicitly.
- Several older tests use plain C `assert()`. If those executables are compiled with `NDEBUG`, the assertions disappear; this page marks the important cases.

## Quick coverage warnings

- **`tests/foundation/test_point_layout.cpp` is currently diagnostic only:** it prints aliasing behavior but contains no active assertion.
- **`tests/search/test_search_tree_voronoi.cpp` is currently diagnostic only:** it prints expected/measured search behavior for BruteForce and KD backends but does not fail on value mismatches.
- **`test_basic`, `test_locks`, `test_hvdatabase`, and `test_boundary` rely substantially or entirely on plain `assert()`**; their effective checking depends on assertions being enabled.
- **The CMake description of `test_qr_normal_solver` is broader than the current source.** The registered comment mentions vertex correction and conditioning/fallback behavior, but the test file currently exercises only the ordinary and Extended/Float128 QR normal solvers. Vertex-correction/fallback coverage must therefore be sought elsewhere or added explicitly.
- **`test_high_voronoi_incremental_vs_batched` has a stale name/CMake description.** Its current source is an ordinary `VoronoiMesh` + explicit `InRangeRaycast` diagnostic matrix; it neither constructs `HighVoronoiMesh` nor compares incremental against batched construction.
- A green test executable therefore does not necessarily mean every behavior it prints or its target name suggests is protected by a regression assertion. When adding a bug regression, prefer an explicit machine-checked expectation.

## Running focused groups

```bash
ctest --test-dir build/release --output-on-failure
ctest --test-dir build/release -L compute --output-on-failure
ctest --test-dir build/release -L highvoronoi --output-on-failure
ctest --test-dir build/release -L integrals --output-on-failure
ctest --test-dir build/release -L threading --output-on-failure
```

## Foundation, public API, storage and synchronization

### `test_basic`

**File:** `tests/foundation/test_basic.cpp`  
**CTest labels:** `foundation`

**`main()`** — Includes the complete public umbrella and checks that `highvoronoi::version()` equals `"0.1.0"`. The check is a plain C `assert`, so this executable becomes only an include/link smoke test when compiled with `NDEBUG`.

### `test_convenience_api`

**File:** `tests/public/test_convenience_api.cpp`  
**CTest labels:** `foundation;api;compute;incremental;highvoronoi`

**`main()`** — Exercises the public Level-1/2 facade end to end. It verifies the default scalar/index/hash/raycast types at compile time; constructs an ordinary Level-1 mesh, then computes, refines and removes; constructs the Level-1 `MultiThread` variant and checks the `ReadWriteLock`/`StaticHash<16>` storage choices; constructs a Level-2 `InRangeRaycast` mesh; instantiates runtime-dimension ordinary and HighVoronoi facades; computes/refines/removes a HighVoronoi mesh; then exercises the integration facade. For integration it expects the Level-1 FastPolygon default to update cells and publish positive finite geometry, an unchanged second call to update zero cells, scalar-function storage to have one bulk component and one component per interface, explicit Level-2 `Polygon{}` to update cells, and the same Level-1 integration call to work on a HighVoronoi wrapper.

### `test_point_layout`

**File:** `tests/foundation/test_point_layout.cpp`  
**CTest labels:** `foundation`

**`main()` — diagnostic only.** Creates a `StaticPointView<double,3>` over `data+1`, prints the view, mutates the backing array and prints the view again to demonstrate aliasing. The formerly intended padding `static_assert` is commented out, and the active program contains no machine-checked expectation; it can therefore exit successfully even if the printed behavior changes.

### `test_locks`

**File:** `tests/foundation/test_locks.cpp`  
**CTest labels:** `foundation;threading`

- **`test_empty_lock()`** — Calls every lock/unlock mode on `EmptyLock`, then runs `with_read_lock` and `with_write_lock`; expects no locked state and the guarded write to take effect.
- **`test_busy_fifo_lock_mutual_exclusion()`** — Eight threads increment one non-atomic counter under `BusyFIFOLock`; expects exactly `8*5000` increments and an unlocked final state.
- **`test_temporary_write_lock_from_read()`** — Enters a read section, temporarily converts it to a write section with `with_write_lock_from_read`, changes a value, and expects the read lock to be reacquired afterward.
- **`test_read_write_lock_exclusion()`** — Starts mixed reader, writer and read-to-write tasks concurrently, tracks simultaneous readers/writers and a protected value, and expects readers to overlap only with readers, writers to be exclusive, every writer update to survive, and all task counters to reach their planned totals.

These checks use plain `assert()`; with `NDEBUG` the concurrency workload still executes but the assertions are disabled.

### `test_atomic_bit_vector`

**File:** `tests/foundation/test_atomic_bit_vector.cpp`  
**CTest labels:** `foundation;threading`

- **`test_non_atomic_basic_bit_operations()`** — Creates a 130-bit `BitVector`; sets/resets/flips bits around 64-bit word boundaries, then `clear()`/`set_all()`; expects exact bit changes and hidden unused tail bits.
- **`test_non_atomic_bounds_and_empty_vector()`** — Exercises an empty vector and out-of-range reads/writes; expects harmless writes and `false` reads.
- **`test_non_atomic_structural_operations()`** — Exercises `reserve`, grow/shrink `resize`, initialization with `false`/`true`, and `push_back` across a word boundary; expects prefix preservation and correct new-bit initialization.
- **`test_basic_bit_operations()` / `test_bounds_and_empty_vector()`** — Repeats the core semantics for `AtomicBitVector`.
- **`test_concurrent_sets_in_shared_words()` / `test_concurrent_resets_in_shared_words()`** — Concurrently sets and clears bits that share packed words; expects no lost `fetch_or`/`fetch_and` updates.

### `test_read_write_list`

**File:** `tests/storage/test_read_write_list.cpp`  
**CTest labels:** `foundation;storage`

**`main()`** — Creates an empty `ReadWriteAddressList`, appends `10,20,30`, resizes to five entries with zero fill, then replaces index 1 with `77`. It expects size, indexing, resize initialization and targeted mutation to preserve all unrelated entries.

### `test_hash_tables`

**File:** `tests/storage/test_hash_tables.cpp`  
**CTest labels:** `foundation;storage`

- **Hash/view tests (`test_hash64`, `test_uint64_view`, `test_hash_functions`, `test_hash_generators`)** — Feed equivalent 32-bit/64-bit views and different keys into the hash functions/generators; expect representation-independent hashes for equal values, distinct fingerprints for different keys, and full nonrepeating in-range probe sequences.
- **`test_queue_hash_table()`** — Starts with capacity 4, inserts 12 keys to force growth, verifies lookup/read-only lookup, erase/reinsert and `clear`; expects all keys to remain discoverable through growth and deletion semantics to be exact.
- **`test_edge_hash_table()`** — Inserts many edge keys, then records first, second and third cell occurrences of one edge; expects the third occurrence to be reported as overfull and `clear()` to reset the table.
- **`test_static_queue_hash_container()` / `test_dynamic_queue_hash_container()`** — Route keys across fixed shards or dynamically created block-ranged tables; expect lookup/erase/reinsert/clear to preserve routing, read-only lookup not to grow a dynamic container, and clear not to shrink its table family.
- **`test_static_edge_hash_container()` / `test_dynamic_edge_hash_container()`** — Repeat two-occurrence/third-occurrence edge semantics through sharded container policies.
- **`test_parameter_types()` / `test_parameter_runtime_forwarding()`** — Verify `SingleThread`/`MultiThread` counts and that `DirectHash`, `StaticHash` and `DynamicHash` runtime parameters are forwarded unchanged.

### `test_hvdatabase`

**File:** `tests/storage/test_hvdatabase.cpp`  
**CTest labels:** `foundation;storage`

- **`test_push_and_read()`** — Inserts one finite vertex, checks address 1, `contains`, duplicate rejection, round-trip of position/signature, erase/tombstone behavior, and reinsertion at a new append-only address.
- **`test_push_and_read_facet()`** — Inserts an infinite/facet record with unsorted signature and direction, expects in-place signature sorting, duplicate rejection, exact round-trip, tombstone reads that leave caller buffers unchanged/empty as documented, and successful reinsertion.
- **`test_block_crossing()` / `test_facet_block_crossing()`** — Use long signatures with a deliberately tiny data block so records cross multiple blocks; expect exact readback of all fields.
- **`run_database_configuration()` / `test_all_database_hash_configurations()`** — Re-run the above against Direct/Static/Dynamic hash containers and both modern/classic queue-table implementations; expect correct block accounting after growth.
- **`test_wide_scalar_and_index_types()`** — Repeats the database suite with `double` and `uint64_t` to ensure wide values also survive cross-block packing.

The file uses plain `assert()` for its expectations.

### `test_neighbour_database_parallel`

**File:** `tests/storage/neighbours/test_neighbour_database_parallel.cpp`  
**CTest labels:** `foundation;storage;threading;neighbours`

- **`test_large_record()`** — Stores and reads a 70,000-entry neighbour record containing duplicates; expects a nonzero address and exact round-trip across database blocks.
- **`test_parallel_database_writes()`** — Multiple threads append disjoint neighbour records; expects every record to remain intact.
- **`test_read_write_address_list_reuse()`** — Appends neighbour-version addresses concurrently to `ReadWriteAddressList` and reads the newest value with `try_back`; expects all publications to survive.
- **`test_dirty_type_selection()`** — Checks that a parallel lock policy selects `AtomicBitVector` for dirty state.

### `test_neighbour_dirty_tracking`

**File:** `tests/storage/neighbours/test_neighbour_dirty_tracking.cpp`  
**CTest labels:** `foundation;storage;threading;neighbours`

**`main()`** — Registers two consumer dirty trackers, marks cells across several packed words, propagates dirty state, clears consumers independently, grows the tracked cell space, and appends new cells. It expects both consumers to receive identical propagated bits without clearing the mesh-owned bits, new cells to appear dirty, expired registrations to be pruned, and the global neighbour version to change only on explicit commit. It also replaces a cell's current neighbour address and verifies that the old append-only record remains readable while normal cell reads follow only the new current address.

### `test_hybrid_neighbour_database`

**File:** `tests/storage/neighbours/test_hybrid_neighbour_database.cpp`  
**CTest labels:** `foundation;storage;engine;neighbours`

- **`test_stored_engine_stored_routing()`** — Writes a stored record, registers an engine range, then writes another stored record. It expects one contiguous logical address space, address 0 to mean “no record”, stored records on both sides to remain readable, engine-local neighbours to gain the configured node offset while duplicates and boundary encodings survive, and the next logical address to follow all segments.
- **`test_registration_requires_authority()`** — Attempts to register an engine without neighbour authority; expects registration to be rejected rather than reserving virtual neighbour records.

### `test_neighbour_storage_hybrid`

**File:** `tests/storage/neighbours/test_neighbour_storage_hybrid.cpp`  
**CTest labels:** `foundation;storage;engine;neighbours`

**`main()`** — Creates hybrid neighbour storage for a new cell, publishes stored records and then virtual engine addresses. It expects a new cell to start dirty with no address, publication to replace exactly one current address without implicitly clearing dirtiness, old append-only records to remain readable through retained addresses, and multiple engine addresses to route through the same hybrid database.

### `test_hvview`

**File:** `tests/foundation/test_hvview.cpp`  
**CTest labels:** `foundation;mapping`

- **`test_switch_view_zero_based()` / `test_switch_view_vectors()`** — Exercise zero-based range switching and vector mapping in both directions; expect public/internal mappings to be reversible and unaffected outside the switched range.
- **`test_combined_view()`** — Composes views and expects the composed mapping/inverse to equal applying the underlying mappings in sequence.
- **`test_shuffle_view()` / `test_shuffle_view_with_explicit_length()`** — Exercise explicit permutations, inverse lookup and an explicit public length; expect bijective mapping over the configured domain.
- **`test_invalid_switch_range()`** — Supplies an invalid switch range and expects rejection rather than an inconsistent view.

### `test_progress_meter`

**File:** `tests/core/test_progress_meter.cpp`  
**CTest labels:** `foundation;threading`

- **`test_basic_timing_and_output()`** — Advances a meter with controlled timing; expects no output before `Delta_T`, one correctly formatted line after expiry, correct percentage/elapsed time, monotone external `TOTAL`, and batched increments to appear in later output.
- **`test_parallel_counter()`** — Multiple threads call `increase`; expects no lost increments and no premature output.
- **`test_parallel_single_reporter()`** — Multiple reporter candidates cross one expired window; expects exactly one emitted report and monotone timestamps.
- **`test_finish_and_elapsed_access()`** — Calls `finish()` and elapsed-time accessors; expects a forced final line with current percentage and readable/up-to-date elapsed totals.

## Nodes, boundaries, meshes, neighbours and mesh views

### `test_voronoi_nodes`

**File:** `tests/core/voronoi_nodes/test_voronoi_nodes.cpp`  
**CTest labels:** `mesh;nodes`

The executable delegates most checks to helper headers in the same directory.

- **`test_stored_branch()` / `test_dynamic_stored_branch()`** in `hvcc_voronoi_nodes_basic_tests_20260803.hpp` — Construct fixed- and dynamic-dimension stored node providers; exercise node access/copy/data interfaces and expect stored coordinates/dimensions to agree.
- **`test_computed_branch()`** — Exposes nodes through a compute-engine/provider branch and expects the common node interface to return the engine coordinates.
- **`test_mixed_branch()`** — Combines stored and computed ranges and expects one continuous index space with the correct provider selected for every index.
- **`test_boundary_check_uses_common_interface()`** — Runs boundary-related node checks through the common provider API rather than provider-specific access.
- **`test_extended_stored_branch()` / `test_extended_computed_base()` / `test_extended_mixed_base()`** in `hvcc_voronoi_nodes_extended_tests_20260803.hpp` — Activate mirror/boundary nodes over stored, computed and mixed bases; expect real and mirror indices/coordinates to be exposed through one extended-node interface.
- **`test_precomputed_stored_base()` / `test_precomputed_computed_base()` / `test_precomputed_mixed_base()`** — Exercise precomputed mirror data for each base type and expect the same public mirror semantics.
- **`test_invalid_mirror_index_is_rejected()`** — Supplies an invalid mirror index and expects rejection.
- `hvcc_voronoi_nodes_compile_contracts_20260803.hpp` contributes compile-time interface checks used by the executable.

### `test_boundary`

**File:** `tests/core/test_boundary.cpp`  
**CTest labels:** `mesh;boundary`

- **`test_reflection_and_projection()`** — Builds planes/cuboids, reflects points and projects to planes; expects the reflected/projected coordinates and signed geometry to match the plane definitions.
- **`test_boundary_and_intersection()`** — Adds boundary planes and evaluates membership/intersection helpers; expects points/segments to be classified against the configured domain correctly.
- **`test_periodic_indices_and_conversion()`** — Creates periodic plane pairs and exercises periodic metadata/index conversion; expects partner indices/shifts to round-trip and non-periodic planes to remain distinct.
- **`test_dynamic_default_and_deduction()`** — Exercises dynamic-dimension/default construction and type deduction; expects runtime dimension/plane data to match the supplied vectors.
- **`test_voronoi_nodes()`** — Uses boundary mirror nodes together with Voronoi-node access and expects mirrored coordinates/indices to agree with boundary geometry.

Expectations are plain `assert()` checks.

### `test_abstract_mesh`

**File:** `tests/mesh/test_abstract_mesh.cpp`  
**CTest labels:** `mesh;mapping`

- **`test_unified_node_point_contract()`** — Reads a node through `operator[]`, `copy_node` and coordinate-wise `get_data`; expects all three paths to expose the same owning `NodePoint` values.
- **`test_store_contains_and_duplicate()`** — Stores a vertex/signature, verifies canonical internal signature scratch and `contains_vertex`, then inserts the duplicate; expects exactly one persistent record.
- **`test_primary_secondary_and_iteration()`** — Stores vertices with different owners and iterates primary, secondary and combined ranges; expects ownership partitions and counts to be correct.
- **`test_erase_vertex()`** — Tombstones one vertex twice and iterates afterward; expects first erase true, second false, `contains` false and tombstones skipped.
- **`test_node_deletion_updates_public_mapping()`** — Deletes one public node; expects dense public numbering to compact onto stable internal nodes and every incident vertex to be removed while unrelated vertices survive.
- **`test_custom_address_list_type()`** — Instantiates `AbstractMesh` with a minimal custom address-list interface and expects ordinary mesh behavior not to require hidden container methods.

### `test_voronoi_mesh`

**File:** `tests/mesh/test_voronoi_mesh.cpp`  
**CTest labels:** `mesh`

**`main()`** — Creates a 2D mesh with six nodes, a four-plane cuboid and ten explicitly stored vertices. It verifies all signatures/points and unique active count; deletes two nodes and expects public numbering to compact, incident vertices to disappear and unaffected signatures to be remapped with boundary-plane indices preserved; inserts four replacement boundary vertices; filters three selected vertex signatures; inserts two final vertices; and finally checks the exact per-cell combined vertex incidence counts. Each stage expects stored sigma/point data and active counts to match the constructed fixture exactly.

### `test_mesh_neighbour_interface`

**File:** `tests/mesh/neighbours/test_mesh_neighbour_interface.cpp`  
**CTest labels:** `mesh;neighbours;boundary`

**`main()`** — Stores an internal neighbour record containing a duplicate ordinary neighbour and a stable high-end boundary neighbour. A dirty read must return `false` while still loading translated public data; after clearing dirtiness the same read must return `true`. Storing a vertex must mark every incident cell dirty, and `compute_adjecents(0)` must return the two co-generators sharing that vertex. Boundary encoding is expected to become `mesh.size()+plane`, with duplicate multiplicity preserved.

### `test_neighbour_mesh_parallel`

**File:** `tests/mesh/neighbours/test_neighbour_mesh_parallel.cpp`  
**CTest labels:** `mesh;neighbours;threading`

**`main()`** — Creates 128 cells and uses eight threads to publish each cell's three-entry neighbour record (including a duplicate) into a shared `VoronoiMesh`, then reads every cell and compares with the exact expected sorted list. A second eight-thread phase marks all cells dirty, after which every dirty bit must be true. The executable returns failure if any record, duplicate, address publication or dirty update is lost.

### `test_neighbour_forwarding`

**File:** `tests/mesh/neighbours/test_neighbour_forwarding.cpp`  
**CTest labels:** `mesh;highvoronoi;mapping;neighbours`

**`main()`** — Builds a `HighVoronoiMesh`, writes one neighbour record, then accesses it through `HighVoronoiDataMeshView` and `HighVoronoiComputeMesh`. It expects neighbour geometry and dirty mutations to forward to the persistent owner, while compile-time checks ensure these temporary views do *not* expose persistent neighbour addresses or consumer-tracker registration. The persistent owner must still expose historical records, propagate to an external dirty tracker and exclusively own the global neighbour version.

### `test_neighbour_geometry`

**File:** `tests/mesh/neighbours/test_neighbour_geometry.cpp`  
**CTest labels:** `mesh;neighbours;degenerate;boundary`

- **`test_general_position()`** — Stores one ordinary 3D vertex and computes adjacency for its active cell; expects exactly the three co-generators.
- **`test_cartesian_3x3x3_cell()`** — Stores the degenerate center vertex of a Cartesian grid; expects 26 shared-vertex adjacents but only six true positive-area facet neighbours, and verifies the adjacency/neighbour workspaces are reusable/clean.
- **`test_degenerate_boundary_vertex()`** — Stores a degenerate vertex involving a boundary mirror; expects two ordinary axis facets plus the boundary facet.
- **`test_workspace_survives_node_growth()`** — Computes a record containing ordinary and boundary neighbours, grows node storage, and expects old boundary scratch/indexing not to alias the new ordinary range.

### `test_spherical_voronoi_mesh`

**File:** `tests/workflows/spherical/test_spherical_voronoi_mesh.cpp`  
**CTest labels:** `mesh;spherical;incremental`

- **`validate_sphere_mesh()`** — Examines S² public vertices after Euclidean-origin-cell construction; expects finite vertices on the unit sphere, signatures of at least three public generators, equal support chord distances, no closer visible generator and no duplicate public emission of the same center-cell record.
- **`validate_rotation_mesh()`** — Validates antipodal S³/+ output: normalized/canonical-hemisphere user nodes and vertices, nonempty signatures, equal projective support distances, no closer visible generator, correct invisible-antipode ownership and no infinite edge on the internal origin cell.
- **`test_s2_neighbours_keep_internal_origin()`** — After spherical compute, verifies neighbour records start dirty/unpublished, explicit `compute_neighbors` clears them, internal records retain origin node 0, axis cells have origin plus four non-opposite neighbours, while public output exposes exactly the four spherical neighbours.
- **`test_antipodal_neighbour_projection_preserves_multiplicity()`** — Computes neighbours for visible and antipodal representatives; expects origin to remain internally and +/- representatives that project to the same public cell not to be deduplicated.
- **`test_s2_initial_refine_remove()`** — Starts with 20 visible S² nodes, appends four by `refine`, removes those four, and expects counts/reports to return to the original state.
- **`test_s3_antipodal_initial_refine_remove()`** — Starts with origin + 10 visible + 10 antipodes, refines by four visible nodes (and four antipodes), removes one public rotation node and expects its invisible antipode to be removed with it.

### `test_mesh_view`

**File:** `tests/mesh/views/test_mesh_view.cpp`  
**CTest labels:** `mesh;mapping`

**`main()`** — Stores four known vertices in an ordinary mesh, then applies a `SwitchView` that swaps selected public nodes. It verifies node coordinates and every affected sigma in view numbering, inverse mapping, view-side vertex deletion propagating to the underlying stable record, and insertion through the view creating a record visible under both view and underlying signatures. Unrelated original vertices must remain unchanged.

### `test_visible_first_mesh`

**File:** `tests/mesh/views/test_visible_first_mesh.cpp`  
**CTest labels:** `mesh;highvoronoi;mapping`

**`main()`** — Builds HighVoronoi state with visible nodes interleaved in stable insertion order with a reference node, constructs `VisibleFirstMesh`, and expects all visible nodes to form the exact leading prefix in visible insertion order while references follow in stable order. `visible_end()` must be the one-past-visible index and reordered coordinates must still identify the same nodes.

## Search backends and local Voronoi geometry

### `test_search_tree`

**File:** `tests/search/test_search_tree.cpp`  
**CTest labels:** `search`

- **`test_fixed_backend()`** — Runs nearest, k-nearest and in-range queries through the generic backend with ordinary and active boundary nodes; expects the closest mirror to win when active, inactive mirrors to be ignored, deterministic merged ordering, skip predicates in mesh-index numbering and recycled result buffers to retain capacity.
- **`test_nanoflann_native_fields()`** — Verifies nanoflann's native index type is `size_t`, that reusable backend-native buffers are distinct from public `Mesh::Index`, results are translated correctly and capacities survive repeated searches.
- **`test_dynamic_dimension()`** — Builds a dynamic-dimension tree and expects its point scratch to have the mesh runtime dimension and to include active boundary nodes.
- **`test_factories_and_custom_keyword()`** — Adds a user-defined search keyword, verifies it is recognized at compile time and constructs a functioning custom tree through the factory.

### `test_search_tree_voronoi`

**File:** `tests/search/test_search_tree_voronoi.cpp`  
**CTest labels:** `search;mesh`

**`run_search_example()` / `main()` — diagnostic only.** Builds the same real 2D Voronoi mesh with BruteForce and KD backends. For each, it writes mesh/query points into recycled search scratch, performs `nn` and `inrange` first with no active boundary mirror and then with the x=1 mirror active, and prints the expected nearest indices/distances and result lists. The source states the expected behavior (ordinary node 0 at distance 0.35 without the mirror; mirror at 0.05 and mirror+node in range when active), but the current executable does not compare those values programmatically and always returns success unless an exception/crash occurs.

### `test_qr_normal_solver`

**File:** `tests/algorithm/numerics/test_qr_normal_solver_20260807.cpp`  
**CTest labels:** `numerics`

- **`test_normal_double_qr()`** — For every possible omitted generator of one deterministic node/signature fixture, calls the ordinary `QRNormalSolver`. Each solve must succeed; the returned vector must have unit norm and its dot product with every pairwise difference of the remaining defining nodes must be zero within the test tolerance.
- **`test_float128_qr()`** — Repeats exactly the same omitted-generator matrix through `ExtendedQRNormalSolver`, whose internal arithmetic uses the configured extended/Float128 type before converting the final normal back to `double`; every solve must again succeed, be normalized and satisfy all orthogonality checks.
- **Coverage boundary:** despite the broader CMake comment, this source does **not** call the vertex-correction API and does not construct an ill-conditioned case that verifies correction/conditioning fallback selection.

### `test_edge_iterator`

**File:** `tests/algorithm/edges/test_edge_iterator.cpp`  
**CTest labels:** `numerics;edge`

- **`test_general_cellwise_iteration()`** — Stores one general-position vertex, runs `EdgeIterator` for every incident cell in queue and systematic modes, and expects identical edge sequences, exactly `d+1` globally owned edges, unique ownership and no FEI cache use.
- **`test_degenerate_cellwise_shared_fei()`** — Stores a degenerate cube vertex, shares one FEI cache between two iterators, forces queue-mode FEI recomputation, then reuses it in systematic mode; expects identical ordered edges, correct initialized-cell bookkeeping, exactly the eight cube edges and unique cell ownership.
- **`test_degenerate_boundary_last_owner()`** — Replays a boundary degeneracy where the last owner was previously skipped; expects cell 4 to emit the specific full edge `{4,5,69}` with `skip=0` and the expected owned-edge count.

### `test_edge_iterator_cell_analysis`

**File:** `tests/algorithm/edges/test_edge_iterator_cell_analysis.cpp`  
**CTest labels:** `numerics;edge;neighbours;degenerate`

**`main()`** — Builds the 16 generators of a 4D hypercube and treats the center as one 16-fold degenerate vertex. For every cell it runs `EdgeIterator::OnCellEdges`, expects `valid_generators()` to be exactly the four Hamming-distance-1 cube neighbours, expects exactly four emitted edges and every edge to contain the active cell. After all 16 cells, the shared FEI cache must still be empty, proving this analysis mode has no persistent FEI side effect.

### `test_raycaster`

**File:** `tests/algorithm/raycast/test_raycaster.cpp`  
**CTest labels:** `numerics;raycast`

- **`test_classic_raycast_on_general_vertex()`** — Starts from a known general-position Voronoi vertex, enumerates all `d+1` edges and Classic-raycasts each; expects forward finite endpoints with `d+1` support generators, valid Voronoi geometry and distinct adjacent vertices.
- **`test_inrange_raycast_on_degenerate_cube()`** — Starts from a `2^d`-generator cube center, raycasts every edge with InRange, and expects the `2d` direct ±axis neighbour cube centers, each with `2^d` support and valid degeneracy.
- **`test_combined_raycast_on_degenerate_cube()`** — Repeats the degenerate cube traversal with Combined and expects the same complete `2d` direct-neighbour set.
- **`test_combined_raycast_on_general_vertex()`** — Repeats the general-position traversal with Combined and expects all `d+1` valid adjacent vertices.
- **`test_combined_boundary_mirror_prepass()`** — Activates a boundary mirror and raycasts toward it; expects a finite endpoint on the upper strip boundary with the mirror generator merged into sigma.

## ComputeVoronoi construction, range behavior and completeness

### `test_compute_voronoi_validation`

**File:** `tests/algorithm/compute/test_compute_voronoi_validation.cpp`  
**CTest labels:** `compute`

- **`test_general_position_compute()`** — Computes a bounded general-position mesh, expects at least one stored vertex and `verify_mesh` validity, then expects `compare_meshes(mesh,mesh)` to report equality.
- **`test_cubical_grid_compute()`** — Computes a strongly degenerate 4D Cartesian grid with InRange; expects exactly `5^4` bounded Cartesian vertices, geometric consistency and full topological completeness.

### `test_compute_voronoi_singlethread_meshview`

**File:** `tests/algorithm/compute/test_compute_voronoi_singlethread_meshview.cpp`  
**CTest labels:** `compute;mapping`

- **`test_independent_equal_meshes_and_difference_detection()`** — Computes two independent meshes from the same random 3D nodes and one mesh with an added generator; expects the first pair to be individually valid and equal with identical vertex counts, and the altered mesh to remain valid but compare unequal with at least one reported difference.
- **`test_computation_through_reordered_mesh_view()`** — Swaps the first/last five public nodes with `SwitchView`, computes through `ReorderedMeshView`, and expects the underlying persistent mesh to be valid and contain exactly the same vertices/count as direct computation.

### `test_compute_voronoi_parallel_smoke`

**File:** `tests/algorithm/compute/test_compute_voronoi_parallel_smoke.cpp`  
**CTest labels:** `compute;threading`

**`main()`** — Computes the same problem three ways: serial reference, mesh-threaded construction and cast-worker-threaded construction. It expects all three meshes to be geometrically valid, both parallel results to compare equal with the serial reference and with each other, and all three to contain the same number of primary vertices.

### `test_compute_voronoi_progress`

**File:** `tests/test_compute_voronoi_progress.cpp`  
**CTest labels:** `compute;threading;progress`

**`main()`** — Runs the same 40-node 3D compute with captured stdout. Non-verbose serial compute must emit nothing; verbose serial and `MultiThread{3}` mesh-parallel runs must emit a `Mesh compute` progress line and reach `100.0 %`.

### `test_compute_voronoi_cartesian_degeneracy`

**File:** `tests/algorithm/compute/test_compute_voronoi_cartesian_degeneracy.cpp`  
**CTest labels:** `compute;degenerate`

**`main()`** — Computes the exact bounded Cartesian degeneracy fixture with InRange. It expects the analytically known number of bounded vertices to be present, all stored vertices to pass geometric validation and `verify_mesh_complete` to report full edge closure/topological completeness.

### `test_compute_voronoi_range_consistency`

**File:** `tests/algorithm/compute/test_compute_voronoi_range_consistency.cpp`  
**CTest labels:** `compute;mapping`

- **`run_random_matrix_for_node_count()` / `run_random_matrix()`** — For 3D/4D random meshes and several node counts/range fractions, compute a full reference and selected/reordered partial ranges. Every full reference must validate, and the selected cells from partial/range computation must match the same cells from full computation in all three random repetitions.
- **`run_historical_case()`** — Replays the historical “27 old + 3 new” geometry; expects the reordered full reference to validate and the first three selected cells from range computation to match full computation exactly.

### `test_infinite_edge_storage`

**File:** `tests/algorithm/compute/test_infinite_edge_storage_20260814.cpp`  
**CTest labels:** `compute;unbounded`

**`main()`** — Eight threads concurrently attempt to store the same 2D infinite edge `(full support, origin, direction)`. Exactly one call may create a persistent record; iteration must return exactly that one record with identical data. A `ReorderedMeshView` then stores the same edge under view numbering and must deduplicate against the wrapped record; view iteration must expose the correctly remapped public sigma and still exactly one edge.

### `test_edge_hash_completeness`

**File:** `tests/algorithm/compute/test_edge_hash_completeness_20260814.cpp`  
**CTest labels:** `compute;edge`

**`main()`** — Pushes occurrences of edge `a` into `EdgeHashTable`: after one finite endpoint the table must be incomplete, after two complete, and after a third occurrence the insertion must report overfull and completeness must become false. A second table verifies that one finite endpoint plus the `infinite_cell` sentinel counts as complete, while a separate finite edge still needs exactly two finite endpoints.

### `test_compute_voronoi_unbounded_completeness`

**File:** `tests/algorithm/compute/test_compute_voronoi_unbounded_completeness_20260814.cpp`  
**CTest labels:** `compute;unbounded`

**`main()`** — Computes the unbounded Voronoi diagram of three non-collinear 2D generators. `verify_mesh_complete` must report a complete mesh with zero finite edge endpoints and exactly three persistent infinite edges. The test then physically erases one infinite-edge database record and requires the next completeness check to fail, proving infinite rays participate in global edge closure.

## Engine-backed geometry and allocation regressions

### `test_cuboid_mesh_engine`

**File:** `tests/engine/test_cuboid_mesh_engine.cpp`  
**CTest labels:** `engine;mesh`

- **`test_geometry_and_incidence()`** — Constructs a `3×2×3` analytic cuboid engine. It expects 18 ordinary nodes, 4 finite box-center vertices, immutable analytic neighbour records, exact flattened-grid coordinates, exact first/last vertex positions and eight-corner signatures, one primary plus seven secondary incidences per 3D finite vertex, and the expected axis-neighbour sets. It invalidates a computed vertex and expects the empty-sigma tombstone convention plus removal from both primary/secondary incidence while native neighbour topology remains unchanged. The engine explicitly reports that it does not provide a complete infinite-edge set.

### `test_compute_mesh_cuboid`

**File:** `tests/engine/test_compute_mesh_cuboid.cpp`  
**CTest labels:** `engine;mesh`

- **`test_compute_mesh_facade()`** — Wraps a cuboid engine in `ComputeMesh`. It expects all 27 computed nodes, the engine's infinite-edge capability flag, native immutable neighbour reads, eight analytic finite vertices each appearing once as primary and seven times as secondary, and geometric Voronoi consistency over all occurrences. Deleting one vertex through the normal `AbstractMesh::erase_vertex` path must invalidate it in the engine and remove it from mesh iteration without creating dirty persistent-neighbour state.

### `test_hybrid_database_cuboid`

**File:** `tests/engine/test_hybrid_database_cuboid.cpp`  
**CTest labels:** `engine;storage`

- **`test_stored_engine_stored_address_space()`** — Inserts stored vertices before and after registering a `3×3×2` cuboid engine. It expects a single logical address space, translated engine signatures/positions through the normal database API, engine signatures in the shared duplicate hash, duplicate stored insertion to see the engine record, and deletion to route to the engine/tombstone while removing the signature from the shared hash.
- **`test_registration_collision_is_transactional()`** — Creates a stored signature that collides with one engine vertex, then attempts engine registration; expects rejection without consuming logical addresses and without disturbing the pre-existing stored hash entry.

### `test_hotpath_allocations`

**File:** `tests/performance/test_hotpath_allocations.cpp`  
**CTest labels:** `foundation;storage;engine;performance`

- **`test_hvdatabase_read_no_allocations()`** — Warms a stored vertex read and then measures allocations around repeated `HVDataBase::read`; expects zero allocations in the read loop.
- **`test_compute_mesh_read_no_allocations()`** — Warms `ComputeMesh` read/contains on an engine-backed signature and repeats them under allocation counting; expects zero allocations and stable lookup.
- **`test_hybrid_engine_read_no_allocations()`** — Registers a cuboid engine behind `HybridDataBase`, warms an engine-record read and repeats it under allocation counting; expects zero allocations.

## Incremental and HighVoronoi workflows

### `test_high_voronoi_nonperiodic_workflow`

**File:** `tests/workflows/high_voronoi/test_high_voronoi_nonperiodic_workflow.cpp`  
**CTest labels:** `highvoronoi`

**`main()`** runs four non-periodic HighVoronoi scenarios in `[-1,1]^3`.
1. It computes a ten-node ordinary source mesh, verifies representative octant/origin coverage and source completeness, appends ten background visible nodes to a HighVoronoi mesh, imports the source mesh, and expects exactly ten appended nodes, twenty visible nodes and stable-order data view.
2. It appends a `10^3` cuboid engine to another HighVoronoi mesh with one pre-existing positive-octant node; expects all 1000 engine nodes, some analytic grid vertices rejected by the foreign node, accepted-vertex count equal to the data view's active vertices, and stable insertion order.
3. It marks three visible nodes inactive in each scenario and expects the public counts/stable-order data views to reflect the deletions before repair.
4. It runs `ComputeHighVoronoi` on both dirty/incomplete states and expects the final bounded meshes to be geometrically consistent/topologically complete while preserving stable data-view ordering.

### `test_high_voronoi_cuboid_inrange`

**File:** `tests/workflows/high_voronoi/test_high_voronoi_cuboid_inrange.cpp`  
**CTest labels:** `highvoronoi;degenerate`

**`main()`** — Imports a `4^3` exact Cartesian cuboid engine into HighVoronoi and uses the InRange path. It expects all 64 engine nodes to appear, a foreign extra node to invalidate at least one analytic grid vertex, the surviving imported partial engine geometry to validate, and deletion of two non-periodic visible nodes to be recorded correctly.

### `test_high_voronoi_incremental_vs_batched`

**File:** `tests/workflows/high_voronoi/test_high_voronoi_incremental_vs_batched.cpp`  
**CTest labels:** `highvoronoi;incremental`

- **`run_case()`** — Constructs a fresh ordinary `VoronoiMesh` for the supplied point set/boundary, explicitly creates KD search + `InRangeRaycast`, then runs one full single-threaded `ComputeVoronoi`. It requires the compute call not to throw, then runs `verify_mesh()` and `verify_mesh_complete()` and requires both geometric consistency and full edge closure. If completeness fails, it prints an additional cell-local full-edge diagnostic (`<2` or `>2` skip occurrences), but those diagnostic counts are not separate expectations.
- **`main()` fixture matrix** — Executes seven independent full-construction cases: exact `4×4×4` Cartesian control; the historical 1007-node “D” final point set; the same set with only the Cartesian component perturbed; exact and perturbed `2×2×2 + six extras`; and exact/perturbed `4×4×4` grids with two generators removed. Every one of the seven meshes must compute, validate geometrically and be topologically complete.
- **Coverage boundary / stale target name:** the current source contains no `HighVoronoiMesh`, no `RefineVoronoi`/`RemoveVoronoi`, and no incremental-vs-batched comparison. The target name and CMake comment therefore describe an older purpose, not what this file presently protects.

### `test_high_voronoi_compute_view_range_equivalence`

**File:** `tests/workflows/high_voronoi/test_high_voronoi_compute_view_range_equivalence.cpp`  
**CTest labels:** `highvoronoi;mapping`

- **`run_one_case()` / random matrix** — For the same trailing node subset, compare: (A) HighVoronoi stable order with `preferred_front` + short compute range, (B) physically reordered ordinary mesh + short range, and (C) physically reordered ordinary mesh + full compute. Only the selected leading cells are compared and must agree.
- **`main()`** — Runs this matrix in 3D and 4D for `N=30,100,300`, tail fractions 5/10/20%, three random runs each, plus the historical `27 old + 3 new` geometry; the historical case explicitly requires view-partial == reordered-partial == reordered-full.

### `test_high_voronoi_periodic_reference`

**File:** `tests/workflows/high_voronoi/test_high_voronoi_periodic_reference.cpp`  
**CTest labels:** `highvoronoi;periodic`

**`main()`** — Builds a partially periodic 3D HighVoronoi mesh with 30 visible nodes and an independent ordinary reference from an explicit `3×3` x/y tiling. It expects HighVoronoi to retain exactly 30 visible nodes, `VisibleFirstMesh` to expose them as an insertion-ordered prefix, visible cells to be geometrically consistent and topologically complete, the full active internal mesh to be consistent, and the explicit reference mesh to validate. Canonicalized periodic reference vertex classes must be covered by the HighVoronoi visible-cell vertices, and every public HighVoronoi vertex must lie inside the external visible domain while covering the projected reference set.

### `test_high_voronoi_periodic_incremental_reference`

**File:** `tests/workflows/high_voronoi/test_high_voronoi_periodic_incremental_reference.cpp`  
**CTest labels:** `highvoronoi;incremental;periodic`

**`main()`** — Starts from 28 visible nodes in a partially periodic domain, computes periodization and requires invisible reference creation without changing the public count or leaving the state unintegrated. It appends three boundary-near visible nodes, recomputes and compares all visible-cell vertices/counts against a fresh explicit periodic reference; no persisted artificial periodic-boundary vertices may remain and central reference cells must not reach the artificial outer x/y faces. At least one inserted node must acquire an invisible neighbour. It then deletes one visible node, repairs again to 30 visible nodes and repeats the same completeness, boundary-cleanliness, vertex-count and explicit-reference equality checks.

### `test_voronoi_incremental_workflows`

**File:** `tests/workflows/incremental/test_voronoi_incremental_workflows.cpp`  
**CTest labels:** `incremental`

The executable runs a dimension/count matrix in 3D (`N=24,50`) and 4D (`N=18,36`). Shared helpers in `voronoi_incremental_test_tools.hpp` validate every repaired state against a freshly recomputed mesh using `verify_mesh_complete`, `compare_meshes` and exact public neighbour lists (duplicates preserved).

- **`run_single_refine()`** — Append one batch; expect `appended_nodes` to equal the batch and the result to equal fresh recomputation.
- **`run_three_refines()`** — Apply batches of 2, 3 and 4 with a new operation object each time; expect `compute()` to return its own stable-slot-sized AFFECTED vector and every stage to equal a fresh reference.
- **`run_remove_with_recompute()`** — Delete a deterministic public set and run immediate removal closure; expect correct removed count, `recomputed=true` and equality with fresh recomputation.
- **`run_remove_refine_remove_refine()`** — Alternate remove/refine/remove/refine; after each repaired state require complete geometry, exact mesh equality and exact neighbour equality with a fresh reference.
- **`run_deferred_remove_then_refine()`** — Call `RemoveVoronoi::remove()` without closure, verify the surviving stored mesh remains geometrically consistent but incomplete, run an independent refinement while the remover stays alive, require the remover's AFFECTED state not to be modified by refinement, then run delayed remover closure and require the AFFECTED vector to extend only when needed to current stable size and the final state to equal fresh recomputation.

### `test_voronoi_refine_highvoronoi_step2_blocks`

**File:** `tests/workflows/high_voronoi/test_voronoi_refine_highvoronoi_step2_blocks.cpp`  
**CTest labels:** `highvoronoi;incremental`

- **`run_high_compute()` / `main()`** — Starts HighVoronoi with 28 visible nodes, performs the historical step-2 insertion sequence and inspects `ComputeHighVoronoi` reports. It expects the three visible step-2 generators to form the first NEW block, one report block per outer round, and every step-2-added node to occur exactly once across those blocks.
- **`validate_refinement_state()`** — Replays the same node sequence through ordinary Voronoi/refinement inside a large fixed box and compares against one-shot references; expects every generator inside the replay boundary and both the baseline and successive refinement states to remain geometrically consistent/topologically complete.

### `test_voronoi_periodic_refine_edge_neighbours`

**File:** `tests/workflows/high_voronoi/test_voronoi_periodic_refine_edge_neighbours.cpp`  
**CTest labels:** `highvoronoi;incremental;periodic`

**`main()`** — Builds a 27-visible-node periodic control and records the first nonempty periodic reference block. It then constructs ordinary comparison meshes `A` and `B0` with identical fixed boundary geometry and verifies both consistent/complete. After materializing exactly the periodic requests/refinement block into `B1`, it requires `A` and `B1` to have identical node counts, coordinates and boundary, and requires `B1` itself to remain consistent/complete. Diagnostic helpers inspect edge candidates/neighbour propagation when snapshots differ.

## Integration storage, views and generic driver

### `test_integral_databases`

**File:** `tests/integration/test_integral_databases.cpp`  
**CTest labels:** `foundation;storage;integrals`

**`main()`** — Exercises `AreaDatabase` with variable-length, scalar and empty records, including records crossing blocks, overwrite-at-same-address and rejection of length-changing overwrite; values are treated as opaque storage and even nonsensical floating payloads must round-trip. It exercises `IntegralDatabase` with scalar/vector records and equal-length overwrite. It then registers one engine adapter behind `HybridAreaDatabase`/`HybridIntegralDatabase`, expects stored/engine/stored logical address routing, readable engine vectors, later stored records to remain accessible, stored overwrite to work, engine overwrite to be rejected, and area/integral databases to reserve independent engine address ranges.

### `test_integral_data`

**File:** `tests/integration/test_integral_data.cpp`  
**CTest labels:** `mesh;storage;integrals;neighbours`

**`main()`** — Publishes neighbour snapshots with duplicates and aligned volume/area/bulk/interface data, then requires `prepare_cell`/`read_cell` to preserve the exact neighbour ordering and neighbour-major interface layout. It verifies disabled options allocate no backing arrays/databases and read back as zero/empty, while sparse enabled subsets remain writable. It deliberately builds an alignment error to ensure layout validation rejects it. It retargets old integral data to a changed neighbour snapshot and expects duplicate occurrences to match one-to-one, surviving area/interface payloads to retain ordinal alignment and newly introduced interfaces to initialize empty/zero. Finally it publishes engine-backed cell data through a shared adapter and requires the same external layout as stored data.

### `test_integral_view`

**File:** `tests/integration/test_integral_view.cpp`  
**CTest labels:** `mesh;view;storage;integrals;neighbours;incremental`

- **`test_view_order_and_integral_alignment()`** — Starts from five stable ordinary cells, deletes one so public numbering compacts, marks one current cell NEW and one DIRTY OLD, and constructs `VoronoiIntegrationView`. It expects `[NEW][DIRTY OLD][CLEAN OLD]` ordering, correct stable↔view↔wrapped mapping and clean geometry still addressable behind the update prefix. Reading committed cells must preserve cell-wide values, translate/sort neighbour indices into view numbering, retain duplicate multiplicity and move area/interface payloads by exactly the same raw-ordinal permutation; stable boundary encoding must become `view.size()+plane`. DIRTY OLD data must remain readable/aligned before recomputation.

### `test_high_voronoi_integral_view`

**File:** `tests/integration/test_high_voronoi_integral_view.cpp`  
**CTest labels:** `mesh;view;storage;integrals;neighbours;highvoronoi;periodic`

- **`test_high_view_order_projection_and_integral_alignment()`** — Builds four visible HighVoronoi cells with two interleaved stable periodic references. It expects integration geometry to contain all six active nodes but only four persistent integral cells, ordered `[NEW visible][DIRTY OLD visible][CLEAN visible][REFERENCES]`. Reference-only dirtiness must not schedule its visible owner; references must resolve to visible owners for integral reads while keeping distinct shifted Euclidean geometry and raw interface identities. Committed area/interface data must remain ordinally aligned across reference-aware translation. It also creates a worker view, expects only the assigned update slice at the local prefix while retaining the global update snapshot and references after all visible context, and verifies integrand-evaluation points are wrapped back into the external fundamental domain.

### `test_integrator_framework`

**File:** `tests/integration/test_integrator_framework.cpp`  
**CTest labels:** `integrals;driver;framework`

**`main()`** — Uses a fake integral/view and tracing algorithm to execute the generic driver. It expects `IntegrationReport.updated_cells` to equal the update prefix, recomputed interfaces to be counted before algorithm mutation, and the trace to show Julia ordering: for each cell `prepare -> integrate`, then only after the complete first pass `cleanup` for all cells, followed by publication/finalization.

### `test_incremental_minors`

**File:** `tests/integration/test_incremental_minors.cpp`  
**CTest labels:** `integrals;polygon;numerics`

- **`test_dimensions()`** — For several small matrix dimensions, incrementally updates the minor hierarchy and compares the top minor with the absolute determinant computed independently; they must match.
- **`test_layer_reuse()`** — Updates higher-order layers of a known 3×3 system and expects lower recycled layers to remain unchanged while the final top minor equals the known determinant magnitude.

### `test_integrator_pass_dispatch`

**File:** `tests/integration/test_integrator_pass_dispatch.cpp`  
**CTest labels:** `integrals;driver;framework;polygon`

- **`test_update_contract_unchanged()`** — Runs an algorithm exposing the legacy `integrate_cell(Update&)` callback and expects cells 0,1,2 in order and the report to count all three.
- **`test_pass_aware_contract()`** — Runs an algorithm exposing `integrate_cell(Pass&,position)`, writes each cell and reads the already-computed previous cell through the pass; expects Julia cellwise ordering, previous values `100,101` to be visible and unchanged report accounting.

## Integration algorithms, incremental behavior and parallel execution

### `test_monte_carlo_integrator`

**File:** `tests/integration/test_monte_carlo_integrator.cpp`  
**CTest labels:** `integrals;monte-carlo;voronoi;geometry`

- **`run_case()`** — Integrates bounded 3D meshes with Monte Carlo and expects a complete published snapshot for every cell, non-negative volume/area data, positive interface/boundary areas and total cell volume within 10% of the unit cube. `main()` runs both 246-node and 100-node cases.
- **`run_incremental_update_case()`** — Integrates an initial 246-cell mesh, refines by 14 cells to 260, then removes those 14. It snapshots CLEAN records and current neighbour topology at every stage. The view must classify exactly 14 NEW cells on refine plus some DIRTY OLD and some reusable CLEAN cells; integration must update exactly NEW+DIRTY, leave CLEAN records bit-for-bit unchanged, publish complete current neighbour snapshots and finite non-negative geometry, and keep total volume within MC tolerance. On removal there must be no NEW survivor, only affected DIRTY survivors plus reusable CLEAN cells; removing the appended nodes must restore the original neighbour topology. Incremental totals are compared with fresh full Monte-Carlo integrations within stochastic tolerance.

### `test_polygon_integrator`

**File:** `tests/integration/test_polygon_integrator.cpp`  
**CTest labels:** `integrals;polygon;geometry;degenerate`

- **`test_degenerate_cartesian_constant()`** — Computes the `2×2×2` Cartesian unit-cube mesh and integrates `f=1` with Polygon; expects eight NEW cells, each volume `1/8`, six facets of area `1/4`, bulk integral equal to volume, interface integral equal to area, exact total volume 1, and zero updates on an unchanged second pass.
- **`test_degenerate_cartesian_vector_affine()`** — Integrates a two-component function (`1` plus an affine component); expects neighbour-major two-component storage, constant component total exactly 1 and the affine component's analytic unit-cube integral exactly.
- **`test_random_general_position_geometry()`** — Integrates a random bounded mesh without a function; expects every cell published, all volumes/areas non-negative and volume sum 1.
- **`test_unbounded_cells_are_finalized_without_integrator()`** — Builds a five-node unbounded cross with four persistent infinite rays and uses a probe algorithm that writes obvious finite values. Only the bounded centre may enter the concrete integrator; four outer cells must be finalized by the planner with `volume=+inf`, complete aligned arrays and componentwise `NaN` for uncomputed function integrals. Finite centre/outer interfaces must be copied from the bounded reciprocal cell, interfaces containing infinite rays must have area `+inf`, and the unchanged second pass must schedule neither finite nor unbounded cells.

### `test_polygon_integrator_4d`

**File:** `tests/integration/test_polygon_integrator_4d.cpp`  
**CTest labels:** `integrals;polygon;geometry;degenerate`

**`main()`** — Integrates the degenerate 4D `2×2×2×2` Cartesian hypercube with Polygon. It expects all 16 cells updated, each volume `1/16`, each cell to expose eight facets of 3D measure `1/8`, and total volume exactly 1.

### `test_polygon_integrator_incremental`

**File:** `tests/integration/test_polygon_integrator_incremental.cpp`  
**CTest labels:** `integrals;polygon;incremental`

**`main()`** — Integrates a bounded mesh with `f=1`, then refines by two generators and later removes them. Initially every NEW cell must be computed and `bulk=volume`, `interface=area`, total volume 1. Refinement/removal must leave some CLEAN cells/survivors so cleanup reuse is exercised; each update must preserve the exact `f=1` identities and total volume 1, and removal must restore the original public cell count.

### `test_polygon_integrator_convergence_4d`

**File:** `tests/integration/test_polygon_integrator_convergence_4d.cpp`  
**CTest labels:** `integrals;polygon;convergence;slow`

**`main()`** — Runs two 4D convergence studies for the deterministic Polygon integrator: successively refined Cartesian grids and nested random refinement. The integrand includes a constant component plus nonlinear components with known analytic integrals. The constant component must integrate to unit hypercube volume at every tested state; nonlinear and combined errors must decrease across Cartesian refinement with negative log-log slopes, and in the nested-random sequence the final nonlinear/combined errors and fitted slopes must improve relative to the initial state. This test is marked `slow` and receives the extended CTest timeout.

### `test_fast_polygon_cache`

**File:** `tests/integration/test_fast_polygon_cache.cpp`  
**CTest labels:** `integrals;polygon;fast-polygon;storage`

**`main()`** — Exercises the serial facet store and shared parallel facet store directly. A fresh key must miss, then round-trip stored volume/integral payload and update hit/miss/entry statistics; `begin_pass` must invalidate previous geometric values. Copies of parallel worker stores must share backing state, concurrent planned misses followed by duplicate publication must yield one readable first-writer entry, and different shard/hierarchy locations must remain independent. Starting the next pass must clear the shared hierarchy. Finally, deliberately colliding hashes with different full canonical keys must remain distinct: collisions may affect diagnostics/probing but never cache identity.

### `test_fast_polygon_integrator`

**File:** `tests/integration/test_fast_polygon_integrator.cpp`  
**CTest labels:** `integrals;polygon;fast-polygon;geometry`

- **`test_degenerate_cartesian_constant()`** — Same `2×2×2`, `f=1` exact geometry contract as Polygon, but through FastPolygon; expects `V=1/8`, six `A=1/4` facets, matching bulk/interface integrals, total 1 and zero work on an unchanged second pass.
- **`test_degenerate_cartesian_vector_affine()`** — Expects two-component neighbour-major layout and exact constant/affine unit-cube integrals.
- **`test_random_general_position_geometry()`** — Expects complete publication, non-negative geometry and total random-cell volume 1.

### `test_fast_polygon_integrator_4d`

**File:** `tests/integration/test_fast_polygon_integrator_4d.cpp`  
**CTest labels:** `integrals;polygon;fast-polygon;geometry;degenerate`

**`main()`** — Integrates the 16-cell degenerate 4D Cartesian hypercube serially and with parallel FastPolygon. Both must update all cells; the parallel run must actually populate/use the shared recursive facet cache and match the serial result cell/interface-wise. Every parallel cell must have volume `1/16`, eight facets of measure `1/8`, and totals must sum to 1.

### `test_fast_polygon_integrator_incremental`

**File:** `tests/integration/test_fast_polygon_integrator_incremental.cpp`  
**CTest labels:** `integrals;polygon;fast-polygon;incremental`

**`main()`** — Repeats the Polygon refine/remove `f=1` workflow with FastPolygon. It expects complete initial computation, exact bulk/volume and interface/area identities, total volume 1, two appended refine cells, CLEAN context left for reuse after refine and remove, exact identities after each update, restored original cell count and total volume 1 after removal.

### `test_fast_polygon_integrator_convergence_4d`

**File:** `tests/integration/test_fast_polygon_integrator_convergence_4d.cpp`  
**CTest labels:** `integrals;polygon;fast-polygon;convergence;slow`

**`main()`** — Runs the same 4D Cartesian and nested-random nonlinear convergence study as the Polygon convergence test but through FastPolygon. Constant integral must remain exactly unit volume; Cartesian nonlinear/combined errors must decrease monotonically with negative log-log slopes, and nested-random final errors/slopes must improve over the initial state. Marked `slow` with the extended timeout.

### `test_heuristic_integrator`

**File:** `tests/integration/test_heuristic_integrator.cpp`  
**CTest labels:** `integrals;heuristic`

- **`test_current_source()`** — Uses a complete source geometry integral and an empty target; Heuristic must compute every NEW target cell, copy authoritative source volume/area, make the `f=1` interface component equal source area and integrate the Cartesian constant/affine fixture exactly.
- **`test_force_source_update_and_dirty_independence()`** — First requires strict mode to reject incomplete source geometry, then enables forced source update and expects source followed by target computation. After mesh refinement, source and target dirty trackers must propagate independently; forced refresh must recompute exactly dirtied target cells, use refreshed source geometry for all `f=1` bulk/interfaces and clear each integral's own dirty tracker on success.
- **`test_self_source_rejected()`** — Attempts to use the target as its own source and expects immediate rejection.

### `test_heuristic_mc_integrator`

**File:** `tests/integration/test_heuristic_mc_integrator.cpp`  
**CTest labels:** `integrals;heuristic;monte_carlo`

**`main()`** — Runs the combined HeuristicMC algorithm. It expects all NEW cells updated, the internal Monte-Carlo stage forced to geometry-only mode with unnecessary bulk sampling disabled, and complete geometry+function records after heuristic completion. The constant interface component must equal finalized MC area; total MC volume and constant/nonlinear bulk integrals must lie near their unit-cube analytic values. An unchanged second pass must update zero cells; after mesh refinement the integral must become dirty and recompute NEW/DIRTY cells.

### `test_integration_dependency_cycle`

**File:** `tests/integration/test_integration_dependency_cycle.cpp`  
**CTest labels:** `integrals;dependency`

**`main()`** — Constructs source-integral dependencies `A -> B -> A` and invokes the generic integration driver. It expects the thread-local dependency/cycle guard to reject the recursive cycle rather than recursing or publishing partial results.

### `test_heuristic_integrator_convergence_4d`

**File:** `tests/integration/test_heuristic_integrator_convergence_4d.cpp`  
**CTest labels:** `integrals;heuristic;convergence;slow`

**`main()`** — Uses deterministic source geometry plus Heuristic function integration for the same 4D Cartesian and nested-random convergence families. The source constant volume must remain 1, nonlinear and combined Cartesian errors must decrease with negative log-log slopes, and the nested-random final nonlinear/combined errors and slopes must improve over the initial state. Marked `slow` with the extended timeout.

### `test_parallel_integration`

**File:** `tests/integration/test_parallel_integration.cpp`  
**CTest labels:** `integrals;driver;threading;polygon;monte-carlo;incremental`

- **`test_worker_neighbour_presentation_permutation()`** — Builds a master integration view with two dirty cells, creates a worker slice and verifies worker-public reordering changes neighbour indices while preserved raw ordinals move area payloads by the identical permutation.
- **`test_serial_progress_overload()`** — Runs the serial progress overload and expects all NEW cells plus a final 100% progress report.
- **`test_polygon_parallel_and_incremental()`** — Compares serial and parallel Polygon on the same initial and refined meshes; expects identical cell/interface data, identical update sets, 100% progress and equality after CLEAN-context reuse.
- **`test_monte_carlo_parallel_cleanup()`** — Runs parallel MC first pass and serial cleanup across worker boundaries; expects symmetric interface areas, total volume near one and zero work on an unchanged pass.
- **`test_fast_polygon_parallel_and_incremental()`** — Confirms serial FastPolygon does not initialize the shared parallel cache; parallel FastPolygon must equal serial initially and after refine, actually hit/store shared facets, rebuild a fresh hierarchy for the refined pass and do zero work when unchanged.
- **`test_heuristic_parallel_and_forced_source_update()`** — Compares serial/parallel Heuristic, including forced source updates. Parallel target must equal serial; a parallel-capable source inherits parallel execution, while an intentionally serial-only source may update serially without preventing the target from remaining parallel.
- **`test_heuristic_mc_parallel_staged_reference()`** — Compares combined parallel HeuristicMC against the staged reference “parallel MC cleanup then parallel Heuristic”, initially and after refine; expects identical data/update sets, `f=1` interfaces built from finalized MC areas, CLEAN reuse and zero work when unchanged.

### `test_high_voronoi_polygon_parallel`

**File:** `tests/integration/test_high_voronoi_polygon_parallel.cpp`  
**CTest labels:** `integrals;driver;threading;polygon;fast-polygon;highvoronoi;periodic`

**`main()`** — Builds a 12-visible-node partially periodic 3D HighVoronoi mesh and requires periodization to create invisible reference nodes. It integrates the same two-component function four ways: Polygon serial/3-worker parallel and FastPolygon serial/3-worker parallel. Every run must update exactly the visible-cell count; serial/parallel pairs must be identical, Polygon and FastPolygon must agree cell/interface-wise, and the parallel FastPolygon cache must report nonzero entries, stores and hits, proving actual shared-facet reuse in periodic geometry.

## How to use this inventory when adding or debugging code

1. Search this page for the invariant or operation you are changing, not only for the class name. For example, a change to neighbour numbering may be covered in mesh views, incremental workflows and integration views simultaneously.
2. Open the named test function and verify that the condition is actually asserted. A printed diagnostic is not regression protection.
3. If a bug crossed several layers, add the narrowest deterministic component regression **and** keep or add an end-to-end workflow regression when the failure depended on orchestration.
4. For numerical/performance changes, separate correctness from performance: first require consistency/completeness/reference equality, then benchmark or check allocations.
5. When adding a new registered test or materially changing what an existing test proves, update this overview in the same change so the documentation remains an accurate map of the suite.
