cmake_minimum_required(VERSION 3.20)

project(
    HighVoronoi
    VERSION 0.1.0
    LANGUAGES CXX
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(Threads REQUIRED)


# ============================================================================
# 1. Build options
# ============================================================================

option(BUILD_SHARED_LIBS
       "Build HighVoronoi as a shared library"
       OFF)

option(HIGHVORONOI_BUILD_TESTS
       "Build the HighVoronoi test suite"
       ON)

option(HIGHVORONOI_BUILD_EXAMPLES
       "Build HighVoronoi examples"
       ON)

option(HIGHVORONOI_BUILD_BENCHMARKS
       "Build manual HighVoronoi performance benchmarks"
       OFF)

option(HIGHVORONOI_BUILD_DOCS
       "Build the HighVoronoi Doxygen manual and API reference"
       OFF)


# ============================================================================
# 2. Library
# ============================================================================
#
# HighVoronoi is mostly template code. Header files are attached to the target
# only so CMake/IDEs know about the complete maintained include tree. They are
# not compiled independently.
#
# CONFIGURE_DEPENDS makes CMake reconfigure when a header is added or removed.
# This avoids the manually maintained header inventory becoming stale.
#
file(GLOB_RECURSE HIGHVORONOI_HEADERS
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/highvoronoi/*.hpp"
)

add_library(HighVoronoi
    src/highvoronoi.cpp
    ${HIGHVORONOI_HEADERS}
)

add_library(HighVoronoi::HighVoronoi ALIAS HighVoronoi)

target_link_libraries(HighVoronoi
    PUBLIC
        Eigen3::Eigen
        Boost::headers
        Threads::Threads
)

target_include_directories(HighVoronoi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(HighVoronoi
    PUBLIC
        cxx_std_17
)

set_target_properties(HighVoronoi PROPERTIES
    CXX_EXTENSIONS OFF
)

target_compile_definitions(HighVoronoi
    PRIVATE
        HIGHVORONOI_BUILDING_LIBRARY
)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(HighVoronoi
        PUBLIC
            HIGHVORONOI_SHARED_LIBRARY
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(HighVoronoi
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
elseif(MSVC)
    target_compile_options(HighVoronoi
        PRIVATE
            /W4
    )
endif()


# ============================================================================
# Documentation (optional)
# ============================================================================
#
# Doxygen is optional for normal library/test/example builds. Enabling this
# option adds a `docs` target which combines docs/*.md with the current
# in-code Doxygen comments and example source browser.
#
if(HIGHVORONOI_BUILD_DOCS)
    include(cmake/HighVoronoiDocs.cmake)
endif()


# ============================================================================
# 3. Tests
# ============================================================================
#
# Every test is registered through one helper:
#
#   highvoronoi_add_test(target source "label;label")
#
# CTest labels allow focused runs, for example:
#
#   ctest --test-dir build/debug -L foundation --output-on-failure
#   ctest --test-dir build/debug -L compute      --output-on-failure
#   ctest --test-dir build/debug -L highvoronoi --output-on-failure
#   ctest --test-dir build/debug -L periodic    --output-on-failure
#
if(HIGHVORONOI_BUILD_TESTS)
    enable_testing()

    function(highvoronoi_add_test TARGET SOURCE LABELS)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
            message(FATAL_ERROR
                "HighVoronoi test source does not exist: ${SOURCE}")
        endif()

        add_executable(${TARGET} ${SOURCE})
        target_link_libraries(${TARGET}
            PRIVATE
                HighVoronoi::HighVoronoi
        )

        add_test(NAME ${TARGET} COMMAND ${TARGET})
        set_tests_properties(${TARGET}
            PROPERTIES
                LABELS "${LABELS}"
        )

        # Keep a registry so forgotten tests can be detected automatically.
        # Test sources may live in thematic subdirectories below tests/.
        set_property(
            GLOBAL APPEND PROPERTY
            HIGHVORONOI_REGISTERED_TEST_SOURCES
            "${SOURCE}"
        )
    endfunction()


    # ------------------------------------------------------------------------
    # 3.1 Foundation: public smoke tests, low-level storage and synchronization
    # ------------------------------------------------------------------------

    # Umbrella-header / compiled-library smoke test.
    highvoronoi_add_test(
        test_basic
        tests/foundation/test_basic.cpp
        "foundation"
    )

    # Public Level-1/Level-2 convenience API and recommended defaults.
    highvoronoi_add_test(
        test_convenience_api
        tests/public/test_convenience_api.cpp
        "foundation;api;compute;incremental;highvoronoi"
    )

    # Fixed/dynamic point layout and Eigen-facing point representation.
    highvoronoi_add_test(
        test_point_layout
        tests/foundation/test_point_layout.cpp
        "foundation"
    )

    # Empty/read-write/FIFO lock semantics and threading primitives.
    highvoronoi_add_test(
        test_locks
        tests/foundation/test_locks.cpp
        "foundation;threading"
    )

    # Packed non-atomic and atomic bit-vector operations.
    highvoronoi_add_test(
        test_atomic_bit_vector
        tests/foundation/test_atomic_bit_vector.cpp
        "foundation;threading"
    )

    # Append/read behavior of the synchronized per-node address list.
    highvoronoi_add_test(
        test_read_write_list
        tests/storage/test_read_write_list.cpp
        "foundation;storage"
    )

    # Queue/edge hash insertion, lookup, deletion, collision and growth rules.
    highvoronoi_add_test(
        test_hash_tables
        tests/storage/test_hash_tables.cpp
        "foundation;storage"
    )

    # HVDataBase append/read/erase and persistent signature deduplication.
    highvoronoi_add_test(
        test_hvdatabase
        tests/storage/test_hvdatabase.cpp
        "foundation;storage"
    )

    # Append/read and parallel block-growth behavior of the neighbour database.
    highvoronoi_add_test(
        test_neighbour_database_parallel
        tests/storage/neighbours/test_neighbour_database_parallel.cpp
        "foundation;storage;threading;neighbours"
    )

    # Per-consumer dirty propagation, single active addresses, and global epoch.
    highvoronoi_add_test(
        test_neighbour_dirty_tracking
        tests/storage/neighbours/test_neighbour_dirty_tracking.cpp
        "foundation;storage;threading;neighbours"
    )

    # Unified logical routing across stored and compute-engine neighbour records.
    highvoronoi_add_test(
        test_hybrid_neighbour_database
        tests/storage/neighbours/test_hybrid_neighbour_database.cpp
        "foundation;storage;engine;neighbours"
    )

    # One current address per cell plus dirty state on a hybrid neighbour DB.
    highvoronoi_add_test(
        test_neighbour_storage_hybrid
        tests/storage/neighbours/test_neighbour_storage_hybrid.cpp
        "foundation;storage;engine;neighbours"
    )

    # Reversible index-view mappings used by reordered mesh facades.
    highvoronoi_add_test(
        test_hvview
        tests/foundation/test_hvview.cpp
        "foundation;mapping"
    )

    # Parallel-safe progress/timing helper with atomic reporter election.
    highvoronoi_add_test(
        test_progress_meter
        tests/core/test_progress_meter.cpp
        "foundation;threading"
    )

    # ------------------------------------------------------------------------
    # 3.2 Nodes, boundaries and mesh/index views
    # ------------------------------------------------------------------------

    # Stored/computed/hybrid VoronoiNodes and extended mirror-node access.
    highvoronoi_add_test(
        test_voronoi_nodes
        tests/core/voronoi_nodes/test_voronoi_nodes.cpp
        "mesh;nodes"
    )

    # Plane geometry, cuboid boundaries and periodic boundary metadata.
    highvoronoi_add_test(
        test_boundary
        tests/core/test_boundary.cpp
        "mesh;boundary"
    )

    # AbstractMesh public/internal mapping and common vertex-storage behavior.
    highvoronoi_add_test(
        test_abstract_mesh
        tests/mesh/test_abstract_mesh.cpp
        "mesh;mapping"
    )

    # Concrete VoronoiMesh node/vertex storage and iteration.
    highvoronoi_add_test(
        test_voronoi_mesh
        tests/mesh/test_voronoi_mesh.cpp
        "mesh"
    )

    # Public/internal neighbour API, dirty state, duplicates and boundary mapping.
    highvoronoi_add_test(
        test_mesh_neighbour_interface
        tests/mesh/neighbours/test_mesh_neighbour_interface.cpp
        "mesh;neighbours;boundary"
    )

    # Parallel neighbour publication and atomic dirty-state updates on VoronoiMesh.
    highvoronoi_add_test(
        test_neighbour_mesh_parallel
        tests/mesh/neighbours/test_neighbour_mesh_parallel.cpp
        "mesh;neighbours;threading"
    )

    # Neighbour/dirty forwarding through HighVoronoi data and compute wrappers.
    highvoronoi_add_test(
        test_neighbour_forwarding
        tests/mesh/neighbours/test_neighbour_forwarding.cpp
        "mesh;highvoronoi;mapping;neighbours"
    )

    # Adjacency candidates versus true facet-neighbours, including degeneracy.
    highvoronoi_add_test(
        test_neighbour_geometry
        tests/mesh/neighbours/test_neighbour_geometry.cpp
        "mesh;neighbours;degenerate;boundary"
    )

    # Spherical origin-cell projection and antipodal S^(d-1)/+- quotient mesh.
    highvoronoi_add_test(
        test_spherical_voronoi_mesh
        tests/workflows/spherical/test_spherical_voronoi_mesh.cpp
        "mesh;spherical;incremental"
    )

    # ReorderedMeshView public/internal mapping and database forwarding.
    highvoronoi_add_test(
        test_mesh_view
        tests/mesh/views/test_mesh_view.cpp
        "mesh;mapping"
    )

    # HighVoronoi view that places all visible nodes in one exact prefix.
    highvoronoi_add_test(
        test_visible_first_mesh
        tests/mesh/views/test_visible_first_mesh.cpp
        "mesh;highvoronoi;mapping"
    )


    # ------------------------------------------------------------------------
    # 3.3 Search backends and local Voronoi geometry
    # ------------------------------------------------------------------------

    # Generic brute-force/nanoflann nearest-neighbour backend contracts.
    highvoronoi_add_test(
        test_search_tree
        tests/search/test_search_tree.cpp
        "search"
    )

    # Search-tree integration with Voronoi mesh and boundary-extension nodes.
    highvoronoi_add_test(
        test_search_tree_voronoi
        tests/search/test_search_tree_voronoi.cpp
        "search;mesh"
    )

    # QR normal solve, vertex correction and conditioning/fallback behavior.
    highvoronoi_add_test(
        test_qr_normal_solver
        tests/algorithm/numerics/test_qr_normal_solver_20260807.cpp
        "numerics"
    )

    # General and degenerate EdgeIterator topology/orientation semantics.
    highvoronoi_add_test(
        test_edge_iterator
        tests/algorithm/edges/test_edge_iterator.cpp
        "numerics;edge"
    )

    # Cell-local all-edge/valid-generator analysis used by neighbour geometry.
    highvoronoi_add_test(
        test_edge_iterator_cell_analysis
        tests/algorithm/edges/test_edge_iterator_cell_analysis.cpp
        "numerics;edge;neighbours;degenerate"
    )

    # Classic/InRange ray casting, descent and finite endpoint geometry.
    highvoronoi_add_test(
        test_raycaster
        tests/algorithm/raycast/test_raycaster.cpp
        "numerics;raycast"
    )


    # ------------------------------------------------------------------------
    # 3.4 ComputeVoronoi construction and completeness
    # ------------------------------------------------------------------------

    # End-to-end bounded construction plus geometric mesh validation.
    highvoronoi_add_test(
        test_compute_voronoi_validation
        tests/algorithm/compute/test_compute_voronoi_validation.cpp
        "compute"
    )

    # Serial ComputeVoronoi through a reordered MeshView.
    highvoronoi_add_test(
        test_compute_voronoi_singlethread_meshview
        tests/algorithm/compute/test_compute_voronoi_singlethread_meshview.cpp
        "compute;mapping"
    )

    # Parallel construction smoke/regression test.
    highvoronoi_add_test(
        test_compute_voronoi_parallel_smoke
        tests/algorithm/compute/test_compute_voronoi_parallel_smoke.cpp
        "compute;threading"
    )

    # ComputeVoronoi progress-meter behavior, including parallel cell counting.
    highvoronoi_add_test(
        test_compute_voronoi_progress
        tests/test_compute_voronoi_progress.cpp
        "compute;threading;progress"
    )

    # Strongly degenerate Cartesian configuration with InRangeRaycast.
    highvoronoi_add_test(
        test_compute_voronoi_cartesian_degeneracy
        tests/algorithm/compute/test_compute_voronoi_cartesian_degeneracy.cpp
        "compute;degenerate"
    )

    # Full compute versus selected-prefix/range computation consistency.
    highvoronoi_add_test(
        test_compute_voronoi_range_consistency
        tests/algorithm/compute/test_compute_voronoi_range_consistency.cpp
        "compute;mapping"
    )

    # Persistence and iteration of genuinely unbounded Voronoi rays.
    highvoronoi_add_test(
        test_infinite_edge_storage
        tests/algorithm/compute/test_infinite_edge_storage_20260814.cpp
        "compute;unbounded"
    )

    # Global EdgeHash completeness accounting and overfull detection.
    highvoronoi_add_test(
        test_edge_hash_completeness
        tests/algorithm/compute/test_edge_hash_completeness_20260814.cpp
        "compute;edge"
    )

    # Completeness of unbounded construction including infinite-edge closure.
    highvoronoi_add_test(
        test_compute_voronoi_unbounded_completeness
        tests/algorithm/compute/test_compute_voronoi_unbounded_completeness_20260814.cpp
        "compute;unbounded"
    )


    # ------------------------------------------------------------------------
    # 3.5 Computed/hybrid mesh engines
    # ------------------------------------------------------------------------

    # CuboidMeshEngine node generation and index layout.
    highvoronoi_add_test(
        test_cuboid_mesh_engine
        tests/engine/test_cuboid_mesh_engine.cpp
        "engine;mesh"
    )

    # ComputeMesh backed by the cuboid engine.
    highvoronoi_add_test(
        test_compute_mesh_cuboid
        tests/engine/test_compute_mesh_cuboid.cpp
        "engine;mesh"
    )

    # Hybrid database behavior for computed cuboid geometry.
    highvoronoi_add_test(
        test_hybrid_database_cuboid
        tests/engine/test_hybrid_database_cuboid.cpp
        "engine;storage"
    )

    # Allocation regression for caller-owned database/engine read buffers.
    highvoronoi_add_test(
        test_hotpath_allocations
        tests/performance/test_hotpath_allocations.cpp
        "foundation;storage;engine;performance"
    )


    # ------------------------------------------------------------------------
    # 3.6 Incremental and HighVoronoi workflows
    # ------------------------------------------------------------------------

    # Basic non-periodic ComputeHighVoronoi workflow and visible output.
    highvoronoi_add_test(
        test_high_voronoi_nonperiodic_workflow
        tests/workflows/high_voronoi/test_high_voronoi_nonperiodic_workflow.cpp
        "highvoronoi"
    )

    # HighVoronoi cuboid construction with the degenerate InRange path.
    highvoronoi_add_test(
        test_high_voronoi_cuboid_inrange
        tests/workflows/high_voronoi/test_high_voronoi_cuboid_inrange.cpp
        "highvoronoi;degenerate"
    )

    # Incremental HighVoronoi insertion versus one batched construction.
    highvoronoi_add_test(
        test_high_voronoi_incremental_vs_batched
        tests/workflows/high_voronoi/test_high_voronoi_incremental_vs_batched.cpp
        "highvoronoi;incremental"
    )

    # Compute-view prefix/range equivalence in HighVoronoi internal numbering.
    highvoronoi_add_test(
        test_high_voronoi_compute_view_range_equivalence
        tests/workflows/high_voronoi/test_high_voronoi_compute_view_range_equivalence.cpp
        "highvoronoi;mapping"
    )

    # Full periodic HighVoronoi geometry versus an explicit 3x3 reference tiling.
    highvoronoi_add_test(
        test_high_voronoi_periodic_reference
        tests/workflows/high_voronoi/test_high_voronoi_periodic_reference.cpp
        "highvoronoi;periodic"
    )

    # Incremental periodic HighVoronoi result versus explicit periodic reference.
    highvoronoi_add_test(
        test_high_voronoi_periodic_incremental_reference
        tests/workflows/high_voronoi/test_high_voronoi_periodic_incremental_reference.cpp
        "highvoronoi;incremental;periodic"
    )

    # Generic refine/remove incremental workflows on ordinary Voronoi meshes.
    highvoronoi_add_test(
        test_voronoi_incremental_workflows
        tests/workflows/incremental/test_voronoi_incremental_workflows.cpp
        "incremental"
    )

    # Regression for HighVoronoi refinement of later insertion blocks.
    highvoronoi_add_test(
        test_voronoi_refine_highvoronoi_step2_blocks
        tests/workflows/high_voronoi/test_voronoi_refine_highvoronoi_step2_blocks.cpp
        "highvoronoi;incremental"
    )

    # Periodic refinement: edge-neighbour/reference-node propagation.
    highvoronoi_add_test(
        test_voronoi_periodic_refine_edge_neighbours
        tests/workflows/high_voronoi/test_voronoi_periodic_refine_edge_neighbours.cpp
        "highvoronoi;incremental;periodic"
    )

    highvoronoi_add_test(
        test_integral_databases
        tests/integration/test_integral_databases.cpp
        "foundation;storage;integrals"
    )

    highvoronoi_add_test(
        test_integral_data
        tests/integration/test_integral_data.cpp
        "mesh;storage;integrals;neighbours"
    )

    highvoronoi_add_test(
        test_integral_view
        tests/integration/test_integral_view.cpp
        "mesh;view;storage;integrals;neighbours;incremental"
    )
    highvoronoi_add_test(
        test_high_voronoi_integral_view
        tests/integration/test_high_voronoi_integral_view.cpp
        "mesh;view;storage;integrals;neighbours;highvoronoi;periodic"
    )
    highvoronoi_add_test(
        test_integrator_framework
        tests/integration/test_integrator_framework.cpp
        "integrals;driver;framework"
    )


    highvoronoi_add_test(
        test_monte_carlo_integrator
        tests/integration/test_monte_carlo_integrator.cpp
        "integrals;monte-carlo;voronoi;geometry"
    )

    # Recycled k-minor hierarchy used by the deterministic polygon integrator.
    highvoronoi_add_test(
        test_incremental_minors
        tests/integration/test_incremental_minors.cpp
        "integrals;polygon;numerics"
    )

    # Backward-compatible first-pass dispatch plus optional pass-aware algorithm hook.
    highvoronoi_add_test(
        test_integrator_pass_dispatch
        tests/integration/test_integrator_pass_dispatch.cpp
        "integrals;driver;framework;polygon"
    )

    # Julia-faithful polygon recursion: degenerate 3D, affine/vector and random mesh.
    highvoronoi_add_test(
        test_polygon_integrator
        tests/integration/test_polygon_integrator.cpp
        "integrals;polygon;geometry;degenerate"
    )

    # Degenerate 4D Cartesian regression for recursive minors/face descent.
    highvoronoi_add_test(
        test_polygon_integrator_4d
        tests/integration/test_polygon_integrator_4d.cpp
        "integrals;polygon;geometry;degenerate"
    )

    # Refine/remove regression including CLEAN-neighbour cleanup reuse.
    highvoronoi_add_test(
        test_polygon_integrator_incremental
        tests/integration/test_polygon_integrator_incremental.cpp
        "integrals;polygon;incremental"
    )

    # ------------------------------------------------------------------------
    # 3.7 Polygon / FastPolygon / Heuristic integration regressions
    # ------------------------------------------------------------------------

    # 4D nonlinear convergence on Cartesian and nested-random refinement.
    highvoronoi_add_test(
        test_polygon_integrator_convergence_4d
        tests/integration/test_polygon_integrator_convergence_4d.cpp
        "integrals;polygon;convergence;slow"
    )

    # Recursive FastPolygon facet-cache behavior.
    highvoronoi_add_test(
        test_fast_polygon_cache
        tests/integration/test_fast_polygon_cache.cpp
        "integrals;polygon;fast-polygon;storage"
    )

    # Serial FastPolygon integration on ordinary Voronoi geometry.
    highvoronoi_add_test(
        test_fast_polygon_integrator
        tests/integration/test_fast_polygon_integrator.cpp
        "integrals;polygon;fast-polygon;geometry"
    )

    # Degenerate 4D FastPolygon recursion/cache regression.
    highvoronoi_add_test(
        test_fast_polygon_integrator_4d
        tests/integration/test_fast_polygon_integrator_4d.cpp
        "integrals;polygon;fast-polygon;geometry;degenerate"
    )

    # FastPolygon refine/remove regression with persistent per-pass facet reuse.
    highvoronoi_add_test(
        test_fast_polygon_integrator_incremental
        tests/integration/test_fast_polygon_integrator_incremental.cpp
        "integrals;polygon;fast-polygon;incremental"
    )

    # Heavy 4D FastPolygon nonlinear convergence regression.
    highvoronoi_add_test(
        test_fast_polygon_integrator_convergence_4d
        tests/integration/test_fast_polygon_integrator_convergence_4d.cpp
        "integrals;polygon;fast-polygon;convergence;slow"
    )

    # Heuristic function integration from an existing geometry source integral.
    highvoronoi_add_test(
        test_heuristic_integrator
        tests/integration/test_heuristic_integrator.cpp
        "integrals;heuristic"
    )

    # Cellwise Monte-Carlo geometry combined with heuristic function integration.
    highvoronoi_add_test(
        test_heuristic_mc_integrator
        tests/integration/test_heuristic_mc_integrator.cpp
        "integrals;heuristic;monte_carlo"
    )

    # Reject cyclic source-integral dependency graphs.
    highvoronoi_add_test(
        test_integration_dependency_cycle
        tests/integration/test_integration_dependency_cycle.cpp
        "integrals;dependency"
    )

    # Heavy 4D Heuristic source-update and nonlinear convergence regression.
    highvoronoi_add_test(
        test_heuristic_integrator_convergence_4d
        tests/integration/test_heuristic_integrator_convergence_4d.cpp
        "integrals;heuristic;convergence;slow"
    )

    # Ordinary Voronoi integration: contiguous worker-view partitioning,
    # Polygon serial/parallel equivalence, MC parallel first pass + serial
    # cleanup, progress reporting, refine regression, and FastPolygon guard.
    highvoronoi_add_test(
        test_parallel_integration
        tests/integration/test_parallel_integration.cpp
        "integrals;driver;threading;polygon;monte-carlo;incremental"
    )

    # Periodic HighVoronoi Polygon/FastPolygon integration: serial/parallel
    # equivalence plus cross-algorithm agreement.
    highvoronoi_add_test(
        test_high_voronoi_polygon_parallel
        tests/integration/test_high_voronoi_polygon_parallel.cpp
        "integrals;driver;threading;polygon;fast-polygon;highvoronoi;periodic"
    )

    # Give the deliberately heavy 4D convergence tests enough room under CTest.
    set_tests_properties(
        test_polygon_integrator_convergence_4d
        test_fast_polygon_integrator_convergence_4d
        test_heuristic_integrator_convergence_4d
        PROPERTIES TIMEOUT 1800
    )


    # ------------------------------------------------------------------------
    # 3.8 Detect test source files that were forgotten in CMake
    # ------------------------------------------------------------------------

    file(GLOB_RECURSE HIGHVORONOI_TEST_SOURCES
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp"
    )

    get_property(
        HIGHVORONOI_REGISTERED_TEST_SOURCES
        GLOBAL PROPERTY HIGHVORONOI_REGISTERED_TEST_SOURCES
    )

    foreach(TEST_SOURCE IN LISTS HIGHVORONOI_TEST_SOURCES)
        list(FIND HIGHVORONOI_REGISTERED_TEST_SOURCES
             "${TEST_SOURCE}"
             REGISTERED_INDEX)
        if(REGISTERED_INDEX EQUAL -1)
            message(WARNING
                "Unregistered HighVoronoi test source: ${TEST_SOURCE}")
        endif()
    endforeach()
endif()


# ============================================================================
# 4. Examples
# ============================================================================

if(HIGHVORONOI_BUILD_EXAMPLES)
    function(highvoronoi_add_example TARGET SOURCE)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}")
            message(FATAL_ERROR
                "HighVoronoi example source does not exist: ${SOURCE}")
        endif()

        add_executable(${TARGET} ${SOURCE})
        target_link_libraries(${TARGET}
            PRIVATE
                HighVoronoi::HighVoronoi
        )
    endfunction()

    # Reproducible bounded 4D general-position construction.
    highvoronoi_add_example(
        highvoronoi_example_general_position
        examples/general_position.cpp
    )

    # Degenerate 4D Cartesian grid with exact expected vertex count.
    highvoronoi_add_example(
        highvoronoi_example_cubic_grid
        examples/cubic_grid.cpp
    )

    # Ordinary VoronoiMesh: initial compute, incremental refine and remove.
    highvoronoi_add_example(
        example_voronoi_refine_remove
        examples/incremental/voronoi_refine_remove.cpp
    )

    # Non-periodic HighVoronoiMesh: visible refine and remove workflow.
    highvoronoi_add_example(
        example_high_voronoi_refine_remove
        examples/incremental/high_voronoi_refine_remove.cpp
    )

    # Periodic HighVoronoiMesh: incremental visible insertion and closure.
    highvoronoi_add_example(
        example_periodic_high_voronoi_refine
        examples/incremental/periodic_high_voronoi_refine.cpp
    )

    # Parallel ray casting inside one mesh branch.
    highvoronoi_add_example(
        example_parallel_cast
        examples/parallel/parallel_cast.cpp
    )

    # Parallel processing of independent mesh branches/cells.
    highvoronoi_add_example(
        example_parallel_mesh
        examples/parallel/parallel_mesh.cpp
    )

    # Combined mesh-level and cast-level parallelism.
    highvoronoi_add_example(
        example_parallel_combined
        examples/parallel/parallel_combined.cpp
    )

    # S^2 construction with incremental refine/remove.
    highvoronoi_add_example(
        example_spherical_refine_remove
        examples/spherical/spherical_refine_remove.cpp
    )

    # Polygon reference versus parallel FastPolygon volume/interface integration.
    highvoronoi_add_example(
        example_fast_polygon_integration
        examples/integration/fast_polygon.cpp
    )
endif()

# ============================================================================
# 5. Manual performance benchmarks
# ============================================================================
#
# Benchmarks are deliberately excluded from CTest and disabled by default.
# Configure a Release tree explicitly, for example:
#
#   cmake -S . -B build/release -G Ninja \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DHIGHVORONOI_BUILD_BENCHMARKS=ON \
#       -DHIGHVORONOI_BUILD_TESTS=OFF \
#       -DHIGHVORONOI_BUILD_EXAMPLES=OFF
#
if(HIGHVORONOI_BUILD_BENCHMARKS)
    add_executable(
        benchmark_compute_voronoi_5d
        benchmarks/benchmark_compute_voronoi_5d.cpp
    )

    target_link_libraries(benchmark_compute_voronoi_5d
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    if(CMAKE_BUILD_TYPE AND
       NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
        message(WARNING
            "HighVoronoi benchmarks are enabled with CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'. "
            "Use Release for comparable timings.")
    endif()
endif()

