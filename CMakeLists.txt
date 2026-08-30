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
        tests/test_basic.cpp
        "foundation"
    )

    # Fixed/dynamic point layout and Eigen-facing point representation.
    highvoronoi_add_test(
        test_point_layout
        tests/test_point_layout.cpp
        "foundation"
    )

    # Empty/read-write/FIFO lock semantics and threading primitives.
    highvoronoi_add_test(
        test_locks
        tests/test_locks.cpp
        "foundation;threading"
    )

    # Packed non-atomic and atomic bit-vector operations.
    highvoronoi_add_test(
        test_atomic_bit_vector
        tests/test_atomic_bit_vector.cpp
        "foundation;threading"
    )

    # Append/read behavior of the synchronized per-node address list.
    highvoronoi_add_test(
        test_read_write_list
        tests/test_read_write_list.cpp
        "foundation;storage"
    )

    # Queue/edge hash insertion, lookup, deletion, collision and growth rules.
    highvoronoi_add_test(
        test_hash_tables
        tests/test_hash_tables.cpp
        "foundation;storage"
    )

    # HVDataBase append/read/erase and persistent signature deduplication.
    highvoronoi_add_test(
        test_hvdatabase
        tests/test_hvdatabase.cpp
        "foundation;storage"
    )

    # Reversible index-view mappings used by reordered mesh facades.
    highvoronoi_add_test(
        test_hvview
        tests/test_hvview.cpp
        "foundation;mapping"
    )


    # ------------------------------------------------------------------------
    # 3.2 Nodes, boundaries and mesh/index views
    # ------------------------------------------------------------------------

    # Stored/computed/hybrid VoronoiNodes and extended mirror-node access.
    highvoronoi_add_test(
        test_voronoi_nodes
        tests/test_voronoi_nodes.cpp
        "mesh;nodes"
    )

    # Plane geometry, cuboid boundaries and periodic boundary metadata.
    highvoronoi_add_test(
        test_boundary
        tests/test_boundary.cpp
        "mesh;boundary"
    )

    # AbstractMesh public/internal mapping and common vertex-storage behavior.
    highvoronoi_add_test(
        test_abstract_mesh
        tests/test_abstract_mesh.cpp
        "mesh;mapping"
    )

    # Concrete VoronoiMesh node/vertex storage and iteration.
    highvoronoi_add_test(
        test_voronoi_mesh
        tests/test_voronoi_mesh.cpp
        "mesh"
    )

    # ReorderedMeshView public/internal mapping and database forwarding.
    highvoronoi_add_test(
        test_mesh_view
        tests/test_mesh_view.cpp
        "mesh;mapping"
    )

    # HighVoronoi view that places all visible nodes in one exact prefix.
    highvoronoi_add_test(
        test_visible_first_mesh
        tests/test_visible_first_mesh.cpp
        "mesh;highvoronoi;mapping"
    )


    # ------------------------------------------------------------------------
    # 3.3 Search backends and local Voronoi geometry
    # ------------------------------------------------------------------------

    # Generic brute-force/nanoflann nearest-neighbour backend contracts.
    highvoronoi_add_test(
        test_search_tree
        tests/test_search_tree.cpp
        "search"
    )

    # Search-tree integration with Voronoi mesh and boundary-extension nodes.
    highvoronoi_add_test(
        test_search_tree_voronoi
        tests/test_search_tree_voronoi.cpp
        "search;mesh"
    )

    # QR normal solve, vertex correction and conditioning/fallback behavior.
    highvoronoi_add_test(
        test_qr_normal_solver
        tests/test_qr_normal_solver_20260807.cpp
        "numerics"
    )

    # General and degenerate EdgeIterator topology/orientation semantics.
    highvoronoi_add_test(
        test_edge_iterator
        tests/test_edge_iterator.cpp
        "numerics;edge"
    )

    # Classic/InRange ray casting, descent and finite endpoint geometry.
    highvoronoi_add_test(
        test_raycaster
        tests/test_raycaster.cpp
        "numerics;raycast"
    )


    # ------------------------------------------------------------------------
    # 3.4 ComputeVoronoi construction and completeness
    # ------------------------------------------------------------------------

    # End-to-end bounded construction plus geometric mesh validation.
    highvoronoi_add_test(
        test_compute_voronoi_validation
        tests/test_compute_voronoi_validation.cpp
        "compute"
    )

    # Serial ComputeVoronoi through a reordered MeshView.
    highvoronoi_add_test(
        test_compute_voronoi_singlethread_meshview
        tests/test_compute_voronoi_singlethread_meshview.cpp
        "compute;mapping"
    )

    # Parallel construction smoke/regression test.
    highvoronoi_add_test(
        test_compute_voronoi_parallel_smoke
        tests/test_compute_voronoi_parallel_smoke.cpp
        "compute;threading"
    )

    # Strongly degenerate Cartesian configuration with InRangeRaycast.
    highvoronoi_add_test(
        test_compute_voronoi_cartesian_degeneracy
        tests/test_compute_voronoi_cartesian_degeneracy.cpp
        "compute;degenerate"
    )

    # Full compute versus selected-prefix/range computation consistency.
    highvoronoi_add_test(
        test_compute_voronoi_range_consistency
        tests/test_compute_voronoi_range_consistency.cpp
        "compute;mapping"
    )

    # Persistence and iteration of genuinely unbounded Voronoi rays.
    highvoronoi_add_test(
        test_infinite_edge_storage
        tests/test_infinite_edge_storage_20260814.cpp
        "compute;unbounded"
    )

    # Global EdgeHash completeness accounting and overfull detection.
    highvoronoi_add_test(
        test_edge_hash_completeness
        tests/test_edge_hash_completeness_20260814.cpp
        "compute;edge"
    )

    # Completeness of unbounded construction including infinite-edge closure.
    highvoronoi_add_test(
        test_compute_voronoi_unbounded_completeness
        tests/test_compute_voronoi_unbounded_completeness_20260814.cpp
        "compute;unbounded"
    )


    # ------------------------------------------------------------------------
    # 3.5 Computed/hybrid mesh engines
    # ------------------------------------------------------------------------

    # CuboidMeshEngine node generation and index layout.
    highvoronoi_add_test(
        test_cuboid_mesh_engine
        tests/test_cuboid_mesh_engine.cpp
        "engine;mesh"
    )

    # ComputeMesh backed by the cuboid engine.
    highvoronoi_add_test(
        test_compute_mesh_cuboid
        tests/test_compute_mesh_cuboid.cpp
        "engine;mesh"
    )

    # Hybrid database behavior for computed cuboid geometry.
    highvoronoi_add_test(
        test_hybrid_database_cuboid
        tests/test_hybrid_database_cuboid.cpp
        "engine;storage"
    )


    # ------------------------------------------------------------------------
    # 3.6 Incremental and HighVoronoi workflows
    # ------------------------------------------------------------------------

    # Basic non-periodic ComputeHighVoronoi workflow and visible output.
    highvoronoi_add_test(
        test_high_voronoi_nonperiodic_workflow
        tests/test_high_voronoi_nonperiodic_workflow.cpp
        "highvoronoi"
    )

    # HighVoronoi cuboid construction with the degenerate InRange path.
    highvoronoi_add_test(
        test_high_voronoi_cuboid_inrange
        tests/test_high_voronoi_cuboid_inrange.cpp
        "highvoronoi;degenerate"
    )

    # Incremental HighVoronoi insertion versus one batched construction.
    highvoronoi_add_test(
        test_high_voronoi_incremental_vs_batched
        tests/test_high_voronoi_incremental_vs_batched.cpp
        "highvoronoi;incremental"
    )

    # Compute-view prefix/range equivalence in HighVoronoi internal numbering.
    highvoronoi_add_test(
        test_high_voronoi_compute_view_range_equivalence
        tests/test_high_voronoi_compute_view_range_equivalence.cpp
        "highvoronoi;mapping"
    )

    # Full periodic HighVoronoi geometry versus an explicit 3x3 reference tiling.
    highvoronoi_add_test(
        test_high_voronoi_periodic_reference
        tests/test_high_voronoi_periodic_reference.cpp
        "highvoronoi;periodic"
    )

    # Incremental periodic HighVoronoi result versus explicit periodic reference.
    highvoronoi_add_test(
        test_high_voronoi_periodic_incremental_reference
        tests/test_high_voronoi_periodic_incremental_reference.cpp
        "highvoronoi;incremental;periodic"
    )

    # Generic refine/remove incremental workflows on ordinary Voronoi meshes.
    highvoronoi_add_test(
        test_voronoi_incremental_workflows
        tests/test_voronoi_incremental_workflows.cpp
        "incremental"
    )

    # Regression for HighVoronoi refinement of later insertion blocks.
    highvoronoi_add_test(
        test_voronoi_refine_highvoronoi_step2_blocks
        tests/test_voronoi_refine_highvoronoi_step2_blocks.cpp
        "highvoronoi;incremental"
    )

    # Periodic refinement: edge-neighbour/reference-node propagation.
    highvoronoi_add_test(
        test_voronoi_periodic_refine_edge_neighbours
        tests/test_voronoi_periodic_refine_edge_neighbours.cpp
        "highvoronoi;incremental;periodic"
    )


    # ------------------------------------------------------------------------
    # 3.7 Detect test source files that were forgotten in CMake
    # ------------------------------------------------------------------------

    file(GLOB HIGHVORONOI_TEST_SOURCES
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
endif()
