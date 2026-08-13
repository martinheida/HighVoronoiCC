
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
# Optionen
# ============================================================================
#
# BUILD_SHARED_LIBS:
#   OFF -> statische Bibliothek (unter Linux typischerweise libHighVoronoi.a)
#   ON  -> dynamische Bibliothek (unter Linux .so, unter Windows .dll)
#
# Aufruf z.B.:
#   cmake --preset debug
#   cmake -S . -B build/shared -DBUILD_SHARED_LIBS=ON
#
option(BUILD_SHARED_LIBS "Build HighVoronoi as a shared library" OFF)

option(HIGHVORONOI_BUILD_TESTS
       "Build HighVoronoi tests"
       ON)

option(HIGHVORONOI_BUILD_EXAMPLES
       "Build HighVoronoi examples"
       ON)

option(HIGHVORONOI_BUILD_BENCHMARKS
       "Build HighVoronoi benchmarks"
       OFF)


# ============================================================================
# 1. Öffentliche Header
# ============================================================================
#
# Alles unter include/highvoronoi/ gehört prinzipiell zur installierbaren
# Header-Struktur der Bibliothek.
#
# Der Benutzer soll aber NUR diesen Header kennen müssen:
#
#     #include <highvoronoi/highvoronoi.hpp>
#
# highvoronoi.hpp ist der "umbrella header" und bindet die übrigen benötigten
# Template-/Modul-Header ein.
#
# WICHTIG:
# Template-Definitionen müssen normalerweise im Header sichtbar sein.
# Neue Template-Dateien daher unter include/highvoronoi/... ablegen und hier
# registrieren. Sie werden NICHT separat kompiliert; die Liste hilft CMake/IDE
# aber dabei, die Dateien dem Library-Target zuzuordnen.
#
set(HIGHVORONOI_PUBLIC_HEADERS
    include/highvoronoi/highvoronoi.hpp

    # Geometrie
    include/highvoronoi/geometry/point.hpp
    include/highvoronoi/geometry/voronoi_nodes.hpp
    include/highvoronoi/geometry/boundary.hpp
    include/highvoronoi/geometry/abstract_mesh.hpp
    include/highvoronoi/geometry/voronoi_mesh.hpp
    include/highvoronoi/geometry/serial_mesh.hpp
    include/highvoronoi/geometry/mesh_view.hpp
    include/highvoronoi/geometry/abstract_search_tree_crtp.hpp
    include/highvoronoi/geometry/brute_force_search_tree_crtp.hpp
    include/highvoronoi/geometry/nanoflann_search_tree_crtp.hpp
    include/highvoronoi/geometry/search_tree_factory_crtp.hpp
    include/highvoronoi/geometry/normal_solver.hpp
    include/highvoronoi/geometry/edge_iterator.hpp
    include/highvoronoi/geometry/raycaster.hpp
    include/highvoronoi/geometry/voronoi_worker.hpp
    include/highvoronoi/geometry/systematic_voronoi.hpp
    include/highvoronoi/geometry/compute_voronoi.hpp
    include/highvoronoi/geometry/mesh_validation.hpp
    
    # Beispiel für einen Template-Header:
    include/highvoronoi/templates/example.hpp

    # internal classes
    include/highvoronoi/detail/float.hpp
    include/highvoronoi/detail/locks.hpp
    include/highvoronoi/detail/fnv_hash.hpp
    include/highvoronoi/detail/hash_functions.hpp
    include/highvoronoi/detail/hash_generators.hpp
    include/highvoronoi/detail/queue_hash_table.hpp
    include/highvoronoi/detail/edge_hash_table.hpp
    include/highvoronoi/detail/queue_hash_table_2.hpp
    include/highvoronoi/detail/hash_table_containers.hpp
    include/highvoronoi/detail/hash_types.hpp
    include/highvoronoi/detail/hvdatabase.hpp
    include/highvoronoi/detail/hvview.hpp
    include/highvoronoi/detail/read_write_list.hpp
    include/highvoronoi/detail/nanoflann.hpp

    # public service for internal classes
    include/highvoronoi/parameters.hpp

    # Später z.B.:
    # include/highvoronoi/geometry/point.hpp
    # include/highvoronoi/geometry/vector.hpp
    # include/highvoronoi/delaunay/delaunay.hpp
    # include/highvoronoi/voronoi/voronoi.hpp
)


# ============================================================================
# 2. Kompilierte C++-Quelldateien
# ============================================================================
#
# Jede .cpp-Datei, die tatsächlich kompiliert werden soll, kommt nach src/
# und wird hier eingetragen.
#
# Beispiel:
#
#   src/highvoronoi.cpp
#   src/geometry/intersection.cpp
#   src/delaunay/search.cpp
#
set(HIGHVORONOI_SOURCES
    src/highvoronoi.cpp

    # Später z.B.:
    # src/geometry/intersection.cpp
    # src/delaunay/search.cpp
)



# ============================================================================
# 4. Bibliotheks-Target
# ============================================================================
#
# Weil STATIC/SHARED hier absichtlich NICHT angegeben ist, entscheidet
# BUILD_SHARED_LIBS:
#
#   -DBUILD_SHARED_LIBS=OFF  -> static
#   -DBUILD_SHARED_LIBS=ON   -> shared
#
add_library(HighVoronoi
    ${HIGHVORONOI_SOURCES}
    ${HIGHVORONOI_PUBLIC_HEADERS}
)

# Alias mit Namespace. Andere CMake-Projekte können später sauber schreiben:
#
#   target_link_libraries(my_program PRIVATE HighVoronoi::HighVoronoi)
#
add_library(HighVoronoi::HighVoronoi ALIAS HighVoronoi)

target_link_libraries(HighVoronoi
    PUBLIC
        Eigen3::Eigen
        Boost::headers
        Threads::Threads
)

# ============================================================================
# 5. Include-Pfade
# ============================================================================
#
# PUBLIC bedeutet:
#   - HighVoronoi selbst darf include/ verwenden.
#   - Programme, die gegen HighVoronoi linken, bekommen include/ ebenfalls.
#
# Dadurch funktioniert:
#
#   #include <highvoronoi/highvoronoi.hpp>
#
target_include_directories(HighVoronoi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


# ============================================================================
# 6. C++-Standard
# ============================================================================
#
# HighVoronoi benötigt mindestens C++17.
# PUBLIC propagiert diese Anforderung auch an abhängige Targets.
#
target_compile_features(HighVoronoi
    PUBLIC
        cxx_std_17
)

set_target_properties(HighVoronoi PROPERTIES
    CXX_EXTENSIONS OFF
)


# ============================================================================
# 7. Export-Makro für Shared Libraries / DLLs
# ============================================================================
#
# Beim Kompilieren der Bibliothek wird HIGHVORONOI_BUILDING_LIBRARY definiert.
# highvoronoi.hpp verwendet dies unter Windows für __declspec(dllexport).
#
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


# ============================================================================
# 8. Compiler-Warnungen
# ============================================================================
#
# PRIVATE: Benutzer deiner Library erben diese Warnflags nicht.
#
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
# 9. Tests
# ============================================================================
#
# Für jeden unabhängigen Test kannst du ein eigenes Executable anlegen.
#
# Schema:
#
#   add_executable(test_xyz tests/test_xyz.cpp)
#   target_link_libraries(test_xyz PRIVATE HighVoronoi::HighVoronoi)
#   add_test(NAME test_xyz COMMAND test_xyz)
#
# Dann gezielt bauen:
#
#   cmake --build build/debug --target test_xyz
#
if(HIGHVORONOI_BUILD_TESTS)
    enable_testing()

    add_executable(test_basic
        tests/test_basic.cpp
    )

    target_link_libraries(test_basic
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_test(
        NAME test_basic
        COMMAND test_basic
    )

    add_executable(test_point_layout
        tests/test_point_layout.cpp
    )

    target_link_libraries(test_point_layout
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_test(
        NAME test_point_layout
        COMMAND test_point_layout
    )


    add_executable(test_voronoi_nodes
        tests/test_voronoi_nodes.cpp
    )

    target_link_libraries(test_voronoi_nodes
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_test(
        NAME test_voronoi_nodes
        COMMAND test_voronoi_nodes
    )

    # Internal synchronization tests. The src/ include path is intentionally
    # PRIVATE so detail/locks.hpp remains outside the public API.
    add_executable(test_locks
        tests/test_locks.cpp
    )

    target_include_directories(test_locks
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_locks
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_locks
        COMMAND test_locks
    )

    add_executable(test_hash_tables
        tests/test_hash_tables.cpp
    )

    target_include_directories(test_hash_tables
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_hash_tables
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_hash_tables
        COMMAND test_hash_tables
    )

    add_executable(test_hvdatabase
        tests/test_hvdatabase.cpp
    )

    target_include_directories(test_hvdatabase
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_hvdatabase
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_hvdatabase
        COMMAND test_hvdatabase
    )

    add_executable(test_hvview
        tests/test_hvview.cpp
    )

    target_include_directories(test_hvview
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_hvview
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_hvview
        COMMAND test_hvview
    )

    add_executable(test_boundary
        tests/test_boundary.cpp
    )

    target_include_directories(test_boundary
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_boundary
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_boundary
        COMMAND test_boundary
    )

    add_executable(test_abstract_mesh
        tests/test_abstract_mesh.cpp
    )

    target_include_directories(test_abstract_mesh
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_abstract_mesh
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_abstract_mesh
        COMMAND test_abstract_mesh
    )

    add_executable(test_read_write_list
        tests/test_read_write_list.cpp
    )

    target_include_directories(test_read_write_list
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_read_write_list
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_read_write_list
        COMMAND test_read_write_list
    )

    add_executable(test_voronoi_mesh
        tests/test_voronoi_mesh.cpp
    )

    target_include_directories(test_voronoi_mesh
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_voronoi_mesh
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_voronoi_mesh
        COMMAND test_voronoi_mesh
    )

    add_executable(test_mesh_view
        tests/test_mesh_view.cpp
    )

    target_include_directories(test_mesh_view
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_mesh_view
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_mesh_view
        COMMAND test_mesh_view
    )

    add_executable(test_search_tree
        tests/test_search_tree.cpp
    )

    target_include_directories(test_search_tree
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_search_tree
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_search_tree
        COMMAND test_search_tree
    )

    add_executable(test_search_tree_voronoi
        tests/test_search_tree_voronoi.cpp
    )

    target_include_directories(test_search_tree_voronoi
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_search_tree_voronoi
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_search_tree_voronoi
        COMMAND test_search_tree_voronoi
    )

    add_executable(test_qr_normal_solver
        tests/test_qr_normal_solver_20260807.cpp
    )

    target_link_libraries(test_qr_normal_solver
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_test(
        NAME test_qr_normal_solver
        COMMAND test_qr_normal_solver
    )

    add_executable(test_edge_iterator
        tests/test_edge_iterator.cpp
    )

    target_link_libraries(test_edge_iterator
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_test(
        NAME test_edge_iterator
        COMMAND test_edge_iterator
    )

    add_executable(test_raycaster
        tests/test_raycaster.cpp
    )

    target_link_libraries(test_raycaster
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_test(
        NAME test_raycaster
        COMMAND test_raycaster
    )

    add_executable(test_compute_voronoi_validation
        tests/test_compute_voronoi_validation.cpp
    )

    target_include_directories(test_compute_voronoi_validation
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_link_libraries(test_compute_voronoi_validation
        PRIVATE
            HighVoronoi::HighVoronoi
            Threads::Threads
    )

    add_test(
        NAME test_compute_voronoi_validation
        COMMAND test_compute_voronoi_validation
    )



endif()

# ============================================================================
# 10. Beispiele
# ============================================================================
#
# Beispiele sind kleine Programme, die die öffentliche API so benutzen wie ein
# externer Anwender.
#
if(HIGHVORONOI_BUILD_EXAMPLES)
    add_executable(highvoronoi_example_general_position
        examples/general_position.cpp
    )

    target_link_libraries(highvoronoi_example_general_position
        PRIVATE
            HighVoronoi::HighVoronoi
    )

    add_executable(highvoronoi_example_cubic_grid
        examples/cubic_grid.cpp
    )

    target_link_libraries(highvoronoi_example_cubic_grid
        PRIVATE
            HighVoronoi::HighVoronoi
    )
endif()


# ============================================================================
# 11. Benchmarks
# ============================================================================
#
# Benchmarks standardmäßig AUS, weil sie später oft zusätzliche Dependencies
# oder längere Build-Zeiten haben.
#
if(HIGHVORONOI_BUILD_BENCHMARKS)

    # Beispiel:
    #
    # add_executable(benchmark_kdtree
    #     benchmarks/benchmark_kdtree.cpp
    # )
    #
    # target_link_libraries(benchmark_kdtree
    #     PRIVATE
    #         HighVoronoi::HighVoronoi
    # )

endif()
