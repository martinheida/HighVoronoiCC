

#include <highvoronoi/high_voronoi.hpp>
#include <highvoronoi/mesh_engines.hpp>

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using HybridDatabase = highvoronoi::HybridDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters>;
using HighMesh = highvoronoi::HighVoronoiMesh<
    Scalar,
    Dimension,
    HybridDatabase>;
using CuboidEngine = highvoronoi::CuboidMeshEngine<
    Scalar,
    Scalar,
    Index,
    Dimension>;
using GridCounts = CuboidEngine::CountPoint;
using Point = HighMesh::NodePoint;
using Boundary = HighMesh::BoundaryType;
using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;
using HighCompute = highvoronoi::ComputeHighVoronoi<
    HighMesh,
    highvoronoi::geometry::KDSearch,
    RayParameters>;

inline constexpr std::size_t DatabaseUnits = 32768;
inline constexpr std::size_t HashCapacity = 32768;
inline constexpr std::size_t PerfectEngineVertices = 27; // (4 - 1)^3
inline constexpr Scalar Tolerance = Scalar{1e-14};

std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
    } else {
        ++failed_checks;
        std::cerr << "    [FAIL] " << description << '\n';
    }
}

void section(std::string_view title) {
    std::cout << "\n============================================================\n"
              << title << '\n'
              << "============================================================\n";
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

Boundary cube_boundary() {
    return Boundary::cuboid(
        point(2.0, 2.0, 2.0),
        point(-1.0, -1.0, -1.0),
        std::vector<Index>{});
}

DatabaseParameters database_parameters() {
    return DatabaseParameters{highvoronoi::DirectHash{HashCapacity}};
}

CuboidEngine make_engine() {
    GridCounts repetitions;
    repetitions << Index{4}, Index{4}, Index{4};
    return CuboidEngine(
        point(0.2, 0.2, 0.2),
        point(0.2, 0.2, 0.2),
        repetitions);
}

void print_complete_state(HighMesh& mesh, std::string_view name) {
    auto view = mesh.data_mesh();
    const auto report = highvoronoi::verify_mesh_complete(
        view,
        Tolerance,
        true);

    std::cout << "    " << name << ": finite edge endpoints = "
              << report.unique_finite_edge_endpoints
              << ", edge closure = "
              << (report.all_edges_have_two_occurrences ? "yes" : "no")
              << '\n';

    check(report.consistency.valid(),
          "mesh is geometrically consistent");
    check(report.complete(),
          "bounded mesh is topologically complete");
}

} // namespace

int main() {
    section("HighVoronoi CuboidEngine + explicit InRangeRaycast regression");
    std::cout << "Domain: [-1,+1]^3, no periodic boundaries\n"
              << "Cuboid nodes: {0.2,0.4,0.6,0.8}^3\n";

    HighMesh mesh(
        Index{Dimension},
        cube_boundary(),
        DatabaseUnits,
        database_parameters());

    // One pre-existing visible node perturbs the perfect Cartesian grid.
    // It should invalidate some analytic CuboidEngine vertices during import.
    (void)mesh.append_visible_node(point(0.53, 0.47, 0.56));

    section("1. Import CuboidEngine");
    const auto append = mesh.append_engine(make_engine());
    std::cout << "    appended grid nodes:       " << append.appended_nodes << '\n'
              << "    perfect engine vertices:  " << PerfectEngineVertices << '\n'
              << "    accepted engine vertices: " << append.accepted_vertices << '\n'
              << "    rejected invalid vertices:" << append.rejected_invalid_vertices << '\n';

    check(append.appended_nodes == 64,
          "all 4^3 CuboidEngine nodes are inserted");
    check(append.accepted_vertices < PerfectEngineVertices,
          "the extra node invalidates at least one analytic grid vertex");

    {
        auto view = mesh.data_mesh();
        const auto consistency = highvoronoi::verify_mesh(view, Tolerance, true);
        check(consistency.valid(),
              "imported partial engine mesh is geometrically consistent");
    }

    section("2. Complete imported mesh with explicit InRangeRaycast");
    RayParameters ray_parameters;
    ray_parameters.variance_tolerance = Scalar{9e-14};
    HighCompute compute(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters);
    const auto first = compute.compute();
    std::cout << "    rounds:                  " << first.rounds << '\n'
              << "    deleted vertices:        " << first.deleted_vertices << '\n'
              << "    invalidated old vertices:" << first.invalidated_old_vertices << '\n'
              << "    new-cell vertices:       " << first.new_cell_vertices << '\n'
              << "    repair vertices:         " << first.repair_vertices << '\n'
              << "    new reference nodes:     " << first.new_reference_nodes << '\n';
    print_complete_state(mesh, "after first completion");

    section("3. Delete two visible grid nodes and repair again");
    // Public node 0 is the perturbing node; the following indices are grid nodes.
    const auto removed = mesh.erase_visible_nodes(
        std::vector<Index>{Index{1}, Index{22}});
    std::cout << "    removed internal nodes: " << removed.removed_internal.size() << '\n';
    check(removed.removed_internal.size() == 2,
          "two non-periodic visible nodes are marked deleted");

    HighCompute recompute(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters);
    const auto second = recompute.compute();
    std::cout << "    rounds:                  " << second.rounds << '\n'
              << "    deleted vertices:        " << second.deleted_vertices << '\n'
              << "    invalidated old vertices:" << second.invalidated_old_vertices << '\n'
              << "    new-cell vertices:       " << second.new_cell_vertices << '\n'
              << "    repair vertices:         " << second.repair_vertices << '\n'
              << "    new reference nodes:     " << second.new_reference_nodes << '\n';
    print_complete_state(mesh, "after delete + repair");

    std::cout << "\nfailed checks: " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}


