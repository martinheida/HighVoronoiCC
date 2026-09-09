
#include <highvoronoi/high_voronoi.hpp>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;
using HighMesh = highvoronoi::HighVoronoiMesh<
    Scalar,
    Dimension,
    Database>;
using Point = HighMesh::NodePoint;
using Boundary = HighMesh::BoundaryType;

using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::InRangeRaycast,
    Scalar>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;

using HighCompute = highvoronoi::ComputeHighVoronoi<
    HighMesh,
    highvoronoi::geometry::KDSearch,
    RayParameters,
    highvoronoi::SingleThread,
    highvoronoi::SingleThread,
    DatabaseParameters,
    EdgeParameters>;

inline constexpr std::size_t DatabaseUnits = 65536;
inline constexpr std::size_t HashCapacity = 8192;

DatabaseParameters database_parameters() {
    return DatabaseParameters{
        highvoronoi::DirectHash{HashCapacity}};
}

EdgeParameters edge_parameters() {
    return EdgeParameters{
        highvoronoi::DirectHash{HashCapacity}};
}

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};
    return parameters;
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point result;
    result << x, y, z;
    return result;
}

Boundary make_boundary() {
    return Boundary::cuboid(
        point(2.0, 2.0, 2.0),
        point(-1.0, -1.0, -1.0),
        std::vector<Index>{});
}

std::vector<Point> make_points(Index count, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{-0.85},
        Scalar{0.85});

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(count));

    while (points.size() < static_cast<std::size_t>(count)) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        bool separated = true;
        for (const Point& existing : points) {
            if ((candidate - existing).norm() < Scalar{0.12}) {
                separated = false;
                break;
            }
        }

        if (separated) {
            points.push_back(candidate);
        }
    }

    return points;
}

void append_points(
    HighMesh& mesh,
    const std::vector<Point>& points) {

    for (const Point& value : points) {
        (void)mesh.append_visible_node(value);
    }
}

HighCompute::Report compute(HighMesh& mesh) {
    HighCompute operation(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters(),
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        database_parameters(),
        edge_parameters());

    return operation.compute();
}

bool print_and_check_complete(
    HighMesh& mesh,
    const char* stage) {

    auto data = mesh.data_mesh();
    const auto report = highvoronoi::verify_mesh_complete(
        data,
        Scalar{1e-12},
        true);

    std::cout
        << stage
        << ": visible nodes=" << mesh.size()
        << ", active internal nodes=" << data.size()
        << ", checked vertices="
        << report.consistency.checked_vertex_occurrences
        << ", complete="
        << (report.complete() ? "yes" : "no")
        << '\n';

    return report.complete();
}

void print_compute_report(
    const char* stage,
    const HighCompute::Report& report) {

    std::cout
        << stage
        << ": rounds=" << report.rounds
        << ", new-cell vertices=" << report.new_cell_vertices
        << ", invalidated old vertices="
        << report.invalidated_old_vertices
        << ", repair vertices=" << report.repair_vertices
        << ", visible nodes=" << report.final_visible_nodes
        << '\n';
}

} // namespace

int main() {
    const std::vector<Point> all_points =
        make_points(Index{30}, 0x48494748564f524fULL);

    HighMesh mesh(
        Index{Dimension},
        make_boundary(),
        std::in_place,
        DatabaseUnits,
        database_parameters());

    // 1. Insert the initial visible generators and integrate them.
    append_points(
        mesh,
        std::vector<Point>(
            all_points.begin(),
            all_points.begin() + 24));

    const auto initial = compute(mesh);
    print_compute_report("initial", initial);

    if (!print_and_check_complete(mesh, "initial")) {
        return 1;
    }

    // 2. Append four further visible generators. ComputeHighVoronoi detects
    //    that they are NEW and runs the incremental refinement path.
    append_points(
        mesh,
        std::vector<Point>(
            all_points.begin() + 24,
            all_points.begin() + 28));

    const auto refined = compute(mesh);
    print_compute_report("after refine", refined);

    if (!print_and_check_complete(mesh, "after refine")) {
        return 1;
    }

    // 3. HighVoronoi deletions are structural first. The following compute
    //    consumes the pending deletion and closes the damaged cells.
    (void)mesh.erase_visible_nodes(
        std::vector<Index>{Index{3}, Index{11}});

    const auto removed = compute(mesh);
    print_compute_report("after remove", removed);

    return print_and_check_complete(mesh, "after remove") ? 0 : 1;
}


