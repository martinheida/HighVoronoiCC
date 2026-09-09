
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
inline constexpr std::size_t HashCapacity = 65536;

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

Boundary periodic_unit_cube() {
    // x and y periodic, z non-periodic.
    return Boundary::cuboid(
        point(1.0, 1.0, 1.0),
        point(0.0, 0.0, 0.0),
        std::vector<Index>{Index{0}, Index{1}});
}

std::vector<Point> refinement_points() {
    return {
        point(0.008, 0.18, 0.37),
        point(0.992, 0.73, 0.63),
        point(0.43, 0.009, 0.84)
    };
}

bool far_enough(
    const Point& candidate,
    const std::vector<Point>& points,
    Scalar minimum_distance) {

    for (const Point& existing : points) {
        if ((candidate - existing).norm() < minimum_distance) {
            return false;
        }
    }
    return true;
}

std::vector<Point> initial_points() {
    const std::vector<Point> future = refinement_points();

    std::mt19937_64 random(0x504552494f444943ULL);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{0.025},
        Scalar{0.975});

    std::vector<Point> result;
    result.reserve(28);

    while (result.size() < 28) {
        Point candidate = point(
            coordinate(random),
            coordinate(random),
            coordinate(random));

        if (!far_enough(candidate, result, Scalar{0.055})) {
            continue;
        }
        if (!far_enough(candidate, future, Scalar{0.085})) {
            continue;
        }

        result.push_back(candidate);
    }

    return result;
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

bool visible_output_is_complete(HighMesh& mesh) {
    highvoronoi::VisibleFirstMesh<HighMesh> view(mesh);

    const auto report = highvoronoi::verify_mesh_complete_cells(
        view,
        Index{0},
        view.visible_end(),
        Scalar{1e-12},
        true);

    std::cout
        << "visible output: visible cells=" << view.visible_end()
        << ", active internal nodes=" << view.size()
        << ", checked visible vertex occurrences="
        << report.consistency.checked_vertex_occurrences
        << ", edge closure="
        << (report.all_edges_have_two_occurrences ? "yes" : "no")
        << '\n';

    return report.complete();
}

bool public_vertices_lie_in_visible_domain(const HighMesh& mesh) {
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for (const auto& vertex : mesh.vertices(cell)) {
            if (!mesh.external_boundary().contains(
                    vertex.position,
                    Scalar{1e-10})) {
                return false;
            }
        }
    }
    return true;
}

void print_report(
    const char* stage,
    const HighCompute::Report& report,
    const HighMesh& mesh) {

    std::cout
        << stage
        << ": rounds=" << report.rounds
        << ", visible=" << mesh.size()
        << ", internal=" << mesh.internal_node_count()
        << ", new references=" << report.new_reference_nodes
        << ", periodic repair vertices="
        << report.periodic_boundary_repair.new_vertices
        << '\n';
}

} // namespace

int main() {
    HighMesh mesh(
        Index{Dimension},
        periodic_unit_cube(),
        std::in_place,
        DatabaseUnits,
        database_parameters());

    // 1. Initial periodic closure.
    append_points(mesh, initial_points());

    const auto initial = compute(mesh);
    print_report("initial", initial, mesh);

    if (mesh.internal_node_count() <= mesh.size()) {
        std::cerr << "Expected invisible periodic reference nodes.\n";
        return 1;
    }

    if (!visible_output_is_complete(mesh) ||
        !public_vertices_lie_in_visible_domain(mesh)) {
        return 1;
    }

    // 2. Append three generators close to periodic faces. The normal
    //    ComputeHighVoronoi path refines the visible mesh, creates any new
    //    invisible references and closes the periodic boundary again.
    append_points(mesh, refinement_points());

    const auto refined = compute(mesh);
    print_report("after periodic refine", refined, mesh);

    if (mesh.size() != Index{31}) {
        std::cerr << "Unexpected visible node count after refinement.\n";
        return 1;
    }

    return visible_output_is_complete(mesh) &&
           public_vertices_lie_in_visible_domain(mesh)
        ? 0
        : 1;
}


