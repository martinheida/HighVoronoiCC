

#include <highvoronoi/voronoi.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;

// The ordinary public default is CombinedRaycast. Keep the example on the
// default path rather than explicitly selecting the legacy InRange variant.
using RayParameters = highvoronoi::RaycastParameters<
    highvoronoi::CombinedRaycast,
    Scalar>;

inline constexpr std::size_t DatabaseUnits = 65536;
inline constexpr std::size_t HashCapacity = 8192;

DatabaseParameters database_parameters() {
    return DatabaseParameters{
        highvoronoi::DirectHash{HashCapacity}};
}

RayParameters ray_parameters() {
    RayParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};
    return parameters;
}

Boundary make_boundary() {
    return Boundary::cuboid(
        Point::Constant(Scalar{6}),
        Point::Constant(Scalar{-3}),
        std::vector<Index>{});
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        DatabaseUnits,
        database_parameters());
}

std::vector<Point> make_points(Index count, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{-1.8},
        Scalar{1.8});

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(count));

    while (points.size() < static_cast<std::size_t>(count)) {
        Point candidate;
        for (int coordinate_index = 0;
             coordinate_index < Dimension;
             ++coordinate_index) {
            candidate[coordinate_index] = coordinate(random);
        }

        bool separated = true;
        for (const Point& existing : points) {
            if ((candidate - existing).norm() < Scalar{0.20}) {
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

Mesh make_mesh(const std::vector<Point>& points) {
    Nodes nodes(static_cast<Index>(points.size()));

    for (Index index = Index{0};
         index < static_cast<Index>(points.size());
         ++index) {
        nodes.set(index, points[static_cast<std::size_t>(index)]);
    }

    return Mesh(
        std::move(nodes),
        make_boundary(),
        make_database());
}

void compute_initial_mesh(Mesh& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    auto raycaster = highvoronoi::make_raycaster(
        tree,
        ray_parameters());

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread>;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{});
    compute.compute();
}

bool print_and_check_complete(
    Mesh& mesh,
    const char* stage) {

    const auto report = highvoronoi::verify_mesh_complete(
        mesh,
        Scalar{1e-12},
        true);

    std::cout
        << stage
        << ": nodes=" << mesh.size()
        << ", checked vertices="
        << report.consistency.checked_vertex_occurrences
        << ", finite edge endpoints="
        << report.unique_finite_edge_endpoints
        << ", complete="
        << (report.complete() ? "yes" : "no")
        << '\n';

    return report.complete();
}

} // namespace

int main() {
    const std::vector<Point> all_points =
        make_points(Index{30}, 0x494e4352454d454eULL);

    std::vector<Point> initial_points(
        all_points.begin(),
        all_points.begin() + 24);

    std::vector<Point> added_points(
        all_points.begin() + 24,
        all_points.begin() + 28);

    Mesh mesh = make_mesh(initial_points);

    // 1. Build the initial ordinary Voronoi mesh.
    compute_initial_mesh(mesh);
    if (!print_and_check_complete(mesh, "initial")) {
        return 1;
    }

    // 2. Refine it by appending four new generators.
    highvoronoi::RefineVoronoi<Mesh> refine(
        mesh,
        added_points,
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    (void)refine.compute();

    std::cout
        << "refine: appended="
        << refine.report().appended_nodes
        << ", new-cell vertices="
        << refine.report().new_cell_vertices
        << ", affected old cells="
        << refine.report().affected_nodes
        << '\n';

    if (!print_and_check_complete(mesh, "after refine")) {
        return 1;
    }

    // 3. Remove two current public generators and immediately repair the holes.
    highvoronoi::RemoveVoronoi<Mesh> remove(
        mesh,
        std::vector<Index>{Index{2}, Index{9}},
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    (void)remove.compute();

    std::cout
        << "remove: removed="
        << remove.report().removed_nodes
        << ", erased vertices="
        << remove.report().erased_vertices
        << ", affected cells="
        << remove.report().affected_nodes
        << ", repair vertices="
        << remove.report().repair_vertices
        << '\n';

    return print_and_check_complete(mesh, "after remove") ? 0 : 1;
}



