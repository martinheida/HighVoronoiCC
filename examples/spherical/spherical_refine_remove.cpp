#include <highvoronoi/spherical_voronoi.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3; // ambient R^3 -> public sphere S^2

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters,
    Dimension>;
using Mesh = highvoronoi::SphereVoronoiMesh<Scalar, Dimension, Database>;
using Point = Mesh::NodePoint;

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        16384,
        DatabaseParameters{highvoronoi::DirectHash{16384}});
}

highvoronoi::RaycastParameters<
    highvoronoi::ClassicRaycast,
    Scalar> ray_parameters() {
    highvoronoi::RaycastParameters<
        highvoronoi::ClassicRaycast,
        Scalar> parameters;
    parameters.variance_tolerance = Scalar{1e-12};
    return parameters;
}

// Deterministic approximately uniform points on S^2.
std::vector<Point> fibonacci_sphere(Index count) {
    constexpr Scalar pi =
        Scalar{3.141592653589793238462643383279502884};
    const Scalar golden_angle = pi * (Scalar{3} - std::sqrt(Scalar{5}));

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(count));

    for (Index i = Index{0}; i < count; ++i) {
        const Scalar z = Scalar{1} -
            Scalar{2} * (static_cast<Scalar>(i) + Scalar{0.5}) /
            static_cast<Scalar>(count);
        const Scalar radius = std::sqrt(
            (std::max)(Scalar{0}, Scalar{1} - z * z));
        const Scalar phi = golden_angle * static_cast<Scalar>(i);

        Point point;
        point << radius * std::cos(phi), radius * std::sin(phi), z;
        points.push_back(point);
    }
    return points;
}

bool print_state(const Mesh& mesh, const char* stage) {
    const auto vertices = mesh.vertices();
    bool unit_sphere = true;
    for (const auto& vertex : vertices) {
        unit_sphere = unit_sphere &&
            std::abs(vertex.position.norm() - Scalar{1}) < Scalar{2e-9};
    }

    std::cout << stage
              << ": visible nodes=" << mesh.size()
              << ", spherical vertices=" << vertices.size()
              << ", all vertices on S^2=" << (unit_sphere ? "yes" : "no")
              << '\n';
    return unit_sphere && !vertices.empty();
}

} // namespace

int main() {
    const std::vector<Point> all_points = fibonacci_sphere(Index{24});

    std::vector<Point> initial(all_points.begin(), all_points.begin() + 20);
    std::vector<Point> added(all_points.begin() + 20, all_points.end());

    // SphericalVoronoiMesh internally computes the Euclidean Voronoi cell of
    // the origin and radially projects that cell's vertices onto S^2.
    Mesh mesh(std::move(initial), make_database());

    mesh.compute(
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());
    if (!print_state(mesh, "initial")) {
        return 1;
    }

    // After the first compute(), add visible spherical generators through
    // refine(); append_node()/append_nodes() are intentionally initial-setup APIs.
    const auto refine_report = mesh.refine(
        std::move(added),
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    std::cout << "refine: appended public nodes="
              << refine_report.appended_public_nodes
              << ", invalidated vertices="
              << refine_report.invalidated_vertices << '\n';
    if (!print_state(mesh, "after refine")) {
        return 2;
    }

    const auto remove_report = mesh.remove(
        std::vector<Index>{Index{2}, Index{7}},
        highvoronoi::geometry::KDSearch{8, 1},
        ray_parameters());

    std::cout << "remove: removed public nodes="
              << remove_report.removed_public_nodes << '\n';
    return print_state(mesh, "after remove") ? 0 : 3;
}
