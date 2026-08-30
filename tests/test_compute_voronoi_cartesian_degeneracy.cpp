#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
inline constexpr Index Side = Index{4};
inline constexpr Index NodeCount = Side * Side * Side;
inline constexpr std::size_t ExpectedVertexCount = 125; // (Side + 1)^3

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::ReadWriteLock, DatabaseParameters>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::NodePoint;
using Boundary = Mesh::BoundaryType;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;
using RayParameters = highvoronoi::RaycastParameters<highvoronoi::InRangeRaycast, Scalar>;

std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
    } else {
        ++failed_checks;
        std::cerr << "    [FAIL] " << description << '\n';
    }
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

Nodes cartesian_nodes() {
    Nodes nodes(NodeCount);
    Index flat = Index{0};

    // Exact Cartesian grid, strictly inside [-1,+1]^3.
    for (Index z = Index{0}; z < Side; ++z) {
        for (Index y = Index{0}; y < Side; ++y) {
            for (Index x = Index{0}; x < Side; ++x) {
                nodes.set(
                    flat++,
                    point(
                        Scalar{0.2} + Scalar{0.2} * x,
                        Scalar{0.2} + Scalar{0.2} * y,
                        Scalar{0.2} + Scalar{0.2} * z));
            }
        }
    }
    return nodes;
}

std::size_t primary_vertex_count(const Mesh& mesh) {
    std::size_t count = 0;
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for ([[maybe_unused]] const auto& vertex : mesh.primary_vertices(cell)) {
            ++count;
        }
    }
    return count;
}

} // namespace

int main() {
    std::cout << "============================================================\n"
              << "ComputeVoronoi Cartesian InRange regression\n"
              << "4x4x4 exact Cartesian grid, all generators strictly inside\n"
              << "============================================================\n";

    auto database = std::make_shared<Database>(
        16384,
        DatabaseParameters{highvoronoi::DirectHash{16384}});
    Mesh mesh(cartesian_nodes(), cube_boundary(), database);

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    RayParameters ray_parameters;
    ray_parameters.variance_tolerance = Scalar{9e-14};
    auto raycaster = highvoronoi::make_raycaster(tree, ray_parameters);

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread,
        DatabaseParameters,
        EdgeParameters>;

    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{},
        std::nullopt,
        DatabaseParameters{highvoronoi::DirectHash{4096}},
        EdgeParameters{highvoronoi::DirectHash{8192}});

    std::cout << "\n[compute] using InRangeRaycast\n";
    compute.compute();

    const std::size_t vertices = primary_vertex_count(mesh);
    std::cout << "    primary vertices: " << vertices << '\n';
    std::cout << "    expected vertices: " << ExpectedVertexCount << '\n';
    check(vertices == ExpectedVertexCount,
          "all bounded Cartesian vertices are present");

    const auto complete = highvoronoi::verify_mesh_complete(
        mesh,
        Scalar{1e-14},
        true);

    std::cout << "    finite edge endpoints: "
              << complete.unique_finite_edge_endpoints << '\n';
    std::cout << "    edge closure: "
              << (complete.all_edges_have_two_occurrences ? "yes" : "no")
              << '\n';

    check(complete.consistency.valid(),
          "Cartesian result is geometrically consistent");
    check(complete.complete(),
          "Cartesian result is topologically complete");

    std::cout << "\nfailed checks: " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}
