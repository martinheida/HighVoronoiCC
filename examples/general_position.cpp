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
#include <random>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 4;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::VertexPoint;
using Boundary = Mesh::BoundaryType;

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        4096,
        DatabaseParameters{highvoronoi::DirectHash{4096}});
}

Boundary make_boundary() {
    // The domain is the box [-6, 6]^4. Boundary planes are represented by
    // mirror generators while a cell is explored.
    const Point dimensions = Point::Constant(Scalar{12});
    const Point offset = Point::Constant(Scalar{-6});
    return Boundary::cuboid(dimensions, offset, std::vector<Index>{});
}

Mesh make_mesh() {
    constexpr Index base_node_count = Index{Dimension + 1};
    constexpr Index extra_node_count = Index{100};
    constexpr Index total_node_count = base_node_count + extra_node_count;

    Nodes nodes(total_node_count);

    // A fixed random seed keeps this example reproducible. Continuous random
    // coordinates give a general-position configuration with probability one.
    std::mt19937_64 base_random(0x4752414d5152ULL);
    std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1});

    for (Index index = Index{0}; index < base_node_count; ++index) {
        Point point;
        do {
            for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
                point[coordinate] = normal(base_random);
            }
        } while (point.norm() < Scalar{0.1});

        point.normalize();
        nodes.set(index, point);
    }

    // Add generators on the coordinate axes and a random outer shell. They
    // make the interior configuration well surrounded before the box boundary
    // closes the remaining outer cells.
    Index next = base_node_count;
    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        Point positive = Point::Zero();
        positive[coordinate] = Scalar{4};
        nodes.set(next++, positive);

        Point negative = Point::Zero();
        negative[coordinate] = Scalar{-4};
        nodes.set(next++, negative);
    }

    std::mt19937_64 shell_random(0x5241594341535445ULL);
    std::uniform_real_distribution<Scalar> radius(Scalar{3}, Scalar{5});

    while (next < total_node_count) {
        Point point;
        do {
            for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
                point[coordinate] = normal(shell_random);
            }
        } while (point.norm() < Scalar{0.1});

        point.normalize();
        point *= radius(shell_random);
        nodes.set(next++, point);
    }

    return Mesh(std::move(nodes), make_boundary(), make_database());
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
    Mesh mesh = make_mesh();

    // Build the nearest-neighbour backend used by the ray caster.
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    highvoronoi::RaycastParameters<
        highvoronoi::ClassicRaycast,
        Scalar> raycast_parameters;
    raycast_parameters.variance_tolerance = Scalar{7e-14};

    auto raycaster = highvoronoi::make_raycaster(tree, raycast_parameters);

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        highvoronoi::SingleThread,
        highvoronoi::SingleThread>;

    // Compute all cells of the mesh. The current public example deliberately
    // uses the serial execution path; parallel construction is under development.
    Compute compute(
        mesh,
        raycaster,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{});
    compute.compute();

    std::cout << "Stored primary vertices: "
              << primary_vertex_count(mesh) << '\n';

    // verify_mesh checks consistency of every stored vertex. It verifies
    // equal generator distances and nearest-neighbour membership, but does not
    // by itself prove that the mesh is complete.
    const auto verification = highvoronoi::verify_mesh(
        mesh,
        Scalar{1e-18},
        true);

    std::cout << "Checked vertex occurrences: "
              << verification.checked_vertex_occurrences << '\n';
    std::cout << "Verification errors: "
              << verification.error_count() << '\n';

    return verification.valid() ? 0 : 1;
}
