
#include <highvoronoi/voronoi.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
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

constexpr Index integer_power(Index base, int exponent) {
    Index result = Index{1};
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}

std::shared_ptr<Database> make_database() {
    return std::make_shared<Database>(
        4096,
        DatabaseParameters{highvoronoi::DirectHash{16384}});
}

Boundary make_boundary() {
    // The generators occupy {-3,-1,1,3}^4. The box [-4,4]^4 lies half a grid
    // spacing beyond the outer generators and closes all cells.
    const Point dimensions = Point::Constant(Scalar{8});
    const Point offset = Point::Constant(Scalar{-4});
    return Boundary::cuboid(dimensions, offset, std::vector<Index>{});
}

Point central_cube_corner(Index index) {
    Point point;
    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        const int bit = Dimension - 1 - coordinate;
        point[coordinate] =
            ((index >> bit) & Index{1}) != Index{0}
                ? Scalar{1}
                : Scalar{-1};
    }
    return point;
}

Mesh make_mesh() {
    constexpr Index values_per_axis = Index{4};
    constexpr Index total_node_count =
        integer_power(values_per_axis, Dimension);
    constexpr Index central_cube_node_count = Index{1} << Dimension;
    constexpr std::array<Scalar, 4> coordinates{
        Scalar{-3}, Scalar{-1}, Scalar{1}, Scalar{3}};

    Nodes nodes(total_node_count);

    // Put the 2^4 generators of the central cube first. This ordering is not
    // required for the final mesh, but makes this deterministic example match
    // the regression configuration used by the test suite.
    Index next = Index{0};
    for (Index index = Index{0}; index < central_cube_node_count; ++index) {
        nodes.set(next++, central_cube_corner(index));
    }

    // Add the remaining generators of the Cartesian grid {-3,-1,1,3}^4.
    for (Index code = Index{0}; code < total_node_count; ++code) {
        Index encoded = code;
        Point point;
        bool belongs_to_central_cube = true;

        for (int coordinate = Dimension - 1; coordinate >= 0; --coordinate) {
            const Index digit = encoded % values_per_axis;
            encoded /= values_per_axis;
            point[coordinate] = coordinates[static_cast<std::size_t>(digit)];
            if (std::abs(point[coordinate]) != Scalar{1}) {
                belongs_to_central_cube = false;
            }
        }

        if (!belongs_to_central_cube) {
            nodes.set(next++, point);
        }
    }

    if (next != total_node_count) {
        throw std::logic_error("Cubic-grid example has wrong node count.");
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

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    // Degenerate Cartesian grids need the in-range ray caster: many generators
    // may belong to one Voronoi vertex and one geometric edge.
    highvoronoi::RaycastParameters<
        highvoronoi::InRangeRaycast,
        Scalar> raycast_parameters;
    raycast_parameters.variance_tolerance = Scalar{9e-14};

    auto raycaster = highvoronoi::make_raycaster(tree, raycast_parameters);

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

    const std::size_t vertex_count = primary_vertex_count(mesh);
    constexpr std::size_t expected_vertex_count =
        static_cast<std::size_t>(integer_power(Index{5}, Dimension));

    std::cout << "Stored primary vertices: " << vertex_count << '\n';
    std::cout << "Expected primary vertices: " << expected_vertex_count << '\n';

    const auto verification = highvoronoi::verify_mesh(
        mesh,
        Scalar{1e-18},
        true);

    std::cout << "Checked vertex occurrences: "
              << verification.checked_vertex_occurrences << '\n';
    std::cout << "Verification errors: "
              << verification.error_count() << '\n';

    const bool complete = vertex_count == expected_vertex_count;
    if (!complete) {
        std::cerr << "Unexpected number of vertices in the cubic grid.\n";
    }

    return complete && verification.valid() ? 0 : 1;
}

