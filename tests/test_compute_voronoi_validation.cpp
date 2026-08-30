#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string_view>
#include <unordered_set>
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
using Sigma = Mesh::Sigma;
using Boundary = Mesh::BoundaryType;

using ClassicParameters =
    highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast, Scalar>;
using InRangeParameters =
    highvoronoi::RaycastParameters<highvoronoi::InRangeRaycast, Scalar>;

inline constexpr Index general_base_node_count = Index{Dimension + 1};
inline constexpr Index general_extra_node_count = Index{100};
inline constexpr Index central_cube_node_count = Index{1} << Dimension;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }
    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

std::shared_ptr<Database> make_database(std::size_t capacity = 4096) {
    return std::make_shared<Database>(
        4096,
        DatabaseParameters{highvoronoi::DirectHash{capacity}});
}

Boundary centered_nonperiodic_box(Scalar size) {
    Point dimensions = Point::Constant(size);
    Point offset = Point::Constant(-size / Scalar{2});
    return Boundary::cuboid(
        dimensions,
        offset,
        std::vector<Index>{});
}

constexpr Index integer_power(Index base, int exponent) {
    Index value = Index{1};
    for (int i = 0; i < exponent; ++i) {
        value *= base;
    }
    return value;
}

Mesh make_general_mesh() {
    constexpr Index total_node_count =
        general_base_node_count + general_extra_node_count;

    Nodes nodes(total_node_count);

    std::mt19937_64 base_random(0x4752414d5152ULL);
    std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1});

    for (Index index = Index{0}; index < general_base_node_count; ++index) {
        Point point;
        do {
            for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
                point[coordinate] = normal(base_random);
            }
        } while (point.norm() < Scalar{0.1});
        point.normalize();
        nodes.set(index, point);
    }

    Index next = general_base_node_count;
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

    // All generators lie inside [-6,6]^4. The boundary makes every cell bounded.
    return Mesh(
        std::move(nodes),
        centered_nonperiodic_box(Scalar{12}),
        make_database());
}

Point central_cube_corner(Index index) {
    Point result;
    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        const int bit = Dimension - 1 - coordinate;
        result[coordinate] =
            ((index >> bit) & Index{1}) != Index{0}
                ? Scalar{1}
                : Scalar{-1};
    }
    return result;
}

Mesh make_cubical_grid_mesh() {
    constexpr Index values_per_axis = Index{4};
    constexpr Index total_node_count =
        integer_power(values_per_axis, Dimension);
    constexpr std::array<Scalar, 4> coordinates{
        Scalar{-3}, Scalar{-1}, Scalar{1}, Scalar{3}};

    Nodes nodes(total_node_count);

    Index next = Index{0};
    for (Index index = Index{0}; index < central_cube_node_count; ++index) {
        nodes.set(next++, central_cube_corner(index));
    }

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
        throw std::logic_error("Cubical-grid fixture has wrong node count.");
    }

    // Grid {-3,-1,1,3}^4, bounded half a grid spacing beyond outer generators.
    return Mesh(
        std::move(nodes),
        centered_nonperiodic_box(Scalar{8}),
        make_database(16384));
}

std::size_t unique_vertex_count(const Mesh& mesh) {
    std::unordered_set<Mesh::Address> addresses;
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for (const auto& vertex : mesh.primary_vertices(cell)) {
            addresses.insert(vertex.address);
        }
    }
    return addresses.size();
}

void test_general_position_compute() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] ComputeVoronoi: bounded general-position 4D mesh\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_general_mesh();

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    ClassicParameters parameters;
    parameters.variance_tolerance = Scalar{7e-14};
    auto raycaster = highvoronoi::make_raycaster(tree, parameters);

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

    const auto count = unique_vertex_count(mesh);
    std::cout << "  stored primary vertices: " << count << '\n';
    check(count > 0, "ComputeVoronoi stores vertices");

    const auto verification = highvoronoi::verify_mesh(
        mesh,
        Scalar{1e-18},
        true);

    std::cout << "  checked vertex occurrences: "
              << verification.checked_vertex_occurrences << '\n';
    std::cout << "  verification errors: "
              << verification.error_count() << '\n';

    check(verification.valid(), "all stored general-position vertices are consistent");

    const auto self_comparison = highvoronoi::compare_meshes(mesh, mesh);
    check(self_comparison.equal(), "compare_meshes accepts a mesh compared with itself");
}

void test_cubical_grid_compute() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] ComputeVoronoi: bounded 3x3x3x3 cubical mesh\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_cubical_grid_mesh();

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    InRangeParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};
    auto raycaster = highvoronoi::make_raycaster(tree, parameters);

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

    const auto count = unique_vertex_count(mesh);
    const std::size_t expected_count =
        static_cast<std::size_t>(integer_power(Index{5}, Dimension));
    std::cout << "  stored primary vertices: " << count << '\n';
    std::cout << "  expected cubical vertices: " << expected_count << '\n';
    check(
        count == expected_count,
        "ComputeVoronoi finds all 5^4 bounded cubical-grid vertices");

    const auto verification = highvoronoi::verify_mesh(
        mesh,
        Scalar{1e-18},
        true);

    std::cout << "  checked vertex occurrences: "
              << verification.checked_vertex_occurrences << '\n';
    std::cout << "  verification errors: "
              << verification.error_count() << '\n';

    check(verification.valid(), "all stored cubical-grid vertices are consistent");
    check(highvoronoi::verify_mesh_complete(mesh, Scalar{1e-10}, true).complete(),
      "cubical-grid mesh is topologically complete");
}

} // namespace

int main() {
    test_general_position_compute();
    test_cubical_grid_compute();

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}
