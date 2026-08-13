#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/edge_iterator.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

// ============================================================================
// One global type configuration for the complete test
// ============================================================================

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 4;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<
    highvoronoi::EmptyLock,
    DatabaseParameters>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using ExtendedNodes = Mesh::ExtendedNodes;
using Point = Mesh::VertexPoint;
using Sigma = Mesh::Sigma;
using Iterator = highvoronoi::EdgeIterator<ExtendedNodes>;
using Candidate = Iterator::Candidate;
using OnQueueEdges = Iterator::OnQueueEdges;

using ClassicParameters =
    highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast, Scalar>;
using InRangeParameters =
    highvoronoi::RaycastParameters<highvoronoi::InRangeRaycast, Scalar>;

inline constexpr Scalar point_tolerance = 2.0e-8;
inline constexpr Index general_base_node_count = Index{Dimension + 1};
inline constexpr Index general_extra_node_count = Index{100};
inline constexpr Index central_cube_node_count = Index{1} << Dimension;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

// ============================================================================
// Minimal test tools
// ============================================================================

void check(bool condition, std::string_view description) {
    ++performed_checks;

    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
        return;
    }

    ++failed_checks;
    std::cerr << "    [FAIL] " << description << '\n';
}

std::shared_ptr<Database> make_database() {
    const DatabaseParameters parameters{
        highvoronoi::DirectHash{512}
    };
    return std::make_shared<Database>(512, parameters);
}

Point zero_point() {
    return Point::Zero();
}

bool same_point(
    const Point& left,
    const Point& right,
    Scalar tolerance = point_tolerance) {

    return (left - right).norm() <= tolerance;
}

void print_point(const Point& point) {
    std::cout << '(';
    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        if (coordinate != 0) {
            std::cout << ',';
        }
        std::cout << std::setprecision(10) << point[coordinate];
    }
    std::cout << ')';
}

void print_indices(const std::vector<Index>& values) {
    std::cout << '(';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << values[i];
    }
    std::cout << ')';
}

void print_edge(const Candidate& edge) {
    std::cout << "      edge minimal=";
    print_indices(edge.indices);
    std::cout << " full=";
    print_indices(edge.full_indices);
    std::cout << " u=";
    print_point(edge.direction);
    std::cout << " du=" << std::setprecision(3)
              << edge.cycle_error << '\n';
}

template <class Result>
void print_raycast(const Result& result, const Sigma& sigma) {
    std::cout << "        -> ";
    if (result.status == highvoronoi::RayCastStatus::Infinite) {
        std::cout << "INFINITE\n";
        return;
    }

    std::cout << "t=" << std::setprecision(10) << result.t
              << " generator=" << result.generator
              << " r=";
    print_point(result.position);
    std::cout << " sigma=";
    print_indices(std::vector<Index>(sigma.begin(), sigma.end()));
    std::cout << '\n';
}

Sigma first_indices(Index count) {
    Sigma sigma;
    sigma.reserve(static_cast<std::size_t>(count));

    for (Index index = Index{0}; index < count; ++index) {
        sigma.push_back(index);
    }

    return sigma;
}

Mesh::VertexRecord load_vertex(Mesh& mesh, Index cell) {
    for (const auto& vertex : mesh.vertices(cell)) {
        return vertex;
    }
    throw std::runtime_error("Expected stored Voronoi vertex.");
}

constexpr Index integer_power(Index base, int exponent) {
    Index value = Index{1};
    for (int i = 0; i < exponent; ++i) {
        value *= base;
    }
    return value;
}

// ============================================================================
// General-position fixture
// ============================================================================

/**
 * The first d+1 generators lie on the unit sphere, so r=0 is known a priori
 * to be equidistant from them. The remaining 100 generators are farther away.
 *
 * The first 2d additional nodes are +/-4 e_i. They guarantee that every
 * outgoing edge has at least one forward candidate, independently of the
 * random outer shell. The remaining outer nodes are deterministic random
 * points with radius in [3,5].
 */
Mesh make_general_raycast_mesh() {
    constexpr Index total_node_count =
        general_base_node_count + general_extra_node_count;

    Nodes nodes(total_node_count);

    std::mt19937_64 base_random(0x4752414d5152ULL);
    std::normal_distribution<Scalar> normal(Scalar{0}, Scalar{1});

    for (Index index = Index{0};
         index < general_base_node_count;
         ++index) {

        Point point;
        do {
            for (int coordinate = 0;
                 coordinate < Dimension;
                 ++coordinate) {
                point[coordinate] = normal(base_random);
            }
        } while (point.norm() < Scalar{0.1});

        point.normalize();
        nodes.set(index, point);
    }

    Index next = general_base_node_count;

    // Guaranteed forward blockers for every possible unit ray direction.
    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        Point positive = Point::Zero();
        positive[coordinate] = Scalar{4};
        nodes.set(next++, positive);

        Point negative = Point::Zero();
        negative[coordinate] = Scalar{-4};
        nodes.set(next++, negative);
    }

    std::mt19937_64 shell_random(0x5241594341535445ULL);
    std::uniform_real_distribution<Scalar> radius(
        Scalar{3},
        Scalar{5});

    while (next < total_node_count) {
        Point point;

        do {
            for (int coordinate = 0;
                 coordinate < Dimension;
                 ++coordinate) {
                point[coordinate] = normal(shell_random);
            }
        } while (point.norm() < Scalar{0.1});

        point.normalize();
        point *= radius(shell_random);
        nodes.set(next++, point);
    }

    return Mesh(std::move(nodes), make_database());
}

// ============================================================================
// Degenerate cubical-grid fixture
// ============================================================================

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

/**
 * Build the generator grid {-3,-1,1,3}^d.
 *
 * There are 4^d generators and therefore 3^d grid cubes. The central cube
 * [-1,1]^d is inserted first, so its 2^d generators keep exactly the same
 * indices 0,...,2^d-1 as in the EdgeIterator cube test.
 */
Mesh make_cubical_grid_mesh() {
    constexpr Index values_per_axis = Index{4};
    constexpr Index total_node_count =
        integer_power(values_per_axis, Dimension);

    constexpr std::array<Scalar, 4> coordinates{
        Scalar{-3},
        Scalar{-1},
        Scalar{1},
        Scalar{3}
    };

    Nodes nodes(total_node_count);

    Index next = Index{0};
    for (Index index = Index{0};
         index < central_cube_node_count;
         ++index) {
        nodes.set(next++, central_cube_corner(index));
    }

    for (Index code = Index{0};
         code < total_node_count;
         ++code) {

        Index encoded = code;
        Point point;
        bool belongs_to_central_cube = true;

        for (int coordinate = Dimension - 1;
             coordinate >= 0;
             --coordinate) {

            const Index digit = encoded % values_per_axis;
            encoded /= values_per_axis;

            point[coordinate] =
                coordinates[static_cast<std::size_t>(digit)];

            if (std::abs(point[coordinate]) != Scalar{1}) {
                belongs_to_central_cube = false;
            }
        }

        if (belongs_to_central_cube) {
            continue;
        }

        nodes.set(next++, point);
    }

    if (next != total_node_count) {
        throw std::logic_error(
            "Cubical-grid fixture generated the wrong number of nodes.");
    }

    return Mesh(std::move(nodes), make_database());
}

std::vector<Point> expected_direct_neighbour_vertices() {
    std::vector<Point> result;
    result.reserve(static_cast<std::size_t>(2 * Dimension));

    for (int coordinate = 0; coordinate < Dimension; ++coordinate) {
        Point negative = Point::Zero();
        negative[coordinate] = Scalar{-2};
        result.push_back(negative);

        Point positive = Point::Zero();
        positive[coordinate] = Scalar{2};
        result.push_back(positive);
    }

    return result;
}

std::optional<std::size_t> matching_expected_vertex(
    const Point& point,
    const std::vector<Point>& expected) {

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (same_point(point, expected[i])) {
            return i;
        }
    }

    return std::nullopt;
}

// ============================================================================
// Test 1: general position + ClassicRayCaster
// ============================================================================

void test_classic_raycast_on_general_vertex() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] general position: EdgeIterator + ClassicRayCaster\n";
    std::cout << "============================================================\n";

    Mesh mesh = make_general_raycast_mesh();
    const Sigma origin_sigma = first_indices(general_base_node_count);
    const Point origin = zero_point();

    const auto address = mesh.store_vertex(origin, origin_sigma);
    check(address != 0, "store known general-position vertex at r = 0");

    const auto stored = load_vertex(mesh, Index{0});

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    ClassicParameters parameters;
    parameters.variance_tolerance = Scalar{7e-14};

    auto raycaster = highvoronoi::make_raycaster(
        tree,
        parameters);

    check(
        raycaster.parameters().variance_tolerance ==
            parameters.variance_tolerance,
        "ClassicRayCaster is constructed through RaycastParameters");

    check(
        raycaster.verify_vertex(stored.sigma, stored.position),
        "the prescribed origin is a valid general-position Voronoi vertex");

    Iterator iterator(raycaster.extended_nodes());
    const std::vector<Index> no_boundary_neighbours;

    std::size_t raycast_count = 0;
    std::vector<Point> endpoints;

    for (Index cell = Index{0};
         cell < general_base_node_count;
         ++cell) {

        std::cout << "\n  [CELL " << cell << "]\n";

        raycaster.activate_cell(
            cell,
            no_boundary_neighbours);

        iterator.reset(
            stored.sigma,
            stored.position,
            cell,
            OnQueueEdges{});

        while (const auto edge_view = iterator.next()) {
            const Candidate edge(*edge_view);
            print_edge(edge);

            Sigma endpoint_sigma(
                edge.full_indices.begin(),
                edge.full_indices.end());

            const auto result = raycaster.cast(
                endpoint_sigma,
                stored.position,
                edge.direction,
                edge.indices,
                stored.sigma,
                highvoronoi::RayCastUsage::WalkRay,
                edge.cycle_error);

            print_raycast(result, endpoint_sigma);
            ++raycast_count;

            check(
                result.status == highvoronoi::RayCastStatus::Finite,
                "classic raycast finds a finite adjacent vertex");

            if (result.status != highvoronoi::RayCastStatus::Finite) {
                continue;
            }

            check(
                result.t > Scalar{0},
                "classic raycast moves forward along the edge");

            check(
                endpoint_sigma.size() ==
                    static_cast<std::size_t>(Dimension + 1),
                "general endpoint has exactly d+1 generators");

            check(
                raycaster.verify_vertex(
                    endpoint_sigma,
                    result.position),
                "classic raycast endpoint is a valid Voronoi vertex");

            bool duplicate = false;
            for (const Point& previous : endpoints) {
                if (same_point(previous, result.position)) {
                    duplicate = true;
                    break;
                }
            }

            check(
                !duplicate,
                "different general edges reach different adjacent vertices");

            endpoints.push_back(result.position);
        }
    }

    check(
        raycast_count ==
            static_cast<std::size_t>(Dimension + 1),
        "all d+1 edges of the general vertex were raycast exactly once");
}

// ============================================================================
// Test 2: cubical grid + InRangeRayCaster
// ============================================================================

void test_inrange_raycast_on_degenerate_cube() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] cubical grid: EdgeIterator + InRangeRayCaster\n";
    std::cout << "============================================================\n";

    constexpr Index total_grid_nodes =
        integer_power(Index{4}, Dimension);
    constexpr Index total_grid_cubes =
        integer_power(Index{3}, Dimension);

    std::cout << "  grid generators: " << total_grid_nodes << '\n';
    std::cout << "  grid cubes:      " << total_grid_cubes << '\n';
    std::cout << "  expected direct neighbours: "
              << 2 * Dimension << '\n';

    Mesh mesh = make_cubical_grid_mesh();
    const Sigma origin_sigma =
        first_indices(central_cube_node_count);
    const Point origin = zero_point();

    const auto address = mesh.store_vertex(origin, origin_sigma);
    check(address != 0, "store central degenerate cube vertex at r = 0");

    const auto stored = load_vertex(mesh, Index{0});

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    InRangeParameters parameters;
    parameters.variance_tolerance = Scalar{9e-14};

    auto raycaster = highvoronoi::make_raycaster(
        tree,
        parameters);

    check(
        raycaster.parameters().variance_tolerance ==
            parameters.variance_tolerance,
        "InRangeRayCaster is constructed through RaycastParameters");

    check(
        raycaster.verify_vertex(stored.sigma, stored.position),
        "the prescribed central cube center is a valid degenerate vertex");

    Iterator iterator(raycaster.extended_nodes());
    const std::vector<Index> no_boundary_neighbours;

    const std::vector<Point> expected =
        expected_direct_neighbour_vertices();
    std::vector<bool> seen(expected.size(), false);

    std::size_t raycast_count = 0;

    for (Index cell = Index{0};
         cell < central_cube_node_count;
         ++cell) {

        std::cout << "\n  [CELL " << cell << "]\n";

        raycaster.activate_cell(
            cell,
            no_boundary_neighbours);

        iterator.reset(
            stored.sigma,
            stored.position,
            cell,
            OnQueueEdges{});

        while (const auto edge_view = iterator.next()) {
            const Candidate edge(*edge_view);
            print_edge(edge);

            Sigma endpoint_sigma(
                edge.full_indices.begin(),
                edge.full_indices.end());

            const auto result = raycaster.cast(
                endpoint_sigma,
                stored.position,
                edge.direction,
                edge.indices,
                stored.sigma,
                highvoronoi::RayCastUsage::WalkRay,
                edge.cycle_error);

            print_raycast(result, endpoint_sigma);
            ++raycast_count;

            check(
                result.status == highvoronoi::RayCastStatus::Finite,
                "in-range raycast finds a finite adjacent cube vertex");

            if (result.status != highvoronoi::RayCastStatus::Finite) {
                continue;
            }

            check(
                result.t > Scalar{0},
                "in-range raycast moves forward along the cube edge");

            check(
                endpoint_sigma.size() ==
                    static_cast<std::size_t>(central_cube_node_count),
                "degenerate endpoint contains exactly 2^d generators");

            check(
                raycaster.verify_vertex(
                    endpoint_sigma,
                    result.position),
                "in-range endpoint is a valid degenerate Voronoi vertex");

            const auto match =
                matching_expected_vertex(
                    result.position,
                    expected);

            check(
                match.has_value(),
                "raycast endpoint is one of the expected +/-2 e_i neighbours");

            if (!match) {
                continue;
            }

            check(
                !seen[*match],
                "the direct neighbouring cube vertex is reached only once");

            seen[*match] = true;
        }
    }

    check(
        raycast_count ==
            static_cast<std::size_t>(2 * Dimension),
        "EdgeIterator and raycast together traverse exactly 2d cube edges");

    check(
        std::all_of(
            seen.begin(),
            seen.end(),
            [](bool value) { return value; }),
        "raycasts find all 2d direct neighbouring cube centers");
}

} // namespace

int main() {
    test_classic_raycast_on_general_vertex();
    test_inrange_raycast_on_degenerate_cube();

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}
