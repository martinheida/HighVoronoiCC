
#include <highvoronoi/parameters.hpp>
#include <highvoronoi/mesh/engine/compute_mesh.hpp>
#include <highvoronoi/mesh/engine/cuboid_mesh_engine.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>

#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
using CuboidEngine = highvoronoi::CuboidMeshEngine<Scalar, Scalar, Index, 3>;
using Mesh = highvoronoi::ComputeMesh<CuboidEngine>;
using Point = CuboidEngine::NodePoint;
using CountPoint = CuboidEngine::CountPoint;

std::size_t performed_checks = 0;
std::size_t failed_checks = 0;

template<class T, class = void>
struct has_neighbour_address : std::false_type {};

template<class T>
struct has_neighbour_address<
    T,
    std::void_t<decltype(std::declval<T&>().neighbour_address(Index{}))>>
    : std::true_type {};


void check(bool condition, std::string_view description) {
    ++performed_checks;
    if (condition) {
        std::cout << "    [OK]   " << description << '\n';
    } else {
        ++failed_checks;
        std::cerr << "    [FAIL] " << description << '\n';
    }
}

Point point(Scalar x, Scalar y, Scalar z) {
    Point p;
    p << x, y, z;
    return p;
}

CountPoint counts(Index x, Index y, Index z) {
    CountPoint p;
    p << x, y, z;
    return p;
}

void test_compute_mesh_facade() {
    std::cout << "\n[TEST] ComputeMesh over CuboidMeshEngine\n";

    CuboidEngine engine(
        point(-1.0, -1.0, -1.0),
        point(1.0, 1.0, 1.0),
        counts(3, 3, 3));
    Mesh mesh(std::move(engine));

    static_assert(std::is_same_v<typename Mesh::Engine, CuboidEngine>,
                  "ComputeMesh must bind the concrete engine type at compile time.");

    check(mesh.size() == Index{27},
          "ComputeMesh exposes all 27 computed ordinary nodes");
    check(!mesh.provides_complete_infinite_edges(),
          "facade forwards the engine infinite-edge completeness capability");

    std::vector<Index> neighbours;
    static_assert(!has_neighbour_address<Mesh>::value,
                  "ComputeMesh must expose neighbour geometry without persistent addresses.");
    check(mesh.neighbours(Index{13}, neighbours) &&
              neighbours == std::vector<Index>({4,10,12,14,16,22}),
          "ComputeMesh reads Cuboid native neighbours directly from the engine");

    std::size_t primary = 0;
    std::size_t secondary = 0;
    for (Index node = 0; node < mesh.size(); ++node) {
        for ([[maybe_unused]] const auto& vertex : mesh.primary_vertices(node)) {
            ++primary;
        }
        for ([[maybe_unused]] const auto& vertex : mesh.secondary_vertices(node)) {
            ++secondary;
        }
    }

    check(primary == 8,
          "ComputeMesh exposes all eight analytic finite vertices exactly once as primary");
    check(secondary == 8 * 7,
          "ComputeMesh exposes seven secondary occurrences per 3D grid vertex");

    const auto verification = highvoronoi::verify_mesh(
        mesh,
        Scalar{1e-20},
        true);
    check(verification.valid(),
          "all finite computed vertices are geometrically Voronoi-consistent");
    check(verification.checked_vertex_occurrences == 8 * 8,
          "verification visits all primary and secondary finite-vertex occurrences");

    auto primary_zero = mesh.primary_vertices(Index{0});
    const auto first = primary_zero.begin();
    check(first != primary_zero.end(),
          "node 0 exposes its computed primary vertex before deletion");
    if (first != primary_zero.end()) {
        const auto address = first->address;
        check(mesh.erase_vertex(address),
              "AbstractMesh erase_vertex routes through ComputeMesh into the engine");
        check(mesh.primary_vertices(Index{0}).empty(),
              "deleted computed vertex disappears through the normal mesh iterator");

        check(!mesh.dirty(Index{13}),
              "ComputeMesh has no persistent neighbour dirty state for immutable engine topology");
        neighbours.clear();
        check(mesh.neighbours(Index{13}, neighbours) &&
                  neighbours == std::vector<Index>({4,10,12,14,16,22}),
              "native engine neighbour topology remains directly readable after vertex invalidation");
    }

    // This is intentionally not a completeness assertion: the current cuboid
    // engine does not enumerate infinite edges of outer cells.
}

} // namespace

int main() {
    test_compute_mesh_facade();
    std::cout << "\nperformed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    return failed_checks == 0 ? 0 : 1;
}



