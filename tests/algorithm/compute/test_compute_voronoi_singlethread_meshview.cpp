
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/detail/hvview.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/mesh/mesh_view.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

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
inline constexpr int Dimension = 3;
inline constexpr Index NodeCount = Index{10};

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::EmptyLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::VertexPoint;
using Boundary = Mesh::BoundaryType;
using ClassicParameters =
    highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast, Scalar>;

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

std::shared_ptr<Database> make_database(std::size_t capacity = 1024) {
    return std::make_shared<Database>(
        2048,
        DatabaseParameters{highvoronoi::DirectHash{capacity}});
}

Boundary centered_cube() {
    Point dimensions = Point::Constant(Scalar{6});
    Point offset = Point::Constant(Scalar{-3});
    return Boundary::cuboid(
        dimensions,
        offset,
        std::vector<Index>{});
}

Nodes make_random_nodes() {
    Nodes nodes(NodeCount);

    std::mt19937_64 random(0x4856475249443344ULL);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{-1.6},
        Scalar{1.6});

    for (Index index = Index{0}; index < NodeCount; ++index) {
        Point point;

        // Avoid nearly coincident generators. The seed makes the fixture fully
        // reproducible while retaining an ordinary random general-position set.
        bool accepted = false;
        while (!accepted) {
            for (int d = 0; d < Dimension; ++d) {
                point[d] = coordinate(random);
            }

            accepted = true;
            for (Index previous = Index{0}; previous < index; ++previous) {
                if ((point - nodes[previous]).norm() < Scalar{0.35}) {
                    accepted = false;
                    break;
                }
            }
        }

        nodes.set(index, point);
    }

    return nodes;
}

Mesh make_mesh(Nodes nodes) {
    return Mesh(
        std::move(nodes),
        centered_cube(),
        make_database());
}

template <class MeshT>
void compute_serial(MeshT& mesh) {
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{4, 1});

    ClassicParameters parameters;
    parameters.variance_tolerance = Scalar{1e-12};
    auto raycaster = highvoronoi::make_raycaster(tree, parameters);

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        MeshT,
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

template <class MeshT>
std::size_t unique_vertex_count(const MeshT& mesh) {
    std::unordered_set<typename MeshT::Address> addresses;
    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        for (const auto& vertex : mesh.primary_vertices(cell)) {
            addresses.insert(vertex.address);
        }
    }
    return addresses.size();
}

void test_independent_equal_meshes_and_difference_detection() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] SingleThread 3D: independent meshes and comparison\n";
    std::cout << "============================================================\n";

    const Nodes original_nodes = make_random_nodes();

    Mesh reference_mesh = make_mesh(original_nodes);
    Mesh copied_mesh = make_mesh(original_nodes);

    compute_serial(reference_mesh);
    compute_serial(copied_mesh);

    const auto reference_verification = highvoronoi::verify_mesh(
        reference_mesh,
        Scalar{1e-16},
        true);
    const auto copied_verification = highvoronoi::verify_mesh(
        copied_mesh,
        Scalar{1e-16},
        true);

    check(reference_verification.valid(),
          "reference random 3D mesh is geometrically consistent");
    check(copied_verification.valid(),
          "independently computed copy is geometrically consistent");

    const auto equal_comparison = highvoronoi::compare_meshes(
        reference_mesh,
        copied_mesh,
        Scalar{0},
        true);

    std::cout << "\n[comparison statistics: independent identical meshes]\n";
    equal_comparison.print_summary(std::cout);

    check(equal_comparison.equal(),
          "compare_meshes accepts independently computed identical node sets");
    check(unique_vertex_count(reference_mesh) == unique_vertex_count(copied_mesh),
          "independent identical meshes contain the same number of vertices");

    Nodes extended_nodes = original_nodes;
    extended_nodes.resize(NodeCount + Index{1});

    Point extra_point;
    extra_point << Scalar{0.173}, Scalar{-0.619}, Scalar{0.881};
    extended_nodes.set(NodeCount, extra_point);

    Mesh extended_mesh = make_mesh(std::move(extended_nodes));
    compute_serial(extended_mesh);

    const auto extended_verification = highvoronoi::verify_mesh(
        extended_mesh,
        Scalar{1e-16},
        true);
    check(extended_verification.valid(),
          "mesh with one additional generator is geometrically consistent");

    const auto different_comparison = highvoronoi::compare_meshes(
        reference_mesh,
        extended_mesh,
        Scalar{0},
        false);

    std::cout << "\n[comparison statistics: additional generator]\n";
    different_comparison.print_summary(std::cout);

    check(!different_comparison.equal(),
          "compare_meshes rejects a mesh built from an additional generator");
    check(different_comparison.error_count() > 0,
          "different meshes produce at least one comparison error");
}

void test_computation_through_reordered_mesh_view() {
    std::cout << "\n============================================================\n";
    std::cout << "[TEST] SingleThread 3D: ComputeVoronoi through MeshView\n";
    std::cout << "============================================================\n";

    const Nodes original_nodes = make_random_nodes();

    Mesh reference_mesh = make_mesh(original_nodes);
    compute_serial(reference_mesh);

    Mesh inner_mesh = make_mesh(original_nodes);

    // For ten nodes SwitchView(5,9) exchanges the first and last five:
    // outer 0..4 -> inner 5..9, outer 5..9 -> inner 0..4.
    highvoronoi::SwitchView<Index> switch_view(Index{5}, Index{9});
    highvoronoi::ReorderedMeshView outer_mesh(inner_mesh, switch_view);

    check(outer_mesh.wrapped_public_index(Index{0}) == Index{5} &&
              outer_mesh.wrapped_public_index(Index{4}) == Index{9} &&
              outer_mesh.wrapped_public_index(Index{5}) == Index{0} &&
              outer_mesh.wrapped_public_index(Index{9}) == Index{4},
          "SwitchView exchanges the first and last five public nodes");

    compute_serial(outer_mesh);

    const auto inner_verification = highvoronoi::verify_mesh(
        inner_mesh,
        Scalar{1e-16},
        true);
    check(inner_verification.valid(),
          "inner mesh remains geometrically consistent after view computation");

    const auto comparison = highvoronoi::compare_meshes(
        reference_mesh,
        inner_mesh,
        Scalar{0},
        true);

    std::cout << "\n[comparison statistics: MeshView computation]\n";
    comparison.print_summary(std::cout);

    check(comparison.equal(),
          "MeshView computation stores exactly the reference vertices in the inner mesh");
    check(unique_vertex_count(reference_mesh) == unique_vertex_count(inner_mesh),
          "MeshView and direct computation produce the same vertex count");
}

} // namespace

int main() {
    test_independent_equal_meshes_and_difference_detection();
    test_computation_through_reordered_mesh_view();

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}


