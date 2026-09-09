
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/validation/mesh_validation.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
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
inline constexpr Index NodeCount = Index{200};
inline constexpr std::size_t ParallelThreads = 3;

using DatabaseParameters = highvoronoi::DataBaseParams<Scalar, Index>;
using Database = highvoronoi::HVDataBase<highvoronoi::ReadWriteLock, DatabaseParameters, Dimension>;
using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::VertexPoint;
using Boundary = Mesh::BoundaryType;
using ClassicParameters =
    highvoronoi::RaycastParameters<highvoronoi::ClassicRaycast, Scalar>;
using EdgeParameters = highvoronoi::EdgeBufferParams<>;

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

std::shared_ptr<Database> make_database() {
    constexpr std::size_t database_hash_capacity = 32768;
    constexpr std::size_t storage_block_units = 32768;

    return std::make_shared<Database>(
        storage_block_units,
        DatabaseParameters{
            highvoronoi::DirectHash{database_hash_capacity}});
}

Boundary centered_cube() {
    Point dimensions = Point::Constant(Scalar{8});
    Point offset = Point::Constant(Scalar{-4});
    return Boundary::cuboid(
        dimensions,
        offset,
        std::vector<Index>{});
}

Nodes make_random_nodes() {
    Nodes nodes(NodeCount);

    std::mt19937_64 random(0x4856504152414c4cULL);
    std::uniform_real_distribution<Scalar> coordinate(
        Scalar{-2.8},
        Scalar{2.8});

    for (Index index = Index{0}; index < NodeCount; ++index) {
        Point point;

        bool accepted = false;
        while (!accepted) {
            for (int d = 0; d < Dimension; ++d) {
                point[d] = coordinate(random);
            }

            accepted = true;
            for (Index previous = Index{0}; previous < index; ++previous) {
                if ((point - nodes[previous]).norm() < Scalar{0.12}) {
                    accepted = false;
                    break;
                }
            }
        }

        nodes.set(index, point);
    }

    return nodes;
}

Mesh make_mesh(const Nodes& nodes) {
    return Mesh(
        Nodes(nodes),
        centered_cube(),
        make_database());
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

template <class MeshThreading, class CastThreading>
double compute_mesh(
    Mesh& mesh,
    MeshThreading mesh_threading,
    CastThreading cast_threading) {

    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    ClassicParameters raycast_parameters;
    raycast_parameters.variance_tolerance = Scalar{1e-12};
    auto raycaster = highvoronoi::make_raycaster(
        tree,
        raycast_parameters);

    using RayCaster = decltype(raycaster);
    using Compute = highvoronoi::ComputeVoronoi<
        Mesh,
        RayCaster,
        MeshThreading,
        CastThreading,
        DatabaseParameters,
        EdgeParameters>;

    const DatabaseParameters queue_parameters{
        highvoronoi::DirectHash{2048}};
    const EdgeParameters edge_parameters{
        highvoronoi::DirectHash{4096}};

    Compute compute(
        mesh,
        raycaster,
        std::move(mesh_threading),
        std::move(cast_threading),
        std::nullopt,
        queue_parameters,
        edge_parameters);

    const auto start = std::chrono::steady_clock::now();
    compute.compute();
    const auto stop = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(stop - start).count();
}

void print_result(
    std::string_view name,
    const Mesh& mesh,
    double seconds) {
    std::cout << "\n" << name << '\n';
    std::cout << "    primary vertices: "
              << unique_vertex_count(mesh) << '\n';
    std::cout << "    compute time:     "
              << std::fixed << std::setprecision(3)
              << seconds << " s\n";
}

template <class Report>
void print_comparison(
    std::string_view name,
    const Report& report) {
    std::cout << "\n[comparison: " << name << "]\n";
    std::cout << std::scientific << std::setprecision(6);
    report.print_summary(std::cout);
}

} // namespace

int main() {
    std::cout << "============================================================\n";
    std::cout << "HighVoronoi 3D parallel smoke test\n";
    std::cout << "nodes: " << NodeCount
              << ", parallel threads: " << ParallelThreads << '\n';
    std::cout << "============================================================\n";

    const Nodes original_nodes = make_random_nodes();

    Mesh serial_mesh = make_mesh(original_nodes);
    Mesh mesh_threaded = make_mesh(original_nodes);
    Mesh worker_threaded = make_mesh(original_nodes);

    std::cout << "\n[1/3] serial reference\n";
    const double serial_seconds = compute_mesh(
        serial_mesh,
        highvoronoi::SingleThread{},
        highvoronoi::SingleThread{});
    print_result("serial reference", serial_mesh, serial_seconds);

    std::cout << "\n[2/3] MeshThreading = MultiThread(3), CastThreading = SingleThread\n";
    const double mesh_parallel_seconds = compute_mesh(
        mesh_threaded,
        highvoronoi::MultiThread{ParallelThreads},
        highvoronoi::SingleThread{});
    print_result("mesh-threaded", mesh_threaded, mesh_parallel_seconds);

    std::cout << "\n[3/3] MeshThreading = SingleThread, CastThreading = MultiThread(3)\n";
    const double worker_parallel_seconds = compute_mesh(
        worker_threaded,
        highvoronoi::SingleThread{},
        highvoronoi::MultiThread{ParallelThreads});
    print_result("worker-threaded", worker_threaded, worker_parallel_seconds);

    std::cout << "\n============================================================\n";
    std::cout << "GEOMETRIC VERIFICATION\n";
    std::cout << "============================================================\n";

    const auto serial_verification = highvoronoi::verify_mesh(
        serial_mesh,
        Scalar{1e-16},
        true);
    const auto mesh_parallel_verification = highvoronoi::verify_mesh(
        mesh_threaded,
        Scalar{1e-16},
        true);
    const auto worker_parallel_verification = highvoronoi::verify_mesh(
        worker_threaded,
        Scalar{1e-16},
        true);

    check(serial_verification.valid(),
          "serial reference mesh is geometrically consistent");
    check(mesh_parallel_verification.valid(),
          "mesh-threaded result is geometrically consistent");
    check(worker_parallel_verification.valid(),
          "worker-threaded result is geometrically consistent");

    std::cout << "\n============================================================\n";
    std::cout << "MESH COMPARISON\n";
    std::cout << "============================================================\n";

    const auto serial_vs_mesh = highvoronoi::compare_meshes(
        serial_mesh,
        mesh_threaded,
        Scalar{0},
        true);
    print_comparison("serial vs mesh-threaded", serial_vs_mesh);
    check(serial_vs_mesh.equal(),
          "mesh-threaded result equals serial reference");

    const auto serial_vs_worker = highvoronoi::compare_meshes(
        serial_mesh,
        worker_threaded,
        Scalar{0},
        true);
    print_comparison("serial vs worker-threaded", serial_vs_worker);
    check(serial_vs_worker.equal(),
          "worker-threaded result equals serial reference");

    const auto mesh_vs_worker = highvoronoi::compare_meshes(
        mesh_threaded,
        worker_threaded,
        Scalar{0},
        true);
    print_comparison("mesh-threaded vs worker-threaded", mesh_vs_worker);
    check(mesh_vs_worker.equal(),
          "both parallel modes produce the same mesh");

    check(
        unique_vertex_count(serial_mesh) == unique_vertex_count(mesh_threaded) &&
        unique_vertex_count(serial_mesh) == unique_vertex_count(worker_threaded),
        "all three runs contain the same number of primary vertices");

    std::cout << "\n============================================================\n";
    std::cout << "performed checks: " << performed_checks << '\n';
    std::cout << "failed checks:    " << failed_checks << '\n';
    std::cout << "============================================================\n";

    return failed_checks == 0 ? 0 : 1;
}


