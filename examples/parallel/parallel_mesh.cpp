/**
 * @file parallel_mesh.cpp
 * @brief Parallel Voronoi construction through several mesh branches.
 *
 * MeshThreading = MultiThread(4)
 * CastThreading = SingleThread
 *
 * ComputeVoronoi partitions the public cells into ReorderedMeshView branches.
 * Every branch owns its SystematicVoronoi state while all branches communicate
 * newly discovered vertices through the top-level ComputeVoronoi coordinator.
 */
#include <highvoronoi/detail/hvdatabase.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/mesh_validation.hpp>
#include <highvoronoi/geometry/raycaster.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Scalar = double;
using Index = std::uint32_t;
inline constexpr int Dimension = 3;
inline constexpr Index NodeCount = Index{100};

// ---------------------------------------------------------------------------
// Hash configuration
// ---------------------------------------------------------------------------
//
// The persistent database and the cell-local EdgeHash use StaticHash<16>.
// Every sub-table owns its own lock, reducing lock contention during parallel
// construction.  The cell-local vertex queue deliberately stays DirectHash:
// VoronoiVertexQueue currently owns one outer queue lock around claim/pop/reset,
// so sharding its internal QueueHash would not remove that outer contention.
//
using HashGenerator = highvoronoi::FNV64_128HashGenerator;

using DatabaseParameters = highvoronoi::DataBaseParams<
    Scalar,
    Index,
    HashGenerator,
    highvoronoi::StaticHash<16>>;

using QueueParameters = highvoronoi::DataBaseParams<
    Scalar,
    Index,
    HashGenerator,
    highvoronoi::DirectHash>;

using EdgeParameters = highvoronoi::EdgeBufferParams<
    HashGenerator,
    highvoronoi::StaticHash<16>>;

// Any parallel construction requires a thread-safe persistent database.
using Database = highvoronoi::HVDataBase<
    highvoronoi::ReadWriteLock,
    DatabaseParameters>;

using Mesh = highvoronoi::VoronoiMesh<Scalar, Dimension, Database>;
using Nodes = Mesh::InternalNodes;
using Point = Mesh::VertexPoint;
using Boundary = Mesh::BoundaryType;

std::shared_ptr<Database> make_database() {
    constexpr std::size_t storage_block_units = 32768;

    return std::make_shared<Database>(
        storage_block_units,
        DatabaseParameters{
            highvoronoi::StaticHash<16>{1024}});
}

Boundary make_boundary() {
    Point dimensions = Point::Constant(Scalar{8});
    Point offset = Point::Constant(Scalar{-4});

    // Explicit empty periodic-axis list = ordinary bounded cuboid.
    return Boundary::cuboid(
        dimensions,
        offset,
        std::vector<Index>{});
}

Nodes make_nodes() {
    Nodes nodes(NodeCount);

    // Deterministic point cloud: every run uses exactly the same generators.
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
int run_example(
    MeshThreading mesh_threading,
    CastThreading cast_threading) {

    Mesh mesh(
        make_nodes(),
        make_boundary(),
        make_database());

    // KDSearch{leaf_max_size, build_thread_count}.
    // KD-tree build threading is independent from Voronoi threading.
    auto tree = highvoronoi::geometry::make_search_tree(
        mesh,
        highvoronoi::geometry::KDSearch{8, 1});

    highvoronoi::RaycastParameters<
        highvoronoi::ClassicRaycast,
        Scalar> raycast_parameters{};
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
        QueueParameters,
        EdgeParameters>;

    const QueueParameters queue_parameters{
        highvoronoi::DirectHash{2048}};

    const EdgeParameters edge_parameters{
        highvoronoi::StaticHash<16>{512}};

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

    const double seconds =
        std::chrono::duration<double>(stop - start).count();

    std::cout << "primary vertices: "
              << unique_vertex_count(mesh) << '\n';
    std::cout << "compute time:     "
              << std::fixed << std::setprecision(3)
              << seconds << " s\n";

    const auto verification = highvoronoi::verify_mesh(
        mesh,
        Scalar{1e-16},
        true);

    std::cout << "mesh valid:       "
              << std::boolalpha
              << verification.valid() << '\n';

    return verification.valid() ? 0 : 1;
}

} // namespace

int main() {
    constexpr std::size_t MeshThreads = 4;

    std::cout << "HighVoronoi parallel mesh example\n";
    std::cout << "MeshThreading: MultiThread(" << MeshThreads << ")\n";
    std::cout << "CastThreading: SingleThread\n\n";

    return run_example(
        highvoronoi::MultiThread{MeshThreads},
        highvoronoi::SingleThread{});
}
