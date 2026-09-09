#pragma once

/**
 * @file voronoi_api.hpp
 * @brief Level-1 and Level-2 ordinary Voronoi convenience API.
 *
 * Level 1:
 *   auto mesh = voronoi_mesh<3>(points, count, boundary);
 *   compute(mesh);
 *   refine(mesh, new_points, new_count);
 *   remove(mesh, {2, 5});
 *
 * Level 2 uses VoronoiConfig to select ray casting and hash policies.
 * Level 3 is the unchanged VoronoiMesh/SearchTree/RayCaster/ComputeVoronoi API.
 */

#include <highvoronoi/api_common.hpp>
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/refine_voronoi.hpp>
#include <highvoronoi/algorithm/incremental/remove_voronoi.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

template <int Dim, class ConfigT>
class ApiVoronoiMesh final {
public:
    using Config = ConfigT;
    using Threading = typename Config::Threading;
    using Index = std::uint32_t;
    using Scalar = double;

    using DatabaseParameters = DataBaseParams<
        Scalar,
        Index,
        typename Config::HashGenerator,
        typename Config::DatabaseHashContainer>;
    using Database = HVDataBase<
        typename Threading::RWLock,
        DatabaseParameters,
        Dim>;
    using Mesh = VoronoiMesh<Scalar, Dim, Database>;

    using Point = typename Mesh::NodePoint;
    using NodePoint = typename Mesh::NodePoint;
    using VertexPoint = typename Mesh::VertexPoint;
    using BoundaryType = typename Mesh::BoundaryType;
    using Sigma = typename Mesh::Sigma;
    using Address = typename Mesh::Address;

    using QueueParameters = DataBaseParams<
        Scalar,
        Index,
        typename Config::HashGenerator,
        DirectHash>;
    using EdgeParameters = EdgeBufferParams<
        typename Config::HashGenerator,
        typename Config::EdgeHashContainer>;

    ApiVoronoiMesh(
        typename Mesh::InternalNodes nodes,
        BoundaryType boundary,
        Config config)
        : config_(std::move(config)),
          runtime_dimension_(nodes.dimension()),
          mesh_(
              std::move(nodes),
              std::move(boundary),
              std::make_shared<Database>(
                  config_.database_block_units,
                  DatabaseParameters{config_.database_hash},
                  static_cast<std::size_t>(runtime_dimension_))) {}

    ApiVoronoiMesh(const ApiVoronoiMesh&) = delete;
    ApiVoronoiMesh& operator=(const ApiVoronoiMesh&) = delete;
    ApiVoronoiMesh(ApiVoronoiMesh&&) = delete;
    ApiVoronoiMesh& operator=(ApiVoronoiMesh&&) = delete;

    [[nodiscard]] Mesh& level3_mesh() noexcept { return mesh_; }
    [[nodiscard]] const Mesh& level3_mesh() const noexcept { return mesh_; }

    [[nodiscard]] Config& config() noexcept { return config_; }
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t thread_count() const noexcept {
        return config_.thread_count;
    }

    [[nodiscard]] Index size() const noexcept { return mesh_.size(); }
    [[nodiscard]] bool empty() const noexcept { return size() == Index{0}; }
    [[nodiscard]] Index dimension() const noexcept { return mesh_.dimension(); }
    [[nodiscard]] const auto& nodes() const noexcept { return mesh_.nodes(); }
    [[nodiscard]] const BoundaryType& boundary() const noexcept {
        return mesh_.boundary();
    }

    [[nodiscard]] auto primary_vertices(Index cell) const {
        return mesh_.primary_vertices(cell);
    }
    [[nodiscard]] auto secondary_vertices(Index cell) const {
        return mesh_.secondary_vertices(cell);
    }
    [[nodiscard]] auto vertices(Index cell) const {
        return mesh_.vertices(cell);
    }
    [[nodiscard]] auto infinite_edges() const {
        return mesh_.infinite_edges();
    }

private:
    Config config_;
    Index runtime_dimension_{};
    Mesh mesh_;
};

namespace api_detail {

template <class ApiMesh>
[[nodiscard]] typename ApiMesh::QueueParameters make_queue_parameters(
    const ApiMesh& mesh) {
    return typename ApiMesh::QueueParameters{mesh.config().queue_hash};
}

template <class ApiMesh>
[[nodiscard]] typename ApiMesh::EdgeParameters make_edge_parameters(
    const ApiMesh& mesh) {
    return typename ApiMesh::EdgeParameters{mesh.config().edge_hash};
}

template <class ApiMesh>
[[nodiscard]] std::size_t run_ordinary_compute(
    ApiMesh& api_mesh,
    std::optional<std::size_t> thread_count,
    bool verbose) {
    using Config = typename ApiMesh::Config;
    using Threading = typename Config::Threading;
    using EngineMesh = typename ApiMesh::Mesh;
    using QueueParameters = typename ApiMesh::QueueParameters;
    using EdgeParameters = typename ApiMesh::EdgeParameters;

    EngineMesh& mesh = api_mesh.level3_mesh();
    const Config& config = api_mesh.config();
    const std::size_t threads = resolve_thread_count<Threading>(
        config.thread_count,
        thread_count);

    auto tree = geometry::make_search_tree(mesh, config.search);
    auto raycaster = make_raycaster(tree, config.raycast);

    using RayCaster = decltype(raycaster);
    using Compute = ComputeVoronoi<
        EngineMesh,
        RayCaster,
        Threading,
        SingleThread,
        QueueParameters,
        EdgeParameters>;

    Compute operation(
        mesh,
        raycaster,
        make_threading<Threading>(threads),
        SingleThread{},
        std::nullopt,
        make_queue_parameters(api_mesh),
        make_edge_parameters(api_mesh));
    operation.compute(verbose || config.verbose);
    return operation.new_vertex_count();
}


} // namespace api_detail

/** Level 2: construct from an explicit VoronoiConfig. */
template <
    int Dim,
    class SourceScalar,
    class SourceIndex,
    class Config,
    std::enable_if_t<IsVoronoiConfigV<Config>, int> = 0>
[[nodiscard]] auto voronoi_mesh(
    const double* points,
    std::size_t point_count,
    const Boundary<Dim, SourceScalar, SourceIndex>& boundary,
    Config config,
    std::size_t dimension = (Dim == Dynamic ? 0 : static_cast<std::size_t>(Dim))) {
    using ApiMesh = ApiVoronoiMesh<Dim, std::decay_t<Config>>;
    using Index = typename ApiMesh::Index;

    const std::size_t resolved_dimension =
        api_detail::resolve_dimension<Dim>(dimension);
    auto nodes = api_detail::make_nodes<Dim, Index>(
        points,
        point_count,
        resolved_dimension);
    auto converted_boundary = api_detail::convert_boundary<
        Dim,
        SourceScalar,
        SourceIndex,
        Index>(boundary, resolved_dimension);

    return ApiMesh(
        std::move(nodes),
        std::move(converted_boundary),
        std::move(config));
}

/** Level 1: all policies except dimension and optional threading are automatic. */
template <
    int Dim,
    class Threading = SingleThread,
    class SourceScalar,
    class SourceIndex,
    std::enable_if_t<api_detail::is_supported_threading_v<Threading>, int> = 0>
[[nodiscard]] auto voronoi_mesh(
    const double* points,
    std::size_t point_count,
    const Boundary<Dim, SourceScalar, SourceIndex>& boundary,
    Threading threading = Threading{},
    std::size_t dimension = (Dim == Dynamic ? 0 : static_cast<std::size_t>(Dim))) {
    const std::size_t resolved_dimension =
        api_detail::resolve_dimension<Dim>(dimension);
    auto config = make_voronoi_config<Threading>(
        point_count,
        resolved_dimension,
        threading);
    return voronoi_mesh<Dim>(
        points,
        point_count,
        boundary,
        std::move(config),
        resolved_dimension);
}

/** Runtime-dimension Level 1 shorthand for a serial convenience mesh. */
template <
    int Dim,
    class SourceScalar,
    class SourceIndex,
    std::enable_if_t<Dim == Dynamic, int> = 0>
[[nodiscard]] auto voronoi_mesh(
    const double* points,
    std::size_t point_count,
    const Boundary<Dim, SourceScalar, SourceIndex>& boundary,
    std::size_t dimension) {
    return voronoi_mesh<Dim>(
        points,
        point_count,
        boundary,
        SingleThread{},
        dimension);
}

/** Compute with the thread count stored by the Level-1/2 mesh. */
template <int Dim, class Config>
[[nodiscard]] std::size_t compute(ApiVoronoiMesh<Dim, Config>& mesh) {
    return api_detail::run_ordinary_compute(mesh, std::nullopt, false);
}

/** Override the stored mesh-thread count for this call (ignored by SingleThread). */
template <int Dim, class Config>
[[nodiscard]] std::size_t compute(
    ApiVoronoiMesh<Dim, Config>& mesh,
    std::size_t thread_count) {
    return api_detail::run_ordinary_compute(mesh, thread_count, false);
}

template <int Dim, class Config>
[[nodiscard]] auto refine(
    ApiVoronoiMesh<Dim, Config>& api_mesh,
    const double* points,
    std::size_t point_count,
    std::optional<std::size_t> thread_count = std::nullopt) {
    using ApiMesh = ApiVoronoiMesh<Dim, Config>;
    using Threading = typename Config::Threading;
    using Refiner = RefineVoronoi<
        typename ApiMesh::Mesh,
        geometry::KDSearch,
        typename Config::RayParameters,
        Threading,
        SingleThread,
        typename ApiMesh::QueueParameters,
        typename ApiMesh::EdgeParameters>;

    if (point_count != 0 && points == nullptr) {
        throw std::invalid_argument("refine() received a null points pointer.");
    }

    std::vector<typename ApiMesh::NodePoint> new_nodes;
    new_nodes.reserve(point_count);
    const std::size_t dimension = static_cast<std::size_t>(api_mesh.dimension());
    for (std::size_t i = 0; i < point_count; ++i) {
        typename ApiMesh::NodePoint point;
        if constexpr (Dim == Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension));
        }
        for (std::size_t d = 0; d < dimension; ++d) {
            point[static_cast<Eigen::Index>(d)] = points[i * dimension + d];
        }
        new_nodes.push_back(std::move(point));
    }

    const std::size_t threads = api_detail::resolve_thread_count<Threading>(
        api_mesh.config().thread_count,
        thread_count);
    Refiner operation(
        api_mesh.level3_mesh(),
        std::move(new_nodes),
        api_mesh.config().search,
        api_mesh.config().raycast,
        api_detail::make_threading<Threading>(threads),
        SingleThread{},
        api_detail::make_queue_parameters(api_mesh),
        api_detail::make_edge_parameters(api_mesh));
    (void)operation.compute(api_mesh.config().verbose);
    return operation.report();
}

template <int Dim, class Config>
[[nodiscard]] auto remove(
    ApiVoronoiMesh<Dim, Config>& api_mesh,
    const std::vector<typename ApiVoronoiMesh<Dim, Config>::Index>& public_nodes,
    std::optional<std::size_t> thread_count = std::nullopt) {
    using ApiMesh = ApiVoronoiMesh<Dim, Config>;
    using Threading = typename Config::Threading;
    using Remover = RemoveVoronoi<
        typename ApiMesh::Mesh,
        geometry::KDSearch,
        typename Config::RayParameters,
        Threading,
        SingleThread,
        typename ApiMesh::QueueParameters,
        typename ApiMesh::EdgeParameters>;

    const std::size_t threads = api_detail::resolve_thread_count<Threading>(
        api_mesh.config().thread_count,
        thread_count);
    Remover operation(
        api_mesh.level3_mesh(),
        public_nodes,
        api_mesh.config().search,
        api_mesh.config().raycast,
        api_detail::make_threading<Threading>(threads),
        SingleThread{},
        api_detail::make_queue_parameters(api_mesh),
        api_detail::make_edge_parameters(api_mesh));
    (void)operation.compute(api_mesh.config().verbose);
    return operation.report();
}

template <int Dim, class Config>
[[nodiscard]] auto remove(
    ApiVoronoiMesh<Dim, Config>& api_mesh,
    std::initializer_list<typename ApiVoronoiMesh<Dim, Config>::Index> public_nodes,
    std::optional<std::size_t> thread_count = std::nullopt) {
    return remove(
        api_mesh,
        std::vector<typename ApiVoronoiMesh<Dim, Config>::Index>(public_nodes),
        thread_count);
}

} // namespace highvoronoi
