#pragma once

/**
 * @file highvoronoi_api.hpp
 * @brief Level-1 and Level-2 HighVoronoi convenience API.
 *
 * The convenience workflow is deliberately symmetric with ordinary Voronoi:
 *
 *   auto mesh = high_voronoi_mesh<3>(points, count, boundary);
 *   compute(mesh);
 *   refine(mesh, added_points, added_count);
 *   remove(mesh, {2, 5});
 *
 * Periodic reference-node closure remains fully internal to ComputeHighVoronoi.
 */

#include <highvoronoi/api_common.hpp>
#include <highvoronoi/storage/hvdatabase.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/algorithm/high_voronoi/compute_high_voronoi.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

template <int Dim, class ConfigT>
class ApiHighVoronoiMesh final {
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
    using Mesh = HighVoronoiMesh<Scalar, Dim, Database>;
    using InputNodes = VoronoiNodes<Scalar, Dim, Index>;

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

    ApiHighVoronoiMesh(
        InputNodes initial_nodes,
        BoundaryType boundary,
        Config config)
        : config_(std::move(config)),
          runtime_dimension_(initial_nodes.dimension()),
          mesh_(
              initial_nodes.dimension(),
              std::move(boundary),
              std::in_place,
              config_.database_block_units,
              DatabaseParameters{config_.database_hash},
              static_cast<std::size_t>(runtime_dimension_)) {
        (void)mesh_.append_visible_nodes(initial_nodes);
    }

    ApiHighVoronoiMesh(const ApiHighVoronoiMesh&) = delete;
    ApiHighVoronoiMesh& operator=(const ApiHighVoronoiMesh&) = delete;
    ApiHighVoronoiMesh(ApiHighVoronoiMesh&&) = delete;
    ApiHighVoronoiMesh& operator=(ApiHighVoronoiMesh&&) = delete;

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
    [[nodiscard]] Index visible_public_count() const noexcept {
        return mesh_.visible_public_count();
    }
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
[[nodiscard]] typename ApiMesh::QueueParameters make_high_queue_parameters(
    const ApiMesh& mesh) {
    return typename ApiMesh::QueueParameters{mesh.config().queue_hash};
}

template <class ApiMesh>
[[nodiscard]] typename ApiMesh::EdgeParameters make_high_edge_parameters(
    const ApiMesh& mesh) {
    return typename ApiMesh::EdgeParameters{mesh.config().edge_hash};
}

template <class ApiMesh>
[[nodiscard]] auto run_high_compute(
    ApiMesh& api_mesh,
    std::optional<std::size_t> thread_count,
    bool verbose) {
    using Config = typename ApiMesh::Config;
    using Threading = typename Config::Threading;
    using Operation = ComputeHighVoronoi<
        typename ApiMesh::Mesh,
        geometry::KDSearch,
        typename Config::RayParameters,
        Threading,
        SingleThread,
        typename ApiMesh::QueueParameters,
        typename ApiMesh::EdgeParameters>;

    const std::size_t threads = resolve_thread_count<Threading>(
        api_mesh.config().thread_count,
        thread_count);
    Operation operation(
        api_mesh.level3_mesh(),
        api_mesh.config().search,
        api_mesh.config().raycast,
        make_threading<Threading>(threads),
        SingleThread{},
        make_high_queue_parameters(api_mesh),
        make_high_edge_parameters(api_mesh));
    return operation.compute(verbose || api_mesh.config().verbose);
}

} // namespace api_detail

/** Level 2: construct a HighVoronoi mesh from an explicit VoronoiConfig. */
template <
    int Dim,
    class SourceScalar,
    class SourceIndex,
    class Config,
    std::enable_if_t<IsVoronoiConfigV<Config>, int> = 0>
[[nodiscard]] auto high_voronoi_mesh(
    const double* points,
    std::size_t point_count,
    const Boundary<Dim, SourceScalar, SourceIndex>& boundary,
    Config config,
    std::size_t dimension = (Dim == Dynamic ? 0 : static_cast<std::size_t>(Dim))) {
    using ApiMesh = ApiHighVoronoiMesh<Dim, std::decay_t<Config>>;
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

/** Level 1: Combined + robust fallback, automatic storage/hashing, SingleThread. */
template <
    int Dim,
    class Threading = SingleThread,
    class SourceScalar,
    class SourceIndex,
    std::enable_if_t<api_detail::is_supported_threading_v<Threading>, int> = 0>
[[nodiscard]] auto high_voronoi_mesh(
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
    return high_voronoi_mesh<Dim>(
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
[[nodiscard]] auto high_voronoi_mesh(
    const double* points,
    std::size_t point_count,
    const Boundary<Dim, SourceScalar, SourceIndex>& boundary,
    std::size_t dimension) {
    return high_voronoi_mesh<Dim>(
        points,
        point_count,
        boundary,
        SingleThread{},
        dimension);
}

template <int Dim, class Config>
[[nodiscard]] auto compute(ApiHighVoronoiMesh<Dim, Config>& mesh) {
    return api_detail::run_high_compute(mesh, std::nullopt, false);
}

template <int Dim, class Config>
[[nodiscard]] auto compute(
    ApiHighVoronoiMesh<Dim, Config>& mesh,
    std::size_t thread_count) {
    return api_detail::run_high_compute(mesh, thread_count, false);
}

template <int Dim, class Config>
[[nodiscard]] auto refine(
    ApiHighVoronoiMesh<Dim, Config>& api_mesh,
    const double* points,
    std::size_t point_count,
    std::optional<std::size_t> thread_count = std::nullopt) {
    if (point_count != 0 && points == nullptr) {
        throw std::invalid_argument("refine() received a null points pointer.");
    }

    const std::size_t dimension = static_cast<std::size_t>(api_mesh.dimension());
    for (std::size_t i = 0; i < point_count; ++i) {
        typename ApiHighVoronoiMesh<Dim, Config>::NodePoint point;
        if constexpr (Dim == Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension));
        }
        for (std::size_t d = 0; d < dimension; ++d) {
            point[static_cast<Eigen::Index>(d)] = points[i * dimension + d];
        }
        (void)api_mesh.level3_mesh().append_visible_node(point);
    }
    if (thread_count) {
        return compute(api_mesh, *thread_count);
    }
    return compute(api_mesh);
}

template <int Dim, class Config>
[[nodiscard]] auto remove(
    ApiHighVoronoiMesh<Dim, Config>& api_mesh,
    const std::vector<typename ApiHighVoronoiMesh<Dim, Config>::Index>& public_nodes,
    std::optional<std::size_t> thread_count = std::nullopt) {
    (void)api_mesh.level3_mesh().erase_visible_nodes(public_nodes);
    if (thread_count) {
        return compute(api_mesh, *thread_count);
    }
    return compute(api_mesh);
}

template <int Dim, class Config>
[[nodiscard]] auto remove(
    ApiHighVoronoiMesh<Dim, Config>& api_mesh,
    std::initializer_list<typename ApiHighVoronoiMesh<Dim, Config>::Index> public_nodes,
    std::optional<std::size_t> thread_count = std::nullopt) {
    return remove(
        api_mesh,
        std::vector<typename ApiHighVoronoiMesh<Dim, Config>::Index>(public_nodes),
        thread_count);
}

} // namespace highvoronoi
