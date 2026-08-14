#pragma once

/**
 * @file voronoi_worker_v2_20260812.hpp
 * @brief Threading-agnostic worker for systematic Voronoi exploration.
 *
 * A VoronoiWorker contains no global threading policy and owns no locks.
 * Synchronization belongs to the queue, edge hash, mesh/database, FEI cache,
 * and the shared vertex-iterator data structures supplied by its parent.
 *
 * Worker-owned state is deliberately limited to the mutable geometric state
 * that cannot be shared between simultaneous ray casts:
 *
 * - one worker-local RayCaster safe copy;
 * - one EdgeIterator used while a found vertex is queued;
 * - one EdgeIterator used while a queued vertex is systematically explored;
 * - reusable numerical and signature scratch buffers.
 *
 * The two EdgeIterators are rebound to the worker-local ExtendedNodes but share
 * the FEIStorageCache of the prototype EdgeIterator owned by
 * SystematicVoronoi.
 */

#include <highvoronoi/geometry/edge_iterator.hpp>
#include <highvoronoi/geometry/raycaster.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

namespace detail {

template <class Point>
[[nodiscard]] Point make_voronoi_worker_point(std::size_t dimension) {
    if constexpr (Point::RowsAtCompileTime == Eigen::Dynamic) {
        return Point(static_cast<Eigen::Index>(dimension));
    } else {
        (void)dimension;
        return Point{};
    }
}

template <class Target, class Source>
void copy_voronoi_worker_point(const Source& source, Target& target) {
    if constexpr (Target::RowsAtCompileTime == Eigen::Dynamic) {
        target.resize(source.size());
    }

    for (Eigen::Index i = 0; i < source.size(); ++i) {
        target[i] = static_cast<typename Target::Scalar>(source[i]);
    }
}

template <class RangeA, class RangeB>
[[nodiscard]] bool same_voronoi_index_range(
    const RangeA& left,
    const RangeB& right) {
    if (static_cast<std::size_t>(left.size()) !=
        static_cast<std::size_t>(right.size())) {
        return false;
    }

    return std::equal(left.begin(), left.end(), right.begin());
}

} // namespace detail

/**
 * @brief Reusable owning queue item used by SystematicVoronoi and its workers.
 */
template <class MeshT, class RayCasterT>
struct VoronoiVertexTask {
    using Mesh = MeshT;
    using RayCaster = RayCasterT;
    using Index = typename Mesh::Index;
    using Sigma = typename Mesh::Sigma;
    using Point = typename RayCaster::Point;

    Sigma sigma;
    Point position;

    explicit VoronoiVertexTask(std::size_t dimension)
        : position(detail::make_voronoi_worker_point<Point>(dimension)) {}
};

/**
 * @brief Geometry worker used by one SystematicVoronoi object.
 *
 * The worker is intentionally unaware of MeshThreading, CastThreading, and the
 * lock types selected by either policy. The parent decides how many workers
 * exist and whether their public work functions are called serially or from
 * several threads.
 */
template <
    class ParentT,
    class MeshT,
    class RayCasterT,
    class EdgeIteratorT,
    class VertexQueueT>
class VoronoiWorker final {
public:
    using Parent = ParentT;
    using Mesh = MeshT;
    using RayCaster = RayCasterT;
    using EdgeIteratorType = EdgeIteratorT;
    using VertexQueue = VertexQueueT;

    using Index = typename Mesh::Index;
    using Sigma = typename Mesh::Sigma;
    using Scalar = typename RayCaster::Scalar;
    using Point = typename RayCaster::Point;
    using MeshPoint = typename Mesh::VertexPoint;
    using Vertex = VoronoiVertexTask<Mesh, RayCaster>;

    VoronoiWorker(
        Mesh& mesh,
        Parent& parent,
        VertexQueue& vertex_queue,
        const RayCaster& raycaster_prototype,
        const EdgeIteratorType& edge_iterator_prototype,
        std::size_t worker_id = 0)
        : mesh_(mesh),
          parent_(parent),
          vertex_queue_(vertex_queue),
          raycaster_(raycaster_prototype.safe_copy()),
          queueing_iterator_(
              raycaster_.extended_nodes(),
              edge_iterator_prototype.storage_cache_handle(),
              raycaster_.parameters().ray_tolerance),
          searching_iterator_(
              raycaster_.extended_nodes(),
              edge_iterator_prototype.storage_cache_handle(),
              raycaster_.parameters().ray_tolerance),
          dimension_(static_cast<std::size_t>(mesh.dimension())),
          current_vertex_(dimension_),
          candidate_vertex_(dimension_),
          descent_vertex_(dimension_),
          descent_point_(detail::make_voronoi_worker_point<Point>(dimension_)),
          direction_(detail::make_voronoi_worker_point<Point>(dimension_)),
          negative_direction_(detail::make_voronoi_worker_point<Point>(dimension_)),
          node_buffer_(detail::make_voronoi_worker_point<Point>(dimension_)),
          reference_node_(detail::make_voronoi_worker_point<Point>(dimension_)),
          basis_buffer_(detail::make_voronoi_worker_point<Point>(dimension_)),
          corrected_point_(detail::make_voronoi_worker_point<Point>(dimension_)),
          mesh_point_buffer_(make_mesh_point()),
          random_(0x4856564f524b4552ULL +
                  static_cast<std::uint64_t>(worker_id) *
                      0x9e3779b97f4a7c15ULL),
          normal_(Scalar{0}, Scalar{1}) {

        static_assert(
            std::is_same_v<Index, typename RayCaster::Index>,
            "VoronoiWorker requires equal mesh and RayCaster index types.");

        static_assert(
            std::is_same_v<Index, typename EdgeIteratorType::Index>,
            "VoronoiWorker requires equal mesh and EdgeIterator index types.");

        if (dimension_ < 2) {
            throw std::invalid_argument(
                "VoronoiWorker requires mesh dimension >= 2.");
        }

        walk_sigma_.reserve(dimension_ + 2);
        descent_sigma_.reserve(dimension_ + 2);
        minimal_support_.reserve(dimension_ + 1);
        internal_sigma_buffer_.reserve(dimension_ + 2);
        external_sigma_buffer_.reserve(dimension_ + 2);

        orthogonal_basis_.reserve(dimension_);
        for (std::size_t i = 0; i < dimension_; ++i) {
            orthogonal_basis_.push_back(
                detail::make_voronoi_worker_point<Point>(dimension_));
        }
    }

    VoronoiWorker(const VoronoiWorker&) = delete;
    VoronoiWorker& operator=(const VoronoiWorker&) = delete;
    VoronoiWorker(VoronoiWorker&&) = delete;
    VoronoiWorker& operator=(VoronoiWorker&&) = delete;

    /** Activate this worker's independent RayCaster boundary state. */
    template <class MirrorIndices>
    void activate_cell(Index cell, const MirrorIndices& mirror_indices) {
        raycaster_.activate_cell(cell, mirror_indices);
    }

    /**
     * @brief Consume already-known mesh vertices from a shared iterator.
     *
     * Existing mesh vertices are not routed through ComputeVoronoi::register:
     * they are already globally persistent. They only need to be introduced
     * into the current SystematicVoronoi queue/edge state.
     */
    template <class SharedVertexIterator>
    void queue_vertices_from_iterator(
        SharedVertexIterator& iterator,
        Index cell) {
        while (iterator.next(current_vertex_)) {
            (void)parent_.queue_vertex(
                current_vertex_,
                queueing_iterator_,
                cell);
        }
    }

    /**
     * @brief Pop and systematically explore vertices until the queue is empty.
     */
    void systematic_explore_all_vertices(Index cell) {
        while (vertex_queue_.pop(current_vertex_)) {
            systematic_explore_vertex(current_vertex_, cell);
        }
    }

    /**
     * @brief Find one initial Voronoi vertex of a cell by descent.
     */
    [[nodiscard]] Vertex descent(Index cell) {
        constexpr std::size_t maximum_restarts = 8;
        constexpr std::size_t maximum_direction_attempts = 100;

        for (std::size_t restart = 0;
             restart < maximum_restarts;
             ++restart) {

            descent_sigma_.clear();
            descent_sigma_.push_back(cell);
            minimal_support_.clear();
            minimal_support_.push_back(cell);
            load_node(cell, descent_point_);

            bool failed = false;

            for (std::size_t level = 0;
                 level < dimension_;
                 ++level) {

                bool found = false;

                for (std::size_t attempt = 0;
                     attempt < maximum_direction_attempts;
                     ++attempt) {

                    if (!random_orthogonal_direction(
                            minimal_support_, direction_)) {
                        continue;
                    }

                    auto result = raycaster_.cast(
                        descent_sigma_,
                        descent_point_,
                        direction_,
                        descent_sigma_,
                        descent_sigma_,
                        RayCastUsage::Descent,
                        Scalar{0});

                    if (result.status == RayCastStatus::Infinite) {
                        negative_direction_ = -direction_;
                        result = raycaster_.cast(
                            descent_sigma_,
                            descent_point_,
                            negative_direction_,
                            descent_sigma_,
                            descent_sigma_,
                            RayCastUsage::Descent,
                            Scalar{0});
                    }

                    if (result.status == RayCastStatus::Infinite) {
                        continue;
                    }

                    descent_point_ = result.position;
                    if (std::find(
                            minimal_support_.begin(),
                            minimal_support_.end(),
                            result.generator) == minimal_support_.end()) {
                        minimal_support_.push_back(result.generator);
                    }

                    found = true;
                    break;
                }

                if (!found) {
                    failed = true;
                    break;
                }
            }

            if (failed) {
                continue;
            }

            descent_point_ = mesh_.boundary().project_inside(descent_point_);

            if (minimal_support_.size() == dimension_ + 1 &&
                raycaster_.correct_vertex(
                    minimal_support_,
                    corrected_point_)) {
                descent_point_ = corrected_point_;
            }

            if (!raycaster_.verify_vertex(
                    descent_sigma_,
                    descent_point_)) {
                continue;
            }

            descent_vertex_.sigma.assign(
                descent_sigma_.begin(),
                descent_sigma_.end());
            descent_vertex_.position = descent_point_;
            return copy_vertex_value(descent_vertex_);
        }

        throw std::runtime_error(
            "Voronoi descent failed to find a valid initial vertex.");
    }

    /**
     * @brief Register a newly found vertex through the top-level coordinator.
     *
     * This is used both for descent vertices and vertices found by WalkRay.
     * ComputeVoronoi stores the vertex in the main database and communicates it
     * to every SystematicVoronoi branch for which it is relevant.
     */
    [[nodiscard]] bool register_found_vertex(const Vertex& vertex) {
        return parent_.register_found_vertex(
            vertex,
            queueing_iterator_,
            internal_sigma_buffer_,
            external_sigma_buffer_,
            mesh_point_buffer_);
    }

    [[nodiscard]] RayCaster& raycaster() noexcept {
        return raycaster_;
    }

    [[nodiscard]] const RayCaster& raycaster() const noexcept {
        return raycaster_;
    }

    [[nodiscard]] EdgeIteratorType& queueing_iterator() noexcept {
        return queueing_iterator_;
    }

    [[nodiscard]] EdgeIteratorType& searching_iterator() noexcept {
        return searching_iterator_;
    }

private:
    [[nodiscard]] MeshPoint make_mesh_point() const {
        if constexpr (Mesh::DimensionAtCompileTime == Dynamic) {
            return MeshPoint(static_cast<Eigen::Index>(dimension_));
        } else {
            return MeshPoint{};
        }
    }

    [[nodiscard]] Vertex copy_vertex_value(const Vertex& source) const {
        Vertex result(dimension_);
        result.sigma = source.sigma;
        result.position = source.position;
        return result;
    }

    void load_node(Index index, Point& target) const {
        raycaster_.extended_nodes().copy_node(index, target.data());
    }

    [[nodiscard]] static std::int64_t edge_token(Index value) {
        const auto maximum = static_cast<std::uintmax_t>(
            (std::numeric_limits<std::int64_t>::max)());
        const auto converted = static_cast<std::uintmax_t>(value);

        if (converted > maximum) {
            throw std::overflow_error(
                "EdgeHash token does not fit into int64_t.");
        }

        return static_cast<std::int64_t>(value);
    }

    /** Julia systematic_explore_vertex. */
    void systematic_explore_vertex(
        const Vertex& vertex,
        Index cell) {

        searching_iterator_.reset(
            vertex.sigma,
            vertex.position,
            cell,
            typename EdgeIteratorType::OnSysVoronoi{});

        while (const auto edge = searching_iterator_.next()) {
            const bool already_complete =
                parent_.edge_hash().pushedge(
                    edge->indices(),
                    edge_token(cell),
                    true);

            if (edge->indices().empty() ||
                edge->indices()[0] != cell ||
                already_complete) {
                continue;
            }

            if (detail::same_voronoi_index_range(
                    edge->full_indices(),
                    vertex.sigma)) {
                continue;
            }

            if (!walk_ray(vertex, *edge, candidate_vertex_)) {
                // Persistent storage for infinite rays is not part of the
                // current mesh API yet.
                continue;
            }

            if (candidate_vertex_.sigma.size() < dimension_ + 1) {
                continue;
            }

            if (detail::same_voronoi_index_range(
                    candidate_vertex_.sigma,
                    vertex.sigma)) {
                continue;
            }

            // Julia verifies only when the candidate is not already known.
            // The persistent mesh still performs the final duplicate-safe
            // insertion in ComputeVoronoi::register_vertex().
            if (!mesh_.contains_vertex(
                    candidate_vertex_.sigma,
                    internal_sigma_buffer_) &&
                !raycaster_.verify_vertex(
                    candidate_vertex_.sigma,
                    candidate_vertex_.position)) {
                continue;
            }

            (void)register_found_vertex(candidate_vertex_);
        }
    }

    /** Julia walkray, with correction delegated to RayCaster. */
    template <class EdgeView>
    [[nodiscard]] bool walk_ray(
        const Vertex& origin,
        const EdgeView& edge,
        Vertex& output) {

        walk_sigma_.assign(
            edge.full_indices().begin(),
            edge.full_indices().end());

        const auto result = raycaster_.cast(
            walk_sigma_,
            origin.position,
            edge.direction(),
            edge.indices(),
            origin.sigma,
            RayCastUsage::WalkRay,
            edge.cycle_error());

        if (result.status == RayCastStatus::Infinite) {
            // Persist the geometric unbounded edge by its complete supporting
            // generator set. Different minimal edges of a degenerate vertex may
            // describe the same ray; the mesh/database therefore deduplicates
            // on full_indices() in stable internal numbering.
            (void)parent_.register_infinite_edge(
                edge.full_indices(),
                origin.position,
                edge.direction(),
                internal_sigma_buffer_);
            return false;
        }

        output.sigma.assign(
            walk_sigma_.begin(),
            walk_sigma_.end());
        output.position = result.position;
        return true;
    }

    /**
     * Generate one random unit vector orthogonal to the current affine support.
     */
    [[nodiscard]] bool random_orthogonal_direction(
        const Sigma& support,
        Point& output) {

        if (support.empty()) {
            return false;
        }

        load_node(support.front(), reference_node_);
        std::size_t basis_count = 0;

        for (std::size_t i = 1; i < support.size(); ++i) {
            load_node(support[i], node_buffer_);
            basis_buffer_ = node_buffer_ - reference_node_;

            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t j = 0; j < basis_count; ++j) {
                    basis_buffer_ -=
                        basis_buffer_.dot(orthogonal_basis_[j]) *
                        orthogonal_basis_[j];
                }
            }

            const Scalar norm = basis_buffer_.norm();
            if (!(norm > Scalar{100} *
                         std::numeric_limits<Scalar>::epsilon())) {
                continue;
            }

            orthogonal_basis_[basis_count] = basis_buffer_ / norm;
            ++basis_count;
        }

        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            output[static_cast<Eigen::Index>(coordinate)] = normal_(random_);
        }

        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t j = 0; j < basis_count; ++j) {
                output -= output.dot(orthogonal_basis_[j]) *
                          orthogonal_basis_[j];
            }
        }

        const Scalar norm = output.norm();
        if (!(norm > Scalar{100} *
                     std::numeric_limits<Scalar>::epsilon())) {
            return false;
        }

        output /= norm;
        return output.allFinite();
    }

    Mesh& mesh_;
    Parent& parent_;
    VertexQueue& vertex_queue_;

    RayCaster raycaster_;
    EdgeIteratorType queueing_iterator_;
    EdgeIteratorType searching_iterator_;

    std::size_t dimension_;

    Vertex current_vertex_;
    Vertex candidate_vertex_;
    Vertex descent_vertex_;

    Sigma walk_sigma_;
    Sigma descent_sigma_;
    Sigma minimal_support_;
    Sigma internal_sigma_buffer_;
    Sigma external_sigma_buffer_;

    Point descent_point_;
    Point direction_;
    Point negative_direction_;
    Point node_buffer_;
    Point reference_node_;
    Point basis_buffer_;
    Point corrected_point_;
    MeshPoint mesh_point_buffer_;
    std::vector<Point> orthogonal_basis_;

    std::mt19937_64 random_;
    std::normal_distribution<Scalar> normal_;
};

} // namespace highvoronoi
