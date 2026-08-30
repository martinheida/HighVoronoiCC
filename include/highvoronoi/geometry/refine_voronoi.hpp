
#pragma once

/**
 * @file refine_voronoi.hpp
 * @brief Incremental node insertion shared by VoronoiMesh and HighVoronoiMesh.
 */

#include <highvoronoi/geometry/incremental_voronoi_backend.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Insert one NEW block, compute its cells and remove obsolete OLD vertices.
 *
 * RefineVoronoi deliberately performs no second ComputeVoronoi(AFFECTED) pass.
 * Adding generators can only cut existing cells: every genuinely new finite
 * vertex contains at least one NEW generator and is therefore discovered while
 * the NEW cells are computed. AFFECTED is retained only to restrict the old
 * vertex invalidation scan.
 */
template <
    class MeshT,
    class SearchKeywordT = geometry::KDSearch,
    class RayParametersT = RaycastParameters<
        InRangeRaycast,
        typename MeshT::NodeScalar>,
    class MeshThreadingT = SingleThread,
    class CastThreadingT = SingleThread,
    class QueueParametersT = DataBaseParams<
        typename MeshT::VertexScalar,
        typename MeshT::Index>,
    class EdgeParametersT = EdgeBufferParams<>>
class RefineVoronoi final {
public:
    using Mesh = MeshT;
    using SearchKeyword = SearchKeywordT;
    using RayParameters = RayParametersT;
    using MeshThreading = MeshThreadingT;
    using CastThreading = CastThreadingT;
    using QueueParameters = QueueParametersT;
    using EdgeParameters = EdgeParametersT;

    using Index = typename Mesh::Index;
    using NodeScalar = typename Mesh::NodeScalar;
    using VertexScalar = typename Mesh::VertexScalar;
    using NodePoint = typename Mesh::NodePoint;
    using VertexPoint = typename Mesh::VertexPoint;
    using Sigma = typename Mesh::Sigma;
    using Address = typename Mesh::Address;

    using AffectedVector = IncrementalVoronoiAffectedVector<
        MeshThreading,
        CastThreading>;
    using Backend = IncrementalVoronoiBackend<Mesh, AffectedVector>;
    using ComputeState = typename Backend::State;
    using ComputeMesh = typename Backend::ComputeMesh;
    using PeriodicRequest = typename Backend::PeriodicRequest;

    struct Settings {
        /** A NEW node invalidates an old vertex only if closer by this slack. */
        VertexScalar nearest_tolerance = VertexScalar{1e-10};

        /**
         * Capture transient periodic-boundary vertices of this NEW compute
         * block. HighVoronoi enables this only for its bootstrap periodization
         * block, before the internal periodic boundary is moved outward.
         */
        bool capture_periodic_boundary_vertices = false;
    };

    struct Report {
        std::size_t appended_nodes = 0;
        std::size_t new_cell_vertices = 0;
        std::size_t affected_nodes = 0;
        std::size_t invalidated_old_vertices = 0;
        std::size_t repair_vertices = 0; // retained for source/report compatibility
    };

    /** Append this NEW block as ordinary/visible nodes before computing it. */
    RefineVoronoi(
        Mesh& mesh,
        std::vector<NodePoint> new_nodes,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{},
        QueueParameters queue_parameters = QueueParameters(
            typename QueueParameters::ContainerMode{}),
        EdgeParameters edge_parameters = EdgeParameters(
            typename EdgeParameters::ContainerMode{}),
        Settings settings = Settings{})
        : mesh_(mesh),
          new_nodes_(std::move(new_nodes)),
          search_keyword_(std::move(search_keyword)),
          ray_parameters_(std::move(ray_parameters)),
          mesh_threading_(std::move(mesh_threading)),
          cast_threading_(std::move(cast_threading)),
          queue_parameters_(std::move(queue_parameters)),
          edge_parameters_(std::move(edge_parameters)),
          settings_(settings),
          expected_public_count_(Backend::public_size(mesh)),
          expected_internal_count_(Backend::internal_size(mesh)),
          old_internal_count_(Backend::internal_size(mesh)),
          affected_(static_cast<std::size_t>(old_internal_count_)) {
        Backend::validate(mesh_);
    }

    /**
     * Compute a NEW block that has already been appended to the owner.
     *
     * HighVoronoi uses this for pending visible nodes and for periodic reference
     * blocks materialized between outer closure rounds.
     */
    RefineVoronoi(
        Mesh& mesh,
        ExistingIncrementalNodesTag,
        std::vector<Index> new_internal_nodes,
        Index old_internal_count,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{},
        QueueParameters queue_parameters = QueueParameters(
            typename QueueParameters::ContainerMode{}),
        EdgeParameters edge_parameters = EdgeParameters(
            typename EdgeParameters::ContainerMode{}),
        Settings settings = Settings{})
        : mesh_(mesh),
          search_keyword_(std::move(search_keyword)),
          ray_parameters_(std::move(ray_parameters)),
          mesh_threading_(std::move(mesh_threading)),
          cast_threading_(std::move(cast_threading)),
          queue_parameters_(std::move(queue_parameters)),
          edge_parameters_(std::move(edge_parameters)),
          settings_(settings),
          expected_public_count_(Backend::public_size(mesh)),
          expected_internal_count_(Backend::internal_size(mesh)),
          old_internal_count_(old_internal_count),
          appended_internal_nodes_(std::move(new_internal_nodes)),
          affected_(static_cast<std::size_t>(expected_internal_count_)),
          nodes_already_appended_(true) {
        Backend::validate(mesh_);
        validate_existing_new_nodes();
    }

    RefineVoronoi(const RefineVoronoi&) = delete;
    RefineVoronoi& operator=(const RefineVoronoi&) = delete;
    RefineVoronoi(RefineVoronoi&&) = delete;
    RefineVoronoi& operator=(RefineVoronoi&&) = delete;

    /** Execute this refinement once and return its private AFFECTED set. */
    [[nodiscard]] AffectedVector& compute() {
        if (computed_) {
            throw std::logic_error(
                "RefineVoronoi::compute() may only be called once.");
        }
        verify_owner_unchanged();
        computed_ = true;

        if (!nodes_already_appended_) {
            if (new_nodes_.empty()) {
                return affected_;
            }
            appended_internal_nodes_ = Backend::append_visible_nodes(
                mesh_,
                new_nodes_);
            report_.appended_nodes = appended_internal_nodes_.size();
        } else {
            report_.appended_nodes = appended_internal_nodes_.size();
        }

        if (appended_internal_nodes_.empty()) {
            return affected_;
        }

        detail::ensure_affected_size(
            affected_,
            Backend::internal_size(mesh_));

        ComputeState state = Backend::make_state(
            mesh_,
            old_internal_count_,
            affected_);
        Backend::set_new_phase(state);
        Backend::set_periodic_boundary_vertex_capture(
            state,
            settings_.capture_periodic_boundary_vertices);

        ComputeMesh new_computation(
            mesh_,
            state,
            appended_internal_nodes_);

        // Imported HighVoronoi vertices already attached to a NEW node must
        // seed exactly the same AFFECTED/periodic events as newly stored records.
        Backend::observe_existing_new_vertices(
            mesh_,
            new_computation,
            appended_internal_nodes_);

        report_.new_cell_vertices = detail::run_incremental_compute(
            new_computation,
            static_cast<Index>(appended_internal_nodes_.size()),
            search_keyword_,
            ray_parameters_,
            mesh_threading_,
            cast_threading_,
            queue_parameters_,
            edge_parameters_);

        periodic_boundary_vertices_ =
            Backend::periodic_boundary_vertices(state);
        periodic_requests_ = Backend::periodic_requests(state);

        // AFFECTED contains OLD generators observed while NEW vertices were
        // stored. Only those old primary lists can contain obsolete vertices.
        invalidate_old_vertices(new_computation, state);

        report_.affected_nodes = detail::active_incremental_affected_nodes(
            mesh_,
            affected_,
            old_internal_count_).size();

        return affected_;
    }

    [[nodiscard]] AffectedVector& affected() noexcept { return affected_; }
    [[nodiscard]] const AffectedVector& affected() const noexcept {
        return affected_;
    }
    [[nodiscard]] const Report& report() const noexcept { return report_; }

    [[nodiscard]] const std::vector<Index>&
    appended_internal_nodes() const noexcept {
        return appended_internal_nodes_;
    }

    [[nodiscard]] const std::vector<PeriodicRequest>&
    periodic_requests() const noexcept {
        return periodic_requests_;
    }

    [[nodiscard]] const std::vector<std::pair<VertexPoint, Sigma>>&
    periodic_boundary_vertices() const noexcept {
        return periodic_boundary_vertices_;
    }

private:
    void validate_existing_new_nodes() const {
        const Index current_internal = Backend::internal_size(mesh_);
        if (old_internal_count_ > current_internal) {
            throw std::invalid_argument(
                "RefineVoronoi OLD watermark exceeds the mesh size.");
        }
        for (const Index internal : appended_internal_nodes_) {
            if (internal < old_internal_count_ ||
                internal >= current_internal ||
                !Backend::is_active_internal(mesh_, internal)) {
                throw std::invalid_argument(
                    "RefineVoronoi existing NEW block contains an invalid node.");
            }
        }
    }

    void verify_owner_unchanged() const {
        if (Backend::public_size(mesh_) != expected_public_count_ ||
            Backend::internal_size(mesh_) != expected_internal_count_) {
            throw std::logic_error(
                "Mesh structure changed after RefineVoronoi construction.");
        }
    }

    [[nodiscard]] bool old_vertex_is_candidate(
        const Sigma& internal_sigma,
        const ComputeState& state) const {
        bool has_ordinary = false;
        for (const Index generator : internal_sigma) {
            if (generator >= state.internal_count()) {
                break;
            }

            has_ordinary = true;
            if (!Backend::is_active_internal(mesh_, generator) ||
                !state.is_old_node(generator) ||
                !state.is_affected(generator)) {
                return false;
            }
        }
        return has_ordinary;
    }

    void invalidate_old_vertices(
        ComputeMesh& new_computation,
        const ComputeState& state) {
        const Index new_count = static_cast<Index>(
            appended_internal_nodes_.size());
        if (new_count == Index{0}) {
            return;
        }

        auto tree = geometry::make_search_tree(
            new_computation,
            search_keyword_);
        auto search_data = tree.make_backend_data();

        VertexPoint position;
        if constexpr (Mesh::DimensionAtCompileTime == Dynamic) {
            position.resize(static_cast<Eigen::Index>(mesh_.dimension()));
        }
        NodePoint generator_point;
        if constexpr (Mesh::DimensionAtCompileTime == Dynamic) {
            generator_point.resize(
                static_cast<Eigen::Index>(mesh_.dimension()));
        }
        Sigma sigma;

        const std::vector<Index> affected_old =
            detail::active_incremental_affected_nodes(
                mesh_,
                affected_,
                old_internal_count_);

        for (const Index primary_internal : affected_old) {
            const auto& addresses = Backend::primary_addresses(
                mesh_,
                primary_internal);
            const std::size_t address_count = addresses.size();

            for (std::size_t p = 0; p < address_count; ++p) {
                const Address address = addresses[p];
                sigma.clear();
                Backend::read_internal_vertex(
                    mesh_,
                    address,
                    position,
                    sigma);
                if (sigma.empty() ||
                    !old_vertex_is_candidate(sigma, state)) {
                    continue;
                }

                Index first_generator = state.internal_count();
                for (const Index generator : sigma) {
                    if (generator < state.internal_count()) {
                        first_generator = generator;
                        break;
                    }
                }
                if (first_generator == state.internal_count()) {
                    continue;
                }

                Backend::copy_internal_node(
                    mesh_,
                    first_generator,
                    generator_point);

                long double radius_squared = 0.0L;
                for (Index coordinate = Index{0};
                     coordinate < mesh_.dimension();
                     ++coordinate) {
                    const long double delta =
                        static_cast<long double>(
                            position[static_cast<Eigen::Index>(coordinate)]) -
                        static_cast<long double>(
                            generator_point[static_cast<Eigen::Index>(coordinate)]);
                    radius_squared += delta * delta;
                }
                const VertexScalar radius = static_cast<VertexScalar>(
                    std::sqrt(radius_squared));

                tree.write_point(search_data, position);
                const auto skip_not_new = [new_count](Index compute_public) {
                    return compute_public >= new_count;
                };
                const auto nearest = tree.nn(search_data, skip_not_new);
                if (!nearest) {
                    continue;
                }

                if (static_cast<VertexScalar>(nearest->distance) +
                        settings_.nearest_tolerance >=
                    radius) {
                    continue;
                }

                if (Backend::erase_internal_vertex(mesh_, address, sigma)) {
                    ++report_.invalidated_old_vertices;
                }
            }
        }

        if (report_.invalidated_old_vertices != 0) {
            Backend::compact_vertex_lists(mesh_);
        }
    }

    Mesh& mesh_;
    std::vector<NodePoint> new_nodes_;
    SearchKeyword search_keyword_;
    RayParameters ray_parameters_;
    MeshThreading mesh_threading_;
    CastThreading cast_threading_;
    QueueParameters queue_parameters_;
    EdgeParameters edge_parameters_;
    Settings settings_;

    Index expected_public_count_ = Index{0};
    Index expected_internal_count_ = Index{0};
    Index old_internal_count_ = Index{0};
    std::vector<Index> appended_internal_nodes_;
    AffectedVector affected_;
    std::vector<PeriodicRequest> periodic_requests_;
    std::vector<std::pair<VertexPoint, Sigma>> periodic_boundary_vertices_;

    Report report_{};
    bool nodes_already_appended_ = false;
    bool computed_ = false;
};

} // namespace highvoronoi


