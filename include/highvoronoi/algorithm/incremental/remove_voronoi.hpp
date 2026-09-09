
#pragma once

/**
 * @file remove_voronoi.hpp
 * @brief Two-phase node removal and delayed local closure.
 */

#include <highvoronoi/algorithm/incremental/incremental_voronoi_backend.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Remove selected nodes now and close the resulting holes later.
 *
 * RemoveVoronoi owns the stable-internal AFFECTED set created by the removal.
 * `remove()` only destroys incident vertices and the selected generators;
 * `compute()` later closes exactly those damaged cells against the mesh in its
 * then-current state. A RefineVoronoi may therefore run between both calls.
 */
template <
    class MeshT,
    class SearchKeywordT = geometry::KDSearch,
    class RayParametersT = RaycastParameters<
        CombinedRaycast,
        typename MeshT::NodeScalar>,
    class MeshThreadingT = SingleThread,
    class CastThreadingT = SingleThread,
    class QueueParametersT = DataBaseParams<
        typename MeshT::VertexScalar,
        typename MeshT::Index>,
    class EdgeParametersT = EdgeBufferParams<>>
class RemoveVoronoi final {
public:
    using Mesh = MeshT;
    using SearchKeyword = SearchKeywordT;
    using RayParameters = RayParametersT;
    using MeshThreading = MeshThreadingT;
    using CastThreading = CastThreadingT;
    using QueueParameters = QueueParametersT;
    using EdgeParameters = EdgeParametersT;

    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using VertexPoint = typename Mesh::VertexPoint;
    using Sigma = typename Mesh::Sigma;
    using AffectedVector = IncrementalVoronoiAffectedVector<
        MeshThreading,
        CastThreading>;
    using Backend = IncrementalVoronoiBackend<Mesh, AffectedVector>;
    using ComputeState = typename Backend::State;
    using ComputeMesh = typename Backend::ComputeMesh;
    using PeriodicRequest = typename Backend::PeriodicRequest;

    struct Report {
        std::size_t removed_nodes = 0;
        std::size_t erased_vertices = 0;
        std::size_t affected_nodes = 0;
        std::size_t repair_vertices = 0;
        bool removed = false;
        bool recomputed = false;
    };

    /** Normal public-node removal. */
    RemoveVoronoi(
        Mesh& mesh,
        const std::vector<Index>& deleted_public_nodes,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{},
        QueueParameters queue_parameters = QueueParameters(
            typename QueueParameters::ContainerMode{}),
        EdgeParameters edge_parameters = EdgeParameters(
            typename EdgeParameters::ContainerMode{}))
        : mesh_(mesh),
          search_keyword_(std::move(search_keyword)),
          ray_parameters_(std::move(ray_parameters)),
          mesh_threading_(std::move(mesh_threading)),
          cast_threading_(std::move(cast_threading)),
          queue_parameters_(std::move(queue_parameters)),
          edge_parameters_(std::move(edge_parameters)),
          initial_public_count_(Backend::public_size(mesh)),
          removal_internal_count_(Backend::internal_size(mesh)),
          deleted_public_nodes_(deleted_public_nodes),
          deleted_internal_flags_(
              static_cast<std::size_t>(removal_internal_count_)),
          affected_(static_cast<std::size_t>(removal_internal_count_)) {
        Backend::validate(mesh_);
        initialize_deleted_nodes_from_public();
    }

    /**
     * Use a deletion already performed structurally by the owning mesh.
     *
     * HighVoronoiMesh uses this form: erase_visible_nodes() marks visible nodes
     * and all their periodic copies inactive immediately, while their incident
     * database records remain available until ComputeHighVoronoi consumes the
     * pending deletion.
     */
    RemoveVoronoi(
        Mesh& mesh,
        ExistingRemovedNodesTag,
        std::vector<Index> deleted_internal_nodes,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{},
        QueueParameters queue_parameters = QueueParameters(
            typename QueueParameters::ContainerMode{}),
        EdgeParameters edge_parameters = EdgeParameters(
            typename EdgeParameters::ContainerMode{}))
        : mesh_(mesh),
          search_keyword_(std::move(search_keyword)),
          ray_parameters_(std::move(ray_parameters)),
          mesh_threading_(std::move(mesh_threading)),
          cast_threading_(std::move(cast_threading)),
          queue_parameters_(std::move(queue_parameters)),
          edge_parameters_(std::move(edge_parameters)),
          initial_public_count_(Backend::public_size(mesh)),
          removal_internal_count_(Backend::internal_size(mesh)),
          deleted_internal_flags_(
              static_cast<std::size_t>(removal_internal_count_)),
          affected_(static_cast<std::size_t>(removal_internal_count_)),
          structure_already_removed_(true) {
        Backend::validate(mesh_);
        initialize_deleted_nodes_from_internal(std::move(deleted_internal_nodes));
    }

    RemoveVoronoi(const RemoveVoronoi&) = delete;
    RemoveVoronoi& operator=(const RemoveVoronoi&) = delete;
    RemoveVoronoi(RemoveVoronoi&&) = delete;
    RemoveVoronoi& operator=(RemoveVoronoi&&) = delete;

    /** Perform only deletion and remember the cells whose closure was lost. */
    [[nodiscard]] AffectedVector& remove() {
        if (removed_) {
            throw std::logic_error(
                "RemoveVoronoi::remove() may only be called once.");
        }
        if (computed_) {
            throw std::logic_error(
                "RemoveVoronoi::remove() cannot run after compute().");
        }

        verify_initial_owner_unchanged();
        collect_affected_and_erase_incident_vertices();

        if (!structure_already_removed_) {
            report_.removed_nodes = Backend::erase_public_nodes(
                mesh_,
                deleted_public_nodes_);
        } else {
            report_.removed_nodes = deleted_internal_nodes_.size();
            Backend::compact_vertex_lists(mesh_);
        }

        removed_ = true;
        report_.removed = true;
        report_.affected_nodes = detail::active_incremental_affected_nodes(
            mesh_,
            affected_,
            removal_internal_count_).size();

        // remove() is itself an observable mutation boundary: its damaged-cell
        // state may intentionally live across a later RefineVoronoi.
        mesh_.propagate_neighbour_dirty();
        return affected_;
    }

    /**
     * Close this remover's original AFFECTED front on the current mesh.
     *
     * `verbose` is forwarded only to the delegated ComputeVoronoi repair
     * phase. The destructive remove() phase itself has no progress meter.
     */
    [[nodiscard]] AffectedVector& compute(bool verbose = false) {
        if (computed_) {
            throw std::logic_error(
                "RemoveVoronoi::compute() may only be called once.");
        }
        if (!removed_) {
            (void)remove();
        }

        detail::ensure_affected_size(
            affected_,
            Backend::internal_size(mesh_));

        const std::vector<Index> affected_front =
            detail::active_incremental_affected_nodes(
                mesh_,
                affected_,
                removal_internal_count_);
        report_.affected_nodes = affected_front.size();

        if (!affected_front.empty()) {
            ComputeState state = Backend::make_state(
                mesh_,
                Backend::internal_size(mesh_),
                affected_);
            Backend::set_repair_phase(state);

            ComputeMesh computation(
                mesh_,
                state,
                affected_front);
            report_.repair_vertices = detail::run_incremental_compute(
                computation,
                static_cast<Index>(affected_front.size()),
                search_keyword_,
                ray_parameters_,
                mesh_threading_,
                cast_threading_,
                queue_parameters_,
                edge_parameters_,
                verbose);
            periodic_requests_ = Backend::periodic_requests(state);
        }

        computed_ = true;
        report_.recomputed = true;

        // The repair phase can create further vertices and therefore further
        // dirty cells. Preserve them for every registered consumer as well.
        mesh_.propagate_neighbour_dirty();
        return affected_;
    }

    [[nodiscard]] AffectedVector& affected() noexcept { return affected_; }
    [[nodiscard]] const AffectedVector& affected() const noexcept {
        return affected_;
    }
    [[nodiscard]] const Report& report() const noexcept { return report_; }
    [[nodiscard]] bool removed() const noexcept { return removed_; }
    [[nodiscard]] bool computed() const noexcept { return computed_; }

    [[nodiscard]] const std::vector<Index>&
    deleted_internal_nodes() const noexcept {
        return deleted_internal_nodes_;
    }

    [[nodiscard]] const std::vector<PeriodicRequest>&
    periodic_requests() const noexcept {
        return periodic_requests_;
    }

private:
    void initialize_deleted_nodes_from_public() {
        deleted_internal_nodes_ = Backend::expand_public_deletions(
            mesh_,
            deleted_public_nodes_);
        initialize_deleted_flags();
    }

    void initialize_deleted_nodes_from_internal(
        std::vector<Index> deleted_internal_nodes) {
        std::sort(deleted_internal_nodes.begin(), deleted_internal_nodes.end());
        deleted_internal_nodes.erase(
            std::unique(
                deleted_internal_nodes.begin(),
                deleted_internal_nodes.end()),
            deleted_internal_nodes.end());

        for (const Index internal : deleted_internal_nodes) {
            if (internal >= removal_internal_count_) {
                throw std::out_of_range(
                    "RemoveVoronoi deleted internal node is out of range.");
            }
        }
        deleted_internal_nodes_ = std::move(deleted_internal_nodes);
        initialize_deleted_flags();
    }

    void initialize_deleted_flags() {
        for (const Index internal : deleted_internal_nodes_) {
            deleted_internal_flags_.set(static_cast<std::size_t>(internal));
        }
    }

    void verify_initial_owner_unchanged() const {
        if (Backend::internal_size(mesh_) != removal_internal_count_ ||
            Backend::public_size(mesh_) != initial_public_count_) {
            throw std::logic_error(
                "Mesh structure changed before RemoveVoronoi::remove().");
        }
    }

    void collect_affected_and_erase_incident_vertices() {
        std::unordered_set<Address> visited;
        VertexPoint position;
        if constexpr (Mesh::DimensionAtCompileTime == Dynamic) {
            position.resize(static_cast<Eigen::Index>(mesh_.dimension()));
        }
        Sigma sigma;

        auto process = [&](const typename Mesh::AddressList& addresses) {
            const std::size_t count = addresses.size();
            for (std::size_t p = 0; p < count; ++p) {
                const Address address = addresses[p];
                if (!visited.insert(address).second) {
                    continue;
                }

                sigma.clear();
                Backend::read_internal_vertex(
                    mesh_,
                    address,
                    position,
                    sigma);
                if (sigma.empty()) {
                    continue;
                }

                for (const Index generator : sigma) {
                    if (generator >= removal_internal_count_) {
                        break;
                    }
                    if (deleted_internal_flags_.test(
                            static_cast<std::size_t>(generator))) {
                        continue;
                    }
                    if (Backend::is_active_internal(mesh_, generator)) {
                        affected_.set(static_cast<std::size_t>(generator));
                    }
                }

                if (Backend::erase_internal_vertex(mesh_, address, sigma)) {
                    ++report_.erased_vertices;
                }
            }
        };

        for (const Index deleted_internal : deleted_internal_nodes_) {
            process(Backend::primary_addresses(mesh_, deleted_internal));
            process(Backend::secondary_addresses(mesh_, deleted_internal));
        }

        Backend::collect_infinite_affected(
            mesh_,
            deleted_internal_flags_,
            affected_);
    }

    Mesh& mesh_;
    SearchKeyword search_keyword_;
    RayParameters ray_parameters_;
    MeshThreading mesh_threading_;
    CastThreading cast_threading_;
    QueueParameters queue_parameters_;
    EdgeParameters edge_parameters_;

    Index initial_public_count_ = Index{0};
    Index removal_internal_count_ = Index{0};
    std::vector<Index> deleted_public_nodes_;
    detail::BitVector deleted_internal_flags_;
    std::vector<Index> deleted_internal_nodes_;
    AffectedVector affected_;
    std::vector<PeriodicRequest> periodic_requests_;

    Report report_{};
    bool structure_already_removed_ = false;
    bool removed_ = false;
    bool computed_ = false;
};

} // namespace highvoronoi
