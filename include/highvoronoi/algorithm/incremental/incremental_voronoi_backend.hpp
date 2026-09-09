

#pragma once

/**
 * @file incremental_voronoi_backend.hpp
 * @brief Compile-time mesh backend used by RemoveVoronoi and RefineVoronoi.
 *
 * The geometric operations are shared. Only storage access, the temporary
 * ComputeVoronoi facade and optional periodic event collection depend on the
 * owning mesh type.
 */

#include <highvoronoi/algorithm/incremental/incremental_voronoi_compute_mesh.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {

/** Use nodes that have already been appended to the owning mesh. */
struct ExistingIncrementalNodesTag final {};
inline constexpr ExistingIncrementalNodesTag existing_incremental_nodes{};

/** Use nodes that have already been marked inactive by the owning mesh. */
struct ExistingRemovedNodesTag final {};
inline constexpr ExistingRemovedNodesTag existing_removed_nodes{};

template <class MeshT, class AffectedVectorT>
struct IncrementalVoronoiBackend;

// -----------------------------------------------------------------------------
// Ordinary VoronoiMesh
// -----------------------------------------------------------------------------

template <typename NodeScalarT,
          int Dim,
          class DatabaseT,
          class AffectedVectorT>
struct IncrementalVoronoiBackend<
    VoronoiMesh<NodeScalarT, Dim, DatabaseT>,
    AffectedVectorT> {
    using Mesh = VoronoiMesh<NodeScalarT, Dim, DatabaseT>;
    using AffectedVector = AffectedVectorT;
    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using AddressList = typename Mesh::AddressList;
    using NodePoint = typename Mesh::NodePoint;
    using VertexPoint = typename Mesh::VertexPoint;
    using Sigma = typename Mesh::Sigma;
    using State = IncrementalVoronoiComputeState<Mesh, AffectedVector>;
    using ComputeMesh = IncrementalVoronoiComputeMesh<Mesh, AffectedVector>;
    using PeriodicRequest = std::pair<Index, Index>;

    static constexpr bool supports_periodization = false;
    static constexpr bool supports_persisted_infinite_edges = true;

    static void validate(const Mesh& mesh) {
        detail::require_nonperiodic_incremental_boundary(mesh);
    }

    [[nodiscard]] static Index internal_size(const Mesh& mesh) noexcept {
        return mesh.internal_nodes_.size();
    }

    [[nodiscard]] static Index public_size(const Mesh& mesh) noexcept {
        return mesh.size();
    }

    [[nodiscard]] static bool is_active_internal(
        const Mesh& mesh,
        Index internal) noexcept {
        return internal < mesh.internal_nodes_.size() &&
               mesh.index_mapping().internal_to_public(internal).has_value();
    }

    [[nodiscard]] static Index public_to_internal(
        const Mesh& mesh,
        Index public_node) {
        return mesh.index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] static std::optional<Index> internal_to_public(
        const Mesh& mesh,
        Index internal) {
        if (internal >= mesh.internal_nodes_.size()) {
            return std::nullopt;
        }
        return mesh.index_mapping().internal_to_public(internal);
    }

    [[nodiscard]] static std::vector<Index> append_visible_nodes(
        Mesh& mesh,
        const std::vector<NodePoint>& points) {
        return mesh.append_nodes(points);
    }

    [[nodiscard]] static State make_state(
        Mesh& mesh,
        Index old_internal_count,
        AffectedVector& affected) {
        return State(internal_size(mesh), old_internal_count, affected);
    }

    static void set_new_phase(State& state) noexcept {
        state.set_phase(IncrementalVoronoiComputePhase::NewCells);
    }

    static void set_repair_phase(State& state) noexcept {
        state.set_phase(IncrementalVoronoiComputePhase::RepairCells);
    }

    static void set_periodic_boundary_vertex_capture(
        State&,
        bool) noexcept {}

    [[nodiscard]] static std::vector<std::pair<VertexPoint, Sigma>>
    periodic_boundary_vertices(const State&) {
        return {};
    }

    [[nodiscard]] static const AddressList& primary_addresses(
        const Mesh& mesh,
        Index internal) {
        return mesh.primary_address_lists_.at(
            static_cast<std::size_t>(internal));
    }

    [[nodiscard]] static const AddressList& secondary_addresses(
        const Mesh& mesh,
        Index internal) {
        return mesh.secondary_address_lists_.at(
            static_cast<std::size_t>(internal));
    }

    static void read_internal_vertex(
        const Mesh& mesh,
        Address address,
        VertexPoint& position,
        Sigma& sigma) {
        mesh.database_->read(address, position, sigma);
    }

    [[nodiscard]] static bool erase_internal_vertex(
        Mesh& mesh,
        Address address,
        const Sigma& sigma) {
        const bool erased = mesh.database_->erase(address, sigma);
        if (erased) {
            mesh.mark_neighbours_dirty(sigma);
        }
        return erased;
    }

    static void copy_internal_node(
        const Mesh& mesh,
        Index internal,
        NodePoint& point) {
        mesh.internal_nodes_.copy_node(internal, point.data());
    }

    static void compact_vertex_lists(Mesh& mesh) {
        mesh.compact_vertex_address_lists();
        mesh.compact_infinite_edge_address_list();
    }

    [[nodiscard]] static std::vector<Index> expand_public_deletions(
        const Mesh& mesh,
        const std::vector<Index>& public_nodes) {
        std::vector<Index> result;
        result.reserve(public_nodes.size());
        for (const Index public_node : public_nodes) {
            if (public_node >= mesh.size()) {
                throw std::out_of_range(
                    "RemoveVoronoi public node index out of range.");
            }
            result.push_back(mesh.index_mapping().public_to_internal(public_node));
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }

    [[nodiscard]] static std::size_t erase_public_nodes(
        Mesh& mesh,
        const std::vector<Index>& public_nodes) {
        detail::BitVector selected(static_cast<std::size_t>(mesh.size()));
        for (const Index public_node : public_nodes) {
            if (public_node >= mesh.size()) {
                throw std::out_of_range(
                    "RemoveVoronoi public node index out of range.");
            }
            selected.set(static_cast<std::size_t>(public_node));
        }
        return mesh.erase_nodes_if(
            [&selected](Index public_node, const auto&) {
                return selected.test(static_cast<std::size_t>(public_node));
            });
    }

    static void collect_infinite_affected(
        const Mesh& mesh,
        const detail::BitVector& deleted_internal,
        AffectedVector& affected) {
        const Index public_count = mesh.size();
        for (const auto& edge : mesh.infinite_edges()) {
            bool touches_deleted = false;
            for (const Index generator : edge.sigma) {
                if (generator >= public_count) {
                    break;
                }
                const Index internal =
                    mesh.index_mapping().public_to_internal(generator);
                if (deleted_internal.test(static_cast<std::size_t>(internal))) {
                    touches_deleted = true;
                    break;
                }
            }
            if (!touches_deleted) {
                continue;
            }
            for (const Index generator : edge.sigma) {
                if (generator >= public_count) {
                    break;
                }
                const Index internal =
                    mesh.index_mapping().public_to_internal(generator);
                if (!deleted_internal.test(static_cast<std::size_t>(internal))) {
                    affected.set(static_cast<std::size_t>(internal));
                }
            }
        }
    }

    static void observe_existing_new_vertices(
        Mesh&,
        ComputeMesh&,
        const std::vector<Index>&) {}

    [[nodiscard]] static std::vector<PeriodicRequest>
    periodic_requests(const State&) {
        return {};
    }
};

namespace detail {

template <class Mesh, class AffectedVector>
[[nodiscard]] std::vector<typename Mesh::Index>
active_incremental_affected_nodes(
    const Mesh& mesh,
    const AffectedVector& affected,
    typename Mesh::Index stable_limit) {
    using Backend = IncrementalVoronoiBackend<Mesh, AffectedVector>;
    using Index = typename Mesh::Index;

    const std::size_t limit = std::min(
        affected.size(),
        static_cast<std::size_t>(stable_limit));

    std::vector<Index> result;
    for (std::size_t stable = 0; stable < limit; ++stable) {
        if (!affected.test(stable)) {
            continue;
        }
        const Index internal = static_cast<Index>(stable);
        if (Backend::is_active_internal(mesh, internal)) {
            result.push_back(internal);
        }
    }
    return result;
}

} // namespace detail

} // namespace highvoronoi




