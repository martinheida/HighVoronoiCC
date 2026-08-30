
#pragma once

/**
 * @file high_voronoi_incremental_backend.hpp
 * @brief HighVoronoiMesh specialization of IncrementalVoronoiBackend.
 *
 * Kept separate from the ordinary backend so VoronoiMesh refinement/removal
 * does not depend on HighVoronoi implementation details or template signatures.
 */

#include <highvoronoi/geometry/incremental_voronoi_backend.hpp>
#include <highvoronoi/geometry/high_voronoi_compute_mesh.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {

// -----------------------------------------------------------------------------
// HighVoronoiMesh
// -----------------------------------------------------------------------------

template <typename NodeScalarT,
          int Dim,
          class DatabaseT,
          class AffectedVectorT>
struct IncrementalVoronoiBackend<
    HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>,
    AffectedVectorT> {
    using Mesh = HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>;
    using AffectedVector = AffectedVectorT;
    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using AddressList = typename Mesh::AddressList;
    using NodePoint = typename Mesh::NodePoint;
    using VertexPoint = typename Mesh::VertexPoint;
    using Sigma = typename Mesh::Sigma;
    using State = HighVoronoiComputeRoundState<Mesh, AffectedVector>;
    using ComputeMesh = HighVoronoiComputeMesh<Mesh, AffectedVector>;
    using PeriodicRequest = std::pair<Index, Index>;

    static constexpr bool supports_periodization = true;
    static constexpr bool supports_persisted_infinite_edges = false;

    static void validate(const Mesh&) noexcept {}

    [[nodiscard]] static Index internal_size(const Mesh& mesh) noexcept {
        return mesh.internal_nodes_.size();
    }

    [[nodiscard]] static Index public_size(const Mesh& mesh) noexcept {
        return mesh.visible_public_count();
    }

    [[nodiscard]] static bool is_active_internal(
        const Mesh& mesh,
        Index internal) noexcept {
        return internal < mesh.internal_nodes_.size() &&
               mesh.active_.test(static_cast<std::size_t>(internal));
    }

    [[nodiscard]] static Index public_to_internal(
        const Mesh& mesh,
        Index public_node) {
        return mesh.visible_public_to_internal(public_node);
    }

    [[nodiscard]] static std::optional<Index> internal_to_public(
        const Mesh& mesh,
        Index internal) {
        if (internal >= mesh.internal_nodes_.size() ||
            !mesh.active_.test(static_cast<std::size_t>(internal))) {
            return std::nullopt;
        }
        return mesh.index_mapping().internal_to_public(internal);
    }

    [[nodiscard]] static std::vector<Index> append_visible_nodes(
        Mesh& mesh,
        const std::vector<NodePoint>& points) {
        std::vector<Index> result;
        result.reserve(points.size());
        for (const NodePoint& point : points) {
            result.push_back(mesh.append_visible_node(point));
        }
        return result;
    }

    [[nodiscard]] static State make_state(
        Mesh& mesh,
        Index old_internal_count,
        AffectedVector& affected) {
        return State(
            internal_size(mesh),
            mesh.external_boundary().size(),
            old_internal_count,
            affected);
    }

    static void set_new_phase(State& state) noexcept {
        state.set_phase(HighVoronoiComputePhase::NewCells);
    }

    static void set_repair_phase(State& state) noexcept {
        state.set_phase(HighVoronoiComputePhase::RepairCells);
    }

    static void set_periodic_boundary_vertex_capture(
        State& state,
        bool enabled) noexcept {
        state.set_capture_periodic_boundary_vertices(enabled);
    }

    [[nodiscard]] static std::vector<std::pair<VertexPoint, Sigma>>
    periodic_boundary_vertices(const State& state) {
        return state.periodic_boundary_vertices();
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
        mesh.database_.read(address, position, sigma);
    }

    [[nodiscard]] static bool erase_internal_vertex(
        Mesh& mesh,
        Address address,
        const Sigma& sigma) {
        return mesh.database_.erase(address, sigma);
    }

    static void copy_internal_node(
        const Mesh& mesh,
        Index internal,
        NodePoint& point) {
        mesh.internal_nodes_.copy_node(internal, point.data());
    }

    static void compact_vertex_lists(Mesh& mesh) {
        mesh.compact_vertex_address_lists();
    }

    [[nodiscard]] static std::vector<Index> expand_public_deletions(
        const Mesh& mesh,
        const std::vector<Index>& public_nodes) {
        detail::BitVector selected(
            static_cast<std::size_t>(mesh.internal_nodes_.size()));

        for (const Index public_node : public_nodes) {
            if (public_node >= mesh.visible_public_count()) {
                throw std::out_of_range(
                    "RemoveVoronoi visible public node index out of range.");
            }
            selected.set(static_cast<std::size_t>(
                mesh.visible_public_to_internal(public_node)));
        }

        for (Index internal = Index{0};
             internal < mesh.internal_nodes_.size();
             ++internal) {
            const auto& reference =
                mesh.references_[static_cast<std::size_t>(internal)];
            if (reference && selected.test(static_cast<std::size_t>(*reference))) {
                selected.set(static_cast<std::size_t>(internal));
            }
        }

        std::vector<Index> result;
        for (Index internal = Index{0};
             internal < mesh.internal_nodes_.size();
             ++internal) {
            if (selected.test(static_cast<std::size_t>(internal)) &&
                mesh.active_.test(static_cast<std::size_t>(internal))) {
                result.push_back(internal);
            }
        }
        return result;
    }

    [[nodiscard]] static std::size_t erase_public_nodes(
        Mesh& mesh,
        const std::vector<Index>& public_nodes) {
        return mesh.erase_visible_nodes(public_nodes).removed_internal.size();
    }

    static void collect_infinite_affected(
        const Mesh&,
        const detail::BitVector&,
        AffectedVector&) noexcept {}

    static void observe_existing_new_vertices(
        Mesh& mesh,
        ComputeMesh& computation,
        const std::vector<Index>& new_internal_nodes) {
        std::unordered_set<Address> visited;
        VertexPoint position;
        if constexpr (Mesh::DimensionAtCompileTime == Dynamic) {
            position.resize(static_cast<Eigen::Index>(mesh.dimension()));
        }
        Sigma sigma;

        auto observe = [&](const AddressList& addresses) {
            const std::size_t count = addresses.size();
            for (std::size_t p = 0; p < count; ++p) {
                const Address address = addresses[p];
                if (!visited.insert(address).second) {
                    continue;
                }
                sigma.clear();
                mesh.database_.read(address, position, sigma);
                if (!sigma.empty()) {
                    computation.compute_database().observe_existing_vertex(sigma);
                }
            }
        };

        for (const Index internal : new_internal_nodes) {
            if (!is_active_internal(mesh, internal)) {
                continue;
            }
            observe(primary_addresses(mesh, internal));
            observe(secondary_addresses(mesh, internal));
        }
    }

    [[nodiscard]] static std::vector<PeriodicRequest>
    periodic_requests(const State& state) {
        return state.requested_mirrors();
    }
};


} // namespace highvoronoi


