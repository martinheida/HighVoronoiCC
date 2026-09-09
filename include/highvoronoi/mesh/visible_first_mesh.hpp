
#pragma once

/**
 * @file visible_first_mesh.hpp
 * @brief Read-through HighVoronoi view with all visible nodes first.
 *
 * VisibleFirstMesh exposes every active stable internal HighVoronoi node through
 * one dense bijective public numbering. Active visible nodes occupy the prefix
 * [0, visible_end()) in visible-public insertion order. All remaining active
 * nodes follow in stable internal insertion order. Vertex records, coordinates,
 * address lists, database storage, and the internal boundary remain owned by the
 * HighVoronoiMesh.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/mapped_voronoi_nodes.hpp>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace highvoronoi {

template <class HighMeshT>
class VisibleFirstMesh final
    : public AbstractMesh<
          typename HighMeshT::NodeScalar,
          typename HighMeshT::VertexScalar,
          typename HighMeshT::Index,
          HighMeshT::DimensionAtCompileTime,
          typename HighMeshT::Database,
          typename HighMeshT::AddressList,
          DenseIndexMapping<typename HighMeshT::Index>> {
public:
    using HighMesh = HighMeshT;
    using NodeScalar = typename HighMesh::NodeScalar;
    using VertexScalar = typename HighMesh::VertexScalar;
    using Index = typename HighMesh::Index;
    using Database = typename HighMesh::Database;
    using Address = typename HighMesh::Address;
    using AddressList = typename HighMesh::AddressList;
    using BoundaryType = typename HighMesh::BoundaryType;

    static constexpr int Dim = HighMesh::DimensionAtCompileTime;
    static constexpr int DimensionAtCompileTime = Dim;

    using IndexMapping = DenseIndexMapping<Index>;
    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        Dim,
        Database,
        AddressList,
        IndexMapping>;
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;

private:
    using PublicActiveNodes = detail::MappedVoronoiNodes<
        typename HighMesh::InternalNodes,
        IndexMapping>;

    using ExtendedNodes = ExtendedVoronoiNodes<PublicActiveNodes>;

public:
    explicit VisibleFirstMesh(HighMesh& owner)
        : Base(owner.dimension(), make_mapping(owner)),
          owner_(owner),
          visible_end_(owner.visible_public_count()),
          public_nodes_(owner_.internal_nodes(), this->index_mapping(), owner_.dimension()),
          extended_nodes_(std::in_place, public_nodes_, owner_.internal_boundary()) {}

    /** One-past-the-end public index of the visible-node prefix. */
    [[nodiscard]] Index visible_end() const noexcept {
        return visible_end_;
    }

    [[nodiscard]] Index visible_size() const noexcept {
        return visible_end_;
    }

    [[nodiscard]] bool is_visible_public(Index public_node) const noexcept {
        return public_node < visible_end_;
    }

    [[nodiscard]] Index global_internal_node(Index public_node) const {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    public_node_of_global_internal(Index internal_node) const {
        return this->index_mapping().internal_to_public(internal_node);
    }

    [[nodiscard]] const PublicActiveNodes& concrete_nodes() const noexcept {
        return public_nodes_;
    }

    [[nodiscard]] ExtendedNodes& concrete_extended_nodes() noexcept {
        return *extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodes& concrete_extended_nodes() const noexcept {
        return *extended_nodes_;
    }

    template <class... Args>
    [[nodiscard]] Address store_vertex(Args&&...) {
        throw std::logic_error(
            "VisibleFirstMesh is read-through and cannot store vertices.");
    }

    template <class... Args>
    [[nodiscard]] Address store_infinite_edge(Args&&...) {
        throw std::logic_error(
            "VisibleFirstMesh is read-through and cannot store infinite edges.");
    }

    template <class... Args>
    [[nodiscard]] bool erase_vertex(Args&&...) {
        throw std::logic_error(
            "VisibleFirstMesh is read-through and cannot erase vertices.");
    }

private:
    [[nodiscard]] static IndexMapping make_mapping(const HighMesh& owner) {
        const Index internal_count = owner.internal_node_count();

        std::vector<Index> public_to_internal;
        public_to_internal.reserve(static_cast<std::size_t>(internal_count));

        std::vector<Index> internal_to_public(
            static_cast<std::size_t>(internal_count),
            IndexMapping::invalid_index());

        detail::BitVector selected(static_cast<std::size_t>(internal_count));

        auto append = [&](Index internal) {
            if (internal >= internal_count ||
                !owner.is_active_internal(internal) ||
                selected.test(static_cast<std::size_t>(internal))) {
                return;
            }

            selected.set(static_cast<std::size_t>(internal));
            const Index public_node = static_cast<Index>(public_to_internal.size());
            public_to_internal.push_back(internal);
            internal_to_public[static_cast<std::size_t>(internal)] = public_node;
        };

        // Preserve visible-public insertion order exactly.
        for (Index visible = Index{0}; visible < owner.visible_public_count(); ++visible) {
            append(owner.visible_public_to_internal(visible));
        }

        // Remaining active nodes keep stable internal insertion order.
        for (Index internal = Index{0}; internal < internal_count; ++internal) {
            append(internal);
        }

        IndexMapping mapping;
        mapping.assign(std::move(public_to_internal), std::move(internal_to_public));
        return mapping;
    }

    [[nodiscard]] const NodesAccess& nodes_impl() const noexcept override {
        return extended_nodes_->inner_nodes();
    }

    [[nodiscard]] ExtendedNodesAccess& extended_nodes_impl() noexcept override {
        return *extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return *extended_nodes_;
    }

    [[nodiscard]] const BoundaryType& boundary_impl() const noexcept override {
        return extended_nodes_->boundary();
    }

    void set_boundary_impl(BoundaryType) override {
        throw std::logic_error(
            "VisibleFirstMesh is a read-through structural view.");
    }

    [[nodiscard]] Database& database_impl() noexcept override {
        return owner_.database();
    }

    [[nodiscard]] const Database& database_impl() const noexcept override {
        return owner_.database();
    }

    [[nodiscard]] Index internal_node_count_impl() const noexcept override {
        return owner_.internal_node_count();
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return owner_.primary_addresses_internal(internal_node);
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return owner_.secondary_addresses_internal(internal_node);
    }

    void register_primary_vertex_impl(Index, Address) override {
        throw std::logic_error("VisibleFirstMesh does not accept vertex storage.");
    }

    void register_secondary_vertex_impl(Index, Address) override {
        throw std::logic_error("VisibleFirstMesh does not accept vertex storage.");
    }

    [[nodiscard]] const AddressList& infinite_edge_addresses_impl() const override {
        return owner_.infinite_addresses_internal();
    }

    void register_infinite_edge_impl(Address) override {
        throw std::logic_error(
            "VisibleFirstMesh does not accept infinite-edge storage.");
    }

    [[nodiscard]] bool load_internal_neighbours_impl(
        Index internal_cell,
        std::vector<Index>& buffer) const override {
        return owner_.internal_neighbours(internal_cell, buffer);
    }

    void store_internal_neighbours_impl(
        Index internal_cell,
        const std::vector<Index>& neighbours) override {
        owner_.store_internal_neighbours(internal_cell, neighbours);
    }

    [[nodiscard]] bool internal_neighbours_dirty_impl(
        Index internal_cell) const override {
        return owner_.internal_neighbours_dirty(internal_cell);
    }

    void set_internal_neighbours_dirty_impl(
        Index internal_cell,
        bool value) override {
        owner_.set_internal_neighbours_dirty(internal_cell, value);
    }

    void compute_neighbors_impl(Index internal_cell) override {
        owner_.compute_internal_neighbours(internal_cell);
    }

    void mark_internal_node_deleted_impl(Index) override {
        throw std::logic_error("Delete nodes through HighVoronoiMesh.");
    }

    void cleanup_vertex_lists_impl() override {}

    HighMesh& owner_;
    Index visible_end_;
    PublicActiveNodes public_nodes_;
    std::optional<ExtendedNodes> extended_nodes_;
};

} // namespace highvoronoi








