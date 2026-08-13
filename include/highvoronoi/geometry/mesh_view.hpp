#pragma once

/**
 * @file mesh_view.hpp
 * @brief Non-owning mesh facade that exposes another mesh in a reordered public numbering.
 *
 * `ReorderedMeshView` changes only the public presentation of a compatible
 * mesh. Node coordinates, vertex records, stable internal indices, address
 * lists, the database, and the boundary remain owned by the wrapped mesh.
 *
 * The supplied reversible index view defines the initial public permutation:
 *
 * @code{.cpp}
 * wrapped_public = index_view / reordered_public;
 * reordered_public = index_view * wrapped_public;
 * @endcode
 *
 * This convention matches the original HighVoronoi.jl `MeshView`: the inverse
 * operation translates an index presented to the facade into the numbering of
 * the wrapped mesh, while the forward operation translates the wrapped public
 * numbering into the facade's public numbering.
 *
 * @par Stable internal numbering
 * The permutation is converted once, at construction, into a mapping from the
 * facade's public indices to the wrapped mesh's stable internal indices. The
 * `HVView` object therefore describes the initial ordering only. When nodes are
 * deleted through the facade, surviving nodes retain their relative facade
 * order while public indices are compacted. Stored vertex signatures are never
 * rewritten.
 *
 * @par Mutation and lifetime
 * The facade does not own the wrapped mesh. The wrapped mesh and every object
 * referenced by the chosen `HVView` implementation must outlive the facade.
 * Structural changes of the wrapped mesh must be performed through this
 * facade while it exists; direct node deletion or replacement of the wrapped
 * mesh's boundary would invalidate the facade's cached stable mapping.
 *
 * Vertex storage, lookup, iteration, and filtering use the common algorithms
 * inherited from `AbstractMesh`. The primitive database and address-list hooks
 * are forwarded to the wrapped mesh through `AbstractMesh`'s protected bridge
 * functions.
 */

#include <highvoronoi/geometry/abstract_mesh.hpp>
#include <highvoronoi/detail/hvview.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Present a compatible mesh in a reversible reordered public numbering.
 *
 * @tparam MeshT Concrete wrapped mesh type. It must derive from the exact
 *         `AbstractMesh` specialization reconstructed from its public aliases.
 * @tparam IndexViewT Reversible index view with nested type `Index` and
 *         operations `operator*` and `operator/`.
 *
 * The class is intentionally non-owning. This keeps creation cheap and allows
 * the same mesh to be inspected under short-lived orderings without moving or
 * copying its storage.
 */
template <class MeshT, class IndexViewT>
class ReorderedMeshView final
    : public AbstractMesh<
          typename MeshT::NodeScalar,
          typename MeshT::VertexScalar,
          typename MeshT::Index,
          MeshT::DimensionAtCompileTime,
          MeshT::NodeMode,
          typename MeshT::Database,
          typename MeshT::AddressList> {
public:
    using WrappedMesh = MeshT;
    using IndexView = IndexViewT;
    using Database = typename WrappedMesh::Database;
    using NodeScalar = typename WrappedMesh::NodeScalar;
    using VertexScalar = typename WrappedMesh::VertexScalar;
    using Index = typename WrappedMesh::Index;

    static constexpr int DimensionAtCompileTime =
        WrappedMesh::DimensionAtCompileTime;
    static constexpr NodeAccessMode NodeMode = WrappedMesh::NodeMode;

    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        DimensionAtCompileTime,
        NodeMode,
        Database,
        typename WrappedMesh::AddressList>;

    using Address = typename Base::Address;
    using AddressList = typename Base::AddressList;
    using BoundaryType = typename Base::BoundaryType;
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;
    using NodePoint = typename Base::NodePoint;
    using VertexPoint = typename Base::VertexPoint;
    using Sigma = typename Base::Sigma;

    static constexpr NodeAccessMode ExtendedNodeMode =
        Base::ExtendedNodeMode;

    static_assert(
        std::is_same_v<typename IndexView::Index, Index>,
        "ReorderedMeshView requires identical mesh and view Index types.");
    static_assert(
        std::is_base_of_v<Base, WrappedMesh>,
        "Wrapped mesh must derive from the compatible AbstractMesh specialization.");

private:
    /**
     * @brief Public ordinary-node adapter applying the facade's current order.
     *
     * All three possible access hooks are implemented. Depending on `NodeMode`,
     * only the matching hook is a virtual override; the remaining functions are
     * ordinary members discarded by compile-time branching.
     */
    class PublicReorderedNodes final : public NodesAccess {
    public:
        using Scalar = NodeScalar;
        using IndexType = Index;

        static constexpr int DimensionAtCompileTime =
            ReorderedMeshView::DimensionAtCompileTime;
        static constexpr NodeAccessMode AccessMode = NodeMode;

        /** @brief Construct a node facade referring to its owning mesh facade. */
        PublicReorderedNodes(
            const ReorderedMeshView& owner,
            Index runtime_dimension)
            : NodesAccess(runtime_dimension),
              owner_(&owner) {}

        /** @brief Return the current dense facade node count. */
        [[nodiscard]] Index size() const noexcept override {
            return static_cast<Index>(
                owner_->public_to_internal_.size());
        }

        [[nodiscard]] NodeScalar get_data(
            Index public_index,
            Index coordinate) const override {
            return owner_->get_data(public_index, coordinate);
        }

    private:
        /** @brief Stored-mode hook returning wrapped stable coordinate storage. */
        [[nodiscard]] const NodeScalar*
        get_stored_node_pointer(Index public_index) const {
            return owner_->stable_node_data(public_index);
        }

        /** @brief Computed-mode hook copying one reordered node. */
        void compute_node(
            Index public_index,
            NodeScalar* target) const {
            owner_->copy_node(public_index, target);
        }

        /** @brief Hybrid-mode hook returning storage when the wrapped node has it. */
        [[nodiscard]] const NodeScalar*
        try_get_stored_node_pointer(Index public_index) const {
            return owner_->try_stable_node_data(public_index);
        }

        const ReorderedMeshView* owner_;
    };

    using ExtendedNodes = ExtendedVoronoiNodes<PublicReorderedNodes>;

    static_assert(
        std::is_base_of_v<ExtendedNodesAccess, ExtendedNodes>,
        "ExtendedVoronoiNodes must satisfy AbstractMesh's inferred extended-node interface.");

public:
    /**
     * @brief Construct a non-owning reordered facade.
     *
     * The view is validated as a bijection of the wrapped mesh's current public
     * ordinary-node range. Indices outside that range are irrelevant here;
     * boundary mirror conversion remains the responsibility of `AbstractMesh`.
     *
     * @param mesh Wrapped mesh, which must outlive this object.
     * @param index_view Reversible initial public permutation.
     */
    ReorderedMeshView(
        WrappedMesh& mesh,
        IndexView index_view)
        : Base(mesh.dimension()),
          mesh_(std::addressof(mesh)),
          index_view_(std::move(index_view)),
          wrapped_nodes_(std::addressof(mesh.nodes())),
          public_nodes_(*this, mesh.dimension()) {
        initialize_mapping();
        reset_extended_nodes();
    }

    /** Prevent construction from a temporary wrapped mesh. */
    ReorderedMeshView(WrappedMesh&&, IndexView) = delete;

    /** @brief Concrete extended nodes used by SearchTree safe copies. */
    [[nodiscard]] ExtendedNodes&
    concrete_extended_nodes() noexcept {
        return *extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodes&
    concrete_extended_nodes() const noexcept {
        return *extended_nodes_;
    }

    /**
     * @brief Return read-only access to the wrapped mesh.
     *
     * Mutable access is deliberately not exposed because direct structural
     * mutation would invalidate the facade's stable public mapping.
     */
    [[nodiscard]] const WrappedMesh& wrapped_mesh() const noexcept {
        return *mesh_;
    }

    /**
     * @brief Return the reversible view that defined the initial ordering.
     *
     * After node deletion this object still describes the original permutation;
     * current compact numbering is represented by the facade's stable maps.
     */
    [[nodiscard]] const IndexView& index_view() const noexcept {
        return index_view_;
    }

    /**
     * @brief Translate a current facade node to the wrapped current public index.
     *
     * The conversion passes through the shared stable internal index and is
     * therefore valid after deletions performed through this facade.
     */
    [[nodiscard]] Index wrapped_public_index(
        Index reordered_public_index) const {
        return resolve_wrapped_public_index(reordered_public_index);
    }

    /**
     * @brief Translate a wrapped current public node to facade numbering.
     *
     * @return `std::nullopt` when the stable node is not visible in this facade.
     */
    [[nodiscard]] std::optional<Index> reordered_public_index(
        Index wrapped_public_index_value) const {
        verify_wrapped_structure();
        const Index internal = Base::internal_node_of(
            *mesh_, wrapped_public_index_value);
        return current_public_of_internal(internal);
    }

private:
    // ---------------------------------------------------------------------
    // AbstractMesh primitive hooks
    // ---------------------------------------------------------------------

    /** @brief Return the reordered ordinary-node adapter. */
    [[nodiscard]] const NodesAccess&
    nodes_impl() const noexcept override {
        return public_nodes_;
    }

    /** @brief Return mutable reordered extended nodes. */
    [[nodiscard]] ExtendedNodesAccess&
    extended_nodes_impl() noexcept override {
        return *extended_nodes_;
    }

    /** @brief Return const reordered extended nodes. */
    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return *extended_nodes_;
    }

    /** @brief Return the boundary owned by the wrapped mesh. */
    [[nodiscard]] const BoundaryType&
    boundary_impl() const noexcept override {
        return mesh_->boundary();
    }

    /**
     * @brief Replace the wrapped boundary and reconstruct reordered mirrors.
     */
    void set_boundary_impl(BoundaryType boundary) override {
        mesh_->set_boundary(std::move(boundary));
        reset_extended_nodes();
    }

    /** @brief Forward mutable database access to the wrapped mesh. */
    [[nodiscard]] Database&
    database_impl() noexcept override {
        return Base::database_of(*mesh_);
    }

    /** @brief Forward const database access to the wrapped mesh. */
    [[nodiscard]] const Database&
    database_impl() const noexcept override {
        return Base::database_of(*mesh_);
    }

    /** @brief Return the wrapped stable internal ordinary-node count. */
    [[nodiscard]] Index
    internal_node_count_impl() const noexcept override {
        return mesh_->internal_size();
    }

    /** @brief Map a facade public node directly to wrapped stable numbering. */
    [[nodiscard]] Index
    public_node_to_internal_impl(Index public_node) const override {
        return public_to_internal_.at(
            static_cast<std::size_t>(public_node));
    }

    /** @brief Map a wrapped stable node to current facade numbering. */
    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        return current_public_of_internal(internal_node);
    }

    /** @brief Forward one stable node's primary address list. */
    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return Base::primary_vertex_addresses_of(
            *mesh_, internal_node);
    }

    /** @brief Forward one stable node's secondary address list. */
    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return Base::secondary_vertex_addresses_of(
            *mesh_, internal_node);
    }

    /** @brief Register a primary address in the wrapped mesh. */
    void register_primary_vertex_impl(
        Index internal_node,
        Address address) override {
        Base::register_primary_vertex_at(
            *mesh_, internal_node, address);
    }

    /** @brief Register a secondary address in the wrapped mesh. */
    void register_secondary_vertex_impl(
        Index internal_node,
        Address address) override {
        Base::register_secondary_vertex_at(
            *mesh_, internal_node, address);
    }

    /**
     * @brief Hide one stable node in both the wrapped mesh and this facade.
     *
     * Public compaction is deferred until the coordinated base-class filter has
     * marked every selected node.
     */
    void mark_internal_node_deleted_impl(
        Index internal_node) override {
        Index& current_public = internal_to_public_.at(
            static_cast<std::size_t>(internal_node));
        if (current_public == invalid_index()) {
            return;
        }

        Base::mark_node_deleted_in(*mesh_, internal_node);
        current_public = invalid_index();
        numbering_dirty_ = true;
    }

    /**
     * @brief Trigger wrapped cleanup and compact the facade's public mapping.
     *
     * `AbstractMesh` intentionally exposes no public cleanup-only operation.
     * Running a no-op coordinated filter on the wrapped mesh reaches its
     * optional cleanup hook without deleting additional nodes or vertices.
     */
    void cleanup_vertex_lists_impl() override {
        const auto cleanup_result = mesh_->filter(
            NoNodeFilter{},
            NoVertexFilter{});
        static_cast<void>(cleanup_result);

        if (numbering_dirty_) {
            rebuild_public_mapping();
            reset_extended_nodes();
            numbering_dirty_ = false;
        }

        expected_wrapped_public_count_ = mesh_->size();
    }

    // ---------------------------------------------------------------------
    // Mapping and node-access helpers
    // ---------------------------------------------------------------------

    /** @brief Sentinel used in the stable internal-to-public map. */
    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return std::numeric_limits<Index>::max();
    }

    /**
     * @brief Validate the supplied permutation and snapshot stable indices.
     */
    void initialize_mapping() {
        const Index public_count = mesh_->size();
        const Index internal_count = mesh_->internal_size();

        public_to_internal_.resize(
            static_cast<std::size_t>(public_count));
        internal_to_public_.assign(
            static_cast<std::size_t>(internal_count),
            invalid_index());

        std::vector<unsigned char> seen_wrapped_public(
            static_cast<std::size_t>(public_count),
            static_cast<unsigned char>(0));

        for (std::size_t reordered_position = 0;
             reordered_position < static_cast<std::size_t>(public_count);
             ++reordered_position) {
            const Index reordered_public =
                static_cast<Index>(reordered_position);
            const Index wrapped_public =
                index_view_ / reordered_public;

            if (wrapped_public >= public_count) {
                throw std::invalid_argument(
                    "Mesh view maps a public node outside the wrapped mesh.");
            }

            unsigned char& already_seen = seen_wrapped_public[
                static_cast<std::size_t>(wrapped_public)];
            if (already_seen != static_cast<unsigned char>(0)) {
                throw std::invalid_argument(
                    "Mesh view is not injective on the wrapped public node range.");
            }
            already_seen = static_cast<unsigned char>(1);

            if ((index_view_ * wrapped_public) != reordered_public) {
                throw std::invalid_argument(
                    "Mesh view forward and inverse mappings are inconsistent.");
            }

            const Index internal = Base::internal_node_of(
                *mesh_, wrapped_public);
            Index& mapped_public = internal_to_public_.at(
                static_cast<std::size_t>(internal));
            if (mapped_public != invalid_index()) {
                throw std::invalid_argument(
                    "Mesh view maps two public nodes to one stable internal node.");
            }

            public_to_internal_[reordered_position] = internal;
            mapped_public = reordered_public;
        }

        expected_wrapped_internal_count_ = internal_count;
        expected_wrapped_public_count_ = public_count;
    }

    /**
     * @brief Rebuild dense facade numbering while preserving facade order.
     */
    void rebuild_public_mapping() {
        std::vector<Index> survivors;
        survivors.reserve(public_to_internal_.size());

        for (const Index internal : public_to_internal_) {
            if (internal_to_public_.at(
                    static_cast<std::size_t>(internal)) != invalid_index()) {
                survivors.push_back(internal);
            }
        }

        public_to_internal_ = std::move(survivors);
        std::fill(
            internal_to_public_.begin(),
            internal_to_public_.end(),
            invalid_index());

        for (std::size_t public_position = 0;
             public_position < public_to_internal_.size();
             ++public_position) {
            const Index internal = public_to_internal_[public_position];
            internal_to_public_[static_cast<std::size_t>(internal)] =
                static_cast<Index>(public_position);
        }
    }

    /** @brief Return the current facade index of a stable node if visible. */
    [[nodiscard]] std::optional<Index> current_public_of_internal(
        Index internal_node) const {
        if (internal_node >= expected_wrapped_internal_count_) {
            throw std::out_of_range(
                "Stable internal node index is outside the mesh view.");
        }

        const Index public_index = internal_to_public_.at(
            static_cast<std::size_t>(internal_node));
        if (public_index == invalid_index()) {
            return std::nullopt;
        }
        return public_index;
    }

    /**
     * @brief Resolve a facade public node through stable numbering.
     */
    [[nodiscard]] Index resolve_wrapped_public_index(
        Index reordered_public_index) const {
        verify_wrapped_structure();

        const Index internal = public_to_internal_.at(
            static_cast<std::size_t>(reordered_public_index));
        const std::optional<Index> wrapped_public =
            Base::public_node_of(*mesh_, internal);

        if (!wrapped_public) {
            throw std::logic_error(
                "Wrapped mesh numbering changed outside ReorderedMeshView.");
        }
        return *wrapped_public;
    }

    /**
     * @brief Detect unsupported direct structural mutation of the wrapped mesh.
     */
    void verify_wrapped_structure() const {
        if (mesh_->internal_size() != expected_wrapped_internal_count_ ||
            mesh_->size() != expected_wrapped_public_count_) {
            throw std::logic_error(
                "Wrapped mesh structure changed outside ReorderedMeshView.");
        }
    }

    /** @brief Copy one facade node into caller-owned coordinate storage. */
    void copy_node(Index public_index, NodeScalar* target) const {
        wrapped_nodes_->copy_node(
            resolve_wrapped_public_index(public_index),
            target);
    }

    [[nodiscard]] NodeScalar get_data(
        Index public_index,
        Index coordinate) const {
        return wrapped_nodes_->get_data(
            resolve_wrapped_public_index(public_index),
            coordinate);
    }

    /** @brief Return stable node storage in Stored mode. */
    [[nodiscard]] const NodeScalar* stable_node_data(
        Index public_index) const {
        if constexpr (NodeMode == NodeAccessMode::Stored) {
            return wrapped_nodes_->stable_node_data(
                resolve_wrapped_public_index(public_index));
        } else if constexpr (NodeMode == NodeAccessMode::Hybrid) {
            return wrapped_nodes_->try_stable_node_data(
                resolve_wrapped_public_index(public_index));
        } else {
            static_cast<void>(public_index);
            return nullptr;
        }
    }

    /** @brief Return stable storage when available in Hybrid mode. */
    [[nodiscard]] const NodeScalar* try_stable_node_data(
        Index public_index) const {
        if constexpr (NodeMode == NodeAccessMode::Stored) {
            return wrapped_nodes_->stable_node_data(
                resolve_wrapped_public_index(public_index));
        } else if constexpr (NodeMode == NodeAccessMode::Hybrid) {
            return wrapped_nodes_->try_stable_node_data(
                resolve_wrapped_public_index(public_index));
        } else {
            static_cast<void>(public_index);
            return nullptr;
        }
    }

    /**
     * @brief Reconstruct boundary mirrors using the reordered ordinary nodes.
     */
    void reset_extended_nodes() {
        extended_nodes_ = std::make_unique<ExtendedNodes>(
            public_nodes_,
            mesh_->boundary());
    }

    WrappedMesh* mesh_;
    IndexView index_view_;
    const NodesAccess* wrapped_nodes_;

    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
    Index expected_wrapped_internal_count_ = Index{0};
    Index expected_wrapped_public_count_ = Index{0};
    bool numbering_dirty_ = false;

    PublicReorderedNodes public_nodes_;
    std::unique_ptr<ExtendedNodes> extended_nodes_;
};

/** Deduce mesh and index-view types from constructor arguments. */
template <class MeshT, class IndexViewT>
ReorderedMeshView(MeshT&, IndexViewT)
    -> ReorderedMeshView<MeshT, IndexViewT>;

/**
 * @brief Concise compatibility alias for explicitly named template types.
 *
 * `ReorderedMeshView` is the preferred descriptive class name. `MeshView` is
 * retained as a short alias for code that already uses the Julia terminology.
 */
template <class MeshT, class IndexViewT>
using MeshView = ReorderedMeshView<MeshT, IndexViewT>;

/**
 * @brief Create a reordered facade with C++17 type deduction.
 */
template <class MeshT, class IndexViewT>
[[nodiscard]] auto make_mesh_view(
    MeshT& mesh,
    IndexViewT index_view) {
    return ReorderedMeshView<MeshT, IndexViewT>(
        mesh,
        std::move(index_view));
}

} // namespace highvoronoi
