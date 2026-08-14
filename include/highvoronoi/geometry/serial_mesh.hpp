#pragma once

/**
 * @file serial_mesh.hpp
 * @brief Sequential composite mesh with one stable global internal numbering.
 *
 * `SerialMesh` concatenates independently stored child meshes into one public
 * mesh. The child meshes own their ordinary nodes and their primary/secondary
 * address lists, while all children and the serial mesh refer to the same
 * vertex database. Vertex records are always written by the outer
 * `SerialMesh`, using its global stable internal numbering.
 *
 * @par Why there is only one SerialMesh class
 * Child meshes are stored as `std::unique_ptr<CompatibleMesh>`, where
 * `CompatibleMesh` is the exact `AbstractMesh` specialization required by this
 * serial mesh. This single representation supports both relevant cases:
 *
 * - a homogeneous sequence of many `VoronoiMesh` objects; and
 * - a heterogeneous sequence of different concrete mesh classes deriving from
 *   the same compatible abstract base.
 *
 * A separate `std::vector<ConcreteMesh>` implementation would not work for the
 * current non-copyable and non-movable mesh hierarchy. A vector of references
 * is not a C++ container element type, and a vector of raw pointers or
 * `std::reference_wrapper` would leave lifetime management to the caller.
 * `std::unique_ptr` gives explicit ownership without adding another mesh class.
 *
 * @par Global internal vertex signatures
 * A child mesh must be empty when appended. Its stable local internal node
 * range is assigned one permanent global offset. Afterwards
 * `SerialMesh::store_vertex()` transforms a public signature directly into
 * global internal numbering and stores that signature in the shared database.
 * Registration is routed back to the appropriate child's local address list.
 * Consequently, a child address list may contain addresses whose database
 * signatures use serial-global indices. Such a child is a storage component of
 * the serial mesh and must not be used independently for vertex iteration.
 *
 * Existing child vertices cannot be accepted without explicitly rebasing every
 * stored signature and rebuilding all registrations. The constructor therefore
 * rejects a child containing any active registered database record.
 *
 * @par Runtime-selectable mapping cache
 * Public/internal conversion can be switched at runtime:
 *
 * - with the cache enabled, two dense vectors provide O(1) conversion;
 * - with the cache disabled, conversion is calculated from child offsets and
 *   child mappings, closely following the original Julia algorithm.
 *
 * The cache can be enabled or disabled while the program is running through
 * `set_index_cache_enabled()`. Disabling it clears the vector sizes but retains
 * their capacity, so re-enabling it can usually rebuild without reallocating.
 * The uncached mode saves the dense conversion tables but performs a block
 * search and a short local scan for each conversion.
 *
 * @par Child ownership and structural mutation
 * Appending transfers exclusive ownership to the serial mesh. Child structure
 * and child public numbering must not be mutated afterwards. Node deletion is
 * represented only in the serial mesh's own stable visibility state; child
 * internal storage and child address lists remain unchanged. Tombstoned
 * database addresses are skipped lazily by the common `AbstractMesh`
 * iterators.
 *
 * @par Boundary handling
 * Only the outer serial mesh converts boundary-mirror indices. Children are
 * used solely for ordinary-node routing. This preserves the high-end internal
 * boundary encoding defined by `AbstractMesh` and prevents child-local boundary
 * numbering from leaking into global database signatures.
 */

#include <highvoronoi/geometry/voronoi_mesh.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Sequential composition of compatible meshes sharing one database.
 *
 * @tparam NodeScalarT Scalar used by ordinary and extended nodes.
 * @tparam Dim Compile-time dimension or `highvoronoi::Dynamic`.
 * @tparam DatabaseT Shared vertex database type.
 * @tparam BaseNodeAccessMode Access mode common to all child `nodes()` objects.
 * @tparam AddressListT Address-list type common to all children. The default is
 *         the same thread-policy-aware list used by `VoronoiMesh`.
 *
 * Every child must derive from the exact `CompatibleMesh` specialization. This
 * means that node scalar, vertex scalar, index type, dimension, node access
 * mode, database type, and address-list type agree at compile time. Concrete
 * child types may otherwise differ.
 */
template <typename NodeScalarT,
          int Dim,
          class DatabaseT,
          NodeAccessMode BaseNodeAccessMode = NodeAccessMode::Stored,
          class AddressListT = detail::MeshAddressList<DatabaseT>>
class SerialMesh final
    : public AbstractMesh<
          NodeScalarT,
          typename DatabaseT::Scalar,
          typename DatabaseT::Index,
          Dim,
          BaseNodeAccessMode,
          DatabaseT,
          AddressListT> {
public:
    using Database = DatabaseT;
    using NodeScalar = NodeScalarT;
    using VertexScalar = typename Database::Scalar;
    using Index = typename Database::Index;

    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        Dim,
        BaseNodeAccessMode,
        Database,
        AddressListT>;

    using CompatibleMesh = Base;
    using Address = typename Base::Address;
    using AddressList = typename Base::AddressList;
    using BoundaryType = typename Base::BoundaryType;
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;
    using NodePoint = typename Base::NodePoint;
    using VertexPoint = typename Base::VertexPoint;
    using Sigma = typename Base::Sigma;

    static_assert(
        std::is_same_v<typename AddressList::value_type, Address>,
        "SerialMesh address lists must store database addresses.");

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr NodeAccessMode NodeMode = BaseNodeAccessMode;
    static constexpr NodeAccessMode ExtendedNodeMode =
        Base::ExtendedNodeMode;

private:
    /** @brief One permanently positioned child mesh. */
    struct ChildBlock {
        std::unique_ptr<CompatibleMesh> mesh;
        Index internal_begin = Index{0};
        Index internal_count = Index{0};
        Index child_public_count = Index{0};
        Index active_public_count = Index{0};
    };

    /** @brief Resolved location of one ordinary serial node. */
    struct NodeLocation {
        std::size_t block_index = 0;
        Index local_internal = Index{0};
        Index local_public = Index{0};
        Index global_internal = Index{0};
    };

    /**
     * @brief Public ordinary-node view forwarding into the child sequence.
     *
     * The class implements all possible access hooks. Only the hooks required
     * by `BaseNodeAccessMode` are virtual overrides in a given instantiation;
     * the remaining methods are harmless ordinary members. This permits one
     * compact adapter for stored, computed, and hybrid child-node access.
     */
    class PublicSerialNodes final : public NodesAccess {
    public:
        using Scalar = NodeScalar;
        using IndexType = Index;

        static constexpr int DimensionAtCompileTime = Dim;
        static constexpr NodeAccessMode AccessMode = BaseNodeAccessMode;

        PublicSerialNodes(
            const SerialMesh& owner,
            Index runtime_dimension)
            : NodesAccess(runtime_dimension),
              owner_(&owner) {}

        /** @brief Return the current dense public ordinary-node count. */
        [[nodiscard]] Index size() const noexcept override {
            return owner_->public_node_count_;
        }

    private:
        /** @brief Stored-mode hook returning stable child coordinate storage. */
        [[nodiscard]] const NodeScalar*
        get_stored_node_pointer(Index public_index) const {
            return owner_->stable_public_node_data(public_index);
        }

        /** @brief Computed/hybrid hook materializing one child node. */
        void compute_node(
            Index public_index,
            NodeScalar* target) const {
            owner_->copy_public_node(public_index, target);
        }

        /** @brief Hybrid-mode hook returning storage when the child has it. */
        [[nodiscard]] const NodeScalar*
        try_get_stored_node_pointer(Index public_index) const {
            return owner_->try_stable_public_node_data(public_index);
        }

        const SerialMesh* owner_;
    };

    using ExtendedNodes = ExtendedVoronoiNodes<PublicSerialNodes>;

public:
    /**
     * @brief Construct an empty serial mesh.
     *
     * @param runtime_dimension Runtime dimension. For fixed `Dim` it must equal
     *        `Dim`; for dynamic dimension it must be positive.
     * @param boundary Boundary owned by the outer serial mesh.
     * @param database Shared database used by this mesh and every child.
     * @param enable_index_cache Whether dense public/internal conversion tables
     *        should be maintained from the beginning.
     */
    SerialMesh(
        Index runtime_dimension,
        BoundaryType boundary,
        std::shared_ptr<Database> database,
        bool enable_index_cache = true)
        : Base(runtime_dimension),
          database_(require_database(std::move(database))),
          index_cache_enabled_(enable_index_cache),
          public_nodes_(*this, runtime_dimension),
          extended_nodes_(public_nodes_, std::move(boundary)) {
        validate_internal_capacity(Index{0});
        if (index_cache_enabled_) {
            rebuild_index_cache();
        }
    }

    /**
     * @brief Construct an empty serial mesh with an empty boundary.
     */
    SerialMesh(
        Index runtime_dimension,
        std::shared_ptr<Database> database,
        bool enable_index_cache = true)
        : SerialMesh(
              runtime_dimension,
              BoundaryType{},
              std::move(database),
              enable_index_cache) {}

    /**
     * @brief Append one empty compatible child and transfer its ownership.
     *
     * The child must use the same database object, must have the same runtime
     * dimension, and must not contain active registered vertices. Its complete
     * stable local internal range receives the next global offset. Existing
     * child-deleted internal slots are retained but remain publicly invisible.
     *
     * `std::unique_ptr<Derived>` converts implicitly to the required
     * `std::unique_ptr<CompatibleMesh>`, so both homogeneous and heterogeneous
     * concrete child types use this same function.
     *
     * @return Zero-based index of the appended child block.
     */
    [[nodiscard]] std::size_t append(
        std::unique_ptr<CompatibleMesh> child) {
        if (!child) {
            throw std::invalid_argument(
                "SerialMesh cannot append a null child mesh.");
        }
        if (child->dimension() != this->dimension()) {
            throw std::invalid_argument(
                "SerialMesh child dimension does not match the parent.");
        }
        if (std::addressof(Base::database_of(*child)) != database_.get()) {
            throw std::invalid_argument(
                "SerialMesh children must use the parent's database object.");
        }
        if (child->internal_size() == Index{0}) {
            throw std::invalid_argument(
                "SerialMesh cannot append a child without internal nodes.");
        }

        require_child_without_active_vertices(*child);

        const Index child_internal_count = child->internal_size();
        const Index child_public_count = child->size();
        const Index new_internal_count = checked_count_sum(
            internal_node_count_,
            child_internal_count,
            "SerialMesh internal node count exceeds Index capacity.");
        const Index new_public_count = checked_count_sum(
            public_node_count_,
            child_public_count,
            "SerialMesh public node count exceeds Index capacity.");

        validate_internal_capacity(new_internal_count);
        validate_public_capacity(new_public_count);

        ChildBlock block;
        block.internal_begin = internal_node_count_;
        block.internal_count = child_internal_count;
        block.child_public_count = child_public_count;
        block.active_public_count = child_public_count;
        block.mesh = std::move(child);

        const std::size_t block_index = blocks_.size();
        blocks_.push_back(std::move(block));
        hidden_internal_nodes_.resize(
            static_cast<std::size_t>(new_internal_count),
            std::uint8_t{0});

        internal_node_count_ = new_internal_count;
        public_node_count_ = new_public_count;

        if (index_cache_enabled_) {
            rebuild_index_cache();
        }
        reset_extended_nodes();
        return block_index;
    }

    /**
     * @brief Construct and append a concrete compatible child in place.
     *
     * This convenience function preserves the concrete return type while the
     * child is stored type-erased through `CompatibleMesh`.
     */
    template <class ChildMesh, class... Arguments>
    [[nodiscard]] ChildMesh& emplace_child(Arguments&&... arguments) {
        static_assert(
            std::is_base_of_v<CompatibleMesh, ChildMesh>,
            "ChildMesh must derive from SerialMesh::CompatibleMesh.");

        auto child = std::make_unique<ChildMesh>(
            std::forward<Arguments>(arguments)...);
        ChildMesh& result = *child;
        (void)append(std::move(child));
        return result;
    }

    /** @brief Return the number of appended child blocks. */
    [[nodiscard]] std::size_t child_count() const noexcept {
        return blocks_.size();
    }

    /**
     * @brief Return one child as a const compatible mesh reference.
     *
     * Mutable child access is deliberately not exposed because changing child
     * numbering after append would invalidate serial offsets and caches.
     */
    [[nodiscard]] const CompatibleMesh& child(
        std::size_t block_index) const {
        return *blocks_.at(block_index).mesh;
    }

    /** @brief Return the shared database handle. */
    [[nodiscard]] const std::shared_ptr<Database>&
    database_handle() const noexcept {
        return database_;
    }

    /** @brief Return whether dense index-conversion tables are active. */
    [[nodiscard]] bool index_cache_enabled() const noexcept {
        return index_cache_enabled_;
    }

    /**
     * @brief Enable or disable dense public/internal conversion tables.
     *
     * Enabling immediately rebuilds both directions from child offsets and the
     * permanent hidden-node state. Disabling clears the vector sizes while
     * retaining capacity for a later rebuild.
     */
    void set_index_cache_enabled(bool enabled) {
        if (enabled == index_cache_enabled_) {
            return;
        }

        index_cache_enabled_ = enabled;
        if (enabled) {
            rebuild_index_cache();
        } else {
            public_to_internal_cache_.clear();
            internal_to_public_cache_.clear();
        }
    }

    /**
     * @brief Return the concrete extended-node object.
     *
     * This typed accessor exposes mirror-specific methods not present in the
     * access-mode-independent base interface.
     */
    [[nodiscard]] ExtendedNodes&
    concrete_extended_nodes() noexcept {
        return extended_nodes_;
    }

    /** @brief Const overload of `concrete_extended_nodes()`. */
    [[nodiscard]] const ExtendedNodes&
    concrete_extended_nodes() const noexcept {
        return extended_nodes_;
    }

private:
    // ---------------------------------------------------------------------
    // AbstractMesh primitive hooks
    // ---------------------------------------------------------------------

    /** @brief Return the concatenated public ordinary-node view. */
    [[nodiscard]] const NodesAccess&
    nodes_impl() const noexcept override {
        return extended_nodes_.inner_nodes();
    }

    /** @brief Return mutable outer extended nodes. */
    [[nodiscard]] ExtendedNodesAccess&
    extended_nodes_impl() noexcept override {
        return extended_nodes_;
    }

    /** @brief Return const outer extended nodes. */
    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return extended_nodes_;
    }

    /** @brief Return the boundary owned by the outer serial mesh. */
    [[nodiscard]] const BoundaryType&
    boundary_impl() const noexcept override {
        return extended_nodes_.boundary();
    }

    /**
     * @brief Replace the outer boundary while no active vertices exist.
     *
     * Internal boundary codes are part of stored signatures, so replacement
     * after vertex storage could reinterpret existing records.
     */
    void set_boundary_impl(BoundaryType boundary) override {
        if (has_active_vertices()) {
            throw std::logic_error(
                "Cannot replace a SerialMesh boundary while active vertices "
                "are stored.");
        }

        validate_internal_capacity(
            internal_node_count_,
            boundary.size());
        validate_public_capacity(
            public_node_count_,
            boundary.size());

        extended_nodes_ = ExtendedNodes(
            PublicSerialNodes(*this, this->dimension()),
            std::move(boundary));
    }

    /** @brief Return mutable access to the shared database. */
    [[nodiscard]] Database&
    database_impl() noexcept override {
        return *database_;
    }

    /** @brief Return const access to the shared database. */
    [[nodiscard]] const Database&
    database_impl() const noexcept override {
        return *database_;
    }

    /** @brief Return the permanent global ordinary-node slot count. */
    [[nodiscard]] Index
    internal_node_count_impl() const noexcept override {
        return internal_node_count_;
    }

    /** @brief Convert a serial public node to global stable internal form. */
    [[nodiscard]] Index
    public_node_to_internal_impl(Index public_node) const override {
        if (index_cache_enabled_) {
            return public_to_internal_cache_.at(
                static_cast<std::size_t>(public_node));
        }
        return calculate_public_to_internal(public_node);
    }

    /** @brief Convert a global stable internal node to current serial public form. */
    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        if (internal_node >= internal_node_count_) {
            throw std::out_of_range(
                "SerialMesh internal node index out of range.");
        }
        if (is_hidden(internal_node)) {
            return std::nullopt;
        }

        if (index_cache_enabled_) {
            const Index public_index = internal_to_public_cache_.at(
                static_cast<std::size_t>(internal_node));
            if (public_index == invalid_index()) {
                return std::nullopt;
            }
            return public_index;
        }

        return calculate_internal_to_public(internal_node);
    }

    /** @brief Route a global internal node to its child's primary list. */
    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        const auto [block_index, local_internal] =
            locate_internal_node(internal_node);
        return Base::primary_vertex_addresses_of(
            *blocks_[block_index].mesh,
            local_internal);
    }

    /** @brief Route a global internal node to its child's secondary list. */
    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        const auto [block_index, local_internal] =
            locate_internal_node(internal_node);
        return Base::secondary_vertex_addresses_of(
            *blocks_[block_index].mesh,
            local_internal);
    }

    /** @brief Register a global primary owner in the corresponding child list. */
    void register_primary_vertex_impl(
        Index internal_node,
        Address address) override {
        const auto [block_index, local_internal] =
            locate_internal_node(internal_node);
        Base::register_primary_vertex_at(
            *blocks_[block_index].mesh,
            local_internal,
            address);
    }

    /** @brief Register a global secondary generator in a child list. */
    void register_secondary_vertex_impl(
        Index internal_node,
        Address address) override {
        const auto [block_index, local_internal] =
            locate_internal_node(internal_node);
        Base::register_secondary_vertex_at(
            *blocks_[block_index].mesh,
            local_internal,
            address);
    }

    /** @brief Return this composite mesh's global infinite-edge list. */
    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return infinite_edge_addresses_;
    }

    /** @brief Register one unbounded edge of the composite mesh. */
    void register_infinite_edge_impl(Address address) override {
        infinite_edge_addresses_.push_back(address);
    }

    /**
     * @brief Permanently hide one global internal node from serial public view.
     *
     * The child itself is not renumbered. This is essential because child
     * address lists are only storage partitions; global database signatures
     * belong to the serial mesh.
     */
    void mark_internal_node_deleted_impl(
        Index internal_node) override {
        if (internal_node >= internal_node_count_) {
            throw std::out_of_range(
                "SerialMesh internal node index out of range.");
        }
        if (is_hidden(internal_node)) {
            return;
        }

        const auto [block_index, local_internal] =
            locate_internal_node(internal_node);
        const std::optional<Index> child_public =
            Base::public_node_of(
                *blocks_[block_index].mesh,
                local_internal);
        if (!child_public) {
            throw std::logic_error(
                "SerialMesh attempted to delete an already invisible child node.");
        }

        hidden_internal_nodes_[
            static_cast<std::size_t>(internal_node)] = std::uint8_t{1};

        ChildBlock& block = blocks_[block_index];
        --block.active_public_count;
        --public_node_count_;
        numbering_dirty_ = true;
    }

    /**
     * @brief Finalize deferred serial-numbering changes after filtering.
     *
     * Child address lists remain lazy. Common iterators already skip database
     * tombstones, so no unsafe mutable access to child-private list storage is
     * required here.
     */
    void cleanup_vertex_lists_impl() override {
        if (!numbering_dirty_) {
            return;
        }

        if (index_cache_enabled_) {
            rebuild_index_cache();
        }
        reset_extended_nodes();
        numbering_dirty_ = false;
    }

    // ---------------------------------------------------------------------
    // Node forwarding
    // ---------------------------------------------------------------------

    /** @brief Resolve one serial public node to child-local numbering. */
    [[nodiscard]] NodeLocation locate_public_node(
        Index public_node) const {
        const Index global_internal =
            index_cache_enabled_
                ? public_to_internal_cache_.at(
                      static_cast<std::size_t>(public_node))
                : calculate_public_to_internal(public_node);

        const auto [block_index, local_internal] =
            locate_internal_node(global_internal);
        const std::optional<Index> local_public =
            Base::public_node_of(
                *blocks_[block_index].mesh,
                local_internal);
        if (!local_public) {
            throw std::logic_error(
                "SerialMesh public mapping refers to an invisible child node.");
        }

        return NodeLocation{
            block_index,
            local_internal,
            *local_public,
            global_internal};
    }

    /** @brief Copy a serial public node through the child's generic interface. */
    void copy_public_node(
        Index public_node,
        NodeScalar* target) const {
        const NodeLocation location = locate_public_node(public_node);
        blocks_[location.block_index].mesh->nodes().copy_node(
            location.local_public,
            target);
    }

    /** @brief Return stable child storage for stored serial nodes. */
    [[nodiscard]] const NodeScalar* stable_public_node_data(
        Index public_node) const {
        if constexpr (BaseNodeAccessMode == NodeAccessMode::Stored) {
            const NodeLocation location = locate_public_node(public_node);
            return blocks_[location.block_index]
                .mesh->nodes()
                .stable_node_data(location.local_public);
        } else {
            return nullptr;
        }
    }

    /** @brief Return child storage when available in stored/hybrid mode. */
    [[nodiscard]] const NodeScalar* try_stable_public_node_data(
        Index public_node) const {
        const NodeLocation location = locate_public_node(public_node);
        const NodesAccess& child_nodes =
            blocks_[location.block_index].mesh->nodes();

        if constexpr (BaseNodeAccessMode == NodeAccessMode::Stored) {
            return child_nodes.stable_node_data(location.local_public);
        } else if constexpr (
            BaseNodeAccessMode == NodeAccessMode::Hybrid) {
            return child_nodes.try_stable_node_data(location.local_public);
        } else {
            return nullptr;
        }
    }

    // ---------------------------------------------------------------------
    // Cached and calculated numbering
    // ---------------------------------------------------------------------

    /** @brief Rebuild both dense serial conversion tables. */
    void rebuild_index_cache() {
        public_to_internal_cache_.clear();
        public_to_internal_cache_.reserve(
            static_cast<std::size_t>(public_node_count_));

        internal_to_public_cache_.assign(
            static_cast<std::size_t>(internal_node_count_),
            invalid_index());

        for (std::size_t block_index = 0;
             block_index < blocks_.size();
             ++block_index) {
            const ChildBlock& block = blocks_[block_index];
            for (Index local_public = Index{0};
                 local_public < block.child_public_count;
                 ++local_public) {
                const Index local_internal = Base::internal_node_of(
                    *block.mesh,
                    local_public);
                const Index global_internal = checked_count_sum(
                    block.internal_begin,
                    local_internal,
                    "SerialMesh global internal index overflow.");

                if (is_hidden(global_internal)) {
                    continue;
                }

                const Index public_index = static_cast<Index>(
                    public_to_internal_cache_.size());
                public_to_internal_cache_.push_back(global_internal);
                internal_to_public_cache_[
                    static_cast<std::size_t>(global_internal)] =
                    public_index;
            }
        }

        if (public_to_internal_cache_.size() !=
            static_cast<std::size_t>(public_node_count_)) {
            throw std::logic_error(
                "SerialMesh child numbering changed after append.");
        }
    }

    /** @brief Calculate public-to-internal conversion without dense tables. */
    [[nodiscard]] Index calculate_public_to_internal(
        Index public_node) const {
        Index remaining = public_node;

        for (const ChildBlock& block : blocks_) {
            if (remaining >= block.active_public_count) {
                remaining = static_cast<Index>(
                    remaining - block.active_public_count);
                continue;
            }

            for (Index local_public = Index{0};
                 local_public < block.child_public_count;
                 ++local_public) {
                const Index local_internal = Base::internal_node_of(
                    *block.mesh,
                    local_public);
                const Index global_internal = static_cast<Index>(
                    block.internal_begin + local_internal);

                if (is_hidden(global_internal)) {
                    continue;
                }
                if (remaining == Index{0}) {
                    return global_internal;
                }
                --remaining;
            }

            throw std::logic_error(
                "SerialMesh block visibility count is inconsistent.");
        }

        throw std::out_of_range(
            "SerialMesh public node index out of range.");
    }

    /** @brief Calculate internal-to-public conversion without dense tables. */
    [[nodiscard]] std::optional<Index>
    calculate_internal_to_public(Index internal_node) const {
        if (is_hidden(internal_node)) {
            return std::nullopt;
        }

        const auto [block_index, local_internal] =
            locate_internal_node(internal_node);
        const ChildBlock& block = blocks_[block_index];
        const std::optional<Index> local_public =
            Base::public_node_of(*block.mesh, local_internal);
        if (!local_public) {
            return std::nullopt;
        }

        std::size_t public_position = 0;
        for (std::size_t preceding = 0;
             preceding < block_index;
             ++preceding) {
            public_position += static_cast<std::size_t>(
                blocks_[preceding].active_public_count);
        }

        for (Index candidate_public = Index{0};
             candidate_public < *local_public;
             ++candidate_public) {
            const Index candidate_internal = Base::internal_node_of(
                *block.mesh,
                candidate_public);
            const Index candidate_global = static_cast<Index>(
                block.internal_begin + candidate_internal);
            if (!is_hidden(candidate_global)) {
                ++public_position;
            }
        }

        return static_cast<Index>(public_position);
    }

    // ---------------------------------------------------------------------
    // Child and block validation
    // ---------------------------------------------------------------------

    /** @brief Locate the child block containing one global internal node. */
    [[nodiscard]] std::pair<std::size_t, Index>
    locate_internal_node(Index internal_node) const {
        if (internal_node >= internal_node_count_) {
            throw std::out_of_range(
                "SerialMesh global internal node index out of range.");
        }

        const auto upper = std::upper_bound(
            blocks_.begin(),
            blocks_.end(),
            internal_node,
            [](Index value, const ChildBlock& block) {
                return value < block.internal_begin;
            });

        if (upper == blocks_.begin()) {
            throw std::logic_error(
                "SerialMesh internal block table is inconsistent.");
        }

        const auto block_iterator = std::prev(upper);
        const std::size_t block_index = static_cast<std::size_t>(
            std::distance(blocks_.begin(), block_iterator));
        const Index local_internal = static_cast<Index>(
            internal_node - block_iterator->internal_begin);

        if (local_internal >= block_iterator->internal_count) {
            throw std::logic_error(
                "SerialMesh internal node does not belong to a child block.");
        }

        return {block_index, local_internal};
    }

    /**
     * @brief Reject children carrying locally numbered active vertex records.
     */
    void require_child_without_active_vertices(
        const CompatibleMesh& child) const {
        std::unordered_set<Address> visited;
        Sigma sigma;
        VertexPoint position = this->make_vertex_point();

        for (Index local_internal = Index{0};
             local_internal < child.internal_size();
             ++local_internal) {
            inspect_child_address_list(
                child,
                Base::primary_vertex_addresses_of(
                    child,
                    local_internal),
                visited,
                sigma,
                position);
            inspect_child_address_list(
                child,
                Base::secondary_vertex_addresses_of(
                    child,
                    local_internal),
                visited,
                sigma,
                position);
        }
    }

    /** @brief Inspect one child address list for an active database record. */
    void inspect_child_address_list(
        const CompatibleMesh& child,
        const AddressList& addresses,
        std::unordered_set<Address>& visited,
        Sigma& sigma,
        VertexPoint& position) const {
        const std::size_t count = addresses.size();
        for (std::size_t list_position = 0;
             list_position < count;
             ++list_position) {
            const Address address = addresses[list_position];
            if (!visited.insert(address).second) {
                continue;
            }

            sigma.clear();
            Base::database_of(child).read(address, position, sigma);
            if (!sigma.empty()) {
                throw std::invalid_argument(
                    "SerialMesh can append only children without active "
                    "registered vertices. Existing signatures would require "
                    "global rebasing.");
            }
        }
    }

    /** @brief Return whether one serial-global internal node is hidden. */
    [[nodiscard]] bool is_hidden(Index internal_node) const noexcept {
        return hidden_internal_nodes_[
                   static_cast<std::size_t>(internal_node)] !=
               std::uint8_t{0};
    }

    /** @brief Return whether any registered serial vertex remains active. */
    [[nodiscard]] bool has_active_vertices() const {
        Sigma sigma;
        VertexPoint position = this->make_vertex_point();

        for (Index internal_node = Index{0};
             internal_node < internal_node_count_;
             ++internal_node) {
            const AddressList& addresses =
                primary_vertex_addresses_impl(internal_node);
            const std::size_t count = addresses.size();
            for (std::size_t list_position = 0;
                 list_position < count;
                 ++list_position) {
                sigma.clear();
                database_->read(
                    addresses[list_position],
                    position,
                    sigma);
                if (!sigma.empty()) {
                    return true;
                }
            }
        }
        return false;
    }

    /** @brief Reset active mirrors after public numbering changes. */
    void reset_extended_nodes() {
        BoundaryType boundary_copy = extended_nodes_.boundary();
        extended_nodes_ = ExtendedNodes(
            PublicSerialNodes(*this, this->dimension()),
            std::move(boundary_copy));
    }

    /** @brief Validate and return a non-null database handle. */
    [[nodiscard]] static std::shared_ptr<Database>
    require_database(std::shared_ptr<Database> database) {
        if (!database) {
            throw std::invalid_argument(
                "SerialMesh requires a non-null database.");
        }
        return database;
    }

    /** @brief Sentinel used by the cached internal-to-public table. */
    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return std::numeric_limits<Index>::max();
    }

    /** @brief Add two Index counts with overflow checking. */
    [[nodiscard]] static Index checked_count_sum(
        Index left,
        Index right,
        const char* message) {
        if (left > std::numeric_limits<Index>::max() - right) {
            throw std::overflow_error(message);
        }
        return static_cast<Index>(left + right);
    }

    /**
     * @brief Ensure ordinary internal slots do not overlap boundary codes.
     */
    void validate_internal_capacity(Index count) const {
        validate_internal_capacity(
            count,
            extended_nodes_.boundary().size());
    }

    /**
     * @brief Ensure ordinary slots do not overlap a specified boundary range.
     */
    static void validate_internal_capacity(
        Index count,
        Index boundary_count) {
        const Index first_boundary_code = static_cast<Index>(
            std::numeric_limits<Index>::max() - boundary_count);
        if (count > first_boundary_code) {
            throw std::overflow_error(
                "SerialMesh ordinary internal nodes overlap boundary codes.");
        }
    }

    /**
     * @brief Ensure public ordinary nodes and public mirrors fit into Index.
     */
    void validate_public_capacity(Index count) const {
        validate_public_capacity(
            count,
            extended_nodes_.boundary().size());
    }

    /** @brief Validate public capacity for a specified boundary size. */
    static void validate_public_capacity(
        Index count,
        Index boundary_count) {
        if (count > std::numeric_limits<Index>::max() - boundary_count) {
            throw std::overflow_error(
                "SerialMesh public nodes and boundary mirrors exceed Index.");
        }
    }

    std::shared_ptr<Database> database_;
    AddressList infinite_edge_addresses_;
    std::vector<ChildBlock> blocks_;
    std::vector<std::uint8_t> hidden_internal_nodes_;

    Index internal_node_count_ = Index{0};
    Index public_node_count_ = Index{0};

    bool index_cache_enabled_ = true;
    bool numbering_dirty_ = false;
    std::vector<Index> public_to_internal_cache_;
    std::vector<Index> internal_to_public_cache_;

    PublicSerialNodes public_nodes_;
    ExtendedNodes extended_nodes_;
};

} // namespace highvoronoi
