#pragma once

/**
 * @file abstract_mesh.hpp
 * @brief Common polymorphic infrastructure for HighVoronoi meshes.
 *
 * `AbstractMesh` separates the algorithms shared by all mesh variants from the
 * concrete storage and numbering structures supplied by derived classes. The
 * base class implements public/internal signature conversion, canonical vertex
 * storage, vertex iteration, node and vertex filtering, and the protected
 * bridge operations required by composite meshes and mesh views.
 *
 * @par Public and internal numbering
 * Public node indices are dense and zero-based. Internal node indices are also
 * zero-based but remain stable for the entire lifetime of a mesh. Deleting a
 * node removes only its public representation. Stored vertex signatures always
 * use stable internal indices.
 *
 * Boundary-mirror indices use different public and internal encodings:
 *
 * @code{.cpp}
 * public mirror index   = public_node_count + plane_index;
 * internal mirror index = max(Index) - 1 - plane_index;
 * @endcode
 *
 * `max(Index)` itself is reserved as an invalid-node marker. For a boundary
 * with `B` planes, valid internal mirror indices therefore occupy
 * `[max(Index) - B, max(Index) - 1]`. Boundary conversion is performed only by
 * the outermost mesh; child meshes and views receive only ordinary-node
 * indices through their ordinary mapping hooks.
 *
 * @par Node-access inference
 * The template parameter `BaseNodeAccessMode` describes `nodes()`. The return
 * type of `extended_nodes()` is inferred through the rules implemented in
 * `voronoi_nodes.hpp`:
 *
 * Extended nodes retain the access mode of their base-node container:
 *
 * - stored base nodes produce stored extended nodes;
 * - computed base nodes produce computed extended nodes;
 * - hybrid base nodes produce hybrid extended nodes.
 *
 * @par Required private virtual overrides
 * Every concrete mesh must override the following private pure virtual hooks.
 * C++ permits a derived class to override private virtual functions even though
 * it cannot call them directly.
 *
 * - `nodes_impl()` returns the ordinary public node-access object.
 * - `extended_nodes_impl()` returns the extended node-access object.
 * - `boundary_impl()` and `set_boundary_impl()` expose and replace the boundary.
 * - `database_impl()` returns the vertex database used by the mesh.
 * - `internal_node_count_impl()` returns the stable number of internal slots.
 * - `public_node_to_internal_impl()` maps a current public ordinary node to its
 *   stable internal index.
 * - `internal_node_to_public_impl()` maps a stable internal ordinary node back
 *   to public numbering, or returns `std::nullopt` for a deleted node.
 * - `primary_vertex_addresses_impl()` returns the addresses primarily owned
 *   by one internal ordinary node. The owner is the smallest ordinary internal
 *   index in the canonical internal signature.
 * - `secondary_vertex_addresses_impl()` returns the remaining addresses visible
 *   at one internal ordinary node.
 * - `register_primary_vertex_impl()` and `register_secondary_vertex_impl()`
 *   append newly stored addresses to the corresponding lists.
 * - `mark_internal_node_deleted_impl()` removes an internal node from the
 *   public numbering without changing any internal index.
 *
 * @par Optional private virtual override
 * `cleanup_vertex_lists_impl()` may compact per-node address lists after
 * database records have been tombstoned. The default implementation performs
 * no work; lazy lists remain correct because iteration skips deleted records.
 *
 * @par Concurrency
 * Structural mesh mutation requires external synchronization. In particular,
 * the convenience `erase_vertex(address)` overload reuses scratch buffers owned
 * by the mesh. Concurrent mutation must either be externally serialized or use
 * independent caller-owned buffers through the three-argument overload.
 */

#include <highvoronoi/geometry/boundary.hpp>
#include <highvoronoi/geometry/point.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>
#include <highvoronoi/geometry/mesh_index_mapping.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/** @brief Predicate that selects no node for deletion. */
struct NoNodeFilter {
    template <class Index, class Point>
    [[nodiscard]] constexpr bool operator()(
        Index,
        const Point&) const noexcept {
        return false;
    }
};

/** @brief Predicate that selects no vertex for deletion. */
struct NoVertexFilter {
    template <class Sigma, class Point>
    [[nodiscard]] constexpr bool operator()(
        const Sigma&,
        const Point&) const noexcept {
        return false;
    }

    template <class DeletedNodeList, class Sigma, class Point>
    [[nodiscard]] constexpr bool operator()(
        const DeletedNodeList&,
        const Sigma&,
        const Point&) const noexcept {
        return false;
    }
};

/**
 * @brief Common polymorphic base for compatible HighVoronoi mesh types.
 *
 * @tparam NodeScalarT Scalar used by ordinary and extended nodes.
 * @tparam VertexScalarT Scalar used by stored Voronoi vertices.
 * @tparam IndexT Unsigned integer type used for node indices and signatures.
 * @tparam Dim Compile-time dimension or `highvoronoi::Dynamic`.
 * @tparam BaseNodeAccessMode Compile-time access mode of `nodes()`.
 * @tparam DatabaseT Concrete database type shared by compatible meshes.
 * @tparam AddressListT Vector-like append-only container used for per-node
 *         address lists. It must provide `size()`, a const `operator[]`, and
 *         `push_back(Address)`.
 *
 * `DatabaseT` must expose the aliases `Scalar` and `Index` and the operations
 * `push(r, sigma)`, `read(address, r, sigma)`, `push_facet(r, sigma, u)`,
 * `read_facet(address, r, sigma, u)`, `contains(sigma)`, and
 * `erase(address, sigma)`.
 */
template <typename NodeScalarT,
          typename VertexScalarT,
          typename IndexT,
          int Dim,
          NodeAccessMode BaseNodeAccessMode,
          class DatabaseT,
          class AddressListT = std::vector<std::size_t>,
          class IndexMappingT = VirtualIndexMapping<IndexT>>
class AbstractMesh {
    static_assert(Dim == Dynamic || Dim > 0,
                  "Mesh dimension must be positive or highvoronoi::Dynamic.");
    static_assert(std::is_integral_v<IndexT>,
                  "Mesh Index must be an integral type.");
    static_assert(std::is_unsigned_v<IndexT>,
                  "Mesh Index must be unsigned.");
    static_assert(std::is_same_v<typename DatabaseT::Scalar, VertexScalarT>,
                  "Database scalar type must match VertexScalar.");
    static_assert(std::is_same_v<typename DatabaseT::Index, IndexT>,
                  "Database index type must match mesh Index.");

public:
    using NodeScalar = NodeScalarT;
    using VertexScalar = VertexScalarT;
    using Index = IndexT;
    using Database = DatabaseT;
    using Address = std::size_t;
    using Sigma = std::vector<Index>;
    using DeletedNodeList = std::vector<Index>;
    using AddressList = AddressListT;
    using IndexMapping = IndexMappingT;

    static constexpr bool UsesVirtualIndexMapping =
        IndexMapping::uses_virtual_dispatch;
    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr NodeAccessMode NodeMode = BaseNodeAccessMode;
    static constexpr NodeAccessMode ExtendedNodeMode =
        detail::ExtendedNodeAccessModeV<BaseNodeAccessMode>;

    [[nodiscard]] IndexMapping& index_mapping() noexcept {
        return index_mapping_;
    }

    [[nodiscard]] const IndexMapping& index_mapping() const noexcept {
        return index_mapping_;
    }

    using NodesAccess = detail::NodeAccessBase<
        NodeMode,
        NodeScalar,
        Dim,
        Index>;

    using ExtendedNodesAccess = detail::NodeAccessBase<
        ExtendedNodeMode,
        NodeScalar,
        Dim,
        Index>;

    using NodePoint = typename NodesAccess::Point;
    using VertexPoint = std::conditional_t<
        Dim == Dynamic,
        DynamicPoint<VertexScalar>,
        StaticPoint<VertexScalar, Dim>>;
    using BoundaryType = Boundary<Dim, NodeScalar, Index>;

    /** @brief Owning value returned by a vertex iterator. */
    struct VertexRecord {
        Sigma sigma;
        VertexPoint position;
        Address address = Address{0};
    };

    /** @brief Owning value returned by the infinite-edge iterator. */
    struct InfiniteEdgeRecord {
        Sigma sigma;
        VertexPoint origin;
        VertexPoint direction;
        Address address = Address{0};
    };

    /**
     * @brief Vector-like concatenation of two existing address lists.
     *
     * The object stores references to the two lists and snapshots their sizes
     * when it is constructed. Entries of the first list are returned before
     * entries of the second list. No combined container is allocated.
     */
    template <class List>
    class CombinedAddressList {
    public:
        CombinedAddressList(
            const List& first,
            const List& second)
            : first_(first),
              second_(second),
              first_size_(first.size()),
              second_size_(second.size()) {}

        [[nodiscard]] std::size_t size() const noexcept {
            return first_size_ + second_size_;
        }

        [[nodiscard]] Address operator[](
            std::size_t position) const {
            if (position < first_size_) {
                return first_[position];
            }
            return second_[position - first_size_];
        }

    private:
        const List& first_;
        const List& second_;
        std::size_t first_size_;
        std::size_t second_size_;
    };

    /**
     * @brief Read-only range over active vertices from a vector-like source.
     *
     * `AddressSource` must provide `size()` and a const `operator[]`. It may be
     * a direct reference to one address list or a lightweight combined source.
     * Each iterator owns and recycles its internal/public signature buffers and
     * its position buffer. Deleted records and records whose internal signature
     * no longer has a complete public representation are skipped.
     */
    template <class AddressSource>
    class VertexRange {
    public:
        /** @brief Input iterator yielding owning `VertexRecord` values. */
        class Iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = VertexRecord;
            using difference_type = std::ptrdiff_t;
            using pointer = const VertexRecord*;
            using reference = const VertexRecord&;

            [[nodiscard]] reference operator*() const noexcept {
                return current_;
            }

            [[nodiscard]] pointer operator->() const noexcept {
                return &current_;
            }

            Iterator& operator++() {
                load_next();
                return *this;
            }

            Iterator operator++(int) {
                Iterator previous(*this);
                ++(*this);
                return previous;
            }

            friend bool operator==(
                const Iterator& left,
                const Iterator& right) noexcept {
                if (left.at_end_ && right.at_end_) {
                    return true;
                }
                return std::addressof(left.mesh_) ==
                           std::addressof(right.mesh_) &&
                       left.position_ == right.position_ &&
                       left.at_end_ == right.at_end_;
            }

            friend bool operator!=(
                const Iterator& left,
                const Iterator& right) noexcept {
                return !(left == right);
            }

        private:
            friend class VertexRange;

            Iterator(
                const AbstractMesh& mesh,
                AddressSource addresses,
                std::size_t position,
                bool at_end)
                : mesh_(mesh),
                  addresses_(
                      std::forward<AddressSource>(addresses)),
                  position_(position),
                  at_end_(at_end) {
                current_.position = mesh_.make_vertex_point();
                if (!at_end_) {
                    load_next();
                }
            }

            /** @brief Advance to the next active publicly representable record. */
            void load_next() {
                while (position_ < addresses_.size()) {
                    const Address address =
                        addresses_[position_++];
                    internal_sigma_.clear();

                    mesh_.database_ref().read(
                        address,
                        current_.position,
                        internal_sigma_);

                    if (internal_sigma_.empty()) {
                        continue;
                    }

                    current_.sigma.clear();
                    if (!mesh_.try_make_public_signature(
                            internal_sigma_,
                            current_.sigma)) {
                        continue;
                    }

                    current_.address = address;
                    at_end_ = false;
                    return;
                }

                at_end_ = true;
            }

            const AbstractMesh& mesh_;
            AddressSource addresses_;
            std::size_t position_ = 0;
            bool at_end_ = true;
            Sigma internal_sigma_;
            VertexRecord current_{};
        };

        /** @brief Return an iterator to the first active record. */
        [[nodiscard]] Iterator begin() const {
            return Iterator(
                mesh_,
                addresses_,
                0,
                false);
        }

        /** @brief Return the end sentinel iterator. */
        [[nodiscard]] Iterator end() const {
            return Iterator(
                mesh_,
                addresses_,
                addresses_.size(),
                true);
        }

        /** @brief Return whether the range contains no active record. */
        [[nodiscard]] bool empty() const {
            return begin() == end();
        }

    private:
        friend class AbstractMesh;

        VertexRange(
            const AbstractMesh& mesh,
            AddressSource addresses)
            : mesh_(mesh),
              addresses_(
                  std::forward<AddressSource>(addresses)) {}

        const AbstractMesh& mesh_;
        AddressSource addresses_;
    };

    /**
     * @brief Read-only range over persisted unbounded Voronoi edges.
     *
     * The address list belongs to the concrete storage mesh. Views may forward
     * that list while this common iterator converts each stable internal edge
     * signature into the public numbering of the mesh on which infinite_edges()
     * was requested.
     */
    class InfiniteEdgeRange {
    public:
        class Iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = InfiniteEdgeRecord;
            using difference_type = std::ptrdiff_t;
            using pointer = const InfiniteEdgeRecord*;
            using reference = const InfiniteEdgeRecord&;

            [[nodiscard]] reference operator*() const noexcept {
                return current_;
            }

            [[nodiscard]] pointer operator->() const noexcept {
                return &current_;
            }

            Iterator& operator++() {
                load_next();
                return *this;
            }

            Iterator operator++(int) {
                Iterator previous(*this);
                ++(*this);
                return previous;
            }

            friend bool operator==(
                const Iterator& left,
                const Iterator& right) noexcept {
                if (left.at_end_ && right.at_end_) {
                    return true;
                }
                return std::addressof(left.mesh_) ==
                           std::addressof(right.mesh_) &&
                       left.position_ == right.position_ &&
                       left.at_end_ == right.at_end_;
            }

            friend bool operator!=(
                const Iterator& left,
                const Iterator& right) noexcept {
                return !(left == right);
            }

        private:
            friend class InfiniteEdgeRange;

            Iterator(
                const AbstractMesh& mesh,
                const AddressList& addresses,
                std::size_t position,
                bool at_end)
                : mesh_(mesh),
                  addresses_(addresses),
                  position_(position),
                  at_end_(at_end),
                  current_{
                      {},
                      mesh_.make_vertex_point(),
                      mesh_.make_vertex_point(),
                      Address{0}} {
                if (!at_end_) {
                    load_next();
                }
            }

            void load_next() {
                while (position_ < addresses_.size()) {
                    const Address address = addresses_[position_++];
                    internal_sigma_.clear();

                    mesh_.database_ref().read_facet(
                        address,
                        current_.origin,
                        internal_sigma_,
                        current_.direction);

                    if (internal_sigma_.empty()) {
                        continue;
                    }

                    current_.sigma.clear();
                    if (!mesh_.try_make_public_signature(
                            internal_sigma_,
                            current_.sigma)) {
                        continue;
                    }

                    current_.address = address;
                    at_end_ = false;
                    return;
                }

                at_end_ = true;
            }

            const AbstractMesh& mesh_;
            const AddressList& addresses_;
            std::size_t position_ = 0;
            bool at_end_ = true;
            Sigma internal_sigma_;
            InfiniteEdgeRecord current_;
        };

        [[nodiscard]] Iterator begin() const {
            return Iterator(mesh_, addresses_, 0, false);
        }

        [[nodiscard]] Iterator end() const {
            return Iterator(mesh_, addresses_, addresses_.size(), true);
        }

        [[nodiscard]] bool empty() const {
            return begin() == end();
        }

    private:
        friend class AbstractMesh;

        InfiniteEdgeRange(
            const AbstractMesh& mesh,
            const AddressList& addresses)
            : mesh_(mesh),
              addresses_(addresses) {}

        const AbstractMesh& mesh_;
        const AddressList& addresses_;
    };

    using SingleVertexRange =
        VertexRange<const AddressList&>;
    using CombinedAddressSource =
        CombinedAddressList<AddressList>;
    using CombinedVertexRange =
        VertexRange<CombinedAddressSource>;

    virtual ~AbstractMesh() = default;

    AbstractMesh(const AbstractMesh&) = delete;
    AbstractMesh& operator=(const AbstractMesh&) = delete;
    AbstractMesh(AbstractMesh&&) = delete;
    AbstractMesh& operator=(AbstractMesh&&) = delete;

    /** @brief Return the current number of publicly visible ordinary nodes. */
    [[nodiscard]] Index size() const noexcept {
        return nodes_impl().size();
    }

    /** @brief Synonym for `size()`. */
    [[nodiscard]] Index length() const noexcept {
        return size();
    }

    /** @brief Return the number of stable internal ordinary-node slots. */
    [[nodiscard]] Index internal_size() const noexcept {
        return internal_node_count_impl();
    }

    /** @brief Return the immutable spatial dimension. */
    [[nodiscard]] Index dimension() const noexcept {
        return dimension_;
    }

    /** @brief Return the ordinary public node-access object. */
    [[nodiscard]] const NodesAccess& nodes() const noexcept {
        return nodes_impl();
    }

    /** @brief Return mutable extended nodes used by cell-local algorithms. */
    [[nodiscard]] ExtendedNodesAccess& extended_nodes() noexcept {
        return extended_nodes_impl();
    }

    /** @brief Return the extended node-access object as const. */
    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes() const noexcept {
        return extended_nodes_impl();
    }

    /** @brief Return the current mesh boundary. */
    [[nodiscard]] const BoundaryType& boundary() const noexcept {
        return boundary_impl();
    }

    /** @brief Replace the boundary by copying it. */
    void set_boundary(const BoundaryType& new_boundary) {
        validate_boundary(new_boundary);
        set_boundary_impl(BoundaryType(new_boundary));
    }

    /** @brief Replace the boundary by moving it. */
    void set_boundary(BoundaryType&& new_boundary) {
        validate_boundary(new_boundary);
        set_boundary_impl(std::move(new_boundary));
    }

    /** @brief Create a vertex point with the mesh's runtime dimension. */
    [[nodiscard]] VertexPoint make_vertex_point() const {
        if constexpr (Dim == Dynamic) {
            return VertexPoint(
                static_cast<Eigen::Index>(dimension_));
        } else {
            return VertexPoint{};
        }
    }

    /**
     * @brief Store a vertex using an internally allocated conversion buffer.
     *
     * This convenience overload creates one temporary `Sigma`. Performance-
     * critical callers should use the overload accepting `internal_buffer`.
     */
    template <class RVector, class SigmaLike>
    [[nodiscard]] Address store_vertex(
        const RVector& position,
        const SigmaLike& public_sigma) {
        Sigma internal_buffer;
        return store_vertex(position, public_sigma, internal_buffer);
    }

    /**
     * @brief Store a vertex using caller-owned conversion storage.
     *
     * `internal_buffer` is cleared, filled with the stable internal indices,
     * sorted, and reused without allocation when its capacity is sufficient.
     * It may be the same `Sigma` object as `public_sigma`; in that case the
     * conversion is performed in place.
     *
     * @return One-based database address, or zero if the canonical signature
     *         already exists.
     */
    template <class RVector, class SigmaLike>
    [[nodiscard]] Address store_vertex(
        const RVector& position,
        const SigmaLike& public_sigma,
        Sigma& internal_buffer) {
        require_vertex_dimension(position);
        make_internal_signature(public_sigma, internal_buffer);

        const Index ordinary_count = internal_size();
        if (internal_buffer.front() >= ordinary_count) {
            throw std::invalid_argument(
                "A stored vertex signature must contain an ordinary node.");
        }

        const Address address =
            database_ref().push(position, internal_buffer);

        if (address == Address{0}) {
            return Address{0};
        }

        register_primary_vertex_impl(
            internal_buffer.front(),
            address);

        for (std::size_t position = 1;
             position < internal_buffer.size() &&
             internal_buffer[position] < ordinary_count;
             ++position) {
            register_secondary_vertex_impl(
                internal_buffer[position],
                address);
        }

        return address;
    }

    /**
     * @brief Persist one unbounded Voronoi edge.
     *
     * The complete supporting edge signature is converted to stable internal
     * numbering and used as the database key. `origin` and `direction` are
     * stored as facet payload. Duplicate insertion returns address zero and
     * does not append another address to the mesh-level infinite-edge list.
     */
    template <class SigmaLike, class RVector, class UVector>
    [[nodiscard]] Address store_infinite_edge(
        const SigmaLike& public_full_edge,
        const RVector& origin,
        const UVector& direction) {
        Sigma internal_buffer;
        return store_infinite_edge(
            public_full_edge,
            origin,
            direction,
            internal_buffer);
    }

    /**
     * @brief Persist one unbounded edge using caller-owned signature scratch.
     */
    template <class SigmaLike, class RVector, class UVector>
    [[nodiscard]] Address store_infinite_edge(
        const SigmaLike& public_full_edge,
        const RVector& origin,
        const UVector& direction,
        Sigma& internal_buffer) {
        require_vertex_dimension(origin);
        require_vertex_dimension(direction);
        make_internal_signature(public_full_edge, internal_buffer);

        const Index ordinary_count = internal_size();
        if (internal_buffer.front() >= ordinary_count) {
            throw std::invalid_argument(
                "A stored infinite edge must contain an ordinary node.");
        }

        const Address address = database_ref().push_facet(
            origin,
            internal_buffer,
            direction);

        if (address == Address{0}) {
            return Address{0};
        }

        register_infinite_edge_impl(address);
        return address;
    }

    /** @brief Return every persisted unbounded edge visible through this mesh. */
    [[nodiscard]] InfiniteEdgeRange infinite_edges() const {
        return InfiniteEdgeRange(
            *this,
            infinite_edge_addresses_impl());
    }

    /**
     * @brief Test a public signature using an internally allocated buffer.
     *
     * Performance-critical callers should use the overload accepting a caller-
     * owned `internal_buffer`.
     */
    template <class SigmaLike>
    [[nodiscard]] bool contains_vertex(
        const SigmaLike& public_sigma) const {
        Sigma internal_buffer;
        return contains_vertex(public_sigma, internal_buffer);
    }

    /**
     * @brief Test a public signature using caller-owned conversion storage.
     *
     * The buffer is canonicalized exactly as in `store_vertex()` and retains
     * its capacity for subsequent calls.
     */
    template <class SigmaLike>
    [[nodiscard]] bool contains_vertex(
        const SigmaLike& public_sigma,
        Sigma& internal_buffer) const {
        make_internal_signature(public_sigma, internal_buffer);
        return database_ref().contains(internal_buffer);
    }

    /**
     * @brief Return vertices primarily owned by one public ordinary node.
     */
    [[nodiscard]] SingleVertexRange primary_vertices(
        Index public_node) const {
        require_public_node(public_node);
        const Index internal_node =
            map_public_node_to_internal(public_node);
        return SingleVertexRange(
            *this,
            primary_vertex_addresses_impl(internal_node));
    }

    /**
     * @brief Return vertices secondarily registered at one public ordinary node.
     */
    [[nodiscard]] SingleVertexRange secondary_vertices(
        Index public_node) const {
        require_public_node(public_node);
        const Index internal_node =
            map_public_node_to_internal(public_node);
        return SingleVertexRange(
            *this,
            secondary_vertex_addresses_impl(internal_node));
    }

    /**
     * @brief Return all visible vertices, primary first and secondary second.
     */
    [[nodiscard]] CombinedVertexRange
    vertices(Index public_node) const {
        require_public_node(public_node);
        const Index internal_node =
            map_public_node_to_internal(public_node);
        return CombinedVertexRange(
            *this,
            CombinedAddressSource(
                primary_vertex_addresses_impl(internal_node),
                secondary_vertex_addresses_impl(internal_node)));
    }

    /** @brief Invoke a function for each active vertex of one public node. */
    template <class Function>
    void for_each_vertex(
        Index public_node,
        Function&& function) const {
        for (const VertexRecord& record : vertices(public_node)) {
            function(record);
        }
    }

    /**
     * @brief Erase one database record using mesh-owned scratch storage.
     *
     * This overload performs no per-call vector allocation after the reusable
     * scratch buffers have acquired sufficient capacity. It is not safe for
     * concurrent calls on the same mesh without external synchronization.
     */
    [[nodiscard]] bool erase_vertex(Address address) {
        return erase_vertex(
            address,
            erase_sigma_buffer_,
            erase_position_buffer_);
    }

    /**
     * @brief Erase one record using caller-owned signature storage.
     *
     * The position buffer remains mesh-owned because its values are needed only
     * to satisfy the database read interface. Concurrent calls still require
     * external synchronization; use the three-argument overload for independent
     * per-thread storage.
     */
    [[nodiscard]] bool erase_vertex(
        Address address,
        Sigma& internal_sigma_buffer) {
        return erase_vertex(
            address,
            internal_sigma_buffer,
            erase_position_buffer_);
    }

    /**
     * @brief Erase one record using entirely caller-owned read buffers.
     *
     * This is the appropriate primitive for a future parallel mutation path in
     * which each worker owns separate scratch storage. Merely writing the same
     * unused position object from multiple C++ threads would still constitute a
     * data race and therefore undefined behaviour.
     */
    [[nodiscard]] bool erase_vertex(
        Address address,
        Sigma& internal_sigma_buffer,
        VertexPoint& position_buffer) {
        prepare_vertex_point(position_buffer);
        internal_sigma_buffer.clear();
        database_ref().read(
            address,
            position_buffer,
            internal_sigma_buffer);

        if (internal_sigma_buffer.empty()) {
            return false;
        }

        return database_ref().erase(
            address,
            internal_sigma_buffer);
    }

    /**
     * @brief Delete every public node selected by a callable predicate.
     *
     * The callable is invoked as
     * `predicate(public_index, owning_node_point)` in ascending public order.
     * A lambda, function object, function pointer, or `std::function` can be
     * used. `NodePredicate&&` is a forwarding reference: lvalues are borrowed,
     * temporary lambdas are accepted directly, and no `std::move` is required.
     *
     * Vertices touching a selected node are tombstoned before the public node
     * mapping is changed.
     */
    template <class NodePredicate>
    std::size_t erase_nodes_if(NodePredicate&& predicate) {
        DeletedNodeList deleted_public_nodes;
        return erase_nodes_if(
            std::forward<NodePredicate>(predicate),
            deleted_public_nodes);
    }

    /**
     * @brief Delete selected nodes and return their old public indices.
     *
     * `deleted_public_nodes` is cleared and filled in strictly ascending order.
     * Its entries refer to the public numbering that existed before deletion.
     * Supplying this buffer permits its capacity to be recycled across calls.
     */
    template <class NodePredicate>
    std::size_t erase_nodes_if(
        NodePredicate&& predicate,
        DeletedNodeList& deleted_public_nodes) {
        Sigma deleted_internal_nodes;
        collect_selected_nodes(
            predicate,
            deleted_public_nodes,
            deleted_internal_nodes);

        if (deleted_internal_nodes.empty()) {
            return 0;
        }

        Sigma internal_sigma_buffer;
        VertexPoint position_buffer = make_vertex_point();

        // Both lists matter because every vertex visible at a deleted node
        // must be removed. Repeated addresses become harmless tombstone reads.
        for (const Index internal_node : deleted_internal_nodes) {
            const CombinedAddressSource addresses(
                primary_vertex_addresses_impl(internal_node),
                secondary_vertex_addresses_impl(internal_node));

            for (std::size_t position = 0;
                 position < addresses.size();
                 ++position) {
                (void)erase_vertex(
                    addresses[position],
                    internal_sigma_buffer,
                    position_buffer);
            }
        }

        erase_infinite_edges_touching(deleted_internal_nodes);

        for (const Index internal_node : deleted_internal_nodes) {
            mark_internal_node_deleted_impl(internal_node);
        }

        cleanup_vertex_lists_impl();
        return deleted_public_nodes.size();
    }

    /**
     * @brief Delete every active vertex selected by a callable predicate.
     *
     * The callable is invoked as `predicate(public_sigma, position)`. Like the
     * node predicate, it may be an lvalue function object or a temporary lambda;
     * no explicit move is needed.
     *
     * Only primary lists are traversed. Every stored vertex therefore appears
     * exactly once without constructing a combined address list.
     */
    template <class VertexPredicate>
    std::size_t erase_vertices_if(VertexPredicate&& predicate) {
        Sigma internal_sigma;
        Sigma public_sigma;
        VertexPoint position = make_vertex_point();
        std::size_t erased = 0;

        const Index count = internal_size();
        for (Index internal_node = Index{0};
             internal_node < count;
             ++internal_node) {
            const AddressList& addresses =
                primary_vertex_addresses_impl(internal_node);

            const std::size_t address_count =
                addresses.size();
            for (std::size_t address_position = 0;
                 address_position < address_count;
                 ++address_position) {
                const Address address =
                    addresses[address_position];
                internal_sigma.clear();
                database_ref().read(address, position, internal_sigma);

                if (internal_sigma.empty()) {
                    continue;
                }

                public_sigma.clear();
                const bool publicly_valid =
                    try_make_public_signature(
                        internal_sigma,
                        public_sigma);

                if (!publicly_valid ||
                    predicate(public_sigma, position)) {
                    if (database_ref().erase(address, internal_sigma)) {
                        ++erased;
                    }
                }
            }
        }

        cleanup_vertex_lists_impl();
        return erased;
    }

    /**
     * @brief Apply a coordinated node-and-vertex filter.
     *
     * The node predicate is evaluated first without modifying the mesh. Its
     * selected public indices are sorted naturally by the ascending traversal.
     * While the old public numbering is still intact, the vertex predicate is
     * invoked for every active vertex as
     *
     * @code{.cpp}
     * vertex_predicate(deleted_public_nodes, public_sigma, position)
     * @endcode
     *
     * A vertex is erased when it touches a selected node or when the vertex
     * predicate returns true. Selected nodes are removed from public numbering
     * only after this vertex pass. Thus both `deleted_public_nodes` and
     * `public_sigma` use the same pre-filter public numbering.
     *
     * @return `(deleted_node_count, deleted_vertex_count)`, where the second
     *         value includes vertices removed because they touched a deleted
     *         node.
     */
    template <class NodePredicate, class VertexPredicate>
    [[nodiscard]] std::pair<std::size_t, std::size_t> filter(
        NodePredicate&& node_predicate,
        VertexPredicate&& vertex_predicate) {
        DeletedNodeList deleted_public_nodes;
        return filter(
            std::forward<NodePredicate>(node_predicate),
            std::forward<VertexPredicate>(vertex_predicate),
            deleted_public_nodes);
    }

    /**
     * @brief Coordinated filter using a caller-owned deleted-node buffer.
     *
     * `deleted_public_nodes` is cleared, filled in ascending pre-filter public
     * order, passed by const reference to every vertex-predicate call, and left
     * available to the caller after the operation.
     */
    template <class NodePredicate, class VertexPredicate>
    [[nodiscard]] std::pair<std::size_t, std::size_t> filter(
        NodePredicate&& node_predicate,
        VertexPredicate&& vertex_predicate,
        DeletedNodeList& deleted_public_nodes) {
        Sigma deleted_internal_nodes;
        collect_selected_nodes(
            node_predicate,
            deleted_public_nodes,
            deleted_internal_nodes);

        std::sort(
            deleted_internal_nodes.begin(),
            deleted_internal_nodes.end());

        Sigma internal_sigma;
        Sigma public_sigma;
        VertexPoint position = make_vertex_point();
        std::size_t erased_vertices = 0;

        const Index count = internal_size();
        for (Index internal_node = Index{0};
             internal_node < count;
             ++internal_node) {
            const AddressList& addresses =
                primary_vertex_addresses_impl(internal_node);

            const std::size_t address_count =
                addresses.size();
            for (std::size_t address_position = 0;
                 address_position < address_count;
                 ++address_position) {
                const Address address =
                    addresses[address_position];
                internal_sigma.clear();
                database_ref().read(address, position, internal_sigma);

                if (internal_sigma.empty()) {
                    continue;
                }

                public_sigma.clear();
                const bool publicly_valid =
                    try_make_public_signature(
                        internal_sigma,
                        public_sigma);

                const bool touches_deleted_node =
                    intersects_sorted(
                        internal_sigma,
                        deleted_internal_nodes);

                const bool predicate_deletes =
                    publicly_valid &&
                    vertex_predicate(
                        deleted_public_nodes,
                        public_sigma,
                        position);

                if (!publicly_valid ||
                    touches_deleted_node ||
                    predicate_deletes) {
                    if (database_ref().erase(address, internal_sigma)) {
                        ++erased_vertices;
                    }
                }
            }
        }

        erase_infinite_edges_touching(deleted_internal_nodes);

        for (const Index internal_node : deleted_internal_nodes) {
            mark_internal_node_deleted_impl(internal_node);
        }

        cleanup_vertex_lists_impl();
        return {
            deleted_public_nodes.size(),
            erased_vertices};
    }

protected:
    /**
     * @brief Construct the common mesh base with an immutable dimension.
     */
    explicit AbstractMesh(Index runtime_dimension)
        : dimension_(checked_dimension(runtime_dimension)),
          index_mapping_(),
          erase_position_buffer_(make_vertex_point()) {}

    explicit AbstractMesh(
        Index runtime_dimension,
        IndexMapping index_mapping)
        : dimension_(checked_dimension(runtime_dimension)),
          index_mapping_(std::move(index_mapping)),
          erase_position_buffer_(make_vertex_point()) {}

    // ---------------------------------------------------------------------
    // Protected bridges for composite meshes and views
    // ---------------------------------------------------------------------

    /** @brief Map another mesh's public ordinary node to stable internal form. */
    [[nodiscard]] static Index internal_node_of(
        const AbstractMesh& mesh,
        Index public_node) {
        mesh.require_public_node(public_node);
        return mesh.map_public_node_to_internal(public_node);
    }

    /** @brief Map another mesh's internal node to public form if still visible. */
    [[nodiscard]] static std::optional<Index> public_node_of(
        const AbstractMesh& mesh,
        Index internal_node) {
        if (internal_node >= mesh.internal_size()) {
            throw std::out_of_range(
                "Internal node index out of range.");
        }
        return mesh.map_internal_node_to_public(internal_node);
    }

    /** @brief Access another compatible mesh's mutable database. */
    [[nodiscard]] static Database& database_of(
        AbstractMesh& mesh) noexcept {
        return mesh.database_impl();
    }

    /** @brief Access another compatible mesh's const database. */
    [[nodiscard]] static const Database& database_of(
        const AbstractMesh& mesh) noexcept {
        return mesh.database_impl();
    }

    /** @brief Access another mesh's primary list for an internal node. */
    [[nodiscard]] static const AddressList&
    primary_vertex_addresses_of(
        const AbstractMesh& mesh,
        Index internal_node) {
        return mesh.primary_vertex_addresses_impl(internal_node);
    }

    /** @brief Access another mesh's secondary list for an internal node. */
    [[nodiscard]] static const AddressList&
    secondary_vertex_addresses_of(
        const AbstractMesh& mesh,
        Index internal_node) {
        return mesh.secondary_vertex_addresses_impl(internal_node);
    }

    /** @brief Access another mesh's global infinite-edge address list. */
    [[nodiscard]] static const AddressList&
    infinite_edge_addresses_of(
        const AbstractMesh& mesh) {
        return mesh.infinite_edge_addresses_impl();
    }

    /** @brief Register an infinite-edge address in another compatible mesh. */
    static void register_infinite_edge_at(
        AbstractMesh& mesh,
        Address address) {
        mesh.register_infinite_edge_impl(address);
    }

    /** @brief Register a primary address in another compatible mesh. */
    static void register_primary_vertex_at(
        AbstractMesh& mesh,
        Index internal_node,
        Address address) {
        mesh.register_primary_vertex_impl(internal_node, address);
    }

    /** @brief Register a secondary address in another compatible mesh. */
    static void register_secondary_vertex_at(
        AbstractMesh& mesh,
        Index internal_node,
        Address address) {
        mesh.register_secondary_vertex_impl(internal_node, address);
    }

    /** @brief Mark another mesh's internal node as publicly deleted. */
    static void mark_node_deleted_in(
        AbstractMesh& mesh,
        Index internal_node) {
        mesh.mark_internal_node_deleted_impl(internal_node);
    }

    /**
     * @brief Apply another mesh's complete boundary-aware public mapping.
     *
     * Composite meshes must use `internal_node_of()` when forwarding an
     * ordinary index to a child. Only the outermost mesh should use this
     * boundary-aware bridge.
     */
    [[nodiscard]] static Index internal_index_of(
        const AbstractMesh& mesh,
        Index public_index) {
        return mesh.public_index_to_internal(public_index);
    }

    /**
     * @brief Apply another mesh's complete boundary-aware internal mapping.
     *
     * Composite meshes must use `public_node_of()` for ordinary child indices.
     */
    [[nodiscard]] static std::optional<Index> public_index_of(
        const AbstractMesh& mesh,
        Index internal_index) {
        return mesh.internal_index_to_public(internal_index);
    }

private:
    // ---------------------------------------------------------------------
    // Required and optional virtual primitives
    // ---------------------------------------------------------------------

    /** @brief Required: return the ordinary public node-access object. */
    [[nodiscard]] virtual const NodesAccess&
    nodes_impl() const noexcept = 0;

    /** @brief Required: return mutable extended nodes. */
    [[nodiscard]] virtual ExtendedNodesAccess&
    extended_nodes_impl() noexcept = 0;

    /** @brief Required: return const extended nodes. */
    [[nodiscard]] virtual const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept = 0;

    /** @brief Required: return the current boundary. */
    [[nodiscard]] virtual const BoundaryType&
    boundary_impl() const noexcept = 0;

    /** @brief Required: replace the boundary and refresh derived caches. */
    virtual void set_boundary_impl(BoundaryType boundary) = 0;

    /** @brief Required: return the mutable shared vertex database. */
    [[nodiscard]] virtual Database&
    database_impl() noexcept = 0;

    /** @brief Required: return the const shared vertex database. */
    [[nodiscard]] virtual const Database&
    database_impl() const noexcept = 0;

    /** @brief Required: return the stable number of internal ordinary nodes. */
    [[nodiscard]] virtual Index
    internal_node_count_impl() const noexcept = 0;

    /** @brief Policy-aware public-to-internal mapping. */
    [[nodiscard]] Index map_public_node_to_internal(Index public_node) const {
        if constexpr (UsesVirtualIndexMapping) {
            return public_node_to_internal_impl(public_node);
        } else {
            return index_mapping_.public_to_internal(public_node);
        }
    }

    /** @brief Policy-aware internal-to-public mapping. */
    [[nodiscard]] std::optional<Index>
    map_internal_node_to_public(Index internal_node) const {
        if constexpr (UsesVirtualIndexMapping) {
            return internal_node_to_public_impl(internal_node);
        } else {
            return index_mapping_.internal_to_public(internal_node);
        }
    }

    /** @brief Required: map a public ordinary node to stable internal form. */
    [[nodiscard]] virtual Index
    public_node_to_internal_impl(Index public_node) const = 0;

    /** @brief Required: map a stable internal node to public form if visible. */
    [[nodiscard]] virtual std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const = 0;

    /** @brief Required: return the primary list of one internal node. */
    [[nodiscard]] virtual const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const = 0;

    /** @brief Required: return the secondary list of one internal node. */
    [[nodiscard]] virtual const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const = 0;

    /** @brief Required: register an address at its unique primary owner. */
    virtual void register_primary_vertex_impl(
        Index internal_node,
        Address address) = 0;

    /** @brief Required: register an address at one secondary node. */
    virtual void register_secondary_vertex_impl(
        Index internal_node,
        Address address) = 0;

    /** @brief Required: return the global infinite-edge address list. */
    [[nodiscard]] virtual const AddressList&
    infinite_edge_addresses_impl() const = 0;

    /** @brief Required: register one persisted infinite-edge address. */
    virtual void register_infinite_edge_impl(
        Address address) = 0;

    /** @brief Required: remove one internal node from public numbering. */
    virtual void mark_internal_node_deleted_impl(
        Index internal_node) = 0;

    /** @brief Optional: compact tombstoned entries from per-node lists. */
    virtual void cleanup_vertex_lists_impl() {}

    // ---------------------------------------------------------------------
    // Common helpers
    // ---------------------------------------------------------------------

    /** @brief Return the mutable database through the virtual hook. */
    [[nodiscard]] Database& database_ref() noexcept {
        return database_impl();
    }

    /** @brief Return the const database through the virtual hook. */
    [[nodiscard]] const Database& database_ref() const noexcept {
        return database_impl();
    }

    /** @brief Validate a public ordinary-node index. */
    void require_public_node(Index public_node) const {
        if (public_node >= size()) {
            throw std::out_of_range(
                "Public mesh-node index out of range.");
        }
    }

    /**
     * @brief Convert a complete public index to stable internal form.
     *
     * Ordinary indices are delegated to the derived mesh. Boundary mirrors are
     * encoded here at the upper end of the `Index` range and are never passed
     * to a child mesh.
     */
    [[nodiscard]] Index public_index_to_internal(
        Index public_index) const {
        const Index public_nodes = size();
        if (public_index < public_nodes) {
            return map_public_node_to_internal(public_index);
        }

        const Index plane = static_cast<Index>(
            public_index - public_nodes);
        if (plane >= boundary().size()) {
            throw std::out_of_range(
                "Public index is neither a node nor a boundary mirror.");
        }

        return static_cast<Index>(
            std::numeric_limits<Index>::max() - Index{1} - plane);
    }

    /**
     * @brief Convert a complete internal index to current public form.
     *
     * Boundary mirrors are recognized before ordinary-node routing. Their
     * plane number is decoded as `max(Index) - (internal_index + 1)`. The
     * parentheses are intentional and prevent an unsafe rearrangement of the
     * unsigned arithmetic.
     */
    [[nodiscard]] std::optional<Index> internal_index_to_public(
        Index internal_index) const {
        const Index maximum = std::numeric_limits<Index>::max();
        if (internal_index == maximum) {
            throw std::out_of_range(
                "The maximum Index value is reserved as an invalid node.");
        }

        const Index boundary_count = boundary().size();
        const Index first_boundary = static_cast<Index>(
            maximum - boundary_count);

        if (boundary_count != Index{0} &&
            internal_index >= first_boundary) {
            const Index plane = static_cast<Index>(
                maximum -
                static_cast<Index>(internal_index + Index{1}));
            return checked_add(size(), plane);
        }

        if (internal_index >= internal_size()) {
            throw std::out_of_range(
                "Internal index is neither a node nor a boundary mirror.");
        }

        return map_internal_node_to_public(internal_index);
    }

    /**
     * @brief Convert and canonicalize a public signature in caller storage.
     */
    template <class SigmaLike>
    void make_internal_signature(
        const SigmaLike& public_sigma,
        Sigma& internal_buffer) const {
        if (public_sigma.size() == 0) {
            throw std::invalid_argument(
                "A Voronoi vertex signature must not be empty.");
        }

        using BareSigmaLike = std::remove_cv_t<
            std::remove_reference_t<SigmaLike>>;

        if constexpr (std::is_same_v<BareSigmaLike, Sigma>) {
            if (std::addressof(public_sigma) ==
                std::addressof(internal_buffer)) {
                for (Index& index : internal_buffer) {
                    index = public_index_to_internal(index);
                }
                canonicalize_signature(internal_buffer);
                return;
            }
        }

        internal_buffer.clear();
        internal_buffer.reserve(
            static_cast<std::size_t>(public_sigma.size()));

        for (std::size_t position = 0;
             position < static_cast<std::size_t>(public_sigma.size());
             ++position) {
            internal_buffer.push_back(
                public_index_to_internal(
                    static_cast<Index>(public_sigma[position])));
        }

        canonicalize_signature(internal_buffer);
    }

    /** @brief Convert an internal signature to sorted public numbering. */
    [[nodiscard]] bool try_make_public_signature(
        const Sigma& internal_sigma,
        Sigma& public_sigma) const {
        public_sigma.clear();
        public_sigma.reserve(internal_sigma.size());

        for (const Index internal_index : internal_sigma) {
            const std::optional<Index> public_index =
                internal_index_to_public(internal_index);
            if (!public_index) {
                public_sigma.clear();
                return false;
            }
            public_sigma.push_back(*public_index);
        }

        std::sort(public_sigma.begin(), public_sigma.end());
        return true;
    }

    /** @brief Sort a signature and reject duplicate generating indices. */
    static void canonicalize_signature(Sigma& sigma) {
        std::sort(sigma.begin(), sigma.end());
        if (std::adjacent_find(sigma.begin(), sigma.end()) != sigma.end()) {
            throw std::invalid_argument(
                "A Voronoi vertex signature must not contain duplicates.");
        }
    }

    /** @brief Validate the dimension of a vertex coordinate vector. */
    template <class RVector>
    void require_vertex_dimension(const RVector& position) const {
        if (static_cast<std::size_t>(position.size()) !=
            static_cast<std::size_t>(dimension())) {
            throw std::invalid_argument(
                "Vertex dimension does not match mesh dimension.");
        }
    }

    /** @brief Ensure a caller-owned dynamic point has the correct size. */
    void prepare_vertex_point(VertexPoint& point) const {
        if constexpr (Dim == Dynamic) {
            point.resize(static_cast<Eigen::Index>(dimension_));
        }
    }

    /**
     * @brief Collect selected public and internal nodes before any mutation.
     */
    template <class NodePredicate>
    void collect_selected_nodes(
        NodePredicate&& predicate,
        DeletedNodeList& deleted_public_nodes,
        Sigma& deleted_internal_nodes) const {
        deleted_public_nodes.clear();
        deleted_internal_nodes.clear();

        const Index public_count = size();
        deleted_public_nodes.reserve(
            static_cast<std::size_t>(public_count));
        deleted_internal_nodes.reserve(
            static_cast<std::size_t>(public_count));

        for (Index public_index = Index{0};
             public_index < public_count;
             ++public_index) {
            const NodePoint node = nodes().node(public_index);
            if (predicate(public_index, node)) {
                deleted_public_nodes.push_back(public_index);
                deleted_internal_nodes.push_back(
                    map_public_node_to_internal(public_index));
            }
        }
    }

    /**
     * @brief Tombstone persisted infinite edges touching deleted internal nodes.
     *
     * Infinite edges are globally addressed rather than registered per node,
     * so node deletion cannot discover them through primary/secondary vertex
     * lists. The complete internal edge signature is therefore read from the
     * global infinite-edge address list and removed from the database hash when
     * it intersects the deleted-node set. Tombstoned list entries remain safe:
     * InfiniteEdgeRange skips them and concrete storage meshes may compact them.
     */
    void erase_infinite_edges_touching(
        const Sigma& deleted_internal_nodes) {
        if (deleted_internal_nodes.empty()) {
            return;
        }

        Sigma internal_sigma;
        VertexPoint origin = make_vertex_point();
        VertexPoint direction = make_vertex_point();
        const AddressList& addresses = infinite_edge_addresses_impl();
        const std::size_t count = addresses.size();

        for (std::size_t position = 0; position < count; ++position) {
            const Address address = addresses[position];
            internal_sigma.clear();
            database_ref().read_facet(
                address,
                origin,
                internal_sigma,
                direction);

            if (internal_sigma.empty()) {
                continue;
            }

            if (intersects_sorted(
                    internal_sigma,
                    deleted_internal_nodes)) {
                (void)database_ref().erase(
                    address,
                    internal_sigma);
            }
        }
    }

    /** @brief Test whether two sorted index vectors have a common entry. */
    [[nodiscard]] static bool intersects_sorted(
        const Sigma& left,
        const Sigma& right) noexcept {
        std::size_t i = 0;
        std::size_t j = 0;

        while (i < left.size() && j < right.size()) {
            if (left[i] < right[j]) {
                ++i;
            } else if (right[j] < left[i]) {
                ++j;
            } else {
                return true;
            }
        }

        return false;
    }

    /** @brief Validate that a replacement boundary has the mesh dimension. */
    void validate_boundary(const BoundaryType& candidate) const {
        if (candidate.dimension() != 0 &&
            candidate.dimension() !=
                static_cast<std::size_t>(dimension_)) {
            throw std::invalid_argument(
                "Boundary dimension does not match mesh dimension.");
        }
    }

    /** @brief Validate and normalize the constructor dimension. */
    [[nodiscard]] static Index checked_dimension(Index dimension) {
        if constexpr (Dim == Dynamic) {
            if (dimension == Index{0}) {
                throw std::invalid_argument(
                    "A dynamic mesh dimension must be positive.");
            }
            return dimension;
        } else {
            static_assert(
                static_cast<unsigned long long>(Dim) <=
                    static_cast<unsigned long long>(
                        std::numeric_limits<Index>::max()),
                "Fixed dimension does not fit into Index.");

            const Index fixed = static_cast<Index>(Dim);
            if (dimension != fixed) {
                throw std::invalid_argument(
                    "Runtime dimension does not match fixed mesh dimension.");
            }
            return fixed;
        }
    }

    /** @brief Add two indices with overflow checking. */
    [[nodiscard]] static Index checked_add(Index left, Index right) {
        if (left > std::numeric_limits<Index>::max() - right) {
            throw std::overflow_error(
                "Mesh index exceeds the range of Index.");
        }
        return static_cast<Index>(left + right);
    }

    Index dimension_;
    IndexMapping index_mapping_;
    Sigma erase_sigma_buffer_;
    VertexPoint erase_position_buffer_;
};

} // namespace highvoronoi
