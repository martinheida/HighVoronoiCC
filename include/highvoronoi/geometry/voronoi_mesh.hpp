#pragma once

/**
 * @file voronoi_mesh.hpp
 * @brief Concrete stored-node Voronoi mesh backed by a shared vertex database.
 *
 * `VoronoiMesh` is the basic concrete mesh implementation used by
 * HighVoronoi. It owns a stable array of internal nodes and stores, for every
 * internal node, two address lists in a shared vertex database: a primary
 * list for vertices uniquely owned by that node and a secondary list for
 * vertices owned by a smaller generating node.
 *
 * Public node numbering is represented by a dense mapping onto the stable
 * internal node array. Deleting a node removes its entry from the public
 * mapping but never moves or renumbers the internal node storage. Vertex
 * signatures stored in the database therefore remain valid for the entire
 * lifetime of the mesh.
 *
 * The database is held through `std::shared_ptr`. A standalone mesh may own the
 * only reference, while a future composite mesh can retain the same database
 * handle. Conversion to a composite mesh's global internal numbering remains
 * the responsibility of that composite mesh.
 *
 * Ordinary public nodes are exposed through a lightweight stored-node view.
 * `ExtendedVoronoiNodes` owns this view, not a copy of the coordinates. Thus
 * the mesh stores ordinary node coordinates exactly once and allocates
 * additional storage only for active boundary reflections.
 */

#include <highvoronoi/geometry/abstract_mesh.hpp>
#include <highvoronoi/detail/read_write_list.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {
namespace detail {

/**
 * @brief Dense public view onto a stable stored-node container.
 *
 * The view does not own coordinates or the mapping. It reads public node
 * `i` from internal node `public_to_internal[i]`. Both referenced objects must
 * outlive the view.
 *
 * @tparam InternalNodes Stored node container owning the stable coordinates.
 */
template <class InternalNodes>
class PublicStoredNodesView final
    : public StoredNodeAccess<
          typename InternalNodes::Scalar,
          InternalNodes::DimensionAtCompileTime,
          typename InternalNodes::Index> {
public:
    using Scalar = typename InternalNodes::Scalar;
    using Index = typename InternalNodes::Index;
    static constexpr int DimensionAtCompileTime =
        InternalNodes::DimensionAtCompileTime;
    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;

    using Base = StoredNodeAccess<
        Scalar,
        DimensionAtCompileTime,
        Index>;
    using Mapping = std::vector<Index>;

    /**
     * @brief Construct a non-owning public-node view.
     *
     * @param internal_nodes Stable internal node storage.
     * @param public_to_internal Dense public-to-internal mapping.
     */
    PublicStoredNodesView(
        const InternalNodes& internal_nodes,
        const Mapping& public_to_internal)
        : Base(internal_nodes.dimension()),
          internal_nodes_(&internal_nodes),
          public_to_internal_(&public_to_internal) {}

    /** @brief Return the current number of public nodes. */
    [[nodiscard]] Index size() const noexcept override {
        return static_cast<Index>(public_to_internal_->size());
    }

    [[nodiscard]] Scalar get_data(
        Index public_index,
        Index coordinate) const override {
        const Index internal_index = public_to_internal_->at(
            static_cast<std::size_t>(public_index));
        return internal_nodes_->get_data(internal_index, coordinate);
    }

private:
    /** @brief Return stable storage of one publicly addressed node. */
    [[nodiscard]] const Scalar*
    get_stored_node_pointer(Index public_index) const override {
        const Index internal_index = public_to_internal_->at(
            static_cast<std::size_t>(public_index));
        return internal_nodes_->stable_node_data(internal_index);
    }

    const InternalNodes* internal_nodes_;
    const Mapping* public_to_internal_;
};

/**
 * @brief Thread-policy-aware list of database addresses used by a mesh node.
 *
 * Database addresses are storage offsets and therefore always use
 * `std::size_t`; they are deliberately independent of the usually smaller
 * node `Index` type.
 */
template <class Database>
using MeshAddressList = ReadWriteAddressList<
    std::vector<std::size_t>,
    typename Database::LockType>;

} // namespace detail

/**
 * @brief Basic concrete Voronoi mesh with stable stored nodes.
 *
 * @tparam NodeScalarT Scalar used for node coordinates.
 * @tparam Dim Compile-time dimension or `highvoronoi::Dynamic`.
 * @tparam DatabaseT Database implementing the contract required by
 *         `AbstractMesh`. In normal HighVoronoi use this is an
 *         `HVDataBase<Lock, DataBaseParams>` specialization.
 *
 * The database scalar and index types determine `VertexScalar` and `Index`.
 * The mesh always exposes stored node access because its public-node view maps
 * directly onto stable contiguous internal node storage.
 */
template <typename NodeScalarT,
          int Dim,
          class DatabaseT>
class VoronoiMesh final
    : public AbstractMesh<
          NodeScalarT,
          typename DatabaseT::Scalar,
          typename DatabaseT::Index,
          Dim,
          NodeAccessMode::Stored,
          DatabaseT,
          detail::MeshAddressList<DatabaseT>> {
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
        NodeAccessMode::Stored,
        Database,
        detail::MeshAddressList<Database>>;

    using Address = typename Base::Address;
    using AddressList = typename Base::AddressList;
    using BoundaryType = typename Base::BoundaryType;

    static_assert(
        std::is_same_v<typename AddressList::value_type, Address>,
        "VoronoiMesh address lists must store database addresses, not node indices.");
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;
    using VertexPoint = typename Base::VertexPoint;
    using Sigma = typename Base::Sigma;

    using InternalNodes = VoronoiNodes<NodeScalar, Dim, Index>;
    using PublicNodes = detail::PublicStoredNodesView<InternalNodes>;
    using ExtendedNodes = ExtendedVoronoiNodes<PublicNodes>;

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;

    /**
     * @brief Construct a mesh using an existing shared database.
     *
     * The initial public numbering is identical to the internal numbering.
     * The database pointer must not be null. Supplying a shared database is the
     * intended construction path when an external owner manages the database.
     *
     * @param nodes Stable internal node coordinates.
     * @param boundary Geometric boundary used for reflected extended nodes.
     * @param database Shared vertex database.
     */
    VoronoiMesh(
        InternalNodes nodes,
        BoundaryType boundary,
        std::shared_ptr<Database> database)
        : Base(nodes.dimension()),
          internal_nodes_(std::move(nodes)),
          public_to_internal_(make_identity_mapping(internal_nodes_.size())),
          internal_to_public_(public_to_internal_),
          primary_address_lists_(
              static_cast<std::size_t>(internal_nodes_.size())),
          secondary_address_lists_(
              static_cast<std::size_t>(internal_nodes_.size())),
          extended_nodes_(
              PublicNodes(internal_nodes_, public_to_internal_),
              std::move(boundary)),
          database_(require_database(std::move(database))) {}

    /**
     * @brief Construct a mesh with an empty boundary.
     *
     * For runtime dimension, the empty boundary has dimension zero and adopts
     * no geometric constraints. It can later be replaced through
     * `set_boundary()`.
     */
    VoronoiMesh(
        InternalNodes nodes,
        std::shared_ptr<Database> database)
        : VoronoiMesh(
              std::move(nodes),
              BoundaryType{},
              std::move(database)) {}

    /** @brief Return the stable internal node storage. */
    [[nodiscard]] const InternalNodes&
    internal_nodes() const noexcept {
        return internal_nodes_;
    }

    /**
     * @brief Return the concrete extended-node container.
     *
     * This typed accessor supplements the access-mode-independent
     * `AbstractMesh::extended_nodes()` interface and exposes operations such as
     * `active_planes()`, `mirror_is_active()`, and `mirror_index()`.
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

    /**
     * @brief Return the shared database handle.
     *
     * The handle is primarily intended for a later composite owner. Sharing
     * the database does not by itself define the composite mesh's global
     * signature numbering.
     */
    [[nodiscard]] const std::shared_ptr<Database>&
    database_handle() const noexcept {
        return database_;
    }

private:
    // ---------------------------------------------------------------------
    // AbstractMesh primitive hooks
    // ---------------------------------------------------------------------

    /** @brief Return the dense public-node view. */
    [[nodiscard]] const NodesAccess&
    nodes_impl() const noexcept override {
        return extended_nodes_.inner_nodes();
    }

    /** @brief Return mutable extended nodes for cell activation. */
    [[nodiscard]] ExtendedNodesAccess&
    extended_nodes_impl() noexcept override {
        return extended_nodes_;
    }

    /** @brief Return extended nodes as a const access interface. */
    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return extended_nodes_;
    }

    /** @brief Return the boundary owned by the extended-node container. */
    [[nodiscard]] const BoundaryType&
    boundary_impl() const noexcept override {
        return extended_nodes_.boundary();
    }

    /**
     * @brief Replace the boundary and reset all active mirror state.
     *
     * Ordinary nodes are not copied. The reconstructed extended container owns
     * another lightweight mapping view onto the same internal node storage.
     */
    void set_boundary_impl(BoundaryType boundary) override {
        if (has_active_vertices()) {
            throw std::logic_error(
                "Cannot replace a VoronoiMesh boundary while active vertices "
                "are stored. Boundary mirror indices are part of vertex "
                "signatures.");
        }

        extended_nodes_ = ExtendedNodes(
            PublicNodes(internal_nodes_, public_to_internal_),
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

    /** @brief Return the stable number of internal node slots. */
    [[nodiscard]] Index
    internal_node_count_impl() const noexcept override {
        return internal_nodes_.size();
    }

    /** @brief Map one dense public node index to stable internal numbering. */
    [[nodiscard]] Index
    public_node_to_internal_impl(Index public_node) const override {
        return public_to_internal_.at(
            static_cast<std::size_t>(public_node));
    }

    /**
     * @brief Map one stable internal node to public numbering.
     *
     * Deleted internal nodes return `std::nullopt`.
     */
    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        const Index public_index = internal_to_public_.at(
            static_cast<std::size_t>(internal_node));
        if (public_index == deleted_node_marker()) {
            return std::nullopt;
        }
        return public_index;
    }

    /** @brief Return addresses primarily owned by one internal node. */
    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return primary_address_lists_.at(
            static_cast<std::size_t>(internal_node));
    }

    /** @brief Return addresses secondarily registered at one internal node. */
    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return secondary_address_lists_.at(
            static_cast<std::size_t>(internal_node));
    }

    /** @brief Register a database address at its unique primary owner. */
    void register_primary_vertex_impl(
        Index internal_node,
        Address address) override {
        primary_address_lists_.at(
            static_cast<std::size_t>(internal_node)).push_back(address);
    }

    /** @brief Register a database address at one secondary generator. */
    void register_secondary_vertex_impl(
        Index internal_node,
        Address address) override {
        secondary_address_lists_.at(
            static_cast<std::size_t>(internal_node)).push_back(address);
    }

    /**
     * @brief Mark an internal node as deleted without changing internal storage.
     *
     * Only the stable internal-to-public entry is tombstoned here.
     * `public_to_internal_` remains unchanged until the base class has marked
     * every selected node and invokes cleanup. Thus all deletions in one
     * operation use the same pre-deletion public numbering and avoid repeated
     * O(n) reconstruction.
     */
    void mark_internal_node_deleted_impl(
        Index internal_node) override {
        Index& public_index = internal_to_public_.at(
            static_cast<std::size_t>(internal_node));
        if (public_index == deleted_node_marker()) {
            return;
        }

        public_index = deleted_node_marker();
        public_numbering_dirty_ = true;
    }

    /**
     * @brief Finalize deferred numbering changes and remove tombstoned addresses.
     *
     * Address compaction is deliberately performed after explicit filter
     * operations rather than during ordinary iteration. Each active database
     * address is tested only once even though it occurs once in a primary list
     * and may also occur in several secondary lists.
     */
    void cleanup_vertex_lists_impl() override {
        if (public_numbering_dirty_) {
            rebuild_public_numbering();
            reset_extended_nodes();
            public_numbering_dirty_ = false;
        }
        compact_vertex_address_lists();
    }

    // ---------------------------------------------------------------------
    // Numbering and cleanup helpers
    // ---------------------------------------------------------------------

    /** @brief Build the initial identity mapping. */
    [[nodiscard]] static std::vector<Index>
    make_identity_mapping(Index count) {
        std::vector<Index> result(
            static_cast<std::size_t>(count));
        for (Index index = Index{0}; index < count; ++index) {
            result[static_cast<std::size_t>(index)] = index;
        }
        return result;
    }

    /** @brief Sentinel used only in the internal-to-public map. */
    [[nodiscard]] static constexpr Index
    deleted_node_marker() noexcept {
        return std::numeric_limits<Index>::max();
    }

    /** @brief Validate and return a non-null shared database. */
    [[nodiscard]] static std::shared_ptr<Database>
    require_database(std::shared_ptr<Database> database) {
        if (!database) {
            throw std::invalid_argument(
                "VoronoiMesh requires a non-null database.");
        }
        return database;
    }

    /**
     * @brief Rebuild both numbering maps after one or more node deletions.
     */
    void rebuild_public_numbering() {
        public_to_internal_.clear();
        public_to_internal_.reserve(
            static_cast<std::size_t>(internal_nodes_.size()));

        for (Index internal_index = Index{0};
             internal_index < internal_nodes_.size();
             ++internal_index) {
            if (internal_to_public_[
                    static_cast<std::size_t>(internal_index)] ==
                deleted_node_marker()) {
                continue;
            }
            public_to_internal_.push_back(internal_index);
        }

        std::fill(
            internal_to_public_.begin(),
            internal_to_public_.end(),
            deleted_node_marker());

        for (std::size_t public_position = 0;
             public_position < public_to_internal_.size();
             ++public_position) {
            const Index internal_index =
                public_to_internal_[public_position];
            internal_to_public_[
                static_cast<std::size_t>(internal_index)] =
                static_cast<Index>(public_position);
        }
    }

    /**
     * @brief Reset active mirrors while preserving nodes and boundary.
     */
    void reset_extended_nodes() {
        BoundaryType boundary_copy = extended_nodes_.boundary();
        extended_nodes_ = ExtendedNodes(
            PublicNodes(internal_nodes_, public_to_internal_),
            std::move(boundary_copy));
    }

    /** @brief Create an owning vertex point with the mesh dimension. */
    [[nodiscard]] VertexPoint make_vertex_point_for_read() const {
        if constexpr (Dim == Dynamic) {
            return VertexPoint(
                static_cast<Eigen::Index>(this->dimension()));
        } else {
            return VertexPoint{};
        }
    }


    /**
     * @brief Return whether at least one registered database record is active.
     *
     * Boundary-plane indices are embedded in internal signatures. Replacing a
     * boundary while records exist could silently reinterpret those indices,
     * so boundary replacement is restricted to an empty mesh state.
     */
    [[nodiscard]] bool has_active_vertices() const {
        std::unordered_set<Address> visited;
        Sigma sigma;
        VertexPoint position = make_vertex_point_for_read();

        for (const AddressList& addresses : primary_address_lists_) {
            const std::size_t count = addresses.size();
            for (std::size_t position_index = 0;
                 position_index < count;
                 ++position_index) {
                const Address address = addresses[position_index];
                if (!visited.insert(address).second) {
                    continue;
                }

                sigma.clear();
                database_->read(address, position, sigma);
                if (!sigma.empty()) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief Remove deleted database addresses from every per-node list.
     */
    void compact_vertex_address_lists() {
        std::unordered_set<Address> all_addresses;
        collect_addresses(primary_address_lists_, all_addresses);
        collect_addresses(secondary_address_lists_, all_addresses);

        std::unordered_set<Address> active_addresses;
        active_addresses.reserve(all_addresses.size());

        Sigma sigma;
        VertexPoint position = make_vertex_point_for_read();
        for (const Address address : all_addresses) {
            sigma.clear();
            database_->read(address, position, sigma);
            if (!sigma.empty()) {
                active_addresses.insert(address);
            }
        }

        compact_address_lists(primary_address_lists_, active_addresses);
        compact_address_lists(secondary_address_lists_, active_addresses);
    }

    /** @brief Insert every address from a list family into a set. */
    static void collect_addresses(
        const std::vector<AddressList>& lists,
        std::unordered_set<Address>& target) {
        for (const AddressList& addresses : lists) {
            const std::size_t count = addresses.size();
            for (std::size_t position = 0; position < count; ++position) {
                target.insert(addresses[position]);
            }
        }
    }

    /** @brief Remove tombstoned addresses from every list in one family. */
    static void compact_address_lists(
        std::vector<AddressList>& lists,
        const std::unordered_set<Address>& active_addresses) {
        for (AddressList& addresses : lists) {
            addresses.erase_if(
                [&active_addresses](Address address) {
                    return active_addresses.find(address) ==
                           active_addresses.end();
                });
        }
    }

    InternalNodes internal_nodes_;
    std::vector<Index> public_to_internal_;
    std::vector<Index> internal_to_public_;
    std::vector<AddressList> primary_address_lists_;
    std::vector<AddressList> secondary_address_lists_;
    ExtendedNodes extended_nodes_;
    std::shared_ptr<Database> database_;
    bool public_numbering_dirty_ = false;
};

} // namespace highvoronoi
