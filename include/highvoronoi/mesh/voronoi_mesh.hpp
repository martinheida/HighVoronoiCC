
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

#include <highvoronoi/mesh/abstract_mesh.hpp>
#include <highvoronoi/algorithm/neighbour_finder.hpp>
#include <highvoronoi/storage/read_write_list.hpp>
#include <highvoronoi/storage/neighbour/neighbour_storage.hpp>
#include <highvoronoi/storage/neighbour/neighbour_dirty_tracker.hpp>

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
    : public AbstractVoronoiNodes<
          typename InternalNodes::Scalar,
          InternalNodes::DimensionAtCompileTime,
          typename InternalNodes::Index> {
public:
    using Scalar = typename InternalNodes::Scalar;
    using Index = typename InternalNodes::Index;
    static constexpr int DimensionAtCompileTime =
        InternalNodes::DimensionAtCompileTime;

    using Base = AbstractVoronoiNodes<
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
    void copy_node_impl(Index public_index, Scalar* target) const override {
        const Index internal_index = public_to_internal_->at(
            static_cast<std::size_t>(public_index));
        internal_nodes_->copy_node(internal_index, target);
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

template <class MeshT, class AffectedVectorT>
class IncrementalVoronoiComputeMesh;

template <class MeshT, class AffectedVectorT>
struct IncrementalVoronoiBackend;

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
 * The mesh exposes the common node interface through a lightweight public view
 * onto stable contiguous internal node storage.
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
          DatabaseT,
          detail::MeshAddressList<DatabaseT>,
          DenseIndexMapping<typename DatabaseT::Index>> {
public:
    using Database = DatabaseT;
    using NodeScalar = NodeScalarT;
    using VertexScalar = typename Database::Scalar;
    using Index = typename Database::Index;

    using IndexMapping = DenseIndexMapping<Index>;

    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        Dim,
        Database,
        detail::MeshAddressList<Database>,
        IndexMapping>;

    using Address = typename Base::Address;
    using AddressList = typename Base::AddressList;
    using BoundaryType = typename Base::BoundaryType;

    static_assert(
        std::is_same_v<typename AddressList::value_type, Address>,
        "VoronoiMesh address lists must store database addresses, not node indices.");
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;
    using NodePoint = typename Base::NodePoint;
    using VertexPoint = typename Base::VertexPoint;
    using Sigma = typename Base::Sigma;
    using NeighbourStorage = detail::NeighbourStorage<
        typename Database::LockType,
        Index>;
    using NeighbourDatabase = typename NeighbourStorage::Database;
    using NeighbourAddress = typename NeighbourStorage::Address;
    using NeighbourDirtyTracker = detail::NeighbourDirtyTracker;
    using NeighbourDirtyTrackerHandle = std::shared_ptr<NeighbourDirtyTracker>;

    using InternalNodes = VoronoiNodes<NodeScalar, Dim, Index>;
    using PublicNodes = detail::PublicStoredNodesView<InternalNodes>;
    using ExtendedNodes = ExtendedVoronoiNodes<PublicNodes>;
    using NeighbourWorkspace = detail::NeighbourFinder<
        InternalNodes,
        BoundaryType,
        VertexScalar>;

    static constexpr int DimensionAtCompileTime = Dim;

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
        : Base(
              nodes.dimension(),
              IndexMapping::identity(nodes.size())),
          internal_nodes_(std::move(nodes)),
          primary_address_lists_(
              static_cast<std::size_t>(internal_nodes_.size())),
          secondary_address_lists_(
              static_cast<std::size_t>(internal_nodes_.size())),
          neighbour_storage_(
              static_cast<std::size_t>(internal_nodes_.size())),
          extended_nodes_(
              PublicNodes(
                  internal_nodes_,
                  this->index_mapping().public_to_internal_data()),
              std::move(boundary)),
          neighbour_workspace_(
              internal_nodes_,
              extended_nodes_.boundary()),
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

    using Base::compute_neighbors;

    // ---------------------------------------------------------------------
    // Persistent neighbour owner API
    // ---------------------------------------------------------------------

    [[nodiscard]] NeighbourDatabase& neighbour_database() noexcept {
        return neighbour_storage_.database();
    }

    [[nodiscard]] const NeighbourDatabase& neighbour_database() const noexcept {
        return neighbour_storage_.database();
    }

    [[nodiscard]] NeighbourAddress neighbour_address(Index public_cell) const {
        if (public_cell >= this->size()) {
            throw std::out_of_range(
                "VoronoiMesh public neighbour cell index out of range.");
        }
        return internal_neighbour_address(
            this->index_mapping().public_to_internal(public_cell));
    }

    [[nodiscard]] NeighbourAddress
    internal_neighbour_address(Index internal_cell) const {
        if (internal_cell >= this->internal_size()) {
            throw std::out_of_range(
                "VoronoiMesh internal neighbour cell index out of range.");
        }
        return neighbour_storage_.address(internal_cell);
    }

    void snapshot_internal_neighbour_addresses(
        std::vector<NeighbourAddress>& addresses) const {
        const std::size_t count =
            static_cast<std::size_t>(this->internal_size());
        addresses.resize(count);
        for (std::size_t cell = 0; cell < count; ++cell) {
            addresses[cell] = neighbour_storage_.address(
                static_cast<Index>(cell));
        }
    }

    [[nodiscard]] bool read_internal_neighbours_at(
        NeighbourAddress address,
        std::vector<Index>& buffer) const {
        buffer.clear();
        return neighbour_storage_.read_address(address, buffer);
    }

    [[nodiscard]] NeighbourDirtyTrackerHandle request_neighbour_dirty_tracker(
        bool include_current_dirty = false) {
        return neighbour_tracking_.request_tracker(
            neighbour_storage_.dirty_vector(),
            include_current_dirty);
    }

    void propagate_neighbour_dirty() {
        neighbour_tracking_.propagate(neighbour_storage_.dirty_vector());
    }

    [[nodiscard]] std::size_t neighbour_version() const noexcept {
        return neighbour_tracking_.version();
    }

    [[nodiscard]] std::size_t advance_neighbour_version() noexcept {
        return neighbour_tracking_.advance_version();
    }


    /** Create caller-owned scratch reusable across neighbour computations. */
    [[nodiscard]] NeighbourWorkspace make_neighbour_workspace() const {
        return NeighbourWorkspace(
            internal_nodes_,
            extended_nodes_.boundary());
    }

    /**
     * Allocation-reusing public-cell neighbour computation with caller-owned
     * scratch.  The ordinary one-argument overload inherited from AbstractMesh
     * uses this mesh's persistent workspace instead.
     */
    void compute_neighbors(
        Index public_cell,
        NeighbourWorkspace& workspace) {
        if (public_cell >= this->size()) {
            throw std::out_of_range(
                "VoronoiMesh public neighbour cell index out of range.");
        }
        const Index internal_cell =
            this->index_mapping().public_to_internal(public_cell);
        this->compute_internal_neighbours_from_geometry(
            internal_cell,
            extended_nodes_.boundary(),
            workspace);
        this->set_dirty(public_cell, false);
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

    // ---------------------------------------------------------------------
    // Stable-node mutation API used by non-periodic incremental algorithms
    // ---------------------------------------------------------------------

    /** @brief Map one current public node to its stable internal slot. */
    [[nodiscard]] Index public_node_to_internal(Index public_node) const {
        if (public_node >= this->size()) {
            throw std::out_of_range(
                "VoronoiMesh public node index out of range.");
        }
        return this->index_mapping().public_to_internal(public_node);
    }

    /** @brief Map one stable internal slot to current public numbering. */
    [[nodiscard]] std::optional<Index>
    internal_node_to_public(Index internal_node) const {
        if (internal_node >= internal_nodes_.size()) {
            throw std::out_of_range(
                "VoronoiMesh internal node index out of range.");
        }
        return this->index_mapping().internal_to_public(internal_node);
    }

    /**
     * @brief Append one ordinary node as a new stable internal slot.
     *
     * Existing stable indices and stored signatures are unchanged. Structural
     * growth requires external synchronization and must not overlap a running
     * ComputeVoronoi phase.
     */
    [[nodiscard]] Index append_node(const NodePoint& point) {
        std::vector<NodePoint> points;
        points.reserve(1);
        points.push_back(point);
        return append_nodes(points).front();
    }

    /**
     * @brief Append ordinary nodes while preserving all existing stable slots.
     *
     * Deleted internal slots remain tombstones; they are never recycled. New
     * stable slots are appended after the complete existing internal range and
     * become new public nodes at the end of the current dense public numbering.
     * Existing vertex records are deliberately left untouched.
     *
     * @return Stable internal indices of the appended nodes.
     */
    [[nodiscard]] std::vector<Index> append_nodes(
        const std::vector<NodePoint>& points) {
        if (points.empty()) {
            return {};
        }
        if (public_numbering_dirty_) {
            throw std::logic_error(
                "VoronoiMesh cannot append while public numbering is dirty.");
        }

        const std::size_t old_internal =
            static_cast<std::size_t>(internal_nodes_.size());
        const std::size_t old_public =
            static_cast<std::size_t>(this->size());
        const std::size_t extra = points.size();
        const std::size_t boundary_count =
            static_cast<std::size_t>(extended_nodes_.boundary().size());
        const std::size_t maximum =
            static_cast<std::size_t>((std::numeric_limits<Index>::max)());

        // max(Index) is invalid. Boundary mirrors occupy the values directly
        // below it, so ordinary stable indices must remain below that range.
        if (boundary_count > maximum ||
            old_internal > maximum - boundary_count ||
            extra > maximum - boundary_count - old_internal) {
            throw std::overflow_error(
                "VoronoiMesh append exceeds stable internal Index capacity.");
        }

        const std::size_t new_internal = old_internal + extra;

        std::vector<Index> public_to_internal =
            this->index_mapping().public_to_internal_data();
        std::vector<Index> internal_to_public =
            this->index_mapping().internal_to_public_data();
        public_to_internal.reserve(old_public + extra);
        internal_to_public.reserve(new_internal);

        internal_nodes_.resize(static_cast<Index>(new_internal));
        primary_address_lists_.resize(new_internal);
        secondary_address_lists_.resize(new_internal);
        neighbour_storage_.resize(new_internal);

        std::vector<Index> appended;
        appended.reserve(extra);

        for (std::size_t offset = 0; offset < extra; ++offset) {
            const Index internal = static_cast<Index>(old_internal + offset);
            const Index public_node = static_cast<Index>(old_public + offset);

            internal_nodes_.set(internal, points[offset]);
            public_to_internal.push_back(internal);
            internal_to_public.push_back(public_node);
            appended.push_back(internal);
        }

        this->index_mapping().assign(
            std::move(public_to_internal),
            std::move(internal_to_public));
        reset_extended_nodes();
        return appended;
    }


private:
    template <class, class>
    friend class IncrementalVoronoiComputeMesh;

    template <class, class>
    friend struct IncrementalVoronoiBackend;

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
        if (has_active_records()) {
            throw std::logic_error(
                "Cannot replace a VoronoiMesh boundary while active vertices "
                "or infinite edges are stored. Boundary mirror indices may be "
                "part of persistent signatures.");
        }

        extended_nodes_ = ExtendedNodes(
            PublicNodes(
                  internal_nodes_,
                  this->index_mapping().public_to_internal_data()),
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
        /**
     * @brief Map one stable internal node to public numbering.
     *
     * Deleted internal nodes return `std::nullopt`.
     */
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

    /** @brief Return the global list of persisted unbounded edges. */
    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return infinite_edge_addresses_;
    }

    /** @brief Register one newly persisted unbounded edge. */
    void register_infinite_edge_impl(Address address) override {
        infinite_edge_addresses_.push_back(address);
    }

    [[nodiscard]] bool load_internal_neighbours_impl(
        Index internal_cell,
        std::vector<Index>& buffer) const override {
        return neighbour_storage_.load(internal_cell, buffer);
    }

    void store_internal_neighbours_impl(
        Index internal_cell,
        const std::vector<Index>& neighbours) override {
        (void)neighbour_storage_.store(internal_cell, neighbours);
    }

    [[nodiscard]] bool internal_neighbours_dirty_impl(
        Index internal_cell) const override {
        return neighbour_storage_.dirty(internal_cell);
    }

    void set_internal_neighbours_dirty_impl(
        Index internal_cell,
        bool value) override {
        neighbour_storage_.set_dirty(internal_cell, value);
    }

    void compute_neighbors_impl(Index internal_cell) override {
        this->compute_internal_neighbours_from_geometry(
            internal_cell,
            extended_nodes_.boundary(),
            neighbour_workspace_);
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
        if (this->index_mapping().is_deleted(internal_node)) {
            return;
        }

        this->index_mapping().mark_deleted(internal_node);
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
        compact_infinite_edge_address_list();
    }

    // ---------------------------------------------------------------------
    // Numbering and cleanup helpers
    // ---------------------------------------------------------------------

    /** @brief Validate one stable internal ordinary-node index. */
    void require_internal_node(Index internal_node) const {
        if (internal_node >= internal_nodes_.size()) {
            throw std::out_of_range(
                "VoronoiMesh internal node index out of range.");
        }
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
        this->index_mapping().rebuild();
    }

    /**
     * @brief Reset active mirrors while preserving nodes and boundary.
     */
    void reset_extended_nodes() {
        BoundaryType boundary_copy = extended_nodes_.boundary();
        extended_nodes_ = ExtendedNodes(
            PublicNodes(
                  internal_nodes_,
                  this->index_mapping().public_to_internal_data()),
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
    [[nodiscard]] bool has_active_records() const {
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

        VertexPoint direction = make_vertex_point_for_read();
        const std::size_t infinite_count = infinite_edge_addresses_.size();
        for (std::size_t i = 0; i < infinite_count; ++i) {
            sigma.clear();
            database_->read_facet(
                infinite_edge_addresses_[i],
                position,
                sigma,
                direction);
            if (!sigma.empty()) {
                return true;
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

    /** @brief Remove tombstoned records from the global infinite-edge list. */
    void compact_infinite_edge_address_list() {
        Sigma sigma;
        VertexPoint origin = make_vertex_point_for_read();
        VertexPoint direction = make_vertex_point_for_read();

        infinite_edge_addresses_.erase_if(
            [this, &sigma, &origin, &direction](Address address) {
                sigma.clear();
                database_->read_facet(
                    address,
                    origin,
                    sigma,
                    direction);
                return sigma.empty();
            });
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
    std::vector<AddressList> primary_address_lists_;
    std::vector<AddressList> secondary_address_lists_;
    NeighbourStorage neighbour_storage_;
    detail::NeighbourTrackingState<typename Database::LockType> neighbour_tracking_;
    AddressList infinite_edge_addresses_;
    ExtendedNodes extended_nodes_;
    NeighbourWorkspace neighbour_workspace_;
    std::shared_ptr<Database> database_;
    bool public_numbering_dirty_ = false;
};

} // namespace highvoronoi


