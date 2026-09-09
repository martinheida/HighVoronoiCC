
#pragma once

/**
 * @file high_voronoi_compute_mesh.hpp
 * @brief Internal compute representation and database event wrapper for HighVoronoi.
 *
 * The ordinary ComputeVoronoi kernel remains completely unaware of refinement,
 * deletion, visibility, references, or periodization. This header supplies the
 * representation that makes that possible:
 *
 * - `HighVoronoiComputeRoundState` contains the thread-safe flags collected in
 *   one outer HighVoronoi reconstruction round.
 * - `HighVoronoiComputeDataBase` wraps the persistent HighVoronoi database. It
 *   receives canonical stable-internal signatures from AbstractMesh::store_vertex
 *   and extracts refinement/periodic events exactly when a new vertex is found.
 * - `HighVoronoiComputeMesh` exposes all active stable internal nodes through a
 *   dense bijective public mapping, optionally moving a requested cell set to
 *   the front for ComputeVoronoi's range-limited execution.
 *
 * No object in this file owns persistent geometry or database records.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/mesh/high_voronoi_mesh.hpp>
#include <highvoronoi/mesh/mapped_voronoi_nodes.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/** Distinguish the two useful collection phases of one refinement round. */
enum class HighVoronoiComputePhase : std::uint8_t {
    NewCells,
    RepairCells
};

/**
 * @brief Thread-policy-aware transient state shared by ComputeVoronoi workers.
 *
 * `round_old_count` separates the already-integrated prefix from the nodes that
 * are new in this outer round. The affected and mirror-request vectors never
 * resize while workers run.
 */
template <class HighMeshT, class AffectedVectorT>
class HighVoronoiComputeRoundState final {
public:
    using HighMesh = HighMeshT;
    using AffectedVector = AffectedVectorT;
    using Index = typename HighMesh::Index;
    using VertexPoint = typename HighMesh::VertexPoint;
    using Sigma = typename HighMesh::Sigma;
    using PeriodicBoundaryVertex = std::pair<VertexPoint, Sigma>;
    using EventLock = std::conditional_t<
        std::is_same_v<AffectedVector, detail::AtomicBitVector>,
        detail::BusyFIFOLock,
        detail::EmptyLock>;

    HighVoronoiComputeRoundState(
        Index internal_count,
        Index boundary_plane_count,
        Index round_old_count)
        : internal_count_(internal_count),
          plane_count_(boundary_plane_count),
          round_old_count_(round_old_count),
          owned_affected_(
              std::in_place,
              static_cast<std::size_t>(internal_count)),
          affected_(&*owned_affected_),
          mirror_requests_(checked_request_count(internal_count, boundary_plane_count)) {
        if (round_old_count_ > internal_count_) {
            throw std::invalid_argument(
                "HighVoronoi round-old watermark exceeds the internal node count.");
        }
    }

    HighVoronoiComputeRoundState(
        Index internal_count,
        Index boundary_plane_count,
        Index round_old_count,
        AffectedVector& affected)
        : internal_count_(internal_count),
          plane_count_(boundary_plane_count),
          round_old_count_(round_old_count),
          affected_(&affected),
          mirror_requests_(checked_request_count(internal_count, boundary_plane_count)) {
        if (round_old_count_ > internal_count_) {
            throw std::invalid_argument(
                "HighVoronoi round-old watermark exceeds the internal node count.");
        }
        if (affected_->size() < static_cast<std::size_t>(internal_count_)) {
            throw std::invalid_argument(
                "HighVoronoi AFFECTED vector is smaller than the mesh.");
        }
    }

    HighVoronoiComputeRoundState(const HighVoronoiComputeRoundState&) = delete;
    HighVoronoiComputeRoundState& operator=(const HighVoronoiComputeRoundState&) = delete;
    HighVoronoiComputeRoundState(HighVoronoiComputeRoundState&&) = delete;
    HighVoronoiComputeRoundState& operator=(HighVoronoiComputeRoundState&&) = delete;

    [[nodiscard]] Index internal_count() const noexcept { return internal_count_; }
    [[nodiscard]] Index plane_count() const noexcept { return plane_count_; }
    [[nodiscard]] Index round_old_count() const noexcept { return round_old_count_; }

    [[nodiscard]] HighVoronoiComputePhase phase() const noexcept { return phase_; }
    void set_phase(HighVoronoiComputePhase phase) noexcept { phase_ = phase; }

    /** Enable capture of transient periodic-boundary vertices for bootstrap repair. */
    void set_capture_periodic_boundary_vertices(bool enabled) noexcept {
        capture_periodic_boundary_vertices_ = enabled;
    }

    /**
     * Capture one periodic-boundary vertex before HighVoronoi suppresses its
     * persistent database insertion. Signatures are already canonical stable
     * internal signatures at this point. The initial periodization front is
     * small, so linear signature deduplication keeps this path simple and is
     * negligible compared with the geometry work.
     */
    void record_periodic_boundary_vertex(
        const VertexPoint& position,
        const Sigma& sigma) {
        if (!capture_periodic_boundary_vertices_) {
            return;
        }

        VertexPoint stored_position = position;
        Sigma stored_sigma = sigma;

        std::lock_guard<EventLock> guard(periodic_boundary_vertices_lock_);
        for (const auto& item : periodic_boundary_vertices_) {
            if (item.second == stored_sigma) {
                return;
            }
        }

        periodic_boundary_vertices_.emplace_back(
            std::move(stored_position),
            std::move(stored_sigma));
    }

    [[nodiscard]] std::vector<PeriodicBoundaryVertex>
    periodic_boundary_vertices() const {
        std::lock_guard<EventLock> guard(periodic_boundary_vertices_lock_);
        return periodic_boundary_vertices_;
    }

    [[nodiscard]] bool is_new_node(Index internal) const noexcept {
        return internal >= round_old_count_ && internal < internal_count_;
    }

    [[nodiscard]] bool is_old_node(Index internal) const noexcept {
        return internal < round_old_count_;
    }

    void mark_affected(Index internal) noexcept {
        if (internal < round_old_count_) {
            affected_->set(static_cast<std::size_t>(internal));
        }
    }

    [[nodiscard]] bool is_affected(Index internal) const noexcept {
        return internal < internal_count_ &&
               affected_->test(static_cast<std::size_t>(internal));
    }

    void request_mirror(Index internal, Index plane) noexcept {
        if (internal >= internal_count_ || plane >= plane_count_) {
            return;
        }
        mirror_requests_.set(flat_index(internal, plane));
    }

    [[nodiscard]] bool mirror_requested(Index internal, Index plane) const noexcept {
        if (internal >= internal_count_ || plane >= plane_count_) {
            return false;
        }
        return mirror_requests_.test(flat_index(internal, plane));
    }

    [[nodiscard]] std::vector<Index> affected_nodes() const {
        std::vector<Index> result;
        for (Index internal = Index{0}; internal < round_old_count_; ++internal) {
            if (is_affected(internal)) {
                result.push_back(internal);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<std::pair<Index, Index>> requested_mirrors() const {
        std::vector<std::pair<Index, Index>> result;
        for (Index internal = Index{0}; internal < internal_count_; ++internal) {
            for (Index plane = Index{0}; plane < plane_count_; ++plane) {
                if (mirror_requested(internal, plane)) {
                    result.emplace_back(internal, plane);
                }
            }
        }
        return result;
    }

private:
    [[nodiscard]] static std::size_t checked_request_count(
        Index internal_count,
        Index plane_count) {
        const std::size_t nodes = static_cast<std::size_t>(internal_count);
        const std::size_t planes = static_cast<std::size_t>(plane_count);
        if (planes != 0 && nodes > (std::numeric_limits<std::size_t>::max)() / planes) {
            throw std::overflow_error("HighVoronoi mirror-request bit count overflow.");
        }
        return nodes * planes;
    }

    [[nodiscard]] std::size_t flat_index(Index internal, Index plane) const noexcept {
        return static_cast<std::size_t>(internal) *
                   static_cast<std::size_t>(plane_count_) +
               static_cast<std::size_t>(plane);
    }

    Index internal_count_ = Index{0};
    Index plane_count_ = Index{0};
    Index round_old_count_ = Index{0};
    HighVoronoiComputePhase phase_ = HighVoronoiComputePhase::NewCells;
    std::optional<AffectedVector> owned_affected_;
    AffectedVector* affected_ = nullptr;
    AffectedVector mirror_requests_;

    bool capture_periodic_boundary_vertices_ = false;
    mutable EventLock periodic_boundary_vertices_lock_;
    std::vector<PeriodicBoundaryVertex> periodic_boundary_vertices_;
};

/**
 * @brief Database-compatible wrapper extracting HighVoronoi events from sigma.
 *
 * AbstractMesh converts ComputeVoronoi's dense compute-public signature to the
 * stable global internal signature before calling `push()`. Consequently this
 * wrapper never has to translate a vertex a second time.
 *
 * A vertex containing a periodic internal-boundary generator is intentionally
 * not persisted: its geometry is still available to SystematicVoronoi's local
 * queue/edge exploration, while `push()` returns address zero so no persistent
 * primary/secondary registration is created.
 */
template <class HighMeshT, class AffectedVectorT>
class HighVoronoiComputeDataBase final {
public:
    using HighMesh = HighMeshT;
    using AffectedVector = AffectedVectorT;
    using UnderlyingDatabase = typename HighMesh::Database;
    using Scalar = typename UnderlyingDatabase::Scalar;
    using Index = typename UnderlyingDatabase::Index;
    using Sigma = typename UnderlyingDatabase::Sigma;
    using VertexPoint = typename UnderlyingDatabase::VertexPoint;
    using LockType = typename UnderlyingDatabase::LockType;
    using Address = std::size_t;

    static constexpr int DimensionAtCompileTime =
        UnderlyingDatabase::DimensionAtCompileTime;
    using RoundState = HighVoronoiComputeRoundState<HighMesh, AffectedVector>;

    HighVoronoiComputeDataBase(HighMesh& mesh, RoundState& state)
        : mesh_(mesh), database_(mesh.database()), state_(state) {}

    HighVoronoiComputeDataBase(const HighVoronoiComputeDataBase&) = delete;
    HighVoronoiComputeDataBase& operator=(const HighVoronoiComputeDataBase&) = delete;
    HighVoronoiComputeDataBase(HighVoronoiComputeDataBase&&) = delete;
    HighVoronoiComputeDataBase& operator=(HighVoronoiComputeDataBase&&) = delete;

    [[nodiscard]] UnderlyingDatabase& underlying_database() noexcept { return database_; }
    [[nodiscard]] const UnderlyingDatabase& underlying_database() const noexcept { return database_; }

    /**
     * Observe an already persisted vertex attached during append_mesh/engine.
     * No database mutation occurs; the same event extraction as push() is used.
     */
    void observe_existing_vertex(const Sigma& sigma) {
        (void)observe_sigma(sigma);
    }

    [[nodiscard]] Address push(
        const VertexPoint& position,
        const Sigma& sigma) {
        const bool periodic_boundary_vertex = observe_sigma(sigma);
        if (periodic_boundary_vertex) {
            state_.record_periodic_boundary_vertex(position, sigma);
            return Address{0};
        }
        return static_cast<Address>(database_.push(position, sigma));
    }

    void read(
        Address address,
        VertexPoint& position,
        Sigma& sigma) const {
        database_.read(address, position, sigma);
    }

    [[nodiscard]] bool contains(const Sigma& sigma) const {
        return database_.contains(sigma);
    }

    bool erase(Address address, const Sigma& sigma) {
        return database_.erase(address, sigma);
    }

    [[nodiscard]] bool register_signature(const Sigma& sigma) {
        return database_.register_signature(sigma);
    }

    [[nodiscard]] bool erase_signature(const Sigma& sigma) {
        return database_.erase_signature(sigma);
    }

    [[nodiscard]] Address push_facet(
        const VertexPoint& origin,
        Sigma& sigma,
        const VertexPoint& direction) {
        const bool periodic_boundary_edge = observe_sigma(sigma);
        if (periodic_boundary_edge) {
            return Address{0};
        }
        return static_cast<Address>(database_.push_facet(origin, sigma, direction));
    }

    void read_facet(
        Address address,
        VertexPoint& origin,
        Sigma& sigma,
        VertexPoint& direction) const {
        database_.read_facet(address, origin, sigma, direction);
    }

private:
    /**
     * Extract affected-node and periodic-copy information from one internal sigma.
     *
     * @return true iff sigma contains at least one periodic internal-boundary
     *         generator and therefore must not be persisted.
     */
    [[nodiscard]] bool observe_sigma(const Sigma& sigma) {
        const Index node_count = mesh_.internal_node_count();

        // Refinement: only the NEW-cell phase establishes the old affected set.
        if (state_.phase() == HighVoronoiComputePhase::NewCells) {
            for (const Index generator : sigma) {
                if (generator >= node_count) {
                    break; // canonical ordinary indices precede high-end boundary codes
                }
                if (mesh_.is_active_internal(generator) &&
                    state_.is_old_node(generator)) {
                    state_.mark_affected(generator);
                }
            }
        }

        bool periodic_boundary_vertex = false;

        // Bootstrap requests are extracted directly from periodic boundary
        // generators. Avoid temporary generator vectors here: push() is a hot
        // path and all information is already present in sigma.
        for (const Index generator : sigma) {
            if (generator < node_count) {
                continue;
            }
            if (!mesh_.is_boundary_internal_index(generator)) {
                throw std::logic_error(
                    "ComputeDataBase received an invalid HighVoronoi internal index.");
            }

            const Index plane = mesh_.decode_boundary_internal_index(generator);
            if (!mesh_.external_boundary()[plane].is_periodic()) {
                continue;
            }

            periodic_boundary_vertex = true;
            for (const Index visible : sigma) {
                if (visible >= node_count) {
                    break;
                }
                if (mesh_.is_active_internal(visible) &&
                    mesh_.is_visible_internal(visible)) {
                    state_.request_mirror(visible, plane);
                }
            }
        }

        // Periodic closure: if a visible generator shares this vertex with an
        // invisible copy shifted through plane p, request the visible generator
        // through partner(p). This is the vertex-local equivalent of Julia's
        // later visible/invisible neighbour scan.
        for (const Index invisible : sigma) {
            if (invisible >= node_count) {
                break;
            }
            if (!mesh_.is_active_internal(invisible) ||
                mesh_.is_visible_internal(invisible) ||
                !mesh_.reference_internal(invisible)) {
                continue;
            }

            const auto& shift =
                mesh_.reference_shift(invisible);
            for (Index plane = Index{0};
                 plane < mesh_.external_boundary().size();
                 ++plane) {
                if (shift[static_cast<std::size_t>(plane)] == 0) {
                    continue;
                }
                const auto partner =
                    mesh_.external_boundary()[plane].periodic_partner();
                if (!partner) {
                    throw std::logic_error(
                        "HighVoronoi reference shift selects a non-periodic plane.");
                }

                for (const Index visible : sigma) {
                    if (visible >= node_count) {
                        break;
                    }
                    if (mesh_.is_active_internal(visible) &&
                        mesh_.is_visible_internal(visible)) {
                        state_.request_mirror(visible, *partner);
                    }
                }
            }
        }

        return periodic_boundary_vertex;
    }

    HighMesh& mesh_;
    UnderlyingDatabase& database_;
    RoundState& state_;
};

/**
 * @brief Dense one-to-one compute representation of all active HighVoronoi nodes.
 *
 * `preferred_front` contains stable internal node indices. They are mapped to
 * compute-public indices `[0, preferred_front.size())`; all remaining active
 * nodes follow in stable insertion order. Stored signatures remain stable
 * HighVoronoi internal indices because this class uses DenseIndexMapping.
 */
template <class HighMeshT, class AffectedVectorT>
class HighVoronoiComputeMesh final
    : public AbstractMesh<
          typename HighMeshT::NodeScalar,
          typename HighMeshT::VertexScalar,
          typename HighMeshT::Index,
          HighMeshT::DimensionAtCompileTime,
          HighVoronoiComputeDataBase<HighMeshT, AffectedVectorT>,
          typename HighMeshT::AddressList,
          DenseIndexMapping<typename HighMeshT::Index>> {
public:
    using HighMesh = HighMeshT;
    using AffectedVector = AffectedVectorT;
    using NodeScalar = typename HighMesh::NodeScalar;
    using VertexScalar = typename HighMesh::VertexScalar;
    using Index = typename HighMesh::Index;
    using Address = typename HighMesh::Address;
    using AddressList = typename HighMesh::AddressList;
    using BoundaryType = typename HighMesh::BoundaryType;
    using RoundState = HighVoronoiComputeRoundState<HighMesh, AffectedVector>;
    using Database = HighVoronoiComputeDataBase<HighMesh, AffectedVector>;
    using IndexMapping = DenseIndexMapping<Index>;

    static constexpr int Dim = HighMesh::DimensionAtCompileTime;
    static constexpr int DimensionAtCompileTime = Dim;

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
    HighVoronoiComputeMesh(
        HighMesh& owner,
        RoundState& state,
        const std::vector<Index>& preferred_front = {})
        : Base(owner.dimension(), make_mapping(owner, preferred_front)),
          owner_(owner),
          database_(owner_, state),
          public_nodes_(owner_.internal_nodes(), this->index_mapping(), owner_.dimension()),
          extended_nodes_(std::in_place, public_nodes_, owner_.internal_boundary()) {}

    [[nodiscard]] HighMesh& owner() noexcept { return owner_; }
    [[nodiscard]] const HighMesh& owner() const noexcept { return owner_; }

    [[nodiscard]] Database& compute_database() noexcept { return database_; }
    [[nodiscard]] const Database& compute_database() const noexcept { return database_; }

    [[nodiscard]] Index stable_internal_node(Index compute_public) const {
        return this->index_mapping().public_to_internal(compute_public);
    }

    [[nodiscard]] std::optional<Index>
    compute_public_node(Index stable_internal) const {
        return this->index_mapping().internal_to_public(stable_internal);
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

private:
    [[nodiscard]] static IndexMapping make_mapping(
        const HighMesh& owner,
        const std::vector<Index>& preferred_front) {
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
            const Index public_index = static_cast<Index>(public_to_internal.size());
            public_to_internal.push_back(internal);
            internal_to_public[static_cast<std::size_t>(internal)] = public_index;
        };

        for (const Index internal : preferred_front) {
            append(internal);
        }
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

    [[nodiscard]] const ExtendedNodesAccess& extended_nodes_impl() const noexcept override {
        return *extended_nodes_;
    }

    [[nodiscard]] const BoundaryType& boundary_impl() const noexcept override {
        return extended_nodes_->boundary();
    }

    void set_boundary_impl(BoundaryType) override {
        throw std::logic_error(
            "Move HighVoronoi's internal boundary only between compute rounds.");
    }

    [[nodiscard]] Database& database_impl() noexcept override { return database_; }
    [[nodiscard]] const Database& database_impl() const noexcept override { return database_; }

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

    void register_primary_vertex_impl(Index internal_node, Address address) override {
        owner_.register_primary_internal(internal_node, address);
    }

    void register_secondary_vertex_impl(Index internal_node, Address address) override {
        owner_.register_secondary_internal(internal_node, address);
    }

    [[nodiscard]] const AddressList& infinite_edge_addresses_impl() const override {
        return owner_.infinite_addresses_internal();
    }

    void register_infinite_edge_impl(Address address) override {
        owner_.register_infinite_internal(address);
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
        throw std::logic_error(
            "HighVoronoiComputeMesh cannot structurally delete nodes during ComputeVoronoi.");
    }

    void cleanup_vertex_lists_impl() override {
        owner_.compact_vertex_address_lists();
    }

    HighMesh& owner_;
    Database database_;
    PublicActiveNodes public_nodes_;
    std::optional<ExtendedNodes> extended_nodes_;
};

} // namespace highvoronoi











