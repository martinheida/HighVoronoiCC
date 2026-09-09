#pragma once

/**
 * @file incremental_voronoi_compute_mesh.hpp
 * @brief Non-periodic compute facade and event tracking for incremental Voronoi updates.
 *
 * This file is the non-periodic subset of the compute-facade idea used by
 * HighVoronoi. The classical ComputeVoronoi kernel remains unchanged.
 *
 * The owning VoronoiMesh keeps stable internal numbering and persistent
 * storage. IncrementalVoronoiComputeMesh only creates a dense temporary public
 * numbering in which a requested stable-node set is moved to the prefix used by
 * ComputeVoronoi's range-limited execution.
 *
 * IncrementalVoronoiComputeDataBase wraps the persistent database. During the
 * NEW-cell phase it sees the canonical stable-internal signature produced by
 * AbstractMesh::store_vertex() and marks every OLD ordinary generator in that
 * signature as AFFECTED. During the repair phase it only delegates database
 * operations and does not enlarge AFFECTED.
 *
 * No visibility, reference, periodic-shift or mirror-request state exists here.
 */

#include <highvoronoi/core/detail/atomic_bit_vector.hpp>
#include <highvoronoi/algorithm/compute_voronoi.hpp>
#include <highvoronoi/mesh/mapped_voronoi_nodes.hpp>
#include <highvoronoi/algorithm/raycaster.hpp>
#include <highvoronoi/search/search_tree_factory_crtp.hpp>
#include <highvoronoi/mesh/voronoi_mesh.hpp>
#include <highvoronoi/parameters.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Thread-policy-dependent AFFECTED storage.
 *
 * Stable internal ordinary-node indices address the bits. If either
 * ComputeVoronoi threading policy is multithreaded, workers may report affected
 * cells concurrently and the atomic implementation is selected.
 */
template <class MeshThreadingT, class CastThreadingT>
using IncrementalVoronoiAffectedVector = std::conditional_t<
    MeshThreadingT::is_multithreaded || CastThreadingT::is_multithreaded,
    detail::AtomicBitVector,
    detail::BitVector>;

/** @brief Event-collection phase of one non-periodic refinement operation. */
enum class IncrementalVoronoiComputePhase : std::uint8_t {
    NewCells,
    RepairCells
};

/**
 * @brief Transient state shared by the compute facade and database wrapper.
 *
 * The state does not own AFFECTED. A caller-provided vector can therefore be
 * accumulated across several remove/refine operations. Structural resizing of
 * that vector happens only outside ComputeVoronoi phases.
 */
template <class MeshT, class AffectedVectorT>
class IncrementalVoronoiComputeState final {
public:
    using Mesh = MeshT;
    using AffectedVector = AffectedVectorT;
    using Index = typename Mesh::Index;

    IncrementalVoronoiComputeState(
        Index internal_count,
        Index old_internal_count,
        AffectedVector& affected)
        : internal_count_(internal_count),
          old_internal_count_(old_internal_count),
          affected_(&affected) {
        if (old_internal_count_ > internal_count_) {
            throw std::invalid_argument(
                "IncrementalVoronoi old-node watermark exceeds internal count.");
        }
        if (affected_->size() < static_cast<std::size_t>(internal_count_)) {
            throw std::invalid_argument(
                "IncrementalVoronoi AFFECTED vector is smaller than the mesh.");
        }
    }

    [[nodiscard]] Index internal_count() const noexcept {
        return internal_count_;
    }

    [[nodiscard]] Index old_internal_count() const noexcept {
        return old_internal_count_;
    }

    [[nodiscard]] IncrementalVoronoiComputePhase phase() const noexcept {
        return phase_;
    }

    void set_phase(IncrementalVoronoiComputePhase phase) noexcept {
        phase_ = phase;
    }

    [[nodiscard]] bool is_old_node(Index internal) const noexcept {
        return internal < old_internal_count_;
    }

    [[nodiscard]] bool is_new_node(Index internal) const noexcept {
        return internal >= old_internal_count_ && internal < internal_count_;
    }

    void mark_affected(Index internal) noexcept {
        if (internal < old_internal_count_) {
            affected_->set(static_cast<std::size_t>(internal));
        }
    }

    [[nodiscard]] bool is_affected(Index internal) const noexcept {
        return internal < internal_count_ &&
               affected_->test(static_cast<std::size_t>(internal));
    }

    [[nodiscard]] AffectedVector& affected() noexcept {
        return *affected_;
    }

    [[nodiscard]] const AffectedVector& affected() const noexcept {
        return *affected_;
    }

private:
    Index internal_count_ = Index{0};
    Index old_internal_count_ = Index{0};
    AffectedVector* affected_ = nullptr;
    IncrementalVoronoiComputePhase phase_ =
        IncrementalVoronoiComputePhase::NewCells;
};

/**
 * @brief Database-compatible wrapper that extracts AFFECTED from stored sigma.
 *
 * AbstractMesh converts the compute-public signature into the compute mesh's
 * stable internal numbering before calling push()/push_facet(). Because the
 * compute mesh uses the owning VoronoiMesh stable indices directly, no second
 * signature translation is necessary here.
 */
template <class MeshT, class AffectedVectorT>
class IncrementalVoronoiComputeDataBase final {
public:
    using Mesh = MeshT;
    using AffectedVector = AffectedVectorT;
    using UnderlyingDatabase = typename Mesh::Database;
    using State = IncrementalVoronoiComputeState<Mesh, AffectedVector>;
    using Scalar = typename UnderlyingDatabase::Scalar;
    using Index = typename UnderlyingDatabase::Index;
    using Sigma = typename UnderlyingDatabase::Sigma;
    using VertexPoint = typename UnderlyingDatabase::VertexPoint;
    using LockType = typename UnderlyingDatabase::LockType;
    using Address = std::size_t;
    using address_type = Address;

    static constexpr int DimensionAtCompileTime =
        UnderlyingDatabase::DimensionAtCompileTime;

    IncrementalVoronoiComputeDataBase(
        Mesh& mesh,
        State& state)
        : database_(*mesh.database_handle()),
          state_(state) {}

    IncrementalVoronoiComputeDataBase(
        const IncrementalVoronoiComputeDataBase&) = delete;
    IncrementalVoronoiComputeDataBase& operator=(
        const IncrementalVoronoiComputeDataBase&) = delete;
    IncrementalVoronoiComputeDataBase(
        IncrementalVoronoiComputeDataBase&&) = delete;
    IncrementalVoronoiComputeDataBase& operator=(
        IncrementalVoronoiComputeDataBase&&) = delete;

    [[nodiscard]] UnderlyingDatabase& underlying_database() noexcept {
        return database_;
    }

    [[nodiscard]] const UnderlyingDatabase& underlying_database() const noexcept {
        return database_;
    }

    /** Observe an already persisted canonical internal signature. */
    template <class SigmaLike>
    void observe_existing_vertex(const SigmaLike& sigma) {
        observe_sigma(sigma);
    }

    [[nodiscard]] Address push(
        const VertexPoint& position,
        const Sigma& sigma) {
        observe_sigma(sigma);
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
        observe_sigma(sigma);
        return static_cast<Address>(
            database_.push_facet(origin, sigma, direction));
    }

    void read_facet(
        Address address,
        VertexPoint& origin,
        Sigma& sigma,
        VertexPoint& direction) const {
        database_.read_facet(address, origin, sigma, direction);
    }

private:
    template <class SigmaLike>
    void observe_sigma(const SigmaLike& sigma) {
        if (state_.phase() != IncrementalVoronoiComputePhase::NewCells) {
            return;
        }

        // Canonical ordinary internal indices precede the high-end boundary
        // encodings. Only OLD ordinary generators seed AFFECTED.
        for (const Index generator : sigma) {
            if (generator >= state_.internal_count()) {
                break;
            }
            if (state_.is_old_node(generator)) {
                state_.mark_affected(generator);
            }
        }
    }

    UnderlyingDatabase& database_;
    State& state_;
};

/**
 * @brief Dense ephemeral compute representation of one ordinary VoronoiMesh.
 *
 * `preferred_front` contains stable internal node indices. Active entries are
 * mapped to compute-public indices `[0, preferred_front.size())`; all remaining
 * active nodes follow in the owning mesh's current public order. Persistent
 * signatures stay in the owning mesh's stable internal numbering.
 */
template <class MeshT, class AffectedVectorT>
class IncrementalVoronoiComputeMesh final
    : public AbstractMesh<
          typename MeshT::NodeScalar,
          typename MeshT::VertexScalar,
          typename MeshT::Index,
          MeshT::DimensionAtCompileTime,
          IncrementalVoronoiComputeDataBase<MeshT, AffectedVectorT>,
          typename MeshT::AddressList,
          DenseIndexMapping<typename MeshT::Index>> {
public:
    using Mesh = MeshT;
    using AffectedVector = AffectedVectorT;
    using NodeScalar = typename Mesh::NodeScalar;
    using VertexScalar = typename Mesh::VertexScalar;
    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using AddressList = typename Mesh::AddressList;
    using BoundaryType = typename Mesh::BoundaryType;
    using State = IncrementalVoronoiComputeState<Mesh, AffectedVector>;
    using Database = IncrementalVoronoiComputeDataBase<Mesh, AffectedVector>;
    using IndexMapping = DenseIndexMapping<Index>;

    static constexpr int Dim = Mesh::DimensionAtCompileTime;
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
        typename Mesh::InternalNodes,
        IndexMapping>;

    using ExtendedNodes = ExtendedVoronoiNodes<PublicActiveNodes>;

public:
    IncrementalVoronoiComputeMesh(
        Mesh& owner,
        State& state,
        const std::vector<Index>& preferred_front = {})
        : Base(
              owner.dimension(),
              make_mapping(owner, preferred_front)),
          owner_(owner),
          database_(owner_, state),
          public_nodes_(owner_.internal_nodes(), this->index_mapping(), owner_.dimension()),
          extended_nodes_(
              std::in_place,
              public_nodes_,
              owner_.boundary()),
          front_size_(static_cast<Index>(preferred_front.size())) {}

    IncrementalVoronoiComputeMesh(
        const IncrementalVoronoiComputeMesh&) = delete;
    IncrementalVoronoiComputeMesh& operator=(
        const IncrementalVoronoiComputeMesh&) = delete;
    IncrementalVoronoiComputeMesh(
        IncrementalVoronoiComputeMesh&&) = delete;
    IncrementalVoronoiComputeMesh& operator=(
        IncrementalVoronoiComputeMesh&&) = delete;

    [[nodiscard]] Mesh& owner() noexcept {
        return owner_;
    }

    [[nodiscard]] const Mesh& owner() const noexcept {
        return owner_;
    }

    [[nodiscard]] Database& compute_database() noexcept {
        return database_;
    }

    [[nodiscard]] const Database& compute_database() const noexcept {
        return database_;
    }

    [[nodiscard]] Index front_size() const noexcept {
        return front_size_;
    }

    /** Map compute-public numbering directly to stable owner numbering. */
    [[nodiscard]] Index stable_internal_node(Index compute_public) const {
        return this->index_mapping().public_to_internal(compute_public);
    }

    /** Map one active stable owner node into compute-public numbering. */
    [[nodiscard]] std::optional<Index>
    compute_public_node(Index internal) const {
        if (internal >= owner_.internal_size()) {
            throw std::out_of_range(
                "IncrementalVoronoiComputeMesh stable node out of range.");
        }
        return this->index_mapping().internal_to_public(internal);
    }

    [[nodiscard]] ExtendedNodes& concrete_extended_nodes() noexcept {
        return *extended_nodes_;
    }

    [[nodiscard]] const ExtendedNodes& concrete_extended_nodes() const noexcept {
        return *extended_nodes_;
    }

private:
    [[nodiscard]] static IndexMapping make_mapping(
        const Mesh& owner,
        const std::vector<Index>& preferred_front) {
        const Index internal_count = owner.internal_size();
        const Index public_count = owner.size();

        if (preferred_front.size() > static_cast<std::size_t>(public_count)) {
            throw std::invalid_argument(
                "IncrementalVoronoiComputeMesh front exceeds active node count.");
        }

        std::vector<Index> public_to_internal;
        public_to_internal.reserve(static_cast<std::size_t>(public_count));
        std::vector<Index> internal_to_public(
            static_cast<std::size_t>(internal_count),
            IndexMapping::invalid_index());
        detail::BitVector selected(
            static_cast<std::size_t>(internal_count));

        auto append_internal = [&](Index internal) {
            if (internal >= internal_count) {
                throw std::out_of_range(
                    "IncrementalVoronoiComputeMesh stable node out of range.");
            }
            if (selected.test(static_cast<std::size_t>(internal))) {
                throw std::invalid_argument(
                    "IncrementalVoronoiComputeMesh front contains a duplicate node.");
            }
            if (!owner.internal_node_to_public(internal)) {
                throw std::invalid_argument(
                    "IncrementalVoronoiComputeMesh front contains a deleted node.");
            }

            selected.set(static_cast<std::size_t>(internal));
            const Index compute_public =
                static_cast<Index>(public_to_internal.size());
            public_to_internal.push_back(internal);
            internal_to_public[static_cast<std::size_t>(internal)] =
                compute_public;
        };

        for (const Index internal : preferred_front) {
            append_internal(internal);
        }

        // The owner's dense public order contains every active stable slot once.
        for (Index owner_public = Index{0};
             owner_public < public_count;
             ++owner_public) {
            const Index internal =
                owner.public_node_to_internal(owner_public);
            if (selected.test(static_cast<std::size_t>(internal))) {
                continue;
            }

            const Index compute_public =
                static_cast<Index>(public_to_internal.size());
            public_to_internal.push_back(internal);
            internal_to_public[static_cast<std::size_t>(internal)] =
                compute_public;
        }

        if (public_to_internal.size() !=
            static_cast<std::size_t>(public_count)) {
            throw std::logic_error(
                "IncrementalVoronoiComputeMesh failed to build a dense active mapping.");
        }

        IndexMapping mapping;
        mapping.assign(
            std::move(public_to_internal),
            std::move(internal_to_public));
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
            "IncrementalVoronoiComputeMesh boundary is fixed for one compute phase.");
    }

    [[nodiscard]] Database& database_impl() noexcept override {
        return database_;
    }

    [[nodiscard]] const Database& database_impl() const noexcept override {
        return database_;
    }

    [[nodiscard]] Index internal_node_count_impl() const noexcept override {
        return owner_.internal_size();
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return owner_.primary_address_lists_.at(
            static_cast<std::size_t>(internal_node));
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return owner_.secondary_address_lists_.at(
            static_cast<std::size_t>(internal_node));
    }

    void register_primary_vertex_impl(
        Index internal_node,
        Address address) override {
        owner_.primary_address_lists_.at(
            static_cast<std::size_t>(internal_node)).push_back(address);
    }

    void register_secondary_vertex_impl(
        Index internal_node,
        Address address) override {
        owner_.secondary_address_lists_.at(
            static_cast<std::size_t>(internal_node)).push_back(address);
    }

    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return owner_.infinite_edge_addresses_;
    }

    void register_infinite_edge_impl(Address address) override {
        owner_.infinite_edge_addresses_.push_back(address);
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
            "IncrementalVoronoiComputeMesh cannot structurally delete nodes.");
    }

    // ComputeVoronoi only appends records. Structural cleanup belongs to the
    // owning mesh's explicit erase/filter operations between compute phases.
    void cleanup_vertex_lists_impl() override {}

    Mesh& owner_;
    Database database_;
    PublicActiveNodes public_nodes_;
    std::optional<ExtendedNodes> extended_nodes_;
    Index front_size_ = Index{0};
};

namespace detail {

/** @brief Reject periodic boundaries in the ordinary incremental algorithms. */
template <class Mesh>
void require_nonperiodic_incremental_boundary(const Mesh& mesh) {
    using Index = typename Mesh::Index;
    const auto& boundary = mesh.boundary();
    for (Index plane = Index{0};
         plane < boundary.size();
         ++plane) {
        if (boundary[plane].is_periodic()) {
            throw std::invalid_argument(
                "RemoveVoronoi/RefineVoronoi require a non-periodic boundary.");
        }
    }
}

/** @brief Ensure caller-owned AFFECTED storage covers all stable mesh slots. */
template <class AffectedVector, class Index>
void ensure_affected_size(
    AffectedVector& affected,
    Index internal_count) {
    const std::size_t required = static_cast<std::size_t>(internal_count);
    if (affected.size() < required) {
        affected.resize(required, false);
    }
}

/** @brief Collect active stable nodes whose AFFECTED bit is set. */
template <class Mesh, class AffectedVector>
[[nodiscard]] std::vector<typename Mesh::Index>
active_affected_nodes(
    const Mesh& mesh,
    const AffectedVector& affected,
    typename Mesh::Index stable_limit) {
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
        const auto public_node = mesh.internal_node_to_public(internal);
        if (!public_node) {
            continue;
        }
        result.push_back(internal);
    }
    return result;
}

/**
 * @brief Run unchanged classical ComputeVoronoi on one selected stable front.
 *
 * `verbose` is forwarded directly to ComputeVoronoi::compute(); incremental
 * orchestration owns no second progress meter.
 */
template <class ComputeMesh,
          class SearchKeyword,
          class RayParameters,
          class MeshThreading,
          class CastThreading,
          class QueueParameters,
          class EdgeParameters>
[[nodiscard]] std::size_t run_incremental_compute(
    ComputeMesh& computation_mesh,
    typename ComputeMesh::Index front_count,
    const SearchKeyword& search_keyword,
    const RayParameters& ray_parameters,
    MeshThreading mesh_threading,
    CastThreading cast_threading,
    QueueParameters queue_parameters,
    EdgeParameters edge_parameters,
    bool verbose = false) {
    using Index = typename ComputeMesh::Index;

    if (front_count == Index{0}) {
        return 0;
    }

    auto tree = geometry::make_search_tree(
        computation_mesh,
        search_keyword);
    auto raycaster = make_raycaster(tree, ray_parameters);

    using RayCaster = decltype(raycaster);
    using Core = ComputeVoronoi<
        ComputeMesh,
        RayCaster,
        MeshThreading,
        CastThreading,
        QueueParameters,
        EdgeParameters>;

    Core core(
        computation_mesh,
        raycaster,
        std::move(mesh_threading),
        std::move(cast_threading),
        static_cast<Index>(front_count - Index{1}),
        std::move(queue_parameters),
        std::move(edge_parameters));
    core.compute(verbose);
    return core.new_vertex_count();
}

} // namespace detail

} // namespace highvoronoi
