#pragma once

/**
 * @file periodic_boundary_repair.hpp
 * @brief Serial repair of Voronoi vertices hidden by the bootstrap periodic boundary.
 *
 * During the first HighVoronoi construction pass the internal periodic boundary
 * still coincides with the visible domain. Finite vertices on those periodic
 * planes are deliberately not persisted. Moving the internal boundary outward
 * later can expose finite vertices whose ordinary generators are all old/visible;
 * such vertices are not discovered by RefineVoronoi's NEW-cell pass.
 *
 * Repair has two separated phases:
 *
 * 1. In the bootstrap geometry, transient periodic-boundary vertices are walked
 *    inward across every edge that drops a periodic boundary generator. The
 *    persistent opposite vertex signature is retained as a repair seed.
 *
 * 2. After periodic reference closure is complete, a sparse mesh wrapper exposes
 *    the complete final node geometry but only transient repair vertex lists.
 *    Unchanged ComputeVoronoi is run serially on the visible cells that currently
 *    contain a seed/new repair vertex. Newly persisted vertices are routed both
 *    to HighVoronoi's real address lists and to the transient repair lists.
 */

#include <highvoronoi/detail/atomic_bit_vector.hpp>
#include <highvoronoi/detail/locks.hpp>
#include <highvoronoi/geometry/compute_voronoi.hpp>
#include <highvoronoi/geometry/edge_iterator.hpp>
#include <highvoronoi/geometry/high_voronoi_compute_mesh.hpp>
#include <highvoronoi/geometry/incremental_voronoi_compute_mesh.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/raycaster.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace highvoronoi {

/** Diagnostic counters for one complete hidden-periodic-vertex repair. */
struct PeriodicBoundaryRepairReport {
    std::size_t captured_boundary_vertices = 0;
    std::size_t opening_edges = 0;
    std::size_t bootstrap_seed_signatures = 0;
    std::size_t resolved_final_seeds = 0;
    std::size_t stale_final_seeds = 0;
    std::size_t sweeps = 0;
    std::size_t computed_visible_cells = 0;
    std::size_t new_vertices = 0;
};

namespace detail {

template <class Sigma>
[[nodiscard]] bool equal_sigma(const Sigma& left, const Sigma& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

template <class Sigma>
void append_unique_sigma(std::vector<Sigma>& target, Sigma sigma) {
    if (std::find_if(
            target.begin(),
            target.end(),
            [&](const Sigma& existing) {
                return equal_sigma(existing, sigma);
            }) == target.end()) {
        target.push_back(std::move(sigma));
    }
}

/**
 * Collect persistent inward counterparts while the bootstrap boundary is still
 * unchanged. Only signatures are retained across the later periodization rounds;
 * no database address is allowed to become stale across refinement.
 */
template <class HighMesh, class SearchKeyword, class RayParameters>
[[nodiscard]] std::vector<typename HighMesh::Sigma>
collect_periodic_boundary_repair_seeds(
    HighMesh& owner,
    const std::vector<std::pair<
        typename HighMesh::VertexPoint,
        typename HighMesh::Sigma>>& boundary_vertices,
    const SearchKeyword& search_keyword,
    const RayParameters& ray_parameters,
    PeriodicBoundaryRepairReport& report) {

    using Index = typename HighMesh::Index;
    using Sigma = typename HighMesh::Sigma;
    using VertexPoint = typename HighMesh::VertexPoint;
    using State = HighVoronoiComputeRoundState<HighMesh, BitVector>;
    using ComputeMesh = HighVoronoiComputeMesh<HighMesh, BitVector>;

    report.captured_boundary_vertices += boundary_vertices.size();
    if (boundary_vertices.empty()) {
        return {};
    }

    State state(
        owner.internal_node_count(),
        owner.external_boundary().size(),
        owner.internal_node_count());
    state.set_phase(HighVoronoiComputePhase::RepairCells);

    ComputeMesh mesh(owner, state);
    auto tree = geometry::make_search_tree(mesh, search_keyword);
    auto raycaster = make_raycaster(tree, ray_parameters);

    using RayCaster = decltype(raycaster);
    using EdgeIteratorType = EdgeIterator<
        typename RayCaster::ExtendedNodes,
        EmptyLock>;

    EdgeIteratorType iterator(
        raycaster.extended_nodes(),
        raycaster.parameters().ray_tolerance);

    std::vector<Index> mirror_indices;
    mirror_indices.reserve(static_cast<std::size_t>(
        raycaster.extended_nodes().mirror_count()));
    for (Index plane = Index{0};
         plane < raycaster.extended_nodes().mirror_count();
         ++plane) {
        mirror_indices.push_back(
            raycaster.extended_nodes().mirror_index(plane));
    }

    Sigma public_sigma;
    Sigma walk_sigma;
    Sigma internal_candidate;
    std::vector<Sigma> seeds;

    const auto is_periodic_public_boundary = [&](Index public_index) {
        if (public_index < mesh.size()) {
            return false;
        }
        const Index plane = static_cast<Index>(public_index - mesh.size());
        return plane < owner.external_boundary().size() &&
               owner.external_boundary()[plane].is_periodic();
    };

    const auto make_public_signature = [&](
        const Sigma& internal,
        Sigma& public_out) {
        public_out.clear();
        public_out.reserve(internal.size());
        for (const Index generator : internal) {
            if (generator < owner.internal_node_count()) {
                const auto public_node =
                    mesh.compute_public_of_global_internal(generator);
                if (!public_node) {
                    public_out.clear();
                    return false;
                }
                public_out.push_back(*public_node);
                continue;
            }
            if (!owner.is_boundary_internal_index(generator)) {
                public_out.clear();
                return false;
            }
            const Index plane = owner.decode_boundary_internal_index(generator);
            public_out.push_back(static_cast<Index>(mesh.size() + plane));
        }
        std::sort(public_out.begin(), public_out.end());
        return true;
    };

    const auto make_internal_signature = [&](
        const Sigma& public_in,
        Sigma& internal_out) {
        internal_out.clear();
        internal_out.reserve(public_in.size());
        for (const Index generator : public_in) {
            if (generator < mesh.size()) {
                internal_out.push_back(mesh.global_internal_node(generator));
                continue;
            }
            const Index plane = static_cast<Index>(generator - mesh.size());
            if (plane >= owner.internal_boundary().size()) {
                throw std::out_of_range(
                    "Repair seed candidate contains an invalid boundary index.");
            }
            internal_out.push_back(owner.encode_boundary_internal_index(plane));
        }
        std::sort(internal_out.begin(), internal_out.end());
    };

    for (const auto& boundary_vertex : boundary_vertices) {
        const VertexPoint& position = boundary_vertex.first;
        const Sigma& internal_sigma = boundary_vertex.second;
        if (internal_sigma.empty()) {
            continue;
        }

        Index primary_internal = owner.internal_node_count();
        for (const Index generator : internal_sigma) {
            if (generator < owner.internal_node_count()) {
                primary_internal = generator;
                break;
            }
        }
        if (primary_internal >= owner.internal_node_count() ||
            !owner.is_active_internal(primary_internal)) {
            continue;
        }

        const auto primary_public =
            mesh.compute_public_of_global_internal(primary_internal);
        if (!primary_public ||
            !make_public_signature(internal_sigma, public_sigma)) {
            continue;
        }

        raycaster.activate_cell(*primary_public, mirror_indices);
        iterator.reset(
            public_sigma,
            position,
            *primary_public,
            typename EdgeIteratorType::OnSysVoronoi{});

        while (const auto edge = iterator.next()) {
            if (edge->indices().empty() ||
                edge->indices()[0] != *primary_public) {
                continue;
            }

            bool drops_periodic_boundary = false;
            for (const Index origin_generator : public_sigma) {
                if (!is_periodic_public_boundary(origin_generator)) {
                    continue;
                }
                if (std::find(
                        edge->full_indices().begin(),
                        edge->full_indices().end(),
                        origin_generator) == edge->full_indices().end()) {
                    drops_periodic_boundary = true;
                    break;
                }
            }
            if (!drops_periodic_boundary) {
                continue;
            }

            ++report.opening_edges;

            const auto try_opening_direction = [&](const auto& direction) {
                walk_sigma.assign(
                    edge->full_indices().begin(),
                    edge->full_indices().end());

                const auto result = raycaster.cast(
                    walk_sigma,
                    position,
                    direction,
                    edge->indices(),
                    public_sigma,
                    RayCastUsage::WalkRay,
                    edge->cycle_error());

                if (result.status == RayCastStatus::Infinite ||
                    walk_sigma.size() <
                        static_cast<std::size_t>(owner.dimension()) + 1) {
                    return false;
                }

                make_internal_signature(
                    walk_sigma,
                    internal_candidate);

                for (const Index generator : internal_candidate) {
                    if (generator < owner.internal_node_count()) {
                        continue;
                    }
                    if (!owner.is_boundary_internal_index(generator)) {
                        continue;
                    }
                    const Index plane =
                        owner.decode_boundary_internal_index(generator);
                    if (owner.external_boundary()[plane].is_periodic()) {
                        return false;
                    }
                }

                // The inward counterpart must already be persistent in the
                // bootstrap mesh. The final repair later resolves the signature
                // again, so intervening refinement may legitimately make it stale.
                if (!owner.database().contains(internal_candidate)) {
                    return false;
                }

                append_unique_sigma(seeds, internal_candidate);
                return true;
            };

            if (!try_opening_direction(edge->direction())) {
                typename RayCaster::Point reverse_direction = -edge->direction();
                (void)try_opening_direction(reverse_direction);
            }
        }
    }

    report.bootstrap_seed_signatures += seeds.size();
    return seeds;
}

} // namespace detail

/**
 * Serial transient state shared by successive sparse repair mesh wrappers.
 * Address lists contain only explicit bootstrap seeds and vertices newly stored
 * by the repair itself. The underlying HighVoronoi address lists remain the
 * persistent source of truth.
 */
template <class HighMeshT>
class HighVoronoiSparseRepairState final {
public:
    using HighMesh = HighMeshT;
    using Index = typename HighMesh::Index;
    using Address = typename HighMesh::Address;
    using AddressList = typename HighMesh::AddressList;
    using Sigma = typename HighMesh::Sigma;
    using VertexPoint = typename HighMesh::VertexPoint;

    explicit HighVoronoiSparseRepairState(HighMesh& owner)
        : owner_(owner),
          primary_(static_cast<std::size_t>(owner.internal_node_count())),
          secondary_(static_cast<std::size_t>(owner.internal_node_count())),
          dirty_(static_cast<std::size_t>(owner.internal_node_count())) {}

    [[nodiscard]] const AddressList& primary(Index internal) const {
        return primary_.at(static_cast<std::size_t>(internal));
    }

    [[nodiscard]] const AddressList& secondary(Index internal) const {
        return secondary_.at(static_cast<std::size_t>(internal));
    }

    void register_primary(Index internal, Address address) {
        primary_.at(static_cast<std::size_t>(internal)).push_back(address);
        mark_dirty_if_visible(internal);
    }

    void register_secondary(Index internal, Address address) {
        secondary_.at(static_cast<std::size_t>(internal)).push_back(address);
        mark_dirty_if_visible(internal);
    }

    /** Resolve a bootstrap signature in the final database and seed all cells. */
    [[nodiscard]] bool add_seed(const Sigma& sigma) {
        const auto address = find_current_address(sigma);
        if (!address) {
            return false;
        }
        if (!seed_addresses_.insert(*address).second) {
            return true;
        }

        const Index ordinary_count = owner_.internal_node_count();
        if (sigma.empty() || sigma.front() >= ordinary_count) {
            return false;
        }

        primary_[static_cast<std::size_t>(sigma.front())].push_back(*address);
        mark_dirty_if_visible(sigma.front());
        for (std::size_t p = 1;
             p < sigma.size() && sigma[p] < ordinary_count;
             ++p) {
            secondary_[static_cast<std::size_t>(sigma[p])].push_back(*address);
            mark_dirty_if_visible(sigma[p]);
        }
        return true;
    }

    /**
     * Consume the current dirty set in visible insertion order. Bits are reset
     * before the compute sweep, so vertices discovered during the sweep mark the
     * next fixpoint iteration even when their cell was processed already.
     */
    [[nodiscard]] std::vector<Index> take_dirty_visible_nodes() {
        std::vector<Index> result;
        result.reserve(static_cast<std::size_t>(owner_.visible_public_count()));

        for (Index visible = Index{0};
             visible < owner_.visible_public_count();
             ++visible) {
            const Index internal = owner_.visible_public_to_internal(visible);
            if (!dirty_.test(static_cast<std::size_t>(internal))) {
                continue;
            }
            dirty_.reset(static_cast<std::size_t>(internal));
            result.push_back(internal);
        }
        return result;
    }

private:
    [[nodiscard]] std::optional<Address>
    find_current_address(const Sigma& sigma) const {
        if (sigma.empty() || sigma.front() >= owner_.internal_node_count()) {
            return std::nullopt;
        }

        VertexPoint position;
        if constexpr (HighMesh::DimensionAtCompileTime == Dynamic) {
            position.resize(static_cast<Eigen::Index>(owner_.dimension()));
        }
        Sigma stored_sigma;

        auto search = [&](const AddressList& addresses) -> std::optional<Address> {
            const std::size_t count = addresses.size();
            for (std::size_t p = 0; p < count; ++p) {
                const Address address = addresses[p];
                stored_sigma.clear();
                owner_.read_internal_vertex(address, position, stored_sigma);
                if (detail::equal_sigma(stored_sigma, sigma)) {
                    return address;
                }
            }
            return std::nullopt;
        };

        if (const auto found = search(
                owner_.primary_addresses_internal(sigma.front()))) {
            return found;
        }
        return search(owner_.secondary_addresses_internal(sigma.front()));
    }

    void mark_dirty_if_visible(Index internal) {
        if (internal < owner_.internal_node_count() &&
            owner_.is_active_internal(internal) &&
            owner_.is_visible_internal(internal)) {
            dirty_.set(static_cast<std::size_t>(internal));
        }
    }

    HighMesh& owner_;
    std::deque<AddressList> primary_;
    std::deque<AddressList> secondary_;
    detail::BitVector dirty_;
    std::unordered_set<Address> seed_addresses_;
};

/** Database facade used by the sparse repair wrapper. */
template <class HighMeshT>
class HighVoronoiSparseRepairDataBase final {
public:
    using HighMesh = HighMeshT;
    using UnderlyingDatabase = typename HighMesh::Database;
    using Scalar = typename UnderlyingDatabase::Scalar;
    using Index = typename UnderlyingDatabase::Index;
    using LockType = typename UnderlyingDatabase::LockType;
    using Address = std::size_t;

    explicit HighVoronoiSparseRepairDataBase(HighMesh& owner)
        : owner_(owner), database_(owner.database()) {}

    [[nodiscard]] UnderlyingDatabase& underlying_database() noexcept {
        return database_;
    }
    [[nodiscard]] const UnderlyingDatabase& underlying_database() const noexcept {
        return database_;
    }

    template <class RVector, class SigmaVector>
    [[nodiscard]] Address push(
        const RVector& position,
        const SigmaVector& sigma) {
        if (contains_periodic_internal_boundary(sigma)) {
            return Address{0};
        }

        // This repair exists solely for vertices hidden by the bootstrap
        // periodic boundary and therefore newly creates only visible-old
        // topology. A genuinely new vertex containing an invisible reference
        // would mean that the normal periodic-reference closure was incomplete;
        // silently accepting it here would bypass mirror-request generation.
        if (contains_invisible_ordinary_generator(sigma) &&
            !database_.contains(sigma)) {
            throw std::logic_error(
                "Sparse periodic repair discovered a new vertex containing "
                "an invisible reference generator.");
        }

        return static_cast<Address>(database_.push(position, sigma));
    }

    template <class RVector, class SigmaVector>
    void read(Address address, RVector& position, SigmaVector& sigma) const {
        database_.read(address, position, sigma);
    }

    template <class SigmaVector>
    [[nodiscard]] bool contains(const SigmaVector& sigma) const {
        return database_.contains(sigma);
    }

    template <class SigmaVector>
    bool erase(Address address, const SigmaVector& sigma) {
        return database_.erase(address, sigma);
    }

    template <class SigmaVector>
    [[nodiscard]] bool register_signature(const SigmaVector& sigma) {
        return database_.register_signature(sigma);
    }

    template <class SigmaVector>
    [[nodiscard]] bool erase_signature(const SigmaVector& sigma) {
        return database_.erase_signature(sigma);
    }

    template <class RVector, class SigmaVector, class UVector>
    [[nodiscard]] Address push_facet(
        const RVector& origin,
        SigmaVector& sigma,
        const UVector& direction) {
        if (contains_periodic_internal_boundary(sigma)) {
            return Address{0};
        }
        return static_cast<Address>(
            database_.push_facet(origin, sigma, direction));
    }

    template <class RVector, class SigmaVector, class UVector>
    void read_facet(
        Address address,
        RVector& origin,
        SigmaVector& sigma,
        UVector& direction) const {
        database_.read_facet(address, origin, sigma, direction);
    }

private:
    template <class SigmaLike>
    [[nodiscard]] bool contains_invisible_ordinary_generator(
        const SigmaLike& sigma) const {
        for (const Index generator : sigma) {
            if (generator >= owner_.internal_node_count()) {
                break;
            }
            if (owner_.is_active_internal(generator) &&
                !owner_.is_visible_internal(generator)) {
                return true;
            }
        }
        return false;
    }

    template <class SigmaLike>
    [[nodiscard]] bool contains_periodic_internal_boundary(
        const SigmaLike& sigma) const {
        for (const Index generator : sigma) {
            if (generator < owner_.internal_node_count()) {
                continue;
            }
            if (!owner_.is_boundary_internal_index(generator)) {
                throw std::logic_error(
                    "Sparse repair database received an invalid boundary index.");
            }
            const Index plane = owner_.decode_boundary_internal_index(generator);
            if (owner_.external_boundary()[plane].is_periodic()) {
                return true;
            }
        }
        return false;
    }

    HighMesh& owner_;
    UnderlyingDatabase& database_;
};

/**
 * Complete final HighVoronoi geometry with sparse transient repair address lists.
 * `preferred_front` contains stable visible internal nodes selected for this
 * serial sweep; all other active nodes remain available to KD/raycast geometry.
 */
template <class HighMeshT>
class HighVoronoiSparseRepairMesh final
    : public AbstractMesh<
          typename HighMeshT::NodeScalar,
          typename HighMeshT::VertexScalar,
          typename HighMeshT::Index,
          HighMeshT::DimensionAtCompileTime,
          NodeAccessMode::Stored,
          HighVoronoiSparseRepairDataBase<HighMeshT>,
          typename HighMeshT::AddressList,
          DenseIndexMapping<typename HighMeshT::Index>> {
public:
    using HighMesh = HighMeshT;
    using State = HighVoronoiSparseRepairState<HighMesh>;
    using NodeScalar = typename HighMesh::NodeScalar;
    using VertexScalar = typename HighMesh::VertexScalar;
    using Index = typename HighMesh::Index;
    using Address = typename HighMesh::Address;
    using AddressList = typename HighMesh::AddressList;
    using BoundaryType = typename HighMesh::BoundaryType;
    using Database = HighVoronoiSparseRepairDataBase<HighMesh>;
    using IndexMapping = DenseIndexMapping<Index>;

    static constexpr int Dim = HighMesh::DimensionAtCompileTime;
    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;

    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        Dim,
        NodeAccessMode::Stored,
        Database,
        AddressList,
        IndexMapping>;
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;

private:
    class PublicActiveNodes final
        : public StoredNodeAccess<NodeScalar, Dim, Index> {
    public:
        using Scalar = NodeScalar;
        static constexpr int DimensionAtCompileTime = Dim;
        static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;
        using AccessBase = StoredNodeAccess<NodeScalar, Dim, Index>;

        PublicActiveNodes(const HighMesh& owner, const IndexMapping& mapping)
            : AccessBase(owner.dimension()), owner_(owner), mapping_(mapping) {}

        [[nodiscard]] Index size() const noexcept override {
            return mapping_.size();
        }

        [[nodiscard]] Scalar get_data(
            Index public_node,
            Index coordinate) const override {
            return owner_.internal_nodes().get_data(
                mapping_.public_to_internal(public_node),
                coordinate);
        }

    private:
        [[nodiscard]] const Scalar* get_stored_node_pointer(
            Index public_node) const override {
            return owner_.internal_nodes().stable_node_data(
                mapping_.public_to_internal(public_node));
        }

        const HighMesh& owner_;
        const IndexMapping& mapping_;
    };

    using ExtendedNodes = ExtendedVoronoiNodes<PublicActiveNodes>;

public:
    HighVoronoiSparseRepairMesh(
        HighMesh& owner,
        State& state,
        const std::vector<Index>& preferred_front)
        : Base(owner.dimension(), make_mapping(owner, preferred_front)),
          owner_(owner),
          state_(state),
          database_(owner_),
          public_nodes_(owner_, this->index_mapping()),
          extended_nodes_(
              std::in_place,
              public_nodes_,
              owner_.internal_boundary()) {}

    [[nodiscard]] Index global_internal_node(Index public_node) const {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    compute_public_of_global_internal(Index internal) const {
        return this->index_mapping().internal_to_public(internal);
    }

    /** One-past-end of the complete visible prefix. */
    [[nodiscard]] Index visible_end() const noexcept {
        return owner_.visible_public_count();
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

        // Requested dirty visible cells form the ComputeVoronoi prefix.
        for (const Index internal : preferred_front) {
            if (!owner.is_visible_internal(internal)) {
                throw std::invalid_argument(
                    "Sparse repair front must contain visible HighVoronoi nodes.");
            }
            append(internal);
        }

        // All remaining visible nodes follow in their public insertion order,
        // so the wrapper retains the same useful visible-prefix invariant as
        // VisibleFirstMesh even though only preferred_front is computed.
        for (Index visible = Index{0};
             visible < owner.visible_public_count();
             ++visible) {
            append(owner.visible_public_to_internal(visible));
        }

        // Invisible active reference nodes remain part of the geometry/KDTree
        // but never enter the range-limited repair compute prefix.
        for (Index internal = Index{0}; internal < internal_count; ++internal) {
            if (owner.is_active_internal(internal) &&
                !owner.is_visible_internal(internal)) {
                append(internal);
            }
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
            "Sparse repair uses the final immutable HighVoronoi internal boundary.");
    }

    [[nodiscard]] Database& database_impl() noexcept override {
        return database_;
    }
    [[nodiscard]] const Database& database_impl() const noexcept override {
        return database_;
    }

    [[nodiscard]] Index internal_node_count_impl() const noexcept override {
        return owner_.internal_node_count();
    }

    [[nodiscard]] Index public_node_to_internal_impl(Index public_node) const override {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        return this->index_mapping().internal_to_public(internal_node);
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return state_.primary(internal_node);
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return state_.secondary(internal_node);
    }

    void register_primary_vertex_impl(Index internal_node, Address address) override {
        owner_.register_primary_internal(internal_node, address);
        state_.register_primary(internal_node, address);
    }

    void register_secondary_vertex_impl(Index internal_node, Address address) override {
        owner_.register_secondary_internal(internal_node, address);
        state_.register_secondary(internal_node, address);
    }

    [[nodiscard]] const AddressList& infinite_edge_addresses_impl() const override {
        return owner_.infinite_addresses_internal();
    }

    void register_infinite_edge_impl(Address address) override {
        owner_.register_infinite_internal(address);
    }

    void mark_internal_node_deleted_impl(Index) override {
        throw std::logic_error(
            "Sparse repair cannot structurally delete HighVoronoi nodes.");
    }

    void cleanup_vertex_lists_impl() override {}

    HighMesh& owner_;
    State& state_;
    Database database_;
    PublicActiveNodes public_nodes_;
    std::optional<ExtendedNodes> extended_nodes_;
};

/** Run the final serial sparse repair to a vertex-discovery fixpoint. */
template <class HighMesh,
          class SearchKeyword,
          class RayParameters,
          class QueueParameters,
          class EdgeParameters>
void run_periodic_boundary_sparse_repair(
    HighMesh& owner,
    const std::vector<typename HighMesh::Sigma>& seed_signatures,
    const SearchKeyword& search_keyword,
    const RayParameters& ray_parameters,
    QueueParameters queue_parameters,
    EdgeParameters edge_parameters,
    std::size_t maximum_sweeps,
    PeriodicBoundaryRepairReport& report) {

    using Index = typename HighMesh::Index;
    using RepairState = HighVoronoiSparseRepairState<HighMesh>;
    using RepairMesh = HighVoronoiSparseRepairMesh<HighMesh>;

    if (seed_signatures.empty()) {
        return;
    }

    RepairState state(owner);
    for (const auto& sigma : seed_signatures) {
        if (state.add_seed(sigma)) {
            ++report.resolved_final_seeds;
        } else {
            ++report.stale_final_seeds;
        }
    }

    for (;;) {
        std::vector<Index> dirty = state.take_dirty_visible_nodes();
        if (dirty.empty()) {
            return;
        }
        if (report.sweeps >= maximum_sweeps) {
            throw std::runtime_error(
                "HighVoronoi periodic boundary repair exceeded its sweep limit.");
        }

        report.computed_visible_cells += dirty.size();
        RepairMesh mesh(owner, state, dirty);
        const std::size_t found = detail::run_incremental_compute(
            mesh,
            static_cast<Index>(dirty.size()),
            search_keyword,
            ray_parameters,
            SingleThread{},
            SingleThread{},
            queue_parameters,
            edge_parameters);

        report.new_vertices += found;
        ++report.sweeps;
    }
}

} // namespace highvoronoi
