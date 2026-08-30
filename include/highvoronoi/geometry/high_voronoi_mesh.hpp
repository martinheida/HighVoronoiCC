#pragma once

/**
 * @file high_voronoi_mesh.hpp
 * @brief Mutable HighVoronoi owner with visible nodes, periodic references,
 *        stable internal numbering, and central vertex storage.
 *
 * HighVoronoiMesh separates three notions which deliberately coincide in a
 * simple VoronoiMesh but no longer coincide after refinement/periodization:
 *
 *  - stable internal nodes: every node ever appended receives one permanent
 *    ordinary-node index;
 *  - active data nodes: every currently active internal node, in stable
 *    insertion order when exposed through HighVoronoiDataMeshView;
 *  - active computation nodes: the same active nodes, but temporarily
 *    reordered so new/affected cells can be presented first;
 *  - visible public nodes: only active nodes intended for the external mesh.
 *
 * The external boundary is fixed geometry visible to the user. The internal
 * boundary starts as an exact copy and may later be expanded only along
 * periodic planes so that newly generated reference nodes are enclosed. Plane
 * order never changes; therefore the standard high-end AbstractMesh boundary
 * encoding remains stable in stored signatures.
 *
 * Invisible periodic nodes always refer to one visible original node and carry
 * a bit mask over external boundary planes. For a mask M,
 *
 *     node(image) = node(reference) + sum(periodic_shift(plane), plane in M).
 *
 * This is the C++ analogue of Julia's `references` + `reference_shifts`.
 * Stable internal indices are never compacted; node deletion only changes
 * active/public state. Vertex records are tombstoned in the central database.
 *
 * HighVoronoiMesh itself exposes the visible node set through AbstractMesh.
 * Its compile-time ProjectedIndexMapping keeps a canonical public-to-internal
 * map for visible cells, while internal-to-public deliberately projects active
 * invisible periodic copies onto the public index of their visible reference.
 * Public vertex iteration reads only the visible cell's own address lists,
 * projects every internal generator through this mapping, and translates the
 * stored vertex position by periodic partner shifts into the external domain.
 * ComputeHighVoronoi never uses this non-bijective public representation; its
 * dedicated HighVoronoiComputeMesh builds a separate dense bijective mapping
 * over all active internal nodes.
 */

#include <highvoronoi/detail/atomic_bit_vector.hpp>
#include <highvoronoi/detail/read_write_list.hpp>
#include <highvoronoi/geometry/abstract_mesh.hpp>
#include <highvoronoi/geometry/search_tree_factory_crtp.hpp>
#include <highvoronoi/geometry/voronoi_nodes.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
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

template <class HighMeshT, class AffectedVectorT>
class HighVoronoiComputeMesh;

template <class HighMeshT, class AffectedVectorT>
class HighVoronoiComputeDataBase;

template <class MeshT, class AffectedVectorT>
struct IncrementalVoronoiBackend;

template <class HighMeshT>
class HighVoronoiDataMeshView;

template <class HighMeshT>
class VisibleFirstMesh;

namespace detail {

template <class Database>
using HighMeshAddressList = ReadWriteAddressList<
    std::vector<std::size_t>,
    typename Database::LockType>;

template <class Index>
[[nodiscard]] constexpr Index highvoronoi_invalid_index() noexcept {
    return std::numeric_limits<Index>::max();
}

template <class Database, class Engine, class Index, class = void>
struct CanRegisterComputeEngine : std::false_type {};

template <class Database, class Engine, class Index>
struct CanRegisterComputeEngine<
    Database,
    Engine,
    Index,
    std::void_t<decltype(
        std::declval<Database&>().register_engine(
            std::declval<std::shared_ptr<Engine>>(),
            std::declval<Index>()))>> : std::true_type {};

template <class Database, class Engine, class Index>
inline constexpr bool CanRegisterComputeEngineV =
    CanRegisterComputeEngine<Database, Engine, Index>::value;

} // namespace detail

/**
 * @brief Central mutable mesh used by the HighVoronoi refinement workflow.
 *
 * @tparam NodeScalarT Scalar used for node coordinates.
 * @tparam Dim Compile-time dimension or highvoronoi::Dynamic.
 * @tparam DatabaseT Central database backend. It may be HVDataBase or
 *         HybridDataBase; only the public database contract is used by normal
 *         storage. `append_engine()` additionally detects `register_engine()`
 *         at compile time and retains an engine when the backend supports it.
 */
template <typename NodeScalarT, int Dim, class DatabaseT>
class HighVoronoiMesh final
    : public AbstractMesh<
          NodeScalarT,
          typename DatabaseT::Scalar,
          typename DatabaseT::Index,
          Dim,
          NodeAccessMode::Stored,
          DatabaseT,
          detail::HighMeshAddressList<DatabaseT>,
          ProjectedIndexMapping<typename DatabaseT::Index>> {
public:
    using Database = DatabaseT;
    using NodeScalar = NodeScalarT;
    using VertexScalar = typename Database::Scalar;
    using Index = typename Database::Index;

    using IndexMapping = ProjectedIndexMapping<Index>;

    using Base = AbstractMesh<
        NodeScalar,
        VertexScalar,
        Index,
        Dim,
        NodeAccessMode::Stored,
        Database,
        detail::HighMeshAddressList<Database>,
        IndexMapping>;

    using Address = typename Base::Address;
    using AddressList = typename Base::AddressList;
    using BoundaryType = typename Base::BoundaryType;
    using NodePoint = typename Base::NodePoint;
    using VertexPoint = typename Base::VertexPoint;
    using Sigma = typename Base::Sigma;
    using NodesAccess = typename Base::NodesAccess;
    using ExtendedNodesAccess = typename Base::ExtendedNodesAccess;
    using InternalNodes = VoronoiNodes<NodeScalar, Dim, Index>;
    using ShiftMask = std::vector<std::uint8_t>;

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;

    struct NodeEraseResult {
        std::vector<Index> requested_visible_internal;
        std::vector<Index> removed_internal;
    };

    struct AppendReport {
        std::size_t appended_nodes = 0;
        std::size_t accepted_vertices = 0;
        std::size_t duplicate_vertices = 0;
        std::size_t rejected_invalid_vertices = 0;
        std::size_t rejected_periodic_vertices = 0;
    };

private:
    /** Visible-node facade onto stable internal coordinate storage. */
    class PublicVisibleNodes final
        : public StoredNodeAccess<NodeScalar, Dim, Index> {
    public:
        using Scalar = NodeScalar;
        static constexpr int DimensionAtCompileTime = Dim;
        static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;
        using AccessBase = StoredNodeAccess<NodeScalar, Dim, Index>;

        PublicVisibleNodes(
            const HighVoronoiMesh& owner,
            Index runtime_dimension)
            : AccessBase(runtime_dimension), owner_(owner) {}

        [[nodiscard]] Index size() const noexcept override {
            return owner_.visible_public_count();
        }

        [[nodiscard]] Scalar get_data(
            Index public_node,
            Index coordinate) const override {
            return owner_.internal_nodes_.get_data(
                owner_.index_mapping().public_to_internal(public_node),
                coordinate);
        }

    private:
        [[nodiscard]] const Scalar*
        get_stored_node_pointer(Index public_node) const override {
            const Index internal =
                owner_.index_mapping().public_to_internal(public_node);
            return owner_.internal_nodes_.stable_node_data(internal);
        }

        const HighVoronoiMesh& owner_;
    };

    using ExtendedNodes = ExtendedVoronoiNodes<PublicVisibleNodes>;

public:
    /**
     * @brief Construct an empty mesh and construct the database in place.
     *
     * `internal_boundary()` initially equals `boundary()` exactly.
     */
    template <class... DatabaseArgs>
    HighVoronoiMesh(
        Index runtime_dimension,
        BoundaryType external_boundary,
        std::in_place_t,
        DatabaseArgs&&... database_args)
        : Base(runtime_dimension, IndexMapping{}),
          internal_nodes_(Index{0}, runtime_dimension),
          visible_nodes_(*this, runtime_dimension),
          external_nodes_(std::in_place, visible_nodes_, external_boundary),
          internal_boundary_(std::move(external_boundary)),
          database_(std::forward<DatabaseArgs>(database_args)...) {
        validate_boundary_dimension(external_nodes_->boundary());
        validate_boundary_dimension(internal_boundary_);
    }

    /** Convenience constructor for HVDataBase/HybridDataBase-like backends. */
    template <
        class D = Database,
        std::enable_if_t<
            std::is_constructible_v<
                D,
                std::size_t,
                const typename D::Parameters&>,
            int> = 0>
    HighVoronoiMesh(
        Index runtime_dimension,
        BoundaryType external_boundary,
        std::size_t database_unit_length,
        const typename D::Parameters& parameters)
        : HighVoronoiMesh(
              runtime_dimension,
              std::move(external_boundary),
              std::in_place,
              database_unit_length,
              parameters) {}

    HighVoronoiMesh(const HighVoronoiMesh&) = delete;
    HighVoronoiMesh& operator=(const HighVoronoiMesh&) = delete;
    HighVoronoiMesh(HighVoronoiMesh&&) = delete;
    HighVoronoiMesh& operator=(HighVoronoiMesh&&) = delete;

    /**
     * @brief Public finite-vertex range with periodic HighVoronoi projection.
     *
     * The range reads only the address list(s) of the canonical visible cell.
     * Invisible generator indices occurring in a stored internal signature are
     * projected to the public index of their visible reference. The stored
     * vertex position is translated through periodic partner shifts until it
     * lies in the external visible domain. No vertex from an invisible copy's
     * own address lists is added to the visible cell.
     */
    template <class AddressSource>
    class PublicVertexRange {
    public:
        using VertexRecord = typename Base::VertexRecord;

        class Iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = VertexRecord;
            using difference_type = std::ptrdiff_t;
            using pointer = const VertexRecord*;
            using reference = const VertexRecord&;

            [[nodiscard]] reference operator*() const noexcept { return current_; }
            [[nodiscard]] pointer operator->() const noexcept { return &current_; }

            Iterator& operator++() { load_next(); return *this; }
            Iterator operator++(int) { Iterator old(*this); ++(*this); return old; }

            friend bool operator==(const Iterator& a, const Iterator& b) noexcept {
                if (a.at_end_ && b.at_end_) return true;
                return std::addressof(a.owner_) == std::addressof(b.owner_) &&
                       a.position_ == b.position_ && a.at_end_ == b.at_end_;
            }
            friend bool operator!=(const Iterator& a, const Iterator& b) noexcept {
                return !(a == b);
            }

        private:
            friend class PublicVertexRange;

            Iterator(
                const HighVoronoiMesh& owner,
                AddressSource addresses,
                std::size_t position,
                bool at_end)
                : owner_(owner),
                  addresses_(std::forward<AddressSource>(addresses)),
                  position_(position),
                  at_end_(at_end),
                  current_{{}, owner.make_vertex_point(), Address{0}} {
                if (!at_end_) load_next();
            }

            void load_next() {
                while (position_ < addresses_.size()) {
                    const Address address = addresses_[position_++];
                    internal_sigma_.clear();
                    owner_.database_.read(
                        address, current_.position, internal_sigma_);
                    if (internal_sigma_.empty()) continue;

                    current_.sigma.clear();
                    if (!owner_.make_public_vertex_signature(
                            internal_sigma_, current_.sigma)) {
                        continue;
                    }
                    owner_.translate_vertex_to_external_domain(current_.position);
                    current_.address = address;
                    at_end_ = false;
                    return;
                }
                at_end_ = true;
            }

            const HighVoronoiMesh& owner_;
            AddressSource addresses_;
            std::size_t position_ = 0;
            bool at_end_ = true;
            Sigma internal_sigma_;
            VertexRecord current_;
        };

        [[nodiscard]] Iterator begin() const {
            return Iterator(owner_, addresses_, 0, false);
        }
        [[nodiscard]] Iterator end() const {
            return Iterator(owner_, addresses_, addresses_.size(), true);
        }
        [[nodiscard]] bool empty() const { return begin() == end(); }

    private:
        friend class HighVoronoiMesh;
        PublicVertexRange(const HighVoronoiMesh& owner, AddressSource addresses)
            : owner_(owner), addresses_(std::forward<AddressSource>(addresses)) {}

        const HighVoronoiMesh& owner_;
        AddressSource addresses_;
    };

    using PublicPrimaryVertexRange = PublicVertexRange<const AddressList&>;
    using PublicCombinedAddressSource = typename Base::template CombinedAddressList<AddressList>;
    using PublicCombinedVertexRange = PublicVertexRange<PublicCombinedAddressSource>;

    [[nodiscard]] PublicPrimaryVertexRange primary_vertices(Index public_node) const {
        const Index internal = visible_public_to_internal(public_node);
        return PublicPrimaryVertexRange(
            *this, primary_address_lists_.at(static_cast<std::size_t>(internal)));
    }

    [[nodiscard]] PublicPrimaryVertexRange secondary_vertices(Index public_node) const {
        const Index internal = visible_public_to_internal(public_node);
        return PublicPrimaryVertexRange(
            *this, secondary_address_lists_.at(static_cast<std::size_t>(internal)));
    }

    [[nodiscard]] PublicCombinedVertexRange vertices(Index public_node) const {
        const Index internal = visible_public_to_internal(public_node);
        return PublicCombinedVertexRange(
            *this,
            PublicCombinedAddressSource(
                primary_address_lists_.at(static_cast<std::size_t>(internal)),
                secondary_address_lists_.at(static_cast<std::size_t>(internal))));
    }

    template <class Function>
    void for_each_vertex(Index public_node, Function&& function) const {
        for (const auto& vertex : vertices(public_node)) {
            function(vertex);
        }
    }

    // ------------------------------------------------------------------
    // Boundaries and database
    // ------------------------------------------------------------------

    [[nodiscard]] const BoundaryType& external_boundary() const noexcept {
        return external_nodes_->boundary();
    }

    [[nodiscard]] const BoundaryType& internal_boundary() const noexcept {
        return internal_boundary_;
    }

    [[nodiscard]] Database& database() noexcept { return database_; }
    [[nodiscard]] const Database& database() const noexcept { return database_; }

    [[nodiscard]] ExtendedNodes& concrete_extended_nodes() noexcept {
        return *external_nodes_;
    }

    [[nodiscard]] const ExtendedNodes& concrete_extended_nodes() const noexcept {
        return *external_nodes_;
    }

    /**
     * @brief Return an AbstractMesh view of the complete active mesh state.
     *
     * Unlike the public HighVoronoiMesh view, this view includes invisible
     * periodic reference nodes. Unlike HighVoronoiComputeMesh, it never
     * reorders nodes for a reconstruction pass: active stable internal nodes
     * are exposed in their original insertion order with deleted nodes simply
     * omitted. The view owns no geometry or database.
     */
    [[nodiscard]] HighVoronoiDataMeshView<HighVoronoiMesh> data_mesh();

    // HighVoronoi's public representation is deliberately read-oriented.
    // Persistent vertex mutation must pass through HighVoronoiComputeMesh so
    // the compute database wrapper can collect refinement/periodic events.
    template <class... Args>
    [[nodiscard]] Address store_vertex(Args&&...) {
        throw std::logic_error(
            "Store HighVoronoi vertices through ComputeHighVoronoi/HighVoronoiComputeMesh.");
    }

    template <class... Args>
    [[nodiscard]] Address store_infinite_edge(Args&&...) {
        throw std::logic_error(
            "Store HighVoronoi infinite edges through the dedicated compute representation.");
    }

    template <class... Args>
    [[nodiscard]] bool erase_vertex(Args&&...) {
        throw std::logic_error(
            "Erase HighVoronoi vertices through HighVoronoi's refinement/removal API.");
    }

    template <class... Args>
    std::size_t erase_nodes_if(Args&&...) {
        throw std::logic_error(
            "Delete HighVoronoi nodes through erase_visible_nodes().");
    }

    template <class... Args>
    std::size_t erase_vertices_if(Args&&...) {
        throw std::logic_error(
            "Erase HighVoronoi vertices through ComputeHighVoronoi.");
    }

    template <class... Args>
    auto filter(Args&&...) {
        throw std::logic_error(
            "Generic AbstractMesh filtering is disabled for HighVoronoiMesh.");
    }

    // ------------------------------------------------------------------
    // Stable node state
    // ------------------------------------------------------------------

    [[nodiscard]] const InternalNodes& internal_nodes() const noexcept {
        return internal_nodes_;
    }

    [[nodiscard]] Index internal_node_count() const noexcept {
        return internal_nodes_.size();
    }

    [[nodiscard]] Index visible_public_count() const noexcept {
        return this->index_mapping().size();
    }

    [[nodiscard]] bool is_active_internal(Index internal_node) const {
        require_internal_node(internal_node);
        return active_.test(static_cast<std::size_t>(internal_node));
    }

    [[nodiscard]] bool is_visible_internal(Index internal_node) const {
        require_internal_node(internal_node);
        return visible_.test(static_cast<std::size_t>(internal_node));
    }

    [[nodiscard]] bool is_reference_internal(Index internal_node) const {
        require_internal_node(internal_node);
        return references_[static_cast<std::size_t>(internal_node)].has_value();
    }

    [[nodiscard]] std::optional<Index>
    reference_internal(Index internal_node) const {
        require_internal_node(internal_node);
        return references_[static_cast<std::size_t>(internal_node)];
    }

    [[nodiscard]] const ShiftMask&
    reference_shift(Index internal_node) const {
        require_internal_node(internal_node);
        return reference_shifts_[static_cast<std::size_t>(internal_node)];
    }

    [[nodiscard]] Index visible_public_to_internal(Index public_node) const {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    internal_to_visible_public(Index internal_node) const {
        require_internal_node(internal_node);
        return this->index_mapping().internal_to_public(internal_node);
    }

    /**
     * @brief Project an active internal ordinary node onto visible numbering.
     *
     * Visible nodes map to themselves. Invisible reference copies map to the
     * same public index as their visible reference. Deleted nodes have no
     * projection. This is the public HighVoronoi signature semantics used by
     * AbstractMesh vertex iterators; the separate compute mapping remains
     * bijective over all active internal nodes.
     */
    [[nodiscard]] std::optional<Index>
    projected_visible_public(Index internal_node) const {
        require_internal_node(internal_node);
        if (!is_active_internal(internal_node)) {
            return std::nullopt;
        }
        return this->index_mapping().internal_to_public(internal_node);
    }

    [[nodiscard]] Index integrated_node_count() const noexcept {
        return integrated_node_count_;
    }

    [[nodiscard]] const std::vector<Index>&
    pending_deleted_internal_nodes() const noexcept {
        return pending_deleted_internal_nodes_;
    }

    /** Commit the current active-node state after a successful reconstruction. */
    void mark_integrated() {
        integrated_node_count_ = internal_node_count();
        pending_deleted_internal_nodes_.clear();
    }

    /** Return active visible nodes appended since the last successful compute. */
    [[nodiscard]] std::vector<Index> pending_new_visible_internal_nodes() const {
        std::vector<Index> result;
        for (Index internal = integrated_node_count_;
             internal < internal_node_count();
             ++internal) {
            if (is_active_internal(internal) && is_visible_internal(internal)) {
                result.push_back(internal);
            }
        }
        return result;
    }

    /** Return every active internal node appended since the last full compute. */
    [[nodiscard]] std::vector<Index> pending_new_internal_nodes() const {
        std::vector<Index> result;
        for (Index internal = integrated_node_count_;
             internal < internal_node_count();
             ++internal) {
            if (is_active_internal(internal)) {
                result.push_back(internal);
            }
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Node insertion/removal
    // ------------------------------------------------------------------

    template <class PointLike>
    [[nodiscard]] Index append_visible_node(
        const PointLike& point,
        NodeScalar boundary_tolerance = NodeScalar{0}) {
        require_point_dimension(point);
        if (!external_boundary().contains(point, boundary_tolerance)) {
            throw std::domain_error(
                "Visible HighVoronoi node lies outside the external boundary.");
        }
        return append_internal_node(
            point,
            true,
            std::nullopt,
            ShiftMask(external_boundary().size(), std::uint8_t{0}));
    }

    template <class NodesLike>
    [[nodiscard]] std::vector<Index> append_visible_nodes(
        const NodesLike& nodes,
        NodeScalar boundary_tolerance = NodeScalar{0}) {
        if (nodes.dimension() != this->dimension()) {
            throw std::invalid_argument(
                "Appended node container has the wrong dimension.");
        }
        std::vector<Index> result;
        result.reserve(static_cast<std::size_t>(nodes.size()));
        NodePoint point = make_node_point();
        for (Index i = Index{0}; i < static_cast<Index>(nodes.size()); ++i) {
            for (Index coordinate = Index{0};
                 coordinate < this->dimension();
                 ++coordinate) {
                point[static_cast<Eigen::Index>(coordinate)] =
                    static_cast<NodeScalar>(nodes.get_data(i, coordinate));
            }
            result.push_back(
                append_visible_node(point, boundary_tolerance));
        }
        return result;
    }

    /**
     * @brief Append one invisible periodic copy of a visible node.
     *
     * `reference_public` uses the current visible public numbering. The point is
     * calculated, not supplied by the caller, so reference and shift cannot
     * disagree.
     */
    [[nodiscard]] Index append_reference_node(
        Index reference_public,
        const ShiftMask& shift_mask) {
        const Index reference = visible_public_to_internal(reference_public);
        return append_reference_node_internal(reference, shift_mask);
    }

    /** Same operation using the stable internal visible reference index. */
    [[nodiscard]] Index append_reference_node_internal(
        Index visible_reference_internal,
        const ShiftMask& shift_mask) {
        require_internal_node(visible_reference_internal);
        if (!is_active_internal(visible_reference_internal) ||
            !is_visible_internal(visible_reference_internal)) {
            throw std::invalid_argument(
                "A periodic reference node must refer to an active visible node.");
        }
        validate_shift_mask(shift_mask);
        if (std::none_of(
                shift_mask.begin(),
                shift_mask.end(),
                [](std::uint8_t value) { return value != 0; })) {
            throw std::invalid_argument(
                "An invisible periodic reference requires a non-empty shift.");
        }
        if (const auto existing = find_reference_copy(
                visible_reference_internal,
                shift_mask)) {
            return *existing;
        }

        NodePoint point = make_node_point();
        copy_internal_node(visible_reference_internal, point);
        point += periodic_shift(shift_mask);
        return append_internal_node(
            point,
            false,
            visible_reference_internal,
            shift_mask);
    }

    /** Find an active invisible copy with the exact reference and shift mask. */
    [[nodiscard]] std::optional<Index> find_reference_copy(
        Index visible_reference_internal,
        const ShiftMask& shift_mask) const {
        for (Index internal = Index{0};
             internal < internal_node_count();
             ++internal) {
            if (!is_active_internal(internal) || is_visible_internal(internal)) {
                continue;
            }
            const auto& reference = references_[static_cast<std::size_t>(internal)];
            if (reference && *reference == visible_reference_internal &&
                reference_shifts_[static_cast<std::size_t>(internal)] == shift_mask) {
                return internal;
            }
        }
        return std::nullopt;
    }

    /**
     * @brief Delete visible nodes and every invisible copy referring to them.
     *
     * The input is interpreted completely in the pre-deletion visible public
     * numbering before any mapping is rebuilt. Vertex repair is intentionally
     * deferred to ComputeHighVoronoi; deleted internal indices are recorded so
     * that reconstruction can remove every incident/stale vertex.
     */
    [[nodiscard]] NodeEraseResult erase_visible_nodes(
        const std::vector<Index>& visible_public_nodes) {
        NodeEraseResult result;
        result.requested_visible_internal.reserve(visible_public_nodes.size());

        std::vector<std::uint8_t> selected(
            static_cast<std::size_t>(internal_node_count()),
            std::uint8_t{0});

        for (const Index public_node : visible_public_nodes) {
            const Index internal = visible_public_to_internal(public_node);
            if (selected[static_cast<std::size_t>(internal)] == 0) {
                selected[static_cast<std::size_t>(internal)] = 1;
                result.requested_visible_internal.push_back(internal);
            }
        }

        for (Index internal = Index{0};
             internal < internal_node_count();
             ++internal) {
            const auto& reference = references_[static_cast<std::size_t>(internal)];
            if (reference &&
                selected[static_cast<std::size_t>(*reference)] != 0) {
                selected[static_cast<std::size_t>(internal)] = 1;
            }
        }

        for (Index internal = Index{0};
             internal < internal_node_count();
             ++internal) {
            if (selected[static_cast<std::size_t>(internal)] == 0 ||
                !is_active_internal(internal)) {
                continue;
            }
            active_.reset(static_cast<std::size_t>(internal));
            result.removed_internal.push_back(internal);
            pending_deleted_internal_nodes_.push_back(internal);
        }

        rebuild_visible_numbering();
        reset_external_nodes();
        return result;
    }

    // ------------------------------------------------------------------
    // Periodic shift/boundary helpers
    // ------------------------------------------------------------------

    [[nodiscard]] NodePoint periodic_shift(const ShiftMask& mask) const {
        validate_shift_mask(mask);
        NodePoint result = make_node_point();
        result.setZero();
        for (Index plane = Index{0}; plane < external_boundary().size(); ++plane) {
            if (mask[static_cast<std::size_t>(plane)] == 0) {
                continue;
            }
            result += external_boundary().periodic_shift(plane);
        }
        return result;
    }

    [[nodiscard]] bool touches_periodic_external_boundary(
        const VertexPoint& point,
        VertexScalar tolerance) const {
        for (Index plane = Index{0}; plane < external_boundary().size(); ++plane) {
            const auto& p = external_boundary()[plane];
            if (!p.is_periodic()) {
                continue;
            }
            const auto distance = static_cast<VertexScalar>(
                std::abs(p.halfspace_value(point)));
            if (distance <= tolerance) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Expand periodic internal planes with a reference-node safety margin.
     *
     * A newly generated periodic reference node must not merely be enclosed by
     * the internal boundary. Its distance to the corresponding internal
     * periodic plane must be at least the distance by which that reference node
     * lies outside the corresponding visible/external plane. For a periodic
     * image this is exactly the distance of the original visible node from the
     * opposite visible periodic plane.
     *
     * Thus, for every candidate point and periodic plane,
     *
     *     required clearance
     *         = max(0, external_plane.halfspace_value(point))
     *
     * and the internal plane is moved only if its current clearance is smaller
     * than this value. `tolerance` is added as a small numerical safety buffer.
     *
     * Non-periodic planes are never moved. The return value contains exactly
     * the zero-based plane indices whose base point changed. Plane ordering and
     * periodic partner indices remain unchanged.
     */
    template <class PointRange>
    [[nodiscard]] std::vector<Index> expand_internal_boundary_to_include(
        const PointRange& points,
        NodeScalar tolerance = NodeScalar{0}) {
        using PlaneType = typename BoundaryType::PlaneType;
        std::vector<PlaneType> planes(
            internal_boundary_.planes().begin(),
            internal_boundary_.planes().end());
        std::vector<Index> modified;

        for (Index plane_index = Index{0};
             plane_index < internal_boundary_.size();
             ++plane_index) {
            PlaneType& plane = planes[static_cast<std::size_t>(plane_index)];
            if (!plane.is_periodic()) {
                continue;
            }

            const auto& external_plane = external_boundary()[plane_index];

            NodeScalar maximum_required_expansion = NodeScalar{0};
            for (const auto& point : points) {
                const NodeScalar internal_violation =
                    static_cast<NodeScalar>(
                        plane.halfspace_value(point));

                const NodeScalar external_violation =
                    static_cast<NodeScalar>(
                        external_plane.halfspace_value(point));

                const NodeScalar required_clearance =
                    std::max(NodeScalar{0}, external_violation);

                const NodeScalar required_expansion =
                    internal_violation + required_clearance;

                if (required_expansion > maximum_required_expansion) {
                    maximum_required_expansion = required_expansion;
                }
            }

            if (maximum_required_expansion <= NodeScalar{0}) {
                continue;
            }

            NodePoint displacement = make_node_point();
            displacement =
                (maximum_required_expansion + tolerance) * plane.normal();
            plane.translate(displacement);
            modified.push_back(plane_index);
        }

        if (!modified.empty()) {
            internal_boundary_ = BoundaryType(
                std::move(planes),
                internal_boundary_.convex());
        }
        return modified;
    }

    [[nodiscard]] static constexpr Index invalid_index() noexcept {
        return detail::highvoronoi_invalid_index<Index>();
    }

    [[nodiscard]] bool is_boundary_internal_index(Index value) const noexcept {
        if (value == invalid_index()) {
            return false;
        }
        const Index count = internal_boundary_.size();
        return count != Index{0} &&
               value >= static_cast<Index>(invalid_index() - count);
    }

    [[nodiscard]] Index decode_boundary_internal_index(Index value) const {
        if (!is_boundary_internal_index(value)) {
            throw std::out_of_range(
                "Index is not a HighVoronoi boundary encoding.");
        }
        return static_cast<Index>(
            invalid_index() - static_cast<Index>(value + Index{1}));
    }

    [[nodiscard]] Index encode_boundary_internal_index(Index plane) const {
        if (plane >= internal_boundary_.size()) {
            throw std::out_of_range("Boundary plane index out of range.");
        }
        return static_cast<Index>(invalid_index() - Index{1} - plane);
    }

    // ------------------------------------------------------------------
    // Internal vertex storage/routing used by reconstruction and imports
    // ------------------------------------------------------------------

    [[nodiscard]] const AddressList&
    primary_addresses_internal(Index internal_node) const {
        require_internal_node(internal_node);
        return primary_address_lists_[static_cast<std::size_t>(internal_node)];
    }

    [[nodiscard]] const AddressList&
    secondary_addresses_internal(Index internal_node) const {
        require_internal_node(internal_node);
        return secondary_address_lists_[static_cast<std::size_t>(internal_node)];
    }

    /** Remove all stale address-list entries owned by one inactive node. */
    void clear_vertex_addresses_internal(Index internal_node) {
        require_internal_node(internal_node);
        auto erase_all = [](Address) { return true; };
        primary_address_lists_[static_cast<std::size_t>(internal_node)]
            .erase_if(erase_all);
        secondary_address_lists_[static_cast<std::size_t>(internal_node)]
            .erase_if(erase_all);
    }

    [[nodiscard]] const AddressList& infinite_addresses_internal() const noexcept {
        return infinite_edge_addresses_;
    }

    void register_primary_internal(Index internal_node, Address address) {
        require_internal_node(internal_node);
        primary_address_lists_[static_cast<std::size_t>(internal_node)]
            .push_back(address);
    }

    void register_secondary_internal(Index internal_node, Address address) {
        require_internal_node(internal_node);
        secondary_address_lists_[static_cast<std::size_t>(internal_node)]
            .push_back(address);
    }

    void register_infinite_internal(Address address) {
        infinite_edge_addresses_.push_back(address);
    }

    template <class PointLike>
    [[nodiscard]] Address store_internal_vertex(
        const PointLike& position,
        Sigma internal_sigma,
        std::optional<Index> primary_owner = std::nullopt) {
        require_point_dimension(position);
        canonicalize_internal_signature(internal_sigma);
        const Index owner = resolve_primary_owner(internal_sigma, primary_owner);
        const Address address = database_.push(position, internal_sigma);
        if (address == Address{0}) {
            return Address{0};
        }
        register_primary_internal(owner, address);
        for (const Index generator : internal_sigma) {
            if (generator >= internal_node_count()) {
                break;
            }
            if (generator != owner) {
                register_secondary_internal(generator, address);
            }
        }
        return address;
    }

    void read_internal_vertex(
        Address address,
        VertexPoint& position,
        Sigma& internal_sigma) const {
        database_.read(address, position, internal_sigma);
    }

    [[nodiscard]] bool erase_internal_vertex(
        Address address,
        const Sigma& internal_sigma) {
        return database_.erase(address, internal_sigma);
    }

    /** Compact all per-node lists by removing database tombstones. */
    void compact_vertex_address_lists() {
        std::unordered_set<Address> addresses;
        collect_addresses(primary_address_lists_, addresses);
        collect_addresses(secondary_address_lists_, addresses);

        std::unordered_set<Address> active_addresses;
        active_addresses.reserve(addresses.size());
        VertexPoint point = make_vertex_point();
        Sigma sigma;
        for (const Address address : addresses) {
            sigma.clear();
            database_.read(address, point, sigma);
            if (!sigma.empty()) {
                active_addresses.insert(address);
            }
        }

        compact_lists(primary_address_lists_, active_addresses);
        compact_lists(secondary_address_lists_, active_addresses);
    }

    /**
     * @brief Append all visible nodes and admissible finite vertices of a source mesh.
     *
     * Source vertices are read through their source-public primary iteration.
     * Invisible/reference nodes of another HighVoronoiMesh are deliberately not
     * imported yet. Boundary generators are accepted only if their source plane
     * geometrically matches a non-periodic external plane. A source vertex on a
     * periodic external plane is dropped. A source vertex outside the external
     * boundary is considered malformed input and throws.
     */
    template <class SourceMesh>
    [[nodiscard]] AppendReport append_mesh(
        const SourceMesh& source,
        VertexScalar nearest_tolerance = VertexScalar{1e-10},
        NodeScalar boundary_tolerance = NodeScalar{1e-10});

    /**
     * @brief Append one concrete compute engine.
     *
     * Nodes are materialized in stable HighVoronoi storage. Invalid, duplicate,
     * or periodic-boundary engine vertices are tombstoned before attachment. If
     * Database exposes `register_engine(shared_ptr<Engine>, node_offset)`, the
     * surviving engine remains attached through that backend; otherwise its
     * surviving finite vertices are materialized as ordinary stored records.
     */
    template <class Engine>
    [[nodiscard]] AppendReport append_engine(
        Engine&& engine,
        VertexScalar nearest_tolerance = VertexScalar{1e-10},
        NodeScalar boundary_tolerance = NodeScalar{1e-10});

private:
    template <class, class>
    friend class HighVoronoiComputeMesh;

    template <class, class>
    friend class HighVoronoiComputeDataBase;

    template <class, class>
    friend struct IncrementalVoronoiBackend;

    template <class>
    friend class HighVoronoiDataMeshView;

    template <class>
    friend class VisibleFirstMesh;

    // AbstractMesh deliberately remains visible-only. Refinement uses the
    // separate HighVoronoiComputeMesh, where all active nodes are one-to-one.
    [[nodiscard]] const NodesAccess& nodes_impl() const noexcept override {
        return external_nodes_->inner_nodes();
    }

    [[nodiscard]] ExtendedNodesAccess& extended_nodes_impl() noexcept override {
        return *external_nodes_;
    }

    [[nodiscard]] const ExtendedNodesAccess&
    extended_nodes_impl() const noexcept override {
        return *external_nodes_;
    }

    [[nodiscard]] const BoundaryType& boundary_impl() const noexcept override {
        return external_nodes_->boundary();
    }

    void set_boundary_impl(BoundaryType boundary) override {
        if (internal_node_count() != Index{0} || has_active_vertex_records()) {
            throw std::logic_error(
                "HighVoronoi external boundary may only be replaced while the mesh is empty.");
        }
        validate_boundary_dimension(boundary);
        internal_boundary_ = boundary;
        external_nodes_.emplace(visible_nodes_, std::move(boundary));
    }

    [[nodiscard]] Database& database_impl() noexcept override { return database_; }
    [[nodiscard]] const Database& database_impl() const noexcept override {
        return database_;
    }

    [[nodiscard]] Index internal_node_count_impl() const noexcept override {
        return internal_node_count();
    }

    [[nodiscard]] Index
    public_node_to_internal_impl(Index public_node) const override {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    internal_node_to_public_impl(Index internal_node) const override {
        return this->index_mapping().internal_to_public(internal_node);
    }

    [[nodiscard]] const AddressList&
    primary_vertex_addresses_impl(Index internal_node) const override {
        return primary_addresses_internal(internal_node);
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return secondary_addresses_internal(internal_node);
    }

    void register_primary_vertex_impl(Index internal_node, Address address) override {
        register_primary_internal(internal_node, address);
    }

    void register_secondary_vertex_impl(Index internal_node, Address address) override {
        register_secondary_internal(internal_node, address);
    }

    [[nodiscard]] const AddressList&
    infinite_edge_addresses_impl() const override {
        return infinite_edge_addresses_;
    }

    void register_infinite_edge_impl(Address address) override {
        register_infinite_internal(address);
    }

    void mark_internal_node_deleted_impl(Index internal_node) override {
        // This hook exists for AbstractMesh compatibility. Public HighVoronoi
        // node deletion should use erase_visible_nodes(), which expands a
        // visible deletion to all periodic reference copies before mutation.
        require_internal_node(internal_node);
        if (!active_.test(static_cast<std::size_t>(internal_node))) {
            return;
        }
        active_.reset(static_cast<std::size_t>(internal_node));
        pending_deleted_internal_nodes_.push_back(internal_node);
        numbering_dirty_ = true;
    }

    void cleanup_vertex_lists_impl() override {
        if (numbering_dirty_) {
            rebuild_visible_numbering();
            reset_external_nodes();
            numbering_dirty_ = false;
        }
        compact_vertex_address_lists();
    }

    template <class PointLike>
    [[nodiscard]] Index append_internal_node(
        const PointLike& point,
        bool visible,
        std::optional<Index> reference,
        ShiftMask shift_mask) {
        validate_internal_capacity(static_cast<Index>(internal_node_count() + Index{1}));
        const Index internal = internal_node_count();
        internal_nodes_.resize(static_cast<Index>(internal + Index{1}));
        internal_nodes_.set(internal, point);
        active_.push_back(true);
        visible_.push_back(visible);
        references_.push_back(reference);
        reference_shifts_.push_back(std::move(shift_mask));
        primary_address_lists_.emplace_back();
        secondary_address_lists_.emplace_back();

        if (visible) {
            (void)this->index_mapping().append_visible(internal);
        } else {
            if (!reference) {
                this->index_mapping().append_hidden(internal);
            } else {
                const auto public_reference =
                    this->index_mapping().internal_to_public(*reference);
                if (!public_reference) {
                    throw std::logic_error(
                        "Invisible HighVoronoi node refers to a non-public reference.");
                }
                this->index_mapping().append_alias(internal, *public_reference);
            }
        }
        return internal;
    }

    void mark_internal_inactive_from_computation(Index internal_node) {
        require_internal_node(internal_node);
        if (!is_active_internal(internal_node)) {
            return;
        }
        active_.reset(static_cast<std::size_t>(internal_node));
        pending_deleted_internal_nodes_.push_back(internal_node);
        rebuild_visible_numbering();
        reset_external_nodes();
    }

    void rebuild_visible_numbering() {
        std::vector<Index> public_to_internal;
        public_to_internal.reserve(static_cast<std::size_t>(internal_node_count()));
        std::vector<Index> internal_to_public(
            static_cast<std::size_t>(internal_node_count()),
            IndexMapping::invalid_index());

        // First assign dense public indices to active visible originals.
        for (Index internal = Index{0};
             internal < internal_node_count();
             ++internal) {
            if (!is_active_internal(internal) || !is_visible_internal(internal)) {
                continue;
            }
            const Index public_index =
                static_cast<Index>(public_to_internal.size());
            public_to_internal.push_back(internal);
            internal_to_public[static_cast<std::size_t>(internal)] = public_index;
        }

        // Then project every active invisible copy onto its visible reference.
        for (Index internal = Index{0};
             internal < internal_node_count();
             ++internal) {
            if (!is_active_internal(internal) || is_visible_internal(internal)) {
                continue;
            }
            const auto& reference = references_[static_cast<std::size_t>(internal)];
            if (!reference || *reference >= internal_node_count()) {
                continue;
            }
            const Index public_index =
                internal_to_public[static_cast<std::size_t>(*reference)];
            if (public_index != IndexMapping::invalid_index()) {
                internal_to_public[static_cast<std::size_t>(internal)] =
                    public_index;
            }
        }

        this->index_mapping().assign(
            std::move(public_to_internal),
            std::move(internal_to_public));
    }

    void reset_external_nodes() {
        BoundaryType boundary_copy = external_nodes_->boundary();
        external_nodes_.emplace(visible_nodes_, std::move(boundary_copy));
    }

    [[nodiscard]] NodePoint make_node_point() const {
        if constexpr (Dim == Dynamic) {
            return NodePoint(static_cast<Eigen::Index>(this->dimension()));
        } else {
            return NodePoint{};
        }
    }

    [[nodiscard]] VertexPoint make_vertex_point() const {
        if constexpr (Dim == Dynamic) {
            return VertexPoint(static_cast<Eigen::Index>(this->dimension()));
        } else {
            return VertexPoint{};
        }
    }

    void copy_internal_node(Index internal_node, NodePoint& target) const {
        require_internal_node(internal_node);
        for (Index coordinate = Index{0};
             coordinate < this->dimension();
             ++coordinate) {
            target[static_cast<Eigen::Index>(coordinate)] =
                internal_nodes_.get_data(internal_node, coordinate);
        }
    }

    template <class PointLike>
    void require_point_dimension(const PointLike& point) const {
        if (static_cast<Index>(point.size()) != this->dimension()) {
            throw std::invalid_argument(
                "Point dimension does not match HighVoronoiMesh dimension.");
        }
    }

    void require_internal_node(Index internal_node) const {
        if (internal_node >= internal_node_count()) {
            throw std::out_of_range("HighVoronoi internal node index out of range.");
        }
    }

    void validate_boundary_dimension(const BoundaryType& boundary) const {
        if (!boundary.empty() &&
            static_cast<Index>(boundary.dimension()) != this->dimension()) {
            throw std::invalid_argument(
                "Boundary dimension does not match HighVoronoiMesh dimension.");
        }
    }

    void validate_internal_capacity(Index count) const {
        const Index reserved = static_cast<Index>(internal_boundary_.size() + Index{1});
        if (count > static_cast<Index>(invalid_index() - reserved)) {
            throw std::length_error(
                "HighVoronoi ordinary-node indices collide with boundary encoding.");
        }
    }

    void validate_shift_mask(const ShiftMask& mask) const {
        if (mask.size() != static_cast<std::size_t>(external_boundary().size())) {
            throw std::invalid_argument(
                "Periodic reference shift mask has the wrong boundary length.");
        }
        for (Index plane = Index{0}; plane < external_boundary().size(); ++plane) {
            if (mask[static_cast<std::size_t>(plane)] != 0 &&
                !external_boundary()[plane].is_periodic()) {
                throw std::invalid_argument(
                    "Periodic shift mask selects a non-periodic boundary plane.");
            }
        }
    }

    [[nodiscard]] bool make_public_vertex_signature(
        const Sigma& internal_sigma,
        Sigma& public_sigma) const {
        public_sigma.clear();
        public_sigma.reserve(internal_sigma.size());

        for (const Index internal : internal_sigma) {
            if (internal < internal_node_count()) {
                if (!is_active_internal(internal)) {
                    public_sigma.clear();
                    return false;
                }
                const auto projected = this->index_mapping().internal_to_public(internal);
                if (!projected) {
                    public_sigma.clear();
                    return false;
                }
                public_sigma.push_back(*projected);
                continue;
            }

            if (!is_boundary_internal_index(internal)) {
                throw std::logic_error(
                    "Stored HighVoronoi vertex contains an invalid internal generator.");
            }
            const Index plane = decode_boundary_internal_index(internal);
            if (external_boundary()[plane].is_periodic()) {
                throw std::logic_error(
                    "Periodic boundary vertices must not persist in HighVoronoi storage.");
            }
            public_sigma.push_back(static_cast<Index>(visible_public_count() + plane));
        }

        std::sort(public_sigma.begin(), public_sigma.end());
        return true;
    }

    void translate_vertex_to_external_domain(VertexPoint& point) const {
        const VertexScalar tolerance =
            VertexScalar{64} * std::numeric_limits<VertexScalar>::epsilon();
        const std::size_t maximum_sweeps =
            std::max<std::size_t>(std::size_t{1},
                                  static_cast<std::size_t>(external_boundary().size()) + 1);

        for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
            bool changed = false;
            for (Index plane = Index{0}; plane < external_boundary().size(); ++plane) {
                const auto& boundary_plane = external_boundary()[plane];
                if (static_cast<VertexScalar>(boundary_plane.halfspace_value(point)) <= tolerance) {
                    continue;
                }
                if (!boundary_plane.is_periodic()) {
                    throw std::logic_error(
                        "Stored HighVoronoi vertex lies outside a non-periodic external boundary.");
                }
                point += external_boundary().periodic_shift(plane)
                             .template cast<VertexScalar>();
                changed = true;
            }
            if (!changed) return;
        }

        if (!external_boundary().contains(point, static_cast<NodeScalar>(tolerance))) {
            throw std::logic_error(
                "Could not translate HighVoronoi vertex into the external periodic domain.");
        }
    }

    static void canonicalize_internal_signature(Sigma& sigma) {
        if (sigma.empty()) {
            throw std::invalid_argument("Voronoi vertex signature must not be empty.");
        }
        std::sort(sigma.begin(), sigma.end());
        if (std::adjacent_find(sigma.begin(), sigma.end()) != sigma.end()) {
            throw std::invalid_argument(
                "Voronoi vertex signature contains duplicate internal indices.");
        }
    }

    [[nodiscard]] Index resolve_primary_owner(
        const Sigma& sigma,
        std::optional<Index> requested) const {
        if (requested) {
            if (*requested >= internal_node_count() ||
                !std::binary_search(sigma.begin(), sigma.end(), *requested)) {
                throw std::invalid_argument(
                    "Requested primary owner is not an ordinary generator of the vertex.");
            }
            return *requested;
        }
        for (const Index generator : sigma) {
            if (generator < internal_node_count()) {
                return generator;
            }
            break;
        }
        throw std::invalid_argument(
            "Stored HighVoronoi vertex requires at least one ordinary generator.");
    }

    [[nodiscard]] bool has_active_vertex_records() const {
        VertexPoint point = make_vertex_point();
        Sigma sigma;
        for (const auto& addresses : primary_address_lists_) {
            const std::size_t count = addresses.size();
            for (std::size_t i = 0; i < count; ++i) {
                sigma.clear();
                database_.read(addresses[i], point, sigma);
                if (!sigma.empty()) {
                    return true;
                }
            }
        }
        return false;
    }

    template <class ListFamily>
    static void collect_addresses(
        const ListFamily& lists,
        std::unordered_set<Address>& target) {
        for (const auto& list : lists) {
            const std::size_t count = list.size();
            for (std::size_t i = 0; i < count; ++i) {
                target.insert(list[i]);
            }
        }
    }

    template <class ListFamily>
    static void compact_lists(
        ListFamily& lists,
        const std::unordered_set<Address>& active_addresses) {
        for (auto& list : lists) {
            list.erase_if([&active_addresses](Address address) {
                return active_addresses.find(address) == active_addresses.end();
            });
        }
    }

    InternalNodes internal_nodes_;
    detail::BitVector active_;
    detail::BitVector visible_;
    std::vector<std::optional<Index>> references_;
    std::vector<ShiftMask> reference_shifts_;

    std::deque<AddressList> primary_address_lists_;
    std::deque<AddressList> secondary_address_lists_;
    AddressList infinite_edge_addresses_;

    PublicVisibleNodes visible_nodes_;
    std::optional<ExtendedNodes> external_nodes_;
    BoundaryType internal_boundary_;
    Database database_;

    Index integrated_node_count_ = Index{0};
    std::vector<Index> pending_deleted_internal_nodes_;
    bool numbering_dirty_ = false;
};

/**
 * @brief Stable-order AbstractMesh view of every currently active internal node.
 *
 * The data view is bijective and exists for diagnostics, import validation and
 * algorithms that need the complete current internal tessellation without a
 * compute-specific front permutation. It owns neither coordinates, address
 * lists nor the database.
 */
template <class HighMeshT>
class HighVoronoiDataMeshView final
    : public AbstractMesh<
          typename HighMeshT::NodeScalar,
          typename HighMeshT::VertexScalar,
          typename HighMeshT::Index,
          HighMeshT::DimensionAtCompileTime,
          NodeAccessMode::Stored,
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
    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;

    using IndexMapping = DenseIndexMapping<Index>;
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

        [[nodiscard]] Scalar get_data(Index public_node, Index coordinate) const override {
            return owner_.internal_nodes_.get_data(
                mapping_.public_to_internal(public_node), coordinate);
        }

    private:
        [[nodiscard]] const Scalar* get_stored_node_pointer(Index public_node) const override {
            return owner_.internal_nodes_.stable_node_data(
                mapping_.public_to_internal(public_node));
        }

        const HighMesh& owner_;
        const IndexMapping& mapping_;
    };

    using ExtendedNodes = ExtendedVoronoiNodes<PublicActiveNodes>;

public:
    explicit HighVoronoiDataMeshView(HighMesh& owner)
        : Base(owner.dimension(), make_mapping(owner)),
          owner_(owner),
          public_nodes_(owner_, this->index_mapping()),
          extended_nodes_(std::in_place, public_nodes_, owner_.internal_boundary_) {}

    [[nodiscard]] Index global_internal_node(Index public_node) const {
        return this->index_mapping().public_to_internal(public_node);
    }

    [[nodiscard]] std::optional<Index>
    public_node_of_global_internal(Index internal_node) const {
        return this->index_mapping().internal_to_public(internal_node);
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
            "HighVoronoiDataMeshView is read-through and cannot store vertices.");
    }

    template <class... Args>
    [[nodiscard]] Address store_infinite_edge(Args&&...) {
        throw std::logic_error(
            "HighVoronoiDataMeshView is read-through and cannot store infinite edges.");
    }

    template <class... Args>
    [[nodiscard]] bool erase_vertex(Args&&...) {
        throw std::logic_error(
            "HighVoronoiDataMeshView is read-through and cannot erase vertices.");
    }

private:
    [[nodiscard]] static IndexMapping make_mapping(const HighMesh& owner) {
        std::vector<Index> public_to_internal;
        public_to_internal.reserve(static_cast<std::size_t>(owner.internal_node_count()));
        std::vector<Index> internal_to_public(
            static_cast<std::size_t>(owner.internal_node_count()),
            IndexMapping::invalid_index());

        for (Index internal = Index{0}; internal < owner.internal_node_count(); ++internal) {
            if (!owner.active_.test(static_cast<std::size_t>(internal))) {
                continue;
            }
            const Index public_index = static_cast<Index>(public_to_internal.size());
            public_to_internal.push_back(internal);
            internal_to_public[static_cast<std::size_t>(internal)] = public_index;
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
        throw std::logic_error("HighVoronoiDataMeshView is a read-through structural view.");
    }

    [[nodiscard]] Database& database_impl() noexcept override { return owner_.database_; }
    [[nodiscard]] const Database& database_impl() const noexcept override { return owner_.database_; }

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
        return owner_.primary_address_lists_.at(static_cast<std::size_t>(internal_node));
    }

    [[nodiscard]] const AddressList&
    secondary_vertex_addresses_impl(Index internal_node) const override {
        return owner_.secondary_address_lists_.at(static_cast<std::size_t>(internal_node));
    }

    void register_primary_vertex_impl(Index, Address) override {
        throw std::logic_error("HighVoronoiDataMeshView does not accept vertex storage.");
    }

    void register_secondary_vertex_impl(Index, Address) override {
        throw std::logic_error("HighVoronoiDataMeshView does not accept vertex storage.");
    }

    [[nodiscard]] const AddressList& infinite_edge_addresses_impl() const override {
        return owner_.infinite_edge_addresses_;
    }

    void register_infinite_edge_impl(Address) override {
        throw std::logic_error("HighVoronoiDataMeshView does not accept infinite-edge storage.");
    }

    void mark_internal_node_deleted_impl(Index) override {
        throw std::logic_error("Delete nodes through HighVoronoiMesh.");
    }

    void cleanup_vertex_lists_impl() override {}

    HighMesh& owner_;
    PublicActiveNodes public_nodes_;
    std::optional<ExtendedNodes> extended_nodes_;
};

template <typename NodeScalarT, int Dim, class DatabaseT>
[[nodiscard]]
HighVoronoiDataMeshView<HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>>
HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>::data_mesh() {
    return HighVoronoiDataMeshView<HighVoronoiMesh>(*this);
}

namespace detail {

template <class HighMesh, class ComputationMesh, class SearchTree>
[[nodiscard]] bool highvoronoi_vertex_survives_nearest_test(
    const HighMesh& owner,
    const ComputationMesh& computation_mesh,
    const SearchTree& tree,
    const typename HighMesh::VertexPoint& position,
    const typename HighMesh::Sigma& internal_sigma,
    typename HighMesh::VertexScalar tolerance) {
    using Index = typename HighMesh::Index;
    using VertexScalar = typename HighMesh::VertexScalar;

    Index first_generator = HighMesh::invalid_index();
    for (const Index generator : internal_sigma) {
        if (generator < owner.internal_node_count()) {
            first_generator = generator;
            break;
        }
    }
    if (first_generator == HighMesh::invalid_index()) {
        throw std::logic_error(
            "HighVoronoi vertex has no ordinary generator.");
    }

    long double radius_squared = 0.0L;
    for (Index coordinate = Index{0}; coordinate < owner.dimension(); ++coordinate) {
        const long double delta =
            static_cast<long double>(position[static_cast<Eigen::Index>(coordinate)]) -
            static_cast<long double>(owner.internal_nodes().get_data(
                first_generator, coordinate));
        radius_squared += delta * delta;
    }
    const long double radius = std::sqrt(radius_squared);

    auto data = tree.make_backend_data(position);
    const auto nearest = tree.nn(
        data,
        [](Index) noexcept { return false; });
    if (!nearest) {
        return false;
    }
    if (nearest->index >= computation_mesh.size()) {
        // No active ordinary candidate was returned. Boundary mirrors are not
        // activated during these consistency queries, so this is defensive.
        return false;
    }
    const Index nearest_internal =
        computation_mesh.global_internal_node(nearest->index);
    if (std::binary_search(
            internal_sigma.begin(),
            internal_sigma.end(),
            nearest_internal)) {
        return true;
    }

    // Equal-distance ties are accepted; only a strictly closer foreign node
    // invalidates the source vertex. This avoids tie-order dependence in
    // degenerate Voronoi configurations.
    return static_cast<long double>(nearest->distance) +
               static_cast<long double>(tolerance) >=
           radius;
}

template <class HighMesh, class SourcePlane>
[[nodiscard]] std::optional<typename HighMesh::Index>
match_external_plane(
    const HighMesh& owner,
    const SourcePlane& source_plane,
    typename HighMesh::NodeScalar tolerance) {
    using Index = typename HighMesh::Index;
    using Scalar = typename HighMesh::NodeScalar;

    for (Index candidate = Index{0};
         candidate < owner.external_boundary().size();
         ++candidate) {
        const auto& target = owner.external_boundary()[candidate];
        long double normal_error_squared = 0.0L;
        long double plane_offset = 0.0L;
        for (Index coordinate = Index{0}; coordinate < owner.dimension(); ++coordinate) {
            const long double sn = static_cast<long double>(
                source_plane.normal()[static_cast<Eigen::Index>(coordinate)]);
            const long double tn = static_cast<long double>(
                target.normal()[static_cast<Eigen::Index>(coordinate)]);
            const long double dn = sn - tn;
            normal_error_squared += dn * dn;
            plane_offset += tn *
                (static_cast<long double>(
                     source_plane.base()[static_cast<Eigen::Index>(coordinate)]) -
                 static_cast<long double>(
                     target.base()[static_cast<Eigen::Index>(coordinate)]));
        }
        const long double tol = static_cast<long double>(tolerance);
        if (std::sqrt(normal_error_squared) <= tol &&
            std::abs(plane_offset) <= tol) {
            return candidate;
        }
    }
    return std::nullopt;
}

} // namespace detail

// --------------------------------------------------------------------------
// Source-mesh import
// --------------------------------------------------------------------------

template <typename NodeScalarT, int Dim, class DatabaseT>
template <class SourceMesh>
[[nodiscard]]
typename HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>::AppendReport
HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>::append_mesh(
    const SourceMesh& source,
    VertexScalar nearest_tolerance,
    NodeScalar boundary_tolerance) {
    if (source.dimension() != this->dimension()) {
        throw std::invalid_argument(
            "Imported mesh dimension does not match HighVoronoiMesh.");
    }

    AppendReport report;
    std::vector<Index> source_to_internal(
        static_cast<std::size_t>(source.size()),
        invalid_index());
    NodePoint node = make_node_point();
    for (Index public_node = Index{0}; public_node < source.size(); ++public_node) {
        for (Index coordinate = Index{0}; coordinate < this->dimension(); ++coordinate) {
            node[static_cast<Eigen::Index>(coordinate)] =
                static_cast<NodeScalar>(source.nodes().get_data(
                    public_node, coordinate));
        }
        source_to_internal[static_cast<std::size_t>(public_node)] =
            append_visible_node(node, boundary_tolerance);
        ++report.appended_nodes;
    }

    HighVoronoiDataMeshView<HighVoronoiMesh> computation(*this);
    auto tree = geometry::make_search_tree(computation, geometry::KDSearch{});

    VertexPoint position = make_vertex_point();
    Sigma translated;
    for (Index source_primary = Index{0};
         source_primary < source.size();
         ++source_primary) {
        for (const auto& vertex : source.primary_vertices(source_primary)) {
            for (Index coordinate = Index{0}; coordinate < this->dimension(); ++coordinate) {
                position[static_cast<Eigen::Index>(coordinate)] =
                    static_cast<VertexScalar>(
                        vertex.position[static_cast<Eigen::Index>(coordinate)]);
            }

            if (!external_boundary().contains(position, boundary_tolerance)) {
                throw std::domain_error(
                    "Imported vertex lies outside the visible HighVoronoi boundary.");
            }
            if (touches_periodic_external_boundary(
                    position,
                    static_cast<VertexScalar>(boundary_tolerance))) {
                ++report.rejected_periodic_vertices;
                continue;
            }

            translated.clear();
            translated.reserve(vertex.sigma.size());
            bool periodic_boundary_generator = false;
            for (const auto source_index_raw : vertex.sigma) {
                const Index source_index = static_cast<Index>(source_index_raw);
                if (source_index < source.size()) {
                    translated.push_back(
                        source_to_internal[static_cast<std::size_t>(source_index)]);
                    continue;
                }

                const Index source_plane = static_cast<Index>(
                    source_index - source.size());
                if (source_plane >= source.boundary().size()) {
                    throw std::logic_error(
                        "Imported mesh vertex contains an invalid boundary index.");
                }
                const auto matched = detail::match_external_plane<HighVoronoiMesh>(
                    *this,
                    source.boundary()[source_plane],
                    boundary_tolerance);
                if (!matched) {
                    throw std::invalid_argument(
                        "Imported boundary vertex does not match the visible HighVoronoi boundary.");
                }
                if (external_boundary()[*matched].is_periodic()) {
                    periodic_boundary_generator = true;
                    break;
                }
                translated.push_back(encode_boundary_internal_index(*matched));
            }

            if (periodic_boundary_generator) {
                ++report.rejected_periodic_vertices;
                continue;
            }
            canonicalize_internal_signature(translated);

            if (!detail::highvoronoi_vertex_survives_nearest_test(
                    *this,
                    computation,
                    tree,
                    position,
                    translated,
                    nearest_tolerance)) {
                ++report.rejected_invalid_vertices;
                continue;
            }

            const Index primary_internal = source_to_internal[
                static_cast<std::size_t>(source_primary)];
            const Address address = store_internal_vertex(
                position,
                translated,
                primary_internal);
            if (address == Address{0}) {
                ++report.duplicate_vertices;
            } else {
                ++report.accepted_vertices;
            }
        }
    }

    return report;
}

// --------------------------------------------------------------------------
// Compute-engine import/attachment
// --------------------------------------------------------------------------

template <typename NodeScalarT, int Dim, class DatabaseT>
template <class Engine>
[[nodiscard]]
typename HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>::AppendReport
HighVoronoiMesh<NodeScalarT, Dim, DatabaseT>::append_engine(
    Engine&& engine_value,
    VertexScalar nearest_tolerance,
    NodeScalar boundary_tolerance) {
    using EngineType = std::decay_t<Engine>;
    static_assert(
        std::is_same_v<typename EngineType::Index, Index>,
        "Compute engine and HighVoronoiMesh must use the same Index type.");
    static_assert(
        std::is_same_v<typename EngineType::VertexScalar, VertexScalar>,
        "Compute engine and HighVoronoiMesh must use the same VertexScalar type.");

    auto engine = std::make_shared<EngineType>(std::forward<Engine>(engine_value));
    if (engine->dimension() != this->dimension()) {
        throw std::invalid_argument(
            "Compute engine dimension does not match HighVoronoiMesh.");
    }

    AppendReport report;
    const Index node_offset = internal_node_count();
    NodePoint node = make_node_point();
    for (Index local = Index{0}; local < engine->node_count(); ++local) {
        engine->copy_node(local, node.data());
        (void)append_visible_node(node, boundary_tolerance);
        ++report.appended_nodes;
    }

    HighVoronoiDataMeshView<HighVoronoiMesh> computation(*this);
    auto tree = geometry::make_search_tree(computation, geometry::KDSearch{});

    typename EngineType::VertexPoint engine_position = [&]() {
        if constexpr (EngineType::DimensionAtCompileTime == Dynamic) {
            return typename EngineType::VertexPoint(
                static_cast<Eigen::Index>(engine->dimension()));
        } else {
            return typename EngineType::VertexPoint{};
        }
    }();
    typename EngineType::Sigma local_sigma;
    VertexPoint position = make_vertex_point();
    Sigma global_sigma;
    std::vector<typename EngineType::Address> to_erase;

    typename EngineType::Address cursor = 0;
    typename EngineType::Address local_address = 0;
    while (engine->next_active_vertex(
               cursor,
               local_address,
               engine_position,
               local_sigma)) {
        for (Index coordinate = Index{0}; coordinate < this->dimension(); ++coordinate) {
            position[static_cast<Eigen::Index>(coordinate)] =
                static_cast<VertexScalar>(
                    engine_position[static_cast<Eigen::Index>(coordinate)]);
        }
        if (!external_boundary().contains(position, boundary_tolerance)) {
            throw std::domain_error(
                "Compute-engine vertex lies outside the visible HighVoronoi boundary.");
        }

        global_sigma.clear();
        global_sigma.reserve(local_sigma.size());
        bool periodic_boundary_generator = false;
        for (const Index local_index : local_sigma) {
            if (local_index < engine->node_count()) {
                global_sigma.push_back(static_cast<Index>(node_offset + local_index));
                continue;
            }
            if (local_index == invalid_index()) {
                throw std::logic_error(
                    "Compute engine returned the reserved invalid index.");
            }
            const Index plane = static_cast<Index>(
                invalid_index() - static_cast<Index>(local_index + Index{1}));
            if (plane >= external_boundary().size()) {
                throw std::invalid_argument(
                    "Compute-engine boundary index is incompatible with HighVoronoi boundary.");
            }
            if (external_boundary()[plane].is_periodic()) {
                periodic_boundary_generator = true;
                break;
            }
            global_sigma.push_back(encode_boundary_internal_index(plane));
        }

        if (periodic_boundary_generator ||
            touches_periodic_external_boundary(
                position,
                static_cast<VertexScalar>(boundary_tolerance))) {
            to_erase.push_back(local_address);
            ++report.rejected_periodic_vertices;
            continue;
        }
        canonicalize_internal_signature(global_sigma);

        if (!detail::highvoronoi_vertex_survives_nearest_test(
                *this,
                computation,
                tree,
                position,
                global_sigma,
                nearest_tolerance)) {
            to_erase.push_back(local_address);
            ++report.rejected_invalid_vertices;
            continue;
        }
        if (database_.contains(global_sigma)) {
            to_erase.push_back(local_address);
            ++report.duplicate_vertices;
        }
    }

    for (const auto address : to_erase) {
        (void)engine->erase_vertex(address);
    }

    if constexpr (detail::CanRegisterComputeEngineV<
                      Database,
                      EngineType,
                      Index>) {
        const auto registration = database_.register_engine(engine, node_offset);

        // Preserve the engine's own primary/secondary classification. Deleted
        // incidence slots are ignored after a direct engine read.
        typename EngineType::Sigma active_sigma;
        for (Index local_node = Index{0}; local_node < engine->node_count(); ++local_node) {
            const Index global_node = static_cast<Index>(node_offset + local_node);
            const std::size_t primary_count = engine->primary_vertex_count(local_node);
            for (std::size_t p = 0; p < primary_count; ++p) {
                const auto local = engine->primary_vertex_address(local_node, p);
                active_sigma.clear();
                engine->read_vertex(local, engine_position, active_sigma);
                if (!active_sigma.empty()) {
                    register_primary_internal(
                        global_node,
                        static_cast<Address>(registration.first_address + local));
                    ++report.accepted_vertices;
                }
            }
            const std::size_t secondary_count = engine->secondary_vertex_count(local_node);
            for (std::size_t p = 0; p < secondary_count; ++p) {
                const auto local = engine->secondary_vertex_address(local_node, p);
                active_sigma.clear();
                engine->read_vertex(local, engine_position, active_sigma);
                if (!active_sigma.empty()) {
                    register_secondary_internal(
                        global_node,
                        static_cast<Address>(registration.first_address + local));
                }
            }
        }
    } else {
        // Backend cannot retain engines: preserve the engine's declared primary
        // owner while materializing each surviving vertex exactly once.
        std::vector<std::optional<Index>> primary_owner(
            static_cast<std::size_t>(engine->vertex_address_capacity()));
        for (Index local_node = Index{0}; local_node < engine->node_count(); ++local_node) {
            const std::size_t count = engine->primary_vertex_count(local_node);
            for (std::size_t p = 0; p < count; ++p) {
                const auto local = engine->primary_vertex_address(local_node, p);
                if (local < engine->vertex_address_capacity()) {
                    primary_owner[static_cast<std::size_t>(local)] =
                        static_cast<Index>(node_offset + local_node);
                }
            }
        }

        cursor = 0;
        while (engine->next_active_vertex(
                   cursor,
                   local_address,
                   engine_position,
                   local_sigma)) {
            global_sigma.clear();
            for (const Index local_index : local_sigma) {
                if (local_index < engine->node_count()) {
                    global_sigma.push_back(static_cast<Index>(node_offset + local_index));
                } else {
                    global_sigma.push_back(local_index);
                }
            }
            canonicalize_internal_signature(global_sigma);
            for (Index coordinate = Index{0}; coordinate < this->dimension(); ++coordinate) {
                position[static_cast<Eigen::Index>(coordinate)] =
                    static_cast<VertexScalar>(
                        engine_position[static_cast<Eigen::Index>(coordinate)]);
            }
            const Address stored = store_internal_vertex(
                position,
                global_sigma,
                primary_owner[static_cast<std::size_t>(local_address)]);
            if (stored == Address{0}) {
                ++report.duplicate_vertices;
            } else {
                ++report.accepted_vertices;
            }
        }
    }

    return report;
}

} // namespace highvoronoi
