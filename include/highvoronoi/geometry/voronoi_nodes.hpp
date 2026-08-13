#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Core>

#include <highvoronoi/geometry/point.hpp>
#include <highvoronoi/geometry/boundary.hpp>

namespace highvoronoi {

/**
 * @file voronoi_nodes.hpp
 * @brief Unified interfaces for stored, computed, hybrid, and reflected nodes.
 *
 * HighVoronoi distinguishes three compile-time node-access modes:
 *
 * - `NodeAccessMode::Stored`: every node has stable contiguous storage and
 *   `operator[]` returns a read-only Eigen view without copying.
 * - `NodeAccessMode::Computed`: nodes are produced on demand and `operator[]`
 *   returns an owning Eigen point.
 * - `NodeAccessMode::Hybrid`: an individual node may either have stable storage
 *   or be computed. `operator[]` returns a `NodeHandle`, which owns a computed
 *   point or borrows stable storage.
 *
 * All three branches derive from `AbstractVoronoiNodes` and therefore share the
 * universal copy interface `copy_node(index, target)`. Algorithms that must be
 * independent of the access mode should use that interface.
 *
 * `ExtendedVoronoiNodes<BaseNodes>` combines the ordinary-node access mode
 * with stable storage for activated mirror nodes:
 *
 * - stored base -> stored extended access,
 * - computed base -> hybrid extended access,
 * - hybrid base -> hybrid extended access.
 *
 * `PrecomputedExtendedVoronoiNodes<BaseNodes>` follows the same access rule but
 * precomputes every node/plane reflection.
 */

// ============================================================================
// Access-mode declarations and common point traits
// ============================================================================

/** @brief Compile-time semantics of node access through `operator[]`. */
enum class NodeAccessMode {
    Stored,
    Computed,
    Hybrid
};

/** @brief Reflection-storage strategy used by extended node containers. */
enum class MirrorStorageMode {
    ActiveCell,
    AllNodes
};

namespace detail {

template <typename Scalar, int Dim>
using OwnedNodePoint = std::conditional_t<
    Dim == Dynamic,
    DynamicPoint<Scalar>,
    StaticPoint<Scalar, Dim>>;

template <typename Scalar, int Dim>
using NodePointView = std::conditional_t<
    Dim == Dynamic,
    DynamicPointView<Scalar>,
    StaticPointView<Scalar, Dim>>;

} // namespace detail

// ============================================================================
// AbstractVoronoiNodes
// ============================================================================

/**
 * @brief Access-mode-independent polymorphic node interface.
 *
 * The common interface deliberately exposes copying rather than a raw pointer.
 * This permits implementations whose points are calculated on demand and have
 * no persistent address. The three access branches add their own statically
 * fixed `operator[]` return types.
 *
 * @tparam ScalarT Coordinate scalar type.
 * @tparam Dim Compile-time dimension or `highvoronoi::Dynamic`.
 * @tparam IndexT Unsigned node-index type.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class AbstractVoronoiNodes {
    static_assert(Dim == Dynamic || Dim > 0,
                  "Node dimension must be positive or highvoronoi::Dynamic.");
    static_assert(std::is_integral_v<IndexT>,
                  "Node index type must be integral.");
    static_assert(std::is_unsigned_v<IndexT>,
                  "Node index type must be unsigned.");

public:
    using Scalar = ScalarT;
    using Index = IndexT;
    using Point = detail::OwnedNodePoint<Scalar, Dim>;
    using PointView = detail::NodePointView<Scalar, Dim>;

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr bool IsDynamic = (Dim == Dynamic);

    virtual ~AbstractVoronoiNodes() = default;

    AbstractVoronoiNodes(const AbstractVoronoiNodes&) = default;
    AbstractVoronoiNodes(AbstractVoronoiNodes&&) noexcept = default;
    AbstractVoronoiNodes& operator=(const AbstractVoronoiNodes&) = default;
    AbstractVoronoiNodes& operator=(AbstractVoronoiNodes&&) noexcept = default;

    /** @brief Return the number of addressable nodes. */
    [[nodiscard]] virtual Index size() const noexcept = 0;

    /** @brief Synonym for `size()`. */
    [[nodiscard]] Index length() const noexcept {
        return size();
    }

    /** @brief Return whether no nodes are addressable. */
    [[nodiscard]] bool empty() const noexcept {
        return size() == Index{0};
    }

    /** @brief Return the common spatial dimension. */
    [[nodiscard]] Index dimension() const noexcept {
        return dimension_;
    }

    /**
     * @brief Return the number of currently active boundary-extension nodes.
     *
     * Ordinary node containers return zero. Extended node containers override
     * this function and expose only the mirror nodes activated for the current
     * cell. This keeps generic search structures from inspecting inactive
     * mirror slots whose stored coordinates may be stale.
     */
    [[nodiscard]] virtual Index
    active_boundary_size() const noexcept {
        return Index{0};
    }

    /**
     * @brief Return one active boundary node in the container's public index space.
     *
     * @param position Zero-based position in the active-boundary sequence.
     * @throws std::out_of_range for ordinary containers and invalid positions.
     */
    [[nodiscard]] virtual Index
    active_boundary_index(Index /* position */) const {
        throw std::out_of_range(
            "Node container has no active boundary-extension nodes.");
    }

    /**
     * @brief Return one coordinate of one node.
     *
     * This interface is intended for adapters that repeatedly access single
     * coordinates, such as nanoflann. Concrete node containers should
     * override it with their natural direct access path. The default keeps
     * existing derived classes source-compatible and fails only if the
     * unsupported operation is actually called.
     */
    [[nodiscard]] virtual Scalar get_data(
        Index /* node */,
        Index /* coordinate */) const {
        throw std::logic_error(
            "get_data(node, coordinate) is not implemented for this node container.");
    }

    /**
     * @brief Copy node `index` into caller-owned contiguous storage.
     *
     * `target` must point to at least `dimension()` writable scalar entries.
     * Like `std::vector::operator[]`, this overload performs no bounds check.
     */
    void copy_node(Index index, Scalar* target) const {
        copy_node_impl(index, target);
    }

    /**
     * @brief Copy node `index` into an owning point object.
     *
     * A dynamic target is resized to `dimension()` before it is filled.
     */
    void copy_node(Index index, Point& target) const {
        if constexpr (IsDynamic) {
            target.resize(static_cast<Eigen::Index>(dimension_));
        }
        copy_node_impl(index, target.data());
    }

    /**
     * @brief Materialize node `index` as an owning point.
     *
     * This function is valid for all access modes and is particularly useful
     * in generic adapters and boundary-validation code.
     */
    [[nodiscard]] Point node(Index index) const {
        Point result = make_point();
        copy_node_impl(index, result.data());
        return result;
    }

    /** @brief Bounds-checked owning-point access. */
    [[nodiscard]] Point node_at(Index index) const {
        require_index(index);
        return node(index);
    }

    /**
     * @brief Activate temporary nodes required for one Voronoi cell.
     *
     * The default implementation performs no operation. Neighbour indices are
     * zero-based global node indices.
     */
    void activate_cell(
        Index cell,
        const Index* neighbours,
        Index neighbour_count) {
        activate_cell_impl(cell, neighbours, neighbour_count);
    }

    /** @brief Convenience overload for contiguous neighbour containers. */
    template <typename NeighbourContainer>
    void activate_cell(
        Index cell,
        const NeighbourContainer& neighbours) {
        activate_cell(
            cell,
            neighbours.data(),
            static_cast<Index>(neighbours.size()));
    }

protected:
    explicit AbstractVoronoiNodes(Index dimension)
        : dimension_(checked_dimension(dimension)) {}

    /** @brief Replace the runtime dimension after validating it. */
    void set_dimension(Index dimension) {
        dimension_ = checked_dimension(dimension);
    }

    /** @brief Create an owning point with the correct runtime dimension. */
    [[nodiscard]] Point make_point() const {
        if constexpr (IsDynamic) {
            return Point(static_cast<Eigen::Index>(dimension_));
        } else {
            return Point{};
        }
    }

    void require_index(Index index) const {
        if (index >= size()) {
            throw std::out_of_range("Voronoi-node index out of range.");
        }
    }

private:
    [[nodiscard]] static Index checked_dimension(Index dimension) {
        if constexpr (IsDynamic) {
            return dimension;
        } else {
            if (dimension != static_cast<Index>(Dim)) {
                throw std::invalid_argument(
                    "Runtime dimension does not match fixed node dimension.");
            }
            return static_cast<Index>(Dim);
        }
    }

    virtual void copy_node_impl(Index index, Scalar* target) const = 0;

    virtual void activate_cell_impl(
        Index /* cell */,
        const Index* /* neighbours */,
        Index /* neighbour_count */) {}

    Index dimension_;
};

// ============================================================================
// Stored access branch
// ============================================================================

/**
 * @brief Intermediate base for node containers with stable contiguous points.
 *
 * Derived classes implement `get_stored_node_pointer(index)`. The pointer must
 * remain valid until the concrete container is structurally modified or
 * destroyed. Reading through `operator[]` is zero-copy.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class StoredNodeAccess
    : public AbstractVoronoiNodes<ScalarT, Dim, IndexT> {
public:
    using Base = AbstractVoronoiNodes<ScalarT, Dim, IndexT>;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Point = typename Base::Point;
    using PointView = typename Base::PointView;

    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;
    static constexpr int DimensionAtCompileTime = Dim;

    [[nodiscard]] PointView operator[](Index index) const {
        return make_view(get_stored_node_pointer(index));
    }

    [[nodiscard]] PointView at(Index index) const {
        this->require_index(index);
        return (*this)[index];
    }

    /** @brief Return stable storage for use by specialized adapters. */
    [[nodiscard]] const Scalar* stable_node_data(Index index) const {
        return get_stored_node_pointer(index);
    }

protected:
    explicit StoredNodeAccess(Index dimension)
        : Base(dimension) {}

    [[nodiscard]] virtual const Scalar*
    get_stored_node_pointer(Index index) const = 0;

private:
    [[nodiscard]] PointView make_view(const Scalar* pointer) const {
        if constexpr (Dim == Dynamic) {
            return PointView(
                pointer,
                static_cast<Eigen::Index>(this->dimension()));
        } else {
            return PointView(pointer);
        }
    }

    void copy_node_impl(Index index, Scalar* target) const final {
        std::copy_n(
            get_stored_node_pointer(index),
            static_cast<std::size_t>(this->dimension()),
            target);
    }
};

// ============================================================================
// Computed access branch
// ============================================================================

/**
 * @brief Intermediate base for nodes calculated on demand.
 *
 * `operator[]` returns an owning point, so no pointer to temporary internal
 * storage escapes from a derived class.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class ComputedNodeAccess
    : public AbstractVoronoiNodes<ScalarT, Dim, IndexT> {
public:
    using Base = AbstractVoronoiNodes<ScalarT, Dim, IndexT>;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Point = typename Base::Point;
    using PointView = typename Base::PointView;

    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Computed;
    static constexpr int DimensionAtCompileTime = Dim;

    [[nodiscard]] Point operator[](Index index) const {
        Point result = this->make_point();
        compute_node(index, result.data());
        return result;
    }

    [[nodiscard]] Point at(Index index) const {
        this->require_index(index);
        return (*this)[index];
    }

protected:
    explicit ComputedNodeAccess(Index dimension)
        : Base(dimension) {}

    virtual void compute_node(Index index, Scalar* target) const = 0;

private:
    void copy_node_impl(Index index, Scalar* target) const final {
        compute_node(index, target);
    }
};

// ============================================================================
// Hybrid node handle and access branch
// ============================================================================

/**
 * @brief Read result of a hybrid node container.
 *
 * A handle either borrows stable contiguous storage or owns a computed point.
 * The coordinate pointer returned by `data()` remains valid for the lifetime of
 * the handle, subject to the usual invalidation rules of borrowed storage.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class NodeHandle {
public:
    using Scalar = ScalarT;
    using Index = IndexT;
    using Point = detail::OwnedNodePoint<Scalar, Dim>;
    using PointView = detail::NodePointView<Scalar, Dim>;

    static constexpr bool IsDynamic = (Dim == Dynamic);

    [[nodiscard]] static NodeHandle borrowed(
        const Scalar* pointer,
        Index dimension) {
        return NodeHandle(BorrowedNode{pointer}, dimension);
    }

    [[nodiscard]] static NodeHandle owning(
        Point point,
        Index dimension) {
        return NodeHandle(std::move(point), dimension);
    }

    [[nodiscard]] bool owns_storage() const noexcept {
        return std::holds_alternative<Point>(storage_);
    }

    [[nodiscard]] const Scalar* data() const noexcept {
        if (const auto* borrowed_node =
                std::get_if<BorrowedNode>(&storage_)) {
            return borrowed_node->pointer;
        }
        return std::get<Point>(storage_).data();
    }

    [[nodiscard]] Index size() const noexcept {
        return dimension_;
    }

    [[nodiscard]] Index dimension() const noexcept {
        return dimension_;
    }

    [[nodiscard]] const Scalar& operator[](Index coordinate) const noexcept {
        return data()[coordinate];
    }

    void copy_to(Scalar* target) const {
        std::copy_n(
            data(),
            static_cast<std::size_t>(dimension_),
            target);
    }

    /**
     * @brief Return an Eigen view valid while this lvalue handle remains alive.
     */
    [[nodiscard]] PointView view() const & {
        if constexpr (IsDynamic) {
            return PointView(
                data(),
                static_cast<Eigen::Index>(dimension_));
        } else {
            return PointView(data());
        }
    }

    /** Prevent a dangling view into a temporary owning handle. */
    PointView view() const && = delete;

private:
    struct BorrowedNode {
        const Scalar* pointer;
    };

    explicit NodeHandle(BorrowedNode borrowed_node, Index dimension)
        : storage_(borrowed_node),
          dimension_(dimension) {}

    explicit NodeHandle(Point point, Index dimension)
        : storage_(std::move(point)),
          dimension_(dimension) {}

    std::variant<BorrowedNode, Point> storage_;
    Index dimension_;
};

/**
 * @brief Intermediate base for a mixture of stored and computed nodes.
 *
 * Derived classes return a stable pointer from
 * `try_get_stored_node_pointer(index)` when available and `nullptr` otherwise.
 * In the latter case `compute_node(index, target)` materializes the point.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class HybridNodeAccess
    : public AbstractVoronoiNodes<ScalarT, Dim, IndexT> {
public:
    using Base = AbstractVoronoiNodes<ScalarT, Dim, IndexT>;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Point = typename Base::Point;
    using PointView = typename Base::PointView;
    using Handle = NodeHandle<Scalar, Dim, Index>;

    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Hybrid;
    static constexpr int DimensionAtCompileTime = Dim;

    [[nodiscard]] Handle operator[](Index index) const {
        if (const Scalar* pointer =
                try_get_stored_node_pointer(index)) {
            return Handle::borrowed(pointer, this->dimension());
        }

        Point result = this->make_point();
        compute_node(index, result.data());
        return Handle::owning(std::move(result), this->dimension());
    }

    [[nodiscard]] Handle at(Index index) const {
        this->require_index(index);
        return (*this)[index];
    }

    /**
     * @brief Return stable storage when available, otherwise `nullptr`.
     */
    [[nodiscard]] const Scalar*
    try_stable_node_data(Index index) const {
        return try_get_stored_node_pointer(index);
    }

protected:
    explicit HybridNodeAccess(Index dimension)
        : Base(dimension) {}

    [[nodiscard]] virtual const Scalar*
    try_get_stored_node_pointer(Index index) const = 0;

    virtual void compute_node(Index index, Scalar* target) const = 0;

private:
    void copy_node_impl(Index index, Scalar* target) const final {
        if (const Scalar* pointer =
                try_get_stored_node_pointer(index)) {
            std::copy_n(
                pointer,
                static_cast<std::size_t>(this->dimension()),
                target);
            return;
        }

        compute_node(index, target);
    }
};

namespace detail {

template <NodeAccessMode Mode,
          typename Scalar,
          int Dim,
          typename Index>
struct NodeAccessBaseSelector;

template <typename Scalar, int Dim, typename Index>
struct NodeAccessBaseSelector<NodeAccessMode::Stored, Scalar, Dim, Index> {
    using type = StoredNodeAccess<Scalar, Dim, Index>;
};

template <typename Scalar, int Dim, typename Index>
struct NodeAccessBaseSelector<NodeAccessMode::Computed, Scalar, Dim, Index> {
    using type = ComputedNodeAccess<Scalar, Dim, Index>;
};

template <typename Scalar, int Dim, typename Index>
struct NodeAccessBaseSelector<NodeAccessMode::Hybrid, Scalar, Dim, Index> {
    using type = HybridNodeAccess<Scalar, Dim, Index>;
};

template <NodeAccessMode Mode,
          typename Scalar,
          int Dim,
          typename Index>
using NodeAccessBase =
    typename NodeAccessBaseSelector<Mode, Scalar, Dim, Index>::type;

template <typename Nodes>
using MatchingNodeAccessBase = NodeAccessBase<
    Nodes::AccessMode,
    typename Nodes::Scalar,
    Nodes::DimensionAtCompileTime,
    typename Nodes::Index>;

/**
 * @brief Access mode exposed by an extended-node container.
 *
 * Mirror nodes always live in stable storage. Therefore an extended container
 * can remain purely stored only when its ordinary base nodes are stored.
 * Computed ordinary nodes become the computed branch of hybrid access, while
 * the stored mirror slots form the stored branch.
 */
template <NodeAccessMode BaseMode>
inline constexpr NodeAccessMode ExtendedNodeAccessModeV =
    BaseMode == NodeAccessMode::Stored
        ? NodeAccessMode::Stored
        : NodeAccessMode::Hybrid;

template <typename Nodes>
using ExtendedNodeAccessBase = NodeAccessBase<
    ExtendedNodeAccessModeV<Nodes::AccessMode>,
    typename Nodes::Scalar,
    Nodes::DimensionAtCompileTime,
    typename Nodes::Index>;

} // namespace detail

// ============================================================================
// VoronoiNodes: ordinary stored nodes
// ============================================================================

/**
 * @brief Owning contiguous storage for ordinary Voronoi nodes.
 *
 * Coordinates are stored node by node in one `std::vector<Scalar>`. Reading
 * through `operator[]` is inherited from `StoredNodeAccess` and returns a
 * read-only Eigen view.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class VoronoiNodes
    : public StoredNodeAccess<ScalarT, Dim, IndexT> {
public:
    using Base = StoredNodeAccess<ScalarT, Dim, IndexT>;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Point = typename Base::Point;
    using PointView = typename Base::PointView;

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr NodeAccessMode AccessMode = NodeAccessMode::Stored;
    static constexpr bool IsDynamic = (Dim == Dynamic);

    using Base::dimension;

    explicit VoronoiNodes(Index length = Index{0})
        : Base(IsDynamic ? Index{0} : static_cast<Index>(Dim)),
          length_(length),
          data_(checked_scalar_count(length, this->dimension())) {}

    VoronoiNodes(Index length, Index dimension)
        : Base(checked_dimension(dimension)),
          length_(length),
          data_(checked_scalar_count(length, this->dimension())) {}

    [[nodiscard]] Index size() const noexcept override {
        return length_;
    }

    /** @brief Change the runtime dimension and resize coordinate storage. */
    void dimension(Index new_dimension) {
        if constexpr (IsDynamic) {
            if (new_dimension == this->dimension()) {
                return;
            }
            data_.resize(checked_scalar_count(length_, new_dimension));
            this->set_dimension(new_dimension);
        } else if (new_dimension != static_cast<Index>(Dim)) {
            throw std::invalid_argument(
                "Cannot change the dimension of fixed-dimensional VoronoiNodes.");
        }
    }

    /** @brief Resize the number of nodes. Existing scalar values are preserved. */
    void resize(Index new_length) {
        data_.resize(checked_scalar_count(new_length, this->dimension()));
        length_ = new_length;
    }

    [[nodiscard]] Scalar* getdata() noexcept {
        return data_.data();
    }

    [[nodiscard]] const Scalar* getdata() const noexcept {
        return data_.data();
    }

    [[nodiscard]] Scalar get_data(
        Index node,
        Index coordinate) const override {
        return data_[
            static_cast<std::size_t>(node) *
                static_cast<std::size_t>(this->dimension()) +
            static_cast<std::size_t>(coordinate)];
    }

    [[nodiscard]] Index scalar_count() const noexcept {
        return static_cast<Index>(data_.size());
    }

    /** @brief Replace one node with an Eigen-compatible vector expression. */
    template <typename Derived>
    void set(Index index, const Eigen::MatrixBase<Derived>& point) {
        this->require_index(index);
        if (static_cast<Index>(point.size()) != this->dimension()) {
            throw std::invalid_argument(
                "Point dimension does not match VoronoiNodes dimension.");
        }

        Scalar* destination =
            data_.data() + index * this->dimension();

        if constexpr (IsDynamic) {
            Eigen::Map<DynamicPoint<Scalar>> target(
                destination,
                static_cast<Eigen::Index>(this->dimension()));
            target = point;
        } else {
            Eigen::Map<StaticPoint<Scalar, Dim>> target(destination);
            target = point;
        }
    }

    /** @brief Replace one node from contiguous scalar storage. */
    void set(Index index, const Scalar* point) {
        this->require_index(index);
        std::copy_n(
            point,
            static_cast<std::size_t>(this->dimension()),
            data_.data() + index * this->dimension());
    }

protected:
    [[nodiscard]] const Scalar*
    get_stored_node_pointer(Index index) const override {
        return data_.data() + index * this->dimension();
    }

private:
    [[nodiscard]] static Index checked_dimension(Index dimension) {
        if constexpr (IsDynamic) {
            return dimension;
        } else {
            if (dimension != static_cast<Index>(Dim)) {
                throw std::invalid_argument(
                    "Explicit dimension does not match fixed VoronoiNodes dimension.");
            }
            return static_cast<Index>(Dim);
        }
    }

    [[nodiscard]] static std::size_t checked_scalar_count(
        Index length,
        Index dimension) {
        if (dimension != Index{0} &&
            length > std::numeric_limits<Index>::max() / dimension) {
            throw std::length_error("VoronoiNodes storage size overflow.");
        }

        const Index count = length * dimension;
        if (count > static_cast<Index>(
                        std::numeric_limits<std::size_t>::max())) {
            throw std::length_error("VoronoiNodes storage size overflow.");
        }
        return static_cast<std::size_t>(count);
    }

    Index length_;
    std::vector<Scalar> data_;
};

// ============================================================================
// Extended nodes
// ============================================================================

namespace detail {

template <typename BaseNodes,
          MirrorStorageMode StorageMode>
class ExtendedVoronoiNodesImpl
    : public ExtendedNodeAccessBase<BaseNodes> {
public:
    using AccessBase = ExtendedNodeAccessBase<BaseNodes>;
    using Scalar = typename BaseNodes::Scalar;
    using Index = typename BaseNodes::Index;
    using Point = typename AccessBase::Point;
    using PointView = typename AccessBase::PointView;
    using BoundaryType = Boundary<
        BaseNodes::DimensionAtCompileTime,
        Scalar,
        Index>;
    using MirrorNodes = VoronoiNodes<
        Scalar,
        BaseNodes::DimensionAtCompileTime,
        Index>;

    static constexpr int DimensionAtCompileTime =
        BaseNodes::DimensionAtCompileTime;
    static constexpr NodeAccessMode AccessMode =
        ExtendedNodeAccessModeV<BaseNodes::AccessMode>;
    static constexpr MirrorStorageMode MirrorMode = StorageMode;

    static_assert(std::is_base_of_v<
                      AbstractVoronoiNodes<
                          Scalar,
                          DimensionAtCompileTime,
                          Index>,
                      BaseNodes>,
                  "BaseNodes must derive from the matching "
                  "AbstractVoronoiNodes specialization.");

    explicit ExtendedVoronoiNodesImpl(
        BaseNodes nodes,
        BoundaryType boundary = BoundaryType{})
        : AccessBase(nodes.dimension()),
          nodes_(std::move(nodes)),
          boundary_(std::move(boundary)),
          mirrors_(boundary_.size(), nodes_.dimension()),
          precomputed_(precomputed_length(), nodes_.dimension()),
          active_(static_cast<std::size_t>(boundary_.size()), std::uint8_t{0}) {
        validate_dimensions();
        active_planes_.reserve(static_cast<std::size_t>(boundary_.size()));
        if constexpr (StorageMode == MirrorStorageMode::AllNodes) {
            precompute_all_reflections();
        }
    }

    [[nodiscard]] Index size() const noexcept override {
        return inner_size() + mirror_count();
    }

    [[nodiscard]] Index inner_size() const noexcept {
        return nodes_.size();
    }

    [[nodiscard]] Index mirror_count() const noexcept {
        return boundary_.size();
    }

    [[nodiscard]] Index mirror_index(Index plane) const {
        if (plane >= mirror_count()) {
            throw std::out_of_range("Boundary-plane index out of range.");
        }
        return inner_size() + plane;
    }

    [[nodiscard]] bool is_mirror_index(Index index) const noexcept {
        return index >= inner_size() && index < size();
    }

    [[nodiscard]] Index mirror_plane(Index index) const {
        if (!is_mirror_index(index)) {
            throw std::out_of_range(
                "Node index does not identify an extended mirror node.");
        }
        return index - inner_size();
    }

    [[nodiscard]] const BaseNodes& inner_nodes() const noexcept {
        return nodes_;
    }

    [[nodiscard]] const BoundaryType& boundary() const noexcept {
        return boundary_;
    }

    [[nodiscard]] const MirrorNodes& mirror_nodes() const noexcept {
        return mirrors_;
    }

    [[nodiscard]] const std::vector<Index>&
    active_planes() const noexcept {
        return active_planes_;
    }

    [[nodiscard]] Index
    active_boundary_size() const noexcept override {
        return static_cast<Index>(active_planes_.size());
    }

    [[nodiscard]] Index
    active_boundary_index(Index position) const override {
        if (position >= active_boundary_size()) {
            throw std::out_of_range(
                "Active boundary-node position out of range.");
        }
        return mirror_index(
            active_planes_[static_cast<std::size_t>(position)]);
    }

    [[nodiscard]] Scalar get_data(
        Index node,
        Index coordinate) const override {
        if (node < inner_size()) {
            return nodes_.get_data(node, coordinate);
        }
        return mirrors_.get_data(node - inner_size(), coordinate);
    }

    [[nodiscard]] std::optional<Index> active_cell() const noexcept {
        return active_cell_;
    }

    [[nodiscard]] bool mirror_is_active(Index plane) const {
        if (plane >= mirror_count()) {
            throw std::out_of_range("Boundary-plane index out of range.");
        }
        return active_[static_cast<std::size_t>(plane)] != std::uint8_t{0};
    }

protected:
    [[nodiscard]] const MirrorNodes&
    all_reflections() const noexcept {
        return precomputed_;
    }

    [[nodiscard]] PointView precomputed_reflection(
        Index cell,
        Index plane) const {
        if constexpr (StorageMode != MirrorStorageMode::AllNodes) {
            throw std::logic_error(
                "Reflections are not precomputed by this node container.");
        } else {
            if (cell >= inner_size() || plane >= mirror_count()) {
                throw std::out_of_range(
                    "Precomputed reflection index out of range.");
            }
            return precomputed_[precomputed_index(cell, plane)];
        }
    }

    // Stored branch hook. For other access modes this is an ordinary method.
    [[nodiscard]] const Scalar*
    get_stored_node_pointer(Index index) const {
        if (index < inner_size()) {
            static_assert(AccessMode == NodeAccessMode::Stored,
                          "Only stored base nodes provide mandatory pointers.");
            return nodes_.stable_node_data(index);
        }
        return mirrors_.stable_node_data(index - inner_size());
    }

    // Hybrid branch hook. For other access modes this is an ordinary method.
    [[nodiscard]] const Scalar*
    try_get_stored_node_pointer(Index index) const {
        if (index >= inner_size()) {
            return mirrors_.stable_node_data(index - inner_size());
        }

        if constexpr (BaseNodes::AccessMode == NodeAccessMode::Stored) {
            return nodes_.stable_node_data(index);
        } else if constexpr (
            BaseNodes::AccessMode == NodeAccessMode::Hybrid) {
            return nodes_.try_stable_node_data(index);
        } else {
            // Computed ordinary nodes have no stable storage. They form the
            // computed branch of the hybrid extended-node interface.
            return nullptr;
        }
    }

    // Computed and hybrid branch hook.
    void compute_node(Index index, Scalar* target) const {
        if (index < inner_size()) {
            nodes_.copy_node(index, target);
        } else {
            mirrors_.copy_node(index - inner_size(), target);
        }
    }

private:
    void activate_cell_impl(
        Index cell,
        const Index* neighbours,
        Index neighbour_count) override {
        if (cell >= inner_size()) {
            throw std::out_of_range(
                "Only an inner node can define the active Voronoi cell.");
        }

        clear_active_mirrors();
        active_cell_ = cell;

        for (Index position = neighbour_count;
             position > Index{0};
             --position) {
            const Index neighbour = neighbours[position - Index{1}];
            if (neighbour < inner_size()) {
                break;
            }

            const Index plane = neighbour - inner_size();
            if (plane >= mirror_count()) {
                throw std::out_of_range(
                    "Mirror neighbour refers to an invalid boundary plane.");
            }
            activate_mirror(cell, plane);
        }
    }

    void clear_active_mirrors() noexcept {
        for (const Index plane : active_planes_) {
            active_[static_cast<std::size_t>(plane)] = std::uint8_t{0};
        }
        active_planes_.clear();
        active_cell_.reset();
    }

    void activate_mirror(Index cell, Index plane) {
        std::uint8_t& is_active =
            active_[static_cast<std::size_t>(plane)];
        if (is_active != std::uint8_t{0}) {
            return;
        }

        if constexpr (StorageMode == MirrorStorageMode::AllNodes) {
            mirrors_.set(
                plane,
                precomputed_[precomputed_index(cell, plane)]);
        } else {
            const Point source = nodes_.node(cell);
            mirrors_.set(plane, boundary_.reflect(source, plane));
        }

        is_active = std::uint8_t{1};
        active_planes_.push_back(plane);
    }

    void precompute_all_reflections() {
        for (Index cell = Index{0}; cell < inner_size(); ++cell) {
            const Point source = nodes_.node(cell);
            for (Index plane = Index{0}; plane < mirror_count(); ++plane) {
                precomputed_.set(
                    precomputed_index(cell, plane),
                    boundary_.reflect(source, plane));
            }
        }
    }

    [[nodiscard]] Index precomputed_length() const {
        if constexpr (StorageMode == MirrorStorageMode::AllNodes) {
            return checked_product(inner_size(), mirror_count());
        } else {
            return Index{0};
        }
    }

    [[nodiscard]] Index precomputed_index(
        Index cell,
        Index plane) const noexcept {
        return cell * mirror_count() + plane;
    }

    [[nodiscard]] static Index checked_product(Index left, Index right) {
        if (right != Index{0} &&
            left > std::numeric_limits<Index>::max() / right) {
            throw std::length_error(
                "Precomputed reflection storage size overflow.");
        }
        return left * right;
    }

    void validate_dimensions() const {
        if (boundary_.dimension() != 0 &&
            boundary_.dimension() !=
                static_cast<std::size_t>(nodes_.dimension())) {
            throw std::invalid_argument(
                "Boundary and node-container dimensions do not match.");
        }

        if (inner_size() >
            std::numeric_limits<Index>::max() - mirror_count()) {
            throw std::length_error("Extended node count overflow.");
        }
    }

    BaseNodes nodes_;
    BoundaryType boundary_;
    MirrorNodes mirrors_;
    MirrorNodes precomputed_;
    std::vector<std::uint8_t> active_;
    std::vector<Index> active_planes_;
    std::optional<Index> active_cell_;
};

} // namespace detail

/**
 * @brief Add cell-local boundary reflections to any node-access branch.
 *
 * The exposed mode follows the storage semantics of the complete extended
 * container:
 *
 * - stored base -> stored extended access,
 * - computed base -> hybrid extended access,
 * - hybrid base -> hybrid extended access.
 *
 * Computed ordinary nodes are materialized on demand, whereas activated mirror
 * nodes borrow the stable mirror slots owned by this container.
 */
template <typename BaseNodes>
class ExtendedVoronoiNodes
    : public detail::ExtendedVoronoiNodesImpl<
          BaseNodes,
          MirrorStorageMode::ActiveCell> {
public:
    using Base = detail::ExtendedVoronoiNodesImpl<
        BaseNodes,
        MirrorStorageMode::ActiveCell>;
    using Base::Base;

    /**
     * @brief Create an independent cell-local copy.
     *
     * BaseNodes is copied according to its own copy semantics. For the mesh
     * node views used by VoronoiMesh/SerialMesh this is a cheap non-owning
     * view copy. Boundary metadata is copied, while mirror storage and all
     * activation state are created afresh and start inactive.
     */
    [[nodiscard]] ExtendedVoronoiNodes safe_copy() const {
        static_assert(
            std::is_copy_constructible_v<BaseNodes>,
            "ExtendedVoronoiNodes::safe_copy requires copyable BaseNodes.");
        return ExtendedVoronoiNodes(
            this->inner_nodes(),
            this->boundary());
    }
};

/**
 * @brief Extended nodes with every node/plane reflection precomputed.
 *
 * `activate_cell` copies only the required reflections into the current mirror
 * slots. The exposed access mode follows the same rule as
 * `ExtendedVoronoiNodes`: stored remains stored, while computed and hybrid
 * ordinary nodes produce hybrid extended access because mirror slots are
 * stored.
 */
template <typename BaseNodes>
class PrecomputedExtendedVoronoiNodes
    : public detail::ExtendedVoronoiNodesImpl<
          BaseNodes,
          MirrorStorageMode::AllNodes> {
public:
    using Base = detail::ExtendedVoronoiNodesImpl<
        BaseNodes,
        MirrorStorageMode::AllNodes>;
    using Index = typename Base::Index;
    using PointView = typename Base::PointView;
    using MirrorNodes = typename Base::MirrorNodes;

    using Base::Base;

    [[nodiscard]] const MirrorNodes&
    precomputed_nodes() const noexcept {
        return this->all_reflections();
    }

    [[nodiscard]] PointView reflected_node(
        Index cell,
        Index plane) const {
        return this->precomputed_reflection(cell, plane);
    }

    /**
     * @brief Safe copying of precomputed storage is deliberately unresolved.
     *
     * A correct implementation should share the immutable precomputed
     * reflection table while duplicating only active mirror slots/state.
     * Recomputing or deep-copying the full table here would violate the
     * intended safe-copy semantics.
     */
    [[nodiscard]] PrecomputedExtendedVoronoiNodes safe_copy() const {
        throw std::logic_error(
            "PrecomputedExtendedVoronoiNodes::safe_copy is not implemented: "
            "the precomputed reflection table must be shared explicitly.");
    }
};

// C++17 deduction guides.
template <typename BaseNodes>
ExtendedVoronoiNodes(
    BaseNodes,
    Boundary<
        BaseNodes::DimensionAtCompileTime,
        typename BaseNodes::Scalar,
        typename BaseNodes::Index>)
    -> ExtendedVoronoiNodes<BaseNodes>;

template <typename BaseNodes>
PrecomputedExtendedVoronoiNodes(
    BaseNodes,
    Boundary<
        BaseNodes::DimensionAtCompileTime,
        typename BaseNodes::Scalar,
        typename BaseNodes::Index>)
    -> PrecomputedExtendedVoronoiNodes<BaseNodes>;

/** @brief Explicit construction helper for active-cell extended nodes. */
template <typename BaseNodes>
[[nodiscard]] auto make_extended_nodes(
    BaseNodes nodes,
    Boundary<
        BaseNodes::DimensionAtCompileTime,
        typename BaseNodes::Scalar,
        typename BaseNodes::Index> boundary)
    -> ExtendedVoronoiNodes<BaseNodes> {
    return ExtendedVoronoiNodes<BaseNodes>(
        std::move(nodes),
        std::move(boundary));
}

/** @brief Explicit construction helper for fully precomputed reflections. */
template <typename BaseNodes>
[[nodiscard]] auto make_precomputed_extended_nodes(
    BaseNodes nodes,
    Boundary<
        BaseNodes::DimensionAtCompileTime,
        typename BaseNodes::Scalar,
        typename BaseNodes::Index> boundary)
    -> PrecomputedExtendedVoronoiNodes<BaseNodes> {
    return PrecomputedExtendedVoronoiNodes<BaseNodes>(
        std::move(nodes),
        std::move(boundary));
}

} // namespace highvoronoi
