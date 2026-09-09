#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <highvoronoi/core/point.hpp>
#include <highvoronoi/core/boundary.hpp>

namespace highvoronoi {

/**
 * @file voronoi_nodes.hpp
 * @brief Unified node interface for stored, computed, composite and reflected nodes.
 *
 * All node providers expose exactly the same public point semantics:
 *
 * - `operator[]`, `at()`, `node()` and `node_at()` return one owning Point type
 *   determined only by Scalar and Dimension;
 * - `get_data(node, coordinate)` is the coordinate-wise hotpath used by search
 *   backends such as nanoflann;
 * - `copy_node(node, target)` writes directly into caller-owned reusable storage.
 *
 * Whether a concrete provider stores coordinates, computes them on demand or
 * routes between several sources is an implementation detail of that provider.
 * It is deliberately not represented in the node type hierarchy.
 */

/** @brief Reflection-storage strategy used by extended node containers. */
enum class MirrorStorageMode {
    ActiveCell,
    AllNodes
};

// ============================================================================
// AbstractVoronoiNodes
// ============================================================================

/**
 * @brief Common polymorphic interface for every Voronoi node provider.
 *
 * The interface is intentionally small. Algorithms that need complete nodes
 * should reuse caller-owned storage through `copy_node()`. Algorithms that need
 * individual coordinates should use `get_data()`. Owning Point-returning access
 * is provided only as a convenience layer.
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
    using Point = PointType<Scalar, Dim>;

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
     * cell.
     */
    [[nodiscard]] virtual Index
    active_boundary_size() const noexcept {
        return Index{0};
    }

    /** @brief Return one active boundary node in public node numbering. */
    [[nodiscard]] virtual Index
    active_boundary_index(Index /* position */) const {
        throw std::out_of_range(
            "Node container has no active boundary-extension nodes.");
    }

    /**
     * @brief Return one coordinate of one node.
     *
     * Concrete hotpath providers should override this directly. The default
     * implementation preserves the small universal contract by materializing
     * one Point and is therefore intended only as a compatibility fallback.
     */
    [[nodiscard]] virtual Scalar get_data(
        Index node_index,
        Index coordinate) const {
        if (coordinate >= dimension_) {
            throw std::out_of_range("Voronoi-node coordinate out of range.");
        }
        const Point result = node(node_index);
        return result[static_cast<Eigen::Index>(coordinate)];
    }

    /**
     * @brief Copy node `index` into caller-owned contiguous storage.
     *
     * `target` must point to at least `dimension()` writable Scalar entries.
     */
    void copy_node(Index index, Scalar* target) const {
        copy_node_impl(index, target);
    }

    /** @brief Copy node `index` into an owning Point object. */
    void copy_node(Index index, Point& target) const {
        if constexpr (IsDynamic) {
            target.resize(static_cast<Eigen::Index>(dimension_));
        }
        copy_node_impl(index, target.data());
    }

    /** @brief Materialize node `index` as the common owning Point type. */
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

    /** @brief Convenience synonym for `node(index)`. */
    [[nodiscard]] Point operator[](Index index) const {
        return node(index);
    }

    /** @brief Bounds-checked convenience synonym for `node_at(index)`. */
    [[nodiscard]] Point at(Index index) const {
        return node_at(index);
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

    /** @brief Create an owning Point with the correct runtime dimension. */
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

    /** @brief Concrete node provider writes one complete node here. */
    virtual void copy_node_impl(Index index, Scalar* target) const = 0;

    virtual void activate_cell_impl(
        Index /* cell */,
        const Index* /* neighbours */,
        Index /* neighbour_count */) {}

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

    Index dimension_;
};

// ============================================================================
// VoronoiNodes: ordinary stored nodes
// ============================================================================

/**
 * @brief Owning contiguous storage for ordinary Voronoi nodes.
 *
 * Coordinates are stored node by node in one `std::vector<Scalar>`. Generic
 * access follows the same owning-Point/copy interface as every other provider.
 * `stable_node_data()` remains available only as a concrete storage-specific
 * optimization; generic mesh and geometry code does not depend on it.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class VoronoiNodes
    : public AbstractVoronoiNodes<ScalarT, Dim, IndexT> {
public:
    using Base = AbstractVoronoiNodes<ScalarT, Dim, IndexT>;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Point = typename Base::Point;

    static constexpr int DimensionAtCompileTime = Dim;
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

    /** @brief Return stable storage for concrete storage-aware code. */
    [[nodiscard]] const Scalar* stable_node_data(Index index) const noexcept {
        return data_.data() + index * this->dimension();
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
    void copy_node_impl(Index index, Scalar* target) const override {
        std::copy_n(
            stable_node_data(index),
            static_cast<std::size_t>(this->dimension()),
            target);
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
// CompositeVoronoiNodes: computed engine ranges + explicitly stored nodes
// ============================================================================

/**
 * @brief Stable global node space backed by engine ranges and stored nodes.
 *
 * The public node contract is identical to every other provider: coordinate
 * access uses get_data(), complete nodes use caller-owned copy_node(), and
 * convenience Point access is inherited from AbstractVoronoiNodes.  The fact
 * that some global indices are supplied by compute engines and others by
 * compact stored coordinates is private routing state, not a node access mode.
 *
 * `is_stored()` is intentionally a concrete-provider query used by
 * CompositeSearchTree to split immutable structured nodes from the mutable
 * sparse overlay. Generic geometry code does not depend on it.
 */
template <typename ScalarT,
          int Dim,
          typename IndexT = std::size_t>
class CompositeVoronoiNodes final
    : public AbstractVoronoiNodes<ScalarT, Dim, IndexT> {
public:
    using Base = AbstractVoronoiNodes<ScalarT, Dim, IndexT>;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Point = typename Base::Point;
    using StoredNodes = VoronoiNodes<Scalar, Dim, Index>;

    static constexpr int DimensionAtCompileTime = Dim;
    static constexpr bool IsDynamic = (Dim == Dynamic);

    explicit CompositeVoronoiNodes(Index runtime_dimension)
        : Base(checked_dimension(runtime_dimension)),
          stored_nodes_(Index{0}, checked_dimension(runtime_dimension)) {}

    [[nodiscard]] Index size() const noexcept override {
        return size_;
    }

    [[nodiscard]] Scalar get_data(
        Index node,
        Index coordinate) const override {
        const Segment& segment = segment_for(node);
        const Index local = static_cast<Index>(node - segment.begin);
        if (segment.kind == SegmentKind::Stored) {
            return stored_nodes_.get_data(
                static_cast<Index>(segment.source_begin + local),
                coordinate);
        }
        return engines_.at(segment.source_begin)->get_data(local, coordinate);
    }

    /** @brief Return true iff this global node belongs to stored overlay data. */
    [[nodiscard]] bool is_stored(Index node) const {
        return segment_for(node).kind == SegmentKind::Stored;
    }

    /** @brief Return true iff this global node belongs to a retained engine. */
    [[nodiscard]] bool is_structured(Index node) const {
        return !is_stored(node);
    }

    /** @brief Append one explicitly stored node and return its stable index. */
    template <class PointLike>
    [[nodiscard]] Index append_stored(const PointLike& point) {
        if (static_cast<Index>(point.size()) != this->dimension()) {
            throw std::invalid_argument(
                "Point dimension does not match CompositeVoronoiNodes dimension.");
        }
        require_capacity(Index{1});

        const Index global = size_;
        const Index stored_local = stored_nodes_.size();
        stored_nodes_.resize(static_cast<Index>(stored_local + Index{1}));
        stored_nodes_.set(stored_local, point);

        if (!segments_.empty() &&
            segments_.back().kind == SegmentKind::Stored &&
            segments_.back().end == global &&
            segments_.back().source_begin +
                    static_cast<std::size_t>(
                        segments_.back().end - segments_.back().begin) ==
                static_cast<std::size_t>(stored_local)) {
            segments_.back().end = static_cast<Index>(global + Index{1});
        } else {
            segments_.push_back(Segment{
                global,
                static_cast<Index>(global + Index{1}),
                SegmentKind::Stored,
                static_cast<std::size_t>(stored_local)});
        }

        size_ = static_cast<Index>(size_ + Index{1});
        return global;
    }

    /**
     * @brief Append one compute-engine node range without materializing it.
     *
     * The returned value is the first stable global index of the retained
     * engine range.
     */
    template <class EngineT>
    [[nodiscard]] Index append_engine(std::shared_ptr<EngineT> engine) {
        static_assert(
            std::is_same_v<typename EngineT::NodeScalar, Scalar>,
            "Engine NodeScalar must match CompositeVoronoiNodes Scalar.");
        static_assert(
            std::is_same_v<typename EngineT::Index, Index>,
            "Engine Index must match CompositeVoronoiNodes Index.");
        static_assert(
            EngineT::DimensionAtCompileTime == Dim,
            "Engine and CompositeVoronoiNodes dimensions must match.");

        if (!engine) {
            throw std::invalid_argument(
                "CompositeVoronoiNodes requires a non-null engine.");
        }
        if (engine->dimension() != this->dimension()) {
            throw std::invalid_argument(
                "Engine dimension does not match CompositeVoronoiNodes.");
        }

        const Index count = engine->node_count();
        require_capacity(count);
        const Index begin = size_;
        if (count == Index{0}) {
            return begin;
        }

        const std::size_t engine_index = engines_.size();
        engines_.push_back(
            std::make_shared<EngineModel<EngineT>>(std::move(engine)));
        segments_.push_back(Segment{
            begin,
            static_cast<Index>(begin + count),
            SegmentKind::Engine,
            engine_index});
        size_ = static_cast<Index>(begin + count);
        return begin;
    }

protected:
    void copy_node_impl(Index node, Scalar* target) const override {
        const Segment& segment = segment_for(node);
        const Index local = static_cast<Index>(node - segment.begin);
        if (segment.kind == SegmentKind::Stored) {
            stored_nodes_.copy_node(
                static_cast<Index>(segment.source_begin + local),
                target);
            return;
        }
        engines_.at(segment.source_begin)->copy_node(local, target);
    }

private:
    enum class SegmentKind : std::uint8_t {
        Stored,
        Engine
    };

    class EngineConcept {
    public:
        virtual ~EngineConcept() = default;
        [[nodiscard]] virtual Scalar get_data(
            Index local_node,
            Index coordinate) const = 0;
        virtual void copy_node(Index local_node, Scalar* target) const = 0;
    };

    template <class EngineT>
    class EngineModel final : public EngineConcept {
    public:
        explicit EngineModel(std::shared_ptr<EngineT> engine)
            : engine_(std::move(engine)) {}

        [[nodiscard]] Scalar get_data(
            Index local_node,
            Index coordinate) const override {
            return engine_->get_data(local_node, coordinate);
        }

        void copy_node(Index local_node, Scalar* target) const override {
            engine_->copy_node(local_node, target);
        }

    private:
        std::shared_ptr<EngineT> engine_;
    };

    struct Segment {
        Index begin = Index{0};
        Index end = Index{0};
        SegmentKind kind = SegmentKind::Stored;
        std::size_t source_begin = 0;
    };

    [[nodiscard]] static Index checked_dimension(Index dimension) {
        if constexpr (IsDynamic) {
            if (dimension == Index{0}) {
                throw std::invalid_argument(
                    "Dynamic CompositeVoronoiNodes requires positive dimension.");
            }
            return dimension;
        } else {
            if (dimension != static_cast<Index>(Dim)) {
                throw std::invalid_argument(
                    "Runtime dimension does not match CompositeVoronoiNodes Dim.");
            }
            return static_cast<Index>(Dim);
        }
    }

    void require_capacity(Index additional) const {
        if (additional > (std::numeric_limits<Index>::max)() - size_) {
            throw std::length_error(
                "CompositeVoronoiNodes global node index overflow.");
        }
    }

    [[nodiscard]] const Segment& segment_for(Index node) const {
        if (node >= size_) {
            throw std::out_of_range(
                "CompositeVoronoiNodes node index out of range.");
        }
        const auto found = std::upper_bound(
            segments_.begin(),
            segments_.end(),
            node,
            [](Index value, const Segment& segment) {
                return value < segment.begin;
            });
        if (found == segments_.begin()) {
            throw std::logic_error(
                "CompositeVoronoiNodes segment routing is inconsistent.");
        }
        const Segment& segment = *std::prev(found);
        if (node >= segment.end) {
            throw std::logic_error(
                "CompositeVoronoiNodes segment routing is inconsistent.");
        }
        return segment;
    }

    Index size_ = Index{0};
    StoredNodes stored_nodes_;
    std::vector<std::shared_ptr<EngineConcept>> engines_;
    std::vector<Segment> segments_;
};

// ============================================================================
// Extended nodes
// ============================================================================

namespace detail {

template <typename BaseNodes,
          MirrorStorageMode StorageMode>
class ExtendedVoronoiNodesImpl
    : public AbstractVoronoiNodes<
          typename BaseNodes::Scalar,
          BaseNodes::DimensionAtCompileTime,
          typename BaseNodes::Index> {
public:
    using Scalar = typename BaseNodes::Scalar;
    using Index = typename BaseNodes::Index;
    static constexpr int DimensionAtCompileTime =
        BaseNodes::DimensionAtCompileTime;
    using Base = AbstractVoronoiNodes<Scalar, DimensionAtCompileTime, Index>;
    using Point = typename Base::Point;
    using BoundaryType = Boundary<
        DimensionAtCompileTime,
        Scalar,
        Index>;
    using MirrorNodes = VoronoiNodes<
        Scalar,
        DimensionAtCompileTime,
        Index>;

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
        : Base(nodes.dimension()),
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

    [[nodiscard]] Point precomputed_reflection(
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

    void copy_node_impl(Index index, Scalar* target) const override {
        if (index < inner_size()) {
            nodes_.copy_node(index, target);
        } else {
            mirrors_.copy_node(index - inner_size(), target);
        }
    }

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

private:
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
            const Point reflection =
                precomputed_[precomputed_index(cell, plane)];
            mirrors_.set(plane, reflection);
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

/** @brief Add cell-local boundary reflections to any node provider. */
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

    /** @brief Create an independent cell-local copy. */
    [[nodiscard]] ExtendedVoronoiNodes safe_copy() const {
        static_assert(
            std::is_copy_constructible_v<BaseNodes>,
            "ExtendedVoronoiNodes::safe_copy requires copyable BaseNodes.");
        return ExtendedVoronoiNodes(
            this->inner_nodes(),
            this->boundary());
    }
};

/** @brief Extended nodes with every node/plane reflection precomputed. */
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
    using Point = typename Base::Point;
    using MirrorNodes = typename Base::MirrorNodes;

    using Base::Base;

    [[nodiscard]] const MirrorNodes&
    precomputed_nodes() const noexcept {
        return this->all_reflections();
    }

    [[nodiscard]] Point reflected_node(
        Index cell,
        Index plane) const {
        return this->precomputed_reflection(cell, plane);
    }

    /**
     * @brief Safe copying of precomputed storage is deliberately unresolved.
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
