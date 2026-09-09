#pragma once

/**
 * @file edge_iterator.hpp
 * @brief Julia-faithful Voronoi edge iterator for general and degenerate vertices.
 *
 * This file ports the two Julia edge iterators behind one C++ interface:
 *
 * - `General_EdgeIterator` for a vertex with exactly `dimension + 1`
 *   generators;
 * - `FastEdgeIterator` for a degenerate vertex with more generators.
 *
 * As in Julia, general-position iteration itself is purely topological:
 * it yields the d-generator edge key plus the omitted generator.  The `u_qr`
 * direction and its error/fallback are materialized lazily only when a caller
 * asks for `direction()` or `cycle_error()`.  Queue/hash-only passes therefore
 * never factor a QR for general-position edges.
 *
 * The degenerate branch follows the Julia algorithm except for one deliberate
 * correction in `scan_for_edge`: a primary generator is rejected only after
 * every admissible supporting-face completion has been tried.  The Julia code
 * rejects the primary after one failed greedy completion, which is weaker than
 * the existence test required by Lemma 2.18 of the HighVoronoi paper.
 *
 * Apart from that correction, the implementation keeps:
 *
 * 1. the hierarchy of projected FEI levels `d, d-1, ..., 2`;
 * 2. `scan_for_edge` and the recursive discovery of valid rays;
 * 3. `rotate2` during the scan phase;
 * 4. the combination walk of `update_edge`;
 * 5. two consecutive `rotate` calls for every Gram-Schmidt update there;
 * 6. the Julia `get_full_edge` half-space test and orientation;
 * 7. the final cycle-error check and, only when necessary, QR correction of
 *    the already constructed basis.
 *
 * Numerical scratch buffers remain owned and reused by EdgeIterator.  The
 * expensive top-level FEI initialization is cached in a shared
 * nested `FEIStorageCache`, matching Julia's `FEIStorage_global`. Unused FEIData
 * geometry buffers remain omitted, while
 * the complete four-entry FEIData `index` state is retained for cache parity.
 */

#include <highvoronoi/algorithm/normal_solver.hpp>
#include <highvoronoi/core/detail/locks.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <functional>
#include <unordered_map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/** Small borrowed contiguous index range, valid until the iterator advances. */
template <typename Index>
class EdgeIndexView {
public:
    EdgeIndexView() = default;

    EdgeIndexView(const Index* data, std::size_t size)
        : data_(data), size_(size) {}

    [[nodiscard]] const Index* begin() const noexcept { return data_; }
    [[nodiscard]] const Index* end() const noexcept { return data_ + size_; }
    [[nodiscard]] const Index* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] const Index& operator[](std::size_t position) const noexcept {
        return data_[position];
    }

private:
    const Index* data_{nullptr};
    std::size_t size_{0};
};

/** Borrowed result of one EdgeIterator increment. */
template <typename Scalar, int Dim, typename Index>
class EdgeView {
public:
    using Point = detail::NormalPoint<Scalar, Dim>;
    using MaterializeFunction = void (*)(void*);

    EdgeView(
        EdgeIndexView<Index> minimal_indices,
        EdgeIndexView<Index> full_indices,
        Index skip,
        const Point& direction,
        const Scalar& cycle_error,
        void* materialize_context = nullptr,
        MaterializeFunction materialize_function = nullptr)
        : minimal_indices_(minimal_indices),
          full_indices_(full_indices),
          skip_(skip),
          direction_(&direction),
          cycle_error_(&cycle_error),
          materialize_context_(materialize_context),
          materialize_function_(materialize_function) {}

    /** Minimal d-generator edge key yielded by the Julia iterator. */
    [[nodiscard]] EdgeIndexView<Index> indices() const noexcept {
        return minimal_indices_;
    }

    /** Explicit name for the same minimal generator set. */
    [[nodiscard]] EdgeIndexView<Index> minimal_indices() const noexcept {
        return minimal_indices_;
    }

    /** Complete set of vertex generators lying in the edge hyperplane. */
    [[nodiscard]] EdgeIndexView<Index> full_indices() const noexcept {
        return full_indices_;
    }

    /**
     * Global generator omitted from the complete supporting edge.
     *
     * Both iterator branches expose the same semantics here: `skip()` is a
     * generator index from the current vertex signature, not a local position
     * inside an FEI buffer. Consequently `skip()` must not occur in
     * `full_indices()`.
     */
    [[nodiscard]] Index skip() const noexcept { return skip_; }

    /**
     * Return the geometric edge direction.
     *
     * General-position views materialize their Julia `u_qr` direction lazily
     * on first access.  Queue/hash-only users therefore pay no QR cost.
     * Degenerate FastEdgeIterator views are already materialized by construction.
     */
    [[nodiscard]] const Point& direction() const {
        materialize_if_needed();
        return *direction_;
    }

    /** Return the direction orthogonality/cycle error, materializing if needed. */
    [[nodiscard]] Scalar cycle_error() const {
        materialize_if_needed();
        return *cycle_error_;
    }

    [[nodiscard]] bool is_degenerate() const noexcept {
        return full_indices_.size() > minimal_indices_.size();
    }

private:
    void materialize_if_needed() const {
        if (materialize_function_ != nullptr) {
            materialize_function_(materialize_context_);
        }
    }

    EdgeIndexView<Index> minimal_indices_;
    EdgeIndexView<Index> full_indices_;
    Index skip_{};
    const Point* direction_{nullptr};
    const Scalar* cycle_error_{nullptr};
    void* materialize_context_{nullptr};
    MaterializeFunction materialize_function_{nullptr};
};

/** Owning copy of EdgeView, useful in tests and persistent diagnostics. */
template <typename Scalar, int Dim, typename Index>
struct EdgeCandidate {
    using Point = detail::NormalPoint<Scalar, Dim>;

    std::vector<Index> indices;
    std::vector<Index> full_indices;
    Index skip{};
    Point direction;
    Scalar cycle_error{0};

    explicit EdgeCandidate(std::size_t dimension)
        : direction(detail::make_normal_point<Scalar, Dim>(dimension)) {}

    explicit EdgeCandidate(const EdgeView<Scalar, Dim, Index>& view)
        : indices(view.indices().begin(), view.indices().end()),
          full_indices(
              view.full_indices().begin(),
              view.full_indices().end()),
          skip(view.skip()),
          direction(view.direction()),
          cycle_error(view.cycle_error()) {}

    [[nodiscard]] bool is_degenerate() const noexcept {
        return full_indices.size() > indices.size();
    }
};

namespace detail {

/** Julia constants converted to explicit C++ names. */
template <typename Scalar>
struct EdgeIteratorTolerance {
    Scalar ray{Scalar{1e-12}};
    Scalar angle{Scalar{1e-12}};
    Scalar correction{Scalar{1e-13}};
};

} // namespace detail

/**
 * Stateful edge iterator bound to one node container.
 *
 * Only a node reference is required from the surrounding Voronoi algorithm.
 * Numerical scratch memory lives in this object and is retained across
 * reset/next cycles.  Expensive FEI initialization state is exchanged through
 * the shared FEIStorageCache owned by this iterator family.
 */
template <class Nodes, class FEILock = detail::EmptyLock>
class EdgeIterator final {
public:
    using Scalar = typename Nodes::Scalar;
    using Index = typename Nodes::Index;

    static constexpr int DimensionAtCompileTime =
        Nodes::DimensionAtCompileTime;

    using Point =
        detail::NormalPoint<Scalar, DimensionAtCompileTime>;
    using View =
        EdgeView<Scalar, DimensionAtCompileTime, Index>;
    using Candidate =
        EdgeCandidate<Scalar, DimensionAtCompileTime, Index>;

    /**
     * Cached result of the expensive top-level FastEdgeIterator
     * initialization.
     *
     * This is the direct C++ counterpart of Julia's FEIStorage.  It remains
     * nested because it has no meaning outside this EdgeIterator type.
     * The full Julia payload is deliberately retained for now so individual
     * fields can later be removed experimentally without changing the
     * iterator interface.
     */
    class FEIStorage final {
    public:
        /** Julia: MVector{5,Int64}; entry 4 is cache/cell marker. */
        std::array<std::size_t, 5> index{};

        /** Julia: free_nodes::Vector{Bool}. */
        std::vector<std::uint8_t> free_nodes;

        /** Julia: valid_nodes::Vector{Bool}. */
        std::vector<std::uint8_t> valid_nodes;

        /** Julia: active_nodes::MVector{dim,Int64}. */
        std::vector<std::size_t> active_nodes;

        FEIStorage() = default;

        FEIStorage(std::size_t signature_size, std::size_t dimension)
            : free_nodes(signature_size),
              valid_nodes(signature_size),
              active_nodes(dimension) {}

        [[nodiscard]] bool initialized() const noexcept {
            return index[4] != 0;
        }

        /** Equivalent of Julia reset(f::FEIStorage). */
        void invalidate() noexcept {
            index[4] = 0;
        }

        [[nodiscard]] Index stored_cell() const {
            if (!initialized()) {
                throw std::logic_error(
                    "FEIStorage has no stored cell because the cache is invalid.");
            }

            const std::size_t zero_based = index[4] - 1;
            if (zero_based >
                static_cast<std::size_t>((std::numeric_limits<Index>::max)())) {
                throw std::overflow_error(
                    "Stored FEI cell does not fit into the public Index type.");
            }
            return static_cast<Index>(zero_based);
        }

        template <class Level>
        void store(const Level& level, Index cell) {
            if (level.index.size() != 4) {
                throw std::logic_error(
                    "FEI level must expose exactly four Julia index values.");
            }

            const std::size_t encoded_cell =
                static_cast<std::size_t>(cell) + std::size_t{1};
            if (encoded_cell == 0) {
                throw std::overflow_error(
                    "FEI cell cannot be encoded with a non-zero cache marker.");
            }

            for (std::size_t i = 0; i < 4; ++i) {
                index[i] = level.index[i];
            }
            index[4] = encoded_cell;

            free_nodes.assign(
                level.free_nodes.begin(),
                level.free_nodes.end());
            valid_nodes.assign(
                level.valid_nodes.begin(),
                level.valid_nodes.end());
            active_nodes.assign(
                level.active_nodes.begin(),
                level.active_nodes.end());
        }

        template <class Level>
        void load(Level& level) const {
            if (!initialized()) {
                throw std::logic_error(
                    "Cannot load an invalid FEIStorage.");
            }
            if (level.index.size() != 4) {
                throw std::logic_error(
                    "FEI level must expose exactly four Julia index values.");
            }
            if (free_nodes.size() != level.free_nodes.size() ||
                valid_nodes.size() != level.valid_nodes.size() ||
                active_nodes.size() != level.active_nodes.size()) {
                throw std::logic_error(
                    "FEIStorage dimensions do not match the current FEI level.");
            }

            for (std::size_t i = 0; i < 4; ++i) {
                level.index[i] = index[i];
            }

            std::copy(
                free_nodes.begin(),
                free_nodes.end(),
                level.free_nodes.begin());
            std::copy(
                valid_nodes.begin(),
                valid_nodes.end(),
                level.valid_nodes.begin());
            std::copy(
                active_nodes.begin(),
                active_nodes.end(),
                level.active_nodes.begin());
        }
    };

    /**
     * Signature-keyed shared FEI cache.
     *
     * Several EdgeIterator instances may share one cache while keeping their
     * complete numerical scratch state independent.  This mirrors Julia's
     * FEIStorage_global owned by the RayCaster.
     */
    class FEIStorageCache final {
    public:
        using Key = std::vector<Index>;
        using Storage = FEIStorage;
        using LockType = FEILock;

        explicit FEIStorageCache(std::size_t dimension)
            : dimension_(dimension) {
            if (dimension_ < 2) {
                throw std::invalid_argument(
                    "FEIStorageCache requires dimension >= 2.");
            }
        }

        FEIStorageCache(const FEIStorageCache&) = delete;
        FEIStorageCache& operator=(const FEIStorageCache&) = delete;
        FEIStorageCache(FEIStorageCache&&) = delete;
        FEIStorageCache& operator=(FEIStorageCache&&) = delete;

        Storage& get_or_create(const Key& sigma) {
            detail::WriteLockGuard<FEILock> guard(lock_);
            const auto found = entries_.find(sigma);
            if (found != entries_.end()) {
                return found->second;
            }

            auto inserted = entries_.emplace(
                sigma,
                Storage(sigma.size(), dimension_));
            return inserted.first->second;
        }

        template <class Sigma>
        Storage& get_or_create(const Sigma& sigma) {
            Key key(sigma.begin(), sigma.end());
            return get_or_create(key);
        }

        Storage* find(const Key& sigma) {
            detail::ReadLockGuard<FEILock> guard(lock_);
            const auto found = entries_.find(sigma);
            return found == entries_.end() ? nullptr : &found->second;
        }

        const Storage* find(const Key& sigma) const {
            detail::ReadLockGuard<FEILock> guard(lock_);
            const auto found = entries_.find(sigma);
            return found == entries_.end() ? nullptr : &found->second;
        }

        template <class Sigma>
        Storage* find(const Sigma& sigma) {
            Key key(sigma.begin(), sigma.end());
            return find(key);
        }

        template <class Sigma>
        const Storage* find(const Sigma& sigma) const {
            Key key(sigma.begin(), sigma.end());
            return find(key);
        }

        [[nodiscard]] std::size_t size() const {
            detail::ReadLockGuard<FEILock> guard(lock_);
            return entries_.size();
        }

        [[nodiscard]] bool empty() const {
            return size() == 0;
        }

        [[nodiscard]] std::size_t dimension() const noexcept {
            return dimension_;
        }

        void reserve(std::size_t count) {
            detail::WriteLockGuard<FEILock> guard(lock_);
            entries_.reserve(count);
        }

        void clear() {
            detail::WriteLockGuard<FEILock> guard(lock_);
            entries_.clear();
        }

    private:
        struct KeyHash {
            [[nodiscard]] std::size_t operator()(const Key& key) const noexcept {
                std::size_t hash = 0;
                for (const Index value : key) {
                    const std::size_t word = std::hash<Index>{}(value);
                    hash ^= word + std::size_t{0x9e3779b9U}
                          + (hash << 6U) + (hash >> 2U);
                }
                return hash;
            }
        };

        std::size_t dimension_;
        std::unordered_map<Key, Storage, KeyHash> entries_;
        mutable FEILock lock_{};
    };

    using Storage = FEIStorage;
    using StorageCache = FEIStorageCache;
    using StorageCacheHandle = std::shared_ptr<FEIStorageCache>;

    struct OnQueueEdges {};
    struct OnSysVoronoi {};

    /**
     * Cell-local analysis mode.
     *
     * Unlike OnQueueEdges/OnSysVoronoi this mode does not apply the global
     * edge-ownership split between generators. It exposes every geometric edge
     * of the current vertex that belongs to `cell` and leaves the top-level
     * FEI valid-node mask untruncated. The shared FEI cache is deliberately not
     * touched so neighbour analysis cannot perturb ComputeVoronoi cache state.
     */
    struct OnCellEdges {};

    enum class FEIAction {
        None,
        Reinitialized,
        Loaded,
        SkippedByCellOwnership
    };

    explicit EdgeIterator(
        const Nodes& nodes,
        Scalar ray_tolerance = Scalar{1e-12})
        : EdgeIterator(
              nodes,
              std::make_shared<StorageCache>(
                  static_cast<std::size_t>(nodes.dimension())),
              ray_tolerance) {}

    EdgeIterator(
        const Nodes& nodes,
        StorageCacheHandle storage_cache,
        Scalar ray_tolerance = Scalar{1e-12})
        : nodes_(&nodes),
          dimension_(static_cast<std::size_t>(nodes.dimension())),
          storage_cache_(std::move(storage_cache)),
          tolerances_{ray_tolerance, Scalar{1e-12}, Scalar{1e-13}},
          vertex_position_(detail::make_normal_point<
              Scalar,
              DimensionAtCompileTime>(dimension_)),
          direction_(detail::make_normal_point<Scalar, DimensionAtCompileTime>(
              dimension_)),
          qr_solver_(dimension_),
          extended_qr_solver_(dimension_),
          qr_direction_(detail::make_normal_point<Scalar, DimensionAtCompileTime>(
              dimension_)),
          node_buffer_(detail::make_normal_point<Scalar, DimensionAtCompileTime>(
              dimension_)),
          random_(0x48564f524f4e4f49ULL),
          uniform_(Scalar{0}, Scalar{1}) {

        if (dimension_ < 2) {
            throw std::invalid_argument(
                "EdgeIterator requires dimension >= 2.");
        }
        if (!storage_cache_) {
            throw std::invalid_argument(
                "EdgeIterator requires a valid shared FEIStorageCache.");
        }
        if (storage_cache_->dimension() != dimension_) {
            throw std::invalid_argument(
                "FEIStorageCache dimension does not match EdgeIterator.");
        }

        levels_.reserve(dimension_ - 1);
        for (std::size_t step = 0; step + 1 < dimension_; ++step) {
            levels_.emplace_back(
                dimension_,
                dimension_ - step);
        }

        minimal_result_.reserve(dimension_);
        full_result_.reserve(dimension_ + 1);
    }

    EdgeIterator(const EdgeIterator&) = delete;
    EdgeIterator& operator=(const EdgeIterator&) = delete;
    EdgeIterator(EdgeIterator&&) = delete;
    EdgeIterator& operator=(EdgeIterator&&) = delete;

    /** Create a second iterator with independent scratch state and the same FEI cache. */
    [[nodiscard]] std::unique_ptr<EdgeIterator> create_shared() const {
        return std::make_unique<EdgeIterator>(
            *nodes_,
            storage_cache_,
            tolerances_.ray);
    }

    [[nodiscard]] StorageCache& storage_cache() noexcept {
        return *storage_cache_;
    }

    [[nodiscard]] const StorageCache& storage_cache() const noexcept {
        return *storage_cache_;
    }

    [[nodiscard]] StorageCacheHandle storage_cache_handle() const noexcept {
        return storage_cache_;
    }

    /**
     * Julia OnQueueEdges semantics.
     *
     * For a degenerate vertex the shared sigma-cache entry is obtained,
     * invalidated, and then recomputed for the current cell.  General-position
     * vertices do not use FEIStorage.
     */
    template <class Sigma, class PointLike>
    void reset(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell,
        OnQueueEdges) {
        reset_from_shared_cache(
            sigma,
            vertex_position,
            cell,
            true);
    }

    /**
     * Julia OnSysVoronoi semantics.
     *
     * For a degenerate vertex the shared sigma-cache entry is loaded when it
     * is initialized.  If no cached initialization exists yet, it is computed
     * once and stored.  General-position vertices do not use FEIStorage.
     */
    template <class Sigma, class PointLike>
    void reset(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell,
        OnSysVoronoi) {
        reset_from_shared_cache(
            sigma,
            vertex_position,
            cell,
            false);
    }

    /**
     * Reset for cell-local geometry analysis without edge ownership truncation.
     *
     * General-position vertices return all d edges incident to `cell`.
     * Degenerate vertices run the existing FEI reduction with `all_rays=true`
     * and `cell_first=true`. No FEIStorage entry is read, invalidated or written.
     */
    template <class Sigma, class PointLike>
    void reset(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell,
        OnCellEdges) {
        reset_cell_analysis(sigma, vertex_position, cell);
    }

    /** Return the next borrowed edge or nullopt after exhaustion. */
    [[nodiscard]] std::optional<View> next() {
        if (mode_ == Mode::General) {
            return next_general();
        }
        if (mode_ == Mode::Degenerate && fast_ready_) {
            return next_fast();
        }
        return std::nullopt;
    }

    [[nodiscard]] std::size_t dimension() const noexcept {
        return dimension_;
    }

    [[nodiscard]] Index cell() const noexcept {
        return cell_;
    }

    [[nodiscard]] FEIAction last_fei_action() const noexcept {
        return last_fei_action_;
    }

    [[nodiscard]] const Point& vertex_position() const noexcept {
        return vertex_position_;
    }

    /** Number of valid top-level rays found by the Julia FEI reset. */
    [[nodiscard]] std::size_t valid_ray_count() const noexcept {
        if (levels_.empty()) {
            return 0;
        }
        return levels_.front().valid_rays();
    }

    /** Invoke `function(global_generator)` for every top-level FEI valid node. */
    template <class Function>
    void for_each_valid_generator(Function&& function) const {
        if (mode_ != Mode::Degenerate || levels_.empty()) {
            return;
        }

        const FastLevel& level = levels_.front();
        for (std::size_t position = 0;
             position < level.sig.size();
             ++position) {
            if (level.valid_nodes[position] != 0) {
                function(level.sig[position]);
            }
        }
    }

    /** Global indices corresponding to the top-level Julia valid-node mask. */
    [[nodiscard]] std::vector<Index> valid_generators() const {
        std::vector<Index> result;
        for_each_valid_generator(
            [&](Index generator) { result.push_back(generator); });
        return result;
    }

    /** Copy valid generators into caller-owned recyclable storage. */
    void valid_generators(std::vector<Index>& result) const {
        result.clear();
        for_each_valid_generator(
            [&](Index generator) { result.push_back(generator); });
    }

private:
    static constexpr std::size_t npos =
        (std::numeric_limits<std::size_t>::max)();

    enum class Mode {
        None,
        General,
        Degenerate
    };

    /** C++ equivalent of the used portion of Julia FEIData. */
    struct FastLevel {
        std::size_t ambient_dim;
        std::size_t current_dim;

        std::vector<Index> sig;
        Point r;
        std::vector<Point> local_xs;
        std::vector<Point> local_cone;
        std::vector<Point> rays;

        std::vector<std::size_t> edge_buffer;
        std::vector<std::uint8_t> free_nodes;
        std::vector<std::uint8_t> valid_nodes;

        // Transient retry state used only by scan_for_edge.  One row per
        // Gram-Schmidt basis slot, one flag per signature position.  It is
        // deliberately not part of FEIStorage: retries never survive a
        // completed supporting-face search.
        std::vector<std::uint8_t> tried_candidates;

        std::vector<std::size_t> active_nodes;

        // Julia FEIData.index: [valid_rays, current_primary,
        // current_secondary, current_plane].  The latter two are retained
        // even though the active Julia path currently does not use them.
        std::array<std::size_t, 4> index{};

        [[nodiscard]] std::size_t& valid_rays() noexcept {
            return index[0];
        }

        [[nodiscard]] const std::size_t& valid_rays() const noexcept {
            return index[0];
        }

        [[nodiscard]] std::size_t& current_primary() noexcept {
            return index[1];
        }

        [[nodiscard]] const std::size_t& current_primary() const noexcept {
            return index[1];
        }

        FastLevel(std::size_t dim, std::size_t cdim)
            : ambient_dim(dim),
              current_dim(cdim),
              r(detail::make_normal_point<
                  Scalar,
                  DimensionAtCompileTime>(dim)) {

            rays.reserve(dim);
            for (std::size_t i = 0; i < dim; ++i) {
                rays.push_back(detail::make_normal_point<
                    Scalar,
                    DimensionAtCompileTime>(dim));
                rays.back().setZero();
            }

            active_nodes.assign(dim, npos);
        }

        void resize(std::size_t count) {
            sig.resize(count);
            edge_buffer.resize(count);
            free_nodes.resize(count);
            valid_nodes.resize(count);
            tried_candidates.resize(ambient_dim * count);

            resize_points(local_xs, count);
            resize_points(local_cone, count);
        }

    private:
        void resize_points(std::vector<Point>& points, std::size_t count) {
            const std::size_t old_size = points.size();
            points.resize(count);
            if constexpr (DimensionAtCompileTime == Dynamic) {
                for (std::size_t i = old_size; i < count; ++i) {
                    points[i].resize(static_cast<Eigen::Index>(ambient_dim));
                }
            }
        }
    };

    template <class Sigma, class PointLike>
    void reset_from_shared_cache(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell,
        bool invalidate_storage) {

        if (static_cast<std::size_t>(sigma.size()) == dimension_ + 1) {
            reset_impl(sigma, vertex_position, cell, nullptr);
            return;
        }

        Storage& storage = storage_cache_->get_or_create(sigma);
        if (invalidate_storage) {
            storage.invalidate();
        }
        reset_impl(sigma, vertex_position, cell, &storage);
    }

    template <class Sigma, class PointLike>
    void reset_cell_analysis(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell) {

        require_reset_input(sigma, vertex_position, cell);

        input_signature_.assign(sigma.begin(), sigma.end());
        cell_ = cell;
        copy_point(vertex_position, vertex_position_);

        minimal_result_.clear();
        full_result_.clear();
        direction_.setZero();
        cycle_error_ = Scalar{0};
        last_fei_action_ = FEIAction::None;
        cell_analysis_mode_ = true;

        if (input_signature_.size() == dimension_ + 1) {
            mode_ = Mode::General;
            reset_general();
            return;
        }

        mode_ = Mode::Degenerate;
        fast_ready_ = reset_level(
            0,
            input_signature_,
            vertex_position_,
            cell_,
            nullptr,
            nullptr,
            true,
            true);
    }

    template <class Sigma, class PointLike>
    void reset_impl(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell,
        Storage* storage) {

        require_reset_input(sigma, vertex_position, cell);

        input_signature_.assign(sigma.begin(), sigma.end());
        cell_ = cell;
        copy_point(vertex_position, vertex_position_);

        minimal_result_.clear();
        full_result_.clear();
        direction_.setZero();
        cycle_error_ = Scalar{0};
        last_fei_action_ = FEIAction::None;
        cell_analysis_mode_ = false;

        if (input_signature_.size() == dimension_ + 1) {
            mode_ = Mode::General;
            reset_general();
            return;
        }

        mode_ = Mode::Degenerate;
        if (storage == nullptr) {
            throw std::logic_error(
                "Degenerate EdgeIterator reset requires FEIStorage.");
        }
        fast_ready_ = reset_level_top(storage);
    }

    template <class Sigma, class PointLike>
    void require_reset_input(
        const Sigma& sigma,
        const PointLike& vertex_position,
        Index cell) const {

        if (static_cast<std::size_t>(vertex_position.size()) != dimension_) {
            throw std::invalid_argument(
                "Vertex dimension does not match EdgeIterator.");
        }
        if (static_cast<std::size_t>(sigma.size()) < dimension_ + 1) {
            throw std::invalid_argument(
                "Voronoi vertex needs at least dimension + 1 generators.");
        }
        if (!std::is_sorted(sigma.begin(), sigma.end())) {
            throw std::invalid_argument(
                "EdgeIterator requires a sorted vertex signature.");
        }
        if (std::adjacent_find(sigma.begin(), sigma.end()) != sigma.end()) {
            throw std::invalid_argument(
                "Vertex signature contains duplicate generators.");
        }
        if (!std::binary_search(sigma.begin(), sigma.end(), cell)) {
            throw std::invalid_argument(
                "Active cell is not contained in the vertex signature.");
        }
    }

    template <class PointLike>
    void copy_point(const PointLike& source, Point& target) const {
        if constexpr (DimensionAtCompileTime == Dynamic) {
            target.resize(static_cast<Eigen::Index>(dimension_));
        }
        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            target[static_cast<Eigen::Index>(coordinate)] =
                static_cast<Scalar>(
                    source[static_cast<Eigen::Index>(coordinate)]);
        }
    }

    void copy_global_node(Index index, Point& target) const {
        nodes_->copy_node(index, target.data());
    }

    static bool normalize(Point& point) {
        const Scalar norm = point.norm();
        if (!(norm > Scalar{0})) {
            return false;
        }
        point /= norm;
        return true;
    }

    // ---------------------------------------------------------------------
    // General_EdgeIterator
    // ---------------------------------------------------------------------

    void reset_general() {
        general_next_ = 0;
        general_end_ = 0;
        general_cell_position_ = npos;
        general_current_omitted_ = npos;
        general_direction_ready_ = false;

        if (cell_analysis_mode_) {
            const auto cell_it = std::find(
                input_signature_.begin(), input_signature_.end(), cell_);
            general_cell_position_ = static_cast<std::size_t>(
                std::distance(input_signature_.begin(), cell_it));
            general_end_ = input_signature_.size();
            return;
        }

        if (input_signature_[0] == cell_) {
            // Cell 0 owns every edge except the deletion of cell 0 itself.
            // Julia emits that one extra candidate and filters it outside the
            // iterator.  The C++ iterator enforces cell ownership internally.
            general_next_ = 1;
            general_end_ = input_signature_.size();
        } else if (input_signature_[1] == cell_) {
            // Cell 1 owns only the edge obtained by deleting cell 0.
            general_next_ = 0;
            general_end_ = 1;
        }
        // Cells at positions >= 2 own no edge of this vertex.
    }

    [[nodiscard]] std::optional<View> next_general() {
        while (cell_analysis_mode_ &&
               general_next_ < general_end_ &&
               general_next_ == general_cell_position_) {
            ++general_next_;
        }
        if (general_next_ >= general_end_) {
            return std::nullopt;
        }

        const std::size_t omitted = general_next_++;
        minimal_result_.clear();
        full_result_.clear();

        for (std::size_t position = 0;
             position < input_signature_.size();
             ++position) {
            if (position == omitted) {
                continue;
            }
            minimal_result_.push_back(input_signature_[position]);
            full_result_.push_back(input_signature_[position]);
        }

        // Julia General_EdgeIterator iteration returns only
        // (_mydeleteat(sig, omitted), sig[omitted]).  The expensive u_qr
        // direction is computed later by get_full_edge(), only when the edge
        // actually survives queue/hash/ownership filtering and is walked.
        general_current_omitted_ = omitted;
        general_direction_ready_ = false;

        return make_view(
            input_signature_[omitted],
            true);
    }

    /** Materialize the current general-position edge direction exactly once. */
    void materialize_general_direction() {
        if (general_direction_ready_) {
            return;
        }
        if (mode_ != Mode::General ||
            general_current_omitted_ == npos) {
            throw std::logic_error(
                "No general-position edge is available for direction materialization.");
        }

        const std::size_t omitted = general_current_omitted_;
        bool ok = qr_solver_.solve(
            *nodes_,
            input_signature_,
            omitted,
            direction_);
        cycle_error_ = ok
            ? qr_solver_.last_direction_error()
            : (std::numeric_limits<Scalar>::max)();

        if (!ok || cycle_error_ > tolerances_.correction) {
            ok = extended_qr_solver_.solve(
                *nodes_,
                input_signature_,
                omitted,
                direction_);
            cycle_error_ = ok
                ? extended_qr_solver_.last_direction_error()
                : (std::numeric_limits<Scalar>::max)();
        }
        if (!ok) {
            throw std::runtime_error(
                "General edge normal is singular even in extended precision.");
        }
        if (cycle_error_ > tolerances_.ray) {
            throw std::runtime_error(
                "Extended QR did not restore an orthogonal general edge direction.");
        }

        general_direction_ready_ = true;
    }

    static void materialize_general_direction_from_view(void* context) {
        static_cast<EdgeIterator*>(context)->materialize_general_direction();
    }


    // ---------------------------------------------------------------------
    // FastEdgeIterator reset hierarchy
    // ---------------------------------------------------------------------

    [[nodiscard]] bool reset_level_top(Storage* storage) {
        return reset_level(
            0,
            input_signature_,
            vertex_position_,
            cell_,
            nullptr,
            storage,
            false,
            false);
    }

    [[nodiscard]] bool reset_level(
        std::size_t step,
        const std::vector<Index>& input_sig,
        const Point& r,
        Index cell_value,
        FastLevel* parent,
        Storage* storage,
        bool all_rays,
        bool cell_first) {

        if (step >= levels_.size()) {
            throw std::logic_error(
                "FastEdgeIterator recursion exceeded FEI level storage.");
        }

        FastLevel& level = levels_[step];
        const std::size_t cdim = level.current_dim;
        const std::size_t lsig = input_sig.size();

        const auto cell_it =
            std::find(input_sig.begin(), input_sig.end(), cell_value);
        if (cell_it == input_sig.end()) {
            throw std::logic_error(
                "Recursive FEI signature does not contain its active cell.");
        }

        std::size_t cell_entry = static_cast<std::size_t>(
            std::distance(input_sig.begin(), cell_it));

        // Exact top-level Julia ownership shortcut:
        // _Cell_entry > lsig - dim in one-based indexing.
        if (step == 0 &&
            cell_entry > lsig - dimension_ &&
            !all_rays) {
            level.valid_rays() = 1;
            if (storage != nullptr) {
                last_fei_action_ = FEIAction::SkippedByCellOwnership;
            }
            return false;
        }

        level.resize(lsig);
        std::copy(input_sig.begin(), input_sig.end(), level.sig.begin());
        std::fill(level.active_nodes.begin(), level.active_nodes.end(), npos);

        std::swap(level.sig[0], level.sig[cell_entry]);
        constexpr std::size_t first_entry = 0;

        if (parent != nullptr) {
            build_recursive_coordinates(level, *parent, r);
        } else {
            build_top_coordinates(level, r);
        }

        std::fill(level.free_nodes.begin(), level.free_nodes.end(), 1);
        level.free_nodes[first_entry] = 0;

        // Julia FEIStorage wraps only the expensive valid-ray discovery.
        // Local coordinates are rebuilt before this branch on every reset.
        if (storage == nullptr || !storage->initialized()) {
            if (step == 0 && storage != nullptr) {
                last_fei_action_ = FEIAction::Reinitialized;
            }
            level.active_nodes[0] = first_entry;
            level.current_primary() = first_entry + 1;

            compute_valid_nodes(step, storage);

            // Julia stores only at the top FEI level (dim == cdim).
            if (step == 0 && storage != nullptr) {
                storage->store(level, cell_value);
            }
        } else {
            // An initialized cache can only be reached at the top level:
            // recursive calls happen while the shared storage is still
            // invalid and the top level stores only after recursion returns.
            if (step != 0) {
                throw std::logic_error(
                    "Initialized FEIStorage reached a recursive FEI level.");
            }
            storage->load(level);
            last_fei_action_ = FEIAction::Loaded;
        }

        // Julia resets active_nodes[2:end] after valid-node discovery/load.
        for (std::size_t i = 1; i < level.active_nodes.size(); ++i) {
            level.active_nodes[i] = npos;
        }

        if (!cell_first) {
            level.active_nodes[0] = cell_entry;

            // Restore original signature ordering and every position-indexed
            // array that Julia swaps back together with it.
            swap_level_positions(level, first_entry, cell_entry);
        } else {
            cell_entry = first_entry;
            level.active_nodes[0] = first_entry;
        }

        level.valid_rays() = 0;
        for (std::size_t position = cell_entry;
             position < lsig;
             ++position) {
            level.valid_rays() +=
                level.valid_nodes[position] != 0 ? 1U : 0U;
        }

        if (!all_rays) {
            for (std::size_t position = 0;
                 position <= cell_entry;
                 ++position) {
                level.valid_nodes[position] = 0;
            }
        }

        return true;
    }

    void build_top_coordinates(FastLevel& level, const Point& r) {
        const std::size_t lsig = level.sig.size();

        copy_global_node(level.sig[0], node_buffer_);
        level.r = r - node_buffer_;

        level.local_xs[0].setZero();
        for (std::size_t position = 1; position < lsig; ++position) {
            copy_global_node(level.sig[position], level.local_xs[position]);
            level.local_xs[position] -= node_buffer_;

            level.local_cone[position] = level.local_xs[position];
            if (!normalize(level.local_cone[position])) {
                throw std::invalid_argument(
                    "Different generator indices have identical coordinates.");
            }
        }

        level.local_cone[0] = level.r;
        if (!normalize(level.local_cone[0])) {
            throw std::invalid_argument(
                "FastEdgeIterator vertex coincides with active generator.");
        }

        // Julia perturbs the initial normal direction by
        // vmin * rand() in every coordinate.
        Scalar vmin = (std::numeric_limits<Scalar>::max)();
        for (std::size_t position = 0; position < lsig; ++position) {
            const Scalar value = std::abs(
                level.local_cone[0].dot(level.local_cone[position]));
            vmin = (std::min)(vmin, value);
        }
        vmin /= Scalar{10} * static_cast<Scalar>(dimension_);

        for (std::size_t coordinate = 0;
             coordinate < dimension_;
             ++coordinate) {
            level.local_cone[0][static_cast<Eigen::Index>(coordinate)] +=
                vmin * static_cast<Scalar>(uniform_(random_));
        }
        if (!normalize(level.local_cone[0])) {
            throw std::runtime_error(
                "FastEdgeIterator failed to initialize random normal.");
        }
    }

    void build_recursive_coordinates(
        FastLevel& level,
        const FastLevel& parent,
        const Point& r) {

        const std::size_t lsig = level.sig.size();
        const std::size_t cdim = level.current_dim;

        for (std::size_t i = 0; i < dimension_; ++i) {
            level.rays[i] = parent.rays[i];
        }

        for (std::size_t position = 0; position < lsig; ++position) {
            const std::size_t parent_position =
                checked_local_position(level.sig[position], parent.sig.size());

            level.local_cone[position] =
                parent.local_cone[parent_position];

            if (position != 0) {
                level.local_xs[position] =
                    parent.local_xs[parent_position];
            }
        }
        level.local_xs[0].setZero();

        const Point& projection_normal = parent.rays[cdim];

        project_and_normalize_twice(
            level.local_cone[0],
            projection_normal);

        level.r = r;
        level.r -= level.r.dot(projection_normal) * projection_normal;
        level.r -= level.r.dot(projection_normal) * projection_normal;
    }

    void compute_valid_nodes(std::size_t step, Storage* storage) {
        FastLevel& level = levels_[step];
        const std::size_t cdim = level.current_dim;
        const std::size_t lsig = level.sig.size();
        constexpr std::size_t first_entry = 0;

        if (dimension_ == 2) {
            std::fill(level.valid_nodes.begin(), level.valid_nodes.end(), 0);

            Scalar minimum_cos = Scalar{2};
            std::size_t best_i = npos;
            std::size_t best_j = npos;

            for (std::size_t k1 = 0; k1 + 1 < lsig; ++k1) {
                if (k1 == first_entry) {
                    continue;
                }
                for (std::size_t k2 = k1; k2 < lsig; ++k2) {
                    if (k2 == first_entry) {
                        continue;
                    }
                    const Scalar value =
                        level.local_cone[k1].dot(level.local_cone[k2]);
                    if (value < minimum_cos) {
                        best_i = k1;
                        best_j = k2;
                        minimum_cos = value;
                    }
                }
            }

            if (best_i != npos) {
                level.valid_nodes[best_i] = 1;
            }
            if (best_j != npos) {
                level.valid_nodes[best_j] = 1;
            }
            return;
        }

        if (cdim == 3) {
            valid_nodes_3d(level, tolerances_.ray);
            return;
        }

        std::fill(level.valid_nodes.begin(), level.valid_nodes.end(), 0);

        std::size_t full_edge_count = 0;
        while (scan_for_edge(
            level,
            tolerances_.ray,
            false,
            full_edge_count)) {

            if (full_edge_count > cdim) {
                const auto begin = level.edge_buffer.begin();
                const auto end = begin +
                    static_cast<std::ptrdiff_t>(full_edge_count);
                const auto cell_it = std::find(begin, end, first_entry);

                if (cell_it == end) {
                    throw std::logic_error(
                        "scan_for_edge returned an edge without active cell.");
                }

                std::vector<Index>& recursive_sig = recursive_signature_buffer_;
                recursive_sig.clear();
                recursive_sig.reserve(full_edge_count);

                for (auto it = cell_it; it != end; ++it) {
                    recursive_sig.push_back(
                        checked_position_to_index(*it));
                }

                if (step + 1 >= levels_.size()) {
                    throw std::logic_error(
                        "Degenerate FEI recursion reached below dimension 2.");
                }

                const bool child_ready = reset_level(
                    step + 1,
                    recursive_sig,
                    level.r,
                    checked_position_to_index(first_entry),
                    &level,
                    storage,
                    false,
                    false);

                if (child_ready) {
                    const FastLevel& child = levels_[step + 1];
                    for (std::size_t child_position = 0;
                         child_position < child.sig.size();
                         ++child_position) {
                        const std::size_t parent_position =
                            checked_local_position(
                                child.sig[child_position],
                                level.sig.size());
                        level.valid_nodes[parent_position] =
                            child.valid_nodes[child_position];
                    }
                }
            } else {
                for (std::size_t i = 0; i < full_edge_count; ++i) {
                    level.valid_nodes[level.edge_buffer[i]] = 1;
                }
            }
        }

        level.valid_nodes[first_entry] = 0;
    }

    void valid_nodes_3d(FastLevel& level, Scalar maximum_angle) {
        if (level.current_dim != 3) {
            throw std::logic_error(
                "valid_nodes_3d called for a non-3D FEI level.");
        }

        const std::size_t first_entry = level.active_nodes[0];
        std::fill(level.valid_nodes.begin(), level.valid_nodes.end(), 0);

        std::size_t full_edge_count = 0;
        while (scan_for_edge(
            level,
            maximum_angle,
            true,
            full_edge_count)) {

            if (full_edge_count == 3) {
                if (level.active_nodes[1] != npos) {
                    level.valid_nodes[level.active_nodes[1]] = 1;
                }
                if (level.active_nodes[2] != npos) {
                    level.valid_nodes[level.active_nodes[2]] = 1;
                }
            } else {
                Scalar minimum_cos = Scalar{2};
                std::size_t best_i = npos;
                std::size_t best_j = npos;

                for (std::size_t k1 = 0;
                     k1 + 1 < full_edge_count;
                     ++k1) {
                    const std::size_t p1 = level.edge_buffer[k1];
                    if (p1 == first_entry) {
                        continue;
                    }

                    for (std::size_t k2 = k1;
                         k2 < full_edge_count;
                         ++k2) {
                        const std::size_t p2 = level.edge_buffer[k2];
                        if (p2 == first_entry) {
                            continue;
                        }

                        const Scalar value =
                            level.local_cone[p1].dot(level.local_cone[p2]);
                        if (value < minimum_cos) {
                            best_i = k1;
                            best_j = k2;
                            minimum_cos = value;
                        }
                    }
                }

                if (best_i != npos) {
                    level.valid_nodes[level.edge_buffer[best_i]] = 1;
                }
                if (best_j != npos) {
                    level.valid_nodes[level.edge_buffer[best_j]] = 1;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Julia scan_for_edge / Gram-Schmidt helpers
    // ---------------------------------------------------------------------

    [[nodiscard]] bool scan_for_edge(
        FastLevel& level,
        Scalar maximum_allowed_angle,
        bool all_edges,
        std::size_t& full_edge_count) {

        const std::size_t cdim = level.current_dim;
        const std::size_t first_entry = level.active_nodes[0];
        const std::size_t lsig = level.sig.size();

        for (;;) {
            if (!next_ray_try(level)) {
                full_edge_count = 0;
                return false;
            }

            level.rays[cdim - 1] = level.local_cone[first_entry];
            level.rays[0] = level.local_cone[level.current_primary()];
            rotate2(level.rays, 0, cdim);

            level.active_nodes[1] = level.current_primary();
            if (cdim > 2) {
                for (std::size_t i = 2;
                     i < level.active_nodes.size();
                     ++i) {
                    level.active_nodes[i] = npos;
                }
            }

            // Lemma 2.18 asks whether *any* supporting face containing the
            // current primary exists.  The Julia implementation tests only
            // one greedy max-angle completion and rejects the primary when
            // that single completion fails.  Search the remaining
            // completions before making that permanent decision.
            const bool found = find_supporting_face(
                level,
                std::size_t{1},
                maximum_allowed_angle,
                first_entry,
                lsig,
                full_edge_count);

            if (!found) {
                // No admissible completion of this primary generated a
                // supporting face.  Only now is the primary exhausted.
                const std::size_t rejected = level.active_nodes[1];
                if (rejected != npos) {
                    level.free_nodes[rejected] = 0;
                }
                continue;
            }

            for (std::size_t i = 0; i < full_edge_count; ++i) {
                level.free_nodes[level.edge_buffer[i]] = 0;
            }

            if (all_edges || level.edge_buffer[0] == first_entry) {
                level.rays[cdim - 1] *= Scalar{-1};
                return true;
            }
        }
    }

    /**
     * Search all max-angle ordered completions of one fixed primary.
     *
     * `basis` is the next Gram-Schmidt basis slot to fill.  Every recursion
     * level owns its own `tried_candidates` mask.  Therefore a candidate that
     * fails after one partial basis is excluded only for that partial basis;
     * it remains available after backtracking to an earlier basis choice.
     */
    [[nodiscard]] bool find_supporting_face(
        FastLevel& level,
        std::size_t basis,
        Scalar maximum_allowed_angle,
        std::size_t first_entry,
        std::size_t lsig,
        std::size_t& full_edge_count) {

        const std::size_t cdim = level.current_dim;

        if (basis + 1 >= cdim) {
            const Scalar edge_tolerance = (std::max)(
                max_basis_angle(level.rays, cdim),
                maximum_allowed_angle);

            full_edge_count = get_full_edge(
                level.rays,
                level.local_cone,
                level.edge_buffer,
                first_entry,
                lsig,
                cdim,
                edge_tolerance);

            return full_edge_count != 0;
        }

        const std::size_t tried_offset = basis * lsig;
        auto tried_begin =
            level.tried_candidates.begin() +
            static_cast<std::ptrdiff_t>(tried_offset);
        std::fill(
            tried_begin,
            tried_begin + static_cast<std::ptrdiff_t>(lsig),
            std::uint8_t{0});

        for (;;) {
            const auto candidate =
                max_angle_candidate(level, basis);

            if (candidate.second == npos) {
                full_edge_count = 0;
                return false;
            }

            // This candidate is consumed only on the current search level.
            // If the completed plane is not supporting, the next iteration
            // tries the next-best candidate while keeping the primary alive.
            level.tried_candidates[tried_offset + candidate.second] = 1;

            // rotate2 modifies only rays[basis] and the current normal.
            // Keep the backtracking state allocation-free.
            const Point saved_basis = level.rays[basis];
            const Point saved_normal = level.rays[cdim - 1];
            const std::size_t saved_active =
                level.active_nodes[basis + 1];

            level.rays[basis] =
                level.local_cone[candidate.second];
            rotate2(level.rays, basis, cdim);
            level.active_nodes[basis + 1] = candidate.second;

            if (find_supporting_face(
                    level,
                    basis + 1,
                    maximum_allowed_angle,
                    first_entry,
                    lsig,
                    full_edge_count)) {
                return true;
            }

            level.rays[basis] = saved_basis;
            level.rays[cdim - 1] = saved_normal;
            level.active_nodes[basis + 1] = saved_active;
        }
    }

    [[nodiscard]] bool next_ray_try(FastLevel& level) const {
        while (level.current_primary() < level.sig.size() &&
               level.free_nodes[level.current_primary()] == 0) {
            ++level.current_primary();
        }
        return level.current_primary() < level.sig.size();
    }

    [[nodiscard]] std::pair<Scalar, std::size_t>
    max_angle_candidate(
        const FastLevel& level,
        std::size_t basis) const {

        const Point& normal = level.rays[level.current_dim - 1];
        const std::size_t tried_offset = basis * level.sig.size();

        Scalar minimum_cos = Scalar{2};
        std::size_t minimum_index = npos;

        for (std::size_t position = 0;
             position < level.sig.size();
             ++position) {
            const Scalar value =
                normal.dot(level.local_cone[position]);

            if (std::abs(value) <= tolerances_.angle ||
                active_contains(level.active_nodes, position) ||
                level.tried_candidates[tried_offset + position] != 0) {
                continue;
            }

            if (value < minimum_cos) {
                minimum_cos = value;
                minimum_index = position;
            }
        }

        return {minimum_cos, minimum_index};
    }

    static bool active_contains(
        const std::vector<std::size_t>& active,
        std::size_t position) {
        return std::find(active.begin(), active.end(), position) != active.end();
    }

    static void ortho_project(
        std::vector<Point>& rays,
        std::size_t i,
        bool doubled) {

        for (std::size_t j = 0; j < i; ++j) {
            rays[i] -= rays[i].dot(rays[j]) * rays[j];
            if (doubled) {
                rays[i] -= rays[i].dot(rays[j]) * rays[j];
            }
            if (!normalize(rays[i])) {
                throw std::runtime_error(
                    "Gram-Schmidt produced a zero basis vector.");
            }
        }
    }

    static void update_normal(
        std::vector<Point>& rays,
        std::size_t i,
        std::size_t current_dim,
        bool doubled) {

        Point& normal = rays[current_dim - 1];
        const Point& basis = rays[i];

        normal -= normal.dot(basis) * basis;
        if (doubled) {
            normal -= normal.dot(basis) * basis;
        }
        if (!normalize(normal)) {
            throw std::runtime_error(
                "Gram-Schmidt produced a zero normal vector.");
        }
    }

    static void rotate(
        std::vector<Point>& rays,
        std::size_t i,
        std::size_t current_dim) {
        ortho_project(rays, i, false);
        update_normal(rays, i, current_dim, false);
    }

    static void rotate2(
        std::vector<Point>& rays,
        std::size_t i,
        std::size_t current_dim) {
        ortho_project(rays, i, true);
        update_normal(rays, i, current_dim, true);
    }

    [[nodiscard]] static Scalar max_basis_angle(
        const std::vector<Point>& rays,
        std::size_t current_dim) {

        Scalar maximum = Scalar{0};
        for (std::size_t i = 0; i + 1 < current_dim; ++i) {
            for (std::size_t j = i + 1; j < current_dim; ++j) {
                maximum = (std::max)(
                    maximum,
                    static_cast<Scalar>(std::abs(rays[i].dot(rays[j]))));
            }
        }
        return Scalar{10} * maximum;
    }

    static void project_and_normalize_twice(
        Point& vector,
        const Point& normal) {
        vector -= vector.dot(normal) * normal;
        if (!normalize(vector)) {
            throw std::runtime_error(
                "Projected FEI cone vector vanished.");
        }
        vector -= vector.dot(normal) * normal;
        if (!normalize(vector)) {
            throw std::runtime_error(
                "Projected FEI cone vector vanished after reorthogonalization.");
        }
    }

    /** Exact control flow of Julia get_full_edge; returns written count. */
    static std::size_t get_full_edge(
        std::vector<Point>& rays,
        const std::vector<Point>& cone,
        std::vector<std::size_t>& edge,
        std::size_t first_entry,
        std::size_t signature_size,
        std::size_t current_dim,
        Scalar maximum_angle) {

        std::size_t count = 0;
        Scalar maximum = Scalar{0};
        Scalar minimum = Scalar{0};
        Point& normal = rays[current_dim - 1];

        for (std::size_t position = 0;
             position < signature_size;
             ++position) {
            const Scalar value = normal.dot(cone[position]);
            const bool in_plane =
                std::abs(value) < Scalar{10} * maximum_angle;

            if (position != first_entry && !in_plane) {
                maximum = (std::max)(maximum, value);
                minimum = (std::min)(minimum, value);
            }

            if (position == first_entry || in_plane) {
                edge[count++] = position;
            }
        }

        if (maximum > maximum_angle && minimum < -maximum_angle) {
            return 0;
        }

        if (minimum < -maximum_angle) {
            normal *= Scalar{-1};
        }

        return count;
    }

    // ---------------------------------------------------------------------
    // Julia update_edge combination walk
    // ---------------------------------------------------------------------

    [[nodiscard]] bool update_fast_edge(std::size_t& dropped_position) {
        FastLevel& level = levels_.front();
        const std::size_t dim = dimension_;
        const std::size_t first_entry = level.active_nodes[0];

        if (level.valid_rays() < dim - 1) {
            return false;
        }

        std::size_t full_edge_count = 0;
        int old_broken_index = static_cast<int>(dim) + 1;

        while (full_edge_count == 0) {
            int start_index = 1; // Julia one-based basis index.

            if (level.active_nodes[1] == npos) {
                level.active_nodes[1] = first_entry;
                fill_up_edge(level, 2);
            } else {
                start_index = old_broken_index + 1;

                while (start_index > old_broken_index) {
                    bool advanced = false;

                    for (int i = static_cast<int>(dim); i >= 1; --i) {
                        if (i == 1) {
                            return false;
                        }

                        const std::size_t slot =
                            static_cast<std::size_t>(i - 1);
                        const std::size_t next = next_valid(
                            level.valid_nodes,
                            level.active_nodes[slot]);

                        const bool acceptable =
                            i == static_cast<int>(dim)
                                ? next != npos
                                : next != level.active_nodes[slot + 1];

                        if (acceptable) {
                            fill_up_edge(level, i);
                            start_index = i - 1;
                            advanced = true;
                            break;
                        }
                    }

                    if (!advanced) {
                        return false;
                    }
                }
            }

            level.rays[dim - 1] = level.local_cone[first_entry];
            bool basis_ok = true;

            // Julia: for i in 1:(start_index-1), rotate twice.
            for (int i = 1; i < start_index; ++i) {
                const std::size_t basis = static_cast<std::size_t>(i - 1);
                rotate(level.rays, basis, dim);
                rotate(level.rays, basis, dim);
            }

            // Julia: for i in start_index:(dim-1).
            for (int i = start_index;
                 i <= static_cast<int>(dim) - 1;
                 ++i) {
                const std::size_t basis = static_cast<std::size_t>(i - 1);
                const std::size_t active_slot = static_cast<std::size_t>(i);
                const std::size_t generator_position =
                    level.active_nodes[active_slot];

                if (generator_position == npos) {
                    return false;
                }

                level.rays[basis] =
                    level.local_cone[generator_position];

                if (std::abs(level.rays[basis].dot(level.rays[dim - 1])) <
                    tolerances_.angle) {
                    old_broken_index = i;
                    basis_ok = false;
                    break;
                }

                rotate(level.rays, basis, dim);
                rotate(level.rays, basis, dim);
            }

            if (!basis_ok) {
                continue;
            }

            full_edge_count = get_full_edge(
                level.rays,
                level.local_cone,
                level.edge_buffer,
                first_entry,
                level.sig.size(),
                dim,
                tolerances_.ray);

            // Julia flips the normal unconditionally after get_full_edge.
            level.rays[dim - 1] *= Scalar{-1};
        }

        // Julia's `dropped` identifies a local FEI position. Keep that local
        // position internally; next_fast() maps it back to the corresponding
        // global generator before exposing EdgeView::skip().
        dropped_position = 0;
        Scalar minimum = level.rays[dim - 1].dot(level.local_xs[0]);
        for (std::size_t position = 0;
             position < level.local_xs.size();
             ++position) {
            const Scalar value =
                level.rays[dim - 1].dot(level.local_xs[position]);
            if (value < minimum) {
                minimum = value;
                dropped_position = position;
            }
        }

        return true;
    }

    void fill_up_edge(FastLevel& level, int julia_index) const {
        // `julia_index` is the 1-based active_nodes slot used by the Julia
        // function fill_up_edge(my_minimal, ..., index, dim).
        for (int i = julia_index;
             i <= static_cast<int>(dimension_);
             ++i) {
            const std::size_t slot = static_cast<std::size_t>(i - 1);
            level.active_nodes[slot] = next_valid(
                level.valid_nodes,
                level.active_nodes[slot]);

            if (i < static_cast<int>(dimension_)) {
                level.active_nodes[slot + 1] = level.active_nodes[slot];
            }
        }
    }

    [[nodiscard]] static std::size_t next_valid(
        const std::vector<std::uint8_t>& valid_nodes,
        std::size_t position) {

        if (position == npos) {
            return npos;
        }

        for (std::size_t candidate = position + 1;
             candidate < valid_nodes.size();
             ++candidate) {
            if (valid_nodes[candidate] != 0) {
                return candidate;
            }
        }
        return npos;
    }

    [[nodiscard]] std::optional<View> next_fast() {
        std::size_t dropped_position = 0;
        if (!update_fast_edge(dropped_position)) {
            return std::nullopt;
        }

        FastLevel& level = levels_.front();
        const std::size_t first_entry = level.active_nodes[0];

        minimal_result_.clear();
        for (std::size_t slot = 0; slot < dimension_; ++slot) {
            const std::size_t position = level.active_nodes[slot];
            if (position == npos || position >= level.sig.size()) {
                throw std::logic_error(
                    "FastEdgeIterator produced incomplete minimal edge.");
            }
            minimal_result_.push_back(level.sig[position]);
        }

        // Julia get_full_edge(sig,r,edge,NF_,xs) calls get_full_edge again
        // after update_edge and then returns `-ray(NF_)`.
        const std::size_t full_edge_count = get_full_edge(
            level.rays,
            level.local_cone,
            level.edge_buffer,
            first_entry,
            level.sig.size(),
            dimension_,
            tolerances_.ray);

        if (full_edge_count == 0) {
            throw std::runtime_error(
                "FastEdgeIterator lost its supporting edge after update_edge.");
        }

        full_result_.clear();
        full_result_.reserve(full_edge_count);
        for (std::size_t i = 0; i < full_edge_count; ++i) {
            full_result_.push_back(level.sig[level.edge_buffer[i]]);
        }

        // Julia assigns an edge to a cell through the minimal edge, not
        // through the complete degenerate supporting edge. The full edge may
        // legitimately contain additional generators with smaller global
        // indices. systematic_explore_vertex applies the same edge[1] ==
        // _Cell condition in Julia.
        if (minimal_result_.empty() || minimal_result_.front() != cell_) {
            throw std::logic_error(
                "FastEdgeIterator produced a minimal edge not owned by the active cell.");
        }

        if (dropped_position >= level.sig.size()) {
            throw std::logic_error(
                "FastEdgeIterator produced an invalid dropped position.");
        }

        const Index skipped_generator = level.sig[dropped_position];
        if (std::find(
                full_result_.begin(),
                full_result_.end(),
                skipped_generator) != full_result_.end()) {
            throw std::logic_error(
                "FastEdgeIterator skip generator belongs to full edge.");
        }

        direction_ = -level.rays[dimension_ - 1];
        cycle_error_ = delta_u(level.rays, dimension_);
        if (cycle_error_ > tolerances_.correction) {
            correct_cycle_error(level, direction_);
        }

        return make_view(skipped_generator);
    }

    /**
     * Julia-compatible orthogonality/cycle error.
     *
     * This is the same quantity used by General_EdgeIterator::delta_u:
     * the sum of absolute projections of the final ray onto the first d-1
     * basis vectors.  FastEdgeIterator stores those basis vectors as rows of
     * the `rays` vector rather than columns of `onb`.
     */
    [[nodiscard]] static Scalar delta_u(
        const std::vector<Point>& rays,
        std::size_t dim) {
        return delta_u(rays, dim, rays[dim - 1]);
    }

    [[nodiscard]] static Scalar delta_u(
        const std::vector<Point>& rays,
        std::size_t dim,
        const Point& direction) {

        Scalar error = Scalar{0};
        for (std::size_t i = 0; i + 1 < dim; ++i) {
            const Scalar norm = rays[i].norm();
            if (norm > Scalar{0}) {
                error += std::abs(rays[i].dot(direction) / norm);
            }
        }
        return error;
    }

    void correct_cycle_error(
        FastLevel& level,
        const Point& old_direction) {

        // Julia first applies u_qr_onb to the already constructed basis and
        // then chooses the sign closest to the iterative direction.  Keep the
        // FEI basis untouched until a candidate has passed the cycle-error
        // check so the extended fallback sees exactly the same input basis.
        bool ok = qr_solver_.solve_basis(level.rays, qr_direction_);
        if (ok) {
            orient_like(qr_direction_, old_direction);
            const Scalar error =
                delta_u(level.rays, dimension_, qr_direction_);
            if (error <= tolerances_.correction) {
                level.rays[dimension_ - 1] = qr_direction_;
                direction_ = qr_direction_;
                cycle_error_ = error;
                return;
            }
        }

        ok = extended_qr_solver_.solve_basis(
            level.rays,
            qr_direction_);
        if (!ok) {
            throw std::runtime_error(
                "QR cycle correction failed even in extended precision.");
        }

        orient_like(qr_direction_, old_direction);
        const Scalar error =
            delta_u(level.rays, dimension_, qr_direction_);
        if (error > tolerances_.ray) {
            throw std::runtime_error(
                "Extended QR did not restore an orthogonal edge direction.");
        }

        level.rays[dimension_ - 1] = qr_direction_;
        direction_ = qr_direction_;
        cycle_error_ = error;
        return;
    }

    static void orient_like(Point& candidate, const Point& reference) {
        // Julia: u3 = dot(u2,u1)>0 ? u1 : -u1
        if (reference.dot(candidate) < Scalar{0}) {
            candidate *= Scalar{-1};
        }
    }

    [[nodiscard]] View make_view(
        Index skip,
        bool lazy_general_direction = false) {
        return View{
            EdgeIndexView<Index>(
                minimal_result_.data(),
                minimal_result_.size()),
            EdgeIndexView<Index>(
                full_result_.data(),
                full_result_.size()),
            skip,
            direction_,
            cycle_error_,
            lazy_general_direction ? static_cast<void*>(this) : nullptr,
            lazy_general_direction
                ? &EdgeIterator::materialize_general_direction_from_view
                : nullptr};
    }

    static void swap_level_positions(
        FastLevel& level,
        std::size_t left,
        std::size_t right) {
        if (left == right) {
            return;
        }

        std::swap(level.sig[left], level.sig[right]);
        std::swap(level.valid_nodes[left], level.valid_nodes[right]);
        std::swap(level.local_xs[left], level.local_xs[right]);
        std::swap(level.local_cone[left], level.local_cone[right]);
    }

    [[nodiscard]] std::size_t checked_local_position(
        Index value,
        std::size_t upper_bound) const {
        const std::size_t result = static_cast<std::size_t>(value);
        if (result >= upper_bound) {
            throw std::out_of_range(
                "Recursive FEI local index is outside parent signature.");
        }
        return result;
    }

    [[nodiscard]] Index checked_position_to_index(
        std::size_t position) const {
        if (position > static_cast<std::size_t>(
                (std::numeric_limits<Index>::max)())) {
            throw std::overflow_error(
                "FEI local position does not fit HighVoronoi Index.");
        }
        return static_cast<Index>(position);
    }

    const Nodes* nodes_;
    std::size_t dimension_;
    StorageCacheHandle storage_cache_;
    detail::EdgeIteratorTolerance<Scalar> tolerances_;

    Mode mode_{Mode::None};
    FEIAction last_fei_action_{FEIAction::None};
    bool cell_analysis_mode_{false};
    Index cell_{};
    Point vertex_position_;
    std::vector<Index> input_signature_;

    // General branch state.
    std::size_t general_next_{0};
    std::size_t general_end_{0};
    std::size_t general_cell_position_{npos};
    std::size_t general_current_omitted_{npos};
    bool general_direction_ready_{false};

    // Degenerate branch state.
    std::vector<FastLevel> levels_;
    bool fast_ready_{false};
    std::vector<Index> recursive_signature_buffer_;

    // Shared result buffers.
    std::vector<Index> minimal_result_;
    std::vector<Index> full_result_;
    Point direction_;
    Scalar cycle_error_{0};

    QRNormalSolver<Scalar, DimensionAtCompileTime> qr_solver_;
    ExtendedQRNormalSolver<Scalar, DimensionAtCompileTime> extended_qr_solver_;
    Point qr_direction_;
    Point node_buffer_;

    // Julia uses rand() for the initial FEI normal perturbation.  A persistent
    // engine keeps the same stochastic role while making C++ tests reproducible.
    std::mt19937_64 random_;
    std::uniform_real_distribution<double> uniform_;
};

} // namespace highvoronoi
