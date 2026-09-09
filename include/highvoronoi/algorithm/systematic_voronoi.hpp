#pragma once

/**
 * @file systematic_voronoi.hpp
 * @brief Cell-level systematic Voronoi construction for one mesh branch.
 *
 * SystematicVoronoi knows only the CastThreading policy. It does not know
 * whether ComputeVoronoi operates one mesh or several communicating mesh
 * views. The parent supplies the queue lock type chosen from the complete
 * Voronoi threading configuration.
 *
 * Synchronization is encapsulated in the data structures that require it:
 *
 * - VertexQueue owns its queue lock and duplicate hash;
 * - EdgeHash uses the complete VoronoiThreading lock supplied as QueueLock;
 * - FEIStorageCache uses the same complete VoronoiThreading lock;
 * - cell_state_lock_ keeps one complete cell generation stable while vertices
 *   are queued and gives reset exclusive ownership of the transition;
 * - prototype_lock_ serializes only the shared prototype EdgeIterator;
 * - ConcurrentVertexIterator owns the lock required to distribute one mesh
 *   vertex iterator among several workers.
 *
 * SystematicVoronoi owns one branch-local generation lock plus one small
 * prototype lock. The generation lock is shared by complete queue transactions
 * and held exclusively while cell-local state is reset. The prototype lock
 * protects only mutable scratch state in prototype_edge_iterator_ when external
 * mesh branches queue vertices through ComputeVoronoi. Worker-owned
 * EdgeIterators remain otherwise unguarded. Shared EdgeHash and FEI cache
 * synchronization is provided by QueueLock, which ComputeVoronoi selects from
 * the complete VoronoiThreading configuration.
 */

#include <highvoronoi/storage/hash/hash_types.hpp>
#include <highvoronoi/core/detail/locks.hpp>
#include <highvoronoi/parameters.hpp>

#include <highvoronoi/algorithm/edge_iterator.hpp>
#include <highvoronoi/algorithm/voronoi_worker.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

namespace detail {

/**
 * @brief Reusable vertex queue with synchronization fully encapsulated.
 *
 * The QueueHash itself deliberately uses EmptyLock: the surrounding QueueLock
 * protects duplicate detection and queue mutation, mirroring Julia's
 * ThreadsafeQueue design where the queue lock protects its internal hash.
 */
template <class QueueLockT, class QueueParametersT, class VertexT>
class VoronoiVertexQueue final {
public:
    using QueueLock = QueueLockT;
    using QueueParameters = QueueParametersT;
    using Vertex = VertexT;
    using Sigma = typename Vertex::Sigma;

    using QueueHash =
        detail::QueueHashFromParams_t<detail::EmptyLock, QueueParameters>;

    VoronoiVertexQueue(
        std::size_t dimension,
        const QueueParameters& parameters)
        : dimension_(dimension),
          queue_hash_(
              detail::make_queue_hash<detail::EmptyLock>(parameters)) {}

    VoronoiVertexQueue(const VoronoiVertexQueue&) = delete;
    VoronoiVertexQueue& operator=(const VoronoiVertexQueue&) = delete;
    VoronoiVertexQueue(VoronoiVertexQueue&&) = delete;
    VoronoiVertexQueue& operator=(VoronoiVertexQueue&&) = delete;

    /** Clear queue entries and duplicate claims without releasing capacity. */
    void reset() {
        detail::WriteLockGuard<QueueLock> guard(lock_);
        size_ = 0;
        queue_hash_.clear();
    }

    /**
     * Claim one signature for the current cell.
     * @return true exactly for the first thread that claims this signature.
     */
    [[nodiscard]] bool claim(const Sigma& sigma) {
        detail::WriteLockGuard<QueueLock> guard(lock_);
        return !queue_hash_.pushqueue(sigma, true);
    }

    /** Enqueue a vertex whose signature was previously claimed. */
    void enqueue_claimed(const Vertex& vertex) {
        detail::WriteLockGuard<QueueLock> guard(lock_);

        if (size_ == slots_.size()) {
            slots_.emplace_back(dimension_);
        }

        Vertex& target = slots_[size_++];
        target.sigma.assign(vertex.sigma.begin(), vertex.sigma.end());
        target.position = vertex.position;
    }

    /** Pop one queued vertex into worker-owned reusable storage. */
    [[nodiscard]] bool pop(Vertex& output) {
        detail::WriteLockGuard<QueueLock> guard(lock_);

        if (size_ == 0) {
            return false;
        }

        Vertex& source = slots_[--size_];
        output.sigma.assign(source.sigma.begin(), source.sigma.end());
        output.position = source.position;
        return true;
    }

    [[nodiscard]] bool empty() const {
        detail::ReadLockGuard<QueueLock> guard(lock_);
        return size_ == 0;
    }

    [[nodiscard]] std::size_t size() const {
        detail::ReadLockGuard<QueueLock> guard(lock_);
        return size_;
    }

    void reserve(std::size_t count) {
        detail::WriteLockGuard<QueueLock> guard(lock_);
        slots_.reserve(count);
    }

    [[nodiscard]] QueueHash& queue_hash() noexcept {
        return queue_hash_;
    }

    [[nodiscard]] const QueueHash& queue_hash() const noexcept {
        return queue_hash_;
    }

private:
    std::size_t dimension_;
    mutable QueueLock lock_{};
    QueueHash queue_hash_;
    std::vector<Vertex> slots_;
    std::size_t size_{0};
};

/**
 * @brief Shared lock-safe consumer around one AbstractMesh vertex range.
 *
 * The wrapped mesh iterator itself remains completely unchanged. Exactly one
 * worker enters next() at a time, copies the current record into worker-owned
 * storage, and advances the iterator. The synchronization is therefore a
 * property of this wrapper, not of VoronoiWorker or SystematicVoronoi.
 */
template <class RangeT, class IteratorLockT>
class ConcurrentVertexIterator final {
public:
    using Range = RangeT;
    using IteratorLock = IteratorLockT;
    using Iterator = decltype(std::declval<const Range&>().begin());

    explicit ConcurrentVertexIterator(Range range)
        : range_(std::move(range)),
          iterator_(range_.begin()),
          end_(range_.end()) {}

    ConcurrentVertexIterator(const ConcurrentVertexIterator&) = delete;
    ConcurrentVertexIterator& operator=(const ConcurrentVertexIterator&) = delete;

    template <class Vertex>
    [[nodiscard]] bool next(Vertex& output) {
        detail::WriteLockGuard<IteratorLock> guard(lock_);

        if (iterator_ == end_) {
            return false;
        }

        const auto& source = *iterator_;
        output.sigma.assign(source.sigma.begin(), source.sigma.end());
        detail::copy_voronoi_worker_point(source.position, output.position);
        ++iterator_;
        consumed_.fetch_add(std::size_t{1}, std::memory_order_relaxed);
        return true;
    }

    [[nodiscard]] std::size_t consumed() const noexcept {
        return consumed_.load(std::memory_order_relaxed);
    }

private:
    Range range_;
    Iterator iterator_;
    Iterator end_;
    mutable IteratorLock lock_{};
    std::atomic<std::size_t> consumed_{0};
};

} // namespace detail

/**
 * @brief Complete systematic Voronoi subroutine for exactly one mesh branch.
 *
 * @tparam MainT Top-level ComputeVoronoi coordinator type.
 * @tparam MeshT Mesh type seen by this branch.
 * @tparam RayCasterT RayCaster type operating on MeshT.
 * @tparam CastThreadingT SingleThread or MultiThread for workers in this branch.
 * @tparam QueueLockT Lock selected by the complete Voronoi threading mode.
 * @tparam QueueParametersT Parameters of the cell-local duplicate queue hash.
 * @tparam EdgeParametersT Parameters of the cell-local edge hash.
 */
template <
    class MainT,
    class MeshT,
    class RayCasterT,
    class CastThreadingT,
    class QueueLockT,
    class QueueParametersT,
    class EdgeParametersT>
class SystematicVoronoi final {
public:
    using Main = MainT;
    using Mesh = MeshT;
    using RayCaster = RayCasterT;
    using CastThreading = CastThreadingT;
    using QueueLock = QueueLockT;
    using QueueParameters = QueueParametersT;
    using EdgeParameters = EdgeParametersT;

    using Index = typename Mesh::Index;
    using Sigma = typename Mesh::Sigma;
    using VertexPoint = typename Mesh::VertexPoint;
    using Vertex = VoronoiVertexTask<Mesh, RayCaster>;

    using CastLock = typename CastThreading::RWLock;
    using SharedLock = QueueLock;
    using ExtendedNodes = typename RayCaster::ExtendedNodes;
    using EdgeIteratorType = EdgeIterator<ExtendedNodes, SharedLock>;

    using VertexQueue = detail::VoronoiVertexQueue<
        QueueLock,
        QueueParameters,
        Vertex>;

    using EdgeHash =
        detail::EdgeHashFromParams_t<SharedLock, EdgeParameters>;

    using Worker = VoronoiWorker<
        SystematicVoronoi,
        Mesh,
        RayCaster,
        EdgeIteratorType,
        VertexQueue>;

    SystematicVoronoi(
        Mesh& mesh,
        Index id,
        Index last_local_cell,
        Main& main,
        const RayCaster& raycaster_prototype,
        CastThreading cast_threading,
        QueueParameters queue_parameters,
        EdgeParameters edge_parameters)
        : id_(id),
          last_local_cell_(last_local_cell),
          main_(main),
          mesh_(mesh),
          cast_threading_(std::move(cast_threading)),
          vertex_queue_(
              static_cast<std::size_t>(mesh.dimension()),
              queue_parameters),
          edge_hash_(
              detail::make_edge_hash<SharedLock>(edge_parameters)),
          prototype_raycaster_(raycaster_prototype.safe_copy()),
          prototype_edge_iterator_(
              prototype_raycaster_.extended_nodes(),
              prototype_raycaster_.parameters().ray_tolerance) {

        static_assert(
            std::is_same_v<Index, typename RayCaster::Index>,
            "SystematicVoronoi requires equal mesh and RayCaster indices.");

        const std::size_t worker_count = cast_threading_.thread_count();
        if (worker_count == 0) {
            throw std::logic_error(
                "SystematicVoronoi requires at least one worker.");
        }

        workers_.reserve(worker_count);
        for (std::size_t worker_id = 0;
             worker_id < worker_count;
             ++worker_id) {
            workers_.push_back(std::make_unique<Worker>(
                mesh_,
                *this,
                vertex_queue_,
                prototype_raycaster_,
                prototype_edge_iterator_,
                worker_id));
        }

        mirror_indices_.reserve(static_cast<std::size_t>(
            prototype_raycaster_.extended_nodes().mirror_count()));

        reset(Index{0});
    }

    SystematicVoronoi(const SystematicVoronoi&) = delete;
    SystematicVoronoi& operator=(const SystematicVoronoi&) = delete;
    SystematicVoronoi(SystematicVoronoi&&) = delete;
    SystematicVoronoi& operator=(SystematicVoronoi&&) = delete;

    /**
     * @brief Compute every local cell 0..last_local_cell_ completely.
     */
    void compute() {
        reset(Index{0});

        while (current_cell() <= last_local_cell_) {
            const Index cell = current_cell();

            auto range = mesh_.vertices(cell);
            using Range = decltype(range);
            using IteratorLock = typename CastThreading::RWLock;
            detail::ConcurrentVertexIterator<Range, IteratorLock>
                vertex_iterator(std::move(range));

            run_workers([&](Worker& worker) {
                worker.queue_vertices_from_iterator(
                    vertex_iterator,
                    cell);
            });

            if (vertex_iterator.consumed() == 0 &&
                vertex_queue_.empty()) {
                Vertex initial = workers_.front()->descent(cell);
                (void)workers_.front()->register_found_vertex(initial);
            }

            run_workers([&](Worker& worker) {
                worker.systematic_explore_all_vertices(cell);
            });

            // One branch-local cell is now complete. The parent owns optional
            // progress reporting and may receive these events concurrently from
            // several mesh branches.
            main_.cell_completed();

            if (cell == last_local_cell_) {
                current_cell_.store(
                    invalid_cell(),
                    std::memory_order_release);

                detail::WriteLockGuard<SharedLock> state_guard(
                    cell_state_lock_);
                vertex_queue_.reset();
                edge_hash_.clear();
                break;
            }

            reset(static_cast<Index>(cell + Index{1}));
        }
    }

    /**
     * @brief Reset all cell-local structures, then publish the new cell index.
     *
     * Publishing the invalid sentinel first makes newly arriving queue attempts
     * fail while reset waits for queue transactions already holding a read lock.
     * The write lock then excludes every complete queue transaction until queue,
     * edge and active-cell geometry state all belong to the new cell.
     */
    void reset(Index cell) {
        if (cell > last_local_cell_) {
            throw std::out_of_range(
                "SystematicVoronoi reset cell exceeds its local range.");
        }

        current_cell_.store(
            invalid_cell(),
            std::memory_order_release);

        detail::WriteLockGuard<SharedLock> state_guard(cell_state_lock_);

        vertex_queue_.reset();
        edge_hash_.clear();

        build_mirror_indices();
        prototype_raycaster_.activate_cell(cell, mirror_indices_);
        for (auto& worker : workers_) {
            worker->activate_cell(cell, mirror_indices_);
        }

        current_cell_.store(cell, std::memory_order_release);
    }

    /**
     * @brief Queue a vertex using a worker-supplied queueing EdgeIterator.
     *
     * This is the common queue-on-find operation for existing vertices and new
     * vertices. Only the first successful queue claim is responsible for the
     * OnQueueEdges pass.
     */
    [[nodiscard]] bool queue_vertex(
        const Vertex& vertex,
        EdgeIteratorType& edge_iterator,
        Index cell) {

        detail::ReadLockGuard<SharedLock> state_guard(cell_state_lock_);

        if (cell != current_cell()) {
            return false;
        }

        if (!std::binary_search(
                vertex.sigma.begin(),
                vertex.sigma.end(),
                cell)) {
            return false;
        }

        if (!vertex_queue_.claim(vertex.sigma)) {
            return false;
        }

        bool all_edges_complete = false;

        if (std::addressof(edge_iterator) ==
            std::addressof(prototype_edge_iterator_)) {
            detail::WriteLockGuard<SharedLock> guard(prototype_lock_);
            all_edges_complete =
                queue_edges(vertex, edge_iterator, cell);
        } else {
            all_edges_complete =
                queue_edges(vertex, edge_iterator, cell);
        }

        if (!all_edges_complete) {
            vertex_queue_.enqueue_claimed(vertex);
        }

        return true;
    }

    [[nodiscard]] bool queue_vertex(
        const Vertex& vertex,
        EdgeIteratorType& edge_iterator) {
        return queue_vertex(
            vertex,
            edge_iterator,
            current_cell());
    }

    /**
     * @brief Queue a vertex received from ComputeVoronoi.
     *
     * External mesh-branch communication deliberately supplies no EdgeIterator.
     * The branch therefore uses its own prototype iterator. Concurrent external
     * calls are serialized only while that prototype's mutable scratch state is
     * used. EdgeHash and FEI cache remain the same shared branch state seen by
     * all local workers.
     */
    [[nodiscard]] bool queue_vertex(const Vertex& vertex) {
        return queue_vertex(
            vertex,
            prototype_edge_iterator_,
            current_cell());
    }

    /**
     * @brief Forward a newly found worker vertex to ComputeVoronoi::register.
     */
    [[nodiscard]] bool register_found_vertex(
        const Vertex& vertex,
        EdgeIteratorType& source_queueing_iterator,
        Sigma& internal_sigma_buffer,
        Sigma& external_sigma_buffer,
        VertexPoint& mesh_point_buffer) {

        // A locally discovered vertex must always be registered in the local
        // branch independently of persistent/global duplicate detection. The
        // worker supplies its own queueing iterator, so this path needs no
        // prototype lock.
        (void)queue_vertex(
            vertex,
            source_queueing_iterator);

        // EdgeIterator ownership ends at the SystematicVoronoi boundary.
        return main_.register_vertex(
            vertex,
            id_,
            internal_sigma_buffer,
            external_sigma_buffer,
            mesh_point_buffer);
    }

    /**
     * @brief Persist one unbounded edge directly in this branch mesh.
     *
     * The mesh converts the complete supporting edge to stable internal
     * numbering. Its database hash performs global duplicate detection and the
     * mesh-level address list records only the successful insertion. No
     * ComputeVoronoi broadcast is required for unbounded edges.
     */
    template <class FullEdge>
    [[nodiscard]] bool register_infinite_edge(
        const FullEdge& full_edge,
        const VertexPoint& origin,
        const VertexPoint& direction,
        Sigma& internal_sigma_buffer) {
        return mesh_.store_infinite_edge(
                   full_edge,
                   origin,
                   direction,
                   internal_sigma_buffer) != typename Mesh::Address{0};
    }

    [[nodiscard]] Index id() const noexcept {
        return id_;
    }

    [[nodiscard]] Index current_cell() const noexcept {
        return current_cell_.load(std::memory_order_acquire);
    }

    [[nodiscard]] Index last_local_cell() const noexcept {
        return last_local_cell_;
    }

    [[nodiscard]] Mesh& mesh() noexcept {
        return mesh_;
    }

    [[nodiscard]] const Mesh& mesh() const noexcept {
        return mesh_;
    }

    [[nodiscard]] VertexQueue& vertex_queue() noexcept {
        return vertex_queue_;
    }

    [[nodiscard]] const VertexQueue& vertex_queue() const noexcept {
        return vertex_queue_;
    }

    [[nodiscard]] EdgeHash& edge_hash() noexcept {
        return edge_hash_;
    }

    [[nodiscard]] const EdgeHash& edge_hash() const noexcept {
        return edge_hash_;
    }

    [[nodiscard]] EdgeIteratorType& prototype_edge_iterator() noexcept {
        return prototype_edge_iterator_;
    }

    [[nodiscard]] const EdgeIteratorType&
    prototype_edge_iterator() const noexcept {
        return prototype_edge_iterator_;
    }

    [[nodiscard]] std::size_t worker_count() const noexcept {
        return workers_.size();
    }

    [[nodiscard]] Worker& worker(std::size_t index) {
        return *workers_.at(index);
    }

private:
    [[nodiscard]] static constexpr Index invalid_cell() noexcept {
        return (std::numeric_limits<Index>::max)();
    }

    [[nodiscard]] static std::int64_t edge_token(Index value) {
        const auto maximum = static_cast<std::uintmax_t>(
            (std::numeric_limits<std::int64_t>::max)());
        const auto converted = static_cast<std::uintmax_t>(value);

        if (converted > maximum) {
            throw std::overflow_error(
                "EdgeHash token does not fit into int64_t.");
        }

        return static_cast<std::int64_t>(value);
    }

    [[nodiscard]] bool queue_edges(
        const Vertex& vertex,
        EdgeIteratorType& edge_iterator,
        Index cell) {

        edge_iterator.reset(
            vertex.sigma,
            vertex.position,
            cell,
            typename EdgeIteratorType::OnQueueEdges{});

        bool all_edges_complete = true;

        while (const auto edge = edge_iterator.next()) {
            all_edges_complete &= edge_hash_.pushedge(
                edge->indices(),
                edge_token(edge->skip()),
                false);
        }

        return all_edges_complete;
    }

    void build_mirror_indices() {
        mirror_indices_.clear();

        const auto& nodes = prototype_raycaster_.extended_nodes();
        const Index count = nodes.mirror_count();

        for (Index plane = Index{0}; plane < count; ++plane) {
            mirror_indices_.push_back(nodes.mirror_index(plane));
        }
    }

    template <class Function>
    void run_workers(Function&& function) {
        if constexpr (!CastThreading::is_multithreaded) {
            function(*workers_.front());
            return;
        }

        if (workers_.size() == 1) {
            function(*workers_.front());
            return;
        }

        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> exceptions(workers_.size());
        threads.reserve(workers_.size());

        for (std::size_t i = 0; i < workers_.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    function(*workers_[i]);
                } catch (...) {
                    exceptions[i] = std::current_exception();
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (const auto& exception : exceptions) {
            if (exception) {
                std::rethrow_exception(exception);
            }
        }
    }

    Index id_;
    Index last_local_cell_;
    Main& main_;
    Mesh& mesh_;
    CastThreading cast_threading_;

    VertexQueue vertex_queue_;
    EdgeHash edge_hash_;

    RayCaster prototype_raycaster_;
    EdgeIteratorType prototype_edge_iterator_;
    mutable SharedLock cell_state_lock_{};
    mutable SharedLock prototype_lock_{};
    std::vector<std::unique_ptr<Worker>> workers_;

    std::atomic<Index> current_cell_{Index{0}};
    std::vector<Index> mirror_indices_;
};

} // namespace highvoronoi
