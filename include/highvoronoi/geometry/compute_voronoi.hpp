#pragma once

/**
 * @file compute_voronoi_20260812.hpp
 * @brief Top-level owner and coordinator of HighVoronoi mesh construction.
 *
 * ComputeVoronoi owns the global construction topology:
 *
 * - MeshThreading: one mesh branch or several parallel mesh branches;
 * - CastThreading: one or several workers inside every mesh branch;
 * - VoronoiThreading: MultiThread iff either of the two policies is multi-
 *   threaded. Its lock type determines the lock used by each branch queue.
 *
 * SystematicVoronoi is intentionally unaware of MeshThreading. VoronoiWorker
 * is unaware of all threading policies.
 *
 * The current implementation completes the one-mesh branch. Construction of
 * parallel MeshView/SwitchView branches is left behind one explicit
 * not-implemented error, as requested. The register_vertex() boundary is
 * already the place where future branches will exchange newly found vertices.
 */

#include <highvoronoi/detail/locks.hpp>
#include <highvoronoi/parameters.hpp>
#include <highvoronoi/geometry/systematic_voronoi.hpp>

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

/**
 * @brief Top-level Voronoi construction coordinator.
 *
 * @tparam MeshT Main mesh type.
 * @tparam RayCasterT RayCaster type for the current one-mesh implementation.
 * @tparam MeshThreadingT SingleThread or MultiThread for mesh branches.
 * @tparam CastThreadingT SingleThread or MultiThread inside each branch.
 */
template <
    class MeshT,
    class RayCasterT,
    class MeshThreadingT = SingleThread,
    class CastThreadingT = SingleThread,
    class QueueParametersT = DataBaseParams<
        typename MeshT::VertexScalar,
        typename MeshT::Index>,
    class EdgeParametersT = EdgeBufferParams<>>
class ComputeVoronoi final {
public:
    using Mesh = MeshT;
    using RayCaster = RayCasterT;
    using MeshThreading = MeshThreadingT;
    using CastThreading = CastThreadingT;
    using QueueParameters = QueueParametersT;
    using EdgeParameters = EdgeParametersT;

    using Index = typename Mesh::Index;
    using Sigma = typename Mesh::Sigma;

    static constexpr bool MeshIsParallel =
        MeshThreading::is_multithreaded;
    static constexpr bool CastIsParallel =
        CastThreading::is_multithreaded;
    static constexpr bool VoronoiIsParallel =
        MeshIsParallel || CastIsParallel;

    using VoronoiThreading = std::conditional_t<
        VoronoiIsParallel,
        MultiThread,
        SingleThread>;

    using QueueLock = typename VoronoiThreading::RWLock;

    using Systematic = SystematicVoronoi<
        ComputeVoronoi,
        Mesh,
        RayCaster,
        CastThreading,
        QueueLock,
        QueueParameters,
        EdgeParameters>;

    using Vertex = typename Systematic::Vertex;
    using EdgeIteratorType = typename Systematic::EdgeIteratorType;

    /**
     * @brief Construct the top-level algorithm for cells 0..range_end.
     *
     * range_end defaults to mesh.size()-1. The caller is responsible for
     * arranging the desired cells in the public prefix 0..range_end before
     * constructing ComputeVoronoi.
     *
     * The MultiThread mesh-branch path currently throws after validating that
     * the mesh database itself uses ReadWriteLock. Its later implementation
     * will build ReorderedMeshView/SwitchView branches here.
     */
    ComputeVoronoi(
        Mesh& mesh,
        const RayCaster& raycaster_prototype,
        MeshThreading mesh_threading,
        CastThreading cast_threading,
        std::optional<Index> range_end = std::nullopt,
        QueueParameters queue_parameters = QueueParameters(
            typename QueueParameters::ContainerMode()),
        EdgeParameters edge_parameters = EdgeParameters(
            typename EdgeParameters::ContainerMode()))
        : mesh_(mesh),
          mesh_threading_(std::move(mesh_threading)),
          cast_threading_(std::move(cast_threading)),
          voronoi_threading_(make_voronoi_threading(
              mesh_threading_,
              cast_threading_)),
          queue_parameters_(std::move(queue_parameters)),
          edge_parameters_(std::move(edge_parameters)),
          range_end_(resolve_range_end(range_end)) {

        static_assert(
            std::is_same_v<Index, typename RayCaster::Index>,
            "ComputeVoronoi requires equal mesh and RayCaster index types.");

        if constexpr (VoronoiIsParallel) {
            static_assert(
                std::is_same_v<
                    typename Mesh::Database::LockType,
                    ReadWriteLock>,
                "Any parallel Voronoi construction requires a mesh/database "
                "using ReadWriteLock because workers may store vertices "
                "concurrently.");
        }

        if constexpr (MeshIsParallel) {
            throw std::logic_error(
                "ComputeVoronoi parallel MeshView construction is not yet "
                "implemented. The one-mesh and CastThreading paths are ready.");
        } else {
            // Keep one common branch representation even in the serial case.
            parallel_meshes_.push_back(std::addressof(mesh_));

            systematic_voronois_.push_back(
                std::make_unique<Systematic>(
                    mesh_,
                    Index{0},
                    range_end_,
                    *this,
                    raycaster_prototype,
                    cast_threading_,
                    queue_parameters_,
                    edge_parameters_));
        }
    }

    ComputeVoronoi(const ComputeVoronoi&) = delete;
    ComputeVoronoi& operator=(const ComputeVoronoi&) = delete;
    ComputeVoronoi(ComputeVoronoi&&) = delete;
    ComputeVoronoi& operator=(ComputeVoronoi&&) = delete;

    /** Start all mesh branches. */
    void compute() {
        if constexpr (!MeshIsParallel) {
            systematic_voronois_.front()->compute();
        } else {
            throw std::logic_error(
                "Parallel mesh-branch execution is not yet implemented.");
        }
    }

    /**
     * @brief Persist and communicate one newly found vertex.
     *
     * In the current one-mesh path the persistent insertion is duplicate-safe
     * through AbstractMesh/HVDataBase. Regardless of whether another thread
     * inserted the vertex first, the branch queue performs its own first-claim
     * test. Therefore the thread that first claims the queue item is exactly
     * the thread responsible for the OnQueueEdges pass.
     *
     * The future MeshThreading path will convert
     *
     *   source public sigma -> stable internal sigma -> target public sigma
     *
     * and call queue_vertex() on every relevant SystematicVoronoi branch.
     */
    template <class MeshPoint>
    [[nodiscard]] bool register_vertex(
        const Vertex& vertex,
        Index source_id,
        EdgeIteratorType& source_queueing_iterator,
        Sigma& internal_sigma_buffer,
        Sigma& external_sigma_buffer,
        MeshPoint& mesh_point_buffer) {

        const std::size_t source = static_cast<std::size_t>(source_id);
        if (source >= systematic_voronois_.size()) {
            throw std::out_of_range(
                "ComputeVoronoi source branch id is invalid.");
        }

        detail::copy_voronoi_worker_point(
            vertex.position,
            mesh_point_buffer);

        Mesh& source_mesh = *parallel_meshes_[source];
        const auto address = source_mesh.store_vertex(
            mesh_point_buffer,
            vertex.sigma,
            internal_sigma_buffer);

        const bool newly_stored =
            address != static_cast<typename Mesh::Address>(0);

        if constexpr (!MeshIsParallel) {
            (void)external_sigma_buffer;

            (void)systematic_voronois_[0]->queue_vertex(
                vertex,
                source_queueing_iterator);
        } else {
            // The algorithmic communication contract is fixed, but actual
            // parallel MeshView objects are deliberately not introduced yet.
            throw std::logic_error(
                "Parallel vertex communication is not yet implemented.");
        }

        if (newly_stored) {
            new_vertices_.fetch_add(
                std::size_t{1},
                std::memory_order_relaxed);
        }

        return newly_stored;
    }

    [[nodiscard]] Mesh& mesh() noexcept {
        return mesh_;
    }

    [[nodiscard]] const Mesh& mesh() const noexcept {
        return mesh_;
    }

    [[nodiscard]] const MeshThreading& mesh_threading() const noexcept {
        return mesh_threading_;
    }

    [[nodiscard]] const CastThreading& cast_threading() const noexcept {
        return cast_threading_;
    }

    [[nodiscard]] const VoronoiThreading&
    voronoi_threading() const noexcept {
        return voronoi_threading_;
    }

    [[nodiscard]] Index range_end() const noexcept {
        return range_end_;
    }

    [[nodiscard]] std::size_t branch_count() const noexcept {
        return systematic_voronois_.size();
    }

    [[nodiscard]] Systematic& systematic_voronoi(std::size_t index) {
        return *systematic_voronois_.at(index);
    }

    [[nodiscard]] const Systematic&
    systematic_voronoi(std::size_t index) const {
        return *systematic_voronois_.at(index);
    }

    [[nodiscard]] std::size_t new_vertex_count() const noexcept {
        return new_vertices_.load(std::memory_order_relaxed);
    }

private:
    [[nodiscard]] Index resolve_range_end(
        std::optional<Index> requested) const {
        if (mesh_.size() == Index{0}) {
            throw std::invalid_argument(
                "ComputeVoronoi requires a non-empty mesh.");
        }

        if (mesh_.size() <= mesh_.dimension()) {
            throw std::invalid_argument(
                "There are not enough nodes to construct a Voronoi diagram.");
        }

        const Index last = requested
            ? *requested
            : static_cast<Index>(mesh_.size() - Index{1});

        if (last >= mesh_.size()) {
            throw std::out_of_range(
                "ComputeVoronoi range_end lies outside the mesh.");
        }

        return last;
    }

    [[nodiscard]] static VoronoiThreading make_voronoi_threading(
        const MeshThreading& mesh_threading,
        const CastThreading& cast_threading) {
        if constexpr (VoronoiIsParallel) {
            const std::size_t mesh_count = mesh_threading.thread_count();
            const std::size_t cast_count = cast_threading.thread_count();

            if (mesh_count >
                (std::numeric_limits<std::size_t>::max)() / cast_count) {
                throw std::overflow_error(
                    "Combined Voronoi thread count overflows size_t.");
            }

            return MultiThread(mesh_count * cast_count);
        } else {
            (void)mesh_threading;
            (void)cast_threading;
            return SingleThread{};
        }
    }

    Mesh& mesh_;

    MeshThreading mesh_threading_;
    CastThreading cast_threading_;
    VoronoiThreading voronoi_threading_;

    QueueParameters queue_parameters_;
    EdgeParameters edge_parameters_;
    Index range_end_;

    // The serial implementation stores the original mesh as branch 0. The
    // later MeshThreading specialization will replace these by owning mesh
    // views without changing SystematicVoronoi or VoronoiWorker.
    std::vector<Mesh*> parallel_meshes_;
    std::vector<std::unique_ptr<Systematic>> systematic_voronois_;

    std::atomic<std::size_t> new_vertices_{0};
};

} // namespace highvoronoi
