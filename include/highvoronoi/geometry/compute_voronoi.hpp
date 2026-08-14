#pragma once

/**
 * @file compute_voronoi.hpp
 * @brief Top-level owner and coordinator of HighVoronoi mesh construction.
 *
 * ComputeVoronoi owns the two independent construction axes:
 *
 * - MeshThreading: one mesh branch or several reordered mesh branches;
 * - CastThreading: one or several geometry workers inside each branch.
 *
 * Parallel mesh branches are non-owning ReorderedMeshViews of the same
 * persistent VoronoiMesh. Each branch moves one disjoint contiguous range of
 * public cells to the beginning of its local numbering. Newly found vertices
 * are first queued locally by the discovering SystematicVoronoi, then stored
 * once through the shared mesh/database and propagated to the other branches
 * in their local public numbering. Target branches perform queue-on-find with
 * their own prototype EdgeIterator; worker EdgeIterators never cross branches.
 *
 * SystematicVoronoi remains unaware of MeshThreading. VoronoiWorker remains
 * unaware of both threading policies.
 */

#include <highvoronoi/detail/hvview.hpp>
#include <highvoronoi/detail/locks.hpp>
#include <highvoronoi/parameters.hpp>
#include <highvoronoi/geometry/mesh_view.hpp>
#include <highvoronoi/geometry/systematic_voronoi.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace highvoronoi {

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

    using BranchIndexView = SwitchView<Index>;
    using ParallelBranchMesh = ReorderedMeshView<Mesh, BranchIndexView>;
    using BranchMesh = std::conditional_t<
        MeshIsParallel,
        ParallelBranchMesh,
        Mesh>;

    using ReboundRayCaster = decltype(
        std::declval<const RayCaster&>().rebind(
            std::declval<BranchMesh&>()));

    using BranchRayCaster = std::conditional_t<
        MeshIsParallel,
        ReboundRayCaster,
        RayCaster>;

    using Systematic = SystematicVoronoi<
        ComputeVoronoi,
        BranchMesh,
        BranchRayCaster,
        CastThreading,
        QueueLock,
        QueueParameters,
        EdgeParameters>;

    using Vertex = typename Systematic::Vertex;
    using EdgeIteratorType = typename Systematic::EdgeIteratorType;

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
            initialize_parallel_branches(raycaster_prototype);
        } else {
            initialize_serial_branch(raycaster_prototype);
        }
    }

    ComputeVoronoi(const ComputeVoronoi&) = delete;
    ComputeVoronoi& operator=(const ComputeVoronoi&) = delete;
    ComputeVoronoi(ComputeVoronoi&&) = delete;
    ComputeVoronoi& operator=(ComputeVoronoi&&) = delete;

    /** @brief Start all mesh branches and rethrow the first branch exception. */
    void compute() {
        if constexpr (!MeshIsParallel) {
            systematic_voronois_.front()->compute();
            return;
        }

        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> exceptions(
            systematic_voronois_.size());
        threads.reserve(systematic_voronois_.size());

        for (std::size_t branch = 0;
             branch < systematic_voronois_.size();
             ++branch) {
            threads.emplace_back([&, branch] {
                try {
                    systematic_voronois_[branch]->compute();
                } catch (...) {
                    exceptions[branch] = std::current_exception();
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

    /**
     * @brief Persist one newly found vertex and propagate it across branches.
     *
     * The source branch has already performed its local queue claim and
     * OnQueueEdges pass before this function is entered. Other branches receive
     * the translated vertex through SystematicVoronoi::queue_vertex(vertex),
     * which uses that branch's prototype EdgeIterator. The prototype is local
     * to the target branch and never transported through ComputeVoronoi.
     */
    template <class MeshPoint>
    [[nodiscard]] bool register_vertex(
        const Vertex& vertex,
        Index source_id,
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

        BranchMesh& source_mesh = *parallel_meshes_[source];
        const auto address = source_mesh.store_vertex(
            mesh_point_buffer,
            vertex.sigma,
            internal_sigma_buffer);

        const bool newly_stored =
            address != static_cast<typename BranchMesh::Address>(0);

        // The source branch has already queued this vertex in
        // SystematicVoronoi::register_found_vertex() with the discovering
        // worker's own EdgeIterator. Do not queue it a second time here.

        if constexpr (MeshIsParallel) {
            make_wrapped_public_signature(
                source_mesh,
                vertex.sigma,
                internal_sigma_buffer);

            for (std::size_t target = 0;
                 target < systematic_voronois_.size();
                 ++target) {
                if (target == source) {
                    continue;
                }

                BranchMesh& target_mesh = *parallel_meshes_[target];
                make_branch_public_signature(
                    target_mesh,
                    internal_sigma_buffer,
                    external_sigma_buffer);

                Vertex communicated(
                    static_cast<std::size_t>(mesh_.dimension()));
                communicated.sigma.assign(
                    external_sigma_buffer.begin(),
                    external_sigma_buffer.end());
                communicated.position = vertex.position;

                // No worker EdgeIterator crosses the branch boundary. The
                // target SystematicVoronoi uses its own prototype iterator,
                // serialized internally when required.
                (void)systematic_voronois_[target]
                    ->queue_vertex(communicated);
            }
        } else {
            (void)external_sigma_buffer;
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
    void initialize_serial_branch(const RayCaster& raycaster_prototype) {
        if constexpr (!MeshIsParallel) {
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

    void initialize_parallel_branches(
        const RayCaster& raycaster_prototype) {
        if constexpr (MeshIsParallel) {
            const std::size_t total_cells =
                static_cast<std::size_t>(range_end_) + std::size_t{1};
            const std::size_t branch_count = std::min(
                mesh_threading_.thread_count(),
                total_cells);

            if (branch_count == 0) {
                throw std::logic_error(
                    "ComputeVoronoi requires at least one mesh branch.");
            }

            owned_parallel_meshes_.reserve(branch_count);
            parallel_meshes_.reserve(branch_count);
            systematic_voronois_.reserve(branch_count);

            const std::size_t base_length = total_cells / branch_count;
            const std::size_t remainder = total_cells % branch_count;
            std::size_t first = 0;

            for (std::size_t branch = 0;
                 branch < branch_count;
                 ++branch) {
                const std::size_t length =
                    base_length + (branch < remainder ? 1 : 0);
                const std::size_t last = first + length - 1;

                auto branch_mesh = std::make_unique<BranchMesh>(
                    mesh_,
                    BranchIndexView(
                        checked_index(first),
                        checked_index(last)));

                BranchMesh& mesh_ref = *branch_mesh;
                auto branch_raycaster =
                    raycaster_prototype.rebind(mesh_ref);

                parallel_meshes_.push_back(std::addressof(mesh_ref));
                owned_parallel_meshes_.push_back(std::move(branch_mesh));

                systematic_voronois_.push_back(
                    std::make_unique<Systematic>(
                        mesh_ref,
                        checked_index(branch),
                        checked_index(length - 1),
                        *this,
                        branch_raycaster,
                        cast_threading_,
                        queue_parameters_,
                        edge_parameters_));

                first = last + 1;
            }
        }
    }

    template <class PublicSigma>
    void make_wrapped_public_signature(
        const BranchMesh& source_mesh,
        const PublicSigma& source_sigma,
        Sigma& output) const {
        output.clear();
        output.reserve(source_sigma.size());

        const Index ordinary_count = mesh_.size();
        for (const Index index : source_sigma) {
            if (index < ordinary_count) {
                output.push_back(
                    source_mesh.wrapped_public_index(index));
            } else {
                output.push_back(index);
            }
        }

        std::sort(output.begin(), output.end());
    }

    void make_branch_public_signature(
        const BranchMesh& target_mesh,
        const Sigma& wrapped_sigma,
        Sigma& output) const {
        output.clear();
        output.reserve(wrapped_sigma.size());

        const Index ordinary_count = mesh_.size();
        for (const Index index : wrapped_sigma) {
            if (index < ordinary_count) {
                const auto target_index =
                    target_mesh.reordered_public_index(index);
                if (!target_index) {
                    throw std::logic_error(
                        "Parallel mesh branch lost a visible ordinary node.");
                }
                output.push_back(*target_index);
            } else {
                output.push_back(index);
            }
        }

        std::sort(output.begin(), output.end());
    }

    [[nodiscard]] Index checked_index(std::size_t value) const {
        const std::size_t maximum = static_cast<std::size_t>(
            (std::numeric_limits<Index>::max)());
        if (value > maximum) {
            throw std::overflow_error(
                "ComputeVoronoi branch index does not fit Mesh::Index.");
        }
        return static_cast<Index>(value);
    }

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

    // Serial: branch 0 points directly to mesh_. Parallel: these point to the
    // owning reordered views below; all views share mesh_'s persistent data.
    std::vector<BranchMesh*> parallel_meshes_;
    std::vector<std::unique_ptr<BranchMesh>> owned_parallel_meshes_;
    std::vector<std::unique_ptr<Systematic>> systematic_voronois_;

    std::atomic<std::size_t> new_vertices_{0};
};

} // namespace highvoronoi
