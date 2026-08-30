
#pragma once

/**
 * @file compute_high_voronoi.hpp
 * @brief HighVoronoi orchestration built from RemoveVoronoi and RefineVoronoi.
 *
 * Pending deletion and insertion remain batched at HighVoronoiMesh level, but
 * their geometry is no longer merged into one reconstruction round:
 *
 *   remover.remove()
 *   refine(pending visible NEW)
 *   refine(periodic NEW) until periodic closure
 *   remover.compute()
 *   refine(periodic NEW created by remove closure) until periodic closure
 *
 * Remove and refine therefore retain independent AFFECTED state and lifetime.
 */

#include <highvoronoi/geometry/remove_voronoi.hpp>
#include <highvoronoi/geometry/refine_voronoi.hpp>
#include <highvoronoi/geometry/high_voronoi_incremental_backend.hpp>
#include <highvoronoi/geometry/periodic_boundary_repair.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace highvoronoi {

template <
    class HighMeshT,
    class SearchKeywordT = geometry::KDSearch,
    class RayParametersT = RaycastParameters<
        InRangeRaycast,
        typename HighMeshT::NodeScalar>,
    class MeshThreadingT = SingleThread,
    class CastThreadingT = SingleThread,
    class QueueParametersT = DataBaseParams<
        typename HighMeshT::VertexScalar,
        typename HighMeshT::Index>,
    class EdgeParametersT = EdgeBufferParams<>>
class ComputeHighVoronoi final {
public:
    using HighMesh = HighMeshT;
    using SearchKeyword = SearchKeywordT;
    using RayParameters = RayParametersT;
    using MeshThreading = MeshThreadingT;
    using CastThreading = CastThreadingT;
    using QueueParameters = QueueParametersT;
    using EdgeParameters = EdgeParametersT;

    using Index = typename HighMesh::Index;
    using NodePoint = typename HighMesh::NodePoint;
    using NodeScalar = typename HighMesh::NodeScalar;
    using VertexScalar = typename HighMesh::VertexScalar;
    using ShiftMask = typename HighMesh::ShiftMask;
    using PeriodicRequest = std::pair<Index, Index>;

    using Remover = RemoveVoronoi<
        HighMesh,
        SearchKeyword,
        RayParameters,
        MeshThreading,
        CastThreading,
        QueueParameters,
        EdgeParameters>;

    using Refiner = RefineVoronoi<
        HighMesh,
        SearchKeyword,
        RayParameters,
        MeshThreading,
        CastThreading,
        QueueParameters,
        EdgeParameters>;

    struct Settings {
        VertexScalar nearest_tolerance = VertexScalar{1e-10};
        NodeScalar boundary_tolerance = NodeScalar{1e-10};
        std::size_t maximum_rounds = 16;
        std::size_t maximum_periodic_repair_sweeps = 16;

        // Kept for source compatibility with the previous diagnostic driver.
        bool debug_new_view_steps = false;
    };

    struct Report {
        std::size_t rounds = 0;
        std::size_t deleted_vertices = 0;
        std::size_t invalidated_old_vertices = 0;
        std::size_t new_cell_vertices = 0;
        std::size_t repair_vertices = 0;
        std::size_t new_reference_nodes = 0;
        std::size_t moved_internal_planes = 0;
        std::size_t final_visible_nodes = 0;
        std::size_t final_internal_nodes = 0;
        PeriodicBoundaryRepairReport periodic_boundary_repair;

        // Diagnostic history: exactly the NEW block computed by every refiner.
        std::vector<std::vector<Index>> new_node_blocks;
    };

    ComputeHighVoronoi(
        HighMesh& mesh,
        SearchKeyword search_keyword = SearchKeyword{},
        RayParameters ray_parameters = RayParameters{},
        MeshThreading mesh_threading = MeshThreading{},
        CastThreading cast_threading = CastThreading{},
        QueueParameters queue_parameters = QueueParameters(
            typename QueueParameters::ContainerMode{}),
        EdgeParameters edge_parameters = EdgeParameters(
            typename EdgeParameters::ContainerMode{}),
        Settings settings = Settings{})
        : mesh_(mesh),
          search_keyword_(std::move(search_keyword)),
          ray_parameters_(std::move(ray_parameters)),
          mesh_threading_(std::move(mesh_threading)),
          cast_threading_(std::move(cast_threading)),
          queue_parameters_(std::move(queue_parameters)),
          edge_parameters_(std::move(edge_parameters)),
          settings_(settings) {}

    ComputeHighVoronoi(const ComputeHighVoronoi&) = delete;
    ComputeHighVoronoi& operator=(const ComputeHighVoronoi&) = delete;
    ComputeHighVoronoi(ComputeHighVoronoi&&) = delete;
    ComputeHighVoronoi& operator=(ComputeHighVoronoi&&) = delete;

    [[nodiscard]] Report compute() {
        Report report;

        // A HighVoronoiMesh marks visible deletions (and all associated
        // references) inactive immediately. Their persisted vertices still
        // exist and are consumed here by the remover's destructive phase.
        std::unique_ptr<Remover> remover;
        const std::vector<Index> deleted =
            mesh_.pending_deleted_internal_nodes();
        if (!deleted.empty()) {
            remover = std::make_unique<Remover>(
                mesh_,
                existing_removed_nodes,
                deleted,
                search_keyword_,
                ray_parameters_,
                mesh_threading_,
                cast_threading_,
                queue_parameters_,
                edge_parameters_);
            (void)remover->remove();
            report.deleted_vertices += remover->report().erased_vertices;
        }

        // Visible nodes are already present in stable HighVoronoi storage.
        // The integrated watermark is the exact OLD/NEW cut for this block.
        std::vector<PeriodicRequest> requests;
        const std::vector<Index> visible_new =
            mesh_.pending_new_visible_internal_nodes();
        if (!visible_new.empty()) {
            requests = refine_existing_block(
                visible_new,
                mesh_.integrated_node_count(),
                report);
            close_periodic_requests(std::move(requests), report);
        }

        // Only removal needs a later closure compute. It runs against the mesh
        // that already contains every visible/reference node generated above.
        if (remover) {
            (void)remover->compute();
            report.repair_vertices += remover->report().repair_vertices;

            // Vertices created by the remove closure may expose new periodic
            // neighbours. Those references are ordinary NEW refine blocks.
            close_periodic_requests(
                remover->periodic_requests(),
                report);
        }

        // Bootstrap periodic boundary vertices can hide finite vertices made
        // exclusively from already-visible generators. They are repaired only
        // after reference-node closure has established the final internal
        // geometry. The repair reuses unchanged serial ComputeVoronoi on sparse
        // transient per-cell address lists.
        run_periodic_boundary_sparse_repair(
            mesh_,
            periodic_repair_seed_signatures_,
            search_keyword_,
            ray_parameters_,
            queue_parameters_,
            edge_parameters_,
            settings_.maximum_periodic_repair_sweeps,
            report.periodic_boundary_repair);
        report.repair_vertices +=
            report.periodic_boundary_repair.new_vertices;

        mesh_.mark_integrated();
        report.final_visible_nodes =
            static_cast<std::size_t>(mesh_.visible_public_count());
        report.final_internal_nodes =
            static_cast<std::size_t>(mesh_.internal_node_count());
        return report;
    }

private:
    struct PlannedReference {
        Index reference_internal = Index{0};
        ShiftMask shift;
        NodePoint point;
    };

    [[nodiscard]] std::vector<PeriodicRequest> refine_existing_block(
        const std::vector<Index>& block,
        Index old_internal_count,
        Report& report) {
        if (block.empty()) {
            return {};
        }
        if (report.rounds >= settings_.maximum_rounds) {
            throw std::runtime_error(
                "ComputeHighVoronoi exceeded the periodization round limit.");
        }

        report.new_node_blocks.push_back(block);
        ++report.rounds;

        typename Refiner::Settings refine_settings;
        refine_settings.nearest_tolerance = settings_.nearest_tolerance;
        const bool bootstrap_periodic_geometry =
            old_internal_count == Index{0};
        refine_settings.capture_periodic_boundary_vertices =
            bootstrap_periodic_geometry;

        Refiner refiner(
            mesh_,
            existing_incremental_nodes,
            block,
            old_internal_count,
            search_keyword_,
            ray_parameters_,
            mesh_threading_,
            cast_threading_,
            queue_parameters_,
            edge_parameters_,
            refine_settings);
        (void)refiner.compute();

        report.new_cell_vertices += refiner.report().new_cell_vertices;
        report.invalidated_old_vertices +=
            refiner.report().invalidated_old_vertices;

        if (bootstrap_periodic_geometry) {
            const auto seeds = detail::collect_periodic_boundary_repair_seeds(
                mesh_,
                refiner.periodic_boundary_vertices(),
                search_keyword_,
                ray_parameters_,
                report.periodic_boundary_repair);
            for (const auto& sigma : seeds) {
                detail::append_unique_sigma(
                    periodic_repair_seed_signatures_,
                    sigma);
            }
        }

        return refiner.periodic_requests();
    }

    void close_periodic_requests(
        std::vector<PeriodicRequest> requests,
        Report& report) {
        while (!requests.empty()) {
            const std::vector<PlannedReference> planned =
                materialize_periodic_requests(requests);
            if (planned.empty()) {
                return;
            }

            const Index old_internal_count = mesh_.internal_node_count();
            const std::vector<Index> appended =
                append_planned_references(planned, report);
            if (appended.empty()) {
                return;
            }

            requests = refine_existing_block(
                appended,
                old_internal_count,
                report);
        }
    }

    [[nodiscard]] std::vector<PlannedReference>
    materialize_periodic_requests(
        const std::vector<PeriodicRequest>& requests) const {
        const Index node_count = mesh_.internal_node_count();
        const Index plane_count = mesh_.external_boundary().size();

        std::vector<std::vector<Index>> requested_planes(
            static_cast<std::size_t>(node_count));

        for (const auto& request : requests) {
            const Index reference = request.first;
            const Index plane = request.second;
            if (reference >= node_count || plane >= plane_count ||
                !mesh_.is_active_internal(reference) ||
                !mesh_.is_visible_internal(reference)) {
                continue;
            }
            if (!mesh_.external_boundary()[plane].is_periodic()) {
                continue;
            }
            requested_planes[static_cast<std::size_t>(reference)]
                .push_back(plane);
        }

        std::vector<PlannedReference> result;
        for (Index reference = Index{0}; reference < node_count; ++reference) {
            auto& candidates =
                requested_planes[static_cast<std::size_t>(reference)];
            if (candidates.empty()) {
                continue;
            }
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(
                std::unique(candidates.begin(), candidates.end()),
                candidates.end());

            ShiftMask current(
                static_cast<std::size_t>(plane_count),
                std::uint8_t{0});
            detail::BitVector taboo(
                static_cast<std::size_t>(plane_count));
            enumerate_shift_combinations(
                reference,
                candidates,
                std::size_t{0},
                current,
                taboo,
                result);
        }
        return result;
    }

    void enumerate_shift_combinations(
        Index reference,
        const std::vector<Index>& candidates,
        std::size_t position,
        ShiftMask& current,
        detail::BitVector& taboo,
        std::vector<PlannedReference>& output) const {
        if (position == candidates.size()) {
            const bool nonempty = std::any_of(
                current.begin(),
                current.end(),
                [](std::uint8_t value) { return value != 0; });
            if (!nonempty ||
                mesh_.find_reference_copy(reference, current).has_value() ||
                planned_contains(output, reference, current)) {
                return;
            }

            NodePoint point = make_node_point();
            mesh_.internal_nodes().copy_node(reference, point.data());
            point += mesh_.periodic_shift(current);
            output.push_back(PlannedReference{
                reference,
                current,
                std::move(point)});
            return;
        }

        enumerate_shift_combinations(
            reference,
            candidates,
            position + 1,
            current,
            taboo,
            output);

        const Index plane = candidates[position];
        if (taboo.test(static_cast<std::size_t>(plane))) {
            return;
        }
        const auto partner =
            mesh_.external_boundary()[plane].periodic_partner();
        if (!partner) {
            throw std::logic_error(
                "HighVoronoi mirror request selects a non-periodic plane.");
        }

        current[static_cast<std::size_t>(plane)] = std::uint8_t{1};
        const bool partner_was_taboo =
            taboo.test(static_cast<std::size_t>(*partner));
        taboo.set(static_cast<std::size_t>(*partner));
        enumerate_shift_combinations(
            reference,
            candidates,
            position + 1,
            current,
            taboo,
            output);
        if (!partner_was_taboo) {
            taboo.reset(static_cast<std::size_t>(*partner));
        }
        current[static_cast<std::size_t>(plane)] = std::uint8_t{0};
    }

    [[nodiscard]] static bool planned_contains(
        const std::vector<PlannedReference>& planned,
        Index reference,
        const ShiftMask& shift) {
        return std::any_of(
            planned.begin(),
            planned.end(),
            [&](const PlannedReference& item) {
                return item.reference_internal == reference &&
                       item.shift == shift;
            });
    }

    [[nodiscard]] std::vector<Index> append_planned_references(
        const std::vector<PlannedReference>& planned,
        Report& report) {
        std::vector<NodePoint> points;
        points.reserve(planned.size());
        for (const auto& item : planned) {
            points.push_back(item.point);
        }

        const std::vector<Index> moved_planes =
            mesh_.expand_internal_boundary_to_include(
                points,
                settings_.boundary_tolerance);
        report.moved_internal_planes += moved_planes.size();

        // Periodic internal-boundary vertices are never persisted by the High
        // compute database, so moving those planes requires no stale-record pass.
        std::vector<Index> appended;
        appended.reserve(planned.size());
        for (const auto& item : planned) {
            if (mesh_.find_reference_copy(
                    item.reference_internal,
                    item.shift)) {
                continue;
            }
            appended.push_back(mesh_.append_reference_node_internal(
                item.reference_internal,
                item.shift));
            ++report.new_reference_nodes;
        }
        return appended;
    }

    [[nodiscard]] NodePoint make_node_point() const {
        if constexpr (HighMesh::DimensionAtCompileTime == Dynamic) {
            return NodePoint(static_cast<Eigen::Index>(mesh_.dimension()));
        } else {
            return NodePoint{};
        }
    }

    HighMesh& mesh_;
    SearchKeyword search_keyword_;
    RayParameters ray_parameters_;
    MeshThreading mesh_threading_;
    CastThreading cast_threading_;
    QueueParameters queue_parameters_;
    EdgeParameters edge_parameters_;
    Settings settings_;
    std::vector<typename HighMesh::Sigma> periodic_repair_seed_signatures_;
};

} // namespace highvoronoi


