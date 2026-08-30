#pragma once

/**
 * @file mesh_cell_validation.hpp
 * @brief Geometric consistency and completeness checks restricted to cell ranges.
 *
 * The consistency check directly inspects only the requested public cells.
 *
 * The completeness check is deliberately different: EdgeIterator has global
 * ownership rules and does not enumerate every incident edge from every cell.
 * Therefore all active cells are traversed exactly as in verify_mesh_complete().
 * Endpoint occurrences are first deduplicated per persistent vertex address and
 * complete supporting edge. Only afterwards are edges filtered to those touching
 * at least one cell in [first_cell, cell_end).
 *
 * This is the correct restriction for views such as VisibleFirstMesh: invisible
 * periodic reference cells may be incomplete, while every geometric edge
 * incident to a visible cell must still have exactly two occurrences.
 */

#include <highvoronoi/geometry/mesh_validation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace highvoronoi {

/** Check geometric consistency only for public cells [first_cell, cell_end). */
template <class Mesh>
[[nodiscard]] MeshVerificationReport<Mesh> verify_mesh_cells(
    Mesh& mesh,
    typename Mesh::Index first_cell,
    typename Mesh::Index cell_end,
    typename Mesh::NodeScalar maximum_variance =
        typename Mesh::NodeScalar{1e-20},
    bool print_errors = false,
    std::ostream& output = std::cerr) {

    using Index = typename Mesh::Index;
    using Error = MeshVerificationError<Mesh>;

    if (first_cell > cell_end || cell_end > mesh.size()) {
        throw std::out_of_range(
            "verify_mesh_cells requires 0 <= first_cell <= cell_end <= mesh.size().");
    }

    MeshVerificationReport<Mesh> report;
    detail::MeshValidationContext<Mesh> context(mesh);

    for (Index cell = first_cell; cell < cell_end; ++cell) {
        context.activate_cell(cell);

        for (const auto& vertex : mesh.vertices(cell)) {
            ++report.checked_vertex_occurrences;
            context.load_position(vertex.position);

            const auto variance = context.variance(vertex.sigma);
            if (!std::isfinite(static_cast<long double>(variance)) ||
                variance > maximum_variance) {
                Error error{
                    MeshVerificationErrorKind::Variance,
                    cell,
                    vertex.address,
                    vertex.sigma,
                    vertex.position,
                    variance,
                    std::nullopt};
                if (print_errors) {
                    detail::print_verification_error(
                        output, error, maximum_variance);
                }
                report.errors.push_back(std::move(error));
            }

            const auto nearest = context.nearest_neighbor();
            if (!nearest || !detail::sigma_contains(vertex.sigma, *nearest)) {
                Error error{
                    MeshVerificationErrorKind::NearestNeighbor,
                    cell,
                    vertex.address,
                    vertex.sigma,
                    vertex.position,
                    variance,
                    nearest};
                if (print_errors) {
                    detail::print_verification_error(
                        output, error, maximum_variance);
                }
                report.errors.push_back(std::move(error));
            }
        }
    }

    return report;
}

/**
 * Check consistency of [first_cell, cell_end) and topological closure of every
 * geometric Voronoi edge incident to at least one cell in that range.
 *
 * Edge discovery must run over the complete mesh because EdgeIterator assigns
 * ownership of an edge to selected generator/cell perspectives. Restricting the
 * traversal itself to the requested cells would miss valid endpoint occurrences.
 */
template <class Mesh>
[[nodiscard]] MeshCompletenessReport<Mesh> verify_mesh_complete_cells(
    Mesh& mesh,
    typename Mesh::Index first_cell,
    typename Mesh::Index cell_end,
    typename Mesh::NodeScalar maximum_variance =
        typename Mesh::NodeScalar{1e-20},
    bool print_errors = false,
    std::ostream& output = std::cerr,
    std::size_t edge_hash_capacity = 256) {

    using Index = typename Mesh::Index;
    using Address = typename Mesh::Address;
    using Sigma = typename Mesh::Sigma;
    using ExtendedNodes = std::remove_reference_t<
        decltype(std::declval<detail::MeshValidationContext<Mesh>&>()
                     .extended_nodes())>;
    using Iterator = EdgeIterator<ExtendedNodes, detail::EmptyLock>;
    using GlobalEdgeHash = detail::EdgeHashTable<detail::EmptyLock>;

    if (first_cell > cell_end || cell_end > mesh.size()) {
        throw std::out_of_range(
            "verify_mesh_complete_cells requires 0 <= first_cell <= cell_end <= mesh.size().");
    }

    MeshCompletenessReport<Mesh> report;
    report.consistency = verify_mesh_cells(
        mesh,
        first_cell,
        cell_end,
        maximum_variance,
        print_errors,
        output);

    detail::MeshValidationContext<Mesh> context(mesh);
    Iterator iterator(context.extended_nodes());

    // Same endpoint discovery semantics as verify_mesh_complete():
    // one physical endpoint occurrence per (persistent vertex address,
    // complete supporting edge), independent of which cell/minimal edge exposed it.
    std::unordered_map<Address, std::vector<Sigma>> edges_by_vertex;

    for (Index cell = Index{0}; cell < mesh.size(); ++cell) {
        context.activate_cell(cell);

        for (const auto& vertex : mesh.vertices(cell)) {
            iterator.reset(
                vertex.sigma,
                vertex.position,
                cell,
                typename Iterator::OnQueueEdges{});

            auto& full_edges = edges_by_vertex[vertex.address];

            while (const auto edge = iterator.next()) {
                Sigma full_edge(
                    edge->full_indices().begin(),
                    edge->full_indices().end());
                std::sort(full_edge.begin(), full_edge.end());

                if (std::find(
                        full_edges.begin(),
                        full_edges.end(),
                        full_edge) != full_edges.end()) {
                    ++report.duplicate_local_edge_representations;
                    continue;
                }

                full_edges.push_back(std::move(full_edge));
            }
        }
    }

    const auto touches_requested_cell =
        [first_cell, cell_end](const Sigma& full_edge) {
            return std::any_of(
                full_edge.begin(),
                full_edge.end(),
                [first_cell, cell_end](Index generator) {
                    return generator >= first_cell && generator < cell_end;
                });
        };

    GlobalEdgeHash selected_edges(edge_hash_capacity);

    for (const auto& [address, full_edges] : edges_by_vertex) {
        (void)address;
        for (const Sigma& full_edge : full_edges) {
            if (!touches_requested_cell(full_edge)) {
                continue;
            }

            // mode=true counts endpoint occurrences, after explicit
            // (address, full_edge) deduplication above.
            (void)selected_edges.pushedge(
                full_edge,
                std::int64_t{0},
                true);
            ++report.unique_finite_edge_endpoints;
        }
    }

    for (const auto& infinite_edge : mesh.infinite_edges()) {
        if (!touches_requested_cell(infinite_edge.sigma)) {
            continue;
        }

        (void)selected_edges.pushedge(
            infinite_edge.sigma,
            GlobalEdgeHash::infinite_cell,
            true);
        ++report.infinite_edges;
    }

    report.all_edges_have_two_occurrences =
        selected_edges.all_edges_complete();

    return report;
}

} // namespace highvoronoi
